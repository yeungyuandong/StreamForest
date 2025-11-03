import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int, 
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).
    """
    protected = 0

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    assert r > 0, r

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src) # , reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


class MemoryManager(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, st_memory_windows=[1, 18], st_memory_tokens=[729, 128], event_split_window=8,
            long_memory_tokens_per_frame=64, long_memory_tokens_quota=5120, sim_weight_g=0.4, time_weight_a=0.2, merge_weight_b=0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        
        self.st_memory_windows = st_memory_windows
        self.st_memory_tokens = st_memory_tokens
        self.event_split_window = event_split_window
        self.long_memory_tokens_per_frame = long_memory_tokens_per_frame
        self.long_memory_tokens_quota = long_memory_tokens_quota
        self.long_memory_current_tokens = 0
        self.long_memory_last_time_pos = -1
        
        self.sim_weight_g = sim_weight_g
        self.time_weight_a = time_weight_a
        self.merge_weight_b = merge_weight_b

        # memory
        self.memory_now = []
        self.memory_short = []
        self.memory_long = []
        
        # sequence id tracking for kv cache reuse
        self.memory_now_seq_ids = []  # 每个元素对应memory_now中每个frame的token的seq_ids
        self.memory_short_seq_ids = []  # 每个元素对应memory_short中每个frame的token的seq_ids
        self.memory_long_seq_ids = []  # 每个元素对应memory_long中每个clip的token的seq_ids
        self.last_output_length = 0  # 上次get_memory_tokens输出的序列长度
        
        # long memory buffer
        self.time_buffer = []
        self.mergecnt_buffer = []
        self.similarity_buffer = []
        
        
        # short memory buffer
        self.last_frame_flat = None
        self.frame_sim_buffer = []
        
    def merge_tokens(self, x, target_num_token):
        r"""
        x = torch.randn(10, 2560, c)
        x = merge_tokens(x, r_merge_list=[1280])
        """
        size = None
        b, p, c = x.shape
        tmp_p = p
        r_merge_list = []
        
        if tmp_p == target_num_token:  #not compress
            return x
        
        assert tmp_p > target_num_token, f"{tmp_p} should greater than {target_num_token}"
        while tmp_p != target_num_token:
            if tmp_p - target_num_token <= (tmp_p // 2):
                r_merge_list.append(tmp_p - target_num_token)
                break
            else:
                r_merge_list.append(tmp_p // 2)
                tmp_p = tmp_p - (tmp_p // 2)
                
        
        head = self.num_attention_heads

        dim = c // head
        for r in r_merge_list:
            metric = x.reshape(b, p, head, dim).mean(2) # [b, p, c//head]
            merge, _ = bipartite_soft_matching(
                metric, 
                r
            )
            x, size = merge_wavg(merge, x, size)
            _, p, _ = x.shape
        # x = x.reshape(-1, c)  # 300, 1024
        return x
    
    def merge_tokens_with_seq_ids(self, x, seq_ids, target_num_token):
        r"""
        合并tokens同时追踪序列ID
        x: [b, p, c] tensor
        seq_ids: [b, p] tensor of sequence ids
        target_num_token: int
        返回: (merged_x, merged_seq_ids)
        """
        size = None
        b, p, c = x.shape
        tmp_p = p
        r_merge_list = []
        
        if tmp_p == target_num_token:  # not compress
            return x, seq_ids
        
        assert tmp_p > target_num_token, f"{tmp_p} should greater than {target_num_token}"
        while tmp_p != target_num_token:
            if tmp_p - target_num_token <= (tmp_p // 2):
                r_merge_list.append(tmp_p - target_num_token)
                break
            else:
                r_merge_list.append(tmp_p // 2)
                tmp_p = tmp_p - (tmp_p // 2)
        
        head = self.num_attention_heads
        dim = c // head
        
        for r in r_merge_list:
            metric = x.reshape(b, p, head, dim).mean(2)  # [b, p, c//head]
            merge, _ = bipartite_soft_matching(metric, r)
            x, size = merge_wavg(merge, x, size)
            
            # 保守策略：只要发生了合并，就将所有seq_id置0
            # 因为合并后的tokens是加权平均，内容已经改变
            _, p_new, _ = x.shape
            seq_ids = torch.zeros((b, p_new), dtype=torch.long, device=x.device)
            
            p = p_new  # 更新p为新的token数量
        
        return x, seq_ids

    def calculate_similarity_between_clips(self, clip1, clip2):
        new_clip = torch.cat([clip1, clip2], dim=0).reshape(-1, clip1.shape[-1])  # [T, C]
        total_tokens = new_clip.shape[0]
        target_tokens = math.ceil((clip1.shape[0] + clip2.shape[0]) / 2) * self.long_memory_tokens_per_frame

        p, c = new_clip.shape
        head = self.num_attention_heads
        dim = c // head
        r = total_tokens - target_tokens  
        assert r <= total_tokens // 2, "The merged tokens must not exceed half! "
        assert r > 0, "Token merging count r must be > 0! "
        
        with torch.no_grad():
            new_clip = new_clip.reshape(p, head, dim).mean(1)
            
            x = new_clip / new_clip.norm(dim=-1, keepdim=True)  
            a, b = x[::2], x[1::2]  
            scores = a @ b.transpose(-1, -2)  
            
            max_scores, _ = scores.max(dim=-1)  
            top_r_scores = max_scores.topk(r, largest=True).values  

        return top_r_scores.mean().item() 

    def _update_short_memory(self):
        overflow = len(self.memory_now) - self.st_memory_windows[0]
        if overflow > 0:
            old_now_batch = self.memory_now[:overflow]
            old_now_seq_ids_batch = self.memory_now_seq_ids[:overflow]
            
            self.memory_now = self.memory_now[overflow:]
            self.memory_now_seq_ids = self.memory_now_seq_ids[overflow:]
            
            old_now_batch = torch.cat(old_now_batch, dim=0)  # [B, p, c]
            old_now_seq_ids_batch = torch.cat(old_now_seq_ids_batch, dim=0)  # [B, p]
            
            short_tokens_batch, short_seq_ids_batch = self.merge_tokens_with_seq_ids(
                old_now_batch, old_now_seq_ids_batch, target_num_token=self.st_memory_tokens[1]
            )  # [B, p', c], [B, p']
            
            self.memory_short.extend(t.unsqueeze(0) for t in short_tokens_batch.unbind(0))
            self.memory_short_seq_ids.extend(s.unsqueeze(0) for s in short_seq_ids_batch.unbind(0))
        return


        
    def _update_event_split(self):
        while len(self.memory_short) >= self.st_memory_windows[1] + self.event_split_window:
            # print("<<<Mark>>> self.memory_short: ", len(self.memory_short), "self.frame_sim_buffer: ", len(self.frame_sim_buffer))
            
            window_sim = torch.stack(self.frame_sim_buffer[:self.event_split_window])
            # print("<<<Mark>>> window_sim len:", len(window_sim), "window_sim:", window_sim)
            
            min_sim_idx = torch.argmin(window_sim)
            # print("<<<Mark>>> min_sim_idx:", min_sim_idx)
            
            split_frame_idx = min_sim_idx + 1
            # print("<<<Mark>>> split_frame_idx:", split_frame_idx)
            
            old_short = torch.cat(self.memory_short[:split_frame_idx], dim=0)
            old_short_seq_ids = torch.cat(self.memory_short_seq_ids[:split_frame_idx], dim=0)
            # print("<<<Mark>>> old_short:", old_short.shape)
            
            self.memory_short = self.memory_short[split_frame_idx:]     #删除短记忆队列中被弹出的帧
            self.memory_short_seq_ids = self.memory_short_seq_ids[split_frame_idx:]
            self.frame_sim_buffer = self.frame_sim_buffer[split_frame_idx:]
            
            # print("<<<Mark>>> self.memory_short: ", len(self.memory_short), "self.frame_sim_buffer: ", len(self.frame_sim_buffer))
            
            event_merged, event_seq_ids = self.merge_tokens_with_seq_ids(
                old_short.reshape(1, -1, self.hidden_size), 
                old_short_seq_ids.reshape(1, -1),
                self.long_memory_tokens_per_frame * old_short.shape[0]
            )
            event_merged = event_merged.reshape(old_short.shape[0], -1, self.hidden_size)
            event_seq_ids = event_seq_ids.reshape(old_short.shape[0], -1)
            # print("<<<Mark>>> event_merged ", event_merged.shape)
            
            # 初始化长记忆
            self.memory_long.append(event_merged)
            self.memory_long_seq_ids.append(event_seq_ids)
            self.long_memory_current_tokens += event_merged.shape[0]*event_merged.shape[1]
            self.time_buffer.append((self.long_memory_last_time_pos*2 + split_frame_idx + 1)/2)
            self.long_memory_last_time_pos+=split_frame_idx
            self.mergecnt_buffer.append(1)
            if len(self.memory_long)>1:
                self.similarity_buffer.append(self.calculate_similarity_between_clips(self.memory_long[-2], self.memory_long[-1]))
        return

    def _update_long_memory(self):
        while self.long_memory_current_tokens > self.long_memory_tokens_quota and len(self.memory_long) > 1:
            input_device = self.memory_long[0].device
            num_events = len(self.memory_long)

            # 1. calculate overall pan
            sim_scores = torch.tensor(self.similarity_buffer, device=input_device)  # [num_events-1]
            time_diffs = torch.tensor([(self.time_buffer[i+1] + self.time_buffer[i])/2 for i in range(num_events-1)], device = input_device)
            merge_cnts = torch.tensor([(self.mergecnt_buffer[i] + self.mergecnt_buffer[i+1])/2 for i in range(num_events-1)], device = input_device)

            sim_penalties = 1 - sim_scores
            time_penalties = time_diffs / (self.long_memory_last_time_pos + len(self.memory_short) + len(self.memory_now))
            merge_penalties = merge_cnts / merge_cnts.max() + 1e-6

            total_penalties = self.sim_weight_g * sim_penalties    +    self.time_weight_a * time_penalties    +    self.merge_weight_b * merge_penalties

            merge_idx = torch.argmin(total_penalties).item()
            # print("<_update_long_memory> merge_idx: ", merge_idx)
            # 2. conduct merge
            clip1, clip2 = self.memory_long[merge_idx], self.memory_long[merge_idx+1]
            clip1_seq_ids, clip2_seq_ids = self.memory_long_seq_ids[merge_idx], self.memory_long_seq_ids[merge_idx+1]
            # print("<_update_long_memory> clip1: ", clip1.shape, "clip2", clip2.shape)
            
            merged_tokens = torch.cat([clip1, clip2], dim=0).reshape(1, -1, self.hidden_size)  # [1, T, C]
            merged_seq_ids = torch.cat([clip1_seq_ids, clip2_seq_ids], dim=0).reshape(1, -1)  # [1, T*tokens_per_frame]
            # print("<_update_long_memory> merged_tokens before: ", merged_tokens.shape)
            
            merged_tokens, merged_seq_ids = self.merge_tokens_with_seq_ids(
                merged_tokens, 
                merged_seq_ids,
                self.long_memory_tokens_per_frame * math.ceil((clip1.shape[0] + clip2.shape[0])/2)
            )
            merged_tokens = merged_tokens.reshape(-1, self.long_memory_tokens_per_frame, self.hidden_size)  # [T', tokens_per_frame, C]
            merged_seq_ids = merged_seq_ids.reshape(-1, self.long_memory_tokens_per_frame)  # [T', tokens_per_frame]
            # print("<_update_long_memory> merged_tokens after: ", merged_tokens.shape)
            
            
            self.memory_long[merge_idx] = merged_tokens
            self.memory_long_seq_ids[merge_idx] = merged_seq_ids
            del self.memory_long[merge_idx+1]
            del self.memory_long_seq_ids[merge_idx+1]
            
            self.long_memory_current_tokens -= (clip1.shape[0] + clip2.shape[0] - merged_tokens.shape[0]) * self.long_memory_tokens_per_frame

            # 3. update time_buffer
            self.time_buffer[merge_idx] = (self.time_buffer[merge_idx] * clip1.shape[0] + self.time_buffer[merge_idx+1] * clip2.shape[0]) / (clip1.shape[0] + clip2.shape[0])
            del self.time_buffer[merge_idx+1]

            # 4. update mergecnt_buffer
            self.mergecnt_buffer[merge_idx] += self.mergecnt_buffer[merge_idx+1]
            del self.mergecnt_buffer[merge_idx+1]

            # 5. 局部update similarity_buffer
            del self.similarity_buffer[merge_idx]
            # update left pair (merge_idx-1, merge_idx)
            if merge_idx - 1 >= 0:
                self.similarity_buffer[merge_idx-1] = self.calculate_similarity_between_clips(self.memory_long[merge_idx-1], self.memory_long[merge_idx])
            # update right pair (merge_idx, merge_idx+1)
            if merge_idx < len(self.memory_long) - 1:
                self.similarity_buffer[merge_idx] = self.calculate_similarity_between_clips(self.memory_long[merge_idx], self.memory_long[merge_idx+1])

            # print("<<<_update_long_memory>>> self.similarity_buffer: ", self.similarity_buffer)
            # print("<<<_update_long_memory>>> self.mergecnt_buffer: ", self.mergecnt_buffer)
            # print("<<<_update_long_memory>>> self.time_buffer: ", self.time_buffer)
        return


    def update(self, new_frame: torch.Tensor):
        """
        new_frame: [H*W, C]
        frame_idx: 当前帧编号 (int)
        return: [1, N, C] memory feature sequence
        """
        new_frame_flat = new_frame.reshape(-1)  # [T, H*W*C]
        if self.last_frame_flat is not None:
            self.frame_sim_buffer.append(F.cosine_similarity(self.last_frame_flat, new_frame_flat, dim=0))
        self.last_frame_flat = new_frame_flat
        
        new_frame_tokens = self.merge_tokens(new_frame.unsqueeze(0), self.st_memory_tokens[0])  # [1, N0, C]
        # 新加入的帧，seq_id全部设为0 (表示需要重新计算)
        new_frame_seq_ids = torch.zeros((1, new_frame_tokens.shape[1]), dtype=torch.long, device=new_frame.device)
        
        self.memory_now.append(new_frame_tokens)
        self.memory_now_seq_ids.append(new_frame_seq_ids)


    def get_memory_tokens(self):
        self._update_short_memory()
        self._update_event_split()
        self._update_long_memory()
        
        x_all = []
        seq_ids_all = []
        
        if len(self.memory_long) > 0:
            long_tokens = torch.cat(self.memory_long, dim=0).reshape(1, -1, self.hidden_size)
            long_seq_ids = torch.cat(self.memory_long_seq_ids, dim=0).reshape(1, -1)
            x_all.append(long_tokens)
            seq_ids_all.append(long_seq_ids)
            
        if len(self.memory_short) > 0:
            short_tokens = torch.cat(self.memory_short, dim=0).reshape(1, -1, self.hidden_size)
            short_seq_ids = torch.cat(self.memory_short_seq_ids, dim=0).reshape(1, -1)
            x_all.append(short_tokens)
            seq_ids_all.append(short_seq_ids)
            
        if len(self.memory_now) > 0:
            now_tokens = torch.cat(self.memory_now, dim=0).reshape(1, -1, self.hidden_size)
            now_seq_ids = torch.cat(self.memory_now_seq_ids, dim=0).reshape(1, -1)
            x_all.append(now_tokens)
            seq_ids_all.append(now_seq_ids)
        
        all_tokens = torch.cat(x_all, dim=1)  # [1, N_total, C]
        all_seq_ids_old = torch.cat(seq_ids_all, dim=1)  # [1, N_total] - 这是从上次输出继承来的seq_ids
        
        # 为本次输出分配新的seq_ids（供下次使用）
        # seq_id表示在"上次输出"中的位置，所以本次输出后要更新为"本次位置"
        all_seq_ids_new = torch.arange(1, all_tokens.shape[1] + 1, dtype=torch.long, device=all_tokens.device).unsqueeze(0)
        
        # 更新last_output_length
        self.last_output_length = all_tokens.shape[1]
        
        # 将新的seq_ids写回各个memory（用于下次判断是否可复用）
        offset = 0
        if len(self.memory_long) > 0:
            long_len = sum([m.shape[0] * m.shape[1] for m in self.memory_long])
            updated_long_seq_ids = all_seq_ids_new[:, offset:offset+long_len]
            offset_within_long = 0
            for i in range(len(self.memory_long)):
                clip_len = self.memory_long[i].shape[0] * self.memory_long[i].shape[1]
                self.memory_long_seq_ids[i] = updated_long_seq_ids[:, offset_within_long:offset_within_long+clip_len].reshape(
                    self.memory_long[i].shape[0], self.memory_long[i].shape[1]
                )
                offset_within_long += clip_len
            offset += long_len
            
        if len(self.memory_short) > 0:
            short_len = sum([m.shape[1] for m in self.memory_short])
            updated_short_seq_ids = all_seq_ids_new[:, offset:offset+short_len]
            offset_within_short = 0
            for i in range(len(self.memory_short)):
                frame_len = self.memory_short[i].shape[1]
                self.memory_short_seq_ids[i] = updated_short_seq_ids[:, offset_within_short:offset_within_short+frame_len]
                offset_within_short += frame_len
            offset += short_len
            
        if len(self.memory_now) > 0:
            now_len = sum([m.shape[1] for m in self.memory_now])
            updated_now_seq_ids = all_seq_ids_new[:, offset:offset+now_len]
            offset_within_now = 0
            for i in range(len(self.memory_now)):
                frame_len = self.memory_now[i].shape[1]
                self.memory_now_seq_ids[i] = updated_now_seq_ids[:, offset_within_now:offset_within_now+frame_len]
                offset_within_now += frame_len
        
        # 返回tokens和旧的seq_ids（旧的seq_ids用于判断哪些可以复用）
        return all_tokens, all_seq_ids_old  # [1, N_total, C], [1, N_total]