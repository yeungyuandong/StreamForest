from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # print("images[0].shape:", images[0].shape)
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        # print("inputs_embeds.shape:", inputs_embeds.shape)
        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
    
    @torch.no_grad()
    def generate_online(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        from llava.model.multimodal_projector.memory_manager import MemoryManager
        from llava.model.qwen_attention import (
            Qwen2AttentionWithPrecomputedKV, 
            set_precomputed_kv_cache, 
            clear_precomputed_kv_cache
        )
        import time
        
        # 性能监控
        perf_stats = {
            'vision_encoding': 0,
            'memory_update': 0,
            'kv_reuse': 0,
            'kv_new_compute': 0,
            'kv_reassemble': 0,
            'llm_prepare': 0,
            'llm_generate': 0,
            'attention_replace': 0,
        }
        
        # 第一步：替换所有attention层为支持预计算KV的版本
        attn_replace_start = time.time()
        original_attentions = []
        for layer_idx, layer in enumerate(self.model.layers):
            original_attentions.append(layer.self_attn)
            layer.self_attn = Qwen2AttentionWithPrecomputedKV(layer.self_attn, layer_idx)
        perf_stats['attention_replace'] = time.time() - attn_replace_start
        
        # Batch推理vision encoder
        vision_start = time.time()
        vision_model = self.get_model().get_vision_tower()
        all_features = []
        concat_images = images[0]
        batch_size=128
        for i in range(0, concat_images.shape[0], batch_size):
            batch = concat_images[i:i + batch_size]
            batch_features = vision_model(batch)
            all_features.append(batch_features)
        image_features = torch.cat(all_features, dim=0)
        perf_stats['vision_encoding'] = time.time() - vision_start
        
        # image_features=self.get_model().get_vision_tower()(images[0].unsqueeze(1))
        
        memory_manager = MemoryManager(self.get_model().get_vision_tower().config.hidden_size, self.get_model().get_vision_tower().config.num_attention_heads,st_memory_windows=[1, 6], st_memory_tokens=[64, 16], event_split_window=4, long_memory_tokens_per_frame=16, long_memory_tokens_quota=800)
        
        # 初始化KV cache相关变量
        past_image_kv_cache = None  # 存储image tokens的KV cache
        last_seq_ids = None  # 上一轮的seq_ids
        last_image_token_count = 0  # 上一轮image tokens的数量
        
        for frame_idx in range(image_features.shape[0]):
            # print("input_new_frame:", image_features[frame_idx].shape)
            memory_manager.update(image_features[frame_idx])
            image_features_now, seq_ids_now = memory_manager.get_memory_tokens()  # [1, N, C], [1, N]
            
            current_image_token_count = image_features_now.shape[1]
            
            # 如果不是第一帧，尝试复用KV cache
            if past_image_kv_cache is not None and last_seq_ids is not None:
                # 功能2: 创建映射 - 找出可复用的token及其新旧位置
                seq_ids_flat = seq_ids_now.squeeze(0)  # [N]
                reusable_mask = seq_ids_flat != 0  # 标记哪些token可以复用
                new_token_mask = seq_ids_flat == 0  # 标记哪些token需要重新计算
                
                # 构建old_position -> new_position的映射
                # seq_ids_flat中非0值表示该token在上一轮中的位置(1-based)
                old_positions = seq_ids_flat[reusable_mask] - 1  # 转换为0-based索引
                new_positions = torch.where(reusable_mask)[0]  # 可复用token在当前轮的位置
                
                # 功能3: 根据映射复用和计算KV cache
                num_reusable = reusable_mask.sum().item()
                num_new = new_token_mask.sum().item()
                
                print(f"\nFrame {frame_idx}: 可复用tokens: {num_reusable}/{current_image_token_count}, 需计算tokens: {num_new}")
                
                # 3.1 对于新增的tokens，需要手动forward计算KV cache
                new_image_kv_cache = None
                if num_new > 0:
                    new_tokens = image_features_now[:, new_token_mask, :]  # [1, num_new, C]
                    new_tokens_embeds = self.get_model().mm_projector.mlp(new_tokens)
                    
                    # 只对新tokens进行forward，获取它们的KV cache
                    outputs_new = self.model(
                        inputs_embeds=new_tokens_embeds,
                        use_cache=True,
                        return_dict=True,
                    )
                    new_image_kv_cache = outputs_new.past_key_values  # tuple of (key, value) for each layer
                
                # 3.2 初始化当前轮完整的image KV cache
                # KV cache结构: tuple of (key, value) for each layer
                # key/value shape: [batch_size, num_heads, seq_len, head_dim]
                current_image_kv_cache = []
                
                for layer_idx in range(len(past_image_kv_cache)):
                    past_key, past_value = past_image_kv_cache[layer_idx]
                    batch_size, num_heads, _, head_dim = past_key.shape
                    
                    # 优化: 使用index_select一次性提取可复用的tokens，减少内存分配
                    if num_reusable > 0:
                        # 从past KV中选择可复用的tokens
                        reused_key = past_key.index_select(2, old_positions)  # [B, H, num_reusable, D]
                        reused_value = past_value.index_select(2, old_positions)
                    
                    # 根据情况组装新的KV cache
                    if num_new > 0 and new_image_kv_cache is not None:
                        new_token_positions = torch.where(new_token_mask)[0]
                        new_key_computed, new_value_computed = new_image_kv_cache[layer_idx]
                        
                        if num_reusable > 0:
                            # 有复用 + 有新计算：需要按照正确顺序拼接
                            # 创建索引映射来确定拼接顺序
                            new_key = torch.empty(batch_size, num_heads, current_image_token_count, head_dim,
                                                dtype=past_key.dtype, device=past_key.device)
                            new_value = torch.empty(batch_size, num_heads, current_image_token_count, head_dim,
                                                  dtype=past_value.dtype, device=past_value.device)
                            new_key[:, :, new_positions, :] = reused_key
                            new_value[:, :, new_positions, :] = reused_value
                            new_key[:, :, new_token_positions, :] = new_key_computed
                            new_value[:, :, new_token_positions, :] = new_value_computed
                        else:
                            # 只有新计算，直接使用
                            new_key = new_key_computed
                            new_value = new_value_computed
                    else:
                        # 只有复用，直接使用（但需要按新位置排列）
                        if num_reusable > 0:
                            new_key = torch.empty(batch_size, num_heads, current_image_token_count, head_dim,
                                                dtype=past_key.dtype, device=past_key.device)
                            new_value = torch.empty(batch_size, num_heads, current_image_token_count, head_dim,
                                                  dtype=past_value.dtype, device=past_value.device)
                            new_key[:, :, new_positions, :] = reused_key
                            new_value[:, :, new_positions, :] = reused_value
                        else:
                            # 理论上不应该到这里（既没复用也没新计算）
                            new_key = torch.empty(batch_size, num_heads, current_image_token_count, head_dim,
                                                dtype=past_key.dtype, device=past_key.device)
                            new_value = torch.empty(batch_size, num_heads, current_image_token_count, head_dim,
                                                  dtype=past_value.dtype, device=past_value.device)
                    
                    current_image_kv_cache.append((new_key, new_value))
                
                past_image_kv_cache = tuple(current_image_kv_cache)
                
            else:
                # 第一帧：所有tokens都需要计算KV cache
                print(f"\nFrame {frame_idx}: 首帧，计算所有 {current_image_token_count} 个tokens的KV cache")
                image_features_now_embeds = self.get_model().mm_projector.mlp(image_features_now)
                
                # Forward计算image tokens的KV cache
                outputs_first = self.model(
                    inputs_embeds=image_features_now_embeds,
                    use_cache=True,
                    return_dict=True,
                )
                past_image_kv_cache = outputs_first.past_key_values
            
            # 准备生成阶段的输入
            image_features_now_embeds = self.get_model().mm_projector.mlp(image_features_now)
            (inputs_now, position_ids_now, attention_mask_now, _, inputs_embeds_now, _) = self.prepare_inputs_labels_for_LLM(
                inputs, position_ids, attention_mask, None, None, images, 
                [image_features_now_embeds], modalities, image_sizes=image_sizes
            )

            # print("inputs_embeds to llm: ", inputs_embeds_now.shape)
            
            start_time = time.time()
            
            # 第二步：设置全局KV cache，让attention层能够访问
            set_precomputed_kv_cache(past_image_kv_cache, current_image_token_count)
            
            try:
                # 调用generate，此时attention层会使用预计算的KV cache
                kwargs.pop("use_cache", None)
                result = super().generate(
                    position_ids=position_ids_now, 
                    attention_mask=attention_mask_now, 
                    inputs_embeds=inputs_embeds_now,
                    use_cache=True, 
                    **kwargs
                )
            finally:
                # 第三步：清理全局KV cache
                clear_precomputed_kv_cache()
            
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            print(f"\n <<< LLM inference time: {elapsed_time:.3f} seconds >>> ")
            print(f"    已复用 {current_image_token_count} 个image tokens的KV cache")
            
            print("result at frame", frame_idx,":", result)
            
            # 更新状态，为下一轮做准备
            last_seq_ids = seq_ids_now
            last_image_token_count = current_image_token_count
        
        # 第四步：恢复原始的attention层
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = original_attentions[layer_idx]
        
        return result

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)