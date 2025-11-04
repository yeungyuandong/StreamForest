import torch
import torch.nn as nn
from typing import Optional, Tuple

# 用于存储预计算的 KV cache
_PRECOMPUTED_KV_CACHE = None
_IMAGE_TOKEN_COUNT = 0  # 存储image tokens的数量


class Qwen2AttentionWithPrecomputedKV(nn.Module):
    """
    支持预计算KV cache的Qwen2Attention包装器
    在prefill阶段，如果有预计算的image tokens KV cache，直接复用
    
    工作原理：
    1. 检测到预计算KV cache时，将hidden_states分割为image和text部分
    2. 跳过image部分，只对text部分调用原始attention（传入预计算的image KV）
    3. 这样避免了重复计算image tokens的KV
    """
    def __init__(self, original_attention, layer_idx):
        super().__init__()
        self.original_attention = original_attention
        self.layer_idx = layer_idx
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        global _PRECOMPUTED_KV_CACHE, _IMAGE_TOKEN_COUNT
        
        # 只在prefill阶段且有预计算KV cache时特殊处理
        # past_key_value为None表示是prefill阶段的第一次调用
        if (_PRECOMPUTED_KV_CACHE is not None and 
            past_key_value is None and 
            _IMAGE_TOKEN_COUNT > 0 and 
            self.layer_idx < len(_PRECOMPUTED_KV_CACHE)):
            
            precomputed_key, precomputed_value = _PRECOMPUTED_KV_CACHE[self.layer_idx]
            precomputed_seq_len = precomputed_key.shape[2]  # 预计算的序列长度（image tokens数量）
            seq_len = hidden_states.shape[1]  # 当前的序列长度（image + text tokens）
            
            # 检查是否是prefill阶段（seq_len >= precomputed_seq_len）
            if seq_len >= precomputed_seq_len:
                # 分离出text tokens部分（跳过前面的image tokens）
                text_hidden_states = hidden_states[:, precomputed_seq_len:, :]
                
                # 调整position_ids（如果存在）
                text_position_ids = position_ids
                if position_ids is not None and position_ids.shape[1] > precomputed_seq_len:
                    text_position_ids = position_ids[:, precomputed_seq_len:]
                
                # 使用预计算的KV作为past_key_value，只计算text tokens的KV
                # 原始attention会自动将past_key_value和新计算的KV拼接
                return self.original_attention(
                    hidden_states=text_hidden_states,
                    attention_mask=attention_mask,  # attention_mask保持不变，因为它涵盖整个序列
                    position_ids=text_position_ids,
                    past_key_value=(precomputed_key, precomputed_value),
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )
        
        # 正常情况：调用原始的attention forward
        return self.original_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )


def set_precomputed_kv_cache(kv_cache, image_token_count):
    """设置预计算的KV cache"""
    global _PRECOMPUTED_KV_CACHE, _IMAGE_TOKEN_COUNT
    _PRECOMPUTED_KV_CACHE = kv_cache
    _IMAGE_TOKEN_COUNT = image_token_count


def clear_precomputed_kv_cache():
    """清理预计算的KV cache"""
    global _PRECOMPUTED_KV_CACHE, _IMAGE_TOKEN_COUNT
    _PRECOMPUTED_KV_CACHE = None
    _IMAGE_TOKEN_COUNT = 0
