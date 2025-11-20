#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


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
        import time
        
        # Batch推理
        vision_model = self.get_model().get_vision_tower()
        all_features = []
        concat_images = images[0]
        batch_size=128
        for i in range(0, concat_images.shape[0], batch_size):
            batch = concat_images[i:i + batch_size]
            batch_features = vision_model(batch)
            all_features.append(batch_features)
        image_features = torch.cat(all_features, dim=0)
        
        # image_features=self.get_model().get_vision_tower()(images[0].unsqueeze(1))
        
        memory_manager = MemoryManager(self.get_model().get_vision_tower().config.hidden_size, self.get_model().get_vision_tower().config.num_attention_heads,st_memory_windows=[1, 6], st_memory_tokens=[64, 16], event_split_window=4, long_memory_tokens_per_frame=16, long_memory_tokens_quota=800)
        
        for frame_idx in range(image_features.shape[0]):
            # print("input_new_frame:", image_features[frame_idx].shape)
            memory_manager.update(image_features[frame_idx])
            image_features_now = memory_manager.get_memory_tokens()
            image_features_now = self.get_model().mm_projector.mlp(image_features_now)
            (inputs_now, position_ids_now, attention_mask_now, _, inputs_embeds_now, _) = self.prepare_inputs_labels_for_LLM(inputs, position_ids, attention_mask, None, None, images, [image_features_now], modalities, image_sizes=image_sizes)

            # print("inputs_embeds to llm: ", inputs_embeds_now.shape)
            
            start_time = time.time()
            
            result = super().generate(position_ids=position_ids_now, attention_mask=attention_mask_now, inputs_embeds=inputs_embeds_now, **kwargs)
            
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            print(f"\n <<< LLM inference time: {elapsed_time:.3f} seconds >>> ")
            
            print("result at frame", frame_idx,":", result)
        
        return result
    
    @torch.no_grad()
    def generate_online_new(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        优化版本的generate_online，使用KV cache复用来加速跨轮推理
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        from llava.model.multimodal_projector.memory_manager import MemoryManager
        import time
        
        # Batch推理vision encoder
        vision_model = self.get_model().get_vision_tower()
        all_features = []
        concat_images = images[0]
        batch_size = 128
        for i in range(0, concat_images.shape[0], batch_size):
            batch = concat_images[i:i + batch_size]
            batch_features = vision_model(batch)
            all_features.append(batch_features)
        image_features = torch.cat(all_features, dim=0)
        
        memory_manager = MemoryManager(
            self.get_model().get_vision_tower().config.hidden_size, 
            self.get_model().get_vision_tower().config.num_attention_heads,
            st_memory_windows=[1, 6], 
            st_memory_tokens=[64, 16], 
            event_split_window=4, 
            long_memory_tokens_per_frame=16, 
            long_memory_tokens_quota=800
        )
        
        # 用于跨轮KV cache复用的变量
        prev_inputs_embeds = None
        cached_kv = None
        cached_past_length = 0
        
        # 统计信息
        total_tokens_processed = 0
        total_tokens_saved = 0
        
        for frame_idx in range(image_features.shape[0]):
            memory_manager.update(image_features[frame_idx])
            image_features_now = memory_manager.get_memory_tokens()
            image_features_now = self.get_model().mm_projector.mlp(image_features_now)
            (inputs_now, position_ids_now, attention_mask_now, _, inputs_embeds_now, _) = \
                self.prepare_inputs_labels_for_LLM(inputs, position_ids, attention_mask, None, None, 
                                                    images, [image_features_now], modalities, image_sizes=image_sizes)
            
            # 找到第一个改变的token位置（连续匹配）
            reuse_length = 0
            if prev_inputs_embeds is not None and cached_kv is not None:
                min_len = min(prev_inputs_embeds.shape[1], inputs_embeds_now.shape[1])
                # 逐个比较token embedding，找到第一个不同的位置
                for i in range(min_len):
                    if torch.allclose(prev_inputs_embeds[:, i, :], inputs_embeds_now[:, i, :], atol=1e-6):
                        reuse_length = i + 1
                    else:
                        break
            
            # 如果有可复用的部分，则截取KV cache
            if reuse_length > 0:
                print(f"\n[Frame {frame_idx}] KV cache复用: {reuse_length}/{inputs_embeds_now.shape[1]} tokens (节省 {reuse_length/inputs_embeds_now.shape[1]*100:.1f}%)")
                
                # 截取past_key_values到reuse_length
                # 处理transformers的Cache对象或tuple list
                past_key_values_reused = []
                try:
                    # 尝试作为DynamicCache处理
                    if hasattr(cached_kv, 'key_cache'):
                        for layer_idx in range(len(cached_kv.key_cache)):
                            key_cached = cached_kv.key_cache[layer_idx][:, :, :reuse_length, :]
                            value_cached = cached_kv.value_cache[layer_idx][:, :, :reuse_length, :]
                            past_key_values_reused.append((key_cached, value_cached))
                    else:
                        # 作为tuple list处理
                        for layer_kv in cached_kv:
                            # layer_kv是一个tuple: (key, value)
                            # key和value的shape: [batch, num_heads, seq_len, head_dim]
                            key_cached = layer_kv[0][:, :, :reuse_length, :]
                            value_cached = layer_kv[1][:, :, :reuse_length, :]
                            past_key_values_reused.append((key_cached, value_cached))
                except Exception as e:
                    print(f"警告: KV cache截取失败 ({e})，将进行完整推理")
                    reuse_length = 0
                    past_key_values_reused = None
                
                if reuse_length > 0:
                    # 只传递新增的inputs_embeds
                    inputs_embeds_new = inputs_embeds_now[:, reuse_length:, :]
                    
                    # 调整position_ids：从reuse_length开始
                    if position_ids_now is not None:
                        position_ids_new = position_ids_now[:, reuse_length:]
                    else:
                        position_ids_new = torch.arange(
                            reuse_length, 
                            reuse_length + inputs_embeds_new.shape[1], 
                            dtype=torch.long, 
                            device=inputs_embeds_new.device
                        ).unsqueeze(0)
                    
                    # 调整attention_mask：保留完整长度
                    if attention_mask_now is not None:
                        attention_mask_new = attention_mask_now
                    else:
                        attention_mask_new = torch.ones(
                            (1, reuse_length + inputs_embeds_new.shape[1]),
                            dtype=torch.long,
                            device=inputs_embeds_new.device
                        )
                else:
                    # 回退到完整推理
                    past_key_values_reused = None
                    inputs_embeds_new = inputs_embeds_now
                    position_ids_new = position_ids_now
                    attention_mask_new = attention_mask_now
            else:
                print(f"\n[Frame {frame_idx}] 无KV cache复用，完整推理 {inputs_embeds_now.shape[1]} tokens")
                past_key_values_reused = None
                inputs_embeds_new = inputs_embeds_now
                position_ids_new = position_ids_now
                attention_mask_new = attention_mask_now
            
            start_time = time.time()
            
            # 使用model的forward来获取KV cache
            # 注意：如果使用了past_key_values，需要确保格式正确（tuple而不是list）
            pkv_to_use = None
            if reuse_length > 0 and past_key_values_reused is not None:
                pkv_to_use = tuple(past_key_values_reused)
            
            outputs = self.model(
                inputs_embeds=inputs_embeds_new,
                attention_mask=attention_mask_new,
                position_ids=position_ids_new,
                past_key_values=pkv_to_use,
                use_cache=True,
                return_dict=True
            )
            
            # 获取logits
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)
            
            # 取最后一个token的logits并生成（这里简化为greedy decoding生成1个token）
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 构造result - 只返回生成的token（与原generate_online格式保持一致）
            result = next_token
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 统计
            total_tokens_processed += inputs_embeds_new.shape[1]
            if reuse_length > 0:
                total_tokens_saved += reuse_length
            
            # 保存当前的inputs_embeds和KV cache用于下一轮
            prev_inputs_embeds = inputs_embeds_now
            cached_kv = outputs.past_key_values
            cached_past_length = inputs_embeds_now.shape[1]
            
            print(f"<<< LLM inference time: {elapsed_time:.3f} seconds >>>")
            print(f"Result at frame {frame_idx}: {result}")
        
        # 打印总体统计信息
        print(f"\n{'='*50}")
        print(f"KV Cache复用统计:")
        print(f"  总帧数: {image_features.shape[0]}")
        print(f"  实际处理tokens: {total_tokens_processed}")
        print(f"  复用节省tokens: {total_tokens_saved}")
        if total_tokens_saved + total_tokens_processed > 0:
            print(f"  复用率: {total_tokens_saved/(total_tokens_saved + total_tokens_processed)*100:.2f}%")
        print(f"{'='*50}\n")
        
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