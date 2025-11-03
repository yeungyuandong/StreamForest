"""
StreamForest 单轮对话测试脚本
用于测试和调试单轮对话功能，详细记录推理耗时
"""
import os
import sys
import argparse
import torch
import json
import time
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import AutoConfig
from llava.video_utils import VIDEO_READER_FUNCS


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="StreamForest单轮对话测试")
    
    # 模型相关参数
    parser.add_argument("--model-path", type=str, 
                        default="./path/StreamForest-Qwen2-7B",
                        help="模型路径")
    parser.add_argument("--model-base", type=str, default=None,
                        help="模型基础路径")
    parser.add_argument("--llm_type", type=str, default="",
                        help="LLM类型")
    
    # 视频相关参数
    parser.add_argument("--video-path", type=str,
                        default="/disk/zdata1/home/liuyunfeng/1026/dataset/VIABench/full_set/blind_videos/3_douyin_2_c/0416.mp4",
                        help="测试视频路径")
    parser.add_argument("--max_num_frames", type=int, default=4096,
                        help="最大帧数")
    parser.add_argument("--question_time", type=int, default=0,
                        help="问题时间点（秒），0表示使用整个视频")
    
    # 推理相关参数
    parser.add_argument("--inference_device", type=str, default="cuda:0",
                        help="推理设备")
    parser.add_argument("--conv-mode", type=str, default="qwen_2",
                        help="对话模式")
    parser.add_argument("--load_8bit", type=lambda x: (str(x).lower() == 'true'), 
                        default=False,
                        help="是否使用8bit加载")
    parser.add_argument("--attn_implementation", type=str, 
                        default="flash_attention_2",
                        help="注意力实现方式")
    
    # Online模式相关参数
    parser.add_argument("--use_online_mode", type=lambda x: (str(x).lower() == 'true'),
                        default=True,
                        help="是否使用Online模式（generate_online）")
    
    # 输出相关参数
    parser.add_argument("--output_dir", type=str,
                        default="./work_dirs/multiround_test",
                        help="输出目录")
    parser.add_argument("--output_name", type=str,
                        default="inference_profile",
                        help="输出文件名（不含扩展名）")
    
    # 时间消息相关参数
    parser.add_argument("--time_msg", type=str, default="",
                        help="时间消息类型")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'),
                        default=False,
                        help="是否强制采样")
    
    args = parser.parse_args(args=[])
    return args


class OnlineChatTester:
    """单轮对话测试类，专注于generate_online推理"""
    
    def __init__(self, args):
        """
        初始化测试器
        
        Args:
            args: 命令行参数对象
        """
        self.args = args
        self.device = args.inference_device
        self.max_num_frames = args.max_num_frames
        self.model_path = args.model_path
        self.use_online_mode = args.use_online_mode
        
        # 推理统计信息
        self.inference_stats = {
            "video_info": {},
            "model_info": {},
            "inference_result": {}
        }
        
        print("=" * 50)
        print("正在加载模型...")
        print(f"模型路径: {self.model_path}")
        print(f"设备: {self.device}")
        print(f"Online模式: {self.use_online_mode}")
        print("=" * 50)
        
        # 加载模型（参考eval_speed_multiround.ipynb的实现）
        model_name = get_model_name_from_path(args.model_path)
        model_name += args.llm_type  # 添加llm_type
        
        model_load_start = time.time()
        
        self.cfg_pretrained = AutoConfig.from_pretrained(
            args.model_path, 
            trust_remote_code=True
        )
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            args.model_path, 
            args.model_base, 
            model_name, 
            load_8bit=args.load_8bit, 
            multimodal=True, 
            trust_remote_code=True,
            attn_implementation=args.attn_implementation
        )
        self.model.to(torch.float16)
        
        model_load_time = time.time() - model_load_start
        
        # 设置pad token
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                print("为Qwen模型设置pad token")
                self.tokenizer.pad_token_id = 151643
        
        # 获取配置（参考eval_speed_multiround.ipynb）
        if getattr(self.model.config, "force_sample", None) is not None:
            self.force_sample = self.model.config.force_sample
        else:
            self.force_sample = args.force_sample
        
        if getattr(self.model.config, "add_time_instruction", None) is not None:
            self.add_time_instruction = self.model.config.add_time_instruction
        else:
            self.add_time_instruction = False
        
        print("模型加载完成！")
        print("=" * 50)
        
        # 保存模型信息
        self.inference_stats["model_info"] = {
            "model_path": self.model_path,
            "model_name": model_name,
            "device": self.device,
            "model_load_time": model_load_time,
            "attn_implementation": args.attn_implementation,
            "load_8bit": args.load_8bit
        }
        
        # 初始化视频信息
        self.video_loaded = False
        self.video_frames = None
        self.image_sizes = None
        self.video_info = {}
        
    def load_video(self, video_path, question_time=0):
        """
        加载视频（参考eval_speed_multiround.ipynb的实现）
        
        Args:
            video_path: 视频路径
            question_time: 问题时间点（秒）
        """
        print(f"\n正在加载视频: {video_path}")
        
        video_load_start = time.time()
        
        # 检查视频是否存在
        if not ('s3://' in video_path or os.path.exists(video_path)):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 确定视频读取类型
        if os.path.isdir(video_path):
            media_dict = {'video_read_type': 'img'}
        else:
            media_dict = {'video_read_type': 'decord'}
        
        # 处理video_path（可能是列表）
        if type(video_path) != str:
            assert len(video_path) == 1, video_path
            video_path = video_path[0]
        
        # 设置时间片段
        if question_time > 0:
            clip = [0, question_time]
        else:
            clip = None
        
        # S3支持
        if 's3://' in video_path:
            from petrel_client.client import Client
            client = Client(conf_path='~/petreloss.conf')
        else:
            client = None
        
        # 读取视频帧
        if 'fps' in media_dict:
            frames, frame_indices, fps, duration = VIDEO_READER_FUNCS[media_dict['video_read_type']](
                video_path=video_path, 
                num_frames=self.max_num_frames, 
                sample='dynamic_fps1', 
                fix_start=None, 
                min_num_frames=4, 
                max_num_frames=self.max_num_frames, 
                client=client, 
                clip=clip, 
                local_num_frames=1,
                fps=media_dict['fps']
            )
        else:
            frames, frame_indices, fps, duration = VIDEO_READER_FUNCS[media_dict['video_read_type']](
                video_path=video_path, 
                num_frames=self.max_num_frames, 
                sample='dynamic_fps1', 
                fix_start=None, 
                min_num_frames=4, 
                max_num_frames=self.max_num_frames, 
                client=client, 
                clip=clip, 
                local_num_frames=1
            )
        
        print(f"视频时长: {duration:.2f}秒")
        print(f"加载帧数: {len(frames)}")
        print(f"FPS: {fps}")
        
        # 预处理帧
        preprocess_start = time.time()
        self.image_sizes = [frames[0].shape[:2]]
        frames_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        self.video_frames = [frames_tensor.to(dtype=self.model.dtype).to(self.device)]
        preprocess_time = time.time() - preprocess_start
        
        self.video_loaded = True
        print("视频加载完成！")
        print(f"视频张量形状: {self.video_frames[0].shape}")
        
        video_load_time = time.time() - video_load_start
        
        # 保存视频信息
        self.video_info = {
            "video_path": video_path,
            "num_frames": len(frames),
            "duration": duration,
            "fps": fps,
            "frame_shape": self.image_sizes[0],
            "question_time": question_time,
            "video_load_time": video_load_time,
            "preprocess_time": preprocess_time
        }
        self.inference_stats["video_info"] = self.video_info
        
        return self.video_info
    
    def ask(self, question, time_msg=""):
        """
        提问并使用generate_online进行推理
        
        Args:
            question: 问题文本
            time_msg: 时间消息（可选）
            
        Returns:
            回答文本和推理统计信息
        """
        if not self.video_loaded:
            raise ValueError("请先加载视频！")
        
        conv_mode = self.args.conv_mode
        
        print(f"\n{'='*50}")
        print(f"问题: {question}")
        print("【使用 Online 模式（generate_online）】")
        print(f"{'='*50}")
        
        # 构建问题文本
        qs = question
        if time_msg and time_msg != "":
            qs = f'{time_msg.strip()}\n{qs}'
        
        # 添加图像token
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        
        # 创建对话模板
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        
        # 获取prompt
        prompt = conv.get_prompt()
        
        # Tokenize
        tokenize_start = time.time()
        input_ids = tokenizer_image_token(
            prompt, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.device)
        tokenize_time = time.time() - tokenize_start
        
        # 设置停止条件
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        # 生成回答
        print("\n正在进行generate_online推理...")
        
        inference_start = time.time()
        
        with torch.inference_mode():
            # 使用Online模式进行推理
            output_ids = self.model.generate_online(
                inputs=input_ids,
                images=self.video_frames,
                attention_mask=attention_masks,
                modalities=["video"],
                image_sizes=self.image_sizes,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=1,  # 生成1个token用于测时间
                num_beams=1,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
        
        inference_time = time.time() - inference_start
        
        # 解码输出
        decode_start = time.time()
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # 移除停止符
        if output_text.endswith(stop_str):
            output_text = output_text[:-len(stop_str)]
        output_text = output_text.strip()
        decode_time = time.time() - decode_start
        
        print(f"\n回答: {output_text}\n")
        
        # 计算性能指标
        num_frames = self.video_info["num_frames"]
        fps = num_frames / inference_time if inference_time > 0 else 0
        time_per_frame = inference_time / num_frames if num_frames > 0 else 0
        
        # 保存推理结果
        self.inference_stats["inference_result"] = {
            "question": question,
            "answer": output_text,
            "tokenize_time": tokenize_time,
            "inference_time": inference_time,
            "decode_time": decode_time,
            "total_time": tokenize_time + inference_time + decode_time,
            "num_frames": num_frames,
            "fps": fps,
            "time_per_frame_ms": time_per_frame * 1000,
            "input_tokens": input_ids.numel(),
            "output_tokens": len(output_ids[0])
        }
        
        # 打印性能统计
        print("\n" + "="*50)
        print("推理性能统计")
        print("="*50)
        print(f"Tokenize 耗时: {tokenize_time*1000:.2f} ms")
        print(f"Inference 耗时: {inference_time:.3f} s")
        print(f"Decode 耗时: {decode_time*1000:.2f} ms")
        print(f"总耗时: {(tokenize_time + inference_time + decode_time):.3f} s")
        print(f"处理帧数: {num_frames}")
        print(f"平均速度: {fps:.3f} fps")
        print(f"单帧耗时: {time_per_frame*1000:.2f} ms")
        print(f"输入tokens数: {input_ids.numel()}")
        print(f"输出tokens数: {len(output_ids[0])}")
        print("="*50 + "\n")
        
        return output_text
    
    def save_inference_profile(self, output_path):
        """
        保存详细的推理性能统计
        
        Args:
            output_path: 输出文件路径
        """
        profile_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "online_inference",
            "model_info": self.inference_stats["model_info"],
            "video_info": self.inference_stats["video_info"],
            "inference_result": self.inference_stats["inference_result"]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n推理性能统计已保存到: {output_path}")


def main():
    """主函数 - 运行单轮对话推理测试"""
    
    # 解析命令行参数
    args = parse_args()
    
    # 单个问题
    question = "What is the main scene of the video?"
    
    print("\n" + "=" * 50)
    print("StreamForest 单轮对话推理测试")
    print("=" * 50)
    print(f"模型: {args.model_path}")
    print(f"视频: {args.video_path}")
    print(f"设备: {args.inference_device}")
    print(f"Online模式: {args.use_online_mode}")
    print(f"问题: {question}")
    print("=" * 50)
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 创建测试器
    tester = OnlineChatTester(args)
    
    # 加载视频
    tester.load_video(args.video_path, args.question_time)
    
    # 进行对话
    print("\n" + "=" * 50)
    print("开始单轮对话推理测试")
    print("=" * 50)
    
    answer = tester.ask(question, "")
    
    # 保存推理性能统计
    output_path = os.path.join(
        args.output_dir, 
        f"{args.output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    tester.save_inference_profile(output_path)
    
    print("\n" + "=" * 50)
    print("单轮对话推理测试完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()

