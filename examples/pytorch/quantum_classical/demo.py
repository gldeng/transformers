#!/usr/bin/env python
# coding=utf-8

import argparse
import torch
import numpy as np
from transformers import (
    QuantumClassicalConfig,
    QuantumClassicalModel,
    AutoTokenizer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden_size", 
        type=int, 
        default=768, 
        help="模型隐藏层大小"
    )
    parser.add_argument(
        "--num_hidden_layers", 
        type=int, 
        default=12, 
        help="Transformer层数"
    )
    parser.add_argument(
        "--num_attention_heads", 
        type=int, 
        default=12, 
        help="注意力头数"
    )
    parser.add_argument(
        "--gram_gamma", 
        type=float, 
        default=0.5, 
        help="GRAM递归步长因子"
    )
    parser.add_argument(
        "--num_recursive_steps", 
        type=int, 
        default=10, 
        help="递归优化次数"
    )
    parser.add_argument(
        "--gram_heads", 
        type=int, 
        default=8, 
        help="GRAM中的注意力头数"
    )
    parser.add_argument(
        "--test_text", 
        type=str, 
        default="这是一个测试句子，用于测试量子经典递归自适应Transformer模型的能力。", 
        help="用于测试的文本"
    )
    parser.add_argument(
        "--tokenizer_name", 
        type=str, 
        default="bert-base-chinese", 
        help="分词器名称"
    )
    args = parser.parse_args()

    # 使用BERT分词器来处理中文
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # 创建配置
    config = QuantumClassicalConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.hidden_size * 4,
        gram_gamma=args.gram_gamma,
        num_recursive_steps=args.num_recursive_steps,
        gram_heads=args.gram_heads,
        vocab_size=tokenizer.vocab_size
    )
    
    # 初始化量子经典递归自适应模型
    print("正在初始化量子经典递归自适应Transformer模型...")
    model = QuantumClassicalModel(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 编码测试文本
    inputs = tokenizer(
        args.test_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # 运行模型
    print(f"正在处理文本: \"{args.test_text}\"")
    print(f"使用递归步骤: {args.num_recursive_steps}")
    
    # 添加时间测量
    import time
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )
    
    elapsed_time = time.time() - start_time
    
    # 分析输出
    hidden_states = outputs.last_hidden_state
    pooled_output = outputs.pooler_output
    
    print(f"处理完成，耗时: {elapsed_time:.4f}秒")
    print(f"输出张量形状: {hidden_states.shape}")
    print(f"池化输出形状: {pooled_output.shape}")
    
    # 计算输出统计信息
    hidden_mean = torch.mean(hidden_states).item()
    hidden_std = torch.std(hidden_states).item()
    pooled_mean = torch.mean(pooled_output).item()
    pooled_std = torch.std(pooled_output).item()
    
    print("\n输出统计分析:")
    print(f"隐藏状态均值: {hidden_mean:.6f}，标准差: {hidden_std:.6f}")
    print(f"池化输出均值: {pooled_mean:.6f}，标准差: {pooled_std:.6f}")
    
    # 测试不同递归步骤
    if args.num_recursive_steps > 1:
        print("\n测试不同递归步骤的效果:")
        steps_to_test = [1, 5, args.num_recursive_steps, args.num_recursive_steps * 2]
        
        results = []
        for steps in steps_to_test:
            # 创建新配置和模型
            test_config = QuantumClassicalConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                num_attention_heads=args.num_attention_heads,
                intermediate_size=args.hidden_size * 4,
                gram_gamma=args.gram_gamma,
                num_recursive_steps=steps,
                gram_heads=args.gram_heads,
                vocab_size=tokenizer.vocab_size
            )
            test_model = QuantumClassicalModel(test_config)
            # 复用权重
            test_model.load_state_dict(model.state_dict())
            
            # 运行模型
            start_time = time.time()
            with torch.no_grad():
                test_outputs = test_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask
                )
            test_time = time.time() - start_time
            
            # 计算输出熵（通过标准差来近似）
            entropy = torch.std(test_outputs.last_hidden_state).item()
            results.append((steps, test_time, entropy))
        
        # 输出结果表格
        print("\n递归步骤 | 耗时(秒) | 信息熵近似")
        print("---------|----------|------------")
        for steps, timing, entropy in results:
            print(f"{steps:9} | {timing:.6f} | {entropy:.6f}")
        
        # 计算熵的减少趋势
        if len(results) > 1:
            entropy_reduction = (results[0][2] - results[-1][2]) / results[0][2] * 100
            time_increase = (results[-1][1] / results[0][1] - 1) * 100
            print(f"\n递归优化分析: 熵减少了 {entropy_reduction:.2f}%，时间增加了 {time_increase:.2f}%")


if __name__ == "__main__":
    main() 