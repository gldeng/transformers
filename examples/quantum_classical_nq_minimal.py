#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用量子经典递归模型（Quantum Classical Model）处理问答数据集
简化版本 - 避免AutoTokenizer冲突
"""

import os
import sys
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm.auto import tqdm
import collections
import string
import re

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 导入我们的自定义模型，但不使用AUTO类
try:
    from src.transformers.models.quantum_classical.configuration_quantum_classical import QuantumClassicalConfig
    from src.transformers.models.quantum_classical.modeling_quantum_classical import QuantumClassicalModel
    from src.transformers.models.bert.tokenization_bert import BertTokenizer
except ImportError:
    print("无法导入量子经典模型，请确保已正确安装或模型文件在正确路径")
    sys.exit(1)

# 为问答任务创建Quantum Classical模型
class QuantumClassicalForQuestionAnswering(torch.nn.Module):
    """
    用于问答任务的量子经典递归模型
    利用递归自适应优化机制处理问答任务
    """
    
    def __init__(self, config):
        super().__init__()
        self.num_labels = 2  # start and end positions
        self.config = config
        
        self.quantum_classical = QuantumClassicalModel(config)
        self.qa_outputs = torch.nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        start_positions=None,
        end_positions=None,
        output_hidden_states=False,
        return_dict=True,
    ):
        outputs = self.quantum_classical(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 获取最后一层的隐藏状态
        if return_dict:
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs[0]
        
        # 预测开始和结束位置
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果提供了正确位置，则计算损失
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
            
        return {
            "loss": total_loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
        }


class SimplifiedQADataset(Dataset):
    """简化的问答数据集类"""
    def __init__(self, tokenizer, dataset, max_length=384):
        self.tokenizer = tokenizer
        self.examples = []
        self.max_length = max_length
        
        print("准备数据集...")
        
        # 检测是否是SQuAD格式
        is_squad = "context" in dataset[0] and "answers" in dataset[0]
        
        for i, example in enumerate(tqdm(dataset, desc="处理数据集")):
            if is_squad:
                # SQuAD格式数据处理
                question = example["question"]
                context = example["context"]
                
                # 标记化
                inputs = tokenizer(
                    question,
                    context,
                    max_length=max_length,
                    truncation="only_second",
                    stride=128,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                # 处理答案位置
                start_positions = 0
                end_positions = 0
                
                if "answers" in example and example["answers"]["answer_start"]:
                    start_char = example["answers"]["answer_start"][0]
                    end_char = start_char + len(example["answers"]["text"][0])
                    
                    # 将字符位置转换为token位置 (简化处理)
                    tokens = tokenizer.encode(context)
                    char_to_token = {}
                    token_idx = 0
                    char_idx = 0
                    
                    for token in tokenizer.tokenize(context):
                        if token.startswith("##"):
                            token = token[2:]
                        char_to_token[char_idx] = token_idx
                        char_idx += len(token)
                        token_idx += 1
                    
                    # 找到最近的token位置
                    closest_start = min(char_to_token.keys(), key=lambda x: abs(x - start_char))
                    start_positions = char_to_token[closest_start]
                    
                    # 结束位置 (简化)
                    end_positions = start_positions + 5
                    if end_positions >= len(tokens):
                        end_positions = len(tokens) - 1
                
                example_data = {
                    "input_ids": inputs["input_ids"][0],
                    "attention_mask": inputs["attention_mask"][0],
                    "token_type_ids": inputs["token_type_ids"][0] if "token_type_ids" in inputs else None,
                    "start_positions": torch.tensor(start_positions),
                    "end_positions": torch.tensor(end_positions)
                }
                
                self.examples.append(example_data)
            else:
                # 非SQuAD格式，仅提取问题和上下文
                question = ""
                context = ""
                
                # 尝试提取问题
                if "question" in example:
                    if isinstance(example["question"], dict) and "text" in example["question"]:
                        question = example["question"]["text"]
                    else:
                        question = str(example["question"])
                
                # 尝试提取上下文
                if "context" in example:
                    context = example["context"]
                elif "document" in example:
                    doc = example["document"]
                    if isinstance(doc, dict) and "text" in doc:
                        context = doc["text"][:2000]  # 限制长度
                    elif isinstance(doc, str):
                        context = doc[:2000]
                
                # 标记化
                inputs = tokenizer(
                    question,
                    context,
                    max_length=max_length,
                    truncation="only_second",
                    stride=128,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                # 简化处理，使用0作为默认位置
                example_data = {
                    "input_ids": inputs["input_ids"][0],
                    "attention_mask": inputs["attention_mask"][0],
                    "token_type_ids": inputs["token_type_ids"][0] if "token_type_ids" in inputs else None,
                    "start_positions": torch.tensor(0),
                    "end_positions": torch.tensor(0)
                }
                
                self.examples.append(example_data)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    """主函数"""
    print("=" * 50)
    print("使用量子经典递归模型处理问答数据集 (简化版)")
    print("=" * 50)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 使用BertTokenizer替代AutoTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # 加载SQuAD数据集
        print("加载SQuAD数据集...")
        dataset = load_dataset("squad", split="train[:200]")
        print(f"加载了SQuAD数据集，样本数: {len(dataset)}")
        
        # 准备简化数据集
        train_dataset = SimplifiedQADataset(tokenizer, dataset)
        
        # 拆分训练和验证集
        dataset_size = len(train_dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # 创建数据加载器
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        eval_dataloader = DataLoader(val_dataset, batch_size=8)
        
        print(f"训练数据加载器大小: {len(train_dataloader)}")
        print(f"评估数据加载器大小: {len(eval_dataloader)}")
        
        # 初始化量子经典模型
        config = QuantumClassicalConfig(
            vocab_size=30522,  # BERT词表大小
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            intermediate_size=512,
            gram_gamma=0.6,  # 递归步长因子
            num_recursive_steps=3,  # 递归优化次数
            gram_heads=4,  # GRAM中的注意力头数
            max_position_embeddings=512
        )
        
        # 创建问答模型
        model = QuantumClassicalForQuestionAnswering(config)
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 移至设备并设置训练参数
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=5e-5)
        
        # 简单训练循环
        print("开始训练...")
        model.train()
        for epoch in range(1):  # 只训练一个epoch
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != "token_type_ids" or v is not None}
                
                outputs = model(**batch)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                
                if step % 10 == 0:
                    print(f"Batch {step}: loss = {loss.item():.4f}")
                    
                if step >= 20:  # 仅运行20个批次进行测试
                    break
            
            avg_loss = total_loss / (step + 1)
            print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
        
        # 保存模型
        model_save_path = "quantum_classical_qa_model_minimal.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"模型保存至: {model_save_path}")
        
        # 显示递归自适应优化机制的效果分析
        print("\n递归自适应优化机制效果分析:")
        print(f"递归步长因子(gamma): {config.gram_gamma}")
        print(f"递归优化次数: {config.num_recursive_steps}")
        print("递归自适应优化允许模型在同一信息上进行多次处理，")
        print("理论上可以提高复杂问答任务的准确性，特别是需要多轮推理的问题。")
            
    except Exception as e:
        print(f"运行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 