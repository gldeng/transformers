#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用量子经典递归模型（Quantum Classical Model）处理Natural Questions数据集
展示如何将递归自适应优化机制应用于问答任务
"""

import os
import sys
import subprocess
import importlib.util
import torch
import numpy as np
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import collections
import string
import re

# 函数检查并安装缺失的依赖
def check_and_install_dependencies():
    required_packages = {
        "datasets": "datasets",
        "torch": "torch",
        "numpy": "numpy",
        "tqdm": "tqdm",
        "sklearn.metrics": "scikit-learn",
        "transformers": "transformers"
    }
    
    missing_packages = []
    
    for module, package in required_packages.items():
        if importlib.util.find_spec(module.split('.')[0]) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少以下依赖包: {', '.join(missing_packages)}")
        install = input("是否自动安装这些依赖? (y/n): ").strip().lower()
        if install == 'y':
            for package in missing_packages:
                print(f"安装 {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("所有依赖已安装")
            print("请重新运行脚本")
            sys.exit(0)
        else:
            print("请手动安装缺少的依赖后再运行脚本")
            print(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)

# 检查依赖
check_and_install_dependencies()

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.transformers.models.quantum_classical.configuration_quantum_classical import QuantumClassicalConfig
    from src.transformers.models.quantum_classical.modeling_quantum_classical import QuantumClassicalModel
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


def load_natural_questions_dataset(subset_size=5000):
    """加载Natural Questions数据集，并取部分用于快速实验"""
    print("加载Natural Questions数据集...")
    
    try:
        # 尝试直接加载数据集
        dataset = load_dataset("natural_questions", split="train")
    except Exception as e:
        print(f"加载完整数据集失败: {e}")
        print("尝试加载简化版本...")
        # 加载验证集，通常更小更易处理
        try:
            dataset = load_dataset("natural_questions", split="validation")
        except:
            # 如果仍然失败，尝试加载squad数据集作为替代
            print("尝试改用SQuAD数据集...")
            dataset = load_dataset("squad", split="train")
    
    # 打印数据集信息
    print(f"数据集特征: {dataset.features}")
    print(f"原始数据集大小: {len(dataset)}个样本")
    
    # 为了快速实验，我们只取部分数据
    if subset_size and subset_size < len(dataset):
        dataset = dataset.select(range(subset_size))
    
    print(f"使用数据集大小: {len(dataset)}个样本")
    
    # 显示一个样本示例
    print("\n数据集样本示例:")
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            print(f"{key}: {value}")
        else:
            print(f"{key}: {type(value)}")
    
    return dataset


def prepare_nq_features(examples, tokenizer, max_length=384, doc_stride=128, max_query_length=64):
    """处理Natural Questions数据集的样本，准备模型输入特征"""
    # 检查数据结构并打印示例
    print("Dataset structure sample:")
    for key in examples:
        print(f"Key: {key}, Type: {type(examples[key])}")
        if isinstance(examples[key], dict):
            print(f"  Subkeys: {examples[key].keys()}")
    
    # 正确提取问题 - Natural Questions格式
    questions = []
    for q in examples["question"]:
        if isinstance(q, dict) and "text" in q:
            questions.append(q["text"].strip())
        elif isinstance(q, str):
            questions.append(q.strip())
        else:
            # 如果问题不符合预期格式，使用空字符串
            questions.append("")
    
    contexts = []
    answers = {"text": [], "answer_start": []}
    
    # 遍历每个样本处理上下文和答案
    for idx in range(len(questions)):
        # 获取文档内容
        if "document" in examples and idx < len(examples["document"]):
            doc = examples["document"][idx]
            # 尝试提取文档内容
            if isinstance(doc, dict) and "tokens" in doc and "token" in doc["tokens"]:
                # 使用前500个标记作为上下文
                doc_tokens = doc["tokens"]["token"][:500]
                context = " ".join(doc_tokens)
            else:
                context = ""
        else:
            context = ""
        
        contexts.append(context)
        
        # 处理答案
        answer_texts = []
        answer_starts = []
        
        if "annotations" in examples and idx < len(examples["annotations"]):
            annotation = examples["annotations"][idx]
            
            # 尝试从短答案中提取
            if isinstance(annotation, dict) and "short_answers" in annotation and annotation["short_answers"]:
                for short_ans in annotation["short_answers"]:
                    if "start_token" in short_ans and "end_token" in short_ans:
                        try:
                            # 处理列表形式的token索引
                            if isinstance(short_ans["start_token"], list):
                                # 如果是空列表，跳过
                                if not short_ans["start_token"]:
                                    continue
                                # 如果非空，取第一个元素
                                start_token = int(short_ans["start_token"][0])
                            else:
                                # 正常情况 - 直接转换
                                start_token = int(short_ans["start_token"])
                                
                            if isinstance(short_ans["end_token"], list):
                                # 如果是空列表，跳过
                                if not short_ans["end_token"]:
                                    continue
                                # 如果非空，取第一个元素
                                end_token = int(short_ans["end_token"][0])
                            else:
                                # 正常情况 - 直接转换
                                end_token = int(short_ans["end_token"])
                            
                            # 如果有文档和标记，提取答案文本
                            if "document" in examples and idx < len(examples["document"]):
                                doc = examples["document"][idx]
                                if isinstance(doc, dict) and "tokens" in doc and "token" in doc["tokens"]:
                                    # 确保不超出范围
                                    tokens = doc["tokens"]["token"]
                                    start_token = max(0, min(start_token, len(tokens)-1))
                                    end_token = max(start_token+1, min(end_token, len(tokens)))
                                    
                                    answer_text = " ".join(tokens[start_token:end_token])
                                    
                                    # 计算在上下文中的字符位置
                                    char_start = 0
                                    if start_token < 500:  # 只有当答案在我们使用的上下文中时
                                        try:
                                            char_start = len(" ".join(tokens[:start_token]))
                                            if char_start > 0:
                                                char_start += 1  # 为空格添加
                                            answer_texts.append(answer_text)
                                            answer_starts.append(char_start)
                                        except Exception as e:
                                            print(f"字符位置计算错误: {e}")
                        except (ValueError, TypeError) as e:
                            print(f"答案标记索引错误: {e}, start_token={short_ans['start_token']}, end_token={short_ans['end_token']}")
        
        # 如果没有找到答案
        if not answer_texts:
            answer_texts = [""]
            answer_starts = [0]
            
        answers["text"].append(answer_texts)
        answers["answer_start"].append(answer_starts)
            
    # 对问题和上下文进行tokenize
    tokenized_examples = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        stride=doc_stride,
        truncation="only_second",
        padding="max_length",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    
    # 将样本ID映射到其特征
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    # 初始化标签
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        # 获取原始样本ID
        sample_idx = sample_mapping[i]
        
        # 提取答案信息
        answer_texts = answers["text"][sample_idx]
        answer_starts = answers["answer_start"][sample_idx]
        
        # 设置CLS标记为默认答案位置
        tokenized_examples["start_positions"].append(0)
        tokenized_examples["end_positions"].append(0)
        
        # 如果没有有效答案，使用CLS标记位置
        if not answer_texts[0]:
            continue
            
        # 获取当前答案
        answer_text = answer_texts[0]
        answer_start_char = answer_starts[0]
        answer_end_char = answer_start_char + len(answer_text)
        
        # 找到标记起始位置
        token_start_index = 0
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= answer_start_char:
            token_start_index += 1
        token_start_index -= 1
        
        # 找到标记结束位置
        token_end_index = len(offsets) - 1
        while token_end_index >= 0 and offsets[token_end_index][1] >= answer_end_char:
            token_end_index -= 1
        token_end_index += 1
        
        # 检查答案是否在上下文中
        if offsets[token_start_index][0] <= answer_start_char and offsets[token_end_index - 1][1] >= answer_end_char:
            tokenized_examples["start_positions"][-1] = token_start_index
            tokenized_examples["end_positions"][-1] = token_end_index
    
    return tokenized_examples


def prepare_natural_questions_dataset(dataset, tokenizer):
    """准备Natural Questions或SQuAD数据集以进行训练和评估"""
    
    # 检测数据集类型并打印结构
    print("数据集结构:")
    first_example = dataset[0]
    for key, value in first_example.items():
        print(f"{key}: {type(value)}")
    
    # 检查是否是SQuAD格式
    is_squad = "context" in first_example and "question" in first_example and "answers" in first_example
    
    if is_squad:
        print("检测到SQuAD格式数据集，使用SQuAD处理逻辑")
        
        # 定义SQuAD数据处理函数
        def process_squad(examples):
            return tokenizer(
                examples["question"],
                examples["context"],
                max_length=384,
                stride=128,
                truncation="only_second",
                padding="max_length",
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
            )
        
        # 拆分数据集
        dataset_split = dataset.train_test_split(test_size=0.2)
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]
        
        # 处理训练数据
        train_tokenized = train_dataset.map(
            process_squad,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        # 设置训练数据的答案位置
        train_features = train_tokenized.map(
            lambda examples: set_squad_answer_positions(examples, train_dataset, tokenizer),
            batched=True
        )
        
        # 处理评估数据
        eval_tokenized = eval_dataset.map(
            process_squad,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        # 设置评估数据的答案位置
        eval_features = eval_tokenized.map(
            lambda examples: set_squad_answer_positions(examples, eval_dataset, tokenizer),
            batched=True
        )
        
        return train_features, eval_features, eval_dataset
    
    else:
        print("使用通用处理逻辑...")
        
        # 将数据集拆分为训练集和评估集
        dataset_split = dataset.train_test_split(test_size=0.2)
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]
        
        # 打印训练集大小
        print(f"训练集大小: {len(train_dataset)}")
        print(f"评估集大小: {len(eval_dataset)}")
        
        # 创建简化版本的特征处理函数
        def simplified_features(examples):
            """简化的特征提取，适用于各种QA数据集格式"""
            # 获取问题和上下文 (根据可用字段自适应)
            questions = []
            contexts = []
            
            for i in range(len(examples["question"]) if "question" in examples else 1):
                # 提取问题
                if "question" in examples:
                    if isinstance(examples["question"][i], dict) and "text" in examples["question"][i]:
                        questions.append(examples["question"][i]["text"])
                    elif isinstance(examples["question"][i], str):
                        questions.append(examples["question"][i])
                    else:
                        questions.append("未知问题")
                else:
                    questions.append("未知问题")
                
                # 提取上下文
                if "context" in examples:
                    contexts.append(examples["context"][i])
                elif "document" in examples and i < len(examples["document"]):
                    doc = examples["document"][i]
                    if isinstance(doc, dict) and "text" in doc:
                        contexts.append(doc["text"][:2000])  # 截取前2000个字符
                    elif isinstance(doc, str):
                        contexts.append(doc[:2000])
                    else:
                        contexts.append("未知上下文")
                else:
                    contexts.append("未知上下文")
            
            # 使用tokenizer处理问题和上下文
            tokenized = tokenizer(
                questions,
                contexts,
                max_length=384,
                stride=128,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt"
            )
            
            # 添加虚拟答案位置
            tokenized["start_positions"] = [0] * len(questions)
            tokenized["end_positions"] = [0] * len(questions)
            
            return tokenized
        
        # 处理训练和评估数据
        train_features = train_dataset.map(
            simplified_features,
            batched=True,
            batch_size=8,
            remove_columns=train_dataset.column_names
        )
        
        eval_features = eval_dataset.map(
            simplified_features,
            batched=True,
            batch_size=8,
            remove_columns=eval_dataset.column_names
        )
        
        return train_features, eval_features, eval_dataset


def set_squad_answer_positions(examples, dataset, tokenizer):
    """为SQuAD格式数据集设置答案的开始和结束位置"""
    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(examples["offset_mapping"]):
        sample_idx = examples["overflow_to_sample_mapping"][i]
        answer = dataset[sample_idx]["answers"][0]
        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])
        
        # 将CLS标记映射到示例中
        sequence_ids = examples.sequence_ids(i)
        
        # 查找上下文的开始和结束索引
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        
        # 如果答案不在上下文中，则将开始/结束位置设置为CLS索引
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # 否则，找到答案的开始和结束标记位置
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    
    examples["start_positions"] = start_positions
    examples["end_positions"] = end_positions
    return examples


def predict_answers(model, eval_dataloader, tokenizer, device):
    """使用模型预测答案"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 前向传播
            outputs = model(**batch)
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]
            
            # 获取最可能的答案位置
            start_indices = torch.argmax(start_logits, dim=1)
            end_indices = torch.argmax(end_logits, dim=1)
            
            # 获取输入ID
            input_ids = batch["input_ids"].cpu().numpy()
            
            # 提取答案文本
            for i in range(len(start_indices)):
                start_idx = start_indices[i].item()
                end_idx = end_indices[i].item()
                
                # 确保开始索引在结束索引之前
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                
                # 提取标记并转换为文本
                answer_tokens = input_ids[i][start_idx:end_idx+1]
                answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                all_predictions.append(answer_text)
    
    return all_predictions


def normalize_answer(s):
    """规范化答案文本进行评估"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def text_eval_metric(references, predictions):
    """计算精确率、召回率和F1分数"""
    normalized_references = [normalize_answer(reference) for reference in references]
    normalized_predictions = [normalize_answer(prediction) for prediction in predictions]
    
    exact_matches = 0
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    
    for reference, prediction in zip(normalized_references, normalized_predictions):
        if not reference and not prediction:
            exact_matches += 1
            precision_sum += 1
            recall_sum += 1
            f1_sum += 1
            continue
        
        reference_tokens = reference.split()
        prediction_tokens = prediction.split()
        
        # 简单的计算精确率和召回率
        common_tokens = set(prediction_tokens) & set(reference_tokens)
        
        # 防止除以零
        if not prediction_tokens:
            precision = 0.0
        else:
            precision = len(common_tokens) / len(prediction_tokens)
            
        if not reference_tokens:
            recall = 0.0
        else:
            recall = len(common_tokens) / len(reference_tokens)
            
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        
        if reference == prediction:
            exact_matches += 1
    
    examples_count = len(normalized_references)
    precision = precision_sum / examples_count
    recall = recall_sum / examples_count
    f1 = f1_sum / examples_count
    exact_match = exact_matches / examples_count
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match
    }


def main():
    """主函数"""
    print("=" * 50)
    print("使用量子经典递归模型处理问答数据集")
    print("=" * 50)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 加载BERT tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # 尝试加载Natural Questions数据集
        try:
            dataset = load_natural_questions_dataset(subset_size=100)
        except Exception as e:
            print(f"加载Natural Questions失败: {e}")
            print("切换到SQuAD数据集...")
            dataset = load_dataset("squad", split="train[:500]")
            print(f"加载了SQuAD数据集，样本数: {len(dataset)}")
        
        # 准备数据集
        try:
            train_features, eval_features, eval_dataset = prepare_natural_questions_dataset(dataset, tokenizer)
            
            # 创建数据加载器
            train_dataloader = DataLoader(train_features, batch_size=8, shuffle=True)
            eval_dataloader = DataLoader(eval_features, batch_size=8)
            
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
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs["loss"]
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                    
                    if step % 50 == 0:
                        print(f"Batch {step}: loss = {loss.item():.4f}")
                        
                    if step >= 100:  # 仅运行100个批次进行测试
                        break
                
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
            
            # 保存模型
            model_save_path = "quantum_classical_qa_model.pt"
            torch.save(model.state_dict(), model_save_path)
            print(f"模型保存至: {model_save_path}")
            
            # 评估
            print("开始评估...")
            predictions = predict_answers(model, eval_dataloader, tokenizer, device)
            
            # 获取参考答案
            references = []
            if "answers" in eval_dataset.features:  # SQuAD格式
                for example in eval_dataset:
                    references.append(example["answers"]["text"][0])
            else:  # Natural Questions格式
                for example in eval_dataset:
                    if "annotations" in example and example["annotations"]:
                        ans = example["annotations"].get("short_answers", [])
                        if ans:
                            # 尝试提取答案文本
                            try:
                                references.append("有答案但格式不支持提取")  # 简化处理
                            except:
                                references.append("")
                        else:
                            references.append("")
                    else:
                        references.append("")
            
            # 评价结果
            print(f"预测答案样例: {predictions[:5]}")
            
            # 如果参考答案有效，计算指标
            if references and all(isinstance(r, str) for r in references):
                metrics = text_eval_metric(references[:len(predictions)], predictions)
                print("\n评估结果:")
                print(f"精确率: {metrics['precision']:.4f}")
                print(f"召回率: {metrics['recall']:.4f}")
                print(f"F1分数: {metrics['f1']:.4f}")
                print(f"完全匹配: {metrics['exact_match']:.4f}")
            
            print("训练和评估完成!")
            
            # 显示递归自适应优化机制的效果
            print("\n递归自适应优化机制效果分析:")
            print(f"递归步长因子(gamma): {config.gram_gamma}")
            print(f"递归优化次数: {config.num_recursive_steps}")
            print("递归自适应优化允许模型在同一信息上进行多次处理，")
            print("理论上可以提高复杂问答任务的准确性，特别是需要多轮推理的问题。")
            
        except Exception as e:
            print(f"数据处理或训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"初始化过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 