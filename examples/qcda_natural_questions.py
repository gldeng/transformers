#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用Google Natural Questions数据集训练和测试QCDA模型用于问答任务
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
from sklearn.metrics import accuracy_score
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
    from src.transformers.models.qcda.configuration_qcda import QCDAConfig
    from src.transformers.models.qcda.modeling_qcda import QCDAPreTrainedModel, QCDAModel
except ImportError:
    print("无法导入QCDA模型，请确保已正确安装或模型文件在正确路径")
    sys.exit(1)

# 为问答任务创建QCDA模型
class QCDAForQuestionAnswering(QCDAPreTrainedModel):
    """问答任务的QCDA模型"""
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2  # start and end positions
        
        self.qcda = QCDAModel(config)
        self.qa_outputs = torch.nn.Linear(config.hidden_size, config.num_labels)
        
        # 初始化权重
        self.post_init()
        
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
        outputs = self.qcda(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs["last_hidden_state"]
        
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
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output
            
        return {
            "loss": total_loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "hidden_states": outputs.get("hidden_states", None),
        }


def load_natural_questions_dataset(subset_size=5000):
    """加载Natural Questions数据集，并取部分用于快速实验"""
    print("加载Natural Questions数据集...")
    
    # 加载数据集 (仅有效问答对)
    dataset = load_dataset("natural_questions", split="train")
    
    # 为了快速实验，我们只取部分数据
    if subset_size and subset_size < len(dataset):
        dataset = dataset.select(range(subset_size))
    
    print(f"数据集大小: {len(dataset)}个样本")
    return dataset


def prepare_nq_features(examples, tokenizer, max_length=384, doc_stride=128, max_query_length=64):
    """处理Natural Questions数据集的样本，准备模型输入特征"""
    # 提取问题和上下文
    questions = [q.strip() for q in examples["question"]["text"]]
    contexts = []
    answers = {"text": [], "answer_start": []}
    
    for annotations in examples["annotations"]:
        # 从第一个长答案中提取上下文
        if annotations["long_answer"] and "start_token" in annotations["long_answer"]:
            start_token = annotations["long_answer"]["start_token"]
            end_token = annotations["long_answer"]["end_token"]
            document_tokens = examples["document"]["tokens"]
            document_text = document_tokens["token"]
            
            # 提取长答案作为上下文
            context = " ".join(document_text[start_token:end_token])
            contexts.append(context)
            
            # 获取短答案
            short_answers_text = []
            short_answers_start = []
            
            if annotations["short_answers"]:
                for short_answer in annotations["short_answers"]:
                    sa_start = short_answer["start_token"] - start_token
                    sa_end = short_answer["end_token"] - start_token
                    if sa_start >= 0 and sa_end <= len(context):
                        answer_text = " ".join(document_text[short_answer["start_token"]:short_answer["end_token"]])
                        short_answers_text.append(answer_text)
                        # 计算字符级别的起始位置
                        char_start = len(" ".join(document_text[start_token:short_answer["start_token"]]))
                        if char_start > 0:  # 添加额外的空格
                            char_start += 1
                        short_answers_start.append(char_start)
            
            # 如果没有短答案，使用长答案开头
            if not short_answers_text:
                short_answers_text = [""]
                short_answers_start = [0]
                
            answers["text"].append(short_answers_text)
            answers["answer_start"].append(short_answers_start)
        else:
            # 如果没有长答案，使用文档的前N个标记
            document_tokens = examples["document"]["tokens"]
            document_text = document_tokens["token"]
            context = " ".join(document_text[:500])  # 使用前500个标记
            contexts.append(context)
            answers["text"].append([""])
            answers["answer_start"].append([0])
    
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
        
        # 获取当前特征的上下文偏移
        cls_index = tokenized_examples["input_ids"][i].index(tokenizer.cls_token_id)
        
        # 获取该特征的序列ID
        sequence_ids = tokenized_examples.sequence_ids(i)
        
        # 标记不属于上下文的偏移为None
        offsets_with_context = [
            (o if sequence_ids[k] == 1 else None) for k, o in enumerate(offsets)
        ]
        offset_mapping[i] = offsets_with_context
        
        # 设置开始和结束位置为CLS索引（如果答案不在当前span中）
        tokenized_examples["start_positions"].append(cls_index)
        tokenized_examples["end_positions"].append(cls_index)
        
        # 检查答案是否在当前span中
        answer_texts = answers["text"][sample_idx]
        answer_starts = answers["answer_start"][sample_idx]
        
        if answer_texts and answer_texts[0]:  # 如果有答案
            answer_text = answer_texts[0]
            answer_start = answer_starts[0]
            
            # 将答案的字符级别位置转换为token级别
            # 找到包含答案开始的token
            token_start = 0
            while token_start < len(offsets_with_context) and (
                offsets_with_context[token_start] is None
                or answer_start >= offsets_with_context[token_start][1]
            ):
                token_start += 1
            
            # 找到包含答案结束的token
            answer_end = answer_start + len(answer_text)
            token_end = token_start
            while token_end < len(offsets_with_context) and (
                offsets_with_context[token_end] is None
                or answer_end > offsets_with_context[token_end][0]
            ):
                token_end += 1
            
            # 如果找到有效的起始和结束token
            if token_start < len(offsets) and token_end < len(offsets):
                tokenized_examples["start_positions"][-1] = token_start
                tokenized_examples["end_positions"][-1] = token_end - 1  # 减1是因为end是包含的
    
    return tokenized_examples


def prepare_natural_questions_dataset(dataset, tokenizer):
    """准备Natural Questions数据集"""
    print("对数据集进行预处理...")
    
    # 将数据集拆分为训练集和验证集
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"训练集大小: {len(train_dataset)}，验证集大小: {len(eval_dataset)}")
    
    # 对数据集进行特征处理
    train_features = train_dataset.map(
        lambda examples: prepare_nq_features(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    eval_features = eval_dataset.map(
        lambda examples: prepare_nq_features(examples, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )
    
    train_features.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]
    )
    
    eval_features.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]
    )
    
    return train_features, eval_features, eval_dataset


def create_dataloaders(train_features, eval_features, batch_size=8):
    """创建训练和评估用的DataLoader"""
    train_dataloader = DataLoader(
        train_features,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )
    
    eval_dataloader = DataLoader(
        eval_features,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )
    
    return train_dataloader, eval_dataloader


def initialize_qa_model():
    """初始化QCDA问答模型"""
    print("初始化QCDA问答模型...")
    
    config = QCDAConfig(
        vocab_size=30522,  # BERT词表大小
        hidden_size=256,
        num_hidden_layers=3,
        num_attention_heads=8,
        intermediate_size=512,
        
        # 量子-经典二元论特殊参数
        quantum_dim=16,
        classical_dim=16,
        interface_dim=8,
        
        # 问答任务参数
        num_labels=2  # 开始和结束位置
    )
    
    model = QCDAForQuestionAnswering(config)
    
    return model


def train_qa_model(model, train_dataloader, eval_dataloader, tokenizer, 
                  eval_examples, eval_dataset, device, num_epochs=3):
    """训练问答模型并在验证集上评估"""
    print(f"使用设备: {device}")
    model.to(device)
    
    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # 开始训练
    print(f"开始训练，共{num_epochs}个epochs...")
    
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        
        # 训练循环
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # 将数据移至设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs["loss"]
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_dataloader)
        
        # 评估模式
        model.eval()
        
        # 创建预测结果
        all_start_logits = []
        all_end_logits = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                outputs = model(**batch)
                
                start_logits = outputs["start_logits"]
                end_logits = outputs["end_logits"]
                
                all_start_logits.append(start_logits.cpu().numpy())
                all_end_logits.append(end_logits.cpu().numpy())
        
        # 连接结果
        start_logits = np.concatenate(all_start_logits)
        end_logits = np.concatenate(all_end_logits)
        
        # 评估结果
        metrics = compute_qa_metrics(
            start_logits, end_logits, eval_features, eval_examples, eval_dataset, tokenizer
        )
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证精确率: {metrics['precision']:.4f}")
        print(f"  验证召回率: {metrics['recall']:.4f}")
        print(f"  验证F1: {metrics['f1']:.4f}")
        
        # 保存最佳模型
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            print(f"  保存新的最佳模型，F1: {best_f1:.4f}")
            torch.save(model.state_dict(), "qcda_nq_best.pt")
    
    return model, best_f1


def compute_qa_metrics(start_logits, end_logits, features, examples, dataset, tokenizer):
    """计算问答模型的评估指标"""
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)
    
    predicted_answers = []
    reference_answers = []
    
    for example_id, example in enumerate(examples):
        feature_indices = example_to_features[example_id]
        
        min_null_score = float("inf")
        valid_answers = []
        
        for feature_index in feature_indices:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            
            # 找到空答案的分数
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            null_score = start_logit[cls_index] + end_logit[cls_index]
            
            # 如果我们比目前的最小分数更好，则保存这个空预测
            if null_score < min_null_score:
                min_null_score = null_score
            
            # 找到最佳答案
            start_indexes = np.argsort(start_logit)[-20:].tolist()
            end_indexes = np.argsort(end_logit)[-20:].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # 跳过无效的答案
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > 50:
                        continue
                    
                    valid_answers.append({
                        "score": start_logit[start_index] + end_logit[end_index],
                        "text": get_answer_text(
                            features[feature_index]["input_ids"][start_index:end_index + 1],
                            tokenizer
                        )
                    })
        
        # 找到最佳的非空答案
        if valid_answers:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]["text"]
        else:
            best_answer = ""
        
        predicted_answers.append(best_answer)
        
        # 获取参考答案文本
        if example["annotations"]["short_answers"]:
            short_answer = example["annotations"]["short_answers"][0]
            doc_tokens = example["document"]["tokens"]["token"]
            reference_text = " ".join(doc_tokens[short_answer["start_token"]:short_answer["end_token"]])
        else:
            reference_text = ""
        
        reference_answers.append(reference_text)
    
    # 计算评估指标
    precision, recall, f1 = text_eval_metric(reference_answers, predicted_answers)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def get_answer_text(token_ids, tokenizer):
    """将token ID转换为文本"""
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    # 将BPE标记合并成单词
    answer = tokenizer.convert_tokens_to_string(tokens)
    return answer


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
    
    return precision, recall, f1


def main():
    """主函数"""
    print("=" * 50)
    print("使用Google Natural Questions数据集训练和测试QCDA模型")
    print("=" * 50)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 加载Natural Questions数据集（使用较小的子集用于快速实验）
    dataset = load_natural_questions_dataset(subset_size=1000)
    
    # 准备数据集
    train_features, eval_features, eval_dataset = prepare_natural_questions_dataset(dataset, tokenizer)
    
    # 创建数据加载器
    train_dataloader, eval_dataloader = create_dataloaders(train_features, eval_features)
    
    # 初始化问答模型
    model = initialize_qa_model()
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    model, best_f1 = train_qa_model(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        eval_dataset,
        dataset,
        device,
        num_epochs=3
    )
    
    # 加载最佳模型权重
    print("加载最佳模型...")
    model.load_state_dict(torch.load("qcda_nq_best.pt"))
    
    # 打印结果
    print(f"训练完成! 最佳F1分数: {best_f1:.4f}")


if __name__ == "__main__":
    main() 