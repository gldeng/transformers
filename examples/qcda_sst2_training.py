#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用SST-2情感分析数据集训练和测试QCDA模型
"""

import os
import sys
import subprocess
import importlib.util

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
            # 需要重新启动脚本以使新安装的模块生效
            print("请重新运行脚本")
            sys.exit(0)
        else:
            print("请手动安装缺少的依赖后再运行脚本")
            print(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)

# 检查依赖
check_and_install_dependencies()

# 现在导入所需的包
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.transformers import AutoTokenizer
    from src.transformers.models.qcda.configuration_qcda import QCDAConfig
    from src.transformers.models.qcda.modeling_qcda import QCDAForSequenceClassification
except ImportError:
    print("无法导入QCDA模型，请确保已正确安装或模型文件在正确路径")
    sys.exit(1)


def load_sst2_dataset():
    """加载SST-2数据集"""
    print("加载SST-2数据集...")
    dataset = load_dataset("glue", "sst2")
    return dataset


def prepare_dataset(dataset, tokenizer, max_length=128):
    """准备数据集以供模型使用"""
    
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    # 对数据集进行tokenize处理
    print("对数据集进行预处理...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # 设置格式为PyTorch张量
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    
    return tokenized_dataset


def create_dataloaders(tokenized_dataset, batch_size=16):
    """创建训练和评估用的DataLoader"""
    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        batch_size=batch_size
    )
    
    eval_dataloader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=batch_size
    )
    
    return train_dataloader, eval_dataloader


def initialize_model():
    """初始化QCDA模型用于序列分类"""
    print("初始化QCDA模型...")
    
    # 使用较小的配置以加速训练
    config = QCDAConfig(
        vocab_size=30522,  # BERT词表大小
        hidden_size=256,
        num_hidden_layers=3,
        num_attention_heads=8,
        intermediate_size=512,
        
        # 量子-经典二元论特殊参数
        quantum_dim=16,  # 使用较小的维度
        classical_dim=16,
        interface_dim=8,
        
        # 分类任务参数
        num_labels=2  # SST-2是二分类
    )
    
    # 创建分类模型
    model = QCDAForSequenceClassification(config)
    
    return model


def train_model(model, train_dataloader, eval_dataloader, device, num_epochs=3):
    """训练模型并在验证集上评估"""
    print(f"使用设备: {device}")
    model.to(device)
    
    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # 开始训练
    print(f"开始训练，共{num_epochs}个epochs...")
    
    best_accuracy = 0.0
    
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
        eval_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # 将数据移至设备
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # 前向传播
                outputs = model(**batch)
                loss = outputs["loss"]
                eval_loss += loss.item()
                
                # 获取预测结果
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].cpu().numpy())
        
        # 计算准确率
        accuracy = accuracy_score(all_labels, all_preds)
        avg_eval_loss = eval_loss / len(eval_dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证损失: {avg_eval_loss:.4f}")
        print(f"  验证准确率: {accuracy:.4f}")
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"  保存新的最佳模型，准确率: {best_accuracy:.4f}")
            torch.save(model.state_dict(), "qcda_sst2_best.pt")
    
    return model, best_accuracy


def evaluate_model(model, eval_dataloader, device):
    """在测试集上评估模型"""
    print("在测试集上评估模型...")
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            logits = outputs["logits"]
            
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"测试集准确率: {accuracy:.4f}")
    
    return accuracy


def main():
    """主函数"""
    print("=" * 50)
    print("使用SST-2数据集训练和测试QCDA模型")
    print("=" * 50)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载BERT tokenizer (使用QCDA兼容)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 加载数据集
    dataset = load_sst2_dataset()
    
    # 准备数据集
    tokenized_dataset = prepare_dataset(dataset, tokenizer)
    
    # 创建数据加载器
    train_dataloader, eval_dataloader = create_dataloaders(tokenized_dataset)
    
    # 初始化模型
    model = initialize_model()
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    model, best_accuracy = train_model(
        model,
        train_dataloader,
        eval_dataloader,
        device,
        num_epochs=3
    )
    
    # 加载最佳模型权重
    print("加载最佳模型...")
    model.load_state_dict(torch.load("qcda_sst2_best.pt"))
    
    # 在验证集上测试
    final_accuracy = evaluate_model(model, eval_dataloader, device)
    
    print(f"训练完成! 最佳验证准确率: {best_accuracy:.4f}")
    print(f"最终测试准确率: {final_accuracy:.4f}")


if __name__ == "__main__":
    main() 