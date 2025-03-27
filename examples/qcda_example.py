#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量子-经典动态注意力（QCDA）模型使用示例
此示例展示如何初始化和使用这个模型进行推理
"""

import torch
from transformers.models.qcda import (
    QCDAConfig, 
    QCDAModel, 
    QCDAForSequenceClassification
)


def main():
    """
    QCDA模型演示函数
    展示如何配置和使用基于量子-经典二元论的模型
    """
    print("初始化QCDA配置...")
    
    # 创建QCDA配置
    config = QCDAConfig(
        vocab_size=30522,  # BERT词表大小
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        
        # 量子-经典二元论特殊参数
        quantum_dim=64,
        classical_dim=64, 
        interface_dim=32,
        beta=1.0,  # 动态注意力调节参数
        gamma=0.1,  # 熵与经典知识调节步长
        eta=0.01,   # 维度自适应学习率
        lambda_factor=0.5,  # 界面域转换优化因子
        
        # 序列分类参数
        num_labels=2
    )
    
    print("配置参数:")
    print(f"- 量子域维度: {config.quantum_dim}")
    print(f"- 经典域维度: {config.classical_dim}")
    print(f"- 界面域维度: {config.interface_dim}")
    print(f"- 动态注意力调节参数 β: {config.beta}")
    print(f"- 熵与经典知识调节步长 γ: {config.gamma}")
    print(f"- 维度自适应学习率 η: {config.eta}")
    print(f"- 界面域转换优化因子 λ: {config.lambda_factor}")
    
    # 初始化QCDA基础模型
    print("\n初始化QCDA基础模型...")
    model = QCDAModel(config)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 初始化用于序列分类的QCDA模型
    print("\n初始化用于序列分类的QCDA模型...")
    classification_model = QCDAForSequenceClassification(config)
    
    # 准备输入数据
    print("\n准备输入数据...")
    input_ids = torch.randint(0, config.vocab_size, (2, 128))
    attention_mask = torch.ones_like(input_ids)
    
    # 运行基础模型
    print("\n运行基础模型进行特征提取...")
    with torch.no_grad():
        base_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # 运行分类模型
    print("\n运行分类模型进行情感分类...")
    with torch.no_grad():
        classification_outputs = classification_model(input_ids=input_ids, attention_mask=attention_mask)
    
    # 打印输出形状
    print(f"\n基础模型输出形状: {base_outputs['last_hidden_state'].shape}")
    print(f"分类模型输出形状: {classification_outputs['logits'].shape}")
    
    # 打印分类预测结果
    logits = classification_outputs["logits"]
    predictions = torch.argmax(logits, dim=-1)
    print(f"分类预测结果: {predictions}")
    
    print("\n量子-经典动态注意力模型示例运行完成!")


if __name__ == "__main__":
    main() 