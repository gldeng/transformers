#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于量子-经典二元论的量子-经典动态注意力机制
详细示例脚本 - 展示各个关键组件的工作原理
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import matplotlib.font_manager as fm
import platform

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 根据不同操作系统设置中文字体
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
elif system == 'Darwin':  # macOS
    # macOS的中文字体路径
    fonts = ['/System/Library/Fonts/PingFang.ttc',
             '/System/Library/Fonts/STHeiti Light.ttc',
             '/System/Library/Fonts/STHeiti Medium.ttc',
             '/Library/Fonts/Arial Unicode.ttf']
    
    # 检查哪些字体可用
    chinese_font = None
    for font in fonts:
        if os.path.exists(font):
            chinese_font = font
            break
    
    if chinese_font:
        # 添加字体并使用
        font_prop = fm.FontProperties(fname=chinese_font)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        # 尝试使用系统已知的字体名称
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Heiti TC', 'Arial Unicode MS']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'AR PL UMing CN']

plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 定义一个函数来设置绘图字体
def set_plot_text_font(ax, font_prop=None):
    """设置图表中所有文本的字体"""
    for text in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        if font_prop:
            text.set_fontproperties(font_prop)
        elif system == 'Darwin' and chinese_font:  # macOS上使用找到的中文字体
            text.set_fontproperties(fm.FontProperties(fname=chinese_font))

try:
    from src.transformers.models.qcda.configuration_qcda import QCDAConfig
    from src.transformers.models.qcda.modeling_qcda import (
        QCDAModel, QCDAForSequenceClassification,
        QuantumStateLayer, ClassicalKnowledgeLayer, QCDynamicAttention,
        EntropyKnowledgeRegulator, AdaptiveDimension, RecursiveInfoStructure,
        InterfaceDomainOptimizer, entropy
    )
except ImportError:
    print("无法导入QCDA模型，请确保已正确安装或模型文件在正确路径")
    sys.exit(1)


def plot_attention_weights(weights, title="量子-经典动态注意力权重"):
    """绘制注意力权重热图"""
    plt.figure(figsize=(10, 8))
    plt.imshow(weights, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("键位置")
    plt.ylabel("查询位置")
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close()


def plot_quantum_amplitude(amplitude, title="量子振幅分布"):
    """绘制量子振幅分布"""
    plt.figure(figsize=(12, 6))
    plt.plot(amplitude.detach().numpy())
    plt.title(title)
    plt.xlabel("量子状态索引")
    plt.ylabel("振幅")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close()


def plot_entropy_knowledge_regulation(entropy_values, knowledge_values, title="熵与知识调节"):
    """绘制熵与知识调节过程"""
    plt.figure(figsize=(12, 6))
    plt.plot(entropy_values.detach().numpy(), label="信息熵")
    plt.plot(knowledge_values.detach().numpy(), label="知识效用")
    plt.title(title)
    plt.xlabel("迭代次数")
    plt.ylabel("数值")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close()


def visualize_interface_domain(quantum_weights, classical_weights, interface_weights, 
                              title="界面域权重分布"):
    """可视化界面域的权重分布"""
    x = np.arange(len(quantum_weights))
    width = 0.25
    
    plt.figure(figsize=(14, 7))
    plt.bar(x - width, quantum_weights.detach().numpy(), width, label="量子域权重")
    plt.bar(x, classical_weights.detach().numpy(), width, label="经典域权重")
    plt.bar(x + width, interface_weights.detach().numpy(), width, label="界面域权重")
    
    plt.title(title)
    plt.xlabel("特征维度")
    plt.ylabel("权重值")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close()


def demonstrate_quantum_state():
    """演示量子态表示层"""
    print("\n1. 量子态表示层演示")
    
    # 创建配置与层
    config = QCDAConfig(quantum_dim=16, hidden_size=64)
    quantum_layer = QuantumStateLayer(config)
    
    # 创建随机输入
    batch_size, seq_len = 2, 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # 前向传播
    quantum_info = quantum_layer(hidden_states)
    
    # 提取信息
    real_part = quantum_info["real"]
    imag_part = quantum_info["imag"]
    amplitude = quantum_info["amplitude"]
    phase = quantum_info["phase"]
    
    print(f"实部形状: {real_part.shape}")
    print(f"虚部形状: {imag_part.shape}")
    print(f"振幅形状: {amplitude.shape}")
    print(f"相位形状: {phase.shape}")
    
    # 可视化第一个批次的第一个序列位置的量子振幅
    plot_quantum_amplitude(amplitude[0, 0], "量子振幅分布示例")
    
    return quantum_info


def demonstrate_classical_knowledge():
    """演示经典知识表示层"""
    print("\n2. 经典知识表示层演示")
    
    # 创建配置与层
    config = QCDAConfig(classical_dim=16, hidden_size=64)
    classical_layer = ClassicalKnowledgeLayer(config)
    
    # 创建随机输入
    batch_size, seq_len = 2, 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # 前向传播
    classical_info = classical_layer(hidden_states)
    
    # 提取信息
    states = classical_info["states"]
    utility = classical_info["utility"]
    
    print(f"经典状态形状: {states.shape}")
    print(f"效用形状: {utility.shape}")
    print(f"效用总和: {utility.sum(-1)}")  # 应该接近1（softmax）
    
    return classical_info


def demonstrate_qc_attention(quantum_info, classical_info):
    """演示量子-经典动态注意力机制"""
    print("\n3. 量子-经典动态注意力机制演示")
    
    # 创建配置与层
    config = QCDAConfig(
        quantum_dim=16, 
        classical_dim=16, 
        hidden_size=64, 
        num_attention_heads=4,
        beta=2.0  # 增大beta以突出动态注意力效果
    )
    attention = QCDynamicAttention(config)
    
    # 创建随机输入
    batch_size, seq_len = 2, 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # 确保量子信息和经典信息与隐藏状态兼容
    if quantum_info is None or classical_info is None:
        # 创建必要的层
        quantum_layer = QuantumStateLayer(config)
        classical_layer = ClassicalKnowledgeLayer(config)
        
        # 获取信息
        quantum_info = quantum_layer(hidden_states)
        classical_info = classical_layer(hidden_states)
    
    # 前向传播
    attention_output = attention(hidden_states, quantum_info, classical_info)
    
    print(f"注意力输出形状: {attention_output.shape}")
    
    # 为了可视化，我们需要访问内部注意力权重
    # 这在实际模型中不可行，这里我们重新计算一个简化版的注意力权重用于演示
    
    # 简化的注意力计算用于可视化
    q = hidden_states @ torch.randn(config.hidden_size, config.hidden_size)
    k = hidden_states @ torch.randn(config.hidden_size, config.hidden_size)
    qk = torch.matmul(q, k.transpose(-1, -2)) / (config.hidden_size ** 0.5)
    
    # 量子振幅因子（简化）
    amp_factor = quantum_info["amplitude"][0, :, 0] ** 2  # 批次0，特征0
    amp_factor = amp_factor.unsqueeze(-1).expand(-1, seq_len)
    
    # 经典效用因子（简化）
    util_factor = classical_info["utility"][0, :, 0]  # 批次0，特征0
    util_factor = util_factor.unsqueeze(0).expand(seq_len, -1)
    
    # 计算动态因子
    dynamic_factor = torch.exp(config.beta * amp_factor * util_factor)
    
    # 应用动态因子并softmax
    weighted_attn = qk[0] * dynamic_factor
    attn_weights = F.softmax(weighted_attn, dim=-1)
    
    # 可视化
    plot_attention_weights(attn_weights.detach().numpy(), 
                         "量子-经典动态注意力权重示例")
    
    return attention_output


def demonstrate_entropy_knowledge_regulation(classical_info):
    """演示信息熵与经典知识动态调节机制"""
    print("\n4. 信息熵与经典知识动态调节机制演示")
    
    # 创建配置与层
    config = QCDAConfig(
        classical_dim=16, 
        hidden_size=64,
        gamma=0.2  # 增大步长以突出效果
    )
    regulator = EntropyKnowledgeRegulator(config)
    
    # 创建随机输入
    batch_size, seq_len = 2, 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # 确保经典信息与隐藏状态兼容
    if classical_info is None:
        # 创建必要的层
        classical_layer = ClassicalKnowledgeLayer(config)
        
        # 获取信息
        classical_info = classical_layer(hidden_states)
    
    # 存储熵和知识效用的变化
    num_iterations = 10
    entropy_values = torch.zeros(num_iterations)
    knowledge_values = torch.zeros(num_iterations)
    
    current_classical = classical_info
    
    # 多次迭代以观察调节效果
    for i in range(num_iterations):
        # 计算熵
        classical_probs = F.softmax(current_classical["states"][0, 0], dim=-1)
        entropy_values[i] = entropy(classical_probs.unsqueeze(0)).item()
        
        # 记录知识效用均值
        knowledge_values[i] = current_classical["utility"][0, 0].mean().item()
        
        # 调节
        current_classical = regulator(hidden_states, current_classical)
    
    print(f"初始熵: {entropy_values[0]:.4f}, 最终熵: {entropy_values[-1]:.4f}")
    print(f"初始知识效用: {knowledge_values[0]:.4f}, 最终知识效用: {knowledge_values[-1]:.4f}")
    
    # 可视化熵与知识效用的变化
    plot_entropy_knowledge_regulation(entropy_values, knowledge_values,
                                    "熵与知识调节动态过程")
    
    return current_classical


def demonstrate_adaptive_dimension(classical_info):
    """演示维度自适应机制"""
    print("\n5. 维度自适应机制演示")
    
    # 创建配置与层
    config = QCDAConfig(
        classical_dim=16, 
        hidden_size=64,
        eta=0.05  # 增大学习率以突出效果
    )
    adaptive_dim = AdaptiveDimension(config)
    
    # 创建随机输入
    batch_size, seq_len = 2, 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # 确保经典信息与隐藏状态兼容
    if classical_info is None:
        # 创建必要的层
        classical_layer = ClassicalKnowledgeLayer(config)
        
        # 获取信息
        classical_info = classical_layer(hidden_states)
    
    # 前向传播
    adapted_states = adaptive_dim(hidden_states, classical_info)
    
    # 打印初始维度和当前维度
    print(f"初始观察者维度: {config.hidden_size // 2}")
    print(f"适应后观察者维度: {adaptive_dim.observer_dim.item():.4f}")
    print(f"适应后状态形状: {adapted_states.shape}")
    
    return adapted_states


def demonstrate_recursive_structure(quantum_info):
    """演示递归信息结构"""
    print("\n6. 递归信息结构演示")
    
    # 创建配置与层
    config = QCDAConfig(quantum_dim=16, hidden_size=64)
    recursive = RecursiveInfoStructure(config)
    
    # 创建随机输入
    batch_size, seq_len = 2, 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    prev_states = torch.randn(batch_size, seq_len, config.hidden_size) * 0.5  # 不同的初始状态
    
    # 多次递归以观察效果
    num_recursions = 5
    states_norm = torch.zeros(num_recursions)
    
    current_states = hidden_states
    
    for i in range(num_recursions):
        # 记录当前状态的范数
        states_norm[i] = torch.norm(current_states).item()
        
        # 递归
        current_states = recursive(current_states, quantum_info, prev_states)
    
    print(f"递归状态范数变化: {states_norm}")
    print(f"递归后状态形状: {current_states.shape}")
    
    return current_states


def demonstrate_interface_optimizer(quantum_info, classical_info):
    """演示界面域转换优化机制"""
    print("\n7. 界面域转换优化机制演示")
    
    # 创建配置与层
    config = QCDAConfig(
        quantum_dim=16, 
        classical_dim=16,
        interface_dim=12,
        hidden_size=64,
        lambda_factor=0.3  # 设置界面域优化因子
    )
    interface_optimizer = InterfaceDomainOptimizer(config)
    
    # 创建随机输入
    batch_size, seq_len = 2, 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # 确保量子信息和经典信息与隐藏状态兼容
    if quantum_info is None or classical_info is None:
        # 创建必要的层
        quantum_layer = QuantumStateLayer(config)
        classical_layer = ClassicalKnowledgeLayer(config)
        
        # 获取信息
        quantum_info = quantum_layer(hidden_states)
        classical_info = classical_layer(hidden_states)
    
    # 前向传播
    interface_output = interface_optimizer(hidden_states, quantum_info, classical_info)
    
    print(f"界面域输出形状: {interface_output.shape}")
    
    # 获取权重用于可视化
    quantum_combined = torch.cat([quantum_info["real"][0, 0], quantum_info["imag"][0, 0]], dim=-1)
    quantum_weights = interface_optimizer.quantum_projector(quantum_combined)
    classical_weights = interface_optimizer.classical_projector(classical_info["states"][0, 0])
    interface_weights = interface_optimizer.interface_projection(hidden_states[0, 0])
    
    # 可视化界面域权重
    visualize_interface_domain(
        F.softmax(quantum_weights, dim=-1),
        F.softmax(classical_weights, dim=-1),
        F.softmax(interface_weights, dim=-1),
        "量子-经典-界面域权重分布"
    )
    
    return interface_output


def demonstrate_full_model():
    """演示完整的QCDA模型"""
    print("\n8. 完整QCDA模型演示")
    
    # 创建配置
    config = QCDAConfig(
        vocab_size=30522,  # BERT词表大小
        hidden_size=256,
        num_hidden_layers=3,
        num_attention_heads=8,
        intermediate_size=1024,
        
        # 量子-经典二元论特殊参数
        quantum_dim=32,
        classical_dim=32, 
        interface_dim=16,
        beta=1.0,  # 动态注意力调节参数
        gamma=0.1,  # 熵与经典知识调节步长
        eta=0.01,   # 维度自适应学习率
        lambda_factor=0.5,  # 界面域转换优化因子
        
        # 序列分类参数
        num_labels=2
    )
    
    # 初始化模型
    model = QCDAModel(config)
    classification_model = QCDAForSequenceClassification(config)
    
    # 打印模型结构
    print(f"QCDA模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 准备输入数据
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    # 运行模型
    with torch.no_grad():
        base_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        classification_outputs = classification_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
    
    # 打印输出形状
    print(f"基础模型输出形状: {base_outputs['last_hidden_state'].shape}")
    print(f"分类模型输出形状: {classification_outputs['logits'].shape}")
    
    # 打印分类预测结果
    logits = classification_outputs["logits"]
    probabilities = F.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)
    
    print(f"分类预测: {predictions}")
    print(f"分类概率: {probabilities}")
    
    return model, classification_model


def main():
    """主函数，演示所有组件"""
    print("基于量子-经典二元论的量子-经典动态注意力模型演示")
    print("=" * 80)
    
    # 1. 量子态表示层
    quantum_info = demonstrate_quantum_state()
    
    # 2. 经典知识表示层
    classical_info = demonstrate_classical_knowledge()
    
    # 3. 量子-经典动态注意力机制
    demonstrate_qc_attention(quantum_info, classical_info)
    
    # 4. 信息熵与经典知识动态调节机制
    regulated_classical = demonstrate_entropy_knowledge_regulation(classical_info)
    
    # 5. 维度自适应机制
    demonstrate_adaptive_dimension(regulated_classical)
    
    # 6. 递归信息结构
    demonstrate_recursive_structure(quantum_info)
    
    # 7. 界面域转换优化机制
    demonstrate_interface_optimizer(quantum_info, regulated_classical)
    
    # 8. 完整模型
    model, classification_model = demonstrate_full_model()
    
    print("\n演示完成! 所有可视化结果已保存为图像文件。")


if __name__ == "__main__":
    main() 