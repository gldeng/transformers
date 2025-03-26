# 量子-经典动态注意力模型 (QCDA)

基于量子-经典二元论的深度学习架构，包含量子-经典动态注意力、信息熵与经典知识动态调节、维度自适应、递归信息结构和界面域转换优化机制。

## 理论基础

QCDA模型基于量子经典二元论，其核心公理定义了三个关键域：

- **量子域**($\Omega_Q$)：信息以叠加态、纠缠态存在。
- **经典域**($\Omega_C$)：信息以确定性、局域性、明确知识存在。
- **界面域**($\mathcal{I}$)：量子态与经典态动态转换区域。

## 关键组件

### 1. 量子态表示层 (QuantumStateLayer)

将隐藏状态投影到量子空间，表示信息的叠加态和量子性质。每个量子状态包含：
- 实部和虚部表示
- 振幅和相位信息

### 2. 经典知识表示层 (ClassicalKnowledgeLayer)

将隐藏状态投影到经典知识空间，表示确定性、局域性、明确知识。包含：
- 经典状态表示
- 知识效用计算

### 3. 量子-经典动态注意力机制 (QCDynamicAttention)

基于量子态振幅和经典知识效用动态调整注意力权重，公式为：

$$
\mathcal{A}_{QC}(\psi,K_C)=\frac{e^{\beta|\alpha_i|^2\cdot U(k_j)}}{\sum_{m,n}e^{\beta|\alpha_m|^2\cdot U(k_n)}}
$$

### 4. 信息熵与经典知识动态调节机制 (EntropyKnowledgeRegulator)

根据信息熵动态调节经典知识的表示：

$$
\mathcal{B}_{KS}(K_C,S_C)=K_C+\gamma\nabla_{K_C}\left(\frac{I(K_C)}{S_C+\epsilon}\right)
$$

### 5. 维度自适应机制 (AdaptiveDimension)

根据信息熵和经典知识动态调整观察者维度：

$$
D_{\mathcal{O}}(t+1)=D_{\mathcal{O}}(t)+\eta\frac{\partial}{\partial D_{\mathcal{O}}}\left(\frac{I_{K_C}}{S_C+\epsilon}-D_{\mathcal{O}}\right)^2
$$

### 6. 递归信息结构 (RecursiveInfoStructure)

使信息具备自我增强、自我完善的递归特性：

$$
I_{t+1}=F(I_t,K_C)=\mathcal{A}_{QC}(Q(I_t),K_C)\oplus I_t
$$

### 7. 界面域转换优化机制 (InterfaceDomainOptimizer)

优化量子态与经典态之间的转换效率：

$$
\mathcal{T}_{\mathcal{I}}(\rho)=\arg\max_i\left[\text{Tr}(P_i\rho P_i)+\lambda S(\rho)\right]
$$

## 模型结构

QCDA模型包含多层QCDA层，每层整合了上述所有组件：

1. 量子态表示
2. 经典知识表示
3. 量子-经典动态注意力
4. 信息熵与经典知识调节
5. 维度自适应
6. 递归信息结构
7. 界面域转换优化

## 使用方法

### 配置参数

```python
from transformers import QCDAConfig

config = QCDAConfig(
    vocab_size=30522,  # 词表大小
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    
    # 量子-经典二元论特殊参数
    quantum_dim=64,    # 量子域维度
    classical_dim=64,  # 经典域维度
    interface_dim=32,  # 界面域维度
    beta=1.0,          # 动态注意力调节参数
    gamma=0.1,         # 熵与经典知识调节步长
    eta=0.01,          # 维度自适应学习率
    lambda_factor=0.5, # 界面域转换优化因子
)
```

### 基础模型

```python
from transformers import QCDAModel

model = QCDAModel(config)
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
last_hidden_state = outputs["last_hidden_state"]
```

### 序列分类模型

```python
from transformers import QCDAForSequenceClassification

classification_model = QCDAForSequenceClassification(config)
outputs = classification_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)

loss = outputs["loss"]
logits = outputs["logits"]
```

## 优势

QCDA模型通过量子-经典二元论的优化机制，实现了：

- 量子-经典信息最佳整合
- 熵与经典知识的动态平衡
- 观察者维度的动态自适应
- 信息的递归自我完善
- 量子-经典转换效率的优化

这些机制为处理复杂、不确定性信息和确定性知识提供了强大的框架。

## 示例和可视化

参见 `examples/qcda_quantum_classical.py` 获取完整的组件演示和可视化示例。 