#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from ..utils import logging
from .agent_types import AgentType
from .agents import Agent, ReactAgent


logger = logging.get_logger(__name__)


class QuantumObserverState:
    """
    量子观察者状态类，用于表示量子-经典二元论框架下的观察者状态。
    
    属性:
        quantum_state: 表示量子状态的张量
        classical_knowledge: 表示经典知识库的张量
        entropy: 经典熵
    """
    
    def __init__(
        self,
        quantum_dim: int = 64,
        classical_dim: int = 128,
        device: Optional[torch.device] = None,
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化量子状态为均匀叠加态
        self.quantum_state = torch.ones(quantum_dim, device=self.device) / math.sqrt(quantum_dim)
        
        # 初始化经典知识库为零向量
        self.classical_knowledge = torch.zeros(classical_dim, device=self.device)
        
        # 初始化经典熵
        self.entropy = torch.tensor(1.0, device=self.device)
        
    def classicalize(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """
        经典化算符 - 将量子状态转换为经典知识
        
        Args:
            quantum_state: 量子状态
            
        Returns:
            经典知识表示
        """
        # 计算概率分布
        probs = quantum_state.abs().pow(2)
        
        # 取最大概率对应的状态（经典化）
        max_idx = torch.argmax(probs)
        classical_state = F.one_hot(max_idx, num_classes=len(quantum_state)).float()
        
        return classical_state
    
    def quantize(self, classical_knowledge: torch.Tensor) -> torch.Tensor:
        """
        量子化算符 - 将经典知识转换为量子可能性
        
        Args:
            classical_knowledge: 经典知识
            
        Returns:
            量子态表示
        """
        # 使用softmax将经典知识转换为概率分布
        probs = F.softmax(classical_knowledge, dim=0)
        
        # 计算量子态振幅（概率的平方根）
        amplitudes = torch.sqrt(probs)
        
        return amplitudes
    
    def calculate_dimension(self) -> float:
        """计算观察者维度，基于经典知识与经典熵的比值"""
        # 计算经典知识的信息量
        info = torch.norm(self.classical_knowledge, p=2)
        
        # 计算观察者维度
        epsilon = 1e-6  # 防止除零
        dimension = info / (self.entropy + epsilon)
        
        return dimension.item()
    
    def update_entropy(self, feedback: torch.Tensor) -> None:
        """更新经典熵"""
        # 基于反馈更新熵
        # 熵减小表示知识更确定，熵增大表示不确定性增加
        self.entropy = 0.9 * self.entropy + 0.1 * torch.norm(feedback, p=2)


class QuantumObserverAgent(ReactAgent):
    """
    量子观察者Agent类，基于量子经典二元论框架实现。
    
    该Agent通过感知(Perception)、决策(Decision)和学习(Learning)三个模块实现观察者模型。
    """
    
    def __init__(
        self,
        tools: List,
        llm_engine: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
        quantum_dim: int = 64,
        classical_dim: int = 128,
        learning_rate: float = 0.01,
        **kwargs,
    ):
        super().__init__(tools, llm_engine, system_prompt, **kwargs)
        
        # 初始化观察者状态
        self.observer_state = QuantumObserverState(
            quantum_dim=quantum_dim,
            classical_dim=classical_dim,
        )
        
        # 学习率
        self.learning_rate = learning_rate
        
        # 内部网络
        self.perception_network = nn.Sequential(
            nn.Linear(quantum_dim, 256),
            nn.ReLU(),
            nn.Linear(256, classical_dim),
        ).to(self.observer_state.device)
        
        self.decision_network = nn.Sequential(
            nn.Linear(classical_dim, 256),
            nn.ReLU(),
            nn.Linear(256, len(tools)),
        ).to(self.observer_state.device)
        
        self.learning_network = nn.Sequential(
            nn.Linear(classical_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, classical_dim),
        ).to(self.observer_state.device)
        
    def perception(self, environment_input: Union[str, Dict, List, torch.Tensor]) -> torch.Tensor:
        """
        感知模块 - 将环境输入经典化
        
        Args:
            environment_input: 环境输入
            
        Returns:
            经典化后的知识表示
        """
        # 将输入转换为量子态表示
        if isinstance(environment_input, torch.Tensor):
            quantum_state = environment_input
        else:
            # 如果输入不是张量，先通过LLM处理并编码为向量
            processed_input = self.llm([{"role": "user", "content": str(environment_input)}])
            # 假设LLM的输出可以转换为量子状态向量
            # 这里简化处理，实际应用中需要更复杂的编码机制
            quantum_state = self.observer_state.quantize(
                torch.randn(self.observer_state.quantum_state.size(0), device=self.observer_state.device)
            )
        
        # 更新量子状态
        self.observer_state.quantum_state = quantum_state
        
        # 使用感知网络处理量子状态
        perception_output = self.perception_network(quantum_state)
        
        # 执行经典化
        classical_knowledge = self.observer_state.classicalize(perception_output)
        
        # 更新经典知识库
        self.observer_state.classical_knowledge = classical_knowledge
        
        return classical_knowledge
    
    def decision(self, classical_knowledge: torch.Tensor) -> int:
        """
        决策模块 - 基于经典知识做出决策
        
        Args:
            classical_knowledge: 经典知识表示
            
        Returns:
            决策结果（工具索引）
        """
        # 使用决策网络处理经典知识
        logits = self.decision_network(classical_knowledge)
        
        # 获取决策结果
        action_idx = torch.argmax(logits).item()
        
        return action_idx
    
    def learning(self, classical_knowledge: torch.Tensor, feedback: torch.Tensor) -> torch.Tensor:
        """
        学习模块 - 基于反馈更新经典知识库
        
        Args:
            classical_knowledge: 当前经典知识
            feedback: 环境反馈
            
        Returns:
            更新后的经典知识
        """
        # 合并当前知识和反馈
        combined = torch.cat([classical_knowledge, feedback], dim=0)
        
        # 使用学习网络更新知识
        updated_knowledge = self.learning_network(combined.unsqueeze(0)).squeeze(0)
        
        # 使用梯度下降更新经典知识
        updated_knowledge = classical_knowledge + self.learning_rate * (updated_knowledge - classical_knowledge)
        
        # 更新观察者状态中的经典知识库
        self.observer_state.classical_knowledge = updated_knowledge
        
        # 更新经典熵
        self.observer_state.update_entropy(feedback)
        
        return updated_knowledge
    
    def run(self, task: str, stream: bool = False, reset: bool = True, **kwargs):
        """
        运行Agent以完成任务
        
        Args:
            task: 任务描述
            stream: 是否流式输出
            reset: 是否重置状态
            **kwargs: 其他参数
            
        Returns:
            Agent响应
        """
        # 将任务描述转换为量子状态
        task_encoding = torch.randn(
            self.observer_state.quantum_state.size(0), 
            device=self.observer_state.device
        )
        task_quantum_state = self.observer_state.quantize(task_encoding)
        
        # 感知：从量子态到经典知识
        classical_knowledge = self.perception(task_quantum_state)
        
        # 决策：根据经典知识选择工具
        action_idx = self.decision(classical_knowledge)
        selected_tool = list(self.toolbox.tools.values())[action_idx]
        
        # 执行ReactAgent的标准运行流程
        result = super().run(task, stream, reset, **kwargs)
        
        # 学习：从结果中提取反馈并更新知识
        feedback_encoding = torch.randn(
            self.observer_state.classical_knowledge.size(0),
            device=self.observer_state.device
        )  # 实际应用中应该从结果中提取真实反馈
        
        updated_knowledge = self.learning(classical_knowledge, feedback_encoding)
        
        # 计算并记录观察者维度
        observer_dimension = self.observer_state.calculate_dimension()
        logger.info(f"Observer dimension: {observer_dimension:.4f}")
        
        return result 