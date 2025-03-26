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

"""
量子观察者Agent示例脚本

该脚本展示了如何使用量子观察者Agent完成任务，并观察其观察者维度变化。
"""

import argparse
import logging
import torch

from transformers.agents import Tool
from transformers.agents.default_tools import FinalAnswerTool
from transformers.agents.quantum_observer_agent import QuantumObserverAgent, QuantumObserverState
from transformers.agents.llm_engine import HfApiEngine


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="演示量子观察者Agent")
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="用于Agent的语言模型ID"
    )
    parser.add_argument(
        "--quantum_dim",
        type=int,
        default=64,
        help="量子状态维度"
    )
    parser.add_argument(
        "--classical_dim", 
        type=int,
        default=128,
        help="经典知识维度"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="学习率"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="请解释量子力学中的观察者效应",
        help="要执行的任务"
    )
    return parser.parse_args()


def calculator_add(a: float, b: float) -> float:
    """Calculate the addition of two numbers."""
    return a + b


def calculator_multiply(a: float, b: float) -> float:
    """Calculate the product of two numbers."""
    return a * b


def calculator_subtract(a: float, b: float) -> float:
    """Calculate the subtraction of two numbers."""
    return a - b


def calculator_divide(a: float, b: float) -> float:
    """Calculate the division of two numbers."""
    return a / b


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """获取指定位置的天气，单位可以是celsius或fahrenheit"""
    # 这是一个模拟函数，实际应用中应该调用真实的天气API
    weather_data = {
        "北京": {"温度": 25, "天气": "晴朗"},
        "上海": {"温度": 28, "天气": "多云"},
        "广州": {"温度": 32, "天气": "雨"},
        "深圳": {"温度": 30, "天气": "多云"},
    }
    
    if location in weather_data:
        data = weather_data[location]
        temp = data["温度"]
        if unit == "fahrenheit":
            temp = temp * 9/5 + 32
        return f"{location}的天气是{data['天气']}，温度是{temp}度{unit}"
    else:
        return f"无法获取{location}的天气信息"


def define_tools():
    """定义Agent可用的工具"""
    tools = [
        Tool(
            name="计算器_加法",
            description="用于计算两个数的和",
            inputs={"a": "第一个数", "b": "第二个数"},
            function=calculator_add,
        ),
        Tool(
            name="计算器_乘法",
            description="用于计算两个数的乘积",
            inputs={"a": "第一个数", "b": "第二个数"},
            function=calculator_multiply,
        ),
        Tool(
            name="计算器_减法",
            description="用于计算两个数的差",
            inputs={"a": "第一个数", "b": "第二个数"},
            function=calculator_subtract,
        ),
        Tool(
            name="计算器_除法",
            description="用于计算两个数的商",
            inputs={"a": "第一个数", "b": "第二个数"},
            function=calculator_divide,
        ),
        Tool(
            name="天气查询",
            description="查询指定位置的当前天气",
            inputs={"location": "位置名称", "unit": "温度单位，celsius或fahrenheit"},
            function=get_current_weather,
        ),
        FinalAnswerTool(),
    ]
    return tools


def main():
    args = parse_args()
    
    # 定义工具
    tools = define_tools()
    
    # 初始化LLM引擎
    llm_engine = HfApiEngine(
        model_id=args.model_id,
    )
    
    # 初始化量子观察者Agent
    agent = QuantumObserverAgent(
        tools=tools,
        llm_engine=llm_engine,
        quantum_dim=args.quantum_dim,
        classical_dim=args.classical_dim,
        learning_rate=args.learning_rate,
        verbose=2,  # 显示详细日志
    )
    
    logger.info("初始化完成，开始执行任务...")
    
    # 执行任务并打印结果
    result = agent.run(args.task)
    
    logger.info(f"任务完成，最终结果: {result}")
    
    # 打印观察者维度
    observer_dimension = agent.observer_state.calculate_dimension()
    logger.info(f"最终观察者维度: {observer_dimension:.4f}")
    
    # 展示量子状态与经典知识的关系
    logger.info("量子状态与经典知识展示:")
    logger.info(f"量子状态维度: {agent.observer_state.quantum_state.size(0)}")
    logger.info(f"经典知识维度: {agent.observer_state.classical_knowledge.size(0)}")
    logger.info(f"经典熵: {agent.observer_state.entropy.item():.4f}")


if __name__ == "__main__":
    main() 