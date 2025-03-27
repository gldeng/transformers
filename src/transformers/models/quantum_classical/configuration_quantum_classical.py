from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..bert.configuration_bert import BertConfig


logger = logging.get_logger(__name__)

QUANTUM_CLASSICAL_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class QuantumClassicalConfig(BertConfig):
    """
    QuantumClassical配置类，基于BertConfig的扩展，添加了GRAM特有的参数
    
    这是一个实现了量子经典递归自适应优化理论的Transformer配置
    通过GRAM（通用递归自适应模块）替代传统前馈网络，实现了递归自适应优化能力
    
    Args:
        gram_gamma (`float`, *optional*, defaults to 0.5):
            递归步长因子，用于控制GRAM模块更新的幅度
        num_recursive_steps (`int`, *optional*, defaults to 10):
            递归优化次数，控制模型的递归深度，理论上可以无限递归
        gram_heads (`int`, *optional*, defaults to 8):
            GRAM中的注意力头数
        **kwargs:
            BertConfig的其他参数
    """

    model_type = "quantum_classical"

    def __init__(
        self,
        gram_gamma=0.5,  # 递归步长因子
        num_recursive_steps=10,  # 递归优化次数
        gram_heads=8,  # GRAM中的注意力头数
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gram_gamma = gram_gamma
        self.num_recursive_steps = num_recursive_steps
        self.gram_heads = gram_heads 