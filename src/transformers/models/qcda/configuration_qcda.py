"""
量子-经典动态注意力机制（QCDA）配置类
"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

QCDA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class QCDAConfig(PretrainedConfig):
    """
    量子-经典动态注意力（Quantum-Classical Dynamic Attention, QCDA）的配置类
    
    这个配置类包含了量子-经典二元论框架所需的所有参数，包括:
    - 量子域参数
    - 经典域参数 
    - 界面域参数
    - 各种优化机制的参数
    
    Args:
        hidden_size (`int`, *可选*, 默认为 768): 
            隐藏层维度
        num_hidden_layers (`int`, *可选*, 默认为 12):
            Transformer编码器的层数
        num_attention_heads (`int`, *可选*, 默认为 12):
            注意力头的数量
        intermediate_size (`int`, *可选*, 默认为 3072):
            前馈网络层的维度
        hidden_act (`str` 或 `Callable`, *可选*, 默认为 "gelu"):
            激活函数
        hidden_dropout_prob (`float`, *可选*, 默认为 0.1):
            隐藏层dropout概率
        attention_probs_dropout_prob (`float`, *可选*, 默认为 0.1):
            注意力概率的dropout比率
        max_position_embeddings (`int`, *可选*, 默认为 512):
            位置编码的最大长度
        initializer_range (`float`, *可选*, 默认为 0.02):
            初始化参数的范围
        layer_norm_eps (`float`, *可选*, 默认为 1e-12):
            层归一化的epsilon值
        
        # 量子-经典二元论框架的特殊参数
        quantum_dim (`int`, *可选*, 默认为 64):
            量子域的维度
        classical_dim (`int`, *可选*, 默认为 64):
            经典域的维度
        interface_dim (`int`, *可选*, 默认为 32):
            界面域的维度
        beta (`float`, *可选*, 默认为 1.0):
            动态注意力的调节参数
        gamma (`float`, *可选*, 默认为 0.1):
            熵与经典知识调节的步长
        eta (`float`, *可选*, 默认为 0.01):
            维度自适应的学习率
        lambda_factor (`float`, *可选*, 默认为 0.5):
            界面域转换的优化因子
    """
    model_type = "qcda"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        # 量子-经典二元论特殊参数
        quantum_dim=64,
        classical_dim=64,
        interface_dim=32,
        beta=1.0,  # 动态注意力调节参数
        gamma=0.1,  # 熵与经典知识调节步长
        eta=0.01,   # 维度自适应学习率
        lambda_factor=0.5,  # 界面域转换优化因子
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        
        # 量子-经典二元论参数
        self.quantum_dim = quantum_dim
        self.classical_dim = classical_dim
        self.interface_dim = interface_dim
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.lambda_factor = lambda_factor 