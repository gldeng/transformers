from .configuration_quantum_classical import QuantumClassicalConfig
from .modeling_quantum_classical import (
    QuantumClassicalModel,
    QuantumClassicalForQuestionAnswering,
)
from .tokenizer import BertTokenizerStandalone

# Import our model mapping patch
from .model_mapping import MODEL_NAMES_MAPPING

from transformers import AutoConfig, AutoModel, AutoModelForQuestionAnswering

# Register the model
AutoConfig.register("quantum_classical", QuantumClassicalConfig)
AutoModel.register(QuantumClassicalConfig, QuantumClassicalModel)
AutoModelForQuestionAnswering.register(QuantumClassicalConfig, QuantumClassicalForQuestionAnswering)

__all__ = [
    "QuantumClassicalConfig",
    "QuantumClassicalModel",
    "QuantumClassicalForQuestionAnswering",
    "BertTokenizerStandalone",
] 