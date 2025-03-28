from transformers import PretrainedConfig

class QuantumClassicalConfig(PretrainedConfig):
    model_type = "quantum_classical"
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        is_decoder=False,
        quantum_dim=128,
        classical_dim=128,
        interface_dim=64,
        gram_gamma=0.5,
        num_recursive_steps=3,
        gram_heads=8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.is_decoder = is_decoder
        self.quantum_dim = quantum_dim
        self.classical_dim = classical_dim
        self.interface_dim = interface_dim
        self.gram_gamma = gram_gamma
        self.num_recursive_steps = num_recursive_steps
        self.gram_heads = gram_heads

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def to_dict(self):
        output = super().to_dict()
        output["model_type"] = self.model_type
        return output 