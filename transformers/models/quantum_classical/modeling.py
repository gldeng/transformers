import torch
import math
from .config import QuantumClassicalConfig

class BertEmbeddings(torch.nn.Module):
    """Embeddings for BERT-like models"""
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        # Register buffer for position ids
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
            
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BertSelfAttention(torch.nn.Module):
    """Self-attention layer for BERT"""
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        # Take the dot product between "query" and "key" to get attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
            
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_probs

class GRAM(torch.nn.Module):
    """
    General Recursive Adaptive Module (GRAM)
    量子-经典二元论框架下的递归自适应模块，替代原始Transformer的FFN
    """
    def __init__(self, config):
        super().__init__()
        self.gamma = config.gram_gamma
        
        # 经典信息压缩算子 C(x)
        self.compress = torch.nn.Linear(config.hidden_size, config.hidden_size)
        
        # 量子域特征扰动算子实现
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=config.hidden_size, 
            num_heads=config.gram_heads, 
            dropout=config.hidden_dropout_prob
        )
        
        # 经典信息重构算子 C^-1(z)
        self.expand = torch.nn.Linear(config.hidden_size, config.hidden_size)
        
        # 规范化层
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, x):
        # 经典信息压缩
        z = self.compress(x)
        
        # 量子域特征扰动
        z, _ = self.attn(z, z, z)
        
        # 经典信息重构
        out = self.expand(z)
        
        # 残差连接与规范化
        out = self.layer_norm(out + x)
        
        return out

class QuantumClassicalLayer(torch.nn.Module):
    """Single layer of the Quantum Classical model"""
    def __init__(self, config):
        super().__init__()
        self.attention = BertSelfAttention(config)
        self.gram = GRAM(config)
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        attention_output, _ = self.attention(hidden_states, attention_mask)
        
        # GRAM processing
        gram_output = self.gram(attention_output)
        
        return gram_output

class RecursiveEncoder(torch.nn.Module):
    """Recursive encoder for quantum-classical processing"""
    def __init__(self, config):
        super().__init__()
        self.num_steps = config.num_recursive_steps
        self.layer = QuantumClassicalLayer(config)
        
    def forward(self, hidden_states, attention_mask=None):
        for _ in range(self.num_steps):
            hidden_states = self.layer(hidden_states, attention_mask)
        return hidden_states

class QuantumClassicalModel(torch.nn.Module):
    """Base model for quantum-classical processing"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = RecursiveEncoder(config)
        
    def get_extended_attention_mask(self, attention_mask):
        # Create extended attention mask for transformer
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        output_hidden_states=False,
        return_dict=True,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify input_ids")
            
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=input_ids.device)
            
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
            
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
        )
        
        sequence_output = encoder_outputs
        
        if not return_dict:
            return (sequence_output,)
            
        return {"last_hidden_state": sequence_output}

class QuantumClassicalForQuestionAnswering(torch.nn.Module):
    """Model for question answering tasks"""
    def __init__(self, config):
        super().__init__()
        self.quantum_classical = QuantumClassicalModel(config)
        self.qa_outputs = torch.nn.Linear(config.hidden_size, 2)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        start_positions=None,
        end_positions=None,
        output_hidden_states=False,
        return_dict=True,
    ):
        outputs = self.quantum_classical(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        sequence_output = outputs[0]
        
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
                
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
            
        return {
            "loss": total_loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
        } 