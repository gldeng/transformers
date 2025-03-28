#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone implementation of Quantum Classical model for question answering
Avoids import conflicts with transformers auto classes
"""

import os
import sys
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
import math

class BertTokenizerStandalone:
    """Simple standalone implementation of BertTokenizer basics"""
    def __init__(self, vocab_file=None):
        # Use a small basic vocabulary for demonstration
        self.vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3}
        self.vocab.update({f"token{i}": i+4 for i in range(30000)})  # Dummy vocab
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
    def tokenize(self, text):
        # Very simplified tokenization - just split by space
        return text.lower().split()
    
    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab["[UNK]"]) for token in tokens]
    
    def encode(self, text, text_pair=None, max_length=None, truncation=False, padding=False):
        tokens = ["[CLS]"] + self.tokenize(text)
        if text_pair:
            tokens += ["[SEP]"] + self.tokenize(text_pair)
        else:
            tokens += ["[SEP]"]
            
        if truncation and max_length and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + ["[SEP]"]
            
        token_ids = self.convert_tokens_to_ids(tokens)
        
        if padding and max_length:
            padding_length = max_length - len(token_ids)
            token_ids = token_ids + [0] * padding_length
            
        return token_ids
    
    def __call__(self, text, text_pair=None, max_length=None, truncation=False, padding=False, return_tensors=None):
        if isinstance(text, str):
            token_ids = self.encode(text, text_pair, max_length, truncation, padding)
            attention_mask = [1] * len(token_ids)
            if padding and max_length:
                attention_mask = [1] * (len(token_ids) - attention_mask.count(0)) + [0] * attention_mask.count(0)
                
            token_type_ids = [0] * len(token_ids)
            if text_pair:
                sep_pos = token_ids.index(self.vocab["[SEP]"])
                token_type_ids[sep_pos+1:] = [1] * (len(token_ids) - sep_pos - 1)
                
            result = {
                "input_ids": token_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
            
            if return_tensors == "pt":
                result = {k: torch.tensor([v]) for k, v in result.items()}
                
            return result
        else:
            # For batch processing
            results = []
            for t, tp in zip(text, text_pair if text_pair else [None] * len(text)):
                results.append(self(t, tp, max_length, truncation, padding))
                
            # Combine results
            batch_result = {
                "input_ids": [],
                "attention_mask": [],
                "token_type_ids": []
            }
            
            for r in results:
                batch_result["input_ids"].append(r["input_ids"])
                batch_result["attention_mask"].append(r["attention_mask"])
                batch_result["token_type_ids"].append(r["token_type_ids"])
                
            if return_tensors == "pt":
                batch_result = {k: torch.tensor(v) for k, v in batch_result.items()}
                
            return batch_result


# Standalone implementation of the Quantum Classical model components

class QuantumClassicalConfig:
    """Simplified configuration for Quantum Classical model"""
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
    ):
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
        
        # Dropout
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        # 经典信息压缩
        compressed = torch.relu(self.compress(x))
        
        # 量子特征扰动通过自注意力实现
        disturbed, _ = self.attn(compressed.transpose(0, 1), compressed.transpose(0, 1), compressed.transpose(0, 1))
        disturbed = torch.tanh(disturbed.transpose(0, 1))  # 量子域特征扰动
        
        # 经典信息重构
        expanded = self.expand(disturbed)
        
        # 加入残差连接并规范化
        return self.layer_norm(x + self.gamma * self.dropout(expanded))


class QuantumClassicalLayer(torch.nn.Module):
    """
    Quantum Classical transformer layer with GRAM module
    """
    def __init__(self, config):
        super().__init__()
        self.attention = BertSelfAttention(config)
        self.attn_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        # GRAM module
        self.gram = GRAM(config)
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        attention_output, _ = self.attention(hidden_states, attention_mask)
        
        # Add & Norm
        attention_output = self.attn_norm(hidden_states + self.attn_dropout(attention_output))
        
        # GRAM module
        layer_output = self.gram(attention_output)
        
        return layer_output


class RecursiveEncoder(torch.nn.Module):
    """
    Encoder with recursive optimization
    """
    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.ModuleList([QuantumClassicalLayer(config) for _ in range(config.num_hidden_layers)])
        self.num_recursive_steps = config.num_recursive_steps
        
    def forward(self, hidden_states, attention_mask=None):
        all_hidden_states = []
        
        for layer in self.layers:
            # Recursive processing
            for _ in range(self.num_recursive_steps):
                hidden_states = layer(hidden_states, attention_mask)
                
            all_hidden_states.append(hidden_states)
            
        return hidden_states, all_hidden_states


class QuantumClassicalModel(torch.nn.Module):
    """
    Simplified Quantum Classical model main implementation
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embeddings = BertEmbeddings(config)
        self.encoder = RecursiveEncoder(config)
        self.pooler = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = torch.nn.Tanh()
        
    def get_extended_attention_mask(self, attention_mask):
        # Create extended attention mask for transformer
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
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
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
            
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        
        sequence_output, all_hidden_states = self.encoder(embedding_output, extended_attention_mask)
        
        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        
        if return_dict:
            return {
                "last_hidden_state": sequence_output,
                "pooler_output": pooled_output,
                "hidden_states": all_hidden_states if output_hidden_states else None
            }
        else:
            return (sequence_output, pooled_output) + (all_hidden_states,) if output_hidden_states else ()


class QuantumClassicalForQuestionAnswering(torch.nn.Module):
    """
    Quantum Classical model for question answering
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = 2
        
        self.quantum_classical = QuantumClassicalModel(config)
        self.qa_outputs = torch.nn.Linear(config.hidden_size, self.num_labels)
        
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
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        sequence_output = outputs["last_hidden_state"]
        
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
                
            # Sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            
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
            "hidden_states": outputs["hidden_states"],
        }


class SimpleQADataset(Dataset):
    """Simple QA dataset for SQuAD format"""
    def __init__(self, dataset, tokenizer, max_length=384):
        self.examples = []
        self.max_length = max_length
        
        print("Preparing dataset...")
        for example in tqdm(dataset, desc="Processing examples"):
            question = example["question"]
            context = example["context"]
            
            # Tokenize
            encoding = tokenizer(
                question,
                context,
                max_length=max_length,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt"
            )
            
            # Get answer start/end position (dummy for simplicity)
            start_position = 0
            end_position = 5
            
            if "answers" in example and len(example["answers"]["text"]) > 0:
                start_position = min(example["answers"]["answer_start"][0], max_length-10)
                end_position = min(start_position + len(example["answers"]["text"][0]), max_length-1)
            
            self.examples.append({
                "input_ids": encoding["input_ids"][0],
                "attention_mask": encoding["attention_mask"][0],
                "token_type_ids": encoding["token_type_ids"][0] if "token_type_ids" in encoding else None,
                "start_positions": torch.tensor(start_position),
                "end_positions": torch.tensor(end_position)
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    """Main function"""
    print("=" * 50)
    print("Standalone Quantum Classical Model for QA")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer = BertTokenizerStandalone()
        
        # Load SQuAD dataset
        print("Loading SQuAD dataset...")
        dataset = load_dataset("squad", split="train[:100]")
        print(f"Loaded SQuAD dataset with {len(dataset)} examples")
        
        # Prepare dataset
        qa_dataset = SimpleQADataset(dataset, tokenizer, max_length=128)  # Smaller context for speed
        
        # Split into train/eval
        train_size = int(0.8 * len(qa_dataset))
        eval_size = len(qa_dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(qa_dataset, [train_size, eval_size])
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=4)
        
        print(f"Training dataloader size: {len(train_dataloader)}")
        print(f"Evaluation dataloader size: {len(eval_dataloader)}")
        
        # Initialize Quantum Classical model with smaller dimensions for faster execution
        config = QuantumClassicalConfig(
            vocab_size=30522,
            hidden_size=128,        # Smaller hidden size
            num_hidden_layers=2,    # Fewer layers
            num_attention_heads=4,  # Fewer heads
            intermediate_size=256,  # Smaller intermediate size
            gram_gamma=0.6,         # Recursive step size factor
            num_recursive_steps=2,  # Number of recursive steps
            gram_heads=2,           # GRAM attention heads
            max_position_embeddings=128  # Match max_length
        )
        
        # Create model
        model = QuantumClassicalForQuestionAnswering(config)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Move to device
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=5e-5)
        
        # Simple training loop
        print("Starting training...")
        model.train()
        for epoch in range(1):
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items() if k != "token_type_ids" or v is not None}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                
                if step % 5 == 0:
                    print(f"Batch {step}: loss = {loss.item():.4f}")
                    
                if step >= 10:  # Just train on a few batches for demo
                    break
            
            avg_loss = total_loss / (step + 1)
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Save model
        model_save_path = "quantum_classical_qa_standalone.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")
        
        # Print recursive optimization effect analysis
        print("\nRecursive Adaptive Optimization Effect Analysis:")
        print(f"Recursive step size factor (gamma): {config.gram_gamma}")
        print(f"Number of recursive steps: {config.num_recursive_steps}")
        print("Recursive adaptive optimization allows the model to process the same information multiple times,")
        print("theoretically improving the accuracy of complex QA tasks, especially for multi-hop reasoning.")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()