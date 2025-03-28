#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified benchmarking for Quantum Classical model
"""

import os
import sys
import time
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to the path so we can import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from the standalone implementation
from examples.standalone_quantum_classical_qa import (
    BertTokenizerStandalone,
    QuantumClassicalConfig,
    QuantumClassicalForQuestionAnswering
)

# Import the dataset class
from examples.quantum_classical_natural_questions import NaturalQuestionsDataset

def print_table(headers, rows):
    """Print a formatted table of results"""
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print headers
    header = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header)
    print("-" * len(header))
    
    # Print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)))

def evaluate_model(model, eval_dataloader, device, tokenizer, num_examples=5):
    """Evaluate the model on the evaluation dataset with enhanced metrics"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # For F1 and EM calculation
    all_predictions = []
    all_labels = []
    
    # For visualization
    examples_to_show = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items() if k != "token_type_ids" or v is not None}
            
            # Get the original input_ids for decoding
            input_ids = batch["input_ids"]
            
            # Get the true start and end positions
            start_positions = batch["start_positions"]
            end_positions = batch["end_positions"]
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs["loss"]
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]
            
            # Get predicted start and end positions
            start_pred = torch.argmax(start_logits, dim=1)
            end_pred = torch.argmax(end_logits, dim=1)
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions and labels for metrics calculation
            for i in range(len(input_ids)):
                # Get the predicted answer spans
                pred_start = start_pred[i].item()
                pred_end = end_pred[i].item()
                true_start = start_positions[i].item() 
                true_end = end_positions[i].item()
                
                # Handle case where end < start
                if pred_end < pred_start:
                    pred_end = pred_start
                
                # Get the tokens for both prediction and true answer
                pred_tokens = input_ids[i][pred_start:pred_end+1]
                true_tokens = input_ids[i][true_start:true_end+1]
                
                # Convert to text (simplified for demonstration)
                pred_text = tokenizer.decode(pred_tokens)
                true_text = tokenizer.decode(true_tokens)
                
                all_predictions.append(pred_text)
                all_labels.append(true_text)
                
                # Store examples to show (only a few)
                if batch_idx < num_examples and i == 0:
                    # Get the full input sequence
                    full_text = tokenizer.decode(input_ids[i])
                    
                    # Extract question and context based on the [SEP] token position
                    sep_pos = input_ids[i].tolist().index(tokenizer.vocab["[SEP]"])
                    question = tokenizer.decode(input_ids[i][1:sep_pos])  # Skip [CLS]
                    context = tokenizer.decode(input_ids[i][sep_pos+1:])  # Skip [SEP]
                    
                    examples_to_show.append({
                        "question": question.strip(),
                        "context": context.strip(),
                        "true_answer": true_text.strip(),
                        "pred_answer": pred_text.strip(),
                        "correct": pred_text.strip() == true_text.strip()
                    })
    
    # Calculate metrics
    exact_match = sum(pred == label for pred, label in zip(all_predictions, all_labels)) / len(all_predictions)
    
    # Calculate F1 score (simplified version)
    f1_scores = []
    for pred, label in zip(all_predictions, all_labels):
        # Convert to sets of words for overlap calculation
        pred_words = set(pred.strip().lower().split())
        label_words = set(label.strip().lower().split())
        
        if len(pred_words) == 0 and len(label_words) == 0:
            f1_scores.append(1.0)  # Both empty means perfect match
            continue
        
        if len(pred_words) == 0 or len(label_words) == 0:
            f1_scores.append(0.0)  # One empty means no match
            continue
        
        # Calculate precision, recall, F1
        common_words = pred_words.intersection(label_words)
        precision = len(common_words) / len(pred_words) if len(pred_words) > 0 else 0
        recall = len(common_words) / len(label_words) if len(label_words) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_loss = total_loss / num_batches
    
    # Print example predictions
    print("\n===== Example Predictions =====")
    for idx, example in enumerate(examples_to_show):
        print(f"\nExample {idx+1}:")
        print(f"Question: {example['question']}")
        print(f"Context: {example['context'][:100]}...")  # Show beginning of context
        print(f"True answer: {example['true_answer']}")
        print(f"Predicted: {example['pred_answer']}")
        print(f"Correct: {'✓' if example['correct'] else '✗'}")
    
    return {
        "loss": avg_loss,
        "exact_match": exact_match,
        "f1": avg_f1,
        "examples": examples_to_show
    }

def benchmark_quantum_classical_variants():
    """Benchmark different configurations of Quantum Classical model"""
    # Create different configurations to compare
    configs = {
        "Small": QuantumClassicalConfig(
            vocab_size=30522,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            gram_gamma=0.5,
            num_recursive_steps=2,
            gram_heads=2,
            max_position_embeddings=384
        ),
        "Medium": QuantumClassicalConfig(
            vocab_size=30522,
            hidden_size=256,
            num_hidden_layers=3,
            num_attention_heads=8,
            intermediate_size=512,
            gram_gamma=0.7,
            num_recursive_steps=3,
            gram_heads=4,
            max_position_embeddings=384
        ),
        "Large": QuantumClassicalConfig(
            vocab_size=30522,
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=1024,
            gram_gamma=0.8,
            num_recursive_steps=4,
            gram_heads=6,
            max_position_embeddings=384
        )
    }
    
    # Table of results
    benchmark_results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("natural_questions", "default", split="validation[:100]")
    tokenizer = BertTokenizerStandalone()
    
    # Prepare dataset
    eval_dataset = NaturalQuestionsDataset(dataset, tokenizer, max_length=384)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8)
    
    # Compare different variants
    for config_name, config in configs.items():
        model_name = f"Quantum Classical ({config_name})"
        print(f"\nEvaluating model: {model_name}")
        
        # Create model
        model = QuantumClassicalForQuestionAnswering(config)
        model.to(device)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameter count: {param_count:,}")
        
        # Track metrics
        results = {
            "parameter_count": param_count,
            "hidden_size": config.hidden_size,
            "layers": config.num_hidden_layers,
            "gram_steps": config.num_recursive_steps
        }
        
        # Measure training time (mock training)
        print("Measuring training time...")
        start_time = time.time()
        
        # Mock training for timing (1 batch)
        model.train()
        batch = next(iter(eval_dataloader))
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items() if k != "token_type_ids" or v is not None}
        
        # Training step
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        
        # Record training time
        train_time = time.time() - start_time
        results["train_time_per_batch"] = train_time
        
        # Measure inference time
        print("Measuring inference time...")
        start_time = time.time()
        
        # Run evaluation
        model.eval()
        eval_results = evaluate_model(model, eval_dataloader, device, tokenizer, num_examples=2)
        
        # Record results
        inference_time = time.time() - start_time
        results["inference_time"] = inference_time / len(eval_dataloader)
        results["loss"] = eval_results["loss"]
        results["exact_match"] = eval_results["exact_match"]
        results["f1"] = eval_results["f1"]
        
        benchmark_results[model_name] = results
    
    # Print comparison table
    print("\n===== Model Comparison =====")
    headers = ["Model", "Params", "Hidden", "Layers", "EM", "F1", "Inf Time", "Train Time"]
    rows = []
    
    for model_name, results in benchmark_results.items():
        rows.append([
            model_name,
            f"{results['parameter_count']/1000000:.2f}M",
            str(results['hidden_size']),
            str(results['layers']),
            f"{results['exact_match']:.4f}",
            f"{results['f1']:.4f}",
            f"{results['inference_time']*1000:.2f}ms",
            f"{results['train_time_per_batch']*1000:.2f}ms"
        ])
    
    # Print as formatted table
    print_table(headers, rows)
    
    return benchmark_results

def evaluate_saved_model(model_path, dataset_path=None, num_examples=10):
    """Evaluate a saved model on the test set or a provided dataset"""
    print(f"Loading model from {model_path}...")
    
    # Load the tokenizer
    tokenizer = BertTokenizerStandalone()
    
    # Load configuration and model
    config = QuantumClassicalConfig(
        vocab_size=30522,
        hidden_size=256,
        num_hidden_layers=3,
        num_attention_heads=8,
        intermediate_size=512,
        gram_gamma=0.7,
        num_recursive_steps=3,
        gram_heads=4,
        max_position_embeddings=384
    )
    
    # Create model and load weights
    model = QuantumClassicalForQuestionAnswering(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load evaluation dataset
    if dataset_path:
        # Custom dataset provided
        print(f"Loading custom dataset from {dataset_path}...")
        # Implementation depends on your dataset format
    else:
        # Use NQ test set
        print("Loading Natural Questions test dataset...")
        dataset = load_dataset("natural_questions", "default", split="validation[:100]")
        
    # Create dataset and dataloader
    eval_dataset = NaturalQuestionsDataset(dataset, tokenizer, max_length=384)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8)
    
    # Run evaluation
    model.eval()
    results = evaluate_model(model, eval_dataloader, device, tokenizer, num_examples)
    
    # Print overall results
    print("\n===== Evaluation Results =====")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Exact Match: {results['exact_match']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    
    return results

def run_performance_analysis():
    """Analyze how different parameters affect model performance"""
    print("=" * 50)
    print("Performance Analysis of Quantum Classical Model")
    print("=" * 50)
    
    # Benchmark different model variants
    results = benchmark_quantum_classical_variants()
    
    # Try to plot results
    try:
        import matplotlib.pyplot as plt
        
        # Extract data
        models = list(results.keys())
        param_counts = [results[m]["parameter_count"]/1000000 for m in models]
        f1_scores = [results[m]["f1"] for m in models]
        em_scores = [results[m]["exact_match"] for m in models]
        inference_times = [results[m]["inference_time"]*1000 for m in models]
        
        # Create plots
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot 1: Model size vs performance
        axes[0].plot(param_counts, f1_scores, 'o-', label='F1 Score')
        axes[0].plot(param_counts, em_scores, 's-', label='Exact Match')
        axes[0].set_xlabel('Model Size (Million Parameters)')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Model Size vs Performance')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Model size vs inference time
        axes[1].plot(param_counts, inference_times, 'o-', color='red')
        axes[1].set_xlabel('Model Size (Million Parameters)')
        axes[1].set_ylabel('Inference Time (ms)')
        axes[1].set_title('Model Size vs Inference Time')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png')
        print("Performance analysis plots saved to 'performance_analysis.png'")
        
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
        print("Raw performance data:")
        for model, data in results.items():
            print(f"\n{model}:")
            for key, value in data.items():
                print(f"  - {key}: {value}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate or analyze Quantum Classical model")
    parser.add_argument("--mode", type=str, choices=["evaluate", "analyze"], 
                        default="analyze", help="Operation mode")
    parser.add_argument("--model_path", type=str, default="quantum_classical_natural_questions.pt",
                        help="Path to saved model")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to custom dataset")
    parser.add_argument("--num_examples", type=int, default=5,
                        help="Number of examples for evaluation")
    
    args = parser.parse_args()
    
    if args.mode == "evaluate":
        evaluate_saved_model(args.model_path, args.dataset_path, num_examples=args.num_examples)
    elif args.mode == "analyze":
        run_performance_analysis()