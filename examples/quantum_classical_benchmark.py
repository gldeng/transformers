#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved benchmarking for Quantum Classical models
"""

import os
import sys
import time
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
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

def decode_tokens(tokenizer, token_ids):
    """Helper function to decode token IDs to text"""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
        
    # Convert token IDs to tokens and join them
    tokens = [tokenizer.ids_to_tokens.get(token_id, "[UNK]") for token_id in token_ids]
    
    # Remove special tokens and join with spaces
    filtered_tokens = [token for token in tokens if token not in ["[PAD]", "[CLS]", "[SEP]"]]
    return " ".join(filtered_tokens)

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
                
                # Convert to text using our helper function instead of tokenizer.decode
                pred_text = decode_tokens(tokenizer, pred_tokens)
                true_text = decode_tokens(tokenizer, true_tokens)
                
                all_predictions.append(pred_text)
                all_labels.append(true_text)
                
                # Store examples to show (only a few)
                if batch_idx < num_examples and i == 0:
                    # Get the full input sequence
                    full_text = decode_tokens(tokenizer, input_ids[i])
                    
                    # Extract question and context based on the [SEP] token position
                    sep_pos = input_ids[i].tolist().index(tokenizer.vocab["[SEP]"])
                    question = decode_tokens(tokenizer, input_ids[i][1:sep_pos])  # Skip [CLS]
                    context = decode_tokens(tokenizer, input_ids[i][sep_pos+1:])  # Skip [SEP]
                    
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

def analyze_dataset_answers():
    """Analyze the dataset to understand answer patterns"""
    print("\n=== Dataset Answer Analysis ===")
    
    # Load dataset
    dataset = load_dataset("natural_questions", "default", split="train[:100]")
    tokenizer = BertTokenizerStandalone()
    
    # Check a few examples in detail
    for i, example in enumerate(dataset[:5]):
        print(f"\nExample {i+1}:")
        question = example.get("question", {}).get("text", "")
        print(f"Question: {question}")
        
        # Check annotations
        annotations = example.get("annotations", {})
        short_answers = annotations.get("short_answers", [])
        if short_answers:
            print(f"Has {len(short_answers)} short answers")
            for j, ans in enumerate(short_answers[:2]):  # Show first 2 answers
                print(f"  Answer {j+1}: Start={ans.get('start_token')}, End={ans.get('end_token')}")
        else:
            print("No short answers")
            
        # Check long answer
        if "long_answer" in annotations:
            long_ans = annotations["long_answer"]
            if long_ans.get("start_token", -1) >= 0:
                print(f"Long answer: Start={long_ans.get('start_token')}, End={long_ans.get('end_token')}")
            else:
                print("No long answer")
    
    # Check fixed position bias in the processed dataset
    qa_dataset = NaturalQuestionsDataset(dataset, tokenizer, max_length=384)
    
    start_positions = []
    end_positions = []
    
    # Collect ground truth positions
    for example in qa_dataset.examples:
        start_positions.append(example["start_positions"].item())
        end_positions.append(example["end_positions"].item())
    
    # Analyze position distributions
    print(f"\nProcessed dataset size: {len(qa_dataset)}")
    print(f"Ground truth start position distribution:")
    start_counts = {}
    for pos in start_positions:
        start_counts[pos] = start_counts.get(pos, 0) + 1
        
    # Sort by most common positions
    for pos, count in sorted(start_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  Position {pos}: {count} answers ({count/len(start_positions)*100:.1f}%)")
        
    print(f"Ground truth end position distribution:")
    end_counts = {}
    for pos in end_positions:
        end_counts[pos] = end_counts.get(pos, 0) + 1
        
    # Sort by most common positions
    for pos, count in sorted(end_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  Position {pos}: {count} answers ({count/len(end_positions)*100:.1f}%)")

def check_answer_position_bias():
    """Check if models are biased toward the fixed answer positions"""
    print("\n=== Checking Position Bias in Answer Predictions ===")
    
    # Test on different models
    models = load_variant_models()
    dataset = load_dataset("natural_questions", "default", split="validation[:50]")
    tokenizer = BertTokenizerStandalone()
    eval_dataset = NaturalQuestionsDataset(dataset, tokenizer, max_length=384)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4)
    
    for name, model in models.items():
        print(f"\n--- Model: {name} ---")
        model.eval()
        
        start_positions = []
        end_positions = []
        
        # Collect predictions
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items() if k != "token_type_ids" or v is not None}
                outputs = model(**batch)
                
                # Get predictions
                start_pred = torch.argmax(outputs["start_logits"], dim=1)
                end_pred = torch.argmax(outputs["end_logits"], dim=1)
                
                # Store positions
                start_positions.extend(start_pred.cpu().tolist())
                end_positions.extend(end_pred.cpu().tolist())
        
        # Analyze position distributions
        print(f"Number of examples: {len(start_positions)}")
        print(f"Start position distribution:")
        start_counts = {}
        for pos in start_positions:
            start_counts[pos] = start_counts.get(pos, 0) + 1
            
        # Sort by most common positions
        for pos, count in sorted(start_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  Position {pos}: {count} predictions ({count/len(start_positions)*100:.1f}%)")
            
        print(f"End position distribution:")
        end_counts = {}
        for pos in end_positions:
            end_counts[pos] = end_counts.get(pos, 0) + 1
            
        # Sort by most common positions
        for pos, count in sorted(end_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  Position {pos}: {count} predictions ({count/len(end_positions)*100:.1f}%)")

def debug_f1_calculation():
    """Print detailed analysis of F1 calculation"""
    print("\n=== F1 Score Calculation Analysis ===")
    
    # Load your model variants
    models = load_variant_models()
    dataset = load_dataset("natural_questions", "default", split="validation[:20]")  # Use fewer examples
    tokenizer = BertTokenizerStandalone()
    eval_dataset = NaturalQuestionsDataset(dataset, tokenizer, max_length=384)
    eval_dataloader = DataLoader(eval_dataset, batch_size=2)
    
    # Check predictions for each model
    for name, model in models.items():
        print(f"\n--- Model: {name} ---")
        model.eval()
        
        # Collect all predictions and labels
        all_examples = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items() if k != "token_type_ids" or v is not None}
                outputs = model(**batch)
                
                # Get predictions
                start_pred = torch.argmax(outputs["start_logits"], dim=1)
                end_pred = torch.argmax(outputs["end_logits"], dim=1)
                
                for i in range(len(batch["input_ids"])):
                    # Extract tokens for both predicted and true answers
                    pred_start = start_pred[i].item()
                    pred_end = end_pred[i].item()
                    true_start = batch["start_positions"][i].item()
                    true_end = batch["end_positions"][i].item()
                    
                    if pred_end < pred_start:
                        pred_end = pred_start
                        
                    # Get the tokens
                    input_ids = batch["input_ids"][i]
                    pred_tokens = input_ids[pred_start:pred_end+1]
                    true_tokens = input_ids[true_start:true_end+1]
                    
                    # Decode
                    pred_text = decode_tokens(tokenizer, pred_tokens)
                    true_text = decode_tokens(tokenizer, true_tokens)
                    
                    # Calculate F1 for this example
                    pred_words = set(pred_text.strip().lower().split())
                    true_words = set(true_text.strip().lower().split())
                    
                    common_words = pred_words.intersection(true_words)
                    precision = len(common_words) / len(pred_words) if len(pred_words) > 0 else 0
                    recall = len(common_words) / len(true_words) if len(true_words) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    all_examples.append({
                        "pred_text": pred_text,
                        "true_text": true_text,
                        "pred_words": pred_words,
                        "true_words": true_words,
                        "common_words": common_words,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1
                    })
        
        # Detailed analysis of the F1 scores
        print(f"Number of examples: {len(all_examples)}")
        avg_f1 = sum(ex["f1"] for ex in all_examples) / len(all_examples)
        print(f"Average F1 score: {avg_f1:.4f}")
        
        # Show distribution of F1 scores
        f1_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        distribution = [0] * (len(f1_bins) - 1)
        for ex in all_examples:
            for i in range(len(f1_bins) - 1):
                if f1_bins[i] <= ex["f1"] < f1_bins[i+1]:
                    distribution[i] += 1
                elif i == len(f1_bins) - 2 and ex["f1"] == f1_bins[i+1]:  # For F1=1.0
                    distribution[i] += 1
                    
        print("F1 score distribution:")
        for i in range(len(f1_bins) - 1):
            print(f"  {f1_bins[i]:.1f}-{f1_bins[i+1]:.1f}: {distribution[i]} examples ({distribution[i]/len(all_examples)*100:.1f}%)")
            
        # Print some example predictions 
        print("\nSample predictions:")
        for i, ex in enumerate(all_examples[:5]):
            print(f"\nExample {i+1}:")
            print(f"  True: '{ex['true_text']}'")
            print(f"  Pred: '{ex['pred_text']}'")
            print(f"  Common words: {ex['common_words']}")
            print(f"  Precision={ex['precision']:.2f}, Recall={ex['recall']:.2f}, F1={ex['f1']:.2f}")

def compare_trained_vs_random():
    """Compare trained model vs. random initialization"""
    print("\n=== Comparing Trained vs Random Models ===")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset("natural_questions", "default", split="validation[:50]")
    tokenizer = BertTokenizerStandalone()
    eval_dataset = NaturalQuestionsDataset(dataset, tokenizer, max_length=384)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4)
    
    # Define configs
    configs = {
        "Small": {"hidden_size": 128, "layers": 2},
        "Medium": {"hidden_size": 256, "layers": 3},
        "Large": {"hidden_size": 384, "layers": 6}
    }
    
    # Test both random and pretrained weights
    results = {}
    
    for size, params in configs.items():
        print(f"\n--- Testing {size} model ---")
        
        # Create config
        config = QuantumClassicalConfig(
            vocab_size=30522,
            hidden_size=params["hidden_size"],
            num_hidden_layers=params["layers"],
            num_attention_heads=params["hidden_size"] // 32,
            intermediate_size=params["hidden_size"] * 2,
            max_position_embeddings=384
        )
        
        # Random weights
        random_model = QuantumClassicalForQuestionAnswering(config)
        random_model.to(device)
        print("Evaluating with random weights...")
        random_results = evaluate_model(random_model, eval_dataloader, device, tokenizer)
        
        # Try to load trained weights if available
        model_path = f"quantum_classical_{size.lower()}.pt"
        try:
            trained_model = QuantumClassicalForQuestionAnswering(config)
            trained_model.load_state_dict(torch.load(model_path))
            trained_model.to(device)
            print(f"Evaluating with trained weights from {model_path}...")
            trained_results = evaluate_model(trained_model, eval_dataloader, device, tokenizer)
        except:
            print(f"No trained weights found at {model_path}")
            trained_results = {"loss": "N/A", "exact_match": "N/A", "f1": "N/A"}
        
        results[f"{size} (Random)"] = random_results
        results[f"{size} (Trained)"] = trained_results
    
    # Print comparison
    print("\n=== Results Comparison ===")
    headers = ["Model", "Loss", "Exact Match", "F1"]
    rows = []
    
    for model_name, res in results.items():
        rows.append([
            model_name, 
            f"{res['loss']:.4f}" if isinstance(res['loss'], float) else res['loss'],
            f"{res['exact_match']:.4f}" if isinstance(res['exact_match'], float) else res['exact_match'],
            f"{res['f1']:.4f}" if isinstance(res['f1'], float) else res['f1']
        ])
    
    print_table(headers, rows)

def run_benchmark_with_randomized_answers():
    """Run the improved benchmark with adjusted settings"""
    # First, run diagnostic functions to understand what's happening
    analyze_dataset_answers()
    check_answer_position_bias()
    debug_f1_calculation()
    compare_trained_vs_random()
    
    # Now run the improved benchmark with adjusted settings
    benchmark_quantum_classical_variants()

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
        run_benchmark_with_randomized_answers()