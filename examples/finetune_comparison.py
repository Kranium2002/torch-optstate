import torch
import time
import psutil
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import argparse

# Try importing required libraries
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("Please install transformers: pip install transformers")
    exit(1)

import torch_optstate
from torch_optstate.wrap import wrap
from torch_optstate.policy.simple import WarmupPolicy

def get_memory_usage(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated(device) / 1024**2  # MB
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2  # MB

def get_peak_memory(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / 1024**2 # MB
    else:
        # CPU peak tracking is harder without external sampling, 
        # so we just return current for now or rely on sampling history
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2

def train_model(
    device, 
    use_wrapper: bool, 
    chunk_size: int = None, 
    steps: int = 50,
    batch_size: int = 16
):
    print(f"\nStarting run: Wrapper={use_wrapper}, Chunk={chunk_size}")
    
    # 1. Setup Model (Fresh copy)
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    model.train()
    
    # 2. Setup Data (Synthetic "Real-world" shape)
    # We use synthetic data to avoid downloading large datasets, but it mimics real inputs
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch_size * steps, seq_len)).to(device)
    attention_mask = torch.ones((batch_size * steps, seq_len)).to(device)
    labels = torch.randint(0, 2, (batch_size * steps,)).to(device)
    
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # 3. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    if use_wrapper:
        # Use Int8 compression after 10 steps of warmup
        policy = WarmupPolicy(warmup_steps=10) 
        optimizer = wrap(optimizer, policy=policy, chunk_size=chunk_size)
        
    # 4. Training Loop
    memory_history = []
    timestamps = []
    start_time = time.time()
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    gc.collect()
    initial_mem = get_memory_usage(device)
    print(f"Initial Memory: {initial_mem:.2f} MB")

    for step, batch in enumerate(dataloader):
        if step >= steps:
            break
            
        b_input_ids, b_mask, b_labels = batch
        
        # Forward
        outputs = model(b_input_ids, attention_mask=b_mask, labels=b_labels)
        loss = outputs.loss
        
        # Backward
        loss.backward()
        
        # Step
        optimizer.step()
        optimizer.zero_grad()
        
        # Record Memory
        current_mem = get_memory_usage(device)
        memory_history.append(current_mem)
        timestamps.append(time.time() - start_time)
        
        if step % 10 == 0:
            print(f"Step {step}: Loss={loss.item():.4f}, Mem={current_mem:.2f} MB")

    peak_mem = get_peak_memory(device) if device.type == 'cuda' else max(memory_history)
    
    # Cleanup
    del model
    del optimizer
    del input_ids
    del attention_mask
    del labels
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return timestamps, memory_history, peak_mem

def main():
    parser = argparse.ArgumentParser(description="Compare standard optimizer vs torch-optstate")
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Chunk size for wrapper")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run Baseline
    print("Running Baseline (Standard AdamW)...")
    t_base, mem_base, peak_base = train_model(device, use_wrapper=False, steps=args.steps, batch_size=args.batch_size)

    # Run Wrapped
    print("Running torch-optstate (Int8 Compression)...")
    t_wrap, mem_wrap, peak_wrap = train_model(device, use_wrapper=True, steps=args.steps, batch_size=args.batch_size)
    
    # Run Wrapped with Chunking
    print(f"Running torch-optstate (Int8 + Chunking={args.chunk_size})...")
    t_chunk, mem_chunk, peak_chunk = train_model(device, use_wrapper=True, chunk_size=args.chunk_size, steps=args.steps, batch_size=args.batch_size)

    # Plotting
    plt.figure(figsize=(12, 6))
    
    plt.plot(t_base, mem_base, label=f'Baseline (Peak: {peak_base:.0f} MB)', alpha=0.8)
    plt.plot(t_wrap, mem_wrap, label=f'OptState Int8 (Peak: {peak_wrap:.0f} MB)', alpha=0.8)
    plt.plot(t_chunk, mem_chunk, label=f'OptState Int8+Chunk (Peak: {peak_chunk:.0f} MB)', alpha=0.8)
    
    plt.title(f"Memory Usage Comparison (DistilBERT Finetuning on {device.type.upper()})")
    plt.xlabel("Time (s)")
    plt.ylabel("Memory Allocated (MB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = "memory_comparison.png"
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Baseline Peak: {peak_base:.2f} MB")
    print(f"OptState Peak: {peak_wrap:.2f} MB ({(1 - peak_wrap/peak_base)*100:.1f}% reduction)")
    print(f"Chunked  Peak: {peak_chunk:.2f} MB ({(1 - peak_chunk/peak_base)*100:.1f}% reduction)")

if __name__ == "__main__":
    main()
