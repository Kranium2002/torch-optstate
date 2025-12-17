import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import matplotlib.pyplot as plt
import psutil
import os
import time
import gc
import numpy as np
import argparse
from typing import List, Dict
import sys

# Add src to path so we can import torch_optstate
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch_optstate
from torch_optstate.policy.simple import WarmupPolicy
from torch_optstate.codecs import Int8MomentumCodec, FP32Codec
from torch_optstate.policy.base import Policy

class Int8AllPolicy(Policy):
    def __init__(self, warmup_steps=0):
        self.warmup_steps = warmup_steps
        self.int8 = Int8MomentumCodec()
        self.fp32 = FP32Codec()
    
    def get_codecs(self, param, state, step):
        if step < self.warmup_steps:
            return {k: self.fp32 for k in state if torch.is_tensor(state[k])}
        return {k: self.int8 for k in state if torch.is_tensor(state[k])}

# Check for datasets library
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not found. Using synthetic data instead.")
    print("To use real data, install it via: pip install datasets")

class MemoryTracker:
    def __init__(self, device):
        self.device = device
        self.timestamps = []
        self.memory_usage = []
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())

    def record(self):
        self.timestamps.append(time.time() - self.start_time)
        if self.device.type == 'cuda':
            # Record in MB
            mem = torch.cuda.memory_allocated(self.device) / 1024 / 1024
        else:
            # Record RSS in MB
            mem = self.process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(mem)

    def get_data(self):
        return self.timestamps, self.memory_usage

def get_tensor_memory(optimizer):
    total_bytes = 0
    # Check standard optimizer state
    if hasattr(optimizer, 'state'):
        for param, state in optimizer.state.items():
            for key, val in state.items():
                if torch.is_tensor(val):
                    total_bytes += val.element_size() * val.numel()
    
    # Check wrapped store
    if hasattr(optimizer, 'store'):
        total_bytes += optimizer.store.get_memory_usage()
        
    return total_bytes / 1024 / 1024 # MB

class SimpleDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_data(tokenizer, num_samples=1000):
    if HAS_DATASETS:
        print("Loading IMDB dataset...")
        try:
            dataset = load_dataset("imdb", split=f"train[:{num_samples}]")
            texts = dataset['text']
            labels = dataset['label']
        except Exception as e:
            print(f"Failed to load IMDB: {e}. Falling back to synthetic.")
            texts = ["This is a sample sentence." for _ in range(num_samples)]
            labels = [0] * num_samples
    else:
        print("Generating synthetic data...")
        texts = ["This is a sample sentence for training." for _ in range(num_samples)]
        labels = [0] * num_samples

    print("Tokenizing data...")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    return SimpleDataset(encodings, labels)

def train_model(optimizer_factory, device, epochs=1, batch_size=16, max_steps=None):
    print(f"Initializing model on {device}...")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.to(device)
    model.train()

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Ensure we have enough data for the requested steps
    needed_samples = (max_steps * batch_size) if max_steps else 500
    # Add a buffer
    needed_samples += batch_size 
    
    dataset = get_data(tokenizer, num_samples=needed_samples) 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create optimizer
    optimizer = optimizer_factory(model.parameters())
    
    # Count parameter tensors
    num_params = sum(1 for _ in model.parameters())
    print(f"Model has {num_params} parameter tensors.")
    
    tracker = MemoryTracker(device)
    
    print("Starting training loop...")
    step = 0
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            t0_step = time.perf_counter()
            optimizer.step()
            t1_step = time.perf_counter()
            step_dt = t1_step - t0_step
            
            # Force GC to verify if memory growth is real or just lazy GC
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            tracker.record()
            
            # Track tensor memory specifically
            tensor_mem = get_tensor_memory(optimizer)
            
            step += 1
            
            # Check for detailed timings from wrapper
            details = ""
            if hasattr(optimizer, 'last_step_timings'):
                timings = optimizer.last_step_timings
                details = f" [Mat: {timings.get('materialize',0)*1000:.1f}ms, Step: {timings.get('step',0)*1000:.1f}ms, Cmt: {timings.get('commit',0)*1000:.1f}ms]"
            
            print(f"Step {step}, Loss: {loss.item():.4f}, Time: {step_dt*1000:.1f}ms{details}, Tensor Mem: {tensor_mem:.2f} MB")
                
            # Clean up to ensure memory measurements are accurate
            del input_ids, attention_mask, labels, outputs, loss

            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break
            
    return tracker.get_data()

def main():
    parser = argparse.ArgumentParser(description='Fine-tune demo with memory tracking')
    parser.add_argument('--steps', type=int, default=1, help='Number of steps to run')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for Torch-OptState (number of tensors, e.g. 100)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Running for {args.steps} steps")
    if args.chunk_size:
        print(f"Using chunk size: {args.chunk_size}")

    # 1. Baseline Run
    print("\n--- Running Baseline (AdamW) ---")
    def baseline_factory(params):
        return torch.optim.AdamW(params, lr=5e-5)
    
    # Clear memory before run
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    base_time, base_mem = train_model(baseline_factory, device, max_steps=args.steps)
    
    # 2. Torch-OptState Run
    print("\n--- Running Torch-OptState (Wrapped AdamW) ---")
    def optstate_factory(params):
        base_opt = torch.optim.AdamW(params, lr=5e-5)
        
        # FIX: Use WarmupPolicy instead of Int8AllPolicy.
        # Int8AllPolicy quantized Variance to Int8, which caused the "Zero Loss" issue 
        # (Variance became 0, causing division by zero in Adam).
        # WarmupPolicy keeps Variance in FP32 (safe) and compresses Momentum to Int8.
        policy = WarmupPolicy(warmup_steps=0) 
        
        return torch_optstate.wrap(base_opt, policy=policy, chunk_size=args.chunk_size)

    # Clear memory before run
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    opt_time, opt_mem = train_model(optstate_factory, device, max_steps=args.steps)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(base_time, base_mem, label='Baseline (AdamW)', alpha=0.7)
    plt.plot(opt_time, opt_mem, label='Torch-OptState', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Memory (MB)')
    plt.title(f'Memory Usage during Fine-tuning ({device.type.upper()})')
    plt.legend()
    plt.grid(True)
    
    output_file = 'memory_comparison.png'
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")
    
    # Print stats
    print(f"\nBaseline Peak: {max(base_mem):.2f} MB")
    print(f"OptState Peak: {max(opt_mem):.2f} MB")
    print(f"Baseline Final: {base_mem[-1]:.2f} MB")
    print(f"OptState Final: {opt_mem[-1]:.2f} MB")

if __name__ == "__main__":
    main()
