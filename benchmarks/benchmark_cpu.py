import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_optstate import wrap, WarmupPolicy
import psutil
import os
import gc

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def benchmark_optimizer(optimizer_name, steps=100, hidden_dim=1024, layers=4):
    print(f"Benchmarking {optimizer_name}...")
    
    # Create a reasonably sized model
    model = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 10)
    )
    
    # Data
    x = torch.randn(32, hidden_dim)
    y = torch.randn(32, 10)
    
    # Optimizer
    if optimizer_name == "AdamW (Baseline)":
        opt = AdamW(model.parameters(), lr=1e-3)
    elif optimizer_name == "AdamW (Wrapped FP32)":
        # Warmup infinity = always FP32
        policy = WarmupPolicy(warmup_steps=steps + 10)
        opt = wrap(AdamW(model.parameters(), lr=1e-3), policy=policy)
    elif optimizer_name == "AdamW (Wrapped Compressed)":
        # Warmup 10 steps, then compress
        policy = WarmupPolicy(warmup_steps=10)
        opt = wrap(AdamW(model.parameters(), lr=1e-3), policy=policy)
    
    # Warmup
    for _ in range(5):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    start_mem = get_memory_usage()
    start_time = time.time()
    
    losses = []
    
    for i in range(steps):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        
    end_time = time.time()
    end_mem = get_memory_usage()
    
    avg_time = (end_time - start_time) / steps * 1000 # ms
    
    # Estimate optimizer state size if possible
    opt_state_size = 0
    if hasattr(opt, 'store'):
        opt_state_size = opt.store.get_memory_usage() / 1024 / 1024 # MB
    else:
        # Estimate for baseline AdamW (2 states per param * 4 bytes)
        for p in model.parameters():
            opt_state_size += p.numel() * 4 * 2 / 1024 / 1024
            
    print(f"  Avg Step Time: {avg_time:.2f} ms")
    print(f"  Peak RSS Increase: {end_mem - start_mem:.2f} MB")
    print(f"  Est. Opt State Size: {opt_state_size:.2f} MB")
    print(f"  Final Loss: {losses[-1]:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    print(f"Initial Memory: {get_memory_usage():.2f} MB")
    benchmark_optimizer("AdamW (Baseline)")
    benchmark_optimizer("AdamW (Wrapped FP32)")
    benchmark_optimizer("AdamW (Wrapped Compressed)")
