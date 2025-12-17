import time
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD, RMSprop, Adagrad
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
    if "AdamW" in optimizer_name:
        base_opt_cls = AdamW
        kwargs = {'lr': 1e-3}
    elif "SGD" in optimizer_name:
        base_opt_cls = SGD
        kwargs = {'lr': 1e-3, 'momentum': 0.9}
    elif "RMSprop" in optimizer_name:
        base_opt_cls = RMSprop
        kwargs = {'lr': 1e-3, 'momentum': 0.9}
    elif "Adagrad" in optimizer_name:
        base_opt_cls = Adagrad
        kwargs = {'lr': 1e-3}
    else:
        raise ValueError(f"Unknown optimizer in {optimizer_name}")

    if "Baseline" in optimizer_name:
        opt = base_opt_cls(model.parameters(), **kwargs)
    elif "Wrapped FP32" in optimizer_name:
        # Warmup infinity = always FP32
        policy = WarmupPolicy(warmup_steps=steps + 10)
        opt = wrap(base_opt_cls(model.parameters(), **kwargs), policy=policy)
    elif "Wrapped Compressed" in optimizer_name:
        # Warmup 10 steps, then compress
        # Configure keys based on optimizer
        momentum_key = 'exp_avg'
        variance_key = 'exp_avg_sq'
        
        if "SGD" in optimizer_name:
            momentum_key = 'momentum_buffer'
            variance_key = 'unused' # SGD has no variance
        elif "RMSprop" in optimizer_name:
            momentum_key = 'momentum_buffer'
            variance_key = 'square_avg'
        elif "Adagrad" in optimizer_name:
            momentum_key = 'unused'
            variance_key = 'sum'
            
        policy = WarmupPolicy(warmup_steps=10, momentum_key=momentum_key, variance_key=variance_key)
        opt = wrap(base_opt_cls(model.parameters(), **kwargs), policy=policy)
    elif "Wrapped Chunked" in optimizer_name:
        # Chunked update for low peak memory
        momentum_key = 'exp_avg'
        variance_key = 'exp_avg_sq'
        
        if "SGD" in optimizer_name:
            momentum_key = 'momentum_buffer'
            variance_key = 'unused'
        elif "RMSprop" in optimizer_name:
            momentum_key = 'momentum_buffer'
            variance_key = 'square_avg'
        elif "Adagrad" in optimizer_name:
            momentum_key = 'unused'
            variance_key = 'sum'
            
        policy = WarmupPolicy(warmup_steps=10, momentum_key=momentum_key, variance_key=variance_key)
        # Chunk size 1 means process 1 param at a time (lowest memory)
        opt = wrap(base_opt_cls(model.parameters(), **kwargs), policy=policy, chunk_size=1)
    
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
        # Estimate for baseline (very rough, assumes 2 states for Adam/RMSprop, 1 for SGD/Adagrad)
        num_states = 2 if "Adam" in optimizer_name or "RMSprop" in optimizer_name else 1
        for p in model.parameters():
            opt_state_size += p.numel() * 4 * num_states / 1024 / 1024
            
    print(f"  Avg Step Time: {avg_time:.2f} ms")
    print(f"  Peak RSS Increase: {end_mem - start_mem:.2f} MB")
    print(f"  Est. Opt State Size: {opt_state_size:.2f} MB")
    print(f"  Final Loss: {losses[-1]:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    print(f"Initial Memory: {get_memory_usage():.2f} MB")
    
    # AdamW
    benchmark_optimizer("AdamW (Baseline)")
    benchmark_optimizer("AdamW (Wrapped FP32)")
    benchmark_optimizer("AdamW (Wrapped Compressed)")
    benchmark_optimizer("AdamW (Wrapped Chunked)")
    
    # SGD
    benchmark_optimizer("SGD (Baseline)")
    benchmark_optimizer("SGD (Wrapped FP32)")
    benchmark_optimizer("SGD (Wrapped Compressed)")
    benchmark_optimizer("SGD (Wrapped Chunked)")

    # RMSprop
    benchmark_optimizer("RMSprop (Baseline)")
    benchmark_optimizer("RMSprop (Wrapped FP32)")
    benchmark_optimizer("RMSprop (Wrapped Compressed)")
    benchmark_optimizer("RMSprop (Wrapped Chunked)")

