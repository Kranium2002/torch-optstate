import time
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD, RMSprop, Adagrad
from torch_optstate import wrap, WarmupPolicy
import psutil
import os
import gc

def get_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def get_gpu_memory_usage():
    return torch.cuda.memory_allocated() / 1024 / 1024 # MB

def reset_peak_memory():
    torch.cuda.reset_peak_memory_stats()

def get_peak_gpu_memory():
    return torch.cuda.max_memory_allocated() / 1024 / 1024 # MB

def benchmark_optimizer_gpu(optimizer_name, steps=100, hidden_dim=4096, layers=4):
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmark.")
        return

    print(f"Benchmarking {optimizer_name} on GPU...")
    
    device = torch.device('cuda')
    
    # Create a larger model for GPU to see VRAM impact
    model = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 10)
    ).to(device)
    
    # Data
    x = torch.randn(32, hidden_dim, device=device)
    y = torch.randn(32, 10, device=device)
    
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
    
    # Warmup
    for _ in range(5):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
        
    gc.collect()
    torch.cuda.empty_cache()
    reset_peak_memory()
    
    start_cpu_mem = get_cpu_memory_usage()
    start_gpu_mem = get_gpu_memory_usage()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    losses = []
    
    for i in range(steps):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    end_cpu_mem = get_cpu_memory_usage()
    peak_gpu_mem = get_peak_gpu_memory()
    
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
    print(f"  Peak GPU VRAM: {peak_gpu_mem:.2f} MB")
    print(f"  CPU RAM Increase: {end_cpu_mem - start_cpu_mem:.2f} MB")
    print(f"  Est. Opt State Size (Compressed): {opt_state_size:.2f} MB")
    print(f"  Final Loss: {losses[-1]:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run GPU benchmark.")
    else:
        print(f"Initial CPU Memory: {get_cpu_memory_usage():.2f} MB")
        print(f"Initial GPU Memory: {get_gpu_memory_usage():.2f} MB")
        
        # AdamW
        benchmark_optimizer_gpu("AdamW (Baseline)")
        benchmark_optimizer_gpu("AdamW (Wrapped Compressed)")
        
        # SGD
        benchmark_optimizer_gpu("SGD (Baseline)")
        benchmark_optimizer_gpu("SGD (Wrapped Compressed)")
