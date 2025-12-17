import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch_optstate import wrap, WarmupPolicy
from torch_optstate.codecs import IdentityCodec, Int8MomentumCodec, FP16Codec
import time
import pandas as pd
import gc
from benchmarks.models import MLP

def get_model(size, device):
    if size == 'small':
        return MLP(input_dim=1024, hidden_dim=1024, layers=3).to(device)
    elif size == 'medium':
        return MLP(input_dim=2048, hidden_dim=2048, layers=5).to(device)
    elif size == 'large':
        return MLP(input_dim=4096, hidden_dim=4096, layers=5).to(device)
    else:
        raise ValueError(f"Unknown size: {size}")

def get_optimizer(opt_name, params):
    if opt_name == 'adamw':
        return AdamW(params, lr=1e-3)
    elif opt_name == 'sgd':
        return SGD(params, lr=1e-3, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

def measure_peak_memory(device):
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device)
    else:
        # CPU memory is hard to measure precisely in python without external tools
        # We'll return 0 for now or use psutil if needed, but let's stick to CUDA for mem
        return 0

def reset_memory(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

def run_benchmark():
    results = []
    
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
        
    sizes = ['small', 'medium'] # Large might be too slow for quick bench
    optimizers = ['adamw']
    policies = ['fp32', 'int8']
    chunk_sizes = [None, 1024] # None = full batch, 1024 = chunked
    
    print(f"Running benchmarks on: {devices}")
    
    for device_name in devices:
        device = torch.device(device_name)
        
        for size in sizes:
            for opt_name in optimizers:
                for policy_name in policies:
                    for chunk_size in chunk_sizes:
                        
                        # Setup
                        reset_memory(device)
                        model = get_model(size, device)
                        opt = get_optimizer(opt_name, model.parameters())
                        
                        # Configure Policy
                        if policy_name == 'fp32':
                            # Use IdentityCodec to simulate standard behavior (keep on device)
                            # Or CPUOffloadCodec if we want to test offloading?
                            # Let's test "Standard" (Identity) vs "Virtualized" (Offload/Compress)
                            # Actually, WarmupPolicy defaults to FP32Codec (Offload)
                            # Let's make explicit policies
                            policy = None # Default Warmup
                        elif policy_name == 'int8':
                            policy = WarmupPolicy(warmup_steps=0) # Immediate Int8
                        
                        # Wrap
                        wrapper = wrap(opt, policy=policy, chunk_size=chunk_size)
                        
                        # Data
                        input_dim = 1024 if size == 'small' else (2048 if size == 'medium' else 4096)
                        x = torch.randn(32, input_dim, device=device)
                        y = torch.randn(32, 10, device=device) # Output dim 10
                        
                        # Warmup
                        for _ in range(3):
                            wrapper.zero_grad()
                            loss = model(x).sum()
                            loss.backward()
                            wrapper.step()
                            
                        # Measure
                        reset_memory(device)
                        start_mem = measure_peak_memory(device)
                        
                        t0 = time.perf_counter()
                        steps = 10
                        for _ in range(steps):
                            wrapper.zero_grad()
                            loss = model(x).sum()
                            loss.backward()
                            wrapper.step()
                        t1 = time.perf_counter()
                        
                        peak_mem = measure_peak_memory(device)
                        mem_usage = peak_mem - start_mem
                        
                        avg_time = (t1 - t0) / steps
                        
                        # Get breakdown from last step
                        timings = wrapper.last_step_timings
                        
                        res = {
                            'device': device_name,
                            'size': size,
                            'opt': opt_name,
                            'policy': policy_name,
                            'chunk': str(chunk_size),
                            'mem_mb': mem_usage / 1024**2,
                            'time_ms': avg_time * 1000,
                            'mat_ms': timings.get('materialize', 0) * 1000,
                            'step_ms': timings.get('step', 0) * 1000,
                            'com_ms': timings.get('commit', 0) * 1000,
                            'ovh_ms': timings.get('overhead', 0) * 1000
                        }
                        results.append(res)
                        print(f"Finished {res}")
                        
                        # Cleanup
                        del model, opt, wrapper, x, y
                        reset_memory(device)

    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df.to_markdown(index=False, floatfmt=".2f"))
    
    # Save
    df.to_csv('benchmark_results.csv', index=False)

if __name__ == "__main__":
    run_benchmark()
