import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch_optstate import wrap, WarmupPolicy
from torch_optstate.codecs import IdentityCodec, Int8MomentumCodec, FP16Codec
import time
import pandas as pd
import gc
import psutil
import os
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
        # Use psutil for CPU RSS
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

def reset_memory(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    else:
        # For CPU, we can't easily "reset" RSS, but we can force GC
        pass

def stop_memory_tracking(device):
    pass

def calculate_theoretical_state_mem(model, opt_name, policy_name):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # AdamW has 2 states (exp_avg, exp_avg_sq), SGD has 1 (momentum)
    states_per_param = 2 if opt_name == 'adamw' else 1
    
    if policy_name in ['baseline', 'fp32']:
        # 4 bytes per element
        return (param_count * states_per_param * 4) / 1024**2
    elif policy_name == 'int8':
        # 1 byte per element (ignoring scale overhead which is negligible)
        return (param_count * states_per_param * 1) / 1024**2
    return 0

def run_benchmark():
    results = []
    
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
        
    sizes = ['small', 'medium'] # Large might be too slow for quick bench
    optimizers = ['adamw']
    policies = ['baseline', 'fp32', 'int8']
    chunk_sizes = [None, 1] # None = full batch, 1 = layer-wise chunking
    
    print(f"Running benchmarks on: {devices}")
    
    for device_name in devices:
        device = torch.device(device_name)
        
        for size in sizes:
            for opt_name in optimizers:
                for policy_name in policies:
                    for chunk_size in chunk_sizes:
                        
                        if policy_name == 'baseline' and chunk_size is not None:
                            continue

                        # Setup
                        reset_memory(device)
                        model = get_model(size, device)
                        # Ensure model has enough params to make chunking interesting
                        # MLP has few params (layers * 2). 
                        # If chunk_size is 1, we process 1 param at a time.
                        
                        opt = get_optimizer(opt_name, model.parameters())
                        
                        # Configure Policy
                        wrapper = None
                        if policy_name == 'baseline':
                            wrapper = opt
                        else:
                            if policy_name == 'fp32':
                                policy = None # Default Warmup (FP32 offload)
                            elif policy_name == 'int8':
                                policy = WarmupPolicy(warmup_steps=0) # Immediate Int8
                            
                            # Wrap
                            wrapper = wrap(opt, policy=policy, chunk_size=chunk_size)
                        
                        # Data
                        input_dim = 1024 if size == 'small' else (2048 if size == 'medium' else 4096)
                        x = torch.randn(32, input_dim, device=device)
                        y = torch.randn(32, 10, device=device) # Output dim 10
                        
                        # Warmup
                        print(f"  Warmup (may compile)...")
                        try:
                            for _ in range(3):
                                wrapper.zero_grad()
                                loss = model(x).sum()
                                loss.backward()
                                wrapper.step()
                        except Exception as e:
                            print(f"  Failed during warmup: {e}")
                            continue
                            
                        # Measure
                        print(f"  Measuring...")
                        reset_memory(device)
                        start_mem = measure_peak_memory(device)
                        
                        t0 = time.perf_counter()
                        steps = 10
                        try:
                            for _ in range(steps):
                                wrapper.zero_grad()
                                loss = model(x).sum()
                                loss.backward()
                                wrapper.step()
                        except Exception as e:
                            print(f"  Failed during measurement: {e}")
                            continue
                        t1 = time.perf_counter()
                        
                        peak_mem = measure_peak_memory(device)
                        
                        if device.type == 'cuda':
                            mem_usage = peak_mem - start_mem
                        else:
                            # For CPU RSS, it's noisy. We take the diff.
                            mem_usage = peak_mem - start_mem
                        
                        avg_time = (t1 - t0) / steps
                        
                        # Get breakdown from last step
                        if hasattr(wrapper, 'last_step_timings'):
                            timings = wrapper.last_step_timings
                        else:
                            timings = {}
                        
                        theo_mem = calculate_theoretical_state_mem(model, opt_name, policy_name)
                        
                        res = {
                            'device': device_name,
                            'size': size,
                            'opt': opt_name,
                            'policy': policy_name,
                            'chunk': str(chunk_size),
                            'theo_mb': theo_mem,
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
    # Use to_string to avoid tabulate dependency
    print(df.to_string(index=False, float_format="%.2f"))
    
    # Save
    df.to_csv('benchmark_results.csv', index=False)

if __name__ == "__main__":
    run_benchmark()
