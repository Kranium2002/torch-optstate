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
        if opt_name == 'adamw':
             # WarmupPolicy default: exp_avg (1 byte) + exp_avg_sq (4 bytes)
             return (param_count * (1 + 4)) / 1024**2
        else:
             # SGD: momentum (1 byte)
             return (param_count * 1) / 1024**2
    elif policy_name == 'int8_all':
        # 1 byte per element
        return (param_count * states_per_param * 1) / 1024**2
    return 0

def run_benchmark():
    results = []
    
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
        
    sizes = ['small', 'medium'] 
    optimizers = ['adamw']
    policies = ['baseline', 'fp32', 'int8', 'int8_all']
    chunk_sizes = [None, 1024, 256, 64, 1] 
    
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
                        opt = get_optimizer(opt_name, model.parameters())
                        
                        # Configure Policy
                        wrapper = None
                        if policy_name == 'baseline':
                            wrapper = opt
                        else:
                            if policy_name == 'fp32':
                                # Ensure it stays in FP32 for the duration of the benchmark
                                policy = WarmupPolicy(warmup_steps=1000000) 
                            elif policy_name == 'int8':
                                policy = WarmupPolicy(warmup_steps=0) # Immediate Int8
                            elif policy_name == 'int8_all':
                                policy = WarmupPolicy(warmup_steps=0, variance_codec=Int8MomentumCodec())
                            
                            # Wrap
                            wrapper = wrap(opt, policy=policy, chunk_size=chunk_size)
                        
                        # Data
                        input_dim = 1024 if size == 'small' else (2048 if size == 'medium' else 4096)
                        x = torch.randn(32, input_dim, device=device)
                        y = torch.randn(32, 10, device=device) 
                        
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
                        print(f"  Measuring {device_name} {size} {opt_name} {policy_name} chunk={chunk_size}...")
                        reset_memory(device)
                        
                        # CPU RSS tracking
                        process = psutil.Process(os.getpid())
                        peak_rss = process.memory_info().rss
                        
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        
                        t0 = time.perf_counter()
                        steps = 10
                        
                        iter_peak_alloc_list = []
                        iter_peak_res_list = []
                        step_extra_alloc_list = []
                        end_alloc_list = []
                        end_res_list = []
                        
                        try:
                            for _ in range(steps):
                                if device.type == 'cuda':
                                    torch.cuda.synchronize()
                                    torch.cuda.reset_peak_memory_stats(device)
                                
                                wrapper.zero_grad(set_to_none=True)
                                loss = model(x).sum()
                                loss.backward()
                                
                                fwd_bwd_peak_alloc = 0
                                fwd_bwd_peak_res = 0
                                pre_step_alloc = 0
                                
                                if device.type == 'cuda':
                                    torch.cuda.synchronize()
                                    fwd_bwd_peak_alloc = torch.cuda.max_memory_allocated(device)
                                    fwd_bwd_peak_res = torch.cuda.max_memory_reserved(device)
                                    pre_step_alloc = torch.cuda.memory_allocated(device)
                                    torch.cuda.reset_peak_memory_stats(device)
                                
                                wrapper.step()
                                
                                if device.type == 'cuda':
                                    torch.cuda.synchronize()
                                    step_peak_alloc = torch.cuda.max_memory_allocated(device)
                                    step_peak_res = torch.cuda.max_memory_reserved(device)
                                    end_alloc = torch.cuda.memory_allocated(device)
                                    end_res = torch.cuda.memory_reserved(device)
                                    
                                    # Metrics
                                    iter_peak_alloc = max(fwd_bwd_peak_alloc, step_peak_alloc)
                                    iter_peak_res = max(fwd_bwd_peak_res, step_peak_res)
                                    step_extra = step_peak_alloc - pre_step_alloc
                                    
                                    iter_peak_alloc_list.append(iter_peak_alloc)
                                    iter_peak_res_list.append(iter_peak_res)
                                    step_extra_alloc_list.append(step_extra)
                                    end_alloc_list.append(end_alloc)
                                    end_res_list.append(end_res)

                                # Sample RSS (CPU)
                                current_rss = process.memory_info().rss
                                if current_rss > peak_rss:
                                    peak_rss = current_rss
                        except Exception as e:
                            print(f"  Failed during measurement: {e}")
                            continue
                            
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                            
                        t1 = time.perf_counter()
                        
                        # Metrics
                        rss_peak_mb = peak_rss / 1024**2
                        rss_end_mb = process.memory_info().rss / 1024**2
                        
                        cuda_iter_peak_mb = 0
                        cuda_iter_res_mb = 0
                        cuda_step_extra_mb = 0
                        cuda_end_alloc_mb = 0
                        cuda_end_res_mb = 0
                        
                        if device.type == 'cuda':
                            if iter_peak_alloc_list:
                                cuda_iter_peak_mb = max(iter_peak_alloc_list) / 1024**2
                                cuda_iter_res_mb = max(iter_peak_res_list) / 1024**2
                                cuda_step_extra_mb = max(step_extra_alloc_list) / 1024**2
                                cuda_end_alloc_mb = end_alloc_list[-1] / 1024**2
                                cuda_end_res_mb = end_res_list[-1] / 1024**2
                            
                        store_mb = 0
                        if hasattr(wrapper, 'store'):
                             store_mb = wrapper.store.get_memory_usage() / 1024**2
                        
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
                            'rss_peak_mb': rss_peak_mb,
                            'rss_end_mb': rss_end_mb,
                            'cuda_iter_peak_mb': cuda_iter_peak_mb,
                            'cuda_iter_res_mb': cuda_iter_res_mb,
                            'cuda_step_extra_mb': cuda_step_extra_mb,
                            'cuda_end_alloc_mb': cuda_end_alloc_mb,
                            'cuda_end_res_mb': cuda_end_res_mb,
                            'store_mb': store_mb,
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
    print(df.to_string(index=False, float_format="%.2f"))
    
    df.to_csv('benchmark_results.csv', index=False)

if __name__ == "__main__":
    run_benchmark()
