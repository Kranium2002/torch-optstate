import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_optstate import wrap, WarmupPolicy, ConfigurablePolicy, FP32Codec, FP16Codec, Int8MomentumCodec
from models import MLP, SimpleCNN, TinyTransformer
import psutil
import os
import gc
import pandas as pd
import numpy as np
import platform
import threading
import statistics
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, List, Optional, Tuple

# --- Configuration ---
NUM_THREADS = 1
torch.set_num_threads(NUM_THREADS)

@dataclass
class BenchmarkConfig:
    model_name: str
    optimizer_name: str
    policy_name: str
    batch_size: int
    num_steps: int
    warmup_steps: int
    seed: int
    device: str = "cpu"

@dataclass
class BenchmarkResult:
    # Config / Repro
    model: str
    optimizer: str
    policy: str
    torch_version: str
    python_version: str
    cpu_model: str
    num_threads: int
    seed: int
    batch_size: int
    num_steps: int
    warmup_steps: int
    
    # Timing
    avg_step_time_ms: float
    step_time_p50_ms: float
    step_time_p90_ms: float
    step_time_std_ms: float
    samples_per_sec: float
    
    # Memory
    model_param_mb: float
    grad_mb: float
    optimizer_state_mb: float
    peak_rss_mb: float
    steady_rss_mb: float
    
    # Correctness
    final_loss: float
    relative_loss_error: float # vs Baseline
    status: str # "OK", "Failed"
    error_msg: str = ""

class MemoryMonitor(threading.Thread):
    def __init__(self, interval=0.001): # 1ms interval
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_rss = 0
        self.process = psutil.Process()
        self.start_rss = 0

    def run(self):
        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss
        while not self.stop_event.is_set():
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except:
                pass
            time.sleep(self.interval)
            
    def stop(self):
        self.stop_event.set()
        self.join()
        return self.peak_rss

def get_cpu_name():
    return platform.processor()

def measure_tensor_bytes(t: torch.Tensor) -> int:
    return t.element_size() * t.numel()

def get_model_size(model: nn.Module) -> Tuple[float, float]:
    param_bytes = 0
    grad_bytes = 0
    for p in model.parameters():
        param_bytes += measure_tensor_bytes(p)
        if p.grad is not None:
            grad_bytes += measure_tensor_bytes(p.grad)
    return param_bytes / 1024**2, grad_bytes / 1024**2

def get_optimizer_state_size(optimizer: optim.Optimizer) -> float:
    total_bytes = 0
    
    # Check if it's our wrapper
    if hasattr(optimizer, 'store'):
        # Use the store's accounting
        total_bytes = optimizer.store.get_memory_usage()
        # Also check if there's anything lingering in optimizer.state (should be empty usually)
        for state in optimizer.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    total_bytes += measure_tensor_bytes(v)
    else:
        # Standard optimizer
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    total_bytes += measure_tensor_bytes(v)
                    
    return total_bytes / 1024**2

def run_benchmark(
    model_factory: Callable[[], nn.Module],
    input_shape: tuple,
    target_shape: tuple,
    optimizer_cls: Any,
    optimizer_kwargs: Dict[str, Any],
    policy_factory: Callable[[], Any],
    config: BenchmarkConfig,
    baseline_loss: Optional[float] = None
) -> BenchmarkResult:
    
    print(f"Running {config.model_name} / {config.optimizer_name} / {config.policy_name}...")
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    mem_monitor = None
    try:
        # Setup Model
        model = model_factory()
        
        # Pre-generate Data
        inputs = []
        targets = []
        
        # Fix data generation for classification tasks
        # MLP: out 10. CNN: out 10. Transformer: out 2.
        # We need LongTensor targets for CrossEntropy
        num_classes = 10
        if "Transformer" in config.model_name:
            num_classes = 2
        
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(config.num_steps + config.warmup_steps):
            if "Transformer" in config.model_name:
                x = torch.randint(0, 1000, input_shape)
            else:
                x = torch.randn(*input_shape)
            
            y = torch.randint(0, num_classes, (input_shape[0],))
            
            inputs.append(x)
            targets.append(y)

        # Setup Optimizer
        base_opt = optimizer_cls(model.parameters(), **optimizer_kwargs)
        policy = policy_factory()
        
        if policy is not None:
            # Handle policy configuration for specific optimizers
            if isinstance(policy, WarmupPolicy):
                if 'SGD' in config.optimizer_name:
                    policy.momentum_key = 'momentum_buffer'
                    policy.variance_key = 'unused_key'
                elif 'RMSprop' in config.optimizer_name:
                    policy.momentum_key = 'momentum_buffer'
                    policy.variance_key = 'square_avg'
                elif 'Adagrad' in config.optimizer_name:
                    policy.momentum_key = 'unused_key'
                    policy.variance_key = 'sum'

            opt = wrap(base_opt, policy=policy)
        else:
            opt = base_opt

        # Warmup
        model.train()
        for i in range(config.warmup_steps):
            opt.zero_grad()
            out = model(inputs[i])
            loss = criterion(out, targets[i])
            loss.backward()
            opt.step()
            
        # Force GC
        gc.collect()
        
        # Start Monitoring
        mem_monitor = MemoryMonitor()
        mem_monitor.start()
        
        step_times = []
        losses = []
        
        # Benchmark Loop
        start_loop = time.perf_counter()
        
        for i in range(config.num_steps):
            idx = i + config.warmup_steps
            x, y = inputs[idx], targets[idx]
            
            t0 = time.perf_counter()
            
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            
            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1000) # ms
            losses.append(loss.item())
            
        peak_rss = mem_monitor.stop()
        mem_monitor = None # Clear ref
        
        # Measure Steady State RSS (after loop, force GC)
        gc.collect()
        steady_rss = psutil.Process().memory_info().rss
        
        # Metrics
        avg_time = statistics.mean(step_times)
        p50_time = statistics.median(step_times)
        p90_time = statistics.quantiles(step_times, n=10)[8] # 9th quantile is 90%
        std_time = statistics.stdev(step_times) if len(step_times) > 1 else 0.0
        
        final_loss = losses[-1]
        if np.isnan(final_loss) or np.isinf(final_loss):
            raise ValueError("Loss is NaN or Inf")
            
        param_mb, grad_mb = get_model_size(model)
        opt_state_mb = get_optimizer_state_size(opt)
        
        rel_error = 0.0
        if baseline_loss is not None and baseline_loss != 0:
            rel_error = abs(final_loss - baseline_loss) / baseline_loss
        
        return BenchmarkResult(
            model=config.model_name,
            optimizer=config.optimizer_name,
            policy=config.policy_name,
            torch_version=torch.__version__,
            python_version=platform.python_version(),
            cpu_model=get_cpu_name(),
            num_threads=NUM_THREADS,
            seed=config.seed,
            batch_size=config.batch_size,
            num_steps=config.num_steps,
            warmup_steps=config.warmup_steps,
            avg_step_time_ms=avg_time,
            step_time_p50_ms=p50_time,
            step_time_p90_ms=p90_time,
            step_time_std_ms=std_time,
            samples_per_sec=config.batch_size / (avg_time / 1000),
            model_param_mb=param_mb,
            grad_mb=grad_mb,
            optimizer_state_mb=opt_state_mb,
            peak_rss_mb=peak_rss / 1024**2,
            steady_rss_mb=steady_rss / 1024**2,
            final_loss=final_loss,
            relative_loss_error=rel_error,
            status="OK"
        )

    except Exception as e:
        if mem_monitor:
            mem_monitor.stop()
            
        return BenchmarkResult(
            model=config.model_name,
            optimizer=config.optimizer_name,
            policy=config.policy_name,
            torch_version=torch.__version__,
            python_version=platform.python_version(),
            cpu_model=get_cpu_name(),
            num_threads=NUM_THREADS,
            seed=config.seed,
            batch_size=config.batch_size,
            num_steps=config.num_steps,
            warmup_steps=config.warmup_steps,
            avg_step_time_ms=0.0,
            step_time_p50_ms=0.0,
            step_time_p90_ms=0.0,
            step_time_std_ms=0.0,
            samples_per_sec=0.0,
            model_param_mb=0.0,
            grad_mb=0.0,
            optimizer_state_mb=0.0,
            peak_rss_mb=0.0,
            steady_rss_mb=0.0,
            final_loss=float('nan'),
            relative_loss_error=0.0,
            status="Failed",
            error_msg=str(e)
        )

def main():
    results = []
    
    # Configurations
    # Reduced steps for quicker iteration, but enough for stats
    STEPS = 100 
    WARMUP = 20
    BATCH_SIZE = 32
    SEED = 42
    
    models = [
        ("MLP", lambda: MLP(hidden_dim=2048, layers=4), (BATCH_SIZE, 784), (BATCH_SIZE,)),
        ("CNN", lambda: SimpleCNN(), (BATCH_SIZE, 1, 28, 28), (BATCH_SIZE,)),
        ("Transformer", lambda: TinyTransformer(d_model=128, nhead=4, num_layers=2), (BATCH_SIZE, 50), (BATCH_SIZE,))
    ]
    
    optimizers = [
        ("AdamW", optim.AdamW, {"lr": 1e-3}),
        ("SGD+Mom", optim.SGD, {"lr": 1e-2, "momentum": 0.9}),
        ("RMSprop", optim.RMSprop, {"lr": 1e-3, "momentum": 0.9}),
        ("Adagrad", optim.Adagrad, {"lr": 1e-2})
    ]
    
    policies = [
        ("Baseline", lambda: None),
        ("Wrapped(FP32)", lambda: ConfigurablePolicy(codecs_map={}, default_codec=FP32Codec())),
        ("Wrapped(Int8Mom)", lambda: WarmupPolicy(warmup_steps=5)),
        ("Wrapped(MixedFP16)", lambda: ConfigurablePolicy(
            codecs_map={
                'exp_avg': FP16Codec(),       # AdamW momentum
                'momentum_buffer': FP16Codec() # SGD/RMSprop momentum
            }, 
            default_codec=FP32Codec() # Keep variance (exp_avg_sq, square_avg, sum) in FP32
        ))
    ]

    # Store baselines: (model_name, opt_name) -> final_loss
    baselines = {}

    for model_name, model_factory, input_shape, target_shape in models:
        for opt_name, opt_cls, opt_kwargs in optimizers:
            for policy_name, policy_factory in policies:
                
                config = BenchmarkConfig(
                    model_name=model_name,
                    optimizer_name=opt_name,
                    policy_name=policy_name,
                    batch_size=BATCH_SIZE,
                    num_steps=STEPS,
                    warmup_steps=WARMUP,
                    seed=SEED
                )
                
                # Get baseline loss if available
                baseline_loss = baselines.get((model_name, opt_name))
                
                res = run_benchmark(
                    model_factory, input_shape, target_shape,
                    opt_cls, opt_kwargs,
                    policy_factory,
                    config,
                    baseline_loss
                )
                
                # Store baseline if this is the baseline run
                if policy_name == "Baseline" and res.status == "OK":
                    baselines[(model_name, opt_name)] = res.final_loss
                
                results.append(asdict(res))

    # Display
    df = pd.DataFrame(results)
    
    # Reorder columns for readability
    cols = [
        'model', 'optimizer', 'policy', 'status', 
        'avg_step_time_ms', 'step_time_std_ms', 
        'optimizer_state_mb', 'peak_rss_mb', 'steady_rss_mb',
        'final_loss', 'relative_loss_error', 'error_msg'
    ]
    # Add remaining columns
    remaining = [c for c in df.columns if c not in cols]
    df = df[cols + remaining]
    
    print("\nBenchmark Results Summary:")
    print(df[['model', 'optimizer', 'policy', 'avg_step_time_ms', 'optimizer_state_mb', 'peak_rss_mb', 'steady_rss_mb', 'status']].to_string(index=False))
    
    # Save
    df.to_csv("benchmark_results.csv", index=False)

if __name__ == "__main__":
    main()
