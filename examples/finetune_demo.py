import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import matplotlib.pyplot as plt
import psutil
import os
import time
import gc
import numpy as np
import argparse
import csv
from typing import List, Dict, Any
import sys

# Add src to path so we can import torch_optstate
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for benchmarks helpers

import torch_optstate
from torch_optstate.policy.simple import WarmupPolicy
from torch_optstate.policy.auto import AdaptiveWarmupPolicy
from torch_optstate.codecs import Int8MomentumCodec, FP32Codec, FP16Codec
from torch_optstate.policy.base import Policy
from torch_optstate.low_memory import wrap_low_memory_adamw, wrap_max_compression_adamw
from torch_optstate.utils import enable_gradient_checkpointing
from benchmarks.models import TinyTransformer

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
            # Record VRAM in MB
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
        item = {}
        for key, val in self.encodings.items():
            if torch.is_tensor(val):
                item[key] = val[idx].clone().detach()
            else:
                item[key] = torch.tensor(val[idx])
        
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_data(tokenizer, num_samples=1000, val_split: float = 0.1, max_length: int = 128, shuffle_seed: int = 42):
    if HAS_DATASETS:
        print("Loading IMDB dataset...")
        try:
            dataset = load_dataset("imdb", split="train").shuffle(seed=shuffle_seed)
            dataset = dataset.select(range(num_samples))
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
    # Ensure inputs are strings to avoid tokenizer type errors
    texts = [str(t) for t in texts]
    labels = [int(l) for l in labels]
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    split_idx = max(1, min(len(labels) - 1, int(len(labels) * (1 - val_split)))) if len(labels) > 1 else len(labels)
    train_ds = SimpleDataset({k: v[:split_idx] for k, v in encodings.items()}, labels[:split_idx])
    val_ds = SimpleDataset({k: v[split_idx:] for k, v in encodings.items()}, labels[split_idx:])
    return train_ds, val_ds

class LargeSyntheticModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=8192, num_layers=8):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ])
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(input_dim, 2)
        
    def forward(self, x, attention_mask=None, labels=None):
        # Ignore mask for synthetic
        for layer in self.layers:
            x = layer(x)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return type('Output', (), {'loss': loss, 'logits': logits})()

def eval_accuracy(model, loader, device, use_large_model=False, use_small_llm: bool = False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if use_large_model or use_small_llm:
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                if use_small_llm:
                    logits = model(inputs)
                else:
                    logits = model(inputs, labels=labels).logits
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids, attention_mask=attention_mask, labels=labels).logits
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    model.train()
    return correct / total if total else 0.0


def train_model(
    optimizer_factory,
    device,
    run_label: str,
    epochs=1,
    batch_size=16,
    max_steps=None,
    use_large_model=False,
    use_small_llm: bool = False,
    small_llm_dim: int = 256,
    small_llm_layers: int = 4,
    val_split: float = 0.1,
    auto_warmup: bool = False,
    auto_patience: int = 5,
    auto_tol: float = 1e-3,
    clip_grad_norm: float = 0.0,
):
    print(f"Initializing model on {device}...")
    
    if use_large_model:
        print("Using Large Synthetic Model (MLP) to demonstrate memory scaling...")
        model = LargeSyntheticModel().to(device)
        tokenizer = None # Not needed
    elif use_small_llm:
        print("Using Tiny Transformer (small LLM) on IMDB to demonstrate compression on sequence models...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = TinyTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=small_llm_dim,
            nhead=4,
            num_layers=small_llm_layers,
            num_classes=2
        ).to(device)
    else:
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        model.to(device)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    model.train()

    if use_large_model:
        # Synthetic data for MLP
        train_samples = 500
        val_samples = max(1, int(train_samples * val_split))
        dataset = SimpleDataset(
            {'input_ids': torch.randn(train_samples, 768)}, # treating input_ids as float features
            [0] * train_samples
        )
        val_dataset = SimpleDataset(
            {'input_ids': torch.randn(val_samples, 768)},
            [0] * val_samples
        )
    else:
        # Ensure we have enough data for the requested steps
        needed_samples = (max_steps * batch_size) if max_steps else 500
        needed_samples += batch_size 
        # TinyTransformer positional encoding supports up to 100 tokens in benchmarks.models
        max_len = 100 if use_small_llm else 128
        dataset, val_dataset = get_data(tokenizer, num_samples=needed_samples, val_split=val_split, max_length=max_len) 
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset is not None else None

    # Create optimizer
    optimizer = optimizer_factory(model.parameters())
    
    # Count parameter tensors
    num_params = sum(1 for _ in model.parameters())
    print(f"Model has {num_params} parameter tensors.")
    
    tracker = MemoryTracker(device)
    run_metrics: List[Dict[str, Any]] = []
    running_correct = 0
    running_total = 0
    
    print("Starting training loop...")
    step = 0
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            outputs = None
            
            if use_large_model or use_small_llm:
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                if use_small_llm:
                    logits = model(inputs)
                    loss = F.cross_entropy(logits, labels)
                else:
                    outputs = model(inputs, labels=labels)
                    logits = outputs.logits
                    loss = outputs.loss
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
            
            # Track cumulative accuracy when logits/labels are available
            preds = logits.argmax(dim=-1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.numel()
            loss.backward()

            if clip_grad_norm and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0_step = time.perf_counter()
            
            optimizer.step()
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1_step = time.perf_counter()
            step_dt = t1_step - t0_step
            
            # Force GC to verify if memory growth is real or just lazy GC
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            tracker.record()
            # Always track CPU RAM (RSS) alongside tensor/VRAM
            cpu_mem = tracker.process.memory_info().rss / 1024 / 1024
            
            # Track tensor memory specifically
            tensor_mem = get_tensor_memory(optimizer)
            
            step += 1
            
            # Check for detailed timings from wrapper
            details = ""
            timings: Dict[str, float] = {}
            if hasattr(optimizer, 'last_step_timings'):
                timings = optimizer.last_step_timings
                details = f" [Mat: {timings.get('materialize',0)*1000:.1f}ms, Step: {timings.get('step',0)*1000:.1f}ms, Cmt: {timings.get('commit',0)*1000:.1f}ms]"
            
            gpu_str = ""
            gpu_mem = None
            gpu_max = None
            if device.type == 'cuda':
                gpu_mem = torch.cuda.memory_allocated(device) / 1024 / 1024
                gpu_max = torch.cuda.max_memory_allocated(device) / 1024 / 1024
                gpu_str = f", GPU: {gpu_mem:.2f} MB (Peak: {gpu_max:.2f} MB)"

            compression_active = False
            if hasattr(optimizer, 'policy'):
                compression_active = getattr(optimizer.policy, 'compression_active', False)
                if not compression_active and hasattr(optimizer.policy, 'warmup_steps'):
                    compression_active = step >= optimizer.policy.warmup_steps

            acc = (running_correct / running_total) if running_total else 0.0
            print(f"Step {step}, Loss: {loss.item():.4f}, Acc: {acc*100:.1f}%, Time: {step_dt*1000:.1f}ms{details}, Tensor Mem: {tensor_mem:.2f} MB, CPU: {cpu_mem:.2f} MB{gpu_str}")
            
            # Record metrics for CSV/export
            run_metrics.append({
                'run': run_label,
                'step': step,
                'loss': loss.item(),
                'accuracy': acc,
                'val_accuracy': None,
                'step_time_ms': step_dt * 1000,
                'tensor_mem_mb': tensor_mem,
                'materialize_ms': timings.get('materialize', 0) * 1000,
                'inner_step_ms': timings.get('step', 0) * 1000,
                'commit_ms': timings.get('commit', 0) * 1000,
                'overhead_ms': timings.get('overhead', 0) * 1000,
                'cpu_mem_mb': cpu_mem,
                'gpu_mem_mb': gpu_mem,
                'gpu_peak_mb': gpu_max,
                'compression_active': compression_active,
            })
                
            # Clean up
            if outputs is not None:
                del outputs
            del loss
            if not use_large_model and not use_small_llm:
                del input_ids, attention_mask, labels
            else:
                del inputs, labels

            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break
            
    val_acc = None
    if 'val_loader' in locals() and val_loader is not None:
        val_acc = eval_accuracy(model, val_loader, device, use_large_model, use_small_llm)
        print(f"Validation Accuracy: {val_acc*100:.2f}%")
        if run_metrics:
            run_metrics[-1]['val_accuracy'] = val_acc
    if device.type == 'cuda':
        print(f"Training Finished. Final GPU: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB, Peak GPU: {torch.cuda.max_memory_allocated(device)/1024**2:.2f} MB")

    times, mem = tracker.get_data()
    return times, mem, run_metrics


def write_metrics_csv(path: str, rows: List[Dict[str, Any]]):
    """
    Save collected per-step metrics to CSV.
    """
    if not rows:
        return
    fieldnames = [
        'run',
        'step',
        'loss',
        'accuracy',
        'val_accuracy',
        'step_time_ms',
        'tensor_mem_mb',
        'materialize_ms',
        'inner_step_ms',
        'commit_ms',
        'overhead_ms',
        'cpu_mem_mb',
        'gpu_mem_mb',
        'gpu_peak_mb',
        'compression_active',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description='Fine-tune demo with memory tracking')
    parser.add_argument('--steps', type=int, default=1, help='Number of steps to run')
    parser.add_argument('--chunk_size', type=int, default=None, help='Chunk size for Torch-OptState (number of tensors). If not set, uses an auto-chosen small chunk.')
    parser.add_argument('--chunk_size_on_cuda', type=int, default=None, help='Chunk size to use on CUDA when --chunk_size is not set.')
    parser.add_argument('--initial_chunk_size', type=int, default=None, help='Optional smaller chunk size for the first step to lower initial peak memory (defaults internally to a tiny chunk).')
    parser.add_argument('--large_model', action='store_true', help='Use a large synthetic model to demonstrate scaling')
    parser.add_argument('--small_llm', action='store_true', help='Use a tiny Transformer (LLM-style) synthetic model')
    parser.add_argument('--small_llm_dim', type=int, default=256, help='Hidden size (d_model) for tiny Transformer LLM mode')
    parser.add_argument('--small_llm_layers', type=int, default=4, help='Number of layers for tiny Transformer LLM mode')
    parser.add_argument('--clip_grad_norm', type=float, default=0.0, help='Apply gradient clipping (L2 norm) if > 0')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help='Optimizer to use')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Number of steps to keep optimizer state in FP32 before switching to int8')
    parser.add_argument(
        '--compression_mode',
        type=str,
        default='default',
        choices=['default', 'fp16_variance', 'int8_variance', 'int8_all'],
        help='default: Int8 momentum, FP32 variance; fp16_variance: Int8 momentum + FP16 variance; int8_variance: Int8 momentum + Int8 variance; int8_all: same as int8_variance'
    )
    parser.add_argument('--min_int8_elements', type=int, default=4096, help='Minimum tensor elements to use int8; smaller tensors use the small-tensor codec.')
    parser.add_argument('--max_compression_fast', action='store_true', help='Force int8 for momentum+variance with no warmup and device-resident state on CUDA.')
    parser.add_argument('--metrics_csv', type=str, default='memory_metrics.csv', help='Path to save per-step metrics CSV')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of data to use for validation')
    parser.add_argument('--auto_warmup', action='store_true', help='Enable adaptive warmup: switch to compression when loss stops improving')
    parser.add_argument('--auto_patience', type=int, default=5, help='Steps without loss improvement before enabling compression when auto_warmup is on')
    parser.add_argument('--auto_tol', type=float, default=1e-3, help='Minimum loss improvement to reset patience when auto_warmup is on')
    parser.add_argument('--pin_memory', action='store_true', help='Pin compressed CPU state to accelerate GPU transfers (otherwise auto-on when using CUDA).')
    parser.add_argument('--device_resident', action='store_true', help='Keep compressed state on device (GPU) for speed; overrides CPU offload.')
    parser.add_argument('--cpu_offload', action='store_true', help='Force CPU offload of compressed state even on CUDA.')
    args = parser.parse_args()
    
    compression_mode = args.compression_mode
    min_int8_elements = args.min_int8_elements
    chunk_size_on_cuda = args.chunk_size_on_cuda

    # Ensure warmup can actually complete within the run for short demos
    effective_warmup = min(args.warmup_steps, max(args.steps - 1, 0))
    if args.max_compression_fast:
        compression_mode = "int8_all"
        min_int8_elements = 0
        effective_warmup = 0
        if chunk_size_on_cuda is None:
            chunk_size_on_cuda = 256
    elif effective_warmup != args.warmup_steps:
        print(f"Clamping warmup_steps from {args.warmup_steps} to {effective_warmup} so compression occurs within {args.steps} steps.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Running for {args.steps} steps with {args.optimizer}")
    if args.chunk_size is not None:
        print(f"Using chunk size: {args.chunk_size}")
    elif chunk_size_on_cuda is not None and device.type == "cuda":
        print(f"Using CUDA chunk size: {chunk_size_on_cuda}")
    else:
        print("Using auto chunk size (small, chunked by default)")
    if args.large_model:
        print("Using Large Synthetic Model")
    if args.small_llm:
        print("Using Tiny Transformer (small LLM)")
    print(f"Warmup steps before int8 compression: {effective_warmup}")
    print(f"Compression mode: {compression_mode}")
    print(f"min_int8_elements: {min_int8_elements}")
    if args.max_compression_fast:
        print("Max compression fast enabled: int8-all, warmup=0.")
    if args.auto_warmup:
        print(f"Auto warmup enabled (patience={args.auto_patience}, tol={args.auto_tol}). Compression will activate when loss plateaus.")
    if args.device_resident and args.cpu_offload:
        print("Both --device_resident and --cpu_offload were set; using device_resident.")

    device_resident = None
    if args.cpu_offload:
        device_resident = False
    elif args.device_resident or args.max_compression_fast:
        device_resident = True

    # 1. Baseline Run
    print(f"\n--- Running Baseline ({args.optimizer.upper()}) ---")
    def baseline_factory(params):
        if args.optimizer == 'sgd':
            return torch.optim.SGD(params, lr=1e-3, momentum=0.9)
        # True baseline: plain AdamW, no wrapping/compression
        return torch.optim.AdamW(params, lr=5e-5, weight_decay=0.01)
    
    # Clear memory before run
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    base_time, base_mem, base_metrics = train_model(
        baseline_factory,
        device,
        run_label='baseline',
        max_steps=args.steps,
        use_large_model=args.large_model,
        use_small_llm=args.small_llm,
        small_llm_dim=args.small_llm_dim,
        small_llm_layers=args.small_llm_layers,
        val_split=args.val_split,
        auto_warmup=False,
        auto_patience=args.auto_patience,
        auto_tol=args.auto_tol,
        clip_grad_norm=args.clip_grad_norm,
    )
    
    # 2. Torch-OptState Run
    print(f"\n--- Running Torch-OptState (Wrapped {args.optimizer.upper()}) ---")
    def optstate_factory(params):
        # Choose variance compression mode
        variance_mode = 'fp32'
        if compression_mode == 'fp16_variance':
            variance_mode = 'fp16'
        elif compression_mode in ['int8_variance', 'int8_all']:
            variance_mode = 'int8'

        if args.optimizer == 'sgd':
            base_opt = torch.optim.SGD(params, lr=1e-3, momentum=0.9)
            policy_cls = AdaptiveWarmupPolicy if args.auto_warmup else WarmupPolicy
            if args.auto_warmup:
                policy = policy_cls(
                    warmup_steps=effective_warmup,
                    momentum_key='momentum_buffer',
                    variance_key='unused',
                    variance_codec=FP32Codec(),
                    min_int8_elements=min_int8_elements,
                    device_resident=device_resident,
                    patience=args.auto_patience,
                    tol=args.auto_tol,
                )
            else:
                policy = policy_cls(
                    warmup_steps=effective_warmup,
                    momentum_key='momentum_buffer',
                    variance_key='unused',
                    variance_codec=FP32Codec(),
                    min_int8_elements=min_int8_elements,
                    device_resident=device_resident,
                )
            return torch_optstate.auto_wrap(
                base_opt,
                policy=policy,
                chunk_size=args.chunk_size,
                chunk_size_on_cuda=chunk_size_on_cuda,
                initial_chunk_size=args.initial_chunk_size,
                pin_memory=args.pin_memory if args.pin_memory else None,
            )

        # AdamW path: use low-memory helper
        if args.max_compression_fast:
            return wrap_max_compression_adamw(
                params,
                lr=5e-5,
                weight_decay=0.01,
                chunk_size=args.chunk_size,
                chunk_size_on_cuda=chunk_size_on_cuda,
                initial_chunk_size=args.initial_chunk_size,
                pin_memory=args.pin_memory if args.pin_memory else None,
                device_resident=device_resident,
            )

        return wrap_low_memory_adamw(
            params,
            lr=5e-5,
            weight_decay=0.01,
            warmup_steps=effective_warmup,
            variance_mode=variance_mode,
            chunk_size=args.chunk_size,
            chunk_size_on_cuda=chunk_size_on_cuda,
            initial_chunk_size=args.initial_chunk_size,
            pin_memory=args.pin_memory if args.pin_memory else None,
            min_int8_elements=min_int8_elements,
            device_resident=device_resident,
        )

    # Clear memory before run
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    opt_time, opt_mem, opt_metrics = train_model(
        optstate_factory,
        device,
        run_label='optstate',
        max_steps=args.steps,
        use_large_model=args.large_model,
        use_small_llm=args.small_llm,
        small_llm_dim=args.small_llm_dim,
        small_llm_layers=args.small_llm_layers,
        val_split=args.val_split,
        auto_warmup=args.auto_warmup,
        auto_patience=args.auto_patience,
        auto_tol=args.auto_tol,
        clip_grad_norm=args.clip_grad_norm,
    )

    # Save combined metrics
    all_metrics = base_metrics + opt_metrics
    write_metrics_csv(args.metrics_csv, all_metrics)
    print(f"\nSaved per-step metrics to {args.metrics_csv}")

    # Plotting
    output_file = 'memory_comparison.png'
    if device.type == 'cuda':
        fig, (ax_gpu, ax_cpu) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        base_gpu = [row['gpu_mem_mb'] for row in base_metrics]
        opt_gpu = [row['gpu_mem_mb'] for row in opt_metrics]
        base_cpu = [row['cpu_mem_mb'] for row in base_metrics]
        opt_cpu = [row['cpu_mem_mb'] for row in opt_metrics]

        ax_gpu.plot(base_time, base_gpu, label='Baseline GPU', alpha=0.7)
        ax_gpu.plot(opt_time, opt_gpu, label='Torch-OptState GPU', alpha=0.7)
        ax_gpu.set_ylabel('GPU VRAM (MB)')
        ax_gpu.set_title(f'Memory Usage during Fine-tuning ({device.type.upper()})')
        ax_gpu.legend()
        ax_gpu.grid(True)

        ax_cpu.plot(base_time, base_cpu, label='Baseline CPU', alpha=0.7)
        ax_cpu.plot(opt_time, opt_cpu, label='Torch-OptState CPU', alpha=0.7)
        ax_cpu.set_xlabel('Time (s)')
        ax_cpu.set_ylabel('CPU RAM (MB)')
        ax_cpu.legend()
        ax_cpu.grid(True)

        fig.tight_layout()
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(base_time, base_mem, label='Baseline (AdamW)', alpha=0.7)
        plt.plot(opt_time, opt_mem, label='Torch-OptState', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Memory (MB)')
        plt.title(f'Memory Usage during Fine-tuning ({device.type.upper()})')
        plt.legend()
        plt.grid(True)

    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")
    
    # Print stats
    print(f"\nBaseline Peak: {max(base_mem):.2f} MB")
    print(f"OptState Peak: {max(opt_mem):.2f} MB")
    print(f"Baseline Final: {base_mem[-1]:.2f} MB")
    print(f"OptState Final: {opt_mem[-1]:.2f} MB")

if __name__ == "__main__":
    main()
