# torch-optstate

**Optimizer State Virtualization for PyTorch**

torch-optstate wraps existing PyTorch optimizers (Adam/AdamW/SGD) to virtualize their state. It compresses and offloads optimizer state when not in use, then materializes it on-the-fly during `step()`, with chunked execution to keep peaks low. New defaults make it plug-and-play on CPU and CUDA.

## Why use this?

- Optimizer state often costs 2–3× model params. Saving there unlocks larger batches/models.
- Compress and offload momentum/variance to CPU while keeping training unchanged.
- Chunked step avoids double-residency spikes; pinned CPU offload keeps VRAM low.
- Policy-driven: choose FP32/FP16/INT8 per-state with warmup or adaptive triggers.

## What’s new (plug-and-play defaults)
1) Auto-chunked step: always chunked; default chunk ≤8, first chunk = 1 to tame peaks.  
2) CUDA-aware pinned offload: compressed state is pinned on CPU automatically when params are on CUDA.  
3) `auto_wrap` helper: one-call wrapping with the defaults above.  
4) Low-memory AdamW helper: `wrap_low_memory_adamw` defaults to tiny first chunk + auto pin.  
5) Adaptive warmup policy (optional): switch to compression when loss plateaus.  
6) Decode scratch cache: reuses decode buffers to reduce per-step allocations.  
7) Chunk-only path: closures are not supported; keeps peak usage low.
8) Small-tensor bypass: int8 compression skips tiny tensors by default to reduce overhead (configurable via `WarmupPolicy`).
9) CUDA fast path: device-resident compression auto-enables on CUDA; `auto_wrap` uses FP16 variance for better speed/memory balance.
10) Max-compression preset: `wrap_max_compression_adamw` for int8-all state with GPU-friendly chunking.

## Installation
```bash
pip install torch-optstate
```
(Research preview.)

## Usage

### 1) Drop-in (auto chunk + auto pin on CUDA)
```python
import torch
from torch.optim import AdamW
import torch_optstate as topt

model = torch.nn.Linear(10, 1).to("cuda")  # or cpu
opt = AdamW(model.parameters(), lr=1e-3)

# One call: auto chunking, tiny first chunk, auto pin if on CUDA
opt = topt.auto_wrap(opt)
```

### 2) Low peak memory AdamW preset
```python
import torch_optstate as topt

opt = topt.wrap_low_memory_adamw(
    model.parameters(),
    variance_mode="int8",   # or "fp16"/"fp32"
    chunk_size=None,        # auto small chunk
    initial_chunk_size=None # defaults to 1
    # pin_memory None -> auto on CUDA
)
```

### 3) Max compression (int8-all, GPU-friendly)
```python
import torch_optstate as topt

opt = topt.wrap_max_compression_adamw(
    model.parameters(),
    chunk_size_on_cuda=256,  # defaults to 256 if chunk_size is None
    initial_chunk_size=1
)
```

### 4) Custom policies (int8 / FP16 / BF16)
```python
from torch_optstate import wrap, WarmupPolicy, FP16Codec, FP32Codec, Int8MomentumCodec

policy = WarmupPolicy(
    warmup_steps=100,
    momentum_key="exp_avg",
    variance_key="exp_avg_sq",
    variance_codec=Int8MomentumCodec(),  # int8 variance
    # min_int8_elements=4096,  # default: skip int8 for tiny tensors
    # device_resident=False,   # force CPU offload even on CUDA
)
opt = wrap(opt, policy=policy, chunk_size=8, initial_chunk_size=1)
```

### 5) GPU offload (pinned CPU) and chunking
- Offload is automatic; pinning is automatic on CUDA (override with `pin_memory`).
- Chunked step is always on; defaults are small to reduce VRAM overlap.

Example CLI (demo) for GPU:
```bash
poetry run python examples/finetune_demo.py \
  --steps 10 \
  --small_llm \
  --max_compression_fast \
  --metrics_csv gpu_metrics.csv
```

## How it works
1. Virtualization: `OptimizerWrapper` intercepts `step()`.  
2. Materialize: compressed state is decoded to full precision for the chunk.  
3. Execute: inner optimizer runs normally.  
4. Commit: updated state is compressed and offloaded; FP32 copies freed.

## Limitations
- Closures are not supported (step is chunked-only).
- Step overhead exists from compress/decompress; usually small vs. forward/backward.
- Tested mainly on AdamW/SGD; other optimizers may need custom policies.

## License
MIT
