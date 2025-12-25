# torch-optstate

**Optimizer State Virtualization for PyTorch**

torch-optstate wraps existing PyTorch optimizers (Adam/AdamW/SGD) to virtualize their state. It compresses and offloads optimizer state when not in use, then materializes it on-the-fly during `step()`, with chunked execution to keep peaks low. New defaults make it plug-and-play on CPU and CUDA.

## Why use this?

- Optimizer state often costs 2–3× model params. Saving there unlocks larger batches/models.
- Compress and offload momentum/variance to CPU while keeping training unchanged (default minimizes VRAM).
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
8) Small-tensor bypass: int8 compression skips tiny tensors by default (configurable via `WarmupPolicy`).
9) CUDA path: compressed state is offloaded to CPU by default to minimize VRAM; set `device_resident=True` to keep it on GPU.
10) Max-compression preset: `wrap_max_compression_adamw` for int8-all state with GPU-friendly chunking.

## Installation
```bash
pip install torch-optstate
```
Optional (Linux): `pip install torch-optstate[triton]` to enable `torch.compile` acceleration.

(Research preview.)

## Usage

### 1) Drop-in (recommended default)
```python
import torch
from torch.optim import AdamW
import torch_optstate as topt

model = torch.nn.Linear(10, 1).to("cuda")  # or cpu
opt = AdamW(model.parameters(), lr=1e-3)

# One call: auto chunking, tiny first chunk, auto pin if on CUDA
# Default policy after warmup: int8 momentum + fp32 variance, offloaded to CPU.
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
    # device_resident=True, # keep compressed state on GPU instead of CPU offload
)
```

### 3) Max compression (int8-all, GPU-friendly)
```python
import torch_optstate as topt

opt = topt.wrap_max_compression_adamw(
    model.parameters(),
    chunk_size_on_cuda=256,  # defaults to 256 if chunk_size is None
    initial_chunk_size=1
    # device_resident=True, # keep compressed state on GPU instead of CPU offload
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
- Offload is default (including after compression); pinning is automatic on CUDA (override with `pin_memory=False`).
- Set `device_resident=True` if you want compressed state to stay on GPU instead.
- Chunked step is always on; defaults are small to reduce VRAM overlap.

Example CLI (demo) for GPU:
```bash
poetry run python -m examples.finetune_demo \
  --steps 10 \
  --small_llm \
  --compression_mode default \
  --metrics_csv gpu_metrics.csv
```
Generates `memory_comparison.png` with GPU VRAM and CPU RAM traces.

## Default benchmark (example)
From `gpu_metrics.csv` (10 steps, `small_11m`, default compression):
- Peak GPU allocated (`gpu_mem_mb`): 2074.91 MB -> 1116.60 MB (-958.32 MB, -46.2%)
- Peak GPU peak (`gpu_peak_mb`): 4715.73 MB -> 3755.91 MB (-959.82 MB, -20.4%)
- Peak CPU RSS (`cpu_mem_mb`): 1698.05 MB -> 2582.95 MB (+884.90 MB, +52.1%)
- Peak tensor state (`tensor_mem_mb`): 1023.24 MB -> 640.89 MB (-382.35 MB, -37.4%)

These numbers reflect the expected trade-off: GPU memory drops while CPU memory rises due to offload.

## Metrics glossary (CSV)
- `run`: `baseline` or `optstate`.
- `step`: Step index (1-based).
- `loss`: Training loss for the step.
- `accuracy`: Running training accuracy up to that step.
- `val_accuracy`: Validation accuracy (filled after eval).
- `step_time_ms`: Total wall-clock time per step.
- `tensor_mem_mb`: Estimated optimizer state size (compressed + uncompressed) in MB.
- `materialize_ms`: Time to decode and load optimizer state for the step.
- `inner_step_ms`: Time spent inside the wrapped optimizer `step()`.
- `commit_ms`: Time to compress and store optimizer state after the step.
- `overhead_ms`: Extra time in the wrapper beyond materialize/step/commit.
- `cpu_mem_mb`: Process RSS in MB.
- `gpu_mem_mb`: Current allocated VRAM (post-step sample) in MB.
- `gpu_peak_mb`: Peak allocated VRAM since last reset in MB.
- `compression_active`: Whether compression is active for the step.

## How it works
1. Virtualization: `OptimizerWrapper` intercepts `step()`.  
2. Materialize: compressed state is decoded to full precision for the chunk.  
3. Execute: inner optimizer runs normally.  
4. Commit: updated state is compressed and offloaded; FP32 copies freed.

## Limitations
- Closures are not supported (step is chunked-only).
- Tested mainly on AdamW/SGD; other optimizers may need custom policies.

Future work: speed improvements are planned in upcoming releases.

## License
MIT
