# torch-optstate

**Optimizer State Virtualization for PyTorch**

`torch-optstate` is a core infrastructure library that wraps existing PyTorch optimizers (like Adam, AdamW, SGD) to virtualize their state. It enables significant memory savings by compressing optimizer states (e.g., momentum) when they are not in use, and seamlessly materializing them on-the-fly during optimization steps.

## ❓ Why use this?

Training large models is often memory-bound. While parameters and gradients take up space, **optimizer states** (like momentum and variance in Adam) can consume **2x to 3x** the memory of the model parameters themselves.

`torch-optstate` solves this by:
1.  **Reducing Memory Footprint**: Compresses optimizer state by **25% to 75%** with minimal accuracy loss.
2.  **Drop-in Compatibility**: Works with your existing `torch.optim` optimizers and training loops.
3.  **Optimizer Offloading**: Stores compressed state in CPU RAM and materializes it on GPU VRAM only when needed. This effectively gives you "infinite VRAM" for optimizer states.
4.  **Policy-Driven**: You control the trade-off between precision, speed, and memory.

### Benchmark Highlights (CPU)

| Model | Optimizer | Policy | State Size | Reduction | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP** | AdamW | Baseline | 76.45 MB | - | OK |
| | | **Int8 Momentum** | **47.78 MB** | **37.5%** | OK |
| | | **Mixed FP16** | **57.34 MB** | **25.0%** | OK |
| **SGD** | SGD+Mom | Baseline | 38.23 MB | - | OK |
| | | **Int8 Momentum** | **9.56 MB** | **75.0%** | OK |
| **Transformer** | AdamW | Baseline | 2.60 MB | - | OK |
| | | **Int8 Momentum** | **1.62 MB** | **37.5%** | OK |

*Benchmarks run on CPU. Int8 Momentum policy keeps variance in FP32 for stability.*

## 📦 Installation

```bash
pip install torch-optstate
```

(Note: This package is currently a research preview.)

## 🛠️ Usage

### 1. Basic Usage (Drop-in)

Simply wrap your existing optimizer. By default, it uses a `WarmupPolicy` that keeps state in FP32 for a few steps before compressing momentum to INT8.

```python
import torch
from torch.optim import AdamW
from torch_optstate import wrap

model = torch.nn.Linear(10, 1)
optimizer = AdamW(model.parameters(), lr=1e-3)

# Wrap the optimizer
# This will automatically manage state compression
optimizer = wrap(optimizer)

# Training loop (standard PyTorch)
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

### 2. Custom Policies

You can define custom policies to control compression behavior.

**Int8 Momentum (Aggressive Compression):**
```python
from torch_optstate import wrap, WarmupPolicy

# Keep in FP32 for 1000 steps, then compress momentum to INT8
# Variance (if present) stays in FP32 for stability.
policy = WarmupPolicy(warmup_steps=1000)
optimizer = wrap(optimizer, policy=policy)
```

**Mixed Precision (FP16 Momentum, FP32 Variance):**
```python
from torch_optstate import wrap, ConfigurablePolicy, FP16Codec, FP32Codec

# Define a policy that stores 'exp_avg' in FP16 and 'exp_avg_sq' in FP32
policy = ConfigurablePolicy(
    codecs_map={
        'exp_avg': FP16Codec(),
        'momentum_buffer': FP16Codec()
    },
    default_codec=FP32Codec() # Fallback for variance
)
optimizer = wrap(optimizer, policy=policy)
```

### 3. Advanced Configuration

You can combine warmup with custom codecs using `ConfigurablePolicy`.

```python
from torch_optstate import wrap, ConfigurablePolicy, FP16Codec, FP32Codec

# Warmup for 100 steps, then switch to FP16 for momentum
policy = ConfigurablePolicy(
    codecs_map={'exp_avg': FP16Codec()},
    default_codec=FP32Codec(),
    warmup_steps=100
)
optimizer = wrap(optimizer, policy=policy)
```

### 4. GPU Offloading (Infinite VRAM)

`torch-optstate` automatically handles device placement. If your model is on GPU, the optimizer state will be stored in compressed form on CPU RAM, and materialized to GPU VRAM only during the `step()`.

```python
# Standard PyTorch GPU setup
model = MyModel().cuda()
optimizer = wrap(AdamW(model.parameters()))

# The wrapper automatically:
# 1. Loads compressed state from CPU RAM
# 2. Moves to GPU VRAM
# 3. Decompresses to FP32 on GPU
# 4. Runs optimizer step on GPU
# 5. Compresses on GPU
# 6. Moves back to CPU RAM
optimizer.step()
```

## 🧠 How It Works

1.  **Virtualization**: The `OptimizerWrapper` intercepts `step()` calls.
2.  **Materialization**: Before the inner optimizer runs, compressed state is decoded to full precision (FP32).
3.  **Execution**: The inner optimizer performs the update using standard PyTorch kernels.
4.  **Commit**: After the update, the new state is compressed (e.g., quantized) and stored in the `StateStore`, and the full-precision state is freed.

## ⚠️ Limitations

- **Step Overhead**: Decompression/Compression adds overhead to the `step()` call. This is often negligible compared to the forward/backward pass of large models.
- **Optimizer Support**: Tested primarily with AdamW and SGD. Other optimizers should work but may require custom policies if they use non-standard state keys.

## 🤝 Contributing

Contributions are welcome! Please check the `tests/` folder for coverage requirements.

## 📄 License

MIT
