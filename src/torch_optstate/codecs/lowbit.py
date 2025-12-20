import torch
import os
from .base import Codec
from typing import Any, Tuple, Callable

def _compile_supported() -> bool:
    if not hasattr(torch, "compile") or os.name == "nt":
        return False
    if os.environ.get("TORCH_COMPILE_DISABLE") in ("1", "true", "True"):
        return False
    try:
        from triton.compiler import compiler as triton_compiler
        if not hasattr(triton_compiler, "triton_key"):
            return False
    except Exception:
        return False
    return True

def _maybe_compile(fn: Callable) -> Callable:
    if not _compile_supported():
        return fn
    compiled = torch.compile(fn)

    def wrapped(*args, **kwargs):
        nonlocal compiled
        if compiled is None:
            return fn(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception:
            compiled = None
            return fn(*args, **kwargs)

    return wrapped

compile_fn = _maybe_compile

@compile_fn
def _int8_encode(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Calculate scale
    abs_max = tensor.abs().max()
    
    # Clamp to avoid inf/nan issues
    if not torch.isfinite(abs_max):
        # Fallback or handle? For now, just clamp to a large number to avoid crash, 
        # but data is already corrupted if we have inf.
        # Ideally we should probably not quantize if inf, but we are inside a compiled function.
        abs_max = torch.nan_to_num(abs_max, nan=0.0, posinf=1e6, neginf=-1e6)

    scale = abs_max / 127.0
    
    # Handle zero case to avoid division by zero or useless scaling
    # Use a scalar float for 1.0 to avoid tensor creation overhead in graph
    scale = torch.where(scale == 0, 1.0, scale)
    
    # Quantize
    quantized = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
    
    # Ensure scale is on CPU and FP32 for stability/storage
    # But we can't move it here if we want to keep this function compilable/device-agnostic?
    # The caller can move it.
    
    return (quantized, scale)

@compile_fn
def _int8_decode(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return quantized.float() * scale

class _TensorCache:
    def __init__(self):
        self._cache = {}

    def get(self, shape, device, dtype):
        # Avoid caching on CPU to prevent extra resident buffers inflating RSS
        if device.type == 'cpu':
            return torch.empty(shape, device=device, dtype=dtype)
        key = (shape, device, dtype)
        buf = self._cache.get(key)
        if buf is None:
            buf = torch.empty(shape, device=device, dtype=dtype)
            self._cache[key] = buf
        return buf

class FP16Codec(Codec):
    """
    Stores tensors in FP16.
    """
    def __init__(self):
        self._decode_cache = _TensorCache()

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.half().cpu()

    def decode(self, packed: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        if device is not None:
            packed = packed.to(device)
        out = self._decode_cache.get(packed.shape, packed.device, torch.float32)
        out.copy_(packed.float())
        return out

    def bytes(self, packed: torch.Tensor) -> int:
        return packed.element_size() * packed.numel()

class BF16Codec(Codec):
    """
    Stores tensors in BF16.
    """
    def __init__(self):
        self._decode_cache = _TensorCache()

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.bfloat16().cpu()

    def decode(self, packed: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        if device is not None:
            packed = packed.to(device)
        out = self._decode_cache.get(packed.shape, packed.device, torch.float32)
        out.copy_(packed.float())
        return out

    def bytes(self, packed: torch.Tensor) -> int:
        return packed.element_size() * packed.numel()

class Int8MomentumCodec(Codec):
    """
    Compresses momentum to INT8 with per-tensor scaling.
    """
    def __init__(self):
        self._decode_cache = _TensorCache()

    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized, scale = _int8_encode(tensor)
        return (quantized.cpu(), scale.cpu())

    def decode(self, packed: Tuple[torch.Tensor, torch.Tensor], device: torch.device = None) -> torch.Tensor:
        quantized, scale = packed
        if device is not None:
            quantized = quantized.to(device)
            scale = scale.to(device)
        out = self._decode_cache.get(quantized.shape, quantized.device, torch.float32)
        out.copy_(_int8_decode(quantized, scale))
        return out

    def bytes(self, packed: Tuple[torch.Tensor, torch.Tensor]) -> int:
        quantized, scale = packed
        return (quantized.element_size() * quantized.numel()) + (scale.element_size() * scale.numel())
