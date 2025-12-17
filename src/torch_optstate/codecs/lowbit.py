import torch
from .base import Codec
from typing import Any, Tuple

# Try to get compile
try:
    if hasattr(torch, 'compile'):
        compile_fn = torch.compile
    else:
        def compile_fn(x): return x
except:
    def compile_fn(x): return x

@compile_fn
def _int8_encode(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Calculate scale
    abs_max = tensor.abs().max()
    scale = abs_max / 127.0
    
    # Handle zero case to avoid division by zero or useless scaling
    scale = torch.where(scale == 0, torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype), scale)
    
    # Quantize
    quantized = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
    
    return (quantized, scale)

@compile_fn
def _int8_decode(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return quantized.float() * scale

class FP16Codec(Codec):
    """
    Stores tensors in FP16.
    """
    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.half().cpu()

    def decode(self, packed: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        if device is not None:
            packed = packed.to(device)
        return packed.float()

    def bytes(self, packed: torch.Tensor) -> int:
        return packed.element_size() * packed.numel()

class BF16Codec(Codec):
    """
    Stores tensors in BF16.
    """
    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.bfloat16().cpu()

    def decode(self, packed: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        if device is not None:
            packed = packed.to(device)
        return packed.float()

    def bytes(self, packed: torch.Tensor) -> int:
        return packed.element_size() * packed.numel()

class Int8MomentumCodec(Codec):
    """
    Compresses momentum to INT8 with per-tensor scaling.
    """
    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized, scale = _int8_encode(tensor)
        return (quantized.cpu(), scale.cpu())

    def decode(self, packed: Tuple[torch.Tensor, torch.Tensor], device: torch.device = None) -> torch.Tensor:
        quantized, scale = packed
        if device is not None:
            quantized = quantized.to(device)
            scale = scale.to(device)
        return _int8_decode(quantized, scale)
