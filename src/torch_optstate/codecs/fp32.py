import torch
from .base import Codec
from typing import Any

class CPUOffloadCodec(Codec):
    """
    Codec that stores tensors in FP32 on CPU.
    This is the default behavior for "FP32" in this library (offloading).
    """
    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        # Avoid a copy when already FP32 on CPU.
        if tensor.device.type == "cpu" and tensor.dtype == torch.float32:
            return tensor.detach()
        # Ensure we store on CPU to save GPU memory
        return tensor.float().cpu()

    def decode(self, packed: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        if device is not None:
            return packed.to(device)
        return packed

    def bytes(self, packed: torch.Tensor) -> int:
        return packed.element_size() * packed.numel()

class IdentityCodec(Codec):
    """
    Codec that does nothing (no compression, no offloading).
    Keeps tensor on original device and dtype.
    """
    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def decode(self, packed: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        if device is not None:
            return packed.to(device)
        return packed

    def bytes(self, packed: torch.Tensor) -> int:
        return packed.element_size() * packed.numel()

# Alias for backward compatibility, but deprecated in spirit
FP32Codec = CPUOffloadCodec
