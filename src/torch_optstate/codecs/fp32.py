import torch
from .base import Codec
from typing import Any

class FP32Codec(Codec):
    """
    Baseline codec that stores tensors in FP32 (no compression).
    """
    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        # Ensure we store on CPU to save GPU memory
        return tensor.float().cpu()

    def decode(self, packed: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        if device is not None:
            return packed.to(device)
        return packed

    def bytes(self, packed: torch.Tensor) -> int:
        return packed.element_size() * packed.numel()
