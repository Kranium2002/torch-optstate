import torch
from .base import Codec
from typing import Any

class FP32Codec(Codec):
    """
    Baseline codec that stores tensors in FP32 (no compression).
    """
    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float()

    def decode(self, packed: torch.Tensor) -> torch.Tensor:
        return packed

    def bytes(self, packed: torch.Tensor) -> int:
        return packed.element_size() * packed.numel()
