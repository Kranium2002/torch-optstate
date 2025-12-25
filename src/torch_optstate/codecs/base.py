from abc import ABC, abstractmethod
from typing import Any, Tuple
import torch

class Codec(ABC):
    """
    Abstract base class for optimizer state compression codecs.
    """

    @abstractmethod
    def encode(self, tensor: torch.Tensor) -> Any:
        """
        Compresses a tensor into a packed representation.
        """
        pass

    @abstractmethod
    def decode(self, packed: Any, device: torch.device = None) -> torch.Tensor:
        """
        Decompresses a packed representation back into a tensor.
        """
        pass

    def batch_encode(self, tensors: list[torch.Tensor]) -> list[Any]:
        """
        Compresses a list of tensors. Can be overridden for performance.
        """
        return [self.encode(t) for t in tensors]

    def batch_decode(self, packed_list: list[Any], device: torch.device = None) -> list[torch.Tensor]:
        """
        Decompresses a list of packed representations. Can be overridden for performance.
        """
        return [self.decode(p, device=device) for p in packed_list]

    @abstractmethod
    def bytes(self, packed: Any) -> int:
        """
        Returns the memory usage of the packed representation in bytes.
        """
        pass
