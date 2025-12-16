from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from ..codecs import Codec

class Policy(ABC):
    """
    Abstract base class for state management policies.
    Decides which codecs to use for compression based on training step and state.
    """

    @abstractmethod
    def get_codecs(self, param: torch.Tensor, state: Dict[str, Any], step: int) -> Dict[str, Codec]:
        """
        Returns a mapping of state keys to Codecs for the given parameter and step.
        """
        pass
