from typing import Dict, Any, Optional
import torch
from .base import Policy
from ..codecs import Codec, FP32Codec

class ConfigurablePolicy(Policy):
    """
    Policy that allows specifying codecs for specific state keys.
    
    Args:
        codecs_map: Dictionary mapping state keys (e.g., 'exp_avg') to Codec instances.
        default_codec: Codec to use for keys not in codecs_map (default: FP32Codec).
        warmup_steps: Number of steps to keep in FP32 before applying compression.
    """
    def __init__(self, codecs_map: Dict[str, Codec], default_codec: Optional[Codec] = None, warmup_steps: int = 0):
        self.codecs_map = codecs_map
        self.default_codec = default_codec or FP32Codec()
        self.warmup_steps = warmup_steps
        self.fp32_codec = FP32Codec()

    def get_codecs(self, param: torch.Tensor, state: Dict[str, Any], step: int) -> Dict[str, Codec]:
        codecs = {}
        
        # During warmup, use FP32 for everything
        if step < self.warmup_steps:
            for key in state:
                if torch.is_tensor(state[key]):
                    codecs[key] = self.fp32_codec
            return codecs

        # After warmup, use configured codecs
        for key in state:
            if torch.is_tensor(state[key]):
                codecs[key] = self.codecs_map.get(key, self.default_codec)
                
        return codecs
