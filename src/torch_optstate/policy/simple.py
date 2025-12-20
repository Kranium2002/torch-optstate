from typing import Dict, Any, Optional
import torch
from .base import Policy
from ..codecs import Codec, FP32Codec, Int8MomentumCodec, FP16Codec

class WarmupPolicy(Policy):
    """
    Policy that keeps state in FP32 for a warmup period, then compresses.
    
    Args:
        warmup_steps: Number of steps to keep in FP32.
        momentum_key: Key for momentum state (default 'exp_avg').
        variance_key: Key for variance state (default 'exp_avg_sq').
        variance_codec: Codec for variance state after warmup (default: FP32).
        min_int8_elements: Minimum tensor size to use int8; smaller tensors use small_tensor_codec.
        small_tensor_codec: Codec for tensors smaller than min_int8_elements (default: FP32).
    """
    def __init__(
        self,
        warmup_steps: int = 100,
        momentum_key: str = 'exp_avg',
        variance_key: str = 'exp_avg_sq',
        variance_codec: Optional[Codec] = None,
        min_int8_elements: int = 4096,
        small_tensor_codec: Optional[Codec] = None,
    ):
        self.warmup_steps = warmup_steps
        self.momentum_key = momentum_key
        self.variance_key = variance_key
        self.min_int8_elements = min_int8_elements
        
        self.fp32_codec = FP32Codec()
        self.int8_codec = Int8MomentumCodec()
        self.fp16_codec = FP16Codec()
        self.variance_codec = variance_codec or self.fp32_codec
        self.small_tensor_codec = small_tensor_codec or self.fp32_codec

    def _select_codec(self, tensor: torch.Tensor, preferred_codec: Codec) -> Codec:
        if (
            self.min_int8_elements
            and isinstance(preferred_codec, Int8MomentumCodec)
            and tensor.numel() < self.min_int8_elements
        ):
            return self.small_tensor_codec
        return preferred_codec

    def get_codecs(self, param: torch.Tensor, state: Dict[str, Any], step: int) -> Dict[str, Codec]:
        codecs = {}
        
        # Default to FP32 for everything initially
        for key in state:
            if torch.is_tensor(state[key]):
                codecs[key] = self.fp32_codec

        if step >= self.warmup_steps:
            if self.momentum_key in state:
                tensor = state[self.momentum_key]
                if torch.is_tensor(tensor):
                    codecs[self.momentum_key] = self._select_codec(tensor, self.int8_codec)
            
            # Also check for 'momentum_buffer' which is used by SGD
            if 'momentum_buffer' in state:
                tensor = state['momentum_buffer']
                if torch.is_tensor(tensor):
                    codecs['momentum_buffer'] = self._select_codec(tensor, self.int8_codec)
            
            if self.variance_key in state:
                tensor = state[self.variance_key]
                if torch.is_tensor(tensor):
                    codecs[self.variance_key] = self._select_codec(tensor, self.variance_codec)
        
        return codecs
