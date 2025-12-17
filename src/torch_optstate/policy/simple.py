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
    """
    def __init__(self, warmup_steps: int = 100, momentum_key: str = 'exp_avg', variance_key: str = 'exp_avg_sq', variance_codec: Optional[Codec] = None):
        self.warmup_steps = warmup_steps
        self.momentum_key = momentum_key
        self.variance_key = variance_key
        
        self.fp32_codec = FP32Codec()
        self.int8_codec = Int8MomentumCodec()
        self.fp16_codec = FP16Codec()
        self.variance_codec = variance_codec or self.fp32_codec

    def get_codecs(self, param: torch.Tensor, state: Dict[str, Any], step: int) -> Dict[str, Codec]:
        codecs = {}
        
        # Default to FP32 for everything initially
        for key in state:
            if torch.is_tensor(state[key]):
                codecs[key] = self.fp32_codec

        if step >= self.warmup_steps:
            if self.momentum_key in state:
                codecs[self.momentum_key] = self.int8_codec
            
            # Also check for 'momentum_buffer' which is used by SGD
            if 'momentum_buffer' in state:
                codecs['momentum_buffer'] = self.int8_codec
            
            if self.variance_key in state:
                codecs[self.variance_key] = self.variance_codec
        
        return codecs
