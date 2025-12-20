from typing import Dict, Any, Optional
import torch
from .base import Policy
from ..codecs import Codec, FP32Codec, Int8MomentumCodec, FP16Codec, BF16Codec, IdentityCodec

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
        device_resident: If True, keep compressed state on the parameter device (e.g., CUDA).
            If False, force CPU offload. If None, default to CPU offload to minimize VRAM.
    """
    def __init__(
        self,
        warmup_steps: int = 100,
        momentum_key: str = 'exp_avg',
        variance_key: str = 'exp_avg_sq',
        variance_codec: Optional[Codec] = None,
        min_int8_elements: int = 4096,
        small_tensor_codec: Optional[Codec] = None,
        device_resident: Optional[bool] = None,
    ):
        self.warmup_steps = warmup_steps
        self.momentum_key = momentum_key
        self.variance_key = variance_key
        self.min_int8_elements = min_int8_elements
        self.device_resident = device_resident
        
        # CPU (offload) codecs
        self._fp32_cpu = FP32Codec()
        self._int8_cpu = Int8MomentumCodec()
        self._fp16_cpu = FP16Codec()
        self._bf16_cpu = BF16Codec()
        # Device-resident codecs
        self._fp32_gpu = IdentityCodec()
        self._int8_gpu = Int8MomentumCodec(offload_to_cpu=False)
        self._fp16_gpu = FP16Codec(offload_to_cpu=False)
        self._bf16_gpu = BF16Codec(offload_to_cpu=False)

        self._variance_kind, self._variance_custom = self._kind_for_codec(variance_codec)
        self._small_kind, self._small_custom = self._kind_for_codec(small_tensor_codec)

        self.fp32_codec = self._fp32_cpu
        self.int8_codec = self._int8_cpu
        self.fp16_codec = self._fp16_cpu
        self.variance_codec = self._resolve_variance_codec()
        self.small_tensor_codec = self._resolve_small_tensor_codec()

    def _use_device_resident(self, device: torch.device) -> bool:
        if self.device_resident is None:
            return False
        return self.device_resident

    def _kind_for_codec(self, codec: Optional[Codec]) -> tuple[str, Optional[Codec]]:
        if codec is None:
            return "fp32", None
        if isinstance(codec, Int8MomentumCodec):
            if getattr(codec, "_offload_to_cpu", True) is False:
                return "custom", codec
            return "int8", None
        if isinstance(codec, FP16Codec):
            if getattr(codec, "_offload_to_cpu", True) is False:
                return "custom", codec
            return "fp16", None
        if isinstance(codec, BF16Codec):
            if getattr(codec, "_offload_to_cpu", True) is False:
                return "custom", codec
            return "bf16", None
        if isinstance(codec, (FP32Codec, IdentityCodec)):
            return "fp32", None
        return "custom", codec

    def _codec_for_kind(
        self,
        kind: str,
        device: torch.device,
        custom: Optional[Codec] = None,
        allow_fp32_device: bool = False,
    ) -> Codec:
        if kind == "custom":
            if custom is None:
                return self._fp32_cpu
            return custom
        use_device = self._use_device_resident(device)
        if kind == "int8":
            return self._int8_gpu if use_device else self._int8_cpu
        if kind == "fp16":
            return self._fp16_gpu if use_device else self._fp16_cpu
        if kind == "bf16":
            return self._bf16_gpu if use_device else self._bf16_cpu
        # fp32
        if allow_fp32_device and use_device:
            return self._fp32_gpu
        return self._fp32_cpu

    def _resolve_variance_codec(self) -> Codec:
        return self._codec_for_kind(self._variance_kind, torch.device("cpu"), self._variance_custom)

    def _resolve_small_tensor_codec(self) -> Codec:
        return self._codec_for_kind(self._small_kind, torch.device("cpu"), self._small_custom)

    def _select_kind(self, tensor: torch.Tensor, preferred_kind: str) -> str:
        if (
            self.min_int8_elements
            and preferred_kind == "int8"
            and tensor.numel() < self.min_int8_elements
        ):
            return self._small_kind
        return preferred_kind

    def get_codecs(self, param: torch.Tensor, state: Dict[str, Any], step: int) -> Dict[str, Codec]:
        codecs = {}
        
        # Default to FP32 for everything initially
        for key in state:
            if torch.is_tensor(state[key]):
                codecs[key] = self._codec_for_kind("fp32", param.device, allow_fp32_device=False)

        if step >= self.warmup_steps:
            if self.momentum_key in state:
                tensor = state[self.momentum_key]
                if torch.is_tensor(tensor):
                    kind = self._select_kind(tensor, "int8")
                    codecs[self.momentum_key] = self._codec_for_kind(
                        kind,
                        param.device,
                        self._small_custom,
                        allow_fp32_device=True,
                    )
            
            # Also check for 'momentum_buffer' which is used by SGD
            if 'momentum_buffer' in state:
                tensor = state['momentum_buffer']
                if torch.is_tensor(tensor):
                    kind = self._select_kind(tensor, "int8")
                    codecs['momentum_buffer'] = self._codec_for_kind(
                        kind,
                        param.device,
                        self._small_custom,
                        allow_fp32_device=True,
                    )
            
            if self.variance_key in state:
                tensor = state[self.variance_key]
                if torch.is_tensor(tensor):
                    kind = self._variance_kind
                    if kind == "int8":
                        kind = self._select_kind(tensor, kind)
                    codecs[self.variance_key] = self._codec_for_kind(
                        kind,
                        param.device,
                        self._variance_custom,
                        allow_fp32_device=False,
                    )
        
        return codecs
