from typing import Iterable, Optional
import torch
from torch.optim import AdamW, Optimizer
from .wrap import wrap
from .policy import WarmupPolicy
from .codecs import Int8MomentumCodec, FP16Codec, FP32Codec, Codec


def make_low_memory_policy(
    warmup_steps: int = 10,
    variance_mode: str = "fp16",
    momentum_key: str = "exp_avg",
    variance_key: str = "exp_avg_sq",
    min_int8_elements: int = 4096,
    small_tensor_codec: Optional[Codec] = None,
    device_resident: Optional[bool] = None,
) -> WarmupPolicy:
    """
    Build a WarmupPolicy aimed at low-memory use:
    - Momentum defaults to INT8 (small tensors may use small_tensor_codec for speed).
    - Variance codec can be FP16, INT8, or FP32 based on variance_mode (small tensors may use small_tensor_codec).
    """
    if variance_mode == "fp16":
        variance_codec: Codec = FP16Codec()
    elif variance_mode in ("int8", "int8_variance", "int8_all"):
        variance_codec = Int8MomentumCodec()
    else:
        variance_codec = FP32Codec()

    return WarmupPolicy(
        warmup_steps=warmup_steps,
        momentum_key=momentum_key,
        variance_key=variance_key,
        variance_codec=variance_codec,
        min_int8_elements=min_int8_elements,
        small_tensor_codec=small_tensor_codec,
        device_resident=device_resident,
    )


def wrap_low_memory_adamw(
    params: Iterable[torch.nn.Parameter],
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 10,
    variance_mode: str = "fp16",
    chunk_size: Optional[int] = 32,
    initial_chunk_size: Optional[int] = 1,
    pin_memory: Optional[bool] = None,
    min_int8_elements: int = 4096,
    small_tensor_codec: Optional[Codec] = None,
    device_resident: Optional[bool] = None,
    chunk_size_on_cuda: Optional[int] = None,
    **adamw_kwargs,
) -> Optimizer:
    """
    Convenience helper: AdamW wrapped with an aggressive low-memory policy.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        warmup_steps: Steps to keep FP32 before compression.
        variance_mode: 'fp16', 'int8', or 'fp32' for variance state.
        chunk_size: Optional chunk size for optimizer step to reduce peak memory.
        chunk_size_on_cuda: If set and chunk_size is None, use this chunk size when parameters are on CUDA.
        initial_chunk_size: Optional smaller chunk size used only for the first step (defaults to 1 for lower first-step peak).
        pin_memory: Pin CPU compressed state to accelerate GPU transfers. Defaults to True when any parameter is on CUDA.
        min_int8_elements: Minimum elements to use INT8 for momentum/variance; smaller tensors use small_tensor_codec.
        small_tensor_codec: Codec for tensors smaller than min_int8_elements (defaults to FP32).
        device_resident: If True, keep compressed state on the parameter device (e.g., CUDA).
            If False, force CPU offload. If None, default to CPU offload to minimize VRAM.
        **adamw_kwargs: Passed through to torch.optim.AdamW.
    """
    base_opt = AdamW(params, lr=lr, weight_decay=weight_decay, **adamw_kwargs)
    policy = make_low_memory_policy(
        warmup_steps=warmup_steps,
        variance_mode=variance_mode,
        momentum_key="exp_avg",
        variance_key="exp_avg_sq",
        min_int8_elements=min_int8_elements,
        small_tensor_codec=small_tensor_codec,
        device_resident=device_resident,
    )
    return wrap(
        base_opt,
        policy=policy,
        chunk_size=chunk_size,
        chunk_size_on_cuda=chunk_size_on_cuda,
        initial_chunk_size=initial_chunk_size,
        pin_memory=pin_memory,
    )


def wrap_max_compression_adamw(
    params: Iterable[torch.nn.Parameter],
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    chunk_size: Optional[int] = None,
    chunk_size_on_cuda: Optional[int] = 256,
    initial_chunk_size: Optional[int] = 1,
    pin_memory: Optional[bool] = None,
    device_resident: Optional[bool] = None,
    **adamw_kwargs,
) -> Optimizer:
    """
    Convenience helper: maximize compression with GPU-friendly defaults.

    - Int8 momentum + int8 variance, no warmup.
    - min_int8_elements=0 (compress all tensors).
    - Defaults to CPU-offloaded compressed state; set device_resident=True to keep it on GPU.
    - chunk_size_on_cuda defaults to a larger value to keep step overhead reasonable.
    """
    base_opt = AdamW(params, lr=lr, weight_decay=weight_decay, **adamw_kwargs)
    policy = make_low_memory_policy(
        warmup_steps=0,
        variance_mode="int8",
        momentum_key="exp_avg",
        variance_key="exp_avg_sq",
        min_int8_elements=0,
        small_tensor_codec=None,
        device_resident=device_resident,
    )
    return wrap(
        base_opt,
        policy=policy,
        chunk_size=chunk_size,
        chunk_size_on_cuda=chunk_size_on_cuda,
        initial_chunk_size=initial_chunk_size,
        pin_memory=pin_memory,
    )
