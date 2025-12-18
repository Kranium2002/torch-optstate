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
) -> WarmupPolicy:
    """
    Build a WarmupPolicy aimed at low-memory use:
    - Momentum is always INT8.
    - Variance codec can be FP16, INT8, or FP32 based on variance_mode.
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
        initial_chunk_size: Optional smaller chunk size used only for the first step (defaults to 1 for lower first-step peak).
        pin_memory: Pin CPU compressed state to accelerate GPU transfers. Defaults to True when any parameter is on CUDA.
        **adamw_kwargs: Passed through to torch.optim.AdamW.
    """
    base_opt = AdamW(params, lr=lr, weight_decay=weight_decay, **adamw_kwargs)
    policy = make_low_memory_policy(
        warmup_steps=warmup_steps,
        variance_mode=variance_mode,
        momentum_key="exp_avg",
        variance_key="exp_avg_sq",
    )
    return wrap(base_opt, policy=policy, chunk_size=chunk_size, initial_chunk_size=initial_chunk_size, pin_memory=pin_memory)
