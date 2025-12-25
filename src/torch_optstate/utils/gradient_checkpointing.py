from typing import Sequence
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential


def enable_gradient_checkpointing(model: nn.Module, use_reentrant: bool = True) -> bool:
    """
    Enable gradient/activation checkpointing on a model.

    - If the model exposes `gradient_checkpointing_enable` (e.g., Hugging Face),
      it will be called.
    - Otherwise, no changes are made and False is returned.
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(use_reentrant=use_reentrant)
        except TypeError:
            # Some implementations don't accept the kwarg
            model.gradient_checkpointing_enable()
        return True
    return False


def checkpoint_sequential_modules(
    modules: Sequence[nn.Module],
    segments: int,
    preserve_rng_state: bool = True,
    use_reentrant: bool = True,
):
    """
    Apply torch.utils.checkpoint.checkpoint_sequential to a sequence of modules.

    This is a thin wrapper so downstream code can avoid importing torch.utils.checkpoint
    directly. You still need to call the returned function inside your forward.
    """
    return checkpoint_sequential(
        modules,
        segments=segments,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=use_reentrant,
    )
