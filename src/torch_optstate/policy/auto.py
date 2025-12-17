from typing import Dict, Any, Optional
import torch
from .simple import WarmupPolicy
from ..codecs import Codec


class AdaptiveWarmupPolicy(WarmupPolicy):
    """
    Warmup policy that can switch to compression early based on loss plateau.
    Call update_on_loss(loss) each step; when loss stops improving for `patience`
    steps within `tol`, compression activates regardless of step count.
    """

    def __init__(
        self,
        warmup_steps: int = 100,
        momentum_key: str = "exp_avg",
        variance_key: str = "exp_avg_sq",
        variance_codec: Optional[Codec] = None,
        patience: int = 5,
        tol: float = 1e-3,
    ):
        super().__init__(warmup_steps, momentum_key, variance_key, variance_codec)
        self.patience = patience
        self.tol = tol
        self.best_loss: Optional[float] = None
        self.stale_steps = 0
        self.compression_active = False

    def update_on_loss(self, loss_value: float) -> bool:
        """
        Track loss; returns True if compression was just activated.
        """
        if self.best_loss is None or loss_value < self.best_loss - self.tol:
            self.best_loss = loss_value
            self.stale_steps = 0
            return False

        self.stale_steps += 1
        if not self.compression_active and self.stale_steps >= self.patience:
            self.compression_active = True
            return True
        return False

    def activate_compression(self):
        self.compression_active = True

    def get_codecs(self, param: torch.Tensor, state: Dict[str, Any], step: int) -> Dict[str, Codec]:
        # If compression already active, behave like warmup is over.
        effective_warmup = 0 if self.compression_active else self.warmup_steps
        codecs = {}

        # Default to FP32 for everything initially
        for key in state:
            if torch.is_tensor(state[key]):
                codecs[key] = self.fp32_codec

        if step >= effective_warmup:
            if self.momentum_key in state:
                codecs[self.momentum_key] = self.int8_codec

            if "momentum_buffer" in state:
                codecs["momentum_buffer"] = self.int8_codec

            if self.variance_key in state:
                codecs[self.variance_key] = self.variance_codec

        return codecs
