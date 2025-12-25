import torch
import time
import copy
from collections import defaultdict
from itertools import chain
from torch.optim import Optimizer
from typing import Optional, Dict, Any, Callable
from .core.state_store import StateStore
from .policy.base import Policy
from .policy.simple import WarmupPolicy

def _auto_pin(param_groups) -> bool:
    """
    Enable pinned CPU storage by default when any parameter lives on CUDA.
    """
    for g in param_groups:
        for p in g["params"]:
            if p.device.type == "cuda":
                return True
    return False

class OptimizerWrapper(Optimizer):
    """
    Wrapper around a PyTorch optimizer that virtualizes its state.
    State is compressed and stored in a StateStore when not in use (i.e. outside of step()).
    """
    def __init__(
        self,
        optimizer: Optimizer,
        policy: Optional[Policy] = None,
        chunk_size: Optional[int] = None,
        chunk_size_on_cuda: Optional[int] = None,
        initial_chunk_size: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        profile: bool = False,
    ):
        self.optimizer = optimizer
        self.policy = policy or WarmupPolicy()
        pin_flag = _auto_pin(self.optimizer.param_groups) if pin_memory is None else pin_memory
        self.store = StateStore(pin_memory=pin_flag)
        self._profile = profile

        self.chunk_size = chunk_size
        self.chunk_size_on_cuda = chunk_size_on_cuda
        self.initial_chunk_size = 1 if initial_chunk_size is None else initial_chunk_size
        self._used_initial_chunk = False
        
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self._global_step = 0
        
        self.last_step_timings = {}
        
        self._update_param_mapping()
        
        if self.optimizer.state:
            for param, state in self.optimizer.state.items():
                step = state.get('step', 0)
                if torch.is_tensor(step):
                     step = step.item()
                
                codecs = self.policy.get_codecs(param, state, step)
                pid = self.param_to_id[param]
                self.store.commit(pid, state, codecs)
            
            self.optimizer.state.clear()

    def _update_param_mapping(self):
        """
        Updates the mapping from parameter objects to stable IDs.
        IDs are assigned sequentially based on group order and parameter order within groups.
        """
        self.param_to_id = {}
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                self.param_to_id[p] = idx
                idx += 1

    def add_param_group(self, param_group: Dict[str, Any]):
        self.optimizer.add_param_group(param_group)
        self._update_param_mapping()

    @property
    def state(self):
        return self.optimizer.state

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            raise NotImplementedError("Chunked stepping with closure is not supported.")

        effective_chunk = None
        if self.initial_chunk_size is not None and not self._used_initial_chunk:
            effective_chunk = self.initial_chunk_size
            self._used_initial_chunk = True
        elif self.chunk_size is not None:
            effective_chunk = self.chunk_size
        elif self.chunk_size_on_cuda is not None and _auto_pin(self.optimizer.param_groups):
            effective_chunk = self.chunk_size_on_cuda
        else:
            total_params = len(self.param_to_id)
            effective_chunk = max(1, min(8, total_params))

        return self._step_chunked(effective_chunk)

    def _step_chunked(self, chunk_size: int):
        """
        Performs the step in chunks to minimize peak memory usage.
        This is only possible if no closure is provided.
        """
        total_materialize = 0.0
        total_step = 0.0
        total_commit = 0.0
        profile_stats = self._init_profile_stats() if self._profile else None
        
        all_original_params = [g['params'] for g in self.optimizer.param_groups]
        
        for g in self.optimizer.param_groups:
            g['params'] = []
            
        for group_idx, group in enumerate(self.optimizer.param_groups):
            original_params = all_original_params[group_idx]
            
            for i in range(0, len(original_params), chunk_size):
                chunk_params = original_params[i : i + chunk_size]
                
                t0 = time.perf_counter()
                
                chunk_pids = [self.param_to_id[p] for p in chunk_params]
                chunk_devices = [p.device for p in chunk_params]
                states = self.store.materialize_batch(chunk_pids, chunk_devices, stats=profile_stats)
                
                for param, state in zip(chunk_params, states):
                    if state:
                        self.optimizer.state[param] = state
                
                t1 = time.perf_counter()
                total_materialize += (t1 - t0)
                
                group['params'] = chunk_params
                
                self.optimizer.step()
                
                t2 = time.perf_counter()
                total_step += (t2 - t1)
                
                for p in chunk_params:
                    if p in self.optimizer.state:
                        state = self.optimizer.state[p]
                        step = state.get('step', self._global_step)
                        if torch.is_tensor(step):
                            step = step.item()
                        codecs = self.policy.get_codecs(p, state, step)
                        self.store.commit(self.param_to_id[p], state, codecs, stats=profile_stats)
                self.optimizer.state.clear()
                
                t3 = time.perf_counter()
                total_commit += (t3 - t2)
                
            group['params'] = []

        for group_idx, group in enumerate(self.optimizer.param_groups):
            group['params'] = all_original_params[group_idx]
            
        self._global_step += 1
        
        timings = {
            'materialize': total_materialize,
            'step': total_step,
            'commit': total_commit,
            'overhead': 0.0 
        }
        if profile_stats is not None:
            encode = profile_stats["encode"]
            decode = profile_stats["decode"]
            timings.update({
                "encode": encode["time_s"],
                "decode": decode["time_s"],
                "encode_tensors": encode["tensors"],
                "decode_tensors": decode["tensors"],
                "encode_bytes": encode["bytes"],
                "decode_bytes": decode["bytes"],
                "encode_elements": encode["elements"],
                "decode_elements": decode["elements"],
                "encode_by_codec": profile_stats["encode_by_codec"],
                "decode_by_codec": profile_stats["decode_by_codec"],
                "materialize_overhead": total_materialize - decode["time_s"],
                "commit_overhead": total_commit - encode["time_s"],
            })
        self.last_step_timings = timings
        
        return None

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        original_state = self.optimizer.state.copy()
        
        cpu_device = torch.device('cpu')
        
        for group in self.param_groups:
            for p in group['params']:
                pid = self.param_to_id[p]
                if pid in self.store._store:
                    self.optimizer.state[p] = self.store.materialize(pid, target_device=cpu_device)
            
        sd = self.optimizer.state_dict()
        sd["optstate_meta"] = {
            "global_step": self._global_step,
            "used_initial_chunk": self._used_initial_chunk,
        }
        
        self.optimizer.state.clear()
        self.optimizer.state.update(original_state)
        
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]):
        state = dict(state_dict)
        meta = state.pop("optstate_meta", None)
        self._load_optimizer_state_dict(state)
        if isinstance(meta, dict):
            self._global_step = meta.get("global_step", self._global_step)
            self._used_initial_chunk = meta.get("used_initial_chunk", self._used_initial_chunk)
        
        for param, state in self.optimizer.state.items():
            step = state.get('step', 0)
            if torch.is_tensor(step):
                step = step.item()
            codecs = self.policy.get_codecs(param, state, step)
            pid = self.param_to_id[param]
            self.store.commit(pid, state, codecs)
            
        self.optimizer.state.clear()

    def _load_optimizer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Custom state_dict loader that preserves optimizer state exactly.
        Avoids the tiny numeric drift seen with Optimizer.load_state_dict on CPU.
        """
        state_dict = state_dict.copy()
        groups = self.optimizer.param_groups
        saved_groups = copy.deepcopy(state_dict["param_groups"])

        if len(groups) != len(saved_groups):
            raise ValueError(
                "loaded state dict has a different number of parameter groups"
            )

        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group "
                "that doesn't match the size of optimizer's group"
            )

        id_map = dict(
            zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        )

        def _cast(param, value, param_id=None, param_groups=None, key=None):
            if isinstance(value, torch.Tensor):
                # Bypass PyTorch's policy logic to ensure bitwise exactness
                # when devices match, avoiding potential casting drift.
                if param.device == value.device and param.dtype == value.dtype:
                    return value.clone()
                return Optimizer._process_value_according_to_param_policy(
                    param, value, param_id, param_groups, key
                )
            if isinstance(value, dict):
                return {
                    k: _cast(
                        param, v, param_id=param_id, param_groups=param_groups, key=k
                    )
                    for k, v in value.items()
                }
            if isinstance(value, (list, tuple)):
                return type(value)(
                    _cast(param, v, param_id=param_id, param_groups=param_groups)
                    for v in value
                )
            return value

        state = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = _cast(
                    param,
                    copy.deepcopy(v),
                    param_id=k,
                    param_groups=state_dict["param_groups"],
                )
            else:
                state[k] = copy.deepcopy(v)

        def update_group(group, new_group):
            new_group["params"] = group["params"]
            if "param_names" in group and "param_names" not in new_group:
                new_group["param_names"] = group["param_names"]
            return new_group

        self.optimizer.param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)
        ]
        self.optimizer.state = state

    def __repr__(self):
        return f"OptimizerWrapper({repr(self.optimizer)})"

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def _init_profile_stats(self) -> Dict[str, Any]:
        return {
            "encode": {"time_s": 0.0, "tensors": 0, "bytes": 0, "elements": 0},
            "decode": {"time_s": 0.0, "tensors": 0, "bytes": 0, "elements": 0},
            "encode_by_codec": {},
            "decode_by_codec": {},
        }

def wrap(
    optimizer: Optimizer,
    policy: Optional[Policy] = None,
    chunk_size: Optional[int] = None,
    chunk_size_on_cuda: Optional[int] = None,
    initial_chunk_size: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    profile: bool = False,
) -> OptimizerWrapper:
    """
    Wraps an existing PyTorch optimizer with state virtualization.
    
    Args:
        optimizer: The optimizer to wrap.
        policy: The policy to use for state compression.
        chunk_size: If set, performs step() in chunks of this size to reduce peak memory.
        chunk_size_on_cuda: If set and chunk_size is None, use this chunk size when parameters are on CUDA.
        initial_chunk_size: Optional smaller chunk size used only for the first step to reduce initial peak (defaults to 1).
        pin_memory: If True, pin CPU-stored compressed tensors to speed GPU transfers. If None (default), enable when any parameter is on CUDA.
        profile: If True, collects encode/decode timing stats in last_step_timings.
    
    Returns:
        An OptimizerWrapper instance.
    """
    return OptimizerWrapper(optimizer, policy, chunk_size, chunk_size_on_cuda, initial_chunk_size, pin_memory, profile)


def auto_wrap(
    optimizer: Optimizer,
    policy: Optional[Policy] = None,
    chunk_size: Optional[int] = None,
    chunk_size_on_cuda: Optional[int] = None,
    initial_chunk_size: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    profile: bool = False,
) -> OptimizerWrapper:
    """
    Plug-and-play helper that applies sensible defaults:
    - Auto-chunking enabled (small first chunk to tame the initial peak).
    - Pinned CPU storage enabled when parameters are on CUDA.
    """
    return OptimizerWrapper(
        optimizer,
        policy=policy,
        chunk_size=chunk_size,
        chunk_size_on_cuda=chunk_size_on_cuda,
        initial_chunk_size=initial_chunk_size,
        pin_memory=pin_memory,
        profile=profile,
    )
