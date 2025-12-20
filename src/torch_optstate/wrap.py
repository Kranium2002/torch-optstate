import torch
import time
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
        initial_chunk_size: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        profile: bool = False,
    ):
        self.optimizer = optimizer
        self.policy = policy or WarmupPolicy()
        # Default to pinning when training on CUDA to make offload truly "plug and play".
        pin_flag = _auto_pin(self.optimizer.param_groups) if pin_memory is None else pin_memory
        self.store = StateStore(pin_memory=pin_flag)
        self._profile = profile

        # Default chunking is small-but-safe; if unspecified we will derive a chunk size
        # once param mapping is known.
        self.chunk_size = chunk_size
        # Optionally use a smaller chunk size for the first step to lower the initial peak.
        # If not provided, default to 1 to minimize the first-step peak.
        self.initial_chunk_size = 1 if initial_chunk_size is None else initial_chunk_size
        self._used_initial_chunk = False
        
        # We don't call super().__init__ because we are proxying.
        # But we need to look like an Optimizer.
        # We share param_groups with the underlying optimizer.
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self._global_step = 0
        
        # Performance stats
        self.last_step_timings = {}
        
        # Initialize param mapping
        self._update_param_mapping()
        
        # Initialize state as empty, we will manage it via store
        # But we need to sync with existing state if any
        if self.optimizer.state:
            for param, state in self.optimizer.state.items():
                # Initial commit with default policy (likely FP32 or whatever policy says for step 0)
                # We don't know the step here easily, assume 0 or extract from state
                step = state.get('step', 0)
                if torch.is_tensor(step):
                     step = step.item()
                
                codecs = self.policy.get_codecs(param, state, step)
                pid = self.param_to_id[param]
                self.store.commit(pid, state, codecs)
            
            # Clear original state to save memory
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
        # We return a view that looks like the state, but we shouldn't really expose it 
        # directly for modification outside of step() if we want to keep consistency.
        # However, for debugging/inspection, we might need to materialize.
        # For now, let's return a proxy or just the underlying empty dict if we want to hide it.
        # But PyTorch internals might access it.
        # Let's return the optimizer's state dict, which we populate during step.
        return self.optimizer.state

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            raise NotImplementedError("Chunked stepping with closure is not supported.")

        # Always use chunked path. If no chunk_size provided, choose a small default based on param count.
        effective_chunk = None
        if self.initial_chunk_size is not None and not self._used_initial_chunk:
            effective_chunk = self.initial_chunk_size
            self._used_initial_chunk = True
        elif self.chunk_size is not None:
            effective_chunk = self.chunk_size
        else:
            total_params = len(self.param_to_id)
            # Use a small default to avoid large overlap on GPU; scale gently with param count.
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
        
        # 1. Backup all params
        all_original_params = [g['params'] for g in self.optimizer.param_groups]
        
        # 2. Empty all groups
        for g in self.optimizer.param_groups:
            g['params'] = []
            
        # 3. Iterate and chunk
        for group_idx, group in enumerate(self.optimizer.param_groups):
            original_params = all_original_params[group_idx]
            
            for i in range(0, len(original_params), chunk_size):
                chunk_params = original_params[i : i + chunk_size]
                
                t1 = time.perf_counter()
                
                # Materialize
                chunk_pids = [self.param_to_id[p] for p in chunk_params]
                chunk_devices = [p.device for p in chunk_params]
                states = self.store.materialize_batch(chunk_pids, chunk_devices, stats=profile_stats)
                
                for param, state in zip(chunk_params, states):
                    if state:
                        self.optimizer.state[param] = state
                
                t2 = time.perf_counter()
                total_materialize += (t2 - t1)
                
                # Set params for this group
                group['params'] = chunk_params
                
                # Step
                self.optimizer.step()
                
                t3 = time.perf_counter()
                total_step += (t3 - t2)
                
                # Commit: stream each param to avoid holding FP32 + compressed for many tensors
                for p in chunk_params:
                    if p in self.optimizer.state:
                        state = self.optimizer.state[p]
                        step = state.get('step', self._global_step)
                        if torch.is_tensor(step):
                            step = step.item()
                        codecs = self.policy.get_codecs(p, state, step)
                        self.store.commit(self.param_to_id[p], state, codecs, stats=profile_stats)
                # Clear optimizer state for this chunk to free FP32 tensors
                self.optimizer.state.clear()
                
                t4 = time.perf_counter()
                total_commit += (t4 - t3)
                
            # Restore params for this group (though we empty it again in next loop? No, we empty all at start)
            # We can leave it empty for now and restore all at end.
            group['params'] = []

        # 4. Restore all
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
        # Materialize everything to generate a standard state_dict
        # We need to temporarily populate optimizer.state
        
        # Save current state of optimizer.state (should be empty)
        original_state = self.optimizer.state.copy()
        
        # We need to map IDs back to params to populate optimizer.state
        # But wait, StateStore uses IDs now.
        # And optimizer.state uses params.
        # We need to iterate over our params and materialize.
        
        # Also, we want to materialize on CPU for state_dict to avoid GPU OOM.
        cpu_device = torch.device('cpu')
        
        for group in self.param_groups:
            for p in group['params']:
                pid = self.param_to_id[p]
                if pid in self.store._store:
                    self.optimizer.state[p] = self.store.materialize(pid, target_device=cpu_device)
            
        # Get state dict
        sd = self.optimizer.state_dict()
        
        # Restore (clear)
        self.optimizer.state.clear()
        self.optimizer.state.update(original_state)
        
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # Load into the underlying optimizer to parse params
        # But wait, load_state_dict expects params to match.
        # We can just call optimizer.load_state_dict, then steal the state.
        
        self.optimizer.load_state_dict(state_dict)
        
        # Now move everything to store
        for param, state in self.optimizer.state.items():
            step = state.get('step', 0)
            if torch.is_tensor(step):
                step = step.item()
            codecs = self.policy.get_codecs(param, state, step)
            pid = self.param_to_id[param]
            self.store.commit(pid, state, codecs)
            
        self.optimizer.state.clear()

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

def wrap(optimizer: Optimizer, policy: Optional[Policy] = None, chunk_size: Optional[int] = None, initial_chunk_size: Optional[int] = None, pin_memory: bool = False, profile: bool = False) -> OptimizerWrapper:
    """
    Wraps an existing PyTorch optimizer with state virtualization.
    
    Args:
        optimizer: The optimizer to wrap.
        policy: The policy to use for state compression.
        chunk_size: If set, performs step() in chunks of this size to reduce peak memory.
        initial_chunk_size: Optional smaller chunk size used only for the first step to reduce initial peak (defaults to 1).
        pin_memory: If True, pin CPU-stored compressed tensors to speed GPU transfers. If None, defaults to True when any parameter is on CUDA.
        profile: If True, collects encode/decode timing stats in last_step_timings.
    
    Returns:
        An OptimizerWrapper instance.
    """
    return OptimizerWrapper(optimizer, policy, chunk_size, initial_chunk_size, pin_memory, profile)


def auto_wrap(optimizer: Optimizer, policy: Optional[Policy] = None, chunk_size: Optional[int] = None, initial_chunk_size: Optional[int] = None, pin_memory: Optional[bool] = None, profile: bool = False) -> OptimizerWrapper:
    """
    Plug-and-play helper that applies sensible defaults:
    - Auto-chunking enabled (small first chunk to tame the initial peak).
    - Pinned CPU storage enabled when parameters are on CUDA.
    """
    return OptimizerWrapper(
        optimizer,
        policy=policy,
        chunk_size=chunk_size,
        initial_chunk_size=initial_chunk_size,
        pin_memory=pin_memory,
        profile=profile,
    )
