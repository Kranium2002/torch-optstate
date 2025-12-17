import torch
import time
from torch.optim import Optimizer
from typing import Optional, Dict, Any, Callable
from .core.state_store import StateStore
from .policy.base import Policy
from .policy.simple import WarmupPolicy

class OptimizerWrapper(Optimizer):
    """
    Wrapper around a PyTorch optimizer that virtualizes its state.
    State is compressed and stored in a StateStore when not in use (i.e. outside of step()).
    """
    def __init__(self, optimizer: Optimizer, policy: Optional[Policy] = None, chunk_size: Optional[int] = None):
        self.optimizer = optimizer
        self.policy = policy or WarmupPolicy()
        self.store = StateStore()
        self.chunk_size = chunk_size
        
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
        if self.chunk_size is not None and closure is None:
            return self._step_chunked()
        
        t0 = time.perf_counter()
        
        # 1. Collect all params
        all_params = []
        all_pids = []
        all_devices = []
        for group in self.param_groups:
            for p in group['params']:
                all_params.append(p)
                all_pids.append(self.param_to_id[p])
                all_devices.append(p.device)

        t1 = time.perf_counter()

        # 2. Materialize all at once
        # This allows the store to batch decompressions
        states = self.store.materialize_batch(all_pids, all_devices)
        
        # 3. Populate optimizer.state
        for param, state in zip(all_params, states):
            if state: # Only if we had state
                self.optimizer.state[param] = state

        t2 = time.perf_counter()

        # 4. Perform the step
        loss = self.optimizer.step(closure)
        self._global_step += 1

        t3 = time.perf_counter()

        # 5. Collect new states and codecs for batch commit
        pids_to_commit = []
        states_to_commit = []
        codecs_list = []

        for group in self.param_groups:
            for p in group['params']:
                if p in self.optimizer.state:
                    state = self.optimizer.state[p]
                    
                    # Determine step
                    # Prefer internal step if available (e.g. Adam), else use global counter
                    step = state.get('step', self._global_step)
                    if torch.is_tensor(step):
                        step = step.item()
                    
                    # Get codecs from policy
                    codecs = self.policy.get_codecs(p, state, step)
                    
                    pids_to_commit.append(self.param_to_id[p])
                    states_to_commit.append(state)
                    codecs_list.append(codecs)
                    
        # 6. Commit batch
        # This allows the store to batch compressions
        if pids_to_commit:
            self.store.commit_batch(pids_to_commit, states_to_commit, codecs_list)
                    
        # 7. Clear optimizer state to free memory
        self.optimizer.state.clear()

        t4 = time.perf_counter()
        
        self.last_step_timings = {
            'materialize': t2 - t1,
            'step': t3 - t2,
            'commit': t4 - t3,
            'overhead': (t1 - t0) + (t4 - t0) - (t4 - t1) # Rough overhead
        }

        return loss
                    
        # 7. Clear optimizer state to free memory
        self.optimizer.state.clear()

        return loss

    def _step_chunked(self):
        """
        Performs the step in chunks to minimize peak memory usage.
        This is only possible if no closure is provided.
        """
        total_materialize = 0.0
        total_step = 0.0
        total_commit = 0.0
        
        # 1. Backup all params
        all_original_params = [g['params'] for g in self.optimizer.param_groups]
        
        # 2. Empty all groups
        for g in self.optimizer.param_groups:
            g['params'] = []
            
        # 3. Iterate and chunk
        for group_idx, group in enumerate(self.optimizer.param_groups):
            original_params = all_original_params[group_idx]
            
            for i in range(0, len(original_params), self.chunk_size):
                chunk_params = original_params[i : i + self.chunk_size]
                
                t1 = time.perf_counter()
                
                # Materialize
                chunk_pids = [self.param_to_id[p] for p in chunk_params]
                chunk_devices = [p.device for p in chunk_params]
                states = self.store.materialize_batch(chunk_pids, chunk_devices)
                
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
                
                # Commit
                pids_to_commit = []
                states_to_commit = []
                codecs_list = []
                
                for p in chunk_params:
                    if p in self.optimizer.state:
                        state = self.optimizer.state[p]
                        step = state.get('step', self._global_step)
                        if torch.is_tensor(step):
                            step = step.item()
                        codecs = self.policy.get_codecs(p, state, step)
                        pids_to_commit.append(self.param_to_id[p])
                        states_to_commit.append(state)
                        codecs_list.append(codecs)
                
                if pids_to_commit:
                    self.store.commit_batch(pids_to_commit, states_to_commit, codecs_list)
                
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
        
        self.last_step_timings = {
            'materialize': total_materialize,
            'step': total_step,
            'commit': total_commit,
            'overhead': 0.0 
        }
        
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

    def add_param_group(self, param_group: Dict[str, Any]):
        self.optimizer.add_param_group(param_group)

    def __repr__(self):
        return f"OptimizerWrapper({repr(self.optimizer)})"

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

def wrap(optimizer: Optimizer, policy: Optional[Policy] = None, chunk_size: Optional[int] = None) -> OptimizerWrapper:
    """
    Wraps an existing PyTorch optimizer with state virtualization.
    
    Args:
        optimizer: The optimizer to wrap.
        policy: The policy to use for state compression.
        chunk_size: If set, performs step() in chunks of this size to reduce peak memory.
    
    Returns:
        An OptimizerWrapper instance.
    """
    return OptimizerWrapper(optimizer, policy, chunk_size)
