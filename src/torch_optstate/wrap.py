import torch
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
                self.store.commit(param, state, codecs)
            
            # Clear original state to save memory
            self.optimizer.state.clear()

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
        
        # 1. Collect all params
        all_params = []
        for group in self.param_groups:
            all_params.extend(group['params'])

        # 2. Materialize all at once
        # This allows the store to batch decompressions
        states = self.store.materialize_batch(all_params)
        
        # 3. Populate optimizer.state
        for param, state in zip(all_params, states):
            if state: # Only if we had state
                self.optimizer.state[param] = state

        # 4. Perform the step
        loss = self.optimizer.step(closure)
        self._global_step += 1

        # 5. Collect new states and codecs for batch commit
        params_to_commit = []
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
                    
                    params_to_commit.append(p)
                    states_to_commit.append(state)
                    codecs_list.append(codecs)
                    
        # 6. Commit batch
        # This allows the store to batch compressions
        if params_to_commit:
            self.store.commit_batch(params_to_commit, states_to_commit, codecs_list)
                    
        # 7. Clear optimizer state to free memory
        self.optimizer.state.clear()

        return loss

    def _step_chunked(self):
        """
        Performs the step in chunks to minimize peak memory usage.
        This is only possible if no closure is provided.
        """
        # Backup original param groups
        original_param_groups = self.optimizer.param_groups
        
        # We need to iterate over groups and chunk them
        # We can't easily chunk across groups because optimizers expect specific group options
        
        for group_idx, group in enumerate(original_param_groups):
            params = group['params']
            
            # Process in chunks
            for i in range(0, len(params), self.chunk_size):
                chunk_params = params[i : i + self.chunk_size]
                
                # 1. Materialize chunk
                states = self.store.materialize_batch(chunk_params)
                for param, state in zip(chunk_params, states):
                    if state:
                        self.optimizer.state[param] = state
                
                # 2. Create temporary group for this chunk
                # We copy the group dict but replace params
                temp_group = group.copy()
                temp_group['params'] = chunk_params
                
                # 3. Inject into optimizer
                # We only provide this single group to the optimizer
                self.optimizer.param_groups = [temp_group]
                
                # 4. Step
                self.optimizer.step()
                
                # 5. Commit chunk
                params_to_commit = []
                states_to_commit = []
                codecs_list = []
                
                for p in chunk_params:
                    if p in self.optimizer.state:
                        state = self.optimizer.state[p]
                        
                        step = state.get('step', self._global_step)
                        if torch.is_tensor(step):
                            step = step.item()
                        
                        codecs = self.policy.get_codecs(p, state, step)
                        
                        params_to_commit.append(p)
                        states_to_commit.append(state)
                        codecs_list.append(codecs)
                
                if params_to_commit:
                    self.store.commit_batch(params_to_commit, states_to_commit, codecs_list)
                
                # 6. Clear state
                self.optimizer.state.clear()
        
        # Restore original groups
        self.optimizer.param_groups = original_param_groups
        self._global_step += 1
        return None

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        # Materialize everything to generate a standard state_dict
        # We need to temporarily populate optimizer.state
        
        # Save current state of optimizer.state (should be empty)
        original_state = self.optimizer.state.copy()
        
        for param in self.store._store:
            self.optimizer.state[param] = self.store.materialize(param)
            
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
            self.store.commit(param, state, codecs)
            
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
