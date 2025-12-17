import torch
from typing import Dict, Any, Optional
from ..codecs import Codec, FP32Codec

class StateStore:
    """
    Central storage for virtualized optimizer state.
    Manages compression, storage, and materialization of state tensors.
    """
    def __init__(self):
        # Storage structure:
        # {
        #   param_id: {
        #     state_key: (codec, packed_data)
        #   }
        # }
        # We use id(param) as key because params are hashable but we want to be sure.
        # Actually, weakref might be better, but for now let's use id(param) or the param itself if it's hashable.
        # PyTorch params are hashable.
        self._store: Dict[torch.Tensor, Dict[str, Any]] = {}
        self._total_bytes = 0

    def materialize(self, param: torch.Tensor) -> Dict[str, Any]:
        """
        Retrieves the full-precision state dictionary for a parameter.
        Decompresses any compressed state.
        """
        if param not in self._store:
            return {}

        state_dict = {}
        for key, (codec, packed) in self._store[param].items():
            if isinstance(codec, Codec):
                state_dict[key] = codec.decode(packed, device=param.device)
            else:
                # Fallback for non-tensor state (e.g. step number)
                state_dict[key] = packed
        return state_dict

    def materialize_batch(self, params: list[torch.Tensor]) -> list[Dict[str, Any]]:
        """
        Batch version of materialize.
        """
        results = [{} for _ in params]
        
        # Group tasks: (codec, device) -> (list of packed_data, list of (param_idx, key))
        tasks = {} 
        
        for i, param in enumerate(params):
            if param not in self._store:
                continue
            
            device = param.device
            for key, (codec, packed) in self._store[param].items():
                if isinstance(codec, Codec):
                    task_key = (codec, device)
                    if task_key not in tasks:
                        tasks[task_key] = ([], [])
                    tasks[task_key][0].append(packed)
                    tasks[task_key][1].append((i, key))
                else:
                    # Non-tensor state
                    results[i][key] = packed
                    
        # Execute batch decodes
        for (codec, device), (packed_list, indices) in tasks.items():
            decoded_list = codec.batch_decode(packed_list, device=device)
            for val, (idx, key) in zip(decoded_list, indices):
                results[idx][key] = val
                
        return results

    def commit(self, param: torch.Tensor, state: Dict[str, Any], codecs: Dict[str, Codec]):
        """
        Compresses and stores the state dictionary for a parameter.
        """
        self.commit_batch([param], [state], [codecs])

    def commit_batch(self, params: list[torch.Tensor], states: list[Dict[str, Any]], codecs_list: list[Dict[str, Codec]]):
        """
        Batch version of commit.
        """
        # Group tasks: codec -> (list of tensors, list of (param_idx, key))
        tasks = {}
        
        # Shared default codec for tensors without explicit codec
        if not hasattr(self, '_default_fp32'):
            self._default_fp32 = FP32Codec()
        
        for i, (param, state, codecs) in enumerate(zip(params, states, codecs_list)):
            # Cleanup old bytes
            self._remove_bytes(param)
            
            new_param_store = {}
            self._store[param] = new_param_store
            
            for key, value in state.items():
                codec = None
                if key in codecs:
                    codec = codecs[key]
                elif torch.is_tensor(value):
                    codec = self._default_fp32
                
                if codec:
                    if codec not in tasks:
                        tasks[codec] = ([], [])
                    tasks[codec][0].append(value)
                    tasks[codec][1].append((i, key))
                else:
                    # Non-tensor
                    new_param_store[key] = (None, value)
                    if isinstance(value, (int, float)):
                        self._total_bytes += 8

        # Execute batch encodes
        for codec, (tensor_list, indices) in tasks.items():
            packed_list = codec.batch_encode(tensor_list)
            for packed, (idx, key) in zip(packed_list, indices):
                param = params[idx]
                self._store[param][key] = (codec, packed)
                self._total_bytes += codec.bytes(packed)

    def _remove_bytes(self, param: torch.Tensor):
        if param not in self._store:
            return
        
        for key, (codec, packed) in self._store[param].items():
            if codec is not None:
                self._total_bytes -= codec.bytes(packed)
            elif isinstance(packed, (int, float)):
                self._total_bytes -= 8

    def get_memory_usage(self) -> int:
        """
        Returns the total estimated memory usage of the stored state in bytes.
        """
        return self._total_bytes
