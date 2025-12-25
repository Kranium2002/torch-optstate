import time
import torch
from typing import Dict, Any, Optional, Union
from ..codecs import Codec, FP32Codec

class StateStore:
    """
    Central storage for virtualized optimizer state.
    Manages compression, storage, and materialization of state tensors.
    """
    def __init__(self, pin_memory: bool = False):
        # Storage structure:
        # {
        #   param_id: {
        #     state_key: (codec, packed_data)
        #   }
        # }
        # Stable integer IDs for serialization.
        self._store: Dict[int, Dict[str, Any]] = {}
        self._total_bytes = 0
        self._pin_memory = pin_memory

    def materialize(self, param_id: int, target_device: torch.device) -> Dict[str, Any]:
        """
        Retrieves the full-precision state dictionary for a parameter.
        Decompresses any compressed state.
        """
        if param_id not in self._store:
            return {}

        state_dict = {}
        for key, (codec, packed) in self._store[param_id].items():
            if isinstance(codec, Codec):
                state_dict[key] = codec.decode(packed, device=target_device)
            else:
                # Fallback for non-tensor state (e.g. step number)
                state_dict[key] = packed
        return state_dict

    def materialize_batch(self, param_ids: list[int], target_devices: list[torch.device], stats: Optional[Dict[str, Any]] = None) -> list[Dict[str, Any]]:
        """
        Batch version of materialize.
        """
        results = [{} for _ in param_ids]
        
        # Group tasks: (codec, device) -> (list of packed_data, list of (param_idx, key))
        tasks = {} 
        
        for i, (pid, device) in enumerate(zip(param_ids, target_devices)):
            if pid not in self._store:
                continue
            
            for key, (codec, packed) in self._store[pid].items():
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
            t0 = time.perf_counter()
            decoded_list = codec.batch_decode(packed_list, device=device)
            t1 = time.perf_counter()
            if stats is not None:
                elements = 0
                for val in decoded_list:
                    try:
                        elements += val.numel()
                    except Exception:
                        pass
                self._update_profile_stats(stats, "decode", codec, packed_list, t1 - t0, elements)
            for val, (idx, key) in zip(decoded_list, indices):
                results[idx][key] = val
                
        return results

    def commit(self, param_id: int, state: Dict[str, Any], codecs: Dict[str, Codec], stats: Optional[Dict[str, Any]] = None):
        """
        Compresses and stores the state dictionary for a parameter.
        """
        self.commit_batch([param_id], [state], [codecs], stats=stats)

    def commit_batch(self, param_ids: list[int], states: list[Dict[str, Any]], codecs_list: list[Dict[str, Codec]], stats: Optional[Dict[str, Any]] = None):
        """
        Batch version of commit.
        """
        # Group tasks: codec -> (list of tensors, list of (param_idx, key))
        tasks = {}
        
        # Shared default codec for tensors without explicit codec
        if not hasattr(self, '_default_fp32'):
            self._default_fp32 = FP32Codec()
        
        for i, (pid, state, codecs) in enumerate(zip(param_ids, states, codecs_list)):
            # Cleanup old bytes
            self._remove_bytes(pid)
            
            new_param_store = {}
            self._store[pid] = new_param_store
            
            for key, value in state.items():
                codec = None
                if codecs and key in codecs:
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
            t0 = time.perf_counter()
            packed_list = codec.batch_encode(tensor_list)
            t1 = time.perf_counter()
            if stats is not None:
                elements = 0
                for tensor in tensor_list:
                    try:
                        elements += tensor.numel()
                    except Exception:
                        pass
                self._update_profile_stats(stats, "encode", codec, packed_list, t1 - t0, elements)
            for packed, (idx, key) in zip(packed_list, indices):
                # Optionally pin CPU tensors to speed GPU transfers later
                packed = self._maybe_pin(packed)
                pid = param_ids[idx]
                self._store[pid][key] = (codec, packed)
                self._total_bytes += codec.bytes(packed)

    def _remove_bytes(self, param_id: int):
        if param_id not in self._store:
            return
        
        for key, (codec, packed) in self._store[param_id].items():
            if codec is not None:
                self._total_bytes -= codec.bytes(packed)
            elif isinstance(packed, (int, float)):
                self._total_bytes -= 8

    def _maybe_pin(self, packed: Any) -> Any:
        # Pin CPU tensors or tuples of tensors if requested
        if not self._pin_memory:
            return packed
        try:
            if isinstance(packed, torch.Tensor) and packed.device.type == 'cpu':
                return packed.pin_memory()
            if isinstance(packed, tuple):
                pinned = []
                changed = False
                for item in packed:
                    if isinstance(item, torch.Tensor) and item.device.type == 'cpu':
                        pinned.append(item.pin_memory())
                        changed = True
                    else:
                        pinned.append(item)
                return tuple(pinned) if changed else packed
        except Exception:
            return packed
        return packed

    def _update_profile_stats(
        self,
        stats: Dict[str, Any],
        section: str,
        codec: Codec,
        packed_list: list[Any],
        elapsed_s: float,
        elements: Optional[int] = None,
    ) -> None:
        base = stats.setdefault(section, {"time_s": 0.0, "tensors": 0, "bytes": 0, "elements": 0})
        base["time_s"] += elapsed_s
        base["tensors"] += len(packed_list)
        bytes_total = 0
        for packed in packed_list:
            try:
                bytes_total += codec.bytes(packed)
            except Exception:
                pass
        base["bytes"] += bytes_total
        if elements is not None:
            base["elements"] += elements

        by_codec = stats.setdefault(f"{section}_by_codec", {})
        name = codec.__class__.__name__
        entry = by_codec.setdefault(name, {"time_s": 0.0, "tensors": 0, "bytes": 0, "elements": 0})
        entry["time_s"] += elapsed_s
        entry["tensors"] += len(packed_list)
        entry["bytes"] += bytes_total
        if elements is not None:
            entry["elements"] += elements

    def get_memory_usage(self) -> int:
        """
        Returns the total estimated memory usage of the stored state in bytes.
        """
        return self._total_bytes
