import torch
import pytest
from torch_optstate.core.state_store import StateStore
from torch_optstate.codecs import FP32Codec, FP16Codec

def test_state_store_basic():
    store = StateStore()
    param_id = 0
    state = {'exp_avg': torch.randn(10), 'step': 1}
    
    codecs = {'exp_avg': FP32Codec()}
    
    store.commit(param_id, state, codecs)
    
    materialized = store.materialize(param_id, target_device=torch.device('cpu'))
    assert torch.allclose(state['exp_avg'], materialized['exp_avg'])
    assert state['step'] == materialized['step']

def test_state_store_memory_accounting():
    store = StateStore()
    param_id = 0
    state = {'exp_avg': torch.randn(10)} # 10 * 4 = 40 bytes
    
    codecs = {'exp_avg': FP32Codec()}
    store.commit(param_id, state, codecs)
    
    assert store.get_memory_usage() == 40
    
    # Update with FP16
    codecs = {'exp_avg': FP16Codec()}
    store.commit(param_id, state, codecs)
    
    assert store.get_memory_usage() == 20 # 10 * 2 = 20 bytes

def test_state_store_multiple_params():
    store = StateStore()
    p1_id = 1
    p2_id = 2
    
    store.commit(p1_id, {'v': torch.randn(10)}, {'v': FP32Codec()}) # 40 bytes
    store.commit(p2_id, {'v': torch.randn(20)}, {'v': FP32Codec()}) # 80 bytes
    
    assert store.get_memory_usage() == 120
    
    store.commit(p1_id, {'v': torch.randn(10)}, {'v': FP16Codec()}) # 20 bytes
    assert store.get_memory_usage() == 100 # 20 + 80
