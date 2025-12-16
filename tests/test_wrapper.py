import torch
import pytest
from torch.optim import AdamW
from torch_optstate import wrap
from torch_optstate.policy import WarmupPolicy

def test_wrapper_equivalence_fp32():
    # Test that wrapped optimizer with FP32 policy behaves exactly like baseline
    
    # Setup
    model1 = torch.nn.Linear(10, 1)
    model2 = torch.nn.Linear(10, 1)
    model2.load_state_dict(model1.state_dict())
    
    opt1 = AdamW(model1.parameters(), lr=1e-3)
    
    # Policy that never compresses (warmup infinity)
    policy = WarmupPolicy(warmup_steps=1000)
    opt2 = wrap(AdamW(model2.parameters(), lr=1e-3), policy=policy)
    
    # Train loop
    for i in range(10):
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        
        # Step 1
        opt1.zero_grad()
        loss1 = (model1(x) - y).pow(2).mean()
        loss1.backward()
        opt1.step()
        
        # Step 2
        opt2.zero_grad()
        loss2 = (model2(x) - y).pow(2).mean()
        loss2.backward()
        opt2.step()
        
        assert torch.allclose(loss1, loss2)
        
        # Check params
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

def test_wrapper_compression_runs():
    # Test that wrapped optimizer runs with compression enabled
    
    model = torch.nn.Linear(10, 1)
    # Policy that compresses immediately
    policy = WarmupPolicy(warmup_steps=0)
    opt = wrap(AdamW(model.parameters(), lr=1e-3), policy=policy)
    
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    for i in range(5):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
        
    # Check that state is compressed in store
    # We can inspect the store directly
    assert opt.store.get_memory_usage() > 0
    
    # Check that optimizer state is empty (cleared)
    assert len(opt.optimizer.state) == 0

def test_state_dict_roundtrip():
    model = torch.nn.Linear(10, 1)
    opt = wrap(AdamW(model.parameters(), lr=1e-3))
    
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    loss = (model(x) - y).pow(2).mean()
    loss.backward()
    opt.step()
    
    sd = opt.state_dict()
    
    # Create new optimizer
    opt2 = wrap(AdamW(model.parameters(), lr=1e-3))
    opt2.load_state_dict(sd)
    
    # Check if state is loaded into store
    # We need to access the internal store to verify
    # Or just step and see if it works
    
    loss = (model(x) - y).pow(2).mean()
    loss.backward()
    opt2.step()
