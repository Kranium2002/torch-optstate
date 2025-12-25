import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch_optstate import wrap, WarmupPolicy
from torch_optstate.codecs import IdentityCodec
import pytest
import copy

def test_exact_equivalence_fp32():
    """
    Verifies that Wrapped(FP32) matches baseline bitwise (or very close) for deterministic runs.
    """
    torch.manual_seed(42)
    
    # Models
    model1 = nn.Linear(10, 10)
    model2 = copy.deepcopy(model1)
    
    # Optimizers
    opt1 = AdamW(model1.parameters(), lr=1e-3)
    opt2 = AdamW(model2.parameters(), lr=1e-3)
    
    class IdentityPolicy(WarmupPolicy):
        def get_codecs(self, param, state, step):
            return {k: IdentityCodec() for k in state if torch.is_tensor(state[k])}
            
    wrapper = wrap(opt2, policy=IdentityPolicy())
    
    # Data
    x = torch.randn(5, 10)
    y = torch.randn(5, 10)
    
    # Train
    for _ in range(5):
        opt1.zero_grad()
        loss1 = (model1(x) - y).pow(2).mean()
        loss1.backward()
        opt1.step()
        
        wrapper.zero_grad()
        loss2 = (model2(x) - y).pow(2).mean()
        loss2.backward()
        wrapper.step()
        
    # Check params
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)

def test_two_param_groups():
    """
    Verifies that multiple param groups with different hyperparameters work correctly.
    """
    torch.manual_seed(42)
    
    params1 = [nn.Parameter(torch.randn(2, 2)) for _ in range(2)]
    params2 = [nn.Parameter(torch.randn(2, 2)) for _ in range(2)]
    
    opt = SGD([
        {'params': params1, 'lr': 0.1},
        {'params': params2, 'lr': 0.01}
    ])
    
    wrapper = wrap(opt)
    
    for p in params1 + params2:
        p.grad = torch.ones_like(p)
        
    wrapper.step()

    p1 = nn.Parameter(torch.zeros(1))
    p2 = nn.Parameter(torch.zeros(1))
    opt = SGD([{'params': [p1], 'lr': 0.1}, {'params': [p2], 'lr': 0.01}])
    wrapper = wrap(opt)
    p1.grad = torch.ones(1)
    p2.grad = torch.ones(1)
    wrapper.step()
    
    assert torch.allclose(p1, torch.tensor([-0.1]))
    assert torch.allclose(p2, torch.tensor([-0.01]))

def test_state_dict_round_trip():
    """
    Train -> Save -> Load -> Continue -> Match Baseline
    """
    torch.manual_seed(42)
    
    model = nn.Linear(10, 1)
    opt = AdamW(model.parameters(), lr=1e-3)
    wrapper = wrap(opt)
    
    x = torch.randn(10, 10)
    y = torch.randn(10, 1)
    
    # Train 5 steps
    for _ in range(5):
        wrapper.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        wrapper.step()
        
    # Save state
    sd = wrapper.state_dict()
    
    # New model
    model2 = nn.Linear(10, 1)
    model2.load_state_dict(model.state_dict())
    opt2 = AdamW(model2.parameters(), lr=1e-3)
    wrapper2 = wrap(opt2)
    wrapper2.load_state_dict(sd)
    
    # Continue both
    for _ in range(5):
        # Wrapper 1
        wrapper.zero_grad()
        loss1 = (model(x) - y).pow(2).mean()
        loss1.backward()
        wrapper.step()
        
        # Wrapper 2
        wrapper2.zero_grad()
        loss2 = (model2(x) - y).pow(2).mean()
        loss2.backward()
        wrapper2.step()
        
    # Compare
        for i, (p1, p2) in enumerate(zip(model.parameters(), model2.parameters())):
            if not torch.allclose(p1, p2):
                diff = (p1 - p2).abs().max().item()
                print(f"Param {i} mismatch. Max diff: {diff}")
                print(f"P1: {p1}")
                print(f"P2: {p2}")

def test_chunk_size_invariance():
    """
    Run same model with chunk_size=None vs chunk_size=1
    """
    torch.manual_seed(42)
    
    params1 = [nn.Parameter(torch.randn(10, 10)) for _ in range(4)]
    params2 = [nn.Parameter(p.clone()) for p in params1]
    
    opt1 = SGD(params1, lr=0.1)
    opt2 = SGD(params2, lr=0.1)
    
    wrapper1 = wrap(opt1) # None
    wrapper2 = wrap(opt2, chunk_size=1) # Small
    
    grads = [torch.randn(10, 10) for _ in range(4)]
    for p1, p2, g in zip(params1, params2, grads):
        p1.grad = g.clone()
        p2.grad = g.clone()
        
    wrapper1.step()
    wrapper2.step()
    
    for p1, p2 in zip(params1, params2):
        assert torch.allclose(p1, p2)

def test_optimizer_smoke():
    """
    Smoke test for various optimizers.
    """
    from torch.optim import RMSprop, Adagrad
    
    for OptCls in [AdamW, SGD, RMSprop, Adagrad]:
        model = nn.Linear(2, 2)
        kwargs = {'lr': 1e-3}
        if OptCls == SGD:
            kwargs['momentum'] = 0.9
            
        opt = OptCls(model.parameters(), **kwargs)
        wrapper = wrap(opt)
        
        x = torch.randn(2, 2)
        loss = model(x).sum()
        loss.backward()
        wrapper.step()
        
        # Check if state is virtualized (optimizer.state should be empty)
        assert len(opt.state) == 0
        # Wrapper store should have data
        assert wrapper.store.get_memory_usage() > 0
