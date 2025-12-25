import torch
from torch.optim import SGD, AdamW
from torch_optstate import wrap
import pytest

def test_chunking_mechanism():
    """
    Verifies that setting chunk_size causes the optimizer to step in chunks.
    """
    # Create a list of parameters
    params = [torch.nn.Parameter(torch.randn(2, 2)) for _ in range(10)]
    
    # Initialize optimizer
    optimizer = SGD(params, lr=0.01)
    
    # Wrap with chunk_size=2.
    # With 10 parameters, we expect 10/2 = 5 chunks.
    wrapper = wrap(optimizer, chunk_size=2, initial_chunk_size=2)
    
    original_step = optimizer.step
    
    params_seen_per_step = []
    
    def mock_step(closure=None):
        count = 0
        for group in optimizer.param_groups:
            count += len(group['params'])
        params_seen_per_step.append(count)
        return None 

    optimizer.step = mock_step
    
    wrapper.step()
    
    optimizer.step = original_step
    
    assert len(params_seen_per_step) == 5
    
    assert all(count == 2 for count in params_seen_per_step)

def test_chunking_preserves_updates():
    """
    Verifies that chunked updates produce the same parameter values as full updates.
    """
    torch.manual_seed(42)
    
    # Setup two identical models/optimizers
    params1 = [torch.nn.Parameter(torch.randn(10, 10)) for _ in range(4)]
    params2 = [torch.nn.Parameter(p.clone()) for p in params1]
    
    opt1 = SGD(params1, lr=0.1)
    opt2 = SGD(params2, lr=0.1)
    
    # Wrap opt1 normally (baseline)
    wrapper1 = wrap(opt1) # chunk_size=None
    
    # Wrap opt2 with chunking (chunk_size=1)
    wrapper2 = wrap(opt2, chunk_size=1)
    
    # Fake gradients
    grads = [torch.randn(10, 10) for _ in range(4)]
    
    for p1, p2, g in zip(params1, params2, grads):
        p1.grad = g.clone()
        p2.grad = g.clone()
        
    # Step
    wrapper1.step()
    wrapper2.step()
    
    # Compare parameters
    for p1, p2 in zip(params1, params2):
        assert torch.allclose(p1, p2), "Parameters diverged between chunked and non-chunked updates"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_chunking_reduces_peak_memory():
    """
    Verifies that chunking actually reduces peak memory usage on GPU.
    """
    # This test requires a model large enough to show difference
    input_dim = 1024
    hidden_dim = 4096 # Large enough to matter
    
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.Linear(hidden_dim, 1)
    ).cuda()
    
    # Baseline (Wrapped but no chunking)
    opt_base = AdamW(model.parameters(), lr=1e-3)
    wrapper_base = wrap(opt_base) # chunk_size=None
    
    # Chunked
    opt_chunk = AdamW(model.parameters(), lr=1e-3)
    wrapper_chunk = wrap(opt_chunk, chunk_size=1) # Extreme chunking
    
    # Data
    x = torch.randn(32, input_dim).cuda()
    y = torch.randn(32, 1).cuda()
    
    # Helper to measure peak mem
    def measure_peak(wrapper):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        wrapper.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        wrapper.step()
        
        return torch.cuda.max_memory_allocated()
    
    # Run baseline
    peak_base = measure_peak(wrapper_base)
    
    # Run chunked
    peak_chunk = measure_peak(wrapper_chunk)
    
    print(f"Peak Base: {peak_base/1024**2:.2f} MB")
    print(f"Peak Chunk: {peak_chunk/1024**2:.2f} MB")
    
    # Assert significant reduction (at least 10%)
    assert peak_chunk < peak_base * 0.9, "Chunking did not significantly reduce peak memory"
