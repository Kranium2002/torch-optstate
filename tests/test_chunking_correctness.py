
import torch
import torch.nn as nn
import copy
from torch.optim import AdamW
from torch_optstate import wrap

def test_chunking_correctness():
    torch.manual_seed(42)
    
    # 1. Setup two identical models and inputs
    input_dim = 100
    output_dim = 10
    
    model_base = nn.Sequential(
        nn.Linear(input_dim, 100),
        nn.ReLU(),
        nn.Linear(100, output_dim)
    )
    
    model_chunk = copy.deepcopy(model_base)
    
    # Ensure they start identical
    for p1, p2 in zip(model_base.parameters(), model_chunk.parameters()):
        assert torch.equal(p1, p2)
        
    # Data
    x = torch.randn(32, input_dim)
    y = torch.randn(32, output_dim)
    
    # 2. Optimizers
    # Baseline: Standard AdamW
    opt_base = AdamW(model_base.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Chunked: Wrapped AdamW with tiny chunk size to force splitting
    # Chunk size 1 means every parameter tensor is its own chunk
    opt_chunk = AdamW(model_chunk.parameters(), lr=1e-3, weight_decay=0.01)
    opt_chunk_wrapped = wrap(opt_chunk, chunk_size=1) 
    
    # 3. Step
    # Baseline
    opt_base.zero_grad()
    loss_base = (model_base(x) - y).pow(2).sum()
    loss_base.backward()
    opt_base.step()
    
    # Chunked
    opt_chunk_wrapped.zero_grad()
    loss_chunk = (model_chunk(x) - y).pow(2).sum()
    loss_chunk.backward()
    opt_chunk_wrapped.step()
    
    # 4. Compare
    print("Comparing parameters after step...")
    max_diff = 0.0
    for (name_base, p_base), (name_chunk, p_chunk) in zip(model_base.named_parameters(), model_chunk.named_parameters()):
        # Check weights
        if not torch.allclose(p_base, p_chunk, atol=1e-6):
            diff = (p_base - p_chunk).abs().max().item()
            print(f"Mismatch in {name_base}: max diff {diff}")
            max_diff = max(max_diff, diff)
        else:
            print(f"Match in {name_base}")
            
    if max_diff == 0.0:
        print("\nSUCCESS: Chunked optimization is mathematically equivalent to baseline.")
    else:
        print(f"\nFAILURE: Max difference {max_diff}")

if __name__ == "__main__":
    test_chunking_correctness()
