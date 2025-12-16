import torch
import pytest
from torch_optstate.codecs import FP32Codec, FP16Codec, BF16Codec, Int8MomentumCodec

def test_fp32_codec():
    codec = FP32Codec()
    tensor = torch.randn(10, 10)
    packed = codec.encode(tensor)
    decoded = codec.decode(packed)
    
    assert torch.allclose(tensor, decoded)
    assert codec.bytes(packed) == tensor.numel() * 4

def test_fp16_codec():
    codec = FP16Codec()
    tensor = torch.randn(10, 10)
    packed = codec.encode(tensor)
    decoded = codec.decode(packed)
    
    assert packed.dtype == torch.float16
    assert torch.allclose(tensor, decoded, atol=1e-3)
    assert codec.bytes(packed) == tensor.numel() * 2

def test_bf16_codec():
    codec = BF16Codec()
    tensor = torch.randn(10, 10)
    packed = codec.encode(tensor)
    decoded = codec.decode(packed)
    
    assert packed.dtype == torch.bfloat16
    # BF16 has lower precision, so higher tolerance
    assert torch.allclose(tensor, decoded, atol=1e-2)
    assert codec.bytes(packed) == tensor.numel() * 2

def test_int8_momentum_codec():
    codec = Int8MomentumCodec()
    tensor = torch.randn(100, 100) * 10.0 # Scale up
    packed = codec.encode(tensor)
    decoded = codec.decode(packed)
    
    quantized, scale = packed
    assert quantized.dtype == torch.int8
    assert scale.dtype == tensor.dtype
    
    # Check error bounds. Int8 quantization error is roughly scale / 127 / 2
    # But we can just check reasonable correlation or error
    error = (tensor - decoded).abs().max()
    max_val = tensor.abs().max()
    # Expected max error is roughly max_val / 127 * 0.5
    expected_error = (max_val / 127.0) * 0.5
    
    # Allow some slack
    assert error <= expected_error * 1.1
    
    # Check bytes
    expected_bytes = (quantized.numel() * 1) + (scale.numel() * 4)
    assert codec.bytes(packed) == expected_bytes

def test_int8_momentum_codec_zeros():
    codec = Int8MomentumCodec()
    tensor = torch.zeros(10, 10)
    packed = codec.encode(tensor)
    decoded = codec.decode(packed)
    
    assert torch.allclose(tensor, decoded)
