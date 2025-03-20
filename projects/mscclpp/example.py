from typing import Tuple, Optional
import pytest
import torch
import torch.multiprocessing as mp
from torch import Tensor
import mscclpp_allreduce
import os

import torch.distributed as dist
from hf_rocm_kernels import skinny_gemm

import timeit

def skinny_gemm_and_ar_pytorch(a, b, d, scale):
    # Perform GEMM
    skinny_gemm(
                skinny_a=a,
                b=b,
                scale_tensor=scale,
                output=d,
                split_k=9,
                b_lanes=5,
            )
    dist.all_reduce(d, op=dist.ReduceOp.SUM)

def init_process(rank, world_size, master_addr):
    """Initialize process group and set environment variables"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)
    
    return rank, world_size

def allreduce_process(rank, world_size):
    torch.cuda.set_device(rank)
    tensor = torch.ones(10, device="cuda") * (rank + 1)
    nccl_allreduce.allreduce(tensor, world_size, rank)
    expected = sum(range(1, world_size + 1)) * torch.ones(10, device="cuda")
    assert torch.allclose(tensor, expected), f"Rank {rank}: Expected {expected}, got {tensor}"

def test_nccl_allreduce():
    world_size = torch.cuda.device_count()
    if world_size < 2:
        pytest.skip("Test requires at least 2 GPUs")
    
    mp.spawn(allreduce_process, args=(world_size,), nprocs=world_size, join=True)
    
def fp8_quantize(
    x_full_precision: Tensor,
    scale: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Quantizes a tensor (x_full_precision) according to a tensor-wise (scale) to float8_e4m3fnuz format. This function is
    meant to mimic the behavior of TGI and thus was inspired by it:
    https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/layers/fp8.py
    For reference on dtypes: https://onnx.ai/onnx/technical/float8.html
    """
    # Scale and clamp in full precision
    finfo = torch.finfo(torch.float8_e4m3fn)
    x_quantized = (x_full_precision * scale.reciprocal()).clamp(min=finfo.min, max=finfo.max)
    # Convert to float8_e4m3fn format, without removing signed zeros
    x_quantized = x_quantized.to(torch.float8_e4m3fn)
    # Remove signed zeros, which correspond to NaNs in float8_e4m3fnuz format
    weight_as_int8 = x_quantized.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    mask = weight_as_int8 == ROCM_FP8_NAN_AS_INT
    weight_as_int8[mask] = 0
    x_quantized = weight_as_int8.view(torch.float8_e4m3fnuz)
    # For the same bits representation, e4m3fnuz value is half of the e4m3fn value, so we should double the scaling 
    # factor to get the same dequantized value.
    return x_quantized, scale * 2.0

    
def generate_skinny_gemm_data(
    m: int, n: int, k: int, seed: Optional[int] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generates random inputs for the skinny_gemm operation. The generated input's shape is determined by (m), (n) and
    (k), and one can pass a (seed) to ensure repeatability."""
    if seed is not None:
        torch.manual_seed(seed)
    scale_tensor = torch.ones(size=(1,), device="cuda", dtype=torch.float32).mul(2).add(1)
    skinny_a = fp8_quantize(
        torch.ones(size=(m, k), device="cuda", dtype=torch.float32),
        scale_tensor,
    )[0]
    b = fp8_quantize(
        torch.ones(size=(n, k), device="cuda", dtype=torch.float32),
        scale_tensor,
    )[0].t()
    output = torch.zeros(size=(m, n), dtype=torch.float16, device="cuda")
    return skinny_a, b, scale_tensor, output


def _benchmark_skinny_gemm(rank, world_size, m: int, n: int, k: int, split_k: int, b_lanes: int):
    """Test for the skinny_gemm operation."""
    try:
        # Initialize process
        rank, world_size = init_process(rank, world_size, "localhost")
        
        # Generate data
        skinny_a, b, scale_tensor, out = generate_skinny_gemm_data(m, n, k, seed=0)
        
        print("out_size: ", out.shape)
        
        # Create AllReduce instance
        allreduce = mscclpp_allreduce.AllReduceEngine(rank, world_size)
        
        # Perform reduction
        #allreduce.reduce(skinny_a, b, out, scale_tensor, split_k, b_lanes)
        #skinny_gemm_and_ar_pytorch(skinny_a, b, out, scale_tensor)
        
        start_torch = torch.cuda.Event(enable_timing=True)
        end_torch = torch.cuda.Event(enable_timing=True)
        start_fused = torch.cuda.Event(enable_timing=True)
        end_fused = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        
        start_fused.record()
        #fused = timeit.timeit(lambda: allreduce.reduce(skinny_a, b, out, scale_tensor, split_k, b_lanes), number=1)
        allreduce.reduce(skinny_a, b, out, scale_tensor, split_k, b_lanes)
        end_fused.record()
        torch.cuda.synchronize()
        start_torch.record()
        #torch = timeit.timeit(lambda: skinny_gemm_and_ar_pytorch(skinny_a, b, out, scale_tensor), number=1)
        skinny_gemm_and_ar_pytorch(skinny_a, b, out, scale_tensor)
        end_torch.record()
        
        #print(f"Fused: {fused} \n")
        #print(f"Pytorch: {torch} \n")
        print(f"Fused: {start_fused.elapsed_time(end_fused)} \n")
        print(f"Pytorch: {start_torch.elapsed_time(end_torch)} \n")
        #torch.set_printoptions(profile="full")
        #print(out)
        
    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise

def test_process():
    M = 8
    #N = 13312
    N=2304
    K = 16384
    B_LANES = 5
    SPLIT_K = 3
    
    # Use all available GPUs
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices available")
    
    # Start multiple processes
    mp.spawn(
        _benchmark_skinny_gemm,
        args=(world_size, M, N, K, SPLIT_K, B_LANES),
        nprocs=world_size,
        join=True,
        start_method='spawn'
    )

if __name__ == "__main__":
    test_process()