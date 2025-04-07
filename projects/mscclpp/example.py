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
    torch.cuda.synchronize()
    dist.all_reduce(d, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    
def fused_allreduce(engine, skinny_a, b, out, scale_tensor, split_k, b_lanes, capturing):
    engine.reduce(skinny_a, b, out, scale_tensor, split_k, b_lanes, capturing)
    torch.cuda.synchronize()

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
        #torch.ones(size=(m, k), device="cuda", dtype=torch.float32),
        torch.ones(size=(m, k), device="cuda", dtype=torch.float32),
        scale_tensor,
    )[0]
    b = fp8_quantize(
        torch.ones(size=(n, k), device="cuda", dtype=torch.float32),
        scale_tensor,
    )[0].t()
    output = torch.zeros(size=(m, n), dtype=torch.float16, device="cuda")
    return skinny_a, b, scale_tensor, output

def generate_random_skinny_gemm_data(
    m: int, n: int, k: int, seed: Optional[int] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generates random inputs for the skinny_gemm operation. The generated input's shape is determined by (m), (n) and
    (k), and one can pass a (seed) to ensure repeatability."""
    if seed is not None:
        torch.manual_seed(seed)
    scale_tensor = torch.rand(size=(1,), device="cuda", dtype=torch.float32).mul(2).add(1)
    skinny_a = fp8_quantize(
        torch.normal(0, 1, size=(m, k), device="cuda", dtype=torch.float32),
        scale_tensor,
    )[0]
    b = fp8_quantize(
        torch.normal(0, 1, size=(n, k), device="cuda", dtype=torch.float32),
        scale_tensor,
    )[0].t()
    output = torch.zeros(size=(m, n), dtype=torch.float16, device="cuda")
    return skinny_a, b, scale_tensor, output

from time import sleep

def create_index_tensor(m, n):
    """
    Creates an (m, n) tensor on CUDA with float16 dtype where each element is its index.
    
    Args:
        m (int): Number of rows
        n (int): Number of columns
        
    Returns:
        torch.Tensor: Tensor of shape (m, n) with index values, float16, on CUDA
    """
    # Create on CUDA with float16 dtype
    indices = torch.arange(m * n, dtype=torch.float16, device='cuda')
    
    # Reshape to (m, n)
    return indices.view(m, n)

def _benchmark_skinny_gemm(rank, world_size, m: int, n: int, k: int, split_k: int, b_lanes: int):
    """Test for the skinny_gemm operation."""
    try:
        # Initialize process
        rank, world_size = init_process(rank, world_size, "localhost")

        # Generate data
        skinny_a, b, scale_tensor, out = generate_skinny_gemm_data(m, n, k, seed=0)
        skinny_a2, b2, scale_tensor2, out2 = generate_skinny_gemm_data(m, n, k, seed=0)

        #print("out_size: ", out.shape)

        # Create AllReduce instance
        comms_a = torch.zeros(size=(m, n), dtype=torch.float16, device="cuda")
        #comms_a.copy_
        comms_b = torch.zeros(size=(m, n), dtype=torch.float16, device="cuda")
        allreduce = mscclpp_allreduce.AllReduceEngine(rank, world_size, 50004, comms_a, comms_b)
        
        #print("comms_a", comms_a.int())

        # Perform reduction
        #allreduce.reduce(skinny_a, b, out, scale_tensor, split_k, b_lanes, False)
        #skinny_gemm(
        #        skinny_a=skinny_a,
        #        b=b,
        #        scale_tensor=scale_tensor,
        #        output=out,
        #        split_k=9,
        #        b_lanes=5,
        #    )
        #out1 = out.clone()
        #out2 = out.clone()
        #comms_a.copy_(out)
        #skinny_gemm_and_ar_pytorch(skinny_a, b, out1, scale_tensor)
        #allreduce.reduce(skinny_a, b, out1, scale_tensor, split_k, b_lanes, False)
        #comms_a.fill_(0)
        #comms_b.fill_(0)
        #allreduce.reduce(skinny_a, b, out2, scale_tensor, split_k, b_lanes, False)
        #skinny_gemm_and_ar_pytorch(skinny_a, b, out2, scale_tensor)
        #print("out1", out1)
        #print("out2", out2)
        
        
        #torch.testing.assert_close(out1, out2, 
        #                        rtol=1e-3, atol=1e-2)
        
        # out3 = out.clone()
        # out4 = out.clone()
        # comms_a.copy_(out)
        # out4.copy_(out)
        # comms_b.fill_(0)
        # comms_a.copy_(out)
        # #skinny_gemm_and_ar_pytorch(skinny_a, b, out, scale_tensor)
        # allreduce.reduce(skinny_a, b, out3, scale_tensor, split_k, b_lanes, False)
        # skinny_gemm_and_ar_pytorch(skinny_a, b, out4, scale_tensor)
        # print("out3", out3)
        # print("out4", out4)
        # torch.testing.assert_close(out3, out4, 
        #                         rtol=1e-3, atol=1e-2)
        
        #dist.all_reduce(comms_a, op=dist.ReduceOp.SUM)

        #start_torch = torch.cuda.Event(enable_timing=True)
        #end_torch = torch.cuda.Event(enable_timing=True)
        #start_fused = torch.cuda.Event(enable_timing=True)
        #end_fused = torch.cuda.Event(enable_timing=True)
        #torch.cuda.synchronize()

        #start_fused.record()
        #fused = timeit.timeit(lambda: allreduce.reduce(skinny_a, b, out, scale_tensor, split_k, b_lanes, False), number=10)
        allreduce.reduce(skinny_a, b, out, scale_tensor, split_k, b_lanes, False)
        #end_fused.record()
        #torch.cuda.synchronize()
        #start_torch.record()
        #pytorch = timeit.timeit(lambda: skinny_gemm_and_ar_pytorch(skinny_a, b, out, scale_tensor), number=10)
        #skinny_gemm_and_ar_pytorch(skinny_a, b, out, scale_tensor)
        #end_torch.record()

        #print(f"Fused: {fused} \n")
        #print(f"Pytorch: {pytorch} \n")
        
        #fused2 = timeit.timeit(lambda: fused_allreduce(allreduce, skinny_a, b, out, scale_tensor, split_k, b_lanes, False), number=10)
        #pytorch2 = timeit.timeit(lambda: skinny_gemm_and_ar_pytorch(skinny_a, b, out, scale_tensor), number=10)


        #print(f"Fused2: {fused2} \n")
        #print(f"Pytorch2: {pytorch2} \n")
        #print(f"Fused: {start_fused.elapsed_time(end_fused)} \n")
        #print(f"Pytorch: {start_torch.elapsed_time(end_torch)} \n")
        #torch.set_printoptions(profile="full")
        print("out", out)
        #print("comms_a", comms_a)
        #print("comms_b", comms_b)

        #allreduce.reduce(skinny_a2, b2, out2, scale_tensor2, split_k, b_lanes, False)
        #print("out2", out2)
        #torch.set_printoptions(profile="full") 
        #shape = (8, 16384)
        
        #rank = dist.get_rank()
        #torch.manual_seed(42 + rank)  # Different seed per rank
        #tensor = torch.randn(*shape, dtype=torch.float16, device="cuda") #* (rank + 1)
        #if (rank == 1):
        #    tensor = torch.ones(*shape, dtype=torch.float16, device="cuda")
        #else:
        #    tensor = torch.ones (*shape, dtype=torch.float16, device="cuda") * rank
            
        #tensor = torch.ones (*shape, dtype=torch.float16, device="cuda") * rank
        
        #tensor = torch.ones(*shape, dtype=torch.float16, device="cuda")
        
        #comms_a = torch.zeros(size=shape, dtype=torch.float16, device="cuda")
        #comms_a.copy_(tensor)
        #comms_b = torch.zeros(size=shape, dtype=torch.float16, device="cuda")
        
        #print("rank:", rank, "comms_b", comms_b)
        #print("comms_b", comms_b)
        #allreduce = mscclpp_allreduce.AllReduceEngine(rank, world_size, 50004, comms_a, comms_b)
        #custom_result = tensor.clone()
        #comms_b.copy_(torch.ones (*shape, dtype=torch.float16, device="cuda") * ((rank + 8 - 1) % 8))
        #allreduce.reduce(skinny_a, b, custom_result, scale_tensor, split_k, b_lanes, False)
        #torch.cuda.synchronize()
        
        # PyTorch allreduce
        #torch_result = tensor.clone()
        #dist.all_reduce(torch_result)
        
        #print("pytorch_result", torch_result)
        #print("custom_result", custom_result)
        #print("comms_a", comms_a)
        #print("comms_b", comms_b)
        
        #torch.testing.assert_close(custom_result, torch_result, 
        #                        rtol=1e-5, atol=1e-5)
    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise
    finally:
        dist.destroy_process_group()

def test_process():
    M = 8
    #N = 13312
    N = 32768
    K = 32768
    #N = 256
    #K = 512
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
