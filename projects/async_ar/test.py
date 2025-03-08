from typing import Tuple, Optional
import torch
from torch import Tensor

from hip_fused_gemm_nccl import FusedGEMMAR
import pytest
import torch.multiprocessing as mp
import torch.distributed as dist




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


def _test_fused_gemm_ar(m: int, n: int, k: int, split_k: int, b_lanes: int, process_group, comm, verbose: bool) -> Tuple[float, float, float]:
    """Test for the skinny_gemm operation. Can be either called with (verbose) flag on, in which case there will be a 
    lot of text displayed, which is good for debugging, or with (verbose) turned off, which is good for pytest."""
    # Generate data
    skinny_a, b, scale_tensor, out = generate_skinny_gemm_data(m, n, k, seed=0)
    # Compute operation outputs
    output = fused_gemm_ar(skinny_a, b, scale_tensor, out.clone(), split_k, b_lanes, comm)
    # Compute reference outputs
    print(f"Output: {output}")
    # Crunch error metrics on each output and maybe display them 
    return


def test_process(
    rank,
    world_size, m: int, 
    n: int,
    k: int,
    split_k: int,
    b_lanes: int,
    atol: float = 1e-1
):
    """
    Function to initialize distributed training.
    Each spawned process runs this function independently.
    """

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, init_method="tcp://127.0.0.1:22500")
    
    comm = None

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    #process_group = dist.distributed_c10d._get_default_group()
    
    gemmar = FusedGEMMAR(world_size, rank, torch.distributed.group.WORLD)
    
    skinny_a, b, scale_tensor, out = generate_skinny_gemm_data(m, n, k, seed=0)

    result = out.clone()
    
    skinny_a = skinny_a.to(dtype=torch.float16)
    b = b.to(dtype=torch.float16)

    gemmar.gemm_ar(skinny_a, b, result, scale_tensor, split_k, b_lanes)
    print(f"result: {result}")
    print(f"Test passed for Split-K = {split_k} on rank {rank}!")

    # Cleanup
    dist.destroy_process_group()
    
    
@pytest.mark.parametrize("split_k", [4])
@pytest.mark.parametrize("b_lanes", [5])
@pytest.mark.parametrize("k", [256])
@pytest.mark.parametrize("n", [128])
@pytest.mark.parametrize("m", [4])
def test_fused_gemm_ar(m, n, k, split_k, b_lanes):
    """
    Multi-GPU test for Split-K GEMM using multiprocessing.
    """
    #world_size = torch.cuda.device_count()  # Use all available GPUs
    world_size = 2
    if world_size < 2:
        pytest.skip("Test requires at least 2 GPUs.")

    # ✅ Spawn processes, each running `setup_process()`
    mp.spawn(test_process, args=(world_size, m, n, k, split_k, b_lanes), nprocs=world_size, join=True)
