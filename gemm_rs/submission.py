from task import input_t, output_t
import torch
from typing import Optional

# Using torch.compile with max-autotune for aggressive optimization.
# full_graph=True ensures the entire function is compiled without breaking back to Python.


@torch.compile(mode="max-autotune")
def compiled_gemm_rs(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> output_t:
    """
    A fully compiled graph for the Gemm-ReduceScatter operation.
    """
    # Allow TF32 for matmul operations for better performance on supported hardware
    torch.backends.cuda.matmul.allow_tf32 = True

    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()

    # matmul
    output = torch.matmul(input, weight.T)
    if bias is not None:
        output = output + bias

    # reduce scatter
    # The output buffer for reduce_scatter must be pre-allocated.
    rs_output = torch.empty((M // world_size, N),
                            dtype=output.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output)

    return rs_output


def custom_kernel(data: input_t) -> output_t:
    """
    Optimized kernel for Gemm-ReduceScatter operation.
    This implementation uses a fully compiled graph with max-autotune.
    """
    input, weight, bias = data
    return compiled_gemm_rs(input, weight, bias)
