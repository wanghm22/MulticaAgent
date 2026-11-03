import logging
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

@libentry()
@triton.jit
def mv_vector_kernel(A, B, C, N, M, stride_an, stride_am, stride_bm, stride_cn,
                     BLOCK_M: tl.constexpr):
    """One program computes one output row using a vector FMA reduction."""
    pid = ext.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    row = pid
    # Keep the accumulator scalar while traversing K in vector chunks.  This
    # avoids materialising an (N,M) tile and maps naturally to RVV reductions.
    acc = tl.zeros((), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        idx = m + offs_m
        mask = idx < M
        a = tl.load(A + row * stride_an + idx * stride_am, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + idx * stride_bm, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(a * b, axis=0)
    tl.store(C + row * stride_cn, acc, mask=row < N)

def mv(inp, vec):
    logger.debug("GEMS MV vector-fma")
    assert inp.ndim == 2 and vec.ndim == 1 and inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    # 256 is a good fit for the K3 RVV implementation; the masked loop also
    # handles dimensions that are not multiples of the tile size.
    block = 256 if M >= 128 else 128
    grid = (N,)
    with torch_device_fn.device(inp.device):
        mv_vector_kernel[grid](inp, vec, out, N, M, inp.stride(0), inp.stride(1),
                               vec.stride(0), out.stride(0), BLOCK_M=block,
                               num_warps=1)
    return out
