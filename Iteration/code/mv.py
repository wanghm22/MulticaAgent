# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""U2 candidate: two-way unrolled, interleaved-load GEMV.

This is a standalone replacement for ``mv.py``.  The launch configuration and
the two-dimensional fp32 accumulator intentionally remain identical to the
baseline; only the reduction loop is unrolled by two.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mv"),
    key=["M", "N"],
    strategy=["align32", "align32"],
    flagtune_op_name="mv",
)
@triton.jit
def mv_u2_kernel(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = ext.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_m = tl.arange(0, BLOCK_M)[None, :]
    n_mask = offset_n < N
    A_ptrs = A + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B + offset_m * stride_bm

    # Keep the baseline's tile accumulator and fp32 arithmetic.  Each
    # iteration computes two independent BLOCK_M chunks, interleaving loads
    # before the FMAs to expose memory-level parallelism.
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    step = 2 * BLOCK_M
    for m in range(0, M, step):
        m0_mask = m + offset_m < M
        a0 = tl.load(A_ptrs, mask=n_mask & m0_mask, other=0.0).to(tl.float32)
        b0 = tl.load(B_ptrs, mask=m0_mask, other=0.0).to(tl.float32)

        # The second half may be a partial tail.  Its mask is independently
        # checked, which is required for odd M/BLOCK_M combinations.
        A_ptrs1 = A_ptrs + BLOCK_M * stride_am
        B_ptrs1 = B_ptrs + BLOCK_M * stride_bm
        m1_mask = m + BLOCK_M + offset_m < M
        a1 = tl.load(A_ptrs1, mask=n_mask & m1_mask, other=0.0).to(tl.float32)
        b1 = tl.load(B_ptrs1, mask=m1_mask, other=0.0).to(tl.float32)

        acc += a0 * b0
        acc += a1 * b1
        A_ptrs += step * stride_am
        B_ptrs += step * stride_bm

    acc = tl.sum(acc, axis=1)
    C_ptrs = C + offset_n * stride_cn
    tl.store(C_ptrs, acc[:, None], mask=n_mask)


def mv(inp, vec):
    logger.debug("GEMS MV (U2 unroll2 interleaved load)")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch_device_fn.device(inp.device):
        mv_u2_kernel[grid](
            inp,
            vec,
            out,
            N,
            M,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
        )
    return out

