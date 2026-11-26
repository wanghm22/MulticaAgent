# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
def mv_kernel(
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
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        m_mask = m + offset_m < M
        a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b
        A_ptrs += BLOCK_M * stride_am
        B_ptrs += BLOCK_M * stride_bm

    acc = tl.sum(acc, axis=1)
    C_ptrs = C + offset_n * stride_cn
    tl.store(C_ptrs, acc[:, None], mask=n_mask)


# I1-MV-O2-DUALACC: exact-shape K3 CPU specialization.
@triton.jit
def _mv_o2_dualacc_kernel(A, B, C):
    rows = tl.program_id(0) * 8 + tl.arange(0, 8)
    cols = tl.arange(0, 256)
    acc0 = tl.zeros((8, 256), dtype=tl.float32)
    acc1 = tl.zeros((8, 256), dtype=tl.float32)
    for j in tl.static_range(0, 2):
        k0 = j * 512 + cols
        k1 = k0 + 256
        # Each vector segment is read once and shared across eight rows.
        a0 = tl.load(A + rows[:, None] * 1024 + k0[None, :]).to(tl.float32)
        b0 = tl.load(B + k0).to(tl.float32)
        a1 = tl.load(A + rows[:, None] * 1024 + k1[None, :]).to(tl.float32)
        b1 = tl.load(B + k1).to(tl.float32)
        acc0 += a0 * b0[None, :]
        acc1 += a1 * b1[None, :]
    result = tl.sum(acc0 + acc1, axis=1)
    tl.store(C + rows, result.to(C.dtype.element_ty))


def _mv_o2_eligible(inp, vec):
    return (
        inp.device.type == "cpu"
        and inp.device == vec.device
        and inp.layout == torch.strided and vec.layout == torch.strided
        and inp.dtype == torch.float16 and vec.dtype == torch.float16
        and tuple(inp.shape) == (160, 1024) and tuple(vec.shape) == (1024,)
        and inp.is_contiguous() and vec.is_contiguous()
        and not inp.requires_grad and not vec.requires_grad
        and not inp.is_conj() and not vec.is_conj()
        and not inp.is_neg() and not vec.is_neg()
    )


def mv(inp, vec):
    logger.debug("GEMS MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    if _mv_o2_eligible(inp, vec):
        with torch_device_fn.device(inp.device):
            _mv_o2_dualacc_kernel[(20,)](inp, vec, out, num_warps=4)
        return out
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch_device_fn.device(inp.device):
        mv_kernel[grid](
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