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
#
# Candidate: aminmax_iter5_o1_candA  (plan AMINMAX-13, base =
# aminmax_iter3_o1_candB -- iter5 baseline, official mean 2.2604 ms,
# insn 22.47M, IPC 0.617, bw 0.94 GB/s, L1d miss 1.13% on [4096,256] fp16
# dim=1; bottleneck = Utilization, NOT bandwidth).
#
# AMINMAX-13 (P0, target: Utilization):
#   Generic N>BLOCK_N path restructured to pure vector state updates with
#   ONE axis=1 reduction after the loop:
#     * load once per tile with other=-inf (== the max identity);
#     * MAX chain updates as a bare `_max = tl.maximum(_max, a)` -- the
#       baseline's extra `tl.where(mask, ..., _max)` select on this chain
#       is dead work, because masked lanes already hold -inf from the
#       load fill (removes one [BLOCK_M, BLOCK_N] select per tile);
#     * MIN chain keeps its identity via fill-then-update
#       `_min = tl.minimum(_min, tl.where(mask, a, +inf))` -- the where
#       now depends only on the load + mask, not on the reduced state,
#       shortening the loop-carried dependency;
#     * the per-loop horizontal tl.min/tl.max stay eliminated (already
#       the baseline structure); a single tl.min/tl.max(axis=1) pair runs
#       after the N-loop.
#   NO additional independent accumulator chains are introduced (iter4
#   proved 2x-chain ILP is a regression on K3: insn 46.7M, IPC 0.589).
#   BLOCK_M=8 / BLOCK_N=256 / num_warps=4 unchanged (BLOCK_M<=4
#   miscompiles on the spine backend).
#   The single-load static kernel (N in 256/512) is kept as the control
#   path: structure unchanged, only the grid-cap row loop below is added.
#
# Grid-cap row loop (correctness, all kernels that index rows/blocks):
#   Every launch is now grid = min(cdiv(M, BLOCK_M), 512) <= 512; the
#   kernel loops over row groups g: rows = (pid + g*512)*BLOCK_M + arange.
#   This FIXES the generic/static paths for M > 4096 (e.g. M=4097:
#   cdiv(4097,8)=513 > 512 was silently dropped by the K3 backend ->
#   uninitialized/NaN output).  dim=None stage-1 gets the same cap+loop.
#   Also carries the guarded per-dim squeeze from AMINMAX-15 so torch-
#   conformant pre-squeezed out= tensors no longer raise IndexError
#   (baseline bug; host-side only, zero kernel impact).
#
# K3 invariants preserved from the baseline (spine backend, verified):
#   * BLOCK_M / BLOCK_MB >= 8 always.
#   * NO scalar (0-d) loads/stores, NO scalar pointer arithmetic; all
#     stores are vector pointers + masks.
#   * Globals read by @triton.jit kernels are tl.constexpr.
#   * fp16/bf16 -> fp32 right after load; +/-inf identity via tl.where.
#   * No autotune / libtuner; @libentry(), ext.program_id, torch_device_fn.
# Risk: dropping the max-chain select relies on vfmax(_max, -inf)==_max;
#   for NaN lanes K3 tl.min/max do not propagate NaN either way (matches
#   baseline bit-for-bit on the NaN tests per the baseline's K3 notes);
#   register pressure of the group loop is unchanged (sequential groups).
import flag_gems  # noqa: F401  K3: registers the Triton CPU driver.
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# fp32 reduction identities, as constexpr so @triton.jit kernels may read them.
_POS_INF = tl.constexpr(float("inf"))
_NEG_INF = tl.constexpr(float("-inf"))

# K3-verified safe launch geometry: reduction tiles need BLOCK_M >= 8 on the
# spine backend; BLOCK_N=256 keeps a 2048-element tile.
_BLOCK_M = 8
_BLOCK_N = 256
_NUM_WARPS = 4
_BLOCK_MB = 8  # stage-1 / fused (dim=None) rows-per-program; tile [8, BLOCK_SIZE]
# AMINMAX-8: number of independent column segments in the static-N kernel.
_SEGMENTS = 2
# K3 hard constraint: the backend silently drops launches with grid > 512.
_GRID_CAP = 512
# (M, N, dtype) -> precomputed grid tuple for the static-N fast kernel.
_FAST_GRID_CACHE = {}
_FAST_GRID_CACHE_MAX = 64


@libentry()
@triton.jit
def aminmax_kernel_1(
    inp,
    min_out,
    max_out,
    M,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_MB: tl.constexpr,
    GRID_CAP: tl.constexpr,
):
    """Stage-1 partial reduction for dim=None.  fp32, +/-inf identity.
    Each program reduces BLOCK_MB consecutive length-BLOCK_SIZE blocks of
    the flat input per group iteration; the group loop covers
    mid_size > GRID_CAP*BLOCK_MB so the launch grid stays <= 512.
    No scalar memory access.
    """
    pid = ext.program_id(0)
    num_groups = (mid_size + BLOCK_MB * GRID_CAP - 1) // (BLOCK_MB * GRID_CAP)
    for g in range(0, num_groups):
        blocks = (pid + g * GRID_CAP) * BLOCK_MB + tl.arange(0, BLOCK_MB)
        elems = tl.arange(0, BLOCK_SIZE)
        offs = blocks[:, None] * BLOCK_SIZE + elems[None, :]
        mask = offs < M
        a = tl.load(inp + offs, mask=mask, other=_NEG_INF).to(tl.float32)
        min_v = tl.where(mask, a, _POS_INF)
        max_v = tl.where(mask, a, _NEG_INF)
        mn = tl.min(min_v, axis=1)
        mx = tl.max(max_v, axis=1)
        bmask = blocks < mid_size
        dtype = inp.type.element_ty
        tl.store(min_out + blocks, mn.to(dtype), mask=bmask)
        tl.store(max_out + blocks, mx.to(dtype), mask=bmask)


@libentry()
@triton.jit
def aminmax_kernel_2(
    min_inp, max_inp, min_out, max_out, mid_size, BLOCK_MID: tl.constexpr
):
    """Stage-2 final reduction for dim=None.  fp32, +/-inf identity.
    The scalar result is broadcast into a length-8 vector and written with a
    masked vector store through a pure vector pointer (lane 0 only): scalar
    stores / scalar pointer arithmetic crash spine-opt on K3.
    """
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    min_a = tl.load(min_inp + offset, mask=mask, other=_NEG_INF).to(tl.float32)
    max_a = tl.load(max_inp + offset, mask=mask, other=_NEG_INF).to(tl.float32)
    mn = tl.min(tl.where(mask, min_a, _POS_INF), axis=0)
    mx = tl.max(tl.where(mask, max_a, _NEG_INF), axis=0)
    lane = tl.arange(0, 8)
    dtype = min_inp.type.element_ty
    mn_vec = tl.zeros([8], tl.float32) + mn
    mx_vec = tl.zeros([8], tl.float32) + mx
    tl.store(min_out + lane, mn_vec.to(dtype), mask=lane < 1)
    tl.store(max_out + lane, mx_vec.to(dtype), mask=lane < 1)


@libentry()
@triton.jit
def aminmax_fused_kernel(
    inp,
    min_out,
    max_out,
    M,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_MB: tl.constexpr,
):
    """Fused single-program dim=None reduction (whole flat input fits one
    [BLOCK_MB, BLOCK_SIZE] tile).  Unchanged from the baseline."""
    blocks = tl.arange(0, BLOCK_MB)
    elems = tl.arange(0, BLOCK_SIZE)
    offs = blocks[:, None] * BLOCK_SIZE + elems[None, :]
    mask = offs < M
    a = tl.load(inp + offs, mask=mask, other=_NEG_INF).to(tl.float32)
    min_v = tl.where(mask, a, _POS_INF)
    max_v = tl.where(mask, a, _NEG_INF)
    mn_rows = tl.min(min_v, axis=1)
    mx_rows = tl.max(max_v, axis=1)
    mn = tl.min(mn_rows, axis=0)
    mx = tl.max(mx_rows, axis=0)
    lane = tl.arange(0, 8)
    dtype = inp.type.element_ty
    mn_vec = tl.zeros([8], tl.float32) + mn
    mx_vec = tl.zeros([8], tl.float32) + mx
    tl.store(min_out + lane, mn_vec.to(dtype), mask=lane < 1)
    tl.store(max_out + lane, mx_vec.to(dtype), mask=lane < 1)


@libentry()
@triton.jit
def aminmax_kernel(
    inp,
    min_out,
    max_out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRID_CAP: tl.constexpr,
):
    """AMINMAX-13 generic per-row min/max reduction.

    Vector state update form: the load fills masked lanes with -inf (the
    max identity), so the max chain is a bare tl.maximum state update with
    NO trailing select; the min chain fills +inf via tl.where on the
    loaded tile (depending only on load+mask, not on the carried state).
    One tl.min/tl.max(axis=1) pair after the loop.  The outer group loop
    keeps grid <= GRID_CAP while covering M > GRID_CAP*BLOCK_M rows.
    """
    dtype = inp.type.element_ty
    pid = ext.program_id(0)
    num_groups = (M + BLOCK_M * GRID_CAP - 1) // (BLOCK_M * GRID_CAP)
    for g in range(0, num_groups):
        rows = (pid + g * GRID_CAP) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        base = inp + rows[:, None] * N
        _min = tl.full([BLOCK_M, BLOCK_N], value=_POS_INF, dtype=tl.float32)
        _max = tl.full([BLOCK_M, BLOCK_N], value=_NEG_INF, dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)
            mask = row_mask[:, None] & (cols[None, :] < N)
            a = tl.load(base + cols[None, :], mask=mask, other=_NEG_INF).to(
                tl.float32
            )
            # masked lanes are -inf == max identity: no select needed here.
            _max = tl.maximum(_max, a)
            _min = tl.minimum(_min, tl.where(mask, a, _POS_INF))
        mn = tl.min(_min, axis=1)
        mx = tl.max(_max, axis=1)
        tl.store(min_out + rows, mn.to(dtype), mask=row_mask)
        tl.store(max_out + rows, mx.to(dtype), mask=row_mask)


@libentry()
@triton.jit
def aminmax_single_kernel(
    inp,
    min_out,
    max_out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Single-load multi-row reduction (BLOCK_N >= N, no N-loop).
    Retained verbatim from the baseline as a valid single-load fallback;
    dispatch prefers the static kernel.  (AMINMAX-13 control path.)
    """
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    cols = tl.arange(0, BLOCK_N)
    offs = rows[:, None] * N + cols[None, :]
    mask = row_mask[:, None] & (cols[None, :] < N)
    a = tl.load(inp + offs, mask=mask, other=_NEG_INF).to(tl.float32)
    min_v = tl.where(mask, a, _POS_INF)
    max_v = tl.where(mask, a, _NEG_INF)
    mn = tl.min(min_v, axis=1)
    mx = tl.max(max_v, axis=1)
    dtype = inp.type.element_ty
    tl.store(min_out + rows, mn.to(dtype), mask=row_mask)
    tl.store(max_out + rows, mx.to(dtype), mask=row_mask)


@libentry()
@triton.jit
def aminmax_static_kernel(
    inp,
    min_out,
    max_out,
    M,
    N: tl.constexpr,
    SEGMENTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GRID_CAP: tl.constexpr,
):
    """AMINMAX-8 static-N segmented per-row min/max (control path).

    Unchanged from the baseline except for the outer group loop: program
    pid handles row groups (pid + g*GRID_CAP) so the launch grid stays
    <= GRID_CAP for M > GRID_CAP*BLOCK_M (fixes M=4097-style silent
    grid drops).  N stays constexpr (addresses constant-fold, column mask
    is compile-time all-true); SEGMENTS=2 explicit segment chains are
    combined once; a single tl.min/tl.max(axis=1) pair finishes each row.
    BLOCK_M=8 (K3-verified; <=4 miscompiles, 16/32 regress).
    """
    tl.static_assert(SEGMENTS == 2)
    SEG_W: tl.constexpr = N // SEGMENTS
    dtype = inp.type.element_ty
    pid = ext.program_id(0)
    cols0 = tl.arange(0, SEG_W)
    cols1 = SEG_W + cols0
    num_groups = (M + BLOCK_M * GRID_CAP - 1) // (BLOCK_M * GRID_CAP)
    for g in range(0, num_groups):
        rows = (pid + g * GRID_CAP) * BLOCK_M + tl.arange(0, BLOCK_M)
        base = inp + rows[:, None] * N
        if (pid + g * GRID_CAP + 1) * BLOCK_M <= M:
            # Full-rows fast path: unmasked loads/stores, no where-identity.
            a0 = tl.load(base + cols0[None, :]).to(tl.float32)
            a1 = tl.load(base + cols1[None, :]).to(tl.float32)
            _min = tl.minimum(a0, a1)
            _max = tl.maximum(a0, a1)
            mn = tl.min(_min, axis=1)
            mx = tl.max(_max, axis=1)
            tl.store(min_out + rows, mn.to(dtype))
            tl.store(max_out + rows, mx.to(dtype))
        else:
            # Tail path: row mask only; +/-inf identity via where AFTER load.
            row_mask = rows < M
            a0 = tl.load(
                base + cols0[None, :],
                mask=row_mask[:, None],
                other=_NEG_INF,
            ).to(tl.float32)
            a1 = tl.load(
                base + cols1[None, :],
                mask=row_mask[:, None],
                other=_NEG_INF,
            ).to(tl.float32)
            min0 = tl.where(row_mask[:, None], a0, _POS_INF)
            max0 = tl.where(row_mask[:, None], a0, _NEG_INF)
            min1 = tl.where(row_mask[:, None], a1, _POS_INF)
            max1 = tl.where(row_mask[:, None], a1, _NEG_INF)
            _min = tl.minimum(min0, min1)
            _max = tl.maximum(max0, max1)
            mn = tl.min(_min, axis=1)
            mx = tl.max(_max, axis=1)
            tl.store(min_out + rows, mn.to(dtype), mask=row_mask)
            tl.store(max_out + rows, mx.to(dtype), mask=row_mask)


def aminmax(inp, dim=None, keepdim=False, *, out=None):
    logger.debug("GEMS AMINMAX")
    # Host fast dispatch for the benchmark hot case -- contiguous 2D
    # (M, N) with N in (256, 512), dim=+/-1, keepdim=False, out=None.
    # Grid is now capped at _GRID_CAP (kernel loops over row groups), so
    # any M is safe here.  All other inputs fall through to the baseline
    # logic below.
    if (
        out is None
        and not keepdim
        and dim is not None
        and inp.dim() == 2
        and inp.size(1) in (256, 512)
        and inp.size(0) > 0
        and inp.is_contiguous()
        and (
            dim == 1
            or dim == -1
            or dim == [1]
            or dim == [-1]
            or dim == (1,)
            or dim == (-1,)
        )
    ):
        M = inp.size(0)
        N = inp.size(1)
        key = (M, N, inp.dtype)
        grid = _FAST_GRID_CACHE.get(key)
        if grid is None:
            if len(_FAST_GRID_CACHE) >= _FAST_GRID_CACHE_MAX:
                _FAST_GRID_CACHE.clear()
            grid = (min(_GRID_CAP, triton.cdiv(M, _BLOCK_M)),)
            _FAST_GRID_CACHE[key] = grid
        dtype = inp.dtype
        min_out = torch.empty((M,), dtype=dtype, device=inp.device)
        max_out = torch.empty((M,), dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            aminmax_static_kernel[grid](
                inp,
                min_out,
                max_out,
                M,
                N=N,
                SEGMENTS=_SEGMENTS,
                BLOCK_M=_BLOCK_M,
                GRID_CAP=_GRID_CAP,
                num_warps=_NUM_WARPS,
            )
        return min_out, max_out
    if dim is None:
        M = inp.numel()
        # Clamp: stage-1 tile is [8, block_size]; K3-verified-safe tiles
        # need BLOCK_N >= 256 columns (small tiles miscompile on spine).
        block_size = max(_BLOCK_N, triton.next_power_of_2(math.ceil(math.sqrt(M))))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        if out is not None:
            min_out = out[0] if isinstance(out, tuple) else out
            max_out = out[1] if isinstance(out, tuple) else out
            if not keepdim:
                min_out = min_out.squeeze()
                max_out = max_out.squeeze()
        else:
            if not keepdim:
                min_out = torch.empty([], dtype=dtype, device=inp.device)
                max_out = torch.empty([], dtype=dtype, device=inp.device)
            else:
                shape = [1] * inp.dim()
                min_out = torch.empty(shape, dtype=dtype, device=inp.device)
                max_out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            if mid_size <= _BLOCK_MB:
                # Whole input fits one [BLOCK_MB, block_size] tile: fused
                # single-program reduction, one launch total.
                aminmax_fused_kernel[(1,)](
                    inp,
                    min_out,
                    max_out,
                    M,
                    BLOCK_SIZE=block_size,
                    BLOCK_MB=_BLOCK_MB,
                    num_warps=_NUM_WARPS,
                )
            else:
                min_mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
                max_mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
                aminmax_kernel_1[(min(_GRID_CAP, triton.cdiv(mid_size, _BLOCK_M)),)](
                    inp,
                    min_mid,
                    max_mid,
                    M,
                    mid_size,
                    BLOCK_SIZE=block_size,
                    BLOCK_MB=_BLOCK_MB,
                    GRID_CAP=_GRID_CAP,
                    num_warps=_NUM_WARPS,
                )
                aminmax_kernel_2[(1,)](
                    min_mid,
                    max_mid,
                    min_out,
                    max_out,
                    mid_size,
                    BLOCK_MID=block_mid,
                    num_warps=_NUM_WARPS,
                )
        return min_out, max_out
    else:
        if isinstance(dim, int):
            dim = [dim]
        # FIX: baseline used a generator (always truthy) -> all(...).
        assert all(i >= -inp.ndim and i < inp.ndim for i in dim), "Invalid dim"
        dtype = inp.dtype
        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = dim_compress(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N
        if out is not None:
            min_out = out[0] if isinstance(out, tuple) else out
            max_out = out[1] if isinstance(out, tuple) else out
        else:
            min_out = torch.empty(shape, dtype=dtype, device=inp.device)
            max_out = torch.empty(shape, dtype=dtype, device=inp.device)
        grid = (min(_GRID_CAP, triton.cdiv(M, _BLOCK_M)),)
        with torch_device_fn.device(inp.device):
            # Static-N fast path for the benchmark's dense rows (control
            # path, unchanged apart from the grid cap).  Any other shape
            # uses the AMINMAX-13 generic kernel.
            if M == 4096 and N in (256, 512) and inp.is_contiguous():
                aminmax_static_kernel[grid](
                    inp,
                    min_out,
                    max_out,
                    M,
                    N=N,
                    SEGMENTS=_SEGMENTS,
                    BLOCK_M=_BLOCK_M,
                    GRID_CAP=_GRID_CAP,
                    num_warps=_NUM_WARPS,
                )
            else:
                aminmax_kernel[grid](
                    inp,
                    min_out,
                    max_out,
                    M,
                    N,
                    BLOCK_M=_BLOCK_M,
                    BLOCK_N=_BLOCK_N,
                    GRID_CAP=_GRID_CAP,
                    num_warps=_NUM_WARPS,
                )
        if not keepdim:
            # Guarded per-dim squeeze (fix ported from AMINMAX-15): squeeze
            # only dims that still exist with size 1, so a torch-conformant
            # pre-squeezed out= no longer raises IndexError; allocated
            # buffers (size-1 reduced dims) squeeze exactly as before.
            for d in sorted(dim, reverse=True):
                if d < min_out.dim() and min_out.size(d) == 1:
                    min_out = min_out.squeeze(d)
                if d < max_out.dim() and max_out.size(d) == 1:
                    max_out = max_out.squeeze(d)
        return min_out, max_out
