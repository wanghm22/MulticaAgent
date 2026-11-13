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

import flag_gems  # noqa: F401  (register CPU backend; per iter3 rule)
from flag_gems import runtime
from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_k = ext.program_id(1)
    pid_m = ext.program_id(0)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        mask = (n_offsets[:, None] < N) & (k_offsets < K)
        input_ptrs = input_ptr + offset
        # Reduce in fp32: some triton backends (e.g. Cambricon MLU) reject
        # fp16/bf16 in tl.exp, and fp32 accumulation is more accurate anyway.
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, 0)
        e = tl.exp(inp - m[None, :])
        z = tl.sum(e, 0)
        out = (e / z).to(output_ptr.dtype.element_ty)
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        # specialization does not improve performance inn this example, as tested
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)  # (TILE_K,)
        z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)  # (TILE_K, )
        m = m_reduced

        # specialization does not improve performance inn this example, as tested
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m[None, :]) / z[None, :]
            tl.store(output_ptr + offsets, o, mask=mask)


@triton.jit
def next_multiple_of(a, b):
    # the smallest x>=a that x%b ==0
    return tl.cidv(a, b) * b


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        # Reduce in fp32: some triton backends (e.g. Cambricon MLU) reject
        # fp16/bf16 in tl.exp, and fp32 accumulation is more accurate anyway.
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = (e / z).to(output_ptr.dtype.element_ty)
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        output_ptr += pid_m * N

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            m_new = tl.maximum(m, inp)
            # it is possible that there are -inf's in the input
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new
        # specialize the last iteration
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        previous_multiple = prev_multiple_of(N, TILE_N)
        # specialize the first iteration
        for start_n in range(0, TILE_N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(
                input_ptr + n_offsets,
                mask=mask,
                other=-float("inf"),
                eviction_policy="evict_first",
            )
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o, mask=mask)
        for start_n in range(TILE_N, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first")
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o)


# Candidate: iter3_P42_softmax_two_pass
# Plan P42: classic stable fp32 two-pass softmax per row (max pass, then
# normalize pass; NO online rescaling) with several rows' load/reduce state
# merged into one program.  Each CTA owns TILE_M rows and walks them as
# TILE_M/2 independent row PAIRS: both rows of a pair are loaded up front,
# then their max -> exp -> sum -> store dependency chains are interleaved in
# the loop body.  Two independent chains in flight per iteration fill the
# reduction dependency bubbles (Judge: stallB 65.9%, IPC 0.345 from the
# serial max->exp->sum chains of the single-row structure).
# Measured on K3 (1024x1024 fp16): this dual-chain mask-free structure at
# TILE_M=16, grid=64, num_warps=4 is the fastest known point (1.12 ms vs
# 1.34 ms rowbatch baseline).  2D masked tiles are forbidden (11-21 ms).
# EVEN constexpr specialization: when the wrapper guarantees N == BLOCK_N
# and M % TILE_M == 0, ALL masks are dropped (unmasked 1D vector
# loads/stores) -- the measured fast path.  The general path keeps full
# tail masks (n < N and r < M), so any N (non-pow2, < 16 via BLOCK_N>=16)
# and any M are correct.
# K3 launch guard: the spine launcher silently drops programs when a grid
# dim exceeds ~512; the wrapper caps every launch at <= 256 programs.
# All-(-inf) rows: m is clamped to 0 (scalar where, free on spine) so
# e = exp(-inf) = 0 and z = 0, then e/z = 0/0 = NaN -- identical to
# torch.softmax's all-(-inf) behavior. NB: a vector tl.where(z==0, 0, e/z)
# guard was measured to cost 3x on spine (3.03ms vs 1.13ms) -- do NOT re-add.
@libentry()
@triton.jit
def softmax_kernel_inner_two_pass(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = ext.program_id(0)
    n_offsets = tl.arange(0, BLOCK_N)
    nmask = n_offsets < N
    for i in range(TILE_M // 2):
        r0 = pid * TILE_M + 2 * i
        r1 = r0 + 1
        offs0 = r0 * N + n_offsets
        offs1 = r1 * N + n_offsets
        if EVEN:
            # iter3_p2_fp16_tile_max (seat O1): keep the loaded tile in its
            # native dtype (fp16) instead of materialising a 4 KB f32 tile for
            # the max pass.  The row max then reduces the 2-byte tile at 16
            # lanes (f32 would be 8) and the tile is widened exactly once, at
            # the exp argument.  fp16->fp32 widening is exact and order
            # preserving, so max(fp16 tile) == max(widened tile) bit for bit.
            inp0 = tl.load(input_ptr + offs0)
            inp1 = tl.load(input_ptr + offs1)
            # Interleaved reduction chains: row0 and row1 max in flight.
            m0 = tl.max(inp0, 0).to(tl.float32)
            m1 = tl.max(inp1, 0).to(tl.float32)
            m0 = tl.where(m0 == float("-inf"), 0.0, m0)
            m1 = tl.where(m1 == float("-inf"), 0.0, m1)
            e0 = tl.exp(inp0.to(tl.float32) - m0)
            e1 = tl.exp(inp1.to(tl.float32) - m1)
            z0 = tl.sum(e0, 0)
            z1 = tl.sum(e1, 0)
            # P2 (iter1_p2_reciprocal): one scalar fp32 reciprocal per row
            # plus a vector fmul, replacing the 32-lane vector fdiv per chunk.
            # r0/r1 MUST stay fp32 scalars (never f16 / never approximate).
            # All-(-inf) rows: z=0 -> r=+inf, e=0 -> 0*inf = NaN, identical to
            # the baseline's 0/0 = NaN (verified, not assumed).
            r0 = 1.0 / z0
            r1 = 1.0 / z1
            o0 = e0 * r0
            o1 = e1 * r1
            tl.store(output_ptr + offs0, o0.to(output_ptr.dtype.element_ty))
            tl.store(output_ptr + offs1, o1.to(output_ptr.dtype.element_ty))
        else:
            mask0 = nmask & (r0 < M)
            mask1 = nmask & (r1 < M)
            # iter3_p2_fp16_tile_max (seat O1): same fp16-tile treatment as
            # the EVEN path.  other=-inf is representable in fp16, so the
            # masked tail still reduces to the true row max; the widening is
            # deferred to the exp argument only.
            inp0 = tl.load(input_ptr + offs0, mask=mask0, other=-float("inf"))
            inp1 = tl.load(input_ptr + offs1, mask=mask1, other=-float("inf"))
            m0 = tl.max(inp0, 0).to(tl.float32)
            m1 = tl.max(inp1, 0).to(tl.float32)
            m0 = tl.where(m0 == float("-inf"), 0.0, m0)
            m1 = tl.where(m1 == float("-inf"), 0.0, m1)
            e0 = tl.exp(inp0.to(tl.float32) - m0)
            e1 = tl.exp(inp1.to(tl.float32) - m1)
            z0 = tl.sum(e0, 0)
            z1 = tl.sum(e1, 0)
            # P2 (iter1_p2_reciprocal): one scalar fp32 reciprocal per row
            # plus a vector fmul, replacing the 32-lane vector fdiv per chunk.
            # r0/r1 MUST stay fp32 scalars (never f16 / never approximate).
            # All-(-inf) rows: z=0 -> r=+inf, e=0 -> 0*inf = NaN, identical to
            # the baseline's 0/0 = NaN (verified, not assumed).
            r0 = 1.0 / z0
            r1 = 1.0 / z1
            o0 = e0 * r0
            o1 = e1 * r1
            tl.store(
                output_ptr + offs0, o0.to(output_ptr.dtype.element_ty), mask=mask0
            )
            tl.store(
                output_ptr + offs1, o1.to(output_ptr.dtype.element_ty), mask=mask1
            )


# Fallback for shapes the pair kernel must not take (N > 4096, or M so large
# that ceil(M / TILE_M) would exceed the 256-program launch cap): per-row
# STREAMING two-pass over TILE_N chunks (pass 1: online max/sum state, pass
# 2: normalize + store), rows walked TILE_M at a time with an outer stride
# so grid stays <= 256 for arbitrarily large M.  Not benchmarked; correctness
# only.  Tail chunks and tail rows are fully masked.
@libentry()
@triton.jit
def softmax_kernel_inner_rowbatch_stream(
    output_ptr,
    input_ptr,
    M,
    N,
    NUM_CTAS,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid_m = ext.program_id(0)
    for rb in range(pid_m * TILE_M, M, NUM_CTAS * TILE_M):
        for i in range(TILE_M):
            r = rb + i
            rmask = r < M
            base = r * N
            # pass 1: streaming max + sum (vector state, reduced at the end)
            m_vec = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
            z_vec = tl.full([TILE_N], value=0.0, dtype=tl.float32)
            for start_n in range(0, N, TILE_N):
                n_offsets = start_n + tl.arange(0, TILE_N)
                mask = (n_offsets < N) & rmask
                inp = tl.load(input_ptr + base + n_offsets, mask=mask, other=-float(
                    "inf"
                )).to(tl.float32)
                m_new = tl.maximum(m_vec, inp)
                all_neg_inf = m_new == float("-inf")
                z_vec = tl.where(
                    all_neg_inf,
                    z_vec,
                    z_vec * tl.exp(m_vec - m_new) + tl.exp(inp - m_new),
                )
                m_vec = m_new
            m = tl.max(m_vec, 0)
            m = tl.where(m == float("-inf"), 0.0, m)
            z = tl.sum(z_vec * tl.exp(m_vec - m), 0)
            # pass 2: normalize + store
            for start_n in range(0, N, TILE_N):
                n_offsets = start_n + tl.arange(0, TILE_N)
                mask = (n_offsets < N) & rmask
                inp = tl.load(input_ptr + base + n_offsets, mask=mask, other=-float(
                    "inf"
                )).to(tl.float32)
                o = tl.where(z == 0, 0.0, tl.exp(inp - m) / z)
                tl.store(
                    output_ptr + base + n_offsets,
                    o.to(output_ptr.dtype.element_ty),
                    mask=mask,
                )


# ------------------------  backward -------------------------------
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("softmax_non_inner"),
    key=[
        "M",
        "N",
        "K",
    ],
)
@triton.heuristics(runtime.get_heuristic_config("softmax_backward_non_inner"))
@triton.jit
def softmax_backward_kernel_non_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    offsets_k = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        mask = (offsets_n < N)[:, None] & (offsets_k < K)
        out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
        scale = tl.sum(out_tile * out_grad_tile, axis=0)
        in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        scale = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            scale += out_tile * out_grad_tile
            offsets_n += TILE_N
            offsets += TILE_N * K
        scale = tl.sum(scale, axis=0)  # (TILE_K)

        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            offsets_n += TILE_N
            offsets += TILE_N * K


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("softmax_inner"),
    key=["M", "N"],
)
@triton.heuristics(
    values=runtime.get_heuristic_config("softmax_backward_inner"),
)
@triton.jit
def softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
        scale = tl.sum(out_tile * out_grad_tile, 1)
        in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        scale = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_last"
            ).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            scale += out_tile * out_grad_tile
            n_offsets += TILE_N
            offsets += TILE_N
        scale = tl.sum(scale, 1)  # (TILE_M,)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_first"
            ).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            n_offsets += TILE_N
            offsets += TILE_N


# =========================================================================
# iter2 - seat O1 - plan iter2_p2_host_dispatch (host-side dispatch cache)
# ONLY host code is changed: the kernels, the geometry rule, BLOCK_N/TILE_M
# values, num_warps, the grid cap (<=256 programs) and every number are the
# baseline's.  No numeric risk.
#
# Why this is worth doing on K3: the official benchmark times the whole Python
# entry point (my_softmax -> softmax -> softmax_out), so the wrapper body is
# inside the measured region.  The baseline body decided, on EVERY call,
# things that only depend on (M, N, dtypes, device): next_power_of_2 + cdiv
# (~7 us measured), tuple(out.shape)/tuple(self.shape) copies (~2 us),
# unconditional self.contiguous(), an always-entered device context, the M/N
# loop, assert and logger.debug.  Measured on RV: module.softmax_out
# 0.806-0.809 ms vs the same kernel launched directly 0.734-0.739 ms.
#
# _LAUNCH_CACHE[(M, N, in_dtype, out_dtype, device)] -> launch geometry, so a
# repeated call does one dict lookup and one launch.  Key completeness
# (Analyzer risk #1): the geometry is a pure function of (M, N); dtypes and
# device are in the key so a dtype/device change can never hit another entry's
# geometry.  Entries are built completely before being stored (risk #3), so a
# concurrent reader never sees a half-initialised tuple.
#
# Output tensors are NEVER cached or reused (Analyzer H18): empty_like costs
# only ~6 us and reusing `out` would break softmax()'s aliasing contract.
#
# torch_device_fn.device (Analyzer risk #2): on the K3 / spacemit CPU backend
# the context manager is a literal no-op -- flag_gems/runtime/backend/
# _spacemit/__init__.py defines _DeviceWrapper.__enter__/__exit__ as `...`
# (verified on RV).  It is kept on every cold path (K > 1, fallback kernel,
# cache miss) and skipped only on the cached fast path.
# =========================================================================
_LAUNCH_CACHE = {}
_LAUNCH_CACHE_MAX = 1024


def _derive_launch_cfg(M, N):
    """Baseline geometry, verbatim: (grid0, tile_m, block_n, even, use_fallback).

    Pure function of (M, N); only called on a cache miss.
    """
    block_n = max(32, triton.next_power_of_2(N))
    tile_m = min(16, max(2, triton.next_power_of_2(triton.cdiv(M, 64))))
    if tile_m % 2 == 1:
        tile_m += 1
    grid0 = triton.cdiv(M, tile_m)
    use_fallback = not (N <= 4096 and grid0 <= 256)
    even = (block_n == N) and (M % tile_m == 0)
    return (grid0, tile_m, block_n, even, use_fallback)


def _launch_rowbatch(out, inp, M, N):
    """Baseline fallback launch (N > 4096 or grid0 > 256). Cold path."""
    with torch_device_fn.device(inp.device):
        tile_m = 4
        num_ctas = min(triton.cdiv(M, tile_m), 256)
        softmax_kernel_inner_rowbatch_stream[(num_ctas, 1, 1)](
            out,
            inp,
            M,
            N,
            num_ctas,
            TILE_N=1024,
            TILE_M=tile_m,
            num_warps=4,
        )

# -------------------------------------------------------------------------
# candB single-point addition on top of candA: also cache the COMPILED kernel
# handle, so the fast path calls CompiledKernel directly instead of going
# through flag_gems' LibEntry.run dispatcher.
#
# Measured on RV (1024x1024 fp16, same process): LibEntry dispatch costs
# ~43 us/call (K[grid](...) 0.734-0.739 ms vs the cached CompiledKernel
# 0.693 ms for the identical kernel and identical output, err 9.5e-07).
# That is the dominant part of the ~66-72 us wrapper overhead -- candA alone
# cannot remove it because the plan's fast path still calls the LibEntry
# object.  Everything else (geometry, values, kernels) is unchanged.
#
# Safety of the cached handle:
#   * The miss path performs exactly ONE launch, through LibEntry.run(...,
#     warmup=False), which is literally what kernel[grid](...) does
#     (triton KernelInterface.__getitem__ == run(grid=..., warmup=False)), and
#     it returns (CompiledKernel, constexprs).  If it does not return a usable
#     handle we fall back to the plain launch and cache kernel=None, which
#     degrades the fast path to candA behaviour.
#   * LibEntry keys its kernel cache on (tensor dtype, data_ptr() % 16 == 0,
#     int arg values, constexprs, num_warps, device).  The cache key below
#     carries the same information -- M, N, both dtypes, device, plus both
#     16-byte alignment flags -- so a handle is only ever reused for the very
#     specialisation it was compiled for.  Int args are M/N (in the key);
#     constexprs are derived from M/N; num_warps is the constant 4.
# -------------------------------------------------------------------------

def softmax_out(self, dim, half_to_float=False, *, out):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("GEMS SOFTMAX_OUT")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    if self.numel() == 0:
        if out.shape != self.shape:
            out.resize_(self.shape)
        zero_(out)
        return out

    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]

    dtype = torch.float32 if half_to_float else self.dtype
    if out.shape != self.shape:
        out.resize_(self.shape)
    if out.dtype != dtype:
        raise RuntimeError(f"_softmax.out: expected out dtype {dtype}, got {out.dtype}")
    K = self.numel() // M // N

    if not self.is_contiguous():
        self = self.contiguous()

    if K > 1:
        grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
        with torch_device_fn.device(self.device):
            softmax_kernel_non_inner[grid](out, self, M, N, K)
        return out

    # Alignment flags mirror LibEntry's `data_ptr() % 16 == 0` tensor
    # specialisation, so a cached CompiledKernel is never reused across
    # specialisations.
    key = (
        M,
        N,
        self.dtype,
        out.dtype,
        self.device,
        self.data_ptr() % 16 == 0,
        out.data_ptr() % 16 == 0,
    )
    entry = _LAUNCH_CACHE.get(key)
    if entry is not None:
        grid0, tile_m, block_n, even, use_fallback, kernel = entry
        if kernel is not None:
            # Direct CompiledKernel launch: skips LibEntry.run's per-call
            # OrderedDict / signature walk / key build / cache probe (~43 us).
            # Args are the kernel's full parameter list in declaration order
            # (output_ptr, input_ptr, M, N, TILE_M, BLOCK_N, EVEN), which is
            # precisely what LibEntry.run passes to kernel[grid].
            kernel[(grid0, 1, 1)](out, self, M, N, tile_m, block_n, even)
            return out
        if use_fallback:
            _launch_rowbatch(out, self, M, N)
            return out
        softmax_kernel_inner_two_pass[(grid0, 1, 1)](
            out,
            self,
            M,
            N,
            TILE_M=tile_m,
            BLOCK_N=block_n,
            EVEN=even,
            num_warps=4,
        )
        return out

    grid0, tile_m, block_n, even, use_fallback = _derive_launch_cfg(M, N)
    kernel = None
    if use_fallback:
        _launch_rowbatch(out, self, M, N)
    else:
        kernel = _launch_two_pass_keep_handle(out, self, M, N, grid0, tile_m, block_n, even)
    if len(_LAUNCH_CACHE) > _LAUNCH_CACHE_MAX:
        _LAUNCH_CACHE.clear()
    _LAUNCH_CACHE[key] = (grid0, tile_m, block_n, even, use_fallback, kernel)
    return out


def softmax(self, dim, half_to_float=False):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("GEMS SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    if self.numel() == 0:
        out = torch.empty(list(self.shape), dtype=self.dtype, device=self.device)
        zero_(out)
        return out

    dtype = torch.float32 if half_to_float else self.dtype
    out = torch.empty_like(self, dtype=dtype)
    return softmax_out(self, dim, half_to_float, out=out)


def _launch_two_pass_keep_handle(out, inp, M, N, grid0, tile_m, block_n, even):
    """One launch, same semantics as kernel[grid](...), returns the handle.

    LibEntry.run(*args, grid=..., warmup=False) performs exactly the launch
    that kernel[grid](*args) performs and additionally returns
    (CompiledKernel, constexprs).  Any surprise (unexpected return shape or an
    exception) is caught; in that case the plain launch below guarantees the
    kernel still runs, and we simply cache kernel=None (candA behaviour).
    """
    launched = False
    kernel = None
    run = getattr(softmax_kernel_inner_two_pass, "run", None)
    if run is not None:
        try:
            res = run(
                out,
                inp,
                M,
                N,
                TILE_M=tile_m,
                BLOCK_N=block_n,
                EVEN=even,
                num_warps=4,
                grid=(grid0, 1, 1),
                warmup=False,
            )
            launched = True
            if isinstance(res, tuple) and res and hasattr(res[0], "__getitem__"):
                kernel = res[0]
        except Exception:
            launched = False
    if not launched:
        softmax_kernel_inner_two_pass[(grid0, 1, 1)](
            out,
            inp,
            M,
            N,
            TILE_M=tile_m,
            BLOCK_N=block_n,
            EVEN=even,
            num_warps=4,
        )
    return kernel


def softmax_backward_out(grad_output, output, dim, input_dtype, *, grad_input):
    logger.debug("GEMS SOFTMAX_BACKWARD_OUT")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    if tuple(grad_input.shape) != tuple(output.shape):
        grad_input.resize_(output.shape)
    if grad_input.dtype != input_dtype:
        raise RuntimeError(
            f"_softmax_backward_data.out: expected grad_input dtype {input_dtype}, got {grad_input.dtype}"
        )
    K = output.numel() // M // N

    with torch_device_fn.device(grad_input.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            softmax_backward_kernel_non_inner[grid](
                output,
                grad_output,
                grad_input,
                M,
                N,
                K,
            )
        else:
            grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
            softmax_backward_kernel_inner[grid](
                output,
                grad_output,
                grad_input,
                M,
                N,
            )
    return grad_input


def softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS SOFTMAX_BACKWARD")
    in_grad = torch.empty_like(output, dtype=input_dtype)
    return softmax_backward_out(
        grad_output, output, dim, input_dtype, grad_input=in_grad
    )
