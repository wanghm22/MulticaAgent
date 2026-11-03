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
"""
P66 topk-chunk16 (iteration 4).

Same pinned-pool multicore design as P65 (per-worker ``os.sched_setaffinity``
pinning + ``torch.set_num_threads(1)``), but with **16 chunks** instead of 8:
each of the 8 pinned workers processes 2 contiguous chunks (1,048,576 / 16 =
65,536 elements per chunk) and the main thread merges 16*k = 160 candidates.

The hypothesis is cache-driven. K3 (SpacemiT X100, 8 AI CPU cores) has **no
last-level cache**: a 131,072-element fp16 chunk (P65) is 256 KiB and spills out
of the per-core L2 while ``torch.topk`` scans it; a 65,536-element fp16 chunk
(P66) is 128 KiB, which fits comfortably in a core's L2, so each
``torch.topk`` runs hotter and the two-chunk sequence per worker can complete in
less wall-time than one double-size chunk. The total scan work is identical to
P65 (same N, same k), so any win is purely from improved cache residency on a
machine without LLC.

Cost of the finer split: the merge stage selects k out of 160 instead of 80 --
still sub-millisecond (160 elements, k=10), so it does not dominate. Each worker
doing two chunks serially means the chunk stage wall-time is ~2x a single
65,536-element topk rather than 1x a 131,072-element topk; if cache locality
makes the smaller topk more than 2x faster per element, P66 beats P65, otherwise
P65 wins -- the Profiler decides.

Correctness is identical to P65/P49: the global top-k is contained in the union
of the per-chunk top-k, so concatenating the 16 per-chunk top-k and re-selecting
recovers the exact global top-k values; indices are global (chunk offset added)
and returned as ``torch.int64``.

Interface: ``topk(x, k, dim=-1, largest=True, sorted=True) -> (values, indices)``
matches ``Operator/code/topk.py`` and ``Operator/benchmark/topk_benchmark.py``.
"""

import itertools
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

import flag_gems  # noqa: F401  -- registers the spine triton driver at import

logger = logging.getLogger(__name__)

# Cores this process is permitted to use (K3 normally exposes all 8 AI cores).
try:
    _AVAIL_CORES = sorted(os.sched_getaffinity(0))
except AttributeError:  # pragma: no cover - non-Linux fallback
    _AVAIL_CORES = list(range(os.cpu_count() or 8))

_NUM_WORKERS = max(1, min(8, len(_AVAIL_CORES)))
_PIN_CORES = _AVAIL_CORES[:_NUM_WORKERS]
# 16 chunks on the 8-core K3 -> 2 chunks per pinned worker. Hard-coded to match
# the plan; the _chunked_topk guard falls back to a single torch.topk when N is
# too small to feed 16 chunks of >= k elements.
_NUM_CHUNKS = 16

torch.set_num_threads(1)

_pool_lock = threading.Lock()
_thread_pool = None
_pin_counter = itertools.count()
_assign_lock = threading.Lock()


def _pin_worker():
    """Pool initializer: pin this worker thread to a distinct available core."""
    if not _PIN_CORES:
        return
    with _assign_lock:
        slot = next(_pin_counter) % _NUM_WORKERS
    core = _PIN_CORES[slot]
    try:
        os.sched_setaffinity(0, {core})
    except (OSError, AttributeError):
        pass


def _get_pool():
    global _thread_pool
    if _thread_pool is None:
        with _pool_lock:
            if _thread_pool is None:
                _thread_pool = ThreadPoolExecutor(
                    max_workers=_NUM_WORKERS, initializer=_pin_worker
                )
    return _thread_pool


def _local_topk(chunk, k, dim, largest, sorted_flag, offset):
    """Top-k on one chunk under inference_mode; returns (values, global_idx)."""
    with torch.inference_mode():
        vals, local_idx = torch.topk(
            chunk, k, dim=dim, largest=largest, sorted=sorted_flag
        )
    return vals, local_idx + offset


def _chunked_topk(x, k, dim, largest, sorted_flag, num_chunks):
    """Parallel chunked top-k along ``dim`` (general ndim)."""
    ndim = x.dim()
    if dim < 0:
        dim += ndim
    n = x.shape[dim]

    # Each chunk must hold at least k elements, otherwise torch.topk(chunk, k)
    # raises; for small inputs the chunked path has no benefit anyway.
    if n < num_chunks * k or k > n:
        with torch.inference_mode():
            return torch.topk(x, k, dim=dim, largest=largest, sorted=sorted_flag)

    chunks = torch.chunk(x, num_chunks, dim=dim)

    offsets = [0]
    for c in chunks:
        offsets.append(offsets[-1] + c.shape[dim])

    n_chunks = len(chunks)
    pool = _get_pool()
    futs = [
        pool.submit(_local_topk, chunks[i], k, dim, largest, sorted_flag, offsets[i])
        for i in range(n_chunks)
    ]
    vals = [None] * n_chunks
    idxs = [None] * n_chunks
    for i, f in enumerate(futs):
        vals[i], idxs[i] = f.result()

    # Merge 16*k = 160 candidates (vs 80 in P65). Still tiny: k-of-160.
    with torch.inference_mode():
        cat_vals = torch.cat(vals, dim=dim)
        cat_idx = torch.cat(idxs, dim=dim)
        merge_vals, merge_sel = torch.topk(
            cat_vals, k, dim=dim, largest=largest, sorted=sorted_flag
        )
        final_idx = cat_idx.gather(dim, merge_sel)
    return merge_vals, final_idx


def topk(x, k, dim=-1, largest=True, sorted=True):
    """Return the k largest (or smallest) elements of x along dim.

    Multicore chunked ``torch.topk`` (16 chunks, pinned pool, 2 chunks/worker)
    for floating-point inputs; integer/bool inputs defer to a single
    ``torch.topk``. Returns (values, indices) with indices as torch.int64.
    """
    logger.debug("GEMS TOPK (P66 chunk16 pinned)")
    if x.is_floating_point():
        return _chunked_topk(x, k, dim, largest, sorted, _NUM_CHUNKS)
    return torch.topk(x, k, dim=dim, largest=largest, sorted=sorted)
