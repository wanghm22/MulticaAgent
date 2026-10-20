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

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x):
    return tl.abs(x)


@triton.jit
def _abs_contiguous_kernel(inp, out, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    """Branch-free, coalesced fast path for the common dense real tensor case."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(inp + offsets, mask=mask)
    tl.store(out + offsets, tl.abs(values), mask=mask)


def abs(A):
    logger.debug("GEMS ABS")
    # A fixed wide tile exposes more independent loads/stores than the generic
    # pointwise launcher on K3.  Keep complex and strided tensors on the generic
    # path so its promotion and indexing rules remain authoritative.
    if A.is_contiguous() and not A.is_complex() and A.numel() != 0:
        out = A.new_empty(A.shape)
        n_elements = A.numel()
        _abs_contiguous_kernel[(triton.cdiv(n_elements, 1024),)](
            A, out, n_elements, BLOCK=1024, num_warps=8
        )
        return out
    return abs_func(A)


def abs_(A):
    logger.debug("GEMS ABS_")
    if A.is_contiguous() and not A.is_complex() and A.numel() != 0:
        n_elements = A.numel()
        _abs_contiguous_kernel[(triton.cdiv(n_elements, 1024),)](
            A, A, n_elements, BLOCK=1024, num_warps=8
        )
        return A
    abs_func(A, out0=A)
    return A
