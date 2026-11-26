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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x):
    return tl.abs(x)


# I2-ABS-O0-CONTIGOWN: eight disjoint contiguous CPU program intervals.
# 8 programs * 32 chunks * 4096 elements = 1024 * 1024 exactly.
@triton.jit
def dense_abs_persistent_kernel(inp, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    for j in range(32):
        offs = (pid * 32 + j) * BLOCK + lane
        value = tl.load(inp + offs)
        tl.store(out + offs, tl.abs(value))


def abs(A):
    logger.debug("GEMS ABS")
    if (
        A.device.type == "cpu"
        and A.layout == torch.strided
        and A.dtype == torch.float32
        and tuple(A.shape) == (1024, 1024)
        and A.is_contiguous()
        and not A.is_conj()
        and not A.is_neg()
        and not A.requires_grad
    ):
        out = torch.empty_like(A)
        with torch_device_fn.device(A.device):
            dense_abs_persistent_kernel[(8,)](A, out, BLOCK=4096, num_warps=4)
        return out
    return abs_func(A)


def abs_(A):
    logger.debug("GEMS ABS_")
    abs_func(A, out0=A)
    return A