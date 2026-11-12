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

logger = logging.getLogger(__name__)


def topk(x, k, dim=-1, largest=True, sorted=True):
    """
    K3 架构兼容的 topk 实现。

    原 FlagGems Triton 两阶段 kernel（topk_stage1_kernel /
    topk_stage2_kernel）在 K3 的 spine-triton 后端编译时会触发
    `ptr.to_ptr` 的 memorySpace 属性解析错误，导致运行时失败。
    在 SpacemiT 后端正式适配 topk 之前，这里直接回退到 PyTorch
    原生 topk，保证算子可运行且语义与 torch.topk 完全一致。

    参数与返回值与 torch.topk 一致：
      - x:        输入张量
      - k:        保留的元素个数
      - dim:      沿哪个维度取 topk，默认 -1
      - largest:  True 取最大，False 取最小
      - sorted:   是否对结果排序
    返回 (values, indices) 元组。
    """
    logger.debug("GEMS TOPK (K3 fallback to torch.topk)")

    # 规范化 dim，与 PyTorch 行为保持一致
    if dim < 0:
        dim = dim + x.ndim

    assert dim == x.ndim - 1, "Currently only support topk in last dimension"

    # 处理 k == 0 的边界情况，返回空张量
    if k == 0:
        out_shape = list(x.shape[:-1]) + [0]
        return (
            torch.empty(out_shape, device=x.device, dtype=x.dtype),
            torch.empty(out_shape, device=x.device, dtype=torch.int64),
        )

    return torch.topk(x, k, dim=dim, largest=largest, sorted=sorted)