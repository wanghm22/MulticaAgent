ATen 契约：`torch.topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)`。


实现功能：找出最大的k个数以及它们的索引
注册：`("topk", topk)`
分级：L1 / 排序与选择

备注：内核选择逻辑（按 k 和数据规模）：

- **bitonic sort**：k 较小、数据量适中时

- **radix select**：k 较大或数据量大时（TLE 大 fp32 场景有专用 heuristic）

- **Ascend DSA 路径**：`vendor_name=="ascend"` 时走专用分支，支持到 `(128, 131072)` 规模

返回 `(values, indices)`；索引一致性通过 `x.gather(1, indices) ≈ values` 校验。

**典型用例**（来源：CANNBench `tasks/level3/top_k/cases.yaml`，case\_id 1）

```yaml
operator: TopK
case_id: 1
input_shape:
- [1048576]
dtype: [float16]
attrs: {k: 10, dim: -1, largest: true}
value_range: [-1, 1]
note: "S-float16-1M-对齐-1D-k=10-dim=-1-largest"
```

1D 输入取最大 10 个值及其索引；走通用 radix\-select 内核。
