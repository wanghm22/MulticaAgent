ATen 契约：`_softmax(Tensor self, int dim, bool half_to_float) -> Tensor`（`torch.softmax` 是其封装）。


实现功能：沿指定维度对输入张量做 Softmax 归一化
注册：`("_softmax", softmax)`、`("_softmax.out", softmax_out)`，及对应反向
分级：L1 / 规约与扫描


备注：其他契约：`numel()==0` 时输出 `zero_`；`half_to_float=True` 时输出强制 fp32；`out` 版校验 dtype。同时实现了反向内核。

**典型用例**（来源：CANNBench `tasks/level2/softmax/cases.yaml`，case\_id 1）

```yaml
operator: Softmax
case_id: 1
input_shape:
- [1024, 1024]
dtype: [float16]
attrs: {dim: -1}
value_range: [-1, 1]
note: "S-float16-1M-对齐-对称小值域-dim=-1 | baseline_kernels: SoftmaxV2×1"
```

沿最末维归约，走 `softmax_kernel_inner`（`K==1`）路径。
 
