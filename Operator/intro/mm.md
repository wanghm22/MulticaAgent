ATen 契约：`torch.mm(Tensor self, Tensor mat2) -> Tensor`。


实现功能：通过分组重排，分块计算累加等方式并行计算矩阵乘法
注册：`("mm", mm)`、`("mm.out", mm_out)`
分级：L3 / BLAS\-3

备注：四级分发链：

|分支|触发条件|
|---|---|
|`syrk_mm`|`is_syrk_transpose_pair(a, b)`：输出对称矩阵，下三角启动域省一半计算|
|`streamk_mm`|`capability[0]==8`（NVIDIA 8\.x）且 fp16/bf16 且 `K > M*5` 且 `K > N*5`|
|`cluster_remote_mm`|集群跨卡远程 dot|
|`general_mm`|兜底，`GROUP_M=8` L2 分组调度，支持 fp64 全精度累加|

`get_higher_dtype` 在 `[fp16, bf16, fp32, fp64]` 上取较高者作为输出 dtype；断言 `a.shape[1]==b.shape[0]`；只做 2D × 2D。

**典型用例**（来源：FlagGems `tests/test_mm.py`，`MNK_SHAPES[1]`）

```yaml
operator: Mm
input_shape:
- [15, 160]      # a (M=15, K=160)
- [160, 1024]    # b (K=160, N=1024)
dtype: [float16]
value_range: [-1, 1]
note: "M-float16-对齐-GEMM-M=15-K=160-N=1024"
```

输出 shape `[15, 1024]`；走 `general_mm` 路径，fp32 累加。