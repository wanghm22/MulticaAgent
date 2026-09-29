ATen 契约：`torch.mv(Tensor self, Tensor vec) -> Tensor`。

实现功能：通过GPU并行功能实现矩阵向量乘法
注册：`("mv", mv)`
分级：L2 / BLAS\-2

备注：`BLOCK_N × BLOCK_M` 二维 tile，fp32 累加后转回 `inp.dtype`；只支持 2D 矩阵 × 1D 向量，无 batch 维。

**典型用例**（来源：FlagGems `tests/test_mv.py`，`MN_SHAPES[1]`）

```yaml
operator: Mv
input_shape:
- [160, 1024]    # matrix (N=160, M=1024)
- [1024]         # vector
dtype: [float16]
value_range: [-1, 1]
note: "M-float16-对齐-GEMV-N=160-M=1024"
```

输出 shape `[160]`；内核按 `BLOCK_N×BLOCK_M` 分块，fp32 累加后转回 fp16。
