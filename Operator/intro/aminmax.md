ATen 契约：`torch.aminmax(Tensor self, int? dim=None, bool keepdim=False) -> (Tensor min, Tensor max)`。

实现功能：计算输入张量的最小值和最大值，并返回包括这二者的元组
注册:`("_aminmax", _aminmax)`、`("aminmax", aminmax)`|
分级:L1 / 规约与扫描

备注：
入口 `def aminmax(inp, dim=None, keepdim=False, *, out=None)`，三条路径：

|路径|触发条件|
|---|---|
|`aminmax_kernel_1` \+ `aminmax_kernel_2`|`dim is None`：两阶段全归约|
|`aminmax_kernel`|指定 `dim`：`BLOCK_M × BLOCK_N` 二维分块|
|输出整理|`keepdim=False` 时 `squeeze(dim=dim)`|

bf16 累加升 fp32：`acc_type = tl.float32 if dtype is tl.bfloat16 else dtype`。填充值取 dtype 极值（`get_dtype_max` / `get_dtype_min`）。

**典型用例**（来源：FlagGems `tests/test_aminmax.py`，`REDUCTION_SHAPES[1]` × `float16`）

```yaml
operator: Aminmax
input_shape:
- [4096, 256]
dtype: [float16]
attrs: {dim: 1, keepdim: false}
value_range: [-1, 1]
note: "M-float16-1M-对齐-2D-dim=1"
```

一次调用返回 `(min, max)` 两个张量；`dim=1` 沿列归约，输出 shape `[4096]`。
