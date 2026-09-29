ATen 契约：`torch.abs(Tensor self) -> Tensor`，另有原地版 `abs_`。

实现功能：将输入的张量所有元素取绝对值，逐元素操作不涉及元素交互
注册：`("abs", abs)`、`("abs_", abs_)`、`("absolute", absolute)`、`("absolute_", absolute_)`
分级：L1 / 逐元素

备注：`promotion_methods=[(0, "COMPLEX_TO_FLOAT")]`：复数输入取模变浮点；标量/任意维走通用路径。

**典型用例**（来源：FlagGems `https://github.com/flagos-ai/FlagGems/tree/master/tests/test_abs.py`，`POINTWISE_SHAPES[2]` × `float32`）

```yaml
operator: Abs
input_shape:
- [1024, 1024]
dtype: [float32]
value_range: [-10, 10]
note: "M-float32-4M-对齐-2D"
```
