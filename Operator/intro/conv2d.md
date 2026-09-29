ATen 契约：`torch.conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[] stride=[1,1], int[] padding=[0,0], int[] dilation=[1,1], int groups=1) -> Tensor`。

实现功能：
注册：`("conv2d", conv2d)`、`("conv2d.padding", conv2d)`
分级：L3 / Convolution

备注：入口三条分支：

|padding 类型|处理|
|---|---|
|int / tuple|直接 `Conv2d.apply(...)`|
|`"same"`|断言 `stride==1`，`ceil(...)` 计算输出尺寸，裁剪 `[..., (oh-ih):, (ow-iw):]`|
|`"valid"`|转 `padding=0` 后 `Conv2d.apply(...)`|

`class Conv2d(torch.autograd.Function)` forward 中：`tl.dot` 要求 `K≥16`，当 `weight_c < 16` 时自动补通道（groups=1 直接 pad，groups\>1 按组 reshape 后 pad）。内核按 `BLOCK_NI_HO_WO × BLOCK_CI × BLOCK_CO` 分块。

**典型用例**（来源：CANNBench `tasks/level3/conv_2d/cases.yaml`，case\_id 1）

```yaml
operator: Conv2D
case_id: 1
input_shape:
- [2, 64, 32, 32]    # x  [N, C_in, H, W]
- [64, 64, 3, 3]     # filter [C_out, C_in, Kh, Kw]
- [64]               # bias [C_out]
dtype: [float16, float16, float16]
attrs:
  strides: [1, 1]
  pads: [1, 1, 1, 1]
  dilations: [1, 1]
value_range:
- [-1, 1]
- [-1, 1]
- [-0.1, 0.1]
note: "S-float16-256K-对齐-3x3k-same_pad | baseline_kernels: TransData×2 + MemSet×1 + Conv2D×1"
```

same padding 下输出 `[2,64,32,32]`，约 131K 元素；3×3 卷积，无 dilation，bias 必填。