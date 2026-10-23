# Layernorm — Compute 维度优化

Roofline 视角：layernorm 算术强度低，compute 不是瓶颈，但计算正确性决定有效算力。本维度关注数值稳定化、mask 越界 lane 的贡献归零、dtype 提升、避免冗余算术与前端编译坑。本章把 layernorm 在 compute 维度的优化前后代码完整摆出来。

## 优化前：mask 填充值直接参与非线性运算

```python
@triton.jit
def layernorm_naive_compute(X, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.)   # 填 0
    mean = tl.sum(x, axis=0) / N
    var = tl.sum((x - mean) * (x - mean), axis=0) / N      # 越界 lane 的 (0-mean)² 虚增方差
    rstd = 1 / tl.sqrt(var + eps)
    y = (x - mean) * rstd
    tl.store(Y + row * N + cols, y, mask=mask)
```

问题不在溢出，而在**越界 lane 的填充值 0 被 `(x - mean)²` 非线性放大**：mask 让越界 lane 读成 0，但 `(0 - mean)²` 不是 0，会虚增方差。结果随越界 lane 数（BLOCK - N）偏移，是数据依赖的静默错误。

## 优化一：tl.where 把越界 lane 的贡献归零

```python
x = tl.load(X + row * N + cols, mask=mask, other=0.).to(tl.float32)
mean = tl.sum(x, axis=0) / N
xc = tl.where(mask, x - mean, 0.)          # ★ 越界 lane 强制贡献 0
var = tl.sum(xc * xc, axis=0) / N
```

这是 layernorm 在 compute 维度的头号项。对照 rmsnorm：rmsnorm 的 `other=0.` 平方后天然是 0，不需要 where；layernorm 的 `(0-mean)²` 不吸收 0，必须显式归零。凡是"mask 填的值参与非线性运算"的场合都要多想一步——求和填 0 没事，平方、exp 就有事。

## 优化二：方差用两趟 (x-mean)²，而非单趟 E[x²]-mean²

两种写法：

```python
# 优化前：单趟 E[x²] − mean²，省一趟访存但有消去误差
ex2 = tl.sum(x * x, axis=0) / N
var = ex2 - mean * mean                     # 大均值小方差时两个大数相减

# 优化后：两趟 (x-mean)²，稳
xc = tl.where(mask, x - mean, 0.)
var = tl.sum(xc * xc, axis=0) / N
```

单趟形式在大均值小方差时有消去误差（两个大数相减损失有效位），两趟形式稳。生产 kernel（var/var_mean）选稳的。compute 维度上这是"用多一点的访存换数值可信"的取舍，layernorm 因 memory-bound 本就跑不满算力，选稳的更划算。

## 优化三：load 后立刻升 f32，归约不丢精度

```python
# 优化前：fp16 直接归约，tl.sum 不提升 dtype
x = tl.load(X + cols, mask=..., other=0.)
acc += x                                     # fp16 部分和，长行误差大

# 优化后：load 后立刻升 f32
x = tl.load(X + cols, mask=..., other=0.).to(tl.float32)
acc += x                                     # f32 累加
```

tl.sum 不会自动提升 dtype，fp16 tile 直接 sum 得到 fp16 部分和，长行下误差超容差。这是 K3 端口层修过的真 bug（ops/02 §2.5 坑 1）。compute 维度上，"在正确的精度上累加"是保证结果有效的前提。

## 优化四：tl.sqrt 只收 f32/fp64，rstd 计算天然安全

```python
rstd = 1 / tl.sqrt(var + eps)
```

var 已经是 f32 标量，tl.sqrt 合法。但若对 f16 向量直接 tl.sqrt 会撞 `_check_dtype`。compute 维度上要记住：math 函数（exp/log/sqrt）只收 f32/fp64 是上游故意设计，kernel 里所有喂给它们的值必须已升 f32。

## 优化五：避免 constexpr 整数的 .to() 前端崩

```python
# 优化前：constexpr 整数直调 .to()
a = tl.full([], 2, tl.int32)
b = a.to(tl.float32)           # 'int' object has no attribute 'to'

# 优化后：用乘法代替 .pow(2)，或用 tl.full 造浮点常量
b = a * a
# 或
two = tl.full([], 2.0, tl.float32)
```

constexpr 整数没有 .to()，`(2).to(tl.float32)` 前端报 `'int' object has no attribute 'to'`（ops/02 §2.5 坑 4）。compute 维度上这是"避免前端编译失败"的优化，写 `a * a` 代替 `a.pow(2)` 最省心。

## 优化六：block-ptr store 显式 cast，避免类型不匹配崩

```python
# 优化前：block-ptr store 无隐式 cast，f32 值写 f16 block 报错
tl.store(block_ptr, f32_val)   # Block element type(fp16) and value element type(fp32) mismatch

# 优化后：显式 cast
tl.store(block_ptr, f32_val.to(Y.type.element_ty))
```

普通指针 store 有隐式 cast，block-ptr store 没有——两种 store 行为不一致，必须分清（ops/02 §2.5 坑 3）。compute 维度上这是"让写出的值类型与目标匹配"的正确性优化。

## 优化前后 compute 对照

| 项 | naive | 优化后 | 收益 |
|---|---|---|---|
| 越界 lane 贡献 | (0-mean)² 虚增方差 | tl.where 归零 | 方差精确 |
| 方差写法 | 单趟 E[x²]-mean² | 两趟 (x-mean)² | 无消去误差 |
| 归约 dtype | f16 | f32 | 长行误差可控 |
| sqrt 输入 | 可能 f16 | f32 标量 | 合法 |
| constexpr 算术 | .to() 崩 | a*a / tl.full | 前端不崩 |
| block-ptr store | 隐式 cast 崩 | 显式 .to() | 类型匹配 |

## 验证 compute 优化生效

取 N=1000、BLOCK=1024（24 个越界 lane），删掉 tl.where 重跑，预测方差偏大还是偏小并验证；golden 用 `torch.nn.functional.layer_norm(x.float(), ...)` 再降回 f16，atol/rtol 1e-2；删掉 .to(tl.float32) 用 N=8192 跑 f16，观察误差随行长滚大。

## 三个维度的关系

tl.where 归零既保 compute 正确也避免重算的隐性 memory 代价；f32 提升既是 compute 精度要求也影响 utilization 的 lane 宽度。调优顺序仍是 memory 趟数先行、utilization 并行和归约形态其次、compute 精度与稳定性收尾。