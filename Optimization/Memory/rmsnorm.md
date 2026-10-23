# RMSNorm — Memory 维度优化

Roofline 视角：rmsnorm 比 layernorm 少一趟归约，访存账更简单，但仍是纯 memory-bound。Memory 维度目标是减少 x 的读趟数、避免中间结果落盘。本章把 rmsnorm 在 memory 维度的优化前后代码完整摆出来。

## 优化前：两趟扫描 + x 重读

朴素融合版行内两趟，每趟重读 x：

```python
@triton.jit
def rmsnorm_naive_kernel(X, Y, W, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * N
    Y += row * N

    # 趟1: sum(x²) —— 读 x 一遍
    acc = tl.zeros([BLOCK], tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        acc += x * x
    ms = tl.sum(acc, axis=0) / N
    rstd = 1 / tl.sqrt(ms + eps)

    # 趟2: y = x * rstd * w —— 再读 x 一遍，读 w，写 y
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        tl.store(Y + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)
```

访存账（每行 N 元素、f16 2B）：x 读两遍 = 2N×2B；w 读一遍 = N×2B；y 写一遍 = N×2B。合计读 3N×2B、写 N×2B，**四趟**。比 layernorm 的六趟少两趟，因为它少一趟归约、没有 bias。

## 优化一：ONE_TILE 整行驻留寄存器

N 小于等于 BLOCK 时一趟 load 整行进寄存器，sum(x²) 和归一化都复用：

```python
@triton.jit
def rmsnorm_one_tile_kernel(X, Y, W, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # ★ 唯一一次读 x：整行进寄存器
    x = tl.load(X + row * N + cols, mask=mask, other=0.).to(tl.float32)

    ms = tl.sum(x * x, axis=0) / N            # 复用 x
    rstd = 1 / tl.sqrt(ms + eps)
    w = tl.load(W + cols, mask=mask).to(tl.float32)   # w 只读一次
    tl.store(Y + row * N + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)
```

访存账变成：x 读 N×2B + w 读 N×2B + y 写 N×2B，**三趟**，x 的重读消失。相比 naive 版四趟，省掉 N×2B 的 x 重读。这是 rmsnorm memory 维度的最优形态，host 启发式 `BLOCK = min(65536 // element_size, next_power_of_2(N))` 就是让它命中。

## 优化二：other=0. 在 rmsnorm 里天然安全

```python
x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
acc += x * x          # 越界 lane 的 0² = 0，天然不贡献
```

与 layernorm 对照：layernorm 趟2 的 `(0 - mean)²` 不吸收 0，必须 tl.where 归零；rmsnorm 的 `0² = 0` 天然吸收，不需要 where。memory 角度，这省掉一遍 tl.where 的寄存器操作，是"mask 填充值被后续运算吸收"带来的间接 memory 收益。

## 优化三：带 stride 的行首指针算术

```python
row = tl.program_id(0)
X += row * stride
Y += row * stride
```

把指针移到行首，后续只写行内偏移。不改变趟数，但让行内访存地址计算少一层乘法、模式规整。

## 优化四：multi-tile 的 f32 向量累加器

N 装不下单 block 时回到多趟，但用 f32 向量累加器、最后只归约一次：

```python
_acc = tl.zeros([BLOCK], dtype=tl.float32)
for off in range(0, N, BLOCK):
    cols = off + tl.arange(0, BLOCK)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    _acc += x * x
ms = tl.sum(_acc, axis=0) / N
```

每趟 x 只读一次、无中间 buffer；f32 累加器避免 f16 长行精度崩坏。

## 优化五：fused_add_rmsnorm 把残差加融进已有趟

Transformer block 里 norm 前总有残差加。优化前两个独立算子：

```python
residual = residual + x          # 一趟读写
y = rmsnorm(residual) * w        # 又两趟读写
```

融合后残差加免费融进 rmsnorm 的趟：

```python
# 趟1: x = residual + x，同时写回新 residual，并累加 x²
acc = tl.zeros([BLOCK], tl.float32)
for off in range(0, N, BLOCK):
    cols = off + tl.arange(0, BLOCK)
    r = tl.load(R + cols, mask=cols < N, other=0.).to(tl.float32)
    xv = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    s = r + xv
    tl.store(R_out + cols, s, mask=cols < N)      # 新 residual 写回
    acc += s * s
ms = tl.sum(acc, axis=0) / N

# 趟2: y = s * rstd * w —— 需要再读一次 residual
```

残差加的中间结果不落额外内存，省掉一个独立 kernel 的完整读写。这是 rmsnorm 在真实 LLM 里最重要的 memory 优化。

## 优化前后访存对照

| 形态 | x/residual 读 | w 读 | y 写 | 总访存 | 适用 |
|---|---|---|---|---|---|
| naive 两趟 | 2N | N | N | 4N | 教学反例 |
| ONE_TILE | N | N | N | 3N | N ≤ BLOCK |
| fused_add | 2N（两趟各读一次） | N | N + N（写新 residual） | 5N | 融合残差 |

## 验证 memory 优化生效

dump IR 数 transfer_read/transfer_write，ONE_TILE 版 x 应只 read 一次；N=1000、BLOCK=1024 用例确认 other=0. 的平方吸收让结果精确不变；fused_add 版对比两个独立算子的墙钟，确认中间结果不落盘。