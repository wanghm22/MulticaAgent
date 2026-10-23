# Layernorm — Memory 维度优化

Roofline 视角：layernorm 每元素只有几次乘加，算术强度极低，性能上限就是带宽。Memory 维度的目标是减少 x 的读趟数和中间结果的落盘。Layernorm 与 softmax 同属"行内多趟归约 + 逐元素"骨架，但多一趟 var、多一个 weight/bias 读，访存账要单独算。本章把 layernorm 在 memory 维度的优化前后代码完整摆出来，逐处说明哪里改了什么、省了多少。

## 优化前：三趟扫描 + 每趟重读 x

朴素融合版已经是"一个 kernel 干完"，但行内仍分三趟，每趟都从内存重新读 x：

```python
@triton.jit
def layernorm_naive_kernel(X, Y, W, B, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * N
    Y += row * N

    # 趟1: mean —— 读 x 一遍
    acc = tl.zeros([BLOCK], tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        acc += x
    mean = tl.sum(acc, axis=0) / N

    # 趟2: var —— 再读 x 一遍
    vacc = tl.zeros([BLOCK], tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        vacc += x * x
    var = tl.sum(vacc, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # 趟3: 归一化 + 仿射 —— 第三遍读 x，再读 w/b，写 y
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y + cols, y, mask=mask)
```

访存账（每行 N 元素、f16 每元素 2B，w/b 各 2B）：x 被读三遍 = 3N×2B；w、b 各读一遍 = 2N×2B；y 写一遍 = N×2B。合计读 5N×2B、写 N×2B，**六趟 N 元素访存**。这就是 memory 维度的基线。

## 优化一：ONE_TILE 整行驻留寄存器

当 N 小于等于 BLOCK 时（BLOCK 取 next_power_of_2(N)，官方启发式还封顶 64KB/元素大小），一趟 load 把整行装进寄存器，mean、var、归一化全部复用寄存器里的 x，不再回内存读：

```python
@triton.jit
def layernorm_one_tile_kernel(X, Y, W, B, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # ★ 唯一一次读 x：整行进寄存器
    x = tl.load(X + row * N + cols, mask=mask, other=0.).to(tl.float32)

    mean = tl.sum(x, axis=0) / N                       # 复用 x
    xc = tl.where(mask, x - mean, 0.)                  # 复用 x
    var = tl.sum(xc * xc, axis=0) / N                  # 复用 xc
    rstd = 1 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=mask).to(tl.float32)    # w 只读一次
    b = tl.load(B + cols, mask=mask).to(tl.float32)    # b 只读一次
    y = xc * rstd * w + b                              # 复用 xc
    tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=mask)
```

访存账变成：x 读 N×2B + w 读 N×2B + b 读 N×2B + y 写 N×2B，**四趟**，x 的三遍重读全部消失。相比 naive 版六趟，省掉的是 2N×2B 的 x 重读。省趟的本质是 x、xc、y 都是寄存器里的 SSA 值，编译器不会把它们落回内存。

## 优化二：带 stride 的行首指针算术

优化前每处偏移都写 `X + row * stride + cols`：

```python
x = tl.load(X + row * stride + off + cols, mask=..., other=0.)
```

优化后先把指针移到行首，后续只写行内偏移：

```python
row = tl.program_id(0)
X += row * stride          # 指针算术，移到本行开头
Y += row * stride
# 之后所有 X + cols / Y + cols 都是行内偏移
```

这不改变访存趟数，但让行内访存模式规整、地址计算少一层乘法，也避免 stride 参与每个 lane 的偏移计算。是 memory 维度"每趟访存地址计算最简"的优化。

## 优化三：mask 填充值选择，避免额外归零遍

layernorm 趟2 里越界 lane 的处理是 memory 与 compute 的交界：

```python
# 优化前：只靠 other=0.，越界 lane 的 (0 - mean)² 会虚增方差
x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
vacc += x * x

# 优化后：显式 tl.where 把越界 lane 的贡献归零
x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
x = tl.where(cols < N, x - mean, 0.)
vacc += x * x
```

这里 tl.where 不增加访存（纯寄存器操作），但它保证了 var 正确，避免"结果错了再重算"的隐性 memory 代价。与 rmsnorm 对照：rmsnorm 的 `other=0.` 平方后天然是 0，不需要 where；layernorm 的 `(0-mean)²` 不吸收 0，必须 where。这是 mask 填充值是否被后续运算吸收的经典对照（ops/02 §2.4、ops/03 §3.3）。

## 优化四：multi-tile 路径的 f32 累加器

N 装不下单 block 时走 multi-tile，结构回到多趟，但每趟都用向量累加器、最后只归约一次，且累加器是 f32：

```python
_mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
for off in range(0, N, BLOCK_SIZE):
    cols = off + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    _mean += x                     # 纯向量加，无水平归约
mean = tl.sum(_mean, axis=0) / N   # 整行只归约一次
```

memory 角度：每趟 x 只读一次、不留中间 buffer；utilization 角度：归约只做一次。f32 累加器还避免了 f16 长行累加的精度崩坏（ops/06 §6.5）。

## 优化五：backward 需要的 mean/rstd 缓存，避免重算

```python
tl.store(Mean + row, mean)
tl.store(Rstd + row, rstd)
```

forward 顺手把每行的 mean/rstd 写进两个 (M,) 的小数组，backward 直接用，不必重新扫一遍 x 算统计量。memory 账：多写 2M 个数（每行 2 个标量，相对 N×M 元素可忽略），省掉 backward 的一整趟 x 读。这是"forward 多写一点点、backward 少读一整趟"的 memory 优化。

## 优化前后访存对照

| 形态 | x 读 | w/b 读 | y 写 | 总访存（N 元素） | 适用 |
|---|---|---|---|---|---|
| naive 三趟 | 3N | 2N | N | 6N | 教学反例 |
| ONE_TILE | N | 2N | N | 4N | N ≤ BLOCK |
| multi-tile | 3N | 2N | N | 6N | N 超单 block |

ONE_TILE 是 layernorm 在 memory 维度的最优解，官方 BLOCK 启发式 `min(65536 // x.element_size(), next_power_of_2(N))` 就是为了让常见行宽命中 4N 形态。

## 验证 memory 优化生效

dump IR 数 .linalgdir 里 transfer_read/transfer_write 次数：ONE_TILE 版 x 应只出现一次 read；对比 naive 版同 shape 墙钟，memory-bound 下 ONE_TILE 应明显更快；加 N=1000、BLOCK=1024 的越界 lane 用例，确认 tl.where 归零让 var 精确。