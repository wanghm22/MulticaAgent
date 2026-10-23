# Reduction — Memory 维度优化

Roofline 视角：归约把 N 个元素读一遍只产出 1 个数，是极端 memory-bound。Memory 维度的目标是数据只从内存过一趟（单 pass），中间留在寄存器。本章把 reduction 在 memory 维度的优化前后代码完整摆出来。

## 优化前：每块先归约再累加标量，中间结果反复落盘

```python
@triton.jit
def row_sum_naive_kernel(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * N
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(X + cols, mask=cols < N, other=0.)
        s += tl.sum(x, axis=0)        # 每块一次水平归约，标量累加
    tl.store(Y + row, s)
```

访存上每块 x 只读一次，看似单 pass，但每块都做一次水平归约，标量累加器 s 在每次归约后被写回。memory 上不额外落盘（s 是寄存器标量），但归约单元反复启停，中间值频繁在 lane 间搬运。

## 优化一：向量累加器 + 最后单次水平归约

```python
@triton.jit(do_not_specialize=["M", "N"])
def row_sum_kernel(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * N

    acc = tl.zeros([BLOCK], dtype=tl.float32)   # 向量累加器（lane 各自累）
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        acc += x                                 # 纯向量加，无水平归约
    tl.store(Y + row, tl.sum(acc, axis=0))       # 最后只归约一次
```

访存账：x 每块读一次、全程只过一趟内存；中间 acc 全程驻留寄存器，水平归约只在行末做一次。这是归约在 memory 维度省趟的标准形态——数据只从内存过一趟，中间留寄存器。

## 优化二：两段式全归约，段1 并行、段2 小数组

全归约（N 元素 → 1 个数）一个 program 串行扫全部 N 太慢。优化前单 program 串行：

```python
@triton.jit
def full_sum_serial_kernel(X, Out, N, BLOCK: tl.constexpr):
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        s += tl.sum(tl.load(X + cols, mask=cols < N, other=0.), axis=0)
    tl.store(Out, s)
```

优化后两段式，段1 完全并行、段2 收小数组：

```python
@triton.jit(do_not_specialize=["N"])
def partial_sum_kernel(X, P, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < N, other=0.).to(tl.float32)
    tl.store(P + pid, tl.sum(x, axis=0))

def full_sum(x):
    N = x.numel()
    BLOCK = 1024
    n_parts = triton.cdiv(N, BLOCK)
    partial = torch.empty(n_parts, dtype=torch.float32)
    partial_sum_kernel[(n_parts,)](x, partial, N, BLOCK=BLOCK)
    return partial.sum()                # 段2：小数组，torch 直接收
```

memory 账：段1 每 program 读自己那段一次、写一个 partial；段2 只读 n_parts 个小标量。总访存仍是 x 过一趟，但并行度拉满。代价是多一次 kernel launch（K3 上 dispatch 开销不可忽略，partial 数组不大时直接 torch.sum 往往最优）。

## 优化三：other=0. 让越界 lane 零贡献，不产生额外读

```python
x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
acc += x          # 越界 lane 填 0，对 sum 零贡献
```

被 mask 的 lane 不发访存，other 只是寄存器值。sum 填 0 天然吸收，越界 lane 不污染结果、不需要额外 where 遍。这是"填充值被归约运算吸收"带来的间接 memory 收益。

## 优化四：f32 累加器，避免 f16 精度崩坏的隐性重算

```python
acc = tl.zeros([BLOCK], dtype=tl.float32)
acc += tl.load(X + cols, mask=..., other=0.).to(tl.float32)
```

f16 直接归约长行误差大（实测 rel 可达 0.6%），结果不可信等于白读。f32 累加器保证结果有效，避免"结果错了再换精度重算"的隐性 memory 代价。

## 优化五：golden 用 float 求和，避免参照系本身不准

```python
# 优化前：f16 输入直接 sum，golden 自己就不准
ref = x.sum()

# 优化后：f16 输入先升 float 再 sum
ref = x.float().sum()
```

这不是 kernel 内的 memory 优化，但属于验证阶段的 memory 账：golden 用 `x.float().sum()` 而不是 `x.sum()`（后者对 f16 输入是 f16 累加，参照系自己就不准）。

## 优化前后访存对照

| 形态 | x 读 | 中间落盘 | 总访存 | 适用 |
|---|---|---|---|---|
| 每块归约标量累加 | N | 无（标量寄存器） | N | 归约单元反复启停 |
| 向量累加器 | N | 无 | N | 单 pass 最优 |
| 两段式全归约 | N + n_parts | partial 小数组 | N + 小 | 全归约并行 |

## 验证 memory 优化生效

dump IR 数 transfer_read，行归约版 x 应只 read 一趟；对比每块归约版与向量累加器版在 N=100000 时的墙钟；加 NaN-guard 切片探针（buf 填 nan、x = buf[:numel].view(shape)），任何越界读都会带 NaN 进输出，逐块扫描定位 OOB。