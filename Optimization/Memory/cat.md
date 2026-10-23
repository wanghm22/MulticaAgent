# Cat / Concat — Memory 维度优化

Roofline 视角：cat 没有任何计算，只有搬运，性能上限就是内存带宽，kernel 的全部难点在"输出位置 ← 输入位置"的映射。Memory 维度的目标是输出每位置恰好写一次、输入每段只读一次、无额外中间 buffer。本章把 cat 在 memory 维度的优化前后代码完整摆出来。

## 优化前：host 循环逐输入 launch 纯 copy kernel

```python
@triton.jit
def copy_kernel(Src, Dst, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(Src + offs, mask=offs < N, other=0.)
    tl.store(Dst + offs, x, mask=offs < N)

def cat_host_loop(a, b):
    c = torch.empty((a.shape[0] + b.shape[0], a.shape[1]), dtype=a.dtype)
    copy_kernel[(triton.cdiv(a.numel(), 1024),)](a, c[:a.shape[0]], a.numel(), BLOCK=1024)
    copy_kernel[(triton.cdiv(b.numel(), 1024),)](b, c[a.shape[0]:], b.numel(), BLOCK=1024)
    return c
```

访存账：a 读一遍写一遍、b 读一遍写一遍，总访存 = 2(a.numel + b.numel)，理论上就是最小。但 host 循环 K 个输入 K 次 launch，memory 上不坏，坏在 dispatch（utilization 篇展开）。memory 上它是"每段纯 copy"的基线。

## 优化一：单 kernel mask-select，两路 load 相加即选择

```python
@triton.jit(do_not_specialize=["M_A", "M_B", "N"])
def cat0_kernel(A, B, C, M_A, M_B, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)                 # 输出行号，0 .. M_A+M_B-1
    col = tl.program_id(1)                 # 列块号
    offs_n = col * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    va = tl.load(A + row * N + offs_n,
                 mask=n_mask & (row < M_A), other=0.)
    vb = tl.load(B + (row - M_A) * N + offs_n,
                 mask=n_mask & (row >= M_A), other=0.)
    tl.store(C + row * N + offs_n, va + vb, mask=n_mask)
```

memory 账：每次输出位置写一次；A 路和 B 路各 load 一次，但被 mask 的那一路根本不发访存（mask load 语义：被 mask 的 lane 直接取 other，不发访存）。所以总访存仍是 a.numel + b.numel 读、c.numel 写，没有额外中间 buffer。va + vb 是纯寄存器选择，不产生 memory 开销。

## 优化二：mask 掉的 lane 绝不访存，负偏移不解引用

```python
vb = tl.load(B + (row - M_A) * N + offs_n,
             mask=n_mask & (row >= M_A), other=0.)
```

row < M_A 时 B 路的 mask 全 False，(row - M_A) 这个负偏移根本不会被解引用——Triton 的 mask load 语义是被 mask 的 lane 直接取 other、不发访存。初学者常以为要先 if row >= M_A 保护偏移计算，不需要，mask 就是保护。memory 角度，这是"不为无效 lane 发访存"的优化。

## 优化三：va + vb 选择，不引入额外读写

```python
tl.store(C + row * N + offs_n, va + vb, mask=n_mask)
```

任何时刻恰有一路是真实值、另一路是 other=0.，相加即选择。搬运类 kernel 的标准无分支技巧（对 int/float 都成立；若是 bool 要换 |）。memory 上 va + vb 是纯寄存器操作，不产生额外 memory 开销。

## 优化四：grid 按输出切，输出每位置写一次

```python
cat0_kernel[(M_A + M_B, triton.cdiv(N, BLOCK_N))](
    a, b, c, M_A, M_B, N, BLOCK_N=BLOCK_N)
```

grid=(M_A+M_B, cdiv(N, BLOCK_N))，行数 128+64=192、列块若干，grid 总量安全。按输出切块，输出每个位置恰好写一次，天然无冲突，不需要同步或 atomic。memory 上这是"输出每位置一次写"的最优形态。

## 优化五：非连续输入先 contiguous 或走 stride 版

```python
# 优化前：kernel 内假设输入连续，a[:, ::2] 这类输入错值
# 优化后：host 侧先 contiguous，或 kernel 传全套 stride
a = a.contiguous()          # host 侧保证连续
```

非连续输入时，要么 host 侧先 contiguous() 或走 stride 版偏移（生产实现传全套 stride 参数），要么 fallback copy_。永远不要在 kernel 里假设输入连续。memory 上这是"非连续访问模式可控"的优化。

## 优化六：do_not_specialize，避免 M_B=1 折成 constexpr

```python
@triton.jit(do_not_specialize=["M_A", "M_B", "N"])
def cat0_kernel(...):
    ...
```

M_B=1 之类的值不该被折成 constexpr。memory 上不直接影响趟数，但避免特化路径带来的调度不一致。

## 优化前后访存对照

| 形态 | 读 | 写 | 中间 buffer | 适用 |
|---|---|---|---|---|
| host 循环 copy | a+b | c | 无 | K 小、最稳 |
| 单 kernel mask-select | a+b（mask 路不发访存） | c | 无 | 2~3 个输入 |
| 单 kernel if/elif 选路 | a+b | c | 无 | K 大（编译器支持不全） |

## 验证 memory 优化生效

跑 cat.py 的 3 组形状（2x2+1x2、100x256+37x256、64x1000+64x1000），判据 torch.equal 逐位相等；非连续输入 a[:, ::2] 验证 fallback 或 stride 版；dump IR 数 transfer_read 确认 mask 路不发访存。