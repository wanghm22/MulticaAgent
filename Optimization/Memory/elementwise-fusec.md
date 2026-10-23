# Elementwise 与融合 — Memory 维度优化

Roofline 视角：逐元素算子读 N 写 N、每元素只算几次乘加，是 memory-bound 的极端。Memory 维度的全部要义是融合：把逐元素链融进相邻大算子，让中间结果不落内存。本章把逐元素与融合在 memory 维度的优化前后代码完整摆出来。

## 优化前：silu 和 mul 分成两个 kernel

```python
@triton.jit
def silu_kernel(X, T, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < N, other=0.).to(tl.float32)
    y = x * tl.sigmoid(x)
    tl.store(T + offs, y.to(T.dtype.element_ty), mask=offs < N)

@triton.jit
def mul_kernel(T, U, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    t = tl.load(T + offs, mask=offs < N, other=0.)
    u = tl.load(U + offs, mask=offs < N, other=0.)
    tl.store(Y + offs, t * u, mask=offs < N)
```

访存账（N 元素、f16 2B）：读 g,u → 写 t=silu(g) → 读 t,u → 写 y，共 **6 趟 N 元素访存**。中间 tensor t 的完整读写是纯浪费。

## 优化一：SwiGLU 融合成一个 kernel

```python
@triton.jit(do_not_specialize=["N"])
def swiglu_kernel(G, U, Y, N, BLOCK: tl.constexpr, MAX_GRID: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * (BLOCK * MAX_GRID)
    for start in tl.range(0, BLOCK * MAX_GRID, BLOCK):
        offs = base + start + tl.arange(0, BLOCK)
        mask = offs < N

        g = tl.load(G + offs, mask=mask, other=0.).to(tl.float32)
        u = tl.load(U + offs, mask=mask, other=0.).to(tl.float32)
        y = (g * tl.sigmoid(g)) * u            # silu(g) * u，全程 f32
        tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)
```

访存账变成：读 g,u → 写 y，共 **3 趟**，中间 t 的完整读写消失，访存直接减半。K3 上算力远富余于带宽，融合省下的每一趟访存都是墙钟。这就是"逐元素算子不值得单独优化，但值得拼命融合"的原因。

## 优化二：epilogue 融合，把逐元素链挂到 GEMM 尾巴

更进一步，SwiGLU 的两个输入来自 gate/up 两个 GEMM。优化前 GEMM 写中间结果、逐元素再读：

```python
# 优化前：GEMM 写 C，再另起 kernel 读 C 做 silu
acc = tl.zeros((BM, BN), dtype=tl.float32)
for k in range(0, K, BK):
    acc += tl.dot(a, b)
tl.store(C_ptrs, acc.to(tl.float16), mask=...)    # 中间结果落盘
# 另一个 kernel 读 C 做 silu 再写
```

优化后 epilogue 在 f32 累加器上直接做，中间不落内存：

```python
acc = tl.zeros((BM, BN), dtype=tl.float32)
for k in range(0, K, BK):
    acc += tl.dot(a, b)                  # ops/01 的 K-loop
# ── epilogue：全在寄存器里的 f32 acc 上做 ──
c = acc + tl.load(BIAS + offs_n)[None, :]     # + bias（广播到每行）
c = c * tl.sigmoid(c)                         # silu
tl.store(C_ptrs, c.to(tl.float16), mask=...)  # 只在最后 cast + 写出
```

memory 账：省掉的是一个 (M, N) f16 中间 tensor 的完整读写，对 memory-bound 的 K3 通常比 epilogue 里多算的几次 sigmoid 贵几个数量级。"中间结果不落内存"是融合的全部要义。

## 优化三：grid 封顶 + 循环吃多 tile，不超运行时上限

优化前直接发超大 grid：

```python
grid = triton.cdiv(N, BLOCK)          # N 大时 grid 可能超上限被静默丢弃
swiglu_kernel[(grid,)](...)
```

优化后封顶 + kernel 内循环：

```python
BLOCK = 1024
MAX_GRID = 512                         # 运行时的 session tile 上限
grid = triton.cdiv(N, BLOCK * MAX_GRID)   # grid 永远 ≤ MAX_GRID
swiglu_kernel[(grid,)](g, u, y, N, BLOCK=BLOCK, MAX_GRID=MAX_GRID)
```

旧版 wheel 对超大 grid 静默丢弃，输出 99% 垃圾值；封顶后一个 program 吃 512K 元素 = 1MB f16×2 读，循环内每次仍是 BLOCK=1024 连续访存，访存模式不变但调度收益（少 6 个 program vs 3072 个）在 dispatch 开销主导的场景反而更快。MAX_GRID 必须作为 constexpr 参数传进 kernel，模块级全局量 jit 函数读不到。

## 优化四：一维扁平化，连续访存最快

```python
N = g.numel()          # 不关心原始 shape
```

逐元素算子对 layout 无感知，扁平处理最简单也最快（连续访存）。不引入 reshape 带来的 stride 变化，保持访存模式最规整。

## 优化五：mask 的 other 会参与计算但被 store mask 挡住

```python
g = tl.load(G + offs, mask=mask, other=0.).to(tl.float32)
u = tl.load(U + offs, mask=mask, other=0.).to(tl.float32)
y = (g * tl.sigmoid(g)) * u            # 越界 lane 算 silu(0)*0=0
tl.store(Y + offs, y, mask=mask)       # 结果被 store mask 挡住，写不出去
```

逐元素 kernel 的 mask 语义与归约不同：归约的填充值会被吸收进结果，逐元素不会——越界 lane 算什么值都无所谓，store mask 挡掉即可。memory 角度，这避免为越界 lane 做额外处理。

## 优化前后访存对照

| 形态 | 读 | 写 | 总访存 | 适用 |
|---|---|---|---|---|
| silu + mul 分开 | g,u,t,u | t,y | 6N | 教学反例 |
| SwiGLU 融合 | g,u | y | 3N | 逐元素二元融合 |
| epilogue 融合 | GEMM 输入 + bias | y | GEMM 输入 + N | 挂 GEMM 尾巴 |

## 验证 memory 优化生效

跑 swiglu.py 对比融合版与两 kernel 版的墙钟；跑 test_mm_silu.py 后 dump IR，数 .linalgdir 里 transfer_write 次数，对照"融合省一趟写"的账；用 MAX_GRID 封顶版到 16M 元素验证全对。