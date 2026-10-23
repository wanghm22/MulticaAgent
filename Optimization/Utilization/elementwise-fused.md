# Elementwise 与融合 — Utilization 维度优化

Roofline 视角：逐元素 compute 极轻，utilization 的关键是让每个 lane 都忙、program 铺满、别被 dispatch 空转拖住。本章把逐元素与融合在 utilization 维度的优化前后代码完整摆出来。

## 优化前：逐元素串行、无向量化

```python
@triton.jit
def swiglu_serial_kernel(G, U, Y, N):
    pid = tl.program_id(0)
    # 标量循环，lane 全空转
    for i in range(0, N):
        g = tl.load(G + i).to(tl.float32)
        u = tl.load(U + i).to(tl.float32)
        tl.store(Y + i, g * tl.sigmoid(g) * u)
```

没有 tl.arange，标量循环，向量 lane 全部空转。grid 再大，每个 program 内部仍是串行，utilization 极低。

## 优化一：向量化 load/compute/store，lane 一次铺满

```python
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < N
g = tl.load(G + offs, mask=mask, other=0.).to(tl.float32)
u = tl.load(U + offs, mask=mask, other=0.).to(tl.float32)
y = (g * tl.sigmoid(g)) * u
tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)
```

一次 load BLOCK 个元素进向量寄存器，所有 lane 并行做 silu 和乘，store 一次写出。lane 从"逐个串行"变成"一次铺满 BLOCK 个"。

## 优化二：grid 封顶 + 循环，避免 dispatch 空转与 program 丢弃

优化前 grid 无上限：

```python
grid = triton.cdiv(N, BLOCK)     # 大 N 时 grid 巨大，dispatch 开销主导
```

优化后封顶 + kernel 内循环：

```python
BLOCK = 1024
MAX_GRID = 512
grid = triton.cdiv(N, BLOCK * MAX_GRID)
swiglu_kernel[(grid,)](g, u, y, N, BLOCK=BLOCK, MAX_GRID=MAX_GRID)
```

utilization 账有两头：grid 太大时 dispatch 开销主导、每个 program 真正算的时间占比低；grid 超运行时上限还会被静默丢弃、program 白跑。封顶后 grid 永远 ≤ MAX_GRID，每个 program 用 tl.range 循环吃多 tile，program 调度一致、dispatch 开销被摊薄。实测封顶版到 16M 元素全对，且 dispatch 主导场景反而更快。

## 优化三：pointwise_dynamic 天然 grid-safe

```python
# FlagGems 大部分二元逐元素算子走 pointwise_dynamic：
# num_ctas = min(max_grid, num_tiles) + kernel 内循环处理多 tile
```

pointwise_dynamic codegen 天然 grid-safe，grid 永远不超上限、不受运行时版本 grid 限制影响。但注意它的 in-place/out 变体走手写平坦 kernel，不享受该保护——历史上 le/maximum/where/clamp 的 in-place 变体失败而 out-of-place 通过就是这个差别。自己写逐元素 kernel 时照抄"MAX_GRID 封顶 + tl.range 循环"骨架即可，grid 安全不依赖运行时升级。

## 优化四：in-place 变体别踩非 grid-safe 路径

```python
# 优化前：in-place 变体走手写平坦 kernel，grid 不封顶
y.add_(x)          # 可能走非 grid-safe 路径

# 优化后：out-of-place 或显式封顶
y = x + y          # 走 pointwise_dynamic，grid-safe
```

utilization 角度，选对路径比调参更重要：out-of-place 走 pointwise_dynamic 天然安全，in-place 手写 kernel 要自己封顶。

## 优化五：一维扁平化，连续访存让 lane 不跨 stride

```python
N = g.numel()
```

扁平处理保持连续访存，lane 读的是相邻地址，向量化 load 一次取 BLOCK 个连续元素，utilization 最高。若按原始多维 shape 处理、stride 不连续，lane 访存跨 stride，向量化效率下降。

## 优化六：逐元素链加长对 utilization 零影响

```python
y = (g * tl.sigmoid(g)) * u + bias     # 三元链
```

逐元素链加长（silu·mul + bias）不改变并行结构，还是那 3 趟访存、同样的 lane 铺满。utilization 上多几次乘加对 memory-bound 的 K3 可忽略，代码结构零变化——这是逐元素融合的便利。

## 优化前后 utilization 对照

| 形态 | 行内并行 | grid | lane 利用 |
|---|---|---|---|
| 标量串行 | 无 | 任意 | 空转 |
| 向量化 | BLOCK 个 lane | cdiv(N,BLOCK) | 一次铺满 |
| 封顶 + 循环 | BLOCK 个 lane | ≤ MAX_GRID | 铺满 + dispatch 摊薄 |
| pointwise_dynamic | BLOCK 个 lane | ≤ max_grid | 天然 grid-safe |

## 验证 utilization 优化生效

对比标量串行版与向量化版墙钟；用 MAX_GRID 封顶前后对比 dispatch 主导场景（小数据量、大 grid）的墙钟；检查 in-place 与 out-of-place 变体是否走不同路径、grid 是否安全。