# Cumsum / Scan — Utilization 维度优化

Roofline 视角：scan 顺序依赖是并行的天敌，utilization 的核心是把"块内串行"压到最小、"块间并行"拉满。本章把 cumsum 在 utilization 维度的优化前后代码完整摆出来。

## 优化前：单 program 串行 scan，只有一条 lane 在工作

```python
@triton.jit
def cumsum_serial_kernel(X, Y, N):
    acc = 0.0
    for i in range(0, N):
        acc += tl.load(X + i).to(tl.float32)
        tl.store(Y + i, acc)
```

grid=(1,)，一个 program 串行扫全部 N，标量累加器，只有一条 lane 在工作，其余执行单元全空转。关键路径 N 次加法，utilization 极低。

## 优化一：块内 tl.cumsum 原语，log BLOCK 轮并行

```python
inp_vals = tl.load(inp + offset, mask=mask, other=0.)
result = tl.cumsum(inp_vals, axis=0)        # Hillis-Steele，log BLOCK 轮移位加
```

tl.cumsum 内部展开成 Hillis-Steele：log BLOCK 轮"每元素加上 2^r 位之前的元素"，每轮所有 lane 并行。块内关键路径从 BLOCK 次串行降到 log BLOCK 轮，utilization 大幅提升。

## 优化二：两趟 scan-then-fan，块间并行拉满

优化前单 program 串行全序列。优化后趟1 grid=(part_num,) 每 program 块内 scan，趟2 grid=(part_num,) 每 program 加前段 base：

```python
grid = (triton.cdiv(n, BLOCK_SIZE),)
scan_part_sum_kernel[grid](x, out, partial_sum, n, part_num, BLOCK_SIZE=BLOCK_SIZE)
add_base_sum_kernel[grid](out, partial_sum, n, part_num, BLOCK_SIZE=BLOCK_SIZE)
```

utilization 账：趟1 的 part_num 个 program 并行做块内 scan，铺满执行单元；趟2 的 part_num 个 program 并行加 base。块间并行拉满，只有块内 scan 仍是 log BLOCK 轮（无法消除，能消除的只有块间串行）。

## 优化三：partial_sum 递归前缀化，段2 输入小到怎么算都快

```python
# 生产实现：对 partial_sum 这个小数组再做一次 scan_then_fan（in-place 前缀化）
# 数组只有 part_num 个元素，递归一层就到底
```

utilization 角度，段2 的输入只有 part_num 个数，怎么算都快。这是"段2 的输入小到怎么算都快"（ops/06 §6.2 形态 B）的思想——不追求段2 的并行度，追求段1 的并行度拉满。

## 优化四：三阶段 block-scan，关键路径从 N 降到 VL+P

```python
# Phase1 grid=(P,) 每 program 向量归约自己块的总和
# Phase2 grid=(1,) 对 P 个块和做排他前缀和（小串行合理）
# Phase3 grid=(P,) 每 program 块内串行 scan 再加块基值
```

raw 三阶段版的关键路径：VL（Phase3 块内）+ P（Phase2）+ 两次额外 launch。N=8192 时 8192 步降到 64+128 步，Phase1 的块归约还是向量化的（一条 vreduce_sum 收掉 64 个元素）。utilization 上 Phase1、Phase3 的 P 个 program 并行铺满，Phase2 的 128 步串行远比再来一次并行 launch + 同步便宜。

## 优化五：host pid 传参并行，raw 层保持无 program_id

```python
@triton.jit
def block_sum_host(X, block_sums, N, P):
    p = tl.program_id(0)                    # ← 并行度在 tl 层
    if p < P:                               # tl 层标量 if，合法
        _sr_call(block_sum_kernel, outputs=[], inputs=[X, block_sums, N, p])
```

raw kernel 不知道自己是第 p 个 program，它只收到一个标量 p——并行语义全在 tl 层，raw 层保持"无 program_id、无 if"。utilization 上并行度由 tl 层 grid 决定，raw 层只负责块内计算。

## 优化六：n 很小时单 program 反而快，别盲目两趟

```python
# n < 4096 时单 program 串行 scan 的墙钟可能低于两趟 launch
```

utilization 上不是并行度越高越好，n 很小时两趟 launch 的 dispatch 开销超过并行收益。老规矩，实测说了算。

## 优化前后 utilization 对照

| 形态 | 并行度 | 关键路径 | lane 利用 |
|---|---|---|---|
| 单 program 串行 | grid=(1,) | N 次加法 | 一条 lane |
| 块内 tl.cumsum | grid=(part_num,) | log BLOCK 轮 + 块间 | 块内并行 |
| 两趟 scan-then-fan | grid=(part_num,)×2 | log BLOCK + part_num | 块间并行拉满 |
| 三阶段 block-scan | grid=(P,)×2 + grid=(1,) | VL + P | 块归约向量化 |

## 验证 utilization 优化生效

对比单 program 串行版与三阶段版在 N=8192 时的墙钟；算关键路径步数（8192 vs 64+128）；跑 test_raw_cumsum_vec.py 对照源码。