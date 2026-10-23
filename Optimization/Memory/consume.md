# Cumsum / Scan — Memory 维度优化

Roofline 视角：scan 把 N 个数变 N 个（每位置是到它为止的归约值），输出长度等于输入，访存上比归约多一倍写。Memory 维度的目标是块内 scan 只过一趟、跨块前缀传递不落多余中间结果。本章把 cumsum 在 memory 维度的优化前后代码完整摆出来。

## 优化前：单 program 串行 scan，每元素读写各一次

```python
@triton.jit
def cumsum_serial_kernel(X, Y, N):
    acc = 0.0
    for i in range(0, N):
        acc += tl.load(X + i).to(tl.float32)    # 每个元素读
        tl.store(Y + i, acc)                     # 每个元素写
```

访存账：x 每个元素读一次、y 每个元素写一次，看似 2N 是最小。但 grid=(1,)、串行 O(N)，关键路径 N 次加法，memory 上虽然趟数少，并行度为零（utilization 篇展开）。memory 上它不是坏在趟数，而是坏在没有分块复用。

## 优化一：块内 tl.cumsum 原语，块内只过一趟

```python
@triton.jit(do_not_specialize=["n_elements", "part_num"])
def scan_part_sum_kernel(inp, out, partial_sum, n_elements, part_num,
                         BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_vals = tl.load(inp + offset, mask=mask, other=0.)
    result = tl.cumsum(inp_vals, axis=0)        # 块内 scan，一步到位
    tl.store(out + offset, result, mask=mask)
    tl.store(partial_sum + pid, tl.sum(inp_vals))
```

块内 tl.cumsum 是原语，编译器内部展开成 Hillis-Steele（log BLOCK 轮移位加），数据在寄存器里完成 scan，块内只从内存读一次、写一次。memory 上块内 scan 的趟数被压到最小，中间移位加不落内存。

## 优化二：两趟 scan-then-fan，段总和用小数组传递

优化前跨块前缀无处传递，只能单 program 串行。优化后两趟：趟1 每段块内 scan + 写段总和，趟2 把前段总和加到本段：

```python
@triton.jit(do_not_specialize=["n_elements", "part_num"])
def add_base_sum_kernel(out, partial_sum, n_elements, part_num,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    if pid > 0:                                  # 标量条件：第 0 段无 base
        base = tl.load(partial_sum + pid - 1)
        out_vals = tl.load(out + offset, mask=mask, other=0.)
        tl.store(out + offset, (out_vals + base).to(out_vals.dtype),
                 mask=mask)
```

memory 账：趟1 读 inp 一遍、写 out 一遍、写 partial_sum 小数组；趟2 读 out 一遍、读 partial_sum 小标量、写 out 一遍。out 被读改写一遍，但 partial_sum 只有 part_num 个小标量，跨块前缀传递的 memory 代价可忽略。这是 scan-then-fan 的 memory 本质：用 O(part_num) 的小数组换掉 O(N) 的串行。

## 优化三：段总和用 tl.sum(inp_vals) 而非 result 尾元素

```python
# 优化前：取 result 尾元素，n % BLOCK != 0 时落在 mask 外，要额外 where
# 优化后：直接对输入求和，天然免疫
tl.store(partial_sum + pid, tl.sum(inp_vals))
```

尾元素在 n % BLOCK != 0 时落在 mask 外，需要额外 where 选取；直接对输入求和天然免疫（越界 lane 已是 0）。memory 上省掉一次 where 的寄存器操作，且 other=0. 必须写上——masked load 不写 other 时越界 lane 是未定义值，会直接污染 partial_sum。

## 优化四：两趟 = 两次 launch，n 小时单 program 反而快

```python
# 优化前：n 很小时（<4096）两趟 launch 的 dispatch 开销超过并行收益
# 优化后：n 小走单 program 串行 scan，n 大走两趟
```

memory 上不是趟数越少越好，K3 上 dispatch 开销不可忽略。n 很小时单 program 串行 scan 反而可能更快——老规矩，实测说了算。这是 memory 维度"趟数 vs launch 开销"的取舍。

## 优化五：行 scan 的三维组织，每行独立只过一趟

```python
# 行 scan（tl.cumsum(x_2d, axis=1)，每行独立）
# grid=(A, part_num, C) 的三维组织：A/C 是独立行/列，中间维切段
```

行 scan 每行独立，A/C 两维是独立的行/列，中间维切段，同一套两趟法逐行并行。memory 上每行只读一遍、写一遍，段间前缀用小数组传递。

## 优化前后访存对照

| 形态 | x 读 | out 读改写 | partial 小数组 | 适用 |
|---|---|---|---|---|
| 单 program 串行 | N | N 写 | 无 | n 小 |
| 两趟 scan-then-fan | N | N 读 + N 写 | O(part_num) | n 大 |

## 验证 memory 优化生效

跑 cumsum.py 的 n=5/64/1000/4096 用例；对比单 program 串行版与两趟版在 n=100/1000/10000 的墙钟交叉点；dump IR 数 transfer_read/transfer_write 确认块内 scan 不落中间 buffer。