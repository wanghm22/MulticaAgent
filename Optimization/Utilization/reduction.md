# Reduction — Utilization 维度优化

Roofline 视角：归约产出少、访存多，utilization 的关键是让每次读进来的数据被所有 lane 充分参与归约，且并行度铺满。本章把 reduction 在 utilization 维度的优化前后代码完整摆出来。

## 优化前：单 program 串行全归约

```python
@triton.jit
def full_sum_serial_kernel(X, Out, N, BLOCK: tl.constexpr):
    s = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        s += tl.sum(tl.load(X + cols, mask=cols < N, other=0.), axis=0)
    tl.store(Out, s)
```

grid=(1,)，一个 program 串行扫全部 N，只有 BLOCK 个 lane 在工作，其余执行单元空转。utilization 极低，长 N 时关键路径等于 N/BLOCK 次迭代。

## 优化一：行归约每 program 一行，行间并行拉满

```python
row_sum_kernel[(M,)](x, y, N, BLOCK=BLOCK)
```

grid=(M,)，行间独立无通信，每行一个 program。M 个 program 铺满所有执行单元，行间零等待。这是形态 A 行归约的标准并行组织。若一个 program 串行多行，行间并行度丢失，utilization 崩掉。

## 优化二：两段式全归约，段1 并行拉满

```python
n_parts = triton.cdiv(N, BLOCK)
partial_sum_kernel[(n_parts,)](x, partial, N, BLOCK=BLOCK)
```

grid=(cdiv(N, BLOCK),)，每 program 归约自己的 BLOCK 段，n_parts 个 program 并行铺开。段1 完全并行，utilization 从单 program 的 1/n_parts 拉满。段2 的输入只有 n_parts 个数，怎么算都快。

## 优化三：lane 内累加、最后归约，归约单元不反复启停

优化前每块先归约再加标量：

```python
s = 0.0
for off in range(0, N, BLOCK):
    cols = off + tl.arange(0, BLOCK)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    s += tl.sum(x, axis=0)         # 每块一次水平归约
```

优化后向量累加器、最后归约一次：

```python
acc = tl.zeros([BLOCK], dtype=tl.float32)
for off in range(0, N, BLOCK):
    cols = off + tl.arange(0, BLOCK)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    acc += x                       # 纯向量加
y = tl.sum(acc, axis=0)            # 整行只归约一次
```

利用率账：tl.sum 是 log2(BLOCK) 轮水平归约（lane 间通信）。优化前做 N/BLOCK 次，优化后只做一次，归约单元不再反复启停，向量 lane 在循环里持续满载。这是块内归约的标准形态（ops/02 趟1 同款）。

## 优化四：do_not_specialize，避免 M/N 特定值触发特化

```python
@triton.jit(do_not_specialize=["M", "N"])
def row_sum_kernel(X, Y, N, BLOCK: tl.constexpr):
    ...
```

避免 M/N 特定值被折成 constexpr 触发特化路径，保持一份编译产物覆盖多 shape，减少编译和调度开销。utilization 角度，特化少意味着 program 调度更一致、不因 shape 变化走不同路径。

## 优化五：argmax 手动实现，利用"只有最大值 lane 贡献下标"

```python
m = tl.max(x, axis=0)
idx = tl.sum(tl.where(x == m, offs, 0), axis=0)    # 只有最大值的 lane 贡献下标
```

或直接用 tl.argmax(x, axis) 一步到位。手动实现时注意索引 dtype：i64 下标与 i32 值混算前先统一类型。utilization 角度，tl.argmax 内部已把比较和下标选择并行化，比手写循环高效。

## 优化前后 utilization 对照

| 形态 | 并行度 | 归约次数 | lane 利用 |
|---|---|---|---|
| 单 program 串行 | grid=(1,) | N/BLOCK 次 | 仅 BLOCK 个 lane |
| 行归约每 program 一行 | grid=(M,) | 每行 1 次 | M×BLOCK 铺满 |
| 每块归约标量累加 | grid=(M,) | N/BLOCK 次 | 归约单元反复启停 |
| 向量累加器 | grid=(M,) | 每行 1 次 | lane 持续满载 |

## 验证 utilization 优化生效

dump IR 数 linalg.reduce，向量累加器版应每行只一次；对比单 program 串行版与两段式在 N=1000000 时的墙钟；用 do_not_specialize 前后对比编译产物数量。