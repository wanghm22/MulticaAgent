# RMSNorm — Utilization 维度优化

Roofline 视角：趟数确定后，关注向量 lane 与归约单元是否空转。rmsnorm 只有一趟归约，utilization 的重点是把这一趟的 lane 用满、并行铺开。本章把 rmsnorm 在 utilization 维度的优化前后代码完整摆出来。

## 优化前：标量累加、行内串行

```python
@triton.jit
def rmsnorm_serial_kernel(X, Y, W, N, eps):
    row = tl.program_id(0)
    X += row * N
    Y += row * N
    s = 0.0
    for i in range(0, N):
        v = tl.load(X + i).to(tl.float32)
        s += v * v                 # 标量累加，lane 空转
    rstd = 1 / tl.sqrt(s / N + eps)
    for i in range(0, N):
        y = tl.load(X + i).to(tl.float32) * rstd * tl.load(W + i).to(tl.float32)
        tl.store(Y + i, y)
```

没有 tl.arange，循环是标量循环，向量 lane 全空转，归约退化成 N 次标量加。行间 grid=(M,) 并行，行内串行，utilization 极低。

## 优化一：向量累加器 + 单次水平归约

优化前每块先归约再加标量：

```python
s = 0.0
for off in range(0, N, BLOCK):
    cols = off + tl.arange(0, BLOCK)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    s += tl.sum(x * x, axis=0)     # 每块一次水平归约
```

优化后 lane 内累加、最后归约一次：

```python
acc = tl.zeros([BLOCK], dtype=tl.float32)
for off in range(0, N, BLOCK):
    cols = off + tl.arange(0, BLOCK)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    acc += x * x                   # 纯向量加
ms = tl.sum(acc, axis=0) / N       # 整行只归约一次
```

利用率账：tl.sum 是 log2(BLOCK) 轮水平归约（lane 间通信），优化前做 N/BLOCK 次，优化后只做一次。归约单元不再反复启停，向量 lane 持续满载。这是 rmsnorm 唯一的水平归约，压到一次就是 utilization 的极限。

## 优化二：每 program 一行，行间并行拉满

```python
rmsnorm_kernel[(M,)](x_arg, y, weight, x_arg.stride(0), N, eps, BLOCK=BLOCK)
```

grid=(M,)，行间独立无通信，每行一个 program。program 调度器把 M 个 program 铺满可用执行单元，行间零等待。若一个 program 串行多行，行内两趟关键路径乘上行数，utilization 崩掉。

## 优化三：ONE_TILE 下唯一一次归约

ONE_TILE 版把归约压到理论最少：

```python
x = tl.load(X + row * N + cols, mask=mask, other=0.).to(tl.float32)
ms = tl.sum(x * x, axis=0) / N     # 整个 kernel 唯一的水平归约
rstd = 1 / tl.sqrt(ms + eps)
```

整个 kernel 只有一次跨 lane 通信，归一化 `x * rstd * w` 是纯逐元素、每 lane 独立。这是 rmsnorm 在 utilization 维度能拿到的最好形态。

## 优化四：归约前升 f32，lane 用满且算得对

```python
x = tl.load(X + cols, mask=..., other=0.).to(tl.float32)
acc += x * x
```

tl.sum 不提升 dtype，fp16 直接归约得到 fp16 部分和，长行误差大。升 f32 后 lane 宽度从 64 变 32 元素，单指令处理元素数减半，但这是正确性硬性要求，且 f32 归约稳定。utilization 的用满 lane 服从算得对。

## 优化五：无 bias、无 mean 缓存，减少不必要的 lane 工作

```python
# layernorm 要存 mean 和 rstd 给 backward
tl.store(Mean + row, mean)
tl.store(Rstd + row, rstd)
# rmsnorm 只需 rstd（ms 可由 rstd 反推），少一个标量 store
tl.store(Rstd + row, rstd)
```

rmsnorm 没有 bias、没有 mean，kernel 里少一个 weight 旁的 bias load、少一个标量 store。utilization 角度，少一条指令就少一次 lane 的额外工作，虽然量小，但方向上是对的。

## 优化前后 utilization 对照

| 形态 | 行内并行 | 归约次数 | lane 利用 |
|---|---|---|---|
| 标量串行 | 无 | N 次标量加 | 空转 |
| 每块归约 | 向量加 | N/BLOCK 次水平归约 | 归约单元反复启停 |
| 向量累加器 | 向量加 | 1 次水平归约 | lane 持续满载 |
| ONE_TILE | 整行向量 | 1 次水平归约 | lane 一次铺满 |

## 验证 utilization 优化生效

dump IR 数 linalg.reduce，ONE_TILE 版应只有一次（sum(x²)）；对比每块归约版与向量累加器版在 N=8192 时的墙钟；删掉 .to(tl.float32) 用 f16 长行跑，观察误差。