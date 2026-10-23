# Layernorm — Utilization 维度优化

Roofline 视角：memory 趟数确定后，关注每趟访存时向量 lane 和归约单元有没有空转。Layernorm 的 utilization 核心与 softmax 同源：向量累加器替代标量累加、每 program 一行拉满行间并行、归约只做一次。本章把 layernorm 在 utilization 维度的优化前后代码完整摆出来。

## 优化前：标量累加、行内串行

```python
@triton.jit
def layernorm_serial_kernel(X, Y, W, B, N, eps):
    row = tl.program_id(0)
    X += row * N
    Y += row * N
    # 标量累加：lane 全空转，归约退化成逐元素比较
    s = 0.0
    for i in range(0, N):
        s += tl.load(X + i).to(tl.float32)
    mean = s / N
    v = 0.0
    for i in range(0, N):
        d = tl.load(X + i).to(tl.float32) - mean
        v += d * d
    rstd = 1 / tl.sqrt(v / N + eps)
    for i in range(0, N):
        y = (tl.load(X + i).to(tl.float32) - mean) * rstd \
            * tl.load(W + i).to(tl.float32) + tl.load(B + i).to(tl.float32)
        tl.store(Y + i, y)
```

没有 tl.arange，循环是标量循环，向量 lane 全部空转；归约退化成 N 次标量加法。行间虽然 grid=(M,) 并行，行内却是串行，长行时关键路径等于 N 次标量操作，utilization 极低。

## 优化一：向量累加器 + 单次水平归约

优化前每块先归约再加标量：

```python
s = 0.0
for off in range(0, N, BLOCK):
    cols = off + tl.arange(0, BLOCK)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    s += tl.sum(x, axis=0)         # 每块都做一次水平归约
```

优化后 lane 内累加、最后归约一次：

```python
acc = tl.zeros([BLOCK], dtype=tl.float32)
for off in range(0, N, BLOCK):
    cols = off + tl.arange(0, BLOCK)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    acc += x                       # 纯向量加，无水平归约
mean = tl.sum(acc, axis=0) / N     # 整行只归约一次
```

利用率账：tl.sum 是 log2(BLOCK) 轮的水平归约，属于 lane 间通信。优化前做 N/BLOCK 次，优化后只做一次，归约单元不再反复启停，向量 lane 在循环里持续满载。var 趟同理，只是累加的是 `x*x`。这是"lane 内累加、最后归约"的标准形态。

## 优化二：每 program 一行，行间并行拉满

```python
layernorm_kernel[(M,)](x_arg, y, weight, bias, mean, rstd,
                       x_arg.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE)
```

grid=(M,)，行间完全独立、无通信，每行一个 program。program 调度器可以把 M 个 program 铺满所有可用执行单元，行与行零等待。若一个 program 串行处理多行，行内多趟的关键路径被乘上行数，utilization 崩掉。这与 softmax 的 grid=(n_rows,) 是同一套并行组织。

## 优化三：ONE_TILE 下两次归约各一次

ONE_TILE 版把行内归约压到理论最少：

```python
x = tl.load(X + row * N + cols, mask=mask, other=0.).to(tl.float32)
mean = tl.sum(x, axis=0) / N          # 1 次水平归约（mean）
var = tl.sum(xc * xc, axis=0) / N     # 1 次水平归约（var）
```

整个 kernel 只有两次水平归约，分别对应 mean 和 var，没有别的跨 lane 通信。归一化那步 `xc * rstd * w + b` 是纯逐元素，每个 lane 独立算，无归约。

## 优化四：归约前升 f32，让 lane 用满且算得对

优化前 fp16 直接归约：

```python
x = tl.load(X + cols, mask=..., other=0.)   # fp16
acc += x                                     # fp16 累加，长行误差大
```

优化后 load 后立刻升 f32：

```python
x = tl.load(X + cols, mask=..., other=0.).to(tl.float32)
acc += x                                     # f32 累加
```

tl.sum 不会自动提升 dtype，fp16 tile 直接 sum 得到 fp16 部分和，长行下误差超容差（实测 abs 可到 0.03，ops/02 §2.4）。升 f32 后 lane 宽度从 64 元素变 32 元素，单条指令处理元素数减半，但这是正确性硬性要求，且 f32 归约数值稳定。utilization 的用满 lane 必须服从算得对。

## 优化五：可选 weight/bias 的守卫不能破坏 utilization

优化前在内联处做 `if W is None` 守卫：

```python
if W is None:
    w = 1.0
else:
    w = tl.load(W + cols, mask=mask)
```

这种写法只护住 load 那一行，若在循环外预建 block desc，`tl.make_block_ptr(base=W)` 在 W 为 None 时前端崩（ops/02 §2.5 坑 2）。更稳的做法是 host 侧保证非 None（传全 1 全 0 tensor），kernel 内不分支，向量 lane 的访存路径保持规整、不因分支产生空转。utilization 维度上，能不用控制流就不用，让所有 lane 走同一条指令流。

## 优化前后 utilization 对照

| 形态 | 行内并行 | 归约次数 | lane 利用 |
|---|---|---|---|
| 标量串行 | 无 | N 次标量加 | 空转 |
| 每块归约 | 向量加 | N/BLOCK 次水平归约 | 归约单元反复启停 |
| 向量累加器 | 向量加 | 1 次水平归约 | lane 持续满载 |
| ONE_TILE | 整行向量 | mean 1 次 + var 1 次 | lane 一次铺满 |

## 验证 utilization 优化生效

dump IR 数 linalg.reduce 次数，ONE_TILE 版应只有两次（mean、var）；对比每块归约版与向量累加器版在 N=8192 时的墙钟；删掉 .to(tl.float32) 用 f16 跑长行，观察误差如何滚大（对照 ops/02 练习 1）。