# RMSNorm — Compute 维度优化

Roofline 视角：rmsnorm 算术强度低，compute 不是瓶颈，但计算正确性决定有效算力。本维度关注数值稳定、mask 填充值吸收、dtype 提升、math 函数输入约束、避免前端编译坑。本章把 rmsnorm 在 compute 维度的优化前后代码完整摆出来。

## 优化前：f16 直接归约与 sqrt

```python
@triton.jit
def rmsnorm_naive_compute(X, Y, W, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.)   # fp16
    ms = tl.sum(x * x, axis=0) / N                          # fp16 归约，误差大
    rstd = 1 / tl.sqrt(ms + eps)                            # ms 若是 fp16，sqrt 撞 _check_dtype
    w = tl.load(W + cols, mask=mask)
    tl.store(Y + row * N + cols, x * rstd * w, mask=mask)
```

两个问题：fp16 归约长行误差大；ms 若保持 fp16，tl.sqrt 直接编译失败（_check_dtype 只收 f32/fp64）。

## 优化一：load 后立刻升 f32

```python
x = tl.load(X + row * N + cols, mask=mask, other=0.).to(tl.float32)
ms = tl.sum(x * x, axis=0) / N            # f32 归约
rstd = 1 / tl.sqrt(ms + eps)              # ms 是 f32 标量，sqrt 合法
```

load 后立刻升 f32，归约和 sqrt 都在 f32 上做。tl.sum 不会自动提升 dtype，这是 K3 端口层修过的真 bug（ops/02 §2.5 坑 1、ops/03 §3.4）。compute 维度上，"在正确精度上累加和开方"是结果有效的前提。

## 优化二：other=0. 天然吸收，无需 where

```python
x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
acc += x * x          # 越界 lane 的 0² = 0
```

对照 layernorm：rmsnorm 没有减均值，越界 lane 填 0 后 `0² = 0` 天然不贡献，不需要 tl.where。这是"mask 填充值被后续运算吸收"的经典正例（ops/03 §3.3）。compute 上少一步 where，结果仍精确。

## 优化三：方差用 mean(x²) 而非 E[x²]-mean²

rmsnorm 定义就是 mean(x²)，没有减均值这一步，天然没有消去误差问题：

```python
ms = tl.sum(x * x, axis=0) / N     # mean square，不是 var
```

对照 layernorm 要选两趟 (x-mean)² 避消去误差，rmsnorm 少一趟且没有这个取舍。compute 维度上这是结构性的简化。

## 优化四：rsqrt 的标量算术链要正确

```python
ms = tl.sum(acc, axis=0) / N              # f32 标量 / index 标量
rstd = 1 / tl.sqrt(ms + eps)              # 或 tle.rsqrt(ms + eps)
```

raw 层这条链是"向量 reduce → f32 标量 / index → rsqrt 标量 → 标量广播回向量乘"（ops/03 §3.5）。compute 上要保证：ms 是 f32 标量、eps 是浮点常量、rsqrt/sqrt 作用在标量上。tl.sqrt 只收 f32/fp64，ms 已是 f32 安全；若对 f16 向量直接 tl.sqrt 会撞 _check_dtype。

## 优化五：避免 constexpr 整数的 .to()

```python
# 优化前
a = tl.full([], 2, tl.int32)
b = a.to(tl.float32)           # 'int' object has no attribute 'to'

# 优化后
b = a * a
# 或
two = tl.full([], 2.0, tl.float32)
```

constexpr 整数没有 .to()（ops/02 §2.5 坑 4）。compute 维度上这是避免前端编译失败的优化。

## 优化六：block-ptr store 显式 cast

```python
# 优化前：block-ptr store 无隐式 cast
tl.store(block_ptr, f32_val)   # f16 block 写 f32 值报 mismatch

# 优化后
tl.store(block_ptr, f32_val.to(Y.type.element_ty))
```

普通指针 store 有隐式 cast，block-ptr 没有（ops/02 §2.5 坑 3）。compute 上保证写出值类型与目标匹配。

## 优化前后 compute 对照

| 项 | naive | 优化后 | 收益 |
|---|---|---|---|
| 归约 dtype | f16 | f32 | 长行误差可控 |
| sqrt 输入 | 可能 f16 | f32 标量 | 合法 |
| 越界 lane | 0² 天然吸收 | 同左 | 无需 where |
| 方差写法 | mean(x²) | 同左 | 无消去误差 |
| constexpr 算术 | .to() 崩 | a*a / tl.full | 前端不崩 |
| block-ptr store | 隐式 cast 崩 | 显式 .to() | 类型匹配 |

## 验证 compute 优化生效

N=1000、BLOCK=1024 用例确认 other=0. 平方吸收让结果精确不变；golden 用 `x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w.float()` 再降回 f16，atol/rtol 1e-2；删掉 .to(tl.float32) 长行跑 f16，观察误差；删 .to 后 f16 喂 tl.sqrt 抄下 _check_dtype 报错原文。

## 三个维度的关系

f32 提升既是 compute 精度要求也影响 utilization 的 lane 宽度；other=0. 天然吸收既是 compute 正确也省掉 memory/utilization 上的 where 操作。调优顺序仍是 memory 趟数先行、utilization 并行和归约形态其次、compute 精度与稳定性收尾。