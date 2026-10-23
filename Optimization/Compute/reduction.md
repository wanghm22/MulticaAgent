# Reduction — Compute 维度优化

Roofline 视角：归约算术强度低，compute 不是瓶颈，但单位元填错、dtype 不提升、容差定错都会让"结果错但看起来跑了"。本维度关注单位元框架、f32 累加、容差按归约长度缩放、避免编译器边界 bug。本章把 reduction 在 compute 维度的优化前后代码完整摆出来。

## 优化前：填错单位元 + f16 累加

```python
@triton.jit
def row_sum_naive_compute(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * N
    acc = tl.zeros([BLOCK], dtype=tl.float16)      # f16 累加
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(X + cols, mask=cols < N, other=1.)   # sum 填 1！错
        acc += x
    tl.store(Y + row, tl.sum(acc, axis=0))
```

两个问题：sum 的单位元是 0，填 1 会让每个越界 lane 多算 1；f16 累加长行误差巨大。

## 优化一：按归约类型选对单位元

```python
# sum: 单位元 0
x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)

# max: 单位元 -inf
x = tl.load(X + cols, mask=cols < N, other=-float('inf')).to(tl.float32)

# min: 单位元 +inf
x = tl.load(X + cols, mask=cols < N, other=float('inf')).to(tl.float32)

# mul: 单位元 1
x = tl.load(X + cols, mask=cols < N, other=1.).to(tl.float32)

# exp 后 sum（softmax 分母）：填 -inf 使 exp 后为 0
x = tl.load(X + cols, mask=cols < N, other=-float('inf')).to(tl.float32)
```

填错单位元是归约 kernel 的第一大逻辑 bug，症状通常不是崩溃而是"部分输入下结果偏一点"（全负行 max 填 0、sum 多算越界 lane 的 1）。单位元表见 ops/06 §6.1。compute 维度上，选对单位元是结果正确的前提。

## 优化二：f32 累加器，长行不丢精度

```python
# 优化前
acc = tl.zeros([BLOCK], dtype=tl.float16)

# 优化后
acc = tl.zeros([BLOCK], dtype=tl.float32)
acc += tl.load(X + cols, mask=..., other=0.).to(tl.float32)
```

tl.sum 不提升 dtype，f16 直接归约 100 万元素误差大到肉眼可见（实测 rel 0.6%）。f32 累加是长归约的硬性要求。

## 优化三：容差按归约长度缩放，不按输出元素数

```python
# 优化前：只按 N 不按 K 定容差，边界舍入 flaky
torch.testing.assert_close(got, ref, atol=1e-4)    # 1M 归约时会误判

# 优化后：按归约长度缩放
# ≤1K: atol 1e-4；~100K: 1e-2；~1M: 1e-1
torch.testing.assert_close(got, ref, atol=1e-1, rtol=1e-3)
```

f32 顺序累加 N 个 ~O(1) 的数，舍入误差按 ~√N·ε 增长。容差按归约长度缩放，否则会把正确的 kernel 判死（历史案例 addmm 1F，atol 只按 N 不按 K，边界舍入 1.05e-4 vs 容差 1e-4，数据依赖 flaky）。

## 优化四：golden 用 float 累加，参照系本身要准

```python
# 优化前
ref = x.sum()                # f16 输入是 f16 累加，golden 自己就不准

# 优化后
ref = x.float().sum()        # f32 累加
```

compute 维度上这是验证阶段的正确性优化：golden 本身要用正确精度算。

## 优化五：NaN-guard 切片探针定位越界读

```python
buf = torch.full((numel + guard,), float('nan'))
x = buf[:numel].view(shape)          # 保持精确 stride/specialization
```

任何越界读都会把 NaN 带进输出，逐块扫描即可定位 OOB 区间。归约 kernel 的"某 shape 区间数据依赖错值"（如旧版编译器只在特定 K 区间触发的越界读 bug）用这招一次定位。compute 维度上这是排错利器。

## 优化六：argmax 的 tie-break 与索引 dtype

```python
# 手动实现时注意 tie-break 和 dtype
m = tl.max(x, axis=0)
idx = tl.sum(tl.where(x == m, offs, 0), axis=0)    # 取首个还是末个取决于实现
# torch.argmax 返回首个最大下标，用 tl.argmax 更省心
idx = tl.argmax(x, axis=0)
```

手动实现要处理 tie-break 语义和 i64/i32 类型统一；tl.argmax 一步到位。compute 维度上这是"结果与 torch 语义对齐"的正确性优化。

## 优化七：cache 会掩盖编译器 bug，先换 cache 再下结论

```python
# 崩溃/错值 shape "一直复现"可能只是缓存 .so 命中
# 先 mv 换 TRITON_CACHE_DIR 再下结论
```

归约长度与向量寄存器宽度的边界（BLOCK < 寄存器宽、行长非整倍数、行长恰等寄存器宽）历史上都出过编译器级 bug，旧版 wheel 症状是 tl.sum 恒差 -16、f32 出 NaN、rc=134 断言崩溃。cache 会掩盖，先换新 cache 复测。compute 维度上这是"判断失败是编译器问题还是自己写错"的关键。

## 优化前后 compute 对照

| 项 | naive | 优化后 | 收益 |
|---|---|---|---|
| 单位元 | sum 填 1 | 按归约类型选 | 结果正确 |
| 累加 dtype | f16 | f32 | 长行误差可控 |
| 容差 | 按 N 不按 K | 按归约长度缩放 | 不误判 |
| golden | x.sum() | x.float().sum() | 参照准 |
| OOB 定位 | 无 | NaN-guard 探针 | 一次定位 |
| argmax | 手写易错 | tl.argmax | tie-break 对齐 |

## 验证 compute 优化生效

把 row_sum 的 other=0. 改成 other=1.、N=1000、BLOCK=1024，预测每行多算多少（24 个越界 lane × 1）并验证；把 acc 改成 f16、N=100000 重跑记录误差；实现 row_max 想清楚越界 lane 填什么；用 NaN-guard 探针复现旧版编译器边界 bug。