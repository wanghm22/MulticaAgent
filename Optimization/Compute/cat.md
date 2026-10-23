# Cat / Concat — Compute 维度优化

Roofline 视角：cat 没有计算，compute 维度谈的是"映射规则正确、偏移不溢出、类型不崩"。本章把 cat 在 compute 维度的优化前后代码完整摆出来。

## 优化前：偏移用 int32、非拼接维假设连续

```python
@triton.jit
def cat_naive_compute(A, B, C, M_A, M_B, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs_n = col * BLOCK_N + tl.arange(0, BLOCK_N)
    # 偏移 row * N 用 int32，大张量溢出
    v = tl.load(A + row * N + offs_n, mask=offs_n < N, other=0.)
    tl.store(C + row * N + offs_n, v, mask=offs_n < N)
```

两个问题：row * N 用 int32，numel 接近 2^31 时溢出、错行而不是崩溃；非拼接维假设连续，非连续输入错值。

## 优化一：mask-select 映射规则，两路相加即选择

```python
va = tl.load(A + row * N + offs_n,
             mask=n_mask & (row < M_A), other=0.)
vb = tl.load(B + (row - M_A) * N + offs_n,
             mask=n_mask & (row >= M_A), other=0.)
tl.store(C + row * N + offs_n, va + vb, mask=n_mask)
```

dim=0 拼接的行映射规则一句话：输出第 r 行 = A 的第 r 行（r < M_A）或 B 的第 r-M_A 行（r ≥ M_A）。两路都 load、mask 选路、相加即选择。compute 上这是"映射规则用无分支表达"的正确性优化，对 int/float 都成立（bool 要换 |）。

## 优化二：大数偏移升 int64，避免 int32 溢出

```python
# 优化前：row * N 用 int32，numel 接近 2^31 时溢出、错行不崩
tl.store(C + row * N + offs_n, ...)

# 优化后：host 侧确认 numel < 2^31，或 kernel 内 .to(tl.int64)
row64 = row.to(tl.int64)
tl.store(C + row64 * N + offs_n, ...)
```

numel 接近 2^31 时 row * N 单项超 int32，溢出症状是错行而不是崩溃。host 侧确认 numel < 2^31，或 kernel 内 .to(tl.int64)。compute 维度上这是"偏移计算不溢出"的正确性优化。

## 优化三：非拼接维连续，host 先 contiguous 或 stride 版

```python
# 优化前：kernel 假设输入连续，a[:, ::2] 这类输入错值
# 优化后：host 侧先 contiguous
a = a.contiguous()
```

非拼接维不连续时，host 侧先 contiguous() 或走 stride 版偏移（生产实现传全套 stride 参数），要么 fallback copy_。永远不要在 kernel 里假设输入连续。compute 上这是"非连续访问模式正确"的优化。

## 优化四：纯搬运判据 torch.equal，任何非零 diff 都是 bug

```python
# 优化前：用容差测试，掩盖搬运错位
torch.testing.assert_close(got, ref, atol=1e-2)

# 优化后：逐位相等
assert torch.equal(got, ref), f"mismatch at {(M_A, M_B, N)}"
```

纯搬运没有算术，结果必须逐位相等（torch.equal）——这是搬运类 kernel 测试的默认判据，任何非零 diff 都是 bug（对照归约/attention 的容差测试）。compute 上这是"搬运正确性判据"的优化。

## 优化五：空张量 cat 合法、单输入、K>2 都测

```python
# 形状：拼接维含 0 的输入（空张量 cat 合法！）、单输入、K>2、非拼接维为 1
# dtype：每种 dtype 各过一遍（纯搬运与 dtype 无关，但 int64 偏移易写错）
# 非连续输入：a[:, ::2] 这类
```

compute 上必测矩阵：拼接维含 0 的空张量（cat 合法）、单输入、K>2、非拼接维为 1；每种 dtype 各过一遍（纯搬运与 dtype 无关，但 int64 偏移计算容易写错）；非连续输入验证 fallback 或 stride 版。

## 优化六：do_not_specialize，避免 M_B=1 特化

```python
@triton.jit(do_not_specialize=["M_A", "M_B", "N"])
def cat0_kernel(...):
    ...
```

M_B=1 之类的值不该被折成 constexpr。compute 上避免特化路径带来的调度不一致。

## 优化前后 compute 对照

| 项 | naive | 优化后 | 收益 |
|---|---|---|---|
| 映射规则 | if/elif 分支 | mask-select 相加 | 无分支、正确 |
| 偏移 dtype | int32 | int64 或确认 <2^31 | 不溢出 |
| 非连续输入 | 假设连续 | contiguous/stride | 不错值 |
| 验证判据 | 容差 | torch.equal | 搬运逐位对 |
| 边界形状 | 只测常规 | 空张量/单输入/K>2 | 覆盖全 |

## 验证 compute 优化生效

跑 cat.py 的 3 组形状判据 torch.equal；把 B 路的 other=0. 改成 other=1.，预测哪一半输出错并验证；构造 row * N 超 2^31 的形状观察 int32 溢出（错行不崩），加 .to(tl.int64) 修复；非连续输入 a[:, ::2] 验证 fallback。

## 三个维度的关系

mask-select 既是 utilization 无分支也保 compute 映射正确；int64 偏移既是 compute 不溢出也影响 memory 的地址计算；contiguous 既是 compute 正确也保 memory 的连续访存。调优顺序仍是 memory 每段只读一次、输出每位置写一次先行，utilization program 铺满与 grid 安全其次，compute 映射规则与偏移 dtype 收尾。