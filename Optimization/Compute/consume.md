# Cumsum / Scan — Compute 维度优化

Roofline 视角：scan 算术强度低，compute 不是瓶颈，但 dtype 提升表、排他前缀顺序、寄存器宽边界、运行时栈空间都会让"结果错或崩"。本章把 cumsum 在 compute 维度的优化前后代码完整摆出来。

## 优化前：dtype 不提升、前缀顺序写反

```python
@triton.jit
def cumsum_naive_compute(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < N, other=0.)   # f16/int16 不提升
    y = tl.cumsum(x, axis=0)                          # f16 累加，误差大
    tl.store(Y + offs, y, mask=offs < N)
```

两个问题：f16/int16 直接 cumsum 累加误差大；跨块前缀若顺序写反（先累加再写）会变成 inclusive 而非 exclusive。

## 优化一：dtype 提升表与 torch.cumsum 语义对齐

```python
inp_vals = tl.load(inp + offset, mask=mask, other=0.)
# dtype 提升表（与 torch.cumsum 语义对齐）：
#   int64/uint64/fp64 保持；其余整数 → int32；其余浮点 → f32
if tl.constexpr(inp_vals.dtype.is_int64() or inp_vals.dtype.is_uint64()
                or inp_vals.dtype.is_fp64()):
    inp_vals = inp_vals
elif tl.constexpr(inp_vals.dtype.is_int()):
    inp_vals = inp_vals.to(tl.int32)
else:
    inp_vals = inp_vals.to(tl.float32)
```

dtype 语义与 torch 对齐：cumsum(int32) 输出仍 int32（包括溢出行为——torch 的 int32 cumsum 溢出会回绕，kernel 也应如此；测试 golden 别随手 .float()，会掩盖溢出语义差异）；int64/fp64 保持原 dtype；其余整数升 int32、其余浮点升 f32。compute 维度上这是"与 torch 语义逐位对齐"的正确性优化。

## 优化二：排他前缀先写再累加，顺序不能反

```python
# 优化前（写反）：inclusive 前缀，每块偏一个块和
acc = 0.0
for i in range(0, P):
    acc = acc + tl.load(block_sums + i, ...)
    tl.store(offsets + i, acc)          # 写的是含自己的和

# 优化后（排他前缀）：先写（不含自己）再累加
acc = 0.0
for i in range(0, P):
    tl.store(offsets + i, acc)          # 先写：前面所有块的总和
    acc = acc + tl.load(block_sums + i, ...)   # 再加上自己
```

Phase2 的循环体内先 sstore 再累加是排他前缀（exclusive）的关键：offsets[p] 必须是"前面所有块的总和"，不含自己。写反顺序整个输出平移一个块和。compute 维度上这是"前缀语义正确"的优化。

## 优化三：整数 scan 必须逐位相等，golden 不 float

```python
# 优化前：golden 随手 .float()，掩盖溢出语义差异
ref = torch.cumsum(xi.float(), dim=0)

# 优化后：整数 scan 逐位相等
got, ref = cumsum(xi), torch.cumsum(xi, dim=0)
assert torch.equal(got, ref)                 # 整数 scan 必须逐位相等
```

整数 scan 必须逐位相等（torch.equal），golden 不 float。compute 维度上这是"溢出行为与 torch 对齐"的验证优化。

## 优化四：行长为寄存器宽边界时调 BLOCK 避开

```python
# 优化前：行长恰好等于寄存器宽（len 15/16 类）→ rc=134 断言崩溃
# 优化后：调整 BLOCK 让行长避开边界
```

行长与向量寄存器宽度的边界关系历史上出过编译器级 bug（ops/06 §6.5 表、ops/09 §9.5 层 2）：行长恰好等于寄存器宽、行长不是寄存器宽整倍数、行长超大（1041 → BLOCK 2048 类）都可能 rc=134 断言崩溃。cache 会掩盖，先换新 TRITON_CACHE_DIR 再下结论。compute 维度上这是"避开编译器边界 bug"的优化。

## 优化五：运行时栈空间，int64 + 大 TILE 用新 wheel

```python
# 优化前：旧版 wheel 协程栈 16KB，int64 + TILE=1024 的 scan 中间量 ~128KB → SIGSEGV
# 优化后：新版 wheel 栈 512KB 量级，执行期崩溃消失
```

Hillis-Steele 每轮分配 (TILE + 2^r) × elemSize 的中间量，kernel 在运行时协程栈上跑。旧版 wheel 栈偏小，int64 + TILE=1024 要 ~128KB，栈越界 SIGSEGV。新版已扩大。compute 维度上这是"运行时执行环境够用"的优化。

## 优化六：分段探针法定位复合算子失败

```python
# repeat_interleave(x, repeats) 拆三段独立跑：
#   段1: cumsum(repeats)        # 算输出偏移
#   段2: repeat_kernel(x, ...)  # 搬运
#   段3: index_select(...)      # 收集
```

复合算子失败时第一反应往往是怀疑最深的那层（scan lowering），而这通常是错的。真实案例：repeat_interleave 数值失败曾长期被归因于 scan，分段探针后失败全在段3 index_select。compute 维度上这是"定位失败层"的方法论优化。

## 优化七：算术扫描先确认"浮点过、整数崩"判据

```python
# 整数 dtype 的 scan/归约断言崩溃判据：
# 同一 kernel，fp16/fp32 全过、int dtype（i1/i16/i32/i64）全崩（rc=134）
# 这是旧版编译器对整型元素的处理限制
```

判据清晰：同一 kernel，fp16/fp32 全过、int dtype 全崩 = 旧版编译器对整型元素的处理限制。碰到先升级 wheel，仍崩按流程报告，附上判据。compute 维度上这是"判断失败归属"的优化。

## 优化前后 compute 对照

| 项 | naive | 优化后 | 收益 |
|---|---|---|---|
| dtype | 不提升 | 按提升表 | 与 torch 对齐 |
| 前缀顺序 | 先累加再写 | 先写再累加 | 排他语义正确 |
| 整数验证 | golden float | torch.equal | 溢出语义对齐 |
| 行长边界 | 崩 | 调 BLOCK 避开 | 不崩 |
| 栈空间 | 旧 wheel SIGSEGV | 新 wheel | 执行不崩 |
| 复合算子失败 | 直接怀疑 scan | 分段探针 | 定位到段 |

## 验证 compute 优化生效

跑 cumsum.py 的 f32 n=5/64/1000/4096 和 int32 n=1000（exact）；把 part_num 改成 4/8 验证 partial_sum[pid-1] 是否还严格前缀；行长为 15/16/17 的二维 cumsum 记录哪些能过哪些崩；跑 test_raw_cumsum_vec.py 对照三阶段。

## 三个维度的关系

dtype 提升既是 compute 语义对齐也影响 utilization 的 lane 宽度；排他前缀顺序既是 compute 正确也影响 memory 的 out 读改写顺序；行长边界既是 compute 崩溃也影响 memory 分块。调优顺序仍是 memory 分块趟数先行、utilization 块间并行其次、compute dtype 与前缀语义收尾。