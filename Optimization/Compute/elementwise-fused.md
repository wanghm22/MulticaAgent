# Elementwise 与融合 — Compute 维度优化

Roofline 视角：逐元素 compute 不是瓶颈，但 math 函数 dtype 检查、libdevice 链路、gelu 的 constexpr 坑都会让"计算直接失败"。本章把逐元素与融合在 compute 维度的优化前后代码完整摆出来。

## 优化前：f16 直接喂 sigmoid/exp

```python
@triton.jit
def swiglu_naive_compute(G, U, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + offs, mask=offs < N, other=0.)   # fp16
    u = tl.load(U + offs, mask=offs < N, other=0.)
    y = (g * tl.sigmoid(g)) * u                       # sigmoid 内部走 exp，只收 f32/fp64
    tl.store(Y + offs, y, mask=offs < N)
```

tl.sigmoid 内部走 exp，只收 fp32/fp64（math.py:_check_dtype，上游故意设计，GPU 同败）。f16 直接喂 = CompilationError。这是 K3 逐元素算子的第一大坑（ops/07 §7.4）。

## 优化一：math 函数前升 f32

```python
g = tl.load(G + offs, mask=offs < N, other=0.).to(tl.float32)
u = tl.load(U + offs, mask=offs < N, other=0.).to(tl.float32)
y = (g * tl.sigmoid(g)) * u            # 全程 f32
tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=offs < N)
```

进 math 函数（sigmoid→exp）前升 f32，是硬性要求。f32 输入时这行是无害 no-op，写上让同一 kernel 两种 dtype 通吃。compute 维度上这是"用正确精度调 math 函数"的前提。

## 优化二：epilogue 天然全程 f32，dtype 约定自动满足

```python
acc = tl.zeros((BM, BN), dtype=tl.float32)
for k in range(0, K, BK):
    acc += tl.dot(a, b)
c = acc + tl.load(BIAS + offs_n)[None, :]     # acc 是 f32，bias 广播
c = c * tl.sigmoid(c)                         # 天然 f32，不撞 _check_dtype
tl.store(C_ptrs, c.to(tl.float16), mask=...)
```

epilogue 运算天然全程 f32（acc 是 f32），dtype 约定自动满足，cast 只在最后写出一刻。compute 维度上这是融合的额外好处：中间累加器本来就是 f32，math 函数不用额外 cast。

## 优化三：gelu tanh 近似的 constexpr 坑

```python
# 优化前：字面量 pow(x, 2) 的 2 是 constexpr，直调类型提升函数前崩
y = 0.5 * x * (1 + tl.math.tanh(0.79788456 * (x + 0.044715 * x.pow(3))))
# 'constexpr_type' object has no attribute 'scalar'

# 优化后：用乘法代替 .pow()，或先 isinstance 解包
y = 0.5 * x * (1 + tl.math.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
```

JIT 内字面量 pow(x, 2) 的 2 是 constexpr，直调类型提升函数前须 isinstance(x, core.constexpr) 解包 .value（gelu_backward 24 例编译崩的根因，ops/07 §7.4）。compute 维度上，写 x*x*x 代替 x.pow(3) 最省心；tl.math.tanh 的输入必须已是 f32。

## 优化四：libdevice 链路白名单与降级

```python
# 优化前：用了 acosh/asinh/atanh/cbrt 这些残留不 lower 的函数
# Dialect 'math' not found（math op 没被降级到底）

# 优化后：用可 lower 清单内的函数
# acos/asin/atan/cos/sin/tan/cosh/sinh/tanh/exp2/expm1/log2/log10/log1p/cttz
```

K3 工具链 math 链路分层：tl.sigmoid/tanh → math.py（_check_dtype）→ libdevice（@core.extern → extern_elementwise → MLIR math dialect）→ 编译器白名单 → 后端降级。acosh/asinh/atanh/cbrt 残留不 lower，新 kernel 避开或先确认。compute 维度上这是"用工具链支持的 math 函数"的优化。

## 优化五：ffs 语义、rint 映射等 shim 坑

```python
# ffs：CUDA 语义 1-based、0 返回 0，用 math.cttz + select 实现
# op 注册名是 math.cttz/ctlz，没有 "math.ctz"
# rint = round-half-to-even，不能映射 math.round
```

这些是 libdevice shim 层的坑（ops/07 §7.4 表）。compute 维度上，遇到"函数不存在/语义不对"先查 shim 形态和后端白名单，别在 kernel 源码里瞎改。

## 优化六：审计全量 math 调用点

```python
# 全量 grep tl.exp|tl.log|tl.sqrt|libdevice. 调用点
# 确认每处都有前置 cast / f32 标量提升 / COMPUTE_FP32 门控之一
```

接手一批逐元素算子时，审计所有 math 调用点，确认每处都有前置 cast / f32 标量提升 / COMPUTE_FP32 门控之一。FlagGems 约 40 个调用点审计结论：有这四类保护的都安全。compute 维度上这是系统性的正确性优化。

## 优化七：别改 math.py 全局自动提升

```python
# 别走这条路：给 math.py 加全局自动提升
# 上游 FlagGems 曾两次尝试修复又都 revert——影响面不可控
```

生产环境当 kernel 源码不许改时，三种解法：kernel 内显式 .to(tl.float32)；heuristic 路由（fp16/bf16 返回 False 强制走 multi-tile）；给 math.py 加全局自动提升（被否决，别走）。compute 维度上这是"用受支持的修复路径"的优化。

## 优化前后 compute 对照

| 项 | naive | 优化后 | 收益 |
|---|---|---|---|
| sigmoid 输入 | f16 编译失败 | 先升 f32 | 合法 |
| epilogue | 可能额外 cast | acc 天然 f32 | dtype 自动满足 |
| gelu 多项式 | x.pow(3) 前端崩 | x*x*x | 前端不崩 |
| libdevice | acosh 等不 lower | 用可 lower 清单 | 后端能降级 |
| ffs/rint | shim 语义错 | 查白名单/shim | 语义正确 |
| 修复路径 | 改 math.py | kernel 内 cast/路由 | 影响面可控 |

## 验证 compute 优化生效

删掉 .to(tl.float32) 重跑抄下 _check_dtype 报错原文；跑 test_mm_silu.py 后 dump IR 确认 sigmoid 变成 exp 相关 math op；用 gelu tanh 近似公式实现 kernel 对 golden torch.nn.functional.gelu(x, approximate='tanh')；全量 grep math 调用点审计。

## 三个维度的关系

融合既是 memory 优化（省趟）也影响 compute（epilogue 天然 f32）；grid 封顶既是 utilization（不空转）也间接是 memory（不读垃圾）。调优顺序仍是 memory 融合趟数先行、utilization 并行和 grid 安全其次、compute math 链路与 dtype 收尾。