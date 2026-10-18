# GEMM — Utilization 方向优化
使用Triton代码
> 优化前：`gemm.py` 用 `tl.dot`（自动映射矩阵引擎），无显式微块打包、无 packed 累加优化。
> 优化后：`test_smt_mm.py`（smt 显式编排，SPLIT_K 分支 + MICRO 打包）。

---

## 优化前代码（无 Utilization 优化）

```python
a = tl.load(a_block_ptr, boundary_check=(0, 1))
b = tl.load(b_block_ptr, boundary_check=(0, 1))
acc = tl.dot(a, b)          # f16 @ f16 → f32，自动提升
tl.store(c_block_ptr, acc.to(c_ptr.dtype.element_ty),
         boundary_check=(0, 1))
```

---

## 优化后代码（SPLIT_K：预打包累加 + packed 累加 + 循环外一次 unpack）

```python
elif SPLIT_K:
    # 累加器先打包：循环里 packed += packed 才能直接相加
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                           dtype=a_ptr.type.element_ty)
    accumulator = smt.view(accumulator, (0, 0),
                           (BLOCK_SIZE_M, BLOCK_SIZE_N),
                           (MICRO_M, MICRO_N))
    sub_num = (K + SUB_BLK_K - 1) // SUB_BLK_K
    for k in tl.range(0, sub_num):
        a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
        a = smt.view(a_descriptor_load, (0, k * SUB_BLK_K),
                     (BLOCK_SIZE_M, SUB_BLK_K), (MICRO_M, MICRO_K))
        b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
        b = smt.view(b_descriptor_load, (k * SUB_BLK_K, 0),
                     (SUB_BLK_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))
        accumulator += smt.dot(a, b)
    # 循环外一次性 unpack——比每轮 unpack 便宜
    accumulator = smt.view(accumulator, (0, 0),
                           (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1))
    c = accumulator.to(c_ptr.dtype.element_ty)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
        offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N], order=[1, 0])
    tl.store(c_block_ptr, c, boundary_check=(0, 1))
```

---

## 优化点拆解（Utilization 方向）

**① 走矩阵引擎（smt.dot / mmt4d / vfwmadot）**
- 方法：用 `smt.dot(a, b)` 替代 `tl.dot(a, b)`。
- 为什么是 Utilization 优化：`tl.dot` 只有一条自动映射通路；`smt` 命名空间让你手动编排，走矩阵引擎直通路径，让矩阵引擎以峰值吞吐工作。
- Roofline：这一步同时**抬高 compute roof**（矩阵引擎 P_peak ≫ 标量 FMA），并决定**能不能贴近这个 roof**。

**② 按 MICRO_M/K/N = 16/8/32 打包**
- 方法：`smt.view(x, offsets, shape, (MICRO_M, MICRO_K))` 把 A/B 打包成硬件微块布局 4-D 张量。
- 为什么是 Utilization 优化：K3 f16 硬件 micro tile 是 16/8/32，打包不对 → 退化成低效路径 → achieved FLOPs 远低于 compute roof。
- Roofline：打包是 “能不能贴上 compute roof” 的关键。

**③ 累加器预打包 + 循环内 packed 累加 + 循环外一次 unpack（SPLIT_K）**
- 方法：`smt.view` 把 `tl.zeros` 出来的普通零矩阵预打包成 packed 格式；循环内 `accumulator += smt.dot(a, b)` 是 packed + packed 直接相加；循环外 `smt.view(..., (1,1))` 一次性解包。
- 为什么是 Utilization 优化：`smt.dot` 返回 packed 格式；如果每轮 dot 后都 unpack 再累加，多出大量布局转换指令，不产生 FLOPs 但占周期。预打包累加 + 循环外一次 unpack 把这部分开销降到最低。
- Roofline：减少非计算指令开销 → 提高 achieved FLOPs 相对 compute roof 的比例。

**④ SPLIT_M 子块独立、无跨迭代累加器**
- 方法：每个 M 子块一次 `smt.dot` 结果就完整，立即 unpack、store；循环内没有累加器跨迭代存活。
- 为什么是 Utilization 优化：减少寄存器压力和依赖链，让矩阵引擎流水更满。
- Roofline：utilization 的调度层优化。

**⑤ constexpr 三选一切分策略**
- 方法：`SPLIT_M / SPLIT_MN / SPLIT_K` 三个 `tl.constexpr` 开关，host 传 0/1 三选一，编译器把没走到的分支整个删掉。
- 为什么是 Utilization 优化：针对不同形状选最能把引擎喂满的调度，零成本特化。
- Roofline：utilization 的调度层优化。

**⑥ tl.dot 的 f16 输入自动得到 f32 输出**
- 方法：`acc = tl.dot(a, b)`，f16 输入自动 f32 累加。
- 为什么是 Utilization 优化：K3 矩阵引擎硬件本身就是 f16 乘 f32 累加（vfwmadot），这是硬件原生行为，引擎以峰值吞吐工作。
- Roofline：utilization。