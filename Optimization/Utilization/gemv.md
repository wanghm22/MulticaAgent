# GEMV — Utilization 方向优化
使用Triton代码
> 优化前：标准 tl 写法（按 N 切块，grid = cdiv(N, BLOCK_N)，每 program BLOCK_N 行）。
> 优化后：vfwmacc 版（转置映射 + 大 NB 少 program + 4×64 微块 + 硬件 spestruct.pack）。

---

## 优化前代码（无 Utilization 优化）

```python
def mv(W, x):
    N, K = W.shape
    y = torch.empty((N,), device=W.device, dtype=W.dtype)
    BLOCK_N, BLOCK_K = 32, min(128, triton.next_power_of_2(K))
    grid = (triton.cdiv(N, BLOCK_N),)   # grid 随 N 线性增长
    mv_kernel[grid](W, x, y, N, K, ..., BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return y
```

---

## 优化后代码（vfwmacc 版：转置映射 + 大 NB 少 program + 4×64 微块 + 硬件 pack）

```python
# 转置映射：y[N] = W[N,K] @ x[K]  ⇔  y[1,N] = x[1,K] @ W^T[K,N]
# raw 层 load_2d_t 用 strided view（strides=[1,N]，硬件 vlse strided 读）
# 直接从行主序 W 里按转置布局读，不需要 host 侧真转置

# 大 NB（少 program）：生产选 NB=256（grid 4），而非 NB=64（grid 16）
# kernel 内再切 4×64 sub-tile：NB=256 的块内用 4 个独立 vector<1×64> acc
# 每 K-block 做 4 次 pack + batch_macc

# A^T pack 用硬件 spestruct.pack（对标 mm 的 pack 例程），
# 取代软件 linalg.generic 循环

# 动态 M：kernel 内 for kb in tle.range(nk)（nk=M/32）K 块循环，
# 列 stride 支持动态 SSA
# 约束：M%32==0 && N%256==0 && f16 && contiguous
```

---

## 优化点拆解（Utilization 方向）

**① 转置映射绕开 batch_macc 的 n≥64 约束**
- 方法：`y[N]=W[N,K]@x[K]`（n=1，不满足）⇔ `y[1,N]=x[1,K]@W^T[K,N]`（n=N，满足）。`W^T` 不需要 host 侧真转置——raw 层 `load_2d_t` 原语用 strided view（`strides=[1,N]`，硬件 vlse strided 读）直接从行主序 W 按转置布局读。
- 为什么是 Utilization 优化：矩阵引擎原子块是 MICRO 16×8×32，M=1 时 16 行里 15 行空，利用率 1/16；pack/unpack 开销相对 1 行计算变得巨大。转置映射让形状满足 `vfwmacc`（batch_macc）约束，零数据搬移。
- Roofline：把“不满足硬件约束的形状”映射到“满足约束的等价形状”，纯“绕约束”，不增 Bytes。

**② 大 NB（少 program）反而更快（调度优化）**
- 方法：实测 NB=64（grid 16）0.165ms、NB=256（grid 4）0.118ms、NB=512/1024 持平；生产选 NB=256。
- 为什么是 Utilization 优化：K3 上 dispatch 开销主导，“更多 program 更并行”的直觉在这里失效。
- Roofline：减少 dispatch 开销，让搬运算的时间占比更高，更贴 memory roof。

**③ kernel 内再切 4×64 sub-tile（微块尺寸优化）**
- 方法：NB=256 的块内用 4 个独立 `vector<1×64>` acc，每 K-block 做 4 次 pack + batch_macc。
- 为什么是 Utilization 优化：64 = f16 一个 scalable 寄存器宽（macc 零吞吐损失）；32 不足一个寄存器宽、会触发编译器崩溃；每块 buf 4KB << L1 32KB。
- Roofline：微块对齐硬件寄存器宽度 + 不超 L1，同样 Bytes 下计算不拖后腿。

**④ 硬件 spestruct.pack 替代软件 linalg.generic 循环**
- 方法：A^T pack 用硬件 `spestruct.pack`，取代软件 `linalg.generic` 循环。
- 为什么是 Utilization 优化：旧软件 pack 在大 M 下是灾难（M=512 只有 mainline 的 0.35x），硬件 pack 救回 1.17x（~3.3x 加速）。
- Roofline：用硬件原语替代软件循环，直接提高搬运效率、逼近带宽峰值。

**⑤ 动态 M + 明确的适用边界**
- 方法：kernel 内 `for kb in tle.range(nk)`（nk=M/32）K 块循环，列 stride 支持动态 SSA。约束：M%32==0 && N%256==0 && f16 && contiguous。
- 为什么是 Utilization 优化：性能总账（N=4096）：M=32 时 2.10x，M=2048 时 1.15x，M=4096 略输 3%——M 大了以后它本质上在跟 GEMM 抢通路，交给 `tl.dot/smt.dot` 更合适。
- Roofline：边界意识，GEMV 专用 kernel 的价值区间就是 M 小（decode）的场景。