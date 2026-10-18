# GEMV — Memory 方向优化
使用Triton代码
> 优化前：`mv_kernel`（标准 tl 写法，每 program BLOCK_N 行，二维 acc + 最后归约）。
> 优化后：`mv_block_style2`（raw 向量版，4 行展开 + 主/尾双循环 strip-mine + phantom 吸收）。

---

## 优化前代码（无 Memory 优化）

```python
@triton.jit(do_not_specialize=["N", "K"])
def mv_kernel(W, x, y, N, K,
              stride_wn, stride_wk, stride_xk, stride_yn,
              BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_k = tl.arange(0, BLOCK_K)[None, :]
    n_mask = offset_n < N

    W_ptrs = W + offset_n * stride_wn + offset_k * stride_wk
    x_ptrs = x + offset_k * stride_xk

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_mask = k + offset_k < K
        w = tl.load(W_ptrs, mask=n_mask & k_mask, other=0.0).to(tl.float32)
        v = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        acc += w * v
        W_ptrs += BLOCK_K * stride_wk
        x_ptrs += BLOCK_K * stride_xk

    row_sum = tl.sum(acc, axis=1)
    tl.store(y + offset_n * stride_yn, row_sum[:, None], mask=n_mask)
```

---

## 优化后代码（raw 向量版：4 行展开 + strip-mine + phantom 吸收）

```python
@tle.raw_kernel
def mv_block_style2(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f32, out=True),
                    K: tle.index, row_base: tle.index, row_end: tle.index):
    nvl = tle.vconfig(-1, 1)             # f16 VLMAX = 64
    Kfloor = (K // nvl) * nvl            # 满 tile 覆盖的 K 区间
    for ni in tle.range(row_base, row_end, 4):        # 每次 4 行
        acc0 = tle.vzero(f32); acc1 = tle.vzero(f32)
        acc2 = tle.vzero(f32); acc3 = tle.vzero(f32)
        for ki in tle.range(0, Kfloor, nvl):          # 主循环：满 tile 快路
            va = tle.vload(A, ki)                     # x 段，4 行共用！
            vb0 = tle.vload(B, ni * K + ki)           # B 第 ni 行段
            vb1 = tle.vload(B, (ni + 1) * K + ki)
            vb2 = tle.vload(B, (ni + 2) * K + ki)
            vb3 = tle.vload(B, (ni + 3) * K + ki)
            acc0 = tle.vmacc(acc0, vb0, va)           # acc += v * 标量广播
            acc1 = tle.vmacc(acc1, vb1, va)
            acc2 = tle.vmacc(acc2, vb2, va)
            acc3 = tle.vmacc(acc3, vb3, va)
        for ki in tle.range(Kfloor, K, nvl):          # 尾循环：0 或 1 次
            nvl = tle.vconfig(K - ki, 1)              # 收窄 → fill-0 pad
            ta = tle.vload(A, ki)                     # 独立临时名！
            tb0 = tle.vload(B, ni * K + ki)
            # ... tb1/tb2/tb3 同构，acc0..3 继续 vmacc
        tle.sstore(C, ni,     tle.vreduce_sum(acc0))  # 水平归约 → 标量 store
        tle.sstore(C, ni + 1, tle.vreduce_sum(acc1))
        tle.sstore(C, ni + 2, tle.vreduce_sum(acc2))
        tle.sstore(C, ni + 3, tle.vreduce_sum(acc3))
```

---

## 优化点拆解（Memory 方向）

**① 4 行展开：x 的访存摊薄 4 倍（核心优化）**
- 方法：`va = tle.vload(A, ki)` 只 load 一次 x 的 K 段，然后喂给 4 个 `vmacc`（`acc0..acc3`），每个 vmacc 对应 B 的一行段。
- 为什么是 Memory 优化：x 是 K 个元素被 N 行共享，展开 4 行让每次 x 读取服务 4 倍计算。这是 memory-bound 算子的经典手法（提高每字节搬运的计算复用）。
- Roofline：最直接的提 AI，x 读一次服务 4 行，有效 AI 上升，工作点沿横轴右移。

**② 主/尾双循环 strip-mine：用迭代次数代替分支**
- 方法：`Kfloor = (K // nvl) * nvl`，主循环 `range(0, Kfloor, nvl)` 走满 tile 快路（零 pad 零分支）；尾循环 `range(Kfloor, K, nvl)` 用 `vconfig(K-ki)` 收窄后 vload 自动 fill-0。整除时尾循环天然跑 0 次。
- 为什么是 Memory 优化：主循环里没有 `if`（default path 无分支），减少控制流开销，让向量单元流水不被打断，提高有效 BW 的达成率。
- Roofline：提高 achieved BW 贴近 memory roof。

**③ phantom 行吸收：免 mask**
- 方法：`N % BLOCK != 0` 时末 program 的 4 行组可能超出 N，`sstore` 无条件执行；host 把 C 分配到 `Np = ceil(N/BLOCK)*BLOCK` 吸收 phantom 写，取结果切 `C[:N]`。
- 为什么是 Memory 优化：用“多写一点被吸收”换掉“每次 store 都带 mask 判断”，省掉边界检查的分支开销。多分配、不越界、零 copy。
- Roofline：减少非搬运算的指令周期，提高 BW 达成率。

**④ 归约推迟到最后（优化前代码里已有，优化后保持）**
- 方法：`acc` 是 `(BLOCK_N, BLOCK_K)` 二维，循环内每个 lane 各自累加，最后 `tl.sum(acc, axis=1)` 做一次水平归约。
- 为什么是 Memory 优化：循环体内全是纯向量乘加，没有归约开销，不因归约打断向量流水。
- Roofline：提高有效 BW 的达成率。

**⑤ other=0.0（边界零代价）**
- 方法：`tl.load(..., mask=..., other=0.0)`。
- 为什么是 Memory 优化：越界 lane 读成 0，乘加贡献为 0，对求和归约是吸收元——边界处理零额外代价，不增加 Bytes。
- Roofline：不增加 Bytes。