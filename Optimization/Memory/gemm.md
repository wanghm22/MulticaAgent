# GEMM — Memory 方向优化
使用Triton代码
> 优化前：`gemm.py`（单 K 块、block_ptr 版），无分块复用、无 TCM 中转，B 被每个 program 从 DDR 重复读取。
> 优化后：`test_smt_mm.py`（smt 显式编排，SPLIT_M / SPLIT_MN 分支）。

---

## 优化前代码（无 Memory 优化）

```python
@triton.jit
def mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
              stride_am, stride_ak, stride_bk, stride_bn,
              stride_cm, stride_cn,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0))
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0))

    a = tl.load(a_block_ptr, boundary_check=(0, 1))
    b = tl.load(b_block_ptr, boundary_check=(0, 1))
    acc = tl.dot(a, b)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    tl.store(c_block_ptr, acc.to(c_ptr.dtype.element_ty),
             boundary_check=(0, 1))
```

---

## 优化后代码（SPLIT_M：B 循环外复用 + SPLIT_MN：TCM 中转）

```python
# SPLIT_M 分支：B 循环外 load+pack 一次，所有 M 子块复用
if SPLIT_M:
    b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
    b = smt.view(b_descriptor_load, (0, 0),
                 (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))
    sub_num = (min(BLOCK_SIZE_M, M - BLOCK_SIZE_M * pid_m) + SUB_BLK_M - 1) // SUB_BLK_M
    for s in smt.parallel(0, sub_num):
        a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
        a = smt.view(a_descriptor_load, (s * SUB_BLK_M, 0),
                     (SUB_BLK_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))
        accumulator = smt.dot(a, b)
        accumulator = smt.view(accumulator, (0, 0),
                               (SUB_BLK_M, BLOCK_SIZE_N), (1, 1))
        c = accumulator.to(c_ptr.dtype.element_ty)
        c_block_ptr = tl.make_block_ptr(
            base=c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
            offsets=[pid_m * BLOCK_SIZE_M + s * SUB_BLK_M,
                     pid_n * BLOCK_SIZE_N],
            block_shape=[SUB_BLK_M, BLOCK_SIZE_N], order=[1, 0])
        tl.store(c_block_ptr, c, boundary_check=(0, 1))

# SPLIT_MN 分支：TCM 片上缓冲中转 B 列条
elif SPLIT_MN:
    b_alloc_ptr = smt.alloc(shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                            type=b_ptr.dtype.element_ty)
    b_alloc_view_ptr = smt.view(b_alloc_ptr, (0, 0),
                                (BLOCK_SIZE_K, BLOCK_SIZE_N),
                                (MICRO_K, MICRO_N))
    for s in smt.parallel(0, total_sub_blocks):
        s_m = s // sub_num_n
        s_n = s % sub_num_n
        a = smt.view(a_descriptor_load, (s_m * SUB_BLK_M, 0),
                     (SUB_BLK_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))
        b_alloc_sub_ptr = smt.view(b_alloc_view_ptr,
                                   (0, s_n * SUB_BLK_N),
                                   (BLOCK_SIZE_K, SUB_BLK_N))
        if s_m == 0:   # 每个列条只有第一行从 DDR 读 B，pack 后存进 TCM
            b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
            b = smt.view(b_descriptor_load, (0, s_n * SUB_BLK_N),
                         (BLOCK_SIZE_K, SUB_BLK_N), (MICRO_K, MICRO_N))
            tl.store(b_alloc_sub_ptr, b, boundary_check=(0, 1, 2, 3))
        # 其余 s_m 都从 TCM 读 B（packed 进、packed 出，不解包）
        b_alloc = tl.load(b_alloc_sub_ptr, boundary_check=(0, 1, 2, 3))
        accumulator = smt.dot(a, b_alloc)
        accumulator = smt.view(accumulator, (0, 0),
                               (SUB_BLK_M, SUB_BLK_N), (1, 1))
        c = accumulator.to(c_ptr.dtype.element_ty)
        c_block_ptr = tl.make_block_ptr(
            base=c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
            offsets=[pid_m * BLOCK_SIZE_M + s_m * SUB_BLK_M,
                     pid_n * BLOCK_SIZE_N + s_n * SUB_BLK_N],
            block_shape=[SUB_BLK_M, SUB_BLK_N], order=[1, 0])
        tl.store(c_block_ptr, c, boundary_check=(0, 1))
```

---

## 优化点拆解（Memory 方向）

**① B 循环外 load+pack 一次复用（SPLIT_M）**
- 方法：把 B 的 `load + smt.view 打包` 移到 `for s in smt.parallel(...)` 循环之外，循环内只取 `a` 子块。
- 为什么是 Memory 优化：B 是 M 方向所有子块的共享操作数，循环外读一次、pack 一次，所有子块复用。优化前每个 program 从 DDR 读 B；优化后 B 只读一次。直接降低 DDR Bytes、提高有效 AI。
- Roofline：把工作点沿横轴右推，脱离 memory roof 斜线。

**② TCM 片上缓冲中转 B 列条（SPLIT_MN）**
- 方法：用 `smt.alloc(...)` 在片上 TCM 开一块 scratch（`b_alloc_ptr`）；`s_m == 0` 时（每个列条的第一行）从 DDR 读 B 存进 TCM；其余 `s_m` 全从 TCM 读。
- 为什么是 Memory 优化：B 的一个 `[K × SUB_BLK_N]` 列条要被所有 M 子块用，优化前每次从 DDR 读太浪费；优化后读一次进 TCM、片上复用多次。DDR 侧 Bytes 大降。
- Roofline：最纯粹的 memory 优化，把 kernel 往 compute-bound 区推。

**③ B 全程 packed 进出 TCM，不解包**
- 方法：`tl.store(b_alloc_sub_ptr, b, ...)` 和 `tl.load(b_alloc_sub_ptr, ...)` 都作用在 4-D packed pointer 上，`smt.dot` 直接吃 packed 形态。
- 为什么是 Memory 优化：减少布局转换（pack/unpack）带来的额外读写——B 从 DDR 到 TCM 再到 dot，全程只有一次布局，不反复转换。
- Roofline：降 Bytes + 降指令。

**④ block_ptr + boundary_check 替代手写 mask**
- 方法：`tl.make_block_ptr(...)` 配 `tl.load(..., boundary_check=(0,1))`。
- 为什么是 Memory 优化：block_ptr 能 lower 成整块的 `tensor.extract_slice`，K3 上能用更宽的向量 load 指令；手写 mask 在 K3 上没有对应指令，会退化成逐元素判断。
- Roofline：同样 Bytes，用更少指令、更接近带宽上限——提高 memory roof 的达成率。

**⑤ 显式传 6 个 stride，支持非连续**
- 方法：host 侧把 `a.stride(0), a.stride(1), b.stride(0), ...` 全部传给 kernel，kernel 不假设 tensor 连续。
- 为什么是 Memory 优化：切片、转置等非连续视图能直接算，不用先 `.contiguous()`——省掉一次全量读写的拷贝。
- Roofline：直接降 Bytes。

**⑥ K 大时分块循环（BK）**
- 方法：`for k in range(0, tl.cdiv(K, BK))`，每次取 `BM×BK` 和 `BK×BN` 两块乘加。
- 为什么是 Memory 优化：`BK=K` 一次展开需 608KB，超 TCM 容量会溢出 hang；分块让中间量留在片上，不溢到 DDR。
- Roofline：受 TCM 容量约束的访存组织。