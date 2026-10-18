# Attention — Memory 方向优化
使用Triton代码
> 优化前：分步 attention（QK^T → mask → softmax → AV 各自独立 kernel，中间矩阵落内存）。
> 优化后：单 tile fused（8.4）+ flash 风格 tiling + online softmax（8.5）+ GQA 共享 KV（8.1）。

---

## 优化前代码（分步实现，中间结果落内存）

```python
# 分步：QK^T 写回内存 → mask 再读再写 → softmax 三趟 → AV 再读 P 和 V
S = torch.matmul(Q, K.transpose(-1, -2)) * scale   # (M,M) 落内存
S = S.masked_fill(~causal_mask, float('-inf'))      # 再读写一遍
P = torch.softmax(S, dim=-1)                        # 再读 S 写 P
O = torch.matmul(P, V)                              # 再读 P 和 V 写 O
```

---

## 优化后代码（单 tile fused + flash tiling + online softmax）

```python
# 单 tile fused（8.4）：QK^T、mask、softmax、AV 全在一个 kernel
@triton.jit(do_not_specialize=["M", "D"])
def attn_fwd_kernel(Q, K, V, O, sm_scale, M, D,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * M * D
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    ...
    q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :],
                mask=m_mask[:, None], other=0.)
    k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :],
                mask=n_mask[:, None], other=0.)
    qk = tl.dot(q, tl.trans(k)) * sm_scale          # QK^T，寄存器内转置
    causal = offs_m[:, None] >= offs_n[None, :]
    qk = tl.where(causal, qk, float('-inf'))        # mask，-inf 不落内存
    m_i = tl.max(qk, axis=1)                        # softmax 在寄存器里
    p = tl.exp(qk - m_i[:, None])
    l_i = tl.sum(p, axis=1)
    v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :],
                mask=n_mask[:, None], other=0.)
    acc = tl.dot(p.to(tl.float16), v) / l_i[:, None] # AV
    tl.store(O + ..., acc.to(O.dtype.element_ty), ...)


# flash 风格 tiling + online softmax（8.5）：M 大时 K/V 沿序列切块流式扫
m_i  = 每行当前见过的最大分数（初始 -inf）
l_i  = 每行当前的 exp 和
acc  = 当前的 P·V 部分和（未归一化）

for kv_block in K/V 的块:
    qk = smt.dot(q, trans_k) * sm_scale          # 本块分数
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1))   # 新的行 max
    p = tl.math.exp(qk - m_ij)                   # 用新 max 重算本块概率
    alpha = tl.math.exp(m_i - m_ij)              # 旧结果的修正系数 ≤1
    l_i = l_i * alpha + tl.sum(p, axis=1)        # 分母同步修正
    acc = acc * alpha                            # 灵魂：旧 acc 整体缩放
    acc += smt.dot(p_cast, v)                    # 加上新块贡献
    m_i = m_ij

out = acc / l_i                                  # 扫完才归一化，一次除法

# GQA 共享 KV（8.1）：不物化复制 KV
off_hkv = off_hq // GROUP_SIZE
```

---

## 优化点拆解（Memory 方向）

**① 单 tile fused：中间结果不落内存**
- 方法：QK^T、mask、softmax、AV 全部在一个 kernel 里完成，`qk`、`p` 都是寄存器里的 tile，不写回全局内存。
- 为什么是 Memory 优化：优化前 S（M×M）、P（M×M）都落内存，每个都是完整读写一遍；融合后省掉这些 N×M 级中间矩阵的多趟访存。
- Roofline：降 Bytes。

**② flash 风格 tiling：QK^T 的 M×M 矩阵不物化落内存**
- 方法：M 大时 K/V 沿序列切块流式扫，边扫边维护运行中的 max/sum/acc，`QK^T` 的 M×M 矩阵不物化。
- 为什么是 Memory 优化：优化前 M=1024 时 qk 是 1024×1024 f32 = 4MB，装不下且要完整读写；flash tiling 让它始终只存在于寄存器/TCM。
- Roofline：降 Bytes，M×M 中间矩阵不落全局内存。

**③ GQA 共享 KV：不物化复制 KV**
- 方法：`off_hkv = off_hq // GROUP_SIZE`，第 hq 个 Q head 用第 `hq // GROUP_SIZE` 组 KV。
- 为什么是 Memory 优化：多个 Q head 共享一组 KV，优化前如果把 KV 物化复制 Hq/Hkv 份，内存和带宽都翻倍；一次整数除法搞定，不需要复制。
- Roofline：省内存也省带宽。

**④ online softmax 的 alpha 修正：只需一遍扫描**
- 方法：max 变了时，把 `acc` 和 `l_i` 同乘 `alpha = exp(m_old - m_new)` 修正；数学上严格等价于“先看完全部再 softmax”。
- 为什么是 Memory 优化：优化前要先把整行读两遍（一遍求 max、一遍求 sum）；online 算法只需一遍扫描。
- Roofline：减少访存趟数。

**⑤ tl.trans 寄存器内转置：不产生访存**
- 方法：`tl.dot(q, tl.trans(k))`，`tl.trans` 是寄存器内转置。
- 为什么是 Memory 优化：不产生额外访存（对照 host 侧真转置会多一次全量读写）。
- Roofline：不增 Bytes。