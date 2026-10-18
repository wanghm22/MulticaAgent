# Attention — Utilization 方向优化
使用Triton代码
> 优化前：单 tile kernel（8.4），`tl.dot` + `tl.max/tl.sum` 归约，无显式微块打包、无 grid-stride、无 causal 拆 STAGE。
> 优化后：生产 flash 骨架（8.5）+ TCM 预算意识（8.6）+ proton 两级计时（8.7）。

---

## 优化前代码（无 Utilization 优化）

```python
qk = tl.dot(q, tl.trans(k)) * sm_scale
causal = offs_m[:, None] >= offs_n[None, :]
qk = tl.where(causal, qk, float('-inf'))
m_i = tl.max(qk, axis=1)
p = tl.exp(qk - m_i[:, None])
l_i = tl.sum(p, axis=1)
acc = tl.dot(p.to(tl.float16), v) / l_i[:, None]
```

---

## 优化后代码（smt.dot + acc_4d + grid-stride + causal 拆 STAGE）

```python
# smt.dot 替代 tl.dot（矩阵引擎直通）
qk = smt.dot(q, trans_k) * sm_scale
acc += smt.dot(p_cast, v)

# accumulator 是 MICRO tile 打包布局（acc_4d），
# 形状按 MICRO_M/K/N=16/8/32 组织，最后 smt.view 解包成普通二维再除 l_i
acc_4d = ...
acc_4d = smt.view(acc_4d, (0, 0), (BLOCK_M, BLOCK_D), (1, 1))

# grid-stride + task 解码：program 总数固定（NUM_CTAS），彻底避开 grid 上限
task_hz_idx = pid // NUM_BLOCKS_M
task_m_idx = pid % NUM_BLOCKS_M
# 再用 pid + num_ctas * block_idx 的 grid-stride 循环遍历所有任务

# causal 拆 STAGE：全 causal 块跳过 mask，对角混合块才逐元素判
# 全 causal 块直接跳过 mask；对角块才逐元素判 offs_m >= offs_n

# D=128 向量通路必须 2×64-chunk（超 f16 单向量寄存器宽 64）
# 矩阵引擎通路按 MICRO tile（K 维 8）自动分块

# TCM 预算：qk/p/acc 三个 BLOCK_M×BLOCK_N 级 tile 同时活着，
# f32 acc 再翻倍。BLOCK_M=BLOCK_N=128、D=128 时单 acc 就 64KB——
# 配置 BLOCK 前先心算总预算，384KB 超了就 rc=134。
```

---

## 优化点拆解（Utilization 方向）

**① smt.dot 替代 tl.dot（矩阵引擎直通）**
- 方法：用 `smt.dot(q, trans_k)` 和 `smt.dot(p_cast, v)` 替代 `tl.dot`。
- 为什么是 Utilization 优化：K3 矩阵引擎直通路径，accumulator 是 MICRO tile 打包布局（acc_4d），以峰值吞吐工作。
- Roofline：utilization。

**② accumulator 按 MICRO tile 打包（acc_4d）**
- 方法：累加器形状按 MICRO_M/K/N=16/8/32 组织成 `acc_4d`，最后 `smt.view` 解包成普通二维再除 l_i。
- 为什么是 Utilization 优化：硬件微块布局对齐，矩阵引擎才能以峰值吞吐工作。
- Roofline：utilization。

**③ grid-stride + task 解码**
- 方法：`task_hz_idx = pid // NUM_BLOCKS_M; task_m_idx = pid % NUM_BLOCKS_M`，再用 `pid + num_ctas * block_idx` 的 grid-stride 循环遍历所有任务，program 总数固定（NUM_CTAS）。
- 为什么是 Utilization 优化：彻底避开 grid 上限问题，调度不浪费。
- Roofline：utilization 的调度层优化。

**④ causal 拆 STAGE**
- 方法：全 causal 块直接跳过 mask，对角混合块才逐元素判 `offs_m >= offs_n`。
- 为什么是 Utilization 优化：纯性能优化，语义与单 tile 版一致，减少非计算的分支/判断开销。
- Roofline：utilization。

**⑤ D=128 向量通路 2×64-chunk**
- 方法：D=128 超过 f16 单向量寄存器宽（64），向量通路必须 2×64-chunk；矩阵引擎通路则按 MICRO tile（K 维 8）自动分块。
- 为什么是 Utilization 优化：对齐硬件寄存器宽度，避免走低效路径。
- Roofline：utilization。

**⑥ TCM 预算意识**
- 方法：qk/p/acc 三个 BLOCK_M×BLOCK_N 级 tile 同时活着，f32 acc 再翻倍。BLOCK_M=BLOCK_N=128、D=128 时单 acc 就 64KB——配置 BLOCK 前先心算总预算，384KB 超了就 rc=134。
- 为什么是 Utilization 优化：知道硬件天花板在哪，避免超预算导致编译器崩溃。
- Roofline：utilization 的边界意识。