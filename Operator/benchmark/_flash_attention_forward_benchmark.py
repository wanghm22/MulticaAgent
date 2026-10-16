import os
import sys
import time
import inspect
from pathlib import Path

import torch
import torch.nn.functional as F

# ========== 0. 环境信息 ==========
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"当前进程 PID: {os.getpid()}")
try:
    print(f"CPU 亲和性: {os.sched_getaffinity(0)}")
except AttributeError:
    pass

# ========== 1. 被测算子：PyTorch 原生 SDPA（CPU/GPU 通用） ==========
def my_fa(q, k, v, scale, is_causal=False):
    """
    q, k, v: [B, S, H, D]  (BSHD)
    返回: out [B, S, H, D]
    """
    # SDPA 要求 [B, H, S, D]
    q_ = q.permute(0, 2, 1, 3)
    k_ = k.permute(0, 2, 1, 3)
    v_ = v.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q_, k_, v_, scale=scale, is_causal=is_causal)
    return out.permute(0, 2, 1, 3).contiguous()


print(f"my_fa signature: {inspect.signature(my_fa)}")


# ========== 2. 构造典型用例（dense BSHD） ==========
# q=[2, 64, 4, 128], k=[2, 96, 4, 128], v=[2, 96, 4, 128]
# dtype=float16, scale=1/sqrt(128), is_causal=False
# 非方形 Q/K (seq_q=64, seq_kv=96), head_dim=128
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

B, S_q, S_kv, H, D = 2, 64, 96, 4, 128
scale = 1.0 / (D ** 0.5)

# CPU 上 fp16 SDPA 支持有限，这里用 float32 保证能跑
if device == "cpu":
    dtype = torch.float32

q = torch.empty((B, S_q, H, D), dtype=dtype, device=device).uniform_(-0.05, 0.05)
k = torch.empty((B, S_kv, H, D), dtype=dtype, device=device).uniform_(-0.05, 0.05)
v = torch.empty((B, S_kv, H, D), dtype=dtype, device=device).uniform_(-0.05, 0.05)

print(f"device={device}, dtype={dtype}")
print(f"用例: q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)} (BSHD)")
print(f"scale={scale:.4f}, is_causal=False")
print(f"value_range=[-0.05, 0.05]")
print(f"输出 shape: out=[{B}, {S_q}, {H}, {D}]")
print(f"非方形 Q/K: seq_q={S_q} != seq_kv={S_kv}, head_dim={D}")


# ========== 3. 统一调用 wrapper ==========
def run_fa():
    return my_fa(q, k, v, scale=scale, is_causal=False)


# ========== 4. 预热 ==========
print("\n预热中...")
for _ in range(5):
    _ = run_fa()
if device == "cuda":
    torch.cuda.synchronize()
print("预热完成")


# ========== 5. 正确性验证 ==========
print("\n--- 正确性验证 ---")

# 参考实现：CPU fp32 BSHD
ref_q = q.cpu().float().permute(0, 2, 1, 3)   # [B, H, S_q, D]
ref_k = k.cpu().float().permute(0, 2, 1, 3)   # [B, H, S_kv, D]
ref_v = v.cpu().float().permute(0, 2, 1, 3)   # [B, H, S_kv, D]

attn_scores = torch.matmul(ref_q, ref_k.transpose(-2, -1)) * scale   # [B, H, S_q, S_kv]
attn_probs = torch.softmax(attn_scores, dim=-1)
ref_out = torch.matmul(attn_probs, ref_v)                              # [B, H, S_q, D]
ref_out = ref_out.permute(0, 2, 1, 3).contiguous()                     # [B, S_q, H, D]

# 调用被测算子
gems_out = run_fa()
gems_out = gems_out.cpu().float()

print(f"gems out shape: {tuple(gems_out.shape)}")
print(f"ref  out shape: {tuple(ref_out.shape)}")

# --- out 数值误差 ---
max_abs_err = (gems_out - ref_out).abs().max().item()
denom = ref_out.abs().clamp_min(1e-3)
max_rel_err = ((gems_out - ref_out).abs() / denom).max().item()
print(f"out max_abs_err = {max_abs_err:.3e}")
print(f"out max_rel_err = {max_rel_err:.3e}")
assert max_abs_err < 1e-2, f"out 绝对误差超限: {max_abs_err}"

# --- lse 一致性（参考侧） ---
# SDPA 不返回 lse，这里只算参考 lse 作为记录
ref_lse = torch.logsumexp(attn_scores, dim=-1)   # [B, H, S_q]
print(f"ref lse shape: {tuple(ref_lse.shape)} (SDPA 不返回 lse，跳过对比)")

print("正确性验证通过 ✓")


# ========== 6. 性能测试循环 ==========
print("\n--- 性能测试循环 ---")
print(">>> 现在可在另一个终端用 perf / miniperf 挂载本进程 <<<")
print("等待 30 秒以便挂载 perf...")
time.sleep(30)
print("开始计时，perf 应已挂载")

duration_sec = 60
iterations = 0
if device == "cuda":
    torch.cuda.synchronize()
start = time.time()

while time.time() - start < duration_sec:
    result = run_fa()
    iterations += 1
    if iterations % 50 == 0:
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  已运行 {elapsed:.1f}s / {duration_sec}s，迭代 {iterations}")

if device == "cuda":
    torch.cuda.synchronize()
end = time.time()

avg_ms = (end - start) / iterations * 1000


# ========== 7. 性能指标 ==========
elem_size = q.element_size()   # fp32 = 4 bytes, fp16 = 2 bytes
bytes_moved = (B * S_q * H * D + B * S_kv * H * D * 2 + B * S_q * H * D) * elem_size
achieved_bw = bytes_moved / (avg_ms / 1000) / 1e9

flops = 2.0 * B * H * S_q * S_kv * D * 2 + 4.0 * B * H * S_q * S_kv
achieved_tflops = flops / (avg_ms / 1000) / 1e12

print(f"\n输入规模: q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)} (BSHD)")
print(f"dtype={dtype}, device={device}")
print(f"平均耗时: {avg_ms:.4f} ms")
print(f"数据搬运量: {bytes_moved / 1e3:.2f} KB")
print(f"实际带宽: {achieved_bw:.2f} GB/s")
print(f"计算量: {flops / 1e6:.4f} MFLOPs")
print(f"实际算力: {achieved_tflops:.4f} TFLOPS")
print(f"算术强度: {flops / bytes_moved:.2f} FLOPs/Byte")
print(f"head_dim={D}, seq_q={S_q}, seq_kv={S_kv} (非方形)")