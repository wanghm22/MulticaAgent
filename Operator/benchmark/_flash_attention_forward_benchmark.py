import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

# ========== 0. 定位 code/flash_attention.py ==========
_current_dir = Path(__file__).resolve().parent                              # benchmark/
_operator_root = _current_dir.parent                                         # Operator/
_fa_source_path = _operator_root / "code" / "flash_attention.py"             # Operator/code/flash_attention.py

# ========== 1. 动态加载 flash_attention 算子 ==========
def load_fa_from_source():
    """直接从 code/flash_attention.py 加载算子实现，绕过库的注册机制。"""
    spec = importlib.util.spec_from_file_location("my_fa_impl", _fa_source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["my_fa_impl"] = module
    spec.loader.exec_module(module)
    return module

fa_module = load_fa_from_source()

# 根据你的 flash_attention.py 实际暴露的函数名调整
if hasattr(fa_module, "_flash_attention_forward"):
    my_fa = fa_module._flash_attention_forward
elif hasattr(fa_module, "flash_attention_forward"):
    my_fa = fa_module.flash_attention_forward
elif hasattr(fa_module, "flash_attention"):
    my_fa = fa_module.flash_attention
else:
    raise AttributeError(
        f"在 {_fa_source_path} 中找不到 _flash_attention_forward / flash_attention_forward 函数。"
        f"模块内可用名称: {[n for n in dir(fa_module) if not n.startswith('_')]}"
    )

print(f"已从 {_fa_source_path} 加载算子: {my_fa.__name__ if hasattr(my_fa, '__name__') else my_fa}")

print(f"当前进程 PID: {os.getpid()}")
try:
    print(f"CPU 亲和性: {os.sched_getaffinity(0)}")
except AttributeError:
    pass

# ========== 2. 构造典型用例 ==========
# 用例: q=[2, 64, 4, 128], k=[2, 96, 4, 128], v=[2, 96, 4, 128]
# dtype=float16, scale=1/sqrt(128), is_causal=False
# value_range=[-0.05, 0.05]
# 非方形 Q/K (seq_q=64, seq_kv=96), head_dim=128 在白名单内
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

B, S_q, S_kv, H, D = 2, 64, 96, 4, 128
scale = 1.0 / (D ** 0.5)   # ≈ 0.0884

q = torch.empty((B, S_q, H, D), dtype=dtype, device=device).uniform_(-0.05, 0.05)
k = torch.empty((B, S_kv, H, D), dtype=dtype, device=device).uniform_(-0.05, 0.05)
v = torch.empty((B, S_kv, H, D), dtype=dtype, device=device).uniform_(-0.05, 0.05)

print(f"用例: q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}")
print(f"dtype={dtype}, scale={scale:.4f}, is_causal=False")
print(f"value_range=[-0.05, 0.05]")
print(f"输出 shape: out=[{B}, {S_q}, {H}, {D}], lse=[{B}, {H}, {S_q}]")
print(f"非方形 Q/K: seq_q={S_q} != seq_kv={S_kv}, head_dim={D} (白名单内)")

# ========== 3. 预热 ==========
print("\n预热中...")
for _ in range(5):
    _ = my_fa(q, k, v, scale=scale, is_causal=False)
if device == "cuda":
    torch.cuda.synchronize()
print("预热完成")

# ========== 4. 正确性验证 ==========
print("\n--- 正确性验证 ---")
# 参考实现：CPU 上用 fp32 手动算标准 attention
# 注意：FlashAttention 的输出语义等价于 softmax(QK^T * scale) @ V
ref_q = q.cpu().float().permute(0, 2, 1, 3)   # [B, H, S_q, D]
ref_k = k.cpu().float().permute(0, 2, 1, 3)   # [B, H, S_kv, D]
ref_v = v.cpu().float().permute(0, 2, 1, 3)   # [B, H, S_kv, D]

# 标准 attention: softmax(Q @ K^T * scale) @ V
attn_scores = torch.matmul(ref_q, ref_k.transpose(-2, -1)) * scale   # [B, H, S_q, S_kv]
attn_probs = torch.softmax(attn_scores, dim=-1)
ref_out = torch.matmul(attn_probs, ref_v)                              # [B, H, S_q, D]
ref_out = ref_out.permute(0, 2, 1, 3).contiguous()                     # [B, S_q, H, D]

# 调用被测算子，接住五元组
gems_result = my_fa(q, k, v, scale=scale, is_causal=False)
gems_out, gems_lse = gems_result[0], gems_result[1]
gems_out = gems_out.cpu().float()

print(f"gems out shape: {tuple(gems_out.shape)}, lse shape: {tuple(gems_lse.shape)}")
print(f"ref  out shape: {tuple(ref_out.shape)}")

# --- 4a. 输出 out 的数值误差 ---
max_abs_err = (gems_out - ref_out).abs().max().item()
denom = ref_out.abs().clamp_min(1e-3)
max_rel_err = ((gems_out - ref_out).abs() / denom).max().item()
print(f"out max_abs_err = {max_abs_err:.3e}")
print(f"out max_rel_err = {max_rel_err:.3e}")
assert max_abs_err < 1e-2, f"out 绝对误差超限: {max_abs_err}"

# --- 4b. lse 一致性（可选，验证数值稳定性） ---
# lse = logsumexp(Q @ K^T * scale, dim=-1)，shape [B, H, S_q]
ref_lse = torch.logsumexp(attn_scores, dim=-1)   # [B, H, S_q]
gems_lse = gems_lse.cpu().float()

# lse 的 shape 可能是 [B, H, S_q] 或 [B, S_q, H]，按你的实现调整
if gems_lse.shape == ref_lse.shape:
    lse_err = (gems_lse - ref_lse).abs().max().item()
    print(f"lse max_abs_err = {lse_err:.3e}")
    assert lse_err < 1e-2, f"lse 误差超限: {lse_err}"
else:
    print(f"lse shape 不匹配: gems={tuple(gems_lse.shape)}, ref={tuple(ref_lse.shape)}，跳过 lse 校验")

print("正确性验证通过 ✓")

# ========== 5. 性能测试循环（供 perf 挂载） ==========
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
    result = my_fa(q, k, v, scale=scale, is_causal=False)
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

# ========== 6. 性能指标 ==========
elem_size = q.element_size()   # fp16 = 2 bytes
# 访存量：读 q + k + v + 写 out (lse 很小可忽略)
bytes_moved = (B * S_q * H * D + B * S_kv * H * D * 2 + B * S_q * H * D) * elem_size
achieved_bw = bytes_moved / (avg_ms / 1000) / 1e9

# 计算量：Q@K^T 是 2*B*H*S_q*S_kv*D，softmax 约 4*B*H*S_q*S_kv，P@V 是 2*B*H*S_q*S_kv*D
# 主导项是两次 matmul
flops = 2.0 * B * H * S_q * S_kv * D * 2 + 4.0 * B * H * S_q * S_kv
achieved_tflops = flops / (avg_ms / 1000) / 1e12

print(f"\n输入规模: q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}")
print(f"平均耗时: {avg_ms:.4f} ms")
print(f"数据搬运量: {bytes_moved / 1e3:.2f} KB (读 q + k + v + 写 out)")
print(f"实际带宽: {achieved_bw:.2f} GB/s")
print(f"计算量: {flops / 1e6:.4f} MFLOPs")
print(f"实际算力: {achieved_tflops:.4f} TFLOPS")
print(f"算术强度: {flops / bytes_moved:.2f} FLOPs/Byte")
print(f"head_dim={D} (白名单内), seq_q={S_q}, seq_kv={S_kv} (非方形)")