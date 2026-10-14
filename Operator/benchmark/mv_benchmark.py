import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

# ========== 0. 定位 code/mv.py ==========
_current_dir = Path(__file__).resolve().parent          # benchmark/
_operator_root = _current_dir.parent                     # Operator/
_mv_source_path = _operator_root / "code" / "mv.py"      # Operator/code/mv.py

# ========== 1. 动态加载 mv 算子 ==========
def load_mv_from_source():
    """直接从 code/mv.py 加载算子实现，绕过库的注册机制。"""
    spec = importlib.util.spec_from_file_location("my_mv_impl", _mv_source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["my_mv_impl"] = module
    spec.loader.exec_module(module)
    return module

mv_module = load_mv_from_source()

# 根据你的 mv.py 实际暴露的函数名调整
if hasattr(mv_module, "mv"):
    my_mv = mv_module.mv
elif hasattr(mv_module, "mv_func"):
    my_mv = mv_module.mv_func
elif hasattr(mv_module, "gemv"):
    my_mv = mv_module.gemv
else:
    raise AttributeError(
        f"在 {_mv_source_path} 中找不到 mv / mv_func / gemv 函数。"
        f"模块内可用名称: {[n for n in dir(mv_module) if not n.startswith('_')]}"
    )

print(f"已从 {_mv_source_path} 加载算子: {my_mv.__name__ if hasattr(my_mv, '__name__') else my_mv}")

# 打印 PID 和 CPU 亲和性，方便 perf 挂载
print(f"当前进程 PID: {os.getpid()}")
try:
    print(f"CPU 亲和性: {os.sched_getaffinity(0)}")
except AttributeError:
    pass

# ========== 2. 构造典型用例 ==========
# 用例: matrix = [160, 1024], vector = [1024], dtype=float16, value_range=[-1, 1]
# 输出: [160], 内核按 BLOCK_N×BLOCK_M 分块, fp32 累加后转回 fp16
device = "cuda" if torch.cuda.is_available() else "cpu"
N, M = 160, 1024
dtype = torch.float16

mat = torch.empty((N, M), dtype=dtype, device=device).uniform_(-1, 1)
vec = torch.empty((M,), dtype=dtype, device=device).uniform_(-1, 1)

print(f"用例: matrix={tuple(mat.shape)}, vector={tuple(vec.shape)}, dtype={dtype}, value_range=[-1, 1]")
print(f"输出 shape: [{N}]")
print(f"matrix 元素数: {mat.numel()}, vector 元素数: {vec.numel()}")

# ========== 3. 预热 ==========
print("\n预热中...")
for _ in range(5):
    _ = my_mv(mat, vec)
if device == "cuda":
    torch.cuda.synchronize()
print("预热完成")

# ========== 4. 正确性验证 ==========
print("\n--- 正确性验证 ---")
# 参考实现：用 float32 在 CPU 上算，再转回 float16 比对
# mv 会做 M=1024 次乘加累加，容差比 abs 宽
ref_mat = mat.cpu().float()
ref_vec = vec.cpu().float()
ref_out = torch.mv(ref_mat, ref_vec)       # [160] float32

gems_out = my_mv(mat, vec).cpu().float()   # [160] float32

max_abs_err = (gems_out - ref_out).abs().max().item()
denom = ref_out.abs().clamp_min(1e-3)      # mv 结果可能接近 0，分母保护放宽
max_rel_err = ((gems_out - ref_out).abs() / denom).max().item()

print(f"max_abs_err = {max_abs_err:.3e}")
print(f"max_rel_err = {max_rel_err:.3e}")
# fp16 输入 + fp32 累加，容差比 abs 宽
assert max_abs_err < 1e-2, f"绝对误差超限: {max_abs_err}"
assert max_rel_err < 1e-2, f"相对误差超限: {max_rel_err}"
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
    out = my_mv(mat, vec)
    iterations += 1
    if iterations % 500 == 0:
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  已运行 {elapsed:.1f}s / {duration_sec}s，迭代 {iterations}")

if device == "cuda":
    torch.cuda.synchronize()
end = time.time()

avg_ms = (end - start) / iterations * 1000

# ========== 6. 性能指标 ==========
# mv 的访存量：读 matrix (N*M) + 读 vector (M) + 写 out (N)，单位是元素数
elem_size = mat.element_size()   # fp16 = 2 bytes
bytes_moved = (N * M + M + N) * elem_size
achieved_bw = bytes_moved / (avg_ms / 1000) / 1e9

# mv 的计算量：2 * N * M FLOPs（乘加各算一次）
flops = 2.0 * N * M
achieved_tflops = flops / (avg_ms / 1000) / 1e12

print(f"\n平均耗时: {avg_ms:.4f} ms")
print(f"数据搬运量: {bytes_moved / 1e3:.2f} KB (读 matrix + 读 vector + 写 out)")
print(f"实际带宽: {achieved_bw:.2f} GB/s")
print(f"计算量: {flops / 1e6:.4f} MFLOPs")
print(f"实际算力: {achieved_tflops:.4f} TFLOPS")
print(f"算术强度: {flops / bytes_moved:.2f} FLOPs/Byte")