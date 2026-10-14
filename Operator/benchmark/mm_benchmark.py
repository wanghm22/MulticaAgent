import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

# ========== 0. 定位 code/mm.py ==========
_current_dir = Path(__file__).resolve().parent          # benchmark/
_operator_root = _current_dir.parent                     # Operator/
_mm_source_path = _operator_root / "code" / "mm.py"      # Operator/code/mm.py

# ========== 1. 动态加载 mm 算子 ==========
def load_mm_from_source():
    """直接从 code/mm.py 加载算子实现，绕过库的注册机制。"""
    spec = importlib.util.spec_from_file_location("my_mm_impl", _mm_source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["my_mm_impl"] = module
    spec.loader.exec_module(module)
    return module

mm_module = load_mm_from_source()

# 根据你的 mm.py 实际暴露的函数名调整
if hasattr(mm_module, "mm"):
    my_mm = mm_module.mm
elif hasattr(mm_module, "mm_func"):
    my_mm = mm_module.mm_func
elif hasattr(mm_module, "general_mm"):
    my_mm = mm_module.general_mm
else:
    raise AttributeError(
        f"在 {_mm_source_path} 中找不到 mm / mm_func / general_mm 函数。"
        f"模块内可用名称: {[n for n in dir(mm_module) if not n.startswith('_')]}"
    )

print(f"已从 {_mm_source_path} 加载算子: {my_mm.__name__ if hasattr(my_mm, '__name__') else my_mm}")

# 打印 PID 和 CPU 亲和性，方便 perf 挂载
print(f"当前进程 PID: {os.getpid()}")
try:
    print(f"CPU 亲和性: {os.sched_getaffinity(0)}")
except AttributeError:
    pass

# ========== 2. 构造典型用例 ==========
# 用例: a = [15, 160], b = [160, 1024], dtype=float16, value_range=[-1, 1]
# 输出: [15, 1024], 走 general_mm 路径, fp32 累加
device = "cuda" if torch.cuda.is_available() else "cpu"
M, K, N = 15, 160, 1024
dtype = torch.float16

a = torch.empty((M, K), dtype=dtype, device=device).uniform_(-1, 1)
b = torch.empty((K, N), dtype=dtype, device=device).uniform_(-1, 1)

print(f"用例: a={tuple(a.shape)}, b={tuple(b.shape)}, dtype={dtype}, value_range=[-1, 1]")
print(f"输出 shape: [{M}, {N}]")
print(f"a 元素数: {a.numel()}, b 元素数: {b.numel()}")

# ========== 3. 预热 ==========
print("\n预热中...")
for _ in range(5):
    _ = my_mm(a, b)
if device == "cuda":
    torch.cuda.synchronize()
print("预热完成")

# ========== 4. 正确性验证 ==========
print("\n--- 正确性验证 ---")
# 参考实现：用 float32 在 CPU 上算，再转回 float16 比对
# 注意：mm 的累积误差比 abs 大，容差要放宽
ref_a = a.cpu().float()
ref_b = b.cpu().float()
ref_out = torch.mm(ref_a, ref_b)           # [15, 1024] float32

gems_out = my_mm(a, b).cpu().float()       # [15, 1024] float32

max_abs_err = (gems_out - ref_out).abs().max().item()
denom = ref_out.abs().clamp_min(1e-3)      # mm 结果可能接近 0，分母保护放宽
max_rel_err = ((gems_out - ref_out).abs() / denom).max().item()

print(f"max_abs_err = {max_abs_err:.3e}")
print(f"max_rel_err = {max_rel_err:.3e}")
# mm 用 fp16 输入、fp32 累加，容差比 abs 宽
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
    out = my_mm(a, b)
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
# mm 的访存量：读 a (M*K) + 读 b (K*N) + 写 out (M*N)，单位是元素数
# 注意：这里按 fp16 的 2 字节算，但输出如果也是 fp16，写也是 2 字节
elem_size = a.element_size()   # fp16 = 2 bytes
bytes_moved = (M * K + K * N + M * N) * elem_size
achieved_bw = bytes_moved / (avg_ms / 1000) / 1e9

# mm 的计算量：2 * M * N * K FLOPs（乘加各算一次）
flops = 2.0 * M * N * K
achieved_tflops = flops / (avg_ms / 1000) / 1e12

print(f"\n平均耗时: {avg_ms:.4f} ms")
print(f"数据搬运量: {bytes_moved / 1e3:.2f} KB (读 a + 读 b + 写 out)")
print(f"实际带宽: {achieved_bw:.2f} GB/s")
print(f"计算量: {flops / 1e6:.4f} MFLOPs")
print(f"实际算力: {achieved_tflops:.4f} TFLOPS")
print(f"算术强度: {flops / bytes_moved:.2f} FLOPs/Byte")