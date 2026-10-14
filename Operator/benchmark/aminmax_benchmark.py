import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

# ========== 0. 定位 code/aminmax.py ==========
_current_dir = Path(__file__).resolve().parent                  # benchmark/
_operator_root = _current_dir.parent                             # Operator/
_aminmax_source_path = _operator_root / "code" / "aminmax.py"    # Operator/code/aminmax.py

# ========== 1. 动态加载 aminmax 算子 ==========
def load_aminmax_from_source():
    """直接从 code/aminmax.py 加载算子实现，绕过库的注册机制。"""
    spec = importlib.util.spec_from_file_location("my_aminmax_impl", _aminmax_source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["my_aminmax_impl"] = module
    spec.loader.exec_module(module)
    return module

aminmax_module = load_aminmax_from_source()

# 根据你的 aminmax.py 实际暴露的函数名调整
if hasattr(aminmax_module, "aminmax"):
    my_aminmax = aminmax_module.aminmax
elif hasattr(aminmax_module, "aminmax_func"):
    my_aminmax = aminmax_module.aminmax_func
else:
    raise AttributeError(
        f"在 {_aminmax_source_path} 中找不到 aminmax / aminmax_func 函数。"
        f"模块内可用名称: {[n for n in dir(aminmax_module) if not n.startswith('_')]}"
    )

print(f"已从 {_aminmax_source_path} 加载算子: {my_aminmax.__name__ if hasattr(my_aminmax, '__name__') else my_aminmax}")

print(f"当前进程 PID: {os.getpid()}")
try:
    print(f"CPU 亲和性: {os.sched_getaffinity(0)}")
except AttributeError:
    pass

# ========== 2. 构造典型用例 ==========
# 用例: input=[4096, 256], dtype=float16, dim=1, keepdim=False
# value_range=[-1, 1], 输出 shape=[4096]
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

R, C = 4096, 256
dim = 1
keepdim = False

x = torch.empty((R, C), dtype=dtype, device=device).uniform_(-1, 1)

print(f"用例: input={tuple(x.shape)}, dtype={dtype}, dim={dim}, keepdim={keepdim}")
print(f"value_range=[-1, 1]")
print(f"输出 shape: min=[{R}], max=[{R}]")

# ========== 3. 预热 ==========
print("\n预热中...")
for _ in range(5):
    _ = my_aminmax(x, dim=dim, keepdim=keepdim)
if device == "cuda":
    torch.cuda.synchronize()
print("预热完成")

# ========== 4. 正确性验证 ==========
print("\n--- 正确性验证 ---")
# 参考实现：CPU 上 fp32 的 torch.aminmax
ref_x = x.cpu().float()
ref_min, ref_max = torch.aminmax(ref_x, dim=dim, keepdim=keepdim)

gems_min, gems_max = my_aminmax(x, dim=dim, keepdim=keepdim)
gems_min = gems_min.cpu().float()
gems_max = gems_max.cpu().float()

print(f"gems min shape: {tuple(gems_min.shape)}, max shape: {tuple(gems_max.shape)}")
print(f"ref  min shape: {tuple(ref_min.shape)}, max shape: {tuple(ref_max.shape)}")

# --- min 一致性 ---
min_abs_err = (gems_min - ref_min).abs().max().item()
denom_min = ref_min.abs().clamp_min(1e-3)
min_rel_err = ((gems_min - ref_min).abs() / denom_min).max().item()
print(f"min max_abs_err = {min_abs_err:.3e}")
print(f"min max_rel_err = {min_rel_err:.3e}")
assert min_abs_err < 1e-2, f"min 绝对误差超限: {min_abs_err}"

# --- max 一致性 ---
max_abs_err = (gems_max - ref_max).abs().max().item()
denom_max = ref_max.abs().clamp_min(1e-3)
max_rel_err = ((gems_max - ref_max).abs() / denom_max).max().item()
print(f"max max_abs_err = {max_abs_err:.3e}")
print(f"max max_rel_err = {max_rel_err:.3e}")
assert max_abs_err < 1e-2, f"max 绝对误差超限: {max_abs_err}"

# --- min <= max 的结构性校验 ---
assert (gems_min <= gems_max).all(), "存在 min > max 的元素"
print("min <= max 结构校验通过")

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
    out_min, out_max = my_aminmax(x, dim=dim, keepdim=keepdim)
    iterations += 1
    if iterations % 200 == 0:
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  已运行 {elapsed:.1f}s / {duration_sec}s，迭代 {iterations}")

if device == "cuda":
    torch.cuda.synchronize()
end = time.time()

avg_ms = (end - start) / iterations * 1000

# ========== 6. 性能指标 ==========
elem_size = x.element_size()   # fp16 = 2 bytes
# 访存量：读全部输入 + 写 min 和 max 两个输出
bytes_moved = (R * C + 2 * R) * elem_size
achieved_bw = bytes_moved / (avg_ms / 1000) / 1e9

# 计算量：每个元素参与 min 和 max 各一次比较
flops = 2.0 * R * C
achieved_tflops = flops / (avg_ms / 1000) / 1e12

print(f"\n输入规模: {R} x {C} = {R*C} 元素 ({R*C*elem_size/1e3:.2f} KB)")
print(f"平均耗时: {avg_ms:.4f} ms")
print(f"数据搬运量: {bytes_moved / 1e3:.2f} KB (读 input + 写 min + 写 max)")
print(f"实际带宽: {achieved_bw:.2f} GB/s")
print(f"比较操作数: {flops / 1e6:.4f} M")
print(f"算术强度: {flops / bytes_moved:.2f} FLOPs/Byte")