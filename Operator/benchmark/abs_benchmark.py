import importlib.util
import sys
import time
from pathlib import Path

import torch

# ========== 0. 根据你的实际目录结构调整这里的路径 ==========
# 假设你的结构是：
#   Operator/
#   ├── code/
#   │   └── abs.py        <-- 你要测的算子实现
#   └── benchmark/
#       └── your_bench.py <-- 本脚本放在这里

# 找到 code/abs.py 的绝对路径
_current_dir = Path(__file__).resolve().parent          # benchmark/
_operator_root = _current_dir.parent                     # Operator/
_abs_source_path = _operator_root / "code" / "abs.py"    # Operator/code/abs.py

# ========== 1. 从源码文件动态加载你的 abs 算子 ==========
def load_abs_from_source():
    """直接从 code/abs.py 加载你的算子实现，绕过库的注册机制。"""
    spec = importlib.util.spec_from_file_location("my_abs_impl", _abs_source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["my_abs_impl"] = module
    spec.loader.exec_module(module)
    return module

abs_module = load_abs_from_source()

# 根据你的 abs.py 实际暴露的函数名调整这里
# 常见命名：abs, abs_func, abs_kernel, my_abs ...
if hasattr(abs_module, "abs"):
    my_abs = abs_module.abs
elif hasattr(abs_module, "abs_func"):
    my_abs = abs_module.abs_func
else:
    raise AttributeError(
        f"在 {_abs_source_path} 中找不到 abs 或 abs_func 函数。"
        f"模块内可用名称: {[n for n in dir(abs_module) if not n.startswith('_')]}"
    )

print(f"已从 {_abs_source_path} 加载算子: {my_abs.__name__ if hasattr(my_abs, '__name__') else my_abs}")

# ========== 2. 构造典型用例 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
shape = (1024, 1024)
dtype = torch.float32

inp = torch.empty(shape, dtype=dtype, device=device).uniform_(-10, 10)
print(f"用例: shape={shape}, dtype={dtype}, value_range=[-10, 10]")

# ========== 3. 预热 ==========
print("\n预热中...")
for _ in range(5):
    _ = my_abs(inp)
if device == "cuda":
    torch.cuda.synchronize()
print("预热完成")

# ========== 4. 正确性验证 ==========
print("\n--- 正确性验证 ---")
ref_inp = inp.cpu()
ref_out = torch.abs(ref_inp)
gems_out = my_abs(inp).cpu()

max_abs_err = (gems_out - ref_out).abs().max().item()
denom = ref_out.abs().clamp_min(1e-8)
max_rel_err = ((gems_out - ref_out).abs() / denom).max().item()

print(f"max_abs_err = {max_abs_err:.3e}")
print(f"max_rel_err = {max_rel_err:.3e}")
assert max_abs_err < 1e-5, f"绝对误差超限: {max_abs_err}"
assert max_rel_err < 1e-4, f"相对误差超限: {max_rel_err}"
print("正确性验证通过 ✓")

# ========== 5. 性能测试循环（供 perf 挂载） ==========
print("\n--- 性能测试循环 ---")
print(">>> 现在可在另一个终端用 perf / miniperf 挂载本进程 <<<")


# 先 sleep 10 秒，给你时间切到另一个终端找 PID 并启动 perf
print("等待 30 秒以便挂载 perf...")
time.sleep(30)
print("开始计时，perf 应已挂载")

duration_sec = 60      # 实际计时窗口，按需要调整
iterations = 0
if device == "cuda":
    torch.cuda.synchronize()
start = time.time()

# 按时间跑，跑满 duration_sec 秒才停
while time.time() - start < duration_sec:
    out = my_abs(inp)
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
bytes_moved = inp.numel() * inp.element_size() * 2
achieved_bw = bytes_moved / (avg_ms / 1000) / 1e9

print(f"\n平均耗时: {avg_ms:.4f} ms")
print(f"实际带宽: {achieved_bw:.2f} GB/s")
print(f"数据搬运量: {bytes_moved / 1e6:.2f} MB (读+写)")