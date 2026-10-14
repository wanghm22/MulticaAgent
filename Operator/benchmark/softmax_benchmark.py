import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

# ========== 0. 定位 code/softmax.py ==========
_current_dir = Path(__file__).resolve().parent                  # benchmark/
_operator_root = _current_dir.parent                             # Operator/
_softmax_source_path = _operator_root / "code" / "softmax.py"    # Operator/code/softmax.py

# ========== 1. 动态加载 softmax 算子 ==========
def load_softmax_from_source():
    """直接从 code/softmax.py 加载算子实现，绕过库的注册机制。"""
    spec = importlib.util.spec_from_file_location("my_softmax_impl", _softmax_source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["my_softmax_impl"] = module
    spec.loader.exec_module(module)
    return module

softmax_module = load_softmax_from_source()

# 根据你的 softmax.py 实际暴露的函数名调整
if hasattr(softmax_module, "softmax"):
    my_softmax = softmax_module.softmax
elif hasattr(softmax_module, "softmax_func"):
    my_softmax = softmax_module.softmax_func
else:
    raise AttributeError(
        f"在 {_softmax_source_path} 中找不到 softmax / softmax_func 函数。"
        f"模块内可用名称: {[n for n in dir(softmax_module) if not n.startswith('_')]}"
    )

print(f"已从 {_softmax_source_path} 加载算子: {my_softmax.__name__ if hasattr(my_softmax, '__name__') else my_softmax}")

print(f"当前进程 PID: {os.getpid()}")
try:
    print(f"CPU 亲和性: {os.sched_getaffinity(0)}")
except AttributeError:
    pass

# ========== 2. 构造典型用例 ==========
# 用例: input=[1024, 1024], dtype=float16, dim=-1
# value_range=[-1, 1], 沿最末维归约, 走 softmax_kernel_inner (K==1) 路径
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

R, C = 1024, 1024
dim = -1

x = torch.empty((R, C), dtype=dtype, device=device).uniform_(-1, 1)

print(f"用例: input={tuple(x.shape)}, dtype={dtype}, dim={dim}")
print(f"value_range=[-1, 1]")
print(f"输出 shape: [{R}, {C}]")

# ========== 3. 预热 ==========
print("\n预热中...")
for _ in range(5):
    _ = my_softmax(x, dim)
if device == "cuda":
    torch.cuda.synchronize()
print("预热完成")

# ========== 4. 正确性验证 ==========
print("\n--- 正确性验证 ---")
# 参考实现：CPU 上 fp32 的 torch.softmax
ref_x = x.cpu().float()
ref_out = torch.softmax(ref_x, dim=dim)

gems_out = my_softmax(x, dim).cpu().float()

print(f"gems output shape: {tuple(gems_out.shape)}")
print(f"ref  output shape: {tuple(ref_out.shape)}")

# --- 4a. 逐元素误差 ---
max_abs_err = (gems_out - ref_out).abs().max().item()
denom = ref_out.abs().clamp_min(1e-3)
max_rel_err = ((gems_out - ref_out).abs() / denom).max().item()
print(f"max_abs_err = {max_abs_err:.3e}")
print(f"max_rel_err = {max_rel_err:.3e}")
assert max_abs_err < 1e-2, f"绝对误差超限: {max_abs_err}"

# --- 4b. 概率归一化校验：每行之和应为 1 ---
row_sum = gems_out.sum(dim=dim)
sum_err = (row_sum - 1.0).abs().max().item()
print(f"行和与 1 的最大偏差 = {sum_err:.3e}")
assert sum_err < 1e-2, f"softmax 行和不为 1: {sum_err}"

# --- 4c. 非负性校验：输出应该全 >= 0 ---
assert (gems_out >= 0).all(), "存在负值输出"
print("非负性校验通过")

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
    out = my_softmax(x, dim)
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
# 访存量：读输入 + 写输出（softmax 输出和输入同 shape）
bytes_moved = (R * C + R * C) * elem_size
achieved_bw = bytes_moved / (avg_ms / 1000) / 1e9

# 计算量估算：每行做 1 次 max 归约 + 1 次 exp + 1 次 sum 归约 + 1 次除法
# 主要开销是 exp，按每元素 1 次 exp + 若干次比较/加法估算
flops = R * C * 4  # 粗略估算：max比较 + sub + exp + div
achieved_tflops = flops / (avg_ms / 1000) / 1e12

print(f"\n输入规模: {R} x {C} = {R*C} 元素 ({R*C*elem_size/1e3:.2f} KB)")
print(f"平均耗时: {avg_ms:.4f} ms")
print(f"数据搬运量: {bytes_moved / 1e3:.2f} KB (读 input + 写 output)")
print(f"实际带宽: {achieved_bw:.2f} GB/s")
print(f"算术强度 (粗估): {flops / bytes_moved:.2f} FLOPs/Byte")
print(f"归约路径: dim={dim} (最末维, 对应 softmax_kernel_inner K==1 路径)")