import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

# ========== 0. 定位 code/topk.py ==========
_current_dir = Path(__file__).resolve().parent               # benchmark/
_operator_root = _current_dir.parent                          # Operator/
_topk_source_path = _operator_root / "code" / "topk.py"       # Operator/code/topk.py

# ========== 1. 动态加载 topk 算子 ==========
def load_topk_from_source():
    """直接从 code/topk.py 加载算子实现，绕过库的注册机制。"""
    spec = importlib.util.spec_from_file_location("my_topk_impl", _topk_source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["my_topk_impl"] = module
    spec.loader.exec_module(module)
    return module

topk_module = load_topk_from_source()

# 根据你的 topk.py 实际暴露的函数名调整
if hasattr(topk_module, "topk"):
    my_topk = topk_module.topk
elif hasattr(topk_module, "topk_func"):
    my_topk = topk_module.topk_func
else:
    raise AttributeError(
        f"在 {_topk_source_path} 中找不到 topk / topk_func 函数。"
        f"模块内可用名称: {[n for n in dir(topk_module) if not n.startswith('_')]}"
    )

print(f"已从 {_topk_source_path} 加载算子: {my_topk.__name__ if hasattr(my_topk, '__name__') else my_topk}")

print(f"当前进程 PID: {os.getpid()}")
try:
    print(f"CPU 亲和性: {os.sched_getaffinity(0)}")
except AttributeError:
    pass

# ========== 2. 构造典型用例 ==========
# 用例: input=[1048576] (1D, 1M 元素), dtype=float16, k=10, dim=-1, largest=True
# value_range=[-1, 1]
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

L = 1048576          # 1M 元素
k = 10
dim = -1
largest = True
sorted_flag = True

x = torch.empty((L,), dtype=dtype, device=device).uniform_(-1, 1)

print(f"用例: input={tuple(x.shape)}, dtype={dtype}, k={k}, dim={dim}, largest={largest}")
print(f"value_range=[-1, 1]")
print(f"输出 shape: values=[{k}], indices=[{k}]")

# ========== 3. 预热 ==========
print("\n预热中...")
for _ in range(5):
    _ = my_topk(x, k, dim, largest, sorted_flag)
if device == "cuda":
    torch.cuda.synchronize()
print("预热完成")

# ========== 4. 正确性验证 ==========
print("\n--- 正确性验证 ---")
# 参考实现：CPU 上 fp32 的 torch.topk
ref_x = x.cpu().float()
ref_values, ref_indices = torch.topk(ref_x, k, dim=dim, largest=largest, sorted=sorted_flag)

gems_values, gems_indices = my_topk(x, k, dim, largest, sorted_flag)
gems_values = gems_values.cpu().float()
gems_indices = gems_indices.cpu()

print(f"gems values shape: {tuple(gems_values.shape)}, indices shape: {tuple(gems_indices.shape)}")
print(f"ref  values shape: {tuple(ref_values.shape)}, indices shape: {tuple(ref_indices.shape)}")

# --- 4a. 值一致性 ---
max_abs_err = (gems_values - ref_values).abs().max().item()
denom = ref_values.abs().clamp_min(1e-3)
max_rel_err = ((gems_values - ref_values).abs() / denom).max().item()
print(f"values max_abs_err = {max_abs_err:.3e}")
print(f"values max_rel_err = {max_rel_err:.3e}")
assert max_abs_err < 1e-2, f"values 绝对误差超限: {max_abs_err}"

# --- 4b. 索引一致性 ---
# 注意：当存在重复值时，topk 的索引不唯一，不能直接比对 indices
# 用 gather 校验：x[indices] 应该等于 values
gathered = x.cpu().float().gather(0, gems_indices)
gather_err = (gathered - gems_values).abs().max().item()
print(f"gather 校验 max_err = {gather_err:.3e}")
assert gather_err < 1e-2, f"gather 校验失败: {gather_err}"

# --- 4c. 索引是否真的指向最大的 k 个（集合一致性）---
# 排序后比对值集合，避免重复值导致的索引顺序歧义
gems_sorted_vals = gems_values.sort(descending=largest).values
ref_sorted_vals = ref_values.sort(descending=largest).values
set_err = (gems_sorted_vals - ref_sorted_vals).abs().max().item()
print(f"排序后值集合 max_err = {set_err:.3e}")
assert set_err < 1e-2, f"值集合不一致: {set_err}"

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
    out_values, out_indices = my_topk(x, k, dim, largest, sorted_flag)
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
# topk 至少要读全部输入；输出很小
bytes_moved = L * elem_size
achieved_bw = bytes_moved / (avg_ms / 1000) / 1e9

print(f"\n平均耗时: {avg_ms:.4f} ms")
print(f"输入规模: {L} 元素 ({bytes_moved / 1e3:.2f} KB)")
print(f"实际带宽 (仅按读输入算): {achieved_bw:.2f} GB/s")
print(f"k={k}, largest={largest}, dim={dim}")