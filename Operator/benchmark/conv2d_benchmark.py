import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

# ========== 0. 定位 code/conv2d.py ==========
_current_dir = Path(__file__).resolve().parent               # benchmark/
_operator_root = _current_dir.parent                          # Operator/
_conv_source_path = _operator_root / "code" / "conv2d.py"     # Operator/code/conv2d.py

# ========== 1. 动态加载 conv2d 算子 ==========
def load_conv2d_from_source():
    """直接从 code/conv2d.py 加载算子实现，绕过库的注册机制。"""
    spec = importlib.util.spec_from_file_location("my_conv2d_impl", _conv_source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["my_conv2d_impl"] = module
    spec.loader.exec_module(module)
    return module

conv_module = load_conv2d_from_source()

# 根据你的 conv2d.py 实际暴露的函数名调整
if hasattr(conv_module, "conv2d"):
    my_conv2d = conv_module.conv2d
elif hasattr(conv_module, "conv2d_func"):
    my_conv2d = conv_module.conv2d_func
else:
    raise AttributeError(
        f"在 {_conv_source_path} 中找不到 conv2d / conv2d_func 函数。"
        f"模块内可用名称: {[n for n in dir(conv_module) if not n.startswith('_')]}"
    )

print(f"已从 {_conv_source_path} 加载算子: {my_conv2d.__name__ if hasattr(my_conv2d, '__name__') else my_conv2d}")

print(f"当前进程 PID: {os.getpid()}")
try:
    print(f"CPU 亲和性: {os.sched_getaffinity(0)}")
except AttributeError:
    pass

# ========== 2. 构造典型用例 ==========
# 用例: x=[2,64,32,32], filter=[64,64,3,3], bias=[64]
# dtype=float16, strides=[1,1], pads=[1,1,1,1], dilations=[1,1]
# value_range: x [-1,1], filter [-1,1], bias [-0.1,0.1]
# same padding, 输出 [2,64,32,32]
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

N, C_in, H, W = 2, 64, 32, 32
C_out, Kh, Kw = 64, 3, 3

x = torch.empty((N, C_in, H, W), dtype=dtype, device=device).uniform_(-1, 1)
weight = torch.empty((C_out, C_in, Kh, Kw), dtype=dtype, device=device).uniform_(-1, 1)
bias = torch.empty((C_out,), dtype=dtype, device=device).uniform_(-0.1, 0.1)

strides = (1, 1)
padding = (1, 1, 1, 1)     # same padding for 3x3, stride 1
dilations = (1, 1)
groups = 1

print(f"用例: x={tuple(x.shape)}, weight={tuple(weight.shape)}, bias={tuple(bias.shape)}")
print(f"dtype={dtype}, strides={strides}, pads={padding}, dilations={dilations}, groups={groups}")
print(f"输出 shape: [{N}, {C_out}, {H}, {W}]")

# ========== 3. 预热 ==========
print("\n预热中...")
for _ in range(5):
    _ = my_conv2d(x, weight, bias, strides, padding, dilations, groups)
if device == "cuda":
    torch.cuda.synchronize()
print("预热完成")

# ========== 4. 正确性验证 ==========
print("\n--- 正确性验证 ---")
# 参考实现：全 fp32 在 CPU 上算，再转回 fp16 比对
# conv2d 的累加维度是 C_in*Kh*Kw=576，容差比 abs 宽
ref_x = x.cpu().float()
ref_w = weight.cpu().float()
ref_b = bias.cpu().float()
ref_out = torch.conv2d(ref_x, ref_w, ref_b, strides, padding, dilations, groups)

gems_out = my_conv2d(x, weight, bias, strides, padding, dilations, groups).cpu().float()

max_abs_err = (gems_out - ref_out).abs().max().item()
denom = ref_out.abs().clamp_min(1e-3)
max_rel_err = ((gems_out - ref_out).abs() / denom).max().item()

print(f"输出 shape: {tuple(gems_out.shape)}")
print(f"max_abs_err = {max_abs_err:.3e}")
print(f"max_rel_err = {max_rel_err:.3e}")
# fp16 输入 + fp32 累加，conv2d 的容差比 abs 宽
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
    out = my_conv2d(x, weight, bias, strides, padding, dilations, groups)
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
# 输出尺寸
H_out = (H + padding[0] + padding[2] - dilations[0] * (Kh - 1) - 1) // strides[0] + 1
W_out = (W + padding[1] + padding[3] - dilations[1] * (Kw - 1) - 1) // strides[1] + 1

elem_size = x.element_size()   # fp16 = 2 bytes
# 访存量：读 x + 读 weight + 读 bias + 写 out
bytes_moved = (N * C_in * H * W + C_out * C_in * Kh * Kw + C_out + N * C_out * H_out * W_out) * elem_size
achieved_bw = bytes_moved / (avg_ms / 1000) / 1e9

# 计算量：2 * N * C_out * H_out * W_out * C_in * Kh * Kw FLOPs
flops = 2.0 * N * C_out * H_out * W_out * C_in * Kh * Kw
achieved_tflops = flops / (avg_ms / 1000) / 1e12

print(f"\n输出 shape: [{N}, {C_out}, {H_out}, {W_out}]")
print(f"平均耗时: {avg_ms:.4f} ms")
print(f"数据搬运量: {bytes_moved / 1e3:.2f} KB (读 x + 读 weight + 读 bias + 写 out)")
print(f"实际带宽: {achieved_bw:.2f} GB/s")
print(f"计算量: {flops / 1e6:.4f} MFLOPs")
print(f"实际算力: {achieved_tflops:.4f} TFLOPS")
print(f"算术强度: {flops / bytes_moved:.2f} FLOPs/Byte")