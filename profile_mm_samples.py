import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch

source = Path(__file__).parent / "Operator/code/mm.py"
spec = importlib.util.spec_from_file_location("profiled_mm", source)
module = importlib.util.module_from_spec(spec)
sys.modules["profiled_mm"] = module
spec.loader.exec_module(module)

torch.manual_seed(0)
M, K, N = 15, 160, 1024
a = torch.empty((M, K), dtype=torch.float16).uniform_(-1, 1)
b = torch.empty((K, N), dtype=torch.float16).uniform_(-1, 1)
for _ in range(5):
    module.mm(a, b)

ref = torch.mm(a.float(), b.float())
got = module.mm(a, b).float()
max_abs_err = (got - ref).abs().max().item()
max_rel_err = ((got - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()

samples_ms = []
iterations_per_sample = 2000
for _ in range(10):
    start = time.perf_counter()
    for _ in range(iterations_per_sample):
        module.mm(a, b)
    elapsed = time.perf_counter() - start
    samples_ms.append(elapsed * 1000 / iterations_per_sample)

ordered = sorted(samples_ms)
p95 = ordered[max(0, int(len(ordered) * 0.95 + 0.999999) - 1)]
mean = statistics.mean(samples_ms)
stdev = statistics.stdev(samples_ms)
bytes_moved = (M * K + K * N + M * N) * a.element_size()
flops = 2 * M * N * K
print(json.dumps({
    "correct": max_abs_err < 1e-2 and max_rel_err < 1e-2,
    "max_abs_err": max_abs_err,
    "max_rel_err": max_rel_err,
    "samples": len(samples_ms),
    "iterations_per_sample": iterations_per_sample,
    "samples_ms": samples_ms,
    "mean_ms": mean,
    "median_ms": statistics.median(samples_ms),
    "min_ms": min(samples_ms),
    "p95_ms": p95,
    "stdev_ms": stdev,
    "cv_percent": stdev / mean * 100,
    "bytes_per_iteration": bytes_moved,
    "flops_per_iteration": flops,
    "arithmetic_intensity_flops_per_byte": flops / bytes_moved,
    "achieved_gbps_from_mean": bytes_moved / (mean / 1000) / 1e9,
    "achieved_gflops_from_mean": flops / (mean / 1000) / 1e9,
}))
