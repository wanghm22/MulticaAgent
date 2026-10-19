status: `ok`

## Run identity

- iteration: `0`
- candidate_id / winner_candidate_id: `mv_baseline_iter0`（本轮唯一候选）
- benchmark_source: `Operator/benchmark/mv_benchmark.py` at commit `823ac99db60a32743ab40641245552582789aa42`
- candidate: `Operator/code/mv.py`, SHA-256 `65278ffe8287c898862cdf510b827cefa07f4d1f6cbe8f536804a7b089fd0ec9`
- benchmark SHA-256: `13634a57834c5221da4b1a16e52511de36314c285e1ecc82cab25220de7cdd37`
- input: matrix `[160,1024]`, vector `[1024]`, `float16`, value range `[-1,1]`; output `[160]`

## Environment

Remote RV host `k3`, Linux `6.18.3-generic`/riscv64; CPU affinity `0-7` (8 × Spacemit A100, max 2.0 GHz, boost disabled). Machine has 16 logical CPUs total, L1d/L1i 1 MiB each across 16 instances, L2 10 MiB across 4 instances, 15 GiB RAM/no swap. Required environment `/home/bianbu/.venv`: PyTorch `2.9.1+debian`, Triton `3.6.0`, FlagGems `0.0.0`; `torch.cuda.is_available()=False`. perf `6.18.3.g27275ec8240c`, paranoid `2`. Full record is in `setting.md`.

## Build and correctness

Dynamic import and Triton JIT/launch completed in every run; no separate build step was required. All five statistical samples passed the benchmark assertion. Per-run `(max_abs_err, max_rel_err)`: `(7.645e-3,4.708e-4)`, `(1.008e-2,4.548e-4)`, `(7.530e-3,4.660e-4)`, `(7.681e-3,4.357e-4)`, `(7.645e-3,4.762e-4)`. The dedicated perf run also passed: `(7.793e-3,4.607e-4)`.

## Reproduction commands

Each statistical sample used the repository benchmark unchanged (5 warmups, correctness check, 30 s attach window, 60 s timed loop):

```bash
ssh -p 3333 root@127.0.0.1
runuser -u bianbu -- bash -lc 'source /home/bianbu/.venv/bin/activate && cd /home/bianbu/mv-profile-pasi83 && exec taskset -c 0-7 python Operator/benchmark/mv_benchmark.py'
```

The valid counter sample located the real Python child rather than the `runuser` wrapper:

```bash
pid=$(pgrep -n -u bianbu -f 'python Operator/benchmark/mv_benchmark.py')
perf stat -p "$pid" -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,cache-misses,stalled-cycles-frontend,stalled-cycles-backend -- sleep 55
```

## Timing statistics

Main metric is average latency reported by the benchmark. Five independent 60 s samples were `1.1277, 1.1259, 1.1228, 1.1266, 1.1220 ms`.

| n | mean | median | min | P95 (linear) | sample stddev | CV |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1.1250 ms | 1.1259 ms | 1.1220 ms | 1.1275 ms | 0.002475 ms | 0.220% |

At mean latency: logical traffic `330,048 B/op`, derived bandwidth `0.293376 GB/s`; work `327,680 FLOP/op`, derived throughput `0.291271 GFLOP/s` (`0.000291271 TFLOP/s`); arithmetic intensity `0.9928 FLOP/B`. These are algorithmic/derived values, not DRAM-controller measurements.

## Hardware counters

One additional correctness-passing run was sampled for 55.006436623 s during its steady loop (Python PID `846517`; whole-process counts, not isolated kernel-only counts):

| counter | value | derived |
|---|---:|---:|
| cycles | 228,839,853,825 | 4.16024 G/s |
| instructions | 215,294,873,930 | IPC 0.94081; 3.91399 G/s |
| L1-dcache-loads | 38,895,160,564 | 707.102 M/s |
| L1-dcache-load-misses | 311,048,749 | 0.79971% of loads; 5.655 M/s |
| L1-dcache-stores | 15,898,292,867 | 289.026 M/s |
| stalled-cycles-frontend | 45,734,710,705 | 19.99% of cycles |
| stalled-cycles-backend | 130,471,827,305 | 57.01% of cycles |
| generic cache-misses | unavailable | perf reports `<not supported>` |
| DRAM bandwidth / execution-unit utilization | unavailable | no usable PMU event exposed in this run |

Five initially collected perf files are retained for audit but explicitly invalid: they attached to the `runuser` wrapper and report `<not counted>`. No value from them is used above.

## Selection and judge_input

- winner_candidate_id: `mv_baseline_iter0`; only valid candidate, so uplift versus baseline/runner-up is `N/A`.
- confidence: **high** for latency repeatability (CV 0.220%, all correctness checks passed); **medium** for bottleneck classification because counters cover the whole process and DRAM/EU counters are unavailable.
- judge_input: latency `1.1250 ms` mean / `1.1259 ms` median / `1.1275 ms` P95, n=5, fixed CPU 0-7; `0.291271 GFLOP/s`, `0.293376 GB/s` logical bandwidth, `0.9928 FLOP/B`; IPC `0.94081`, L1D load miss rate `0.79971%`, frontend stall `19.99%`, backend stall `57.01%` over a 55.006 s whole-process sample. The high backend-stall share is the strongest observed signal, but the available evidence is insufficient to label the kernel definitively memory- or compute-bound.

## Artifacts

- Exact candidate: repository commit `823ac99db60a32743ab40641245552582789aa42`, `Operator/code/mv.py` (hash above; source not modified).
- Raw logs: attached `pasi83_mv_baseline_iter0_logs.tar.gz`; within it, `run_1.log`…`run_5.log`, `run_6_perf.log`, `perf_6_valid.stat`, plus the five invalid wrapper-PID perf records retained for audit.
