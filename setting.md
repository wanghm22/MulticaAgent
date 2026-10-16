# RV benchmark machine

- SSH endpoint: `root@127.0.0.1:3333`
- Hostname: `k3`
- OS/kernel: Linux `6.18.3-generic`, `riscv64`
- CPU: 16 logical CPUs, 1 thread/core; 8 × Spacemit A100 (max 2.0 GHz) and 8 × Spacemit X100 (max 2.4 GHz)
- Cache: 16 × 64 KiB L1d (1 MiB total), 16 × 64 KiB L1i (1 MiB total), 4 × 2.5 MiB L2 (10 MiB total)
- Memory: 15 GiB RAM, no swap
- Benchmark CPU affinity: CPUs `0-7`
- Python environment: activate with `cd /home/bianbu && source .venv/bin/activate`
- Python: 3.14.4
- PyTorch: 2.9.1+debian
- Triton: 3.6.0 (RV/K3 environment)
- FlagGems: 0.0.0
- perf: 6.18.3.g27275ec8240c; `perf_event_paranoid=2`; PMUs: `cpu`, `rvtrace`, software/tracepoint/breakpoint
- Accelerator note: no CUDA GPU; the K3/RV Triton environment executes through the CPU-backed device stack.

Observed on 2026-10-16 (Asia/Shanghai).
