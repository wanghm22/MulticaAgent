# RV/K3 profiler environment

- SSH endpoint: `root@127.0.0.1:3333`
- OS/kernel: Linux `6.18.3-generic`, `riscv64`
- CPU: 16 logical CPUs; 8 × Spacemit A100 (up to 2.0 GHz) and 8 × Spacemit X100 (up to 2.4 GHz), 1 thread/core
- Cache: L1d 1 MiB total (16 instances), L1i 1 MiB total (16 instances), L2 10 MiB total (4 instances)
- Python environment: `/home/bianbu/.venv`; activate with `source /home/bianbu/.venv/bin/activate`
- Profiling tool: `/usr/bin/perf`
- Benchmark affinity: CPUs `0-7`

Captured on 2026-10-16 with `uname -a`, `lscpu`, and `perf list`.
