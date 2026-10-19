# RV benchmark machine

- Endpoint: `ssh -p 3333 root@127.0.0.1`
- Benchmark account: `bianbu` (launched by root with `runuser`)
- Python environment: `/home/bianbu/.venv` (activate with `source /home/bianbu/.venv/bin/activate`)
- Host/kernel: `k3`, Linux `6.18.3-generic`, `riscv64`
- CPU: 16 logical CPUs, 1 thread/core; CPUs 0-7 Spacemit A100 (8 cores, max 2.0 GHz), CPUs 8-15 Spacemit X100 (8 cores, max 2.4 GHz); boost disabled
- Cache: L1d 1 MiB total (16 instances), L1i 1 MiB total (16 instances), L2 10 MiB total (4 instances)
- Memory: 15 GiB RAM, no swap
- Benchmark affinity used by profiler: CPUs `0-7`
- Software observed 2026-10-19: Python 3 (venv), PyTorch `2.9.1+debian`, Triton `3.6.0`, FlagGems `0.0.0`, `torch.cuda.is_available() == False`
- perf: `6.18.3.g27275ec8240c`, `/proc/sys/kernel/perf_event_paranoid = 2`; root attaches counters to the actual Python PID located with `pgrep`

This is a CPU-only K3/RV target. The distribution-provided Triton/FlagGems environment above is required; the system Python is not a valid benchmark environment.
