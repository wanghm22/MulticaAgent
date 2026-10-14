## RV profiling host

- Connection: `ssh -p 3333 root@127.0.0.1`
- Hostname: `k3`
- Architecture: `riscv64`
- SoC/CPU: 8 × Spacemit A100 (max 2.0 GHz) + 8 × Spacemit X100 (max 2.4 GHz), 16 logical CPUs total
- Cache: L1d 1 MiB (16 instances), L1i 1 MiB (16 instances), L2 10 MiB (4 instances)
- Memory: 15 GiB, no swap
- Kernel: Linux 6.18.3-generic
- Python environment: `/home/bianbu/.venv` (activate with `cd /home/bianbu && source .venv/bin/activate`)
- Python/packages observed 2026-10-14: Python 3.14.4, torch 2.9.1+debian, Triton 3.6.0, FlagGems 0.0.0
- perf: 6.18.3.g27275ec8240c; `kernel.perf_event_paranoid=2`
- Accelerator note: `torch.cuda.is_available() == False`; this is the CPU-backed K3/RV Triton environment.
