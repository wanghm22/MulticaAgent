# RV benchmark environment

- Captured: 2026-10-18 (Asia/Shanghai)
- Connection: `ssh -p 3333 root@127.0.0.1`
- Hostname: `k3`
- OS: Bianbu 4.0 (Resolute Raccoon), Linux 6.18.3-generic, riscv64
- CPU: 16 logical CPUs, 1 thread/core; 8 × Spacemit A100 (max 2.0 GHz) and 8 × Spacemit X100 (max 2.4 GHz)
- Cache: L1d 1 MiB total (16 instances), L1i 1 MiB total (16 instances), L2 10 MiB (4 instances)
- Benchmark affinity: CPUs 0-7 (`taskset -c 0-7`)
- Python environment: activate with `source /home/bianbu/.venv/bin/activate` (the task login directory `/root` has no `.venv`)
- Python: 3.14.4
- PyTorch: 2.9.1+debian
- Triton: 3.6.0 (RV/K3 CPU backend); `torch.cuda.is_available() == False`
- perf: 6.18.3.g27275ec8240c; `kernel.perf_event_paranoid=2`; profiling commands run as root

The machine has no GPU or conventional CUDA Triton driver. All operator benchmarks must run inside the virtual environment above and use the RV/K3 CPU backend.
