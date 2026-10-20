# RV benchmark host

- SSH endpoint: `root@127.0.0.1:3333`
- Hostname: `k3`
- Architecture: `riscv64`, Spacemit K3/A100+X100, 16 logical CPUs
- Kernel: `6.18.3-generic`
- CPU affinity used by operator benchmarks: CPUs `0-7`
- Python environment: `/home/bianbu/.venv` (activate with `source /home/bianbu/.venv/bin/activate`)
- Performance tool: `perf 6.18.3.g27275ec8240c`
- Benchmark user: `bianbu` when accessing `/dev/shm/tcm_sync_standalone` (owned by `bianbu:bianbu`, mode `0600`)

Captured on 2026-10-20 (Asia/Shanghai).
