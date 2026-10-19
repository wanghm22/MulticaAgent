## RV benchmark host

- SSH endpoint: `root@127.0.0.1:3333`
- Hostname: `k3`
- Architecture: `riscv64`
- CPU: 8× Spacemit A100 (up to 2.0 GHz) + 8× Spacemit X100 (up to 2.4 GHz), 16 logical CPUs total
- Cache: L1d 1 MiB total (16 instances), L1i 1 MiB total (16 instances), L2 10 MiB (4 instances)
- OS/kernel: Linux 6.18.3-generic
- Triton environment: `/home/bianbu/.venv` (activate with `source /home/bianbu/.venv/bin/activate`)
- Performance tooling: `/usr/bin/perf`; passwordless `sudo` is enabled for this root SSH account
- GPU: none; the virtual environment supplies the K3 CPU-targeted Triton/FlagGems/PyTorch stack
