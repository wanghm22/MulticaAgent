## RV/K3 profiling host

- SSH endpoint: `root@127.0.0.1:3333`
- Architecture: `riscv64`
- Kernel: `Linux 6.18.3-generic`
- CPU topology: 16 logical CPUs; 8 Spacemit A100 cores and 8 Spacemit X100 cores, one thread per core
- CPU frequency range: 614.4–2000 MHz (A100), 614.4–2400 MHz (X100); boost disabled
- Cache: 64 KiB L1d and 64 KiB L1i per core (1 MiB each total); 10 MiB L2 total
- Triton environment: `/home/bianbu/.venv`; activate with `source /home/bianbu/.venv/bin/activate`
- Profiling tool: `perf 6.18.3.g27275ec8240c`
- Benchmark CPU affinity: CPUs `0-7` (A100 cluster)

Recorded on 2026-10-19 for reproducible operator profiling.
