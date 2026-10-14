## Profiler RV host

- Endpoint: `ssh -p 3333 root@127.0.0.1`
- Architecture/host: `riscv64`, hostname `k3`, Spacemit A100/X100, 16 logical CPUs
- Benchmark account: `bianbu` (entered from the required root SSH session)
- Required environment activation: `source /home/bianbu/.venv/bin/activate`
- Python: `/home/bianbu/.venv/bin/python`
- Runtime: torch `2.9.1+debian`, Triton `3.6.0+spacemit.a5`, FlagGems `0.0.0`, CUDA unavailable
- Kernel/perf: Linux `6.18.3-generic`, perf `6.18.3.g27275ec8240c`
- CPU frequency boost: disabled; reported max clocks are 2.0 GHz (A100) and 2.4 GHz (X100)

The login directory for `root` has no usable project environment. Always activate the
environment above before importing torch/Triton/FlagGems. Running the benchmark as
`bianbu` also supplies access to the CPU-backed Triton runtime shared-memory objects.
