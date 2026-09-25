Based on the hardware's inherent capabilities and the measured current hardware metrics, we can determine where the current performance bottleneck lies. According to the Roofline principle, we classify bottlenecks into Compute, Memory, and Utilization. You need to determine the current bottleneck based on the machine's current performance and read the corresponding .md file to find the provided solutions that may be effective.

Signs that compute is the bottleneck: 
The actual FLOPs of the operator / theoretical peak FLOPs is low.
Execution unit (ALU/FPU/vector unit) utilization is high, but the cycle count is still large.
The instruction count is close to the theoretical lower bound, but the per-instruction throughput is low.

Signs that memory is the bottleneck:
High cache miss rate (L1/L2/LLC).
Memory bandwidth is nearly saturated, but compute utilization is low.
Load/Store stalls dominate.

Signs that Utilization is the bottleneck:
Low IPC, but it is neither a compute wall nor a bandwidth wall.
Many pipeline stalls: dependencies, branches, structural conflicts.
Execution units are sometimes idle and sometimes congested.
Uneven load across multiple cores.