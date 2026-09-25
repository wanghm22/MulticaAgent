The performance deficiency comes from memory, that is, data supply cannot keep up with computation, or memory access efficiency is low. The following methods may be effective: 
Solution 1: Use blocking and tiling to make the working set fit into L1/L2 cache; Solution 2: Rearrange data into contiguous access; 
Solution 3: Use prefetching via __builtin_prefetch, prefetch.r, etc.; 
Solution 4: Use huge pages and aligned allocation to reduce TLB misses; 
Solution 5: Use operator fusion to avoid memory access of intermediate tensors; Solution 6: For multi-core, pay attention to cache coherence and false sharing.