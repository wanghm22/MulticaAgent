PASI-265：七算子完成三轮优化（RV/K3 CPU）。

范围为 abs、aminmax、conv2d、mm、mv、softmax、topk；排除的 `_flash_attention_forward.py` 与初始提交一致。三个 Orchestrator 每轮各交付七个独立候选，共63个；Profiler 逐算子筛选正确且有可信收益的版本，保留基线时记录继承来源。

以下是最初源码与最终源码的新一轮同场交替测量，单位 ms/call。每份实现100个预热后归一化样本，10个随机交错块，计入派发、分配及全部运行时工作；JIT/构建/预热不计入。加速为原始均值/最终均值，未相乘跨轮比率。

|算子|原始 ms|最终 ms|加速|配对95%区间|最终代码来源|
|---|---:|---:|---:|---|---|
|abs|4.245012|1.306774|3.248×|[3.199, 3.299]|`pasi265_i2_o0_abs_contigown`|
|aminmax|4.858407|2.341626|2.075×|[2.071, 2.079]|`pasi265_i3_o2_aminmax_fold128`|
|conv2d|25.298554|9.700301|2.608×|[2.586, 2.630]|`pasi265_i3_o1_conv2d_bk64`|
|mm|1.096002|0.239668|4.573×|[4.543, 4.604]|`pasi265_i3_o2_mm_bk32`|
|mv|0.565932|0.271834|2.082×|[2.058, 2.106]|`pasi265_i2_o2_mv_unroll2`|
|softmax|1.234717|0.798422|1.546×|[1.531, 1.562]|`pasi265_i3_o1_softmax_rows32`|
|topk|13.562756|4.254786|3.188×|[3.173, 3.202]|`pasi265_i2_o1_topk_cpuheap10`|

原 benchmark 的断言、参考实现和容差保持不变；seed 265/266/267、代表性回退、topk 索引/唯一性/gather/排序及特殊值检查均通过。最终七份源码的字节SHA与Profiler实测快照一致，完整SHA见 `final_manifest.json`；三轮选择来源见 `selection_history.json`。

环境为 RV/K3 RISC-V AI CPU，Python 3.14.4、Torch 2.9.1+debian、Triton 3.6.0/Spacemit、FlagGems 5.0.2、CPUDriver。入口CPU0–7，Triton/spert工作线程使用CPU8–15。六项保留Triton实现，topk使用自包含Python文件内嵌的C++17小堆实现；首次调用会构建扩展。快路径受输入守卫约束，并保留通用回退；这些数据只代表指定benchmark和稳态条件。

PMU仅报告可用的cycles/instructions/L1及通用stall。DRAM实测带宽、细分执行单元利用率和匹配硬件屋顶不可用，不据此声称唯一瓶颈。单进程配对区间为测量启发式，不代表跨机器或跨进程的普遍保证。

完整Profiler报告、原始样本/命令、全部候选、知识卡与可移植SQLite通过Multica PASI-265附件交付。原始350点样本保留；正式结果使用iteration_evaluations/iteration_summary等扩展表，旧evaluations表仅含初始基线。复现与查询工具位于证据包各轮profiler和memory_metrics目录。
