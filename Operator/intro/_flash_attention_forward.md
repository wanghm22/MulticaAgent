ATen 契约：

实现功能： FlashAttention 前向传播的核心入口函数，它负责在 GPU 上以分块（tiling）的方式高效计算注意力，避免将完整的注意力分数矩阵写入显存
注册：`("_flash_attention_forward", _flash_attention_forward)` 及 `_scaled_dot_product_flash_attention` 族|
分级：L4 / NeuralNetwork

备注：关键约束：

- 不支持 varlen：`assert cumulative_sequence_length_q is None`（传非 None 会断言失败）

- head dim 白名单：`(16, 32, 64, 96, 128, 192, 256)`；不在列表时填充计算后裁剪输出

- 返回五元组：`(out, lse, philox_seed, philox_offset, p)`，`lse` 供反向使用

- 额外支持 `softcap`、`alibi_slopes`、滑动窗口注意力（`window_size_left/right`）

**典型用例**（来源：FlagGems `tests/test_flash_attention.py`，`FLASH_ATTENTION_FORWARD_CONFIGS[1]`）

```yaml
operator: FlashAttentionForward
input_shape:
- [2, 64, 4, 128]    # query  [batch, seq_q,  num_head, head_dim]
- [2, 96, 4, 128]    # key    [batch, seq_kv, num_head, head_dim]
- [2, 96, 4, 128]    # value  [batch, seq_kv, num_head, head_dim]
dtype: [float16, float16, float16]
attrs:
  scale: 0.0884      # 1/sqrt(128)
  is_causal: false
value_range: [-0.05, 0.05]
note: "S-float16-非方-batch=2-head=4-seq_q=64-seq_kv=96-d=128"
```

非方形 Q/K（seq\_q≠seq\_kv）路径；head\_dim=128 在白名单内；返回五元组，`lse` 供反向使用。

