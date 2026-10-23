# Cat / Concat — Utilization 维度优化

Roofline 视角：cat 纯搬运，utilization 的关键是 program 铺满、分支最小化、grid 安全。本章把 cat 在 utilization 维度的优化前后代码完整摆出来。

## 优化前：单 program 串行搬运全部输出

```python
@triton.jit
def cat_serial_kernel(A, B, C, M_A, M_B, N):
    for r in range(0, M_A + M_B):
        for c in range(0, N):
            if r < M_A:
                v = tl.load(A + r * N + c)
            else:
                v = tl.load(B + (r - M_A) * N + c)
            tl.store(C + r * N + c, v)
```

grid=(1,)，单 program 串行搬运全部输出，只有一条 lane 在工作，其余执行单元全空转。utilization 极低。

## 优化一：grid 按输出切块，program 铺满

```python
cat0_kernel[(M_A + M_B, triton.cdiv(N, BLOCK_N))](
    a, b, c, M_A, M_B, N, BLOCK_N=BLOCK_N)
```

grid=(M_A+M_B, cdiv(N, BLOCK_N))，每个 program 一个输出行×列块，行数和列块数相乘得到 program 总数，铺满执行单元。每 program 处理 BLOCK_N 个连续元素，lane 一次铺满。utilization 从单 program 串行拉满到行×列块并行。

## 优化二：无分支 mask-select，所有 lane 走同一指令流

优化前 if/elif 按行选路：

```python
if r < M_A:
    v = tl.load(A + r * N + c)
else:
    v = tl.load(B + (r - M_A) * N + c)
```

优化后两路都 load、mask 选路：

```python
va = tl.load(A + row * N + offs_n, mask=n_mask & (row < M_A), other=0.)
vb = tl.load(B + (row - M_A) * N + offs_n, mask=n_mask & (row >= M_A), other=0.)
tl.store(C + row * N + offs_n, va + vb, mask=n_mask)
```

utilization 角度，无分支写法让所有 lane 走同一指令流，不因 row < M_A 产生控制流分叉；SIMT/SIMD 硬件上分支会导致部分 lane 空转。能用 mask/select 表达的选择就不要用 if/elif，对编译器更友好、对硬件更友好。

## 优化三：host 循环 vs 单 kernel，dispatch 开销取舍

```python
# host 循环：K 次 launch，K 小最稳
# 单 kernel + mask-select：1 次 launch，2~3 个输入
# 单 kernel + 指针 if/elif：1 次 launch，K 大但编译器支持不全
```

utilization 上三种方案的账：host 循环 K 次 launch，K 小时 dispatch 开销可接受、最稳；单 kernel mask-select 1 次 launch，2~3 个输入时最优；单 kernel 指针 if/elif 1 次 launch，K 大时省 launch，但历史编译器对"控制流携带指针类型"支持不完整，会 failed to legalize 'cf.br'。K3 dispatch 开销不可忽略，K 大时单 kernel 有真实吸引力，但通用直觉是能用 mask/select 就不用 if/elif。

## 优化四：grid 安全性，不超运行时上限

```python
# grid=(M_A+M_B, cdiv(N, BLOCK_N))，行数 128+64=192、列块若干，总量安全
# 行多列窄时可退化成 grid=(M_A+M_B,) 每行内循环列块
```

utilization 上 grid 总量要离运行时上限远。旧版 wheel 对超大 grid 有上限、超限静默丢弃；新版运行时自动分块。行多列窄时退化成 grid=(M_A+M_B,) 每行内循环列块，program 数更少但每个吃更多数据。

## 优化五：多维推广，映射规则不变、lane 连续

```python
# dim=0 的 3D/4D cat：行号解码成多维下标（r → (b, m) 等）、偏移按 stride 算
```

utilization 上多维推广只要把行号解码成多维下标、偏移按 stride 算，映射规则不变。非拼接维不连续时 host 先 contiguous 或走 stride 版偏移，保持 lane 连续访存。

## 优化前后 utilization 对照

| 形态 | 并行度 | 分支 | grid 安全 |
|---|---|---|---|
| 单 program 串行 | grid=(1,) | if/elif | 安全但空转 |
| host 循环 copy | grid=(cdiv(numel,BLOCK),)×K | 无 | 安全、K 次 launch |
| 单 kernel mask-select | grid=(M_A+M_B, cdiv(N,BLOCK_N)) | 无 | 安全、1 次 launch |
| 单 kernel 指针 if/elif | 同左 | 有 | 编译器支持不全 |

## 验证 utilization 优化生效

对比单 program 串行版与 mask-select 版墙钟；对比 host 循环与单 kernel 在 K=2/10 时的 dispatch 开销；确认 mask-select 版所有 lane 走同一指令流、无控制流分叉。