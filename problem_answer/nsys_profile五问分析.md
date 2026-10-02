# Nsight Systems Profiling 五问分析

整理日期：2026-09-30  
对应代码：`cs336_systems/benchmarking213.py`、`cs336_systems/benchmarking214.py`、`cs336-basics/cs336_basics/transformer.py`、`cs336-basics/cs336_basics/optimization.py`  
对应 profile：`benchmark_profile.1.nsys-rep`、`benchmark_profile.2.nsys-rep`、`benchmark_profile.3.nsys-rep`

## 0. 结论适用范围

本文所有数值只对应当前已经实际采集的一组配置：

| 配置项 | 当前值 | 来源 |
|---|---:|---|
| 模型规模 | medium | `benchmarking214.py:13` |
| `d_model` | 1024 | `benchmarking214.py:21-25` |
| `d_ff` | 4096 | `benchmarking214.py:21-25` |
| 层数 | 24 | `benchmarking214.py:21-25` |
| attention heads | 16 | `benchmarking214.py:21-25` |
| context length | 512 | `benchmarking214.py:59` |
| batch size | 4 | `benchmarking214.py:79` |
| dtype | FP32 | `benchmarking214.py:55` |
| 预热次数 | 5 | `benchmarking214.py:114,144-146` |
| 正式测量次数 | 10 | `benchmarking214.py:115,147-150` |

题目完整要求是选择两种模型规模，并对三个大于 128 的 2 的幂次 context length 进行 profile。当前结果只完成了 `medium × 512` 这一格，所以下面的回答可作为这一格的完整分析和其他组合的分析模板，不能当作整个实验矩阵已经完成。

## 1. 数据来源和三个 profile 的对应关系

`benchmarking214.py:120` 按以下顺序运行三个模式，并用同名 `benchmark` NVTX range 分别捕获：

| 文件 | 模式 | 包含的工作 |
|---|---|---|
| `benchmark_profile.1.nsys-rep` | `forward` | forward |
| `benchmark_profile.2.nsys-rep` | `forward_backward` | forward、loss、backward |
| `benchmark_profile.3.nsys-rep` | `full_step` | forward、loss、backward、AdamW optimizer |

代码中的 NVTX 标签位于 `benchmarking214.py:96-110`，外层正式测量标签位于 `benchmarking214.py:147-150`。预热在外层 `benchmark` 标签之前，因此没有进入这三个 profile。

Python 标准库 `perf_counter` 的对照结果来自之前终端输出：

| 模式 | Python 测量结果 |
|---|---:|
| forward | `52.948 ± 0.103 ms` |
| forward + backward | `169.038 ± 0.646 ms` |
| full step | `184.027 ± 1.169 ms` |

这些 Python 时间的计时代码位于 `benchmarking213.py:145-160`。每次 `run_step` 后都执行 `torch.cuda.synchronize()`，因此计时包含等待 GPU 完成的时间。

### 1.1 四种 nsys report 各自表示什么

- `nvtx_pushpop_sum`：CPU 线程进入和退出 NVTX range 之间的持续时间。CUDA 是异步的，所以内层 `forward` range 的 CPU 时间不一定等于 GPU 完成 forward 的时间。
- `nvtx_gpu_proj_sum`：把某个 NVTX range 启动的 GPU 操作投影到 GPU 时间线，更适合读取该阶段对应的 GPU 时间。
- `cuda_gpu_kern_sum`：按 CUDA kernel 名称汇总真正的 GPU 执行时间、调用次数和比例，用于找 GPU 热点。
- `cuda_api_sum`：按 CUDA API 名称汇总 CPU 侧的提交、同步和内存操作时间；例如 `cudaDeviceSynchronize` 的时间主要是 CPU 等待 GPU，并不是一个模型计算 kernel。

## 2. 问题 (a)：forward pass 总时间是多少，是否与 Python 计时相符？

### 可提交的 1–2 句答案

在 medium、context length 512、batch size 4、FP32 配置下，Nsight Systems 的 GPU 投影显示单次 forward 平均为 **53.015 ms**；Python `perf_counter` 的结果为 **52.948 ± 0.103 ms**。两者仅相差 **0.067 ms（约 0.13%）**，可以认为一致。

### 详细推导

`benchmark_profile.1.nsys-rep` 的 `nvtx_gpu_proj_sum` 给出：

```text
Range      Range Instances  Proj Avg (ns)
:forward   10               53014618.8
```

换算为毫秒：

```text
53014618.8 ns / 1,000,000 = 53.0146188 ms
```

Python 结果为 `52.948 ms`，所以：

```text
绝对差 = 53.0146188 - 52.948 = 0.0666188 ms
相对差 = 0.0666188 / 52.948 ≈ 0.126%
```

还可以用外层 `benchmark` range 交叉验证。`nvtx_pushpop_sum` 中外层 range 的总时间是 `531372005 ns`，包含 10 次 forward：

```text
531372005 ns / 10 = 53.1372005 ms/次
```

它与 Python 结果相差约 `0.189 ms（0.36%）`，同样非常接近。这里不使用 `nvtx_pushpop_sum` 中内层 `:forward` 的 `35.377 ms` 作为最终 forward 延迟，因为该值主要反映 CPU 异步提交 CUDA 工作的时间，GPU 在 range 退出后仍可能继续执行。

### 答案来源

1. **Python 对照时间**：之前的终端输出 `forward: 52.948 ± 0.103 ms`；对应计时代码为 `benchmarking213.py:145-160`。
2. **Nsight forward GPU 时间**：`benchmark_profile.1.nsys-rep` 的 `nvtx_gpu_proj_sum`，`:forward` 行的 `Proj Avg (ns)`。
3. **Nsight 外层交叉验证**：同一 report 的 `nvtx_pushpop_sum`，`:benchmark` 行的 `Total Time (ns)`，再除以 10。

## 3. 问题 (b)：哪个 CUDA kernel 累计时间最多？单次 forward 调用多少次？forward+backward 时是否相同？

### 可提交的 1–2 句答案

仅做 forward 时，累计 GPU 时间最多的是 `cutlass_80_simt_sgemm_128x256_8x4_tn_align1`，10 次 forward 共执行 730 次、耗时 230.156 ms，因此单次 forward 调用 **73 次**。加入 backward 后，第一名变为 `cutlass_80_simt_sgemm_256x128_8x4_nn_align1`（10 次共 1690 次、276.032 ms），所以不是同一个 kernel。

### 详细数据

`benchmark_profile.1.nsys-rep` 的 `cuda_gpu_kern_sum` 第一行：

| 模式 | 第一名 kernel | Total Time | Time | 10 次中的 Instances | 每次调用数 |
|---|---|---:|---:|---:|---:|
| forward | `cutlass_80_simt_sgemm_128x256_8x4_tn_align1` | 230.156 ms | 46.2% | 730 | 73 |
| forward + backward | `cutlass_80_simt_sgemm_256x128_8x4_nn_align1` | 276.032 ms | 约 17.4% | 1690 | 169 |

单次 forward 的调用次数来自：

```text
730 instances / 10 forward passes = 73 instances/forward
```

73 也能与代码结构互相印证：24 个 Transformer block 中每个 FFN 有 `W1`、`W2`、`W3` 三个线性层，共 `24 × 3 = 72` 个，再加最终 `lm_head`，正好是 73 个。这个对应关系来自 `transformer.py:70-76`、`transformer.py:194-197`；它是根据 kernel 调用数和模型结构做出的映射。

forward+backward 的第一名变成 `...256x128...nn...`，说明 backward 引入了不同转置方向和矩阵形状的 GEMM。它的 `169` 次/训练样本也与所有线性层数量一致：每个 block 有 attention 的 4 个线性层和 FFN 的 3 个线性层，`24 × 7 + 1 lm_head = 169`。

### 答案来源

1. **forward 第一名、累计时间、调用次数**：`benchmark_profile.1.nsys-rep` 的 `cuda_gpu_kern_sum` 第一行。
2. **forward+backward 第一名、累计时间、调用次数**：`benchmark_profile.2.nsys-rep` 的 `cuda_gpu_kern_sum` 第一行。
3. **单次调用数**：两个 report 都包含 10 次测量，分别用 `730/10` 和 `1690/10` 计算。
4. **与模型结构的对应**：`transformer.py:70-76,134-137,194-197`。

## 4. 问题 (c)：forward 中还有哪些非矩阵乘法 kernel 占用了不可忽略的时间？

### 可提交的 1–2 句答案

除 GEMM 外，forward 中最显著的是 `exp_kernel_cuda`（32.636 ms，约 6.5%），以及除法、mask `where`、加减、乘法等 elementwise kernels（单类约 2.4%–4.1%）。softmax 的 max/sum reduction、RMSNorm/SiLU/RoPE/残差连接产生的逐元素与归约 kernel 也贡献了可见开销。

### 详细数据和代码对应

下面均来自 `benchmark_profile.1.nsys-rep` 的 `cuda_gpu_kern_sum`，时间是 10 次 forward 的累计值：

| 非 GEMM kernel | Total Time | GPU kernel 时间占比 | 主要代码来源 |
|---|---:|---:|---|
| `exp_kernel_cuda` | 32.636 ms | 6.5% | attention softmax 的两次 `exp`，以及 FFN 的 SiLU `exp` |
| elementwise division | 20.498 ms | 4.1% | softmax 除法、RMSNorm/SiLU 等除法 |
| `where_kernel_impl` | 12.934 ms | 2.6% | causal attention mask |
| elementwise add/subtract | 12.436 ms | 2.5% | softmax 平移、残差/广播运算 |
| elementwise multiply | 12.157 ms | 2.4% | scale、RoPE、RMSNorm、SiLU 等 |
| `MaxOps` reduction | 4.147 ms | 0.8% | softmax 求最大值 |
| sum reduction | 3.177 ms | 0.6% | softmax 分母求和 |

代码中的直接证据包括：

- `transformer.py:77-81`：自定义 softmax 包含 max、减法、两次 `torch.exp`、sum 和除法。
- `transformer.py:117-120`：attention 包含缩放、mask、softmax。
- `transformer.py:55-59`：RMSNorm 包含 square、mean、sqrt、乘除。
- `transformer.py:74-76`：SiLU/FFN 包含 neg、exp、加法、除法和乘法。
- `transformer.py:104-109`：RoPE 包含多次逐元素乘加及 stack/flatten。
- `transformer.py:155`：每层创建 causal mask 并调用 `tril`。

### 答案来源

1. **各 kernel 的时间、比例和调用次数**：`benchmark_profile.1.nsys-rep` 的 `cuda_gpu_kern_sum`。
2. **kernel 对应的高层操作**：`transformer.py:55-81,104-120,155-158`。
3. **映射方法**：结合 kernel 名称（如 `exp`、`where`、`DivFunctor`、`MaxOps`、`sum_functor`）与源码中的对应运算，而不是只凭时间猜测。

## 5. 问题 (d)：完整训练步中矩阵乘法的时间比例如何变化？其他 kernel 呢？

### 可提交的 1–2 句答案

forward-only 时 GEMM 占全部 GPU kernel 时间的 **70.85%**，完整训练步中下降到 **56.98%**，减少约 **13.87 个百分点**。原因不是矩阵乘法消失，而是 backward 和逐参数 AdamW 更新加入了大量 add、mul、div、sqrt、copy 等 elementwise kernels，使非 GEMM 比例从 29.15% 上升到 43.02%。

### 详细计算

这里将名称包含 `cutlass::Kernel` 或 `magma_sgemm` 的 kernel 归为矩阵乘法。分类结果来自 `.sqlite` 中 `CUPTI_ACTIVITY_KIND_KERNEL` 的实际 GPU 起止时间：

| 模式 | 全部 GPU kernel 时间 | GEMM 时间 | GEMM 比例 | 非 GEMM 比例 |
|---|---:|---:|---:|---:|
| forward-only | 498.432 ms | 353.155 ms | 70.85% | 29.15% |
| full step | 1701.448 ms | 969.471 ms | 56.98% | 43.02% |

计算方式：

```text
forward GEMM fraction = 353.155 / 498.432 = 70.85%
full-step GEMM fraction = 969.471 / 1701.448 = 56.98%
变化 = 56.98% - 70.85% = -13.87 percentage points
```

完整训练步中耗时最多的非 GEMM 类别包括：

| full-step 非 GEMM kernel | Total Time | 占全部 GPU kernel 时间 |
|---|---:|---:|
| vectorized add | 110.828 ms | 6.51% |
| vectorized multiply | 104.093 ms | 6.12% |
| elementwise divide | 94.012 ms | 5.53% |
| direct copy | 55.782 ms | 3.28% |
| 另一类 vectorized divide | 52.261 ms | 3.07% |
| unary multiply | 48.842 ms | 2.87% |
| negation | 47.937 ms | 2.82% |

这与 AdamW 源码一致：`optimization.py:60-63` 对每个参数分别执行动量更新、二阶矩更新、weight decay、sqrt、除法和参数更新。当前实现没有使用 fused AdamW，因此会产生大量逐参数 elementwise kernel；这也是 full step 中非 GEMM 比例明显上升的重要原因。

### 答案来源

1. **forward-only 全部 kernel 和 GEMM 累计时间**：`benchmark_profile.1.sqlite` 的 `CUPTI_ACTIVITY_KIND_KERNEL`。
2. **full-step 全部 kernel、GEMM 和其他 kernel 时间**：`benchmark_profile.3.sqlite` 的同一表。
3. **矩阵乘法分类规则**：kernel 名称包含 `cutlass::Kernel` 或 `magma_sgemm`；当前 report 中实际矩阵乘法均落入这两类。
4. **非 GEMM 增长的代码原因**：`optimization.py:47-64` 的逐参数、自定义、非 fused AdamW。

## 6. 问题 (e)：attention 中 softmax 与矩阵乘法的运行时间和 FLOPs 如何比较？

### 可提交的 1–2 句答案

在每次 forward 中，attention 的两次 batched matrix multiplication 合计约 **3.169 ms**，而自定义 softmax 的 max、减法、两次 exp、sum 和除法合计约 **6.732 ms**，softmax 反而约慢 **2.12 倍**。但两次 attention matmul 约有 **103.08 GFLOPs/forward**，按当前 softmax 实现约 2.42 billion 标量操作估算，前者约多 **42.7 倍**；这说明高算术强度、优化良好的 GEMM 比由多个内存访问、归约和 transcendental kernel 组成的 softmax 更能有效利用 GPU。

### 6.1 此处比较的操作范围

这里的“attention matrix multiplications”专指 `scaled_dot_product_attention` 中 softmax 前后的两次矩阵乘法：

```text
Q @ K^T  -> attention scores
softmax(scores)
P @ V    -> attention output
```

对应源码为 `transformer.py:117-120`，不包含 `W_Q/W_K/W_V/W_O` 四个线性投影。

### 6.2 运行时间如何得到

当前代码只给整个 `forward` 加了 NVTX 标签，没有分别给 `QK^T`、softmax 和 `PV` 加细粒度标签。因此下面的对应关系不是直接读取操作名，而是综合 **kernel 名称、grid 形状、调用次数和源码结构** 得到的。

attention 的两个 batched GEMM 在 10 次 forward 中分别是：

| kernel | 10 次累计时间 | Instances | 单次 forward 调用数 |
|---|---:|---:|---:|
| `magma_sgemmEx_kernel` | 16.770 ms | 240 | 24 |
| `cutlass_80_simt_sgemm_64x64_8x5_nn_align1` | 14.915 ms | 240 | 24 |

每个 kernel 每次 forward 恰好调用 24 次，与模型的 24 层一一对应。因此两次 attention matmul 的每次 forward 时间为：

```text
(16.770 + 14.915) ms / 10 = 3.1685 ms
```

自定义 softmax 的源码 `transformer.py:77-81` 会执行 max、减法、exp、sum、再次 exp 和除法。根据 attention score 张量对应的 grid 形状和每层调用次数，10 次 forward 的累计时间为：

| softmax 组成 | 10 次累计时间 | Instances |
|---|---:|---:|
| max reduction | 4.147 ms | 240 |
| 减去最大值（broadcast add/subtract kernel） | 12.436 ms | 240 |
| 两次 `exp` | 29.508 ms | 480 |
| sum reduction | 3.177 ms | 240 |
| 除以分母 | 18.048 ms | 240 |
| **合计** | **67.316 ms** | — |

所以每次 forward：

```text
softmax time = 67.316 / 10 = 6.7316 ms
attention matmul time = 3.1685 ms
runtime ratio = 6.7316 / 3.1685 ≈ 2.12
```

这里没有把 `weight / sqrt(d_k)` 的 scale kernel 和 causal-mask `where` kernel 算进 softmax 本身；如果将它们也算作 attention-score normalization 周边开销，非 GEMM 时间还会更大。

### 6.3 FLOPs/操作量如何得到

当前配置：

```text
B = 4
H = 16
S = 512
d_head = d_model / H = 1024 / 16 = 64
L = 24
```

若一次乘加记为 2 FLOPs，则一层中 `QK^T` 和 `PV` 两次矩阵乘法合计为：

```text
2 matmuls × 2 × B × H × S² × d_head
= 4 × 4 × 16 × 512² × 64
= 4,294,967,296 FLOPs/layer
```

24 层合计：

```text
4,294,967,296 × 24 = 103,079,215,104 FLOPs
≈ 103.08 GFLOPs/forward
```

当前 softmax 对每个 score 元素近似进行 max、减法、两次 exp、sum 和除法，共约 6 个标量操作。按这种简化计数：

```text
6 × B × H × S² × L
= 6 × 4 × 16 × 512² × 24
= 2,415,919,104 operations/forward
```

因此矩阵乘法的理论 FLOPs/标量操作数约为 softmax 的：

```text
103,079,215,104 / 2,415,919,104 ≈ 42.7 倍
```

这个 FLOP 比值只用于数量级比较：`exp`、比较和 reduction 的硬件代价不能严格当作普通的一次浮点加法。关键现象是，虽然 attention matmul 的算术工作量高一个数量级以上，但其测得运行时间反而更短；原因是 GEMM 具有更高算术强度并受到 CUTLASS/MAGMA 高度优化，而当前 softmax 被拆成多个独立、偏内存带宽和归约受限的 kernel，并且源码还重复计算了 `torch.exp(shifted_x)`。

### 答案来源

1. **attention 计算结构**：`transformer.py:110-120`。
2. **head dimension、层数、batch 和 context length**：`benchmarking214.py:21-25,59,79`。
3. **两个 attention GEMM 的时间和次数**：`benchmark_profile.1.sqlite` 的 kernel 名称、grid 和 instance 统计。
4. **softmax 各 kernel 的时间和次数**：同一 `.sqlite` 中与 score 张量形状对应的 exp、reduction、broadcast add/divide kernels。
5. **FLOPs**：根据源码中的两个 batched matmul 及实际张量形状手工计算；softmax 使用当前实现的六步近似操作计数。

## 7. 五问结果汇总

| 问题 | 当前 `medium × 512` 的结论 | 最主要的数据来源 |
|---|---|---|
| (a) forward 总时间 | 53.015 ms；与 Python 52.948 ms 相符 | `.1` 的 `nvtx_gpu_proj_sum` + Python stdout |
| (b) 最耗时 kernel | forward 为 `...128x256...tn...`，73 次/forward；forward+backward 第一名不同 | `.1/.2` 的 `cuda_gpu_kern_sum` |
| (c) 非 GEMM 热点 | exp 6.5%，以及 div/where/add/mul/reduction kernels | `.1` 的 `cuda_gpu_kern_sum` + Transformer 源码 |
| (d) GEMM 比例变化 | 70.85% → 56.98%，下降 13.87 个百分点 | `.1/.3` 的 kernel 事件统计 |
| (e) attention softmax vs matmul | 6.732 ms vs 3.169 ms；softmax 慢 2.12×，尽管 matmul 操作量约大 42.7× | `.1` 的 kernel 事件 + attention 源码 + FLOPs 计算 |

## 8. 复现统计的命令

查看四类标准 report：

```bash
nsys stats \
  --report nvtx_pushpop_sum,nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_api_sum \
  benchmark_profile.1.nsys-rep
```

将文件名替换为 `.2.nsys-rep` 或 `.3.nsys-rep`，即可分别查看 forward+backward 和 full-step。

需要注意：`nvtx_gpu_proj_sum` 中 backward 的投影可能不完整，因为 PyTorch autograd 的部分 kernel 由其他工作线程启动，不一定继承主线程的 `backward` NVTX range。因此本文关于 forward/backward 总体 kernel 排名与比例的结论使用完整 capture 的 `cuda_gpu_kern_sum`/SQLite kernel 事件，而没有用异常偏小的 `:backward Proj Avg`。

## 9. 完成整道作业还需要补充什么

题面要求两种模型规模、三个大于 128 的 2 的幂次 context length，并让最长长度取显存能够容纳的最大值。当前还需要为其余配置重复生成三种模式的 profile，然后对每个配置复用本文的五问分析；尤其应比较 context length 增大后 attention softmax、两个 `S²` attention GEMM、其他线性层 GEMM 的比例如何变化。
