# Section 4：GPU Kernels 学习路线与任务清单

整理日期：2026-10-02  
对应材料：[本项目作业说明](../cs336_assignment2_systems.pdf) 第 15–29 页、[测试适配器](../tests/adapters.py)、[Attention 测试](../tests/test_attention.py)。  
范围：说明 Section 4 要学什么、各题要做什么、如何衔接及验收；以下不是已经完成的实现或实验报告。

## 1. 这一节的主线

Section 2 已经训练了你测时间、测显存并解释瓶颈；Section 3 讨论了 autograd 保存的张量及重计算。Section 4 把这些观察集中到 attention：普通实现会产生形状约为 `[batch, query_length, key_length]` 的注意力矩阵，序列越长，内存和数据搬运越昂贵。本节先用基准测试确认问题，再学习 Triton 的 tile 和指针，最后自己实现 FlashAttention-2 的前向、反向并验证性能。

```text
普通 PyTorch attention：先知道慢在哪里、占内存在哪里
        ↓
torch.compile 对照：自动融合能改善多少
        ↓
Triton weighted-sum 示例：学会 tile、指针和 autograd 接口
        ↓
FlashAttention-2 前向：在线 softmax，不写出完整注意力矩阵
        ↓
FlashAttention-2 反向：用保存的 L 重算概率并求梯度
        ↓
正确性测试与基准测试：比较时间、显存限制和适用条件
```

必做题合计 **29 分**：`pytorch_attention` 2 分、`torch_compile` 2 分、`flash_forward` 15 分、`flash_backward` 5 分、`flash_benchmarking` 5 分。作业还给出一个 **可选** 的 Triton backward 练习；weighted-sum 是教学示例，不是单独计分题。

## 2. 按题目看要做什么

| 阶段 | 题目 | 要交付什么 | 做完后应能回答 |
|---|---|---|---|
| 4.1.1 | `pytorch_attention` | 参数扫描的 forward/backward 时间及 OOM 表；一个 OOM 配置的显存估算；1–2 段分析 | 为什么长序列 attention 会耗尽显存？ |
| 4.2 | `torch_compile` | 编译前后的 attention 时间表，以及完整 Transformer 的性能对照表 | 自动编译能解决哪些开销，哪些问题仍存在？ |
| 4.2.1 | weighted-sum 示例 | 阅读并理解示例，无单独交付物 | Triton program 如何加载 tile、处理边界和接入 autograd？ |
| 4.2.2 | `flash_forward` | PyTorch 分块参考实现、Triton 前向 kernel、causal mask，接入适配器并通过测试 | 怎样不存整张注意力矩阵仍算出 `O` 与 `L`？ |
| 4.2.2 后半 | `flash_backward` | 用 PyTorch 和 `torch.compile` 实现 backward，通过梯度测试 | 怎样用 `Q,K,V,O,L,dO` 重算 `P` 并得到 `dQ,dK,dV`？ |
| 4.2.2 后半 | `flash_benchmarking` | `triton.testing.do_bench` 测得的 forward、backward、完整往返时间表 | 自己的实现在哪些形状、精度下有收益或限制？ |
| 4.2.3 | OPTIONAL Triton backward | 可选的分块 Triton backward | 如何进一步减少 backward 的数据搬运与同步？ |

### 2.1 `pytorch_attention`：建立普通实现的基线

题目要求 batch size 固定为 **8**，不使用多头维度；扫描 `d ∈ {16, 32, 64, 128}` 与序列长度 `S ∈ {256, 1024, 4096, 8192, 16384}` 的笛卡尔积，共 **20 组**。每组随机生成 `Q,K,V`，预热后分别测 **100 次 forward** 和 **100 次 backward**，每次 forward/backward 后都调用 `torch.cuda.synchronize()`；还要记录 backward 开始前占用的显存。无法运行的组合记录 `OOM`，不要把 OOM 悄悄从表中删掉。

随后选一个**能找到的最小 OOM 配置**做显存估算。例如在无 head 维度、`Q` 和 `K` 序列等长时，一张 attention score/probability 矩阵有 `8 × S × S` 个元素；若每元素 4 字节，仅一张矩阵就需要 `8S² × 4` 字节。实际峰值还包括 `Q,K,V`、softmax 中间值、输出及 backward 所需张量，所以这只是分析的一项，不等于完整峰值。最后解释 saved tensors 如何随 `S` 增长，以及你打算如何避免保存完整的 `S × S` 矩阵。

这里可以复用 Section 2 的计时脚手架；做 100 次 backward 时，要确保每次都有有效计算图和梯度，不能对已经释放的同一计算图反复调用 `backward()`。

### 2.2 `torch_compile`：给自动优化一个公平对照

第一部分在上题**完全相同的 20 组形状与测量条件**下，比较普通 PyTorch attention 和 `torch.compile` 后的版本，交付 forward/backward 时间表。第二部分把 Section 2 的完整 Transformer 端到端基准也增加 compiled model，对照普通模型的 forward，以及包含 backward、optimizer step 的训练时间。

要把编译和预热成本排除在稳态计时之外，并为两组使用相同输入形状、dtype、设备和同步方式。这道题的价值是建立基线：自动融合可能有帮助，但普通 attention 中的 `S × S` 中间矩阵及其数据搬运仍是后续自定义 kernel 要处理的主要问题。

### 2.3 weighted-sum 示例：正式写 Triton 前要读懂的接口

作业用 `weighted_sum(x, weight) = (x * weight).sum(-1)` 展示 Triton 前向和反向。它把行分给不同的 Triton program：`tl.program_id(0)` 确定当前行 tile；`tl.make_block_ptr` 指定基址、总形状、stride、起始 offset、tile 形状和维度顺序；`tl.load`/`tl.store` 用边界检查处理不能整除 tile 的情况；`advance` 移动到下一个 tile。

反向示例让每个 program 直接写自己负责的 `grad_x` tile；对于需要跨 program 汇总的 `grad_weight`，先写入 partial buffer，再在 kernel 外求和。`torch.autograd.Function` 的 `forward(ctx, ...)` 负责保存 backward 所需张量并启动 kernel，`backward(ctx, grad_out)` 负责返回与 forward 输入一一对应的梯度。读懂这个例子，重点是能自己解释 **grid、program、tile、stride、block pointer、边界检查和梯度归约** 分别在做什么。

### 2.4 `flash_forward`：本节最重的实现题

先写一个**纯 PyTorch 的分块参考版本**，再写 Triton kernel，最后加 causal masking；这三步在题目里分别是 (a)、(b)、(c)。参考版用于逐项对照 `S`、行最大值 `m`、归一化和 `L`；它可以较慢，目的在于为 Triton 版提供正确性参照。题目要求 tile 至少为 `16 × 16`，测试输入维度是至少 16 的 2 的幂；PyTorch 参考版在 (a) 可以先忽略 `is_causal`。

Triton 版的 grid 按题面设为 `(T_q, batch_size)`：一个 program 负责一个 batch 中的一个 query tile，在 kernel 里循环所有 key tiles。需实现作业 Algorithm 1 的**在线 softmax**，保持 `m`（迄今为止每行最大分数）、`l`（以当前 `m` 为基准的指数和）以及尚未最终归一化的输出累加器 `O_acc`。每读取一个 key tile，先计算局部分数，再更新 `m`，用 `exp(m_old - m_new)` 缩放旧的 `l` 和 `O_acc`，累加本 tile 的贡献；循环结束后输出 `O = O_acc / l`，以及 `L = m + log(l)`。这些行级累加量应使用 FP32，以减少数值误差；矩阵乘法使用 `tl.dot`，并留意概率和 `V` 相乘前的数据类型。

causal 版给函数增加可选的 `is_causal=False` 最后参数。根据**全局** query/key 下标构造 `key_index <= query_index` 的掩码，题面要求被遮挡的 score 加 `-1e6`；同时将 `is_causal` 存进 `ctx`，让 backward 沿用同一种 mask。不要只比较 tile 内的局部下标，否则离对角线较远的 tile 会遮错位置。

接口上有一个关键区别：kernel 可以同时写出 `O` 和 `L`，但 `torch.autograd.Function.forward(ctx, Q, K, V, is_causal=False)` **对外返回 `O`**；`L` 与 `Q,K,V,O` 要通过 `ctx.save_for_backward` 留给 backward。仓库的 forward 测试会从 saved tensors 中取出 `L` 并检查它的形状与数值。

### 2.5 `flash_backward`：用 `L` 重算，而不是保存完整概率矩阵

题目要求输入 `Q,K,V,O,dO,L`，输出 `dQ,dK,dV`，使用 PyTorch 操作并通过 `torch.compile` 编译这个 backward 计算，不要求在必做部分写 Triton backward。关键行向量是

$$
D_i = \sum_d O_{id}\,dO_{id}.
$$

随后按题面公式 13–19 重算 `S=QKᵀ/√d`、`P_ij=exp(S_ij-L_i)`，再求 `dV=PᵀdO`、`dP=dO Vᵀ`、`dS_ij=P_ij(dP_ij-D_i)`、`dQ=dS K/√d` 与 `dK=dSᵀQ/√d`。实现 causal 情况时，重算 `P` 必须使用与 forward 一致的 mask。由于 forward 有 `Q,K,V,is_causal` 四个输入，自定义 autograd 的 `backward` 应返回四项：`dQ,dK,dV,None`。

这一必做版本的 backward 使用常规 PyTorch 矩阵运算，重算过程中仍可能临时生成完整的 `N_q × N_k` 的 `S/P/dS`。因此，“Triton forward 不保存完整注意力矩阵”的显存优势不等于“本题必做的整个 forward + backward 都已摆脱平方级显存”；很长序列的完整往返仍可能 OOM。可选的分块 Triton backward 才进一步处理 backward 的这部分开销。

### 2.6 `flash_benchmarking`：分别比较三类时间

用 `triton.testing.do_bench` 比较自己的**Triton forward + PyTorch/compiled backward** 与普通 PyTorch attention。表中要分别列出 forward、backward、完整 forward + backward 的延迟；三者不是同一个指标，不能只报一个端到端数字。题面指定 **单张 B200**、batch size **1**、**causal mask**，并扫描从 128 到 65536 的多个 2 的幂序列长度、`d ∈ {16,32,64,128}`、`torch.bfloat16` 与 `torch.float32`；tile 大小可能需随输入调整。

随机输入须在计时区间外生成。若某组合 OOM 或当前设备不是 B200，需如实注明条件和未能完成的组合；其他 GPU 的实测值可用于开发时观察趋势，不能直接当作题面要求的 B200 结果。

### 2.7 可选：4.2.3 Triton backward

Algorithm 2 给出了进一步实现 Triton backward 的方向：可以分两遍重算 `P`，一遍为 `dK,dV`，另一遍为 `dQ`，以减少跨 program 的同步需求。它适合必做的 forward、PyTorch backward 和测试都通过后再做；**不是**完成 Section 4 必做题的前置条件。

## 3. 贯穿全节的几个数学量

| 量 | 形状（单 batch、无多头维度） | 用途 |
|---|---|
| `Q` | `[N_q, d]` | query；按 `B_q` 行切 tile |
| `K,V` | `[N_k, d]` | key/value；按 `B_k` 行切 tile |
| `S=QKᵀ/√d` | `[N_q,N_k]` | 标准实现的大分数矩阵；FlashAttention 只临时生成小 tile |
| `P=softmax(S)` | `[N_q,N_k]` | 标准实现的大概率矩阵；FlashAttention 不把整张矩阵长期写入 HBM |
| `O=PV` | `[N_q,d]` | 对外返回的 attention 输出 |
| `L_i=log Σ_j exp(S_ij)` | `[N_q]` | 每个 query 行的 logsumexp；供 backward 重算 `P` |
| `D_i=Σ_d O_id dO_id` | `[N_q]` | backward 中求 `dS` 的行级校正项 |

要能口头解释在线 softmax 的核心：不同 key tile 的分数最大值可能不同，所以合并新 tile 时必须先把旧累加器换算到**新的最大值基准**，再相加。直接把每个 tile 单独 softmax 后拼起来是不对的，因为各 tile 使用了不同的分母。

## 4. 推荐完成顺序与验收点

1. **建立 PyTorch 基线**：先让 `pytorch_attention` 的小形状跑通计时和显存记录，再扩大扫描范围、记录 OOM 并完成显存估算。
2. **加入编译对照**：以相同输入比较 eager/compiled attention；再复用 Section 2 的脚本测完整 Transformer，记录编译预热和稳态测量的边界。
3. **手推并核对小 tile**：用几个 query/key 行手算 `m、l、O_acc、L` 的更新，再写纯 PyTorch 分块版，与 `torch.softmax`/`torch.logsumexp` 对照。
4. **接上 PyTorch autograd 接口**：在 [适配器](../tests/adapters.py) 返回你的类，先运行 `uv run pytest -k test_flash_forward_pass_pytorch`。
5. **实现 Triton forward 和 causal**：先让非 causal 小形状正确，再加入 mask，运行 `uv run pytest -k test_flash_forward_pass_triton`；逐项比较 `O` 和 `L`。
6. **实现 backward**：按公式检查 `D、P、dS` 的形状和广播，再运行 `uv run pytest -k test_flash_backward`，最后可运行 `uv run pytest tests/test_attention.py`。
7. **完成性能表**：区分 forward、backward、完整往返，记录设备、dtype、形状、tile、OOM 与实际延迟。

仓库中适配器的准确函数名是 `get_flashattention_autograd_function_pytorch` 和 `get_flashattention_autograd_function_triton`。PDF 在 Triton forward 的一处交付文字中将后者简写成 `get_flash_autograd_function_triton`；编写适配器时以[当前测试文件](../tests/test_attention.py)的实际 import 为准。该文件还会分别测试 Triton 的 `is_causal=False/True`，而 `test_flash_backward` 同时覆盖 PyTorch 与 Triton 路径。

## 5. 一页复习

- **先测普通 attention**：随着序列长度增加，`[B,S,S]` 的 score/probability 是显存问题的核心。
- **再测 `torch.compile`**：得到自动优化的对照基线。
- **学会 Triton tile**：program id 选 tile；stride、offset 与边界检查决定读写位置。
- **FlashAttention 前向**：逐个 key tile 更新 `m、l、O_acc`，最终得到 `O、L`，避免存完整 `P`。
- **FlashAttention 反向**：使用 `L` 和 `D` 重算 `P`、`dS`，得到 `dQ,dK,dV`。
- **最后测性能**：正确性先于速度，表格注明实际硬件和无法运行的配置。
