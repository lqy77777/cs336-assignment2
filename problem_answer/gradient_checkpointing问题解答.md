# Memory-Optimal Gradient Checkpointing 问题解答

整理日期：2026-10-02  
对应题目：`cs336_assignment2_systems.pdf` 中的 `gradient_checkpointing`  
分析范围：Transformer block 堆叠产生的 activation/residual 显存，不把参数、优化器状态等误算进理论上的 activation memory 复杂度。

## 1. 先理解 checkpoint 为什么能够节省峰值显存

`torch.utils.checkpoint.checkpoint(function, *args)` 的核心不是让 residual 永远消失，而是改变它的存活时间：第一次 forward 主要保留 checkpoint 区域的输入，不让区域内部的大量 residual 一直存活到 backward；等 backward 真正需要这些 residual 时，再从保存的输入重新执行一次局部 forward。重算产生的 residual 会被当前局部 backward 消耗并释放，因此不同 checkpoint 区域的 residual 不必同时存在。

例如，四个 block 不做 checkpoint 时，在 forward 结束处可能同时存在 `r1, r2, r3, r4`；若将其分成两个 checkpoint 区域，backward 可以先重算并使用 `r3, r4`，释放后再重算并使用 `r1, r2`。总共仍然生成过所有 residual，并且还多做了 forward，但峰值时刻不再需要同时保存四份 residual。因此，checkpoint 优化的是“某个时刻同时存活的 tensor 数量”，而不是 tensor 的累计创建量。

## 2. 符号与问题中的量

| 符号 | 含义 |
|---|---|
| $N$ | 顺序堆叠的 Transformer block 数量，本题 XL 配置中为 32 |
| $A$ | 一个 checkpoint 入口 activation 的大小，即 checkpoint 为以后重算长期保存的输入 |
| $R$ | 一个普通 Transformer block 为 backward 保存的全部 residual 大小 |
| $s$ | 一个非嵌套 checkpoint 区域中包含的 Transformer block 数量 |
| peak allocated memory | PyTorch 报告的运行过程中实际 tensor allocation 峰值，本次用 `torch.cuda.max_memory_allocated()` 测量 |

题目假设单个 block 的 residual 比每个 checkpoint 的少量管理开销大得多。要注意，checkpoint 输入 activation 本身仍然是真实的激活显存，不能仅因为它比 $R$ 小就从渐近分析中完全删除。

---

## 3. (a) 忽略计算代价时的最小峰值激活显存策略

### 3.1 可直接用于作业提交的 4 句话

> 我将连续的 Transformer blocks 递归地近似二等分，对左右两个半区间分别使用 checkpoint，并在每个 checkpoint 内继续递归，直到叶子区间只包含一个 block。反向传播时，只需同时保留 checkpoint 树中的一条根到叶路径，以及当前一个 block 被物化出来的 residual。由于平衡递归树的深度为 $O(\log N)$，峰值激活内存为 $O(\log N)$。每个 block 在每层递归中至多被重新计算一次，因此总计算量为 $O(N\log N)$。

### 3.2 为什么普通的单层分段只有 $O(\sqrt N)$ 显存

先考虑不允许嵌套的扁平分段。若设置 $K$ 个 checkpoint 区域，那么需要长期保存约 $K$ 个区域入口；每个区域包含约 $N/K$ 个 block，backward 重算一个区域时会临时物化约 $N/K$ 个 block 的 residual。因此忽略常数后，峰值激活显存为

$$
M(N,K)=O\left(K+\frac{N}{K}\right).
$$

令两项平衡，即 $K\approx\sqrt N$，可以得到 $O(\sqrt N)$ 峰值显存。这个结果比不做 checkpoint 的 $O(N)$ 好，但还不是允许嵌套时的最低结果。

这里还有两个容易产生误解的极端方案：

1. 只用一个 checkpoint 包住全部 $N$ 个 block，第一次 forward 确实只保存很少的边界，但 backward 重算这个巨大区域时仍可能同时物化 $N$ 个 block 的 residual，峰值仍可达到 $O(N)$。
2. 在一个普通循环里分别 checkpoint 每个 block 是“扁平的多个 checkpoint”，而不是递归嵌套；它会长期保存 $N$ 个 block 输入，因此单凭这种写法仍是 $O(N)$ 个边界 activation。

### 3.3 平衡递归 checkpoint

更好的做法是把 `[0, N)` 近似二等分，对左右半段分别调用 checkpoint，并且在传给 checkpoint 的函数内部继续二等分。递归直到叶子只剩一个 block，此时允许该 block 在局部 backward 前物化自己的 residual。

以 $N=8$ 为例，checkpoint 区域形成一棵平衡树：

```text
[0, 8)
├── [0, 4)
│   ├── [0, 2)
│   │   ├── block 0
│   │   └── block 1
│   └── [2, 4)
└── [4, 8)
    ├── [4, 6)
    └── [6, 8)
```

backward 处理 `block 7` 时，只需要展开类似 `[0,8) → [4,8) → [6,8) → block 7` 的一条路径，而不需要同时展开整棵树。平衡树的深度为 $O(\log N)$，每层只留下常数个 checkpoint 边界，所以峰值显存满足

$$
M(N)=M(\lceil N/2\rceil)+O(A),\qquad M(1)=O(R),
$$

从而

$$
M(N)=O(R+A\log N)=O(\log N).
$$

每一层递归累计重算 $O(N)$ 个 block，递归层数为 $O(\log N)$，因此 forward 与重算工作的递推关系可以写成

$$
C(N)=2C(N/2)+O(N)=O(N\log N).
$$

正常的 $O(N)$ backward 工作不会改变总计算的 $O(N\log N)$ 渐近阶。

### 3.4 代码草图的参数、变量和返回值

| 名称 | 含义 |
|---|---|
| `x` | 当前区间的入口 activation |
| `blocks` | 按执行顺序保存的 Transformer blocks，例如 `nn.ModuleList` |
| `lo` | 当前区间包含的第一个 block 下标 |
| `hi` | 当前区间不包含的结束下标，因此区间写作 `[lo, hi)` |
| `mid` | 将当前区间近似二等分的位置 |
| 返回值 | 依次执行 `blocks[lo:hi]` 后得到的 activation |

### 3.5 短代码草图

```python
from torch.utils.checkpoint import checkpoint


def recursive_blocks(x, blocks, lo, hi):
    if hi - lo == 1:
        return blocks[lo](x)

    mid = (lo + hi) // 2

    x = checkpoint(
        lambda z: recursive_blocks(z, blocks, lo, mid),
        x,
        use_reentrant=False,
    )
    x = checkpoint(
        lambda z: recursive_blocks(z, blocks, mid, hi),
        x,
        use_reentrant=False,
    )
    return x


def forward_with_recursive_checkpointing(x, blocks):
    return recursive_blocks(x, blocks, 0, len(blocks))
```

逐步理解这段代码：

1. `hi - lo == 1` 是递归终点。最终至少要临时物化一个 block 的 residual 才能计算该 block 的 backward，所以叶子直接执行一个 block。
2. `mid` 让左右子区间最多相差一个 block，使递归深度保持为 $O(\log N)$；若每次只切下一个 block，深度会退化成 $O(N)$。
3. 左半段的 checkpoint 函数内部再次调用 `recursive_blocks`，因此左半段执行期间还会建立更小的 checkpoint，这才是真正的嵌套。
4. 左半段输出成为右半段输入，右半段也采用相同递归策略，最终返回值与普通顺序执行全部 blocks 相同。

---

## 4. (b) 只允许一层重算时的最佳策略

### 4.1 可直接用于作业提交的 5 句话

> 在不允许嵌套 checkpoint 时，我对每个完整的 Transformer block 分别使用一次 checkpoint，因为更大的区域虽然减少了 checkpoint 输入的存储量，却会在重算时同时物化多个 blocks 的大量 residual。若将一个 block 进一步拆成 attention 和 FFN 两个 checkpoint，则会走向另一个极端，使长期存活的 block 边界 activation 数量翻倍。在 RTX 5090、PyTorch 2.11.0、batch size 4、sequence length 2048 和 XL block 维度下，半个 block、一个 block 和两个 blocks 三种 checkpoint 区域的 peak allocated memory 分别为 14.127 GiB、12.487 GiB 和 16.808 GiB。因此，在测试的三个相邻 checkpoint 粒度中，每个完整 Transformer block 使用一个非嵌套 checkpoint 的实测峰值最低，为 12.487 GiB。由于包含 32 个独立参数层的 FP32 实验在 backward 时超过了当前 GPU 的容量，这组对比数据使用一个编译后的 XL block 共享参数并连续执行 32 次来隔离 activation-memory 行为，不能视为完整独立参数模型的绝对峰值。

### 4.2 理论预测

若每个 checkpoint 区域包含 $s$ 个完整 block，则总共有约 $N/s$ 个区域。第一次 forward 结束后需要保存约 $(N/s)A$ 的区域入口，而 backward 重算某一区域时需要临时物化约 $sR$ 的 residual，因此与分段粒度相关的峰值近似为

$$
M(s)\approx \frac{N}{s}A+sR+M_{\text{fixed}}.
$$

把 $s$ 暂时看成连续变量，可得理论平衡点

$$
s^*=\sqrt{\frac{NA}{R}}.
$$

本题输入 activation 的 FP32 大小为

$$
A=4\times2048\times2560\times4\ \text{bytes}=80\ \text{MiB}.
$$

题面前文测得一个编译后的 XL Transformer block 保存约 $R=3651.31$ MiB residual，且 $N=32$，所以

$$
s^*\approx\sqrt{\frac{32\times80}{3651.31}}\approx0.84.
$$

这个估计说明最优粒度应该在一个完整 block 附近：两个 blocks 一组会让重算 residual 明显增大；拆成 attention、FFN 两个半 block 虽可缩小单次重算区域，却会把长期保存的边界 activation 数量从 32 增加到 64。由于 attention 和 FFN 的 residual 大小并不相等，最终仍需实测三个相邻候选，而不能只依赖这个连续近似。

### 4.3 本次 profiling 条件

| 项目 | 本次设置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，31.358 GiB |
| PyTorch / CUDA runtime | PyTorch 2.11.0+cu130 / CUDA 13.0 |
| 逻辑 Transformer block 数 | 32 |
| batch size / sequence length | 4 / 2048 |
| `d_model` / `d_ff` / `num_heads` | 2560 / 10240 / 32 |
| activation 与参数 dtype | FP32 |
| 编译方式 | `torch.compile(..., fullgraph=True)` |
| checkpoint 实现 | `use_reentrant=False`，没有嵌套 checkpoint |
| 测量范围 | 一个 XL block 共享参数并顺序应用 32 次，执行 forward、`sum()` 和 backward |
| 测量方法 | 预热一次，清除梯度和无用 tensor，`reset_peak_memory_stats()` 后正式测量，CUDA 同步后读取 `max_memory_allocated()` |

共享同一个 block 的理由是：题目研究的是 32 个相同 block 调用所产生的 activation/residual 存活规律，而当前 31.358 GiB GPU 无法公平容纳 FP32 的 32 份独立 XL 参数、完整梯度和所有候选策略。实际尝试 32 个独立 FP32 blocks 时，参数本身占 12.501 GiB，checkpoint forward 结束后的 allocated memory 为 15.231 GiB，但 backward 随后发生 CUDA OOM。共享参数不会改变 checkpoint 边界数量和每次 block 调用产生的 residual 形状，但会显著降低常驻参数与梯度内存。因此，下表适合验证“哪一种 activation checkpoint 粒度更低”，绝不能冒充完整独立参数 XL 模型在其他硬件上的绝对峰值。

### 4.4 三个相邻粒度的定义与结果

| checkpoint 粒度 | checkpoint 区域数 | peak allocated | peak reserved | 相对一个 block |
|---|---:|---:|---:|---:|
| 半个 block：attention 与 FFN 分开 | 64 | 14.127 GiB | 14.984 GiB | +1.640 GiB |
| 一个完整 block | 32 | **12.487 GiB** | 14.988 GiB | 最低 |
| 两个完整 blocks | 16 | 16.808 GiB | 19.303 GiB | +4.320 GiB |

这里用 `max_memory_allocated()` 作为题目要求的 peak memory；`max_memory_reserved()` 是 PyTorch caching allocator 向 CUDA 申请并保留的内存，包含当时未被 tensor 实际使用的缓存块，因此作为辅助信息列出，但不用于决定最优策略。三组测试在正式测量前的常驻 allocated baseline 都约为 0.411 GiB。

结果符合理论上的两侧权衡：

- 半个 block 的单次重算区域较小，但 64 个长期保存的边界输入以及分隔 attention/FFN 带来的额外存活量使峰值升到 14.127 GiB。
- 一个 block 同时控制了边界数量与局部 residual，峰值最低，为 12.487 GiB。
- 两个 blocks 减少了边界输入数量，但 backward 重算时需要同时支持更大的局部计算图，峰值升到 16.808 GiB。

### 4.5 profiling 代码结构

候选粒度的核心写法如下。这里的 `attn_part` 返回 residual connection 后的 `y`，`ffn_part` 接收 `y` 并返回完整 block 输出；两个函数顺序执行与原始 Transformer block 的 forward 一致。

```python
def attn_part(x, block):
    return x + block.Attention(block.rms1(x))


def ffn_part(y, block):
    return y + block.FFN(block.rms2(y))


# 更小候选：半个 block 一个 checkpoint
for _ in range(32):
    x = checkpoint(
        lambda z: attn_part(z, block), x, use_reentrant=False
    )
    x = checkpoint(
        lambda z: ffn_part(z, block), x, use_reentrant=False
    )

# 最优候选：一个完整 block 一个 checkpoint
for _ in range(32):
    x = checkpoint(block, x, use_reentrant=False)

# 更大候选：两个 blocks 一个 checkpoint
def two_blocks(x):
    x = block(x)
    return block(x)


for _ in range(16):
    x = checkpoint(two_blocks, x, use_reentrant=False)
```

峰值测量的关键顺序是：

```python
# 先执行一次完整 warm-up，然后清除 warm-up 留下的梯度和 tensor。
block.zero_grad(set_to_none=True)
del warmup_x, warmup_y
gc.collect()
torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()
y = checkpointed_forward(x)
y.sum().backward()
torch.cuda.synchronize()

peak_gib = torch.cuda.max_memory_allocated() / 1024**3
print(f"peak allocated: {peak_gib:.3f} GiB")
```

如果以后在另一张 GPU 或完整独立参数模型上重新测试，提交时应替换为那次实测值，并保证三种粒度使用完全相同的 dtype、编译设置、输入和测量边界。

---

## 5. 最值得记住的结论

1. checkpoint 节省的是峰值存活内存，而不是累计计算量；backward 重算的 residual 会被局部消费和释放。
2. 只做一层 checkpoint 时，区域过大会让重算 residual 太多，区域过小会让长期保存的边界输入太多。
3. 允许任意嵌套且忽略计算代价时，平衡递归 checkpoint 达到 $O(\log N)$ 峰值激活显存和 $O(N\log N)$ 计算。
4. 本次 XL 激活实验中，一个完整 Transformer block 一个非嵌套 checkpoint 的峰值最低，实测为 12.487 GiB。
5. 理论 activation-memory 复杂度与完整训练峰值不是同一个量；完整训练还包含参数、梯度、优化器状态、临时 workspace 和 allocator 行为。

## 6. 参考资料

- PyTorch `torch.utils.checkpoint` 文档：<https://docs.pytorch.org/docs/stable/checkpoint>
- Tianqi Chen et al., *Training Deep Nets with Sublinear Memory Cost*：<https://arxiv.org/abs/1604.06174>
- 本项目作业说明：`cs336_assignment2_systems.pdf`
