# Section 3.1：Autograd Residuals 详解

- 对应代码：`cs336_systems/single.py`
- 对应作业：CS336 Assignment 2，Section 3.1
- 整理日期：2026-09-30

## 1. 这一节到底在做什么

Section 3.1 的核心目的，是观察 PyTorch 为了进行反向传播，在前向传播阶段保存了哪些张量，以及这些张量为什么会占用大量内存。

可以用一句话概括：

> 这段代码在“偷看”PyTorch autograd 为 backward 保存和读取了哪些中间张量。

这一节暂时没有优化 RMSNorm，也没有手动实现 backward。它只是用 `saved_tensors_hooks` 观察 autograd 的内部行为，为后面的算子融合和 activation checkpointing 做准备。

这里的整体逻辑是：

```text
3.1 Autograd Residuals
观察：backward 到底保存了什么？

        ↓

3.1.1 Operator Fusion
优化：减少细粒度算子造成的多余保存

        ↓

3.2 Activation Checkpointing
进一步优化：不长期保存激活，在 backward 时重新计算
```

## 2. 为什么 forward 需要保存张量

神经网络训练的基本过程是：

```text
forward → 得到 loss → backward → 得到梯度
```

反向传播要计算导数，而很多导数依赖前向传播时的输入或中间结果。

例如：

```python
y = x**2
```

它的导数是：

$$
\frac{\partial y}{\partial x}=2x
$$

当 backward 开始时，PyTorch 必须知道 forward 时的 `x`。因此，执行 `y = x**2` 时，autograd 会保存 `x`。

可以将它理解为：

```text
forward：
    计算 y = x²
    把 x 放进一个抽屉里

backward：
    从抽屉里取出 x
    计算 grad_x = grad_y × 2x
```

这个被保存的 `x` 就叫作：

- saved tensor；
- autograd residual；
- 为 backward 保存的激活。

### 2.1 注意 residual 的两种含义

这里的 residual 不是 Transformer 中的 residual connection。

| 名称 | 含义 |
|---|---|
| Transformer residual | 残差连接或 residual stream 中的 hidden state |
| autograd residual | 为了 backward 保存的任意张量 |

## 3. 不是所有操作都需要保存输入

例如：

```python
y = x * 3
```

它的导数是：

$$
\frac{\partial y}{\partial x}=3
$$

这个导数不依赖 `x` 的具体数值，因此 backward 不一定需要保存 `x`。

但如果是：

```python
y = x * x
```

则有：

$$
\frac{\partial y}{\partial x}=2x
$$

backward 需要知道原来的 `x`，所以必须保存它。

再例如：

```python
z = a * b
```

梯度是：

$$
\frac{\partial z}{\partial a}=b,
\qquad
\frac{\partial z}{\partial b}=a
$$

为了计算 `a.grad`，需要保存 `b`；为了计算 `b.grad`，需要保存 `a`。

因此，PyTorch 中的每个算子都会根据自己的 backward 公式，决定在 forward 时需要保存哪些张量。

## 4. `single.py` 中的输入和 RMSNorm

代码首先创建输入：

```python
x = torch.randn((4, 512, 2560), requires_grad=True)
```

形状可以理解为：

```text
(batch_size, sequence_length, hidden_size)
= (4, 512, 2560)
```

`requires_grad=True` 表示需要计算损失关于 `x` 的梯度。执行 backward 后，结果会保存在：

```python
x.grad
```

这个 FP32 张量包含：

$$
4\times512\times2560=5,242,880
$$

个元素，占用：

$$
5,242,880\times4\text{ bytes}=20\text{ MiB}
$$

RMSNorm 的前向传播是：

```python
def forward(self, x):
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    x = x * rms
    return self.weight * x
```

对应数学公式：

$$
y=w\odot
\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}
$$

## 5. PyTorch 实际看到的是多个独立算子

数学上，RMSNorm 看起来只有一个公式。但是 PyTorch 看到的是一系列独立操作：

```text
x
↓
pow(2)
↓
mean
↓
add eps
↓
rsqrt
↓
multiply x
↓
multiply weight
↓
y
```

为了便于分析，可以把所有中间结果写出来：

```python
squared = x.pow(2)
mean_square = squared.mean(-1, keepdim=True)
stabilized = mean_square + eps
rms = torch.rsqrt(stabilized)
normalized = x * rms
y = weight * normalized
```

每个算子都有自己的 backward，而每个 backward 可能需要不同的前向张量。

## 6. RMSNorm 的 backward 为什么需要这些张量

### 6.1 最后一步：乘以 `weight`

前向传播：

```python
y = weight * normalized
```

假设上游传来的梯度为：

```text
grad_y = ∂loss / ∂y
```

计算 `weight` 的梯度时：

$$
\frac{\partial L}{\partial weight}
=
\frac{\partial L}{\partial y}\odot normalized
$$

所以需要保存前向传播的 `normalized`。

计算 `normalized` 的梯度时：

$$
\frac{\partial L}{\partial normalized}
=
\frac{\partial L}{\partial y}\odot weight
$$

所以还需要保存 `weight`。

因此，最后一次乘法的 backward 需要：

```text
normalized：形状 (4, 512, 2560)
weight：形状 (2560,)
```

日志中可能对应：

```text
Saving residual: shape=torch.Size([4, 512, 2560]), grad_fn=<MulBackward0 ...>
Saving residual: shape=torch.Size([2560]), grad_fn=None
```

`weight.grad_fn=None` 是因为它是叶子参数，并不代表它不需要梯度。

### 6.2 中间步骤：`normalized = x * rms`

前向传播：

```python
normalized = x * rms
```

计算 `x` 的这一部分梯度时：

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial normalized}\odot rms
$$

因此需要保存 `rms`。

计算 `rms` 的梯度时：

$$
\frac{\partial L}{\partial rms}
=
\sum_d
\frac{\partial L}{\partial normalized_d}x_d
$$

因此还需要保存原始输入 `x`。

这里需要的张量形状为：

```text
x:   (4, 512, 2560)
rms: (4, 512, 1)
```

### 6.3 `rsqrt` 操作

前向传播：

```python
rms = torch.rsqrt(stabilized)
```

也就是：

$$
rms=stabilized^{-1/2}
$$

它的导数是：

$$
\frac{\partial rms}{\partial stabilized}
=
-\frac{1}{2}stabilized^{-3/2}
$$

为了计算这个导数，backward 需要 forward 时的输入或输出。PyTorch 的具体实现会保存 `rsqrt` 的相关结果，因此日志中可能看到：

```text
shape=torch.Size([4, 512, 1])
grad_fn=<RsqrtBackward0 ...>
```

### 6.4 `x.pow(2)` 操作

前向传播：

```python
squared = x.pow(2)
```

导数是：

$$
\frac{\partial squared}{\partial x}=2x
$$

所以它的 backward 也需要原始输入 `x`。

## 7. 为什么同一个形状可能打印多次

作业中的输出类似：

```text
Saving residual: [4, 512, 2560]
Saving residual: [4, 512, 1]
Saving residual: [4, 512, 1]
Saving residual: [4, 512, 2560]
Saving residual: [4, 512, 2560]
Saving residual: [2560]
```

原因是不同 backward 节点都可能需要相应张量：

```text
PowBackward          需要 x
MulBackward          需要 x 和 rms
另一个 MulBackward   需要 normalized 和 weight
```

因此，在计算图中可能有多个 backward 节点分别持有对 `x` 或其他张量的引用。

但是需要注意：

> 同一个形状被打印多次，不一定表示 PyTorch 重新分配并复制了多份完整数据。

多个 saved tensor 可能引用同一个底层 storage。因此，不能把日志中所有张量的大小简单相加，并将结果直接当成真实显存增加量。

这段代码观察的是 autograd 的保存逻辑，而不是精确的显存分配情况。精确显存分析应使用 PyTorch memory snapshot 或 Nsight Systems 等工具。

## 8. `saved_tensors_hooks` 在这里的作用

代码使用：

```python
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward()
```

它给 autograd 的 saved tensor 机制安装了一对钩子。

### 8.1 `pack_hook`

当某个 forward 操作准备保存张量时，PyTorch 相当于执行：

```python
packed = pack_hook(tensor)
```

原代码是：

```python
def pack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t
```

它只打印信息，然后返回张量。因此，它不会改变模型的数学计算，也不会主动减少显存。

当前 PyTorch 更建议使用 `t.detach()`，避免返回值继续持有不必要的 autograd 引用：

```python
def pack_hook(t):
    print(f"Saving residual: shape={t.shape}, dtype={t.dtype}, grad_fn={t.grad_fn}")
    return t.detach()
```

### 8.2 `unpack_hook`

当 backward 需要之前保存的张量时，PyTorch 相当于执行：

```python
tensor = unpack_hook(packed)
```

原代码为：

```python
def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t
```

它打印正在读取的张量，然后将其交给相应的 backward 算子。

完整生命周期是：

```text
forward 操作需要保存 x
        ↓
pack_hook(x)
        ↓
打印 Saving residual
        ↓
计算图保存 pack_hook 的返回值
        ↓
开始 backward
        ↓
某个 backward 节点需要 x
        ↓
unpack_hook(saved_x)
        ↓
打印 Loading residual
        ↓
使用恢复后的 x 计算梯度
```

## 9. `y.sum().backward()` 做了什么

`y` 的形状是：

```text
(4, 512, 2560)
```

为了方便调用 backward，代码先执行：

```python
loss = y.sum()
```

这样得到一个标量：

```text
loss.shape == ()
```

然后执行：

```python
loss.backward()
```

反向传播大致按照以下方向进行：

```text
loss = sum(y)
        ↓
y = weight * normalized
        ↓
normalized = x * rms
        ↓
rms = rsqrt(mean(x²) + eps)
        ↓
x
```

每走到一个 backward 节点，autograd 就会读取该节点需要的 saved tensors，最终计算出：

```python
x.grad
ln.weight.grad
```

## 10. 3.1 希望我们得出什么结论

### 10.1 训练比推理更耗显存

推理通常不需要 backward：

```python
with torch.no_grad():
    y = model(x)
```

中间结果用完之后就可以释放。

训练时，PyTorch 必须判断：

```text
这个张量之后的 backward 会不会用到？
```

如果会用到，就要让它一直存活到 backward，因此增加激活显存。

### 10.2 简单公式可能包含很多细粒度操作

RMSNorm 的数学公式很短，但原始 PyTorch 实现包含：

```text
pow → mean → add → rsqrt → multiply → multiply
```

autograd 分别处理这些操作，每个操作保存自己的 backward 所需张量，因此可能出现冗余保存。

### 10.3 为后面的优化方法做铺垫

Section 3.1 发现问题：

```text
autograd 保存了很多中间张量
```

Section 3.1.1 引入 operator fusion：

```text
把多个细粒度操作融合成一个整体操作
→ 只保存整体 backward 真正需要的张量
```

Section 3.2 引入 activation checkpointing：

```text
不长期保存一部分激活
→ backward 时重新执行 forward 得到它们
→ 用额外计算换取显存
```

## 11. 最值得记住的要点

1. autograd 为了计算 backward，会在 forward 中保存必要的输入或中间结果。
2. 这些张量称为 saved tensors 或 autograd residuals。
3. `pack_hook` 在张量被保存时调用，`unpack_hook` 在 backward 读取张量时调用。
4. `single.py` 中的 hooks 只用于观察，没有减少内存，也没有改变梯度计算。
5. RMSNorm 虽然公式简单，但由多个独立算子组成，每个算子都可能保存自己的张量。
6. hooks 的日志表示逻辑上的保存行为，不一定等于新增的物理内存分配。
7. 3.1 的最终目的，是解释训练中的激活显存从哪里来，并为算子融合和 activation checkpointing 建立基础。

