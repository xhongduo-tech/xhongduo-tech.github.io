## 核心结论

Gradient Accumulation，中文常叫“梯度累积”，指的是：连续处理多个小批次数据，把它们产生的梯度累加起来，等累积到指定次数后，再执行一次参数更新。

它解决的问题不是“让单次前向计算放进更多样本”，而是“让一次参数更新基于更多样本的平均梯度”。所以它是“大 Batch 的模拟训练”，不是“真正把 Batch 变大”。

核心公式是：

$$
B_{eff} = b \times k
$$

其中：

| 术语 | 白话解释 | 典型含义 |
|---|---|---|
| `micro-batch` | 单次真正送进模型的小批次 | 显存实际能承受的 batch |
| `accumulation steps` | 攒多少次梯度后更新一次 | 记作 $k$ |
| `effective batch size` | 一次参数更新等效看到多少样本 | 记作 $B_{eff}$ |

玩具例子：显存一次只能放 2 张图，但希望一次更新基于 6 张图。做法是把 6 张图拆成 3 次，每次放 2 张图，前两次只计算梯度不更新参数，第三次算完后统一更新一次。此时：

$$
B_{eff} = 2 \times 3 = 6
$$

真实工程例子：训练长上下文 LLM 时，单卡可能只能跑 `micro_batch=1`。如果希望一次更新至少看见 16 条样本，可以设置 `accumulation_steps=16`。单次显存仍然只承担 1 条样本的前向和反向，但每次 `optimizer.step()` 对应 16 条样本的累积梯度。

---

## 问题定义与边界

训练神经网络时，显存主要被模型参数、优化器状态、前向激活值、反向梯度占用。这里的“激活值”是前向计算中间结果，反向传播需要用它们计算梯度，所以不能随便丢掉。模型越大、输入越长、分辨率越高，单个样本占用的显存越多。

Gradient Accumulation 面对的是这个问题：目标 batch size 放不进显存，但又希望一次参数更新有较大的样本统计规模。

它不解决所有“大 batch 相关问题”。尤其要明确三个概念：

| 概念 | 计算方式 | 关注点 |
|---|---:|---|
| `micro-batch size` | 单次前向实际输入的样本数 | 单步显存占用 |
| `effective batch size` | `micro_batch × accumulation_steps` | 单次参数更新看到的样本数 |
| `global batch size` | `micro_batch × accumulation_steps × world_size` | 多卡训练下全局一次更新的样本数 |

其中 `world_size` 是分布式训练里的进程或 GPU 数量。

新手版例子：训练长上下文 LLM 时，单卡只能放 `batch=1`，但希望模型每次更新都“看见”更多样本。可以连续跑 8 个 `batch=1`，每次只反向传播并累积梯度，第 8 次之后再统一更新一次。此时有效 batch 是 8，但单次显存容量没有变大。

边界是：

| 不等价对象 | 原因 |
|---|---|
| 不等价于真正的大 batch 单步训练 | BatchNorm 等层仍然看到的是 micro-batch 统计 |
| 不等价于更大的单步显存容量 | 单次前向仍只能处理 micro-batch |
| 不等价于自动提升吞吐 | 多次前向/反向后才更新一次，训练控制更复杂 |
| 不等价于数据并行扩展 | 数据并行会同时在多卡计算，梯度累积通常是时间上串行累积 |

因此，Gradient Accumulation 的精确定义是：在显存受限时，通过延迟参数更新，让一次更新使用多个 micro-batch 的累积梯度。

---

## 核心机制与推导

在 PyTorch 里，`loss.backward()` 默认会把梯度累加到参数的 `.grad` 字段中，而不是自动覆盖。`.grad` 可以理解为每个参数旁边的“梯度缓存”。只要不调用 `optimizer.zero_grad()` 清空它，多次 `backward()` 的结果就会继续加进去。

标准流程是：

| 步骤 | 操作 | 说明 |
|---:|---|---|
| 1 | 前向计算 | 用一个 micro-batch 得到预测 |
| 2 | 计算 loss | 得到当前 micro-batch 的损失 |
| 3 | 反向传播 | `loss.backward()` 把梯度加到 `.grad` |
| 4 | 重复 $k$ 次 | 累积 $k$ 个 micro-batch 的梯度 |
| 5 | 参数更新 | `optimizer.step()` 更新参数 |
| 6 | 清空梯度 | `optimizer.zero_grad()` 开始下一轮累积 |

设 micro-batch size 为 $b$，累积步数为 $k$，则：

$$
B_{eff} = b \times k
$$

如果每个 micro-batch 的 loss 都是按样本均值计算，并且每个 micro-batch 大小一致，那么第 $i$ 个 micro-batch 的梯度是：

$$
\nabla_\theta L_i
$$

一次参数更新应该使用的平均梯度是：

$$
g_{update} = \frac{1}{k}\sum_{i=1}^{k}\nabla_\theta L_i
$$

这就是为什么代码里经常写：

```python
loss = loss / accumulation_steps
loss.backward()
```

因为每次先除以 $k$，再累加 $k$ 次，最终 `.grad` 中保存的就是平均梯度，而不是放大 $k$ 倍的梯度。

玩具例子：设 `b=2`、`k=3`，三个 micro-batch 的梯度分别是 `2、4、6`。如果直接累加，得到：

$$
2 + 4 + 6 = 12
$$

如果要模拟大 batch 的平均梯度，需要除以 3：

$$
\frac{2 + 4 + 6}{3} = 4
$$

这对应“6 个样本一起参与平均后再更新”的效果。

多卡训练时还要把 GPU 数量算进去：

$$
global\ batch = micro\_batch \times accumulation\_steps \times world\_size
$$

例如每张卡 `micro_batch=2`，累积 8 步，使用 4 张卡：

$$
global\ batch = 2 \times 8 \times 4 = 64
$$

---

## 代码实现

最小 PyTorch 写法如下：

```python
optimizer.zero_grad()

for step, batch in enumerate(loader):
    outputs = model(batch["x"])
    loss = criterion(outputs, batch["y"])

    # 除以 accumulation_steps，是为了让累积后的梯度仍然是平均梯度
    loss = loss / accumulation_steps
    loss.backward()

    # 到达一个 effective batch 后，才真正更新参数
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

`zero_grad()` 不能每个 micro-batch 都调用。否则前面 micro-batch 的梯度刚算出来就被清掉，累积就不存在了。正确位置是在一次参数更新之后。

下面是一个可运行的纯 Python 玩具例子，用数字模拟梯度累积：

```python
def accumulated_gradient(micro_gradients, accumulation_steps):
    grad_buffer = 0.0
    updates = []

    for i, grad in enumerate(micro_gradients, start=1):
        # 模拟 loss / accumulation_steps 后再 backward
        grad_buffer += grad / accumulation_steps

        if i % accumulation_steps == 0:
            updates.append(grad_buffer)
            grad_buffer = 0.0

    return updates


updates = accumulated_gradient([2.0, 4.0, 6.0], accumulation_steps=3)

assert updates == [4.0]
assert (2.0 + 4.0 + 6.0) / 3 == updates[0]
```

AMP，中文常叫“自动混合精度”，指的是用 FP16/BF16 等低精度格式加速训练、降低显存，同时用缩放技术避免梯度下溢。使用 AMP 时，`GradScaler` 的顺序要注意：先对每个缩放后的 loss 做 `backward()`，等一个 effective batch 累积完成后，再 `step()` 和 `update()`。

```python
scaler = torch.cuda.amp.GradScaler()
optimizer.zero_grad(set_to_none=True)

for step, batch in enumerate(loader):
    with torch.cuda.amp.autocast():
        outputs = model(batch["x"])
        loss = criterion(outputs, batch["y"])
        loss = loss / accumulation_steps

    scaler.scale(loss).backward()

    if (step + 1) % accumulation_steps == 0:
        # 如果需要梯度裁剪，应先 unscale
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

这里的关键是：`step()` 只能在累积完成后执行。前面每次 `backward()` 都只是把梯度放进 `.grad`，最后一次才结算。

真实工程例子：训练一个 7B 参数 LLM，单卡 24GB 显存可能只能放下 `micro_batch=1` 和较短上下文。为了让训练更稳定，可以设 `accumulation_steps=32`，再配合 AMP 和 activation checkpointing。activation checkpointing 是一种用额外计算换显存的方法：前向时少存一部分激活值，反向时重新计算它们。

---

## 工程权衡与常见坑

Gradient Accumulation 会改变“训练步”的含义。micro-step 是每次前向/反向；update-step 是每次 `optimizer.step()`。大多数日志、评估、保存 checkpoint、学习率调度，应该按 update-step 计算。

例如 `accumulation_steps=8` 时，跑 8 次前向/反向才算 1 次真正更新。如果日志每个 micro-step 都记录学习率，而 scheduler 每个 update-step 才更新一次，曲线看起来就会错位。

常见问题如下：

| 常见坑 | 后果 | 修正方式 |
|---|---|---|
| `loss` 没有缩放 | 梯度放大 $k$ 倍，等价于隐式增大学习率 | `loss = loss / accumulation_steps` |
| 每个 micro-batch 都 `zero_grad()` | 梯度无法累积 | 只在 `optimizer.step()` 后清空 |
| scheduler 按 micro-step 走 | 学习率变化过快 | 通常按 update-step 调度 |
| AMP 提前 `step()` | 参数过早更新 | 累积完成后再 `scaler.step()` |
| AMP 提前或重复 `unscale_()` | 梯度裁剪和缩放状态混乱 | 只在更新前 unscale |
| BatchNorm 统计不稳定 | 小 batch 统计偏差大 | 换 LayerNorm/GroupNorm，或冻结 BN |
| 最后一个不满 batch 仍按固定 $k$ 缩放 | 梯度偏小或偏大 | 按实际累积样本数重新缩放 |

错误写法：

```python
for batch in loader:
    optimizer.zero_grad()

    loss = criterion(model(batch["x"]), batch["y"])
    loss.backward()

    optimizer.step()
```

这段代码每个 micro-batch 都清空并更新，实际上没有梯度累积。

正确写法：

```python
optimizer.zero_grad(set_to_none=True)

for step, batch in enumerate(loader):
    loss = criterion(model(batch["x"]), batch["y"])
    loss = loss / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

最后一个不满 batch 要特别处理。假设计划 `accumulation_steps=8`，但 epoch 末尾只剩 3 个 micro-batch。如果仍然每个 loss 除以 8，最终梯度会偏小。更严谨的做法是按实际样本数加权，或者在数据加载阶段使用 `drop_last=True` 丢弃不满组的数据。前者更充分利用数据，后者实现更简单。

BatchNorm 是另一个关键边界。BatchNorm，中文叫“批归一化”，会用当前 mini-batch 的均值和方差归一化特征。即使梯度累积让一次参数更新看起来像大 batch，BatchNorm 在每次前向时看到的仍然只是 micro-batch。因此 `micro_batch=2, accumulation_steps=16` 不会让 BatchNorm 的统计行为等价于 `batch=32`。

---

## 替代方案与适用边界

Gradient Accumulation 是低成本显存方案，但不是所有训练问题的最优解。它适合“显存不够，但希望维持较大有效 batch”的场景；如果模型本来能直接跑大 batch，直接使用真大 batch 更简单，统计行为也更接近目标设定。

| 方案 | 解决什么 | 优点 | 局限 |
|---|---|---|---|
| Gradient Accumulation | 显存放不下目标 batch | 实现简单，不需要多卡 | 吞吐未必高，BN 不自然等价 |
| 真大 batch 训练 | 直接扩大单步 batch | 行为最直接，控制简单 | 显存要求高 |
| 数据并行 / 多卡扩展 | 用多张卡分摊 batch | 提升吞吐，扩大 global batch | 需要通信和分布式配置 |
| activation checkpointing | 降低激活显存 | 可训练更深模型或更长序列 | 反向时要重算，速度变慢 |
| mixed precision | 降低显存和提升速度 | 工程收益明显 | 需要处理数值稳定性 |

适用边界可以这样判断：

| 场景 | 是否适合 Gradient Accumulation |
|---|---|
| 单卡显存不足，但目标是更大有效 batch | 适合 |
| 训练 LLM、扩散模型、大图像模型 | 常见适用 |
| 模型大量依赖 BatchNorm，且 micro-batch 很小 | 需要谨慎 |
| 主要瓶颈是训练吞吐 | 不一定适合 |
| batch 极小且梯度噪声很大 | 需要实验验证 |
| 已有多卡资源且通信效率高 | 数据并行可能更直接 |

新手版判断规则：如果模型能直接跑目标 batch，就先用真大 batch；如果显存不够，再用梯度累积；如果梯度累积后速度太慢，再考虑混合精度、activation checkpointing、多卡数据并行。

真实工程中常见组合是：`micro_batch=1 or 2`，`accumulation_steps=8/16/32`，再配合 AMP 和 checkpointing。这样能把训练从“完全跑不起来”变成“可接受速度下稳定运行”。但一旦引入 scheduler、分布式训练、梯度裁剪和 BatchNorm，就要把 update-step、global batch、归一化统计分别处理清楚。

---

## 参考资料

1. [PyTorch: Zeroing out gradients](https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html)  
解释 PyTorch 中梯度默认累积到 `.grad`，以及为什么需要 `zero_grad()` 清空梯度。

2. [PyTorch: Automatic Mixed Precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)  
说明 AMP、`GradScaler`、梯度累积、`unscale_()`、`step()` 的正确使用时机。

3. [PyTorch: BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)  
说明 BatchNorm 使用 mini-batch 统计量，这是梯度累积不能自然等价真大 batch 的关键原因。

4. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)  
BatchNorm 原始论文，用于理解批统计量对训练行为的影响。

5. [torchtune: Memory Optimizations](https://docs.pytorch.org/torchtune/0.5/tutorials/memory_optimizations.html)  
说明 effective batch size、gradient accumulation、显存优化在训练工程中的关系。
