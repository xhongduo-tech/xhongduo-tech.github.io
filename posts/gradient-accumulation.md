## 核心结论

梯度累积（gradient accumulation，白话说就是“把多次小批次算出来的梯度先存起来，攒够了再更新一次参数”）是显存受限时模拟大 batch size 的直接方法。它不改变单次前向和反向所需的激活显存，因此单卡只能放下 `batch_size=4` 时，仍然可以通过多次累积得到等效 `batch_size=32` 的训练行为。

核心公式只有两个：

$$
\text{total\_batch\_size} = \text{batch\_size} \times \text{accumulation\_steps}
$$

$$
g_{\text{effective}} = \frac{1}{K}\sum_{i=1}^{K} g_i
$$

其中 $K=\text{accumulation\_steps}$，$g_i$ 是第 $i$ 个微批次（micro-batch，白话说就是“显存真正一次能吃下去的小批次”）的梯度。

新手版可以直接这样理解：把 8 次 `batch_size=4` 的小训练迭代，当成 1 次“拼接”出的 `batch_size=32`。显存始终只承受 4 个样本，但参数更新时看到的是 32 个样本平均后的梯度。

它的收益是显存占用接近单个微批次，它的代价是一次参数更新前要做更多次前向和反向，所以 wall-clock time（白话说就是“真实训练耗时”）通常更长。它也不是严格无条件等价，遇到 BatchNorm 这类依赖批内统计的模块时，要特别谨慎。

---

## 问题定义与边界

大 batch 的意义不是“样本多看起来高级”，而是梯度更稳定。随机梯度下降每一步只看一个批次，批次越小，梯度噪声越大；批次越大，梯度更接近整批数据的平均方向。

但训练限制往往不在算力，而在显存。一次性放下更大的 batch，需要同时保存更多样本的激活值、梯度和部分中间状态。大模型、长序列、高分辨率输入都会把这个问题放大。

典型场景是：

- 显存最多只能容纳 `batch_size=4`
- 经验上训练希望等效 `batch_size=32`
- 做法不是硬塞 32 个样本，而是每次喂 4 个，累计 8 次后再更新一次参数

这就是梯度累积的边界：它解决的是“更新所依赖的样本数”与“单次显存容量”之间的矛盾，但不解决单个样本本身太大导致的 OOM。如果一个样本就放不下，梯度累积没有帮助，应该考虑缩短序列、降低分辨率、AMP 或 checkpointing。

下面这个表可以先建立直觉：

| 可容纳 batch | 目标等效 batch | accumulation_steps | 是否可行 | 主要代价 |
|---|---:|---:|---|---|
| 4 | 8 | 2 | 可行 | 更新频率减半 |
| 4 | 16 | 4 | 可行 | 前后向次数增多 |
| 4 | 32 | 8 | 可行 | wall-clock 明显上升 |
| 4 | 128 | 32 | 技术上可行 | 吞吐下降，调度更敏感 |
| 1 | 32 | 32 | 可能不稳定 | 微批次过小，统计噪声大 |

玩具例子最适合说明问题：显存像一个一次只能装 4 个苹果的箱子，但你想按 32 个苹果的平均重量来决定运输方案。梯度累积不是造更大的箱子，而是连续称 8 次，每次 4 个，最后把 8 次结果平均后再决策。

真实工程例子是训练语言模型。比如单卡 24GB 显存训练中等规模 Transformer，`seq_len=2048` 时一次只能放 2 到 4 个样本，但为了让学习率调度和优化行为更稳定，工程上常把等效 batch 拉到 32、64 甚至更高，这时梯度累积几乎是默认配置。

---

## 核心机制与推导

标准训练循环里，每个 batch 都会做一次：

1. 前向计算 loss
2. 反向传播得到梯度
3. `optimizer.step()` 更新参数
4. `optimizer.zero_grad()` 清空梯度

梯度累积只改两件事：

1. 不在每个微批次后立刻更新参数
2. 不在每个微批次后立刻清空梯度

也就是把多个微批次的梯度先加到一个“总梯度池”里，攒够之后再统一更新。这个“池”并不是单独的数据结构，通常就是参数对象上的 `.grad` 张量。

设一个大 batch 被拆成 $K$ 个微批次，每个微批次 loss 为 $\mathcal{L}_i$。如果大 batch 的目标 loss 定义为平均值，那么：

$$
\mathcal{L} = \frac{1}{K}\sum_{i=1}^{K}\mathcal{L}_i
$$

对参数 $\theta$ 求梯度：

$$
\nabla_\theta \mathcal{L} = \frac{1}{K}\sum_{i=1}^{K}\nabla_\theta \mathcal{L}_i
$$

所以等效实现有两种常见写法：

| 写法 | 做法 | 结果 |
|---|---|---|
| loss 先除以 `K` | 每次 `loss = loss / K` 再 `backward()` | `.grad` 里直接是平均梯度 |
| 梯度后处理 | 每次直接 `backward()`，更新前手动除以 `K` | 理论等价，但实现更麻烦 |

工程里更常见第一种，因为简单且不容易漏。

这也解释了为什么下面这个式子重要：

```text
total_batch_size = batch_size × accumulation_steps
effective_grad = sum_{i=1}^{accumulation_steps} grad_i / accumulation_steps
```

它的含义不是“训练速度变快”，而是“每次参数更新所参考的样本数变大”。于是两个超参数会被连带影响：

- 学习率：很多经验规则基于总 batch size，而不是微批次大小
- 调度器：scheduler 应该按真实参数更新次数走，而不是按微批次数走

这是新手最容易忽略的点。比如以前每个 batch 更新一次参数，现在改成每 8 个微批次才更新一次，那么同样训练一个 epoch，`optimizer.step()` 的次数会变成原来的 `1/8`。如果 scheduler 还按微批次调用，warmup 和 decay 都会快 8 倍，训练曲线会直接错位。

再强调一次“桶”的直觉：每个小 batch 都往桶里倒一勺梯度，但如果每勺都没先除以 `K`，最后桶里会变成总和而不是平均值，相当于偷偷把梯度放大了 `K` 倍。

---

## 代码实现

下面先给一个可运行的玩具例子，证明“直接用大 batch 算一次”和“拆成多个微批次累积”在平均梯度上是一致的。

```python
import numpy as np

def mse_grad(x, y, w):
    # 线性模型 y_hat = x * w
    y_hat = x * w
    grad = np.mean(2 * x * (y_hat - y))
    return grad

# 8 个样本，模拟目标大 batch
x = np.array([1., 2., 3., 4., 5., 6., 7., 8.])
y = np.array([2., 4., 6., 8., 10., 12., 14., 16.])
w = 0.5

# 一次性大 batch 的梯度
full_grad = mse_grad(x, y, w)

# 拆成 4 个微批次，每个 batch_size=2，累积 4 次
accum_steps = 4
micro_batch_size = 2
accum_grad = 0.0

for i in range(accum_steps):
    start = i * micro_batch_size
    end = start + micro_batch_size
    grad_i = mse_grad(x[start:end], y[start:end], w)
    accum_grad += grad_i / accum_steps  # 等效平均梯度

assert np.allclose(full_grad, accum_grad, atol=1e-10)

lr = 0.1
w_full = w - lr * full_grad
w_accum = w - lr * accum_grad

assert np.allclose(w_full, w_accum, atol=1e-10)
print("full_grad =", full_grad)
print("accum_grad =", accum_grad)
print("same update =", np.allclose(w_full, w_accum))
```

上面这段代码说明一件事：如果 loss 或梯度按 `accumulation_steps` 做了归一化，参数更新结果可以与大 batch 对齐。

训练循环里的标准写法通常如下：

```python
accumulation_steps = 8
optimizer.zero_grad()

for step, batch in enumerate(train_loader):
    loss = model(batch)
    loss = loss / accumulation_steps  # 归一化，让多个小 batch 等效大 batch
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()      # 只在累积满时更新参数
        optimizer.zero_grad() # 更新后再清梯度
        scheduler.step()      # 调度器也按“真实更新次数”前进
```

这段伪码有三个关键点：

1. `loss / accumulation_steps` 不是装饰，它决定梯度量级是否正确。
2. `optimizer.step()` 的频率下降了，所以日志、学习率调度、EMA 更新都要看真实 update step。
3. 一个 epoch 结束时如果最后不足 `accumulation_steps`，需要补一次更新，否则尾部样本的梯度会被丢掉。

真实工程例子可以看大模型训练。假设：

- 单卡显存只能支持 `micro_batch_size=2`
- 多卡数 `world_size=4`
- `accumulation_steps=8`

那么全局等效 batch 是：

$$
\text{global\_batch\_size} = 2 \times 4 \times 8 = 64
$$

这 64 才是你设置学习率经验规则、warmup 步数和吞吐统计时真正应该参考的 batch 大小。

---

## 工程权衡与常见坑

梯度累积不是“白拿大 batch”，它只是把显存压力换成时间和工程复杂度。

最常见的问题如下：

| 问题 | 现象 | 规避 |
|---|---|---|
| wall-clock 变长 | 一次更新前要重复多次前后向 | 接受吞吐下降，结合 AMP 提升速度 |
| scheduler 步频错误 | 学习率变化过快或过慢 | 按 `optimizer.step()` 次数调用 |
| 忘记归一化 loss | 梯度放大，训练发散 | 每个微批次 `loss /= accumulation_steps` |
| 尾批次没更新 | 最后几个样本白算 | epoch 结尾补 `step()` |
| BatchNorm 不稳定 | 验证指标振荡 | 改 LayerNorm/GroupNorm，或降低 momentum |
| 梯度裁剪位置错误 | 裁剪阈值失真 | 在累积完成后、`optimizer.step()` 前裁剪 |

BatchNorm（批归一化，白话说就是“用当前批次的均值和方差来做归一化”）是最典型的坑。它的问题不在梯度累积本身，而在它依赖“当前微批次”的统计量。你把一个大 batch 拆成多个微批次时，梯度可以跨步累加，但 BatchNorm 的运行均值和方差不会跨步合并，因此不再等价于真正的大 batch。

新手常见误区是：我已经把 8 个小 batch 累积成等效 32 了，为什么 CNN 训练还抖得厉害？答案通常是微批次上的 BatchNorm 统计不稳定。比如图像分类里 `batch_size=2` 跑 ResNet，每个微批次统计都很偏，训练和验证曲线会明显振荡。这时常见处理是：

- 把 BatchNorm 换成 LayerNorm 或 GroupNorm
- 冻结 BatchNorm 统计
- 调低 BatchNorm momentum，让运行统计更新更慢

另一个坑是日志解释。你看到 loss 每个微批次都在打印，但参数每 8 步才更新一次，所以“step loss”“global step”“tokens per update”必须区分清楚。很多训练脚本指标对不上，本质上是把 micro-step 当成 update-step 了。

---

## 替代方案与适用边界

如果问题是“总 batch 放不下”，梯度累积很合适；如果问题是“单个样本都放不下”，它就不够了。

常见替代方案如下：

| 方案 | 核心思路 | 适用场景 | 主要代价 |
|---|---|---|---|
| 梯度累积 | 多个微批次后再更新 | 想要更大等效 batch | 时间更长 |
| AMP | 用半精度减少显存 | 大多数 GPU 训练 | 数值稳定性要关注 |
| Gradient Checkpointing | 少存激活，反向时重算 | 深层网络、长序列 | 额外计算开销 |
| 模型拆分/并行 | 把模型分布到多设备 | 单卡放不下模型 | 工程复杂 |
| 缩短输入/缩小模型 | 直接减少计算图规模 | 资源极度受限 | 可能损失效果 |

AMP（自动混合精度，白话说就是“部分张量用更低精度存和算”）和梯度累积经常一起用。前者主要减单次显存，后者主要扩等效 batch，两者是互补关系，不是二选一。

checkpointing 也类似。它通过少存激活、反向时重算来省显存，适合“模型很深”或“序列很长”的场景。相比之下，梯度累积并不减少单步激活峰值。

如果 BatchNorm 让累积出问题，新手视角下可以这样判断：

- 如果模型本来就能用 LayerNorm，优先换 LayerNorm
- 如果是 CNN 且很依赖 BatchNorm，先尝试更大微批次或 GroupNorm
- 如果单卡方案始终不稳定，再考虑多卡分布式训练，而不是死扛超大的 `accumulation_steps`

一个实际边界是：`accumulation_steps` 不能无限增大。理论上能继续累，工程上会遇到这些限制：

- 单次参数更新太慢，训练反馈周期过长
- 调度器和日志粒度过粗
- 梯度变得“过旧”，尤其在强数据增强或动态采样下更明显
- 微批次过小导致归一化层、dropout 统计行为更不稳定

所以它是折中方案，不是通用最优解。

---

## 参考资料

下面这些资料足够覆盖“定义、机制、实现、坑点”四个层面：

| 来源 | 重点 | 作用 |
|---|---|---|
| Hugging Face Accelerate | 梯度累积的定义、直观解释、基本代码 | 适合建立第一层概念 |
| PyTorch Torchtune 文档 | 内存优化视角、与 batch/lr 的关系 | 适合理解训练配置联动 |
| Uplatz 技术分析 | 工程场景、wall-clock、BatchNorm 等坑 | 适合理解真实训练代价 |
| PyTorch 论坛 BatchNorm 讨论 | 微批次统计与 BN 不等价问题 | 适合定位归一化层风险 |

参考列表：

- Hugging Face Accelerate, Gradient Accumulation  
  https://huggingface.co/docs/accelerate/main/usage_guides/gradient_accumulation  
  重点是定义清楚：多个微批次累积梯度，再进行一次优化器更新，直观上等效更大 batch。

- PyTorch Torchtune, Memory Optimizations  
  https://docs.pytorch.org/torchtune/0.3/tutorials/memory_optimizations.html  
  重点是把梯度累积放进“显存优化工具箱”里看，并强调总 batch size 与训练配置的关系。

- Uplatz, Gradient Accumulation: A Comprehensive Technical Guide  
  https://uplatz.com/blog/gradient-accumulation-a-comprehensive-technical-guide-to-training-large-scale-models-on-memory-constrained-hardware/  
  重点是工程权衡，尤其是 wall-clock 开销、微批次设计和实际训练中的不完全等价问题。

- PyTorch Forums, BatchNorm with Gradient Accumulation  
  可在 PyTorch 论坛检索相关讨论  
  重点是说明 BatchNorm 的运行统计基于每个微批次，不会因为梯度累积自动变成大 batch 统计。
