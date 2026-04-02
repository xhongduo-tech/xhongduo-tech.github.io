## 核心结论

8-bit 优化器指的是把优化器内部保存的状态改成 8-bit 整数存储，而不是把整个训练都改成 8-bit。对 Adam/AdamW 而言，最占显存的不是参数本身，而是一阶矩 $m_t$ 和二阶矩 $v_t$，也就是“优化器记住过去梯度历史的两份缓存”。bitsandbytes 的 `Adam8bit` / `AdamW8bit` 做的事情，是把这两份状态按 block 分组后量化成 INT8 存储，更新时再临时反量化到浮点数参与计算。

它的核心收益非常直接：优化器状态显存大约下降 75%。如果用最常见的 FP32 Adam，单个参数通常需要两份 FP32 状态；改成 8-bit 后，这两份状态从每个元素 4 字节变成 1 字节，外加很小的缩放因子开销，因此整体近似从 8 字节降到 2 字节。

| 项目 | FP32 Adam 状态 | 8-bit Adam 状态 | 结论 |
|---|---:|---:|---|
| 每个参数的 $m,v$ 存储 | 8 字节 | 约 2 字节 | 约省 75% |
| 7B 模型优化器状态 | 约 56 GB | 约 14 GB | 节省约 42 GB |
| 更新精度 | 高 | 接近 FP32 | 通常精度差很小 |

玩具例子可以先这样理解：把优化器里连续 4096 个数字看成一组，先找这一组里绝对值最大的数，再把整组都按这个最大值缩放到 $[-127,127]$，保存成整数。需要更新时，再把它们按比例放大回近似浮点数，算完 Adam 更新，再压缩回去。它不是“直接拿整数做 Adam”，而是“整数存、浮点算”。

真实工程例子是 QLoRA。QLoRA 把基础模型权重量化到 4-bit，再配合 8-bit 优化器训练 LoRA 参数，使 65B 级模型在单张 48GB GPU 上微调成为可能。这里 8-bit 优化器解决的是“状态显存太大”的问题，不是所有显存问题。

---

## 问题定义与边界

问题先要定义清楚：8-bit 优化器优化的是 optimizer state memory，也就是优化器状态显存，而不是参数显存、激活显存、KV cache 显存。很多初学者会把“训练显存”混成一个总数，但工程上至少要拆成三类：

| 显存来源 | 含义 | 8-bit 优化器是否直接解决 |
|---|---|---|
| 模型参数 | 权重本身占用的显存 | 部分无关 |
| 激活 | 前向中间结果，反向要回传 | 基本不解决 |
| 优化器状态 | Adam 的 $m,v$ 等缓存 | 直接解决 |

Adam 的问题在于它比 SGD 多存两份历史统计量。这里“一阶矩”就是梯度的指数滑动平均，可以白话理解成“近期梯度方向的平滑记忆”；“二阶矩”就是梯度平方的指数滑动平均，可以白话理解成“近期梯度波动大小的记忆”。这两份状态通常都用 FP32 保存，所以模型一大，显存就被迅速吃满。

边界也要说清楚。8-bit 优化器并不等于“任何场景都省很多”。如果你的训练主要卡在激活，比如长序列 Transformer、超大 batch、重视觉 backbone，那么只压缩优化器状态可能还不够。相反，如果你是在大语言模型微调里卡在 Adam 状态，收益就很明显。

bitsandbytes 还有一个重要边界条件：不是所有 tensor 都值得量化。小 tensor 元素少，量化后的收益很有限，但误差相对更敏感，所以库里默认只对足够大的参数块启用 8-bit。常见参数是 `min_8bit_size=4096`，意思是元素数小于 4096 的 tensor 默认继续用 32-bit 状态。

| tensor 大小 | 默认策略 | 原因 |
|---|---|---|
| $\ge 4096$ 元素 | 优先 8-bit | 显存收益明显，统计更稳定 |
| $< 4096$ 元素 | 保留 32-bit | 节省有限，误差更敏感 |
| Embedding 等特殊层 | 常保守处理 | 梯度分布可能不均匀 |

一个面向新手的边界判断方法是：如果你把优化器关掉，显存一下子降很多，那 8-bit 优化器可能有价值；如果你把 batch size 从 8 改到 2 才能跑通，那更可能是激活显存问题。

---

## 核心机制与推导

bitsandbytes 的关键技术是 block-wise dynamic quantization。这里“block-wise”是“按小块分别量化”，白话就是“不拿整张大表只用一个缩放范围，而是每一小段自己找尺度”；“dynamic”是“缩放范围不是固定写死，而是根据当前 block 的数值动态计算”。

设某个 block 记为 $B_i$，其中每个元素是优化器状态里的一个值，例如某段一阶矩。先计算该 block 的绝对值最大值：

$$
absmax_i = \max(|B_i|)
$$

然后把 block 内每个值映射到 INT8 区间：

$$
q_j = \operatorname{round}\left(\frac{B_j}{absmax_i} \cdot 127\right)
$$

反量化时再恢复近似值：

$$
B_j \approx q_j \cdot \frac{absmax_i}{127}
$$

这三步的直觉很简单：先归一化，再压到整数范围，使用时再按比例放回去。

为什么要按 block 做，而不是整个 tensor 用一个 `absmax`？因为优化器状态往往分布不均匀。如果整个 tensor 里只有少数位置特别大，一个全局最大值会把大多数较小数值压得很粗，量化误差会上升。按 block 分开做，局部尺度更贴合数据分布。

Adam 的核心更新公式本身并没有变：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

其中 $g_t$ 是当前梯度，$\beta_1,\beta_2$ 是指数滑动平均系数。工程上的关键点是：梯度统计和更新逻辑仍然在 FP32 精度下完成。8-bit 优化器量化的是“存储状态”，不是“把 Adam 公式改成整数数学”。

完整流程可以写成：

| 步骤 | 做什么 | 精度 |
|---|---|---|
| 1 | 从显存中读取 INT8 状态 $m,v$ | INT8 |
| 2 | 按 block scale 反量化 | FP32 近似值 |
| 3 | 用当前梯度执行 Adam 更新 | FP32 |
| 4 | 把新状态重新按 block 量化 | INT8 |
| 5 | 写回显存 | INT8 |

玩具例子最容易说明这个过程。假设某个 block 的最大绝对值是 $0.4$，其中一个值是 $0.32$。先归一化：

$$
0.32 / 0.4 = 0.8
$$

再映射到 INT8：

$$
0.8 \times 127 = 101.6 \approx 102
$$

所以保存时存成整数 `102`。使用时再恢复：

$$
102 \times 0.4 / 127 \approx 0.321
$$

恢复值和原值 $0.32$ 很接近。接着在 FP32 中参与 $m_t, v_t$ 更新，更新后再按新的 block 范围量化保存。

这也是为什么 8-bit Adam 和 FP32 Adam 的最终效果通常很接近。真正影响收敛的主要计算仍然在浮点域里完成，量化误差集中在“状态缓存的存储与恢复”环节，而 block-wise 设计又把这种误差限制在可控范围内。

真实工程例子可以看大模型微调。以 7B 级模型为例，Adam 的状态如果全用 FP32，大约要 56GB；压成 8-bit 后约 14GB。对 QLoRA 这类方案来说，基础权重已经用 NF4 量化，LoRA 可训练参数量不大，但整体仍需要优化器状态。如果没有 8-bit optimizer，单卡 48GB 经常会直接爆显存；有了它，训练预算才落到单卡可承受区间。

---

## 代码实现

对使用者来说，8-bit 优化器的最大优点是接入成本低。训练循环通常不用改，只需要把 `torch.optim.AdamW` 换成 `bitsandbytes` 的对应实现。内部的量化、反量化、状态管理都由库完成。

下面先用一个纯 Python 玩具实现说明原理。这个例子不是高性能实现，只是把“按 block 量化、反量化、更新”的流程写清楚。

```python
from math import isclose

def quantize_blockwise(values, block_size=4):
    q_values = []
    scales = []
    for i in range(0, len(values), block_size):
        block = values[i:i + block_size]
        absmax = max(abs(x) for x in block) if block else 0.0
        scale = absmax / 127.0 if absmax != 0 else 1.0
        q_block = [int(round(x / scale)) if absmax != 0 else 0 for x in block]
        q_values.extend(q_block)
        scales.append(scale)
    return q_values, scales

def dequantize_blockwise(q_values, scales, block_size=4):
    values = []
    for block_id, scale in enumerate(scales):
        start = block_id * block_size
        block = q_values[start:start + block_size]
        values.extend([q * scale for q in block])
    return values

def adam_step(param, grad, m, v, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
    new_m = beta1 * m + (1 - beta1) * grad
    new_v = beta2 * v + (1 - beta2) * (grad ** 2)
    update = lr * new_m / ((new_v ** 0.5) + eps)
    new_param = param - update
    return new_param, new_m, new_v

# 玩具状态
m_states = [0.32, -0.10, 0.05, 0.40]
q_m, scales = quantize_blockwise(m_states, block_size=4)
restored_m = dequantize_blockwise(q_m, scales, block_size=4)

# 量化后恢复值应接近原值
assert isclose(restored_m[0], 0.32, rel_tol=0.02, abs_tol=0.01)
assert q_m[0] == 102  # 0.32 / (0.4/127) ≈ 101.6 -> 102

# 模拟一次 Adam 更新：先反量化，再在浮点域更新
param, grad = 1.0, 0.2
new_param, new_m, new_v = adam_step(param, grad, restored_m[0], 0.01)

# 更新后参数应下降，新的状态可再量化存回去
assert new_param < param
q_new_m, new_scales = quantize_blockwise([new_m, -0.10, 0.05, 0.40], block_size=4)
assert len(q_new_m) == 4
assert len(new_scales) == 1
```

真正使用 bitsandbytes 时，代码要简单得多：

```python
import torch
import bitsandbytes as bnb

model = torch.nn.Linear(1024, 2).cuda()

optimizer = bnb.optim.Adam8bit(
    model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.999),
    min_8bit_size=4096,
)

x = torch.randn(8, 1024, device="cuda")
y = torch.randint(0, 2, (8,), device="cuda")

criterion = torch.nn.CrossEntropyLoss()

for _ in range(3):
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()  # 内部自动执行 dequantize -> update -> requantize
```

这里 `min_8bit_size=4096` 很关键。它控制“多大的 tensor 才启用 8-bit 状态”。对初级工程师来说，可以先保持默认值，不要为了省一点显存盲目调小。因为小 tensor 的量化收益低，但不稳定风险更高。

如果你原来写的是：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
```

那么很多情况下只要替换成：

```python
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-5)
```

其余训练逻辑不动即可。这也是它在微调场景中很受欢迎的原因。

---

## 工程权衡与常见坑

8-bit 优化器并不是“无脑开启”的开关，而是一个明确针对状态显存的工程折中。折中点主要在三处：数值稳定性、收益分布、系统瓶颈。

第一类坑是小 tensor。小 tensor 即使量化，省下的字节数也有限，但每个 block 的尺度更容易被极端值影响，量化误差反而更突出。所以默认的 `min_8bit_size=4096` 是一个保守且实用的经验值。很多训练不稳的问题，都是因为把这个阈值调得过低。

第二类坑是 Embedding。Embedding 可以白话理解成“把 token id 映射成向量的查表层”。它的梯度分布往往不均匀，热门 token 和冷门 token 的更新频率差别很大。对这类层，bitsandbytes 提供了 Stable Embedding 等保守设计，本质上是避免在最容易出问题的地方过度量化。

第三类坑是误解显存来源。如果你的任务是长上下文训练，激活显存远大于优化器状态，那么只换 8-bit optimizer 仍然可能 OOM。此时更有效的是 activation checkpointing、缩短序列长度、减 batch、ZeRO 分片等手段。

| 常见坑 | 现象 | 原因 | 建议 |
|---|---|---|---|
| `min_8bit_size` 设太小 | loss 抖动、训练不稳 | 小 tensor 量化误差更敏感 | 先用默认 4096 |
| Embedding 也强制量化 | 收敛变差 | 梯度分布不均匀 | 对特殊层保守处理 |
| 以为能解决全部 OOM | 仍爆显存 | 激活才是主瓶颈 | 配合 checkpoint/ZeRO |
| 直接在 INT8 上做更新 | 数值偏差明显 | Adam 公式不该在整数域执行 | 必须 dequantize 后更新 |
| 盲目追求极限压缩 | 调参困难 | 收益和风险不对称 | 先保稳定，再压显存 |

这里要特别强调一件事：不要把“状态存成 INT8”和“更新过程用 INT8 计算”混为一谈。正确流程必须是 `dequantize -> update -> requantize`。如果有人为了省一次临时浮点展开，试图直接在量化整数上更新，数值行为就已经不是 Adam 了。

真实工程里，一个典型场景是微调带大词表的语言模型。你可能已经把主体线性层都交给 `Adam8bit`，但词嵌入层仍建议谨慎处理。如果 batch 中 token 分布长尾明显，Embedding 状态往往比普通 MLP 层更敏感。这时工程目标不是“所有层都 8-bit”，而是“整体显存和稳定性最优”。

---

## 替代方案与适用边界

8-bit 优化器解决的是 Adam 状态过大问题，但训练显存优化通常要组合拳。常见方案可以按“它主要压哪一部分显存”来分类：

| 方法 | 主要压缩对象 | 优点 | 代价 | 适用边界 |
|---|---|---|---|---|
| 8-bit optimizer | 优化器状态 | 接入简单，省约 75% 状态显存 | 仍需浮点更新 | 状态显存是主瓶颈 |
| NF4 / 4-bit 权重量化 | 模型参数 | 参数显存大幅下降 | 有量化近似误差 | 微调大模型 |
| ZeRO | 参数/梯度/状态分片 | 多卡下节省明显 | 通信复杂度上升 | 多卡训练 |
| Activation Checkpointing | 激活 | 显著降激活显存 | 计算更慢 | 长序列、大 batch |
| CPU Offloading | GPU 显存总占用 | 单卡也能撑更大模型 | PCIe 传输慢 | 显存极紧但可接受降速 |

8-bit optimizer 和 NF4 并不冲突，反而经常一起出现。前者压优化器状态，后者压模型权重。QLoRA 就是典型组合：基础模型权重用 NF4 存储，训练只针对 LoRA 适配器进行，优化器状态再用 8-bit 保存。这样一来，参数、状态两个大头都被压下去了。

但如果你的瓶颈是激活显存，结论就不同。比如超长上下文监督微调，即使参数和优化器都压了，反向传播中间激活仍可能吃掉大部分显存。这时 8-bit optimizer 不是主解法，而是辅助解法。

一个面向初学者的判断表可以这样记：

| 你当前最缺什么 | 优先考虑 |
|---|---|
| Adam 状态太大 | 8-bit optimizer |
| 模型参数装不下 | 4-bit / 8-bit 权重量化 |
| 长序列反向爆显存 | activation checkpointing |
| 单卡不够，多卡可用 | ZeRO |
| GPU 显存太小但能接受降速 | CPU offloading |

所以，8-bit 优化器的适用边界不是“所有训练都该开”，而是“当优化器状态已经成为主要显存成本时，它是收益极高、代价相对小的方案”。

---

## 参考资料

- Dettmers 等，《8-bit Optimizers via Block-wise Quantization》，OpenReview。核心论文，定义了 block-wise quantization 与 8-bit optimizer 的实验结果。
- Hugging Face bitsandbytes Optimizers 文档。官方使用说明，包含 `Adam8bit`、`AdamW8bit`、`min_8bit_size`、Stable Embedding 等实践细节。
- Hugging Face bitsandbytes Explanations 文档。解释 8-bit optimizer 的设计动机、数值稳定性与适用场景。
- QLoRA / FinLoRA 相关资料。说明 4-bit 权重量化与 8-bit optimizer 的组合如何支持单卡大模型微调。
- EngineersOfAI 显存对比整理。用于理解 7B 级模型下 FP32 Adam 与 8-bit optimizer 的状态显存量级差异。
