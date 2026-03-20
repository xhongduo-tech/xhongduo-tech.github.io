## 核心结论

量化感知训练（Quantization Aware Training，QAT，意思是“训练时就把低精度误差模拟进去”）的本质，不是先把模型变成 INT8/INT4 再训练，而是在前向计算里插入伪量化节点，用浮点数模拟离散化，再在反向传播里用梯度直通估计器（Straight-Through Estimator，STE，意思是“把本来不可导的量化操作近似当成可导”）继续更新 FP32 参数。

这和训练后量化（Post-Training Quantization，PTQ，意思是“模型训完后再做四舍五入压缩”）的差别，决定了精度上限。PTQ 的量化误差不会参与原始 loss 优化，所以一旦某一层对误差敏感，尤其是 Hessian（损失曲率矩阵，白话讲就是“参数微小变化会把 loss 放大多少”）较大的方向被粗暴舍入，精度就会明显下降。QAT 则把这种误差提前暴露给优化器，让参数向“对量化噪声更稳”的区域收敛。

核心公式是：

$$
\text{FakeQuant}(x)=\text{clamp}\left(\text{round}\left(\frac{x}{\text{scale}}+zp\right), q_{\min}, q_{\max}\right)\cdot \text{scale}
$$

其中 `scale` 是缩放因子，`zp` 是零点，`clamp` 表示截断到量化范围。前向看起来像量化，反向常用 STE 近似为：

$$
\frac{\partial \text{FakeQuant}(x)}{\partial x}\approx 1
$$

所以“低比特前向 + 高精度反向”成为可能。即便是 4-bit，只要前向先量化/反量化、反向仍保留 FP32 梯度，模型也可能接近原始 FP32 精度。

一个新手最常见的真实流程是：先把 PyTorch 里的 FP32 模型训到收敛，再继续做一段 QAT，把权重 `W` 在前向中经过 FakeQuant 模拟 INT8，loss 仍按原任务计算，最后导出真正的 INT8 模型。这样做的目标不是“训练一个更强模型”，而是“训练一个对低精度部署更不敏感的模型”。

---

## 问题定义与边界

训练时量化要解决的核心问题是：部署需要低比特，但模型训练通常发生在高精度浮点空间，两者之间存在分布断层。PTQ 只在训练结束后做离散化，等于假设模型参数附近的小扰动都不重要；这个假设在很多层并不成立。

可以把 PTQ 和 QAT 的区别概括成下面这张表：

| 维度 | PTQ | QAT |
|---|---|---|
| 执行时机 | 训练后 | 训练中 |
| 前向是否模拟量化误差 | 否 | 是 |
| 反向是否可导 | 不涉及训练 | 依赖 STE 近似可导 |
| 误差控制点 | scale、group、校准集 | scale、group、训练过程共同优化 |
| 对敏感层的适应能力 | 弱 | 强 |
| 训练成本 | 低 | 较高 |
| 典型场景 | 快速部署 | 高精度低比特部署 |

对白话解释：PTQ 像“考试交卷后再改字迹”，内容已经定了；QAT 像“平时练习时就戴着限制条件做题”，模型会主动学会适应误差。

这里要明确边界。第一，QAT 不等于所有层都必须量化。Embedding、输出头、少数归一化层，有时保留高精度更稳。第二，INT8 和 INT4 的难度不同。INT8 的动态范围仍相对宽，QAT 常能逼近 FP32；INT4 尤其是权重量化时，误差明显更大，通常需要更细的分组、特殊码本，或者只量化权重不量化激活。第三，激活顺序（act-order，意思是“先按激活重要性排序再量化”）和块大小（blocksize，意思是“多少个参数共享一个 scale”）会直接影响误差分布，不能一刀切。

玩具例子：一层只有两个权重，原本是 `[0.27, 1.92]`。如果 PTQ 直接量化到很粗的 4-bit 网格，`0.27` 可能被映射成 `0.246`，误差是 `0.024`。如果这个位置刚好对应敏感方向，loss 就会抬升。QAT 会在训练阶段不断看到这个 `0.024` 的扰动，逐步把参数推向更稳定的位置，比如 `0.245` 或 `0.31`，使量化后输出更可控。

---

## 核心机制与推导

QAT 的第一层机制是 FakeQuant。它在前向中做三件事：缩放到整数域、四舍五入、截断，再映射回浮点域。这样算子既保留了“量化后的数值格点”，又能继续跑普通浮点图。

但 `round` 和 `clamp` 天生不可导或几乎处处梯度为 0。如果直接把它们放进训练图，优化器几乎收不到有效梯度。所以引入 STE。它的含义不是“量化真的可导”，而是工程上强行规定：在反向传播时，把伪量化节点近似看成恒等映射。可以用一句话记住：

前向按量化后的值算，反向按原始值传梯度。

文字图示如下：

`x -> round/clamp -> x_q -> loss`
反向时不是用 `d round / dx = 0`，而是近似为 `dx_q/dx ≈ 1`

这等价于告诉优化器：虽然部署时参数会上格点，但你先按“格点附近的连续空间”去更新。

再看 GPTQ。GPTQ 是一种后处理量化方法，但它解释了为什么“不同权重的量化误差不能等价看待”。如果损失在参数附近可以局部二阶近似：

$$
\Delta L \approx \frac{1}{2}\Delta w^T H \Delta w
$$

其中 $H$ 是 Hessian。若某个方向曲率大，说明这个方向上的小误差也会造成大损失。GPTQ 的做法是：逐列量化当前权重，并把该列产生的误差通过 Hessian 的逆传播到后续列，近似公式可写成：

$$
\Delta W[:, j] \mathrel{-}= e_i \cdot H^{-1}[j, i]
$$

白话解释是：第 `i` 列已经被量化出误差 `e_i`，那后面的列不要装作没发生，而要用二阶信息把这部分误差“摊回去”。

继续看前面的最小例子。某权重 `0.27` 被 NF4（NormalFloat4，意思是“针对近似正态分布设计的 4-bit 码本”）映射到 `0.246`，误差约为 `0.024`。如果该分量对应的 Hessian 对角元素很小，说明这个位置对 loss 不敏感，那么这个舍入代价低，可以接受。若对角元素很大，就意味着误差危险，QAT 会倾向于在训练中把参数移动到更适合的格点附近；GPTQ 则会在后续列上做补偿，减少总输出漂移。

NF4 的统计意义在于：它不是均匀整数格点，而是假设权重近似服从零均值正态分布，把更多离散码位分配到高密度区域，也就是靠近 0 的位置。这比线性 4-bit 对 LLM 权重更友好，因为大模型里大量权重本来就聚集在零附近。

---

## 代码实现

下面给出一个最小可运行的 Python 例子，演示手写 FakeQuant 和 STE。这个例子不依赖 PyTorch，重点是把机制写清楚。

```python
import math

def fake_quantize(x, scale, qmin, qmax, zp=0):
    q = round(x / scale + zp)
    q = max(qmin, min(qmax, q))
    return (q - zp) * scale

def ste_fake_quant(x, scale, qmin, qmax, zp=0):
    # 前向返回量化值；反向在深度学习框架里近似把梯度直接传给 x
    return fake_quantize(x, scale, qmin, qmax, zp)

# toy example: INT8 symmetric quant
scale = 0.1
x = 0.27
x_q = fake_quantize(x, scale=scale, qmin=-128, qmax=127)

assert abs(x_q - 0.3) < 1e-9
assert abs(x - x_q) == 0.03

# NF4-like toy mapping example,直接写死一个码表近邻
nf4_codebook = [-1.0, -0.696, -0.525, -0.394, -0.284, -0.185, -0.091, 0.0,
                0.079, 0.160, 0.246, 0.337, 0.440, 0.563, 0.723, 1.0]

def nearest_codebook_value(x, codebook):
    return min(codebook, key=lambda c: abs(c - x))

w = 0.27
w_q = nearest_codebook_value(w, nf4_codebook)
err = abs(w - w_q)

assert abs(w_q - 0.246) < 1e-9
assert abs(err - 0.024) < 1e-9
print("ok")
```

如果用 PyTorch，最常见的写法是保留 FP32 参数，在 `forward` 里做 fake quant。伪代码如下：

```python
import torch

def fake_quant_ste(x, scale, qmin=-127, qmax=127):
    x_q = torch.clamp(torch.round(x / scale), qmin, qmax) * scale
    return x + (x_q - x).detach()

W = torch.nn.Parameter(torch.randn(128, 128))
x = torch.randn(32, 128)

scale = W.detach().abs().max() / 127
W_q = fake_quant_ste(W, scale)

y = x @ W_q
loss = y.pow(2).mean()
loss.backward()
```

这里 `x + (x_q - x).detach()` 是常见 STE 技巧。前向结果等于 `x_q`，但反向梯度对 `x` 近似恒等传递。

真实工程例子通常更复杂。比如在 LLM 微调里，主权重加载为 4-bit NF4，前向通过 bitsandbytes 做块量化与反量化；LoRA 适配器保持 FP16/BF16 可训练，基础模型冻结。这不是完整 QAT，但保留了“低比特主权重 + 高精度可训练路径”的思想，工程收益很高。

再用表格总结一个典型流程：

| 步骤 | 输入 dtype | 计算 | 输出 dtype |
|---|---|---|---|
| 原始权重存储 | FP32/BF16 | 参数更新 | FP32/BF16 |
| 前向伪量化 | FP32/BF16 | round + clamp + dequant | FP32 |
| 主干矩阵乘 | FP32 激活 + 伪量化权重 | 常规 GEMM | FP32/BF16 |
| 反向传播 | FP32/BF16 | 用 STE 传梯度 | FP32/BF16 |
| 最终导出 | FP32 权重 | 真量化打包 | INT8/INT4 |

PyTorch 已有 `observer` 和 `FakeQuantize` 机制，适合标准 INT8 QAT。NF4、双重量化、4-bit 训练路径则通常依赖 bitsandbytes 等库。

---

## 工程权衡与常见坑

最常见的错误，是把 `torch.round` 直接塞进训练图，却不做 STE。这样梯度几乎处处为 0，结果不是“训练得慢”，而是“几乎不更新”。第二类错误，是把所有层统一按同一 group size、同一 scale 策略量化，忽略了不同层对误差的敏感性。第三类错误，是只看权重量化误差大小，不看误差是否落在高 Hessian 方向。

下面是高频坑位表：

| 问题 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| 没有 STE | loss 不下降 | round/clamp 梯度消失 | 用自定义 fake quant 或框架内置 QAT |
| scale 统计太粗 | 近零权重偏移严重 | 少量离群值拉大范围 | 改用 per-channel 或更小 blocksize |
| 敏感层直接 PTQ | 精度突然崩 | 高 Hessian 方向被粗暴舍入 | 这些层优先 QAT 或保留高精度 |
| act-order 未调 | 某些 token 输出异常 | 重要激活先后顺序影响误差传播 | 对大模型量化时启用重要性排序 |
| 双重量化过强 | 收敛波动大 | scale 本身又被粗量化 | 先验证 scale 分布，再决定是否启用 |
| 激活也做 4-bit | 训练不稳 | 激活动态范围远比权重复杂 | 优先只量化权重，激活保留更高精度 |

一个实用检查方法，是在训练前后统计 fake quant 误差分布，比如看每层的 $\|W-W_q\|_2$、最大误差、接近零区域的偏移比例。若误差主要集中在少数层，就不要继续全模型一刀切，而应回退到混合精度策略。

还有一个常被忽略的点：QAT 会改变优化地形。因为前向里的“噪声”不是随机噪声，而是由量化格点决定的结构化噪声，所以学习率、warmup、校准区间都可能需要重调。直接把原 FP32 训练超参数搬过来，经常不是最优。

---

## 替代方案与适用边界

如果你的目标只是“尽快把模型压到能跑”，而不是极限保精度，PTQ 仍然是最低成本方案。若有少量训练数据但不想全量 QAT，可以做 QAT-lite：先 PTQ，再用少量数据做几轮量化感知微调，通常已经能修复一部分精度损失。

如果没有训练数据，GPTQ 这类后处理方法更合适。它利用 Hessian 近似做逐层或逐列误差补偿，不依赖完整训练流程，部署友好，但它不能像 QAT 一样真正“重塑参数分布”。

如果有数据但算力有限，QLoRA 往往是现实里更常见的方案。QLoRA 的核心是：主模型权重用 NF4 存储并冻结，只训练小规模 LoRA 适配器。这样显存主要花在适配器和优化器状态上，而不是全量模型参数。对单卡环境、有限预算、需要尽快做领域微调的团队，这通常比完整 QAT 更实用。

可以用下表快速选型：

| 方案 | 是否需要训练数据 | 算力成本 | 精度上限 | 适合场景 |
|---|---|---|---|---|
| PTQ | 否或少量校准集 | 最低 | 中 | 快速部署 |
| GPTQ | 否或极少量样本 | 低 | 中高 | 无法训练但要压到 4-bit |
| QAT | 是 | 高 | 最高 | 对精度要求高的 INT8/INT4 部署 |
| QLoRA | 是 | 中低 | 高 | 单卡/低显存微调大模型 |

真实工程里常见的组合不是“只选一个”，而是分阶段做。比如先用 GPTQ 验证 4-bit 可行性，再对关键层做 QAT；或者先用 QLoRA 完成业务微调，再评估是否值得为最终部署补一轮全模型 QAT。

边界也要说清楚：当目标硬件根本不支持高效 INT4 推理时，训练侧做再复杂的 QAT 也没有部署收益；当任务极端依赖长尾数值稳定性时，输出头或部分激活路径保留更高精度往往比盲目全量化更有效。

---

## 参考资料

| 来源 | 聚焦内容 | 阅读顺序 |
|---|---|---|
| NVIDIA TensorRT QAT Blog | QAT、FakeQuant、STE 的工程解释 | 1 |
| PyTorch Quantization 文档与博客 | `FakeQuantize`、observer、训练图里的量化机制 | 2 |
| GPTQ 论文与实现资料 | Hessian 加权误差补偿、逐列量化 | 3 |
| QLoRA 论文与 bitsandbytes 资料 | NF4、双重量化、LoRA 微调路径 | 4 |
| sanowl 的 QLoRA 说明页 | 面向工程实践的 QLoRA 直观梳理 | 5 |

- NVIDIA: https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/
- PyTorch: https://docs.pytorch.org/blog/quantization-aware-training/
- GPTQ paper: https://arxiv.org/abs/2210.17323
- QLoRA paper: https://arxiv.org/abs/2305.14314
- QLoRA explanation: https://sanowl.github.io/qlora.html
