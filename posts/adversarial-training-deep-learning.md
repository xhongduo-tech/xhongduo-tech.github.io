## 核心结论

对抗训练是一类直接针对“最坏情况输入扰动”优化的训练方法。白话说，它不是只让模型在正常样本上答对题，而是要求模型在“被人故意轻微篡改”的样本上也尽量不出错。

它的核心目标通常写成：

$$
\min_\theta \mathbb{E}_{(x,y)}\Big[\max_{\delta \in S}\mathcal{L}(f_\theta(x+\delta), y)\Big]
$$

其中，$\theta$ 是模型参数，$\delta$ 是加到输入上的扰动，$S$ 是扰动允许出现的范围。这个式子的意思是：内层先找最容易让模型犯错的扰动，外层再更新参数，让这种最坏扰动造成的损失尽量小。

结论可以先记住三条：

| 问题 | 结论 |
|---|---|
| 对抗训练在做什么 | 在训练时主动生成对抗样本，再用这些样本更新模型 |
| 为什么有效 | 它把训练目标从“平均样本正确”改成“最坏小扰动下也尽量正确” |
| 代价是什么 | 训练时间、显存、实现复杂度都会明显上升 |

一个玩具例子可以直观看清楚。设某个像素值为 $x=0.3$，损失对它的梯度是 $-0.2$，扰动上限 $\varepsilon=0.05$。FGSM 会构造：

$$
x_{adv}=x+\varepsilon \cdot \mathrm{sign}(\nabla_x\mathcal{L})=0.3+0.05\cdot(-1)=0.25
$$

像素只从 `0.30` 变成 `0.25`，人眼几乎看不出来，但模型可能因此把“猫”判成“狗”。对抗训练就是不断把这类样本喂给模型，让模型学会忽略这种危险但微小的变化。

---

## 问题定义与边界

先把术语定清楚。

对抗样本，指的是“看起来和原样本几乎一样，但能让模型更容易出错的输入”。鲁棒性，指的是“输入发生小范围变化时，模型预测仍然稳定的能力”。

最常见的约束是 $L_\infty$ 扰动约束：

$$
S=\{\delta:\|\delta\|_\infty \le \varepsilon\}
$$

白话说，$L_\infty$ 约束就是“每一个输入维度最多只能改动 $\varepsilon$ 这么多”。对图像任务，它常表示每个像素最多改一点点；对文本或离散符号任务，这种连续扰动定义就不一定直接适用。

自然训练和对抗训练的区别可以压缩成下面这张表：

| 维度 | 自然训练 | 对抗训练 |
|---|---|---|
| 输入 | 原始样本 $x$ | 原始样本 $x$ 加上扰动后的 $x+\delta$ |
| 优化目标 | 降低平均训练损失 | 降低最坏扰动下的损失 |
| 假设 | 训练分布和测试分布接近 | 测试时可能存在恶意或极端扰动 |
| 优势 | 快、简单、自然精度高 | 鲁棒性强，抗攻击能力更好 |
| 代价 | 训练成本低 | 训练成本高，调参难 |

这里要明确边界。

第一，对抗训练优化的是“指定威胁模型”下的鲁棒性。威胁模型，白话说，就是“你假设攻击者能怎么改输入”。如果你只按 $L_\infty$ 小扰动训练，它并不自动保证对旋转、裁剪、遮挡、语义改写同样鲁棒。

第二，对抗训练通常会引入自然精度和鲁棒精度的张力。自然精度，指干净样本上的准确率；鲁棒精度，指对抗样本上的准确率。很多模型会出现“鲁棒性上升，但干净样本精度略降”的现象，因为决策边界被推得更保守。

第三，它更适合高风险预测场景，而不是所有任务都必须使用。比如自动驾驶视觉、语音识别、金融风控、内容安全等系统，一次小扰动造成的错误代价很高，这时鲁棒性收益更值得买单。

---

## 核心机制与推导

对抗训练之所以成立，是因为它把“找最坏样本”和“训练模型”拆成了两个嵌套优化问题。

### 1. min-max 结构

$$
\min_\theta \mathbb{E}_{(x,y)}\Big[\max_{\delta\in S}\mathcal{L}(f_\theta(x+\delta), y)\Big]
$$

内层 $\max$ 是攻击过程：在允许的扰动集合 $S$ 内，找让损失最大的 $\delta$。外层 $\min$ 是训练过程：更新模型参数，让这些最坏扰动也难以造成大损失。

由于内层一般没有解析解，所以工程里常用近似算法。

### 2. FGSM：一步近似

FGSM，全称 Fast Gradient Sign Method，白话说就是“只走一步的梯度攻击”。公式是：

$$
x^{adv}=x+\varepsilon \cdot \mathrm{sign}(\nabla_x \mathcal{L}(f_\theta(x), y))
$$

它的直觉很直接：如果损失对输入某一维的梯度是正，说明把这一维往大调会让损失变大；如果是负，就往小调。`sign` 只取方向，不管具体大小，因此实现简单、速度快。

玩具例子可以再看一次。若二维输入 $x=[0.3, 0.7]$，梯度为 $[-0.2, 1.4]$，$\varepsilon=0.05$，则：

$$
\mathrm{sign}(\nabla_x\mathcal{L})=[-1, 1]
$$

$$
x^{adv}=[0.25, 0.75]
$$

模型看到的只是很小的改动，但损失沿着上升方向被推动了。

### 3. PGD：多步逼近最坏扰动

PGD，Projected Gradient Descent，在攻击视角里通常是“投影梯度上升”。白话说，就是“不是只走一步，而是走很多小步，每一步后都拉回允许扰动范围”。

更新形式常写为：

$$
x_{t+1}^{adv}=\Pi_{B_\varepsilon(x)}\Big(x_t^{adv}+\alpha \cdot \mathrm{sign}(\nabla_{x_t^{adv}}\mathcal{L}(f_\theta(x_t^{adv}),y))\Big)
$$

其中 $\Pi_{B_\varepsilon(x)}$ 是投影操作，意思是把结果裁回到以原始样本 $x$ 为中心、半径为 $\varepsilon$ 的合法区域内。

FGSM 可以看成 PGD 只走一步的特例。PGD 更接近真正的 worst-case，所以鲁棒训练更强，但成本更高。

### 4. TRADES：把自然精度和鲁棒性显式分开

TRADES 的关键点不是只盯着“分类标签对不对”，而是约束“原样本预测分布”和“对抗样本预测分布”不要差太多。KL 散度，白话说，就是“两个概率分布相差多大”的度量。

TRADES 目标常写为：

$$
\mathcal{L}_{TRADES}
=
\mathcal{L}_{CE}(f_\theta(x), y)
+
\beta \cdot \max_{\delta \in S}
\mathrm{KL}(f_\theta(x)\,\|\,f_\theta(x+\delta))
$$

其中：

- 第一项保证干净样本分类正确。
- 第二项要求对抗样本的预测分布不要偏离原样本太多。
- $\beta$ 控制两者权衡。$\beta$ 大，模型更偏向鲁棒；$\beta$ 小，模型更偏向自然精度。

从工程角度看，TRADES 比“只做交叉熵对抗训练”更容易显式调节精度和鲁棒性的平衡。

### 5. 真实工程例子

以语音识别为例，输入不是图像像素，而是声学特征或波形片段。攻击者可以加入非常小的背景噪声，使“打开车门”被识别成别的命令。工程上可以在训练中对高损失音频片段生成 FGSM 或 PGD 扰动，再训练模型保持预测稳定。这样做的目标不是让模型对所有噪声都免疫，而是在给定噪声幅度约束内提升最坏情况表现。

---

## 代码实现

下面用一个最小可运行 Python 例子说明 FGSM 的核心计算逻辑。这里不用深度学习框架，只演示“梯度方向 + 裁剪”的机制。

```python
import numpy as np

def fgsm_attack(x, grad, eps, x_min=0.0, x_max=1.0):
    x = np.asarray(x, dtype=np.float32)
    grad = np.asarray(grad, dtype=np.float32)
    x_adv = x + eps * np.sign(grad)
    return np.clip(x_adv, x_min, x_max)

# 玩具例子：二维输入
x = np.array([0.3, 0.7], dtype=np.float32)
grad = np.array([-0.2, 1.4], dtype=np.float32)
eps = 0.05

x_adv = fgsm_attack(x, grad, eps)

assert np.allclose(x_adv, np.array([0.25, 0.75], dtype=np.float32))
assert np.max(np.abs(x_adv - x)) <= eps + 1e-6
assert np.all((0.0 <= x_adv) & (x_adv <= 1.0))

print("x     =", x)
print("x_adv =", x_adv)
```

真正训练时，流程比上面多两件事：一是梯度来自模型损失对输入的反向传播；二是对抗样本生成后，还要继续用它参与参数更新。

下面是更接近真实训练循环的伪代码，逻辑适用于 PyTorch 一类框架：

```python
for x, y in train_loader:
    x = x.to(device)
    y = y.to(device)

    # 1. 生成对抗样本
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss_adv_gen = cross_entropy(logits, y)
    grad = autograd.grad(loss_adv_gen, x_adv)[0]

    x_adv = x_adv + eps * grad.sign()
    x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)   # 投影回 epsilon 球
    x_adv = x_adv.clamp(0.0, 1.0).detach()

    # 2. 用 clean / adv 计算训练损失
    logits_clean = model(x)
    logits_adv = model(x_adv)

    loss_clean = cross_entropy(logits_clean, y)
    loss_adv = cross_entropy(logits_adv, y)

    loss = 0.5 * loss_clean + 0.5 * loss_adv

    # 3. 更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

如果改成 PGD，只要把“生成对抗样本”部分改成多步迭代即可。每一步都要重新对当前 `x_adv` 求梯度，再投影回合法范围。

对大模型训练，还会补两种优化：

| 优化手段 | 作用 |
|---|---|
| 混合精度训练 | 降低显存占用，减少部分算力开销 |
| gradient checkpointing | 以重算换显存，适合深层网络 |
| 只对高损失样本做 PGD | 降低每个 batch 的攻击成本 |
| 冻结底层部分参数 | 减少反向传播开销 |

---

## 工程权衡与常见坑

对抗训练最常见的问题不是“不会写”，而是“能跑但效果不对”。

第一类坑是 FGSM-AT 的 catastrophic overfitting。这个词的意思是“训练中突然失去鲁棒性，对抗精度快速塌掉”。原因通常是一步攻击太弱，模型学会了针对这一步攻击的局部模式，而不是学到真正稳定的边界。常见修复手段有三种：加入随机初始化、改用更强的多步 PGD、增加正则或一致性约束。

第二类坑是成本失控。若 PGD 用 $K$ 步，则一个 batch 的开销大致接近自然训练的 $(K+1)$ 倍。因为每一步攻击都要额外前向和反向。比如自然训练 1 次前反传，PGD-10 可能接近 11 倍量级，实际还会受显存、通信和混合精度实现影响。

第三类坑是评估不严谨。很多实验只报告干净样本准确率，或者用弱攻击评估强鲁棒性，这会导致错误结论。正确做法是：训练时和评估时分开设计攻击强度，评估攻击通常应强于训练攻击，至少要覆盖多个 $\varepsilon$ 和多步 PGD 设定。

第四类坑是大模型“根本装不下”。对抗训练需要对输入求梯度，常常还要多次展开攻击步骤，这会让激活保存、优化器状态、梯度缓存同时变大。没有混合精度、checkpointing、参数分片时，模型可能连单卡 batch 都放不进去。

下面这张表适合做工程选型：

| 方法 | 训练成本 | 鲁棒性 | 主要优点 | 主要风险 |
|---|---|---|---|---|
| FGSM-AT | 低 | 中 | 快，适合入门和资源紧张场景 | 容易过拟合到弱攻击 |
| PGD-AT | 高 | 高 | 更接近 worst-case，效果稳定 | 时间和显存成本高 |
| TRADES | 中到高 | 高 | 可显式调自然精度/鲁棒性平衡 | 调参复杂，KL 项实现需谨慎 |

一个真实工程例子是“高损失样本筛选”。设一个 batch 有 256 个样本，不一定每个样本都值得做 10 步 PGD。可以先做一次普通前向，选出 loss 最高的 top-k，比如 64 个样本，再只对这部分样本生成对抗扰动。这样能显著减少攻击计算量，代价是鲁棒性提升可能不如全量样本稳定，需要通过验证集确认收益。

---

## 替代方案与适用边界

对抗训练不是唯一的鲁棒性方案，也不是所有项目的默认选择。

如果你的任务主要面对自然噪声而不是恶意攻击，普通数据增强、噪声注入、标签平滑、正则化、输入预处理，往往更便宜，也更容易落地。比如图像分类中常见的随机裁剪、颜色抖动，能提升对一般分布漂移的稳健性，但它们不等价于 worst-case 鲁棒性。

如果你的系统不能承受训练成本，还可以考虑推理期方法，例如异常样本检测、输入去噪、集成预测。它们的逻辑是“先识别可疑输入，再拒绝或修正”，但注意这不等于真正提高模型本体的鲁棒边界。

适用边界可以用一个简单表格概括：

| 场景 | 是否适合对抗训练 | 原因 |
|---|---|---|
| 安全敏感视觉识别 | 适合 | 小扰动误判代价高 |
| 语音指令识别 | 适合 | 背景噪声和恶意音频都可能触发错误 |
| 一般推荐系统排序 | 视情况而定 | 通常更关心分布漂移而非恶意微扰 |
| 超大模型全参数训练 | 受限 | 计算和显存成本可能过高 |

还可以给一个粗略经验边界。若你的扰动预算 $\varepsilon$ 很小，对抗训练更接近“局部平滑”；若 $\varepsilon$ 过大，训练出来的样本可能已经偏离真实数据流形。数据流形，白话说，就是“真实样本通常会落在的那块结构化区域”。一旦扰动超出这块区域，模型学到的可能不是鲁棒性，而是对不自然样本的过度适配。

在大模型上，尤其是高分辨率视觉模型、语音模型、多模态模型，常见工程组合是：

$$
\text{AT 可行性} \approx f(\text{显存}, \text{攻击步数}, \text{模型规模}, \text{batch size})
$$

这不是严格数学定理，但工程上非常实用：模型越大、攻击步数越多、batch 越大，训练越容易卡死。因此大模型常采用“轻量 FGSM 预热 + 局部 PGD 微调 + 混合精度 + checkpointing”的折中路线，而不是全程重型 PGD。

---

## 参考资料

- [Goodfellow et al., Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
  说明：FGSM 的经典论文，适合入门理解“一步梯度攻击”的来源。

- [Madry et al., Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)
  说明：PGD 对抗训练的代表工作，适合理解 min-max 鲁棒优化的标准表述。

- [Zhang et al., Theoretically Principled Trade-off between Robustness and Accuracy (TRADES)](https://arxiv.org/abs/1901.08573)
  说明：TRADES 原始论文，适合理解自然精度与鲁棒性的显式权衡。

- [TensorFlow 官方 FGSM 教程](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm)
  说明：有可运行示例和可视化，适合第一次动手做对抗样本生成。

- [MDPI 综述：Adversarial Machine Learning and Deep Learning](https://www.mdpi.com/1999-5903/13/12/300)
  说明：对 PGD、TRADES、鲁棒误差分解有系统梳理，适合做概念对照。

- [Emergent Mind: Adversarial Training](https://www.emergentmind.com/topics/adversarial-training)
  说明：适合快速回顾概念、关键方法和近年发展方向。

- [Emergent Mind: Fast Gradient Sign Method](https://www.emergentmind.com/topics/fast-gradient-sign-method-fgsm)
  说明：适合快速理解 FGSM、PGD、快速对抗训练和常见问题。
