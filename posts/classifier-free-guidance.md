## 核心结论

Classifier-Free Guidance，简称 CFG，本质上是一种“在同一个生成模型里同时学会有条件生成和无条件生成，再在推理时把两者差值放大”的方法。这里的“无条件”可以先理解成“不看 prompt 也要给出一个合理生成方向”。它不需要额外训练分类器，这一点比早期的 classifier guidance 更实用。

CFG 的标准采样公式是：

$$
\hat{\epsilon}_{\text{cfg}}
=
(1+w)\,\epsilon_\theta(x_t,t,y)-w\,\epsilon_\theta(x_t,t,\varnothing)
$$

其中 $y$ 是条件，比如文本 prompt；$\varnothing$ 是空条件；$w$ 是 guidance scale，也叫引导强度。它的作用不是简单把条件分支乘以一个倍数，而是把“条件输出相对无条件输出多出来的那部分”做线性外插。白话说，模型本来知道“自然会生成什么”，CFG 额外强调“为了满足 prompt，应该往哪个方向再推一把”。

一个新手可直接记住的玩具例子是：如果某一步条件预测是 `0.6`，无条件预测是 `0.1`，取 `w=4`，那么

$$
(1+4)\times 0.6 - 4\times 0.1 = 2.6
$$

结果不是把 `0.6` 变成 `3.0`，而是把“条件比无条件多出的信号”放大后再加回去。所以 CFG 强调的是“差值”，不是“单边倍增”。

CFG 最重要的工程结论只有两个：

| guidance scale `w` | prompt 对齐 | 多样性 | 常见现象 |
|---|---:|---:|---|
| 低，例如 `0~2` | 较弱 | 较高 | 更自由，但可能“不听话” |
| 中，例如 `3~7` | 较强 | 中等 | 通常是质量与多样性的折中区间 |
| 高，例如 `8+` | 很强 | 较低 | 可能出现样式收缩、细节失真、模式坍缩 |

真实工程里，文本到图像系统经常把 CFG 当成默认控制旋钮：调低时图像更散、更有随机性；调高时更贴 prompt，但不同采样结果会越来越像。对于“零基础到初级工程师”，最值得建立的直觉是：CFG 不是“让模型更聪明”，而是“让模型在采样时更偏向条件方向”。

---

## 问题定义与边界

条件生成的核心任务是：已知条件 $y$，从目标分布 $p(x\mid y)$ 中生成样本。这里“条件生成”可以先理解成“带要求地生成”，比如“生成一张戴红帽子的猫”。扩散模型本来擅长从噪声逐步还原数据，但 prompt 信号在高噪声阶段往往很弱，导致模型可能只生成“像图像”，却不一定“像你要的图像”。

早期做法是 classifier guidance：额外训练一个分类器，用分类器的梯度去推动采样轨迹。这条路的问题是明显的：

| 方法 | 需要额外模型吗 | 训练复杂度 | 推理耦合度 | 主要问题 |
|---|---:|---:|---:|---|
| classifier guidance | 需要 | 高 | 高 | 额外分类器难训练，噪声下鲁棒性差 |
| CFG | 不需要 | 中 | 中 | 需要双分支推理，`w` 过大易失真 |

CFG 解决的是“如何不依赖额外分类器，也能让采样更听条件”的问题。它的做法是在训练阶段引入 null condition，也就是空条件。白话说，就是故意让模型有时“看 prompt”，有时“装作没看到 prompt”，这样同一个网络就会同时学到两种预测：

$$
\epsilon_\theta(x_t,t,y)
\quad\text{和}\quad
\epsilon_\theta(x_t,t,\varnothing)
$$

其中 $\epsilon_\theta$ 是噪声预测器，可以理解成“模型认为当前噪声该怎么去掉”。

训练时常见做法是：对每个样本，以概率 $p_{\text{drop}}$ 把原始条件 $y$ 替换为 $\varnothing$。于是模型学到的其实是一个混合任务：

$$
\mathcal{L}
=
\mathbb{E}_{x,\epsilon,t,y'}
\left[
\|\epsilon-\epsilon_\theta(x_t,t,y')\|^2
\right]
$$

其中

$$
y'=
\begin{cases}
y, & \text{以概率 } 1-p_{\text{drop}} \\
\varnothing, & \text{以概率 } p_{\text{drop}}
\end{cases}
$$

这个机制的边界也很明确。

第一，`w` 不能无限增大。因为采样时做的是线性外插，外插过强会把样本硬推向少数高分模式，导致 mode collapse。这里“mode collapse”可以先理解成“本来应该有很多合理答案，最后只剩几种模板化答案”。

第二，null 条件训练比例不能太低。如果 $p_{\text{drop}}$ 太小，模型的无条件分支学得不准，那么推理时拿它做“基线”就不稳定，最终差值也会失真。

第三，CFG 不是免费午餐。推理时每一步通常要算两次网络：一次条件，一次无条件。因此它提升可控性的同时，也增加了推理成本。

一个适合新手的类比是：训练阶段相当于让同一个学生同时练两种题目，“按要求答题”和“完全不看题干自由作答”。到了考试时，不是只看“按要求答题”的结果，而是比较这两种答案的偏差，再把偏差放大。这样得到的不是普通答案，而是“更有针对性”的答案。

---

## 核心机制与推导

CFG 的核心推导可以从“条件分支 = 无条件基线 + 条件增量”出发。把条件预测写成：

$$
\epsilon_\theta(x_t,t,y)
=
\epsilon_\theta(x_t,t,\varnothing)
+
\Delta_\theta(x_t,t,y)
$$

其中 $\Delta_\theta$ 表示“条件额外带来的方向变化”。那么 CFG 采样可写成：

$$
\hat{\epsilon}_{\text{cfg}}
=
\epsilon_\theta(x_t,t,\varnothing)
+
(1+w)\Delta_\theta(x_t,t,y)
$$

因为

$$
\Delta_\theta(x_t,t,y)
=
\epsilon_\theta(x_t,t,y)-\epsilon_\theta(x_t,t,\varnothing)
$$

所以代回去得到：

$$
\hat{\epsilon}_{\text{cfg}}
=
\epsilon_\theta(x_t,t,\varnothing)
+
(1+w)\big(
\epsilon_\theta(x_t,t,y)-\epsilon_\theta(x_t,t,\varnothing)
\big)
$$

整理后就是常见形式：

$$
\hat{\epsilon}_{\text{cfg}}
=
(1+w)\epsilon_\theta(x_t,t,y)-w\epsilon_\theta(x_t,t,\varnothing)
$$

这说明 CFG 做的事情不是“只信条件分支”，而是“以无条件分支为参考，把条件偏移量外插到更远的位置”。这也是为什么 `w=0` 时退化为普通条件采样，而 `w>0` 时条件被逐步强化。

玩具例子最能说明这个机制。设某一步：

- 条件预测：`0.6`
- 无条件预测：`0.1`
- guidance scale：`w=4`

则：

$$
\hat{v}_{\text{cfg}}=(1+4)\times0.6-4\times0.1=2.6
$$

从分解角度看更直观：

- 无条件基线：`0.1`
- 条件增量：`0.6 - 0.1 = 0.5`
- 放大后：`0.1 + 5 × 0.5 = 2.6`

所以关键不是 `0.6` 本身，而是“相对 `0.1` 多出来的 `0.5`”。

在流匹配，Flow Matching，指“直接学习连续时间速度场而不是离散去噪步骤”的生成框架里，CFG 有等价形式。把噪声预测换成速度预测 $v_\theta$，可写成：

$$
\hat{v}_{\text{cfg}}
=
(1+s)v_\theta(x_t,t,y)-s\,v_\theta(x_t,t,\varnothing)
$$

这里的 $s$ 与扩散模型里的 $w$ 扮演同一角色，都是引导强度。也就是说，CFG 不是某个特定采样器私有的技巧，而是一种更一般的“条件方向外插”机制。

可以把整条流程压缩成一个简单图：

```text
训练阶段：
(x, y) -> 随机丢弃条件 -> 模型学习 cond / uncond 两种预测

采样阶段：
x_t -> 模型算 cond
x_t -> 模型算 uncond
两者线性组合 -> 得到 cfg 预测 -> 更新 x_{t-1}
```

真实工程例子是文本到图像系统。比如输入 prompt：“a red vintage car parked on a rainy street at night”。如果不使用 CFG，模型可能只生成“夜景街道上的车”；使用中等强度 CFG 后，“red”“vintage”“rainy”这些描述更容易被保留下来；如果把 `w` 拉得过高，图像可能会过度重复某些显著特征，比如车总处于极其相似的角度、雨夜光斑结构高度模板化，这就是多样性被压缩的直接表现。

从概率解释看，CFG 近似于沿着 $\nabla_x \log p(y\mid x_t)$ 的方向推采样轨迹。即便不展开严格推导，工程上也可记住这层含义：CFG 在做“让当前样本更像能解释该条件的样本”。

---

## 代码实现

训练实现的关键点只有一个：随机把部分条件替换成 null token。这里的“token”可以白话理解成“给模型看的条件占位符”。对文本条件模型来说，通常会准备一个空 prompt 或专门的空嵌入。

下面先给一个可运行的 Python 玩具实现，只演示 CFG 线性组合本身：

```python
def cfg_combine(cond, uncond, w):
    return (1 + w) * cond - w * uncond

# 玩具例子
value = cfg_combine(cond=0.6, uncond=0.1, w=4)
assert abs(value - 2.6) < 1e-9

# 边界检查：w=0 时应退化为条件分支
assert cfg_combine(cond=1.2, uncond=-0.5, w=0) == 1.2

# 如果 cond == uncond，说明条件没有提供额外信息，CFG 不应改变结果
assert cfg_combine(cond=0.8, uncond=0.8, w=7) == 0.8
```

训练阶段的 Python 风格伪代码如下：

```python
import random

NULL_PROMPT = ""

def maybe_drop_condition(prompt, drop_prob=0.1):
    if random.random() < drop_prob:
        return NULL_PROMPT
    return prompt

def train_step(model, x0, prompt, t, noise, drop_prob=0.1):
    used_prompt = maybe_drop_condition(prompt, drop_prob)
    xt = q_sample(x0, t, noise)           # 前向加噪
    pred_noise = model(xt, t, used_prompt)
    loss = mse_loss(pred_noise, noise)
    return loss
```

这个训练过程有两个实际含义：

1. 模型参数是共享的，不是两套网络。
2. cond / uncond 的区别只体现在条件输入是否被替换为空。

采样阶段则是每一步分别算条件与无条件输出，再线性组合：

```python
def sample_step(model, xt, t, prompt, w):
    eps_cond = model(xt, t, prompt)
    eps_uncond = model(xt, t, "")
    eps_cfg = (1 + w) * eps_cond - w * eps_uncond
    x_prev = ddim_or_solver_update(xt, t, eps_cfg)
    return x_prev
```

如果写成完整循环，大致是：

```python
def generate(model, xT, prompt, timesteps, w):
    x = xT
    for t in timesteps:
        x = sample_step(model, x, t, prompt, w)
    return x
```

真实工程中，这种写法的成本是明确的：每个时间步要做两次前向。假设原始采样要跑 30 步，那么加 CFG 后等价于 60 次模型调用量级。为降低开销，常见优化包括：

| 优化点 | 作用 | 适用说明 |
|---|---|---|
| batch 合并 cond/uncond | 一次前向同时算两支 | 最常见，吞吐更高 |
| 共享 timestep embedding | 减少重复计算 | 依赖模型实现细节 |
| KV cache 或中间缓存 | 降低重复编码成本 | 更适合大文本编码器场景 |
| 分段启用 CFG | 只在关键步使用强引导 | 兼顾质量与速度 |

在很多图像系统里，会把条件和无条件拼成同一个 batch，一次丢给 U-Net 或 Transformer，再把结果拆开组合。这种做法不会改变数学形式，但会显著改善 GPU 利用率。

---

## 工程权衡与常见坑

CFG 最常见的误区是把 `w` 当成“越大越好”的质量旋钮。实际上它调的是“对齐和多样性之间的平衡”，不是单边提升。高 `w` 往往让图像更听 prompt，但也更容易把样本压到少数模式附近。

下面是工程上最常见的坑：

| 问题 | 现象 | 原因 | 常见补救 |
|---|---|---|---|
| mode collapse | 多次采样结果越来越像 | 高 `w` 过度外插，样本收缩 | 降低 `w`、分段缩放、CFGibbs |
| 早期高噪误导 | 构图一开始就跑偏 | 高噪阶段条件信号本来就不稳 | early step 用更小 `w` |
| 无条件分支不准 | CFG 效果忽好忽坏 | 训练时 null 条件比例太低 | 提高 `p_drop`，重训无条件能力 |
| 过饱和或伪细节 | 图像“很像但很假” | 外插过强放大误差 | 动态阈值、zero-init、减小后段 `w` |
| 推理成本翻倍 | 延迟和显存压力增加 | 每步双分支前向 | batch 合并、减少步数、按阶段启用 |

mode collapse 之所以重要，是因为它不是“随机性下降一点”这么简单，而是会系统性损害分布覆盖。白话说，本来 prompt “a cat” 应该能对应长毛猫、短毛猫、室内、室外、不同姿态和光照；高 `w` 之后，模型可能总掉进几种最稳妥的“猫模板”。

从理论上讲，标准 CFG 只放大了吸引项，却没有完整补偿分布中的斥力结构。一些讨论会把这类缺失与 Rényi 校正项联系起来。对工程师来说，不需要先吃透所有理论细节，但要知道结论：标准 CFG 在高引导强度下会偏向更尖锐、更窄的分布。

一个新手容易观察到的现象是：把 `w` 从 5 调到 12，画面会变得“更像 prompt”，但不同随机种子的结果开始高度相似，有时甚至连主体构图都差不多。这就是典型的样本收缩。更稳妥的做法通常是前期小 `w`、后期大 `w`，因为采样早期噪声大，模型对语义结构的判断本来就不可靠，此时强拉条件更容易把轨迹带偏。

真实工程例子可以看 Flow Matching 家族模型。CFG-Zero* 针对这类模型观察到：在某些阶段，速度预测误差较大，标准 CFG 会把误差一起放大，导致图像质量下降。其修正思路包括“优化缩放”和“零初始化”两类策略，本质上都是在控制：什么时候该强引导，什么时候该谨慎引导。对于文本到图像、文本到视频这类高维生成任务，这种修正比在小玩具数据上更有价值，因为误差累积更严重。

---

## 替代方案与适用边界

CFG 很常用，但不是唯一选择。是否继续使用标准 CFG，取决于你更在意什么：实现简单、prompt 对齐、多样性、还是推理成本。

先看几种主要方案的对比：

| 方法 | 核心思路 | 优点 | 缺点 | 更适合的边界 |
|---|---|---|---|---|
| CFG | cond 与 uncond 线性外插 | 简单、通用、无需额外分类器 | 高 `w` 易收缩，推理双前向 | 通用文本到图像/音频生成 |
| CFGibbs | 在 CFG 基础上加入校正项 | 更好缓解高引导下分布失真 | 实现更复杂，理论门槛高 | 追求高保真且关注分布覆盖 |
| classifier guidance | 用外部分类型梯度引导 | 数学解释直接 | 需额外分类器，耦合重 | 条件标签明确、可训练鲁棒分类器 |
| 分段缩放 | 不同步数使用不同 `w` | 便宜、工程收益高 | 需要调度经验 | 高噪早期不稳定的采样器 |
| CFG-Zero* | 在 Flow Matching 上改进缩放与初始化 | 对 prompt 对齐和质量更稳 | 依赖具体模型族 | Flow Matching 文本到图像/视频 |

如果目标是“快速做出一个能用的条件生成系统”，标准 CFG 仍然是首选，因为实现成本最低，几乎所有扩散/流匹配框架都能直接接入。

如果目标是“高引导强度下还要保留多样性”，那么标准 CFG 的边界就会暴露。此时有两个现实方向。

第一，使用分段缩放。白话说就是“开始时保守，后面再发力”。例如前 30% 步骤用 `w=1.5`，中间 40% 用 `w=4`，最后 30% 用 `w=6`。这种策略像分段加速，开始时先保证轨迹别跑偏，后面在低噪区域再强化 prompt。

第二，使用带校正的变体，如 CFGibbs 或针对 Flow Matching 的 CFG-Zero*。这类方法的共同目标是减少“只会往条件方向猛推，但没有处理外推副作用”的问题。

需要特别指出的是，CFG 适用的前提是模型真的学会了合理的无条件分支。如果你的训练数据太少、条件丢弃概率设计不合理，或者空条件嵌入本身就处理得差，那么 CFG 的公式虽然没错，实际效果也可能很差。换句话说，CFG 依赖一个有效的“对照组”，这个对照组就是无条件预测。

对初级工程师，最实用的策略通常是：

1. 先用标准 CFG 跑通。
2. 从中等 `w` 开始，不要一上来拉满。
3. 观察不同随机种子的多样性。
4. 如果 prompt 对齐不够，再考虑分段缩放。
5. 如果模型是 Flow Matching 体系，再评估 CFG-Zero* 一类改进。

---

## 参考资料

1. Ho, Jonathan, and Tim Salimans. “Classifier-Free Diffusion Guidance.” NeurIPS 2021 Workshop.  
   经典原始思路来源，给出 CFG 的基本训练与采样公式。  
   URL: https://arxiv.org/abs/2207.12598

2. Emergent Mind, “Classifier-Free Guidance Approach.”  
   适合做理论入门，整理了标准 CFG 公式、模式收缩讨论以及相关校正思路。文中对高 `w` 下的分布偏移问题有较清晰总结。  
   URL: https://www.emergentmind.com/topics/classifier-free-guidance-approach

3. Emergent Mind, “Joint Classifier-Free Guidance for Diffusion Models.”  
   可用于理解 CFG 在更广义生成框架中的写法，包括与流匹配表述的联系。阅读时注意页面版本与更新时间。  
   URL: https://www.emergentmind.com/topics/joint-classifier-free-guidance-cfg

4. Fan et al., “CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching Models.”  
   重点看 Flow Matching 场景下为什么标准 CFG 会放大误差，以及 zero-init、优化缩放如何改善文本到图像/视频生成。  
   URL: https://weichenfan.github.io/webpage-cfg-zero-star/

5. Lipman et al., “Flow Matching for Generative Modeling.”  
   如果需要理解 $v_{\text{cfg}}=(1+s)v_{\text{cond}}-sv_{\text{uncond}}$ 这类速度场写法，建议结合 Flow Matching 原论文一起看。  
   URL: https://arxiv.org/abs/2210.02747

6. 阅读提示  
   不同项目会对 guidance scale 的记号、是否把 `1+w` 写成 `w`、以及 null prompt 的具体实现略有差异。引用公式时应先确认论文版本、代码实现和采样器定义是否一致。
