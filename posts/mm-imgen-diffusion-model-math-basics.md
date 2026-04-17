## 核心结论

扩散模型的数学核心可以压缩成一句话：先用一个固定、可解析的高斯过程把真实样本逐步推向标准高斯，再学习一个逆过程，把“纯噪声”一步步拉回数据分布。这里的 **score** 指的是 $\nabla_x \log p(x)$，白话解释就是“当前点往哪里移动，概率密度会上升最快”。

它之所以成立，不是因为模型直接学会了“画图”，而是因为训练时我们人为知道每个样本被加了多少噪声，于是可以把“生成复杂图像”转写成“在不同噪声尺度下预测噪声方向”。这件事等价于 **denoising score matching**，白话解释就是“通过去噪来逼近真实分布的梯度”。

前向过程通常写成：

$$
q(x_t|x_{t-1})=\mathcal{N}\left(\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I\right)
$$

令 $\alpha_t = 1-\beta_t,\ \bar\alpha_t=\prod_{s=1}^t \alpha_s$，则可以直接从 $x_0$ 采样到任意时刻：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal N(0,I)
$$

逆向过程写成：

$$
p_\theta(x_{t-1}|x_t)=\mathcal N(\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))
$$

其中网络常见做法不是直接输出图像，而是预测噪声 $\hat\epsilon_\theta(x_t,t)$。只要它在多个噪声层级上逼近正确的 score，逆向链就能把标准高斯逐步还原成多模态图像分布。

一个最小玩具例子足够说明这件事。设一维标量 $x_0=0.5$，$\beta_1=0.02$，所以 $\alpha_1=0.98$。若采样到噪声 $z=0.3$，则

$$
x_1=\sqrt{0.98}\cdot 0.5+\sqrt{0.02}\cdot 0.3\approx 0.5378
$$

若模型预测 $\hat\epsilon\approx 0.3$，则可按

$$
\hat x_0=\frac{x_1-\sqrt{1-\alpha_1}\hat\epsilon}{\sqrt{\alpha_1}}
$$

把原值近似恢复回去。这里最关键的不是“记住图像”，而是知道“噪声往哪里减”。

| 维度 | 前向扩散 | 逆向生成 |
|---|---|---|
| 目标 | 把数据分布推向高斯 | 把高斯拉回数据分布 |
| 条件分布 | $q(x_t|x_{t-1})$ | $p_\theta(x_{t-1}|x_t)$ |
| 是否固定 | 固定，不学习 | 学习得到 |
| 关键变量 | $\beta_t,\alpha_t,\bar\alpha_t$ | $\mu_\theta,\Sigma_\theta,\hat\epsilon_\theta$ |
| 每步做什么 | 加一点噪声 | 去一点噪声 |
| 网络真正学什么 | 不需要网络 | score 或噪声预测 |

---

## 问题定义与边界

扩散模型解决的问题，不是“如何一次性从随机向量生成图像”，而是“如何把一个困难的全局建模问题，拆成很多个局部去噪问题”。这里的 **局部** 指某个固定噪声尺度附近的小范围分布结构，白话解释就是“先只学会在模糊程度为某一级时怎么修正”。

设真实数据来自 $p_{\text{data}}(x_0)$。前向过程定义为一个固定马尔可夫链：

$$
q(x_{1:T}|x_0)=\prod_{t=1}^{T} q(x_t|x_{t-1})
$$

当 $T$ 足够大且噪声调度合适时，$q(x_T)$ 会接近标准高斯 $\mathcal N(0,I)$。这一步给出了边界：只有当前向链确实把复杂分布“洗”成一个简单分布时，逆向从高斯出发才有明确起点。

训练时难点在于，我们并不知道真实逆分布 $q(x_{t-1}|x_t)$ 的闭式形式；但如果额外条件化在 $x_0$ 上，则

$$
q(x_{t-1}|x_t,x_0)
$$

是高斯，可解析。于是可以通过最大化 ELBO，把学习任务转成一系列 KL 项最小化。对 DDPM 来说，进一步化简后，常见目标变成预测噪声：

$$
\mathcal L_{\text{simple}} = \mathbb E_{x_0,\epsilon,t}\left[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right]
$$

这和 denoising score matching 在本质上是一回事。DSM 写成：

$$
J_D(\theta)=\frac12\mathbb E_{x,\epsilon}\left[\left\|s_\theta(x+\sigma\epsilon)+\frac{\epsilon}{\sigma}\right\|^2\right]
$$

其中 $s_\theta$ 逼近的是 $\nabla_x \log p_\sigma(x)$。白话解释：如果你知道样本是“干净信号 + 某个强度的高斯噪声”，那么最合理的去噪方向，就等于这个噪声分布下的密度上升方向。

为什么这能避开直接建模整个图像分布的困难？因为在小噪声时，模型只需要恢复局部结构；在大噪声时，模型只需要决定粗布局。不同时间步承担不同难度层级，而不是让单一步骤同时处理纹理、语义、布局和细节。

以 $t=1$ 为例，如果 $\beta_1$ 很小，那么

$$
q(x_1|x_0)=\mathcal N(\sqrt{1-\beta_1}x_0,\beta_1 I)
$$

意味着图像主体结构几乎没变，只是被轻微扰动。此时逆向要学的不是整个世界模型，而只是“在很小扰动下，如何把样本拉回高密度区域”。这也是扩散模型对初学者最容易忽视的一点：它强在分解问题，而不是单步表达力神奇。

边界也要明确：

| 问题 | 扩散模型是否擅长 | 原因 |
|---|---|---|
| 高质量图像生成 | 擅长 | 局部去噪目标稳定 |
| 强条件控制 | 中等 | 需要额外条件注入与 guidance |
| 长文本精确渲染 | 不擅长 | 字符是高精度离散结构 |
| 严格符号推理 | 不擅长 | score 主要学习连续空间几何 |
| 多模态联合生成 | 可做但复杂 | 一致性与质量存在权衡 |

---

## 核心机制与推导

前向过程是整个数学闭环的起点。每一步加入少量高斯噪声：

$$
q(x_t|x_{t-1})=\mathcal N(\sqrt{1-\beta_t}x_{t-1},\beta_t I)
$$

把它递推展开，可以得到：

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t
$$

继续展开到初始时刻：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
$$

这条式子非常关键，因为它告诉我们：训练时不需要真的一步一步加噪，可以直接随机采样 $t$ 和 $\epsilon$，一次构造出 $x_t$。这让训练成本大幅下降。

接着看逆向。理想情况下我们想要：

$$
q(x_{t-1}|x_t)
$$

但它难以直接求解。好消息是，当条件化在 $x_0$ 上时，后验是高斯：

$$
q(x_{t-1}|x_t,x_0)=\mathcal N(\tilde\mu_t(x_t,x_0),\tilde\beta_t I)
$$

其中均值可写成 $x_t$ 和 $x_0$ 的线性组合。于是如果网络能根据 $x_t,t$ 估计出 $x_0$ 或 $\epsilon$，就能近似恢复这个后验均值。

常见参数化是噪声预测。因为

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
$$

所以有

$$
x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}{\sqrt{\bar\alpha_t}}
$$

若网络输出 $\hat\epsilon_\theta(x_t,t)$，则可以得到 $\hat x_0$，再代入后验均值公式，形成逆向采样。很多入门材料直接写成：

$$
\mu_\theta(x_t,t)\approx \frac{x_t-\sqrt{1-\alpha_t}\hat\epsilon_\theta(x_t,t)}{\sqrt{\alpha_t}}
$$

严格地说，不同论文在系数上略有差异，但思想一致：网络预测噪声，采样器用这个噪声修正当前状态。

为什么“预测噪声”会等价于“预测 score”？因为高斯扰动下，两者线性相关。对扰动分布 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$，其 score 满足与噪声项成比例的关系，因此最小化噪声 MSE，本质上就是在逼近各噪声层上的 $\nabla_x\log p_t(x)$。

可以把整个推导链理解成四步：

| 步骤 | 数学对象 | 作用 |
|---|---|---|
| 1 | $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ | 直接构造任意噪声层训练样本 |
| 2 | $q(x_{t-1}|x_t,x_0)$ | 给出可解析的理想后验 |
| 3 | $\epsilon_\theta(x_t,t)$ 或 $s_\theta(x_t,t)$ | 用网络替代未知量 |
| 4 | $p_\theta(x_{t-1}|x_t)$ | 形成可迭代的生成过程 |

再看一个玩具例子。假设一维数据只由两个峰构成，比如 $x_0\in\{-2,2\}$。如果没有噪声，分布是不连续的双峰，很难直接从标准高斯一步映射过去；但当前向扩散进行到中间层时，这两个峰被抹平并局部重叠，score 场会变得平滑。逆向生成就不再是“一步跨越两个模式”，而是“沿着概率梯度慢慢往某个峰走”。这解释了扩散模型为什么天然适合多模态分布。

---

## 代码实现

实现层面最重要的事实是：训练时只需采样一个时间步，不需要真的跑完整条链。输入是一张干净样本 $x_0$，再随机采样时间步 $t$ 和噪声 $\epsilon$，构造：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
$$

然后让网络预测 $\hat\epsilon=\epsilon_\theta(x_t,t)$，损失用 MSE：

$$
\mathcal L=\|\hat\epsilon-\epsilon\|^2
$$

下面给出一个可运行的 Python 玩具实现，只演示公式是否自洽，不依赖深度学习框架：

```python
import math

def forward_sample(x0: float, beta_t: float, eps: float):
    alpha_t = 1.0 - beta_t
    x_t = math.sqrt(alpha_t) * x0 + math.sqrt(1.0 - alpha_t) * eps
    return x_t, alpha_t

def recover_x0(x_t: float, alpha_t: float, eps_pred: float):
    return (x_t - math.sqrt(1.0 - alpha_t) * eps_pred) / math.sqrt(alpha_t)

# 玩具例子
x0 = 0.5
beta_1 = 0.02
eps = 0.3

x1, alpha_1 = forward_sample(x0, beta_1, eps)
x0_hat = recover_x0(x1, alpha_1, eps)

# 数值应接近原值
assert abs(x0_hat - x0) < 1e-9

# 如果噪声预测错了，恢复会偏移
x0_bad = recover_x0(x1, alpha_1, 0.0)
assert abs(x0_bad - x0) > 0.01

print(round(x1, 6), round(x0_hat, 6), round(x0_bad, 6))
```

如果换成真实训练流程，伪代码基本就是：

```python
# x0: 干净图像
# t: 随机采样的时间步
# eps: 标准高斯噪声
x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * eps
eps_pred = model(x_t, t)
loss = mse(eps_pred, eps)
```

采样时则从纯噪声开始，逐步逆推：

```python
x_t = sample_standard_gaussian()

for t in reversed(range(1, T + 1)):
    eps_pred = model(x_t, t)
    mu = compute_mean_from_eps(x_t, t, eps_pred)
    if t > 1:
        z = sample_standard_gaussian()
    else:
        z = 0.0
    x_t = mu + sigma_t * z
```

这里的 **时间步嵌入** 指把离散步数 $t$ 编码成网络可理解的向量，白话解释就是“告诉模型当前噪声有多重”。没有它，模型不知道自己是在处理“几乎干净的图”还是“几乎纯噪声”。

真实工程例子通常不是直接在像素空间训练超大模型，而是进入 **latent diffusion**。白话解释就是先用自编码器把图像压到更小的连续潜空间，再在潜空间做扩散。这样做的直接收益是显存和计算量大幅下降，代价是最终质量受编码器上限约束。

例如文本生成图像系统里，流程通常是：

| 阶段 | 输入 | 输出 | 作用 |
|---|---|---|
| VAE 编码 | 原始图像 | 潜变量 $z$ | 压缩空间维度 |
| 文本编码 | prompt | 文本条件向量 $c$ | 提供语义控制 |
| 扩散 U-Net | $z_t,t,c$ | $\hat\epsilon$ | 在潜空间去噪 |
| VAE 解码 | 去噪后潜变量 | 图像 | 恢复像素内容 |

这也是为什么现代图像生成系统常把“数学上是扩散模型”和“工程上是条件 latent diffusion”同时成立。

---

## 工程权衡与常见坑

扩散模型训练稳定、样本质量高，但工程问题并没有因此消失。它最典型的坑，不是“训不起来”，而是“训起来以后对齐不精确”。

先看一个真实工程例子。多模态 latent diffusion 会把图像、文本甚至结构化信号分别编码到共享潜空间，再通过 masked diffusion 或 soft condition 进行联合建模。假设提示词要求“白色背景、绿色箭头、中央一个十字形符号”，模型可能会出现三类冲突：箭头颜色对了但形状错，十字形结构存在但位置偏移，文字或符号边缘糊掉。根因不是单一模块坏了，而是不同模态在噪声层上的约束强度不同，导致联合 score 场不一致。

常见坑可以直接归纳如下：

| 常见坑 | 根因 | 典型现象 | 应对策略 |
|---|---|---|---|
| 文本控制弱 | score 更擅长连续视觉局部结构 | 计数错误、关系错位 | 更强文本编码器、交叉注意力改进、guidance 调度 |
| 字体失真 | 字符是高精度离散结构 | 拼写错误、字母粘连 | 局部文字损失、分区生成、分辨率优先采样 |
| 多模态一致性差 | 各模态噪声同步困难 | 图文不对应、结构冲突 | latent mask、soft condition、阶段化生成 |
| 评估困难 | 常规 FID 不测语义细节 | 看起来像对，实际上不对 | 引入 VLM 评估、任务化基准、人工抽检 |
| 采样慢 | 需要多步迭代 | 在线服务延迟高 | DDIM、蒸馏、少步采样器 |

需要特别强调“文字生成为什么经常失败”。字符不是普通纹理，而是一种小尺度、高约束、离散组合结构。扩散模型的 score 在局部像素梯度上工作得很好，但“把 r 和 n 区分成 m 还是 rn”这种问题，往往需要更精确的结构先验。也就是说，模型可能知道这里“应该像文字”，却不知道“必须是哪几个字母”。

另一个典型坑是 guidance 过强。**guidance** 指采样时人为加强条件信号，白话解释就是“让模型更听 prompt 的话”。它能提高对齐，但强度过大时会让样本多样性下降，甚至出现高对比、边缘过锐、局部重复等现象，本质上接近 mode collapse。

工程上还要处理噪声调度。$\beta_t$ 太小，前向链混合不充分，逆向学习到的是狭窄局部；$\beta_t$ 太大，早期就破坏了太多结构，逆向难度会上升。初学者容易把它看成一个次要超参数，实际上它决定了每个时间步承担的“信息破坏量”。

---

## 替代方案与适用边界

DDPM 不是唯一选择。它只是“离散时间、显式加噪、显式去噪”这一路线里最经典的一种。实际工程里更常见的是若干变体。

| 方法 | 核心思想 | 优点 | 代价或边界 |
|---|---|---|---|
| DDPM | 离散时间步逐步去噪 | 数学清晰、训练稳定 | 采样步数多 |
| Latent Diffusion | 在潜空间而非像素空间扩散 | 高分辨率更省算力 | 质量受编码器约束 |
| Score-based SDE/ODE | 把扩散写成连续时间随机微分方程或常微分方程 | 采样器设计灵活，适合逆问题 | 理解与实现更复杂 |
| CFG | 采样时增强条件分支 | 不重训即可提升对齐 | guidance 大时损失多样性 |
| Feedback Guidance | 用反馈信号动态修正采样 | 对复杂约束更灵活 | 需要额外反馈模型，链路更重 |

如果任务是高分辨率图像生成，Latent Diffusion 往往比像素空间 DDPM 更合理，因为像素空间维度太高，算力消耗太大。如果任务是图像修复、超分辨率、条件逆问题，score-based SDE/ODE 很有吸引力，因为它方便把观测约束写进采样过程。

一个实际判断标准是：你到底更在意“质量”“速度”还是“控制力”。

1. 只追求高质量离线生成，DDPM 或其改进采样器足够。
2. 追求大分辨率和可部署性，Latent Diffusion 更现实。
3. 追求 prompt 强控制，CFG 常常是默认选项，但不能把强度拉得过高。
4. 需要复杂反馈闭环，比如多阶段审核或外部判别器纠偏，Feedback Guidance 更合适。

边界同样明确。扩散模型适合连续空间、近似可微、允许逐步修正的任务；不适合要求严格离散精确输出、一步低延迟、或必须保证逻辑一致性的任务。比如生成海报底图很合适，但要求“一次就把 12 位订单号无误印在图中”就不合适，至少不能只靠纯扩散主干完成。

---

## 参考资料

| 名称 | 出处 | 关键贡献 |
|---|---|---|
| Score Matching and Diffusion Models – III | Vishnu Boddeti 课程讲义 | 前向/逆向高斯链、DSM 与 score matching 推导 |
| Building Diffusion Model’s Theory from Ground Up | ICLR 相关博文 | 用去噪视角解释扩散模型为什么成立 |
| Multi-Modal Latent Diffusion | Entropy 2024 | 多模态 latent diffusion 的工程流程与一致性权衡 |
| 文字生成失真分析相关文章 | 工程博客与研究讨论 | 解释扩散模型为何难以稳定渲染文字 |
| 条件引导与反馈引导相关论文 | OpenReview 等 | 分析质量、多样性、对齐之间的采样期权衡 |

1. Vishnu Boddeti, *Score Matching and Diffusion Models – III*. 重点看前向高斯链、逆向高斯近似、DSM 目标。
2. *Building Diffusion Model’s Theory from Ground Up*. 适合建立“噪声预测为何可行”的直觉。
3. *Multi-Modal Latent Diffusion*, Entropy 2024. 适合理解多模态共享潜空间、masked diffusion、coherence 权衡。
4. 条件控制与反馈引导相关论文。适合理解 CFG、反馈引导与多样性损失问题。
5. 关于图像中文字渲染失败的工程分析文章。适合理解扩散模型在离散结构上的局限。
