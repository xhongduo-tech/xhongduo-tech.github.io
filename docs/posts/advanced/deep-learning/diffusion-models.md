---
title: 扩散模型：前向加噪与反向去噪
date: 2026-08-07
---

# 扩散模型：前向加噪与反向去噪

<div class="epigraph">
<p>打碎一面镜子容易，把它拼回去才是真正的挑战。</p>
<footer>—— 依据「加噪易、去噪难」的精神改写</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ Ho 等《DDPM》（2020）、Sohl-Dickstein 等（2015） ｜ 2026-08-07</p>
</div>

## 为什么从扩散模型开始

GAN 用「博弈」生成但训练不稳定，VAE 用「一次重建」但生成模糊。**扩散模型（diffusion model）**走了一条「慢工出细活」的路：**把数据一步步加噪成纯噪声（前向），再让模型一步步把噪声还原成数据（反向）**。这个「渐进式去噪」让生成从「一次到位」变成「逐步精修」——训练稳定（无博弈）、生成质量超越 GAN，2020 年的 **DDPM**（Denoising Diffusion Probabilistic Models）让扩散模型成为图像生成的新王（Stable Diffusion、Midjourney 的基础）。

扩散模型的思想可以追溯到热力学：**墨水滴入清水会自然扩散开（前向）**；而「把扩散开的水墨重新聚成一滴」（反向）需要学习。扩散模型的「前向加噪」是固定的（数学给定），「反向去噪」是可学习的（神经网络）——**「学反向过程 = 学生成」**。本节把前向加噪、反向去噪、训练目标（预测噪声）讲透——它是当前生成模型浪潮的技术根基。<span class="marginnote">「扩散模型的物理灵感」：Sohl-Dickstein 等 2015 年首次把「热力学扩散」引入机器学习——「<strong>不可逆的扩散过程（加噪）是给定的，可逆的聚集过程（去噪）是可学的</strong>」。2020 年 DDPM 用「简单的预测噪声目标」把它变成实用的生成模型——「<strong>物理学的直觉 + 深度学习的实现</strong>」，是「AI for Science」反向影响「AI 自身」的经典案例。</span>

## 1 前向加噪：把数据一步步变成噪声

**前向过程（forward process）**：从一个数据样本 $\boldsymbol{x}_0$ 出发，逐步加高斯噪声，$T$ 步后变成纯噪声 $\boldsymbol{x}_T \sim \mathcal{N}(0, \boldsymbol{I})$：

$$
q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}) = \mathcal{N}\big(\boldsymbol{x}_t; \sqrt{1-\beta_t}\,\boldsymbol{x}_{t-1}, \beta_t\boldsymbol{I}\big)
$$

其中 $\beta_t$ 是**噪声调度（noise schedule）**——每步加多少噪声（从小到大递增）。

**关键性质（重参数化的便利）**：**一步到位**地算出「第 $t$ 步的加噪结果」（无需逐步迭代）：

$$
\boldsymbol{x}_t = \sqrt{\bar{\alpha}_t}\,\boldsymbol{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})
$$

其中 $\bar{\alpha}_t = \prod_{i=1}^{t}\alpha_i$（$\alpha_i = 1-\beta_i$）——**「给定 $\boldsymbol{x}_0$，任意步 $t$ 的加噪样本可直接采样」**。这是扩散模型训练「并行化」的关键。

**「前向过程是『确定性 + 噪声』的递推」**——它不需要学习，是「固定」的破坏过程。<span class="marginnote">「为什么加噪是可并行计算的」：如果必须「逐步加噪」才能得到 $\boldsymbol{x}_t$，训练要跑 $t$ 步前向（慢）；但高斯噪声的「可加性」让「$t$ 步加噪」可以「一步算完」（$\boldsymbol{x}_t$ 是 $\boldsymbol{x}_0$ 与一个噪声的线性组合）——「<strong>高斯分布的『闭合性』让扩散模型的训练可以并行</strong>」。</span>

**易错点：** 前向过程的「$T$」要足够大（通常 1000 步），让「$\boldsymbol{x}_T$ 几乎等于纯噪声」——**「加噪到『忘光』，反向才有意义」**。噪声调度 $\beta_t$ 的设计（线性、余弦）影响训练难度与生成质量。

## 2 反向去噪：学「把噪声拼回数据」

**反向过程（reverse process）**：从一个纯噪声 $\boldsymbol{x}_T$ 出发，逐步去噪，还原成数据 $\boldsymbol{x}_0$：

$$
p_\theta(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t) = \mathcal{N}\big(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_\theta(\boldsymbol{x}_t, t), \boldsymbol{\Sigma}_\theta(t)\big)
$$

**「反向过程是可学习的」**——$\boldsymbol{\mu}_\theta$ 是神经网络（预测「上一步该是什么」），方差 $\boldsymbol{\Sigma}_\theta$ 通常固定或简化为标量。

**生成 = 反向过程**：从 $\boldsymbol{x}_T \sim \mathcal{N}(0,\boldsymbol{I})$ 采样，$T$ 步去噪，得到 $\boldsymbol{x}_0$——**「生成就是反向去噪」**。

**关键简化（DDPM 的洞见）**：反向过程不需要预测「$\boldsymbol{x}_{t-1}$ 本身」，而是预测「**噪声 $\boldsymbol{\epsilon}$**」——因为 $\boldsymbol{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\boldsymbol{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon})$，「预测噪声」等价于「预测 $\boldsymbol{x}_0$ 的方向」。**「去噪网络学的是『把 $\boldsymbol{x}_t$ 里的噪声 $\boldsymbol{\epsilon}$ 分离出来』」**。<span class="marginnote">「预测噪声 vs 预测数据」：DDPM 用「预测噪声」得到极简单的训练目标（L2 回归噪声）；直觉上「预测噪声」比「预测数据」更稳定——「数据空间大而复杂，噪声空间小而简单（都是标准高斯）」。「<strong>学『分离噪声』比学『生成数据』更简单</strong>」——这是扩散模型「训练简单」的核心。</span>

**易错点：** 反向的「逐步去噪」是**串行**的（第 $t$ 步依赖第 $t+1$ 步）——**生成不能并行**（这是扩散模型「生成慢」的原因，DDIM 等加速采样见下一篇）。「训练并行（前向一步到位）、生成串行（反向逐步）」是扩散模型的节奏。

## 3 训练目标：预测噪声

**DDPM 的训练目标**极其简单——**预测「加进去的噪声」**：

$$
L = \mathbb{E}_{\boldsymbol{x}_0, t, \boldsymbol{\epsilon}}\Big[\big\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)\big\|^2\Big]
$$

其中 $\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)$ 是去噪网络对噪声的预测。**「给定 $\boldsymbol{x}_t$ 和时间步 $t$，预测当初加进去的噪声 $\boldsymbol{\epsilon}$」**——L2 回归。

**训练流程**：

1. 采样一个数据 $\boldsymbol{x}_0$。
2. 随机选一个时间步 $t$。
3. 采样噪声 $\boldsymbol{\epsilon}$，计算 $\boldsymbol{x}_t = \sqrt{\bar{\alpha}_t}\boldsymbol{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$。
4. 让网络从 $\boldsymbol{x}_t, t$ 预测 $\boldsymbol{\epsilon}$，最小化 L2 误差。

**「每个时间步都在训练」**——网络学会「在任意噪声水平下去噪」。这解释了为什么训练稳定：**「每个样本的梯度是独立的（无博弈、无对抗）」**。<span class="marginnote">「扩散模型训练稳定 vs GAN 不稳定」：扩散的每个训练样本「独立地」贡献梯度（预测噪声是回归任务），没有「两个网络互相博弈」的不稳定来源——「<strong>回归目标天生稳定，博弈目标天生不稳</strong>」。这个「稳定性」是扩散模型 2020 年后超越 GAN 的根本原因——「<strong>简单稳定的训练，胜过精巧不稳定的博弈</strong>」。</span>

**易错点：** 训练时「时间步 $t$」是**随机采样**的（从 $1..T$ 均匀采样）——网络必须学会「任意噪声水平」的去噪。**「随机时间步 = 覆盖所有去噪难度」**——这是「多尺度学习」的一种形式。

## 4 公式解析：为什么「预测噪声」等价于「去噪」

把「预测噪声」与「反向去噪」的等价关系写清楚。前向的闭合形式：

$$
\boldsymbol{x}_t = \sqrt{\bar{\alpha}_t}\,\boldsymbol{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}
$$

- **第一步，反解 $\boldsymbol{x}_0$**：移项得 $\boldsymbol{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\big(\boldsymbol{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}\big)$——**知道噪声 $\boldsymbol{\epsilon}$，就知道干净数据 $\boldsymbol{x}_0$**。
- **第二步，看「预测噪声」的意义**：$\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t) \approx \boldsymbol{\epsilon}$ 时，代入上式得「预测的干净数据 $\hat{\boldsymbol{x}}_0$」——**「预测噪声」=「隐式预测干净数据」**。
- **第三步，读反向的均值**：反向的均值 $\boldsymbol{\mu}_\theta(\boldsymbol{x}_t, t)$ 可以写成「用预测噪声构造的 $\hat{\boldsymbol{x}}_0$ 再向前加噪一步」——**「反向去噪 = 先估 $\hat{\boldsymbol{x}}_0$（预测噪声），再推 $\boldsymbol{x}_{t-1}$」**。<span class="marginnote">「预测噪声的『多尺度』含义」：不同 $t$ 对应不同的噪声水平——$t$ 小时（噪声少）预测「小噪声」（精修细节），$t$ 大时（噪声多）预测「大噪声」（恢复结构）——「<strong>一个网络学『从结构到细节』的全谱去噪</strong>」。这个「多尺度」让扩散模型既会「重建结构」又会「补细节」——是它生成质量高的关键。</span>

## 5 条件扩散：从 DDPM 到 Stable Diffusion

纯 DDPM 生成「随机图像」；实用的生成是「条件生成」——**条件扩散（conditional diffusion）**：

$$
\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t, \boldsymbol{c})
$$

其中 $\boldsymbol{c}$ 是条件（类别、文本）。**Stable Diffusion** 用「文本 → 图像」：把文本经 CLIP 编码成条件，注入去噪网络——「给定文本描述，生成对应图像」。

**现代形态**：

- **潜空间扩散（LDMs）**：在「低维潜空间」（而非像素空间）做扩散——省算力（Stable Diffusion 的基础）。
- **引导采样（guidance）**：分类器引导/无分类器引导——增强「条件遵循」（生成的图更贴文本）。
- **DDIM**：加速采样（下一篇）。

**「扩散模型的『条件化』继承自 cGAN 的范式」**——「条件生成」是现代生成模型的通用框架，扩散模型只是换了个「稳定的引擎」。<span class="marginnote">「Stable Diffusion 的技术栈」：它是「扩散模型 + 潜空间 + CLIP 条件 + 无分类器引导」的组合——每一项都在「扩散」这个骨架上加了「实用化」的组件。「<strong>扩散模型是引擎，条件注入与加速采样是驾驶系统</strong>」——理解 DDPM（本节）与 DDIM（下节），就看懂了 Stable Diffusion 的核心。</span>

**易错点：** 「条件扩散」的「条件」注入方式（拼接、cross-attention、AdaGN）影响「条件遵循度」——**「条件怎么注入」是条件生成模型的实现细节，但决定了「听不听话」**。现代文本到图像用「cross-attention」（文本条件与图像特征的注意力交互）注入。

## 6 小结

- **扩散模型**：前向加噪（固定、可并行）→ 反向去噪（可学习）——「生成 = 反向去噪」。
- **前向闭合形式**：$\boldsymbol{x}_t = \sqrt{\bar{\alpha}_t}\boldsymbol{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$——「一步算完任意步加噪」。
- **训练目标**：预测噪声 $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)\|^2$——「回归目标，天生稳定」（无博弈）。
- **预测噪声 = 隐式预测干净数据**——多尺度去噪（大噪声修结构、小噪声补细节）。
- **条件扩散**：$\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t, \boldsymbol{c})$——文本到图像（Stable Diffusion）的框架。
- 训练稳定 + 生成质量高——2020 年后取代 GAN 成为生成新王；生成串行（慢）→ DDIM 加速。

在下一节，我们解决扩散模型的「慢」——把「1000 步去噪」压缩到「几十步」，这就是 **DDPM 与 DDIM：采样加速**。
