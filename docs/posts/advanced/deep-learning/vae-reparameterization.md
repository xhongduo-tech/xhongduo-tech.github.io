---
title: 变分自编码器（VAE）与重参数化技巧
date: 2026-08-07
---

# 变分自编码器（VAE）与重参数化技巧

<div class="epigraph">
<p>我们无法直接看到本质，但可以让它按我们的方式显形。</p>
<footer>—— 依据变分推断的精神改写</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 花书《深度学习》§20.9、Kingma & Welling（2014） ｜ 2026-08-07</p>
</div>

## 为什么从变分自编码器开始

自编码器能「压缩-重建」，但它学出的表示 $\boldsymbol{h}$ 没有「概率分布」——你不能「采样一个表示，生成一张新图」。**变分自编码器（Variational Autoencoder, VAE）**（Kingma & Welling, 2014）给表示装上「概率」：假设隐变量 $\boldsymbol{z}$ 服从先验分布（如标准正态），模型学习「从 $\boldsymbol{z}$ 生成数据」——于是「**从先验采样 $\boldsymbol{z}$，就能生成新的数据**」。VAE 是第一个「既能学表示、又能生成」的深度生成模型。

VAE 的数学框架（变分推断 + 重参数化）看似复杂，但它的直觉非常清晰：**编码器学「把数据映射到隐分布」、解码器学「从隐变量重建数据」**，两者用「证据下界（ELBO）」联合优化。**重参数化技巧（reparameterization trick）**则是让「采样」可微的关键——没有它，VAE 无法用反向传播训练。本节把 VAE 的「生成模型视角」「ELBO」「重参数化」层层拆开——它是理解一切「隐变量生成模型」（扩散模型、流模型）的数学地基。<span class="marginnote">「VAE 的历史地位」：它是「深度生成模型」的第一块基石，与 GAN（同时期）一起开启了「生成模型时代」。它也是「变分推断 + 深度学习」的结合——把「贝叶斯推断」（第一级《概率论》与《最大似然估计》）的「变分法」用神经网络实现。「<strong>VAE = 贝叶斯推断 × 深度学习</strong>」——它是「深度概率建模」的代表作。</span>

## 1 从自编码器到生成模型：给隐变量一个分布

普通自编码器：$\boldsymbol{x} \to \boldsymbol{h} \to \hat{\boldsymbol{x}}$——$\boldsymbol{h}$ 是「确定性的压缩」。

**VAE 的转变**：假设数据由「隐变量 $\boldsymbol{z}$」生成：

1. **先验**：$\boldsymbol{z} \sim p(\boldsymbol{z})$（如 $\mathcal{N}(0, \boldsymbol{I})$）——「隐变量从标准正态采样」。
2. **似然**：$\boldsymbol{x} \sim p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})$（解码器）——「从隐变量生成数据」。
3. **后验**：$p(\boldsymbol{z} \mid \boldsymbol{x})$（编码器）——「给定数据，隐变量是什么分布」。

**「生成 = 从先验采样 $\boldsymbol{z}$，再经解码器生成 $\boldsymbol{x}$」**——这是 VAE 的生成视角：隐变量是「潜在原因」，数据是「可见结果」。

**学习目标**：最大化数据的边际似然 $p(\boldsymbol{x}) = \int p(\boldsymbol{x}\mid\boldsymbol{z})p(\boldsymbol{z})d\boldsymbol{z}$——但积分不可解（$\boldsymbol{z}$ 高维连续）。**「变分推断」出场**：用一个「可学习的后验近似 $q_\phi(\boldsymbol{z}\mid\boldsymbol{x})$」替代真实后验。<span class="marginnote">「为什么后验不可解」：$p(\boldsymbol{z}\mid\boldsymbol{x}) = \frac{p(\boldsymbol{x}\mid\boldsymbol{z})p(\boldsymbol{z})}{p(\boldsymbol{x})}$，分母 $p(\boldsymbol{x})$ 是「对所有 $\boldsymbol{z}$ 的积分」——高维积分不可计算。变分推断的应对：「<strong>不精确求后验，而是找一个『近似的』后验 $q_\phi$，让 $q_\phi$ 尽量接近真实后验</strong>」——「近似替代精确」是变分法的核心思想（第二级《最优化理论》的变分法同源）。</span>

**易错点：** VAE 的「编码器」学的是「后验的**分布**」（均值和方差），不是「确定的向量」——编码器输出 $\boldsymbol{\mu}_\phi(\boldsymbol{x}), \boldsymbol{\sigma}_\phi(\boldsymbol{x})$，表示「$\boldsymbol{z}$ 的条件分布 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$」。「<strong>编码器 = 后验分布，不是点估计</strong>」——这是 VAE 与自编码器的本质区别。

## 2 ELBO：VAE 的训练目标

VAE 最大化**证据下界（Evidence Lower Bound, ELBO）**：

$$
\log p(\boldsymbol{x}) \ge \underbrace{\mathbb{E}_{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}\big[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z})\big]}_{\text{重建项}} - \underbrace{D_{\text{KL}}\big(q_\phi(\boldsymbol{z}\mid\boldsymbol{x}) \,\|\, p(\boldsymbol{z})\big)}_{\text{正则项}}
$$

**两项的直觉**：

- **重建项**：让「从后验采样 $\boldsymbol{z}$ → 解码重建」的概率最大——「编码器给的隐变量要能重建出 $\boldsymbol{x}$」（与自编码器相同）。
- **KL 正则项**：让「后验 $q_\phi$」接近「先验 $p(\boldsymbol{z})$」——「隐变量分布不能跑太远，要贴标准正态」（这样「从先验采样也能生成」）。

**「重建 + 正则」的平衡**：重建项逼「表示有用」，KL 项逼「表示像先验」——两者拉扯，让隐变量「既有信息、又规则化」。**「VAE = 自编码器的重建目标 + 一个『分布约束』」**。<span class="marginnote">「ELBO 的名字」：因为 $\log p(\boldsymbol{x})$ = ELBO + KL(近似后验 || 真实后验) ≥ ELBO——ELBO 是「证据（$\log p(\boldsymbol{x})$）的下界」。最大化 ELBO = 「最小化 KL（近似后验距真实后验）」，同时「最大化重建」。「<strong>ELBO 同时做两件事：让近似后验靠近真实后验（推断），让重建更准（生成）</strong>」——一个目标、两个效果，这是 VAE 的优雅之处。</span>

**易错点：** ELBO 的「KL 项」可以**解析计算**（高斯对高斯有闭式），「重建项」用**蒙特卡洛采样**估计（从 $q_\phi$ 采样一个 $\boldsymbol{z}$ 算重建）——**「KL 解析、重建采样」**是 VAE 训练的标准做法。

## 3 重参数化技巧：让「采样」可微

VAE 的反向传播卡在一个地方：重建项要「从 $q_\phi(\boldsymbol{z}\mid\boldsymbol{x})$ 采样 $\boldsymbol{z}$」——但**「采样」不可微**（梯度无法穿过随机节点）。

**重参数化技巧（reparameterization trick）**：把「采样」改写为「确定性变换 + 外部噪声」：

$$
\boldsymbol{z} = \boldsymbol{\mu}_\phi(\boldsymbol{x}) + \boldsymbol{\sigma}_\phi(\boldsymbol{x}) \odot \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})
$$

**「从 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ 采样」 = 「从标准正态采样 $\boldsymbol{\epsilon}$，再线性变换」**——随机性被「挪」到 $\boldsymbol{\epsilon}$（外部），$\boldsymbol{\mu}, \boldsymbol{\sigma}$ 变成「确定性函数」，梯度可以穿过它们。<span class="marginnote">「重参数化的核心」：它把「随机性」从「参数路径」里剥离——$\boldsymbol{z}$ 的随机性全来自 $\boldsymbol{\epsilon}$（与参数无关），$\boldsymbol{\mu}, \boldsymbol{\sigma}$ 是「确定的」→ 梯度可以流过「$\boldsymbol{\mu}, \boldsymbol{\sigma}$ 到 $\boldsymbol{z}$」的路径。「<strong>『把随机性挪到外部』，让『参数路径』可微</strong>」——这个技巧让「随机模型的梯度训练」成为可能，是深度生成模型的「可训练性」基石（扩散模型的采样也有类似的处理）。</span>

**易错点：** 重参数化要求「采样分布」可以写成「确定性变换 + 简单噪声」——高斯分布可以（$\boldsymbol{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$），伯努利/离散分布**不能**（离散采样不可重参数化）。**「重参数化的可用性取决于分布」**——这也是 VAE 的隐变量用高斯的原因之一。

## 4 公式解析：重参数化的梯度流

把「重参数化为什么能让梯度流过」写清楚。VAE 的重建项梯度：

$$
\nabla_{\phi}\mathbb{E}_{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}\big[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z})\big]
$$

**不用重参数化**：梯度要「穿过采样」——$\nabla_\phi$ 无法作用在「$q_\phi$ 的样本」上（采样不可微）。

**用重参数化**：$\boldsymbol{z} = \boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi\odot\boldsymbol{\epsilon}$，于是期望变成「对固定的 $\boldsymbol{\epsilon}$」：

$$
\mathbb{E}_{\boldsymbol{\epsilon}\sim\mathcal{N}(0,\boldsymbol{I})}\big[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid \boldsymbol{\mu}_\phi(\boldsymbol{x}) + \boldsymbol{\sigma}_\phi(\boldsymbol{x})\odot\boldsymbol{\epsilon})\big]
$$

- **第一步，看期望的「转移」**：期望从「对 $q_\phi$ 采样」变成「对固定的 $\boldsymbol{\epsilon}$ 采样」——$\boldsymbol{\epsilon}$ 与参数无关。
- **第二步，看梯度的「畅通」**：$\log p$ 现在是「$\boldsymbol{\mu}_\phi, \boldsymbol{\sigma}_\phi$ 的确定性函数」——$\nabla_\phi$ 可以沿「$\phi \to \boldsymbol{\mu},\boldsymbol{\sigma} \to \boldsymbol{z} \to \log p$」的路径流畅计算。
- **第三步，读本质**：**重参数化把「随机梯度估计」变成「确定性路径 + 外部噪声的梯度」**——「随机性不进梯度路径」。<span class="marginnote">「重参数化 = 得分函数（REINFORCE）的替代」：没有重参数化，梯度要用的「对 $q_\phi$ 的对数导数」（REINFORCE/score function）——方差大、训练慢；重参数化给「低方差的梯度估计」。这个「<strong>低方差梯度估计</strong>」是深度生成模型训练的核心工程问题（VAE 用重参数化、GAN 用判别器、扩散用去噪目标——各有各的「低方差技巧」）。</span>

## 5 VAE 的能力、局限与现代形态

**VAE 的能力**：

- **生成**：从先验采样 $\boldsymbol{z}$，解码生成新数据。
- **表示**：编码器给出「数据的隐表示」（可用于下游任务）。
- **平滑潜在空间**：隐空间是「连续的、规则的」——插值两个 $\boldsymbol{z}$ 能生成「中间样本」（这是 VAE 相对 GAN 的优势之一）。

**VAE 的局限**：

- **生成模糊**：VAE 的重建用「逐像素独立」的高斯似然，倾向「平均所有可能」——生成图像常「模糊」。
- **先验太简单**：标准高斯先验不能匹配「真实数据的复杂隐结构」——「后验坍缩」等问题。

**现代形态**：

- **β-VAE**：加大 KL 项的权重，让隐变量更「解耦」（每个维度对应一个语义因子）。
- **VQ-VAE**：把隐变量离散化（codebook 量化）——生成清晰图像、也是「离散 token」生成（与 LLM 结合）的基础。
- **扩散模型**：把「去噪重建」推向「逐步去噪」——生成质量超过 VAE（第七篇《扩散模型》）。

**「VAE 的『概率生成』思想被一切后续生成模型继承」**——扩散模型、流模型都是「隐变量生成模型」的不同实现。<span class="marginnote">「VAE 的『模糊』问题」的根源：VAE 用「均方重建」训练，而 MSE 的「最优解」是「条件均值」——对「一张图可以是多个合法结果」的情形，MSE 会「平均」出「糊成一团的图」。这引出生成模型的「模式坍缩/模糊」根本矛盾：<strong>「分布 vs 均值」——生成模型要学「分布」（多样），MSE 逼它学「均值」（模糊）</strong>。扩散模型用「逐步去噪 + 每步预测噪声」绕过这个矛盾——「生成质量的跃迁，源自『不再用均值重建』」。</span>

**易错点：** VAE 的「重建项」可以是 MSE（连续数据）或交叉熵（二值数据）——**「似然的选择 = 对数据分布的假设」**（高斯 → MSE、伯努利 → BCE）。「重建项的形式，编码了『数据长什么样』的假设」。

## 6 小结

- **VAE**：隐变量生成模型——先验 $p(\boldsymbol{z})$、似然 $p(\boldsymbol{x}\mid\boldsymbol{z})$、变分后验 $q_\phi(\boldsymbol{z}\mid\boldsymbol{x})$。
- **编码器 = 后验分布**（输出均值/方差），解码器 = 生成——「从隐变量生成数据」。
- **ELBO** = 重建项 − KL 正则项——「表示有用」与「像先验」的平衡。
- **重参数化技巧**：$\boldsymbol{z} = \boldsymbol{\mu} + \boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$——把随机性挪到外部，让梯度可穿过。
- 能力：生成 + 表示 + 平滑隐空间；局限：生成模糊（MSE 逼均值）。
- 现代形态：β-VAE（解耦）、VQ-VAE（离散化）、扩散模型（逐步去噪）——「概率生成」的思想被继承。

在下一节，我们看另一条「生成之路」——不学「重建」，而是学「博弈」：生成器与判别器的对抗，这就是**生成对抗网络（GAN）：博弈与训练动态**。
