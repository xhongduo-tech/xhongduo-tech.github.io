---
title: 扩散模型基础：DDPM
date: 2026-08-11
---

# 扩散模型基础：DDPM

<div class="epigraph">
<p>万物流转，无物常驻（Everything flows, nothing stands still）。</p>
<footer>—— 赫拉克利特（Heraclitus）</footer>
</div>

<div class="article-byline">
<p>第四级 · 高阶专题 · DiT / Sora（论文解析） ｜ 对标教材 Ho et al., Denoising Diffusion Probabilistic Models (2020) §2 ｜ 2026-08-11</p>
</div>

## 为什么从「加噪—去噪」开始

大模型这个词在今天几乎等于「语言模型」，可 2024 年让世界惊叹的 Sora，生成的是视频。Sora 的底座不是 GPT 式的自回归，而是**扩散模型（diffusion model）**：先让一段纯噪声慢慢「长出」画面。要读懂 DiT 和 Sora，第一步就是吃透它们的引擎——由 Jonathan Ho、Ajay Jain 与 Pieter Abbeel 在 2020 年提出的 **DDPM（Denoising Diffusion Probabilistic Models，去噪扩散概率模型）**。<span class="marginnote">DDPM 发表于 NeurIPS 2020，但它的思想源头可追溯到 2015 年 Sohl-Dickstein 等人的「非均衡热力学视角的深度无监督学习」——把统计物理里的扩散过程搬进了深度学习。</span>

本章是《DiT / Sora》专题的发动机。上一级课程里我们见过 GAN 与 VAE：GAN 靠「造假者与鉴定者的博弈」，VAE 靠「把图像压进隐变量再解码」。扩散模型走出第三条路——**不学一次生成，而学 T 次去噪**。这条路的直觉很简单，数学却干净得惊人，而 Sora 的所有技术都建立在它的三条公式之上。

## 1 前向过程：有序地打散信息

想象把一滴墨汁滴进一杯水。墨会自发地扩散，颜色越来越淡，最终均匀充满整杯水；这个过程你无法靠搅动「收回」墨滴。扩散模型的前向过程（forward process）就是把这个物理过程做成可计算的版本：**从干净数据 $x_0$ 出发，逐小步加入高斯噪声，T 步之后图像彻底淹没在噪声里。**

形式化地说，前向过程是一条马尔可夫链<span class="marginnote">马尔可夫链：下一步的状态只依赖当前状态、与更早的历史无关。这一假设让前向过程可以被写成一串简单的转移，是概率论与随机过程（第二级《概率论与数理统计》）的核心概念。</span>：

$$
q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1}), \qquad
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right)
$$

其中 $\beta_t \in (0,1)$ 是预设的**方差表（variance schedule）**，控制每一步加多少噪声。DDPM 采用线性表：$\beta_1 = 10^{-4}$ 逐渐增大到 $\beta_T = 0.02$，$T = 1000$ 步。每走一步，信号按 $\sqrt{1-\beta_t}$ 缩小一点、噪声按 $\sqrt{\beta_t}$ 注入一点，信息被「有秩序地」稀释，而不是一下子打烂。

**DDPM 最重要的数学捷径是：任意时刻 $t$ 的加噪结果，可以不用一步步累乘，而是从 $x_0$ 一步算出来。** 定义 $\alpha_t = 1 - \beta_t$，$\bar\alpha_t = \prod_{s=1}^t \alpha_s$，则边缘分布可以写成一个高斯：

$$
q(x_t \mid x_0) = \mathcal{N}\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\right)
$$

再用**重参数化技巧**（从高斯采样写成「均值 + 标准差 × 标准正态」）把它变成生成式训练最爱用的形式：

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

这行公式是整个扩散模型的地基。它的含义是：**前向过程根本不需要跑 $t$ 次，一个公式就能把 $x_0$ 变成任意噪声水平的 $x_t$。** 剩下的工作就是教网络做反方向的事。

## 2 反向过程：学会「倒放」

如果向前是打散，那生成就是**倒放**：从 $t=T$ 的纯噪声 $x_T \sim \mathcal{N}(0,I)$ 出发，一步步「预测」上一步更干净的图像，直到 $x_0$。问题是 $q(x_{t-1} \mid x_t)$ 我们不知道——好在当 $\beta_t$ 很小时，反向转移也近似是高斯，于是可以拿一个神经网络去拟合它：

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1};\ \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t)\right)
$$

DDPM 的关键洞察是：**让网络直接预测噪声 $\epsilon$ 比预测均值 $\mu$ 更容易、效果也更好。** 因为在反向公式里均值由噪声线性表出：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right)
$$

于是整个模型变成一个「去噪器」$\epsilon_\theta$：输入带噪图像 $x_t$ 和时间步 $t$，输出它估计混进去的那份噪声。训练好之后，采样就是反复执行「用预测的噪声把 $x_t$ 擦回 $x_{t-1}$」：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\right) + \sigma_t z,\qquad z \sim \mathcal{N}(0,I)
$$

最后一步 $x_0$ 就是生成结果。<span class="marginnote">与 VAE、GAN 相比，扩散模型不需要对抗训练，损失函数是朴素的回归；代价是采样要走 1000 步——后来的 DDIM、加速采样器都在解决「怎么少走几步」。这一对比详见第三级《生成模型》专题。</span>

一个直觉坐标有助于记住整套参数：**信号噪声比（SNR）**。在第 $t$ 步，信号部分 $\sqrt{\bar\alpha_t}x_0$ 的能量正比于 $\bar\alpha_t$，噪声部分 $\sqrt{1-\bar\alpha_t}\epsilon$ 的能量正比于 $1-\bar\alpha_t$，于是

$$
\mathrm{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t}
$$

训练初期 $t=1$ 时 $\bar\alpha_t \approx 1$，信噪比极高，图像几乎还是原样；$t=T$ 时 $\bar\alpha_t \to 0$，信噪比趋近于零，图像彻底被噪声淹没。**扩散模型本质上是在「信噪比从极高到极低」的整条谱带上均匀地学习去噪**——这就是为什么时间条件 $t$ 必须显式告诉网络「现在信噪比有多低」，也是为什么方差表 $\beta_t$ 的选择如此关键：它决定了这 1000 步里，信号衰减与噪声注入的节奏。

## 3 公式解析：一张图看懂 DDPM 的损失

在动笔拆公式之前，值得先把「为什么网络不直接生成图像」的直觉立住：网络并不输出图像，而是输出「这份噪声里混的是什么」。当 $t$ 很小时，噪声几乎透明，预测它等于在学图像本身的细节；当 $t$ 很大时，图像已面目全非，预测噪声近乎在猜随机数。把这两端的技能压进同一个网络，恰好逼出了一种由粗到细的图像理解能力——这也解释了为什么扩散模型擅长生成高保真图像。

训练目标非常干净，用标准的变分下界（ELBO，与 VAE 同源）可以推出来；但 Ho 等人发现，把各项权重剥掉之后的**简化损失**训练最稳、效果最好：

$$
L_{\mathrm{simple}}(\theta) = \mathbb{E}_{t,\,x_0,\,\epsilon}\left[\left\|\epsilon - \epsilon_\theta\left(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t\right)\right\|^2\right]
$$

拆成三步看：

- **第一步，看输入 $x_t$ 怎么来**：括号里正是重参数化公式 $\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$。训练时随机采一个干净图 $x_0$、随机采一个噪声 $\epsilon$、随机采一个时间步 $t$，一步拼出「$t$ 时刻的带噪图」。
- **第二步，看网络要预测什么**：$\epsilon_\theta(x_t, t)$ 是去噪器输出的「猜测噪声」，$t$ 以正弦嵌入的形式作为条件喂进网络——这就是**时间步条件**，它告诉网络「你现在面对的是轻噪声还是重噪声」。<span class="marginnote">同一个网络要处理 1000 种噪声水平，所以「时间条件」必不可少；DiT 的核心贡献之一，就是把这种条件注入升级成了更聪明的 AdaLN-Zero，见本专题第三篇。</span>
- **第三步，看损失是什么**：真实噪声 $\epsilon$ 与预测噪声 $\epsilon_\theta$ 的均方误差——一次朴素的回归。期望对 $t$ 均匀取样 $t \sim \{1,\dots,T\}$，对每个噪声水平都做「去噪比赛」。网络在训练里唯一要学的，就是**在每个噪声水平下认出自己混进去的那份噪声**。

这条损失把「生成图像」这个听起来玄幻的任务，翻译成了「把网络训练成一个精确的噪声解算器」——而噪声的均方误差，就是第一级微积分里再熟悉不过的平方损失。

## 4 DDPM 的位置：从像素到 latent 再到视频

理解 DDPM 之后，就能看清一条清晰的升级线：

- **像素级扩散**：DDPM 直接在 $256 \times 256 \times 3$ 的像素上做扩散。训练贵、采样慢、高分辨率力不从心。
- **latent 扩散（LDM）**：Rombach 等人 2022 年提出用 VAE 把图像先压进低维隐空间（latent space），再在 latent 上跑扩散——这就是 Stable Diffusion 的底座。DiT 正是接在这条线上：它处理的输入是 $32 \times 32 \times 4$ 的 latent，而非像素。
- **视频扩散**：Sora 把「压缩网络 + 扩散 Transformer」整套搬到了视频上，压缩的维度从 2D 变成 3D（空间 × 时间）。

所以 DDPM 不是终点，而是所有后续工作的**公共引擎**。理解了这三条公式——前向加噪、反向去噪、简化损失——你手里就握住了 Sora 的钥匙。

最后补一句采样视角：DDPM 原文用 1000 步采样，速度很慢；后续的 DDIM 把去噪路径改造成确定性的常微分方程离散化，几百步甚至几十步就能出图，代价是质量与多样性略降。到了视频这类「天然贵」的任务，加速采样更是刚需——60 秒视频逐帧走 1000 步是天文数字，采样器与缩放律在工程上同样重要。DiT 正是在采样端与模型端同时优化，才把扩散生成推进到了可商用的水准。

## 5 小结

- 扩散模型分**前向过程**（逐小步加噪，信息有序打散）与**反向过程**（网络逐小步去噪，从纯噪声长出图像）。
- 重参数化公式 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ 让任意时刻的加噪可以**一步算出来**，无需跑 $T$ 步。
- 网络学习的目标是**预测噪声** $\epsilon$ 而非均值，损失是最朴素的均方误差 $L_{\mathrm{simple}}$。
- 时间步 $t$ 以嵌入形式作为条件注入网络，是扩散模型「条件机制」的起点。
- 从像素级 DDPM 到 latent 扩散（LDM）再到视频扩散（Sora），是一条连续的升级线。

在下一节，我们将回答这个问题：**为什么 2023 年 Peebles 与 Xie 决定把扩散模型背后的 U-Net 换成 Transformer？** 这就是 DiT——把「去噪器」做成可缩放的通用架构。
