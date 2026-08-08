---
title: DDPM 与 DDIM：采样加速
date: 2026-08-07
---

# DDPM 与 DDIM：采样加速

<div class="epigraph">
<p>走 1000 步是对耐心的考验，走 50 步才是对智慧的考验。</p>
<footer>—— 依据采样加速的精神改写</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ Ho 等《DDPM》（2020）、Song 等《DDIM》（2021） ｜ 2026-08-07</p>
</div>

## 为什么从 DDPM 与 DDIM 开始

DDPM 生成本质上要跑 **$T$ 步反向去噪**（$T$ 常取 1000）——**生成一张图要做 1000 次网络前向**，慢得难以实用（Stable Diffusion 早期生成一张图要几十秒）。**DDIM（Denoising Diffusion Implicit Models）**（Song 等, 2021）在不重新训练的前提下，把采样步数从 1000 压到 **20–50 步**——速度提升 20–50 倍，图像质量几乎不损。它通过两个关键思想实现：**跳步采样**（不必每一步都走）与**确定性采样**（DDIM 的生成过程是确定性的，还额外获得了「可复现」与「潜空间插值」能力）。

理解 DDIM，等于理解「扩散模型的采样加速」——这是扩散模型从「实验室」走向「产品」（实时生成）的关键工程。本节把 DDPM 采样慢的原因、DDIM 的跳步与确定性原理、以及「采样加速」的完整谱系（DDIM、DPM-Solver、LCM）讲透。<span class="marginnote">「DDIM 的『implicit』」：DDIM 的全名是「Denoising Diffusion Implicit Models」——「隐式」指它的采样过程不依赖「马尔可夫链」（DDPM 的逐级 Markov 假设），而是定义一个「隐式」的确定性轨迹——「<strong>不重新训练，只改采样方式，就能大幅加速</strong>」。这个「训练与采样解耦」的洞见，让「采样加速」成为独立的研究方向。</span>

## 1 为什么 DDPM 采样慢：每一步都要跑网络

DDPM 的生成流程：

1. 采样 $\boldsymbol{x}_T \sim \mathcal{N}(0, \boldsymbol{I})$。
2. 对 $t = T, T-1, \dots, 1$：用去噪网络算 $\boldsymbol{x}_{t-1} = \boldsymbol{\mu}_\theta(\boldsymbol{x}_t, t) + \sigma_t \boldsymbol{z}$。

**每一步都要一次网络前向**——1000 步 = 1000 次前向。而「网络」通常是 U-Net（几十亿参数）——**一次前向就很贵，1000 次贵得离谱**。

**为什么 DDPM 要 1000 步？** 因为 DDPM 假设「反向是马尔可夫链」——每步只能走「一小步」（噪声从 $\boldsymbol{x}_t$ 到 $\boldsymbol{x}_{t-1}$ 只减少 $\beta_t$）。**「马尔可夫的『小步走』约束了步长」**——这是 DDPM 慢的结构性原因。<span class="marginnote">「马尔可夫链 vs 非马尔可夫」：DDPM 的反向是「马尔可夫」的（$\boldsymbol{x}_{t-1}$ 只依赖 $\boldsymbol{x}_t$，且每步只能走一步的噪声）；DDIM 打破了「马尔可夫」约束——「$\boldsymbol{x}_{t-1}$ 可以依赖『预测的 $\hat{\boldsymbol{x}}_0$』，从而跳过中间步骤」。「<strong>打破马尔可夫假设 = 打破每步只能走一小步的限制</strong>」。</span>

**易错点：** DDPM 的「每步都要跑网络」不是「采样算法笨」，而是「马尔可夫链的结构限制」——「加速的关键是『打破链式限制』，不是『优化每步的计算』」。

## 2 DDIM 的核心思想：跳步 + 确定性

**DDIM 的两个关键改动**：

**改动一：跳步采样（skip steps）**。DDPM 每步从 $\boldsymbol{x}_t$ 走 $\boldsymbol{x}_{t-1}$（必须相邻）；DDIM 允许从 $\boldsymbol{x}_t$ **直接跳到 $\boldsymbol{x}_s$**（$s < t$，任意间隔）。做法：用「预测噪声」先估 $\hat{\boldsymbol{x}}_0$，再从 $\hat{\boldsymbol{x}}_0$ 直接「加回」到 $\boldsymbol{x}_s$：

$$
\boldsymbol{x}_s = \sqrt{\bar{\alpha}_s}\,\hat{\boldsymbol{x}}_0(\boldsymbol{x}_t) + \sqrt{1-\bar{\alpha}_s}\,\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)
$$

**「跳步」的合法性**：因为「预测噪声 $\boldsymbol{\epsilon}_\theta$ 已包含了对 $\boldsymbol{x}_t$ 的全部信息」——从 $\hat{\boldsymbol{x}}_0$ 出发，可以直接构造任意步 $\boldsymbol{x}_s$，无需一步步走。**「从 1000 步里均匀抽 20 步，就能近似完成去噪」**。<span class="marginnote">「跳步的直觉」：去噪网络已经「学会」了「任意噪声水平」的去噪（训练时随机 $t$ 都训练），所以「直接跳到大步幅」是「网络本来就会的技能」——「<strong>训练时的『随机时间步』让网络天然支持任意步幅采样</strong>」。DDIM 只是把这个「能力」用于「更少的采样步」。</span>

**改动二：确定性采样（deterministic sampling）**。DDIM 的反向更新**去掉随机噪声项**：

$$
\boldsymbol{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{\boldsymbol{x}}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\,\boldsymbol{\epsilon}_\theta
$$

对比 DDPM 的 $\boldsymbol{x}_{t-1} = \boldsymbol{\mu}_\theta + \sigma_t\boldsymbol{z}$（含随机 $\boldsymbol{z}$）——**DDIM 的采样是确定性的**（同样的 $\boldsymbol{x}_T$ 生成同样的 $\boldsymbol{x}_0$）。

**确定性带来的两个额外好处**：

1. **可复现**：同一种子 → 同一张图（利于调试与「编辑」）。
2. **潜空间插值**：两个 $\boldsymbol{x}_T$ 的插值 → 对应生成的插值——「**在噪声空间里做语义插值**」（生成「猫与狗的中间体」）。<span class="marginnote">「确定性采样的『语义插值』」：DDIM 的确定性让「噪声 $\boldsymbol{x}_T$」成为一个「有语义的潜变量」——$\boldsymbol{x}_T$ 的微小变化对应生成的微小语义变化。于是「$\boldsymbol{x}_T^A$ 与 $\boldsymbol{x}_T^B$ 的插值」生成「A 与 B 的中间体」——这个「潜空间插值」是「图像编辑」（Stable Diffusion 的 img2img、风格混合）的数学基础。DDPM 的随机采样做不到这点（每次生成都换一个随机轨迹）。</span>

**易错点：** DDIM 的「确定性」是「给定 $\boldsymbol{x}_T$」的确定性——如果你换 $\boldsymbol{x}_T$（重新采样噪声），生成的图当然不同。**「确定性是『同种子同结果』，不是『每次生成都一样的图』」**。

## 3 公式解析：DDIM 的采样更新

把 DDIM 的单步更新写成数学。给定 $\boldsymbol{x}_t$，先预测噪声 $\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)$，估出干净数据：

$$
\hat{\boldsymbol{x}}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\big(\boldsymbol{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_\theta\big)
$$

再「跳」到 $\boldsymbol{x}_s$：

$$
\boldsymbol{x}_s = \sqrt{\bar{\alpha}_s}\,\hat{\boldsymbol{x}}_0 + \sqrt{1-\bar{\alpha}_s - \sigma_s^2}\,\boldsymbol{\epsilon}_\theta + \sigma_s\boldsymbol{z}
$$

- **第一步，看 $\hat{\boldsymbol{x}}_0$**：从「预测噪声」反解「干净数据」——「网络已经看到 $\boldsymbol{x}_t$，它推断的『去噪后的样子』」。
- **第二步，看跳步**：$\boldsymbol{x}_s$ 是「$\hat{\boldsymbol{x}}_0$ 向前加噪到 $s$ 步」——**只要 $s < t$，无论差多少步，都可以直接构造**（这就是「跳步」）。
- **第三步，看随机项**：$\sigma_s$ 控制「随机性」——$\sigma_s = 0$ 时是**纯确定性 DDIM**；$\sigma_s > 0$ 时退化为「带噪声的 DDPM 风格采样」。**「$\sigma_s$ 是『确定性 ↔ 随机性』的旋钮」**——0 给确定性（可复现、可插值），>0 给多样性（随机生成）。<span class="marginnote">「DDIM 的『随机性旋钮』」：$\sigma_s=0$（纯 DDIM）生成「确定、可复现但多样性略低」；$\sigma_s>0$ 增加随机性、多样性更高但损失确定性。「<strong>确定性 vs 多样性</strong>」的取舍在采样里是永恒的主题（与《序列采样》的温度类似）——DDIM 用 $\sigma_s$ 给了这个「旋钮」。</span>

## 4 采样加速的完整谱系

DDIM 开启了「采样加速」的研究，此后方法层出不穷：

| 方法 | 加速方式 | 步数 | 特点 |
| --- | --- | --- | --- |
| DDPM | 原始逐级采样 | 1000 | 基准（慢） |
| **DDIM** | 跳步 + 确定性 | 20–50 | 不重训、可复现 |
| DPM-Solver | 高阶数值求解器 | 10–20 | 用「常微分方程求解」加速 |
| LCM | 一致性蒸馏 | 1–4 步 | 重新蒸馏训练 |
| 蒸馏（progressive） | 网络蒸馏 | 1–8 步 | 重训、可实时 |

**「两步走」的加速哲学**：DDIM/DPM-Solver 是「**不改训练、只改采样**」（免费加速）；LCM/蒸馏是「**重新训练一个更快的生成器**」（牺牲训练换生成速度）。**「先试免费加速，不够再重训」**是实践顺序。<span class="marginnote">「DDIM 与『常微分方程（ODE）』的连接」：DDIM 的确定性采样可以被理解为「在『概率流 ODE』上做数值积分」——它每步走「欧拉步」。这个「采样 = ODE 求解」的视角，让「数值求解器」方法（DPM-Solver）得以应用——「<strong>把采样加速变成『数值分析』问题</strong>」（与第二级《数值分析》的 ODE 数值解同源）。</span>

**易错点：** 「加速采样」≠「加速训练」——DDIM 只加速「生成（采样）」；训练仍要完整的 1000 步加噪目标。「<strong>训练慢是训练的问题，生成慢是采样的问题，两者可以分别优化</strong>」。

## 5 从加速到实时：扩散模型的实用化

采样加速让扩散模型「可用」：

**Stable Diffusion**：DDIM 的 50 步 → 一张图 1–2 秒（普通 GPU）。
**实时生成**：LCM 的 1–4 步 → 一张图 0.1 秒（实时绘画、视频生成）。
**图像编辑**：DDIM 的「潜空间插值」→ 编辑、风格迁移、img2img。

**「采样加速是扩散模型『上桌』的关键工程」**——没有它，「生成一张图要 1000 步」的扩散模型只能躺在论文里。「<strong>把模型变快 20 倍，往往比把模型变好 20% 更重要</strong>」——采样加速是「工程让科学可用」的典范。

**易错点：** 加速幅度与质量有「甜点」——DDIM 到 10 步以下质量下降明显；LCM 的 1–4 步要「蒸馏训练」来保证质量。**「步数与质量的权衡」是采样加速的永恒主题**——「越快越糊，越慢越好」的边界由任务决定。<span class="marginnote">「步数-质量曲线」是采样研究的核心评估：横轴是采样步数、纵轴是生成质量（FID）——「<strong>在尽量少的步数下达到尽量低的质量损失</strong>」是加速方法的目标。DDIM 的 20–50 步在曲线上是「甜点」（质量损失小、加速大）；更少的步数需要「蒸馏」来维持质量——「<strong>加速与质量的帕累托前沿</strong>」。</span>

## 6 小结

- **DDPM 采样慢**：马尔可夫链「每步只能走一小步」——1000 步 = 1000 次网络前向。
- **DDIM**：**跳步采样**（从 $\hat{\boldsymbol{x}}_0$ 直接构造任意步）+ **确定性采样**（去掉随机噪声）——20–50 步，质量几乎不损。
- DDIM 的数学：先预测噪声 → 反解 $\hat{\boldsymbol{x}}_0$ → 跳到 $\boldsymbol{x}_s$——「跳步」的合法性来自「训练时随机时间步」。
- $\sigma_s$ 是「确定性 ↔ 多样性」的旋钮——确定性给「可复现 + 潜空间插值」。
- 加速谱系：DDIM（免费）/DPM-Solver（ODE 求解）/LCM（蒸馏）——「先试免费加速，不够再重训」。
- 采样加速是扩散模型「可用」的关键——「把模型变快 20 倍，比变好 20% 更重要」。

在下一节，我们换一条「生成路线」——用「可逆变换」精确计算概率，这就是**流模型与归一化流（Normalizing Flow）**。
