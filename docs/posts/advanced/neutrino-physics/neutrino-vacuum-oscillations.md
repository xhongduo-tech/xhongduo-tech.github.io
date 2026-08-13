---
title: 真空中的中微子振荡
date: 2026-08-07
---

# 真空中的中微子振荡

<div class="epigraph">
<p>中微子在旅行中忘记了自己的身份——这既不是缺陷，而是它们质量差异的指纹。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第四级 · 中微子物理 ｜ Giunti–Kim《Fundamentals of Neutrino Physics and Astrophysics》第6章 ｜ 2026-08-07</p>
</div>

## 为什么振荡公式是整个领域的心脏

PMNS 矩阵回答了「中微子由什么混合而成」，
真空振荡则回答「混合如何随时间与距离显现」。
振荡概率公式——尤其是两味形式 $P = \sin^2 2\theta\, \sin^2(\Delta m^2 L/4E)$——是全部中微子实验分析的工具箱：反应堆测消失、加速器测出现、太阳与大气测缺失，
最终都归结为拟合这些公式中的 $\Delta m^2$ 与 $\theta$。
理解它的推导，就理解了中微子物理的一半方法论。

## 1 传播中的相位演化

一个质量为 $m_i$ 的中微子以能量 $E$ 传播，
其量子态随时间演化携带相位 $e^{-i(Et - p_i L)}$。
在相对论极限（$m_i \ll E$）下展开：

$$E = \sqrt{p^2 + m_i^2} \approx p + \frac{m_i^2}{2E}$$

于是飞行距离 $L$ 后的相位为 $e^{-i m_i^2 L / 2E}$（忽略共同的整体相位）。
**关键点：不同质量态的相位差由质量平方差 $\Delta m^2$ 驱动**：

$$\phi_i - \phi_j = \frac{\Delta m_{ij}^2\, L}{2E}, \qquad \Delta m_{ij}^2 \equiv m_i^2 - m_j^2$$

中微子出发时是味态（三种质量态相干叠加），飞行中相位差累积，
到达探测器时叠加结果改变——**振荡就是干涉**。
<span class="marginnote">用「双缝」类比：$\nu_1$、$\nu_2$ 是两条「缝」，
出发时相干叠加成 $\nu_e$；
飞行到 $\phi$ 差达到 $\pi$ 时，
$\nu_1$ 与 $\nu_2$ 干涉相消，
$\nu_e$ 成分最小、$\nu_\mu$ 成分最大。
与光干涉的唯一区别：相位差不是光程差，而是<strong>质量差</strong>的累积。
</span>

## 2 两味振荡公式

为建立直觉，先看两味近似（如 $\nu_e \leftrightarrow \nu_\mu$）。
混合矩阵退化为一个角：

$$\nu_e = \cos\theta\, \nu_1 + \sin\theta\, \nu_2, \qquad \nu_\mu = -\sin\theta\, \nu_1 + \cos\theta\, \nu_2$$

设初始为纯 $\nu_e$。飞行距离 $L$ 后，
$\nu_1$ 与 $\nu_2$ 分别获得相位。
计算 $\lvert\langle \nu_\mu \mid \nu(L)\rangle\rvert^2$ 得到振荡概率：

$$P(\nu_\alpha \to \nu_{\beta\neq\alpha}) = \sin^2 2\theta\ \sin^2\!\left(\frac{\Delta m^2 L}{4E}\right)$$

这是中微子物理**最著名的一条公式**，值得逐项解剖：

- **$\sin^2 2\theta$（振幅因子）**：决定振荡「多强」。$\theta=45°$ 时 $\sin^2 2\theta=1$，完全互换；$\theta=0$ 或 $90°$ 时为零，不振荡。<span class="marginnote">注意振荡需要两个条件同时满足：<strong>非零混合角</strong>（$\sin^2 2\theta\neq 0$）与<strong>非零质量差</strong>（$\Delta m^2\neq 0$）。两者缺一，中微子就「不换装」。标准模型里中微子无质量（$\Delta m^2=0$），所以振荡的观测直接证明质量存在——这就是 1998 年历史性发现的逻辑。</span>
- **$\Delta m^2 L/4E$（相位因子）**：振荡「周期」。$\Delta m^2$ 越大或 $L/E$ 越大，相位演化越快。
- **存活概率**：$P(\nu_\alpha\to\nu_\alpha) = 1 - \sin^2 2\theta \sin^2(\Delta m^2 L/4E)$。实验测「消失」（存活<1）与「出现」（其他味>0）互为补充。

## 3 振荡长度与实验调谐

定义**振荡长度**（相位差 $2\pi$ 对应的距离）：

$$L_{\mathrm{osc}} = \frac{4\pi E}{\Delta m^2} \approx 2.48\ \mathrm{km}\ \frac{E[\mathrm{GeV}]}{\Delta m^2[\mathrm{eV^2}]}$$

这个量是实验设计的核心：**探测器距离应调到 $L \sim L_{\mathrm{osc}}$ 附近**，
使振荡在到达时充分发展。代入实际数字：

- 反应堆反中微子 $E\sim 3\ \mathrm{MeV}$、$\Delta m^2_{31}\sim 2.5\times10^{-3}\ \mathrm{eV^2}$：$L_{\mathrm{osc}}\approx 1.8\ \mathrm{km}$——所以大亚湾、双魁把探测器放在约 1–2 km 处测「大气标度」振荡。
- 同样反应堆、$\Delta m^2_{21}\sim 7.5\times10^{-5}\ \mathrm{eV^2}$：$L_{\mathrm{osc}}\approx 60\ \mathrm{km}$——KamLAND 选在约 180 km 处测「太阳标度」振荡。
- 大气中微子能量 GeV 级、穿越地球直径 $L\sim 10^4\ \mathrm{km}$：天然覆盖两个标度。

**辨析｜易错点：** 振荡长度公式里 $L/E$ 与 $\Delta m^2$ 是**互补**的：同样一个 $\Delta m^2$，
换一种 $L/E$ 就能看到完全不同的振荡图样。
**同一个实验的「标度灵敏度」由 $L/E$ 决定，
而不是单独由距离或能量决定。
** 这也是为什么长基线实验要「长基线」（L 大）配「高能量」（E 大）以保持合适的 $L/E$。

## 4 三味情形：消失与出现的完整图景

实际世界是三味振荡，但**当前参数下存在极好的近似**：

$$P(\nu_e \to \nu_e) \approx 1 - \sin^2 2\theta_{13}\,\sin^2\!\left(\frac{\Delta m_{31}^2 L}{4E}\right) - c_{13}^4\,\sin^2 2\theta_{12}\,\sin^2\!\left(\frac{\Delta m_{21}^2 L}{4E}\right)$$

这是「反应堆消失公式」：短基线（~1 km）时第二项（太阳标度）尚未发展，
主要看 $\theta_{13}$ 驱动的第一项；
长基线（~180 km）时两项叠加。
<span class="marginnote">三味振荡有<strong>两个独立的 $\Delta m^2$</strong>（$\Delta m^2_{21}$ 与 $\Delta m^2_{31}\approx\Delta m^2_{32}$），
对应两个振荡标度。
习惯上称「太阳标度」（$\Delta m^2_{21}$，
太阳与 KamLAND 主导）与「大气标度」（$\Delta m^2_{31}$，
大气、反应堆短基线、加速器主导）。
两者相差约 30 倍，所以近似总能成立。
</span>

加速器长基线实验测的则是**出现概率** $P(\nu_\mu\to\nu_e)$，
它同时依赖 $\theta_{13}$、$\theta_{23}$、$\Delta m^2_{31}$ 与 CP 相 $\delta$——这让出现通道成为探测 CP 破坏的关键，
详见本专题《三味振荡与CP破坏》。

## 5 公式解析：为什么是「$\sin^2$ 而不是 $\sin$」

推导两味公式时出现 $\sin^2$ 而非 $\sin$，
物理上大有讲究：

- **第一步，振幅叠加**：$\langle \nu_\mu|\nu(L)\rangle = -s c (e^{-i\phi_1} - e^{-i\phi_2})$。两个质量态振幅**相减**（源于混合矩阵的 $-s$ 元），差相位 $\Delta\phi$。
- **第二步，模方**：$\lvert -s c (e^{-i\phi_1}-e^{-i\phi_2})\rvert^2 = 4s^2c^2 \sin^2(\Delta\phi/2)$，其中 $4s^2c^2 = \sin^2 2\theta$。
- **第三步，代入相位**：$\Delta\phi/2 = \Delta m^2 L/4E$，得 $P=\sin^2 2\theta\,\sin^2(\Delta m^2 L/4E)$。

**$\sin^2$ 意味着概率总在 $0$ 与 $\sin^2 2\theta$ 之间振荡**：探测到的味成分周期性地回归与消失，
绝不会「单向转走」。
这带来一个实用的推论——**只要测量多个距离或能量的存活率，
就能同时提取 $\Delta m^2$（振荡频率）与 $\theta$（振荡幅度）**，
参数拟合由此成为可能。

## 6 小结

- 中微子振荡 = **质量态干涉**：不同质量态相位差 $\Delta m^2 L/2E$ 累积，味态叠加结果随之改变。
- 两味公式 **$P=\sin^2 2\theta\,\sin^2(\Delta m^2 L/4E)$**，需要非零混合角与非零质量差。
- 振荡长度 $L_{\mathrm{osc}}=4\pi E/\Delta m^2$：实验按 $L\sim L_{\mathrm{osc}}$ 调谐探测器距离。
- 三味振荡由**两个标度**（$\Delta m^2_{21}$ 太阳、$\Delta m^2_{31}$ 大气）主导，出现/消失通道互补。
- $\sin^2$ 结构让概率周期性回归，使实验能同时拟合 $\Delta m^2$ 与 $\theta$