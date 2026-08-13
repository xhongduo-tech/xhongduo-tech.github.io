---
title: 零差探测与量子态断层成像
date: 2026-08-07
---

# 零差探测与量子态断层成像

<div class="epigraph">
<p>要看清一个量子态，你不能只看它——你得在无数个方向上投影它，再把影子拼回原形。</p>
<footer>—— 沃格尔与里斯肯（U. Leonhardt），《测量量子态》一书的中心思想</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子光学 ｜ D. F. Walls & G. J. Milburn, Quantum Optics 第9章 ｜ 2026-08-07</p>
</div>

## 为什么从零差探测开始

光子计数只能数「有几个光子」，丢失了相位信息。
要同时抓住光场的振幅与相位——尤其是测量压缩态的正交分量、重构 
Wigner 
函数——需要**零差探测（homodyne detection）**。
它的原理：把待测信号光与一束强本地振荡（LO）在分束器上干涉，
用差分电流读出信号场在 LO 相位方向的投影。旋转 LO 相位，
就能扫遍相空间的所有方向；把每个方向的边缘分布收集起来，经逆 Radon 
变换，
就能**重构出整个量子态**——这就是**量子态断层成像（tomography）**。<span class="marginnote">断层成像的名字与医学 
CT 同源：CT 是绕着人体扫 X 光投影再重构截面，
量子断层是绕着相空间扫正交分量投影再重构 Wigner 函数——数学上都是 
Radon 逆变换。</span>

## 1 平衡零差探测的物理

把信号场 $\hat{a}_s$ 
与本地振荡 $\hat{a}_{LO} = |\alpha_{LO}|e^{i\theta}$（视为经典强场，$\alpha_{LO} = |\alpha_{LO}|e^{i\theta}$）在 
50:50 分束器上混合。两个输出口的场算符

$$\hat{a}_1 = \frac{1}{\sqrt{2}}(\hat{a}_s + \alpha_{LO}), \qquad \hat{a}_2 = \frac{1}{\sqrt{2}}(\hat{a}_s - \alpha_{LO})$$

两路光强分别正比于 $\hat{a}_1^\dagger\hat{a}_1$ 
与 $\hat{a}_2^\dagger\hat{a}_2$，差分电流

$$\hat{I}_- \propto \hat{a}_1^\dagger\hat{a}_1 - \hat{a}_2^\dagger\hat{a}_2 \approx |\alpha_{LO}|\left(\hat{a}_s e^{-i\theta} + \hat{a}_s^\dagger e^{i\theta}\right) = 2|\alpha_{LO}|\,\hat{X}_\theta$$

其中 **$\hat{X}_\theta = \frac{1}{2}(\hat{a}_s e^{-i\theta} + \hat{a}_s^\dagger e^{i\theta})$ 是信号场沿相位 $\theta$ 方向的正交分量**。<span class="marginnote">「平衡」二字来自两个探测器的对称结构：
LO 的经典噪声在差分中抵消，
只剩信号场的量子正交分量——这是零差探测能测到散粒噪声以下的量子信号的秘诀。</span>

**重点：调 LO 相位 $\theta$，就选择测量哪一个正交分量。** $\theta = 0$ 
测 $X_1$，$\theta = \pi/2$ 测 $X_2$，
中间相位测任意旋转方向。这是断层扫描的「投影方向旋钮」。

## 2 正交分量与 Wigner 边缘

在《量子相空间》一节我们已建立关键性质：**Wigner 函数的边缘积分 = 正交分量测量概率**：

$$P_\theta(x_\theta) = \int dx_{\theta + \pi/2}\, W(x_\theta, x_{\theta + \pi/2})$$

零差探测每次测量给出一个随机数 $x_\theta$，
大量重复测量统计出 $P_\theta(x_\theta)$——这正是 
Wigner 函数在 $\theta$ 方向的边缘分布。
测遍 $\theta \in [0, \pi)$，就得到一套边缘分布族。

对压缩真空，各方向的边缘分布都是高斯，但宽度随 $\theta$ 
变化：$\Delta x_\theta^2 = \frac{1}{4}(\cosh 2r - \sinh 2r\cos(2\theta - \phi))$。
测出不同方向的方差，即可读出压缩参数 $r$ 
与方向 $\phi$——**零差测量是压缩态的天然诊断仪**。<span class="marginnote">对高斯态（相干、
压缩、热），断层成像可以「偷懒」：只需测少数方向的方差，
整个态就由协方差矩阵确定。但非高斯态（Fock 态、
猫态）必须测满全角度。</span>

## 3 从边缘分布重构 Wigner 函数

**逆 Radon 变换**把边缘分布族还原为二维分布：

$$W(x_1, x_2) = \frac{1}{(2\pi)^2}\int_0^\pi d\theta \int_{-\infty}^{\infty} d\xi\, |\xi|\, \tilde{P}_\theta(\xi)\, e^{i\xi(x_1\cos\theta + x_2\sin\theta)}$$

其中 $\tilde{P}_\theta(\xi)$ 
是边缘分布的傅里叶变换。实际重构有三条路线：

- **逆 Radon 滤波反投影**：经典层析的标准算法，直接用于量子态；
- **最大似然估计（MaxLik）**：把重构当作参数估计，在物理态空间里找最可能的密度算符——当前首选，保证结果正定；
- **贝叶斯重构**：加入先验，估计误差棒，适合少量数据。

**辨析｜易错点：** 逆 Radon 
变换要求**完备的角覆盖**（$\theta$ 从 $0$ 
到 $\pi$）。只测两三个方向就声称「重构了 Wigner 
函数」是常见错误——那只能给出方差信息，画不出负值区域。Fock 态、
猫态等非高斯态的负 Wigner 值，
只有在满角覆盖下才能被忠实重建。<span class="marginnote">最大似然重构对测量装置的校准非常敏感：
探测效率 $\eta \lt  1$ 会把真空噪声掺进数据，直接压平 
Wigner 负值。实验中必须标定效率并做「反卷积」修正。</span>

## 4 公式解析：零差测量量 $\hat{X}_\theta = \frac{1}{2}(\hat{a}_s e^{-i\theta} + \hat{a}_s^\dagger e^{i\theta})$

这条式子定义「零差探测到底在测什么」，拆成三步：

**第一步，分束器混合**：$\hat{a}_1 = (\hat{a}_s + \alpha_{LO})/\sqrt{2}$、$\hat{a}_2 = (\hat{a}_s - \alpha_{LO})/\sqrt{2}$。强 LO 使 $\hat{a}_s^\dagger\alpha_{LO}$ 与 $\alpha_{LO}^*\hat{a}_s$ 项远大于其他项，差分时 LO 强度项 $\propto |\alpha_{LO}|^2$ 与 $\alpha_{LO}^*\hat{a}_s^\dagger\hat{a}_s$ 项相消。
**第二步，读出正交分量**：留下 $\propto \alpha_{LO}\hat{a}_s^\dagger e^{i\theta} + \alpha_{LO}^*\hat{a}_s e^{-i\theta} = 2|\alpha_{LO}|\hat{X}_\theta$。因子 $\frac{1}{2}$ 保证 $\hat{X}_\theta$ 与 $\hat{X}_{\theta+\pi/2}$ 满足不确定关系 $\Delta X_\theta\Delta X_{\theta+\pi/2} \geq 1/4$（$\hbar = 1$）。
- **第三步，经典类比**：$\hat{X}_\theta$ 是场的「以 $\theta$ 为参考相位」的振幅分量——正好对应经典光电场 $\propto \cos(\omega t + \theta)$ 的投影。零差测量把量子场的正交分量「翻译」成经典电流，供电子学分析。

## 5 断层成像的应用与极限

- **量子态验证**：重构 Wigner 函数、计算保真度、见证纠缠——量子光学实验的「期末考」；
- **压缩/纠缠定量**：直接读出压缩参数、EPR 方差积；
- **量子计算读出**：连续变量量子计算的测量基就是正交分量，断层成像是其校准工具；
- **单光子态重构**：SPDC 预报的单光子态重构出负的 Wigner 函数——非经典性的「照片」；
- **实时状态监测**：快速零差 + 实时重构已用于量子反馈与量子增强计量。

**物理极限**：探测效率 $\eta$、暗计数、电子噪声、LO 
相位噪声共同决定重构保真度。理想极限下，
断层成像给出态的完整描述——但它消耗大量重复测量，
且对非高斯态需要指数级资源。
这就是「量子态层析的指数困境」——与量子计算本身的复杂性一样，
源自态空间维度指数膨胀。<span class="marginnote">层析的采样复杂度（需 $O(1/\epsilon^2)$ 
次测量达到保真度 $1-\epsilon$）是量子表征理论的经典结论，
也是「量子验证」研究的起点。</span>

## 6 小结

- 平衡零差探测：信号 + 强 LO 在分束器混合，差分电流读出 $\hat{X}_\theta$。
- LO 相位 $\theta$ 选择测量方向；$\theta \in [0,\pi)$