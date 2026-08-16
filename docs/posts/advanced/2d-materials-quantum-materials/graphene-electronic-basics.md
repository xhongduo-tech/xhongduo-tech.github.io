---
title: 石墨烯电子学基础
date: 2026-08-07
---

# 石墨烯电子学基础

<div class="epigraph">
<p>石墨烯中的电子并不像普通金属那样被描述为薛定谔方程的解，而是由相对论性的狄拉克方程描述——它们在凝聚态物理中实现了粒子物理里无法实现的实验。</p>
<footer>—— K. S. Novoselov（Nobel Lecture: Graphene: Materials in the Flatland, 2011）</footer>
</div>

<div class="article-byline">
<p>第四级 · 二维材料与量子材料 ｜ Dresselhaus 等《Physics of Graphene》；Geim & Novoselov Nobel Lecture (Rev. Mod. Phys. 83, 2011) ｜ 2026-08-07</p>
</div>

## 为什么从石墨烯电子学开始

上一节我们认识了二维材料家族，其中石墨烯是「零号主角」——半金属、零带隙、蜂窝晶格。这一节要回答最核心的问题：**为什么石墨烯里的电子像无质量粒子？** 答案埋在一个让人意外的对比里：石墨烯中的电子用薛定谔方程描述会失败，而用相对论性的狄拉克方程描述反而精确成立，只不过光速 $c$ 换成了约 $c/300$ 的费米速度。这一发现让「相对论量子力学」第一次在桌面上唾手可及。<span class="marginnote">粒子物理里要造加速器才能看到狄拉克费米子；石墨烯里它们天生就在，且无需任何高速——这是「从极限到大模型」中「实验室就在指尖」的绝佳例证。</span>

本节对标 Dresselhaus 等人的《Physics of Graphene》与 Geim/Novoselov 的诺贝尔演讲，从紧束缚模型出发，一步步推出狄拉克锥、赝自旋与 Klein 隧穿。这一节的概念也是后续《输运性质》《拓扑量子材料》两篇的物理基础。

## 1 紧束缚模型：石墨烯的电子从哪里来

石墨烯的价电子构型是 $2s^2 2p^2$。蜂窝晶格中，碳原子以 $sp^2$ 杂化形成面内 $\sigma$ 键，剩下的一个 $p_z$ 轨道垂直于平面，形成 $\pi$ 键体系。**导电的主要是 $\pi$ 电子**：$\sigma$ 带的带宽大、远离费米能级，$\pi$ 带则在费米能级附近决定输运。

用**紧束缚近似（tight-binding approximation）**描述 $\pi$ 电子是最自然的起点。只考虑最近邻跃迁，哈密顿量写为：

$$\mathcal{H} = -t \sum_{\langle i,j\rangle, s} \left( a_{s}^{\dagger}(\mathbf{R}_i) b_s(\mathbf{R}_j) + \text{h.c.} \right)$$

其中 $a_s^{\dagger}$ 与 $b_s$ 分别是 A 子格与 B 子格上的产生、湮灭算符，$s$ 为自旋指标，$t \approx 2.8\,\text{eV}$ 是最近邻跃迁能。<span class="marginnote">$t$ 取正值，源于 $p_z$ 轨道与最近邻轨道之间的重叠积分，其大小约 2.8 eV 对应 3500 K 左右的能量尺度——远超室温，所以石墨烯的电子结构在室温下稳定可用。</span>

对波矢 $\mathbf{k}$ 作傅里叶变换，得到 $2 \times 2$ 的 Bloch 哈密顿量：

$$H(\mathbf{k}) = \begin{pmatrix} 0 & f(\mathbf{k}) \\ f^*(\mathbf{k}) & 0 \end{pmatrix}, \qquad f(\mathbf{k}) = -t \sum_{i=1}^{3} e^{i\mathbf{k}\cdot\boldsymbol{\delta}_i}$$

这里 $\boldsymbol{\delta}_i$ 是最近邻矢量，$\boldsymbol{\delta}_1 = (a/\sqrt{3})(0, 1)$ 等。对角化即得能带 $E_{\pm}(\mathbf{k}) = \pm \lvert f(\mathbf{k}) \rvert$。**矩阵的非对角元 $f(\mathbf{k})$ 是 A/B 两子格耦合的体现，正是它把「双原子基」翻译成了能带结构**——这也是上一节预告过的赝自旋的数学来源。

## 2 狄拉克点与线性色散

现在关键一步：求 $E_{\pm}(\mathbf{k})$ 的零点。计算 $\lvert f(\mathbf{k})\rvert = 0$ 会发现，它只在六角布里渊区的两个不等价顶点消失，即 **K 点与 K′ 点**（也叫谷，valley）：

$$\mathbf{K} = \left(\frac{2\pi}{3a}, \frac{2\pi}{3\sqrt{3}a}\right), \qquad \mathbf{K}' = -\mathbf{K}$$

在这两个点附近，能带是**线性的**。令 $\mathbf{k} = \mathbf{K} + \mathbf{q}$ 并把 $f(\mathbf{k})$ 对 $\mathbf{q}$ 展开，可得色散关系：

$$E_{\pm}(\mathbf{q}) \approx \pm \hbar v_F \lvert \mathbf{q} \rvert$$

其中费米速度 $v_F = \frac{\sqrt{3} a t}{2\hbar} \approx 10^6\,\text{m/s}$。<span class="marginnote">$v_F \approx c/300$，但即便打 90 折，电子在石墨烯中也可以无散射地跑过微米级距离——室温弹道输运是后面输运篇的主角。</span>

**重点：普通金属与半导体的能带在极值处是抛物线的 $E \propto \lvert\mathbf{k}\rvert^2$，而石墨烯在 K/K′ 点是锥形（圆锥）的 $E \propto \lvert\mathbf{q}\rvert$。** 线性色散意味着有效质量为零，电子与空穴的能带在狄拉克点相接但不断裂，二者之间没有带隙，也没有态密度零点处的平台——石墨烯因此是「零带隙半导体」，或更准确地说，是一种**半金属（semimetal）**。锥形色散在实验上被角分辨光电子能谱（ARPES）直接拍到，这是《表征技术》一篇会讲到的实验证据。

线性色散还带来一个反常的态密度：二维体系的态密度 $D(E) = \lvert E\rvert/(2\pi \hbar^2 v_F^2)$，在狄拉克点 $E=0$ 处恰好为零。这意味着**费米能级恰好在狄拉克点时，体系既无电子也无空穴，电导趋于一个最小值而非零**——这个「最小电导」约 $4e^2/(\pi h)$，是石墨烯输运中最先被测量的奇特量之一，也是输运篇要回访的对象。石墨烯的费米能级还可以通过背栅电场在导带、狄拉克点、价带之间连续扫过，这使得同一块样品既是电子导体又是空穴导体——场效应实验的最初动机正是这里。

## 3 无质量狄拉克费米子与赝自旋

线性色散最深刻的后果，是把石墨烯的低能电子描述为**无质量狄拉克费米子**。在 K 点附近，有效哈密顿量退化为：

$$H_{\text{eff}} = \hbar v_F \begin{pmatrix} 0 & q_x - i q_y \\ q_x + i q_y & 0 \end{pmatrix} = \hbar v_F \, \boldsymbol{\sigma} \cdot \mathbf{q}$$

这里的 $\boldsymbol{\sigma}=(\sigma_x,\sigma_y,\sigma_z)$ 是泡利矩阵，但作用对象不是真实自旋，而是 A/B 子格指标——故称**赝自旋（pseudospin）**。<span class="marginnote">赝自旋是石墨烯「身在凝聚态、心在相对论」的关键：把 A/B 子格自由度类比为「上/下自旋」，电子在晶格里的位置组合就被编码成了一个二能级系统。</span>对比狄拉克方程 $H = c\,\boldsymbol{\alpha}\cdot\mathbf{p} + mc^2\beta$：我们的方程里没有质量项 $mc^2\beta$，且光速被换成了 $v_F$。**石墨烯 = 无质量狄拉克方程的凝聚态实现**。

由 $H_{\text{eff}}$ 的本征方程可以证明：电子本征态满足

$$\boldsymbol{\sigma} \cdot \hat{\mathbf{q}} \, \lvert \psi \rangle = \pm \lvert \psi \rangle$$

即**赝自旋总是平行（或反平行）于动量方向**。这个「自旋锁定向动量的锁」，物理上叫**手性（chirality）**。它带来了两个著名推论：

- **贝里相位为 $\pi$**：电子绕狄拉克点走一圈，波函数获得 $\pi$ 相位，表现为电子干涉实验中不存在弱局域化（而是出现弱反局域化）——输运篇会细讲。
- **Klein 隧穿**：高势垒对无质量相对论粒子形同虚设，这是下一节的焦点。

这些性质可以放进一张核心对比表，与「普通抛物带电子」并排看，差异一目了然（本文属公式密集主题，本表作为知识锚点补充而非替代公式解析）：

| 性质 | 普通电子（抛物线能带） | 石墨烯狄拉克电子 |
| --- | --- | --- |
| 色散 | $E = \hbar^2 k^2/(2m^*)$ | $E = \pm \hbar v_F \lvert k\rvert$ |
| 有效质量 | 有限 $m^*$ | 零（无质量） |
| 波函数手性 | 无 | 赝自旋锁定动量 |
| 绕闭合回路的贝里相位 | 0 或 $2\pi$ | $\pi$ |
| 势垒隧穿 | 指数衰减 | 正入射 $T=1$（Klein） |
| 量子霍尔效应 | 整数量子化 $\sigma_{xy} = n e^2/h$ | 半整数量子化 $\sigma_{xy} = (n+\tfrac12)4e^2/h$ |

最后一行特别值得一提：石墨烯量子霍尔电导的平台出现在 $\pm 4(n+1/2)e^2/h$——其中 $4$ 来自自旋与谷的双重简并，$1/2$ 则正是零级朗道能级被手性保护、跨过狄拉克点的直接体现。这条半整数量子霍尔效应是 2005 年被实验确认的，也是「无质量狄拉克费米子」最硬的实验证据。

## 4 公式解析：从紧束缚能带推出费米速度

把「$E_{\pm} = \pm\hbar v_F \lvert \mathbf{q}\rvert$」这条核心公式拆成三步：

- **第一步，写出 $f(\mathbf{k})$ 并在 K 点展开**：$f(\mathbf{K}+\mathbf{q}) = -t\sum_i e^{i(\mathbf{K}+\mathbf{q})\cdot\boldsymbol{\delta}_i} = -t\sum_i e^{i\mathbf{K}\cdot\boldsymbol{\delta}_i}e^{i\mathbf{q}\cdot\boldsymbol{\delta}_i}$。由于 $e^{i\mathbf{K}\cdot\boldsymbol{\delta}_i}$ 的相位在三个最近邻上恰好为 $1, e^{i2\pi/3}, e^{-i2\pi/3}$，其和为零——这正是狄拉克点存在的原因。
- **第二步，对小 $\mathbf{q}$ 线性化**：对三个 $e^{i\mathbf{q}\cdot\boldsymbol{\delta}_i}$ 只保留到一阶，得到 $f \approx -t\, i \mathbf{q} \cdot \sum_i \boldsymbol{\delta}_i e^{i\mathbf{K}\cdot\boldsymbol{\delta}_i}$，求和结果是一个常复数矢量，即 $f \approx \frac{\sqrt{3} a t}{2}(q_x - i q_y)$ 的相因子。
- **第三步，取模长**：$\lvert f\rvert = \frac{\sqrt{3} a t}{2}\sqrt{q_x^2+q_y^2}$，代入 $a = 0.246\,\text{nm}$、$t = 2.8\,\text{eV}$、$\hbar$，得 $v_F = \sqrt{3}at/(2\hbar) \approx 10^6\,\text{m/s}$。

这条链条的妙处在于：**一个看似全新的物理量 $v_F$，其实完全由晶格常数与跃迁能两个已知量决定**。石墨烯电子学的「相对论」不是上帝给的，而是紧束缚模型在 K 点附近线性化的必然产物。

## 5 Klein 隧穿：势垒形同虚设

1929 年，Oskar Klein 发现相对论粒子穿越高势垒时透射率趋于 1——这就是著名的 **Klein 佯谬（Klein paradox）**，在真空中几乎无法直接验证，因为需要极高势垒。<span class="marginnote">正入射的狄拉克电子遇到势垒时，会以「空穴反粒子」的形式在势垒内部传播，从而以 100% 概率穿过——正粒子、反粒子配对使隧穿不再被指数压制。</span>

石墨烯让 Klein 佯谬成了日常实验。无质量狄拉克费米子正入射到任意高、任意宽的势垒，透射率恒为：

$$T = 1 \quad (\text{正入射})$$

而偏离正入射时透射率迅速衰减，$T$ 随入射角的平方下降。这意味着石墨烯势垒的「透明」是高度方向依赖的：像透镜一样只放行垂直入射的成分。这一效应已在微米级的 p-n 结实验中被直接观察到——电子几乎「无摩擦」地穿过原本应该阻挡它的能垒。<span class="marginnote">Klein 隧穿是「赝自旋守恒」的直接推论：势垒只要不翻转赝自旋，就不会反射电子。这也解释了为何石墨烯弹道输运能在室温长距离维持——输运篇会用量子霍尔效应再验证一次。</span>

## 6 双谷结构：K 与 K′ 简并

蜂窝晶格时间反演对称性保证 K 点与 K′ 点的能带严格简并，于是每个狄拉克点又带一个「谷」指标。四重简并（2 自旋 × 2 谷）是石墨烯低能电子学的完整图景。谷自由度可以作为信息载体——这就是**谷电子学（valleytronics）**的雏形，本专题《过渡金属硫族化合物》一篇会借 TMDC 展开。而 Berry 相位 $\pi$ 与赝自旋手性，则为《拓扑量子材料》一篇的贝里曲率铺好了道路。

值得强调的是，石墨烯的「零带隙」既是它的优点也是它的缺点：无带隙让载流子迁移率极高、狄拉克物理干净纯粹，但也让场效应晶体管无法关闭——这正是器件应用篇里「打开带隙」的诸多努力的根源。理解石墨烯，就是在理解一对天生的矛盾。

## 7 小结

- 石墨烯导电主要靠 $\pi$ 电子，紧束缚模型用跃迁能 $t\approx 2.8$ eV 描述最近邻耦合。
- 能带在 K/K′ 点线性交叉，色散为 $E_{\pm}=\pm\hbar v_F\lvert\mathbf{q}\rvert$，$v_F\approx 10^6$ m/s，是「无质量」的根源。
- 低能有效哈密顿量 $H=\hbar v_F\,\boldsymbol{\sigma}\cdot\mathbf{q}$ 是无质量狄拉克方程，$\boldsymbol{\sigma}$ 作用在 A/B 子格赝自旋上。
- 手性使赝自旋锁定动量，贝里相位为 $\pi$，电子干涉呈现弱反局域化。
- Klein 隧穿：正入射透射率 $T=1$，势垒对石墨烯电子几乎透明，且透射高度依赖入射方向。
- 四重简并（自旋 × 谷）与半整数量子霍尔效应 $\sigma_{xy}=(n+\tfrac12)4e^2/h$，是石墨烯「无质量」的直接实验签名。

在下一节，我们将从「材料是从哪来的」这一实际问题出发，系统比较**二维材料制备**的四种路线：机械剥离、CVD、分子束外延与化学插层剥离。那里会看到，把上一节的物理照进真实器件，第一步永远是先把一张「足够好」的单层造出来。
