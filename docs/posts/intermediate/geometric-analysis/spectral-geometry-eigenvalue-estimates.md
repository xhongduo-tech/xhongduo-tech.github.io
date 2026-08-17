---
title: 谱几何（特征值估计、Cheeger 不等式、等周常数）
date: 2026-08-07
---

# 谱几何（特征值估计、Cheeger 不等式、等周常数）

<div class="epigraph">
<p>「你能听出鼓的形状吗？」</p>
<footer>—— 马克 · 卡茨（Mark Kac），《American Mathematical Monthly》（1966）</footer>
</div>

<div class="article-byline">
<p>第二级 · 几何分析 ｜ Peter Li《Geometric Analysis》特征值章 ｜ Jost 谱章 ｜ 2026-08-07</p>
</div>

## 为什么从谱几何开始

热核的短时渐近展开把特征值交给几何，反过来的问题——**「听出鼓的形状」**——问谱（所有特征值）能否决定流形的几何。答案大体是否定的（Milnor 的反例），但谱几何真正的力量在另一方向：**用几何量给特征值划界**。特征值的上下界、Cheeger 不等式、等周常数，构成「几何 ↔ 分析」最精确的计量表，也是前面所有工具（比较定理、热核、Sobolev）的汇流处。

从课程体系看，本篇把《流形上的椭圆算子》《热方程与热核》《Sobolev 空间与 PDE 工具》三篇整合成「谱」的完整图像，并首次把**等周常数（isoperimetric constant）**当作主角——它是把「几何规模」翻译成「谱规模」的汇率。它与第一级《变分法》、第四级《泛函分析》中算子谱理论自然衔接。

<span class="marginnote">卡茨 1966 年的经典文章《Can One Hear the Shape of a Drum?》提出了这个著名问题。1992 年 Gordon、Webb、Wolpert 用构造性反例（两个保谱但不同胚的平面区域）回答了「听不出」。但对黎曼流形，谱仍编码海量几何信息——Weyl 律给体积、短时展开给曲率积分，这就是「谱 → 几何」的正向方向，也是本专题的主旋律。</span>

## 1 Laplace 算子谱与 Rayleigh 商

设 $(M,g)$ 紧致无边。Laplace–Beltrami 算子（正的，见《流形上的椭圆算子》篇）的谱是离散的：存在可数特征值

$$0 = \lambda_0 < \lambda_1 \le \lambda_2 \le \cdots \to \infty$$

对应正交完备的特征函数 $\{\varphi_k\}$。第一个正特征值 $\lambda_1$ 是谱几何的主角，它有变分刻画——**Rayleigh 商（Rayleigh quotient）**

$$\lambda_1 = \inf_{u \perp \text{常数}} \frac{\int_M |\nabla u|^2\,dV}{\int_M u^2\,dV}$$

**几何意义**：$\lambda_1$ 是「函数在流形上能振荡得多剧烈」的度量——$\lambda_1$ 大意味着任何非平凡函数都必须迅速变化，即流形「紧密」；$\lambda_1$ 小意味着流形「空旷」或有「瓶颈」。<span class="marginnote">$\lambda_1$ 与物理联系：鼓膜最低固有频率、弦的最低泛音、量子的基态能量（除零点）都是同一个 Rayleigh 商。圆盘半径 $R$ 时 $\lambda_1 = j_{0,1}^2/R^2$（$j_{0,1}$ 是 Bessel 函数第一零点），球面 $S^n$ 时 $\lambda_1 = n$——这些显式值充当比较的标尺。</span>

## 2 Cheeger 不等式与等周常数

**Cheeger 常数（Cheeger constant）** 把「瓶颈」量化：对紧流形 $M$，

$$h(M) = \inf_{\Omega} \frac{\operatorname{Vol}_{n-1}(\partial\Omega)}{\min\{\operatorname{Vol}(\Omega), \operatorname{Vol}(M\setminus\Omega)\}}$$

其中 $\Omega$ 遍历 $M$ 的正则子区域（边界测度与体积用诱导度量）。$h$ 是「把一个区域切下来所需面积与区域体积之比」的下确界——瓶颈越小，$h$ 越小。

**Cheeger 不等式（Cheeger's inequality）** 给出 $\lambda_1$ 的下界：

$$\lambda_1 \ge \frac{h(M)^2}{4}$$

证明概要：设 $u$ 是 $\lambda_1$ 的特征函数，对水平集 $\{u > t\}$ 用余面积公式（coarea formula）与 Cheeger 常数的定义，把 $\int|\nabla u|^2$ 与 $h^2\int u^2$ 挂钩。直观：**若流形有细瓶颈（$h$ 小），则存在变化缓慢的函数（特征函数被瓶口拉平），故 $\lambda_1$ 小**；反过来 $\lambda_1$ 控制着瓶口能多细。

**等周常数（isoperimetric constant）** $I(M)$ 是 Cheeger 常数的「同族兄弟」，要求更强的指数形态：$\operatorname{Vol}(\partial\Omega) \ge I\,\min\{\operatorname{Vol}\Omega, \operatorname{Vol}(M\setminus\Omega)\}^{1-\frac1n}$。在 Ricci 下界 + 体积上界下，$I$（进而 $h$）有一致的下界——这正是前面 Bishop–Gromov 体积比较的用武之地：**几何曲率下界 → 等周下界 → $\lambda_1$ 下界**。<span class="marginnote">Cheeger 不等式证明里的「余面积公式」是几何测度论的核心：$\int_M |\nabla u|\,g(u)\,dV = \int g(t)\, \operatorname{Vol}_{n-1}(\{u=t\})\,dt$，它把「梯度积分」翻译成「水平集面积积分」，正是把 $\lambda_1$ 与 $h$ 联系起来的钥匙。</span>

## 3 特征值估计：从下界到上界

**定理（$\lambda_1$ 下界，Li–Yau / Gromov）**：设 $\operatorname{Ric} \ge (n-1)K$，$K\le0$，$\operatorname{diam}\le D$，则存在常数 $c_n > 0$ 使

$$\lambda_1 \ge \frac{c_n}{D^2\,e^{c_n\sqrt{-K}\,D}}$$

这个估计的路线图是完整地串联前几篇的工具链：

- 用 Bishop–Gromov 体积比较控制体积增长；
- 体积增长给出等周常数 $I$ 的下界；
- 等周下界 ⇒ Sobolev/Nash 不等式（常数与 $D, K$ 挂钩）；
- Nash 不等式 ⇔ 热核上界 ⇔ 谱下界（$e^{-\lambda_1 t}$ 是热核长时间的主导项）。

**上界估计**同样深刻：**Cheng 的极大值原理 + 比较**给出 $\lambda_1 \le$（模型空间球的谱），且对 $\operatorname{Ric}\ge(n-1)K$ 有（Cheng, 1975）

$$\lambda_1(M) \le \lambda_1(B_{\kappa})$$

其中 $B_\kappa$ 是截面曲率 $\kappa$ 的模型空间测地球。**$\lambda_1$ 被正曲率压低、被负曲率抬高**——这是「曲率 → 谱」的双向绑定。Yang–Yau 与 Korevaar 的估计则在固定亏格曲面上给 $\lambda_1$ 上界。

**$\lambda_1$ 的可积性视角**：$\lambda_1$ 也可以由热核长时间行为读出：$e^{-\lambda_1 t} \sim \frac{1}{\operatorname{Vol}(M)}$（$t\to\infty$ 主导项）。反过来，给定 $\lambda_1$ 下界与体积，热核的长时渐近被钉住——这是「谱 → 热核 → 几何」环路的最后一环。

**Weyl 渐近律（Weyl's asymptotic law）**给出整体谱的分布：对 $N(\lambda) = \#\{\lambda_k \le \lambda\}$，

$$N(\lambda) \sim \frac{\omega_n}{(2\pi)^n}\,\operatorname{Vol}(M)\,\lambda^{n/2}, \qquad \lambda \to \infty$$

其中 $\omega_n$ 是单位球体积。**谱的密度决定体积**——这是「听出鼓的大小」的正向答案。更精细的热核迹展开（Minakshisundaram–Pleijel）再把曲率积分也「听」出来。<span class="marginnote">Weyl 律是谱几何的「宪法」：它保证谱携带体积信息。Milnor（1964）给出两个保谱但不等距的 16 维闭流形，说明谱不决定几何；但「$\lambda_1$ 有界 + 谱分布」这类联合信息在现代（Cheeger–Müller、热核不变量）仍不断给出新的刚性结果。</span>

## 4 公式解析：Cheeger 不等式

把 Cheeger 不等式当作一条公式解析来拆解，看它的每一步如何「从几何走向谱」：

$$\lambda_1 \ge \frac{h^2}{4}$$

- **第一步，起点**：设 $u$ 是 $\lambda_1$ 的特征函数，取两个水平集 $\Omega_+ = \{u > 0\}$，$\Omega_- = \{u < 0\}$。由定义 $\lambda_1$ 是 Rayleigh 商的下确界，只需对合适的函数证明「范数比 ≥ $h^2/4$」。
- **第二步，余面积公式**：把 $\int_{\{u>0\}}|\nabla u|^2$ 分解为水平集面积积分；每一层的面积被 $h$ 与「该层下方的体积」夹住：$\operatorname{Vol}_{n-1}(\{u=t\}) \ge h\,\min\{V_+(t), V_-(t)\}$。
- **第三步，两个平凡估计**：利用 $|u|$ 的分布函数，把「$\min\{V_+,V_-\}$」与「$\int u^2$」通过一维不等式 $\int V(t)\,dt \ge \frac12\int u^2$ 之类的积分比较连起来。
- **第四步，组装**：得到 $\int|\nabla u|^2 \ge \frac{h^2}{4}\int u^2$，正是 $\lambda_1 \ge h^2/4$。

**直觉内核**：瓶颈（$h$ 小）允许「跨瓶口缓变」的特征函数，压低了 $\lambda_1$；反之流形无瓶颈（$h$ 大）则任何函数都必须在短距离内变化，$\lambda_1$ 必须大。**Cheeger 常数是谱下界的「瓶颈测度」，等周常数是它的积分版。**

## 5 谱几何的现代走向

- **特征函数与结点集**：特征函数 $\varphi_k$ 的零集（结点）分割流形；**结点定理**（Courant）说第 $k$ 个特征函数的结点把流形分成至多 $k$ 个区域。谱 → 拓扑结构。
- **$\lambda_1$ 的极值问题**：固定体积/亏格，找最大化 $\lambda_1$ 的度量——**这是当前活跃的谱几何优化问题**（Nadirashvili 的圆盘猜想、等谱问题与模空间）。
- **谱收敛**：Cheeger–Gromov 流形收敛理论中，谱序列的收敛性是「几何收敛」的伴随信号；$\lambda_1$ 的连续性刻画了流形收敛的精细程度。
- **与数据科学**：图拉普拉斯与 Cheeger 不等式是谱聚类（spectral clustering）、流形学习（拉普拉斯特征映射）的数学根基——这是几何分析通向当代机器学习（含大模型的嵌入）最直接的接口。

| 谱量 | 几何输入 | 典型估计 | 方向 |
| --- | --- | --- | --- |
| $\lambda_1$ 下界 | $\operatorname{Ric}\ge(n-1)K$，直径 $D$ | $\lambda_1 \ge c_n/(D^2 e^{c_n\sqrt{-K}D})$ | 几何 → 谱 |
| $\lambda_1$ 上界 | 曲率下界、模型球 | Cheng：$\lambda_1 \le \lambda_1(B_\kappa)$ | 几何 → 谱 |
| $N(\lambda)$ | 体积 | Weyl：$N(\lambda)\sim\frac{\omega_n}{(2\pi)^n}\operatorname{Vol}\,\lambda^{n/2}$ | 谱 → 几何 |
| 等周/Cheeger | 瓶颈、等周常数 | $\lambda_1 \ge h^2/4$ | 几何 → 谱 |

<span class="marginnote">「谱 → 几何」的反方向（卡茨的问题）虽然整体否定，但局部的谱不变量（Weyl 律的体积项、短时展开的曲率积分）异常丰富。在现代数据科学中，谱聚类就是把「Cheeger 不等式」当作切割图的最优算法依据——鼓的听觉在这里变成了网络的分割，见第三级《图论与数据结构》中谱方法一节的交叉。</span>

**辨析｜易错点：** Cheeger 不等式的方向是 $\lambda_1 \ge h^2/4$（下界）；它的逆不等式 $\lambda_1 \le C h^2$ 对图成立（Cheeger–Alon–Milman 双向界），但对黎曼流形**不**普遍成立。另外 $\lambda_1$ 对直径的依赖是指数的（上式中的 $e^{c_n\sqrt{-K}D}$），不能只记多项式版本。

**术语速查**：

| 记号 / 术语 | 含义 | 要点 |
| --- | --- | --- |
| 特征值 $\lambda_k$ | $0=\lambda_0<\lambda_1\le\lambda_2\to\infty$ | 离散谱，特征函数正交完备 |
| Rayleigh 商 | $\inf_{u\perp 1}\frac{\int\|\nabla u\|^2}{\int u^2}$ | $\lambda_1$ 的变分刻画 |
| Cheeger 常数 $h$ | $\inf\frac{\mathrm{Vol}_{n-1}(\partial\Omega)}{\min\mathrm{Vol}}$ | 瓶颈的度量 |
| 等周常数 $I$ | 面积 $\ge I\cdot(\text{体积})^{1-1/n}$ | Bishop–Gromov 给出下界 |
| 余面积公式 | $\int\|\nabla u\| g(u)dV = \int g(t)\mathrm{Vol}_{n-1}(\{u=t\})dt$ | 谱 ↔ 等周 的钥匙 |
| Weyl 律 | $N(\lambda)\sim\frac{\omega_n}{(2\pi)^n}\mathrm{Vol}\,\lambda^{n/2}$ | 谱密度决定体积 |
| 谱聚类 | 图拉普拉斯 + Cheeger 不等式 | 谱几何 → 数据科学接口 |

## 6 小结

- **谱**：$0=\lambda_0<\lambda_1\le\lambda_2\to\infty$，Rayleigh 商刻画 $\lambda_1$——振荡有多剧烈。
- **Cheeger 常数** $h$：瓶颈的度量，$\lambda_1 \ge h^2/4$（Cheeger 不等式）。
- **等周常数**：由 Bishop–Gromov 体积比较给出下界，喂给 Sobolev/Nash ⇒ 谱下界。
- **Weyl 律**：$N(\lambda)\sim\frac{\omega_n}{(2\pi)^n}\operatorname{Vol}\,\lambda^{n/2}$，谱密度决定体积。
- **双向绑定**：曲率下界压低 $\lambda_1$（Cheng），谱聚类把 Cheeger 不等式带入数据科学。

在下一节，我们把所有线索汇合，走向几何分析的前沿——**Perelman 工作概览、正质量定理与广义相对论中的几何分析**，为这个专题画上一个望向未来的句点。
