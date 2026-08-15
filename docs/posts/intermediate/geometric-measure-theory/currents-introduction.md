---
title: 电流（currents）引论
date: 2026-08-07
---

# 电流（currents）引论

<div class="epigraph">
<p>曲面是微分形式的奴隶：它由「在它上面积分什么」所唯一决定。</p>
<footer>—— 自 H. Federer, *Geometric Measure Theory\*（意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 几何测度论 ｜ H. Federer, *Geometric Measure Theory\*, Ch.4 ｜ 2026-08-07</p>
</div>

## 为什么从 current 开始

上一节的 varifold 用「位置 + 切空间」的测度化来换取极限的封闭性。电流（current）是另一套、也是更经典的一套管法：**干脆不把曲面当作「点的集合」，而是当作「在它上面可以做积分」的泛函。** 一支 $m$ 维电流，是一个把「$m$ 次微分形式」映成实数的连续线性泛函——积分 $\int_T \omega$ 本身就是曲面 $T$ 的身份证明。这个「对偶视角」有三个不可替代的好处：定向被天然纳入（反对称形式的对偶），边界算子 $\partial$ 是现成的（由 Stokes 公式 $\partial T(\omega) = T(\mathrm{d}\omega)$ 定义），且**紧致性定理**（Federer–Fleming）保证「质量有界的电流列必有弱收敛子列」。电流因此成为极小曲面问题、以及 Plateau 问题证明的标准框架。<span class="marginnote">「电流」这个名字来自物理学：像电流一样，它携带「方向 + 强度 + 流动」的信息；英文 current 兼有「当前」与「流动」两义，Federer 取其「流动」意。</span>

## 1 从微分形式对偶出发

先回顾微分形式。$\mathbb{R}^n$ 上的 $m$ 次微分形式（$m$-form）是反对称 $m$-线性形式场，记作 $\omega$；在光滑曲面 $M$ 上，积分 $\int_M \omega$ 是经典微积分的内容。电流把这件事倒过来：

**核心概念（$m$ 维电流）**：$\mathbb{R}^n$ 上的一个 **$m$ 维电流（$m$-current）** $T$，是紧支撑光滑 $m$-形式空间上的连续线性泛函。即对每个光滑 $m$-形式 $\omega$，给一个实数 $T(\omega)$，且 $T$ 对 $\omega$ 是线性的、在分布意义下连续。<span class="marginnote">「电流 = 形式的对偶」与泛函分析里「分布 = 测试函数的对偶」同构：测度论研究对象从「集合」升格为「作用在测试对象上的泛函」，从而获得极限封闭性。0 维电流就是对偶于 0-形式（函数）的分布。</span>

**核心概念（质量）**：电流 $T$ 的**质量（mass）**定义为

$$
\mathbf{M}(T) \;=\; \sup\left\{ T(\omega) : \|\omega\|_\infty \le 1 \right\}
$$

其中 $\|\omega\|_\infty \le 1$ 指形式的逐点范数不超过 1。质量度量「$T$ 有多大」，是面积概念的抽象化。<span class="marginnote">质量是电流的「面积」：对整值电流（见第 3 节），$\mathbf{M}(T)$ 恰等于带重数的 $\mathcal{H}^m$ 面积。质量有界是紧致性定理的前提，也是变分问题的自然约束。</span>

## 2 边界算子：Stokes 公式变成定义

电流理论最优雅的一步，是用 Stokes 公式**定义**边界。光滑曲面的 Stokes 公式说 $\int_{\partial M} \omega = \int_M \mathrm{d}\omega$。

**核心概念（电流的边界）**：电流 $T$ 的**边界**是 $(m-1)$ 维电流 $\partial T$，定义为

$$
\partial T(\omega) \;=\; T(\mathrm{d}\omega), \qquad \text{对一切 } (m-1)\text{-形式 } \omega
$$

这个定义把 Stokes 公式从「定理」变成「边界算子的定义」：**$\partial$ 是 $\mathrm{d}$ 的（负）伴随。** 由 $\mathrm{d}^2 = 0$ 立刻得到

$$
\partial^2 T \;=\; 0
$$

即「边界的边界为零」——拓扑学中经典的圈-边界同调关系在电流框架里自动成立。<span class="marginnote">$\partial^2 = 0$ 是整个同调理论的基石。电流提供了「奇异链」的测度论替代：电流的同调类可以完全用积分定义，而紧致性定理保证每个同调类里有面积最小的电流代表元——这正是 Plateau 问题的解。</span>

**辨析｜易错点：** 电流的边界与原点的定向、重数息息相关。$\partial(\partial T) = 0$ 是恒等式，但「$\partial T$ 的质量 = 边界面积」不等于「$T$ 内部结构」——一条边界相同的电流可以有不同的内部。极小曲面问题正是「固定边界、最小化内部质量」。

## 3 整流电流与整值电流

电流太宽泛，需要挑出「由曲面长出来」的子类，就像 varifold 里挑整流 varifold。

**核心概念（整流电流）**：$m$ 维电流 $T$ 称为**整流电流（rectifiable current）**，如果存在 $m$ 维整流集 $M$、整数重数 $\theta \in L^1_{\mathrm{loc}}(M)$ 与可测定向 $\vec{\xi}$（$M$ 的 $m$ 向量切场），使得

$$
T(\omega) \;=\; \int_M \langle \omega(x),\, \vec{\xi}(x)\rangle\; \theta(x)\; \mathrm{d}\mathcal{H}^m(x)
$$

若 $T$ 与 $\partial T$ 都是整流电流，则 $T$ 称为**整值电流（integral current）**。<span class="marginnote">整流电流对应「可数片定向曲面」：$\vec{\xi}$ 给每片定向，$\theta$ 给每片层数（整数）。整值电流额外要求边界也是整流电流——这保证了「曲面没有无限粗糙的边缘」，是 Federer–Fleming 紧致性成立的舞台。</span>

**重点：整值电流的紧凑定理（Federer–Fleming）**：若一族整值电流的质量与边界质量一致有界（$\mathbf{M}(T_i) + \mathbf{M}(\partial T_i) \le C$），则存在子列弱收敛到一个整值电流。这条定理把「面积有界 + 边界有界」的序列压出极限，是变分法里「存在性」的标准证明引擎。<span class="marginnote">弱收敛的精确含义是 $T_i(\omega) \to T(\omega)$ 对每个紧支撑光滑形式成立。质量与边界质量的双重有界性保证了极限不「漏质量」也不「长毛边」——两个条件缺一不可。</span>

## 4 公式解析：面积最小化的存在性证明

用电流语言，Plateau 问题（给定边界曲线 $\Gamma$，找张在 $\Gamma$ 上面积最小的曲面）的证明骨架如下。

**第一步，把问题放进电流框架**：设 $\Gamma$ 是一条可求长曲线（1 维整值电流），考虑集合

$$
\mathcal{C} \;=\; \{\, T : T \text{ 是整值电流},\; \partial T = \Gamma \,\}
$$

面积最小化 = 在 $\mathcal{C}$ 中求 $\mathbf{M}(T)$ 的最小值。

**第二步，取极小化序列**：取 $T_i \in \mathcal{C}$ 使 $\mathbf{M}(T_i) \to \inf_{T \in \mathcal{C}} \mathbf{M}(T)$。由于 $\partial T_i = \Gamma$ 固定，$\mathbf{M}(\partial T_i) = \mathbf{M}(\Gamma)$ 有界，质量又有界，Federer–Fleming 紧致性给出子列 $T_{i_j} \rightharpoonup T_*$。

**第三步，验证极限是解**：边界在弱收敛下连续（$\partial T_* = \lim \partial T_{i_j} = \Gamma$），质量在弱收敛下是下半连续的（$\mathbf{M}(T_*) \le \liminf \mathbf{M}(T_{i_j}) = \inf$），因此 $T_* \in \mathcal{C}$ 且 $\mathbf{M}(T_*)$ 达到下确界。**$T_*$ 就是 Plateau 问题的面积最小化曲面。**

**重点：存在性靠「紧致性 + 下半连续」两件套，而不是构造。** 这个「先取极小化序列、用紧致性拿极限、用下半连续性验证」的三段式，是整个变分法的通用引擎。<span class="marginnote">经典光滑变分法无法证明 Plateau 存在性，因为极小化序列在光滑曲面类里没有极限（会皱缩成非光滑对象）；电流框架把曲面类扩大到包含极限，存在性便自动成立。正则性（极限是否光滑）是另一回事——见第 10 篇。</span>

## 5 电流、varifold 与同调

电流与 varifold 是两套互补的框架，选择取决于问题。

**电流**：携带**定向**与**整数重数**，有边界算子 $\partial$ 与同调结构。适合有向问题：极小子流形、Plateau 问题、几何测度论中的同调表示。
**varifold**：不要求定向，携带**切空间分布**与**实数重数**。适合无向问题：肥皂膜、平均曲率流、无向极值曲面。

两者都能做变分与紧致性，但**边界**是电流独有：varifold 没有自然的边界算子，而电流的 $\partial$ 让它直接进入同调理论。<span class="marginnote">在同调里，$T \mapsto \partial T$ 定义链映射，整流电流 / 整值电流构成链复形，其同调与奇异同调同构（对相当广的系数）。「每个同调类有面积最小代表元」——这是 Federer 与 Fleming 最早证明 Plateau 问题的路线，也是「几何 + 拓扑」结合的范例。</span>

**比较表**：电流与 varifold 的系统对照。

| 特征 | $m$ 维电流 | $m$ 维 varifold |
| --- | --- | --- |
| 定义 | 形式的对偶泛函 | $G(n,m) \times \mathbb{R}^n$ 上测度 |
| 方向 | 定向 $m$-向量 $\vec{\xi}$ | 无向切空间 $T_xM$ |
| 重数 | 整数（整值电流） | 非负实数 |
| 边界 | $\partial T$ 有定义 | 无 |
| 同调 | 有（$\partial^2 = 0$） | 无 |
| 紧致性 | 质量 + 边界质量有界 | 质量有界 |

## 6 电流的切片与几何应用

电流不仅有边界，还可以**切片**：对整值电流 $T$ 与 Lipschitz 函数 $u$，可定义「$T$ 沿水平集 $u = t$ 的切片」$\langle T, u, t \rangle$，它满足**切片公式**

$$
\int \mathbf{M}(\langle T,u,t\rangle)\; \mathrm{d}t \;\le\; \mathrm{Lip}(u)\; \mathbf{M}(T)
$$

这个公式是余面积公式（第 6 篇）在电流语言里的对应：把「沿目标切片」变成「把电流切成水平集层」。切片在几何分析里至关重要：证明极小曲面正则性、估计奇异集维数，都要把电流切成低维层再逐层分析。<span class="marginnote">切片算子 $\langle T,u,t \rangle$ 的直观：把电流 $T$ 当作「布」，用函数 $u$ 当作「刀」，沿 $u = t$ 这一层把布切下一片。切片公式保证「切出来的布总质量」有界。</span>

电流与同调的联系同样深刻。整值电流构成一个链复形（$\partial^2 = 0$），其同调在相当广的系数下与奇异同调同构（Federer–Fleming）。由此得到几何结论：**每个同调类中都存在面积最小的整值电流代表元**，它对应着「同调约束下的最优曲面」。这是代数拓扑概念在变分框架里的落地。

**辨析｜易错点：** 切片公式是「$\le$」而不是「$=$」：水平集切片可能把面积「漏掉」（沿临界方向），所以只有单侧估计。初学者易误用为等式，需注意 $\mathrm{Lip}(u)$ 的因子与临界集的存在。

## 7 小结

- **电流**：$m$-形式空间上的连续线性泛函；质量 $\mathbf{M}(T)$ 是面积概念的抽象。
- **边界** $\partial T(\omega) = T(\mathrm{d}\omega)$：由 Stokes 公式定义，且 $\partial^2 = 0$，进入同调。
- **整流 / 整值电流**：由整流集 + 整数重数 + 定向给出；整值电流要求边界也整流。
- **Federer–Fleming 紧致性**：质量与边界质量一致有界的整值电流列有弱收敛子列。
- **Plateau 存在性**：固定边界 $\Gamma$