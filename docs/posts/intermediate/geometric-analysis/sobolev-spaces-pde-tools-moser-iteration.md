---
title: Sobolev 空间与流形上的 PDE 工具（Moser 迭代、De Giorgi–Nash 估计）
date: 2026-08-07
---

# Sobolev 空间与流形上的 PDE 工具（Moser 迭代、De Giorgi–Nash 估计）

<div class="epigraph">
<p>「自然界不作跳跃（Natura non facit saltus）。」</p>
<footer>—— 相传为莱布尼茨（G. W. Leibniz）与牛顿（I. Newton）传统中的自然哲学箴言</footer>
</div>

<div class="article-byline">
<p>第二级 · 几何分析 ｜ Peter Li《Geometric Analysis》Ch. 5（Sobolev Inequalities）｜ Jost PDE 工具章 ｜ 2026-08-07</p>
</div>

## 为什么从 Sobolev 空间开始

热方程告诉我们「解有多光滑」（抛物正则性），但还没有回答更基本的量级问题：**给定函数的积分信息，能推出多大的范数上界？** 答案在 **Sobolev 空间（Sobolev spaces）** 与 **Sobolev 不等式（Sobolev inequalities）** 中——它们是流形上一切非线性问题（Yamabe、调和映射、Ricci 流）的「能量引理」。而 **Moser 迭代** 与 **De Giorgi–Nash 估计** 则是把「弱解存在」升级为「强解光滑」的两把通用锤子。

从课程体系看，本篇把第三级《泛函分析与 Sobolev 空间》的工具搬到黎曼流形，并第一次用到 Bishop–Gromov 体积比较——**等周不等式经体积比较给出流形上的 Sobolev 常数**。这是「几何信息（Ricci 下界）→ 分析常数（Sobolev 常数）→ 方程结论」的完整链条。

<span class="marginnote">「自然界不作跳跃」正好描述 De Giorgi–Nash 的成果：弱解的连续性（不跳跃）是方程本身的强制性结果，而非人为假设。De Giorgi（1957）与 Nash（1958）几乎同时独立证明：散度型椭圆/抛物方程的弱解是 Hölder 连续的——这个结果被广泛认为是 PDE 正则性理论的分水岭。</span>

## 1 Sobolev 空间与嵌入定理

**Sobolev 空间（Sobolev space）** $W^{k,p}(M)$ 是「导数也属于 $L^p$」的函数空间：

$$W^{k,p}(M) = \big\{ u : \|\nabla^j u\|_{L^p} < \infty,\ j=0,\dots,k \big\}$$

最常见的是 $W^{1,2} = H^1$，范数 $\|u\|_{H^1}^2 = \int_M (u^2 + |\nabla u|^2)dV$。它是热方程、特征值、变分问题的自然工作空间——能量泛函在其上定义，弱解在其中存在。

**Sobolev 嵌入定理（Sobolev embedding）**：对 $p<n$，$W^{1,p} \hookrightarrow L^{p^*}$，临界指数 $p^* = \frac{np}{n-p}$（这里 $n = \dim M$）。特别地 $H^1 \hookrightarrow L^{\frac{2n}{n-2}}$。嵌入是**连续**的：存在常数 $C$ 使

$$\|u\|_{L^{2n/(n-2)}} \le C\,\big(\|\nabla u\|_{L^2} + \|u\|_{L^2}\big)$$

**关键点：常数 $C$ 依赖流形（曲率），这就是「几何进入分析」的通道。** 由 Bishop–Gromov 体积比较，Ricci 下界给 $C$ 一个只依赖维数、Ricci 下界与半径的界。<span class="marginnote">临界指数 $p^*$ 的来历：$|\nabla u|\in L^p$，要估计 $\int |u|^{p^*} $，用「$u$ 的分布函数 + 等周不等式」逐步提升——每一轮把指数 $p$ 提到 $p+\frac{p^2}{n}$，叠代趋于 $p^*$。这就是后面 Moser 迭代的原型，Sobolev 嵌入本身就是「迭代到收敛」的结果。</span>

**一个欧氏原型**：$\mathbb{R}^n$ 上经典 Sobolev 不等式 $\|u\|_{L^{2^*}}\le C_n\|\nabla u\|_2$ 的极小值是「塔朗蒂泡」$u(x)=(1+|x|^2)^{-(n-2)/2}$——与《Yamabe 问题》篇的最佳 Sobolev 常数同一来源。流形上的 Sobolev 常数不再等于欧氏值，但曲率下界保证它与欧氏常数可比，这正是「局部欧氏 + 全局曲率」的典型结合。

## 2 Nash 不等式与等周常数的关联

**Nash 不等式（Nash inequality）** 是流形上最灵活的 Sobolev 型不等式：存在常数 $C$ 使得

$$\|u\|_{L^2}^{1+\frac{2}{n}} \le C\,\|\nabla u\|_{L^2}\,\|u\|_{L^1}^{\frac{2}{n}}$$

它等价于「热核上界」（对 $e^{-t\Delta}$ 在时间方向作 Jensen 型处理）——Nash 正是用它证明了热核的全局上界，即上一篇末的估计在 Ricci 下界情形的另一种证法。而 Nash 不等式的常数由**等周不等式（isoperimetric inequality）**控制：对足够正则的 $\Omega\subset M$，

$$\operatorname{Vol}(\partial\Omega) \ge h\,\min\{\operatorname{Vol}(\Omega), \operatorname{Vol}(M\setminus\Omega)\}^{1-\frac1n}$$

等周常数 $h$（Cheeger 常数）在 Ricci 下界与体积上界下有一致的下界——于是「几何 → 等周 → Nash → 热核」链条闭合。这正是《谱几何》篇 Cheeger 不等式的来源。

**要点：Sobolev、Nash、等周、热核上界，四者在紧致流形上互相蕴含。** 它们是同一个「函数有多集中」的事实的不同侧面。<span class="marginnote">这套等价性在欧氏与流形上都成立，构成所谓的「分析等价环」：等周 ⇔ Sobolev ⇔ Nash ⇔ 热核上界 ⇔ 谱下界。做几何分析的人常按手头问题挑最顺手的一环，其余自动跟上。</span>

**辨析：为什么需要等周而非仅体积**：等周不等式控制「边界面积相对体积的比例」，比「体积增长」强得多。只有体积增长（Bishop–Gromov）不足以导出 Sobolev 常数——函数可能集中在测度很小的细长区域，等周正是对这种「细长集中」的唯一约束。

## 3 公式解析：Moser 迭代

**Moser 迭代（Moser iteration，1961）** 是证明「弱解的 $L^\infty$ 界 / 正则性」的机器。核心思想：把 Sobolev 不等式当作「提升器」，每轮把解的 $L^q$ 范数提升到更高指数，无限迭代后达到 $L^\infty$。

设 $u$ 满足椭圆不等式 $-\Delta u \le 0$ 且 $u \ge 0$，取测试函数 $u^p$（$p>0$）：

- **第一步，能量形式**：对 $-u^p\Delta u$ 积分，用分部积分得

$$\int |\nabla u|^2 u^{p-1} \le 0 \ \Rightarrow\ \int |\nabla(u^{p/2})|^2 \le \text{(控制项)}$$

把 $u^{p/2}$ 当作新函数 $w$，则 $w \in H^1$。
- **第二步，代入 Sobolev**：对 $w$ 用 Sobolev 嵌入 $\|w\|_{L^{2^*}} \le C\|\nabla w\|_{L^2}$，得到

$$\|u\|_{L^{p\,2^*}} \le C\,\|u\|_{L^p}^{p/2} \ \text{ 型的提升关系}$$

其中 $2^* = \frac{2n}{n-2} > 2$——指数从 $p$ 跳到 $p\cdot\frac{2^*}{2}$，**每轮指数被乘上一个固定倍数 $\kappa = \frac{2^*}{2} = \frac{n}{n-2} > 1$**。
- **第三步，迭代**：从 $p_0 = 2$ 出发，$p_{k+1} = \kappa\, p_k$，指数 $p_k \to \infty$。常数 $C^{1/p_k}$ 的乘积收敛（因为 $\sum 1/p_k < \infty$），取极限得

$$\|u\|_{L^\infty} \le C\, \|u\|_{L^2}$$

- **第四步，为什么奏效**：迭代每一轮把「低阶范数 → 高阶范数」的常数都压进一个收敛的乘积里；只要初始范数有限，极限就是本质有界。**「提升器 + 收敛因子」**的模板在热方程（对时间的超收缩估计）、特征函数（$L^\infty$ 界）、Yamabe 解上都反复出现。

## 4 De Giorgi–Nash 估计：弱解自动连续

有了 $L^\infty$ 界，还差连续性。**De Giorgi–Nash 估计（De Giorgi–Nash estimate）** 断言：散度型椭圆方程

$$\operatorname{div}(a(x)\nabla u) = 0, \qquad a(x)\ \text{椭圆、有界、可测}$$

的弱解是 **Hölder 连续的**：存在 $\alpha \in (0,1)$ 与 $C$（只依赖椭圆常数与维数），使得

$$|u(x) - u(y)| \le C\, d(x,y)^\alpha \,\|u\|_{L^2}$$

证明的核心是 **De Giorgi 的等值集切片（level-set slicing）**：把「$u$ 超过某高度的集合」逐层剥离，用能量不等式控制其体积衰减，推出「$u$ 在某尺度内振荡被控制」——振荡按几何级数衰减即得 Hölder 连续性。<span class="marginnote">De Giorgi 的方法是纯定量的：不假设任何光滑性，只靠能量不等式与 Sobolev，就推出振荡的几何衰减。它是「弱解 → 正则性」的典范，1957 年 De Giorgi 用它解决 Hilbert 第 19 问题（极小曲面的正则性）；Nash 同年独立给出抛物版本，用于证明热核正则性。两人后来都得了诺贝尔经济学奖——并非为了这个结果，却是分析学界津津乐道的轶事。</span>

在流形上，把 Laplace–Beltrami 算子写作 $\operatorname{div}(\sqrt g\, g^{ij}\partial_j)$，测度加权后仍是散度型算子，De Giorgi–Nash 自动适用——**流形上调和函数、热方程解、Yamabe 方程弱解都是 Hölder 连续的**。配合前面的 Moser 迭代（给出 $L^\infty$），两者合并就是完整的「弱解正则性阶梯」：

| 步骤 | 工具 | 得到的结果 |
| --- | --- | --- |
| 存在性 | 变分法 / 单调算子 | 弱解 $u \in H^1$ |
| 有界性 | Moser 迭代 | $u \in L^\infty$ |
| 连续性 | De Giorgi–Nash | $u \in C^{\alpha}$（Hölder） |
| 光滑性 | 椭圆正则性（更高阶） | $u \in C^\infty$ |

### 4.1 Schauder 估计与 L^p 估计

正则性阶梯的顶端由 **Schauder 估计（Schauder estimates）** 提供：对系数光滑的椭圆方程，解的 Hölder 导数按系数的 Hölder 半范数被控制：

$$\|u\|_{C^{2,\alpha}} \le C\big(\|f\|_{C^\alpha} + \|u\|_{L^\infty}\big)$$

配合前面的 $L^\infty$（Moser）与 $C^\alpha$（De Giorgi–Nash），构成完整的椭圆正则性谱系。**L^p 估计（Calderón–Zygmund）** 给出 $\|\nabla^2 u\|_{L^p}\le C\|f\|_{L^p}$，是 Sobolev 空间自洽性的保证——三者在流形上经「局部化 + 坐标拉回」后全部成立。

**Moser 迭代在热方程中的角色**：对热方程做「时间方向的超收缩估计」，从 $L^2$ 出发逐轮提升到 $L^\infty$，得到热核的 $L^\infty$ 上界——这是热核高斯界的另一条独立证明路线，与 Li–Yau 梯度估计互为补充。

**辨析｜易错点：** Moser 迭代给的是 $L^\infty$ 界，De Giorgi–Nash 给的是 Hölder 连续性，两者不互相蕴含（连续未必本质有界，有界未必连续）；对一般有界可测系数的方程，只能保证 Hölder，不能保证更高光滑性——光滑性需要系数本身光滑（椭圆正则性）。

## 5 小结

- **Sobolev 空间与嵌入**：$W^{1,p}\hookrightarrow L^{np/(n-p)}$，常数经 Bishop–Gromov 由 Ricci 下界控制。
- **Nash 不等式**：$\|u\|_{L^2}^{1+2/n} \le C\|\nabla u\|_2\|u\|_1^{2/n}$，等价于热核上界，常数来自等周常数。
- **Moser 迭代**：用 Sobolev 作「指数提升器」，迭代到 $L^\infty$；模板泛用于热方程与特征函数。
- **De Giorgi–Nash**：散度型椭圆/抛物弱解自动 Hölder 连续；在流形上给出调和函数与热解的连续性。
- **正则性阶梯**：存在性（$H^1$）→ 有界（$L^\infty$）→ 连续（$C^\alpha$）→ 光滑（$C^\infty$）。

在下一节，我们把这些 PDE 工具对准一个具体的非线性几何问题——**Yamabe 问题与共形几何**：在一个共形类里寻找常标量曲率的度量，它正是 Sobolev 最佳常数与临界指数理论的直接战场。
