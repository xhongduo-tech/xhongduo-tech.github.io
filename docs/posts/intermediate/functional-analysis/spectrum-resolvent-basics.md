---
title: 线性算子的谱、正则点与预解式
date: 2026-08-07
---

# 线性算子的谱、正则点与预解式

<div class="epigraph">
<p>特征值的概念太窄，容纳不下无穷维的全部奥秘——谱，才是特征值的完整继承人。</p>
<footer>—— 约翰 · 冯 · 诺伊曼（John von Neumann），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§9.1 ｜ 2026-08-07</p>
</div>

## 为什么「谱」取代「特征值」

线性代数里，矩阵的特征值 $\lambda$ 满足 $Ax = \lambda x$。可无穷维算子的问题在于：**很多「自然」的数不是特征值，却在谱里**。例如乘法算子 $M_t f(t) = tf(t)$ 于 $L^2[0,1]$，每个 $\lambda \in [0,1]$ 都「很像特征值」，但 $tf = \lambda f$ 没有非零 $L^2$ 解（$f$ 只在 $\lambda$ 点取值，几乎处处为零）——$\lambda$ 不是特征值，却是谱点。因此谱的定义必须更宽：**不看「$Tx = \lambda x$ 有无解」，而看「$\lambda I - T$ 是否可逆」**。这个「可逆性」的视角让谱成为无穷维特征值理论的正确框架。<span class="marginnote">一个直觉：特征值是「$\\lambda I - T$ 有非平凡核」；谱则是「$\\lambda I - T$ 不可逆」——不可逆的原因不止核非平凡（还有值域不闭、值域不稠密等），所以谱 ⊇ 特征值。量子力学里「连续谱」就是这么来的：能量 $\lambda$ 不在点谱里，却对应「非正常本征态」。</span>

## 1 正则点与谱

**定义**：设 $X$ 是复 Banach 空间，$T \in \mathcal{B}(X)$，$\lambda \in \mathbb{C}$。

若 $\lambda I - T$ 是**双射**且逆有界（由逆算子定理，双射 + 有界自动 ⟹ 逆有界），则称 $\lambda$ 为 $T$ 的**正则点（regular point）**；
**谱（spectrum）** $\sigma(T)$ = 全体非正则点；
**预解集（resolvent set）** $\rho(T)$ = 全体正则点 = $\mathbb{C} \setminus \sigma(T)$。

**为什么 $\lambda$ 是「数」而不是「向量」**：特征值问题的核心是「$\lambda I - T$ 可逆吗」。可逆性是算子的性质，而 $\lambda I$ 是「数乘算子」，于是「数 $\lambda$」能否从算子中「减掉并求逆」成为判别标准。<span class="marginnote">记号习惯：谱用 $\\sigma(T)$（sigma），预解集用 $\\rho(T)$（rho）。「预解（resolvent）」一词来自「解决 $\\lambda I - T$ 的可逆问题」——$\\rho$ 里的 $\\lambda$ 让 $\\lambda I - T$ 可解。这两个记号在谱理论里是最常用的。</span>

**核心要点：谱 = 「$\lambda I - T$ 不可逆的 $\lambda$」的集合**。它把「特征值」从「方程 $Tx = \lambda x$」扩展为「算子 $\lambda I - T$ 的可逆性」——这是无穷维的必然推广。

## 2 预解式

**定义**：对 $\lambda \in \rho(T)$，定义**预解式（resolvent）**

$$
R_\lambda(T) = (\lambda I - T)^{-1}
$$

它是「$\lambda$ 不在谱里时，$\lambda I - T$ 的逆」——一个从 $\rho(T)$ 到 $\mathcal{B}(X)$ 的算子值函数。

**预解式方程（resolvent identity）**：对 $\lambda, \mu \in \rho(T)$，

$$
R_\lambda - R_\mu = (\mu - \lambda) R_\lambda R_\mu
$$

证明：$R_\lambda - R_\mu = R_\lambda(\mu I - T)R_\mu - R_\lambda(\lambda I - T)R_\mu = (\mu - \lambda)R_\lambda R_\mu$。<span class="marginnote">预解式方程是谱理论的地基恒等式：它把「不同 $\lambda$ 的预解式之差」表示成「乘积」——这使 $R_\lambda$ 成为算子值的「解析函数」的雏形。第九章后文与复分析的联系（谱半径公式）都从这里生长。</span>

## 3 例子：乘法算子与移位算子的谱

**例一（乘法算子 $M_t$）**：$M_t f(t) = tf(t)$ 于 $L^2[0,1]$。

对 $\lambda \notin [0,1]$，$(\lambda - t)^{-1}$ 有界，$R_\lambda f = f/(\lambda - t)$，$\lambda \in \rho$。
对 $\lambda \in [0,1]$，$(\lambda - t)^{-1}$ 无界（在 $t = \lambda$ 处爆炸），$\lambda I - M_t$ 不可逆（值域不闭）。
故 $\sigma(M_t) = [0,1]$，**且全是连续谱**（没有点谱——$tf = \lambda f$ 无非零 $L^2$ 解）。<span class="marginnote">这是连续谱的模范例子：<strong>谱是「乘子函数的值域」$[0,1]$，但没有任何一点是特征值</strong>。量子力学的位置算子 $Qf(x) = xf(x)$ 的谱是 $\\mathbb{R}$——全是连续谱，对应「位置可以取任何实数但不是固定本征态」。</span>

**例二（单侧移位 $S$）**：$S(x_1,x_2,\ldots) = (0,x_1,x_2,\ldots)$ 于 $l^2$。

- $\|S\| = 1$，故 $|\lambda| > 1$ 时 $\lambda \in \rho$（诺伊曼级数 $R_\lambda = \sum S^n/\lambda^{n+1}$）。
- $|\lambda| < 1$ 时，$Sx = \lambda x$ 有解 $x = (1, \lambda, \lambda^2, \ldots)$（若 $\lambda^2$ 和收敛）——点谱 $\{|\lambda| < 1\}$。
- $|\lambda| = 1$ 时属于谱但非点谱（连续谱部分）。$\sigma(S) = \overline{D}$（闭单位圆盘）。

## 4 公式解析：预解式方程

把预解式方程的证明拆成三步：

$$
R_\lambda - R_\mu = (\mu - \lambda) R_\lambda R_\mu
$$

- **第一步，插入单位**：$R_\lambda - R_\mu = R_\lambda\, I - I\, R_\mu = R_\lambda(\mu I - T)R_\mu - R_\lambda(\lambda I - T)R_\mu$——因为 $R_\lambda(\lambda I - T) = I$ 且 $(\mu I - T)R_\mu = I$。
- **第二步，相减**：$R_\lambda(\mu I - T)R_\mu - R_\lambda(\lambda I - T)R_\mu = R_\lambda\big((\mu I - T) - (\lambda I - T)\big)R_\mu$。
- **第三步，化简**：$(\mu I - T) - (\lambda I - T) = (\mu - \lambda)I$，故 $R_\lambda - R_\mu = (\mu - \lambda)R_\lambda R_\mu$。

**关键**：恒等式完全来自「$R_\lambda$ 是 $(\lambda I - T)$ 的逆」这一条定义——没有用到 $T$ 的任何具体性质。**预解式方程是「取逆」这一操作的纯代数后果**。

## 5 例题精讲：谱的计算

**例题一：对角算子 $M_\lambda$ 的谱**。

- $M_\lambda x = (\lambda_n x_n)$ 于 $l^2$。$\sigma(M_\lambda) = \overline{\{\lambda_n\}}$（特征值的闭包）。
- $M_\lambda$ 紧 ⟺ $\lambda_n \to 0$ ⟺ 谱只聚在 0。
- 非紧情形（$\lambda_n$ 不趋于 0），谱有非零聚点。

**例题二：积分算子 $T_K$ 的谱**。

- $T_K$ 紧，谱 = $\{0\} \cup \{\text{特征值}\}$，且特征值可数、只聚在 0（Riesz-Schauder，下节）。
- Volterra 算子 $V$ 的谱 = $\{0\}$（$V$ 是拟幂零算子：$\|V^n\|^{1/n} \to 0$）。
- $V$ 的谱只有 0，但 $V \neq 0$——谱不能区分「零」与「拟零」。

**例题三：$T$ 与 $T^n$ 的谱**。

- $\sigma(T^n) = \{\lambda^n : \lambda \in \sigma(T)\}$（谱映射定理的雏形）。
- 推论：$\sigma(T)$ 非空（下节）⟹ $\sigma(T^n)$ 非空。
- 谱映射定理让「算子幂的谱」从「原谱」直接读出。

**核心要点**：谱计算的三个例子——对角（特征值闭包）、紧算子（聚在 0）、幂（谱映射）——展示谱如何从算子读出。

**辨析｜易错点：** 谱 ⊇ 特征值，但可以严格大。$M_t$ 的谱 $[0,1]$ 没有点谱；$V$ 的谱 $\{0\}$ 没有非零特征值。**「没有特征值」不等于「谱小」**——谱是更大的对象。

## 6 常见误区与辨析

**误区一：以为谱就是特征值的集合**。

- 谱 ⊇ 特征值，连续谱、剩余谱都不是特征值。
- $M_t$ 的谱 $[0,1]$ 没有点谱。

**误区二：把「不可逆」当「核非平凡」**。

- 不可逆有三种原因：核非平凡、值域不稠、值域不闭。
- 谱的分类正是按这三种原因。

**误区三：忘记谱的定义依赖复空间**。

- 实空间上旋转矩阵无实谱，需复化。
- 谱非空是复 Banach 空间的定理。

**核心要点：谱 = 「$\lambda I - T$ 不可逆」的 $\lambda$**——比特征值宽得多。


## 7 小结

- **谱** $\sigma(T)$ = $\lambda I - T$ 不可逆的 $\lambda$；**预解集** $\rho(T)$ = 可逆的 $\lambda$。
- **预解式** $R_\lambda = (\lambda I - T)^{-1}$；预解式方程 $R_\lambda - R_\mu = (\mu-\lambda)R_\lambda R_\mu$。
- **例子**：$M_t$ 谱 $[0,1]$（连续谱）；移位算子谱为闭单位圆盘；对角算子谱为特征值闭包。
- **谱 ⊇ 特征值**：连续谱不是特征值，但属于谱。
- **定位**：谱理论用「可逆性」取代「特征方程」，为第九章全部内容奠基。

在下一节，我们分类谱——**点谱、连续谱与剩余谱**，把「不可逆」的三种原因分别刻画。
