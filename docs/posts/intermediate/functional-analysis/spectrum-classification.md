---
title: 谱的分类：点谱、连续谱与剩余谱
date: 2026-08-07
---

# 谱的分类：点谱、连续谱与剩余谱

<div class="epigraph">
<p>不可逆的原因有三种：核非平凡、值域不稠、值域不闭——谱于是被分成三块。</p>
<footer>—— 约翰 · 冯 · 诺伊曼（John von Neumann），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§9.2 ｜ 2026-08-07</p>
</div>

## 为什么谱要分类

$\lambda \in \sigma(T)$ 意味着 $\lambda I - T$ 不可逆。但「不可逆」有三种不同的原因：**核非平凡**（$(\lambda I - T)x = 0$ 有非零解——这就是特征值）、**值域不稠密**（方程「几乎无解」）、**值域稠密但不闭**（方程有解但解不连续依赖于右端）。这三种原因对应谱的三类：**点谱、剩余谱、连续谱**。分类的意义在于：不同类的谱点对「方程 $(\lambda I - T)x = y$ 的可解性」给出完全不同的回答——这正是谱理论的核心信息。<span class="marginnote">一个比喻：点谱像「实根」（$\\lambda I - T$ 有非平凡核），连续谱像「极限点」（$\\lambda$ 被特征值逼近但本身不是），剩余谱像「洞」（值域够不着）。对自伴算子（第九章后文），剩余谱为空——这是自伴性的重要福利。</span>

## 1 谱的三分类

**定义**：设 $T \in \mathcal{B}(X)$，$X$ 复 Banach 空间，$\lambda \in \sigma(T)$。

1. **点谱（point spectrum）** $\sigma_p(T)$：$\lambda I - T$ **不是单射**——存在非零 $x$ 使 $Tx = \lambda x$。$\lambda$ 是**特征值**，$x$ 是**特征向量**。
2. **连续谱（continuous spectrum）** $\sigma_c(T)$：$\lambda I - T$ **单射**、**值域稠密**，但**值域不闭**（逆存在但无界）。
3. **剩余谱（residual spectrum）** $\sigma_r(T)$：$\lambda I - T$ **单射**但**值域不稠密**（逆存在且定义在非稠密子空间上）。

**三类的并 = 全谱**：$\sigma(T) = \sigma_p(T) \cup \sigma_c(T) \cup \sigma_r(T)$（互不相交）。<span class="marginnote">记忆法：三类的分界是「$\\lambda I - T$ 的单射性」与「值域的稠密性」两个问题：<strong>核非零 ⟹ 点谱；值域不稠 ⟹ 剩余谱；都正常但值域不闭 ⟹ 连续谱</strong>。自伴算子没有剩余谱（值域问题与核问题互相对偶），这是它格外干净的原因。</span>

**辨析｜易错点：** 连续谱里的 $\lambda$ **不是**特征值（单射），但 $\lambda I - T$ 也不可逆（逆无界）。所以「$\lambda \in \sigma_c$」时方程 $(\lambda I - T)x = y$ 可能对某些 $y$ 有解，但解不连续依赖 $y$——「几乎可解但不稳定」。

## 2 例子：三类的具体面貌

**例一（$M_t$ 于 $L^2[0,1]$）**：$\sigma(M_t) = [0,1]$ **全是连续谱**。

$tf = \lambda f$ 无非零 $L^2$ 解（$f$ 几乎处处为零）——没有点谱。
$(\lambda - t)^{-1}$ 无界但定义在稠密子空间 $\{f : f/(\lambda - t) \in L^2\}$ 上——值域稠密、逆无界，连续谱。<span class="marginnote">量子力学的直观：位置算子的谱 $\\mathbb{R}$ 全是连续谱。波函数可以「集中在 $\\lambda$ 附近」但永远不是「精确在 $\\lambda$」——这正是连续谱的物理含义：<strong>连续谱 = 「极限意义」的本征值，没有真正的本征态</strong>。</span>

**例二（移位算子 $S$ 于 $l^2$）**：$\sigma(S) = \overline{D}$（闭单位盘）。

点谱：$|\lambda| < 1$（$Sx = \lambda x$ 有解 $x = (1, \lambda, \lambda^2, \ldots)$）。
边界 $|\lambda| = 1$：属于连续谱或剩余谱（取决于具体算子；对 $S$ 是连续谱部分）。

**例三（前移位 $S^*$）**：$\sigma(S^*) = \overline{D}$，但分类不同——$S^*$ 有**剩余谱**（$|\lambda| < 1$ 时 $S^*x = \lambda x$ 无解，值域不稠）。<span class="marginnote">$S$ 与 $S^*$ 的谱相同（$\overline{D}$），但分类不同：$S$ 在盘内是点谱，$S^*$ 在盘内是剩余谱。这说明<strong>谱的分类不是「谱集合」的固有属性，而是算子的属性</strong>——同一个谱集合可以有不同的内部结构。谱映射理论（对偶）会揭示点谱与剩余谱的互换关系。</span>

## 3 点谱与剩余谱的对偶互换

**定理（对偶谱的互换）**：对 $T \in \mathcal{B}(X)$，点谱与剩余谱在对偶下互换：

$$
\sigma_p(T^*) \supseteq \sigma_r(T), \qquad \sigma_r(T^*) \supseteq \sigma_p(T)
$$

（更精细地，$\sigma_p(T^*) = \sigma_r(T) \cup$（$\sigma_p(T)$ 的一部分）之类的关系在自反空间里成立。）直觉：$\lambda \in \sigma_r(T)$ 时 $\operatorname{ran}(\lambda I - T)$ 不稠密，由 $\ker(\lambda I - T^*) = (\operatorname{ran}(\lambda I - T))^\perp$ 非零，$\lambda$ 是 $T^*$ 的特征值。

**例**：$S$（移位）的盘内是点谱，$S^*$ 的盘内是剩余谱——正是这个互换的体现。对自伴算子 $T = T^*$，点谱与剩余谱的互换导致**剩余谱为空**（因为 $\sigma_p(T) = \sigma_p(T^*)$，互换后 $\sigma_r(T)$ 空）。<span class="marginnote">这是自伴算子谱理论的第一条好消息：<strong>自伴算子的谱 = 点谱 + 连续谱，没有剩余谱</strong>。量子力学的「可观测量」（自伴算子）因此只有「本征值 + 连续谱」两种能量——这是谱定理与量子力学测量理论的基础。</span>

## 4 公式解析：剩余谱与对偶的关系

把「$\sigma_r(T)$ 通过 $T^*$ 变成点谱」的机制写清：

$$
\lambda \in \sigma_r(T) \iff \overline{\operatorname{ran}(\lambda I - T)} \neq X \iff \ker(\lambda I - T^*) \neq \{0\}
$$

- **第一步（剩余谱的定义）**：$\lambda I - T$ 单射但值域不稠密，即 $\overline{\operatorname{ran}(\lambda I - T)} \subsetneq X$。
- **第二步（正交补）**：$\overline{\operatorname{ran}(\lambda I - T)} \neq X$ ⟺ 存在非零 $f$ 正交于值域，即 $(\operatorname{ran}(\lambda I - T))^\perp \neq \{0\}$。
- **第三步（对偶核）**：由 §7.7，$\ker(\lambda I - T^*) = (\operatorname{ran}(\lambda I - T))^\perp \neq \{0\}$——$\lambda$ 是 $T^*$ 的特征值。

**关键**：整个论证只用了「正交补」关系 $\ker T^* = (\operatorname{ran}T)^\perp$。**剩余谱在对偶下「变成」点谱**——这是谱分类最深层的结构。

## 5 例题精讲：谱分类的三个判别

**例题一：判断 $M_t$ 的谱类型**。

- $\sigma(M_t) = [0,1]$。对 $\lambda \in [0,1]$，$tf = \lambda f$ 无 $L^2$ 解（点谱空）。
- $(\lambda - t)^{-1}$ 无界，值域稠密——连续谱。
- 结论：$\sigma(M_t) = \sigma_c(M_t) = [0,1]$，点谱剩余谱皆空。

**例题二：判断对角算子 $M_\lambda$ 的谱类型**。

- $M_\lambda x = (\lambda_n x_n)$。$\sigma_p(M_\lambda) = \{\lambda_n\}$（$e_n$ 是特征向量）。
- 其余谱点是连续谱（若 $\lambda$ 是 $\{\lambda_n\}$ 的聚点但不是某个 $\lambda_n$）。
- $\sigma(M_\lambda) = \overline{\{\lambda_n\}}$。

**例题三：自伴紧算子的谱**。

- 自伴紧算子（如 Hermite 核积分算子）：$\sigma$ = 特征值 + $\{0\}$（0 可能是连续谱）。
- 特征值可数、实、只聚在 0（Riesz-Schauder + 自伴）。
- 没有剩余谱——自伴性的福利。

**核心要点**：谱分类的三个判别——乘法（全连续）、对角（特征值 + 聚点）、自伴紧（特征值 + 0）——覆盖了最常见的算子类型。

**辨析｜易错点：** 谱分类依赖「值域是否稠密」「值域是否闭」，这些是**拓扑**性质，与范数等价无关。$\lambda I - T$ 的值域稠密性在等价范数下不变，所以谱分类是「几何」的而非「度量」的。

## 6 例题精讲：谱分类的练习

**练习一：$M_t$ 的谱分类**。

- $\sigma(M_t) = [0,1]$，全连续谱。
- 无点谱（$tf = \lambda f$ 无 $L^2$ 解）、无剩余谱（自伴）。
- 连续谱的模范。

**练习二：对角算子 $M_\lambda$ 的谱分类**。

- 点谱 $= \{\lambda_n\}$（$e_n$ 特征向量）。
- 聚点（非特征值）属连续谱。
- $\sigma = \overline{\{\lambda_n\}}$。

**练习三：移位算子 $S$ 的谱分类**。

- $|\lambda| < 1$：点谱（$x = (1,\lambda,\lambda^2,\ldots)$）。
- $|\lambda| = 1$：连续谱部分。
- $\sigma(S) = \overline{D}$。

**核心要点**：谱分类三练习——乘法（连续）、对角（点 + 连续）、移位（盘内点谱）——覆盖常见类型。

**辨析｜易错点：** 谱分类依赖「值域稠密/闭」，是拓扑性质。$S$ 与 $S^*$ 谱相同但分类不同（点谱 vs 剩余谱）。


## 7 小结

- **三类谱**：点谱（核非平凡）、连续谱（值域稠密不闭）、剩余谱（值域不稠密）；$\sigma = \sigma_p \cup \sigma_c \cup \sigma_r$。
- **$M_t$ 的谱**：$[0,1]$ 全是连续谱——连续谱的模范。
- **对偶互换**：$\sigma_r(T) \to \sigma_p(T^*)$；自伴算子无剩余谱。
- **自伴福利**：谱 = 点谱 + 连续谱，无剩余谱（量子力学测量的基础）。
- **定位**：谱分类把「不可逆」的三种原因分开，为谱性质（下节）做准备。

在下一节，我们研究谱的整体性质——**非空性与紧性**：每个有界算子的谱都是非空紧集。
