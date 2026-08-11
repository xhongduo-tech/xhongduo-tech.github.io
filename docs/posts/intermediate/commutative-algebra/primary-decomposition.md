---
title: 准素分解
date: 2026-08-11
---

# 准素分解

<div class="epigraph">
<p>抽象代数……使数学中最古老的部分——数论——重现青春。</p>
<footer>—— 埃米 · 诺特（Emmy Noether）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 交换代数 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从准素分解开始

整数理论里最漂亮的一句话：**每个正整数都能唯一分解为素数的乘积**。交换代数想把这个故事搬到一般环的理想上。但一般环里，理想未必能分解成「素理想」的乘积——退一步，却能唯一分解成**准素理想**的交。这正是 Lasker（1905）证明、Noether（1921）用链条件推广的**准素分解（primary decomposition）**定理，它让「因子分解」从算术进入代数，也成为整门交换代数的基石之一。

几何上这句话有极美的翻译：一个代数簇是「不可约分支」的并，而理想分解记录的就是这些分支。这篇先把准素理想本身讲透，再给出分解定理与唯一性，最后落到几何。<span class="marginnote">埃米 · 诺特（1882—1935）是近代抽象代数的奠基人，1921 年的准素分解论文被视为「抽象代数学的里程碑」。她生前在哥廷根，因纳粹迫害于 1933 年移居美国，1935 年病逝。爱因斯坦在《纽约时报》上悼念她是「有史以来最伟大的女性数学家」。</span>

## 1 幂零与准素

先看一个数论类比。对素数 $p$，整数分解里 $p^e$ 与 $p$ 的关系是：$p^e$ 的「根」是 $(p)$。把这个「取根」推广到理想，就有了上一节定义的**根** $\sqrt{\mathfrak{a}} = \{x \mid x^n \in \mathfrak{a}\}$。

**准素理想（primary ideal）**：理想 $\mathfrak{q}$ 称为准素理想，若 $xy \in \mathfrak{q}$ 且 $x \notin \mathfrak{q}$ 则存在 $n$ 使 $y^n \in \mathfrak{q}$。

$\mathbb{Z}$ 中准素理想恰是 $(p^e)$（$p$ 素数）。$(12) = (4) \cap (3)$ 就是一次分解。
- $k[x,y]$ 中 $(x^2, y)$ 是准素理想：它对应「沿 $x$-轴多重相交」的代数条件。
- 素理想都是准素的；「准素」意味着「根的某次幂」，即元素差一点就是素理想。

**重点：$\mathfrak{q}$ 是准素理想当且仅当 $\sqrt{\mathfrak{q}}$ 是素理想。** 这句话给出准素概念的另一种写法：$A/\mathfrak{q}$ 的所有零因子都幂零，即 $A/\mathfrak{q}$ 的幂零根 $\operatorname{nil}(A/\mathfrak{q})$ 是素理想。<span class="marginnote">记号：$\mathfrak{q}$ 准素且 $\sqrt{\mathfrak{q}} = \mathfrak{p}$ 时称 $\mathfrak{q}$ 是<strong>$\mathfrak{p}$-准素</strong>。注意「根是素理想」只是准素的必要不充分条件——根是素理想不一定准素，这是最经典的易错点之一。</span>

**辨析｜易错点：** 「$\sqrt{\mathfrak{q}}$ 是素理想」**不等于**「$\mathfrak{q}$ 是准素的」。反例：$k[x,y]/(x^2, xy)$ 的像环中，取 $\bar{x}$ 生成的理想，其根是 $(\bar{x})$，素理想；但 $\bar{x}\bar{y} = 0 \in \mathfrak{q}$ 而 $\bar{y} \notin \mathfrak{q}$、$\bar{x}^n \notin \mathfrak{q}$ 对所有 $n$。判断准素性必须回到定义，别只查根。

## 2 Lasker–Noether 定理

**Lasker–Noether 定理**：设 $A$ 是 Noether 环，$\mathfrak{a}$ 是 $A$ 的真理想，则存在有限个准素理想 $\mathfrak{q}_1, \dots, \mathfrak{q}_n$ 使

$$\mathfrak{a} = \mathfrak{q}_1 \cap \mathfrak{q}_2 \cap \cdots \cap \mathfrak{q}_n.$$

并且可以要求：
1. 各 $\mathfrak{p}_i = \sqrt{\mathfrak{q}_i}$ 两两不同（消去冗余）；
2. 没有 $\mathfrak{q}_i \supseteq \bigcap_{j \neq i} \mathfrak{q}_j$（各分量不可省）。

满足这两条叫**最简（irredundant）分解**。

**重点：分解存在性靠的就是链条件。** 证明思路是「最小反例」+ 无穷升链：若某理想无分解，取极大者 $\mathfrak{a}$；若 $\mathfrak{a}$ 不准素，由定义找到 $xy \in \mathfrak{a}$、$x \notin \mathfrak{a}$、$y \notin \sqrt{\mathfrak{a}}$，作理想链 $\mathfrak{a} + (x) \subsetneq \cdots$，用 Noether 链条件逼出矛盾。<span class="marginnote">这里首次用上链条件——Noether 环定义就是「理想升链稳定」，下一节《链条件与 Noether 环》会系统讲它。可以说准素分解是链条件给出的第一份分红。</span>

**唯一性（第一唯一性定理）**：最简分解中，集合 $\{\sqrt{\mathfrak{q}_1}, \dots, \sqrt{\mathfrak{q}_n}\}$ 由 $\mathfrak{a}$ 唯一决定，与分解选取无关。<span class="marginnote">第一唯一性：根集 $\operatorname{Ass}(A/\mathfrak{a})$ 唯一。第二唯一性：对「极小」根 $\mathfrak{p}_i$（不含于其他根者），对应的分量 $\mathfrak{q}_i$ 也唯一；「嵌入」根的分量才可能不唯一。这是本节的深度所在。</span>

## 3 几何：不可约分支

为什么准素分解重要？几何上它有直接的翻译。设 $k$ 代数闭，$A = k[x_1, \dots, x_n]$，理想 $\mathfrak{a}$ 的零点集是

$$V(\mathfrak{a}) = \{\mathbf{x} \in k^n \mid f(\mathbf{x}) = 0,\ \forall f \in \mathfrak{a}\}.$$

若 $\mathfrak{a} = \mathfrak{q}_1 \cap \cdots \cap \mathfrak{q}_n$ 是最简分解，则

$$V(\mathfrak{a}) = V(\mathfrak{q}_1) \cup \cdots \cup V(\mathfrak{q}_n) = V(\sqrt{\mathfrak{q}_1}) \cup \cdots \cup V(\sqrt{\mathfrak{q}_n}).$$

**重点：一个代数簇被准素分解拆成「不可约分支」的并。** 每个 $V(\mathfrak{q}_i)$ 是不可约分支（对极小根），整条曲线、曲面由分支拼成。<span class="marginnote">例：$k[x,y]$ 中 $\mathfrak{a} = (xy)$ 的零点集是两条坐标轴 $V(x) \cup V(y)$；分解 $(xy) = (x) \cap (y)$ 的两个准素分量正是两条分支。把平面换成曲面 $(x^2 y)$：分解 $(x^2y) = (x^2) \cap (y)$ 中 $(x^2)$ 的分支「多重」，记下相重数——准素分量连「多重分支」的信息都保留着。</span>

嵌入素理想（embedded prime）对应的是「分支被另一分支吞掉」的退化情形：比如 $(x^2, xy) = (x) \cap (x^2, y)$ 中，$(\bar{x})$ 是嵌入的，它没有自己的分支，只记录「原点处更高阶的重叠」。几何学家对嵌入素理想又爱又怕——它藏起局部奇点信息，却让分解不再唯一。

## 4 公式解析：$\mathbb{Z}$ 中的分解与唯一性

在 $\mathbb{Z}$ 上，准素分解就是算术基本定理。设

$$m = p_1^{e_1} p_2^{e_2} \cdots p_r^{e_r}, \qquad \mathfrak{a} = (m).$$

则最简分解

$$\mathfrak{a} = (p_1^{e_1}) \cap (p_2^{e_2}) \cap \cdots \cap (p_r^{e_r})$$

拆解这条公式：

- **第一步，翻译**：$(m) = \bigcap_i (p_i^{e_i})$，因为 $m \mid n$ 当且仅当每个 $p_i^{e_i} \mid n$——交集对应「同时整除」。
- **第二步，验证准素**：$\sqrt{(p_i^{e_i})} = (p_i)$ 是素理想，故每个分量 $\mathfrak{p}_i$-准素，这里 $\mathfrak{p}_i = (p_i)$。
- **第三步，看唯一性**：根集 $\{(p_1), \dots, (p_r)\}$ 由 $m$ 唯一决定（就是它的素因子），而且每个根都是极小的（$\mathbb{Z}$ 中任意两个不同素数理想互不包含）——所以第二唯一性下各分量也唯一。这正是算术基本定理的环论翻版。

**辨析｜易错点：** 分解成「素理想乘积」与「准素理想交集」是两回事：$(p_1^{e_1}\cdots p_r^{e_r}) = \prod (p_i^{e_i})$ 是乘积，而分解定理给的是**交集**。一般环里，用交集远比用乘积普遍——即便 $\mathbb{Z}$ 中二者恰好给同一结果，含义也不同。

## 5 小结

- **准素理想**：$xy \in \mathfrak{q}$、$x \notin \mathfrak{q}$ 推出 $y^n \in \mathfrak{q}$；等价于根是素理想（但根是素理想不保证准素）。
- **Lasker–Noether 定理**：Noether 环中每个理想都有最简准素分解。
- **唯一性**：根集 $\operatorname{Ass}(A/\mathfrak{a})$ 唯一（第一唯一性）；极小根对应的分量唯一（第二唯一性），嵌入分量不唯一。
- **几何翻译**：$V(\mathfrak{a})$ = 不可约分支的并；准素分量保留多重信息。
- $\mathbb{Z}$ 上准素分解即算术基本定理。

在下一节，我们开始研究分解存在背后的发动机——**链条件**：为什么「升链稳定」能推出 Noether 环，而 Noether 环又是如何界定「可以有限分解」的。
