---
title: 根系分类与 Dynkin 图
date: 2026-08-11
---

# 根系分类与 Dynkin 图

<div class="epigraph">
<p>数学最美的分类，往往是把「无限」归结为「一张小清单」。</p>
<footer>—— 欧根 · 维格纳（Eugene Wigner，意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 李代数与李群 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么会出现一张清单

上一节我们抽象出根系统后，一个自然的疑问涌上来：**究竟有多少种不同的根系统？** Killing 与 Cartan 在上世纪初给出了震撼的回答：不可分解的根系统（进而复半单李代数）**只有有限多种**，且全部列在一张图上。这张清单就是 **Dynkin 图（Dynkin diagram）**。<span class="marginnote">Killing 1890 年实际上列出了全部类型（尽管证明不完整），Cartan 1894 年的博士论文补全了分类。Dynkin 图（Dynkin 1947）用一套统一的图示把分类压缩成极易识别的图案——物理学家的「表格化」直觉又一次胜出。</span>

## 1 Cartan 矩阵：把几何编码成整数

根系统的全部信息可以编码进一个矩阵。选简单根 $\Pi = \{\alpha_1, \dots, \alpha_\ell\}$，定义 **Cartan 矩阵**：

$$A_{ij} = \left\langle \alpha_i, \alpha_j^\vee \right\rangle = \frac{2\langle \alpha_i, \alpha_j\rangle}{\langle \alpha_j, \alpha_j\rangle}$$

Cartan 矩阵是整数矩阵，对角线为 $2$，非对角元 $\leq 0$，且 $A_{ij} \neq 0 \iff A_{ji} \neq 0$。<span class="marginnote">Cartan 矩阵就是第 8 篇整数性公理下简单根对的「全部角度信息」。三个数值 $-1, -2, -3$ 分别对应夹角 $120°, 135°, 150°$（两两正交时非对角元为 0）。</span>

**Cartan 矩阵唯一决定根系统（在同构意义下）**，进而唯一决定复半单李代数。于是「分类根系统」变成了「分类满足一定条件的整数矩阵」。

## 2 Dynkin 图：分类的可视化

**Dynkin 图（Dynkin diagram）**：画法如下——

每个简单根画一个**节点**；
- 两个节点之间画连线，边数 = 两对角化角度对应的边数：$0$（正交）、$1$（$120°$）、$2$（$135°$）、$3$（$150°$）；
- 若两简单根长度不同（仅当 $A_{ij} \neq A_{ji}$ 时），在边上画**箭头**指向较短根。

**辨析｜易错点：** $B_\ell$ 与 $C_\ell$ 的图在节点排布上完全一样，唯一区别是**双箭头方向**——箭头永远指向**短根**一侧。只看节点不看箭头就会把 $\mathfrak{so}(2\ell+1)$ 错认成 $\mathfrak{sp}(2\ell)$。另外，连接两个节点的边数只可能取 $0,1,2,3$，**不存在 $4$ 条边**（$G_2$ 是极端的 $150°$，已经是最钝的简单根夹角）。

**核心分类定理**：不可分解根系统恰好对应以下 Dynkin 图（连通的）：

| 记号 | 图 | 秩 | 典型例子 |
| --- | --- | --- | --- |
| $A_\ell$（$\ell\ge1$） | 单链 $\circ-\circ-\cdots-\circ$ | $\ell$ | $\mathfrak{sl}(\ell+1,\mathbb{C})$ |
| $B_\ell$（$\ell\ge2$） | 链末一根短（双箭头） | $\ell$ | $\mathfrak{so}(2\ell+1,\mathbb{C})$ |
| $C_\ell$（$\ell\ge3$） | 链末一根短（反向双箭头） | $\ell$ | $\mathfrak{sp}(2\ell,\mathbb{C})$ |
| $D_\ell$（$\ell\ge4$） | 末节点双叉 | $\ell$ | $\mathfrak{so}(2\ell,\mathbb{C})$ |
| $E_6, E_7, E_8$ | 树状 | $6,7,8$ | 例外李代数 |
| $F_4$ | 混合长短 | $4$ | 例外 |
| $G_2$ | 夹角 $30°$ | $2$ | 例外 |

四种经典族 $A, B, C, D$ 加上五个例外 $E_6, E_7, E_8, F_4, G_2$，就是全部。<span class="marginnote">例外的出现是分类理论最令人惊讶的礼物：数学家本不期待它们，但它们真实存在于 $\mathfrak{g}_2$ 等李代数中，并在物理（八重法、例外群）与几何中一再现身。</span>

![经典 Dynkin 图：A4 单链、D4 末叉、E6/E7/E8 树、F4 与 G2 的长短根箭头](/images/lie-algebra/root-classification-dynkin-diagrams-1.svg)

## 3 公式解析：为何角度只有四种

分类的有限性最终归结为一条简单的算术事实。设 $\alpha, \beta$ 是两个不共线的简单根，$\theta$ 为它们的夹角，$m = 4\cos^2\theta = A_{ij} A_{ji}$（整数）。由于夹角在 $(90°, 180°)$ 之间（简单根夹角为钝角），$\cos^2\theta < 1$，故 $m < 4$ 且 $m \in \{0,1,2,3\}$：

$m = 0$：$\theta = 90°$，正交，不连线；
- $m = 1$：$\theta = 120°$，单边；
- $m = 2$：$\theta = 135°$，双边；
- $m = 3$：$\theta = 150°$，三边。

- **第一步，写出整数性**：$A_{ij} A_{ji} = m \in \mathbb{Z}$ 且 $> 0$。
- **第二步，用几何约束**：$A_{ij} A_{ji} = 4\cos^2\theta$，而 $A_{ij} \leq 0$，所以 $m = 4\cos^2\theta \in \{1,2,3\}$（排除正交的 0）。
- **第三步，反解角度**：$\cos\theta = -\sqrt{m}/2$，得到四个角度。

**核心要点**：反射的整数性（Cartan 整数必为整数）+ 简单根夹角为钝角 ⟹ 只有四种可能角度 ⟹ 图只可能有四类边。再结合连通图上「不允许环、不允许三叉以上」的组合论证（Humphreys §11），就逼出那张清单。<span class="marginnote">这种「先列可能、再用公理筛掉、最后证明无遗漏」的模式，是 19 世纪末分类数学的范式：Killing 用类似方法枚举了全部例外根系。</span>

## 4 从 Dynkin 图回到李代数

分类定理的完整表述（Humphreys §12 等）分两步：

**存在性**：上表每个 Dynkin 图都对应一个实根系统，进而对应一个复半单李代数（由 Serre 关系式给出生成元与关系）。

**唯一性**：两个复半单李代数同构 ⟺ 它们的 Dynkin 图相同。

**Serre 关系式（Serre relations）**（给 Cartan 矩阵 $A$）：李代数由生成元 $e_i, f_i, h_i$（$i=1,\dots,\ell$）与关系式生成，其中关键的一族是：

$$(\operatorname{ad}e_i)^{1 - A_{ij}} e_j = 0, \qquad (\operatorname{ad}f_i)^{1 - A_{ij}} f_j = 0 \quad (i \neq j)$$

例如 $A_{ij} = -1$ 时 $(\operatorname{ad}e_i)^2 e_j = 0$——这是「$\mathfrak{sl}(2)$ 三元组如何粘合在一起」的精确答案。<span class="marginnote">Serre（1955 年）把分类的「存在性」也做成显式的：从 Cartan 矩阵直接给出李代数的生成元与关系。这意味着整个复半单李代数理论可以「数据化」——任何软件包里 $\mathfrak{sl}_n, \mathfrak{so}_n$ 等都由一组矩阵数据生成。</span>

## 5 小结

- **Cartan 矩阵** $A_{ij} = \langle\alpha_i, \alpha_j^\vee\rangle$ 是整数矩阵，唯一决定根系统与李代数。
- **Dynkin 图**：节点 = 简单根，边数 = 角度（0/1/2/3），箭头指较短根。
- **分类定理**：不可分解根系统 = 经典族 $A_\ell, B_\ell, C_\ell, D_\ell$ + 例外 $E_6, E_7, E_8, F_4, G_2$。
- 角度只有四种（$90°/120°/135°/150°$），源于整数性 + 钝角约束；组合论证排除环与高叉。
- **Serre 关系式**给出存在性的显式构造；分类定理因此既完整又可实现。
- 对复半单李代数：同构 ⟺ 同 Dynkin 图；分类问题彻底解决。

在下一节，我们将离开「结构」进入「构造」：引入**万有包络代数与 PBW 定理**，为最高权表示提供代数土壤。
