---
title: 半单李代数与 Killing 型
date: 2026-08-11
---

# 半单李代数与 Killing 型

<div class="epigraph">
<p>我见过许多将对称性奉为圭臬的人，但真正的力量在于对称性背后的不变量。</p>
<footer>—— 埃米 · 诺特（Emmy Noether，意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 李代数与李群 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么引入 Killing 型

可解与幂零描述了「退化」的一端，这一端比较容易处理。另一半则深藏结构：**半单（semisimple）**李代数——不含非零可解理想的李代数。半单李代数的结构极为丰富，而进入它的钥匙是一把对称双线性型：**Killing 型**。<span class="marginnote">Killing 型是 Wilhelm Killing 在 1888 年前后独立于 Sophus Lie 完成的李代数分类工作的一部分——他正是通过这把「度量」成功刻画了复半单李代数，比 Cartan 的分类更早奠定根基。Killing 本人未能完全完成证明，Cartan 的博士论文补全了它。</span>

## 1 定义：从伴随表示生成的不变量

**Killing 型（Killing form）**：$L$ 上的对称双线性型 $\kappa$ 定义为

$$\kappa(x, y) = \operatorname{tr}(\operatorname{ad}x \circ \operatorname{ad}y)$$

即取伴随算子复合后的迹。<span class="marginnote">迹 $tr$ 本身是最基本的「不变量」：相似变换下不变。因此 Killing 型对同构、同态都有良好的自然性——它从李代数「自身作用」的痕迹中提取信息，不依赖任何外部的选择。</span>

它自动具备两个关键性质：

**对称性**：$\kappa(x,y) = \kappa(y,x)$，来自迹的对称性 $\operatorname{tr}(AB) = \operatorname{tr}(BA)$。
- **不变性**：$\kappa([x,y], z) = \kappa(x, [y,z])$，即括号对 Killing 型可「移动」。这使 $\kappa$ 成为「不变双线性型」，相当于李代数的度量结构。<span class="marginnote">不变性可从循环性 $\operatorname{tr}([x,y]w) = \operatorname{tr}(x[y,w])$ 推出；它对后文「正交补」结构、以及根空间分解的几何化至关重要。</span>

**半单（semisimple）**：若 $L$ 不含非零可解理想（等价地，不含非零阿贝尔理想），则称 $L$ **半单**。<span class="marginnote">「半单」直译即「一半是单的」——严格说它要求的是「无平凡可解理想」，比「无平凡理想」弱。全体单李代数的理想直和恰是半单李代数（见第 5 节）。</span>

**单（simple）**：$L$ 无平凡理想且 $[L, L] \neq 0$，则称 $L$ **单**。半单包含「若干单的代数和」；单是「不能再分解」的构件。

## 2 Cartan 判别法：退化性把两类分开

**Cartan 判别法（Cartan's criterion）** 是全书第一个「判别结构」的利器：

> $\mathbb{F}$ 特征零，$L$ 是 $\mathbb{F}$ 上有限维李代数。
> 1. $L$ **可解** ⟺ $\kappa(L, [L, L]) = 0$（即 $\kappa$ 在 $L$ 与 $[L,L]$ 上退化为零）。
> 2. $L$ **半单** ⟺ $\kappa$ **非退化**（即对每个非零 $x$ 存在 $y$ 使 $\kappa(x, y) \neq 0$）。

**辨析｜易错点：** 「$\kappa(L, [L,L]) = 0$」不等于「$\kappa$ 对 $[L,L]$ 限制后为零」——它是说 $\kappa(x, y) = 0$ 对所有 $x \in L, y \in [L,L]$。初学者常误读为「Killing 型平凡」。判别法第二句才是关键应用：**半单 ⟺ 非退化**，这把「无退化理想」的代数条件翻译成了「非退化度量」的线性条件，也直接让正交补、对偶空间等工具进场。

$\mathfrak{sl}(n, \mathbb{C})$ 的 Killing 型可以直接算出：$\kappa(x, y) = 2n \operatorname{tr}(xy)$（在 $\mathfrak{sl}(n)$ 上）。当 $n \geq 2$ 时非退化，故 $\mathfrak{sl}(n,\mathbb{C})$ 半单——实际上它是单的。<span class="marginnote">计算细节：$\operatorname{ad}x$ 在标准基上的矩阵迹需要较长推导，Humphreys 习题提供了组合证明。此处记住公式即可——它是后续 $\mathfrak{sl}(2)$ 表示论中归一化的基础。</span>

## 3 结构定理：半单的代数和与分解

**半单结构定理（第一形式）**：特征零有限维半单李代数 $L$ 可写成其单理想 $L_i$ 的**理想直和**：

$$L = L_1 \oplus L_2 \oplus \cdots \oplus L_t$$

其中每个 $L_i$ 是单理想，且这种分解在排列次序意义下**唯一**。<span class="marginnote">这相当于半单李代数的「素数分解」：单理想是素因子，直和是相乘。与第 1 篇的「理想直和」概念精确对上——$[L_i, L_j] = 0$ 当 $i \neq j$。</span>

**半单结构定理（第二形式）**：特征零半单李代数 $L$ 满足

$$L = \operatorname{Der}(L), \qquad [L, L] = L$$

即每个导子都是内导子 $\operatorname{ad}x$，且 $L$ 由括号自生成。这正是「半单 = 完整 + 自我充足」的代数含义。<span class="marginnote">后文 Weyl 定理（第 5 篇）将在此基础上推出完全可约性；根空间分解（第 7 篇）则把半单的「度量」进一步翻译成「格」——几何化由此展开。</span>

## 4 公式解析：非退化性的判定

用 $\mathfrak{sl}(2, \mathbb{C})$ 验证 Cartan 判别法第二句。基为 $e, f, h$，伴随算子按第 3 篇的矩阵计算，得到 Killing 型矩阵：

$$\kappa = \begin{pmatrix} 0 & 4 & 0 \\ 4 & 0 & 0 \\ 0 & 0 & 8 \end{pmatrix} \qquad \text{(在基 } e, f, h \text{ 下)}$$

三步拆解：

- **第一步，算 $\operatorname{ad}x \operatorname{ad}y$ 的迹**：如 $\kappa(e, f) = \operatorname{tr}(\operatorname{ad}e \operatorname{ad}f)$。已知 $\operatorname{ad}e(f) = -h$，$\operatorname{ad}f(h) = 2f$，$\operatorname{ad}f(e) = -h$ 等，逐项取迹得 4。
- **第二步，排矩阵**：对角元 $\kappa(h,h) = \operatorname{tr}((\operatorname{ad}h)^2) = 2^2 + (-2)^2 = 8$，其余由对称性填齐。
- **第三步，判行列式**：$\det \kappa = 4 \cdot 4 \cdot 8 \neq 0$（主行列式非零），故非退化——$\mathfrak{sl}(2,\mathbb{C})$ 半单。

**核心要点**（判别法两方向）：

| 结构 | Killing 型表现 | 含义 |
| --- | --- | --- |
| 可解 | $\kappa(L, [L,L]) = 0$ | 度量在交换子方向上「塌缩」 |
| 半单 | $\kappa$ 非退化 | 度量满秩，无零方向 |

## 5 小结

- **Killing 型** $\kappa(x,y) = \operatorname{tr}(\operatorname{ad}x \operatorname{ad}y)$ 是天然的对称不变双线性型，天然地携带结构信息。
- **Cartan 判别法**：可解 ⟺ $\kappa(L,[L,L])=0$；**半单 ⟺ $\kappa$ 非退化**。
- **半单结构定理**：特征零半单 = 单理想的唯一理想直和；且 $L = \operatorname{Der}(L) = [L, L]$。
- 半单是「无退化」与「自充足」的同义语；它是本专题后半所有深结构的舞台。
- $\mathfrak{sl}(n,\mathbb{C})$（$n \ge 2$）是单而非半单原型，其 Killing 型 $\kappa(x,y) = 2n\operatorname{tr}(xy)$。

在下一节，我们将问：半单李代数的表示能否「拆开」——这引出本专题的中心定理之一，**Weyl 完全可约性定理**。
