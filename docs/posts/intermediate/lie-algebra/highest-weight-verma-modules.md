---
title: 最高权表示与 Verma 模
date: 2026-08-11
---

# 最高权表示与 Verma 模

<div class="epigraph">
<p>每一个不可约表示，都在某个意义上由它的「极点」所决定。</p>
<footer>—— 哈米什 · 康利-钱德拉（Harish-Chandra，意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 李代数与李群 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么需要无穷维的 Verma 模

第 6 篇我们看到 $\mathfrak{sl}(2)$ 的不可约表示由最高权（整数）标记。对一般的半单李代数，同样的图景成立，但构造更微妙：**并不是每个「候选最高权」都能直接得到有限维不可约表示**。我们需要先造一个「万能的最大表示」——**Verma 模**——然后看它坍缩成什么。<span class="marginnote">Verma 模（Dayanand Verma, 1968）是最高权理论的枢纽：它把所有「重量足够大的表示」统一收进一个对象里。1990 年前后 Kazhdan–Lusztig 猜想证明它（经仿射 Hecke 代数）的完备推广，至今仍是表示论核心研究方向之一。</span>

## 1 权与最高权

设 $H$ 是半单 $L$ 的 Cartan 子代数，$H^*$ 的整元素集称为**权格（weight lattice）**：

$$\Lambda = \{ \lambda \in H^* \mid \lambda(h_\alpha) \in \mathbb{Z} \ \forall \alpha \in \Phi \}$$

**权（weight）**：$L$ 的表示 $V$ 中，$H$ 的联合特征空间 $V_\lambda = \{ v \mid h v = \lambda(h) v \}$ 非零时称 $\lambda$ 是 $V$ 的权。**最高权（highest weight）**：在某个「偏序」下最大的权。<span class="marginnote">对简单根 $\Pi$，定义 $\lambda \ge \mu \iff \lambda - \mu$ 是简单根的非负整数组合。最高权即偏序极大元。偏序把「最大」的含混概念变成了可计算的比较规则。</span>

**Weyl 群作用**：$W$ 自然地作用在 $H^*$ 上（通过余根对偶），权格在 $W$ 作用下不变。这是表示论中「Weyl 群作用于权」的基础。

## 2 Verma 模的构造

设 $\lambda \in \Lambda$。**Verma 模（Verma module）** $M(\lambda)$ 定义为：

$$M(\lambda) = U(L) / \langle n_+ ,\ h - \lambda(h) \mid h \in H \rangle$$

更具体地，$M(\lambda)$ 是 $U(L)$ 模，由单个最高权向量 $v_\lambda$（$e_\alpha v_\lambda = 0$ 对所有正根 $\alpha$，$h v_\lambda = \lambda(h) v_\lambda$）生成，且 $U(L)$ 自由作用在其上。<span class="marginnote">$n_+ = \bigoplus_{\alpha > 0} L_\alpha$ 是正根空间；$U(n_-) = U(\bigoplus_{\alpha<0}L_\alpha)$ 是负根方向的包络代数。PBW 定理给出：$M(\lambda) \cong U(n_-) \otimes \mathbb{C}v_\lambda$ 作为向量空间——Verma 模就是「从最高权往下撒 $U(n_-)$ 的全部元素」。</span>

**核心事实（Verma 模的结构）**：

作为 $U(n_-)$-模，$M(\lambda)$ 由 $U(n_-)$ 生成且自由（PBW），因此基由负根方向的严格有序单项式给出。
- $M(\lambda)$ 有**唯一**的极大子模 $M'(\lambda)$（这是关键引理，靠 PBW 的「有序性」证明）。
- **最高权定理**：不可约最高权表示 $L(\lambda) = M(\lambda)/M'(\lambda)$。每个不可约 $L$-模都是某个 $L(\lambda)$，且 $L(\lambda) \cong L(\mu) \iff \lambda = \mu$。<span class="marginnote">也就是说：不可约表示 ⟺ 最高权 ⟺ 权格 $\Lambda$ 的一个元素。这是 $\mathfrak{sl}(2)$「$n$ 标记表示」对一般半单代数的精确推广。</span>

## 3 有限维性与支配整权

什么时候 $L(\lambda)$ 是有限维的？答案是「支配整权」。**支配整权（dominant integral weight）**：

$$\lambda \in \Lambda^+ \iff \lambda(h_\alpha) \ge 0 \quad \forall \alpha \in \Phi^+ \text{（或对简单根）}$$

**核心定理（最高权表示的分类）**：对复半单 $L$，

> 不可约有限维表示 $L(\lambda)$ ⟺ $\lambda$ 是支配整权。且 $L(\lambda)$ 的权谱关于 Weyl 群对称，最高权恰为 $\lambda$。

**辨析｜易错点：** Verma 模 $M(\lambda)$ 对**任何** $\lambda$ 都有定义，但不可约化后 $L(\lambda)$ 有限维**仅当** $\lambda$ 支配整。初学者常误以为「最高权表示 = 有限维表示」——实际上最高权表示始终存在，有限维性是额外的「支配整」条件。对 $\lambda = -1$（负权），$L(\lambda)$ 是无穷维的。<span class="marginnote">支配整权的有限性依赖 $\mathfrak{sl}(2)$ 子代数的表示论：从每个简单根出发的 $\mathfrak{sl}(2)$ 三元组要求 $\lambda(h_{\alpha_i})$ 为非负整数才能截断——这与第 6 篇「权必为整数」一脉相承。</span>

## 4 公式解析：Verma 模的基与权重

以 $\mathfrak{sl}(2,\mathbb{C})$ 为例完全算清 $M(\lambda)$。基取 $n_- = \mathbb{C}f$，PBW 给出

$$M(\lambda) = \operatorname{span}\{ f^k v_\lambda \mid k \ge 0 \}, \qquad h(f^k v_\lambda) = (\lambda - 2k) f^k v_\lambda$$

- **第一步，用 $U(n_-)$**：$M(\lambda) = U(n_-) \cdot v_\lambda$，而 $U(n_-)$ 的基是 $f^k$（$k \ge 0$）。
- **第二步，算权**：$h f^k v_\lambda = (\lambda - 2k) f^k v_\lambda$（利用 $[h, f] = -2f$ 归纳），权依次为 $\lambda, \lambda-2, \lambda-4, \dots$。
- **第三步，看有限性**：当 $\lambda = n$ 非负整数时，$f^{n+1} v_\lambda = 0$ 恰好成立（由 $f e$ 反推关系），$M(n)$ 坍缩为 $n+1$ 维不可约表示 $V_n$；当 $\lambda$ 为负或半整数时，$M(\lambda)$ 是无穷维的。

**核心要点**：Verma 模的结构由 PBW 定理完全决定——它就是「从最高权出发，让负根方向自由生成」的自由模。唯一的复杂性在于它可能不「精简」（有极大子模要商掉），而**何时精简**正是支配整条件的物理意义。<span class="marginnote">在共形场论与仿射李代数中，Verma 模的「退化」决定 Kac 行列式与特征标公式——第 12 篇的 Weyl 特征标公式正是处理「$L(\lambda)$ 的权多重度」的精密工具。</span>

## 5 小结

- **权格** $\Lambda$ 与**支配整权** $\Lambda^+$：最高权表示的分类空间。
- **Verma 模** $M(\lambda)$：$U(L)/\langle n_+, h - \lambda(h)\rangle$，由最高权向量生成、$U(n_-)$ 自由模。
- **最高权定理**：每个不可约模 = 某个 $L(\lambda) = M(\lambda)/M'(\lambda)$，且 $\lambda$ 唯一确定表示。
- **有限维分类**：$L(\lambda)$ 有限维 ⟺ $\lambda \in \Lambda^+$；此时权谱 $W$-对称。
- PBW + 支配整条件是理论的地基；Verma 模把「任意权」都纳入可计算的轨道。

在下一节，我们将为最高权表示的权谱写出显式公式——**Weyl 特征标公式**，并理解为什么分母中出现 $\rho$ 与 Weyl 群。
