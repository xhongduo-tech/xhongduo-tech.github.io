---
title: 最高权表示与 Verma 模
date: 2026-08-07
---

# 最高权表示与 Verma 模

<div class="epigraph">
<p>每一个不可约表示，都在某个意义上由它的「极点」所决定。</p>
<footer>—— 哈米什 · 康利-钱德拉（Harish-Chandra，意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 李代数与李群 ｜ Humphreys §20-21 ｜ 2026-08-07</p>
</div>

## 为什么需要无穷维的 Verma 模

第 6 篇我们看到 $\mathfrak{sl}(2)$ 的不可约表示由最高权（整数）标记。对一般的半单李代数，同样的图景成立，但构造更微妙：**并不是每个「候选最高权」都能直接得到有限维不可约表示**。我们需要先造一个「万能的最大表示」——**Verma 模**——然后看它坍缩成什么。<span class="marginnote">Verma 模（Dayanand Verma, 1968）是最高权理论的枢纽：它把所有「重量足够大的表示」统一收进一个对象里。1990 年前后 Kazhdan–Lusztig 猜想证明它（经仿射 Hecke 代数）的完备推广，至今仍是表示论核心研究方向之一。</span>

一个更高的视角：最高权表示的分类其实是「权格上的数据」与「表示对象」之间的字典。权格 $\Lambda$ 是离散的、组合的、可枚举的；表示空间 $L(\lambda)$ 是线性的、解析的、携带维数的。Verma 模与支配整条件就是这本字典的「查表规则」——读表时，$\lambda$ 的每个整数取值给出一个 $L(\lambda)$，而「非整」与「负」的 $\lambda$ 被标记为无穷维。理解这条字典机制，就等于抓住了半单李代数表示论的总纲。

## 1 权与最高权

设 $H$ 是半单 $L$ 的 Cartan 子代数，$H^*$ 的整元素集称为**权格（weight lattice）**：

$$\Lambda = \{ \lambda \in H^* \mid \lambda(h_\alpha) \in \mathbb{Z} \ \forall \alpha \in \Phi \}$$

**权（weight）**：$L$ 的表示 $V$ 中，$H$ 的联合特征空间 $V_\lambda = \{ v \mid h v = \lambda(h) v \}$ 非零时称 $\lambda$ 是 $V$ 的权。**最高权（highest weight）**：在某个「偏序」下最大的权。<span class="marginnote">对简单根 $\Pi$，定义 $\lambda \ge \mu \iff \lambda - \mu$ 是简单根的非负整数组合。最高权即偏序极大元。偏序把「最大」的含混概念变成了可计算的比较规则。</span>

偏序的直观：$\lambda \ge \mu$ 意味着「$\mu$ 处在 $\lambda$ 的下方」，且差能被简单根的非负整组合凑出。对 $A_2$，若取 $\lambda = \omega_1 + \omega_2$（最高的支配整权），则所有低于它的权都可由它减去非负组合得到。最高权是这条偏序链的顶——表示论中一切「从高到低」的生成（$f$ 作用、负根生成）都以这条偏序为轨道，顺着它一步步下降。

**Weyl 群作用**：$W$ 自然地作用在 $H^*$ 上（通过余根对偶），权格在 $W$ 作用下不变。这是表示论中「Weyl 群作用于权」的基础。具体地，$W$ 的生成元是简单根反射 $s_i(\lambda) = \lambda - \langle \lambda, \alpha_i^\vee\rangle \alpha_i$，而 Cartan 整数 $\langle\lambda, \alpha_i^\vee\rangle$ 对 $\lambda \in \Lambda$ 取整数值——整系数的线性组合仍在 $\Lambda$ 中，故权格在 $W$ 下封闭。这把「群作用于权」从抽象声明变成可逐点计算的操作，第 12 篇的特征标公式将反复用到这条规则。

## 2 Verma 模的构造

设 $\lambda \in \Lambda$。**Verma 模（Verma module）** $M(\lambda)$ 定义为：

$$M(\lambda) = U(L) / \langle n_+ ,\ h - \lambda(h) \mid h \in H \rangle$$

更具体地，$M(\lambda)$ 是 $U(L)$ 模，由单个最高权向量 $v_\lambda$（$e_\alpha v_\lambda = 0$ 对所有正根 $\alpha$，$h v_\lambda = \lambda(h) v_\lambda$）生成，且 $U(L)$ 自由作用在其上。<span class="marginnote">$n_+ = \bigoplus_{\alpha > 0} L_\alpha$ 是正根空间；$U(n_-) = U(\bigoplus_{\alpha<0}L_\alpha)$ 是负根方向的包络代数。PBW 定理给出：$M(\lambda) \cong U(n_-) \otimes \mathbb{C}v_\lambda$ 作为向量空间——Verma 模就是「从最高权往下撒 $U(n_-)$ 的全部元素」。</span>

自由性的精确含义：$M(\lambda) \cong U(n_-) \otimes \mathbb{C}v_\lambda$ 意味着「$U(n_-)$ 中没有任何元素会意外地把 $v_\lambda$ 杀死」。这由 PBW 保证：$U(n_-)$ 的元素在标准单项式基下彼此线性无关，而 $v_\lambda$ 被定义成「最高权」——$e_\alpha$ 杀它，$f_\alpha$ 则自由作用。若 PBW 不成立，$U(n_-)$ 会出现隐藏关系，Verma 模就会「缩小」，最高权定理将失去地基。

**核心事实（Verma 模的结构）**：

作为 $U(n_-)$-模，$M(\lambda)$ 由 $U(n_-)$ 生成且自由（PBW），因此基由负根方向的严格有序单项式给出。
- $M(\lambda)$ 有**唯一**的极大子模 $M'(\lambda)$（这是关键引理，靠 PBW 的「有序性」证明）。
- **最高权定理**：不可约最高权表示 $L(\lambda) = M(\lambda)/M'(\lambda)$。每个不可约 $L$-模都是某个 $L(\lambda)$，且 $L(\lambda) \cong L(\mu) \iff \lambda = \mu$。<span class="marginnote">也就是说：不可约表示 ⟺ 最高权 ⟺ 权格 $\Lambda$ 的一个元素。这是 $\mathfrak{sl}(2)$「$n$ 标记表示」对一般半单代数的精确推广。</span>

**术语速查表**：

| 术语 | 记号 | 含义 |
| --- | --- | --- |
| 权格 | $\Lambda$ | $H^*$ 中满足 $\lambda(h_\alpha) \in \mathbb{Z}$ 的泛函全体 |
| 支配整权 | $\Lambda^+$ | 在简单根上取非负整数值的权 |
| 最高权向量 | $v_\lambda$ | 满足 $e_\alpha v_\lambda = 0$、$h v_\lambda = \lambda(h) v_\lambda$ 的向量 |
| Verma 模 | $M(\lambda)$ | $U(L)/\langle n_+, h - \lambda(h)\rangle$ |
| 极大子模 | $M'(\lambda)$ | $M(\lambda)$ 的极大真子模，商出不可约模 |
| 不可约最高权模 | $L(\lambda)$ | $M(\lambda)/M'(\lambda)$，分类的最终对象 |

## 3 有限维性与支配整权

什么时候 $L(\lambda)$ 是有限维的？答案是「支配整权」。**支配整权（dominant integral weight）**：

$$\lambda \in \Lambda^+ \iff \lambda(h_\alpha) \ge 0 \quad \forall \alpha \in \Phi^+ \text{（或对简单根）}$$

**核心定理（最高权表示的分类）**：对复半单 $L$，

> 不可约有限维表示 $L(\lambda)$ ⟺ $\lambda$ 是支配整权。且 $L(\lambda)$ 的权谱关于 Weyl 群对称，最高权恰为 $\lambda$。

**辨析｜易错点：** Verma 模 $M(\lambda)$ 对**任何** $\lambda$ 都有定义，但不可约化后 $L(\lambda)$ 有限维**仅当** $\lambda$ 支配整。初学者常误以为「最高权表示 = 有限维表示」——实际上最高权表示始终存在，有限维性是额外的「支配整」条件。对 $\lambda = -1$（负权），$L(\lambda)$ 是无穷维的。<span class="marginnote">支配整权的有限性依赖 $\mathfrak{sl}(2)$ 子代数的表示论：从每个简单根出发的 $\mathfrak{sl}(2)$ 三元组要求 $\lambda(h_{\alpha_i})$ 为非负整数才能截断——这与第 6 篇「权必为整数」一脉相承。</span>

**数值例**：对 $\mathfrak{sl}(2,\mathbb{C})$，权格 $\Lambda = \mathbb{Z}$，支配整权 $\Lambda^+ = \{0, 1, 2, \dots\}$。$\lambda = 3$ 对应四维表示 $V_3$（$L(3) \cong V_3$）；而 $\lambda = -1$ 的 Verma 模 $M(-1)$ 不可约且无穷维——负最高权根本不是支配整权，却依然拥有合法的最高权表示。这个对比说明：**「存在最高权表示」与「存在有限维最高权表示」是两个完全不同的问题**，前者对所有权成立，后者只对 $\Lambda^+$ 成立。

**辨析｜支配整与「整」**：注意区分三个层次——「权格 $\Lambda$」要求 $\lambda(h_\alpha) \in \mathbb{Z}$（整权）；「支配整 $\Lambda^+$」进一步要求非负；「严格支配」则要求 $> 0$。$\rho = \tfrac12\sum_{\alpha>0}\alpha$ 属于严格支配整权。第 12 篇 Weyl 特征标公式里出现 $\lambda + \rho$ 的原因，正是要利用这个「严格」来避免分母为零——一个看似微小的记号选择，背后是防止除零的考虑。

值得再强调一次：**判定 $\lambda \in \Lambda^+$ 只需检查简单根，不需要检查全部正根**。因为任意正根都是简单根的非负整组合，若 $\lambda$ 在每个简单根上非负，则它在每个正根上自动非负。这个「从 $\ell$ 个简单根出发覆盖所有 $\frac12(\dim L - \ell)$ 个正根」的简化，让支配整条件从「检查半打不等式」压缩成「检查 $\ell$ 个不等式」——在实际计算最高权时极其省力。

## 4 公式解析：Verma 模的基与权重

以 $\mathfrak{sl}(2,\mathbb{C})$ 为例完全算清 $M(\lambda)$。基取 $n_- = \mathbb{C}f$，PBW 给出

$$M(\lambda) = \operatorname{span}\{ f^k v_\lambda \mid k \ge 0 \}, \qquad h(f^k v_\lambda) = (\lambda - 2k) f^k v_\lambda$$

- **第一步，用 $U(n_-)$**：$M(\lambda) = U(n_-) \cdot v_\lambda$，而 $U(n_-)$ 的基是 $f^k$（$k \ge 0$）。
- **第二步，算权**：$h f^k v_\lambda = (\lambda - 2k) f^k v_\lambda$（利用 $[h, f] = -2f$ 归纳），权依次为 $\lambda, \lambda-2, \lambda-4, \dots$。
- **第三步，看有限性**：当 $\lambda = n$ 非负整数时，$f^{n+1} v_\lambda = 0$ 恰好成立（由 $f e$ 反推关系），$M(n)$ 坍缩为 $n+1$ 维不可约表示 $V_n$；当 $\lambda$ 为负或半整数时，$M(\lambda)$ 是无穷维的。

**第三步（续）**：$h(f^k v_\lambda) = (\lambda - 2k) f^k v_\lambda$ 给出的权谱是 $\lambda, \lambda-2, \lambda-4, \dots$——一维算术列。这正是 $\mathfrak{sl}(2)$ 有限维表示的权谱「截断」前的情形；截断点 $k = n+1$ 由 $f^{n+1} v_\lambda = 0$ 决定。把这个「截断条件」翻译成最高权语言：$e$ 方向的 $\mathfrak{sl}(2)$ 链要求 $\lambda$ 是非负整数，$f$ 方向则无条件。对一般半单代数，截断条件逐个加在**每个简单根方向**上——支配整条件就是这么长出来的。

**核心要点**：Verma 模的结构由 PBW 定理完全决定——它就是「从最高权出发，让负根方向自由生成」的自由模。唯一的复杂性在于它可能不「精简」（有极大子模要商掉），而**何时精简**正是支配整条件的物理意义。<span class="marginnote">在共形场论与仿射李代数中，Verma 模的「退化」决定 Kac 行列式与特征标公式——第 12 篇的 Weyl 特征标公式正是处理「$L(\lambda)$ 的权多重度」的精密工具。</span>

再算 $\mathfrak{sl}(3,\mathbb{C})$ 的一个 Verma 模的权谱：取 $\lambda = \omega_1$（第一基本权），正根为 $\alpha_1, \alpha_2, \alpha_1 + \alpha_2$。由 $M(\lambda) \cong U(n_-) \otimes \mathbb{C}v_\lambda$，负根方向有 $f_{\alpha_1}, f_{\alpha_2}$ 两个生成元，于是 $M(\omega_1)$ 的权谱含 $\omega_1, \omega_1-\alpha_1, \omega_1-\alpha_2, \omega_1-\alpha_1-\alpha_2, \omega_1-2\alpha_1-\alpha_2, \dots$。当 $\lambda$ 支配整时这些权里的负权部分会被极大子模吸收，剩下有限个权；这正是第 12 篇 Weyl 特征标公式要精确计数的对象。

## 5 权谱与 Weyl 群对称

有限维表示 $L(\lambda)$ 的权谱有一个漂亮的对称性：**它被 Weyl 群 $W$ 整体不变**。若 $\mu$ 是权且重数为 $m_\mu$，则 $w\mu$ 也是权且重数相同，对一切 $w \in W$ 成立。因此权谱总是以 Weyl 群的轨道为「砖块」拼成。

**为什么必然对称？** 因为 $W$ 由根反射生成，而每对根 $\pm\alpha$ 对应一个 $\mathfrak{sl}(2)$ 三元组（第 7 篇）。$\mathfrak{sl}(2)$ 的表示论告诉我们：整条权链 $n, n-2, \dots, -n$ 关于 0 对称；对一般表示，每个 $\mathfrak{sl}(2)$ 子链把权谱沿 $\alpha$ 方向对称化。所有方向对称化的叠加，就是整个 $W$ 作用的不变性。

**举例**：$A_2$ 中 $L(\omega_1)$（标准三维表示）的权谱 $\{\omega_1, \omega_1-\alpha_1, \omega_1-\alpha_2\}$——三个权构成 $W \cong S_3$ 的一条轨道，六个群元素把三角形顶点搬来搬去但不离开这三点。这也解释了为什么「最高权」的选择看似任意、权谱却与选择无关：$W$ 轨道把任何「最高权」的决定性信息都锁进了同一个谱。

**数值自检**：对 $\mathfrak{sl}(2)$，$V_3$ 的权谱 $\{-3,-1,1,3\}$。$W = \{1, s\}$ 只有两个元素，$s$ 把 $\mu$ 送到 $-\mu$：$3 \leftrightarrow -3$、$1 \leftrightarrow -1$——恰好两两配对，谱在 $W$ 下封闭。任何一个支配整权的有限维表示，其权谱都满足这条「W-封闭性」。

这条对称性还回答了「为什么负权不能出现在不可约模里」：若 $-\lambda$ 是权而 $\lambda$ 不是，$W$ 反射会立刻把它配对回来。事实上「最高权」之所以能作为标签，正因为 $W$ 的轨道让每个 $W$-等价类都有唯一的支配代表——这就是为什么我们可以说「权格 / $W$ 的支配元」而不会造成歧义。下一节的 Weyl 特征标公式，正是把这条轨道结构写成了显式的求和。

## 6 小结

- **权格** $\Lambda$ 与**支配整权** $\Lambda^+$：最高权表示的分类空间。
- **Verma 模** $M(\lambda)$：$U(L)/\langle n_+, h - \lambda(h)\rangle$，由最高权向量生成、$U(n_-)$ 自由模。
- **最高权定理**：每个不可约模 = 某个 $L(\lambda) = M(\lambda)/M'(\lambda)$，且 $\lambda$ 唯一确定表示。
- **有限维分类**：$L(\lambda)$ 有限维 ⟺ $\lambda \in \Lambda^+$；此时权谱 $W$-对称。
- **可积性视角**：有限维表示对应「可积最高权模」，其权谱被 Weyl 群对称地封锁；无穷维 Verma 模是「未截断」的极限。
- 对 $\mathfrak{sl}(2)$，$M(n)$ 坍缩为 $V_n$、$M(-1)$ 保持无穷维——支配整是唯一的分水岭。
- PBW + 支配整条件是理论的地基；Verma 模把「任意权」都纳入可计算的轨道。
- Verma 模是最高权理论的「通解」，不可约模是它在支配整条件下的「特解」。
- 判定支配整只需检查 $\ell$ 个简单根方向——这是最高权计算里最省力的实用技巧。

在下一节，我们将为最高权表示的权谱写出显式公式——**Weyl 特征标公式**，并理解为什么分母中出现 $\rho$ 与 Weyl 群。
