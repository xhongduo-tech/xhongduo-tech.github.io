---
title: 全纯凸性、Runge 域与包络的存在性
date: 2026-08-07
---

# 全纯凸性、Runge 域与包络的存在性

<div class="epigraph">
<p>Runge 逼近说：在好的区域上，多项式并不逊色于任意全纯函数——解析的世界是被多项式撑起来的。</p>
<footer>—— 仿 卡尔 · 龙格（Carl Runge）与 安德烈 · 韦伊（André Weil），《解析函数的多项式逼近》</footer>
</div>

<div class="article-byline">
<p>第二级 · 多复变函数论 ｜ Hörmander 第2章；Krantz 第2章 ｜ 2026-08-07</p>
</div>

## 为什么从 Runge 域开始

前几篇我们反复用「全纯凸包」定义好区域，但有个尴尬：**定义里的 $\mathcal{O}(D)$ 是整个全纯函数空间，太大、太抽象**。能不能用更小的函数族（比如**多项式**）来代替？单复变中，Runge 定理早已给出漂亮的答案：在多项式凸的区域（如凸域）上，全纯函数可被多项式一致逼近。多复变中，**Runge 域**与 **Oka–Weil 定理**（下一节）把这个思想推广到 $\mathbb{C}^n$：若紧集 $K$ 关于多项式是凸的，则 $K$ 上的全纯函数可被多项式一致逼近。<span class="marginnote">Runge 域连接两件大事：一是<strong>逼近</strong>（用简单对象近似一般对象），二是<strong>凸性</strong>（多项式凸包 $\widehat K_{\mathcal{P}}$ 与全纯凸包 $\widehat K_{\mathcal{O}}$ 的关系）。Oka 的伟大工作证明：全纯凸域上两者重合。</span>

## 1 多项式凸性与 Runge 域

记 $\mathcal P$ 为 $\mathbb{C}^n$ 上的**多项式**全体。对紧集 $K$，定义**多项式凸包（polynomially convex hull）**：

$$
\widehat K_{\mathcal P} = \left\{ z \in \mathbb{C}^n : |p(z)| \leq \sup_K |p|, \; \forall p \in \mathcal P \right\}
$$

$K$ 称为**多项式凸**的，若 $\widehat K_{\mathcal P} = K$。注意这里凸包在全空间 $\mathbb{C}^n$ 里取，不是限制在某个区域中。<span class="marginnote">多项式凸包的几何直觉：把 $K$ 放进 $\mathbb{C}^n$，看哪些「额外点」无法被任何多项式用模区分——那些点就是凸包的填充。单复变中，多项式凸集 = 补集连通的紧集（Runge 定理的几何面）。</span>

**Runge 域**：区域 $D \subset \mathbb{C}^n$ 称为 **Runge 域**，若对每个紧集 $K \Subset D$，多项式凸包 $\widehat K_{\mathcal P}$ 与 $D$ 的交仍相对紧于 $D$，即 $\widehat K_{\mathcal P} \cap D \Subset D$。<span class="marginnote">直观：$D$ 不阻止多项式凸包「长出」$D$ 外的部分，但长出 $D$ 的那部分不能靠回 $D$ 的边界——或者说 $D$ 是「对多项式凸包稳定」的。凸域当然是 Runge 域；多圆柱、多球也都是。</span>

## 2 Runge 逼近定理（单复变经典）

回顾单变量：$K \subset \mathbb{C}$ 紧，$K$ 连通补集。则任何在 $K$ 邻域上全纯的函数都能被**多项式一致逼近**（在 $K$ 上）。等价说法：$K$ 是多项式凸的（因为单复变中 $\widehat K_{\mathcal P}$ 正是「补上被 $K$ 围住的洞」）。

这个定理的**证明精神**：先用 Runge 核（有理函数逼近），再用多项式逼近有理函数——两个步骤都由 Cauchy 积分完成。<span class="marginnote">为什么「补集连通」？因为若 $K$ 围住一个洞，洞内一点的值由边界积分决定，而边界积分涉及 $1/(\zeta-z)$——这是有理函数而非多项式。只有洞被填平，有理函数的极点才无处藏身，才能降级为多项式。</span>

多复变中，**Oka–Weil 定理**（下节主角）把这条推到：若 $K$ 是**多项式凸**紧集，则 $K$ 邻域上的全纯函数可被多项式一致逼近。

## 3 全纯凸包与多项式凸包：Oka 的洞察

在一般区域 $D$ 上，全纯凸包 $\widehat K_{\mathcal{O}(D)}$ 比多项式凸包 $\widehat K_{\mathcal P}$ **小**（因为 $\mathcal P \subset \mathcal{O}(D)$，约束更多……等等，约束更多包更小）。实际上对 $K \Subset D$：

$$
\widehat K_{\mathcal{O}(D)} \;\subset\; \widehat K_{\mathcal P} \cap D
$$

**Oka 的深刻定理**：若 $D$ 是**全纯凸**的，则上述包含是**等式**：

$$
\widehat K_{\mathcal{O}(D)} = \widehat K_{\mathcal P} \cap D
$$

这有多重后果：在全纯凸域中，全纯凸包可用（更易处理的）多项式凸包代替；且 $D$ 是 Runge 域当且仅当 $D$ 全纯凸且「$\widehat K_{\mathcal P}$ 不触边」。<span class="marginnote">Oka 证明这条时用了著名的 <strong>Oka 引理</strong>（把局部数据通过幂级数拼接）——它是多复变「局部到整体」的又一个范本。等号的意义：全纯函数「包得住」的区域，多项式同样包得住——解析刚性的全部力量浓缩在多项式里。</span>

## 4 公式解析：凸包包含关系与 Oka 等式

$$
\widehat K_{\mathcal{O}(D)} \;=\; \widehat K_{\mathcal P} \cap D \qquad (\text{$D$ 全纯凸})
$$

- **第一步，先证 $\subset$**：任意 $p \in \mathcal P \subset \mathcal{O}(D)$ 都在约束族里，故 $\widehat K_{\mathcal{O}(D)}$ 中的点自动满足 $|p(z)| \leq \sup_K|p|$，即 $z \in \widehat K_{\mathcal P}$；又 $z \in D$ 显然，故 $\subset$ 无代价成立。
- **第二步，理解难的是 $\supset$**：要证「多项式约束住了」的点，也逃不过「所有全纯函数的约束」。这是非平凡的：$\mathcal{O}(D)$ 比 $\mathcal P$ 大得多。Oka 的做法是对任意 $f \in \mathcal{O}(D)$ 与任意 $z_0 \in \widehat K_{\mathcal P}\cap D$，构造多项式列 $p_\nu \to f$（Runge 逼近），从而 $|f(z_0)| = \lim |p_\nu(z_0)| \leq \limsup \sup_K|p_\nu| = \sup_K|f|$。
- **第三步，抓住灵魂**：等式的本质是 **逼近定理 + 凸性** 的合体——Runge 逼近把「一般全纯函数」降维到「多项式」，凸性保证逼近在 $K$ 与 $z_0$ 上一致可行。全纯凸域恰是逼近可行的区域。

## 5 辨析与延伸：Runge 域的四个易错点

**辨析 1：多项式凸包 ≠ 全纯凸包**。多项式族 $\mathcal P$ 是 $\mathcal O(D)$ 的子集，约束更多、包更小——所以 $\widehat K_{\mathcal O} \subseteq \widehat K_{\mathcal P} \cap D$。初学者常把方向搞反。Oka 定理的贡献正是在全纯凸域上把包含变成等式。<span class="marginnote">记忆：函数族越大，约束越多，凸包越小。$\mathcal P \subset \mathcal O(D)$ ⟹ $\widehat K_{\mathcal O} \subseteq \widehat K_{\mathcal P}$。</span>

**辨析 2：Runge 域不是「多项式逼近的区域」的简单同义**。Runge 域要求多项式凸包不触边界，但不要求凸包被 $D$ 完全吸收——凸包可以伸出 $D$，只要伸出的部分不靠回边界。精确地说：$\widehat K_{\mathcal P} \cap D \Subset D$。

**辨析 3：单复变中 Runge 域 = 全纯域**。$n=1$ 时全纯域 = 任意开集，而 Runge 域的条件退化为平凡条件——所以「Runge 域」在 $n=1$ 是平凡概念，不值得单列。它和全纯域、伪凸域一样，都是 $n \geq 2$ 才「活」起来的对象。

**辨析 4：逼近与延拓的区别**。Runge 逼近说的是「用多项式**近似**全纯函数」；延拓说的是「把函数**扩展**到更大的区域」。两者不同但相关：延拓定理常通过逼近列构造。混淆「一致逼近」与「延拓」是最常见的概念错误。

**误区清单**：

- **误区 1**：以为「凸域 = Runge 域」方向反了。
  正解：凸域是 Runge 域（充分），但 Runge 域不一定是凸域。
- **误区 2**：以为「多项式凸包必在区域 D 内」。
  正解：多项式凸包在全空间 $\mathbb{C}^n$ 中取，可以伸出 $D$。
- **误区 3**：以为「逼近是在 $D$ 上整体进行的」。
  正解：逼近是在**紧集** $K$ 上的一致逼近，不是全区域一致逼近。
- **误区 4**：以为「Oka 等式无代价」。
  正解：$\widehat K_{\mathcal O} = \widehat K_{\mathcal P} \cap D$ 需要 $D$ 全纯凸，是深刻定理。

**术语表**：

| 中文 | 英文 | 说明 |
| --- | --- | --- |
| 多项式凸包 | polynomially convex hull | 由多项式模定义 |
| 全纯凸包 | holomorphic hull | 由全纯函数模定义 |
| Runge 域 | Runge domain | 多项式凸包不触边界 |
| Runge 逼近 | Runge approximation | 紧集上多项式一致逼近 |
| 补集连通 | connected complement | 单复变 Runge 条件 |
| Oka 等式 | Oka's identity | 全纯凸域上两凸包重合 |

## 6 历史注记与知识树

**历史**：Runge 定理（1885）是逼近理论的起点；Weil（1932）把「补集连通」推广到「多项式凸」；Oka（1936）去掉紧集限制得到全纯凸域版本。三十年里，逼近理论从单复变的工具成长为多复变的引擎。

**知识树**：

- 向后：Weierstrass 逼近定理（实分析）、单复变 Runge 定理（第二级《复变函数》）。
- 向前：Oka–Weil 定理（本组第 11 篇）、Levi 问题（本组第 10 篇）。
- 横向：凸分析的凸包理论（第二级《凸分析》）——「凸包」的多复变翻版。

**一句话记忆**：Runge 域 = 多项式凸包不触边的区域；Oka 等式 = 全纯凸域上「全纯凸包 = 多项式凸包 ∩ D」。

## 7 小结

- **多项式凸包** $\widehat K_{\mathcal P}$：由多项式模约束定义的紧集凸包；多项式凸集 = 凸包等于自身。
- **Runge 域**：多项式凸包不触边界的区域；凸域、多圆柱都是。
- **Runge 逼近**：单复变中「补集连通」紧集上全纯函数可被多项式一致逼近。
- **Oka 等式**：全纯凸域中全纯凸包 $=$ 多项式凸包 $\cap D$——逼近与凸性的完美统一。

多项式凸性提供了「全纯凸性」最可操作的版本：在全纯凸域上，一个抽象的凸包计算被降维成多项式模的检查——这是逼近与凸性统一带给我们的实际好处。

在下一节，我们迎来本组核心：**Levi 问题**——证明「伪凸 $\Rightarrow$