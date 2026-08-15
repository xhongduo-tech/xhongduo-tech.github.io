---
title: 相伴素与支集
date: 2026-08-07
---

# 相伴素与支集

<div class="epigraph">
<p>一个模的相伴素理想是它的「素谱」；支集则是它的「领土」。</p>
<footer>—— 交换代数课堂传统（经 Atiyah–Macdonald 系统化）</footer>
</div>

<div class="article-byline">
<p>第二级 · 交换代数 ｜ Atiyah–Macdonald Ch. 4 ｜ 2026-08-07</p>
</div>

## 为什么从相伴素开始

模论里，我们总想「把模拆成更小的块」。向量空间拆成直和；一般模拆不成，但有一族「最小零件」——**相伴素理想**（associated primes）：模中元素的零化子，恰好是素理想的那些。它们记录模的「病历卡」：模的非零部位、杀掉它的地方、以及准素分解的根。而**支集**（support）回答「模在哪里非零」——一个纯粹关于局部的集合。<span class="marginnote">相伴素的概念在 1920 年代 Krull 与 Noether 学派处理准素分解时逐渐成形：分解 $\mathfrak{a} = \cap \mathfrak{q}_i$ 的根 $\sqrt{\mathfrak{q}_i}$ 正是 $A/\mathfrak{a}$ 的相伴素。它在现代交换代数里比准素分解本体更常用——几乎每个定理都要先问「Ass 是什么」。</span>

这一篇把 $\operatorname{Ass} M$ 与 $\operatorname{Supp} M$ 讲清、讲透、讲准，让它们成为后续（以及回顾）所有局部论断的标准工具。

## 1 相伴素：模的病历卡

**相伴素（associated prime）**：设 $M$ 是 $A$-模，素理想 $\mathfrak{p}$ 称为 $M$ 的相伴素，若存在 $m \in M$，$m \neq 0$，使

$$\mathfrak{p} = \operatorname{ann}(m) = \{a \in A \mid am = 0\}.$$

$M$ 的相伴素全体记为 $\operatorname{Ass}_A M$（或 $\operatorname{Ass} M$）。

标准例子：
- 作为 $\mathbb{Z}$-模，$\operatorname{Ass} \mathbb{Z} = \{(0)\}$（$1$ 的零化子）；$\operatorname{Ass} \mathbb{Z}/6\mathbb{Z} = \{(2), (3)\}$；$\operatorname{Ass} \mathbb{Z}/p^\infty \mathbb{Z} = \{(p)\}$。
- $A$-模 $A/\mathfrak{a}$ 的相伴素 = 准素分解的根集 $\{\sqrt{\mathfrak{q}_i}\}$（第1篇《准素分解》）。
- $k[x,y]/(x^2, xy)$ 的相伴素：$\{(x), (x, y)\}$——其中 $(x,y)$ 是嵌入素理想。

**核心对照表：相伴素与支集**

| 模 $M$ | $\operatorname{Ass} M$ | $\operatorname{Supp} M$ |
| --- | --- | --- |
| $\mathbb{Z}$（$\mathbb{Z}$-模） | $\{(0)\}$ | $\operatorname{Spec} \mathbb{Z}$ |
| $\mathbb{Z}/6\mathbb{Z}$ | $\{(2), (3)\}$ | $\{(2), (3)\}$ |
| $\mathbb{Z}/p^\infty\mathbb{Z}$ | $\{(p)\}$ | $\{(p)\}$ |
| $k[x,y]/(x^2, xy)$ | $\{(x), (x,y)\}$ | $V((x)) = \{x=0\}$ |
| $A/\mathfrak{a}$ | 准素分解的根集 | $V(\mathfrak{a})$ |

最后一行是一切的母版：**$A/\mathfrak{a}$ 的相伴素就是第1篇准素分解的根集，支集就是零点集**——《准素分解》与《零点定理》在这里用同一个表格握手。

**重点：$\operatorname{Ass} M$ 的极小元素恰是 $\operatorname{Supp} M$ 的极小元素，也恰是「$\operatorname{ann} M$ 的极小素因子」。** 嵌入素理想（非极小的相伴素）是「多余的分支」，它们不贡献支集的边界，却贡献「深度杀手」（见《深度》）。<span class="marginnote">一句话直觉：$\operatorname{Ass} M$ =「$M$ 的元素被哪些素理想零化」；极小相伴素是「真正构成 $M$ 支撑的分支」，嵌入相伴素是「被吞进别的分支里的退化点」。</span>

**重点（Noether 模的关键性质）：$M \neq 0$ 时 $\operatorname{Ass} M \neq \emptyset$，且 Noether 模的 $\operatorname{Ass} M$ 是有限集。** 非空的证明用 Zorn：取零化子极大的元素，其零化子必为素理想。<span class="marginnote">有限性来自「降链 + 反链性质」：相伴素构成的反链在 Noether 环上有限（由升链条件保证）。「有限张病历卡」让 Noether 模可以逐个素理想地审问。</span>

这条「有限」是后续一切论证的发动机：深度、维数、局部上同调（第2篇）的「逐个 $\mathfrak{p}$ 检查」都靠它收束成有限步的验证——非有限生成的模没有这份便利，这也是为什么本专题几乎总默认有限生成。

## 2 支集：模的领土

**支集（support）**：$M$ 的支集

$$\operatorname{Supp} M = \{\mathfrak{p} \in \operatorname{Spec} A \mid M_{\mathfrak{p}} \neq 0\}.$$

即「$M$ 在哪些点局部化后仍非零」。标准例子：
- $\operatorname{Supp} \mathbb{Z} = \operatorname{Spec} \mathbb{Z}$（局部化 $\mathbb{Z}_{(p)} \neq 0$ 处处）；
- $\operatorname{Supp} \mathbb{Z}/6\mathbb{Z} = \{(2), (3)\}$；
- 有限生成 $A$-模 $M$ 的 $\operatorname{Supp} M = V(\operatorname{ann} M)$——由生成元的零化子决定。<span class="marginnote">「$M_{\mathfrak{p}} \neq 0$」比「$\mathfrak{p} \in \operatorname{Ass} M$」宽：$\mathbb{Z}$ 作为 $\mathbb{Z}$-模 $\operatorname{Supp} = \operatorname{Spec}\mathbb{Z}$ 而 $\operatorname{Ass} = \{(0)\}$。支集是「领土」（很大），相伴素是「重镇」（很少）。</span>

**重点：$\operatorname{Ass} M \subseteq \operatorname{Supp} M$，且两者有相同的极小元。** 于是

$$\operatorname{Supp} M = \bigcup_{\mathfrak{p} \in \operatorname{Ass} M} V(\mathfrak{p}),$$

支集被有限个相伴素的闭包「盖住」——Noether 模的支集是有限个不可约闭集的并。<span class="marginnote">几何翻译：$M$ 的支集在 $\operatorname{Spec} A$ 上像「闭子簇的有限并」，每块 $V(\mathfrak{p})$ 对应一个极小相伴素；嵌入相伴素只影响 $\operatorname{Ass}$ 不影响支集边界——与《准素分解》的「嵌入分支不贡献几何」完全一致。</span>

对有限生成模，支集完全由生成元决定：$M = (m_1, \dots, m_r)$ 时 $\operatorname{ann} M = \bigcap_i \operatorname{ann}(m_i)$，故 $\operatorname{Supp} M = V(\operatorname{ann} M)$。算例：$M = k[x]/(x) \oplus k[x]/(x-1)$ 有 $\operatorname{ann} M = (x)\cap(x-1) = (x^2-x)$，支集是 $\{(x), (x-1)\}$ 两个点——与逐点算 $M_{\mathfrak{p}}$ 结果一致。**「$M$ 的支集」与「$\operatorname{ann} M$ 的零点集」在这里是同一句话。**

## 3 局部化下的行为

**重点：局部化把相伴素「滤掉」那些与 $S$ 相交的：**

$$\operatorname{Ass}_{S^{-1}A}\big(S^{-1}M\big) = \{\mathfrak{p} \in \operatorname{Ass} M \mid \mathfrak{p} \cap S = \emptyset\}.$$

特别地，$\operatorname{Ass}_{A_{\mathfrak{p}}} M_{\mathfrak{p}} = \{\mathfrak{q} \in \operatorname{Ass} M \mid \mathfrak{q} \subseteq \mathfrak{p}\}$——**在点 $\mathfrak{p}$ 处局部看，相伴素恰好是原相伴素里「包含于 $\mathfrak{p}$」者**。这与第1篇《局部化》的素理想对应完全接榫。<span class="marginnote">这条公式是「局部判别整体」的又一范本：从 $\operatorname{Ass} M_{\mathfrak{p}}$（点附近的病历）拼回 $\operatorname{Ass} M$（全谱的病历）。它也让「$\operatorname{Ass}$ 是局部-有限的」成为「用局部化审问模」的合法手续。</span>

**支集的局部化版本**：$\operatorname{Supp} M = \{\mathfrak{p} \mid M_{\mathfrak{p}} \neq 0\}$ 天然局部——判别支集归属时，只要看一个局部化是否为零。两者联手给出：

$$M = 0 \iff \operatorname{Supp} M = \emptyset \iff \operatorname{Ass} M = \emptyset \qquad (\text{Noether})$$

**辨析｜易错点：** $\operatorname{Ass} M \neq \emptyset$ 只对 $M \neq 0$ 且（有限生成时）才稳定成立；但对**任意** $A$-模，$M \neq 0$ ⇒ $\operatorname{Ass} M \neq \emptyset$ 在 Noether 环上成立。初学者常默认「支集非空 ⇔ 模非零」——对有限生成模正确，对一般模（如某些大模）$\operatorname{Supp} M$ 可能非空而 $\operatorname{Ass} M$ 空。**先确认有限生成。**

用 $\mathbb{Z}/6\mathbb{Z}$ 把局部化公式走一遍：$\operatorname{Ass} = \{(2), (3)\}$。在点 $(2)$ 处局部化，$S = \mathbb{Z}\setminus(2)$，$3 \in S$ 与 $(3)$ 相交，故 $(3)$ 被滤掉，剩下 $\operatorname{Ass}_{\mathbb{Z}_{(2)}}(\mathbb{Z}/6)_{(2)} = \{(2)\}$——点 $(2)$ 附近只看见「$2$ 的功劳」。这正对应 $(\mathbb{Z}/6)_{(2)} \cong \mathbb{Z}/2$。**局部化把不相干的素理想「按点过滤」**，这条公式是「审问模」的标准手续。

## 4 公式解析：支集 = 零化子的簇

对有限生成 $A$-模 $M$：

$$\operatorname{Supp} M = V(\operatorname{ann} M), \qquad \operatorname{Supp} M = \bigcup_{\mathfrak{p} \in \operatorname{Ass} M} V(\mathfrak{p}).$$

- **第一步，第一式**：$M_{\mathfrak{p}} \neq 0 \iff \mathfrak{p} \supseteq \operatorname{ann} M$。方向「$\supseteq$ 使非零」：$\operatorname{ann} M \subseteq \mathfrak{p}$ 时，$M$ 中「不被 $\mathfrak{p}$ 之外的元素零化」的非零元素在局部化中存活；反方向：若 $\mathfrak{p} \not\supseteq \operatorname{ann} M$，取 $a \in \operatorname{ann} M \setminus \mathfrak{p}$，则 $a$ 在 $A_{\mathfrak{p}}$ 中可逆、$aM = 0$ 推得 $M_{\mathfrak{p}} = 0$。<span class="marginnote">第二式是关键的「盖住」：有限生成时 $\operatorname{Supp} M$ 由极小相伴素的闭包并出；嵌入相伴素不改变并。两式合一：<strong>领土 = 极小重镇的闭包并 = 零化子的簇</strong>。</span>
- **第二步，第二式**：$\operatorname{Ass} M \subseteq \operatorname{Supp} M$ 显然（$\mathfrak{p} \in \operatorname{Ass}$ 时 $M_{\mathfrak{p}}$ 含 $\mathfrak{p}$-零化的非零元）；反向用「极小元相同」——$\operatorname{Supp}$ 的极小元必为相伴素（局部化 + 零化子论证），而 $\operatorname{Supp}$ 由极小元的闭包并覆盖。
- **第三步，用途**：$M$ 的维数 $\dim M = \dim \operatorname{Supp} M = \max_{\mathfrak{p} \in \operatorname{Ass} M} \dim A/\mathfrak{p}$——**模的维数被其相伴素的维数支配**。这正是《维数理论》与《深度》里反复使用的「维数 = 支撑闭集维数」的精确版本。
- **第四步，特例**：若 $M$ 是 $A/\mathfrak{p}$ 的有限扩张，则 $\dim M = \dim A/\mathfrak{p}$ 且 $\operatorname{Ass} M = \{\mathfrak{p}\}$ 是单点集。于是「$\operatorname{Ass}$ 是单点」⇔「支集是不可约闭集」——几何里的「子簇」在模论里就是「相伴素单点」的模。

**辨析｜易错点：** $\operatorname{Ass} M$ 与「$A/\mathfrak{p}$ 是 $M$ 的子商」的关系：$\mathfrak{p} \in \operatorname{Ass} M$ ⇔ $A/\mathfrak{p}$ 嵌入 $M$。这不是「商」，是「嵌入」——初学者把方向弄反就全盘皆错。判据永远回到 $\mathfrak{p} = \operatorname{ann}(m)$ 这个原始定义。

**术语速查表**

| 术语 | 一句话含义 |
| --- | --- |
| 相伴素 $\operatorname{Ass} M$ | 元素零化子为素者 |
| 支集 $\operatorname{Supp} M$ | 局部化非零的点 |
| 极小相伴素 | 支集的极小元，决定边界 |
| 嵌入素理想 | 非极小的相伴素，深度杀手 |
| $\operatorname{ann}(m)$ | 零化单个元素的理想 |
| $\operatorname{ann} M$ | 零化整个模的元素之集 |

## 5 小结

- **相伴素** $\operatorname{Ass} M$：元素零化子为素者；极小元 = 支集的极小元 = $\operatorname{ann} M$ 的极小素因子；Noether 模有限。
- **支集** $\operatorname{Supp} M = \{\mathfrak{p} \mid M_{\mathfrak{p}} \neq 0\}$；有限生成时 $= V(\operatorname{ann} M)$。
- 局部化公式：$\operatorname{Ass}_{S^{-1}A} S^{-1}M = \{\mathfrak{p} \in \operatorname{Ass} M : \mathfrak{p} \cap S = \emptyset\}$。
- $\dim M = \max\{\dim A/\mathfrak{p} : \mathfrak{p} \in \operatorname{Ass} M\}$；$M = 0 \iff \operatorname{Ass} M = \emptyset$（Noether）。

在下一节，我们做整个专题的「天空之眼」收束：**局部上同调**——用导出函子把 $\operatorname{Supp}$、深度、维数、对偶性全部装进同调语言，看这最后一块拼图如何把所有线索连成环。
