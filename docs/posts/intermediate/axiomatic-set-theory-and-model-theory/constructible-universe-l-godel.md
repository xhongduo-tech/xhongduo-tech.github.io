---
title: 可构造宇宙 L 与 Gödel 的一致性结果
date: 2026-08-07
---

# 可构造宇宙 L 与 Gödel 的一致性结果

<div class="epigraph">
<p>宇宙 L 是最小的宇宙：它只装那些「被一阶定义逼迫存在」的集合，于是选择与连续统都只能在它的屋檐下安身。</p>
<footer>—— 库尔特 · 哥德尔（Kurt Gödel）</footer>
</div>

<div class="article-byline">
<p>第二级 · 公理集合论与模型论 ｜ Jech, <em>Set Theory</em> 第13章；Kunen 第8章 ｜ 2026-08-07</p>
</div>

## 为什么从可构造宇宙开始

从第1篇到第2篇，我们一直在 ZFC 的地基上盖房子。但「ZFC 的一致性」本身呢？Gödel 第二不完备定理说：若 ZFC 一致，ZFC 无法证明自己一致。于是集合论者转向**相对一致性**：要证「ZFC + $\neg$CH 一致」，只需证「ZFC 一致 ⇒ ZFC + $\neg$CH 一致」——这需要一座「模型加工厂」。**可构造宇宙 $L$** 就是第一台这样的机器。<span class="marginnote">Gödel 在 1938 年构造 $L$，证明 ZFC + CH + AC 的相对一致性；Cohen 在 1963 年用力迫证明 $\neg$CH 的相对一致性。$L$ 是最先被严格研究的「内模型」，也是「内模型程序」（inner model program）的开端——后来诞生了 $L[\mu]$、核心模型等更深的理论。</span>

今天的目标：定义 $L$ 的建造过程（用「可定义子集」取代「一切子集」），证明 $L$ 满足 ZF，然后展示 $L$ 里为什么 CH 与 AC 都成立。$L$ 与 $V$ 的最大区别是**极简主义**：$L$ 只保留「被一阶公式从已有集合里定义出来的子集」，于是它比 $V$ 小得多，却依然自洽地承载整个 ZFC。

## 1 可定义闭包：给宇宙装上「省料的引擎」

$V$ 的层级用幂集 $\mathcal{P}(V_\alpha)$ 建造——允许一切子集。$L$ 换成一个更省料的算子：**可定义子集**。

**可定义子集**：$X \subseteq A$ 是 $A$ 的**可定义子集**，若存在一阶公式 $\varphi(v_0, \dots, v_n)$ 与参数 $a_1, \dots, a_n \in A$，使得

$$
X = \{x \in A : (A, \in) \vDash \varphi(x, a_1, \dots, a_n)\}
$$

记 $X \in \mathrm{Def}(A)$。<span class="marginnote">可定义性用「$A$ 里的一阶公式」定义——不许引用 $A$ 之外的集合，也不许用二阶量词。$\mathrm{Def}(A) \subseteq \mathcal{P}(A)$，但通常小得多。例：$\mathrm{Def}(\omega) = \mathrm{Def}(V_\omega)$ 仍然可数，因为一阶公式只有可数多条。</span>

直觉：可定义子集是「$A$ 内部能指认的子集」。$V$ 允许「在外部偶然存在」的子集，$L$ 只保留「内部公式能点名」的子集。

## 2 $L$ 的层级：只用定义造集合

可构造层级用超限递归定义：

$$
L_0 = \emptyset, \qquad L_{\alpha+1} = \mathrm{Def}(L_\alpha), \qquad L_\lambda = \bigcup_{\alpha\lt \lambda} L_\alpha \;(\lambda \text{ 极限})
$$

并令

$$
L = \bigcup_{\alpha \in \mathrm{On}} L_\alpha
$$

**可构造宇宙 $L$**。它与 $V$ 的结构逐层平行：$L_\alpha \subseteq V_\alpha$ 且 $L_\alpha$ 是传递集；$L_\omega = V_\omega$（遗传有限集在有限步内被公式定义）；但 $L_{\omega+1} \subsetneq V_{\omega+1}$——$V_{\omega+1}$ 里有不可数多实数，$L_{\omega+1}$ 里只有可数多个（因为公式可数）。<span class="marginnote">关键观察：$L_{\omega+1}$ 是<strong>可数</strong>的，而 $V_{\omega+1}$ 不可数。于是「$V$ 里有不可数多实数的证据」在 $L$ 里并不存在——$L$ 里实数的数量是「从 $L$ 内部看」的，这可能小于 $V$ 里的数量。这就是 $L \neq V$ 的第一道裂缝。</span>

**要点**：$L_\alpha$ 的每一步都「被公式限制」，所以整座 $L$ 是「最小的传递模型类」——任何包含一切序数的传递模型都包含 $L$。这是 $L$ 的核心特征。

## 3 $L$ 满足 ZF：可构造宇宙是合法宇宙

要断言「$L$ 是 ZF 的模型」，需验证每条公理在 $L$ 中成立。难点是分离公理与替换公理——它们对「$L$ 里的任意公式」都要成立。

**关键工具：反射 + 可定义闭包的绝对性**。证明思路是逐条验证：

**外延、空集、配对、并、幂集**：靠 $L_\alpha$ 的传递性（幂集稍复杂：$L$ 里的幂集是 $\mathcal{P}^L(x) = \mathcal{P}(x) \cap L$，即 $L$ 内部能定义的子集）。
**分离**：设 $X, p \in L$，$\varphi$ 是 $L$ 中公式。用**反射原理**（第2篇）找到 $\alpha$ 使 $\varphi$ 在 $L_\alpha$ 中反映，则 $\{x \in X : (L \vDash \varphi(x,p))\} = \{x \in X : (L_\alpha \vDash \varphi(x,p))\}$，后者在 $\mathrm{Def}(L_\alpha)$ 中，故仍在 $L$。
- **替换**：类似，用「像的秩」压回某个 $L_\alpha$。
- **良基公理**：$L \subseteq V$ 是传递的，$V$ 中良基所以 $L$ 中良基。

**定理（Gödel）**：$L \vDash \mathrm{ZF}$。<span class="marginnote">关键是「反射原理」在 $L$ 里也成立——因为 $L$ 是「用公式搭起来的」，公式集合在足够高的 $L_\alpha$ 里被照见。这使分离、替换得以在 $L$ 内部完成，不必借助 $L$ 外的集合。</span>

**辨析｜易错点：** $L \vDash \mathrm{ZF}$ 不意味着「$L$ 的元素构成 ZF 的集合」——$L$ 是真类，不是集合。相对一致性表述为：**ZF 的每个模型都含一个「内部 $L$」满足 ZF**，由此「ZF 一致 ⇒ ZF + V=L 一致」。

## 4 公式解析：$L$ 里为什么 CH 成立

$L$ 里连续统假设成立的关键，是「每个实数都在某个 $L_\alpha$ 里被可数地定义出来」——因此从 $L$ 内部看，实数只有 $\aleph_1$ 多个。核心公式是**构造序数（constructibility rank）**：

$$
\mathrm{otp}(x) = \min\{\alpha : x \in L_{\alpha+1}\}
$$

把「$x$ 最早出现在哪层」记下来。Gödel 的关键引理：

**压缩引理（condensation）**：若 $M$ 是「$L_\alpha$ 的初等子结构」（$\alpha$ 极限），则 $M$ 的传递坍缩（Mostowski collapse）恰是某个 $L_\beta$（$\beta \le \alpha$）。

- **第一步（实数有构造序数）**：$x \subseteq \omega \in L$ 时，存在 $\alpha$ 使 $x \in L_{\alpha+1}$，定义 $\alpha_x$ 为最小者。
- **第二步（压缩引理给出 $\omega_1^L$ 个）**：对每个 $x \subseteq \omega$，取「能看见 $\alpha_x$ 的最小的 $L_\alpha$ 初等子结构」，压缩引理把它压成 $L_\beta$，而压缩保持「$x$ 被定义」——于是 $x$ 在 $L_\beta$ 且 $\beta \lt  \omega_1^L$（因为初等子结构可数时 $\beta$ 可数）。
- **第三步（结论）**：$\{x \subseteq \omega\}^L \subseteq \bigcup_{\beta \lt  \omega_1^L} \mathcal{P}(L_\beta)$ 的势为 $\aleph_1^L$，故 $L \vDash 2^{\aleph_0} = \aleph_1$。

**要点**：压缩引理是 $L$ 的「省料定理」——任何「小初等子结构」都自动坍缩成 $L$ 的一个前段。它保证「能看见实数的最小编码」都是可数的，于是实数全体在 $L$ 内部只有 $\aleph_1$ 个。AC 的证明同理：$L$ 有全局良序（按构造序数排序），选择函数可逐层定义。

**辨析｜易错点：** $L \vDash \mathrm{CH}$ 指的是**在 $L$ 内部**实数有 $\aleph_1^L$ 个。若 $L \subsetneq V$，$V$ 里可能有多得多（不可数多）的实数——「$L$ 里的 $\aleph_1$」与「$V$ 里的 $\aleph_1$」不一定是同一个基数（严格说 $\aleph_1^L \le \aleph_1^V$，可相等也可更小）。

## 6 动手推导：为什么 $L_{\omega+1}$ 只有可数多个实数

把「$L$ 里实数可数」的关键一步算出来，理解 $L$ 的「极简」来自哪里。

- **第一步，公式只有可数多条**：一阶公式由有限符号表生成，符号表可数 ⟹ 公式全体可数（有限字符串的可数并）。参数 $a_1,\dots,a_n \in L_\omega$ 也只有可数多（$L_\omega = V_\omega$ 可数）。
- **第二步，可定义子集可数**：$\mathrm{Def}(L_\omega)$ 由「公式 + 参数」配对生成，可数 × 可数 = 可数。故 $L_{\omega+1} = \mathrm{Def}(L_\omega)$ 可数。
- **第三步，对比 $V_{\omega+1}$**：$V_{\omega+1} = \mathcal{P}(V_\omega)$，而 $V_\omega$ 可数 ⟹ $\mathcal{P}(V_\omega)$ 不可数（康托尔定理）。于是 $L_{\omega+1} \subsetneq V_{\omega+1}$——**$L$ 从第一层就「缺」不可数多个实数**。
- **第四步，要点**：$V$ 的幂集「无差别地收进一切子集」，$L$ 的可定义闭包只收「被公式点名的子集」。这个「点名」限制贯穿 $L$ 的每一层，最终让「$L$ 里实数只有 $\aleph_1$ 个」（由压缩引理严格化）。

**辨析｜易错点：** $L_{\omega+1}$ 可数**不等于**「$L$ 可数」——$L$ 是超穷并，从 $\omega+1$ 层往上每层可能继续增长（虽然仍可能「相对小」）。可数性只对「前 $\omega+1$ 层」成立。初学者常把「$L_{\omega+1}$ 可数」误推广成「$L$ 可数」。

### 更进一步：$L$ 中的 $\Diamond$ 与广义连续统

$L$ 不只给出 CH 与 AC，它还承载一整套「$L$ 组合原则」——最重要的两个是 **$\Diamond$（diamond）** 与 **GCH**：

- **$L \vDash \mathrm{GCH}$**：对每个无穷基数 $\kappa$，$2^\kappa = \kappa^+$。证明仍用压缩引理：$L$ 里每个 $\kappa$ 的子集在某个「看见它的 $L_\alpha$」里被可数定义，压缩后落入 $L_{\kappa^+}$，于是 $|\mathcal{P}(\kappa)^L| \le \kappa^+$。
- **$L \vDash \Diamond$**：$\Diamond$ 序列存在（Jensen 1972），它是「$L$ 组合学」的引擎——上一节的 Suslin 树、以及力迫法里「$L$ 里反例最多」的事实都靠它。

$L$ 的这些「饱和」的组合性质，与它「极小」的宇宙形成奇妙对照：**最小的内模型反而拥有最强的组合原则**。原因在于 $L$ 太「穷」——穷到每个子集都被一层层的可定义切片钉死，于是「处处可预测」。这提醒我们：集合论里「小而穷」往往比「大而富」更可控，这正是力迫法反复利用 $L$ 的原因。

### 补充：$L$ 与 $V$ 的关系一句话

一句话总结 $L$ 与 $V$：**$L \subseteq V$ 总是成立（$L$ 是 $V$ 的传递内模型），但「$L = V$」是独立于 ZFC 的命题**（Gödel 在 $L$ 内部显然 $L=V$；力迫可以造出 $V[G] \supsetneq L$ 的模型，那里 $L \neq V$）。

- 「$V = L$」断言「每个集合都是可构造的」——它推出 GCH、$\Diamond$、一切组合原则。
- 多数集合论者认为 $V = L$「太限制」——它排除了可测基数等大基数（$L$ 里没有可测基数，Scott 定理）。
- $L$ 的地位：它是「内模型程序的起点」——「$L$ 里有什么、缺什么」是判断新内模型（$L[\mu]$、核心模型）的基准。

**辨析｜易错点：** 「$L \subseteq V$」是 ZF 的定理（$L$ 的定义绝对），但「$L = V$」不是——它依赖「每个集合可构造」这一额外断言。初学者常把「$L$ 是内模型」与「$L = V$」混为一谈；前者总是真，后者独立。

## 9 小结

- **可定义子集** $\mathrm{Def}(A)$：用 $A$ 内的一阶公式加参数定义出的子集；通常远小于 $\mathcal{P}(A)$。
- **可构造层级**：$L_0=\emptyset$，$L_{\alpha+1}=\mathrm{Def}(L_\alpha)$，$L_\lambda=\bigcup_{\alpha\lt \lambda}L_\alpha$；$L=\bigcup_\alpha L_\alpha$。
- **$L \vDash \mathrm{ZF}$**：反射原理保证分离/替换在 $L$ 内可完成；$L$ 是真类的内模型。
- **压缩引理**：$L_\alpha$ 的初等子结构坍缩成 $L_\beta$；由此每个实数的编码可数化，$L \vDash \mathrm{CH}$ 且 $L \vDash \mathrm{AC}$。
- **相对一致性**：ZF 一致 ⇒ ZF + V=L（因而 ZF + CH + AC）一致；$L$ 是最小传递模型类。

在下一节，我们解剖 $L$ 的「可定义」机制本身：绝对性、良定义性——为什么有些公式在任何模型里都「说同一件事」，这些公式是 $L$