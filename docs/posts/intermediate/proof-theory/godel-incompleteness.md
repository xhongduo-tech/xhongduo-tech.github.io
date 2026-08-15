---
title: Gödel 不完全性定理的证明论视角
date: 2026-08-07
---

# Gödel 不完全性定理的证明论视角

<div class="epigraph">
<p>存在这样一些算术命题：它们在系统内既不可证，也不可反驳——然而它们是真的。</p>
<footer>—— 库尔特 · 哥德尔（Kurt Gödel），《论〈数学原理〉及相关系统中的形式不可判定命题》（1931）</footer>
</div>

<div class="article-byline">
<p>第二级 · 证明论 ｜ A. S. Troelstra & H. Schwichtenberg《Basic Proof Theory》第10章 ｜ 2026-08-07</p>
</div>

## 为什么从 Gödel 定理开始

前面几篇反复提及「Gödel 第二定理」「不完全性」，现在到了正面拆解的时刻。**Gödel 不完全性定理**是 20 世纪数学最深刻的发现之一，而从证明论的视角看它，叙事最为清晰：它不是一个关于「某些奇怪的命题」的孤立结论，而是**算术化 + 自指 + 可表示性**三件工具的必然结果。<span class="marginnote">哥德尔 1931 年发表论文时年仅 25 岁，论文的题目——「论〈数学原理〉及相关系统中的形式不可判定命题」——几乎预告了后续的一切。希尔伯特听到结果时的反应是「数学里不该有不可知」；几十年后，Gödel 定理成了每一个数学基础叙事的核心章节。</span>

对证明论而言，Gödel 定理不是终点而是转折点：它宣告「在系统内部证明系统一致」之路断绝，逼出了序数分析（第 7、8 篇）与证明挖掘（第 11 篇）这两条新路。理解它，才能理解证明论为什么长成今天的样子。

## 1 算术化

Gödel 定理的第一块基石是**算术化（arithmetization）**：把「公式」「证明」这些符号对象编码成自然数。这就是**哥德尔配数（Gödel numbering）**。

先给每个基本符号配一个数（比如 $0 \mapsto 1$、$S \mapsto 2$、$\forall \mapsto 9$ 等），再把一串符号 $s_1 s_2 \cdots s_k$ 编成它们配数的某种「打包」——例如用哥德尔发明的「指数乘积」编码。这样：

每个公式 $\varphi$ 有一个配数 $\ulcorner \varphi \urcorner$；
每个推导序列有一个配数（一个更大的数）；
「$y$ 是 $x$ 的一个证明」成为一个**关于数字的算术关系** $\mathrm{Proof}(x, y)$。

**关键一跃**：原本关于「符号」的元数学陈述——「公式 $\varphi$ 可证」——变成了关于「数字」的算术陈述「$\exists y\,\mathrm{Proof}(\ulcorner\varphi\urcorner, y)$」。数学第一次能够**谈论自己**。<span class="marginnote">为什么可以这样编码而不产生「两种语言」的混淆？因为编码是可逆且可机械计算的：给定一个数，能机械判定它是不是某个公式/证明的编码。这种「可机械转换」正是算术化被接受的底线，也正是它在 PRA 层面可形式化的原因。</span>

## 2 可表示性与证明谓词

第二块基石是**可表示性（representability）**——第 6 篇已经见过：每个原始递归关系 $R$ 都有公式 $\varphi_R$ 使 PA 能证明其成立与否。证明谓词恰好是原始递归的：

$$
\mathrm{Proof}_{\mathrm{PA}}(x, y) \iff \text{「} y \text{ 是编号为 } x \text{ 的公式在 PA 中的证明」}
$$

由此可以定义**可证性谓词（provability predicate）** $\mathrm{Bew}(x) \equiv \exists y\,\mathrm{Proof}_{\mathrm{PA}}(x, y)$，读作「编号为 $x$ 的公式可证」。它在 PA 内部是一个合法的公式，具备三条教科书性质：

- **可证封闭**：若 $\mathrm{PA} \vdash \varphi$，则 $\mathrm{PA} \vdash \mathrm{Bew}(\ulcorner \varphi \urcorner)$；
- **分配律**：$\mathrm{PA} \vdash \mathrm{Bew}(\ulcorner \varphi \to \psi \urcorner) \to (\mathrm{Bew}(\ulcorner \varphi \urcorner) \to \mathrm{Bew}(\ulcorner \psi \urcorner))$；
- **四则内化**：$\mathrm{PA} \vdash \mathrm{Bew}(\ulcorner \varphi \urcorner) \to \mathrm{Bew}(\ulcorner \mathrm{Bew}(\ulcorner \varphi \urcorner) \urcorner)$。

这三条合称**可证性条件（derivability conditions）**，是后续一切自指论证的引擎。<span class="marginnote">可证性条件让「可证」在算术内部表现得像个模态算子——它满足正规模态逻辑 $\mathbf{K4}$ 的全部公理。希尔伯特–贝奈斯与洛布（Löb）正是在这三条条件上搭起完整的证明逻辑（provability logic），那里甚至能推出「可证性不动点定理」。</span>

## 3 公式解析：对角线引理

自指靠的是**对角线引理（diagonal lemma / 不动点引理）**。它对每个一元公式 $\varphi(x)$ 都造出一个「说自己满足 $\varphi$」的句子：

$$
\mathrm{PA} \vdash \gamma \;\longleftrightarrow\; \varphi(\ulcorner \gamma \urcorner)
$$

分四步拆解：

- **第一步，定义代入函数**：设 $\mathrm{sub}(n, m)$ 是「把编号为 $n$ 的公式中自由变元全部替换为数字 $m$ 的记法所得的公式的编号」。它是原始递归的，因而可在 PA 中表示。
- **第二步，构造自指对象**：取公式 $\psi(x) \equiv \varphi(\mathrm{sub}(x, x))$，即「把 $x$ 代入 $x$ 自身」。令 $m = \ulcorner \psi \urcorner$。
- **第三步，计算代入**：考察 $\gamma \equiv \psi(m)$，即 $\varphi(\mathrm{sub}(m, m))$。而 $\mathrm{sub}(m, m)$ 恰是「把 $\psi$ 的自由变元换成 $m$」——得到的正是 $\gamma$ 自身！
- **第四步，收网**：于是 $\gamma$ 等价于「$\varphi(\ulcorner \gamma \urcorner)$」，即 $\gamma$ 说「我满足 $\varphi$」。**自指从「技巧」变成「可证明的算术事实」**。

对角线引理的价值在于它是通用的：令 $\varphi$ 取「不可证」，得到第一定理；取「一致性」，得到第二定理；取「我的否定可证」，得到 Löb 定理。**同一个引理，喂不同的 $\varphi$，吐出不同的深刻结论**。

## 4 第一不完全性定理

现在令 $\varphi(x) \equiv \neg \mathrm{Bew}(x)$，对角线引理给出一个句子 $\gamma$，使得

$$
\mathrm{PA} \vdash \gamma \;\longleftrightarrow\; \neg \mathrm{Bew}(\ulcorner \gamma \urcorner)
$$

$\gamma$ 说的是：「我不被证明。」论证分两路：

- 若 $\mathrm{PA} \vdash \gamma$，则可证封闭给出 $\mathrm{Bew}(\ulcorner\gamma\urcorner)$，与 $\gamma$ 的内容「$\neg \mathrm{Bew}(\ulcorner\gamma\urcorner)$」冲突——一致性被破坏。
- 若 $\mathrm{PA} \vdash \neg\gamma$，则 $\gamma$ 等价于「$\neg\mathrm{Bew}$」给出 $\mathrm{Bew}(\ulcorner\gamma\urcorner)$ 成立，即「$\gamma$ 可证」为真；但我们已经假定 $\gamma$ 不可证（否则第一条矛盾）。「$\gamma$ 可证」为真而「$\gamma$ 可证」不可证，需要**$\omega$-一致性**来排除。<span class="marginnote">$\omega$-一致性是比普通一致性更强的假设：它要求 PA 不能「同时证出 $A(0), A(1), A(2), \dots$ 又证出 $\exists x\,\neg A(x)$」。Rosser 在 1936 年把「可证」换成「存在一个其否定更早被证的证明」，用普通一致性即可完成论证——这是对 Gödel 原始证明的著名改良。</span>

于是：**$\gamma$ 与 $\neg\gamma$ 都不可证，但 $\gamma$ 在标准模型中为真**——因为 $\gamma$ 的内容（「我不被证明」）恰好如实描述了它的处境。第一不完全性定理成立：

> **定理（Gödel I）**：任何包含 PA、且一致（甚至 $\omega$-一致）的递归公理化理论，都存在既不可证也不可反驳的句子。

## 5 第二不完全性定理

第二定理几乎是从第一定理「白送」出来的。令 $\mathrm{Con}(\mathrm{PA})$ 为「PA 一致」的形式化句子，例如 $\neg \mathrm{Bew}(\ulcorner 0 = 1 \urcorner)$。把第一定理的证明在 PA **内部**重做一遍，会得到：

$$
\mathrm{PA} \vdash \mathrm{Con}(\mathrm{PA}) \;\to\; \gamma
$$

即「PA 一致蕴含 $\gamma$」——因为 $\gamma$ 的不可证性，恰是一致性的一个推论。若 PA 能证明 $\mathrm{Con}(\mathrm{PA})$，则 PA 也能证明 $\gamma$，矛盾于第一定理。于是：

> **定理（Gödel II）**：若 PA 一致，则 $\mathrm{PA} \nvdash \mathrm{Con}(\mathrm{PA})$。

这个证明论视角下的第二定理有着**精确的证明论含义**：任何一致性证明都必须使用 PA 之外的原理。这直接引出第 7、8 篇的整个事业——序数分析就是回答「要用多少超限原理」的学科。<span class="marginnote">第二定理还有一个常常被忽略的方向：$\mathrm{Con}(\mathrm{PA})$ 与 Gödel 句子 $\gamma$ 在 PA 内<strong>等价</strong>。换句话说，「PA 一致」与「存在一个不可证的真句子」在 PA 看来是同一件事的两面。这个等价在证明逻辑里是洛布定理的直接推论。</span>

## 6 证明论视角下的意义

把 Gödel 定理放回证明论的脉络，它留下的不是绝望，而是一张精确的「能力地图」：

- **一致与完全不可兼得**：任何「能表达算术」的一致系统都必然不完备。
- **自证清白不可得**：一致性证明必须走出系统，这正是序数分析存在的理由。
- **可证性可被算术化**：证明谓词在算术内部运转，证明论因此获得「可形式化的元数学」。
- **限制即信息**：Gödel 定理没有杀死数学，而是告诉数学家「你的工具在哪里失效」——而失效的边界，恰恰是序数、不可判定命题、高速增长函数这些新对象的出生地。

**辨析｜易错点：** 第二定理常被误读为「PA 的一致性无法证明」。准确表述是「**PA 不能证明自己的**一致性」——在更强元理论中证明它完全可以，这正是 Gentzen 做的事。把「在系统内不可证」错当成「绝对不可证」，是理解不完全性时最普遍的滑坡。

还有一层当代回响值得点出：Gödel 定理常被称为「数学里第一个真正的复杂性定理」——它首次揭示了「知晓」与「证明」之间不可消除的差距。这个差距今天在计算复杂性、密码学与机器学习理论里以各种面貌反复出现：**证明能力与计算能力共享同一条边界**。

## 7 小结

- **算术化**把公式与证明编码成数字，数学由此能谈论自己。
- **可证性谓词** $\mathrm{Bew}$ 满足三条可证性条件，在算术内部扮演模态算子。
- **对角线引理**为每个 $\varphi(x)$ 造出「说我满足 $\varphi$