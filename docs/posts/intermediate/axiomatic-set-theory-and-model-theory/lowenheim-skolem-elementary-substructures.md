---
title: Löwenheim-Skolem 定理、初等子结构与初等链
date: 2026-08-07
---

# Löwenheim-Skolem 定理、初等子结构与初等链

<div class="epigraph">
<p>一阶逻辑说不出「我是不可数的」——于是每个不可数理论都藏着可数的灵魂。</p>
<footer>—— 利奥波德 · 勒文海姆（Leopold Löwenheim）</footer>
</div>

<div class="article-byline">
<p>第二级 · 公理集合论与模型论 ｜ Marker, <em>Model Theory</em> 第2章 ｜ 2026-08-07</p>
</div>

## 为什么从 Löwenheim-Skolem 开始

紧致性告诉模型论「模型可以有多大」，而 **Löwenheim-Skolem 定理** 补上另一半：「模型可以有多小」。它断言：**只要一个理论有某个无限模型，它就有任意大的模型，也有任意小的模型（小到可数）**。这意味着一阶逻辑表达不了「模型的基数」——一个理论可以同时有可数模型与不可数模型，它们初等等价却大小悬殊。<span class="marginnote">最著名的后果是 <strong>Skolem 悖论</strong>：$\mathsf{ZFC}$ 如果有模型，就有<strong>可数</strong>模型（Löwenheim-Skolem）；但 $\mathsf{ZFC}$ 又证明「$\mathcal{P}(\omega)$ 不可数」。于是这个可数模型内部「认为自己有不可数集合」——「不可数」是一阶语言里无法自我指认的相对概念。</span>

今天先把 Löwenheim-Skolem 的两个方向（向上、向下）讲透，再引入**初等子结构**（模型间「真理保持」的内嵌）与**初等链**（模型塔的并），并指出它们如何协同——初等链的并保持真理，是模型论里「造大模型」的标准手法，也是第3节饱和模型的前奏。

## 1 向下 Löwenheim-Skolem：可数小模型存在

**向下 Löwenheim-Skolem 定理**：设 $\mathcal{L}$ 是可数语言（或 $|\mathcal{L}| \le \kappa$），$\mathcal{M}$ 是 $\mathcal{L}$-结构，$A \subseteq M$。则存在 $\mathcal{N} \preceq \mathcal{M}$（初等子结构，见 §2）使得 $A \subseteq N$ 且 $|N| \le \max(|A|, |\mathcal{L}|, \aleph_0)$。

特别地：**每个无限结构都有可数初等子结构**。

**证明骨架（Skolem 闭包 / Tarski-Vaught 判据）**：

1. 从 $A$ 出发，逐步加入「见证元素」：对每个公式 $\exists x\, \psi(x, \bar a)$（$\bar a$ 是已收集元素），若 $\mathcal{M} \vDash \exists x\,\psi(x,\bar a)$，就选一个见证 $b \in M$ 加入集合。
2. 反复 $\omega$ 步，得到 $N$（可数，因为公式可数、每步加入可数多见证）。
3. **Tarski-Vaught 判据**：$N$ 满足「每个存在公式的见证都在 $N$ 里」，由此归纳得 $\mathcal{N} \preceq \mathcal{M}$。<span class="marginnote">Tarski-Vaught 判据：$\mathcal{N} \subseteq \mathcal{M}$ 是初等子结构，当且仅当对每个公式 $\exists x \psi(x, \bar b)$（$\bar b \in N$），若 $\mathcal{M} \vDash \exists x \psi(x,\bar b)$ 则存在 $c \in N$ 使 $\mathcal{M} \vDash \psi(c, \bar b)$。它把「初等」化约为「存在见证在内部」。</span>

**要点**：证明的核心是「闭包」——把见证元素不断加入，直到模型对存在断言「自足」。这与 Henkin 构造的「见证常数」是同一思想的两面：模型内部必须有所有存在断言的见证。

## 2 初等子结构：真理保持的内嵌

**初等子结构（elementary substructure）** $\mathcal{N} \preceq \mathcal{M}$：$\mathcal{N} \subseteq \mathcal{M}$ 是子结构，且对每个公式 $\varphi(\bar x)$ 与每个 $\bar b \in N$：

$$
\mathcal{N} \vDash \varphi(\bar b) \iff \mathcal{M} \vDash \varphi(\bar b)
$$

「**$\mathcal{N}$ 里看到的真理 = $\mathcal{M}$ 里看到的真理**」——$\mathcal{N}$ 是 $\mathcal{M}$ 里「说得一样」的缩印。<span class="marginnote">注意与「子结构」的区别：子结构只要求「符号解释一致」（常数在、函数封闭、关系一致）；初等子结构还要求「所有一阶公式的真理一致」——强得多。例如 $(\mathbb{N},\lt )$ 是 $(\mathbb{Z},\lt )$ 的子结构但<strong>不是</strong>初等子结构（$\exists x\,(x\lt 0)$ 在 $\mathbb{Z}$ 真、在 $\mathbb{N}$ 假）。</span>

**初等嵌入（elementary embedding）** $j: \mathcal{M} \to \mathcal{N}$：保序且保一切公式真理的映射（$\mathcal{M} \vDash \varphi(\bar a) \iff \mathcal{N} \vDash \varphi(j\bar a)$）。初等嵌入是模型论「比较模型」的基本语言——力迫里的「初等嵌入」概念（大基数理论）正是它的集合论化身。

**辨析｜易错点：** 「$\mathcal{N} \preceq \mathcal{M}$」的真理保持**只对 $\mathcal{N}$ 的元素**做参数——不是「$\mathcal{N}$ 能谈 $\mathcal{M}$ 的一切」。$\mathcal{N}$ 里没有的元素对应的公式 $\exists x \varphi(x)$ 即使 $\mathcal{M}$ 里有见证，$\mathcal{N}$ 里也可能没有——但只要 $\mathcal{M} \vDash \exists x\varphi(x)$ 且参数在 $\mathcal{N}$ 里，就必有见证在 $\mathcal{N}$ 里（这正是 Tarski-Vaught 判据）。

## 3 向上 Löwenheim-Skolem 与初等链

**向上 Löwenheim-Skolem 定理**：若 $\mathcal{M}$ 是无限结构，则对任意基数 $\kappa \ge |\mathcal{M}|$，存在初等扩张 $\mathcal{N} \succ \mathcal{M}$ 使得 $|\mathcal{N}| \ge \kappa$。

**证明（紧致性 + Skolem 化）**：加 $\kappa$ 个新常数 $\{c_\alpha\}_{\alpha\lt \kappa}$，考虑理论

$$
T^* = \mathrm{Th}(\mathcal{M}) \cup \{c_\alpha \neq c_\beta : \alpha \neq \beta\}
$$

每个有限子集有模型（$\mathcal{M}$ 无限，可给有限多个 $c$ 配不同元素），紧致性给出模型 $\mathcal{N}$，它包含 $\kappa$ 个互异元素（$c_\alpha$ 的取值），且 $\mathcal{N} \succ \mathcal{M}$（因为含 $\mathrm{Th}(\mathcal{M})$）。<span class="marginnote">向上定理的证明完美示范了「紧致性 = 加常数的存在性」：要多大，就加多少个「互异常数」。这也再次说明一阶逻辑的「基数盲目性」——它无法表达「我的模型只有 $\aleph_0$ 大」。</span>

**初等链（elementary chain）**：模型塔 $\mathcal{M}_0 \preceq \mathcal{M}_1 \preceq \cdots \preceq \mathcal{M}_n \preceq \cdots$（$\alpha\lt \beta$ 时 $\mathcal{M}_\alpha \preceq \mathcal{M}_\beta$）。

**初等链定理（Tarski-Vaught）**：设 $\{\mathcal{M}_\alpha\}_{\alpha\lt \delta}$ 是初等链，则并 $\bigcup_{\alpha\lt \delta} \mathcal{M}_\alpha$ 是每个 $\mathcal{M}_\alpha$ 的初等扩张。

**要点**：初等链定理是「造大模型」的标准工具：一段段往上爬，极限处取并，真理保持不变。它把「无限扩张」化约为「可数步扩张」，是饱和模型、超积理论的常用脚手架。

## 4 公式解析：Skolem 悖论的谜底

把 Skolem 悖论的推理链写出来，拆开「不可数」的歧义：

$$
\mathsf{ZFC} \text{ 有可数模型 } \mathcal{M} = (M, \in^{\mathcal{M}}), \qquad \mathcal{M} \vDash \exists x\, (x \text{ 是 } \mathcal{P}(\omega) \text{ 的集})
$$

- **向下 LS**：$\mathsf{ZFC}$ 若有模型，就有可数模型 $M$（$|M| = \aleph_0$）。
- **ZFC 内部**：$\mathsf{ZFC} \vdash$「$\mathcal{P}(\omega)$ 不可数」——这是定理，故 $\mathcal{M} \vDash$「$\mathcal{P}(\omega)$ 不可数」。
- **矛盾？**：$M$ 可数，其元素至多 $\aleph_0$ 个——但 $\mathcal{M}$ 里「$\mathcal{P}(\omega)$」是某个 $p \in M$，而「$p$ 不可数」是**$\mathcal{M}$ 内部**的断言（$\mathcal{M} \vDash \neg \exists f\, (f: \omega \leftrightarrow p)$）。
- **谜底**：$\mathcal{M}$ 里「$\omega$」也是 $\mathcal{M}$ 的元素（一个「伪自然数集」）。$\mathcal{M}$ 内部**看不到** $p$ 到 $\omega$ 的双射，但这只是因为**该双射不在 $\mathcal{M}$ 里**——在 $V$ 里，$p$ 与 $\mathcal{M}$ 的「$\omega$」可能都是可数集，存在 $V$ 里的双射，但它不是 $\mathcal{M}$ 的元素。

**要点**：「不可数」是一阶语言里**相对**的断言：$\mathcal{M} \vDash$「$x$ 不可数」意思是「$\mathcal{M}$ 里不存在双射 $f:\omega \to x$」——$f$ 必须**是 $\mathcal{M}$ 的元素**。模型论把「大小」翻译成「模型内部的关系结构」，Skolem 悖论由此消解。

**辨析｜易错点：** 悖论的名字是「悖论」，实则是**非矛盾**——$\mathcal{M}$ 可数而 $\mathcal{M} \vDash$「$x$ 不可数」同时成立，因为「不可数」的判定函数被限制在 $\mathcal{M}$ 内部。初学者常常忘记「存在量词 $\exists f$ 的 $f$ 必须在模型里」，才会觉得矛盾。

## 6 动手推导：$(\mathbb{Z},\lt )$ 不是 $(\mathbb{R},\lt )$ 的初等子结构

把「子结构 ≠ 初等子结构」用具体例子钉死，建立 Tarski-Vaught 判据的直觉。

- **第一步，是子结构吗**：$(\mathbb{Z},\lt ) \subseteq (\mathbb{R},\lt )$——整数集是实数的子集，序关系一致。是子结构。
- **第二步，是初等子结构吗**：考虑公式 $\varphi(x) \equiv \exists y\, (x \lt  y \lt  x+1)$。对 $x = 0$：在 $(\mathbb{R},\lt )$ 里 $\varphi(0)$ 真（取 $y = 1/2$）；在 $(\mathbb{Z},\lt )$ 里 $\varphi(0)$ 假（没有整数严格夹在 $0$ 与 $1$ 之间）。
- **第三步，初等性被破坏**：$(\mathbb{Z},\lt ) \vDash \lnot \varphi(0)$ 而 $(\mathbb{R},\lt ) \vDash \varphi(0)$——同一公式同一参数，真值不同，故不是初等子结构。
- **第四步，Tarski-Vaught 视角**：判据要求「若 $\mathcal{M} \vDash \exists y \psi(y, \bar b)$ 则见证在 $\mathcal{N}$ 里」。这里见证 $y=1/2$ 不在 $\mathbb{Z}$ 里，判据失败——正是「初等」失败的原因。
- **第五步，要点**：子结构只管「符号解释一致」，初等子结构还要求「一切公式真理一致」。整数是实数的子结构，但不是初等子结构——差距恰恰在「存在量词的见证不在子结构里」。

**辨析｜易错点：** 初等子结构要求「**每个**公式的真理保持」，不是「某些公式」。$\mathrm{tp}$ 相同的元素对初等性至关重要，但「元素相似」不等于「结构初等」——$(\mathbb{Z},\lt )$ 里每个整数在 $\mathbb{R}$ 里都有同型对应，可整个结构仍非初等子结构，因为「见证缺席」。

### 更进一步：初等链定理的一个典型用法

初等链定理「$\bigcup_\alpha \mathcal{M}_\alpha$ 是初等扩张」看起来平淡，却是一切「造大模型」的标准脚手架。看两个典型场景：

- **造 $\kappa$-饱和模型**：从任意模型 $\mathcal{M}_0$ 出发，反复「实现一切 $\lt \kappa$ 参数的型」——每步取一个「更大」的模型 $\mathcal{M}_{\alpha+1} \succ \mathcal{M}_\alpha$，极限处取并。初等链定理保证极限仍是初等扩张，于是「型的实现」逐级累积，最终得到 $\kappa$-饱和模型。**饱和 = 初等链的极限**。
- **模型论里「并」的合法性**：任何初等链的并都「说同一句话」——这使「把可数步骤拼接成超穷过程」成为模型论的标准操作。力迫、超积、Morley 定理的证明里，这种「链 + 极限」的模式无处不在。

**要点**：初等链定理是「模型论的超限递归」——它把「无穷多步扩张」压缩成「一个初等扩张」，从而让「逐步实现型」「逐步逼近饱和」这些过程有了严格的落点。它与集合论里的超限递归共享同一个骨架：**极限处取并，性质保持**。

### 补充：Skolem 悖论与「模型相对性」的普适教训

Skolem 悖论常被哲学地讨论，但它的模型论教训非常具体：**一阶性质的相对性**。$\mathcal{M} \vDash \varphi$ 永远是关于「$\mathcal{M}$ 内部结构」的断言，不是关于「对象本身」的断言。

- 这个教训在集合论里最尖锐：$\mathsf{ZFC}$ 的每个模型里「序数」都是该模型「内部世界」的序数。$L$ 与 $V$ 的序数可能不同（$L$ 里 $\aleph_1$ 可以是 $V$ 里的某个可数序数）。
- 模型论把「真理」从「绝对」改成「相对于模型」——这是现代数学语义学的基石，也是力迫法「改模型」操作合法的前提。
- 日常应用：程序验证里「模型检查」检查的是「模型是否满足公式」——同样的公式在不同模型（不同状态空间）里不同真值，正是相对性的日常体现。

**要点**：相对性不是缺陷，是精度。Skolem 悖论之所以「不悖」，正因为我们学会把「真」读作「在某模型里真」——这套读法贯穿本专题第3篇（力迫改模型）与第4篇（模型论）。

## 9 小结

- **向下 Löwenheim-Skolem**：无限结构有任意小的初等子结构（小到可数）；Skolem 闭包 + Tarski-Vaught 判据。
- **初等子结构** $\mathcal{N} \preceq \mathcal{M}$：$\mathcal{N}$ 里看到的真理 = $\mathcal{M}$ 里看到的真理，比「子结构」强得多——$(\mathbb{Z},\lt )$ 是 $(\mathbb{R},\lt )$ 的子结构但不是初等子结构。
- **向上 Löwenheim-Skolem**：无限结构有任意大的初等扩张——要多大，就加多少个「互异常数」，紧致性定理再次出场。
- **初等链定理**：$\bigcup_{\alpha \lt \delta}\mathcal{M}_\alpha$ 是每个 $\mathcal{M}_\alpha$ 的初等扩张——「造大模型」的标准脚手架，饱和模型的构造正由此起步。
- **Skolem 悖论的消解**：「不可数」是一阶语言里的相对断言——判定双射必须躺在模型内部，模型可数与「模型认为 $x$ 不可数」并不矛盾。