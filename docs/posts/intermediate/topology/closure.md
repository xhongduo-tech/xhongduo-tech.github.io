---
title: 闭包（Closure）的定义与性质
date: 2026-08-07
---

# 闭包（Closure）的定义与性质

<div class="epigraph">
<p>无穷！任何其他问题都不曾如此深刻地撼动人类的心灵。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert，1925 年演讲《论无穷》）</footer>
</div>

<div class="article-byline">
<p>第二级 · 拓扑学 ｜ 尤承业《基础拓扑学讲义》第一章 §1.2 ｜ 2026-08-07</p>
</div>

## 为什么从「一个集合贴着什么」开始

前几篇我们一直在问「哪些集合开、哪些集合闭」。可真实世界里的集合大多不这么听话：$(0,1)$ 既不开也不闭，$\mathbb{Q}$ 更是既不疏也不闭。面对一个「随便给」的子集 $A$，拓扑学要回答的第一个问题是：**$A$ 到底贴着多少空间？** 它不只是 $A$ 自己那一片，还包括它边界上「一碰就着」的所有点——把这些点统统收进来，就得到了 $A$ 的**闭包（closure）**。

闭包是点集拓扑里最有用的「粘合剂」之一。它把「集合」升级成「闭集」，让一切关于闭集的手段都能作用于任意集合；它是**稠密性**的定义工具（$\overline{\mathbb{Q}} = \mathbb{R}$）、是**连续映射**的等价判据（$f(\overline{A}) \subseteq \overline{f(A)}$）的原料，也是**分离公理**里点与闭集分离的测试对象。可以说，闭包是继开集、邻域、基之后，理解拓扑空间的第四根支柱。

## 1 闭包的定义：最小的闭集外壳

**闭包（closure）**：设 $A$ 是拓扑空间 $X$ 的子集。称 $X$ 中包含 $A$ 的**一切闭集之交**为 $A$ 的**闭包**，记作 $\overline{A}$：
$$\overline{A} = \bigcap \{ F \subseteq X \mid A \subseteq F, \ F \text{ 是闭集} \}$$

由这个定义，三个事实立刻浮出水面：

- **$A \subseteq \overline{A}$**：每个参与交的闭集都含 $A$，交自然也含 $A$。
- **$\overline{A}$ 是闭集**：任意多个闭集之交仍是闭集（闭集公理的并集版本取补即得）。
- **$A$ 是闭集，当且仅当 $A = \overline{A}$**：若 $A$ 闭，则 $A$ 自己是「含 $A$ 的闭集」之一，且是最小的，故交恰为 $A$；反之若 $A = \overline{A}$，而 $\overline{A}$ 闭，故 $A$ 闭。<span class="marginnote">「闭包」这个名字由此说得通：$\overline{A}$ 是「把 $A$ 用最小的一层闭壳包起来」的结果。$A$ 若是闭的，这层壳就贴得严丝合缝，壳等于 $A$ 本身；$A$ 若不闭，壳就把 $A$ 连同它缺的那几块边界一并裹住。</span>

闭包因此可以一句话总结：**$\overline{A}$ 是包含 $A$ 的最小闭集。** 这层「最小闭壳」的直觉，贯穿后面所有性质。

## 2 邻域刻画：什么是「贴着」

定义虽然干净，却不好直接计算——「一切闭集之交」听起来很抽象。下面这条定理给出了判断「$x$ 在不在闭包里」的贴身标准。

**定理（邻域刻画）**：$x \in \overline{A}$，当且仅当 $x$ 的**每一个开邻域都与 $A$ 相交**：
$$x \in \overline{A} \quad\Longleftrightarrow\quad \forall\, U \text{ 开},\ x \in U \implies U \cap A \neq \emptyset$$

右边那句话的直觉：$x$ 周围无论取多小的开区域，都扫得到 $A$ 的点——$x$ 就「贴着」$A$。这样的 $x$ 叫 $A$ 的**贴点（adherent point）**。闭包 = $A$ 的全部贴点之集。

![集合 A 的闭包](/images/topology/closure-1.svg)

图里的点 $x$ 落在 $A$ 的边界上：它不属于 $A$（$A$ 是青色区域内部），但 $x$ 的每个开邻域都与 $A$ 相交——于是 $x \in \overline{A}$。**边界上的点，正是「贴点」的典型代表。** <span class="marginnote">贴点与「极限点」是有区别的：贴点允许「邻域与 $A$ 的交点就是 $x$ 自己」，而极限点要求「交里必须有 $A$ 的、异于 $x$ 的点」。前者是闭包的语言，后者是下一节《极限点与闭包的等价刻画》的主角。</span>

## 3 闭包的基本性质：一套「粘合」运算律

把闭包看作一种运算 $\overline{(\cdot)}$，它满足一套整洁的规则，合称**闭包运算律**：

**扩大**：$A \subseteq \overline{A}$。
**幂等**：$\overline{\overline{A}} = \overline{A}$。
**单调**：$A \subseteq B \implies \overline{A} \subseteq \overline{B}$。
**并的等式**：$\overline{A \cup B} = \overline{A} \cup \overline{B}$。
**交的包含**：$\overline{A \cap B} \subseteq \overline{A} \cap \overline{B}$。
**端点**：$\overline{\emptyset} = \emptyset$，$\overline{X} = X$。

前三条几乎不用动脑；第四条与第五条是重点，值得停下来对比。**并是等式，交只是包含**——这是闭包运算里最容易翻车的地方。为什么交不能是等式？因为交集的闭包可能被两边的边界「喂」出额外的点，却不一定这些点同时属于两边的闭包。反例就在数轴上：$A = (0,1)$、$B = (1,2)$，则 $\overline{A \cap B} = \overline{\emptyset} = \emptyset$，而 $\overline{A} \cap \overline{B} = [0,1] \cap [1,2] = \{1\}$——点 $1$ 从两边的边界钻了出来。<span class="marginnote">「并保等式、交只保包含」是拓扑学里反复出现的主题：闭包（还有内部、测度、积分）这些「封口」运算都对并友好、对交苛刻。背后的原因是它们都由「包含所有贴点」这种存在性定义，存在性对「或」天然稳定、对「且」天然不稳。</span>

**辨析｜易错点：**不要从「$A \subseteq B \implies \overline{A} \subseteq \overline{B}$」推出「$\overline{A} = \overline{B} \implies A = B$」。闭包运算是「损失信息」的：不同的集合可以有同一个闭包。比如 $\mathbb{Q}$ 与 $\mathbb{R} \setminus \mathbb{Q}$ 的闭包都是 $\mathbb{R}$，可它们互不相交。闭包记住了「贴着什么」，却记不住「具体是谁」。

## 4 公式解析：邻域刻画的两步证明

现在把第 2 节的定理拆成证明链。核心要证的是双向蕴含，两个方向都用**逆否命题**做，格外干净。

**方向一：$x \in \overline{A} \implies$ 每个开邻域都碰到 $A$。**

- **第一步，假设反面**：假设存在含 $x$ 的开集 $U$ 使 $U \cap A = \emptyset$。
- **第二步，翻成闭集**：$U \cap A = \emptyset$ 等价于 $A \subseteq X \setminus U$；而 $U$ 开，故 $X \setminus U$ 闭。
- **第三步，挤进交集**：于是 $X \setminus U$ 是「含 $A$ 的闭集」之一，它必须被 $\overline{A}$ 这个「一切含 $A$ 的闭集之交」包含：
$$A \subseteq X \setminus U \quad\Longrightarrow\quad \overline{A} \subseteq X \setminus U$$
- **第四步，推出矛盾**：$x \in U$ 而 $U \cap (X \setminus U) = \emptyset$，故 $x \notin X \setminus U$；又 $\overline{A} \subseteq X \setminus U$，所以 $x \notin \overline{A}$，与假设矛盾。故每个开邻域必碰到 $A$。

**方向二：每个开邻域都碰到 $A$ $\implies$ $x \in \overline{A}$。**

- **第五步，假设反面**：假设 $x \notin \overline{A}$。
- **第六步，挖出隔离开集**：$\overline{A}$ 是闭集，故 $U = X \setminus \overline{A}$ 是开集；又 $x \notin \overline{A}$，故 $x \in U$。
- **第七步，检验隔离**：$U \cap A = (X \setminus \overline{A}) \cap A = \emptyset$，因为 $A \subseteq \overline{A}$。于是找到了一个含 $x$ 的开集，它与 $A$ 不相交——与「每个开邻域都碰到 $A$」矛盾。故 $x \in \overline{A}$。

两条证明殊途同归：**闭包之外的点，总能被一个开集「隔离」在 $A$ 之外。** 这正是闭包作为「贴点集合」的拓扑本质——贴得住，就进闭包；隔得开，就不进。

把同样的手法用到第 3 节的「并的等式」，还能顺手再走一遍：

- **$\supseteq$**：$A \subseteq A \cup B \implies \overline{A} \subseteq \overline{A \cup B}$，同理 $\overline{B} \subseteq \overline{A \cup B}$，故 $\overline{A} \cup \overline{B} \subseteq \overline{A \cup B}$。
- **$\subseteq$**：$\overline{A} \cup \overline{B}$ 是**有限个闭集之并**，因而是闭集；它含 $A \cup B$，故是「含 $A \cup B$ 的闭集」之一，于是最小闭壳 $\overline{A \cup B}$ 被它包含。

两条包含合起来就是等式。注意第二步里「有限个闭集之并闭」是关键——若换成无限并，闭包就未必是闭的了，这也是闭集公理坚持「有限并」的原因之一。

## 5 例子：闭包在几个空间里长什么样

**实数轴通常拓扑：**

- $\overline{(0,1)} = [0,1]$：开区间的闭包补上了两个端点。
- $\overline{\mathbb{Q}} = \mathbb{R}$：有理数的闭包是整个实数轴——因为每个无理数附近都有有理数，任何开邻域都碰到 $\mathbb{Q}$。<span class="marginnote">「$\overline{\mathbb{Q}} = \mathbb{R}$」正是「$\mathbb{Q}$ 在 $\mathbb{R}$ 中稠密」的精确说法。稠密性是实分析的基本功：实数轴上的连续函数由它在 $\mathbb{Q}$ 上的取值完全决定，因为 $\mathbb{Q}$ 贴住了整个 $\mathbb{R}$。</span>
$\overline{\mathbb{Z}} = \mathbb{Z}$：整数是闭集，闭包就是自己。
$\overline{\{1/n \mid n \ge 1\}} = \{1/n\} \cup \{0\}$：点列 $\{1/n\}$ 的极限点 $0$ 被闭包收编——闭包「看到」了序列的极限。

**几个极端拓扑：**

**离散拓扑**：每个集合都闭，故 $\overline{A} = A$ 对一切 $A$——闭包运算形同虚设。
**平凡拓扑**：只有 $\emptyset$ 与 $X$ 两个闭集。$A = \emptyset$ 时 $\overline{A} = \emptyset$；$A \neq \emptyset$ 时唯一含 $A$ 的闭集是 $X$，故 $\overline{A} = X$。任何非空集合都「贴」满整个空间。
**余有限拓扑**：闭集是有限集与 $X$。$A$ 有限时 $\overline{A} = A$；$A$ 无限时唯一含它的闭集是 $X$，故 $\overline{A} = X$。<span class="marginnote">余有限拓扑里「闭包」被有限性彻底支配：集合一旦无限，闭包立刻膨胀到整个空间。这与余有限拓扑「开集靠有限补集撑场」的性格一脉相承——闭合也在听「有限」指挥。</span>

## 6 闭包与连续性：一条通往后续的桥梁

闭包不只是一个孤立概念，它是**连续映射**的等价判据之一，预告着第一篇后半程的重头戏：

**定理（预告）**：$f: X \to Y$ 连续，当且仅当对 $X$ 的每个子集 $A$，都有
$$f\big(\overline{A}\big) \subseteq \overline{f(A)}$$

直觉：连续映射不能「撕裂」空间——若 $x$ 贴着 $A$，那么 $f(x)$ 也必须贴着 $f(A)$。这条不等式把「连续」从「开集原像是开集」的全局语言，翻译成了「贴点被贴点」的局部语言。日后讲连续映射的等价刻画时，它会与「开集原像」「基」「逐点连续」并列，构成理解连续性的四副面孔。<span class="marginnote">这条判据对「不连续映射」格外灵敏：只要某处把「贴着的点」映射到「不贴的地方」，不等式立刻被戳穿。它是检验映射连续性的第一个「局部放大镜」。</span>

## 7 小结

- **闭包**：$\overline{A}$ = 包含 $A$ 的一切闭集之交 = 包含 $A$ 的最小闭集。
- **邻域刻画**：$x \in \overline{A} \iff$ $x$ 的每个开邻域都与 $A$ 相交（$x$ 是 $A$ 的贴点）。
- **运算律**：扩大、幂等、单调；**并保等式** $\overline{A \cup B} = \overline{A} \cup \overline{B}$，**交只保包含** $\overline{A \cap B} \subseteq \overline{A} \cap \overline{B}$。
- **辨析**：闭包运算是「损失信息」的，$\overline{A} = \overline{B}$ 推不出 $A = B$（$\mathbb{Q}$ 与无理数集同闭包）。
- **例子**：$\overline{(0,1)} = [0,1]$、$\overline{\mathbb{Q}} = \mathbb{R}$；离散拓扑闭包恒等，平凡拓扑非空集闭包为全空间。
- **桥梁**：$f$ 连续 $\iff f(\overline{A}) \subseteq \overline{f(A)}$；$\overline{A} = X$ 即「$A$ 稠密」。

在下一节，我们把闭包的「镜子」摆出来：**内部（interior）** 问的是「$A$ 肚子里含着多少开集」，**边界（boundary）** 问的是「$A$ 里里外外的分界线在哪里」。闭包、内部、边界三者两两互补，合起来构成对任意集合的一份完整「拓扑体检报告」。
