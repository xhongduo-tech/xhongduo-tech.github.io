---
title: 次微分（次梯度、Moreau-Rockafellar 理论）
date: 2026-08-07
---

# 次微分（次梯度、Moreau-Rockafellar 理论）

<div class="epigraph">
<p>受过教育的心智的标志，是安于主题所允许的精确度，而不是在只有近似为真的地方强求精确。</p>
<footer>—— 亚里士多德（Aristotle）</footer>
</div>

<div class="article-byline">
<p>第二级 · 凸分析 ｜ Rockafellar《Convex Analysis》第23章；Boyd《Convex Optimization》§3.2 ｜ 2026-08-07</p>
</div>

## 为什么从次微分开始

微积分的导数在尖点处失灵，而现代优化的目标函数到处是尖点：$\ell_1$ 正则的零点、ReLU 的折点、合页损失的边界、SVM 的间隔处。**次微分（subdifferential）**把「唯一的导数」升级成「一簇次梯度」，让不可微的凸函数在每个点都有意义地「可导」。这一篇是第3篇次梯度理论的综述与升华，重点落在**Moreau–Rockafellar 理论**——它回答「次微分如何算加法、复合、逐点极大」，是工程上求次梯度的运算律全集。<span class="marginnote">一句话记忆次梯度：<strong>$g$ 是 $f$ 在 $x$ 处的次梯度，当且仅当直线 $z = f(x) + \langle g, y - x\rangle$ 是上图在 $(x, f(x))$ 处的支撑超平面</strong>——支撑斜率的一簇，就是次梯度集合。可微点这一簇缩成一个点 $\{\nabla f(x)\}$，退化不丢。</span>

在「从极限到大模型」的主线上，次微分把微积分的「导数」从光滑世界推进到非光滑世界：第2篇的凸函数允许尖点，本节的次梯度让尖点也有「斜率」可言。深度学习里 ReLU 在 $0$ 处「取导数 0 或 1」的约定，就是从这个集合里挑元素。

## 1 次梯度：不可微点的「可导」答案

**次梯度（subgradient）**：设 $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ 是正常凸函数，$x \in \operatorname{dom} f$。向量 $g$ 是 $f$ 在 $x$ 处的次梯度，若

$$f(y) \ge f(x) + \langle g, y - x \rangle, \qquad \forall\, y$$

全体次梯度构成**次微分（subdifferential）** $\partial f(x)$。几何上，$g$ 对应上图在 $(x, f(x))$ 处的一条支撑超平面。<span class="marginnote">注意次梯度是用<strong>全局不等式</strong>定义的：一条支撑线必须在整个定义域上「托住」$f$，而不只是局部切线。这带来一个副作用——边界点处可能没有任何次梯度（支撑超平面只存在于上图闭包的支撑点）。好在 $\operatorname{ri}(\operatorname{dom} f)$ 内次微分一定非空，这由第12节连续性定理保证。</span>

**存在性定理**：若 $x \in \operatorname{ri}(\operatorname{dom} f)$，则 $\partial f(x) \ne \emptyset$；若 $x \in \operatorname{int}(\operatorname{dom} f)$，则 $\partial f(x)$ 非空、紧、凸。<span class="marginnote">「非空、紧、凸」三条性质是次微分最优性的物理基础：极小化条件 $0 \in \partial f(x)$ 的「$0$」必须真的落在集合里，紧凸性保证它在算法里可以被逼近（次梯度法、近端法）。</span>三个核心例子：$f(x) = |x|$ 在 $0$ 处 $\partial f(0) = [-1,1]$；$f(x) = \|x\|_1$ 在零分量处取 $[-1,1]$、非零分量处取 $\{\operatorname{sign}(x_i)\}$；$f(x) = \max(0, 1 - x)$（合页）在 $x=1$ 处 $\partial = [-1, 0]$。

把常见函数的次微分列成速查表，是工程求导的第一站：

| 函数 $f$ | 次微分 $\partial f(x)$ |
| --- | --- |
| $|x|$ | $\operatorname{sign}(x)$（$x\ne0$）；$[-1,1]$（$x=0$） |
| $\|x\|_1$ | 逐分量 $\operatorname{sign}(x_i)$；零分量取 $[-1,1]$ |
| $\|x\|_2$ | $\{x/\|x\|_2\}$（$x\ne0$）；$\{g : \|g\|_2\le 1\}$（$x=0$） |
| $\delta_C(x)$ | 法锥 $\mathcal{N}_C(x)$ |
| $\max(0,1-x)$ | $-1$（$x<1$）；$[-1,0]$（$x=1$）；$0$（$x>1$） |

这张表覆盖了 LASSO、SVM、投影算法里九成的次微分查询。

## 2 次微分与最优性：0 ∈ ∂f(x) 的充要条件

次微分把「最优性」压缩成一条干净的代数条件：

**极小化充要条件**：$x^*$ 极小化正常凸函数 $f$ 当且仅当

$$0 \in \partial f(x^*)$$

证明是定义的三行重写：$0 \in \partial f(x^*)$ ⟺ $f(y) \ge f(x^*) + \langle 0, y - x^*\rangle = f(x^*)$ 对一切 $y$ ⟺ $x^*$ 是全局极小点。<span class="marginnote">对比光滑情形：$\nabla f(x^*) = 0$ 只是驻点，还要二阶条件定性质；而<strong>$0 \in \partial f(x^*)$ 直接就是全局最优的充要条件</strong>——凸性的全部「好消息」都浓缩在这条无条件的判定里。非凸情形用 Clarke 次梯度推广，但「含 0」只给出驻点不再保证全局。</span>

**重点：** 这条条件与约束并存时变形为「对偶问题里 $0 \in \partial_x L(x^*, \lambda^*) + \mathcal{N}(x^*)$」——KKT 的次微分形态。无约束时它是最简形态；带约束时加上法锥（正常锥）项。这就是第5篇 KKT 的凸分析形式：**用次微分把「约束是等式还是不等式」统一成「属于哪个锥」**。

**辨析｜易错点：** 「$0 \in \partial f(x)$」与「$\partial f(x) = \{0\}$」完全不同——前者只需含 0（绝对值在 $0$ 处即最优），后者要求唯一。次梯度不唯一是常态而非例外；「次梯度 = 0」与「含 0」的差别，是初学最易滑倒处。

**辨析｜易错点：** 光滑情形 $-\nabla f(x)$ 保证函数下降；次梯度则不然——它只是支撑斜率，不是下降方向。最直观的反例：$f(x) = |x|$ 在全局最小点 $x^* = 0$ 处取次梯度 $g = 1 \in [-1,1]$，沿 $-g$ 方向走到 $x = -t$（$t > 0$），函数值从 0 升到 $t$——「负次梯度」反而让函数上升。所以次梯度法不能要求每步都下降，它的收敛证明靠的是「期望下降」与距离减小的折中，而非单调下降。这正是次梯度法收敛慢（$\mathcal{O}(1/\sqrt{k})$）而梯度法快（$\mathcal{O}(1/k)$）的深层原因。

把「梯度 vs 次梯度」这对孪生概念并排对照，五条差异一次看清：

| 维度 | 梯度 $\nabla f$ | 次微分 $\partial f$ |
| --- | --- | --- |
| 定义 | 唯一的切线向量 | 支撑斜率构成的一簇 |
| 可微点 | 单点 $\{\nabla f(x)\}$ | 退化成单点 $\{\nabla f(x)\}$ |
| 下降性 | $-\nabla f$ 保证下降 | $-g$ 不保证下降 |
| 最优性 | 驻点，还需二阶条件 | $0 \in \partial f(x)$ 即全局最优（充要） |
| 收敛速率 | $\mathcal{O}(1/k)$（光滑强凸） | $\mathcal{O}(1/\sqrt{k})$ |

这张表是「为什么需要次微分」与「为什么次微分法慢」的浓缩答案——前者看第二行，后者看第五行。

## 3 Moreau–Rockafellar 理论：次微分的运算规则

次微分理论的核心成就是**Moreau–Rockafellar 和规则**：设 $f_1, \dots, f_m$ 为正常凸函数，若

$$\bigcap_{i=1}^m \operatorname{ri}(\operatorname{dom} f_i) \ne \emptyset$$

则对一切 $x$，

$$\partial (f_1 + \dots + f_m)(x) = \partial f_1(x) + \dots + \partial f_m(x)$$

（右侧是 Minkowski 和）。**包含 $\supseteq$ 无条件成立**（次梯度可加）；反向需要相对内部相交的条件——它防止两个函数的「尖点」错位导致和函数在某处凸性退化。<span class="marginnote">Moreau–Rockafellar 的名字里藏着两个贡献者：Jean-Jacques Moreau（法国力学与分析学家）与 R. Tyrrell Rockafellar。<strong>它是对偶理论的一块基石</strong>——第4篇 Fenchel 对偶的强对偶、第5篇 KKT 的约束规格，本质上都要调用这条和规则或其变体。</span>

**算一个用和规则的次微分**：$f(x) = |x_1| + |x_2| + x_1^2$。拆成 $f_1 = \|x\|_1$、$f_2 = x_1^2$，两者的定义域都是全空间，相对内部相交条件自动满足。于是 $\partial f(x) = \partial f_1(x) + \partial f_2(x)$，逐分量写出：

$$\partial f(x) = \big( \partial |x_1| + 2x_1 \big) \times \partial |x_2|$$

在 $x = (0, 1)$ 处，$\partial f = ([-1,1] + 0) \times \{1\} = [-1,1] \times \{1\}$——「第一个分量卡在区间里，第二个分量光滑」。**和规则把复合函数的次微分拆成已知块的拼接**，这就是工程求导的标准动作。

**算一个逐点极大次微分**：$f(x) = \max(x^2, |x|)$。在普通点只有一个分量「掌权」：$x = 2$ 时 $f = x^2$，$\partial f(2) = \{4\}$；$x = \tfrac12$ 时 $f = |x|$，$\partial f(\tfrac12) = \{1\}$。在接缝处多个分量同时取到极大，次微分取它们的凸组合：$x = 0$ 处 $x^2 = |x| = 0$，$\partial f(0) = \operatorname{conv}(\{0\} \cup [-1,1]) = [-1,1]$；$x = 1$ 处 $\partial f(1) = \operatorname{conv}(\{2\} \cup \{1\}) = [1,2]$。**逐点极大规则的实操要点：先找「谁掌权」，再在接缝处做凸组合**——这背后正是第2篇「逐点极大保凸」在微分层的回响。

**配套运算规则**：**复合规则**（仿射复合 $f \circ A$ 的次微分 = $A^T \partial f(Ax)$ 在条件成立时）、**逐点极大**（$\partial \max_i f_i(x) = \operatorname{conv}\{\partial f_i(x) : i \in \mathcal{I}(x)\}$，$\mathcal{I}(x)$ 是取到极致的指标集）、**共轭对偶**（$g \in \partial f(x) \iff x \in \partial f^*(g)$）。最后一条是**Fenchel 双共轭的次微分版本**：次梯度与共轭互相转置，是「对偶 = 共轭」在微分层面的回响。<span class="marginnote">逐点极大规则有直觉直译：$f = \max(f_1, f_2)$ 在某点由哪个分量「掌权」，次梯度就取那个分量的次梯度；多个分量同时取到极大时，取它们的凸组合——这就是「活动集」（active set）的次微分表达。</span>

**重点：** 这组规则让「求次微分」变成可机械执行的运算：把函数拆成已知块（绝对值、范数、指示函数、合页、逐点极大），逐层套用和、复合、极大规则。工程上求次梯度（LASSO、SVM、ReLU 网络）走的正是这条路。

## 4 公式解析：和规则为什么需要「相对内部相交」

和规则的难点在反向包含 $\subseteq$。看最简单情形 $f = f_1 + f_2$ 与 $g \in \partial f(x)$ 但想证明 $g = g_1 + g_2$：

$$f_1(y) - f_1(x) + f_2(y) - f_2(x) \ge \langle g, y - x \rangle, \qquad \forall y$$

- **第一步，写成辅助函数**：定义 $h(y) = f_1(y) + f_2(y) - \langle g, y\rangle$。条件 $g \in \partial f(x)$ 正是「$h$ 在 $y = x$ 处取最小值」。
- **第二步，用共轭转移**：$h$ 的最小值条件等价于 $0 \in \partial h(x)$。问题是 $\partial h = \partial f_1 + \partial f_2 + \{-g\}$ 吗？——这正是要证的内容，直接循环了。换路线：$h$ 的下确界为 $h(x)$，而 $\inf_y h(y) = \inf_y [f_1(y) + f_2(y) - \langle g,y\rangle] = -(f_1^* + f_2^*)$（Fenchel 共轭定义）。
- **第三步，上确界转下确界**：由 Fenchel–Moreau 的次微分形式 $f^{**} = f$，可以把「$f_1 + f_2$ 的下确界」展开成「$f_1^*, f_2^*$ 的共轭的共轭」。条件 $\operatorname{ri}(\operatorname{dom} f_1) \cap \operatorname{ri}(\operatorname{dom} f_2) \ne \emptyset$ 保证对偶的驻点存在，进而把「和为 $\inf$」拆成「分别取 $\inf$」。
- **第四步，读出反向包含**：拆解后得到存在 $g_1 \in \partial f_1(x)$、$g_2 \in \partial f_2(x)$ 使 $g = g_1 + g_2$——反向包含成立。条件正是为了让两个函数在「相对内部」相遇，避免边界处的退化。

**这条证明的启示**：<span class="marginnote">Moreau–Rockafellar 的证明必须经过共轭（Fenchel–Moreau）这一站，这正是它被称为「对偶性在次微分上的化身」的原因。<strong>想给次微分建立运算规则，最终要借助对偶理论本身</strong>——运算律与对偶性互为表里。</span>

**从这里走向近端算子**：把和规则与「次梯度含 0」联用，得到投影不等式；更进一步，$\operatorname{prox}_f(x) = \arg\min_z f(z) + \tfrac12\|z - x\|^2$ 满足「$x - \operatorname{prox}_f(x) \in \partial f(\operatorname{prox}_f(x))$」——近端算子正是次微分的「可逆运算」，是近端梯度法、ADMM 在每次迭代里求的闭式解。<span class="marginnote">近端算子可以看作「次微分的积分」：由 $u = x - \operatorname{prox}_f(x)$ 是 $f$ 在 $\operatorname{prox}_f(x)$ 处的次梯度，近端映射把「求 $f$ 的极小」转换成「解 $z + \partial f(z) \ni x$」——一个含次微分的方程。LASSO 的软阈值、投影到凸集，全是它的特例。</span>

### 术语速查：次微分世界的名词对照

| 术语 | 一句话定义 | 出处 |
| --- | --- | --- |
| 次梯度 | 满足全局支撑不等式 $f(y) \ge f(x) + \langle g, y-x\rangle$ 的 $g$ | 本篇 |
| 次微分 | 全体次梯度的集合 $\partial f(x)$，内部非空、紧、凸 | 本篇 |
| 法锥 | 指示函数 $\delta_C$ 的次微分 $\mathcal{N}_C(x)$ | 本篇 / 第5篇 |
| Moreau–Rockafellar 和规则 | 相对内部相交时 $\partial \sum f_i = \sum \partial f_i$ | 本篇 |
| 逐点极大规则 | $\partial \max_i f_i = \operatorname{conv}\{\partial f_i : i \in \mathcal{I}\}$ | 本篇 |
| 共轭对偶 | $g \in \partial f(x) \iff x \in \partial f^*(g)$ | 本篇 / 第4篇 |
| 近端算子 | $\operatorname{prox}_f(x) = \arg\min_z f(z) + \tfrac12\|z-x\|^2$ | 本篇 |
| 软阈值算子 | $\operatorname{prox}_{\lambda\|\cdot\|_1}$，$\ell_1$ 稀疏的执行者 | 本篇 |

## 5 小结

- **次梯度** $g$：$f(y) \ge f(x) + \langle g, y-x\rangle$ 对所有 $y$；$\partial f(x)$ = 上图支撑斜率集合，内部非空、紧、凸。
- **最优性**：$x^*$ 全局极小 ⟺ $0 \in \partial f(x^*)$——无条件、充要。
- **Moreau–Rockafellar 和规则**：$\bigcap \operatorname{ri}(\operatorname{dom} f_i) \ne \emptyset$ 时 $\partial \sum f_i = \sum \partial f_i$。
- **配套规则**：仿射复合（$A^T$）、逐点极大（活动集凸组合）、共轭对偶（$g \in \partial f(x) \iff x \in \partial f^*(g)$）。
- 次微分把「求导」变成「机械运算」，是 LASSO、SVM、近端算法的共同分析基础。
- 近端算子满足 $x - \operatorname{prox}_f(x) \in \partial f(\operatorname{prox}_f(x))$，是次微分的可逆运算。

在下一节，我们把最优性从「无约束」推进到「带约束」——**KKT 条件**的凸分析形式：法锥、拉格朗日乘子与互补松弛如何在同一张图上统一。
