---
title: 最优性条件（KKT 条件的凸分析形式）
date: 2026-08-07
---

# 最优性条件（KKT 条件的凸分析形式）

<div class="epigraph">
<p>把问题表述好，问题就解决了一半。</p>
<footer>—— 约翰 · 杜威（John Dewey）</footer>
</div>

<div class="article-byline">
<p>第二级 · 凸分析 ｜ Rockafellar《Convex Analysis》第28章；Boyd《Convex Optimization》§5.5 ｜ 2026-08-07</p>
</div>

## 为什么从 KKT 条件开始

无约束极小化的最优性只是「$0 \in \partial f(x)$」，带约束时多了一重张力：目标想往下降，约束却把点钉在可行域里。**KKT 条件（Karush–Kuhn–Tucker）**把「目标梯度」与「约束梯度」的对抗写成一组可解的方程——它是带约束最优化的终点判断，也是第4篇对偶理论落地成算法的接口。<span class="marginnote">KKT 名字的三位来源：William Karush（1939 年硕士论文）、Harold W. Kuhn 与 Albert W. Tucker（1951 年发表）。条件本身被 Lagrange（1788 年左右）的乘子法先行实践，KKT 的价值在于把它推广到不等式约束并给出严格的充分性。</span>这一篇是第5篇 KKT 的综述，重点在「凸分析形式」：用法锥与次微分把等式、不等式约束统一成一个几何对象。

KKT 是「从极限到大模型」主线里工程价值最高的一页：内点法、SQP、增广拉格朗日，全部围绕「解 KKT 方程组」转。能写出一组问题的 KKT 条件，就等于把它交给了数值优化器。

## 1 从无约束到带约束：最优性的三股合力

考虑凸优化问题

$$\min_x f_0(x) \quad \text{s.t.} \quad f_i(x) \le 0,\ i = 1,\dots,m, \qquad Ax = b$$

最优性由三股力合成：**目标**想沿 $- \nabla f_0(x)$ 下降；**不等式约束**在边界处把可行方向挡回（只有 $\nabla f_i$ 的「向内」方向可用）；**等式约束**把运动限制在 $Ax = b$ 的仿射流形上。<span class="marginnote">把 $x^*$ 处的「可行下降方向」画出来：不等式约束 $i$ 若取等（活动约束），可行方向必须满足 $\nabla f_i(x^*)^T d \le 0$；等式约束则要求 $A d = 0$。KKT 说：<strong>没有方向同时「让目标下降」且「对所有约束可行」时，就是最优</strong>——最优性 = 可行锥与下降锥不交，再用分离定理转成乘子语言。</span>

**重点：** 最优处必须有「目标梯度」落在「约束梯度的锥组合」里——否则存在一个方向既下降又可行。这条「梯度属于锥」的直觉，是 KKT 一切形式的几何内核。

## 2 KKT 条件：四族条件的结构

对可微凸问题，KKT 条件（在约束规格下是充分必要条件）由四族组成，设 $x^*, \lambda^* \ge 0, \nu^*$：

**（1）原始可行性**：$f_i(x^*) \le 0$（$i = 1..m$）、$Ax^* = b$——$x^*$ 必须在可行域里。

**（2）对偶可行性**：$\lambda^* \ge 0$——不等式乘子非负（等式的 $\nu^*$ 无符号约束）。

**（3）互补松弛（complementary slackness）**：

$$\lambda_i^* f_i(x^*) = 0, \qquad i = 1,\dots,m$$

**（4）驻点条件（stationarity）**：

$$\nabla f_0(x^*) + \sum_{i=1}^m \lambda_i^* \nabla f_i(x^*) + A^T \nu^* = 0$$

互补松弛是「要么约束取等，要么乘子为零」——活动约束才有影子价格，非活动约束的乘子必须归零。<span class="marginnote">互补松弛的名字很形象：$\lambda_i$ 与 $f_i(x^*)$（≤0）二者「互补」——乘积为零。它把「哪些约束被激活」编码成代数条件，而「活动集」（active set）正是内点法与活性集算法迭代时反复猜测的对象。</span>KKT 的每一条都可被第5篇逐条验证；凸性保证「KKT 点即全局最优」。

把四族条件及其角色汇总：

| 条件 | 表达式 | 作用 |
| --- | --- | --- |
| 原始可行 | $f_i(x^*) \le 0,\ Ax^* = b$ | 点必须在可行域 |
| 对偶可行 | $\lambda^* \ge 0$ | 乘子符号正确 |
| 互补松弛 | $\lambda_i^* f_i(x^*) = 0$ | 活动集编码 |
| 驻点 | $\nabla f_0 + \sum \lambda_i \nabla f_i + A^T\nu = 0$ | 梯度平衡 |

**算一个完整 KKT**：求解 $\min \tfrac12(x_1^2 + x_2^2)$ s.t. $x_1 + x_2 \ge 1$。把约束写成 $-x_1 - x_2 + 1 \le 0$，Lagrange $L = \tfrac12(x_1^2+x_2^2) + \lambda(-x_1-x_2+1)$。驻点：$\partial L/\partial x_1 = x_1 - \lambda = 0$、$\partial L/\partial x_2 = x_2 - \lambda = 0$，故 $x_1 = x_2 = \lambda$。互补松弛：$\lambda(-x_1-x_2+1) = 0$。若 $\lambda = 0$ 则 $x = 0$，违反约束；故 $\lambda > 0$，约束取等：$2\lambda = 1$，$\lambda = \tfrac12$，$x_1 = x_2 = \tfrac12$。**一个只有两条约束的小问题，四族条件逐个动用，得出精确解 $(\tfrac12, \tfrac12)$**。

**算一个带等式约束的完整 KKT**：$\min \tfrac12 x_1^2 + \tfrac12 x_2^2 + x_1$ s.t. $x_1 - x_2 = 0$，$x_1 \ge 0$。Lagrange $\tfrac12 x_1^2 + \tfrac12 x_2^2 + x_1 + \nu(x_1 - x_2) - \lambda x_1$（注意 $x_1 \ge 0$ 写为 $-x_1 \le 0$，故乘子前是 $-\lambda$）。驻点：$x_1 + 1 + \nu - \lambda = 0$、$x_2 - \nu = 0$。互补松弛：$\lambda x_1 = 0$。原始可行：$x_1 - x_2 = 0$、$x_1 \ge 0$。分两种情形：① $\lambda = 0$：$x_1 + 1 + \nu = 0$、$x_2 = \nu$、$x_1 = x_2$，得 $x_1 + 1 + x_1 = 0$，$x_1 = -\tfrac12$，违反 $x_1 \ge 0$。② $\lambda > 0$：则 $x_1 = 0$（互补松弛），$x_2 = 0$（等式），驻点得 $0 + 1 + \nu - \lambda = 0 \Rightarrow \lambda = 1 + \nu$，且 $0 - \nu = 0 \Rightarrow \nu = 0$，于是 $\lambda = 1 \ge 0$。全体条件满足，$x^* = (0,0)$ 是解。**这里互补松弛的「分支」与无约束情况形成鲜明对比**：无约束的话 $\nabla f = 0$ 直接给出 $(-\tfrac12, -\tfrac12)$，但约束把解「推」到了边界上。

## 3 凸分析形式：法锥与次微分

KKT 的凸分析形式把四族条件压缩成两条集合包含。定义**法锥（normal cone）**：对凸集 $C$，在 $x \in C$ 处的法锥是

$$\mathcal{N}_C(x) = \{ v \mid \langle v, y - x \rangle \le 0,\ \forall\, y \in C \}$$

法锥收集「与外向方向内积非正」的所有向量——几何上就是支撑超平面法向的集合。<span class="marginnote">法锥是次微分的「集合版本」：指示函数 $\delta_C$ 的次微分 $\partial \delta_C(x) = \mathcal{N}_C(x)$。于是「约束」与「目标」可以在同一个次微分语言里加和——这正是 Moreau–Rockafellar 和规则（第5篇）能统一处理带约束情形的通道。</span>

**KKT 的凸分析形式**：设 $x^*$ 可行，则 $x^*$ 最优当且仅当存在乘子使

$$0 \in \partial f_0(x^*) + \sum_{i: f_i(x^*) = 0} \lambda_i \partial f_i(x^*) + \mathcal{N}_{\{Ax = b\}}(x^*)$$

等式约束用仿射集的法锥（即 $\{A^T \nu\}$ 的全体），不等式用活动集的次微分——**「约束种类」被消化成「锥/次微分」**，不再区分等式与不等式。<span class="marginnote">把目标、不等式、等式三块放进展开式里：$0 \in \partial_x L(x^*, \lambda^*, \nu^*)$——$L$ 是 Lagrange 函数。所以 KKT 的凸分析形式一句话：<strong>原点属于 Lagrange 函数对 $x$ 的次微分</strong>。可微时它退回 $0 = \nabla_x L$，不可微时次微分接住一切。</span>这是第5篇「次梯度最优性条件」与「KKT」的统一视图。

**非光滑 KKT 的一例**：求解 $\min |x|$ s.t. $x \ge 1$。目标在 $x=1$ 处次微分 $\partial |x| = \{1\}$（$x>0$），法锥 $\mathcal{N}_{[1,\infty)}(1) = \{v : v \le 0\}$（向左的方向）……KKT 的凸分析形式写 $0 \in \{1\} + \mathcal{N}_{[1,\infty)}(1)$，即存在 $v \le 0$ 使 $1 + v = 0$，$v = -1$——成立。最优解 $x^* = 1$。**这里 $\partial f_0$ 不是梯度而是一整个集合，KKT 的凸分析形式照样工作。**

**辨析｜易错点：** 初学者常把 $0 \in \partial f_0(x^*) + \sum \lambda_i \partial f_i(x^*) + \mathcal{N}(x^*)$ 直接等同于可微形式的 $0 = \nabla f_0 + \sum \lambda_i \nabla f_i + A^T \nu$。实际上，凸分析形式是「集合包含」，可微形式是「向量等式」——前者是后者的严格推广。当 $f_0$ 或 $f_i$ 在 $x^*$ 不可微时，$\partial f_i$ 可能是多值集合，驻点条件变成「存在某个连线能找零」而非「所有方向都为零」。这条区别在线性规划（目标线性、处处可微）和 $\ell_1$ 正则化问题（LASSO，零点处不可微）之间划了一条清晰的界线：对 LASSO，KKT 条件是「$0$ 属于某个区间」，不是「等于某个确定向量」。

## 4 公式解析：从 Fermat 到 KKT 的推导

KKT 不是天降的清单，它由「可行方向 + 分离定理」推出来。以仅有不等式约束（$Ax = b$ 省略）为例：

$$x^*\ \text{最优} \iff \text{不存在可行下降方向}$$

- **第一步，刻画可行方向锥**：$x^*$ 处活动约束集 $\mathcal{I}(x^*) = \{i : f_i(x^*) = 0\}$。可行方向 $d$ 须满足 $\langle \nabla f_i(x^*), d \rangle \le 0$ 对 $i \in \mathcal{I}(x^*)$（在约束函数可微且约束规格成立时，这是充要的）。
- **第二步，下降方向锥**：$d$ 让目标下降须 $\langle \nabla f_0(x^*), d \rangle < 0$。最优性 = 这两个锥不交。
- **第三步，分离定理出手**：两个凸锥不相交 ⟹ 存在超平面分离，得到系数 $\lambda_i \ge 0$（$i \in \mathcal{I}$）使 $\nabla f_0(x^*) + \sum_{i \in \mathcal{I}} \lambda_i \nabla f_i(x^*) = 0$——这是 Gordan/Farkas 型对偶定理的直接应用。
- **第四步，补齐符号与零乘子**：非活动约束令 $\lambda_i = 0$，互补松弛自动成立；对偶可行性 $\lambda_i \ge 0$ 来自分离方向的选择。四族条件一次成型。

**这条推导的要点**：<span class="marginnote">KKT 的深层证明完全建立在第3篇的分离定理上——<strong>最优性条件 = 分离定理在锥语言下的转写</strong>。这也解释了为什么第4篇强对偶（Slater）、第5篇 KKT 约束规格本质上是同一个「锥不交」条件在不同高度的投影。</span>约束规格（Slater、LICQ）保证「可行方向锥」能被精确刻画——没有它，KKT 只是必要条件。

## 5 连接：KKT 与算法

KKT 条件的工程角色是「算法的心脏」。**内点法**把不等式约束用对数障碍吸收，把 KKT 方程组改写成一个可解的牛顿系统，迭代让互补松弛从 $\lambda_i f_i = -\mu$ 逐渐趋零；**活性集方法**则显式猜测「哪些约束取等」，在一个固定活动集上解等式 KKT，再校验互补松弛。<span class="marginnote">这些算法不是「用 KKT 验算」，而是<strong>把 KKT 当成需要被数值求解的方程</strong>——内点法解的是「互补松弛的 $\mu$-扰动版」，活性集法解的是「固定活动集的等式 KKT」。理解 KKT，就能看懂这两类算法的每一步在干嘛。</span>在「从极限到大模型」的数值优化课程里，你会在 QP 求解器、SQP 与增强拉格朗日里反复遇见本节的四族条件。

再看两个最熟悉的机器学习案例如何落在四族条件上，把抽象落回具体：

- **SVM**：原始问题的 KKT 驻点给出 $w^* = \sum_i \alpha_i y_i x_i$（$w$ 是支撑向量的加权和），互补松弛给出「$\alpha_i > 0 \Rightarrow y_i(w^Tx_i + b) = 1$」——只有间隔边界上的样本（支撑向量）进入 $w$ 的表达式。
- **LASSO**：$\min \tfrac12\|Ax - b\|^2 + \lambda\|x\|_1$ 的 KKT 写 $0 \in A^T(Ax^* - b) + \lambda \partial\|x^*\|_1$。对 $x_j^* = 0$ 的分量，互补松弛换成「次微分条件」：$|(A^T(Ax^* - b))_j| \le \lambda$——这就是「梯度不够大就不激活变量」的稀疏机制。

两条案例共同说明：**互补松弛决定「选谁」，驻点条件决定「权重多少」**——几何上的活动集，在算法里就是支撑向量与激活变量集合。

### 术语速查：最优性条件的名词对照

| 术语 | 一句话定义 | 出处 |
| --- | --- | --- |
| 拉格朗日函数 | $L = f_0 + \sum\lambda_i f_i + \nu^T(Ax-b)$，约束入目标 | 本篇 / 第4篇 |
| 驻点条件 | $0 = \nabla_x L$（可微）或 $0 \in \partial_x L$（不可微） | 本篇 |
| 互补松弛 | $\lambda_i f_i(x^*) = 0$，活动集的代数签名 | 本篇 |
| 法锥 | $\mathcal{N}_C(x) = \{v : \langle v, y-x\rangle \le 0,\ \forall y \in C\}$ | 本篇 |
| 约束规格 | Slater / LICQ，保证「KKT 充要」的前提 | 本篇 |
| 活动集 | 在 $x^*$ 处取等的约束指标集 $\mathcal{I}(x^*)$ | 本篇 |
| 内点法 | 用对数障碍 + 牛顿法数值解 $\mu$-扰动 KKT | 本篇 |
| 支撑向量 | $\alpha_i > 0$ 的样本，即活动约束对应点 | 本篇 |

## 6 小结

- **KKT 四族条件**：原始可行、对偶可行（$\lambda \ge 0$）、互补松弛（$\lambda_i f_i(x^*) = 0$）、驻点（梯度平衡）——凸问题下充要。
- **凸分析形式**：$0 \in \partial f_0(x^*) + \sum \lambda_i \partial f_i(x^*) + \mathcal{N}(x^*)$——等式/不等式统一成锥与次微分。
- **法锥** $\mathcal{N}_C(x) = \{v : \langle v, y-x\rangle \le 0\, \forall y \in C\}$，是指示函数的次微分。
- 推导路径：可行/下降锥不交 → **分离定理** → 乘子存在——KKT 是分离定理的转写。
- 约束规格（Slater/LICQ）是「锥不交 ⟹ 乘子存在」的前提。
- 内点法与活性集方法都是「数值求解 KKT 方程组」的两种姿势。

在下一节，我们转向凸集的「骨架」——**极值表示**：极点、极值方向与 Minkowski 定理如何用少数点重建整个凸集。
