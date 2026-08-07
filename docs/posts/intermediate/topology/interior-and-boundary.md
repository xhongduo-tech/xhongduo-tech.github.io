---
title: 内部与边界
date: 2026-08-07
---

# 内部与边界

<div class="epigraph">
<p>在数学中，你并不理解事物，你只是对它们习以为常。</p>
<footer>—— 约翰 · 冯 · 诺依曼（John von Neumann）</footer>
</div>

<div class="article-byline">
<p>第二级 · 拓扑学 ｜ 尤承业《基础拓扑学讲义》第一章 ｜ Munkres《Topology》§17 ｜ 2026-08-07</p>
</div>

## 为什么从内部与边界开始

上一课我们用**闭包**回答了「哪些点被 $A$ 吸引」，把「附着」精确化了。但闭包只告诉我们「谁算贴边」，还没有区分**三种本质不同的点**：完全在 $A$ 肚子里、完全在 $A$ 外面、以及正正骑在 $A$ 的边界上。内部与边界这两个概念，就是把空间里的每个点按「与 $A$ 的亲疏」分成三类。这个三分法在数学分析里早就以「开区间、闭区间、端点」的直觉存在，在拓扑学里被提升为**只依赖开集、与度量无关**的纯拓扑概念——于是「边界」不再依赖长度或距离，而成为连续变形下的不变量。<span class="marginnote">分析里「端点」依赖实数轴的顺序与距离，拓扑的「边界」只依赖开集结构。两者在多数情形重合，但后者更强——它能在没有度量的空间（如有限拓扑空间、商空间）里照常工作。</span>

## 1 内部：含于 A 的最大开集

**内部（interior）**：设 $A$ 是拓扑空间 $X$ 的子集，把 $A$ 内所有开集取并，得到的集合称为 $A$ 的**内部**，记作 $\operatorname{int} A$ 或 $A^\circ$：

$$\operatorname{int} A = \bigcup \{U \subset A \mid U \text{ 是 } X \text{ 的开集}\}$$

这个定义有一处值得停下来确认：为什么「取所有含于 $A$ 的开集的并」就一定还是开集、并且还含于 $A$？因为开集的**任意并**仍是开集（开集公理第三条），且每个开集都含于 $A$，并集自然含于 $A$。所以 $\operatorname{int} A$ 是含于 $A$ 的**最大开集**，它的存在性是公理保证的，不需要额外构造。<span class="marginnote">「取所有……再取并」是一个反复出现的构造手法：含于 $A$ 的所有开集的并给出内部，包含 $A$ 的所有闭集的交给出闭包。一个管「最里面的开壳」，一个管「最外面的铁皮」，两者互为对偶。</span>

用邻域语言可以更直觉地刻画内部中的点：

$$x \in \operatorname{int} A \iff A \text{ 是 } x \text{ 的邻域}$$

也就是说，$x$ 在 $A$ 内部，当且仅当存在一个开集 $U$ 使得 $x \in U \subset A$——$x$ 有一小片「纯粹属于 $A$ 的地盘」。这一点在实轴上非常直观：$[0,1]$ 的内部是 $(0,1)$，因为只有开区间里的点才拥有完全落进 $[0,1]$ 的小邻域；端点 $0$ 的任何一个邻域都会溢出到负半轴去。

## 2 边界：与 A 若即若离的点

**边界（boundary / frontier）**：$A$ 的**边界**定义为闭包去掉内部：

$$\partial A = \operatorname{cl} A \setminus \operatorname{int} A$$

边界上的点有什么特征？它同时「够得着」$A$ 和 $A$ 的补集：**$x \in \partial A$ 当且仅当 $x$ 的每个邻域都与 $A$ 相交、也与 $X \setminus A$ 相交。** 左边那半句说明 $x \in \operatorname{cl} A$，右边那半句说明 $x \notin \operatorname{int} A$——两个条件合起来正好是边界的定义。

看几个例子：

- 在 $\mathbb{R}$ 中，$\partial(0,1) = \{0, 1\}$，$\partial[0,1] = \{0,1\}$——**开区间与闭区间有相同的边界**，因为边界只看「贴没贴到外面」，不看端点算不算自己人。
- 在 $\mathbb{R}^2$ 中，单位开圆盘 $B = \{(x,y) \mid x^2 + y^2 < 1\}$ 的边界是单位圆周 $S^1 = \{(x,y) \mid x^2 + y^2 = 1\}$。
- 在 $\mathbb{R}$ 中，$\partial \mathbb{Q} = \mathbb{R}$：有理数无处不稠，也无处是内部，所以每个实数都骑在有理数集的边界上。<span class="marginnote">$\mathbb{Q}$ 的边界是整个实数轴——这是「处处稠密但内部为空」的集合的典型命运。这种集合在实数论与测度论里是主角，见第二级《实变函数与测度论》。</span>

## 3 三分宇宙：内部、外部与边界的划分

把「外部」也补进来，就得到对空间的一个漂亮三分。**外部（exterior）**定义为补集的内部：

$$\operatorname{ext} A = \operatorname{int}(X \setminus A)$$

三个集合互不相交，且并起来是全体：

$$X = \operatorname{int} A \;\dot\cup\; \partial A \;\dot\cup\; \operatorname{ext} A$$

其中 $\dot\cup$ 表示不交并。这个三分法把「属于 $A$ 与否」和「接近 $A$ 与否」两个独立信息组装在一起：内部里的点既属于又接近，外部里的点既不属也不接近，而边界上的点**最微妙——不属、却接近**。

由内部与边界立即可得两个与开闭集挂钩的准则：

- $A$ 是开集 $\iff \operatorname{int} A = A$（开集就是「没有边界点被算进来」的集，它等于自己的内部）。
- $A$ 是闭集 $\iff \partial A \subset A$（闭集把边界全部收编）。

第二个准则值得验证一下：若 $A$ 闭，则 $\operatorname{cl} A = A$，自然 $\partial A \subset A$；反过来若 $\partial A \subset A$，则 $\operatorname{cl} A = \operatorname{int} A \cup \partial A \subset A \cup A = A$，而 $A \subset \operatorname{cl} A$ 恒真，故 $\operatorname{cl} A = A$，$A$ 闭。**一个集合闭不闭，只看它有没有把边界「兜住」。**

### 例子：实轴上的内部与边界速查

把 $\mathbb{R}$ 里的典型集合的四个量（内部、闭包、边界、外部）列成一张表，是记忆与做题的捷径：

| 集合 $A$ | $\operatorname{int} A$ | $\operatorname{cl} A$ | $\partial A$ | $\operatorname{ext} A$ |
| --- | --- | --- | --- | --- |
| $(0,1)$ | $(0,1)$ | $[0,1]$ | $\{0,1\}$ | $(-\infty,0)\cup(1,\infty)$ |
| $[0,1]$ | $(0,1)$ | $[0,1]$ | $\{0,1\}$ | $(-\infty,0)\cup(1,\infty)$ |
| $\mathbb{Q}$ | $\emptyset$ | $\mathbb{R}$ | $\mathbb{R}$ | $\emptyset$ |
| $\mathbb{Z}$ | $\emptyset$ | $\mathbb{Z}$ | $\mathbb{Z}$ | $\mathbb{R}\setminus\mathbb{Z}$ |
| $\{1/n \mid n\ge 1\}$ | $\emptyset$ | $\{1/n\}\cup\{0\}$ | $\{1/n\}\cup\{0\}$ | $\mathbb{R}\setminus(\{1/n\}\cup\{0\})$ |

注意表格第三、四行里 $\partial A = \operatorname{cl} A$——当内部为空时，边界就是整个闭包。而开区间与闭区间的内部、闭包、边界**完全相同**，区别只在「$A$ 自己收不收边界」。

### 内部与边界的运算规律

内部与边界还满足几条常用运算律，证明都直接从定义走：

- $\operatorname{int}(A \cap B) = \operatorname{int} A \cap \operatorname{int} B$（内部对有限交可交换）。
- $\operatorname{cl}(A \cup B) = \operatorname{cl} A \cup \operatorname{cl} B$（闭包对有限并可交换）。
- 一般地 $\operatorname{int}(A \cup B) \supset \operatorname{int} A \cup \operatorname{int} B$——并的内部只大不小，反过来不一定成立。

这些运算律在第二级《实变函数》与《数学分析》里计算 Lebesgue 测度、处理开集覆盖时是常用工具。

## 4 公式解析：边界与闭包的交换关系

边界有两个等价刻画，其中一个在计算时格外好用：

$$\partial A = \operatorname{cl} A \cap \operatorname{cl}(X \setminus A)$$

对这条公式做三步拆解：

- **第一步，回忆闭包的双重身份**：$\operatorname{cl} A$ 是「包含 $A$ 的最小闭集」，等价地，$x \in \operatorname{cl} A$ 当且仅当 $x$ 的每个邻域都与 $A$ 相交。后一种「邻域视角」是理解下面推导的钥匙。
- **第二步，验证 $x \in \partial A$ 的两个条件**：$x \in \partial A = \operatorname{cl} A \setminus \operatorname{int} A$ 意味着「每个邻域都与 $A$ 相交」（来自 $x \in \operatorname{cl} A$），并且「$A$ 不是 $x$ 的邻域」。后一句等价于「每个邻域都跑到 $A$ 外面去」，即每个邻域都与 $X \setminus A$ 相交。
- **第三步，合成**：「每个邻域都与 $A$ 相交」正是 $x \in \operatorname{cl} A$，「每个邻域都与 $X \setminus A$ 相交」正是 $x \in \operatorname{cl}(X \setminus A)$。两者同时成立就是 $x$ 属于交集，于是 $\partial A = \operatorname{cl} A \cap \operatorname{cl}(X \setminus A)$。

这个公式揭示了边界的**自对偶性**：$\partial A = \partial (X \setminus A)$——一个集合和它的补集共享同一条边界。直观上，$A$ 的边界把 $A$ 与补集隔开，从哪边看都是「那堵墙」。

## 5 辨析｜易错点

**辨析｜易错点：** 初学内部与边界，有四处容易踩坑：

- **内部与开核混淆**：$\operatorname{int} A$ 是「最大的开子集」，但它不一定是 $A$ 的「所有内点」在视觉上的总和——在离散拓扑里每个子集都开，于是 $\operatorname{int} A = A$ 恒成立，边界恒空。边界是否为空，是**依赖具体拓扑**的判断，不能凭直觉的「形状」下结论。
- **「边界很薄」是度量世界的偏见**：在普通欧氏空间中边界确实「没有面积」，但拓扑定义本身与体积无关。$\partial \mathbb{Q} = \mathbb{R}$ 就是一个「厚边界」的例子。
- **把 $\partial A \subset A$ 误当成「边界由 $A$ 决定」**：边界由 $A$ 与它的补集共同决定。$A$ 的开闭性质只反映「边界算不算自己人」，边界本身的形状是双方共同的。
- **$\operatorname{cl} A$ 与 $\operatorname{int} A \cup \partial A$ 的关系**：两者相等，但 $\operatorname{cl} A \neq \operatorname{int} A \cup \partial(A \setminus \operatorname{int} A)$ 这类「套娃」式子极易算错。固定公式只有三条：$\operatorname{cl} A = \operatorname{int} A \cup \partial A$、$\partial A = \operatorname{cl} A \cap \operatorname{cl}(X\setminus A)$、$X = \operatorname{int} A \dot\cup \partial A \dot\cup \operatorname{ext} A$。

## 6 小结

- **内部** $\operatorname{int} A$：含于 $A$ 的最大开集，$x \in \operatorname{int} A \iff A$ 是 $x$ 的邻域。
- **边界** $\partial A = \operatorname{cl} A \setminus \operatorname{int} A$：$x \in \partial A$ 当且仅当 $x$ 的每个邻域都与 $A$ 及其补集同时相交。
- **三分宇宙**：$X = \operatorname{int} A \;\dot\cup\; \partial A \;\dot\cup\; \operatorname{ext} A$，内部、边界、外部互不相交。
- **开闭判定**：$A$ 开 $\iff \operatorname{int} A = A$；$A$ 闭 $\iff \partial A \subset A$。
- **自对偶**：$\partial A = \partial (X \setminus A)$，由 $\partial A = \operatorname{cl} A \cap \operatorname{cl}(X\setminus A)$ 保证。

在下一节，我们将回到闭包本身，问一个更精细的问题：究竟什么样的点会被闭包「吸引」进来？这就是**极限点（聚点）与闭包的等价刻画**。
