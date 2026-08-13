---
title: 交换 C\*-代数与 Gelfand 变换
date: 2026-08-07
---

# 交换 C\*-代数与 Gelfand 变换

<div class="epigraph">
<p>数学是给不同事物取同一个名字的艺术。</p>
<footer>—— 亨利 · 庞加莱（Henri Poincaré）</footer>
</div>

<div class="article-byline">
<p>第二级 · 算子代数 ｜ Murphy《C\*-Algebras and Operator Theory》第1章 ｜ 2026-08-07</p>
</div>

## 为什么从 Gelfand 变换开始

上一节的末尾留下一个断言：**每个交换 C\* 代数都同构于某个 $C(X)$**。这一节就把它证明出来，并同时揭示一个更惊人的事实：那个 $X$ 不是外部给定的，而是**从代数内部自己长出来的**——它由全体「特征」构成，是代数的「灵魂图谱」。这就是 **Gelfand 变换**，它像一台 X 光机，把抽象交换代数照成一张连续函数的照片。

Gelfand 变换的地位怎么强调都不过分：它是连续函数演算（第 4 篇）的抽象根源，是谱定理（第 13 篇）的证明引擎，也是「非交换几何」里把几何重新定义为「算子代数」的第一步。<span class="marginnote">庞加莱的这句话恰好预言了 Gelfand 的成就：不同面目（$C(X)$、$C_0(X)$、$c_0$、对角矩阵代数、$C(\mathbb{T})$）的交换 C\* 代数，在 Gelfand 变换眼里是同一个概念的不同实例。</span>

## 1 特征：代数的「观察点」

设 $A$ 是交换含幺 C\* 代数。一个**特征（character / multiplicative linear functional）**是非零线性泛函 $\tau:A\to\mathbb{C}$，满足

$$\tau(ab)=\tau(a)\tau(b), \qquad \tau(1)=1.$$

特征自动是连续的（范数为 1），且满足 $\tau(a^*)=\overline{\tau(a)}$。

**定理（特征与极大理想一一对应）**：映射 $\tau\mapsto\ker\tau$ 在「特征」与「极大理想」之间建立双射。每个极大理想恰是某个特征的核，每个特征唯一决定一个极大理想。<span class="marginnote">代数的「极大理想」是「函数在某个点消失」的抽象翻译：$C(X)$ 里，每个点 $x$ 对应极大理想 $\{f:f(x)=0\}$，每个极大理想都是这种「在某点消失」的集合。特征就是「在该点取值」的求值映射 $\mathrm{ev}_x$。</span>

**特征空间（spectrum）** $\Delta(A)$：$A$ 的全体特征，配上**弱-$\ast$ 拓扑**（即逐点收敛拓扑），它是紧 Hausdorff 空间。把 $\Delta(A)$ 想成「抽象点空间」：交换 C\* 代数的点，不再是几何里的坐标，而是代数上的乘法线性泛函。

## 2 Gelfand 变换：把代数元变成函数

有了特征空间，定义

$$\Gamma: A \longrightarrow C(\Delta(A)), \qquad \Gamma(a)(\tau) = \widehat{a}(\tau) = \tau(a).$$

即：$a$ 的 **Gelfand 变换** $\widehat{a}$ 是特征空间上的连续函数，它在每个「观察点」$\tau$ 的取值，就是 $\tau$ 对 $a$ 的读数。这是**对偶**的胜利：把「代数元」重新看成一个「在点空间上变化的函数」。

**关键性质（逐个验证）**：$\Gamma$ 是代数同态（$\widehat{ab}=\widehat{a}\widehat{b}$，$\widehat{1}=1$）、保对合（$\widehat{a^*}=\overline{\widehat{a}}$）。但它现在还不一定保范——这正是要证的。

**命题（谱的翻译）**：对每个 $a\in A$ 与 $\lambda\in\mathbb{C}$：

$$\lambda\in\sigma(a) \iff \exists\,\tau\in\Delta(A) \text{ 使 } \tau(a)=\lambda, \qquad \sigma(a) = \widehat{a}(\Delta(A)).$$

谱就是「Gelfand 变换的像」——**谱等于函数 $\widehat{a}$ 的值域**。这把我们带回第 3 篇的乘法算子直觉：抽象算子对每个「点」$\tau$ 给出一个「标量读数」，全体读数排成一串，就是谱。<span class="marginnote">这条翻译是整套理论的支点：代数里最神秘的谱，在特征空间里退化成「函数取值的集合」。自伴算子的实谱、正规算子的范数等于谱半径，都能从这里一眼看穿。</span>

## 3 定理：交换 C\*-代数 = C(X)

**定理（Gelfand–Naimark，交换情形）**：设 $A$ 是交换含幺 C\* 代数。则 Gelfand 变换 $\Gamma:A\to C(\Delta(A))$ 是**等距 $\ast$-同构**。

证明的三个关键步骤：

**单射**：$\widehat{a}=0$ 意味着 $\sigma(a)=\{0\}$，即 $r(a)=0$。对正规元 $r(a)=\|a\|$（第 7 篇），故 $a=0$。
**保范**：$\|\widehat{a}\|_\infty=\sup|\tau(a)|=r(a)=\|a\|$（正规元用范数=谱半径）。
- **满射**：$\Gamma(A)$ 是 $C(\Delta(A))$ 中的闭 $\ast$-子代数，且分离点、含常数函数，由 **Stone–Weierstrass 定理**它必是全体 $C(\Delta(A))$。<span class="marginnote">Stone–Weierstrass 定理是「用多项式/简单函数逼近任意连续函数」的终极版本：一个分离点且含常数的闭子代数就是整座函数代数。它在这里充当「稠密延拓」的万能钥匙——和函数演算里的 Weierstrass 逼近是同一个家族的两位成员。</span>

**无幺情形**：$A$ 无幺时，$\Delta(A)$ 是局部紧但非紧的空间，$\Gamma$ 给等距 $\ast$-同构 $A\cong C_0(\Delta(A))$。把 $A$ 单位化（第 12 篇）后，$\Delta$ 就紧化了。

## 4 公式解析：$\widehat{a}(\tau)=\tau(a)$

$$
\widehat{a}(\tau) = \tau(a), \qquad \|\widehat{a}\|_\infty = \|a\|
$$

- **第一步，看定义**：$\widehat{a}$ 的「自变量」是特征 $\tau$，不是数也不是向量。函数演算把元素 $a$ 当成「在 $\tau$ 处取值 $\tau(a)$ 的复值函数」。这个反转视角是整个理论的灵魂：**元素变函数，泛函变点**。
- **第二步，看为什么 $\|\widehat{a}\|_\infty=\|a\|$**：$\|\widehat{a}\|_\infty=\sup_\tau|\tau(a)|$。对正规 $a$，$\tau(a)$ 遍历 $\sigma(a)$（谱翻译），故上确界 = 谱半径 $r(a)$；而 C\* 恒等式给出 $r(a)=\|a\|$。三步环环相扣：谱翻译 → 谱半径 → C\* 恒等式。
- **第三步，看它统一了什么**：对 $C(X)$ 本身，$\Delta(C(X))\cong X$（点 $x$ 对应求值 $\mathrm{ev}_x$），Gelfand 变换就是恒等映射。所以「每个交换 C\* 代数 ≅ $C(X)$」是自指的重言——抽象理论把具体对象认成了自己，这正是庞加莱「给不同事物取同一个名字」的数学形态。

**例子**：
- $A=C(\mathbb{T})$：$\Delta(A)\cong\mathbb{T}$，Gelfand 变换是恒等；
- $A=\ell^\infty$：$\Delta(A)\cong\beta\mathbb{N}$（Stone–Čech 紧化），Gelfand 变换给出 $\ell^\infty\cong C(\beta\mathbb{N})$；
- $A=C^*(S)$（单侧移位生成的 C\* 代数，见第 14 篇 Toeplitz 代数中的对角部分）：其交换子代数的特征空间回到 $C(\mathbb{T})$，这就是为何 Toeplitz 算子要拿圆周上的函数做符号。

## 5 Gelfand 变换的实际用法：函数演算的重生

有了 $A\cong C(\Delta(A))$，第 4 篇的连续函数演算变得「免费」：

对正规元 $a$，取包含 $a$ 与 $1$ 的交换 C\* 子代数 $B=C^*(a,1)$。Gelfand 变换给 $\sigma(a)\cong\Delta(B)$（谱翻译），于是

$$f\ \longmapsto\ \Gamma^{-1}(f\circ\widehat{a}\,), \qquad f\in C(\sigma(a))$$

就定义了 $f(a)$，且自动满足谱映射定理、范数等式、函数复合——**第 4 篇的整套构造被吸收进 Gelfand 理论**。<span class="marginnote">这是「抽象理论反过来解释具体构造」的典范：函数演算不是零散的技巧，而是交换理论的自然推论。第 13 篇谱定理将进一步把 $C(\sigma(a))$ 推广到有界 Borel 函数，得到谱测度。</span>

**辨析｜易错点：**Gelfand 变换对**一般 Banach 代数**也有定义，但**不一定等距、不一定满射**——非 C\* 的 Banach 代数（如 $L^1(\mathbb{T})$ 卷积代数）的 Gelfand 变换只是同态，谱会「变形」。Gelfand 变换的完美性（等距 $\ast$-同构）是 C\* 恒等式独有的恩赐，把它误当成 Banach 代数的普遍性质，是常见误区。

## 6 例：Gelfand 变换亲手算

把 Gelfand 变换在三个具体代数上各算一遍，抽象的机器就透明了。

**$A=C(\mathbb{T})$**：特征 = 求值映射 $\mathrm{ev}_z$（$z\in\mathbb{T}$），$\Delta(A)\cong\mathbb{T}$，Gelfand 变换是恒等（$f\mapsto f$）。「$C(X)\cong C(\Delta(C(X)))$」是重言，却是一切例子的基准。

**$A=\ell^\infty$**：$\Delta(A)\cong\beta\mathbb{N}$（Stone–Čech 紧化，一个很大的空间）。Gelfand 变换 $\ell^\infty\to C(\beta\mathbb{N})$ 是等距同构——「有界序列」被重新认成「$\beta\mathbb{N}$ 上的连续函数」。超滤子在这里化身特征。

**$A=c_0$**：无幺，$\Delta(c_0)=\mathbb{N}$（离散、局部紧），Gelfand 变换给出 $c_0\cong C_0(\mathbb{N})$。无幺情形的 $C_0$ 版本在此显形。

**$A=C^*(S)$ 的交换子代数（Toeplitz 预告）**：由酉 $S$（第 14 篇）生成的交换 C\* 代数是 $C(\sigma(S))=C(\mathbb{T})$——这就是为什么 Toeplitz 算子要用圆周函数做符号。

**$A=C^*(a,1)$（单元素生成）**：$\Delta(A)\cong\sigma(a)$（谱翻译），Gelfand 变换 $\widehat a:\tau\mapsto\tau(a)$ 就是「把 $a$ 看成谱上的恒等函数」——这是连续函数演算（第 4 篇）的抽象版本。

**一句话总结**：Gelfand 变换把「抽象代数」翻译回「函数代数」，特征空间就是「抽象点」，谱就是「函数值域」。

## 7 延伸：特征空间与谱的拓扑

$\Delta(A)$ 不只是集合，它带着拓扑，而拓扑里藏着信息。

**弱-$\ast$ 拓扑**：$\Delta(A)$ 配 $A^*$ 的弱-$\ast$ 拓扑（逐点收敛）。$\tau_\alpha\to\tau$ ⟺ $\tau_\alpha(a)\to\tau(a)$ 对所有 $a$。含幺时 $\Delta(A)$ 紧（Alaoglu），无幺时局部紧。

**紧化与单位化**：$A$ 无幺时，$A^+=A\oplus\mathbb{C}1$ 的单位化对应 $\Delta(A^+)=\Delta(A)\cup\{\infty\}$（一点紧化）。「加单位元」与「紧化」在 Gelfand 世界里是同一件事。

**谱 = 值域（再强调）**：$\sigma(a)=\widehat a(\Delta(A))$。这个「谱 = 连续函数的值域」的陈述，让「自伴 ⟹ 实谱」变成「实值函数的像在 $\mathbb{R}$ 里」的平凡事实——Gelfand 变换让谱的直觉变得「透明」。

**连通性**：$\Delta(A)$ 的连通分支对应 $A$ 的「幂等元分解」（$p^2=p$）。投影 $p$ 与 $\Delta(A)$ 的闭开子集一一对应——拓扑与代数的字典又添一条。

**与 K 理论**：$K_0(C(X))$ = $X$ 的复向量丛（第 25 篇）。交换代数的 K 理论 = 空间的拓扑 K 理论——Gelfand 变换把「代数 K 理论」与「拓扑 K 理论」缝合。

## 8 延伸：Gelfand 变换之后的道路

Gelfand 变换是交换理论的终点，也是非交换理论的起点。

**非交换的困难**：$A$ 非交换时没有特征（乘法线性泛函不够用），Gelfand 变换无从定义。取而代之的是表示论（第 9、11 篇）：用 $B(\mathcal{H})$ 上的表示「代替」特征空间。

**原始谱（$\mathrm{Prim}(A)$）**：非交换代数的「点空间」由不可约表示的核（原始理想，第 12 篇）充当，配 Jacobson 拓扑。这是非交换几何（Connes）的出发点——用理想拓扑代替点集拓扑。

**谱测度**：对单个正规算子，Gelfand 变换 + Riesz–Markov 给出谱测度（第 13 篇）。交换理论在单算子上的「完整爆发」就是谱定理。

**分类**：交换 C\* 代数由 $C_0(\Delta(A))$ 完全决定，故分类平凡；非交换 C\* 代数的分类（第 26 篇）则靠 K 理论与迹——Gelfand 变换的「直接对象」丢了，不变量却更丰富。

**一句话总结**：Gelfand 变换是「交换 = 函数论」的宣言；非交换世界里，它让位给表示、原始谱与 K 理论——但那条「代数 ⟷ 几何」的字典，永远是它的遗产。

## 9 小结

- **特征** $\tau$：非零乘法线性泛函；与极大理想一一对应；特征空间 $\Delta(A)$ 是紧 Hausdorff 空间。
- **Gelfand 变换** $\Gamma$：$\widehat{a}(\tau)=\tau(a)$，把代数元变成特征空间上的函数。
- **谱翻译**：$\sigma(a)=\widehat{a}(\Delta(A))$——谱 = Gelfand 变换的值域。
- **交换 Gelfand–Naimark 定理**：$\Gamma$ 是等距 $\ast$-同构 $A\cong C(\Delta(A))$（无幺时 $A\cong C_0(\Delta(A))$），满射靠 Stone–Weierstrass。
- **函数演算重生**：正规元的 $f(a)$ 通过 Gelfand 变换 + 复合实现，谱映射与范数等式自动成立。
- **界限**：完美等距性是 C\* 独有的；一般 Banach 代数的 Gelfand 变换会丢失信息。

在下一节，我们把这个交换定理推向最完整的形式——**Gelfand–Naimark 定理**：证明每个（未必交换的）C\* 代数都能忠实嵌入某个 $B(\mathcal{H})$