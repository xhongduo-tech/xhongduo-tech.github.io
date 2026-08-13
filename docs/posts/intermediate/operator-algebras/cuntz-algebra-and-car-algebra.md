---
title: Cuntz 代数与 CAR 代数
date: 2026-08-07
---

# Cuntz 代数与 CAR 代数

<div class="epigraph">
<p>科学无法解开自然的终极奥秘，因为我们自己正是这奥秘的一部分。</p>
<footer>—— 马克斯 · 普朗克（Max Planck）</footer>
</div>

<div class="article-byline">
<p>第二级 · 算子代数 ｜ Davidson《C\*-Algebras by Example》第12章 ｜ 2026-08-07</p>
</div>

## 为什么从 Cuntz 代数与 CAR 代数开始

第 17 篇的 AF 代数是「有限维拼图」，处处可控。而这一篇的两位主角站在完全相反的极端：**Cuntz 代数** $\mathcal{O}_n$ 由等距生成、**纯无限**且**单**——它是非交换世界里「最疯狂、最简洁」的一族代数；**CAR 代数**则是费米子（电子、质子）的量子场论代数，看似平凡（UHF $2^\infty$）却在数学物理中无处不在。

把两者放一起讲不是偶然：它们分别对应量子世界的两种统计——**玻色型**（$\mathcal{O}_n$ 的等距生成元是玻色子式的）与**费米型**（CAR 的反对易子）。Cuntz 代数以「$\sum S_iS_i^*=1$」这一条关系定义，却拥有令人惊叹的深度（分类、K 理论、表示全是前沿海题）；CAR 代数用「$\{a_i,a_j^*\}=\delta_{ij}$」的反对易关系定义，是量子场论与统计力学的标准舞台。理解两者，就理解了「由生成元与关系定义 C\* 代数」的整个范式的成败。

## 1 Cuntz 代数：一条关系定义的宇宙

**Cuntz 代数** $\mathcal{O}_n$（$n\ge2$）：由 $n$ 个**等距** $S_1,\dots,S_n$ 生成的含幺 C\* 代数，满足

$$S_i^*S_i = 1 \quad (1\le i\le n), \qquad \sum_{i=1}^n S_iS_i^* = 1.$$

第一条说每个 $S_i$ 保内积（等距）；第二条说「$n$ 个互不正交的像子空间 $S_iS_i^*\mathcal{H}$ 恰好铺满整个空间」。<span class="marginnote">等距 $S_i$ 就像「每步的移位算子」，而 $\sum S_iS_i^*=1$ 要求这些移位「覆盖一切」。Cuntz（1977）证明：$\mathcal{O}_n$ 是<strong>单</strong> C\* 代数，且 $\mathcal{O}_n\cong\mathcal{O}_m$ 当且仅当 $n=m$——表示理论里，不同的 $n$ 给出不同的代数。</span>

**定理（Cuntz 代数基本定理）**：$\mathcal{O}_n$ 是**简单、纯无限、可核（nuclear）**的 C\* 代数，且 $\mathcal{O}_n$ 有唯一迹态……不对，$\mathcal{O}_n$ **没有**任何迹态——它太「大」了。它的 K 理论：$K_0(\mathcal{O}_n)\cong\mathbb{Z}/(n-1)\mathbb{Z}$，$K_1(\mathcal{O}_n)=0$。

**纯无限（purely infinite）**：每个非零正元素 $a$ 都「无穷大」——存在 $b$ 使 $b^*b=a$ 且 $bb^*<a$（即 $a$ 的 Murray–von Neumann 等价类「吸收」一个更小的投影）。纯无限代数没有迹态，所有投影要么是 0 要么等价于「无穷多个自己」的部分。<span class="marginnote">纯无限是 C\* 世界「非 AF」的极端形态：AF 代数投影「很小」、可数可加地铺起来；$\mathcal{O}_n$ 的投影「无限大」、自我增殖。Cuntz 代数让「简单 + 纯无限」成为可研究的新物种，第 26 篇单 C\* 代数的分类正是围绕「有限（AF 型）— 无限（纯无限型）」的两极展开。</span>

## 2 CAR 代数：费米子的代数

**CAR 代数（canonical anticommutation relation algebra）**：由生成元 $a(f)$（$f\in$ 某可分 Hilbert 空间 $\mathcal{K}$）生成的 C\* 代数，满足

$$a(f)a(g)+a(g)a(f)=0, \qquad a(f)a(g)^*+a(g)^*a(f)=\langle f,g\rangle\,1.$$

第二个式子写成 $\{a(f),a(g)^*\}=\langle f,g\rangle$——**反对易子**（anticommutator）。$a(f)$ 是**消灭算子**，$a(f)^*$ 是**产生算子**。<span class="marginnote">反对易关系是费米子的律法：$a(f)^2=0$（两个相同费米子不能在同一态——Pauli 不相容原理的代数形式），而 $a(f)a(g)^*$ 与 $a(g)^*a(f)$ 相差一个标量。CAR 代数把所有「费米子多体问题」的代数结构压缩成这一条关系。</span>

**Fock 表示**：在费米子 Fock 空间 $\bigwedge\mathcal{K}=\bigoplus_k\bigwedge^k\mathcal{K}$ 上，$a(f)$ 作用于外代数（反对称张量），满足 CAR。**唯一性定理（Jordan–Wigner）**：对可分 $\mathcal{K}$，CAR 代数有唯一的不可约表示（Fock 表示）——**CAR 代数是最「忠实」的 C\* 代数**，它只有一个真面。

**与 UHF 的关系**：取 $\mathcal{K}=\mathbb{C}^N$ 的基 $e_1,\dots,e_N$，$a_i=a(e_i)$。则 CAR 代数 $\cong M_{2^N}$；$N\to\infty$ 时给出 **UHF $2^\infty$ 代数**——第 17 篇的 $M_{2^\infty}$。所以 **CAR 代数 = UHF $2^\infty$**，是 AF 代数，K 理论 $K_0\cong\mathbb{Z}[1/2]$。

## 3 两极对照：有限与无限

把 $\mathcal{O}_n$ 与 CAR 代数并排，C\* 代数的「极性」一目了然：

**对比表（核心要点）**：

| 性质 | Cuntz 代数 $\mathcal{O}_n$ | CAR 代数（UHF $2^\infty$） |
| --- | --- | --- |
| 生成元关系 | 等距，$\sum S_iS_i^*=1$ | 反对易子，$\{a_i,a_j^*\}=\delta_{ij}$ |
| 结构 | **简单、纯无限** | **简单、有限（AF）** |
| 迹态 | 无 | 唯一迹态（Fock 真空态） |
| K 理论 | $K_0=\mathbb{Z}/(n-1)$，$K_1=0$ | $K_0=\mathbb{Z}[1/2]$，$K_1=0$ |
| 物理 | 玻色型 / 图 C\* 代数 | 费米子 / 量子场论 |

**辨析｜易错点：**「简单」不代表「小」或「平凡」。$\mathcal{O}_n$ 与 CAR 代数都简单，但一个是纯无限（投影无穷大、无迹），一个是有限（迹态存在、投影可数）。**「简单」只承诺「没有非平凡闭理想」，不承诺任何「有限性」**。把「简单」误读成「结构少」，会错过纯无限世界里最丰富的一层。

## 4 公式解析：$\sum_{i=1}^n S_iS_i^*=1$ 与 $\{a(f),a(g)^*\}=\langle f,g\rangle$

$$
\text{Cuntz:}\ \sum_{i=1}^n S_iS_i^* = 1, \qquad \text{CAR:}\ a(f)a(g)^*+a(g)^*a(f)=\langle f,g\rangle\,1
$$

- **第一步，看 Cuntz 的 $\sum S_iS_i^*=1$**：$S_iS_i^*$ 是向「$S_i$ 的像」的投影。等式说 $n$ 个像子空间**两两正交且直和等于全空间**（互不正交来自 $S_i^*S_j=0$ 当 $i\neq j$，铺满来自和等于 1）。它把「一个 $n$ 叉移位」定义为「每步有 $n$ 个选择」——$\mathcal{O}_n$ 因此是「$n$-进制树」上算子的代数。
- **第二步，看 CAR 的反对易子**：$a(f),a(g)^*$ 不交换，而是「反交换再加常数」。$f=g$ 时 $\{a,a^*\}=1$，$a^2=0$——**一个态至多一个费米子**。这就是 Pauli 不相容原理的代数外壳。
- **第三步，看两条关系为何殊途同归**：都是「用一条关系制造非交换性」，但 Cuntz 用**交换性缺陷**（$S_iS_j^*$ 的交叉项被 $\sum$ 压平），CAR 用**反对易子**。两条路分别长出「纯无限」与「有限」两种不同的简单代数——**关系的形式决定代数的性格**。
- **第四步，看它们如何进入物理**：Cuntz 关系编码「$n$ 个自由玻色子模」或「图上的平移」；CAR 关系直接是二次量子化的费米子场。两条公式分别是非交换几何与量子场论的「第一推动」。

## 5 用武之地：从量子场论到非交换几何

**Cuntz 代数的应用**：
- **图 C\* 代数与 Cuntz–Krieger 代数**：$\mathcal{O}_n$ 是最简单的 Cuntz–Krieger 代数（转移矩阵全 1 的图），其推广统治「子移位系统的 C\* 代数」——动力系统 + 算子代数在此交汇。<span class="marginnote">Cuntz–Krieger 代数 $\mathcal{O}_A$（由矩阵 $A$ 的 0/1 决定哪些 $S_iS_j^*$ 允许）把「图的边转移」编成 C\* 代数，其 K 理论恰好是转移矩阵的「指标」，把动力系统的拓扑熵与代数的 K 群连在一起。</span>
- **分类理论**：$\mathcal{O}_n$ 是**纯无限简单代数分类**的基准例（Kirchberg–Phillips 定理用 K 理论分类 $\mathcal{O}_n$ 的稳定同类），第 26 篇会再遇。

**CAR 代数的应用**：
- **量子场论**：费米场二次量子化 = CAR 代数的 Fock 表示；真空态 = 唯一迹态（由 CAR 唯一性，真空态自动唯一）。
- **统计力学与凝聚态**：自由费米气、超导 BCS 理论的 Bogoliubov 变换都是 CAR 代数的自同构；纠缠熵、量子信息里的费米子纠缠也用 CAR 语言。

**辨析｜易错点：**CAR 代数「有唯一表示」与 $\mathcal{O}_n$「表示极多」形成鲜明对比——但别以为 CAR 代数「简单到没内容」。它的结构全在**自同构群**（Bogoliubov 变换）与**子代数**（如单粒子哈密顿量的时间演化）里，表示唯一只是「舞台唯一」，舞台上的剧目依然无穷。

## 6 例：Cuntz 代数的 K 理论

$\mathcal{O}_n$ 的 K 理论计算是「由生成元与关系算 K 群」的样板。

**$K_0(\mathcal{O}_n)=\mathbb{Z}/(n-1)\mathbb{Z}$**：$[1]=(n-1)[S_1]$（因为 $\sum S_iS_i^*=1$ 且每个 $S_iS_i^*\sim 1$）。所以 $n$ 份「单位投影」加起来等于 1，模 $n-1$ 后给出有限群。

**$K_1(\mathcal{O}_n)=0$**：$\mathcal{O}_n$ 的酉元都「同伦平凡」——纯无限代数没有「环」结构。

**为什么 $n\ne m$ 时 $\mathcal{O}_n\not\cong\mathcal{O}_m$**：$K_0$ 不同（$\mathbb{Z}/(n-1)$ vs $\mathbb{Z}/(m-1)$）。Cuntz 1977 年的结论「$\mathcal{O}_n\cong\mathcal{O}_m$ ⟺ $n=m$」由 K 理论给出最干净证明。

**纯无限与 K 理论**：纯无限代数没有投影的「有限信息」，$K_0$ 全是「无穷投影的稳定类」——所以 $K_0$ 可以很小（有限群）。这与 AF 代数的 $K_0$（自由群）形成对照。

**一句话总结**：$\mathcal{O}_n$ 的 K 理论 = $\mathbb{Z}/(n-1)$——一个参数 $n$ 完全决定 K 群，而 K 群完全决定同构类。

## 7 延伸：CAR 代数的 Fock 表示

CAR 代数的「唯一表示」值得亲手构造一次。

**Fock 空间**：$\mathcal{F}(\mathcal{K})=\bigoplus_{k\ge0}\bigwedge^k\mathcal{K}$（反对称张量）。$\bigwedge^0\mathcal{K}=\mathbb{C}$（真空），$\bigwedge^1=\mathcal{K}$（单粒子），$\bigwedge^k$（$k$ 粒子，费米子不重复）。

**消灭与产生**：$a(f)$ 在外代数上作用（楔乘的伴随）；$a(f)^*$ 是楔乘。验证反对易子：$a(f)a(g)+a(g)a(f)=0$（反对称），$a(f)a(g)^*+a(g)^*a(f)=\langle f,g\rangle$。

**真空态**：$\xi_0\in\bigwedge^0$，$a(f)\xi_0=0$（真空被消灭算子杀掉）。真空态 = 唯一迹态——费米子场的基态。

**唯一性定理（Jordan–Wigner / Segal）**：可分 $\mathcal{K}$ 上 CAR 代数的不可约表示唯一（Fock）。「一个舞台演全场」——费米子代数没有别的表示。

**物理**：自由费米子场 = CAR 代数的 Fock 表示；$a(f)^*a(f)$ 是粒子数算子。Pauli 原理 $a(f)^2=0$ 在 Fock 表示里自动成立。

**一句话总结**：CAR 代数 = Fock 空间上的外代数算子；唯一性让「费米子只有一个世界」成为数学定理。

## 8 延伸：玻色型与费米型的对偶

$\mathcal{O}_n$ 与 CAR 代数分属两种统计，它们的对偶关系值得点破。

**玻色子 vs 费米子**：玻色子（光子、声子）允许多个占据同态，用交换关系/等距描述；费米子（电子）禁止重复占据，用反对易关系。$\mathcal{O}_n$ 的等距 $S_i$「每个方向都能叠」，CAR 的 $a(f)$「每个态至多一个」。

**CCR vs CAR**：正则对易关系（CCR，玻色）$[a(f),a(g)^*]=\langle f,g\rangle$ 与正则反对易关系（CAR，费米）$\{a(f),a(g)^*\}=\langle f,g\rangle$。只差一个符号，却长出截然不同的代数（无界 vs 有界、可积 vs 唯一）。

**交换统计定理**：物理上，自旋-统计定理（泡利）说粒子要么是玻色子要么是费米子——算子代数把它编码成「C\* 代数是 CCR 型还是 CAR 型」。

**$\mathcal{O}_n$ 与 CAR 的桥梁**：$C^*(\mathbb{F}_{n-1})$（自由群约化）与 $\mathcal{O}_n$ 密切相关；而 $C^*_r(\mathbb{F}_2)$ 的「自由概率」（Voiculescu）被视为「非交换玻色统计」。统计、概率、C\* 代数在这里合流。

**一句话总结**：$\mathcal{O}_n$ 与 CAR 是「交换律 vs 反对易律」的两极；从统计物理到自由概率，这一对极是量子世界的两条主轨。

## 9 小结

- **Cuntz 代数** $\mathcal{O}_n$：$n$ 个等距 + $\sum S_iS_i^*=1$；**简单、纯无限、可核**；无迹态；$K_0=\mathbb{Z}/(n-1)$。
- **CAR 代数**：反对易关系 $\{a(f),a(g)^*\}=\langle f,g\rangle$；Fock 表示唯一；$\cong$ UHF $2^\infty$，**简单、有限、有唯一迹态**。
- **两极**：Cuntz（纯无限）vs CAR（有限），都由单条关系生成，却性格迥异。
- **物理**：Cuntz ↔ 玻色/图动力系统，CAR ↔ 费米子/量子场论，Pauli 原理 = $a(f)^2=0$