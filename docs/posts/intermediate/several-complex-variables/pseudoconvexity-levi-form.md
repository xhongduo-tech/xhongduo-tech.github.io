---
title: 伪凸域与 Levi 形式
date: 2026-08-07
---

# 伪凸域与 Levi 形式

<div class="epigraph">
<p>伪凸性是多复变对「凸性」的最终裁决：它既非拓扑的，也非代数的，而是属于函数论本身。</p>
<footer>—— 仿 埃里希 · 卡勒（Erich Kähler）遗风，转引自 Krantz《多复变函数论》</footer>
</div>

<div class="article-byline">
<p>第二级 · 多复变函数论 ｜ Hörmander 第2章；史济怀 第3章 ｜ 2026-08-07</p>
</div>

## 为什么从伪凸域开始

第 1 篇组确立了三个互相等价的「好区域」面孔：**全纯域**（存在不可延拓函数）、**全纯凸域**（凸包不溢出）、以及（预告的）**伪凸域**。前两个是分析条件，用起来往往不便；而**伪凸性**是一个纯粹的**局部几何条件**——它只看区域边界的曲率，就像实分析里「凸集」只看边界是往外鼓还是往里凹。把「好区域」化简到「边界曲率条件」，正是 Levi 问题的全部意义。<span class="marginnote">本组的核心定理（Levi 问题的解答，第 10 篇）将宣告：对 $\mathbb{C}^n$ 中具有光滑边界的区域，<strong>全纯域 $\iff$ 伪凸域</strong>。今天这篇先定义伪凸性与 Levi 形式，把曲率这个「引擎」造出来。</span>

## 1 实凸性的回顾：切空间与法向

先回到实分析找直觉。区域 $D = \{ \rho \lt  0 \}$，其中 $\rho: \mathbb{R}^N \to \mathbb{R}$ 是定义函数（$\rho = 0$ 为边界，$\nabla \rho \neq 0$ 在边界上）。$D$ 在边界点 $p$ 处**凸**，直观上要求 $D$ 落在切平面的「一侧」，二阶条件就是 Hessian 沿切方向半正定。

多复变中，边界 $\partial D = \{ \rho = 0 \}$ 是实超曲面（$\mathbb{R}^{2n-1}$ 维）。但**复结构**带来全新的切空间分解：切空间 $T_p \partial D$ 里有一个复子空间 $T_p^{\mathbb{C}} \partial D$（**复切空间 / CR 切空间**），由满足 $\partial \rho(p) \cdot v = 0$ 的复切向量 $v$ 张成。伪凸性关心的是：**曲率在复切方向上是否非负**——这是比「实凸性」更弱也更重要的一阶信息。<span class="marginnote">为什么只看复切方向？因为复切方向是「全纯延拓能自由移动」的方向：沿复切方向，函数信息通过 Cauchy 积分「自动传播」。非复切方向（法向）则对应真正的边界约束。Levi 形式的几何直觉：<strong>复切方向上的凹坑无法被填平，就会导致延拓</strong>。</span>

## 2 Levi 形式：定义

设 $D \subset \mathbb{C}^n$ 有 $C^2$ 定义函数 $\rho$（$\partial D = \{\rho = 0\}$，$\nabla \rho \neq 0$），$p \in \partial D$。$D$ 在 $p$ 处的 **Levi 形式（Levi form）**是限制在复切空间上的复 Hessian 型：

$$
\mathcal L_p(w) = \sum_{j,k=1}^n \frac{\partial^2 \rho}{\partial z_j \partial \bar z_k}(p)\, w_j \bar w_k, \qquad w \in T_p^{\mathbb{C}}\partial D
$$

其中 $T_p^{\mathbb{C}}\partial D = \{ w \in \mathbb{C}^n : \sum_j \partial \rho/\partial z_j(p)\, w_j = 0 \}$。<span class="marginnote">Levi 形式依赖定义函数 $\rho$ 的选择，但<strong>在复切方向上的符号</strong>（正定/半正定/不定）不依赖——换定义函数 $\rho \to e^{\psi}\rho$ 只是乘一个正因子。所以「Levi 形式半正定」是一个几何不变量。</span>

## 3 伪凸域：三种等价定义

**定义（光滑边界情形）**：$C^2$ 边界的区域 $D = \{\rho \lt  0\}$ 称为**伪凸（pseudoconvex）**的，若在边界的每一点 $p$，Levi 形式在复切方向上半正定：

$$
\mathcal L_p(w) \geq 0, \qquad \forall w \in T_p^{\mathbb{C}}\partial D
$$

若处处**正定**，称为**强伪凸（strongly pseudoconvex）**。

但伪凸性有更深刻、不依赖边界的整体定义，且彼此等价：

1. **Levi 形式版**：如上（需光滑边界）。
2. **psh 定义函数版**：存在 $D$ 的连续 psh 定义函数 $\rho$（在边界邻域 $\rho \lt  0$ 内 psh，边界上 $\rho = 0$），且随 $z \to \partial D$ 有 $\rho(z) \to 0$。
3. **全纯凸版**：$D$ 全纯凸（Cartan–Thullen）。

2 与 3 的等价在一般（无光滑边界）情形也成立，这使「伪凸」成为**不依赖边界光滑性**的纯粹概念。<span class="marginnote">重要区分：<strong>强伪凸</strong>要求 Levi 形式正定，是「严格」版本；<strong>伪凸</strong>只要求半正定。两者对 $\bar\partial$ 方程的正则性（第 4 篇组）差别巨大——强伪凸域上有一整套局部理论，一般伪凸域则要依靠整体 L² 方法。</span>

## 4 公式解析：Levi 形式的「复切向投影」

$$
\mathcal L_p(w) = \sum_{j,k} \frac{\partial^2 \rho}{\partial z_j \partial \bar z_k}(p) w_j \bar w_k, \qquad w \in T_p^{\mathbb{C}}\partial D
$$

- **第一步，认出矩阵**：$(\partial^2\rho/\partial z_j\partial\bar z_k)$ 正是上一节 psh 函数判据中的复 Hessian。所以 Levi 形式 = **限制到复切空间的复 Hessian**。
- **第二步，理解复切条件**：$w \in T_p^{\mathbb{C}}\partial D$ 意味着 $\partial \rho(p)\cdot w = \sum \partial\rho/\partial z_j(p) w_j = 0$——$w$ 是「零阶方向导数沿边界消失」的复方向。实切方向（满足 $\mathrm{Re}$ 条件）不一定复切。
- **第三步，为什么符号不变**：换定义函数 $\rho \mapsto e^{\psi}\rho$，复 Hessian 变为 $e^{\psi}(\partial\bar\partial\rho + \rho\partial\bar\partial\psi + \partial\rho\wedge\bar\partial\psi + \bar\partial\psi\wedge\partial\rho)$；沿边界 $\rho=0$ 且限制在复切方向（$\partial\rho$ 作用为零），只剩 $e^{\psi}\partial\bar\partial\rho$——正因子的半正定性不变。这就是「符号是不变量」的严格来源。

## 5 辨析与延伸：Levi 形式的五个易错点

**辨析 1：Levi 形式 ≠ 实 Hessian**。Levi 形式是复 Hessian $\partial\bar\partial\rho$ 在复切方向上的限制；实 Hessian 作用在实切方向。两者完全不同。一个边界可以实凸而 Levi 不定，或实凹而 Levi 正定。<span class="marginnote">例子：$\rho = |z_1|^2 - |z_2|^2$ 的边界点处，实 Hessian 有正负特征值，Levi 形式（在复切方向）也受 $z_1,z_2$ 两个方向的贡献控制——符号取决于具体切方向。不要用实直觉猜 Levi 符号。</span>

**辨析 2：Levi 形式依赖定义函数，但符号不依赖**。换定义函数 $\rho \to e^{\psi}\rho$ 会改变 Levi 形式的值，但**在复切方向上的符号**不变。所以「半正定」是几何不变量，而「Levi 形式的具体数值」不是。做题时若发现 Levi 形式算出的值变了，先检查是不是换了定义函数。

**辨析 3：伪凸 vs 强伪凸**。伪凸要求 Levi 形式半正定（允许零方向）；强伪凸要求正定（无零方向）。两者在正则性理论中差别巨大：强伪凸有次椭圆估计（第 21 篇），一般伪凸只有 L² 存在性。**「强」不是「更凸一点」，而是「正则性可用」的开关**。

**辨析 4：伪凸是单边条件**。伪凸只禁止 Levi 形式取负值，允许任意退化。这与实凸性（要求所有方向）形成对比——伪凸是「半」凸，是复结构特有的「单边」条件。

**辨析 5：边界不光滑时怎么办**。Levi 形式需要 $C^2$ 边界；一般区域用「psh 定义函数」或「局部伪凸」定义。**Levi 形式是光滑边界的特权**，不是伪凸性的普遍定义。

**速查表**：

| 对象 | 定义域 | 条件 | 意义 |
| --- | --- | --- | --- |
| Levi 形式 | 复切空间 | $\sum \rho_{j\bar k} w_j\bar w_k$ | 边界复曲率 |
| 伪凸 | 边界各点 | $\mathcal L \geq 0$ | 好区域候选 |
| 强伪凸 | 边界各点 | $\mathcal L > 0$ | 正则性保证 |
| 凹 | 边界各点 | $\mathcal L \leq 0$ | 坏区域（可延拓） |

## 6 术语对照表

| 中文 | 英文 | 一句话说明 |
| --- | --- | --- |
| 伪凸域 | pseudoconvex domain | Levi 形式半正定的区域 |
| 强伪凸域 | strongly pseudoconvex domain | Levi 形式正定的区域 |
| Levi 形式 | Levi form | 复 Hessian 在复切方向的限制 |
| 定义函数 | defining function | 刻画边界的方程 $\rho=0$ |
| 复切空间 | CR tangent space | 被复结构保持的切方向 |
| 复 Hessian | complex Hessian | $\partial\bar\partial$ 的二阶项 |
| 全纯凸 | holomorphically convex | 全纯凸包不溢出 |
| 多重次调和 | plurisubharmonic | 沿复直线次调和 |
| 边界值 | boundary value | 函数在边界上的限制 |
| 特征边界 | distinguished boundary | 多圆柱的 $T^n$ 骨架 |
| 弱 Levi 条件 | weak Levi condition | Levi 形式半正定 |
| 局部伪凸 | locally pseudoconvex | 每点邻域伪凸 |
| psh 定义函数 | psh defining function | 伪凸的 psh 刻画 |
| 凹域 | concave domain | Levi 形式半负定 |
| 零方向 | null direction | Levi 形式为零的切方向 |

**记忆口诀**：Levi 形式只看**复切方向**；伪凸是**单边**条件（只禁负方向）；「强」是正则性的开关而非程度的量词。

**补充辨析**：Levi 形式与 psh 的关系——一个 $C^2$ 函数 $\rho$ 在 $D$