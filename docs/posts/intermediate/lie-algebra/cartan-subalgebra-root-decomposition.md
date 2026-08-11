---
title: Cartan 子代数与根空间分解
date: 2026-08-11
---

# Cartan 子代数与根空间分解

<div class="epigraph">
<p>对称性的内在格点，正是它所有表现的蓝图。</p>
<footer>—— 埃利 · 嘉当（Élie Cartan，意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 李代数与李群 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么需要一个对角化框架

$\mathfrak{sl}(2,\mathbb{C})$ 的表示论给我们的启示是：**对角的 $h$ 是尺子，升降的 $e, f$ 是梯子**。对一般的半单李代数，我们要把这条经验推广——需要一个足够大的「尺子子代数」（Cartan 子代数），把整个代数分解为它的特征空间（根空间）的直和。这一步称为**根空间分解（root space decomposition）**，它把李代数这个「代数对象」翻译成一个「格上的几何对象」（根系统），后者正是第 8、9 篇分类理论的输入。<span class="marginnote">类比线性代数：一个算子可对角化 ⟺ 全空间分解为特征空间的直和。Cartan 子代数是把这一思想推广到「一族交换算子同时对角化」。</span>

## 1 定义与存在性

**Cartan 子代数（Cartan subalgebra）**：$L$ 的**极大交换**子代数 $H$，满足：$H$ 的每个元素在 $\operatorname{ad}H$ 下都对角化（即 $H$ 由 $\operatorname{ad}$-对角化的元素张成），且没有更大的包含它的交换子代数。

**存在性**：任意有限维李代数都有 Cartan 子代数；对半单李代数，Cartan 子代数在 $\operatorname{ad}H$ 下的作用可同时对角化。<span class="marginnote">存在性可用「正则元素」构造：取 $L$ 中 $x$ 使 $\operatorname{ad}x$ 的零空间极小，则该零空间就是 Cartan 子代数。这是 Humphreys §8 的标准构造。</span>

**根空间分解（root space decomposition）**：设 $H$ 是 $L$ 的 Cartan 子代数，则

$$L = H \oplus \bigoplus_{\alpha \in \Phi} L_\alpha, \qquad L_\alpha = \{ x \in L \mid [h, x] = \alpha(h) x \ \forall h \in H \}$$

其中 $\alpha: H \to \mathbb{C}$ 是非零线性泛函（**根（root）**），$L_\alpha$ 是根空间，$\Phi$ 是根的集合。<span class="marginnote">每个根 $\alpha$ 是一个线性泛函——它把 Cartan 子代数的元素映射为标量特征值。第 6 篇的 $h$ 相当于 $H = \mathbb{C}h$，根只有 $\alpha(h)=2$ 与 $-\alpha$，根空间 $\mathbb{C}e, \mathbb{C}f$。</span>

## 2 根空间的基本性质

根空间分解的威力来自一系列结构性质：

**根的配对**：$\dim L_\alpha = 1$（半单情形），且 $[\ L_\alpha, L_\beta\ ] \subseteq L_{\alpha + \beta}$——括号把根相加。
- **对根**：若 $\alpha$ 是根，则 $-\alpha$ 也是根，且 $L_\alpha \oplus L_{-\alpha} \oplus [L_\alpha, L_{-\alpha}]$ 构成一个 $\mathfrak{sl}(2)$ 三元组。<span class="marginnote">这正是第 6 篇的承诺兑现：每对根 $\pm\alpha$ 给出一个标准三元组 $\{e_\alpha, e_{-\alpha}, h_\alpha\}$，其中 $h_\alpha \in H$ 满足 $\alpha(h_\alpha) = 2$。全体半单结构被切成许多个 $\mathfrak{sl}(2)$。</span>
**整数性**：$\beta(h_\alpha)$ 对任意根 $\beta$ 取整数值——这是后面根系公理中「整数性条件」的来源。
- **根的零性**：根空间在 Killing 型下正交：$\kappa(L_\alpha, L_\beta) = 0$ 除非 $\alpha + \beta = 0$。<span class="marginnote">这使 Killing 型在根空间上几乎「对角」——它是第 8 篇构造 Weyl 群反射的几何基础。</span>

**辨析｜易错点：** Cartan 子代数并不唯一，但**维数固定**（称为秩，rank）。根的个数 $\dim L - \operatorname{rank} L$ 也固定。初学者常误以为「取哪个 Cartan 子代数都行但根不同」——事实是根的**构型**在同构意义下唯一，根系统的分类因此不依赖选择。

## 3 实例：$\mathfrak{sl}(3,\mathbb{C})$ 的根

我们计算最经典的非平凡例子 $\mathfrak{sl}(3,\mathbb{C})$（迹零 $3\times3$ 矩阵），其 Cartan 子代数取对角矩阵：

$$H = \{ \operatorname{diag}(a_1, a_2, a_3) \mid a_1 + a_2 + a_3 = 0 \}$$

定义线性泛函 $\epsilon_i(\operatorname{diag}(a_1,a_2,a_3)) = a_i$。则六个根空间对应六个矩阵单位 $E_{ij}$（$i \neq j$）：

$$\alpha_{ij} = \epsilon_i - \epsilon_j, \qquad [h, E_{ij}] = (\epsilon_i - \epsilon_j)(h)\, E_{ij}$$

**核心事实**：根集为 $\Phi = \{ \pm(\epsilon_1 - \epsilon_2),\ \pm(\epsilon_2 - \epsilon_3),\ \pm(\epsilon_1 - \epsilon_3) \}$，共 6 个根，构成平面上的**六边形**（若把 $\epsilon_i$ 视为 $\mathbb{R}^3$ 的坐标，约束 $\sum a_i = 0$ 使它们在二维平面内）。秩为 2，根个数为 6，正好等于 $\dim \mathfrak{sl}(3) - 2 = 8 - 2 = 6$。<span class="marginnote">这就是 $A_2$ 根系：六边形。它是第 9 篇 Dynkin 图分类中第一个非平凡成员。八矩阵单位中三个对角元（$H$）加六个非对角元（根空间）正好凑出 $\dim = 8$。</span>

![$\mathfrak{sl}(3,\mathbb{C})$ 的根系统：六个根构成正六边形，正根用实心点、负根用空心点标注](/images/lie-algebra/cartan-subalgebra-root-decomposition-1.svg)

## 4 公式解析：根向量与 Cartan 子代数的对偶

根空间分解中核心公式是：

$$[h, e_\alpha] = \alpha(h)\, e_\alpha$$

这条式子已经熟悉，但根 $\alpha$ 是**泛函**而非数——我们把它彻底拆开：

- **第一步，分清层次**：$h \in H$ 是代数元素，$\alpha \in H^*$ 是它的对偶泛函，$e_\alpha \in L_\alpha$ 是根空间向量。三者分属三个空间。
- **第二步，理解配对**：$\alpha(h)$ 是泛函作用在元素上得到的**标量**。当 $H = \mathbb{C}h_0$ 一维时，$\alpha$ 就退化为第 6 篇的数 $\pm 2$；多维度时它是一族标量，对应 $h$ 的各个分量。
- **第三步，读懂分解**：$L_\alpha$ 是 $H$ 的「联合特征空间」——一个向量同时是 $H$ 中所有元素的特征向量，特征值由 $\alpha$ 统一编码。这就是「$H$ 同时对角化」的精确含义。

**核心要点**：根空间分解把半单李代数翻译为「以 $H^*$ 为底座、以根集为格点的几何对象」。这一翻译的有效性建立在三条性质上：根空间一维、括号相加根、$\pm\alpha$ 配对成 $\mathfrak{sl}(2)$ 三元组。

## 5 小结

- **Cartan 子代数** $H$ 是极大交换且 $\operatorname{ad}$-对角化的子代数，维数 = **秩**；半单情形存在性由正则元素保证。
- **根空间分解** $L = H \oplus \bigoplus_\alpha L_\alpha$：$L_\alpha$ 是 $H$ 的联合特征空间，$\alpha \in H^*$ 为根。
- 关键结构：根一维、$[L_\alpha, L_\beta] \subseteq L_{\alpha+\beta}$、$\pm\alpha$ 给出 $\mathfrak{sl}(2)$ 三元组、$\beta(h_\alpha) \in \mathbb{Z}$。
- $\mathfrak{sl}(3,\mathbb{C})$ 的根集是平面上的**六边形**（$A_2$ 根系），秩 2、6 个根。
- 根系统把李代数翻译成几何格点，为第 8 篇的**根系公理**与第 9 篇的 **Dynkin 图分类**提供原料。

在下一节，我们将抽取根空间分解的「几何精华」，给出**根系公理**的抽象定义，并认识控制对称的 **Weyl 群**。
