---
title: 根系公理与 Weyl 群
date: 2026-08-11
---

# 根系公理与 Weyl 群

<div class="epigraph">
<p>把根系统从它生长的地方剥离出来，对称性才显出全貌。</p>
<footer>—— 威廉 · 基灵（Wilhelm Killing，意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 李代数与李群 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么把根抽象成公理

上一节我们从一个具体李代数的根空间分解里看到了「六边形」：$\mathfrak{sl}(3,\mathbb{C})$ 的六个根在 $H^*$ 的实平面上摆出一个完美对称的构型。这个构型里藏着比李代数本身更本质的组合信息——**根系统（root system）**。<span class="marginnote">Killing 在 1888–1890 年惊人地先于一切具体模型，仅从抽象的公理出发枚举了所有可能的复半单李代数。他把「几何 + 反射对称」当成了第一性原理，这正是本节公理化的历史由来。</span>

## 1 根系统的公理

设 $E$ 是有限维欧几里得空间，配内积 $\langle \cdot, \cdot \rangle$。$E$ 中的**根系统（root system）**是有限集 $\Phi$，满足四条公理：

1. **生成性**：$\Phi$ 张成 $E$，且 $0 \notin \Phi$。
2. **对称性**：若 $\alpha \in \Phi$，则 $-\alpha \in \Phi$，且 $\alpha$ 与 $-\alpha$ 是 $\Phi$ 中仅有的 $E$ 中与 $\alpha$ 共线的向量（精确说：$\Phi \cap \mathbb{R}\alpha = \{\pm\alpha\}$）。
3. **反射不变性**：对每个 $\alpha \in \Phi$，反射 $\sigma_\alpha$（$\sigma_\alpha(\beta) = \beta - 2\frac{\langle \beta, \alpha\rangle}{\langle \alpha, \alpha\rangle}\alpha$）把 $\Phi$ 映到 $\Phi$。
4. **整数性**：对所有 $\alpha, \beta \in \Phi$，$\langle \beta, \alpha\rangle / \langle \alpha, \alpha\rangle \in \mathbb{Z}$。<span class="marginnote">第四条是「Cartan 整数」的来源：上一节根空间性质里的 $\beta(h_\alpha) = 2\langle \beta,\alpha\rangle/\langle\alpha,\alpha\rangle$ 正是这个比值，它必为整数。这是最微妙也最强的一条公理。</span>

**Weyl 群（Weyl group）**：由全体反射 $\sigma_\alpha$（$\alpha \in \Phi$）生成的、$E$ 上正交变换的群，记作 $W$。<span class="marginnote">Weyl 群是根系统全部对称性的「骨架」：它作用在根集上、作用在权格上，甚至作用在表示论中「最高权」的集合上。它也是 Coxeter 群的第一个自然例子。</span>

## 2 简单根与正根

要让 $W$ 和 $\Phi$ 变得可枚举，我们需要选择一套「生成元」。选一个线性泛函 $f \in E^*$ 使 $f(\alpha) \neq 0$ 对所有 $\alpha$ 成立，把根分为**正根**与**负根**：

$$\Phi^+ = \{ \alpha \in \Phi \mid f(\alpha) > 0 \}, \qquad \Phi = \Phi^+ \sqcup (-\Phi^+)$$

**简单根（simple root）**：$\Phi^+$ 中不能被写成 $\Phi^+$ 中两个根之和的根，全体记为 $\Pi = \{\alpha_1, \dots, \alpha_\ell\}$。它们有若干关键性质：

- **基的性质**：$\Pi$ 是 $E$ 的一组基，且每个正根都能写成简单根的非负整数线性组合。
- **反射的生成**：$W$ 由简单根对应的反射 $s_i = \sigma_{\alpha_i}$ 生成。
- **关系式**：设 $m_{ij} = \langle \alpha_i, \alpha_j\rangle$ 的对偶整数 $a_{ij} = 2\langle\alpha_i,\alpha_j\rangle/\langle\alpha_j,\alpha_j\rangle$，则 $W$ 有 Coxeter 关系 $(s_i s_j)^{m_{ij}} = 1$，其中 $m_{ij} = 2, 3, 4, 6$（由角度 $\theta_{ij}$ 决定，$\cos\theta_{ij} = -\sqrt{m_{ij}}/2$... 精确地说 $\cos\theta_{ij} = \tfrac12\sqrt{\#\text{边}}$）。<span class="marginnote">简单根角度只有四种可能：$90°$（乘积阶 2）、$120°$（阶 3）、$135°$（阶 4）、$150°$（阶 6）。这四种角度就是第 9 篇 Dynkin 图连线方式的全部来源——组合有限性由此而来。</span>

**辨析｜易错点：** 正根与简单根的选择依赖线性泛函 $f$，看似「人择」。但结论是**不变量**：简单根反射生成的 $W$、根的个数、以及根集的整体构型都与 $f$ 无关。分类理论只依赖这些不变信息。

## 3 公式解析：反射公式的几何含义

根系统最核心的公式是反射：

$$\sigma_\alpha(\beta) = \beta - \left\langle \beta, \alpha^\vee \right\rangle \alpha, \qquad \alpha^\vee = \frac{2\alpha}{\langle \alpha, \alpha\rangle}$$

其中 $\alpha^\vee$ 是**余根（coroot）**。三步拆解：

- **第一步，看系数**：$\langle \beta, \alpha^\vee\rangle = 2\langle \beta, \alpha\rangle/\langle\alpha,\alpha\rangle$——它就是公理 4 的 Cartan 整数，必为整数。
- **第二步，看动作**：$\sigma_\alpha$ 把 $\beta$ 反射到关于「$\alpha$ 垂直平面」的镜像。系数为整数 ⟹ 根集在反射下「整齐地」排列，不产生小数偏移。
- **第三步，看意义**：$\sigma_\alpha(\beta) = \beta - n\alpha$（$n \in \mathbb{Z}$）意味着：从 $\beta$ 出发沿 $\alpha$ 方向往返，根的位置差恰是 $\alpha$ 的整数倍。这就是根格（root lattice）$\mathbb{Z}\Phi$ 的来历。

**核心要点**：反射 + 整数性是根系理论的两大引擎——反射给出对称群 $W$，整数性保证所有权落在格上，二者结合让「有限构型」的分类变成可能的（第 9 篇）。<span class="marginnote">物理学的根格观：在粒子物理中根对应规范玻色子的量子数；在晶体学中根格对应晶体倒格矢的对称性。同一套数学在不同学科反复出现。</span>

## 4 例子：$A_2$ 根系统的 Weyl 群

回到 $\mathfrak{sl}(3,\mathbb{C})$ 的六边形根系统（$A_2$）。取简单根 $\alpha_1 = \epsilon_1 - \epsilon_2$，$\alpha_2 = \epsilon_2 - \epsilon_3$（夹角 $120°$）。

**正根**：$\alpha_1, \alpha_2, \alpha_1 + \alpha_2$（三个）；负根为其相反数。
- **Weyl 群**：由反射 $s_1, s_2$ 生成，$s_1 s_2$ 的阶为 3（$120°$ 反射乘积是 $60°$ 旋转，三次回到恒等）。$W \cong S_3$，六阶，与根数相同（根对反射的轨道大小为 1 或 3，$W$ 传递地作用在根上）。
- **根格**：$\mathbb{Z}\Phi = \{ m\alpha_1 + n\alpha_2 \}$ 是六角格，即石墨烯、蜂窝晶格的数学骨架。<span class="marginnote">蜂窝格正是 $A_2$ 根格的几何实现——这是数学格点与材料科学的一次直接相遇，也解释了为何石墨烯的六角对称性如此普遍。</span>

## 5 小结

- **根系统**：$E$ 上有限集 $\Phi$，满足生成性、对称性、反射不变、整数性四公理。
- **Weyl 群** $W$：由根反射生成的有限正交群，是根系统的对称群；由简单根反射 $s_i$ 生成，满足 Coxeter 关系。
- **简单根** $\Pi$：正根中不可分解的元素，构成 $E$ 的基；所有正根是其非负整数组合。
- 根对之间的夹角只有 $90°/120°/135°/150°$ 四种，这是组合有限性的来源。
- **余根** $\alpha^\vee$ 与 Cartan 整数把反射改写成整数系数的平移，根格由此而来。
- $A_2$ 的 Weyl 群是 $S_3$，六阶，作用在六边形根上传递。

在下一节，我们将问：所有可能的根系统有多少种？答案指向人类数学中最美的分类定理之一——**根系分类与 Dynkin 图**。
