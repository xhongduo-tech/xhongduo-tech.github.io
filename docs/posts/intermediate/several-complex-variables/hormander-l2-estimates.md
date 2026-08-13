---
title: Hörmander L² 估计与加权解存在性定理
date: 2026-08-07
---

# Hörmander L² 估计与加权解存在性定理

<div class="epigraph">
<p>L² 估计是一把双刃剑：它既给出解的存在性，也给出解的大小。</p>
<footer>—— 拉尔斯 · 赫尔曼德（Lars Hörmander），《多复变分析引论》（An Introduction to Complex Analysis in Several Variables）</footer>
</div>

<div class="article-byline">
<p>第二级 · 多复变函数论 ｜ Hörmander 第4章；Krantz 第3章 ｜ 2026-08-07</p>
</div>

## 为什么从 Hörmander L² 估计开始

前一篇铺好了加权 L² 空间，证明了 $\bar\partial$ 的闭性。本篇完成最关键的一跳：**证明伪凸域上 $\bar\partial u = g$ 恒有解，并给出解的 L² 范数控制**。这就是 **Hörmander L² 估计**（1965），它把多复变从「定性理论」推向「定量理论」，并成为整个领域最通用的工具——从 Levi 问题的现代证明，到代数几何的乘理想层（Nadel），再到复几何的正性理论，全都建立在这条估计上。<span class="marginnote">Hörmander 这篇 1965 年论文（"L² estimates and existence theorems for the $\bar\partial$ operator"）被公认为现代多复变的奠基之作。它首次证明了：<strong>在伪凸域上，$\bar\partial$ 方程存在满足显式范数界的光滑解</strong>——不依赖任何边界光滑性，只要求权函数 psh。</span>

## 1 定理陈述：加权解存在性

**Hörmander 定理**：设 $D \subset \mathbb{C}^n$ 开集，$\varphi \in \mathrm{psh}(D)$。则对任意 $g \in L^2_{(0,1)}(D, \mathrm{loc})$ 满足 $\bar\partial g = 0$ 与

$$
\int_D |g|^2 e^{-\varphi} \, d\lambda \lt  \infty
$$

存在 $u \in L^2(D, \mathrm{loc})$ 使 $\bar\partial u = g$，且满足**估计**：

$$
\int_D |u|^2 e^{-\varphi} \, d\lambda \;\leq\; \int_D |g|^2 e^{-\varphi} \, d\lambda
$$

**关键是：$D$ 甚至不需要伪凸？** 事实上定理要求权 $\varphi$ 是 psh 且 $D$ 上 $\partial\bar\partial\varphi$ 正定（强 psh）才保证解的唯一性意义。Hörmander 的完整版本要求 $\varphi$ 在 $D$ 上「强 psh」（$\partial\bar\partial\varphi \geq \omega$ 某个正形式）。<span class="marginnote">当 $D$ 伪凸时，可取 $\varphi$ 为强 psh 穷竭函数（Levi 问题证明中的 $\varphi = -\log(-\rho)+C|z|^2$），于是 $H^{0,1}_{\bar\partial}(D) = 0$——这就是第 3 篇组消失定理的 L² 证明。所以<strong>Levi 问题的现代证明路径是：伪凸 ⟹ 有强 psh 穷竭函数 ⟹ Hörmander 定理 ⟹ $\bar\partial$ 可解 ⟹ 全纯域</strong>。</span>

## 2 证明的三大支柱

**支柱 1：闭值域与对偶**。$\bar\partial$ 像闭（上一节），故 $g = \bar\partial u$ 有解 ⟺ $g \perp \ker \bar\partial^*_\varphi$。而「$g \perp \ker\bar\partial^*$」由「$\bar\partial g = 0$ 且 $g$ 满足可积条件」推出（闭性 + 对偶论证）。

**支柱 2：基本恒等式 / 曲率估计**。对 $u \in C_c^\infty$，有

$$
\|\bar\partial u\|^2_\varphi + \|\bar\partial^*_\varphi u\|^2_\varphi \;\geq\; \int_D \langle \partial\bar\partial\varphi, u \wedge \bar u \rangle \, e^{-\varphi}
$$

当 $\varphi$ 强 psh（$\partial\bar\partial\varphi \geq \omega$）时右端 $\geq \int |u|^2 e^{-\varphi}$，得到**先验估计**：

$$
\|u\|^2_\varphi \;\leq\; \|\bar\partial u\|^2_\varphi + \|\bar\partial^*_\varphi u\|^2_\varphi
$$

**支柱 3：从先验估计到存在性**。先验估计 + Hahn–Banach/Riesz 表示定理：定义线性泛函 $T(\bar\partial^*_\varphi v) = \langle g, v\rangle_\varphi$，由估计知其有界连续，延拓到整个空间，Riesz 表示给出 $u$ 满足 $\bar\partial u = g$ 且范数界成立。<span class="marginnote">支柱 3 是「先验估计 ⇒ 存在性」的标准手法，也出现在椭圆方程理论（如 $-\Delta u = f$ 的 Lax–Milgram 论证）里。它的优雅在于：<strong>不构造解，而是证明「解的存在」由范数界的对偶性必然推出</strong>。这是泛函分析在 PDE 中的最典型应用。</span>

## 3 为什么 psh 权是「刚刚好」的条件

把权函数的角色再深化一层。**若 $\varphi$ 不是 psh**，定理的结论可能崩坏：

$\varphi$ 在某方向复 Hessian 为负 ⟹ 基本恒等式右端出现负项 ⟹ 先验估计失效 ⟹ 解可能不存在或不在 $L^2_\varphi$ 中。
反例：$D = \mathbb{C}$，$\varphi = |z|^2/2$（psh），$\bar\partial u = g$ 有解；若取 $\varphi = -|z|^2/2$（不是 psh），则权 $e^{-(-|z|^2/2)} = e^{+|z|^2/2}$ 在无穷处增长，$g = \bar z$（$\bar\partial(\bar z)=0$ 吗？否，$\bar\partial \bar z = d\bar z \neq 0$）……精确反例需小心构造，但精神明确：**权的凹性破坏估计**。<span class="marginnote">这个「psh ⟺ 可解」的等价关系，是 Hörmander 理论的精髓：<strong>权函数的 psh 性既是充分的，也几乎必要</strong>。它把「分析可解性」与「复几何凸性」焊死在一起——再次呼应 psh 函数的多复变中心地位。</span>

## 4 公式解析：核心 L² 估计

$$
\boxed{\; \int_D |u|^2 e^{-\varphi} \leq \int_D |g|^2 e^{-\varphi}, \qquad \bar\partial u = g \;}
$$

- **第一步，量纲与意义**：这是「解的 $L^2_\varphi$ 范数被数据的 $L^2_\varphi$ 范数控制」——**常数 1 的范数估计**。它说：解的「能量」不超过数据的「能量」，即 $\bar\partial$ 的「逆」是压缩的。常数可取任意 $\geq 0$，Hörmander 版本给出 1，是最优的。
- **第二步，从先验估计到它**：先验估计 $\|u\|_\varphi \leq \|\bar\partial u\|_\varphi + \|\bar\partial^* u\|_\varphi$ 给出的是「和解 $\bar\partial^* v$ 的组合控制」；通过对偶（支柱 3）转成「对 $\bar\partial u = g$ 的解范数控制」。中间用「$g \perp \ker \bar\partial^*$ 的加权正交分解」。
- **第三步，为什么常数是 1**：当 $\partial\bar\partial\varphi \geq 0$（psh）时基本恒等式中没有额外正项，先验估计的常数就是 1。若 $\partial\bar\partial\varphi \geq \varepsilon \omega$（严格 psh），常数可 < 1，解更「收缩」——**权越凸，解越小**。

## 5 辨析与延伸：Hörmander 估计的五个要点

**辨析 1：常数 1 是「免费」但非「平凡」**。$\|u\|_\varphi \leq \|g\|_\varphi$ 中常数 1 来自 psh 权的「无额外正项」。若权严格 psh（$\partial\bar\partial\varphi \geq \varepsilon\omega$），常数可以 < 1，解更收缩。**常数的大小反映「权有多凸」——凸性越强，解越被压缩**。<span class="marginnote">这个观察在代数几何中开花结果：Nadel 的乘理想层理论利用「$\varphi$ 的奇性程度」构造理想的奇性描述，正性与消失定理的强度都由 $\varphi$ 的凸性（奇性）编码。</span>

**辨析 2：$L^2$ 估计 vs 逐点估计**。Hörmander 方法给出的是**积分**（$L^2$）估计，不直接给逐点界。但通过加权技巧可以反过来推逐点界：取 $\varphi$ 在指定点「挖一个峰」，解的 $L^2$ 范数界转化为该点值的界。**$L^2$ 估计 + 权函数技巧 = 任意点估计**——这是后文构造「爆掉」函数的惯用手法。

**辨析 3：存在性 ≠ 构造性**。Hörmander 定理给出解的存在与范数界，但**不显式构造**解（Riesz 表示给出抽象存在）。对应用来说这通常够用；需要显式解时才用积分表示（Cauchy 型核）。**「存在 + 范数界」是分析的黄金标准，无需显式公式**。

**辨析 4：为什么这个定理「通吃」**。$\bar\partial u = g$ 是线性方程；其解空间结构（存在性、唯一性模 $\ker\bar\partial$）由范数估计完全控制。把任意「构造全纯函数」问题化为 $\bar\partial$ 方程，再用 Hörmander 估计，就得到一个万能构造法——这正是现代多复变的「工厂模式」。

**误区清单**：

- **误区 1**：以为「L² 估计给出显式解」。
  正解：给出存在 + 范数界，不显式构造。
- **误区 2**：以为「常数 1 是凑巧」。
  正解：来自 psh 权的基本恒等式，有几何意义。
- **误区 3**：以为「强 psh 权才可解」。
  正解：psh 即可（存在性）；强 psh 提供更好的正则性。
- **误区 4**：以为「Hörmander 方法只解决 Levi 问题」。
  正解：它是通吃工具，Cousin 问题、延拓、构造全纯函数都靠它。

**术语表**：

| 中文 | 英文 | 说明 |
| --- | --- | --- |
| 先验估计 | a priori estimate | 假设解存在推范数界 |
| 闭值域 | closed range | 值域闭定理 |
| 对偶论证 | duality argument | 泛函延拓 |
| Riesz 表示 | Riesz representation | 内积 → 线性泛函 |
| 强 psh | strongly psh | 复 Hessian 正定 |
| 乘理想层 | multiplier ideal sheaf | Nadel 构造 |

## 6 历史注记与知识树

**历史**：Hörmander 1965 年论文同时给出 $L^2$ 估计与存在性定理，是近代多复变的奠基文献之一。Andreotti–Vesentini（1965）独立得到类似结果。此后：Nadel（1989）把权函数方法引入代数几何（乘理想层）；Demailly 系统发展「解析 Monge–Ampère 方程 + L² 方法」；Ohsawa–Takegoshi（1987）证明 $L^2$ 延拓定理，至今仍是几何不等式的重要来源。

**知识树**：

- 向后：加权 L² 空间与闭性（本组第 19 篇）、psh 函数（第 1 组）。
- 向前：强伪凸域正则性（本组第 21 篇）——从存在到光滑。
- 横向：PDE 的椭圆理论（第二级《偏微分方程》）——先验估计 ⟹ 存在性的标准手法。

**一句话记忆**：Hörmander 估计 = 伪凸域上 $\bar\partial$ 方程有解且 $\|u\| \leq \|g\|$；三大支柱 = 闭值域 + 曲率恒等式 + 对偶论证。

## 7 小结

- **Hörmander 定理**：伪凸域（强 psh 权）上 $\bar\partial u = g$ 有解，且 $\|u\|_\varphi \leq \|g\|_\varphi$。
- **三大支柱**：闭值域 + 对偶、基本恒等式（曲率估计）、先验估计 ⇒ 存在性。
- **psh 权是「刚刚好」**：权的凸性 ⟺ 可解性，把分析可解性与复几何凸性焊在一起。
- **推论**：$H^{0,q}_{\bar\partial}(D)=0$（全纯凸域）的 L² 证明——Levi 问题现代路径。

在下一节，我们研究解的光滑性：**强伪凸域上 $\bar\partial$ 方程的正则性**——解从 $L^2$ 升级到 $C^\infty$