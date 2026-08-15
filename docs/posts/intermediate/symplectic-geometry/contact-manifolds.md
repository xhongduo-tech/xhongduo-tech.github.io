---
title: 接触流形与辛化
date: 2026-08-07
---

# 接触流形与辛化

<div class="epigraph">
<p>接触几何是辛几何的奇维表亲：它描述「带方向的光线如何填满空间」。</p>
<footer>—— 海因里希 · 格吕贝尔 精神续写（接触几何教学传统）</footer>
</div>

<div class="article-byline">
<p>第二级 · 辛几何 ｜ McDuff & Salamon 第4章；Cannas 第5章 ｜ 2026-08-07</p>
</div>

## 为什么从接触流形开始

辛流形永远是偶维的，但物理世界里有大量「奇维」的几何：光学里的光前波阵面、热力学里的状态曲面、力学里的**等能面**（$H = \text{const}$）。奇维流形上放不下辛形式，但能放一个更弱的对象——**接触结构**：一个处处「最大不可积」的超平面分布。它与辛几何的关系不是巧合：**接触流形通过辛化（symplectization）变成辛流形**，反过来，辛流形的许多边界是接触流形。这一篇建立这个桥梁，并且预告后面伪全纯曲线理论里「把接触流形填成辛流形」的问题（填充问题，第3篇）。<span class="marginnote">等能面为什么是接触的？在 $2n$ 维辛流形上，$H$ 的水平集是 $2n-1$ 维超曲面，其上由 $\omega$ 诱导的自然超平面分布 $\xi = \ker(\omega|_H)$ 恰是接触结构。力学与几何在这里再次握手。</span>

## 1 接触结构的定义

**接触形式（contact form）**：$(2n+1)$ 维流形 $Y$ 上的 1-形式 $\alpha$，满足

$$
\alpha \wedge (d\alpha)^n \neq 0 \quad \text{处处非零}
$$

这是**体积形式条件**：$\alpha \wedge (d\alpha)^n$ 是 $Y$ 上的处处非零体积形式，故 $Y$ 可定向且 $\alpha$ 在逐点意义上「最大不可积」。<span class="marginnote">与辛条件对比：辛形式要求 $\omega^n \neq 0$（$\omega$ 非退化），接触形式要求 $\alpha \wedge (d\alpha)^n \neq 0$。辛是偶维、闭、非退化；接触是奇维、不闭、最大不可积。</span>

**接触结构（contact structure）**：超平面分布

$$
\xi := \ker \alpha
$$

由 $\alpha$ 定义，但接触结构本身是「分布」，两个相差一个非零函数倍数的 $\alpha$（$\alpha' = f\alpha$）给出同一个 $\xi$。定义 $d\alpha|_\xi$：它是 $\xi$ 上的非退化 2-形式（逐点），所以 $(\xi, d\alpha|_\xi)$ 在每个切平面上是个辛向量空间——**接触分布的每片叶子是「潜在辛」的**。

**Reeb 向量场（Reeb vector field）** $R_\alpha$：满足

$$
\iota_{R_\alpha} d\alpha = 0, \qquad \alpha(R_\alpha) = 1
$$

的唯一向量场。它横截于 $\xi$（因为 $\alpha(R) = 1 \neq 0$），且 $R$ 的流保持 $\alpha$（$\mathcal{L}_R\alpha = \iota_R d\alpha + d(\alpha(R)) = 0$）。Reeb 流是接触流形上的「时间演化」——在光学里，它是光线的方向场。

## 2 基本例子

**例1（标准 $\mathbb{R}^3$）**：$\alpha = dz - y\, dx$。则 $d\alpha = -dy \wedge dx = dx \wedge dy$，$\alpha \wedge d\alpha = (dz - y\,dx) \wedge dx \wedge dy = dz \wedge dx \wedge dy \neq 0$。接触结构 $\xi = \ker(dz - y\,dx)$ 由向量场 $\partial_y$ 与 $\partial_x + y\partial_z$ 张成。Reeb 场是 $\partial_z$。<span class="marginnote">$\mathbb{R}^3$ 的标准接触结构有个直观画面：把每点的平面想像成「倾斜的瓦片」，沿 $y$ 方向旋转——这就是为什么它也叫「旋转接触结构」。它是最基本的紧致接触结构的局部模型。</span>

**例2（球面 $S^3$）**：把 $S^3 \subset \mathbb{C}^2$ 看成单位球，$\alpha$ 是「沿切向的分量」$\alpha = \sum (x_j dy_j - y_j dx_j)|_{S^3}$。它是接触形式，且 $S^3$ 的接触结构与它作为「复结构的水平集」自然相关。

**例3（余切丛的单位余球面）**：$Q$ 的余切丛 $T^*Q$ 上，刘维尔 1-形式 $\lambda$ 在单位余球面 $S^*Q = \{(q,p): |p|=1\}$ 上的限制 $\alpha = \lambda|_{S^*Q}$ 是接触形式。**接触流形与辛流形的标准桥梁，第一条就来自余切丛。**

## 3 辛化与接触填充

**辛化（symplectization）**：接触流形 $(Y, \alpha)$ 的辛化是辛流形

$$
\mathrm{Symp}(Y, \alpha) = (Y \times \mathbb{R}_t, \; d(e^t \alpha))
$$

直接计算 $d(e^t \alpha) = e^t (dt \wedge \alpha + d\alpha)$。它的非退化性等价于 $\alpha \wedge (d\alpha)^n \neq 0$——**辛化成功当且仅当 $\alpha$ 是接触形式**。所以「接触」正是「能辛化」的精确条件。<span class="marginnote">更几何的版本用「锥」：$Y$ 的锥 $CY = Y \times \mathbb{R}_+$ 上放 $\omega = d(r\alpha)$（$r = e^t$）。对标准接触结构，辛化就是去掉原点的 $\mathbb{R}^{2n} \setminus \{0\}$，接触流形是锥的「截面」。这就是「接触是辛的奇维截面」说法的来源。</span>

**反向问题（填充）**：给定接触流形 $(Y, \xi)$，能否找到紧辛流形 $(W, \omega)$ 使 $\partial W = Y$ 且 $\xi = \ker(\omega|_Y)$？这样的 $(W, \omega)$ 叫 $Y$ 的**辛填充（symplectic filling）**。不是所有接触流形都可填充，判断可填充性是辛拓扑的核心问题之一（第3篇《辛嵌入与填充问题》展开）。<span class="marginnote">伪全纯曲线的 Gromov 紧致性（第3篇）给出了深刻的不可填充性障碍：某些接触流形（如过度扭转的）没有辛填充。这是「接触几何 + 辛几何 + 全纯曲线」三方合力的名场面。</span>

## 4 公式解析：辛化的非退化性

**核心公式：**

$$
d(e^t \alpha) = e^t (dt \wedge \alpha + d\alpha) \quad \text{在 } Y \times \mathbb{R} \text{ 上非退化} \iff \alpha \wedge (d\alpha)^n \neq 0
$$

拆解：

- **第一步，计算**：$d(e^t\alpha) = dt \wedge e^t\alpha + e^t d\alpha = e^t(dt \wedge \alpha + d\alpha)$。这是莱布尼茨法则，$e^t$ 因子保留。
- **第二步，取幂**：$(d(e^t\alpha))^{n+1} = e^{(n+1)t} (dt \wedge \alpha + d\alpha)^{n+1}$。展开时只有「全含 $d\alpha$ 或含一个 $dt\wedge\alpha$」的项有贡献：$(dt\wedge\alpha + d\alpha)^{n+1} = (n+1) dt \wedge \alpha \wedge (d\alpha)^n$（因为 $d\alpha$ 是 2-形式，$dt\wedge\alpha$ 是 2-形式，二者交换，二次以上交叉项为零）。
- **第三步，判读**：$Y \times \mathbb{R}$ 是 $2n+2$ 维，辛形式非退化当且仅当 $(d(e^t\alpha))^{n+1}$ 是处处非零体积形式。上一步显示它正比于 $dt \wedge \alpha \wedge (d\alpha)^n$——处处非零当且仅当 $\alpha \wedge (d\alpha)^n$ 处处非零。
- **第四步，结论**：**辛化非退化 ⟺ $\alpha$ 是接触形式**。这个等价把「奇维的接触条件」翻译成「偶维的非退化条件」，两个概念因此被焊成一体。

**直觉总结：** 辛化在接触流形上「竖」起一个径向方向（$t$ 或 $r$），把超平面分布 $\xi$ 逐点「转」成辛形式。接触条件保证锥是辛的，辛条件保证截面是接触的——**奇偶维的两种几何其实是同一个几何的两副面孔**。

## 5 Gray 稳定性与 Legendrian 子流形

**Gray 稳定性定理**：接触结构的同伦形变（固定 $\xi_0$，$\xi_t$ 连续变化）都可通过同痕实现：存在 $\psi_t$ 使 $\psi_t^*\xi_t = \xi_0$。这是接触版的 Moser/Darboux——接触结构在同伦意义下没有局部形变余地。<span class="marginnote">对比辛的 Moser 稳定性：辛要求上同调类相同，接触的同伦自动「可拉直」。但接触结构仍然有整体刚性——紧致流形上（如 $S^3$）存在同伦但不等价的接触结构，这是高维接触几何的活跃领域。</span>

**Legendrian 子流形（Legendrian submanifold）**：$Y$ 中维数 $n$、且处处与 $\xi$ 相切的子流形（接触版的 Lagrangian）。在辛化里，Legendrian 子流形提升为 Lagrangian 锥。它们在后来的 Lagrangian 接触同调、结理论的 Legendrian 分类中扮演核心角色。<span class="marginnote">在 $\mathbb{R}^3$ 的标准接触结构里，Legendrian 结是「处处与 $\xi$ 相切」的曲线，其前投影 $yz$-平面的图自带「无正切」约束。Legendrian 结理论由此产生——它是接触几何与拓扑结合的丰产区。</span>

## 6 紧致接触结构：tight 与 overtwisted

紧致接触流形上，接触结构分成两类，这个二分是 Eliashberg 的重大贡献：

**紧致接触结构（tight contact structure）**：不存在「过扭转圆盘」（overtwisted disk）——即不存在嵌入圆盘 $D^2 \hookrightarrow Y$，使 $\partial D^2$ 是 Legendrian 边界且 $\xi$ 与圆盘相切于边界内部。**紧致结构是「几何刚性」的**：它们保持 $Y$ 的许多拓扑信息，且与辛填充兼容（见第3篇《辛嵌入与填充问题》）。

**过扭转结构（overtwisted contact structure）**：存在至少一个过扭转圆盘。**Eliashberg 定理**：过扭转结构（在固定同伦类里）由同伦数据完全分类——「柔软」（flexible），且**不可被辛填充**。<span class="marginnote">「紧致 vs 过扭转」是接触几何里的「刚性 vs 柔性」二分（对比第3篇 Gromov 非压缩篇的刚性/柔性）。$S^3$ 上的标准接触结构是紧致的；$S^3$ 上还存在着无穷多个过扭转结构。判断紧致性常常需要深层的全纯曲线论证——这是高维接触几何的活跃前沿。</span>

**与辛化的联系**：紧致接触结构可辛化且「几何」；过扭转结构辛化后「塌缩」。所以**填充问题（第3篇）与紧致性直接挂钩**：可辛填充 ⟹ 紧致（但紧致未必可填充）。

**例（$S^3$）**：标准接触结构 $\xi_{\mathrm{std}}$ 是紧致的；对每个整数 $k$，存在过扭转结构 $\xi_k$（Darboux 球 + 过扭转修正），它们同伦但不同胚。**同伦等价但接触不等价**——接触结构比「拓扑」细。

**接触几何与光学**：接触几何的名字来自「接触（contact）」，其原型是几何光学：光线族生成波前，波前之间「接触」的条件由接触形式描述。更精确地说，光线的空间（相空间的光线模型）是一个接触流形，其 Reeb 流就是「沿光线传播」——**接触流形是「光线传播」的几何舞台**。这个古老起源提醒我们：接触结构不是孤立的抽象，而是「传播现象」的数学语言。

**热力学的接触结构**：热力学里，热力学状态空间（如 $p$-$V$-$T$）带一个自然的接触形式 $dU - T dS + p dV = 0$，其 Legendre 子流形对应「平衡态流形」。**接触几何在这里是「热力学第二定律的几何化」**——可逆过程的路径沿 Legendre 子流形，不可逆过程「离开」它。这提醒我们：接触结构横跨光学、力学、热力学三大物理分支，是「约束传播」的统一语言。

**与辛化的最终对话**：接触流形与其辛化 $Y \times \mathbb{R}$ 的关系，也可以反过来读——**辛流形带一个「径向」方向时，其「截面」自动是接触流形**（当径向方向是「辛向量场」时）。这就把「辛流形带什么样的全局方向」变成「接触几何的分类问题」——这正是第3篇「辛嵌入与填充」与第4篇「ECH 谱」的技术出发点：ECH 就是在「辛化的流形」上对 Reeb 轨道做 Floer 理论。

**Legendrian 的「前端」直觉**：在 $\mathbb{R}^3$ 标准接触结构里，Legendrian 结投影到 $yz$-平面时出现「尖点」与「横截交叉」两种奇点，且**交叉处上下层满足「左/右」交替规则**——这就是 Legendrian 结的「前端图（front）」理论。前端图让 Legendrian 分类变成「可画」的组合问题，是接触拓扑里最直观、也最实用的工具之一（Chekanov 多项式的计算就发生在前端图上）。

## 7 小结

- **接触形式** $\alpha$：$\alpha \wedge (d\alpha)^n \neq 0$ 处处非零——奇维版本的非退化条件；**接触结构** $\xi = \ker\alpha$ 是最大不可积超平面分布。
- **Reeb 向量场**：横截 $\xi$、保持 $\alpha$ 的规范向量场；给出接触流形上的「时间」。
- **例子**：$\mathbb{R}^3$ 标准接触结构、$S^3$、余切丛的单位余球面。
- **辛化**：$(Y \times \mathbb{R}, d(e^t\alpha))$ 是辛流形 ⟺ $\alpha$ 是接触形式——**接触正是「能辛化」的精确条件**。
- **Legendrian 子流形**：处处切 $\xi$ 的 $n$ 维子流形，接触版的 Lagrangian；辛化中提升为 Lagrangian 锥。
- **紧致 vs 过扭转**：Eliashberg 二分——过扭转可同伦分类且不可辛填充，紧致与填充问题（第3篇）挂钩。
- **应用**：光学（光线空间 = 接触流形，Reeb 流 = 光线传播）、热力学（状态空间的 Legendre 子流形 = 平衡态流形）。
- **数值检验**：$\mathbb{R}^3$ 标准形式 $\alpha = dz - y\,dx$ 满足 $\alpha \wedge d\alpha = dz \wedge dx \wedge dy \neq 0$——接触条件逐点可验，是「最大不可积」最直接的落地。

在下一节，我们将进入可积系统与 Liouville-Arnold 定理：当守恒量的数目多到等于自由度，相空间被纤维化成环面，哈密顿流变成环面上的直线运动。可积系统的「环面纤维化」与本文接触结构的「潜在辛分布」共享同一个思想——**把动力系统的可解性读成几何结构的规整性**。