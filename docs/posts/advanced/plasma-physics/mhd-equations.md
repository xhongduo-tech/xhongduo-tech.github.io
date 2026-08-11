---
title: 磁流体力学方程
date: 2026-08-11
---

# 磁流体力学方程

<div class="epigraph">
<p>磁流体力学把等离子体当作一种导电的流体来处理——在这门学科里，磁力线像琴弦一样绷在流体上，物质与场互相拖着走。</p>
<footer>—— 弗朗西斯 · 陈（Francis F. Chen, <em>Introduction to Plasma Physics and Controlled Fusion</em>）</footer>
</div>

<div class="article-byline">
<p>第四级 · 高阶专题 · 等离子体物理 ｜ 对标教材 Chen Ch.3 ｜ 2026-08-11</p>
</div>

## 为什么从单粒子到流体

单粒子轨道（上一节）告诉我们「一个粒子怎么动」，但托卡马克里有 $10^{20}$ 个粒子——逐个追踪是不现实的，也是不必要的。
当体系尺度远大于回旋半径、时间尺度远大于回旋周期时，我们关心的是**密度、速度、压强、磁场**这些宏观量随时间的演化。
这就把等离子体压缩成一种**导电流体**，方程组即**磁流体力学（Magnetohydrodynamics, MHD）**。聚变装置的平衡、
稳定性（第 7 篇）、以及太阳风与行星磁场的相互作用，全都用它建模。<span class="marginnote">MHD 之于等离子体，
恰如「流体力学 + 电磁学」之于空气：空气压缩产生声波，导电流体压缩产生磁声波——后文阿尔芬波、磁声波都是 MHD 的产物。</span>

## 1 从千万个粒子到一个流体元

把分布函数对速度空间求矩（这在第 8 篇动理学那里会正式展开），就得到流体量：**数密度** $n$、**平均速度** 
$\mathbf{u}$、**压强张量** $\overleftrightarrow{P}$。压强来自粒子的无规热运动——理想气体定律 
$p = nkT$ 在等离子体中仍然成立，只是要分电子与离子两支。这一节我们从「一张表」出发，只写单流体（合并两族粒子后的整体）版本的方程，
它已经能解释磁约束的绝大部分宏观行为。
<span class="marginnote">单流体近似成立的条件：体系尺度远大于回旋半径与德拜长度、时间尺度远大于碰撞时间与回旋周期——磁约束
装置芯部通常满足，而边界区域、稀薄天体等离子体则要回到双流体甚至动理学。</span>

## 2 连续性方程：质量不灭

质量守恒：一个流体元里质量的减少，等于流出去的流量。

$$
\frac{\partial \rho}{\partial t} + \nabla\cdot(\rho\,\mathbf{u}) = 0
$$

其中 $\rho = n(m_i + m_e) \approx n m_i$（电子太轻，几乎不贡献质量）。
这和普通流体力学的连续性方程一模一样——**等离子体把能量交给场，但质量守恒绝不妥协**。
<span class="marginnote">在环形装置里，这个方程配合「等离子体不与壁接触」的边界条件，决定了约束时间：
粒子一旦横越磁力线撞上壁，连续性方程便会在那一点报警。</span>

## 3 运动方程：动量与洛伦兹力

动量守恒给出单流体运动方程：

$$
\rho\left(\frac{\partial \mathbf{u}}{\partial t} + 
\mathbf{u}\cdot\nabla\mathbf{u}\right) = -\nabla p + 
\mathbf{J}\times\mathbf{B}
$$

右边两项：$-\nabla p$ 是**压强梯度力**（与普通流体相同），$\mathbf{J}\times\mathbf{B}$ 
是**洛伦兹力**——整团流体携带电流 $\mathbf{J}$ 在磁场里受力。磁力与流体力在同一个方程里角力，这就是 MHD 的「磁」
与「流体」之结合。

需要补充电磁方程组（麦克斯韦 + 欧姆定律）才能闭合。静磁近似下 
$\nabla\times\mathbf{B} = \mu_0 \mathbf{J}$，于是洛伦兹力可以改写为：

$$
\mathbf{J}\times\mathbf{B} = 
\frac{1}{\mu_0}(\mathbf{B}\cdot\nabla)\mathbf{B} - 
\nabla\frac{B^2}{2\mu_0}
$$

第一项是**磁张力**（磁力线像橡皮筋，弯曲了要拉直），第二项是**磁压**（$\nabla B^2/2\mu_0$）。**磁力线张力 + 
磁压，就是「磁场对流体施加的全部力」。**<span class="marginnote">把 $B^2/2\mu_0$ 与热压 $p$ 对比：
$p \sim B^2/2\mu_0$时，流体压强与磁压相当，比值就是后文的 $\beta$——聚变界孜孜以求的大 $\beta$ 装置，
本质是「用同样大的磁场压住更胖的等离子体」。
</span>

## 4 广义欧姆定律与磁冻结

等离子体的欧姆定律（单流体、含电阻率 $\eta$）：

$$
\mathbf{E} + \mathbf{u}\times\mathbf{B} = \eta\,\mathbf{J}
$$

把 $\mathbf{J}$ 代入法拉第定律 
$\partial\mathbf{B}/\partial t = -\nabla\times\mathbf{E}$，得到磁场演化方程：

$$
\frac{\partial\mathbf{B}}{\partial t} = 
\nabla\times(\mathbf{u}\times\mathbf{B}) + 
\frac{\eta}{\mu_0}\nabla^2\mathbf{B}
$$

第一项说**磁场随流体一起运动**，第二项是磁场被电阻「吃掉」的扩散。两相对比定义**磁雷诺数** 
$R_m = \mu_0 L u/\eta$：若$R_m \gg 1$，扩散项可忽略，得到**理想 MHD** 的磁冻结（frozen-in 
flux）定理：

$$
\frac{\partial\mathbf{B}}{\partial t} = \nabla\times(\mathbf{u}\times\mathbf{B})
$$

**磁冻结**：等离子体与磁力线「焊死」在一起，流体质点只能沿磁力线走，磁力线随流体运动而运动。
<span class="marginnote">太阳风把太阳磁场一路拖到地球轨道，日冕物质抛射把一团「冻住磁场」
的等离子体抛向地球——都是磁冻结的表现。而在托卡马克中，磁冻结意味着等离子体无法横越磁力线扩散：输运（第 6 篇）只能靠碰撞与湍流「悄悄钻空子」
。</span>

## 5 平衡方程与 β 值

静态平衡（$\mathbf{u}=0$）下运动方程退化为力平衡：

$$
\nabla p = \mathbf{J}\times\mathbf{B}
$$

等离子体压强梯度由洛伦兹力撑住。两侧点乘 $\mathbf{B}$：
$\mathbf{B}\cdot\nabla p = 0$——**压强沿磁力线不变**；再点乘 $\nabla p$：
$\mathbf{J}\cdot\nabla p = 0$——电流沿压强梯度方向的分量为零。这两条几何约束，直接决定了约束装置的磁场位形。

由此定义**等离子体 β 值**：

$$
\beta = \frac{p}{B^2/2\mu_0}
$$

$\beta$ 衡量「热压与磁压之比」。磁约束装置要高效，希望 $\beta$ 大；但 $\beta$ 过大等离子体就会撑破磁场（第 7 
篇的不稳定性）。托卡马克典型 $\beta\sim 1\%$，太阳日冕 $\beta \ll 1$，太阳风的 $\beta$ 则跨越多个量级。

**辨析｜易错点：** 不要把 MHD 的 $\mathbf{E}+\mathbf{u}\times\mathbf{B}=0$ 
误读成「电场为零」。理想导体的欧姆定律是「本构关系」$\mathbf{J}=\sigma\mathbf{E}'$（$\mathbf{E}'$ 
为流体静止系中的电场），在 $\sigma\to\infty$ 极限下它约束的是**电场与运动的耦合**，而不是电场本身。

## 6 公式解析：磁冻结方程

$$
\frac{\partial\mathbf{B}}{\partial t} = \nabla\times(\mathbf{u}\times\mathbf{B})
$$

三步拆解：

- **第一步，看左边**：$\partial\mathbf{B}/\partial t$ 是磁场随时间的**变化率**——这条方程是磁场的「演化宪法」。
- **第二步，看右边**：$\nabla\times(\mathbf{u}\times\mathbf{B})$。叉积 $\mathbf{u}\times\mathbf{B}$ 是流体运动产生的电场（动生电动势）；对它的旋度再取进来，就是「这个电场又怎样改变磁场」——法拉第定律的自洽环。<span class="marginnote">对照法拉第定律 $\partial\mathbf{B}/\partial t = -\nabla\times\mathbf{E}$，把理想欧姆定律 $\mathbf{E}=-\mathbf{u}\times\mathbf{B}$ 代进去，两边负号相消，得证。</span>
**第三步，几何直觉**：想象一团流体带着磁力线整体移动。流体元附近的磁通量 $\Phi = \int\mathbf{B}\cdot d\mathbf{S}$ 保持为常数——**磁通不穿破流体质点**。用数学语言说：磁通量守恒 $\Leftrightarrow$ $\mathbf{B}/\rho$ 沿流线演化，磁场随密度一起被「拧」变形。<span class="marginnote">「$\mathbf{B}/\rho$ 随流线演化」是磁冻结最实用的表述：太阳表面磁管被对流举起、磁场被放大、最终爆发耀斑，全程都遵守这条关系。</span>

## 7 小结

- 单流体 MHD 由**连续性方程**（质量守恒）与**运动方程**（$\rho d\mathbf{u}/dt = -\nabla p + \mathbf{J}\times\mathbf{B}$）构成。
- 洛伦兹力等价于**磁张力 + 磁压**，$\beta = p/(B^2/2\mu_0)$ 是磁约束的效率指标。
- 理想 MHD 的欧姆定律 $\mathbf{E}+\mathbf{u}\times\mathbf{B}=0$ 导出**磁冻结**：磁力线与流体焊死。
- 磁雷诺数 $R_m$ 判定冻结与扩散谁占上风；真实等离子体总有电阻，冻结会被打破（磁重联、破裂）。
- 平衡方程 $\nabla p = \mathbf{J}\times\mathbf{B}$ 给出磁约束装置的基本几何约束。

在下一节，我们让流体动起来、探一探它振动起来的声音——**等离子体振荡与色散关系**。
