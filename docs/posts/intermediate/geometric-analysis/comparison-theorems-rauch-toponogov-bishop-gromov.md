---
title: 比较定理（Rauch、Toponogov、Bishop–Gromov 体积比较）
date: 2026-08-07
---

# 比较定理（Rauch、Toponogov、Bishop–Gromov 体积比较）

<div class="epigraph">
<p>「凡是可测的，就去测它；凡是不可测的，就设法让它变得可测。」</p>
<footer>—— 伽利略 · 伽利雷（Galileo Galilei），常被引用的科学方法论箴言</footer>
</div>

<div class="article-byline">
<p>第二级 · 几何分析 ｜ Peter Li《Geometric Analysis》Ch. 2（Comparison Geometry）｜ Jost 比较几何章 ｜ 2026-08-07</p>
</div>

## 为什么从比较定理开始

测地线的研究告诉我们「曲率决定测地线的聚焦」。比较定理把这个直觉变成数学的杠杆：**用常曲率空间（球面、欧氏空间、双曲空间）作为标尺，把任意流形的几何量夹在两条标尺之间。** 若已知截面曲率或 Ricci 曲率的上下界，就能控制测地线的距离、角、体积的增长——这是几何分析「用曲率界换几何结论」的第一个范式，之后的热核估计、谱估计、Ricci 流都存在性证明全部建立在这类比较之上。

从课程体系看，本篇是从「局部几何」（度量、联络、曲率）走向「整体几何」（直径、体积、拓扑）的第一座桥：Bonnet–Myers 用正 Ricci 曲率推出流形紧致且直径有界，这是「曲率控制拓扑」的最早也是最深刻的例子，与第四级《广义相对论》里「正能量密度导致时空聚焦」的思想一脉相承。

<span class="marginnote">「用常曲率模型空间作标尺」的思想可上溯到黎曼本人：他在就职演说中就建议用常曲率空间作为度量比较的基准。现代形式的 Rauch 比较定理出现在 1951 年前后（Rauch、Berger 等），Toponogov 三角比较则来自苏联学派，Bishop–Gromov 体积比较由 Bishop 提出、Gromov 推广，是当代几何分析最常用的工具之一。</span>

## 1 模型空间与截面曲率比较

**模型空间（model spaces）**：截面曲率恒为常数 $\kappa$ 的完备单连通黎曼流形——$\kappa>0$ 是球面 $S^n$，$\kappa=0$ 是欧氏空间 $\mathbb{R}^n$，$\kappa<0$ 是双曲空间 $\mathbb{H}^n$。它们各自有显式的距离函数、体积公式，是完美的「对照实验组」。

**常曲率空间的完整清单**：完备单连通、截面曲率恒为 $\kappa$ 的流形只有三种——$\kappa>0$ 的球面 $S^n$（半径 $1/\sqrt\kappa$）、$\kappa=0$ 的欧氏空间 $\mathbb{R}^n$、$\kappa<0$ 的双曲空间 $\mathbb{H}^n$（可用 Poincaré 半空间模型实现）。三者之间无法互相等距，构成比较理论的三个「标尺刻度」；非单连通的常曲率空间是这些空间的商（如透镜空间 $S^n/\mathbb{Z}_k$）。

**Rauch 比较定理（Rauch comparison theorem）**（大意）：设 $M,\tilde M$ 的截面曲率满足 $K(\sigma) \le \tilde K(\tilde\sigma) = \kappa$，沿两条对应的测地线取 Jacobi 场，则**在无共轭点的区间内，$M$ 上的 Jacobi 场长度不小于 $\tilde M$ 上对应者**。反之若 $K \ge \kappa$，则 $M$ 上的 Jacobi 场更短。

用几何语言说：**曲率越小，测地线发散得越快（Jacobi 场长得越快）；曲率越大，测地线聚得越紧。**<span class="marginnote">直观对照：平面（$K=0$）上平行线永不相交，双曲空间（$K<0$）上平行线发散、三角内角和小于 $\pi$，球面（$K>0$）上测地线聚焦、内角和大于 $\pi$。Rauch 定理就是把这三档比较推广到「曲率有界」的一般流形。</span>

由 Rauch 定理的 Jacobi 场长度控制，取 $J$ 为沿径向的偏离场，即可推出**关于距离函数的截面曲率比较**：例如沿测地线的径向 Jacobi 场 $\xi(t)$ 满足

$$\frac{\xi''}{\xi} + K(\xi,\dot\gamma) \le 0 \quad\Rightarrow\quad \xi(t) \le \sin_{\kappa}(t)\, \xi(1)\ \text{（当 }K\ge\kappa\text{）}$$

其中 $\sin_\kappa$ 是模型空间的「正弦函数」：$\kappa>0$ 为 $\frac{1}{\sqrt\kappa}\sin(\sqrt\kappa\,t)$，$\kappa=0$ 为 $t$，$\kappa<0$ 为 $\frac{1}{\sqrt{-\kappa}}\sinh(\sqrt{-\kappa}\,t)$。

## 2 Toponogov 三角比较

**Toponogov 定理（Toponogov's theorem）**把比较推广到「大」的三角形：设 $(M,g)$ 的截面曲率 $K \ge \kappa$，取测地线三角形 $\triangle pqr$（每条边是长度的测地线）。则这个三角形不比模型空间 $\kappa$ 中的对应三角形「更瘦」：对应角满足

$$\angle_q(p,r) \ge \angle_q^{\kappa}(p,r), \qquad c^2 \ge a^2 + b^2 - 2ab\cos\kappa(\gamma)$$

其中 $\cos_\kappa$ 是模型空间的余弦函数。换言之，**正曲率让三角形的边「更弯向外」、角更大**。

<span class="marginnote">Toponogov 的意义在于它是「全局」的：Rauch 是局部的（无共轭点区间），而 Toponogov 允许三角形跨越割迹。它把截面曲率下界的局部信息整合为整体的距离比较，是后面流形收敛理论（Cheeger–Gromov）中「曲率有界 ⇒ 预紧」证明的基石。</span>

Toponogov 的比较是**刚性**的：如果一个大三角形处处取到模型空间的极值，且流形足够正则，则流形本身在相应区域内与模型空间等距——这类「比较达到极限即刚性」的结果（与等周不等式、Splitting 定理一起）是几何分析中最优雅的一类结论。

## 3 公式解析：Bishop–Gromov 体积比较

截面曲率比较关注测地线的聚焦，而 **Ricci 曲率只控制体积**。**Bishop–Gromov 体积比较（Bishop–Gromov volume comparison）**给出流形上测地球体积的单调性：

$$\frac{\operatorname{Vol}(B(p,r))}{\operatorname{Vol}(B_{\kappa}(r))} \ \text{关于 } r \text{ 单调递减}$$

其中 $B_\kappa(r)$ 是模型空间中以半径为 $r$ 的球（球面情形取 $r \le \pi/\sqrt\kappa$）。逐项拆解：

- **第一步，条件**：假设 $\operatorname{Ric}_g \ge (n-1)\kappa g$——注意这里只需 **Ricci 下界**，不要求截面曲率。
- **第二步，机理**：径向距离函数 $r(x)=d(p,x)$ 的 Laplace 满足 **Laplace 比较**：$\Delta r \le (n-1)\sqrt{-\kappa}\coth(\sqrt{-\kappa}\,r)$（$\kappa\le0$ 时；$\kappa>0$ 用 $\cot$）。$|\nabla r|=1$ 与 $\Delta r$ 的上界联合，通过体积元的演化方程 $\partial_r \log(\sqrt{\det g_{radial}})$ 控制球体积。
- **第三步，为什么单调**：设 $\theta(r)$ 为 $p$ 处单位球面测度在径向映射下的推前，$\theta$ 满足的 ODE 逐点被模型空间的 $\theta_\kappa$ 所控；比例 $\theta/\theta_\kappa$ 沿 $r$ 单调不增，积分后即得体积比的单调性。

**两个直接推论**是几何分析的高频工具：

**Bonnet–Myers 定理**：若 $\operatorname{Ric} \ge (n-1)\kappa > 0$，则 $\operatorname{diam}(M) \le \pi/\sqrt\kappa$，且 $M$ 紧致、基本群有限。由 $r\to\pi/\sqrt\kappa$ 时模型球体积趋于零、而流形球体积有限即可。

**一个具体算例**：$n=3$，$\kappa=1$（对照单位球 $S^3$，$\operatorname{Ric}=2g$）。Bishop–Gromov 说任何满足 $\operatorname{Ric}\ge 2g$ 的三维流形，其测地球体积不超过球面同半径测地球体积；且当 $r\to\pi$ 时球面球体积趋于总体积——直接推出流形直径不超过 $\pi$。这正是 Bonnet–Myers 的直观来源：正 Ricci 曲率把体积「压缩」进有限半径。

**等周常数与覆盖引理**：Bishop–Gromov 给出了球族体积的多项式控制，从而在切球计数（packing）论证中提供常数——这是后面热核估计、Moser 迭代、谱下界里「体积因子」的统一来源。

## 4 比较定理在几何分析中的用法

比较定理不是孤立的几个漂亮定理，而是一个**工具箱**。下列场景会在本专题反复出现：

- **热核的上界**：Bishop–Gromov 控制体积增长，配合 Sobolev 不等式得到热核上界（见《热方程与热核》篇）。
- **特征值下界**：体积比较给出等周常数，等周常数喂给 Cheeger 不等式（见《谱几何》篇）。
- **Ricci 流的短时间存在性**：需要曲率张量的量纲一致估计，其中用到 Laplacian 比较与体积比较来统一处理曲率生长（见《Ricci 流引论》篇）。
- **正质量定理**：Schoen–Yau 的极小曲面方法把「平均曲率与面积比较」翻译为质量的非负性（见《前沿专题》篇）。

| 比较定理 | 曲率条件 | 控制对象 | 直接推论 |
| --- | --- | --- | --- |
| Rauch | 截面曲率有界 | Jacobi 场长度、距离函数 | 共轭点估计、最小测地线稳定性 |
| Toponogov | 截面曲率 $K\ge\kappa$ | 大三角形边长、角度 | 曲率有界 ⇒ 收敛预紧、刚性 |
| Laplace 比较 | Ricci 下界 | 径向距离的 $\Delta r$ | 体积元演化、热核估计 |
| Bishop–Gromov | Ricci 下界 | 球体积增长 | Bonnet–Myers、等周常数、覆盖引理 |

<span class="marginnote">记忆口诀：「截面曲率控制距离和角，Ricci 曲率控制体积，标量曲率控制 Yamabe 型不变量。」四层曲率各自对应一组比较定理，这是几何分析的分工图。</span>

### 4.1 比较定理的统一视角

比较定理可以统一看成「一族 Riccati 方程的比较」：沿测地线，径向 Jacobi 场模长 $J$ 满足 $J'' + K J = 0$（一维振荡方程），而模型空间的对应量满足 $J_\kappa'' + \kappa J_\kappa = 0$。由 **Sturm 比较定理**，$K \ge \kappa$ 时 $J$ 比 $J_\kappa$ 更早归零——测地线更早聚焦。这个「Sturm 比较」是上述全部比较定理的最小公分母。

### 4.2 刚性：比较达到极限

比较定理的极限情形通常自动「刚性」——若某个比较的等号处处成立，流形本身就在对应区域与模型空间等距：

**Cheeger–Gromoll 分割定理（splitting theorem）**：若完备流形满足 $\operatorname{Ric}\ge 0$ 且含一条直线（两端都测地完备的直线），则 $M$ 等距分解为 $N \times \mathbb{R}$，其中 $N$ 仍满足 $\operatorname{Ric}\ge0$。反复运用即得：$\operatorname{Ric}\ge0$ 的流形分解为 $\mathbb{R}^k \times N$，$N$ 紧致。这条定理是「Ricci 非负 + 无界 ⇒ 圆柱分解」的精确表述，也是 Perelman 处理 Ricci 流「不塌缩 + 长柱」情形的几何依据。

**横向连接**：比较定理的思想在第四级《广义相对论》中继续深化——**Raychaudhuri 方程**正是「Laplace/体积比较」在时空（伪黎曼流形）中的对应物，它把正能量条件转化为测地线束的聚焦，是奇点定理的核心。

**辨析｜易错点：** Bishop–Gromov 的条件是 Ricci 下界而非截面曲率下界；Ricci 曲率下界并不保证截面曲率有界，所以不能指望从体积比较推出距离比较。反过来，Rauch 需要截面曲率——它的结论比体积更强，但前提也更苛刻。

**术语速查**：

| 记号 / 术语 | 含义 | 备注 |
| --- | --- | --- |
| 模型空间 $M^n_\kappa$ | 截面曲率恒为 $\kappa$ 的单连通空间 | $S^n$ / $\mathbb{R}^n$ / $\mathbb{H}^n$ |
| $\sin_\kappa(t)$ | 模型空间的正弦函数 | $\kappa$ 为负时换为 $\sinh$ |
| Jacobi 场 | 测地线变分场，满足 $J''+KJ=0$ | 反映测地线聚焦 |
| 共轭点 | Jacobi 场在两点都为零 | Rauch 定理的有效区间端点 |
| $\Delta r$ Laplace 比较 | $\operatorname{Ric}\ge(n-1)\kappa$ 时 $\Delta r \le (n-1)\sqrt{-\kappa}\coth(\sqrt{-\kappa}\,r)$ | Bishop–Gromov 的核心输入 |
| 割迹 | 测地线不再最短的点集 | Toponogov 可跨越割迹 |
| Splitting 定理 | $\operatorname{Ric}\ge0$ + 直线 ⇒ 圆柱分解 | Cheeger–Gromoll |

## 5 小结

- **模型空间**（球面 / 欧氏 / 双曲）是曲率比较的标尺，几何量被夹在 $\sin_\kappa$、$\cos_\kappa$ 之间。
- **Rauch 定理**：截面曲率上界 ⇒ Jacobi 场增长更慢；下界 ⇒ 聚焦更快。局部比较。
- **Toponogov 定理**：截面曲率下界 ⇒ 大三角的边角被模型空间夹住；全局比较，是收敛理论基石。
- **Bishop–Gromov**：Ricci 下界 ⇒ 球体积比单调递减；推出 **Bonnet–Myers** 与等周常数。
- 比较定理是「用曲率界换几何量」的第一范式，之后各篇均以它为工具。

在下一节，我们开始给流形装上分析学的引擎——**Laplace–Beltrami 算子与 Hodge 理论**：把梯度、散度、调和形式这些欧氏工具整体地搬到流形上，并看它们如何读出流形的拓扑。
