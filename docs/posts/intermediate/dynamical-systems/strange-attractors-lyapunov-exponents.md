---
title: 奇怪吸引子与 Lyapunov 指数
date: 2026-08-11
---

# 奇怪吸引子与 Lyapunov 指数

<div class="epigraph">
<p>确定性的背后藏着偶然，偶然之中又隐藏着确定性——我们正在学习同时看见两者。</p>
<footer>—— 约瑟夫 · 福特（Joseph Ford），混沌理论先驱</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 动力系统 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从奇怪吸引子开始

Lorenz 篇我们见过奇怪吸引子的「蝴蝶翅膀」，但只凭图像说话不够严谨。这一篇给它配两件定量工具：**Lyapunov 指数**测量「误差放大多快」（混沌的强度），**分形维数**测量「吸引子的几何有多奇怪」。<span class="marginnote">「奇怪」一词来自 Ruelle 与 Takens（1971）：他们把「对初值敏感 + 分形结构」的吸引子命名为 strange attractor，用以区别于平庸的极限环与不动点。后来才意识到「奇怪」应严格与「分形维数」「正 Lyapunov 指数」挂钩。</span>

对 AI 与数据科学，这两件工具早已出圈：**Lyapunov 指数被用于时间序列预测的可行性评估（数据是否混沌），分形维数被用于识别数据集的本质复杂度**。而对理论，它们是「混沌 = 正指数 + 分形几何」这一定量的现代定义的基石。

## 1 奇怪吸引子：正式定义

**吸引子（attractor）** $\Lambda \subset \mathbb{R}^n$：对某个开邻域 $U \supset \Lambda$，所有 $x \in U$ 的轨线都收敛进 $\Lambda$，且 $\Lambda$ 是**不可约**的（内部没有更小的真吸引子，即拓扑传递）。平凡吸引子：稳定不动点、稳定极限环、稳定环面。

**奇怪吸引子（strange attractor）**：在上述基础上，**至少有一个正的 Lyapunov 指数**（对初值敏感），且**分形维数非整数**（几何上是分形）。<span class="marginnote">注意「奇怪」历史上指「非周期的、敏感的」，现代约定俗成把「正 Lyapunov 指数」作为混沌吸引子的核心判据。Lorenz 吸引子、Rossler 吸引子、Chua 电路都是典型例子。</span>

**辨析｜易错点：**「吸引子」不等于「系统最终状态」。Lorenz 吸引子中轨线永不重复、永不静止，但统计上稳定——它是**几何上的极限集**而非单一状态。同样，**「奇怪」是几何结论，「混沌」是动力学结论**：正指数管混沌（时间）、分形管奇怪（空间），二者相关但不等同。

## 2 Lyapunov 指数：测量误差放大

设 $n$ 维系统 $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$，取无穷小扰动 $\delta\mathbf{x}(t)$ 沿轨线演化，其 Jacobian 为 $D\mathbf{f}(\mathbf{x}(t))$。**Lyapunov 指数**定义为

$$
\lambda(\mathbf{x}_0, \mathbf{v}) = \lim_{t\to\infty} \frac{1}{t} \ln \frac{|\delta\mathbf{x}(t)|}{|\delta\mathbf{x}(0)|}, \qquad \delta\mathbf{x}(0) = \mathbf{v}.
$$

对「几乎所有」扰动 $\mathbf{v}$，极限收敛到同一个值——**最大 Lyapunov 指数** $\lambda_{\max}$。它回答「初值差一点，多久后分道扬镳」：

- $\lambda_{\max} > 0$：误差指数放大，**混沌**；
- $\lambda_{\max} < 0$：误差指数收缩，收敛到不动点；
- $\lambda_{\max} = 0$：误差既不放大也不收缩（对应周期或准周期运动）。<span class="marginnote">对周期轨道，沿轨道切向的指数恒为 0，其余负——这就是为什么极限环「中性稳定」于相位移而吸引于横向。$\lambda$ 的完整谱排序 $\lambda_1 \ge \lambda_2 \ge \dots$，其和为 $n$ 维相体积收缩率。</span>

**数值估计**：初始单位球面沿轨线演化成椭球，主轴按 $e^{\lambda_i t}$ 伸缩——$\lambda_i$ 就是主轴对数伸缩率的长期平均。程序上，对 $\delta\mathbf{x}$ 每隔几步做一次 **Gram–Schmidt 重正交化**（避免数值溢出），累积对数伸缩率再除以总时间，即可估计全谱。这个算法是时间序列混沌检测（Wolf 算法、Benettin 算法）的基础。

## 3 分形维数与奇怪几何

混沌吸引子的几何特征是**分形**：整体有限维、局部无限细节、自相似。三种常用维数：

**Hausdorff 维数 $D_H$**：覆盖刻画的「度规」维数，定义严谨但难算；
- **盒维数（box-counting）** $D_B = \lim_{\varepsilon\to 0} \frac{\log N(\varepsilon)}{\log (1/\varepsilon)}$，其中 $N(\varepsilon)$ 是覆盖吸引子所需边长 $\varepsilon$ 的盒子数；
- **关联维数（correlation dimension）** $D_2$：从时间序列直接估计（Grassberger–Procaccia 算法），最常用于实验数据。<span class="marginnote">Lorenz 吸引子 $D_H \approx 2.06$——比平面（2）大一点、比立体（3）小一点。分形维数介于整数之间，正是「既不是面、也不是体」的几何结构的定量写照。分形几何的创始人曼德博（Mandelbrot）在本专题属于《分形几何》的分支。</span>

**估计程序**：取长时间序列 $\{x_i\}$，按嵌入维数构造重构相空间（Takens 嵌入），统计「距离小于 $\varepsilon$ 的点对比例」$C(\varepsilon) \propto \varepsilon^{D_2}$，在对数坐标里直线的斜率就是关联维数。<span class="marginnote">嵌入维数要 ≥ $2D+1$（Takens 定理），关联维数于是能从<strong>单一标量时间序列</strong>重建动力学的几何——天气预报、心电信号、股票序列的混沌检验都靠这套「只观察一个变量，还原整个相空间」的魔法。见第三级《时间序列分析》。</span>

## 4 吸引子的分形结构与耗散

为什么混沌系统有奇怪吸引子、而保守系统没有？答案在**耗散**。对 Lorenz 系统，$\mathrm{div}\,\mathbf{F} = -(\sigma + 1 + \beta) < 0$：相体积被持续压缩。同时混沌要求误差放大（正指数）——这两个矛盾要求由分形结构同时满足：**体积被压缩（整体收缩），但方向被拉伸（局部发散），于是物质在有限区域内被无限折叠**。

**折叠与拉伸（baker's map 思想）**：把「拉伸—折叠」迭代，就像揉面团——体积不变但表面积无限增长，混沌吸引子的分形结构就是无数次拉伸-折叠的残留。<span class="marginnote">KAM 系统（保守、无耗散）里没有这种折叠的动力学原因，因此没有奇怪吸引子；耗散是「体积收缩」与「误差放大」能共存的必要条件——这是「奇怪吸引子只出现在耗散系统」的直觉解释。</span>

## 5 公式解析：Lyapunov 指数的定义式

$$
\lambda = \lim_{t\to\infty}\frac{1}{t}\ln\frac{|\delta\mathbf{x}(t)|}{|\delta\mathbf{x}(0)|}
$$

- **$\delta\mathbf{x}(t)$**：扰动向量，沿变分方程 $\dot{\delta\mathbf{x}} = D\mathbf{f}(\mathbf{x}(t))\,\delta\mathbf{x}$ 演化——它是「误差动力学」本身。
- **对数比值**：$\ln(|\delta\mathbf{x}(t)|/|\delta\mathbf{x}(0)|)$ 是到时刻 $t$ 为止误差被放大的总对数倍数；除以 $t$ 得单位时间的平均对数放大率。
- **极限 $t\to\infty$**：去掉瞬态与局部起伏，只保留长期渐近的、与初值几乎无关的值——这保证了 $\lambda$ 是系统（而非某条轨线）的固有属性。
- **几何含义**：$|\delta\mathbf{x}(t)| \approx |\delta\mathbf{x}(0)| e^{\lambda t}$——正 $\lambda$ 意味着**指数级的错误放大**。天气预报的「两周不可预测性」正是 $\lambda \approx 0.9$ 与「误差容忍阈值」的对数除法：$\frac{1}{0.9}\ln(\text{可容忍误差}/\text{初始误差})$。

## 6 小结

- **吸引子 vs 奇怪吸引子**：平凡吸引子（点/环/环面）之外，正 Lyapunov 指数 + 分形维数 = 奇怪吸引子。
- **Lyapunov 指数**：$\lambda>0$ 混沌、$\lambda<0$ 收敛、$\lambda=0$ 周期/准周期；重正交化算法可数值估计全谱。
- **分形维数**：盒维数、关联维数从时间序列即可估计，Lorenz 吸引子 $D_H \approx 2.06$。
- **耗散是关键**：体积收缩（Lorenz 的 $\mathrm{div}\,\mathbf{F}<0$）+ 方向拉伸 = 分形折叠，保守系统没有奇怪吸引子。

在下一节，我们将从连续时间切到离散时间：一个看似玩具的**logistic 映射**，如何用一张倍周期分岔图点亮整个混沌理论的烟火。
