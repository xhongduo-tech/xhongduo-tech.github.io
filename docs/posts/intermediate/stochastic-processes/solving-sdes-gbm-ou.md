---
title: 常见 SDE 的求解：几何布朗运动与 OU 过程
date: 2026-08-07
---

# 常见 SDE 的求解：几何布朗运动与 OU 过程

<div class="epigraph">
<p>把方程「积分因子」地一乘，随机方程便现出闭式——解 SDE 是 Itô 公式最锋利的用武之地。</p>
<footer>—— 安德雷 · 柯尔莫哥洛夫（Andrey Kolmogorov）</footer>
</div>

<div class="article-byline">
<p>第二级 · 随机过程 ｜ 张波《应用随机过程》§8.4 ｜ 2026-08-07</p>
</div>

## 解 SDE 的「心法」

大多数 SDE 没有闭式解，但有三个「幸运儿」——**几何布朗运动（GBM）**、**Ornstein-Uhlenbeck 过程（OU）**、**CIR 过程**——它们结构特殊，可以用 Itô 公式直接解出显式表达式。这三个模型覆盖了金融建模的半壁江山：GBM 管股价，OU 管利率/均值回复，CIR 管非负利率。

解 SDE 的标准心法只有一条：**猜一个「解的形状」，用 Itô 公式验证，再定参数**。常见的「形状」是取对数（GBM）或乘积分因子（OU）。<span class="marginnote">解 SDE 与解 ODE 的对照：<strong>ODE 用积分因子、分离变量；SDE 在这些动作上要加 Itô 修正</strong>——因为 $d(\ln X)$ 不再是 $dX/X$，还要减 $\frac12(dX)^2/X^2$。Itô 公式就是那个「修正后的微积分」。</span>

本节目标：完整求解 GBM 与 OU 过程，并掌握「对数化 / 积分因子」两种解法套路。

## 1 求解 GBM：对数化

**GBM 的 SDE**：
$$
dS = \mu S\, dt + \sigma S\, dB.
$$

**解法（对数化）**：设 $Y = \ln S$。由 Itô 公式（$g(x) = \ln x$，$g' = 1/x$，$g'' = -1/x^2$）：
$$
dY = \frac{1}{S} dS - \frac{1}{2S^2} (dS)^2 = \big(\mu - \frac{\sigma^2}{2}\big) dt + \sigma dB.
$$
**$Y$ 是带漂移布朗运动**（$dY$ 的系数不含 $S$！），积分得
$$
Y(t) = Y(0) + \big(\mu - \frac{\sigma^2}{2}\big) t + \sigma B(t).
$$
**结论**：
$$
S(t) = S_0 \exp\Big( \big(\mu - \frac{\sigma^2}{2}\big) t + \sigma B(t) \Big).
$$
**这与第七节的 GBM 定义完全吻合——Itô 公式把「定义」与「方程」缝合。**<span class="marginnote">「对数化」为什么行：<strong>$dS$ 的系数 $\mu S$、$\sigma S$ 正比于 $S$，取对数后系数变成常数</strong>——随机系数消掉了，$Y$ 成为常系数带漂移布朗。能对数化的 SDE 就叫「几何」型（GBM）；不能的就要其他套路。</span>

## 2 求解 OU：积分因子

**Ornstein-Uhlenbeck 过程（OU）**：
$$
dX = \theta\, (\alpha - X)\, dt + \sigma\, dB.
$$
$\theta > 0$ 是**均值回复速度**，$\alpha$ 是**长期均值**，$\sigma$ 是波动。**当 $X$ 高于 $\alpha$ 时，漂移 $\theta(\alpha - X)$ 为负，把它拉回；低于 $\alpha$ 时为正，推它上去——「均值回复」机制。**

**解法（积分因子）**：乘 $e^{\theta t}$：
$$
d\big( e^{\theta t} X \big) = e^{\theta t}\big( dX + \theta X dt \big) = e^{\theta t} \big( \theta\alpha\, dt + \sigma dB \big).
$$
（这里用乘积法则或直接验算。）积分：
$$
e^{\theta t} X(t) = X(0) + \alpha\big( e^{\theta t} - 1 \big) + \sigma \int_0^t e^{\theta s}\, dB(s).
$$
**结论**：
$$
X(t) = X(0) e^{-\theta t} + \alpha \big( 1 - e^{-\theta t} \big) + \sigma e^{-\theta t} \int_0^t e^{\theta s}\, dB(s).
$$
**$X(t)$ 是「初值衰减」+「均值吸引」+「噪声累积」三部分之和。**<span class="marginnote">积分因子的直觉：<strong>$e^{\theta t}$ 把「均值回复」变成「常数漂移」</strong>——回复项 $\theta X dt$ 乘上积分因子后恰好被吸收进全微分。这和对 $y' + \theta y = f$ 用积分因子解 ODE 一模一样，只是多出的 Itô 积分项 $\sigma \int e^{\theta s} dB(s)$ 是随机贡献。</span>

## 3 OU 的性质：均值回复与平稳分布

由解式算矩（Itô 积分期望 0、方差由等距）：

**均值**：
$$
E[X(t)] = X(0)e^{-\theta t} + \alpha(1 - e^{-\theta t}) \;\xrightarrow{t\to\infty}\; \alpha.
$$
**均值指数回复到 $\alpha$，回复速率 $\theta$。**

**方差**：
$$
\mathrm{Var}(X(t)) = \frac{\sigma^2}{2\theta}\big( 1 - e^{-2\theta t} \big) \;\xrightarrow{t\to\infty}\; \frac{\sigma^2}{2\theta}.
$$
**方差收敛到常数 $\sigma^2/(2\theta)$——OU 有稳态方差。**

**平稳分布**：$X(t)$ 渐近服从
$$
\mathrm{Normal}\Big( \alpha,\; \frac{\sigma^2}{2\theta} \Big).
$$
**OU 是「带高斯平稳分布」的扩散——它绕着 $\alpha$ 做高斯波动，波动幅度由波动率与回复速度之比决定。**<span class="marginnote">OU 的平稳分布同时说明它的应用地位：<strong>它是 Vasicek 利率模型（第十篇）的引擎，也是统计物理里朗之万方程的连续极限</strong>。「回复到均值 + 高斯噪声」几乎是所有「围绕均衡波动」现象的通用模型。</span>

## 4 公式解析：验证 OU 解式满足方程

**目标：把 OU 的解式代回 SDE，用 Itô 公式确认它是真正的解。**

第一步，写解式并求微分。$X(t) = X(0)e^{-\theta t} + \alpha(1-e^{-\theta t}) + \sigma e^{-\theta t} Z(t)$，其中 $Z(t) = \int_0^t e^{\theta s} dB(s)$（Itô 积分）。

第二步，对 $Z$ 用 Itô 公式。$dZ = e^{\theta t} dB$（被积函数确定、可直接读微分），且 $Z$ 是鞅（Itô 积分鞅性）。

第三步，求 $dX$。对「$\sigma e^{-\theta t} Z$」用乘积法则：
$$
d(\sigma e^{-\theta t} Z) = \sigma e^{-\theta t} dZ - \theta \sigma e^{-\theta t} Z\, dt = \sigma dB - \theta \cdot \sigma e^{-\theta t} Z\, dt.
$$
第四步，合并非随机项。$d(\text{确定部分}) = -\theta X(0)e^{-\theta t}dt + \theta\alpha e^{-\theta t}dt$。相加：
$$
dX = \theta(\alpha - X) dt + \sigma dB.
$$
**恰好回到原方程——验证成功。**

**这个推导为什么重要**：它演示了「构造解 + 用 Itô 公式验证」的标准闭环——**解 SDE 的最后一步永远是验证**。掌握了「积分因子 + 乘积法则 + 鞅项微商」这套动作，OU 类方程（含 Vasicek、CIR）的求解就是流水线。

## 5 解法套路对照

| 方程类型 | 系数结构 | 套路 | 解的形态 |
| --- | --- | --- | --- |
| GBM $dS = \mu S dt + \sigma S dB$ | 系数 $\propto S$ | 对数化 | 指数（对数正态） |
| OU $dX = \theta(\alpha - X)dt + \sigma dB$ | 回复项线性 | 积分因子 | 高斯（均值回复） |
| CIR $dX = \theta(\alpha - X)dt + \sigma\sqrt X dB$ | 扩散 $\propto\sqrt X$ | 无闭式 | 非中心卡方（非负） |

**GBM 与 OU 是「能闭式求解」的两座山头；CIR 有解析的转移密度但无简单路径解。**<span class="marginnote">学习地图：<strong>掌握 GBM 的对数化与 OU 的积分因子，就掌握了解 SDE 的两大核心动作</strong>。CIR、Heston 等只是它们的变体——要么多个 $\sqrt X$，要么多个状态变量，处理工具仍是 Itô 公式。</span>

## 6 小结

- **GBM** $dS = \mu S dt + \sigma S dB$：取对数 ⟹ $S(t) = S_0 e^{(\mu-\sigma^2/2)t + \sigma B(t)}$。
- **OU** $dX = \theta(\alpha - X)dt + \sigma dB$：积分因子 $e^{\theta t}$ ⟹ 解式 = 初值衰减 + 均值吸引 + 噪声累积。
- **OU 性质**：$E \to \alpha$、$\mathrm{Var} \to \sigma^2/(2\theta)$、平稳分布 $N(\alpha, \sigma^2/2\theta)$。
- **验证闭环**：构造解 + Itô 公式 + 乘积法则——最后一步永远验证。
- 套路：对数化（几何型）/ 积分因子（线性回复型）；CIR 是 OU + $\sqrt X$ 扩散。

**数值验证**：OU 的平稳分布 $N(\alpha, \sigma^2/2\theta)$ 可直接用模拟检验——欧拉-丸山离散化跑长轨道，画直方图对比理论密度。两个参数的分工一目了然：$\theta$ 大则轨道紧贴均值（回复快），$\sigma$ 大则散开（波动强）。「模拟验证解析解」是随机过程建模的收尾动作，本专题从泊松到 OU 一路如此。

到这里，第八篇《随机积分初步》全部结束。从下一篇起，我们把视角从「演化」转向「平稳」：**平稳过程**——统计特性不随时间平移而改变的随机过程。
