---
title: 二次规划的典型应用：SVM 与投资组合优化
date: 2026-08-07
---

# 二次规划的典型应用：SVM 与投资组合优化

<div class="epigraph">
<p>最好的分类边界与最优的资产配比，都是同一个二次规划的化身。</p>
<footer>—— 类比自 QP 在机器学习与金融中的双栖地位</footer>
</div>

<div class="article-byline">
<p>第二级 · 最优化理论 ｜ 《最优化方法》二次规划章、Boyd《Convex Optimization》§8 ｜ 2026-08-07</p>
</div>

## 为什么用两个「完全不同」的应用收尾第七篇

SVM 与投资组合优化一个在机器学习、一个在金融，看似八竿子打不着——但它们**共享同一个数学模型**：凸二次规划。本节把两个问题都「翻译」成 QP，展示 QP 理论（积极集、内点、KKT、灵敏度）如何在同一套语言里解决两类真实世界问题。<span class="marginnote">这种「不同领域、同一模型」的现象正是凸优化的魅力：建模把问题还原成标准形式，剩下的交给通用算法。SVM 与 Markowitz 组合既是 QP 理论的招牌案例，也是「从极限到大模型」里从「数学工具」到「产业应用」的桥——理解它们，你就看到了优化如何成为机器学习的肌肉与金融的风控器。</span>

## 1 软间隔 SVM：原始 QP

给定数据 $(x_i, y_i)$（$y_i \in \{\pm1\}$），软间隔 SVM 找分类超平面 $w^Tx + b = 0$，最大化间隔同时容忍少量误分类：

$$
\min_{w, b, \xi}\ \frac12\|w\|_2^2 + C\sum_{i=1}^{m}\xi_i \quad \mathrm{s.t.}\quad y_i(w^Tx_i + b) \ge 1 - \xi_i,\ \ \xi_i \ge 0
$$

- **目标**：$\frac12\|w\|^2$ 是间隔倒数的平方（间隔越大越好）；$C\sum\xi_i$ 惩罚「软间隔违反量」。
- **约束**：$y_i(w^Tx_i+b) \ge 1 - \xi_i$ 要求样本「正确且离边界至少 1」，$\xi_i$ 允许少量放松。
- **QP 结构**：$P = \begin{bmatrix}I&0\\0&0\end{bmatrix}$（半正定）、约束全线性——**凸 QP**。<span class="marginnote">变量是 $(w, b, \xi)$，$w$ 的维数等于特征维数——特征很多时原始 QP 的变量很多。<strong>SVM 的经典做法是转对偶 QP</strong>：对偶变量 $\alpha_i$ 的个数 = 样本数，且只通过内积 $x_i^Tx_j$ 出现——这为<strong>核技巧</strong>打开大门（把内积换成核函数）。这是「对偶」在机器学习里最著名的应用。</span>

## 2 SVM 的对偶 QP 与支持向量

写出 SVM 的拉格朗日对偶（推导略，用第三篇对偶理论的机械流程）：

$$
\max_\alpha\ \sum_i \alpha_i - \frac12\sum_{i,j}\alpha_i\alpha_j y_i y_j x_i^Tx_j \quad \mathrm{s.t.}\quad \sum_i\alpha_i y_i = 0,\ \ 0 \le \alpha_i \le C
$$

这是**盒子约束 + 一个等式约束的 QP**（$P_{ij} = y_iy_jx_i^Tx_j$ 半正定）。解出 $\alpha^*$ 后：

- **支持向量**：$\alpha_i^* > 0$ 的样本在间隔边界上——由互补松弛 $\alpha_i(y_i(w^Tx_i+b) - 1 + \xi_i) = 0$ 决定。<span class="marginnote">互补松弛在这里「选择」了支持向量：$\alpha_i^* = 0$ 的样本在间隔外（不影响解），$0 < \alpha_i^* < C$ 的在边界上，$\alpha_i^* = C$ 的在误分类区。<strong>支持向量个数通常远小于样本数</strong>——这就是 SVM「稀疏表示」的来源，也是核方法高效的原因。</span>
- **决策函数**：$f(x) = \sum_i \alpha_i^* y_i \langle x_i, x\rangle + b^*$——**只需支持向量**参与预测。

**求解**：小规模用积极集法（SMO 是专为 SVM 优化的坐标下降变体），大规模用内点法。

## 3 Markowitz 投资组合：均值-方差 QP

**均值-方差组合优化（Markowitz）**在收益与风险间权衡。设 $x_i$ 是第 $i$ 种资产的投资权重，$\mu$ 为期望收益向量，$\Sigma$ 为协方差矩阵（$\Sigma \succeq 0$）：

$$
\min_x\ \frac12 x^T\Sigma x - \lambda\, \mu^T x \quad \mathrm{s.t.}\quad \mathbf{1}^Tx = 1,\ \ x \ge 0
$$

- **目标**：$\frac12x^T\Sigma x$ 是组合方差（风险），$-\lambda\mu^Tx$ 是收益项（负号表示最大化），$\lambda$ 是风险厌恶系数。
- **约束**：$\mathbf{1}^Tx = 1$（权重和为一）、$x \ge 0$（不许做空，可选）。<span class="marginnote">$\lambda$ 是「风险-收益」权衡的旋钮：$\lambda = 0$ 只最小风险（得最小方差组合），$\lambda \to \infty$ 只追收益。$\lambda$ 扫过整个范围，最优解画出<strong>有效前沿（efficient frontier）</strong>——这是 Markowitz 1952 年诺奖工作的核心图形，也是现代资产配置的理论地基。</span>
**QP 结构**：$\Sigma \succeq 0$ ⇒ 凸 QP；**唯一性**：$\Sigma \succ 0$（资产非冗余）时严格凸、解唯一。

## 4 公式解析：把两个问题对齐到同一个 QP 骨架

把 SVM 对偶与 Markowitz 并排，看它们共享的数学内核：

| 部件 | SVM 对偶 QP | Markowitz QP |
| --- | --- | --- |
| 变量 | $\alpha$（对偶变量） | $x$（权重） |
| 二次矩阵 $P$ | $y_iy_jx_i^Tx_j$（半正定） | $\Sigma$（半正定） |
| 线性项 $q$ | $-\mathbf{1}$ | $-\lambda\mu$ |
| 等式约束 | $\sum\alpha_iy_i = 0$ | $\mathbf{1}^Tx = 1$ |
| 盒子约束 | $0 \le \alpha \le C$ | $x \ge 0$ |
| 解的含义 | 支持向量权重 | 资产配置权重 |

**第一步，对齐**：两个问题都是「半正定二次目标 + 线性/盒子约束」——同一个 QP 模板。
**第二步，KKT 的角色**：SVM 用互补松弛挑支持向量，Markowitz 用互补松弛判断「哪些资产值得持有」（$x_i > 0$ 的资产 vs $x_i = 0$ 的空仓）。
**第三步，灵敏度**：SVM 的 $C$、Markowitz 的 $\lambda$ 都是「超参数」——灵敏度分析回答「调一格 $C$ 或 $\lambda$，解怎么变」，与第四篇的影子价格同源。<span class="marginnote">这条「同构表」是 QP 应用方法论的核心：<strong>看到一个新问题，先识别它的 $P, q, A, b$</strong>，一旦落入「凸二次 + 线性约束」框架，全部 QP 理论立即生效——全局最优、有效算法、灵敏度解读。SVM 与 Markowitz 只是这张表的两行。</span>

## 5 工程与延伸

**核技巧**：SVM 对偶只依赖内积 $\langle x_i,x_j\rangle$，换成核 $k(x_i,x_j)$ 即得非线性分类——QP 骨架不变，只是 $P$ 换成核矩阵（仍半正定）。
**大规模**：SMO 分解法逐个坐标更新，避免整矩阵分解——上百万样本可行。
**现代延伸**：投资组合加交易成本、换手率惩罚仍是 QP；SVM 的替代（hinge 损失的邻近梯度）则走向一阶法——但 QP 形式仍是「精确、可解释」的标杆。<span class="marginnote">在「从极限到大模型」里，这两条线都通向深处：SVM → 核方法 → 表示学习；Markowitz → 风险平价 → 智能贝塔。它们的共同起点都是今天这节 QP——一个「不同领域共用同一算法」的绝佳示范。</span>

## 6 小结

- **SVM**：原始 QP（$w, b, \xi$）与对偶 QP（$\alpha$）；对偶开启核技巧。
- **支持向量**：互补松弛选出的 $\alpha_i^* > 0$ 样本——稀疏表示。
- **Markowitz**：$\frac12x^T\Sigma x - \lambda\mu^Tx$，约束 $\mathbf{1}^Tx = 1$、$x\ge0$。
- **同构**：两个问题共享「半正定二次 + 线性约束」骨架，一张表对齐。
- **方法论**：识别 $P,q,A,b$ → 落入 QP 框架 → 全套理论即时生效。

在下一节，我们进入第八篇，走向离散世界——**整数规划建模：0-1 变量与逻辑约束**。
