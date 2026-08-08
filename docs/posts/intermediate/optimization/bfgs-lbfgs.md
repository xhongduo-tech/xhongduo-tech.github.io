---
title: BFGS 公式与有限内存 L-BFGS
date: 2026-08-07
---

# BFGS 公式与有限内存 L-BFGS

<div class="epigraph">
<p>记住最近的几步，就足够走出很远。</p>
<footer>—— 类比自 L-BFGS 的有限内存哲学</footer>
</div>

<div class="article-byline">
<p>第二级 · 最优化理论 ｜ 《最优化方法》无约束优化章、Nocedal & Wright §6 ｜ 2026-08-07</p>
</div>

## 为什么 BFGS 成了拟牛顿的「默认王者」

DFP 打开了拟牛顿的大门，但数值上常不够稳。**BFGS（Broyden–Fletcher–Goldfarb–Shanno）**几乎在同一时间被四人独立发现，它对**Hessian 本身**（而非其逆）做更新，数值稳定性和收敛表现全面超越 DFP，成为教科书与工程库的默认拟牛顿法。<span class="marginnote">而<strong>L-BFGS（limited-memory BFGS）</strong>进一步解决 BFGS 的存储问题：$B_k$（或 $H_k$）是稠密 $n\times n$ 矩阵，$n=10^6$ 时存不下。L-BFGS 干脆<strong>不显式存矩阵</strong>，只存最近 $m$ 步的 $(s_i, y_i)$ 对（$m$ 常取 5–20），用一套精巧的「两循环递归」直接算方向——内存从 $O(n^2)$ 降到 $O(mn)$。这就是今天大规模无约束优化的主力算法（也是许多 LLM 微调器的基础）。</span>

## 1 BFGS 更新公式

BFGS 更新的是 Hessian 近似 $B_k$（满足 $B_{k+1}s_k = y_k$）：

$$
B_{k+1} = B_k - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k} + \frac{y_k y_k^T}{y_k^T s_k}
$$

对照 DFP（更新 $H$ 且第一项分母是 $y^THy$），BFGS 把「角色互换」：$B$ 与 $H$、$s$ 与 $y$ 对调。<span class="marginnote">记忆技巧：BFGS 公式把 DFP 里的 $H \to B$、$y \to s$、$s \to y$ 全部互换即得。两者在数学上是「对偶」的：一个近似 Hessian，一个近似其逆。BFGS 胜在数值稳定——即使 $B_0$ 离谱，更新也不太容易放大病态。</span>实际使用时需要的是方向 $d = -B^{-1}\nabla f$，所以常写成**逆形式**

$$
H_{k+1} = \left(I - \rho_k s_k y_k^T\right) H_k \left(I - \rho_k y_k s_k^T\right) + \rho_k s_k s_k^T, \qquad \rho_k = \frac{1}{y_k^Ts_k}
$$

这个形式是后面 L-BFGS「两循环」的地基。BFGS 的收敛性质：超线性，且对正定二次函数有限步（$n$ 步内）精确收敛。

## 2 L-BFGS：不存矩阵，只存记忆

L-BFGS 的核心观察：**BFGS 的 $H_k$ 由初始 $H_0$ 与最近的所有 $(s_i,y_i)$ 决定**。若只保留最近 $m$ 对，$H_k$ 就被「截断」成：

$$
H_k \approx \text{从 } H_0 \text{ 与最近 } m \text{ 对 } (s_i, y_i) \text{ 递归构造}
$$

迭代时更新列表：新对 $(s_k,y_k)$ 加入，最老的对被丢弃——一个滚动窗口。<span class="marginnote">$m$ 是「记忆长度」：太小（如 $m=1$）近似太粗，接近梯度下降；太大（如 $m=100$）逼近全 BFGS 但内存与计算随 $m$ 涨。经验值 $m \in [5, 20]$，多数问题 $m=10$ 左右性能最佳。这是 L-BFGS 唯一重要的超参数。</span>初始 $H_0$ 常取对角缩放 $\gamma_k I$，$\gamma_k = \frac{s_{k-1}^Ty_{k-1}}{y_{k-1}^Ty_{k-1}}$——用最近一步的曲率比例做整体缩放，几乎零成本地大幅加速。

## 3 两循环递归（two-loop recursion）

L-BFGS 计算方向 $d_k = -H_k\nabla f_k$ 时，**不构造 $H_k$**，而是用「两循环递归」从记忆里直接算出 $H_k\nabla f_k$。算法（记 $\rho_i = 1/(y_i^Ts_i)$）：

```text
q = ∇f_k
// 第一循环（反向：从最新到最旧）
for i = k−1, …, k−m:
    α_i = ρ_i · s_iᵀ q
    q ← q − α_i y_i
// 中间步：用对角缩放 H₀ = γ_k I
r = H₀ q
// 第二循环（正向：从最旧到最新）
for i = k−m, …, k−1:
    β = ρ_i · y_iᵀ r
    r ← r + (α_i − β) s_i
返回搜索方向 d_k = −r
```

**第一循环（反向）**：沿存储的 $(s_i,y_i)$ 从新到旧，「解压」梯度，把最新的曲率信息加权进去。
**中间**：用 $H_0$（对角缩放）做初始作用。
**第二循环（正向）**：从旧到新把权重「重新组合」，得到 $H_k\nabla f_k$ 的精确乘积。<span class="marginnote">两循环递归是 L-BFGS 的「发动机」：它把「乘以 $H_k$」从 $O(n^2)$（矩阵乘法）降到 $O(mn)$（$2m$ 次内积与缩放），且<strong>不需要任何矩阵存储</strong>。对 $n=10^6$、$m=10$，每步只需 $O(10^7)$ 次运算——正是大规模问题的量级。正确性：递归输出的就是完整 BFGS 更新 $m$ 次后的 $H_k\nabla f_k$（若 $H_0$ 为初始矩阵）。</span>

## 4 公式解析：L-BFGS 方向 = 截断 BFGS 方向

把「两循环递归没算错」验证一下：它输出的 $r$ 必须等于「完整 BFGS 更新 $m$ 次后的 $H_k$」作用在 $\nabla f_k$ 上。核心是观察 BFGS 逆更新是一个**秩 2 复合**：

$$
H_{k+1} = T_k H_k T_k^T + \rho_k s_k s_k^T, \qquad T_k = I - \rho_k s_k y_k^T
$$

- **第一步，$H_k$ 的展开**：把逆更新从 $k-1$ 往前迭代展开，$H_k$ 是「初始 $H_0$ 被一系列 $T$ 变换 + 一系列秩 1 项」的复合。
- **第二步，作用在梯度上**：$H_k\nabla f$ 展开后是若干「$T$ 链 + 外积项」作用于 $\nabla f$ 的和。
- **第三步，递归的精妙**：第一循环反向推进时，$\alpha_i$ 累积「新信息加权」，得到 $q$；第二循环正向时把 $T$ 链的转置作用补上——**两条循环正好走完 $H_k\nabla f$ 展开式的全部项**，且每项只用到已存的 $(s_i,y_i)$。

**结论：两循环递归是「展开 BFGS 更新、去掉中间矩阵」的代数值化简**——同样的数学，$O(mn)$ 的实现。

## 5 L-BFGS 的工程地位

L-BFGS 是无约束/约束优化里「性价比」最高的算法之一，四处可见：

- **大规模最小二乘与逻辑回归**：scikit-learn、Liblinear 的默认求解器。
- **深度学习的某些场景**：全批量（非 SGD）小模型、超参数调优、泛化较好的结构化目标。
- **约束优化的子问题**：SQP 与增广拉格朗日法的内层常用 L-BFGS。
- **与线搜索配合**：Wolfe 线搜索保证 $s_k^Ty_k > 0$，是 L-BFGS 稳定性的前提。<span class="marginnote">L-BFGS 与 SGD 家族的分工很清晰：<strong>L-BFGS 适合「能算全批量梯度」的中大规模问题、精度要求高</strong>；SGD 适合「样本海量、梯度有噪声」的深度学习。前者确定性强、收敛快；后者扛噪声、内存恒定。理解这条分界线，选求解器就有了准绳。</span>

## 6 小结

- **BFGS**：更新 Hessian 近似 $B_{k+1} = B_k - \frac{B_ks_ks_k^TB_k}{s_k^TB_ks_k} + \frac{y_ky_k^T}{y_k^Ts_k}$，超线性、数值稳，拟牛顿默认。
- **逆形式**：$H$ 的秩 2 复合更新，是 L-BFGS 的地基。
- **L-BFGS**：只存最近 $m$ 对 $(s_i,y_i)$，内存 $O(mn)$，$m \in [5,20]$。
- **两循环递归**：不构造矩阵，直接算 $H_k\nabla f$，$O(mn)$ 一步。
- **工程地位**：大规模无约束优化的主力；与 SGD 按「能否全批量算梯度」分工。

在下一节，我们进入无约束优化的另一条大路——**共轭梯度法：线性共轭梯度**。
