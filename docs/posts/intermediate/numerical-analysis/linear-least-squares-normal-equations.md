---
title: 矛盾方程组与线性最小二乘问题
date: 2026-08-07
---

# 矛盾方程组：方程比未知数多时怎么办

<div class="epigraph">
<p>当你无法精确满足所有约束时，最好的解是让所有约束「都差一点点」而不是「一个完全满足、其余全错」。</p>
<footer>—— 最小二乘的哲学</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§3.8 ｜ 2026-08-07</p>
</div>

## 为什么从矛盾方程组开始

前面的最小二乘是从「数据拟合」进入的；这一节我们从**线性代数的纯粹视角**再走一遍：方程组 $A\mathbf{x}=\mathbf{b}$ 有 $m$ 个方程、$n$ 个未知数，$m>n$（**超定方程组 / 矛盾方程组**）。绝大多数时候无精确解——$\mathbf{b}$ 不在 $A$ 的列空间里。这类「无解」的方程组在工程里多如牛毛：GPS 定位（冗余卫星）、相机标定（冗余观测）、线性回归（样本多于参数）。**线性最小二乘问题**研究的就是：无精确解时，找使残差范数最小的 $\mathbf{x}$。<span class="marginnote">「矛盾方程组」的名字来自旧教材：方程之间互相打架、无一致解。但「无解」不等于「没意义」——<strong>每个工程测量都自带冗余，正是冗余让噪声可被平均掉</strong>。把「矛盾」变成「最小二乘」，是从「对错」走向「好坏」的思维跃迁。</span>

本节统一三套视角：几何（正交投影）、代数（法方程）、数值（QR/SVD 分解）。

## 1 问题提法与几何图像

**线性最小二乘问题（linear least squares problem）**：给定 $A\in\mathbb{R}^{m\times n}$（$m>n$）与 $\mathbf{b}\in\mathbb{R}^m$，求 $\mathbf{x}^*\in\mathbb{R}^n$ 使

$$
\lVert \mathbf{b} - A\mathbf{x}^*\rVert_2 = \min_{\mathbf{x}\in\mathbb{R}^n}\lVert \mathbf{b} - A\mathbf{x}\rVert_2
$$

**几何图像**：$A$ 的列张成 $\mathbb{R}^m$ 的一个 $n$ 维子空间 $\mathrm{Col}(A)$。$\mathbf{b}$ 一般不在 $\mathrm{Col}(A)$ 里。$A\mathbf{x}$ 遍历 $\mathrm{Col}(A)$，要找离 $\mathbf{b}$ 最近的那个点——**正是 $\mathbf{b}$ 在 $\mathrm{Col}(A)$ 上的正交投影 $\hat{\mathbf{b}}=\mathrm{proj}_{\mathrm{Col}(A)}\mathbf{b}$**。残差 $\mathbf{r}=\mathbf{b}-A\mathbf{x}^*$ 垂直于 $\mathrm{Col}(A)$，即

$$
A^\top(\mathbf{b} - A\mathbf{x}^*) = \mathbf{0}
$$

**这是最小二乘解的完整几何图景：解存在 ⇔ 投影存在；投影唯一 ⇔ $A$ 列满秩。**<span class="marginnote">在线性代数（第二级）里你已经见过 $\hat{\mathbf{y}}=A(A^\top A)^{-1}A^\top\mathbf{y}$ 这个投影公式；数值分析补充的是<strong>「怎么算得稳」</strong>：直接套公式会病态，QR 分解才是工程正解。</span>

## 2 法方程与正规解：何时有解、解如何表示

由正交条件 $A^\top(A\mathbf{x}^*-\mathbf{b})=0$ 得**法方程**

$$
(A^\top A)\mathbf{x}^* = A^\top \mathbf{b}
$$

- 若 $A$ **列满秩**（$\mathrm{rank}(A)=n$），$A^\top A$ 正定，唯一解 $\mathbf{x}^*=(A^\top A)^{-1}A^\top\mathbf{b}$。
- 若 $A$ **列亏秩**，$A^\top A$ 奇异，法方程有无穷多解；工程上取**极小范数解** $\mathbf{x}^+=A^+\mathbf{b}$，其中 $A^+$ 是**伪逆（pseudoinverse）**。

**正规解（normal solution）** 指在全部最小二乘解中范数最小者，它存在且唯一。伪逆的稳定性最强，但数值上直接算伪逆也病态。<span class="marginnote">伪逆 $A^+$ 由摩尔-彭罗斯（Moore-Penrose）公理唯一确定，可通过 SVD 稳定计算：$A=U\Sigma V^\top$ 时 $A^+=V\Sigma^+U^\top$。<strong>SVD 同时给出「解」与「解的可信度」（奇异值即放大倍数）</strong>——是秩亏问题的最终裁判。</span>

**条件数与解的可靠性：** $\mathrm{cond}(A)$ 决定 $\mathbf{x}^*$ 对 $\mathbf{b}$ 扰动的敏感度。注意**法方程的条件数是 $\mathrm{cond}(A)^2$**——直接解法方程，条件数被平方恶化，这就是为什么数值上要绕开法方程。

## 3 公式解析：三种算法与病态的博弈

解最小二乘有三种路径，稳定性递增：

| 方法 | 运算 | 条件数 | 适用 |
| --- | --- | --- | --- |
| 法方程 | 算 $A^\top A$ 再 Cholesky | $\mathrm{cond}(A)^2$ | 仅良态小问题 |
| **QR 分解** | $A=QR$，解 $R\mathbf{x}=Q^\top\mathbf{b}$ | $\mathrm{cond}(A)$ | 标准选择 |
| **SVD** | $A=U\Sigma V^\top$，$\mathbf{x}=V\Sigma^{-1}U^\top\mathbf{b}$ | $\mathrm{cond}(A)$，可诊断秩亏 | 病态/秩亏 |

**为什么 QR 是标准选择？** 拆解 QR 路径：

- **第一步，正交化。** 对 $A$ 做 QR 分解 $A=QR$，其中 $Q\in\mathbb{R}^{m\times n}$ 列正交（$Q^\top Q=I$），$R$ 是上三角。残差范数平方：

$$
\lVert \mathbf{b}-A\mathbf{x}\rVert_2^2 = \lVert Q^\top\mathbf{b}-R\mathbf{x}\rVert_2^2 + \lVert \text{正交补分量的范数} \rVert_2^2
$$

- **第二步，正交补分量与 $\mathbf{x}$ 无关。** $\mathbf{b}$ 分解成「$Q$ 列空间里的部分」$QQ^\top\mathbf{b}$ 与「正交补部分」$\mathbf{b}-QQ^\top\mathbf{b}$。后者无法被 $A\mathbf{x}$ 触及，最小化时不起作用。
- **第三步，解三角方程组。** 剩下要最小化 $\lVert Q^\top\mathbf{b}-R\mathbf{x}\rVert_2^2$，即解上三角系统 $R\mathbf{x}^*=Q^\top\mathbf{b}$——回代即可。**QR 路径没有平方条件数，还顺便给出正交基。**

**残差的几何恒等式**：$\lVert\mathbf{b}-A\mathbf{x}^*\rVert_2^2 = \lVert\mathbf{b}\rVert_2^2 - \lVert A\mathbf{x}^*\rVert_2^2$（勾股定理），工程上用来核对实现。

## 4 数值例子：QR 路径 vs 法方程路径

用一个小例子演示两条路径的差别。数据点 $(0,1),(1,2),(2,3.1),(3,3.9)$，线性拟合 $y=a_0+a_1x$。设计矩阵 $A=\begin{pmatrix}1&0\\1&1\\1&2\\1&3\end{pmatrix}$，$\mathbf{b}=\begin{pmatrix}1\\2\\3.1\\3.9\end{pmatrix}$。

法方程路径：$A^\top A=\begin{pmatrix}4&6\\6&14\end{pmatrix}$，$A^\top\mathbf{b}=\begin{pmatrix}10\\21.1\end{pmatrix}$。解 $a_0=1.03,\ a_1=0.975$（约）。

QR 路径：对 $A$ 做 QR（过程略），回代得到同样的 $a_0,a_1$。**良态下两条路径数值一致；病态下（数据范围大、$n$ 高）法方程开始失真，QR 保持稳定。**

```python
import numpy as np

# 良态小例子：直线拟合，两条路径给出同一组系数
x = np.array([0., 1., 2., 3.])
y = np.array([1., 2., 3.1, 3.9])
A = np.vstack([np.ones_like(x), x]).T          # 设计矩阵

a_normal = np.linalg.solve(A.T @ A, A.T @ y)   # 法方程
Q, R = np.linalg.qr(A)                          # QR（reduced）
a_qr = np.linalg.solve(R, Q.T @ y)              # 回代
print(a_normal, a_qr)                           # 同一组系数（约 [1.03, 0.98]）

# 病态对照：窄区间上的高次拟合，法方程条件数平方恶化
t = np.linspace(1.0, 1.2, 15)                   # 区间窄 → Vandermonde 病态
V = np.vander(t, 10, increasing=True)           # 次数 0..9
b = np.sin(t)
c_normal = np.linalg.solve(V.T @ V, V.T @ b)    # 法方程：cond(A)²
c_qr = np.linalg.lstsq(V, b, rcond=None)[0]     # QR/SVD：cond(A)
print(np.linalg.cond(V), np.linalg.cond(V.T @ V))   # cond(A) 与 cond(A)²
print(np.linalg.norm(c_normal - c_qr))          # 两条路径的系数开始分叉
```

两条路径结果一致，但**在病态例子上跑一遍就会发现法方程开始出现误差——这就是「为什么不直接解法方程」的实战理由**。<span class="marginnote">演示病态差异的经典例子：拟合高次多项式时（如 $n=15$），法方程路径的系数可能偏离真实几个数量级，而 $A^\top A$ 依旧可靠。<strong>凡是方程数目大的拟合，一律 $A^\top A$ / QR / SVD，别手写 $A^\top A$。</strong></span>

## 5 辨析：超定、欠定与适定的谱系

| 情形 | 方程数 $m$ vs 未知数 $n$ | 解的情况 | 处理 |
| --- | --- | --- | --- |
| 适定 | $m=n$ | 唯一解（若 $A$ 可逆） | 高斯消去/LU |
| **超定（矛盾）** | $m>n$ | 一般无解，有最小二乘解 | QR / SVD / 法方程 |
| 欠定 | $m<n$ | 无穷多解，有极小范数解 | 伪逆 $A^+$ |

**辨析｜易错点：** 超定 ≠ 无意义。超定方程组的「最优解」不是让方程「几乎全满足」，而是让**残差平方和最小**——残差不会全部为零，而是被「摊到每个方程上」。「方程组无解」在工程语言里是「观测有噪声，取最小二乘」；在纯数学里是「$\mathbf{b}\notin\mathrm{Col}(A)$」。两种读法同一件事。

## 6 小结

- **线性最小二乘**：$A\mathbf{x}\approx\mathbf{b}$（$m>n$）无精确解时，求残差范数最小的 $\mathbf{x}$；解 = $\mathbf{b}$ 在 $\mathrm{Col}(A)$ 上的正交投影。
- **法方程** $A^\top A\mathbf{x}^*=A^\top\mathbf{b}$：条件数平方恶化，仅适合良态小问题。
- **QR 路径** $R\mathbf{x}^*=Q^\top\mathbf{b}$：条件数保持 $\mathrm{cond}(A)$，标准选择；**SVD** 处理秩亏并给诊断。
- 列亏秩时解不唯一，取**极小范数解** $\mathbf{x}^+=A^+\mathbf{b}$（伪逆）。
- 适定、超定、欠定分别对应唯一解、最小二乘解、极小范数解——矩阵的「形状」决定解的「性质」。

至此，函数逼近与曲线拟合的十章写完了。下一章，我们开始一个全新的主题：**数值积分与数值微分**——从插值思想出发，把定积分与导数也变成「可算的近似」。
