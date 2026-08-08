---
title: Christoffel 记号与联络系数
date: 2026-08-07
---

# Christoffel 记号与联络系数

<div class="epigraph">
<p>坐标系的弯曲要用一套记号来补偿——这就是 Christoffel 记号的工作。</p>
<footer>—— 埃尔温 · 布鲁诺 · 克里斯托费尔（Elwin Bruno Christoffel）</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分几何 ｜ 陈维桓《微分几何》§4.4 ｜ 2026-08-07</p>
</div>

## 为什么从 Christoffel 记号开始

协变导数是「曲面上的求导」，但它的坐标公式里藏着一堆神秘的符号 $\Gamma^k_{ij}$——这就是 **Christoffel 记号（Christoffel symbols）**。它们是坐标基向量二阶导数的切向分量，在坐标下编码了「坐标系如何弯曲」。

为什么值得为这套记号单开一节？因为**整个内蕴几何的计算都要靠它**：测地线方程、平行移动、高斯曲率的内蕴公式、Gauss-Codazzi 方程——全部以 $\Gamma$ 为脚手架。而且 $\Gamma$ 有一个优美的性质：**它们完全由第一基本形式（度量）决定**——再一次证明「度量决定内蕴几何」。<span class="marginnote">Christoffel（1829—1900）是德国数学家，1869 年发表这套记号。它们的本质：坐标基向量 $\mathbf{x}_u,\mathbf{x}_v$ 是「会动」的（随位置变化），二阶导 $\mathbf{x}_{uu}$ 展开到切平面上需要一套系数。在现代黎曼几何里，$\Gamma^k_{ij}$ 就是「联络」在坐标基下的分量——「联络系数」的名字由此而来。</span>

## 1 坐标基的二阶导：问题的起点

坐标基 $\{\mathbf{x}_u, \mathbf{x}_v\}$ 随位置变化（曲面弯曲），所以二阶偏导 $\mathbf{x}_{uu}, \mathbf{x}_{uv}, \mathbf{x}_{vv}$ 一般不为零。把它们**分解到切平面与法向**：

$$
\mathbf{x}_{uu} = \Gamma^1_{11}\,\mathbf{x}_u + \Gamma^2_{11}\,\mathbf{x}_v + \text{法向分量}
$$

**定义（Christoffel 记号）**：$\mathbf{x}_{ij}$（$i,j \in \{1,2\}$，表示对第 $i$、第 $j$ 个坐标求偏导）在切平面上的展开系数记为 $\Gamma^k_{ij}$：

$$
\mathbf{x}_{ij} = \sum_{k=1}^2 \Gamma^k_{ij}\,\mathbf{x}_k + \text{法向分量}
$$

**重点：$\Gamma^k_{ij}$ 是「坐标基向量的变化率在切平面内的展开系数」。** 法向分量我们暂时忽略（它是第二基本形式相关，即第二篇的 $L,M,N$）。$\Gamma$ 只捕捉「坐标系自身怎么弯」——完全内蕴。<span class="marginnote">记号说明：$\Gamma^1_{11}$ 读作「Gamma 上 1 下 11」。上标是展开到哪个基向量，下标是两个求导方向。$\Gamma^k_{ij} = \Gamma^k_{ji}$（对曲面的 Levi-Civita 联络，$\mathbf{x}_{uv} = \mathbf{x}_{vu}$ 保证对称性）——「无挠」性质在坐标下的表现。</span>

## 2 由第一基本形式计算 $\Gamma$

**关键事实：$\Gamma^k_{ij}$ 只依赖 $E,F,G$ 及其一阶导数。** 具体公式（用度量矩阵的逆）：

$$
\Gamma^k_{ij} = \frac{1}{2}\sum_{m=1}^2 g^{km}\Big(\frac{\partial g_{jm}}{\partial x^i} + \frac{\partial g_{im}}{\partial x^j} - \frac{\partial g_{ij}}{\partial x^m}\Big)
$$

其中 $g_{11} = E$、$g_{12} = g_{21} = F$、$g_{22} = G$，$(g^{km})$ 是度量矩阵的逆 $\frac{1}{EG-F^2}\begin{pmatrix}G & -F\\-F & E\end{pmatrix}$，$x^1 = u, x^2 = v$。

这是**六个**独立的 $\Gamma$（因对称性 $2\times2\times2$ 中有 4 个独立，再减对称剩 3 个：$\Gamma^1_{11}, \Gamma^1_{12}, \Gamma^1_{22}, \Gamma^2_{11}, \Gamma^2_{12}, \Gamma^2_{22}$）。<span class="marginnote">记忆技巧：公式结构是「度量导数之和减第三项」。虽然六个符号看起来吓人，但实践中常用三个技巧偷懒：① 选正交坐标（$F=0$）——$\Gamma$ 公式大简化；② 若 $E,G$ 只依赖一个坐标，大量偏导为零；③ 几何对称性常让许多 $\Gamma$ 为零。真正手算时，$\Gamma$ 往往「白给」。</span>

### 例：球面的 Christoffel 记号

球面 $S^2_R$ 用经纬度 $(\theta,\phi)$：$E = R^2\cos^2\phi$、$F=0$、$G = R^2$。算得非零的只有

$$
\Gamma^1_{12} = \Gamma^1_{21} = -\tan\phi, \qquad \Gamma^2_{11} = \sin\phi\cos\phi
$$

其余为零。**球面坐标系只需两个非零 $\Gamma$**——经纬度坐标的「弯曲」全在这两个数里。

## 3 公式解析：$\Gamma$ 怎么从度量算出来

以 $\Gamma^1_{11}$ 为例，走一遍完整推导，理解公式为何长这样：

- **第一步，对 $E$ 求导**：$E = \mathbf{x}_u\cdot\mathbf{x}_u$，对 $u$ 求导：
  $$
  E_u = 2\,\mathbf{x}_{uu}\cdot\mathbf{x}_u
  $$
  左边是已知的度量导数，右边是未知的二阶导与基向量的内积。
- **第二步，引入 $\Gamma$**：把 $\mathbf{x}_{uu} = \Gamma^1_{11}\mathbf{x}_u + \Gamma^2_{11}\mathbf{x}_v + \perp$ 代入，与 $\mathbf{x}_u$ 点乘（法向分量消失）：
  $$
  \frac{1}{2}E_u = \Gamma^1_{11}\,E + \Gamma^2_{11}\,F
  $$
  这是一个关于两个未知 $\Gamma$ 的线性方程。类似地，对 $F_u$、$G_u$ 等再取几个方程，联立解出全部 $\Gamma$。
- **第三步，统一成标准公式**：把「解线性方程组」的步骤写成闭式，就是上面那个带逆度量矩阵的求和公式。**公式虽长，本质是「从度量导数的线性方程组解出 $\Gamma$」。**

**辨析｜易错点：** $\Gamma$ 是**不是**张量（它在坐标变换下不按张量法则变）。它们是「联络系数」——像坐标系的「加速度计」，记录坐标怎么弯。正因为不是张量，它们的数值依赖坐标，但「协变导数的整体行为」（测地线、平行移动）不依赖坐标。**别把 $\Gamma$ 当成几何量，它是坐标工具。**

## 4 为什么 $\Gamma$ 完全由度量决定

$\Gamma$ 从 $E,F,G$ 算出的这个事实，是「度量决定内蕴几何」的又一明证：

- **度量 $g_{ij}$**（即 $E,F,G$）给定 ⟹ $\Gamma^k_{ij}$ 由公式唯一确定。
- $\Gamma$ 给定 ⟹ 协变导数确定（上一节）⟹ 平行移动、测地线确定。
- **所以：只需度量，即可推导出曲面的全部内蕴几何。**

这条链条是黎曼几何的纲领：**在任意流形上，给一个度量（黎曼度量），一切都自动长出来**——测地线、平行移动、曲率。$\Gamma$ 是度量与几何之间的「转换器」。<span class="marginnote">反过来有一个微妙点：$\Gamma$ 的公式里用了「度量矩阵的逆」，而逆的存在依赖 $EG-F^2\neq0$（正则性）。所以「度量 ⟹ $\Gamma$」在正则曲面处处成立，但在退化点（如球面极点）坐标公式失效——那里必须换坐标卡。这就是「图册」存在的必要性的又一次显现。</span>

## 5 $\Gamma$ 的用途：内蕴计算的脚手架

$\Gamma$ 几乎出现在所有内蕴计算里：

**测地线方程**（下一节）：$u'' + \Gamma^1_{11}(u')^2 + 2\Gamma^1_{12}u'v' + \Gamma^1_{22}(v')^2 = 0$——测地线的坐标方程直接由 $\Gamma$ 写成。
**平行移动**：$Dw/dt = 0$ 展开成 $\dot w^k + \Gamma^k_{ij}\dot x^i w^j = 0$。
**曲率张量**（第八篇）：黎曼曲率张量 $R^l_{ijk}$ 由 $\Gamma$ 及其导数构成。
**Gauss-Codazzi 方程**：内蕴相容性条件，也以 $\Gamma$ 为零件。

**重点：$\Gamma$ 是「度量 → 内蕴几何」的翻译官。** 学会了它，测地线、平行移动、曲率的全部计算都是机械代入——这也是为什么黎曼几何的学习曲线「过了 $\Gamma$ 就平坦了」。<span class="marginnote">对数值/工程读者：在三角网格上，「离散 Christoffel 记号」是计算测地距离、平行移动的标准工具（如 heat method）。理解 $\Gamma$ 的几何含义（坐标基变化率的切向分量），是读懂这些数值算法的前提。</span>

### 例：极坐标下的 Christoffel 记号

用极坐标计算平面度量的 $\Gamma$，感受「坐标弯曲」与「真实弯曲」的区别。平面极坐标 $E=1$、$F=0$、$G=r^2$，非零 $\Gamma$ 只有

$$
\Gamma^r_{\theta\theta} = -r, \qquad \Gamma^\theta_{r\theta} = \Gamma^\theta_{\theta r} = \frac{1}{r}
$$

（可代入公式 $2\Gamma^k_{ij}g_{km} = \partial_i g_{jm} + \partial_j g_{im} - \partial_m g_{ij}$ 验证。）

**重点：平面是平坦的（$K=0$），但极坐标下 $\Gamma \neq 0$——$\Gamma$ 反映「坐标系的弯曲」而非「空间的弯曲」。** 这就是为什么 $\Gamma$ 不是张量、而曲率张量是张量：换坐标 $\Gamma$ 就变（极坐标有非零 $\Gamma$、直角坐标全为零），但曲率不变（都是 0）。**「坐标系会骗人，曲率不会」**——$\Gamma$ 是坐标的「表象」，$R$ 是空间的「真实」。

## 6 小结

- **Christoffel 记号** $\Gamma^k_{ij}$：坐标基二阶导在切平面内的展开系数；$\mathbf{x}_{ij} = \sum_k \Gamma^k_{ij}\mathbf{x}_k + \perp$。
- **由度量计算**：$\Gamma^k_{ij} = \frac12 g^{km}(g_{jm,i} + g_{im,j} - g_{ij,m})$，完全由 $E,F,G$ 决定。
- 六个独立记号；球面经纬度只需两个非零 $\Gamma$。
- $\Gamma$ 不是张量（坐标相关），但协变导数的整体行为坐标无关。
- 用途：测地线方程、平行移动、曲率张量、Gauss-Codazzi——内蕴计算的脚手架。

在下一节，我们使用 $\Gamma$ 做第一件大事：**平行移动**——定义「向量沿曲线不变」的精确意义，并揭示曲率如何让平行移动绕圈后「失真」。
