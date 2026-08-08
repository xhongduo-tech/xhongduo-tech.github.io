---
title: Levi-Civita 联络
date: 2026-08-07
---

# Levi-Civita 联络

<div class="epigraph">
<p>在众多联络中，度量选中了一个——它既尊重度量，又不带扭转，这个联络以它的发现者命名。</p>
<footer>—— 图利奥 · 列维-奇维塔（Tullio Levi-Civita）</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分几何 ｜ 陈维桓《微分几何》§8.4 ｜ 2026-08-07</p>
</div>

## 为什么从 Levi-Civita 联络开始

仿射联络有无限多个（任意系数 $\Gamma^k_{ij}$）。但黎曼流形 $(M,g)$ 有一个「自然选择」——**Levi-Civita 联络**：它是**唯一**同时满足两个好条件的联络：

1. **度量相容（metric-compatible）**：$\nabla g = 0$——求导与度量「互不干扰」。
2. **无挠（torsion-free）**：$\Gamma^k_{ij} = \Gamma^k_{ji}$——坐标变换的「对称性」。

这两个条件**唯一确定**一个联络，且它的系数正是我们从曲面就认识的 **Christoffel 记号**。这就是**黎曼几何基本定理**。这一节建立 Levi-Civita 联络，并解释为什么它如此特殊。<span class="marginnote">曲面上的协变导数（投影回切平面）<strong>就是</strong> Levi-Civita 联络——曲面情形下它自动满足度量相容与无挠。所以你在第四篇已经「使用」了 Levi-Civita 联络，只是没命名。「由度量决定的联络」从曲面到任意维流形是同一个对象——「度量 ⟹ 联络」的完美推广。</span>

## 1 度量相容与无挠

**定义（度量相容，metric compatibility）**：联络 $\nabla$ 与度量 $g$ **相容**，如果

$$
X\big(g(Y,Z)\big) = g(\nabla_X Y, Z) + g(Y, \nabla_X Z)
$$

「对 $g(Y,Z)$ 求导，用联络展开」——这正是曲面协变导数的「保内积」性质（第四篇）的公理化。

**定义（无挠，torsion-free）**：联络**无挠**，如果

$$
\nabla_X Y - \nabla_Y X = [X, Y]
$$

坐标下即 $\Gamma^k_{ij} = \Gamma^k_{ji}$——联络系数对下标对称。

**重点：度量相容 = 「求导与内积可交换」；无挠 = 「协变导数是 Lie 导数的对称化」。** 两个条件各有几何含义：前者让平行移动保长度保夹角，后者让「二阶导数无扭转」。<span class="marginnote">直观理解两个条件：度量相容 ⟹ 平行移动保持长度与夹角（测地线匀速、正交保持正交）；无挠 ⟹ 「先沿 $X$ 再沿 $Y$ 与先沿 $Y$ 再沿 $X$」的联络差异只由 Lie 括号给出（没有额外的「扭转」）。缺任何一个，联络都会「怪」——度量不相容则平行移动不保长，有挠则平行移动「拧着走」。</span>

## 2 Levi-Civita 联络的系数：Christoffel 记号

**定理（Levi-Civita 联络的坐标系数）**：度量相容 + 无挠唯一确定联络系数

$$
\Gamma^k_{ij} = \frac{1}{2}\sum_m g^{km}\Big(\frac{\partial g_{jm}}{\partial x^i} + \frac{\partial g_{im}}{\partial x^j} - \frac{\partial g_{ij}}{\partial x^m}\Big)
$$

**这正是第四篇的 Christoffel 记号公式——只是 $g_{ij}$ 取代了 $E,F,G$。** 由度量 $g$ 完全决定。

**重点：Levi-Civita 联络系数 = Christoffel 记号 = 度量的函数。** 曲面上你算过的 $\Gamma$（如球面经纬度），就是 Levi-Civita 联络的系数。**「度量 ⟹ 联络」在任意维流形上由这条公式实现。**<span class="marginnote">公式结构：「度量导数之和减第三项」，乘逆度量。二维时 $g_{11}=E,g_{12}=F,g_{22}=G$，代入得到曲面版的 Christoffel 记号——你在《Christoffel 记号》一节已经见过它。「$n$ 维推广 = 下标扩展」：从 $u,v$ 到 $1,\dots,n$，公式一字不变。</span>

## 3 公式解析：为什么「度量相容 + 无挠」唯一确定联络

这是黎曼几何基本定理，拆开看推理链：

- **第一步，写出三个对称组合**：由度量相容，
  $$
  \partial_i g_{jk} = g(\nabla_{\partial_i}\partial_j, \partial_k) + g(\partial_j, \nabla_{\partial_i}\partial_k)
  $$
  把 $\nabla_{\partial_i}\partial_j = \Gamma^m_{ij}\partial_m$ 代入，得到 $\partial_i g_{jk} = \Gamma^m_{ij}g_{mk} + \Gamma^m_{ik}g_{jm}$ 型等式。
- **第二步，循环相减**：考虑 $\partial_i g_{jk} + \partial_j g_{ik} - \partial_k g_{ij}$（无挠使 $\Gamma$ 对称），三项代入并利用对称性，交叉项抵消，剩下
  $$
  \partial_i g_{jk} + \partial_j g_{ik} - \partial_k g_{ij} = 2\Gamma^m_{ij}g_{mk}
  $$
- **第三步，乘逆度量解出**：两边乘逆 $g^{kn}$ 求和，得
  $$
  \Gamma^n_{ij} = \frac{1}{2}g^{kn}(\partial_i g_{jk} + \partial_j g_{ik} - \partial_k g_{ij})
  $$
  ——唯一确定。

**重点：两个条件给出一组「线性方程」，方程可解（度量正定 ⟹ 逆存在），解唯一。** 黎曼几何基本定理因此成立：**每条黎曼流形上恰有一个 Levi-Civita 联络。**<span class="marginnote">这个「循环相减」技巧是黎曼几何的基本手法：把「想求的量」（$\Gamma$）通过「已知的量的导数」（$\partial g$）表达出来，用对称性消去交叉项。「$g$ 的导数 ⟹ $\Gamma$」的推导是「度量决定联络」的机械证明——理解它，就理解了为什么 Levi-Civita 联络「非它不可」。</span>

## 4 Levi-Civita 联络的几何含义

Levi-Civita 联络是黎曼流形的「自然导数」，它的几何含义丰富：

**平行移动保度量**：沿任何曲线平行移动，保持长度与夹角——「向量被无扭曲地搬运」。
**测地线是「最直的路」**：$\nabla_{\gamma'}\gamma' = 0$——速度沿自身平行，长度不变（匀速）。
**梯度、散度、拉普拉斯**：都用 Levi-Civita 联络定义——流形上的微积分全部基于它。
**曲率张量**：由联络的不可交换性定义（下一节）——曲率是联络的「曲」。

**重点：Levi-Civita 联络是黎曼流形的「标准求导」——一切几何分析都以它为基础。** 梯度、散度、Laplacian、曲率，全是它的派生对象。<span class="marginnote">物理里，Levi-Civita 联络的「度量相容」保证「自由落体的世界线测地线上，固有时匀速流逝」；「无挠」保证「对称的物理定律」。广义相对论用 Levi-Civita 联络描述引力——「物质决定度量，度量决定联络，联络给出测地线（自由落体）」。</span>

## 5 Levi-Civita 联络的地位

Levi-Civita 联络是整个黎曼几何的枢纽：

**黎曼几何基本定理**：度量 ⟹ 唯一联络——「度量是黎曼流形的完整输入」。
**曲率的地基**：曲率张量 $R$ 由 $\nabla$ 定义（下一节）——「联络的曲」。
**自然梯度**：深度学习里的自然梯度 = 用 Levi-Civita 联络（Fisher 度量）做的协变梯度（第九篇）。
**规范场论对比**：物理的规范联络不必是无挠的——Levi-Civita 是「黎曼流形的规范联络」。<span class="marginnote">对比「自然梯度」与「普通梯度」：普通梯度 $\nabla f$ 不是坐标无关的（换坐标就变），自然梯度 $g^{-1}\nabla f$ 是（用度量升级）——这正是「度量相容的联络」保证的性质。流形优化里「正确的梯度」必须是协变的——Levi-Civita 联络是它唯一的选择。</span>

### 例：球面经纬度的 Christoffel 记号

用球面具体算出 Levi-Civita 联络的系数，验证「度量 ⟹ 联络」。单位球面 $g = \cos^2\phi\,d\theta^2 + d\phi^2$，代入公式得非零 $\Gamma$：

$$
\Gamma^\theta_{\theta\phi} = \Gamma^\theta_{\phi\theta} = -\tan\phi, \qquad \Gamma^\phi_{\theta\theta} = \sin\phi\cos\phi
$$

**测地线方程**（用这些 $\Gamma$）：

$$
\theta'' - 2\tan\phi\,\theta'\phi' = 0, \qquad \phi'' + \sin\phi\cos\phi(\theta')^2 = 0
$$

正是球面测地线（大圆）的方程。

**重点：一个度量（球面 $g$）⟹ 唯一的联络（$\Gamma$）⟹ 唯一的测地线方程——「度量决定一切」的完整链条在球面上显形。** 这套计算是黎曼几何基本定理（下一节）的具体操作：Levi-Civita 联络不是「任选」的，而是由度量唯一钦定的「自然联络」。

## 6 小结

- **度量相容** $\nabla g = 0$：求导与内积可交换，平行移动保长保角。
- **无挠**：$\nabla_X Y - \nabla_Y X = [X,Y]$，坐标下 $\Gamma^k_{ij}=\Gamma^k_{ji}$。
- **Levi-Civita 联络**：唯一同时满足两者的联络；系数 = Christoffel 记号。
- 系数公式：$\Gamma^k_{ij} = \frac12 g^{km}(g_{jm,i} + g_{im,j} - g_{ij,m})$——度量的函数。
- **黎曼几何基本定理**：每条黎曼流形有唯一 Levi-Civita 联络（下一节详述）。
- 地位：梯度/散度/拉普拉斯/曲率/自然梯度的共同基础。

在下一节，我们给出「度量 ⟹ 联络」的完整定理：**黎曼几何基本定理**。
