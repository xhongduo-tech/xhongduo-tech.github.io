---
title: 第二基本形式
date: 2026-08-07
---

# 第二基本形式

<div class="epigraph">
<p>第一基本形式描述曲面如何度量，第二基本形式描述曲面如何弯曲；两者合起来，才构成曲面的完整局部几何。</p>
<footer>—— 加斯帕尔 · 蒙日（Gaspard Monge）</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分几何 ｜ 陈维桓《微分几何》§3.2 ｜ 2026-08-07</p>
</div>

## 为什么从第二基本形式开始

第一基本形式 $I$ 告诉我们曲面上**长度、夹角、面积**怎么量——它描述「度量」。但它完全捕捉不到「弯曲」：把一张纸卷成圆柱，第一基本形式分毫不变。弯曲是曲面在三维空间里的属性，它藏在**法向变化**里，而法向变化的量化工具就是**第二基本形式（second fundamental form）** $II$。

如果说第一基本形式是「内在的眼睛」，第二基本形式就是「外在的眼睛」——它看曲面如何偏离自己的切平面。两者合起来，才是曲面局部几何的完整画像。这一节我们把 $II$ 定义清楚，给出坐标下的系数 $L,M,N$（注意别与法向量 $\mathbf{N}$ 混淆），并建立它与高斯映射微分的联系。<span class="marginnote">第二基本形式的标准记号是 $II(v,w) = \langle dN_p(v), w\rangle$（取负号）。$II$ 的坐标系数沿用 Gauss 的记号 $L,M,N$——这是曲面论里最容易和「法向量 $\mathbf{N}$」「第一基本形式的 $E,F,G$」弄混的一组字母，学的时候要格外当心。</span>

## 1 从法向变化到弯曲度量

回顾高斯映射的微分 $dN_p: T_pS \to T_pS$。它是「法向随位置的变化率」，即弯曲的量化。现在用它定义一个**双线性形式**。

**定义（第二基本形式）**：设 $S$ 是带单位法场 $\mathbf{N}$ 的正则曲面，$p\in S$。第二基本形式是切平面上的双线性形式

$$
II_p: T_pS \times T_pS \longrightarrow \mathbb{R}, \qquad II_p(v,w) = -\big\langle dN_p(v),\, w\big\rangle
$$

（负号是记号约定，让凸曲面（如球面）的 $II$ 为正，见下文例子。）

**重点：$II$ 是「把 $dN_p$ 作用后与 $w$ 做内积」**。$dN_p(v)$ 是「沿 $v$ 走法向的变化」，它与 $w$ 的内积衡量「法向变化在 $w$ 方向上的分量」——正是「$v$ 方向的弯曲在 $w$ 方向上的表现」。<br/>为什么有负号？因为凸曲面的法向变化与切向移动方向相反（球面上向外走，法向指向球心、反方向转），加负号使凸曲面的 $II$ 正定。

## 2 坐标下的系数：$L,M,N$

在坐标卡 $\mathbf{x}(u,v)$ 下，用坐标基计算 $II$ 的四个系数：

$$
L = II(\mathbf{x}_u, \mathbf{x}_u) = \mathbf{x}_{uu}\cdot\mathbf{N}, \qquad
M = II(\mathbf{x}_u, \mathbf{x}_v) = \mathbf{x}_{uv}\cdot\mathbf{N},
$$
$$
N = II(\mathbf{x}_v, \mathbf{x}_v) = \mathbf{x}_{vv}\cdot\mathbf{N}
$$

**为什么 $\mathbf{x}_{uu}\cdot\mathbf{N}$ 等于 $-\langle dN(\mathbf{x}_u), \mathbf{x}_u\rangle$？** 因为 $\mathbf{x}_u\cdot\mathbf{N} = 0$（$\mathbf{N}$ 垂直于切平面），两边对 $u$ 求导：

$$
\mathbf{x}_{uu}\cdot\mathbf{N} + \mathbf{x}_u\cdot\mathbf{N}_u = 0 \quad\Longrightarrow\quad
\mathbf{x}_{uu}\cdot\mathbf{N} = -\mathbf{x}_u\cdot\mathbf{N}_u = -\langle dN(\mathbf{x}_u), \mathbf{x}_u\rangle = II(\mathbf{x}_u,\mathbf{x}_u)
$$

**这个求导技巧是第二基本形式全部理论的钥匙**：把对 $\mathbf{N}$ 的微分「搬」到对 $\mathbf{x}$ 的微分上，从而能用 $\mathbf{x}$ 的二阶导数计算。<span class="marginnote">上式是「$\mathbf{x}_u\cdot\mathbf{N}=0$ 两边求导」的产物。它的几何含义：切向量的变化在法向的分量 = 法向变化在切向的负分量。这正是「弯曲」的两个等价视角——从曲面的二阶导看，或从法向的变化看。</span>

于是对任意切向量 $v = du\,\mathbf{x}_u + dv\,\mathbf{x}_v$：

$$
II(v,v) = L\,du^2 + 2M\,du\,dv + N\,dv^2
$$

与第一基本形式的展开式结构完全平行——**只是系数换成了 $L,M,N$**。

## 3 例：球面、平面与圆柱面的第二基本形式

用具体例子建立直觉。

- **平面 $z=0$**：$\mathbf{x}_{uu} = \mathbf{x}_{uv} = \mathbf{x}_{vv} = 0$，故 $L=M=N=0$。第二基本形式恒为零——平面不弯。
- **球面 $S^2_R$**：法向 $\mathbf{N} = (x,y,z)/R$（径向），$\mathbf{x}_{uu}\cdot\mathbf{N} = -1/R$ 类。计算得 $L = N = -1/R$ 或 $L=N=1/R$（取决于法向取向）。取外法向时：
  $$
  II(v,v) = \frac{1}{R}\,I(v,v)
  $$
  第二基本形式与第一基本形式成比例——球面「处处同曲率地弯」，这正是球面的标志。
- **圆柱面**：参数化 $\mathbf{x}(u,v) = (\cos u, \sin u, v)$，得 $L=1$（径向）、$M=0$、$N=0$。于是
  $$
  II(v,v) = du^2, \qquad I(v,v) = du^2 + dv^2
  $$
  弯曲只沿 $u$ 方向（横向），$v$ 方向（轴向）完全不弯——圆柱「单向弯曲」。<span class="marginnote">把这三个例子记住，曲面论的一大半直觉就建立了：平面 $II=0$、球面 $II = I/R$、圆柱 $II = du^2$。它们分别对应「不弯」「双向同弯」「单向弯」三种典型，后面主曲率、高斯曲率都是在这三个原型上做组合。</span>

## 4 公式解析：为什么 $II(v,v) = L\,du^2 + 2M\,du\,dv + N\,dv^2$

这条式子与第一基本形式的展开完全同构，逐项拆：

- **第一步，切向量分解**：$v = du\,\mathbf{x}_u + dv\,\mathbf{x}_v$，其中 $(du,dv)$ 是 $v$ 在坐标基下的分量（沿 $u$-方向走多少、沿 $v$-方向走多少）。
- **第二步，双线性展开**：
  $$
  \begin{aligned}
  II(v,v) &= II(du\,\mathbf{x}_u + dv\,\mathbf{x}_v,\ du\,\mathbf{x}_u + dv\,\mathbf{x}_v)\\
  &= du^2\,II(\mathbf{x}_u,\mathbf{x}_u) + 2\,du\,dv\,II(\mathbf{x}_u,\mathbf{x}_v) + dv^2\,II(\mathbf{x}_v,\mathbf{x}_v)
  \end{aligned}
  $$
  双线性性 + 对称性（$II(v,w)=II(w,v)$，由内积与对称的 $dN$ 保证）给出交叉项系数 2。
- **第三步，代入定义**：$II(\mathbf{x}_u,\mathbf{x}_u) = L$、$II(\mathbf{x}_u,\mathbf{x}_v)=M$、$II(\mathbf{x}_v,\mathbf{x}_v)=N$，得
  $$
  II(v,v) = L\,du^2 + 2M\,du\,dv + N\,dv^2
  $$

**重点：$II$ 的几何意义是「曲面在 $v$ 方向偏离切平面的二阶速率」。** 沿 $v$ 方向走，曲面离开切平面的距离约等于 $\frac{1}{2}II(v,v)\,t^2$（二阶项）——$II(v,v)$ 正是这个「下坠加速度」。$II(v,v) > 0$ 表示曲面朝法向凸起，$< 0$ 表示朝法向凹陷，$=0$ 表示该方向「不弯」。

## 5 第二基本形式与曲面理论的分工

把两个基本形式摆在一起，曲面的「局部画像」才算完整：

| 基本形式 | 系数 | 度量对象 | 内蕴? |
| --- | --- | --- | --- |
| 第一基本形式 $I$ | $E,F,G$ | 长度、夹角、面积 | **内蕴**（只靠曲面自身） |
| 第二基本形式 $II$ | $L,M,N$ | 弯曲、法向变化 | **外蕴**（依赖嵌入 $\mathbb{R}^3$） |

**重点：$I$ 是内蕴的，$II$ 是外蕴的。** 卷纸不改变 $I$ 但改变 $II$（平面卷成圆柱，$II$ 从 0 变成 $du^2$）。「哪些量只用 $I$ 就能算」正是「内蕴几何」的研究范围——这是第四篇的主题，而 Gauss 绝妙定理（高斯曲率居然只用 $I$ 就能算！）将把这条线索推到顶点。<span class="marginnote">$II$ 依赖「曲面怎么放在 $\mathbb{R}^3$ 里」；换一种嵌入（比如把同一片曲面弯成不同形状），$II$ 就变。这也是为什么 $II$ 无法仅由「曲面上的居民」观测到——曲面上的蚂蚁能量长度夹角（$I$），却不知道自己的曲面在三维里怎么弯（$II$）。「内蕴 vs 外蕴」由此成为曲面论最深刻的分野。</span>

## 6 小结

- **第二基本形式** $II_p(v,w) = -\langle dN_p(v), w\rangle$：用高斯映射微分的负内积定义，度量弯曲。
- 坐标系数 $L,M,N$：$L = \mathbf{x}_{uu}\cdot\mathbf{N}$、$M = \mathbf{x}_{uv}\cdot\mathbf{N}$、$N = \mathbf{x}_{vv}\cdot\mathbf{N}$；$II(v,v)=L\,du^2+2M\,du\,dv+N\,dv^2$。
- 关键求导技巧：$\mathbf{x}_u\cdot\mathbf{N}=0$ 两边求导，把 $dN$ 换成 $\mathbf{x}$ 的二阶导。
- 例子：平面 $II=0$、球面 $II=I/R$、圆柱 $II=du^2$——「不弯 / 双向同弯 / 单向弯」三原型。
- $I$ 内蕴、$II$ 外蕴；两者合起来才是曲面局部几何全貌。

在下一节，我们把 $dN_p$ 与 $II$ 的联系做成正式对象：**Weingarten 映射与形状算子**——一个编码了全部弯曲信息的线性算子。
