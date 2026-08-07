---
title: 视图变换（View Transformation）：相机标架
date: 2026-08-07
---

# 视图变换（View Transformation）：相机标架

<div class="epigraph">
<p>世界不动，动的只是观察者的眼睛。</p>
<footer>—— 图形学课堂常谈</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机图形学 ｜ GAMES101 & 虎书（Fundamentals of Computer Graphics）§7.2 ｜ 2026-08-07</p>
</div>

## 为什么需要视图变换

模型变换把物体摆进了世界坐标系。但世界坐标系的坐标轴是固定的——它的原点在世界中心、$x/y/z$ 轴是「世界的方向」。而我们看到的画面取决于**相机（camera）**的位置与朝向：同一个世界，从正面看与从侧面看截然不同。<span class="marginnote">视图变换的哲学：<strong>渲染不关心「世界是什么样」，只关心「相机看到了什么」</strong>。于是我们把整个坐标系变成「以相机为中心的坐标系」——相机在原点上、看向 $-z$ 方向、$y$ 轴向上——再在这个坐标系里渲染，一切就简单了。</span>

**视图变换（view transformation）** 就是「世界 → 相机」这一步：把世界坐标系里的所有点，重新表示在以相机为原点、以相机朝向为 $z$ 轴的坐标系中。它和模型变换地位对等，只是方向相反。

## 1 定义相机：位置、朝向、上方向

一个相机（或者说视点）用三个量完整描述：

- **位置（position）** $\vec{e}$：相机在世界坐标里的位置。
- **朝向（look-at / gaze）** $\vec{g}$：相机看向的方向。
- **上方向（up）** $\vec{t}$：相机的「头顶」指向，用于确定相机的翻滚。

这三个量构成相机的**标架（frame）**。为什么需要上方向？光有位置和朝向还不够——你可以看向正前方，但你的头还能左歪右歪，画面会跟着旋转。上方向 $\vec{t}$ 把「歪头」这个自由度钉死。<span class="marginnote">真实相机里 $\vec{t}$ 通常取世界坐标的 $y$ 轴（天空方向），但在过山车视角、飞行模拟里，相机会翻滚，$\vec{t}$ 随姿态变化——所以它必须作为输入给出，而不是默认「向上」。</span>

## 2 从「相机在世界里」到「相机就是世界」

视图变换的目标是构造一个矩阵 $V$，使得变换后：

1. 相机位置 $\vec{e}$ 落到原点 $(0,0,0)$。
2. 看向方向 $\vec{g}$ 指向 $-z$ 轴。
3. 上方向 $\vec{t}$ 指向 $y$ 轴。

这三条把「相机坐标系」完全定死了。构造 $V$ 的经典做法是两步复合——先平移、再旋转：

$$
V = R_{\text{view}}\; T_{\text{view}}
$$

**第一步，平移**：把相机搬到原点。

$$
T_{\text{view}} = \begin{pmatrix} 1 & 0 & 0 & -e_x \\ 0 & 1 & 0 & -e_y \\ 0 & 0 & 1 & -e_z \\ 0 & 0 & 0 & 1 \end{pmatrix}
$$

**第二步，旋转**：把相机的三个轴（右、上、后）转到世界坐标轴。这一步最妙的技巧是**求逆**：与其直接找「把相机轴转到世界轴」的矩阵，不如先写「把世界轴转到相机轴」的矩阵——因为后者的列就是相机轴的坐标，而正交矩阵的逆就是转置，于是前者免费得到。

## 3 构造旋转部分：巧用正交矩阵

设相机的三个轴向（在世界坐标系中）为：

$$
\hat{\mathbf{w}} = \frac{\vec{g}}{|\vec{g}|} \quad(\text{看向的反方向，相机坐标的 }-z),\qquad \hat{\mathbf{u}} = \frac{\vec{t}\times\hat{\mathbf{w}}}{|\vec{t}\times\hat{\mathbf{w}}|} \quad(\text{右方向}),\qquad \hat{\mathbf{v}} = \hat{\mathbf{w}}\times\hat{\mathbf{u}} \quad(\text{上方向})
$$

`叉积`在这里连续登场：$u$ 由「上 × 后」得到右，$v$ 由「后 × 右」得到上——保证 $u, v, w$ 构成右手正交系。<span class="marginnote">三个轴必须两两正交且满足右手定则，否则渲染会「拧」起来。$\hat{\mathbf{u}}$ 从 $\vec{t}\times\hat{\mathbf{w}}$ 得来，天然与 $\vec{w}$ 正交；$\hat{\mathbf{v}}$ 再用一次叉积封闭成完整正交基。</span>

「世界轴转到相机轴」的矩阵是：

$$
R_{\text{world}\to\text{cam}} = \begin{pmatrix} \hat{\mathbf{u}}_x & \hat{\mathbf{u}}_y & \hat{\mathbf{u}}_z & 0 \\ \hat{\mathbf{v}}_x & \hat{\mathbf{v}}_y & \hat{\mathbf{v}}_z & 0 \\ \hat{\mathbf{w}}_x & \hat{\mathbf{w}}_y & \hat{\mathbf{w}}_z & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}
$$

它的每行是世界轴在相机轴上的投影（即「世界坐标 → 相机坐标」的系数）。因为它正交，逆等于转置，所以「相机轴转到世界轴」就是转置；而我们需要的正是逆方向，于是：

$$
R_{\text{view}} = \left(R_{\text{world}\to\text{cam}}\right)^\top
$$

## 4 公式解析：视图矩阵为什么可以免费求逆

把关键一步单独拆开看。设相机轴 $\hat{\mathbf{u}}, \hat{\mathbf{v}}, \hat{\mathbf{w}}$ 是正交单位向量组，那么「世界 → 相机」坐标变换矩阵 $R$ 的每一行就是「把世界坐标投影到相机轴」：

$$
R = \begin{pmatrix} \hat{\mathbf{u}}^\top \\ \hat{\mathbf{v}}^\top \\ \hat{\mathbf{w}}^\top \end{pmatrix}, \qquad R\,\vec{x} = \begin{pmatrix} \hat{\mathbf{u}}\cdot\vec{x} \\ \hat{\mathbf{v}}\cdot\vec{x} \\ \hat{\mathbf{w}}\cdot\vec{x} \end{pmatrix}
$$

- **第一步，正交性**：$\hat{\mathbf{u}}, \hat{\mathbf{v}}, \hat{\mathbf{w}}$ 两两正交且单位化，故 $R R^\top = \mathbf{I}$，$R$ 是正交矩阵。
- **第二步，转置即逆**：正交矩阵满足 $R^{-1} = R^\top$。所以「相机轴 → 世界」的矩阵（我们想要的视图旋转）直接取转置。
- **第三步，列的意义**：$R^\top$ 的三列恰好是 $\hat{\mathbf{u}}, \hat{\mathbf{v}}, \hat{\mathbf{w}}$ 的世界坐标——符合「矩阵的列是变换后基向量的坐标」这一贯约定。

**辨析｜易错点：** 构造相机轴时方向最容易错。相机看向 $\vec{g}$，但 OpenGL / 图形学约定相机朝 $-z$ 看，所以 $\vec{w}$ 取 $\vec{g}$ 的**反方向**（$-\vec{g}$ 归一化）。若把 $\vec{w}$ 取成 $\vec{g}$ 本身，画面会前后颠倒。另一个坑是 $\vec{u}$ 与 $\vec{v}$ 的叉积方向：必须保证 $u \times v = w$，否则视图矩阵行列式为负，世界会被镜像。

## 5 视图矩阵的完整形态

把平移与旋转复合起来，视图矩阵是：

$$
V = \begin{pmatrix} \hat{\mathbf{u}}_x & \hat{\mathbf{u}}_y & \hat{\mathbf{u}}_z & -\hat{\mathbf{u}}\cdot\vec{e} \\ \hat{\mathbf{v}}_x & \hat{\mathbf{v}}_y & \hat{\mathbf{v}}_z & -\hat{\mathbf{v}}\cdot\vec{e} \\ \hat{\mathbf{w}}_x & \hat{\mathbf{w}}_y & \hat{\mathbf{w}}_z & -\hat{\mathbf{w}}\cdot\vec{e} \\ 0 & 0 & 0 & 1 \end{pmatrix}
$$

看到那个 $- \hat{\mathbf{u}}\cdot\vec{e}$ 了吗？它是「把相机平移到原点后，再把相机位置 $\vec{e}$ 用相机轴重新表示并取负」——平移与旋转的耦合被整齐地折叠进第四列。<span class="marginnote">这列乘积 $-\hat{\mathbf{u}}\cdot\vec{e}$ 是很多初学者的盲区：它不能只写 $-e_x$，因为旋转后相机位置的三个分量已经不是世界坐标的 $e_x, e_y, e_z$，而是要在相机轴上的投影。</span>

把视图矩阵作用到所有物体上，等于「整个世界被搬到以相机为中心的坐标系」。此后无论相机怎么动，我们都站在相机视角渲染——这就是《正交/透视投影》之前必经的一步。

## 6 视图变换与模型变换的对称性

模型变换与视图变换其实是同一枚硬币的两面：

| | 模型变换 | 视图变换 |
| --- | --- | --- |
| 方向 | 局部 → 世界 | 世界 → 相机 |
| 几何 | 物体被搬动 | 观察者被搬动 |
| 直觉 | 把模型放到世界 | 把相机放到原点 |
| 逆操作 | $M^{-1}$ 世界 → 局部 | $V^{-1}$ 相机 → 世界 |

现代渲染里，两者常被合称为 **ModelView 矩阵**（$V \cdot M$），即「先把物体从局部搬到世界，再搬到相机坐标系」。合在一起后，投影变换直接作用于相机空间——这条管线正是下一节《正交投影变换》的起点。

## 7 小结

- 相机由**位置 $\vec{e}$、朝向 $\vec{g}$、上方向 $\vec{t}$** 定义。
- 视图变换 = 平移（相机到原点）+ 旋转（相机轴对齐坐标轴），$V = R_{\text{view}}\,T_{\text{view}}$。
- 构造旋转靠**正交矩阵转置即逆**：先写「世界 → 相机」，再转置得「相机 → 世界」。
- 相机朝 $-z$ 看、$\vec{w}$ 取 $-\vec{g}$，符号是最高频错误。
- 模型变换与视图变换合称 ModelView，是投影前的最后一步。

在下一节，我们终于可以把三维世界「拍扁」成二维了——从**正交投影变换**开始。
