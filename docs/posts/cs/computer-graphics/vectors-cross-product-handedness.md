---
title: 向量：叉积与左右手坐标系
date: 2026-08-07
---

# 向量：叉积与左右手坐标系

<div class="epigraph">
<p>直线属于人类，曲线属于上帝。</p>
<footer>—— 安东尼 · 高迪（Antoni Gaudí）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机图形学 ｜ GAMES101 第2讲 / 虎书 第2、5章 ｜ 2026-08-07</p>
</div>

## 为什么从「叉积」开始

上一篇我们认识了点积：两个向量的点积给出一个**标量**，告诉我们「两个方向有多接近」。今天的主角是它的孪生兄弟——**叉积（cross product）**。如果说点积回答「方向相近吗」，叉积回答的是：**给定两个方向，如何构造出同时垂直于它们的新方向？**

叉积在图形学里是「造方向」的发动机。三角形的**法向量**靠它造出来，某点在三角形**内还是外**靠它判定，相机坐标系的**第三根轴**靠它补齐，一个多边形的**朝向（正面/背面）**也靠它分辨。正因如此，理解叉积不仅是算几个分量，更要理解它背后的**朝向约定**——而左右手坐标系，正是这个约定最直观的载体。

## 1 叉积的定义：输出一个向量

**叉积（cross product）**：给定三维向量 $\mathbf{a}$、$\mathbf{b}$，叉积 $\mathbf{a} \times \mathbf{b}$ 是一个**新向量**，满足三条规则：

- **垂直于两者**：$\mathbf{a} \times \mathbf{b}$ 同时垂直于 $\mathbf{a}$ 与 $\mathbf{b}$。<span class="marginnote">「同时垂直于两个向量」的结果其实有两个候选方向（上下各一个），到底取哪个由右手定则决定——这正是左/右手坐标系的根源。</span>
- **长度等于平行四边形面积**：$|\mathbf{a} \times \mathbf{b}| = |\mathbf{a}|\,|\mathbf{b}|\,\sin\theta$，其中 $\theta$ 是 $\mathbf{a}$ 与 $\mathbf{b}$ 的夹角。当两者平行时 $\sin\theta = 0$，叉积为零向量。
- **方向由右手定则确定**：四指从 $\mathbf{a}$ 弯向 $\mathbf{b}$，大拇指指向即为 $\mathbf{a} \times \mathbf{b}$ 的方向。

**辨析｜易错点：** 点积输出**标量**，叉积输出**向量**——两者名字里都有「积」，但结果类型完全不同。另外叉积**不满足交换律**，而是**反交换律**：

$$
\mathbf{a} \times \mathbf{b} = -\mathbf{b} \times \mathbf{a}
$$

交换两个向量的顺序，方向会反转。这是初学者最常见的陷阱：三角形求法线时一旦把边的顺序写反，法线就会指向背面。

## 2 叉积的坐标公式：分量怎么算

给定 $\mathbf{a} = (a_x, a_y, a_z)$、$\mathbf{b} = (b_x, b_y, b_z)$，叉积的每个分量按「交叉相乘再相减」得到：

$$
\mathbf{a} \times \mathbf{b} =
\begin{pmatrix}
a_y b_z - a_z b_y \\
a_z b_x - a_x b_z \\
a_x b_y - a_y b_x
\end{pmatrix}
$$

**直觉记忆法**：把坐标轴单位向量排成 $\mathbf{x}=(1,0,0)$、$\mathbf{y}=(0,1,0)$、$\mathbf{z}=(0,0,1)$。对标准基，叉积遵循循环规则：

$$
\mathbf{x} \times \mathbf{y} = \mathbf{z}, \qquad
\mathbf{y} \times \mathbf{z} = \mathbf{x}, \qquad
\mathbf{z} \times \mathbf{x} = \mathbf{y}
$$

即「$\mathbf{x} \to \mathbf{y} \to \mathbf{z} \to \mathbf{x}$」按顺序叉乘给出下一个轴；逆序则为负（如 $\mathbf{y} \times \mathbf{x} = -\mathbf{z}$）。任意向量叉积的分量公式，正是这条规则逐项展开的结果——每个分量都对应「本轮循环的交叉乘积之差」。<span class="marginnote">反循环方向给出负号，这与你熟知的「右手螺旋」完全一致；换到左手坐标系时，这套符号会整体反转。</span>

## 3 叉积的三大应用

叉积是图形学里出镜率最高的「方向判定器」，三个经典场景：

### 3.1 求法向量

平面上两个不共线向量 $\mathbf{p}$、$\mathbf{q}$ 张成该平面，叉积 $\mathbf{n} = \mathbf{p} \times \mathbf{q}$ 给出平面法向（未归一化）。对三角形来说，取两条边：

$$
\mathbf{n} = (\mathbf{b} - \mathbf{a}) \times (\mathbf{c} - \mathbf{a})
$$

**注意：** 顶点的**环绕方向**（逆时针/顺时针）直接决定法线朝外还是朝内。建模软件与引擎通常约定「逆时针顶点顺序朝外」，因此法线方向与顶点顺序强绑定——这是后面着色与背面剔除的基础。<span class="marginnote">「逆时针朝外」并非数学定律，而是行业约定：OpenGL 默认 CCW 为正面，DirectX 也支持这一约定。判断「逆时针」需要先明确视线方向，这正是下一节左右手坐标系要处理的。</span>

### 3.2 判定点是否在三角形内

给定三角形 $ABC$ 与点 $P$，如何判断 $P$ 在三角形内部？用叉积的**方向一致性**：

$$
\mathbf{e}_1 = (\mathbf{b}-\mathbf{a}) \times (\mathbf{p}-\mathbf{a}), \quad
\mathbf{e}_2 = (\mathbf{c}-\mathbf{b}) \times (\mathbf{p}-\mathbf{b}), \quad
\mathbf{e}_3 = (\mathbf{a}-\mathbf{c}) \times (\mathbf{p}-\mathbf{c})
$$

若三个叉积指向**同一方向**（与法线点积同号），则 $P$ 在三角形内；若方向不一致，$P$ 在外部。直觉：沿三角形边走一圈，内部点始终位于每条边的同一侧，而外部点会「越过」某条边。这正是光栅化里「点在三角形内吗」的标准判据。

### 3.3 建立正交基

给定一个向量 $\mathbf{w}$，想构造一组互相垂直的单位向量？用叉积补出垂直于它的方向：

$$
\mathbf{u} = \frac{\mathbf{w}}{|\mathbf{w}|}, \qquad
\mathbf{v} = \frac{\mathbf{z} \times \mathbf{u}}{|\mathbf{z} \times \mathbf{u}|}, \qquad
\mathbf{w}' = \mathbf{u} \times \mathbf{v}
$$

这组 $\{\mathbf{u}, \mathbf{v}, \mathbf{w}'\}$ 就是一张**正交基**——相机坐标系的右、上、前三个方向正是这么构造的。我们在下一篇《法向量与正交基》中会专门展开。

## 4 公式解析：叉积长度 = 平行四边形面积

叉积的几何意义藏在它的模长公式里，值得拆开看：

$$
|\mathbf{a} \times \mathbf{b}| = |\mathbf{a}|\,|\mathbf{b}|\,\sin\theta
$$

- **第一步，识别 $|\mathbf{b}|\sin\theta$**：把 $\mathbf{b}$ 分解为平行于 $\mathbf{a}$ 与垂直于 $\mathbf{a}$ 的两个分量，其中垂直于 $\mathbf{a}$ 的分量长度恰为 $|\mathbf{b}|\sin\theta$。
- **第二步，底乘高**：以 $|\mathbf{a}|$ 为底、$|\mathbf{b}|\sin\theta$ 为高，平行四边形面积为「底 × 高」=$|\mathbf{a}|\,|\mathbf{b}|\,\sin\theta$。
- **第三步，几何图像**：$\mathbf{a}$、$\mathbf{b}$ 张成一个平行四边形，叉积的模长就是它的面积；当两向量垂直时 $\sin\theta=1$，面积最大；平行时面积为 0。

**辨析｜易错点：** 叉积的模长等于面积，但**叉积本身是向量不是面积**。面积是标量，方向信息（朝内/朝外）在叉积的向量方向里。另外，二维向量没有叉积——二维「叉积」通常退化为一个标量 $a_x b_y - a_y b_x$，它其实就是三维叉积的 $\mathbf{z}$ 分量，代表「二维平行四边形的有向面积」，常被用来做二维多边形朝向判定。

## 5 左右手坐标系

叉积方向依赖右手定则，而「右手」本身就预设了一个**坐标系约定**。区分两个系统：

- **右手坐标系（right-handed）**：$\mathbf{x} \times \mathbf{y} = \mathbf{z}$。拇指指向 $\mathbf{x}$，食指指向 $\mathbf{y}$，中指指向 $\mathbf{z}$，三者成右手势。OpenGL、数学惯例多用右手系。
- **左手坐标系（left-handed）**：$\mathbf{x} \times \mathbf{y} = -\mathbf{z}$。用左手比划同样动作，中指方向恰好反过来。DirectX、Unity 世界坐标系常用左手系。<span class="marginnote">左手/右手只是「镜像关系」：把其中一个的所有坐标整体镜像（例如 $\mathbf{x} \to -\mathbf{x}$），右手系就变成左手系。真正的物理空间不分左右，分左右的是我们建立坐标轴的方式。</span>

**如何快速判断一个坐标系是左还是右？** 把手伸向 $\mathbf{x}$、$\mathbf{y}$ 轴的正方向做叉积，看 $\mathbf{z}$ 是否朝向约定方向。更实用的判据：**绕 $\mathbf{z}$ 轴正方向逆时针旋转时，$\mathbf{x}$ 是否转到 $\mathbf{y}$**——在右手系里 $R_z(90°)$ 把 $\mathbf{x}$ 转到 $\mathbf{y}$。

**辨析｜易错点：** 叉积公式本身不分左右手，但**你把公式套进哪个坐标系，得到的「朝外」就定义在哪个系统里**。同一组顶点数据，在右手系里算出的法线，到左手系里可能指向相反方向。因此移植渲染代码时，坐标系约定必须同步核对——这是跨引擎移植最常见的隐蔽 bug。

![右手定则与叉积方向：x × y = z，z 指向屏幕外；交换顺序得到反向，即 a×b = -(b×a)](/images/computer-graphics/vectors-cross-product-handedness-1.svg)

## 6 小结

- **叉积**输出一个**向量**：垂直于 $\mathbf{a}$、$\mathbf{b}$，模长 $|\mathbf{a}|\,|\mathbf{b}|\,\sin\theta$ 等于平行四边形面积，方向由右手定则确定。
- 坐标公式：$\mathbf{a} \times \mathbf{b} = (a_y b_z - a_z b_y,\ a_z b_x - a_x b_z,\ a_x b_y - a_y b_x)$；标准基遵循 $\mathbf{x}\times\mathbf{y}=\mathbf{z}$ 的循环。
- **反交换律**：$\mathbf{a} \times \mathbf{b} = -\mathbf{b} \times \mathbf{a}$；两向量平行时叉积为零向量。
- 三大应用：**求法向量**、**判定点在三角形内**、**建立正交基**。
- 叉积方向依赖**左右手坐标系约定**：右手系 $\mathbf{x}\times\mathbf{y}=\mathbf{z}$，左手系相反；移植代码时必须核对坐标系。

在下一节，我们将把叉积与点积合起来，系统构造图形学里最重要的结构——**法向量与正交基**：如何求法线、如何归一化、如何用点积与叉积构造一组互相垂直的坐标系，为相机标架与变换打下基础。
