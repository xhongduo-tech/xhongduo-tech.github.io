---
title: 向量：法向量与正交基
date: 2026-08-07
---

# 向量：法向量与正交基

<div class="epigraph">
<p>简单是可靠性的前提。</p>
<footer>—— 埃兹格 · 迪杰斯特拉（Edsger W. Dijkstra）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机图形学 ｜ GAMES101 第2讲 / 虎书 第2章 ｜ 2026-08-07</p>
</div>

## 为什么从「法向量与正交基」开始

上一篇我们用叉积造出了垂直于平面的向量，但「造出来」只是第一步。真正的渲染世界里，**法向量**是判断「光照怎么打」的核心，而 **正交基** 是构造「相机看向哪」的骨架。可以说：点积负责测量，叉积负责定向，而法向量与正交基负责**把方向组织成一套可用的坐标系**。

法向量直接决定物体表面明暗：Blinn-Phong 光照里的 $\mathbf{N} \cdot \mathbf{L}$、环境光遮蔽里的遮挡判断、法线贴图里的逐像素扰动，全都建立在「这个面的法向量朝哪」之上。正交基则无处不在：相机标架、切线空间、球面坐标变换。这一篇我们把「怎么求法向量」「怎么构造一组正交基」讲透，它们将在下一篇变换篇成为矩阵的物理直觉。

## 1 法向量：表面的「朝向」

**法向量（normal vector）**：曲面上某点处垂直于切平面、且通常归一化为单位长度的向量 $\mathbf{n}$。<span class="marginnote">「法向量」一词源于拉丁文 normalis，意为「垂直的」；在英文图形学文献中 normal 常直接指单位法向量。</span>

对**平面**来说，法向量处处相同；对**曲面**（如球面）来说，每一点的法向量随位置变化——球面上点 $\mathbf{p}$ 的法向量就是 $\mathbf{p}/|\mathbf{p}|$，恰好指向球心连线方向。

### 1.1 求一个面的法向量

三角形 $ABC$ 是图形学的基本图元，其法向量由两条边的叉积给出：

$$
\mathbf{n} = \frac{(\mathbf{b}-\mathbf{a}) \times (\mathbf{c}-\mathbf{a})}{|(\mathbf{b}-\mathbf{a}) \times (\mathbf{c}-\mathbf{a})|}
$$

分子是叉积（垂直于平面），分母是归一化（长度为 1）。**分母不能省**——光照公式里 $\mathbf{N} \cdot \mathbf{L}$ 要求 $\mathbf{N}$ 是单位向量，否则点积结果会混入模长干扰。

### 1.2 顶点法向量：从面法线平均

每个三角形有自己的面法线，但共享顶点的多个三角形如何共用一个法向量？**顶点法向量（vertex normal）**取共享该顶点的所有面法线的加权平均：

$$
\mathbf{n}_v = \frac{\sum_i \mathbf{n}_i}{\left|\sum_i \mathbf{n}_i\right|}
$$

这样球体表面由许多小三角面拼成时，顶点法向量指向平滑过渡的方向，光照过渡自然——这就是 **Gouraud / Phong 着色**里「顶点法向量」的来源。若不平滑而是保持每个面各自法线，则看到的是硬棱边（flat shading）。<span class="marginnote">加权平均的权重可用各三角形面积或角度；面积加权更贴近「占多少面积就贡献多少方向」。简单实现常用等权平均，视觉差异通常很小。</span>

**辨析｜易错点：** 求法向量必须先归一化再参与光照计算；但若做**顶点法线平均**，则应**先平均再归一化**——先归一化每个面法线再平均，会让面积大的面对朝向的影响被低估。

## 2 正交基：一组「互相垂直的单位向量」

**正交基（orthonormal basis）**：一组两两垂直、且每个长度都为 1 的向量，它们张成整个空间。在三维空间里就是三个单位向量 $\{\mathbf{u}, \mathbf{v}, \mathbf{w}\}$，满足：

$$
\mathbf{u} \cdot \mathbf{v} = 0, \quad \mathbf{u} \cdot \mathbf{w} = 0, \quad \mathbf{v} \cdot \mathbf{w} = 0, \qquad |\mathbf{u}| = |\mathbf{v}| = |\mathbf{w}| = 1
$$

正交基的威力：**任意向量 $\mathbf{x}$ 都能唯一分解为这三个方向上的分量之和**：

$$
\mathbf{x} = (\mathbf{x} \cdot \mathbf{u})\,\mathbf{u} + (\mathbf{x} \cdot \mathbf{v})\,\mathbf{v} + (\mathbf{x} \cdot \mathbf{w})\,\mathbf{w}
$$

其中每个系数 $\mathbf{x} \cdot \mathbf{u}$ 是 $\mathbf{x}$ 在该方向上的投影长度。这就是「用点积做坐标分解」——它把任意向量翻译成一组数（坐标），是矩阵变换、傅里叶展开等一切「换坐标系」操作的雏形。

## 3 构造正交基：Gram–Schmidt 正交化

给定一组线性无关的向量 $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$，如何得到一组正交基？**Gram–Schmidt 正交化**逐一向量消除「已被先前方向覆盖」的成分：

### 第一步，归一化第一向量

$$
\mathbf{u}_1 = \frac{\mathbf{a}_1}{|\mathbf{a}_1|}
$$

### 第二步，从 $\mathbf{a}_2$ 中减去它在 $\mathbf{u}_1$ 上的投影

$$
\mathbf{v}_2 = \mathbf{a}_2 - (\mathbf{a}_2 \cdot \mathbf{u}_1)\,\mathbf{u}_1, \qquad
\mathbf{u}_2 = \frac{\mathbf{v}_2}{|\mathbf{v}_2|}
$$

几何直觉：$\mathbf{a}_2$ 沿 $\mathbf{u}_1$ 方向的分量是投影 $(\mathbf{a}_2 \cdot \mathbf{u}_1)\mathbf{u}_1$，**减去它**后剩下的部分与 $\mathbf{u}_1$ 垂直。

### 第三步，从 $\mathbf{a}_3$ 中减去它在 $\mathbf{u}_1, \mathbf{u}_2$ 上的投影

$$
\mathbf{v}_3 = \mathbf{a}_3 - (\mathbf{a}_3 \cdot \mathbf{u}_1)\mathbf{u}_1 - (\mathbf{a}_3 \cdot \mathbf{u}_2)\mathbf{u}_2, \qquad
\mathbf{u}_3 = \frac{\mathbf{v}_3}{|\mathbf{v}_3|}
$$

得到 $\{\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3\}$ 即为一组正交基。<span class="marginnote">这与你熟悉的「解方程消元」异曲同工：每一步都在剔除已经确定的坐标方向，剩下的部分张成正交补。</span>

### 一个更轻量的构造法

如果只想从单个向量 $\mathbf{a}$ 出发构造正交基（常见于相机标架），可以用「叉积制造垂直」：

1. $\mathbf{u} = \mathbf{a}/|\mathbf{a}|$
2. 任选一个不与之平行的辅助向量 $\mathbf{t}$，令 $\mathbf{w} = \mathbf{u} \times \mathbf{t}$，归一化
3. 再令 $\mathbf{v} = \mathbf{w} \times \mathbf{u}$，自动与两者垂直且已归一化

这样得到的 $\{\mathbf{u}, \mathbf{v}, \mathbf{w}\}$ 构成正交基，且 $\mathbf{v} = \mathbf{w} \times \mathbf{u}$ 保证了它保持右手系。相机坐标系（look-at 矩阵）正是这样从「视线方向」出发搭起来的。

**辨析｜易错点：** 构造正交基时若辅助向量 $\mathbf{t}$ 与 $\mathbf{u}$ **平行**，叉积为零向量，归一化会除零崩溃。实际代码要检测叉积模长接近 0 时换一个辅助向量——这是矩阵构造里隐蔽而常见的崩溃源。

## 4 公式解析：向量沿正交基分解

前面给出的分解公式值得完整拆解，它是正交基价值的核心：

$$
\mathbf{x} = (\mathbf{x} \cdot \mathbf{u})\,\mathbf{u} + (\mathbf{x} \cdot \mathbf{v})\,\mathbf{v} + (\mathbf{x} \cdot \mathbf{w})\,\mathbf{w}
$$

- **第一步，理解投影系数**：$\mathbf{x} \cdot \mathbf{u} = |\mathbf{x}| \cos\theta$ 是 $\mathbf{x}$ 在 $\mathbf{u}$ 方向上的有符号投影长度。因为 $\mathbf{u}$ 是单位向量，点积直接给出「沿 $\mathbf{u}$ 走了多远」。
- **第二步，为什么必须正交**：若 $\mathbf{u}$、$\mathbf{v}$ 不正交，投影到 $\mathbf{u}$ 的分量里会混入 $\mathbf{v}$ 的成分，分解不唯一；正交保证三个方向互不干扰，分解唯一且系数干净。
- **第三步，为什么必须单位长**：系数 $\mathbf{x} \cdot \mathbf{u}$ 要以 $\mathbf{u}$ 为单位 1 才等于「长度」。若 $\mathbf{u}$ 长度不为 1，系数应写作 $\frac{\mathbf{x}\cdot\mathbf{u}}{|\mathbf{u}|^2}$，麻烦且易错。

**这组分解系数 $(\mathbf{x}\cdot\mathbf{u},\ \mathbf{x}\cdot\mathbf{v},\ \mathbf{x}\cdot\mathbf{w})$ 就是 $\mathbf{x}$ 在新坐标系下的坐标。** 把所有新坐标写成矩阵乘向量的形式，就得到变换矩阵——这恰好是下一篇《矩阵乘法与线性变换》的起点：正交基是坐标系的骨架，矩阵是把坐标系间的映射。

## 5 小结

- **法向量**是表面的朝向，平面法线由叉积归一化得到，**顶点法线**由邻接面法线加权平均而来。
- 法向量参与光照前**必须归一化**；顶点法线要**先平均再归一化**。
- **正交基**：两两垂直且单位长的一组向量；任意向量可沿它唯一分解，系数即投影长度。
- **Gram–Schmidt 正交化**：逐步减去已确定方向的投影，剩下部分与已构造向量垂直。
- 单向量构造正交基可用「叉积制造垂直」，注意辅助向量不能与目标平行。
- 正交基的坐标分解是**矩阵变换的物理直觉**——下一篇我们将看到这些系数如何排成矩阵。

在下一节，我们将把正交基放进更大的框架——**矩阵**：矩阵乘法如何表示线性变换，一个矩阵如何把向量从一组坐标搬到另一组坐标，为后面的 2D/3D 变换铺路。
