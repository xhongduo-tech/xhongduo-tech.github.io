---
title: 对坐标的曲面积分
date: 2026-08-07
---

# 对坐标的曲面积分

<div class="epigraph">
<p>通量是向量场穿过曲面的流量——方向与侧，决定了它的正负。</p>
<footer>—— 詹姆斯 · 克拉克 · 麦克斯韦（James Clerk Maxwell）</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等数学 ｜ 同济《高等数学》下册 §11.5 ｜ 2026-08-07</p>
</div>

## 为什么从对坐标的曲面积分开始

水流穿过一块曲面，每秒穿过多少水？这不是「曲面上的总量」，而是「向量场（速度场）穿过曲面的流量」——它依赖曲面选哪一侧（法向量方向）。**对坐标的曲面积分（第二型曲面积分）**正是「向量场在有向曲面上的通量」的数学表述。它是「流量」的语言：电场线穿过曲面的电通量、流体穿过曲面的流量、磁场穿过线圈的磁通量——全是对坐标的曲面积分。它与对面积曲面积分的区别（有向 vs 无向）正是下一节高斯公式（通量定理）的入口。<span class="marginnote">通量的直观：<strong>$\iint_\Sigma \mathbf{F}\cdot\mathbf{n}\,dS$ = 场 $\mathbf{F}$ 每秒穿过曲面 $\Sigma$ 的「净流量」</strong>。$\mathbf{n}$ 是曲面的单位法向量，$\mathbf{F}\cdot\mathbf{n}$ 是「场沿法向的分量」（真正穿过曲面的部分），乘面积微元累加。侧（法向）选反，通量变号。</span>

## 1 有向曲面与对坐标曲面积分的定义

**有向曲面**：指定了「侧」（法向量方向）的曲面。对 $\Sigma: z = z(x,y)$，取法向量向上（$\mathbf{n}$ 与 $z$ 轴正方向夹角为锐角）称取**上侧**，反之取下侧。闭合曲面的外侧、内侧类似。

**对坐标的曲面积分**：设 $\mathbf{F} = P\mathbf{i} + Q\mathbf{j} + R\mathbf{k}$，$\Sigma$ 是有向光滑曲面，定义

$$\iint_\Sigma P\,dy\,dz + Q\,dz\,dx + R\,dx\,dy$$

其中 $dy\,dz$、$dz\,dx$、$dx\,dy$ 是**有向面积微元**——$P\,dy\,dz$ 表示「场在 $x$ 方向的分量穿过垂直于 $x$ 轴的投影面积」。向量形式：

$$\iint_\Sigma \mathbf{F}\cdot\mathbf{n}\,dS = \iint_\Sigma \mathbf{F}\cdot d\mathbf{S}$$

其中 $d\mathbf{S} = \mathbf{n}\,dS$ 是**有向面积向量**。**反向变号**：取曲面另一侧（法向量反向），积分变号。<span class="marginnote">三个「有向投影面积」的含义：$dy\,dz$ 是「面积微元在 $yz$ 平面的有向投影」——它量度「场沿 $x$ 方向的分量穿过该微元」。把 $d\mathbf{S} = (dy\,dz,\ dz\,dx,\ dx\,dy)$ 看成有向面积向量，则 $\mathbf{F}\cdot d\mathbf{S} = P\,dy\,dz + Q\,dz\,dx + R\,dx\,dy$——这是「场与面积向量的点积」，正是「穿流量」的代数化。</span>

## 2 对坐标曲面积分的计算

设 $\Sigma: z = z(x,y)$ 取**上侧**，投影区域 $D_{xy}$，则

$$\iint_\Sigma R(x,y,z)\,dx\,dy = \iint_{D_{xy}} R(x,y,z(x,y))\,dx\,dy$$

**上侧取正号、下侧取负号**。类似地：

$$\iint_\Sigma P\,dy\,dz = \pm\iint_{D_{yz}} P(x(y,z),y,z)\,dy\,dz$$

（取前侧为正、后侧为负）；$Q\,dz\,dx$ 对应右/左侧。

**公式解析：为什么「上侧为正」**

- **第一步，写有向面积向量**：$z=z(x,y)$ 上侧时，法向量 $\mathbf{n}$ 与 $z$ 轴夹角为锐角，$d\mathbf{S} = \mathbf{n}\,dS$ 的 $z$ 分量为正。
- **第二步，投影**：$R\,dx\,dy$ 是「$R$ 乘有向投影面积」，上侧时 $dx\,dy$ 的投影为正（法向朝上），取下侧则投影为负。
- **第三步，约定**：所以「上侧 +、下侧 −」。对 $P\,dy\,dz$，法向量朝 $x$ 正方向（前侧）为正。

**关键**：对坐标曲面积分的计算 = **把曲面投影到对应坐标面 + 按侧定符号 + 二重积分**。「投影 + 符号」是对坐标曲面积分的独有步骤。

## 3 两类曲面积分的联系

设 $\Sigma$ 的单位法向量 $\mathbf{n} = (\cos\alpha, \cos\beta, \cos\gamma)$，则

$$dy\,dz = \cos\alpha\,dS, \qquad dz\,dx = \cos\beta\,dS, \qquad dx\,dy = \cos\gamma\,dS$$

于是

$$\iint_\Sigma P\,dy\,dz + Q\,dz\,dx + R\,dx\,dy = \iint_\Sigma (P\cos\alpha + Q\cos\beta + R\cos\gamma)\,dS$$

——**对坐标曲面积分 = 对面积曲面积分中「场与单位法向量点积」的积分**。这正是「通量 = 法向分量 × 面积」的数学表述。<span class="marginnote">两类曲面积分的联系与两类曲线积分完全平行：<strong>曲线看切向量（$\mathbf{F}\cdot\mathbf{T}ds$），曲面看法向量（$\mathbf{F}\cdot\mathbf{n}dS$）</strong>。切向量把「线」的两型相连，法向量把「面」的两型相连。这个「向量场与几何对象的点积」的统一视角，是理解整个场论积分的关键。</span>

## 4 公式解析：计算通量

求向量场 $\mathbf{F} = (0, 0, z)$ 穿过上半球面 $z=\sqrt{1-x^2-y^2}$ 上侧的通量：

**第一步，识别非零项**：$\mathbf{F}\cdot d\mathbf{S} = z\,dx\,dy$（只有 $R=z$ 项）。
**第二步，投影并定符号**：上半球面取上侧，符号为正，投影区域是单位圆盘 $D$。
**第三步，代 $z$ 并积分**：$\iint_\Sigma z\,dx\,dy = \iint_D \sqrt{1-x^2-y^2}\,dx\,dy$。
**第四步，极坐标**：$\int_0^{2\pi}\int_0^1 \sqrt{1-r^2}\,r\,dr\,d\theta = 2\pi\cdot\frac13 = \frac{2\pi}{3}$。

**关键**：对坐标曲面积分的四步——**识别 $P,Q,R$ → 投影定符号 → 代曲面方程 → 二重积分**。本题通量 $\frac{2\pi}{3}$ 恰是「单位球的上半球体积」，印证了「通量 = 场穿过的总量」的几何直觉。

## 5 通量与曲面积分的应用

对坐标曲面积分（通量）是场论与工程的核心语言：

**通量（flux）**：流体流量、电通量 $\Phi_E = \iint \mathbf{E}\cdot d\mathbf{S}$、磁通量 $\Phi_B = \iint \mathbf{B}\cdot d\mathbf{S}$——法拉第定律、高斯定律都用通量表述。<span class="marginnote">通量是「穿过曲面的净流量」：<strong>场从一侧穿入计负、从另一侧穿出计正，净通量 = 穿出减穿入</strong>。闭合曲面外侧的通量 = 曲面内「源与汇」的净强度——这正是下一节高斯公式（散度定理）的物理内容。</span>
- **闭合曲面的通量**：$\oint\kern{-5pt}\iint_\Sigma \mathbf{F}\cdot d\mathbf{S}$ 表示「单位时间净流出闭合曲面的量」——若为正，内部有源；为负，内部有汇。
- **曲面的几何应用**：通量计算在电磁学、流体力学、热传导里是标准操作，也是高斯公式（第 68 节）的直接载体。

## 6 小结

- **对坐标曲面积分**：$\iint_\Sigma P\,dy\,dz + Q\,dz\,dx + R\,dx\,dy = \iint_\Sigma \mathbf{F}\cdot\mathbf{n}\,dS$——通量。
- **与侧有关**：反向变号；「上侧 +、下侧 −」「前侧 +、后侧 −」等按投影坐标面定。
- 计算：**投影到坐标面 + 按侧定符号 + 二重积分**。
- 两类曲面联系：$dy\,dz=\cos\alpha\,dS$ 等，通量 = 法向分量 × 面积。
- 应用：流体流量、电通量、磁通量——闭合曲面通量是高斯公式的载体。

在下一节，我们将学习把「闭合曲面通量」与「内部散度」相连的定理——**高斯公式、通量与散度**。
