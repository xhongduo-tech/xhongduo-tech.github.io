---
title: 第一基本形式
date: 2026-08-07
---

# 第一基本形式

<div class="epigraph">
<p>度量是几何的灵魂：长度、角度、面积皆由此而生。</p>
<footer>—— 伯恩哈德 · 黎曼（Bernhard Riemann）</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分几何 ｜ 陈维桓《微分几何》§2.7 ｜ 2026-08-07</p>
</div>

## 为什么从第一基本形式开始

曲面在每一点有一张切平面 $T_pS$，但光有切平面还不够——我们还不知道**在这张平面上如何量长度和角度**。切平面是一个向量空间，而向量空间只有加上**内积**才能谈「长度」「夹角」。第一基本形式（first fundamental form）就是给切平面配上的这个内积。

它为什么如此关键？因为**曲面上的一切度量都从它派生**：曲线的长度、两方向的夹角、区域的面积、以及后面的测地线、等距变换，全部以第一基本形式为起点。它还是「内蕴」的——只依赖曲面自身的度量结构，不依赖曲面如何嵌入 $\mathbb{R}^3$。这一节我们要把它定义清楚，并记下它的标准记号 $E,F,G$。<span class="marginnote">第一基本形式也叫「度量张量」「黎曼度量」（在流形语境下）。$E,F,G$ 这三个符号自 Gauss 以来沿用两百年，是曲面论的「字母表」前三个字母。看到 $E\,du^2 + 2F\,du\,dv + G\,dv^2$，就看到了曲面度量。</span>

## 1 切平面上要有内积

$\mathbb{R}^3$ 自带标准内积（点积），切平面 $T_pS \subset \mathbb{R}^3$ 继承这个内积。因此「切向量 $v$ 的长度」就是它在 $\mathbb{R}^3$ 里的长度 $\sqrt{v\cdot v}$，两切向量的夹角由 $\cos\theta = \dfrac{v\cdot w}{\|v\|\|w\|}$ 给出。

**定义（第一基本形式）**：曲面 $S$ 在 $p$ 处的第一基本形式是切平面 $T_pS$ 上的内积

$$
I_p: T_pS \times T_pS \longrightarrow \mathbb{R}, \qquad I_p(v,w) = v \cdot w
$$

（点积继承自 $\mathbb{R}^3$）。对切向量 $v$，$I_p(v,v) = \|v\|^2$ 给出其长度的平方。

在坐标卡 $\mathbf{x}(u,v)$ 下，取坐标基 $\{\mathbf{x}_u, \mathbf{x}_v\}$，内积被三个函数完全确定：

$$
E = \mathbf{x}_u\cdot\mathbf{x}_u, \qquad F = \mathbf{x}_u\cdot\mathbf{x}_v, \qquad G = \mathbf{x}_v\cdot\mathbf{x}_v
$$

于是对任意 $v = du\,\mathbf{x}_u + dv\,\mathbf{x}_v$，有

$$
I(v,v) = E\,(du)^2 + 2F\,du\,dv + G\,(dv)^2
$$

**重点：$E,F,G$ 携带切平面内积的全部信息。** 一旦知道了 $E,F,G$（作为 $u,v$ 的函数），曲面上任何长度、角度、面积都能算。$E,F,G$ 排成矩阵 $\begin{pmatrix} E & F \\ F & G \end{pmatrix}$，它是对称正定矩阵——正定性由内积性质保证，是「度量」之所以为度量的根本。<span class="marginnote">「正定」意味着矩阵 $\det = EG - F^2 > 0$，等价于 $E>0$ 且 $EG-F^2>0$。它与正则性 $\mathbf{x}_u\times\mathbf{x}_v\neq0$ 互通：$EG-F^2 = \|\mathbf{x}_u\times\mathbf{x}_v\|^2$。正则 ⟺ 度量正定，两个视角一个结论。</span>

## 2 长度：第一基本形式的第一个孩子

曲面上曲线的长度，是最直接的应用。设 $\alpha: [a,b] \to S$ 是曲面上的曲线，$v = \alpha'(t)$ 是其速度。曲线上每点的切向量在坐标基下为 $v = u'\mathbf{x}_u + v'\mathbf{x}_v$，于是

$$
\|\alpha'(t)\| = \sqrt{E(u')^2 + 2F u'v' + G(v')^2}
$$

曲线全长：

$$
L(\alpha) = \int_a^b \|\alpha'(t)\|\,dt = \int_a^b \sqrt{E\,(u')^2 + 2F\,u'v' + G\,(v')^2}\;dt
$$

**重点：弧长完全由 $(u(t),v(t))$ 与 $E,F,G$ 决定，与三维空间无关。** 一只被「困在曲面里」的蚂蚁，沿曲面爬行测量到的长度，正是这条公式——它只看见 $E,F,G$，看不见曲面的三维摆放。<span class="marginnote">这就是「内蕴」的第一次显形：曲面上曲线的长度是曲面自身属性。同样一条曲线，如果曲面在三维里弯曲变形但保持 $E,F,G$ 不变，蚂蚁量到的长度分毫不变。第四篇的「等距变换」正是以此定义——保持 $E,F,G$ 的变换。</span>

## 3 夹角与正交

两条切向量 $v, w \in T_pS$ 的夹角 $\theta$：

$$
\cos\theta = \frac{I(v,w)}{\sqrt{I(v,v)}\sqrt{I(w,w)}} = \frac{E\,du\,\delta u + F(du\,\delta v + dv\,\delta u) + G\,dv\,\delta v}{\sqrt{E\,du^2 + 2F\,du\,dv + G\,dv^2}\;\sqrt{E\,\delta u^2 + 2F\,\delta u\,\delta v + G\,\delta v^2}}
$$

（其中 $v = du\,\mathbf{x}_u + dv\,\mathbf{x}_v$，$w = \delta u\,\mathbf{x}_u + \delta v\,\mathbf{x}_v$。）

**参数曲线的正交性**有一个漂亮结论：两族参数曲线（$u$=常数的曲线与 $v$=常数的曲线）在每一点正交 ⟺ $F = 0$。因为 $u$-曲线的切向是 $\mathbf{x}_u$，$v$-曲线的切向是 $\mathbf{x}_v$，两者夹角余弦正比于 $\mathbf{x}_u\cdot\mathbf{x}_v = F$。<span class="marginnote">$F=0$ 的坐标卡叫<strong>正交参数化</strong>（正交坐标）。它让一切公式大幅简化（长度公式、后面 Christoffel 记号都变简单）。地球的经纬度在赤道附近接近正交，但两极处退化；工程上「测地坐标系」特意构造正交参数化来简化计算。</span>

## 4 公式解析：为什么 $I(v,v) = E\,du^2 + 2F\,du\,dv + G\,dv^2$

这条式子是第一基本形式的「招牌」，逐项拆解：

- **第一步，写出切向量**：$v = du\,\mathbf{x}_u + dv\,\mathbf{x}_v$，其中 $du = u'(t)$、$dv = v'(t)$ 是曲线速度在坐标下的分量（对固定切向量而言就是两个数）。
- **第二步，内积展开**：用双线性性
  $$
  \begin{aligned}
  I(v,v) &= (du\,\mathbf{x}_u + dv\,\mathbf{x}_v)\cdot(du\,\mathbf{x}_u + dv\,\mathbf{x}_v)\\
  &= du^2\,(\mathbf{x}_u\cdot\mathbf{x}_u) + 2\,du\,dv\,(\mathbf{x}_u\cdot\mathbf{x}_v) + dv^2\,(\mathbf{x}_v\cdot\mathbf{x}_v)
  \end{aligned}
  $$
  中间的交叉项系数 2 来自「$(a+b)^2 = a^2 + 2ab + b^2$」——把内积写成平方就是两次交叉。
- **第三步，代入记号**：$\mathbf{x}_u\cdot\mathbf{x}_u = E$，$\mathbf{x}_u\cdot\mathbf{x}_v = F$，$\mathbf{x}_v\cdot\mathbf{x}_v = G$，得到
  $$
  I(v,v) = E\,du^2 + 2F\,du\,dv + G\,dv^2
  $$

**注意记号习惯**：$du^2$ 表示 $(du)^2$（微分一次方的平方），不是二阶微分。这是 Gauss 以来的传统记号，读的时候要保持清醒——$du^2$ 是「$du$ 的平方」，不是「$u$ 的二阶微分」。

**辨析｜易错点：** $E,F,G$ 依赖坐标卡，但它们描述的内积不依赖。换坐标卡，$E,F,G$ 会按坐标变换的「张量法则」变换，但内积 $I$ 本身不变。**$E,F,G$ 是坐标相关，内积是坐标无关**——这和上一节「矩阵随基变、线性映射不变」是同一个道理。

## 5 第一基本形式的几何地位

第一基本形式不是孤立的一个式子，它是曲面论整座大厦的「地基测量图」：

- **长度、夹角、面积**：全部由 $E,F,G$ 导出（本节 + 下一节）。
- **内蕴几何**：凡只依赖 $E,F,G$ 的量（测地线、Gauss 曲率的内蕴公式）不随嵌入改变——Gauss 绝妙定理（第四篇）会把这个事实推到顶点。
- **等距变换**：保持 $E,F,G$ 的变换，就是保持曲面度量结构不变的变换。
- **黎曼几何**：把 $E,F,G$ 抽象成「度量张量」$g_{ij}$，在任意流形上重写一遍，就是第八篇黎曼几何的全部起点。

**重点：第一基本形式是曲面「内在的眼睛」**——它让曲面上的居民不看三维环境也能量出全部几何。这一观念，是微分几何从「经典曲面论」跨向「现代黎曼几何」的枢纽。<span class="marginnote">在黎曼几何（第八篇）里，度量张量 $g$ 是流形上每点切空间的正定内积，连续地随点变化。第一基本形式就是「度量张量在二维曲面上的名字」。从 $E,F,G$ 到 $g_{ij}$，只是把下标从「三个」换成「张量形式」，思想完全一致。</span>

### 例：极坐标下平面度量的 $E,F,G$

看一个最简单的「非平凡度量」：用极坐标描述平面。$\mathbf{x}(r,\theta) = (r\cos\theta, r\sin\theta)$，则

$$
\mathbf{x}_r = (\cos\theta, \sin\theta), \qquad \mathbf{x}_\theta = (-r\sin\theta, r\cos\theta)
$$

$$
E = \mathbf{x}_r\cdot\mathbf{x}_r = 1, \qquad F = \mathbf{x}_r\cdot\mathbf{x}_\theta = 0, \qquad G = r^2
$$

第一基本形式

$$
I = dr^2 + r^2\,d\theta^2
$$

**重点：极坐标下平面度量有 $F=0$（正交）、$E=1$、$G=r^2$——$G$ 随 $r$ 变化，反映「离原点越远，$\theta$ 方向的长度刻度越大」。** 这是「度量系数携带坐标几何」的第一个例子：平面本身平坦（$K=0$），但极坐标的系数 $E,F,G$ 并不全为 1——**坐标的选择影响 $E,F,G$，但不影响内蕴几何**。换坐标后 $E,F,G$ 变了，度量描述的还是同一个平面。

## 6 小结

- **第一基本形式** $I_p(v,w) = v\cdot w$：切平面上的内积，继承自 $\mathbb{R}^3$。
- 坐标下由三个函数编码：$E=\mathbf{x}_u^2$、$F=\mathbf{x}_u\cdot\mathbf{x}_v$、$G=\mathbf{x}_v^2$；$I(v,v)=E\,du^2+2F\,du\,dv+G\,dv^2$。
- **长度** $L=\int\sqrt{E u'^2+2F u'v'+G v'^2}\,dt$；**夹角** $\cos\theta = I(v,w)/\sqrt{I(v,v)I(w,w)}$。
- 参数曲线正交 ⟺ $F=0$；正则 ⟺ $EG-F^2>0$。
- $E,F,G$ 是坐标相关，内积是坐标无关；第一基本形式是「内蕴几何」与黎曼几何的起点。

在下一节，我们将把第一基本形式用到极致：系统地给出**长度、夹角与面积元**的全部度量公式，并首次看到「面积元」$dA = \sqrt{EG-F^2}\,du\,dv$ 的登场。
