---
title: 正交曲线坐标与度量系数
date: 2026-08-11
---

# 正交曲线坐标与度量系数

<div class="epigraph">
<p>坐标系的自由，是物理学家最常被忽视的武器。</p>
<footer>—— 卡尔 · 雅可比（Carl Gustav Jacob Jacobi）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础科学 · 向量与张量初步 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从坐标系更换开始

前几讲的全部公式都写在直角坐标里。但物理问题天生有对称性：原子是球对称的，导线是柱对称的。硬用直角坐标描述球对称问题，就像用方形盘子盛汤——不是不能，是处处别扭。**坐标系是描述工具的带宽，选择它应该服从问题的形状，而不是反过来。**<span class="marginnote">本讲对应 Arfken 第 2 章 §2.1–§2.5。球坐标与柱坐标是其中最常用的两种；本讲给出的度量系数方法适用于一切正交曲线坐标。</span>

更换坐标不是「换个标签」那么轻松：坐标微元 $dq_i$ 与实际长度微元 $ds_i$ 之间有一个**拉伸比例**——这就是**度量系数（scale factor）**。梯度、散度、旋度在这些坐标系里长什么样，全部由度量系数决定。学懂这一讲，你会获得一个可以「自动生成」任何正交坐标系下算子公式的机器。

## 1 曲线坐标：用三族曲面定位

在三维空间里，用三个参数 $(q_1, q_2, q_3)$ 定位一个点，每个参数固定时给出一个坐标曲面：

- **直角坐标**：$q_1=x,q_2=y,q_3=z$，坐标曲面是三族平行平面。
- **柱坐标** $(\rho, \phi, z)$：$\rho=$ 常数是同心圆柱面，$\phi=$ 常数是过 $z$ 轴的半平面，$z=$ 常数是水平面。
- **球坐标** $(r, \theta, \phi)$：$r=$ 常数是同心球面，$\theta=$ 常数是圆锥面，$\phi=$ 常数是过 $z$ 轴的半平面。

位置向量 $\mathbf r(q_1,q_2,q_3)$ 的微分

$$
d\mathbf r = \frac{\partial \mathbf r}{\partial q_1}dq_1 + \frac{\partial \mathbf r}{\partial q_2}dq_2 + \frac{\partial \mathbf r}{\partial q_3}dq_3
$$

三个偏导 $\partial\mathbf r/\partial q_i$ 分别是「只沿 $q_i$ 增大方向」的切向量。它们不必是单位向量——长度就是度量系数。

## 2 度量系数：长度微元与坐标微元的比值

**度量系数（scale factor）**：

$$
h_i = \left|\frac{\partial \mathbf r}{\partial q_i}\right|
$$

于是沿 $q_i$ 方向的真实长度微元是 $ds_i = h_i\, dq_i$。三个坐标方向的正交单位向量 $(\hat{\mathbf e}_1,\hat{\mathbf e}_2,\hat{\mathbf e}_3)$ 满足

$$
\hat{\mathbf e}_i = \frac{1}{h_i}\frac{\partial \mathbf r}{\partial q_i}
$$

在**正交曲线坐标**里，这三个单位向量处处互相垂直，位置微分简化为<span class="marginnote">直角坐标的 $h_i\equiv 1$，所以从来没人注意到「$dx$ 就是长度」这件事——坐标太舒服时，物理结构反而被藏起来了。</span>

$$
d\mathbf r = h_1 dq_1\, \hat{\mathbf e}_1 + h_2 dq_2\, \hat{\mathbf e}_2 + h_3 dq_3\, \hat{\mathbf e}_3
$$

**线元**、**体积元**随之而来：

$$
ds^2 = h_1^2 dq_1^2 + h_2^2 dq_2^2 + h_3^2 dq_3^2, \qquad dV = h_1 h_2 h_3\, dq_1 dq_2 dq_3
$$

体积元公式的直觉：在 $P$ 处沿三个坐标方向各走 $dq_i$，扫出的是一个小「六面体」，边长分别是 $h_1dq_1, h_2dq_2, h_3dq_3$，正交时体积恰为三边之积。

## 3 三大坐标系：把度量系数装进口袋

最常用的两组度量系数必须熟记：

| 坐标系 | $q_1,q_2,q_3$ | 度量系数 $h_1,h_2,h_3$ | 体积元 |
| --- | --- | --- | --- |
| 直角 | $x,y,z$ | $1,1,1$ | $dx\,dy\,dz$ |
| 柱 | $\rho,\phi,z$ | $1,\ \rho,\ 1$ | $\rho\,d\rho\,d\phi\,dz$ |
| 球 | $r,\theta,\phi$ | $1,\ r,\ r\sin\theta$ | $r^2\sin\theta\,dr\,d\theta\,d\phi$ |

记住它们的诀窍是几何：柱坐标里「绕 $z$ 轴转一点角度」扫过的弧长是 $\rho\,d\phi$；球坐标里「沿 $\theta$ 方向」扫过的是 $r\,d\theta$，「沿 $\phi$ 方向」扫过的是 $r\sin\theta\,d\phi$。<span class="marginnote">$r\sin\theta$ 就是「到 $z$ 轴的距离」——它负责把「绕轴转动」翻译成真实的弧长。</span>

## 4 微分算子公式：度量系数驱动的通用机器

有了度量系数，一切算子都可以机械地写出来（$V_i$ 是 $\mathbf V$ 沿 $\hat{\mathbf e}_i$ 的分量）：

$$
\nabla\phi = \frac{1}{h_1}\frac{\partial\phi}{\partial q_1}\hat{\mathbf e}_1 + \frac{1}{h_2}\frac{\partial\phi}{\partial q_2}\hat{\mathbf e}_2 + \frac{1}{h_3}\frac{\partial\phi}{\partial q_3}\hat{\mathbf e}_3
$$

$$
\nabla\cdot\mathbf V = \frac{1}{h_1 h_2 h_3}\left[\frac{\partial(h_2 h_3 V_1)}{\partial q_1} + \frac{\partial(h_1 h_3 V_2)}{\partial q_2} + \frac{\partial(h_1 h_2 V_3)}{\partial q_3}\right]
$$

$$
\nabla^2\phi = \frac{1}{h_1 h_2 h_3}\left[\frac{\partial}{\partial q_1}\left(\frac{h_2 h_3}{h_1}\frac{\partial\phi}{\partial q_1}\right) + \frac{\partial}{\partial q_2}\left(\frac{h_1 h_3}{h_2}\frac{\partial\phi}{\partial q_2}\right) + \frac{\partial}{\partial q_3}\left(\frac{h_1 h_2}{h_3}\frac{\partial\phi}{\partial q_3}\right)\right]
$$

三个式子共享一个结构：**凡是「法向面积元」出现的地方（$h_2h_3$ 等），就乘进对相应坐标的求导里**。散度里 $V_1$ 乘以「$q_1$ 面元面积」$h_2h_3$，拉普拉斯里 $\partial\phi/\partial q_1$ 除以 $h_1$ 又乘回 $h_2h_3$——这些系数正是 Gauss 定理中「相邻面通量要按面积加权」的忠实执行。

## 5 公式解析：球坐标下的拉普拉斯算子

把球坐标度量系数 $h_r=1,\ h_\theta=r,\ h_\phi=r\sin\theta$ 代入拉普拉斯公式：

$$

\nabla^2\phi = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial\phi}{\partial r}\right) + \frac{1}{r^2\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial\phi}{\partial\theta}\right) + \frac{1}{r^2\sin^2\theta}\frac{\partial^2\phi}{\partial\phi^2}

$$

逐项解剖：

- **第一步，看 $r$ 项**：$\dfrac{1}{r^2}\dfrac{\partial}{\partial r}\left(r^2\dfrac{\partial\phi}{\partial r}\right)$ 不是简单的 $\partial^2/\partial r^2$，多出的 $1/r^2\cdot r^2$ 来自「球壳面积随 $r^2$ 增长」——同样大小的温度梯度，在半径更大处对应更大面积的球壳，通量被稀释。
- **第二步，看 $\theta$ 项**：$\dfrac{1}{r^2\sin\theta}\dfrac{\partial}{\partial\theta}\left(\sin\theta\dfrac{\partial\phi}{\partial\theta}\right)$。因子 $1/r^2$ 给 $\theta$、$\phi$ 项统一的「维数」，让三项相加后各项都是「单位长度平方分之一」。
- **第三步，看 $\phi$ 项**：$\dfrac{1}{r^2\sin^2\theta}\dfrac{\partial^2\phi}{\partial\phi^2}$。分母里 $\sin^2\theta$ 提醒你：越接近极点，绕 $z$ 轴一步扫过的弧长越短，同样的 $\phi$ 变化对应更小的实际长度，梯度被「放大」。

这条式子几乎是物理学最常出现的公式之一：氢原子薛定谔方程、亥姆霍兹方程、静电学、天体引力问题全部经由它。**记不住没关系——只要记住「度量系数代入通用机器」这条生成规则，任何坐标系的算子公式都能当场推出来。**

## 6 小结

- **曲线坐标**：用三族坐标曲面定位；柱坐标、球坐标由问题的对称性决定。
- **度量系数** $h_i = |\partial\mathbf r/\partial q_i|$：真实长度微元与坐标微元的比值；线元 $ds^2 = \sum h_i^2 dq_i^2$，体积元 $dV = h_1h_2h_3\,dq_1dq_2dq_3$。
- **三坐标度量系数**：直角 $(1,1,1)$；柱 $(1,\rho,1)$；球 $(1,r,r\sin\theta)$。
- **通用机器**：梯度、散度、拉普拉斯都能由度量系数统一生成；球坐标拉普拉斯算子是物理学的常客。

在下一节，我们要回答一个更深刻的问题：这些「坐标变换下」的法则，能不能只依赖一套统一的记号，与坐标系的选择彻底脱钩？能——把指标写在字母上、让求和符号自己循环，这就是**张量定义与指标记号**。
