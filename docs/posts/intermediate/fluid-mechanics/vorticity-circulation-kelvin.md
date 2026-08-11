---
title: 涡量、环量与 Kelvin 定理
date: 2026-08-11
---

# 涡量、环量与 Kelvin 定理

<div class="epigraph">
<p>我相信涡旋原子是唯一的真实原子。</p>
<footer>—— 威廉 · 汤姆森（William Thomson，即开尔文勋爵，1867）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 流体力学 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么盯住"旋转"不放

上一章的势流理论漂亮，但只适用无旋流动。真实的流体处处有旋：河流中的漩涡、飞机机翼后缘脱落的涡、大气里的气旋。它们之间的差别不在速度大小，而在**涡量（vorticity）**——流体微团的旋转角速度。把注意力从"速度"转向"涡量"，会得到一个惊人的好处：**无粘流动中涡量输运方程变成了"冻结定理"，涡动力学比速度动力学简单得多。**<span class="marginnote">1867 年开尔文提出"涡旋原子"假说——把原子想象成以太中永恒的涡环，希望用流体力学的涡量守恒解释原子的稳定性。假说本身失败了（因为涡会耗散、结构不稳定），但它把涡量研究推上流体力学舞台中央，催生了亥姆霍兹与开尔文本人的涡定理。</span>

本章的主角是三个环环相扣的结果：**Stokes 定理**（环量与涡量的桥梁）、**Kelvin 环量定理**（无粘流动中环量守恒）、**Helmholtz 涡定理**（涡线与涡管的永久性）。它们共同构成"涡动力学"，也是升力理论、大气环流、湍流能量级联的底层语法。

## 1 涡量：旋转的"速度梯度账本"

**核心概念：** 涡量定义为速度场的旋度：

$$\boldsymbol{\omega} = \nabla \times \boldsymbol{v}$$

涡量是流体微团角速度的两倍：$\boldsymbol{\omega}=2\boldsymbol{\Omega}$。前面在《流体运动学》里把速度梯度分解为对称（应变率）与反对称（旋转）部分，反对称部分的特征向量正是 $\boldsymbol{\omega}$。<span class="marginnote">固体力学里刚体旋转用一个角速度向量 $\boldsymbol{\Omega}$ 描述；流体力学里每个微团有自己的旋转，于是角速度升级为一个<strong>场</strong> $\boldsymbol{\omega}$。涡量像头发丝一样遍布流场，方向代表旋转轴，大小代表旋转快慢。</span>

与速度场不同，涡量场是**无散的**（$\nabla\cdot\boldsymbol{\omega}=\nabla\cdot(\nabla\times\boldsymbol{v})\equiv0$），所以涡量线不能有起点终点——涡线要么闭合（涡环），要么延伸到无穷远/边界。这一拓扑约束是后面一切涡结构理论的基础。

## 2 环量：把"旋转的总量"放在一起

单看一点的涡量不够，工程与理论更关心一块区域的总旋转。**环量（circulation）**定义为速度沿闭曲线的线积分：

$$\Gamma = \oint_C \boldsymbol{v}\cdot d\boldsymbol{l}$$

Stokes 定理把环量与涡量连接起来：

$$\Gamma = \oint_C \boldsymbol{v}\cdot d\boldsymbol{l} = \iint_S (\nabla\times\boldsymbol{v})\cdot\boldsymbol{n}\,dS = \iint_S \boldsymbol{\omega}\cdot\boldsymbol{n}\,dS$$

**重点：环量是"涡量的通量"**——边界上的速度环量等于该边界围成面上涡量的总量。<span class="marginnote">这与散度定理（高斯定理）完全平行：通量是对"散度"的面积分，环量是对"旋度"的面积分。两个定理是矢量分析的孪生定理，也解释了为什么"环量"这个看似边界上的量，能用来刻画内部的旋转总量。</span>

环量的实际意义在升力理论中登峰造极：绕机翼一圈的环量 $\Gamma$ 直接决定单位展长升力 $L'=\rho U\Gamma$（库塔-茹科夫斯基定理，见《势流理论与复势》）——**飞机能飞，本质是机翼给空气一个净环量，空气还给机翼一个升力。**

## 3 Kelvin 环量定理：无粘流动的"旋转守恒"

取一个随流体质点一起运动的闭合物质线（material loop），其环量的物质导数是什么？对无粘、正压（$p=p(\rho)$）、外力守恒的流动，可以证明：

$$\frac{D\Gamma}{Dt} = \frac{D}{Dt}\oint \boldsymbol{v}\cdot d\boldsymbol{l} = 0$$

这就是**Kelvin 环量定理（Kelvin's circulation theorem）**：在理想流体中，沿任何物质闭合曲线的环量守恒。

**核心概念：** 环量是"守恒量"——不随时间改变。<span class="marginnote">物理上这来自理想流体的两项"无"：无粘（没有切应力把旋转泄掉）、正压+守恒体力（压强梯度与重力不会给流体"上劲"产生环量）。现实中海洋、大气的环量都源自这两项被破坏的时刻——边界层粘性"注入"涡量，正是机翼产生升力的物理通道。</span>凯尔文定理的一个著名推论：**无旋流动永远无旋**——若某一时刻处处无旋，之后始终无旋。这给势流理论（上一章）提供了"安全证书"：来流无旋则绕流无旋，势流解自洽成立。

## 4 Helmholtz 涡定理：涡线的"冻结"

Kelvin 定理的场版本是**Helmholtz 涡定理（Helmholtz's vortex theorems，1858）**：

1. **涡线随流体质点移动**——涡线就像被"冻结"在流体里，随流动一起变形、平移，但始终保持是涡线。
2. 涡管强度（管内的环量 $\Gamma$）沿涡管不变——"涡量是守恒的输运量"。
3. 涡管不能起止于流体内部——要么闭环，要么到边界。

**辨析｜易错点：** 涡线"冻结"与涡线"穿过流体"的区别是理解关键。在无粘理想流里，涡量是**被动输运**的：流体怎么动，涡量就怎么跟。但注意——它并不"扩散"（没有 $\nu\nabla^2\boldsymbol{\omega}$ 项）。涡量的耗散/扩散只来自粘性，这正是下一节的输运方程里的关键差异项。<span class="marginnote">想想真实的烟圈：它形成后能飘很远而不散，因为涡环的结构由 Kelvin 定理"锁住"了环量；它最终消散，则是因为空气粘性（其实很小）缓慢地扩散涡量。涡的"永生"（无粘）与"衰亡"（粘性）之争，就是这两项的拉锯。</span>

## 5 涡量输运方程：粘性把"冻结"打破

对不可压缩流动，取 Navier-Stokes 方程的旋度，消去压强项（$\nabla\times(-\nabla p)\equiv0$），得到**涡量输运方程（vorticity transport equation）**：

$$\frac{\partial\boldsymbol{\omega}}{\partial t} + (\boldsymbol{v}\cdot\nabla)\boldsymbol{\omega} = (\boldsymbol{\omega}\cdot\nabla)\boldsymbol{v} + \nu\nabla^2\boldsymbol{\omega}$$

**重点：这是 Navier-Stokes 方程的"旋转视角"。** 三项各司其职：左边是涡量的随体输运；$(\boldsymbol{\omega}\cdot\nabla)\boldsymbol{v}$ 是**涡拉伸/涡倾斜**（三维速度不均匀拉伸涡线，把涡量"拧大"——龙卷风拉长时旋转加剧）；$\nu\nabla^2\boldsymbol{\omega}$ 是**粘性扩散**。<span class="marginnote">压强从方程里消失了——这是涡量视角的第一个红利。第二个红利是直观：湍流的能量级联本质上就是涡拉伸的级联（大涡拉成小涡），见《湍流简介与 Reynolds 应力》。第三个红利是数值：很多 CFD 算法直接解这个方程（涡量-流函数法）。</span>二维流动中 $\boldsymbol{\omega}\cdot\nabla=0$（涡量与速度场垂直），涡拉伸项消失，涡量只剩"对流+扩散"，行为类似于被动标量——二维与三维湍流的根本差异就埋在这里。

## 6 公式解析：Kelvin 定理的证明骨架

$$\frac{D\Gamma}{Dt} = \frac{D}{Dt}\oint \boldsymbol{v}\cdot d\boldsymbol{l} = \oint \frac{D\boldsymbol{v}}{Dt}\cdot d\boldsymbol{l} + \oint \boldsymbol{v}\cdot\frac{D\,d\boldsymbol{l}}{Dt}$$

- **第一步，把物质导数送进积分**：积分随物质线运动，所以 $D/Dt$ 不能直接穿过积分号，必须用莱布尼茨规则——对线元 $d\boldsymbol{l}$ 本身也要取物质导数，这产生两项。
- **第二步，处理第二项**：线元的随体变化率就是速度梯度 $\frac{D\,d\boldsymbol{l}}{Dt} = (\boldsymbol{v}\cdot\nabla)\,d\boldsymbol{l}$。于是 $\boldsymbol{v}\cdot\frac{D\,d\boldsymbol{l}}{Dt} = \boldsymbol{v}\cdot[(\boldsymbol{v}\cdot\nabla)d\boldsymbol{l}]$，这是"速度大小梯度的闭路积分"，等于零（因为它是 $\nabla(\frac12 v^2)$ 沿闭路的积分）。
- **第三步，处理第一项**：对理想流体，$\frac{D\boldsymbol{v}}{Dt}=-\frac{1}{\rho}\nabla p+\boldsymbol{g}$。它的闭路积分：$\oint\nabla p\cdot d\boldsymbol{l}=0$（梯度场的闭路积分为零），正压性使 $\oint\frac{1}{\rho}\nabla p\,d\boldsymbol{l}=0$，重力项也归零。
- **第四步，合起来**：两项都为零，故 $\frac{D\Gamma}{Dt}=0$。**环量守恒的证明，本质是"梯度场闭路积分为零"这一纯矢量代数事实的两次应用。**<span class="marginnote">注意证明里"无粘"和"正压"各负责消掉一项：无粘保证 $\mu=0$ 项不存在，正压保证 $1/\rho$ 能从梯度里"提出来"。任何一个条件不满足，定理就失效——所以粘性流、非正压流里环量不守恒，这正是海洋环流、大气涡旋的成因。</span>

## 7 小结

- **涡量** $\boldsymbol{\omega}=\nabla\times\boldsymbol{v}$ 是微团角速度的两倍；涡量场无散，涡线只能闭合或到边界。
- **环量** $\Gamma=\oint\boldsymbol{v}\cdot d\boldsymbol{l}$；Stokes 定理把它与涡量通量连接。
- **Kelvin 环量定理**：理想流体中物质环量守恒；推论是无旋流永远无旋。
- **Helmholtz 涡定理**：涡线随流体质点"冻结"移动，涡管强度不变，涡管不能终止于流体内部。
- **涡量输运方程**：涡量 = 对流 + 拉伸 + 粘性扩散；压强消失，湍流级联与涡方法均由此起步。

在下一节，我们走到雷诺数的另一个极端：$Re\ll1$，惯性完全退场，Navier-Stokes 坍缩为线性的 Stokes 方程——那里有细菌的游泳、微尘的沉降与著名的 **Stokes 曳力公式**。
