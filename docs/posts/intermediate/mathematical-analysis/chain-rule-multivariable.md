---
title: 复合函数的微分法：链式法则与全微分形式不变性
date: 2026-08-07
---

# 复合函数的微分法：链式法则与全微分形式不变性

<div class="epigraph">
<p>一个变量通过中间变量间接影响函数值——多元链式法则把这种「影响的传导」拆成每一条路径的偏导数乘积之和。</p>
<footer>—— 欧拉（Leonhard Euler），《微分学原理》（节意）</footer>
</div>

<div class="article-byline">
<p>第二级 · 数学分析 ｜ 华东师大《数学分析》§17.2 ｜ 2026-08-07</p>
</div>

## 为什么多元链式法则是「反向传播」

一元链式法则 $\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}$ 只有一条链。多元复合则是一张**网**：$z=f(u,v)$、$u=\varphi(x,y)$、$v=\psi(x,y)$——$x$ 通过 $u$ 和 $v$ 两条路径影响 $z$。多元链式法则说：**总影响 = 每条路径的「偏导乘积」之和**。

这条法则正是深度学习反向传播的数学原理：**神经网络就是一个巨大的复合函数，损失对每个参数的偏导，按链式法则沿计算图逐层相乘相加地「传回去」**。从「从极限到大模型」的主线看，本节是通往反向传播的最后一站之一。<span class="marginnote">反向传播（第四级《机器学习》）就是「多元链式法则的工程实现」：把网络前向计算画成计算图，从输出端向输入端逐层应用链式法则，每个节点只做局部计算（乘一个 Jacobian）。你会在 §17.2 学到「总导数 = 各路径导数之和」，而反向传播正是这个公式在成千上万层上的自动化。<strong>学链式法则时想一次反向传播，这条主线就通了。</strong></span>

## 1 多元链式法则

**定理（链式法则 / Chain Rule）：设 $z=f(u,v)$ 可微，$u=\varphi(x,y),\ v=\psi(x,y)$ 的偏导数存在，则复合函数 $z=f(\varphi(x,y),\psi(x,y))$ 的偏导数为**

$$\frac{\partial z}{\partial x}=\frac{\partial z}{\partial u}\frac{\partial u}{\partial x}+\frac{\partial z}{\partial v}\frac{\partial v}{\partial x},$$

$$\frac{\partial z}{\partial y}=\frac{\partial z}{\partial u}\frac{\partial u}{\partial y}+\frac{\partial z}{\partial v}\frac{\partial v}{\partial y}.$$

**公式解析：为什么是「乘积之和」**

**第一步，画依赖图**。$z\to(u,v)$，$u\to x,y$，$v\to x,y$——$x$ 到 $z$ 有**两条路径**：$x\to u\to z$ 与 $x\to v\to z$；

**第二步，每条路径贡献乘积**。路径「$x\to u\to z$」的贡献是「$u$ 对 $x$ 的变化率 × $z$ 对 $u$ 的变化率」=$\frac{\partial u}{\partial x}\cdot\frac{\partial z}{\partial u}$；

**第三步，求和**。总变化率 = 两条路径贡献之和。**「路径求积、多路径求和」是链式法则的口诀**——$x$ 影响 $u$ 也影响 $v$，两个渠道都要算。

**示范**：$z=u^2v$，$u=\sin(xy)$、$v=x+y$。

$$\frac{\partial z}{\partial x}=(2uv)(y\cos(xy))+(u^2)(1)=2uv\cdot y\cos(xy)+u^2.$$

**直接代入后再求导**可以核对：$z=\sin^2(xy)(x+y)$，对 $x$ 求导（乘积法则 + 链式）结果一致。

> **辨析｜易错点：**链式法则的**每条路径都不能漏**。$x$ 出现在 $u$ 和 $v$ 里时，两条路径都要算——漏掉一条就出错。**依赖图是防漏的最佳工具**：先画「谁依赖谁」，再按图索骥。另一个易错点：**记号 $\frac{\partial z}{\partial u}$ 与 $\frac{\partial z}{\partial x}$ 的含义不同**——前者固定 $v$、后者固定 $y$，是「不同语境下的偏导」。中间变量多的复杂复合（三层以上），依赖图几乎不可省略。

## 2 单变量路径与「全导数」

当 $z=f(u,v)$ 且 $u=\varphi(t),\ v=\psi(t)$ 都只是 $t$ 的函数时，$z$ 退化为 $t$ 的一元函数，链式法则给出**全导数**：

$$\frac{dz}{dt}=\frac{\partial z}{\partial u}\frac{du}{dt}+\frac{\partial z}{\partial v}\frac{dv}{dt}.$$

注意：**左边是 $\frac{dz}{dt}$（全导数），右边是 $\frac{\partial z}{\partial u}$（偏导）**——记号差异反映「$z$ 只通过 $t$ 依赖变量」vs「$z$ 直接依赖 $u,v$」。

**示范**：$z=x^2+y^2$，$x=\cos t$、$y=\sin t$（单位圆上的函数值）：

$$\frac{dz}{dt}=2x(-\sin t)+2y(\cos t)=-2\cos t\sin t+2\sin t\cos t=0.$$

**$z=x^2+y^2=1$ 恒为常数，导数为 0**——「沿单位圆移动，$x^2+y^2$ 不变」的链式验证。

## 3 一阶全微分形式不变性

**定理（全微分形式不变性）：设 $z=f(u,v)$ 可微，$u,v$ 是自变量或可微的中间变量，则无论 $u,v$ 的身份如何，都有**

$$dz=\frac{\partial z}{\partial u}\,du+\frac{\partial z}{\partial v}\,dv.$$

**公式解析：三步拆解**

**第一步，$u,v$ 为自变量**。$dz=f_udu+f_vdv$——全微分定义（§17.1），直接成立；

**第二步，$u,v$ 为中间变量**。设 $u=\varphi(x,y)$、$v=\psi(x,y)$，则由链式法则

$$\frac{\partial z}{\partial x}=f_uu_x+f_vv_x,\qquad\frac{\partial z}{\partial y}=f_uu_y+f_vv_y,$$

于是 $dz=\frac{\partial z}{\partial x}dx+\frac{\partial z}{\partial y}dy$，代入并归并 $dx,dy$ 的系数：

$$dz=f_u(u_xdx+u_ydy)+f_v(v_xdx+v_ydy)=f_u\,du+f_v\,dv,$$

**第三步，结论**。$du=u_xdx+u_ydy$、$dv=v_xdx+v_ydy$（中间变量的全微分），代入即得 $dz=f_udu+f_vdv$——**形式与 $u,v$ 为自变量时完全一致**。

**要点**：**一阶全微分形式不变**——$dz=f_udu+f_vdv$ 无论 $u,v$ 是自变量还是中间变量都成立。这条性质让「换元求微分」变得安全：先对 $u,v$ 求微分，再代入 $du,dv$ 的展开。它与一元的一阶微分形式不变性（§5.5）完全平行。

> **辨析｜易错点：**形式不变性**只对一阶全微分成立**。高阶微分 $d^2z$ 在中间变量下**不再形式不变**（多出含 $d^2u,d^2v$ 的项，§5.6 一元教训的多元版）。所以「$d^2z=f_{uu}du^2+\cdots$」这种二阶公式只在 $u,v$ 为自变量时成立；$u,v$ 是中间变量时必须加修正项。**「一阶可换、二阶要小心」**是微分形式不变性的完整纪律。

## 4 链式法则与反向传播

把链式法则写成「反向传播」的形态。设损失 $\mathcal L=f(u,v)$，参数 $x$ 通过 $u,v$ 影响 $\mathcal L$：

$$\frac{\partial\mathcal L}{\partial x}=\frac{\partial\mathcal L}{\partial u}\frac{\partial u}{\partial x}+\frac{\partial\mathcal L}{\partial v}\frac{\partial v}{\partial x}.$$

**反向传播的计算顺序**：**先算输出端**（$\frac{\partial\mathcal L}{\partial u},\frac{\partial\mathcal L}{\partial v}$，即「损失对隐藏层输出」的梯度），**再向输入端逐层回传**（乘上 $\frac{\partial u}{\partial x}$）——**梯度从损失端「倒着流」到每个参数**。<span class="marginnote">反向传播的效率秘密在于「共享中间梯度」：$\frac{\partial\mathcal L}{\partial u}$ 只算一次，被所有「下游」参数复用（因为 $u$ 可能被多个参数影响）。这比「对每个参数单独做前向扰动」快指数级——正是「链式法则 + 动态规划」的组合。深度学习框架（PyTorch、TensorFlow）的自动求导，本质是「把链式法则编译成计算图上的后向遍历」。你在第三级《深度学习》里学反向传播时，会发现它只是今天这条公式的工程封装。</span>

**示范（一个微型网络）**：$\mathcal L=(wx+b-y)^2$，参数 $w,b$。令 $u=wx+b$、$v=u-y$，$\mathcal L=v^2$。则

$$\frac{\partial\mathcal L}{\partial w}=\frac{\partial\mathcal L}{\partial v}\frac{\partial v}{\partial u}\frac{\partial u}{\partial w}=2v\cdot1\cdot x=2(wx+b-y)x,$$

$$\frac{\partial\mathcal L}{\partial b}=2v\cdot1\cdot1=2(wx+b-y).$$

**这正是线性回归的梯度**——链式法则自动给出每个参数的偏导。

## 5 链式法则的谱系

| 情形 | 公式 |
| --- | --- |
| 一元复合 | $\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}$ |
| 二元复合（二元中间） | $\frac{\partial z}{\partial x}=f_uu_x+f_vv_x$ |
| 单变量路径（全导数） | $\frac{dz}{dt}=f_u\frac{du}{dt}+f_v\frac{dv}{dt}$ |
| 全微分形式不变 | $dz=f_udu+f_vdv$（任何身份） |

**所有公式共享「路径求积、多路径求和」**——一元是单路径，多元是多路径。

## 6 小结

- **链式法则**：$\frac{\partial z}{\partial x}=f_uu_x+f_vv_x$——各路径偏导乘积之和。
- **防漏工具**：依赖图——画「谁依赖谁」，按图索骥。
- **全导数**：$z=f(u(t),v(t))$ 时 $\frac{dz}{dt}=f_u\frac{du}{dt}+f_v\frac{dv}{dt}$——左边全导、右边偏导。
- **全微分形式不变性**：$dz=f_udu+f_vdv$ 对任何中间变量成立；**一阶可换、二阶小心**。
- **反向传播**：梯度从输出端倒流，逐层乘 Jacobian——链式法则的工程化。

在下一节，我们研究「沿任意方向的变化率」：**方向导数与梯度**。梯度是「最陡上升的方向」，它是多元微分最重要的几何对象，也是优化算法的心脏。
