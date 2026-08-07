---
title: 多元复合函数的求导法则
date: 2026-08-07
---

# 多元复合函数的求导法则

<div class="epigraph">
<p>变化沿着依赖之链传播——这就是链式法则。</p>
<footer>—— 戈特弗里德 · 威廉 · 莱布尼茨（Gottfried Wilhelm Leibniz）</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等数学 ｜ 同济《高等数学》下册 §9.4 ｜ 2026-08-07</p>
</div>

## 为什么从多元复合函数的求导法则开始

一元链式法则 $\frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx}$ 你已经熟练；多元情形下，变量之间的依赖关系变成了一张「网络」——$z$ 依赖 $u,v$，$u,v$ 又依赖 $x,y$。**多元复合函数的求导法则**就是这张依赖网络上的传播规则：沿每条「依赖路径」把导数相乘，所有路径相加。它是整个反向传播算法的数学核心——深度神经网络里的梯度回传，正是多元链式法则的大规模执行。<span class="marginnote">多元链式法则的本质：<strong>「变化沿着依赖路径传播，路径间相加」</strong>。若 $z$ 经 $u$ 和 $v$ 两条路径依赖 $x$，则 $\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial z}{\partial v}\frac{\partial v}{\partial x}$——每条路径一项，路径越多项越多。这个「路径求和」结构就是计算图反向传播的数学形式。</span>

## 1 中间变量是多个、自变量是一个

若 $z = f(u,v)$，$u = u(x)$，$v = v(x)$，则 $z$ 是 $x$ 的一元复合函数，全导数为

$$\frac{dz}{dx} = \frac{\partial z}{\partial u}\frac{du}{dx} + \frac{\partial z}{\partial v}\frac{dv}{dx}$$

**重点：这里用「全导数」$\frac{dz}{dx}$（不是偏导）**——因为 $z$ 最终只是 $x$ 的一元函数。中间用偏导（对 $u,v$）、外层用全导（对 $x$），正是「路径求和」的第一种形态。

例：$z = u^2v$，$u = \sin x$，$v = e^x$。$\frac{dz}{dx} = 2uv\cos x + u^2 e^x$。

## 2 中间变量与自变量都是多个

一般情形：$z = f(u,v)$，$u = u(x,y)$，$v = v(x,y)$。则

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial z}{\partial v}\frac{\partial v}{\partial x}$$

$$\frac{\partial z}{\partial y} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial y} + \frac{\partial z}{\partial v}\frac{\partial v}{\partial y}$$

**规则**：$\frac{\partial z}{\partial x}$ = 对每条「从 $z$ 经中间变量到 $x$ 的路径」，把路径上各导数相乘，再对所有路径求和。用**链式图**（依赖关系图）可以一目了然：$z \to u \to x$、$z \to u \to y$、$z \to v \to x$、$z \to v \to y$。<span class="marginnote">画「依赖图」是多元链式法则最实用的工具：每个变量一个节点，箭头表示依赖，求 $\frac{\partial z}{\partial x}$ 就找「$z$ 到 $x$」的所有路径，每条路径的导数相乘再相加。到第四级《深度学习》，计算图的反向传播正是把这个规则机械化。</span>

## 3 多元链式法则的变形

**中间变量即自变量**：$z = f(x, y)$，$y = y(x)$，则 $z$ 是 $x$ 的一元函数

$$\frac{dz}{dx} = f_x + f_y\frac{dy}{dx}$$

这里 $f_x$ 是「$z$ 直接随 $x$ 的部分」，$f_y\frac{dy}{dx}$ 是「经 $y$ 间接随 $x$ 的部分」——**直路 + 弯路，两路相加**。

**全微分形式不变性**：$dz = f_u du + f_v dv$ 无论 $u,v$ 是自变量还是中间变量都成立——与一元「一阶微分形式不变性」平行。<span class="marginnote">一阶微分形式不变性是链式法则的「封装」：先按外层函数写 $dz = f_u du + f_v dv$，再展开 $du = u_x dx + u_y dy$、$dv = v_x dx + v_y dy$，代入即得偏导公式。先写形式、后补细节——这个「延迟展开」的技巧让复杂的链式求导变得有条理。</span>

## 4 公式解析：二阶偏导的链式应用

设 $z = f(u,v)$，$u = xy$，$v = \frac{x}{y}$，求 $\frac{\partial z}{\partial x}$、$\frac{\partial z}{\partial y}$：

- **第一步，画依赖图**：$z \to u \to x,y$；$z \to v \to x,y$。
- **第二步，写 $\frac{\partial z}{\partial x}$**：$\frac{\partial z}{\partial x} = f_u\cdot\frac{\partial u}{\partial x} + f_v\cdot\frac{\partial v}{\partial x} = f_u\cdot y + f_v\cdot\frac{1}{y}$。
- **第三步，写 $\frac{\partial z}{\partial y}$**：$\frac{\partial z}{\partial y} = f_u\cdot x + f_v\cdot\left(-\frac{x}{y^2}\right)$。
- **第四步，验证一阶微分不变性**：$dz = f_u(x\,dy + y\,dx) + f_v\left(\frac{dx}{y} - \frac{x\,dy}{y^2}\right)$，整理后 $dx$ 系数即 $\frac{\partial z}{\partial x}$、$dy$ 系数即 $\frac{\partial z}{\partial y}$，一致。

**关键**：求二阶偏导（如 $\frac{\partial^2 z}{\partial x^2}$）时，**$f_u, f_v$ 仍是 $u,v$ 的函数**，要继续用链式法则求它们的偏导——这是多元链式最繁琐也最易错的地方，务必牢记「$f_u$ 里还有 $u,v$，而 $u,v$ 又依赖 $x,y$」。

## 5 多元链式法则与反向传播

多元链式法则是深度学习的数学引擎：

- **前向传播**：输入 $x$ 经各层参数 $W$ 逐层变换到输出 $\hat y$——正是复合函数 $z = f_3(f_2(f_1(x)))$。
- **反向传播**：损失 $L$ 对每个参数的偏导，沿「$L \to$ 各层」的依赖路径用链式法则逐层回传。<span class="marginnote">反向传播 = 多元链式法则在计算图上的「工程化」：每一层只需计算「局部导数」（本层输出对输入的导数）并乘以上游传来的梯度。这就是「局部计算、全局传播」——你在这里学的路径求和规则，是 PyTorch 自动求导的数学基础。</span>
- **Jacobian 矩阵**：多元函数的「导数」是 Jacobian，链式法则在向量形式下就是「Jacobian 矩阵相乘」——正是反向传播矩阵化后的计算。

**应用：全导数的物理意义**。若 $z = f(x,y)$ 而 $x, y$ 都随时间 $t$ 变化（如质点的温度 $T(x(t),y(t))$），则

$$\frac{dT}{dt} = T_x \frac{dx}{dt} + T_y \frac{dy}{dt}$$

——沿轨迹的温度变化率 = 各方向的偏导 × 该方向的速度分量之和，这是「全导数」在运动学里的标准用法，也通向方向导数（第 54 节）的直觉。

## 6 小结

- **多元链式法则**：沿每条依赖路径相乘、对所有路径求和。
- $z = f(u,v)$，$u,v$ 依赖 $x$：$\frac{dz}{dx} = f_u\frac{du}{dx} + f_v\frac{dv}{dx}$（全导数）。
- $u,v$ 都依赖 $x,y$：$\frac{\partial z}{\partial x} = f_u u_x + f_v v_x$，同理对 $y$。
- **一阶微分形式不变性**：$dz = f_u du + f_v dv$ 恒成立，展开即得偏导。
- 二阶偏导要继续对 $f_u, f_v$ 用链式法则；链式法则是反向传播的数学核心。

在下一节，我们将用偏导求解「方程隐含着函数」的求导问题——**隐函数的求导公式**。
