## 核心结论

动量方法是在梯度下降中显式加入历史位移，用上一轮或多轮的移动方向影响当前更新。它的本质不是简单把步长调大，而是让算法在连续下降方向上积累速度，在震荡方向上互相抵消。

Heavy-ball 动量和 Nesterov 加速的核心差别不在“有没有动量”，而在“梯度在哪个点计算”。Heavy-ball 在当前点 $x_k$ 计算梯度，然后叠加历史位移；Nesterov 先根据历史位移外推到 $y_k$，再在 $y_k$ 计算梯度并修正。

| 方法 | 更新形式 | 梯度计算点 |
|---|---|---|
| GD | $x_{k+1}=x_k-\eta\nabla f(x_k)$ | $x_k$ |
| HB | $x_{k+1}=x_k-\eta\nabla f(x_k)+\beta_k(x_k-x_{k-1})$ | $x_k$ |
| NAG | $y_k=x_k+\beta_k(x_k-x_{k-1}),\;x_{k+1}=y_k-\eta\nabla f(y_k)$ | $y_k$ |

玩具例子：把梯度下降想成“每次都看脚下地形再走一步”。动量像“记住上一脚往哪走”。Heavy-ball 是带惯性往前冲；Nesterov 是先预判自己会冲到哪里，再看那里的坡度，所以更容易提前刹车。

在 $L$-smooth convex 问题中，经典 Nesterov 加速梯度法可以达到

$$
f(x_k)-f^\*\le \frac{2L\|x_0-x^\*\|^2}{(k+1)(k+2)}=O(1/k^2)
$$

这比普通梯度下降的 $O(1/k)$ 更快。但这个结论有明确边界：它属于凸、光滑、特定参数设置下的一阶优化理论，不能直接套到非凸深度学习训练中。

---

## 问题定义与边界

本文讨论的问题是最小化一个函数：

$$
\min_x f(x)
$$

其中 $f$ 是目标函数，$x$ 是参数，$x^\*$ 表示最优解，即让 $f(x)$ 取得最小值的参数。

几个术语先固定：

| 术语 | 白话解释 | 数学含义 |
|---|---|---|
| $L$-smooth | 坡度变化不会突然变得无限陡 | 梯度 Lipschitz 连续 |
| convex | 两点之间连线不会低于函数图像 | 任意局部下降方向都不会被坏局部最小值误导 |
| $x^\*$ | 最优参数 | $f(x^\*)=\min_x f(x)$ |
| $\eta$ | 每次沿梯度方向走多远 | 步长或学习率 |
| $\beta_k$ | 历史位移保留多少 | 第 $k$ 步的动量系数 |

$L$-smooth 的常用定义是：

$$
\|\nabla f(x)-\nabla f(y)\|\le L\|x-y\|
$$

它表示梯度变化有上限。这个条件很重要，因为步长 $\eta=1/L$ 的经典选择依赖它。

convex 表示函数是凸的。对任意 $x,y$ 和 $\lambda\in[0,1]$，有：

$$
f(\lambda x+(1-\lambda)y)\le \lambda f(x)+(1-\lambda)f(y)
$$

新手版本：可以把凸光滑问题想成在一个平滑的碗里找最低点。动量和 Nesterov 加速在这种地形上有清晰理论。若地形有多个坑、坡度突然变化、梯度噪声很大，比如深度网络训练，算法行为就不能只用经典凸优化结论解释。

| 场景 | 是否直接适用经典 NAG 结论 | 备注 |
|---|---|---|
| 凸光滑优化 | 是 | 可用 $O(1/k^2)$ |
| 非凸优化 | 否 | 多看驻点复杂度 |
| 深度学习训练 | 部分适用 | 常需 warmup、restart、调度器 |
| 非光滑问题 | 否 | 需要额外处理 |

真实工程例子：大规模逻辑回归、线性分类器、CTR 预估中的部分凸损失问题，通常更接近 Nesterov 理论假设。深度网络训练虽然也会使用 NAG 或带动量的 SGD，但目标函数非凸、梯度有噪声、学习率会调度，因此不能说训练误差必然按 $O(1/k^2)$ 下降。

---

## 核心机制与推导

普通梯度下降只使用当前梯度：

$$
x_{k+1}=x_k-\eta\nabla f(x_k)
$$

它的问题是，在狭长谷底中会沿陡峭方向来回摆动，沿平缓方向前进很慢。动量方法加入历史位移 $x_k-x_{k-1}$。这个量表示“上一轮从哪里走到哪里”，也就是最近一次移动方向。

Heavy-ball 的更新为：

$$
\text{HB: }\;x_{k+1}=x_k-\eta\nabla f(x_k)+\beta_k(x_k-x_{k-1})
$$

它等于“当前点梯度修正 + 历史速度”。如果连续多步梯度方向相近，历史速度会让移动更快；如果方向来回变化，历史速度会抵消一部分震荡。

Nesterov 的更新为：

$$
\text{NAG: }\;y_k=x_k+\beta_k(x_k-x_{k-1}),\quad x_{k+1}=y_k-\eta\nabla f(y_k)
$$

它先按动量外推到 $y_k$，再在 $y_k$ 计算梯度。这个差别很小，但影响很大。HB 是“到了当前位置再看坡度，同时带着惯性走”；NAG 是“先估计惯性会把自己带到哪里，再看那个位置的坡度并修正”。

更新流程可以写成：

| 方法 | 流程 |
|---|---|
| HB | 当前点求梯度 -> 加历史位移 -> 更新 |
| NAG | 先外推 -> 外推点求梯度 -> 修正更新 |

在长而窄的山谷中，HB 像带惯性的点，可能已经接近谷底仍继续前冲。NAG 在外推点 $y_k$ 看到更靠前位置的坡度，梯度修正会更早抑制过冲。因此 NAG 通常比 HB 更容易减少来回摆动。

经典 Nesterov 参数常写为：

$$
\eta=\frac{1}{L},\quad \beta_k=\frac{k-1}{k+2}
$$

在 $L$-smooth convex 条件下，可以得到：

$$
f(x_k)-f^\*\le \frac{2L\|x_0-x^\*\|^2}{(k+1)(k+2)}=O(1/k^2)
$$

这里的复杂度边界表示：若想让误差 $f(x_k)-f^\*\le \varepsilon$，NAG 需要的迭代次数量级是 $O(1/\sqrt{\varepsilon})$，而普通梯度下降通常是 $O(1/\varepsilon)$。

Nesterov 推导常使用估计序列。估计序列是一组辅助函数，用来从上方或可控方向逼近目标函数，并把每一步的下降关系串起来。一个典型形式是：

$$
\Phi_{k+1}(x)=(1-\delta_k)\Phi_k(x)+\delta_k\bigl[f(y_k)+\langle\nabla f(y_k),x-y_k\rangle\bigr]
$$

其中 $\Phi_k$ 是 surrogate，意思是用于分析的替代函数。括号里的

$$
f(y_k)+\langle\nabla f(y_k),x-y_k\rangle
$$

是 $f$ 在 $y_k$ 处的一阶线性近似。对凸函数来说，这个线性近似是全局下界。估计序列技术通过精心选择 $\delta_k$、$y_k$ 和 $x_k$，维持类似

$$
f(x_k)\le \min_x \Phi_k(x)
$$

的关系，再推出 $O(1/k^2)$ 的收敛率。

这里不需要把估计序列背成技巧。要抓住它的作用：它不是实现算法必需的代码结构，而是证明“为什么外推点求梯度能达到加速率”的分析工具。

---

## 代码实现

实现 HB 和 NAG 时需要保存两个状态：当前参数 $x_k$ 和上一步参数 $x_{k-1}$。也可以等价地保存速度变量，但用两个点更容易看出公式。

如果只记住一句代码差异，就是：HB 在 `x` 上算梯度，NAG 在 `y` 上算梯度。

```python
# Heavy-ball
x_prev = x0
x = x0
for k in range(T):
    g = grad(x)
    x_next = x - eta * g + beta * (x - x_prev)
    x_prev, x = x, x_next
```

```python
# Nesterov accelerated gradient
x_prev = x0
x = x0
for k in range(T):
    y = x + beta * (x - x_prev)
    g = grad(y)
    x_next = y - eta * g
    x_prev, x = x, x_next
```

| 变量 | 含义 | 需要保存 |
|---|---|---|
| `x_prev` | 上一步参数 | 是 |
| `x` | 当前参数 | 是 |
| `y` | 外推点 | 否，临时变量 |
| `g` | 梯度 | 否，临时变量 |

下面是一个可运行的二维二次函数例子。二次函数的最优点是原点，形式为：

$$
f(x)=\frac{1}{2}x^\top A x
$$

其中 $A$ 是正定矩阵。正定矩阵可以理解为“所有方向上都是向上的碗”。

```python
import numpy as np

A = np.array([[10.0, 0.0],
              [0.0, 1.0]])

def f(x):
    return 0.5 * x @ A @ x

def grad(x):
    return A @ x

def heavy_ball(x0, eta=0.08, beta=0.6, steps=5):
    x_prev = x0.copy()
    x = x0.copy()
    path = [x.copy()]
    for _ in range(steps):
        g = grad(x)
        x_next = x - eta * g + beta * (x - x_prev)
        x_prev, x = x, x_next
        path.append(x.copy())
    return np.array(path)

def nesterov(x0, eta=0.08, beta=0.6, steps=5):
    x_prev = x0.copy()
    x = x0.copy()
    path = [x.copy()]
    for _ in range(steps):
        y = x + beta * (x - x_prev)
        g = grad(y)
        x_next = y - eta * g
        x_prev, x = x, x_next
        path.append(x.copy())
    return np.array(path)

x0 = np.array([1.0, 1.0])
hb_path = heavy_ball(x0)
nag_path = nesterov(x0)

print("HB first 3:", hb_path[:3])
print("NAG first 3:", nag_path[:3])

assert hb_path.shape == nag_path.shape == (6, 2)
assert f(nag_path[-1]) < f(x0)
assert f(hb_path[-1]) < f(x0)
assert not np.allclose(hb_path[1], nag_path[1])
```

这个例子不是为了证明 NAG 永远优于 HB，而是为了验证两者确实在执行不同更新。第一步中，HB 直接在 $x_0$ 算梯度；NAG 会先构造外推点。真实工程里，这种差异会体现在损失曲线的震荡幅度、达到同等精度所需迭代数，以及对学习率和动量系数的敏感度上。

---

## 工程权衡与常见坑

动量不是越大越好。$\beta$ 控制历史位移保留多少，$\eta$ 控制当前梯度修正强度。两者共同决定稳定性。若 $\beta$ 大、$\eta$ 也大，算法可能不是加速，而是震荡甚至发散。

常见现象是：训练 loss 先下降，随后大幅来回抖动。这个现象不一定说明模型结构错误，更常见原因是动量太大、步长太高，或者两者组合过激。

| 问题现象 | 可能原因 | 优先处理 |
|---|---|---|
| 震荡明显 | $\beta$ 过大 | 先减 $\beta$ |
| 发散 | $\eta$ 过大 | 再减 $\eta$ |
| 前期收敛慢 | 动量不足 | 适度增大 $\beta$ |
| 非凸训练不稳定 | 目标不满足凸假设 | 加 warmup / restart / 调度 |

几个常见坑需要明确避免：

| 常见坑 | 为什么错 | 正确处理 |
|---|---|---|
| 把 HB 和 NAG 当成同一个算法 | 梯度计算点不同 | 检查梯度是在 `x` 还是 `y` 上算 |
| 把 $O(1/k^2)$ 当成非凸保证 | 该结论依赖凸性 | 非凸问题看驻点复杂度或实验曲线 |
| 只调 $\eta$ 不调 $\beta$ | 两者共同影响稳定性 | 联合调参 |
| 忽略 restart | 加速项可能积累错误方向 | 不稳定时重置动量 |
| 认为 NAG 必然比 HB 好 | 参数和问题结构会改变结果 | 用验证集和曲线判断 |

restart 是重启动量的机制，意思是在特定条件下把历史速度清空。例如当目标值上升、方向不一致、或满足某种重启准则时，将动量状态重置。它的作用不是改变 NAG 的基本定义，而是在不稳定场景下防止错误动量长期积累。

真实工程例子：在图像分类网络中使用 SGD+Nesterov 时，常见配置会包含学习率 warmup、cosine decay、weight decay 和动量。warmup 是训练初期逐步增大学习率，避免一开始梯度不稳定时更新过猛。此时 NAG 是优化器的一部分，不是单独保证收敛速度的全部原因。

---

## 替代方案与适用边界

NAG 是经典一阶加速方法。一阶方法只使用函数值和梯度，不使用 Hessian 矩阵。它在凸光滑问题上有清晰理论优势，但不是所有问题的默认最优选择。

| 方法 | 适合场景 | 优点 | 局限 |
|---|---|---|---|
| GD | 简单凸优化 | 稳定、易理解 | 慢 |
| HB | 需要惯性加速 | 实现简单 | 易震荡 |
| NAG | 凸光滑问题 | 理论加速明显 | 参数敏感，非凸需谨慎 |
| 自适应方法 | 稀疏/尺度差异大 | 鲁棒性更强 | 可能丢失部分理论最优性 |

自适应方法指 Adam、RMSProp、AdaGrad 这类优化器。它们会根据历史梯度大小自动调整不同参数维度的步长。对于稀疏特征、不同维度尺度差异很大的问题，自适应方法经常比固定学习率的 NAG 更容易调。

何时不用经典 NAG：

| 场景 | 原因 | 可考虑方案 |
|---|---|---|
| 非光滑目标 | 梯度不连续或不存在 | 近端梯度、次梯度、平滑化 |
| 梯度噪声很大 | 外推点梯度可能更不稳定 | SGD 动量、Adam、调小动量 |
| 强非凸训练 | 凸加速率不成立 | warmup、restart、学习率调度 |
| 强约束问题 | 更新后可能离开可行域 | 投影梯度、约束优化方法 |
| 需要极强鲁棒性 | 加速可能放大不稳定 | 保守步长或自适应优化器 |

新手版本：如果问题本身是“山路平滑、路径长”，NAG 往往更合适；如果问题是“地形噪声大、坑很多、信号不稳定”，就不能只靠 NAG 的经典形式硬推。

一个反例式提醒：即使在简单一维二次函数 $f(x)=\frac{1}{2}x^2$ 上，若 $\eta$ 和 $\beta$ 取值过大，动量项也可能导致来回震荡。加速方法不是免调参方法。理论里的 $\eta=1/L$ 和 $\beta_k=(k-1)/(k+2)$ 来自特定假设；工程实现里的固定 $\beta=0.9$ 或 $0.99$ 是经验配置，需要结合目标函数、batch size、梯度噪声和学习率调度一起判断。

---

## 参考资料

| 文献 | 作用 | 建议阅读顺序 |
|---|---|---|
| Nesterov, *Introductory Lectures on Convex Optimization* | 建立基本定义、光滑凸优化与加速主线 | 1 |
| Nesterov 1983 原始条目 | 理解加速梯度方法的来源 | 2 |
| Heavy-ball 的现代 Lyapunov 分析工作 | 理解 HB 稳定性和二次函数行为 | 3 |
| Estimate sequence 相关论文 | 理解估计序列如何推出加速率 | 4 |
| 非凸加速与 restart 相关论文 | 理解非凸场景的边界和工程改进 | 5 |

如果只想建立一条稳妥学习路径，先读教材中的 Nesterov 加速章节，再看原始论文和估计序列相关材料，最后读非凸加速与 restart 的现代工作。

- Nesterov, *Introductory Lectures on Convex Optimization*, Springer.
- Nesterov, “A method for solving the convex programming problem with convergence rate $O(1/k^2)$”, 1983.
- Polyak, “Some methods of speeding up the convergence of iteration methods”, 1964.
- An accelerated Lyapunov function for Polyak’s Heavy-ball on convex quadratics.
- Estimate sequence methods for accelerated gradient iterates.
- Accelerated Gradient Methods for Nonconvex Nonlinear and Stochastic Programming.
- Restarted Nonconvex Accelerated Gradient Descent.
