## 核心结论

近端梯度法（Proximal Gradient Method）处理的是一类分解目标：
$$
F(x)=f(x)+g(x)
$$
其中 $f$ 是光滑项，白话说就是“梯度变化不至于突然失控”的部分；$g$ 是非光滑项，白话说就是“可能带绝对值、阈值、稀疏约束这类尖点结构”的部分。

它的核心价值不在于把问题“变简单”，而在于把两类难点拆开处理：

1. 对 $f$ 做普通梯度下降，因为它可导、局部线性近似可靠。
2. 对 $g$ 做近端更新，因为它不可微，但常常有结构化闭式解。

对应更新式是
$$
x_{t+1}=\operatorname{prox}_{\alpha g}\bigl(x_t-\alpha \nabla f(x_t)\bigr)
$$
其中 $\alpha$ 是步长，白话说就是“每次走多远”。

这一步可以理解成“先走再拉回”：先按 $f$ 的梯度往下降方向走一步，再用近端算子把解拉回到 $g$ 偏好的结构上。对稀疏学习，这个“拉回”通常就是把小权重直接压成零。

最常见的例子是 L1 正则：
$$
g(x)=\lambda \|x\|_1
$$
它的近端算子是软阈值：
$$
\operatorname{prox}_{\alpha g}(v)_i=\operatorname{sign}(v_i)\max(|v_i|-\lambda\alpha,0)
$$
这意味着每一维都先减去阈值 $\lambda\alpha$，不够大的直接归零。

在合适步长下，ISTA 的目标函数误差收敛率是 $O(1/t)$；FISTA 在此基础上加入 Nesterov 外插，白话说就是“利用前几步趋势做预测”，可提升到 $O(1/t^2)$。但 FISTA 更快不代表总是更稳，非强凸或非凸问题中常见振荡，工程上通常要配合重启策略。

| 方法 | 更新核心 | 理论收敛率 | 适合场景 | 风险 |
|---|---|---:|---|---|
| ISTA | 梯度步 + 近端步 | $O(1/t)$ | 稳定、易实现 | 速度偏慢 |
| FISTA | ISTA + 动量外插 | $O(1/t^2)$ | 大规模凸优化 | 易振荡 |
| 纯梯度下降 | 只处理可导目标 | 依问题而定 | 无非光滑项 | 不能直接处理 L1 等 |

---

## 问题定义与边界

近端梯度法不是“任何优化问题都能套”的通用模板，它有明确边界。

第一，$f$ 需要有 Lipschitz 连续梯度。这个术语的白话意思是：梯度不能变化得过快，存在某个常数 $L_f$ 使得
$$
\|\nabla f(x)-\nabla f(y)\|\le L_f\|x-y\|
$$
这样做梯度步时，局部二次上界才成立，下降才有保证。

第二，$g$ 不要求可导，但必须“近端可算”。所谓近端可算，白话说就是：
$$
\operatorname{prox}_{\alpha g}(v)=\arg\min_z\left\{g(z)+\frac{1}{2\alpha}\|z-v\|^2\right\}
$$
这个子问题要么有闭式解，要么能非常便宜地解出来。若每次近端都要再嵌一个重优化器，整体算法就失去意义。

第三，步长要受 $L_f$ 限制。对标准凸问题，常用条件是
$$
0<\alpha\le \frac{1}{L_f}
$$
如果 $\alpha$ 过大，梯度那一步就可能直接越过下降区，导致目标值不降反升。若 $L_f$ 不好估计，工程上通常用回溯线搜索，白话说就是“先大胆走，走坏了再缩步长”。

一个新手可理解的图景是：

- $f$ 像平滑山坡，梯度告诉你局部向下走的方向。
- $g$ 像结构边界，规定你最终更偏好稀疏、低秩、分组或满足某种掩码。
- 近端算子就是每次走完后，把你重新拉回这个结构偏好区域。

玩具例子可以直接看二维向量。设初值
$$
x_0=[2,-1]^\top,\quad \lambda=1,\quad \alpha=0.5
$$
假设某一步梯度后得到
$$
v=x_0-\alpha \nabla f(x_0)=[1.5,-0.5]^\top
$$
再做 L1 近端：
$$
x_1=\operatorname{prox}_{\alpha\lambda\|\cdot\|_1}(v)=[1,0]^\top
$$
第一维从 $1.5$ 缩到 $1.0$，第二维因为刚好落到阈值边界，直接变成 $0$。这就是“先走再拉回”的最小样例。

---

## 核心机制与推导

近端算子的定义是
$$
\operatorname{prox}_{\alpha g}(v)=\arg\min_z\left\{g(z)+\frac{1}{2\alpha}\|z-v\|^2\right\}
$$
第二项是二次惩罚，白话说就是“别离当前梯度步结果 $v$ 太远”；第一项是结构偏好，白话说就是“但也别忘了稀疏、平滑、分组这些要求”。

### 从局部近似得到 ISTA

对光滑项 $f$，在 $x_t$ 处做一阶近似并加二次上界：
$$
f(z)\approx f(x_t)+\nabla f(x_t)^\top(z-x_t)+\frac{1}{2\alpha}\|z-x_t\|^2
$$
于是原问题的下一步可以近似为
$$
x_{t+1}=\arg\min_z\left\{
\nabla f(x_t)^\top(z-x_t)+\frac{1}{2\alpha}\|z-x_t\|^2+g(z)
\right\}
$$
整理二次项可得
$$
x_{t+1}=\arg\min_z\left\{
g(z)+\frac{1}{2\alpha}\|z-(x_t-\alpha\nabla f(x_t))\|^2
\right\}
$$
这正是
$$
x_{t+1}=\operatorname{prox}_{\alpha g}(x_t-\alpha\nabla f(x_t))
$$

### L1 近端为什么是软阈值

取
$$
g(z)=\lambda\|z\|_1=\lambda\sum_i |z_i|
$$
则近端子问题可按坐标分解：
$$
\min_{z_i}\ \lambda|z_i|+\frac{1}{2\alpha}(z_i-v_i)^2
$$
每一维独立求解。结论是：

- 若 $v_i>\lambda\alpha$，最优值为 $z_i=v_i-\lambda\alpha$
- 若 $v_i<-\lambda\alpha$，最优值为 $z_i=v_i+\lambda\alpha$
- 若 $|v_i|\le \lambda\alpha$，最优值为 $z_i=0$

合并写成
$$
z_i=\operatorname{sign}(v_i)\max(|v_i|-\lambda\alpha,0)
$$
这叫软阈值。术语“软”是因为它不是简单裁剪，而是把超过阈值的量整体向零收缩。

### L2 近端为什么只是缩放

若
$$
g(z)=\frac{\lambda}{2}\|z\|_2^2
$$
则近端问题为
$$
\min_z\ \frac{\lambda}{2}\|z\|^2+\frac{1}{2\alpha}\|z-v\|^2
$$
一阶条件直接给出
$$
\lambda z+\frac{1}{\alpha}(z-v)=0
$$
所以
$$
z=\frac{1}{1+\lambda\alpha}v
$$
L2 近端不是制造稀疏，而是整体缩小幅值。

### ISTA 与 FISTA 对比

| 方法 | 更新式 |
|---|---|
| ISTA | $x_{t+1}=\operatorname{prox}_{\alpha g}(x_t-\alpha\nabla f(x_t))$ |
| FISTA | 先在外插点 $y_t$ 算梯度，再做近端：$x_{t+1}=\operatorname{prox}_{\alpha g}(y_t-\alpha\nabla f(y_t))$ |
| FISTA 动量 | $t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2},\ y_{k+1}=x_{k+1}+\frac{t_k-1}{t_{k+1}}(x_{k+1}-x_k)$ |

FISTA 的逻辑不是改变近端，而是在“在哪个点算梯度”上做加速。它利用历史方向进行外插，因此理论更快，但也更容易因为预测过头而抖动。

一个简化伪代码是：

```text
x0 = init
y0 = x0
t0 = 1
for k = 0,1,2,...
    x_{k+1} = prox_{alpha g}(y_k - alpha * grad_f(y_k))
    t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2
    y_{k+1} = x_{k+1} + ((t_k - 1) / t_{k+1}) * (x_{k+1} - x_k)
```

如果发现目标值上升，工程上常做“重启”：把 $y_{k+1}$ 直接重置成 $x_{k+1}$，暂时取消动量。

---

## 代码实现

最基础的实现是 Lasso，也就是
$$
\min_x \frac12\|Ax-b\|_2^2+\lambda\|x\|_1
$$
其中 $\frac12\|Ax-b\|_2^2$ 是拟合误差，白话说就是“预测和真实差多少”；$\lambda\|x\|_1$ 是稀疏正则，白话说就是“尽量少用特征”。

每轮更新分两步：

1. 梯度步：$v=x-\alpha A^\top(Ax-b)$
2. 近端步：对 $v$ 做软阈值

这正是“先走再拉回”。

下面给出一个可运行的 Python 版本，包含 `assert` 校验：

```python
import numpy as np

def prox_l1(v, lam, alpha):
    return np.sign(v) * np.maximum(np.abs(v) - lam * alpha, 0.0)

def objective(A, b, x, lam):
    residual = A @ x - b
    return 0.5 * residual @ residual + lam * np.abs(x).sum()

def ista_lasso(A, b, lam, alpha, max_iter=200):
    x = np.zeros(A.shape[1], dtype=float)
    history = []

    for _ in range(max_iter):
        grad = A.T @ (A @ x - b)
        v = x - alpha * grad
        x = prox_l1(v, lam, alpha)
        history.append(objective(A, b, x, lam))

    return x, history

# 玩具数据
A = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])
b = np.array([1.0, -0.2, 0.7])

# 对二次项，L = ||A^T A||_2
L = np.linalg.norm(A.T @ A, 2)
alpha = 1.0 / L
lam = 0.2

x, hist = ista_lasso(A, b, lam, alpha, max_iter=100)

# 软阈值的基本性质
assert np.allclose(prox_l1(np.array([1.5, -0.5]), 1.0, 0.5), np.array([1.0, -0.0]))

# 目标值应整体下降到较稳定区间
assert hist[-1] <= hist[0] + 1e-8
assert objective(A, b, x, lam) <= objective(A, b, np.zeros(2), lam) + 1e-8

print("solution:", x)
print("final objective:", hist[-1])
```

参数含义如下：

| 参数 | 含义 | 工程建议 |
|---|---|---|
| $x_0$ | 初始解 | 常用零向量或上轮 warm start |
| $\alpha$ | 步长 | 优先取 $1/L_f$ 或回溯线搜索 |
| $\lambda$ | 正则强度 | 越大越稀疏，但偏差更大 |
| `max_iter` | 最大迭代数 | 需配合停止准则，不宜只看轮数 |

如果要改成 FISTA，只需增加两个状态量：历史解和动量系数。近端函数本身不变，这正是近端梯度框架的工程优势：梯度模块和结构模块是解耦的。

真实工程例子是 LoRA 剪枝或稀疏注意力。LoRA 是低秩适配，白话说就是“只训练一个小的低秩增量矩阵，而不是全量改大模型权重”。当希望让 LoRA 增量进一步稀疏时，可以把训练目标写成
$$
\mathcal{L}(W)+\lambda\|W\|_1
$$
然后对数据损失 $\mathcal{L}$ 做梯度，对稀疏项做近端。这样不会把稀疏性硬塞进优化器内部，而是显式地在每步更新后执行稀疏投影式压缩。

---

## 工程权衡与常见坑

近端梯度法在公式上简洁，但工程质量高度依赖细节。

| 问题 | 典型表现 | 原因 | 缓解措施 |
|---|---|---|---|
| 步长过大 | 目标值震荡、不下降 | 不满足 $\alpha \le 1/L_f$ | 回溯线搜索、谱范数估计 |
| 步长过小 | 收敛很慢 | 下降保守 | 自适应步长、FISTA 加速 |
| prox 不易解 | 单步代价太高 | $g$ 结构过复杂 | 换近似 prox、换 ADMM |
| FISTA 振荡 | 目标值上下跳 | 外插过强 | 单调 FISTA、重启 |
| 稀疏过强 | 精度明显下降 | $\lambda$ 太大 | 做正则路径扫描 |
| 数值尺度差 | 某些维度不收敛 | 特征未归一化 | 预处理、对角预条件 |

最常见的坑是把“能写出 prox 定义”和“能高效算 prox”混为一谈。定义总能写，但若每次都要求解复杂组合约束，算法可能还不如直接换方法。

### 2:4 稀疏化的工程场景

2:4 稀疏化指每连续 4 个权重里只保留 2 个非零。白话说就是“硬件喜欢这种固定模式稀疏，因为访存和算子都更规整”。这类结构常见于 AI 加速器友好的模型压缩。

若把每组 4 个权重记为 $w^{(j)}\in\mathbb{R}^4$，目标可以写成
$$
\min_W f(W)+g(W)
$$
其中 $g$ 不再是简单 L1，而是“每组最多 2 个非零”的结构惩罚或约束。此时近端步通常变成：对每个长度为 4 的小组，保留绝对值最大的 2 个元素，其余置零。它仍然保持“分组可分解”，所以能嵌进近端框架。

这类方法的价值不是理论形式更漂亮，而是每轮更新都直接产出硬件可执行的掩码结构。相比训练完再后处理剪枝，近端式训练能更早让模型适应目标稀疏模式。

但这里也有边界：2:4 约束已经不是简单凸正则，FISTA 的动量可能导致结构切换频繁，出现训练不稳。工程上常见做法是：

- 前期只做软稀疏正则，后期再切到结构化 prox
- 或者在固定间隔才触发一次 2:4 prox，而不是每步都做
- 或者一旦目标值回升就重启动量

---

## 替代方案与适用边界

近端梯度法适合“$f$ 光滑、$g$ 的 prox 简单”这类问题。一旦近端子问题本身变复杂，就要考虑替代方案。

| 方法 | 约束适配 | 优点 | 限制 |
|---|---|---|---|
| ISTA/FISTA | 光滑损失 + 简单 prox | 实现简单、结构清晰 | 复杂 prox 时失效 |
| ADMM | 可拆成多个子变量与约束 | 对复杂约束更灵活 | 每轮有额外乘子与子问题 |
| 原始-对偶方法 | 适合线性算子复合项 | 处理约束更自然 | 参数调节更敏感 |
| 投影梯度法 | $g$ 是硬约束集合指标函数 | 投影直观 | 投影本身可能困难 |
| 二阶或拟牛顿近端法 | 强凸或病态问题 | 迭代轮数少 | 单步代价高 |

### 什么时候继续用 ISTA/FISTA

- $g$ 是 L1、组 Lasso、核范数、简单投影约束
- 数据规模大，单步必须便宜
- 需要可解释、易维护的训练流程

### 什么时候换 ADMM

- 约束之间耦合很强
- 需要同时满足多个结构条件
- 近端不能闭式分解，但拆变量后能分别求解

### 什么时候看原始-对偶方法

- 存在线性算子复合，如总变差、稀疏变换域约束
- 希望同时处理原变量和对偶变量
- 需要比简单近端梯度更自然地表达约束

### 一个判断标准

如果你写出
$$
\operatorname{prox}_{\alpha g}(v)
$$
后，发现每步都要再跑几十次内层迭代，那通常已经不是近端梯度法最舒服的适用区间。此时继续坚持 ISTA/FISTA，往往只是形式统一，未必是总成本最低。

---

## 参考资料

1. EmergentMind, *Proximal Gradient Methods: ISTA & FISTA*, 2024/01/29 更新。适合快速建立定义、更新式和收敛率框架，重点看 ISTA/FISTA 对比与常见 prox 形式。  
   链接：https://www.emergentmind.com/topics/proximal-gradient-methods-ista-fista

2. ScienceDirect, *Proximal subgradient norm minimization of ISTA and FISTA*, 2026。适合进一步阅读收敛分析与变体理论，重点看近端残差、子梯度范数与复杂度表述。  
   链接：https://www.sciencedirect.com/science/article/abs/pii/S1063520325001022

3. OpenReview, *A Proximal Operator for Inducing 2:4-Sparsity*, 2026。适合看结构化稀疏 prox 在硬件友好训练中的用法，重点看 2:4 掩码如何通过近端步骤直接产生。  
   链接：https://openreview.net/pdf?id=jFC8SS8kWU

4. Beck and Teboulle, *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*, 2009。FISTA 经典文献，适合补足加速更新的原始推导。

5. Parikh and Boyd, *Proximal Algorithms*, 2014。近端方法综述，适合系统补课 prox、投影、ADMM 与分裂方法之间的关系。
