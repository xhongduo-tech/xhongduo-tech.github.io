---
title: 一阶方程组与高阶方程的数值解法
date: 2026-08-07
---

# 从单方程到方程组：一切 ODE 都在一阶系统里重写

<div class="epigraph">
<p>世界是耦合的，方程组才是常态——而高阶方程不过是方程组的一个视图。</p>
<footer>—— ODE 数值解法的降维智慧</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§9.8 ｜ 2026-08-07</p>
</div>

## 为什么从方程组开始

前面所有方法都是对单个方程 $y'=f(t,y)$ 讲的。但现实是**方程组**：三体运动（6 个方程）、化学反应网络、电路、机械多体——全是耦合的一阶系统。**好消息：所有单步法/多步法几乎不加修改就能向量化**——把 $y$ 换成向量、$f$ 换成向量值函数即可。而**高阶方程**（$y^{(n)}=g(t,y,y',\dots,y^{(n-1)})$）只需一个「换元术」——引入中间变量化成等价的一阶系统。这节把「降维术」与「向量化」讲透。<span class="marginnote">「高阶 → 一阶系统」的换元术：<strong>设 $y_1=y,\ y_2=y',\ \dots,\ y_n=y^{(n-1)}$，则原高阶方程变成 $n$ 个一阶方程</strong>。这是 ODE 数值解法的「统一化」——求解器只需要一个接口（一阶系统），通吃一切 ODE。</span>

本节给出高阶化一阶的换元术、向量化 RK，并讨论方程组的刚性。

## 1 高阶方程降维：换元术

对 $n$ 阶方程

$$
y^{(n)} = g\left(t,\ y,\ y',\ \dots,\ y^{(n-1)}\right)
$$

设 $y_1=y,\ y_2=y',\ \dots,\ y_n=y^{(n-1)}$，则：

$$
\begin{cases}
y_1' = y_2 \\
y_2' = y_3 \\
\vdots \\
y_{n-1}' = y_n \\
y_n' = g(t, y_1, y_2, \dots, y_n)
\end{cases}
$$

**一个 $n$ 阶方程 ⇔ 一个 $n$ 维一阶系统**。初值 $y(t_0),y'(t_0),\dots,y^{(n-1)}(t_0)$ 映射为 $\mathbf{y}(t_0)=(y_1(t_0),\dots,y_n(t_0))^\top$。

**示例**：弹簧-质量-阻尼 $my''+cy'+ky=0$，设 $y_1=y,\ y_2=y'$：

$$
\begin{pmatrix}y_1\\y_2\end{pmatrix}' = \begin{pmatrix}y_2\\ -\frac{c}{m}y_2-\frac{k}{m}y_1\end{pmatrix}
$$

初值 $(y(0),y'(0))$。**两行一阶方程完美封装二阶物理**。<span class="marginnote">这个换元术的哲学：<strong>「把高阶的『记忆』变成状态的『维度』」</strong>——$n$ 阶系统需要记住前 $n-1$ 阶导，于是用 $n$ 维状态向量承载。这是控制理论里「状态空间」表示的精髓，也预告了现代深度学习里「ODE 作为 ResNet 的连续化」。</span>

## 2 向量化：RK4 不加修改

所有单步法对一阶系统**逐字照搬**，只需把 $y,f$ 换成向量。**向量 RK4**：

$$
\mathbf{k}_1 = \mathbf{f}(t_k,\mathbf{y}_k)
$$

$$
\mathbf{k}_2 = \mathbf{f}\left(t_k+\frac{h}{2},\ \mathbf{y}_k+\frac{h}{2}\mathbf{k}_1\right)
$$

$$
\mathbf{k}_3 = \mathbf{f}\left(t_k+\frac{h}{2},\ \mathbf{y}_k+\frac{h}{2}\mathbf{k}_2\right)
$$

$$
\mathbf{k}_4 = \mathbf{f}\left(t_k+h,\ \mathbf{y}_k+h\mathbf{k}_3\right)
$$

$$
\mathbf{y}_{k+1} = \mathbf{y}_k + \frac{h}{6}\left(\mathbf{k}_1+2\mathbf{k}_2+2\mathbf{k}_3+\mathbf{k}_4\right)
$$

**唯一区别：$\mathbf{k}_i$ 是向量**。Python 里用 numpy 数组，代码几乎不变。<span class="marginnote">向量化的意义：<strong>「方法不关心方程个数——只关心向量维数」</strong>。写一次 RK4，从单摆（2 维）到太阳系（N 体，3N 维）通吃。这是数值软件「一次编写、处处求解」的根基。</span>

**数值例子（简谐振子）**：$y''=-y,\ y(0)=0,\ y'(0)=1$（解 $\sin t$）。化系统后 RK4 步长 0.1，到 $t=1$ 误差约 $10^{-8}$（四阶）——与单方程精度一致。

## 3 实现

```python
import numpy as np

def rk4_system(f, t0, y0, h, n):
    """f: (t, y_vec) -> vec；y0 是 numpy 向量"""
    t, y = t0, np.array(y0, float)
    ys = [y.copy()]
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h
        ys.append(y.copy())
    return np.array(ys)

# 洛伦兹吸引子（3 维系统）
def lorenz(t, y):
    x, yy, z = y
    return np.array([10*(yy-x), x*(28-z)-yy, x*yy - 8/3*z])

traj = rk4_system(lorenz, 0, [1,1,1], 0.01, 1000)
```

**工程注意**：向量维数影响**雅可比/稳定域**——系统越大、特征值分布越宽，刚性越常见。`solve_ivp` 接受向量初值即自动向量化。

## 4 方程组的刚性：更多特征值，更多麻烦

系统的「刚性」由**雅可比矩阵的特征值分布**决定：

$$
\mathbf{J}=\frac{\partial\mathbf{f}}{\partial\mathbf{y}}, \qquad \text{刚性比}=\frac{\max|\mathrm{Re}\,\lambda|}{\min|\mathrm{Re}\,\lambda|}
$$

**刚性比巨大**（如 $10^6$）⇒ 显式方法步长受「最快特征值」限制 ⇒ 步数爆炸。多体问题（质量悬殊）、化学反应（快慢组分共存）都高刚性。<span class="marginnote">预告：<strong>「系统的特征值谱越宽，刚性越强，显式方法越绝望」</strong>——这就是下一节硬性问题的定义性背景。工程判断：对系统先估雅可比谱（格什戈林圆盘！），决定显式还是隐式。</span>

**多体问题的典型情况**：近距遭遇（特征值瞬间巨大）时，自适应显式方法步长骤减——这是天体模拟「卡顿」的数值根源，隐式或专用辛积分是解药。

## 5 辨析：降维术的代价与边界

**辨析｜易错点：** 高阶化一阶系统**增加了状态维度**，但不改变方法的阶与稳定性——**「降维」是形式变换，不是精度损失**。但要注意：

- 初始条件必须**完整**给定（$n$ 阶方程要 $n$ 个初值）。
- 系统维度大时，**每步成本随维度线性增长**（RK 每步 $s\times d$ 次标量求值）。
- 高阶方程的直接多步法（如 Nyström 法）可省维度，但现代库默认走一阶系统路线（更通用）。

**工程结论：一律化一阶系统**——简单、通用、库函数友好；除非极端大规模且需要专用结构。

## 6 小结

- **降维术**：$y_1=y,\dots,y_n=y^{(n-1)}$ 把 $n$ 阶方程化成 $n$ 维一阶系统——求解器只需一个接口。
- **向量化**：RK4 等方法对向量方程逐字照搬（$\mathbf{k}_i$ 变向量），numpy 一行向量化。
- 初值完整对应：$n$ 阶方程需要 $n$ 个初始条件。
- **刚性比** $\dfrac{\max|\mathrm{Re}\,\lambda|}{\min|\mathrm{Re}\,\lambda|}$：系统特征值谱越宽越刚性，显式方法越吃力。
- 工程默认：一律化一阶系统 + 通用求解器；大规模刚性系统用隐式。

在下一节，我们直面数值 ODE 的「最难点」：**刚性问题与绝对稳定域**——为什么显式方法在刚性问题上崩溃，隐式方法如何成为唯一解。
