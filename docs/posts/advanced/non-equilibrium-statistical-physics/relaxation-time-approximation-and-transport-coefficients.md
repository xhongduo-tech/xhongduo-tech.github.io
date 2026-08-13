---
title: 弛豫时间近似与输运系数
date: 2026-08-07
---

# 弛豫时间近似与输运系数

<div class="epigraph">
<p>一切理论的最高目标，是让不可约的基本要素尽可能简单、尽可能少，而不必放弃对任何一个经验事实的恰当表示。</p>
<footer>—— 阿尔伯特 · 爱因斯坦（Albert Einstein，1933）</footer>
</div>

<div class="article-byline">
<p>第四级 · 非平衡统计物理 ｜ 陈式刚《非平衡统计力学》第8章 ｜ 2026-08-07</p>
</div>

## 为什么从弛豫时间近似开始

玻尔兹曼方程的碰撞积分是五重积分，精确求解几乎不可能。但为了算出粘滞系数、热导率这些实验可测的量，我们需要一种实用的近似。

**弛豫时间近似（relaxation time approximation，RTA）**是动理学理论最常用、也最优雅的简化：它把碰撞项这一复杂积分换成「分布以单一时间尺度 $\tau$ 指数弛豫到局域平衡」的简单形式。代价是精度有限，收获是几乎一切输运系数都能闭式算出，且与实验量级吻合。它是从「方程」到「数字」之间最快的一座桥。

## 1 弛豫时间近似

记 $f^0$ 为局域平衡（麦克斯韦）分布。RTA 假设碰撞项正比于分布与局域平衡的偏差：

$$
\boxed{\left(\frac{\partial f}{\partial t}\right)_{coll} = -\frac{f - f^0}{\tau}}
$$

$\tau$ 是**弛豫时间**，量纲为时间，物理上约等于平均碰撞间隔。这个近似的直觉：碰撞总是把分布「拉回」局域平衡，拉回的速度与偏离程度成正比，就像摩擦把速度拉回零。<span class="marginnote">RTA 的精确版是 BGK 模型（Bhatnagar-Gross-Krook，下一讲），它用「局域平衡分布 + 单一弛豫时间」替换碰撞项，并调整 $f^0$ 的参量使守恒律自动成立。RTA 保留了玻尔兹曼方程的一切结构优点，只是把碰撞的细节压缩成一个参数 $\tau$。</span>

把 RTA 代入玻尔兹曼方程：

$$
\frac{\partial f}{\partial t} + \mathbf{v}\cdot\nabla_{\mathbf{q}}f + \mathbf{F}\cdot\nabla_{\mathbf{p}}f = -\frac{f - f^0}{\tau}
$$

**辨析｜易错点：** RTA 假设所有模式以**同一**时间尺度弛豫，这在真实气体中不严格——高角动量碰撞模式弛豫更快。对粘滞与热导这种「低阶矩输运」，单 $\tau$ 够用；但精确的气体动理学（查普曼-恩斯科格展开）会给出不同矩的不同弛豫率。RTA 的误差典型在百分之几十，但**结构与量级完全正确**——这正是它作为教学工具与工程工具的价值。

## 2 稳态解：偏离平衡的一阶修正

在定态、缓变的输运问题里，设分布写成局域平衡加小修正：

$$
f = f^0 + \delta f, \qquad |\delta f| \ll f^0
$$

在 RTA 下，稳态玻尔兹曼方程给出：

$$
\delta f = -\tau\left(\mathbf{v}\cdot\nabla_{\mathbf{q}}f^0 + \mathbf{F}\cdot\nabla_{\mathbf{p}}f^0\right)
$$

这正是「偏离平衡 = 弛豫时间 × 自由流动的驱动力」。$f^0$ 依赖局域密度 $n(\mathbf{q})$、流速 $\mathbf{u}(\mathbf{q})$、温度 $T(\mathbf{q})$，它们随位置的梯度就是输运的引擎。把这个 $\delta f$ 代回宏观量（应力张量、热流）的定义，就能读出输运系数。

## 3 公式解析：粘滞系数

取均匀密度、但流速有梯度的情形（剪切流动）。设流速沿 $x$ 方向、梯度沿 $y$ 方向：$\mathbf{u} = u_x(y)\hat{\mathbf{x}}$。局域平衡为 $f^0(n,T,\mathbf{u}(y))$，代入 $\delta f$ 公式，得分布偏离：

$$
\delta f = -\tau\, v_y\,\frac{\partial u_x}{\partial y}\,\frac{\partial f^0}{\partial u_x}
$$

- **$\tau\, v_y$**：弛豫时间乘垂直速度分量——粒子在两次碰撞间「携带着」速度信息跑多远。$v_y$ 越大，粒子越能跨越梯度携带动量。
- **$\partial u_x/\partial y$**：速度梯度——输运的驱动力。没有梯度，分布不偏离平衡。
- **$\partial f^0/\partial u_x$**：局域平衡对宏观流速的敏感度，它把「分布修正」与「流速」联系起来。

粘滞应力张量 $P_{xy} = \int m\,v_x v_y f\,d\mathbf{v}$ 只含 $\delta f$ 的贡献（$f^0$ 各向同性，积分为零）：

$$
P_{xy} = -\tau\int m\,v_x v_y\left(v_y\frac{\partial u_x}{\partial y}\frac{\partial f^0}{\partial u_x}\right)d\mathbf{v}
$$

完成对麦克斯韦分布的各向同性积分，得到 $P_{xy} = -\eta\,\partial u_x/\partial y$，其中：

$$
\boxed{\eta = n m\, \tau\,\langle v_x^2\rangle = n k_BT\,\tau}
$$

这就是 RTA 给出的**剪切粘滞系数**。它与实验量级一致，且揭示了输运系数的通用结构：**输运系数 ≈（粒子密度）×（能量尺度）×（弛豫时间）**。

## 4 热导率与扩散系数

同样的套路可以算出其它输运系数。对温度梯度驱动的热流：

$$
\kappa = n k_B\, \tau\,\frac{5}{2}\frac{k_BT}{m} = \frac{5}{2} n k_B D
$$

其中 $D$ 是自扩散系数。对粒子密度梯度：

$$
D = \tau\,\frac{k_BT}{m} = \frac{1}{3}\bar v\,\lambda
$$

最后一步用了关系 $\tau = \lambda/\bar v$（弛豫时间 = 平均自由程 ÷ 平均热速度）。由此得到动理学理论的经典结果：

| 输运系数 | 表达式 | 平均自由程形式 |
| --- | --- | --- |
| 扩散系数 $D$ | $k_BT\tau/m$ | $\frac{1}{3}\bar v\lambda$ |
| 粘滞系数 $\eta$ | $nk_BT\tau$ | $\frac{1}{3}nm\bar v\lambda$ |
| 热导率 $\kappa$ | $\frac{5}{2}nk_BD$ | $\frac{1}{3}n\bar v\lambda c_v$ |

三个系数共享同一个骨架：**$\frac{1}{3} \times$（携带者密度）×（平均速度）×（平均自由程）×（每单位携带量）**。$\lambda$ 越大（气体越稀、碰撞越少），输运越有效——直觉上，自由程长的分子能「把信息带得更远」才散播出去。<span class="marginnote">一个惊人的预言：理想气体粘滞系数 $\eta \propto n\bar v\lambda$，而 $n\lambda \sim 1/\sigma$（截面）与密度无关——所以<strong>理想气体的粘滞系数与压强无关</strong>！麦克斯韦 1860 年预言这一反常现象，多年后由实验证实。这是动理学理论最早、最漂亮的定量胜利。</span>

## 5 输运系数的普适逻辑

弛豫时间近似的意义远超「算几个系数」。它揭示了输运的**通用机制**：

1. **驱动力**：密度/流速/温度梯度打破局域平衡；
2. **偏离**：分布函数产生 $\delta f \sim -\tau\times$（自由流动项）；
3. **通量**：$\delta f$ 的三阶矩给出宏观流；
4. **系数**：流 ∝ 梯度，比例系数就是输运系数，正比于 $\tau$。

这套逻辑在等离子体、半导体、中子输运、辐射转移里反复重现——**只要把「碰撞」换成「散射」「复合」或「吸收」，把 $\tau$ 换成对应的弛豫率，同一个公式骨架就适用于完全不同的物理。**<span class="marginnote">从 Green-Kubo 的视角（第4篇），输运系数其实是「微观流自关联函数的积分」，而 $\tau$ 正是关联函数的衰减时间。RTA 相当于假设关联函数单指数衰减 $C(\tau) = C(0)e^{-t/\tau}$，积分得 $\tau\times$（涨落强度）。两种方法殊途同归——RTA 是唯象捷径，Green-Kubo 是严格公式。</span>

## 6 小结

- **弛豫时间近似**把碰撞项替换为 $-(f-f^0)/\tau$，用一个参数 $\tau$ 抓住碰撞的全部效应。
- 稳态一阶解 $\delta f = -\tau(\mathbf{v}\cdot\nabla f^0 + \mathbf{F}\cdot\nabla_p f^0)$ 是输运的统一出发点。
- RTA 算出**剪切粘滞** $\eta = nk_BT\tau$、**热导率** $\kappa = \frac52 nk_B D$、**扩散系数** $D = \frac13\bar v\lambda$。
- 三系数共享骨架 $\frac13 \times$ 密度 $\times$ 平均速度 $\times$ 平均自由程——这是输运的普适结构。
- 理想气体粘滞系数与压强无关是动理学理论的著名预言，后经实验证实。

在下一节，我们把弛豫时间近似升级为更系统的工具：**BGK 模型与矩方法**。BGK 保留了 RTA 的简洁，却通过「矩守恒」约束 $f^0$