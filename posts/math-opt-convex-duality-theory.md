## 核心结论

凸优化的对偶理论，是把原问题的约束通过拉格朗日函数合并到目标里，构造一个“总是不会高估原问题最优值”的下界问题，再从这些下界里找最高的一个。

原问题像“在所有可行方案里找最便宜的”，对偶问题像“所有约束一起出价，给出一个不会低估真实最优值的最高报价”。这里的“报价”不是随便猜，而是由拉格朗日乘子控制的数学下界。

核心链路是：

```text
原问题 -> 拉格朗日函数 -> 对偶函数 -> 对偶问题 -> KKT
```

核心公式是：

$$
p^\star \ge d^\star,\quad \text{在 Slater 条件下常有 } p^\star=d^\star
$$

其中 $p^\star$ 是原问题最优值，$d^\star$ 是对偶问题最优值。$p^\star-d^\star$ 叫对偶间隙，用来衡量对偶下界距离真实最优值还有多远。

一个最小玩具例子：

$$
\min_x x^2\quad \text{s.t. } x\ge 1
$$

原问题的最优解是 $x^\star=1$，最优值是 $1$。把约束改写成 $1-x\le 0$ 后，可以推出对偶问题的最优值也是 $1$。这说明在这个例子里强对偶成立，也就是 $p^\star=d^\star=1$。

---

## 问题定义与边界

标准凸优化问题常写成：

$$
\min_x f_0(x)\quad
\text{s.t. } f_i(x)\le 0,\ i=1,\dots,m;\quad h_j(x)=0,\ j=1,\dots,p
$$

这里 $x$ 是决策变量，也就是要被优化算法选择的量。$f_0(x)$ 是目标函数，表示要最小化的成本、误差、风险或能耗。$f_i(x)\le 0$ 是不等式约束，表示“不能违反的限制”。$h_j(x)=0$ 是等式约束，表示“必须精确满足的条件”。

凸优化对这些元素有明确要求：

| 元素 | 要求 | 作用 |
|---|---|---|
| `f_0(x)` | 凸 | 目标函数 |
| `f_i(x)` | 凸 | 不等式约束 |
| `h_j(x)` | 仿射 | 等式约束 |

凸函数的白话解释是：函数图像上任意两点连线不会低于函数图像本身。仿射函数的白话解释是：形如 $Ax+b$ 的线性变换加常数项。

这些要求不是形式主义。若目标函数和不等式约束都是凸的，可行域和目标形状就更稳定；若等式约束是任意非线性的，常常会破坏凸性。例如 $x^2=1$ 的可行点是 $x=1$ 和 $x=-1$，两个点的中点 $0$ 不可行，因此不是凸集合。

适用边界需要说清楚：对偶理论最稳定地适用于凸问题。非凸问题里也可以写拉格朗日对偶，但通常只能得到下界，不能直接保证全局最优。并且，`凸` 不等于 `自动强对偶`，还要看约束资格条件，例如 Slater 条件。

Slater 条件的白话解释是：不仅存在可行点，而且存在一个严格满足所有不等式约束的点。对于凸问题，若存在 $x$ 使得 $f_i(x)<0$ 且 $h_j(x)=0$，很多常见情形下可以保证强对偶。

---

## 核心机制与推导

拉格朗日函数把目标函数和约束合并：

$$
L(x,\lambda,\nu)=f_0(x)+\sum_{i=1}^m \lambda_i f_i(x)+\sum_{j=1}^p \nu_j h_j(x),\quad \lambda\ge 0
$$

$\lambda_i$ 是不等式约束的拉格朗日乘子，白话解释是“第 $i$ 个限制的价格”。因为不等式约束是 $f_i(x)\le 0$，所以要求 $\lambda_i\ge 0$。$\nu_j$ 是等式约束的乘子，可以是正数、负数或零，因为等式约束没有“只允许单侧违反”的方向。

对偶函数定义为：

$$
g(\lambda,\nu)=\inf_x L(x,\lambda,\nu)
$$

$\inf$ 是下确界，白话解释是“所有可能值里的最大下界”；如果最小值能取到，它就等于最小值。

对偶问题是：

$$
\max_{\lambda\ge 0,\nu} g(\lambda,\nu)
$$

推导步骤可以写成：

1. 写出 $L(x,\lambda,\nu)$
2. 对 $x$ 取 $\inf$
3. 得到 $g(\lambda,\nu)$
4. 最大化 $g$

为什么 $g$ 是下界？对任何原问题可行的 $x$，都有 $f_i(x)\le 0$，且 $\lambda_i\ge 0$，所以 $\lambda_i f_i(x)\le 0$；等式约束满足 $h_j(x)=0$，所以等式项为 $0$。因此：

$$
L(x,\lambda,\nu)\le f_0(x)
$$

又因为 $g(\lambda,\nu)=\inf_x L(x,\lambda,\nu)$，所以：

$$
g(\lambda,\nu)\le L(x,\lambda,\nu)\le f_0(x)
$$

这对所有可行 $x$ 都成立，所以对最优可行解也成立：

$$
g(\lambda,\nu)\le p^\star
$$

再对所有对偶可行的 $\lambda,\nu$ 最大化，仍然不会超过 $p^\star$：

$$
d^\star \le p^\star
$$

这就是弱对偶。弱对偶不需要凸性，几乎总成立。强对偶更强，要求 $d^\star=p^\star$。在凸优化里，Slater 条件是常用的强对偶保证。

回到玩具例子：

$$
\min_x x^2 \quad \text{s.t. } x\ge 1
$$

改写为：

$$
1-x\le 0
$$

拉格朗日函数为：

$$
L(x,\lambda)=x^2+\lambda(1-x),\quad \lambda\ge 0
$$

对 $x$ 求极小值：

$$
\frac{\partial L}{\partial x}=2x-\lambda=0 \Rightarrow x=\frac{\lambda}{2}
$$

代回：

$$
g(\lambda)=\lambda-\frac{\lambda^2}{4},\quad \lambda\ge 0
$$

最大化 $g$：

$$
g'(\lambda)=1-\frac{\lambda}{2}=0 \Rightarrow \lambda^\star=2
$$

所以：

$$
d^\star=g(2)=1
$$

原问题也有 $p^\star=1$，因此对偶间隙为：

$$
p^\star-d^\star=0
$$

KKT 条件是最优性条件，白话解释是：如果一个点同时满足原问题可行、对偶可行、互补松弛和梯度平衡，那么它就是最优点；在满足强对偶的凸问题中，它通常也是充分必要条件。

可微情形下，KKT 条件写成：

$$
\begin{cases}
f_i(x^\star)\le 0,\ h_j(x^\star)=0 \\
\lambda_i^\star\ge 0 \\
\lambda_i^\star f_i(x^\star)=0 \\
\nabla f_0(x^\star)+\sum_i \lambda_i^\star \nabla f_i(x^\star)+\sum_j \nu_j^\star \nabla h_j(x^\star)=0
\end{cases}
$$

互补松弛 $\lambda_i^\star f_i(x^\star)=0$ 的白话解释是：一个约束如果没有卡住最优解，它的价格就是零；一个约束如果有正价格，它必须正好卡在边界上。

---

## 代码实现

下面用纯 Python 验证玩具例子。代码的目标不是用数值优化器求最优，而是验证手动推导出的原问题值、对偶值和 KKT 条件一致。

```python
# 原问题：
#   minimize x^2
#   subject to x >= 1
#
# 标准形式：
#   1 - x <= 0
#
# 拉格朗日函数：
#   L(x, lambda) = x^2 + lambda * (1 - x), lambda >= 0
#
# 对 x 求 inf：
#   dL/dx = 2x - lambda = 0
#   x = lambda / 2
#
# 对偶函数：
#   g(lambda) = lambda - lambda^2 / 4
#
# 对偶问题：
#   maximize g(lambda), lambda >= 0
#   g'(lambda) = 1 - lambda / 2 = 0
#   lambda* = 2

def primal_objective(x):
    return x * x

def constraint_value(x):
    # 标准约束 1 - x <= 0
    return 1 - x

def dual_function(lam):
    assert lam >= 0
    return lam - lam * lam / 4

x_star = 1.0
lambda_star = 2.0

p_star = primal_objective(x_star)
d_star = dual_function(lambda_star)

# 原问题可行性
assert constraint_value(x_star) <= 0

# 对偶可行性
assert lambda_star >= 0

# 互补松弛
assert abs(lambda_star * constraint_value(x_star)) < 1e-12

# 梯度平衡：d/dx [x^2 + lambda(1-x)] = 2x - lambda
assert abs(2 * x_star - lambda_star) < 1e-12

# 强对偶：p* = d*
assert abs(p_star - d_star) < 1e-12
assert p_star == 1.0
```

如果使用 `cvxpy`，代码框架可以更接近真实工程写法：

```python
import cvxpy as cp

x = cp.Variable()
objective = cp.Minimize(cp.square(x))
constraints = [x >= 1]

prob = cp.Problem(objective, constraints)
prob.solve()

assert abs(x.value - 1.0) < 1e-6
assert abs(prob.value - 1.0) < 1e-6
```

真实工程例子可以看无线通信里的功率分配。系统要在多个用户之间分配发射功率，目标可能是最小化总功率，同时满足每个用户的 SINR 或 QoS 约束。SINR 是信号与干扰加噪声比，白话解释是“接收信号相对于干扰和噪声的清晰程度”。QoS 是服务质量，白话解释是“业务必须达到的最低体验要求”。

这类问题里的对偶变量可以解释为每个约束的边际代价：某个用户的 QoS 约束越难满足，它对应的对偶变量往往越大。工程上可以用这些变量做资源定价、分布式求解和系统瓶颈分析。类似思想也出现在网络流量整形、LASSO 稀疏回归、支持向量机和水位填充算法中。

---

## 工程权衡与常见坑

工程上最重要的问题不是“能不能写出对偶公式”，而是“什么时候能信对偶结果”。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 以为凸问题一定强对偶 | 误判结果 | 检查 `Slater 条件` |
| 只看 `λ>=0` | 忽略 `g` 可能无定义 | 同时检查 `dom g` |
| 把 KKT 当非凸充分条件 | 得到假结论 | 非凸只当必要条件 |
| 等式约束写成任意非线性形式 | 破坏标准形式 | 等式约束用仿射 |
| 只看 `p^\star=d^\star` | 忽略可达性 | 还要检查 primal/dual 是否达到最优 |

第一个坑是把“凸”和“强对偶”混为一谈。凸性让问题更稳定，但强对偶还依赖约束资格条件。Slater 条件就是最常用的一类检查方式。

第二个坑是只检查 $\lambda\ge 0$。在例子 $L(x,\lambda)=x^2+\lambda(1-x)$ 中，$\lambda\ge 0$ 只是对偶可行的一部分。还要看：

$$
g(\lambda)=\inf_x L(x,\lambda)
$$

是否是有限值。新手版解释是：不是任何乘子都能用，错误的乘子会让 $\inf_x L(x,\lambda,\nu)$ 变成 $-\infty$。一旦 $g=-\infty$，这个乘子虽然形式上满足某些符号约束，但没有提供有意义的下界。

第三个坑是误用 KKT。对凸问题，在合适条件下，KKT 可以成为充分必要条件；对非凸问题，KKT 通常只是局部最优的必要条件，不能保证全局最优。

第四个坑是忽略可达性。可达性指最优值是否真的由某个点取到。可能出现原问题或对偶问题的最优值存在，但没有具体解达到它。工程实现里，如果只比较数值，不检查解是否可行、是否达到，可能会误判算法结果。

---

## 替代方案与适用边界

对偶理论不是唯一工具。它适合解释约束价格、构造下界、做问题分解，但不一定是所有优化任务的最快路径。

| 方法 | 适合场景 | 优点 | 局限 |
|---|---|---|---|
| 原始问题直接求解 | 规模小、结构简单 | 直观 | 不利于分解 |
| 对偶分解 | 大规模、可分布式 | 可解释、可拆分 | 依赖强对偶或良好松弛 |
| ADMM | 结构化约束问题 | 实用性强 | 收敛与参数敏感 |
| 非凸优化 | 复杂真实模型 | 表达能力强 | 无强对偶保障 |

小规模凸问题，直接交给成熟求解器通常更简单。例如几十个变量的二次规划，没有必要先手工推导复杂对偶形式。

大规模可分解问题，对偶方法更有价值。例如多个机器、多个地区、多个用户共享同一类资源时，原问题可能耦合在一个总约束上。通过对偶变量把共享约束“定价”，各子问题可以分开求解，再由乘子协调整体资源。

ADMM 是交替方向乘子法，白话解释是：把复杂问题拆成几个子问题交替优化，同时用乘子约束它们的一致性。它经常用于分布式优化、信号处理和机器学习模型训练。

非凸问题里，对偶仍然有价值，但角色通常变成松弛或下界分析。松弛的白话解释是：把原本难满足的限制放宽，先解一个更容易的问题。此时对偶结果可以告诉我们“最优值不可能低于多少”，但不能保证直接得到全局最优解。

如果目标只是“算出一个数”，对偶不一定最快；如果还想知道“每个约束值多少钱”，或者要把一个大问题拆给多个子系统，对偶理论就很有用。

---

## 参考资料

| 顺序 | 资料 | 目的 |
|---|---|---|
| 1 | 讲义 | 建立直觉 |
| 2 | 教材 | 补全理论 |
| 3 | 课程页 | 串联章节 |
| 4 | 论文 | 扩展边界 |

1. Stanford 讲义《Duality》PDF：<https://see.stanford.edu/materials/lsocoee364a/05Duality.pdf>
2. Boyd & Vandenberghe, *Convex Optimization*, Chapter 5 Duality：<https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>
3. Stanford EE364A 课程页：<https://see.stanford.edu/Course/EE364A>
4. Jeyakumar, V., Lee, G. M., & Dinh, N. “On strong and total Lagrange duality for convex optimization problems.” *Journal of Mathematical Analysis and Applications*, 2008：<https://doi.org/10.1016/j.jmaa.2007.04.071>
