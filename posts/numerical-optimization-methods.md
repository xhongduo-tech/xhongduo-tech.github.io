## 核心结论

线搜索和信赖域，解决的是同一个问题：**方向已经有了，步子到底该迈多大**。区别在于控制变量不同。

线搜索先选方向 $p_k$，再找步长 $\alpha_k$，更新写成
$$
x_{k+1}=x_k+\alpha_k p_k.
$$
它的核心目标是避免“步子太大导致反弹”或“步子太小导致收敛很慢”。最常见的判据是 Armijo 条件和 Wolfe 条件。

信赖域先承认一个事实：**当前点附近的局部模型只在有限范围内可信**。于是它不直接问“步长是多少”，而是问“在半径 $\Delta_k$ 的球内，哪个步子最值得走”。典型子问题是
$$
\min_{\|p\|\le \Delta_k} m_k(p),\qquad
m_k(p)=f(x_k)+g_k^Tp+\frac12 p^TB_kp.
$$
这里的二次模型，就是“用梯度和曲率近似原函数的局部抛物面”。

两类方法都能给出全局收敛结论，但适用环境不同：

| 方法 | 先决定什么 | 主要控制量 | 典型接受规则 | 更适合的场景 |
|---|---|---|---|---|
| 线搜索 | 先定方向 | 步长 $\alpha_k$ | Armijo / Wolfe | 下降方向可靠、函数较平滑 |
| 信赖域 | 先定局部区域 | 半径 $\Delta_k$ | 比值 $\rho_k$ | 曲率不稳、Hessian 可能不定 |
| 线搜索 + BFGS | 方向由拟牛顿给出 | 步长与曲率条件联动 | Armijo-Wolfe | 中高维无约束光滑优化 |
| 信赖域 + LM | 用阻尼控制曲率 | 半径或阻尼参数 | 预测/实际降幅比 | 非线性最小二乘、参数估计 |

对初学者，最重要的不是记住所有变体，而是记住两个结论：

1. 线搜索的核心是“**固定方向，筛选一个足够安全的步长**”。
2. 信赖域的核心是“**固定一个可信范围，在范围里解一个局部优化问题**”。

如果只记一句话，可以记成：

| 你已经知道什么 | 还不知道什么 | 典型方法 |
|---|---|---|
| 方向大致是对的 | 沿方向走多远 | 线搜索 |
| 局部模型大致是对的 | 在模型可信范围内怎么走 | 信赖域 |

---

## 问题定义与边界

我们讨论的是**无约束最小化**：
$$
\min_{x\in \mathbb{R}^n} f(x).
$$

这里“无约束”就是没有额外的等式或不等式条件，变量可以在整个 $\mathbb{R}^n$ 上取值。优化过程通常分成两步：

1. 选择方向 $p_k$
2. 决定沿这个方向走多远

方向可能来自：

- 负梯度方向 $p_k=-\nabla f(x_k)$
- 牛顿方向 $p_k=-[\nabla^2 f(x_k)]^{-1}\nabla f(x_k)$
- 拟牛顿方向，如 BFGS 给出的方向
- 共轭梯度或截断共轭梯度给出的近似二阶方向

“下降方向”指的是沿该方向走一个足够小的正步长，目标函数会下降。数学上是
$$
\nabla f(x_k)^T p_k < 0.
$$

这个条件为什么重要，可以直接从一阶展开看出来。对足够小的 $\alpha>0$，
$$
f(x_k+\alpha p_k)
= f(x_k) + \alpha \nabla f(x_k)^T p_k + O(\alpha^2).
$$
如果 $\nabla f(x_k)^T p_k<0$，那么一阶项先把函数值往下拉；如果这个内积不小于零，再好的线搜索也不可能凭空把错误方向修成下降方向。

线搜索的边界很明确：**它默认方向已经可信**。如果你给它的方向本身不是下降方向，Armijo 或 Wolfe 再精细也救不了。

信赖域的边界不同：它不要求先拿到一个完全可靠的步长，而是要求你能构造一个局部模型 $m_k(p)$。如果 Hessian 不稳定，或者牛顿步过大，信赖域会通过约束
$$
\|p\|\le \Delta_k
$$
把步子压回模型仍然可信的范围内。

这两类方法与“固定学习率”也不是一回事。固定学习率直接预设一个数 $\eta$，更新写成
$$
x_{k+1}=x_k-\eta \nabla f(x_k),
$$
它不检查这一步是否真的足够好；线搜索和信赖域则会在每一步显式验证“这个步子为什么可以走”。

一个玩具例子足够说明差异。设
$$
f(x)=(x-2)^2.
$$
在 $x_0=0$ 处，

- 梯度 $g_0=f'(0)=-4$
- Hessian $H_0=2$

若取负梯度方向，则
$$
p_0=-g_0=4.
$$

此时：

- 线搜索要解决的是：沿着 $p_0=4$，选哪个 $\alpha$ 合适
- 信赖域要解决的是：在 $|p|\le \Delta$ 内，二次模型最优的 $p$ 是多少

这个边界非常关键。很多初学者把两者都理解成“调学习率”，这是不准确的。线搜索直接调步长；信赖域本质上调的是**模型可信范围**。

为避免混淆，可以把它们放到同一张表里看：

| 问题 | 线搜索怎么回答 | 信赖域怎么回答 |
|---|---|---|
| “方向往哪走？” | 通常由外部方法先给出 | 通常由局部模型内部决定 |
| “步子多大？” | 通过 $\alpha_k$ 直接控制 | 通过 $\Delta_k$ 间接约束 |
| “为什么接受这一步？” | 满足下降/曲率条件 | 预测降幅和实际降幅一致 |
| “如果模型不准怎么办？” | 缩小步长 | 缩小信赖域 |

---

## 核心机制与推导

### 1. 线搜索：Armijo 与 Wolfe

Armijo 条件又叫“充分下降条件”，意思是下降不能只是“比原来小一点”，而是要小到有意义。公式是
$$
f(x_k+\alpha p_k)\le f(x_k)+c_1\alpha \nabla f(x_k)^T p_k,
$$
其中 $0<c_1<1$，常用 $c_1=10^{-4}$。

因为 $\nabla f(x_k)^T p_k<0$，右边其实比 $f(x_k)$ 更小，所以它要求这一步必须产生足够的函数值下降。

只看 Armijo，会出现一个问题：把 $\alpha$ 压得极小几乎总能通过，因为当 $\alpha\to 0$ 时，左边和右边都会逼近 $f(x_k)$。这会导致“每步都安全，但整体很慢”。

于是再加 Wolfe 条件来约束曲率：
$$
\nabla f(x_k+\alpha p_k)^T p_k \ge c_2 \nabla f(x_k)^T p_k,
$$
其中 $c_1<c_2<1$，常取 $c_2=0.9$。它的直观含义是：走完这一步后，沿当前方向的下降趋势不能还像起点那样陡，否则说明步长可能过短。

如果使用强 Wolfe 条件，则写成
$$
\left|\nabla f(x_k+\alpha p_k)^T p_k\right|
\le c_2 \left|\nabla f(x_k)^T p_k\right|.
$$
它比普通 Wolfe 更常用于拟牛顿法，因为它对新点处的方向导数控制更强，更利于保持 Hessian 近似的数值稳定性。

三条量放在一起看更直观：

| 条件 | 数学对象 | 解决的问题 |
|---|---|---|
| 下降方向 | $\nabla f(x_k)^T p_k < 0$ | 这个方向是否值得尝试 |
| Armijo | 函数值 $f$ | 降得够不够多 |
| Wolfe / 强 Wolfe | 方向导数 $\nabla f^Tp$ | 步长是不是过短 |

一个常见实现是回溯线搜索。先试 $\alpha=1$，不满足就缩成 $\beta\alpha$，其中 $\beta\in(0,1)$，常用 $0.5$。它通常只检查 Armijo，因为这样实现最简单、最稳。

回溯线搜索的伪代码可以写成：

$$
\alpha \leftarrow 1,\qquad
\text{while Armijo 不成立: }\alpha \leftarrow \beta \alpha.
$$

这背后的逻辑不是“找到最优步长”，而是“找到一个足够好的步长”。工程上这很重要，因为一次精确线搜索往往代价太高，近似地找到可接受步就够了。

还有一个初学者常忽略的事实：线搜索并不独立存在，它依赖方向生成器。比如：

| 方向生成器 | 典型步长策略 |
|---|---|
| 梯度下降 | Armijo 回溯 |
| 共轭梯度 | Wolfe / 强 Wolfe |
| BFGS / L-BFGS | 强 Wolfe 更常见 |
| 牛顿法 | $\alpha=1$ 起试，不行再回溯 |

### 2. 信赖域：二次模型与 $\rho_k$

信赖域先构造局部模型
$$
m_k(p)=f(x_k)+g_k^Tp+\frac12 p^TB_kp.
$$

这里：

- $g_k=\nabla f(x_k)$ 是梯度，表示一阶变化趋势
- $B_k$ 是 Hessian 或其近似，表示曲率
- 二次模型就是“在当前点附近用二次函数近似原函数”

如果 $f$ 二阶可微，那么 Taylor 展开给出
$$
f(x_k+p)
= f(x_k)+g_k^Tp+\frac12 p^T\nabla^2 f(x_k)p+O(\|p\|^3).
$$
所以当 $\|p\|$ 足够小时，二次模型通常是合理近似；当步子过大时，三阶及更高阶项不再可忽略，模型就会失真。这正是“信赖域”名字的来源。

然后解
$$
\min_{\|p\|\le \Delta_k} m_k(p).
$$

算出候选步 $p_k$ 后，不直接相信它，而是比较“模型预测的下降”和“真实函数的下降”是否接近：
$$
\rho_k=\frac{f(x_k)-f(x_k+p_k)}{m_k(0)-m_k(p_k)}.
$$

其中分母
$$
m_k(0)-m_k(p_k)= -g_k^T p_k-\frac12 p_k^TB_kp_k
$$
叫作**预测降幅**；分子
$$
f(x_k)-f(x_k+p_k)
$$
叫作**实际降幅**。

这个比值的解释很直接：

- $\rho_k\approx 1$：模型预测很准，可以信任
- $\rho_k\ll 1$：模型过于乐观，实际下降远低于预测
- $\rho_k<0$：走了这一步反而变差，应拒绝

典型更新规则是：

| $\rho_k$ 范围 | 动作 | 含义 |
|---|---|---|
| $\rho_k < 0.25$ | 缩小 $\Delta_k$ | 模型不可信，区域太大 |
| $0.25 \le \rho_k < \eta$ | 常见实现会拒绝步，半径不变或小幅缩小 | 下降不够可靠 |
| $\rho_k \ge \eta$ | 接受步 | $\eta$ 常取 0.1 或 0.2 |
| $\rho_k > 0.75$ 且 $\|p_k\|=\Delta_k$ | 放大 $\Delta_k$ | 模型可靠且边界已打满 |

上面这套规则有两个关键点：

1. **接受步和改半径是两件事**。是否更新 $x_{k+1}$，由 $\rho_k$ 是否足够大决定；是否放大或缩小半径，还要看模型到底准不准。
2. **信赖域关心的不只是“这一步降没降”**。它更关心“模型说会降多少，实际是不是差不多也降了这么多”。

在实现子问题时，常见的不是精确求解，而是近似求解。几种典型方法如下：

| 子问题求解器 | 适用情况 | 特点 |
|---|---|---|
| Cauchy point | 最简单的下降方向近似 | 便宜、稳，但可能偏保守 |
| Dogleg | 非线性最小二乘、Gauss-Newton 框架 | 结合陡降步和牛顿步 |
| 截断共轭梯度 | 中高维稀疏问题 | 不必显式求逆 Hessian |
| Levenberg-Marquardt | 残差平方和问题 | 用阻尼参数控制曲率 |

把线搜索和信赖域的控制逻辑并排看，差异会更清楚：

| 维度 | 线搜索 | 信赖域 |
|---|---|---|
| 先验假设 | 方向可信 | 局部模型可信 |
| 外层循环在调什么 | $\alpha_k$ | $\Delta_k$ |
| 典型失败模式 | 方向差，导致一直缩步 | 模型差，导致一直缩半径 |
| 成功信号 | 很快找到满足条件的 $\alpha$ | $\rho_k$ 长期接近 1 |

### 3. 玩具例子：$f(x)=(x-2)^2$

在 $x_0=0$ 处：

- $f(0)=4$
- $g_0=-4$
- 取负梯度方向 $p_0=4$

若线搜索取 $\alpha=0.1$，则新点
$$
x_1=0+0.1\times 4=0.4,
$$
函数值
$$
f(x_1)=(0.4-2)^2=2.56.
$$

Armijo 右侧为
$$
f(x_0)+c_1\alpha g_0^Tp_0=4+c_1\cdot 0.1\cdot (-16).
$$
若 $c_1=10^{-4}$，右侧是 $3.99984$，显然 $2.56\le 3.99984$，条件成立。

如果进一步试 $\alpha=1$，会得到
$$
x_1=4,\qquad f(4)=4.
$$
这一步并没有下降，因此会被 Armijo 直接拒绝。这个例子说明：即使方向是正确的，步长过大仍然会反弹。

再看信赖域。二次模型在这个例子里与原函数局部一致，因为 Hessian 恒为 2。未约束最优步满足
$$
g_0 + H_0 p = 0
\Rightarrow -4 + 2p = 0
\Rightarrow p^*=2.
$$
若当前信赖域半径 $\Delta_0=1$，则 $p^*=2$ 超出边界，只能截断成 $p=1$。此时：

- 实际下降：$f(0)-f(1)=4-1=3$
- 预测下降：$m_0(0)-m_0(1)=3$

所以
$$
\rho_0=1.
$$

这说明模型完全可信，步子应被接受，而且因为步子碰到了边界，通常还会扩大半径。

如果把半径换成 $\Delta_0=3$，则未约束最优步 $p^*=2$ 落在信赖域内部，直接接受即可。此时一步到达最优点 $x=2$，并且
$$
f(2)=0.
$$

这个一维例子很简单，但它已经把两类方法的“思考方式”完整展示出来了：

| 方法 | 决策问题 | 本例里的答案 |
|---|---|---|
| 线搜索 | 方向 $p_0=4$ 已给定，$\alpha$ 取多少 | 取小于 1 的合适步长 |
| 信赖域 | 模型已给定，在 $|p|\le\Delta$ 里取哪个 $p$ | 若 $\Delta=1$ 取 $p=1$，若 $\Delta\ge 2$ 取 $p=2$ |

---

## 代码实现

下面给出一个**可直接运行**的最小 Python 实现，分别演示 Armijo 回溯线搜索和二维信赖域。代码只依赖标准库，保存为 `demo.py` 后可直接运行。

前半段用二维二次函数演示回溯线搜索，后半段用同一个目标函数演示信赖域的 Cauchy step。这样做有两个好处：

1. 代码能直接运行并打印结果
2. 不再局限于一维，更容易看清“方向”和“半径”在多维中的含义

```python
from math import sqrt


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def scale(alpha, v):
    return [alpha * x for x in v]


def norm(v):
    return sqrt(dot(v, v))


# 目标函数: f(x, y) = (x - 2)^2 + 4 (y + 1)^2
def f(x):
    return (x[0] - 2.0) ** 2 + 4.0 * (x[1] + 1.0) ** 2


def grad(x):
    return [2.0 * (x[0] - 2.0), 8.0 * (x[1] + 1.0)]


# Hessian 是常数矩阵 [[2, 0], [0, 8]]
def hess_mul(v):
    return [2.0 * v[0], 8.0 * v[1]]


def quadratic_model(x, p):
    g = grad(x)
    Bp = hess_mul(p)
    return f(x) + dot(g, p) + 0.5 * dot(p, Bp)


def armijo_backtracking(x, p, alpha0=1.0, c1=1e-4, beta=0.5, max_iter=50):
    gx = grad(x)
    descent = dot(gx, p)
    if descent >= 0:
        raise ValueError("p must be a descent direction")

    fx = f(x)
    alpha = alpha0
    for _ in range(max_iter):
        trial = add(x, scale(alpha, p))
        if f(trial) <= fx + c1 * alpha * descent:
            return alpha
        alpha *= beta

    raise RuntimeError("Armijo backtracking failed")


def cauchy_step(x, Delta):
    g = grad(x)
    g_norm = norm(g)
    if g_norm == 0.0:
        return [0.0, 0.0]

    Bg = hess_mul(g)
    gBg = dot(g, Bg)

    # Cauchy step 沿负梯度方向，在 trust region 内选择模型最优的长度
    if gBg <= 0:
        tau = Delta / g_norm
    else:
        alpha_star = dot(g, g) / gBg
        tau = min(alpha_star, Delta / g_norm)

    return scale(-tau, g)


def trust_region_step(x, Delta, eta=0.1):
    p = cauchy_step(x, Delta)
    predicted = quadratic_model(x, [0.0, 0.0]) - quadratic_model(x, p)
    if predicted <= 0:
        raise RuntimeError("predicted reduction must be positive")

    x_trial = add(x, p)
    actual = f(x) - f(x_trial)
    rho = actual / predicted

    if rho < 0.25:
        new_Delta = 0.25 * Delta
    elif rho > 0.75 and abs(norm(p) - Delta) < 1e-12:
        new_Delta = min(2.0 * Delta, 10.0)
    else:
        new_Delta = Delta

    if rho >= eta:
        return x_trial, new_Delta, p, rho, True
    return x, new_Delta, p, rho, False


def demo_line_search():
    print("== Line Search Demo ==")
    x = [0.0, 0.0]
    for k in range(5):
        g = grad(x)
        p = scale(-1.0, g)
        alpha = armijo_backtracking(x, p)
        x = add(x, scale(alpha, p))
        print(
            f"iter={k:02d}, alpha={alpha:.4f}, x={x}, "
            f"f(x)={f(x):.6f}, ||g||={norm(grad(x)):.6f}"
        )
    print()


def demo_trust_region():
    print("== Trust Region Demo ==")
    x = [0.0, 0.0]
    Delta = 0.5
    for k in range(8):
        x, Delta, p, rho, accepted = trust_region_step(x, Delta)
        print(
            f"iter={k:02d}, accepted={accepted}, rho={rho:.4f}, "
            f"Delta={Delta:.4f}, p={p}, x={x}, f(x)={f(x):.6f}"
        )
        if norm(grad(x)) < 1e-8:
            break
    print()


if __name__ == "__main__":
    demo_line_search()
    demo_trust_region()
```

这段代码体现了两条完全不同的控制流：

- 线搜索不断试探 $\alpha$
- 信赖域先解子问题得到 $p$，再用 $\rho$ 决定接受与否，并调整 $\Delta$

如果你运行它，会看到一个很清楚的现象：

- 在线搜索部分，方向固定为负梯度，程序主要在找一个能通过 Armijo 的 $\alpha$
- 在信赖域部分，程序主要在根据 $\rho$ 调整 $\Delta$，而不是直接搜索某个“最优学习率”

为了让初学者把输出读懂，下面给一个字段说明表：

| 输出字段 | 含义 |
|---|---|
| `alpha` | 线搜索接受的步长 |
| `p` | 本轮实际走的步向量 |
| `Delta` | 当前信赖域半径 |
| `rho` | 实际降幅 / 预测降幅 |
| `accepted` | 是否接受候选步 |
| `||g||` | 梯度范数，常用来判断是否接近驻点 |

如果想从这份最小实现继续往工程版推进，通常会沿下面这条路线扩展：

| 最小实现 | 工程版常见替换 |
|---|---|
| Armijo 回溯 | 强 Wolfe 线搜索 |
| Cauchy step | Dogleg 或截断共轭梯度 |
| 显式 Hessian | Hessian-vector product 或拟牛顿近似 |
| 二维 toy function | Rosenbrock、逻辑回归、非线性最小二乘 |

一个真实工程例子是**相机标定**。相机内外参估计通常写成非线性最小二乘问题，目标是让投影误差最小。Levenberg-Marquardt 可以看作一种带阻尼的信赖域方法：当模型不可靠时增大阻尼，相当于缩小可信区域；当模型预测和真实误差吻合时减小阻尼，加快收敛。这类问题里，单纯牛顿步很容易因为曲率不稳而失控，信赖域通常更稳。

---

## 工程权衡与常见坑

线搜索的优点是实现简单，单步开销低，容易和梯度下降、共轭梯度、BFGS 拼接。缺点是它把“方向正确”当成前提。如果方向质量差，线搜索只能把坏方向的损失降到较小，不能从根本上修复方向。

信赖域的优点是稳健。即使 Hessian 不定，也能通过半径约束、阻尼或正定近似避免一次走飞。缺点是每轮都要解一个子问题，代码复杂度和单步计算量更高。

两类方法的工程权衡可以压缩成一张表：

| 维度 | 线搜索 | 信赖域 |
|---|---|---|
| 单轮实现复杂度 | 低 | 中到高 |
| 单轮计算成本 | 通常较低 | 通常较高 |
| 对方向质量的依赖 | 高 | 中 |
| 对曲率失真的容忍度 | 较弱 | 较强 |
| 典型成功场景 | 光滑、方向可靠 | 曲率复杂、牛顿步易失控 |

常见坑主要有六类：

1. 把 Armijo 当成“越严格越好”。不是。$c_1$ 太大，会要求过强下降，导致频繁缩步；太小则容易接受质量不高的步。
2. Wolfe 条件设置过紧。这样会增加函数和梯度评估次数，尤其在高维问题上成本明显上升。
3. 信赖域里只看是否下降，不看 $\rho_k$。这会让半径更新失真，模型明明已经不可信，算法却还在大范围试探。
4. Hessian 非正定时直接用牛顿步。此时二次模型可能沿某些方向无下界，容易产生错误的大步。
5. 把“步被拒绝”理解成算法失败。不是。在线搜索和信赖域里，拒绝步本身就是正常控制逻辑的一部分。
6. 没有记录监控指标。优化器经常不是“数学错了”，而是“你没看到它什么时候开始失控”。

建议至少记录下面这些量：

| 监控项 | 线搜索关注点 | 信赖域关注点 |
|---|---|---|
| $\|\nabla f(x_k)\|$ | 是否接近驻点 | 是否接近驻点 |
| 步长大小 | $\alpha_k$ 是否长期过小 | $\|p_k\|$ 是否长期卡边界 |
| 函数下降量 | 是否出现震荡 | 实际下降是否稳定 |
| 模型误差 | 一般不单独记录 | $|\rho_k-1|$ 是否持续很大 |
| 拒绝步比例 | 回溯次数是否过多 | 步是否频繁被拒绝 |

如果要给新手一个排障顺序，通常按这个顺序看最有效：

1. 先看方向是不是下降方向，即 $\nabla f(x_k)^Tp_k<0$ 是否成立。
2. 再看函数值是否真的下降，而不是只看参数更新了没。
3. 然后看步长或半径是不是长期被压得过小。
4. 最后再怀疑 Hessian、拟牛顿更新或模型近似是否失真。

真实工程里，BFGS 配合 Armijo-Wolfe 很常见。它的重要性在于：line search 不是“给 BFGS 加一个保险”，而是 BFGS 收敛理论的一部分。步长规则如果换得过于随意，很多理论保证会一起消失。

在神经网络训练里，完整线搜索不像传统数值优化那样常用，因为一次函数评估就可能意味着一次大批量前向和反向传播，成本高，而且随机梯度噪声会破坏精确判据。但在小批量较大、目标较平滑、或做二阶微调时，回溯线搜索仍然是可用工具。信赖域思想也会以近似形式出现，比如用 KL 散度或二次近似去约束参数更新范围，但那已经不是本文讨论的经典无约束信赖域框架。

---

## 替代方案与适用边界

如果问题满足“方向便宜、函数平滑、每次评估成本不算夸张”，线搜索通常是更直接的选择。比如逻辑回归、L2 正则线性模型、很多经典凸优化问题，都可以从梯度法或 BFGS + Armijo 开始。

如果问题属于**非线性最小二乘**，而且曲率信息很重要，信赖域通常更自然。Levenberg-Marquardt、Dogleg 都属于这一路线，典型应用包括：

- 相机标定
- 位姿估计
- 曲线拟合
- 机器人参数辨识
- Bundle adjustment

下面给出一个简化选择表：

| 问题特征 | 更推荐 |
|---|---|
| 下降方向容易算，Hessian 不必显式构造 | 线搜索 |
| 曲率信息可靠，想利用二阶速度 | 线搜索 + 牛顿/拟牛顿 |
| Hessian 可能不定，牛顿步常常过大 | 信赖域 |
| 非线性最小二乘，残差结构明显 | LM / Dogleg / 信赖域 |
| 目标含噪，精确函数评估代价高 | 固定步长或自适应一阶法，少用严格线搜索 |
| 需要更强稳健性，能接受更高单步成本 | 信赖域 |

如果还想再简化，可以按“评估贵不贵、模型准不准”两条轴来判断：

| 情况 | 更常见选择 | 原因 |
|---|---|---|
| 函数评估便宜，方向容易拿到 | 线搜索 | 多试几个 $\alpha$ 成本可接受 |
| 函数评估贵，但局部模型有结构 | 信赖域 | 更愿意把计算花在模型和子问题上 |
| 问题噪声大、目标不稳定 | 固定步长或自适应一阶法 | 精确判据本身会失真 |
| 问题规模大、Hessian 稀疏 | 截断 CG 型信赖域 | 不必显式构造 Hessian |

还要明确一个边界：**现代深度学习里常见的 Adam、RMSProp，并不是线搜索，也不是经典信赖域**。它们是基于梯度统计量做坐标级缩放的自适应一阶法，目标是处理噪声和尺度差异，不是通过 Armijo 或 $\rho_k$ 直接验证局部模型。

同样，学习率调度器也不等于线搜索。比如余弦退火、指数衰减、warmup，本质上是预设时间表；线搜索则是根据当前点的函数和梯度反馈即时决策。

所以，不要把所有“自动调步长”的算法都归到这两类里。线搜索和信赖域的共同特征是：**每一步都试图用明确的数学判据解释为什么这个步子可以走**。

如果只想要一个实践上的起点，可以这样选：

1. 光滑无约束问题，先试 `BFGS + strong Wolfe`。
2. 非线性最小二乘问题，先试 `LM / Dogleg / trust region`。
3. Hessian 明显不稳或牛顿步经常失控，优先考虑信赖域。
4. 目标噪声很大、每次评估非常贵，不要先上严格线搜索。

---

## 参考资料

1. Jorge Nocedal, Stephen J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006.  
2. FitBenchmarking, “Trust Region”. https://fitbenchmarking.readthedocs.io/en/stable/users/algorithms/trust_region.html  
3. Qiujiang Jin, Ruichen Jiang, Aryan Mokhtari, “Non-asymptotic Global Convergence Analysis of BFGS with the Armijo-Wolfe Line Search”, NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/1e269abc604816c35f600ae14b354efd-Paper-Conference.pdf  
4. NEOS Guide, “Levenberg-Marquardt Method”. https://neos-guide.org/guide/algorithms/lmm/  
5. Cornell Optimization Wiki, “Trust-region methods”. https://optimization.cbe.cornell.edu/index.php?title=Trust-region_methods  
6. Wikipedia, “Line search”. https://en.wikipedia.org/wiki/Line_search
