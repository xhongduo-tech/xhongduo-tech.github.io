## 核心结论

ADMM，交替方向乘子法，是一种把“大优化问题拆成两个较小子问题轮流解”的方法。它最适合这类模板：

$$
\min_{x,z}\ f(x)+g(z)\quad \text{s.t.}\quad Ax+Bz=c
$$

这里的“可分”意思是目标函数能拆成两部分，各自只依赖一块变量；“线性约束”意思是两块变量之间只通过线性关系耦合。

它的核心不是直接硬解约束，而是把约束违背量写进增广拉格朗日函数：

$$
L_\rho(x,z,y)=f(x)+g(z)+y^\top(Ax+Bz-c)+\frac{\rho}{2}\|Ax+Bz-c\|_2^2
$$

然后反复做三步：

1. 固定 $z,y$，更新 $x$
2. 固定 $x,y$，更新 $z$
3. 用当前约束残差更新对偶变量 $y$

标准两块凸问题下，ADMM 具有渐进收敛性质，常见目标残差或可行性度量可达到 $O(1/t)$ 的迭代复杂度量级。工程上最关键的超参数是 $\rho$，它决定“优先压约束违背”还是“优先让对偶变量更平稳”。

| 更新阶段 | 公式 | 作用 |
|---|---|---|
| 原始变量 $x$ 更新 | $x^{k+1}=\arg\min_x L_\rho(x,z^k,y^k)$ | 先优化第一块变量 |
| 原始变量 $z$ 更新 | $z^{k+1}=\arg\min_z L_\rho(x^{k+1},z,y^k)$ | 再优化第二块变量 |
| 对偶变量 $y$ 更新 | $y^{k+1}=y^k+\rho r^{k+1}$ | 用残差反馈约束违背 |
| 原始残差 | $r^{k+1}=Ax^{k+1}+Bz^{k+1}-c$ | 衡量约束是否满足 |
| 对偶残差 | 常写作 $s^{k+1}=\rho A^\top B(z^{k+1}-z^k)$ | 衡量相邻迭代变化是否过猛 |

玩具例子最能看清机制。设

$$
f(x)=\frac12(x-1)^2,\qquad g(z)=\frac12 z^2,\qquad x=z
$$

取 $\rho=1$，初值 $x^0=z^0=y^0=0$。第一轮就得到 $x^1=\frac12,\ z^1=\frac14,\ y^1=\frac14$；继续迭代后会收敛到 $x^\star=z^\star=\frac12$。这说明 ADMM 会一边优化局部目标，一边把两块变量往一致性约束上拉回去。

---

## 问题定义与边界

ADMM 处理的是“两块目标可分，但约束把它们绑在一起”的问题。标准形式是：

$$
\min_{x,z}\ f(x)+g(z)\quad \text{s.t.}\quad Ax+Bz=c
$$

边界有三层。

第一层，$f$ 和 $g$ 最好能单独求解，至少各自的子问题要比原问题容易。否则拆分没有意义。

第二层，耦合约束通常要求是线性的，即 $Ax+Bz=c$。这样二次罚项

$$
\frac{\rho}{2}\|Ax+Bz-c\|_2^2
$$

才能稳定地把“局部最优”和“全局一致”连接起来。

第三层，最经典、结论最稳的是“两块变量、凸目标”。多块 ADMM、非凸 ADMM 也常用，但收敛条件会更严格，不能把标准结论直接照搬。

为什么不直接把变量并成一个大变量 $w$ 去做梯度下降？因为拆分后的两个子问题常常来自不同来源。

真实工程例子：联邦学习里，每个客户端维护本地模型参数 $x_i$，服务器维护全局一致变量 $z$。目标常写成：

$$
\min_{\{x_i\},z}\ \sum_{i=1}^N f_i(x_i)+g(z)\quad \text{s.t.}\quad x_i-z=0,\ \forall i
$$

这里 $f_i$ 是每个客户端自己的损失，$g(z)$ 可以是正则项。之所以不直接合并成单变量，是因为数据留在客户端，本地子问题和全局一致性问题天然分离。ADMM 的价值就在这里：客户端本地解自己的 $x_i$，服务器只处理聚合变量和对偶变量，不需要上传原始数据。

---

## 核心机制与推导

把原问题写成增广拉格朗日后：

$$
L_\rho(x,z,y)=f(x)+g(z)+y^\top(Ax+Bz-c)+\frac{\rho}{2}\|Ax+Bz-c\|_2^2
$$

ADMM 的三步迭代是：

$$
x^{k+1}=\arg\min_x L_\rho(x,z^k,y^k)
$$

$$
z^{k+1}=\arg\min_z L_\rho(x^{k+1},z,y^k)
$$

$$
y^{k+1}=y^k+\rho\big(Ax^{k+1}+Bz^{k+1}-c\big)
$$

这里“对偶变量”可以理解成约束的价格信号：哪个方向更违反约束，后续更新就会被更强地拉回。

继续看玩具例子：

$$
\min_{x,z}\ \frac12(x-1)^2+\frac12 z^2\quad \text{s.t.}\quad x-z=0
$$

此时 $A=1,B=-1,c=0$，所以

$$
L_\rho(x,z,y)=\frac12(x-1)^2+\frac12 z^2+y(x-z)+\frac{\rho}{2}(x-z)^2
$$

令 $\rho=1$。

先做 $x$ 步。固定 $z^k,y^k$，解

$$
x^{k+1}=\arg\min_x \frac12(x-1)^2+y^k(x-z^k)+\frac12(x-z^k)^2
$$

对 $x$ 求导并令其为零：

$$
(x-1)+y^k+(x-z^k)=0
$$

整理得

$$
2x=1+z^k-y^k
$$

所以

$$
x^{k+1}=\frac{1+z^k-y^k}{2}
$$

再做 $z$ 步。固定 $x^{k+1},y^k$，解

$$
z^{k+1}=\arg\min_z \frac12 z^2+y^k(x^{k+1}-z)+\frac12(x^{k+1}-z)^2
$$

对 $z$ 求导并令其为零：

$$
z-y^k-(x^{k+1}-z)=0
$$

整理得

$$
2z=x^{k+1}+y^k
$$

所以

$$
z^{k+1}=\frac{x^{k+1}+y^k}{2}
$$

最后更新对偶变量：

$$
y^{k+1}=y^k+(x^{k+1}-z^{k+1})
$$

从 $x^0=z^0=y^0=0$ 开始：

- 第 1 轮：$x^1=\frac12,\ z^1=\frac14,\ y^1=\frac14$
- 第 2 轮：$x^2=\frac12,\ z^2=\frac38,\ y^2=\frac38$
- 第 3 轮：$x^3=\frac12,\ z^3=\frac7{16},\ y^3=\frac7{16}$

可以看到 $x^k$ 很快锁到 $\frac12$，$z^k,y^k$ 逐步逼近 $\frac12$。  
原始残差是

$$
r^k=x^k-z^k
$$

它会趋近于 0，说明约束越来越满足。对偶残差反映变量变化速度，若它很大，通常表示步子过猛或 $\rho$ 偏大。

$\rho$ 的直观作用可以直接记成一句话：

- $\rho$ 小：更像“先顾各自目标”，约束收紧得慢
- $\rho$ 大：更像“先逼一致性”，但可能引起对偶震荡

---

## 代码实现

下面给一个新手友好的可运行版本。为了让接口保持通用，这个实现把 $x$ 子问题和 $z$ 子问题都用简单的梯度下降内循环近似求解，适合教学，不适合高性能生产环境。

```python
import numpy as np

def admm_solver(f_grad, g_grad, A, B, c, rho, max_iter=200, tol=1e-8,
                x0=None, z0=None, y0=None, inner_steps=200, lr=0.1, verbose=False):
    A = np.atleast_2d(np.array(A, dtype=float))
    B = np.atleast_2d(np.array(B, dtype=float))
    c = np.atleast_1d(np.array(c, dtype=float))

    n_x = A.shape[1]
    n_z = B.shape[1]
    m = A.shape[0]

    x = np.zeros(n_x) if x0 is None else np.array(x0, dtype=float)
    z = np.zeros(n_z) if z0 is None else np.array(z0, dtype=float)
    y = np.zeros(m) if y0 is None else np.array(y0, dtype=float)

    history = []

    for k in range(max_iter):
        z_prev = z.copy()

        # x-update: minimize f(x) + y^T(Ax + Bz - c) + (rho/2)||Ax + Bz - c||^2
        for _ in range(inner_steps):
            residual_x = A @ x + B @ z - c
            grad_x = f_grad(x) + A.T @ y + rho * A.T @ residual_x
            x = x - lr * grad_x

        # z-update: minimize g(z) + y^T(Ax + Bz - c) + (rho/2)||Ax + Bz - c||^2
        for _ in range(inner_steps):
            residual_z = A @ x + B @ z - c
            grad_z = g_grad(z) + B.T @ y + rho * B.T @ residual_z
            z = z - lr * grad_z

        # primal residual
        r = A @ x + B @ z - c

        # dual update
        y = y + rho * r

        # dual residual: standard form
        s = rho * (A.T @ (B @ (z - z_prev)))

        r_norm = np.linalg.norm(r)
        s_norm = np.linalg.norm(s)
        history.append((r_norm, s_norm))

        if verbose and (k % 10 == 0 or k == max_iter - 1):
            print(f"iter={k:03d}, r={r_norm:.3e}, s={s_norm:.3e}, x={x}, z={z}, y={y}")

        # rho can be adapted here if needed:
        # if r_norm > 10 * s_norm: rho *= 2
        # elif s_norm > 10 * r_norm: rho /= 2

        if r_norm < tol and s_norm < tol:
            break

    return x, z, y, history


# 玩具例子:
# min 0.5*(x-1)^2 + 0.5*z^2  s.t. x - z = 0
def f_grad(x):
    return x - 1.0

def g_grad(z):
    return z

A = np.array([[1.0]])
B = np.array([[-1.0]])
c = np.array([0.0])

x, z, y, history = admm_solver(
    f_grad, g_grad, A, B, c,
    rho=1.0, max_iter=200, tol=1e-7,
    inner_steps=100, lr=0.1, verbose=False
)

assert abs(x[0] - 0.5) < 1e-3
assert abs(z[0] - 0.5) < 1e-3
assert abs(x[0] - z[0]) < 1e-3
print("solution:", x[0], z[0], y[0])
```

这段代码对应 ADMM 的标准骨架：

1. 初始化 $x,z,y$
2. 先解 $x$ 子问题
3. 再解 $z$ 子问题
4. 计算原始残差 $r^k$
5. 更新对偶变量 $y$
6. 计算对偶残差 $s^k$
7. 判断是否收敛

如果子问题有闭式解，工程上应优先写闭式更新；如果没有闭式解，才考虑内层迭代器、近端算子或专门数值求解器。

真实工程例子：在模型压缩的量化感知训练里，可以把“连续权重”记为 $x$，把“满足量化约束的辅助变量”记为 $z$，约束写成 $x-z=0$。这样训练时既能优化任务损失，又能逐步把参数拉向量化可部署的结构。

---

## 工程权衡与常见坑

ADMM 好用，但它不是“随便套就稳”。

| 常见坑 | 典型现象 | 缓解方式 |
|---|---|---|
| $\rho$ 太小 | 原始残差下降慢，约束长期不满足 | 增大 $\rho$，或用残差比值自适应调参 |
| $\rho$ 太大 | 对偶变量震荡，子问题变硬 | 降低 $\rho$，加松弛因子 |
| 子问题解不准 | 外层迭代看似在跑，但整体不收敛 | 设定内层精度，必要时逐步提高求解精度 |
| 非凸目标 | 可能停在局部最优，且理论保证变弱 | 明确只求稳定可用解，不宣称全局最优 |
| 多块直接扩展 | 两块版稳定，多块版可能失稳 | 用带近端项的多块变体，或重写成两块结构 |
| 通信开销大 | 分布式场景每轮同步成本高 | 减少同步频率，做局部多步更新 |

联邦学习是最典型的真实工程例子。每个客户端求自己的 $x_i$ 子问题，服务器聚合全局变量 $z$ 和对偶变量 $y_i$。优势是数据不出本地，传的是参数或乘子信息，不是原始样本。问题在于：

- $\rho$ 太小，各客户端容易“各练各的”，全局一致性差
- $\rho$ 太大，服务器每轮都强力拉齐，本地异构数据会导致震荡
- 通信轮数多时，ADMM 的理论优点可能被网络延迟吃掉

所以实际系统里常见的做法是：根据原始残差和对偶残差的比例自适应调整 $\rho$。粗略规则是：

- 若 $\|r^k\| \gg \|s^k\|$，说明约束违背更严重，增大 $\rho$
- 若 $\|s^k\| \gg \|r^k\|$，说明对偶变化过快，减小 $\rho$

这本质上是在平衡“更快满足约束”和“不要更新过猛”两件事。

---

## 替代方案与适用边界

ADMM 不是所有约束优化问题的默认最优解。选择它，通常是因为“目标可分、约束线性、分布式协调重要”。

| 算法 | 适用场景 | 主要优势 | 局限 |
|---|---|---|---|
| 梯度法 | 无约束或弱约束、单机训练 | 实现最简单 | 难直接处理耦合约束 |
| 投影梯度法 | 约束集合容易投影 | 每步直观，易实现 | 若投影本身很难，成本高 |
| 罚函数法 | 想快速把约束塞进目标 | 代码改动小 | 罚项过大时病态，过小时不可行 |
| ADMM | 可分目标 + 线性耦合约束 + 分布式 | 能拆分、适合并行、对局部结构友好 | 调参敏感，多块/非凸要谨慎 |
| 自适应/松弛 ADMM | 标准 ADMM 收敛慢或震荡 | 更稳，常更快 | 理论和实现都更复杂 |

把 ADMM 和原始罚函数法放在同一个问题上比较，差别很清楚。

若直接做罚函数法：

$$
\min_{x,z}\ f(x)+g(z)+\frac{\mu}{2}\|Ax+Bz-c\|^2
$$

你把约束丢进目标后，就变成一个单体问题。优点是实现简单；缺点是 $\mu$ 一大，数值条件会变差，优化器容易难走。

ADMM 等于在罚函数法基础上，再显式保留对偶变量，用“局部最优 + 约束反馈”交替推进。  
所以通常：

- 单机、小规模、投影容易时，投影梯度或罚函数法更省事
- 分布式、多端协同、局部子问题结构清晰时，ADMM 更合适
- 若标准 ADMM 对 $\rho$ 很敏感，可考虑自适应罚项和松弛因子版本。2025 年的一些改进工作就是沿这条线：按迭代信息动态调 $\rho$，并在乘子更新中加入松弛，目标是提升稳定性和实际效率

适用边界也要说清：若问题不是两块凸结构，而是高耦合、多块、强非凸，ADMM 仍可能可用，但不能默认继承“标准两块凸 ADMM”的全部收敛结论。

---

## 参考资料

1. Optimization Notes, “Alternating Direction Method of Multipliers”  
   https://optimizationnotes.readthedocs.io/en/latest/convex_optimization/admm.html  
   用途：适合先建立最基础的三步迭代公式、scaled form 和 prox 视角。新手应先读这一篇，把 $x$ 更新、$z$ 更新、对偶更新的结构读熟。

2. Stephen Boyd 的 ADMM 资料入口  
   https://stanford.edu/~boyd/admm.html  
   用途：适合继续扩展到分布式优化、统计学习和经典应用。若想把 ADMM 放回更大的优化方法谱系里，这个入口很有价值。

3. Lu, Zhu, Dang, 2024, “Symmetric ADMM-Based Federated Learning with a Relaxed Step”  
   https://www.mdpi.com/2227-7390/12/17/2661  
   用途：展示了联邦学习里的 ADMM 变体如何处理局部变量、全局变量、双重对偶更新和通信效率问题。读这篇可以把“客户端子问题 + 服务器协调”看成 ADMM 的工程落地。

4. Peng et al., 2025, “An Improvement of the Alternating Direction Method of Multipliers to Solve the Convex Optimization Problem”  
   https://www.mdpi.com/2227-7390/13/5/811  
   用途：聚焦自适应罚项和乘子松弛更新。若你已经理解标准 ADMM，再读这篇能明白为什么工程上经常不把 $\rho$ 固定死。

建议阅读顺序：

1. 先读 Optimization Notes，建立公式骨架
2. 再看 Boyd 的资料入口，理解 ADMM 在分布式优化中的位置
3. 然后读 2024 联邦学习论文，看真实系统中的变量拆分与通信流程
4. 最后读 2025 自适应改进论文，理解标准 ADMM 为什么需要调参与扩展
