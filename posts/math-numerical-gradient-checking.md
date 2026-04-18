## 核心结论

梯度检查是用数值差分近似 $\partial L / \partial \theta_i$，再和反向传播得到的解析梯度比较，用来验证 backward 实现是否正确。

它的目标不是“算出更好的梯度”。训练时真正使用的梯度仍然来自反向传播。梯度检查只在调试阶段低频使用，用来发现自定义 loss、自定义算子、手写 backward、复杂张量广播等位置的实现错误。

最常用的方法是中心差分。设损失函数为 $L(\theta)$，参数向量为 $\theta$，第 $i$ 维单位向量为 $e_i$，数值梯度定义为：

$$
\tilde g_i(h)=\frac{L(\theta+h e_i)-L(\theta-h e_i)}{2h}
$$

新手版理解：如果反向传播说某个参数的梯度是 `g`，梯度检查会把这个参数轻微往右挪一点，再轻微往左挪一点，分别计算两次损失。若损失变化对应的斜率和 `g` 基本一致，说明这一维梯度大概率正确；若差很多，说明实现可能有 bug。

| 参数维度 | 解析梯度 `g_bp` | 数值梯度 `g_num` | 相对误差 | 判断 |
|---:|---:|---:|---:|---|
| 0 | 2.000000 | 2.000000 | 1e-10 | 通过 |
| 1 | -0.500000 | -0.499999 | 1e-6 | 通过 |
| 2 | 3.100000 | 2.400000 | 1.27e-1 | 需要排查 |

梯度检查的判断重点通常是相对误差，而不是只看绝对误差：

$$
\text{rel\_err}=\frac{|g_{\text{num}}-g_{\text{bp}}|}{\max(\epsilon, |g_{\text{num}}|+|g_{\text{bp}}|)}
$$

其中 $\epsilon$ 是防止分母为 0 的小常数。

---

## 问题定义与边界

损失函数 `loss function` 是把模型参数、输入和标签映射成一个标量误差的函数。记作 $L(\theta)$，其中 $\theta$ 是所有待求导参数拼成的向量。梯度检查关注的是某一维参数 $\theta_i$ 的偏导数：

$$
g_i=\frac{\partial L}{\partial \theta_i}
$$

解析梯度 `analytic gradient` 是由反向传播或自动微分系统根据链式法则算出的梯度。数值梯度 `numeric gradient` 是只通过多次前向计算损失近似出来的梯度。梯度检查就是比较二者是否一致。

适用边界如下：

| 场景 | 是否适合梯度检查 | 原因 |
|---|---:|---|
| 自定义 loss | 是 | 可以逐维比对损失对参数的导数 |
| 自定义 CUDA 算子 | 是 | backward 容易出现索引、广播、累加错误 |
| 手写反向传播 | 是 | 可以定位某一层或某一维的梯度错误 |
| 含 dropout 的训练模式 | 否 | 随机性会污染前后两次损失 |
| 数据增强在线随机变化 | 否 | 每次前向不是同一个函数 |
| ReLU 的 kink 点附近 | 谨慎 | 数值梯度可能跨过不可导点 |
| 大规模训练主循环 | 否 | 成本太高，不适合作为训练手段 |

`kink 点` 是函数不可导或左右导数不一致的位置，例如 ReLU 在 0 处、max 在两个输入相等处、hinge loss 在间隔边界处。中心差分需要函数在局部平滑，否则左右两边看到的可能不是同一段函数。

玩具例子：取 $f(x)=x^3$，在 $x=1$ 处检查导数。真实导数是 $f'(x)=3x^2$，所以 $f'(1)=3$。中心差分为：

$$
\frac{(1+h)^3-(1-h)^3}{2h}=3+h^2
$$

若 $h=10^{-2}$，数值梯度是 $3.0001$，绝对误差是 $10^{-4}$。这不是 backward 错了，而是中心差分本身有截断误差。

真实工程例子：你实现了一个自定义对比学习 loss，训练时 loss 能下降，但这不能证明 backward 正确。因为错误梯度也可能在某些数据上让 loss 暂时下降。更稳的做法是固定一小批输入，关闭随机增强，把参数切到 `float64`，抽几维参数做中心差分，对照自动微分或手写 backward 的结果。

---

## 核心机制与推导

中心差分比单边差分更稳定，核心原因是左右两侧的低阶误差会抵消一部分。单边差分是：

$$
\frac{L(\theta+h e_i)-L(\theta)}{h}
$$

它只看参数往一个方向移动后的变化，截断误差通常是一阶 $O(h)$。中心差分同时看两边：

$$
\tilde g_i(h)=\frac{L(\theta+h e_i)-L(\theta-h e_i)}{2h}
$$

对一维函数做 Taylor 展开：

$$
L(\theta+h e_i)=L(\theta)+h g_i+\frac{h^2}{2}L''+\frac{h^3}{6}L^{(3)}+O(h^4)
$$

$$
L(\theta-h e_i)=L(\theta)-h g_i+\frac{h^2}{2}L''-\frac{h^3}{6}L^{(3)}+O(h^4)
$$

两式相减后，常数项和二阶项抵消，得到：

$$
\tilde g_i(h)=g_i+\frac{h^2}{6}L^{(3)}(\xi)+O(h^4)
$$

这说明中心差分的截断误差是 $O(h^2)$。截断误差 `truncation error` 是用有限步长近似极限导数带来的误差。步长 $h$ 越大，这部分误差越明显。

但 $h$ 不能无限小。浮点数 `floating point number` 是计算机用有限位数表示实数的方法，它会产生舍入误差。舍入误差 `rounding error` 是真实数值被存进有限精度格式时丢失低位造成的误差。当 $h$ 很小时，$L(\theta+h e_i)$ 和 $L(\theta-h e_i)$ 非常接近，相减会放大舍入误差。

实际误差常用下面的模型理解：

$$
E(h)\approx C_t h^2 + C_r\frac{u}{h}
$$

其中 $C_t$ 与函数的高阶导数有关，$C_r$ 与计算过程中的浮点误差放大有关，$u$ 是机器精度单位圆整误差。这个式子给出一个 U 型关系：$h$ 太大时截断误差主导，$h$ 太小时舍入误差主导，中间有一个较好的区间。

只看数量级，中心差分常见的理论最优步长接近 $u^{1/3}$。双精度 `float64` 的机器精度约为 $2.22\times 10^{-16}$，$u^{1/3}$ 约在 $10^{-5}$ 量级，所以很多梯度检查默认从 `1e-5` 或 `1e-6` 附近试起。单精度 `float32` 的机器精度约为 $1.19\times 10^{-7}$，对应步长更大，且误差阈值必须放宽。

---

## 代码实现

梯度检查实现的关键不是公式复杂，而是保证前向函数稳定、确定、可重复。最小流程是：

| 步骤 | 目的 |
|---|---|
| 固定随机种子 | 让每次前向使用同一批数据和同一状态 |
| 切到评估模式 | 关闭 dropout、batchnorm 训练态更新等噪声 |
| 使用 `float64` | 降低舍入误差 |
| 抽样少量参数 | 避免全量逐维检查成本过高 |
| 做中心差分 | 得到数值梯度 |
| 比较相对误差和绝对误差 | 避免零附近误判 |

下面是一个可运行的纯 Python 例子，检查一个平方损失的梯度，并包含 Kahan 求和：

```python
import math
import random

def kahan_sum(xs):
    s = 0.0
    c = 0.0
    for x in xs:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s

def loss_fn(theta, x, y):
    # 线性模型：pred = theta[0] * x + theta[1]
    terms = []
    for xi, yi in zip(x, y):
        pred = theta[0] * xi + theta[1]
        err = pred - yi
        terms.append(0.5 * err * err)
    return kahan_sum(terms) / len(x)

def analytic_grad(theta, x, y):
    g0_terms = []
    g1_terms = []
    for xi, yi in zip(x, y):
        pred = theta[0] * xi + theta[1]
        err = pred - yi
        g0_terms.append(err * xi)
        g1_terms.append(err)
    n = len(x)
    return [kahan_sum(g0_terms) / n, kahan_sum(g1_terms) / n]

def numeric_grad(loss, theta, i, h=1e-5):
    theta_pos = theta[:]
    theta_neg = theta[:]
    theta_pos[i] += h
    theta_neg[i] -= h
    return (loss(theta_pos) - loss(theta_neg)) / (2.0 * h)

def rel_error(g_num, g_bp, eps=1e-12):
    return abs(g_num - g_bp) / max(eps, abs(g_num) + abs(g_bp))

random.seed(0)
x = [0.0, 1.0, 2.0, 3.0]
y = [1.0, 3.0, 5.0, 7.0]
theta = [1.8, 0.7]

loss = lambda th: loss_fn(th, x, y)
g_bp = analytic_grad(theta, x, y)

for i in range(len(theta)):
    g_num = numeric_grad(loss, theta, i, h=1e-5)
    err = rel_error(g_num, g_bp[i])
    assert err < 1e-8, (i, g_num, g_bp[i], err)

assert abs(loss_fn([2.0, 1.0], x, y)) < 1e-12
```

这段代码里的 `kahan_sum` 是补偿求和。补偿求和 `compensated summation` 是用额外变量记录前一次加法中被舍入掉的低位，从而降低大量求和时的误差累积。它的核心形式是：

```text
y = x_k - c
t = s + y
c = (t - s) - y
s = t
```

真实工程里，损失常常是很多样本项、很多 token 项或很多像素项的聚合。如果前向损失本身因为裸加而不稳定，那么数值梯度会先被损失误差污染，再拿它去判断 backward，就会得到误导性结论。Kahan、pairwise summation、分块求和都能缓解这个问题。

阈值建议要区分精度：

| 精度 | 推荐步长起点 | 相对误差经验阈值 | 说明 |
|---|---:|---:|---|
| `float64` | `1e-5` 到 `1e-6` | `1e-7` 到 `1e-9` | 适合严格梯度检查 |
| `float32` | `1e-2` 到 `1e-3` | `1e-3` 到 `1e-5` | 只能粗筛，误差更大 |
| 混合精度 | 不建议直接检查 | 需放宽 | loss scaling 和 cast 会干扰判断 |

---

## 工程权衡与常见坑

梯度检查最常见的问题不是公式写错，而是检查环境不是同一个确定函数。比如训练模式下的 dropout 每次随机丢弃不同神经元，那么 $L(\theta+h e_i)$ 和 $L(\theta-h e_i)$ 实际来自两个不同函数。此时差分结果没有清晰含义。

| 常见坑 | 现象 | 规避方式 |
|---|---|---|
| `h` 太大 | 数值梯度偏离真实梯度 | 减小步长，观察误差是否下降 |
| `h` 太小 | 梯度抖动大 | 避免低于浮点精度可承受范围 |
| `float32` 做严格检查 | 相对误差明显偏大 | 优先切到 `float64` |
| 只看绝对误差 | 小梯度附近误判 | 主看相对误差，零附近补看绝对误差 |
| ReLU/max/hinge 附近 | 左右差分不稳定 | 避开 kink 点或分段检查 |
| dropout 未关闭 | 每次 loss 不一致 | 使用 eval 模式并固定随机性 |
| batchnorm 训练态 | running stats 改变 | 冻结统计量 |
| 大量求和裸加 | 损失低位不稳定 | 用 Kahan、pairwise 或分块求和 |
| 全量参数逐维扫 | 检查极慢 | 随机抽样关键参数 |

绝对误差 `absolute error` 是 $|g_{\text{num}}-g_{\text{bp}}|$。它适合判断非常接近 0 的梯度。相对误差 `relative error` 是把误差除以梯度规模后的比例，更适合比较不同量级的参数。工程上通常两个都看：普通梯度主看相对误差，接近 0 的梯度补看绝对误差。

一个实用判断模板是：

| 情况 | 判断方式 |
|---|---|
| $|g_{\text{num}}|+|g_{\text{bp}}|$ 不小 | 主看相对误差 |
| 两者都接近 0 | 看绝对误差是否足够小 |
| 数值梯度和解析梯度符号相反 | 高优先级排查 |
| 只有少数维度失败 | 排查索引、广播、mask |
| 大量维度整体偏一个倍数 | 排查平均、求和、缩放系数 |

为什么 `h` 不能越小越好：因为中心差分分子是两个非常接近的数相减。两个大数相减得到小数时，低位有效数字容易丢失，这叫消减误差 `cancellation error`。当分子已经被舍入误差污染，再除以很小的 $2h$，误差会被进一步放大。

---

## 替代方案与适用边界

梯度检查适合小规模、低频率、调试阶段。它不适合大规模训练主循环，也不适合线上推理。原因很直接：检查一维参数需要两次前向，检查 $n$ 维参数需要 $2n$ 次前向，成本随参数维度线性增长。

| 方法 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| 中心差分梯度检查 | 调试 backward | 直观、可靠、精度较好 | 慢 |
| 单边差分 | 粗略验证 | 实现简单 | 截断误差更大 |
| 随机方向检查 | 高维参数 | 成本更低 | 不能逐维定位 |
| 自动微分对照 | 框架内部验证 | 快，覆盖常规算子 | 不能证明自定义 backward 正确 |
| 分模块检查 | 复杂模型调试 | 易定位问题 | 需要拆出稳定子函数 |

随机方向检查 `random directional check` 是不逐维检查，而是随机采样一个方向 $v$，比较方向导数：

$$
\frac{L(\theta+h v)-L(\theta-h v)}{2h}
\quad \text{与} \quad
g^\top v
$$

它适合高维参数的快速粗筛。如果方向检查失败，说明整体梯度有问题；如果方向检查通过，也不能保证每一维都正确，因为某些错误可能在随机方向上被抵消。

真实工程中可以采用分层策略：先用随机方向检查确认整体没有明显错误，再对关键参数、边界参数、mask 分支、广播分支做逐维中心差分。对于自定义 CUDA 算子，还应准备极小输入形状，例如 batch size 为 1、通道数为 2、包含边界索引的输入，这样更容易定位 backward 的错误来源。

梯度检查的适用边界可以概括为三点：第一，函数必须尽量确定；第二，检查点应避开不可导位置；第三，数值精度要足够支撑差分。只要这三点不满足，失败结果就不一定说明 backward 错，也可能是检查方法本身不适用。

---

## 参考资料

| 资料 | 适合看什么 |
|---|---|
| [CS231n Gradient Checks](https://cs231n.github.io/neural-networks-3/) | 中心差分、相对误差阈值、双精度建议 |
| [NumPy `finfo` 文档](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html) | `eps`、机器精度、浮点表示 |
| [UIUC CS357 Finite Difference Methods](https://courses.grainger.illinois.edu/cs357/fa2023/notes/ref-19-finite-difference.html) | 前向差分与中心差分的截断阶数 |
| [Numerical Validation of Compensated Summation Algorithms with Stochastic Arithmetic](https://www.sciencedirect.com/science/article/pii/S1571066115000481) | 补偿求和如何降低舍入误差 |
| [A Class of Fast and Accurate Summation Algorithms](https://research.manchester.ac.uk/en/publications/a-class-of-fast-and-accurate-summation-algorithms/) | 低精度下的准确求和与误差界分析 |
