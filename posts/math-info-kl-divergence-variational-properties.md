## 核心结论

KL 散度的变分性质说的是：KL 散度不仅可以通过概率密度直接计算，也可以写成一个“对函数取最优”的问题。

核心公式是 Donsker-Varadhan 变分表示：

$$
D_{\mathrm{KL}}(P\Vert Q)=\sup_f \left\{\mathbb{E}_P[f]-\log \mathbb{E}_Q[e^f]\right\}
$$

其中 $P$ 和 $Q$ 是两个概率分布，$f$ 是定义在同一个样本空间上的实值函数。实值函数就是输入一个样本、输出一个分数的函数。

这条公式的直观含义是：不是直接计算两个分布的差异，而是去找一个最会区分 $P$ 和 $Q$ 的打分函数。这个函数在 $P$ 下尽量打高分，在 $Q$ 下尽量压低指数平均。谁分得开，谁对应的目标值就大。

最优函数本质上是对数密度比：

$$
f^*(x)=\log \frac{dP}{dQ}(x)+c
$$

这里 $\frac{dP}{dQ}$ 叫 Radon-Nikodym 导数，在连续密度场景下可以先理解成密度比 $\frac{p(x)}{q(x)}$；$c$ 是任意常数，因为目标函数对整体平移不敏感。

这件事的重要性在于：KL 散度从一个“需要知道分布密度”的量，变成了一个“可以通过优化函数估计”的量。MINE，也就是 Mutual Information Neural Estimation，正是把这个函数 $f$ 换成神经网络 $T_\theta$，用样本训练来估计互信息。

---

## 问题定义与边界

KL 散度定义为：

$$
D_{\mathrm{KL}}(P\Vert Q)=\mathbb{E}_P\left[\log \frac{dP}{dQ}\right]
$$

它衡量的是：如果真实分布是 $P$，但用 $Q$ 来编码或建模，会多付出多少信息代价。这里的单位如果使用自然对数，就是 nats。

KL 不是对称距离。通常：

$$
D_{\mathrm{KL}}(P\Vert Q)\neq D_{\mathrm{KL}}(Q\Vert P)
$$

所以 $D_{\mathrm{KL}}(P\Vert Q)$ 的方向不能随便换。

最重要的边界条件是 $P\ll Q$，读作“$P$ 绝对连续于 $Q$”。白话解释是：只要某个事件在 $Q$ 下概率为 0，它在 $P$ 下也必须概率为 0。否则 $P$ 会把概率放到 $Q$ 完全不支持的地方，此时 $D_{\mathrm{KL}}(P\Vert Q)=\infty$。

玩具例子：样本空间是 $\{0,1\}$。令

$$
P=(0.8,0.2),\quad Q=(0.5,0.5)
$$

两个分布都在 0 和 1 上有正概率，所以可以正常比较。若改成 $P(1)=0.2$，但 $Q(1)=0$，则 $P$ 在事件 $1$ 上有质量，而 $Q$ 不支持这个事件，KL 会变成无穷大。

| 条件 | 白话含义 | 不满足时的后果 |
|---|---|---|
| $P\ll Q$ | $P$ 不把概率放到 $Q$ 完全没有的位置 | $D_{\mathrm{KL}}(P\Vert Q)$ 可能为 $\infty$ |
| $\mathbb{E}_Q[e^f]<\infty$ | $f$ 在 $Q$ 下的指数平均必须有限 | 变分目标没有有限意义 |
| $\mathbb{E}_P[f]$ 有定义 | $f$ 在 $P$ 下的平均分能算 | 第一项无法稳定解释 |
| KL 有限 | 对数密度比在 $P$ 下可积 | 变分式可以作为有限优化目标 |

因此，DV 表示不是无条件公式。它默认 $P$、$Q$ 在同一个空间上，并且函数 $f$ 的指数矩满足基本可积性要求。

---

## 核心机制与推导

关键步骤是用 $f$ 构造一个新的分布：

$$
G_f(dx)=\frac{e^{f(x)}Q(dx)}{\mathbb{E}_Q[e^f]}
$$

$G_f$ 是把原分布 $Q$ 按照 $e^{f(x)}$ 重新加权后的分布。白话解释是：$f(x)$ 越大，$x$ 在新分布 $G_f$ 里的权重越高。

设

$$
Z_f=\mathbb{E}_Q[e^f]
$$

则

$$
G_f(dx)=\frac{e^{f(x)}}{Z_f}Q(dx)
$$

所以：

$$
\log \frac{dG_f}{dQ}(x)=f(x)-\log Z_f
$$

进一步得到：

$$
f(x)-\log \mathbb{E}_Q[e^f]=\log \frac{dG_f}{dQ}(x)
$$

对 $P$ 取期望：

$$
\mathbb{E}_P[f]-\log \mathbb{E}_Q[e^f]
=
\mathbb{E}_P\left[\log \frac{dG_f}{dQ}\right]
$$

把右侧拆成两个 KL：

$$
\mathbb{E}_P\left[\log \frac{dG_f}{dQ}\right]
=
\mathbb{E}_P\left[\log \frac{dP}{dQ}\right]
-
\mathbb{E}_P\left[\log \frac{dP}{dG_f}\right]
$$

也就是：

$$
\mathbb{E}_P[f]-\log \mathbb{E}_Q[e^f]
=
D_{\mathrm{KL}}(P\Vert Q)-D_{\mathrm{KL}}(P\Vert G_f)
$$

由于 KL 散度非负：

$$
D_{\mathrm{KL}}(P\Vert G_f)\ge 0
$$

所以任意 $f$ 都给出一个下界：

$$
\mathbb{E}_P[f]-\log \mathbb{E}_Q[e^f]\le D_{\mathrm{KL}}(P\Vert Q)
$$

取等号的条件是：

$$
D_{\mathrm{KL}}(P\Vert G_f)=0
$$

这等价于：

$$
G_f=P
$$

也就是：

$$
\frac{e^{f(x)}Q(dx)}{\mathbb{E}_Q[e^f]}=P(dx)
$$

整理得到：

$$
e^{f(x)}\propto \frac{dP}{dQ}(x)
$$

因此：

$$
f^*(x)=\log \frac{dP}{dQ}(x)+c
$$

这说明 DV 公式不是凭空构造的优化目标。它的机制是：每个 $f$ 都会把 $Q$ 扭成一个新分布 $G_f$；$f$ 越接近对数密度比，$G_f$ 越接近 $P$；当 $G_f=P$ 时，目标值刚好等于真实 KL。

继续看前面的玩具例子：

$$
P=(0.8,0.2),\quad Q=(0.5,0.5)
$$

若把状态顺序写成 $(1,0)$，最优打分可取：

$$
f^*(1)=\log\frac{0.8}{0.5}=\log 1.6
$$

$$
f^*(0)=\log\frac{0.2}{0.5}=\log 0.4
$$

于是：

$$
\mathbb{E}_P[f^*]=0.8\log 1.6+0.2\log 0.4
$$

并且：

$$
\mathbb{E}_Q[e^{f^*}]=0.5\cdot 1.6+0.5\cdot 0.4=1
$$

所以变分目标等于：

$$
0.8\log 1.6+0.2\log 0.4\approx 0.1927
$$

这和直接计算 $D_{\mathrm{KL}}(P\Vert Q)$ 完全一致。

---

## 代码实现

下面先用一个可运行的 Python 例子验证离散场景下的结论。代码里直接计算 KL，同时用最优 $f^*$ 计算 DV 目标。

```python
import math

P = [0.8, 0.2]
Q = [0.5, 0.5]

def kl(p, q):
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))

f_star = [math.log(pi / qi) for pi, qi in zip(P, Q)]

ep_f = sum(pi * fi for pi, fi in zip(P, f_star))
eq_exp_f = sum(qi * math.exp(fi) for qi, fi in zip(Q, f_star))
dv_value = ep_f - math.log(eq_exp_f)

true_kl = kl(P, Q)

assert abs(eq_exp_f - 1.0) < 1e-12
assert abs(dv_value - true_kl) < 1e-12
assert round(true_kl, 6) == 0.192745

print(true_kl)
```

真实工程例子是 MINE。互信息定义为：

$$
I(X;Y)=D_{\mathrm{KL}}(P_{XY}\Vert P_XP_Y)
$$

其中 $P_{XY}$ 是联合分布，表示真实配对的 $(x,y)$；$P_XP_Y$ 是边缘分布乘积，表示 $x$ 和 $y$ 独立采样后的配对。

把 DV 表示代入：

$$
I(X;Y)
=
\sup_T
\left\{
\mathbb{E}_{P_{XY}}[T(x,y)]
-
\log \mathbb{E}_{P_XP_Y}[e^{T(x,y)}]
\right\}
$$

MINE 的做法是用神经网络 $T_\theta(x,y)$ 近似最优函数 $T$。在训练时：

1. 取一批真实联合样本 `(x, y)`；
2. 计算 `T_theta(x, y)`，估计 $\mathbb{E}_{P_{XY}}[T]$；
3. 打乱 batch 内的 `y`，构造 `(x, y_shuffle)`；
4. 用打乱样本近似 $P_XP_Y$；
5. 最大化 `mean(T_joint) - logmeanexp(T_shuffle)`。

简化伪代码如下：

```python
# joint_batch: 来自真实数据的 (x, y)
# shuffled_batch: 保留 x，打乱 y 后得到的 (x, y_shuffle)

T_joint = T_theta(joint_batch)
T_shuffle = T_theta(shuffled_batch)

def logmeanexp(values):
    m = values.max()
    return m + log(mean(exp(values - m)))

mine_lower_bound = mean(T_joint) - logmeanexp(T_shuffle)
loss = -mine_lower_bound
loss.backward()
optimizer.step()
```

这里的 `logmeanexp` 是数值稳定版本。它避免直接计算 `exp(T_shuffle)` 时因为分数过大导致溢出。

如果用 PyTorch 实现，核心结构通常是：

```python
import torch

def logmeanexp(x, dim=0):
    m = torch.max(x, dim=dim, keepdim=True).values
    return (m + torch.log(torch.mean(torch.exp(x - m), dim=dim, keepdim=True))).squeeze(dim)

def mine_loss(model, x, y):
    batch_size = x.shape[0]
    perm = torch.randperm(batch_size, device=x.device)

    joint_batch = torch.cat([x, y], dim=-1)
    shuffled_batch = torch.cat([x, y[perm]], dim=-1)

    t_joint = model(joint_batch)
    t_shuffle = model(shuffled_batch)

    lower_bound = t_joint.mean() - logmeanexp(t_shuffle, dim=0)
    loss = -lower_bound
    return loss
```

工程上，`joint_batch` 近似 $P_{XY}$，`shuffled_batch` 近似 $P_XP_Y$。如果 `x` 和 `y` 是图像、文本向量或模型隐变量，`T_theta` 可以是 MLP、CNN 或 Transformer 后接一个标量输出头。

---

## 工程权衡与常见坑

DV 表示给出的是理论上成立的上确界，但工程训练只是在有限 batch、有限模型容量、有限优化步数下做近似。因此 MINE 的结果通常应被理解为互信息下界估计，而不是精确互信息。

| 常见坑 | 原因 | 规避方式 |
|---|---|---|
| batch 太小 | 负样本少，$P_XP_Y$ 近似质量差 | 增大 batch，跨 batch 维护负样本池 |
| 输出尺度过大 | $e^{T_\theta}$ 对大值极敏感 | 加权重衰减、梯度裁剪、限制输出范围 |
| $Q$ 构造不纯 | shuffle 后仍残留配对关系 | 检查数据采样方式，避免同源强相关样本进入负样本 |
| $\mathbb{E}_Q[e^{T_\theta}]$ 爆炸 | 指数项导致数值溢出 | 使用 logmeanexp，监控最大 logit |
| 经验估计有偏 | $\log$ 作用在 batch 均值外侧 | 用滑动平均或更大 batch 降低波动 |
| 训练目标上升但估计不可信 | 网络记住 batch 结构 | 使用验证集估计，检查打乱策略 |

batch 太小是最常见的问题。比如一个 batch 里只有 16 个样本，那么每个 $x$ 只看到 15 个错误配对的 $y$。如果真实数据维度很高，这些负样本覆盖不了 $P_XP_Y$ 的形状，模型学到的分界面会很局部，互信息估计值容易偏低并且抖动。

另一个问题是指数项。假设某个负样本的 $T_\theta$ 从 5 增加到 20，$e^{T_\theta}$ 会从约 148 增加到约 $4.85\times 10^8$。这会让 `log E_Q[e^T]` 被少数样本支配，训练曲线看起来突然崩掉。

真实工程中，如果用 MINE 估计一个视觉编码器的输入图像 $X$ 和 latent 表示 $Z$ 的互信息，通常不能只看训练集上的 MINE 值。更稳妥的做法是：固定编码器，在验证集上重新采样 joint 和 shuffled pair；同时观察估计值是否随 batch size 明显变化。如果 batch size 翻倍后估计值大幅漂移，说明当前估计还不稳定。

---

## 替代方案与适用边界

DV/MINE 不是估计 KL 或互信息的唯一方法。它的优势是可以处理高维、连续、显式密度不可得的场景；代价是优化不稳定、估计有偏、结果依赖模型容量和采样方式。

| 方法 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| 直接计算 | 低维离散变量，概率表可枚举 | 精确、简单、无训练 | 不适合高维连续变量 |
| Monte Carlo 密度比估计 | 已知或可评估 $p(x),q(x)$ | 实现直接，统计含义清楚 | 需要显式密度或密度比 |
| DV/MINE | 高维连续变量，只有样本，无显式密度 | 可微、可接入神经网络训练 | 方差大，数值不稳定 |
| InfoNCE | 对比学习、表示学习 | 通常比 DV 更稳定 | 受负样本数量限制，存在上界约束 |
| MMD 等分布距离 | 只需比较分布差异 | 稳定，避免指数爆炸 | 不是 KL，也不直接给互信息 |

低维离散变量时，直接枚举通常更好。例如两个二值变量一共只有 4 种联合状态，直接统计联合概率表和边缘概率表，就能精确计算：

$$
I(X;Y)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
$$

这时强行训练 MINE 反而增加误差来源。

连续高维表示学习中，MINE 更有价值。例如你想知道一个语音编码器的 latent 向量是否保留了说话人信息，但你没有显式密度模型。此时可以把真实的 `(latent, speaker_feature)` 当作联合样本，把打乱后的配对当作独立样本，用神经网络估计互信息下界。

如果任务只需要排序或阈值判断，粗估方法可能已经足够。比如你只想比较两个编码器哪个保留的信息更多，那么稳定的相对指标有时比不稳定的绝对互信息估计更实用。此时可以同时尝试 InfoNCE、分类探针和 MINE，而不是只依赖单一数值。

DV 变分表示的适用边界可以概括为：理论上它是 KL 的精确表示；工程上它是一个可优化但敏感的下界估计器。用它之前要先判断问题是否真的需要“可微的 KL 或互信息估计”。

---

## 参考资料

理论原文：

- Donsker, M. D. and Varadhan, S. R. S. 1975. *Asymptotic evaluation of certain Markov process expectations for large time, I*. Communications on Pure and Applied Mathematics. https://doi.org/10.1002/cpa.3160280102
- NYU Scholars 页面：https://nyuscholars.nyu.edu/en/publications/asymptotic-evaluation-of-certain-markov-process-expectations-for--4

互信息神经估计：

- Belghazi, M. I. et al. 2018. *Mutual Information Neural Estimation*. PMLR. https://proceedings.mlr.press/v80/belghazi18a.html
- Microsoft Research 论文页：https://www.microsoft.com/en-us/research/publication/mine-mutual-information-neural-estimation/

开源实现：

- mine-pytorch：https://github.com/gtegner/mine-pytorch

后续应用论文：

- 显式使用 Donsker-Varadhan 表示的后续论文示例：https://link.springer.com/article/10.1007/s10618-021-00759-3

如果要把这套方法真正用到模型里，建议先读 MINE 论文和实现，再回看 DV 变分表示，理解会更快。
