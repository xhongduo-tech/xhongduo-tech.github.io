## 核心结论

香农熵、互信息、KL 散度，分别回答三个不同问题：

1. 熵 $H(X)$：一个随机变量有多难预测。更精确地说，它是按真实分布加权后的平均信息量，也就是无损编码时的理论平均码长下限。
2. 互信息 $I(X;Y)$：知道 $Y$ 之后，$X$ 的不确定性减少了多少。它衡量的是两个变量共享了多少可用信息，而不是只看线性相关。
3. KL 散度 $D(P\|Q)$：如果真实分布是 $P$，却用近似分布 $Q$ 去描述，会多付出多少平均信息代价。它不是距离，而是“错误建模所带来的额外编码成本”。

三者的核心公式可以统一在“对数概率”上理解：

$$
H(X)=-\sum_x p(x)\log p(x)
$$

$$
I(X;Y)=H(X)-H(X|Y)
$$

$$
D(P\|Q)=\sum_x p(x)\log\frac{p(x)}{q(x)}
$$

如果对数底取 $\log_2$，单位就是比特 bit；如果取自然对数 $\ln$，单位就是 nat。工程里两种都能用，但整条链路必须统一，否则同一个量会因为单位不同而数值不一致。

最小玩具例子：抛一枚偏硬币，正面概率 $0.75$，反面概率 $0.25$。它的熵是：

$$
H(X)=-(0.75\log_2 0.75+0.25\log_2 0.25)\approx 0.81\ \text{bit}
$$

这说明：长期来看，要把每次结果无损编码下来，平均至少需要约 $0.81$ 比特，而不是 1 比特。原因不是“某次能用 0.81 个比特表示”，而是当样本足够多时，可以通过变长编码把整体平均码长压到接近这个下限。

在机器学习里，这三个量不是抽象名词，而是直接决定目标函数的工具：

| 概念 | 它衡量什么 | 典型用途 |
|---|---|---|
| 熵 | 单个变量的不确定性 | 决策树划分、策略探索、压缩极限 |
| 互信息 | 两个变量共享的信息 | 特征选择、表示学习、信道容量 |
| KL 散度 | 两个分布的差异成本 | VAE、蒸馏、RL 正则、分布拟合 |

还可以把三者再压缩成一句话：

| 问题 | 对应量 | 直观回答 |
|---|---|---|
| 这个变量本身有多乱？ | $H(X)$ | 平均要花多少信息才能描述它 |
| 另一个变量能帮我减少多少不确定性？ | $I(X;Y)$ | 看完 $Y$ 后，$X$ 还少猜多少 |
| 我拿错分布建模会损失多少？ | $D(P\|Q)$ | 比最优编码多花多少代价 |

---

## 问题定义与边界

本文只讨论离散随机变量，也就是取值是一个有限集合的变量，例如“类别标签是猫/狗”“动作是左/右/停”“硬币结果是正/反”。这样做的原因很直接：离散情形下，所有定义都能写成显式求和，初学者可以直接算清楚每一步，也更容易把公式和代码一一对应起来。

边界先说清楚：

- 不讨论连续变量的微分熵。
- 不讨论测度论定义。
- 不讨论高维估计器偏差，只讲离散、有限状态、有限概率表。
- 重点放在编码与信道视角，因为这是理解三者关系最直接的框架。

一个二分类工程问题可以作为统一背景。假设我们要判断邮件是否垃圾邮件，目标变量 $Y\in\{\text{正常},\text{垃圾}\}$，特征变量 $X$ 可以是“是否包含某关键词”“是否来自陌生域名”等离散特征。此时：

- 熵 $H(Y)$ 表示：如果什么特征都没看，只看总体分布，标签本身有多难猜。
- 条件熵 $H(Y|X)$ 表示：看完某个特征后，标签还剩多少不确定性。
- 互信息 $I(X;Y)$ 表示：这个特征到底提供了多少有效信息。
- KL 散度 $D(P\|Q)$ 表示：如果模型预测分布 $Q$ 偏离真实分布 $P$，会付出多少额外代价。

可以把这几个量先粗略对齐到一个表里：

| 指标 | 输入对象 | 目标 | 典型解释 | 适用边界 |
|---|---|---|---|---|
| $H(X)$ | 单个分布 $P_X$ | 测不确定性 | 平均编码下限 | 离散变量 |
| $H(X|Y)$ | 联合分布 $P_{XY}$ | 测条件下剩余不确定性 | 已知 $Y$ 后还要补多少信息 | 离散变量对 |
| $I(X;Y)$ | 联合分布 $P_{XY}$ | 测相关性 | 知道 $Y$ 后减少多少不确定性 | 离散变量对 |
| $D(P\|Q)$ | 两个分布 $P,Q$ | 测近似损失 | 用 $Q$ 表示 $P$ 多花多少代价 | 要求若 $p(x)>0$ 则最好有 $q(x)>0$ |

要注意，互信息和相关系数不是一回事。相关系数主要描述线性关系，而互信息描述的是一般依赖关系。一个经典例子是：若 $Y=X^2$，且 $X$ 关于 0 对称，那么相关系数可能接近 0，但互信息仍然大于 0，因为知道 $X$ 后仍然能显著缩小对 $Y$ 的判断范围。

再强调两个初学者最容易混淆的边界：

| 概念对比 | 是否相同 | 原因 |
|---|---|---|
| 高熵 vs 高价值信息 | 否 | 熵只描述不确定性，不评价信息有没有业务价值 |
| 互信息大 vs 因果关系强 | 否 | 互信息只说明依赖，不说明因果方向 |
| KL 小 vs 分布完全相同 | 否 | 只说明近似误差小，不代表逐点完全一致 |

---

## 核心机制与推导

### 1. 熵为什么是“平均信息量”

单个事件 $x$ 的信息量定义为：

$$
\text{info}(x)=-\log p(x)
$$

直觉是：越罕见的事，看到之后越“意外”，提供的信息越多。比如概率 $1/2$ 的事件信息量是 1 bit，概率 $1/4$ 的事件信息量是 2 bit。

这个定义不是拍脑袋定的，而是满足三个直觉要求后的自然结果：

1. 事件越罕见，信息量越大。
2. 独立事件的信息量应当可加。
3. 信息量应当连续变化，概率轻微变化时数值不应跳变。

满足这三个条件的函数，形式上就会落到对数上。

把单个事件的信息量按真实概率做平均，就得到熵：

$$
H(X)=\mathbb{E}[-\log p(X)] = -\sum_x p(x)\log p(x)
$$

这说明熵本质上是“平均惊讶程度”，也是“最优无损编码所需的平均信息量”。

再看一个更完整的离散例子。设三分类变量 $X\in\{A,B,C\}$，其分布为：

$$
P(X=A)=0.5,\quad P(X=B)=0.25,\quad P(X=C)=0.25
$$

则三个事件的信息量分别是：

$$
-\log_2 0.5 = 1,\quad -\log_2 0.25 = 2,\quad -\log_2 0.25 = 2
$$

平均后得到：

$$
H(X)=0.5\times 1 + 0.25\times 2 + 0.25\times 2 = 1.5\ \text{bit}
$$

这个结果比“均匀三分类”的 $\log_2 3 \approx 1.585$ bit 更低，因为这里的分布更偏，因而更容易压缩。

玩具例子还是偏硬币：

- 若 $P(\text{正})=0.75$，$P(\text{反})=0.25$
- 正面信息量是 $-\log_2 0.75\approx 0.415$
- 反面信息量是 $-\log_2 0.25=2$

平均以后：

$$
H(X)=0.75\cdot(-\log_2 0.75)+0.25\cdot 2\approx 0.81
$$

因为“正面”很常见，所以长期平均下来，不需要每次都花 1 bit 去描述它。

熵还有两个必须记住的性质：

| 性质 | 公式 | 含义 |
|---|---|---|
| 非负性 | $H(X)\ge 0$ | 不确定性不可能是负数 |
| 均匀时最大 | 若 $X$ 有 $n$ 个等概率取值，则 $H(X)=\log n$ | 越均匀越难猜 |
| 确定时为 0 | 若某个值概率为 1，则 $H(X)=0$ | 完全可预测时没有不确定性 |

如果把熵和编码联系起来，可以这样理解：熵不是某种“玄学复杂度”，而是长期平均意义下的压缩极限。哈夫曼编码、算术编码这些方法，本质上都在逼近这个极限。

### 2. 互信息为什么等于“不确定性的减少量”

条件熵 $H(X|Y)$ 表示在知道 $Y$ 后，$X$ 还剩多少不确定性。离散形式为：

$$
H(X|Y)=\sum_y p(y)H(X|Y=y)
= -\sum_{x,y} p(x,y)\log p(x|y)
$$

于是：

$$
I(X;Y)=H(X)-H(X|Y)
$$

这行公式的意思非常直接：原来要猜 $X$ 需要这么多信息，看了 $Y$ 以后少掉了一部分，少掉的那部分就是共享信息。

同一个量也可以写成对称形式：

$$
I(X;Y)=H(X)+H(Y)-H(X,Y)
$$

这说明互信息对 $X,Y$ 是对称的，因此：

$$
I(X;Y)=I(Y;X)
$$

如果展开到联合分布，会得到：

$$
I(X;Y)=\sum_{x,y} p(x,y)\log \frac{p(x,y)}{p(x)p(y)}
$$

这一步很关键，因为它把互信息写成了“联合分布”和“独立分布”的差异。分母 $p(x)p(y)$ 表示“如果两者独立，本该出现的概率”；分子 $p(x,y)$ 表示“真实一起出现的概率”。两者越不一样，互信息越大。

机制链路可以写成：

$$
H(X)\rightarrow H(X|Y)\rightarrow I(X;Y)=H(X)-H(X|Y)
$$

以及：

$$
I(X;Y)=D_{KL}(P_{XY}\|P_XP_Y)
$$

这说明互信息其实就是一个特殊的 KL 散度：它比较的是“真实联合分布”和“假设独立时的联合分布”。

这个结论还有一个直接推论：

$$
I(X;Y)\ge 0
$$

因为 KL 散度总是非负。并且只有在 $X$ 与 $Y$ 独立时，互信息才为 0。

下面给一个四格表例子，比“$Y=X$”更接近真实数据分析场景。假设：

| $X$ 是否含关键词 | $Y$ 是否垃圾邮件 | 概率 |
|---|---|---|
| 0 | 0 | 0.50 |
| 0 | 1 | 0.10 |
| 1 | 0 | 0.15 |
| 1 | 1 | 0.25 |

先算边缘分布：

$$
P(X=1)=0.40,\quad P(Y=1)=0.35
$$

于是标签熵为：

$$
H(Y)=-(0.65\log_2 0.65+0.35\log_2 0.35)\approx 0.934
$$

再算条件熵：

$$
H(Y|X)=P(X=0)H(Y|X=0)+P(X=1)H(Y|X=1)
$$

其中：

$$
P(Y=1|X=0)=\frac{0.10}{0.60}\approx 0.167,\quad
P(Y=1|X=1)=\frac{0.25}{0.40}=0.625
$$

代入可得：

$$
H(Y|X)\approx 0.801
$$

所以互信息约为：

$$
I(X;Y)=H(Y)-H(Y|X)\approx 0.934-0.801=0.133\ \text{bit}
$$

这个数不大，但含义明确：该关键词确实提供了信息，只是提供得不算多。它能帮助判断垃圾邮件，但远没有达到“看见就几乎能确定标签”的程度。

继续用硬币例子。若 $Y=X$，也就是接收端收到的是完全无噪声的拷贝，那么：

$$
H(X|Y)=0
$$

因为知道 $Y$ 就等于知道 $X$。于是：

$$
I(X;Y)=H(X)-0=H(X)\approx 0.81\ \text{bit}
$$

这就是“一个确定的反馈把信息全部传递”的严格表达。

### 3. KL 散度为什么表示“近似代价”

交叉熵 $H(P,Q)$ 表示：真实样本来自 $P$，但编码器按 $Q$ 的概率分配去编码，平均要花多少代价：

$$
H(P,Q)=-\sum_x p(x)\log q(x)
$$

而真实最优代价是：

$$
H(P)=-\sum_x p(x)\log p(x)
$$

两者相减：

$$
H(P,Q)-H(P)=\sum_x p(x)\log\frac{p(x)}{q(x)}=D(P\|Q)
$$

于是：

$$
D(P\|Q)=H(P,Q)-H(P)
$$

这行式子非常重要。它说明 KL 散度不是一个抽象“距离”，而是一个非常具体的量：你因为用了错误分布 $Q$，比理论最优编码平均多花了多少。

还是偏硬币，令真实分布 $P=(0.75,0.25)$，近似分布 $Q=(0.5,0.5)$，则：

$$
D(P\|Q)=0.75\log_2\frac{0.75}{0.5}+0.25\log_2\frac{0.25}{0.5}\approx 0.19\ \text{bit}
$$

含义是：如果你误以为这是一枚公平硬币，那么长期编码时每次平均多浪费约 0.19 bit。

这里有三个常见误解需要拆开：

| 误解 | 正确认识 |
|---|---|
| KL 是距离 | 不是。它不对称，也不满足三角不等式 |
| KL 越小越说明两个分布逐点相同 | 不是。只说明平均代价小 |
| 可以随意交换 $P,Q$ | 不行。$D(P\|Q)$ 与 $D(Q\|P)$ 通常不同 |

看一个简单反例。设：

$$
P=(0.99,0.01),\quad Q=(0.90,0.10)
$$

则 $D(P\|Q)$ 关注的是：“真实很少发生的第二类事件，在 $Q$ 里被高估了多少”；而 $D(Q\|P)$ 关注的是：“$Q$ 以为第二类事件挺常见，但真实里很少见，这种乐观会付出什么代价”。方向不同，惩罚重点也不同。

KL 的非负性也值得明确写出：

$$
D(P\|Q)\ge 0
$$

且当且仅当 $P=Q$（在 $P$ 的支持集上几乎处处相等）时取 0。

另一个工程上非常关键的边界是支持集问题。若存在某个 $x$ 满足：

$$
p(x)>0,\quad q(x)=0
$$

则

$$
D(P\|Q)=\infty
$$

因为真实会出现的事件，在近似分布中被赋予了零概率。编码视角下，这等价于“编码器根本没有给这个事件留码字”。这也是为什么实际实现里通常要做平滑或加 `eps`。

### 4. 信道容量如何连到互信息

信道容量是信道在最优输入分布下能传递的最大信息率：

$$
C=\max_{P_X} I(X;Y)
$$

白话说，不仅信道本身重要，输入怎么发也重要。同一条信道，不同输入分布得到的互信息不同。容量就是把输入分布调到最好时的上限。

以二元对称信道（Binary Symmetric Channel, BSC）为例。设输入比特有概率 $\varepsilon$ 被翻转，则：

$$
P(Y\neq X)=\varepsilon,\quad P(Y=X)=1-\varepsilon
$$

这条信道的容量为：

$$
C=1-H_2(\varepsilon)
$$

其中 $H_2(\varepsilon)$ 是二元熵函数：

$$
H_2(\varepsilon)=-\varepsilon\log_2\varepsilon-(1-\varepsilon)\log_2(1-\varepsilon)
$$

几个典型点：

| 翻转概率 $\varepsilon$ | 容量 $C$ | 含义 |
|---|---|---|
| 0 | 1 bit | 完全无噪声，每次传 1 bit 有效信息 |
| 0.1 | $1-H_2(0.1)\approx 0.531$ bit | 有噪声，但仍有可观传输能力 |
| 0.5 | 0 bit | 输出与输入独立，完全无法传信息 |

这也是为什么“信息论”会同时出现在压缩、通信、机器学习里：它们都在问同一个问题，信息如何表示、传递和逼近。

---

## 代码实现

下面给出一个可运行的 Python 实现。它做四件事：

- `entropy(dist)` 计算离散分布熵
- `cross_entropy(p, q)` 计算交叉熵
- `kl_divergence(p, q)` 计算 KL 散度
- `mutual_info(joint)` 从联合分布表计算互信息

代码只依赖 Python 标准库，可以直接运行：

```python
from __future__ import annotations

import math
from typing import Dict, Hashable, Mapping, Tuple

State = Hashable
Dist = Mapping[State, float]
JointDist = Mapping[Tuple[State, State], float]


def normalize(dist: Dist) -> Dict[State, float]:
    total = float(sum(dist.values()))
    if total <= 0:
        raise ValueError("distribution sum must be positive")
    return {k: float(v) / total for k, v in dist.items()}


def validate_nonnegative(dist: Dist, name: str) -> None:
    for k, v in dist.items():
        if v < 0:
            raise ValueError(f"{name}[{k!r}] must be non-negative, got {v}")


def get_log(base: float):
    if base == math.e:
        return math.log
    if base == 2:
        return math.log2
    if base <= 0 or base == 1:
        raise ValueError("log base must be positive and not equal to 1")
    return lambda x: math.log(x, base)


def entropy(dist: Dist, base: float = 2) -> float:
    validate_nonnegative(dist, "dist")
    probs = normalize(dist)
    log_fn = get_log(base)

    h = 0.0
    for p in probs.values():
        if p > 0.0:
            h -= p * log_fn(p)
    return h


def cross_entropy(p: Dist, q: Dist, base: float = 2, eps: float = 1e-12) -> float:
    validate_nonnegative(p, "p")
    validate_nonnegative(q, "q")
    p_norm = normalize(p)
    q_norm = normalize(q)
    log_fn = get_log(base)

    value = 0.0
    for state, pk in p_norm.items():
        if pk == 0.0:
            continue
        qk = q_norm.get(state, 0.0)
        qk = max(qk, eps)  # 工程实现里防止 log(0)
        value -= pk * log_fn(qk)
    return value


def kl_divergence(p: Dist, q: Dist, base: float = 2, eps: float = 1e-12) -> float:
    return cross_entropy(p, q, base=base, eps=eps) - entropy(p, base=base)


def mutual_info(joint: JointDist, base: float = 2, eps: float = 1e-12) -> float:
    validate_nonnegative(joint, "joint")
    total = float(sum(joint.values()))
    if total <= 0:
        raise ValueError("joint distribution sum must be positive")

    pxy = {(x, y): float(v) / total for (x, y), v in joint.items()}

    px: Dict[State, float] = {}
    py: Dict[State, float] = {}
    for (x, y), p in pxy.items():
        px[x] = px.get(x, 0.0) + p
        py[y] = py.get(y, 0.0) + p

    log_fn = get_log(base)
    mi = 0.0
    for (x, y), p in pxy.items():
        if p == 0.0:
            continue
        denom = max(px[x] * py[y], eps)
        mi += p * log_fn(p / denom)
    return mi


def conditional_entropy(joint: JointDist, base: float = 2, eps: float = 1e-12) -> float:
    validate_nonnegative(joint, "joint")
    total = float(sum(joint.values()))
    if total <= 0:
        raise ValueError("joint distribution sum must be positive")

    pxy = {(x, y): float(v) / total for (x, y), v in joint.items()}
    py: Dict[State, float] = {}
    for (_, y), p in pxy.items():
        py[y] = py.get(y, 0.0) + p

    log_fn = get_log(base)
    value = 0.0
    for (x, y), p in pxy.items():
        if p == 0.0:
            continue
        p_x_given_y = p / max(py[y], eps)
        value -= p * log_fn(p_x_given_y)
    return value


def main() -> None:
    # 例 1：偏硬币
    p = {"H": 0.75, "T": 0.25}
    q = {"H": 0.50, "T": 0.50}

    h = entropy(p)
    ce = cross_entropy(p, q)
    dkl = kl_divergence(p, q)

    # 例 2：Y = X 的联合分布
    joint_same = {
        ("H", "H"): 0.75,
        ("H", "T"): 0.0,
        ("T", "H"): 0.0,
        ("T", "T"): 0.25,
    }
    mi_same = mutual_info(joint_same)
    hx_given_y = conditional_entropy(joint_same)

    print("biased coin")
    print("H(X)      =", round(h, 6))
    print("H(P, Q)   =", round(ce, 6))
    print("D(P||Q)   =", round(dkl, 6))
    print("H(X|Y)    =", round(hx_given_y, 6))
    print("I(X;Y)    =", round(mi_same, 6))

    # 例 3：邮件关键词与垃圾标签的联合分布
    joint_mail = {
        (0, 0): 0.50,
        (0, 1): 0.10,
        (1, 0): 0.15,
        (1, 1): 0.25,
    }
    mi_mail = mutual_info(joint_mail)
    print("\nmail example")
    print("I(X;Y)    =", round(mi_mail, 6))

    # 基本校验
    assert abs(h - 0.811278) < 1e-6
    assert abs(ce - 1.0) < 1e-9
    assert abs(dkl - 0.188722) < 1e-6
    assert abs(hx_given_y - 0.0) < 1e-9
    assert abs(mi_same - h) < 1e-9
    assert mi_mail > 0.0


if __name__ == "__main__":
    main()
```

这段代码可以直接保存为 `info_theory_demo.py` 后运行。预期输出的关键数值是：

```text
biased coin
H(X)      = 0.811278
H(P, Q)   = 1.0
D(P||Q)   = 0.188722
H(X|Y)    = 0.0
I(X;Y)    = 0.811278

mail example
I(X;Y)    = 0.132844
```

代码里有三个设计点值得说明：

| 设计点 | 原因 |
|---|---|
| 先 `validate_nonnegative` 再 `normalize` | 避免负概率输入被静默吞掉 |
| `cross_entropy` 单独实现 | 更清楚地展示 $H(P,Q)=H(P)+D(P\|Q)$ |
| 对分母加 `eps` | 工程里避免零概率导致数值爆炸，但理论分析时要知道这只是近似处理 |

如果想把这个实现用于更接近工程的输入，可以把联合分布直接理解为计数表。例如：

- 决策树里，统计“特征是否出现”和“标签是否为垃圾邮件”的四格计数表，再归一化后算互信息。
- A/B 实验里，统计“是否曝光某策略”和“是否点击”的联合分布，评估一个策略信号到底提供了多少可用信息。
- 推荐系统里，统计“用户是否看过某类内容”和“是否发生点击/转化”，用互信息筛选有区分度的离散特征。

一个真实工程例子是决策树。假设目标标签是“是否坏账”，候选特征是“是否逾期超过 30 天”。如果这个特征让标签熵从 $H(Y)$ 降到 $H(Y|X)$，则信息增益就是：

$$
\text{Gain}(Y,X)=H(Y)-H(Y|X)=I(X;Y)
$$

因此，决策树优先选择互信息高的特征做分裂，因为它们能最快降低标签不确定性。

---

## 工程权衡与常见坑

信息论公式很简洁，但真正落到代码和训练目标时，坑主要集中在数值稳定性、估计偏差和目标权重三类问题。

| 常见坑 | 具体表现 | 风险 | 规避方式 |
|---|---|---|---|
| $\log 0$ | 直接对 0 取对数 | `nan` 或 `inf` | 对 $p=0$ 项跳过；对 $q$ 加 `eps` |
| 混用单位 | 一部分用 `log_2`，另一部分用 `\ln` | 数值量纲不一致 | 全流程统一 bit 或 nat |
| 忘记归一化 | 概率和不为 1 | 结果失真 | 输入前先 normalize |
| 把 KL 当距离 | 误以为对称 | 推理错误 | 明确 $D(P\|Q)\neq D(Q\|P)$ |
| 正则过强 | KL 权重过大 | 模型退化到先验 | 把权重当超参数调 |
| 稀疏计数过少 | 联合分布很多格子为 0 | 估计噪声大 | 平滑、合并桶、增加样本 |
| 直接比较不同粒度的离散化 | 分桶方式不同 | 结论不可比 | 固定分桶规则后再比较 |
| 用经验分布代替真实分布却忽略样本量 | 小样本下数值波动大 | 过拟合或误判特征价值 | 报告样本量，并做平滑或交叉验证 |

其中最容易被初学者忽视的是：KL 散度不是对称的。$D(P\|Q)$ 和 $D(Q\|P)$ 衡量的是完全不同的问题，因为“用谁去近似谁”不同。真实分布和近似分布的位置不能随意交换。

真实工程例子可以看强化学习。很多方法会在奖励外加一个 KL 正则项，例如：

$$
\mathbb{E}[R]-\alpha D_{KL}(\pi\|\pi_0)
$$

这里 $\pi$ 是当前策略，$\pi_0$ 是参考策略，$\alpha$ 是正则强度。这样做的目的是限制策略更新不要跳太远，避免训练不稳定。

但如果 $\alpha$ 过大，会出现一个直接后果：策略几乎只会贴着 $\pi_0$ 走，探索被压扁，学习不到真正更优的动作分布。也就是说，KL 正则本来是“防抖器”，权重过大就会变成“刹车锁死”。

VAE 里也有类似问题。ELBO 中的 KL 项：

$$
D_{KL}(q_\phi(z|x)\|p(z))
$$

用于约束潜变量后验接近先验。如果这项权重过强，潜空间会塌缩，模型忽略输入；如果过弱，采样质量又会变差。这不是公式错，而是工程上必须平衡重建项和正则项。

再补一个分类任务里的常见误区。监督学习中常写交叉熵损失：

$$
\mathcal{L}=-\sum_x p(x)\log q(x)
$$

如果标签是 one-hot，这个式子看起来像“只在真实类上取对数”；但本质上仍然是用模型分布 $q$ 去逼近真实分布 $p$。因此训练时最小化交叉熵，本质上是在最小化：

$$
H(P,Q)=H(P)+D(P\|Q)
$$

由于数据分布 $P$ 给定时 $H(P)$ 是常数，所以优化重点落在 KL 上。

---

## 替代方案与适用边界

KL、互信息、熵很常用，但不是所有场景都该直接套它们。

第一类替代是 JS 散度。它是 Jensen-Shannon divergence 的缩写，可以理解为“对称化且更稳定的分布差异指标”。定义为：

$$
JS(P,Q)=\frac{1}{2}D_{KL}(P\|M)+\frac{1}{2}D_{KL}(Q\|M),
\quad M=\frac{1}{2}(P+Q)
$$

它的特点是：

- 对称；
- 总是有限；
- 更适合把两个分布放在平等地位上比较。

第二类替代是交叉熵。交叉熵不是 KL 的替身，而是包含了熵和 KL 的总编码成本：

$$
H(P,Q)=H(P)+D(P\|Q)
$$

所以在监督学习里，最小化交叉熵本质上等价于在固定真实分布熵的情况下最小化 KL。这也是分类模型普遍用交叉熵损失的原因。

第三类边界是连续变量或高维潜空间。此时直接显式枚举分布几乎不可行，常见做法是通过变分技巧把 KL 放进 ELBO，用采样和重参数化近似优化，而不是手工写概率表。

第四类边界是互信息估计。互信息在理论上很干净，但在高维连续空间上直接估计通常很难，原因包括：

- 联合分布难估计；
- 采样方差大；
- 离散化方式会强烈影响结果。

因此工程里常见的是代理目标，而不是直接求真实互信息，例如对比学习损失、变分下界、判别器估计。

对比如下：

| 度量 | 是否对称 | 核心作用 | 适用边界 |
|---|---|---|---|
| KL 散度 | 否 | 衡量用 $Q$ 近似 $P$ 的代价 | 概率建模、VAE、RL 正则 |
| JS 散度 | 是 | 稳定比较两个分布 | 生成模型、分布相似性分析 |
| 交叉熵 | 否 | 真实编码成本 + 近似误差 | 分类训练、语言模型训练 |
| 熵 | 不适用 | 衡量单分布不确定性 | 压缩、探索、特征纯度分析 |
| 互信息 | 是 | 衡量变量间共享信息 | 特征选择、通信、表示学习 |

还要再强调一个适用边界：互信息在理论上很好，但高维连续变量上通常很难直接估计。工程里常常不直接求真实互信息，而是用对比学习损失、变分下界、判别器近似等办法替代。也就是说，概念本身是底层原则，实际优化常常靠代理目标。

如果只想记一个选择规则，可以用下面这张表：

| 你的问题 | 优先看什么 |
|---|---|
| “这个变量本身乱不乱？” | 熵 |
| “这个特征对标签有没有帮助？” | 互信息 |
| “模型分布和真实分布差多少代价？” | KL 散度 |
| “两个分布要对称比较，且不想数值太脆弱？” | JS 散度 |
| “我要训练分类模型，直接用什么损失？” | 交叉熵 |

---

## 参考资料

- Shannon, C. E. “A Mathematical Theory of Communication.” *Bell System Technical Journal*, 27(3), 379-423, 1948; 27(4), 623-656, 1948.
- Cover, T. M., and Thomas, J. A. *Elements of Information Theory*. 2nd ed., Wiley, 2006.
- MacKay, D. J. C. *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press, 2003.
- Murphy, K. P. *Probabilistic Machine Learning: An Introduction*. MIT Press, 2022.
- Bishop, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006.
- Goodfellow, I., Bengio, Y., and Courville, A. *Deep Learning*. MIT Press, 2016.
- 信息论教材中关于熵、条件熵、互信息、信道容量的标准章节，可作为公式推导的主参考。
- 概率建模与深度生成模型资料中关于交叉熵、KL 散度、ELBO 的章节，可用于理解机器学习中的目标函数联系。
- 决策树、表示学习、强化学习文献中关于信息增益、KL 正则与策略约束的章节，可用于理解工程用法。
- 若需要在线检索，优先查阅教材、课程讲义和论文原文，而不是只看二手摘要。
