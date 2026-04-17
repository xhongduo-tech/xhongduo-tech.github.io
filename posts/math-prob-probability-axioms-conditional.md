## 核心结论

概率论先定义“什么叫合法的概率”，再讨论“在已知信息下如何更新概率”。这两步分别对应 Kolmogorov 三公理与条件概率。

Kolmogorov 三公理把概率定义为概率空间 $(\Omega,\mathcal{F},P)$ 上的一个测度。测度可以白话理解为“给每个可讨论事件分配一份合法份额”的规则。三个公理是：

$$
P(A)\ge 0
$$

$$
P(\Omega)=1
$$

对于两两互斥的事件 $A_1,A_2,\dots$，

$$
P\left(\bigcup_{i=1}^{\infty}A_i\right)=\sum_{i=1}^{\infty}P(A_i)
$$

它们保证三件事：概率不会是负数，总份额固定为 1，互不重叠的路径可以直接相加。

条件概率则回答“在已知 $B$ 已经发生的前提下，$A$ 还剩多少概率”：

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}, \quad P(B)>0
$$

这里的“条件”可以白话理解为“把原来的世界缩小到只剩下 $B$ 成立的部分，再看 $A$ 占多少”。

一个最小玩具例子是公平六面骰。设 $A=\{6\}$，$B=\{2,4,6\}$，即“结果是偶数”。因为

$$
P(A\cap B)=P(\{6\})=\frac16,\quad P(B)=\frac12
$$

所以

$$
P(6\mid B)=\frac{1/6}{1/2}=\frac13
$$

这同时验证了乘法公式：

$$
P(A\cap B)=P(A\mid B)P(B)=\frac13\cdot\frac12=\frac16
$$

这套公式不是只给骰子和抽球用。贝叶斯公式、全概率公式、VAE 的 ELBO、扩散模型中的后验推导，都是从这里长出来的。

| 公理/公式 | 数学形式 | 直觉含义 |
|---|---|---|
| 非负性 | $P(A)\ge 0$ | 份额不能是负数 |
| 规范性 | $P(\Omega)=1$ | 全部可能性加起来就是 1 |
| 可列可加性 | $P(\cup_i A_i)=\sum_i P(A_i)$ | 互斥路径可以叠加 |
| 条件概率 | $P(A\mid B)=P(A\cap B)/P(B)$ | 已知 $B$ 后重新计算份额 |

---

## 问题定义与边界

严格写法里，概率不是“凭感觉给一个数”，而是在概率空间 $(\Omega,\mathcal{F},P)$ 上定义的。

- $\Omega$ 是样本空间，白话解释是“所有可能结果的全集”。
- $\mathcal{F}$ 是事件集合，通常叫 $\sigma$-代数，白话解释是“哪些结果集合允许被拿来谈概率”。
- $P$ 是概率测度，白话解释是“给这些事件分配合法概率的规则”。

例如掷一次骰子时：

- $\Omega=\{1,2,3,4,5,6\}$
- 事件“偶数”可写为 $\{2,4,6\}$
- 事件“点数大于 4”可写为 $\{5,6\}$

如果是离散且有限的情况，初学者可以把概率理解成“每个结果在全空间中占的份额”。只要满足两件事，它才是合法概率：

1. 每个份额在 $[0,1]$ 内。
2. 所有基本结果的份额加起来等于 1。

这也是三公理在有限情形下最直接的直觉版本。

条件概率有明确边界：只有当 $P(B)>0$ 时，$P(A\mid B)$ 才有定义。原因很简单，公式分母不能为 0：

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}
$$

如果 $P(B)=0$，就不能直接说“在 $B$ 发生的条件下”。因为 $B$ 在原概率模型里没有可分配的正份额，不能直接把它当成一个新的归一化空间。

| 事件 | 描述 | 概率条件（需满足） |
|---|---|---|
| $\Omega$ | 全部可能结果 | $P(\Omega)=1$ |
| $A$ | 任意可讨论事件 | $P(A)\ge 0$ |
| $A\cup B$ | 两个事件至少一个发生 | 若互斥，则 $P(A\cup B)=P(A)+P(B)$ |
| $A\mid B$ | 在 $B$ 已知成立时看 $A$ | 必须满足 $P(B)>0$ |

这里还要说明一个常见误区：条件概率不是“给原概率打补丁”，而是“在新的有效样本空间里重新归一化”。已知 $B$ 成立后，原来在 $B^c$ 中的那些概率全部被丢弃，只保留 $B$ 内部，再把总量重新缩放成 1。

---

## 核心机制与推导

条件概率最重要的作用，是把复杂联合事件拆成一步一步的局部条件。

从定义直接变形：

$$
P(A\cap B)=P(A\mid B)P(B)
$$

也可以写成：

$$
P(A\cap B)=P(B\mid A)P(A)
$$

这就是乘法公式的两事件版本。继续推广到多个事件：

$$
P(A_1\cap A_2\cap \cdots \cap A_n)
=
P(A_1)\,P(A_2\mid A_1)\,P(A_3\mid A_1\cap A_2)\cdots P(A_n\mid A_1\cap\cdots\cap A_{n-1})
$$

更紧凑地写：

$$
P\left(\bigcap_{i=1}^n A_i\right)
=
\prod_{i=1}^n P\left(A_i \mid \bigcap_{j=1}^{i-1}A_j\right)
$$

这个公式的意义是：联合概率不是一次性硬算，而是沿着“先发生什么，再在此前提下发生什么”的链条展开。

还是用骰子例子验证一次。设：

- $A=\{6\}$
- $B=\{2,4,6\}$

则 $A\subseteq B$，所以

$$
P(A\cap B)=P(A)=\frac16
$$

而条件概率给出：

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}=\frac{1/6}{1/2}=\frac13
$$

再代回乘法公式：

$$
P(A\cap B)=P(A\mid B)P(B)=\frac13\cdot\frac12=\frac16
$$

前向与后向是一致的。这就是概率推理链条自洽的意思。

接下来就是贝叶斯公式。贝叶斯公式是条件概率的直接变形，不是新理论。由

$$
P(\theta\mid x)=\frac{P(\theta\cap x)}{P(x)},\quad
P(x\mid \theta)=\frac{P(\theta\cap x)}{P(\theta)}
$$

可得

$$
P(\theta\mid x)=\frac{P(x\mid \theta)P(\theta)}{P(x)}
$$

其中：

- 先验 $P(\theta)$：观测前对参数或假设的相信程度。
- 似然 $P(x\mid \theta)$：假设 $\theta$ 成立时，观测 $x$ 出现的可能性。
- 后验 $P(\theta\mid x)$：看到数据后更新得到的概率。
- 证据 $P(x)$：归一化常数，保证后验加起来等于 1。

常写成比例形式：

$$
P(\theta\mid x)\propto P(x\mid \theta)P(\theta)
$$

但这个“$\propto$”只表示“差一个归一化常数”，不表示可以永远不算分母。真正的分母来自全概率公式：

$$
P(x)=\sum_i P(x\mid \theta_i)P(\theta_i)
$$

连续情形则变成积分：

$$
p(x)=\int p(x\mid z)p(z)\,dz
$$

这一步叫边缘化。边缘化可以白话理解为“把隐藏变量的所有可能性都加总掉，只保留观测变量的总概率”。

真实工程里，生成模型几乎都在做这件事。比如 VAE 假设联合分布

$$
p_\theta(x,z)=p_\theta(x\mid z)p(z)
$$

那么数据概率就是

$$
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz
$$

这是一个典型的全概率积分版。扩散模型里也一样，会围绕条件分布和后验分布做推导，例如解析形式的

$$
q(x_{t-1}\mid x_t,x_0)
$$

本质上仍然是“已知部分变量后，另一个变量的条件分布是什么”。

---

## 代码实现

下面先写一个离散事件版的条件概率函数。为了让初学者能直接运行，事件用 Python 的 `set` 表示，基本结果到概率的映射用字典表示。

```python
from typing import Dict, Set, Hashable

Outcome = Hashable

def prob(event: Set[Outcome], probs: Dict[Outcome, float]) -> float:
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-9, f"probabilities must sum to 1, got {total}"
    for x, p in probs.items():
        assert p >= 0.0, f"negative probability for {x}: {p}"
    return sum(probs[x] for x in event)

def conditional_prob(event: Set[Outcome], condition: Set[Outcome], probs: Dict[Outcome, float]) -> float:
    p_condition = prob(condition, probs)
    if p_condition == 0:
        raise ValueError("P(B)=0, conditional probability is undefined")
    return prob(event & condition, probs) / p_condition

# 玩具例子：公平六面骰
dice_probs = {i: 1/6 for i in range(1, 7)}
A = {6}
B = {2, 4, 6}

p_A_given_B = conditional_prob(A, B, dice_probs)
p_A_and_B = prob(A & B, dice_probs)

print("P(6|even) =", p_A_given_B)
print("P(6∩even) =", p_A_and_B)

assert abs(p_A_given_B - 1/3) < 1e-9
assert abs(p_A_and_B - 1/6) < 1e-9
assert abs(p_A_given_B * prob(B, dice_probs) - p_A_and_B) < 1e-9

# 常见坑：零概率条件
try:
    conditional_prob({1}, set(), dice_probs)
    assert False, "should have raised ValueError"
except ValueError:
    pass
```

这个实现刻意做了两件工程上必须做的检查：

1. 概率表必须归一化，总和为 1。
2. 条件事件的概率不能为 0。

再看一个离散贝叶斯更新的最小实现。假设有两个模型假设 $\theta_1,\theta_2$，看见观测 $x$ 后更新后验。

```python
def bayes_update(prior, likelihood):
    assert len(prior) == len(likelihood)
    assert all(p >= 0 for p in prior)
    assert abs(sum(prior) - 1.0) < 1e-9

    unnormalized = [p * l for p, l in zip(prior, likelihood)]
    evidence = sum(unnormalized)
    if evidence == 0:
        raise ValueError("evidence is zero, cannot normalize posterior")
    posterior = [u / evidence for u in unnormalized]
    return posterior, evidence

# 真实工程味道的简化例子：
# theta0: 邮件正常
# theta1: 邮件垃圾
# x: 观测到“包含大量营销关键词”
prior = [0.8, 0.2]
likelihood = [0.1, 0.7]

posterior, evidence = bayes_update(prior, likelihood)

print("posterior =", posterior)
print("evidence =", evidence)

assert abs(sum(posterior) - 1.0) < 1e-9
assert posterior[1] > prior[1]  # 看到该特征后，“垃圾邮件”后验上升
```

这就是“先验 × 似然，再归一化”的直接实现。很多工程系统不会显式写成贝叶斯公式，但内部逻辑就是这一套。

| 输入项 | 示例 | 作用 |
|---|---|---|
| `prior` | `[0.8, 0.2]` | 观测前各假设的概率 |
| `likelihood` | `[0.1, 0.7]` | 给定假设时观测出现的概率 |
| `unnormalized` | `[0.08, 0.14]` | 未归一化后验 |
| `evidence` | `0.22` | 归一化常数 |
| `posterior` | `[0.3636, 0.6364]` | 观测后的合法概率分布 |

---

## 工程权衡与常见坑

概率公式在纸上很短，在工程里出错往往是边界没处理清楚。

第一个坑是对零概率事件直接套条件概率公式。理论上 $P(B)=0$ 时 $P(A\mid B)$ 不可直接这样定义；工程上更直接，分母为 0 会让程序崩掉，或者更糟，返回 `nan` 后继续传播。

第二个坑是把全概率公式用在不完备或不互斥的划分上。全概率要求 $\{B_i\}$ 构成一个分割，也就是“互斥且完备”。如果少了一块，求和会偏小；如果重叠了，求和会重复计算。

第三个坑是把“先验 × 似然”误当成后验。它只是未归一化后验。只要不除以证据项，最终结果就不是合法概率分布。

第四个坑是把经验频率和理论概率混用。频率可以白话理解为“实验中观察到的比例”，概率是模型层面的分配规则。样本很小时，频率抖动很大，不能机械当成真实概率。

下面是常见错误与规避方式。

| 坑 | 典型后果 | 规避策略 |
|---|---|---|
| $P(B)=0$ 仍计算 $P(A\mid B)$ | 除零、`nan`、逻辑失真 | 先检查 `P(B) > 0`，否则报错或改建模 |
| 全概率分解不是互斥完备分割 | 和不等于真实 $P(x)$ | 明确验证各分块覆盖全集且无重叠 |
| 忽略归一化常数 | 后验不和为 1 | 显式计算证据并做归一化 |
| 把频率直接当概率 | 小样本误判 | 加入先验、置信区间或更多数据 |
| 默认独立 | 联合概率估计偏差大 | 明确写出条件项，不随意拆乘 |

最小防御式伪代码如下：

```python
if P_B <= 0:
    raise ValueError("conditional probability undefined")
```

真实工程例子可以看垃圾邮件过滤、CTR 预估、故障诊断。比如某监控系统根据多个告警信号估计“机器是否即将故障”。如果直接把多个信号概率相乘，默认它们独立，通常会高估或低估风险。正确做法是明确条件结构，至少要问：这些特征是否共享同一原因，是否需要条件化建模。

---

## 替代方案与适用边界

解析公式很重要，但并不代表任何问题都能手算出闭式答案。闭式答案可以白话理解为“能直接写成明确公式的结果”。一旦模型复杂、变量连续、维度高，解析条件概率往往算不出来，这时就需要替代方案。

第一类替代方案是频率估计。比如抛硬币 1000 次，用正面出现比例近似 $P(\text{正面})$。它适合样本足够大、问题结构简单的情况，但对稀有事件和小样本不稳定。

第二类替代方案是贝叶斯更新。即使分布未知，也可以先给一个先验，再根据新数据逐步更新：

$$
posterior \propto prior \times likelihood
$$

这类方法适合要持续吸收新证据的场景，比如在线推荐、风控、故障告警。

第三类替代方案是变分推断。变分推断可以白话理解为“用一个容易算的分布去逼近真实但难算的后验”。VAE 就是这个思路的标准例子。它不直接求难算的后验 $p_\theta(z\mid x)$，而是引入近似分布 $q_\phi(z\mid x)$，优化 ELBO：

$$
\log p_\theta(x)\ge
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-
\mathrm{KL}(q_\phi(z\mid x)\|p(z))
$$

这里：

- 重构项 $\mathbb{E}_{q}[\log p_\theta(x\mid z)]$ 要求生成出的 $x$ 尽量像真实数据。
- KL 项衡量两个分布差异，白话解释是“别让近似后验偏离先验太远”。

这就是一个新手也应该知道的真实工程例子。VAE 不是绕开概率，而是把贝叶斯后验拟合问题转成可优化的目标函数。

扩散模型同样如此。正向过程把数据一步步加噪，逆向过程学习去噪。中间一个关键对象是：

$$
q(x_{t-1}\mid x_t,x_0)
$$

它表示“已知当前更噪的状态 $x_t$ 和原始样本 $x_0$ 时，上一步 $x_{t-1}$ 的后验分布”。很多推导之所以能成立，就是因为高斯分布族在条件化后仍可解析处理。

因此，适用边界可以总结为：

| 方法 | 适用场景 | 边界 |
|---|---|---|
| 概率公理 + 条件概率 | 定义清楚、事件可精确表达 | 需要明确定义事件与分布 |
| 频率估计 | 样本多、结构简单 | 小样本和稀有事件不稳 |
| 贝叶斯更新 | 持续接收证据、需要不确定性表达 | 复杂后验常难解析 |
| 变分推断/VAE | 高维潜变量生成模型 | 得到的是近似后验，不是精确后验 |
| 扩散模型后验拟合 | 连续高维生成任务 | 依赖特定噪声结构与训练近似 |

所以最稳妥的理解顺序是：先掌握概率公理和条件概率，再理解贝叶斯与全概率，最后再进入 VAE、扩散、ELBO。后面的模型很复杂，但底层语法没有变。

---

## 参考资料

1. Wikipedia, “Probability axioms”. 用于查看 Kolmogorov 三公理与概率空间的标准定义。
2. Wikipedia, “Bayes' theorem”. 用于查看贝叶斯公式、先验、似然、后验、证据的标准表达。
3. Wikipedia, “Law of total probability”. 用于查看全概率公式及其适用前提。
4. Kingma, D. P. and Welling, M., “Auto-Encoding Variational Bayes”. VAE 的原始论文，ELBO 推导从联合分布与近似后验出发。
5. Angus Turner, “Denoising Diffusion Probabilistic Models” 相关笔记。用于理解扩散模型中的条件高斯与后验形式。
6. 概率论教材或测度论入门材料。若要真正理解 $(\Omega,\mathcal{F},P)$ 与 $\sigma$-代数，建议读正式教材而不是只看博客。
7. 生成模型相关综述。适合把边缘化积分、变分推断、后验近似放到统一框架下理解。

建议优先读原始定义和原始论文。博客适合建立直觉，但涉及条件分布、边缘化、ELBO 时，最终应回到正式文献核对符号与假设。
