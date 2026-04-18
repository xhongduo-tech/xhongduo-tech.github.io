## 核心结论

Radon-Nikodym 定理回答的问题是：一个测度能否写成另一个测度上的加权积分。它不是传统微积分里“函数对变量求导”的点导数，而是“测度相对测度”的导数。

核心条件是绝对连续性：

$$
P \ll \mu
$$

意思是只要参考测度 $\mu$ 认为某个集合 $A$ 的大小为 0，目标测度 $P$ 也必须认为它的大小为 0：

$$
\mu(A)=0 \Rightarrow P(A)=0
$$

在常用的 $\sigma$-有限条件下，Radon-Nikodym 定理保证存在一个可测函数 $p$，使得：

$$
P(A)=\int_A p\,d\mu
$$

并记作：

$$
p=\frac{dP}{d\mu}
$$

这个 $p$ 就是 Radon-Nikodym 导数。

| 概念 | 含义 |
|---|---|
| 绝对连续 $P \ll \mu$ | $\mu(A)=0$ 时 $P(A)=0$ |
| RN 导数 $dP/d\mu$ | 把 $P$ 写成相对 $\mu$ 的密度 |
| pdf | $\mu=\lambda$ 时的 RN 导数 |

新手版理解：可以把“总面积为 1 的布料”看成一个概率空间。参考测度 $\mu$ 描述每块布料的面积，目标测度 $P$ 描述每块布料上承载的概率重量。RN 导数就是“单位面积上放了多少概率重量”。如果换一把不同规格的尺子，数值会变，但描述的仍然是同一个分布。

概率密度 `pdf` 只是一个特例：当参考测度是 Lebesgue 测度 $\lambda$，也就是普通连续空间里的长度、面积、体积时，$p=dP/d\lambda$ 才是通常说的概率密度函数。

---

## 问题定义与边界

测度是给集合分配“大小”的规则。概率测度是总大小为 1 的测度。密度函数不是分布本身，而是分布相对于某个参考测度的表示方式。分布则是随机变量取值规律的整体对象。

| 对象 | 说明 |
|---|---|
| $\mu$ | 参考测度，用来定义“基准大小” |
| $P$ | 目标测度或概率测度 |
| $P \ll \mu$ | $P$ 受 $\mu$ 控制 |
| $\sigma$-有限 | 空间可分解为可数个有限测度集合，是 RN 存在性常用前提 |

RN 定理的边界很重要：不是任意两个测度都能写出 RN 导数。必须先确认：

$$
\mu(A)=0 \Rightarrow P(A)=0
$$

然后才有资格写：

$$
P(A)=\int_A p\,d\mu
$$

玩具例子：在实数轴上，连续均匀分布可以相对 Lebesgue 测度写出 pdf。但一个只在点 $0$ 上取值的分布，也就是 $P(\{0\})=1$，不能写成普通 Lebesgue pdf。因为 Lebesgue 测度认为单点集合 $\{0\}$ 长度为 0，但这个概率测度认为它的概率是 1，所以不满足 $P \ll \lambda$。

离散分布和连续分布使用的“密度语言”不同。离散变量通常相对 counting measure，也就是“数点个数”的测度，写成 pmf。连续变量通常相对 Lebesgue measure 写成 pdf。混合分布同时含有离散点质量和连续部分时，通常不能只靠一个 Lebesgue pdf 完整表达。

不适用的典型情况包括：非 $\sigma$-有限时不能直接套 RN 定理；目标测度不绝对连续于参考测度时不存在 RN 导数；离散-连续混合分布不能强行压成一个普通 pdf。

---

## 核心机制与推导

RN 导数的本质是“换参考测度后的权重函数”。一旦 $P$ 可以写成 $p=dP/d\mu$，所有概率、期望、KL 都可以统一成积分形式。

从定义出发：

$$
P(A)=\int_A p\,d\mu
$$

若 $g$ 是一个可积函数，则期望可以写成：

$$
E_P[g]=\int g\,dP=\int g(x)p(x)\,d\mu(x)
$$

这一步的含义是：先用 $p(x)$ 把参考测度 $\mu$ 加权成目标测度 $P$，再对函数 $g$ 求平均。

推导图如下：

| 层次 | 表达 |
|---|---|
| 参考测度 | $\mu$ |
| 相对权重 | $p=dP/d\mu$ |
| 概率 | $P(A)=\int_A p\,d\mu$ |
| 期望 | $E_P[g]=\int g p\,d\mu$ |
| KL | $KL(Q\|P)=\int \log(dQ/dP)\,dQ$ |

新手版例子：如果 $p(x)$ 已经告诉我们每个位置相对 $\mu$ 有多少概率重量，那么区间概率就是把这个区间内的重量加起来。算期望时，再把函数值 $g(x)$ 乘进去，然后积分。

KL 散度是衡量两个概率测度差异的量。若 $Q \ll P$，可以写成：

$$
KL(Q\|P)=\int \log\left(\frac{dQ}{dP}\right)dQ
$$

这说明 KL 的核心也是 RN 导数：需要知道 $Q$ 相对 $P$ 的密度比。

在变分推断中，$Q$ 是近似后验，$P$ 是真实后验。常见目标是最大化 ELBO：

$$
ELBO=E_Q[\log p(x,z)-\log q(z)]
$$

并且有：

$$
ELBO=\log p(x)-KL(Q\|P)
$$

所以最大化 ELBO 等价于让近似后验 $Q$ 靠近真实后验 $P$。

真实工程例子：normalizing flows 把简单分布变成复杂分布。设 $\epsilon \sim \nu$，密度为 $r(\epsilon)$，通过可逆可微变换 $z=T_\phi(\epsilon)$ 得到新分布：

$$
Q_\phi=T_{\phi\#}\nu
$$

这里 $T_{\phi\#}\nu$ 叫 pushforward measure，意思是把原测度通过函数 $T_\phi$ 推到新空间上。若 $T_\phi$ 可逆可微，则密度变换为：

$$
q_\phi(z)=r(\epsilon)\left|\det J_{T_\phi}^{-1}(z)\right|
$$

Jacobian 行列式是体积修正项，用来补偿变换后局部空间被拉伸或压缩的比例。

---

## 代码实现

下面用最小 Python 例子验证一个 RN 导数。取参考测度为 $[0,1]$ 上的 Lebesgue 测度，定义：

$$
p(x)=2x,\quad 0\le x\le 1
$$

则：

$$
P([0,1])=\int_0^1 2x\,dx=1
$$

```python
import numpy as np
from scipy.integrate import quad

def p(x):
    return 2 * x if 0 <= x <= 1 else 0.0

prob_all, _ = quad(p, 0, 1)
prob_half, _ = quad(p, 0, 0.5)
ex, _ = quad(lambda x: x * p(x), 0, 1)

print(prob_all, prob_half, ex)

assert abs(prob_all - 1.0) < 1e-10
assert abs(prob_half - 0.25) < 1e-10
assert abs(ex - 2 / 3) < 1e-10
```

这段代码对应的数学对象如下：

| 代码项 | 数学含义 |
|---|---|
| `p(x)` | RN 导数 / 密度 |
| `quad` | 积分 $\int$ |
| `prob_all` | $P([0,1])$ |
| `prob_half` | $P([0,1/2])$ |
| `ex` | $E[X]$ |

在 flow 或变分推断中，代码结构通常类似：

```python
# epsilon ~ N(0, I)
# z = T_phi(epsilon)
# log q_phi(z) = log r(epsilon) - log|det J_T_phi(epsilon)|
# optimize ELBO = E_q[log p(x, z) - log q_phi(z)]
```

| 代码项 | 数学含义 |
|---|---|
| `epsilon` | 基分布样本 |
| `T_phi` | 可逆变换 |
| `z` | 变换后的隐变量 |
| `log r(epsilon)` | 基分布的 log 密度 |
| `det J` | 变换时的体积修正 |
| `log q_phi(z)` | 近似后验的 log 密度 |
| `ELBO` | 变分推断优化目标 |

---

## 工程权衡与常见坑

工程上最容易错的不是 RN 定理本身，而是没有说清楚参考测度，或者把 density 当 probability。

| 常见坑 | 结果 | 规避方式 |
|---|---|---|
| 不写参考测度 | 密度含义不清 | 先写 $dP/d\mu$ |
| 把 pdf 当概率 | 误解数值大小 | 概率是对集合积分 |
| 混合分布硬套 pdf | 公式失真 | 分解离散/连续部分 |
| 忽略 $\sigma$-有限 | 定理可能不适用 | 先确认假设 |
| 忘记 Jacobian | flow 公式错误 | 先检查可逆可微 |
| 用单点值判断 RN 导数 | 受“几乎处处”限制 | 只讨论积分意义 |

错误写法：

```text
p(0.3)=1.4，所以 x=0.3 的概率是 1.4。
```

正确写法：

```text
p(0.3)=1.4 表示 x=0.3 附近相对于 Lebesgue 测度的局部概率密度。
真正的概率必须写成 P(A)=∫_A p(x) dx。
```

RN 导数只要求“几乎处处唯一”。几乎处处的意思是：允许在参考测度为 0 的集合上不同。两个密度函数即使在单点上取值不同，只要它们在积分意义上给出相同概率测度，就可以代表同一个 RN 导数。

混合分布是另一个常见坑。例如某个模型输出：以 30% 概率直接取 0，以 70% 概率从连续分布采样。这个分布有一个离散原子 $\{0\}$，也有连续部分。它不能用一个普通 Lebesgue pdf 完整概括，因为单点质量无法被 Lebesgue 密度表达。

在 normalizing flows 里，公式：

$$
q_\phi(z)=r(\epsilon)|\det J^{-1}|
$$

依赖可逆性、可微性和维度匹配。如果变换不是双射，或者输入输出维度不同，就不能直接使用这个标准 change-of-variables 公式。

---

## 替代方案与适用边界

RN 定理是统一表达测度变化的通用语言，但不是每个工程任务都需要从测度论开始。

| 场景 | 可用方法 | 适用性 |
|---|---|---|
| 离散变量 | pmf / counting measure | 简单直接 |
| 连续变量 | pdf / Lebesgue measure | 标准情形 |
| 混合分布 | 分块测度或混合模型 | 更稳妥 |
| 流模型 | change-of-variables | 要求可逆可微 |
| VI / ELBO | KL + RN 导数 | 训练可优化 |

如果数据全是离散标签，例如分类任务中的类别编号，直接使用 counting measure 和 pmf 就够了。此时 $P(\{k\})$ 就是类别 $k$ 的概率，不需要引入 Lebesgue pdf。

如果数据是连续观测，例如传感器读数、图像像素归一化值、连续隐变量，Lebesgue 测度下的 pdf 更自然。此时概率仍然不是密度点值，而是区域积分。

如果数据同时包含离散和连续结构，例如“用户是否点击”和“点击后的停留时长”，就应该分块建模：离散部分用 Bernoulli 或 categorical，连续部分用条件密度。强行找一个单一 pdf 往往会让公式看似统一，实际语义错误。

在流模型中：

$$
q_\phi(z)=r(\epsilon)|\det J^{-1}|
$$

只在变换满足条件时成立。若模型使用不可逆网络、降维映射、离散采样步骤，就需要其他方法，例如变分下界、score-based 方法、隐式分布估计或数值采样。

RN 定理的价值在于把“概率密度”“密度比”“KL”“测度变换”放进同一个框架里。对于初级工程师，先掌握 pmf、pdf、积分概率、变换公式已经足够处理大量任务；当遇到混合分布、后验近似、flow、重要性采样、密度比估计时，再回到 RN 导数会更清楚。

---

## 参考资料

理论入门：

- Berkeley Stat 210a Course Reader: Probability  
  https://stat210a.berkeley.edu/fall-2024/reader/probability.html

严格证明：

- Lean mathlib: Radon-Nikodym theorem  
  https://leanprover-community.github.io/mathlib4_docs/Mathlib/MeasureTheory/Measure/Decomposition/RadonNikodym.html

工程应用：

- Blei, Kucukelbir, McAuliffe, *Variational Inference: A Review for Statisticians*  
  https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1285773

- Rezende & Mohamed, *Variational Inference with Normalizing Flows*  
  https://huggingface.co/papers/1505.05770

如果目标是“会用”，优先看概率直觉和 flow / VI 应用；如果目标是“会证”，优先看测度论与 RN 定理证明。
