## 核心结论

Rényi 熵是 Shannon 熵的 `order-α` 广义形式。这里的“熵”指分布的不确定性度量：分布越平均，不确定性越高；分布越集中，不确定性越低。

对离散分布 $P=(p_1,\dots,p_n)$，Rényi 熵定义为：

$$
H_\alpha(P)=\frac{1}{1-\alpha}\log\sum_i p_i^\alpha,\quad \alpha>0,\alpha\ne 1
$$

当 $\alpha\to 1$ 时，它收敛到 Shannon 熵：

$$
\lim_{\alpha\to 1}H_\alpha(P)=-\sum_i p_i\log p_i
$$

$\alpha$ 的作用不是简单放大或缩小熵值，而是改变不同概率事件的权重。$\alpha<1$ 时，小概率事件被相对抬高；$\alpha>1$ 时，高概率事件被更强地强调。

玩具例子：取 $P=(0.9,0.1)$，使用自然对数，单位是 `nats`。

| 指标 | 数值 | 解释 |
|---|---:|---|
| $H_1(P)$ | 0.325 | Shannon 熵 |
| $H_{0.5}(P)$ | 0.470 | 稀有事件 $0.1$ 权重上升，熵更高 |
| $H_2(P)$ | 0.198 | 主峰 $0.9$ 被强调，熵更低 |

α-散度是 KL 散度的一族参数化推广。散度是两个分布之间差异的度量：散度越大，说明用 $Q$ 近似 $P$ 的代价越高。α-散度的价值不只是换一个公式，而是把“更看重尾部”还是“更看重主峰”的偏好显式暴露出来。

---

## 问题定义与边界

本文主要讨论离散分布。连续分布可以把求和换成积分，但需要额外处理可积性、密度函数、支持集等问题。支持集是一个分布可能取到非零概率的区域；如果 $P$ 在某个位置有概率而 $Q$ 完全不给概率，很多散度会发散。

Shannon 熵、KL 散度、Rényi 熵、α-散度的关系可以先按下面理解。

| 名称 | 定义对象 | 是否有参数 | 主要用途 |
|---|---|---:|---|
| Shannon 熵 | 单个分布 $P$ | 否 | 衡量平均不确定性 |
| KL 散度 | 两个分布 $P,Q$ | 否 | 衡量用 $Q$ 近似 $P$ 的代价 |
| Rényi 熵 | 单个分布 $P$ | $\alpha$ | 用可调阶数衡量不确定性 |
| α-散度 | 两个分布 $P,Q$ | $\alpha$ | 用可调偏好衡量分布差异 |

本文采用一种常见归一化形式：

$$
D_\alpha(P\Vert Q)=\frac{1}{\alpha(\alpha-1)}
\left(\sum_i p_i^\alpha q_i^{1-\alpha}-1\right),
\quad \alpha\ne 0,1
$$

注意：不同论文里的 $\alpha$ 可能有重参数化，例如把 $\alpha$ 换成 $(1-\alpha)/2$ 或改变前后向顺序。因此工程上不能只看“α-散度”这个名字，必须先对齐公式。

仍取 $P=(0.9,0.1)$，再取 $Q=(0.5,0.5)$。同一对分布在不同 $\alpha$ 下会得到不同惩罚强度：

| $\alpha$ | $D_\alpha(P\Vert Q)$ | 解释 |
|---:|---:|---|
| 0.5 | 0.422 | 更重视分布重叠与尾部覆盖 |
| 2.0 | 0.320 | 更强调主峰区域的差异 |

所以，“散度值大不大”不能脱离 $\alpha$ 单独解释。比较两个实验结果时，必须确认它们使用的是同一个公式、同一个 $\alpha$、同一个对数单位。

---

## 核心机制与推导

理解 Rényi 熵的关键是 $p_i^\alpha$。它相当于对概率质量做一次幂变换。

当 $0<p_i<1$ 时：

| 条件 | 对小概率项的影响 | 对大概率项的影响 | 优化倾向 |
|---|---|---|---|
| $\alpha<1$ | 相对抬高 | 相对削弱 | 更照顾尾部，更倾向模式覆盖 |
| $\alpha=1$ | 原始权重 | 原始权重 | 回到 Shannon/KL |
| $\alpha>1$ | 相对压低 | 相对强调 | 更关注主峰，更倾向模式寻优 |

“模式”指分布中的高概率区域。模式覆盖是尽量覆盖多个可能解释；模式寻优是优先贴近最主要的解释。

用两点分布看得最清楚。设 $P=(0.9,0.1)$。

当 $\alpha=0.5$ 时：

$$
\sum_i p_i^{0.5}=\sqrt{0.9}+\sqrt{0.1}
$$

平方根会把 $0.1$ 拉高到约 $0.316$，它相对 $0.9$ 的存在感变强。因此 $H_{0.5}$ 更高。

当 $\alpha=2$ 时：

$$
\sum_i p_i^2=0.9^2+0.1^2=0.82
$$

平方会把 $0.1$ 压到 $0.01$，尾部几乎不影响结果。因此 $H_2$ 更低。

$\alpha\to1$ 时，Rényi 熵回到 Shannon 熵。令

$$
f(\alpha)=\log\sum_i p_i^\alpha
$$

则 $H_\alpha(P)=f(\alpha)/(1-\alpha)$，且 $f(1)=\log\sum_i p_i=0$。对 $f$ 求导：

$$
f'(\alpha)=
\frac{\sum_i p_i^\alpha \log p_i}{\sum_i p_i^\alpha}
$$

代入 $\alpha=1$：

$$
f'(1)=\sum_i p_i\log p_i
$$

由洛必达法则：

$$
\lim_{\alpha\to1}H_\alpha(P)
=
\lim_{\alpha\to1}\frac{f(\alpha)}{1-\alpha}
=
-f'(1)
=
-\sum_i p_i\log p_i
$$

α-散度也可以从重加权角度理解。核心项是：

$$
\sum_i p_i^\alpha q_i^{1-\alpha}
$$

它衡量 $P$ 和 $Q$ 在各个位置上的加权重叠。$\alpha$ 改变的是重叠区域中谁的权重更大。放到变分推断里，$P$ 可以是真实后验，$Q$ 可以是近似后验。真实后验是观察数据后的理想概率分布；近似后验是模型实际能计算或优化的替代分布。

真实工程例子：在 VAE 或贝叶斯神经网络中，后验可能有多个模式。例如一张模糊图片既可能解释成数字 `3`，也可能解释成数字 `8`。如果目标更偏 $\alpha<1$，近似分布会更愿意覆盖多个解释；如果目标更偏 $\alpha>1$，近似分布会更倾向压到主要解释上，减少尾部和噪声的影响。

---

## 代码实现

先写最小可复现版本，再讨论数值稳定性。下面代码使用 `numpy`，默认自然对数，所以单位是 `nats`。如果要用 `bits`，需要把结果除以 $\log 2$。

```python
import numpy as np

def normalize(p):
    p = np.asarray(p, dtype=float)
    assert np.all(p >= 0), "probabilities must be non-negative"
    s = p.sum()
    assert s > 0, "sum of probabilities must be positive"
    return p / s

def shannon_entropy(p):
    p = normalize(p)
    mask = p > 0
    return -np.sum(p[mask] * np.log(p[mask]))

def renyi_entropy(p, alpha, tol=1e-8):
    p = normalize(p)
    assert alpha > 0, "alpha must be positive"
    if abs(alpha - 1.0) < tol:
        return shannon_entropy(p)
    return np.log(np.sum(p ** alpha)) / (1.0 - alpha)

def alpha_divergence(p, q, alpha, tol=1e-8, eps=1e-12):
    p = normalize(p)
    q = normalize(q)

    # Small clipping avoids undefined powers when q_i is zero and alpha > 1.
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = normalize(p)
    q = normalize(q)

    if abs(alpha - 1.0) < tol:
        mask = p > 0
        return np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask])))

    assert alpha != 0.0, "this normalized form excludes alpha=0"
    overlap = np.sum((p ** alpha) * (q ** (1.0 - alpha)))
    return (overlap - 1.0) / (alpha * (alpha - 1.0))

P = np.array([0.9, 0.1])
Q = np.array([0.5, 0.5])

h1 = renyi_entropy(P, 1.0)
h05 = renyi_entropy(P, 0.5)
h2 = renyi_entropy(P, 2.0)

d05 = alpha_divergence(P, Q, 0.5)
d2 = alpha_divergence(P, Q, 2.0)

print(round(h1, 3), round(h05, 3), round(h2, 3))
print(round(d05, 3), round(d2, 3))

assert abs(h1 - 0.3250829733914482) < 1e-9
assert abs(h05 - 0.47000362924573563) < 1e-9
assert abs(h2 - 0.19845093872383818) < 1e-9
assert abs(d05 - 0.4222912360003366) < 1e-9
assert abs(d2 - 0.32) < 1e-9
assert h05 > h1 > h2
```

示例输出：

```text
0.325 0.47 0.198
0.422 0.32
```

实现时需要固定输入输出约定。

| 项目 | 约定 | 注意事项 |
|---|---|---|
| 输入分布 | 非负数组 | 先归一化，避免用户传入计数 |
| 熵输出 | 标量 | 默认 `nats`，不要和 `bits` 混用 |
| 散度输出 | 标量 | 只在同一公式和同一 $\alpha$ 下比较 |
| $\alpha\approx1$ | 走极限分支 | 不要直接代入原式 |
| 零概率 | 掩码或平滑 | 避免 `log(0)`、`0` 的负幂 |

---

## 工程权衡与常见坑

第一个坑是符号不统一。论文 A 的 $\alpha$ 和论文 B 的 $\alpha$ 可能不是同一个参数。尤其在信息几何、变分推断、GAN 稳定训练中，常见形式包括 Rényi divergence、Amari α-divergence、power divergence。它们有关联，但不能把数值直接混在一起比较。

第二个坑是 $\alpha\to1$。公式里有分母 $1-\alpha$ 或 $\alpha(\alpha-1)$，直接代入会除以零。正确做法是走极限分支：

$$
\alpha\to1 \quad\Rightarrow\quad H_\alpha(P)\to H(P),\quad D_\alpha(P\Vert Q)\to KL(P\Vert Q)
$$

第三个坑是支持集不重叠。如果 $P_i>0$ 但 $Q_i=0$，KL 中会出现 $\log(P_i/Q_i)$，散度可能发散。α-散度在某些 $\alpha$ 下也会遇到类似问题。工程上常见处理是加 $\epsilon$ 平滑、裁剪概率、或在建模阶段保证 $Q$ 的支持集覆盖 $P$。

| 坑点 | 触发条件 | 后果 | 规避办法 |
|---|---|---|---|
| 直接代入 $\alpha=1$ | 分母为零 | `NaN` 或异常大数 | 单独实现 Shannon/KL 分支 |
| `log(0)` | 概率为 0 | `inf` | 对 $p_i=0$ 做掩码 |
| $q_i=0,p_i>0$ | 支持集不覆盖 | 散度发散 | 平滑、截断、重新设计近似族 |
| 单位混用 | 有的代码用 `log2`，有的用 `ln` | 数值差 $\log2$ 倍 | 明确 `nats` 或 `bits` |
| 参数化不同 | 不同论文公式不同 | 复现实验失败 | 先写出公式再比较 |

真实工程中还要考虑梯度稳定性。若把 α-散度放进训练目标，$\alpha$ 过大可能让梯度集中在少数高概率样本上，导致训练不稳定；$\alpha$ 过小可能让模型过度关注尾部，增加方差。这里的选择不是“越大越好”或“越小越好”，而是与任务偏好绑定。

---

## 替代方案与适用边界

当目标是标准概率拟合、模型比较、信息增益解释时，Shannon 熵和 KL 散度通常更直接。它们定义稳定、解释成熟、工具链完备，不需要额外调 $\alpha$。

当任务关心鲁棒性、多模态覆盖、异常点抑制、可调不确定性偏好时，Rényi 熵和 α-散度更有价值。它们把偏好写进目标函数，而不是藏在经验调参里。

| 方法 | 主要对象 | 优点 | 适用场景 |
|---|---|---|---|
| KL 散度 | 两个分布 | 标准、可解释、常用 | 最大似然、ELBO、信息增益 |
| Rényi 熵/散度 | 分布或分布对 | 可调尾部与主峰偏好 | 多模态后验、稳健推断 |
| JS 散度 | 两个分布 | 对称、有界 | GAN、分布比较 |
| Wasserstein 距离 | 两个分布 | 支持集不重叠时仍有几何意义 | 生成模型、最优传输 |

VAE 或贝叶斯神经网络是典型场景。如果后验明显多模态，使用 $\alpha<1$ 的目标可能更利于覆盖多个解释，减少不确定性低估。如果数据中存在噪声、离群点，或工程目标更关心主要模式，$\alpha>1$ 可能更合适。

但不要为了“更高级”而引入 $\alpha$。如果一个普通 KL 目标已经能解释问题、稳定训练、满足评估指标，引入额外超参数只会增加搜索成本和复现难度。论文复现时尤其要注意作者使用的是 Rényi divergence、Amari α-divergence，还是某种经过缩放和重参数化的目标。

---

## 参考资料

| 资料 | 支持内容 | 对应章节 |
|---|---|---|
| Rényi, 1961: *On Measures of Entropy and Information* | Rényi 熵的定义来源 | 核心结论、核心机制与推导 |
| Amari & Cichocki, 2010: *Information geometry of divergence functions* | 从信息几何理解散度函数族 | 问题定义与边界、核心机制与推导 |
| Li & Turner, 2016: *Rényi Divergence Variational Inference* | Rényi 目标在变分推断中的应用 | 核心机制与推导、替代方案与适用边界 |
| MDPI, 2020: *Utilizing Amari-Alpha Divergence to Stabilize the Training of GANs* | α-散度在生成模型训练稳定性中的应用 | 工程权衡与常见坑、替代方案与适用边界 |

- Rényi, 1961: [On Measures of Entropy and Information](https://cir.nii.ac.jp/crid/1572261550246171008?lang=en)
- Amari & Cichocki, 2010: [Information geometry of divergence functions](https://doi.org/10.2478/v10175-010-0019-1)
- Li & Turner, 2016: [Rényi Divergence Variational Inference](https://papers.nips.cc/paper/6208-renyi-divergence-variational-inference)
- MDPI, 2020: [Utilizing Amari-Alpha Divergence to Stabilize the Training of Generative Adversarial Networks](https://www.mdpi.com/1099-4300/22/4/410)
