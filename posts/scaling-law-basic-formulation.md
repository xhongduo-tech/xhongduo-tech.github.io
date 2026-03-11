## 核心结论

Scaling Law，直译为“扩展定律”，指模型性能随资源扩大而呈现稳定数学规律。对语言模型，Kaplan 等人在 2020 年给出的核心结论是：验证损失 $L$ 会随着参数量 $N$ 和训练 token 数 $D$ 的增加按幂律下降，而不是线性下降。

最常见的三个公式是：

$$
L(N) \propto N^{-\alpha}
$$

$$
L(D) \propto D^{-\beta}
$$

$$
L(N, D)=L_0 + A N^{-\alpha} + B D^{-\beta}
$$

其中：

| 符号 | 含义 | 白话解释 |
| --- | --- | --- |
| $L$ | 验证损失 | 模型在没见过的数据上的平均错误程度 |
| $N$ | 参数量 | 模型里可学习权重的规模，通常指非 embedding 参数 |
| $D$ | 训练 token 数 | 训练时实际喂给模型的文本总量 |
| $L_0$ | 极限损失 | 资源无限大时理论上逼近的下界 |
| $\alpha$ | 参数缩放指数 | 参数增加时，损失下降有多快 |
| $\beta$ | 数据缩放指数 | 数据增加时，损失下降有多快 |

Kaplan 论文中的经验拟合结果常被写成：

- 参数方向：$\alpha \approx 0.076$
- 数据方向：$\beta \approx 0.095$

这两个数很小，结论非常直接：扩大资源确实有效，但边际收益递减。参数增加 10 倍，参数不足带来的那部分损失只会乘上 $10^{-0.076}\approx 0.83$；数据增加 10 倍，数据不足带来的那部分损失乘上 $10^{-0.095}\approx 0.80$。

玩具例子可以直接算。假设某模型当前满足：

$$
L_0 = 1.5,\quad A N^{-\alpha}=0.4,\quad B D^{-\beta}=0.3
$$

那么总损失是：

$$
L = 1.5 + 0.4 + 0.3 = 2.2
$$

如果把参数和数据都各扩大 10 倍，那么两项分别变成：

$$
0.4 \times 10^{-0.076}\approx 0.33
$$

$$
0.3 \times 10^{-0.095}\approx 0.24
$$

于是新损失约为：

$$
L' \approx 1.5 + 0.33 + 0.24 = 2.07
$$

这说明两件事：第一，扩模型和扩数据都有效；第二，二者都不是“翻十倍就大幅变好”，而是缓慢、稳定、可预测地变好。

---

## 问题定义与边界

这里讨论的不是“训练损失”，而是“验证损失”。验证损失指模型在未参与训练的数据上的误差，更接近泛化能力。Scaling Law 主要用于预测“如果我把模型做大、把数据喂多，泛化会改善多少”。

边界必须先说清楚，否则公式容易被误用。

第一，$N$ 不等于所有参数的机械总数。Kaplan 语境里常强调非 embedding 参数，也就是不把词表嵌入层的规模混进去。白话说，真正关心的是模型主体计算能力，而不是词表表面积。

第二，$D$ 是训练过程中实际看到的 token 总量，不是原始文件大小，也不是样本条数。对白话文本、代码、数学语料，这个量的质量和分布也会影响拟合。

第三，算力预算 $C$ 是约束条件。它通常用 FLOP 表示，也就是浮点运算次数。白话说，$C$ 是你能花掉的总训练计算资源。实际训练里，$N$、$D$ 和 $C$ 不是彼此独立的自由变量，因为更大的模型和更多的数据都会消耗更多计算。

可以把问题写成：

$$
\text{在给定算力 } C \text{ 下，怎样分配 } N \text{ 和 } D \text{，使 } L(N,D) \text{ 最低？}
$$

一个初学者常见误区是只扩参数，不扩数据。比如把模型从 $10^7$ 参数加到 $10^8$，但训练 token 还停留在 $10^{11}$。这时参数不足项会下降，但数据不足项几乎不变，总损失下降幅度有限。直观上，模型“脑子变大了”，但“看过的书”没变多，泛化不会同步改善。

下面这个表可以帮助划边界：

| 变量 | 典型问题 | 误用方式 | 正确理解 |
| --- | --- | --- | --- |
| $N$ | 模型要多大 | 只看总参数，不看有效参数 | 关注参与主体计算的参数规模 |
| $D$ | 数据要喂多少 | 用样本数代替 token 数 | 用训练 token 总量建模 |
| $C$ | 预算有多少 | 忽略算力和训练时间约束 | 在固定 FLOP 下联合分配资源 |
| $L_0$ | 最低能到哪里 | 当成 0 或直接忽略 | 它代表不可消除的剩余误差下界 |

所以，Scaling Law 不是一句“越大越好”。它真正回答的是：在当前数据分布、模型族、优化器和预算范围内，扩大哪种资源更值。

---

## 核心机制与推导

幂律，白话说，就是“资源每扩大一个固定倍数，指标按固定比例缩放”。它不是一次函数，也不是指数函数。把幂律取对数后，会变成直线，这就是它容易拟合的原因。

### 1. 单变量形式

先看参数量固定数据充足时的近似：

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha}
$$

这里 $N_c$ 是一个尺度常数，白话说，它把“多大才算开始进入有效下降区间”编码进公式。

两边取对数：

$$
\log L = \alpha \log N_c - \alpha \log N
$$

所以如果把 $\log N$ 作为横轴、$\log L$ 作为纵轴，点大致落在一条斜率为 $-\alpha$ 的直线上。数据方向同理：

$$
L(D) = \left(\frac{D_c}{D}\right)^{\beta}
$$

取对数后：

$$
\log L = \beta \log D_c - \beta \log D
$$

这就是为什么论文里常先画 log-log 图，再做最小二乘拟合。最小二乘，白话说，就是找一条线，让所有点到这条线的误差平方和最小。

### 2. 为什么联合公式是“相加”

单变量公式只能描述“只有一个瓶颈在起作用”的情况，但真实训练里常常同时存在两种不足：

- 参数不足：模型容量不够，无法吸收规律
- 数据不足：模型有容量，但样本不够，无法学满

因此可以把总损失拆成三部分：

$$
L(N,D)=L_0+\Delta L_N+\Delta L_D
$$

其中：

$$
\Delta L_N = A N^{-\alpha}
$$

$$
\Delta L_D = B D^{-\beta}
$$

代回去得到：

$$
L(N,D)=L_0 + A N^{-\alpha} + B D^{-\beta}
$$

这个“相加”不是凭直觉拍脑袋，而是经验建模：把参数不足和数据不足视为两类可分离的附加误差。$L_0$ 是再怎么扩资源也难以消掉的下界，可能来自数据噪声、任务本身不确定性、模型族偏差等。

### 3. 数值推导例子

继续用玩具例子：

$$
L_0=1.5,\quad A N^{-\alpha}=0.4,\quad B D^{-\beta}=0.3
$$

总损失是 $2.2$。如果 $N$ 增加 10 倍，且 $\alpha=0.076$，那么参数项变成：

$$
0.4 \cdot 10^{-0.076}\approx 0.33
$$

如果 $D$ 也增加 10 倍，且 $\beta=0.095$，那么数据项变成：

$$
0.3 \cdot 10^{-0.095}\approx 0.24
$$

于是：

$$
L' \approx 1.5 + 0.33 + 0.24 = 2.07
$$

这个例子有两个读法。

第一种读法是“收益递减”。10 倍资源并没有把附加损失砍半。

第二种读法是“联合扩展优于单边扩展”。如果你只扩参数而不扩数据，那么只能把 $0.4$ 压到 $0.33$，但 $0.3$ 还原地不动，总损失只有：

$$
1.5 + 0.33 + 0.3 = 2.13
$$

比同时扩展更差。

### 4. 真实工程例子

真实工程里，训练团队会先用一批较小实验点拟合参数，再推算更大预算下的配置。比如已有一组不同 $(N,D)$ 组合对应的验证损失，先拟合出 $A,B,\alpha,\beta,L_0$，再在固定算力预算 $C$ 下搜索最优点。

Kaplan 体系下常见经验关系是：

$$
N \propto C^{0.73}, \qquad D \propto C^{0.27}
$$

意思是：在固定 FLOP 预算下，参数规模应比数据规模增长得更快。白话说，在那个实验区间里，继续做大模型通常比等比例堆数据更划算。但这不是普适真理，它依赖具体模型族和训练区间，后续 Chinchilla 工作就给出了不同的最优分配结论。

---

## 代码实现

下面给一个最小可运行版本，分两步做：

1. 用单变量 log-log 线性回归估计 $\alpha,\beta$
2. 在指数已知或近似已知时，用网格搜索拟合 $L_0,A,B$

这个实现不依赖 `scipy`，只用 `numpy`，便于直接运行。

```python
import math
import numpy as np

# 生成一组玩具数据
TRUE_L0 = 1.50
TRUE_A = 1.20
TRUE_B = 0.90
TRUE_ALPHA = 0.076
TRUE_BETA = 0.095

def loss_fn(N, D, L0, A, B, alpha, beta):
    return L0 + A * (N ** (-alpha)) + B * (D ** (-beta))

Ns = np.array([1e6, 3e6, 1e7, 3e7, 1e8], dtype=float)
Ds = np.array([1e9, 3e9, 1e10, 3e10, 1e11], dtype=float)

# 构造参数扫描数据：固定 D 很大，只看 N
D_large = 1e15
L_by_N = np.array([loss_fn(N, D_large, TRUE_L0, TRUE_A, TRUE_B, TRUE_ALPHA, TRUE_BETA) for N in Ns])

# 构造数据扫描数据：固定 N 很大，只看 D
N_large = 1e12
L_by_D = np.array([loss_fn(N_large, D, TRUE_L0, TRUE_A, TRUE_B, TRUE_ALPHA, TRUE_BETA) for D in Ds])

# 单变量拟合前，先减去已知或预估的 L0；真实工程里 L0 需要联合估计
L0_guess = 1.50

def fit_power_law(xs, ys, floor):
    residual = ys - floor
    assert np.all(residual > 0), "ys - floor 必须为正，log 才有定义"
    coef = np.polyfit(np.log(xs), np.log(residual), 1)
    slope, intercept = coef
    exponent = -slope
    scale = math.exp(intercept)
    return exponent, scale

alpha_hat, A_hat = fit_power_law(Ns, L_by_N, L0_guess)
beta_hat, B_hat = fit_power_law(Ds, L_by_D, L0_guess)

assert abs(alpha_hat - TRUE_ALPHA) < 0.01
assert abs(beta_hat - TRUE_BETA) < 0.01

# 联合拟合：指数固定为前一步估计值，搜索 L0，线性最小二乘解 A/B
pairs = [
    (1e6, 1e9), (1e6, 1e10), (1e7, 1e10), (1e7, 1e11),
    (1e8, 1e10), (1e8, 1e11), (3e7, 3e10), (3e6, 3e9)
]
y = np.array([loss_fn(N, D, TRUE_L0, TRUE_A, TRUE_B, TRUE_ALPHA, TRUE_BETA) for N, D in pairs])

best = None
for L0_try in np.linspace(1.3, 1.7, 401):
    X = []
    target = []
    for (N, D), yi in zip(pairs, y):
        X.append([N ** (-alpha_hat), D ** (-beta_hat)])
        target.append(yi - L0_try)
    X = np.array(X)
    target = np.array(target)
    coeffs, *_ = np.linalg.lstsq(X, target, rcond=None)
    A_try, B_try = coeffs
    pred = L0_try + X @ coeffs
    mse = np.mean((pred - y) ** 2)
    if best is None or mse < best[0]:
        best = (mse, L0_try, A_try, B_try)

mse, L0_hat, A2_hat, B2_hat = best

def predict_loss(N, D):
    return L0_hat + A2_hat * (N ** (-alpha_hat)) + B2_hat * (D ** (-beta_hat))

toy_before = predict_loss(1e7, 1e11)
toy_after = predict_loss(1e8, 1e12)

assert toy_after < toy_before
assert abs(L0_hat - TRUE_L0) < 0.05

print("alpha_hat =", round(alpha_hat, 4))
print("beta_hat  =", round(beta_hat, 4))
print("L0_hat    =", round(L0_hat, 4))
print("toy_before=", round(toy_before, 4))
print("toy_after =", round(toy_after, 4))
```

这段代码体现了一个工程上常见的做法：先用“近似单变量区间”估指数，再做联合拟合。原因是如果一开始就把 $L_0,A,B,\alpha,\beta$ 全部丢进黑盒优化，容易不稳定，尤其样本点少时更明显。

真实工程中可以进一步做三件事：

| 步骤 | 做法 | 目的 |
| --- | --- | --- |
| 数据清洗 | 只保留同一训练配方下的实验点 | 避免把不同优化器和 tokenizer 差异混进拟合 |
| 分段拟合 | 先看 log-log 图，再决定哪些点进入回归 | 避免把明显偏离幂律区间的点强行拟合 |
| 置信区间 | 对实验点做 bootstrap | 判断 $\alpha,\beta$ 是否稳定 |

如果你用 `scipy.optimize.curve_fit`，可以直接对联合公式做非线性最小二乘，但前提仍然是：初值要合理，数据区间要干净。

---

## 工程权衡与常见坑

第一个权衡是“单变量方便，联合模型更可信”。单变量式子好画、好讲、好拟合，但一旦真正要做资源分配，联合模型更接近决策需要。

| 方法 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 单变量拟合 | 简单、直观、样本需求低 | 忽略另一个瓶颈 | 教学、初步摸底 |
| 联合拟合 | 能指导 $N$/$D$ 分配 | 参数更多、拟合更脆弱 | 预算规划、实验排期 |

第二个权衡是“Kaplan 指数不是万能常数”。$\alpha \approx 0.076$、$\beta \approx 0.095$ 是在特定模型族、数据和训练区间中拟合出来的经验结果。换模型结构、换 tokenizer、换数据质量、换优化策略，指数可能变化。把论文里的数字原样抄到自己项目里，通常不够严谨。

第三个常见坑是忽略 $L_0$。如果把公式错写成：

$$
L(N,D)=A N^{-\alpha} + B D^{-\beta}
$$

那么当资源扩大时，模型会被错误地预测成可以无限逼近 0 损失。这会高估扩展收益，尤其在已经接近饱和的区间。

第四个常见坑是只按参数幂律做预算。比如团队只看到“参数更值钱”，于是把 90% FLOP 都给模型规模，数据没有跟上。结果是训练损失也许继续下降，但验证损失改善有限，因为数据项仍是主要瓶颈。

可以用一个检查表避免误判：

- 是否明确了 $N$ 的定义，尤其是否排除了不一致的 embedding 统计口径
- 是否使用训练 token 数而不是文档数、样本数
- 是否只拟合同一训练配方下的数据点
- 是否先画过 log-log 图，确认点大致落在线性区间
- 是否保留了 $L_0$ 而不是默认它为 0
- 是否验证了指数在当前区间内稳定，而不是被极少数点牵着走
- 是否在最终决策前，用联合公式而不是单变量公式做预算搜索

---

## 替代方案与适用边界

Scaling Law 适合回答“继续扩大预训练规模值不值”，但它不是唯一方法，也不是任何预算下都优先。

当你已有较强的基座模型时，问题往往不再是“从零训练多大模型”，而是“怎样在有限资源下适配新任务”。这时 LoRA，白话说，就是只训练少量低秩增量参数的微调方法，可能比继续扩大 $N$ 更现实。类似地，数据增强、蒸馏、检索增强，也可能比盲目加参数更有效。

下面给一个边界表：

| 资源范围 | 典型约束 | 推荐策略 |
| --- | --- | --- |
| 数据和算力都充足 | 预算大，目标是逼近最优损失 | 用联合 scaling law 做全局分配 |
| 算力紧张，显存受限 | 大模型放不下或训练太慢 | 增加高质量数据，或用参数高效微调 |
| 数据极少 | 幂律区间可能尚未建立 | 先补数据质量，不要急着外推指数 |
| 已有强基座模型 | 从零训练成本过高 | LoRA、蒸馏、继续预训练、检索增强 |

给一个真实感更强的场景：如果预算是 $10^{23}$ FLOP，但硬件显存已经卡住，继续把模型做大可能根本训不起来。这时单看 Kaplan 式的“参数优先”会给出错误直觉。更可行的路线可能是：

- 提高 $D$，补充更高质量 token
- 保持基座不变，做继续预训练
- 用 LoRA 之类的低秩适配降低训练成本
- 改善数据去重、采样和课程学习，而不是只堆参数

所以，Scaling Law 的适用边界可以概括成一句话：它最适合“同一模型族、同一训练配方、足够多实验点、明确预算约束”的规模规划问题；一旦跨到别的模型机制或极小样本区域，最好重新拟合，甚至换方法。

---

## 参考资料

| 分类 | 资料 | 时间 | 用途 |
| --- | --- | --- | --- |
| 原始研究 | Kaplan et al. - *Scaling Laws for Neural Language Models*（OpenAI Research） | 2020 | 原始幂律结论、参数/数据/算力关系来源 |
| 解读 | Yue Shui - *Scaling laws* | 2025 | 面向读者的推导说明与数值直觉 |
| 工程总结 | Emergent Mind - *Kaplan & Chinchilla Scaling Laws* | 2025 | 总结 Kaplan 与后续 Chinchilla 的工程差异 |

- Kaplan et al. 2020. *Scaling Laws for Neural Language Models*. OpenAI Research.
- Yue Shui. 2025. *Scaling laws*.
- Emergent Mind. 2025. *Kaplan & Chinchilla Scaling Laws*.
