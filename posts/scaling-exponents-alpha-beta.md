## 核心结论

扩展定律研究的是一个经验规律：当模型参数量 $N$、训练 token 数 $D$ 变大时，验证损失 $L$ 往往按幂律下降。幂律就是“输入放大若干倍，输出按固定指数缩放”的关系。语言模型里常用的形式是：

$$
L(N,D)=\frac{A}{N^\alpha}+\frac{B}{D^\beta}+E
$$

其中 $\alpha$ 衡量“加参数”带来的收益，$\beta$ 衡量“加数据”带来的收益，$E$ 是无法继续降下去的损失底噪。

Kaplan 与 Chinchilla 的关键分歧，不在于都承认幂律，而在于对指数的估计完全不同。Kaplan 论文给出的近似值是 $\alpha\approx0.076,\ \beta\approx0.095$；Chinchilla 路线常见拟合值约为 $\alpha\approx0.34,\ \beta\approx0.28$。这会直接改变训练策略：前者更容易得出“做更大的模型，即使训练不完全也划算”，后者更接近“模型和数据应一起增长，很多大模型其实训练 token 不够”。

先看一个最小玩具例子。若只把参数增加 10 倍，损失项按 $10^{-\alpha}$ 缩放：

| 指数设定 | 参数增大 10 倍后的参数项比例 | 数据增大 10 倍后的数据项比例 | 直观含义 |
| --- | --- | --- | --- |
| Kaplan: $\alpha=0.076,\ \beta=0.095$ | $10^{-0.076}\approx0.84$ | $10^{-0.095}\approx0.80$ | 单独加参数或加数据，收益都偏慢 |
| Chinchilla: $\alpha=0.34,\ \beta=0.28$ | $10^{-0.34}\approx0.46$ | $10^{-0.28}\approx0.52$ | 参数和数据都更“值钱” |

这张表说明一件事：$\alpha$ 越小，不是“模型没用”，而是“参数翻倍带来的边际收益更慢”。因此，指数估得偏小，会把最优策略推向“更大的模型、更早停训”。

---

## 问题定义与边界

问题可以写得很明确：给定训练预算 $C$，怎样选择模型大小 $N$ 和训练数据量 $D$，让验证损失最小。

对于稠密 Transformer，训练计算量常近似写成：

$$
C \propto N D
$$

白话讲，就是“参数越多、token 越多，训练总算力消耗越高”，而且两者大致相乘决定总成本。

于是目标变成：

$$
\min_{N,D}\ L(N,D)=\frac{A}{N^\alpha}+\frac{B}{D^\beta}+E
\quad\text{s.t.}\quad ND=\tilde C
$$

这里的边界很重要，因为 Kaplan 和 Chinchilla 不是只差了一组数字，而是实验口径不同。

| 维度 | Kaplan 式做法 | Chinchilla 式做法 |
| --- | --- | --- |
| 拟合视角 | 常从固定 $N$ 或固定 $D$ 的截面看 | 直接联合拟合整个损失曲面 |
| 训练长度 | 常见设定是固定总 token 预算 | 为不同模型单独选更接近收敛的训练长度 |
| 学习率调度 | 早期实验常更简化 | 更强调按模型规模调整训练计划 |
| 风险 | 小模型可能没收敛，数据收益被低估 | 实验成本更高，但指数更稳 |

如果把损失曲面想成一张地形图：横轴是 $N$，纵轴是 $D$，等高线是相同 loss。Kaplan 更像是沿几条切线去估地形坡度；Chinchilla 更像是测完整块地形，再找最低谷。当地形不是完全规则平面时，只看切片很容易把坡度看平。

因此，这个问题的边界不是“谁对谁错”，而是“你采到的数据点，是否真的代表了同等收敛水平下的模型”。

---

## 核心机制与推导

核心推导并不复杂，但很容易在符号上写反。

由约束 $ND=\tilde C$ 得：

$$
D=\frac{\tilde C}{N}
$$

代回损失函数：

$$
L(N)=A N^{-\alpha}+B\left(\frac{\tilde C}{N}\right)^{-\beta}+E
= A N^{-\alpha}+B\tilde C^{-\beta}N^{\beta}+E
$$

对 $N$ 求导并令其为 0：

$$
-\alpha A N^{-\alpha-1}+\beta B \tilde C^{-\beta}N^{\beta-1}=0
$$

整理得：

$$
N^{\alpha+\beta}\propto \tilde C^{\beta}
$$

所以

$$
N^* \propto \tilde C^{\beta/(\alpha+\beta)},\qquad
D^* \propto \tilde C^{\alpha/(\alpha+\beta)}
$$

这一步的白话解释是：参数最优增长速度由“数据指数 $\beta$”控制，数据最优增长速度由“参数指数 $\alpha$”控制，因为两者在预算约束下是此消彼长的。

代入 Kaplan 指数：

$$
\frac{\beta}{\alpha+\beta}=\frac{0.095}{0.076+0.095}\approx0.556,\qquad
\frac{\alpha}{\alpha+\beta}\approx0.444
$$

得到：

$$
N^*\propto C^{0.556},\qquad D^*\propto C^{0.444}
$$

含义是：预算增长时，更偏向先把模型做大。

再代入 Chinchilla 指数：

$$
\frac{\beta}{\alpha+\beta}=\frac{0.28}{0.34+0.28}\approx0.452,\qquad
\frac{\alpha}{\alpha+\beta}\approx0.548
$$

得到：

$$
N^*\propto C^{0.452},\qquad D^*\propto C^{0.548}
$$

这更接近“参数和数据共同增长，数据略快一点”。由于指数接近 0.5，工程上常把它简化理解成 $N$ 和 $D$ 近似等比例扩张。

下面把差异压缩成一张表：

| 指数来源 | $\alpha$ | $\beta$ | $N^*$ 随算力增长 | $D^*$ 随算力增长 | 策略倾向 |
| --- | --- | --- | --- | --- | --- |
| Kaplan | 0.076 | 0.095 | $C^{0.556}$ | $C^{0.444}$ | 偏大模型、偏早停 |
| Chinchilla | 0.34 | 0.28 | $C^{0.452}$ | $C^{0.548}$ | 模型与数据共同增长 |

为什么会差这么大？机制上主要有三点。

第一，拟合方式不同。若只看固定 $D$ 的切片，小模型往往更容易受“没训练够”影响，测到的 loss 会偏高，从而把“增加参数”的收益夸大。

第二，学习率调度不同。学习率调度就是“训练过程中步长怎么变”。若训练后期没有合理衰减，小模型可能在接近最优区间时震荡，大模型反而更稳定，这会系统性扭曲拟合结果。Chinchilla 路线更强调 cosine decay，也就是把学习率按余弦曲线逐步降到很低。

第三，收敛判断不同。收敛不是“跑完固定 token 数”，而是“在该配置下继续训练，收益已经很小”。如果所有模型都强行训练同样长，实际上是在比较“有的基本收敛，有的明显欠训”的混合样本。

真实工程例子就是 Gopher 与 Chinchilla。Gopher 约 280B 参数、300B token；Chinchilla 约 70B 参数、1.4T token，在相近训练算力下，后者验证损失和下游表现更好。这个例子说明：只堆参数而不补足训练 token，常会落到“模型太大、训练太短”的欠训区。

---

## 代码实现

如果要在自己的实验中估计 $\alpha,\beta$，一个实用方法是：先收集一批不同 $(N,D)$ 组合下、尽量接近收敛点的验证损失，再对

$$
L(N,D)=A/N^\alpha + B/D^\beta + E
$$

做联合拟合。

下面给一个可运行的 Python 玩具脚本。它用合成数据模拟损失曲面，再通过网格搜索 $\alpha,\beta$，对固定指数下的 $A,B,E$ 用最小二乘求解。这个方法不依赖 `scipy`，只用 `numpy` 就能跑通。

```python
import numpy as np

# 真实参数：用来生成玩具数据
TRUE_A = 1.2
TRUE_B = 0.9
TRUE_E = 1.05
TRUE_ALPHA = 0.34
TRUE_BETA = 0.28

def loss_fn(N, D, A, B, E, alpha, beta):
    return A / (N ** alpha) + B / (D ** beta) + E

# 一组玩具实验点：不同模型大小 N 和训练 token 数 D
N_vals = np.array([70e6, 140e6, 300e6, 700e6, 1.5e9, 3e9], dtype=float)
D_vals = np.array([5e9, 10e9, 20e9, 40e9, 80e9, 160e9], dtype=float)

records = []
for N in N_vals:
    for D in D_vals:
        y = loss_fn(N, D, TRUE_A, TRUE_B, TRUE_E, TRUE_ALPHA, TRUE_BETA)
        # 加一点很小的噪声，模拟实验波动
        y += np.random.default_rng(0).normal(0, 0.002)
        records.append((N, D, y))

data = np.array(records, dtype=float)
N = data[:, 0]
D = data[:, 1]
y = data[:, 2]

def fit_given_alpha_beta(N, D, y, alpha, beta):
    # 对固定 alpha, beta，A/B/E 是线性参数，可直接最小二乘
    X = np.column_stack([
        1.0 / (N ** alpha),
        1.0 / (D ** beta),
        np.ones_like(N),
    ])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    A, B, E = coeffs
    pred = X @ coeffs
    mse = np.mean((pred - y) ** 2)
    return A, B, E, mse

best = None
for alpha in np.linspace(0.10, 0.50, 161):
    for beta in np.linspace(0.10, 0.45, 141):
        A, B, E, mse = fit_given_alpha_beta(N, D, y, alpha, beta)
        if A <= 0 or B <= 0:
            continue
        if best is None or mse < best["mse"]:
            best = {
                "alpha": alpha,
                "beta": beta,
                "A": A,
                "B": B,
                "E": E,
                "mse": mse,
            }

print(best)

# 断言：拟合出的指数应接近真实值
assert abs(best["alpha"] - TRUE_ALPHA) < 0.05
assert abs(best["beta"] - TRUE_BETA) < 0.05
assert best["A"] > 0 and best["B"] > 0 and best["E"] > 0
```

把这个玩具例子迁移到真实工程时，流程通常是：

1. 记录每个训练 run 的 `params`, `tokens_seen`, `val_loss`。
2. 对每个模型单独选择接近收敛的最佳点，而不是强行取同一训练步。
3. 过滤异常 run，比如数值爆炸、明显欠训、数据污染。
4. 用联合拟合估计 $\alpha,\beta$，同时报告置信区间或残差。

如果训练流程可控，建议每个模型都使用相同形状、不同总长度的 cosine decay。原因很直接：不同模型的最优训练长度不同，若 schedule 不跟着 token 总量缩放，比较会失真。一个常见做法是先设定目标 token 数 $D_i$，再把 warmup 比例固定、decay 终点对齐到该模型自己的总步数。

---

## 工程权衡与常见坑

工程里最常见的错误，不是公式写错，而是把“不公平的实验点”拿去做公平拟合。

| 常见坑 | 具体表现 | 后果 | 应对方式 |
| --- | --- | --- | --- |
| 固定所有模型的训练 token | 小模型可能还在继续降，大模型提前被比较 | 指数被扭曲 | 每个模型单独寻找近收敛点 |
| 学习率 schedule 不缩放 | 同一 LR 曲线套所有规模 | 小模型/大模型收敛质量不同 | 按目标 token 数缩放 cosine decay |
| 只看单一切片 | 固定 $D$ 或固定 $N$ 分别拟合 | 丢失二维耦合结构 | 联合拟合整个 $(N,D)$ 曲面 |
| 忽略误差区间 | 只报一个 $\alpha,\beta$ | 外推时过度自信 | 报残差、bootstrap 区间 |
| 远距离外推 | 用 100M 到 1B 的数据预测 100B | 偏差快速放大 | 尽量控制在 10× 内，超出时保守 |

再看一个训练计划对比。设总训练算力相同，有两种方案：

| 方案 | 模型 | 训练 tokens | 风险 |
| --- | --- | --- | --- |
| 大模型 × 短训练 | 70B | 300B | 欠训，验证损失未到底 |
| 小模型 × 长训练 | 20B | 1.0T | 更接近收敛，但要求更稳定的数据管线 |

很多团队在现实里偏爱第一种，因为“大模型名义上更先进”。但如果目标是给定预算下的最低验证损失，第二种往往更接近 Chinchilla 前沿。真正的难点不在结论，而在执行：长训练意味着更严格的数据去重、更稳定的学习率衰减、更可靠的 checkpoint 恢复，以及更长时间的监控成本。

还有一个常被低估的坑是外推。扩展定律是经验拟合，不是自然常数。若你的观测范围只覆盖到 1B 参数，预测到 5B 参数通常还算可用；直接推到 100B，就应该把结果当成“带巨大误差条的规划值”，而不是采购单。

---

## 替代方案与适用边界

并不是任何团队都必须完整复刻 Chinchilla 式联合拟合。选择方法，取决于你能否负担“让每个点都接近收敛”的实验成本。

| 方法 | 适用场景 | 优点 | 局限 |
| --- | --- | --- | --- |
| Kaplan 式切片拟合 | 预算很紧、只能做少量实验 | 快，样本需求低 | 容易混入欠训偏差 |
| Chinchilla 式联合拟合 | 能系统控制训练计划 | 指数更稳定，能直接找最优前沿 | 实验成本高 |
| 局部经验回归 | 只关心一小段规模区间 | 对当前项目实用 | 不适合远距离外推 |
| 多模型集成外推 | 需要给管理层做预算预测 | 能表达不确定性 | 实现更复杂 |

可以把它理解成两种团队画像。

研究型团队通常有较强的训练控制力，能接受多跑一批小模型，也更愿意花成本验证“是否真的收敛”。这类团队更适合 Chinchilla 式方法，因为他们真正需要的是可靠指数，而不是快速出一个近似答案。

企业训练团队往往更关心交付时间与预算封顶。如果数据受限、GPU 窗口紧、只能在有限 run 上做粗估，那么 Kaplan 式切片拟合仍然有价值，但要明确它更像“早期规划工具”，不是精确物理定律。

适用边界还包括三条。

第一，架构要相对稳定。Transformer、MoE、检索增强模型的缩放行为可能不同，旧指数不能直接平移。

第二，数据分布要基本一致。如果小模型用的是高质量精选集，大模型被迫吃进更杂的数据，指数会变。

第三，外推距离要受控。经验上，10× 内的预测比 100× 外推可靠得多。超过 100× 时，建议至少做两件事：一是报告保守区间，二是用多组拟合方法交叉验证，而不是相信单条回归线。

---

## 参考资料

1. OpenAI, *Scaling laws for neural language models*. 用途：Kaplan 路线的原始论文入口，给出语言模型损失随模型规模、数据规模、算力变化的幂律框架。  
   https://openai.com/index/scaling-laws-for-neural-language-models/

2. NeurIPS 2022, *An empirical analysis of compute-optimal large language model training*. 用途：Chinchilla 原始论文入口，给出“参数和训练 token 应近似等比例增长”的核心结论，并展示 Chinchilla 与 Gopher 的同算力对比。  
   https://proceedings.neurips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract-Conference.html

3. Emergent Mind, *Compute-Optimal Scaling Laws*. 用途：汇总 $L(N,D)$ 形式、常见指数和 compute-optimal 指数表达，便于快速对照不同论文口径。  
   https://www.emergentmind.com/topics/compute-optimal-scaling-laws

4. Michael Brenndoerfer, *Predicting Model Performance: Scaling Laws & Forecasting*. 用途：总结扩展定律在外推上的经验边界，特别是“10× 内较稳、100× 外推风险显著升高”的实践提醒。  
   https://mbrenndoerfer.com/writing/predicting-model-performance-scaling-laws

5. Takara TLDR, *Training Compute-Optimal Large Language Models*. 用途：快速查看 Chinchilla 论文摘要、70B 参数与 1.4T token 的关键实验设置。  
   https://tldr.takara.ai/p/2203.15556
