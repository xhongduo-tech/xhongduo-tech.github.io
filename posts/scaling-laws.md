## 核心结论

缩放定律研究的是：当模型参数量、训练数据量、训练计算量持续增加时，语言模型的验证集损失会如何变化。这里的“损失”可以先理解成“模型平均还会犯多少错”，常用指标是交叉熵 loss。

它的核心经验公式可以写成：

$$
L(N, D) \simeq A N^{-\alpha} + B D^{-\beta} + E
$$

其中：

- $N$ 是参数量，也就是模型里可学习数字的总数
- $D$ 是训练 tokens 数，也就是喂给模型的文本总量
- $E$ 是不可消除误差，可以理解为当前任务和数据下的理论下限
- $\alpha,\beta$ 是经验指数，表示“继续堆参数”或“继续堆数据”还能换来多大收益

Kaplan 等人在 2020 年发现，语言模型的 loss 随着参数量、数据量、算力增加，近似按幂律下降。幂律的白话解释是：继续加资源仍然有效，但收益会越来越慢，不是线性变好。论文中常被引用的一个量是参数缩放指数大约 $\alpha \approx 0.076$。

后来的 Chinchilla 论文进一步修正了一个关键结论：很多大模型不是“参数太少”，而是“数据太少、训练不够”。在固定训练算力下，单纯做更大的模型往往不如把一部分算力拿去增加训练 tokens。其经验结论可以概括为：

$$
N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}
$$

这里的 $C$ 是训练计算预算。白话说，最优做法不是把预算主要砸在参数上，而是让参数和数据一起增长。

一个最常用的工程近似是 Chinchilla 的 token/parameter 比例约为 $20:1$。因此，70B 参数模型的计算最优训练 tokens 大约是：

$$
D \approx 20 \times 70\text{B} = 1.4\text{T tokens}
$$

这条规则的实践意义很直接：在同样训练预算下，“稍小一些但喂更多数据的模型”，经常比“参数更大但没喂够数据的模型”更强，而且推理成本更低。

玩具例子可以这样理解。你有固定 100 点训练预算。Kaplan 风格的直觉更接近“多买更复杂的脑子”，Chinchilla 风格的直觉更接近“脑子和练习题一起加”。如果你只扩大模型，不增加足够样本，模型就像一本很厚的笔记本，但只记了很少内容，最后并不会更会做题。

---

## 问题定义与边界

缩放定律解决的问题不是“模型能不能继续变强”，而是：

在给定训练预算 $C$ 的情况下，应该如何分配参数量 $N$ 和训练数据量 $D$，使验证集损失 $L(N,D)$ 尽可能小？

对于自回归 Transformer，一个常见近似是：

$$
C \approx 6ND
$$

它表示一次完整训练的浮点计算量与参数量和 token 数近似成正比。系数 6 来自前向与反向传播的数量级估算，工程上常用于粗略预算，而不是精确记账。

因此，问题变成带约束优化：

$$
\min_{N,D} L(N,D)
\quad \text{s.t.} \quad C \approx 6ND
$$

这件事的重要性在于，算力是贵的，数据清洗也是贵的。如果配比错误，会出现两类典型浪费：

- 模型过大但训练 token 不够，表现为欠训练，很多参数没有被充分利用
- 数据很多但模型太小，表现为容量不够，模型吃不下更多模式

下面这个表先给出两种常见经验结论的差异：

| 视角 | 固定算力下最优参数量 | 固定算力下最优数据量 | 直观含义 |
| --- | --- | --- | --- |
| Kaplan 2020 | $N^* \propto C^{0.73}$ | $D^* \propto C^{0.27}$ | 更偏向堆参数，较早停止训练 |
| Chinchilla 2022 | $N^* \propto C^{0.5}$ | $D^* \propto C^{0.5}$ | 参数和数据均衡增长 |
| 工程口径 | 参数更大不一定更优 | 数据往往比想象中更缺 | 先确认是否已经“喂够” |

这里的边界也要说清楚。

第一，缩放定律主要描述的是大规模预训练阶段的统计规律，不直接保证下游任务、指令微调、RLHF 之后仍保持同样指数。

第二，公式是经验拟合，不是物理定律。换优化器、换 tokenizer、换去重策略、换数据质量，常数项会变，指数也可能偏移。

第三，论文里的“最优”通常指固定训练计算下的最优，不一定等于“固定推理成本最优”或“固定延迟最优”。训练和部署的目标不完全一样。

真实工程里，这个问题更像预算分配。你手里只有一笔“电费”。这笔钱既可以买更大的模型，也可以让模型多读更多文本。缩放定律的作用，就是帮你判断现在真正短缺的是参数，还是 tokens。

---

## 核心机制与推导

从公式出发：

$$
L(N, D) \simeq A N^{-\alpha} + B D^{-\beta} + E
$$

如果训练预算固定，且满足：

$$
C \approx 6ND
$$

那么可以把 $D$ 写成：

$$
D \approx \frac{C}{6N}
$$

代回损失公式：

$$
L(N) \simeq A N^{-\alpha} + B\left(\frac{C}{6N}\right)^{-\beta} + E
= A N^{-\alpha} + B\left(\frac{6N}{C}\right)^{\beta} + E
$$

接下来对 $N$ 求最小值，本质上是在平衡两项误差：

- 参数不足误差 $A N^{-\alpha}$
- 数据不足误差 $B D^{-\beta}$

最优点通常出现在两种误差贡献同量级的位置。直观上，如果左边很大，就说明模型太小；如果右边很大，就说明数据太少。最优解来自两边同时下降到一个平衡点。

Kaplan 的实验拟合得到一种更偏向参数的最优分配，结论近似为：

$$
N^* \propto C^{0.73}, \quad D^* \propto C^{0.27}
$$

这意味着算力增加时，更多预算应该分给更大的模型，而不是更长训练数据。

但 Chinchilla 重新做了更系统的实验后发现，许多历史大模型其实处在“参数多、训练少”的区域。它的修正结论是：

$$
N^* \propto C^{0.5}, \quad D^* \propto C^{0.5}
$$

也就是每当算力翻倍，最优参数量和最优训练 tokens 都应大致翻倍。进一步换成更工程化的表达，就是：

$$
\frac{D}{N} \approx 20
$$

这里 $D/N$ 的单位是“每个参数对应多少训练 tokens”。比如 7B 参数模型，对应大约 140B tokens；70B 参数模型，对应大约 1.4T tokens。

玩具例子：

假设有两个方案，训练预算相同。

| 方案 | 参数量 | 训练 tokens | 结果趋势 |
| --- | --- | --- | --- |
| A | 100B | 300B | 参数偏大，训练不足 |
| B | 70B | 1.4T | 更接近 Chinchilla 最优 |

方案 A 看起来“模型更大”，但如果 tokens 明显不够，很多参数只是增加了推理成本，没有转化成对应的泛化能力。方案 B 的参数更少，却可能因为训练更充分而在验证集上更好。

真实工程例子是 Chinchilla 与 Gopher 的对比。两者训练计算预算相近，但 Gopher 参数约 280B，而 Chinchilla 约 70B，后者使用了更多训练数据。结果是 Chinchilla 在多个基准上优于更大的 Gopher。这个结果改变了行业对“越大越好”的早期直觉。

再看一个经常被引用的现代例子。Llama 2 70B 预训练使用约 2T tokens，对应 token/parameter 比大约是：

$$
\frac{2\text{T}}{70\text{B}} \approx 28.6
$$

它高于 Chinchilla 的 20:1 经验值。这说明现代实践往往会略偏向更多数据，原因包括数据质量差异、训练稳定性、目标 benchmark 和后续对齐流程的不同。

---

## 代码实现

下面给出一个可运行的 Python 脚本，用来估算固定算力预算下的参数量与 tokens 分配。这里不追求论文级精确拟合，只实现两个工程上常用的近似方案：

- `kaplan`：按 $N^* \propto C^{0.73}, D^* \propto C^{0.27}$
- `chinchilla`：按 $D/N \approx 20$ 且 $C \approx 6ND$

```python
import math

def compute_optimal_allocation(C, scheme="chinchilla"):
    """
    C: training compute budget in FLOPs
    returns: (N, D), where
      N = parameter count
      D = training tokens
    """
    if C <= 0:
        raise ValueError("C must be positive")

    if scheme == "chinchilla":
        # C ≈ 6ND and D/N ≈ 20
        # => C ≈ 120 N^2
        N = math.sqrt(C / 120.0)
        D = 20.0 * N
        return N, D

    if scheme == "kaplan":
        # Use proportional exponents only, then solve scale by ND = C/6
        # N = k * C^0.73, D = m * C^0.27, and k*m = 1/6
        # choose k = m = sqrt(1/6) for a simple toy implementation
        k = math.sqrt(1.0 / 6.0)
        m = math.sqrt(1.0 / 6.0)
        N = k * (C ** 0.73)
        D = m * (C ** 0.27)

        # Renormalize so that 6ND = C exactly
        scale = math.sqrt((C / 6.0) / (N * D))
        N *= scale
        D *= scale
        return N, D

    raise ValueError("scheme must be 'chinchilla' or 'kaplan'")


def format_units(x):
    if x >= 1e12:
        return f"{x / 1e12:.2f}T"
    if x >= 1e9:
        return f"{x / 1e9:.2f}B"
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    return f"{x:.2f}"


# 70B model under Chinchilla rule needs about 1.4T tokens
N_70b = 70e9
D_70b = 20 * N_70b
assert D_70b == 1.4e12

# Check compute constraint for computed allocation
C = 120 * (70e9 ** 2)
N, D = compute_optimal_allocation(C, scheme="chinchilla")
assert abs(N - 70e9) / 70e9 < 1e-9
assert abs(D - 1.4e12) / 1.4e12 < 1e-9
assert abs(6 * N * D - C) / C < 1e-9

for budget in [1e20, 1e21, 1e22]:
    n_c, d_c = compute_optimal_allocation(budget, "chinchilla")
    n_k, d_k = compute_optimal_allocation(budget, "kaplan")
    print("budget =", f"{budget:.1e}")
    print("  chinchilla:", format_units(n_c), "params,", format_units(d_c), "tokens")
    print("  kaplan    :", format_units(n_k), "params,", format_units(d_k), "tokens")
```

如果你希望做简单可视化，可以加上这段绘图代码：

```python
import matplotlib.pyplot as plt

budgets = [10 ** x for x in range(19, 24)]
chinchilla_ns, chinchilla_ds = [], []
kaplan_ns, kaplan_ds = [], []

for c in budgets:
    n, d = compute_optimal_allocation(c, "chinchilla")
    chinchilla_ns.append(n)
    chinchilla_ds.append(d)

    n, d = compute_optimal_allocation(c, "kaplan")
    kaplan_ns.append(n)
    kaplan_ds.append(d)

plt.figure(figsize=(8, 5))
plt.loglog(budgets, chinchilla_ns, label="Chinchilla params")
plt.loglog(budgets, chinchilla_ds, label="Chinchilla tokens")
plt.loglog(budgets, kaplan_ns, "--", label="Kaplan params")
plt.loglog(budgets, kaplan_ds, "--", label="Kaplan tokens")
plt.xlabel("Training compute budget C")
plt.ylabel("Scale")
plt.legend()
plt.tight_layout()
plt.show()
```

这个脚本的用途不是替代实验，而是先做资源预估。比如你计划训练一个 30B 到 70B 级别模型，可以先把预算换成 FLOPs，再看当前数据池是否足以支撑目标参数量。

---

## 工程权衡与常见坑

缩放定律最容易被误用的地方，不在公式本身，而在把“经验前沿”当成“硬规则”。

常见坑如下：

| 坑 | 现象 | 后果 | 对策 |
| --- | --- | --- | --- |
| 套用 Kaplan 低估数据需求 | 模型越做越大，tokens 没同步增加 | 欠训练，验证集不划算 | 先检查 $D/N$ 是否过低 |
| 只看参数不看 token | PRD 里只写模型规模，不写训练时长 | 预算失控且收益差 | 参数、tokens、FLOPs 一起评审 |
| 忽略数据质量 | token 数很大，但重复、低质、噪声多 | 幂律前沿变差 | 去重、清洗、混合采样 |
| 学习率不匹配 | 模型和数据比例合理，但训练不稳定 | 无法接近理论最优 | 重新扫 learning rate 和 warmup |
| batch 过大或过小 | token 吞吐和优化噪声失衡 | 收敛效率变差 | 结合梯度噪声尺度调参 |
| 把训练最优当部署最优 | 训练更省，但推理太贵 | 线上成本爆炸 | 联合考虑推理延迟和吞吐 |

真实工程例子：

某团队有 300B tokens 的高质量语料，计划训练一个 100B 级模型，因为“参数大更先进”。按 Chinchilla 直觉，这个配置的 $D/N=3$，远低于 20，明显偏欠训练。结果通常是：训练 loss 还在下降，但数据已经消耗完；模型上线后推理成本很高，效果却不一定胜过一个 15B 到 20B、训练更充分的模型。

一个更稳的工程流程是：

1. 先估算可用高质量 tokens，而不是先拍脑袋定模型尺寸。
2. 用 $D/N \approx 20$ 做第一版 compute-optimal 草案。
3. 再检查推理预算、显存、训练时长，必要时向更小模型回退。
4. 最后用小规模试验验证学习率、batch size、数据混合比例。

这里要强调，20:1 不是万能常数。它会受到数据质量、训练目标、上下文长度、优化器设置影响。低质量语料很多时，表面 token 足够，实际有效信息密度可能不够；这时简单追求 token 数，会让你误以为自己处在最优区域。

---

## 替代方案与适用边界

Kaplan 和 Chinchilla 不是谁“彻底错误”，而是适用区域不同。

| 场景 | 特征 | 更适合的思路 |
| --- | --- | --- |
| 数据富裕 | 高质量 tokens 充足，可持续扩充 | 更接近 Chinchilla，参数和数据同步增长 |
| 数据稀缺 | 垂直领域数据少，扩充成本高 | 可适度偏向更大模型，但要加强正则化 |
| 推理成本敏感 | 线上延迟和显存很贵 | 倾向较小但训练充分的模型 |
| 研究探索 | 关注能力上限而非部署成本 | 可尝试更大参数，接受欠训练风险 |

如果你只有 100B 级别的专有 tokens，而且再收集数据很难，那么完全照搬 20:1 就不现实。比如按 20:1，100B tokens 只能支撑大约 5B 参数模型；但你的任务可能确实需要更大容量。这时可以采用“偏 Kaplan”策略：增大模型，但明确承认自己处在数据稀缺边界，并配合更强的去重、正则化、数据增广或迁移学习。

另一类替代方案，是不直接通过“更多参数 + 更多 tokens”来分配算力，而是改训练和部署形态，例如：

- 混合精度训练，降低单位 token 成本
- MoE，让总参数增大但每个 token 激活参数较少
- 剪枝与蒸馏，把训练得到的能力压缩到更小模型
- LoRA 或参数高效微调，在已有底座上做低成本适配

这些方法没有否定缩放定律，而是在问另一个问题：如果总算力不变，能否通过更聪明的计算路径，把预算用得更值。

因此可以这样记：

- 讨论“预训练最优分配”，优先看 Chinchilla
- 讨论“数据极稀缺时如何退而求其次”，Kaplan 风格结论仍有参考价值
- 讨论“上线成本”，还要再加一层推理约束，不能只盯训练最优

---

## 参考资料

- Kaplan et al., *Scaling Laws for Neural Language Models*, 2020：提出语言模型 loss 随参数量、数据量、计算量呈幂律变化，并给出固定算力下偏向大模型的最优分配结论。
- Hoffmann et al., *Training Compute-Optimal Large Language Models*, 2022：即 Chinchilla 论文，修正了“参数优先”的早期结论，强调很多大模型实际上欠训练。
- Touvron et al., *Llama 2: Open Foundation and Fine-Tuned Chat Models*, 2023：提供现代大模型训练 tokens 的工程实例，可用于理解 Chinchilla 之后行业的实际取舍。
- Emergent Mind 对 Kaplan / Chinchilla 的综述：适合快速复盘公式、指数和后续验证工作。
- TTSugriy 的 scaling laws 教程：适合从直观角度理解“固定算力下如何分参数和数据”。
- TildAlice 的 Chinchilla 复盘：适合补充常见误解、实验设置与工程影响。
