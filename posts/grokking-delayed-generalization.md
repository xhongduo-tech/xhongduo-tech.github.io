## 核心结论

Grokking 指的是一种非常反直觉的训练现象：模型先把训练集“做对”，却长时间不会做测试集，继续训练很久之后，测试性能才突然跃升。这里的“突然”不是指一步完成，而是指相对整个训练过程，泛化提升集中发生在一个很窄的后期窗口里。

更精确地说，设训练精度为 $A_{\text{train}}(t)$，测试精度为 $A_{\text{test}}(t)$，$T_{\text{fit}}$ 表示训练集几乎被完全拟合的时间，$T_{\text{grok}}$ 表示测试集性能开始明显跃升的时间，那么延迟泛化窗口定义为：

$$
\Delta_{\text{grok}} = T_{\text{grok}} - T_{\text{fit}}
$$

当 $\Delta_{\text{grok}} \gg 0$ 时，就出现了典型的 Grokking。

在 Power 等人的 modular addition 实验里，模型在小数据、强容量的条件下，训练集精度很早接近 100%，测试集却长期接近随机；继续训练一个数量级后，测试精度才从接近乱猜跃升到 95% 甚至更高。这说明模型的学习过程不是“稳定线性进步”，而是“先记忆，再在后期突然得到可泛化规则”。

对初学者，可以把它理解成两阶段：

1. 第一阶段，模型像背答案，只会做见过的题。
2. 第二阶段，模型经过长时间参数调整后，才真正学到“出题规律”。

它和“涌现能力”的联系在于：两者都体现了能力获取不是平滑增加，而是在某个条件下出现非线性跃迁。

---

## 问题定义与边界

Grokking 讨论的不是一般意义上的“训练变好”，而是一个更窄的现象：**过拟合之后的延迟泛化**。过拟合，白话讲，就是模型只会记住训练样本，却不会把规律用到新样本上。延迟泛化，白话讲，就是这种“会用规律”的能力不是一开始就出现，而是在训练后期才出现。

一个典型设置是 modular addition，也就是模加法任务。模运算，白话讲，就是结果超过某个数后重新从 0 开始计数。例如模 5 下，$3+4=2$，因为 $7 \bmod 5 = 2$。

在素数 $p=97$ 的任务中，输入是 $(a,b)$，目标是预测：

$$
y = (a+b)\bmod 97
$$

全部样本共有 $97^2=9409$ 个。如果只取其中约 40% 作为训练集，其余作为测试集，那么模型很容易先“记住训练样本”，却很难立刻学到真正的模加法规则。这是 Grokking 常见的实验土壤。

下面这张表可以帮助区分“容易发生 Grokking”和“通常不发生”的条件。

| 条件 | 更容易发生 Grokking | 通常不明显或不发生 |
|---|---|---|
| 数据规模 | 小训练集 | 大训练集或全量数据 |
| 任务类型 | 算法任务、规则任务 | 高噪声统计任务 |
| 模型容量 | 足够大，能先记忆 | 容量太小，连记忆都难 |
| 正则化 | 有 weight decay | 无 weight decay |
| 初始化 | 较大初始化更常见 | 初始化太小 |
| 训练时长 | 远超 $T_{\text{fit}}$ | 早停 |
| 曲线形态 | 训练早收敛，测试后期跃升 | 训练测试同步提升 |

这里的边界很重要。Grokking 不是“所有模型最后都会突然泛化”的普遍规律。它更像一种特定训练动力学现象，常见于：

- 数据很小
- 规则性很强
- 模型容量足够大
- 正则化与初始化配置合适
- 训练时间极长

如果把 modular addition 的训练集扩展到接近全量，测试集和训练集的分布差异就变小，模型更容易在早期直接学到通用规则，此时训练曲线和测试曲线会更同步，延迟现象可能明显减弱甚至消失。

玩具例子可以这样理解。假设只有下面几道模 5 题被放入训练集：

| 输入 | 输出 |
|---|---|
| $(0,0)$ | 0 |
| $(1,2)$ | 3 |
| $(2,2)$ | 4 |
| $(4,4)$ | 3 |

如果模型容量很大，它完全可以把这四个输入当成四个“键值对”硬记下来。这时它在训练集上能拿满分，但遇到 $(3,4)$ 时仍不会算，因为它还没学到“先求和再取模”这个规则。

---

## 核心机制与推导

理解 Grokking，关键是把“拟合训练集”和“学到可泛化规则”分开看。

一个常见解释是：优化过程存在早期偏置和晚期偏置。偏置，白话讲，就是优化器更倾向走向某一类解，而不是所有解等概率出现。

早期阶段，模型倾向于找到一个能快速把训练误差压低的解。这种解可以看作更接近 kernel 解。kernel 在这里可以粗略理解为“依赖局部相似性或样本记忆的解法”。它的好处是见效快，坏处是泛化差。

晚期阶段，在长时间训练和 weight decay 的共同作用下，参数范数会持续被压缩。weight decay，白话讲，就是训练时不断轻微惩罚过大的参数值。这样一来，优化会慢慢偏向更简单、更规则化的表示，也就是更接近 min-norm 或 max-margin 的解。min-norm 指“参数尽量小的解”，max-margin 指“分类边界尽量稳健的解”。它们往往更有机会对应真实规则，而不是样本记忆。

可以把这一转变写成一个示意：

$$
\text{early phase: memory/kernel} \;\longrightarrow\; \text{late phase: rule/margin}
$$

训练与测试精度的典型形态是：

- $A_{\text{train}}(t)$ 在较早阶段快速接近 1
- $A_{\text{test}}(t)$ 长时间停留在低水平
- 当参数进入更有泛化性的区域后，$A_{\text{test}}(t)$ 在较短时间窗内快速上升

示意图可以写成：

```text
accuracy
1.0 | train  ────────────────┐
    |                        │
0.8 |                        │         test
    |                        │        /
0.6 |                        │      /
    |                        │    /
0.4 |                        │  /
    |               test ____|_/
0.2 |
    +---------------------------------> steps (log scale)
                T_fit       T_grok
```

为什么会有这么长的滞后？直觉上看，训练误差为零时，参数空间里仍然有很多“都能做对训练集”的解。优化器先到达的，往往是容易找到的记忆解；而真正具有规则结构的解虽然泛化更好，但不一定最先被碰到。继续训练配合权重衰减，相当于在零训练误差的解集合中持续“筛选”更简单的那个，最后才表现为测试性能突增。

这也是为什么大初始化有时反而更容易观察到 Grokking。初始化，白话讲，就是训练开始前参数的初始尺度。较大的初始化可能让模型早期更容易进入记忆型区域，而晚期再在正则化压力下向泛化型区域转移，形成明显的“两阶段”对比。如果初始化太小，或者正则太弱，这个阶段切换可能根本不明显。

一个更具体的实验结论是：在 modular addition 这类任务上，训练精度达到 100% 并不代表已经学到算法。真正的算法性表示，往往要在后续训练中才逐渐形成。也就是说：

$$
A_{\text{train}}(T_{\text{fit}})\approx 1
\quad\not\Rightarrow\quad
A_{\text{test}}(T_{\text{fit}})\approx 1
$$

而是要等到：

$$
t \ge T_{\text{grok}} \gg T_{\text{fit}}
$$

测试性能才会明显提升。

真实工程例子可以类比到小样本规则学习系统。比如一个内部风控引擎要根据一组离散规则判断交易是否合法，训练数据很少，但每条规则都有明确组合结构。如果模型很大，它可能先记住训练案例的具体模式，验证集表现一般；继续训练并施加合适正则后，它才会真正学到“规则组合关系”，这时对未见组合的判断才突然变准。这和模加法不是同一个任务，但动力学相似：先记忆，后规则化。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现，目标不是完整复现论文的长时间训练，而是把任务定义、延迟指标和监控逻辑写清楚。代码包含两个部分：

1. 生成 modular addition 数据
2. 用一个“模拟日志”的方式展示如何计算 $T_{\text{fit}}$、$T_{\text{grok}}$ 和 $\Delta_{\text{grok}}$

```python
from dataclasses import dataclass

def mod_add(a: int, b: int, p: int) -> int:
    return (a + b) % p

def build_dataset(p: int):
    data = []
    for a in range(p):
        for b in range(p):
            data.append(((a, b), mod_add(a, b, p)))
    return data

@dataclass
class LogPoint:
    step: int
    train_acc: float
    test_acc: float

def find_fit_and_grok(logs, fit_threshold=0.999, grok_threshold=0.95):
    t_fit = None
    t_grok = None

    for item in logs:
        if t_fit is None and item.train_acc >= fit_threshold:
            t_fit = item.step
        if t_fit is not None and t_grok is None and item.test_acc >= grok_threshold:
            t_grok = item.step

    if t_fit is None or t_grok is None:
        raise ValueError("logs do not contain a full grokking event")

    return t_fit, t_grok, t_grok - t_fit

# 玩具例子：模 5 数据集
toy = build_dataset(5)
assert len(toy) == 25
assert mod_add(3, 4, 5) == 2
assert mod_add(4, 4, 5) == 3

# 模拟一段典型 grokking 日志
logs = [
    LogPoint(step=10**3, train_acc=0.40, test_acc=0.21),
    LogPoint(step=10**4, train_acc=0.92, test_acc=0.20),
    LogPoint(step=10**5, train_acc=1.00, test_acc=0.22),
    LogPoint(step=10**6, train_acc=1.00, test_acc=0.24),
    LogPoint(step=3*10**6, train_acc=1.00, test_acc=0.30),
    LogPoint(step=10**7, train_acc=1.00, test_acc=0.97),
]

t_fit, t_grok, delta = find_fit_and_grok(logs)
assert t_fit == 10**5
assert t_grok == 10**7
assert delta == 10**7 - 10**5

print("T_fit =", t_fit)
print("T_grok =", t_grok)
print("Delta_grok =", delta)
```

如果你真要做接近论文设定的实验，训练脚本需要明确几个工程点：

| 配置项 | 典型做法 | 作用 |
|---|---|---|
| 任务 | modular addition, $p=97$ | 构造规则明确的小算法数据集 |
| 训练集比例 | 约 40% | 给模型留下“记忆而非直接泛化”的空间 |
| 模型 | 宽 MLP 或小 Transformer | 容量要足够大 |
| 学习率 | 如 0.002 | 保持稳定优化 |
| weight decay | 如 $10^{-4}$ | 推动后期偏向更简单解 |
| 初始化 | 较大尺度 | 让阶段切换更明显 |
| 训练步数 | 远超 $T_{\text{fit}}$ | 不然观察不到延迟泛化 |
| 日志频率 | 每 $10^4$ 步记录一次 | 便于观测后期跃迁 |

训练循环的核心不是“怎么尽快停”，而是“怎么不要错过后期相变”。

```python
# 伪代码，强调监控逻辑而非框架细节

config = {
    "lr": 0.002,
    "weight_decay": 1e-4,
    "max_steps": 10_000_000,
    "log_every": 10_000,
    "fit_threshold": 0.999,
    "grok_threshold": 0.95,
    "disable_early_stopping": True,
}

T_fit = None
T_grok = None

for step in range(1, config["max_steps"] + 1):
    train_one_step()

    if step % config["log_every"] == 0:
        train_acc = evaluate(train_loader)
        test_acc = evaluate(test_loader)

        if T_fit is None and train_acc >= config["fit_threshold"]:
            T_fit = step

        if T_fit is not None and T_grok is None and test_acc >= config["grok_threshold"]:
            T_grok = step

        print(
            f"step={step} "
            f"train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f} "
            f"T_fit={T_fit} T_grok={T_grok}"
        )

if T_fit is not None and T_grok is not None:
    print("Delta_grok =", T_grok - T_fit)
```

日志模板建议直接输出对数尺度关注的关键点，例如：

```text
step=100000 train_acc=1.0000 test_acc=0.2280 T_fit=100000 T_grok=None
step=1000000 train_acc=1.0000 test_acc=0.2410 T_fit=100000 T_grok=None
step=10000000 train_acc=1.0000 test_acc=0.9730 T_fit=100000 T_grok=10000000
```

这样你不会把“训练早就满分了”误判成“实验已经结束了”。

---

## 工程权衡与常见坑

Grokking 最容易误导工程师的地方，是它和常规训练直觉相反。常规经验是：验证集不涨，就该停；训练集先满分、测试集不动，多半说明过拟合。但在 Grokking 场景里，这个判断可能恰好把最重要的后期阶段提前切掉。

下面是常见坑与对应建议。

| 常见坑 | 后果 | 建议 |
|---|---|---|
| 提前 early stopping | 永远只能看到过拟合，看不到后期泛化 | 明确关闭早停，预留长训练预算 |
| 不加 weight decay | 记忆解长期稳定，测试不跃升 | 保持小但非零的 weight decay |
| 初始化太小 | 阶段切换不明显，甚至不出现 | 尝试更大初始化尺度 |
| 数据太多 | 训练测试同步提高，看不到延迟 | 用更小训练集观察机制 |
| 日志太稀 | 错过测试集跃升窗口 | 固定步长记录，最好看 log-scale |
| 只看 loss 不看 accuracy | 看不出规则型跃迁 | 同时记录训练/测试准确率 |
| 把 Grokking 当常规策略 | 在大任务上浪费计算 | 先判断任务是否具备小数据强规则结构 |

第一个坑最常见。很多训练平台默认用 validation patience 做早停，而 Grokking 恰恰要求你在“验证集长期不动”时继续训练。也就是说，标准工程流程会系统性地把这个现象裁掉。

第二个坑是忽略正则。没有 weight decay，模型可能长期停留在记忆型解附近。此时你看到的是“训练集持续满分，测试集始终随机”，很容易误以为模型彻底失败。实际上它不是学不会，而是优化偏置没有被推向更规则化的方向。

第三个坑是误用观察尺度。因为 $T_{\text{fit}}$ 和 $T_{\text{grok}}$ 可能相差一个甚至多个数量级，用线性横轴很容易把关键变化挤在图的右侧一小段，看不清楚。工程上通常应该同时保存原始步数和对数尺度视图。

真实工程里，还要考虑成本。为了验证一个小规则任务是否会 Grokking，你可能需要比常规训练多跑 10 倍甚至更多步数。如果这只是一个生产分类器，而不是研究问题，那么这种算力投入通常不划算。也就是说，Grokking 更像是“理解训练动力学的窗口”，而不是默认推荐的生产训练流程。

---

## 替代方案与适用边界

如果你的任务满足下面几个特征，Grokking 值得观察：

- 数据量小
- 任务本质是规则学习
- 模型容量足够大，存在先记忆后规则化的空间
- 你关心的是“模型何时真正学到算法结构”，而不只是短期验证分数

如果你的任务是大规模 NLP、推荐系统、视觉分类等常规统计学习问题，通常不需要刻意等待 Grokking。原因很简单：这些场景里数据更大、噪声更高、目标更统计化，标准正则化和早停往往更有效，也更便宜。

可以用下面这张决策表做粗判断。

| 场景 | 是否建议专门观察 Grokking | 原因 |
|---|---|---|
| 小样本模运算、奇偶校验、规则组合学习 | 建议 | 延迟泛化现象更典型 |
| 研究模型是否真的学到算法 | 建议 | 可用 $\Delta_{\text{grok}}$ 度量规则获取延迟 |
| 大规模文本分类 | 不建议 | 常规泛化更重要，Grokking 不明显 |
| 工业推荐/CTR 预测 | 不建议 | 数据噪声大，长时间等待成本高 |
| 算力受限的小团队项目 | 通常不建议 | 继续长跑的性价比低 |
| 教学实验或论文复现 | 建议 | 现象清晰，便于理解隐式偏置 |

替代方案主要有三类。

第一类是传统早停。早停，白话讲，就是验证集不再提升时停止训练。它适用于绝大多数实用任务，目标是用更少计算换更稳的泛化。

第二类是数据增强和样本扩充。如果问题本质上是“训练样本太少，模型只能死记硬背”，那么增加覆盖范围通常比等待后期动力学更直接。对规则任务来说，这相当于让模型更早见到足够多的输入组合。

第三类是显式结构约束。比如把任务写成更适合规则学习的模型、加入归纳偏置、手工设计输入表示，或者直接采用符号方法与神经网络结合。这样做的核心思想是：不要等模型在很晚的训练阶段“自己悟出来”，而是尽早把正确结构放进模型里。

所以，Grokking 不是“比传统泛化更高级”的方法，而是一个提醒：**低测试分数并不总意味着模型永远学不会，也可能意味着它还停留在记忆阶段。** 但是否值得等它“悟出来”，取决于任务规模、算力预算和业务目标。

---

## 参考资料

1. Power et al., *Grokking: Generalisation beyond overfitting on small algorithmic datasets*。原始实验论文，提出 Grokking 命名，并在 modular arithmetic 等小算法数据集上展示“先过拟合、后泛化”的现象。

2. Lyu et al., *Dichotomy of early and late phase implicit biases can provably induce grokking*。机制推导论文，重点解释早期 kernel 偏置与晚期 margin/min-norm 偏置的切换，以及 weight decay、大初始化和延迟泛化之间的关系。

3. EmergentMind, *Grokking: Delayed Generalization*。社区整理材料，适合快速建立整体图景，尤其是 $\Delta_{\text{grok}}$ 这类描述延迟窗口的表达方式。
