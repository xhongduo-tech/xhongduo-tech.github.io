## 核心结论

微调超参数搜索，指的是在固定数据、固定基座模型、固定训练预算下，把一组训练参数当作待优化变量，直接用验证集指标选出最优组合。形式化地说，若超参数向量记为 $\lambda$，最优目标可以写成：

$$
\lambda^* = \arg\max_\lambda S_{val}(\text{train}(\theta; \lambda))
$$

其中 $S_{val}$ 是验证集分数，白话解释就是“训练完以后，不看你训练集背得多熟，只看你在没见过的数据上表现多好”。

在 LoRA 微调里，最常见、也最值得优先搜索的四个参数是学习率 $\eta$、batch size $m$、LoRA rank $r$、dropout $p$。它们不是附属细节，而是结果本身的一部分。只盯着单个参数看，往往会得出错误结论，因为这四个量存在明显耦合。

| 超参数 | 作用 | 直接影响 |
|---|---|---|
| 学习率 $\eta$ | 每次更新走多大步 | 收敛速度、是否震荡或发散 |
| batch size $m$ | 每次用多少样本估计梯度 | 梯度噪声、显存占用、可承受学习率 |
| LoRA rank $r$ | 低秩适配器的容量 | 可学习信息量、参数量、过拟合风险 |
| dropout $p$ | 训练时随机屏蔽一部分激活 | 正则强度、泛化能力、欠拟合风险 |

玩具例子：同一个分类任务里，$\eta=2\times10^{-4}$ 在 $m=8$ 时可能震荡，但把 $m$ 改成 $16$ 后，梯度更稳定，这个学习率反而变得可用。再把 $r$ 从 $8$ 提到 $16$，模型容量增加，验证集可能上涨；但如果不配合稍大的 $p$，也可能很快过拟合。

实际建议很明确：先用随机搜索或小网格建立基线，再用 Bayesian optimization，也就是“根据历史试验结果决定下一组试验”的方法，缩小搜索范围。对大模型，优先在更便宜的小模型上找稳定区间，再迁移到目标模型复验。

---

## 问题定义与边界

本文讨论的搜索对象定义为：

$$
\lambda = (\eta, m, r, p)
$$

其中：

- 学习率 $\eta$：控制参数更新步长，白话解释就是“每次改模型时下手多重”。
- batch size $m$：每步训练使用的样本数，白话解释就是“每次做决定前看多少例子”。
- LoRA rank $r$：低秩分解的维度，白话解释就是“给适配器多大表达空间”。
- dropout $p$：随机丢弃激活的概率，白话解释就是“训练时故意打乱一点，防止死记硬背”。

优化目标通常写成最小化验证损失：

$$
\lambda^* = \arg\min_\lambda L_{val}(\theta_\lambda)
$$

这里的边界必须讲清楚。超参数搜索成立的前提是：基座模型固定、训练数据固定、验证集固定、训练预算固定。如果你同时换了模型架构、tokenizer、优化器，或者改了数据清洗规则，那就不是单纯的超参数搜索，而是在比较不同训练系统。

| 参数名 | 含义 | 变化会影响什么 | 常见误区 |
|---|---|---|---|
| $\eta$ | 更新步长 | 收敛速度与稳定性 | 只在线性区间微调，不做对数采样 |
| $m$ | 小批量大小 | 梯度方差、吞吐、显存 | 只当作“显存参数”，忽略对最优学习率的影响 |
| $r$ | LoRA 容量 | 表达能力、参数量 | 误以为越大越好 |
| $p$ | 正则强度 | 泛化与欠拟合 | 训练集分数高就把它调成 0 |

新手最容易混淆的一点是：搜索结论只在当前条件下有效。比如你在中文客服分类任务上，用 AdamW、固定 3 个 epoch、固定验证集找到一组最佳参数；当你把任务换成指令跟随生成，或者把基座从 1.8B 换成 7B，这组参数最多算“可参考起点”，不能当成通用最优解。

---

## 核心机制与推导

超参数搜索是一个外层优化问题。内层做的是训练，外层做的是“试哪组参数更值得”。机制链条可以写成：

外层搜索策略 $\rightarrow$ 内层训练 $\rightarrow$ 验证集打分 $\rightarrow$ 更新下一次搜索策略

先看 batch size。小批量梯度写成：

$$
\hat{g}_m = \frac{1}{m}\sum_i \nabla \ell_i
$$

其方差通常满足：

$$
Var(\hat{g}_m) \propto \frac{1}{m}
$$

这句话的意思很直接：$m$ 越大，梯度估计通常越稳，训练抖动越小。所以当 $m$ 从 8 改成 16 时，模型往往能承受更大的 $\eta$。这就是为什么“只调学习率，不调 batch size”经常会把结论调偏。

再看 LoRA。LoRA 是低秩适配，白话解释就是“不直接改整块大矩阵，而是只学一个更小的补丁”。其形式为：

$$
W' = W + \Delta W
$$

$$
\Delta W = (\alpha / r)BA
$$

其中 $A$ 和 $B$ 是可训练的小矩阵，$r$ 控制低秩空间大小。$r$ 越小，参数越省、训练越便宜，但容量也越受限；$r$ 越大，适配器更能表达任务信息，但过拟合风险和显存开销也更高。

dropout 的机制则是：

$$
z = m \odot h,\quad m \sim Bernoulli(1-p)
$$

含义是训练时随机屏蔽一部分激活。$p$ 越大，正则越强，模型越不容易把训练样本死记下来；但过大又会让可用容量下降，表现成欠拟合。

把这四个参数放在一起，就能看出耦合关系：

| 参数 | 对收敛的影响 | 对稳定性的影响 | 对容量的影响 | 对泛化的影响 |
|---|---|---|---|---|
| $\eta$ | 决定快慢 | 过大易发散 | 不直接改变容量 | 间接影响最优点质量 |
| $m$ | 较大时常更平滑 | 方差下降 | 不直接改变容量 | 影响可承受的 $\eta$ |
| $r$ | 容量大时更易拟合任务 | 过大可能更敏感 | 直接增加容量 | 过大时可能过拟合 |
| $p$ | 过大时收敛变慢 | 可减少过拟合抖动 | 降低有效容量 | 适度时提升泛化 |

玩具例子：假设验证 F1 如下。

| 试验 | $\eta$ | $m$ | $r$ | $p$ | 验证 F1 |
|---|---:|---:|---:|---:|---:|
| A | $1e{-4}$ | 8 | 8 | 0.05 | 84.1 |
| B | $2e{-4}$ | 8 | 16 | 0.05 | 84.8 |
| C | $2e{-4}$ | 16 | 16 | 0.10 | 85.2 |

如果你只扫 $r$，可能以为“16 一定优于 8”；如果只扫 $\eta$，可能误判“$2e{-4}$ 就是最佳”；但真正有效的是组合 $(\eta,m,r,p)$ 一起变化后的结果，C 的优势来自联动，而不是某个参数单独变好。

真实工程例子：做中文客服问答分类，基座是 7B 模型，单卡 24GB，只能使用 LoRA。常见做法不是直接在 7B 上暴力试几十组，而是先在同架构 1B 到 3B 代理模型上搜索 $\eta \in [1e^{-5}, 5e^{-4}]$、$m \in \{4,8,16\}$、$r \in \{8,16,32\}$、$p \in \{0,0.05,0.1\}$，找出稳定区间，再迁移到 7B 上做少量复搜。原因不是“小模型结果一定等于大模型结果”，而是很多趋势具有可迁移性，能先排除明显差的区域。

---

## 代码实现

工程上最小可行框架只有四步：定义搜索空间、跑一次训练、在验证集打分、把结果回报给搜索器。重点不是把代码写得多复杂，而是让每次 trial 可比较、可复现。

先看一个可运行的 Python 玩具版本。它不依赖深度学习库，只模拟“参数越接近目标区间，得分越高”的搜索逻辑，方便理解流程。

```python
import math
from itertools import product

def evaluate_config(lr, batch_size, rank, dropout):
    # 一个玩具打分函数：离设定的“好区域”越近，分数越高
    score = 0.0
    score -= abs(math.log10(lr) - math.log10(2e-4)) * 8
    score -= abs(batch_size - 16) * 0.6
    score -= abs(rank - 16) * 0.25
    score -= abs(dropout - 0.1) * 40
    return round(85 + score, 3)

def grid_search():
    lrs = [1e-5, 5e-5, 1e-4, 2e-4]
    batch_sizes = [4, 8, 16]
    ranks = [8, 16, 32]
    dropouts = [0.0, 0.05, 0.1]

    best_cfg = None
    best_score = float("-inf")

    for lr, bs, r, p in product(lrs, batch_sizes, ranks, dropouts):
        score = evaluate_config(lr, bs, r, p)
        if score > best_score:
            best_score = score
            best_cfg = {"lr": lr, "batch_size": bs, "rank": r, "dropout": p}

    return best_cfg, best_score

best_cfg, best_score = grid_search()

assert best_cfg["batch_size"] == 16
assert best_cfg["rank"] == 16
assert abs(best_cfg["dropout"] - 0.1) < 1e-12
assert best_score > 84.0

print(best_cfg, best_score)
```

上面这段代码体现的不是深度学习细节，而是 HPO 的基本结构：固定候选空间，逐组评估，用验证分数选最好。

如果落到 Hugging Face `Trainer` + Optuna，结构通常如下：

```python
import optuna
from transformers import Trainer, TrainingArguments

def define_search_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [4, 8, 16]
        ),
        "lora_r": trial.suggest_categorical("lora_r", [8, 16, 32]),
        "lora_dropout": trial.suggest_categorical("lora_dropout", [0.0, 0.05, 0.1]),
    }

def train_one_trial(model_init, train_dataset, eval_dataset, trial):
    params = define_search_space(trial)

    model = model_init(
        lora_r=params["lora_r"],
        lora_dropout=params["lora_dropout"],
    )

    args = TrainingArguments(
        output_dir=f"./outputs/trial_{trial.number}",
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params["per_device_train_batch_size"],
        evaluation_strategy="epoch",
        save_strategy="no",
        num_train_epochs=3,
        logging_steps=20,
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics["eval_f1"]

def objective(trial):
    score = train_one_trial(model_init, train_dataset, eval_dataset, trial)
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print(study.best_trial.params)
print(study.best_trial.value)
```

搜索空间示例如下：

| 参数 | 搜索空间 | 建议 |
|---|---|---|
| $\eta$ | $[1e^{-5}, 5e^{-4}]$ | 用对数采样 |
| $m$ | $\{4, 8, 16\}$ | 受显存约束，离散枚举 |
| $r$ | $\{8, 16, 32\}$ | 从低到中等容量开始 |
| $p$ | $\{0, 0.05, 0.1\}$ | 小范围离散尝试 |

一个实用伪代码是：

```text
for trial in searcher:
    cfg = define_search_space(trial)
    train_one_trial(cfg, fixed_budget=True)
    score = evaluate_on_dev(cfg)
    report_to_optimizer(trial, score)
select_best_cfg()
```

这里最重要的工程纪律有两条。

第一，统一训练预算。所有 trial 要么都跑相同步数，要么都跑相同 token 数，要么都跑相同墙钟时间。否则“谁更好”这个问题就不成立。

第二，记录完整配置。除了 $\eta,m,r,p$，还要记录随机种子、数据切分、基座模型版本、LoRA target modules、warmup、weight decay。因为你最终要复现实验，而不是只在屏幕上看到一次好看的分数。

---

## 工程权衡与常见坑

超参数搜索的最大现实约束是预算。四个参数各取 5 档，完整网格就是 $5^4=625$ 次试验。对大模型来说，这通常不可接受。因此工程问题不是“怎样搜得最全”，而是“怎样用有限预算搜得最值”。

先看几种常见方法的预算差异：

| 方法 | 预算利用率 | 优点 | 缺点 |
|---|---|---|---|
| 网格搜索 | 低到中 | 简单、可解释 | 高维下爆炸，浪费在不重要维度 |
| 随机搜索 | 中到高 | 高维更有效，起点成本低 | 不利用历史结果 |
| Bayesian optimization | 高 | 会把试验集中到高潜力区域 | 实现更复杂，对噪声敏感 |

常见坑基本都来自“不可比”或“错误归因”。

| 常见坑 | 为什么错 | 如何规避 |
|---|---|---|
| 只调 $\eta$ 不调 $m$ | $m$ 改变梯度方差，最优 $\eta$ 常随之变化 | 至少把 $\eta$ 和 $m$ 联合搜索 |
| $\eta$ 用线性网格 | 学习率跨数量级变化，线性刻度采样效率差 | 用对数采样或对数网格 |
| 网格维度过多 | 组合数爆炸，预算被平均摊薄 | 先随机搜索，再局部精搜 |
| 训练预算不统一 | 多训练的试验天然更容易高分 | 固定步数、token 数或时间 |
| 只看训练集 | 训练分高不代表泛化好 | 以验证集指标为唯一选择标准 |
| 小模型结果盲迁移 | 模型尺度变化会改变最优区间 | 迁移后在目标模型上做少量复搜 |
| $r$ 盲目加大 | 容量增加也会抬高过拟合和显存成本 | 从 8/16/32 这类小集合开始试 |

真实工程里还有两个隐蔽坑。

一个是验证集噪声太大。比如验证集只有几百条，且类别分布不稳，那么你看到的 0.3 或 0.5 个 F1 提升，可能只是抽样波动。这时不该急着得出“参数 B 优于参数 A”的结论，而应增加验证集规模、固定多次 seed，或者至少复跑最佳候选。

另一个是把 trial 之间的训练条件偷偷改了。比如某轮试验因为显存不够，把梯度累积步数、混合精度、截断长度一起改了；表面上你只改了 $m$，实际已经改了有效 batch 和计算图行为。这个结果就不能直接跟其他 trial 横向比较。

---

## 替代方案与适用边界

不是所有任务都值得做完整 HPO。是否搜索，首先取决于预算、噪声和迁移价值。

| 方法 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 网格搜索 | 简单透明 | 组合爆炸 | 参数很少、每维候选极少 |
| 随机搜索 | 在高维下通常优于粗网格 | 不能主动利用历史结果 | 预算很低，先找可用区域 |
| Bayesian optimization | 样本效率高 | 需要搜索器与稳定打分 | 预算中等，希望少试几次 |
| 小模型到大模型迁移 | 大幅降低前期成本 | 迁移不一定精确 | 同架构、同数据分布、多尺度模型族 |

什么时候不该大规模搜：

- 数据量太少。验证集本身不稳定时，搜得越精细，越可能是在拟合噪声。
- 训练成本过高。单次 trial 就要几小时甚至几天时，先用经验默认值和少量随机试验更现实。
- 目标模型与代理模型差异太大。比如从 encoder-only 模型迁移到 decoder-only 模型，或者 tokenizer 完全不同，这种迁移价值有限。
- 需求不要求极致最优。若只是先把系统跑通，稳定的默认配置通常比昂贵搜索更划算。

对新手，一个实用决策规则是：

1. 实验机会少于 10 次：优先随机搜索，不做大网格。
2. 实验机会在 10 到 50 次：先随机，再用 Bayesian optimization 聚焦。
3. 已有小模型代理：先在小模型上找稳定区间，再到大模型上少量验证。
4. 验证集波动明显：先修评估，再谈搜参。

核心判断标准不是“有没有先进搜索器”，而是“当前问题值不值得把预算花在搜索上”。

---

## 参考资料

| 参考资料 | 作用 | 对应章节 |
|---|---|---|
| Bergstra & Bengio, 2012 | 说明高维下随机搜索通常比粗网格更有效 | 核心结论、工程权衡 |
| Snoek et al., 2012 | 说明 Bayesian optimization 的基本思想与优势 | 核心结论、替代方案 |
| Hu et al., 2021 | 解释 LoRA 的低秩适配机制与 rank 含义 | 核心机制与推导 |
| Hugging Face Transformers HPO 文档 | 给出 Trainer 做 HPO 的工程接口 | 代码实现 |
| Hugging Face PEFT LoRA 文档 | 给出 `r`、`lora_dropout`、`lora_alpha` 的官方定义 | 问题定义、代码实现 |
| Optuna 文档 | 给出 TPE、采样与 study 的具体用法 | 代码实现、替代方案 |
| Yang et al., 2022 | 支撑“小模型先搜，再迁移到大模型”的思路 | 核心结论、替代方案 |

1. [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html)
2. [Practical Bayesian Optimization of Machine Learning Algorithms](https://dash.harvard.edu/entities/publication/73120378-b6b3-6bd4-e053-0100007fdf3b)
3. [LoRA: Low-Rank Adaptation of Large Language Models](https://huggingface.co/papers/2106.09685)
4. [Hugging Face Transformers: Hyperparameter search](https://huggingface.co/docs/transformers/en/hpo_train)
5. [Hugging Face PEFT: LoRA docs](https://huggingface.co/docs/peft/en/package_reference/lora)
6. [Optuna Documentation](https://optuna.readthedocs.io/en/stable/index.html)
7. [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://huggingface.co/papers/2203.03466)
