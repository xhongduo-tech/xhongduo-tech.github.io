## 核心结论

课程学习（Curriculum Learning, CL）是一种数据排序策略：先训练容易样本，再逐步加入困难样本，最终仍然回到全量训练数据。

它不是让模型结构更复杂，也不是替代损失函数，而是改变训练过程中样本出现的顺序：

```text
D -> π -> S_t
全量数据 -> 难度排序 -> 第 t 阶段可见的训练子集
```

“容易样本”不能只按人类直觉定义。更可靠的做法是让难度分数尽量贴近模型当前能力、任务目标和数据质量。例如文本分类中，可以先训练句式简单、标签清晰的样本，再加入长句、歧义句、噪声句；但这里的“简单”最好由模型置信度、标注一致性或验证集表现来校验。

课程学习的主要价值是让优化更稳定、收敛更平滑，并在部分任务上带来小幅指标提升。它通常不是数量级提升的方法，在 NER、情感分析等任务中，更常见的是 1-3% 左右的收益。它也有明显风险：如果课程太长或太硬，模型会在早期过度适应简单样本，导致困难样本欠拟合。

易到难排序可以写成：

```text
easy       medium        hard
[0.10] -> [0.35] -> [0.70] -> [0.92]
```

课程学习与随机混洗的差异如下：

| 训练方式 | 样本顺序 | 优点 | 风险 |
|---|---:|---|---|
| 随机混洗 | 每轮随机 | 简单、稳定基线强 | 早期可能被噪声和难例干扰 |
| 课程学习 | 从易到难 | 优化更平滑，早期梯度更稳定 | 难度定义错会拖累训练 |
| 过硬课程 | 长时间只看 easy 样本 | 前期 loss 降得快 | hard 样本欠拟合 |

---

## 问题定义与边界

给定训练集：

$$
D = \{(x_i, y_i)\}_{i=1}^{n}
$$

其中 $x_i$ 是输入样本，$y_i$ 是标签。课程学习要做的第一件事，是给每个样本定义一个难度分数：

$$
d_i = g(x_i, y_i)
$$

这里的 $g$ 是难度函数，白话说就是“用某种规则判断这条样本有多难”。分数越小，表示越容易。然后按难度从小到大排序：

$$
\pi: d_{\pi(1)} \le d_{\pi(2)} \le ... \le d_{\pi(n)}
$$

$\pi$ 是排序后的样本下标序列。训练时不一定一开始使用全部样本，而是在第 $t$ 个阶段使用前 $m_t$ 个样本：

$$
S_t = \{\pi(1), ..., \pi(m_t)\}
$$

一个玩具例子：有 3 条样本，难度分数分别是 `0.2, 0.5, 0.9`。训练顺序就是先 `0.2`，再 `0.5`，最后 `0.9`。对新手来说，这等价于先让模型看最容易判断的样本，再逐步加入更容易出错的样本。

常见难度来源如下：

| 难度来源 | 难度定义 | 适用任务 | 主要风险 |
|---|---|---|---|
| 样本长度 | 文本越长越难 | 分类、NER、摘要 | 长度不一定等于难度 |
| 模型置信度 | 置信度越低越难 | 分类、抽取、匹配 | 依赖辅助模型质量 |
| 标注一致性 | 标注者越不一致越难 | 医疗、法律、内容审核 | 低一致性可能是脏数据 |
| 噪声水平 | 噪声越高越难 | 弱监督、爬虫数据 | 噪声不该直接当难例学习 |

课程学习的边界也很明确：它解决的是“训练样本如何排序或加权”的问题，不直接解决模型容量不足、标签体系错误、数据分布偏移等问题。如果困难样本本身是错标样本，把它们放到后期并不会自动修复标签，反而可能让模型在训练末期被错误信号拉偏。

---

## 核心机制与推导

课程学习可以从两个等价视角理解。

第一种是逐步扩大训练集合。第 $t$ 阶段的训练目标为：

$$
L_t(\theta) = \sum_{i \in S_t} \ell(f_\theta(x_i), y_i)
$$

其中 $\theta$ 是模型参数，$f_\theta$ 是模型，$\ell$ 是损失函数。白话说，模型在不同阶段看到的训练集不一样，早期只看容易样本，后期逐步看到更多样本。

第二种是逐步增加样本权重：

$$
L_t(\theta) = \sum_i w_i(t)\ell(f_\theta(x_i), y_i)
$$

其中 $w_i(t) \in [0,1]$。$w_i(t)$ 是第 $t$ 阶段第 $i$ 条样本的权重，白话说就是“这条样本现在对训练有多大影响”。容易样本早期权重大，困难样本后期权重逐渐变大。

Bengio 等人的原始表述可以理解为训练分布随阶段变化：

$$
Q_\lambda(z) \propto W_\lambda(z)P(z)
$$

$P(z)$ 是原始数据分布，$W_\lambda(z)$ 是课程权重函数。随着 $\lambda$ 增大，更多样本被纳入训练，最后 $Q_\lambda$ 接近原始全量数据分布。

训练阶段可以这样扩展样本集合：

```text
阶段 1: [easy]
阶段 2: [easy, medium]
阶段 3: [easy, medium, hard]
阶段 4: [all data shuffled]
```

如果一个辅助模型对三条样本金标的置信度是 `[0.95, 0.72, 0.41]`，可以定义：

$$
d_i = 1 - p_\phi(y_i|x_i)
$$

$p_\phi(y_i|x_i)$ 是辅助模型对正确标签的概率。置信度越高，难度越低。因此前期先学 `0.95` 和 `0.72`，后期再加入 `0.41`。这不是说低置信度样本不重要，而是说它们更适合在模型已有基础能力之后再加入。

难度分数可以这样计算：

| 方法 | 公式 | 含义 |
|---|---|---|
| 置信度 | $d_i = 1 - \max_k p_\phi(k|x_i)$ | 模型越没把握越难 |
| 金标概率 | $d_i = 1 - p_\phi(y_i|x_i)$ | 对正确标签越没把握越难 |
| 长度 | $d_i = |x_i|$ | 输入越长越难 |
| 一致性 | $d_i = 1 - a_i$ | 标注者越不一致越难 |

真实工程例子：在医疗 NER 中，早期可以训练“固定格式日期”“明确药名”“标准医院名”等边界清楚的实体；中期加入长病历句子；后期再加入嵌套实体、缩写、低标注一致性样本。这样做的目标是先学稳定边界，再处理复杂上下文。

---

## 代码实现

工程实现一般分成 4 步：打分、排序、分阶段采样、训练时逐步放开更难样本。关键是把课程策略做成可配置组件，而不是写死在训练循环里。

输入数据结构可以设计为：

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | string | 输入文本 |
| `label` | int/string | 标签 |
| `difficulty` | float | 难度分数，越小越容易 |
| `stage` | int | 样本首次进入训练的阶段 |

最小可运行示例：

```python
from dataclasses import dataclass
from typing import List


@dataclass
class Sample:
    text: str
    label: int
    difficulty: float
    stage: int = 0


class CurriculumScheduler:
    def __init__(self, samples: List[Sample], stages: int):
        self.samples = sorted(samples, key=lambda x: x.difficulty)
        self.stages = stages
        n = len(samples)
        for rank, sample in enumerate(self.samples):
            sample.stage = min(stages - 1, rank * stages // n)

    def visible_samples(self, epoch: int) -> List[Sample]:
        current_stage = min(self.stages - 1, epoch)
        return [s for s in self.samples if s.stage <= current_stage]


samples = [
    Sample("good movie", 1, 0.2),
    Sample("bad", 0, 0.1),
    Sample("not bad but too long and unclear", 1, 0.8),
    Sample("the plot is predictable", 0, 0.5),
]

scheduler = CurriculumScheduler(samples, stages=3)

epoch0 = scheduler.visible_samples(epoch=0)
epoch1 = scheduler.visible_samples(epoch=1)
epoch2 = scheduler.visible_samples(epoch=2)

assert [s.difficulty for s in epoch0] == [0.1, 0.2]
assert [s.difficulty for s in epoch1] == [0.1, 0.2, 0.5]
assert [s.difficulty for s in epoch2] == [0.1, 0.2, 0.5, 0.8]

for epoch in range(3):
    batch = scheduler.visible_samples(epoch)
    # 真实训练中，这里会执行 forward、loss、backward、optimizer.step
    loss = sum(s.difficulty for s in batch) / len(batch)
    print(epoch, len(batch), round(loss, 3))
```

伪训练循环可以写成：

```python
for epoch in range(num_epochs):
    train_subset = scheduler.visible_samples(epoch)
    loader = build_dataloader(train_subset, shuffle=True)

    for batch in loader:
        logits = model(batch["text"])
        loss = loss_fn(logits, batch["label"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

实际项目中，`difficulty` 可以离线预计算后写入数据文件，也可以在训练前由辅助模型生成。更推荐离线生成，因为它可复现、可审计，也方便和随机混洗基线做对比。

---

## 工程权衡与常见坑

课程学习最重要的风险是：难度定义错了，训练顺序就会错。训练顺序错了，不只是“没有收益”，还可能比随机混洗更差。

| 坑点 | 后果 | 规避办法 |
|---|---|---|
| 只按人类直觉定义难度 | 排序和模型学习状态脱节 | 用模型置信度或 dev 集误差校验 |
| 课程太长 | hard 样本训练不足 | 设置较短 warmup，后期全量混洗 |
| 课程太硬 | early stage 分布过窄 | 每阶段混入少量随机样本 |
| 把噪声当困难样本 | 后期被错标样本拉偏 | 先做去噪和一致性筛选 |
| 只按长度排序 | 长度与任务难度不一致 | 组合长度、置信度、一致性等特征 |

课程过长导致 hard 样本欠拟合，可以用下面的示意理解：

```text
epoch:        1     2     3     4     5
easy:       train train train train train
medium:       -   train train train train
hard:         -     -     -     -   train

结果：hard 样本只在最后出现，梯度更新次数不足。
```

真实工程中必须看 dev 集验证曲线，而不是只看训练 loss。合理的课程学习通常表现为：训练前期 loss 更平滑，dev 指标不低于随机混洗，并且全量阶段后 hard 子集指标没有明显落后。如果只有 easy 子集提升，而 hard 子集下降，说明课程过硬或持续时间过长。

另一个常见问题是把“晚学”误解成“重要性低”。课程学习不是降低困难样本的重要性，而是延迟它们对优化的主导作用。后期必须让模型充分接触全量样本，否则最终模型会偏向简单模式。

---

## 替代方案与适用边界

课程学习不是唯一的数据策略。它和自步学习、难例挖掘、主动学习相似，但目标不同。

自步学习（Self-paced Learning）是让模型根据当前损失自动选择样本，白话说就是“模型觉得自己能学哪些，就先学哪些”。难例挖掘（Hard Example Mining）是主动加强困难样本，白话说就是“专门补模型最容易错的地方”。主动学习（Active Learning）是选择最值得标注的样本，白话说就是“把标注预算花在最有信息量的数据上”。

| 方法 | 核心动作 | 适用阶段 | 主要目标 |
|---|---|---|---|
| 课程学习 | 预先或半预先从易到难排序 | 训练过程 | 稳定优化 |
| 自步学习 | 模型动态选择易样本 | 训练过程 | 自动调节学习节奏 |
| 难例挖掘 | 提高难样本权重 | 中后期训练 | 修复短板 |
| 随机混洗 | 每轮随机采样 | 默认基线 | 简单稳健 |

什么时候适合用课程学习：

| 场景 | 是否适合 | 原因 |
|---|---|---|
| 难度可由置信度估计 | 适合 | 排序信号可靠 |
| 标注一致性可获得 | 适合 | 能区分清晰样本和歧义样本 |
| 噪声极高且未清洗 | 谨慎 | 困难样本可能只是脏样本 |
| 长尾覆盖最重要 | 不一定 | 难例挖掘可能更直接 |
| 数据量很小 | 谨慎 | 切分阶段可能进一步减少有效样本 |

医疗 NER 中，课程学习适合先学“格式稳定、边界清楚”的实体，再学长句、嵌套实体和低一致性样本。情感分类中，可以先学情绪词明显、句子较短的样本，再学转折、反讽和长文本。二者的共同点是：早期先建立稳定的基础判别能力，后期再覆盖复杂样本。

但如果目标是尽快提升稀有实体召回率，难例挖掘可能更直接。因为课程学习强调“先打地基”，而难例挖掘强调“专门补短板”。两者可以组合：前几个 epoch 使用课程学习，后几个 epoch 使用全量混洗加 hard example reweighting。

---

## 参考资料

1. [Curriculum Learning](https://icml.cc/2009/papers/119.pdf)：Bengio 等人提出课程学习的经典论文，用于理解基本思想和训练分布变化。
2. [Self-paced Curriculum Learning](https://cdn.aaai.org/ojs/9608/9608-13-13136-1-2-20201228.pdf)：介绍自步学习和课程学习的结合方式，用于理解自动选择样本的变体。
3. [Active Curriculum Learning](https://aclanthology.org/2021.internlp-1.6/)：讨论主动学习与课程学习结合，用于理解样本选择和标注策略的关系。
4. [A Sentiwordnet Strategy for Curriculum Learning in Sentiment Analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC7298176/)：情感分析任务中的课程学习应用，用于观察文本分类场景下的经验收益。
5. [Learn Like a Pathologist: Curriculum Learning by Annotator Agreement](https://openaccess.thecvf.com/content/WACV2021/html/Wei_Learn_Like_a_Pathologist_Curriculum_Learning_by_Annotator_Agreement_for_WACV_2021_paper.html)：使用标注一致性构造课程，用于理解“难度来自标注分歧”的方法。
