## 核心结论

FLAN（Fine-tuned LAnguage Net）是一种指令微调范式：把多个 NLP 任务统一改写成“指令-输入-输出”格式，再用监督微调让语言模型学习“按自然语言要求完成任务”。

监督微调是指：给模型一批带标准答案的样本，让模型生成标准答案的概率变高。FLAN 的关键不在于训练一个只会做情感分类的模型，而在于把分类、问答、摘要、翻译、抽取等任务放进同一个训练接口，让模型学习任务意图。

传统微调更像“只针对一道题型训练”；FLAN 更像“让模型先学会读题，再做题”。这里的“读题”不是人类理解，而是模型在统计上学到：不同指令文本通常对应不同输出形式。

| 对比项 | 传统单任务微调 | FLAN 指令微调 |
|---|---|---|
| 任务范围 | 通常只服务一个任务 | 混合多个任务 |
| 输入形式 | 任务专用字段 | 自然语言指令 + 输入 |
| 输出接口 | 标签、类别或固定格式 | 统一文本生成 |
| 泛化能力 | 依赖目标任务数据 | 更关注零样本、少样本泛化 |

统一目标函数可以写成：

$$
\mathcal{L}(\theta)=\frac{1}{N}\sum_{i=1}^{N}-\log p_\theta(y_i\mid x_i)
$$

其中，$x_i$ 是“指令 + 输入”，$y_i$ 是目标输出。损失越低，表示模型越倾向于在给定指令后生成正确答案。

---

## 问题定义与边界

FLAN 解决的问题是：如何让预训练语言模型通过一轮监督微调，获得更强的指令跟随能力。预训练语言模型是先在大量文本上学习语言规律的模型，但它不一定天然知道“请总结”“请分类”“请抽取金额”这些指令应该如何执行。

零样本泛化是指：模型在没有见过某个具体测试任务训练样本的情况下，仍然能根据任务描述给出可用答案。FLAN 的核心发现是，指令微调可以显著提升这种能力，且任务种类越丰富，收益通常越明显。

但 FLAN 不是万能微调方法。它依赖高质量任务数据、合理模板、任务采样平衡和足够大的基础模型。原始 FLAN 论文也强调，模型规模和任务数量是关键因素；在较小模型上，指令微调不一定总是带来收益。

| 项目 | FLAN 的边界 |
|---|---|
| 解决什么 | 提升模型按指令完成多类任务的能力 |
| 不解决什么 | 不保证事实正确、不自动消除偏见、不替代领域数据 |
| 需要什么数据 | 多任务、有答案、可改写成指令的数据 |
| 模板依赖程度 | 高，模板会影响模型学到的任务表达 |

多任务混合训练目标可以写成：

$$
\mathcal{L}(\theta)=\sum_{t\in T}\lambda_t\;\mathbb{E}_{(x,y)\sim D_t}\big[-\log p_\theta(y\mid x)\big],\quad \sum_t \lambda_t=1
$$

这里 $T$ 是任务集合，$D_t$ 是第 $t$ 个任务的数据，$\lambda_t$ 是任务权重。任务权重决定不同任务在训练中出现的比例。

---

## 核心机制与推导

FLAN 的核心机制是统一输入接口。统一输入接口是指：不管原任务是分类、问答还是摘要，都转换成一段自然语言提示，再让模型生成文本答案。

玩具例子：

| 原始任务 | 指令模板 | 输入 | 输出 |
|---|---|---|---|
| 情感分类 | 判断下列文本情感是正面还是负面 | I love it | positive |
| 问答 | 回答下列国家的首都 | France | Paris |
| 摘要 | 用一句话总结下列文本 | A long article... | A short summary |
| 信息抽取 | 抽取句子中的金额 | The price is 30 dollars | 30 dollars |

训练时，这些样本都会变成同一种形式：

```text
指令：判断下列文本情感是正面还是负面
输入：I love it
答案：positive
```

推导上，FLAN 没有发明新的损失函数。它仍然是在最大化正确输出的条件概率：

$$
\mathcal{L}(\theta)=\frac{1}{N}\sum_{i=1}^{N}-\log p_\theta(y_i\mid x_i)
$$

区别在于，$x_i$ 不再只是原始输入，而是包含任务描述的完整提示：

$$
x_i = \text{instruction}_i + \text{input}_i
$$

如果模型看到“判断情感”时更倾向输出 `positive` 或 `negative`，看到“回答首都”时更倾向输出城市名，说明它学到了一部分任务条件分布。这里的条件分布是指：在给定输入条件下，不同输出出现的概率分布。

多任务混合比单任务更稳，是因为模型不会只把某一种输入格式和某一种输出格式绑定在一起。任务越多，模型越容易从不同样本中抽象出“指令文本决定任务类型”这一规律。

原始 FLAN 的评估还按任务簇留出测试。任务簇是语义上相近的一组任务，例如自然语言推理、阅读理解、问答。按任务簇留出比随机留出更严格，因为它减少了“训练集见过相似题型，测试集只是换个数据集”的泄漏风险。

---

## 代码实现

FLAN 式训练的工程核心是数据管线，不是特殊模型结构。先把原始任务数据转换成统一指令格式，再按普通语言模型监督微调训练。

| 步骤 | 作用 | 示例 |
|---|---|---|
| 原始任务数据 | 保留输入和答案 | 文本：I love it；标签：positive |
| 指令改写 | 加入任务说明 | 判断下列文本情感 |
| 统一拼接 | 形成训练 prompt | 判断下列文本情感：I love it |
| 批量采样 | 混合不同任务 | 分类、问答、摘要轮流出现 |
| 监督微调 | 提高目标答案概率 | 输出 positive |

下面是一个可运行的最小 Python 例子。它不训练真实神经网络，只演示 FLAN 的样本格式、损失计算和断言检查：

```python
import math

samples = [
    {
        "task": "sentiment",
        "instruction": "判断下列文本情感是正面还是负面",
        "input": "I love it",
        "output": "positive",
        "prob": 0.8,
    },
    {
        "task": "qa",
        "instruction": "回答下列国家的首都",
        "input": "France",
        "output": "Paris",
        "prob": 0.5,
    },
]

def build_prompt(sample):
    return f"{sample['instruction']}：{sample['input']}"

def average_negative_log_likelihood(items):
    return sum(-math.log(item["prob"]) for item in items) / len(items)

prompts = [build_prompt(sample) for sample in samples]
loss = average_negative_log_likelihood(samples)

assert prompts[0] == "判断下列文本情感是正面还是负面：I love it"
assert samples[1]["output"] == "Paris"
assert round(loss, 3) == 0.458

after_training = [{"prob": 0.9}, {"prob": 0.9}]
new_loss = average_negative_log_likelihood(after_training)

assert round(new_loss, 3) == 0.105
assert new_loss < loss
```

这个玩具例子说明了 FLAN 优化的对象：不是让模型记住某个字符串，而是提高“给定指令后生成正确答案”的概率。

真实工程例子：客服助手通常同时需要做意图分类、订单号抽取、投诉摘要、风险判断和回复草稿生成。如果每个任务都训练一个专用模型，系统会有多个接口、多个部署单元和多套评估流程。用 FLAN 式指令微调，可以把它们统一成：

```text
指令：抽取用户消息中的订单号
输入：我昨天买的订单 A12345 还没发货
输出：A12345
```

```text
指令：判断这条消息是否包含退款风险
输入：再不给我处理我就投诉
输出：是
```

这样新增任务时，通常可以先补充少量指令样本继续微调，或直接用已有指令模型做零样本试运行。

---

## 工程权衡与常见坑

FLAN 的效果不是“数据越多越好”，而是任务多样性、模板覆盖、采样平衡和模型规模共同决定。任务多样性是指训练集中任务类型足够丰富，不只是同一类任务换几个数据集。

| 常见坑 | 后果 | 规避方法 |
|---|---|---|
| 只用一种模板 | 模型记住固定格式，换问法掉点 | 同一任务使用多模板改写 |
| 任务不平衡 | 大任务压制小任务 | 分层采样或重加权 |
| 训练测试泄漏 | 指标虚高 | 按任务簇划分训练和测试 |
| 只堆数据量 | 低质量重复样本稀释收益 | 先保证任务覆盖，再扩数据 |
| 训练和推理提示不一致 | 零样本效果下降 | 保持指令风格稳定 |

风险优先级可以这样看：

| 风险 | 优先级 | 原因 |
|---|---:|---|
| 任务泄漏 | 高 | 会直接让评估结果失真 |
| 任务采样失衡 | 高 | 会让模型偏向大数据任务 |
| 模板单一 | 中高 | 会削弱指令泛化 |
| 输出格式不稳定 | 中 | 会增加工程解析成本 |
| 数据轻微重复 | 中低 | 影响取决于重复规模 |

一个常见错误是把所有任务都写成过于相似的模板，例如全部以“请完成任务：”开头。这样模型可能学到的是模板表面形式，而不是任务差异。更稳的做法是为同一任务准备多种表达，但保持输出规范清晰。例如情感分类可以有“判断情感”“这句话是正面还是负面”“给出情绪标签”等模板，但输出始终限制为 `positive` 或 `negative`。

---

## 替代方案与适用边界

FLAN 适合“一个模型做很多事”的场景，尤其适合任务类型多、目标任务样本少、希望提升零样本能力的系统。但如果任务边界很窄，传统方案可能更便宜。

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| 传统单任务微调 | 单一分类或抽取任务 | 简单、稳定、成本低 | 泛化到新任务能力弱 |
| 多头多任务学习 | 多任务但输出结构固定 | 各任务可独立控制 | 接口复杂，新增任务要改结构 |
| 提示词零样本/少样本 | 无训练预算或快速验证 | 不需要训练 | 对提示词敏感，稳定性有限 |
| LoRA 等参数高效微调 | 训练资源有限 | 成本低，易部署多个版本 | 仍依赖数据和模板质量 |
| FLAN 指令微调 | 多任务、通用指令跟随 | 泛化能力更强，接口统一 | 数据组织和评估更复杂 |

可以用一个简化判断式理解适用性：

$$
\text{FLAN收益} \propto \text{任务多样性} \times \text{模板质量} \times \text{模型规模}
$$

如果你只做一个固定标签分类器，例如“把日志分成 INFO、WARN、ERROR”，且有几十万条稳定标注数据，单任务微调或小模型分类器可能更直接。如果你要做客服助手、数据分析助手、内容审核助手这类多能力系统，FLAN 式指令微调更合适。

需要极低延迟时，也要谨慎选择 FLAN。统一生成式接口虽然灵活，但生成文本通常比专用分类头慢。工程上常见折中是：高频、强结构任务用专用模型；复杂、长尾、多变任务交给指令模型。

---

## 参考资料

| 名称 | 类型 | 用途 | 推荐顺序 |
|---|---|---|---:|
| Finetuned Language Models are Zero-Shot Learners | 原始论文 | 理解 FLAN 的核心实验与结论 | 1 |
| Introducing FLAN | Google Research 博客 | 快速理解动机和方法 | 2 |
| google-research/FLAN | 官方仓库 | 查看模板和任务处理方式 | 3 |
| The Flan Collection | 后续扩展论文 | 理解任务平衡、混合提示和数据设计 | 4 |

1. [Finetuned Language Models are Zero-Shot Learners](https://research.google/pubs/finetuned-language-models-are-zero-shot-learners/)
2. [Introducing FLAN: More generalizable Language Models with Instruction Fine-Tuning](https://research.google/blog/introducing-flan-more-generalizable-language-models-with-instruction-fine-tuning/)
3. [google-research/FLAN](https://github.com/google-research/FLAN)
4. [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://research.google/pubs/the-flan-collection-designing-data-and-methods-for-effective-instruction-tuning/)
