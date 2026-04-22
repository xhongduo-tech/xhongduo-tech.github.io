## 核心结论

持续预训练（Continued Pretraining）是在通用预训练模型的基础上，继续用领域无标注语料执行同类自监督训练，让模型更适应某个领域的数据分布。

它不是重新训练一个模型，也不是给模型直接灌入某个任务的答案。更准确地说，它是在模型已经学会通用语言能力之后，再拿医疗、法律、代码、金融研报等专业文本继续训练一段时间，使模型更熟悉这些领域的词汇、表达方式、格式习惯和局部知识。

新手版可以理解为：先让模型学会中文，再拿医疗、法律、代码等专业语料继续补课。补课之后，模型更容易读懂“主诉”“鉴别诊断”“违约责任”“HTTP 429”“反向传播”这类领域表达。

| 维度 | 收益 | 风险 |
|---|---|---|
| 领域词汇 | 更理解专业缩写、术语、固定表达 | 学到错误缩写、乱码、模板噪声 |
| 领域句式 | 更适应病历、合同、代码注释等文本结构 | 对普通对话、通用写作能力有损伤 |
| 局部知识 | 更容易生成符合领域分布的内容 | 可能把过时或重复内容当成高频知识 |
| 下游任务 | 后续微调需要更少标注数据 | 训练过强会导致灾难性遗忘 |

灾难性遗忘（catastrophic forgetting）指模型学习新分布时，原来已经具备的通用能力明显下降。持续预训练的核心工程问题就是：领域能力要涨，但通用能力不能掉得太多。

---

## 问题定义与边界

领域适应（domain adaptation）是让模型从原来的数据分布迁移到目标领域数据分布。持续预训练是实现领域适应的一种方法，目标是缩小通用预训练语料与目标领域语料之间的分布差异。

这里的“数据分布”可以用白话理解为：文本长什么样、常出现哪些词、句子如何组织、问题通常怎么表达。通用模型会说中文，但不一定懂病历缩写、合同条款或 API 文档的固定写法，因此需要领域适应。

持续预训练解决的是“模型见过的数据和目标场景不一致”的问题，不直接解决“模型结构太弱”“任务标签太少”“实时知识缺失”这些问题。

| 方法 | 主要目标 | 训练数据 | 是否需要标签 | 典型用途 |
|---|---|---|---|---|
| 持续预训练 | 适应领域文本分布 | 领域无标注语料 | 不需要 | 医疗文本、法律文书、代码语料适应 |
| 监督微调 | 学会具体任务映射 | 输入-输出标注样本 | 需要 | 分类、抽取、摘要、问答 |
| 指令微调 | 学会按指令回答 | 指令-回答数据 | 通常需要 | 对话助手、任务泛化 |
| 检索增强 | 引入外部知识 | 文档库与检索结果 | 不一定 | 实时知识、企业知识库问答 |

DAPT（Domain-Adaptive Pretraining）指面向某个领域继续预训练，例如用大量医学论文继续训练通用语言模型。TAPT（Task-Adaptive Pretraining）指面向某个具体任务的数据继续预训练，例如在情感分类任务的无标注评论文本上继续训练。二者都属于领域适应路径，只是适应范围不同。

玩具例子：一个通用模型读到“BP 120/80，HR 76，否认胸痛”时，可能只把它当普通英文缩写混杂文本；经过医疗语料持续预训练后，它更可能把 BP 理解为血压，HR 理解为心率，并学到病历中“否认某症状”的常见表达。

真实工程例子：一家做企业合同审查的团队，使用通用中文大模型处理合同条款抽取。模型能理解一般中文，但对“不可抗力”“连带责任”“管辖法院”“违约金上限”等条款结构不稳定。团队可以先用大量历史合同、法律法规、裁判文书做持续预训练，再用少量标注样本做监督微调，通常比只做监督微调更稳。

---

## 核心机制与推导

持续预训练通常沿用原模型的自监督目标。自监督训练指模型从文本本身构造训练信号，不需要人工标注答案。对于自回归语言模型，常见目标是给定前面的 token，预测下一个 token。

设领域语料为 $D_{domain}$，模型参数为 $\theta$，一段 token 序列为 $x_1, x_2, ..., x_T$。纯领域持续预训练的损失可以写成：

$$
\mathcal{L}_{domain}(\theta) = - \mathbb{E}_{x \sim D_{domain}} \sum_{t=1}^{T} \log p_{\theta}(x_t \mid x_{<t})
$$

这表示：模型在领域文本上预测下一个 token，预测得越准，损失越低。训练完成后，模型会更偏向领域语料中的词汇和表达。

问题在于，如果只看领域数据，模型会不断向领域分布移动。领域效果可能增强，但通用能力可能下降。为了缓解这个问题，工程上常把领域语料和通用语料混合训练：

$$
\mathcal{L}_{mix}(\theta) = \alpha \mathcal{L}_{domain}(\theta) + (1-\alpha)\mathcal{L}_{general}(\theta)
$$

其中 $\alpha$ 是领域语料权重。$\alpha$ 越大，训练越偏领域；$\alpha$ 越小，训练越保守。白话说，就是训练时一部分 batch 看领域数据，一小部分 batch 仍然看通用数据。

| $\alpha$ | 训练含义 | 领域收益 | 通用能力风险 |
|---:|---|---|---|
| 0.2 | 主要保持通用分布，少量领域补充 | 慢 | 低 |
| 0.5 | 通用和领域大致平衡 | 中等 | 中等 |
| 0.8 | 主要学习领域分布 | 快 | 高 |
| 1.0 | 只看领域语料 | 最高但不稳定 | 最高 |

可以把持续预训练看成一次受控的参数迁移。模型原本的参数 $\theta_0$ 已经编码了通用语言能力；继续训练得到新参数 $\theta_1$。如果领域数据质量高、训练步数适中、混合比例合理，$\theta_1$ 会在领域任务上更好。如果训练过久或数据太窄，$\theta_1$ 会过度贴合领域语料，导致普通问答、常识推理或通用写作能力下降。

这也是为什么持续预训练通常不只看训练损失。训练损失下降只能说明模型更会预测当前训练语料，不能说明它整体更有用。至少需要同时观察领域验证集损失和通用验证集损失：前者判断领域适应是否有效，后者判断遗忘是否可控。

---

## 代码实现

持续预训练的最小闭环包括：准备语料、tokenize、构造 data collator、训练、验证、保存。代码实现的关键通常不是模型结构，而是数据管线和训练配比，尤其是采样比例、padding、max length、学习率和评估集。

| 步骤 | 输入 | 输出 | 关键点 |
|---|---|---|---|
| 语料准备 | 领域文本、通用文本 | 清洗后的文本行 | 去重、去乱码、去模板垃圾 |
| tokenize | 文本 | token id | 控制 max length |
| batch 构造 | token id 列表 | 张量 batch | 动态 padding 或定长切块 |
| 训练 | batch | loss | 控制学习率与领域比例 |
| 验证 | 领域/通用验证集 | 两组指标 | 同时监控收益和遗忘 |
| 保存 | 模型参数 | checkpoint | 保留最好版本 |

一个输入批次的玩具例子如下：

| 样本来源 | 文本 | token 后的形态 |
|---|---|---|
| 领域语料 | 患者主诉胸痛 3 小时 | `[101, 2457, 5632, ...]` |
| 通用语料 | 今天北京天气晴朗 | `[101, 791, 1921, ...]` |

下面代码不是完整训练器，而是一个可运行的最小示例，用来演示混合采样比例如何工作。真实训练时可以把这里的 batch 来源替换成 Hugging Face `datasets` 和 `Trainer`。

```python
import random

domain_texts = [
    "患者主诉胸痛三小时，既往有高血压病史。",
    "合同一方未按期履行付款义务，应承担违约责任。",
    "函数返回 HTTP 429 表示请求过于频繁。"
]

general_texts = [
    "今天的天气适合散步。",
    "这本书介绍了基础数学概念。",
    "用户可以在页面上搜索文章。"
]

def sample_mixed_batch(domain, general, alpha=0.7, batch_size=10, seed=0):
    """alpha 表示从领域语料采样的概率。"""
    random.seed(seed)
    batch = []
    source_count = {"domain": 0, "general": 0}

    for _ in range(batch_size):
        if random.random() < alpha:
            batch.append(random.choice(domain))
            source_count["domain"] += 1
        else:
            batch.append(random.choice(general))
            source_count["general"] += 1

    return batch, source_count

batch, count = sample_mixed_batch(domain_texts, general_texts, alpha=0.7, batch_size=100, seed=42)

assert len(batch) == 100
assert count["domain"] + count["general"] == 100
assert 55 <= count["domain"] <= 85
assert any("胸痛" in text or "合同" in text or "HTTP 429" in text for text in batch)

print(count)
```

Hugging Face 风格的训练框架可以写成：

```python
# 伪代码：展示结构，不直接运行
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

domain_ds = load_dataset("text", data_files="domain.txt")
general_ds = load_dataset("text", data_files="general.txt")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=2048,
    )

domain_tokens = domain_ds.map(tokenize, batched=True)
general_tokens = general_ds.map(tokenize, batched=True)

# 关键：按 alpha 控制领域和通用 batch 比例
train_ds = mix_dataset(domain_tokens, general_tokens, alpha=0.7)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset={
        "domain": domain_valid,
        "general": general_valid,
    },
    data_collator=collator,
    args=training_args,
)

trainer.train()
trainer.save_model("continued-pretrained-model")
```

真实工程中，验证集至少要分成两类：一类来自目标领域，用来确认领域损失是否下降；另一类来自通用文本或通用任务，用来确认通用能力是否下降。只保留领域最优 checkpoint 是危险的，因为它可能正好对应通用能力明显退化的阶段。

---

## 工程权衡与常见坑

工程上最常见的问题不是“训不动”，而是“领域分数涨了，通用能力掉了”。所以持续预训练必须双验证集监控。领域验证集看适应效果，通用验证集看遗忘程度。

一个简单的早停规则可以是：领域验证损失连续下降，但通用验证损失相对初始值上升超过阈值，例如 5%，就停止训练或回退到更早 checkpoint。这个阈值没有固定标准，要根据业务容忍度决定。法律审查、医疗辅助、代码生成这类高风险场景，通常不能只追求领域分数。

| 坑 | 现象 | 规避方式 |
|---|---|---|
| 领域语料太窄 | 模型只会某类格式，泛化差 | 混入多来源领域文本 |
| 重复文本过多 | loss 降得快，但实际能力不涨 | 去重、限制模板占比 |
| 脏数据太多 | 输出乱码、错误缩写、奇怪格式 | 清洗 HTML、表格残片、OCR 噪声 |
| 只看领域指标 | 领域任务变好，通用问答变差 | 同时评估领域和通用验证集 |
| 学习率过大 | 训练不稳定，能力波动明显 | 使用较小学习率和 warmup |
| 训练过久 | 灾难性遗忘加重 | 早停、降低 $\alpha$、保留 checkpoint |
| max length 不匹配 | 长文档结构学不到 | 根据领域文档长度设置切块策略 |

医疗语料是典型例子。如果病历中有大量缩写、表格、乱码、复制粘贴模板，模型不会自动区分哪些是医学知识，哪些是数据噪声。它只会根据训练目标学习高频模式。数据中“无明显异常”被模板化重复一百万次，模型就会更倾向于生成这种套话。

代码领域也类似。真实工程语料里可能有过时代码、错误注释、泄漏密钥、自动生成文件和重复依赖锁文件。如果不清洗，持续预训练会把这些内容一起学进去。结果可能是模型更熟悉代码格式，但也更容易生成旧 API、错误依赖版本或无意义样板代码。

因此，持续预训练前的数据处理通常比训练脚本更重要。最低限度要做：文件类型过滤、重复文本删除、超短和超长样本处理、隐私信息清理、乱码检测、训练集与验证集去重。对于专业领域，还需要让领域专家抽样检查语料质量。

---

## 替代方案与适用边界

持续预训练适合“希望模型整体更懂某个领域”的场景，但它不是所有问题的首选方案。如果目标只是让模型完成一个明确任务，监督微调可能更直接。如果目标是回答最新企业文档，检索增强可能更合适。如果目标是让模型更会遵循指令，指令微调更接近目标。

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| 持续预训练 | 领域语料多，想提升整体领域理解 | 不需要标签，改善底层分布适应 | 成本较高，可能遗忘 |
| 监督微调 | 任务明确，有标注数据 | 目标直接，评估清晰 | 对领域底层理解提升有限 |
| 指令微调 | 需要模型按人类指令完成任务 | 改善交互和任务格式 | 依赖高质量指令数据 |
| RAG | 知识频繁变化，需要可追溯来源 | 更新快，容易引用文档 | 不改变模型参数，检索质量很关键 |

如果你只是做病历分类，直接标注少量样本做监督微调可能更划算。因为分类任务的目标很明确，模型只需要学会从输入映射到类别。如果你要让模型整体理解医疗文本，例如病历摘要、诊断依据生成、医学问答、术语解释都要更稳，那么持续预训练更合适。

如果领域语料很少，持续预训练收益通常有限。几千条短文本不足以稳定改变模型分布，反而容易过拟合。此时更合理的路线可能是：整理少量高质量标注数据做监督微调，或者把领域文档放入检索系统，用 RAG 在推理时提供上下文。

如果领域知识变化很快，也不适合完全依赖持续预训练。比如公司内部制度、产品价格、接口状态每天变化，持续预训练更新成本高、发布周期长，还难以保证记忆准确。RAG 更适合这类场景，因为文档更新后可以直接进入检索库。

持续预训练的适用边界可以概括为三点：第一，有足够多、足够干净的领域无标注语料；第二，领域表达与通用语料确实存在明显差异；第三，业务允许通过训练、评估和回滚来管理通用能力退化风险。满足这三点时，它才是值得投入的领域适应方法。

---

## 参考资料

1. [Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964)
2. [Overcoming catastrophic forgetting in neural networks](https://www.pnas.org/doi/10.1073/pnas.1611835114)
3. [Hugging Face Transformers - Language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling)
4. [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/index)
5. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
