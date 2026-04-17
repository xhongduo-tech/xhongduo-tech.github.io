## 核心结论

mGPT 的结果说明了一件很具体的事：只要模型在预训练阶段已经把多种语言压进同一个参数空间，英语 few-shot 示例里的“任务模式”就可能直接迁移到别的语言，哪怕这些语言没有做过专门的对齐训练。这里的“few-shot”是指只给模型看极少量示例；“跨语言迁移”是指在一种语言学到的做题方式，被另一种语言直接复用。

论文中的 mGPT-1.3B 使用单一的 100k shared BPE 词表。BPE 可以理解为“把文本切成高频子词片段”的分词方法。它在 61 种语言、25 个语系上预训练，随后发现一个涌现现象：英文 prompt 和英文示例，不只是能做英语任务，还能把“判断标签、补全答案、跟随格式”的行为模式迁到低资源语言任务上。

这个现象不是“任何语言都同样好”，而是有明显边界。共享子词越多、脚本越接近、预训练语料越足、语系距离越近，迁移通常越稳。相反，脚本完全不同、语料极少、标签词切分得很碎时，效果会明显下降。

先看一个最小结论表：

| 结论 | 含义 | 工程解释 |
|---|---|---|
| 英语 few-shot 模式可迁移 | 英文示例能帮助俄语、印尼语、亚美尼亚语等任务推理 | 不必为每个语言单独做 prompt 调优 |
| 共享词表很关键 | 不同语言会复用部分子词或共享统计结构 | 同一 tokenizer 比“每语种一个 tokenizer”更利于迁移 |
| 迁移不是免费午餐 | 脚本、语料量、标签设计都会限制效果 | 部署前必须做语言级验证 |
| 更多示例不一定更好 | 4-shot 在部分任务上反而退化 | few-shot 需要把示例数当超参数调 |

一个容易误解的点需要先纠正。公开翻译实验里，俄语到亚美尼亚语的原始 mGPT 在英语 prompt、1-shot 设置下，BLEU 大约是 0.009；换成目标语 prompt 后，BLEU 提升到约 0.054。约 0.386 这个数字对应的是 ROUGE，不是 BLEU。这个例子仍然成立，但要把指标说对。

---

## 问题定义与边界

问题可以写成一句话：如果我只有英语 prompt 模板和少量英语示例，能不能把它们直接拿去做别的语言任务，而不再为每种低资源语言单独训练或翻译一套模板？

这里的“低资源语言”是指可用训练数据很少的语言；“对齐训练”是指显式教模型把不同语言映射到同一任务接口的训练过程。mGPT 想证明，部分情况下这一步可以省掉。

论文考察的边界主要有三层：

| 边界维度 | 范围 | 含义 |
|---|---|---|
| 预训练语言 | 61 种语言，25 个语系 | 覆盖广，但不是全球所有语言 |
| 下游评测 | 33 种语言上的 cross-lingual NLU 等任务 | 说明结论来自多任务，不是单一数据集巧合 |
| 任务类型 | 分类、序列标注、常识推理、翻译等 | 迁移不只发生在“选 A/B/C”这类简单任务 |

玩具例子可以这样理解。假设你有一个自然语言推断任务，标签只有 `entailment`、`neutral`、`contradiction`。你给模型两个英语示例：

- Premise: A man is eating. Hypothesis: A person is having lunch. Label: entailment
- Premise: A dog is flying. Hypothesis: An animal is on the ground. Label: contradiction

接着你把真正待判断的句子换成俄语，但仍然沿用英语模板和英语标签。mGPT 仍可能根据上下文选择正确标签。它未必“懂俄语标签词”，但它学会了“看完上下文后，给候选答案打分并选最合理者”的过程。

真实工程例子更直接。假设你做一个多语言客服工单分类系统，线上每天出现英语、俄语、印尼语、哈萨克语请求。传统做法往往是每种语言准备一套标注数据，或先翻译到英语再分类。mGPT 这类现象说明，你可以先用一套英语 few-shot 模板快速冷启动多语言版本，把“退款”“账号问题”“配送异常”作为统一标签空间，再按表现决定哪些语言值得额外微调。

但它的边界也很清楚：

- 它不等于“模型自动学会所有语言”。
- 它不等于“英文 prompt 永远优于目标语 prompt”。
- 它不等于“跨脚本、超低资源语言也能稳定迁移”。

---

## 核心机制与推导

mGPT 在分类任务里使用的核心决策规则很朴素。给定输入上下文 $x$ 和候选标签集合 $\mathcal{Y}$，对每个候选标签构造一个完整 prompt，然后计算该 prompt 中标签部分 token 的负对数概率和，选择最小者：

$$
\hat{y}=\arg\min_{y\in\mathcal{Y}} \sum_{t\in \text{prompt}_y} -\log P(t \mid x, \text{demo}, t_{<})
$$

这里的“负对数概率”可以理解为“模型觉得这个 token 有多不自然”。总和越小，说明模型越相信这条标签续写。

为什么这个规则会产生跨语言迁移？关键不是“英语标签 magically 变成了俄语标签”，而是下面三件事同时成立。

第一，共享词表让不同语言进入同一套离散符号系统。shared BPE 的含义不是所有语言都切成一样的词，而是很多高频片段、标点、数字、专名、拉丁字符、借词、甚至部分形态变化，会落到重叠或相邻的子词空间里。这相当于给模型提供了跨语言可复用的积木。

第二，自回归模型学到的是“续写分布”，不是固定分类头。自回归的白话解释是“模型每次预测下一个 token”。当 few-shot 示例展示了任务格式后，模型不是调用一个专门的俄语分类器，而是在统一的生成空间里继续补全最像答案的 token 序列。因此，任务模式本身可以跨语言复用。

第三，预训练中的语料不平衡既是问题，也是迁移来源。高资源语言在参数里塑造了更稳定的“任务接口”；低资源语言虽然数据少，但只要它们共享脚本、共享子词、共享一些句法或语义结构，就可能被这个接口“带起来”。这就是所谓“意外跨语言泛化”的一个合理假说：模型并没有显式学会“把英语规则翻译过去”，而是在统一参数空间里形成了语言无关的局部推理模板。

论文还做了相关性分析，发现脚本和模型规模与表现存在显著关系，而预训练语料量与效果也有关联，只是线性相关未必总是很强。对初学者来说，可以记成一句工程判断：共享脚本和共享子词通常比“语言名字像不像”更重要。

下面用一个极小的打分例子说明。假设候选标签只有 `yes` 和 `no`。模型看到上下文后，对 `yes` 的 token 概率更高：

| 候选标签 | token 序列 | 负 log 概率和 | 结果 |
|---|---|---:|---|
| `yes` | `["yes"]` | 0.22 | 选中 |
| `no` | `["no"]` | 1.61 | 落选 |

如果目标输入换成低资源语言，但上下文结构、示例格式和标签决策模式仍然可复用，模型就可能继续把 `yes` 打到更低的损失。它不是在“先翻译再分类”，而是在同一个条件生成框架里直接比较候选答案。

---

## 代码实现

下面的代码不是完整 mGPT 推理器，而是把论文里的 scoring 思路缩成一个可运行的玩具版。它展示了 three-step 流程：构造候选 prompt、累加标签 token 的负 log 概率、取最小值。

```python
import math

def score_candidate(log_probs):
    # 输入是候选标签各 token 的对数概率
    return -sum(log_probs)

def predict(candidates):
    scores = {label: score_candidate(lp) for label, lp in candidates.items()}
    return min(scores, key=scores.get), scores

# 玩具例子：英语示例引导出的任务模式，被复用于另一种语言输入
candidates = {
    "entailment": [math.log(0.80)],
    "neutral": [math.log(0.15)],
    "contradiction": [math.log(0.05)],
}

label, scores = predict(candidates)

assert label == "entailment"
assert scores["entailment"] < scores["neutral"] < scores["contradiction"]

print(label)
print(scores)
```

真实工程里的流程会多两步。

第一步，使用共享 tokenizer。tokenizer 可以理解为“把字符串切成模型认识的 token 编号”的工具。英文 prompt、英文 demonstrations、俄语或亚美尼亚语输入，都必须走同一套词表，否则你就失去了共享子词带来的跨语言桥梁。

第二步，只对候选答案部分计分，而不是对整个 prompt 计分。因为前面的示例和上下文对所有候选都相同，真正决定类别的是标签续写那几段 token。

伪代码如下：

```python
def build_prompt(demos_en, input_text, candidate_label):
    prompt = ""
    for d in demos_en:
        prompt += f"Text: {d['text']}\nLabel: {d['label']}\n\n"
    prompt += f"Text: {input_text}\nLabel: {candidate_label}"
    return prompt

def classify_with_lm(model, tokenizer, demos_en, input_text, labels):
    best_label = None
    best_score = float("inf")

    for label in labels:
        prompt = build_prompt(demos_en, input_text, label)
        input_ids = tokenizer.encode(prompt)

        # label_span 表示“最后这个候选标签”对应的 token 范围
        label_span = tokenizer.encode(label)
        log_probs = model.token_log_probs(input_ids)

        # 只累加标签 token 的负 log 概率
        score = 0.0
        for i in range(len(label_span)):
            score += -log_probs[-len(label_span) + i]

        if score < best_score:
            best_score = score
            best_label = label

    return best_label
```

真实工程例子可以是多语言 NER 或工单分类服务。你保留一套英语 demonstrations，例如：

- “Order arrived damaged” -> `delivery_issue`
- “I was charged twice” -> `payment_issue`

上线时输入可以是西班牙语、俄语、印尼语文本，只要标签空间固定，模型就能按相同 scoring loop 决策。这样做的价值不是拿到最终最优精度，而是用极低成本把多语言链路先跑起来。

---

## 工程权衡与常见坑

第一个坑是示例数。很多人默认 few-shot “越多越好”，但 mGPT 的实验说明并不是这样。更多示例会让 prompt 变长、噪声变多、语言分布更复杂，最后把本来脆弱的跨语言泛化压坏。尤其在低资源语言上，1-shot 或 2-shot 可能比 4-shot 更稳。

第二个坑是脚本差异。脚本就是文字系统，例如 Latin、Cyrillic、Arabic。模型即使参数共享，如果目标语言大部分 token 都被切成低频碎片，负 log 概率就会上升，分类和生成都会变差。

第三个坑是标签词设计。很多初学者只关心输入语言，忽略输出标签本身。如果标签选成极不常见、被切分很碎、或语义很接近的词，评分会很不稳定。工程上常常要测试 `yes/no`、`true/false`、数字标签、或短英文标签哪种更稳。

第四个坑是把“翻译实验”理解成“翻译质量已经够用”。俄语到亚美尼亚语的结果恰好说明相反：英语 prompt 虽然能唤醒目标语言输出，但原始 mGPT 在该设置下 BLEU 仍然很低。目标语 prompt 能显著改善，但这更像“说明迁移存在”，不是“说明可直接上线”。

下面给一个趋势表：

| 因素 | 常见趋势 | 工程含义 |
|---|---|---|
| 示例数从 0 到 1 | 常有明显提升 | few-shot 确实能激活任务模式 |
| 示例数从 1 到 4 | 可能提升，也可能退化 | 必须逐语言调 |
| 共享脚本更多 | perplexity 往往更低 | 更适合直接迁移 |
| 语料更大 | 通常更稳，但不是充分条件 | 高资源语言更像“支点” |
| hate-speech 等高噪声任务 | 可能接近随机 | 不要把成功经验外推到所有任务 |

一个实用 checklist：

| 检查项 | 为什么要看 |
|---|---|
| 目标语言是否与高资源语言共享脚本 | 决定 tokenizer 是否能有效复用 |
| 标签词是否短、稳定、易分词 | 决定 scoring 噪声 |
| 1-shot、2-shot、4-shot 是否都试过 | few-shot 数量常是决定性超参数 |
| 输入是否包含大量专名、URL、代码混杂文本 | 这些内容会改变 token 分布 |
| 是否需要把 prompt 翻成目标语言再试一次 | 英语模板不是总最优 |

---

## 替代方案与适用边界

如果把 mGPT 的英语 few-shot 迁移当成一种低成本启动方案，那么至少还有三种常见替代路径。

| 方案 | 实现难度 | 多语种精度上限 | 数据需求 | 适用场景 |
|---|---|---|---|---|
| 直接用英语 prompt | 低 | 中 | 低 | 先验证任务是否可迁移 |
| 改成目标语 prompt | 中 | 中到高 | 低到中 | 目标语言已有母语模板能力 |
| 翻译桥接到英语 | 中 | 中 | 中 | 现成英文链路很强，但目标语模型弱 |
| 多语种微调 | 高 | 高 | 高 | 业务稳定、值得长期投入 |

什么时候优先选英语 prompt？

- 目标语言与高资源语言共享较多子词。
- 任务是分类、排序、抽取这类格式稳定任务。
- 你手上只有英语 demonstrations，没有目标语标注集。
- 你当前目标是冷启动，而不是最终 SOTA。

什么时候不该强行用英语 prompt？

- 目标语言脚本与训练主力语言差异很大。
- 任务输出需要高质量自然生成，而不是短标签。
- 业务容错率低，例如审核、医疗、法务场景。
- 你已经发现目标语 prompt 明显更优。

还是用翻译任务举一个边界例子。若目标语言既低资源，又与主流语言脚本完全不同，那么英语 prompt 即使能“产生点目标语输出”，也不代表质量可接受。这时优先级通常是：先试目标语 prompt；如果仍差，再试翻译桥接；只有在成本极端受限时，才把英语 prompt 直接上线做弱基线。

所以，mGPT 的真正价值不是“证明英语万能”，而是给出一个更实用的判断框架：在共享 tokenizer、多语预训练、任务格式稳定这三个条件同时存在时，英语 few-shot 示例可以成为多语言系统的最小可用接口。

---

## 参考资料

1. Shliazhko, O., Fenogenova, A., Tikhonova, M., Kozlova, A., Mikhailov, V., Shavrina, T. “mGPT: Few-Shot Learners Go Multilingual”, TACL 2024. https://aclanthology.org/2024.tacl-1.4/  
2. 论文 PDF. https://aclanthology.org/2024.tacl-1.4.pdf  
3. SberDevices 在 Habr 的实验文章，包含英语 prompt 与目标语 prompt 的翻译对比表。https://habr.com/ru/companies/sberdevices/articles/755108/
