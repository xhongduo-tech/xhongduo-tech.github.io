## 核心结论

合成数据是“由程序或模型生成、但目标上尽量保留真实数据统计特征的数据”。白话讲，它不是随便伪造样本，而是用可控方式补足真实数据拿不到、拿不全、拿不起的部分。

它成立的前提，不是“模型会编数据”，而是“生成过程被真实分布约束”。如果没有这个约束，模型学到的只是另一个模型的习惯，而不是业务世界本身。

对初级工程师最重要的结论有两条：

1. 合成数据最适合做“补集”，不适合做“全集”。真实数据稀缺时，它能快速补齐长尾场景、极端事件、标注缺口，显著降低人工成本。
2. 训练时必须保留真实数据锚点，并对合成样本做质量控制。否则容易出现 Model Autophagy Disorder，简称 MAD，中文可理解为“模型自噬”：模型反复吃自己或同类模型生成的内容，信息不断变旧、变窄，最后泛化能力下降。

一个典型结果是金融情感分类。用 Mixtral-8x7B 给 `financial_phrasebank` 自动打标签，再把真实数据和合成数据一起训练 RoBERTa，小模型可做到约 94% 准确率，而训练与推理成本约 2.7 美元。相比之下，若直接依赖高价闭源大模型大规模标注，成本可能到数千美元，碳排放也更高。这里的工程含义很直接：先让大模型做“教师”，再让小模型做“学生”，通常是更可部署的路线。

| 方案 | 标注成本 | 训练成本 | 准确率 | 碳排放 |
|---|---:|---:|---:|---:|
| 人工或高价闭源模型直接大规模标注 | 很高，示例约 3061 美元 | 低到中 | 高，但依赖外部服务 | 很高，示例约 735-1100 kg |
| 仅用合成数据训练学生模型 | 低 | 低 | 初期可用，后期不稳定 | 低 |
| 真实数据 + 合成数据混合训练 | 低到中，示例约 2.7 美元总成本 | 低 | 约 94% | 很低，示例约 0.12 kg |

核心判断标准不是“是否用了合成数据”，而是“真实分布是否仍然主导训练目标”。

---

## 问题定义与边界

“合成数据的使用”要解决的是一个很具体的问题：真实数据不足，但任务又必须上线。这里的“不足”包括四类：

1. 数量不足：新业务刚启动，没有足够标注样本。
2. 事件稀缺：极端风险、异常故障、欺诈样本天然少。
3. 采集受限：隐私、监管、商业保密限制原始数据流出。
4. 标注昂贵：需要领域专家判断，单条样本成本高。

边界也必须先说清。合成数据不是替代现实世界的万能钥匙，它只能在“真实规律已知一部分、生成目标可验证”的场景里发挥价值。比如文本分类、信息抽取、客服问答回放、风控演练，这些任务有明确标签空间，适合先生成再筛选。反过来，如果任务依赖极细的现实因果结构，比如医疗诊断、司法裁判、交易执行，合成数据只能做辅助，不能直接代表真实世界。

下面这张表更适合工程判断：

| 场景 | 可用真实数据量 | 合成数据需求 | 隐私/监管考虑 |
|---|---|---|---|
| 金融舆情分类 | 少量历史样本，覆盖不足 | 补标签、补长尾表达 | 高，需防止泄露公司与客户信息 |
| 极端风险事件识别 | 极少，真实事件罕见 | 扩充罕见场景 | 高，需审查误导性模式 |
| 客服意图分类 | 中等，但新意图持续出现 | 快速补新类样本 | 中，需脱敏对话内容 |
| 自动驾驶仿真 | 真实事故数据极少 | 扩充危险边界场景 | 高，需保证仿真条件可追溯 |

玩具例子先看一个最简单的。

假设你要训练一个“句子情绪分类器”，标签只有“正面、负面、中性”三类，但手里真实数据只有 30 条。此时可以先写 10 条真实句子作为骨干样本，让教师模型按同一风格改写、扩写并补标签，例如：

- 真实句子：`公司本季度利润增长超预期`
- 教师生成：`管理层上调全年业绩指引，市场反应积极`
- 标签：正面

这个例子里，合成数据的价值不是创造新事实，而是扩充“同类表达方式”。

真实工程例子则更典型。某金融公司要做品牌舆情监控，但关于某个垂直品牌的历史舆情样本太少，人工标注又慢。常见做法是：先抽一批代表性文本，由 LLM 按统一提示词输出正面、负面、中性标签，再把结果映射到内部标准枚举，最后用这些“教师标签 + 少量人工校验样本”去微调一个内部可部署的小模型。这样做的重点不是追求教师模型永远在线，而是把外部大模型的能力转移成内部低成本推理能力。

---

## 核心机制与推导

核心机制可以概括为四步：

1. 教师生成  
教师模型是能力更强、知识更广的大模型。它负责根据 prompt 生成标签、解释、改写样本或新样本。白话讲，它像一个临时高级标注员。

2. 质量评价  
不是教师吐出的东西都能进训练集。必须检测合成样本是否贴近真实数据分布。常见指标之一是 PPL，即 Perplexity，中文常译“困惑度”，可理解为“一个语言模型读这段文本时觉得多不多见、顺不顺”。

$$
PPL = \exp\left(-\frac{1}{N}\sum_{t=1}^{N}\log p(w_t \mid w_{<t})\right)
$$

其中，$w_t$ 是第 $t$ 个 token，$p(w_t \mid w_{<t})$ 是在前文条件下该 token 的概率。若一批合成文本的 PPL 明显高于真实文本，通常说明它们偏离了真实语料风格，或者带有教师模型自己的模板化口吻。

3. 真实/合成混合  
把筛过的合成数据按比例加入真实数据。比例没有通用常数，但原则很明确：真实数据负责“锚定任务”，合成数据负责“填充覆盖”。

4. 学生学习  
学生模型一般选成本更低、部署更稳定的小模型，如 RoBERTa、DistilBERT、轻量分类器。它最终服务于线上系统。

可以把流程理解成：

`教师模型 -> 生成标签/样本 -> 质量评价 -> 与真实数据混合 -> 学生模型训练 -> 部署`

为什么不能只用合成数据？因为合成数据大多来自已有模型分布，而不是直接来自真实世界。如果下一轮训练继续拿这些数据再生成更多数据，就会形成“模型输出喂给模型”的闭环。闭环本身不是问题，问题在于闭环里没有足够的新信息输入。MAD 本质上是信息熵塌缩：表达越来越像模板，边界越来越窄，最后模型对现实输入的适应性下降。

从分布角度看，可以把目标写成：

$$
P_{train}(x, y) = \alpha P_{real}(x, y) + (1-\alpha) P_{synt}(x, y)
$$

其中 $\alpha$ 是真实数据占比。若 $P_{synt}$ 与 $P_{real}$ 足够接近，适当降低 $\alpha$ 可以节省成本；若两者偏差很大，则必须提高真实数据比重，并增加人工校验。

一个实用判断是：

- 若合成样本在验证集上带来召回率提升，说明它补到了覆盖面。
- 若它只提高训练集表现、却拉低验证集表现，通常说明它补的是噪声或模板偏差。

---

## 代码实现

下面给一个可运行的简化 Python 示例。它不直接调用真实大模型 API，而是用规则函数模拟“教师打标签”，重点展示流程：生成、筛选、混合、训练前检查。

```python
from math import exp, log
from collections import Counter

LABEL_MAP = {"positive": 2, "neutral": 1, "negative": 0}

real_samples = [
    {"text": "company profit increased strongly", "label": "positive"},
    {"text": "revenue stayed flat this quarter", "label": "neutral"},
    {"text": "guidance was cut after weak sales", "label": "negative"},
]

raw_unlabeled = [
    "management raised full year outlook",
    "demand remained unchanged in europe",
    "operating margin fell sharply",
]

def teacher_label(text: str) -> dict:
    t = text.lower()
    if any(k in t for k in ["raised", "increased", "strongly"]):
        return {"text": text, "label": "positive", "reason": "performance or outlook improved"}
    if any(k in t for k in ["fell", "cut", "weak"]):
        return {"text": text, "label": "negative", "reason": "performance deteriorated"}
    return {"text": text, "label": "neutral", "reason": "no clear positive or negative signal"}

def toy_ppl(text: str, unigram_probs: dict, floor: float = 1e-6) -> float:
    tokens = text.lower().split()
    n = len(tokens)
    avg_neg_log = -sum(log(unigram_probs.get(tok, floor)) for tok in tokens) / n
    return exp(avg_neg_log)

def fit_unigram(texts):
    counts = Counter()
    total = 0
    for text in texts:
        toks = text.lower().split()
        counts.update(toks)
        total += len(toks)
    return {tok: c / total for tok, c in counts.items()}

# 1. 用真实数据拟合一个极简语言分布，作为“贴近真实分布”的近似参考
real_texts = [x["text"] for x in real_samples]
unigram_probs = fit_unigram(real_texts)

# 2. 教师模型生成标签
synthetic_samples = [teacher_label(x) for x in raw_unlabeled]

# 3. 质量控制：按 PPL 过滤过于偏离真实分布的句子
filtered_synthetic = []
for item in synthetic_samples:
    ppl = toy_ppl(item["text"], unigram_probs)
    item["ppl"] = ppl
    if ppl < 1e6:  # 示例阈值，真实工程应由验证集调参
        filtered_synthetic.append(item)

# 4. 合并真实与合成数据
train_dataset = real_samples + filtered_synthetic

# 5. 枚举映射
encoded = [{"text": x["text"], "label_id": LABEL_MAP[x["label"]]} for x in train_dataset]

assert len(synthetic_samples) == 3
assert all(x["label"] in LABEL_MAP for x in synthetic_samples)
assert len(train_dataset) >= len(real_samples)
assert set(x["label_id"] for x in encoded).issubset({0, 1, 2})

print(encoded)
```

上面代码里的 `toy_ppl` 只是“玩具例子”，即最小可理解版本。它用一元语言模型近似真实语料分布，目的是让初学者看明白：合成样本不是直接进训练，而是要先看它和真实样本“像不像”。

真实工程会再多三层逻辑：

1. 提示词约束  
让教师模型只输出固定 JSON 结构，减少解析失败。

2. 质量过滤  
除了 PPL，还会加标签置信度、关键词一致性、去重、长度阈值、人工抽检。

3. 训练闭环  
小模型上线后，把线上低置信度样本再送回教师模型或人工标注，继续补数据。

一个接近真实工程的伪代码如下：

```python
# 真实工程例子：金融舆情分类
from datasets import Dataset, concatenate_datasets

def llm_generate_label(client, text: str) -> dict:
    prompt = f"""
    你是金融情绪标注器。
    输入文本：{text}
    只输出 JSON:
    {{"label":"positive|neutral|negative","reason":"<20 words>"}}
    """
    # resp = client.chat.completions.create(...)
    # return parse_json(resp)
    return {"label": "neutral", "reason": "stub"}

def map_enum(label: str) -> int:
    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    return mapping[label]

def build_synthetic_dataset(client, unlabeled_texts, ppl_scorer, ppl_threshold):
    rows = []
    for text in unlabeled_texts:
        result = llm_generate_label(client, text)
        ppl = ppl_scorer(text)
        if ppl <= ppl_threshold:
            rows.append({
                "text": text,
                "label": map_enum(result["label"]),
                "source": "synthetic",
                "ppl": ppl,
            })
    return Dataset.from_list(rows)

def mix_datasets(real_ds, synt_ds, synt_ratio=0.5):
    # 实际项目通常还会按类别重采样，而不只是直接拼接
    return concatenate_datasets([real_ds, synt_ds])

# train_dataset = mix_datasets(real_ds, synt_ds)
# trainer = Trainer(model=student_model, train_dataset=train_dataset, eval_dataset=eval_ds)
# trainer.train()
```

这里最关键的不是 API 细节，而是数据管线顺序不能反：

先生成，再过滤，再混合，再训练，再评估。

如果顺序错成“先全量生成并直接训练，最后看效果”，通常已经把噪声引进来了。

---

## 工程权衡与常见坑

合成数据项目失败，往往不是因为“生成不出来”，而是因为“生成得太容易，所以大家忘了边界”。

| 问题 | 现象 | 影响 | 规避措施 |
|---|---|---|---|
| MAD，自噬 | 训练集越来越像模型腔调 | 泛化下降，线上误判增多 | 保留真实数据锚点，周期性引入新真实样本 |
| 偏差放大 | 合成样本反复强化某类表达 | 某些类别过拟合，公平性变差 | 做类别平衡、人工抽检、分布对齐 |
| 低质量模板化 | 文本重复、句式单一 | 验证集收益小，训练损失虚假变好 | 去重、PPL 过滤、多提示词生成 |
| 标签漂移 | 教师标准变动 | 学生标签语义不稳定 | 固定 prompt、固定枚举、版本化管理 |
| 监管与隐私风险 | 合成文本仍可推断真实个体或机构 | 合规问题，甚至无法上线 | 按真实数据同等级审查，先脱敏后生成 |

最常见的误区是“合成数据越多越好”。这在工程上通常是错的。新增合成样本的边际收益会快速下降，因为教师模型生成的内容存在强模板性。当你发现下面几种现象时，通常说明该停了：

1. 验证集指标不再提升，甚至开始回落。
2. 合成样本的重复率上升。
3. PPL 分布与真实样本差距扩大。
4. 学生模型在线上低置信度样本上的表现没有改善。

另一个常见坑是忽略“标签语义一致性”。例如教师模型有时把“业绩不及预期”标成负面，有时因为“长期战略积极”又标成中性。如果 prompt 没写清“以短期市场反应为准”还是“以长期基本面为准”，学生模型学到的就是摇摆规则。

实际项目里，建议至少保留一个专家校验环节。它不需要审核全部样本，但要定期抽样审查以下内容：

- 标签定义是否稳定
- 合成文本是否偏离行业用语
- 是否出现敏感信息重构
- 是否对少数类产生系统性误标

工程上最稳妥的心法是：把合成数据当作“放大器”，不要把它当作“真相来源”。

---

## 替代方案与适用边界

如果真实数据能拿到，优先顺序通常不是“先合成”，而是：

1. 先采集真实数据
2. 再做真实数据增强
3. 最后才用合成数据补缺口

原因很简单。真实数据包含任务真正要解决的噪声、歧义和分布漂移，而这些恰恰是模型上线后必须面对的内容。合成数据更像一种工程杠杆，用来降低成本、缩短冷启动时间，而不是替代现实。

常见替代路线如下：

| 约束类型 | 首选方案 | 合成数据作用 | 边界条件 |
|---|---|---|---|
| 真实数据容易获取 | 真实采集 + 人工标注 | 少量补长尾 | 不应替代主数据源 |
| 真实数据少但可脱敏 | 脱敏后标注 + 数据增强 | 补不足类别 | 需验证脱敏不破坏语义 |
| 隐私限制强 | 联邦学习、脱敏、局部合成 | 降低明文暴露 | 合成内容仍需合规审查 |
| 极端事件稀缺 | 仿真器 + 少量真实样本 | 扩危险边界 | 仿真必须覆盖真实物理约束 |
| 教师模型成本高 | 开源模型教师 + 小模型学生 | 降成本 | 需接受教师上限较低 |

这里要区分两类“替代方案”：

第一类是数据增强。它是在真实样本上做保守变换，例如同义改写、回译、裁剪、噪声注入。优点是更贴近真实分布，缺点是扩展性有限。

第二类是仿真生成。它更适合多模态、高风险、稀有事件任务，比如自动驾驶、机器人、工业异常。此时比起纯 LLM 生成文本，基于规则或物理世界的仿真器往往更可靠，因为它能显式编码边界条件。

什么时候停止增加合成样本？一个实用标准是同时看三件事：

1. 验证集收益是否停止增长
2. 样本质量指标是否恶化
3. 业务覆盖是否已经足够

如果这三项里有两项已经不再改善，就不应继续堆合成数据，而应回到真实数据采集或专家修订。

所以适用边界可以压缩成一句话：

真实数据负责“校准世界”，合成数据负责“扩展覆盖”；当“扩展”开始伤害“校准”时，就到了它的边界。

---

## 参考资料

1. 2024-02-16，Hugging Face 博客，《合成数据：利用开源技术节约资金、时间和减少碳排放》  
重点：给出 `financial_phrasebank` 上“教师生成标签 + 学生微调”的低成本案例，提供成本、准确率与碳排放对比，是本文工程例子的主要依据。

2. 2024-10-30，53AI，《合成数据用于大模型训练的3点理解》  
重点：强调合成数据不能脱离真实数据质量控制，指出模型自噬、偏差放大与监管风险，是“常见坑”部分的重要来源。

3. 2024-11-12，经济管理文库，《合成数据在生成式人工智能时代的价值与风险》  
重点：从“保留真实分布统计特征”的定义出发，讨论合成数据的价值、适用场景与风险边界，为“问题定义与边界”部分提供理论框架。
