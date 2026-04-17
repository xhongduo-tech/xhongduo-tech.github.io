## 核心结论

多语言 Tokenizer 的“词表分配公平性”问题，本质上是：同样表达一段意思，不同语言被切成的 token 数量差很多，而这个差距不是随机噪声，而是训练语料分布、脚本系统和分词算法共同造成的系统性偏差。

最常见的现象是英文被分得更“省”，中文、藏文、缅文、掸语、桑塔利语等语言被分得更“碎”。这里的“碎”指一句话会被拆成更多子词单元。对子词模型来说，更多 token 直接意味着更高训练成本、更高推理费用、更短可用上下文，以及更差的跨语言公平性。

一个常用度量是 token premium：

$$
p_L = \mathbb{E}_s\left[\frac{|t(S_L)|}{|t(S_{\text{en}})|}\right]
$$

其中 $S_L$ 是语言 $L$ 的句子，$t(\cdot)$ 是 tokenizer 输出的 token 序列，$|t(S)|$ 表示 token 数。若 $p_L=2$，意思是“同义句在该语言里平均需要英文两倍 token”。

对初级工程师来说，最重要的结论有三条：

| 结论 | 直接影响 | 工程含义 |
|---|---|---|
| 词表不是平均分给所有语言的 | 非英文常常更费 token | 成本估算不能只看英文样本 |
| token 多不只是计费问题 | 还会挤占 context window | 长上下文场景会先伤到高 premium 语言 |
| 更大的词表不一定自动公平 | 词表容量仍可能被高资源语言吃掉 | 需要显式评估而不是凭感觉上线 |

玩具例子很直观。假设英文句子 `The model needs more tokens for Chinese.` 被切成 8 个 token，而对应中文句子“这个模型处理中文需要更多 token。”被切成 16 个 token，那么中文 premium 就是：

$$
p_{zh} = 16 / 8 = 2.0
$$

这不是“中文更复杂”这么简单，而是“在当前词表设计下，中文被分配到的高效子词单位更少”。

---

## 问题定义与边界

Tokenizer 是“把文本切成模型可处理离散单元的组件”。在大模型里，这个单元通常不是词，而是子词，也就是词的一部分。BPE 和 SentencePiece 都属于这类方法。

讨论“公平性”时，边界要先讲清楚。这里不是讨论社会公平，也不是讨论翻译质量，而是讨论“相同语义内容进入模型时，不同语言是否需要大致相近的 token 预算”。

这个问题至少有三层：

| 层次 | 定义 | 关注点 |
|---|---|---|
| 计费层 | 每种语言平均消耗多少 token | API 成本、吞吐、上下文占用 |
| 建模层 | 模型是否能用相近粒度理解不同语言 | 表征质量、跨语言迁移 |
| 资源层 | 词表容量是否被少数语言长期占用 | 多语模型是否天然偏向高资源语言 |

除了 token premium，常见辅助指标还有 fertility。fertility 可以理解成“每个原始语言单位平均裂成多少 token”。若把空格分词后的词、字符或对齐片段记作基本单位 $A$，则：

$$
f = \frac{|T(A)|}{|A|}
$$

$f$ 越大，说明切分越碎。对无空格语言，basic unit 的定义要谨慎，否则会把脚本差异和 tokenizer 差异混在一起。

这里有一个容易混淆的边界：中文 token 多，不一定说明模型一定差；但如果中文 token 系统性地比英文多 1.5 到 3 倍，那么在固定上下文和固定预算下，中文用户确实先吃亏。这个结论是成本和容量层面的，不需要等到下游任务精度下降才成立。

再看一个玩具例子。假设上下文窗口固定 8K：

| 语言 | 平均每条客服消息 token 数 | 同一窗口可容纳轮数 |
|---|---:|---:|
| 英文 | 120 | 约 66 轮 |
| 中文 | 240 | 约 33 轮 |
| 掸语 | 480 | 约 16 轮 |

模型窗口没变，但高 premium 语言能放进去的有效对话历史明显更少。这就是“同样预算跑不同距离”。

---

## 核心机制与推导

为什么会不公平，核心在三个机制。

第一，训练语料分布不均。BPE 或 unigram tokenizer 会优先把高频片段收进词表。高资源语言，尤其英文，训练数据多、重复模式强，更容易获得完整词或高频词干。低资源语言因为数据少、频率低，常常只能保留更短、更碎的片段。

第二，脚本和预处理差异。中文、日文等语言没有天然空格边界；藏文、缅文、掸文等脚本的字符组合形式复杂；byte-level 方案又会把字符编码层面的细节暴露出来。结果是同样一个“词”，英文可能是一个合并后的稳定子词，别的语言却要拆成多个部分。

第三，词表容量是稀缺资源。无论 50K、100K 还是 250K，词表都不是无限的。某些语言多拿一个稳定 token，别的语言就可能少拿一个。这和缓存、带宽、参数预算一样，本质上是资源分配问题。

可以把 premium 推导成一个平均开销比。设平行语料中一对等义句为 $(S_L, S_{en})$，则：

$$
p_L = \frac{1}{N}\sum_{i=1}^{N}\frac{|t(S_L^{(i)})|}{|t(S_{en}^{(i)})|}
$$

如果再引入平均语义单位数 $u(S)$，则每语义单位 token 成本近似为：

$$
c_L \approx \frac{|t(S_L)|}{u(S_L)}
$$

当 $u(S_L)\approx u(S_{en})$ 时，$p_L$ 就近似描述了“表达同样信息要多花多少离散计算步”。

这会进一步影响注意力成本。自注意力计算量常写成 $O(n^2)$，这里 $n$ 是序列长度，也就是 token 数。若某语言 token 数从 $n$ 变成 $2n$，那么注意力部分的开销近似变成原来的 4 倍。虽然真实系统还有 KV cache、batching、prefill/decode 等因素，但方向不变：token 多会放大成本。

看三个常见多语模型的策略差异：

| 模型 | 典型分词思路 | 优点 | 问题 |
|---|---|---|---|
| XLM-R | 大词表 SentencePiece | 常见语种覆盖广 | 仍对低资源和非拉丁脚本不均衡 |
| mT5 | 大词表 unigram/SentencePiece | 泛化稳定 | 词表更大但不保证公平 |
| BLOOM | byte-level BPE | 几乎任何文本都能编码 | 覆盖不等于高效，某些语言仍极碎 |

真实工程例子是多语客服系统。若英文平均每轮 90 token，中文 170 token，掸语 420 token，那么在相同 GPU 上做批量推理时，会出现三个连锁反应：

| 指标 | 英文 | 中文 | 掸语 |
|---|---:|---:|---:|
| 单轮输入 token | 90 | 170 | 420 |
| 预填充延迟相对值 | 1.0 | 1.9 | 4.7 |
| 同 batch 可并发会话数相对值 | 1.0 | 0.53 | 0.21 |
| 上下文消耗速度相对值 | 1.0 | 1.9 | 4.7 |

这时即使模型本身“支持多语言”，系统层面也已经不公平了。

---

## 代码实现

下面给一个可运行的最小示例。它不依赖 Hugging Face，只模拟“同义句在不同语言下的 token 数”并计算 premium、fertility。这样先把指标算明白，再替换成真实 tokenizer。

```python
from statistics import mean

pairs = [
    {
        "en": "The model needs more tokens for Chinese.",
        "zh": "这个模型处理中文需要更多token。",
    },
    {
        "en": "We should measure tokenizer fairness before deployment.",
        "zh": "我们应该在部署前测量分词器公平性。",
    },
    {
        "en": "A larger vocabulary does not guarantee fairness.",
        "zh": "更大的词表并不保证公平。",
    },
]

# 玩具 tokenizer:
# 英文按空格切，中文按“每个汉字或连续英文串”切。
def toy_tokenize_en(text: str):
    return text.replace(".", "").split()

def toy_tokenize_zh(text: str):
    tokens = []
    buf = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)
        elif ch.isascii() and ch.isalnum():
            buf.append(ch)
        elif buf:
            tokens.append("".join(buf))
            buf = []
    if buf:
        tokens.append("".join(buf))
    return tokens

def count_premium(pairs):
    ratios = []
    zh_fertility = []
    en_fertility = []

    for item in pairs:
        en_tokens = toy_tokenize_en(item["en"])
        zh_tokens = toy_tokenize_zh(item["zh"])

        ratios.append(len(zh_tokens) / len(en_tokens))

        # 这里把“空格词数”或“汉字数”作为玩具版基本单位
        en_units = len(item["en"].replace(".", "").split())
        zh_units = sum(1 for ch in item["zh"] if "\u4e00" <= ch <= "\u9fff") + ("token" in item["zh"])

        en_fertility.append(len(en_tokens) / en_units)
        zh_fertility.append(len(zh_tokens) / zh_units)

    return {
        "premium_zh_vs_en": mean(ratios),
        "en_fertility": mean(en_fertility),
        "zh_fertility": mean(zh_fertility),
    }

result = count_premium(pairs)

assert result["premium_zh_vs_en"] > 1.0
assert result["zh_fertility"] >= result["en_fertility"]

print(result)
```

这段代码表达的是指标定义，不是生产级分词。换成真实工程实现时，一般流程如下：

1. 准备平行语料或高质量对齐语料。
2. 用真实 tokenizer 分别编码各语言句子。
3. 统计平均 token 数、premium、fertility。
4. 输出按语言排序的报表。
5. 将高 premium 语言纳入压测和成本模型。

如果要接真实 tokenizer，核心逻辑仍是这几行：

```python
def premium(counts_lang, counts_en):
    return sum(c_l / c_e for c_l, c_e in zip(counts_lang, counts_en)) / len(counts_en)
```

真正要注意的不是代码复杂度，而是数据对齐质量。若英文句子比中文句子信息更少，算出来的 premium 就会失真。

---

## 工程权衡与常见坑

第一类坑是只按英文估算成本。很多团队会拿英文样本压测吞吐，然后默认“多语差不多”。这是错的。若线上有大量高 premium 语言，实际 QPS、显存占用、平均延迟都会偏离预估。

第二类坑是把“大词表”当成答案。词表从 50K 扩到 250K，确实可能改善部分语言，但如果训练目标仍按全局频率最大化，新增容量仍可能优先奖励高频语言。容量增加不等于分配公平。

第三类坑是以为 byte-level 就天然公平。byte-level 的白话解释是“任何文本都先拆成字节再合并”。它解决的是“能不能编码”，不是“是否高效编码”。很多脚本在 byte-level 下仍然很吃亏，因为一个字符可能映射成多个字节，再加上 merge 不足，最终 token 依旧很多。

第四类坑是忽略上下文挤占。很多产品只看每轮计费，不看历史对话压缩速度。结果是英文客服能保留 30 轮上下文，中文只能保留 15 轮，某些低资源语言只剩几轮，随后检索召回、工具调用和多轮一致性全部下降。

可操作的排查清单如下：

| 常见坑 | 现象 | 缓解方式 |
|---|---|---|
| 只测英文 | 线上多语延迟异常 | 建语言分层压测集 |
| 只看平均值 | 尾部语言严重超标 | 统计 P95/P99 token 数 |
| 用非对齐语料比较 | premium 波动很大 | 用平行语料或严格控制主题 |
| 只算输入 token | 忽略上下文滚动成本 | 模拟完整多轮会话 |
| 误把覆盖率当效率 | “能编码”但仍很碎 | 同时看 premium 和 fertility |

真实工程里，建议把 tokenizer fairness 当成发布门禁项。比如规定：任何进入全球站点的模型，在目标语种集合上的 premium 报表必须齐全；若某语种超过阈值，例如 $p_L > 2.5$，则要单独评估上线风险，包括费用、上下文削减策略和降级方案。

---

## 替代方案与适用边界

替代思路主要有四类。

第一类是脚本感知的预切分。也就是在训练 tokenizer 前，先按语言或脚本特征做合理切分，让无空格语言、复杂脚本语言不要在起跑线上就吃亏。它的优点是实现成本相对可控，缺点是规则维护会增加。

第二类是 core vocabulary。做法是给跨语言高频、稳定、功能性强的片段预留核心词表，再把剩余容量用于统计学习。它能减少“全部容量被高资源语言卷走”的问题，但设计不好会牺牲整体压缩率。

第三类是 superword tokenizer。所谓 superword，可以理解成“尽量学到更长、更完整的高频片段”，而不是过度碎片化。对高 premium 语言，这往往比单纯扩大词表更有效，但训练和调参更复杂。

第四类是 byte 或 character 级方案。优点是覆盖最稳，不容易 OOV；缺点是序列通常更长。它适合极低资源、强鲁棒性优先、延迟不敏感的场景，不适合把上下文效率看得很重的在线系统。

对比可以压缩成一张表：

| 方案 | 优势 | 限制 | 适用边界 |
|---|---|---|---|
| 脚本感知预切分 | 能直接改善非拉丁脚本 | 需要语言规则 | 多语生产系统 |
| Core vocabulary | 控制资源分配偏置 | 设计不当会浪费容量 | 需要显式公平约束的模型 |
| Superword | 能明显降 premium | 训练实现更复杂 | 追求效率的多语模型 |
| Byte/char-level | 覆盖最稳 | token 长度常偏大 | 低资源、鲁棒性优先 |

初学者可以用一个简单直觉理解替代方案：如果某语言现在一句话要 4 倍 token，目标不是“让它和英文完全一样”，而是先把最浪费的拆分消掉，降到 1.5 到 2 倍，系统就会明显稳定得多。工程上，公平性通常不是追求绝对相等，而是把极端不均衡拉回可控范围。

---

## 参考资料

- Multilingual Tokenizers. EmergentMind, 2025. 讨论多语言 tokenizer 的公平性定义、token premium、fertility、NSL、STRR 等指标。
- Petrov, A. et al. Language Model Tokenizers Introduce Unfairness. NeurIPS 2023. 给出 XLM-R、mT5、BLOOM 在 FLORES-200 等数据上的跨语言 token 不均衡现象。
- Explaining and Mitigating Crosslingual Tokenizer Inequities, 2025. 讨论词表大小、预切分和 superword 等缓解方法。
- BigScience BLOOM 相关技术报告。可用于理解 byte-level BPE 的覆盖优势与效率局限。
