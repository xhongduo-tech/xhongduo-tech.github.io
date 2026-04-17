## 核心结论

多语言统一分词，指的是让多种语言共享同一个固定大小的词表。词表可以理解为“模型可直接识别的基本词块清单”。mBERT 使用约 119K 的 WordPiece 词表，XLM-R 使用 250K 的 SentencePiece Unigram 词表，目标都一样：在有限 embedding 容量内，同时覆盖 100 多种语言。

它的核心矛盾不是“能不能把多种语言放进一个词表”，而是“有限容量下如何分配覆盖权”。高资源语言，白话讲就是训练语料很多的语言，比如英语、西班牙语，会自然占据更多高频词位；低资源语言，白话讲就是训练数据少的语言，比如尼泊尔语、藏语、某些东南亚语言，如果直接按原始语料比例训练，往往只能得到很少的专属 token，结果就是一句同样长度的话会被切成更多 token。

这会直接影响三件事：

| 影响项 | 原因 | 后果 |
| --- | --- | --- |
| 压缩率 | 低资源语言缺少整词或高频子词 | 句子被切得更碎 |
| 计算成本 | token 数变多 | 训练、推理更慢 |
| 公平性 | 不同语言表达同样信息需要不同 token 预算 | 上下文窗口和费用不对等 |

所以，多语言统一分词的实务结论很明确。

第一，词表规模不能太小。经验上，每种语言往往需要大约 1K 到 2K 个“核心 token”，也就是最常见、最能压缩文本的词块。支持 100 种语言时，共享词表通常至少要到 100K 量级，真正稳定可用的区间往往是 128K 到 250K。

第二，不能直接按语料原始比例训练词表。否则高资源语言会“挤占”词表。XLM-R 的做法是对语言采样做幂次平滑，也就是用 $p_l \propto n_l^\alpha$ 调整不同语言进入训练的概率，其中 $n_l$ 是该语言语料规模，$\alpha$ 常取 0.3 左右。它的作用不是让所有语言完全一样多，而是把差距从“非常悬殊”拉回到“还能共享”。

第三，多语言分词不是纯词表问题，而是系统容量分配问题。词表越大，embedding 表越大；词表越小，序列越长。工程上必须在“参数开销”和“token 压缩率”之间取平衡。

| 词表大小 | 英文平均 token/word | embedding 表内存（768 维，32-bit） |
| --- | --- | --- |
| 32K | 1.3-1.5 | 约 96MB |
| 100K | 1.1-1.3 | 约 300MB |
| 250K | 对多语言压缩更友好 | 约 750MB |

玩具例子可以这样理解：如果共享词表只有 30K，而你要覆盖 100 种语言，那么平均每种语言只能分到 300 个“有辨识力的位置”，这显然不够。即使共享子词能复用，不同文字系统、不同词形变化、不同常用前后缀都会抢位置，因此词表会很快失衡。

---

## 问题定义与边界

“多语言统一分词”解决的不是翻译问题，也不是语言建模本身，而是一个更底层的问题：如何把不同语言的原始文本映射为统一的 token 序列，让同一个 Transformer 编码器处理。

这里有三个边界需要先定清楚。

第一，它讨论的是“一个共享词表服务多种语言”，不是“每种语言单独一个 tokenizer”。如果每种语言单独训练分词器，覆盖率通常更高，但跨语言共享 embedding 的程度会下降，部署复杂度也会上升。

第二，它讨论的是“词表覆盖与序列长度”，不是“模型一定能理解语言”。分词做得差，模型几乎一定吃亏；分词做得好，也不等于模型自动具备强语义能力。分词器是输入压缩层，不是语言能力本身。

第三，它讨论的是“跨语言公平性”。公平性在这里不是社会学概念，而是工程度量：同样语义量的文本，不同语言是否需要差不多的 token 数来表达。常见指标包括 fertility、parity、NSL。

其中 NSL，Normalized Sequence Length，可理解为“同一批数据在两个 tokenizer 下的总长度比值”：

$$\text{NSL}(T_\lambda, T_\beta) = \frac{\sum_i \mathrm{len}(T_\lambda(D_i))}{\sum_i \mathrm{len}(T_\beta(D_i))}$$

如果一个 tokenizer 让某种语言的 NSL 长期偏高，说明它把该语言切得更碎。

一个对新手最直观的例子是同一段新闻。假设英文版本被切成 20 个 token，而印地语版本被切成 70 个 token。两段文本传达的信息量相近，但后者在训练和推理中要占用更多显存、更多时间、更多上下文窗口。对于按 token 计费的服务，这还意味着更高成本。这个现象就叫 token 不公平。

因此，多语言统一分词的真实问题可以表述为：

> 在固定词表容量下，如何让高资源语言不要占光词位，同时让低资源语言保持可接受的压缩率，并尽量保留跨语言共享子词带来的迁移收益。

---

## 核心机制与推导

主流方法通常分两步。

第一步，把多语言语料混在一起训练统一词表。常见算法是 BPE、WordPiece、SentencePiece Unigram。它们共同的目标都是从原始字符片段中学出一组高收益的词块。所谓“高收益”，白话讲就是既要出现得够频繁，又要能显著减少序列长度。

第二步，不直接使用原始语料分布，而是对语言采样做平滑。XLM-R 常见的公式是：

$$p_l=\frac{n_l^\alpha}{\sum_k n_k^\alpha}$$

这里：

- $n_l$ 是语言 $l$ 的语料规模
- $\alpha$ 是平滑系数
- $p_l$ 是训练时抽到该语言样本的概率

$\alpha=1$ 时，完全按原始语料比例采样；$\alpha$ 越小，越接近“各语言更均衡”。

为什么这个公式有效？因为幂次函数会压缩大数和小数之间的差距。

看一个玩具例子。假设高资源语言语料量是 500M token，低资源语言是 5M token，原始规模差 100 倍。若直接按比例采样，低资源语言几乎没有发言权。

若取 $\alpha = 0.3$：

$$500^{0.3} \approx 6.45,\quad 5^{0.3} \approx 1.62$$

此时采样权重比不再是 100:1，而约为 6.45:1.62，也就是约 4:1。差距仍然存在，但已经从“完全碾压”变成“可共同训练”。

这一步的意义不仅在预训练样本分布上，也会反过来影响词表学习。因为词表训练时看到的低资源语言片段更多，它们更容易拿到稳定的高频子词位置，而不是被迫退化成大量字符级拆分。

接着看词表预算为什么会被推到 100K 以上。假设你要覆盖 100 种语言，并希望每种语言至少有 1.5K 个“核心 token”，这些 token 包括：

- 高频整词
- 常见词根、词缀
- 该文字系统里稳定出现的字符组合
- 常见数字、标点、URL 片段等共享模式

那么仅这部分预算就需要：

$$100 \times 1500 = 150000$$

如果按 2K 核心 token 估算，就是 200K。再考虑跨语言共享 token 和通用符号，128K 到 250K 成为比较合理的区间，这正是 mBERT 与 XLM-R 一类模型的经验规模。

真实工程例子是 XLM-R。它在大规模多语言语料 CC100 上做预训练，使用 250K 的 SentencePiece Unigram 词表，并通过幂次平滑提升低资源语言在训练中的出现频率。这个设计不是偶然，而是为了避免“英语压缩得很好，其他语言被切碎”的失衡。

一句话概括机制链路：

> 统一词表带来跨语言共享，温度采样缓解资源失衡，大词表保障压缩率，三者缺一不可。

---

## 代码实现

工程里最先要落地的通常不是“训练完整 tokenizer”，而是先算清楚不同语言该以什么比例进入训练流程。下面给一个最小可运行示例，用 Python 计算语言权重、验证归一化、并做一次批次采样。

```python
import random
from collections import Counter

def build_language_weights(lang_counts, alpha=0.3):
    assert alpha > 0
    assert len(lang_counts) > 0
    assert all(count > 0 for count in lang_counts.values())

    raw = {lang: count ** alpha for lang, count in lang_counts.items()}
    total = sum(raw.values())
    probs = {lang: value / total for lang, value in raw.items()}

    # 基本正确性检查
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    return probs

def sample_batch(language_probs, batch_size, seed=0):
    assert batch_size > 0
    rng = random.Random(seed)
    langs = list(language_probs.keys())
    weights = [language_probs[lang] for lang in langs]
    batch = rng.choices(langs, weights=weights, k=batch_size)
    assert len(batch) == batch_size
    return batch

if __name__ == "__main__":
    # 一个玩具例子：英语语料远大于尼泊尔语和斯瓦希里语
    lang_counts = {
        "en": 500_000_000,
        "ne": 5_000_000,
        "sw": 20_000_000,
    }

    probs = build_language_weights(lang_counts, alpha=0.3)

    # 英语仍然更常见，但不再是按原始 100:1 直接碾压
    assert probs["en"] > probs["sw"] > probs["ne"]
    assert probs["en"] < 0.7

    batch = sample_batch(probs, batch_size=10000, seed=42)
    freq = Counter(batch)

    observed = {lang: freq[lang] / 10000 for lang in probs}
    for lang in probs:
        # 采样频率应接近理论概率
        assert abs(observed[lang] - probs[lang]) < 0.03

    print("language_probs =", probs)
    print("observed_freq =", observed)
```

这段代码表达了一个核心事实：多语言公平不是“让所有语言一样多”，而是“让小语种不会低到无法学习”。

如果继续往前走，典型工程流程是：

1. 统计每种语言的语料量 $n_l$。
2. 用 $n_l^\alpha$ 计算采样权重。
3. 按该权重对原始语料做重采样，得到混合训练集。
4. 在混合训练集上训练 SentencePiece 或 WordPiece。
5. 评估各语言的平均序列长度、UNK 比例、词级任务保留率。
6. 必要时调整词表大小或 $\alpha$。

真实工程例子可以设成一个多语搜索系统。假设系统需要同时支持英语、印地语、阿拉伯语、泰语和印尼语。如果直接用英文主导语料训练 50K 词表，英语查询可能平均 6 个 token，泰语和印地语可能变成 15 到 30 个 token。结果是：

- 查询编码耗时增加
- 索引侧 token 对齐变差
- 同样长度的上下文能容纳的非英语信息更少
- 下游排序模型对低资源语言更不稳定

这时最直接的工程动作通常不是“换模型”，而是重新设计 tokenizer 训练配比与词表大小。

---

## 工程权衡与常见坑

第一个权衡是词表大小和参数成本。

词表越大，embedding 表越大。对于隐藏维度 768 的模型，参数量和内存几乎按词表大小线性增长。250K 词表显著改善多语言压缩率，但仅输入 embedding 就是很可观的内存开销。这对小模型、边缘部署和移动端尤其敏感。

第二个权衡是共享程度和语言专属性。

共享越强，跨语言词块复用越多，迁移学习通常更好；但共享过度时，不同文字系统和低资源语言会被拆得过细。语言专属 token 多一些，压缩率好，但跨语言对齐可能下降，词表也会膨胀。

第三个权衡是句级任务和词级任务。

句级任务，如分类、NLI，往往更能容忍分词粒度变化；词级任务，如 POS、NER、依存分析，对词边界更敏感。一个 tokenizer 对句级任务效果不错，不代表它对词级任务也稳。

常见坑主要有五类。

第一，按原始语料比例直接训练统一词表。结果通常是英语占据大量整词和高频子词，其他语言被迫更多依赖字符级切分。

第二，只看总 perplexity，不看跨语言长度分布。很多系统整体指标能过，但某些语言 token premium 极高。所谓 token premium，白话讲就是“同样内容比英语多花多少 token”。

| 语言 | 相对英语 token premium |
| --- | --- |
| Shan | 4.43 |
| Dzongkha | 7.36 |
| 某些低资源文字系统语言 | 可高达 4-15 倍区间 |

第三，只扩词表，不改采样。如果语料分布仍极端倾斜，大词表也可能继续被高资源语言吃掉。

第四，忽略脚本差异。拉丁字母、阿拉伯字母、天城文、汉字体系的分词需求并不一样。统一词表并不等于统一预处理，script-aware normalization，也就是“按文字系统做规范化”，通常仍然必要。

第五，没有 fallback 机制。byte fallback 可以理解为“当字符没进词表时，退回到字节级表示”。它不能解决压缩率问题，但能避免完全 OOV，也就是超出词表导致无法稳定编码。

工程上更稳的做法通常是：

- 控制最低语言覆盖预算
- 对低资源语言做上采样
- 监控平均 token 长度和 token premium
- 保留 byte fallback 兜底
- 用真实下游任务而不是单一压缩率指标做最终决策

---

## 替代方案与适用边界

统一词表不是唯一方案，它只是大规模多语言模型里最常见、最方便部署的方案。

第一类替代方案是 language-specific tokenizer，也就是每种语言单独分词。这种方法在单语任务或少量语言任务中往往效果更稳，尤其适合形态变化很强、文字系统特殊、且训练数据相对充足的语言。缺点是部署复杂，跨语言共享 embedding 变弱。

第二类是 cluster-based tokenizer。做法是先把语言按文字系统或统计相似性分组，再给每组训练一个词表。它介于“完全统一”和“完全分开”之间，适合 10 到 30 种语言的中规模系统。

第三类是 parity-aware BPE 或公平性约束词表。它在合并子词时不只看频率，还看不同语言之间的覆盖均衡，目标是让各语言的压缩率更接近。

第四类是 byte-level 或 character-level 方案。优点是几乎没有 OOV，覆盖最稳定；缺点是序列通常更长，训练和推理成本更高。它们更适合极端开放字符集场景，而不是追求高压缩率的主流多语预训练。

可用一个简单指标辅助判断是否该继续坚持统一词表。Parity Ratio 可写成：

$$\mathrm{Parity}_{X,Y}(T) = \frac{\mathrm{mean}_{s\in S_X}|T(s)|}{\mathrm{mean}_{t\in S_Y}|T(t)|}$$

如果这个值长期显著偏离 1，说明同等文本量在两种语言上的 token 长度不对齐，可能需要重新分配词表或切换方案。

一个新手容易理解的例子是新增亚美尼亚语支持。假设当前系统已经服务 80 种语言，但亚美尼亚语进入后平均序列长度明显偏高。这时有三种常见决策：

| 方案 | 适用情况 | 代价 |
| --- | --- | --- |
| 扩大全局词表 | 语言总数继续增长，部署仍要求单词表 | embedding 成本上升 |
| 为该语言补充核心 token | 只需修正少数覆盖缺口 | 训练和兼容逻辑更复杂 |
| 独立或分组 tokenizer | 该语言长期任务重要且分布特殊 | 系统复杂度明显上升 |

适用边界也要说清楚。如果系统只支持 3 到 5 种语言，而且这些语言都属于相近文字系统，没必要机械追求 250K 级别的超大共享词表。相反，如果系统要覆盖 100 多种语言，且包含大量低资源语言，统一词表就必须把“公平性”当成一等公民，而不是训练结束后再补救。

---

## 参考资料

- BERT-base Multilingual Cased 与 mBERT 词表规模介绍：<https://www.emergentmind.com/topics/bert-base-multilingual-cased>
- XLM-R 词表与多语言训练讨论：<https://discuss.huggingface.co/t/xlm-r-vs-llama-7b-tokenization/172649>
- XLM-R 与温度采样机制综述：<https://next.gr/ai/large-language-models/massively-multilingual-models-mbert-xlm-r>
- XLM-R 多语言训练背景与 250K SentencePiece Unigram：<https://www.emergentmind.com/topics/xlm-r>
- 多语言 tokenizer 公平性、核心 token、byte fallback 综述：<https://www.emergentmind.com/topics/multilingual-tokenizers>
- token 长度不公平与 token premium 讨论：<https://liner.com/review/language-model-tokenizers-introduce-unfairness-between-languages>
- 词表大小与 embedding 成本、序列长度权衡：<https://www.systemoverflow.com/learn/ml-nlp-systems/tokenization-preprocessing/vocabulary-size-trade-offs-and-sequence-length-impact>
- 多语言词表设计与替代策略综述：<https://www.emergentmind.com/topics/multilingual-sentencepiece-bpe-tokenizer>
- Parity 等公平性指标扩展讨论：<https://www.emergentmind.com/topics/multilingual-gpt-scale-tokenizers>
- Limisiewicz et al. 2023, 词汇覆盖与跨语言任务影响：<https://aclanthology.org/2023.findings-acl.350.pdf>
