## 核心结论

预训练数据不是“把网页抓下来就能喂模型”。真正可用的流程，通常是从 Common Crawl 的 WARC 原始抓取开始，先做正文抽取，再依次做语言识别、规则过滤、风险内容拦截、多级去重和质量分类，最后才得到可训练文本。这里的“质量过滤”就是用规则或分类器判断文本是否像正常文章，而不是乱码、导航栏、广告页或批量拼接页。

对零基础读者，可以先记住一个新手版流程：把抓到的网页变成纯文本段落，先删掉太短、太长、特殊符号太多的内容；再筛掉 NSFW 或恶意站点；然后按 URL、段落、文档三层去重；最后用质量打分器剔除低质量样本。剩下的数据，才有资格进入 tokenizer 和训练集切片。

工程上最重要的不是某一个神奇模型，而是“逐层串联、每层可监控”。因为数据脏、重复、偏题，往往不是同一个原因造成的。拆成多层后，才能回答三个关键问题：坏数据从哪一步漏进来的，重复样本在哪一层最严重，质量损失是否来自阈值设得过严。

| 阶段 | 输入 | 处理 | 输出 | 典型监控指标 |
|---|---|---|---|---|
| 原始抓取 | WARC/HTML | 正文抽取 | 纯文本文档 | 抽取成功率、空文档率 |
| 语言识别 | 纯文本 | fastText 等语言分类 | 英文文档 | `en` 通过率、置信度分布 |
| 规则过滤 | 英文文档 | 长度、重复行、特殊字符比例 | 格式较干净文本 | 过滤率、长度分布 |
| 风险过滤 | 格式较干净文本 | NSFW/恶意域名/黑名单 | 安全文本 | NSFW 命中率、域名命中率 |
| 去重 | 安全文本 | URL/段落/文档去重 | 非重复文本 | 去重率、簇大小分布 |
| 质量过滤 | 非重复文本 | 分类器打分 | 高质量语料 | 分数分布、保留率 |

---

## 问题定义与边界

这篇文章讨论的是“大规模英文预训练语料”的生产流程，不讨论多语言对齐、图像文本配对，也不讨论监督微调数据。边界要先讲清楚，否则“过滤”两个字会变成空话。

这里的“英文”不是人工阅读确认，而是语言分类器给出足够高的英文概率。FineWeb 数据卡中给出的是 fastText 语言过滤，英文分数低于 0.65 的文档会被移除。这里的“阈值”就是一条数值边界，低于它直接丢弃，高于它进入下一步。

这里的“干净”也不是文学意义上的“写得好”，而是满足几个工程条件：

| 定义对象 | 一个可操作定义 | 为什么要这样定义 |
|---|---|---|
| 英文 | 语言分类器判断为 `en`，且分数超过阈值 | 防止多语言噪声混入英文训练 |
| 长度合适 | 太短或太长都剔除 | 太短信息量低，太长常含模板、日志或拼接页 |
| 特殊字符正常 | 非字母数字符号比例不过高 | 降低乱码、表格残片、网页碎片 |
| 非风险内容 | 域名和文本不过 NSFW/恶意规则 | 降低安全、合规和品牌风险 |
| 非重复 | URL、段落、文档层面都不过度重复 | 避免浪费算力和放大记忆风险 |
| 高质量 | 像自然文章而不是导航栏、SEO 垃圾页 | 提高单位 token 的训练价值 |

一个玩具例子最容易理解。假设抓到 5 个网页：

1. “How to boil eggs” 正常英文教程。
2. 同一网页带追踪参数的另一个 URL。
3. 一篇英文正文后面拼了 300 行“Related posts”。
4. 全是特殊字符和碎片 HTML。
5. 正文正常，但来自成人站点。

这 5 个网页里，真正适合训练的可能只有第 1 个。第 2 个会在 URL 级或文档级去重时移除，第 3 个可能在规则过滤或质量分类里被打低分，第 4 个会在特殊字符比例中过滤，第 5 个会在 NSFW 或域名黑名单阶段拦截。

一个最小伪代码可以写成：

```text
if is_english(text) and length_ok(text) and symbol_ratio_ok(text):
    if not is_nsfw(url, text):
        keep(text)
```

这不是最终系统，但它已经说明了边界：先定义什么不能进，再谈怎样选出更好的内容。

---

## 核心机制与推导

大规模语料里，去重是最值钱也最容易被误解的一步。因为“完全相同”很好查，但预训练里更常见的是“近似重复”：同一篇文章被转载、改标题、删几句、插广告、换段落顺序。MinHash-LSH 的作用，就是用较低成本找到这些“看起来不是一样、实质上很像”的文档。

先定义术语。`k-gram` 是把文本切成长度为 `k` 的连续片段；对白话读者，可以理解成“滑动窗口截出来的一串小碎片”。如果文档 $d$ 的 5-gram 集合记作 $S_d$，那么它们的 Jaccard 相似度是：

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

意思很直接：两个集合共有多少片段，占总片段的比例有多大。完全相同则 $J=1$，完全不重合则 $J=0$。

MinHash 的关键结论是：如果用一个随机哈希函数对集合里的元素取最小值，那么两个集合“最小哈希相同”的概率，等于它们的 Jaccard 相似度。于是文档签名可以写成：

$$
\text{sig}(d)=\left[\min_{s\in S_d} h_j(s)\right]_{j=1}^{m}
$$

这里的 $m$ 是哈希函数个数。白话解释：每个哈希函数都给文档做一次“极简指纹”，很多次组合起来，就能近似表示文本相似度。

为什么它能把复杂度从 $O(n^2)$ 降下来？朴素方法要比较所有文档对，$n$ 篇文档要做大约 $n(n-1)/2$ 次比较。LSH 的做法是把签名切成多个 band，只比较落入同一桶的候选。于是总成本近似变成：

$$
O(n^2)\rightarrow O(nk)
$$

这里的 $k$ 不是前面的 `k-gram`，而是每篇文档平均进入的候选数。只要桶设计得合理，$k \ll n$，成本就能大幅下降。

FineWeb 所依赖的 datatrove 默认参数是 5-gram、14 个桶、每桶 8 个哈希，也就是总签名长度 $m=14\times 8=112$。LSH 的候选概率近似为：

$$
P(\text{candidate}|s)=1-(1-s^r)^b
$$

其中 $s$ 是 Jaccard 相似度，$b=14$，$r=8$。对应的经验阈值大约是：

$$
t\approx \left(\frac{1}{b}\right)^{1/r}=\left(\frac{1}{14}\right)^{1/8}\approx 0.72
$$

如果两个文档的 Jaccard 相似度是 0.8，则：

$$
1-(1-0.8^8)^{14}\approx 0.924
$$

也就是说，它们有约 92.4% 的概率会被送入候选集合，再做精确比对。这不是“已经判定重复”，而是“高概率进入复审名单”。

| 参数 | 含义 | 作用 |
|---|---|---|
| 5-gram | 文本切片长度 | 太小会误报，太大会漏掉轻微改写 |
| 14 buckets | band 数量 | 增加候选召回，但也增加桶处理开销 |
| 8 hashes/bucket | 每个 band 的行数 | 增加选择性，减少误报 |
| 阈值约 0.72 | LSH 转折点 | 高于此值更容易被拉入候选 |
| $s=0.8$ 时召回约 92.4% | 近重复召回水平 | 适合作为生产基线 |

玩具例子可以这样看。文档 A 是“今天发布模型训练报告”，文档 B 是“今天发布了模型训练报告全文”，文档 C 是“晚饭吃面条”。A 和 B 的 5-gram 重合很多，A 和 C 几乎没有重合。MinHash-LSH 不会让 A 去和所有文档全量比较，而是优先把 A 和 B 放进同一候选桶，再忽略掉大部分明显不相关的文档。

真实工程例子是 Common Crawl 的多 dump 处理。FineWeb 的经验不是把所有 crawl 一次性做全局 dedup，而是每个 crawl 先单独去重，再做采样和混合。原因很实际：如果一次性全局去重，候选对数量、桶文件大小、调度开销都会明显膨胀，反而更难控制质量和成本。

质量过滤通常放在去重之后。原因是如果先给大量重复文档打分，分数分布会被污染，而且你会为重复样本重复付推理成本。质量分类器本质上是一个“文本像不像正常训练材料”的判别器，常结合 Gopher/C4 风格规则和自定义特征，去识别列表页、重复行、格式错乱、SEO 农场页等。

---

## 代码实现

下面给一个可运行的极简 Python 版本，演示“规则过滤 + 近似去重 + 质量打分”的核心思路。它不是 FineWeb 生产实现，但结构是对的。

```python
import re
import hashlib
from collections import defaultdict

def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())

def is_english_like(text: str) -> bool:
    # 玩具版：真实工程应使用 fastText/GlotLID
    tokens = tokenize(text)
    if not tokens:
        return False
    common_en = {"the", "and", "is", "to", "of", "in", "for", "that"}
    hit = sum(t in common_en for t in tokens)
    return hit / max(len(tokens), 1) >= 0.05

def special_char_ratio(text: str) -> float:
    if not text:
        return 1.0
    bad = sum(not (ch.isalnum() or ch.isspace() or ch in ".,;:!?'-") for ch in text)
    return bad / len(text)

def rule_filter(text: str, min_len=40, max_len=5000, max_special_ratio=0.15) -> bool:
    if len(text) < min_len or len(text) > max_len:
        return False
    if special_char_ratio(text) > max_special_ratio:
        return False
    if not is_english_like(text):
        return False
    return True

def shingles(text: str, k=5):
    toks = tokenize(text)
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i+k]) for i in range(len(toks) - k + 1)}

def stable_hash(s: str, seed: int) -> int:
    payload = f"{seed}:{s}".encode("utf-8")
    return int(hashlib.md5(payload).hexdigest(), 16)

def minhash_signature(text: str, num_hashes=32, k=5):
    grams = shingles(text, k=k)
    if not grams:
        return [2**128 - 1] * num_hashes
    sig = []
    for seed in range(num_hashes):
        sig.append(min(stable_hash(g, seed) for g in grams))
    return sig

def lsh_bands(signature, bands=8):
    rows = len(signature) // bands
    assert rows * bands == len(signature)
    out = []
    for b in range(bands):
        chunk = tuple(signature[b*rows:(b+1)*rows])
        out.append((b, chunk))
    return out

def jaccard(a: str, b: str, k=5) -> float:
    sa, sb = shingles(a, k=k), shingles(b, k=k)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

def quality_score(text: str) -> float:
    toks = tokenize(text)
    unique_ratio = len(set(toks)) / max(len(toks), 1)
    punctuation_ok = 1.0 if re.search(r"[.!?]$", text.strip()) else 0.6
    return 0.7 * unique_ratio + 0.3 * punctuation_ok

docs = [
    "The model training report explains how data filtering and deduplication work in practice.",
    "The model training report explains how data filtering and deduplication work in practice today.",
    "%%%% #### <<<< raw html fragment >>>> $$$$",
    "Buy now buy now buy now buy now buy now buy now"
]

filtered = [d for d in docs if rule_filter(d)]
assert len(filtered) == 2

sigs = [minhash_signature(d, num_hashes=32, k=3) for d in filtered]
buckets = defaultdict(list)
for idx, sig in enumerate(sigs):
    for key in lsh_bands(sig, bands=8):
        buckets[key].append(idx)

candidates = set()
for ids in buckets.values():
    if len(ids) >= 2:
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                candidates.add((ids[i], ids[j]))

assert candidates, "近似重复样本应该进入候选集合"

dup_pairs = []
for i, j in candidates:
    if jaccard(filtered[i], filtered[j], k=3) >= 0.5:
        dup_pairs.append((i, j))

assert dup_pairs == [(0, 1)]

scores = [quality_score(d) for d in filtered]
assert all(0 <= s <= 1 for s in scores)
```

这段代码对应的工程模块可以概括为：

| 模块 | 输入 | 输出 | 监控指标 |
|---|---|---|---|
| `rule_filter` | 原始文本 | 是否通过 | 规则过滤率、长度分布 |
| `minhash_signature` | 通过规则的文本 | 签名向量 | 签名生成成功率、耗时 |
| `lsh_bands` | 签名向量 | 桶键 | 平均桶大小、超大桶比例 |
| `jaccard` | 候选文档对 | 精确相似度 | 候选转重复率 |
| `quality_score` | 去重后文档 | 质量分数 | 分数直方图、保留率 |

真实工程流水线通常是：

```text
Common Crawl WARC
-> 正文抽取
-> fastText 语言识别
-> 规则过滤
-> NSFW / 恶意域名拦截
-> URL 去重
-> MinHash-LSH 段落/文档去重
-> 质量分类器
-> 输出到 jsonl/parquet
```

每一步都应该落监控。比如语言识别通过率突然下降，可能是正文抽取坏了；去重率突然上升，可能是某个 dump 出现大规模模板页；质量分数整体变低，可能是规则过滤放得太宽。

---

## 工程权衡与常见坑

最大的误区，是把去重理解成“最后清一下就行”。不对。重复样本一旦混入训练，你会为同一信息反复支付 token、带宽、训练步数和显存时间。重复还会放大记忆风险，因为模型更容易记住被多次重复的内容。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 只做最终文档级去重 | URL 变体、转载页、段落复用大量漏过 | URL、段落、文档多层去重 |
| 规则过滤过宽 | 乱码、导航栏、SEO 页进入训练 | 先做阈值基线，再看样本回查 |
| 规则过滤过严 | 好文章被误删，召回不足 | 做误杀抽样，按 dump 分桶检查 |
| 全局一次性去重 | 候选爆炸，作业成本高 | 先按 dump 局部 dedup，再混合 |
| MinHash 参数固定不调 | 漏掉轻改写，或误报太多 | 以 5-gram、14x8 为基线做白盒调优 |
| 去重前先跑重模型打分 | 成本高，分数分布失真 | 先粗过滤和去重，再做质量分类 |
| 只看总保留率 | 定位不到问题来源 | 每层独立监控、独立抽样 |

MinHash 参数的取舍本质上是召回和精度的平衡。`k-gram` 变小，文本稍微相似就可能撞上，召回升高但误报增加；`k-gram` 变大，只有非常接近的文档才会被命中，误报下降但漏报增加。14 个桶、每桶 8 个哈希的好处，是在 $s \approx 0.72$ 左右形成一条比较实用的 S 曲线：高于阈值的近重复有较高机会进候选，低于阈值的大多数文档不会被拉进来。

真实工程里还要考虑“配比”。数据不是过滤完就结束，还要决定不同来源各占多少。The Pile 的典型做法是多来源混合，并对较高质量子集给更高 epoch 或权重；SlimPajama 给出一套更固定的来源比例，例如 CommonCrawl 52.2%、C4 26.7%，其余分配给 GitHub、Books、ArXiv、Wikipedia、StackExchange 等。这个动作叫“数据混合”，白话解释就是训练时不同来源被抽到的频率怎么分配。

这里的坑也很多。如果你只保留最“像教科书”的内容，模型会更整洁，但网络语言覆盖会变差；如果你让 Common Crawl 占比过高，模型泛化广，但噪声和模板文本也会抬头。FineWeb、The Pile、SlimPajama 的不同配方，本质上是对“质量、覆盖、多样性、成本”四个目标的不同折中。

---

## 替代方案与适用边界

MinHash-LSH 不是唯一方案，只是 trillion-token 量级下很实用的方案。它适合做表层文本近似重复检测，尤其是转载、轻微改写、模板复用这类问题。

| 方案 | 核心思路 | 成本 | 精度特点 | 适用规模 |
|---|---|---|---|---|
| MinHash-LSH | 基于 k-gram 的集合相似 | 低到中 | 对表层近重复强 | 超大规模网页语料 |
| SimHash | 基于特征签名的近似哈明距离 | 低 | 实现简单，但边界更粗 | 大规模快速预筛 |
| Embedding 去重 | 用向量相似度找语义重复 | 中到高 | 能抓语义近似，但成本高 | 中小规模高价值语料 |
| LM 质量打分 | 用语言模型困惑度或分类头评分 | 高 | 质量判断更强 | 去重后的小一层数据 |
| 纯规则过滤 | 长度、字符、黑名单 | 很低 | 可解释性强，但上限低 | 冷启动或资源紧张场景 |

如果你只有几 GB 到几十 GB 数据，embedding 去重是可选项。因为数据量不大，向量计算和 ANN 检索的成本还能接受，而且它能抓到“措辞不同、语义相同”的重复。但一旦进入 Common Crawl 这种量级，向量化每个文档、建全局索引、做持续增量更新，成本会迅速上升。此时 MinHash-LSH 的优势是便宜、稳定、可并行、易解释。

质量过滤也可以不用独立分类器，而改用语言模型分数、困惑度或小模型判别器。但这只适合在规则过滤和粗去重之后使用。否则你会把大量算力浪费在明显垃圾文本上。

一个实用判断标准是：

1. 数据规模极大，先用规则过滤 + MinHash-LSH。
2. 数据规模中等，且文本价值高，可以补 embedding 去重。
3. 数据规模较小，但质量要求很高，可以上更强的质量模型。
4. 无论哪种方案，NSFW/恶意域名/PII 处理都应前置，不能省略。

---

## 参考资料

| 资料 | 关键贡献 | 使用场景 |
|---|---|---|
| [FineWeb 数据集卡](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | 给出从 Common Crawl 到正文抽取、fastText 语言过滤、质量过滤、5-gram 14x8 MinHash 去重、PII 处理的完整流程 | 作为流程主线和参数基线 |
| [datatrove MinHash 参数说明](https://leeroopedia.com/index.php/Heuristic%3AHuggingface_Datatrove_MinHash_Parameter_Tuning) | 解释 5-gram、14 buckets、8 hashes、阈值约 0.72 和 $s=0.8$ 时约 92.4% 候选召回 | 说明去重公式与参数推导 |
| [The Pile 代码仓库](https://github.com/EleutherAI/the-pile) | 展示 22 个子集、权重、epochs 和 effective size 的混合方式 | 说明多来源语料的采样权重设计 |
| [The Pile 论文摘要页](https://www.scixplorer.org/abs/2021arXiv210100027G/abstract) | 说明 The Pile 的目标是用 22 个高质量子集提高多样性与泛化 | 解释为什么不能只靠单一 Common Crawl |
| [SlimPajama 数据集卡](https://huggingface.co/datasets/cerebras/SlimPajama-627B) | 给出经清洗和去重后的多来源开放语料 | 作为混合比例的另一套工程基线 |
| [Oumi 对 SlimPajama 的整理](https://oumi.ai/docs/en/latest/_modules/oumi/datasets/pretraining/slim_pajama.html) | 明确列出 CommonCrawl 52.2%、C4 26.7% 等来源占比 | 用于配比表述 |
| [Scaling Data Deduplication 经验总结](https://dev.to/e_b680bbca20c348/lessons-from-scaling-data-deduplication-for-trillion-token-llms-4d63) | 强调重复样本的算力浪费、参数错配和分阶段 dedup 的必要性 | 作为工程坑点总结 |
| [Apertus 技术报告 PDF](https://mboether.com/assets/pdf/hernandez2025apertus.pdf) | 给出长上下文训练中 70% Stage 5、20% FineWeb-Long、10% Institutional Books 的混合例子 | 说明“过滤后还要继续做数据配比” |
