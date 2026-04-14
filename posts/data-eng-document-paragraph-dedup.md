## 核心结论

预训练数据去重不是“把一样的文本删掉”这么简单，而是要在不同粒度上分阶段处理，目标是让模型看到更多独特信号。这里的“独特信号”可以理解为真正带来新知识、新表达或新上下文的文本，而不是同一内容的重复拷贝。

最稳妥的工程做法不是只选一种去重方法，而是采用嵌套策略：先做 URL 级去重，再做文档级去重，最后做段落级或句级去重。原因很直接。URL 级去重成本最低，能先删掉同一个页面被重复抓取的情况；文档级去重擅长发现“几乎相同但不是逐字一致”的网页副本；段落级去重再负责清掉文档内部或跨文档重复出现的长模板、免责声明、广告段。

可以把它想象成三道筛子。先用 URL 去重把同源抓取合并，再在剩余网页里用 MinHash 找出模板副本，最后再删掉相同段落。只做前两步，会漏掉“文档不同但段落大量复用”的情况；只做最后一步，又容易把合法重复结构删过头。

| 粒度 | 典型方法 | 能抓到什么 | 主要风险 |
| --- | --- | --- | --- |
| URL 级 | URL 规范化、URL 哈希 | 同一地址重复抓取 | 抓不到改写副本 |
| 文档级 | 全文哈希、MinHash、LSH | 高相似网页、模板页、轻改副本 | 可能漏掉局部重复 |
| 段落级 | 滑动窗口、n-gram、句级 fuzzy match | 共享段落、页脚、广告块 | 阈值不当会过度删除 |

FineWeb 这类大规模预训练数据集的经验说明，文档级 MinHash/LSH 适合做主力近重复检测，但它不该单独承担全部去重任务。标准流程更接近“URL 预去重 + 文档级近重复聚类 + 段落级精筛”，而不是一次全局哈希完事。

---

## 问题定义与边界

去重的核心问题不是“两个文本一不一样”，而是“从训练角度看，它们是否提供了足够不同的新信息”。如果两个网页只有时间戳、分页号、广告位不同，那么对模型来说它们几乎在重复训练同一模式；如果两篇文章共享一段背景介绍，但后半部分给出不同结论，那么它们不该被整个删掉。

因此，必须先定义边界：

1. 什么算重复。
2. 重复判断在什么粒度上做。
3. 阈值设多高。
4. 在什么数据范围内比较。

文档级去重通常把整篇文档看作一个对象，判断“整体是否近似相同”。段落级去重则把文档拆成更小单元，判断“局部内容是否重复”。前者更保守，后者更激进。前者的问题是漏检，后者的问题是误删。

一个常见误区是跨快照做全局文档级去重。快照可以理解为某一时间点抓到的一整批网页副本。网页会被反复爬取，所以不同快照之间天然存在大量重复。如果把所有年份、所有快照放在一起全局去重，旧快照中的内容可能被大量踢掉。FineWeb 的实验里，最老快照在和后续大量快照做全局去重时，最多有 90% 的原始过滤后数据被移除。这并不一定代表这些内容“没价值”，而只是它们在后续快照里也出现过。

玩具例子很容易说明这个边界问题。假设 2023 年和 2024 年都抓到某技术文档站点：

- 2023 版：API v1 文档，包含旧参数说明。
- 2024 版：API v2 文档，前 85% 内容一样，只改了少量字段和示例。

如果跨快照全局去重，2023 版可能整体被判成重复而删除。但对模型训练来说，v1 和 v2 都有价值，因为它们携带版本演化信息。这里真正该做的是：每个 snapshot 内独立文档级去重，跨 snapshot 只做更轻的规则，或者干脆不做文档级全局合并。

文档级 MinHash 的阈值本质上也是边界定义。FineWeb 使用 112 个哈希、14 个 bucket、每个 bucket 8 个哈希，其匹配概率为：

$$
P_{match}=1-(1-s^8)^{14}
$$

其中 $s$ 是两个文档的 n-gram 相似度。这个公式的含义很直白：相似度越高，至少有一个 bucket 完全碰撞的概率越高，也就越可能被认定为近重复。它不是硬阈值，而是一条概率曲线。

---

## 核心机制与推导

文档级去重常用 MinHash。MinHash 可以理解为“用一小组摘要值近似表示一篇文档的集合内容”，这样就不用两两比较全文。LSH 是局部敏感哈希，白话说就是“让相似对象更容易落到同一个桶里”，从而把原本昂贵的全量比较变成只比较候选对。

FineWeb 的做法可以拆成四步：

1. 把文档切成 5-gram。
2. 用 112 个哈希函数计算 MinHash 签名。
3. 把 112 个值分成 14 个 bucket，每个 bucket 8 个值。
4. 只要两个文档在任意一个 bucket 上 8 个值完全相同，就认为它们是候选重复，再做聚类和过滤。

这里的 5-gram 指连续 5 个词组成的片段，白话说就是把文档滑动切成很多长度固定的小片段。这样做比按字符或整句更稳，因为它既能保留局部语义结构，又不会因为单个词变化就完全失配。

为什么是 14 个 bucket、每个 8 个？因为这组参数决定了“多相似才容易撞桶”。若两个文档的 n-gram 相似度为 $s$，则某一个 bucket 完全匹配的概率约为 $s^8$；14 个 bucket 中至少命中一个的概率就是：

$$
P_{match}=1-(1-s^8)^{14}
$$

这条公式解释了文档级 MinHash 的工程价值：它不是要求全文相同，而是对高相似文档给出快速、可调的高召回匹配。

| n-gram 相似度 $s$ | $P_{match}$ | 含义 |
| --- | --- | --- |
| 0.60 | $1-(1-0.6^8)^{14}\approx 21.4\%$ | 大多不会被判重复 |
| 0.75 | $1-(1-0.75^8)^{14}\approx 77.0\%$ | 高概率被捕捉 |
| 0.85 | $1-(1-0.85^8)^{14}\approx 98.8\%$ | 几乎必被识别 |
| 0.90 | $1-(1-0.9^8)^{14}\approx 100\%$ | 基本必命中 |

这组概率很关键。它说明文档级 MinHash 不是拿来抓“略微相似”的内容，而是抓“整体高度相似”的网页副本。比如两个文档的 $s=0.75$，重复确认概率接近 77%，这意味着系统对明显相似文档已经足够敏感，但仍会保留一部分边界样本，不至于把所有相似文本都砍掉。超过 0.85 时，识别概率接近 98.8%，几乎就是稳定命中。

玩具例子：

- 文档 A：教程正文 1000 词。
- 文档 B：其中 820 词一样，只改了标题、作者简介和一段广告。

如果用全文精确哈希，A 和 B 一定不同，完全抓不到。
如果用 5-gram + MinHash，A 和 B 的局部片段高度重合，$s$ 会很高，于是大概率被聚到同一重复簇。

真实工程例子是新闻站、文档站和电商页。它们常见的重复模式不是整页逐字一致，而是“模板相同 + 主体少量变动”。例如一个商品页的标题、价格、库存经常更新，但页头、参数表结构、评价模板和推荐区域高度重复。文档级 MinHash 会把这类页面拉成候选重复簇；接着你再决定是保留最新版本、保留信息最全版本，还是按 snapshot 分开保留。

段落级去重处理的是另一类问题：整篇文档不相同，但局部模板重复严重。例如 200 万篇博客都带同一段“转载声明”和同一段风险提示。这里如果只做文档级去重，重复模板会保留下来；如果做段落级去重，就可以通过滑动窗口和 n-gram 匹配删掉这些高频片段。

---

## 代码实现

工程上不建议直接从段落级去重开始。正确顺序通常是“便宜的规则先做，昂贵的 fuzzy match 后做”。一个简化流水线如下：

| Stage | 输入 | 处理 | 输出 |
| --- | --- | --- | --- |
| URL 预去重 | 原始抓取文档 | URL 规范化、URL 哈希 | 唯一 URL 文档 |
| 精确文档去重 | 唯一 URL 文档 | 全文哈希 | 完全相同文档被移除 |
| 文档级近重复 | 剩余文档 | 5-gram + MinHash + LSH + 聚类 | 保留每个重复簇代表文档 |
| 段落级精筛 | 保留文档 | 滑动窗口、n-gram/句级 fuzzy | 删除高频模板段 |
| 质检回扫 | 最终文档 | 长度阈值、删除比例控制 | 可训练语料 |

下面的 Python 代码不是 FineWeb 的完整实现，但足够展示核心思路：先做 URL 和全文哈希预去重，再做简化的段落窗口去重。代码可运行，并且包含 `assert`。

```python
from hashlib import md5
from collections import Counter
from urllib.parse import urlsplit, urlunsplit

def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower() or "https"
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, "", ""))

def exact_doc_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return md5(normalized.encode("utf-8")).hexdigest()

def ngrams(words, n=5):
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}

def jaccard_5gram(a: str, b: str) -> float:
    wa = a.split()
    wb = b.split()
    if len(wa) < 5 or len(wb) < 5:
        return 0.0
    sa = ngrams(wa, 5)
    sb = ngrams(wb, 5)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

def paragraph_windows(text: str, window_size=2):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    windows = []
    for i in range(len(paras) - window_size + 1):
        windows.append("\n\n".join(paras[i:i+window_size]))
    return windows

def dedup_pipeline(docs, doc_threshold=0.75, para_threshold=0.9, min_para_chars=80):
    seen_urls = set()
    seen_hashes = set()
    kept_docs = []
    removed_docs = []

    # Stage 1: URL + exact hash
    for doc in docs:
        url = normalize_url(doc["url"])
        text = doc["text"]

        if url in seen_urls:
            removed_docs.append((doc["id"], "duplicate_url"))
            continue
        seen_urls.add(url)

        h = exact_doc_hash(text)
        if h in seen_hashes:
            removed_docs.append((doc["id"], "duplicate_exact_doc"))
            continue
        seen_hashes.add(h)
        kept_docs.append({"id": doc["id"], "url": url, "text": text})

    # Stage 2: simplified doc-level near dedup
    final_docs = []
    for doc in kept_docs:
        is_dup = False
        for kept in final_docs:
            if jaccard_5gram(doc["text"], kept["text"]) >= doc_threshold:
                removed_docs.append((doc["id"], f"near_duplicate_of:{kept['id']}"))
                is_dup = True
                break
        if not is_dup:
            final_docs.append(doc)

    # Stage 3: paragraph/window cleanup inside kept docs
    global_windows = Counter()
    for doc in final_docs:
        for w in paragraph_windows(doc["text"], 2):
            if len(w) >= min_para_chars:
                global_windows[w] += 1

    cleaned_docs = []
    for doc in final_docs:
        paras = [p.strip() for p in doc["text"].split("\n\n") if p.strip()]
        cleaned_paras = []
        for p in paras:
            if len(p) < min_para_chars:
                cleaned_paras.append(p)
                continue
            # 简化版 fuzzy：只有在高频且长段落时才删除
            if global_windows[p] >= 2 and jaccard_5gram(p, p) >= para_threshold:
                continue
            cleaned_paras.append(p)
        cleaned_docs.append({**doc, "text": "\n\n".join(cleaned_paras)})

    return cleaned_docs, removed_docs

docs = [
    {
        "id": "a",
        "url": "HTTPS://example.com/post/1/",
        "text": "alpha beta gamma delta epsilon zeta eta theta iota kappa\n\n免责声明：仅供学习使用。"
    },
    {
        "id": "b",
        "url": "https://example.com/post/1",
        "text": "alpha beta gamma delta epsilon zeta eta theta iota kappa\n\n免责声明：仅供学习使用。"
    },
    {
        "id": "c",
        "url": "https://mirror.example.org/post/1",
        "text": "alpha beta gamma delta epsilon zeta eta theta iota lambda\n\n免责声明：仅供学习使用。"
    },
]

cleaned, removed = dedup_pipeline(docs, doc_threshold=0.75)

assert len(cleaned) == 1
assert removed[0][1] == "duplicate_url"
assert removed[1][1].startswith("near_duplicate_of:")
assert cleaned[0]["id"] == "a"
```

如果把这段简化流程换成真正的大规模实现，文档级部分通常会演化成下面的伪代码：

```python
for doc in snapshot:
    doc = normalize_url_and_text(doc)
    if seen_url(doc.url):
        continue
    if exact_hash_hit(doc.text):
        continue

    sig = compute_minhash(doc.text, num_hashes=112, ngram=5)
    buckets = split_into_buckets(sig, num_buckets=14, bucket_size=8)

    if bucket_match(buckets):
        mark_candidate_duplicate(doc)
    else:
        keep(doc)

clusters = transitive_cluster(candidate_pairs)
kept_docs = keep_one_doc_per_cluster(clusters)

for doc in kept_docs:
    doc = paragraph_fuzzy_filter(
        doc,
        min_chars=120,
        repeated_ratio_threshold=0.3,
        window_size=3,
    )
    write(doc)
```

真实工程里，`bucket_match` 之后不能立即删除，因为 A 和 B 可能命中，B 和 C 也命中，但 A 和 C 没直接命中，这时要做传递聚类，把它们并成一个重复簇，再从簇里保留一个代表文档。

---

## 工程权衡与常见坑

文档级与段落级去重最难的地方不在算法，而在阈值和作用范围。

第一类坑是跨快照全局文档级去重。理论上它能删掉更多重复，实际上很可能把旧网页、旧版本文档、历史新闻稿大面积删掉。FineWeb 的结果说明，最老 snapshot 在这种策略下最多会损失 90% 数据，而最终训练效果并没有因此更好。更稳妥的做法是：文档级 MinHash 仅在单个 snapshot 内执行，跨 snapshot 只保留轻量规则，比如 URL 级或更温和的 exact match。

第二类坑是段落级阈值过低。很多网页共享页脚、版权说明、风险提示、站点导航。如果你把任何重复段都删掉，最后可能把每篇文末的免责声明都删光，甚至破坏正文结构。段落级去重必须加长度约束、重复率约束、位置约束。短段、导航段、页脚段通常单独处理，不能混在同一阈值里一刀切。

第三类坑是保留策略过于简单。重复簇里“随机保留一个”在研究实验里很常见，但工程上未必合理。你常常更想保留 token 更多、正文更长、噪声更少、时间更新或来源更可信的版本。否则去重虽然完成了，留下来的却不是最佳样本。

真实工程例子：一个技术文档站每天重新生成静态页面，URL 带版本号，正文 95% 相同，差异集中在一个参数表和发布日期。若只做 URL 去重，几乎没效果；若跨版本文档级全局去重，又会把历史版本删除；更合理的是在“同一版本集”里做文档级近重复，再在段落级删掉统一页脚和导航模板。

| 坑/问题 | 后果 | 规避策略 |
| --- | --- | --- |
| 跨快照全局文档级去重 | 旧快照被大量清空，多样性下降 | 文档级去重按独立 snapshot 执行 |
| 段落阈值太低 | 合法模板或固定说明被误删 | 设最小长度、最小重复率、位置限制 |
| 只做全文精确哈希 | 轻微改写副本漏检 | 在精确哈希后加 MinHash |
| 只做段落级去重 | 计算昂贵，且容易过删 | 先做 URL/文档级粗筛 |
| 重复簇随机保留 | 留下低质量样本 | 按长度、质量分、时间戳选择代表文档 |

---

## 替代方案与适用边界

不是所有场景都需要 MinHash。去重方法的选择取决于数据源、变化模式和算力预算。

如果你的抓取源本来就很干净，比如固定 API 导出的知识库页面，或者每个内容对象天然有唯一 ID，那么 URL 哈希或全文精确哈希通常已经够用。它们实现简单、速度快、误删风险低。但只要页面模板会轻微变动、广告和时间戳会变化、同一内容会被镜像转载，精确哈希就会迅速失效。

MinHash 适合中间地带：文本主体高度相似，但不要求逐字相同。它的优势是召回近重复文档，劣势是要维护签名、bucket、候选对和聚类流程，工程复杂度高于精确哈希。

段落级或句级 fuzzy matching 更适合做补充，而不是主去重器。因为它成本更高，也更容易“误伤”共享结构。但对于长文档、论坛帖子、产品说明书、法律文本，这一步很有价值。尤其是当整篇文档保留下来是合理的，但其中某些大段模板会在全库重复上万次时，段落级去重能有效减少训练中的无意义重复。

| 方法 | 适用边界 | 成本 | 主要不足 |
| --- | --- | --- | --- |
| URL hash | 同源抓取、地址稳定 | 低 | 抓不到镜像和改写 |
| 全文精确哈希 | 完全重复文本 | 低 | 对轻微改动极敏感 |
| MinHash + LSH | 高相似文档、模板副本 | 中 | 只能近似，需聚类 |
| 句级/段落级 fuzzy | 局部模板重复、共享段 | 中高 | 容易过删，需要额外约束 |

因此，一个实用决策是：

- 内容源唯一、变化小：先用 URL/哈希。
- 页面模板多、轻改副本多：加文档级 MinHash。
- 长模板段反复出现：再补句级或段落级 fuzzy match。

这也是“URL/哈希消掉完全重复，再在剩余文档上用 MinHash；如果还有重复段落，再引入句级模糊匹配”的真正含义。它不是算法堆叠，而是按成本和风险逐层收缩搜索空间。

---

## 参考资料

| 资料 | 贡献/用途 |
| --- | --- |
| FineWeb NeurIPS 2024 论文 | 给出 FineWeb 的完整数据构建流程、独立 snapshot 去重实验结论、文档级 MinHash 设计与效果 |
| FineWeb Supplemental / Appendix E | 给出 5-gram、112 个哈希、14 个 bucket、匹配概率数值等具体参数 |
| Hugging Face `datatrove` 仓库 | 展示 FineWeb 去重流水线的工程实现思路，包括 signature、bucket、cluster、filter 等阶段 |
| 工程实践类去重文章与说明 | 用于补充 URL 级、段落级和分层去重的工程直觉 |

参考链接：
- FineWeb 论文：https://proceedings.neurips.cc/paper_files/paper/2024/file/370df50ccfdf8bde18f8f9c2d9151bda-Paper-Datasets_and_Benchmarks_Track.pdf
- FineWeb Supplemental：https://papers.nips.cc/paper_files/paper/2024/file/370df50ccfdf8bde18f8f9c2d9151bda-Supplemental-Datasets_and_Benchmarks_Track.pdf
- datatrove 仓库：https://github.com/huggingface/datatrove
- 一篇分层去重实践说明：https://callsphere.tech/blog/document-level-deduplication-llm-training?utm_source=openai
