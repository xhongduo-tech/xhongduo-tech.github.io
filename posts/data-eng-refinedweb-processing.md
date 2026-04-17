## 核心结论

RefinedWeb 的核心结论很直接：**网页数据不一定天然低质，关键在于处理管线是否足够严格**。这里的“管线”指按固定顺序执行的一串数据处理步骤。RefinedWeb 证明，只要对 Common Crawl 这类大规模原始网页做系统性的过滤、语言识别、质量筛选和多层去重，就可以得到一份足以支撑大模型预训练的高质量语料，而且训练效果能接近甚至局部超过 The Pile。

它的设计理念不是“先收集最贵的人工精选文本”，而是“先拿到足够大的网页集合，再把坏样本一层层剥掉”。这和很多新手的直觉不同。很多人以为网页文本太脏，必须依赖书籍、百科、论文库才行。RefinedWeb 的价值就在于反驳这个前提：**高质量并不只来自来源高贵，也可以来自处理严格**。

一个最小玩具例子可以说明这种“乘法式提升”。假设起点是 1000 万份原始网页文档，经过多级 MDR（MacroData Refinement，宏观数据精炼）处理后：

| 阶段 | 输入文档数 | 输出文档数 | 丢弃率 | 作用 |
|---|---:|---:|---:|---|
| 原始网页 | 10,000,000 | 10,000,000 | 0% | Common Crawl 原始输入 |
| URL/语言过滤后 | 10,000,000 | 5,000,000 | 50% | 去掉非英语、黑名单站点、异常 URL |
| 质量过滤后 | 5,000,000 | 3,800,000 | 24% | 去掉噪声文档、结构异常文本 |
| 文档/段落去重后 | 3,800,000 | 3,360,000 | 11.6% | 去掉近重复和模板内容 |

可以用一个简单公式表达这个过程：

$$
final = initial \times \prod_i (1 - discard\_rate_i)
$$

在上面的例子里：

$$
3.36\text{M} \approx 10\text{M} \times (1-0.50)\times(1-0.24)\times(1-0.116)
$$

这个例子重要，不是因为数字本身必须完全固定，而是因为它说明了一个工程事实：**每一级只要带来有限改进，串起来就会形成显著提升**。RefinedWeb 的贡献正是在这里。它不是发明了某一个神奇过滤器，而是把多个中等强度但可复现的筛选器串成了稳定的工业流程。

真实工程例子是 Falcon 系列模型。论文报告中，基于 RefinedWeb 训练的 1.3B 和 7.5B 模型，在多个基准上达到或接近用 The Pile 训练的同量级模型。这意味着：如果你的目标是构建一个大规模英文通用语料库，RefinedWeb 这条路线可以替代昂贵、许可复杂、人工维护重的“手工拼盘式”数据集。

---

## 问题定义与边界

RefinedWeb 解决的问题不是“怎样从互联网上下载更多文本”，而是：**怎样从 Common Crawl 这种原始网页快照中，自动构建一份可用于预训练的高质量英文语料库**。

这里有三个边界必须先说清楚。

第一，它主要面向**通用网页文本**，不是书籍、百科、学术论文的精选合集。也就是说，它默认输入是噪声很多、结构不稳定、重复严重的网页抓取结果。

第二，它追求的是**自动化可扩展处理**，而不是人工逐站审核。因为数据规模通常是数百亿到数千亿 token，人力筛选在这个量级上不可行。

第三，它关注的是**预训练可用性**，不是网页归档完整性。换句话说，很多网页内容虽然“真实存在”，但如果它对语言模型训练没有价值，或者会引入重复和污染，就应该被删掉。

对初学者来说，可以把它理解成这样一条流程：

`URL过滤 → 内容提取 → 语言识别 → 质量检测 → 重复剔除`

其中每一步都在回答一个不同的问题：

| 步骤 | 它在检查什么 | 为什么必须做 |
|---|---|---|
| URL 过滤 | 这个网页来源是否明显不适合进入语料 | 黑名单站点、恶意站点、重复 URL 会污染整个库 |
| 内容提取 | 网页里哪些是正文，哪些是导航栏、广告、脚注 | 原始 HTML 不是训练文本，必须先抽正文 |
| 语言识别 | 这段文本是不是目标语言 | 混入大量非英语会降低语料一致性 |
| 质量检测 | 文本是否像“人写的正常内容” | 噪声、乱码、列表垃圾、模板页会拖低质量 |
| 重复剔除 | 这段内容是否已经出现过 | 重复会让模型过拟合常见模板和热门站点 |

更具体地，边界条件通常会写成若干规则。例如：

- 域名命中 blocklist 的 URL 直接丢弃
- 提取正文后，英文比例需要足够高，例如可近似理解为“英语占主体”
- 文档中不能充满乱码、超长重复行、异常标点密度
- 同一 URL 或高度相似文档不能跨 dump 重复保留
- 长度过短、信息密度过低的页面不进入最终集合

这里的“dump”就是 Common Crawl 的一次抓取批次，可以理解为某个时间点的一整轮网页快照。**跨 dump 去重**非常关键，因为同一页面往往会在多轮抓取中重复出现。如果只在单轮内部去重，训练时仍然会多次看到同一内容。

一个新手友好的例子是：先用域名黑名单去掉色情、赌博、恶意下载站，再用 fastText 做语言识别，确认文本主体是英语，剩下的内容才进入后续质量和去重检查。这样做的目的不是“道德审查”，而是保证数据分布可控、文本结构稳定、任务目标明确。

---

## 核心机制与推导

RefinedWeb 的核心机制可以概括成一句话：**MDR 不是单点过滤器，而是串联式收缩系统**。每一级都让数据规模减少一点，但数据纯度提高一点。连续几级之后，整体质量会出现明显跃升。

数学上，它就是前面提到的乘法过程：

$$
N_{k+1} = N_k \cdot (1-r_k)
$$

其中 $N_k$ 是第 $k$ 级输入规模，$r_k$ 是这一层的丢弃率。最终结果是：

$$
N_{final} = N_0 \cdot \prod_{k=1}^{m}(1-r_k)
$$

为什么这种结构有效？因为网页噪声不是单一来源。URL 问题、语言问题、模板问题、近重复问题、低质量问题，它们彼此相关但不完全重合。单一规则抓不住全部噪声，只能用多级策略分而治之。

RefinedWeb 的去重尤其关键，因为它不是只做一种 dedup。这里的“dedup”就是去重，意思是尽量不让模型重复看到同一内容。

| 去重层级 | 主要方法 | 解决的问题 | 特点 |
|---|---|---|---|
| URL 级 | URL 规范化与跨 dump 去重 | 同一地址被重复抓取 | 最便宜，最先做 |
| 文档级近重复 | MinHash + LSH | 相似页面、转载页、镜像页 | 快，适合大规模粗筛 |
| 精确长片段重复 | 后缀数组或精确匹配 | 大段复制粘贴、站点模板 | 更准，但更重 |
| 段落级/行级 | 模板段落检测 | 页头页脚、版权声明、导航文案 | 对网页数据尤其重要 |

先说文档级 MinHash。MinHash 是一种近似集合相似度算法，白话讲，就是**不用逐字比较全文，也能快速估计两篇文档是否很像**。论文讨论的参数常写成 `b=20, r=450, n=9000`，可以理解为：先从文档中抽很多哈希特征，再分成多个 band 做局部敏感哈希（LSH），把可能相似的文档先召回出来。它不是最后的精确判决器，而是高效候选生成器。

MinHash 背后的目标通常是近似文档的 Jaccard 相似度。若两个文档的 shingle 集合分别为 $A$ 和 $B$，则：

$$
J(A,B)=\frac{|A \cap B|}{|A \cup B|}
$$

在 LSH 里，两篇文档被判为候选重复的概率近似为：

$$
P(\text{candidate}) = 1-(1-s^r)^b
$$

其中 $s$ 是相似度，$b$ 是 band 数，$r$ 是每个 band 的行数。这个函数的意义是：相似度越高，被召回为候选重复的概率越大。工程上不追求公式推导得多漂亮，而是关心它能否在可接受成本下把大量近重复文档捞出来。

但仅靠文档级 MinHash 不够。原因是很多网页并不是整篇相同，而是共享大段模板。例如站点统一的“关于我们”“隐私政策”“相关推荐”“版权声明”。如果只做整篇近重复，这些段落仍会在大量网页中反复出现，于是模型会学到很多无价值模板。因此还要做**精确长片段匹配**，例如删掉长度不小于 50 词的重复片段。

可以把这一层写成伪代码：

```text
for paragraph in document:
    if exact_match_length(paragraph, corpus_index) >= 50_words:
        discard(paragraph)
    elif template_similarity(paragraph, template_bank) > theta:
        discard(paragraph)
    else:
        keep(paragraph)
```

这里的 `template_bank` 可以理解为“常见模板段落库”，也就是那些在很多页面里重复出现、但信息价值很低的段落集合。`theta` 是阈值，意思是“相似到什么程度就认为它只是模板”。

这种多层 dedup 的价值，可以用一个玩具例子说明：

- 文档 A：教程正文 800 词 + 页脚模板 120 词
- 文档 B：教程正文完全不同 900 词 + 相同页脚模板 120 词

如果只做文档级相似度，A 和 B 可能不算重复，因为大部分正文不同。
但对语言模型来说，那 120 词模板出现了上百万次就会形成统计偏置。段落级去重正是为了解这个问题。

---

## 代码实现

工程实现上，可以把 RefinedWeb 式管线拆成六个模块：

1. URL 过滤
2. 正文抽取
3. 语言检测
4. 质量过滤
5. MinHash 文档去重
6. 长片段与段落模板去重

一个实用的数据结构设计如下：

| 模块 | 输入 | 输出 | 关键数据结构 |
|---|---|---|---|
| URL 过滤 | 原始 URL 列表 | 合法 URL | `set`、规范化 URL 字符串 |
| 正文抽取 | HTML | 纯文本正文 | 文本字符串、元信息 |
| 语言检测 | 文本 | 语言标签与置信度 | `lang`, `score` |
| 质量过滤 | 文本 | 保留/丢弃标记 | 统计特征字典 |
| MinHash 去重 | 文本 shingles | 重复簇 | 签名向量、LSH 桶 |
| 模板去重 | 段落列表 | 干净段落 | 倒排索引、后缀数组或 n-gram 索引 |

下面给一个可运行的 Python 玩具实现。它不等价于论文全流程，但能体现“过滤率可量化”和“段落级规则化筛选”的核心思想。

```python
import re
from collections import Counter

def quality_features(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    chars = len(text)
    if chars == 0:
        return {
            "chars": 0,
            "line_count": 0,
            "empty_ratio": 1.0,
            "alpha_ratio": 0.0,
            "digit_ratio": 0.0,
            "repeated_line_ratio": 0.0,
        }

    alpha = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    all_lines = [line.strip() for line in text.splitlines()]
    empty_ratio = 1 - (len(lines) / max(len(all_lines), 1))

    repeated_line_ratio = 0.0
    if lines:
        counts = Counter(lines)
        repeated = sum(v for v in counts.values() if v > 1)
        repeated_line_ratio = repeated / len(lines)

    return {
        "chars": chars,
        "line_count": len(lines),
        "empty_ratio": empty_ratio,
        "alpha_ratio": alpha / chars,
        "digit_ratio": digits / chars,
        "repeated_line_ratio": repeated_line_ratio,
    }

def keep_paragraph(text: str) -> bool:
    f = quality_features(text)
    if f["chars"] < 80:
        return False
    if f["alpha_ratio"] < 0.60:
        return False
    if f["digit_ratio"] > 0.20:
        return False
    if f["empty_ratio"] > 0.40:
        return False
    if f["repeated_line_ratio"] > 0.30:
        return False
    return True

good = """
RefinedWeb builds a large-scale corpus from web data.
It removes low-quality pages and repeated template paragraphs.
The goal is to keep useful natural language text for pretraining.
"""

bad = """
1234567890
BUY NOW
BUY NOW
BUY NOW

@@@@ #### !!!! 9999
"""

assert keep_paragraph(good) is True
assert keep_paragraph(bad) is False
```

如果把它扩展成实际管线，结构通常类似下面这样：

```text
for url in urls:
    if blocked(url) or seen_url(url):
        continue

    html = fetch_or_load(url)
    text = trafilatura_extract(html)
    if not text:
        continue

    lang, score = detect_language(text)
    if lang != "en" or score < min_lang_score:
        continue

    if not pass_quality_rules(text):
        continue

    sig = minhash_signature(text)
    if near_duplicate(sig, lsh_index):
        continue

    clean_paragraphs = []
    for para in split_paragraphs(text):
        if is_template_or_exact_duplicate(para, paragraph_index):
            continue
        clean_paragraphs.append(para)

    save(clean_paragraphs)
```

这里每一步都应该产出保留率指标，例如：

- `url_keep_rate`
- `lang_keep_rate`
- `quality_keep_rate`
- `doc_dedup_keep_rate`
- `paragraph_keep_rate`

因为没有这些指标，你就不知道是规则太松还是太狠。RefinedWeb 的工程价值不只在“过滤”，还在“可监控地过滤”。

一个真实工程例子是：你维护一个 50B token 级别的英文网页语料库。若质量过滤前后 token 数下降 5%，但下游模型 loss 几乎没变，说明这层规则可能太保守；如果只删 5%，但人工抽样仍看到大量导航栏和版权模板，说明规则太弱。**数据管线调参的依据不是感觉，而是保留率、抽样检查和训练指标的联合反馈。**

---

## 工程权衡与常见坑

RefinedWeb 这类系统不是“规则越多越好”，而是要在**规模、纯度、成本、可复现性**之间做平衡。

先看常见坑：

| 坑 | 影响 | 解决办法 |
|---|---|---|
| 只在单个 dump 内做 URL 去重 | 同一页面跨时间重复出现，训练被放大污染 | 做跨 dump 全局 URL 指纹 |
| 只做文档级去重 | 模板段落在不同页面反复保留 | 增加段落级和长片段去重 |
| 质量规则写死且不监控 | 很容易误删有效文本或放过噪声 | 记录每级丢弃率并做抽样复核 |
| 语言识别只看短文本 | 标题页、导航页误判多 | 先抽正文，再对足够长度文本检测 |
| 过度去重 | 规模急剧缩水，知识覆盖面下降 | 结合保留率曲线设阈值 |
| 不可复现的人工筛选 | 后续无法重跑、无法审计 | 优先用程序规则和固定版本工具 |

其中最容易被低估的是：**URL 去重必须跨 dump**。很多站点首页、文章页、分类页会在每月抓取中反复进入数据池。模型训练时如果多次见到相同热门网页，相当于人为提高这些内容的权重，最后学到的是抓取频率，不是语言世界的真实分布。

另一个关键权衡是：**只做文档 dedup 不够，但做到段落 dedup 会更贵**。原因很直接。文档级比较对象数量较少，而段落级会把每篇文档拆成很多片段，索引规模和计算量都会膨胀。但网页语料的模板污染恰恰集中在段落级，所以这部分成本通常值得付。

可以这样理解：

- 只做文档 dedup：适合先快速压掉大规模转载和镜像
- 加上段落 dedup：适合消灭页头页脚、版权声明、推荐语等模板噪声

实践中建议画两条曲线：

- 去重率 vs 最终保留 token 比例
- 去重率 vs 小模型验证集损失

前者看数据量损失，后者看训练收益。阈值不应该凭直觉设，而应看曲线拐点。比如段落相似度阈值再调严一点，保留率大幅下跌，但验证损失几乎不改善，这通常说明你开始误删正常内容了。

一个典型误区是看到“重复率高”就一路加码过滤。问题在于，某些高频短语是正常语言组成，不应该因为出现次数多就删掉。真正该删的是**结构性冗余**，比如大量网页共享的模板段、固定免责声明、批量生成的 SEO 页面，而不是自然语料中的高频表达。

---

## 替代方案与适用边界

RefinedWeb 不是唯一方案，它适合的是一种很具体的场景：**你需要海量、通用、可扩展、以网页为主的预训练语料，同时又不能接受原始 Common Crawl 的噪声和重复。**

和其他路线对比更容易看清它的边界：

| 方案 | 数据来源 | 质量控制方式 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|---|---|
| Raw Common Crawl | 原始网页 | 很弱或没有 | 量最大、最便宜 | 重复和噪声极高 | 只适合做原料，不适合直接训练 |
| The Pile | 多源人工拼接 | 来源级筛选为主 | 质量高、来源清晰 | 维护复杂、授权和拼接成本高 | 中高质量多源语料构建 |
| RefinedWeb | Common Crawl + 严格处理 | 管线式过滤与去重 | 可扩展、可复现、质量可控 | 工程实现复杂 | 大规模网页预训练语料 |
| Bergamot 类网页管线 | 网页为主 | 规则与模型结合 | 更灵活，可按任务定制 | 细节依赖实现 | 垂直领域或多语言网页清洗 |

对新手可以用一个白话类比，但不能把类比当定义。RefinedWeb 更像是：**先在入口处装网关，再用多级筛子把网页原料逐层净化，最后才送进训练系统**。它不是把整个网页仓库直接倒给模型，也不是只从少数“高贵文本源”里挑食材。

如果你已经有高质量手工语料，比如授权书籍、维基百科、技术文档、问答对，那直接使用这些数据通常更省事，且质量上限更高。但它们的问题是规模不够、覆盖不全、授权复杂，难以支撑超大规模通用模型。

如果你的目标是垂直领域，例如法律、医学、代码、金融，RefinedWeb 也未必是最优。因为它强调的是“广覆盖网页文本的系统净化”，不是“特定领域知识密度最大化”。这时更合理的做法往往是：以 RefinedWeb 类网页语料做通用底座，再混入高质量领域数据。

所以它的适用边界可以总结为：

- 需要百亿到千亿 token 级别的通用英文语料
- 主要原料来自 Common Crawl 或类似网页快照
- 不能依赖大量人工审核
- 希望流程能重复执行、可审计、可调参

如果这些条件不成立，RefinedWeb 不一定是最省成本的路线。

---

## 参考资料

1. Penedo, Guilherme 等. *The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only*. 论文，NeurIPS 2023，2023-09/12。链接：https://openreview.net/forum?id=kM5eGcdCzq  
   说明：核心来源，给出 RefinedWeb 的设计目标、MDR 管线、训练对比和主要实验结果。

2. NeurIPS Proceedings 页面：*The RefinedWeb Dataset for Falcon LLM*. 论文页面，2023-12。链接：https://proceedings.neurips.cc/paper_files/paper/2023/hash/fa3ed726cc5073b9c31e3e49a807789c-Abstract-Datasets_and_Benchmarks.html  
   说明：正式会议版本入口，可用于核对论文发表信息与摘要。

3. NeurIPS PDF：*The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only*. 论文 PDF，2023。链接：https://papers.nips.cc/paper/2023/file/fa3ed726cc5073b9c31e3e49a807789c-Paper-Datasets_and_Benchmarks.pdf  
   说明：包含更完整的实验表、处理流程和训练结果，是写工程细节时最可靠的文本来源。

4. Emergent Mind: *RefinedWeb Dataset*. 技术笔记/综述，发布时间以页面为准。链接：https://www.emergentmind.com/topics/refinedweb-dataset  
   说明：对 MDR 管线、MinHash 去重、后缀数组模板移除等工程点做了二次整理，适合作为阅读提要。

5. AI Wiki: *The Pile* 相关条目。百科/笔记，发布时间以页面为准。链接：https://aiwiki.ai/wiki/the_pile  
   说明：适合用来理解 The Pile 的多源拼接思路，以及为什么 RefinedWeb 要强调“网页数据也能做高质量预训练”。
