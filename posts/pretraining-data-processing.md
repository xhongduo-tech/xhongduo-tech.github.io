## 核心结论

预训练数据处理不是“清洗一下网页”这么简单，它本质上是在控制训练信号的来源。这里的“训练信号”指模型每一步参数更新所依赖的样本贡献。对大模型来说，最重要的闭环是：收集、语言识别、质量过滤、去重、格式标准化。其中真正决定效果上限的，通常是质量过滤和去重。

质量过滤解决的是“坏数据太多”的问题。坏数据包括乱码、模板页、广告页、极短碎片、机器拼接文本、异常符号堆积文本，也包括语法勉强通顺但信息密度极低的文本。去重解决的是“同一内容被重复学习太多次”的问题。重复样本会让梯度集中在少数模式上，削弱模型对长尾知识和表达方式的覆盖。

当前最成熟的组合是“规则过滤 + 分类器或语言模型评分 + 精确去重 + MinHash/LSH 近似去重”。规则过滤先快速去掉明显垃圾，分类器或困惑度评分再做细筛，最后用哈希和近似相似检索去掉重复与近似重复。这一顺序通常比“先去重再过滤”更稳，因为大量低质文本本身也会制造无意义的重复簇。

一个面向新手的最小流程可以概括为：先把抓来的网页按语言和编码分桶，再用“文本长度够不够、特殊字符多不多”筛掉明显垃圾；然后用一个小模型或参考语言模型判断文本是否“像百科、书籍、技术文档”；最后给每段文本算指纹，把完全一样或几乎一样的段落合并，只保留代表样本。

下表给出一个典型效果对比，数字是说明趋势的工程量级，不是固定常数：

| 处理方式 | 训练 token 量 | 信噪比 SNR | CORE 分数 |
|---|---:|---:|---:|
| 未经过滤、未去重 | 76B | 1.00 | 24.7 |
| 仅规则过滤 | 61B | 1.18 | 25.4 |
| 过滤 + 精确去重 | 52B | 1.29 | 26.1 |
| 过滤 + 精确去重 + MinHash/LSH | 45B | 1.41 | 26.8 |

困惑度常用来衡量文本是否接近高质量参考分布。它的定义是：

$$
\mathrm{Perplexity}(D)=\exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(w_i\mid w_{<i})\right)
$$

这里的“困惑度”可以直白理解为：一段文本让参考语言模型感到多“意外”。值越低，通常说明它越像参考高质量文本。

---

## 问题定义与边界

预训练数据处理要回答两个问题。

第一，哪些文本不值得训练。这里的“不值得”不等于“有错误字词”，而是指它们对模型泛化能力帮助很小，甚至会引入噪声。常见判据包括文本长度、可打印字符比例、重复行比例、语言一致性、URL/广告词密度、分类器质量分数等。

第二，哪些文本虽然表面不同，但其实表达的是同一内容。比如同一篇新闻被不同站点转载，代码仓库 README 只改了项目名，论坛回答重复粘贴。它们如果大量保留，会让模型过度学习同一模式。

这个问题的边界主要有三类。

| 维度 | 典型做法 | 主要边界 |
|---|---|---|
| 质量过滤 | 规则 + 分类器评分 | 规则简单但粗糙；分类器更准但会有领域偏差 |
| 精确去重 | 全文哈希、行哈希、段落哈希 | 只能抓完全一致或规范化后完全一致的重复 |
| 近似去重 | MinHash + LSH | 能抓改写和局部变体，但需要阈值与索引设计 |

如果对所有文本做两两相似度比较，复杂度接近 $O(N^2)$，数据规模到亿级后不可行。所以工程上不会直接做“所有文档两两比较”，而是先构造能快速召回候选重复对的索引，再做精筛。

另一个核心边界是“不要把难样本误删”。信号强噪比可以帮助理解这个问题：

$$
\mathrm{SNR}=\frac{\mathbb{E}[||\nabla_\theta \mathcal{L}_{\text{clean}}||^2]}{\mathbb{E}[||\nabla_\theta \mathcal{L}_{\text{noisy}}||^2]}
$$

这里的“信噪比”可以直白理解为：有用样本对参数更新的贡献，相对于噪声样本有多强。过滤的目标是提高 SNR，但如果把专业论文、法律条文、代码说明书这类“难但有价值”的文本也删掉，SNR 不一定真的提高，只是参考分布变窄了。

玩具例子：假设你抓了三类文本，共 1000 段。A 是百科类，B 是广告模板，C 是科研摘要。若只按“句子越像百科越好”打分，A 会大量保留，B 会被删掉，但 C 也可能被误删，因为科研文本术语密、句式长、分布和百科不同。这就是过滤边界。

真实工程例子：一个多语种网页语料池可能同时包含 UTF-8 正文、错误编码页面、导航菜单、转载新闻、站点镜像页。你不能只用“能不能解码”做质量判定，也不能只用英文参考集去评估中文或代码文本，否则会直接引入系统性偏差。

---

## 核心机制与推导

质量过滤通常分两层。

第一层是启发式规则。这里的“启发式”可以直白理解为：不靠复杂模型，只靠经验阈值先做快速裁剪。比如最小长度、平均词长、特殊字符比例、数字比例、重复 n-gram 比例、停用词覆盖、语言检测置信度。它们便宜，适合做第一道门。

第二层是统计评分或分类器评分。常见做法是先准备一个高质量参考集，例如百科、书籍、人工精选技术文档，再训练一个小分类器区分“高质量/低质量”，或者直接让参考语言模型给文本打困惑度分数。困惑度低，表示文本更接近参考分布；分类器高分，表示它更像高质量样本。

但不能把“像参考集”误当成“绝对高质量”。如果参考集全是百科，那么医学论文、编译器源码注释、数据库错误分析报告都可能因为风格不同而得低分。因此更稳的方案是用混合参考集：百科、书籍、问答、技术文档、代码文档分别建参考，再做加权评分。

去重的核心是相似度估计。最常见的是 Jaccard 相似度：

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

这里的 $A,B$ 通常不是原始字符串，而是文本切分后的 shingles，也就是固定长度 token 片段集合。白话讲，就是把一段话拆成很多连续短片段，再看两段文本共享了多少短片段。

直接算所有文档的 Jaccard 还是太贵，所以会引入 MinHash。MinHash 的直觉是：不用保存完整集合，只保存若干个“最小哈希签名”，这些签名在统计上可以近似反映 Jaccard 相似度。再进一步，把签名切成多个 band，放进 LSH 桶中。同一 band 命中的文档才成为候选对。这样就把“海量全比较”改成了“少量候选精查”。

如果签名被切成 $b$ 个 band、每个 band 有 $r$ 行，那么两个文档以概率被判为候选的大致趋势是：

$$
P(\text{candidate}) \approx 1-(1-s^r)^b
$$

其中 $s$ 是真实相似度。这个公式的直观含义是：相似度高的文档，更容易在某个 band 上完全碰撞，于是被召回；相似度低的文档，不容易进入候选集。于是我们用很低的成本，逼近“只比可能重复的那些对”。

下表总结两类去重机制的差异：

| 方法 | 能处理的重复类型 | 召回率 | 资源成本 | 可扩展性 |
|---|---|---|---|---|
| 精确哈希 | 完全相同、规范化后相同 | 低到中 | 低 | 很高 |
| MinHash/LSH | 近似改写、局部修改、转载 | 中到高 | 中 | 高 |

玩具例子：  
文本 A：“大模型训练需要高质量数据和去重流程。”  
文本 B：“训练大模型时，需要去重和高质量数据处理流程。”  
两句字面不同，但 3-gram 片段有明显重叠。全文哈希会认为它们不同，MinHash 则可能把它们判成近似重复。

真实工程例子：数十亿网页文档上，通常先做 URL 级、正文哈希级、段落哈希级精确去重，把完全复制的页面先拿掉；再在段落或文档粒度构建 MinHash LSH 索引，召回疑似近似重复的文档，再做更细的相似度确认和代表样本保留。

---

## 代码实现

下面给出一个可运行的最小实现。它不是生产级系统，但能把“规则过滤 + 简单评分 + 近似去重”的主干串起来。

```python
import math
import re
from collections import Counter

def rule_filter(text: str) -> bool:
    text = text.strip()
    if len(text) < 20:
        return False
    special_ratio = sum(1 for ch in text if not (ch.isalnum() or '\u4e00' <= ch <= '\u9fff' or ch.isspace())) / max(len(text), 1)
    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    if special_ratio > 0.25:
        return False
    if digit_ratio > 0.5:
        return False
    return True

def toy_perplexity_score(text: str, ref_words: set[str]) -> float:
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    if not tokens:
        return float("inf")
    # 用“参考词覆盖率”近似困惑度，值越小越好
    covered = sum(tok in ref_words for tok in tokens)
    p = max(covered / len(tokens), 1e-6)
    return math.exp(-math.log(p))

def shingles(text: str, k: int = 3) -> set[str]:
    text = re.sub(r"\s+", " ", text.strip().lower())
    if len(text) < k:
        return {text}
    return {text[i:i+k] for i in range(len(text) - k + 1)}

def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(len(a | b), 1)

def dedup_pipeline(texts, ref_words, ppl_threshold=2.5, tau=0.65):
    kept = []
    for text in texts:
        if not rule_filter(text):
            continue
        score = toy_perplexity_score(text, ref_words)
        if score > ppl_threshold:
            continue
        is_dup = False
        s1 = shingles(text)
        for old in kept:
            if jaccard(s1, shingles(old)) >= tau:
                is_dup = True
                break
        if not is_dup:
            kept.append(text)
    return kept

ref_words = {"模型", "训练", "数据", "质量", "去重", "文本", "文档", "预训练", "语言"}
docs = [
    "预训练模型依赖高质量数据与稳定的去重流程。",
    "训练预训练模型时，依赖高质量数据和稳定去重流程。",
    "!!!! 12345 #### 广告点击这里 ####",
    "数据质量过滤通常先做规则过滤，再做模型评分。"
]

result = dedup_pipeline(docs, ref_words)
assert len(result) == 2
assert any("高质量数据" in x or "高质量数据与稳定" in x for x in result)
assert all("广告" not in x for x in result)
print(result)
```

这段代码对应的模块关系如下：

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `rule_filter` | 原始文本 | `bool` | 过滤明显垃圾 |
| `toy_perplexity_score` | 文本、参考词表 | `float` | 近似质量分数，越低越好 |
| `shingles` | 文本 | 集合 | 生成局部片段指纹 |
| `jaccard` | 两个集合 | `float` | 衡量近似重复程度 |
| `dedup_pipeline` | 文本列表 | 过滤后文本列表 | 串联全流程 |

如果写成工程伪代码，主流程通常是这样：

```python
for text in raw_corpus:
    if not rule_filter(text):
        continue

    score = lm.perplexity(text)   # 或 classifier.predict_proba(text)
    if score > threshold:
        continue

    if exact_hash_index.contains(normalize(text)):
        continue

    candidates = minhash_index.query(text)
    if any(sim(text, c) > tau for c in candidates):
        continue

    output.append(standardize(text))
```

真实工程里还会增加三层能力：

1. 分片处理。把语料按语言、来源、日期、域名或 shard 切开，降低单机内存压力。
2. 代表样本选择。重复簇里不一定留第一条，而是保留最长、元信息最完整、质量分最高的一条。
3. 标准化输出。统一编码、换行、元数据字段、段落边界、文档 ID，为后续 tokenizer 和混料环节服务。

---

## 工程权衡与常见坑

最大的坑不是“规则太少”，而是“规则和评分都太像同一种文本”。如果你的参考集过于偏百科风格，那么模型会越来越擅长生成百科式答案，却不一定更懂真实工程语境。

下表展示过滤强度的典型权衡：

| 策略 | 阈值特征 | 高质样本保留 | 领域覆盖 | 风险 |
|---|---|---:|---:|---|
| 温和过滤 | 困惑度阈值宽、符号比例宽 | 高 | 高 | 噪声残留较多 |
| 中等过滤 | 困惑度和规则联合筛选 | 中高 | 中高 | 通常最稳 |
| 强过滤 | 只留最低分样本 | 中低 | 低 | 误删专业长尾内容 |

玩具例子：如果你把全部文本按困惑度排序，只保留最低 10%，留下的可能几乎全是百科、新闻和标准问答。数学证明、科研摘要、编程报错分析这类“形式复杂但信息密”的文本，会被系统性压缩。

真实工程例子：训练代码模型时，若直接使用通用自然语言质量分类器，很多高质量仓库文档会被错杀。原因不是它们差，而是它们包含路径、命令、栈追踪、配置片段，这些形式和普通文章差异很大。

去重也有相同问题。去重率越高，不代表越好。比如一个医学领域 shard 只有很少量样本，其中大量页面模板相似、术语重复度高。如果阈值 $\tau$ 设得太低，系统会把很多实际有价值的病例说明当成近似重复删掉。

因此通常需要 token budget。这里的“token budget”可以直白理解为：每个来源或领域最多保留多少训练 token。一个简单表达是：

$$
\sum_{d \in \text{source}_i} \mathrm{tokens}(d) \le B_i
$$

其中 $B_i$ 是来源 $i$ 的预算。它的作用不是只控总量，而是防止某一类来源因为规模大、重复多、分数高而挤占全部训练配额。

常见规避策略有三种：

| 问题 | 现象 | 规避方式 |
|---|---|---|
| 数据质量幻觉 | 评分高的不一定最有用 | 使用多参考集、多任务验证 |
| 去重过强 | 长尾领域被削薄 | 按 shard 保留代表样本 |
| 来源失衡 | Web 文本压过书籍/代码 | 为各来源设 token budget |

---

## 替代方案与适用边界

不是所有团队都需要全套“LM 打分 + MinHash/LSH + 分布式索引”。选择方案时，关键看语料规模、算力、延迟要求、目标模型类型。

第一类替代方案是“Bloom Filter + Suffix Array + 精确哈希”。Bloom Filter 可以直白理解为：一个很省内存的存在性筛子，用来快速判断“这个东西大概率见过没有”。它适合先挡住显式重复。Suffix Array 更适合做子串级重复检测，尤其对长段复制很有效。它们上线快，适合先把最明显的冗余降下去。

第二类替代方案是“先规则缩候选，再小模型评分”。对资源受限团队，这比全量计算困惑度更现实。你可以先把 100% 数据用规则筛到 30%，再让一个轻量分类器对候选集打分，而不是让大语言模型遍历全部数据。

第三类替代方案是“仅规则过滤”。它成本最低，但边界也最强，适合早期验证，不适合最终高质量语料建设。

| 方案 | 精度 | 覆盖范围 | 计算成本 | 适用场景 |
|---|---:|---:|---:|---|
| 纯规则过滤 | 低到中 | 只覆盖明显垃圾 | 很低 | 冷启动、快速试验 |
| Bloom + Suffix + 精确哈希 | 中 | 明显重复与局部复制 | 低到中 | 先做大规模降重 |
| MinHash + LSH | 中到高 | 近似重复、转载改写 | 中 | 大规模网页语料主流方案 |
| 规则 + 分类器 + MinHash/LSH | 高 | 质量与重复同时控制 | 中到高 | 正式预训练数据管线 |

一个实际可落地的折中方案是：先用 Bloom Filter 和正文哈希去掉完全复制，再用规则过滤砍掉明显垃圾，最后只对剩余候选做 MinHash/LSH 和质量评分。这样能把最贵的计算集中在真正值得判断的那一部分数据上。

---

## 参考资料

| 来源 | 贡献 | 建议阅读章节 |
|---|---|---|
| Data Filtering Stage for LLMs | 概括预训练数据过滤与去重主流程 | 问题定义与边界、核心机制与推导 |
| NVIDIA NeMo Curator 文档 | 给出启发式规则与工程参数思路 | 代码实现、工程权衡与常见坑 |
| DataComp-LM / NeurIPS 相关论文 | 展示过滤、去重、配比对 benchmark 的影响 | 核心结论、工程权衡与常见坑 |
| Milvus MinHash LSH 工程实践 | 说明大规模近似去重与索引加速方式 | 核心机制与推导、替代方案与适用边界 |
| 关于数据质量幻觉的研究 | 说明“高分不等于高价值”的风险 | 工程权衡与常见坑 |

参考链接：  
- https://rahatibnrafiq.github.io/llm_data_filtering/?utm_source=openai  
- https://docs.nvidia.com/nemo/curator/26.02/curate-text/process-data/quality-assessment/heuristic.html?utm_source=openai  
- https://papers.nips.cc/paper_files/paper/2024/file/19e4ea30dded58259665db375885e412-Paper-Datasets_and_Benchmarks_Track.pdf?utm_source=openai  
- https://blog.milvus.io/blog/minhash-lsh-in-milvus-the-secret-weapon-for-fighting-duplicates-in-llm-training-data.md?utm_source=openai  
- https://openreview.net/forum?id=vSBACt34gS&utm_source=openai
