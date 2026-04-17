## 核心结论

文本数据的预处理流水线，本质上是把原始语料按固定顺序送过一组清洗模块，输出模型能稳定消费的文本集合。对零基础读者，可以把“流水线”理解成一条传送带：每个工位只负责一种问题，比如去掉 HTML、识别语言、删掉重复、过滤噪声；但对工程实现来说，它不是简单相加，因为前一步的输出质量会直接决定后一步是否还能判断正确。

一个常见误区是把预处理理解成“越干净越好”。这不准确。真实目标不是把文本洗到最少，而是在噪声、重复、结构保留和领域信号之间找到平衡。比如法律条文、配置模板、接口文档常常长得很像，如果去重过猛，会把本来应该保留的版本差异一起删除；反过来，如果清洗太轻，模型会大量学习到导航栏、广告语、乱码和模板页。

工程上通常用三组指标来约束这个平衡：

$$
Retention = \frac{\text{清洗后有效 token}}{\text{原始 token}}
$$

$$
Duplicate\ Rate = \frac{\text{确认重复样本数}}{\text{总样本数}}
$$

$$
Effective\ Token\ Ratio = \frac{\text{清洗去重后 token}}{\text{清洗前 token}}
$$

其中 token 可以先简单理解为“模型处理文本时使用的最小片段”，不必先纠结具体分词器。Retention 太低，通常说明删过头了；Duplicate Rate 太低不一定是好事，可能是去重器太弱；Effective Token Ratio 太低，则说明大量 token 被浪费在无效内容上。

玩具例子可以直接看 1000 篇网页文章：先做 HTML 清理和编码统一，剩下 950 篇；再去重，删掉 250 篇近似重复；再做语言检测，剔除 100 篇非目标语言；最后只剩约 600 篇可信内容。这个例子说明，流水线并不是“最后一步补救”，而是决定训练集形状的主流程。

---

## 问题定义与边界

文本预处理流水线要解决的问题，可以定义为：对来源混杂、格式不一致、质量不稳定的文本集合做统一处理，输出适合训练、检索或推理使用的标准化文本。这里的“标准化”不是只改编码或空格，而是同时处理格式、语言、重复、句边界和噪声。

边界也必须说清楚。预处理不是信息抽取，也不是语义理解系统。它的职责通常停留在“让文本更可用”，而不是“理解文本含义并重写内容”。例如它可以删 HTML 标签、统一全角半角、识别是否为中文、判断是否疑似模板页；但它不应该擅自改写句子意思，也不应该用激进摘要替代原文。

不同语料的边界不同：

| 阶段 | 主要作用 | 典型输入 | 过度处理的风险 |
|---|---|---|---|
| 编码识别 | 统一成 UTF-8，消除乱码 | 爬虫文本、OCR 导出 | 错误转码导致字符丢失 |
| 格式统一 | 去 HTML、脚本、样式、控制符 | 网页、论坛、文档 | 把正文中的合法标记也删掉 |
| 语言检测 | 只保留目标语言 | 多语言网页、评论区 | 中英混写被误判 |
| 分句 | 切成句子或段落 | 连续正文 | 把缩写、小数点误切开 |
| 标准化 | 统一空白、标点、大小写 | OCR、论坛文本 | 领域符号被错误归一 |
| 去重 | 去完全重复或近似重复 | 采样语料、镜像站点 | 删掉必要的版本差异 |
| 噪声过滤 | 过滤模板页、短垃圾、异常字符 | 爬虫、日志、对话 | 删掉少见但有价值的专业文本 |

新手版边界可以这样理解：HTML 页面不能直接丢给模型，因为模型会把 `<script>`、导航栏、广告按钮也当成普通文本学进去。于是第一步常常是去标签、去脚本，再做语言和句子切分。但如果页面本身是技术文档，里面的 `<code>`、`<pre>`、JSON 片段可能恰好是核心信息，不能一刀切全部删光。这就是边界问题。

所以预处理流水线不是一个“万能清洗器”，而是一组围绕任务目标设计的保守操作。任务如果是问答系统，可能更重视句子边界和段落完整性；任务如果是大规模预训练，则更关心重复率、token 浪费和整体质量分布。

---

## 核心机制与推导

预处理的难点不在某一个模块，而在顺序耦合。所谓“顺序耦合”，就是前一步的输出会改变后一步的判断条件。白话说，同样的文本，先清理再判断，和先判断再清理，结果可能完全不同。

一个最典型的例子是“先去 HTML，再分句”。如果顺序反过来，句子切分器会先看到很多残缺标签、属性值和脚本片段，例如：

```text
<p>模型训练需要数据</p><script>track()</script>
```

如果先分句，可能得到 `"<p>模型训练需要数据</p><script>track()"` 这样的脏片段；后续去重和语言检测再基于这些片段工作，就容易误判。先清理再分句，才能让后续模块看到真正的自然语言。

可以把流水线简化成下面这个机制图：

`原始文本 -> 编码统一 -> 格式清理 -> 语言检测 -> 分句/分段 -> 标准化 -> 去重 -> 质量过滤 -> 输出`

每一步对指标的影响不同：

| 阶段 | 对 Retention 的影响 | 对 Duplicate Rate 的影响 | 对 Effective Token Ratio 的影响 |
|---|---|---|---|
| 编码统一 | 防止乱码导致无效丢弃 | 间接影响 | 提升有效 token 占比 |
| 格式清理 | 通常下降，因为删掉标签噪声 | 间接影响 | 显著提升 |
| 语言检测 | 下降，剔除非目标语言 | 几乎不变 | 提升任务相关性 |
| 去重 | 下降，删掉重复内容 | 直接决定 | 减少 token 浪费 |
| 质量过滤 | 进一步下降 | 可能略变 | 提升剩余文本密度 |

这几个指标不是孤立的。假设原始 token 数量为 $T_0$，格式清理后为 $T_1$，去重后为 $T_2$，质量过滤后为 $T_3$，那么：

$$
Retention = \frac{T_3}{T_0}
$$

$$
Effective\ Token\ Ratio = \frac{T_2}{T_1}
$$

如果你发现 $T_1$ 相比 $T_0$ 大幅下降，说明格式清理阶段删了很多东西，要检查是否误删正文；如果 $T_2$ 几乎等于 $T_1$，说明去重器可能没有抓住近似重复；如果 $T_3$ 再次剧烈下降，说明质量规则可能太激进。

玩具例子可以继续推导。设 1000 篇文档原始总 token 为 1,000,000。HTML 清理后剩 850,000，去重后剩 680,000，语言过滤后剩 620,000，质量过滤后剩 600,000。那么：

- $Retention = 600000 / 1000000 = 60\%$
- 若确认重复样本 250 篇，则 $Duplicate\ Rate = 250 / 1000 = 25\%$
- $Effective\ Token\ Ratio = 680000 / 850000 \approx 80\%$

这个结果通常比单看“最后剩多少篇”更有价值，因为篇数无法反映长文与短文差异，而 token 更接近模型真实消耗。

真实工程例子是网页预训练语料。爬虫抓下来的页面里，常见噪声包括导航栏、版权声明、分页按钮、镜像站重复页和多语言混排。工程上不会只靠一条正则处理，而是采用“规则 + 轻量模型 + 去重”的分层策略：规则先快速过滤极短文本、异常字符占比过高文本；轻量语言模型判断语种和可读性；最后再用近似去重处理大规模重复。这样做的原因很直接：规则快但粗，模型准一些但贵，去重最耗资源，所以要把明显垃圾尽量提前筛掉。

---

## 代码实现

下面给一个可以运行的 Python 玩具实现。它不追求工业级鲁棒性，但能把主流程串起来：读取文本、去 HTML、做简单语言检测、按规范归一、再基于哈希去重，并计算基础指标。

```python
import re
import html
import hashlib
from typing import List, Dict

def strip_html(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def detect_lang_simple(text: str) -> str:
    # 非工业实现，只做玩具示例
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    if cjk_count >= latin_count:
        return "zh"
    return "en"

def quality_score(text: str) -> float:
    if not text:
        return 0.0
    visible_chars = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))
    return visible_chars / max(len(text), 1)

def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def preprocess_pipeline(docs: List[str], target_lang: str = "zh") -> Dict[str, object]:
    raw_chars = sum(len(x) for x in docs)
    after_clean = []
    for doc in docs:
        cleaned = normalize_text(strip_html(doc))
        after_clean.append(cleaned)

    clean_chars = sum(len(x) for x in after_clean)

    lang_kept = []
    for doc in after_clean:
        if detect_lang_simple(doc) == target_lang:
            lang_kept.append(doc)

    seen = set()
    deduped = []
    duplicate_count = 0
    for doc in lang_kept:
        h = doc_hash(doc)
        if h in seen:
            duplicate_count += 1
            continue
        seen.add(h)
        deduped.append(doc)

    final_docs = [doc for doc in deduped if len(doc) >= 8 and quality_score(doc) >= 0.5]
    final_chars = sum(len(x) for x in final_docs)

    metrics = {
        "raw_chars": raw_chars,
        "clean_chars": clean_chars,
        "final_chars": final_chars,
        "retention": final_chars / raw_chars if raw_chars else 0.0,
        "duplicate_rate": duplicate_count / len(docs) if docs else 0.0,
        "effective_token_ratio": final_chars / clean_chars if clean_chars else 0.0,
        "docs_out": final_docs,
    }
    return metrics

docs = [
    "<html><body><h1>机器学习入门</h1><p>监督学习是一类从标注数据学习映射关系的方法。</p></body></html>",
    "<html><body><h1>机器学习入门</h1><p>监督学习是一类从标注数据学习映射关系的方法。</p></body></html>",
    "<div>Buy now!!! Limited offer!!!</div>",
    "<p>深度学习需要数据、算力和目标函数。</p>",
    "<p>Hello world, this is an English page.</p>",
]

result = preprocess_pipeline(docs, target_lang="zh")

assert result["raw_chars"] > result["final_chars"]
assert 0 <= result["retention"] <= 1
assert 0 <= result["duplicate_rate"] <= 1
assert len(result["docs_out"]) == 2
```

这段代码故意保持简单，目的是让每一步的责任可见。输入输出关系可以概括成表：

| 输入 | 操作 | 输出 |
|---|---|---|
| 原始 HTML 文本 | `strip_html` | 去标签后的正文 |
| 清洗后文本 | `detect_lang_simple` | 目标语言筛选结果 |
| 目标语言文本 | `doc_hash` | 完全重复检测键 |
| 候选文本 | `quality_score` | 质量过滤结果 |

如果要更接近真实工程，可以把这条链扩展成：

1. `remove_html` 或正文抽取器，去模板和脚本。
2. 编码统一成 UTF-8，记录非法字符比例。
3. 语言检测，常见实现是 `langid` 或 fastText 语言识别模型。
4. 分句或分段，确保后续训练样本长度合适。
5. 近似去重，常见方法是 shingling、MinHash、LSH。
6. 质量打分，例如压缩比、字符分布、token-type ratio。

真实工程例子：做预训练语料时，可能先用规则过滤字符异常页，再用轻量语言模型去掉非目标语言，最后在亿级文本上做 MinHash 去重。这样即使单个步骤都不完美，整体误差也能被分层吸收。

---

## 工程权衡与常见坑

最常见的坑不是“某个模型不准”，而是“流程设计没留后路”。预处理一旦把内容删掉，很多信息就不可逆了，所以每一步都应该能解释、能回放、能抽样复查。

下面是常见问题与规避方式：

| 常见坑 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| 去重过猛 | 合同、法规、配置模板大量消失 | 相似文本被当成同一份 | 加白名单，按领域单独设阈值 |
| 顺序错误 | 语言检测误判、分句混乱 | 脏 HTML 先进入后续模块 | 先格式清理，再做语言和分句 |
| 语言误判 | 中英混写文本被删 | 检测器只适合纯文本 | 对混写样本设置灰区，不直接删除 |
| 质量规则过窄 | 专业术语文档被判垃圾 | 稀有词比例高、符号多 | 面向领域重写规则 |
| 只看保留率 | 数据留下很多，但质量差 | 指标单一 | 同时看重复率和有效 token 占比 |
| 无日志 | 无法追查是哪一步删掉内容 | 没有阶段性记录 | 每一步输出计数和抽样日志 |

新手最容易忽略的是“领域感知”。所谓领域感知，就是规则要知道自己在处理什么类型的文本。白话说，同样是大量重复，新闻站的页脚是噪声，API 文档的参数表可能不是噪声；同样是符号很多，广告页可能是垃圾，数学文章却可能很有价值。

例如合同模板看起来高度重复，但版本号、责任条款、时间字段可能恰好是关键差异。如果只按字符相似度强行去重，就会把有效样本删成只剩一条。这个问题通常要靠 domain-aware whitelist 解决，也就是“按领域定义白名单”。白话说，先承认某些结构重复是正常的，不要默认它们就是垃圾。

日志记录至少要覆盖三类信息：样本数变化、token 变化、抽样样本。下面是一个极简思路：

```python
def log_stage(stage_name: str, docs_before, docs_after):
    print({
        "stage": stage_name,
        "count_before": len(docs_before),
        "count_after": len(docs_after),
        "chars_before": sum(len(x) for x in docs_before),
        "chars_after": sum(len(x) for x in docs_after),
        "sample_before": docs_before[:1],
        "sample_after": docs_after[:1],
    })
```

这类日志看起来简单，但非常关键。没有它，你只能看到“最后数据变少了”；有了它，才能回答“是 HTML 清理删多了，还是语言检测阈值太严格”。

---

## 替代方案与适用边界

不是所有项目都需要完整流水线。流水线的复杂度应该和数据规模、任务目标、预算一起决定。

如果是低资源场景，比如几千篇中文教程做一个小型分类器，完全可以用简化方案：HTML 清理、编码统一、语言检测、完全去重，已经足够。因为数据量不大，近似去重的收益有限，复杂质量模型也未必值回成本。

如果是大规模预训练或高质量问答库，就需要强化方案：正文抽取、句段切分、近似去重、质量评分、领域白名单、人工抽样复核。这时目标不是“能跑”，而是让每多保留一个 token 都尽量有学习价值。

可以用下面这张表看替代做法：

| 方案 | 主要做法 | 适用场景 | 可以放宽的维度 |
|---|---|---|---|
| 简化版 | HTML 清理 + 语言检测 + 完全去重 | 小数据集、原型验证 | 可不做近似去重 |
| 平衡版 | 规则过滤 + 语言检测 + 分句 + 去重 | 中等规模训练集 | 质量模型可以轻量化 |
| 强化版 | 分层过滤 + MinHash/LSH + 质量评分 + 白名单 | 预训练、大规模检索库 | 基础规则不能省 |
| 领域定制版 | 按领域定制清洗与去重策略 | 法律、医疗、代码、配置 | 通用阈值不能直接套用 |

这里也要强调适用边界。MinHash 和 LSH 适合大规模近似去重，但它们解决的是“找相似文本”的效率问题，不会自动理解什么相似值得删、什么相似必须留。对于小规模项目，shingling 加普通哈希就可能足够。所谓 shingling，可以先白话理解为“把文本切成重叠短片段，再比较这些片段是否相似”。

不同任务的目标也不同：

- 预训练任务更关注覆盖面和重复率，允许保留更多风格差异。
- 问答系统更关注可读性和段落边界，宁可少一些，也要稳定。
- 生成任务若面向垂直领域，更应该保留术语、格式和模板结构。

所以不能把某一套清洗阈值当成通用标准。最稳妥的方法是先定义任务目标，再反推希望达到的 retention、duplicate rate 和有效 token 占比范围。

---

## 参考资料

| 来源 | 核心贡献 | 可复用策略 |
|---|---|---|
| Latitude: Ultimate Guide to Preprocessing Pipelines for LLMs | 系统梳理文本预处理流水线的阶段与目标 | 用统一链路组织编码、格式、语言和噪声处理 |
| EMNLP 2024 Industry: Scaling Parameter-Constrained Language Models with Quality Data | 强调高质量数据、过滤层级和度量指标的重要性 | 用 retention、重复率、token 相关指标监控清洗效果 |
| Aman AI: Data Filtering / Quality Metrics Primer | 汇总规则过滤、质量度量、分层筛选思路 | 把规则、轻量模型和质量评分组合使用 |
| Scale/Exchange: Preprocessing Techniques in NLP | 介绍语言检测、标准化、分词分句等基础模块 | 先做基础清洗，再做语言和结构处理 |
| 关于去重误删的工程讨论文章 | 说明 dedup 过强会误删必要数据 | 为法律、模板、配置等高重复领域增加白名单 |

- Latitude: https://latitude.so/blog/ultimate-guide-to-preprocessing-pipelines-for-llms
- EMNLP 2024 Industry paper: https://aclanthology.org/2024.emnlp-industry.8/
- Aman AI primer: https://aman.ai/primers/ai/data-filtering/
- Scale Exchange blog: https://exchange.scale.com/en/public/blogs/preprocessing-techniques-in-nlp-a-guide
- 去重误删讨论文: https://medium.com/%40duckweave/dedupe-deletes-the-data-you-needed-9e4224f0da95
