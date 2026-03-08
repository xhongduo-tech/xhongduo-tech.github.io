## 核心结论

多语言数据采集与处理，核心不是“把更多网页抓回来”，而是“先判断每条文本属于哪种语言，再按语言差异决定后续采集、验证和格式化路径”。语言识别器就是做这件事的工具：它把一段文本映射为 ISO 语言代码，例如 `en` 表示英语、`zh` 表示中文、`am` 表示阿姆哈拉语。对工程系统来说，这一步相当于先分桶，再决定每个桶该走哪条流水线。

真正的难点在于分布失衡。网络上的语言分布，并不等于现实世界的说话人口分布。高资源语言，指公开文本、标注数据、工具链都很多的语言；低资源语言，指这些资源稀缺、线上可抓取内容少、社区工具不完整的语言。如果直接按网页量抓取，英语、西班牙语、法语会不断扩张，阿姆哈拉语、提格里尼亚语、老挝语之类会在最初的数据入口就被淹没。

工程上更可行的办法，是把 `fastText`、`langdetect`、`CLD3` 这类识别器放在入口，形成“检测→采样→验证→格式化”的闭环。检测阶段决定语种；采样阶段按脚本、资源量和任务价值动态配比；验证阶段做置信度复核、去重和编码检查；格式化阶段统一成可训练、可检索、可审计的数据结构。这样做的结果不是让所有语言一样多，而是避免低资源语言被系统性排除。

一个真实工程例子是紧急响应 SMS 分流。假设收到一批灾害相关短信，系统先逐条识别为 `en`、`zh`、`am`。英语走常规网页补采与现有语料合并；中文走现有问答库和论坛增补；阿姆哈拉语因为公开网页少，就转到 volunteer crowdsourcing，由本地志愿者或社区翻译网络补充。最后三路数据再统一做 UTF-8 编码、字段对齐、去重和质量打分。对零基础工程师来说，可以把它理解成一句话：先分好语种，再分别找最适配的采集渠道。

为了让这个结论更容易落地，可以把整个系统看成两个连续问题：

| 问题 | 要回答什么 | 失败后果 |
| --- | --- | --- |
| 识别问题 | 这条文本大概率是哪种语言、哪种脚本 | 走错下游流程，低资源语被误分走 |
| 资源分配问题 | 识别完后，预算和采样比例怎么分 | 高资源语越采越多，低资源语长期缺席 |

因此，语言识别不是独立模块，而是数据治理的入口控制器。只要入口没有把语言差异显式建模，后面的抓取、清洗、标注、训练都会持续放大偏差。

| 语言识别器 | 适合场景 | 采样目标 | 补充渠道 |
| --- | --- | --- | --- |
| fastText | 中长文本、批量离线处理 | 稳定给大规模语料打初标签 | 网页抓取、现有语料库 |
| langdetect | 轻量原型、小样本实验 | 快速分桶与统计频率 | 手工复核、问卷导入 |
| CLD3 | 短文本、混杂文本、移动端片段 | 强化短信/社媒短句识别 | 众包、社区采集、翻译会话 |

简易流程图：

```text
原始文本
  ↓
语言检测
  ↓
按脚本+资源量分桶
  ↓
目标采样 / 补充采集
  ↓
质量验证
  ↓
统一格式化与入库
```

---

## 问题定义与边界

这类系统要解决的问题，不是“识别语言”本身，而是“在语言分布严重偏移的情况下，如何稳定采到可用且多样的数据”。偏移的来源主要有两个。第一，Web 内容天然偏向高联网率国家和高商业价值语言。第二，同一种脚本内部也会出现强势语言挤压弱势语言，例如拉丁字母脚本下英语会掩盖很多非主流语言。

为了让问题可操作，通常需要先定义三个维度。语言代码，也就是 ISO 639 之类的标准缩写，用来统一标识语种。脚本，指语言书写所依赖的字符系统，比如 Latin、Arabic、Devanagari、Ethiopic。资源量，指该语言现有网页、语料、标注集、工具和社区支持的总量。真正的采样策略，往往不是只按语言名，而是按“语言代码 + 脚本 + 资源量等级”共同决定。

对新手来说，这三个维度可以这样理解：

| 维度 | 直白解释 | 例子 | 为什么单独看它不够 |
| --- | --- | --- | --- |
| 语言代码 | 这段文本属于哪种语言 | `zh`、`en`、`am` | 只知道语言，不知道它怎么写 |
| 脚本 | 这段文本用什么字符系统书写 | Latin、Arabic、Ethiopic | 同一种语言可能有多种书写形式 |
| 资源量 | 这个语言在线上和工具链里“富不富” | 英语高资源，阿姆哈拉语低资源 | 只知道资源量，不知道当前文本长什么样 |

一个常见判断指标是虚拟存在度：

$$
V_i=\frac{\text{Web 内容百分比}_i}{\text{全球 L1+L2 说话人百分比}_i}
$$

这里 $L1$ 是母语人数，$L2$ 是第二语言人数。$V_i>1$ 表示该语言在网络上的供给高于现实人口占比，属于“过剩”；$V_i<1$ 表示该语言在网络上供给不足，属于“稀缺”。这个指标不直接告诉你该采多少，但能告诉你“是否需要额外补偿”。

这个式子可以直接读成一句判断规则：

- 分子大，说明网上这种语言很多。
- 分母大，说明现实里会说这种语言的人很多。
- 比值大于 1，说明它在线上的存在感高于现实人口占比。
- 比值小于 1，说明它在线上的存在感低于现实人口占比。

玩具例子可以非常简单。假设有 300 条多语种问卷文本，语言检测后得到：Arabic 50 条、Hindi 30 条、English 220 条。如果只看数量，英语已经占多数；但如果 Arabic 和 Hindi 的后续目标任务是地区服务支持，那它们的单位样本价值可能更高。这时你至少要保证低资源桶里每个语种保留一定样本，而不是因为英文太多就把小语种过滤掉。

再看一个更具体的对比：

| 语言 | 检测到的数量 | 直接按数量采样会怎样 | 加入业务价值后应怎样 |
| --- | --- | --- | --- |
| English | 220 | 继续扩张，成本最低 | 适度保留即可 |
| Arabic | 50 | 容易被当成少量边角样本 | 重点保留，补充地区数据 |
| Hindi | 30 | 很快在清洗阶段被阈值过滤 | 设置最小保留量 |

问题边界也需要提前说清楚：

| 边界 | 说明 |
| --- | --- |
| 数据来源边界 | 只处理可合法访问的网络内容、开放平台数据和授权众包数据 |
| 采样下限边界 | 每个低资源语种至少保留一条及以上样本，不能在入口被清零 |
| 脚本边界 | 不能只依赖 Latin 脚本特征做识别，否则会漏掉 Arabic、Devanagari、Ethiopic 等 |
| 质量边界 | 识别结果不是事实本身，低置信度样本必须允许回退到人工复核 |

如果把边界说得再工程一点，可以再加三条常被忽略的约束：

| 边界 | 说明 |
| --- | --- |
| 编码边界 | 输入文本必须先统一为 UTF-8 或能无损转成 UTF-8 |
| 审计边界 | 每条样本要能追溯来源、检测结果、复核状态 |
| 隐私边界 | 短信、问卷、工单类文本要先做脱敏，再流转到标注或存储环节 |

对新手来说，最容易上手的做法是先按脚本分桶，再细分语言。比如一批问卷，先分为 Arabic 脚本、Devanagari 脚本、Latin 脚本三桶。Latin 桶可以优先网页抓取补样；Arabic 或 Ethiopic 桶如果网页极少，就直接转 crowdsourcing。只要手动做过两三个桶，就能理解这个系统的边界在哪里。

---

## 核心机制与推导

三类常见语言识别器的机制并不相同。`fastText` 常用于基于子词特征的文本分类，子词可以理解为“比单词更短、比字符更有结构的信息单元”；它对拼写变化、未登录词和中长文本比较稳。`langdetect` 基于字符 n-gram 和概率模型，n-gram 就是长度为 n 的连续字符片段；它实现轻、上手快，但对很短文本和混语文本容易波动。`CLD3` 采用神经化的字符 n-gram 表征，更擅长短句、网页片段和移动端文本。三者互补，而不是谁绝对替代谁。

如果只用一句话区分三者：

| 工具 | 核心特征 | 优势 | 弱点 |
| --- | --- | --- | --- |
| `fastText` | 子词分类 | 大规模离线标注稳定 | 对极短文本不占优 |
| `langdetect` | 传统字符统计 | 安装简单、原型快 | 结果波动更大 |
| `CLD3` | 神经化短文本识别 | 对短句和片段更友好 | 集成链路相对麻烦 |

实际管线通常这样设计：

1. 入口文本先跑一个主识别器，例如 fastText。
2. 对短文本、低置信度文本、脚本不明确文本，再用 CLD3 或第二识别器复核。
3. 根据语言代码、脚本和资源等级分桶。
4. 用虚拟存在度 $V_i$ 计算采样权重，抑制过剩语言，补偿稀缺语言。
5. 对低资源桶调用众包、社区项目或领域平台补样。
6. 对所有数据做统一验证、去重、编码和格式化。

这个机制里最重要的不是“识别一次”，而是“识别结果是否参与后续决策”。如果识别完只把结果打在日志里，不影响采样和路由，那系统实际上没有利用多语言信息。

采样权重最简单的推导是：

$$
w_i=\frac{1}{V_i}
$$

如果某语言在网上已经很多，$V_i$ 大，权重就小；如果某语言在线上稀缺，$V_i$ 小，权重就大。工程里通常不会直接无限放大，而会加上下界和上界，例如：

$$
w_i=\text{clip}\left(\frac{1}{V_i}, w_{\min}, w_{\max}\right)
$$

这样可以避免极低资源语言因样本太少而被权重放大到不稳定，也能避免英语这类高资源语言被压到完全不可用。

仅靠 $V_i$ 仍然不够，因为它只回答“稀不稀缺”，没有回答“任务重不重要”。一个更常见的工程写法是把业务价值和质量风险一起乘进去：

$$
W_i=\text{clip}\left(\frac{1}{V_i}\times B_i\times Q_i,\; w_{\min},\; w_{\max}\right)
$$

其中：

- $B_i$ 表示业务价值系数，例如某语言直接关系到当地服务覆盖，则 $B_i$ 更高。
- $Q_i$ 表示质量修正系数，例如某语言当前识别误判很多、来源质量不稳，则先降低扩采速度。
- `clip` 表示把结果限制在可控区间内，避免某一个系数把整体打爆。

对新手来说，可以把这个式子理解成三步：

1. 先看它在线上是不是被低估。
2. 再看这个语言对当前任务是不是重要。
3. 最后看当前这批数据值不值得继续放大。

再进一步，还可以参考网络全球化指数：

$$
CGI(L)=\frac{L1+L2}{L1}(L)\times S(L)\times C(L)
$$

其中 $\frac{L1+L2}{L1}(L)$ 表示该语言的跨母语传播能力，$S(L)$ 表示覆盖国家比例，$C(L)$ 表示联网率或接入能力。它不是采样权重本身，但可以辅助判断某语种未来在线增长潜力。如果一个语言当前网页少，但 $CGI(L)$ 较高，说明未来更值得提前布局工具与采集渠道。

这两个公式可以分工理解：

| 公式 | 关注点 | 用途 |
| --- | --- | --- |
| $V_i$ | 当前线上供给是否失衡 | 决定要不要补偿 |
| $W_i$ | 结合业务价值后的实际采样力度 | 决定现在采多少 |
| $CGI(L)$ | 未来增长潜力 | 决定是否提前铺基础设施 |

玩具例子如下。假设检测后得到：

- `en`: 700 条，$V_{en}=1.4$
- `zh`: 180 条，$V_{zh}=0.9$
- `am`: 20 条，$V_{am}=0.2$

那么原始权重大致为：

- $w_{en}=1/1.4\approx0.71$
- $w_{zh}=1/0.9\approx1.11$
- $w_{am}=1/0.2=5$

这表示英语应被限流，中文基本维持，阿姆哈拉语需要强补偿。此时系统就不应该再继续花主要预算去抓英文网页，而是把阿姆哈拉语转向志愿者采集、双语对译会话或者社区平台。

如果再加入业务价值，例如灾区工单主要来自阿姆哈拉语社区，可以设：

- $B_{en}=1.0$
- $B_{zh}=1.0$
- $B_{am}=1.5$

并且为了防止阿姆哈拉语样本质量过低，设：

- $Q_{en}=1.0$
- $Q_{zh}=1.0$
- $Q_{am}=0.8$

那么：

$$
W_{am}=\text{clip}(5\times1.5\times0.8,\;0.5,\;4.0)=4.0
$$

也就是它仍会被顶到上限，说明即使考虑质量风险，它依然值得强补偿。

真实工程例子是人道主义短信处理。灾害事件发生后，系统先对 SMS、社媒短帖、求助表单做识别。如果 `am`、`ti`、`so` 等低资源语言条目偏少，但任务又高度依赖本地语言，系统就不能只按网页量增补，而应通过社区翻译网络、志愿者标注和领域平台补足。最后统一进入同一验证栈，包括重复内容压缩、时间戳标准化、编码一致性检查和敏感信息脱敏。

---

## 代码实现

下面给一个可运行的简化实现。它不直接调用真实识别器，而是用一个小规则模拟“检测→加权→路由→格式化”流程，重点是把工程结构讲清楚。真实项目里，只需要把 `detect_lang_with_confidence` 替换成 fastText、langdetect 或 CLD3 的调用。

```python
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple
import json
import unicodedata


LANG_TO_SCRIPT = {
    "en": "Latin",
    "zh": "Han",
    "am": "Ethiopic",
    "unknown": "Unknown",
}

LOW_RESOURCE_LANGS = {"am"}


def normalize_text(text: str) -> str:
    """统一兼容字符、空白和首尾空格。"""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.strip().split())


def detect_lang_with_confidence(text: str) -> Tuple[str, float]:
    """
    简化版语言检测。
    规则很粗糙，但可运行，适合演示完整流程。
    """
    if not text:
        return "unknown", 0.0

    has_han = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    has_ethiopic = any("\u1200" <= ch <= "\u137f" for ch in text)
    latin_letters = sum(("a" <= ch.lower() <= "z") for ch in text)

    if has_han:
        return "zh", 0.98
    if has_ethiopic:
        return "am", 0.96
    if latin_letters >= 3:
        return "en", 0.85
    return "unknown", 0.40


def second_pass_detector(text: str) -> Tuple[str, float]:
    """
    二次复核器。
    真实工程中可换成 CLD3 或另一套模型。
    """
    # 这里仍然用同一套简化规则，只是人为给出稍保守的置信度
    lang, conf = detect_lang_with_confidence(text)
    return lang, min(conf, 0.80)


def get_script(lang: str) -> str:
    return LANG_TO_SCRIPT.get(lang, "Unknown")


def get_resource_level(lang: str) -> str:
    if lang in LOW_RESOURCE_LANGS:
        return "low"
    if lang in {"en", "zh"}:
        return "high"
    return "medium"


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_weight(virtual_presence: float, business_value: float, quality_factor: float) -> float:
    raw = (1.0 / virtual_presence) * business_value * quality_factor
    return clip(raw, 0.5, 4.0)


def route_by_bucket(lang: str, resource_level: str, confidence: float) -> str:
    if lang == "unknown" or confidence < 0.70:
        return "manual_review"
    if resource_level == "low":
        return "crowd"
    return "web"


def redact_pii(text: str) -> str:
    """
    示例级脱敏：把纯数字串替换为 <NUM>。
    真实工程应做手机号、邮箱、地址等更细规则。
    """
    tokens = []
    for token in text.split():
        if token.isdigit():
            tokens.append("<NUM>")
        else:
            tokens.append(token)
    return " ".join(tokens)


def process_corpus(
    corpus: List[str],
    virtual_presence: Dict[str, float],
    business_value: Dict[str, float],
    quality_factor: Dict[str, float],
) -> Tuple[Counter, List[Dict[str, object]]]:
    stats = Counter()
    routed_records: List[Dict[str, object]] = []

    for raw_text in corpus:
        normalized = normalize_text(raw_text)
        lang, confidence = detect_lang_with_confidence(normalized)

        if confidence < 0.70:
            lang, confidence = second_pass_detector(normalized)

        script = get_script(lang)
        resource_level = get_resource_level(lang)
        route = route_by_bucket(lang, resource_level, confidence)

        vp = virtual_presence.get(lang, 1.0)
        bv = business_value.get(lang, 1.0)
        qf = quality_factor.get(lang, 1.0)
        weight = compute_weight(vp, bv, qf)

        record = {
            "text": redact_pii(normalized),
            "lang": lang,
            "script": script,
            "resource_level": resource_level,
            "confidence": round(confidence, 2),
            "weight": round(weight, 2),
            "channel": route,
        }

        stats[lang] += 1
        routed_records.append(record)

    return stats, routed_records


def main() -> None:
    corpus = [
        "Need clean water urgently",
        "需要紧急医疗支援",
        "እባክዎ ምግብ ያስፈልጋል",
        "Need shelter for 23 children",
        "   需要   饮用水   和  药品   ",
    ]

    virtual_presence = {
        "en": 1.42,
        "zh": 0.90,
        "am": 0.20,
        "unknown": 1.00,
    }

    business_value = {
        "en": 1.00,
        "zh": 1.00,
        "am": 1.50,
        "unknown": 0.80,
    }

    quality_factor = {
        "en": 1.00,
        "zh": 1.00,
        "am": 0.80,
        "unknown": 0.60,
    }

    stats, routed = process_corpus(
        corpus=corpus,
        virtual_presence=virtual_presence,
        business_value=business_value,
        quality_factor=quality_factor,
    )

    assert stats["en"] == 2
    assert stats["zh"] == 2
    assert stats["am"] == 1

    assert any(item["channel"] == "crowd" and item["lang"] == "am" for item in routed)
    assert any(item["text"] == "Need shelter for <NUM> children" for item in routed)
    assert all(item["text"] == normalize_text(item["text"]) for item in routed)

    print("Language counts:")
    print(dict(stats))
    print("\nProcessed records:")
    print(json.dumps(routed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

这段代码可以直接运行，输出结果会类似这样：

```text
Language counts:
{'en': 2, 'zh': 2, 'am': 1}

Processed records:
[
  {
    "text": "Need clean water urgently",
    "lang": "en",
    "script": "Latin",
    "resource_level": "high",
    "confidence": 0.85,
    "weight": 0.7,
    "channel": "web"
  },
  {
    "text": "需要紧急医疗支援",
    "lang": "zh",
    "script": "Han",
    "resource_level": "high",
    "confidence": 0.98,
    "weight": 1.11,
    "channel": "web"
  },
  {
    "text": "እባክዎ ምግብ ያስፈልጋል",
    "lang": "am",
    "script": "Ethiopic",
    "resource_level": "low",
    "confidence": 0.96,
    "weight": 4.0,
    "channel": "crowd"
  }
]
```

这段实现对应四个模块，而不是只有一个“识别函数”：

| 模块 | 作用 | 代码里的位置 | 真实工程替换点 |
| --- | --- | --- | --- |
| 识别模块 | 给文本打语言标签并返回置信度 | `detect_lang_with_confidence` | 接入 fastText、langdetect、CLD3 |
| 复核模块 | 对低置信度文本再次判断 | `second_pass_detector` | 接入第二模型或人工复核队列 |
| 路由模块 | 根据语言、脚本、资源量决定渠道 | `route_by_bucket` | 接入 crowd API、爬虫任务队列 |
| 规范化模块 | 统一编码、空白、脱敏 | `normalize_text`、`redact_pii` | 增加去重、日志、字段校验 |

如果写成更接近生产环境的伪代码，大致如下：

```python
for chunk in corpus:
    chunk = normalize_text(chunk)
    lang, conf = detect_lang(chunk)

    if conf < 0.7:
        lang, conf = second_pass_detector(chunk)

    script = get_script(chunk, lang)
    resource = get_resource_level(lang)
    bucket = build_bucket(lang=lang, script=script, resource=resource)

    if bucket.is_low_resource:
        send_to_crowd(chunk, lang=lang, script=script)
    elif conf < 0.7:
        send_to_manual_review(chunk, lang=lang)
    else:
        enqueue_web_scraper(chunk, lang=lang)

normalize_and_dedup()
validate_encoding()
redact_sensitive_info()
log_metrics()
```

如果你是第一次写这种系统，建议先按下面顺序实现，而不是一开始就上完整平台：

1. 先做“输入文本 → 语言标签”。
2. 再做“语言标签 → 采集渠道”。
3. 然后补“归一化、脱敏、日志”。
4. 最后才加“权重调度、二次复核、人工回退”。

监控项至少应包括：每个语种采集数量、每个渠道产出数量、识别置信度分布、被人工回退的比例、去重后保留率、编码异常率。没有这些日志，后面你就无法知道“低资源语种是真的少，还是识别器把它误判掉了”。

还可以把监控拆成三组，方便排查：

| 监控组 | 关键指标 | 用来发现什么问题 |
| --- | --- | --- |
| 识别监控 | 语种分布、置信度分布、二次复核率 | 识别器是否漂移 |
| 数据监控 | 去重率、空文本率、乱码率 | 清洗是否破坏原始数据 |
| 路由监控 | `web/crowd/manual_review` 占比 | 资源是否被错误分配 |

---

## 工程权衡与常见坑

最大工程坑通常不在识别，而在 tokenizer。tokenizer 是把文本切成模型可处理片段的组件。很多多语言 tokenizer 对 Latin 脚本更友好，对非 Latin 脚本会切得更碎，形成更高的 token fertility，也就是“同样一段信息，被拆成更多 token”。这会直接导致序列更长、推理更贵、对齐更差。

例如同一条告警消息，英语可能被切成 12 个 token，俄语被切成 25 个，阿姆哈拉语或缅甸语可能更高。对按 token 计费的 API 或定长上下文模型来说，成本和截断风险都会上升。新手最容易忽略这一点，因为表面看只是“字符不一样”，但对模型来说是“序列长度翻倍”。

可以用一个很简单的比值描述这种差异：

$$
F(L)=\frac{\text{token 数}(L)}{\text{语义等价英文 token 数}}
$$

其中 $F(L)$ 越大，说明该语言在当前 tokenizer 下越“吃亏”。如果某种语言表达同样的信息总是需要更多 token，那么它在训练、推理、检索中的单位成本都会更高。

| 问题 | 影响 | 对策 |
| --- | --- | --- |
| Token fertility 上升 | API 成本高、序列更长 | Script-aware tokenizer |
| 跨语言对齐误差 | 平行语料错位、检索效果差 | Token alignment/transfer |
| Vocabulary 爆裂 | 词表膨胀、训练不稳定 | 动态 vocabulary 管理 |
| 混合编码与兼容字符 | 去重失败、识别漂移 | UTF-8 统一、NFKC 归一化 |
| 短文本误判 | 路由错误、低资源样本流失 | 双检测器复核、置信度阈值 |

另一个坑是把“语言识别结果”当作最终真值。现实里常见混语、转写和拼写变体。比如阿拉伯语内容可能夹英文缩写，印地语可能用 Latin 转写输入，社媒文本还会有表情、重复字符、错拼。对这些内容，单一识别器经常不稳定，所以应允许二次检测和人工回退。

下面是几个常见误判来源：

| 场景 | 例子 | 为什么容易错 |
| --- | --- | --- |
| 混语 | `Need water ASAP 请尽快处理` | 两种语言同时出现 |
| 转写 | `namaste doston` | 语言是印地语，但脚本是 Latin |
| 极短文本 | `ok`、`hi`、`no` | 信息量太低 |
| 噪声文本 | `heeellppp!!!` | 重复字符和符号破坏统计特征 |

还有一个典型错误是只按语言，不按脚本建策略。塞尔维亚语可能出现 Latin 和 Cyrillic 两种书写；印地语和印地语转写文本在 tokenizer 和识别器上的行为完全不同。如果你不把脚本视为一等公民，就会在采样和规范化阶段遇到很多隐蔽错误。

另一个经常发生的问题是“入口阈值过硬”。比如把低于 `0.8` 置信度的样本全部丢弃。这样做对英语新闻可能问题不大，但对短短信、口语化问卷、低资源语社媒文本，会直接造成系统性漏召回。更稳妥的做法不是直接丢，而是改路由：

| 置信度区间 | 建议动作 |
| --- | --- |
| `>= 0.85` | 直接进入自动流程 |
| `0.70 ~ 0.85` | 第二识别器复核 |
| `< 0.70` | 人工抽检或低风险延迟处理 |

最后一个常见坑是没有把“来源差异”纳入评估。网页、短信、论坛、问卷、字幕、OCR 文本的噪声形态完全不同。你在干净网页上调出来的阈值，放到短信流里通常会失效。因此识别器评估集至少要按来源拆分，而不是只看一份总准确率。

---

## 替代方案与适用边界

采集渠道没有“最好”，只有“在当前语言和任务下更合适”。网页抓取速度最快，但天然偏向主导语言。众包质量可控，但成本和时间更高。社区驱动平台覆盖特定语种更好，但响应速度和数据结构不一定统一。工程上最好把它们视为可切换的补充渠道，而不是单选题。

| 渠道 | 优点 | 局限 | 适用条件 |
| --- | --- | --- | --- |
| 自动网页抓取 | 规模大、速度快 | 偏向高资源语言、噪声高 | 主流语言、资讯类文本 |
| Crowdsourcing | 可定向补低资源语 | 慢、要做质检 | 低资源语、任务型文本 |
| Common Voice | 社区驱动、适合语音 | 主要是语音，不是通用文本 | 语音识别、口语数据 |
| Tatoeba | 多语句对丰富 | 句子短、领域受限 | 对译、基础并行语料 |
| Lanfrica | 聚焦非洲语言 | 覆盖面区域性强 | 非洲低资源语 |
| DEEP/HumSet | 适合人道主义数据协同 | 平台接入与规范要求更高 | 灾害响应、社会治理类任务 |

一个简单的切换规则可以是：某语种采集量少于 100 条时，不再继续等网页自然积累，而是直接转向 Tatoeba、志愿者社区或 crowdsourcing。这个阈值不是固定真理，但它能让系统在低资源语种上尽早触发人工和社区补偿，而不是一直被动等待。

还可以把这个切换规则写得更明确一点：

| 当前状态 | 推荐渠道 |
| --- | --- |
| 高资源语、文本量充足 | 网页抓取为主 |
| 低资源语、文本量不足 | 众包或社区采集为主 |
| 有原文但缺平行译文 | 翻译会话或双语标注 |
| 短文本噪声高 | 人工抽检 + 二次复核 |

还有一种替代策略是“识别→翻译会话”。也就是先识别出低资源语言文本，再发起人工或半自动翻译对话，把原文和译文一起回灌到数据管线。它不等于直接机器翻译洗数据，而是把翻译过程当成数据采集和验证的一部分。对完全无法稳定抓取原生网页的语种，这种方式常常比盲目抓网更有效，但前提是你能接受更高的人力成本和更低的吞吐量。

这种方案的优点和限制可以单独看：

| 方案 | 优点 | 限制 |
| --- | --- | --- |
| 识别后直接抓网 | 吞吐量高 | 低资源语覆盖差 |
| 识别后发起翻译会话 | 原文和译文可同时积累 | 成本更高、速度更慢 |
| 识别后转社区采集 | 更贴近本地语言实际用法 | 需要社区组织能力 |

适用边界也要明确。如果任务目标是构建开放领域预训练语料，网页抓取仍然是主力；如果目标是灾害响应、问卷分析、社区服务，低资源语的定向采集权重就应该更高。不要把一个面向海量预训练的策略，硬套到一个强调语言公平性的业务场景里。

可以用一张表快速判断“当前任务更像哪一类”：

| 任务目标 | 主优先级 | 更合适的策略 |
| --- | --- | --- |
| 开放领域预训练 | 规模 | 网页抓取 + 去重清洗 |
| 客服/工单分流 | 准确路由 | 识别 + 二次复核 + 领域词表 |
| 灾害响应 | 低资源语覆盖 | 社区采集 + 翻译会话 |
| 多语检索 | 对齐质量 | 平行语料 + 统一格式化 |

---

## 参考资料

| 类别 | 资源 | 用途 |
| --- | --- | --- |
| 工具文档 | fastText language identification | 了解批量语言识别模型与标签格式 |
| 工具文档 | `langdetect` PyPI | 轻量原型与 Python 接入 |
| 工具文档 | CLD3 GitHub / Python bindings | 短文本与网页片段语言识别 |
| 指标 | OBDILCI / Internet Language Observatory V6 | 查看 Web 内容占比、说话人口占比、虚拟存在度 |
| 调研 | Frontiers: humanitarian NLP survey | 理解低资源语、人道主义场景与数据采集闭环 |
| 综述 | Emergent Mind: multilingual tokenization | 理解多语言 tokenizer 偏差与工程对策 |
| 调查 | Tokenization Mismatches Across Languages | 理解跨语言切分失衡与下游影响 |

工具文档：
- fastText language identification：适合建立第一层批量识别能力，重点看支持的语言标签、模型输入格式和批量推理方式。
- `langdetect`：适合原型验证，重点看 API 调用方式和短文本表现。
- CLD3：适合短文本与网页片段，重点看语言覆盖范围、概率输出和绑定方式。

指标与数据分布：
- OBDILCI V6：可用来估计网页内容占比与说话人口占比的偏差，是虚拟存在度思路的直接参考源。
- Internet Language Observatory 相关报告：可辅助理解语言在线存在度不等于人口规模。
- 公开人口与联网率报告：可补充 $CGI(L)$ 中的覆盖国家和接入能力信息。

调研与综述：
- 人道主义 NLP 调研：重点看低资源语言在灾害场景、公共服务场景中的标注与采集约束。
- 多语言 tokenization 综述：重点看不同脚本在 token 长度、词表覆盖、下游任务上的不平衡。
- Tokenization mismatch 调查：重点看平行语料错位、跨语检索误差和生成长度偏差。

建议阅读顺序如下：

1. 先看工具文档，搞清楚识别器能输出什么。
2. 再看语言分布指标，理解为什么不能只按网页数量采样。
3. 最后看 tokenization 和人道主义场景资料，理解后续训练和应用阶段会放大哪些问题。

如果只保留最小参考集合，至少应包括四类：

| 最小集合 | 作用 |
| --- | --- |
| 一个识别器文档 | 知道如何把文本转成语言标签 |
| 一个 Web 分布指标源 | 知道哪些语言被线上高估或低估 |
| 一个低资源场景调研 | 知道为什么要引入社区或众包 |
| 一个 tokenization 综述 | 知道后续模型阶段的成本偏差从哪里来 |
