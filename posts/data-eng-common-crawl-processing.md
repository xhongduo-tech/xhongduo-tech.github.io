## 核心结论

Common Crawl 是一套公开的网页抓取归档。白话说，它不是“已经清洗好的训练集”，而是“能自己做训练集的原材料仓库”。截至 2026 年 4 月 14 日，官方 `Get Started` 页面列出的可用主抓取包含 `CC-MAIN-2026-12`、`CC-MAIN-2026-08`、`CC-MAIN-2026-04` 等版本，数据存放在 AWS `us-east-1` 区域，可通过 `s3://commoncrawl/` 或 `https://data.commoncrawl.org/` 访问。

对工程实现最重要的结论有三点。

1. 先分清三层格式，再决定处理入口。`WARC` 是原始 HTTP 响应包，`WAT` 是元数据层，`WET` 是抽出的纯文本层。做快速实验优先从 `WET` 开始，做高质量训练语料通常回到 `WARC` 重新抽正文。
2. 处理链路本质上是一个筛选流水线：下载分片 → 抽正文 → 语言识别 → 质量过滤 → 去重 → 分桶输出。顺序不能乱，因为后一步通常依赖前一步降低噪声和规模。
3. 大规模作业最优实践是在 `us-east-1` 就地处理。白话说，数据在哪个机房，你的计算最好也放在哪个机房，否则会遇到额外网络费用、吞吐变差，以及 S3 访问细节带来的报错。

| 格式 | 内容 | 适用阶段 |
| --- | --- | --- |
| WARC | 原始 HTTP 响应、请求信息、部分元数据 | 原生抽取、重跑正文解析、保留最大信息量 |
| WAT | 从 WARC 计算出的元数据，常见为 JSON | 链接图、HTTP 头分析、URL 索引 |
| WET | 从 WARC 提取出的纯文本 | 快速 NLP 实验、语言检测、轻量过滤 |

一个最小上手命令是直接走 HTTP 下载单个样本：

```bash
curl https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-12/segments/1674172666062.83/warc/CC-MAIN-20260122054713-20260122070247-00000.warc.gz -o sample.warc.gz
```

这条命令的意义不是“下载完整数据”，而是先验证你的网络、路径格式和本地解压链路是否正常。

---

## 问题定义与边界

“处理 Common Crawl 数据”不是单一动作，而是把超大规模网页归档变成可用文本语料。白话说，你面对的不是几万个文件，而是按月发布、每次包含数十亿页面的网页快照。真正的问题不是“能不能下载”，而是“用什么入口、在什么规模、按什么标准过滤”。

这里的边界要先划清。

| 访问方式 | 适用场景 | 限制 |
| --- | --- | --- |
| `s3://commoncrawl/...` | AWS 内部作业，EC2、EMR、Spark 集群 | 需认证访问；Requester Pays 请求要显式声明 |
| `https://data.commoncrawl.org/...` | 本地调试、小样本验证 | 速度慢，不适合大批量全量同步 |
| Columnar Index / Athena | 先查 URL 再回读 WARC 片段 | 配置更复杂，适合精准检索，不适合新手直接入门 |

如果你从 S3 读数据，还要理解 `Requester Pays`。白话说，S3 需要你明确承认“请求费用由我承担”。AWS 文档要求下载请求带上 `x-amz-request-payer: requester`，否则常见结果就是 `403`。AWS CLI 的等价写法是：

```bash
aws s3 cp s3://commoncrawl/crawl-data/CC-MAIN-2026-12/segments/.../wet/...wet.gz . --request-payer requester
```

因此，工程边界通常是这样划分的：

- 本地机器：只做样本验证、解析器调试、规则联调。
- AWS `us-east-1`：做真正的批处理、质量过滤、去重和落盘。
- 不建议：在其他区域全量拉取再处理，因为主要成本常常不是 CPU，而是跨区网络和低效 I/O。

一个“玩具例子”可以说明边界。假设你只想理解 WET 长什么样，那么下载一个 `100MB` 级别的样本就够了；假设你要做一个英文预训练子集，目标是 `5TB` 原始文本，那就必须上分布式集群，否则单机根本不是正确起点。

---

## 核心机制与推导

Common Crawl 处理链路可以抽象成下面这个函数组合：

$$
\text{raw crawl} \xrightarrow{\text{extract}} \text{text}
\xrightarrow{\text{lang detect}} \text{language bucket}
\xrightarrow{\text{quality filter}} \text{clean text}
\xrightarrow{\text{dedup}} \text{training corpus}
$$

这里每个词都要理解清楚。

- 正文抽取：从 HTML 中找出主内容。白话说，就是把导航栏、评论区、页脚、广告块尽量剥掉。
- 语言识别：判断文本属于哪种语言。白话说，就是别把英文、法文、乱码混在一起训练。
- 质量过滤：用规则或语言模型打分。白话说，就是尽量删掉模板页、列表页、垃圾页、重复页。
- 去重：删除相同或近似相同内容。白话说，就是同一篇文章不要在训练集里出现十遍。

CCNet 是这一类流水线的典型代表。它的思路不是追求最复杂的抽取，而是先把 Common Crawl 变成可控的单语语料。常见步骤是：

1. 读取 WET 文本段。
2. 做归一化：小写化、数字占位、去重音符。
3. 对归一化结果做哈希，做精确去重。
4. 做语言识别，筛出目标语言。
5. 用 KenLM 计算困惑度。困惑度是语言模型对文本“顺不顺”的打分，数值越低通常越自然。
6. 按阈值保留高质量文档，并按语言、质量桶输出。

可以写成更接近实现的形式：

$$
h(p) = \text{SHA1}(\text{normalize}(p))[:64]
$$

$$
\text{keep}(p) =
\mathbf{1}\big[\text{lang}(p)=L \land \text{ppl}(p) < \tau \big]
$$

其中：

- `normalize` 是文本标准化。
- `SHA1` 是哈希函数。白话说，它把一段文本压成一个固定长度的指纹。
- `ppl` 是 perplexity，困惑度。
- $\tau$ 是质量阈值。

看一个玩具例子。假设某个 `1 GB` 的 WET 分片里有 `1,000,000` 个段落：

| 步骤 | 剩余段落数 | 说明 |
| --- | ---: | --- |
| 原始读取 | 1,000,000 | 还没做任何清洗 |
| 归一化 + 精确去重 | 300,000 | 重复模板、镜像页、大量转载被合并 |
| 语言识别保留中文/英文目标集 | 240,000 | 去掉其他语言和识别不稳定样本 |
| 质量阈值过滤 | 120,000 | 去掉乱码、短页、列表页、低自然度文本 |

这个数字不是固定真值，但它解释了为什么“先筛后算”很重要。你不应该先对一百万段文本都跑重模型，再去重；正确做法是尽量先用便宜规则缩小规模。

真实工程里，像 RedPajama-V2 这样的数据工程会把多个 Common Crawl 快照统一处理，产出按 `snapshot / shard / language / bucket` 组织的结果文件。白话说，输出不是一个大文件，而是一大批按语言和质量层次整理好的分片，这样下游训练可以按桶采样，也能做不同质量配比。

---

## 代码实现

初学者最容易犯的错，是一上来就去搭 Spark 集群。更稳妥的做法是先在单机上验证三个动作：读记录、做归一化、做去重判定。下面这个 Python 例子不依赖 Common Crawl 专用库，但它复现了 CC 类流水线里最核心的一段逻辑。

```python
import hashlib
import re
import unicodedata
from typing import Iterable, List

def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\d", "0", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fingerprint64(text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]

def dedup_texts(texts: Iterable[str]) -> List[str]:
    seen = set()
    kept = []
    for text in texts:
        fp = fingerprint64(text)
        if fp not in seen:
            seen.add(fp)
            kept.append(text)
    return kept

samples = [
    "Hello 2025 World!",
    "hello 2026 world!",
    "Café costs 123 dollars",
    "Cafe costs 999 dollars",
    "A completely different paragraph."
]

result = dedup_texts(samples)

assert normalize_text("Café 123") == "cafe 000"
assert fingerprint64("Hello 2025 World!") == fingerprint64("hello 2026 world!")
assert len(result) == 3
assert "A completely different paragraph." in result
```

这个例子说明两件事。

1. “归一化后去重”比“原文直接去重”更接近真实工程。
2. 数字占位和去重音符会主动扩大“相同文本”的定义，这能清掉模板页，但也可能误杀合法内容。

如果你要从原始 `WARC` 直接抽正文，NVIDIA NeMo Curator 给了更接近生产的管线接口。它把流程拆成 URL 生成、下载、遍历记录、HTML 抽取四步，支持 `Trafilatura`、`jusText`、`Resiliparse` 等正文提取器。一个小规模验证可以写成：

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.download import CommonCrawlDownloadExtractStage
from nemo_curator.stages.text.download.html_extractors import TrafilaturaExtractor
from nemo_curator.stages.text.io.writer import JsonlWriter

stage = CommonCrawlDownloadExtractStage(
    start_snapshot="2020-50",
    end_snapshot="2020-50",
    download_dir="./downloads",
    html_extraction=TrafilaturaExtractor(min_extracted_size=200),
    url_limit=10,
)

pipeline = Pipeline(name="cc-text", stages=[stage, JsonlWriter("./out")])
pipeline.run()
```

真实工程例子通常会再往前走一步：不是只抽 `10` 个 URL，而是在 `us-east-1` 上开 Spark 或 Ray 作业，批量跑完整个快照。Common Crawl 官方的 `cc-pyspark` 仓库就提供了从 WARC/WAT/WET 读取、统计、抽取并落成表的示例，适合用来做分布式模板。

---

## 工程权衡与常见坑

最常见的工程分歧，是“直接用 WET”还是“回到 WARC 重抽正文”。

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 直接用 WET | 开发快、I/O 小、实现简单 | 容易保留导航、脚注、广告和模板噪声 | 快速实验、语言识别、原型验证 |
| 从 WARC 重抽 | 质量更高，可控性更强 | CPU 成本高，解析链更复杂 | 训练语料、长期数据资产建设 |

坑主要集中在五类。

1. `403` 访问错误  
原因常常不是路径错，而是没有显式声明 `Requester Pays`。S3 路径、IAM 配置、请求头三者缺一不可。

2. 跨区处理导致账单异常  
Common Crawl 数据在 `us-east-1`。如果你在别的区域拉大批量数据，网络费用可能比算力贵。

3. WET 不是“干净文本”  
WET 是提取过的纯文本，不是人工校验过的高质量正文。目录页、版权声明、推荐阅读、面包屑导航都可能还在。

4. 精确去重不等于近似去重  
哈希去重只能删掉“完全一样或归一化后完全一样”的文本。对改写转载、局部拼接、模板轻改页面，需要 `MinHash` 或 LSH 这类近似去重方法。

5. 过滤过强导致数据分布失真  
如果你把“短文本、代码片段、表格页面、论坛问答”全判成低质量，最后得到的可能是一个过于单一的百科化语料，而不是真实网页分布。

这也是为什么 FineWeb 一类方案选择从 `WARC` 重新做 `Trafilatura + fastText + 质量规则 + MinHash`。它牺牲了算力成本，换来更干净的文本输入。这种取舍在预训练里通常是值得的，因为一次高质量清洗会影响整个模型的学习上限。

---

## 替代方案与适用边界

不是所有团队都应该从原始 Common Crawl 开始。对很多初级工程师来说，更现实的问题是：我到底该处理原始数据，还是直接用别人处理好的版本？

| 数据集/方案 | 处理基础 | 适用边界 |
| --- | --- | --- |
| CCNet | 主要基于 WET，做语言识别、困惑度分桶、去重 | 多语语料构建、轻处理流水线 |
| FineWeb | 从 WARC 重抽正文，再做语言与质量过滤、MinHash | 英文高质量训练集 |
| RedPajama-V2 | 基于 CCNet 产出文本，再附带大量质量信号 | 想复用公开质量特征、按桶重加权训练 |
| 直接用 Common Crawl 原始层 | WARC/WAT/WET 全量原料 | 需要自定义规则、做可复现实验平台 |

适用边界可以直接按目标来选。

- 目标是“理解格式、跑通链路”  
先下一个 WET 或 WARC 样本，单机解析就够。

- 目标是“做课程项目或小模型实验”  
优先考虑现成的 FineWeb、RedPajama-V2、OSCAR 之类处理后数据，别一开始就吞整套原始归档。

- 目标是“搭团队内部数据工厂”  
才有必要从原始 Common Crawl 做一遍自定义清洗，并把规则、版本、去重策略、质量阈值全部产品化。

一个真实工程判断标准是：如果你的目标是“拿到英文高质量网页文本训练一个中小模型”，直接复用 FineWeb 往往比重做整套 Common Crawl 更划算；如果你的目标是“要做中文、技术博客、论坛、代码片段混合语料，而且过滤标准要自己定义”，那就必须回到原始 `WARC` 或至少自己控制正文抽取和去重阶段。

---

## 参考资料

| 资料 | 说明 | 作用 |
| --- | --- | --- |
| [Common Crawl Get Started](https://commoncrawl.org/get-started) | 访问方式、快照列表、区域要求 | 校验最新 crawl、下载入口、`us-east-1` 要求 |
| [Common Crawl About](https://commoncrawl.org/about) | 项目概览、月度发布、托管区域 | 确认数据发布节奏和基础背景 |
| [Common Crawl 文件格式说明](https://commoncrawl.org/blog/web-archiving-file-formats-explained) | WARC/WAT/WET 的用途差异 | 理解格式分层 |
| [AWS Requester Pays 文档](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ObjectsinRequesterPaysBuckets.html) | `x-amz-request-payer: requester` 规则 | 解释 `403` 和 CLI 参数来源 |
| [CCNet 论文](https://arxiv.org/abs/1911.00359) | 单语语料抽取、去重、语言识别、困惑度过滤 | 理解经典处理链 |
| [NeMo Curator Common Crawl 文档](https://docs.nvidia.com/nemo/curator/latest/curate-text/load-data/common-crawl.html) | `CommonCrawlDownloadExtractStage` 与正文提取器 | 代码参考 |
| [commoncrawl/cc-pyspark](https://github.com/commoncrawl/cc-pyspark) | Spark 处理 WARC/WAT/WET 示例 | 分布式处理模板 |
| [FineWeb Dataset Card](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | WARC 重抽、fastText、质量规则、MinHash | 现代高质量网页语料范式 |
| [RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) | CCNet 输出、质量信号、去重标记 | 质量分桶与下游训练参考 |
