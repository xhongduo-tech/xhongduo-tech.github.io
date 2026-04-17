## 核心结论

代码数据清洗可以定义为：把来自 GitHub、Stack Overflow、Kaggle 等来源的源码与配套文本，在进入训练集或对外发布前，经过一组可重复执行的过滤流程，删掉不合规、低质量、重复或含敏感信息的样本。

对初级工程师来说，可以先记住一个最简版本：先按语言和文件格式分类，再检查许可证是否在白名单内，然后做质量打分，接着去重、抽取代码与文本的对应关系，最后扫描并脱敏密钥与个人信息；任一步不合格，就直接丢弃。

六道“筛子”分别解决不同问题：

| 筛子 | 解决的问题 | 不做会怎样 |
| --- | --- | --- |
| 语言过滤 | 保证样本属于目标编程语言或文件类型 | Python 训练集混入 HTML、配置文件、二进制片段 |
| 许可证过滤 | 保证数据可合法使用 | 训练后模型可能继承有争议的数据来源 |
| 质量过滤 | 过滤乱码、演示代码、低信号代码 | 模型学到错误风格或噪声模式 |
| 去重 | 防止同一代码被反复学习 | 模型过拟合热门仓库，评估结果虚高 |
| 对齐提取 | 找到代码与注释、问答、文档的对应关系 | 难以做代码生成、解释、修复类任务 |
| 隐私处理 | 删除密钥、邮箱、手机号等敏感信息 | 训练数据泄露，发布风险高 |

工程上最重要的不是“有过滤”，而是“每一步都能审计”。也就是每个样本为什么被保留、为什么被删除，都要写入 metadata。这样团队才能复现结果，新手也能排查问题。

---

## 问题定义与边界

本文讨论的“代码数据清洗”，输入不是单一代码仓库，而是多源异构数据：源码文件、README、API 文档、问答解释、提交说明、仓库元数据。输出则是一个可用于训练或共享的数据集，其中每条样本都满足语言、许可、质量、重复、对齐、隐私这六类约束。

边界要划清三点。

第一，本文讨论的是“进入模型前的数据处理”，不是模型训练后的安全对齐。训练前数据不干净，后面再补救，成本会更高。

第二，本文讨论的是“样本级与文件级清洗”，不是仓库构建系统修复。比如某个项目测试跑不过，不一定要把项目修好，而是把“测试通过率”“lint 结果”“依赖可解析性”转成质量信号。

第三，本文讨论的是“可投入训练或发布”的数据，不是“互联网上能抓到的数据”。能抓到，不等于能用。

一个玩具例子足够说明六道门槛为什么必要。假设原始数据只有三个文件：

| 文件 | 语言 | 许可证 | 与 A 的相似度 | 是否含敏感信息 | 结果 |
| --- | --- | --- | --- | --- | --- |
| A | Python | MIT | 1.00 | 否 | 保留 |
| B | Python | MIT | 0.99 | 否 | 近重复，删除 |
| C | HTML | GPL | 0.10 | 含邮箱 | 删除 |

这个例子里：
A 通过语言门、许可门、质量门、去重门和隐私门，所以保留。
B 看起来合法，但几乎是 A 的复制品，去重后应删除。
C 即使内容本身可读，也因为许可证不在白名单，且含隐私信息，不应进入数据集。

可以把流程写成简化伪代码：

```text
for sample in raw_dataset:
    if not pass_language(sample): drop
    elif not pass_license(sample): drop
    else:
        score = quality(sample)
        if score < threshold: drop
        elif is_duplicate(sample): drop
        else:
            pair = extract_alignment(sample)
            clean = redact_sensitive_info(pair)
            save(clean, metadata)
```

这个顺序也有原因。语言和许可证是“硬门槛”，失败就直接删；质量、去重、对齐和隐私更像“处理链”，需要继续计算、标注或改写。

---

## 核心机制与推导

去重是代码数据里最容易被低估的一步。原因很简单：公开代码里复制、改名复制、轻微改写复制非常多。若不处理，模型会反复见到同一逻辑，导致训练分布失真。

近重复常用一个近似指标表示：

$$
DedupScore(a, b) \approx Jaccard(tokens(a), tokens(b)) = \frac{|tokens(a)\cap tokens(b)|}{|tokens(a)\cup tokens(b)|}
$$

当

$$
DedupScore(a, b) \ge 0.85
$$

时，通常可以把两个文件视为近重复候选。这里的 token 可以理解为“把代码切成关键字、标识符、符号后的离散单元”。

为什么不是只用一个哈希值？因为精确哈希只能发现完全一样的文件，不能发现“变量名改了、注释删了、顺序微调了”的复制代码。所以工程上常见的组合是：

1. `Exact Hash`
   用 `SHA1`、`MD5` 之类的精确摘要，先删完全相同的文件。
2. `MinHash`
   用少量签名近似表示 token 集合，降低比较成本。
3. `LSH`
   即局部敏感哈希，把相似样本分到同一桶里，只在候选桶内做进一步比较。

新手可以把它理解成两层筛法：先用“身份证号”找一模一样的人，再用“长相相似度”找长得几乎一样的人。

质量过滤也不是单一规则，而是多个弱信号组合成分数。一个常见思路是：

$$
quality = w_1 \cdot avg\_line\_length + w_2 \cdot alphanum\_fraction + w_3 \cdot \log(stars+1) + w_4 \cdot test\_pass\_rate
$$

其中：
`avg_line_length` 表示平均行长，用来排除极端压缩或乱码文件；
`alphanum_fraction` 表示字母数字占比，用来识别二进制污染、模板噪声或异常字符堆积；
`stars` 是仓库受关注度；
`test_pass_rate` 是测试通过率。

这不是说 star 高就一定质量高，而是把它当成一个弱先验。真正稳妥的做法是“硬规则 + 软分数”结合：

| 类型 | 例子 | 作用 |
| --- | --- | --- |
| 硬规则 | 文件可解码、扩展名合法、许可证允许 | 明确不合格直接删除 |
| 软分数 | 行长分布、字符占比、仓库星数、测试通过率 | 对边界样本排序与裁剪 |

真实工程里，数据源还不止源码。Stack Overflow 问答、README、Kaggle Notebook 的说明文本都可以用来构造代码-文本对齐样本。所谓“对齐”，就是判断哪段自然语言与哪段代码在讲同一件事。最简单的方式是利用结构信号：函数上方注释、文档里的代码块与说明段、问答中的 accepted answer 与 snippet。同一份源码如果没有高质量文本对齐，它可能适合预训练，但不一定适合指令微调。

---

## 代码实现

一个可落地的实现结构，通常是“采集索引 + 多阶段管道 + 审计元数据”。可以把它理解成一个小型 ETL。ETL 的白话解释是：先抽取数据，再做转换，最后写入目标存储。

各阶段输入输出可以先画清楚：

| 阶段 | 输入变量 | 输出变量 | 作用 |
| --- | --- | --- | --- |
| 收集索引 | `raw_sources` | `records` | 统一 GitHub/问答/竞赛数据的元数据 |
| 语言分类 | `records` | `lang_records` | 标记语言、扩展名、可解码性 |
| 许可检查 | `lang_records` | `licensed_records` | 保留许可证白名单样本 |
| 质量评分 | `licensed_records` | `scored_records` | 计算 lint、测试、字符统计等分数 |
| 去重 | `scored_records` | `unique_records` | 删除完全重复与近重复 |
| 对齐提取 | `unique_records` | `aligned_records` | 构造 code-text 对 |
| 脱敏 | `aligned_records` | `clean_records` | 替换密钥、邮箱、手机号等 |
| 审计落盘 | `clean_records` | `dataset.jsonl` | 保存样本与过滤原因 |

下面给一个可运行的玩具实现。它不依赖外部库，只演示流程结构和近重复判断：

```python
import math
import re
from hashlib import sha1

ALLOW_LICENSES = {"MIT", "Apache-2.0", "BSD-3-Clause"}
TARGET_LANGS = {"python", "javascript"}

def tokenize(code: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code))

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def quality_score(avg_line_length: float, alphanum_fraction: float, stars: int, test_pass_rate: float) -> float:
    return (
        0.2 * avg_line_length
        + 2.0 * alphanum_fraction
        + 0.5 * math.log(stars + 1)
        + 3.0 * test_pass_rate
    )

def redact_sensitive(text: str) -> str:
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
    text = re.sub(r"sk-[A-Za-z0-9]{10,}", "[REDACTED_KEY]", text)
    return text

def exact_hash(code: str) -> str:
    return sha1(code.encode("utf-8")).hexdigest()

samples = [
    {
        "id": "A",
        "lang": "python",
        "license": "MIT",
        "code": "def add(x, y):\n    return x + y\n",
        "text": "Add two numbers. contact me at dev@example.com",
        "avg_line_length": 14,
        "alphanum_fraction": 0.72,
        "stars": 20,
        "test_pass_rate": 1.0,
    },
    {
        "id": "B",
        "lang": "python",
        "license": "MIT",
        "code": "def add(a, b):\n    return a + b\n",
        "text": "same function",
        "avg_line_length": 14,
        "alphanum_fraction": 0.73,
        "stars": 5,
        "test_pass_rate": 1.0,
    },
    {
        "id": "C",
        "lang": "html",
        "license": "GPL",
        "code": "<html>secret sk-1234567890ABCDE</html>",
        "text": "template",
        "avg_line_length": 38,
        "alphanum_fraction": 0.40,
        "stars": 1,
        "test_pass_rate": 0.0,
    },
]

clean = []
seen_hash = set()

for s in samples:
    if s["lang"] not in TARGET_LANGS:
        continue
    if s["license"] not in ALLOW_LICENSES:
        continue
    score = quality_score(s["avg_line_length"], s["alphanum_fraction"], s["stars"], s["test_pass_rate"])
    if score < 5.0:
        continue

    h = exact_hash(s["code"])
    if h in seen_hash:
        continue

    duplicated = False
    for kept in clean:
        sim = jaccard(tokenize(s["code"]), tokenize(kept["code"]))
        if sim >= 0.85:
            duplicated = True
            break
    if duplicated:
        continue

    s["text"] = redact_sensitive(s["text"])
    seen_hash.add(h)
    clean.append(s)

assert len(clean) == 1
assert clean[0]["id"] == "A"
assert "[REDACTED_EMAIL]" in clean[0]["text"]
```

这段代码对应的工程含义很直接：
先过滤语言和许可证；
再算质量分；
再做精确去重与近重复去重；
最后脱敏。

真实工程例子会复杂得多。比如一个 GitHub 仓库里的 Python 文件，可以这样进入管道：

1. 抓取仓库与文件索引，记录 `repo_name`、`path`、`license`、`stars`。
2. 用扩展名和解析器判断是否为 Python 源码。
3. 调用 lint 工具和测试结果生成质量信号。
4. 先做文件哈希，再进入 MinHash + LSH 近重复检索。
5. 抽取 docstring、README 段落、issue 说明，与函数体建立候选对齐。
6. 扫描邮箱、API key、用户名、访问令牌，必要时替换为占位符。
7. 把每一步的 `drop_reason`、`quality_score`、`dedup_cluster_id`、`pii_hits` 写入审计字段。

关键不是代码写得多复杂，而是每个阶段都输出结构化 metadata，后续才能回答“为什么这个文件没进训练集”。

---

## 工程权衡与常见坑

第一类坑是隐私与密钥泄露。公开仓库并不等于无敏感信息。最常见的是邮箱、手机号、身份证号、云平台密钥、数据库连接串。工程上通常会把“规则扫描”和“模型扫描”结合起来：规则扫描负责高精度命中，例如邮箱正则、API key 模式；模型扫描负责识别上下文里的隐私片段，比如名字、地址、工单号等。像 DataFog 这类工具代表的是“模型辅助识别”路线。

这里有个常被忽略的点：不是所有命中的文件都应该直接物理删除。更稳妥的做法是输出一个“敏感文件保留清单”，记录哪些文件被替换、哪些文件被隔离、哪些文件需要人工复核。这样做的意义是避免误删高价值样本，同时保留审计证据。

第二类坑是过度去重。近重复阈值设得太低，会把大量“同题不同解”“同接口不同实现”的样本错杀。尤其在低资源语言、冷门框架、专业领域代码中，这种误删会让数据集更偏。实践里常见的经验是把精确去重做得严格，把近重复去重做得保守，并允许保留一部分簇内代表样本，而不是整个簇全删。

第三类坑是许可证误判。仓库级许可证和文件级许可证可能不一致，第三方 vendored 代码也可能混在项目里。只看仓库根目录的 `LICENSE`，风险很高。至少要把“仓库许可证”和“文件归属路径”一起纳入判断。

下面是常见坑与规避策略：

| 常见坑 | 具体表现 | 规避策略 |
| --- | --- | --- |
| PII 泄露 | 邮箱、手机号、密钥进入训练集 | 正则扫描 + 模型扫描 + 脱敏替换 + 保留审计清单 |
| 过度去重 | 不同实现被当成重复样本删掉 | 精确去重严格，近重复去重保守，人工抽样复核 |
| 许可误判 | 仓库可用但局部文件不可用 | 仓库级与文件级联合判断，维护白名单 |
| 质量误杀 | 小众项目 star 少但代码质量高 | 不把 star 当硬门槛，只作弱信号 |
| 对齐错配 | README 段落配错代码块 | 优先利用结构关系，低置信度样本不进入监督数据 |

一个实用原则是：对“合规错误”宁可保守，对“质量错误”要留复核通道。原因是许可证和隐私一旦漏检，后果通常比保留一些普通样本更严重。

---

## 替代方案与适用边界

并不是每个团队都需要一上来就做完整六阶段流水线。如果场景是内部实验、小模型训练、私有仓库整理，轻量方案通常更划算。

轻量级替代方案可以是：只做语言过滤、许可证过滤、基础哈希去重，再加少量人工复核。它省掉了 MinHash + LSH、复杂质量评分和大规模隐私模型扫描，适合样本规模较小、数据源相对可控的团队。

全流程与轻量替代可以直接对比：

| 方案 | 包含步骤 | 适用情况 | 不适用情况 |
| --- | --- | --- | --- |
| 全流程清洗 | 语言、许可、质量、去重、对齐、隐私全覆盖 | 公开发布、大模型预训练、跨源数据集 | 极小团队、一次性内部实验 |
| 轻量替代 | 语言、许可、简单 hash 去重、人工复核 | 私有 repo、小模型、原型验证 | 高风险合规场景、要公开共享的数据集 |

如果团队资源更少，还有两种常见折中。

1. 先自动打分，再人工复核高风险样本。
这适合样本量中等、合规要求较高的团队。机器负责缩小范围，人负责最终判断。

2. 借用已有数据集，再补本地规则。
这适合没有能力自建抓取与去重流水线的团队。比如直接基于现成代码数据集继续过滤，重点补充你自己的许可证、领域标签和隐私规则。

但边界也要说明清楚。轻量方案不适合直接用于公开训练集发布，因为它通常缺少系统化的近重复处理、对齐置信控制和隐私审计。对外发布的数据，核心不是“够不够快”，而是“出了问题能不能解释”。

---

## 参考资料

1. BigCode 数据集项目资料：适合看整体数据管道，里面通常能找到过滤、PII、许可、质量控制等脚本思路，支撑本文关于六道筛子的整体流程。
2. The Stack 或其去重版本的数据卡片：适合看语言分布、许可证处理、近重复定义、数据声明，支撑本文关于 `DedupScore`、许可白名单和近重复清洗的部分。
3. Code2Doc 一类的代码文档对齐论文：适合理解如何从代码与自然语言中抽取监督样本，支撑本文关于 code-text alignment 的部分。
4. DataFog 一类敏感信息检测工具资料：适合了解正则规则与模型识别如何结合，支撑本文关于 PII、密钥扫描和脱敏审计的部分。
5. 公开代码数据集的 dataset card 与 data statement：适合学习“为什么删、删了什么、剩下什么”的披露方式，支撑本文关于可审计 metadata 的工程要求。
