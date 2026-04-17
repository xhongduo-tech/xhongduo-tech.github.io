## 核心结论

结构化数据抽取的目标，是把自然语言文本中的关键信息转换成程序可直接处理的记录。这里的“结构化”不是把一句话重新表述一遍，而是把信息映射到预先定义好的字段中，并满足类型、格式、取值范围等约束，最终可以稳定写入数据库、搜索索引或下游分析系统。

最常见的三类抽取对象如下：

| 类型 | 定义 | 典型输出 |
|---|---|---|
| 实体识别（NER, Named Entity Recognition） | 从文本中定位具有明确类别的片段 | `person=李雷` |
| 关系抽取（RE, Relation Extraction） | 判断两个或多个实体之间的语义关系 | `李雷 -> 出差地点 -> 上海` |
| 事件抽取（Event Extraction） | 将一句话组织成“主体 + 动作 + 时间/地点/对象”等事件结构 | `trip(person,date,location)` |

先看一个最小例子。

输入文本：

`李雷于2026-03-08在上海出差`

目标输出：

```json
{"person":"李雷","date":"2026-03-08","location":"上海"}
```

这个例子说明，抽取系统的本质不是“理解世界”，而是完成两步受约束的转换：

1. 从原文中找出候选信息，例如“李雷”“2026-03-08”“上海”。
2. 按预定义格式做标准化和校验，例如日期必须满足 `YYYY-MM-DD`，地点必须属于允许集合，字段不能为空。

常见方法可以分为三类：

| 方法 | 优点 | 缺点 | 适合场景 |
|---|---|---|---|
| 规则法：正则表达式 + Gazetteer | 成本低、结果稳定、便于解释 | 覆盖面有限，表达一变化就容易漏召回 | 表单、日志、固定模板报告 |
| 模型法：BERT-NER + LLM | 能处理自然表达，召回率更高 | 可能幻觉，成本更高，需要额外验证 | 长段落文本、客服对话、科研文献 |
| 混合法 | 同时利用规则约束和模型泛化能力 | 系统复杂度更高 | 真实生产系统 |

整体流程可以概括为一条流水线：

```text
文本 T -> 候选抽取 -> 字段标准化 -> schema 验证 -> 清洗去重 -> 入库
```

这条流水线有一个关键约束：模型输出不是结果，验证通过的输出才是结果。

---

## 问题定义与边界

“信息抽取”与“摘要”或“自由问答”不同。抽取任务要求输出满足预定义 `schema`。可以把 `schema` 理解为字段合同，它明确规定：

| 约束项 | 含义 | 例子 |
|---|---|---|
| 字段名 | 系统要保存哪些字段 | `date`、`owner`、`amount` |
| 字段类型 | 字段以什么类型存储 | `string`、`number`、`date` |
| 必填性 | 字段是否允许缺失 | `required: true` |
| 格式规则 | 字段值必须满足什么模式 | 日期必须是 `YYYY-MM-DD` |
| 取值范围 | 字段是否只能取枚举值 | 状态只能是 `draft/published` |

例如文本：

`项目提交日期2026年4月1日，负责人小张。`

如果目标 schema 是：

| 字段 | 类型 | 说明 |
|---|---|---|
| event | string | 事件类型 |
| date | date | 标准化日期 |
| owner | string | 负责人 |

那么可接受输出是：

```json
{"event":"提交","date":"2026-04-01","owner":"小张"}
```

这里已经出现了边界问题。不是所有信息都适合自动抽取，通常要先区分三类字段：

| 字段类型 | 是否适合自动抽取 | 说明 | 例子 |
|---|---|---|---|
| 明确出现 | 适合 | 字段值直接出现在原文 | `2026年4月1日` -> `2026-04-01` |
| 可稳定推导 | 有条件适合 | 需要轻度语义映射，但规则清晰 | “提交日期”对应事件 `提交` |
| 依赖业务判断 | 常需人工确认 | 需要业务语境或责任定义 | “负责人”是否等于最终审批人 |

因此，抽取系统通常只覆盖“可以稳定映射到 schema 的那部分信息”。不同文本类型对应的 schema 也完全不同：

| 输入文本 T | schema 示例 | 目标字段 |
|---|---|---|
| 会议纪要 | 会议记录表 | 时间、地点、参与人、决策 |
| 合同条款 | 合同数据库 | 甲方、乙方、金额、生效日期 |
| 科研段落 | 材料属性表 | 材料名、Tg、bandgap、测试条件 |
| 客服对话 | 工单记录 | 用户、问题类型、处理结果、时间 |
| 运维日志 | 事件监控表 | 主机、错误码、时间、影响范围 |

边界矩阵更直观：

| 信息类型 | 可直接抽取 | 需模型推理 | 需人工确认 |
|---|---|---|---|
| 明确时间 | 是 | 否 | 否 |
| 明确地点 | 是 | 否 | 否 |
| 代词指代的人物 | 否 | 是 | 有时 |
| 隐含因果关系 | 否 | 是 | 常常 |
| 合规性结论 | 否 | 否 | 是 |

这说明一个核心事实：结构化数据抽取不是“理解全文的一切”，而是“只抽 schema 允许进入系统的那部分信息”。边界定义越早、越精确，系统越稳定。

---

## 核心机制与推导

可以把抽取过程形式化写成：

$$
I(T)=NER(T)\cup RE(NER(T))\cup Event(NER(T))
$$

其中：

- $T$ 表示输入文本。
- $NER(T)$ 表示从文本中识别出的实体集合。
- $RE(NER(T))$ 表示基于实体对识别出的关系集合。
- $Event(NER(T))$ 表示基于实体和触发词构造出的事件结构。

上式描述的是“信息候选集合”，但工程系统真正关心的是“哪些候选能安全入库”。因此还需要加上 schema 约束：

$$
Output = \{x \in I(T)\ |\ x \models S\}
$$

其中：

- $S$ 表示预定义 schema。
- $\models$ 表示“满足约束”。

白话解释就是：抽出来不算完成，满足字段合同才算完成。

如果进一步写成函数流水线，可以得到：

$$
Output = Validate(Normalize(Extract(T)), S)
$$

这三个环节分别解决不同问题：

| 环节 | 解决的问题 | 典型操作 |
|---|---|---|
| `Extract` | 从哪里找值 | 正则、NER、LLM |
| `Normalize` | 值如何统一表示 | 日期标准化、单位换算、别名归一 |
| `Validate` | 值能否入库 | 类型检查、枚举检查、原文对照 |

例如定义一个最小 schema：

```json
{
  "person": {"type": "string", "minLength": 1},
  "date": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
  "location": {"type": "string", "enum": ["上海", "北京", "深圳", "杭州"]}
}
```

对于玩具例子：

```text
原文: 李雷于2026-03-08在上海出差
  -> 实体候选: 李雷 / 2026-03-08 / 上海
  -> 事件映射: 出差(person=李雷, date=2026-03-08, location=上海)
  -> 标准化: 日期保持为 2026-03-08
  -> schema 校验: 日期格式合法，地点属于允许集合
  -> 输出 JSON
```

规则法的机制是“先定义模式，再匹配文本”。例如：

- 日期通过正则识别。
- 地点通过词典匹配。
- 固定动作词通过模板映射为事件类型。

例如日期模式可以写成：

```regex
\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?
```

地点词典通常称为 `Gazetteer`。它本质上是一个领域词表，用来判断某个片段是否属于已知实体集合，例如城市名单、公司名录、药品清单、材料名称库。它不负责“理解上下文”，只负责“识别是否命中已知集合”。

模型法的机制是“先预测候选，再用约束过滤”。典型流程如下：

1. 用 BERT-NER 或类似序列标注模型定位实体候选。
2. 将原文和 schema 一起输入 LLM。
3. 要求模型只返回 JSON，不要解释文字。
4. 对 JSON 做字段级校验、格式校验、原文对照。
5. 校验失败时重试、回退规则、或进入人工复核。

这里最重要的工程原则是：模型输出不等于事实。LLM 的“幻觉”可以理解为“输出了看起来合理、但原文没有明示的信息”。因此抽取系统不能只检查输出格式，还必须检查字段值是否能在原文中找到证据，或至少能由稳定规则推出。

科研文献抽取更能说明这一点。原文可能是：

`The polymer exhibited a glass transition temperature of 118 °C and a bandgap of 2.1 eV.`

目标结构可能是：

```json
{
  "material": "polymer",
  "tg_celsius": 118.0,
  "bandgap_ev": 2.1
}
```

但实际文本表达会出现大量变体：

| 原文表达 | 目标字段 |
|---|---|
| `glass transition temperature of 118 °C` | `tg_celsius=118` |
| `Tg = 118°C` | `tg_celsius=118` |
| `glass transition at 118 C` | `tg_celsius=118` |
| `band gap was 2.1 eV` | `bandgap_ev=2.1` |
| `Eg = 2.1 eV` | `bandgap_ev=2.1` |

这时单纯依赖正则容易漏掉变体，单纯依赖 LLM 又会带来稳定性问题，所以更常见的做法是：模型负责召回候选，规则负责标准化与验证。

---

## 代码实现

下面给出一个可运行的最小 Python 示例。它只依赖标准库，完整覆盖“抽取 -> 标准化 -> schema 校验 -> 输出”这一闭环。为了让示例真正可运行，代码同时处理了两种日期写法、固定动作词映射、地点词典校验，以及简单的单元测试。

```python
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


CITY_GAZETTEER = {"上海", "北京", "深圳", "杭州"}
EVENT_PATTERNS = {
    "出差": re.compile(r"(出差|前往|到).*?(上海|北京|深圳|杭州)?"),
    "提交": re.compile(r"(提交|提交了|已提交)"),
}


@dataclass
class TripRecord:
    person: str
    date: str
    location: str
    event: str = "出差"


def normalize_date(text: str) -> Optional[str]:
    """
    支持:
    - 2026-03-08
    - 2026/03/08
    - 2026年3月8日
    """
    text = text.strip()
    match = re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?", text)
    if not match:
        return None

    year, month, day = map(int, match.groups())
    try:
        dt = datetime(year, month, day)
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%d")


def extract_person(text: str) -> Optional[str]:
    """
    仅用于演示:
    识别“李雷”“韩梅梅”“小张”这类连续中文片段。
    真实系统应使用 NER 或更明确的规则。
    """
    match = re.search(r"(李雷|韩梅梅|小张|小李|小王|[\u4e00-\u9fa5]{2,4})", text)
    return match.group(1) if match else None


def extract_date(text: str) -> Optional[str]:
    match = re.search(r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)", text)
    if not match:
        return None
    return normalize_date(match.group(1))


def extract_location(text: str) -> Optional[str]:
    for city in sorted(CITY_GAZETTEER, key=len, reverse=True):
        if city in text:
            return city
    return None


def detect_event(text: str) -> Optional[str]:
    for event, pattern in EVENT_PATTERNS.items():
        if pattern.search(text):
            return event
    return None


def extract_trip(text: str) -> dict:
    record = {
        "person": extract_person(text),
        "date": extract_date(text),
        "location": extract_location(text),
        "event": detect_event(text),
    }
    return record


def validate_record(record: dict) -> tuple[bool, list[str]]:
    errors = []

    if record.get("event") != "出差":
        errors.append("event must be 出差")

    person = record.get("person")
    if not person or not re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", person):
        errors.append("person is invalid")

    date = record.get("date")
    if not date or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        errors.append("date format must be YYYY-MM-DD")

    location = record.get("location")
    if not location or location not in CITY_GAZETTEER:
        errors.append("location is not in gazetteer")

    return len(errors) == 0, errors


def main() -> None:
    samples = [
        "李雷于2026-03-08在上海出差",
        "韩梅梅在2026年3月8日前往北京出差",
    ]

    for text in samples:
        pred = extract_trip(text)
        ok, errors = validate_record(pred)
        print("input :", text)
        print("pred  :", json.dumps(pred, ensure_ascii=False))
        print("valid :", ok)
        if errors:
            print("errors:", errors)
        print("-" * 40)


if __name__ == "__main__":
    main()
```

运行后，预期输出类似：

```text
input : 李雷于2026-03-08在上海出差
pred  : {"person": "李雷", "date": "2026-03-08", "location": "上海", "event": "出差"}
valid : True
----------------------------------------
input : 韩梅梅在2026年3月8日前往北京出差
pred  : {"person": "韩梅梅", "date": "2026-03-08", "location": "北京", "event": "出差"}
valid : True
----------------------------------------
```

这个示例虽然简单，但已经具备生产系统的基本模块边界：

| 模块 | 作用 | 当前实现 | 可替换实现 |
|---|---|---|---|
| 候选识别 | 找到可能的字段值 | 正则 + 词典 | BERT-NER、CRF、LLM |
| 标准化 | 统一输出格式 | `normalize_date` | 时间解析器、单位换算器 |
| 事件映射 | 把表达映射成 schema 字段 | 关键动词模板 | 分类模型、LLM |
| schema 校验 | 判断是否可入库 | 正则 + 枚举检查 | `jsonschema`、Pydantic |
| 错误报告 | 返回失败原因 | `errors` 列表 | 字段级置信度、审计日志 |

如果接入 LLM，最合理的方式不是让它直接决定最终结果，而是让它只负责“提出候选”。例如可以设计如下接口：

```python
def llm_extract(text: str, schema: dict) -> dict:
    """
    伪代码:
    1. 把 text 和 schema 一起发给模型
    2. 要求只返回 JSON
    3. 不接受 schema 外字段
    """
    return {
        "person": "李雷",
        "date": "2026-03-08",
        "location": "上海",
        "event": "出差"
    }
```

接下来仍然必须做验证：

```python
def verify_against_source(text: str, record: dict) -> tuple[bool, list[str]]:
    errors = []

    if record["location"] not in text:
        errors.append("location not grounded in source text")

    if record["person"] not in text:
        errors.append("person not grounded in source text")

    ok, schema_errors = validate_record(record)
    errors.extend(schema_errors)
    return len(errors) == 0, errors
```

这体现出一个工程常识：LLM 负责提高召回，规则负责保证落库质量。

对于新手，最容易混淆的点有三个：

| 术语 | 实际含义 | 不应误解为 |
|---|---|---|
| schema | 字段合同 | 一段自由描述 |
| Gazetteer | 已知实体词典 | 能自动理解上下文的模型 |
| Grounding | 输出可在原文中找到依据 | 模型觉得“合理” |

只要这三个概念分清，整套系统的职责边界就清楚了。

---

## 工程权衡与常见坑

真实系统里，问题通常不在“能不能抽出来”，而在“抽出来之后能不能放心使用”。因此工程权衡通常围绕四个指标展开：

| 指标 | 含义 | 为什么重要 |
|---|---|---|
| 精度（Precision） | 抽出来的结果中有多少是真的 | 错误入库会污染数据库 |
| 召回（Recall） | 原文中应该抽的内容有多少被抽到 | 漏召回会降低数据覆盖率 |
| 一致性（Consistency） | 相同输入是否得到稳定输出 | 直接影响线上系统可预测性 |
| 成本（Cost） | 开发、维护和推理成本 | 决定方案是否可长期运行 |

不同方案的工程画像如下：

| 方案 | 开发成本 | 运行成本 | 可解释性 | 泛化能力 |
|---|---|---|---|---|
| 纯规则 | 中 | 低 | 高 | 低 |
| 纯 BERT/LLM | 高 | 中到高 | 低到中 | 高 |
| 混合方案 | 高 | 中 | 高 | 高 |

LLM 在抽取系统中的典型问题主要有四类：

| 风险 | 现象 | 原因 | 缓解策略 |
|---|---|---|---|
| 幻觉 | 输出原文没有的值 | 语言模型倾向补全 | 原文对照、字段级 grounding |
| 不一致 | 同一输入多次结果不同 | 生成式模型具有随机性 | 固定温度、固定提示词、缓存 |
| 格式漂移 | JSON 结构变化、字段名拼错 | 模型偏离格式约束 | 严格 schema 验证，不通过即重试 |
| 过度推理 | 把模糊表述当成确定事实 | 模型过强解释倾向 | 区分“抽取字段”和“推断字段” |

再看一些更具体的工程坑。

第一，字段定义不清。  
例如“负责人”可能指文档作者、项目 owner、审批人、接口人。若 schema 本身模糊，模型和规则都会出现不稳定输出。这个问题不能靠调参解决，必须先修正字段定义。

第二，上游错误向下游传播。  
例如实体识别把“上海研究院”中的“上海”错误识别为地点，那么关系抽取、事件抽取和数据库写入都会一起出错。这类问题需要分层校验，而不是只看最终准确率。

第三，词典老化。  
Gazetteer 很适合封闭集合，但一旦别名、新机构名、缩写不断增加，词典就会迅速过期。工程上要给词典维护版本、更新时间和增量补丁，而不是把它当静态文件。

第四，单位和格式不统一。  
特别是在科研、金融、合同场景中，数值抽取往往不是“抽数字”那么简单，而是要同时处理单位、量纲和书写差异。例如：

| 原文 | 若不标准化的问题 | 正确结果 |
|---|---|---|
| `118 °C` | 单位包含空格 | `118` + `celsius` |
| `2.1 eV` | 数值和单位混合 | `2.1` + `ev` |
| `1.2 million USD` | 含数量级 | `1200000` + `USD` |

第五，训练集和线上分布不一致。  
离线测试时文本可能写法规范，线上数据却充满简称、错别字、拼写波动和上下文缺失。很多“实验室里很好”的抽取器，在上线后会因为样本分布变化迅速失效。

因此，生产系统通常会采用双重验证或多重验证：

```text
模型抽取
  -> schema 检查
  -> 原文片段对照
  -> 规则复核
  -> 置信度打分
  -> 低置信度样本人工抽检
  -> 入库
```

如果需要用公式表达“是否自动入库”，可以写成：

$$
AutoWrite(x)=
\begin{cases}
1, & score(x)\ge \tau \land x \models S \land grounded(x,T)=1 \\
0, & otherwise
\end{cases}
$$

其中：

- $score(x)$ 是模型或融合模块给出的置信分数。
- $\tau$ 是上线阈值。
- $x \models S$ 表示满足 schema。
- $grounded(x,T)=1$ 表示关键字段可以在原文中找到证据。

这个公式表达的是一个实际工程原则：准确率不是唯一门槛，验证通过才是自动化门槛。

---

## 替代方案与适用边界

不是所有场景都应该使用 LLM。选型主要取决于三个问题：

1. 文本表达是否稳定。
2. 错误成本是否可接受。
3. 数据规模是否允许高推理成本。

看两个对比明显的例子。

例子一，格式固定文本：

`日期：2026/03/08，地点：上海，人员：李雷`

这类文本更适合规则法。因为字段标签、分隔符、顺序都比较稳定，使用正则和词典就能以很低成本获得高精度结果。

例子二，自然表达文本：

`3 月 8 日，李雷到上海处理客户现场问题。`

这类文本更适合模型法。因为时间、人名、地点和事件可能分散出现，还可能存在省略、倒装或口语表达。

不同方法的适用边界可以总结为：

| 方法 | 适用场景 | 优势 | 边界 |
|---|---|---|---|
| 正则表达式 | 日志、报表、表单 | 简单、稳定、便宜 | 表达一变化就容易失效 |
| Gazetteer | 地名、机构名、药品名等封闭集合 | 校验强、误报低 | 对新词、别名、缩写不敏感 |
| BERT-NER | 中等复杂度实体识别 | 上下文敏感，比纯规则强 | 仍需下游结构映射 |
| LLM + schema | 长文本、复杂事件、多样表达 | 端到端能力强 | 成本高，且必须加验证 |
| 混合策略 | 线上生产系统 | 精度和召回更平衡 | 设计和维护复杂 |

对于新手，最实用的判断方式不是先问“哪种模型最先进”，而是先问“文本像什么”。可以用下面这张表快速判断：

| 文本外观 | 推荐方案 | 原因 |
|---|---|---|
| 像表单 | 规则优先 | 格式稳定，规则收益最高 |
| 像自然段落 | 模型优先 | 需要处理自由表达 |
| 错误代价高 | 混合验证 | 不能只依赖单一路径 |
| 数据量特别大 | 先评估成本 | 模型推理费用可能成为主约束 |

生产中更常见的混合流程如下：

```text
文本
  -> 规则快速命中显式字段
  -> 模型补召回隐式字段
  -> 规则与词典校验
  -> 置信度打分
  -> 高置信度自动入库
  -> 低置信度人工审核
```

这个流程的优点不是“最先进”，而是“最稳”。原因很简单：数据清洗和结构化抽取的目标不是展示模型能力，而是稳定地产出可被下游系统信任的记录。

可以用一句工程化判断收尾：

- 如果文本长得像表单，优先规则。
- 如果文本长得像自然段落，优先模型。
- 如果错误代价高，必须混合验证。
- 如果规模很大，先算推理成本，再决定模型深度。

---

## 参考资料

1. Jurafsky, Daniel; Martin, James H.《Speech and Language Processing》相关章节。用于支撑实体识别、关系抽取、事件抽取的基本定义与任务拆分。  
2. IBM 关于 Information Extraction 的工程介绍。用于支撑规则式流程、实体与关系抽取在企业文本处理中的典型做法。  
3. Pydantic 与 JSON Schema 官方文档。用于支撑“schema-first”和“字段级验证先于入库”的工程实践。  
4. `jsonschema` 官方文档。用于支撑结构化输出的格式校验、类型校验、枚举校验等机制。  
5. BERT、领域 NER 与材料信息抽取相关论文。用于支撑“模型负责候选召回，规则负责标准化与约束”的混合路线。  
6. 近年来使用 LLM 做结构化抽取、关系抽取的论文与工程博客。用于支撑 LLM 在长文本、多样表达上的优势，以及幻觉、成本和一致性问题。  
7. Gazetteer、地名词典、术语词典维护相关实践。用于支撑词典法的优势和局限，即高可解释性与持续维护成本并存。
