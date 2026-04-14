## 核心结论

多格式数据标准化，是指在数据进入统一存储层之前，把不同来源、不同文件格式、不同字段命名习惯的数据，收敛成一套可以校验、可以比较、可以直接消费的标准表示。白话解释：上游系统各说各话，标准化阶段负责把它们翻译成同一种“机器能稳定理解的语言”。

它不是“顺手清洗一下”这么简单，而是 ETL 或数据融合流程里最关键的收口点。因为下游的 BI 报表、特征工程、检索系统、模型训练都默认一件事：同名字段有同样含义，同类数据有同样类型，同样的时间和数值可以直接比较。如果这里不统一，下游每一层都要重复处理一次，最终会出现口径不一致。

可以把核心流程压缩成一条式子：

$$
Schema = infer(SampleRows, TypeRules) \rightarrow align(FieldMapping) \rightarrow normalize(Unit, Timezone, Encoding, Precision)
$$

其中：
- `infer` 是推断，意思是根据样本行猜出字段和类型。
- `align` 是对齐，意思是把不同源里的异名字段映射到统一字段。
- `normalize` 是规范化，意思是把单位、时间、编码、精度变成统一标准。

一个最小玩具例子：

| 输入字段 | 样本值 | 推断类型 | 映射后标准字段 | 标准单位 |
|---|---:|---|---|---|
| `velocity_kph` | 110 | FLOAT | `canonical.speed` | km/h |
| `speed_mps` | 30.5 | FLOAT | `canonical.speed` | km/h |
| `speed` | 68 | FLOAT | `canonical.speed` | km/h |

这张表的意思很直接：三个系统都在表达“速度”，但名字不同、单位不同。标准化阶段先识别它们都是数值，再通过字段映射表统一成 `canonical.speed`，最后全部换算到 km/h。这样仪表盘、搜索索引、模型特征只读一个字段即可。

---

## 问题定义与边界

“多格式数据标准化”解决的是载体差异和表达差异，不直接解决业务口径差异。白话解释：它负责让数据“长得一样”，不负责定义“指标怎么算”。

它通常覆盖五类不一致：

| 维度 | 典型问题 | 是否应在标准化阶段处理 |
|---|---|---|
| 字段名 | `user_id`、`uid`、`memberId` 含义相同 | 是 |
| 字段类型 | 同一列在不同源里分别是 `STRING`、`INT`、`FLOAT` | 是 |
| 单位 | km/h、m/s、mph 混用 | 是 |
| 时间格式 | `2026/04/14 08:00`、Unix 时间戳、带本地时区字符串 | 是 |
| 文本编码 | UTF-8、UTF-8 with BOM、GBK 混杂 | 是 |
| 业务指标定义 | GMV 是否含税、订单数是否去重 | 否 |

因此边界要明确：

1. 标准化阶段负责 ingestion 之前或 ingestion 当中的格式统一。
2. 它输出的是 canonical schema。白话解释：canonical schema 就是团队约定的“唯一标准字段模型”。
3. 真正的业务指标计算应该放在其上的视图层、语义层或模型层。

时间字段的目标格式通常可以写成：

$$
parse(任意格式) \rightarrow format(ISO\ 8601 + UTC/Z)
$$

ISO 8601 是时间字符串标准，白话解释：它规定时间怎么写，避免 `04/05/2026` 这种既可能是 4 月 5 日，也可能是 5 月 4 日的歧义。典型输出是：

`2026-04-14T08:00:00Z`

这里的 `Z` 表示 UTC，也就是世界协调时间。

一个新手容易理解的边界例子：

你从对象存储里收到三类文件：
- CSV：字段名第一列带 BOM，出现 `ï»¿Name`
- JSON：时间写成 `"2026/04/14 16:00:00 +0800"`
- XML：金额单位有时是元，有时是分

标准化阶段应该做的是：
- 去 BOM，统一为 UTF-8 无 BOM
- 时间统一成 ISO 8601 的 UTC 表示
- 金额统一成一种单位，比如分
- 输出到同一张 `canonical_facts` 表

但“订单完成金额是否扣除退款”，不在这一步解决。

---

## 核心机制与推导

标准化一般分三层：结构对齐、值规范化、输出校验。

第一层是结构对齐。结构对齐的核心不是“读文件”，而是“把不同结构映射到同一语义”。CSV 是列式表格，JSON 允许嵌套对象，XML 常见属性与节点混用，Parquet 带强类型和列元数据。它们的存储形式不同，但标准化系统关心的是最终字段能否对齐。

基本推导过程如下：

$$
\text{CanonicalField} = align(infer(SourceField, SampleRows), MappingRules)
$$

其中：
- `SampleRows` 是采样数据，白话解释：先看一部分样本，不必一开始就扫描全量。
- `TypeRules` 是类型规则，白话解释：规定 `"123"` 能否转成整数，`"true"` 能否转成布尔。
- `MappingRules` 是字段映射表，白话解释：提前声明哪些原字段应归到哪个标准字段。

第二层是值规范化，包括数字、时间、文本三个最常见对象。

数值标准化公式：

$$
normalize(value, unit\_dict, precision\_rules) \rightarrow convert(canonical\_unit)
$$

例如一个传感器输入：

```json
{"velocity_kph": 110}
```

如果某条下游路由要求显示 mph，可以临时输出 `68.35 mph`；但 canonical 层仍应保留 km/h 或团队约定的统一单位，避免多次来回转换造成误差。也就是说，展示层可以变，底层事实值不要反复变。

时间标准化的关键不是“能 parse 就行”，而是保住时区语义。比如：
- `"2026-04-14 08:00:00"` 如果没有时区，语义不完整
- `"1713081600"` 需要知道它是秒还是毫秒
- `"2026/04/14 08:00 CST"` 还要判断 `CST` 指中国标准时间还是美国中部时间

文本标准化则更底层。它的目标式子可以写成：

$$
encode(raw) \rightarrow UTF\text{-}8\ \text{without BOM}
$$

BOM 是字节顺序标记，白话解释：某些文件会在开头塞几个额外字节帮助识别编码，但很多解析器会把它当成正文字符，导致列名脏掉。

下面是常见字段的标准化路径：

| 字段类型 | 标准化步骤 | 常见操作 |
|---|---|---|
| 数值 | 解析数值、识别单位、统一精度、转换到标准单位 | `float()`、单位词典、`round/Decimal` |
| 时间 | 识别格式、补充时区、转 UTC、格式化 ISO 8601 | `parse`、`astimezone(UTC)` |
| 文本 | 解码、去 BOM、统一 UTF-8、清理控制字符 | `utf-8-sig`、`strip` |
| 布尔 | 把 `0/1`、`yes/no`、`true/false` 对齐 | 映射字典 |
| 枚举 | 把来源侧状态值映射到标准状态集合 | 状态映射表 |

真实工程例子更能说明问题。假设一个跨云数据平台接收三路数据：
- 电商订单 CSV 从 S3 批量落盘
- 设备遥测 JSON 从消息队列写入对象存储
- 供应商主数据 XML 从 FTP 同步

如果不做统一，后面会出现：
- 订单金额有的是元，有的是分
- 遥测时间有的是本地时区，有的是 Unix 毫秒
- 供应商名称编码不一致，搜索时同一名称查不全
- 用户 ID 有的叫 `uid`，有的叫 `customer_id`

标准化系统的职责就是在入仓前把它们全部变成同一组 canonical 字段和类型，让仓库层只承接“已经统一”的数据，而不是把混乱原样放进去等下游兜底。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不依赖外部框架，只展示三件核心事：字段映射、单位转换、时间与编码标准化。

```python
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import codecs

FIELD_MAPPING = {
    "velocity_kph": ("canonical.speed", "kph"),
    "speed_mps": ("canonical.speed", "mps"),
    "speed": ("canonical.speed", "kph"),
    "event_time": ("canonical.event_time", "datetime"),
    "name": ("canonical.name", "text"),
}

TYPE_RULES = {
    "canonical.speed": "float",
    "canonical.event_time": "datetime",
    "canonical.name": "string",
}

def infer_type(value):
    if isinstance(value, (int, float)):
        return "float"
    if isinstance(value, str):
        try:
            float(value)
            return "float"
        except ValueError:
            pass
        return "string"
    return "unknown"

def to_kph(value, unit):
    value = Decimal(str(value))
    if unit == "kph":
        result = value
    elif unit == "mps":
        result = value * Decimal("3.6")
    elif unit == "mph":
        result = value * Decimal("1.60934")
    else:
        raise ValueError(f"unsupported unit: {unit}")
    return result.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def normalize_datetime(value):
    # 这里只演示两种常见输入：ISO 8601 和 Unix 秒
    if isinstance(value, int):
        dt = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, str):
        # 把 +08:00 这类带时区字符串转成 UTC
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError("timezone required for string datetime")
        dt = dt.astimezone(timezone.utc)
    else:
        raise ValueError("unsupported datetime value")
    return dt.isoformat().replace("+00:00", "Z")

def normalize_text(raw_bytes):
    # utf-8-sig 会自动去掉 UTF-8 BOM
    text = raw_bytes.decode("utf-8-sig")
    return text.strip()

def standardize_row(row):
    output = {}
    for source_field, value in row.items():
        if source_field not in FIELD_MAPPING:
            continue
        target_field, semantic = FIELD_MAPPING[source_field]

        if semantic in ("kph", "mps", "mph"):
            output[target_field] = float(to_kph(value, semantic))
        elif semantic == "datetime":
            output[target_field] = normalize_datetime(value)
        elif semantic == "text":
            if isinstance(value, bytes):
                output[target_field] = normalize_text(value)
            else:
                output[target_field] = str(value).strip()
    return output

toy_row_1 = {"velocity_kph": 110, "event_time": "2026-04-14T16:00:00+08:00", "name": codecs.BOM_UTF8 + "Alice".encode("utf-8")}
toy_row_2 = {"speed_mps": 30.5, "event_time": 1776153600, "name": " Bob "}

out1 = standardize_row(toy_row_1)
out2 = standardize_row(toy_row_2)

assert out1["canonical.speed"] == 110.00
assert out1["canonical.event_time"] == "2026-04-14T08:00:00Z"
assert out1["canonical.name"] == "Alice"

assert out2["canonical.speed"] == 109.80
assert out2["canonical.event_time"].endswith("Z")
assert out2["canonical.name"] == "Bob"
```

这个玩具例子对应的输入输出关系如下：

| 原字段 | 推断类型 | 目标字段 | 处理规则 |
|---|---|---|---|
| `velocity_kph` | `float` | `canonical.speed` | 保持 km/h |
| `speed_mps` | `float` | `canonical.speed` | 乘以 3.6 转 km/h |
| `event_time` | `string/int` | `canonical.event_time` | 转 UTC 后输出 ISO 8601 |
| `name` | `bytes/string` | `canonical.name` | 转 UTF-8，无 BOM，去首尾空格 |

如果要写成更接近生产环境的伪代码，流程通常是：

```python
schema = infer(sample_rows, type_rules)

for field in schema:
    target = mapping_table.get(field.name)
    if not target:
        continue

    for row in source_rows:
        raw_value = row[field.name]
        normalized_value = normalize(raw_value, target.unit, target.type, target.timezone)
        emit(target.name, normalized_value)
```

这里有两个工程原则很重要：

1. 映射表要配置化，不要把 `velocity_kph -> canonical.speed` 这种规则硬编码在几十个 if 里。
2. 单位、精度、时区、编码要有元数据，不要只靠字段名猜。

真实工程里，通常还会增加三层能力：
- schema override：覆盖自动推断结果
- validation：校验标准化结果是否符合 canonical schema
- reject channel：把无法标准化的数据发到隔离区，而不是静默吞掉

---

## 工程权衡与常见坑

自动推断很方便，但它本质上是启发式规则，不是事实真相。白话解释：它只能“猜”，不能保证永远猜对。

最常见的坑如下：

| 坑类型 | 触发场景 | 结果 | 应对策略 |
|---|---|---|---|
| 推断退化 | 采样行过少、列前几百行全空 | 类型被推成 `STRING` | 提供 schema override，扩大采样，关键列强制类型 |
| 混合类型 | 同列既有数字又有 `"N/A"` | 下游无法聚合 | 先做脏值清洗，再转标准类型 |
| 单位缺失 | `speed=68` 但没说明是 mph 还是 km/h | 数值不可比较 | 单位元数据与列绑定，无单位则拒收 |
| 时间无时区 | `"2026-04-14 08:00:00"` | UTC 转换可能错 8 小时甚至更多 | 约定源系统必须带时区，或按源级默认时区补齐 |
| UTF-8 BOM | CSV 表头带 BOM | 字段名变成 `ï»¿Name` | 统一按 `utf-8-sig` 解码 |
| 浮点误差 | 金额或单位反复换算 | 出现 `0.3000000004` 之类结果 | 金额用整数最小单位或 `Decimal` |

关于 BOM，很多新手第一次碰到会误以为“文件内容坏了”。其实文件没坏，是读取方式错了。比如 CSV 第一列名本来是 `Name`，结果读出来成了 `ï»¿Name`。原因不是列名真长这样，而是解析器把 BOM 当普通字符了。解决方案是把输入统一解码为 UTF-8 无 BOM。

可以把文本处理抽象成：

$$
text\_canonical = encode(raw) \rightarrow UTF\text{-}8\ without\ BOM
$$

另一个常见坑是“标准化阶段做太多业务逻辑”。比如把订单状态直接映射成 GMV 是否计入，把用户生命周期标签也顺手算出来。这样做短期方便，长期会导致 canonical 层被业务耦合污染。正确做法是：标准化只保证“字段含义明确且格式统一”，复杂业务指标放到其上的模型层。

真实工程例子里，这个边界尤其重要。假设你在湖仓一体平台接订单数据：
- 原始表保留 `refund_amount_cent`
- 标准化层只保证它是整数、单位是分、币种字段齐全
- 业务层再定义 `net_gmv = paid_amount_cent - refund_amount_cent`

如果你在标准化阶段直接生成 `net_gmv`，后来退款口径变化，所有历史逻辑都要回灌重算，代价很高。

---

## 替代方案与适用边界

并不是所有场景都要自己写一套“推断 + 对齐 + 规范化”流程。标准化方案的选择，取决于源数据稳定性、团队控制力、延迟要求和改造成本。

常见替代方案如下：

| 方案 | 核心思路 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| 显式 schema / DDL | 提前定义完整字段和类型 | 稳定、可控、错误早暴露 | 对变更不够灵活 | 数据源稳定、接口受控 |
| ingestion 标准化 | 入仓前统一格式和语义 | 下游最干净、复用最高 | 前期建设成本高 | 多源异构、多人消费 |
| downstream 视图转换 | 先落原始数据，再在视图层转换 | 上线快、改动小 | 每层重复处理，延迟更高 | 无法改管道、过渡期项目 |
| 仓库内自动推断 | 利用仓库自身能力识别 schema | 实施快 | 推断能力有限，单位/编码仍需补 | 格式较单一、简单接入 |

视图层转换的思路可以写成：

$$
downstream\_view = SELECT\ convert\_units(canonical)\ FROM\ normalized\_table
$$

这类方案适合什么情况？比如源系统只产 JSON，字段和类型长期稳定，团队暂时没有精力建设 ingestion 标准化框架，那么可以直接在仓库里定义 external table 或视图，把最小必要的转换放到查询层完成。

但它的边界很明显：
- 如果下游消费方很多，重复转换会放大成本
- 如果字段语义经常变，视图口径容易分叉
- 如果有模型训练、特征服务、跨团队复用，运行时转换通常不够稳定

因此经验上可以这样判断：
1. 单一来源、结构稳定，优先显式 schema。
2. 多来源异构、长期复用，优先 ingestion 标准化。
3. 无法动现有管道、只是短期交付，才考虑 downstream 视图兜底。

---

## 参考资料

| 资料类别 | 链接/说明 |
|---|---|
| 时间标准 | ISO 8601 介绍：https://www.iso8601.com/ |
| 产品文档 | BigQuery schema auto-detection：https://cloud.google.com/bigquery/docs/schema-detect |
| 单位标准化示例 | Oracle IoT 单位归一化场景：https://docs.oracle.com/en-us/iaas/Content/internet-of-things/normalize-units-scenario.htm |
| 方法综述 | 数据标准化概述：https://prospeo.io/s/data-normalization |
| 编码实践 | 重点关注 UTF-8、UTF-8 with BOM、解码器是否支持 `utf-8-sig` |
| 阅读顺序建议 | 先理解 ISO 8601 的 UTC 表达，再看 schema inference 的限制，最后补单位和编码治理策略 |
