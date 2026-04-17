## 核心结论

分布式数据清洗不是“把一段清洗代码丢到多台机器上并行执行”这么简单。真正决定性能和稳定性的核心，是先选对分区键，再把能在分区内完成的操作尽量留在本地，只把必须全局协调的操作交给 shuffle。这里的 shuffle 可以白话理解为“跨机器重新分发数据，让同一类 key 落到一起”。

对初学者最重要的判断标准只有两个：

1. 这个清洗操作是否必须看到“全局数据”才能正确执行。
2. 这个操作是否会把大量数据在网络中重排。

过滤、字段纠错、格式标准化，通常都可以在分区内完成；`join`、`groupBy`、全局去重这类操作，通常需要全局协调。分布式清洗的效率，往往不取决于 CPU 算力本身，而取决于你触发了多少次 shuffle、每次 shuffle 搬了多少数据、状态保存了多久。

一个最常用的资源估算起点是：

$$
N_{partition}=\left\lceil \frac{D}{128\text{MB}} \right\rceil
$$

其中 $D$ 是待处理数据的解压后体量，128MB 对应 Spark 读文件时常见的默认分区上限。工程上还常用一个保守经验：单核可用内存至少按“分区大小的 4 到 6 倍”预留，避免排序、哈希聚合、序列化和 spill 把任务拖慢。spill 可以白话理解为“内存放不下，临时写磁盘”。

如果只记一个实践原则，可以记这一句：

$$
\text{总成本} \approx \text{本地清洗成本} + \text{shuffle 成本} + \text{状态维护成本}
$$

本地清洗通常便宜，shuffle 和长期状态通常昂贵。分布式清洗设计的大部分优化，本质上都在压低后两项。

---

## 问题定义与边界

本文说的“分布式数据清洗”，指的是在 Spark、Dask、Ray 这类分布式执行引擎上，把大规模原始数据拆成多个分区，并行执行以下操作：

| 清洗算子 | 白话解释 | 是否通常可局部完成 | 是否常触发全局协调 |
| --- | --- | --- | --- |
| `filter` | 删除明显无效记录 | 是 | 否 |
| 格式化 | 统一时间、金额、编码格式 | 是 | 否 |
| 字段纠错 | 修复脏值、空值、枚举映射 | 是 | 否 |
| `mapPartitions` 自定义清洗 | 对每个分区批量处理 | 是 | 否 |
| 局部去重 | 先在单分区内删重复 | 是 | 否 |
| `groupBy` / 聚合 | 按 key 汇总统计 | 否 | 是 |
| `join` | 把两张表按 key 对齐 | 否 | 是 |
| 全局 `dropDuplicates` | 所有分区联合判重 | 否 | 是 |
| 流式状态去重 | 保留历史 key 防止重复 | 否 | 是 |

这里先澄清几个初学者最容易混淆的术语：

| 术语 | 简明定义 | 初学者常见误解 |
| --- | --- | --- |
| 分区 | 一份可独立调度和处理的数据块 | 以为等同于磁盘文件 |
| 任务 | 执行引擎分配给一个核的一段工作 | 以为“一个作业只有一个任务” |
| shuffle | 按 key 重新分发数据 | 以为只是“排序” |
| watermark | 允许迟到数据的时间边界 | 以为是“时间字段格式化” |
| 状态 | 为了后续判重或聚合而保留的历史信息 | 以为只在数据库里才有状态 |

边界也要说清楚。不是所有清洗都值得分布式化：

- 数据量只有几 GB，单机 pandas 或 DuckDB 往往更简单。
- 逻辑极度依赖全局排序、跨表多次回填、复杂事务写入时，分布式收益会下降。
- 如果输出要求“同一 key 严格唯一”，那你必须定义唯一键、乱序容忍范围、失败恢复策略，否则“清洗正确”没有可验证含义。

判断要不要上分布式，可以先看这张表：

| 数据规模与约束 | 更常见的合适方案 |
| --- | --- |
| 几 GB，规则简单，单次批处理 | pandas / DuckDB |
| 几十 GB 到数百 GB，已有大量 Python 清洗代码 | Dask |
| TB 级，涉及多次 `join`、聚合、流批一体 | Spark |
| 多阶段自定义流水线，算子链复杂，含模型推断或样本处理 | Ray |

一个新手最容易理解的流程是：先选 key 决定分区，再对每个 chunk 做局部去重和格式清洗，只有在联接、全局去重或按 key 汇总时，才把相同 key 的记录 shuffle 到一起。

再进一步说，分布式清洗的“正确性”通常依赖三个边界条件：

1. 你用什么字段判定两条记录“其实是同一条”。
2. 你允许多晚到达的数据仍然被视为有效。
3. 任务失败后，系统按什么状态恢复，避免重复写和漏写。

这三个问题不定义清楚，代码即使跑通，结果也未必可信。

---

## 核心机制与推导

### 1. 分区数先由数据量决定

如果原始 1TB 压缩 CSV 解压后约为 10TB，按 128MB 分区估算：

$$
N_{partition}=\left\lceil \frac{10\times 1024\times 1024\text{MB}}{128\text{MB}} \right\rceil \approx 81{,}920
$$

这意味着任务数量的量级就是 8 万。这里的任务可以白话理解为“调度器分发给某个核执行的一小块工作”。

如果每个核同一时刻只稳定处理 1 个分区，且按每分区 512MB 到 768MB 的可用内存预算，则全内存并行的理论资源会非常大。实际集群不会一次性开出 8 万核，而是靠多波次调度完成；但这个公式能帮助你及早意识到，10TB 清洗不是“多开几台机器”就够了。

这个估算至少有三个作用：

- 帮你预估任务数会不会大到让调度器有压力。
- 帮你反推单分区大小是否过大，容易 spill。
- 帮你判断后续 `repartition` 后分区数该落在什么量级，而不是随手写一个常数。

可以把资源估算写成更完整的工程视角：

$$
\text{总波次数} \approx \left\lceil \frac{N_{partition}}{N_{core}} \right\rceil
$$

其中 $N_{core}$ 是同一时刻实际可并行执行的核数。  
例如有 1,024 个可用核，8.2 万个分区至少需要约 80 波次以上才能跑完。此时即使单个任务只跑 2 分钟，总体作业时间也不会低。

对新手来说，最重要的认识不是“公式多精确”，而是：

- 分区数过大，调度变重。
- 分区数过小，单任务变胖。
- 分区大小不是越大越好，也不是越小越好。

### 2. 为什么 reduceByKey 思路比 groupByKey 更省

对按 key 去重、计数、合并这样的操作，应该尽量先做分区内预聚合，再做跨节点汇总。原因是：

- `groupByKey` 更像“先把所有同 key 数据都搬到一起，再开始算”
- `reduceByKey` 更像“先在本地把同 key 压缩一遍，再把压缩后的结果发出去”

因此网络传输量更小，shuffle 压力更低。放到 DataFrame 语义里，对应的思想是优先使用可预聚合的聚合算子，而不是先把明细全量打散重组。

看一个极简例子。假设某个 key 在 100 个分区里各出现 1 万次：

| 做法 | 本地阶段 | 网络传输 |
| --- | --- | --- |
| `groupByKey` 思路 | 基本不压缩 | 约传 100 万条原始记录 |
| `reduceByKey` 思路 | 每分区先聚成 1 条或少量条目 | 约传 100 条局部结果 |

这就是为什么“先局部、再全局”是分布式清洗的基本方法。  
即使最终仍然需要全局协调，先在本地把数据缩小，通常也能显著降低网络和内存压力。

对 DataFrame 使用者，可以把这个原则翻译成更直接的话：

- 优先写成“过滤 + 标准化 + 可预聚合聚合”
- 避免“先收齐明细再手工处理”
- 如果只是为了算最小值、最大值、计数、去重数量，不要把整组明细全搬过去

### 3. `repartition` 和 `coalesce` 的差异

- `repartition` 会主动重分布数据，通常会触发 shuffle，适合“按新 key 重新洗牌”。
- `coalesce` 通常用于减少分区数，尽量避免全量 shuffle，适合“下游并行度太高，准备收缩任务数”。

所以“先按业务 key 做一次 `repartition`，后续局部清洗尽量不再打散”，通常比中间反复改分区策略稳定得多。

一个常见误区是：看到任务慢，就不断插入 `repartition(200)`、`repartition(500)`、`repartition("user_id")`。  
这往往不是优化，而是在反复制造 shuffle。

可以直接用下面这张表记忆：

| 操作 | 常见目的 | 代价特点 | 适合的时机 |
| --- | --- | --- | --- |
| `repartition(n)` | 重新均匀打散并调整并行度 | 通常有 shuffle | 需要重新平衡负载时 |
| `repartition("key")` | 让同 key 数据落到同类分区 | 通常有 shuffle | 后续要按该 key 聚合、去重、join 时 |
| `coalesce(n)` | 缩减分区数 | 常可少 shuffle | 输出前收缩任务数时 |

如果后续多个阶段都围绕同一个 `user_id` 工作，那么最稳定的做法通常是：

1. 先按 `user_id` 重分区一次。
2. 在这个分布上完成尽可能多的局部清洗。
3. 只在必须换 key 时再触发下一次 shuffle。

### 4. 玩具例子

假设有 12 条订单，唯一键是 `order_id`，其中 3 条重复，2 条时间格式错误。

如果直接全局去重，12 条数据都可能参与一次全局重排；但如果先按 `order_id % 4` 分成 4 个分区，重复订单天然更可能被送到同一分区，你可以先在每个分区里做：

- 时间字段标准化
- 非法金额过滤
- 局部去重

只有跨分区仍可能重复的 key，才进入最终全局判重阶段。数据量小时差异不明显，数据到十亿行时，代价差异会非常大。

把这个过程展开成更容易理解的流水线：

| 阶段 | 做什么 | 是否需要看全局 |
| --- | --- | --- |
| 阶段 1 | 按 `order_id` 相关规则分区 | 否 |
| 阶段 2 | 修时间格式、统一邮箱大小写、过滤坏金额 | 否 |
| 阶段 3 | 在分区内删除明显重复记录 | 否 |
| 阶段 4 | 对残余重复 key 做全局合并 | 是 |
| 阶段 5 | 输出结果并记录进度 | 取决于输出系统 |

这里最重要的不是“局部去重一定完全正确”，而是“先把便宜且高收益的工作做掉，再把小很多的数据交给全局阶段”。

---

## 代码实现

下面先用一个可运行的 Python 玩具实现说明“按 key 分区 -> 分区内清洗 -> 全局合并”的思路。代码补上了时间解析、结果打印和断言，直接在 Python 3.10+ 可运行：

```python
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from math import ceil
from typing import Iterable


rows = [
    {"user_id": 1, "ts": "2026-03-01 10:00:00", "email": "A@EXAMPLE.COM "},
    {"user_id": 2, "ts": "2026/03/01 10:01:00", "email": "b@example.com"},
    {"user_id": 1, "ts": "2026-03-01 10:00:00", "email": " a@example.com"},
    {"user_id": 3, "ts": "", "email": "bad"},
    {"user_id": 2, "ts": "2026-03-01 10:01:00", "email": "B@example.com"},
]


def normalize_ts(ts: str) -> str | None:
    if not ts or not ts.strip():
        return None

    candidates = ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]
    for fmt in candidates:
        try:
            parsed = datetime.strptime(ts.strip(), fmt)
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return None


def normalize_email(email: str) -> str:
    return email.strip().lower()


def is_valid_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1]


def partition_key(row: dict, n: int = 2) -> int:
    return row["user_id"] % n


def clean_partition(partition_rows: Iterable[dict]) -> list[dict]:
    seen = set()
    output = []

    for row in partition_rows:
        normalized_ts = normalize_ts(row["ts"])
        normalized_email = normalize_email(row["email"])

        if normalized_ts is None:
            continue
        if not is_valid_email(normalized_email):
            continue

        dedup_key = (row["user_id"], normalized_ts, normalized_email)
        if dedup_key in seen:
            continue

        seen.add(dedup_key)
        output.append(
            {
                "user_id": row["user_id"],
                "ts": normalized_ts,
                "email": normalized_email,
            }
        )

    return output


parts: dict[int, list[dict]] = defaultdict(list)
for row in rows:
    parts[partition_key(row, n=2)].append(row)

local_cleaned = []
for partition_id, partition_rows in parts.items():
    cleaned_rows = clean_partition(partition_rows)
    print(f"partition={partition_id}, in={len(partition_rows)}, out={len(cleaned_rows)}")
    local_cleaned.extend(cleaned_rows)

global_seen = set()
final_rows = []
for row in local_cleaned:
    key = (row["user_id"], row["ts"], row["email"])
    if key in global_seen:
        continue
    global_seen.add(key)
    final_rows.append(row)

print("final_rows =", final_rows)

assert len(final_rows) == 2
assert final_rows[0]["email"] == "a@example.com"
assert final_rows[1]["email"] == "b@example.com"
assert ceil((10 * 1024 * 1024) / 128) == 81920
```

这段代码当然不是生产方案，但它把分布式实现里最重要的两层结构说清楚了：

1. 分区内先做便宜且局部的清洗。
2. 全局只处理真正需要一致性的那一步。

如果把这段代码翻译成“分布式脑图”，它对应的是：

| Python 玩具步骤 | 分布式含义 |
| --- | --- |
| `partition_key` | 决定数据如何分布 |
| `clean_partition` | 每个 worker 对本地数据做清洗 |
| `seen` 局部集合 | 分区内局部去重状态 |
| `global_seen` | 最终全局一致性阶段 |
| `assert` | 用最小样例验证逻辑正确性 |

下面看 Spark 版示例，贴近真实工程。这里不再写成过于理想化的伪代码，而是写成更接近可执行的 Structured Streaming 样式：

```python
from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .appName("distributed-cleaning-demo")
    .getOrCreate()
)

raw = (
    spark.readStream
    .format("json")
    .schema("""
        order_id STRING,
        user_id STRING,
        event_time STRING,
        email STRING,
        amount DOUBLE
    """)
    .load("/data/orders")
)

cleaned = (
    raw
    .withColumn("event_time", F.to_timestamp("event_time"))
    .withColumn("email", F.lower(F.trim("email")))
    .filter(F.col("event_time").isNotNull())
    .filter(F.col("email").contains("@"))
    .filter(F.col("amount").isNotNull() & (F.col("amount") >= 0))
    .repartition("user_id")
)

deduped = (
    cleaned
    .withWatermark("event_time", "2 hours")
    .dropDuplicates(["order_id", "event_time"])
)

query = (
    deduped
    .writeStream
    .format("parquet")
    .option("path", "/warehouse/clean_orders")
    .option("checkpointLocation", "/checkpoint/clean_orders")
    .outputMode("append")
    .start()
)

query.awaitTermination()
```

这段代码里有四个关键点：

- `repartition("user_id")`：让后续按用户维度的处理尽量在稳定分区上进行。
- `filter(...)`：把明显无效的数据尽早删掉，减少后续 shuffle 体积。
- `withWatermark("event_time", "2 hours")`：告诉引擎“迟到超过 2 小时的数据，不再无限期保留判重状态”。
- `checkpointLocation`：把进度和状态写到可靠存储，失败后可以恢复，而不是整条流从头重跑。

初学者还需要知道一个事实：  
`repartition("user_id")` 并不代表后面的所有操作都“完全不需要 shuffle”。如果后续又按 `order_id` 做 `join` 或按另一个 key 聚合，仍然可能触发新的 shuffle。它的作用是把一段连续处理尽量建立在同一套分布之上，而不是保证全程零重排。

真实工程例子可以是电商订单清洗：每天从对象存储读取多 TB 订单、退款、用户行为日志。订单表按 `user_id` 或 `order_id` 分区后先做字段标准化与非法值过滤；退款关联需要一次基于 `order_id` 的 shuffle；最终流式去重需要基于 `order_id + event_time` 并结合 watermark 约束状态大小。

Dask 和 Ray 的思路类似，但接口风格不同。

Dask 常见写法是：先用 `map_partitions` 做局部清洗，再在必要时 `set_index` 或重分区。例如：

```python
import dask.dataframe as dd
import pandas as pd

df = dd.read_parquet("/data/orders/*.parquet")

def clean_partition(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.copy()
    pdf["email"] = pdf["email"].str.strip().str.lower()
    pdf["event_time"] = pd.to_datetime(pdf["event_time"], errors="coerce")
    pdf = pdf[pdf["event_time"].notna()]
    pdf = pdf[pdf["email"].str.contains("@", na=False)]
    return pdf.drop_duplicates(subset=["order_id", "event_time"])

cleaned = df.map_partitions(clean_partition)
result = cleaned.set_index("user_id")
```

Ray 更适合把每个清洗阶段封装成 pipeline task，尤其适合复杂自定义算子和超大规模数据流水线。它的重点通常不是 SQL 表达能力，而是任务编排的灵活性。

---

## 工程权衡与常见坑

分布式清洗最常见的问题，不是“代码跑不起来”，而是“能跑，但慢、贵、还不稳定”。

| 常见坑 | 现象 | 根因 | 规避方式 |
| --- | --- | --- | --- |
| 流式 `dropDuplicates` 不设 watermark | 状态越来越大，最终 GC/OOM | 状态无上界 | 用 `withWatermark`，优先有界状态去重 |
| 分区太大 | 单任务内存爆、spill 严重 | 每块数据超出 worker 稳定处理范围 | 控制单分区体量，结合内存压测 |
| 分区太小 | 调度开销高，任务数爆炸 | 任务管理成本超过计算成本 | 合并小分区，避免百万级任务图 |
| 频繁 `repartition` | 网络占满，stage 很慢 | 重复 shuffle | 只在关键边界重分区一次 |
| 用 `groupByKey` 收全量明细 | 网络量暴涨 | 没有本地预聚合 | 先局部聚合，再全局合并 |
| 断点续跑不完整 | 失败后重复写或漏写 | 无 checkpoint 或 sink 非幂等 | checkpoint + 幂等输出 |

特别要强调一个坑：Spark Structured Streaming 中，不带事件时间边界的去重会长期保留状态。对新手来说，最安全的心智模型是：只要你在流里做“记住历史”的操作，就必须回答“这段历史要保留多久”。回答不了，就会付出状态膨胀的代价。

还可以把常见问题再压缩成一套排查顺序：

1. 先看是不是过滤太晚，导致脏数据也参与了 shuffle。
2. 再看是不是分区键选错，导致同类 key 过于分散。
3. 再看是不是分区大小失衡，产生数据倾斜。
4. 最后看是不是状态没回收，或者输出不是幂等的。

数据倾斜尤其值得单独说。所谓倾斜，就是少数 key 的数据量远大于其他 key，导致：

- 大多数任务很快完成
- 少数任务拖很久
- 总作业时间被最慢的几个任务决定

例如订单数据里某个平台账号、机器人账号或热门商品可能聚集大量记录。此时即使总分区数合理，单个热点 key 也可能把某个任务压得很重。典型应对方式包括：

- 预先观察 key 分布，避免盲选分区键
- 对热点 key 做拆分或加盐
- 能先过滤的先过滤，先减少热点上的无效数据

Dask 也有对应问题。官方建议避免过大的分区，也避免过小的分区；Parquet 场景下常建议单分区落在约 100 到 300MiB 的内存规模。工程上常把“单 chunk 不超过 worker 内存的大约十分之一”当作保守起点，再根据 dashboard 观察峰值内存和任务排队情况调整。这是经验启发，不是硬规则。

新手做工程时，至少应该盯四类指标：

| 指标 | 说明 | 异常时通常意味着什么 |
| --- | --- | --- |
| shuffle 读写量 | 网络重排的数据规模 | 重分区太多或聚合太重 |
| spill 到磁盘量 | 内存放不下后写磁盘 | 分区太大或聚合状态太重 |
| task 时长分布 | 每个任务耗时差异 | 倾斜或局部热点 |
| 状态大小 | 流式状态存储占用 | watermark 不合理或去重范围过大 |

---

## 替代方案与适用边界

不同引擎解决的是不同类型的问题，不存在绝对最优。

| 引擎 | 优势 | 短板 | 适用边界 |
| --- | --- | --- | --- |
| Spark | SQL 能力强，shuffle/流批一体成熟，checkpoint/watermark 完整 | Python 自定义逻辑写起来不如纯 Python 灵活 | 大规模表清洗、流式管道、湖仓场景 |
| Dask | Python 原生，和 pandas/NumPy 迁移成本低 | 大规模 shuffle 和超复杂任务图时更敏感 | 中等规模数据、Python 算法型清洗 |
| Ray | 自定义任务编排灵活，适合复杂 pipeline | 需要更强的工程控制能力 | 多阶段 AI 数据流水线、复杂算子链 |
| Data-Juicer on Ray | 针对数据处理流水线做了算子和分布式优化 | 场景偏数据集构建，不是通用 SQL 引擎 | 超大样本集清洗、文本 dedup、训练集构建 |

真实边界可以这样理解：

- 如果你主要在做表级过滤、格式化、关联、批流统一，优先 Spark。
- 如果你已有大量 pandas 清洗逻辑，且数据量是几十 GB 到数百 GB，优先 Dask 做低成本扩展。
- 如果你的流程像“分词、规则过滤、质量打分、MinHash-LSH 去重、分阶段落盘”，Ray 或 Data-Juicer 更自然。

为了避免“选型全靠印象”，可以再补一个更具体的判断表：

| 你面临的问题 | 更可能优先的方案 |
| --- | --- |
| 需要 SQL、窗口、流式去重、checkpoint | Spark |
| 主要是 pandas 风格处理，团队熟 Python | Dask |
| 需要把自定义模型推断、规则算子、异构任务串起来 | Ray |
| 目标是大规模文本/样本集构建与去重 | Data-Juicer on Ray |

一个有代表性的真实工程例子是 Data-Juicer 的 Ray 分布式模式：公开资料展示过在数千 CPU 核规模上处理数十亿到数百亿样本，并提供 MinHash-LSH 去重和分布式流水线能力。这类方案适合云上超大规模数据集构建；但如果你只是清洗几亿行业务日志，用 Spark 往往更直接。

这里的核心不是“哪个框架更强”，而是“你的主要代价来自哪里”：

- 如果代价主要来自大表 shuffle，Spark 更成熟。
- 如果代价主要来自 Python 自定义逻辑迁移成本，Dask 更顺手。
- 如果代价主要来自多阶段流水线编排，Ray 更灵活。

---

## 参考资料

| 资料 | 作用 |
| --- | --- |
| Apache Spark SQL Performance Tuning: https://spark.apache.org/docs/3.5.1/sql-performance-tuning.html | 支撑 128MB 文件分区上限这一估算起点 |
| Apache Spark Structured Streaming Guide: https://spark.apache.org/docs/4.0.2/streaming/apis-on-dataframes-and-datasets.html | 支撑 watermark、流式去重、状态边界、流式 join 约束 |
| Spark Dataset API: https://spark.apache.org/docs/4.0.0/api/java/org/apache/spark/sql/Dataset.html | 支撑 `dropDuplicatesWithinWatermark` 的行为说明 |
| Dask Best Practices: https://docs.dask.org/en/stable/best-practices.html | 支撑“避免过大分区、避免过小任务、结合内存与核心数估算 chunk” |
| Dask DataFrame + Parquet: https://docs.dask.org/en/latest/dataframe-parquet.html | 支撑 Parquet 常见 100-300MiB 分区建议 |
| Data-Juicer Distributed Processing: https://modelscope.github.io/data-juicer/en/v1.4.2/docs/Distributed.html | 支撑 Ray 模式、超大规模样本处理、MinHash-LSH 去重案例 |

本文中的两个经验数字需要按“保守估算”理解，而不是当成固定常数：

- `N_partition = ceil(D / 128MB)` 是基于 Spark 常见默认读分区上限的起算公式，不是所有格式和配置下都强制成立。
- “每核内存约为分区大小的 4 到 6 倍”与“chunk 先按 worker 内存十分之一试探”属于工程估算启发，最终要用实际数据分布、倾斜程度、dashboard 指标和 spill 比例回调。

如果把全文压缩成一个可执行的判断框架，可以归纳为四步：

1. 先定义唯一键、迟到边界、恢复策略。
2. 再确定分区键，让尽可能多的清洗停留在本地。
3. 把过滤、标准化、局部去重前置，尽量缩小 shuffle 输入。
4. 只把必须全局协调的步骤留给 shuffle，并持续观察状态、spill 和倾斜。
