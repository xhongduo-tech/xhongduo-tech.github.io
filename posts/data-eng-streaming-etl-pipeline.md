## 核心结论

流式 ETL 是一条“数据持续流入、持续加工、持续落库”的管线。ETL 指 Extract、Transform、Load，也就是抽取、转换、加载。它和传统批处理 ETL 的区别，不是“代码写法不同”，而是系统默认面对的是**无界数据**。无界数据指没有天然结束点的数据流，比如订单、日志、埋点、监控事件。

一条可落地的流式 ETL 架构，通常由三层组成：

| 组件 | 白话解释 | 核心职责 | 不该承担的职责 |
|---|---|---|---|
| Kafka / Pulsar | 像可持久化的消息传送带 | 接收生产者数据、按分区保存、提供重放能力 | 复杂时间窗口计算、状态 join |
| Flink | 像“理解时间和状态”的流处理引擎 | 事件时间、窗口、状态、checkpoint、join、聚合 | 长期存储业务结果 |
| ClickHouse / Iceberg / OLAP / DB | 最终结果仓库 | 查询、分析、下游消费 | 承担流计算过程中的中间状态 |

初学者最容易误解的一点是：**exactly-once 不是某一个组件单独提供的能力，而是消息系统、计算引擎、结果写入端协同出来的结果。**

如果把一次 checkpoint 记作 $t_k$，那么可以把核心一致性逻辑写成：

$$
recoverState(t_k) + replayRecords(from\ offsets_k) \Rightarrow exactly\text{-}once\ result
$$

意思是：故障后恢复到第 $k$ 次检查点保存的状态，再从当时保存的 source offset 继续重放，最终结果等价于“每条数据只被正确处理一次”。

玩具例子可以这样理解：Kafka 像一排邮件箱，生产者不断投信；Flink 像按时间处理信件的邮差，并把中间记账写进随身记事本；数据库像最终档案室。即使处理中途断电，只要 checkpoint 已经把“记事本内容”和“上次读到哪封信”一起保存，下次重启就能继续，不会多算也不会漏算。

---

## 问题定义与边界

流式 ETL 要解决的不是“怎么把数据搬运出去”，而是“怎么在持续到来的数据上，稳定地做变换、关联、容错和落地”。

这里有两个基础概念：

- 有界数据：有明确结束点的数据集，比如某天全部订单文件。
- 无界数据：不断到来的事件流，比如支付事件、用户点击流。

二者的计算难点不同：

| 类型 | 特征 | 常见处理方式 | 难点 |
|---|---|---|---|
| 有界数据 | 数据总量固定、最终会读完 | 批处理、一次性扫描 | 成本和时延 |
| 无界数据 | 数据持续到达、没有“结束” | 流处理、窗口计算 | 时间语义、乱序、状态恢复 |

流式 ETL 的系统边界必须清楚：

| 层级 | 负责什么 | 容错依赖什么 | 扩展靠什么 |
|---|---|---|---|
| Source 层：Kafka/Pulsar | 持久化消息、分区顺序、偏移管理 | 副本、日志持久化 | 增加 partition / topic |
| Compute 层：Flink | 时间戳、watermark、状态、窗口、join | checkpoint / savepoint / state backend | 增加并行度、key group 重分配 |
| Sink 层：存储系统 | 最终写入和查询 | 事务、幂等写、两阶段提交 | 分片、分区、集群扩容 |

这里的 watermark 是“系统对事件时间推进位置的估计”。白话说，它告诉算子：“大概率不会再有比这个时间更早的数据来了。”没有 watermark，窗口就无法知道何时可以安全关闭。

真实工程例子：电商平台要把“下单流”“支付流”“风控特征流”合并后写入 ClickHouse 做实时看板。Kafka 负责顶住流量高峰并保留可重放日志，Flink 负责按用户或订单 ID 做 join 和聚合，ClickHouse 负责提供秒级分析查询。任何一层职责不清，都会导致故障时难以恢复。

---

## 核心机制与推导

流式 ETL 真正难的部分有三个：时间、状态、恢复。

先看 checkpoint。checkpoint 可以理解成“全局一致快照”。Flink 会在数据流中插入 checkpoint barrier。barrier 是一种特殊标记，表示“到这里为止，当前处理位置可以形成一个一致状态”。

一次 checkpoint 通常会记录：

| 记录内容 | 作用 |
|---|---|
| Source offset | 恢复后知道从哪里重新读消息 |
| Operator state / Keyed state | 恢复中间计算进度 |
| Backend location | 知道状态文件保存在哪里 |
| Checkpoint ID | 对齐一次全局一致快照 |

为什么这能保证 exactly-once？因为故障后恢复的不是“应用代码跑到哪一行”，而是“整条数据流在某个时间点的统一处理位置”。

再看 join。流式 join 不是把两张完整表拿出来做笛卡尔组合，而是“在有限时间范围内，用状态暂存两边事件，等匹配条件成立时输出”。

假设订单流为 A，支付流为 B，匹配条件是同一个 `order_id`，并且支付时间在下单前 2 秒到后 1 秒内：

$$
b.timestamp \in [a.timestamp - 2s,\ a.timestamp + 1s]
$$

同时 key 相同，才输出 join 结果。

这就是 interval join。它的本质是：

1. 按 key 分区。
2. 把 A 流最近一段时间的数据放进 keyed state。
3. 把 B 流最近一段时间的数据也放进 keyed state。
4. 新事件到来时，在另一个状态区间里查找可匹配事件。
5. 超过时间边界后清理旧状态。

玩具例子：

- 订单 A：`order_id=1001, ts=10:00:00`
- 支付 B：`order_id=1001, ts=10:00:01`
- 规则：`[-2s, +1s]`

因为 `10:00:01 ∈ [09:59:58, 10:00:01]`，所以匹配成功。

如果支付在 `10:00:05` 才来，就不属于这个 join 窗口，Flink 不会输出结果。

这里还会遇到 key group。key group 是“key 到并行任务之间的稳定映射单元”。白话说，它不是直接把某个用户永远绑死在某台机器上，而是先映射到更细的逻辑桶，再由这些桶分配给并行实例。这样扩缩容时状态可以按桶迁移，不需要全部重算。

---

## 代码实现

下面先给一个可运行的 Python 玩具例子，模拟 interval join 的核心逻辑。它不是 Flink 生产代码，但能帮助理解“为什么需要保存短期状态”。

```python
from collections import defaultdict

def interval_join(left_events, right_events, lower_ms=-2000, upper_ms=1000):
    right_by_key = defaultdict(list)
    for e in right_events:
        right_by_key[e["key"]].append(e)

    result = []
    for a in left_events:
        for b in right_by_key[a["key"]]:
            delta = b["ts"] - a["ts"]
            if lower_ms <= delta <= upper_ms:
                result.append((a["key"], a["value"], b["value"]))
    return result

orders = [
    {"key": "1001", "ts": 10000, "value": "order_created"},
    {"key": "1002", "ts": 20000, "value": "order_created"},
]

payments = [
    {"key": "1001", "ts": 11000, "value": "paid"},
    {"key": "1002", "ts": 25050, "value": "paid"},
]

joined = interval_join(orders, payments)
assert joined == [("1001", "order_created", "paid")]
print(joined)
```

下面是简化后的 Flink DataStream 作业，展示 Kafka Source、Watermark、interval join 和输出逻辑：

```python
from pyflink.common import Duration, WatermarkStrategy
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer
from pyflink.datastream.functions import RuntimeContext
import json

env = StreamExecutionEnvironment.get_execution_environment()
env.enable_checkpointing(10000)  # 每 10 秒做一次 checkpoint
env.get_checkpoint_config().set_checkpoint_timeout(60000)
env.set_parallelism(2)

source_orders = KafkaSource.builder() \
    .set_bootstrap_servers("localhost:9092") \
    .set_topics("orders") \
    .set_group_id("etl-job") \
    .set_starting_offsets(KafkaOffsetsInitializer.earliest()) \
    .set_value_only_deserializer(SimpleStringSchema()) \
    .build()

source_payments = KafkaSource.builder() \
    .set_bootstrap_servers("localhost:9092") \
    .set_topics("payments") \
    .set_group_id("etl-job") \
    .set_starting_offsets(KafkaOffsetsInitializer.earliest()) \
    .set_value_only_deserializer(SimpleStringSchema()) \
    .build()

wm = WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(3))

orders = env.from_source(source_orders, wm, "orders") \
    .map(lambda s: json.loads(s)) \
    .key_by(lambda x: x["order_id"])

payments = env.from_source(source_payments, wm, "payments") \
    .map(lambda s: json.loads(s)) \
    .key_by(lambda x: x["order_id"])

joined = orders.interval_join(payments) \
    .between(Duration.of_seconds(-2), Duration.of_seconds(1)) \
    .process(lambda left, right: {
        "order_id": left["order_id"],
        "user_id": left["user_id"],
        "pay_amount": right["amount"]
    })

joined.print()
env.execute("streaming-etl-demo")
```

如果 sink 端要做严格 exactly-once，通常还会要求 Kafka 事务或两阶段提交。两阶段提交可以理解成“先预写、checkpoint 成功后再正式提交”的协议，用来避免任务失败时写出半条结果。

checkpoint 常用参数如下：

| 参数 | 作用 | 常见建议 |
|---|---|---|
| `checkpointing.interval` | checkpoint 周期 | 5s 到 60s，取决于吞吐和恢复目标 |
| `checkpoint.timeout` | 单次 checkpoint 超时 | 不能小于大状态快照所需时间 |
| `externalized` | 作业取消后是否保留 checkpoint | 生产环境通常保留 |
| `max concurrent checkpoints` | 同时进行的 checkpoint 数 | 状态大时不要过高 |
| `maxParallelism` | key group 上限 | 先规划好，避免未来扩容受限 |

真实工程例子：订单实时宽表作业从 Kafka 读“下单流”和“支付流”，再从维表系统补充商户属性，最后通过事务 sink 写入 ClickHouse 或 Iceberg。这个作业真正难的地方不是 join 语法，而是确保 checkpoint 周期、状态后端和 sink 提交节奏能一起稳定工作。

---

## 工程权衡与常见坑

流式 ETL 最大的代价不是代码复杂，而是**长期运行下的状态成本**。

最常见的问题是状态膨胀。状态膨胀指算子保存的中间数据越来越大，导致 checkpoint 变慢、恢复变慢、内存和磁盘持续上涨。stream join 尤其容易出现这个问题，因为两边历史数据都要留一段时间。

| 问题 | 典型指标 | 原因 | 对策 |
|---|---|---|---|
| 状态膨胀 | checkpoint duration、state size 持续上升 | join 窗口过大、TTL 缺失 | 缩短窗口、设置 TTL、拆分链路 |
| Backpressure | `backPressuredTimeMsPerSecond` 高 | 下游 sink 吞吐不足 | 提升 sink 并行度、批量写、异步 I/O |
| CPU 忙 | `busyTimeMsPerSecond` 高 | 算子计算重、序列化开销大 | 优化数据格式、减少热点 key |
| checkpoint 超时 | failed checkpoints 增多 | 状态太大或存储太慢 | 增加超时、增量 checkpoint、换更快 backend |

backpressure 就是“上游推得太快，下游接不住”。白话说，像流水线最后一道工位卡住，前面的产品会逐级堆积。在 Flink WebUI 里，如果某个算子的 `backPressuredTimeMsPerSecond` 很高，而下游 sink 的 busy time 也高，通常说明瓶颈在落库而不是在 join。

新手常犯的坑有四类：

1. 只看吞吐，不看恢复时间。高吞吐不是全部，状态过大时恢复十几分钟，实时链路就失去价值。
2. 只配 checkpoint，不验证 sink 事务。source 和 state 能恢复，不代表结果库不会重复写。
3. 忽略乱序数据。现实日志常常晚到，如果 watermark 太激进，会丢掉本应参与计算的事件。
4. 热点 key。比如某个超级商户或热门直播间的事件全集中到一个 key，会让单并行实例过载。

一个典型误区是把 join 状态保留 1 小时，只因为“怕错过迟到事件”。这通常会把 checkpoint 拖成灾难。更合理的做法是根据业务 SLA 设窗口，比如支付通常在下单后几分钟内完成，那就按业务时延分布设置状态 TTL，而不是无限放大保护范围。

---

## 替代方案与适用边界

不是所有 ETL 都应该上 Flink。

如果需求只是“每天把日志汇总入仓库”，而不是“秒级响应并容错恢复”，那 Spark Batch、离线 SQL、定时调度通常更简单，开发和运维成本都更低。

| 方案 | 优势 | 劣势 | 适合场景 |
|---|---|---|---|
| Spark Batch / 定时 SQL | 简单、便宜、易维护 | 实时性差 | 日报、T+1 汇总、离线数仓 |
| Kafka Streams | 与 Kafka 绑定紧、轻量 | 复杂状态和多流计算能力较弱 | 中小规模 Kafka 内部流处理 |
| Flink + Kafka/Pulsar | 时间语义强、状态能力强、容错完善 | 运维和调优复杂 | 实时风控、实时宽表、在线特征计算 |

Kafka 和 Pulsar 也有明显边界差异：

| 维度 | Kafka | Pulsar |
|---|---|---|
| 分区模型 | Topic-Partition | Topic-Partition，底层可分层管理 |
| 存储架构 | 计算与存储耦合更常见 | BookKeeper 架构更强调分离 |
| 事务生态 | 成熟，和 Flink 配合常见 | 也可做事务，但工程选型看团队经验 |
| 运维心智 | 生态成熟、资料多 | 架构更灵活，理解成本更高 |
| 适用业务 | 通用消息总线、成熟数据平台 | 多租户、分层存储、云环境弹性场景 |

白话理解：Pulsar 更像“自带冷热分层仓库的消息系统”，Kafka 更像“成熟稳定、工具链极丰富的工业标准总线”。两者都能接 Flink，但如果团队已经大量使用 Kafka，并且目标是稳定做实时 ETL，Kafka 往往是更低风险的选择；如果你很看重多租户隔离、分层存储和云原生弹性，Pulsar 的吸引力会更大。

---

## 参考资料

| 资料 | 主要内容 | 适合解决什么问题 |
|---|---|---|
| Apache Flink Architecture | Flink 架构、作业图、运行模型 | 理解 Flink 在整条链路中的角色 |
| Apache Flink Stateful Stream Processing | 状态、checkpoint、容错机制 | 理解 exactly-once 为什么成立 |
| Apache Flink Joining / Interval Join | interval join 语义和 API | 实现双流按事件时间关联 |
| Apache Flink Backpressure Docs | WebUI 指标与背压排查 | 线上性能定位和调优 |
| Kafka / Pulsar 官方文档 | 分区、存储、事务、消费者语义 | 选择消息系统和理解 source 语义 |
| 实时数据平台实践文章 | 电商、日志、风控等真实案例 | 参考端到端架构拆分方式 |

推荐阅读顺序：

1. 先读 Flink Architecture，建立“消息系统负责缓冲，Flink 负责时间和状态”的整体图。
2. 再读 Stateful Processing，理解 checkpoint、state backend、恢复链路。
3. 最后读 Joining 和 Backpressure 文档，把“能跑”推进到“能长期稳定跑”。
