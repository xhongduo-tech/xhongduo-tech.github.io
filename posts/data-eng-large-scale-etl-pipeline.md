## 核心结论

大规模 ETL（Extract, Transform, Load，白话说就是“把数据取出来、整理干净、再放进目标系统”）的核心，不是单纯把数据搬得更快，而是在吞吐、正确性、延迟和可恢复性之间做可控平衡。吞吐高但重复写入严重，系统仍然不可用；延迟低但失败后无法重跑，结果也不可靠。

一个可落地的 ETL 流程通常包含 6 个环节：抽取、标准化、校验、去重、分区写入、失败重跑。大规模场景下，批处理（一次处理一大批数据）和流处理（持续处理新到数据）往往同时存在。批处理负责历史回补和低成本吞吐，流处理负责低延迟更新，两者共享同一套主键、版本、去重规则，才能避免“历史一套逻辑、实时一套逻辑”造成的数据分叉。

先看一个玩具例子。假设你有两个来源：订单库和支付库。订单库里用户 ID 是整数，支付库里用户 ID 是字符串；同一笔订单可能被上游重复推送 3 次。如果你只是把两边数据直接拼起来写入仓库，结果通常是字段类型不一致、重复记录增加、下游报表翻倍。正确做法是先统一字段格式，再基于唯一键和版本去重，最后按日期分区写入，这样即使重跑，也不会再多写一次。

下面这个表说明 ETL 里几个常见目标的方向：

| 指标 | 定义 | 目标方向 | 说明 |
|---|---|---|---|
| 吞吐 $T$ | 单位时间处理记录数 | 越大越好 | 决定系统能否扛住数据量 |
| 延迟 $L$ | 从抽取到可查询的时间 | 越小越好 | 决定数据新鲜度 |
| 成功率 $S$ | 成功任务数 / 总调度数 | 越大越好 | 反映稳定性 |
| 重复率 $R$ | 重复写入数 / 写入总数 | 越小越好 | 反映是否会污染数据 |
| 幂等性 $I$ | 重跑后结果是否不变 | 必须保障 | 反映可恢复性 |

---

## 问题定义与边界

ETL 的边界首先要定义清楚，否则团队很容易只盯一个指标。例如只追求吞吐，就可能接受高重复率；只追求低延迟，就可能牺牲校验和重跑能力。工程上更稳妥的做法，是先把指标写成可观测对象，再决定架构。

常见定义如下：

$$
T=\frac{\text{处理记录数}}{\text{秒}}
$$

$$
L=\text{从抽取开始到加载完成的平均端到端时间}
$$

$$
S=\frac{\text{成功任务数}}{\text{总调度数}}
$$

$$
R=\frac{\text{重复写入记录数}}{\text{总写入记录数}}
$$

目标通常是最大化 $T, S$，最小化 $L, R$，同时保证幂等性 $I$。这里“幂等性”可以理解成：同一批数据重复执行多次，最终结果与执行一次相同。

典型流程的边界也要明确：

| 环节 | 作用 | 失败后是否可重跑 | 常见风险 |
|---|---|---|---|
| 抽取 | 从数据库、日志、消息队列取数 | 应可重跑 | 漏拉、重复拉取、源端限流 |
| 标准化 | 统一字段名、类型、时间格式 | 应可重跑 | 类型漂移、时区错误 |
| 校验 | 检查空值、范围、枚举合法性 | 应可重跑 | 脏数据混入 |
| 去重 | 消除重复事件或旧版本记录 | 必须可重跑 | 主键设计错误 |
| 分区写入 | 按日期、业务线等组织存储 | 应可重跑 | 小文件、热点分区 |
| 失败重跑 | 恢复中断批次或重放消息 | 核心能力 | 重复写入、状态错乱 |

一个新手容易忽略的问题是：批和流不是替代关系，而是边界不同。批处理适合大历史量，比如每天跑一次历史回补，每小时处理 500 万到 1000 万行；流处理适合增量，比如 Kafka 每秒持续进消息，要求几十秒内可见。两者必须共用去重规则，否则批处理回补时会把流处理已经写过的数据再写一遍。

可以把它理解为同一条时间线上的两种速度。批处理解决“过去的数据补齐没有”，流处理解决“现在的数据够不够新”。如果只做批，数据会旧；如果只做流，历史修复和回补会很痛苦。

---

## 核心机制与推导

真正决定大规模 ETL 能否长期稳定运行的，不是“有没有 Spark”或“是不是上了消息队列”，而是两套机制：幂等写入机制和批流一致机制。

第一套机制是幂等。最常见实现是“唯一键 + 版本号”。唯一键用于标识“这是谁”，版本号用于标识“这是谁的最新状态”。例如订单表可以用 `(order_id)` 作为唯一键，用 `event_time` 或 `version` 作为版本。写入时不是简单 `INSERT`，而是：

$$
\text{MERGE ON key, and update only when } incoming\_version > current\_version
$$

更具体一点，可写成：

$$
\text{if } k_{in}=k_{cur} \land v_{in}>v_{cur}, \text{ then update;}
$$

$$
\text{if } k_{in}=k_{cur} \land v_{in}\le v_{cur}, \text{ then ignore;}
$$

$$
\text{if no matching key, then insert.}
$$

这条规则的意义很直接：旧数据即使重放，也不会覆盖新数据；同一版本重复送达，也不会重复写入。

第二套机制是批流共存。CDC（Change Data Capture，白话说就是“记录一条数据是新增、更新还是删除”）常用 `I/U/D` 标记事件类型。流处理不断消费 CDC 事件，批处理则周期性做历史扫描或回补。只要两者都遵守同一套 `key + version + op` 规则，最终写入层就能收敛为一致结果。

下面是批处理和流处理的工程差异：

| 维度 | 批处理 | 流处理 |
|---|---|---|
| 目标 | 高吞吐、低成本处理历史 | 低延迟处理增量 |
| 延迟 | 分钟到小时 | 秒到分钟 |
| 吞吐 | 通常更高 | 通常受状态和实时约束 |
| 状态管理 | 批次边界清晰 | 要持续维护状态 |
| 常见工具 | Spark Batch、Hive、SQL Warehouse | Kafka、Flink、Spark Structured Streaming |
| 典型风险 | 回补过慢、窗口过大 | 状态膨胀、乱序、重复消费 |

玩具例子可以很简单。假设同一个用户资料发生了三次变更：

| user_id | version | city | op |
|---|---|---|---|
| 1 | 1 | Beijing | I |
| 1 | 2 | Shanghai | U |
| 1 | 2 | Shanghai | U |

如果不做幂等，第三条会被再次写入，导致重复；如果按 `user_id` 匹配、只接受更大版本，则最终只保留 `version=2` 的结果。

真实工程例子更接近这样：Kafka 中持续进入订单 CDC 事件，字段里有 `I/U/D`，下游用 Spark 先做字段标准化，比如金额统一成分、时间统一成 UTC、布尔值统一成 `0/1`，然后写入 Bronze 层（原始明细层，白话说就是“尽量保留原始信息的第一层存储”）。再从 Bronze 合并到 Silver 层（清洗后的分析层），合并规则基于复合键和版本执行 `MERGE`。这样即使 Kafka 某个分区被重放，或某个批次失败后重跑，最终表也不会出现同一订单多次插入。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现，演示“唯一键 + 版本号”的幂等去重。这个例子不依赖大数据框架，但逻辑与真实 ETL 一致。

```python
def merge_events(events):
    latest = {}
    for e in events:
        key = e["id"]
        version = e["version"]
        current = latest.get(key)

        if current is None or version > current["version"]:
            latest[key] = e
        elif version == current["version"]:
            # 同版本重复到达，忽略
            continue
        else:
            # 旧版本晚到，忽略
            continue
    return latest


events = [
    {"id": "A", "version": 1, "value": 10},
    {"id": "A", "version": 2, "value": 20},
    {"id": "A", "version": 2, "value": 20},  # 重复事件
    {"id": "B", "version": 3, "value": 30},
    {"id": "B", "version": 2, "value": 25},  # 旧版本晚到
]

result = merge_events(events)

assert result["A"]["value"] == 20
assert result["A"]["version"] == 2
assert result["B"]["value"] == 30
assert result["B"]["version"] == 3
assert len(result) == 2
```

这个例子说明，ETL 的关键不是“接到一条就写一条”，而是“按业务键决定该不该写”。

在真实工程里，常见做法是先写 Bronze，再 `MERGE` 到 Silver。下面是简化后的 Spark SQL 伪代码：

```sql
-- 1. 读取 Kafka CDC 事件，先做标准化
CREATE OR REPLACE TEMP VIEW cdc_cleaned AS
SELECT
  CAST(order_id AS STRING) AS order_id,
  CAST(user_id AS STRING) AS user_id,
  CAST(amount * 100 AS BIGINT) AS amount_cent,
  CAST(event_time AS TIMESTAMP) AS event_time,
  CAST(version AS BIGINT) AS version,
  UPPER(op) AS op,
  DATE(CAST(event_time AS TIMESTAMP)) AS dt
FROM kafka_bronze_raw
WHERE order_id IS NOT NULL
  AND version IS NOT NULL;

-- 2. 不急着 DISTINCT，先保留全量，再做后置去重
CREATE OR REPLACE TEMP VIEW cdc_dedup AS
SELECT *
FROM (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY order_id, version, op
           ORDER BY event_time DESC
         ) AS rn
  FROM cdc_cleaned
) t
WHERE rn = 1;
```

然后做幂等合并：

```sql
MERGE INTO silver_orders AS target
USING cdc_dedup AS source
ON target.order_id = source.order_id
WHEN MATCHED AND source.op = 'D' AND source.version >= target.version THEN
  DELETE
WHEN MATCHED AND source.op IN ('I', 'U') AND source.version > target.version THEN
  UPDATE SET
    target.user_id = source.user_id,
    target.amount_cent = source.amount_cent,
    target.event_time = source.event_time,
    target.version = source.version,
    target.dt = source.dt
WHEN NOT MATCHED AND source.op IN ('I', 'U') THEN
  INSERT (order_id, user_id, amount_cent, event_time, version, dt)
  VALUES (source.order_id, source.user_id, source.amount_cent, source.event_time, source.version, source.dt);
```

这里有两个关键点。

第一，不要在上游动不动就写 `SELECT DISTINCT`。`DISTINCT` 的本质通常需要全局去重，会触发大范围 shuffle（数据重分发，白话说就是“把很多机器上的数据重新洗牌”）。更常见、更稳的策略是 `UNION ALL` 保留明细，然后在靠近目标层时，按业务键和版本做窗口去重。

第二，分区写入要和查询模式匹配。例如订单按 `dt` 分区，是因为大多数报表按天查；如果按随机哈希分区，虽然写入均匀，但下游按天读时成本会升高。分区不是越细越好，过细会导致小文件问题，元数据压力也会上升。

---

## 工程权衡与常见坑

大规模 ETL 的很多问题，不是算法错，而是工程权衡没做对。下面这几个坑最常见。

| 常见坑 | 现象 | 根因 | 规避策略 |
|---|---|---|---|
| 滥用 `SELECT DISTINCT` | 作业突然变慢，shuffle 爆炸 | 过早全局去重 | 改为 `UNION ALL` + 后置窗口去重 |
| 没有唯一键 | 重跑后数据翻倍 | 无法判断同一业务实体 | 设计业务主键或复合键 |
| 没有版本控制 | 旧数据覆盖新数据 | 只按 key 更新 | 增加 version / event_time 比较 |
| 分区过细 | 文件过多，查询变慢 | 过度追求写入并行 | 按查询维度做粗粒度分区 |
| 抽取无节制并发 | 源库被打挂 | 忽略源端能力边界 | 限流、分片、增量抽取 |
| 流批逻辑分裂 | 回补后口径不一致 | 两套规则两套表 | 统一去重和 MERGE 规则 |

一个典型的真实工程坑是：团队在 JDBC 抽取阶段用了 `SELECT DISTINCT`，当输入速率接近 200 MB/s 时，Spark 为了做全局去重触发大规模 shuffle，结果吞吐下降、延迟上升，甚至挤占了后续写入资源。后来改为 `UNION ALL` 保留全部原始记录，只在落 Silver 前基于 `(business_key, version)` 做窗口去重，吞吐才稳定下来。

另一个坑是把“成功写入”误当成“结果正确”。例如任务成功率 $S$ 很高，但重复率 $R$ 也在上升，这说明系统只是“总能写进去”，并不代表“写得对”。因此监控不能只看作业是否失败，还要看重复写入率、端到端延迟、数据新鲜度、晚到数据比例等业务指标。

还有一个容易被低估的问题是失败重跑。很多新手把重跑理解成“再跑一遍”。实际上，真正困难的是“再跑一遍后结果不能变脏”。如果系统没有 checkpoint（检查点，白话说就是“保存处理中间状态的恢复点”）、没有批次水位线、没有幂等主键，那么每次重跑都可能把旧数据再插一遍。

---

## 替代方案与适用边界

没有一种 ETL 架构能同时把吞吐、延迟、成本、一致性都做到最好，所以需要按场景选型。

| 方案 | 适用场景 | 优点 | 瓶颈 |
|---|---|---|---|
| 纯批处理 | 报表为主、小时级可见即可 | 架构简单、成本低、吞吐高 | 数据不够新鲜 |
| 纯流处理 | 实时监控、风控、秒级响应 | 延迟低 | 状态复杂、回补困难 |
| 批 + 流混合 | 既要历史完整，又要实时更新 | 兼顾回补与新鲜度 | 规则统一难度高 |
| 微批 | 秒级到分钟级延迟可接受 | 比纯流简单，控制资源更容易 | 延迟不如真流低 |

如果延迟预算小于 30 秒，通常优先考虑流处理 CDC 管道，再用 `MERGE` 保证幂等；如果历史数据很多，仍然需要批处理做回补。对于极高负载但允许窗口延迟的业务，微批常常是更稳的工程选择，因为它比持续流更容易控制 JDBC 连接数、元数据压力和小文件数量。

可以用一个简单判断来理解边界：

1. 如果业务看重“现在发生了什么”，优先流或微批。
2. 如果业务看重“历史是否完整、口径是否稳定”，必须保留批处理回补链路。
3. 如果目标库写入成本高，比如数据库连接敏感、元数据管理重，尽量让流只写 Bronze，把复杂合并留给批或微批处理的 Silver 层。

因此，大规模 ETL 最稳的落地模式往往不是“全流”或“全批”，而是“源头增量捕获 + 中间层保留原始事实 + 目标层按主键版本合并”。

---

## 参考资料

- Dataflow 模型相关资料：用于理解大规模数据系统里正确性、延迟、成本之间的权衡框架，支撑本文对“平衡而非单点最优”的结论。
- OneUptime 的 ETL 实践文章：用于说明 ETL 的常规流程、幂等、监控和批流边界，支撑本文对标准化、校验、重跑的流程描述。
- Microsoft Q&A 关于 CDC、Kafka、Spark、Hyperscale 的讨论：用于支撑本文中的真实工程例子，包括 `I/U/D` 标记、复合键 `MERGE`、吞吐与 JDBC 瓶颈。
- MoldStud 关于 ETL 故障分析的文章：用于提供吞吐、延迟、成功率等指标定义，支撑本文中的公式和监控指标设计。
- Celonis ETL 最佳实践文档：用于支撑 `SELECT DISTINCT`、`UNION` 可能导致高成本重排，以及后置去重更稳妥的工程结论。
