## 核心结论

增量数据采集的目标很直接：只拿“自上次成功同步之后发生变化的数据”，不重复扫描整张表。这里的“位点”是核心术语，白话讲就是“系统记住自己上次读到哪里了”的一个坐标。这个坐标可以是时间戳、自增 ID，或者数据库日志里的 offset。

最常见的新手版做法是保存上次提取时间，例如上次成功同步结束后，系统写下 `2026-04-13 20:00:00`。下一次执行时，只查：

```sql
SELECT *
FROM orders
WHERE updated_at > '2026-04-13 20:00:00';
```

这样就只处理更晚发生变化的订单，不需要把旧数据再扫一遍。处理完成后，再把新的位点写回存储，供下一轮继续使用。

增量采集的价值不在“绝对正确”，而在“用更低代价逼近需要的一致性”。如果业务只要求每天同步一次报表，可能全量更简单；如果要求分钟级同步、还要感知删除，那么就需要更严谨的增量机制，甚至直接上 CDC。

最短的对比可以先看下面这张表：

| 方案 | 扫描开销 | 实时性 | 删除感知 |
| --- | --- | --- | --- |
| 全量采集 | 高 | 低 | 有，前提是目标端做全量覆盖 |
| 增量采集 | 低到中 | 中到高 | 取决于位点机制，CDC 最强 |

---

## 问题定义与边界

问题定义可以写成一句话：如何在不反复全表扫描的前提下，尽量完整地拿到“新变化”的数据。

“变化”这个词要先说清。对一张业务表来说，变化通常包含三类：

| 变化类型 | 白话解释 | 常见来源 |
| --- | --- | --- |
| Insert | 新增了一行 | 新订单、新用户 |
| Update | 旧行内容变了 | 订单状态更新、地址修改 |
| Delete | 某行被删了 | 软删、物理删除、归档删除 |

如果你同步的是订单表，新手最直观的理解就是：只关心自上次同步以来 `updated_at` 变了，或者 `id` 比上次更大的订单，其他没变的记录不动。

因此，最基础的“变化集合”可以写成：

$$
\Delta(t) = \{row \mid timestamp(row) > ts_{last}\} \cup \{row \mid id(row) > id_{last}\}
$$

这表示：在当前同步时刻 $t$，要采集的数据是“时间戳比上次位点更新的记录”与“ID 比上次位点更大的记录”的并集。这里的 $\Delta(t)$ 不是数学上的高深概念，白话讲就是“这次要补抓的数据集合”。

但边界也很明确：

1. 如果源库只有 `updated_at`，没有删除日志，那么物理删除通常看不见。
2. 如果业务系统机器时间不准，`updated_at` 可能乱跳，导致漏采或重复。
3. 如果一条记录先更新后回滚，或者跨事务提交顺序复杂，单纯按业务列查询不一定能还原真实变更顺序。
4. 如果同步过程中失败，位点什么时候写回，会直接决定是否丢数或重数。

所以“增量采集”不是一个单一技术，而是一套边界清晰的工程约定：你先定义什么叫变化，再定义怎么记录变化，再定义失败后如何恢复。

玩具例子可以帮助建立直觉。

假设订单表当前只有 5 行，系统上次同步位点是 `ts_last = 2026-04-13 20:00:00`：

| id | updated_at | status |
| --- | --- | --- |
| 1 | 2026-04-13 19:10:00 | paid |
| 2 | 2026-04-13 19:30:00 | paid |
| 3 | 2026-04-13 20:10:00 | shipped |
| 4 | 2026-04-13 21:00:00 | canceled |
| 5 | 2026-04-14 07:50:00 | paid |

这次增量采集只需要抓 `id=3,4,5`。因为它们的 `updated_at` 晚于上次位点。这样理解之后，增量采集的本质就很清楚了：不是“查所有数据”，而是“查位点之后的数据”。

---

## 核心机制与推导

增量采集真正稳定，依赖两件事：位点管理和事件顺序。

先看位点。位点可以是三类：

| 位点类型 | 白话解释 | 典型场景 |
| --- | --- | --- |
| 时间戳 `ts_last` | 记住上次同步到哪个时间 | 表里有可靠 `updated_at` |
| 自增 ID `id_last` | 记住上次处理到哪个主键 | 只关心 append-only 新增 |
| 日志 offset | 记住数据库日志读到哪条事件 | CDC、流式同步 |

如果只用时间戳，公式通常是：

$$
\Delta_{ts}(t)=\{row \mid updated\_at > ts_{last}\}
$$

如果只用自增 ID，则是：

$$
\Delta_{id}(t)=\{row \mid id > id_{last}\}
$$

两者组合时，系统常见做法是按 `(timestamp, id)` 做稳定排序。原因很实际：很多记录的 `updated_at` 会相同，只用时间戳会在边界位置重复或漏掉，所以要加一个次序更稳定的列来打破并列。

更严谨的查询条件通常长这样：

```sql
WHERE updated_at > :ts_last
   OR (updated_at = :ts_last AND id > :id_last)
ORDER BY updated_at, id
LIMIT 1000
```

这个条件比 `updated_at > :ts_last AND id > :id_last` 更安全。原因是我们需要表达“先按时间推进，时间相同再按 ID 推进”，而不是要求两个字段都同时更大。前者是字典序推进，后者会错误过滤掉一部分数据。

位点更新逻辑可以抽象成下面这个流程：

1. 读取上次成功位点 `(ts_last, id_last)`。
2. 查询位点之后的一批数据。
3. 按顺序处理并写入目标端。
4. 处理成功后，把本批最后一条记录的 `(updated_at, id)` 写成新位点。
5. 下一轮从新位点继续。

简化流程图可以写成：

`读取旧位点 -> 查询增量批次 -> 写目标端 -> 成功后提交新位点 -> 下轮继续`

这里“成功后提交新位点”非常关键。因为位点本质上是断点续传的依据。断点续传的白话解释是“任务中断后，下次能从上次结束处继续”。如果数据还没写入目标端，你就提前把位点推进了，失败后就会漏数；如果数据写完了但位点没写回，下次会重放这批数据，所以目标端必须支持幂等。幂等的白话解释是“同一条数据重复处理多次，结果不变”。

再看 CDC。CDC 是 Change Data Capture，白话讲就是“直接监听数据库的变更日志，而不是反复查业务表”。它不是通过 `SELECT ... WHERE updated_at > ...` 推断变化，而是直接从事务日志里拿到 `insert/update/delete` 事件。

CDC 的核心链路是：

1. 数据库产生事务日志，例如 MySQL binlog、PostgreSQL WAL。
2. CDC 组件读取日志并解析成结构化事件。
3. 每条事件带上表名、操作类型、主键、变更前后值、日志位点。
4. 下游系统消费这些事件并更新目标端。
5. 消费位点和日志位点共同支持断点恢复。

这时位点不再是业务字段，而是日志 offset。好处是：

- 删除可以被显式捕获。
- 顺序更接近事务提交顺序。
- 不需要给 OLTP 表持续施加条件查询压力。

真实工程里，Kafka Connect JDBC 的 `timestamp+incrementing` 模式和 Debezium 的 CDC 模式，就是这两类路线的代表。

一个典型的 JDBC 增量模式会按 `(updated_at, id)` 轮询；而 Debezium 则直接读取 WAL/binlog，把每次 `insert/update/delete` 发布到 Kafka 主题。前者实现简单，后者一致性和实时性更强。

---

## 代码实现

先给一个可运行的 Python 玩具实现，模拟“按 `(updated_at, id)` 位点分页抓取”的逻辑。这个例子重点不在数据库驱动，而在位点推进规则。

```python
from datetime import datetime

rows = [
    {"id": 1, "updated_at": "2026-04-13 19:10:00", "status": "paid"},
    {"id": 2, "updated_at": "2026-04-13 20:10:00", "status": "paid"},
    {"id": 3, "updated_at": "2026-04-13 20:10:00", "status": "shipped"},
    {"id": 4, "updated_at": "2026-04-13 21:00:00", "status": "canceled"},
    {"id": 5, "updated_at": "2026-04-14 07:50:00", "status": "paid"},
]

def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

def incremental_fetch(rows, last_ts, last_id, limit=1000):
    # 字典序推进：先比较时间戳，再比较 ID
    result = []
    for row in rows:
        row_ts = parse_ts(row["updated_at"])
        if row_ts > last_ts or (row_ts == last_ts and row["id"] > last_id):
            result.append(row)

    result.sort(key=lambda r: (parse_ts(r["updated_at"]), r["id"]))
    batch = result[:limit]

    if not batch:
        return [], (last_ts, last_id)

    new_last = (parse_ts(batch[-1]["updated_at"]), batch[-1]["id"])
    return batch, new_last

last_ts = parse_ts("2026-04-13 20:00:00")
last_id = 0

batch1, checkpoint1 = incremental_fetch(rows, last_ts, last_id, limit=2)
assert [r["id"] for r in batch1] == [2, 3]
assert checkpoint1[1] == 3

batch2, checkpoint2 = incremental_fetch(rows, checkpoint1[0], checkpoint1[1], limit=10)
assert [r["id"] for r in batch2] == [4, 5]
assert checkpoint2[1] == 5
```

这个例子说明了三件事：

1. 位点不是“当前系统时间”，而是“本批最后一条成功处理记录的位置”。
2. 排序和过滤条件必须一致，否则分页会乱。
3. 断点恢复时，直接从上次位点重新拉取后续批次即可。

如果写成 SQL 伪代码，可以更接近实际系统：

```sql
SELECT id, updated_at, status
FROM orders
WHERE updated_at > :ts_last
   OR (updated_at = :ts_last AND id > :id_last)
ORDER BY updated_at, id
LIMIT 1000;
```

处理逻辑可以配成下面这种伪代码：

```text
load checkpoint(ts_last, id_last)
loop:
  batch = query_incremental(ts_last, id_last, limit=1000)
  if batch is empty:
    break

  for row in batch:
    upsert_to_target(row)   # 目标端需要幂等

  ts_last, id_last = last(batch).updated_at, last(batch).id
  save_checkpoint(ts_last, id_last)
```

真实工程例子可以看 Kafka Connect JDBC Source。一个典型配置如下：

```properties
name=orders-jdbc-source
connector.class=io.confluent.connect.jdbc.JdbcSourceConnector
tasks.max=1

connection.url=jdbc:postgresql://db:5432/app
connection.user=app_user
connection.password=secret

table.whitelist=orders
mode=timestamp+incrementing
timestamp.column.name=updated_at
incrementing.column.name=id

topic.prefix=jdbc-
poll.interval.ms=10000
batch.max.rows=1000

offset.storage.topic=connect-offsets
offset.storage.partitions=25
offset.storage.replication.factor=1
```

这里几个关键参数的含义必须说清：

| 参数 | 作用 | 白话解释 |
| --- | --- | --- |
| `mode=timestamp+incrementing` | 使用双位点增量 | 先按时间，再按 ID |
| `timestamp.column.name` | 时间位点列 | 告诉连接器用哪个时间字段判断更新 |
| `incrementing.column.name` | 递增位点列 | 用来解决同时间戳并列问题 |
| `offset.storage.topic` | offset 存储位置 | 把“上次读到哪”持久化到 Kafka |

如果需求升级为“要看见删除、要更低延迟、要少打业务库”，做法通常切到 Debezium。代码侧反而更简单：启动连接器，把数据库日志变成 Kafka 事件，然后消费者按主键做 upsert 或 delete。真正需要你自己保证的，是消费者幂等和目标端顺序处理能力。

---

## 工程权衡与常见坑

增量采集不是“更高级的全量”，而是拿复杂性换性能。下面这些坑是最常见的。

| 问题 | 影响 | 对策 |
| --- | --- | --- |
| 只靠 `updated_at`，看不见 delete | 目标端残留脏数据 | 用 CDC，或源表做软删标记 |
| 时钟漂移导致时间戳不准 | 漏采、重采 | 统一数据库时间源，不依赖应用机时间 |
| 排序条件和位点条件不一致 | 分页边界错乱 | 过滤和排序都用同一组列 |
| 失败后先提交位点 | 直接漏数据 | 先写目标端，再提交位点 |
| 重试导致重复写入 | 下游数据重复 | 目标端按主键幂等 upsert |
| 高频轮询打 OLTP | 影响线上业务 | 拉大轮询间隔、加索引，或切到 CDC |
| 增量列无索引 | 每次查询退化成全表扫 | 给 `updated_at` 或组合索引建索引 |

新手最容易犯的错，是把“当前时间”写成位点。比如这轮任务从 08:00 开始，查完一批数据后直接把位点写成 `08:00`。如果在 08:00 到 08:01 之间，又有一条旧事务晚提交，而它的业务时间小于 08:00，就可能被永久漏掉。更稳妥的做法是把位点推进到“本批最后一条实际处理记录的排序键”。

另一个常见坑是 delete。举个最简单的例子：你只靠

```sql
WHERE updated_at > :ts_last
```

抓订单更新。某个订单被物理删除后，这一行已经不存在了，你根本查不出来，因此目标端还会保留那条订单。这个问题不是 SQL 写得不够复杂，而是信息源本身不够。要解决，就要么改模型做软删字段，例如 `is_deleted=1`，要么改机制用 CDC。

真实工程里还要注意目标端语义。如果目标端是数仓明细层，通常需要保留变更历史，那就不能简单覆盖；如果目标端是缓存或宽表，可以用主键 upsert。位点机制只解决“拿到变化”，并不自动解决“怎么落地变化”。

---

## 替代方案与适用边界

工程上通常有三种主路线：全量、SQL 增量、CDC。它们没有谁绝对更好，关键看数据量、实时性和一致性要求。

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 全量抽取 | 最简单，逻辑直观 | 成本高、慢 | 数据量小、同步窗口充裕 |
| SQL 增量 | 实现快、改造小 | delete 弱、依赖业务列 | 已有 `updated_at` 或自增 ID |
| CDC | 实时、能看见 delete、顺序更稳 | 部署复杂、依赖日志权限 | 低延迟同步、审计、流式处理 |

如果 OLTP 表不大，每天夜间同步一次，直接全量可能反而最稳。因为系统复杂度低，失败恢复也简单，代价只是多扫一些数据。

如果表已经有可靠的 `updated_at`，业务只要求“新增和更新尽快同步，不强求删除秒级可见”，SQL 增量是成本最低的方案。它对初级工程师也最友好，因为核心就是“读位点、查增量、写位点”。

如果你要做实时数仓、事件流处理、缓存刷新，或者必须正确处理 delete，那么 CDC 更合适。比如 Debezium 读取 PostgreSQL WAL，把订单表每次 insert/update/delete 推到 Kafka，后面的风控、报表、搜索索引都订阅这个主题。这时增量不再依赖表扫描，而是天然变成流式场景中的事件消费。

Kafka Connect 的增量同步模式，就是这类工程实践的典型中间态：

1. 业务简单时，用 JDBC Source 的 `timestamp+incrementing`。
2. 一致性要求提高时，切到 Debezium CDC。
3. 下游统一从 Kafka 消费，减少源库多方直连。

可以把适用边界总结成一句更实用的话：

- 数据少、窗口宽：全量。
- 逻辑简单、有时间戳：SQL 增量。
- 要 delete、要低延迟、要日志级一致性：CDC。

---

## 参考资料

1. APXML, *Full vs Incremental Extraction*  
   重点是全量与增量采集的定义、优缺点对比，以及增量方案在 delete、状态管理方面的天然限制。  
   https://apxml.com/courses/intro-etl-pipelines/chapter-2-the-extraction-stage/full-vs-incremental-extraction?utm_source=openai

2. Hevo Data, *Incremental Data Load vs Full Load*  
   重点是时间戳、自增 ID、位点管理的基本机制，适合建立“上次同步到哪里”这个核心概念。  
   https://hevodata.com/learn/incremental-data-load-vs-full-load/?utm_source=openai

3. CodeStudy, *Kafka Connect JDBC Source Incremental*  
   重点是 Kafka Connect JDBC 的增量配置方式，以及与 Debezium CDC 的工程落地关系，适合继续看配置示例和流程图。  
   https://www.codestudy.net/blog/kafkaconnect-jdbc-source-incremental/?utm_source=openai

可按以上链接继续阅读更完整的流程图、连接器参数说明和 CDC 实践细节。
