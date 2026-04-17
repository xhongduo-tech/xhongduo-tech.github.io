## 核心结论

CDC，Change Data Capture，直译是“变更数据捕获”。它解决的不是“怎么定时把整张表再扫一遍”，而是“怎么把数据库已经提交的变更，按顺序、带上下文地持续发出去”。

对工程实践来说，数据库 CDC 集成的核心结论有三条：

1. 最稳定的做法不是轮询业务表，而是读取数据库自己的复制日志，例如 MySQL 的 binlog、PostgreSQL 的 WAL。因为这些日志本来就是数据库为复制和恢复准备的，所以天然带有顺序、事务边界和删除信息。
2. Debezium 的本质不是“同步工具”，而是“把数据库变更翻译成标准事件的源端连接器”。它运行在 Kafka Connect 中，负责读取 binlog/WAL，输出结构化 change event，再由 Kafka 解耦下游消费者。
3. CDC 不是“接上就行”。真正决定系统是否可上线的，是三件事：Schema 演进处理、延迟监控、以及多源合并时的冲突解决。前两者决定稳定性，后一者决定数据是否可信。

一个最小判断标准是：如果你的目标是让搜索、缓存、实时数仓、推荐系统尽快看到数据库最新状态，那么 CDC 往往比“每 5 分钟扫一次表”更准确、更便宜，也更容易拿到删除和中间状态。

再补一条新手常忽略的判断标准：CDC 传播的是“事实事件”，不是“最终表快照”。这意味着你拿到的不只是“现在是什么”，而是“这个状态是怎样一步一步变成现在的”。很多实时系统真正依赖的正是这个过程信息。

---

## 问题定义与边界

先定义问题。数据库 CDC 集成要解决的不是“数据迁移”，而是“把源库已经发生的变更，以低延迟、可消费的方式传播出去”。

它通常有以下目标：

| 目标 | 说明 | 对下游的直接价值 |
| --- | --- | --- |
| 降低源库压力 | 不再频繁 `SELECT updated_at > xxx` 扫表 | 减少索引扫描、锁竞争和 IO |
| 降低延迟 | 从分钟级批处理变成秒级或毫秒级传播 | 搜索、缓存、风控更快看到最新状态 |
| 捕获完整变更 | 不只拿到新增和更新，还能拿到删除 | 下游能正确清理脏数据 |
| 解耦下游 | 搜索、缓存、数仓、风控各自消费同一份事件 | 一个源头，多种消费 |
| 保留变更上下文 | 带上事务位点、表名、提交时间 | 便于排障、审计、回放 |

边界也要先说清楚：

| 边界 | 不要误解成什么 |
| --- | --- |
| CDC 负责传播变更 | 不保证你下游天然就“最终正确” |
| Debezium 负责读日志并发事件 | 不负责替你设计主键、去重、宽表口径 |
| Kafka 保证分区内顺序 | 不保证跨分区全局顺序 |
| Schema Registry 追踪模式变化 | 不等于业务字段变化就一定兼容 |
| 源库日志能反映提交事实 | 不等于所有业务语义都天然完整 |

这里的“业务语义”要单独解释。数据库日志知道一行被插入、更新、删除，但它不知道“这次状态变更是否表示支付成功、退款完成、风控放行”。如果下游需要的是业务事件而不是数据变更，CDC 可能还要和应用层事件一起使用。

玩具例子最容易理解。

你有一张 MySQL `users` 表：

| id | name | status |
| --- | --- | --- |
| 1 | alice | active |

之后发生三次操作：

1. 插入 `id=2, name=bob`
2. 更新 `id=1, status=suspended`
3. 删除 `id=2`

如果你用轮询，可能只能在下一次任务里看到“当前表状态”，看不到“bob 曾经存在过又被删了”这个过程。  
如果你用 CDC，Kafka 中会出现三条事件：`c`（create）、`u`（update）、`d`（delete）。这就是 CDC 的价值：它传播的是“变化”，不是某个时刻的静态快照。

把这个差异写成表会更清楚：

| 方案 | 能否看到 bob 被创建 | 能否看到 bob 被删除 | 能否还原中间状态 |
| --- | --- | --- | --- |
| 轮询业务表 | 可能看不到 | 通常看不到 | 通常不能 |
| 读取 binlog/WAL | 能 | 能 | 能 |

在 Debezium 的语境里，一个数据库源通常由一个 connector 管理；每张表的事件会落到类似 `server.schema.table` 或 `topicPrefix.schema.table` 的主题，例如：

| 日志类型 | Debezium 连接器 | Kafka 主题示例 |
| --- | --- | --- |
| MySQL binlog | `io.debezium.connector.mysql.MySqlConnector` | `mysql1.inventory.users` |
| PostgreSQL WAL | `io.debezium.connector.postgresql.PostgresConnector` | `pgserver.sales.orders` |

这里的 `server` 或 `topicPrefix` 不是随便写的字符串，而是源端身份标识。多源场景下，它相当于“数据谱系标签”，后续合并、隔离和排查问题都靠它。

---

## 核心机制与推导

先看传播链路：

数据库提交事务  
$\rightarrow$ binlog/WAL 记录变更  
$\rightarrow$ Debezium 解析日志并组装事件  
$\rightarrow$ Kafka 持久化事件  
$\rightarrow$ 下游消费者更新搜索、缓存、数仓、特征库

这条链路里最关键的是事件结构。Debezium 常见事件里会包含：

| 字段 | 含义 | 为什么重要 |
| --- | --- | --- |
| `op` | 操作类型，`c/u/d/r` 分别表示新增、更新、删除、快照读取 | 决定下游该插入、更新、删除还是标记快照 |
| `before` | 更新前或删除前的旧值 | 支持审计、差异计算、删除处理 |
| `after` | 更新后的新值 | 下游构造最新状态的依据 |
| `source` | 来源信息，例如库、表、事务位点、提交时间 | 用于幂等、追踪和问题定位 |
| `ts_ms` | 事件被处理或发出的时间戳 | 用于计算延迟 |
| `transaction` | 事务元数据，部分配置下可用 | 用于理解批量提交边界 |

这意味着消费者不是“猜发生了什么”，而是直接按事实处理。

一个典型事件可以抽象成：

```json
{
  "op": "u",
  "before": { "id": 1, "status": "active" },
  "after":  { "id": 1, "status": "suspended" },
  "source": {
    "db": "orders_db",
    "table": "users",
    "file": "mysql-bin.000123",
    "pos": 456789
  },
  "ts_ms": 1710000000123
}
```

### 1. 为什么 CDC 比轮询更准确

轮询本质上依赖一个条件，例如：

$$
\text{SELECT * FROM table WHERE updated\_at > last\_cursor}
$$

这个模式有四个问题：

1. 删除通常拿不到，因为被删的行已经不存在。
2. 高频更新会丢中间状态，只能看到“最后一次结果”。
3. 如果 `updated_at` 精度不够，可能重复或遗漏。
4. 频繁扫表会占用索引和 IO。

可以把它写成一个更一般的判断式。设一行记录在时间区间 $\Delta t$ 内更新了 $n$ 次，轮询间隔为 $T$。如果 $n > 1$ 且这 $n$ 次更新都发生在同一个轮询窗口内，那么轮询最多只能观察到窗口末尾状态，而 CDC 能观察到全部 $n$ 次变更。

也就是说：

$$
\text{ObservedStates}_{\text{polling}} \le 1,\quad
\text{ObservedStates}_{\text{CDC}} = n
$$

其中前者是“一个轮询窗口内最多观测到一个最终状态”，后者是“日志里保留了全部提交事件”。

CDC 避开了这四点，因为它读的是数据库提交后写入的日志，而不是业务表当前快照。

### 2. 延迟怎么定义

CDC 是否可用，不能只看“有没有消息”，而要看端到端延迟。最简单的定义是：

$$
L_{\text{E2E}} = t_{\text{consume}} - t_{\text{commit}}
$$

其中：

- $t_{\text{commit}}$：事务在源库提交的时间
- $t_{\text{consume}}$：下游真正处理这条事件的时间

这个总延迟可以继续拆成三段：

$$
L_{\text{E2E}} = L_{\text{DB}} + L_{\text{Debezium}} + L_{\text{Consumer}}
$$

分别表示：

- $L_{\text{DB}}$：数据库日志生成或复制延迟
- $L_{\text{Debezium}}$：Debezium 从日志解析到写入 Kafka 的延迟
- $L_{\text{Consumer}}$：Kafka 到下游系统的消费延迟

这套拆分很重要，因为告警必须能定位责任点。  
如果总延迟升高，但 Kafka lag 很低，问题大概率在源库复制或 Debezium connector；如果 Debezium 正常、Kafka 堆积严重，瓶颈就在下游消费者。

一个最小监控面板通常至少有这几项：

| 指标 | 解释 | 异常时通常意味着什么 |
| --- | --- | --- |
| `commit_ts -> connector_ts` | 源库到 Debezium 的延迟 | 日志复制、连接器读日志变慢 |
| `connector_ts -> kafka_append_ts` | Debezium 到 Kafka 的延迟 | Connect 写 Kafka 受阻 |
| `kafka_lag` | 消费组落后消息数 | 消费者处理能力不足 |
| `process_ts -> sink_visible_ts` | 下游写入后何时可见 | Elasticsearch/OLAP/缓存写入变慢 |

### 3. Schema 演进为什么是 CDC 的硬问题

Schema，模式，直白说就是“这条数据长什么样，有哪些字段、类型是什么”。

例如 `orders` 表原本是：

| id | amount |
| --- | --- |
| bigint | decimal(10,2) |

后来新增一列 `currency`，再后来把 `amount` 精度从 `decimal(10,2)` 改成 `decimal(18,4)`。  
这类变化在源库里很正常，但在 CDC 链路上会影响三层：

1. Debezium 是否能识别 DDL 或字段变化。
2. Kafka 中的序列化格式是否兼容旧版本消费者。
3. 下游表结构是否允许平滑升级。

把常见变更放在一起会更容易判断风险：

| 变更类型 | 源库侧风险 | 消费侧风险 | 推荐做法 |
| --- | --- | --- | --- |
| 新增可空字段 | 低 | 低到中 | 优先采用 |
| 新增非空字段无默认值 | 中 | 高 | 先加默认值或改成可空 |
| 扩大数值精度 | 低 | 中 | 先检查序列化和下游列类型 |
| 改字段名 | 中 | 高 | 视为破坏性变更，走兼容期 |
| 删字段 | 低 | 高 | 先停用，再删除 |
| 改主键 | 高 | 很高 | 通常要重建消费链路 |

所以 CDC 上线前必须回答一个问题：你的消费者是“严格依赖字段固定形状”，还是“能容忍字段新增和默认值补齐”。如果不能回答，Schema 演进迟早会在生产里打断同步。

---

## 代码实现

先看一个最小 Debezium MySQL connector 配置。它的作用是读取 `orders_db.orders` 的 binlog，并把事件写到 Kafka。

```json
{
  "name": "mysql-orders",
  "config": {
    "connector.class": "io.debezium.connector.mysql.MySqlConnector",
    "database.hostname": "mysql",
    "database.port": "3306",
    "database.user": "debezium",
    "database.password": "dbz_pw",
    "database.server.id": "1234",
    "topic.prefix": "mysql_orders",
    "database.include.list": "orders_db",
    "table.include.list": "orders_db.orders",
    "snapshot.mode": "initial",
    "include.schema.changes": "true",
    "schema.history.internal.kafka.bootstrap.servers": "kafka:9092",
    "schema.history.internal.kafka.topic": "schema.history.orders",
    "key.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter": "org.apache.kafka.connect.json.JsonConverter",
    "key.converter.schemas.enable": "true",
    "value.converter.schemas.enable": "true"
  }
}
```

几个关键字段要理解：

| 配置项 | 作用 | 常见错误 |
| --- | --- | --- |
| `connector.class` | 指定连接器类型 | 类名写错导致 connector 无法启动 |
| `database.server.id` | MySQL 复制身份，必须唯一 | 与现有复制实例冲突 |
| `topic.prefix` | 事件主题前缀，也是源端标识 | 多源场景前缀设计混乱 |
| `snapshot.mode` | 初次启动是否做快照 | 大表直接全量快照压垮源库 |
| `schema.history.internal.kafka.topic` | 记录表结构历史 | topic 未创建或权限不足 |
| `table.include.list` | 指定要接入的表 | 漏配后以为“CDC 没生效” |

真实工程里，注册通常通过 Kafka Connect REST API：

```bash
curl -X POST http://connect:8083/connectors \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "mysql-orders",
    "config": {
      "connector.class": "io.debezium.connector.mysql.MySqlConnector",
      "database.hostname": "mysql",
      "database.port": "3306",
      "database.user": "debezium",
      "database.password": "dbz_pw",
      "database.server.id": "1234",
      "topic.prefix": "mysql_orders",
      "database.include.list": "orders_db",
      "table.include.list": "orders_db.orders",
      "snapshot.mode": "initial",
      "include.schema.changes": "true",
      "schema.history.internal.kafka.bootstrap.servers": "kafka:9092",
      "schema.history.internal.kafka.topic": "schema.history.orders"
    }
  }'
```

如果要先验证 Kafka Connect 是否正常，再注册 connector，最小检查命令通常是：

```bash
curl http://connect:8083/connectors
```

返回 `[]` 表示服务正常但还没有已注册 connector；返回连接错误则说明先别排查 Debezium 配置，应该先排查 Connect 服务本身。

接下来给一个可运行的 Python 玩具实现。它不依赖 Kafka，只模拟 Debezium 事件的应用逻辑，重点演示四件事：

1. `c/u/d/r` 事件怎么还原状态
2. 幂等处理怎么避免重复写
3. 如何计算端到端延迟
4. 如何用版本号避免旧事件覆盖新事件

```python
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Event:
    op: str                   # c/u/d/r
    pk: int
    before: Optional[dict]
    after: Optional[dict]
    commit_ts_ms: int
    process_ts_ms: int
    source_event_id: str      # 幂等去重
    version: int              # 防止乱序覆盖


def apply_events(events: List[Event]):
    state: Dict[int, dict] = {}
    versions: Dict[int, int] = {}
    seen = set()
    lags = []

    for e in events:
        if e.source_event_id in seen:
            continue
        seen.add(e.source_event_id)

        lag = e.process_ts_ms - e.commit_ts_ms
        if lag < 0:
            raise ValueError(f"negative lag for event {e.source_event_id}")
        lags.append(lag)

        current_version = versions.get(e.pk, -1)
        if e.version < current_version:
            continue

        if e.op in ("c", "r"):
            if e.after is None:
                raise ValueError(f"{e.op} event requires after")
            state[e.pk] = dict(e.after)
            versions[e.pk] = e.version
        elif e.op == "u":
            if e.after is None:
                raise ValueError("u event requires after")
            current = dict(state.get(e.pk, {}))
            current.update(e.after)
            state[e.pk] = current
            versions[e.pk] = e.version
        elif e.op == "d":
            state.pop(e.pk, None)
            versions[e.pk] = e.version
        else:
            raise ValueError(f"unsupported op: {e.op}")

    metrics = {
        "count": len(lags),
        "max_lag_ms": max(lags) if lags else 0,
        "avg_lag_ms": mean(lags) if lags else 0.0,
    }
    return state, metrics


def main():
    events = [
        Event("r", 1, None, {"id": 1, "name": "alice", "status": "active"}, 1000, 1080, "e0", 1),
        Event("u", 1, {"id": 1, "name": "alice", "status": "active"}, {"status": "suspended"}, 2000, 2150, "e1", 2),
        Event("c", 2, None, {"id": 2, "name": "bob", "status": "active"}, 3000, 3090, "e2", 1),
        Event("d", 2, {"id": 2, "name": "bob", "status": "active"}, None, 4000, 4200, "e3", 2),
        Event("u", 1, {"id": 1, "name": "alice", "status": "active"}, {"status": "active"}, 1500, 2300, "late-old", 1),
        Event("u", 1, {"id": 1, "name": "alice", "status": "active"}, {"status": "suspended"}, 2000, 2150, "e1", 2),
    ]

    state, metrics = apply_events(events)

    assert state == {
        1: {"id": 1, "name": "alice", "status": "suspended"}
    }
    assert metrics["count"] == 5
    assert metrics["max_lag_ms"] == 800
    print("final_state =", state)
    print("metrics =", metrics)


if __name__ == "__main__":
    main()
```

运行输出应当类似：

```text
final_state = {1: {'id': 1, 'name': 'alice', 'status': 'suspended'}}
metrics = {'count': 5, 'max_lag_ms': 800, 'avg_lag_ms': 254}
```

这个玩具例子对应真实系统里的三个基本事实：

| 事实 | 白话解释 | 对下游的要求 |
| --- | --- | --- |
| at-least-once | 系统宁可重复发，也不会轻易漏发 | 必须做幂等 |
| partition order only | 同一分区通常有序，不代表全局有序 | 合并时要定义排序依据 |
| delete is first-class event | 删除不是“缺一行”，而是“有一条删除事件” | 必须显式处理 `op=d` |

真实工程例子可以看实时数仓。

业务库 `orders`、`payments`、`shipments` 三张表分别由不同服务维护。  
你希望在数仓里得到一张实时订单宽表 `dwd_order_realtime`，用于看支付转化和履约时延。

做法通常是：

1. 每个源库用独立 Debezium connector。
2. Kafka 中形成多个主题，例如 `mysql_orders.orders_db.orders`、`mysql_payments.pay_db.payments`。
3. Flink、Kafka Streams 或自写消费者按订单主键合并流。
4. Sink 到 OLAP 或湖仓时使用 UPSERT，而不是 append-only 盲写。
5. 用 Schema Registry 和 schema history topic 处理字段新增与类型变更。

这类架构的收益很直接，但收益不是“天然获得”的，而是建立在幂等、主键设计、延迟监控和变更治理都做对的前提上。真正可复用的结论不是某个团队拿到了多少吞吐，而是下面这张工程映射表：

| 需求 | CDC 链路上的实现点 |
| --- | --- |
| 秒级搜索更新 | 读日志 + 消费端 UPSERT 索引 |
| 实时宽表 | 多主题按主键 join + 版本控制 |
| 删除同步 | 显式消费 `d` 事件 |
| 历史追踪 | 保留 `before/after/source` |
| 低耦合多下游 | Kafka 作为事件总线 |

---

## 工程权衡与常见坑

CDC 最大的误区是“接上 Kafka 就算完成”。实际上，大多数故障都出在链路边缘，而不是 connector 本身。

先看常见坑：

| 坑 | 现象 | 根因 | 规避措施 |
| --- | --- | --- | --- |
| 重复事件 | 重启后下游重复写入 | at-least-once 语义 | 以主键做 UPSERT，记录事件唯一 ID |
| 删除丢失 | 搜索或数仓里旧数据删不掉 | 只消费 `after`，忽略 `op=d` | 显式处理删除事件 |
| Schema Drift | 消费端反序列化失败 | 源库 DDL 与下游不兼容 | 统一兼容策略，变更前预演 |
| 初始快照太重 | 启动时拖慢源库 | 大表在高峰期做快照 | 低峰执行，必要时增量快照 |
| 多源写回环 | A 改 B，B 又触发 A | 缺少来源隔离 | 按源端标识隔离，做来源过滤 |
| 监控不足 | 延迟升高却不知道卡在哪 | 只看“消息有没有到” | 分段监控 DB、Debezium、Consumer |
| 主键设计不稳 | 宽表反复插入新行 | 业务主键不唯一或会变 | 统一稳定主键 |
| 乱序覆盖 | 旧状态把新状态覆盖掉 | 多分区、多源或重试导致乱序 | 引入版本号或时间戳比较 |

### 1. 幂等不是可选项

如果下游是 Elasticsearch、Redis、数仓宽表，只要写入逻辑不是幂等，就一定会被重复事件放大。  
最常见做法有两种：

1. 按主键直接 UPSERT。
2. 保存 `source` 中的事务位点或事件唯一标识，重复则丢弃。

如果还要解决乱序，通常再加一个版本字段，例如 `updated_at`、`source.ts_ms` 或业务侧递增版本号，只接受“更新版本更大”的事件。

可以把消费规则写成一个简单判定式：

$$
\text{accept}(e)=
\begin{cases}
1, & \text{if } version(e) \ge version(current) \\
0, & \text{otherwise}
\end{cases}
$$

它不完美，但足够表达一个工程事实：去重只解决“重复”，不解决“乱序”；乱序需要版本比较。

### 2. Schema 演进最怕“新增非空字段”

这类变更在业务开发眼里很普通，但在 CDC 下游可能直接炸掉。  
例如源表新增 `country_code VARCHAR NOT NULL`，但历史事件和旧消费者都没有这个字段。结果可能是：

- 旧消费者反序列化失败
- 下游写库因非空约束报错
- 宽表补数逻辑出现空洞

稳妥做法通常是：

1. 先新增可空字段或带默认值字段。
2. 等所有下游完成适配。
3. 再逐步收紧约束。

这类变更最好按“兼容窗口”执行：

| 阶段 | 源表动作 | 消费端动作 |
| --- | --- | --- |
| 第 1 阶段 | 新增可空字段 | 忽略未知字段也能正常运行 |
| 第 2 阶段 | 开始写入新字段 | 下游完成适配并回填 |
| 第 3 阶段 | 校验覆盖率 | 观察空值、默认值和错误率 |
| 第 4 阶段 | 收紧约束 | 再改成非空或去掉旧字段 |

### 3. 多源合并没有“自动正确”这一说

多源 CDC 的难点不是把事件读出来，而是合并规则。  
如果两个系统都维护用户资料，一个改昵称，一个改手机号，你必须先定义冲突策略：

| 冲突策略 | 含义 | 适用场景 | 风险 |
| --- | --- | --- | --- |
| Last Write Wins | 时间新的覆盖旧的 | 简单同步，允许最终覆盖 | 时钟不一致会误判 |
| Source Priority | 指定某源优先级更高 | 主数据系统明确 | 低优先级更新可能永远失效 |
| Field Ownership | 字段归属不同系统 | 微服务边界清晰 | 需要长期维护字段边界 |
| Manual Reconcile | 冲突进入人工队列 | 高价值、低频冲突 | 成本高，吞吐低 |

没有规则就直接 merge，结果往往不是“偶尔错一点”，而是整个下游口径不稳定。

新手最容易忽略的一点是：多源冲突不是技术细节，而是业务规则。技术系统只能执行规则，不能替你发明规则。

---

## 替代方案与适用边界

Debezium 很强，但不是所有团队都应该第一天就上 Debezium + Kafka Connect + 完整监控栈。

先看适用边界：

| 场景 | 是否适合 Debezium | 原因 |
| --- | --- | --- |
| 已有 Kafka 基础设施，追求低延迟、多下游复用 | 很适合 | 事件总线和连接器体系已经具备 |
| 需要实时数仓、搜索、缓存联动 | 很适合 | 同一份变更可复用到多个下游 |
| 只有一两张表，目标只是同步到 BI | 可能过重 | 轮询或轻量同步可能成本更低 |
| 团队缺少 Kafka/Connect 运维能力 | 需要谨慎 | 故障定位和容量管理成本高 |
| 需要大规模多源并发接入 | 很适合 | Debezium 连接器生态成熟 |

替代方案主要有三类。

### 1. 轮询增量抽取

最简单，直接扫 `updated_at`。  
优点是实现快、依赖少。  
缺点是删除难处理、延迟高、扫表重、容易漏边界。

适合：早期验证、小数据量、容忍分钟级延迟。

补一条判断经验：如果你只需要每天一次同步，或者只是把一两张表导入 BI，轮询往往已经够用。不要为了“技术上更先进”把系统复杂度抬高一个数量级。

### 2. 应用层双写或事件外发

业务代码在写数据库时，同时发一条消息。  
优点是事件定义可控。  
缺点是一致性难，代码侵入大，历史表接入成本高。

适合：新系统、强业务语义事件，而不是想回收旧数据库变更。

这里要分清两种事件：

| 类型 | 例子 | 是否等同于 CDC |
| --- | --- | --- |
| 数据变更事件 | “orders 表第 123 行状态变了” | 是或接近 |
| 业务语义事件 | “订单已支付” | 不等同于 CDC |

如果你需要的是“业务语义事件”，应用层外发通常比纯 CDC 更直接。

### 3. UI-first CDC 平台

例如一些托管式或可视化 CDC 工具。  
优点是上手快、运维少。  
缺点是灵活性、可控性、成本模型可能不如自建。

| 维度 | Debezium + Kafka Connect | UI-first CDC |
| --- | --- | --- |
| 基础设施 | 需要 Kafka、Connect 等组件 | 平台封装较多 |
| 灵活性 | 高，可深度定制 | 中等，受平台能力约束 |
| 运维复杂度 | 高 | 低到中 |
| 排障深度 | 深，可看位点和连接器细节 | 取决于平台暴露能力 |
| 适合团队 | 有数据平台能力的团队 | 想快速上线的小团队 |

一个常见且合理的落地策略是“批量 + 增量混合”：

1. 先用传统 ETL 把历史全量装进数仓。
2. 再用 CDC 接增量。
3. 下游统一做 UPSERT 或 merge。

这样可以避免让 CDC 承担“回灌全部历史”这个不擅长的任务，也能让实时链路更干净。

最后给一个选择建议表，帮助新手快速判断：

| 你的约束 | 更优先考虑 |
| --- | --- |
| 只有少量表，延迟不敏感 | 轮询增量 |
| 需要业务语义事件 | 应用层事件外发 |
| 需要低延迟、多下游复用、删除同步 | Debezium CDC |
| 不想自运维太多组件 | UI-first CDC 平台 |
| 既要历史全量又要实时增量 | 批量 + CDC 混合 |

---

## 参考资料

- Debezium Documentation: MySQL Connector  
  核心贡献：解释 MySQL binlog 读取、快照模式、schema history、事件结构。  
  建议阅读方式：先看 connector 配置，再看 event payload。  
  https://debezium.io/documentation/reference/stable/connectors/mysql.html

- Debezium Documentation: PostgreSQL Connector  
  核心贡献：解释 WAL、logical decoding、slot、publication 等 PostgreSQL 侧前置概念。  
  建议阅读方式：重点看 replication slot 和 publication 的要求。  
  https://debezium.io/documentation/reference/stable/connectors/postgresql.html

- Debezium Documentation: Event Changes 与 Message Structure  
  核心贡献：说明 `before`、`after`、`op`、`source`、`ts_ms` 等字段含义。  
  建议阅读方式：把字段定义和自己的消费逻辑一一对照。  
  https://debezium.io/documentation/reference/stable/transformations/event-changes.html

- Apache Kafka Documentation: Kafka Connect  
  核心贡献：解释 Connect worker、connector、task、offset 管理与 REST API。  
  建议阅读方式：先看 Connect 架构，再看 connector 生命周期。  
  https://kafka.apache.org/documentation/#connect

- MySQL Reference Manual: The Binary Log  
  核心贡献：解释 binlog 是什么、记录粒度是什么、为什么能用于复制和恢复。  
  建议阅读方式：重点看 row-based logging 与 replication 相关部分。  
  https://dev.mysql.com/doc/refman/8.0/en/binary-log.html

- PostgreSQL Documentation: Logical Decoding Concepts  
  核心贡献：解释 WAL 如何被逻辑解码、slot 如何保证消费位置。  
  建议阅读方式：先理解 replication slot，再理解输出插件。  
  https://www.postgresql.org/docs/current/logicaldecoding-explanation.html

- Confluent Schema Registry Documentation  
  核心贡献：说明 schema 兼容策略，例如 backward、forward、full。  
  建议阅读方式：对照自己的“新增字段、删字段、改类型”流程看兼容级别。  
  https://docs.confluent.io/platform/current/schema-registry/index.html

- Martin Kleppmann: Turning the Database Inside Out  
  核心贡献：从系统设计角度说明为什么数据库日志可以成为数据流系统的事实来源。  
  建议阅读方式：把它当作架构层解释，不要把它当作具体操作手册。  
  https://www.confluent.io/blog/turning-the-database-inside-out-with-apache-samza/
