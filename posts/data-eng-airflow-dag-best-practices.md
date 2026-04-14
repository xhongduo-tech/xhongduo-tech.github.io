## 核心结论

Airflow 的 DAG 编排，核心不是“把任务连起来”，而是把一条数据管线拆成一组**可重试、可观测、可限流、可演进**的任务节点。对初级工程师来说，最重要的实践有六条：

1. 每个任务都按“数据库事务”来设计。事务的白话解释是：一次执行要么完整成功，要么失败后可以安全重来。对应到 Airflow，就是任务输出必须尽量**幂等**。幂等的白话解释是：同一个任务重复执行多次，最终结果应与执行一次一致。
2. 依赖关系要显式表达，用 `>>`、`<<` 或 `set_downstream` 清楚描述拓扑，不要把顺序关系藏进脚本内部。
3. 等待外部系统时优先用 deferrable Sensor 或 `mode="reschedule"`。否则等待中的任务会长期占住 worker 资源。
4. 动态 DAG 适合“结构相同、参数不同”的场景，但动态生成逻辑必须轻量，不能把大量 I/O、网络请求、数据库查询放在 DAG 文件顶层。
5. 重试策略不是“失败就无限重跑”，而是围绕临时故障、幂等写入、告警升级来设计。网络抖动可重试，逻辑错误通常不该盲目重试。
6. 资源管理要前置建模，用 `pool`、`pool_slots`、`queue`、`priority_weight` 控制谁先跑、谁少跑、谁不能并发跑。

可以把调度优先级先理解成一个近似公式：

$$
effective\_priority = priority\_weight + \sum priority\_weight(\text{downstream tasks})
$$

当 `weight_rule=downstream` 时，上游任务会因为“后面还有很多任务等着它”而获得更高的有效优先级。这不是业务逻辑，而是调度策略。

---

## 问题定义与边界

本文讨论的是：如何把 Airflow DAG 设计成适合生产环境长期运行的形式。重点覆盖任务粒度、依赖表达、幂等性、Sensor、动态 DAG、重试与错误处理、资源隔离、测试与 CI/CD。

先明确边界。DAG 是**调度描述**，不是把所有业务都塞进去的容器。它负责定义“何时执行、按什么依赖执行、失败后如何处理、资源如何分配”，不负责替代数据库、消息队列、流处理平台或底层计算引擎。

| 范围 | 描述 |
| --- | --- |
| 必须完成 | 幂等任务设计、显式依赖、合理 Sensor、动态配置生成、资源限制、DAG 测试与发布验证 |
| 可以涉及 | 重试参数、超时、告警、失败回调、分环境配置 |
| 不在此讨论 | 自定义调度器、替换 orchestration 平台、修改 executor 底层实现 |

一个常见误区是把“DAG 能跑起来”当作完成。生产环境里真正的问题是：重跑是否安全、失败是否可恢复、外部依赖是否会阻塞资源、多个客户或多条管线是否能稳定共存。

这里给一个玩具例子。假设你有一个最简单的三步流程：

1. 等待 `/data/orders/2026-04-14.ready`
2. 解压 `orders.zip`
3. 导入 `warehouse.orders`

如果第二步失败，第三步不能执行；如果第三步执行到一半失败，下次重跑不能重复写入同一批订单；如果第一步一直等文件到来，不能把整个 worker 卡死。这三个要求，分别对应依赖、幂等、Sensor 模式。

真实工程例子更复杂。比如金融 ETL：先等待外部 SFTP 报文，再按客户配置生成多个“清洗 -> 校验 -> 入仓”任务链，其中调用外部风控 API 的任务必须走同一个 `api_pool`，因为该 API 有供应商限流。这个 DAG 不是只求“成功”，而是要求在高峰期也能可控地失败、重试、排队和恢复。

---

## 核心机制与推导

### 1. 任务粒度为什么不能过粗也不能过细

任务粒度就是一次 task 应该做多大工作。太粗的问题是失败影响面大，任何一个小步骤报错都要重跑整大段流程；太细的问题是调度开销高、日志碎片化、依赖图失真。

经验上，一个任务应满足三点：

1. 输入边界清楚
2. 输出边界清楚
3. 失败后可独立重跑

比如“下载文件、解压、清洗、写入仓库、更新指标表”放在一个任务里，粒度过粗。更合理的是拆成多个任务，因为这些步骤失败原因不同、资源需求不同、幂等策略也不同。

### 2. 依赖表达为什么要外置

Airflow 调度的是**任务图**。如果你在一个 Python 脚本里写：

```python
download()
clean()
load()
```

那 Airflow 只看见“一个任务”，而不是三步。这样监控不到中间状态，也无法对某一步单独重试。

因此依赖应在 DAG 层显式表达，例如：

- `wait_file >> extract >> load`
- `start >> [clean_a, clean_b] >> merge`

这样做的价值不是语法整洁，而是把可视化、重试、告警、并发控制全部建立在真实拓扑之上。

### 3. 幂等性为什么是 Airflow 最核心的约束

Airflow 默认就支持失败重试和手动重跑，所以任务必须能承受“同一逻辑被重复执行”。数学上可以把幂等写成：

$$
f(f(x)) = f(x)
$$

白话解释是：把同一批输入重复处理两次，结果不能越来越错。

常见做法有三类：

| 场景 | 非幂等写法 | 幂等写法 |
| --- | --- | --- |
| 数据入库 | 直接 `INSERT` | 按业务主键 `MERGE` / `UPSERT` |
| 文件输出 | 直接覆盖不完整文件 | 先写临时文件，再原子重命名 |
| 分区计算 | 直接追加 | 先删目标分区，再重建该分区 |

这也是为什么很多团队要求“每个 Airflow task 像数据库事务”。不是说任务内部真的开数据库事务，而是要求它具备类似的可恢复边界。

### 4. Sensor 为什么容易把集群拖慢

Sensor 是“等待条件成立”的任务，比如等文件、等表、等上游系统产出。问题在于，传统 `poke` 模式会周期性检查，但任务本身一直占着 worker slot。slot 的白话解释是：一个 worker 能同时执行的名额。

如果有 100 个 DAG 都在等文件，而每个 Sensor 都长时间占 slot，那么真正该计算的任务可能反而没资源执行。于是要优先使用：

- deferrable Sensor
- 或 `mode="reschedule"`

二者的思路都是：条件未满足时，把等待状态交还给系统，而不是一直霸占 worker。

### 5. Pool、Queue、Priority 如何共同决定谁先执行

这三个概念经常混淆：

- `pool`：资源池，用来限制某类任务总并发
- `queue`：任务队列，用来把任务路由到特定 worker
- `priority_weight`：优先权重，用来决定同池竞争时谁先拿到资源

如果 `maintenance` pool 只有 2 个 slots，那么一个 `pool_slots=2` 的重任务可以独占整个池。这个机制很适合数据库维护、外部 API、GPU 任务等敏感资源。

再看优先级。默认 `weight_rule=downstream` 时，上游任务会累加下游权重，所以“堵住整条链路入口”的任务会比末端任务更容易拿到 slot。这符合调度直觉：先把主干打通，再让尾部细节慢慢收尾。

| 规则 | 含义 | 适用场景 |
| --- | --- | --- |
| `downstream` | 自身 + 所有下游权重 | 希望优先打通上游主干 |
| `upstream` | 自身 + 所有上游权重 | 希望后段任务更优先完成 |
| `absolute` | 只使用显式指定值 | 权重已人工设计好，且大 DAG 需更快解析 |

### 6. 动态 DAG 的价值与风险

动态 DAG 是指根据配置批量生成结构相似的 DAG 或任务。适合多租户、多表、多客户、多区域的数据管线。

价值很直接：减少重复代码，统一结构，改一次模板即可影响多条流程。

风险也很直接：如果 DAG 文件顶层为了生成这些对象去读远程配置、查数据库、扫对象存储，调度器每次解析 DAG 都会变慢。Airflow 的 DAG 文件会被频繁解析，所以顶层逻辑必须接近“纯配置展开”。

真实工程里，更推荐把元数据放在本地可访问配置文件中，由 DAG 顶层轻量读取，再生成任务图；而不是在 import 时请求外部服务。

---

## 代码实现

先给一个可运行的 Python 玩具代码。它不依赖 Airflow，本质是演示“幂等装载”的核心思想：同一批记录重复装载，最终表中不应出现重复主键。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Record:
    order_id: str
    amount: int
    ds: str

def load_partition(existing_rows, incoming_rows):
    """
    幂等装载示例：
    1. 先删除目标分区
    2. 再按主键去重写入
    """
    if not incoming_rows:
        return list(existing_rows)

    target_ds = incoming_rows[0].ds
    kept = [r for r in existing_rows if r.ds != target_ds]

    dedup = {}
    for row in incoming_rows:
        dedup[row.order_id] = row

    return kept + list(dedup.values())

old_rows = [
    Record("A001", 100, "2026-04-13"),
    Record("A002", 200, "2026-04-14"),
]

batch = [
    Record("A002", 200, "2026-04-14"),
    Record("A003", 300, "2026-04-14"),
    Record("A003", 300, "2026-04-14"),
]

first = load_partition(old_rows, batch)
second = load_partition(first, batch)

assert len(first) == 3
assert len(second) == 3
assert first == second
assert sorted([r.order_id for r in second]) == ["A001", "A002", "A003"]
```

这段代码要表达的不是具体 SQL 写法，而是任务设计原则：按分区重建、按业务键去重、允许安全重跑。

下面是一个更接近实际项目的 Airflow DAG 片段，展示等待外部文件、显式依赖、资源控制和重试策略：

```python
from datetime import datetime, timedelta

from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "data-platform",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

with DAG(
    dag_id="orders_pipeline",
    start_date=datetime(2026, 4, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["工程实践", "数据工程"],
) as dag:

    wait_ready = FileSensor(
        task_id="wait_ready_file",
        filepath="/data/inbox/orders/{{ ds }}/_READY",
        poke_interval=60,
        timeout=60 * 60,
        mode="reschedule",
    )

    extract_orders = BashOperator(
        task_id="extract_orders",
        bash_command="python scripts/extract_orders.py --ds {{ ds }}",
        pool="io_pool",
        priority_weight=3,
    )

    load_dw = BashOperator(
        task_id="load_dw",
        bash_command="python scripts/load_orders.py --ds {{ ds }}",
        pool="warehouse_pool",
        pool_slots=2,
        priority_weight=5,
    )

    wait_ready >> extract_orders >> load_dw
```

这个例子里有四个关键点：

1. `wait_ready` 使用 `mode="reschedule"`，避免等待期间占用 worker。
2. `extract_orders` 和 `load_dw` 分配到不同 `pool`，让 I/O 和仓库写入分开限流。
3. `load_dw` 使用更高的 `priority_weight`，表示它在资源竞争时更重要。
4. `execution_timeout` 和 `retries` 是写在任务边界上的，而不是散落在脚本内部。

真实工程例子可以进一步扩展成“按客户动态生成任务链”：

```python
CUSTOMERS = [
    {"name": "bank_a", "source_path": "/sftp/bank_a", "pool": "api_pool"},
    {"name": "bank_b", "source_path": "/sftp/bank_b", "pool": "api_pool"},
]

for cfg in CUSTOMERS:
    clean = BashOperator(
        task_id=f"clean_{cfg['name']}",
        bash_command=f"python scripts/clean.py --customer {cfg['name']}",
        pool=cfg["pool"],
        priority_weight=4,
    )
```

这里的关键不是 `for` 循环本身，而是“结构稳定、参数外置”。如果客户越来越多，这种写法比复制十几份 DAG 文件更可维护。但前提是 `CUSTOMERS` 的来源要轻量、稳定、可测试。

CI 侧最基本要做两件事：

1. 用 `DagBag` 或直接 import 验证 DAG 能被加载，没有 `import_errors`
2. 跑单元测试，验证关键函数的幂等行为、参数展开、依赖结构

对 Airflow 项目来说，CI 的价值不是跑出业务结果，而是提前发现“调度器根本读不进 DAG”的低级故障。

---

## 工程权衡与常见坑

### 1. 顶层解析过重

DAG 文件会被调度器反复解析。如果你在顶层写数据库查询、HTTP 请求、复杂循环、巨量配置展开，调度器会变慢，UI 也会受影响。顶层只应保留轻量配置加载和 DAG 定义。

### 2. Sensor 用错模式

很多初学者先写出能工作的 Sensor，再发现集群越来越慢。根因通常不是业务量，而是大量等待任务占满 slot。凡是“等待时间远大于执行时间”的场景，都应优先考虑 deferrable 或 `reschedule`。

### 3. 重试策略不区分故障类型

重试适合处理临时故障，比如网络抖动、短暂锁冲突、上游对象延迟可见。不适合处理确定性 bug，比如 SQL 语法错误、字段不存在、代码逻辑错。后者重试 10 次也不会成功，只会浪费资源并延迟告警。

### 4. 没有 pool，导致关键任务被普通任务挤压

如果所有任务都去默认池里抢资源，那么“生成日报截图”和“写核心事实表”在调度器眼里可能同样重要。这显然不合理。资源敏感任务必须有独立池，外部限流接口也必须集中管理。

### 5. 动态 DAG 生成失控

动态生成的目标是减少重复，不是制造不可理解的黑箱。一个常见坑是把 500 个客户的配置全部在一个文件里展开成极其庞大的任务图，导致解析、展示、排查都很困难。这时要重新拆分 DAG，而不是继续堆循环。

下面这张表可以直接作为排查清单：

| 常见坑 | 现象 | 规避措施 |
| --- | --- | --- |
| 顶层加载慢 | DAG 列表刷新慢、解析超时 | 顶层只做轻量配置读取，避免远程 I/O |
| Sensor 占 slot | worker 长时间忙但无实质计算 | 改用 deferrable 或 `mode="reschedule"` |
| 写入不幂等 | 重跑后重复数据、脏分区 | 用主键去重、分区重建、临时表切换 |
| 无资源隔离 | 关键任务排队严重 | 显式设置 `pool`、`pool_slots`、`priority_weight` |
| CI 缺失 | 上线后才发现 DAG import 失败 | 在 PR 阶段跑 DAG 加载测试和单元测试 |

---

## 替代方案与适用边界

Airflow 适合有清晰依赖关系、以批处理和定时调度为主的数据编排。但不是所有等待和编排问题都必须由 DAG + Sensor 解决。

第一类替代方案是**事件驱动**。如果上游系统天然会在数据就绪时发消息，比如推送到 Kafka、SQS 或 Webhook，那么直接走事件触发通常比“每分钟检查一次文件是否存在”更高效。Sensor 的本质是轮询；事件驱动的本质是通知。

第二类替代方案是**外部元数据驱动而非硬编码**。如果多个流程只有数据源、目标表、字段映射不同，那么优先把这些差异放入配置文件，再用统一模板生成 DAG，而不是人工复制几十份近似代码。

第三类替代方案是**限流优先于盲目并发**。很多团队在任务慢时先想到“多开并发”，但真实瓶颈往往是数据库连接数、外部 API 限流、对象存储吞吐或下游锁竞争。此时继续加并发只会放大失败率。更合理的方式是通过 `pool` 和 `priority_weight` 精确限制关键资源。

适用边界可以概括成下面几条：

1. 如果流程主要是定时批处理，并且依赖关系明确，Airflow 很合适。
2. 如果流程主要依赖外部事件即时触发，优先评估事件驱动方案。
3. 如果 DAG 数量很多但结构高度重复，动态 DAG 合适；如果每条流程差异巨大，强行动态化会降低可读性。
4. 如果任务本身无法做到幂等，Airflow 仍能调度，但生产可维护性会明显变差，因为重试和补跑都会变危险。

换句话说，Airflow 最擅长的是“把已经可重复执行的工作，稳定地按图执行起来”。它不负责替你修正一个本身不可重跑的任务实现。

---

## 参考资料

- Apache Airflow 官方 Best Practices：任务幂等、减少顶层复杂度、动态 DAG、测试与部署  
  https://airflow.apache.org/docs/apache-airflow/2.4.2/best-practices.html
- Apache Airflow Priority Weights：`priority_weight`、`weight_rule` 的调度行为  
  https://airflow.apache.org/docs/apache-airflow/2.4.3/concepts/priority-weight.html
- Apache Airflow Pools：`pool`、`pool_slots` 的资源限制机制  
  https://airflow.apache.org/docs/apache-airflow/2.3.0/concepts/pools.html
- Astronomer Sensors 指南：Sensor 模式、deferrable operator、Triggerer  
  https://www.astronomer.io/docs/learn/what-is-a-sensor/
- Azure Learn CI/CD pattern with Airflow：DAG 验证与发布流程示例  
  https://learn.microsoft.com/en-us/azure/data-factory/ci-cd-pattern-with-airflow
