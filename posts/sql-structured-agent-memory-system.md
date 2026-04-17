## 核心结论

基于 SQL 的结构化 Agent 记忆系统，适合存储“字段明确、查询条件明确、需要精确命中”的记忆，而不是所有记忆都交给向量检索。这里的“结构化”意思是：数据能拆成确定的列，例如 `user_id`、`type`、`status`、`project_id`、`created_at`，查询时可以直接写条件过滤。

这类记忆最典型的是三种：

1. 用户偏好，例如语言、地区、通知方式、订阅级别。
2. 任务状态，例如当前处于 `planning`、`deploying`、`failed` 哪个阶段。
3. 配置参数，例如超时时间、模型名、权限开关、重试次数。

原因很直接。向量检索本质是“相似度匹配”，也就是找“看起来像”的内容；SQL 本质是“条件匹配”，也就是找“满足这些字段条件”的内容。对于“上次给 42 号用户保存的 timeout 是多少”这种问题，后者更符合问题结构。

在精确实体、数值、排序、聚合这些任务上，SQL 方案通常比纯向量方案更可靠。TAG/BIRD 相关实验显示，结构化查询在精确匹配上的准确率可稳定在 40% 以上，而一些向量或检索基线常落在 0% 到 10%。用公式写就是：

$$
\Delta_{\text{exact}} = \text{ExactMatch}_{SQL} - \text{ExactMatch}_{Vector} \approx 30\%\sim 40\%
$$

如果把 Agent 记忆拆开看，可以得到一个简单结论：

| 记忆类型 | 推荐存储 | 原因 |
| --- | --- | --- |
| 用户偏好、配置、状态、ID 关联 | SQL 表 + JSON 列 | 需要精确过滤、审计、更新 |
| 长文本经验、模糊语义线索 | 向量库 | 需要相似语义召回 |
| 同时有字段过滤和语义搜索 | SQL + 向量混合 | 先缩小范围，再做语义匹配 |

从工程角度看，SQLite 作为嵌入式数据库尤其适合单机 Agent、本地工具 Agent、桌面端助手、轻量服务。它没有额外进程，部署简单，事务完整，还支持 JSON1 扩展来处理半结构化字段。对于“读一条状态”“按用户取最新偏好”这类查询，延迟通常低于 1ms 量级，足够实时注入上下文。

---

## 问题定义与边界

“基于 SQL 的结构化 Agent 记忆系统”不是指把所有上下文都塞进数据库，而是指把那些能够稳定抽取字段的事实性记忆保存下来，并通过 SQL 精确查询。这里的“事实性记忆”可以理解为“有确定答案、不是靠语义猜测的记忆”。

例如下面这些问题都属于结构化记忆问题：

- 用户 123 上次选择的部署区域是什么？
- 当前会话 89 的任务状态是否还在 `deploying`？
- 项目 A 的默认重试次数是几次？
- 过去 24 小时内，这个用户是否已经授权某个动作？

这些问题有共同点：都能写成 `WHERE` 条件。也就是说，查询目标不是“找语义接近的文本”，而是“找满足条件的那一条或那几条记录”。

一个新手容易混淆的点是：Agent 记忆不等于聊天历史。聊天历史是原始语料，结构化记忆是从语料中提取出的稳定事实。比如用户说“以后都用中文回复”，原始聊天是一段文本，结构化记忆则可能是：

- `user_id=123`
- `type='preference'`
- `key='language'`
- `value='zh-CN'`

玩具例子可以这样看。

用户第一次对 Agent 说：

> 以后部署默认区域改成 `cn-east`，超时时间设成 30 秒。

如果只把这句话存成向量，下一次检索时，系统要依赖 embedding 是否把“部署区域”“默认区域”“超时时间”这些词正确拉近；如果直接拆成结构化字段，就能明确写成：

- `region = 'cn-east'`
- `timeout = 30`

然后查询：

```sql
SELECT json_extract(params, '$.region'), json_extract(params, '$.timeout')
FROM memories
WHERE user_id = 42 AND type = 'preference'
ORDER BY created_at DESC
LIMIT 1;
```

这就是问题边界。它适合“字段明确”的场景，不适合纯语义模糊场景。下面这个表可以直接区分：

| 维度 | SQL 结构化记忆适合 | SQL 结构化记忆不适合 |
| --- | --- | --- |
| 实体/数值 | 用户 ID、金额、状态、时间、配置 | 仅有自然语言描述、无固定字段 |
| 查询方式 | 精确过滤、排序、聚合、去重 | 纯语义相似、开放式联想 |
| 可审计性 | 需要知道“哪条记录命中了” | 只关心“语义像不像” |
| 更新方式 | 需要事务、覆盖、版本控制 | 只做文档追加和召回 |
| 成本结构 | 希望低成本本地部署 | 已经为大规模语义检索建好基础设施 |

所以，结构化 Agent 记忆的边界不是“替代向量”，而是“把本来就该用数据库做的部分拿回来”。如果问题天然是关系型、条件型、状态型，就不应该强行先 embedding 再检索。

---

## 核心机制与推导

核心机制只有一句话：把记忆表示成“可过滤的事实”，再让数据库做它最擅长的事。

### 1. 记忆模型：从文本变成事实行

最小记忆单元通常包含以下字段：

| 字段 | 含义 | 例子 |
| --- | --- | --- |
| `user_id` | 记忆归属用户 | `42` |
| `session_id` | 当前会话或任务流 | `89` |
| `type` | 记忆类型 | `preference` / `task_state` |
| `status` | 当前状态 | `deploying` |
| `params` | 半结构化参数 | `{"timeout":30,"region":"cn-east"}` |
| `created_at` | 发生时间 | `2026-03-19 10:00:00` |

这里的“半结构化”意思是：主干字段固定，但细节参数可能变化，所以放进 JSON。SQLite 的 JSON1 扩展允许你对 JSON 做读写，比如 `json_extract` 读取、`json_patch` 合并更新，这样不需要为了每个小参数都加一列。

### 2. 精确匹配为什么比向量更稳

向量检索的流程通常是：

1. 把文本转成 embedding。
2. 用相似度搜索邻近向量。
3. 从召回结果里再让模型挑答案。

误差可能出现在每一层：文本表达不稳定、embedding 丢失细粒度数值、召回排序不准、最终生成再误读。

SQL 查询则不同。只要字段设计正确，查询本身是确定性的。例如：

```sql
SELECT *
FROM memories
WHERE user_id = 42
  AND type = 'preference'
ORDER BY created_at DESC
LIMIT 1;
```

这个查询不会因为“偏好”和“习惯”语义接近就召回别人的记录，也不会因为“30 秒”和“半分钟”嵌入距离略有偏差而错拿一条近似结果。

因此，SQL 方案的精度优势并不神秘，它来自少了一层“近似相似度”误差。对精确任务来说，可以把检索误差简化成：

$$
P(\text{错误}) = P(\text{字段设计错误}) + P(\text{写入错误}) + P(\text{查询条件错误})
$$

而向量方案常常还要加上：

$$
+ P(\text{embedding 表示损失}) + P(\text{近邻召回错误}) + P(\text{重排错误})
$$

链路越长，确定性越弱。

### 3. SQLite 为什么够用

SQLite 是嵌入式数据库，也就是数据库引擎直接作为库嵌入应用，不需要单独启动数据库服务。对 Agent 来说，这有几个直接好处：

- 本地可运行，部署简单。
- 单文件数据库，便于备份和迁移。
- 支持事务，能保证多步写入一致。
- 支持索引，适合高频点查。
- 支持 JSON1，适合“固定骨架 + 灵活参数”的记忆模式。

对于大多数“一个用户一条最近状态”“按项目取配置”“按时间窗口过滤事件”的查询，复杂度主要取决于索引是否命中。若为 `(user_id, type, created_at)` 建复合索引，那么最新偏好读取通常就是一次非常短的 B-Tree 路径查找。直观上可以理解为：

$$
T_{query} \approx O(\log N)
$$

而且 `LIMIT 1` 会让数据库尽早停止扫描。对于 10 万到 100 万条量级的本地结构化记忆，这种模式通常足够快。

### 4. 一个玩具例子

假设只有两条记录：

| id | user_id | type | params |
| --- | --- | --- | --- |
| 1 | 42 | preference | `{"language":"zh-CN","region":"cn-east"}` |
| 2 | 43 | preference | `{"language":"en-US","region":"us-west"}` |

用户 42 再次进入系统时，Agent 需要知道回复语言。用 SQL 直接查：

```sql
SELECT json_extract(params, '$.language')
FROM memories
WHERE user_id = 42 AND type = 'preference'
LIMIT 1;
```

结果是唯一且确定的。如果改用向量检索，系统要把“上次我的语言偏好是什么”变成 embedding，再从两条文本描述中找相似项。数据量小时看不出差异，规模一大，错误来源会明显增加。

### 5. 一个真实工程例子

真实工程里更常见的不是“语言偏好”，而是“任务状态回忆”。例如一个 DevOps Agent 帮用户部署服务。一次部署过程跨多轮对话、多个工具调用，关键状态可能包括：

- 当前环境：`prod`
- 当前阶段：`deploying`
- 镜像版本：`v1.8.2`
- 超时：`30`
- 重试上限：`2`
- 回滚开关：`true`

这时最有价值的记忆不是一段自然语言摘要，而是一条可恢复执行状态的结构化记录。系统重启后，只要查到这条记录，就能判断下一步该继续等待、重试还是回滚。这个场景里，SQL 的事务、一致性、时间排序、条件过滤都比向量库更贴近需求。

---

## 代码实现

下面给出一个最小可运行实现。它使用 Python 标准库里的 `sqlite3`，不需要第三方依赖。代码演示三件事：

1. 建表和索引。
2. 写入用户偏好与任务状态。
3. 精确读取最新状态，并用 `assert` 验证结果。

```python
import sqlite3
import json
from datetime import datetime, timezone

def utc_now():
    return datetime.now(timezone.utc).isoformat()

conn = sqlite3.connect(":memory:")
conn.execute("""
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_id INTEGER,
    project_id TEXT,
    type TEXT NOT NULL,
    status TEXT,
    params TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
)
""")

conn.execute("""
CREATE INDEX idx_memories_user_type_time
ON memories(user_id, type, created_at DESC)
""")

conn.execute("""
CREATE INDEX idx_memories_session_status_time
ON memories(session_id, status, created_at DESC)
""")

def write_memory(user_id, session_id, project_id, mem_type, status, params):
    conn.execute(
        """
        INSERT INTO memories(user_id, session_id, project_id, type, status, params, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            session_id,
            project_id,
            mem_type,
            status,
            json.dumps(params, ensure_ascii=True),
            utc_now(),
        ),
    )
    conn.commit()

def latest_preference(user_id):
    row = conn.execute(
        """
        SELECT params
        FROM memories
        WHERE user_id = ? AND type = 'preference'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (user_id,),
    ).fetchone()
    return json.loads(row[0]) if row else None

def latest_timeout(session_id):
    row = conn.execute(
        """
        SELECT json_extract(params, '$.timeout')
        FROM memories
        WHERE session_id = ? AND status = 'deploying'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    return row[0] if row else None

# 写入两位用户的偏好
write_memory(
    user_id=42,
    session_id=1001,
    project_id="proj-a",
    mem_type="preference",
    status="active",
    params={"language": "zh-CN", "region": "cn-east"}
)

write_memory(
    user_id=43,
    session_id=1002,
    project_id="proj-b",
    mem_type="preference",
    status="active",
    params={"language": "en-US", "region": "us-west"}
)

# 写入部署状态
write_memory(
    user_id=42,
    session_id=89,
    project_id="proj-a",
    mem_type="task_state",
    status="deploying",
    params={"timeout": 30, "model": "gpt-4.1", "retry_limit": 2}
)

pref = latest_preference(42)
timeout = latest_timeout(89)

assert pref["language"] == "zh-CN"
assert pref["region"] == "cn-east"
assert timeout == 30

print("all assertions passed")
```

这段代码的关键点不是 Python，而是存储模型：

- 固定筛选字段单独成列：`user_id`、`session_id`、`type`、`status`
- 易变参数放进 JSON：`params`
- 所有读取都显式带过滤条件和排序条件

如果换成 SQL 视角，最常见的读写语句通常就是下面几类：

```sql
-- 写入一条偏好
INSERT INTO memories(user_id, type, status, params, created_at)
VALUES (
  42,
  'preference',
  'active',
  json_object('language', 'zh-CN', 'region', 'cn-east'),
  CURRENT_TIMESTAMP
);

-- 读取某用户最新偏好
SELECT *
FROM memories
WHERE user_id = 42 AND type = 'preference'
ORDER BY created_at DESC
LIMIT 1;

-- 读取部署中的 timeout
SELECT json_extract(params, '$.timeout') AS timeout
FROM memories
WHERE session_id = 89 AND status = 'deploying'
ORDER BY created_at DESC
LIMIT 1;
```

真实工程里还会再加两类能力。

第一类是“幂等写入”。“幂等”意思是同一个操作重复执行，结果不应重复脏写。例如同一个 `session_id + step_id` 只能有一条当前状态，可用唯一索引或 `UPSERT` 处理。

第二类是“版本回放”。也就是不覆盖旧值，而是保留事件流。因为 Agent 出错时，排查往往需要回答两个问题：

- 它当时看到了什么记忆？
- 它为什么根据那条记忆做出了这个动作？

SQL 天然适合回答这两个问题，因为每次写入都能带时间戳、事务边界和可复现查询条件。

---

## 工程权衡与常见坑

结构化记忆不是“建一张表就结束”，真正的工程质量取决于写入规范和查询纪律。下面是最常见的坑。

| 坑 | 典型表现 | 规避手段 |
| --- | --- | --- |
| 垃圾输入直接入库 | 同一个概念写成多个字段名，后续无法稳定查询 | 先做字段标准化和枚举约束 |
| 忽略元数据过滤 | 查偏好时漏掉 `user_id`，导致跨用户串记忆 | 强制带 `user_id/project_id/session_id` |
| JSON 乱长 | 所有内容都塞进 `params`，最后无法建索引 | 高频过滤字段必须提升为独立列 |
| 覆盖式更新过多 | 只保留最新值，排障时看不到历史 | 关键状态保留事件日志 |
| 提前引入向量层 | 本来是精确匹配问题，却增加 embedding 成本 | 先 SQL，确认需要模糊语义再补向量 |
| 缺少生命周期策略 | 旧任务状态长期堆积，影响维护 | 设置 TTL、归档、冷热分层 |

最容易出事故的是第二类：漏过滤条件。

例如你写了：

```sql
SELECT *
FROM memories
WHERE type = 'preference'
ORDER BY created_at DESC
LIMIT 1;
```

这条 SQL 在语法上完全正确，但在业务上几乎一定有问题。因为“最新偏好”是谁的偏好？如果没加 `user_id`，系统可能把 A 用户的地区偏好注入给 B 用户。这个 bug 比“检索不到”更危险，因为它会表现为“检索到了错误但看似合理的答案”。

另一个常见坑是把所有动态字段都扔进 JSON。JSON 很灵活，但灵活不等于随意。判断标准很简单：如果一个字段会频繁出现在 `WHERE`、`ORDER BY`、`JOIN` 里，就不该只存在于 JSON 里，而应该提升为独立列。比如：

- `status`
- `project_id`
- `user_id`
- `created_at`

这些都应是显式列，而不是 `params.status`、`params.project_id`。

真实工程例子可以看一个客服 Agent。它要记住用户订阅级别、退款资格、最近工单状态。如果你把这些信息全部以文本摘要方式存储，系统每次都要重新理解一遍“用户是否可退款”；如果把它们结构化为：

- `subscription_tier`
- `refund_eligible`
- `ticket_status`

那么权限判断就是普通 SQL 条件判断，速度快，逻辑可审计，也便于法务和风控复查。

最后要注意“Garbage In, Garbage Out”。这句话的意思是：输入是错的，系统只会更稳定地复用错误。SQL 不会自动修正脏数据，它只是更快地命中脏数据。所以写入前的清洗、归一化、字段约束是结构化记忆系统的一部分，不是外围装饰。

---

## 替代方案与适用边界

SQL 不是万能记忆层，它解决的是“可结构化事实”的问题。下面是更实用的选型表：

| 场景 | 首选方案 | 原因 |
| --- | --- | --- |
| 用户偏好、任务状态、配置参数 | SQL | 需要精确命中、事务、审计 |
| 长文档经验回忆、开放式问答 | 向量检索 | 需要语义近邻召回 |
| 图结构关系推理 | 图数据库 | 关注节点关系和路径 |
| 混合问答 | SQL + 向量 | 先过滤实体，再补语义召回 |

什么时候必须引入替代方案？

第一种情况是纯语义查询。比如“用户过去提过哪些和性能焦虑相关的担忧”，这里“性能焦虑”不是稳定字段，而是一种语义主题，用向量检索更自然。

第二种情况是多模态记忆。图片、音频、界面截图这类内容无法直接落在传统关系表的字段过滤上，通常需要 embedding 或专用索引。

第三种情况是超大规模语义召回。如果系统目标是跨百万文档做相关片段搜索，SQL 不是最优语义检索工具，即使它可以做 LIKE、FTS，也未必适合复杂语义搜索。

但要注意，很多系统误把“有文本”当成“必须向量化”。实际上，一个问题是否适合 SQL，不看输入是不是文本，而看输出是不是精确事实。

例如：

- “列出上次购买金额最大的三个项目”是 SQL 问题，因为它要求排序和数值比较。
- “找几段和部署失败原因语义接近的历史经验”才更像向量问题。

前者可以直接写：

```sql
SELECT item_name, amount
FROM purchases
WHERE user_id = 42
ORDER BY amount DESC
LIMIT 3;
```

这类查询如果交给纯向量 RAG，常见问题不是“慢一点”，而是“根本答错”。因为排序、聚合、实体对齐不是向量相似度擅长的任务。

所以更稳妥的架构通常不是“二选一”，而是分层：

1. 结构化事实进 SQL。
2. 非结构化经验进向量库。
3. 查询时先判断问题类型。
4. 若需要混合，先用 SQL 缩小候选集，再做语义召回。

这种分层的核心价值不是“更先进”，而是让每种存储系统处理自己擅长的问题。

---

## 参考资料

- Memori, “Why Use SQL Databases for AI Agent Memory”, 讨论结构化记忆的优势、成本对比与可维护性。https://memorilabs.ai/blog/why-use-sql-databases-for-ai-agent-memory/
- Biswal et al., “Text2SQL is Not Enough: Unifying AI and Databases with TAG”, CIDR 2025，讨论 SQL 形态任务与检索基线在精确匹配上的差异。https://www.vldb.org/cidrdb/papers/2025/p11-biswal.pdf
- SQLite 官方文档，JSON1 Extension，说明 `json_extract`、`json_patch` 等 JSON 读写能力。https://www.sqlite.org/json1.html
- Sparkco, “Persistent Memory for AI Agents: Comparing PAG, Memory.md and SQLite Approaches”, 给出 SQLite 持久化记忆的工程案例与延迟讨论。https://sparkco.ai/blog/persistent-memory-for-ai-agents-comparing-pag-memorymd-and-sqlite-approaches
- BSWEN, “AI Agent Memory Storage: SQL vs Vector”, 讨论 metadata 过滤、脏数据与选型误区。https://docs.bswen.com/blog/2026-03-06-agent-memory-storage/
