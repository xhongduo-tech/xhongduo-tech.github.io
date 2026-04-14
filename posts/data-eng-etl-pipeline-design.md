## 核心结论

ETL 和 ELT 的区别，不在于有没有“转换”这一步，而在于**转换发生在加载前还是加载后**。ETL（Extract, Transform, Load，先转后载）是把数据在进入目标系统前先清洗、校验、脱敏；ELT（Extract, Load, Transform，先载后转）是先把原始数据放进仓库，再在仓库里按分析需求做转换。

---

先给结论：

| 维度 | ETL | ELT |
| --- | --- | --- |
| 转换时机 | 加载前 | 加载后 |
| 原始数据是否先进仓 | 通常不会完整进入分析仓 | 通常会先保留原始数据 |
| 典型目标 | 合规、稳定报表、强质量控制 | 自助分析、探索式建模、快速试验 |
| 质量控制点 | 入口处校验、脱敏、标准化 | 仓库内模型校验、作业编排、数据治理 |
| 对仓库算力依赖 | 相对较低 | 相对较高 |
| 对原始数据复用支持 | 较弱 | 较强 |

如果一句话概括：**要在落地前守住合规和质量，优先 ETL；要保留原始数据并支持反复分析，优先 ELT；真实工程里常常两者并行。**

给初学者一个直观类比。把数据管线想成厨房：

- ETL 像厨师在入口处把菜洗净、切好、去掉不能上桌的部分，再送到餐厅。
- ELT 像先把整料搬进食材库，后面哪个团队要做什么菜，再去仓库里加工。

这个类比不严格，但足够帮助理解“转换时机”这个核心差异。真正的工程判断，仍然要落在数据敏感度、仓库算力、重放能力和监控体系上。

## 问题定义与边界

一条数据管线至少有三个动作：抽取、转换、加载。抽取是把数据从源系统拿出来；转换是把原始字段整理成目标格式；加载是把结果写进目标系统。讨论 ETL vs ELT 时，真正要区分的是两个维度：

- 抽取策略 $S$：全量还是增量
- 转换位置 $T$：加载前还是加载后

形式化地写：

$$
S = \{full,\ incremental\}
$$

$$
T = \{before\_load,\ after\_load\}
$$

$$
ETL \subset S \times \{before\_load\}
$$

$$
ELT \subset S \times \{after\_load\}
$$

这里的含义很直接：ETL 和 ELT 都可以做全量，也都可以做增量；它们的分界线不是“抽多少”，而是“何时转换”。

---

边界要先说清楚，否则“该选 ETL 还是 ELT”会变成口号。

第一，**隐私与合规边界**。如果原始数据包含身份证号、手机号、交易明细等敏感信息，而且目标仓库不是所有人都该看到，那么就不能轻易把原始数据先落进去再说。这时常常必须在加载前脱敏，ETL 更合适。

第二，**访问模式边界**。如果分析团队经常提出新问题，昨天看留存，今天看漏斗，明天还要重建口径，那么保留原始数据的价值很大，ELT 更合适。

第三，**计算边界**。ELT 假设仓库本身有足够算力和治理能力，能支撑大量 SQL、模型作业、回填和重跑。仓库弱、预算紧、作业体系不成熟时，纯 ELT 容易把问题推迟到后面集中爆炸。

第四，**变更规模边界**。如果表很大但每天只变一点，增量抽取的收益非常明显。

玩具例子：一张 20GB 的账本表，每晚只有 1% 更新，也就是 0.2GB。

- 如果做全量，每晚复制 20GB。
- 如果做增量，只复制 0.2GB。
- 节省的复制量是 $20 - 0.2 = 19.8GB$。

这就是为什么“先决定全量还是增量”通常比“先争论 ETL 还是 ELT”更落地。ETL/ELT 决定转换位置，full/incremental 决定搬多少数据，两者是正交的。

在这个例子里：

- 增量 ETL：只传 0.2GB 变更，并在落地前完成脱敏和字段校验。
- 增量 ELT：同样只传 0.2GB，但先把原始变更写入仓库，后续再由分析团队在仓库中建模。

## 核心机制与推导

先看抽取层。CDC（Change Data Capture，变更数据捕获，白话说就是“只抓新增、更新、删除的变化”）和时间戳增量，是最常见的两种增量方式。

- CDC 更接近数据库真实变更，能处理更新和删除。
- 时间戳增量实现简单，但依赖源表的更新时间字段可靠。

如果你的源系统只有 `updated_at`，那就要接受一个现实：只靠时间戳不一定能准确表达删除事件，也可能被回写、时钟漂移或补数据破坏。

---

再看转换层。转换规则本质上是在定义“什么样的数据才允许进入目标层”。ETL 和 ELT 都做转换，但职责位置不同。

ETL 的机制是：

1. 从源头读出变更
2. 在中间层完成标准化、脱敏、字段映射、质量校验
3. 只把合格结果写入目标系统

ELT 的机制是：

1. 先把原始变更写入仓库
2. 在仓库里用 SQL、UDF、模型作业做清洗和聚合
3. 再把结果暴露给下游报表或应用

这里有一个关键工程要求：**转换必须可重放，并且写入必须幂等**。幂等（idempotent，白话说就是“同一批数据重复执行多次，最终结果不应变坏”）决定了重试是否安全。

如果一次任务失败后重跑，会发生什么？

- 没有幂等：重复插入，数据翻倍
- 有幂等：重复执行后，目标表仍然只保留一份正确结果

因此，增量管线至少要有两个控制点：

| 控制点 | 作用 | 典型实现 |
| --- | --- | --- |
| Checkpoint | 记录处理到哪里 | LSN、binlog offset、watermark 时间戳 |
| Idempotent Write | 保证重复执行不写脏 | Upsert、Merge、按业务主键去重 |

---

玩具例子可以继续展开。

源表日变更量 0.2GB，字段有：

| order_id | user_phone | amount | updated_at |
| --- | --- | --- | --- |
| A001 | 13800001111 | 100 | 2026-04-14 10:00:00 |

ETL 路径会在加载前先把手机号脱敏成 `138****1111`，再校验金额非负，最后写入合规库。ELT 路径则会先把这一行原始数据落进 Snowflake，再跑仓库内 SQL 生成分析表。

这两条路都能工作，但监控重点不同：

| 指标 | ETL 更关注什么 | ELT 更关注什么 |
| --- | --- | --- |
| Freshness（新鲜度，白话说就是“数据离现在有多近”） | 抽取和入口处理是否延迟 | 仓库内转换作业是否排队、失败 |
| Completeness（完整性，白话说就是“该来的数据是否都到了”） | 入口校验、字段缺失、主键重复 | 原始层到账是否完整、模型层是否漏算 |
| Validity（有效性） | 脱敏、格式校验、业务规则校验 | SQL 模型约束、指标口径一致性 |

可以把 SLO（Service Level Objective，服务等级目标，白话说就是“团队给自己定的可接受目标”）写成：

$$
freshness \le 30\ min
$$

$$
completeness \ge 99.9\%
$$

真实工程例子：金融监管平台处理交易账本时，监管要求敏感字段在落地前就脱敏，且金额、币种、交易状态要做严格校验。这种场景下，ETL 不是“老技术”，而是合规边界的一部分。与此同时，产品分析团队仍然希望保留原始埋点和日志，以便后续做漏斗分析和用户画像，于是同一 CDC 源又进入 Snowflake 走 ELT。这就是典型混合架构。

## 代码实现

下面给一个简化的 `etl_runner.py` 思路。目标不是模拟完整平台，而是把“抽取→转换→加载→更新位点”这条主线讲清楚。

```python
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class State:
    last_lsn: int


SOURCE_CHANGES = [
    {"lsn": 101, "order_id": "A001", "user_phone": "13800001111", "amount": 100},
    {"lsn": 102, "order_id": "A002", "user_phone": "13900002222", "amount": 200},
    {"lsn": 103, "order_id": "A001", "user_phone": "13800001111", "amount": 100},  # 重放
]


def extract_cdc(last_lsn: int) -> List[Dict]:
    return [row for row in SOURCE_CHANGES if row["lsn"] > last_lsn]


def mask_phone(phone: str) -> str:
    return phone[:3] + "****" + phone[-4:]


def run_transforms(changes: List[Dict]) -> List[Dict]:
    staged = []
    for row in changes:
        assert row["amount"] >= 0, "amount must be non-negative"
        staged.append(
            {
                "order_id": row["order_id"],
                "user_phone_masked": mask_phone(row["user_phone"]),
                "amount": row["amount"],
                "source_lsn": row["lsn"],
            }
        )
    return staged


def load_target(staged: List[Dict], target: Dict[str, Dict]) -> None:
    # 用业务主键做 upsert，保证幂等
    for row in staged:
        target[row["order_id"]] = row


def update_state(changes: List[Dict], state: State) -> State:
    if changes:
        state.last_lsn = max(row["lsn"] for row in changes)
    return state


def load_incremental(state: State, target: Dict[str, Dict]) -> State:
    changes = extract_cdc(state.last_lsn)
    staged = run_transforms(changes)
    load_target(staged, target)
    return update_state(changes, state)


state = State(last_lsn=100)
target = {}

state = load_incremental(state, target)

assert state.last_lsn == 103
assert len(target) == 2
assert target["A001"]["user_phone_masked"] == "138****1111"
assert target["A002"]["amount"] == 200
```

这段代码体现了三个合同：

1. 抽取层只关心“从哪个位点之后继续读”
2. 转换层只关心“把原始数据变成合格数据”
3. 加载层只关心“如何安全写入目标表”

其中最重要的一行不是脱敏，而是 Upsert。Upsert（更新或插入，白话说就是“有就更新，没有就插入”）让重试变得安全。

---

如果是 ELT，通常会把原始变更先写入 `raw_orders`，再在仓库里做转换，例如：

```sql
merge into mart_orders as t
using (
  select
    order_id,
    concat(substr(user_phone, 1, 3), '****', substr(user_phone, -4)) as user_phone_masked,
    amount,
    max(source_lsn) as source_lsn
  from raw_orders
  where amount >= 0
  group by 1, 2, 3
) as s
on t.order_id = s.order_id
when matched then update set
  user_phone_masked = s.user_phone_masked,
  amount = s.amount,
  source_lsn = s.source_lsn
when not matched then insert (
  order_id, user_phone_masked, amount, source_lsn
) values (
  s.order_id, s.user_phone_masked, s.amount, s.source_lsn
);
```

这就是“先 load，再 transform”的典型写法。对于新手，可以把它理解为三句话：

- 先读取变更
- 再跑脱敏和校验 SQL
- 最后把结果并入目标表

真实工程里还会把代码拆成：

- `extract/`：CDC 客户端、source connector
- `transform/`：SQL 模型、Python UDF、规则配置
- `load/`：仓库写入器、Oracle/Snowflake/BigQuery 适配
- `state/`：checkpoint 持久化
- `monitoring/`：指标与告警

这种拆分比把一切写进一个超长脚本更可测，也更容易定位故障。

## 工程权衡与常见坑

真正难的不是把数据搬过去，而是把系统做成“失败可恢复、延迟可观测、口径可解释”。

先看常见坑：

| 常见坑 | 具体表现 | 后果 | 规避措施 |
| --- | --- | --- | --- |
| 单一巨型 DAG | 所有业务逻辑堆在一个流程里 | 任一环节失败都会拖垮全链路 | 拆分成源同步、标准化、主题建模三层 |
| 忽视幂等 | 重跑后重复写入 | 指标翻倍、账务错误 | 业务主键去重、Merge/Upsert、批次号控制 |
| 缺少监控 | 任务失败后没人知道 | 延迟和漏数悄悄扩散 | 建 freshness/completeness 告警 |
| 只写 SQL 不做测试 | 规则改了无人验证 | 口径漂移、线上回归 | 数据契约、断言、样例回放 |
| 混批流但无迟到策略 | 迟到事件覆盖旧数据失败 | 汇总口径不稳定 | watermark、补算窗口、回填流程 |
| 原始层权限过宽 | 敏感字段对太多人可见 | 合规风险 | 分级权限、列级脱敏、入口 ETL |

---

金融监管平台是一个典型真实工程例子。交易账本进入监管报送库前，必须先做脱敏和字段校验，然后再写入 Oracle。这里如果直接把原始交易数据落到分析仓，再慢慢处理，风险不是“技术债”，而是合规事故。

但同一个平台里，分析团队又需要按用户、渠道、设备、时段重跑历史分析。于是最常见的做法不是“全站只允许 ETL”或者“全站只允许 ELT”，而是：

- 监管链路：ETL，前置脱敏和强校验
- 分析链路：ELT，保留原始日志和变更
- 上游共享同一套 CDC 增量源

这就是混合架构的合理性。它不是妥协，而是按不同下游的约束分层设计。

可以把监控目标写得非常明确：

$$
freshness \le 30\ min,\ completeness \ge 99.9\%
$$

这类 SLO 必须绑定告警和 runbook。runbook（故障处置手册，白话说就是“出问题后按什么步骤查和修”）至少要回答三件事：

- 位点落后时先看哪里
- 漏数时如何补算
- 重跑时如何避免重复写入

没有这些文档，所谓“自动化管线”通常只是在自动制造排障成本。

## 替代方案与适用边界

如果把选择题做得足够简单，可以按下面这张表判断：

| 场景 | 推荐方式 | 原因 | 边界条件 |
| --- | --- | --- | --- |
| 强合规、强脱敏、下游固定报表 | ETL | 必须在落地前处理敏感数据 | 入口规则稳定，业务口径相对固定 |
| 自助分析、探索式查询、频繁改口径 | ELT | 保留原始数据更灵活 | 仓库算力和治理能力必须跟上 |
| 既有监管链路又有分析链路 | 混合 | 同一增量源服务不同目标 | 需要清晰分层和权限管理 |
| 小团队、低数据量、简单同步 | 轻量 ETL 或简单 ELT | 先追求可维护性 | 不要过度设计平台 |
| 高频事件流、近实时指标 | 流式 ETL/ELT | 延迟要求高 | 必须设计 watermark 和迟到处理 |

---

对初学者来说，可以先记一个非常实用的判断规则：

- **数据敏感，先处理再落地**：偏 ETL
- **问题多变，先保存再分析**：偏 ELT
- **两类需求同时存在**：混合

再回到厨房类比：

- 敏感菜、要先去骨去刺、不能让原料直接上桌的，像 ETL
- 普通原料先进储藏室，后面按菜单再做的，像 ELT

这个类比的边界也要明确。现实系统不是厨房，数据一旦落错地方，问题会在权限、审计、下游报表、重跑成本上放大。所以工程上真正关键的是：

- 增量位点是否可靠
- 转换是否可重放
- 写入是否幂等
- 监控是否能尽早发现 freshness/completeness 失控
- 原始层和主题层的权限是否分清

金融行业的混合方案很有代表性。交易账本进入监管库前走 ETL，在中间层脱敏并写入 Oracle；产品和运营分析则把同一 CDC 源同步到 Snowflake，在仓库中做 ELT，形成行为分析数据集。这样既守住合规边界，又保留分析灵活性。

所以最终不是“ETL 过时，ELT 先进”，而是：**不同边界对应不同架构位置，混合往往才是长期解。**

## 参考资料

1. System Design Space, *Data Pipeline / ETL / ELT Architecture*, 2026-03-15。覆盖数据流模式、ETL/ELT 架构差异、质量控制与常见设计问题。
2. BladePipe, *What Is Change Data Capture?*, 2026。重点说明 CDC 的工作方式，以及为什么增量抽取能减少不必要复制。
3. dbt Labs, *Data movement patterns explained*, 2026-03-10。说明 ETL、ELT 等数据移动模式在现代数据仓库中的适用场景。
4. Airbyte, *Incremental Load vs Full Load ETL*, 2026-01-06。给出 20GB 表每日仅变更 1% 的最小数值例子，可直接支撑“增量复制显著节省成本”的论断。
5. CloseLoop, *ETL vs ELT: Differences, Benefits, Use Cases*, 2026。用于理解金融等敏感场景下 ETL 与 ELT 并行的真实工程模式。
