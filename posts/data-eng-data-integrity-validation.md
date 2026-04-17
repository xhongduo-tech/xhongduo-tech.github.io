## 核心结论

数据完整性校验不是“看起来差不多就行”，而是把数据是否可信拆成三层规则同时验证：Schema 校验、业务规则校验、统计校验。Schema 校验就是检查“长得对不对”，例如列名、类型、必填字段；业务规则校验就是检查“逻辑对不对”，例如订单金额不能超过授信额度；统计校验就是检查“整体分布有没有突然异常”，例如近 7 天缺失率不能超过 2%。

这三类规则要一起用，因为它们覆盖的问题不同。只做 Schema，你能发现列丢了，但发现不了“金额字段类型没变、含义却错了”；只做业务规则，你可能在数据已经大面积漂坏后才发现；只做统计，你又很难精确定位是哪一条规则被破坏。工程上更稳妥的做法是：先用 JSON Schema 或等价规则定义结构，再用 Great Expectations（简称 GX，意思是“把数据预期写成可执行规则”）组织 expectation suite，最后用 checkpoint 把“执行校验、记录结果、报警、阻断、修复”串起来。

GX 文档里的一个关键点是：`ExpectationSuiteValidationResult.success` 表示整套规则是否全部通过，本质上可以写成

$$
\text{suite.success} = \bigwedge_{i=1}^{n} \text{expectation}_i.\text{success}
$$

也就是“所有 expectation 都成功，整套才成功”。这个聚合结果再进入 Checkpoint 的 actions，驱动后续动作：

`ExpectationSuite -> ValidationResult -> Checkpoint -> Actions -> Data Docs`

玩具例子可以很简单：一个 `users.csv` 至少有 `user_id`、`country`、`age` 三列，`country` 只能是 `CN/US/JP`，`age` 不能为负数。真实工程例子则通常是跨表或跨系统：例如金融转账表 `transfers.amount` 必须和余额表 `transfer_balance.total_amount` 对齐，否则下游账务汇总就会错。

---

## 问题定义与边界

数据完整性校验，指的是用一组明确的、可执行的规则，验证字段、行、表以及跨表关系是否符合预期，再决定数据能不能进入下游。这里的“完整性”不是单指“有没有丢行”，而是更广义的“结构没坏、关系没断、逻辑没偏、统计没漂”。

可以把三类校验先分清楚：

| 类型 | 关注点 | 示例 expectation | 常见失败模式 |
| --- | --- | --- | --- |
| Schema 校验 | 列是否存在、类型是否正确、必填是否满足 | `order_id` 必须存在；`amount` 必须是数值 | 上游改列名、CSV 推断错类型、必填列空值增多 |
| 业务规则校验 | 单行或多行是否满足业务逻辑 | `order.total_amount <= customer.credit_limit` | 字段仍合法，但业务含义错了 |
| 统计校验 | 整体分布是否稳定 | 近 7 天缺失率 `< 2%`，均值波动不超过阈值 | 批量漂移、脏数据突然增多、采集链路局部失效 |

边界也要讲清楚。第一，完整性校验不只针对单表。只检查“订单表里有 `order_id`”不够，还要检查订单里的 `customer_id` 是否能在客户表中找到，也就是参照完整性，白话说就是“引用出去的东西要真存在”。第二，它不应只依赖应用层。应用层常常只拦在线写入，但离线同步、补数、回灌、第三方导入都可能绕过应用逻辑。第三，它也不该是一次性动作。数据质量是持续过程，规则需要长期执行、记录、比较和复审。

对初学者来说，可以先记一个最小集合：

- “订单表至少有 `order_id`、`customer_id`”是 Schema 校验。
- “订单金额不能超过客户授信额度”是业务规则校验。
- “近 7 天手机号缺失率低于 2%”是统计校验。

只有把三者放在一起，才构成真正的网状防护。

---

## 核心机制与推导

先看 GX 里的几个核心对象。

Expectation，可以理解成“一条可验证的预期”。例如 `ExpectColumnToExist("order_id")`。Expectation Suite，就是“一组 expectation 的集合”。Validation Result，就是“一次执行后的结果对象”。Checkpoint，则是“把一次或多次校验和后续动作封装在一起的执行入口”。

因此，基本机制是：

1. 定义 Expectation Suite。
2. 用数据批次运行校验，得到 Validation Result。
3. 读取 `success` 和明细失败项。
4. 交给 Checkpoint 的 actions 做后处理。
5. 生成 Data Docs 和监控记录。

这个过程可以写成更直观的流程图：

```text
ExpectationSuite
      |
      v
ValidationResult
      |
      v
Checkpoint.actions
      |
      +--> block
      +--> alert
      +--> fix
      +--> store/report
      |
      v
Data Docs
```

为什么说 `ValidationResult.success` 是中枢？因为它把很多离散规则收敛成一个统一布尔值。比如严格模式下有 8 条 expectation，只要有 1 条失败，整套就是失败。你可以把它理解为逻辑与：

$$
\text{ExpectationSuiteValidationResult.success}
=
\text{e}_1.success \land \text{e}_2.success \land \cdots \land \text{e}_n.success
$$

这就是“通过才放行”的数学基础。

玩具例子里，这个机制非常直接。假设有一个注册用户表：

- `user_id` 必须存在且唯一。
- `country` 只能是 `CN/US/JP`。
- `age` 必须大于等于 0。
- 最近一天空邮箱比例不能超过 1%。

如果前 3 条都通过，最后一条失败，那么整套结果就是失败。工程上可以定义为“警告但不阻断”，因为邮箱缺失可能影响触达，但不一定影响账务。相反，如果 `user_id` 不唯一，则通常应立即阻断，因为主键冲突会污染整个下游维表或事实表。

真实工程例子更典型的是跨表完整性。以转账系统为例，`transfers` 表记录交易摘要，`transfer_balance` 表记录余额分解。即使 `transfers.amount` 的类型仍是 `DECIMAL`，如果它和 `transfer_balance.total_amount` 不相等，下游总账就会错。这个问题靠单表 schema 校验发现不了，只能通过 join 后的业务或完整性 expectation 发现。

因此可把机制写成：

$$
\text{ValidationResult} \rightarrow \text{Checkpoint.actions} \rightarrow \{\text{block}, \text{alert}, \text{fix}, \text{report}\}
$$

其中 `block` 是阻断下游，`alert` 是发通知，`fix` 是自动修复或放入修复队列，`report` 是沉淀校验报告供复盘。

---

## 代码实现

先给一个可运行的 Python 玩具实现，不依赖 GX，目的是把“schema + 业务 + 统计 + 聚合结果”讲透：

```python
from collections import Counter

rows = [
    {"user_id": 1, "country": "CN", "age": 18, "email": "a@example.com"},
    {"user_id": 2, "country": "US", "age": 30, "email": ""},
    {"user_id": 3, "country": "JP", "age": 22, "email": "c@example.com"},
]

required_fields = {"user_id": int, "country": str, "age": int, "email": str}
allowed_countries = {"CN", "US", "JP"}

def schema_check(data):
    for row in data:
        for field, expected_type in required_fields.items():
            assert field in row, f"missing field: {field}"
            assert isinstance(row[field], expected_type), f"type error: {field}"
    return True

def business_check(data):
    user_ids = [r["user_id"] for r in data]
    assert len(user_ids) == len(set(user_ids)), "user_id must be unique"
    for row in data:
        assert row["country"] in allowed_countries, "invalid country"
        assert row["age"] >= 0, "age must be non-negative"
    return True

def statistical_check(data, max_missing_rate=0.5):
    missing_email = sum(1 for r in data if r["email"] == "")
    rate = missing_email / len(data)
    assert rate < max_missing_rate, f"email missing rate too high: {rate}"
    return True

results = [
    schema_check(rows),
    business_check(rows),
    statistical_check(rows),
]

suite_success = all(results)
assert suite_success is True
print("validation passed:", suite_success)
```

这个例子对应的就是最小版 `ExpectationSuiteValidationResult.success = all(expectation.success)`。

如果切到 GX，思路也是一样，只是规则对象、结果对象和动作框架已经标准化。下面是接近真实项目的示意代码，包含 schema expectation、跨表 SQL expectation、checkpoint action：

```python
import great_expectations as gx
import great_expectations.expectations as gxe
from great_expectations.checkpoint import (
    SlackNotificationAction,
    UpdateDataDocsAction,
)

context = gx.get_context()

# 1. 数据源与数据资产
datasource = context.data_sources.add_postgres(
    "warehouse",
    connection_string=CONNECTION_STRING,
)
transfers_asset = datasource.add_table_asset(
    name="transfers",
    table_name="integrity_transfers",
)
batch_def = transfers_asset.add_batch_definition_whole_table("transfers_batch")

# 2. expectation suite：schema + 业务/完整性
suite = context.suites.add(gx.ExpectationSuite(name="transfers_integrity_suite"))
suite.add_expectation(gxe.ExpectColumnToExist(column="transfer_id"))
suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="amount", type_="DOUBLE_PRECISION"))
suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="transfer_id"))

class ExpectTransferAmountsToMatch(gxe.UnexpectedRowsExpectation):
    description = "transfers.amount must equal transfer_balance.total_amount"
    unexpected_rows_query = """
        select *
        from {batch} t
        join integrity_transfer_balance b using (transfer_balance_id)
        where t.amount <> b.total_amount
    """

validation_definition = context.validation_definitions.add(
    gx.ValidationDefinition(
        name="transfers_validation",
        data=batch_def,
        suite=suite,
    )
)

# 3. checkpoint：记录、通知、更新报告
action_list = [
    SlackNotificationAction(
        name="slack_on_failure",
        slack_token="${validation_notification_slack_webhook}",
        slack_channel="${validation_notification_slack_channel}",
        notify_on="failure",
        show_failed_expectations=True,
    ),
    UpdateDataDocsAction(name="update_data_docs"),
]

checkpoint = gx.Checkpoint(
    name="transfers_checkpoint",
    validation_definitions=[validation_definition],
    actions=action_list,
    result_format={"result_format": "COMPLETE"},
)

result = checkpoint.run()
print("checkpoint success:", result.success)
```

这里有两个工程点要注意。

第一，GX 内置 expectation 适合表达通用约束，例如列存在、类型、唯一性、范围、集合成员等；复杂跨表关系可以通过自定义 SQL expectation 补上。官方文档里明确给出了“查询返回任意异常行则失败”的模式，这对完整性校验非常实用。

第二，失败后的动作不应只剩“抛异常”。更常见的分级是：

| 失败类型 | 处理方式 | 典型场景 |
| --- | --- | --- |
| 阻断 | 终止流水线，不让数据入仓或入宽表 | 主键不唯一、金额对账失败 |
| 警告 | 发 Slack/邮件，但允许继续 | 标签列缺失率轻微升高 |
| 修复 | 自动填默认值或写入修复队列 | 非关键维度字段缺失 |
| 记录 | 落盘并更新 Data Docs | 所有校验都应做 |

---

## 工程权衡与常见坑

完整性校验的难点不在“会不会写规则”，而在“规则是否足够覆盖真实风险”。

最常见的误区是只做 Schema 校验。列都在、类型都对，不代表业务正确。比如 `transfers.amount` 仍然是数值列，但由于 join 错位，金额和余额表不一致，这类错误只有跨表校验能抓到。另一类误区是只在单一系统校验。你在数仓里验证通过，不代表它和交易系统、CRM、支付网关仍然一致；一旦多源同步有延迟或漏数，单系统视角会给出“假通过”。

下面这张表基本覆盖了常见坑：

| 常见坑 | 结果 | 规避措施 |
| --- | --- | --- |
| 只做单表 Schema 校验 | 发现不了跨表关系断裂 | 增加 referential integrity 和 cross-table expectation |
| 没有明确 NULL 策略 | 误报或漏报 | 区分“允许为空”“缺失即失败”“缺失可修复” |
| Schema 变更后不复审 expectation | 规则与真实表结构脱节 | 把 expectation 版本管理纳入发布流程 |
| 只报警不阻断 | 错数据继续流向下游 | 为关键表定义硬阻断门槛 |
| 只看单次结果，不看趋势 | 漂移长期积累后才暴露 | 生成 Data Docs 和按日监控统计 |
| 忽略多源一致性 | 系统间账不平 | 用 multi-source expectation 或对账任务补齐 |

新手最容易低估的是 NULL 策略。NULL 不是一个单纯的“坏值”，而是业务语义的一部分。比如 `middle_name` 可以为空，`order_id` 不可以为空，`coupon_code` 为空可能表示“未使用优惠券”，而 `payment_status` 为空则几乎一定是上游异常。如果不先定义字段语义，只会得到一堆没有行动价值的告警。

真实工程里还要考虑执行成本。跨表 join、跨系统对账、统计窗口计算都可能很贵。实践上通常会把规则分层：

- 入湖或入仓前，跑轻量 schema 校验和关键唯一性校验。
- 核心事实表装载后，跑业务规则和跨表完整性。
- 按日或按小时，跑统计漂移和多系统对账。

这样能在时效和成本之间取得平衡。

---

## 替代方案与适用边界

完整性校验并不是只能靠 GX。常见替代方案有三类。

第一类是数据库约束，例如 `NOT NULL`、`CHECK`、`UNIQUE`、`FOREIGN KEY`。这类约束离数据最近，执行稳定，适合单数据库核心约束。第二类是应用层校验，例如写入 API 时检查字段和业务逻辑。它适合在线交易，但覆盖不了离线补数和外部导入。第三类是 ETL 脚本里的 ad-hoc 校验，也就是临时 SQL 或 Python 脚本，优点是灵活，缺点是难复用、难观测、难统一管理。

对比可以直接看表：

| 方案 | 可观测性 | 自动化动作 | 跨系统一致性 | 适用边界 |
| --- | --- | --- | --- | --- |
| 数据库约束/传统脚本 | 弱 | 弱 | 弱到中 | 小型、单库、规则稳定 |
| 应用层校验 | 中 | 中 | 弱 | 在线写入链路 |
| GX expectation + checkpoint | 强 | 强 | 中到强 | 多数据源、需要统一报告和告警 |

所以边界很清楚：

- 小型单数据库项目，先把数据库约束做好，已经能解决一批问题。
- 如果只是一次性迁移，临时脚本也够用。
- 一旦进入“多表、多源、持续校验、需要报告和动作编排”的阶段，GX 这类框架价值才真正体现出来。

对初级工程师，一个实用判断标准是：当你开始需要“同一套规则每天跑、失败时自动通知、结果能沉淀成报告、还能比较历史趋势”，就已经越过了“只靠脚本”的边界。

---

## 参考资料

- [Great Expectations: Validate data schema with GX](https://docs.greatexpectations.io/docs/reference/learn/data_quality_use_cases/schema/)：查看 schema expectation 的官方定义与示例，适合确认列存在、类型、列集合等规则的写法。
- [Great Expectations: Validate data integrity with GX](https://docs.greatexpectations.io/docs/reference/learn/data_quality_use_cases/integrity/)：查看跨表完整性、业务规则、自定义 SQL expectation 与 multi-source expectation 的官方示例。
- [Great Expectations: Checkpoint API](https://docs.greatexpectations.io/docs/reference/api/checkpoint_class/)：确认 Checkpoint 的职责、参数和 `run()` 返回结果。
- [Great Expectations: Create a Checkpoint with Actions](https://docs.greatexpectations.io/docs/core/trigger_actions_based_on_results/create_a_checkpoint_with_actions/)：查看 `SlackNotificationAction`、`UpdateDataDocsAction` 等 action 的配置方式。
- [Great Expectations: Validation Result](https://legacy.017.docs.greatexpectations.io/docs/0.16.16/terms/validation_result/)：复核 `ExpectationSuiteValidationResult.success`、`results`、`statistics` 等字段含义。
- [Great Expectations: Data Docs](https://legacy.017.docs.greatexpectations.io/docs/0.14.13/terms/data_docs/)：确认 Data Docs 如何展示 expectation suite 与 validation result，以及如何作为团队共享的数据质量报告。
- [JSON Schema 官方文档](https://json-schema.org/)：查看结构化 schema 约束的基础规范，适合理解“字段存在、类型、required、枚举值”等表达方式。
