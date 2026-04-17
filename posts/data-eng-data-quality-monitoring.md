## 核心结论

数据质量监控不是“出问题再查日志”，而是先定义一份可验证的契约，再持续度量契约是否被满足。这个契约通常写成 SLA。SLA 是服务等级协议，白话讲，就是你公开承诺“什么时间、以什么质量，把什么数据交付出来”。对数据产品来说，SLA 不能只写“表会生成”，还要写清楚“多久更新一次、允许缺多少、哪些字段必须一致、分布偏移到什么程度算失真”。

一套能落地的数据质量监控，至少要覆盖四类指标：

| 维度 | 解决的问题 | 常见指标 | 典型阈值 |
| --- | --- | --- | --- |
| 新鲜度 Freshness | 数据是否按时到达 | `Now - max(load_timestamp)` | `< 60 min` |
| 完整性 Completeness | 数据量是否明显缺失或异常膨胀 | `observed_rows / expected_rows` | `0.8 - 1.2` |
| 一致性 Consistency | 多表、多系统、多口径是否对齐 | 主键覆盖率、金额汇总差、维表匹配率 | 差异 `< 0.1%` |
| 分布偏移 Distribution Shift | 数据模式是否悄悄变了 | PSI、K-S、Wasserstein | `PSI <= 0.1` 稳定，`> 0.25` 告警 |

只看单一指标是不够的。新鲜度正常，不代表内容正确；行数正常，不代表字段分布没漂移；分布稳定，也不代表关联关系没断。工程上通常把多个指标组合成一个 SLI。SLI 是服务等级指标，白话讲，就是被 SLA 引用的可测量数字。只要任意关键指标越过阈值，就视为 SLA 违约。

一个新手容易理解的玩具例子是“城市供电”。如果把数据表看成电网，那么：

- 新鲜度看的是“最新一次供电时间”
- 完整性看的是“线路有没有断，电有没有明显少送”
- 一致性看的是“多个变电站读数是否统一”
- 分布偏移看的是“负载曲线是否突然换了形状”

对应公式可以直接写成：

$$
\text{Freshness Lag} = \text{Now} - \max(\text{load\_timestamp})
$$

$$
\text{Completeness Ratio} = \frac{\text{observed\_rows}}{\text{expected\_baseline}}
$$

$$
\text{PSI} = \sum_i (p_i - q_i)\ln\frac{p_i}{q_i}
$$

真正有效的监控不是多上几个图表，而是把“指标采集、阈值判断、状态聚合、告警降噪、责任人通知、Runbook 执行”串成一条自动链路。Great Expectations、dbt tests、自研脚本三者并不冲突，它们解决的是不同层面的校验问题：结构约束、规则校验、特殊业务统计。

---

## 问题定义与边界

数据质量监控的目标，不是证明“所有数据永远正确”，而是在约定边界内保证“数据产品可用且可信”。这里的“可用”至少包括三种失败形态：

| 失败类型 | 定义 | 典型表现 | 是否计入 Downtime |
| --- | --- | --- | --- |
| 不可用 | 数据无法查询或结果为空 | 任务没跑完、表不存在、查询 0 行 | 是 |
| 失真 | 数据能查到，但数量或值明显错误 | 行数骤降、金额总和异常 | 是 |
| 异常偏移 | 数据量看似正常，但模式已偏离 | 类别占比突变、数值分布漂移 | 视业务影响而定，关键链路通常计入 |

Downtime 是数据停机时间，白话讲，就是下游无法放心使用数据的那段时间。它不一定等于“数据库挂了”。如果一张销售日报在早上 9 点前必须可读，但 9 点时只有 30% 的数据写入完成，这也是数据停机。

因此，边界要先画清楚：

1. 监控对象是谁  
   是原始落地层、清洗层、宽表层，还是直接给业务看的报表层。不同层级阈值不同，不能混在一起。

2. 用户承诺是什么  
   比如“每天早上 9 点前，销售报表 99.9% 可读取，核心指标误差低于 0.1%”。

3. SLI 怎么定义  
   例如：
   - 新鲜度：`lag < 60 分钟`
   - 完整性：`行数在 7 日均值的 80%-120%`
   - 一致性：`订单明细汇总 = 报表汇总`
   - 分布偏移：`PSI <= 0.25`

4. 持续窗口是什么  
   偶发抖动不应该立刻叫醒人。监控系统要定义“连续 N 个窗口超标才升级”。

看一个真实工程例子。假设你维护一条销售报表链路，SLA 写成：

“每天北京时间 09:00 前，销售报表可读率达到 99.9%，核心收入指标误差小于 0.1%，数据延迟不超过 30 分钟。”

这句话需要翻译成机器可判断的规则：

| 指标 | 定义 | 阈值 | 告警动作 |
| --- | --- | --- | --- |
| Freshness lag | 当前时间减去最新加载时间 | `< 30 min` 正常，`30-60 min` warning，`> 60 min` critical | 通知 owner，超 60 分钟建 incident |
| Row completeness | 当日行数 / 近 7 日同周期基线 | `0.9-1.1` 正常，`0.8-0.9` 或 `1.1-1.2` warning，超出即 critical | 发送 ChatOps 告警 |
| Revenue consistency | 明细汇总与报表汇总差异 | `< 0.1%` | 触发数据核对 Runbook |
| Category PSI | 今日品类占比与历史基线比较 | `<= 0.1` 正常，`0.1-0.25` 观察，`> 0.25` 告警 | 提醒分析是否口径变化或埋点故障 |

边界之外的东西不要混入这个系统。例如，业务增长本身导致 GMV 提升，不属于质量问题；但埋点缺失导致“访问量突然腰斩”，属于质量问题。监控的职责是发现数据管线或数据定义是否失真，不是替代业务分析。

---

## 核心机制与推导

四类核心指标里，最容易落地的是新鲜度和完整性，最容易被忽略的是一致性和分布偏移。

先看新鲜度。设最新一条成功加载记录时间为 $t_{max}$，当前时间为 $t_{now}$，则：

$$
L = t_{now} - t_{max}
$$

其中 $L$ 就是延迟。只要 $L$ 超过 SLA 窗口，数据即使“存在”，也不能算合格。这类规则适合流水表、事件表、CDC 同步表。

再看完整性。完整性不是“有数据就算完整”，而是和预期基线比较。常用定义：

$$
C = \frac{R_{obs}}{R_{exp}}
$$

其中 $R_{obs}$ 是当前观测行数，$R_{exp}$ 是预期行数，通常来自历史同周期均值、中位数，或业务日历修正后的预测值。若 $C < 0.8$，一般说明有缺失；若 $C > 1.2$，可能是重复写入、重跑累加或上游脏数据扩散。

一致性更像“交叉验证”。比如订单事实表汇总收入，应与报表宽表收入对齐；用户维表中活跃用户数，应与明细去重后的结果接近。一致性校验常写成差值或差异率：

$$
D = \frac{|A - B|}{\max(|B|, \epsilon)}
$$

这里 $\epsilon$ 是一个很小的数，避免分母为零。若 $D > 0.001$，就意味着差异超过 0.1%。

分布偏移解决的是“总量正常，但形状错了”。例如用户来源渠道平时是“自然流量 40%、广告 35%、私域 25%”，今天突然变成“广告 80%”。总行数可能还在正常范围，但数据很可能出了问题。PSI 是常用指标，白话讲，它衡量“当前分布与基线分布差多远”。公式是：

$$
\text{PSI} = \sum_i (p_i - q_i)\ln\frac{p_i}{q_i}
$$

其中 $p_i$ 是基线分布第 $i$ 个桶的占比，$q_i$ 是当前分布的占比。经验上：

| PSI 区间 | 含义 | 建议动作 |
| --- | --- | --- |
| `<= 0.1` | 稳定 | 不告警 |
| `0.1 - 0.25` | 轻度偏移 | 观察，记录趋势 |
| `> 0.25` | 显著偏移 | 告警并排查 |

玩具例子可以这样算。某流水表要求 60 分钟内更新，近 7 日同时间窗口平均 10000 行，今天来了 8200 行，最新加载时间距现在 75 分钟，渠道分布 PSI 为 0.28。那它的状态不是单一故障，而是：

- 新鲜度违约：75 分钟 > 60 分钟
- 完整性接近下界：8200 / 10000 = 0.82
- 分布显著偏移：PSI = 0.28

此时即使报表还能打开，也应该视为 SLA 违约。因为“可查询”不等于“可信”。

真实工程里，多个指标通常按“最严重状态优先”聚合。可以把状态定义为：

- `OK`：全部核心指标正常
- `WARNING`：出现轻微异常，但未达到强制处理级别
- `CRITICAL`：任一核心指标严重违约
- `DOWN`：完全不可用或不可查询

SLA 违约时长则可写成：

$$
\text{Violation Duration} = t_{recover} - t_{breach}
$$

如果 08:40 首次检测发现违约，09:25 恢复正常，那么这次 downtime 或 violation duration 就是 45 分钟。只有把这段时间持续记录下来，后续才能统计月度可用性：

$$
\text{Availability} = 1 - \frac{\text{Downtime}}{\text{Total Scheduled Time}}
$$

这也是为什么“是否把失真计入停机”必须提前定义。否则月报里的可用率没有可比性。

---

## 代码实现

实现时不要把监控写成一大段 if-else。更稳妥的结构是四步：

1. 采集指标：读取最新加载时间、行数、对账结果、分布统计
2. 判断状态：每个指标独立给出 `OK/WARNING/CRITICAL`
3. 聚合结果：取最严重状态，并计算违约持续时间
4. 执行动作：写入事件表、发送告警、触发 Runbook

下面给一个可运行的 Python 最小实现。它不依赖外部库，重点是把“指标 -> 状态 -> 告警”这条链路写清楚。

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import log

OK = "OK"
WARNING = "WARNING"
CRITICAL = "CRITICAL"

@dataclass
class MetricStatus:
    name: str
    value: float
    status: str
    message: str

def calc_freshness_lag_minutes(now: datetime, last_load_time: datetime) -> float:
    return (now - last_load_time).total_seconds() / 60.0

def calc_completeness_ratio(observed_rows: int, expected_rows: int) -> float:
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    return observed_rows / expected_rows

def calc_psi(baseline, current, eps=1e-6) -> float:
    if len(baseline) != len(current):
        raise ValueError("baseline/current length mismatch")
    total = 0.0
    for p, q in zip(baseline, current):
        p = max(p, eps)
        q = max(q, eps)
        total += (p - q) * log(p / q)
    return total

def judge_freshness(lag_minutes: float) -> MetricStatus:
    if lag_minutes > 60:
        return MetricStatus("freshness", lag_minutes, CRITICAL, "lag > 60 min")
    if lag_minutes > 30:
        return MetricStatus("freshness", lag_minutes, WARNING, "lag > 30 min")
    return MetricStatus("freshness", lag_minutes, OK, "freshness ok")

def judge_completeness(ratio: float) -> MetricStatus:
    if ratio < 0.8 or ratio > 1.2:
        return MetricStatus("completeness", ratio, CRITICAL, "ratio outside [0.8, 1.2]")
    if ratio < 0.9 or ratio > 1.1:
        return MetricStatus("completeness", ratio, WARNING, "ratio outside [0.9, 1.1]")
    return MetricStatus("completeness", ratio, OK, "completeness ok")

def judge_psi(psi: float) -> MetricStatus:
    if psi > 0.25:
        return MetricStatus("distribution_shift", psi, CRITICAL, "psi > 0.25")
    if psi > 0.1:
        return MetricStatus("distribution_shift", psi, WARNING, "psi > 0.1")
    return MetricStatus("distribution_shift", psi, OK, "distribution stable")

def aggregate_status(statuses):
    order = {OK: 0, WARNING: 1, CRITICAL: 2}
    return max(statuses, key=lambda s: order[s.status]).status

def build_alert_payload(table_name: str, owner: str, statuses):
    overall = aggregate_status(statuses)
    return {
        "table": table_name,
        "owner": owner,
        "overall_status": overall,
        "violations": [s.message for s in statuses if s.status != OK],
    }

# toy example
now = datetime(2026, 4, 14, 9, 0, 0)
last_load_time = now - timedelta(minutes=75)
lag = calc_freshness_lag_minutes(now, last_load_time)
ratio = calc_completeness_ratio(observed_rows=8200, expected_rows=10000)
psi = calc_psi([0.4, 0.35, 0.25], [0.2, 0.55, 0.25])

statuses = [judge_freshness(lag), judge_completeness(ratio), judge_psi(psi)]
payload = build_alert_payload("sales_daily_report", "data-oncall", statuses)

assert round(lag, 1) == 75.0
assert round(ratio, 2) == 0.82
assert aggregate_status(statuses) == CRITICAL
assert payload["table"] == "sales_daily_report"
assert "lag > 60 min" in payload["violations"]
```

这段代码表达了三个关键点。

第一，规则与采集分离。`calc_*` 负责算值，`judge_*` 负责解释值。这样阈值调整时不需要改采集逻辑。

第二，状态按严重级别聚合。只要有一个核心指标进入 `CRITICAL`，整体就是 `CRITICAL`。

第三，告警载荷里必须带责任人和问题摘要，方便后续接 ChatOps 或工单系统。

如果把它放进真实工程链路，常见流程是：

- Airflow、Dagster 或调度器跑完任务后写入 `last_load_time`
- dbt tests 校验唯一性、非空、关系完整性
- Great Expectations 校验业务规则和统计约束
- 自研脚本计算 PSI、K-S 或 Wasserstein 距离
- 聚合器把结果写入 `incident_queue`
- 告警服务根据级别推送到 Slack、飞书、PagerDuty，并附上 Runbook

一个真实工程例子是“销售报表链路”：

- `dbt tests` 保证 `order_id` 唯一、`shop_id` 不为空、事实表能关联到门店维表
- `Great Expectations` 保证 `total_amount >= 0`、退款率不超过阈值、分区行数不低于历史基线
- 自研分布脚本检查渠道分布 PSI 和客单价 Wasserstein 距离
- 聚合器判断 08:55 是否满足 09:00 前可读 SLA
- 若不满足，自动通知报表 owner，并在消息里附上“先查上游 CDC，再查宽表去重逻辑”的 Runbook 链接

这套结构的重点不是工具名，而是责任链闭环。没有责任人、没有恢复流程、没有违约时长记录，告警就只是噪声。

---

## 工程权衡与常见坑

监控系统最难的部分不是写公式，而是控制误报和漏报。阈值过紧会把 on-call 团队拖垮，阈值过松又会让坏数据悄悄流入业务。

常见做法是两级告警加持续窗口：

| 级别 | 条件 | 动作 |
| --- | --- | --- |
| Warning | 连续 2 个窗口轻微超标 | 通知群组，不拉人值班 |
| Critical | 单次严重超标，或连续 3 个窗口 warning | 触发 incident，通知 owner/on-call |
| Down | 不可查询、表缺失、关键链路 0 行 | 立即升级，计入 downtime |

这里的“窗口”是离散观测周期，比如每 5 分钟或每 15 分钟采样一次。持续窗口的意义是降噪。比如 CDC 延迟偶尔抖一下，5 分钟后恢复，这不值得建事故单；但如果连续 3 个窗口都超标，就说明问题在持续。

常见坑可以直接列出来：

| 坑 | 后果 | 对策 |
| --- | --- | --- |
| 只监控任务成功失败 | 任务成功但结果错误，监控失效 | 增加行数、汇总、一致性、分布指标 |
| 只看总量不看分布 | 总行数正常但类目比例异常 | 增加 PSI、K-S、类别占比校验 |
| 阈值写死，不区分节假日 | 大促或节假日频繁误报 | 用业务日历和同周期基线 |
| 每次超标都立即报警 | on-call 疲劳，最终忽略告警 | warning/critical 分层，要求持续窗口 |
| 不定义 downtime | 无法统计 SLA 可用率 | 约定“不可用/失真/不可查询”哪些计入停机 |
| 没有 owner 和 Runbook | 发现问题后没人处理 | 每条规则绑定责任人、处理步骤、升级路径 |
| 所有规则都上 GE 或 dbt | 工具职责混乱，维护成本升高 | 结构规则用 dbt，复杂断言用 GE，特殊统计自研 |

这里有一个工程上非常关键的判断：不是所有异常都应该叫做“数据质量事故”。如果广告投放策略真的变了，渠道分布变化可能是业务真实变化，不是质量问题。所以分布偏移告警通常需要二次确认：先标记“异常”，再判断是“业务变化”还是“数据错误”。这也是为什么分布指标往往先进入 `WARNING`，由人或额外规则做升级。

另一个常见坑是“期望值”的来源太粗糙。很多团队直接拿昨天行数做今天基线，这在周末、月初、促销日都会失真。更稳妥的办法是：

- 按星期几或小时段建基线
- 排除历史异常天
- 对节假日和大促做特殊日历修正
- 对新表或低频表允许更宽阈值

最后要强调，downtime 定义必须写入制度，而不是写在工程师脑子里。建议至少统一这三类是否计入：

- 不可用：表不存在、查询失败、返回空结果
- 失真：核心指标误差超过阈值
- 不可查询：权限、元数据、分区损坏导致下游无法消费

只有定义清楚，SLA 追踪才有意义。

---

## 替代方案与适用边界

实际选型时，不要问“哪个工具最好”，而要问“哪类问题由哪类工具最省成本”。

| 工具/方案 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| dbt tests | 模型层结构校验、主键唯一、非空、关系完整性 | 简单、贴近 SQL 模型、易纳入 CI/CD | 对复杂统计和漂移检测支持弱 |
| Great Expectations | 复杂规则、字段范围、分布约束、数据文档化 | 表达力强，适合做规则资产沉淀 | 配置与运维成本更高 |
| 自研方案 | 特殊业务指标、PSI/K-S/Wasserstein、定制告警聚合 | 灵活，能贴合业务 SLA | 需要自己维护平台能力 |
| 定时 SQL + 告警脚本 | 小团队、低成本、起步阶段 | 上手快，成本低 | 扩展性差，规则管理容易散落 |

适用边界可以这样理解。

如果你是刚起步的数据团队，只有少量核心报表，先用 `dbt tests + 定时 SQL + 简单 webhook` 就够了。先把“唯一性、非空、汇总对账、新鲜度”这些硬规则跑起来，比一开始就引入完整平台更现实。

如果你已经有几十到上百张核心表，需要系统化沉淀规则、统一结果展示、支持复杂断言，那么 Great Expectations 更合适。它适合承载“列值范围、分布期望、条件组合”等规则。

如果你有特殊业务需求，比如“新客占比 PSI”“广告渠道分布 Wasserstein 距离”“推荐结果点击率漂移”，那通常要上自研。因为这些指标往往带强业务语义，不适合完全依赖通用框架。

看一个完整的真实链路例子。销售报表系统里：

- `dbt tests` 保证 `order_id` 唯一、`customer_id` 不为空、事实表能关联维表
- `Great Expectations` 保证 `total_amount` 非负、分区行数在历史基线范围、退款率不异常
- 自研脚本计算渠道分布 PSI，并在超阈值时发 Slack 告警
- 聚合器按 09:00 SLA 计算是否违约以及违约分钟数
- 月度报表统计总 downtime 和每类事故占比

这个组合的优点是职责清晰。dbt 管结构，GE 管业务规则，自研管高级统计和告警编排。

边界也要清楚。数据质量监控不等于数据治理，不等于血缘系统，不等于元数据目录，也不等于 APM。它的核心任务只有一个：在 SLA 约定下，尽早发现“数据已不可信”的状态，并把问题送到正确的人手里。

---

## 参考资料

1. Great Expectations + dbt 结合与数据合同实践（Calmops）  
2. 数据质量 KPI：Freshness、Completeness、Bias 等解释（Datacult）  
3. 数据产品 SLA、SLO 与 Runbook 响应实践（dbt Labs / DesignGurus）  
4. 分布偏移检测方法：PSI、K-S、Wasserstein（System Overflow / AllDaysTech）  
5. 数据停机时间定义、影响与统计口径（TheDataOps / Sparvi）
