## 核心结论

模型更新不是“把新文件传上去”这么简单，它本质上是一个受控发布系统。受控发布的意思是：每一个新模型都必须有明确身份、明确部署路径、明确退出路径。身份靠版本号，路径靠分阶段流量切换，退出路径靠自动回滚。

最小闭环可以写成三件事：

1. 用语义化版本标识模型与相关代码、配置、特征定义。
2. 用蓝绿、金丝雀或滚动更新把新模型逐步引入真实流量。
3. 用监控和阈值判断是否继续推进，否则立即回滚到上一个稳定版本。

语义化版本是“让版本号表达变更级别”的规则，常写为：

$$
semantic\_version := major.minor.patch
$$

其中：

- `major`：破坏性改动，旧调用方式、输入输出契约或特征定义不再兼容。
- `minor`：兼容性增强，在不破坏旧接口的前提下提升能力。
- `patch`：缺陷修复，不改变主要行为边界。

对新手来说，可以把模型当成一个 APP release。`v2.1.0` 表示一次兼容增强，`v2.1.1` 表示修小问题。正确做法不是直接 100% 切流量，而是先给 1% 流量跑 30 分钟，确认准确率、延迟、业务 KPI 都正常，再逐步加量。

结论可以再压缩成一句：用“版本号 + 渐进流量 + SLO 守门 + 自动回滚”形成闭环，才能把模型更新从手工操作变成工程系统。

---

## 问题定义与边界

模型更新要同时解决两个目标：服务可用性和行为可追踪性。

服务可用性很好理解，就是线上不能因为换模型而大面积超时、报错或指标劣化。行为可追踪性是指：出问题时必须知道当前是谁、在什么时候、把哪个版本、以什么配置、部署到了哪里，并且能恢复到哪个旧版本。

这里的“边界”必须提前写清楚，否则团队会在事故发生时才争论。至少要定义以下四类边界：

| 边界项 | 需要回答的问题 | 典型规则 |
|---|---|---|
| 版本边界 | 什么算破坏性改动 | 输入 schema 变化、输出标签集变化、特征含义变化都算 major |
| 部署边界 | 每一阶段放多少流量 | 1% → 5% → 25% → 50% → 100% |
| 监控边界 | 看哪些指标 | 准确率、p99 延迟、错误率、CTR/转化率等 |
| 回滚边界 | 什么情况立即回退 | 任一核心指标超阈值，或出现统计显著回归 |

如果没有这些边界，所谓“上线”只是把风险留到生产环境处理。

一个适合新手理解的玩具例子是邮件垃圾分类模型。旧版本 `v1.3.2` 已稳定运行，新版本 `v1.4.0` 只是在原有特征上新增了域名信誉分，属于兼容增强。团队设定分阶段流量为 1%、10%、50%、100%，并规定：

- 精确率不能下降超过 0.3%
- p99 延迟不能高于 120ms
- 错误率不能高于 0.2%

那么这次发布就不再是“感觉应该没问题”，而是“只有满足这些约束才允许继续”。

你可以把这个过程理解为交通管控。每次更新不是一次性开闸，而是按阶段放行车辆：1% → 5% → 25%。只要任一阶段出现事故，就立刻收回到旧路线。这个比喻只用于理解节奏，真正工程里判断依据不是“看起来堵不堵”，而是预先定义好的监控阈值。

下面这张表可以帮助理解发布阶段的边界：

| 阶段 | 流量比例 | 观察时间 | 检查指标 | 是否允许继续 |
|---|---:|---:|---|---|
| Stage 1 | 1% | 30 分钟 | p99、错误率、核心 KPI | 全部达标才继续 |
| Stage 2 | 5% | 30-60 分钟 | p99、准确率、核心 KPI | 全部达标才继续 |
| Stage 3 | 25% | 1-2 小时 | 多维指标 + 资源占用 | 全部达标才继续 |
| Stage 4 | 50% | 2-4 小时 | 全链路稳定性 | 全部达标才继续 |
| Stage 5 | 100% | 持续监控 | 回归监测 | 仍需保留回滚能力 |

边界还包括保留策略。旧模型要保留多久，不是存档问题，而是恢复能力问题。若旧版本镜像、权重文件、配置、依赖环境已经被清掉，那么“回滚”在纸面上存在，在生产中不存在。

---

## 核心机制与推导

模型更新的核心机制不是单点技术，而是两套规则的组合：版本规则和发布规则。

版本规则回答“这个版本能不能直接替换”。发布规则回答“即使能替换，也应该怎样把它安全送到线上”。

先看版本规则。语义化版本背后的工程意义不是好看，而是约束团队行为：

- `major` 增长时，默认不能直接全量替换，必须重点验证调用契约、特征兼容性、回滚脚本和数据恢复。
- `minor` 增长时，通常允许灰度导入，因为接口边界不变。
- `patch` 增长时，说明预期行为接近旧版本，重点是验证修复是否引入副作用。

再看发布规则。假设有一组发布阶段 $i \in \{1,2,\dots,n\}$，每个阶段分配流量比例 $f_i$，并采集该阶段的关键指标 $M_i$。系统允许继续推进的条件是：这些指标都落在安全区间 $SLO_i$ 内。SLO 是“服务等级目标”，白话讲就是系统必须守住的最低服务标准。

可以写成：

$$
deploy(v_{new}) \Rightarrow \{f_1, f_2, \dots, f_n\}
$$

对每个阶段执行判定：

$$
if\ \exists i: M_i \notin SLO_i \Rightarrow rollback(target\_version)
$$

否则：

$$
M_i \in SLO_i \Rightarrow proceed(f_{i+1})
$$

这套推导的关键点在于：推进和回滚都不是人工拍板，而是对阶段指标做判定。

更具体一点，可以把每阶段的决策函数写成：

$$
Decision_i =
\begin{cases}
continue, & \text{if } latency_{p99} < L_i \land error\_rate < E_i \land KPI \ge K_i \\
rollback, & \text{otherwise}
\end{cases}
$$

这里：

- $latency_{p99}$ 是 p99 延迟，意思是 99% 请求都低于该耗时，用来衡量长尾慢请求。
- $error\_rate$ 是错误率，表示失败请求占比。
- $KPI$ 是关键业务指标，比如 CTR、转化率、召回率收益等。

一个玩具例子可以帮助把公式落地。假设一个推荐排序模型从 `v2.0.3` 升级到 `v2.1.0`，发布计划如下：

| 阶段 | 流量 $f_i$ | 条件阈值 |
|---|---:|---|
| 1 | 1% | p99 < 220ms，CTR 不下降，错误率 < 0.1% |
| 2 | 5% | p99 < 220ms，CTR 提升至少 0.2% |
| 3 | 25% | p99 < 230ms，转化率不下降 |
| 4 | 100% | 连续稳定 2 小时 |

如果在 5% 阶段看到 p99 从 205ms 升到 238ms，即使 CTR 有小幅提升，也应中止推进。因为上线目标不是单一指标最大化，而是满足整体 SLO。

一个真实工程例子是信息流排序模型更新。假设线上稳定版本是 `v2.2.5`，新版本 `v3.0.0` 修改了特征拼接方式和打分校准逻辑。由于这类改动可能改变输入分布解释方式，应该按 major 版本处理。发布流程可以是：

- 先在蓝环境部署 `v3.0.0`，保持绿环境继续承接主流量。
- 将 1% 流量切到蓝环境，持续 30 分钟。
- 检查 p99 是否低于 250ms，CTR 是否提升 0.5%，错误率是否未升高。
- 若通过，再进入 5% 阶段。
- 如果 5% 阶段出现延迟反弹 15ms，且收益不显著，则立即把流量切回 `v2.2.5`，并保留蓝环境供排查。

这就是“版本规则 + 流量规则 + 指标规则”的联合机制。模型更新真正难的地方，不在“如何传文件”，而在“如何把前进条件和退出条件明确成代码”。

---

## 代码实现

工程里通常需要一个模型注册表。注册表是“记录每个模型元数据的地方”，最简单时可以是一个 JSON 文件，复杂时可以是数据库或专门的 Model Registry。它至少应包含版本、创建时间、状态、部署环境、上一个稳定版本。

下面给一个可运行的 Python 玩具实现。它不是完整平台，只演示三件事：读取注册表、分阶段检查指标、在不满足阈值时回滚并写日志。

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class ModelRecord:
    version: str
    created_at: str
    status: str  # "stable", "candidate", "rolled_back"
    previous_stable: Optional[str] = None


@dataclass
class RollbackLog:
    reason: str
    who: str
    timestamp: str
    from_version: str
    to_version: str


@dataclass
class Registry:
    models: Dict[str, ModelRecord] = field(default_factory=dict)
    rollback_logs: List[RollbackLog] = field(default_factory=list)

    def add_model(self, record: ModelRecord) -> None:
        self.models[record.version] = record

    def mark_stable(self, version: str) -> None:
        for model in self.models.values():
            if model.status == "stable":
                model.status = "candidate"
        self.models[version].status = "stable"

    def rollback(self, from_version: str, to_version: str, reason: str, who: str) -> None:
        self.models[from_version].status = "rolled_back"
        self.models[to_version].status = "stable"
        self.rollback_logs.append(
            RollbackLog(
                reason=reason,
                who=who,
                timestamp=datetime.utcnow().isoformat(),
                from_version=from_version,
                to_version=to_version,
            )
        )


def metrics_ok(metrics: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    return (
        metrics["p99_ms"] <= thresholds["p99_ms"]
        and metrics["error_rate"] <= thresholds["error_rate"]
        and metrics["ctr_delta"] >= thresholds["ctr_delta"]
    )


def deploy_model(
    registry: Registry,
    new_version: str,
    previous_version: str,
    stage_metrics: List[Dict[str, float]],
    stage_thresholds: List[Dict[str, float]],
    who: str = "release-bot",
) -> str:
    for metrics, thresholds in zip(stage_metrics, stage_thresholds):
        if not metrics_ok(metrics, thresholds):
            registry.rollback(
                from_version=new_version,
                to_version=previous_version,
                reason=f"metrics out of SLO: {metrics}",
                who=who,
            )
            return previous_version

    registry.mark_stable(new_version)
    return new_version


registry = Registry()
registry.add_model(ModelRecord(version="v2.2.5", created_at="2026-03-01", status="stable"))
registry.add_model(ModelRecord(version="v3.0.0", created_at="2026-03-27", status="candidate", previous_stable="v2.2.5"))

stage_metrics = [
    {"p99_ms": 230, "error_rate": 0.001, "ctr_delta": 0.006},
    {"p99_ms": 266, "error_rate": 0.001, "ctr_delta": 0.004},  # 第二阶段超阈值
]
stage_thresholds = [
    {"p99_ms": 250, "error_rate": 0.002, "ctr_delta": 0.005},
    {"p99_ms": 250, "error_rate": 0.002, "ctr_delta": 0.005},
]

result = deploy_model(
    registry=registry,
    new_version="v3.0.0",
    previous_version="v2.2.5",
    stage_metrics=stage_metrics,
    stage_thresholds=stage_thresholds,
)

assert result == "v2.2.5"
assert registry.models["v3.0.0"].status == "rolled_back"
assert registry.models["v2.2.5"].status == "stable"
assert len(registry.rollback_logs) == 1
assert registry.rollback_logs[0].from_version == "v3.0.0"
```

这个例子故意让第二阶段失败，目的是说明上线逻辑必须内置“失败分支”。很多初学者写部署脚本只写 `deploy()`，没有写 `rollback()`，这相当于只设计了起飞流程，没有设计降落流程。

如果把它扩展到真实工程，常见流程大致是：

```python
def rollout(version, stages, previous_version):
    for traffic in stages:
        deploy_traffic(version=version, percent=traffic)
        metrics = collect_metrics(version=version, window="30m")
        if metrics_within_slo(metrics):
            continue
        log_rollback(reason="SLO breach", who="release-bot", timestamp=now())
        rollback(previous_version)
        return "rolled_back"
    return "completed"
```

在生产环境中，`deploy_traffic` 往往接服务网格、网关或负载均衡配置；`collect_metrics` 接 Prometheus、Datadog 或内部监控平台；`log_rollback` 则写入注册表、审计日志或事件中心。重点不是工具名，而是三个动作必须打通：

- 部署动作可自动执行
- 指标采集可自动判断
- 回滚动作可自动落库并可审计

真实工程例子可以再具体一点。假设电商搜索排序模型更新后，团队定义：

- 1% 流量跑 30 分钟
- p99 < 180ms
- 下单转化率不能下降超过 0.2%
- 实时错误率 < 0.05%

如果部署脚本在 1% 阶段检测到转化率回落 0.6%，系统应立刻：

1. 把流量切回旧版本
2. 将 `v4.1.0` 标记为 `rolled_back`
3. 记录原因、操作者、时间戳
4. 通知值班人员
5. 保留失败环境，便于复盘

这样回滚就不是“出事后临时手工改配置”，而是“系统内置的常规路径”。

---

## 工程权衡与常见坑

模型更新最大的误区，是把它只当成模型效果问题。事实上，它同时是部署问题、可观测性问题、回滚问题、流程治理问题。

第一个常见坑是没有记录回滚原因。没有回滚日志，事故复盘就会退化成口头记忆。你会知道“昨天回滚过”，但不知道是延迟超标、特征缺失、资源打满，还是实验配置写错。解决办法很直接：每次回滚都写入状态表，至少包含 `who / why / when / from / to`。

第二个常见坑是旧模型不存在。很多团队只保存最新权重，认为 Git 里有训练代码就够了。实际上回滚依赖的不是源码本身，而是完整可运行工件：模型权重、推理配置、依赖版本、特征处理逻辑、容器镜像都要可复现。否则你以为可以回退，实际要重新拼装环境，恢复时间会从分钟变成小时。

第三个常见坑是只看离线准确率。离线 AUC 很好，不代表线上一定更好。线上还要看延迟、超时、资源占用、缓存命中、业务 KPI。特别是大模型或复杂特征管道，线上瓶颈常常不是算法本身，而是特征读取和推理资源。

第四个常见坑是切流量太快。上线速度当然重要，但快不是一次性切满。流量节奏是在“风险暴露速度”和“发布效率”之间权衡。业务价值很高但监控成熟度低时，应该慢一点；系统低风险且回滚极快时，节奏可以更激进。

下面这张表总结常见坑与规避方式：

| 常见坑 | 具体表现 | 风险 | 规避行为 |
|---|---|---|---|
| 没有回滚日志 | 只知道回滚了，不知道为何回滚 | 无法复盘，重复踩坑 | 注册表记录原因、操作者、时间戳 |
| 旧版本未保留 | 旧模型镜像或权重已删除 | 回滚变慢甚至失败 | 始终保留健康待命版本 |
| 只看单一指标 | 只盯准确率，不看延迟和 KPI | 局部最优破坏整体服务 | 建立多维指标门禁 |
| 切流量过快 | 1% 后直接到 100% | 问题放大过快 | 使用固定阶段和观察窗口 |
| 版本语义混乱 | 小改也升 major，或大改只升 patch | 团队误判风险级别 | 统一版本规范并写入流程 |
| 回滚不自动 | 出问题时人工 SSH 改配置 | 恢复慢且容易二次失误 | 将回滚做成发布系统内置动作 |

还有一个容易忽略的坑是“数据恢复”。如果新模型上线同时伴随特征变更、索引重建、缓存预热、阈值重标定，那么回滚不只是切模型版本，还可能涉及恢复特征视图、重新切换数据源、清理有毒缓存。因此回滚脚本不能只写 `switch_model(old)`，而要明确配套资源是否也需要恢复。

---

## 替代方案与适用边界

蓝绿、金丝雀、滚动更新是最常见的模型发布方案，但不是所有业务都该用同一种。

蓝绿部署是“同时保留两套完整环境，再切流量”的方式。白话讲，就是新旧系统并排站好，确认新系统健康后再切过去。它的优点是切换快、回退也快，适合对恢复时间敏感的业务；缺点是资源成本高，因为两套环境要同时存在。

金丝雀发布是“先给极少量真实流量观察，再逐步增加”的方式。它最适合在线推荐、搜索、广告这类对线上反馈极其敏感的场景，因为可以边观察边放量。但它要求监控足够成熟，否则小流量阶段可能看不出异常。

滚动更新是“逐台替换实例”的方式。它常用于普通服务部署，资源成本相对低，但在模型更新场景里要特别注意版本混跑期间的行为一致性。如果模型版本对输入 schema 很敏感，滚动更新可能引入更复杂的兼容问题。

影子流量是“复制真实请求给新模型，但结果不对外生效”的方式。白话讲，就是新模型先旁听，不直接参与决策。它适合高风险行业，比如金融、医疗、风控，因为可以先看性能和行为分布，再决定是否进入真实灰度。但它不能直接验证真实业务收益，因为用户最终仍看到旧模型结果。

灰度实验或在线 A/B 是把不同版本长期暴露给不同用户群，重点比较长期业务指标。它适合优化类问题，比如推荐排序策略比较，但不适合作为唯一安全手段，因为它默认新版本已经有资格进入用户可见流量。

下面是对照表：

| 方案 | 核心做法 | 优点 | 局限 | 适用边界 |
|---|---|---|---|---|
| 蓝绿部署 | 两套环境并存，切换流量 | 回滚快，切换清晰 | 资源成本高 | 高可用要求强，能承担双环境成本 |
| 金丝雀发布 | 1% 到 100% 逐步放量 | 风险可控，适合在线优化 | 依赖高质量监控 | 推荐、搜索、广告等可渐进验证场景 |
| 滚动更新 | 逐批替换实例 | 成本较低 | 混跑期间复杂 | 版本兼容性强、基础设施成熟 |
| 影子流量 | 新模型旁路接收真实请求 | 风险低，适合敏感业务 | 不验证真实用户收益 | 金融、医疗、风控等高风险场景 |
| 灰度实验 | 用户分桶比较版本效果 | 能评估长期业务收益 | 不能替代安全发布 | 已通过基础稳定性验证后的收益比较 |

一个简化场景是金融风控模型。由于误判成本高，团队可能先让新模型在 shadow 模式接收 100% 请求，只比较延迟、资源占用、预测分布漂移和人工审核样本，再决定是否进入 canary。相比之下，资讯推荐等低风险业务，常常可以直接走蓝绿或金丝雀。

版本策略也可以和标签结合使用。例如：

- `v2.3.1-stable`
- `v2.4.0-experimental`
- `v3.0.0-rc1`

这里的 `stable`、`experimental`、`rc` 不是替代语义化版本，而是补充发布状态。这样不同团队可以共享同一套版本体系，却使用不同上线策略。研究团队可以频繁产出 `experimental`，平台团队只允许 `stable` 进入生产灰度。

最终选择什么方案，取决于三件事：

- 业务对错误的容忍窗口有多大
- 团队是否有足够的监控和自动化能力
- 是否愿意为更快回滚支付额外资源成本

没有一种方案对所有场景都最优。真正正确的做法，是让发布策略和风险等级匹配。

---

## 参考资料

1. PrepLoop，Blue-Green and Canary Deployment Patterns for Model Rollout  
   重点：包含蓝绿与金丝雀发布的步骤化说明，适合建立分阶段放量的直观认识。

2. OneUptime，MLops Model Rollback  
   重点：聚焦模型回滚机制、回滚日志和恢复流程，适合理解“为什么旧版本必须始终可用”。

3. LinkedIn Advice，What Are Best Practices for Versioning ML Models  
   重点：讨论模型版本化、语义化版本、代码与配置协同管理，适合补足版本治理视角。

4. ML Journey 相关部署经验材料  
   重点：强调线上监控、业务 KPI 与模型效果联合判断，适合理解“准确率之外还要看什么”。
