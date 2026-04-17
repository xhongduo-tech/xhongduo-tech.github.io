## 核心结论

模型服务的灰度发布，本质上是把新版本先放到少量真实流量中运行，用真实请求验证“能不能跑、跑得稳不稳、业务效果有没有变差”，再决定继续放量还是立即回滚。对模型系统来说，它不是一个“发布动作”，而是一套“受控试错机制”。

灰度发布最重要的不是流量比例本身，而是三件事同时成立：

1. 流量切分可重复。同一用户在灰度阶段尽量稳定落到同一版本，避免一次请求打到旧版、下一次请求打到新版，导致体验混乱。
2. 回滚路径可立即执行。旧版本必须保持可用，回滚不能依赖重新构建、重新拉起、重新预热。
3. 决策依据可量化。是否放量，不能靠感觉，要看基础稳定性指标、业务指标、成本指标，以及统计显著性。

一个适合初学者记忆的节奏如下：

| 阶段 | 典型流量 | 主要目标 | 决策点 |
| --- | --- | --- | --- |
| Canary | 1%～5% | 验证基础可用性 | 延迟、错误率、资源正常 |
| 增量放量 | 10%～50% | 验证业务效果 | KPI 无显著退化，成本可接受 |
| 全量切换 | 100% | 完成替换 | 旧版可下线，保留回滚窗口 |

AWS SageMaker 的 canary 文档给了一个很直观的流程：先把一部分流量切到新 fleet，再等待 baking period，例如 `WaitIntervalInSeconds=600`，也就是 10 分钟；如果 CloudWatch 告警没有触发，再把剩余流量切过去。这正是“先小范围试，再全量切换”的标准灰度流程。  
参考：[AWS SageMaker Canary](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails-blue-green-canary.html)

---

## 问题定义与边界

问题定义很简单：当你要把旧模型 $M_{old}$ 替换成新模型 $M_{new}$ 时，怎样在不让全部用户承担风险的前提下验证新模型。

这里的“灰度发布”只解决上线风险控制，不解决模型研发质量本身。也就是说，灰度能帮你降低“坏模型一下子影响所有用户”的风险，但不能替代离线评估、回放测试、shadow test。

流量切分比例通常写成：

$$
canary\_share = \frac{canary\_requests}{total\_requests}
$$

白话解释：全部请求里，有多少比例先给新版本。

玩具例子：某排序模型每天处理 10 万次请求。你先拿 1% 流量给新版本，也就是 1000 次请求。如果新版出现严重超时或明显准确率下降，最多影响这 1000 次，而不是 10 万次。这就是灰度的价值。

真实工程例子：推荐、搜索排序、广告出价、风控评分这类模型服务通常请求量大、用户路径长、业务影响直接。一次全量发布如果让延迟抬高 50ms，或者让点击率下降 1%，损失往往立刻放大到全站。因此这类系统几乎都需要灰度，而不是直接全量替换。

边界也要说清楚：

| 场景 | 是否适合灰度 |
| --- | --- |
| 在线推理服务 | 适合，核心场景 |
| 批处理离线任务 | 通常不适合，更像作业切换 |
| 强状态服务且新旧不兼容 | 有条件适合，先做兼容设计 |
| 极低流量服务 | 谨慎，样本可能不足以判断 |

---

## 核心机制与推导

灰度发布通常包含三个机制：流量切分、指标判定、自动回滚。

先看流量切分。常见有两类：

| 策略 | 做法 | 适用点 |
| --- | --- | --- |
| 百分比切流 | 网关按权重把 5% 请求路由到新版本 | 简单直接，适合基础稳定性验证 |
| 用户分桶 | 对 `user_id` 做哈希，固定部分用户进入新版本 | 保证同一用户稳定落桶，适合体验对比 |

“用户分桶”的白话解释是：把用户按一个稳定规则分组，不是每次随机抽签。常见写法是 `hash(user_id) % 100 < 5` 表示 5% 用户进入 canary。这样用户今天和明天大概率仍在同一版本里，便于观察长期行为。

再看判定指标。指标至少分两层：

1. Fast guardrails：快速护栏，意思是几秒到几分钟就能观察到的基础健康指标，比如成功率、P99 延迟、CPU、内存。
2. Slow metrics：慢指标，意思是需要更长窗口才能稳定的业务指标，比如准确率、点击率、转化率、用户满意度、单次请求成本。

为什么需要统计显著性？因为业务指标有噪声。假设旧版转化率是 10%，你看到新版是 9.8%，这未必说明新版真的更差，可能只是样本波动。只有样本量足够，差异才有解释力。

对于比例类指标，一个常见近似样本量关系是：

$$
n \propto \frac{1}{\Delta^2}
$$

其中 $\Delta$ 是你想检测的最小差异。白话解释：你想看得越细，所需样本越大。如果你只想发现“明显坏了”，几千样本可能够；如果你想识别 0.5% 的轻微下降，往往需要十万级样本甚至更多。这也是为什么 canary 常常先用 fast guardrails 控风险，再用更长窗口判断业务指标。

控制循环可以概括成：

```python
if all(fast_guardrail_pass and slow_guardrail_pass for window in last_n):
    raise_canary(weight + step)
elif consecutive_failures >= threshold:
    rollback_to(stable_version)
```

这段逻辑的重点不是语法，而是节奏：小步放量、窗口评估、连续失败再回滚，避免一次偶发抖动就误伤部署。

---

## 代码实现

下面给一个可运行的 Python 玩具实现，展示“用户分桶 + 指标判定 + 自动回滚”的最小闭环：

```python
from dataclasses import dataclass

def bucket_user(user_id: str, share_percent: int) -> str:
    # 用稳定哈希把用户固定分到 old/new
    bucket = sum(ord(c) for c in user_id) % 100
    return "new" if bucket < share_percent else "old"

@dataclass
class Metrics:
    error_rate: float      # 错误率
    p99_latency_ms: float  # P99 延迟
    accuracy: float        # 准确率
    satisfaction: float    # 用户满意度
    cost_per_1k: float     # 每千次请求成本

def should_promote(old: Metrics, new: Metrics) -> bool:
    fast_ok = new.error_rate <= 0.01 and new.p99_latency_ms <= 300
    slow_ok = (
        new.accuracy >= old.accuracy - 0.005 and
        new.satisfaction >= old.satisfaction - 0.01 and
        new.cost_per_1k <= old.cost_per_1k * 1.10
    )
    return fast_ok and slow_ok

stable = Metrics(error_rate=0.005, p99_latency_ms=220, accuracy=0.942, satisfaction=0.91, cost_per_1k=2.0)
candidate = Metrics(error_rate=0.004, p99_latency_ms=240, accuracy=0.940, satisfaction=0.905, cost_per_1k=2.1)

assert bucket_user("alice", 5) in {"old", "new"}
assert bucket_user("alice", 5) == bucket_user("alice", 5)  # 同一用户稳定落桶
assert should_promote(stable, candidate) is True

bad_candidate = Metrics(error_rate=0.03, p99_latency_ms=520, accuracy=0.938, satisfaction=0.88, cost_per_1k=2.6)
assert should_promote(stable, bad_candidate) is False
```

真实工程里，这段逻辑通常不在业务代码里手写，而是在网关、服务网格、发布平台或云厂商控制面执行。例如 AWS SageMaker 可以通过 `UpdateEndpoint` 配置 canary 流量和自动回滚：

```python
import boto3

client = boto3.client("sagemaker")

client.update_endpoint(
    EndpointName="prod-endpoint",
    EndpointConfigName="new-endpoint-config",
    DeploymentConfig={
        "BlueGreenUpdatePolicy": {
            "TrafficRoutingConfiguration": {
                "Type": "CANARY",
                "CanarySize": {"Type": "CAPACITY_PERCENT", "Value": 30},
                "WaitIntervalInSeconds": 600
            },
            "TerminationWaitInSeconds": 600,
            "MaximumExecutionTimeoutInSeconds": 1800
        },
        "AutoRollbackConfiguration": {
            "Alarms": [{"AlarmName": "p99-latency-alarm"}]
        }
    }
)
```

这里要注意，SageMaker 的 canary 是建立在 blue/green 之上的：旧 fleet 继续承接大部分流量，新 fleet 先接一小部分，观察期通过后再切换。  
参考：[AWS Blue/Green Deployments](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails-blue-green.html)

---

## 工程权衡与常见坑

灰度发布最大的误区，是把它理解成“调一个 5% 流量开关就结束”。实际上，真正的工程难点在指标体系和回滚准备。

常见坑如下：

| 坑 | 影响 | 规避方式 |
| --- | --- | --- |
| 只看延迟和错误率 | 业务指标变差后才发现 | 同时看准确率、满意度、CTR、成本 |
| 不做用户粘性 | 同一用户来回切版本 | 用用户分桶或 session stickiness |
| 旧版未保温 | 回滚时冷启动过慢 | 保持旧版运行并预热 |
| 样本不足就下结论 | 误判上线或误判回滚 | 延长观察窗口或扩大样本 |
| 不看分群指标 | 少数高价值人群受损被均值掩盖 | 按设备、地域、渠道、用户等级拆分 |

一个很典型的错误是只看 P95，不看 P99。P95 正常不代表长尾体验正常。对于在线模型服务，P99 往往直接对应最差一批真实用户的等待时间。如果模型有缓存失效、特征服务抖动、GPU 抢占、批处理队列拥堵，P99 比平均值更早暴露问题。

另一个常见问题是“业务指标和成本指标冲突”。例如新模型准确率提升了 0.3%，但推理成本上涨 40%，GPU 利用率逼近上限。此时不能只看效果，要看单位收益是否值得。灰度的意义之一，就是让你在真实成本曲线上做判断，而不是只在离线实验里乐观估计。

---

## 替代方案与适用边界

灰度发布不是唯一方案，常见替代方式还有蓝绿部署和 A/B 测试。

| 策略 | 流量方式 | 主要目标 | 适用边界 |
| --- | --- | --- | --- |
| Canary | 逐步放量 | 控上线风险 | 高频在线服务 |
| Blue/Green | 0%/100% 切换 | 秒级切回旧版 | 核心接口、强稳定性要求 |
| A/B 测试 | 长时间并行分流 | 比较业务效果 | 业务决策、实验平台成熟场景 |

三者关系不要混淆：

1. 蓝绿部署解决“快速切换环境”。
2. Canary 解决“逐步暴露风险”。
3. A/B 测试解决“验证业务效果差异是否成立”。

很多成熟团队会把三者组合使用：先 shadow test，再 1% canary 验证系统稳定，再 10%～50% 做带统计检验的 A/B，最后全量切换。这样既控制技术风险，也控制业务风险。

适用边界也很明确：如果你的服务流量极低，灰度的统计价值会下降；如果你的模型依赖强状态且新旧版本特征不兼容，必须先做协议兼容和数据双写，否则灰度本身就会制造额外故障。

---

## 参考资料

- [AWS SageMaker: Use canary traffic shifting](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails-blue-green-canary.html)
- [AWS SageMaker: Blue/Green Deployments](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails-blue-green.html)
- [AWS SageMaker: Deployment guardrails for updating models in production](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails.html)
- [SysDesAi: Canary Release](https://www.sysdesai.com/learn/deployment-operations/canary-release)
- [SystemOverflow: ML Specific Guardrails and Metrics in Canary Analysis](https://www.systemoverflow.com/learn/ml-infrastructure-mlops/automated-rollback/ml-specific-guardrails-and-metrics-in-canary-analysis)
- [SystemOverflow: Implementing the Canary Control Loop](https://www.systemoverflow.com/learn/ml-infrastructure-mlops/automated-rollback/implementing-the-canary-control-loop)
- [SystemOverflow: Canary Deployments and Automated Rollback for ML Models](https://www.systemoverflow.com/learn/ml-monitoring-observability/model-performance-degradation/canary-deployments-and-automated-rollback-for-ml-models)
- [Netdata: What Is Canary Deployment?](https://www.netdata.cloud/academy/canary-deployment/)
