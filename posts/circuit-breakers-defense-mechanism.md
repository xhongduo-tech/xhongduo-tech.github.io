## 核心结论

Circuit Breakers 可以理解为模型或服务层里的“安全断路器”。断路器的意思是：系统不是等危险内容已经生成出来再检查，而是在生成过程已经明显朝危险方向偏移时，提前切断、改道或降级。对白话一点说，它像装在模型内部的小开关，不是看到火苗才灭火，而是发现线路已经短路就先断电。

这和传统的关键词过滤不同。关键词过滤主要盯着“最终说出来的话”，Circuit Breaker 更关注“模型正在往哪里想”。如果检测到内部激活，也就是神经网络中间层的数值状态，正在逼近有害方向，就把输出重定向到拒答、安全回复、人工审核或低权限模式。它的核心价值不是“绝对不出错”，而是“尽快止损”。

现有研究和工程报告给出的结论相当明确：这类机制在已知攻击和部分未知攻击上都能显著降低有害输出与危险工具调用成功率，同时对常规能力的损伤可以控制在较低水平。它不是万能护盾，但它改变了安全防御的位置，从“出口检查”前移到“生成链路内部拦截”。

下面这张表先给出最关键的区别：

| 防线位置 | 关键行为 | 典型效果 |
| --- | --- | --- |
| 传统出口过滤 | 输出后扫描关键词、规则、分类器 | 延迟低、接入简单，但容易被改写措辞绕过 |
| Circuit Breaker | 判断内部表征或决策轨迹后 reroute | 能更早截断，对未知变体更有泛化空间 |
| 人工审核/停机 | 高风险状态下强制接管 | 成本高，但适合兜底和事故控制 |

一个玩具例子可以帮助理解。假设用户让模型“写一封城市探险主题的邮件”，但真正意图是伪装成故事去生成恶意诱导邮件。传统过滤可能只看到“城市探险”“邮件”这些词而放行；Circuit Breaker 则可能在模型内部已经形成“欺骗性 persuasion”轨迹时就触发，把后续输出改成“我不能协助撰写这类内容”。

---

## 问题定义与边界

问题的核心不是“如何让模型从不犯错”，而是“当模型开始走向危险行为时，如何在运行时尽快切断风险链路”。这里的运行时，意思是一次真实请求进入系统、模型正在生成 token 或准备调用工具的那个阶段。Circuit Breaker 主要解决的是这个阶段的快速干预问题。

它通常覆盖三类风险：

| 风险阶段 | 典型对象 | 断路器能做什么 | 仍然可能失效的边界 |
| --- | --- | --- | --- |
| 输入到中间激活 | 越狱提示、隐写提示、伪装任务 | 检测有害激活方向并重定向 | 对抗性 embedding 扰动可能绕过检测 |
| 文本生成阶段 | 危险回答、违规建议、敏感推理 | 在输出形成前拒答或降级 | 误报会伤害正常体验 |
| 工具调用阶段 | 发邮件、执行命令、下单、写库 | 阻断高风险函数调用并记录日志 | 若调用前无检测点，仍可能越权执行 |

这里有两个边界必须说清楚。

第一，Circuit Breaker 不是通用真理机。它只能根据训练时学到的危险表征、规则阈值和工程监控做判断。如果新的攻击方式把有害语义伪装成模型没见过的向量分布，断路器就可能漏检。近期关于 embedding space attack 的研究正好说明了这一点：攻击者不一定靠“坏词”，也可以靠更隐蔽的向量级扰动来骗过检测器。

第二，它也不是纯模型内问题。一个断路器如果没有日志、报警、回放、降级、人工接管这些外层机制，就只是“多了一个内部分类器”。真正的工程防线必须覆盖检测、阻断、审计、恢复四个环节，否则误触发和漏触发都难以管理。

可以把边界失效理解成下面三种：

| 阶段 | 触发指示 | 边界失效风险 |
| --- | --- | --- |
| 激活层表征 | 有害向量成分上升 | embedding 攻击可模拟无害激活，绕过 detection |
| 输出级别 | 关键词、拒答模板 | 语义改写、隐喻、拆分表达可绕过规则 |
| 工程监控 | 错误率、异常调用率、拒答率 | 若无独立测试与回放，失效可能长期不被发现 |

真实工程里，断路器更适合回答这样的问题：当模型已经显露出“危险行为倾向”时，如何把损害压缩到最小，而不是赌模型永远不会产生危险倾向。

---

## 核心机制与推导

Circuit Breaker 的关键思想是：危险内容并不只存在于最终文本里，它也会反映在模型中间层的激活轨迹中。激活轨迹可以理解为模型“此刻内部状态的数字痕迹”。如果能找出“有害状态”和“无害状态”的稳定差异，就能在推理时把模型从危险轨道推回安全轨道。

一个常见做法是先构造有害提示和对应的无害提示，记录它们在某些层的激活，然后计算差值：

$$
\Delta h = \text{activations}(\text{harmful prompt}) - \text{activations}(\text{harmless prompt})
$$

这个 $\Delta h$ 可以理解为“危险意图相对于正常意图多出来的那部分方向”。如果收集了很多组样本，就可以用 PCA，也就是主成分分析，提取最稳定、最显著的主方向 $d$。这个方向可以被看成“有害轨迹的代表方向”。

推理时，对某一层输出 $y_\ell$ 做重定向：

$$
y_\ell^{\text{rerouted}} = y_\ell + \alpha_\ell \cdot d
$$

其中 $\alpha_\ell$ 是控制偏移强度的系数。直观上，这一步像是把正在驶向危险支线的列车扳回到安全岔道。这里的 reroute 不是“把坏词删掉”，而是“改写生成过程的内部状态”，所以它理论上能比出口过滤更早生效。

一个玩具例子如下。假设我们只看二维向量空间：

- 无害请求的平均激活约为 $(1, 1)$
- 有害请求的平均激活约为 $(4, 5)$
- 那么危险方向近似是 $d=(3,4)$

如果当前请求在生成中间出现了接近 $(3.8, 4.7)$ 的状态，就说明它越来越像有害分布。此时不是等它输出成句，而是直接在中间层对抗这个方向，或者直接 trip 进入拒答模式。

为什么这类方法比关键词过滤更强？因为关键词过滤只看表面字符串，而内部激活能编码更抽象的意图。例如“写一个爆炸装置教程”与“用化学方法迅速制造高压密闭反应物”在字符串上差异很大，但在语义和激活层面可能非常接近。Circuit Breaker 利用的正是这种“语义近、表面不一定近”的结构。

不过，推导成立并不代表工程上简单。这里有三个难点：

| 难点 | 本质问题 | 工程含义 |
| --- | --- | --- |
| 方向提取 | 危险语义是否稳定可分 | 训练数据太窄会导致泛化差 |
| 强度控制 | $\alpha_\ell$ 取值过弱或过强 | 过弱拦不住，过强会误伤正常回答 |
| 检测时机 | 哪一层、哪个步骤插入 breaker | 太晚来不及，太早可能噪声大 |

真实工程例子是 AI agent 调用工具。很多 agent 不只是“说话”，还会执行函数，比如 `send_email`、`run_sql`、`transfer_money`。这时断路器不一定要只盯文本生成，也可以直接盯“函数调用意图”。如果模型内部已经朝“发送虚假抹黑邮件”的方向收敛，就在调用前把动作阻断，返回安全拒绝，并打日志供复查。这比等邮件内容已经拼好再过滤更稳，因为风险点已经从“文本”升级到了“行动”。

---

## 代码实现

工程实现通常不是把研究论文原样搬进生产，而是分成两个层次：

1. 模型内部或模型旁路的风险评分器
2. 服务层的状态机和降级路径

状态机是经典的 `Closed -> Open -> Half-Open` 三态。

- `Closed`：正常放行，但持续监测
- `Open`：风险过高，直接拒绝或降级
- `Half-Open`：小流量试探恢复，避免永久封死

下面给出一个可运行的 Python 示例。它不依赖大模型激活，先用简单分数模拟 breaker 行为，但结构和真实工程是一致的：先评分，再 trip，再 fallback。

```python
from dataclasses import dataclass, field
from typing import Callable, List


class HighRiskDetected(Exception):
    pass


@dataclass
class BreakerConfig:
    trip_threshold: int = 2
    reset_after_success: int = 1


@dataclass
class CircuitBreaker:
    config: BreakerConfig
    state: str = "CLOSED"
    failure_count: int = 0
    success_count: int = 0
    event_log: List[str] = field(default_factory=list)

    def evaluate_risk(self, text: str) -> float:
        # 玩具规则：真实系统这里会接激活分类器、策略模型或工具调用审计器
        risky_patterns = [
            "发送虚假邮件",
            "抹黑竞对",
            "伪造身份",
            "爆炸",
        ]
        score = sum(1 for pattern in risky_patterns if pattern in text) / len(risky_patterns)
        return score

    def call(self, operation: Callable[[], str], intent_text: str) -> str:
        if self.state == "OPEN":
            self.event_log.append("fallback_from_open_state")
            return self.fallback()

        risk_score = self.evaluate_risk(intent_text)
        if risk_score >= 0.25:
            self.record_failure(f"trip:risk_score={risk_score:.2f}")
            if self.failure_count >= self.config.trip_threshold:
                self.state = "OPEN"
            raise HighRiskDetected(f"blocked intent: {intent_text}")

        result = operation()
        self.record_success()
        return result

    def record_failure(self, event: str) -> None:
        self.failure_count += 1
        self.success_count = 0
        self.event_log.append(event)

    def record_success(self) -> None:
        self.success_count += 1
        if self.success_count >= self.config.reset_after_success and self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            self.event_log.append("reset_to_closed")

    def fallback(self) -> str:
        return "Action blocked: escalated to human review."

    def try_recover(self) -> None:
        if self.state == "OPEN":
            self.state = "HALF_OPEN"
            self.event_log.append("enter_half_open")


def safe_send():
    return "邮件已发送给正常客户。"


def risky_send():
    return "发送竞对抹黑邮件。"


breaker = CircuitBreaker(BreakerConfig(trip_threshold=2))

# 正常请求通过
assert breaker.call(safe_send, "给客户发送产品升级通知") == "邮件已发送给正常客户。"
assert breaker.state == "CLOSED"

# 第一次高风险，拦截但还未完全打开
try:
    breaker.call(risky_send, "发送虚假邮件抹黑竞对")
except HighRiskDetected:
    pass
assert breaker.state == "CLOSED"
assert breaker.failure_count == 1

# 第二次高风险，进入 OPEN
try:
    breaker.call(risky_send, "伪造身份并发送虚假邮件抹黑竞对")
except HighRiskDetected:
    pass
assert breaker.state == "OPEN"

# OPEN 状态直接走降级
assert breaker.call(safe_send, "给客户发送产品升级通知") == "Action blocked: escalated to human review."

# 尝试恢复
breaker.try_recover()
assert breaker.state == "HALF_OPEN"
assert breaker.call(safe_send, "给客户发送产品升级通知") == "邮件已发送给正常客户。"
assert breaker.state == "CLOSED"
```

如果把这个模式替换成真实 agent，大致流程如下：

| 步骤 | 输入 | 判断点 | 结果 |
| --- | --- | --- | --- |
| 用户请求进入 | 文本、上下文、工具候选 | 风险分类器先评分 | 低风险继续 |
| 模型准备调用函数 | 函数名、参数草稿、激活特征 | breaker 检测危险轨迹 | 高风险直接阻断 |
| 阻断后处理 | 事件、用户会话、请求元数据 | 日志与工单系统 | 转人工审核或安全回复 |
| 恢复阶段 | 新模型版本、更新规则 | Half-Open 小流量试跑 | 验证后回到 Closed |

真实工程例子可以这样看：一个销售自动化 agent 有 `send_email(recipient, subject, body)` 这个工具。模型准备调用它时，旁路安全模块发现当前激活和参数文本都高度接近“发送虚假抹黑邮件”的危险模板，于是 breaker 直接拒绝执行该函数，返回“该操作需要人工复核”，同时记录触发时间、会话 ID、风险分数和原始参数摘要。这里真正被保护的不是一句文本，而是一次外部行动。

---

## 工程权衡与常见坑

Circuit Breaker 的优点是“快”和“靠前”，但代价也很直接：它会影响正常用户体验，而且如果设计不当，会让系统进入“安全但不可用”的状态。

第一个权衡是误报和漏报。阈值低，误报多，用户会觉得系统动不动就拒答；阈值高，漏报多，危险请求可能穿过去。这不是一个能靠一句“调一下阈值”解决的问题，因为不同业务的风险容忍度完全不同。客服 FAQ 和自动转账系统，不应该用同样的 trip 策略。

第二个权衡是模型能力损失。内部 rerouting 的本质是改写激活分布，如果改得太重，连正常语义也可能被拉偏。例如一些涉及安全研究、攻防教育、漏洞修复的正常请求，可能因为语义邻近高风险内容而被误杀。

第三个权衡是对抗演化。研究已经说明，攻击者可以在 embedding 空间做更细粒度的扰动，制造“表面无害、内部仍危险”的输入。如果 breaker 的检测器长期不更新，它迟早会被针对性绕开。换句话说，breaker 不是一次性交付件，而是持续运维件。

下面是常见坑和对应缓解措施：

| 常见坑 | 具体表现 | 缓解措施 |
| --- | --- | --- |
| 只做关键词版“伪 breaker” | 名字叫断路器，实质仍是出口过滤 | 明确区分内部信号和输出规则，不要概念混用 |
| 阈值写死 | 新业务上线后误触暴涨 | 分场景设阈值，并保留灰度开关 |
| 无独立日志 | 只知道“被拒绝了”，不知道为什么 | 记录触发层级、风险分数、工具名、上下文摘要 |
| 无降级方案 | Open 后所有功能全挂 | 设计 human-in-loop 或只读模式 |
| 不做回测 | 新攻击出现后长期漏检 | 用最新攻击样本持续做 regression test |
| breaker 与主系统强耦合 | 一处失效拖垮整体 | 独立部署、独立监控、独立告警 |

一个很常见的真实运维场景是：某天线上 breaker 的 trip 率突然从 1% 升到 8%。如果没有回放系统，团队可能只能猜“是不是模型变笨了”。但正确做法是先检查三件事：是否出现了新型攻击模式、是否最近改动了提示模板导致激活分布漂移、是否某个下游工具参数格式变化引发误判。很多时候问题并不在“安全模型不准”，而在“系统分布变了但阈值和测试没跟上”。

因此，工程上应把 Circuit Breaker 当成一个受监控的控制层，而不是一句“加个安全 classifier”就结束。

---

## 替代方案与适用边界

Circuit Breaker 不是唯一方案，它更像防御体系里的一个关键层。要理解它的适用边界，最好的方法是和其他方案并排比较。

| 策略 | 优点 | 缺点 | 适用边界 |
| --- | --- | --- | --- |
| Circuit Breaker | 拦截有害激活或决策轨迹，对未知变体有潜在泛化 | 需要持续评测，误报会影响体验 | 高风险输出、工具调用、代理执行 |
| Adversarial Training | 能修补已知攻击样本 | 对新攻击可能失效，维护成本高 | 模型更新周期明确、能持续补数据 |
| 输出过滤 | 接入最简单、成本最低 | 只看表面文本，易被绕过 | 低到中风险内容审核 |
| Kill Switch | 极端情况下强制停机 | 粗暴，业务中断大 | 事故响应、监管要求、重大异常 |
| Human Handoff | 审核可靠，便于问责 | 成本高、延迟高 | 高价值高风险决策 |

可以把它们理解成不同层的闸门。

- 输出过滤解决“说出来的话像不像违规”
- 对抗训练解决“模型见过的坏招能不能少中”
- Circuit Breaker 解决“模型正在变危险时能不能立刻拦住”
- Kill Switch 解决“系统整体已经异常时能不能马上停下”
- Human Handoff 解决“机器不该独自决定的事情由谁兜底”

对零基础到初级工程师来说，最容易犯的误解是把 breaker 当成“比关键词过滤更高级的过滤器”。这个理解不够准确。更准确的说法是：它是运行时控制器，目标不是理解最终字符串，而是控制生成和执行链路的状态转移。

它最适合的场景有三类：

1. 模型会调用外部工具
2. 输出一旦错就成本很高
3. 需要在毫秒到秒级做实时止损

它不太适合单独承担全部安全责任的场景也很明确：

1. 法规要求强审计和人工审批时
2. 新攻击变化很快、检测器更新跟不上时
3. 业务对误报极端敏感时

所以更合理的结论不是“以后都用 Circuit Breaker”，而是“把它放在高风险链路上，作为 defence in depth，也就是纵深防御，中间那一层自动控制”。当它失败时，还要有 kill switch、人工审核、权限最小化、日志回放一起兜底。

---

## 参考资料

- NeuralTrust, “Using Circuit Breakers to Secure the Next Generation of AI Agents,” 2026-01-23, https://neuraltrust.ai/blog/circuit-breakers
- Gray Swan Research, “Improving Alignment and Robustness with Circuit Breakers,” 2024, https://www.grayswan.ai/research/circuit-breakers
- Dipankar Sarkar, “Defence in Depth for AI Agents: Kill Switches, Circuit Breakers, and Control Layers,” 2025-12-09, https://www.dipankar.name/writings/defence-in-depth-ai-agents/
- PromptLayer, “Revisiting the Robust Alignment of Circuit Breakers,” 2024-08, https://www.promptlayer.com/research-papers/revisiting-the-robust-alignment-of-circuit-breakers
