## 核心结论

AI 项目的 PR 审查，不是检查“这段代码能不能运行”，而是检查“这次改动会不会让系统在某一类输入、某一类用户、某一条链路上失控”。这里的“PR”是 pull request，也就是一次待合并的代码或配置改动集合。对 AI 项目来说，PR 里的有效改动对象通常不只包括代码，还包括提示词、模型版本、索引、特征、评测集、权限配置和上线策略。

一个直接可用的总框架是风险函数：

$$
R(PR)=\sum_i w_i \cdot p_i \cdot (1-c_i)
$$

其中，$w_i$ 是风险后果的严重度，白话说就是“出事了有多严重”；$p_i$ 是发生概率，也就是“这次改动把问题带进来的可能性”；$c_i$ 是覆盖率，也就是“测试、评测、监控、回滚能挡住多少”。覆盖率越高，剩余风险越低。

审查结论最好分成两层：

| 类型 | 含义 | 典型例子 | 处理方式 |
|---|---|---|---|
| `hard gate` | 硬门槛，不满足就不能合并 | 数据泄漏、权限扩大、无回滚、无批准数据源约束 | 直接阻断 |
| `soft score` | 软评分，会影响优先级和质量，但不一定阻断 | 命名、注释、代码重复、文档不清 | 记录后修复或跟进 |

玩具例子：一个 PR 看起来只是把客服机器人的 system prompt 从“尽量拒答不确定问题”改成“尽量给出完整回答”。如果 reviewer 只把它当成文案优化，就会漏掉真正风险：拒答策略改变了，模型在敏感问题上的放行率可能上升。这类改动即使一行代码没变，也可能比一次普通重构更危险。

真实工程例子：某个 RAG 客服系统的 PR 同时改了提示词和检索索引。RAG 是 retrieval-augmented generation，白话说就是“先查资料，再让模型生成答案”。审查重点不该停留在 prompt 语气是否更自然，而应检查：检索是否仍只命中批准知识源、拒答边界是否变化、是否有 prompt injection 防护、灰度和回滚是否可执行。真正的审查目标是系统稳定性，而不是局部代码整洁度。

---

## 问题定义与边界

AI PR 的特殊性在于，错误往往不是单层错误，而是跨层耦合错误。所谓“耦合”，白话说就是多个环节互相依赖，一个地方改了，另一个地方会跟着变。传统后端 PR 多数集中在函数逻辑、接口兼容性、异常处理；AI PR 还要额外看输入从哪来、知识从哪来、模型如何生成、评测如何验证、上线后如何监控。

建议先把审查边界拆清楚：

| 层级 | 审查对象 | 典型风险 | reviewer 应问的问题 |
|---|---|---|---|
| 代码层 | 业务逻辑、接口、异常处理 | 分支缺陷、并发问题、回归 bug | 改动是否破坏原有调用契约 |
| 数据层 | 训练集、知识库、索引、特征 | 脏数据、未批准数据源、分布漂移 | 数据来源是否可追踪、可批准 |
| 提示词层 | system prompt、模板、工具调用指令 | 拒答边界变化、注入脆弱性 | prompt 是否扩大行为边界 |
| 评测层 | 基准集、切片集、回归集 | 平均分掩盖局部退化 | 是否覆盖高风险切片 |
| 服务层 | API、流量分桶、缓存、降级 | 线上行为与离线不一致 | 失败时怎么降级 |
| 运维层 | 监控、告警、灰度、回滚 | 出问题无法发现或无法撤回 | 是否能快速定位和止损 |

再把问题分成阻断类和排序类：

| 结论类型 | 属于什么问题 | 示例 |
|---|---|---|
| `hard gate` | 安全、隐私、权限、数据源合法性、无回滚 | 检索索引混入未批准文档 |
| `soft score` | 可读性、局部复杂度、注释、命名一致性 | prompt 模板命名不统一 |

玩具例子：一个小型问答机器人，把回答模板从“若不确定则返回不知道”改成“结合上下文尽量回答”。代码层可能完全通过，单元测试也全绿，但提示词层发生了边界漂移。这里 reviewer 如果只看函数，就会把高风险改动误判成低风险改动。

真实工程例子：一个 RAG PR 只改了索引构建脚本，代码看起来只是“新增一个数据目录”。但上线后发现新目录包含未经批准的内部文档，导致回答引用来源失控。这不是简单的数据 bug，而是数据层、权限层、服务层共同失守。结论是：reviewer 必须明确自己在审哪一层，不能把“模型效果差”“检索污染”“上线无监控”混成一句“这 PR 看起来问题不大”。

---

## 核心机制与推导

风险拆分比主观印象更稳定。原因很简单：人类 reviewer 很容易高估自己熟悉的代码风险，低估自己不熟悉的数据和提示词风险。把每个风险项拆成严重度、概率、覆盖率，能让讨论从“我感觉没问题”变成“哪一层没有被验证”。

最小数值例子如下。假设某个 PR 同时改了 prompt 和检索索引，团队设定阈值 $T=3$，超过就不能合并。

| 风险项 | 严重度 $w_i$ | 概率 $p_i$ | 覆盖率 $c_i$ | 风险值 $w_i p_i (1-c_i)$ | 是否阻断 |
|---|---:|---:|---:|---:|---|
| 数据泄漏 | 5 | 0.3 | 0.1 | 1.35 | 是 |
| 切片评测缺失 | 4 | 0.5 | 0.2 | 1.60 | 否，但接近阻断 |
| 回滚缺失 | 3 | 0.4 | 0.5 | 0.60 | 是 |

总风险：

$$
R = 1.35 + 1.60 + 0.60 = 3.55
$$

因为 $R > T$，这次 PR 不该直接合并。更重要的是，即使团队不采用这个公式，光看风险结构也足够得出结论：数据泄漏和无回滚已经触碰 `hard gate`，所以必须先补。

这个框架成立的原因，是它抓住了 AI PR 的三个本质约束：

1. 严重度不对称。泄漏用户数据和命名不统一，后果不是一个量级。
2. 概率不可忽略。AI 系统很多错误不是必现，而是“在某类输入上才触发”。
3. 覆盖率决定你有没有底气。离线分数高，不等于高风险切片被覆盖。

玩具例子：一个文本分类器 PR 把阈值从 0.8 降到 0.6，平均召回率上升，但敏感类别误报率显著提高。如果没有专门的敏感切片评测，$c_i$ 实际接近 0，剩余风险就很高。

真实工程例子：一个企业知识助手切换 embedding 模型并重建索引。embedding 是把文本映射成向量，白话说就是“把句子变成可计算相似度的数字表示”。如果 reviewer 只看离线平均命中率，可能会通过；但如果高权限文档与普通文档分桶策略变了，权限穿透风险的严重度极高，应该直接进入 `hard gate` 检查。

---

## 代码实现

工程上最有效的方法，不是写一个复杂审查平台，而是把关键检查项变成重复执行的流程。最小实现通常包括三部分：PR checklist、自动化校验、人工结论模板。

一个简化的 PR checklist 可以这样写：

```text
- 是否改动了数据、索引、模型版本、prompt 或权限配置
- 是否说明了影响链路：检索、生成、评测、上线
- 是否补充了切片评测，而不只提供平均指标
- 是否验证仅命中批准数据源
- 是否检查 prompt injection 防护与拒答策略变化
- 是否具备灰度发布、监控、告警和回滚方案
- 是否记录了版本号、数据快照或配置 hash
```

下面是一个可运行的 `python` 玩具实现，把 `hard gate` 和风险阈值合并成一个判定器：

```python
from dataclasses import dataclass

@dataclass
class RiskItem:
    name: str
    severity: float
    probability: float
    coverage: float
    hard_gate: bool = False

    def score(self) -> float:
        return self.severity * self.probability * (1 - self.coverage)

def evaluate_pr(risks, threshold=3.0):
    total = sum(r.score() for r in risks)
    blocked = any(r.hard_gate and r.score() > 0 for r in risks)
    return {
        "total_risk": round(total, 2),
        "blocked": blocked or total > threshold,
    }

risks = [
    RiskItem("data_leak", severity=5, probability=0.3, coverage=0.1, hard_gate=True),
    RiskItem("slice_eval_missing", severity=4, probability=0.5, coverage=0.2),
    RiskItem("rollback_missing", severity=3, probability=0.4, coverage=0.5, hard_gate=True),
]

result = evaluate_pr(risks, threshold=3.0)
assert abs(result["total_risk"] - 3.55) < 1e-9
assert result["blocked"] is True
print(result)
```

这个例子不复杂，但它体现了一个关键原则：把“经验判断”编码成“可重复判断”。哪怕最后仍要人工拍板，至少 reviewer 先被提醒去看真正高风险的地方。

真实工程例子：RAG PR 可以在 CI 中加三个自动检查。
1. 检查索引元数据，只允许白名单数据源进入构建。
2. 跑高风险切片评测，例如敏感问答、权限边界问答、注入样例。
3. 检查是否存在回滚配置，例如旧索引版本号、旧 prompt 版本号、流量开关。

这类实现未必要大而全，但必须可执行。否则 checklist 很容易退化成形式主义。

---

## 工程权衡与常见坑

AI PR 审查的第一大陷阱，是只看平均指标。平均值会把长尾问题抹平，而真实线上事故往往来自长尾。所谓“长尾”，白话说就是“总量不大，但一旦命中后果很重的那一小部分输入”。

真实工程例子：离线总准确率从 91% 提升到 93%，看起来很好；但法务敏感问题切片从 88% 降到 71%，结果线上投诉明显增加。平均分掩盖了高风险局部退化，这类 PR 应该被卡住，而不是被“总体提升”放行。

第二大陷阱，是把数据、特征、提示词、索引当成“外部资源”，不做版本控制。结果是 reviewer 只能看到代码 diff，却看不到真正改变系统行为的东西。

常见坑可以直接整理成表：

| 常见坑 | 造成的后果 | 规避方式 |
|---|---|---|
| 只看平均指标 | 高风险切片退化被掩盖 | 强制提交切片评测结果 |
| 只看离线，不看线上 | 训练/服务偏斜无法发现 | 加灰度、监控、告警 |
| 数据和索引不做版本记录 | 无法复现问题和回滚 | 记录版本号、快照、hash |
| 不检查权限边界 | 检索到未授权内容 | 白名单数据源和访问控制 |
| 评测集泄漏 | 离线结果虚高 | 分离训练、验证、回归数据 |
| 过度抽象 | 审查成本变高，真实风险被埋 | 先用最小可审查结构 |

玩具例子：一个 prompt 模板系统为了“未来扩展”，提前抽象出五层继承和十几个可插拔策略。结果 reviewer 很难一眼看出实际生效的 prompt 是什么，风险反而更高。这里的坑不是“代码高级不高级”，而是“可审查性下降”。

还有一个常见误区：把 reviewer 的责任理解成“证明没问题”。正确做法是“优先发现不可接受的问题”。因为 AI 系统输入空间太大，任何 reviewer 都不可能证明系统在所有输入上都正确。

---

## 替代方案与适用边界

不是所有 AI 项目都需要同样重的审查流程。审查强度应该跟改动范围和风险边界匹配，而不是一刀切。核心判断标准是：这次改动是否触碰数据、模型、索引、权限、上线策略这些高风险边界。

| 审查档位 | 适用场景 | 重点检查 |
|---|---|---|
| 轻量审查 | 纯 UI 文案、样式调整、非行为型日志改动 | 可读性、兼容性、是否误伤展示层 |
| 标准审查 | 普通业务逻辑、非敏感 prompt 微调、常规接口改动 | 单测、回归、切片评测、监控说明 |
| 高风险审查 | 数据源变更、索引重建、模型切换、权限改动、拒答策略改动 | `hard gate` 全量检查、灰度、回滚、批准记录 |

玩具例子：把按钮文案从“提交”改成“发送”，属于轻量审查；把 system prompt 从“未知时拒答”改成“尽量补全答案”，即使只改几行文本，也应至少进入标准审查，很多场景下应直接按高风险审查处理。

真实工程例子：一个客服系统 PR 同时改 UI 提示文案和索引配置。前者主要影响体验，后者影响答案来源和安全边界。两部分不能按同一标准审。最稳妥的做法是拆 PR：展示层改动走轻量审查，索引改动走高风险审查。这样 reviewer 的注意力不会被混淆，回滚也更清晰。

替代方案上，如果团队还很小，没有完善 CI，可以先从“人工 checklist + 强制评测附件 + 明确回滚说明”开始；如果团队成熟，再逐步把白名单校验、切片评测、风险打分放进自动化。关键不是工具多，而是底线明确：一旦涉及高风险边界，就不能只按普通代码风格审查。

---

## 参考资料

1. [What to look for in a code review](https://google.github.io/eng-practices/review/reviewer/looking-for.html)
2. [The Standard of Code Review](https://google.github.io/eng-practices/review/reviewer/standard.html)
3. [The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)
4. [Data Validation for Machine Learning](https://research.google/pubs/data-validation-for-machine-learning/)
5. [Model Cards for Model Reporting](https://research.google/pubs/model-cards-for-model-reporting/)
6. [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml/)
7. [Hidden Technical Debt in Machine Learning Systems](https://research.google/pubs/hidden-technical-debt-in-machine-learning-systems/)
