## 核心结论

自动化测试 Agent 不是“会写测试脚本的大模型”，而是一个闭环系统：先感知，再认知，再执行，最后学习。感知，白话说就是“先把需求、代码、日志、页面状态读明白”；认知就是“决定测什么、先测谁、用什么策略”；执行就是“真的去跑脚本、调工具、收结果”；学习就是“把这次跑出来的经验回写，影响下一轮计划”。因此，它比传统脚本的关键优势不在于“更聪明”，而在于“能自我修正”。

对零基础读者，可以把它想成一个 QA 小组：一个人读需求，一个人设计场景，一个人执行测试，一个人复盘经验。自动化测试 Agent 是把这四个角色合成一套可持续运转的机器人，并且把每次失败、覆盖率、风险判断都吃进下一轮计划。

传统脚本和 Agent 的差异可以先看下面这张表：

| 传统脚本自动化 | 自动化测试 Agent |
| --- | --- |
| 输入源通常是人工写好的固定用例 | 输入源可以是需求文档、代码、日志、UI、历史缺陷 |
| 反馈通道弱，跑完多半只留一份报告 | 反馈通道强，失败日志、覆盖率、断言结果会回写知识库 |
| 应变能力低，页面一改脚本就容易碎 | 应变能力更高，可重规划、重打分、重选策略 |
| 适合稳定流程和低分支系统 | 适合高分支、高风险、需求变化快的系统 |

它为什么成立，可以用一个状态更新公式概括：

$$
T_{t+1}=T_t+f(\text{反馈}(\text{Perception},\text{Cognition},\text{Action}))
$$

其中 $T_t$ 是第 $t$ 轮测试计划。白话说就是：

“下一轮测试计划 = 上一轮计划 + 这轮执行后拿到的结果和失败笔记”。

这个公式的价值在于，它说明 Agent 不是一次性生成器，而是迭代系统。只要反馈足够真实，Agent 的计划会逐步向高风险路径、高价值场景和高失败密度区域收敛。

一个很直观的玩具例子是贷款审批流程。假设流程里有 32 个节点、6 个分支。Machines 2025 的实验里，`BankLoanCheck` 工作流在 “All States Once” 策略下达到 100% 覆盖耗时约 18.33 分钟；同一模型用 Concolic 策略约为 33.91 分钟。这说明策略不是“越高级越好”，而是不同策略在路径深度、求解成本和覆盖速度之间有稳定折衷。

真实工程里，这种能力更适合银行、发票处理、医院流程这类 RPA 场景。原因不是它们“更复杂”这么简单，而是它们同时具备三种约束：分支多、外部依赖多、真实数据不能随便用。Agent 只有把控制流、输入约束、Mock 环境和反馈学习放在一起，才可能稳定工作。

---

## 问题定义与边界

自动化测试 Agent 要解决的问题，不是“自动生成一些测试数据”，而是：如何从持续增长的需求文档、代码变更、历史日志、页面状态中，自动提取测试目标，并覆盖复杂控制流，同时让测试结果反过来修正下一轮决策。

对新手可以这样理解：你把一本需求书、一张 UI 截图、一堆失败日志交给机器人，让它自己决定该测哪些路径、造哪些输入、调用什么工具、怎样判断这次测得值不值。这个问题难，不是因为“写代码难”，而是因为输入本身就不完整、不一致、甚至互相矛盾。

一个简洁的流程图如下：

```text
需求/代码/日志/UI
        |
        v
      感知
   (提取目标)
        |
   [边界1: 文档缺失]
        v
      认知
   (规划策略)
        |
   [边界2: 风险排序失真]
        v
      执行
   (跑脚本/求解/Mock)
        |
   [边界3: 输入约束缺失]
        v
      学习
   (汇总反馈/更新知识库)
        |
   [边界4: 无反馈闭环]
        v
       回写
```

这里的边界主要有四类。

第一类是感知边界。需求写得越模糊，感知模块越容易抽错测试目标。比如“用户异常时应提示错误”这句话，如果没说明异常的定义，是输入为空、权限不足、超时，还是后端 500，Agent 很容易只测到最表面的空值校验。

第二类是控制流边界。很多系统看起来只是几个按钮，实际上内部有大量条件分支。没有控制流图，Agent 只能靠观察行为猜路径，覆盖会明显不稳定。

第三类是输入边界。缺少输入约束时，Agent 会生成“不现实”的数据。比如身份证长度不对、贷款金额超出业务范围、住院日期早于出生日期。这些输入在数学上可行，但在业务上无效，会污染测试结果。

第四类是反馈边界。没有稳定的反馈机制，Agent 只会不断“产出更多测试”，却不知道哪些测试真的带来了价值。结果就是越来越忙，但越来越偏。

因此，这类系统的定义边界很明确：它适合“输入来源多、控制流复杂、反馈能沉淀”的系统；不适合“业务规则极少、页面简单、人工脚本已足够便宜”的系统。

---

## 核心机制与推导

自动化测试 Agent 的核心不是单个模型，而是模块化链路。

感知模块负责把非结构化信息转成结构化信号。非结构化，白话说就是“人能看懂、机器直接不好用的内容”，例如需求文本、报错栈、截图、聊天记录。认知模块拿这些信号做策略选择，比如先测高风险模块，还是先追覆盖率；先做符号执行，还是先做真实输入回放。执行模块调用脚本、浏览器、符号执行器、Concolic 引擎或 Mock 服务。学习模块把结果回写知识库，重新调整下一轮优先级。

这个过程可以抽象成：

$$
T_{t+1}=T_t+\alpha R_t+\beta F_t+\gamma C_t
$$

其中：

- $R_t$ 表示失败日志和根因信号
- $F_t$ 表示断言成功/失败等执行反馈
- $C_t$ 表示覆盖率变化
- $\alpha,\beta,\gamma$ 表示不同反馈的权重

如果某个支付流程本轮失败很多，而且失败集中在“身份校验后跳转支付页”这一段，那么下一轮权重就应该更偏向该路径，而不是平均分配资源。这就是“基于反馈重排策略”。

新手版理解可以更简单：

“测试计划 = 上一次计划 +（执行结果 + 失败笔记）”。

学习模块则像共享笔记本：哪个输入组合经常触发问题，哪个页面改动后老脚本不稳定，哪个接口在 Mock 下和真实环境差异大，这些经验都会影响下次测试顺序。

下面这张表说明不同反馈会怎样改变认知模块：

| 反馈信号类型 | 对认知调整的影响 |
| --- | --- |
| 成功 | 降低该路径优先级，避免重复投入 |
| 失败 | 提高相邻路径和同类输入的优先级 |
| 覆盖率提升 | 保留当前策略权重，继续扩展 |
| 覆盖停滞 | 切换探索策略，如从脚本回放转向 Concolic |
| 断言波动 | 标记环境不稳定，增加人工确认或 Mock |
| 无效输入过多 | 收紧输入约束，补充业务边界 |

玩具例子可以用一个三步贷款检查流程：

1. 输入贷款金额
2. 判断是否超过额度
3. 判断用户信用等级

如果传统脚本只写两条用例，可能只覆盖“低额度+高信用”和“高额度+低信用”。Agent 则会把分支条件抽成约束，例如：

$$
amount \le 50000,\quad credit \in \{A,B,C\}
$$

然后根据执行结果继续探索未覆盖分支。如果发现 `credit=C` 且 `amount>30000` 总是触发失败日志，它会提升这一簇路径的优先级，而不是机械重复前两条脚本。

真实工程例子则来自 RPA。比如银行退款或医院流程自动化，经常需要跨多个系统：先读 Excel，再登录 CRM，再请求支付平台，再生成回执。这里的错误不只来自代码，还来自 UI 状态、接口响应、权限、数据格式和跨系统时序。Agent 的价值正是在于把这些反馈统一入账，而不是每一层都靠人工排查。

---

## 代码实现

落地时，最重要的不是先选模型，而是先定统一数据模型。否则感知模块输出一套字段，执行模块需要另一套字段，学习模块又写第三套字段，多 Agent 协作会很快失控。

一个最小实现里，至少要有两个统一对象：`TestPlan` 和 `Feedback`。前者描述测什么，后者描述结果如何。

模块关系可以画成这样：

```text
[Perception] -> [Cognition] -> [Executor] -> [Learning]
      |              |              |             |
 requirements      TestPlan       Feedback     KnowledgeBase
```

下面给一个可运行的 Python 玩具实现。它没有真正调用浏览器或求解器，但能说明闭环怎样成立。

```python
from dataclasses import dataclass, field

@dataclass
class TestPlan:
    targets: list[str]
    strategy: str
    priority: dict[str, int] = field(default_factory=dict)

@dataclass
class Feedback:
    passed: list[str]
    failed: list[str]
    coverage_gain: int

class Cognition:
    def build_plan(self, requirements: list[str], risk_notes: dict[str, int]) -> TestPlan:
        priority = {req: risk_notes.get(req, 1) for req in requirements}
        strategy = "concolic" if max(priority.values()) >= 3 else "all_states_once"
        return TestPlan(targets=requirements, strategy=strategy, priority=priority)

class Executor:
    def run(self, plan: TestPlan) -> Feedback:
        failed = [t for t in plan.targets if "高风险" in t]
        passed = [t for t in plan.targets if t not in failed]
        return Feedback(passed=passed, failed=failed, coverage_gain=len(plan.targets))

class KnowledgeBase:
    def __init__(self):
        self.risk_notes = {}

    def update(self, feedback: Feedback):
        for item in feedback.failed:
            self.risk_notes[item] = self.risk_notes.get(item, 1) + 1

requirements = ["登录成功", "高风险:贷款超额拒绝", "支付回执生成"]
kb = KnowledgeBase()
plan = Cognition().build_plan(requirements, kb.risk_notes)
result = Executor().run(plan)
kb.update(result)

assert "高风险:贷款超额拒绝" in result.failed
assert kb.risk_notes["高风险:贷款超额拒绝"] == 2
assert plan.strategy in {"concolic", "all_states_once"}
```

这段代码对应的流程是：

```python
plan = cognition.build_plan(requirements)
result = executor.run(plan)
knowledge_base.update(feedback=result)
```

第一步 `build_plan` 的作用，是把需求转成可执行计划，同时决定策略。第二步 `run` 的作用，是让计划进入真实执行层。第三步 `update` 的作用，是把执行结果沉淀下来，让下一轮更偏向高风险区域。

真实工程里，这套最小实现会再加三层能力。

第一层是感知增强。输入不只有需求文本，还会有 Git diff、错误日志、接口文档、截图、录屏和历史缺陷单。

第二层是执行增强。执行器不只是跑 UI 脚本，还要能切换不同策略，比如“All States Once”优先追节点覆盖，Concolic 优先沿着真实执行轨迹翻转约束，符号执行优先穷举逻辑路径。

第三层是学习增强。知识库不只是存“通过/失败”，还要存失败模式、路径特征、输入约束、外部依赖质量和人工确认记录。这样多 Agent 才能共享经验，而不是每个 Agent 都从零猜。

---

## 工程权衡与常见坑

自动化测试 Agent 的第一大权衡，是“自治程度”与“可控性”的冲突。自治越强，探索能力越好；但如果需求、控制流和输入约束不完整，它也更容易走偏。

第二大权衡，是“覆盖率”与“时间成本”的冲突。路径越深，约束越复杂，求解与执行时间越长。BankLoanCheck 的实验已经说明，不同策略的速度和路径深度并不一致，不能把所有任务都丢给同一方法。

第三大权衡，是“真实环境”与“可重复性”的冲突。真实环境更接近生产，但也更脆弱、更昂贵、更难复现；Mock 环境更稳定，却可能掩盖线上依赖问题。

常见坑和矫正措施如下：

| 坑 | 矫正措施 |
| --- | --- |
| 文档缺失，Agent 误解需求 | 增加多模态感知，联合代码、日志、UI 信息 |
| 反馈缺失，只会盲目扩张测试集 | 强制记录日志、断言、覆盖率，并回写知识库 |
| 输入失控，生成不真实数据 | 用约束求解器、字段边界、人工标注限制输入空间 |
| 外部依赖不稳定 | 使用 API Mock、数据库 Mock、沙箱环境 |
| 页面轻微变化导致误判 | 增加容错定位、结构化断言，不只靠像素或文本 |
| 多 Agent 冲突 | 统一 `TestPlan`/`Feedback` schema，避免口径不一 |

对新手来说，最容易忽略的是“人工确认环节”。很多人以为 Agent 自动化了，就应该完全无人值守。现实相反，越是高风险流程，越需要把人工确认当成保险丝。比如医院账单、银行退款、合规审批，Agent 可以自动探索和生成建议，但在关键断点上最好有人批准，或者至少有规则投票。

另一个典型问题是只靠文档。文档不全时，Agent 就像盲人摸象。正确做法不是“换一个更强模型”，而是补充观测源：读取日志、抓运行时页面、加控制流注解、导入历史缺陷，必要时让人工补最关键的业务边界。

最后一个高频坑是输入生成脱离业务。论文里反复强调注解、CSV Mock、变量边界和人机协同，其本质就是防止求解器找到“数学上合法、业务上荒谬”的输入。没有这个约束，再高的覆盖率也可能没有业务价值。

---

## 替代方案与适用边界

传统脚本自动化仍然有明确价值。对于页面结构稳定、业务分支少、规则变化慢的小型 Web 应用，用 Selenium、Playwright 或接口测试脚本往往更便宜、更直观、更容易维护。适用边界通常是：

- 测试目标稳定
- 输入空间有限
- 覆盖率要求不极端
- 人工维护脚本成本可接受

弱 Agent 也是可行替代。所谓弱 Agent，就是只负责“执行 + 简单回写”，不做复杂感知和策略学习。它适合已有成熟用例库、只是想减少脚本调度和结果汇总成本的团队。这样的系统比纯脚本更灵活，但还达不到真正的闭环优化。

相比之下，完整 Agent 更适合以下场景：

- 分支多，手工枚举路径成本高
- 需要把需求、日志、代码一起读
- 存在敏感数据，必须通过 Mock 和约束控制测试输入
- 需要多 Agent 共享经验，如同一组织内多个业务系统共用缺陷知识
- 存在复杂控制流图，适合插拔符号执行、Concolic 和约束求解

对新手可以用两个并列判断。

传统脚本：适合小型 Web 应用、固定表单、稳定回归路径。
边界清单：页面变化少、业务简单、分支少、团队能持续维护脚本。

Agent：适合银行、医院、RPA、合规流程这类高分支高约束系统。
边界清单：需要控制流图、输入约束、Mock 环境、知识共享与反馈闭环。

一个典型真实工程例子是银行 RPA。这里不仅有大量条件分支，还涉及敏感数据和跨系统调用。如果继续用纯脚本，问题通常不是“不能跑”，而是“跑得不稳、覆盖不深、结果不可信”。这时 Agent 把符号执行、Concolic、约束注解和 Mock 环境组合起来，优势才真正显现。

---

## 参考资料

1. Twinkle Joshi, Dishant Gala, *Architecting Agentic AI for Modern Software Testing: Capabilities, Foundations, and a Proposed Scalable Multi-Agent System for Automated Test Generation*  
   URL: https://www.jisem-journal.com/index.php/journal/article/view/10768  
   核心贡献：给出自动化测试 Agent 的三层架构，明确感知、认知、执行与持续学习如何协同，适合作为总体框架入门。

2. Ciprian Paduraru, Marina Cernat, Adelina-Nicoleta Staicu, *A Unified Framework for Automated Testing of Robotic Process Automation Workflows Using Symbolic and Concolic Analysis*  
   URL: https://www.mdpi.com/2075-1702/13/6/504  
   核心贡献：给出符号执行、Concolic、All States Once 三类策略在真实 RPA 工作流中的实验数据，包括 `BankLoanCheck` 等案例，适合理解策略折衷与工程落地。

3. João Vitorino et al., *Constrained adversarial learning for automated software testing: a literature review*  
   URL: https://link.springer.com/article/10.1007/s42452-025-07073-3  
   核心贡献：从综述视角讨论“约束”在自动化测试中的地位，帮助理解为什么输入边界、约束求解和搜索策略对测试有效性非常关键。

阅读顺序建议是：先看 JISEM 理解整体架构，再看 MDPI 理解实验和策略细节，最后看 Springer 的综述补足“约束驱动测试”的大背景。
