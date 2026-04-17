## 核心结论

Agent 的多步推理里，真正危险的不是“最后一步答错”，而是“某个中间结果先错，后续步骤都在错的前提上继续算”。中间结果验证，就是在每一步推理后插入一个 verifier。verifier 可以理解为“验收器”，专门检查当前这一步是否还能安全传给下一步。

如果把每一步都看成一次独立成功事件，整条链路的成功率不是相加，而是相乘：

$$
P=\prod_{i=1}^{N} p_i
$$

当每一步成功率都近似相同为 $p$ 时，可简化为：

$$
P=p^N
$$

这意味着多步任务天然存在指数衰减。单步只错一点，链路一长，最终成功率会急剧下降。

一个玩具例子最容易看清这件事。假设三步算式推理里，每一步未经验证的成功率是 $0.9$，那么整条链成功率是：

$$
P=0.9^3\approx 72.9\%
$$

如果在每一步后加入验证，把单步成功率提高到 $0.95$，则变成：

$$
P=0.95^3\approx 85.7\%
$$

三步任务里差距已经很明显；步数继续增加时，这个差距会被继续放大。

因此，核心结论有三条：

| 结论 | 含义 | 工程意义 |
|---|---|---|
| 错误会级联放大 | 某一步出错会污染后续状态 | 不能只检查最终答案 |
| 验证要前移到中间步 | 在每一步后做验收 | 能提前截断错误链 |
| 验证信号要多样化 | 规则、语义、工具交叉验证 | 单一 verifier 容易漏判 |

公开总结材料里，一个常见经验是：在数学推理任务中，只看最终结果时准确率大约在 45% 左右；加入中间步验证后，可提升到约 67%。代价是每一步通常增加约 200ms 的延迟。在多步、高价值任务里，这个代价通常是值得的，因为它买到的是“整条链不崩”。

---

## 问题定义与边界

这里讨论的不是“模型知识是否足够”，而是“多步推理链是否能保持状态正确”。状态，白话说，就是后续步骤默认相信的中间事实、变量值、工具输出和约束条件。

问题可以定义为：

> 在一个由 $N$ 个步骤组成的 Agent 工作流中，若每一步都依赖前一步的中间结果，则任何一个中间结果错误都可能沿链路传播，并在后续步骤中被放大；因此必须在关键步骤后验证中间状态，而不是只在最后一步验收最终答案。

若第 $i$ 步的错误率是 $p_i$，则整条链路成功率为：

$$
P=\prod_{i=1}^{N}(1-p_i)
$$

如果每一步错误率相同，都是 $p$，则：

$$
P=(1-p)^N
$$

例如 20 步工作流里，每步错误率是 5%，则：

$$
P=0.95^{20}\approx 35.85\%
$$

也就是说，就算单步看起来“还不错”，整条链最终也只有约三分之一概率完全正确。这正是多步 Agent 难以直接投入生产的根本原因之一。

可以把它类比成分布式系统里的“每跳验签”。验签，白话说，就是确认消息没有被篡改、格式合法、来源可信。如果一个服务把错误状态写进下游数据库，后面的服务即使逻辑完全正确，也只是在错误输入上继续计算。Agent 的多步推理也是同一类问题：一旦中间状态失真，后续正确推理也失去意义。

这篇文章讨论的边界主要有三类：

| 边界 | 包含 | 不包含 |
|---|---|---|
| 任务类型 | 数学推理、工具调用、工作流编排、检索后加工 | 纯单轮闲聊 |
| 验证对象 | 中间文本、结构化参数、工具结果、逻辑约束 | 训练阶段参数更新 |
| 目标 | 降低级联错误，提高整链可靠性 | 追求最低延迟 |

所以，中间结果验证更适合“链路长、代价高、状态敏感”的任务，而不是所有场景都必须上。

---

## 核心机制与推导

中间结果验证常见有三种信号源：规则验证、LLM 语义验证、工具交叉验证。它们分别解决不同类型的问题。

| 验证信号 | 作用 | 典型实现 | 擅长发现的问题 |
|---|---|---|---|
| 规则验证 | 检查格式、范围、约束 | JSON Schema、正则、数值范围、字段必填 | 参数缺失、格式错、数值越界 |
| LLM 语义验证 | 判断当前步骤在语义和逻辑上是否合理 | verifier prompt、过程奖励模型、单独审稿模型 | 推理跳步、概念混淆、因果不连贯 |
| 工具交叉验证 | 用外部可执行系统复核结果 | Python 计算、SQL 执行、检索回查、求解器 | 算错、查错、执行结果与文本不一致 |

三者互补的原因很直接。

规则验证成本最低，但只能看“形状是否对”。比如订单金额字段是否为正数、日期是否符合 `YYYY-MM-DD`。它不能判断“这个金额虽然格式合法，但语义上不合理”。

LLM 语义验证能看“意思是否通顺、结论是否接得上前文”。但它本身仍是概率模型，可能出现“看起来会讲道理，实际上判断错”的情况。

工具交叉验证最可靠，因为它把文本断言重新变成可执行事实，例如用 Python 复算、用数据库复查、用检索系统核对来源。但它覆盖面有限，只能验证那些可以外部执行或查询的部分。

因此，工程上常用做法不是三选一，而是组合打分。一个简单的融合形式可以写成：

$$
S = w_r S_r + w_l S_l + w_t S_t
$$

其中：

- $S_r$：规则验证得分
- $S_l$：LLM 语义验证得分
- $S_t$：工具交叉验证得分
- $w_r,w_l,w_t$：对应权重，且通常满足 $w_r+w_l+w_t=1$

当 $S \ge \tau$ 时，当前步骤通过；否则退回重试或改写。$\tau$ 是阈值，白话说就是“最低放行标准”。

再看一个新手能理解的例子。假设 Agent 要预订差旅行程，分三步：

1. planner 先生成机票方案  
2. verifier 检查是否超预算、时间是否冲突、机场代码是否合法  
3. executor 再去下单  

如果第二步没有做验证，planner 一旦写错日期，后续搜索、比价、下单都会围绕错误日期运行。越往后，错误越像“事实”，回滚成本也越高。

真实工程里，这个机制通常不是“一次验证就完事”，而是一个环：

1. 生成候选步骤
2. 验证当前步骤
3. 如果失败，返回错误类型和修正提示
4. 让 reasoner 基于反馈重写该步或局部回滚
5. 再验证
6. 通过后再进入下一步

这就是“生成 -> 验证 -> 精修”的闭环。它的价值不在于 verifier 永远正确，而在于把错误暴露得更早，让修正成本从“整条链重做”变成“局部返工”。

---

## 代码实现

下面给出一个最小可运行示例。它不是完整 Agent，而是一个“多步推理 + 中间验证”的骨架，重点展示三件事：

1. 每步产出后立刻验证  
2. 验证失败时返回反馈并重试  
3. 最终只接受整条链都通过的结果  

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Step:
    name: str
    value: int


@dataclass
class Verdict:
    ok: bool
    reason: str = ""


class ToyReasoner:
    def solve(self, x: int) -> List[Step]:
        # 故意制造一个错误中间步：第二步应为 x * 2，但先给出 x * 3
        return [
            Step("add_one", x + 1),
            Step("double", (x + 1) * 3),
            Step("minus_three", ((x + 1) * 3) - 3),
        ]

    def refine(self, x: int, bad_index: int, feedback: str) -> List[Step]:
        if "double" in feedback:
            corrected = [
                Step("add_one", x + 1),
                Step("double", (x + 1) * 2),
                Step("minus_three", ((x + 1) * 2) - 3),
            ]
            return corrected
        return self.solve(x)


class ToyVerifier:
    def check_step(self, x: int, steps: List[Step], i: int) -> Verdict:
        step = steps[i]
        if step.name == "add_one":
            return Verdict(step.value == x + 1, "add_one invalid")
        if step.name == "double":
            expected = steps[i - 1].value * 2
            return Verdict(step.value == expected, "double invalid")
        if step.name == "minus_three":
            expected = steps[i - 1].value - 3
            return Verdict(step.value == expected, "minus_three invalid")
        return Verdict(False, "unknown step")

    def final_ok(self, x: int, steps: List[Step]) -> bool:
        expected = ((x + 1) * 2) - 3
        return steps[-1].value == expected


def multi_step_with_verifier(x: int, max_iters: int = 3) -> Optional[List[Step]]:
    reasoner = ToyReasoner()
    verifier = ToyVerifier()

    steps = reasoner.solve(x)

    for _ in range(max_iters):
        failed: Optional[Tuple[int, str]] = None

        for i in range(len(steps)):
            verdict = verifier.check_step(x, steps, i)
            if not verdict.ok:
                failed = (i, verdict.reason)
                break

        if failed is None:
            return steps if verifier.final_ok(x, steps) else None

        bad_index, feedback = failed
        steps = reasoner.refine(x, bad_index, feedback)

    return None


result = multi_step_with_verifier(5)
assert result is not None
assert [s.value for s in result] == [6, 12, 9]
assert result[-1].value == 9
```

这个例子里，`ToyReasoner` 第一次会在 `double` 这一步故意算错，`ToyVerifier` 会在中间步把它拦下，然后 `refine` 用反馈把错误修正。

如果把这个骨架放大到真实 Agent，常见结构会变成：

```python
def run_agent(task, reasoner, verifier, tool_runner, max_retries=3):
    trace = []
    state = init_state(task)

    while not state.done:
        step = reasoner.next_step(state, trace)

        verdict = verifier.check(
            step=step,
            state=state,
            trace=trace,
            tools=tool_runner,
        )

        if verdict.ok:
            state = apply_step(state, step, tool_runner)
            trace.append((step, verdict))
            continue

        if verdict.retryable and state.retry_count < max_retries:
            step = reasoner.refine(
                state=state,
                trace=trace,
                feedback=verdict.feedback,
            )
            state.retry_count += 1
            continue

        return {"ok": False, "trace": trace, "error": verdict.feedback}

    return {"ok": True, "trace": trace, "result": state.result}
```

这类实现里，验证器通常至少返回四种信息：

| 字段 | 含义 |
|---|---|
| `ok` | 是否通过 |
| `feedback` | 为什么失败，供重试使用 |
| `retryable` | 是否值得重试 |
| `score/confidence` | 当前步骤可信度 |

真实工程例子是订单处理 Agent。它可能要完成“读取订单 -> 计算折扣 -> 校验库存 -> 生成扣减指令 -> 写库”。其中“生成扣减指令”是高风险节点，因为一旦写错，会直接污染库存数据。这里就应同时做三层验证：

1. 规则层：SKU、数量、仓库 ID 是否齐全且格式正确  
2. 语义层：扣减指令是否与订单内容一致  
3. 工具层：调用库存查询接口复核变更前后数量是否合理  

只有都过关，才允许写库。这样做的目标不是让 Agent “更聪明”，而是让系统“更不容易把错做实”。

---

## 工程权衡与常见坑

中间结果验证不是免费午餐。最直接的代价是延迟和算力。

| 验证频率 | 单步收益（准确率提升） | 新增延迟（ms） | 适用场景 |
|---|---:|---:|---|
| 全步验证 | 通常最高，可显著降低级联错误 | 200ms-1000ms/步 | 高风险、长链路任务 |
| 关键节点验证 | 中等 | 200ms-500ms/关键步 | 工具调用、写操作前后 |
| 仅最终验证 | 最低 | 接近 0-200ms | 低风险、短链任务 |

如果一个工作流有 10 步，每步都加 200ms，单是验证就多出 2 秒。在交互场景里，这会明显影响体感。因此常见策略不是“所有步骤都验”，而是“只验贵的、危险的、不确定的”。

另一个常见坑是把 self-critique 当验证。self-critique，白话说，就是模型自己评价自己。但“自己反思”不等于“自己能独立纠错”。小模型尤其容易出现一种现象：嘴上说“我前面可能错了”，接着却沿用原来的错误中间状态继续往下推。这类现象可以概括为“看起来像纠错，实际上没改错”。

常见坑可以归纳如下：

| 坑 | 表现 | 后果 | 规避方式 |
|---|---|---|---|
| 只看最终答案 | 中间步骤无人检查 | 错误链到最后才暴露 | 在关键中间步设验证门 |
| 把 self-critique 当独立验证 | 同一模型自说自话 | 错误被包装成“反思” | 让 verifier 独立于 reasoner |
| 规则过严 | 合理候选被误杀 | 重试增多、成本上升 | 区分硬约束与软约束 |
| 工具验证覆盖不足 | 只验证可执行部分 | 语义错误漏过 | 与语义验证联合使用 |
| 每步都重模型验证 | 延迟爆炸 | 用户体验差 | 采用分层验证和采样验证 |

还有一个工程上容易忽视的问题：验证器本身也会错。解决方式不是幻想 verifier 绝对正确，而是把它设计成“低成本、可解释、可组合”。例如优先让规则检查拦住明显错误，让工具检查复核可执行事实，再把 LLM verifier 放在最需要语义判断的位置。这样即使 LLM verifier 偶尔误判，也不至于单点失效。

---

## 替代方案与适用边界

并不是所有任务都值得做“全链中间结果验证”。如果任务便宜、短、可并行，那么一些替代方案更合算。

| 替代策略 | 典型场景 | 适用边界 | 延迟影响 |
|---|---|---|---|
| best-of-n | 开放式生成、多候选答案 | 能接受多次采样成本 | 中到高 |
| self-consistency | 数学题、逻辑题 | 问题可从不同路径重复求解 | 中 |
| heuristic pruning | 工作流裁剪、搜索树剪枝 | 有清晰启发式规则 | 低 |
| 仅工具校验 | 计算、数据库、代码执行 | 结果可外部执行验证 | 低到中 |
| 关键节点验证 | 写操作、交易、审批流 | 只需控制高风险步骤 | 低到中 |

这些方案和“每步验证”的区别在于，它们并不总是盯住每个中间状态。

比如 best-of-n 是先生成多个完整候选，再从中选最好的一条；self-consistency 是让模型走多条推理路径，看多数是否一致；heuristic pruning 是用启发式规则提前剪掉低质量候选。这些方法有用，但它们更像“结果层筛选”，而不是“过程层验收”。

因此，适用边界可以简单总结：

1. 如果任务很短，最终可直接校验，优先用最终验证或工具校验。  
2. 如果任务很长，且中间状态会驱动后续动作，优先用中间结果验证。  
3. 如果任务里只有少数步骤高风险，就只在关键节点设验证门。  
4. 如果任务必须高并发、低延迟，先考虑规则验证和启发式剪枝，再决定是否引入 LLM verifier。  

一个真实工程例子是客服订单流程。查询订单、读取物流、解释政策这些“只读步骤”风险相对低；但“修改库存”“发起退款”“关闭工单”这些写操作风险高。此时没必要让每一步都走重验证，而应只在状态变更前后触发多信号检查。这样可以把可靠性预算花在真正危险的位置。

所以，中间结果验证不是默认全开，而是按风险密度部署。最合理的做法通常是：

- 低成本规则检查默认全开
- 工具交叉验证放在可执行节点
- LLM 语义验证只覆盖高不确定度步骤

这比“所有步骤都上最贵的 verifier”更符合工程现实。

---

## 参考资料

- Medium, “Test-Time Scaling Part 2: The Verification Revolution”  
  https://medium.com/%40nilanshut/test-time-scaling-part-2-the-verification-revolution-cfb69882b3e5
- Emergent Mind, “Multi-LLM Verification Pipeline”  
  https://www.emergentmind.com/topics/multi-llm-verification-pipeline
- Preprints, “Reasoning and Planning …”  
  https://www.preprints.org/manuscript/202512.2242/v1
- LinkedIn, Sophie Halbeisen, “Compounding Error Problem …”  
  https://www.linkedin.com/posts/sophie-halbeisen-5449a23a_i-cant-stop-thinking-about-the-compounding-activity-7401711284502700032-NflS
- Manthan Gupta, “Agentic System Patterns That Increased Accuracy by 50%”  
  https://manthanguptaa.in/posts/agentic_systems_pattern/
- Hugging Face, “ProofCore demo README”  
  https://huggingface.co/spaces/Flamehaven/proofcore-demo/blob/main/README.md
- Emergent Mind, “When Small Models Are Right for Wrong Reasons”  
  https://www.emergentmind.com/papers/2601.00513
- Emergent Mind, “Step-by-Step Fact Verification”  
  https://www.emergentmind.com/topics/step-by-step-fact-verification
- Omniit, “Testing Agentic RAG …”  
  https://omniit.ai/blogs/testing-agentic-rag-retrieval-accuracy-source-grounded-answers-and-multi-step-workflow-assurance
