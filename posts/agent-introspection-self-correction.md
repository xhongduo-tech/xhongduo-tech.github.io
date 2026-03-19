## 核心结论

Agent 的自我纠错，核心不是“再答一次”，而是把一次回答拆成三个阶段：`错误检测 -> 原因归因 -> 修正生成`。这里的“内省”就是模型对自己刚刚产出的内容再做一次检查，而不是直接相信第一版输出。

一个对新手最实用的理解是：你写完一道题，不是立刻重写全部，而是先判断“哪里像错了”，再判断“为什么错”，最后只改高风险部分。否则，修改本身也会制造新错误。

玩具例子可以这样看：

- 检测：第一次算出 `27 + 2x = 39` 的解是 `x=5`，回代发现 `27+10=37`，不等于 39，说明答案有问题。
- 归因：不是题目理解错，而是代数移项错了。
- 修正：把 `2x=12`，改为 `x=6`。

但工程上最重要的结论不是“三阶段”本身，而是“选择性触发”。如果没有外部反馈，模型并不稳定地知道自己哪里错。ICLR 2025 的 SCoRe 论文显示，基线模型在 MATH 上从第一次尝试到第二次尝试，准确率会从 52.6% 掉到 41.4%，也就是“越改越错”；某些离线自纠错训练设置里，正确答案被改错的比例 $\Delta_{c \to i}$ 可达 19.6%。这说明“永远要求模型反思并重写”不是稳健策略。

可以把整个机制画成一条非常实用的流水线：

```text
初始回答
  ↓
检测器：一致性检查 / 置信度 / 外部验证器
  ↓
风险评分 d
  ↓
若 d <= τ：直接输出
若 d >  τ：进入归因与修正
  ↓
生成修正版
  ↓
再次校验，择优输出
```

一个最小经验法则是：

| 场景 | 是否建议纠错 | 原因 |
|---|---:|---|
| 低置信度、多个候选互相冲突 | 是 | 原答案本来就不稳定，重试有收益 |
| 高置信度、候选高度一致 | 否 | 原答案更可能已正确，继续改会引入翻转风险 |
| 可接外部验证器 | 强烈建议 | 验证器能显著降低“把对的改错” |
| 无外部反馈且任务难验证 | 保守使用 | 纯内省常抓不住真正错误 |

---

## 问题定义与边界

先把问题说清楚。这里讨论的不是“让 Agent 无限循环思考”，而是：**在没有人工即时反馈的情况下，Agent 如何判断自己刚生成的结果是否值得重写**。

“置信度”第一次出现时，可以把它理解成：模型对当前答案把握有多大。  
“一致性”第一次出现时，可以把它理解成：同一个问题用不同采样方式回答时，结论是否稳定。  
“外部验证器”第一次出现时，可以把它理解成：一个独立于主回答器的检查模块，比如单元测试、规则引擎、检索核验器、第二个模型或形式化工具。

边界要划清楚：

1. 自我纠错不等于真正确认。它只能提高发现错误的概率，不能替代外部真值。
2. 自我纠错最适合“可局部验证”的任务，比如数学步骤、代码测试、结构化抽取、表单填充。
3. 对开放式任务，例如产品方案、文风改写、抽象总结，如果没有可验证标准，内省收益通常小于结构化审稿或外部证据检索。
4. 高置信且高一致的回答，默认不应重写。因为“再改一次”本身也是一次带噪声的采样。

可以把触发边界写成工程规则：

| 条件 | 动作 | 主要风险 |
|---|---|---|
| `c` 低于阈值 | 触发纠错 | 可能多算一次，增加时延 |
| 多个候选答案不一致 | 触发纠错 | 可能出现票数高但仍错误 |
| 外部验证失败 | 强制纠错 | 需要额外工具成本 |
| `c` 高且候选一致 | 跳过纠错 | 少数高置信错误会漏掉 |
| 无法验证、任务高度开放 | 谨慎纠错 | 容易把本来合理的答案改坏 |

对新手来说，可以记成一句话：**只有在“看起来不稳”时才回头改，不要把“认真”误当成“每题重做一遍”。**

---

## 核心机制与推导

一个可落地的做法，是把“是否进入纠错”定义成一个风险分数。先给出两个信号：

- 一致性率 $s$：不同候选输出之间有多像。
- 置信度 $c$：模型自己对当前答案的把握程度。

文中给出的一个简化写法是：

$$
s=\frac{\text{最长一致子串长度}}{\text{总 token 数}}
$$

这个定义很直观：如果多个候选的关键答案段落基本一致，那么 $s$ 接近 1；如果差异很大，$s$ 就低。

再定义风险分数：

$$
d=\max(1-s, 1-c)
$$

当

$$
d>\tau
$$

时，认为当前回答“高风险”，进入纠错流程；否则直接输出。

这套写法的含义很简单：

- $1-s$ 大，表示“不同采样之间说法冲突大”。
- $1-c$ 大，表示“模型自己也没底”。
- 取 `max`，表示两者只要有一个足够危险，就值得进入检查。

为什么不能只看检测能力的“总体准确率”？因为纠错流水线真正关心的是两个更细的指标：

- 精确率 precision：判成“没问题”的答案里，真没问题的比例。
- 召回率 recall：真正有问题的答案里，被你抓出来的比例。

对自纠错来说，这两个指标分别对应两类代价：

| 指标 | 白话解释 | 代价 |
|---|---|---|
| 错误召回率 | 真错的能抓出多少 | 低了会放过错误 |
| 正确精确率 | 判为“可直接过”的答案有多可靠 | 低了会误放错误或误保留错误路径 |
| 纠错成功率 $\Delta_{i \to c}$ | 错答案被修成对答案的比例 | 低了说明重写没带来收益 |
| 翻转率 $\Delta_{c \to i}$ | 对答案被改成错答案的比例 | 高了说明策略危险 |

ACL 2025 的 S2R 给出了很有代表性的验证结果。以 Qwen2.5-Math-7B 为例，SFT 初始化下，自验证准确率约 61.58%，错误召回约 66.83%，正确精确率约 84.94%；经过 outcome-level RL 后，分别提升到约 66.49%、70.11%、87.85%。这说明检测器不是“全知全能”，但它越能稳住 precision 和 recall，后续修正才越有价值。

再看 ICLR 2025 SCoRe 的一个关键对比：

| 方法 | Accuracy@t1 | Accuracy@t2 | $\Delta_{i \to c}$ | $\Delta_{c \to i}$ |
|---|---:|---:|---:|---:|
| Base model | 52.6% | 41.4% | 4.6% | 15.8% |
| STaR + DStaR | 55.4% | 41.2% | 5.4% | 19.6% |
| STaR + D+StaR | 53.6% | 54.0% | 2.6% | 2.2% |
| Pair-SFT + DSFT | 52.4% | 54.2% | 5.4% | 3.6% |
| SCoRe | 60.0% | 64.4% | 5.8% | 1.4% |

这张表说明一个非常关键的工程事实：**纠错系统的价值，不在于它会不会改，而在于它能不能知道“什么时候别改”。**

真实工程例子可以看代码 Agent。主模型先生成补丁，检测器再做三类检查：

1. 单元测试是否失败。
2. 静态规则是否报警。
3. 多次采样出的补丁思路是否一致。

如果三者都通过，就不要再重写；如果任一失败，再进入“归因 -> 修补 -> 回归测试”。这时，Agent 的自纠错不是文学式反思，而是一个带阈值的控制系统。

---

## 代码实现

下面给一个最小可运行的 Python 版本。它不是完整 Agent，而是把“检测 -> 归因 -> 修正”的触发逻辑做成可测试模块。

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Candidate:
    text: str
    confidence: float  # 0~1

def longest_common_prefix_ratio(texts: List[str]) -> float:
    assert texts, "texts must not be empty"
    tokens_list = [t.split() for t in texts]
    min_len = min(len(tokens) for tokens in tokens_list)
    if min_len == 0:
        return 0.0

    prefix_len = 0
    for i in range(min_len):
        token = tokens_list[0][i]
        if all(tokens[i] == token for tokens in tokens_list[1:]):
            prefix_len += 1
        else:
            break

    total = max(len(tokens_list[0]), 1)
    return prefix_len / total

def risk_score(candidates: List[Candidate]) -> float:
    assert candidates, "candidates must not be empty"
    s = longest_common_prefix_ratio([c.text for c in candidates])
    c = max(0.0, min(1.0, candidates[0].confidence))
    d = max(1 - s, 1 - c)
    return d

def should_reflect(candidates: List[Candidate], tau: float = 0.45) -> bool:
    return risk_score(candidates) > tau

def diagnose_reason(candidates: List[Candidate]) -> str:
    texts = [c.text for c in candidates]
    if len(set(texts)) > 1:
        return "candidate_conflict"
    if candidates[0].confidence < 0.6:
        return "low_confidence"
    return "unclear"

def rewrite_answer(original: str, reason: str) -> str:
    if reason == "candidate_conflict":
        return f"[REWRITE: resolve conflict] {original}"
    if reason == "low_confidence":
        return f"[REWRITE: add verification] {original}"
    return original

# 玩具例子：高一致 + 高置信，不触发纠错
stable = [
    Candidate("answer is 6 because 27 plus 2x equals 39", 0.92),
    Candidate("answer is 6 because 27 plus 2x equals 39", 0.88),
]
assert should_reflect(stable, tau=0.45) is False

# 玩具例子：候选冲突，触发纠错
unstable = [
    Candidate("answer is 5 because 27 plus 2x equals 39", 0.55),
    Candidate("answer is 6 because 27 plus 2x equals 39", 0.58),
]
assert should_reflect(unstable, tau=0.45) is True
assert diagnose_reason(unstable) == "candidate_conflict"
assert rewrite_answer(unstable[0].text, diagnose_reason(unstable)).startswith("[REWRITE:")
```

这段代码的设计重点不是算法多复杂，而是模块边界清晰：

- `longest_common_prefix_ratio` 负责检测一致性。
- `risk_score` 负责合成风险分数。
- `diagnose_reason` 负责做最小归因。
- `rewrite_answer` 负责修正动作。

如果放到 Agent 系统里，伪代码通常是这样：

```python
answer = generator(prompt)
candidates = sampler(prompt, k=3)

if risk_score(candidates) > tau:
    reason = diagnose_reason(candidates)
    revised = rewriter(prompt, answer, reason)
    final = verifier_select(answer, revised)
else:
    final = answer
```

其中 `verifier_select` 的作用很关键：不是“只要重写了就用新版”，而是让原版和新版再比较一次，避免新版覆盖正确旧版。

---

## 工程权衡与常见坑

自我纠错最常见的误区，是把它理解成“多一步推理一定更稳”。实际上，它更像带副作用的重试机制。

下面这张表是工程里最容易踩的坑：

| 常见坑 | 现象 | 避险策略 |
|---|---|---|
| 过度触发 | 每次都反思，时延和翻转率一起升高 | 只在低置信或不一致时触发 |
| 内在错误盲区 | 更容易发现用户输入中的错，较难发现自己刚写的错 | 加显式自检标记，或引入外部验证器 |
| 归因过粗 | 明明是检索证据不足，却被当作推理错误重写 | 把“事实缺失 / 推理冲突 / 格式错误”拆开 |
| 修正覆盖原答案 | 新版不一定更好，却直接替换旧版 | 原版与新版做择优保留 |
| 指标只看最终准确率 | 看不到“对的被改错” | 必须单独统计 $\Delta_{c \to i}$ |
| 把开放任务硬做成自纠错 | 文风类任务不断自改，质量反而漂移 | 这类任务优先做审稿式约束，不做强纠错 |

“内在错误盲区”是近两年很值得重视的点。Self-Correction Bench 发现，模型面对同样的错误，放在“别人写的内容”里更容易识别，放在“自己刚写的内容”里反而不容易改。这个现象说明，自我纠错能力不只是推理能力，还和训练数据里“模型是否经常看到带纠错标记的轨迹”有关。

一个很便宜的小技巧，是加纠错触发词。比如在重检阶段前，补一个显式标记：

```python
def add_reflection_trigger(text: str) -> str:
    return "Wait. Re-check the previous reasoning for contradictions.\n" + text

triggered = add_reflection_trigger("The final answer is 5.")
assert triggered.startswith("Wait.")
```

根据 Self-Correction Bench 的实验，`Wait`、`But`、`However` 这类标记能明显提升模型进入“纠错模式”的概率。它不是魔法，本质上是在提示模型切换生成分布，从“继续顺着原答案往下写”切到“重新审视前文”。

但这个技巧只能提召回，不能单独解决精度问题。换句话说，它能让模型更愿意查错，不代表它查得对。所以它更适合放在“已经被检测器判高风险”的场景，而不是所有回答都强塞一个 `Wait`。

---

## 替代方案与适用边界

如果系统允许接入外部信息，纯自纠错通常不是第一选择。更稳的做法往往是“验证器 + 选择性重写”。

可以把常见方案做个横向比较：

| 方案 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| 纯自纠错 | 无工具、无检索、无真值反馈 | 接入简单，成本低 | 容易把对的改错 |
| 自纠错 + 一致性 | 能做多采样 | 对低稳答案更敏感 | 计算成本上升 |
| 自纠错 + 外部验证器 | 有规则、测试、检索、执行器 | 稳定性最好 | 需要工具链 |
| 多模型验证 + 一致性 | 高价值任务，如医疗、代码、硬件 | 误改风险更低 | 成本最高、编排复杂 |

真实工程例子可以看多模型验证流水线：

```text
主模型生成方案
  ↓
独立验证器打分
  ↓
若低分：检索/测试/第二模型复核
  ↓
仅在复核失败时重写
  ↓
最终输出
```

这类方法在材料科学、临床、RTL 验证等任务里更常见。二级验证器本质上是在回答一个更小的问题：“这份答案是否值得被相信？”这个问题通常比“直接从零生成正确答案”更容易做对。近年的一些案例显示，多阶段验证能明显降低幻觉率或提升可用精度；例如 HalluMat 报告其多阶段验证流程可将材料科学内容的幻觉率降低约 30%。这类结果的共同点不是“模型更会自省了”，而是**把纠错决定交给了更可校验的中间层**。

所以适用边界可以总结为：

1. 没有外部反馈时，用“低置信 / 不一致触发”的纯自纠错。
2. 有外部验证时，用“验证失败才重写”。
3. 高风险任务，例如医疗建议、金融决策、自动改代码，不要只依赖纯内省。
4. 如果任务本身没有清晰可验证目标，自我纠错要降级为“结构审稿”，不要伪装成真验证。

---

## 参考资料

| 标题 | 领域 | 主要贡献 | URL |
|---|---|---|---|
| When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs | 综述 | 系统总结自纠错何时有效，核心结论是纯提示式自纠错在一般任务中证据不足，外部反馈更可靠 | https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713/125177/When-Can-LLMs-Actually-Correct-Their-Own-Mistakes |
| Training Language Models to Self-Correct via Reinforcement Learning | ICLR 2025 | 提出 SCoRe；给出 `Acc@t1/Acc@t2`、$\Delta_{i \to c}$、$\Delta_{c \to i}$，说明无选择性自纠错会明显翻车 | https://proceedings.iclr.cc/paper_files/paper/2025/file/871ac99fdc5282d0301934d23945ebaa-Paper-Conference.pdf |
| S2R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning | ACL 2025 | 把自验证和自纠错联动起来，报告 verification accuracy、error recall、correct precision 等关键指标 | https://aclanthology.org/2025.acl-long.1104.pdf |
| Self-Correction Bench: Revealing and Addressing the Self-Correction Blind Spot in LLMs | 基准测试 | 提出“自我纠错盲区”，显示模型更容易纠正外部错误而非自身错误，并分析 `Wait/But/However` 触发作用 | https://arxiv.org/abs/2507.02778 |
| CorrectBench: A Benchmark of Self-Correction in LLMs | 基准测试 | 系统比较内生自纠错、工具增强纠错、微调纠错三类范式的表现与成本 | https://correctbench.github.io/ |
| HalluMat: Detecting Hallucinations in LLM-Generated Materials Science Content Through Multi-Stage Verification | 材料科学验证 | 展示多阶段验证器如何降低专业领域内容幻觉率，适合作为“外部验证优先”的工程案例 | https://arxiv.org/abs/2512.19008 |

上面的资料可以按三个方向读：

- 先读综述，建立边界：什么时候纯自纠错几乎不该被神化。
- 再读 SCoRe 和 S2R，理解检测、修正、训练信号怎么拆。
- 最后读 Self-Correction Bench 和 CorrectBench，看为什么“会推理”不等于“会自查”。
