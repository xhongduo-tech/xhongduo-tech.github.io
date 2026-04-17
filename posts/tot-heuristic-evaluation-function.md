## 核心结论

ToT，Tree of Thoughts，直译是“思维树”，本质是把一次推理拆成多个中间节点，再用评估函数决定哪些节点值得继续展开。这里的“评估函数”可以先理解成“给候选思路打分的规则”。它不是附属模块，而是搜索质量的核心：生成能力决定你能想出多少分支，评估能力决定你能不能把预算花在对的分支上。

最常见的三类节点评估范式是 `value scoring`、`vote comparison`、`classification`。`value scoring` 是“直接打分”，给每条候选思路一个数值；`vote comparison` 是“放在一起比”，让模型从多个候选里投票选优；`classification` 是“分档判断”，例如标成 `complete / partial / impossible`。三者不是互斥关系，工程上常常组合使用。

一个新手版的玩具例子是：模型每轮先提出 5 个下一步，再用一个简短的 value prompt 给每个候选打 1 到 5 分，保留前 2 个继续扩展，其余先剪掉。这就是最小可用的 ToT。

形式化地，若某个候选思路节点记为 $v$，其启发式分数可写为：

$$
S(v)=\text{score}(\text{thought }v)\approx \text{LLM}(v \mid \text{prompt})
$$

这里“启发式”指“不是严格证明正确，而是提供足够有用的近似判断”。评估也有成本。若每轮要评估 $N$ 个候选，粗略可写成：

$$
C_{eval}\approx N\times(\text{tokens}_{\text{solution}}+\text{tokens}_{\text{eval\_CoT}})
$$

含义很直接：你评估的候选越多、评估提示越长、评估时还要写一段解释性思维链，成本就越高。ToT 的设计重点不是“把评估做得最复杂”，而是在准确率和预算之间找到可持续的平衡。

---

## 问题定义与边界

问题可以定义得很具体：给定某一层的候选思路集合 $V=\{v_1,v_2,\dots,v_n\}$，我们希望在有限预算内，选出最值得保留的子集 $K\subseteq V$，用于下一层继续搜索。输入是候选思路和上下文，输出是保留分支列表，约束是 token、延迟、调用次数以及外部工具可用性。

这件事的难点不在“评分”本身，而在“评分要足够快，还不能太瞎”。如果你完全不评估，搜索会指数膨胀；如果你评估过重，成本会反过来吞掉搜索收益；如果你评估过早、过严，又可能把潜在正确路径直接剪掉。

新手版直觉是：你草稿纸上写了两个思路，不会同时把两条都写到底，而是先判断哪个更像对的。ToT 只是把这个过程系统化了。区别在于，人能凭经验快速看出“这个方向虽然还没答案，但是有戏”；而 LLM 需要 prompt、规则或 verifier 来承担这件事。

下面这张表把问题边界压缩成一个工程视角：

| 维度 | 典型选择 | 输出形式 | 主要约束 |
|---|---|---|---|
| 输入条件 | 仅 LLM 自评 | 分数 / 投票 / 分类 | 成本低，但容易自信过高 |
| 输入条件 | LLM + 外部 verifier | 可行/违规/通过 等信号 | 需要工具链或规则系统 |
| 输出粒度 | 连续分数 | 排序最细，便于 beam search | 校准困难 |
| 输出粒度 | 投票结果 | 适合相对比较 | 丢失绝对置信度 |
| 输出粒度 | 离散分类 | 最省 token | 难以细粒度排序 |
| 预算约束 | token 优先 | 少评估、阈值粗 | 易误剪枝 |
| 预算约束 | 准确率优先 | 多评估、保留更多分支 | 成本和时延上升 |

边界也要说清楚。ToT 适合“中间状态可以被局部判断”的任务，比如数学步骤、代码修复计划、网页操作计划。它不太适合“中间状态根本无法评估”的任务，比如高度主观且没有外部反馈的开放写作，除非你愿意接受 vote 这种相对弱但可用的比较方式。

---

## 核心机制与推导

三类评估机制可以看成三种不同的信息压缩方式。

| 方法 | 典型输出 | 信息粒度 | 成本 | 更适合的任务 |
|---|---|---|---|---|
| `value scoring` | 0-1、1-5、0-100 分 | 细 | 中 | 数学、规划、代码等可局部判断任务 |
| `vote comparison` | 票数、胜者 | 中 | 中到高 | 写作、方案对比、风格或整体性判断 |
| `classification` | `sure / likely / impossible` | 粗 | 低 | 大规模快速过滤、前置剪枝 |

`value scoring` 的优点是能排序。假设每层生成分支数为 $g$，保留宽度为 $b$，搜索深度为 $d$。若每个保留节点都展开 $g$ 个候选，则单层待评估数量近似为 $N_t=b_t\times g$。总评估成本可近似写成：

$$
C_{eval}^{total}\approx \sum_{t=1}^{d} N_t \cdot c_{eval}
=\sum_{t=1}^{d}(b_t\cdot g)\cdot c_{eval}
$$

其中 $c_{eval}$ 是单个候选的平均评估成本。如果 $b_t$ 固定为 $b$，则：

$$
C_{eval}^{total}\approx d\cdot b\cdot g\cdot c_{eval}
$$

这说明 `beam width`，束宽，可理解为“每层保留几条活跃分支”，直接决定成本上限。评估函数越可靠，$b$ 就可以越小；评估函数越不稳定，$b$ 就必须放宽，否则会误杀好分支。

`vote comparison` 更像锦标赛。它不问“每个候选绝对有多好”，而问“几个候选里谁更好”。这在创意写作、摘要组织、计划完整性之类任务中很有价值，因为这类任务很难给出稳定绝对分，却比较容易做相对判断。代价是：如果你需要完整排序，投票结果往往不够细。

`classification` 最省。它把连续判断压缩成少数几个桶，例如 `complete / partial / impossible`。这很适合前置过滤：先用低成本分类去掉明显坏分支，再对剩余分支做细评分。工程上经常是“两段式”：先 `classification`，后 `value scoring`。

一个简单的搜索循环可以写成：

```text
生成候选 thoughts
-> 对每个 thought 做 value / vote / classification
-> 根据阈值或 top-k 选择保留分支
-> 继续下一层
-> 达到终止条件或预算耗尽后输出最优答案
```

玩具例子：解一个多步算式题，当前有两个候选步骤。
候选 A：先把两个大数相减，能快速缩小范围。
候选 B：先做一个看似复杂的乘法。
如果 value prompt 给 A 打 4/5、给 B 打 2/5，那么即使 A 还没得到答案，它也更可能进入下一层。这里评估的不是“最终正确性”，而是“继续展开的潜力”。

真实工程例子：在代码代理里，模型面对一个失败测试，可能生成三种修复计划。
第一种只改表面断言；
第二种追踪调用链并修复状态同步；
第三种直接绕过测试。
这时可先用 classification 判断“是否违反约束”，把第三种剪掉；再对前两种用 value scoring，优先保留能解释根因且改动面可控的计划。如果有静态分析或单元测试，它们就是外部 verifier，能把“看起来像对”变成“验证上更像对”。

---

## 代码实现

Princeton 的 ToT 实现把评估显式做成 `value` 和 `vote` 两种方法：前者独立评估状态，后者把多个状态放在一起比较。核心不是某个复杂算法，而是“候选生成 prompt”和“评估 prompt”的接口分离，这样你可以单独替换评分逻辑。

下面是一个可运行的简化 Python 版本，展示 `value scoring`、`classification` 和带权投票聚合。代码里的“评估器”先用规则模拟，真实工程里可以替换为 LLM 调用或外部 verifier。

```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Candidate:
    text: str
    value_score: float = 0.0
    label: str = "unknown"   # complete / partial / impossible

def mock_value_eval(text: str) -> float:
    score = 0.0
    if "约束" in text:
        score += 0.35
    if "验证" in text:
        score += 0.35
    if "猜" in text:
        score -= 0.40
    if "回滚" in text or "测试" in text:
        score += 0.20
    return max(0.0, min(1.0, score))

def mock_classify(text: str) -> str:
    if "绕过测试" in text or "忽略错误" in text:
        return "impossible"
    if "验证" in text or "约束" in text:
        return "complete"
    return "partial"

def weighted_vote(cands: List[Candidate]) -> Tuple[str, float]:
    # 简化版：票权 = value_score
    winner = max(cands, key=lambda c: c.value_score)
    return winner.text, winner.value_score

def select_top_k(cands: List[Candidate], k: int, threshold: float) -> List[Candidate]:
    filtered = [c for c in cands if c.label != "impossible" and c.value_score >= threshold]
    return sorted(filtered, key=lambda c: c.value_score, reverse=True)[:k]

candidates = [
    Candidate("先定位失败测试涉及的状态同步，再补验证步骤"),
    Candidate("直接猜一个常见空指针修复"),
    Candidate("绕过测试并忽略错误日志"),
]

for c in candidates:
    c.value_score = mock_value_eval(c.text)
    c.label = mock_classify(c.text)

selected = select_top_k(candidates, k=2, threshold=0.3)
winner_text, winner_score = weighted_vote(selected)

assert len(selected) == 1
assert selected[0].label == "complete"
assert winner_text == "先定位失败测试涉及的状态同步，再补验证步骤"
assert 0.0 <= winner_score <= 1.0

print("selected:", selected)
print("winner:", winner_text, winner_score)
```

这段代码体现了三个实现要点。

第一，评估 prompt 的输入不能只喂候选本身，通常还要带上 `previous thoughts + current candidate + task constraints`。否则评估器只看到一句孤立的话，很难判断它在整个搜索上下文中的价值。

第二，投票统计有两种常见写法。`simple majority` 是简单多数票，便宜、直观；`confidence-weighted` 是按置信度加权，适合已经有分数或概率输出的场景，但更依赖校准质量。

第三，外部 verifier 应该是独立模块。比如代码代理里，可以把“是否通过单测”“是否新增 lint 错误”“是否触发安全规则”做成单独检查器，再把这些结果映射回搜索分数。这样做的价值不是让 LLM 更聪明，而是让评估信号更可信。

---

## 工程权衡与常见坑

ToT 的收益来自更强的搜索，代价来自更多的调用。原始 ToT 工作在 Game of 24 上报告了明显精度提升，GitHub 仓库也说明主实验是 `b=5` 的搜索设置；附录成本分析常被二次引用为：ToT 每题成本约 \$0.74，100 次 CoT best-of 约 \$0.47，而 ToT 的成功率更高。这类数字的意义不是“越贵越好”，而是说明评估函数如果设计得当，额外预算可以换到更高质量的搜索路径。

| 常见坑 | 风险 | 缓解措施 |
|---|---|---|
| 模型自评过于自信 | 错分支被高分保留 | 引入 verifier、降低自评分权重 |
| 投票粒度过粗 | 只能选胜者，难以排序 | 投票前先过滤，投票后再补分数 |
| 分类桶太少 | `partial` 里混入大量差异 | 给 `partial` 再做二次 value 评分 |
| 评估 prompt 不含上下文 | 分数失真 | 输入中加入历史 thoughts 与约束 |
| 生成和评估用同一偏见 | 错误相互放大 | 多角色 prompt，或不同模型分工 |
| 不设预算上限 | 成本爆炸、延迟失控 | 设置 max depth、max nodes、timeout |
| 分数未校准 | 0.9 和 0.6 不具可比性 | 用历史数据做阈值标定或悲观惩罚 |

“校准”可以直白理解成“分数是否配得上实际正确率”。如果一个评估器经常把错误分支打到 0.95，它并不是不会排序，而是置信度失真。工程上常见做法是加入保守项，例如当候选缺少外部支持、解释不完整、或与历史成功模式偏离较大时，对原始分数施加一个惩罚项：

$$
S'(v)=S(v)-\lambda\cdot \text{penalty}(v)
$$

这里的 $\lambda$ 是惩罚强度。这个思路和一些 test-time search / calibration 工作相通，本质上都是避免“高分错解”主导搜索。

预算策略要提前定义。常见控制杆有四个：最大深度、每层最大候选数、评估阈值、超时终止。不要等到线上成本失控再补。对零基础到初级工程师来说，一个非常实用的经验是：先把 `classification` 做成廉价门卫，再把贵的 `value scoring` 用在少数通过初筛的分支上。

---

## 替代方案与适用边界

不是所有任务都该上 ToT。简单任务用单次 CoT，往往更便宜也更稳定。只有当问题具备“多步、易走错、可局部评估”这三个条件时，ToT 的启发式评估函数才真正有价值。

| 方法 | 准确率潜力 | 成本 | 适合任务复杂度 | 主要限制 |
|---|---|---|---|---|
| 单次 CoT | 低到中 | 低 | 低 | 容易一次走偏 |
| CoT best-of-N | 中 | 中到高 | 中 | 重采样多，但缺少结构化剪枝 |
| CoT beam search | 中到高 | 中到高 | 中到高 | 需要稳定的局部评分 |
| ToT + value | 高 | 中到高 | 高 | 分数校准要求高 |
| ToT + vote | 中到高 | 高 | 高 | 比较次数多，排序较粗 |
| ToT + classification | 中 | 低到中 | 中到高 | 信息粒度不足 |

新手版判断标准很简单。做四则运算、单轮问答、简单提取，单次 CoT 就够。解谜、多步规划、代码修复、工具使用决策，ToT 更合适。再往前一步，如果任务能被程序或规则外部验证，比如单测、数学 checker、网页 DOM 约束，那么 ToT 的评估函数会更可靠，因为它不再完全依赖模型自评。

ToT 的边界也很明确：如果令牌预算极紧、延迟要求极低、或者中间状态根本不可判，那么维护一棵树并不划算。那时直接采样多个完整答案，再用轻量 verifier 选优，往往更现实。

---

## 参考资料

- Tree of Thoughts: Deliberate Problem Solving with Large Language Models，arXiv / NeurIPS 2023，https://arxiv.org/abs/2305.10601 。引用原因：ToT 框架原始定义、Game of 24 等核心实验结果。
- Princeton ToT 官方代码仓库，GitHub，https://github.com/princeton-nlp/tree-of-thought-llm 。引用原因：`value` / `vote` 的实现接口、`b` 束宽和实验参数说明。
- Tree of Thoughts: Deliberate Problem Solving with Large Language Models，Emergent Mind，https://www.emergentmind.com/topics/tree-of-thoughts-tot-framework 。引用原因：对 ToT 状态评估与搜索流程的概括性整理。
- Henderik A. Proper 相关文章与 PDF 资料页，https://www.erikproper.eu/ 。引用原因：对 ToT 中 value、vote、classification 三类评估方式的总结性表述来源。
- What is Tree of Thoughts (ToT) Prompting? Complete Guide，ArticSledge，https://www.articsledge.com/post/tree-of-thoughts-tot-prompting 。引用原因：二次整理的成本对比表，便于说明 accuracy 与 cost 的权衡。
- Test-Time Scaling for Multistep Reasoning in Small Language Models via A* Search，OpenReview PDF，https://openreview.net/pdf/a8b02bf2994beeac053e51786824391b0842f0dc.pdf 。引用原因：展示测试时搜索中自评启发式与分数驱动扩展的相关思路。
- Thought calibration: Efficient and confident test-time scaling，ACL Anthology，https://aclanthology.org/2025.emnlp-main.722/ 。引用原因：说明“评估分数需要校准”这一工程问题，不是只看排序不看置信度。
