## 核心结论

Test-Time Compute，直白说，就是**模型上线后，在回答当前这一题时额外多花一些算力**。它不改参数，不重新训练，而是在推理阶段增加思考长度、采样次数、候选重排或搜索深度，换取更高的正确率。

它重要的原因是：训练时扩展，也就是继续堆参数、堆数据、堆训练 FLOPs，长期看会出现收益递减；但很多难题在推理阶段仍然存在“再想一会就能答对”的空间。于是，工程上出现了第二条路：**模型不一定更大，但每道难题可以允许它算得更久一点**。

这条路线的核心结论可以压缩成三句：

| 方法 | 额外推理算力怎么花 | 典型收益形态 | 主要代价 |
|---|---|---:|---:|
| 单样本 | 只生成一次 | 基线 | 最低延迟 |
| Self-Consistency | 生成 $N$ 条推理链后投票 | 前期提升明显，随后饱和 | 成本近似线性随 $N$ 增长 |
| Best-of-N | 生成 $N$ 个候选后用奖励模型选最优 | 提升常见，但边际收益递减 | 易被奖励模型“带偏” |
| 搜索式方法（如 MCTS + PRM） | 在中间步骤做树搜索与过程打分 | 对复杂推理题更强 | 实现复杂、延迟更高 |

第一，**多次采样不是“重复浪费”**。如果模型在同一道题上有时能答对、有时会走偏，那么多采几次再聚合，就能把“偶然答对”的概率放大成“较稳定答对”。

第二，**推理时算力扩展的收益不是无限的**。Self-Consistency 的收益通常先快后慢；Best-of-N 常见经验是随 $N$ 增长继续变好，但接近对数收益，而不是线性收益。

第三，**更好的策略不是一刀切地让每题都采样 64 次**。更合理的是先估计题目难度，再动态分配预算。简单题尽快早停，困难题才启用更多候选、投票和搜索。这就是 compute-optimal 的核心思想：**固定总预算下，把算力投给最需要的题和最有效的步骤**。

一个直接的公开例子是 OpenAI 对 o1 的展示：AIME 2024 上，单样本、64 条共识、再到大规模重排，效果逐级上升。这说明额外推理算力确实能把同一个模型的上限继续往上推。

---

## 问题定义与边界

先给定义。

**Test-Time Compute**：模型推理时额外消耗的计算资源。白话说，就是**回答当前这道题时临时加的“思考预算”**。它通常表现为下面四类动作：

| 形式 | 具体做法 | 直白解释 |
|---|---|---|
| 更长推理链 | 让模型写出更多中间步骤 | 让它别急着下结论 |
| 多样本采样 | 同一道题生成多份答案 | 允许它“多做几遍” |
| 候选重排 | 先生成多个候选，再挑最好的 | 先广撒网，再精选 |
| 搜索树 | 在中间状态上继续展开分支 | 不只看终点，还看路径 |

这和训练时扩展的边界不同。

**训练时计算扩展**，白话说，就是**在模型出厂前多投入训练资源**。它是“长期打基础”；而 Test-Time Compute 是“考试现场多做几步草稿”。两者都能提升能力，但适用边界不同：

| 维度 | 训练时扩展 | 推理时扩展 |
|---|---|---|
| 发生阶段 | 训练前或训练中 | 线上推理时 |
| 是否改参数 | 是 | 否 |
| 成本分布 | 一次性大投入 | 按请求逐次付费 |
| 调度方式 | 所有样本统一 | 可按题目难度动态分配 |
| 主要风险 | 训练成本高、迭代慢 | 延迟升高、调度复杂 |

因此，Test-Time Compute 的问题定义不是“如何让模型一直多想”，而是：

$$
\text{在固定线上预算下，如何把额外计算分配给最值得分配的请求与步骤？}
$$

把这个式子拆开看，重点有两层：

1. **固定线上预算**：线上 GPU、时延、吞吐量都有限，预算不是无限池子。
2. **最值得分配**：不是所有请求都值同样多的算力，真正关键的是“挑题花钱”。

这个定义自带几个边界。

第一，它主要适用于**困难但可验证的任务**。例如数学题、代码题、逻辑推理题。这类任务往往有清晰答案，或者至少能用过程奖励模型、单元测试、规则检查器去打分。相反，纯开放式写作、审美判断这类任务，很难通过额外采样稳定换来确定性收益。

第二，它不是“零成本提升”。线上系统中，额外采样意味着更多 token、更长尾延迟、更高 GPU 占用，甚至更复杂的请求调度。

第三，它不是“只要 $N$ 足够大就一定更好”。如果奖励模型本身不可靠，Best-of-N 会出现 reward hacking。白话说，就是**模型学会讨好打分器，而不是更接近真实正确答案**。如果多数投票里的样本高度相关，Self-Consistency 也会失效，因为“错得很一致”仍然是错。

一个玩具例子可以先看清边界。

题目：`17 × 19 = ?`

假设某模型单次作答时：
- 60% 概率答对 `323`
- 25% 概率答成 `333`
- 15% 概率答成 `313`

如果你只采样 1 次，正确率就是 60%。如果采样 5 次并多数投票，正确答案更可能成为票数最多的那个结果。这个提升来自“重复独立试验后的聚合”。

但如果模型在这个题上 95% 都答 `333`，只是偶尔碰巧答对，那么多投票几乎没有意义。于是问题核心就变成：**你要先知道这题值不值得多算**。

一个真实工程例子是客服或代码助手系统。简单问题如“接口地址是什么”，直接单样本即可；复杂问题如“为什么这个并发 bug 只在生产环境出现”，就更适合启用多候选分析、基于日志证据的重排，甚至分步骤搜索。否则所有请求都走重型推理，系统成本会迅速失控。

为了让边界更清楚，可以把“适合做 Test-Time Compute 的任务”压缩成一个表：

| 条件 | 是否重要 | 原因 |
|---|---|---|
| 任务较难 | 高 | 太简单的题没必要追加预算 |
| 结果可验证 | 高 | 没有验证信号，重排和搜索容易失真 |
| 单题价值高 | 中到高 | 否则额外成本不划算 |
| 可容忍延迟 | 高 | 在线预算最终要落到时延 SLA 上 |
| 错误路径多样 | 中 | 如果总是同一种错，多采样帮助有限 |

---

## 核心机制与推导

这一节只讲三个最常见机制：Self-Consistency、Best-of-N、搜索式推理。

### 1. Self-Consistency：多次独立作答，再做聚合

**Self-Consistency**，白话说，就是**同一道题让模型独立做多遍，最后按答案投票**。它依赖一个前提：模型并不是永远错在同一个地方，而是存在“有时走对路径”的概率。

如果一次采样答对的概率是 $p$，采样 $N$ 次并做多数投票，那么最终正确率近似等于“二项分布中正确票数超过一半”的概率：

$$
P_{\text{SC}}(N, p) = \sum_{k=\lceil N/2 \rceil}^{N} \binom{N}{k} p^k (1-p)^{N-k}
$$

式子里每一项的含义是：
- $N$：总共采样多少次
- $k$：其中答对了多少次
- $\binom{N}{k}$：在 $N$ 次里选出 $k$ 次答对的组合数
- $p^k (1-p)^{N-k}$：某个具体答对/答错排列出现的概率

这个式子说明两件事：

1. 当 $p > 0.5$ 时，多数投票会放大正确率。
2. 当 $p < 0.5$ 时，多数投票会放大错误率。

所以 Self-Consistency 不是万能药。它最适合“模型基础能力已经接近会做，但不稳定”的题。

继续看上面的玩具例子。若单次正确率 $p=0.6$，则：

| 采样数 $N$ | 多数投票正确率 |
|---:|---:|
| 1 | 0.600 |
| 3 | $3p^2(1-p)+p^3=0.648$ |
| 5 | $\sum_{k=3}^{5}\binom{5}{k}p^k(1-p)^{5-k}=0.683$ |
| 9 | 约 0.733 |

这个趋势很重要：**从 1 到 3 的提升通常比从 5 到 9 更明显**。这就是“前期收益大，后期收益递减”。

这里要补一个新手常见误解。Self-Consistency 不是简单把文本投票，而是把**最终可比对的答案**投票。比如数学题投最后数值、代码题投是否通过测试、选择题投选项。要是把整段自然语言解释直接投票，聚合结果往往很脆弱，因为不同表述可能本质相同，也可能表述相似但结论不同。

### 2. Best-of-N：多生成，再用评分器选最优

**Best-of-N**，白话说，就是**先生成 $N$ 个候选，再让一个外部评分器选最优答案**。评分器可以是奖励模型，也可以是规则检查器、测试集、执行器，甚至人工偏好模型。

它和 Self-Consistency 的区别在于：
- Self-Consistency 用“票数”聚合
- Best-of-N 用“分数”聚合

前者更像民主投票，后者更像专家评审。

一个直观例子：

| 候选 | 最终答案 | 测试通过数 | 奖励分 |
|---|---|---:|---:|
| A | 正确 | 8/10 | 0.72 |
| B | 正确 | 10/10 | 0.95 |
| C | 错误 | 2/10 | 0.18 |
| D | 错误但措辞很完整 | 1/10 | 0.81 |

如果评分器真的反映“任务完成度”，系统会选 B；如果评分器更偏好“解释写得像样”，它可能选 D。这就是 Best-of-N 的强点和风险同时存在的地方。

它的收益直觉是：候选越多，抽到高质量样本的概率越高。但这个提升不会一直线性增长。一个常见分析是，随着 $N$ 增长，额外收益更接近对数规律，而不是线性规律：

$$
\text{gain}(N) \propto \log N
$$

有些论文会从分布偏移或 KL 角度给出更细的推导，例如：

$$
KL_{\text{BoN}} \approx \log N - \frac{N-1}{N}
$$

这里不需要把公式当成精确工程定律，记住直觉就够了：**从 1 增加到 8 往往值得，从 64 增加到 128 就常常不值**。

### 3. 搜索式推理：不只比终点，还比中间路径

第三类是**搜索式方法**，典型代表是 MCTS，也就是蒙特卡洛树搜索。白话说，它不是只在最终答案上挑选，而是**把中间推理步骤看成一棵树，在树上选择、展开、回传分数**。

如果把复杂任务拆成状态转移：

$$
s_0 \rightarrow s_1 \rightarrow s_2 \rightarrow \cdots \rightarrow s_T
$$

那么每一步可以获得一个过程奖励 $r_t$，总价值写成：

$$
V = \sum_{t=1}^{T} r_t
$$

其中 $r_t$ 由 **Process Reward Model（PRM，过程奖励模型）** 给出。PRM 的意思是：**不是只给最终答案打分，而是给中间步骤也打分**。它的价值在于，可以更早发现“这条路已经走歪”。

一个非常简化的预算分配框架可以写成：

$$
B = B_{\text{sample}} + B_{\text{rerank}} + B_{\text{search}}
$$

其中总预算 $B$ 根据难度函数 $d(x)$ 动态分配：

$$
(B_{\text{sample}}, B_{\text{rerank}}, B_{\text{search}}) = g(d(x))
$$

直白解释就是：
- 简单题：多分给单样本或少量重采样
- 中等题：多做 Self-Consistency
- 困难题：增加搜索和过程打分预算

可以把它理解成一个调度表：

| 条件 | 动作 | 原因 |
|---|---|---|
| 置信度高 | 早停直接返回 | 继续采样浪费 |
| 置信度中等 | 追加样本并投票 | 结果尚不稳定 |
| 候选差异大 | 启用重排模型 | 需要更细的区分 |
| 中间步骤易验证 | 启用 PRM / 搜索 | 路径信息有价值 |

真实工程例子是数学推理系统。单次生成往往会出现“前两步对，第三步算错”的情况。如果只比较最终答案，很多潜在好路径会被误杀；如果用 PRM 给中间步骤打分，再配合树搜索，就能把搜索预算优先投到更靠谱的分支上。这也是 MCTS + PRM 往往在复杂推理任务上优于纯 Best-of-N 的原因。

为了让三个机制的边界更清楚，可以直接对比：

| 机制 | 聚合对象 | 需要什么验证信号 | 更适合什么场景 |
|---|---|---|---|
| Self-Consistency | 最终答案 | 可比较的标准答案 | 模型“会做但不稳定” |
| Best-of-N | 候选整体质量 | 奖励模型、规则、执行结果 | 有较强外部评分器 |
| 搜索 + PRM | 中间步骤和最终答案 | 过程奖励、局部可验证步骤 | 长链推理、复杂规划 |

---

## 代码实现

下面给一个**最小可运行版本**，用来模拟“按难度动态分配推理预算 + Self-Consistency 投票 + 简化重排”的逻辑。它不是生产级实现，但足够说明机制，而且可以直接运行。

```python
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence


@dataclass(frozen=True)
class Budget:
    samples: int
    rerank_topk: int
    confidence_threshold: float


def allocate_budget(difficulty: float) -> Budget:
    """
    根据难度分配推理预算。
    difficulty 取值范围约定为 [0, 1]。
    """
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("difficulty must be between 0 and 1")

    if difficulty < 0.3:
        return Budget(samples=1, rerank_topk=1, confidence_threshold=1.0)
    if difficulty < 0.7:
        return Budget(samples=5, rerank_topk=3, confidence_threshold=0.60)
    return Budget(samples=9, rerank_topk=5, confidence_threshold=0.55)


def majority_confidence(answers: Sequence[str]) -> float:
    """
    返回当前最高票答案所占比例。
    例如 ["a", "a", "b"] -> 2 / 3
    """
    if not answers:
        return 0.0
    top_count = Counter(answers).most_common(1)[0][1]
    return top_count / len(answers)


def vote_answer(answers: Sequence[str]) -> str:
    """
    返回最高票答案。
    若有并列，Counter.most_common 的顺序会受首次出现位置影响。
    """
    if not answers:
        raise ValueError("answers must not be empty")
    return Counter(answers).most_common(1)[0][0]


def top_k_candidates(answers: Sequence[str], k: int) -> List[str]:
    """
    取出现频率最高的前 k 个候选，用于后续重排。
    """
    return [ans for ans, _ in Counter(answers).most_common(k)]


def rerank_with_reward(
    candidates: Sequence[str],
    reward_fn: Callable[[str], float],
) -> str:
    """
    用外部评分器选择分数最高的候选。
    """
    if not candidates:
        raise ValueError("candidates must not be empty")

    scored = [(reward_fn(candidate), candidate) for candidate in candidates]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def solve(
    prompt: str,
    difficulty: float,
    sampler: Callable[[str], str],
    reward_fn: Callable[[str], float],
) -> str:
    """
    一个简化版在线推理流程：
    1. 先按难度分配预算
    2. 多次采样
    3. 若投票已足够集中，则早停
    4. 否则对高频候选做一次重排
    """
    budget = allocate_budget(difficulty)
    answers: List[str] = []

    for _ in range(budget.samples):
        answers.append(sampler(prompt))

        # 至少拿到 3 个样本再判断是否早停，避免过早收缩
        if len(answers) >= 3:
            if majority_confidence(answers) >= budget.confidence_threshold:
                return vote_answer(answers)

    candidates = top_k_candidates(answers, budget.rerank_topk)
    return rerank_with_reward(candidates, reward_fn)


class DeterministicSampler:
    """
    用固定输出序列模拟“同一道题有时对、有时错”的现象。
    每次调用按顺序返回一个答案，跑到结尾后循环。
    """

    def __init__(self, outputs: Sequence[str]) -> None:
        if not outputs:
            raise ValueError("outputs must not be empty")
        self._outputs = list(outputs)
        self._index = 0

    def __call__(self, _prompt: str) -> str:
        value = self._outputs[self._index % len(self._outputs)]
        self._index += 1
        return value


def reward_fn(answer: str) -> float:
    """
    简化版奖励函数：
    在真实系统里，这里可以替换成测试通过率、规则检查器或奖励模型分数。
    """
    reward_table = {
        "323": 1.0,
        "333": 0.4,
        "313": 0.2,
    }
    return reward_table.get(answer, 0.0)


def main() -> None:
    sampler = DeterministicSampler(
        outputs=["323", "323", "333", "323", "313", "323", "333", "323", "323"]
    )

    result = solve(
        prompt="17 * 19 = ?",
        difficulty=0.8,
        sampler=sampler,
        reward_fn=reward_fn,
    )

    print("final answer:", result)

    # 基本断言，确保示例逻辑可运行
    assert result == "323"
    assert allocate_budget(0.1).samples == 1
    assert allocate_budget(0.5).samples == 5
    assert allocate_budget(0.9).samples == 9
    assert abs(majority_confidence(["a", "a", "b"]) - 2 / 3) < 1e-9
    assert vote_answer(["x", "x", "y"]) == "x"
    assert top_k_candidates(["a", "b", "a", "c", "a", "b"], 2) == ["a", "b"]


if __name__ == "__main__":
    main()
```

运行方式：

```bash
python3 ttc_demo.py
```

预期输出：

```text
final answer: 323
```

这段代码对应的工程流程是：

| 模块 | 作用 | 直白解释 |
|---|---|---|
| `allocate_budget` | 按难度分预算 | 难题多花钱，简单题少花钱 |
| `sampler` | 生成候选答案 | 同一道题做多遍 |
| `majority_confidence` | 计算投票集中度 | 看答案是否已经基本一致 |
| `vote_answer` | 多数投票 | 用票数聚合 |
| `rerank_with_reward` | 奖励模型重排 | 用评分器选最好 |

如果把它扩展到真实工程中的代码助手，可以这样理解：

1. 输入一个复杂报错和一段堆栈信息。
2. 先用轻量难度模型判断是“简单定位”还是“复杂因果分析”。
3. 如果是简单问题，直接单样本回答。
4. 如果是复杂问题，生成多条“问题原因 + 修复建议”候选。
5. 用规则检查器验证候选里是否引用了真实日志、是否匹配代码上下文、是否能通过测试。
6. 若多数候选已经集中在同一个根因，早停；否则进入重排或搜索。

关键调参项可以先这样理解：

| 参数 | 含义 | 常见取值思路 |
|---|---|---|
| `samples` | 采样次数 | 先小后大，逐步增加 |
| `confidence_threshold` | 早停阈值 | 任务越稳定可设越高 |
| `rerank_topk` | 参与重排的候选数 | 不宜过大，否则重排本身太贵 |
| `difficulty` | 难度分数 | 可由轻量模型或规则估计 |
| `reward_fn` | 评分器 | 优先用可验证信号，不要只靠黑盒偏好分 |

一个很实用的落地模式是：**先投票，后重排**。因为投票便宜，重排贵。先用少量样本看是否已出现明显共识，只有共识不足时才调用奖励模型或 PRM。这样能显著降低平均成本。

如果再往前走一步，一个更接近生产系统的流程通常长这样：

| 阶段 | 典型输入 | 典型输出 |
|---|---|---|
| 难度估计 | 请求文本、上下文长度、历史失败率 | `difficulty` |
| 采样 | Prompt + 不同随机种子/温度 | 多个候选 |
| 聚合 | 候选答案集合 | 投票结果或候选 shortlist |
| 验证/重排 | 候选、测试、规则、奖励模型 | 最终答案 |
| 监控 | 延迟、token、通过率 | 是否继续扩预算 |

这也是 Test-Time Compute 真正落地时最容易被忽略的一点：**它不是一个单点技巧，而是一条完整的在线决策链**。

---

## 工程权衡与常见坑

最常见的误区是把 Test-Time Compute 理解成“只要预算更大，效果就更好”。这在工程上通常是错的。

第一类坑是 **reward hacking**。这个术语首次出现时可以直接理解成：**模型学会迎合评分规则，而不是学会真正做对任务**。例如代码任务里，奖励模型可能偏好“解释很完整、术语很多”的答案，但真正重要的是补丁能否通过测试。于是 Best-of-N 越大，系统越容易从大量候选中挑出“最会讨好评分器”的那个，而不是真正正确的那个。

第二类坑是样本相关性过强。Self-Consistency 的前提接近“多次采样有一定独立性”。如果温度过低、提示词过死、模型总走同一条错误路径，那么采样 16 次和采样 2 次没有本质区别。

第三类坑是延迟尾部失控。线上系统最怕的不是平均延迟，而是 P95/P99。困难题一旦触发深搜索，可能把单请求延迟从 2 秒拉到 20 秒。用户看到的就是“系统偶尔很慢”，而不是“平均还行”。

第四类坑是把所有请求都按困难题处理。这样系统很快会把预算烧在不需要的地方。真正有效的是分层调度。

第五类坑是验证信号和目标错位。比如你想优化“代码修复成功率”，但重排阶段只看“回答像不像一个专业解释”，那额外推理就会把预算浪费在语言表面，而不是任务完成度上。

可以用一个风险表总结：

| 方法 | 成本 | 延迟 | 收益形态 | 主要风险 |
|---|---:|---:|---|---|
| 单样本 | 低 | 低 | 基线 | 上限低 |
| Self-Consistency | 中 | 中 | 先快后慢 | 样本相关性强时无效 |
| Best-of-N | 中到高 | 中到高 | 常见对数级递减 | reward hacking |
| MCTS + PRM | 高 | 高 | 难题收益更大 | 实现复杂、PRM 误判会误导搜索 |

监控指标不能只看最终正确率，还要看过程信号：

| 监控项 | 含义 | 异常信号 |
|---|---|---|
| Vote agreement | 投票一致度 | 一致度长期过低，说明样本分散或任务过难 |
| PRM score spread | 候选得分分布 | 得分差过小，说明评分器区分度不足 |
| Early-stop rate | 早停比例 | 过低说明预算浪费，过高可能过早收缩 |
| Cost per solved task | 每个已解决任务的平均成本 | 持续上升说明额外推理未转化为收益 |
| Pass@k / Exact match | 最终任务质量 | 与成本一起看，判断是否值得 |

真实工程里，代码修复代理就是典型场景。你可以让代理生成 8 个补丁，再跑测试选择最好。但如果评分器只看“通过了一个脆弱的单测”，它就可能挑出过拟合补丁。更稳妥的做法是：

- 奖励信号同时包含单元测试、静态检查、风格约束
- 对原始模型分布加约束，避免候选过度偏离
- 为超高预算任务设置硬上限和超时回退
- 分离“探索指标”和“上线指标”，不要只看离线最好成绩
- 单独跟踪高预算请求的 P95/P99，避免平均值掩盖问题

一个对新手尤其重要的判断标准是：**额外推理算力必须绑定“可验证信号”和“可中止机制”**。否则它只是更贵，而不是更好。

把常见坑翻成更具体的工程动作，大致是这样：

| 坑 | 上线前该做什么 |
|---|---|
| 奖励模型不可靠 | 先做小规模人工抽检，对比“高分候选”是否真的更对 |
| 样本高度相关 | 调整温度、提示模板、解题路径约束 |
| 预算爆炸 | 设置最大样本数、最大搜索深度、全链路超时 |
| 难度估计失真 | 保留回退策略，不要完全依赖一个黑盒分数 |
| 监控缺失 | 同时记录质量、时延、token、预算命中率 |

---

## 替代方案与适用边界

Test-Time Compute 不是唯一选择。至少还有三类相关路线需要一起看。

**Sleep-time Compute**，白话说，就是**把一部分本该在线上花的思考，提前搬到离线阶段去做**。例如提前生成题库、构造推理轨迹、蒸馏出更擅长长推理的小模型。它适合高并发、低延迟场景，因为线上不想再做太多搜索。

**Thinking-Optimal Scaling**，白话说，就是**让模型学会不同长度的思考方式，再在推理时选择合适长度**。重点不是一味变长，而是找到“足够正确且尽量短”的推理。

**MCTS + PRM** 属于更重的在线搜索路线。它适合复杂推理、步骤可验证、单题价值高的任务，比如数学证明、复杂代码修复、规划问题。

可以把几条路线做个对照：

| 路线 | 主要投入阶段 | 延迟 | 硬件压力 | 可解释性 | 适合任务 |
|---|---|---:|---:|---:|---|
| Test-Time Compute | 在线推理 | 中到高 | 中到高 | 中 | 难题、可验证任务 |
| Sleep-time Compute | 离线准备 | 低 | 离线高、在线低 | 中 | 高并发服务 |
| Thinking-Optimal | 训练与推理结合 | 中 | 中 | 中 | 需要控制思考长度 |
| MCTS + PRM | 在线搜索 | 高 | 高 | 高 | 数学、代码、规划 |

如果只看“零基础到初级工程师”的落地建议，可以按下面的决策顺序理解：

| 问题条件 | 更适合的方案 |
|---|---|
| 请求量大、延迟敏感 | 先做 Sleep-time Compute 或蒸馏 |
| 任务可验证、单题价值高 | 优先 Test-Time Compute |
| 中间步骤很重要 | 用 PRM 或搜索 |
| 奖励模型不可靠 | 少用激进的 Best-of-N，多用投票和规则验证 |
| 简单题很多、难题很少 | 做动态预算分配，不要全量重推理 |

一个简单 checklist：

1. 任务是否能验证最终结果，或至少验证中间步骤？
2. 线上是否能容忍额外延迟？
3. 难题占比是否足够高，值得做动态调度？
4. 奖励模型是否可信，还是容易被“刷分”？
5. 是否能设置早停、超时和回退路径？

如果这些问题里前两项都是否定，那么强行上 Test-Time Compute 通常不划算。反过来，如果任务高价值、难度高、可验证，那么它往往是比单纯扩参更有性价比的路径。

最后把边界压缩成一句更直白的话：

- 如果问题**不难**，别加太多推理预算。
- 如果结果**不可验证**，别迷信 Best-of-N。
- 如果任务**高价值且可验证**，推理时算力扩展通常值得认真做。

---

## 参考资料

- Snell et al. “Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning” (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html
- OpenAI, “Learning to reason with LLMs”: https://openai.com/index/learning-to-reason-with-llms/
- Emergent Mind, “Self-Consistency in Chain-of-Thought Reasoning”: https://www.emergentmind.com/topics/self-consistency-in-chain-of-thought-reasoning
- Emergent Mind, “Best-of-N”: https://www.emergentmind.com/topics/best-of-n-bon
- ACL 2025 相关 Best-of-N / KL 分析论文: https://aclanthology.org/2025.acl-long.649.pdf
- Microsoft Research, “Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning”: https://www.microsoft.com/en-us/research/publication/towards-thinking-optimal-scaling-of-test-time-compute-for-llm-reasoning/
- Microsoft Research, “rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking”: https://www.microsoft.com/en-us/research/publication/rstar-math-small-llms-can-master-math-reasoning-with-self-evolved-deep-thinking/
- Awesome Inference-Time Scaling: https://github.com/ThreeSR/Awesome-Inference-Time-Scaling
