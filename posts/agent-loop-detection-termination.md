## 核心结论

Agent 推理循环检测的目标，不是单纯发现“有重复”，而是在任务尚未收敛、但系统已经开始重复消耗 token、工具配额和等待时间时，尽早识别“当前路径不再产生有效增量”。

生产级方案不能依赖单一指标。更稳妥的做法，是把三类信号叠加使用：

1. 行为签名相似度。行为签名指“这一轮做了什么、输入了什么、得到了什么”的压缩表示，用来判断当前动作与历史动作是否本质相同。
2. n-gram 重复率。n-gram 指连续的短序列片段，这里可理解为“最近几步动作或输出的局部重复程度”。
3. 状态覆盖度。状态覆盖度指“系统是否真的进入了新状态、接触了新对象、获得了新证据或产生了新中间结果”。

一个常用判断式是：

$$
\text{loop\_score}_t=\alpha \cdot \text{sim}(B_t,B_{t-k})+\beta \cdot N_t+\gamma \cdot (1-C_t)
$$

其中：

- $B_t$ 是第 $t$ 步的行为签名
- $\text{sim}$ 是行为相似度函数
- $N_t$ 是重复率
- $C_t$ 是状态覆盖度
- $\alpha,\beta,\gamma$ 是三类信号的权重

直观理解很简单：越像以前、越重复、越没有新状态，就越接近死循环。

工程上，单次高分通常还不够。更可靠的规则是：当 `loop_score > 0.7`，并且连续两轮没有新状态时，触发四级干预链路：

| 级别 | 动作 | 目标 | 典型系统行为 |
|---|---|---|---|
| 1 | 提示 | 让 Agent 自行识别重复并换策略 | 注入提醒，要求更换关键词、来源或方法 |
| 2 | 重规划 | 重新拆解任务，重写子步骤 | 清空当前短计划，重新生成执行路径 |
| 3 | 子目标切换 | 放弃当前路径，改走备选路径 | 从网页搜索切到结构化数据库、从修复切到定位 |
| 4 | 强制终止 | 切断继续消耗，返回失败原因 | 抛出异常、记录行为链、通知上层调度器 |

玩具例子：一个简单问答 Agent 在 `search -> summarize -> search -> summarize` 之间反复来回。如果每次搜索结果都相同，摘要也只是换措辞，那么行为相似度高、n-gram 重复率高、状态覆盖度低，会被判定为循环。

真实工程例子：一个网络情报 Agent 在 `research -> web_search -> analysis` 链路中，多次查询同一关键词集合，并反复访问已见过的网页。系统应先追加提示：“已有结果重复，请切换检索词、扩大时间范围或改用结构化来源。”若仍无进展，再进入 circuit breaker。circuit breaker 可以直接理解为“检测到危险模式后强制切断”的保护机制。

这个结论可以压缩成一句话：循环检测真正要拦住的，不是“重复动作”，而是“没有新状态的重复动作”。

---

## 问题定义与边界

问题定义很明确：在多轮推理、多工具调用的场景中，Agent 可能反复执行同一类动作序列，却没有获得新的状态信息，导致成本持续上升、响应越来越慢、任务仍不收敛。

这里最容易混淆的边界，不是“是否发生重复”，而是“重复是否带来了新信息”。因为很多正常任务天然包含重试。重试不是缺陷，往往还是必要机制。

常见合法回溯包括：

- 搜索第一次失败，第二次换关键词后成功
- 表单第一次提交报错，第二次补齐字段后成功
- 代码修复第一次没过测试，第二次修改后通过
- API 第一次超时，第二次命中备用节点后恢复
- 网页第一次元素定位失败，第二次等待 DOM 稳定后成功

这些都属于正常推进。回溯的定义是“暂时返回前一步重新尝试”，它不等于死循环。

因此，循环检测必须同时回答两个问题：

1. 当前动作像不像之前已经做过的动作？
2. 当前动作有没有把系统带到新状态？

如果只回答第一个问题，会把很多健康重试误判为循环；如果只回答第二个问题，又可能对明显的重复行为反应过慢，因为系统要等很久才会发现“虽然看似有一点变化，但本质上仍在原地消耗”。

下面这个表格可以帮助区分边界：

| 信号 | 含义 | 正常情况 | 异常情况 | 为什么单独使用不够 |
|---|---|---|---|---|
| 行为签名相似度 | 当前步骤与历史步骤的相似程度 | 相似，但目标参数、约束条件或候选对象变了 | 高度相似，参数和结果也几乎不变 | 会误伤正常重试 |
| n-gram 重复率 | 最近动作或输出短序列的重复比例 | 局部重复，但整体在推进 | 局部和整体都重复 | 容易把格式化输出误判为循环 |
| 状态覆盖度 | 新状态占已访问状态空间的比例 | 持续访问新页面、新对象、新中间结果 | 长时间停留在旧状态集合 | 状态定义不好时容易失真 |
| 连续无新状态轮数 | 连续多少轮没有实质进展 | 0 或 1 轮 | 2 轮及以上 | 只看它会发现得太晚 |
| 干预建议 | 是否需要系统介入 | 仅记录日志 | 提示、重规划或终止 | 依赖前面几项作为输入 |

举一个边界例子。假设一个 Agent 连续 5 次执行 `research -> web_search -> analysis`：

- 如果 5 次分别进入了不同网页，拿到了不同证据片段，那么这是“有探索的重复”，不该直接判死循环。
- 如果 5 次都落回同一组搜索词、同一批网页、同一种摘要结论，那么这是“无探索的重复”，应判定为循环。

再看一个新手更容易理解的对照表：

| 场景 | 动作是否重复 | 状态是否变化 | 应否判循环 |
|---|---|---|---|
| 用不同关键词再次搜索 | 是 | 是 | 否 |
| 重试同一 API，但命中了不同数据分片 | 是 | 是 | 否 |
| 反复访问同一 URL，摘要只换措辞 | 是 | 否 | 是 |
| 连续生成计划，但子步骤和目标对象都一样 | 是 | 否 | 是 |
| 修复同一 bug，但测试失败信息每次都不同 | 部分重复 | 是 | 否 |

所以，循环检测的本质不是识别“重复动作”，而是识别“没有新状态的重复动作”。这是边界判断的核心。

---

## 核心机制与推导

循环检测需要的是一个连续量，而不是简单的布尔值。原因很直接：真实工程中很少存在一个绝对清晰的边界，更多情况是“越来越像循环”。连续得分更适合做策略升级。

### 1. 三类信号

第一类是行为签名 $B_t$。  
它可以由工具名、归一化输入、关键参数、输出摘要、错误码、目标对象 ID 等拼接而成，再做哈希或向量化。对白话解释，它相当于给“这一轮到底发生了什么”打一个可比较的指纹。

一个简单的行为签名可以写成：

$$
B_t=\text{hash}(\text{tool}_t,\text{args}_t^{*},\text{result}_t^{*},\text{error}_t,\text{state}_t^{*})
$$

其中上标 `*` 表示“归一化后”。所谓归一化，就是去掉无关噪声，只保留真正影响任务进展的字段。例如：

- 把 URL 中无关追踪参数删掉
- 把搜索时间戳对齐到小时或日期
- 把模型输出中的随机措辞替换成结构化摘要
- 把错误堆栈折叠成错误类型码

第二类是 n-gram 重复率 $N_t$。  
它统计最近窗口内动作序列或输出片段的重复程度。对白话解释，它看的是“最近几步是不是总在做差不多的事、说差不多的话”。

如果动作序列窗口为 $A_{t-w+1:t}$，则长度为 $n$ 的动作 n-gram 集合可以定义为：

$$
G_t^{(n)}=\{(A_i,A_{i+1},\dots,A_{i+n-1})\}_{i=t-w+1}^{t-n+1}
$$

一种简单的重复率可写为：

$$
N_t=\frac{\sum_{g \in G_t^{(n)}} \max(\text{count}(g)-1,0)}{|G_t^{(n)}|}
$$

这个定义的含义是：如果一个片段只出现一次，不算重复；从第二次开始才累计重复成本。

第三类是状态覆盖度 $C_t$。  
它统计系统是否真正探索到了新状态。对白话解释，它看的是“系统有没有真的打开过新门，而不是在原来的门口来回转”。

在有限窗口内，一个实用定义是：

$$
C_t=\frac{|\mathcal{S}_{t,w}^{\text{unique}}|}{|\mathcal{S}_{t,w}|}
$$

其中：

- $\mathcal{S}_{t,w}$ 表示最近 $w$ 轮访问到的状态序列
- $\mathcal{S}_{t,w}^{\text{unique}}$ 表示去重后的状态集合

如果最近 8 轮访问了 8 个完全不同的状态，则 $C_t=1$；如果最近 8 轮只有 2 个状态来回切换，则 $C_t=2/8=0.25$，说明探索明显不足。

常见状态定义如下：

| 任务类型 | 可选状态定义 |
|---|---|
| 网页搜索 | URL、域名、SERP 结果集合摘要、文档哈希 |
| 浏览器自动化 | DOM 哈希、页面路由、表单字段状态、焦点元素 |
| 代码代理 | 文件路径、git diff 指纹、测试失败集合、编译错误类型 |
| 数据工作流 | 表名、分区、主键集合、查询计划摘要 |
| 多 Agent 协作 | 当前子目标、共享黑板版本号、证据集摘要 |

### 2. 综合得分

将三者组合：

$$
\text{loop\_score}_t=\alpha \cdot \text{sim}(B_t,B_{t-k})+\beta \cdot N_t+\gamma \cdot (1-C_t)
$$

解释如下：

- $\text{sim}(B_t,B_{t-k})$ 越高，说明当前行为越像过去第 $t-k$ 步
- $N_t$ 越高，说明最近局部模式越重复
- $(1-C_t)$ 越高，说明新状态越少，探索越弱

$\alpha,\beta,\gamma$ 是权重，满足：

$$
\alpha+\beta+\gamma=1,\quad \alpha,\beta,\gamma \ge 0
$$

常见经验如下：

| 系统特征 | 建议提高的权重 | 原因 |
|---|---|---|
| 工具链固定、动作模板稳定 | $\alpha$ | 行为相似最能暴露“同样的步骤在重放” |
| 文本输出长、复述成本高 | $\beta$ | 文本局部重复更容易提前暴露空转 |
| 强依赖环境探索 | $\gamma$ | 是否进入新状态比措辞是否变化更关键 |
| 高风险外部工具调用 | $\alpha,\gamma$ | 既要识别重复调用，也要确认是否真的有新结果 |

如果没有历史数据可调参，可以先用一组保守初值：

$$
\alpha=0.4,\quad \beta=0.3,\quad \gamma=0.3
$$

这样做的理由是：行为相似通常最稳定，n-gram 和状态覆盖更依赖具体业务定义。

### 3. 为什么要看 $k$

公式里比较的是 $B_t$ 和 $B_{t-k}$，而不是只看前一轮。因为很多循环不是“一步一模一样”，而是长度为 2 到 5 的环。

典型长度为 2 的环：

- `search -> summarize -> search -> summarize`
- `plan -> call_tool -> plan -> call_tool`

典型长度为 3 或 4 的环：

- `locate -> edit -> test -> locate`
- `research -> compare -> summarize -> research`

如果只和上一轮比较，系统会错过这类“隔一步或隔几步重复”的模式。因此 $k$ 应覆盖潜在环长。工程上常取最近 3 到 8 步窗口，分别与历史片段比较，取最大相似度或平均相似度：

$$
\text{sim}^{*}(B_t)=\max_{k \in K}\text{sim}(B_t,B_{t-k}),\quad K=\{2,3,\dots,8\}
$$

这个写法的直觉是：只要当前行为和最近某个历史位置高度相似，就值得提高警惕。

### 4. 连续无新状态为什么重要

单轮高分可能只是偶然，并不代表系统已经真的陷入循环。最常见的例子是临时超时重试。它会让当前轮与上一轮很像，但不能因此立刻熔断。

因此一般要再加一个门槛：

$$
\text{trigger}_t=
\mathbf{1}\left(
\text{loop\_score}_t>\tau
\land
\text{no\_new\_state\_streak}\ge m
\right)
$$

其中：

- $\tau$ 是触发阈值，例如 0.7
- $m$ 是连续无新状态的最小轮数，常见取值为 2

这条规则的工程意义很强：

- `loop_score` 负责回答“像不像循环”
- `no_new_state_streak` 负责回答“是否已经持续没有进展”

只有这两个条件同时满足，系统才从“怀疑”升级为“介入”。

### 5. 数值玩具例子

设：

- $\text{sim}(B_t,B_{t-2})=0.76$
- $N_t=0.82$
- $C_t=0.30$
- $\alpha=0.4,\beta=0.3,\gamma=0.3$

则：

$$
\text{loop\_score}_t
=0.4 \times 0.76+0.3 \times 0.82+0.3 \times (1-0.30)
$$

即：

$$
\text{loop\_score}_t
=0.304+0.246+0.21
=0.76
$$

这个结果高于 0.7，说明当前轮已经非常可疑。但如果这是第一次无新状态，仍可以只触发“提示”而不是终止。

再假设下一轮仍没有新状态：

- $\text{sim}(B_{t+1},B_{t-1})=0.79$
- $N_{t+1}=0.84$
- $C_{t+1}=0.25$

则：

$$
\text{loop\_score}_{t+1}
=0.4 \times 0.79+0.3 \times 0.84+0.3 \times 0.75
=0.793
$$

如果此时 `no_new_state_streak = 2`，就应从“记录风险”升级到“主动干预”。

真实工程中，这个分数不是为了追求数学上的完美，而是为了给策略机提供统一输入。系统真正关心的问题是：

- 何时提醒
- 何时重规划
- 何时切换路径
- 何时直接停止

---

## 代码实现

下面给出一个可运行的 Python 简化实现。它不依赖外部模型，只使用标准库，通过可解释规则模拟行为签名、重复率、状态覆盖和四级响应。代码可以直接保存为 `loop_detector.py` 后运行。

```python
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Sequence, Tuple


class LoopDetectedError(RuntimeError):
    """Raised when the detector decides the agent should be terminated."""


@dataclass(frozen=True)
class StepRecord:
    action: str
    args_digest: str
    output_digest: str
    state_id: str


@dataclass
class LoopConfig:
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3
    threshold: float = 0.7
    compare_ks: Tuple[int, ...] = (2, 3, 4)
    ngram_size: int = 2
    window_size: int = 8
    min_history: int = 4
    max_no_new_state_before_terminate: int = 4

    def __post_init__(self) -> None:
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 1e-9:
            raise ValueError("alpha + beta + gamma must equal 1.0")
        if self.window_size < self.ngram_size * 2:
            raise ValueError("window_size should be at least 2 * ngram_size")


def tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token]


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union)


@dataclass
class LoopDetector:
    config: LoopConfig
    history: List[StepRecord] = field(default_factory=list)
    recent_states: Deque[str] = field(default_factory=deque)
    no_new_state_streak: int = 0

    def behavior_signature_similarity(self, current: StepRecord, past: StepRecord) -> float:
        action_score = 1.0 if current.action == past.action else 0.0
        args_score = jaccard_similarity(tokenize(current.args_digest), tokenize(past.args_digest))
        output_score = jaccard_similarity(tokenize(current.output_digest), tokenize(past.output_digest))
        state_score = 1.0 if current.state_id == past.state_id else 0.0

        # 四个分量相加后仍落在 [0, 1]
        return (
            0.35 * action_score
            + 0.25 * args_score
            + 0.25 * output_score
            + 0.15 * state_score
        )

    def max_behavior_similarity(self) -> float:
        current = self.history[-1]
        similarities: List[float] = []

        for k in self.config.compare_ks:
            if len(self.history) > k:
                past = self.history[-1 - k]
                similarities.append(self.behavior_signature_similarity(current, past))

        return max(similarities, default=0.0)

    def ngram_repeat_rate(self) -> float:
        actions = [record.action for record in self.history[-self.config.window_size :]]
        n = self.config.ngram_size
        if len(actions) < n * 2:
            return 0.0

        grams = [tuple(actions[i : i + n]) for i in range(len(actions) - n + 1)]
        counts = Counter(grams)

        repeated_occurrences = sum(max(count - 1, 0) for count in counts.values())
        return repeated_occurrences / len(grams)

    def state_coverage(self) -> float:
        states = list(self.recent_states)[-self.config.window_size :]
        if not states:
            return 1.0
        return len(set(states)) / len(states)

    def intervention_level(self, score: float) -> str:
        if score < self.config.threshold or self.no_new_state_streak == 0:
            return "observe"
        if self.no_new_state_streak == 1:
            return "prompt"
        if self.no_new_state_streak == 2:
            return "replan"
        if self.no_new_state_streak == 3:
            return "switch_subgoal"
        return "terminate"

    def add_step(self, step: StepRecord) -> Tuple[float, str]:
        if self.recent_states and step.state_id == self.recent_states[-1]:
            self.no_new_state_streak += 1
        else:
            self.no_new_state_streak = 0

        self.history.append(step)
        self.recent_states.append(step.state_id)

        if len(self.recent_states) > self.config.window_size:
            self.recent_states.popleft()

        if len(self.history) < self.config.min_history:
            return 0.0, "observe"

        similarity = self.max_behavior_similarity()
        repetition = self.ngram_repeat_rate()
        coverage = self.state_coverage()

        score = (
            self.config.alpha * similarity
            + self.config.beta * repetition
            + self.config.gamma * (1.0 - coverage)
        )

        level = self.intervention_level(score)
        if level == "terminate":
            raise LoopDetectedError(
                f"loop detected: score={score:.3f}, no_new_state_streak={self.no_new_state_streak}"
            )

        return score, level


def demo() -> None:
    detector = LoopDetector(LoopConfig())

    steps = [
        StepRecord("research", "company profile q1", "seed urls page 1", "search:company-q1"),
        StepRecord("web_search", "company profile q1", "top 10 links page 1", "serp:company-q1"),
        StepRecord("analysis", "summarize evidence", "company founded in 2019", "doc:profile-1"),
        StepRecord("research", "company profile q1", "seed urls page 1", "doc:profile-1"),
        StepRecord("web_search", "company profile q1", "top 10 links page 1", "doc:profile-1"),
        StepRecord("analysis", "summarize evidence", "company founded in 2019", "doc:profile-1"),
        StepRecord("research", "company profile q1", "seed urls page 1", "doc:profile-1"),
    ]

    print("step | score | level | no_new_state_streak")
    print("-" * 44)

    for index, step in enumerate(steps, start=1):
        try:
            score, level = detector.add_step(step)
            print(f"{index:>4} | {score:>5.3f} | {level:<14} | {detector.no_new_state_streak}")
        except LoopDetectedError as exc:
            print(f"{index:>4} |  n/a  | terminate      | {detector.no_new_state_streak}")
            print(exc)
            break


if __name__ == "__main__":
    demo()
```

这段代码可以直接运行。一个典型输出会类似这样：

```text
step | score | level | no_new_state_streak
--------------------------------------------
   1 | 0.000 | observe        | 0
   2 | 0.000 | observe        | 0
   3 | 0.415 | observe        | 0
   4 | 0.637 | observe        | 1
   5 | 0.717 | replan         | 2
   6 | 0.801 | switch_subgoal | 3
   7 |  n/a  | terminate      | 4
loop detected: score=0.836, no_new_state_streak=4
```

这个实现体现了几个关键工程点。

第一，检测逻辑必须放在每轮执行之后立即运行。否则循环在日志上已经清晰可见，但成本也已经失控。

第二，检测结果不要只有 `True/False`。更有用的是返回干预级别，因为“提示”和“强制终止”对应的是完全不同的调度行为。

第三，状态判断不要只看“是否曾经见过”，还要看“最近是否持续停在同一类状态”。代码里用的是简化版规则：如果连续步骤命中同一状态，就增加 `no_new_state_streak`。真实系统里，状态相等往往要经过归一化后再比较。

第四，行为签名最好拆成多个可解释分量，而不是只做一个黑盒 embedding 相似度。这样在误判时更容易回放和调试。

下面给出一个更贴近生产环境的数据结构示例：

| 字段 | 含义 | 例子 |
|---|---|---|
| `action` | 工具或动作类型 | `web_search`、`click`、`run_test` |
| `args_digest` | 输入参数摘要 | `query=company+profile,date<=2025` |
| `output_digest` | 输出结果摘要 | `10 links, 8 duplicates, 0 new domains` |
| `state_id` | 当前状态标识 | `serp:company-q1`、`dom:checkout-form-v3` |
| `error_code` | 失败类型 | `TIMEOUT`、`VALIDATION_ERROR` |
| `cost` | 当前轮成本 | token、API 调用数、等待时间 |

如果系统更复杂，还可以继续扩展：

$$
\text{loop\_score}_t=
\alpha \cdot \text{sim}(B_t,B_{t-k})
+\beta \cdot N_t
+\gamma \cdot (1-C_t)
+\delta \cdot E_t
$$

其中 $E_t$ 可以表示“重复错误率”或“重复失败模式强度”。例如连续 3 次都因为同一个权限错误失败，即使动作表面上不完全一样，也可能已经进入无效路径。

真实工程例子可以这样落地：在一个多 Agent 情报系统中，调度器每轮记录：

```text
tool_name + normalized_args + result_digest + state_hash + error_code
```

若连续多轮都命中高 `loop_score`，则按顺序执行：

1. 给当前 Agent 注入提示：“不要重复使用相同检索词和来源，优先寻找新实体、新时间范围或新证据类型。”
2. 若下一轮仍无新状态，要求重写计划，并说明当前路径已被判定为低收益。
3. 若还失败，切换子目标，例如从“网页搜索”改为“结构化数据库查询”。
4. 最后终止，并把最近行为链、得分曲线和状态摘要写入日志与告警系统。

这才是完整的“检测 + 干预 + 终止”闭环。

---

## 工程权衡与常见坑

循环检测最常见的问题，不是“完全检测不到”，而是“检测太早”或“检测太晚”。前者会误伤正常探索，后者会让系统在明显空转时继续烧钱。

| 常见坑 | 现象 | 后果 | 应对方式 |
|---|---|---|---|
| 只看 n-gram 重复 | 文本复述一多就报警 | 误伤合法重试 | 必须叠加状态覆盖度 |
| 只看状态覆盖 | 新状态很多，但动作已明显空转 | 响应过慢 | 再加行为相似度 |
| 没有短期缓存 | 要到很多轮后才发现重复 | 成本暴涨 | 保存最近 3 到 8 轮行为签名 |
| 只有软提示，没有硬终止 | Agent 被提醒后仍继续绕圈 | 无限消耗 | 设计终止阈值和 circuit breaker |
| 状态定义过粗 | 不同页面都算同一状态 | 误判循环 | 使用更细粒度 state hash |
| 状态定义过细 | 微小变化都算新状态 | 漏判循环 | 做归一化或聚类后再计数 |
| 输入未归一化 | 时间戳、随机 ID 导致“看起来总是新行为” | 检测失效 | 去除无关字段，保留关键参数 |
| 输出未压缩 | 长文本每次都略有差异 | 相似度不稳定 | 先抽取结构化摘要再比对 |

一个典型误判例子，是 Web 自动化里的“查找 -> 填写 -> 撤销 -> 再查找”。  
如果只看动作序列，它高度重复；但如果页面字段值、校验状态、焦点位置在变化，那么状态并没有停滞。这种情况下不应过早打断。

另一个常见坑是忽略缓存和归一化。比如搜索参数只有时间戳不同，其他完全一样。如果不做归一化，系统会把这些请求当成“新行为”，导致重复检测失效。相反，如果把真正重要的参数也去掉了，例如查询关键词、站点范围、时间窗口，又会把有效探索误判为重复。

因此，工程上通常需要三层归一化：

1. 动作归一化：统一工具名、参数顺序和默认值表示。
2. 输出归一化：去掉噪声字段，只保留关键信息摘要。
3. 状态归一化：只保留决定任务进展的状态特征。

可以把这三层理解成三个问题：

- 动作归一化在回答：“这两个调用本质上是不是同一件事？”
- 输出归一化在回答：“这两次返回的有效信息是不是同一类结果？”
- 状态归一化在回答：“系统是否真的进入了不同阶段？”

还有一个现实权衡是阈值选择。阈值低，能更早节省成本，但也更容易打断有效搜索；阈值高，误判少，但系统已经浪费了不少 token 和工具预算。调阈值不能只靠直觉，最好基于历史运行数据回放。

一个简单的调参流程如下：

| 步骤 | 做法 | 目标 |
|---|---|---|
| 1 | 回放历史任务轨迹 | 找出真实循环样本和正常重试样本 |
| 2 | 记录每轮 `loop_score` | 看循环与非循环的分布是否可分 |
| 3 | 试不同阈值 | 比较误报率和漏报率 |
| 4 | 评估成本收益 | 计算提前终止节省的 token 与错误打断造成的损失 |
| 5 | 分任务类型建阈值 | 搜索型、代码型、浏览器型任务分别调参 |

如果系统允许，还可以把“阈值”升级成“任务类型相关策略”。例如：

- 搜索型任务允许更多重试，因为探索空间大
- 浏览器表单任务允许更少重试，因为状态空间通常更稳定
- 代码修复任务对测试失败集合更敏感，因为它能直接反映新状态

---

## 替代方案与适用边界

不是所有系统都需要完整的三信号融合。有些场景更适合从简单方案开始，再逐步升级。

| 方案 | 核心思路 | 适用场景 | 弱点 |
|---|---|---|---|
| 迭代计数器 | 超过最大轮数就停 | 原型系统、低复杂度任务 | 不能区分有进展的重试 |
| Token 预算 | 超过 token 或费用预算就停 | 成本敏感系统 | 只能控成本，不能识别原因 |
| 语义相似度 | 比较多轮输出是否语义接近 | 文本型 Agent、研究型 Agent | 需要额外计算，可能受表述变化干扰 |
| 状态覆盖度 | 看是否进入新状态 | 浏览器自动化、流程执行 | 状态定义困难 |
| 三信号融合 | 相似度 + 重复率 + 覆盖度 | 生产级多工具 Agent | 实现复杂度更高 |

最简单的替代方案是“最大轮数 + token 预算”。优点是实现非常快，适合作为第一层保险丝；缺点也很明显：它只能回答“要不要停”，不能回答“为什么要停”。

更细粒度的替代方案是“语义相似度 + 状态哈希”。  
它适用于高价值任务，例如情报检索、代码代理、复杂工作流。它能在第五次重复前看出“内容虽然换了词，但本质结论没有变化”，然后触发 circuit breaker。

一个实用的分层设计是：

1. 第一层：迭代计数和 token 预算，保证绝不失控。
2. 第二层：行为签名和 n-gram，快速发现显式重复。
3. 第三层：状态覆盖和语义相似度，区分健康重试与真正空转。
4. 第四层：响应策略机，决定提示、重规划、切子目标还是终止。

这个分层设计的价值在于：前两层成本低、触发快，后两层判断更准、解释性更强。

下面给出不同任务类型的推荐方案：

| 任务类型 | 推荐最小方案 | 推荐生产方案 |
|---|---|---|
| 一次性文本生成 | 最大轮数 + token 预算 | 语义相似度 + 预算控制 |
| 网页研究 Agent | 行为签名 + 状态覆盖 | 三信号融合 + 干预链路 |
| 浏览器自动化 | 状态覆盖 + 错误模式检测 | 三信号融合 + DOM 状态哈希 |
| 代码代理 | 行为签名 + 测试失败集合 | 三信号融合 + diff/测试状态 |
| 数据工作流 Agent | 状态覆盖 + 预算控制 | 三信号融合 + 查询计划摘要 |

这套设计的适用边界也很清楚。

如果任务几乎没有“状态”概念，例如一次性纯文本生成，那么状态覆盖度很难定义，此时应退化为“语义相似度 + 预算控制”。

如果任务是浏览器操作、文件系统操作或数据库事务，那么状态覆盖度通常非常关键。因为这些系统里，“是否进入新状态”往往比“措辞是否变化”更能说明问题。

如果任务高度依赖外部环境，且环境经常非确定性波动，例如搜索引擎排序变化、网页动态渲染、分布式服务延迟，那么循环检测最好不要只用硬阈值。更稳妥的做法是：

- 对状态做归一化
- 对短窗口和长窗口同时统计
- 把“重复失败模式”作为附加特征
- 把终止前的最后一次介入留给“重规划”而不是直接熔断

换句话说，替代方案不是“对错之分”，而是“成本、复杂度、准确率”的权衡。系统越复杂、单轮成本越高，就越值得做完整的三信号融合。

---

## 参考资料

- [TrackAI: Loop Detection & Breaking: Stop Infinite Agent Loops](https://trackai.dev/tracks/observability/debugging-tracing/loop-detection/?utm_source=openai)
- [Michael Brenndoerfer: Agent Evaluation: Metrics, Benchmarks and Safety Standards](https://mbrenndoerfer.com/writing/agent-evaluation-metrics-benchmarks-safety?utm_source=openai)
- [Martin Fowler: Circuit Breaker](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Yao et al., ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
