## 核心结论

测试时计算扩展，指模型在推理阶段额外消耗计算资源来换取更高正确率。这里的“额外计算”不是重新训练模型，也不是增加参数量，而是在回答同一个问题时，多生成几个候选解、多走几步推理链，或者多做一次验证。它解决的是“模型并非完全不会，而是第一次没答对”的问题。

更准确地说，测试时计算扩展把一次性回答，改成了“生成候选解 -> 检查候选解 -> 选出更可靠结果”的过程。只要任务满足两个条件，它通常就有效：

| 条件 | 含义 | 典型任务 |
| --- | --- | --- |
| 正确解可被搜索到 | 模型有非零概率在某次尝试中走到正确路径 | 数学题、代码修复、规划搜索 |
| 正确解可被区分出来 | 系统能判断哪个候选更可信 | 单元测试、判题器、规则检查、多数投票 |

因此，测试时计算扩展的价值不在于“让模型更聪明”，而在于“让同一个模型把已有能力兑现得更充分”。对复杂推理任务，这个区别很重要。模型可能本来就具备解题能力，但单次生成会因为随机采样、局部推理错误、过早收敛到错误思路而失败。增加测试时预算，本质上是在增加命中正确轨迹的机会。

若设单次尝试成功概率为 $p_x$，最大可逼近成功率为 $F_{\max}$，那么预算为 $N$ 时，总体成功率可写成：

$$
F(N)=F_{\max}\bigl(1-(1-p_x)^N\bigr)
$$

这个式子表达的是“至少成功一次”的概率。若每次尝试近似独立，那么连续 $N$ 次都失败的概率是 $(1-p_x)^N$，取反后就得到至少一次成功。乘上 $F_{\max}$，表示系统并不一定能逼近 100%，因为基础模型、提示词、验证器本身可能存在能力上限。

它的直接结论有三个：

| 结论 | 数学表现 | 工程含义 |
| --- | --- | --- |
| 预算增加会提高成功率 | $F(N)$ 随 $N$ 单调上升 | 多采样通常有效 |
| 收益递减 | 后续增益逐步变小 | 不应无上限堆预算 |
| 存在平台期 | $F(N)$ 逼近 $F_{\max}$ | 到平台期后应换策略 |

边际增益更准确地写成离散形式：

$$
\Delta F(N)=F(N+1)-F(N)=F_{\max}p_x(1-p_x)^N
$$

也就是说，前几次追加预算最有价值，后面每多一次尝试，新增收益都会更小。这正是“收益递减”的数学形式。

对初学者可以把它理解成两句话：

1. 模型没有换脑子，只是被允许多试几次。
2. 多试几次能提高命中率，但不会无限涨分。

例如一道乘法题，模型第一次口算错了；如果允许它列出中间步骤，再做一次验算，正确率往往会上升。这个增益来自“更多尝试 + 更强约束”，而不是来自“模型参数突然变大”。

---

## 问题定义与边界

测试时 compute budget，指推理阶段可以显式控制的资源总量。它不是抽象概念，而是具体的 token、时间和验证次数。工程上最常见的预算有三类：

| 资源 | 含义 | 典型做法 | 直接成本 |
| --- | --- | --- | --- |
| 采样次数 | 对同一问题生成多少个候选解 | `best-of-N`、多数投票、并行采样 | 调用次数增加 |
| 思考长度 | 每个候选允许消耗多少推理 token | 更长 chain-of-thought、更长草稿 | 单次延迟和 token 成本增加 |
| 验证开销 | 对候选解做多少次检查 | 单元测试、编译、规则验证、执行器 | CPU/GPU/外部服务成本增加 |

这里的 budget 最终都会落到三个业务指标上：

| 指标 | 问题 | 例子 |
| --- | --- | --- |
| 延迟 | 用户要等多久 | 接口 SLA 规定 2 秒内返回 |
| 成本 | 每次请求花多少钱 | token 成本、沙箱执行成本 |
| 稳定性 | 结果波动是否可接受 | 同题多次调用不能差异太大 |

SLA 是 service level agreement，这里可以简单理解成“接口返回时间的硬约束”。如果一个在线问答系统承诺 1 秒内返回，那么再好的测试时扩展，只要把平均延迟拉到 5 秒，就不适合作为默认路径。

测试时计算扩展并不对所有任务都有效。关键要看任务到底属于“可推导问题”还是“事实检索问题”。

| 任务类型 | 额外 compute 的典型效果 | 原因 |
| --- | --- | --- |
| 数学推理 | 通常有效 | 不同推理路径可竞争，答案可验证 |
| 代码生成 | 通常有效 | 可编译、可测试、可 benchmark |
| 规划搜索 | 常常有效 | 路径质量可比较，搜索可剪枝 |
| 逻辑谜题 | 常常有效 | 约束满足可检查 |
| 事实查询 | 可能无效甚至变差 | 多想不会凭空补足缺失知识 |
| 法律引用/医学事实 | 风险高 | 会把未知事实包装成更流畅的错误 |

这一区分很重要。测试时扩展更适合“会做但第一次不一定做对”的任务，不适合“根本不知道答案”的任务。

先看一个正例。问题是“37 乘 48 等于多少”。模型单次生成可能心算出错，但如果允许它先拆成

$$
37\times 48 = 37\times(50-2)=1850-74=1776
$$

再让系统核对结果是否一致，正确率通常会上升。因为这是推导型任务，正确答案可以通过中间步骤和算术规则验证。

再看一个反例。问题是“某国 2025 年某监管细则第几款写了什么”。如果模型没有可靠检索链路，多给它 5 倍思考 token，并不会让参数里的知识自动更新。它只会更长时间地补全文字，可能生成更完整、也更自信的错误答案。

为了决定预算加到哪里停，可以引入一个剩余误差阈值 $\varepsilon$。若希望与上限的差距满足

$$
F_{\max}-F(N)\le \varepsilon
$$

则由上式可得：

$$
F_{\max}(1-p_x)^N \le \varepsilon
$$

进一步得到预算上限估计：

$$
\hat N=\left\lceil \frac{\ln(\varepsilon/F_{\max})}{\ln(1-p_x)} \right\rceil
$$

这个式子的含义是：如果系统已经很接近上限，再继续增加预算，新增收益可能不值得那一点额外延迟。

还可以补一个更工程化的量：若系统每次尝试成功就立刻早停，那么期望尝试次数为

$$
\mathbb{E}[T]=\sum_{i=1}^{N}(1-p_x)^{i-1}
=\frac{1-(1-p_x)^N}{p_x}
$$

它说明早停会显著降低平均成本。即便最大预算是 $N$，平均并不一定真的跑满 $N$ 轮。

---

## 核心机制与推导

测试时计算扩展最小可分成四个步骤：

1. 生成多个候选解。
2. 对候选解做验证或打分。
3. 在预算内继续搜索或提前停止。
4. 聚合并输出最终答案。

它的核心不是“让模型写更长文本”，而是“把一次性生成改成有搜索和筛选的求解过程”。

最小概率推导很直接。设一次尝试成功概率为 $p_x$，并假设不同尝试近似独立。则：

$$
P(\text{单次失败})=1-p_x
$$

连续 $N$ 次都失败的概率是：

$$
P(\text{全部失败})=(1-p_x)^N
$$

因此，至少成功一次的概率为：

$$
P(\text{至少成功一次})=1-(1-p_x)^N
$$

考虑系统上限 $F_{\max}$ 后，就得到：

$$
F(N)=F_{\max}\bigl(1-(1-p_x)^N\bigr)
$$

这个模型虽然简单，但足够解释大多数现象。它对应的是“best-of-N + 完美验证器”的近似情形。真实系统里，尝试之间通常并不完全独立，验证器也不完美，因此实际曲线会更复杂，但“先快后慢”的递减收益特征通常仍然成立。

### 一个可计算的玩具例子

假设：

- 单次成功概率 $p_x=0.2$
- 理论上限 $F_{\max}=1$

则不同预算下的成功率为：

| 预算 $N$ | 失败率 $(1-p_x)^N$ | 成功率 $F(N)$ | 新增收益 $\Delta F(N)$ |
| --- | ---: | ---: | ---: |
| 1 | 0.8000 | 0.2000 | 0.1600 |
| 2 | 0.6400 | 0.3600 | 0.1280 |
| 4 | 0.4096 | 0.5904 | 0.0819 |
| 8 | 0.1678 | 0.8322 | 0.0336 |
| 16 | 0.0281 | 0.9719 | 0.0056 |

如果换成更强的单次能力，比如 $p_x=0.5$，结果会变成：

| 预算 $N$ | 成功率 $F(N)$ | 新增收益 $\Delta F(N)$ |
| --- | ---: | ---: |
| 1 | 0.5000 | 0.2500 |
| 2 | 0.7500 | 0.1250 |
| 4 | 0.9375 | 0.0313 |
| 8 | 0.9961 | 0.0020 |
| 16 | 0.99998 近似 | 很小 |

这个对比说明了一个工程事实：强模型和弱模型都可能受益于测试时预算，但受益方式不同。

| 模型状态 | 单次成功率 | 预算扩展的典型效果 |
| --- | --- | --- |
| 强模型 | 已经较高 | 少量预算就能触顶，边际收益很快变小 |
| 中等模型 | 中等 | 最适合做预算优化，常有明显增益 |
| 很弱的模型 | 很低 | 如果验证器也弱，额外预算可能只是浪费 |

还可以把成功率曲线与 pass@k 联系起来。pass@k 的直观含义是“采样 $k$ 次，至少有一个候选正确的概率”。若验证器完美，它本质上就是上面的 $F(N)$。因此，测试时扩展最常见的增益形式，就是让 pass@k 高于 pass@1。

真实研究里，常把这类增长进一步拟合成幂律或对数收益曲线，并讨论 test-time scaling exponent。对初学者可以先把它理解成一句话：预算增长确实能带来性能增长，但增长速度通常明显低于线性。

这一点在推理模型报告里很常见。以 DeepSeek-R1 的公开报告为例，AIME 2024 上单次采样 `pass@1` 已经很高，而使用多样采样和一致性聚合后，结果还能继续提升。这类现象说明模型内部并不只有一条解题轨迹，额外预算能挖出被单次解码遗漏的正确路径。

把它和预训练 scaling law 放在一起看，会更清楚：

| 维度 | 预训练扩展 | 测试时扩展 |
| --- | --- | --- |
| 改变什么 | 参数、数据、训练 FLOPs | 推理时的采样、思考长度、验证 |
| 发生在哪个阶段 | 训练前或训练中 | 模型部署后 |
| 目标 | 提升基础能力 | 提升同一能力的兑现率 |
| 代价结构 | 前期成本极高、边际大 | 后期按请求付费、边际递减 |

因此，测试时计算扩展不是预训练的替代品，而是补充项。前者负责“同一个模型还能榨出多少潜力”，后者负责“模型本身到底有多强”。

---

## 代码实现

工程上，一个可复用的测试时扩展模块通常包含四部分：

| 模块 | 作用 | 最小实现 |
| --- | --- | --- |
| `sampler` | 生成候选解 | 多次调用模型或搜索器 |
| `validator` | 判断候选是否可信 | 单元测试、规则检查、判题器 |
| `controller` | 控制预算与早停 | 限制轮数、token、总时间 |
| `aggregator` | 聚合候选 | 多数投票、最高分选择、一致性判断 |

下面给一个可运行的 Python 玩具实现。它不依赖真实大模型，而是用随机过程模拟“单次尝试成功概率为 $p_x$”的情况，同时演示四件事：

1. 预算增加时，经验成功率会提高。
2. 经验成功率会逼近理论曲线。
3. 早停会降低平均尝试次数。
4. 验证器存在时，系统会优先返回已验证正确的候选。

```python
import math
import random
from collections import Counter
from dataclasses import dataclass

WRONG_POOL = ["1716", "1766", "1774", "1786", "1806"]

@dataclass
class SolveResult:
    prediction: str
    verified: bool
    attempts_used: int
    samples: list[str]

def sample_candidate(correct_answer: str, p_success: float) -> str:
    """Simulate one model attempt."""
    if random.random() < p_success:
        return correct_answer
    wrong_choices = [x for x in WRONG_POOL if x != correct_answer]
    return random.choice(wrong_choices)

def verify(candidate: str, expected: str) -> bool:
    """A toy validator: exact-match judge."""
    return candidate == expected

def aggregate(samples: list[str]) -> str:
    """Fallback aggregator when no candidate is verified."""
    return Counter(samples).most_common(1)[0][0]

def solve_with_budget(expected: str, p_success: float, budget: int, early_stop: bool = True) -> SolveResult:
    samples: list[str] = []

    for round_id in range(1, budget + 1):
        candidate = sample_candidate(expected, p_success)
        samples.append(candidate)

        if verify(candidate, expected):
            if early_stop:
                return SolveResult(
                    prediction=candidate,
                    verified=True,
                    attempts_used=round_id,
                    samples=samples,
                )

    # If early stop is disabled, return the first verified candidate if any.
    for idx, candidate in enumerate(samples, start=1):
        if verify(candidate, expected):
            return SolveResult(
                prediction=candidate,
                verified=True,
                attempts_used=budget,
                samples=samples,
            )

    return SolveResult(
        prediction=aggregate(samples),
        verified=False,
        attempts_used=budget,
        samples=samples,
    )

def estimate_metrics(trials: int, p_success: float, budget: int, early_stop: bool = True) -> tuple[float, float]:
    hits = 0
    total_attempts = 0

    for _ in range(trials):
        result = solve_with_budget(
            expected="1776",
            p_success=p_success,
            budget=budget,
            early_stop=early_stop,
        )
        if result.prediction == "1776":
            hits += 1
        total_attempts += result.attempts_used

    success_rate = hits / trials
    avg_attempts = total_attempts / trials
    return success_rate, avg_attempts

def theoretical_success(p_success: float, budget: int, f_max: float = 1.0) -> float:
    return f_max * (1 - (1 - p_success) ** budget)

def theoretical_avg_attempts_with_early_stop(p_success: float, budget: int) -> float:
    if p_success == 0:
        return float(budget)
    return (1 - (1 - p_success) ** budget) / p_success

if __name__ == "__main__":
    random.seed(7)

    trials = 20000
    p_success = 0.2

    print("budget | empirical_success | theory_success | empirical_avg_attempts | theory_avg_attempts")
    for budget in [1, 2, 4, 8]:
        empirical_success, empirical_avg_attempts = estimate_metrics(
            trials=trials,
            p_success=p_success,
            budget=budget,
            early_stop=True,
        )
        theory_success = theoretical_success(p_success, budget)
        theory_avg_attempts = theoretical_avg_attempts_with_early_stop(p_success, budget)

        print(
            f"{budget:>6} | "
            f"{empirical_success:>16.4f} | "
            f"{theory_success:>13.4f} | "
            f"{empirical_avg_attempts:>22.4f} | "
            f"{theory_avg_attempts:>18.4f}"
        )

    # Basic monotonicity checks.
    s1, _ = estimate_metrics(trials, p_success, 1, early_stop=True)
    s4, _ = estimate_metrics(trials, p_success, 4, early_stop=True)
    s8, _ = estimate_metrics(trials, p_success, 8, early_stop=True)

    assert s4 >= s1
    assert s8 >= s4
```

这段代码可以直接运行，输出会接近下面这种形态：

```text
budget | empirical_success | theory_success | empirical_avg_attempts | theory_avg_attempts
     1 |           0.2000 |        0.2000 |                 1.0000 |             1.0000
     2 |           0.3600 |        0.3600 |                 1.8000 |             1.8000
     4 |           0.5900 |        0.5904 |                 2.9520 |             2.9520
     8 |           0.8320 |        0.8322 |                 4.1611 |             4.1611
```

这个玩具实现有两个现实含义。

第一，它说明测试时扩展并不是“预算加倍，延迟一定也加倍”。如果启用了早停，平均尝试次数可能明显低于最大预算。

第二，它说明验证器很重要。如果没有验证器，系统只能在多个候选之间做启发式选择，例如多数投票；一旦候选共享同一类错误，多数投票就会失效。

真实工程里，代码生成是最典型的落地场景。系统可以这样组织：

| 阶段 | 做什么 | 为什么 |
| --- | --- | --- |
| 候选生成 | 生成多个函数实现 | 单次生成不稳定，保留多条路径 |
| 语法检查 | 编译或执行静态检查 | 先淘汰明显错误 |
| 单元测试 | 执行测试集 | 判断功能正确性 |
| 性能验证 | 跑 benchmark | 在多个正确解中选择更优者 |
| 输出聚合 | 选择通过测试且性能最好版本 | 把“语言质量”转成“系统确认的正确性” |

如果要把它做成线上服务，预算控制一般采用分层策略，而不是一上来就跑满预算：

1. 先做一次低成本单样本调用。
2. 若判断为高难度问题，再进入多采样。
3. 若仍失败，再进入强验证器或更大模型。
4. 最后才走人工流程或拒答。

这种调度方式的目标不是追求最高单题分数，而是优化整体吞吐、平均成本和尾延迟。

---

## 工程权衡与常见坑

测试时计算扩展最常见的误区，是把“更多推理”误认为“更多事实”。这两者不是一回事。额外推理可以提升搜索和筛选质量，但不会凭空补充缺失知识。

| 常见坑 | 典型表现 | 为什么会发生 | 规避方式 |
| --- | --- | --- | --- |
| 幻觉放大 | 想得越久，叙述越完整，但事实仍错 | 语言生成在补全，不是在检索 | 先检索，再推理；无法验证时拒答 |
| 长链退化 | 推理链越长，越容易偏离目标 | 中间错误会沿链条累积 | 限制推理长度，改用并行采样 |
| 验证器太弱 | 错解通过筛选 | 打分器只会偏好“像正确答案”的文本 | 优先使用可执行验证 |
| 全量跑满预算 | 平均延迟和成本过高 | 没有按难度路由，所有请求同待遇 | 分层预算，按难度调度 |
| 多数投票失效 | 多个候选都重复同一错误 | 采样缺乏多样性，错误高度相关 | 提升采样多样性，改变提示词或温度 |
| 过度相信 pass@k | 离线评测提升，线上体验一般 | 用户真正看到的是单次延迟和最终答案 | 同时看延迟、成本、稳定性 |

这里的“拒答”常记作 `abstain`，意思是系统主动承认“当前没有足够证据可靠回答”。在高风险任务里，拒答通常比胡乱生成更有工程价值。

一个典型反例是法律或医学问答。假设模型需要回答“某药品说明书在 2026 年 2 月版本中的禁忌症变化”。如果系统没有可靠文档检索，它不应该因为第一次答案不稳定，就继续让模型生成更长的思维链。更合理的路径是：

1. 先查文档原文。
2. 查不到就直接返回“无法可靠确认”。
3. 只有查到原文后，才在证据上做归纳和推理。

否则，长推理只是在放大错误叙事。

下面是一个更接近实际服务的伪代码框架：

```python
def guarded_reasoning(query, budget):
    for round_id in range(budget.num_samples):
        draft = generate(query, max_tokens=budget.reason_tokens)

        if looks_fact_sensitive(query):
            evidence = retrieve(query)
            if not evidence:
                return "无法可靠确认，建议先检索原文"
            if not grounded(draft, evidence):
                continue

        if passes_rule_check(draft):
            return draft

    return fallback_model_or_abstain(query)
```

这里最重要的不是代码细节，而是调度顺序：

- 先判断任务类型。
- 对事实敏感任务优先检索。
- 对可验证推理任务优先验证。
- 不要默认“更多思考”永远是最优策略。

还可以用一个简单决策表帮助新手判断：

| 现象 | 更可能的问题 | 优先处理手段 |
| --- | --- | --- |
| 同题多次结果不一致 | 搜索不足或采样噪声大 | 多采样 + 聚合 |
| 结果稳定但经常事实错 | 知识缺失 | 检索增强，不是加思考 |
| 结果大致对但最后一步错 | 局部推理失误 | 加验证器或中间检查 |
| 长答案更差 | 长链漂移 | 缩短链路、拆步验证 |

---

## 替代方案与适用边界

测试时计算扩展不是唯一可选策略。实际工程里，你通常在比较下面几种方案：

| 策略 | 主要成本 | 开发复杂度 | 适用场景 | 典型问题 |
| --- | --- | --- | --- | --- |
| 更大模型 | 单次调用更贵 | 低 | 需要通用能力快速提升 | 成本高，部署重 |
| 多轮采样 | 请求次数增加 | 中 | 数学、代码、规划 | 延迟随样本数上升 |
| 微调 | 前期训练贵，推理便宜 | 高 | 稳定垂直任务 | 数据和标注成本高 |
| Prompt 工程 | 最便宜 | 低 | 格式约束、轻量任务 | 提升有限 |
| 检索增强 | 建知识库和召回链路 | 中 | 事实密集、知识常更新 | 召回质量成瓶颈 |
| 工具调用 | 依赖执行器或外部系统 | 中到高 | 计算、数据库、搜索 | 系统复杂度增加 |

对初学者，一个简单判断规则足够实用：

| 问题本质 | 更合适的优先策略 |
| --- | --- |
| 模型会做，但第一次常答错 | 测试时计算扩展 |
| 模型不知道，缺事实 | 检索增强或换模型 |
| 任务稳定、重复出现很多次 | 微调或蒸馏 |
| 只是输出格式不稳定 | Prompt 工程 |
| 结果必须可执行验证 | 多采样 + 验证器 |

还可以用一个简单价值函数帮助做预算选择：

$$
\text{Value}(N)=\frac{F(N)-F(1)}{\text{Latency}(N)-\text{Latency}(1)}
$$

它表示每增加一单位延迟，换回多少额外正确率。这个量越高，说明加预算越划算；越低，则说明已经接近平台期。

如果进一步引入单位成本 $C(N)$，还可以写成：

$$
\text{ROI}(N)=\frac{F(N)-F(1)}{C(N)-C(1)}
$$

工程上通常并不会真的用一个公式拍板，但这两个量有助于建立直觉：预算不是越大越好，而是要看新增收益是否值得。

一个常见的线上组合策略是：

1. 先用中等模型做单次回答。
2. 检测到高难度推理特征时，开启多采样。
3. 若候选分歧很大，再启用验证器。
4. 若仍不能确认，回退到更大模型、工具链路或人工流程。

因此，测试时扩展更适合被看成“推理路由器的一部分”，而不是孤立存在的万能按钮。

它尤其适合：

- 数学推理
- 代码生成与修复
- 可执行验证的搜索任务
- 高价值、低频、允许更长延迟的请求

它不适合默认用于：

- 强事实问答
- 强实时场景
- 低容错高风险场景
- 无法验证、也无法检索的开放式问题

---

## 参考资料

下面给一组更完整、且彼此有衔接关系的参考资料。顺序按“先建立基本概念，再看验证器和真实系统，最后看风险边界”排列。

| 来源 | 主题摘要 | 建议阅读顺序 |
| --- | --- | --- |
| Jason Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*（2022） | 说明为什么显式中间推理步骤能提升复杂推理任务表现，是理解“多想几步”最早也最关键的背景材料 | 1 |
| Xuezhi Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*（2022） | 给出“多次采样 + 聚合”这一最经典的测试时扩展范式，适合理解 `best-of-N` 和多数投票 | 2 |
| Karl Cobbe et al., *Training Verifiers to Solve Math Word Problems*（2021） | 解释为什么验证器重要，并展示“生成多个候选，再由验证器选优”的收益 | 3 |
| Charlie Snell et al., *Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters*（2024/ICLR 2025） | 系统讨论测试时计算扩展的效率边界、任务难度分层和 compute-optimal 调度 | 4 |
| DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*（2025） | 公开推理模型在数学、代码等任务上使用更长推理和一致性聚合后的结果，是现实系统案例 | 5 |
| Various analyses on reasoning limits such as *Reasoning or Reciting?*（2023） | 讨论为什么很多看似“会推理”的结果其实夹杂复述、记忆和事实幻觉，适合理解任务边界 | 6 |

如果你想按问题类型来读，可以这样分：

| 你最关心的问题 | 优先读什么 |
| --- | --- |
| 为什么多采样有效 | Wei 2022 + Wang 2022 |
| 为什么验证器很关键 | Cobbe 2021 |
| 怎样做预算最划算 | Snell 2024 |
| 推理模型在真实评测上表现如何 | DeepSeek-R1 2025 |
| 为什么事实型任务不该盲目加思考 | *Reasoning or Reciting?* 及相关失败分析 |

参考链接：

- Wei et al. 2022: https://arxiv.org/abs/2201.11903
- Wang et al. 2022: https://arxiv.org/abs/2203.11171
- Cobbe et al. 2021: https://arxiv.org/abs/2110.14168
- Snell et al. 2024: https://arxiv.org/abs/2408.03314
- DeepSeek-R1 2025: https://arxiv.org/abs/2501.12948
- Reasoning or Reciting? 2023: https://arxiv.org/abs/2307.02477

阅读顺序也可以压缩成一句话：先理解“为什么多试几次会提高命中率”，再理解“为什么验证器决定增益能否兑现”，最后再看“哪些任务不能靠多想解决”。
