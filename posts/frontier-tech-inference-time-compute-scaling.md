## 核心结论

推理时计算 Scaling，指的是不改模型参数，只在推理阶段追加计算预算，用更多采样、分支搜索、反思、工具调用或验证步骤，把复杂任务的正确率继续往上推。白话说，就是模型不一定要“重新训练得更聪明”，也可以靠“多试几次并挑对的”变得更强。

它最核心的经验公式可以写成：

$$
\text{Inference-Time Gain} \approx \text{额外采样数} \times \text{验证器质量}
$$

这里“验证器”是能检查答案对不对的机制，白话说就是裁判。裁判越可靠，多采样带来的收益越能落到最终结果上。

对零基础读者，最直观的理解是：单次输出像只做一遍题；推理时计算 Scaling 像做多轮实验，先收集多个候选答案，再用测试、规则或外部工具挑出最靠谱的一个。o1 类方法之所以有效，核心不只是“会想”，而是把更多 token、更多中间步骤、更多候选分支换成更高的最终命中率。

一个常见误解是“只要让模型输出更长思维链就一定更强”。这不成立。额外 token 只有在它真的扩大了解空间，或者让验证更容易时，才会产生收益。否则只是更慢、更贵，而且不一定更准。

玩具例子：让模型解一个两位数乘法题，单次输出可能算错；如果让它独立做 5 次，再用程序校验哪个答案满足乘法结果，最终正确率通常会明显上升。

真实工程例子：代码补丁生成。单次生成一个 patch 并直接交付，风险很高；如果生成 5 到 20 个 patch，再跑单元测试、静态检查和回归检查，往往能用较便宜模型达到甚至超过更贵模型单次生成的效果。

| 方案 | 候选数 $n$ | 是否验证 | 典型成本 | 典型正确率 |
|---|---:|---|---|---|
| 单次输出 | 1 | 否/弱 | 低 | 基线 |
| 多采样 | $n>1$ | 弱验证 | 中 | 有提升，但容易饱和 |
| 多采样 + 强验证 | $n>1$ | 强验证 | 中到高 | 提升最大 |
| 搜索 + 工具调用 + 验证 | 分支式 | 强验证 | 高 | 复杂任务最强 |

---

## 问题定义与边界

更正式地说，Inference-Time Scaling 是在测试时增加计算量，以提高复杂任务表现，而不是靠训练期追加参数、数据或训练轮数。常见手段包括：

1. 多采样：同一个问题生成多个独立候选。
2. 搜索：把解题过程拆成多步，探索多个分支。
3. 反思：让模型检查前一步是否有错。
4. 工具调用：调用解释器、检索、计算器、编译器等外部系统。
5. 验证与重排：对候选进行打分、过滤、排序。

它的收益边界很明确：任务必须尽量“可验证”。“可验证”不是说必须有人工评审，而是最好有可程序化的正确性信号，比如测试是否通过、公式是否成立、约束是否满足、最终路径是否合法。没有这个条件，多采样通常只能提高“候选覆盖率”，却不一定能提高“最终选中正确答案的概率”。

这一点可以写成一个简单边界：

$$
0 \le \text{coverage}(n) \le 1,\qquad
\text{final\_accuracy}(n) \le \text{coverage}(n)\cdot q_v
$$

其中 $\text{coverage}(n)$ 表示前 $n$ 个候选里至少有一个正确答案的概率，$q_v$ 表示验证器把正确答案挑出来的能力。白话说，候选里就算已经有正确答案，如果裁判认不出来，最终结果还是上不去。

下面这个表格说明了任务边界：

| 任务类型 | 是否容易自动验证 | 多采样收益 | 原因 |
|---|---|---|---|
| 数学题 | 高 | 高 | 可直接检查结果或推导 |
| 代码修复 | 高 | 高 | 可跑测试、编译、lint |
| 路径规划 | 中到高 | 高 | 可检查约束与代价 |
| 开放式问答 | 中 | 中等 | 很难定义唯一正确 |
| 小说创作 | 低 | 低 | 几乎没有客观验证器 |
| 营销文案 | 低 | 低 | 多样性高但正确性弱 |

玩具例子：让模型写一个“反转字符串”函数。这个任务适合多采样，因为你可以立刻跑输入输出测试。

真实工程例子：SWE-bench Lite 这类代码修复任务适合推理时计算 Scaling，因为每个补丁都能被仓库测试集自动验证。相反，如果任务是“写一篇更有感染力的品牌故事”，多采样只能带来风格多样性，不能稳定带来“正确率”提升。

---

## 核心机制与推导

这一类方法最重要的观测，不是“多试几次总会更好”这么简单，而是它呈现出近似可预测的缩放规律。论文中常用的经验形式是：

$$
\text{coverage}(n)\approx 1-\exp(-\alpha n^\beta)
$$

其中：

- $\alpha$ 控制起步速度，白话说就是前几次采样有没有明显收获。
- $\beta$ 控制增长形状，白话说就是收益衰减得快不快。
- $n$ 是采样次数。

把上式做变形：

$$
1-\text{coverage}(n)\approx \exp(-\alpha n^\beta)
$$

再取对数：

$$
\log(1-\text{coverage}(n)) \approx -\alpha n^\beta
$$

如果继续对右侧做幂律近似观察，常见实验里会出现接近 log 线性的趋势。直观含义是：样本数不断增加时，未覆盖概率持续下降，但下降速度会越来越慢，也就是典型的 diminishing returns，边际收益递减。

可以用一个新手友好的图像来理解：横轴是采样次数 $n$，纵轴是 coverage。曲线一开始上升很快，之后逐渐变平。原因不是后面的采样没用，而是“容易命中的解”前面已经命中，剩下的是更难碰到的解。

下面给一个参数对比：

| $\alpha$ | $\beta$ | 曲线特征 | 解释 |
|---:|---:|---|---|
| 0.2 | 0.5 | 起步慢，后期平缓 | 模型很少直接命中正确思路 |
| 0.5 | 0.7 | 较理想 | 多采样能持续带来收益 |
| 0.8 | 0.9 | 起步很快 | 前几个样本就能覆盖大量正确解 |
| 0.3 | 0.3 | 很快饱和 | 问题空间窄，额外采样价值低 |

“多采样 + 验证”为何有效？因为最终成功率不是单次生成能力，而是两段概率的组合：

$$
P(\text{成功}) = P(\text{至少有一个候选正确}) \times P(\text{验证器选中它})
$$

第一项靠采样和搜索提高，第二项靠验证器提高。二者缺一不可。

玩具例子：猜一个四位密码，每次模型只会排除一部分组合。单次成功率很低，但如果每次尝试都能得到“位置对了几个”的反馈，那么多轮搜索会迅速缩小空间。这里反馈就是验证信号。

真实工程例子：代码修复里，候选 patch 的“覆盖率”可能随样本数增长而显著上升；但如果测试集过弱，错误 patch 也可能通过，验证器就会把噪声当成正确解，最终收益被吞掉。这就是为什么工程上经常是“验证瓶颈”先到，而不是“采样瓶颈”先到。

---

## 代码实现

工程上最好把系统拆成三层：采样器、验证器、聚合器。采样器负责生成候选，验证器负责给每个候选打分，聚合器负责选最优。这样模型、规则和排序逻辑可以解耦。

下面是一个可运行的 Python 玩具实现。它不依赖真实大模型，而是用模拟候选来展示“采样 -> 验证 -> 选最优”的主流程。

```python
from dataclasses import dataclass
from typing import List, Callable
import math

@dataclass
class Candidate:
    text: str
    score: float

def mock_model_samples(question: str, n: int) -> List[str]:
    # 模拟模型输出：其中少数候选是正确的
    pool = [
        "12 * 13 = 156",
        "12 * 13 = 146",
        "12 * 13 = 154",
        "12 * 13 = 156",
        "12 * 13 = 166",
    ]
    return [pool[i % len(pool)] for i in range(n)]

def verifier(answer: str) -> float:
    # 验证器：检查等式是否成立，成立得 1.0，否则 0.0
    try:
        left, right = answer.split("=")
        a, b = left.strip().split("*")
        lhs = int(a.strip()) * int(b.strip())
        rhs = int(right.strip())
        return 1.0 if lhs == rhs else 0.0
    except Exception:
        return 0.0

def select_best(question: str, n: int,
                sampler: Callable[[str, int], List[str]],
                judge: Callable[[str], float]) -> Candidate:
    candidates = []
    for text in sampler(question, n):
        candidates.append(Candidate(text=text, score=judge(text)))
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[0]

def coverage(alpha: float, beta: float, n: int) -> float:
    return 1 - math.exp(-alpha * (n ** beta))

best = select_best("12*13=?", 5, mock_model_samples, verifier)
assert best.text == "12 * 13 = 156"
assert best.score == 1.0
assert coverage(0.5, 0.7, 10) > coverage(0.5, 0.7, 1)
```

如果换成真实系统，接口通常是这样：

```python
def solve(prompt, n, model, verifier):
    candidates = []
    for _ in range(n):
        answer = model.sample(prompt)      # 1. 采样，可能含更长 CoT
        score = verifier(answer)           # 2. 验证，可能跑测试/规则/工具
        candidates.append((answer, score))
    candidates.sort(key=lambda x: x[1], reverse=True)
    if candidates[0][1] < 0.5:
        return model.sample(prompt)        # fallback：退化为单次输出
    return candidates[0][0]
```

候选排序表可以长这样：

| 候选答案 | 验证分数 | 排名 |
|---|---:|---:|
| `12 * 13 = 156` | 1.0 | 1 |
| `12 * 13 = 156` | 1.0 | 2 |
| `12 * 13 = 154` | 0.0 | 3 |
| `12 * 13 = 146` | 0.0 | 4 |
| `12 * 13 = 166` | 0.0 | 5 |

真实工程例子：代码代理修 bug 时，`model.sample` 生成 patch，`verifier` 跑单元测试、类型检查、仓库编译和安全规则，`select_best` 再优先选择“测试全过且改动最小”的补丁。这比只看模型自己说“我修好了”要可靠得多。

---

## 工程权衡与常见坑

推理时计算 Scaling 不免费。它把训练期一次性投入，换成了线上每个请求的重复投入，所以成本、延迟和非确定性会一起上升。

最重要的工程权衡有三个：

1. 采样预算。$n$ 越大，覆盖率越高，但成本近似线性增长。
2. 验证预算。验证器越强，选对概率越高，但验证本身也可能很贵。
3. 延迟预算。并行采样可以降墙钟时间，但会抬高瞬时资源占用。

一个粗略的成本式子是：

$$
\text{Total Cost} \approx n \cdot C_{sample} + n \cdot C_{verify} + C_{aggregate}
$$

如果验证器很弱，增加 $n$ 只是放大噪声；如果验证器很贵，收益可能被成本吞没。

常见坑如下：

| 坑点 | 影响 | 缓解策略 |
|---|---|---|
| 没有可靠验证器 | 多采样后难以选对 | 先缩小任务到可验证子问题 |
| 只延长输出 token | 更慢更贵，不一定更准 | 优先增加独立候选和外部验证 |
| 测试集过弱 | 错误答案被误判为正确 | 补强测试、加入静态检查 |
| 候选高度相似 | coverage 增长慢 | 提高采样多样性，调整温度或策略 |
| 工具调用链过长 | 延迟和失败点上升 | 控制最大步数，设置超时 |
| 在线服务低延迟要求 | 用户体验下降 | 只对高价值请求启用多采样 |

玩具例子：同一道计算题采样 20 次，但验证器只是“答案长度看起来像整数”，这种验证几乎没意义。看起来做了很多计算，实际没有提高正确率。

真实工程例子：在 CI/CD 中生成 5 个代码补丁并跑全量测试，效果可能很好；但如果全量测试要 25 分钟，那么 5 个候选就是 125 分钟级别的验证成本。此时通常要改成“先快速 smoke test 过滤，再对前 1 到 2 个候选跑重测试”，否则系统吞吐会崩。

验证失败时的降级策略通常要显式写清楚：

```python
def safe_pick(candidates, min_score=0.8):
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    best_answer, best_score = candidates[0]
    if best_score < min_score:
        return "fallback_single_shot"
    return best_answer
```

---

## 替代方案与适用边界

推理时计算 Scaling 不是唯一路线，它和更大模型、提示工程、微调是互补关系。

| 方案 | 核心投入 | 优点 | 缺点 | 最适合场景 |
|---|---|---|---|---|
| Inference-Time Scaling | 在线推理算力 | 不改模型即可增强复杂任务 | 延迟和成本随请求上升 | 数学、代码、规划 |
| Prompt Engineering | 提示设计 | 成本低、落地快 | 上限有限 | 轻量优化 |
| Fine-tuning | 训练数据与训练成本 | 领域稳定性更强 | 需要数据和训练流程 | 固定垂类任务 |
| Larger Model | 更大模型调用费或训练费 | 单次能力强 | 成本高，受供应商限制 | 通用高难任务 |

两类成本可以结构化地比较：

$$
\text{Cost}_{ITS} \approx Q \cdot n \cdot (C_{sample}+C_{verify})
$$

$$
\text{Cost}_{train} \approx C_{data}+C_{train}+Q \cdot C_{serve}
$$

其中 $Q$ 是请求量。若请求量不大、任务又高度可验证，Inference-Time Scaling 往往划算；若请求量极大且任务模式稳定，训练或微调后的长期均摊成本可能更低。

它的适用边界也很清楚：

1. 可验证任务，优先考虑它。
2. 高价值低频请求，适合它。
3. 开放式创作、强实时服务，不应优先依赖它。

玩具例子：写小说段落时，没有统一正确答案，多采样只能产生多个版本，无法稳定定义“哪个一定更对”。这类任务更适合 prompt 设计或风格微调。

真实工程例子：客服系统如果要求 300ms 内响应，复杂的多分支搜索通常不现实；但离线代码审查、批量数学求解、自动规划排程这类任务，对时延不敏感而对正确率敏感，推理时计算 Scaling 就非常合适。

---

## 参考资料

| 文献 | 主张 | 数据或结论支持 |
|---|---|---|
| Bradley Brown et al., *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling* | 重复采样可形成推理时 Scaling，coverage 随样本数增长呈可预测规律 | SWE-bench Lite 上 DeepSeek-Coder-V2-Instruct 从单样本 15.9% 提升到 250 样本 56%；并指出在可自动验证任务上收益最明显。链接：https://huggingface.co/papers/2407.21787 |
| Vidhisha Balachandran et al., *Inference-Time Scaling for Complex Tasks: Where We Stand and What Lies Ahead* | 推理时 Scaling 的收益依赖任务类型、反馈强度与验证质量，额外 token 不必然转化为更高准确率 | 微软研究总结了并行多次调用与顺序反馈式推理两类扩展路径，并强调复杂任务上的收益梯度和边界。链接：https://www.microsoft.com/en-us/research/publication/inference-time-scaling-for-complex-tasks-where-we-stand-and-what-lies-ahead/ |
| Eric Zhao et al., *Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification* | 验证能力本身也可被扩展，自验证是当前瓶颈之一 | 论文指出前沿模型原生自验证能力偏弱，但通过比较多个候选、改变输出风格和加强审查，可提升最终搜索效果。链接：https://huggingface.co/papers/2502.01839 |
| Microsoft Research 文章 *Eureka Inference-Time Scaling Insights* | 从工程视角总结推理时 Scaling 的并行与顺序模式 | 强调“多跑几次模型”只能给出下界，强反馈与强验证器仍有大量可挖空间。链接：https://www.microsoft.com/en-us/research/articles/eureka-inference-time-scaling-insights-where-we-stand-and-what-lies-ahead/ |
