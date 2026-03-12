## 核心结论

自洽性，白话说就是“不要只信模型第一次写出来的那份代码，而是让它多写几份，再看哪些结果彼此一致”，在代码生成里最有效的落地点不是“比较字符串是否一样”，而是“执行测试，比较功能是否等价”。

这件事在代码任务上成立，是因为代码答案通常没有唯一写法。两个实现的源代码可以完全不同，但只要对同一组输入给出相同且正确的输出，它们就在功能上是一致的。于是，多路径采样加执行验证，能够把“偶然写对一次”变成“稳定选对一次”。

CodeT 的核心结果很直接：对同一道题采样多个候选解，再生成测试并做双重执行一致性筛选，`code-davinci-002` 在 HumanEval 上的 pass@1 从 47.0% 提升到 65.8%。MPSC 再往前走一步，不把测试或规范当成天然更可靠的“裁判”，而是把 Solution、Specification、Test 三类输出都视为模型对同一问题的不同视角，在图上联合打分；论文摘要报告 ChatGPT 在 HumanEval、MBPP、CodeContests 上分别提升 15.91%、6.43%、9.37%。

下面这张表先给出最重要的对比：

| 方法 | 验证信号 | HumanEval pass@1 | 说明 |
| --- | --- | ---: | --- |
| 单次 greedy / baseline | 无 | 47.0% | 只取一次输出，最便宜，也最不稳定 |
| Self-Consistency 思路迁移到代码 | 多样本一致性 | 取决于验证器质量 | 关键不是文本一致，而是功能一致 |
| CodeT | 生成测试 + 双重执行 | 65.8% | 用执行结果替代答案比对 |
| MPSC | 解 / 规范 / 测试三视角图优化 | 在 ChatGPT 上显著增益 | 让不同视角互相校准 |

---

## 问题定义与边界

问题可以写成：给定自然语言需求 $q$，模型生成多个候选程序 $\{c_1,\dots,c_n\}$，我们需要从中选出最可能正确的一个，而不是盲选第一份。

这里有三个边界必须先说清楚。

第一，代码生成不是标准问答。问答任务常用“最终答案相同”做投票；代码任务不能这么做，因为不同实现可能都正确。这里的“一致”应理解为“在足够多的输入上行为一致”。

第二，测试并不总是可靠。测试，白话说就是“拿一组输入输出样例去试代码是否按预期工作”。如果测试太弱，错误实现也可能通过；如果测试本身是模型生成的，它也会带偏差。

第三，采样数量 $N$ 决定上限，预算决定下限。$N$ 太小，多样性不够；$N$ 太大，成本和延迟上升。工程里常见范围是 3 到 40，不存在脱离预算的“最优值”。

一个玩具例子能说明边界。题目是“计算列表平均值”，模型给出三个版本：

- A：`sum(nums) / len(nums)`，正确
- B：手写循环累加后除以长度，正确
- C：把 `len(nums)` 错写成 `len(nums)-1`，错误

如果只取第一次输出，碰到 C 就直接失败。若采样 3 次并执行测试，例如 `avg([2,4,6]) == 4`、`avg([1]) == 1`，A 和 B 会通过，C 会暴露。这里真正起作用的不是“多数代码长得像”，而是“多数代码在可执行行为上相同”。

流程可以简化成下面这样：

```text
问题描述
  ↓
采样多个 Solution / Specification / Test
  ↓
两两验证是否一致
  - 代码 vs 测试：能否通过
  - 代码 vs 规范：是否满足前后置条件
  - 规范 vs 测试：测试是否落在规范允许范围内
  ↓
构图与打分
  ↓
选择得分最高且执行稳定的代码
```

---

## 核心机制与推导

Self-Consistency 原本用于链式推理，核心做法是“采样多条推理路径，再对最终答案做边缘化或投票”。迁移到代码生成后，投票对象从“答案字符串”变成“功能行为”。

CodeT 先做了第一层改造：生成很多候选代码，再生成很多测试。随后运行所有“代码-测试”对，并进一步看不同代码在测试上的输出是否形成一致群体。论文把这个过程称为 dual execution agreement，白话说就是“既看代码能不能过测试，也看多个代码是否在同一批测试上表现一致”。

MPSC 做了第二层改造：不再默认测试比代码更可信，而是把三类对象都放进一个三部图
$$
G=(V,E),\quad V=V_{\text{sol}}\cup V_{\text{spec}}\cup V_{\text{test}}
$$
其中节点是候选解、规范、测试，边权 $W_{ij}$ 表示两个节点在功能上有多一致。

它的 inter-consistency，也就是“跨视角一致性”，定义为：
$$
L_{\text{inter}}=\sum_{(v_i,v_j)\in E} W_{ij}\bigl(f(v_i)-f(v_j)\bigr)^2
$$
直觉是：如果一个解和一个测试、或者一个解和一个规范高度一致，它们的分数就不该差太多。

它的 intra-consistency，也就是“同视角内部一致性”，定义为：
$$
L_{\text{intra}}=\frac{1}{2}\sum_{v_i\in V}\left|f(v_i)-\phi(v_i)\right|^2
$$
这里 $\phi(v_i)$ 是节点自身的内部一致性估计，白话说就是“这个候选在同类候选里像不像一个高置信样本”。在代码场景里，这个量可来自词汇相似度、语义相似度、Bayes risk 风格的重排序信号等。

最终优化目标是：
$$
\min_f \ \alpha L_{\text{inter}} + (1-\alpha)L_{\text{intra}}
$$
其中 $\alpha$ 是平衡项，用来决定“更信跨视角共识”还是“更信单视角内部稳定性”。

看一个最小例子。假设有三个候选解 A、B、C：

| 候选 | 通过生成测试 | 满足规范 | 与其他候选行为一致 |
| --- | --- | --- | --- |
| A | 是 | 是 | 与 B 高一致 |
| B | 是 | 是 | 与 A 高一致 |
| C | 否 | 部分失败 | 与 A/B 低一致 |

优化后通常会得到 $f(A)\approx f(B)>f(C)$。这就是“投票”在图上的版本。它不要求 A 和 B 代码一样，只要求它们在不同视角下描述的是同一个功能。

---

## 代码实现

工程实现可以先从最小闭环开始：多采样、自动测试、执行筛选、再排序。下面这段 Python 代码可直接运行，演示一个简化版“功能自洽性选择器”。

```python
from typing import Callable, List, Tuple

def choose_best(candidates: List[Callable[[str], str]],
                tests: List[Tuple[str, str]]) -> int:
    scores = []
    for fn in candidates:
        passed = 0
        outputs = []
        for x, expected in tests:
            y = fn(x)
            outputs.append(y)
            if y == expected:
                passed += 1
        scores.append((passed, outputs))

    # 第一层：按通过测试数排序
    best_pass = max(p for p, _ in scores)
    best_ids = [i for i, (p, _) in enumerate(scores) if p == best_pass]

    # 第二层：在高分候选里做简单一致性打分
    def agreement(i: int) -> int:
        out_i = scores[i][1]
        total = 0
        for j, (_, out_j) in enumerate(scores):
            if i == j:
                continue
            total += sum(a == b for a, b in zip(out_i, out_j))
        return total

    return max(best_ids, key=agreement)

def reverse_ok(s: str) -> str:
    return s[::-1]

def reverse_ok_loop(s: str) -> str:
    out = []
    for ch in s:
        out.insert(0, ch)
    return "".join(out)

def reverse_bad(s: str) -> str:
    return s  # bug: 没有反转

tests = [("abc", "cba"), ("a", "a"), ("ab", "ba")]
cands = [reverse_ok, reverse_ok_loop, reverse_bad]

best = choose_best(cands, tests)
assert best in (0, 1)
assert cands[best]("hello") == "olleh"
```

这段代码对应四步：

1. `candidates`：采样多个代码实现。
2. `tests`：准备测试集，可以是人工测试，也可以是模型生成测试。
3. `passed`：执行测试，用功能是否通过替代文本比对。
4. `agreement`：在通过数接近时，优先选与其他候选行为更一致的实现。

如果把它扩展成更接近论文的伪代码，会是这样：

```python
for problem in problems:
    solutions = sample_model(problem, n=5)      # 多路径采样
    tests = sample_tests(problem, n=20)         # 生成测试
    specs = sample_specs(problem, n=5)          # 生成前/后置条件
    run_results = run_all(solutions, tests)     # 执行验证
    graph = build_multiview_graph(solutions, specs, tests, run_results)
    scores = optimize_score(graph)              # 图优化：inter + intra
    best = select_best(scores, run_results)     # 选稳定且高分解
```

真实工程例子通常出现在 CI 或 AI coding agent。做法不是“让模型直接改主分支代码”，而是：

- 让模型对同一修复任务生成 5 到 10 个补丁
- 自动补充回归测试和边界测试
- 在隔离环境执行所有补丁
- 对通过测试的候选再做一致性重排
- 只把最稳定的那个送到人工 review 或自动合并队列

这样做的收益不是让模型“更聪明”，而是把模型原本波动很大的单次输出，变成一个更可控的候选选择过程。

---

## 工程权衡与常见坑

最大的收益来自稳定性，最大的代价来自采样与执行。

| 采样数 | 额外测试执行 | 成本 | 延迟 | 常见收益 |
| ---: | ---: | --- | --- | --- |
| 1 | 低 | 最低 | 最低 | 基线 |
| 3 | 中 | 可接受 | 可并行 | 明显减少偶发错误 |
| 5 | 中高 | 常见工程折中 | 可并行 | 更适合上线前筛选 |
| 10+ | 高 | 显著上升 | 需要批处理 | 适合离线评测或高价值任务 |

一个常被引用的实现示例是：在 AQuA 复现中，5 条 reasoning paths 的 Self-Consistency 评估成本从约 `$0.15` 增到 `$0.75`。这不是通用报价，但能说明量级关系：多采样通常是线性涨成本的。

常见坑主要有五个。

第一，错的共识。5 个候选里 3 个都犯同一个边界错误，多数投票仍然会错。比如都把空列表平均值处理成 0，却违背规范要求抛异常。

第二，测试过弱。只测常规样例，不测空输入、负数、重复元素、极端长度，错误实现会伪装成一致。

第三，多样性不足。温度太低、提示词太死，采样出来只是同一段代码的小改写，自洽性几乎没有新增信息。

第四，验证器本身有偏差。模型生成的规范、测试、参考实现都可能一起错，所以不能无条件把“测试通过”理解为“真实正确”。

第五，执行环境污染。真实工程里如果依赖网络、时间、随机数、数据库状态，不同候选的执行结果可能不可重复，导致一致性信号失真。

因此，部署时更合理的做法是：先用 CodeT 风格执行筛选挡住明显错误，再用 MPSC 风格图排序处理“都差不多过了测试，但哪个更稳”的问题。

---

## 替代方案与适用边界

如果任务不是“生成一个可执行程序”，自洽性的变体选择也会不同。

| 方法 | 核心思想 | 更适合的任务 | 局限 |
| --- | --- | --- | --- |
| Self-Consistency | 多路径采样后多数选择 | 单一答案推理 | 少数正确答案易被淹没 |
| Ranked Voting Self-Consistency | 每条路径给出排序答案，再做排序投票 | 多选题、可排序候选 | 需要模型能稳定给出排名 |
| Mirror-Consistency | 让模型反思少数派与多数派冲突 | 不确定性高、需校准置信度 | 额外反思步骤增加时延 |
| MPSC | 多视角构图联合优化 | 代码生成、可执行验证 | 实现复杂，依赖验证基础设施 |

Ranked Voting 的关键改动是：每次生成不只给一个答案，而是给一个候选排序，再用 instant-runoff、Borda count、MRR voting 等规则综合所有排序。它适合“候选可枚举”的任务，比如多选题，不适合直接拿来选开放式代码实现。

Mirror-Consistency 的核心是：不要忽略少数派。白话说就是“模型自己看看，为什么有些采样结果和多数不一样”。如果这些少数派反复出现，说明问题可能不简单，应该再采样、再反思、再校准置信度。它对减少过度自信很有价值，但不直接替代代码执行验证。

所以，适用边界可以压缩成一句话：

- 能执行，就优先用执行验证驱动自洽性。
- 候选可排序，就考虑 Ranked Voting。
- 不确定性高、少数意见重要，就考虑 Mirror-Consistency。
- 需要把代码、测试、规范统一起来，就用 MPSC。

---

## 参考资料

- Self-Consistency Improves Chain of Thought Reasoning in Language Models  
  https://arxiv.org/abs/2203.11171

- CodeT: Code Generation with Generated Tests  
  https://openreview.net/forum?id=ktrw68Cmu9c

- Enhancing Large Language Models in Coding Through Multi-Perspective Self-Consistency  
  https://aclanthology.org/2024.acl-long.78/

- Mirror-Consistency: Harnessing Inconsistency in Majority Voting  
  https://aclanthology.org/2024.findings-emnlp.135/

- Ranked Voting based Self-Consistency of Large Language Models  
  https://aclanthology.org/2025.findings-acl.744/

- Self-consistency 复现仓库中的成本示例  
  https://github.com/akpe12/Self-consistency
