## 核心结论

CodeT 的核心不是“让模型多写几份代码再投票”，而是让模型**同时生成候选代码和候选测试**，再把两者做全量交叉执行。这里的“双重执行验证”指两层信号同时参与排序：

1. 一个候选解通过了多少测试。
2. 有多少别的候选解与它在同一组测试上表现一致。

如果把代码解看成“答卷”，把测试看成“验收题”，那么 CodeT 选的不是“最像标准答案的文字”，而是“在最多验收题上表现稳定，而且有最多其他答卷复现同样功能”的那一组。

论文报告的代表性结果是：在 HumanEval 上，`code-davinci-002` 的 pass@1 从 47.0% 提升到 65.8%；后续 Parsel + GPT-4 再结合 CodeT 分数做筛选，可把 HumanEval 的 pass@1 推到 85.1%。这说明在代码任务里，“功能一致性”通常比“文本相似”或“链路投票”更接近真实正确性。

| 方法 | 基础模型/流程 | HumanEval pass@1 |
|---|---|---:|
| 原始 Codex 采样后随机取 1 个 | `code-davinci-002` baseline | 47.0% |
| CodeT | `code-davinci-002` + 生成测试 + 双执行一致性 | 65.8% |
| GPT-4 + Parsel + CodeT score | 结构化中间表示 + 生成测试筛选 | 85.1% |

玩具例子可以这样理解：你有 8 份“番茄汤食谱”和 20 次“试喝测试”。一份食谱如果在更多测试下味道都对，而且还有多位厨师做出了相同味道，这份食谱就更可信。CodeT 做的就是这种“试喝 + 同味共识”的联合排序。

---

## 问题定义与边界

问题定义很具体：给定一道编程题，模型会采样出多个候选程序，但我们最后通常只能交付 1 个。难点不在“能不能生成”，而在“怎么从一堆候选里选出最可能真的对的那个”。

传统做法有两个短板：

| 做法 | 核心依赖 | 主要问题 |
|---|---|---|
| 只看模型概率/置信度 | 语言模型自身打分 | 文字流畅不等于功能正确 |
| 只做 self-consistency | 多条推理链投票 | 推理过程一致，不代表代码跑出来一致 |
| 只靠人工测试 | 人写测试集 | 成本高，很多真实场景根本没有 |

这里的 `pass@1` 是“只取 1 个候选时的正确率”，白话就是“用户真正拿到的那一个程序有多大概率能用”。CodeT 的目标正是提升这个指标，而不是单纯堆高 `pass@k`。

它的边界也必须说清楚：

1. CodeT 不是证明器。它不能形式化证明程序正确，只是在“生成测试可覆盖到的范围内”做更强的筛选。
2. 它依赖生成测试的质量。如果测试本身错了，或者只覆盖了很窄的输入区域，共识就可能被错误放大。
3. 它更适合“可以执行、可以判定输出是否正确”的任务。对交互式程序、外部副作用很多的程序、依赖环境状态的程序，落地难度更高。

新手版可记成一条流程：

```text
题目
  ↓
采样 N 个代码解
  ↓
采样 M 个测试用例
  ↓
执行 N × M 的交叉矩阵
  ↓
统计“谁过了哪些测试”
  ↓
找出通过模式最稳定、成员最多的功能群组
  ↓
从最高分群组里选一个解
```

这和“谁写得最像参考答案”不是一回事。CodeT 关心的是**行为**，不是表面文本。

---

## 核心机制与推导

先定义几个符号。设候选代码集合为 $X=\{x_1,\dots,x_N\}$，测试集合为 $Y=\{y_1,\dots,y_M\}$。对任一候选解 $x$，记它通过的测试集合为

$$
T(x)=\{y\in Y \mid x \text{ passes } y\}
$$

这里的 “passes” 就是“代码在这个测试输入上跑通，且输出等于期望输出”。白话说，就是这份程序把这道验收题做对了。

CodeT 的关键假设来自 RANSAC。RANSAC 是一种“从有噪声的数据里找最大共识簇”的经典算法。迁移到代码生成里，假设是：

1. 错误程序往往很多样，彼此不容易在大量测试上恰好表现一致。
2. 正确或接近正确的程序，虽然实现细节不同，但会在同一批测试上给出一致行为。

因此，CodeT 不只看“某个解过了多少题”，还看“有多少别的解和它属于同一功能簇”。

在论文的朴素实现里，可以把具有相同通过集合的程序归成一组。若某组程序集合为 $S$，它们共同通过的测试集合为 $T$，则该共识集分数可写为

$$
f(S, T)=|S|\times|T|
$$

含义很直接：

- $|T|$ 越大，说明这组程序通过的测试越多。
- $|S|$ 越大，说明这种功能模式被更多独立采样复现。
- 两者相乘，等于“支持这组功能假设的解-测对数量”。

这就是 Dual Execution Agreement 的直觉来源。

看一个玩具例子。假设有 3 个程序 `a、b、c`，以及 3 个测试 `t1、t2、t3`：

| 解\测 | t1 | t2 | t3 |
|---|---|---|---|
| a | 1 | 1 | 0 |
| b | 1 | 1 | 0 |
| c | 1 | 0 | 0 |

其中 `1` 表示通过，`0` 表示失败。

- `a` 和 `b` 的通过集合都是 `{t1, t2}`，所以它们构成同一个功能簇，分数是 $2\times 2=4$。
- `c` 的通过集合是 `{t1}`，单独成组，分数是 $1\times 1=1$。

于是系统会优先选 `a/b` 这一组，而不是只因为 `c` 也过了一个测试就把它看成同等可信。

这也是它与 self-consistency 的根本区别。经典 self-consistency 是“让模型走多条推理链，再对最终答案做多数投票”。它验证的是**推理叙述的一致性**。CodeT 验证的是**可执行行为的一致性**。在代码任务里，两个程序可以写法完全不同，但只要在关键输入上表现一致，它们就属于同一个功能群；反过来，两个推理解释写得很像，程序也可能在边界条件上完全不同。

真实工程例子更容易看出差异。比如自动修复一个 PR 里的 bug，模型给出 20 个补丁。单看自然语言解释，很多补丁都说“修复了空指针问题”；但真正把补丁编译后跑在 50 个自动生成回归测试上，可能只有 4 个补丁稳定通过，其中 3 个补丁又恰好在同一组测试上行为一致。此时 CodeT 会把这 3 个补丁推到前面，因为它们不是“说得像对”，而是“跑出来更像对”。

---

## 代码实现

下面用一个最小可运行版本模拟 CodeT 的朴素打分。这里不真的执行任意 Python 代码，而是直接给出“解 × 测试”的通过矩阵，用它演示如何分组和计算共识分数。

```python
from collections import defaultdict

def codet_rank(pass_matrix):
    """
    pass_matrix: dict[str, dict[str, bool]]
    返回按 CodeT 朴素分数排序后的功能簇
    """
    # 1) 每个解映射到它通过的测试集合
    passed_tests_by_solution = {}
    for solution, test_results in pass_matrix.items():
        passed = tuple(sorted(t for t, ok in test_results.items() if ok))
        passed_tests_by_solution[solution] = passed

    # 2) 按“通过集合”分组，这些解被视为同一功能簇
    groups = defaultdict(list)
    for solution, passed in passed_tests_by_solution.items():
        groups[passed].append(solution)

    # 3) 计算分数 = 解数 × 测试数
    ranked = []
    for passed_tests, solutions in groups.items():
        score = len(solutions) * len(passed_tests)
        ranked.append({
            "solutions": tuple(sorted(solutions)),
            "tests": passed_tests,
            "score": score,
        })

    ranked.sort(key=lambda x: (-x["score"], -len(x["solutions"]), -len(x["tests"])))
    return ranked


matrix = {
    "a": {"t1": True,  "t2": True,  "t3": False},
    "b": {"t1": True,  "t2": True,  "t3": False},
    "c": {"t1": True,  "t2": False, "t3": False},
    "d": {"t1": False, "t2": False, "t3": True},
}

ranked = codet_rank(matrix)

assert ranked[0]["solutions"] == ("a", "b")
assert ranked[0]["tests"] == ("t1", "t2")
assert ranked[0]["score"] == 4

# 单独簇的分数较低
assert any(item["solutions"] == ("c",) and item["score"] == 1 for item in ranked)
assert any(item["solutions"] == ("d",) and item["score"] == 1 for item in ranked)
```

如果把它改成真实系统，流程通常是：

1. 采样 $N$ 个代码解。
2. 采样 $M$ 个测试。
3. 执行全部 $N\times M$ 对。
4. 记录每个解的通过向量。
5. 按通过向量聚类。
6. 用 $|S|\times|T|$ 或其变体排序。
7. 从最高分簇中取一个候选作为最终输出。

简化伪代码如下：

```python
solutions = sample_solutions(problem, n=N)
tests = sample_tests(problem, m=M)

matrix = {}
for s in solutions:
    matrix[s] = {}
    for t in tests:
        matrix[s][t] = run_and_check(s, t)

groups = group_by_pass_pattern(matrix)

best_group = None
best_score = -1
for group in groups:
    score = len(group.solutions) * len(group.passed_tests)
    if score > best_score:
        best_group = group
        best_score = score

answer = pick_one(best_group.solutions)
```

真实工程例子里，一般还会加两层缓存：

| 缓存项 | 作用 |
|---|---|
| 执行结果缓存 | 相同代码和相同测试不重复运行 |
| 输出哈希缓存 | 避免重复比较大对象或长文本输出 |

如果高温采样产生大量重复解或重复测试，可以先做去重，再执行矩阵。否则你统计到的“共识”里会混入大量重复样本，放大虚高分数。

---

## 工程权衡与常见坑

CodeT 的收益来自“测试 + 共识”双信号，但它的风险也主要来自测试。

最典型的问题是**错误测试**。错误测试就是“期望输出写错了”的测试。白话讲，就是判卷标准本身错了。只要这种测试混进来，错误程序就可能被奖励，正确程序反而被惩罚。

另一个问题是**测试毒性**。这里的毒性可以理解成“测试把排序往错误方向推”的能力。比如一批过于简单的测试只能覆盖主路径，很多错误解都会通过，于是系统形成了一个很大的假共识簇。

这正是后续工作反复强调的点。HARDTESTGEN 直接指出，现有公开数据里，APPS 上通过测试的程序中有相当高比例实际上仍然错误；CodeContests 上也有很多程序语义对但复杂度不达标。这说明“能过一批合成测试”不等于“真实正确”。

| 问题 | 现象 | 影响 | 对策 |
|---|---|---|---|
| 错误测试 | 期望输出写错 | 正确解被误杀 | 用参考解或多 oracle 过滤 |
| 重复测试 | 输入模式高度相似 | 共识被虚假放大 | 去重、聚类、提高采样多样性 |
| 覆盖不足 | 只测主路径 | 边界 bug 留存 | 加边界提示词、加入 adversarial tests |
| 只测功能不测性能 | 小样例能过，大样例超时 | 选中低效算法 | 补充规模测试和复杂度压力测试 |
| 沙箱不严 | 代码读写环境或死循环 | 执行风险高 | 超时、资源隔离、系统调用限制 |

可以用“认证考试题过于简单”来理解：如果题库全是送分题，很多人都能高分，分数就不再能代表真实能力。CodeT 只是更会用题库，不会自动把坏题库变成好题库。

工程上还有两个常见误区：

1. 误把 CodeT 当成多数投票。它不是“哪类代码文本最多就赢”，而是“哪类执行行为支持最多就赢”。
2. 误把高温采样当成越高越好。温度高能带来多样性，但也会带来大量无效样本、重复测试和执行成本，最后不一定更优。

---

## 替代方案与适用边界

最接近的替代方案是 self-consistency。它的优点是简单，不需要执行环境；缺点是没有真正验证程序行为。在数学问答里，多条推理链投票常常有效；但在代码任务里，两个“解释看起来一样”的程序，可能一个漏了边界条件，一个没有。

MBR-EXEC 也是相关方法。MBR 是 minimum Bayes risk，白话就是“选一个与其他候选平均最不冲突的答案”。当它基于执行结果计算风险时，会比纯文本方法更强，但它的目标仍是“与其他样本整体更接近”，而不一定像 CodeT 那样显式寻找“解集 × 测试集”的共识簇。

| 方法 | 输入 | 核心验证信号 | 优势 | 局限 |
|---|---|---|---|---|
| Self-consistency | 多条推理链/多个答案 | 最终答案投票 | 简单，成本低 | 不执行代码 |
| MBR-EXEC | 多个候选解 + 执行结果 | 与其他候选的平均一致性 | 比文本投票更贴近行为 | 不一定显式利用测试簇结构 |
| CodeT | 候选解 + 生成测试 | 通过测试数 + 功能共识簇 | 对代码任务更贴近真实正确性 | 强依赖测试质量 |
| Parsel + CodeT | 中间语言 + 多实现 + 生成测试 | 结构约束 + 双执行筛选 | 上限更高 | 流程更复杂，成本更高 |

适用边界可以一句话概括：

- 如果没有可执行环境，CodeT 很难落地。
- 如果题目天然可执行、可自动判定输出，CodeT 很有价值。
- 如果任务复杂到“生成测试比生成代码还贵”，可以先用结构化中间表示、规范约束或轻量投票做第一轮筛选，再对前几名用 CodeT 精排。

对新手最实用的判断标准是：你到底想验证“模型会不会说”，还是“程序能不能跑”。前者可以用 self-consistency，后者更适合 CodeT。

---

## 参考资料

- Bei Chen et al.，*CodeT: Code Generation with Generated Tests*，ICLR 2023。论文给出 Dual Execution Agreement、$|S|\times|T|$ 共识打分，以及 `code-davinci-002` 在 HumanEval 上从 47.0% 到 65.8% 的结果。[https://openreview.net/pdf?id=ktrw68Cmu9c](https://openreview.net/pdf?id=ktrw68Cmu9c)
- Maxwell Nye et al.，*Show Your Work: Scratchpads for Intermediate Computation with Language Models* 不直接相关；本题更相关的是 Parsel 论文：*Parsel: Algorithmic Reasoning with Language Models by Composing Decompositions*，NeurIPS 2023，给出 GPT-4 + Parsel + CodeT score 在 HumanEval 上 85.1% 的结果。[https://proceedings.neurips.cc/paper_files/paper/2023/file/6445dd88ebb9a6a3afa0b126ad87fe41-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/6445dd88ebb9a6a3afa0b126ad87fe41-Paper-Conference.pdf)
- Zhenyu Wang et al.，*Multi-Perspective Self-Consistency Improves Code Generation*，ACL 2024。文中将 Self-consistency、MBR-EXEC、CodeT 作为对照，适合看这些后验重排方法的差别。[https://aclanthology.org/2024.acl-long.78.pdf](https://aclanthology.org/2024.acl-long.78.pdf)
- Zhongmou He et al.，*HARDTESTGEN: A High-Quality RL Verifier Generation Pipeline for LLM Algorithmic Coding*，ICLR 2026。论文强调合成测试质量不足会带来高假阳性，是理解 CodeT 工程边界的重要补充。[https://openreview.net/pdf/f564b1a7536d9ae6db501b3dd71d2312678bd462.pdf](https://openreview.net/pdf/f564b1a7536d9ae6db501b3dd71d2312678bd462.pdf)
