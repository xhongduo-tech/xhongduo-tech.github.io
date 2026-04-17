## 核心结论

把大模型的数学推理接到 Lean 4 上，关键收益不是“模型更会算题”，而是“每一步都能被证明器验收”。Lean 4 是一个形式化证明系统，白话说，它像一个只接受严格语法和严格逻辑的编译器。模型负责提出下一步证明动作，Lean 负责判断这一步是否合法、是否把目标推进、是否已经完成证明。这种组合把“会说理由”变成“能交付可验证对象”。

LeanDojo-v2 的价值在于把这条链路工程化：仓库追踪、证明状态提取、前提检索、tactic 生成、搜索执行、训练与部署都放进同一套工具中。对新手可以直接理解成：模型先看当前题目状态，再从题库里找可能有用的定理，猜下一步操作，Lean 说“通过”或“报错”，直到证明完成。

一个常见误解是“只要 LLM 足够强，就可以直接写出完整证明”。现实不是这样。当前效果最好的一类方案不是纯生成，而是“检索增强 + 搜索 + Lean 验证”。在旧版 LeanDojo / ReProver 路线里，模型并不是一次写完整个 proof，而是不断预测 tactic。tactic 是证明中的一步操作，白话说，类似“把当前题拆成子题”或“调用某个已知定理”。

另一个必须分清的点是 autoformalization。它不是证明本身，而是把自然语言数学命题翻译成 Lean 能接受的形式化语句。Poiroux 等人在 2024 年的结果表明，GPT-4o 做 Lean 4 autoformalization 时，失败主因不是“数学意义理解错”，而是“类型检查没过”；他们报告 86.6% 的错误起始于 type-check failure。类型检查可以理解成“变量、定理、结构之间的接口是否对得上”。

里程碑结果上，OpenAI 在 ICML 2022 的 expert iteration 工作里，full curriculum 模型在 miniF2F-valid 上做到 pass@8 = 41.2%。这里的 `pass@8` 指每道题给 8 次独立尝试，至少有一次成功的比例。这个数字说明“验证驱动的数据增强”很有效，但也同时暴露出瓶颈：模型仍然不擅长稳定串联多步非平凡推理，尤其是需要中间引理时。

| 结论 | 为什么重要 |
| --- | --- |
| LLM 负责生成，Lean 负责验收 | 把“会解释”变成“可审计” |
| ReProver 依赖检索前提 + proof state | 降低凭空 hallucination 的概率 |
| autoformalization 要先过类型检查 | 不合法的形式句子不能进入证明阶段 |
| 41.2% 说明路线可行但远未解决 | 长链推理和 lemma 生成仍是瓶颈 |

---

## 问题定义与边界

问题可以拆成两层。

第一层是 theorem proving，也就是“给定一个已经形式化的 Lean 命题，自动生成证明”。输入是 Lean 当前的 proof state，输出是下一步 tactic 或完整 proof。proof state 可以理解成“当前还没证明完的目标”和“此时可用的局部变量、假设”。

第二层是 autoformalization，也就是“把自然语言命题翻译成 Lean 语句”。输入是非形式化文本，输出是一个能被 Lean 解析并通过类型检查的 formal statement。只有这一步成功，后面才谈得上证明。

二者边界必须分开看：

| 模块 | 输入 | 输出 | Lean 的作用 |
| --- | --- | --- | --- |
| Autoformalization | 自然语言命题 | Lean 语句 | 解析 + 类型检查 |
| Tactic generation | proof state + 检索前提 | 下一步 tactic | 执行 tactic 并返回新状态 |
| Proof search | 多个候选 tactic | 完整 proof tree | 判断哪条路径能闭合全部目标 |

LeanDojo-v2 主要解决的是接口问题。LLM 不能直接“理解” Lean 运行时内部结构，所以需要额外层把仓库中的 theorem、AST、proof state、premise usage 提取出来，整理成训练和推理可消费的数据。AST 是抽象语法树，白话说，就是代码的树状结构表示，方便程序读取而不是给人读。

玩具例子可以先看一个很小的命题：证明“若 `a = b`，则 `a + 1 = b + 1`”。人类会说“等式两边同时加一”。Lean 环境里，这不是一句自然语言，而是一个可执行动作，比如 `simpa [h]` 或 `rw [h]`。模型必须生成合法 tactic，Lean 执行后才知道是不是推进了目标。

真实工程例子则是 mathlib4 仓库。LeanDojo-v2 会追踪仓库快照，提取 theorem 和 proof state，写入数据库，再训练 RetrievalTrainer 学“当前状态下哪些前提最相关”，最后由 LeanPG 或其他 prover 发起搜索。也就是说，系统不是只盯着一道题，而是在整个形式化数学代码库上建立“题目状态 -> 可用定理 -> 下一步动作”的数据闭环。

数学上，可以把 tactic 预测写成条件概率问题：
$$
p(t \mid s, P)=\mathrm{softmax}(f_\theta(s, P))
$$
其中 $s$ 是 proof state，$P$ 是检索到的前提集合，$t$ 是候选 tactic。直观上，模型根据“当前卡在哪”和“手边有哪些定理”，给每个下一步动作打分。

---

## 核心机制与推导

ReProver 的核心不是单独一个语言模型，而是“检索 + 生成 + 搜索”的组合。

第一步，系统从 mathlib 中检索相关前提。前提就是可能要调用的定理、定义或引理。第二步，把这些前提和当前 proof state 拼接后送入 encoder-decoder 模型，生成多个 tactic 候选。第三步，不是直接相信最高分候选，而是交给 best-first search。best-first search 是一种优先扩展最有希望节点的搜索策略，白话说，先试模型最看好的那几步。

可以把搜索节点 $n$ 的评分写成：
$$
\mathrm{score}(n)=\log p_\theta(t_1,\dots,t_k \mid s_0,P)-\lambda \cdot \mathrm{cost}(n)
$$
这里前一项表示这条路径在模型看来有多合理，后一项表示搜索代价，比如深度、展开节点数或未解决目标数量。$\lambda$ 越大，搜索越保守。

新手版理解是这样的：模型先报几个“可能有效”的步骤，搜索器把它们排优先级，Lean 真正执行，执行成功才保留，失败就剪枝。于是整个过程不是“模型写答案”，而是“模型提方案，证明器裁决”。

autoformalization 的机制不同。它的难点不是 proof search，而是句子合法性。Poiroux 等人的方法非常直接：对同一条自然语言命题采样多个 Lean 候选表达式，让 Lean 先做类型检查，删除非法句子，再对剩余候选做 self-consistency。self-consistency 可以理解成“多次独立作答后，选最稳定、最常出现的版本”。

若候选集合为 $C=\{c_1,\dots,c_m\}$，类型检查通过的集合为 $C_{\mathrm{ok}}$，则一个简单投票分数可写成：
$$
\mathrm{vote}(c)=\sum_{i=1}^{m}\mathbf{1}[c_i=c \land c_i \in C_{\mathrm{ok}}]
$$
最终选择
$$
c^\*=\arg\max_{c \in C_{\mathrm{ok}}}\mathrm{vote}(c)
$$

玩具例子：自然语言命题“任意自然数 $n$ 满足 $n+0=n$”。模型可能采样出 8 个 Lean 候选，其中 5 个因为变量类型、命名空间或量词写法错误而无法 type-check，剩下 3 个其实表达同一件事，只是语法略不同。这时先过滤，再投票，效果会明显好于只取第一条。

下面的表格说明这种机制的直觉：

| 采样数 | 类型检查前错误率 | 类型检查后可投票候选数 | 结果稳定性 |
| --- | --- | --- | --- |
| 1 | 高 | 0 或 1 | 很差 |
| 4 | 中高 | 1 到 2 | 一般 |
| 8 | 中 | 2 到 4 | 明显提升 |
| 16 | 更低但成本更高 | 3 到 8 | 提升趋缓 |

真实工程例子是 miniF2F。miniF2F 是从竞赛数学题中整理出的形式化基准，白话说，它更像“陌生考试题”，不像 mathlib 那样容易靠相似样本记忆。OpenAI 2022 的 expert iteration 结果里，miniF2F-valid 的 `pass@8` 从 PACT 的 29.3% 提高到 full curriculum 模型的 41.2%。这说明仅靠 proof search 不够，必须把“搜索得到的成功轨迹”反过来作为训练数据，形成数据增强闭环。

| 方法 | miniF2F-valid pass@1 | pass@8 | pass@64 |
| --- | --- | --- | --- |
| PACT (2021) | 23.9% | 29.3% | - |
| $\theta_1$ | 28.5% | 35.5% | 41.2% |
| $\theta^{mathlib}_9$ | 31.3% | 38.3% | 44.1% |
| $\theta^{full}_9$ | 33.6% | 41.2% | 47.3% |

---

## 代码实现

下面用一个极简 Python 示例模拟“候选 formal statement 过滤 + self-consistency”。它不是 LeanDojo 源码，但保留了核心接口思想：先判断候选是否合法，再在合法候选里投票。

```python
from collections import Counter

def pick_formalization(candidates, typecheck_ok):
    """
    candidates: 多次采样得到的候选 formal statement
    typecheck_ok: 一个函数，输入候选字符串，返回是否通过类型检查
    """
    legal = [c for c in candidates if typecheck_ok(c)]
    assert len(candidates) > 0
    if not legal:
        return None
    counts = Counter(legal)
    best, freq = counts.most_common(1)[0]
    assert freq >= 1
    return best

def fake_typecheck(stmt: str) -> bool:
    # 极简模拟：必须含有 theorem 和 :=
    return "theorem" in stmt and ":=" in stmt

samples = [
    "theorem add_zero_nat : forall n : Nat, n + 0 = n := by simp",
    "forall n : Nat, n + 0 = n",  # 非法，缺 theorem 和 :=
    "theorem add_zero_nat : forall n : Nat, n + 0 = n := by simp",
    "theorem bad_stmt : forall n : Nat, n + true = n := by simp",  # 这里只模拟通过，不做真正类型推理
]

best = pick_formalization(samples, fake_typecheck)
assert best == "theorem add_zero_nat : forall n : Nat, n + 0 = n := by simp"
print(best)
```

如果把 theorem proving 也抽象一下，推理循环大致如下：

```python
def best_first_prove(initial_state, retrieve, generate_tactics, run_tactic, max_steps=64):
    frontier = [(0.0, initial_state, [])]  # (负分数, 当前状态, 历史路径)
    visited = set()

    while frontier and max_steps > 0:
        frontier.sort(key=lambda x: x[0])
        neg_score, state, path = frontier.pop(0)
        max_steps -= 1

        if state["goal_closed"]:
            return path

        key = state["text"]
        if key in visited:
            continue
        visited.add(key)

        premises = retrieve(state)
        tactics = generate_tactics(state, premises)

        for tactic, logp in tactics:
            result = run_tactic(state, tactic)
            if result["ok"]:
                frontier.append((neg_score - logp, result["next_state"], path + [tactic]))

    return None
```

这段伪代码对应 LeanDojo / ReProver 的主流程：`retrieve` 像 RetrievalTrainer 产出的检索器，`generate_tactics` 像 ReProver，`run_tactic` 则由 Lean 环境执行并反馈。

再看训练侧，可以把 state 字段和训练目标对应起来：

| Lean state 字段 | 含义 | 训练用途 |
| --- | --- | --- |
| `goal` | 当前待证明目标 | tactic 生成主输入 |
| `locals` | 当前局部变量与假设 | 决定可用推理上下文 |
| `premises` | 检索出的相关定理 | 检索训练与 RAG 输入 |
| `next_tactic` | 人类或成功轨迹中的下一步 | 监督学习标签 |
| `result` | tactic 执行反馈 | 搜索剪枝与 RL 信号 |

真实工程例子里，LeanDojo-v2 的流水线可以概括成：

| 阶段 | 输入 | 产物 |
| --- | --- | --- |
| Repository tracing | Lean 仓库快照 | theorem、AST、proof state、premise 标注 |
| Dataset management | 多仓库追踪结果 | 动态数据库 |
| Retrieval training | state -> 相关前提 | dense retriever |
| Prover inference | state + 前提 | tactic 或 whole-proof |
| Lean execution | tactic | 新状态、错误、完成信号 |

---

## 工程权衡与常见坑

第一个坑是把“数学正确”误当成“Lean 可接受”。在形式化系统里，语义正确但类型不对，依然等于失败。Poiroux 等人的数据说明这个问题很严重：GPT-4o 做 Lean 4 autoformalization 时，86.6% 的错误从类型检查失败开始。工程上这意味着，采样 8 个候选再过滤，往往比单次“更聪明的提示词”更划算。

第二个坑是长链推理。LLM 对 1 到 2 步非平凡操作还可能稳定，但一旦需要 3 步以上，并且其中包含中间引理构造，成功率会明显下降。所谓 lemma，就是为了主证明而临时提出的中间命题，白话说，相当于“先证明一个小台阶，再踩这个台阶上去”。

可以把这件事写成一个粗糙的成功率模型：
$$
P(\text{full proof}) \approx \prod_{i=1}^{k} P(\text{step}_i \mid \text{history})
$$
如果每一步都不是非常稳，链条一长，总成功率会指数下降。这就是为什么 41.2% 不是一个“小优化还没做完”的问题，而是深层搜索与规划问题。

第三个坑是检索噪声。检索到太少前提，模型缺工具；检索到太多前提，输入变长、干扰增大、生成更不稳定。对初级工程师来说，一个实用经验是先把检索做好，再谈更大的生成模型。因为错误前提集会直接污染 tactic 预测条件分布。

第四个坑是把搜索预算当作免费资源。best-first search、beam search、自一致性采样都在花算力。一个典型权衡如下：

| 问题 | 便宜做法 | 更稳做法 | 代价 |
| --- | --- | --- | --- |
| autoformalization | 单样本直出 | 多样本 + type-check + voting | 延迟更高 |
| tactic 预测 | 贪心生成 | best-first / beam search | 节点数增加 |
| 长链证明 | 直接硬搜 | 显式生成 lemma / cut | 系统更复杂 |

真实工程中，过滤流程最好顺手统计错误类型，否则系统会反复在同一类失败上烧算力。下面是一个最小错误统计片段：

```python
from collections import Counter

def summarize_errors(results):
    counter = Counter()
    for r in results:
        if r["stage"] == "typecheck" and not r["ok"]:
            counter["type_error"] += 1
        elif r["stage"] == "tactic" and not r["ok"]:
            counter["tactic_error"] += 1
        elif r["stage"] == "search_timeout":
            counter["timeout"] += 1
        else:
            counter["success"] += 1
    assert sum(counter.values()) == len(results)
    return counter
```

如果想进一步缓解长链问题，可以把 lemma 规划看成两级决策：
$$
\max_{l_1,\dots,l_m} \; P(l_1,\dots,l_m \mid s)\cdot P(\text{prove target}\mid s,l_1,\dots,l_m)
$$
这里先选哪些中间引理值得提出，再分别证明它们。难点在于，提出“有用但不多余”的引理本身就是高难任务。

---

## 替代方案与适用边界

纯 proof search 仍然有价值。它不依赖通用 LLM 的语言能力，系统结构也更可控。如果问题深度小、可用 tactic 模式稳定、目标分布接近已有库，那么纯搜索很合适。它像“在明确规则下穷举高概率步骤”，适合局部封闭问题。

但纯 proof search 的上限也明显。面对 miniF2F 这类更开放、与库中已有证明距离更远的问题，仅靠穷举 tactic 调度通常不够。OpenAI 2022 的结果已经说明，在固定算力下，expert iteration 比 proof-search-only 更有效，因为它能把搜索发现的新证明回灌成训练数据。

另一类替代方案是端到端 autoformalization，也就是自然语言直接翻 Lean 语句，再继续让模型给 proof。它适合结构简单、模板强、已有大量对齐数据的任务，例如教材式定理、定义重述、简单代数恒等式。但一旦命题需要复杂依赖、隐式上下文或高级类型结构，类型失败会快速上升。

可以用一个简单的效率/完整性直觉表示：
$$
\text{utility} = \alpha \cdot \text{success rate} - \beta \cdot \text{compute cost} + \gamma \cdot \text{completeness}
$$
纯 proof search 往往在 completeness 上更好，因为它更系统；LLM + verification 往往在 success rate 上更高，因为它更会猜高价值步骤；端到端翻译则在简单场景里效率高，但对上下文敏感。

| 方案 | 优点 | 缺点 | 适用边界 |
| --- | --- | --- | --- |
| Pure proof search | 可控、可分析、对小深度问题稳 | 开放题上效率差 | tactic 空间较规整 |
| ReProver + Lean 验证 | 成功率高、可审计 | 依赖检索与搜索预算 | 中等复杂度 theorem proving |
| End-to-end autoformalization | 流程短 | 类型失败重 | 模板化命题翻译 |

基线 proof search 的典型 loop 可以写成这样：

```python
def baseline_search(state, candidate_tactics, run_tactic, depth_limit=5):
    agenda = [(state, [])]
    while agenda:
        cur, path = agenda.pop()
        if cur["goal_closed"]:
            return path
        if len(path) >= depth_limit:
            continue
        for tac in candidate_tactics(cur):
            nxt = run_tactic(cur, tac)
            if nxt["ok"]:
                agenda.append((nxt["next_state"], path + [tac]))
    return None
```

这段代码说明了它的适用边界：只要深度不大、候选分支不爆炸，就还能工作；一旦需要长链、引理构造或大规模前提选择，就会迅速失控。

---

## 参考资料

| 资料 | 年份 | 关键信息 |
| --- | --- | --- |
| LeanDojo-v2: A Comprehensive Library for AI-Assisted Theorem Proving in Lean | 2025 | Lean 4 端到端框架，覆盖仓库追踪、数据管理、检索训练、proof search、部署 |
| LeanDojo: Theorem Proving with Retrieval-Augmented Language Models | 2023 | 提出 ReProver，核心是“检索前提 + proof state -> tactic”，并与 best-first search 结合 |
| Improving Autoformalization using Type Checking | 2024 | 说明 autoformalization 主要瓶颈是类型检查；GPT-4o Lean 4 错误中 86.6% 起于 type-check failure；过滤 + self-consistency 把 ProofNet 提到 53.2% |
| Formal Mathematics Statement Curriculum Learning | 2022 | expert iteration 在 miniF2F-valid 上达到 pass@8 = 41.2%，说明“搜索产生数据，再训练模型”优于 proof-search-only |

1. LeanDojo 官方页面：https://leandojo.org/leandojo  
2. Poiroux 等，Improving Autoformalization using Type Checking：https://atcbosselut.github.io/publication/poiroux-2024-typechecking/  
3. OpenAI，Formal Mathematics Statement Curriculum Learning（ICML 2022 PDF）：https://cdn.openai.com/papers/Formal_Mathematics_Statement_Curriculum_Learning__ICML_2022.pdf
