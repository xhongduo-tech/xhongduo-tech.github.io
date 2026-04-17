## 核心结论

BFCL，全称 Berkeley Function-Calling Leaderboard，是 Berkeley 团队推出的函数调用评测体系，目标不是只看“模型会不会输出一个函数名”，而是统一衡量模型在真实工具调用链路中的完整能力：是否选对工具、是否按 schema 填对参数、是否在不该调用时正确 abstain，以及是否能在多轮和多工具场景中把任务做完。[OpenReview](https://openreview.net/forum?id=2GmDdhBdDk&noteId=2a2WPB0QnE)、[官方博客](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)

对工程团队来说，BFCL最有价值的地方在于它把“函数调用成功”拆成了几个可量化环节。工具调用不是一句“能用 function calling”就结束，真正决定效果的是三层能力：语法是否对齐、参数是否准确、执行或 abstain 是否正确。语法对齐可以理解为“格式合法”；参数精度可以理解为“值填得对”；执行正确则表示“真实跑起来结果没错”。

| 维度 | 指标 | 说明 |
| --- | --- | --- |
| 语法对齐 | AST 匹配 | 是否按函数定义构造出合法调用结构 |
| 参数精度 | Parameter match | 每个字段是否填对、类型是否匹配 |
| 执行结果 | Execution / abstain | 该调用时调用，不该调用时不乱调 |

一个最小但很典型的例子来自 `exec_simple_0`。用户问的是“单次掷出六点的概率是 60%，连续掷 20 次，恰好 5 个六点的概率是多少？”如果给定工具 `calc_binomial_probability`，正确调用必须是 `calc_binomial_probability(n=20, k=5, p=0.6)`。这里 `n` 是试验次数，白话说就是一共掷多少次；`k` 是成功次数，也就是要几个六点；`p` 是单次成功概率。如果模型把 `p` 写成 `1/6`、把 `k` 写成 `6`、或者把参数名拼错，`exact_match` 都会失败。[Inspect API 示例](https://ukgovernmentbeis.github.io/inspect_evals/evals/assistants/bfcl/)

结论可以压缩成一句话：BFCL评估的不是“模型能不能像样地输出 JSON”，而是“模型能不能把自然语言需求稳定翻译成可执行、可验证、可恢复状态的工具调用”。

---

## 问题定义与边界

BFCL可以形式化为一个三元组 $(q, T, g)$。

其中，$q$ 是用户输入的问题；$T=\{t_1,\dots,t_N\}$ 是可用工具集合；$g$ 是标准答案，通常是一个兼容 AST 的目标调用序列，也可能是空调用。AST，抽象语法树，可以理解为“把代码调用结构化之后得到的树状表示”，这样评测器可以比较结构而不是只比较字符串。[EmergentMind](https://www.emergentmind.com/topics/berkeley-function-calling-benchmark-bfcl)

每个工具 $t_i$ 往往用 JSON schema 描述，至少包含：

$$
t_i = (\text{name}, \text{description}, \text{properties}, \text{required})
$$

这里的 schema 很关键。schema 可以理解为“接口合同”，它规定了函数名、参数名、类型、是否必填。BFCL的很多分数，本质上都建立在“模型是否遵守这份合同”之上。

先看一个玩具例子。假设只有一个函数：

```json
{
  "name": "calc_binomial_probability",
  "description": "Calculates the probability of getting k successes in n trials.",
  "parameters": {
    "type": "dict",
    "properties": {
      "n": {"type": "integer"},
      "k": {"type": "integer"},
      "p": {"type": "float"}
    },
    "required": ["n", "k", "p"]
  }
}
```

用户说：“帮我估算 20 次掷骰子正好 5 个六点。”  
如果系统上下文已经说明“六点概率是 0.6”，那么模型应当输出调用；如果没有给出概率，严格来说就缺参了，好的系统应当追问，而不是瞎填 `1/6`。这就是 BFCL 的一个重要边界：它不仅评估“会不会调”，还评估“什么时候该问清楚，什么时候该 abstain”。

因此，BFCL覆盖的边界至少有四层：

| 边界 | 含义 | 评测关注点 |
| --- | --- | --- |
| 执行 vs 非执行 | 是否真的跑工具 | 输出结构对不对、结果对不对 |
| 单函数 vs 多函数 | 一个工具还是多个工具 | 选择精度、排序与组合 |
| 单轮 vs 多轮 | 一次问答还是连续上下文 | 状态保持、缺参澄清 |
| relevance vs irrelevance | 该不该调用 | 避免过度调用 |

这里有个容易被忽略的点：空调用也是答案。用户如果只是问“帮我解释一下二项分布是什么”，那正确行为可能是直接自然语言回答，而不是硬凑一个函数调用。BFCL把这种能力单独拉出来做 relevance 或 irrelevance 检查，就是为了防止模型“见工具就想调”。

---

## 核心机制与推导

BFCL之所以比“单条 tool call demo”更严格，是因为它把工具调用拆成了一条完整判定链：

`用户意图识别 → 工具选择 → 参数抽取 → 结构校验 → 执行验证或 abstain`

这条链任何一环错，最终都可能失败。

从类别上看，BFCL早期覆盖 simple、multiple、parallel、parallel-multiple、relevance、REST、多语言等任务；后续版本继续扩展到多轮、长上下文和 agentic 场景。agentic 可以理解为“像代理一样连续调度工具完成任务”的能力。V4 又把 Web Search、Memory、Format Sensitivity 单独拉出来，测试模型能否在多轮环境中恢复状态、读写记忆、并适应不同工具格式。[Hugging Face 数据集卡](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)、[BFCL V4 Web Search](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)、[BFCL V4 Memory](https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html)、[BFCL V4 Format](https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html)

| 类别 | 挑战 | 说明 |
| --- | --- | --- |
| Simple | 单调用 | 只需选择一个函数并填参 |
| Parallel | 同步多调用 | 一次输出多个彼此独立的调用 |
| Multiple / Parallel-Multiple | 选函数再组合 | 多个候选函数中选对并可能重复调用 |
| Multi-turn / Nested | 链式依赖 | 前一步结果影响后一步输入 |
| Relevance / Irrelevance | Abstain | 识别不需要调用工具 |

为什么 BFCL会强调 AST 校验？因为只做字符串比较太脆弱。比如参数顺序不同、空格不同，不该影响正确性；但函数名错、字段名错、字段值错，就必须判错。AST 的作用就是把“表面格式”与“真实结构”分开。

一个简化推导如下。假设总共有 $N$ 个样本，第 $i$ 个样本预测结果为 $\hat{y}_i$，标准答案为 $y_i$。则 AST 准确率可以写成：

$$
\mathrm{ASTAcc}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\mathrm{AST}(\hat{y}_i)=\mathrm{AST}(y_i))
$$

执行准确率则更严格：

$$
\mathrm{ExecAcc}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\mathrm{exec}(\hat{y}_i)=\mathrm{exec}(y_i))
$$

这里的 $\mathbf{1}$ 是指示函数，白话说就是“对记 1，错记 0”。

这意味着一件重要的工程事实：AST 对了，不代表执行一定对。比如函数名对、参数类型也对，但把日期填成明天而不是今天，AST 可能仍然合法，业务结果却错了。反过来，执行对了也未必说明模型鲁棒，因为它可能碰巧用错参数却在某些函数上得到相同输出。

真实工程例子可以看 BFCL V4 的 Web Search + Memory。比如用户先问“帮我查一下某家公司 2024 年增长最快的业务线”，后面再追问“把刚才那个结论记住，并生成一段结构化摘要”。模型至少要做到三件事：

1. 先调搜索工具拿外部信息。
2. 再把关键结果写入 memory。
3. 在后续回合读取 memory，并按格式工具输出结果。

如果第二步写入的 key 设计错，或者第三步忘了前面对话上下文，最终就不是“单次函数调用失败”，而是“多工具编排一致性失败”。这正是 BFCL 新版本要测的东西。

---

## 代码实现

如果你想自己感受 BFCL，最直接的方法是跑一个最小 split。Inspect API 文档给出的示例命令是：

```bash
inspect eval inspect_evals/bfcl \
  --model openai/gpt-4o-mini \
  --category exec_simple \
  --shuffle false
```

这个命令适合做 smoke test，也就是“先快速确认系统链路能跑通”。[Inspect API](https://ukgovernmentbeis.github.io/inspect_evals/evals/assistants/bfcl/)

下面用一个可运行的 Python 小脚本，模拟 BFCL 里最核心的一步：检查模型输出是否与目标调用完全一致，并顺手验证 `exec_simple_0` 的二项分布结果。二项分布，白话说就是“重复做很多次相同试验，恰好成功若干次的概率模型”。

```python
from math import comb

def calc_binomial_probability(n: int, k: int, p: float) -> float:
    assert isinstance(n, int) and n >= 0
    assert isinstance(k, int) and 0 <= k <= n
    assert isinstance(p, float) and 0.0 <= p <= 1.0
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def exact_match(pred: str, gold: str) -> bool:
    return pred.strip() == gold.strip()

gold = "calc_binomial_probability(n=20, k=5, p=0.6)"
pred_ok = "calc_binomial_probability(n=20, k=5, p=0.6)"
pred_bad = "calc_binomial_probability(n=20, k=5, p=0.1667)"

prob = calc_binomial_probability(n=20, k=5, p=0.6)

assert exact_match(pred_ok, gold) is True
assert exact_match(pred_bad, gold) is False
assert 0.0 <= prob <= 1.0
assert round(prob, 6) == round(comb(20, 5) * (0.6 ** 5) * (0.4 ** 15), 6)

print(prob)
```

如果你把这个思路扩展到完整评测器，通常会有三层校验：

| 层级 | 输入 | 输出 | 失败条件 |
| --- | --- | --- | --- |
| Schema 校验 | 函数定义 + 模型参数 | 合法 JSON / 调用对象 | 缺少 required、类型不匹配 |
| AST 校验 | 预测调用 + 标准答案 | 结构是否一致 | 函数名、字段名、值不一致 |
| 执行校验 | 调用对象 + 执行环境 | 业务结果 | 运行时报错、结果不一致 |

在 NeMo Evaluator 里，也可以直接跑 BFCL 的多个版本，文档明确列出了 `bfclv3`、`bfclv3_ast`、`bfclv2`、`bfclv2_ast` 等类型，并支持通过并行度配置控制吞吐。[NVIDIA NeMo](https://docs.nvidia.com/nemo/microservices/latest/evaluate/evaluation-types/bfcl.html)

一个工程上更接近真实系统的伪流程是：

$$
\text{调用前验证 schema} \rightarrow \text{生成 call} \rightarrow \text{AST match} \rightarrow \text{执行或 abstain}
$$

这里“调用前验证 schema”不能省。很多线上事故不是模型不会调用，而是应用方自己给错了 schema，导致模型再强也只能在错误合同上工作。

---

## 工程权衡与常见坑

第一个坑是把“模型能力问题”和“接口定义问题”混在一起。很多团队看到 BFCL 分数不高，第一反应是换模型；但真实原因往往是 schema 设计混乱，比如字段名含义不清、`required` 漏写、描述文本和参数名冲突。对 BFCL 来说，schema 是评测输入的一部分，不是背景噪声。

第二个坑是把 AST 正确当成业务正确。比如货币单位、时区、日期边界、分页 cursor，这些都可能在结构合法的前提下让结果错掉。也就是说，AST 更像“结构层验收”，不是“业务层验收”。

第三个坑是多轮状态漂移。状态漂移，白话说就是“模型到后面忘了前面说过什么”。在 BFCL V3/V4 里，这个问题会被放大。上一轮已经搜索过，下一轮却重复搜索；上一轮已经写入 memory，这一轮却找不到；或者工具列表在中间回合没有重新声明，导致模型调用了一个当前上下文并不存在的工具。这类错误在线上 agent 系统里非常常见。[BFCL V4 官方博客](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)

第四个坑是并行调用的一致性。并行不是“多输出几个调用”这么简单，而是多个调用之间不能互相污染。比如用户说“帮我查北京、上海、广州今天的天气”，如果模型输出了三个 `get_weather` 调用，但把三个城市都填成北京，形式上是并行，实质上是参数复制错误。

第五个坑是 abstain 做不好。很多模型过度积极，凡是看到工具就调用；也有些模型过度保守，该调用时却直接给口头答案。BFCL把 relevance 和 irrelevance 单独拿出来，就是因为线上错误里这两类都很多。

工程上更稳的做法通常是：

1. 先把 schema 设计成“少而清晰”的必填参数。
2. 对缺参场景单独定义追问策略。
3. 每轮调用前后都记录结构化状态，而不是只保留自然语言历史。
4. 对每一步单独打分：选型错、填参错、执行错、abstain 错分别统计。

这样做的好处是，一旦线上失败，你能知道是“工具选错了”还是“工具选对了但值填错了”，而不是只看到一个模糊的失败率。

---

## 替代方案与适用边界

BFCL很强，但它不是所有评测需求的唯一答案。

如果你的目标是评估“基础函数调用是否合格”，那么 v1/v2 风格的 simple、multiple、parallel 已经足够，因为它们重点考查单轮结构化调用、工具选择和参数对齐。

如果你的目标是“多轮工具链是否稳定”，那么应该优先看 v3，因为它更强调多轮、缺函数、缺参数、长上下文等问题。也就是说，你不该用 `exec_simple` 去判断一个 agent 系统的长期状态管理能力。

如果你的目标是“多工具代理能否在真实任务里持续工作”，那么 v4 更合适，因为它引入了 Web Search、Memory、Format Sensitivity 等 agentic 子任务。这些任务更接近真实应用里的“搜索一下再记住，然后按指定格式输出”。[BFCL V4 博客](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)

| 版本 / 形态 | 适用场景 | 特性 |
| --- | --- | --- |
| BFCL v1 / v2 | 单轮、基础工具调用 | AST、Exec、多函数、多语言 |
| BFCL v3 | 多轮调用、链式任务 | 缺参澄清、上下文追踪、长对话 |
| BFCL v4 | Agentic 多工具 | 搜索、记忆、格式敏感、状态恢复 |
| Inspect / NeMo 自定义接入 | 企业内部 API | 可替换数据、复用评测流程 |

还要看到 BFCL 的边界。它非常擅长判断“工具调用结构对不对”，但对一些更高层的业务目标，比如用户体验、策略性追问质量、长周期任务完成质量，往往还需要补充更贴近产品的任务集。换句话说，BFCL更像函数调用层的标准化体检，不是整个 agent 产品的全部验收标准。

如果你的内部工具很多、schema 很特殊、甚至根本不公开，那更实际的做法通常不是放弃 BFCL，而是沿用 BFCL 的评测思想：保留 $(q, T, g)$ 结构、保留 AST/Exec/abstain 三层评分，再把数据换成你的内部 API。NeMo 这类评测器就支持这种接法。[NVIDIA NeMo](https://docs.nvidia.com/nemo/microservices/latest/evaluate/evaluation-types/bfcl.html)

---

## 参考资料

- OpenReview: The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models  
  https://openreview.net/forum?id=2GmDdhBdDk&noteId=2a2WPB0QnE
- BFCL 数据集卡与分类说明  
  https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
- Inspect API: BFCL 评测与 `exec_simple_0` 示例  
  https://ukgovernmentbeis.github.io/inspect_evals/evals/assistants/bfcl/
- BFCL V4 Web Search  
  https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html
- BFCL V4 Memory  
  https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html
- BFCL V4 Format Sensitivity  
  https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html
- NVIDIA NeMo: BFCL Evaluation Type  
  https://docs.nvidia.com/nemo/microservices/latest/evaluate/evaluation-types/bfcl.html
- EmergentMind: Berkeley Function-Calling Benchmark  
  https://www.emergentmind.com/topics/berkeley-function-calling-benchmark-bfcl
