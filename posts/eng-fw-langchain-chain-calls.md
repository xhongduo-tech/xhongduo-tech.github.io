## 核心结论

LangChain 的链式调用，本质上是把多个可复用步骤按顺序拼成一条流水线。这里的“链”可以先理解成一个输入进来、输出出去的处理盒子。盒子里可以是大模型调用、检索器、工具执行、格式化器，甚至另一个链。

它成立的关键，不是“会调模型”，而是“每一步都遵守统一接口”。经典 `Chain` 接口以字典作为输入输出，现代 LangChain 更常用 `RunnableSequence` 和 `|` 管道语法，但核心思想没变：前一步的结果要能稳定地喂给下一步。用数学写就是：

$$
y = f_n(f_{n-1}(\cdots f_1(x)\cdots))
$$

这带来三个直接收益：

| 价值 | 白话解释 | 为什么对工程有用 |
| --- | --- | --- |
| 可组合 | 小步骤能像积木一样拼接 | 需求变化时改一段，不必重写整条流程 |
| 可观测 | 每一步都能挂日志、回调、trace | 出错时知道是哪一段坏了 |
| 可治理 | 可以插入 memory、middleware、权限控制 | 适合真实生产系统，而不只是 demo |

对初学者最重要的一点是：链式调用不是为了“显得高级”，而是为了把一个大问题拆成多个职责清晰的小问题。复杂任务不要试图塞进一个超长 Prompt。那样通常更贵、更难调、也更难复现。

---

## 问题定义与边界

“链式调用”解决的问题，是多步骤任务如何稳定执行。这里的“稳定”不是回答永远正确，而是系统结构可预测：第一步做什么、第二步吃什么输入、第三步如何记录状态，都能提前说明。

玩具例子最容易看懂。假设你做一个旅行规划器：

1. 第一个链根据目的地生成活动建议。
2. 第二个链根据“活动建议”和用户约束生成日程。

这时边界很清楚：第二个链只读 `activities`、`days`、`budget` 这些必要字段，而不是把所有原始对话历史都塞进去。这样做的目的，是避免上下文失控。上下文就是模型本轮能看到的信息，可以先理解成“本次考试允许翻阅的材料”。

| 链名 | 输入键 | 输出键 | 主要关注点 |
| --- | --- | --- | --- |
| 活动建议链 | `destination` | `activities` | 只负责发散候选活动 |
| 行程生成链 | `activities`, `days`, `budget` | `itinerary` | 只负责组合成计划 |
| 审查链 | `itinerary` | `review` | 只负责检查冲突和遗漏 |

边界一旦不清，常见问题就来了：

| 失控方式 | 结果 |
| --- | --- |
| 每一步都传全量聊天记录 | token 暴涨，延迟升高 |
| 输出字段命名混乱 | 后续链拿不到正确值 |
| 把检索、推理、格式化混在一条 Prompt 里 | 调试困难，责任不清 |
| 工具权限不显式传递 | 模型知道要做什么，却不能做 |

所以链式调用的边界不是“能不能连起来”，而是“每一步是否只处理自己该处理的信息”。

---

## 核心机制与推导

从机制上看，链式调用就是函数复合。每个链段 $f_i$ 都接收一个结构化输入，并返回一个结构化输出。结构化的意思是：不是随便拼文本，而是用明确键名表达语义，比如 `question`、`documents`、`summary`、`answer`。

经典 `Chain` 体系里，`invoke` 负责执行，`prep_inputs` 会在执行前补齐 memory 等输入，`prep_outputs` 会在执行后校验输出并决定是否把输入一起带回。白话讲，就是“开工前把工具箱准备好，完工后把结果收拾整齐”。

玩具例子：公司名生成链和口号生成链。

- `f_1({"product": "环保水瓶"}) -> {"company_name": "AquaLeaf"}`
- `f_2({"company_name": "AquaLeaf"}) -> {"slogan": "更少塑料，更久陪伴"}`

组合后就是：

$$
f(x)=f_2(f_1(x))
$$

箭头可以写成：

`product` → `company_name` → `slogan`

真实工程里只是把盒子换大一些。例如一个 RAG 流水线。RAG 是“检索增强生成”，白话讲就是先找资料，再让模型基于资料回答，而不是只靠模型记忆。

| 阶段 | 职责 | 输入 | 输出 |
| --- | --- | --- | --- |
| 检索链 | 从知识库找相关文档 | `question` | `documents` |
| 摘要链 | 压缩文档，减少噪声 | `documents` | `summary` |
| 回答链 | 基于摘要回答并给出引用 | `question`, `summary` | `answer` |

这里还有一个常被忽略的推导：链式调用不一定更省 token，但它让 token 成本可控。设检索后文档长度为 $D$，摘要长度为 $S$，最终问题长度为 $Q$，那么“摘要后再回答”的主要输入规模近似是：

$$
Cost \approx D + (S + Q)
$$

如果直接把全部文档和问题扔进最终回答链，则近似是：

$$
Cost \approx D + Q
$$

看起来差别不一定大，因为摘要本身也花 token。但工程上的收益在于：你把“长上下文理解”和“最终回答生成”拆开了，于是可以分别优化、缓存、监控和替换模型。

---

## 代码实现

先给一个可运行的 Python 玩具实现，不依赖 LangChain，也能看懂“链式数据流”是什么。重点是输出键必须与下游输入键对齐。

```python
from typing import Dict

def company_chain(inputs: Dict[str, str]) -> Dict[str, str]:
    product = inputs["product"]
    if "water" in product.lower():
        name = "AquaLeaf"
    else:
        name = "NovaCraft"
    return {"company_name": name}

def slogan_chain(inputs: Dict[str, str]) -> Dict[str, str]:
    company_name = inputs["company_name"]
    slogan = f"{company_name}: build less waste, ship more value."
    return {"slogan": slogan}

def sequential_invoke(inputs: Dict[str, str]) -> Dict[str, str]:
    step1 = company_chain(inputs)
    merged = {**inputs, **step1}
    step2 = slogan_chain(merged)
    return {**merged, **step2}

result = sequential_invoke({"product": "eco-friendly water bottles"})
assert result["company_name"] == "AquaLeaf"
assert "AquaLeaf" in result["slogan"]
print(result)
```

上面这段代码已经覆盖了链式调用最核心的规则：

1. 每一步只读自己需要的键。
2. 每一步只产出明确命名的键。
3. 中间结果可以合并，再传给后续步骤。

如果换成 LangChain，初学者经常在旧教程里看到 `LLMChain` 和 `SequentialChain`。这些写法能帮助理解“链”的概念，但从官方 API 演进看，现代写法更推荐 `RunnableSequence` 或 `prompt | model | parser`。

经典示意如下：

```python
from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate

company_prompt = PromptTemplate(
    input_variables=["product"],
    template="Generate a company name for a product that is {product}."
)

slogan_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Write a slogan for the company named {company_name}."
)

# 省略 llm 初始化
company_chain = LLMChain(llm=llm, prompt=company_prompt, output_key="company_name")
slogan_chain = LLMChain(llm=llm, prompt=slogan_prompt, output_key="slogan")

sequential_chain = SequentialChain(
    chains=[company_chain, slogan_chain],
    input_variables=["product"],
    output_variables=["company_name", "slogan"],
    verbose=True,
)
```

更接近当前 LangChain 方向的写法，是把步骤视为 `Runnable`，用 `|` 串联。这样天然支持同步、异步、批处理和流式输出。对真实系统，这比只会写一个大 Prompt 更接近生产可用形态。

真实工程例子可以是“检索 → 摘要 → 最终回答”：

1. 检索器返回文档列表。
2. 摘要器把长文档压到固定窗口。
3. 回答器只吃 `question + summary`。
4. 回调记录 token、延迟、错误率。

这时你不是在“调一个模型”，而是在“编排一条可观测流水线”。

---

## 工程权衡与常见坑

链式调用最大的工程价值，是把失败变成可定位问题；但它也会引入更多步骤、更多延迟和更多接口约束。

最常见的坑不是模型太笨，而是上下文工程做得不完整。上下文工程可以理解成“把模型完成任务所需的信息，以正确格式、在正确时机塞进去”。LangChain 官方把它分成 transient context 和 persistent context。前者是本次调用临时可见的信息，后者是能跨轮次保留下来的状态。

| 常见坑 | 影响 | 缓解策略 |
| --- | --- | --- |
| 漏传关键字段 | 下游链输出偏差或直接报错 | 为每步定义清晰输入键并做校验 |
| 输出键命名不一致 | 链接不上，调试困难 | 统一 schema，写最小断言测试 |
| 历史消息过长 | token 成本和延迟上升 | 摘要旧消息，只保留最近关键轮次 |
| 所有步骤都用同一个大模型 | 成本过高 | 检索/摘要用便宜模型，最终回答用强模型 |
| 链条过长 | 单点错误会层层传染 | 在关键步骤落日志、保存中间结果 |
| 路由条件写死过早 | 可维护性差 | 先用顺序链，需求稳定后再引入路由 |

一个典型真实策略是 `SummarizationMiddleware`。它的作用不是“让回答更聪明”，而是“当会话太长时，先压缩旧消息，再继续执行”。这属于生命周期中间件，即在模型调用前后插入横切逻辑。横切逻辑就是所有链都可能共用的规则，比如摘要、日志、审计、权限过滤。

实际效果是：

1. 达到 token 阈值。
2. 旧消息被单独总结成摘要。
3. 摘要写回状态，替代原始长消息。
4. 后续链继续运行，但上下文窗口更小。

这种设计的代价是：摘要可能丢细节。所以不能把它理解成“免费压缩”。它只是用可接受的信息损失，换取成本和稳定性。

---

## 替代方案与适用边界

顺序链适合线性任务，即步骤先后关系明确，且大部分请求都走同一条路径。比如“检索后回答”“生成后审查”“提取后入库”。这种场景优点是结构简单、易测、容易插监控。

但如果输入类型差异很大，线性链就会浪费资源。比如用户问题里有的要数学求解，有的要代码建议，有的只是普通问答。你不希望所有请求都经过同样的链。此时更合适的是路由链，现代 LangChain 更常见的实现是 `RunnableBranch` 或基于 `RunnableLambda` 的动态路由。

可以把它理解成程序里的 `if/else`：

| 链类型 | 是否线性 | 输入输出数量 | 何时选用 | 复杂度 |
| --- | --- | --- | --- | --- |
| 顺序链 | 是 | 通常固定 | 固定流程、多步加工 | 低 |
| 路由链 | 否 | 多分支 | 输入类型差异大、需按语义分流 | 中 |
| 图式工作流 | 否，且可回环 | 多节点多状态 | 有回退、重试、并行、人工审批 | 高 |

一个简单路由例子：

- 包含“微分、积分、方程”走数学链。
- 包含“Python、报错、接口”走编程链。
- 其他走通用问答链。

这时路由的收益在于“把请求送到最合适的专家链”，代价则是“要多一次判断”。如果这个判断还靠 LLM 做语义分类，就会增加延迟和 token 成本。所以经验规则是：

1. 能用顺序链解决，就先别上路由。
2. 能用显式规则路由，就先别上 LLM 路由。
3. 当流程出现回环、并行、人工审批、恢复点时，顺序链也不够了，应考虑更图式的编排工具。

也就是说，LangChain 的链式调用不是“越复杂越高级”，而是“在最小复杂度下，把流程拆到刚好可治理”。

---

## 参考资料

- LangChain `Chain` 基类与 `invoke`、`prep_inputs`、`prep_outputs`：<https://api.python.langchain.com/en/latest/langchain/chains/langchain.chains.base.Chain.html>
- LangChain Chains 总览：<https://api.python.langchain.com/en/latest/langchain/chains.html>
- LangChain `RunnableSequence`：<https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.base.RunnableSequence.html>
- LangChain `RunnableBranch`：<https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.branch.RunnableBranch.html>
- LangChain Context Engineering（Python）：<https://docs.langchain.com/oss/python/langchain/context-engineering>
- LangChain `LLMChain` 文档，含已弃用提示：<https://api.python.langchain.com/en/latest/langchain/chains/langchain.chains.llm.LLMChain.html>
- GeeksforGeeks，《Sequential Chains in LangChain》：<https://www.geeksforgeeks.org/artificial-intelligence/sequential-chains-in-langchain/>
- FastGPTPlus，《When to Use Router Chain in LangChain》：<https://fastgptplus.com/en/posts/langchain-router-chain-guide-2025>
- Blockchain Council，《LangChain in 2026: Building Reliable Agents and RAG Pipelines》：<https://www.blockchain-council.org/ai/langchain/>
