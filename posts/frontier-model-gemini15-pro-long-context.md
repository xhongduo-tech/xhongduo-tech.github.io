## 核心结论

Gemini 1.5 Pro 的突破点不是“把输入框放大”，而是把“长上下文真的能被利用”这件事往前推进了一大步。这里的上下文窗口，指模型一次推理时能看到的全部输入范围；token 是模型处理文本、代码、音视频片段时使用的最小计数单位，不等于“字”或“词”。

2024 年 2 月 Google 公布 Gemini 1.5 Pro 时，生产可用上下文窗口达到 1,000,000 token，研究实验延伸到 10,000,000 token。按官方给出的等效量，1M token 大致可覆盖 70 万词、1 小时视频、11 小时音频，或一个 3 万行级别代码库。它的意义不是“能塞更多材料”，而是把原本需要“检索一次、问一次、再检索”的多轮流程，压缩成“一次把大语料放进去再推理”。

但长上下文不等于稳定。Gemini 1.5 Pro 在单针检索，也就是“在大文本里找一个确定目标”时，召回率几乎贴近满分；到了多针检索，也就是“同时找很多目标”时，召回率会明显下降。工程上要接受一个现实：窗口更长，通常意味着可处理范围更大；不意味着每个角落的信息都同样可靠，也不意味着成本、时延和提示复杂度会自动消失。

| 上下文规模 | 典型代表 | 大致等效量 | 工程含义 |
| --- | --- | --- | --- |
| 128k token | 传统长上下文上限 | 长文档、单仓库局部模块 | 适合中等长度分析 |
| 1M token | Gemini 1.5 Pro 公布的生产级能力 | 约 70 万词、约 5 万行代码、1 小时视频 | 可以把“整套材料”放进一次推理 |
| 10M token | Gemini 1.5 系列研究实验 | 远超单次常规业务输入 | 证明可用边界，不等于日常都该这样用 |

---

## 问题定义与边界

长上下文要解决的问题，不是“模型看不到前文”，而是“前文越长，关键信息越容易被稀释”。稀释的意思很直接：你把资料越堆越多，真正有用的那几句越像掉进草堆里的针，模型未必能稳定把它们找回来。

对零基础读者，可以把它理解成两类任务：

| 任务类型 | 真正困难点 | 长上下文是否直接有效 |
| --- | --- | --- |
| 单文档总结 | 信息量大，但目标简单 | 通常有效 |
| 跨文档问答 | 需要在多个位置建立对应关系 | 有效，但提示结构很重要 |
| 整仓代码理解 | 函数、模块、调用链分散 | 非常适合长上下文 |
| 多目标检索 | 一次要找很多“针” | 容易掉召回 |
| 精确事实问答 | 只需极少相关片段 | RAG 往往更省 |

Gemini 1.5 Pro 的边界主要有三条。

第一，长上下文能力本质上是“可用信息范围扩大”，不是“记忆无限”。模型仍然要在有限计算预算内决定看哪里、看多深。

第二，needle in a haystack 这类评测说明，单个目标的召回可以很高，但当目标数量增加时，表现会下降。召回率，白话说就是“该找到的信息里，实际找回了多少”。如果一次插入 100 个 needle，模型就从“几乎总能找到”变成“只能找到一部分”。

第三，成本与时延通常随输入长度近似线性增长。线性增长的意思是：输入翻 10 倍，账单和等待时间大概率也要跟着涨，不会因为模型更聪明就免费。

可以把多针场景下的经验边界写成一个启发式关系，不是物理定律，但足够指导工程判断：

$$
R(n, T) \approx R_0(T)\cdot e^{-kn}
$$

其中 $R$ 是召回率，$n$ 是同时要找的目标数，$T$ 是上下文长度，$k$ 是与提示设计、目标分布有关的经验常数。它表达的意思很简单：上下文很长时，目标数越多，召回越容易掉。

玩具例子：把 30 本教材拼成一个大文本，只问“第三本里定义过的某个术语在哪一页”，长上下文很合适；如果改成“把这 30 本书里所有重要定义各找一遍并按主题归类”，就从单针问题变成多针问题，难度会立刻上升。

---

## 核心机制与推导

Gemini 1.5 Pro 之所以能把长上下文推到百万级，不是单靠更大显存，而是靠两类机制配合。

第一类是 MoE，Mixture-of-Experts，中文通常叫混合专家。白话解释：不是让一个统一大网络处理所有 token，而是让路由器把不同输入片段分配给不同“专家”子网络，只激活其中一部分参数。这样做的好处是模型总参数可以很大，但每次真正参与计算的参数不需要同比增长。

第二类是高效注意力。注意力机制的作用，是让当前 token 去看其他位置的信息；高效注意力的目标，是尽量保留长距离依赖，同时减少不必要的全量两两计算。直白说，不是让每个位置都盯着 100 万个位置看，而是尽量只看更相关的那部分。

这两类机制解决的是同一个问题：上下文变长以后，信息总量增长了，但单位 token 可分到的计算预算不能无限增长。如果还是朴素全量处理，成本会过快失控。

论文里一个很关键的观测，是负对数似然在长序列上仍然持续下降。负对数似然，白话说就是“模型对下一个 token 预测得有多不自信”，越低越好。经验拟合形式为：

$$
L(T) = \alpha T^{\beta} + \gamma,\quad \beta < 0
$$

这里 $T$ 是可利用的上下文长度，$L(T)$ 越低说明预测越好。$\beta < 0$ 表示随着上下文增长，损失还在下降，也就是额外上下文不是纯噪声，模型确实从更长输入里拿到了增量信息。

这条式子的工程意义比公式本身更重要：如果 $L(T)$ 在很长区间内还持续下降，说明“继续加上下文”仍有价值；如果趋近平台，说明继续加只是在烧钱。

可以把它理解成分工式阅读流程：

1. 路由器先决定哪些片段该交给哪些专家。
2. 注意力模块在局部和远距离之间建立必要连接。
3. 解码阶段把各处证据汇总成最终输出。

玩具例子：一份超长项目文档里同时有 API 设计、数据库迁移、异常日志和测试记录。MoE 的直观类比是把“接口说明”“SQL 变更”“日志模式”“测试行为”分别交给不同分析员先看，再把结果汇总，而不是要求一个分析员从头到尾死记全文。

真实工程例子：把一个服务端仓库、接口文档和 bug 报告一起输入。模型如果只能做短上下文，它往往只能看局部函数；长上下文下，它更可能同时看到“报错栈位置”“调用链上游”“配置差异”“文档里约束条件”，然后给出跨文件、跨材料的因果解释。

---

## 代码实现

工程里通常不会每次都把 1M token 原样重传，而是拆成“稳定前缀 + 变化查询”。稳定前缀，指多次请求都相同的那部分上下文，例如整套代码库、固定规范文档或长视频转写；变化查询，指每次临时问的问题。

下面是一个可运行的 Python 玩具实现，用来模拟“长前缀缓存 + 动态问题”的成本收益。它不调用真实 Gemini API，但逻辑与实际做法一致：固定前缀尽量复用，避免重复支付大输入成本。

```python
from dataclasses import dataclass

@dataclass
class Pricing:
    input_per_million: float
    cache_per_million: float
    storage_per_million_per_hour: float

def request_cost(tokens: int, pricing: Pricing) -> float:
    return tokens / 1_000_000 * pricing.input_per_million

def cached_request_cost(prefix_tokens: int, query_tokens: int, hours: float, pricing: Pricing) -> float:
    cache_write = prefix_tokens / 1_000_000 * pricing.cache_per_million
    storage = prefix_tokens / 1_000_000 * pricing.storage_per_million_per_hour * hours
    query = query_tokens / 1_000_000 * pricing.input_per_million
    return cache_write + storage + query

pricing = Pricing(
    input_per_million=2.50,              # 示例：>200k 输入价
    cache_per_million=0.25,              # 示例：缓存写入价
    storage_per_million_per_hour=4.50    # 示例：缓存存储价
)

prefix_tokens = 800_000
query_tokens = 20_000
hours = 1.0
repeat_queries = 10

without_cache = repeat_queries * request_cost(prefix_tokens + query_tokens, pricing)
with_cache = cached_request_cost(prefix_tokens, query_tokens, hours, pricing) \
    + (repeat_queries - 1) * request_cost(query_tokens, pricing)

assert without_cache > with_cache
assert round(without_cache, 2) == 20.5
assert round(with_cache, 2) == 6.95

print("without_cache =", round(without_cache, 2))
print("with_cache =", round(with_cache, 2))
```

这个玩具例子说明一件事：如果你只问一次，缓存未必划算；如果你对同一份 80 万 token 前缀反复提问，缓存通常明显省钱。

真实接口层面的伪代码通常会长这样：

```python
def query_gemini(cached_content_name: str, user_query: str):
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=user_query,
        config={
            "cached_content": cached_content_name
        }
    )
    usage = response.usage_metadata
    assert usage["cached_content_token_count"] >= 0
    return response.text, usage
```

这里的关键字段是 `cached_content_token_count`。它表示这次请求里有多少 token 命中了缓存。命中高，说明你的“稳定前缀”设计是对的；命中低，说明每次请求的前缀变化太大，缓存收益会很差。

真实工程例子：团队把完整代码库、架构文档和接口契约作为缓存前缀，后续每次只提交“分析这个 bug”“为这个 PR 生成风险说明”“画出模块调用关系”这类短查询。这样做的价值，不是让模型更聪明，而是把同一批重材料只处理一次。

---

## 工程权衡与常见坑

第一类坑是“长上下文替代了一切”。这通常是错的。长上下文适合“大量原始材料必须同时被看到”的任务，不适合所有问答。只问一个函数签名，先检索再提问往往更快更便宜。

第二类坑是“单针表现好，所以多针也会稳”。这也不成立。单针检索更像“已知要找什么”；多针检索更像“同时做很多个找针任务”，模型的注意力和输出空间都会被竞争。

第三类坑是“塞得越满越好”。实际相反。上下文越长，提示越要结构化。最有效的做法通常不是直接堆材料，而是先给材料分区并标注用途，例如“目录结构”“关键约束”“待回答问题”“输出格式”。

下面这张表更接近工程决策时该看的指标：

| 输入规模 | 典型命中风险 | 成本/时延趋势 | 建议对策 |
| --- | --- | --- | --- |
| 0-128k | 风险较低 | 可控 | 正常提示即可 |
| 128k-512k | 关键信息易被冲淡 | 明显上升 | 分段标题、显式索引、限制问题数 |
| 512k-1M | 多针召回明显波动 | 高 | 用缓存复用前缀，单次只问少量核心问题 |
| 1M 以上 | 结果稳定性依赖任务设计 | 很高 | 只在必须全量上下文时使用，优先做分层分析 |

真实工程例子：把 3 万多行服务端代码、API 文档和线上 bug 报告一次送入模型，确实可能一次产出调用链和改造建议；但如果同一请求再加上 20 个不相关问题、多个日志片段和历史会议纪要，答案就容易变散，甚至遗漏关键因果链。

一个常见补救策略是“三层提示”：

1. 顶层定义任务和输出格式。
2. 中层给材料分区和编号。
3. 底层只问 1 到 3 个核心问题。

这比“把所有问题一次写成一大段”稳定得多。

---

## 替代方案与适用边界

长上下文不是 RAG 的替代品，而是另一条路线。RAG，检索增强生成，白话说就是先从外部知识库里找出少量相关片段，再把这些片段交给模型回答。它的核心价值不是模型更强，而是把无关材料挡在外面。

两者适用边界可以直接对比：

| 方案 | token 使用 | 延迟 | 成本控制 | 适合场景 |
| --- | --- | --- | --- | --- |
| 长上下文 | 高 | 较高 | 依赖缓存和问题收敛 | 跨文档推理、整仓代码理解、长视频分析 |
| RAG | 低到中 | 较低 | 好控制 | 精确问答、知识库客服、频繁小问题 |
| 长上下文 + 缓存 | 首次高，后续低 | 首次高，后续改善 | 适合重复查询 | 同一长文档反复问不同问题 |
| RAG + 重排序 | 中 | 中 | 稳定 | 对召回准确性要求高的事实型任务 |

如果你每次都要围绕同一份 50 万到 100 万 token 的材料工作，例如大型 API 文档、法律条文集合、完整代码仓库，那么长上下文加缓存会显著减少系统复杂度。因为你不用维护“先检索哪几段、再怎么拼回上下文”的链路。

但如果你的真实需求只是“问某个函数的参数意义”或“查某条规定的准确表述”，RAG 往往更合适。原因很简单：问题只需要几段证据，就没有必要把 100 万 token 全带上。

一句话判断边界：

- 必须同时看到大量原始材料，选长上下文。
- 只需要少量高相关证据，优先 RAG。
- 会反复问同一批大材料，加缓存。
- 问题很多且彼此无关，拆成多次请求。

---

## 参考资料

1. Google 官方博客，2024 年 2 月发布的 Gemini 1.5 Pro 介绍，包含 1M 生产窗口与 10M 研究实验说明：https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/
2. Gemini 1.5 技术报告《Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context》，包含长上下文 NLL 曲线、needle-in-a-haystack 与多针评测：https://indico.cern.ch/event/1381905/contributions/5814657/attachments/2820312/4924824/gemini_v1_5_report.pdf
3. Google Cloud 关于 needle-in-a-haystack 的说明，补充了单针与多针检索表现差异：https://cloud.google.com/blog/products/ai-machine-learning/the-needle-in-the-haystack-test-and-how-gemini-pro-solves-it
4. Google AI for Developers 的 context caching 文档，说明缓存机制、TTL 与 `cached_content_token_count` 用法：https://ai.google.dev/gemini-api/docs/caching/
5. Google AI for Developers 的定价页面，用于核对上下文缓存与长提示的价格结构：https://ai.google.dev/pricing
