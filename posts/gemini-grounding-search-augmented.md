## 核心结论

Gemini 的 Grounding 可以理解为“让模型在回答前，按需调用 Google Search，再把搜索依据连同答案一起返回”。这里的“grounding”就是“把回答绑定到外部证据”，白话说，就是不只给答案，还给出处。

它和传统 RAG 的核心区别不是“都能检索”，而是“检索链路放在哪”。传统 RAG 通常由业务方自己维护抓取、切块、向量化、召回、重排、拼提示词；Gemini Grounding 则把“是否搜、搜什么、如何融合、如何给引用”都内置进模型调用流程。对公开互联网、时效性强的问题，这条链路更短。

需要特别校正一个常见误解：当前主流用法是 `google_search` 工具，开发者只负责开启工具，模型自己决定是否发起搜索。`dynamic_threshold` 这种显式阈值配置，公开文档只在 Gemini 1.5 的旧工具 `google_search_retrieval` 中出现。也就是说，“触发条件”这个概念仍然存在，但在新接口里大多是模型内部决策，而不是你直接传一个阈值。

---

## 问题定义与边界

Grounding 解决的不是“模型不会答”，而是“模型可能需要最新、可核查、外部世界的信息”。比如“法国首都是哪里”属于稳定事实，模型内置知识通常足够；“2026 年 3 月最新的模型价格怎么计费”则属于快变事实，更适合搜索增强。

可以把问题粗分成三类：

| 问题类型 | 例子 | 是否更适合 Grounding | 原因 |
| --- | --- | --- | --- |
| 稳定事实 | 牛顿第二定律是什么 | 通常不需要 | 知识长期不变 |
| 慢变事实 | 某产品长期架构设计 | 看情况 | 文档可能更新，但不是分钟级变化 |
| 快变事实 | 最新价格、比赛结果、政策变更 | 很适合 | 需要实时网页证据 |

玩具例子：用户问“Euro 2024 冠军是谁”。这类问题的答案依赖比赛结果，模型更可能触发搜索，返回答案时还会附带 `webSearchQueries`、`groundingChunks`、`groundingSupports` 等元数据。

边界也很明确。Grounding 依赖公开网页，不等于能访问你的私有知识库；如果你要问公司内部 SOP、客服工单、设计文档，就应该考虑 Vertex AI Search 或传统 RAG。另一个边界是“有开关不等于一定搜到结果”：官方文档明确说明，即使启用了 Grounding，如果来源相关性低，或者模型输出无法形成完整归因，响应里也可能没有 `groundingMetadata`，这时结果实际上退化成普通生成。

---

## 核心机制与推导

先看最小机制。对输入问题 $q$，系统内部会做一个“是否值得搜索”的判断。公开文档没有给出当前 `google_search` 的具体打分公式，但可以用一个抽象式理解：

$$
use\_search = \mathbf{1}[r(q) \ge \theta]
$$

其中 $r(q)$ 表示“该问题需要外部新信息”的内部评分，$\theta$ 表示触发阈值。对当前新接口，这个 $\theta$ 主要由模型内部控制；对 Gemini 1.5 的旧式 `google_search_retrieval`，开发者可以显式设置 `dynamic_threshold`。

一旦触发搜索，模型会自动生成一个或多个查询词，得到结果集合：

$$
S = \{s_1, s_2, \dots, s_n\}
$$

每个结果通常至少包含来源链接和摘要片段。随后模型不是简单把片段原样拼进去，而是先综合这些结果，再生成自然语言答案。最后，系统把“答案中的哪一段对应哪些来源”编码进 `groundingSupports`：

$$
support_i = ([start_i, end_i), I_i), \quad I_i \subseteq \{0,\dots,n-1\}
$$

这里 `[start_i, end_i)` 表示答案文本中的字符区间，$I_i$ 表示这段话对应的来源索引。白话说，就是“答案第 20 到 60 个字符，依据的是第 0 和第 1 个网页”。

这套设计有两个工程价值。

第一，引用不是“整段答案统一挂一个参考链接”，而是可以做到局部归因。这样用户能看出“哪句话对应哪条网页证据”。

第二，检索和生成不是两个松散拼接的服务，而是在同一次模型调用里完成。你看到的 `groundingMetadata` 本质上就是这个内部推理链路留下的结构化痕迹。

真实工程例子：做一个“技术新闻问答”页面，用户问“Gemini 3 的 Search Grounding 现在怎么计费”。如果模型只靠旧知识，很可能答错时间或计费单位；启用 Grounding 后，模型会拉取当前文档，再把“按 prompt 计费”还是“按 query 计费”的差异映射回最终答案。这个场景里，Grounding 的价值不是让回答更长，而是把“答案是否跟得上今天的文档”这件事变得可控。

---

## 代码实现

下面先给一个可运行的玩具实现，模拟如何把 `groundingSupports` 转成行内引用。这个例子不依赖 Gemini SDK，可以直接运行。

```python
def add_citations(text, supports, chunks):
    # 倒序插入，避免前面插入后把后面的索引推偏
    for support in sorted(supports, key=lambda x: x["end"], reverse=True):
        refs = []
        for idx in support["chunk_indices"]:
            refs.append(f"[{idx+1}]({chunks[idx]['uri']})")
        text = text[:support["end"]] + "".join(refs) + text[support["end"]:]
    return text


answer = "Spain won Euro 2024. It was Spain's fourth European Championship title."
chunks = [
    {"uri": "https://example.com/a"},
    {"uri": "https://example.com/b"},
]
supports = [
    {"start": 0, "end": 20, "chunk_indices": [0]},
    {"start": 21, "end": len(answer), "chunk_indices": [0, 1]},
]

rendered = add_citations(answer, supports, chunks)

assert "[1](https://example.com/a)" in rendered
assert "[2](https://example.com/b)" in rendered
print(rendered)
```

如果接真实 Gemini API，请求本身很短，关键是把 `google_search` 工具放进配置里：

```python
from google import genai
from google.genai import types

client = genai.Client()

config = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())]
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Gemini 3 的 Google Search Grounding 现在如何计费？",
    config=config,
)

print(response.text)

meta = response.candidates[0].grounding_metadata
if meta:
    print(meta.web_search_queries)
    print(meta.grounding_chunks)
```

这段代码体现了当前接口的关键点：你不需要自己先调用搜索 API，再把网页内容手工塞回 prompt；只要开启工具，模型会自行决定是否搜索、生成什么查询、如何把来源映射进回答。

---

## 工程权衡与常见坑

Grounding 不是“免费升级版 RAG”，而是对特定问题更省工程的一条路径。它的优势和代价要一起看。

| 维度 | Gemini Grounding | 传统 RAG |
| --- | --- | --- |
| 公开网页时效性 | 强 | 取决于你多久重抓 |
| 私有知识接入 | 弱 | 强 |
| 引用归因 | 内建 | 需自己做 |
| 运维复杂度 | 低 | 高 |
| 可控性 | 中 | 高 |

常见坑主要有四个。

第一，启用了工具，不代表一定有 `groundingMetadata`。官方文档说明，如果来源相关性不足，或者回答无法形成完整归因，元数据可能缺失。工程上不能默认 `response.candidates[0].grounding_metadata` 一定存在，必须判空，否则前端渲染引用时会直接报错。

第二，Search Suggestions 有展示要求。Vertex AI 文档明确要求：如果响应返回了 Search Suggestions，生产环境里需要按要求展示，而不是只取文本答案把建议入口丢掉。很多团队只关心模型文本，忽略这部分合规要求，后面才补 UI。

第三，计费口径会随模型代际变化。到 2026 年 3 月这个时间点，官方文档写明：Gemini 3 上的 Google Search Grounding 从 2026 年 1 月 5 日开始按“每个搜索查询”计费；而 Gemini 2.5 及更早模型通常按“每个 grounded prompt”计费。一个 prompt 触发多次查询时，Gemini 3 的成本会上升更明显。

第四，外部搜索能提高“最新性”，不等于自动保证“业务正确性”。网页结果可能彼此冲突、标题党、地区口径不同。你仍然需要在产品层定义降级策略，例如：
1. 没有 `groundingMetadata` 时，明确提示“未检索到足够可靠的网页依据”。
2. 引用数量过少时，降低答案确定性措辞。
3. 涉及金融、医疗、法律时，把 Grounding 当证据辅助，而不是最终裁决。

---

## 替代方案与适用边界

如果你的问题主要来自公开互联网，并且时效性强，Gemini Grounding 很合适。典型场景是新闻摘要、赛事结果、价格查询、政策更新、工具文档变化。

如果你的问题主要来自私有知识，或者你需要严格控制召回范围，传统 RAG 或 Vertex AI Search 更合适。这里的“私有知识”就是“外部搜索引擎看不到，但业务必须知道的数据”，比如内部文档、工单记录、企业 FAQ。

| 场景 | 推荐方案 | 原因 |
| --- | --- | --- |
| 最新公开信息问答 | Gemini Grounding with Google Search | 时效性强，内建引用 |
| 企业私有知识问答 | Vertex AI Search / RAG | 数据可控，权限可管 |
| 混合公开与私有信息 | 分层架构或多源 Grounding | 同时处理两类证据 |
| 高监管场景 | RAG + 人工审核 | 需要更强审计链路 |

一个实用判断标准是：你更缺“最新网页”，还是更缺“自己的知识底座”。前者优先 Grounding，后者优先 RAG。不要把两者当成谁替代谁，它们更像是两种检索边界不同的工程组件。

---

## 参考资料

- [Google AI for Developers: Grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search)
- [Google AI for Developers: Grounding](https://ai.google.dev/gemini-api/docs/grounding)
- [Google Cloud Vertex AI: Grounding with Google Search](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/grounding/grounding-with-google-search)
- [Google Cloud Vertex AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)
- [Google Cloud: Grounding with Vertex AI Search](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/grounding-with-vertex-ai-search)
- [Google Cloud Architecture: RAG with Gen AI and Gemini on Vertex AI](https://cloud.google.com/architecture/rag-genai-gemini-enterprise-vertexai)
