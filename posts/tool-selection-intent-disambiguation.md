## 核心结论

工具选择的意图识别，本质上不是“让模型自己想想该调哪个函数”，而是把“用户问题”和“工具描述”做一次受约束的匹配。这里的“受约束”，不是限制模型能力，而是主动减少歧义：不要让模型在几十个相似工具里裸选。

当工具数量超过 20 个之后，工具选择准确率通常会明显下降。原因通常不是模型突然不会推理，而是候选工具过多、描述重叠、边界不清，模型面对的是一个高歧义决策空间。像 `search_web`、`browse_url`、`fetch_page` 都和“查网页”有关，如果描述又都写成“用于获取网页内容”，误选就会迅速增多。

实践里最有效的三种改进手段，通常是这三件事：

1. 工具分组：先按领域聚类，例如“搜索类”“文件类”“日程类”，先缩小竞争范围。
2. 两阶段选择：先用 embedding 检索出 top-k 候选，再让 LLM 在小集合里做最终判别。
3. 负样例描述：在工具说明里明确写“何时不用”，也就是 `when NOT to use`。

这三件事的共同目标只有一个：把“相似工具互相干扰”这件事压下去。很多系统的改进并不是来自更强的模型，而是来自更干净的候选集和更清晰的边界描述。把 50 个工具直接平铺给模型，往往会进入“看起来都能用”的状态；而分组、检索预筛和负样例描述叠加之后，工具选择才会重新回到可控区间。

先看两阶段路由的角色分工：

| 阶段 | 作用 | 价值 |
| --- | --- | --- |
| Embedding 检索 | 把语义最贴近的 $k$ 个工具挑出 | 让 LLM 只需面对可控决策空间 |
| LLM 排序 | 用自然语言理解对候选做最终判断 | 处理示例、负例、`Do NOT use` 这类细粒度边界 |

可以用一个点餐例子理解。假设菜单里有 40 道菜，如果全部平铺给服务员，“奶油蘑菇汤”和“蘑菇炒饭”都可能只剩下“蘑菇”这个表面特征；但如果先分成“汤”“主食”“甜品”，再在菜名旁边注明“勿用于主食”“勿用于甜点”，误点率会立刻下降。工具选择也是同一个结构：先减小候选，再强化边界。

---

## 问题定义与边界

这里说的“工具选择意图识别”，是指在一个多工具系统里，把用户请求映射到正确工具的过程。它关心的不是工具内部怎么执行，而是第一步“该选谁”。

问题边界先说清楚：

1. 这不是问答任务。目标不是直接生成答案，而是先决定是否调用工具、调用哪个工具。
2. 这不是参数抽取。参数抽取发生在选中工具之后，例如已经决定调用 `weather_api`，再从问题里抽出城市和日期。
3. 这也不等于整个路由系统。完整路由还包括“不该调用任何工具”“需要多个工具串联”“调用失败后回退”“是否向用户澄清”等问题。本文只聚焦单次工具选择的准确性与消歧。

为什么工具一多就容易出问题？可以用一个简单量来描述决策空间的歧义程度：

$$
\lambda=\frac{K\cdot \bar{R}_q}{N}
$$

其中：

- $N$ 是工具总数。
- $K$ 是当前真正展示给模型看的候选数。
- $\bar{R}_q$ 是每个查询平均命中的相关工具数。直白说，就是“看起来都像能回答这个问题的工具有几个”。

这个式子不是标准工业指标，而是一个很实用的分析视角：  
如果 $K$ 很大，而且每个查询又经常对应多个相似工具，那么模型面对的就不再是“答案明显”的选择题，而是“多个选项都像对”的歧义题。

表格更直观：

| 配置 | $K$ | $\bar{R}_q$ | $N$ | $\lambda$ | 结论 |
| --- | --- | --- | --- | --- | --- |
| 只展示 5 个工具 | 5 | 4 | 58 | 0.34 | 选择仍可控 |
| 展示 20 个工具 | 20 | 4 | 58 | 1.38 | 开始明显退化 |
| 展示全部 58 个工具 | 58 | 4 | 58 | 4.0 | 进入高歧义区 |

这里最容易误解的一点是：工具总数 $N$ 大，不一定必然出错；真正危险的是“展示给模型的候选数 $K$ 很大”，同时“相似候选很多”。所以问题核心不是简单删工具，而是减少每次决策时的有效竞争者。

再看一个对新手更直观的例子。假设一台自动售货机里有 58 种冷饮，其中很多名字都像“Cola”“Zero Cola”“Diet Cola”“Sparkling Cola”。用户只说“来杯可乐”，从人类角度已经有歧义；如果每个商品说明又都写成“含气饮料、适合解渴”，那就是多个候选高度重叠。模型并不是缺知识，而是缺边界。

真实工程里也是一样。下面这几个工具都和“网页”有关，但职责并不相同：

| 工具 | 正确用途 | 不该处理的情况 |
| --- | --- | --- |
| `search_web` | 不知道具体网址，需要先检索公开信息 | 已经给出明确 URL |
| `browse_url` | 已有明确 URL，需要抓取或读取页面内容 | 需要先搜候选网页 |
| `fetch_api` | 请求结构化接口数据 | 面向普通网页内容抓取 |
| `open_local_doc` | 打开本地文档或知识库文件 | 面向外部互联网搜索 |

如果没有这些边界说明，用户问“帮我看看 OpenAI 官网最新文档”时，模型就可能在“先搜索官网入口”“直接打开文档链接”“调用 API 文档抓取器”之间来回摇摆。

为了让边界更明确，可以把“用户输入形态”和“工具决策”对应起来：

| 用户表达 | 首要判断 | 更可能的工具 |
| --- | --- | --- |
| “帮我查一下……” | 目标网址未知 | `search_web` |
| “看这个链接里的内容……” | 目标网址已知 | `browse_url` |
| “查某城市明天气温” | 结构化事实查询 | `weather_lookup` |
| “算一下涨了多少百分比” | 数值运算 | `calculator` |
| “这个词是什么意思” | 不一定需要工具 | `none` 或普通回答 |

这张表的意义不是做硬编码，而是提醒一个事实：很多误选不是因为模型没看懂语义，而是因为系统根本没有把“未知网址”和“已知网址”这类决策边界写出来。

---

## 核心机制与推导

一个有效的工具选择系统，通常不是让 LLM 从头在全部工具中挑，而是把选择拆成两个层次：

1. 语义检索层：先用 embedding 找到最像的候选。
2. 语义判别层：再由 LLM 在小集合中做最终决策。

第一步可以写成：

$$
t^{(1)}=\arg\max_{t\in \text{Tools}} \text{sim}(q,\text{desc}(t))
$$

其中：

- $q$ 是用户查询。
- $\text{desc}(t)$ 是工具描述的向量表示。
- $\text{sim}$ 是相似度函数，最常见是余弦相似度。

这一步的含义很直接：先不让模型处理复杂流程，只做一个简单任务，“这个问题和哪个工具描述最像”。

但工程里通常不会只取一个工具，而是取前 $k$ 个候选：

$$
C_k(q)=\operatorname{TopK}_{t\in\text{Tools}} \ \text{sim}(q,\text{desc}(t))
$$

这里的 $C_k(q)$ 表示候选集合。它解决的不是“最后选谁”，而是“先让哪些工具进入决赛圈”。

为什么这一步有效？因为它先解决了集合规模问题。全量 50 个工具里，LLM 很容易被无关但表面相近的描述干扰；embedding 检索的作用，是先把明显不相关的大部分工具踢出去，把候选数从 $N$ 压到 $k$，其中 $k\ll N$。

例如用户问：

> 找一下苹果公司最新股价。

embedding 检索可能得到下面这样的 top-k：

| 工具 | 相似度 | 说明 |
| --- | --- | --- |
| `finance_api` | 0.812 | 直接覆盖股票、行情、代码、股价 |
| `search_web` | 0.671 | 也能处理“最新公开信息” |
| `calculator` | 0.233 | 与“数值”相关，但不负责实时数据 |
| `weather_lookup` | 0.041 | 基本无关 |

这个结果不代表相似度第二名就一定能完成任务，它只代表“它在语义空间里也像相关工具”。关键收益在于：后续 LLM 不必再在 50 个工具里找，而只需在 top-3 或 top-5 里判断谁最合理。

第二步可以抽象成一个条件判别：

$$
t^{(2)}=\arg\max_{t\in C_k(q)} P\bigl(t \mid q,\text{desc}(t),\text{neg}(t),\text{examples}(t)\bigr)
$$

其中：

- $\text{neg}(t)$ 是工具的负样例描述，即“不该用我的场景”。
- $\text{examples}(t)$ 是少量示例，帮助模型学习边界。

这里可以把两阶段的职责彻底分开看：

| 层 | 主要目标 | 最怕什么错误 |
| --- | --- | --- |
| 检索层 | 召回正确工具，不漏掉它 | 正确工具没进 top-k |
| 判别层 | 在小集合中消歧，拒绝近似但错误工具 | 候选都进来了却选错 |

如果只做 embedding，会遇到一个问题：向量相似不等于功能正确。比如“股票价格”“天气温度”“汇率换算”都属于“查询最新数值”，它们在向量空间里未必离得很远。此时 LLM 仍然需要根据工具说明中的边界做最后判断。

负样例描述在这里尤其关键。所谓负样例，直白说，就是告诉模型“别在这些情况用我”。例如：

| 工具 | 正向描述 | 负向描述 |
| --- | --- | --- |
| `search_web` | 适合不知道具体链接、需要搜索公开网页时使用 | 不要用于访问用户已经给出的 URL |
| `browse_url` | 适合读取已知页面内容 | 不要用于从搜索引擎检索新页面 |
| `calculator` | 适合四则运算、百分比、汇率推算 | 不要用于查询实时股价或天气 |

这种写法会直接改变排序质量，因为它不只提供“我能做什么”，还提供“我不能做什么”。在高度相似的工具之间，这种排除信息通常比正向能力描述更有区分度。

把这套机制和前面的 $\lambda$ 放在一起，逻辑就清楚了：

1. 直接把全部工具给 LLM，$K$ 很大，$\lambda$ 容易上升。
2. embedding 检索把 $K$ 从 $N$ 压到 $k$，歧义随之下降。
3. 工具分组进一步降低每个查询的相关重叠数 $\bar{R}_q$。
4. 负样例描述继续压缩“多个工具都像对”的空间。

所以这三种方法不是彼此独立，而是在共同降低决策熵。工程上看，它们分别作用于三个层面：

| 方法 | 影响对象 | 本质作用 |
| --- | --- | --- |
| 分组 | 候选池范围 | 减少无关竞争者 |
| top-k 检索 | 决赛圈大小 | 限制模型注意力分散 |
| 负样例 | 候选间边界 | 提高相邻工具可分性 |

---

## 代码实现

下面给一个最小可运行版本。它不依赖外部模型，直接用纯 Python 实现一个玩具路由器，目的是把流程讲清楚：定义工具、计算相似度、筛出 top-k、做阈值回退、在 top-k 内用“正向词”和“负向词”做二次判别。

这个版本可以直接运行：

```python
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


TOOLS: List[Dict[str, object]] = [
    {
        "name": "search_web",
        "desc": "搜索公开网页信息。适用于不知道具体网址、需要查找最新公开资料、新闻、网页入口。",
        "not_use_for": "不要用于读取用户已经给出的明确URL；不要用于本地文件读取。",
        "positive_terms": ["搜索", "查找", "最新", "公开", "网页", "官网", "入口", "新闻"],
        "negative_terms": ["链接", "url", "网页内容", "本地文件"],
    },
    {
        "name": "browse_url",
        "desc": "读取指定URL的网页内容。适用于用户已经提供明确链接，需要抓取或阅读页面正文。",
        "not_use_for": "不要用于全网搜索；不要用于查找未知网页。",
        "positive_terms": ["链接", "url", "读取", "打开", "网页内容", "页面内容", "http", "https"],
        "negative_terms": ["搜索", "查找入口", "全网"],
    },
    {
        "name": "weather_lookup",
        "desc": "查询城市天气、气温、降雨、风力等信息。",
        "not_use_for": "不要用于股票价格、网页搜索、数学计算。",
        "positive_terms": ["天气", "气温", "降雨", "风力", "温度", "明天", "今天", "城市"],
        "negative_terms": ["股票", "网页", "百分比", "链接"],
    },
    {
        "name": "finance_api",
        "desc": "查询股票、基金、汇率等金融市场数据，适用于最新股价、涨跌幅、代码查询。",
        "not_use_for": "不要用于纯数学运算；不要用于读取网页正文。",
        "positive_terms": ["股票", "股价", "基金", "汇率", "涨跌幅", "代码", "市值", "行情"],
        "negative_terms": ["天气", "链接", "网页正文"],
    },
    {
        "name": "calculator",
        "desc": "执行数学计算，适用于加减乘除、百分比、同比、环比、汇率换算。",
        "not_use_for": "不要用于获取实时信息；不要用于搜索网页。",
        "positive_terms": ["计算", "多少", "百分比", "涨了", "降了", "同比", "环比", "换算"],
        "negative_terms": ["最新股价", "天气", "网页", "链接"],
    },
]

VOCAB = sorted(
    {
        token
        for tool in TOOLS
        for token in (
            tool["positive_terms"]
            + tool["negative_terms"]
            + tokenize_seed(tool["desc"])
            + tokenize_seed(tool["not_use_for"])
        )
    }
)


def tokenize_seed(text: str) -> List[str]:
    parts = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]{1,6}", text.lower())
    return [p for p in parts if p.strip()]


def tokenize(text: str) -> List[str]:
    text = text.lower()
    matched = []
    for term in VOCAB:
        if term in text:
            matched.append(term)
    return matched


def vectorize(text: str) -> List[float]:
    counts = Counter(tokenize(text))
    return [float(counts.get(term, 0.0)) for term in VOCAB]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def lexical_score(query: str, tool: Dict[str, object]) -> float:
    q = query.lower()
    pos_hits = sum(1 for t in tool["positive_terms"] if t in q)
    neg_hits = sum(1 for t in tool["negative_terms"] if t in q)
    return pos_hits * 0.12 - neg_hits * 0.10


def route_query(
    query: str,
    top_k: int = 3,
    retrieval_threshold: float = 0.08,
    clarify_margin: float = 0.06,
) -> Dict[str, object]:
    qv = vectorize(query)
    scored: List[Tuple[str, float, float, float]] = []

    for tool in TOOLS:
        emb = cosine_similarity(qv, vectorize(tool["desc"] + " " + tool["not_use_for"]))
        lex = lexical_score(query, tool)
        final = emb + lex
        scored.append((tool["name"], round(emb, 3), round(lex, 3), round(final, 3)))

    scored.sort(key=lambda x: x[3], reverse=True)
    top_candidates = scored[:top_k]
    best = top_candidates[0]

    if best[3] < retrieval_threshold:
        return {
            "decision": None,
            "reason": "no_match",
            "top_k": top_candidates,
        }

    if len(top_candidates) > 1 and (best[3] - top_candidates[1][3]) < clarify_margin:
        return {
            "decision": None,
            "reason": "need_clarification",
            "top_k": top_candidates,
        }

    return {
        "decision": best[0],
        "reason": "selected",
        "top_k": top_candidates,
    }


def demo() -> None:
    queries = [
        "帮我找苹果公司最新股价",
        "读取这个链接的网页内容：https://example.com/docs",
        "上海明天气温多少",
        "苹果股价比上周涨了多少百分比",
        "帮我看看 OpenAI 官网最新文档",
    ]

    for q in queries:
        result = route_query(q)
        print(f"query: {q}")
        print(f"decision: {result['decision']}, reason: {result['reason']}")
        print(f"top_k: {result['top_k']}")
        print("-" * 60)


if __name__ == "__main__":
    demo()
```

这段代码有几个点值得说明。

第一，它把“工具描述”和“何时不用”都纳入了打分。  
这和很多新手最容易写出来的版本不同。新手常见写法只有一句工具描述，例如“用于搜索信息”“用于读取网页”，这样几乎没有边界。这里把 `desc` 和 `not_use_for` 一起纳入表示，目的就是让工具说明本身具备消歧能力。

第二，它把流程拆成了两步。  
`cosine_similarity` 对应“检索层”，先看 query 和工具描述在向量空间里像不像；`lexical_score` 对应一个非常简化的“二次判别层”，用命中词和负向词做微调。真实生产环境里，这个第二步通常由 LLM 完成，这里只是为了让流程在本地可跑、可解释。

第三，它没有在低置信度时强行选择。  
两个分支很关键：

- 最高分低于 `retrieval_threshold`，返回 `no_match`
- top-1 和 top-2 分差小于 `clarify_margin`，返回 `need_clarification`

这两个回退条件，直接决定了系统会不会“瞎猜”。在真实系统里，“不知道”和“需要澄清”通常比“自信地选错”更可接受。

下面看一组典型输入和期望输出：

| 查询 | 期望决策 | 原因 |
| --- | --- | --- |
| `帮我找苹果公司最新股价` | `finance_api` | 目标是实时金融数据 |
| `读取这个链接的网页内容：https://...` | `browse_url` | 已知 URL，应直接读取 |
| `上海明天气温多少` | `weather_lookup` | 结构化天气查询 |
| `苹果股价比上周涨了多少百分比` | `need_clarification` 或多工具 | 既要金融数据，又要计算 |
| `帮我看看 OpenAI 官网最新文档` | `search_web` | 用户未给 URL，先搜索入口 |

如果要把它改成更接近生产的版本，一般会做下面几件事：

1. 用真实 embedding 模型替换手写向量，例如 `text-embedding-3-large`、`bge`、`e5` 一类模型。
2. 预计算工具向量并缓存，避免每个请求重复编码全部工具。
3. 把 top-k 候选、分数、负样例说明一起交给 LLM 做最终排序。
4. 对“无匹配”“分差过小”“可能多工具”建立明确回退策略。
5. 把路由结果、候选集、最终执行结果写入日志，用来做离线回放。

一个更接近生产环境的伪流程如下：

```text
1. 接收用户查询 q
2. 计算 q 的 embedding
3. 在工具索引中检索 top-k 候选
4. 若 top-1 分数低于阈值，则返回 none 或普通对话
5. 若 top-1 与 top-2 分差过小，则进入澄清流程
6. 否则把 top-k 的工具名、描述、负样例、少量示例发给 LLM
7. LLM 输出最终工具，或输出需要多工具/需要澄清
8. 执行后记录日志，进入回放评估
```

如果把这个流程应用到“股票价格查询”，就很容易理解各层职责。假设系统里有 50 多个工具，embedding 阶段可能只保留 `finance_api`、`search_web`、`calculator`。之后 LLM 再根据描述判断：

- 如果用户要“最新股价”，优先 `finance_api`
- 如果用户要“和上周相比涨了多少百分比”，可能需要 `finance_api + calculator`
- 如果系统根本没有金融接口，才退回 `search_web`

所以 embedding 不是替代 LLM，而是在给 LLM 创造一个可判断的局部空间。

---

## 工程权衡与常见坑

工具选择不是纯算法问题，它直接受延迟、token 成本、设备资源和工具治理质量影响。

先看一个工程权衡表：

| 设计点 | 好处 | 代价 |
| --- | --- | --- |
| 全量交给 LLM 选 | 实现最简单，原型快 | 工具一多就降准，token 成本高 |
| Embedding 先筛选 | 延迟更稳，候选可控 | 要维护向量索引、阈值与召回率 |
| 分层路由 | 大工具集下准确率更高 | 系统结构更复杂，观测点更多 |
| 负样例描述 | 消歧效果明显 | 需要持续维护文档质量 |
| 澄清回退 | 降低误调用成本 | 会增加一轮交互 |

对大多数系统，真正困难的不是“实现一个路由器”，而是长期治理工具清单。工具一旦增长，系统问题往往会从模型问题转成配置问题：命名混乱、描述重叠、示例失效、工具下线后文档没同步更新。

在资源受限环境里，Less-is-More 这类方法的价值会更明显。它的核心思想很直接：设备越弱，越不应该每次把完整工具集搬进上下文。先在粗粒度层级判断，再在细粒度层级确认，可以同时减少 token、延迟和功耗。

一个简化的层级路由结构可以写成：

| 层级 | 选择对象 | 典型输出 |
| --- | --- | --- |
| Level 1 | 领域组 | `搜索类` / `办公类` / `金融类` |
| Level 2 | 组内候选 | top-k 工具 |
| Level 3 | 最终工具 | `finance_api` |
| Level 4 | 执行策略 | 单工具 / 多工具 / 需澄清 |

这类设计在边缘设备上尤其有价值，例如车机、手表、本地助手、小模型终端。因为这些系统不仅怕错，还怕慢、怕耗电、怕上下文塞不下。

最常见的坑，通常有下面几类。

第一，工具描述写得太宽泛。  
例如把 `search_web` 写成“用于查找信息”，把 `weather_lookup` 写成“用于查询最新信息”。这两句都没错，但边界几乎没有。模型看到的是两个大而空的能力集合。

第二，没有写负样例。  
“什么时候不用我”往往比“什么时候用我”更重要。尤其在相邻工具之间，负样例是最直接的分隔线。

第三，工具名称相似且描述重叠。  
像 `search_web`、`web_search`、`internet_search` 这类名字，本身就在制造干扰。如果功能没有明显差别，应该合并；如果确有区别，应该把差别写进名称本身，例如 `search_web_by_query` 与 `read_webpage_by_url`。

第四，只看 top-1，不看分差。  
假设 top-1 是 0.31，top-2 是 0.30，这种情况下系统其实并不确定。正确做法通常不是直接调用，而是继续判别，或向用户追问“你是要搜索网页，还是读取已有链接？”

第五，评测数据过于干净。  
真实用户输入往往短、脏、歧义大，例如“查一下苹果现在多少”。这里的“苹果”可能是公司、手机价格、水果热量，也可能是某个 App 的订阅价格。评测如果只用规范请求，会系统性高估效果。

第六，只评估“有没有选中”，不评估“错得多严重”。  
工具选择有两种错误严重度完全不同：

| 错误类型 | 例子 | 风险 |
| --- | --- | --- |
| 同领域误选 | `search_web` 选成 `browse_url` | 多半还能回退 |
| 跨领域误选 | `weather_lookup` 选成 `finance_api` | 结果通常完全失真 |
| 误触发工具 | 本可直接回答却调用外部接口 | 增加延迟和成本 |
| 漏掉澄清 | 本应追问却直接猜测 | 用户体验最差 |

因此工程上最好建立回放机制。简单说，就是把真实流量里的查询、候选、最终结果、失败样本记录下来，定期看三类误差：

- 误选到同领域但错误工具
- 本该不用工具却误调用
- 本该澄清却直接猜测

这比只盯平均准确率更有价值。平均准确率会掩盖严重错误，而错误分布更接近真实业务风险。

最后再补一个容易被忽略的点：  
工具描述本身就是训练数据的一部分。即使你不做参数微调，工具名、工具说明、负样例、示例，也在实时塑造模型的决策边界。很多系统路由效果差，不是模型不够强，而是“在线提示词里的工具文档”质量太低。

---

## 替代方案与适用边界

工具路由并不是只有 hybrid 一条路。常见可以分成三类：

| 策略 | 优点 | 缺点 | 适合 |
| --- | --- | --- | --- |
| LLM-based | 语义灵活，可顺带做解释和多步规划 | 高延迟，高 token 成本，工具多时易混淆 | 工具 < 20、边界差异大 |
| Similarity-based | 低延迟，部署简单，易并行 | 区分力有限，受召回上限约束 | 高频、低歧义、模式稳定请求 |
| Hybrid | 精度与效率更平衡，适合大工具集 | 结构复杂，需要治理索引和阈值 | 工具 ≥ 20 且相似度高的系统 |

先说 LLM-only。  
如果系统只有 5 个工具，而且每个工具差异很大，例如“天气”“计算器”“地图”“发邮件”“翻译”，那直接把全部工具描述给 LLM 完全可行。它的优点是开发简单、灵活，模型还能顺带解释为什么这么选。

再说 Similarity-only。  
如果场景固定、问题模式高度重复，比如一个客服机器人只在“查物流”“查退款”“查订单”之间切换，那么 embedding 加阈值可能已经够用。它快、便宜、易部署。但它不适合细粒度差异很多的开放场景，因为相似度无法稳定表达复杂边界。

Hybrid 的适用边界最广，但它更像一个工程系统，而不是一个单点技巧。你需要持续维护：

- 工具清单
- 工具命名规范
- 工具描述
- 负样例
- 分组结构
- 阈值策略
- 澄清逻辑
- 离线评测与线上回放

所以不是所有项目都该一上来做 hybrid。一个实用判断标准是下面四个问题：

1. 工具数是否已经接近或超过 20。
2. 是否经常出现语义相近工具。
3. 错误调用的代价是否较高。
4. 是否受延迟和 token 成本约束。

如果前两项成立，hybrid 基本就值得考虑；如果后两项也成立，那分层路由通常就是必要配置，而不是可选优化。

可以再把三种策略放到一个更具体的决策表里：

| 场景 | 更合适的方案 | 原因 |
| --- | --- | --- |
| 工具少、边界差异大 | LLM-only | 直接判断成本低，灵活性高 |
| 高频标准问句 | Similarity-only | 快且便宜，维护简单 |
| 工具多且高度相似 | Hybrid | 需要先缩小候选再做消歧 |
| 边缘设备、本地模型 | 分层 Hybrid | 同时控制上下文、延迟和功耗 |
| 高风险操作系统 | Hybrid + 澄清 | 错误调用代价高，必须保守 |

最后落回一个简单判断：

- 工具少、差异大，用 LLM-only。
- 工具多、歧义高，用 hybrid。
- 请求高频、模式固定，用 similarity-only。
- 错误代价高时，不要追求“每次都必须选一个”，而要优先支持 `none` 和“需要澄清”。

不要把所有问题都塞进同一种路由框架。真正稳定的系统，通常不是“某个模型特别强”，而是它知道什么时候该选、什么时候不该选、什么时候先问清楚再选。

---

## 参考资料

- Michael Brenndoerfer, “Tool Selection for LLM Agents: Routing Strategies and Implementation.”  
  https://mbrenndoerfer.com/writing/tool-selection-llm-agents-routing-strategies

- ICLR Blogposts 2026, “The 99% Success Paradox: When Near-Perfect Retrieval Equals Random Selection.”  
  https://iclr-blogposts.github.io/2026/blog/2026/bits-over-random/

- Varatheepan Paramanayakam, Andreas Karatzas, Iraklis Anagnostopoulos, Dimitrios Stamoulis, “Less is More: Optimizing Function Calling for LLM Execution on Edge Devices.”  
  https://www.engr.siu.edu/staff/iraklis.anagnostopoulos/files/papers/Less_is_More_Optimizing_Function_Calling_for_LLM_Execution_on_Edge_Devices.pdf

- Cailin Winston, René Just, “A Taxonomy of Failures in Tool-Augmented LLMs.”  
  https://homes.cs.washington.edu/~rjust/publ/tallm_testing_ast_2025.pdf

- Anthropic Docs, “Pricing” 与 “Token-efficient tool use.”  
  https://docs.anthropic.com/en/docs/about-claude/pricing  
  https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/token-efficient-tool-use

- Anthropic Docs, “Computer use tool.”  
  https://docs.anthropic.com/en/docs/build-with-claude/computer-use
