## 核心结论

Agent 的长期记忆不能只靠上下文窗口。上下文窗口只是模型在一次调用里临时能看到的内容，请求结束后就消失；长期记忆要求信息能跨会话保存、被再次找回、能够更新，并且在需要时可以追溯来源。

工程上更稳妥的方案，不是在 `KV cache`、向量数据库、结构化数据库三者里选一个，而是做分层：

| 层 | 解决的问题 | 典型内容 | 典型代价 |
| --- | --- | --- | --- |
| `KV cache` | 复用已经算过的前缀 | 固定 system prompt、工具 schema、长文档公共前缀 | 依赖前缀稳定，不适合长期归档 |
| 向量数据库 | 从大量历史里按语义召回候选 | 历史对话片段、知识片段、案例 | 模糊匹配强，但不擅长权限和精确约束 |
| 结构化数据库 | 按字段精确过滤与审计 | `user_id`、时间、状态、权限、租户 | 精确但不擅长语义相似 |

这三层的目标不同，所以延迟预算也不同。一个常见的端到端近似式是：

$$
T_{total}=T_{embed}+T_{retrieve}+T_{rerank}+T_{decode}
$$

其中：

- `embed`：把文本编码成向量，便于做相似度比较
- `retrieve`：从外部记忆中取回候选内容
- `rerank`：对候选再做排序、过滤或业务规则打分
- `decode`：模型逐 token 生成回答

如果只保留最粗略的主链路，也可以写成：

$$
T_{total}\approx T_{embed}+T_{retrieve}+T_{decode}
$$

Sparkco 给出的参考预算是索引/嵌入 `<10ms`、热检索 `<50ms`、冷数据获取 `<200ms`。这不是定律，而是在线 Agent 常见的工程目标：用户愿意等待一点检索，但不愿意为每次“回忆”都付出秒级延迟。

`KV cache` 的收益主要来自减少重复前缀的预填充计算。Glean 在 Azure PTU 上的实验观察到，每个命中的 cached token 大约节省 `0.15ms` 的 TTFT；如果跨请求复用 5000 个前缀 token，数量级上可节省约 `750ms`。这个数量级说明了一件很实际的事：固定前缀越长、复用率越高，`KV cache` 越值钱；跨会话的个性化事实越多，外部持久化层越重要。

一句话概括本文结论：

- `KV cache` 负责“把已经算过的公共前缀算一次、复用很多次”
- 向量库负责“从大量历史里模糊找可能相关的候选”
- 结构化库负责“把不该进来的结果挡在外面，并留下可审计的边界”

---

## 问题定义与边界

先把“长期记忆”拆开。这里讨论的不是模型参数里隐含的知识，也不是一次会话内部的短期上下文，而是 Agent 在多轮、多天、甚至多月后，还能重新取回与当前任务相关的信息，例如：

- 用户长期偏好
- 历史订单和操作记录
- 某个任务过去执行到哪一步
- 外部知识库中的文档、规则、案例
- 审计时需要追溯的检索来源

它的核心问题不是“能不能存下来”，而是下面三件事要同时成立：

| 目标 | 含义 | 失败表现 |
| --- | --- | --- |
| 持久化 | 会话结束后信息仍然存在 | 第二天完全忘记用户偏好 |
| 可检索 | 新问题到来时能及时找回相关信息 | 明明存过，却召回不到 |
| 可控性 | 召回满足权限、时间和业务约束 | 找到了相似内容，但不是该用户能看的 |

很多新手把“记住内容”和“正确回忆内容”混为一谈。两者不是一回事。一个系统即使把所有历史都存下来了，如果没有检索索引、过滤规则和时效策略，依然等于“存了但不会用”。

为什么不能把所有东西都塞进 `KV cache`？因为 `KV cache` 本质上是推理期缓存，通常驻留在 GPU HBM 或其附近的高性能内存里，目标是服务当前或近期复用的前缀，而不是做海量、长期、跨会话归档。Cloudidr 给出的分层表可以帮助理解这个边界：

| Tier | Latency | 最优用例 | 限制 |
| --- | --- | --- | --- |
| GPU HBM3e | 10-30ns | 当前上下文、KV cache、活跃推理 | 容量昂贵，不适合长期存储 |
| DDR5 / DRAM | 50-100ns | 热向量、热元数据、近期 session | 容量仍有限，成本高于冷存储 |
| NVMe SSD | 60μs | 向量索引、会话归档、知识库 | 随机访问更慢，不能承担 active attention |

这张表对应一个直接结论：长期记忆必须落到更便宜、更大容量的外部介质上，但外部介质延迟更高，所以系统必须区分热数据和冷数据，而不是把所有信息都放在同一层。

下面用一个更具体的例子说明边界。假设你在做一个“背单词 Agent”：

| 信息类型 | 更适合放在哪一层 | 原因 |
| --- | --- | --- |
| 固定的评分规则、输出模板、批改标准 | `KV cache` | 所有用户共享，前缀稳定，重复率高 |
| 用户过去常错的单词、混淆原因、例句偏好 | 向量库 | 需要按语义找“类似错误”和“相近知识点” |
| 用户 ID、当前等级、最近 7 天学习记录 | 结构化库 | 需要精确过滤，不能靠模糊匹配 |

如果你把全部历史都扔进向量库，系统会“能记住，但回忆太慢”；如果你只靠 `KV cache`，系统会“眼前很快，但隔天就失忆”；如果你只靠结构化数据库，系统会“过滤很准，但不会联想相似经验”。

因此，长期记忆的边界不是“是否保存”，而是：

1. 哪些内容需要跨会话保留
2. 哪些内容要按语义召回
3. 哪些内容必须按字段精确控制
4. 哪些内容值得放入热层，哪些必须下沉冷层

---

## 核心机制与推导

先看 `KV cache` 的作用。Transformer 在处理前缀时，会为每一层自注意力保存历史 token 的 `key` 和 `value`。白话说，模型不是每生成一个新 token 都把整段前文重新算一遍，而是把“前面已经算过、后面还会继续引用”的中间结果缓存起来。

因此，`KV cache` 优化的是重复前缀带来的 **prefill** 成本，而不是让模型“天然拥有长期记忆”。它解决的是同一个或高度相似的前缀在不同请求中被重复计算的问题。

如果前缀长度为 $L$，每个 cached token 带来的平均节省为 $c$ 毫秒，那么 TTFT 的近似收益可以写成：

$$
\Delta T_{prefill}\approx c \times L
$$

将 Glean 给出的经验值 $c \approx 0.15ms$ 代入，若共享前缀长度为 $L=5000$，则有：

$$
\Delta T_{prefill}\approx 0.15 \times 5000=750ms
$$

这个公式不复杂，但它非常有用，因为它回答了一个工程决策问题：什么时候值得为前缀复用投入缓存设计？答案通常是：

- 公共前缀长
- 请求量大
- 多个用户共享相同模板
- 系统对 TTFT 很敏感

反过来，如果前缀中掺杂了大量动态字段，例如用户名、当前时间、trace id、每次变化的检索结果，那么缓存键会频繁变化，命中率就会明显下降。

再看外部记忆。向量检索的基本路径是：

1. 把查询文本转成 embedding
2. 在索引中找最近邻候选
3. 将候选送入模型，或者再做重排后送入模型

它解决的是“语义相近”，不是“业务正确”。语义相似意味着两段文本表达的意思接近，但不代表它们在权限、时间、租户、状态上也是正确的。

因此，更合理的长期记忆主链路通常不是单段检索，而是分阶段检索：

| 阶段 | 作用 | 常见输入 | 常见输出 |
| --- | --- | --- | --- |
| 语义宽召回 | 先从大量历史中抓一批可能相关的候选 | 查询 embedding | `top_k` 候选片段 |
| 结构化精筛 | 用业务字段把不合法候选过滤掉 | `user_id`、时间、状态、租户 | 合法候选集 |
| 重排 / 打分 | 结合相似度、时间衰减、规则分数排序 | 候选集 | 最终上下文 |
| 模型生成 | 使用最终上下文回答 | 上下文 + 当前问题 | 最终输出 |

可以把它理解为“先模糊找，再严格裁，再按业务优先级排序”。

如果把这条链路写成一个简单的打分式，可以表示为：

$$
Score(d, q)=\alpha \cdot Sim(d,q)+\beta \cdot Freshness(d)+\gamma \cdot BusinessRule(d)
$$

其中：

- $Sim(d,q)$：文档与查询的语义相似度
- $Freshness(d)$：时间衰减或新鲜度分数
- $BusinessRule(d)$：业务规则分，例如是否为当前用户、是否处于有效状态
- $\alpha,\beta,\gamma$：不同维度的权重

这类打分在生产里不一定显式写成公式，但工程上一定会存在，因为“最相似”通常不等于“最该返回”。

Sparkco 给出的预算可以当作线上系统的目标值：

| 阶段 | 描述 | 目标延迟 | 说明 |
| --- | --- | --- | --- |
| Embed & Index | 为新内容生成 embedding 或更新索引 | `<10ms` | 在线写入不应长期阻塞主链路 |
| Hot retrieval | 热层检索与重排 | `<50ms` | 适合近期高频记忆 |
| Cold fetch | 冷层归档读取 | `<200ms` | 可更慢，但不宜成为默认路径 |

下面用一个企业差旅 Agent 说明这三层如何配合。用户问：

> 上周三我订的上海到东京机票怎么改签？

这时系统里的三层分别承担不同职责：

| 层 | 实际承担的内容 |
| --- | --- |
| `KV cache` | 固定 system prompt、改签政策摘要、工具 schema、输出模板 |
| 向量库 | 历史订单说明、航司政策解释、相似改签案例、历史对话摘要 |
| 结构化库 | `user_id=123`、订单日期是 `2026-03-04`、状态为“已出票未使用” |

如果只做语义召回，系统可能会拿到别人的东京机票记录；如果只做结构化过滤，又可能只找到订单本身，找不到“改签政策解释”和“相似案例”。所以长期记忆真正的关键不是某一个组件，而是检索链路的组合方式。

---

## 代码实现

下面给出一个最小可运行的 Python 示例。它不依赖第三方库，目的是演示三件事：

1. `KV cache` 只负责复用稳定前缀
2. 外部记忆先做语义召回
3. 结构化字段负责保证结果可控

代码是玩具实现，不代表生产质量，但可以直接运行。

```python
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Optional


VOCAB = ["机票", "改签", "退票", "东京", "酒店", "订单", "支付", "航班"]


@dataclass
class MemoryRecord:
    user_id: int
    text: str
    day: str
    status: str
    emb: List[float] = field(default_factory=list)


class KVCascadeMemory:
    def __init__(self) -> None:
        self.kv_cache: Dict[str, str] = {}
        self.records: List[MemoryRecord] = []

    def put_prefix_cache(self, prefix_key: str, cached_value: str) -> None:
        self.kv_cache[prefix_key] = cached_value

    def invalidate_prefix_cache(self, prefix_key: str) -> None:
        self.kv_cache.pop(prefix_key, None)

    def add_record(self, user_id: int, text: str, day: str, status: str) -> None:
        self.records.append(
            MemoryRecord(
                user_id=user_id,
                text=text,
                day=day,
                status=status,
                emb=self.embed(text),
            )
        )

    def embed(self, text: str) -> List[float]:
        # 极简 bag-of-words 向量，只演示“编码 -> 相似度 -> 排序”的路径
        return [float(text.count(token)) for token in VOCAB]

    def cosine(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sqrt(sum(x * x for x in a))
        nb = sqrt(sum(x * x for x in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def vector_search(
        self,
        query: str,
        user_id: int,
        day: Optional[str] = None,
        status: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict[str, object]]:
        q_emb = self.embed(query)

        filtered = [
            r
            for r in self.records
            if r.user_id == user_id
            and (day is None or r.day == day)
            and (status is None or r.status == status)
        ]

        ranked = sorted(
            filtered,
            key=lambda r: self.cosine(q_emb, r.emb),
            reverse=True,
        )[:top_k]

        return [
            {
                "text": r.text,
                "day": r.day,
                "status": r.status,
                "score": round(self.cosine(q_emb, r.emb), 4),
            }
            for r in ranked
        ]

    def retrieve(
        self,
        query: str,
        user_id: int,
        day: Optional[str] = None,
        status: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, object]:
        prefix_key = "travel_agent_v1"

        if prefix_key in self.kv_cache:
            return {
                "route": "kv_hit",
                "prefix": self.kv_cache[prefix_key],
                "candidates": [],
            }

        candidates = self.vector_search(
            query=query,
            user_id=user_id,
            day=day,
            status=status,
            top_k=top_k,
        )

        return {
            "route": "vector_fallback",
            "prefix": None,
            "candidates": candidates,
        }


def main() -> None:
    memory = KVCascadeMemory()

    memory.add_record(
        user_id=123,
        text="东京机票订单已出票，可改签一次，需补差价",
        day="2026-03-04",
        status="issued",
    )
    memory.add_record(
        user_id=123,
        text="酒店订单支持入住前一天免费取消",
        day="2026-03-04",
        status="booked",
    )
    memory.add_record(
        user_id=999,
        text="东京机票已退票完成",
        day="2026-03-04",
        status="refunded",
    )

    # 1) 没有 KV 命中时，走外部记忆检索
    result = memory.retrieve(
        query="上周三东京机票怎么改签",
        user_id=123,
        day="2026-03-04",
        top_k=3,
    )

    assert result["route"] == "vector_fallback"
    candidates = result["candidates"]
    assert len(candidates) >= 1
    assert candidates[0]["text"].startswith("东京机票订单已出票")
    assert all(c["status"] != "refunded" for c in candidates)

    print("第一次检索结果：")
    print(result)

    # 2) 命中 KV cache 后，直接复用稳定前缀
    memory.put_prefix_cache(
        "travel_agent_v1",
        "固定系统提示词 + 改签政策摘要 + 工具 schema",
    )

    result2 = memory.retrieve(
        query="上周三东京机票怎么改签",
        user_id=123,
        day="2026-03-04",
    )

    assert result2["route"] == "kv_hit"
    assert result2["prefix"].startswith("固定系统提示词")

    print("\n第二次检索结果：")
    print(result2)


if __name__ == "__main__":
    main()
```

运行方式：

```bash
python3 memory_demo.py
```

这个例子的重点不是“做了一个强大的向量数据库”，而是把执行路径拆清楚。第一次查询时，没有命中前缀缓存，系统只能从外部记忆里做检索；第二次查询时，固定前缀已经缓存，可以直接复用。

为了让新手更容易理解，可以把这个玩具系统想成一个两段式流程：

| 路径 | 负责什么 | 为什么快 / 为什么慢 |
| --- | --- | --- |
| `kv_hit` | 复用固定前缀 | 不用重复算前缀，所以快 |
| `vector_fallback` | 召回历史候选 | 要先编码、再检索、再排序，所以更慢 |

但生产系统通常还要再补四块。

### 1. 异步写入

新对话不应该同步阻塞在 embedding 和索引更新上，否则用户会直接感受到写入成本。更常见的做法是：

- 主链路先回答
- 后台把高价值事件写入队列
- 异步更新热层索引和冷层归档

一个很粗略的写入链路可以写成：

$$
Write_{online}=T_{enqueue} \ll T_{embed}+T_{index}
$$

也就是说，在线请求尽量只承担“入队”，把真正昂贵的编码和索引更新后移。

### 2. 前缀版本化

`system prompt`、工具 schema、输出格式一旦变化，旧的 `KV cache` 可能就不能安全复用。因此缓存键不能只写成业务名，通常要带版本号，例如：

```text
travel_agent:v3:policy_2026_03
```

这样做的目的不是“看起来专业”，而是避免老前缀污染新请求。

### 3. 元数据独立存储

向量库里可以放 embedding，但 `tenant_id`、`user_id`、`created_at`、`visibility` 这类字段必须能独立过滤。否则你只能得到“语义接近”的内容，无法保证“业务合法”的内容。

### 4. 记忆压缩与摘要

如果把每条对话都原样存入长期记忆，索引会很快膨胀。更常见的做法是：

- 原始日志进冷层
- 高价值事实抽取成结构化记录
- 会话摘要进热层检索
- 低价值闲聊设置较短 TTL

这一步的目标不是少存，而是把“以后还值得回忆的内容”和“一次性噪声”分开。

---

## 工程权衡与常见坑

最常见的误判，是把“语义相似”当成“业务可用”。向量库擅长找相似内容，但它不会天然理解权限、时序、租户边界和资源状态。只要系统涉及多用户、合规、审计，就必须把结构化过滤放在主路径中，而不是事后补救。

第二个坑，是高估 `KV cache` 的泛化能力。它非常适合“相同前缀 + 不同后缀”的场景，但不适合“每次提示词都在变”的场景。Glean 引述 Azure 的建议也指出：混合不同 workload 会降低 cache hit rate，必要时要拆分部署。直白地说，客服问答、代码生成、分析报告如果共用同一个前缀缓存池，命中率通常不会好看。

下面把常见 trade-off 摊平看：

| 风险 | 原因 | 直接后果 | 规避方式 |
| --- | --- | --- | --- |
| 向量检索变慢 | 数据规模上升、过滤维度增加、写入频繁 | 响应延迟抖动 | 热冷分层、先 metadata 过滤再 ANN、异步索引 |
| 召回到过期事实 | 旧事实没有失效策略 | 输出过时结论 | TTL、事件追加、版本号、时间衰减重排 |
| `KV cache` 命中率低 | prompt 混入动态字段、多个 workload 共池 | TTFT 改善很弱 | 稳定前缀、变量后置、按业务拆缓存池 |
| 热层成本过高 | 低频内容长期占用高性能内存 | 成本高、热点被挤出 | 只保最近热点，下沉冷层 |
| 不可审计 | 检索过程只有相似度，没有日志 | 出错时无法追溯 | 记录检索日志、引用链路、版本号 |

对于新手来说，有两个错误特别高频。

### 错误一：把动态信息塞进公共前缀

例如把下面这些内容都拼进 system prompt：

- `CurrentUser=张三`
- 当前时间
- 当前 trace id
- 每次查询得到的临时检索结果

这样做的结果是每次前缀都不同，`KV cache` 基本没有复用空间。更合理的拆法是：

| 放前缀 | 放后缀 / 动态输入 |
| --- | --- |
| 系统角色定义 | 当前用户 |
| 工具说明 | 当前时间 |
| 固定输出格式 | 当次检索结果 |
| 稳定安全规则 | 本次任务参数 |

原则很简单：越稳定的内容越应该靠前，越动态的内容越应该靠后。

### 错误二：默认“所有历史都做 embedding”

这个思路看似完整，实际往往最先把系统拖垮。原因有三点：

1. 写入量大时，embedding 队列会堆积
2. 检索时索引变大，延迟上升
3. 很多闲聊内容本来就没有长期价值

更合理的策略通常是按事件价值分层：

| 事件类型 | 是否进长期记忆 | 理由 |
| --- | --- | --- |
| 用户明确偏好 | 是 | 后续高概率复用 |
| 关键业务状态变更 | 是 | 需要追踪和审计 |
| 可复用知识摘要 | 是 | 有检索价值 |
| 一次性寒暄 | 否或短 TTL | 价值低，噪声高 |
| 中间推理草稿 | 通常否 | 噪声大，稳定性差 |

### 错误三：把“检索到了”当成“可以直接喂给模型”

检索结果通常只是候选，不是最终上下文。生产里往往还需要：

- 去重
- 脱敏
- 时间排序
- 权限过滤
- token 预算裁剪
- 引用来源保留

否则模型看到的上下文要么太长，要么掺杂无关信息，要么包含不该暴露的数据。

---

## 替代方案与适用边界

如果你的场景以“重复问题多、知识库稳定、在线响应要求高”为主，统一栈往往比自己拼多套系统更简单。Redis 在 2026 年的 Agent 组件文章里强调的重点，不是某个单独算法，而是把语义缓存、向量检索、状态和限流能力放到更少的系统边界内，减少网络跳数和跨系统协调成本。对于企业 FAQ、客服 Agent、内部知识问答，这类方案通常很实用。

如果你的场景是“固定系统提示词很长，而且会被大量用户复用”，那么 NVIDIA TensorRT-LLM 的 `KV cache early reuse` 会更有吸引力。它优化的是高并发下共享前缀的 TTFT，而不是长期记忆本身。NVIDIA 给出的数据是最多可达 `5x` 的 TTFT 加速，并且更细粒度的 block size 还能带来额外收益。它适合 GPU 推理服务已经标准化、前缀高度稳定的环境。

把常见方案压缩成一张对照表，可以更容易看清取舍：

| 方案 | 优势 | 适用场景 | 限制 |
| --- | --- | --- | --- |
| `KV cache` 复用 | 重复前缀极快，显著降低 TTFT | 固定 system prompt、长模板、共享文档前缀 | 不能替代长期存储，前缀变动会失效 |
| 向量数据库 | 语义召回灵活，适合历史经验回忆 | 历史对话、知识片段、案例库 | 权限和时间控制弱，规模大时延迟明显 |
| 结构化数据库 | 精确过滤、易审计、事务能力强 | 用户资料、订单、事件日志、权限系统 | 不擅长模糊语义召回 |
| Redis 统一栈 | 状态、缓存、检索放在更少系统里 | 企业问答、在线 Agent 平台 | 容量规划和 SLA 设计要求更高 |
| TensorRT-LLM KV reuse | 固定前缀场景下极致优化 TTFT | 高并发 GPU 推理服务 | 依赖 NVIDIA 生态，不解决长期记忆归档 |

还可以再从“系统复杂度”和“业务要求”两个维度看边界。

### 什么时候简单方案就够了

如果满足下面条件，通常不需要一开始就做复杂分层：

- 单租户
- 数据量不大
- 权限模型简单
- 主要是 FAQ 或知识问答
- 对审计要求不高

这时可以先用“结构化元数据 + 简单向量检索 + 少量缓存”的轻量架构。

### 什么时候必须做严格分层

如果系统出现下面特征，分层就基本不是优化项，而是必需项：

- 多租户隔离
- 金融、医疗、企业内部系统等合规场景
- 历史数据增长很快
- 高并发且对 TTFT 敏感
- 需要明确知道“答案引用了哪些记忆”

最终的适用边界可以收敛成三句判断：

- 只要需求是“跨会话还记得”，就必须有外部持久化
- 只要需求是“低延迟”，就不能默认每次都从冷层全量检索
- 只要需求是“权限正确”，就不能只靠向量相似度

所以，长期记忆真正的问题不是“要不要 `KV cache`”，而是：

1. 哪些信息应留在热层
2. 哪些信息应下沉到持久化层
3. 什么时候优先复用前缀，什么时候回退到外部检索
4. 检索结果怎样经过结构化约束和审计后再送给模型

---

## 参考资料

- Glean, *How KV caches impact time to first token for LLMs*  
  https://www.glean.com/blog/glean-kv-caches-llm-latency
- Sparkco, *Agent Context Windows in 2026: How to Stop Your AI from Forgetting Everything*  
  https://sparkco.ai/blog/agent-context-windows-in-2026-how-to-stop-your-ai-from-forgetting-everything
- Cloudidr, *The Memory Architecture of AI: From Context Windows to Infinite Agent Memory*  
  https://www.cloudidr.com/blog/ai-memory-architecture
- Redis, *Agentic AI System Components: Building Production-Ready Agents*  
  https://redis.io/blog/agentic-ai-system-components/
- NVIDIA, *5x Faster Time to First Token with NVIDIA TensorRT-LLM KV Cache Early Reuse*  
  https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/
- NVIDIA Docs, *KV Cache Reuse (a.k.a. prefix caching)*  
  https://docs.nvidia.com/nim/large-language-models/1.2.0/kv-cache-reuse.html
- Medium, CortexFlow, *Long-Term Memory for AI Agents*  
  https://medium.com/the-software-frontier/long-term-memory-for-ai-agents-1d93516c08ae
- Medium, Ryjoxtechnologies, *Why Agent Memory Speed Matters: Sub-Millisecond Retrieval Enables Adaptive Reasoning*  
  https://medium.com/%40ryjoxtechnologies/why-agent-memory-speed-matters-sub-millisecond-retrieval-enables-adaptive-reasoning-40714bdbc48a
