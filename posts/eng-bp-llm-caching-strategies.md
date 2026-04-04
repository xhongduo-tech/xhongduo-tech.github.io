## 核心结论

LLM 应用里的“缓存”不是单一技术，而是一条按风险和成本递增的复用链路：先查精确缓存，再查语义缓存，再吃提供方的前缀缓存，最后才真正调用模型。这样设计的原因很直接：完全相同的请求，没必要再付一次推理费；语义相近且答案稳定的请求，也常常可以复用已有结果。

在生产环境里，多层缓存通常能带来 30% 到 50% 的推理成本下降，同时把命中请求的延迟压到实时推理的十分之一甚至百分之一量级。它成立的前提不是“模型很聪明”，而是“业务里确实存在重复、近重复、稳定前缀和稳定答案”。客服 FAQ、表单抽取、分类路由、知识库问答，都属于这类场景。

但缓存不是“命中越高越好”。缓存系统真正要保证的是：便宜地复用正确答案，而不是便宜地复用错误答案。对 LLM 来说，错误通常来自三类问题：内容过期、语义误匹配、模型或数据版本漂移。所以一个合格的缓存策略必须把 `版本号`、`TTL` 和 `相似度阈值` 放进同一套设计里。

下面这张表可以先建立整体直觉：

| 缓存层级 | 命中条件 | 典型实现 | 成本 | 主要风险 |
| --- | --- | --- | --- | --- |
| 精确缓存 | 输入完全一致 | `hash(prompt+params+version)` 查 Redis | 最低 | 版本漏绑导致旧答案继续命中 |
| 语义缓存 | 语义相近且通过校验 | 向量检索 + 相似度阈值 + 实体验证 | 低到中 | 不同实体被误判为同一问题 |
| 前缀缓存 | 长前缀一致 | 模型提供方 KV/prefix cache | 中 | prompt 前缀不稳定，命中率下降 |
| 推理层/KV 缓存 | 会话上下文可复用 | GPU KV cache / paged KV | 中到高 | 显存打满，延迟断崖式上升 |

一个新手最容易理解的玩具例子是客服 FAQ。假设 100 个请求里有 25 个都在问“退款多久到账”，并且 system prompt、模型参数、知识库版本都相同，那么这 25 个请求只要第一次跑过模型，后面 24 次都可以直接命中精确缓存。这里不需要“理解语义”，只需要校验请求哈希相同即可。命中后延迟会从秒级掉到毫秒级，这就是缓存价值最直观的来源。

---

## 问题定义与边界

先定义问题。LLM 推理贵，贵在两件事：一是 token 计算本身有成本，二是请求多了以后，显存、带宽、并发调度都会变成瓶颈。如果应用里有大量重复请求，却每次都重新走完整推理，相当于主动烧钱。

因此，缓存策略要回答三个边界问题：

| 问题 | 需要回答什么 | 如果没定义清楚会怎样 |
| --- | --- | --- |
| 哪些请求可缓存 | 是否稳定、是否重复、是否允许复用 | 把创作类任务错误缓存，体验下降 |
| 缓存多久 | 可接受多长时间的陈旧窗口 | 旧政策、旧价格、旧库存继续被返回 |
| 如何判定“相同” | 完全相同还是语义相近 | 相似问题被错配到错误答案 |

一个典型边界案例是 HR 机器人。系统原来回答“年假是 15 天”，后来政策改成“20 天”。如果缓存 key 只绑定 prompt，不绑定 `policy_version`，并且 TTL 设成 24 小时，那么在更新后的 24 小时内，旧答案都可能继续返回。这个问题不是模型能力不够，而是缓存边界定义错了。

可以把决策过程理解成下面这条规则：

1. 内容是否稳定。
2. 稳定多久。
3. 是否允许近似命中。
4. 命中后是否还要做附加校验。

如果是“今天上海天气如何”这种时间敏感问题，答案的可复用窗口很短，缓存收益很低，错误成本很高，就不该把语义缓存开得很激进。如果是“Python 的 list 和 tuple 区别”这种稳定知识，缓存窗口可以长很多。

所以，LLM 缓存的边界不是“能不能缓存”，而是“在什么新鲜度、风险和命中率约束下缓存才值得”。

---

## 核心机制与推导

缓存命中的判断，不是一句“像不像”就够了。生产上更接近下面这组联合条件：

$$
K = hash(system\_prompt + user\_prompt + params + data\_version)
$$

$$
TTL \le staleness\_window
$$

$$
semantic\_score(q, q_c) \ge threshold
$$

并且还要满足：

$$
entity(q) = entity(q_c)
$$

其中：

- `K` 是精确缓存键，意思是“把影响答案的全部输入压成一个唯一标识”。
- `TTL` 是缓存有效期，白话讲就是“这条答案最多能存多久”。
- `staleness_window` 是业务允许的最大陈旧窗口。
- `semantic_score` 是语义相似度，常见做法是余弦相似度。
- `entity` 是实体校验，意思是“问题里真正决定答案的对象是否一致”。

这几条条件组合起来，形成一条安全的降级链：

1. 先查精确缓存。完全一致，直接返回。
2. 精确未命中，再查语义缓存。只有相似度足够高并且实体一致才返回。
3. 再没有，就走前缀缓存或完整推理。
4. 推理结果写回缓存，供后续复用。

为什么语义缓存必须加“实体校验”？看一个玩具例子就明白：

- 问题 A：“电子产品退款期限是多少？”
- 问题 B：“生鲜商品退款期限是多少？”

这两个问题句式非常接近，向量相似度可能高到 0.94。但如果电子产品是 30 天、生鲜是 7 天，那么只靠“相似度大于 0.90”命中，就会把答案复用错。这里“电子产品”和“生鲜商品”就是实体，必须单独抽出来校验。

这也是 LLM 缓存和传统网页缓存的本质区别。传统缓存更关心 URL 是否相同；LLM 缓存还要处理“文本表面相近，但语义关键点不同”的问题。

从工程上看，多层缓存成立的核心逻辑是“让最便宜、最确定的判断先执行”。因为精确缓存查询最便宜、风险最低，所以永远放第一层；语义缓存会增加 embedding、向量检索和误命中风险，所以放第二层；真正的模型推理最贵，放最后。

---

## 代码实现

下面给一个最小可运行的 Python 例子，演示“精确缓存 + TTL + 版本绑定 + 简单实体验证”的骨架。这里不用外部依赖，重点是把机制讲清楚。

```python
import hashlib
import time
from dataclasses import dataclass

@dataclass
class CacheEntry:
    answer: str
    expire_at: float
    entity: str
    semantic_text: str

class LLMCache:
    def __init__(self):
        self.exact_store = {}

    def make_key(self, system_prompt, user_prompt, params, data_version):
        raw = f"{system_prompt}|{user_prompt}|{params}|{data_version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def put_exact(self, key, answer, ttl_seconds, entity, semantic_text):
        self.exact_store[key] = CacheEntry(
            answer=answer,
            expire_at=time.time() + ttl_seconds,
            entity=entity,
            semantic_text=semantic_text,
        )

    def get_exact(self, key):
        entry = self.exact_store.get(key)
        if not entry:
            return None
        if time.time() > entry.expire_at:
            del self.exact_store[key]
            return None
        return entry

def extract_entity(text: str) -> str:
    if "电子" in text:
        return "electronics"
    if "生鲜" in text:
        return "groceries"
    return "generic"

def fake_llm(user_prompt: str) -> str:
    if "电子" in user_prompt:
        return "电子产品退款期限为30天"
    if "生鲜" in user_prompt:
        return "生鲜商品退款期限为7天"
    return "请咨询人工客服"

def handle_request(cache, system_prompt, user_prompt, params, data_version, ttl_seconds=3600):
    key = cache.make_key(system_prompt, user_prompt, params, data_version)

    exact_hit = cache.get_exact(key)
    if exact_hit:
        return {"source": "exact_cache", "answer": exact_hit.answer}

    # 这里只演示精确缓存；真实系统可以在这里插入语义检索
    answer = fake_llm(user_prompt)
    cache.put_exact(
        key=key,
        answer=answer,
        ttl_seconds=ttl_seconds,
        entity=extract_entity(user_prompt),
        semantic_text=user_prompt,
    )
    return {"source": "llm", "answer": answer}

cache = LLMCache()
system_prompt = "你是退款政策助手"
params = {"temperature": 0}
data_version = "policy_v2"

r1 = handle_request(cache, system_prompt, "电子产品退款期限是多少？", params, data_version)
r2 = handle_request(cache, system_prompt, "电子产品退款期限是多少？", params, data_version)

assert r1["source"] == "llm"
assert r2["source"] == "exact_cache"
assert r2["answer"] == "电子产品退款期限为30天"

# 版本变化后，即使 prompt 相同，也不应命中旧缓存
r3 = handle_request(cache, system_prompt, "电子产品退款期限是多少？", params, "policy_v3")
assert r3["source"] == "llm"

print("all assertions passed")
```

上面这段代码体现了三个关键点：

1. key 不只包含用户问题，还包含 `system_prompt`、参数和 `data_version`。
2. 缓存过期依赖 TTL，而不是永久有效。
3. 版本变化后自动进入新的 key 空间，旧缓存虽然还在，但不会再被命中。

真实工程例子通常更复杂。比如一个知识库客服系统，完整链路可能是：

1. 计算精确 key，查 Redis。
2. 未命中时，对用户问题做 embedding，去向量库查近邻。
3. 若相似度高于阈值，再校验实体、租户、语言、文档版本。
4. 通过后返回语义缓存。
5. 否则走 RAG 检索和 LLM 推理。
6. 结果回写 Redis 与向量库。
7. 高频问题提前预热，冷启动前先写入缓存。

如果你只准备做第一版，建议先上线“精确缓存 + 版本绑定 + TTL + 请求合并”。请求合并的意思是多个相同请求同时到达时，只让一个请求真正调用模型，其他请求等待同一个结果。它的收益通常比一开始就做复杂语义缓存更稳定。

---

## 工程权衡与常见坑

最常见的坑，不是“没做缓存”，而是“做了缓存却没有定义风险约束”。

| 坑 | 后果 | 规避措施 |
| --- | --- | --- |
| 语义阈值太低 | 不同实体共用错误答案 | 提高阈值，加入实体级验证，仅在低风险域启用 |
| key 没绑版本 | 数据更新后继续返回旧答案 | 在 key 中加入 `data_version/model_version` |
| TTL 过长 | 旧价格、旧政策长时间存活 | TTL 不超过业务允许陈旧窗口 |
| prompt 前缀不稳定 | 前缀缓存几乎不命中 | 固定 system prompt，把变量放后面 |
| 只看命中率不看正确率 | 高命中掩盖质量下降 | 同时监控 hit rate、latency、quality |
| GPU KV 不分页 | 高并发下显存打满 | paged KV + LRU + 限流 + 请求合并 |

其中最危险的是“语义误匹配”。因为它不会像程序崩溃那样显眼，而是安静地返回一条看起来很合理、但其实是别的问题的答案。对于金融、医疗、法务、人事政策这类高风险领域，保守做法通常是只开精确缓存，不开语义缓存，或者语义缓存命中后仍然强制二次校验。

再看一个真实工程例子。聊天系统把会话历史的 KV cache 放在 GPU 显存里，本意是减少重复计算；但当并发达到 10,000、每个会话约 2,000 token 时，KV 数据可能吃掉 5 到 10GB 显存。一旦显存顶满，系统就要把一部分 KV 页换到主机内存，跨 PCIe 读回会很慢，p99 延迟可能从 800ms 直接跳到 5s。这个现象说明：模型层缓存也不是“越多越好”，而是必须和显存预算、并发模型、调度策略一起设计。

所以工程权衡的实质是：

- 更高命中率，通常意味着更宽松的匹配规则。
- 更宽松的匹配规则，通常意味着更高的错误风险。
- 更高的新鲜度，通常意味着更短 TTL。
- 更短 TTL，通常意味着更低命中率。

没有免费的最优点，只能按业务风险选一个可接受区间。

---

## 替代方案与适用边界

不是所有 LLM 应用都适合重缓存。判断是否值得做缓存，可以先看下面这张表：

| 场景 | 推荐策略 | 是否适合缓存 |
| --- | --- | --- |
| FAQ / 客服 / 政策问答 | 精确缓存 + 版本绑定，必要时加语义缓存 | 适合 |
| 结构化抽取 / 分类 | 精确缓存优先 | 适合 |
| RAG 问答且文档更新较慢 | 精确缓存 + 文档版本 | 较适合 |
| 实时行情 / 天气 / 库存 | 极短 TTL 或直接不缓存 | 谨慎 |
| 创意写作 / 头脑风暴 | 通常禁用答案缓存 | 不适合 |
| 高风险法务/医疗建议 | 只做非常保守的精确缓存 | 谨慎 |

创作类应用是最典型的反例。用户问“给我写一个有反转的科幻短篇开头”，即使输入完全一样，用户也往往期待不同表达。此时缓存会伤害“新颖性”，也就是产品价值本身。对这类场景，最多保留 system prompt 的前缀缓存，而不要缓存最终答案。

另一类边界是“命中率太低，不值得付复杂度”。如果你的请求空间高度离散，几乎每次都不一样，那么你为了做语义缓存而引入向量库、阈值调参、实体校验和误命中监控，最后省下来的推理费可能还不够覆盖工程成本。此时更合理的路径是：

1. 先做前缀稳定化，吃到模型提供方的 prefix cache。
2. 再做精确缓存，观察真实重复率。
3. 只有当重复率、相似请求比例和错误成本都合适时，再上语义缓存。

换句话说，缓存不是 LLM 系统的默认配置，而是一种有前提的成本优化手段。前提包括：请求有复用性、答案有稳定性、错误可控、监控到位。

---

## 参考资料

- System Overflow, “Failure Modes and Edge Cases in LLM Caching”  
  https://www.systemoverflow.com/learn/ml-llm-genai/llm-caching-optimization/failure-modes-and-edge-cases-in-llm-caching
- Viqus Blog, “LLM Caching Strategies: Cut Your Inference Bill Without Cutting Corners”  
  https://viqus.ai/blog/llm-caching-strategies-production
- OneUpTime, “How to Build LLM Caching Strategies”  
  https://oneuptime.com/blog/post/2026-01-30-llm-caching-strategies/view
