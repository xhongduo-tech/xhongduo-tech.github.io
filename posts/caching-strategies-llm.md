## 核心结论

缓存策略在大模型服务里不是一个点，而是两层协同：

| 机制 | 白话解释 | 直接作用 | 典型存储位置 |
| --- | --- | --- | --- |
| 请求缓存 | 把“同样的问题”对应的“同样结果”先存起来 | 避免重复推理，直接返回结果 | 进程内存、Redis |
| 模型缓存 | 把模型运行时要反复用到的中间状态先留住 | 减少模型加载、前缀计算、上下文重建时间 | GPU 显存、CPU 内存、Redis |

结论只有两条。

第一，请求缓存和模型缓存解决的是两个不同瓶颈。请求缓存处理“重复输入”，模型缓存处理“重复计算准备”。前者命中时甚至不用调用模型，后者即使请求没命中，也能让推理更快开始。

第二，缓存不是“开了就快”，而是“命中率高且失效策略正确才快”。命中率、TTL、容量上限、淘汰策略这四个指标决定缓存是否真的在降低延迟，而不是单纯占内存。

一个最小直觉例子是客服系统。用户反复提“账单问题”，服务端先用问题文本生成缓存键，到 Redis 查询。命中后直接返回答案，不再调用 LLM。这就是请求缓存。若没有命中，但模型权重已经常驻显存、相同系统提示词的前缀 KV Cache 已经保留，那么这次 miss 仍会比“完全冷启动”更快，这就是模型缓存。

---

## 问题定义与边界

先把两个概念分清。

请求缓存，指把“输入到输出”的映射缓存下来。术语里的“映射”可以理解成“一一对应记录”。如果输入完全一致，系统就复用之前的输出。它主要优化重复问答、FAQ、固定模板生成、批量同题摘要这类场景。

模型缓存，指把模型运行中高成本、可复用的部分缓存下来。这里的“KV Cache”是 Transformer 解码时保存历史 token 计算结果的缓冲区，可以理解成“前文已经算过的注意力中间结果，不必再算一遍”。模型缓存常见对象包括模型权重、tokenization 结果、embedding、公共前缀的 KV Cache。

问题的数学刻画很简单。设命中率为 $h$，命中耗时为 $T_{\text{hit}}$，未命中耗时为 $T_{\text{miss}}$，则平均请求时间为：

$$
T_{\text{avg}}=(1-h)T_{\text{miss}}+hT_{\text{hit}}
$$

这说明缓存价值不是“有没有缓存”，而是“命中率能不能高到足以拉低平均值”。

边界也要明确：

| 场景 | 请求缓存效果 | 模型缓存效果 | 原因 |
| --- | --- | --- | --- |
| FAQ、固定问答 | 高 | 中 | 输入重复度高 |
| 带固定系统提示词的对话 | 中 | 高 | 公共前缀可复用 |
| 每次都不同的长文写作 | 低 | 中 | 输入高变异，请求缓存几乎失效 |
| 高频知识库检索问答 | 中 | 高 | 检索结果和前缀结构常重复 |
| 实时行情、秒级变化数据 | 低 | 低到中 | 内容变化快，缓存寿命很短 |

玩具例子：常见 10 个 FAQ，比如“怎么开发票”“如何修改密码”。这些问题答案固定，请求缓存非常有效。

真实工程例子：企业内部知识库问答。用户问法不完全一样，但系统提示词、检索模板、回答骨架高度一致，这时请求缓存命中未必高，但 embedding 缓存、检索结果缓存、公共前缀 KV Cache 仍然能明显降低成本。

---

## 核心机制与推导

先看请求缓存的流程：

`用户请求 -> 规范化输入 -> 生成缓存键 -> 查缓存 -> hit 直接返回 / miss 调模型并回写缓存`

再看模型缓存的流程：

`用户请求 -> 模型已预热 -> 复用权重/前缀KV/embedding -> 补充剩余计算 -> 返回结果`

两层缓存的关键差异是命中位置不同。

请求缓存的命中发生在“模型调用之前”。一旦 hit，推理成本接近 0，只剩序列化、网络、缓存读写的成本。

模型缓存的命中发生在“模型执行内部”。即使最终还要生成新 token，也可能省掉模型加载、公共前缀重算、向量重复编码等步骤。

下面用公式看收益。假设某接口：

- miss 时延 $T_{\text{miss}}=27\text{ms}$
- hit 时延 $T_{\text{hit}}=1\text{ms}$

当命中率为 75% 时：

$$
T_{\text{avg}}=0.25\times27+0.75\times1=7.5\text{ms}
$$

当命中率提升到 90% 时：

$$
T_{\text{avg}}=0.1\times27+0.9\times1=3.6\text{ms}
$$

命中率只提高了 15 个百分点，平均时延却下降了 52%。原因很直接：缓存命中的代价远小于 miss，所以 $h$ 对最终结果有乘数效应。

这里有一个容易忽略的推导点。很多人以为“缓存命中率 90% 才有价值”。其实不对。只要满足：

$$
T_{\text{hit}} \ll T_{\text{miss}}
$$

即使命中率只有 30% 到 50%，也可能有明显收益。真正要避免的是另一类情况：缓存键设计很差，导致读写开销接近模型调用本身，那缓存就失去了意义。

玩具例子：  
用户连续 100 次问“账单什么时候出”。假设其中 90 次完全相同。请求缓存会让这 90 次直接返回；剩下 10 次才走模型。这个时候系统吞吐量会随命中率几乎线性上升，因为 GPU 被释放出来处理真正的新请求。

真实工程例子：  
一个多租户 RAG 系统使用固定系统提示词，例如“你是企业知识库助手，请严格依据检索结果回答”。不同用户提问不同，但前 200 个 token 的系统提示和格式模板几乎相同。如果服务端支持前缀 KV Cache 复用，那么即使请求缓存 miss，模型也不必对这 200 个 token 重新构建完整注意力状态，首 token 延迟会下降。

因此，推理服务通常不是只做一种缓存，而是分层做：

1. 最外层做请求缓存，直接拦截重复输入。
2. 中间层做检索结果、embedding 缓存，减少外部依赖开销。
3. 模型层做权重常驻和 KV Cache 复用，缩短推理准备时间。

---

## 代码实现

下面先给一个最小可运行的 Python 示例，模拟请求缓存的核心逻辑。这个例子不依赖 Redis，但接口形式与 Redis 很接近，便于迁移到真实工程。

```python
import hashlib
import time


class SimpleTTLCache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        item = self.store.get(key)
        if not item:
            return None
        value, expire_at = item
        if expire_at is not None and time.time() > expire_at:
            del self.store[key]
            return None
        return value

    def set(self, key, value, ttl_seconds=None):
        expire_at = None if ttl_seconds is None else time.time() + ttl_seconds
        self.store[key] = (value, expire_at)


def normalize_input(text: str) -> str:
    # 统一空白与大小写，避免“同义格式差异”导致缓存 miss
    return " ".join(text.strip().lower().split())


def cache_key(model_name: str, temperature: float, prompt: str) -> str:
    normalized = normalize_input(prompt)
    raw = f"{model_name}|{temperature}|{normalized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def fake_llm_generate(prompt: str) -> str:
    # 模拟高成本推理
    time.sleep(0.01)
    if "账单" in prompt:
        return "账单通常在每月最后一天生成。"
    return f"answer:{prompt}"


def answer_with_cache(cache, model_name, temperature, prompt, ttl_seconds=60):
    key = cache_key(model_name, temperature, prompt)
    cached = cache.get(key)
    if cached is not None:
        return {"source": "cache", "answer": cached}

    result = fake_llm_generate(prompt)
    cache.set(key, result, ttl_seconds=ttl_seconds)
    return {"source": "model", "answer": result}


cache = SimpleTTLCache()

r1 = answer_with_cache(cache, "gpt-demo", 0.0, "账单问题")
r2 = answer_with_cache(cache, "gpt-demo", 0.0, "  账单问题  ")

assert r1["source"] == "model"
assert r2["source"] == "cache"
assert r1["answer"] == r2["answer"] == "账单通常在每月最后一天生成。"
```

这个例子体现了三个关键点。

第一，缓存键不能只用原始文本，至少要带上模型名、温度等影响输出的参数。否则同样的问题在不同模型配置下会错误复用。

第二，输入要先规范化。这里的“规范化”就是把不影响语义的格式差异消掉，例如多余空格、大小写差异。否则用户输入“账单问题”和“ 账单问题 ”会变成两个键，命中率被白白浪费。

第三，TTL 必须显式设置。TTL 是“生存时间”，也就是条目自动失效的时间。对静态 FAQ 可以设得长一些，对时效性强的数据必须设得短。

如果切到 Redis，逻辑通常就是：

```python
import hashlib
import json
import redis

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def make_key(model_name, temperature, prompt):
    raw = f"{model_name}|{temperature}|{' '.join(prompt.strip().lower().split())}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def get_answer(prompt, model_name="gpt-demo", temperature=0.0, ttl=86400):
    key = make_key(model_name, temperature, prompt)

    cached = r.get(key)
    if cached:
        return json.loads(cached)

    # 真实场景这里调用 LLM
    result = {"answer": f"generated:{prompt}", "source": "model"}
    r.set(key, json.dumps(result, ensure_ascii=False), ex=ttl)
    return result
```

模型缓存的实现会更分层。典型做法如下：

| 层级 | 缓存对象 | 典型做法 |
| --- | --- | --- |
| 进程启动层 | 模型权重 | 服务启动时预热，保持常驻 |
| 请求前处理层 | tokenizer / embedding | 对重复文本缓存编码结果 |
| 推理层 | 前缀 KV Cache | 对公共 system prompt 或模板复用 |
| 外围依赖层 | 检索结果、FAQ 答案 | 放入 Redis，跨实例共享 |

真实工程例子可以这样落地。  
一个账单客服系统有 8 个应用实例，前面挂负载均衡。如果只用进程内存缓存，每个实例都要自己积累热点，命中率会被流量打散。把 FAQ 请求缓存放到 Redis 后，某个实例刚算出的“账单怎么查询”结果，其他实例也能直接复用，这就是分布式缓存的价值。

---

## 工程权衡与常见坑

缓存最常见的问题不是“不会写”，而是“写完后指标变差但没发现”。

| 坑 | 现象 | 规避措施 |
| --- | --- | --- |
| 漏设 TTL | 缓存越积越多，占满内存 | 为所有缓存条目设置默认 TTL，按数据时效分层 |
| 键设计不一致 | 明明是同一个问题却总 miss | 统一规范化、序列化和哈希规则 |
| 高基数 key | 每个请求都是新键，命中率极低 | 限制参与键的字段，只保留真正影响结果的参数 |
| 错误复用 | 不同模型配置返回了同一缓存 | 把模型名、版本、温度、系统提示词纳入 key |
| 只看命中率，不看节省成本 | hit 很高但收益不明显 | 同时监控 hit latency、miss latency、token 节省量 |
| Redis 被冷数据塞满 | 热点数据被挤掉，命中率反降 | 配置 LRU/LFU 和容量上限 |
| 缓存雪崩 | 大量 key 同时过期，请求瞬时打爆后端 | 给 TTL 加随机抖动，错开过期时间 |

先解释三个淘汰策略。

LRU，Least Recently Used，最近最少使用。白话是“很久没用过的先淘汰”。  
LFU，Least Frequently Used，最不常用。白话是“总共用得少的先淘汰”。  
TTL，Time To Live，生存时间。白话是“到点就过期”。

这三者没有谁绝对更强，而是解决不同问题：

- TTL 控制数据是否过时。
- LRU 控制近期热点是否保留。
- LFU 控制长期高频项是否保留。

对 FAQ、模板回复这类长期热点，LFU 往往比 LRU 更稳。因为某些高价值条目即使某一小时没被访问，也不该轻易淘汰。对波峰波谷明显的业务，LRU 更适合，因为它能快速适应热点迁移。

玩具例子：  
FAQ 缓存 TTL 设为 3 天，淘汰策略选 LFU。结果“账单问题”“发票申请”“修改手机号”这些高频问答会长期保留，命中率稳定。

真实工程例子：  
某团队上线缓存后只监控了 Redis QPS，没有监控 eviction 和 keyspace hit ratio。两周后发现平均延迟反而升高。排查结果是用户自由输入生成了大量高基数 key，冷数据塞满 Redis，热点 FAQ 反而被挤掉。修复方式不是“加更大 Redis”，而是先收敛 key 设计，再按业务区分 TTL，并切换到 LFU。

还有一个非常典型的坑是“缓存不考虑版本”。如果系统提示词、模型版本、知识库索引版本变化了，旧缓存可能逻辑上已经失效。工程上通常会给 key 加版本前缀，例如：

`v3:model-name:hash(prompt)`

这样一旦提示词模板升级，只需切版本号，不必逐条删 key。

---

## 替代方案与适用边界

不是所有场景都适合做严格的请求缓存。输入高度动态时，请求缓存会迅速退化。

这时可以考虑近似缓存，也叫 semantic cache。它不是要求“文本完全相同”，而是要求“语义足够接近”。所谓“语义接近”，就是用 embedding 向量衡量两个问题是否表达同一件事。如果相似度超过阈值，就复用之前答案，或者至少复用检索结果。

但近似缓存有边界。它适合 FAQ、知识库问答、不追求逐字精确一致的场景；不适合法律条款、金融报价、库存价格这类对时效和精确性极敏感的请求。

| 场景 | 适合缓存层 | 说明 |
| --- | --- | --- |
| 固定 FAQ | 请求缓存 | 完全匹配即可，收益最高 |
| 多轮对话固定前缀 | 模型缓存 | 公共 system prompt 和前缀可复用 |
| 知识库问答 | 请求缓存 + semantic cache + 检索缓存 | 问法不同但语义重复 |
| 实时行情查询 | 短 TTL 请求缓存或不缓存 | 数据更新快，过期风险高 |
| Serverless 推理 | 预热 + 模型常驻优先 | 容器冷启动比请求缓存更关键 |
| 多租户平台 | 分布式请求缓存 + 分租户隔离 | 防止租户间污染与误命中 |

玩具例子：  
“如何开发票”和“发票怎么开”文本不同，但语义接近。严格请求缓存会 miss，semantic cache 可能 hit。

真实工程例子：  
金融资讯问答系统里，“今天苹果股价多少”这种问题时效极强。这里不应把请求缓存设成几小时，否则用户会拿到旧结果。更合理的做法是只缓存 embedding、检索结果或最近几分钟的数据，并给 TTL 设很短。超过窗口就必须回源。

还有一种容易被忽略的替代方案是“先做预热，不急着做复杂缓存”。如果你的瓶颈主要来自模型冷启动，而不是重复请求，那么把模型权重常驻、把常用 LoRA 提前加载、把服务实例维持 warm state，往往比设计复杂的请求缓存更直接。尤其在 Serverless 或弹性缩容频繁的环境里，冷启动成本可能比缓存 miss 更致命。

---

## 参考资料

1. Redis 关于语义缓存优化的文章，适合理解请求缓存、embedding 缓存和命中率优化思路。  
2. Hakia 的缓存策略说明，适合建立 TTL、LRU、LFU 与容量治理的工程直觉。  
3. Redis Enterprise 关于大规模缓存的资料，给出了平均请求时间公式与典型缓存分层方法。  
4. Redis 关于缓存优化策略的综述，适合整理容量规划、淘汰策略和监控指标。  
5. 若工程里使用 RAG 或知识库检索，建议继续查阅所用推理框架对 prefix cache、KV cache、paged attention 的官方文档，因为不同推理引擎的缓存粒度和失效条件差异很大。
