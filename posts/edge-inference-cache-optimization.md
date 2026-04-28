## 核心结论

边缘推理缓存优化，指的是**把可重复使用的推理前缀中间结果提前存起来，下次同类请求直接复用**。白话说，不是每次都从头把同一段提示词重新“读一遍模型”，而是把已经算过的前半段结果拿出来接着用。

它的目标不是把缓存做得尽可能大，而是在很小的内存预算下，用更高的命中率换更低的首 token 延迟。对大语言模型来说，请求通常分成两段：前缀预填充和后续解码。前者要把输入 token 逐步送进模型，成本高；后者是逐 token 生成，成本相对稳定。于是只要前缀能复用，平均时延就能明显下降。一个常用写法是：

$$
L = T_{lookup} + (1-p)T_{pre} + T_{dec}
$$

其中，$L$ 是平均时延，$p$ 是命中率，$T_{lookup}$ 是查找与拷贝缓存的成本，$T_{pre}$ 是前缀预填充成本，$T_{dec}$ 是解码成本。只有当 $pT_{pre} > T_{lookup}$ 时，缓存才真正节省时间。

下面这张表先给出最实用的判断：

| 场景 | 是否适合缓存 | 原因 |
|---|---|---|
| 固定 system prompt | 适合 | 前缀重复率高 |
| 热门 FAQ | 适合 | 问题模板稳定，命中率高 |
| 表单生成、固定工作流模板 | 适合 | 指令前缀一致 |
| 每次都不同的自由问答 | 不太适合 | 可复用部分少 |
| 高频改模板的实验环境 | 不太适合 | key 失效快 |
| 多租户敏感数据混跑且隔离弱 | 谨慎 | 有串租户风险 |

玩具例子很直观。门店里的边缘盒子每天回答“退货规则”“营业时间”“会员积分”这几类问题，品牌规则和 FAQ 说明几乎不变。第一次请求时把这段公共前缀算出来并缓存，后面同类请求就直接复用，只对用户最后几句话继续推理。真正有价值的缓存，基本都长在这种“固定模板 + 高频重复”场景里。

真实工程里更明确：同一 `model_id` 下，若固定模板前缀命中，就可以跳过大部分预填充阶段，只做 `lookup + copy + decode`。这时收益来自少算了一大段前缀，而不是少传了一段文本。边缘缓存优化本质上优化的是**模型内部计算路径**，不是字符串处理速度。

---

## 问题定义与边界

先把“缓存什么”说清楚。边缘推理缓存通常缓存的不是原始文本，而是以下几类对象：

| 缓存对象 | 白话解释 | 典型用途 | 主要约束 |
|---|---|---|---|
| tokenized 前缀 | 文本分词后的 token 序列 | 快速判断是否同前缀 | 依赖 tokenizer 版本 |
| KV cache block | Transformer 每层注意力要复用的键值状态 | 跳过重复预填充 | 占内存大 |
| 可复用中间状态 | 某些框架内部保存的中间表示 | 针对特定实现优化 | 兼容性差 |

KV cache，白话说就是“模型已经读过前文后留下的记忆快照”。后面的 token 生成会反复依赖它，所以一旦前缀固定，KV 就有复用价值。

边缘场景的边界比云端硬得多，因为资源更紧：

| 维度 | 边缘设备常见约束 |
|---|---|
| 内存预算 | 只有几 GB 到十几 GB，可用空间非常紧 |
| 功耗 | 长时间满载会触发功耗限制 |
| 温度 | 温度升高可能降频，延迟波动加剧 |
| 带宽 | 本地磁盘、板载内存、总线带宽都有限 |
| 模型驻留 | 模型权重本身已占掉大头 |
| 多租户隔离 | 错误共享缓存可能造成数据泄露 |

还要区分什么能缓存、什么不能缓存、什么必须隔离：

| 类别 | 结论 | 说明 |
|---|---|---|
| 固定 system prompt | 能缓存 | 典型公共前缀 |
| 稳定模板 | 能缓存 | 如客服、工单、总结模板 |
| 用户私有上下文 | 需隔离后缓存 | 不能跨用户共享 |
| 实时变化文档 | 谨慎缓存 | 版本漂移会导致错用 |
| 原始自然语言全文 | 不能直接当结果缓存 | 文本相同不代表计算路径相同 |
| 跨模型/跨 tokenizer 复用 | 不能 | 中间状态不兼容 |

必要符号也统一一下：

| 符号 | 含义 |
|---|---|
| $p$ | 缓存命中率 |
| $T_{pre}$ | 前缀预填充耗时 |
| $T_{lookup}$ | 缓存查找与拷贝耗时 |
| $T_{dec}$ | 解码耗时 |
| $B$ | 可用缓存总预算 |
| $m_i$ | 第 $i$ 个缓存条目的内存占用 |

一个容易被忽视的边界是：**同一段文本，只要 tokenizer、模板、模型版本变了，旧缓存就可能完全失效**。因此缓存 key 不能只看原始文本，更不能只看“这串字长得一样”。

---

## 核心机制与推导

核心机制其实只有两步：先判断值不值得缓存，再决定缓存谁。

先看值不值得。平均时延公式是：

$$
L = T_{lookup} + (1-p)T_{pre} + T_{dec}
$$

不做缓存时，可近似写成：

$$
L_0 = T_{pre} + T_{dec}
$$

两者相减得到平均收益：

$$
\Delta = L_0 - L = pT_{pre} - T_{lookup}
$$

所以只有当：

$$
pT_{pre} > T_{lookup}
$$

缓存才带来正收益。这个结论很重要，因为它说明命中率再高，如果查找和搬运成本也很高，收益也可能被吃掉。

看一个最小数值例子。假设前缀长度是 `600` token，预填充成本是 `0.3 ms/token`，那么：

$$
T_{pre} = 600 \times 0.3 = 180 ms
$$

如果命中率 $p=0.7$，查找加拷贝成本 $T_{lookup}=12ms$，则平均收益：

$$
\Delta = 0.7 \times 180 - 12 = 114ms
$$

这说明每个请求平均能省约 `114 ms`。

如果命中率掉到 $p=0.05$，则：

$$
\Delta = 0.05 \times 180 - 12 = -3ms
$$

这时缓存反而拖慢系统。也就是说，“有缓存”不等于“有收益”。

再看容量约束。边缘设备不能无限存，通常满足：

$$
\sum_i m_i \le B
$$

当缓存空间固定时，应该优先保留“高复用、低体积、低拷贝成本”的条目。工程上常写成一个排序分数，例如：

$$
score_i = \frac{reuse_i \times prefill\_saved_i}{size_i + copy\_cost_i}
$$

分数高的优先留下。白话说，就是“省时多、占地少”的优先。

玩具例子可以类比成桌面资料管理。你经常查 3 页公式表，把它们放手边最值；如果把整本书、所有附件和偶尔才用一次的表都铺满桌子，真正常用的内容反而不容易拿到。这就是为什么缓存不是越大越好，而是越“准”越好。

真实工程例子更典型。一个本地知识助手部署在零售门店边缘服务器上，`model_id` 固定，system prompt 固定，FAQ 模板固定，只有用户最后一句不同。此时请求前 400 到 800 个 token 往往高度重复，命中后可以直接复用对应 KV block，TTFT（首 token 时间，白话说就是“用户第一次看到输出要等多久”）会明显下降。如果后来业务把模板从“门店客服模板 v3”切到“营销导购模板 v4”，即便文本差异不大，模板 hash 变了，旧缓存也应该视为不可复用。

---

## 代码实现

实现上至少要拆成五个模块：`key 构造`、`lookup`、`verify`、`insert`、`evict`。这样才能把版本隔离、命中判断、失败回退和内存上限控制清楚。

先看一个简化实现。它不是完整推理框架，但能把核心策略跑通。

```python
from collections import OrderedDict
import hashlib

class PrefixCache:
    def __init__(self, budget_bytes: int):
        self.budget_bytes = budget_bytes
        self.used_bytes = 0
        self.store = OrderedDict()  # key -> {"size": int, "payload": object, "tenant": str}

    def make_key(self, model_id, tokenizer_hash, template_hash, prefix_tokens):
        raw = f"{model_id}|{tokenizer_hash}|{template_hash}|{' '.join(map(str, prefix_tokens))}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def lookup(self, key, tenant):
        item = self.store.get(key)
        if item is None:
            return None
        assert item["tenant"] == tenant, "cross-tenant reuse is forbidden"
        self.store.move_to_end(key)
        return item["payload"]

    def insert(self, key, payload, size, tenant):
        if size > self.budget_bytes:
            return False
        while self.used_bytes + size > self.budget_bytes and self.store:
            old_key, old_item = self.store.popitem(last=False)
            self.used_bytes -= old_item["size"]
        self.store[key] = {"payload": payload, "size": size, "tenant": tenant}
        self.used_bytes += size
        return True

# toy example
cache = PrefixCache(budget_bytes=100)
key = cache.make_key("qwen2.5-7b", "tok_v1", "tmpl_v3", [1, 2, 3, 4])
assert cache.lookup(key, "tenant_a") is None

ok = cache.insert(key, payload={"kv_blocks": 4}, size=40, tenant="tenant_a")
assert ok is True
assert cache.lookup(key, "tenant_a") == {"kv_blocks": 4}

key2 = cache.make_key("qwen2.5-7b", "tok_v1", "tmpl_v4", [1, 2, 3, 4])
assert key != key2  # template change must invalidate reuse
assert cache.lookup(key2, "tenant_a") is None
```

这个 key 设计里，`model_id`、`tokenizer_hash`、`template_hash` 必须放进去，原因分别是：

| 字段 | 为什么必须进 key |
|---|---|
| `model_id` | 不同模型内部状态不兼容 |
| `tokenizer_hash` | 分词结果一变，token 序列就变 |
| `template_hash` | 模板空格、占位符顺序变化都可能影响前缀 |
| `prefix_tokens` | 真正决定是否同前缀的直接内容 |
| `tenant_id` 或隔离域 | 防止跨用户、跨租户误复用 |

请求流程可以压成下面的逻辑：

1. 接收请求，抽取公共前缀。
2. 用 `model_id + tokenizer_hash + template_hash + prefix_tokens` 生成 key。
3. 查缓存，若命中则先校验租户隔离、版本匹配、条目未过期。
4. 校验通过则复用 KV block，直接进入 decode。
5. 若未命中或校验失败，则正常做 prefill。
6. 若该前缀复用价值高，则插入缓存。
7. 若超预算，按 `LRU`、`TTL` 或评分淘汰。

缓存层级也要设计清楚：

| 缓存层级 | 存储介质 | 适合内容 | 常见淘汰策略 |
|---|---|---|---|
| 进程内内存 | RAM / VRAM | 高频短前缀、热 KV block | LRU |
| 本地磁盘映射 | SSD | 次热条目、可恢复元数据 | TTL + 大小上限 |
| 跨进程共享层 | 共享内存或专门服务 | 多 worker 公共前缀 | 评分淘汰 |

真实工程里通常不会只写一个字典，而会附带统计模块，至少记录 `hit rate`、`TTFT`、`p95`、内存占用和淘汰次数。否则你只知道“系统里有缓存”，却不知道“缓存到底有没有帮上忙”。

---

## 工程权衡与常见坑

边缘推理缓存最容易犯的错，是把它当成“默认总是有益”的优化。实际上它是高约束优化，必须算账。

先看常见坑：

| 常见坑 | 问题 | 规避措施 |
|---|---|---|
| 只按原始文本做 key | 文本相同但 tokenizer 或模板不同，可能错复用 | key 加入模型、tokenizer、模板版本 |
| 只看平均延迟 | 平均值好看但 `p95` 变差 | 同时监控 `TTFT`、`p95`、抖动 |
| 缓存过大 | 挤占模型运行内存，导致 OOM 或降频 | 给缓存设置硬预算 |
| 跨用户共享 | 可能泄露私有上下文 | 默认按租户隔离 |
| 低复用硬上缓存 | 命中率低，查找成本白付 | 先测重复率再启用 |
| 忽略失效策略 | 模板更新后还命中旧条目 | 加版本号和 TTL |

收益和风险通常不是同向增长的：

| 选择 | 可能收益 | 主要风险 |
|---|---|---|
| 扩大缓存容量 | 提高热前缀保留率 | 挤占模型内存 |
| 放宽共享范围 | 提高命中率 | 安全隔离变差 |
| 降低校验成本 | 降低 `T_lookup` | 错用缓存概率上升 |
| 只保留超热点前缀 | 稳定收益 | 长尾请求无帮助 |

有一个真实失败模式很常见。测试环境里只有一套模板、几类固定问题，命中率能到 80% 以上；上线后接入多个租户，模板频繁改版，运营团队还不断 A/B 测试提示词，结果命中率掉到 10% 以下。此时缓存层不仅没降 TTFT，反而增加了查找、序列化、搬运开销，`p95` 还变差。原因不是缓存机制错了，而是工作负载变了。

所以指标不能只看平均值，至少看下面这些：

| 指标 | 作用 |
|---|---|
| `hit rate` | 判断缓存是否真的被用到 |
| `TTFT` | 衡量首 token 用户体验 |
| `p95` / `p99` | 观察尾延迟 |
| 内存占用 | 防止挤爆运行时 |
| OOM 次数 | 检查预算是否不合理 |
| 淘汰次数 | 判断容量是否过小 |

---

## 替代方案与适用边界

缓存优化不是唯一手段，而且很多时候不是第一优先级。它和量化、PagedAttention、批处理、流式保留、冷热分层是互补关系。

| 方案 | 适用条件 | 主要收益 | 主要代价 |
|---|---|---|---|
| 前缀缓存 | 前缀重复率高 | 降低预填充时延 | 需要命中率支撑 |
| 量化 | 算力或显存紧张 | 降低内存和算力成本 | 精度可能下降 |
| PagedAttention | 长上下文、KV 管理复杂 | 提高 KV 管理效率 | 实现复杂度高 |
| 批处理 | 并发高、请求可聚合 | 提高吞吐 | 单请求时延可能变差 |
| 流式保留策略 | 会话型长上下文 | 保留关键历史 | 需要选择性裁剪 |
| 冷热分层 | 热点明显、容量有限 | 提高整体命中效率 | 维护更复杂 |

可以用一个简单判断流程：

1. 先看请求前缀重复率是否稳定高于一个阈值，比如 30% 到 40%。
2. 再看前缀预填充是否占总时延的大头。
3. 再确认本地内存预算允许保留这部分 KV。
4. 最后确认多租户隔离和版本失效机制能否做对。

如果四步里前两步都不成立，就不要优先做缓存。比如随机问答、临时检索、每次提示词都大变的实验型场景，更应该优先考虑量化、批处理或者更高效的 KV 管理，而不是继续扩缓存层。

一个新手能理解的例子是做饭。如果你每天都做同一道菜，提前切菜、分装酱料很值；如果每天菜单都完全随机，预处理反而浪费时间。对应到工程里，RAG 热文档问答、固定客服模板、品牌知识助手适合前缀缓存；随机自由问答、提示词快速迭代实验、跨租户混杂负载，更适合先做量化、批处理或 PagedAttention。

结论可以压成一句话：**当请求存在稳定可复用前缀，且预填充成本高、命中率可观、隔离可控时，优先做缓存；当请求重复率低或负载变化快时，优先做量化、KV 管理和调度优化。**

---

## 参考资料

1. [vLLM Automatic Prefix Caching](https://docs.vllm.ai/)：看工程实现里如何做前缀复用、block 管理和命中逻辑。
2. [PagedAttention: Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180)：理解为什么 KV 管理是长上下文推理的核心瓶颈。
3. [LMCache](https://github.com/LMCache/LMCache)：看多层缓存、远近端缓存和可部署方案的工程化思路。
4. [Hugging Face KV Cache Documentation](https://huggingface.co/docs/transformers/main/en/kv_cache)：理解 KV cache 的基本概念、作用和实现边界。
5. [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)：看高性能推理框架如何处理缓存、paged KV 和推理调度。
6. [SGLang Documentation](https://docs.sglang.ai/)：看面向实际服务系统的前缀复用、调度与长上下文优化。
