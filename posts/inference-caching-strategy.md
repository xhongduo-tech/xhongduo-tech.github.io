## 核心结论

推理缓存的目标不是“把请求都存起来”，而是把重复计算变成重复复用，本质上是用显存换延迟、换吞吐。

最容易混淆的点是：`KV cache`、`prefix cache`、`output cache` 不是一回事。`KV cache` 是 Transformer 推理时保存历史注意力状态的数据结构，白话说，就是“模型已经算过的上下文中间结果”；`prefix cache` 是复用一段相同前缀对应的 KV；`output cache` 是直接复用整次请求的最终回答。

在工程上，真正有价值的判断不是“要不要缓存”，而是“缓存哪一层、按什么 key 命中、何时淘汰、如何隔离租户”。如果这些边界没定清楚，缓存会从优化手段变成负担：显存被占满、碎片增多、热前缀反复失效、峰值延迟上升。

| 类型 | 缓存对象 | 命中条件 | 主要收益 |
|---|---|---|---|
| KV cache | 当前请求已生成 token 的注意力状态 | 同一请求继续生成 | 避免重复计算历史 token |
| Prefix cache | 相同前缀对应的 KV 块 | 同模板、同 tokenizer、同 token 序列 | 降低 prefill 成本 |
| Output cache | 最终输出文本或 token 序列 | 同模型、同输入、同参数、同 seed、同租户 | 直接跳过整次生成 |

玩具例子：企业知识库机器人每次都带一段 1200 token 的系统提示和合规模板。如果这段前缀不变，那么第二个请求开始就不必重新做完整 prefill，这就是前缀缓存。若某个标准问答在 `temperature=0` 下总是返回同一答案，还可以用输出缓存直接返回结果。

真实工程例子：多租户企业问答平台里，公共政策文档、系统规则、工具说明经常被重复拼进 prompt。前缀缓存能显著降低长上下文的首 token 延迟；但如果不同租户共用同一缓存池且没有 namespace，一个租户的突发流量会把别人的热前缀挤掉，最终所有人的命中率一起下降。

---

## 问题定义与边界

本文讨论的是推理服务侧缓存，不讨论训练时的数据缓存，也不讨论模型权重加载缓存。前者优化的是“这次请求怎么少算一些 token”，不是“模型文件怎么更快放进显存”。

判断是否可缓存，不能靠“语义差不多”，而要看是否满足“复用层的等价条件”。对前缀缓存来说，关键是 token 序列一致；对输出缓存来说，关键是整次请求的可复现条件一致。两个用户都说“解释这份政策”，只要 chat template 不同、工具调用包装不同、tokenizer 版本不同，缓存层面就不是同一个请求。

一个常见误区是把“文本看起来一样”当成“前缀一样”。实际上，下面这两种输入即使自然语言接近，也可能完全不能复用：

1. 同一句用户问题，但系统提示顺序不同。
2. 同一段 prompt，但一个请求走 chat completions 模板，另一个走 responses 模板。
3. 同一份历史对话，但一个保留了 `<think>` 或 reasoning token，另一个在回放时去掉了。

| 场景 | 判断 | 原因 |
|---|---|---|
| 固定系统提示 + 固定模板 + 长文档问答 | 可缓存 | 前缀长且重复率高 |
| 标准 FAQ，`temperature=0` | 可缓存 | 输出高度确定 |
| 开放式创作，`temperature>0` | 不可直接做输出缓存 | 输出不稳定 |
| 带外部工具副作用的请求 | 需谨慎缓存 | 结果可能受实时数据影响 |
| 同文本但 tokenizer / template 版本变化 | 不可缓存 | token 序列或解析边界已变 |
| 推理模型包含 thinking token 的多轮会话 | 需谨慎缓存 | 可复用视图可能与用户后续输入不一致 |

所以本文边界很明确：缓存是一个“严格等价复用”问题，不是“语义近似匹配”问题。近似匹配属于检索或语义缓存范畴，那是另一类系统设计。

---

## 核心机制与推导

前缀缓存复用的是“前缀对应的 KV 状态”。白话说，模型已经把这段前缀看过并算出中间结果，只要下一个请求在这段前缀之前完全一样，就可以接着往后算，而不是从头开始。

设：

- $B$：每个 block 包含的 token 数
- $S_{kv}$：每个 token 的 KV 显存占用
- $C$：缓存中 block 数
- $L_p$：可复用前缀长度
- $L_o$：可复用输出长度
- $H_p$：前缀缓存命中率
- $H_o$：输出缓存命中率

则缓存显存占用可近似写成：

$$
M_{cache} = C \times B \times S_{kv}
$$

这个式子只表达一个核心事实：缓存不是抽象概念，它直接吃显存，而且通常近似线性增长。

一次请求的节省时间可粗略写成：

$$
Saved \approx H_p \times T_{prefill}(L_p) + H_o \times T_{decode}(L_o)
$$

其中 $T_{prefill}(L_p)$ 是长度为 $L_p$ 的前缀预填充时间，$T_{decode}(L_o)$ 是生成 $L_o$ 个输出 token 的解码时间。这个式子不追求精确预测，而是说明：前缀缓存主要节省 prefill，输出缓存主要节省 decode。

再看收益与占用的矛盾。若某类缓存条目的命中收益为 $Benefit_i$，占用字节为 $Bytes_i$，淘汰时更合理的思路通常不是只看最近最少使用，而是看单位显存收益：

$$
Score_i = \frac{Benefit_i}{Bytes_i}
$$

分数越低，越应该被淘汰。因为一个超长但几乎不复用的前缀，会比一个中等长度但高频命中的前缀更浪费显存。

玩具例子：假设 $S_{kv}=256\ \text{KB/token}$，$L_p=1024$，则一段前缀的 KV 大小约为：

$$
1024 \times 256\ \text{KB} \approx 256\ \text{MB}
$$

如果 8 个请求共享这段前缀，而缓存只存一份，就能少做 7 次同样的 prefill。这个例子说明，长前缀下，命中一次前缀缓存的收益通常很大。

| 符号 | 含义 | 工程作用 | 容易误解点 |
|---|---|---|---|
| $B$ | block 大小 | 决定缓存切分粒度 | 不是请求长度 |
| $S_{kv}$ | 每 token KV 占用 | 估算显存成本 | 与模型结构强相关 |
| $C$ | block 数 | 控制缓存容量 | 不是请求数 |
| $L_p$ | 前缀长度 | 估算 prefill 节省 | 只有可复用部分才算 |
| $L_o$ | 输出长度 | 估算输出缓存收益 | 不等于总回复长度 |
| $H_p$ | 前缀命中率 | 衡量模板复用效果 | 高命中不代表高收益 |
| $H_o$ | 输出命中率 | 衡量结果复用效果 | 只适合高确定性场景 |

从实现机制看，现代推理框架常把 KV 切成 block，再按“前缀 + 当前 block token”做索引。vLLM 文档明确把 block 唯一性建立在父哈希、block token 和额外哈希上，额外哈希可包含 LoRA、模态输入哈希和多租户隔离用的 cache salt。这说明工程上真正的 key 从来不只是 `prompt_hash`，而是“影响 KV 正确性的全部条件集合”。

输出缓存更严格。它不是复用中间状态，而是复用最终答案，所以 key 至少要覆盖：`tenant_id`、`model_id`、`tokenizer_version`、`template_hash`、`normalized_prompt_hash`、`sampling_params_hash`、`seed`。漏掉任何一个，都可能把错误答案返回给错误请求。

---

## 代码实现

实现顺序应该是：先定义 key 的语义边界，再写 lookup / insert / eviction。很多线上故障都不是 LRU 算法本身有问题，而是 key 设计先天不完整。

先看前缀 key。关键不是“把字符串 hash 一下”，而是“对规范化后的 token 视图做 hash”。

```python
from dataclasses import dataclass
from hashlib import sha256
import json

def stable_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()

def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())

def build_prefix_key(
    tenant_id: str,
    model_id: str,
    tokenizer_version: str,
    chat_template_hash: str,
    prefix_token_ids: list[int],
    extra_salt: str = "",
) -> str:
    return stable_hash({
        "tenant_id": tenant_id,
        "model_id": model_id,
        "tokenizer_version": tokenizer_version,
        "chat_template_hash": chat_template_hash,
        "prefix_token_ids": prefix_token_ids,
        "extra_salt": extra_salt,
    })

k1 = build_prefix_key("t1", "m1", "tok-v1", "tpl-a", [1, 2, 3])
k2 = build_prefix_key("t1", "m1", "tok-v1", "tpl-a", [1, 2, 3])
k3 = build_prefix_key("t2", "m1", "tok-v1", "tpl-a", [1, 2, 3])

assert k1 == k2
assert k1 != k3
```

上面这个可运行例子说明两件事：同租户、同模型、同模板、同 token 前缀应命中；跨租户默认不应共享。是否允许跨租户共享，是安全策略，不是默认行为。

再看请求流程。前缀缓存和输出缓存必须走两条路径：

```python
def handle_request(req, prefix_cache, output_cache, model_runner):
    norm_prompt = normalize_text(req.prompt)
    token_ids = req.tokenize(norm_prompt)

    prefix_key = build_prefix_key(
        tenant_id=req.tenant_id,
        model_id=req.model_id,
        tokenizer_version=req.tokenizer_version,
        chat_template_hash=req.chat_template_hash,
        prefix_token_ids=token_ids[:req.reusable_prefix_len],
        extra_salt=req.cache_salt,
    )

    if req.enable_output_cache:
        output_key = stable_hash({
            "tenant_id": req.tenant_id,
            "model_id": req.model_id,
            "tokenizer_version": req.tokenizer_version,
            "chat_template_hash": req.chat_template_hash,
            "prompt_hash": stable_hash(token_ids),
            "params_hash": stable_hash(req.sampling_params),
            "seed": req.seed,
        })
        cached = output_cache.get(output_key)
        if cached is not None:
            return {"source": "output_cache", "text": cached}

    kv = prefix_cache.get(prefix_key)
    if kv is not None:
        text = model_runner.decode_from_kv(kv, token_ids, req.sampling_params)
        source = "prefix_cache"
    else:
        kv, text = model_runner.prefill_and_decode(token_ids, req.sampling_params)
        prefix_cache.put(prefix_key, kv)
        source = "model"

    if req.enable_output_cache and req.is_deterministic():
        output_cache.put(output_key, text)

    return {"source": source, "text": text}
```

这里最重要的工程判断是 `req.is_deterministic()`。典型条件包括：`temperature=0`、无实时工具副作用、无外部时间敏感依赖、seed 固定、模板稳定。否则输出缓存容易把“上次碰巧生成的答案”错误复用到“这次本该生成不同答案”的请求上。

最后看淘汰。简单 LRU 往往不够，因为一个很长但低频的前缀会占掉大量显存。

```python
def eviction_score(entry) -> float:
    # benefit 近似 = 命中次数 * 每次节省毫秒
    benefit = entry.hit_count * entry.saved_ms_per_hit
    # 加 1 防止除零
    return benefit / (entry.bytes_used + 1)

def choose_victim(entries):
    candidates = [e for e in entries if e.ref_count == 0]
    assert candidates, "no evictable entries"
    return min(candidates, key=eviction_score)
```

真实系统里，`saved_ms_per_hit` 还会按 prefill 成本、decode 成本、租户优先级、前缀长度、是否为公共模板做加权。核心思想是：淘汰应近似最小化未来损失，而不是机械删除最老对象。

| Key 字段 | 用于前缀缓存 | 用于输出缓存 | 说明 |
|---|---|---|---|
| `tenant_id` | 是 | 是 | 做租户隔离 |
| `model_id` | 是 | 是 | 模型不同不可混用 |
| `tokenizer_version` | 是 | 是 | 分词变化会破坏等价性 |
| `chat_template_hash` | 是 | 是 | 模板差异会改 token 序列 |
| `prompt_hash` / `prefix_token_ids` | 是 | 是 | 前缀缓存更看 token 视图 |
| `params_hash` | 否或弱相关 | 是 | 采样参数直接影响输出 |
| `seed` | 通常否 | 是 | 保证确定性复现 |
| `cache_salt` | 常用 | 可选 | 多租户安全隔离 |

---

## 工程权衡与常见坑

缓存容量、命中率、显存压力三者始终拉扯。缓存池变大，短期看命中率会上升；但它也会压缩 batch 空间、增加内存管理复杂度，并放大碎片和驱逐抖动。对长上下文服务，这种矛盾尤其明显。

真实工程例子：一个多租户问答平台有 50 个企业租户，大家都复用一套法规模板，但其中一个大租户在工作日白天不断提交新文档问答。若缓存不按租户做 namespace 或配额，这个租户会持续写入新的长前缀，把其他租户的公共模板挤出。表面现象可能不是“缓存错了”，而是“命中率忽高忽低、free queue 抖动、p95 延迟升高”。

| 坑位 | 后果 | 规避方式 |
|---|---|---|
| 模板不统一 | 相同文本无法命中 | 固定 chat template 和字段顺序 |
| tokenizer 不一致 | token 序列不等价 | 将 tokenizer 版本纳入 key |
| 输出缓存用于随机生成 | 返回错误或不自然答案 | 仅对高确定性请求启用 |
| 把 reasoning token 混入可复用内容 | 形成死分支、浪费显存 | 只缓存未来可见且可回放的 token 视图 |
| 缺少 tenant namespace | 租户相互污染 | 加 `tenant_id` 或 `cache_salt` |
| 只看 LRU 不看收益 | 长低频前缀霸占显存 | 引入 `benefit / byte` 权重 |
| 只统计命中率 | 错判优化效果 | 同时看 saved ms、GPU 利用率、p95 延迟 |

故障排查时，可以用下面这张对照表先缩小范围：

| 现象 | 可能原因 |
|---|---|
| 命中率突然从高位掉到很低 | 模板变更、tokenizer 升级、租户热点挤压 |
| 显存占用高但收益不明显 | 长前缀低复用、输出缓存误开、淘汰策略失衡 |
| 多轮推理模型越跑越慢 | thinking token 被写入可复用前缀，形成不可达分支 |
| 高并发时尾延迟升高 | 热块反复驱逐、缓存池与 batch 争抢显存 |

这里要特别强调 reasoning token 的坑。若服务端把模型内部 thinking token 也插入前缀树，而客户端下一轮构造 prompt 时又不会带回这些 token，那么这些 KV 会变成“在树里存在、但未来请求永远匹配不到”的死数据。SGLang 的相关 issue 就暴露了这种问题：它不仅浪费显存，还会让本可复用的答案部分被堵在死分支后面，导致再次计算。

---

## 替代方案与适用边界

缓存不是通用解。低复用、高随机性、强外部依赖的请求，即使缓存做对了，也可能收益很低。

前缀缓存、输出缓存和其他优化手段解决的是不同问题。`batching` 是把多个请求一起算，提升设备利用率；prompt 压缩是减少输入长度；speculative decoding 是加速生成阶段；缓存则是复用过去已经算过的结果。它们可以叠加，但不要混为一谈。

开放式创作类请求通常不适合输出缓存，因为温度、top-p、上下文细微差别都会改变答案；但它仍可能适合前缀缓存，只要系统提示和角色设定稳定。相反，标准化问答、模板化提取、固定分类任务，往往最适合输出缓存。

| 方案 | 适用场景 | 主要收益 | 风险点 |
|---|---|---|---|
| 前缀缓存 | 长前缀、高复用、模板稳定 | 降低 prefill 延迟 | 吃显存，要求 token 严格一致 |
| 输出缓存 | 高确定性、强重复、低副作用 | 直接跳过生成 | 容易因 key 不全而误命中 |
| 不缓存 | 高随机、低复用、强实时依赖 | 简化系统 | 放弃复用收益 |
| Prompt 压缩 | 前缀过长但重复不高 | 降低输入长度 | 可能损失语义信息 |
| Batching | 高并发、请求可排队 | 提升吞吐 | 可能增加单请求等待 |
| Speculative decoding | 解码阶段重 | 降低生成延迟 | 依赖额外模型或复杂控制 |

可以用一条简单规则做初筛：

- 高确定性、强重复、低副作用：优先考虑输出缓存。
- 长前缀、高复用、模板稳定：优先考虑前缀缓存。
- 高随机、强交互、强实时外部依赖：谨慎缓存，必要时不缓存。
- 命中收益不高但输入很长：先考虑 prompt 压缩。
- 设备利用率低、队列长：优先考虑 batching。

所以，“推理缓存策略设计”的核心不是发明一个更花哨的 LRU，而是把复用层级、等价条件、显存预算和租户边界统一到一个可观测、可解释、可调整的系统里。

---

## 参考资料

| 主题 | 链接 | 适合解决的问题 |
|---|---|---|
| 机制基础 | vLLM Automatic Prefix Caching | 前缀缓存的哈希组成、完整 block 缓存边界、多租户 salt |
| 机制基础 | vLLM Prefix Caching Implementation Details | 为什么前缀缓存要求“前缀 token + 当前 block token”唯一对应 KV block |
| 机制基础 | PagedAttention | 为什么 KV 管理会受碎片和重复分配影响 |
| 工程实现 | SGLang RadixAttention | 前缀树式复用在高性能服务框架中的工程落地 |
| 分布式 / 多租户 | SGLang distributed cache architecture issue | 分布式环境下计算与缓存解耦、cache-aware routing 的必要性 |
| 边界与问题 | SGLang reasoning tokens pollute radix cache issue | reasoning token 为什么会污染前缀缓存并制造不可达分支 |

1. [vLLM: Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
2. [vLLM: Prefix Caching Implementation Details](https://docs.vllm.ai/en/v0.5.3.post1/automatic_prefix_caching/details.html)
3. [PagedAttention: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://huggingface.co/papers/2309.06180)
4. [SGLang README: RadixAttention for prefix caching](https://github.com/sgl-project/sglang)
5. [SGLang issue: distributed cache architecture](https://github.com/sgl-project/sglang/issues/7761)
6. [SGLang issue: reasoning tokens pollute radix cache](https://github.com/sgl-project/sglang/issues/22373)
