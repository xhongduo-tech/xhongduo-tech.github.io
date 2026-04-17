## 核心结论

前缀缓存（Prefix Caching）指把请求前半段已经算过的 KV Cache 复用到后续请求里。KV Cache 可以理解为“模型已经读过前文后留下的中间记忆”。只要多个请求的前缀 token 完全一致，后续请求就不必重复做同一段 prefill，只需要继续处理尾部不同的 token。

它解决的是“相同系统提示、相同 few-shot、相同长文档前缀被反复计算”的浪费。对线上服务最直接的收益是 TTFT（Time To First Token，首 token 延迟）下降，以及单位 GPU 吞吐上升。粗略近似下，在尾部新增 token 长度稳定、排队时间不变时，可以写成：

$$
TTFT_{new} \approx TTFT_{original} \times (1 - hit\_rate)
$$

这里的 `hit_rate` 是前缀缓存命中率。命中率越高，需要重算的 prefill 越少，TTFT 就越低。

玩具例子最容易理解。假设三个请求分别是：

1. `系统提示 + 产品说明书 + 问题A`
2. `系统提示 + 产品说明书 + 问题B`
3. `系统提示 + 产品说明书 + 问题C`

如果前两部分完全相同，那么第一个请求会生成对应的 KV Cache；第二、第三个请求直接复用这部分缓存，只计算最后那句问题。直观上就是“第一次做重活，后面直接接力”。

真实工程里，这种模式大量出现在客服机器人、RAG 问答、批量评测、长文档多问一答、SQL 代理和代码审查助手中。只要开头大段上下文高度重复，前缀缓存就是低风险、高回报的优化。

---

## 问题定义与边界

问题本质很简单：LLM 的 prefill 阶段要先把输入 token 编码成注意力可用的 KV Cache。prefill 可以理解为“先把整段提示读完并建立记忆”。如果每个请求都带相同的系统提示或相同文档前缀，重复 prefill 就是在重复烧 GPU。

前缀缓存不是“相似就行”，而是“token 精确一致才行”。这里的精确一致不是语义一致，而是分词后的 token 序列一致。多一个空格、换一个标点、模型版本变化、模板字段顺序变化，都可能导致无法复用。

| 前缀特征 | 是否可复用 | 常见失败条件 |
| --- | --- | --- |
| 固定系统提示 | 通常可复用 | 字符串有空格、换行、大小写差异 |
| 固定 few-shot 示例 | 通常可复用 | 示例顺序变化、示例内容更新 |
| 固定对话历史前半段 | 部分可复用 | 多轮对话插入不同消息导致后续 token 整体偏移 |
| RAG 固定文档块 | 可复用 | 检索结果顺序不同、拼接模板不同 |
| 动态用户画像 | 往往不可复用 | 每个用户前缀都不同 |
| 跨模型版本请求 | 不应复用 | tokenizer 或模型参数变化 |

一个典型新手场景是客服系统。每次请求都带：

`你是专业客服，请遵守公司售后规则……`

如果这段系统提示固定，前缀缓存就能把它记住，后续请求只算用户新问题。但如果某次模板末尾多了一个空格，或者换成了另一版 prompt，哈希就不同，命中率会立刻下降。

边界还包括部署层面。vLLM 的前缀缓存本质上是引擎内 KV block 的索引复用，天然更偏单实例或单节点内命中；如果要跨节点复用，仅共享“有无命中”还不够，还要能定位缓存数据所在位置并完成搬运，因此需要额外的分层缓存或远端存储设计。SGLang 的 HiCache 正是在这个方向上继续扩展。

---

## 核心机制与推导

vLLM 的思路是按 block 管理 KV Cache。block 可以理解为“固定大小的 token 积木块”。常见配置里 block size 可以是 16 token；为了说明机制，下面用 16 token 举例。

设一个共享前缀长 2048 token，那么它会被拆成：

$$
2048 / 16 = 128 \text{ blocks}
$$

第一次请求到来时，系统对这 128 个 block 做 prefill，并在 block 填满后写入缓存索引。vLLM 的核心做法是对每个 block 计算与其前缀相关的哈希，逻辑上类似：

$$
block\_hash_i = hash(prefix\_tokens_{<i} + block\_tokens_i)
$$

这样设计的原因是：光看当前 block 自己的 token 不够，因为同样的 16 个 token 出现在不同上下文里，语义位置不同，不能随便复用。把前缀也纳入哈希，才能保证“这是这条路径上的这个块”。

查询时，从左到右按 block 检查。只要前面的 block 连续命中，后面的共享前缀就能直接接上；一旦中间某个 block miss，后面的块也不能当成命中前缀处理，因为前缀路径已经断了。

命中率可以写成：

$$
hit\_rate = \frac{hits}{queries}
$$

更贴近监控系统时，也常用：

$$
hit\_rate = \frac{rate(prefix\_cache\_hits\_total)}{rate(prefix\_cache\_queries\_total)}
$$

TTFT 的近似分解可以写成：

$$
TTFT \approx T_{queue} + T_{prefill} + T_{first\_decode}
$$

前缀缓存主要降低的是 $T_{prefill}$。如果共享前缀占输入的大头，那么：

$$
T_{prefill,new} \approx (1-hit\_rate)\cdot T_{prefill,old}
$$

于是得到前面的经验公式：

$$
TTFT_{new} \approx TTFT_{original}\times(1-hit\_rate)
$$

这不是严格物理定律，因为排队、调度、尾部长度、批处理都会影响结果，但在“同类请求很多、共享前缀长、尾部较短”的场景里，这个近似足够指导工程判断。

再看一个数值玩具例子。共享前缀 2048 token，尾部问题 200 token。三次请求只有最后一句不同：

- 无前缀缓存：三次都要 prefill 2048 token
- 有前缀缓存：第一次 prefill 2048 token，第二和第三次只处理尾部 200 token

如果 prefill 本来占 TTFT 的大头，那么第二次开始 TTFT 会明显下降。你可以把它理解成“拼图前 90% 已经拼好，后面只补缺口”。

真实工程例子是企业知识库问答。系统提示固定，文档上下文由检索器返回，但很多用户都在问同一份产品手册。只要拼接模板和文档顺序稳定，大段前缀就会复用成功，第二个请求开始，GPU 不再重复读那几千 token。

---

## 代码实现

实现最小闭环需要三件事：

1. 把前缀切成 block
2. 为 block 建立稳定索引
3. 维护引用计数和淘汰策略

下面先看简化伪代码，逻辑和真实系统一致：

```python
# prefix_tokens 必须是分词后的 token 序列，而不是原始字符串
# block_size 以固定长度切分，块满后才能稳定进入缓存

for block in split_into_blocks(prefix_tokens, block_size=16):
    prefix_hash = hash(parent_hash, block.tokens)

    if prefix_hash in prefix_cache:
        connect_block(prefix_cache[prefix_hash])   # 命中：直接把已有 KV block 接到当前请求
        refcount_increment(prefix_cache[prefix_hash])
    else:
        kv_block = compute_block(block.tokens)     # 未命中：执行 prefill
        prefix_cache.insert(prefix_hash, kv_block)
        refcount_increment(kv_block)

    parent_hash = prefix_hash
```

下面给一个可以运行的 Python 玩具实现。它不真的计算 Transformer 的 KV，只模拟“按 block 命中、统计重算量、用 LRU+refcount 淘汰”的核心流程。

```python
from collections import OrderedDict
from dataclasses import dataclass

BLOCK_SIZE = 4

@dataclass
class CacheEntry:
    tokens: tuple
    refcount: int = 0

class PrefixCache:
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.table = OrderedDict()  # key -> CacheEntry

    def _make_key(self, parent_key, block_tokens):
        return (parent_key, tuple(block_tokens))

    def lookup_or_insert(self, parent_key, block_tokens):
        key = self._make_key(parent_key, block_tokens)
        if key in self.table:
            entry = self.table.pop(key)
            self.table[key] = entry  # LRU touch
            entry.refcount += 1
            return True, key
        self.evict_if_needed()
        self.table[key] = CacheEntry(tokens=tuple(block_tokens), refcount=1)
        return False, key

    def release(self, keys):
        for key in keys:
            if key in self.table:
                self.table[key].refcount -= 1
                assert self.table[key].refcount >= 0

    def evict_if_needed(self):
        if len(self.table) < self.capacity:
            return
        for key, entry in list(self.table.items()):
            if entry.refcount == 0:
                del self.table[key]
                return
        raise RuntimeError("no evictable block")

def split_blocks(tokens, block_size=BLOCK_SIZE):
    return [tokens[i:i + block_size] for i in range(0, len(tokens), block_size)]

def process_request(cache, tokens):
    parent = None
    hits = 0
    computed = 0
    used_keys = []
    for block in split_blocks(tokens):
        hit, key = cache.lookup_or_insert(parent, block)
        used_keys.append(key)
        if hit:
            hits += 1
        else:
            computed += 1
        parent = key
    return hits, computed, used_keys

cache = PrefixCache(capacity=16)

req1 = "SYS DOC DOC DOC Q1".split()
req2 = "SYS DOC DOC DOC Q2".split()
req3 = "SYS DOC DOC DOC Q3".split()

hits1, computed1, keys1 = process_request(cache, req1)
cache.release(keys1)

hits2, computed2, keys2 = process_request(cache, req2)
cache.release(keys2)

hits3, computed3, keys3 = process_request(cache, req3)
cache.release(keys3)

assert computed1 > computed2
assert computed1 > computed3
assert hits2 >= 1 and hits3 >= 1
```

这段代码表达了四个工程事实：

- 缓存键不是“当前块内容” alone，而是“父路径 + 当前块”
- 命中时不重算，只连接已有块
- 正在被请求使用的块不能淘汰，所以要有 `refcount`
- 缓存满时优先淘汰 `refcount=0` 的旧块，所以常配合 LRU

vLLM 的重点是 block-hash 索引和本地块管理。SGLang 则从 RadixAttention 演进到 HiCache，把 GPU、Host、远端存储做成分层缓存。前者更像“本机查哈希表复用 block”，后者更像“用树形元数据描述前缀路径，并决定 KV 在 GPU、CPU 还是更远存储的哪一层”。这就是两者实现差异的核心。

---

## 工程权衡与常见坑

前缀缓存的收益很高，但它不是“打开开关就一定快”。很多线上命中率低，不是算法不行，而是输入治理做得差。

| 坑 | 触发条件 | 规避措施 |
| --- | --- | --- |
| 前缀微小差异 | 多一个空格、换行、时间戳、随机 ID | 用固定模板生成 prompt，禁止无关动态字段混入前缀 |
| 命中率看似高但 TTFT 不明显 | 尾部新 token 很长，或排队占主导 | 拆分 TTFT 到 queue/prefill/decode 分别看 |
| KV 内存满 | 长上下文多并发，缓存块太多 | 调整 block 数、GPU 利用率、Host 分层缓存 |
| 淘汰过快 | LRU 容量过小，热点前缀被反复驱逐 | 扩大缓存池，按业务前缀分组 |
| 跨节点命中差 | 请求落到不同实例，本地缓存不共享 | 做会话粘性，或使用分层/远端 KV 方案 |
| 模型升级后误判可复用 | tokenizer 或模型权重变化 | 按模型版本隔离缓存命名空间 |

最常见的坑是“看起来一样，实际 token 不一样”。例如：

- 模板 A：`你是专业客服。`
- 模板 B：`你是专业客服。 `

B 末尾多一个空格，分词后很可能已经不同。哈希不同，命中率可能直接掉到 0。工程上应当把系统提示做成常量，不要在字符串拼接时混入隐形差异。

另一个权衡是空间换时间。KV Cache 很占显存，尤其长上下文模型更明显。缓存块保留得越多，命中率可能越高；但占用越大，能承载的并发和上下文长度就越受限。vLLM 用 `ref_cnt=0` 且最近最少使用的块优先淘汰，本质是在“未来可能复用”与“当前必须分配内存”之间做平衡。

成本也能粗略量化。设一批请求平均前缀长度为 $P$，尾部新增长度为 $T$，总请求数为 $N$，其中除第一次外都命中，那么原始 prefill 成本约为：

$$
N \cdot P
$$

启用前缀缓存后近似变为：

$$
P + N \cdot T
$$

如果 $P \gg T$，且 $N$ 足够大，节省量会非常可观。这也是为什么共享长文档问答、批量评测一类任务最适合前缀缓存。

---

## 替代方案与适用边界

不是所有重复请求都应该先想到前缀缓存。如果每个请求开头都不同，命中率上不去，收益就会很差。

| 方案 | 适用条件 | 主要优势 | 主要边界 |
| --- | --- | --- | --- |
| 前缀缓存 | 大量请求共享相同前缀 | 显著降低 prefill，TTFT 改善直接 | 依赖 token 级一致 |
| 请求批处理 | 同时到达的请求多 | 提高 GPU 利用率和吞吐 | 不一定降低单请求 TTFT |
| Embedding 缓存 | 重复做检索或相似度计算 | 降低检索阶段成本 | 不解决生成阶段 prefill |
| Prompt 预编译/模板固化 | 前缀结构不稳定但可治理 | 提高缓存命中前提 | 需要改业务拼接逻辑 |
| 分层 KV 缓存 | 单机显存不足、跨实例复用强 | 扩大可复用容量 | 引入搬运与网络延迟 |

一个典型反例是动态用户画像 RAG。假设系统提示每次都拼入当前用户偏好、权限标签、实验组参数，那么虽然“主题相近”，但 token 前缀经常不同，前缀缓存会变得不可靠。这时更合适的优化通常是：

- 先把动态字段移到尾部，尽量固化真正的共享前缀
- 检索侧做 embedding 缓存
- 生成侧做 batching 或异步流水线

如果必须跨节点共享 KV，SGLang 的 HiCache 更接近这类需求。它把 GPU 当 L1、Host 当 L2、分布式存储当 L3，并用 HiRadixTree 维护前缀路径与存储位置。代价是系统更复杂，且网络与 I/O 延迟开始成为边界条件。换句话说，跨节点可复用不再只是“命不中哈希表”，而是“命中了还能不能足够快搬回来”。

---

## 参考资料

- [vLLM Automatic Prefix Caching 设计文档](https://docs.vllm.ai/en/stable/design/prefix_caching.html)：说明 block 级哈希、`ref_cnt`、free queue 与 LRU eviction 的核心实现。
- [vLLM Automatic Prefix Caching 旧版设计页](https://docs.vllm.ai/design/prefix_caching.html)：补充 `cache_salt`、多模态额外哈希和块路径哈希思路。
- [vLLM Engine Arguments](https://docs.vllm.ai/en/v0.8.0/serving/engine_args.html)：给出 `--enable-prefix-caching` 与 block size 配置边界。
- [SGLang HiCache System Design and Optimization](https://docs.sglang.ai/advanced_features/hicache_design.html)：解释 HiCache 的 GPU/Host/L3 分层缓存与 HiRadixTree 元数据组织。
- [SGLang Server Arguments](https://docs.sglang.ai/backend/server_arguments.html)：给出 `--enable-hierarchical-cache`、`--disable-radix-cache` 等服务参数。
- [NVIDIA NIM KV Cache Reuse](https://docs.nvidia.com/nim/large-language-models/1.12.0/kv-cache-reuse.html)：给出“90% 以上共享前缀时 TTFT 常见约 2x 改善”的官方使用边界。
- [NVIDIA TensorRT-LLM KV Cache Early Reuse](https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/)：讨论更细粒度 block 对复用率与 TTFT 的影响。
