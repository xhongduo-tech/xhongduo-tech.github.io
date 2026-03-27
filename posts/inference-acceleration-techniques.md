## 核心结论

大模型推理加速里，`KV Cache`、`Speculative Decoding`、`PagedAttention` 解决的是三个不同层面的瓶颈，但它们可以叠加。

`KV Cache` 的白话解释是：把已经算过的注意力中间结果留下来，下一步直接复用，不再把整段前缀重新算一遍。它主要减少重复计算。

`PagedAttention` 的白话解释是：把 KV 缓存像操作系统页表一样分块管理，只给真正用到的位置分配显存，不提前占满最大上下文。它主要减少显存浪费和碎片。

`Speculative Decoding` 的白话解释是：让一个更快的草稿模型先猜多个 token，再由目标模型一次性验证，验证通过的部分直接接入输出。它主要减少“目标模型每次只推进 1 个 token”的低效率。

这三者组合后的本质是：把自回归生成从“每个 token 都重新跑很多旧工作”，改成“旧结果复用、KV 按需分配、目标模型批量确认多个 token”。如果只记一句话，可以记成：

$$
\text{推理加速} = \text{减少重复算} + \text{减少无效占用} + \text{增加每轮推进长度}
$$

一个经验判断是：

| 技术 | 主要解决问题 | 代价 | 最适合场景 |
|---|---|---|---|
| KV Cache | 重复计算前缀 | 显存占用随上下文增长 | 几乎所有生成场景 |
| PagedAttention | KV 预留浪费、碎片 | 内核实现更复杂 | 高并发、长短请求混跑 |
| Speculative Decoding | 单步生成吞吐低 | 需要草稿模型或额外结构 | 低时延、高吞吐服务 |

所以，工程上通常不是“三选一”，而是先上 KV Cache，再在高并发服务里引入 PagedAttention，再根据 acceptance rate 决定是否叠加 speculative decoding。这里的 `acceptance rate` 可以理解为“草稿模型猜中的比例”。

---

## 问题定义与边界

先定义问题。大语言模型的标准生成方式是 `autoregressive decoding`，即自回归解码，意思是“每次只生成下一个 token，再把它接回输入继续生成”。问题在于，这个流程天然串行。

如果没有 KV Cache，第 $t$ 个 token 生成时，模型会重新处理前面 $1 \sim t-1$ 的全部上下文。对注意力层来说，这意味着大量前缀被重复计算。新人最容易忽略的一点是：模型慢，不只是参数多，更因为它在重复做已经做过的事。

KV Cache 解决了“算力重复”问题，但引入了新的边界：缓存必须放在显存里，且序列越长、层数越多、头数越多，缓存越大。一个常见近似公式是：

$$
M_{kv} \approx 2 \times L \times S \times H \times D \times B \times \text{bytes}
$$

其中：

- $L$ 是层数
- $S$ 是序列长度
- $H$ 是注意力头数
- $D$ 是每个头的维度
- $B$ 是 batch size
- 前面的 $2$ 表示同时存 `Key` 和 `Value`

这说明 KV Cache 把时间换成了空间。

接着是第二个问题：很多系统为了简单，会按“最大上下文长度”给每个请求预留整块 KV 空间。比如一个请求实际只生成 500 个 token，但系统按 2048 token 预留，这样利用率只有：

$$
500 / 2048 \approx 24.4\%
$$

也就是说，约 76% 的 KV 预留根本没被用到。这还不是最糟的。更麻烦的是，请求长短不一，释放时间也不同，显存会出现碎片，最后表现为“还有显存，但分配失败”。

这正是 PagedAttention 的边界：它不改变 Transformer 的数学定义，它只是把 KV 的内存管理从“连续大块预留”改成“固定大小块按需分配”。

第三个问题是：即使已经用了 KV Cache，目标模型在标准解码里每次前向传播通常还是只确认 1 个 token。GPU 很强，但这种逐 token 串行流程很难把吞吐吃满。Speculative Decoding 的边界就在这里：它不减少目标模型参数量，而是想办法让目标模型一次确认多个 token。

一个玩具例子可以把三个问题区分清楚：

| 情况 | 没有优化 | 加上 KV Cache | 再加 PagedAttention | 再加 Speculative |
|---|---|---|---|---|
| 生成第 501 个 token | 前 500 个前缀又算一遍 | 前 500 个复用缓存 | 前 500 个复用且只占已用页 | 目标模型可能一次确认多个 token |
| 资源瓶颈 | 算力浪费 | 显存上升 | 内存管理复杂 | 需要高 acceptance |

所以本文讨论的边界很明确：只讨论推理阶段，不讨论训练；只讨论基于 Transformer 的自回归生成，不展开非自回归模型；只讨论工程上已经广泛使用的三类技术，不把量化、蒸馏、MoE 路由当主角。

---

## 核心机制与推导

先看 KV Cache。

在自注意力里，每个 token 都会产生 $Q$、$K$、$V$。第 $t$ 步真正需要的是：新 token 的 $Q_t$ 去和历史的所有 $K_{1:t}$ 做匹配，再用匹配权重对 $V_{1:t}$ 加权求和。历史 token 的 $K$ 和 $V$ 一旦算过，后续并不会变，所以可以缓存。

因此，第 $t$ 步不需要重算前 $t-1$ 个 token 的 $K/V$，只需要补算当前 token 的 $Q_t, K_t, V_t$，再把 $K_t, V_t$ 追加进缓存。这就是 KV Cache 的机制核心。

如果没有缓存，整体注意力代价会随着序列增长不断重复前缀；有缓存后，每一步只新增一个 token 的投影和一次“新 Q 对旧 K/V”的计算。直观上，它把“每步重走整段路”改成“每步只在旧路后面再走一步”。

再看 PagedAttention。

PagedAttention 把序列的 KV 缓存切成固定大小 block，例如每块 16 个 token。逻辑上，一个序列还是连续的；物理上，它对应的是若干离散 block。于是系统只需要维护一张 `block table`，也就是“逻辑块号 -> 物理块地址”的映射表。

这样做的直接结果有三个：

1. 只在需要时申请新 block，而不是一次性按最大长度预留。
2. 请求结束时，相关 block 可以立刻回收到空闲池。
3. 多个请求若共享前缀，可以共享物理 block，再通过 `copy-on-write` 处理分叉。

这里的 `copy-on-write` 可以白话理解为：先共用，真要修改时再复制。

PagedAttention 的内存成本可以写成更接近实际使用量的形式：

$$
M_{paged} \approx 2 \times N_{active\ blocks} \times b \times H \times D \times \text{bytes}
$$

其中：

- $N_{active\ blocks}$ 是当前活跃 block 数
- $b$ 是 block size

关键区别是，它不再按“最大长度”计费，而按“当前活跃块数”计费。这也是它能把 KV 浪费从大比例预留压到很低的原因。

再看 Speculative Decoding。

它通常包含两个角色：

- `draft model`：草稿模型，更小更快，先猜接下来 $\gamma$ 个 token
- `target model`：目标模型，真正决定输出的模型

流程是：

1. 草稿模型先生成 $\gamma$ 个候选 token。
2. 目标模型对这段候选做一次验证。
3. 从左到右找到最长一致前缀。
4. 一致的部分直接接受，不一致处由目标模型接管。

它的收益来自“目标模型一次前向，不只推进 1 个 token”。衡量这个收益的常用公式是期望推进长度 $\tau$：

$$
\tau = \frac{1 - \alpha^{\gamma+1}}{1-\alpha}
$$

其中：

- $\alpha$ 是 acceptance rate，即候选 token 被接受的概率
- $\gamma$ 是每轮草稿提出的 token 数

这个公式的直觉是：$\alpha$ 越高，草稿越靠谱；$\gamma$ 越大，每轮潜在推进越多，但验证失败风险也更高。

看一个玩具数值例子。设 $\gamma = 5$，$\alpha = 0.8$：

$$
\tau = \frac{1 - 0.8^6}{1-0.8}
= \frac{1 - 0.262144}{0.2}
\approx 3.689
$$

这意味着目标模型每次验证平均能推进约 3.7 个 token，而不是标准解码里的 1 个 token。实际吞吐不会正好提升 3.7 倍，因为还有草稿模型开销、调度开销、缓存管理开销，但方向上是成立的。

把三者连起来看，完整推导逻辑是：

- KV Cache 让“历史上下文”不重复算。
- PagedAttention 让“历史上下文对应的缓存”不无效占显存。
- Speculative Decoding 让“目标模型对未来的推进”不再一次只走 1 步。

一个真实工程例子是在线聊天服务。假设同时有三类请求：

- 短问答：输出 50 到 150 token
- 普通对话：输出 300 到 800 token
- 长文生成：输出 2000+ token

如果只用连续预分配，短请求会浪费大量 KV 空间，并且和长请求混在一起时产生碎片；如果只用 KV Cache，不做分页，高并发下仍会很快吃满显存；如果只做 speculative 但 acceptance 不高，可能收益不稳定。于是工程上最常见的组合就是：`KV Cache + PagedAttention` 打底，再对高频业务流量评估是否叠加 speculative decoding。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不依赖 GPU，也不实现真正的 Transformer 注意力，而是只模拟两个关键点：

- 分页式 KV 缓存如何按 block 追加和截断
- speculative decoding 如何根据 acceptance 前缀推进

```python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class PagedKVCache:
    block_size: int
    free_blocks: List[int]
    block_tables: Dict[str, List[int]] = field(default_factory=dict)
    physical_blocks: Dict[int, List[str]] = field(default_factory=dict)
    lengths: Dict[str, int] = field(default_factory=dict)

    def _alloc_block(self) -> int:
        assert self.free_blocks, "out of blocks"
        block_id = self.free_blocks.pop(0)
        self.physical_blocks[block_id] = []
        return block_id

    def append_token(self, seq_id: str, token: str) -> None:
        pos = self.lengths.get(seq_id, 0)
        block_idx = pos // self.block_size
        offset = pos % self.block_size

        table = self.block_tables.setdefault(seq_id, [])
        while len(table) <= block_idx:
            table.append(self._alloc_block())

        block_id = table[block_idx]
        block = self.physical_blocks[block_id]

        if offset == len(block):
            block.append(token)
        else:
            block[offset] = token

        self.lengths[seq_id] = pos + 1

    def get_tokens(self, seq_id: str) -> List[str]:
        total = self.lengths.get(seq_id, 0)
        result = []
        for block_id in self.block_tables.get(seq_id, []):
            result.extend(self.physical_blocks[block_id])
        return result[:total]

    def truncate(self, seq_id: str, new_len: int) -> None:
        old_len = self.lengths.get(seq_id, 0)
        assert 0 <= new_len <= old_len

        keep_blocks = (new_len + self.block_size - 1) // self.block_size if new_len else 0
        table = self.block_tables.get(seq_id, [])

        while len(table) > keep_blocks:
            block_id = table.pop()
            self.physical_blocks.pop(block_id, None)
            self.free_blocks.append(block_id)

        if keep_blocks > 0 and new_len % self.block_size != 0:
            last_block_id = table[-1]
            keep = new_len % self.block_size
            self.physical_blocks[last_block_id] = self.physical_blocks[last_block_id][:keep]

        self.lengths[seq_id] = new_len

    def release(self, seq_id: str) -> None:
        for block_id in self.block_tables.get(seq_id, []):
            self.physical_blocks.pop(block_id, None)
            self.free_blocks.append(block_id)
        self.block_tables.pop(seq_id, None)
        self.lengths.pop(seq_id, None)


def speculative_step(draft_tokens: List[str], target_tokens: List[str]) -> Tuple[List[str], str]:
    accepted = []
    for d, t in zip(draft_tokens, target_tokens):
        if d == t:
            accepted.append(d)
        else:
            return accepted, t
    return accepted, target_tokens[len(accepted)]


cache = PagedKVCache(block_size=4, free_blocks=list(range(10)))
seq_id = "chat-1"

for token in ["我", "想", "学", "推", "理", "加", "速"]:
    cache.append_token(seq_id, token)

assert cache.get_tokens(seq_id) == ["我", "想", "学", "推", "理", "加", "速"]
assert len(cache.block_tables[seq_id]) == 2  # 7 个 token，按 block_size=4 需要 2 块

draft = ["技", "术", "原", "理"]
target = ["技", "术", "细", "节", "补"]

accepted, fallback = speculative_step(draft, target)
assert accepted == ["技", "术"]
assert fallback == "细"

old_len = cache.lengths[seq_id]
for token in accepted:
    cache.append_token(seq_id, token)

assert cache.lengths[seq_id] == old_len + 2
cache.truncate(seq_id, 5)
assert cache.get_tokens(seq_id) == ["我", "想", "学", "推", "理"]
```

这段代码对应的工程含义如下。

`PagedKVCache` 里最关键的数据结构是 `block_tables`。它记录每个请求用了哪些物理块。真实系统里块内不是字符串 token，而是每层的 `K/V tensor`。但管理逻辑是一致的：按块追加，按块回收，必要时截断。

`truncate` 在 speculative decoding 里尤其重要。因为草稿模型可能先生成了多个候选，但只有一部分被目标模型接受，所以缓存必须回退到“最后一个已确认 token”的位置。否则缓存内容会和真实输出不一致。

如果把这个玩具实现映射到真实推理引擎，可以得到更接近生产环境的伪代码：

```python
def serve_one_round(seq, draft_model, target_model, kv_cache, gamma):
    draft_tokens = draft_model.generate(seq, gamma, kv_cache="draft")
    verified = target_model.verify(seq, draft_tokens, kv_cache="target")

    accepted_prefix = longest_match_prefix(draft_tokens, verified.candidates)
    kv_cache.truncate(seq.id, seq.len + len(accepted_prefix))

    for tok in accepted_prefix:
        seq.append(tok)

    if len(accepted_prefix) < len(draft_tokens):
        seq.append(verified.next_token_from_target)

    return seq
```

真实工程例子可以想成 vLLM 或 TensorRT-LLM 这样的服务框架。它们不会在 Python 层逐 token 存字符串，而是在 GPU 上管理分页 KV block、调度多个请求、合并验证 batch，并把 attention kernel 改到能读取 block table 或等价映射。这也是为什么“概念简单，工程实现难”的原因：难点不在公式，而在显存布局、调度和 kernel 配合。

---

## 工程权衡与常见坑

第一类坑是“以为开了 KV Cache 就够了”。

这在单请求测试里经常成立，但在真实服务里不成立。原因是单请求只暴露计算问题，不暴露内存管理问题。一旦并发上来，请求长度差异变大，连续预分配会快速出现两件事：高浪费和高碎片。结果就是 GPU 利用率看起来不低，但可服务并发上不去。

第二类坑是 block size 选得不合适。

`block size` 可以理解为“每页装多少 token”。太小，页表更长，管理开销更高，kernel 访存更碎；太大，最后一页的尾部浪费变大，分页收益下降。工程上它不是越小越好，而是要在“元数据开销”和“尾部浪费”之间找平衡。

第三类坑是忽视 PagedAttention 的 kernel 成本。

分页管理本身并不自动带来最快的 attention。因为原本连续的 KV 现在变成逻辑连续、物理离散，kernel 要多做一次地址映射，访存模式也更复杂。如果内核没针对分页布局做优化，吞吐可能被额外开销吃掉。这也是为什么有些实现会引入 `vAttention` 一类方案，用虚拟内存保留连续视图。

第四类坑是 speculative decoding 只看“理论加速”，不看 acceptance。

决定 speculative 是否划算的不是它听起来多先进，而是草稿模型和目标模型在你的业务上是否足够一致。若 acceptance rate 很低，流程就会退化成：

- 草稿模型先做一遍
- 目标模型又来一遍
- 接受的 token 很少

这时你等于给每轮生成增加了额外工作。新人常见误区是把 speculative 当成“无条件加速器”，这不对，它本质上是一个有前提的优化。

第五类坑是缓存一致性。

在普通单模型生成里，缓存只需要一直追加；在 speculative、beam search、prefix sharing 等场景里，缓存可能需要分叉、回退、共享、复制。如果这些状态管理不严谨，会出现最难排查的一类错误：输出偶发错误，但模型本身没问题，真正出错的是 cache。

下面这个表格可以帮助判断几个常见工程现象：

| 现象 | 更可能的原因 | 优先排查 |
|---|---|---|
| 并发一高就 OOM | KV 预分配浪费、碎片严重 | 是否使用分页 KV |
| GPU 利用率不低但 tok/s 不高 | 仍是逐 token 目标模型验证 | 是否能上 speculative |
| 上了分页后反而慢 | kernel 不适配离散 block | attention 实现与访存模式 |
| 上了 speculative 收益不稳定 | acceptance 低或分布不均 | 分任务统计 acceptance |
| 输出偶发异常 | cache truncate / COW 出错 | 缓存状态机 |

一个真实工程例子是在线 API 服务。白天流量里短问答很多，晚上批量摘要很多。如果系统仍按固定最大上下文预留 KV，那么白天的请求大多在浪费显存，晚上又因为长序列把碎片放大。更合理的做法是：

- 基础层用 KV Cache 保证不重算前缀
- 显存管理层用 PagedAttention 把活跃 KV 压到接近真实用量
- 对高频、模式稳定的业务流量单独测 acceptance，再决定是否启用 speculative

这比“一上来就全局开启所有优化”更稳。

---

## 替代方案与适用边界

PagedAttention 不是唯一的分页思路。一个重要替代方案是 `vAttention`。它的白话解释是：底层仍用虚拟内存映射，让上层 kernel 看到的地址尽量像连续内存，这样可以少改已有 attention kernel。它适合“不想重写太多 kernel，但又需要低碎片率”的团队。

Speculative decoding 也不是只能依赖“单独草稿模型 + 目标模型”的双模型方案。还有几类常见变体：

| 方案 | 思路 | 优点 | 代价 | 适用边界 |
|---|---|---|---|---|
| Draft + Target | 小模型猜，大模型验 | 通用性强 | 双模型部署复杂 | 服务端低时延场景 |
| SWIFT / Self-Speculative | 同一模型内部跳层或门控猜测 | 不必维护两个模型 | 需要结构支持 | 模型可改造场景 |
| Medusa / ReDrafter | 给目标模型增加多 token 预测头 | 推进更直接 | 需要训练或微调 | 有训练资源的团队 |
| vAttention | 用虚拟内存优化 KV 布局 | 少改现有 kernel | 依赖底层能力 | 已有成熟 kernel 资产 |

怎么选，可以按瓶颈倒推。

如果你的瓶颈是“单卡显存不够，并发上不去”，优先级通常是 `KV Cache -> PagedAttention`。因为这是最直接的容量问题。

如果你的瓶颈是“单请求时延高，尤其是 decode 阶段慢”，应重点评估 speculative decoding。因为此时真正限制你的不是前缀是否重算，而是目标模型每轮只确认一个 token。

如果你的团队没有能力改底层内核，但又不接受分页带来的访存惩罚，那么 `vAttention` 一类路线更现实。

如果你的场景很小，比如单机离线生成、并发很低、上下文也不长，那么复杂分页和 speculative 可能都不值得。因为你引入的实现复杂度、调试成本、监控成本，可能比加速收益更大。

所以适用边界可以压缩成一句话：高并发先管内存，高时延先管推进长度，底层能力不足时优先选兼容现有 kernel 的方案。

---

## 参考资料

- FlashAttention, “KV Cache Explained”：[https://flashattn.dev/blog/kv-cache-explained](https://flashattn.dev/blog/kv-cache-explained)
- BentoML, “Speculative Decoding”：[https://bentoml.com/llm/inference-optimization/speculative-decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding)
- Emergent Mind, “PagedAttention”：[https://www.emergentmind.com/topics/pagedattention](https://www.emergentmind.com/topics/pagedattention)
- Emergent Mind, “PagedAttention and Fine-Grained Caching”：[https://www.emergentmind.com/topics/pagedattention-and-fine-grained-caching](https://www.emergentmind.com/topics/pagedattention-and-fine-grained-caching)
- NVIDIA Developer Blog, “An Introduction to Speculative Decoding for Reducing Latency in AI Inference”：[https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference)
- TensorRT-LLM Documentation, “Speculative Decoding”：[https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html)
- MarkTechPost, “vLLM vs TensorRT-LLM vs HF TGI vs LMDeploy”：[https://www.marktechpost.com/2025/11/19/vllm-vs-tensorrt-llm-vs-hf-tgi-vs-lmdeploy-a-deep-technical-comparison-for-production-llm-inference/](https://www.marktechpost.com/2025/11/19/vllm-vs-tensorrt-llm-vs-hf-tgi-vs-lmdeploy-a-deep-technical-comparison-for-production-llm-inference/)
- Introl, “Speculative Decoding LLM Inference Speedup Guide”：[https://introl.com/blog/speculative-decoding-llm-inference-speedup-guide-2025](https://introl.com/blog/speculative-decoding-llm-inference-speedup-guide-2025)
