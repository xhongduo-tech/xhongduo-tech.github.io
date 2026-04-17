## 核心结论

KV Cache 是 Transformer 推理阶段保存历史注意力结果的缓存区。白话说，它把“前面已经算过的 token 信息”先存起来，后面生成新 token 时直接读，不再整段重算。

它解决的是推理速度问题，但代价是显存持续增长。对大语言模型来说，KV Cache 的核心内存公式通常可写为：

$$
\text{KV\_bytes} \approx 2 \times L \times B \times S \times H_{kv} \times D \times \text{dtype\_bytes}
$$

其中：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $2$ | K 和 V 两份 | 每个 token 要同时存 Key 和 Value |
| $L$ | layer 数 | 模型有多少层，就要存多少层的历史 |
| $B$ | batch size | 一次并行处理多少请求 |
| $S$ | sequence length | 当前上下文有多少 token |
| $H_{kv}$ | KV 头数 | 真正需要缓存的注意力头数 |
| $D$ | head dimension | 每个头的向量长度 |
| dtype\_bytes | 每个元素字节数 | FP16/BF16 通常是 2 字节 |

如果模型采用普通多头注意力，且 $H_{kv} \times D = d_{model}$，也常写成：

$$
\text{KV\_bytes} \approx 2 \times L \times B \times S \times d_{model} \times \text{dtype\_bytes}
$$

结论非常直接：KV Cache 对上下文长度、批大小、层数近似线性增长。上下文翻 4 倍，KV Cache 也大致翻 4 倍；并发翻 4 倍，KV Cache 也近似翻 4 倍。实际部署里，长上下文场景常常不是先卡模型权重，而是先卡 KV Cache。

一个常见估算例子是 7B 级模型，32 层、32 个 KV 头、head\_dim=128、FP16、单请求 4K 上下文：

$$
2 \times 32 \times 1 \times 4096 \times 32 \times 128 \times 2
= 2{,}147{,}483{,}648 \text{ bytes}
\approx 2 \text{ GB}
$$

这说明“只有一个请求、只有 4K 上下文”时，KV Cache 就可能已经到 2GB 量级。很多初学者第一次 OOM，不是因为模型太大，而是因为没有先算这块缓存。

---

## 问题定义与边界

问题定义可以压缩成一句话：每生成一个新 token，模型都要把这个 token 在每一层产生的 K 和 V 追加到缓存里，所以缓存会随着对话变长而不断膨胀。

这里先解释几个术语。

注意力机制是模型在“当前 token 和历史 token 之间建立关联”的计算方式。KV Cache 则是注意力机制的推理优化版本，专门为了避免重复计算历史部分。

这个问题有几个明确边界。

第一，它主要讨论的是推理，不是训练。训练时通常会保留更多激活值，内存结构不同；而 KV Cache 是推理中最关键的“历史记忆区”。

第二，它讨论的是显存占用，不是磁盘占用。模型权重落盘多大，和运行时的 KV Cache 是两回事。很多人看见“8B 模型量化后只有 5GB”，就以为 24GB 显卡一定能轻松跑超长上下文，这个判断经常错。

第三，它和上下文长度强相关。假设权重已经固定，输入从 4K 增到 32K，不是“多一点显存”，而是近似按 8 倍放大 KV Cache。对部署者来说，这通常比“模型从 7B 换 13B”更容易突然触发 OOM。

下面给一个面向初学者的对比表，仍用上面的 7B 近似配置，FP16，batch=1：

| 上下文长度 | 估算 KV Cache |
|---|---:|
| 1K | 约 0.5 GB |
| 4K | 约 2 GB |
| 8K | 约 4 GB |
| 32K | 约 16 GB |
| 128K | 约 64 GB |

这张表的意义不是给出所有模型的精确值，而是建立一个判断习惯：长上下文的主要代价是缓存，不只是“前处理变慢”。

一个玩具例子是这样的。假设你在纸上做表格记录历史：第一轮只记 10 行，第二轮记 100 行，第三轮记 1000 行。每增加一行，后续查表会更快，因为不用重算；但表本身会越来越大。KV Cache 本质上就是这个“不断追加的历史表”，只不过每一行是每层、每头的向量。

一个真实工程例子更能说明边界。有人在 24GB 的消费级 GPU 上部署 8B 模型，发现权重量化后只占几 GB，于是把 `max_model_len` 直接拉到 128K，再开多个并发请求。结果不是模型加载失败，而是运行一段时间后显存被 KV Cache 吃满。这类事故的根因通常不是“模型太大”，而是“上下文和并发预算根本没算”。

---

## 核心机制与推导

Transformer 的自注意力里，每个 token 会生成 Query、Key、Value 三组向量。白话说，Query 是“我现在要找什么”，Key 是“我能提供什么索引”，Value 是“我真正携带的内容”。

推理时，历史 token 的 Query 没必要再用，但历史 token 的 Key 和 Value 后面还要不断参与新 token 的注意力计算，所以要缓存下来。这就是 KV Cache 的来源。

先看单层、单 token 的存储量。对某一层而言，一个 token 需要保存：

$$
2 \times H_{kv} \times D \times \text{dtype\_bytes}
$$

原因很简单：

- `2` 是 K 和 V 两份
- 每份有 $H_{kv}$ 个头
- 每个头长度是 $D$
- 每个元素占 `dtype_bytes` 字节

再乘上层数 $L$，得到“单 token 全模型缓存量”：

$$
\text{bytes\_per\_token} = 2 \times L \times H_{kv} \times D \times \text{dtype\_bytes}
$$

再乘上下文长度 $S$，得到“单请求总缓存量”：

$$
\text{bytes\_per\_request} = 2 \times L \times S \times H_{kv} \times D \times \text{dtype\_bytes}
$$

最后再乘 batch 大小 $B$，得到“批量总缓存量”：

$$
\text{KV\_bytes} = 2 \times L \times B \times S \times H_{kv} \times D \times \text{dtype\_bytes}
$$

这个推导最重要的不是公式本身，而是它说明了增长路径：

`每 token` → `每请求` → `批量并发`

也就是说，KV Cache 不是一次性分配后不变，而是随着 token 生成持续追加。预填充阶段也叫 prefill，意思是先把整段输入上下文跑一遍并建立缓存；解码阶段也叫 decode，意思是每次生成一个 token，同时继续向缓存追加一行。

再看一个更具体的数值推导。假设某模型配置为 80 层、8 个 KV 头、head\_dim=128、BF16，每元素 2 字节。那么单 token 缓存量是：

$$
2 \times 80 \times 8 \times 128 \times 2 = 327{,}680 \text{ bytes}
$$

约等于 0.3125 MB。若上下文是 4K，batch=1，则总缓存约为：

$$
327{,}680 \times 4096 \approx 1.25 \text{ GB}
$$

这个量级已经足够说明问题：即使是单请求，长上下文也会快速吃掉显存。

这里还有一个容易忽略的点：公式里的 $H_{kv}$ 不一定等于注意力总头数。很多现代模型使用 GQA，Grouped Query Attention，意思是多个 Query 头共享一组较少的 KV 头。白话说，查询头很多，但真正要缓存的 K/V 头更少。因此估算时必须看 KV 头数，而不是总头数。把这两个数混了，结果会直接差几倍。

---

## 代码实现

最实用的做法不是死记公式，而是把它写成一个估算函数，在部署前先跑一遍预算。

```python
def compute_kv_bytes(layers, kv_heads, head_dim, seq_len, batch, bytes_per_elem):
    return 2 * layers * kv_heads * head_dim * seq_len * batch * bytes_per_elem

def to_gib(num_bytes):
    return num_bytes / (1024 ** 3)

# 玩具例子：7B 风格配置，32层、32 KV头、head_dim=128、FP16、4K上下文、batch=1
kv_bytes_7b_4k = compute_kv_bytes(
    layers=32,
    kv_heads=32,
    head_dim=128,
    seq_len=4096,
    batch=1,
    bytes_per_elem=2,
)

kv_gib_7b_4k = to_gib(kv_bytes_7b_4k)
print(round(kv_gib_7b_4k, 3))  # 2.0

assert kv_bytes_7b_4k == 2147483648
assert abs(kv_gib_7b_4k - 2.0) < 1e-9

# 真实工程例子：Llama 3.1 70B 常见近似配置，80层、8 KV头、head_dim=128、FP16
# 32K上下文、8并发
kv_bytes_70b_32k_b8 = compute_kv_bytes(
    layers=80,
    kv_heads=8,
    head_dim=128,
    seq_len=32768,
    batch=8,
    bytes_per_elem=2,
)

kv_gib_70b_32k_b8 = to_gib(kv_bytes_70b_32k_b8)
print(round(kv_gib_70b_32k_b8, 2))  # 40.0

assert abs(kv_gib_70b_32k_b8 - 40.0) < 1e-9
```

这段代码可以直接运行。它做了两件事：

1. 把公式写成统一函数，避免手算出错。
2. 用 `assert` 固定关键样例，防止后续改参数时把单位或公式改坏。

需要注意，真实部署里常见文章会给出约 40GB 到 43GB 的范围，这种差异通常来自几个因素：

- 是否按 GiB 还是 GB 计算
- 模型配置是否完全一致
- 运行时是否有分页、对齐、额外元数据开销
- 某些框架是否为管理方便预留额外缓存块

也就是说，公式适合做容量级估算，而不是替代 profiler 做字节级精确核算。

如果要把它接进实际部署脚本，通常会这样用：

- 从模型配置里读取 `num_hidden_layers`
- 读取 `num_key_value_heads`
- 读取 `head_dim`
- 根据 `torch_dtype` 映射出 `bytes_per_elem`
- 根据服务设置的 `max_model_len` 和 `max_num_seqs` 估算上限
- 若预算超出显存，则主动下调上下文或并发

这类“加载前预估”比 OOM 后再排查更有工程价值。

---

## 工程权衡与常见坑

部署阶段最常见的误判是只盯着模型权重，不盯 KV Cache。模型权重是静态成本，KV Cache 是随上下文和并发增长的动态成本。长对话、多用户、长 system prompt，这三件事叠加后，缓存经常比权重更先成为瓶颈。

下面给一个工程视角的简表：

| 场景 | 上下文 | 并发 | KV Cache 变化趋势 | 风险 |
|---|---:|---:|---|---|
| 单用户调试 | 4K | 1 | 基线 | 通常安全 |
| 长文问答 | 32K | 1 | 约为 4K 的 8 倍 | 先卡缓存 |
| 在线服务 | 8K | 8 | 约为单用户的 8 倍 | 峰值显存高 |
| 长上下文并发 | 32K | 8 | 同时受上下文和并发放大 | 极易 OOM |

几个典型坑最值得单独指出。

第一，错误使用总头数代替 KV 头数。对 GQA 模型，这会明显高估或低估，取决于你拿的是哪个配置字段。估算前应先确认是 `num_attention_heads` 还是 `num_key_value_heads`。

第二，忽略数据类型。FP16 和 BF16 一般都是 2 字节，FP8 是 1 字节左右的量级感知。只改了权重量化，不代表 KV Cache 也自动等比例下降。很多框架中，权重和 KV 的 dtype 可以不同。

第三，忽略 batch 与并发的线性关系。服务端的“8 个并发用户”在很多实现里并不等于操作系统层面的 8 个连接，而是推理引擎同时维护 8 份活动序列。对显存来说，这通常就意味着近似 8 倍的缓存占用。

第四，没有给上下文长度设硬上限。没有上限时，问题不是“偶尔慢一点”，而是某个超长请求可能直接把在线服务拖死。

一个真实工程例子是 70B 级模型做 32K 长上下文、8 路并发。即使按较乐观估算，KV Cache 也已经在 40GB 量级；再加模型权重、运行时工作区、采样缓冲和碎片化，80GB 卡也不一定宽松。这个案例说明，部署设计的第一步不该是“能不能加载模型”，而该是“目标上下文和并发下，KV 预算是多少”。

---

## 替代方案与适用边界

既然问题是 KV Cache 线性膨胀，优化方向也就比较清楚：减少要缓存的内容，或减少缓存的精度，或提升缓存分配方式。

先看常见方案：

| 方案 | 核心思路 | 最适合场景 | 限制 |
|---|---|---|---|
| GQA | 降低 KV 头数 | 新模型架构设计 | 需要模型本身支持 |
| MQA | 多个查询头共享单组 KV | 极致压缩 KV | 表达能力可能受约束 |
| FP8 KV Cache | 降低缓存元素字节数 | 显存紧张的在线服务 | 精度和实现兼容性要验证 |
| PagedAttention | 分页管理缓存 | 长上下文、动态并发 | 依赖特定推理框架 |
| Prefix Caching | 复用公共前缀 | system prompt 很长且重复 | 仅对共享前缀有效 |
| 截断上下文 | 强制缩短输入 | 资源硬约束环境 | 可能损失信息 |
| 检索增强 | 外部召回替代超长上下文 | 知识库问答 | 引入检索系统复杂度 |

GQA 可以理解成“减少真正要存的 K/V 份数”。这通常是最干净的方案，因为它直接从模型结构上降低缓存成本。MQA 更激进，相当于进一步把 KV 共享到更少的头。

KV Cache 量化则是“每个元素少占几个字节”。比如从 FP16 降到 FP8，理论上缓存体积可接近减半。若某模型 32K 上下文的 KV Cache 约 86GB，那么仅从字节数角度看，切到 FP8 后就可能降到约 43GB 量级。当然，这只是容量近似，不代表零成本；精度损失和框架支持需要单独验证。

PagedAttention 是“分页缓存管理”。白话说，它不再要求每个请求都拿到一整块连续大显存，而是把 KV 切成更小页面按需管理，更适合多用户动态服务。Prefix Cache 则是“前缀复用”，如果大量请求都有同一个 system prompt 或模板前缀，就不必每次都从头构建同样的缓存。

这些方案的适用边界需要讲清楚：

- 如果模型本身不支持 GQA/MQA，部署层做不到凭空减少 KV 头数。
- 如果业务对输出稳定性很敏感，KV 量化需要先做质量回归。
- 如果请求之间前缀不共享，Prefix Cache 收益会很有限。
- 如果真实需求只是“少量请求偶尔长一点”，最便宜的方法常常不是上复杂优化，而是直接限制 `max_model_len`。

工程上最稳妥的顺序通常是：

1. 先用公式算清楚理论上限。
2. 再限制上下文和并发，拿到可运行基线。
3. 然后按需要引入 FP8 KV、PagedAttention、Prefix Cache 等优化。
4. 最后才考虑是否必须换更大显卡或多卡方案。

---

## 参考资料

- DeepWiki, “KV Cache Memory Calculator”: https://deepwiki.com/ModelEngine-Group/unified-cache-management/9.1-kv-cache-memory-calculator?utm_source=openai
- ML Journey, “What Is KV Cache and Why It Affects LLM Speed”: https://mljourney.com/what-is-kv-cache-and-why-it-affects-llm-speed/?utm_source=openai
- InsiderLLM, “KV Cache: Why Context Length Eats Your VRAM”: https://insiderllm.com/guides/kv-cache-optimization-guide/?utm_source=openai
- Spheron, “KV Cache Optimization Guide”: https://www.spheron.network/blog/kv-cache-optimization-guide/?utm_source=openai
- Hugging Face, “KV Cache Basics”: https://huggingface.co/blog/atharv6f/kv-cache-basics?utm_source=openai
