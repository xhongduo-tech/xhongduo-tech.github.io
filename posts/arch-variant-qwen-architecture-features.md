## 核心结论

Qwen 系列的核心架构并不是“重新发明 Transformer”，而是在标准 decoder-only Transformer 上，针对三个真实瓶颈做了连续演进：一是用 GQA，分组查询注意力，白话说就是“很多 query 头共用更少的 key/value 头”，直接降低 KV cache 开销；二是在 Q/K 投影后引入常被概括为 QK-Norm 的归一化思路，白话说就是“先把注意力输入的尺度压稳，再去算分数”，减少长上下文和多语种训练中的数值失控；三是配合 YaRN 和更大的词表，把可用上下文从 32K 扩到 128K，进一步延展到 1M 级别的推理场景。

如果只看 Qwen2.5-7B 的公开配置，它已经能代表这一设计方向：28 层、hidden size 3584、28 个注意力头、4 个 KV 头、FFN 隐层 18944、词表 152064，基础上下文 32K，并通过 YaRN 支持到 128K。这个组合的重点不是“参数表好看”，而是让一个 7B 级开源模型在长文本、代码、多语种任务里更稳，更省显存，也更接近生产可用。

| 组件 | 白话解释 | 解决的问题 | 对 Qwen 的意义 |
| --- | --- | --- | --- |
| GQA | 多个 query 头共用少量 KV 头 | KV cache 太大、推理显存高 | 长上下文推理更省内存 |
| QK-Norm | 先把 Q/K 尺度归一再做注意力 | logit 爆炸、softmax 饱和 | 多语种和长上下文更稳 |
| YaRN | 对 RoPE 长度外推做插值缩放 | 训练 32K，推理想跑 128K/1M | 长度扩展更可控 |
| 152K 词表 | 更大的子词表 | 中文和代码分词碎片多 | 压缩率更高，token 更省 |
| Dual Chunk Attention | 分块算长注意力 | 1M 上下文全注意力太贵 | 超长上下文更快 |

---

## 问题定义与边界

Qwen 的架构优化主要针对两个边界条件。

第一，多语种混合训练。多语种训练不是简单“数据变多”，而是分布变杂。中文、英文、日文、韩文、代码，它们的 token 频率、长度分布、局部模式都不同，导致不同 batch 中的 $q^\top k$ 方差波动更大。注意力分数一旦波动过大，softmax 就会很快变尖，模型几乎只盯住极少数 token，训练梯度就容易抖。

第二，超长上下文推理。标准注意力的计算复杂度近似是 $O(n^2)$，而 KV cache 的显存占用近似随序列长度线性增长。序列从 8K 拉到 128K，再到 1M，问题不只是“慢”，还包括数值外推、位置编码失真、激活内存爆炸、chunk prefill 调度困难。

Qwen2.5-7B 的一个最小配置例子可以帮助理解边界：

| 项目 | Qwen2.5-7B 公开配置 |
| --- | --- |
| 层数 | 28 |
| hidden size | 3584 |
| 注意力头数 | 28 |
| KV 头数 | 4 |
| head dim | $3584 / 28 = 128$ |
| FFN 隐层 | 18944 |
| 词表大小 | 152064 |
| 基础最大位置 | 32768 |
| 滑窗参数 | 131072 |
| 长度扩展方式 | YaRN |

玩具例子：假设一个 head 的维度是 128，如果未归一化的 $q$ 和 $k$ 每个分量都偏大，那么点积会累加 128 项。哪怕每项平均只有 4，最后也会得到几百量级的 logit。softmax 在这种输入下几乎退化成“只选一个位置”，训练和推理都会变脆。QK-Norm 的目标就是先把这个量级压回可控范围。

真实工程例子：做仓库级代码问答时，用户可能一次性上传 20 万到 100 万 token 的代码、README、设计文档和 issue。此时瓶颈不是单步生成，而是 prefill，也就是“把整段上下文先喂进去”的阶段。Qwen2.5-1M 相关公开资料里强调的 chunked prefill、稀疏注意力和 DCA，本质都在解决这个问题。

---

## 核心机制与推导

先看 QK-Norm。这里“Norm”指归一化，白话说就是“先把向量长度拉回统一量级，再参与计算”。常见写法是对每个 head 的 $q,k$ 分别做 RMS 归一化，再乘上可学习缩放参数：

$$
\tilde q=\frac{q}{\sqrt{\frac{1}{d}\sum q^2+\varepsilon}}\odot g_q,\qquad
\tilde k=\frac{k}{\sqrt{\frac{1}{d}\sum k^2+\varepsilon}}\odot g_k
$$

然后再做旋转位置编码 RoPE，并计算注意力分数：

$$
L_{ij}=\frac{\mathrm{RoPE}(\tilde q_i)\mathrm{RoPE}(\tilde k_j)^\top}{\sqrt{d_h}}
$$

这里有三个效果。

第一，限制尺度。RMSNorm 不关心均值，只关心均方根，适合直接控制向量幅值。  
第二，引入可学习温度。$g_q,g_k$ 不是固定常数，而是可训练的缩放向量，模型可以自己学“该放大多少”。  
第三，减弱长序列外推时的尖峰。RoPE 会把位置信息混到 Q/K 里，如果原始幅值已经很大，旋转后的点积更容易出现极端值。

可以把这条流水线理解成：

`Q/K 线性投影 -> RMS 归一化 -> 可学习缩放 -> RoPE -> 点积 -> softmax`

再看 GQA。标准多头注意力里，每个 query 头都对应一个 key 头和一个 value 头。GQA 把多个 query 头映射到同一组 KV 头，相当于“Q 细分、KV 共享”。以 Qwen2.5-7B 为例，28 个 query 头只配 4 个 KV 头，KV cache 大约按 $4/28=1/7$ 的比例缩小。这对长上下文推理非常关键，因为推理期的主要显存压力常常不是参数本身，而是缓存的 K/V。

YaRN 则是长度外推方法。RoPE 原本在训练长度内表现稳定，但超出训练范围后，相对位置会进入模型没见过的区间。YaRN 的思想可以粗看成“重新映射位置尺度，让超长位置落在模型还能处理的频率范围内”。所以 Qwen2.5-7B 基础配置虽然是 32K，但可以通过 YaRN 支持到 128K。要注意，这不代表“训练过 128K 的全部分布”，而是说外推退化可接受。

Dual Chunk Attention 可以理解为长上下文版的分层注意力。chunk 就是块，白话说就是“把超长序列切成多个段”。块内做稠密注意力，保证局部信息充分交互；块间不是完全断开，而是通过特殊的跨块连接和位置重映射保留远程依赖。它的核心价值是：在不做完整 $O(n^2)$ 全注意力的前提下，尽量保住跨块检索能力。Qwen 的 1M 长上下文方案把它和稀疏注意力、chunked prefill 一起使用，属于“训练稳定性 + 推理效率”的联合设计。

---

## 代码实现

下面先给一个可运行的玩具实现，演示 QK-Norm 如何压缩点积尺度。这个例子只用 Python 标准库，不依赖第三方包。

```python
import math

def rms_norm(vec, eps=1e-6):
    rms = math.sqrt(sum(x * x for x in vec) / len(vec) + eps)
    return [x / rms for x in vec]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

# 模拟一个未归一化的 Q/K
q = [10.0, -8.0, 6.0, 4.0]
k = [9.0, -7.0, 5.0, 3.0]

raw_logit = dot(q, k) / math.sqrt(len(q))

# QK-Norm: 先归一化，再乘可学习尺度；这里先用 1.0 代替学习参数
q_norm = rms_norm(q)
k_norm = rms_norm(k)
norm_logit = dot(q_norm, k_norm) / math.sqrt(len(q_norm))

assert raw_logit > norm_logit
assert abs(dot(q_norm, q_norm) / len(q_norm) - 1.0) < 1e-4
assert abs(dot(k_norm, k_norm) / len(k_norm) - 1.0) < 1e-4

print("raw_logit =", round(raw_logit, 4))
print("norm_logit =", round(norm_logit, 4))
```

这个例子说明的不是“归一化后一定更准”，而是“归一化后注意力输入的尺度更可控”。这是训练稳定性的前提，不是性能的全部来源。

如果写成接近 PyTorch 的伪代码，QK-Norm 与 GQA 的核心 forward 可以写成下面这样：

```python
# q: [B, T, Hq, Dh]
# k,v: [B, T, Hkv, Dh]
# Hq > Hkv, 例如 28 vs 4

def qk_norm(x, gain, eps=1e-6):
    rms = sqrt(mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * gain

def attention_forward(x):
    q = Wq(x)
    k = Wk(x)
    v = Wv(x)

    q = reshape_heads(q, num_q_heads)
    k = reshape_heads(k, num_kv_heads)
    v = reshape_heads(v, num_kv_heads)

    q = qk_norm(q, gq)
    k = qk_norm(k, gk)

    q = apply_rope(q, position_ids)
    k = apply_rope(k, position_ids)

    # GQA: 扩展共享的 KV 到对应的 query head 组
    k = repeat_kv(k, num_q_heads // num_kv_heads)
    v = repeat_kv(v, num_q_heads // num_kv_heads)

    scores = matmul(q, transpose(k)) / sqrt(head_dim)
    probs = softmax(scores, dim=-1)
    out = matmul(probs, v)
    return merge_heads(out)
```

再看 Dual Chunk Attention 的调度思路。下面不是官方实现，而是帮助理解的数据流伪码：

```python
def dual_chunk_attention(hidden_states, chunk_size):
    chunks = split_into_chunks(hidden_states, chunk_size)
    outputs = []

    for i, chunk in enumerate(chunks):
        # 1. 块内全注意力，保住局部建模能力
        local_out = full_attention(chunk, chunk)

        # 2. 块间稀疏连接，只取部分历史块
        cross_context = select_sparse_history(chunks, current_index=i)

        # 3. 对跨块位置做重映射，避免超长距离直接进入注意力
        cross_out = sparse_cross_attention(
            query=chunk,
            key_value=cross_context,
            position_remap="DCA_or_YaRN"
        )

        outputs.append(local_out + cross_out)

    return concat(outputs)
```

真实工程例子：如果你要做“百万 token 的企业文档问答”，常见流程不是把 1M token 一次性做完整全注意力，而是：
1. prefill 阶段按 chunk 输入，减少激活峰值；
2. 块内保留 dense attention；
3. 块间用 DCA 或稀疏模式保留远距离检索；
4. 生成阶段继续使用 GQA 降低 KV cache 压力。  
这就是 Qwen 长上下文方案的工程意义，它不是单个技巧，而是一整套组合拳。

---

## 工程权衡与常见坑

Qwen 这一套设计并不是“无代价升级”，而是把复杂度从算力和显存，转移到训练稳定性和推理调度上。

| 常见坑 | 现象 | 原因 | 处理方式 |
| --- | --- | --- | --- |
| 不做 QK-Norm | softmax 过尖、训练抖动 | $q^\top k$ 方差过大 | 在 Q/K 投影后做 RMS 归一 |
| KV 头不降 | 长上下文显存爆 | KV cache 线性增长过快 | 用 GQA/MQA |
| 直接硬扩 RoPE | 超长长度精度掉得快 | 位置频率超出训练分布 | 用 YaRN 或其他外推方法 |
| chunk 太大 | prefill 慢、激活显存高 | 单块计算过重 | 调小 chunk size |
| chunk 太小 | 跨块信息断裂 | 稀疏连接不足 | 增加跨块连接或 refinement |
| 稀疏规则只在短序列调 | 到 1M 长度时精度崩 | 稀疏模式不泛化 | 做长序列专门 refinement |

这里最容易误解的一点是：QK-Norm 不是为了“让注意力值落到 [-1,1] 这个硬区间”，而是为了把尺度控制在稳定范围内。经过可学习缩放后，logit 仍然可以大于 1。工程上真正关心的是方差和梯度是否可控，而不是某个绝对边界。

另一个常见坑是把 Dual Chunk Attention 误解成“简单滑动窗口”。二者相似，但不等价。滑动窗口通常只看固定邻域，超出窗口的信息直接看不见；DCA 的目标是让跨块注意力仍然存在，只是用更便宜、更稳定的方式表达。因此它更适合文档检索、长代码仓分析、多轮长对话总结这类“远距离信息真的有用”的任务。

---

## 替代方案与适用边界

如果你的任务并不需要 128K 或 1M，上述复杂设计未必值得。

| 方案 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 标准 MHA + RoPE | 4K 到 8K、单语种、小模型 | 实现最简单 | 长上下文成本高 |
| MQA | 极端省 KV cache 的推理场景 | 显存最省 | 表达能力可能更受限 |
| GQA | 主流长上下文推理 | 性能和成本平衡好 | 仍需 careful tuning |
| 滑动窗口注意力 | 本地依赖明显的序列 | 算法直接 | 远距离信息容易丢 |
| 纯稀疏注意力 | 超长上下文、吞吐优先 | 速度快 | 稀疏模式难调 |
| Qwen 式组合栈 | 多语种 + 长上下文 + 工程部署 | 稳定性与效率兼顾 | 系统复杂度更高 |

新手可以用一个简单判断：

如果你只做 8K 聊天机器人，标准 Transformer 已经够用。  
如果你做 32K 到 128K 文档问答，GQA + YaRN 是很有价值的。  
如果你做 1M 级仓库分析、长小说理解、跨多份超长文档检索，才需要 DCA、稀疏 prefill、chunked pipeline 这一层工程体系。

还有一个准确性边界要说清：公开资料里，Qwen2.5-72B 在开源模型中表现很强，很多指标接近更大的 Llama-3.1-405B-Instruct；Qwen2.5-Turbo 和 Qwen2.5-Plus 的官方比较对象分别更接近 GPT-4o-mini 与 GPT-4o。把“Qwen2.5-72B 接近 GPT-4o”当作笼统口号并不严谨，最好按具体模型和 benchmark 分开说。

---

## 参考资料

| 分类 | 资料 | URL |
| --- | --- | --- |
| 官方博客 | Qwen2.5 总览，含 18T tokens、128K、多语种与整体 benchmark | https://qwenlm.github.io/blog/qwen2.5/ |
| 官方配置 | Qwen2.5-7B-Instruct 配置，含 28 层、3584 hidden、28/4 GQA、18944 FFN、152064 词表 | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json |
| 官方页面 | Qwen2.5 模型页，含 Qwen2.5-72B 与 Qwen2.5-72B-Instruct 评测表 | https://qwen2.org/qwen2-5/ |
| 官方长上下文博客 | Qwen2.5-Turbo 1M context、Passkey Retrieval、RULER、推理加速 | https://qwenlm.github.io/blog/qwen2.5-turbo/ |
| 官方开源长上下文博客 | Qwen2.5-1M，含 DCA、chunked prefill、3x 到 7x 加速说明 | https://qwenlm.github.io/blog/qwen2.5-1m/ |
| 技术报告 | Qwen2.5 Technical Report，总体训练和后训练设计 | https://huggingface.co/papers/2412.15115 |
| 框架文档 | Hugging Face Qwen2 文档，概述 GQA、RoPE、DCA、YaRN | https://huggingface.co/docs/transformers/en/model_doc/qwen2 |
| 早期词表背景 | Qwen 早期 tokenizer 设计，说明 150K+ 词表与中文、代码压缩导向 | https://huggingface.co/Qwen/Qwen-1_8B |
