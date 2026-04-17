## 核心结论

Decoder-only 架构可以理解为“只保留生成那一半的 Transformer”。它的核心做法是：输入先变成 token embedding 和 position embedding，再经过多层带因果掩码的自注意力与前馈网络，最后通过 softmax 预测下一个 token。这个过程同时完成两件事：一是形成上下文表示，二是执行生成，所以一个模型既能做续写，也能做很多理解类任务。

这里的“因果掩码”指一种遮挡规则，白话说就是“当前位置只能看左边，不能偷看右边”。因此模型在训练时必须老老实实根据历史 token 预测下一个 token，这就是 next-token prediction。它看起来只是生成目标，但实际上会逼着模型学习语法、事实、推理线索和任务模式，所以也能支持问答、分类、摘要、代码补全等任务。

Decoder-only 成为主流，不只是因为“能生成文本”，更因为它在工程上形成了闭环：

| 任务类型 | 传统理解 | Decoder-only 的统一方式 |
|---|---|---|
| 理解任务 | 通常抽取整句表示再接分类头 | 把任务写成文本条件，让模型继续生成答案 |
| 生成任务 | 原生支持 | 直接按下一个 token 生成 |
| 训练目标 | 常有多种预训练目标 | 统一成 next-token prediction |
| 部署形式 | 往往多头多模块 | 单一模型、单一推理链路 |

从训练规律看，Decoder-only 上已经积累了大量缩放实验数据。常见经验公式写成

$$
L(N)=\alpha N^{-p}+\beta
$$

其中 $L$ 是损失，白话说就是“模型错得有多厉害”；$N$ 是参数规模。这个式子表示：模型通常不是“突然变强”，而是随着参数和数据增加按幂律逐步下降。但这个规律只在观测区间内可靠，超出区间仍要重新拟合。

---

## 问题定义与边界

先把边界说清楚。Decoder-only 指的是只保留 decoder stack，也就是一串重复堆叠的“自注意力 + 前馈层 + 残差连接 + 归一化”模块，不包含单独的 encoder。它最适合“从左到右生成”的任务，因为每个位置只能读取左侧上下文。

训练目标可以写成：给定 token 序列 $x_1,x_2,\dots,x_T$，模型在每个位置预测下一个 token：

$$
P(x_2,\dots,x_T|x_1)=\prod_{t=1}^{T-1} P(x_{t+1}|x_{\le t})
$$

这就是“每一步都做局部预测，整体拼成全句概率”。白话说，模型不是一次性理解整段后再输出，而是不断回答“接下来最可能是什么”。

因果掩码的数学形式通常写成：

$$
M_{ij}=
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

含义很直接：如果第 $i$ 个位置想看第 $j$ 个位置，而 $j$ 在未来，就把分数减到负无穷，softmax 后权重变成 0。

可以用一个玩具例子理解。输入句子是：

`The cat sits on the`

训练时，模型在最后一个位置要预测的就是 `mat`、`chair`、`floor` 这类候选 token 的概率分布。它不会看到真实下一个词，只能根据左边上下文猜。这种训练会迫使模型学会“冠词后面常接名词”“cat 常和 sits 连在一起”“on the 后面常是地点名词”等统计规律。

如果把遮挡关系画成词流示意，可以理解为：

| 当前位置 | 能看到的 token | 看不到的 token |
|---|---|---|
| `The` | `The` | `cat sits on the` |
| `cat` | `The cat` | `sits on the` |
| `sits` | `The cat sits` | `on the` |
| `on` | `The cat sits on` | `the` |
| `the` | `The cat sits on the` | 未来词 |

这也是它的边界所在。它天然不具备“完整双向同时编码整句”的结构偏好，所以当任务特别依赖全局双向对齐时，别的架构可能更直接。

---

## 核心机制与推导

Decoder-only 的主体计算仍然是多头注意力。注意力可以理解为“每个 token 动态决定该参考哪些历史 token”。单头公式是：

$$
\text{head}_h=\text{softmax}\left(\frac{Q_hK_h^\top}{\sqrt{d_k}}+M\right)V_h
$$

其中 $Q,K,V$ 分别是 query、key、value。白话说，$Q$ 像“我要找什么”，$K$ 像“我这里有什么标签”，$V$ 像“真正要拿走的信息内容”。

假设一个两层小模型做玩具推导：

- hidden size $d=8$
- head 数 $h=2$
- 每个 head 的维度 $d_k=d_v=4$
- 序列长度 $T=3$

那么输入矩阵 $X\in \mathbb{R}^{3\times 8}$。对第 1 个 head，有：

- $W_Q^{(1)}\in \mathbb{R}^{8\times 4}$
- $W_K^{(1)}\in \mathbb{R}^{8\times 4}$
- $W_V^{(1)}\in \mathbb{R}^{8\times 4}$

于是得到：

- $Q^{(1)}=XW_Q^{(1)}\in \mathbb{R}^{3\times 4}$
- $K^{(1)}=XW_K^{(1)}\in \mathbb{R}^{3\times 4}$
- $V^{(1)}=XW_V^{(1)}\in \mathbb{R}^{3\times 4}$

再算 $QK^\top$，得到 $3\times 3$ 的相似度矩阵，表示“3 个位置彼此关注的强度”。加上因果掩码后，右上角未来位置被屏蔽。softmax 后每一行的权重和为 1，再乘 $V$，得到该 head 的输出。多个 head 拼接后再过一次输出投影矩阵 $W_O$ 回到 $d=8$ 维，然后做残差连接。所谓“残差连接”，白话说就是“原输入别丢掉，新信息在原表示上增量修正”。

随后是 FFN，前馈网络可以理解为“每个 token 自己单独做一次更强的非线性变换”。LLaMA 系列常用 SwiGLU，它比普通 ReLU FFN 表达能力更强。

更完整的缩放规律常写成：

$$
L(N,D)=E+\frac{a}{N^\alpha}+\frac{b}{D^\beta}
$$

这里 $D$ 是数据量，$E$ 可以理解为当前任务和数据分布决定的下界项。这个式子表达的不是“参数越大一定无限变好”，而是“参数和数据都重要，缺一边都会卡住”。

以 LLaMA-2-7B 为例，可以把典型配置记成下面这张表：

| 项目 | 数值 | 含义 |
|---|---|---|
| 层数 | 32 | decoder block 重复 32 次 |
| hidden size | 4096 | 每个 token 的主表示维度 |
| attention heads | 32 | 注意力并行头数 |
| GQA 组数 | 4 | 多个 query 共享较少的 KV 头，减少缓存 |
| FFN 隐层 | 11008 | SwiGLU 中间层宽度，约 $2.69\times4096$ |
| 词表 | 32000 | token 种类数 |
| 上下文长度 | 4096 | 一次最多处理的 token 数 |
| 总参数 | 约 6.7B | 约 67 亿参数 |

工程上还会关心 FLOPs 和 KV cache。KV cache 指“把历史 token 的 key 和 value 存起来，下一个 token 直接复用”。对白话理解，它像做阅读理解时把前面每句话的索引卡片放在手边，不必每次从头整理。

粗略估算一层 KV cache 的元素数：

$$
\text{KV elements per layer} = 2 \times T \times n_{kv} \times d_{head}
$$

其中 2 表示 K 和 V 两份，$T$ 是序列长度，$n_{kv}$ 是 KV 头数。总缓存再乘层数和每个元素字节数即可。GQA 的意义之一就是把 $n_{kv}$ 降下来，显著省内存。

真实工程例子是在线对话服务。用户发来 1500 token 的长提示词时，系统先做 prefill，把整段 prompt 并行跑完，得到整条上下文的 KV cache；然后进入 decode，每次只生成 1 个新 token，并把新 token 的 K/V 追加到缓存。前者偏算力密集，后者偏带宽密集，两者瓶颈完全不同。

---

## 代码实现

下面用一个极简 Python 例子展示“shift 后做 next-token loss”的核心逻辑。它不是完整神经网络，但可以运行，并验证训练标签确实是“向右错一位”。

```python
import math

def causal_mask(seq_len: int):
    mask = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            row.append(0.0 if j <= i else float("-inf"))
        mask.append(row)
    return mask

def shift_for_next_token(tokens):
    # 输入 [x1, x2, x3, x4]
    # 训练时用 [x1, x2, x3] 预测 [x2, x3, x4]
    return tokens[:-1], tokens[1:]

def cross_entropy_from_probs(probs, targets):
    loss = 0.0
    for p, t in zip(probs, targets):
        loss -= math.log(p[t])
    return loss / len(targets)

tokens = [10, 20, 30, 40]
x, y = shift_for_next_token(tokens)

assert x == [10, 20, 30]
assert y == [20, 30, 40]

mask = causal_mask(4)
assert mask[0][0] == 0.0
assert mask[0][1] == float("-inf")
assert mask[3][2] == 0.0

# 假设模型对 3 个位置分别给出 3 类概率
probs = [
    [0.1, 0.7, 0.2],  # 位置1预测目标类别1
    [0.2, 0.2, 0.6],  # 位置2预测目标类别2
]
targets = [1, 2]
loss = cross_entropy_from_probs(probs, targets)

assert 0 < loss < 1.0
print("ok, loss =", round(loss, 4))
```

完整流程的伪代码可以写成：

```python
def decoder_only_forward(input_ids, kv_cache=None):
    x = token_embedding(input_ids) + position_embedding(input_ids)

    new_cache = []
    for layer_idx, layer in enumerate(layers):
        residual = x
        x = layer_norm_1(x)

        attn_out, layer_cache = masked_multi_head_attention(
            x,
            past_kv=None if kv_cache is None else kv_cache[layer_idx]
        )
        x = residual + attn_out

        residual = x
        x = layer_norm_2(x)
        x = residual + feed_forward(x)

        new_cache.append(layer_cache)

    logits = lm_head(x)
    return logits, new_cache
```

KV cache 的更新逻辑重点在“只为新 token 算一次 K/V，再接到历史后面”：

```python
def update_kv_cache(past_k, past_v, new_x):
    new_k = project_to_k(new_x)   # shape: [1, n_kv_heads, d_head]
    new_v = project_to_v(new_x)   # shape: [1, n_kv_heads, d_head]

    if past_k is None:
        k = new_k
        v = new_v
    else:
        k = concat_along_time(past_k, new_k)
        v = concat_along_time(past_v, new_v)

    return k, v
```

可以把 prefill 和 decode 的区别总结成：

| 阶段 | 输入特点 | 并行度 | 主要收益 | 主要瓶颈 |
|---|---|---|---|---|
| Prefill | 整段 prompt | 高 | 一次性建立全部历史 KV | 算力、矩阵乘 |
| Decode | 每次 1 个 token | 低 | 复用历史 KV，避免重算 | 显存带宽、缓存读写 |

这也是“为什么有缓存后每步近似是增量 O(1)”的直观来源。严格地说，总成本仍随序列长度增长，因为要和更长的历史做注意力，但不会重复重算历史 token 的 K/V 与中间表示，实际增量代价远小于从头再跑一遍。

---

## 工程权衡与常见坑

生产环境里最容易被忽视的一点，不是模型结构本身，而是推理阶段被拆成两种完全不同的工作负载。

Prefill 处理整个 prompt，矩阵乘法大，GPU 算力容易吃满；Decode 每次只吐 1 个 token，虽然计算量小，但需要频繁读取每层历史 KV cache，因此常常变成 memory-bound，也就是“不是算不动，而是数据搬运跟不上”。

经验上，prefill 吞吐可以到每秒 500 到 2000 token，而 decode 单流常只有每秒 20 到 100 token。也就是说，预填充阶段吞吐大约是单 token 解码的 10 到 20 倍。这个差距说明：如果把两阶段混在同一套调度里，资源利用很容易失衡。

| 问题 | 典型表现 | 根因 | 常见缓解策略 |
|---|---|---|---|
| Prefill 堵塞 | 长 prompt 请求拖慢整机 | compute-bound | 分离 prefill/decode，批量化 prompt |
| Decode 变慢 | 长对话越聊越卡 | KV cache 线性增长，带宽吃紧 | GQA、分页缓存、滑动窗口 |
| 显存不够 | 并发一上来就 OOM | 每请求都带一份历史缓存 | chunked prefill、cache 压缩、分层卸载 |
| 延迟抖动 | 99p 很高 | 长短请求混部争资源 | disaggregated serving 独立调度 |

KV cache 的内存复杂度可以粗略写成：

$$
\text{Memory} \propto L \times T \times n_{kv} \times d_{head}
$$

其中 $L$ 是层数，$T$ 是当前上下文长度。它对 token 数是线性增长的，所以长上下文、多轮对话、高并发三件事叠加时，内存压力会很快出现。

一个常见坑是“只盯单 token 延迟，不看阶段结构”。如果业务主要是长 prompt 短输出，例如文档问答、代码库分析，瓶颈往往在 prefill；如果业务是短 prompt 长输出，例如创作、陪伴式对话，瓶颈往往在 decode。部署方案必须按流量结构选，而不是只看模型参数量。

另一个坑是误用 Scaling Law。幂律关系不是承诺书，而是经验拟合。跨语言、跨领域、跨 tokenizer 设置后，指数项可能明显变化。放大训练前最好先做小模型试验，而不是直接假设“大十倍一定划算”。

---

## 替代方案与适用边界

Decoder-only 并不是唯一正确答案，它只是当前通用文本生成场景里最有规模优势的答案。

| 架构 | 强项 | 弱项 | 更适合的任务 |
|---|---|---|---|
| Decoder-only | 统一生成与理解，部署简单 | 天然是单向上下文 | 对话、写作、代码生成、统一指令模型 |
| Encoder-Decoder | 输入编码和输出生成分工明确 | 系统更复杂 | 翻译、摘要、结构化生成 |
| Encoder-only | 双向表示强 | 不擅长自回归生成 | 检索、分类、匹配、序列标注 |

为什么“需要大量双向 context”时别的架构可能更适合？例如问答检索中的 reranker，要判断查询和候选文档是否高度匹配。这个任务核心不是一步步生成，而是“同时看两边，再做细粒度交互”。Encoder-only 模型天然能双向编码整段输入，往往更适合做高质量匹配表示。再例如机器翻译，encoder-decoder 可以把源句完整压成表示，再专门生成目标句，结构上更贴合“先理解输入，再输出另一种序列”。

但如果你的目标是统一接口、统一训练、统一部署，让模型既能聊天，又能写代码，还能按提示做抽取和分类，Decoder-only 通常更有工程优势。它的任务适配方式简单：把任务写成文本提示即可，不必额外设计太多任务头。

关于 Scaling Law，一个务实的小实验设计是：

1. 固定 tokenizer 和数据清洗流程。
2. 训练 3 到 4 个小模型，比如 100M、300M、700M、1.3B。
3. 每个规模下再改变数据量，观察验证集交叉熵。
4. 拟合 $\alpha,\beta$ 后，再决定下一档参数和数据预算。

这样做的原因很简单：你要先知道“当前数据分布下，究竟是缺模型还是缺数据”，否则扩容方向可能错。

---

## 参考资料

1. EmergentMind, *Decoder-Only Transformers*  
   重点：Decoder-only 的基本结构、因果掩码、next-token prediction 如何统一理解与生成。  
   链接：https://www.emergentmind.com/topics/decoder-only-transformers-8d3ea15e-255d-46ad-96ac-dbee4d7b246a

2. EmergentMind, *Decoder-Only Scaling Laws*  
   重点：损失随参数量与数据量的幂律变化，为什么要区分观测区间内拟合与区间外外推。  
   可用于核对文中 $L(N)=\alpha N^{-p}+\beta$ 与 $L(N,D)=E+\frac{a}{N^\alpha}+\frac{b}{D^\beta}$ 的背景。

3. EmergentMind, *LLaMA-2-7B*  
   重点：LLaMA-2-7B 的层数、hidden size、head 数、FFN 尺寸、词表和上下文长度。  
   文中的配置数字应以原文和模型发布资料为准，尤其是不同实现里的 GQA/头部细节。

4. EmergentMind, *Prefill-Decode Disaggregation Architecture*  
   重点：推理服务中 prefill 与 decode 的负载差异，以及为何独立调度两阶段可以提升吞吐并降低尾延迟。  
   如果要验证“prefill 500–2000 token/s、decode 20–100 token/s”，应先看测试场景是否是单流、多流、哪种 GPU、多少 batch，再比较数字，不能脱离上下文直接横比。

5. MLJourney / Medium, *What Is KV Cache and Why It Affects LLM Speed*  
   重点：KV cache 的作用、为何能避免重复计算、以及为什么长序列 decode 会逐渐受内存带宽限制。  
   这类文章适合建立工程直觉，但具体实现仍应与框架源码和部署系统文档交叉验证。
