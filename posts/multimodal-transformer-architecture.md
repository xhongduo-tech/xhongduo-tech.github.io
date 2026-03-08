## 核心结论

多模态 Transformer 的核心任务，不是“同时看图和读字”，而是把视觉和文本表示映射到同一语义空间，再决定它们在什么阶段发生信息交换。这里的“语义空间”指的是：图像和文本都被编码成向量后，向量越接近，通常表示语义越接近。

新手先记住一句话：视觉编码器把图像切成 token，语言编码器把句子切成 token，然后模型用三种主流方式让两种 token 发生交互。

第一种是早期融合：在输入层就把视觉 token 和文本 token 放进同一个 Transformer，一开始就联合建模。  
第二种是晚期融合：图像和文本先各自编码，最后只比较全局向量是否接近。  
第三种是中期融合：先分别编码，再在中间若干层通过 cross-attention 让文本去查询图像，或让图像去查询文本。

工程上没有一种融合方式对所有任务都最优。检索、分类、去重、召回，通常更适合晚期融合；视觉问答、文档理解、图像描述生成，通常需要中期融合；而把所有视觉 token 和文本 token 从第一层就完全混合的早期融合，表达力最强，但计算和显存压力也最大。

| 融合方式 | 融合位置 | 典型结构 | 优点 | 缺点 | 更适合的任务 |
| --- | --- | --- | --- | --- | --- |
| 早期融合 | 输入层 | 单一 Transformer 共同处理图文 token | 细粒度对齐最强，联合建模最直接 | $O((n_v+n_t)^2)$ 成本高，显存压力大 | 小规模联合理解、结构化场景、多模态推理 |
| 中期/混合融合 | 编码器内部 | 视觉编码器 + 语言模型 + cross-attention / adapter / query bridge | 对齐能力和效率平衡较好，易接入现有 LLM | 架构更复杂，训练与调参更难 | VQA、OCR 问答、图文生成、图表理解 |
| 晚期融合 | 输出层 | 双塔编码器 + 对比学习 | 训练和检索效率高，可大规模召回，向量索引友好 | 细节交互弱，对局部证据不敏感 | 图文检索、零样本分类、向量库召回 |

---

## 问题定义与边界

问题可以精确成一句话：给定图像 $x_v$ 和文本 $x_t$，学习两个编码函数 $f_v(x_v)$、$f_t(x_t)$，以及必要时的跨模态交互模块 $g(\cdot)$，使模型既能判断“图和文是否对应”，也能在需要时让文本生成显式依赖图像中的局部细节。

更形式化一点，可以写成：

$$
v = f_v(x_v), \quad t = f_t(x_t)
$$

如果任务只要求匹配，可以直接比较 $v$ 和 $t$ 的相似度：

$$
s(x_v, x_t) = \mathrm{sim}(v, t)
$$

如果任务要求细粒度理解或生成，则需要交互模块：

$$
h = g(v, t), \quad y = \mathrm{Decoder}(h)
$$

这里的边界必须先说清。

第一，视觉 token 比文本 token 更容易膨胀。token 可以理解为模型一次处理的最小片段。文本一句话可能只有几十到几百个 token，但一张图像切成 patch 后，轻易就会得到几百到上千个视觉 token。Vision Transformer 的基本做法，就是把图像切成固定大小的 patch，再把每个 patch 映射成一个向量。

以一张 $224 \times 224$ 的图像为例，如果 patch 大小是 $16 \times 16$，那么视觉 token 数量为：

$$
n_v = \frac{224}{16} \times \frac{224}{16} = 14 \times 14 = 196
$$

如果图像分辨率变成 $448 \times 448$，而 patch 大小不变，则：

$$
n_v = \frac{448}{16} \times \frac{448}{16} = 28 \times 28 = 784
$$

可以看到，分辨率翻倍后，视觉 token 数量不是翻倍，而是接近变成 4 倍。

第二，Transformer 的全连接注意力成本近似与 token 数平方成正比。若图像产生 $n_v=1024$ 个视觉 token，文本有 $n_t=128$ 个 token，早期融合每层要处理的注意力规模近似为：

$$
(n_v+n_t)^2 = 1152^2 = 1{,}327{,}104
$$

这只是一个头、一个层上的数量级直觉，没有把多头注意力、batch、大模型隐藏维度、前馈层、缓存和反向传播算进去。它真正要说明的是：图文越早全量混合，成本越容易失控。

对比一下三类方案的计算直觉：

| 方案 | 主要交互对象 | 近似成本直觉 | 说明 |
| --- | --- | --- | --- |
| 早期融合 | 全部视觉 token + 全部文本 token | $O((n_v+n_t)^2)$ | 每层都做图文全量交互，表达强但很贵 |
| 中期融合 | 文本 token 对视觉 token 或视觉摘要 token | $O(n_t \cdot n_v)$ 或 $O(n_t \cdot m)$ | 若先把视觉压缩到 $m$ 个摘要 token，可显著降成本 |
| 晚期融合 | 图像全局向量 vs 文本全局向量 | $O(d)$ 或 batch 内相似度矩阵 | 训练靠对比损失，推理侧最省 |

再用一个玩具例子解释差别。假设图片内容是一只红球放在木桌上，文本是 `"a red ball on a table"`。

如果用晚期融合，模型更容易学到“这张图和这句话整体匹配”。  
如果用户问“球在桌子的左边还是右边”，晚期融合通常不够，因为它保留的是全局摘要，局部布局信息往往已经丢失。  
如果用中期融合，文本 token `"red"`、`"ball"`、`"table"` 可以分别去查询不同视觉区域，模型更容易学到颜色、物体和位置之间的对应关系。

所以这篇文章的边界是：只讨论视觉-语言 Transformer 的融合架构，不展开数据清洗、标注噪声处理、RLHF、工具调用、多模态输出生成，也不讨论音频和视频输入的统一建模。

---

## 核心机制与推导

先看最重要的跨模态注意力。注意力可以理解为：一组 token 去另一组 token 中查找和自己最相关的信息，再把查到的信息聚合回来。

设文本表示为 $T \in \mathbb{R}^{n_t \times d}$，视觉表示为 $V \in \mathbb{R}^{n_v \times d}$。如果让文本查询图像，则常见写法是：

$$
Q = T W_Q,\quad K = V W_K,\quad U = V W_V
$$

其中：

| 符号 | 含义 | 直白理解 |
| --- | --- | --- |
| $Q$ | Query | “我想找什么” |
| $K$ | Key | “我这里有哪些可被匹配的索引” |
| $U$ | Value | “真正被取回并聚合的信息” |
| $W_Q, W_K, W_V$ | 可学习参数 | 把原始表示投影到注意力所需空间 |

跨模态注意力输出为：

$$
\mathrm{Attn}(T,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)U
$$

这个公式可以拆成三步理解。

第一步，算相关性：

$$
S = \frac{QK^\top}{\sqrt{d_k}}
$$

矩阵 $S \in \mathbb{R}^{n_t \times n_v}$ 的第 $(i,j)$ 项，表示第 $i$ 个文本 token 与第 $j$ 个视觉 token 的匹配程度。

第二步，做归一化：

$$
A = \mathrm{softmax}(S)
$$

这样每个文本 token 都会得到一组和为 1 的视觉注意力权重。

第三步，按权重聚合视觉信息：

$$
O = AU
$$

于是，第 $i$ 个文本 token 输出的是“它从整张图里读回来的视觉证据”。

举个更具体的例子。假设问题是“这张图里有猫吗”。文本中的 `"cat"` token 会倾向于给毛发纹理、猫耳轮廓、猫脸区域更高权重；如果问题改成“桌子是什么颜色”，则 `"color"`、`"table"` 等 token 会更关注桌面区域。这就是 cross-attention 的价值：文本生成的每一步，都能根据当前词语去图像中重新取证，而不是只依赖一份固定的全局图像摘要。

如果不做中间交互，而是采用 CLIP / ALIGN 这类晚期融合，机制就变成“双塔编码 + 对比学习”。模型不追求逐 token 的精细对齐，而是让正确图文对在向量空间更接近，让错误图文对更远。

设图像全局向量为 $v_i$，文本全局向量为 $t_i$，温度参数为 $\tau$，则 batch 内的图到文对比损失写作：

$$
\mathcal{L}_{img} = -\frac{1}{N}\sum_i \log \frac{\exp(\langle v_i,t_i\rangle/\tau)}{\sum_j \exp(\langle v_i,t_j\rangle/\tau)}
$$

文到图对比损失写作：

$$
\mathcal{L}_{txt} = -\frac{1}{N}\sum_i \log \frac{\exp(\langle t_i,v_i\rangle/\tau)}{\sum_j \exp(\langle t_i,v_j\rangle/\tau)}
$$

总损失为：

$$
\mathcal{L} = \frac{1}{2}(\mathcal{L}_{img}+\mathcal{L}_{txt})
$$

这组公式的直觉并不复杂。对于每张图，模型希望正确文本在整个 batch 里得分最高；对于每段文本，模型也希望正确图像在整个 batch 里得分最高。batch 内其余样本天然就成了负样本，因此不需要额外人工构造复杂的负采样逻辑。

如果把三类代表性模型放到一条线上看，会更容易理解：

| 模型/家族 | 公开可确认的主结构特征 | 融合特点 | 典型能力 |
| --- | --- | --- | --- |
| CLIP | 图像编码器 + 文本编码器 + 对比损失 | 晚期融合 | 图文检索、零样本分类、向量召回 |
| ALIGN | 双塔编码器 + 大规模噪声图文对比学习 | 晚期融合，强调规模 | 大规模检索、跨模态召回 |
| Flamingo / BLIP-2 类 | 冻结视觉编码器 + 适配器 / Q-Former + LLM | 中期融合，靠 bridge 做跨模态注入 | 看图问答、描述生成、条件对话 |
| GPT-4V | 公开材料未披露完整连接细节 | 从能力表现推断属于深度融合型系统 | 通用视觉问答、文档理解 |

这里要保留一个严格边界：GPT-4V 的完整架构没有被 OpenAI 公开披露，因此“它具体是不是某种 cross-attention 变体”只能算基于公开能力和行业范式的推断，不应当当作官方已确认事实。公开论文里，Flamingo 和 BLIP-2 更适合作为“中期融合如何落地”的可解释样板。

工程上再看一个真实场景。做电商“以图搜货”时，用户上传一张鞋子照片，系统需要在千万商品库中找相似商品。这个任务的首要目标是高吞吐召回和低成本索引，而不是逐词解释图像细节，因此 CLIP / ALIGN 式晚期融合通常更合适。相反，如果做“发票截图问答”或“图表问答”，用户会问“总金额是多少”“蓝线在 2024 年 7 月为什么下降”，这时系统必须让文本访问局部视觉证据，仅靠全局向量几乎不够，因此中期融合更合理。

---

## 代码实现

实现时最重要的是把接口拆清楚：视觉编码器负责输出视觉 token，语言编码器负责输出文本 token，融合层决定是在输入层拼接、在中间做 cross-attention，还是最后只做相似度学习。

下面给一个最小可运行的 Python 例子，演示两件事：

1. `scaled dot-product attention` 如何让文本 token 查询视觉 token。  
2. CLIP 风格的双向对比损失如何计算。

代码只依赖 `numpy`，可以直接运行。

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def l2_normalize(x, axis=-1, eps=1e-12):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def cross_attention(text_tokens, image_tokens):
    """
    text_tokens:  [n_t, d]
    image_tokens: [n_v, d]
    返回:
      attended: [n_t, d]
      weights:  [n_t, n_v]
    """
    if text_tokens.ndim != 2 or image_tokens.ndim != 2:
        raise ValueError("text_tokens and image_tokens must be 2D arrays")
    if text_tokens.shape[1] != image_tokens.shape[1]:
        raise ValueError("text_tokens and image_tokens must share the same hidden size")

    q = text_tokens
    k = image_tokens
    v = image_tokens
    d_k = q.shape[-1]

    scores = q @ k.T / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    attended = weights @ v
    return attended, weights


def clip_loss(image_embeds, text_embeds, temperature=0.07):
    """
    image_embeds: [batch, d]
    text_embeds:  [batch, d]
    """
    if image_embeds.shape != text_embeds.shape:
        raise ValueError("image_embeds and text_embeds must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    image_embeds = l2_normalize(image_embeds, axis=-1)
    text_embeds = l2_normalize(text_embeds, axis=-1)

    logits = image_embeds @ text_embeds.T / temperature
    labels = np.arange(logits.shape[0])

    p_i2t = softmax(logits, axis=1)
    p_t2i = softmax(logits.T, axis=1)

    loss_i = -np.mean(np.log(p_i2t[labels, labels] + 1e-12))
    loss_t = -np.mean(np.log(p_t2i[labels, labels] + 1e-12))
    loss = 0.5 * (loss_i + loss_t)
    return loss, logits


def main():
    # 3 个文本 token，4 个视觉 token，隐藏维度 d=4
    # 可以把它理解成三个词在查询四个图像区域
    text_tokens = np.array([
        [1.0, 0.0, 0.0, 0.0],   # "red"
        [0.0, 1.0, 0.0, 0.0],   # "ball"
        [0.0, 0.0, 1.0, 0.0],   # "table"
    ], dtype=np.float64)

    image_tokens = np.array([
        [0.9, 0.1, 0.0, 0.0],   # 红色区域
        [0.1, 0.8, 0.0, 0.0],   # 球体区域
        [0.0, 0.1, 0.9, 0.0],   # 桌面区域
        [0.0, 0.0, 0.0, 1.0],   # 背景区域
    ], dtype=np.float64)

    attended, weights = cross_attention(text_tokens, image_tokens)

    print("=== Cross Attention Weights ===")
    print(np.round(weights, 3))
    print("row sums:", np.round(weights.sum(axis=1), 6))
    print()

    # 3 对图文样本。对角线位置是正确匹配
    image_embeds = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.7, 0.7, 0.0],
    ], dtype=np.float64)

    text_embeds = np.array([
        [0.95, 0.05, 0.0],
        [0.05, 0.95, 0.0],
        [0.60, 0.80, 0.0],
    ], dtype=np.float64)

    loss, logits = clip_loss(image_embeds, text_embeds, temperature=0.07)

    print("=== CLIP-style Logits ===")
    print(np.round(logits, 3))
    print()
    print("clip-style loss:", round(float(loss), 6))

    # 基本正确性检查
    assert attended.shape == text_tokens.shape
    assert weights.shape == (3, 4)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert logits.shape == (3, 3)
    assert loss >= 0.0


if __name__ == "__main__":
    main()
```

这段代码可以从机制上读出两件事。

第一，`weights` 的每一行对应一个文本 token 对全部视觉 token 的注意力分布。比如 `"red"` 这个 token 应该更关注红色区域；`"ball"` 更关注球体区域；`"table"` 更关注桌面区域。  
第二，`logits` 是 batch 内所有图文两两配对的相似度矩阵。理想情况下，对角线位置应该比非对角线位置更高，因为对角线代表正确图文对。

如果把它扩展成工程代码，数据流通常是下面三种。

```python
# 1) early fusion
v_tokens = vision_encoder(image)        # [n_v, d]
t_tokens = text_encoder(input_ids)      # [n_t, d]
joint = concat(v_tokens, t_tokens)      # [n_v + n_t, d]
h = joint_transformer(joint)
out = task_head(h)

# 2) mid fusion / cross-attn
v_tokens = vision_encoder(image)        # [n_v, d]
t_states = llm_embed(input_ids)         # [n_t, d]

for block in decoder_blocks:
    t_states = block.self_attn(t_states)
    if block.has_cross_attn:
        t_states = block.cross_attn(
            query=t_states,
            key=v_tokens,
            value=v_tokens,
        )
    t_states = block.ffn(t_states)

out = lm_head(t_states)

# 3) late fusion
v = image_tower(image)                  # [d]
t = text_tower(input_ids)               # [d]
loss = contrastive_loss(v, t)
```

对新手来说，可以把这三段伪代码分别理解成三种决策：

| 方案 | 系统在什么时候交流图文信息 | 代价特点 |
| --- | --- | --- |
| early fusion | 一上来就交流 | 最贵，但联合最深 |
| mid fusion | 编码一部分后再交流 | 平衡最好，最常见 |
| late fusion | 最后只交换摘要结果 | 最省，适合检索 |

工程里还有一个常见做法：先把大量视觉 token 压缩成少量“查询 token”再送进语言模型。BLIP-2 的 Q-Former 就属于这一路线。它的本质不是“再造一个大模型”，而是做一个桥接器，把原本很多的视觉 token 压成更少、更适合语言模型消费的视觉摘要。

可以把这个思路写成：

$$
V \in \mathbb{R}^{n_v \times d}
\quad \xrightarrow{\text{Q-Former / Query Bottleneck}} \quad
Z \in \mathbb{R}^{m \times d}, \quad m \ll n_v
$$

这样后续 cross-attention 的成本就从 $O(n_t \cdot n_v)$ 下降到 $O(n_t \cdot m)$。这正是很多中期融合架构能在大视觉输入下仍然可用的关键原因。

---

## 工程权衡与常见坑

真正把模型做起来，难点不在“能不能连上”，而在“连到什么程度还跑得动、训得稳、上线后还能控成本”。

第一类坑是 token 数失控。高分辨率图像、长文档页面、OCR 文本叠加在一起时，早期融合会让显存、延迟和训练不稳定性一起上升。你以为只是多输入了一张图，实际上是给 Transformer 增加了几百到几千个 token。

第二类坑是晚期融合过度依赖全局向量。它很擅长回答“是否相关”，不擅长回答“相关在哪里、为什么相关”。所以它在检索上往往很强，但一旦进入细节问答、定位解释、证据提取，性能就会明显受限。

第三类坑是视觉编码器和语言模型的表示空间不兼容。简单说，就是图像特征会说一种“表示语言”，而语言模型只熟悉另一种“表示语言”。这时通常需要投影层、adapter、query token、Q-Former 或 cross-attn bridge 做翻译。

第四类坑是训练目标和上线目标不一致。很多团队在训练时用的是“图文是否匹配”的对比学习目标，但上线后要求系统“按问题逐步生成答案”。这两类目标不是同一件事。只靠对比学习训练出的双塔模型，通常不能直接承担精细生成任务。

第五类坑是错误地把“更深融合”当成“更强能力”。更早、更深的融合确实提高了表达力，但也会带来更长训练周期、更复杂的数据需求以及更难排查的错误来源。模型能力提升必须和任务收益相匹配，否则只是增加成本。

把权衡压缩成表格更清晰：

| 融合方式 | 计算压力 | 细粒度对齐 | 工程风险 | 适配任务 |
| --- | --- | --- | --- | --- |
| 早期融合 | 最高 | 最强 | 显存爆炸、长上下文难训、部署成本高 | 小图小文本、强结构联合推理 |
| 中期融合 | 中等 | 强 | 模块耦合复杂、训练不稳定、桥接设计敏感 | VQA、图表问答、OCR 生成、看图对话 |
| 晚期融合 | 最低 | 弱 | 细节丢失、生成能力弱、难做局部证据解释 | 检索、分类、去重、召回 |

有一个比喻可以保留，但必须落到机制上。早期融合像把所有角色都拉进同一个大会议室，从第一分钟开始一起讨论，信息交流最充分，但会议成本最高；晚期融合像各自先开小会，最后只交换总结；中期融合像只让关键岗位在几个固定节点碰头，因此在表达力和成本之间更平衡。

真实工程例子：做企业知识库里的“图片 + 文档问答”。如果你把 PDF 页图、OCR 文本、用户问题从第一层全量拼接，延迟通常很难接受，更常见的可行方案是：

1. 先用视觉编码器提取页面区域特征。  
2. 再用 OCR 或版面分析模型抽出文本块。  
3. 只把压缩后的视觉摘要和问题 token 送入 cross-attn。  
4. 必要时先做检索，再做小范围深度融合。

常见规避方法也比较固定：

| 问题 | 常见做法 | 作用 |
| --- | --- | --- |
| 视觉 token 太多 | 增大 patch、做下采样、只取关键区域 | 直接降低 $n_v$ |
| cross-attn 太贵 | 用 query bottleneck / Q-Former | 把视觉 token 压缩成固定小集合 |
| 训练不稳定 | 只在少数层插入 cross-attn，冻结部分底座参数 | 降低耦合和梯度干扰 |
| 检索和生成都要做 | 两阶段系统：先晚期融合召回，再中期融合重排或生成 | 同时兼顾吞吐和精度 |
| 文档理解输入太长 | 先做版面切块、区域选择、候选页检索 | 控制上下文长度 |

对新手来说，可以把工程选择总结成一个顺序问题：

先问“我要不要看局部证据”。  
如果不要，优先考虑晚期融合。  
如果要，再问“我能否承担大规模 token 交互”。  
如果不能，优先考虑中期融合而不是直接上早期融合。

---

## 替代方案与适用边界

如果需求是“给海量图片建向量库，让文字能搜图，图片也能搜文”，首选通常不是复杂的跨模态生成模型，而是 CLIP / ALIGN 这类晚期融合。它们训练简单、推理便宜、索引友好，特别适合检索、分类和召回。

如果需求是“看图后回答细节问题”，例如“这张电路板上烧坏的是哪一块”“表格第三列比第二列高多少”，就需要中期融合。因为模型必须在生成每个词时访问图像证据，而不是只靠一个全局向量去回忆整张图。

如果需求是“输入图像、文本、音频、视频，统一在一个网络里端到端处理”，那已经进入原生多模态模型范畴。OpenAI 在 GPT-4o system card 中公开说明了它是 end-to-end 的 omni 模型，这和经典“先有图像塔，再接语言模型”的拼接式方案不同。但这已经超出本文只讨论视觉-语言 Transformer 融合架构的范围。

下面把场景和推荐方案对应起来：

| 场景 | 推荐方案 | 原因 |
| --- | --- | --- |
| 图文检索、以图搜图、零样本分类 | CLIP / ALIGN 式晚期融合 | 吞吐高，训练目标直接对齐检索，向量索引友好 |
| 视觉问答、图表问答、发票问答 | cross-attn 中期融合 | 需要词级别访问视觉证据 |
| 图像描述、对话式看图生成 | 视觉编码器 + 适配器 / Q-Former + LLM | 兼顾生成能力和视觉条件控制 |
| 极小规模、强结构、多模态联合推理 | 早期融合 | 对齐最强，但只在成本可控时值得用 |

还可以再加一个新手最容易混淆的区分表：

| 任务类型 | 你真正要优化的东西 | 常见首选 |
| --- | --- | --- |
| 匹配 | 正确样本能否排到前面 | 晚期融合 |
| 重排 | 候选结果里谁更相关 | 晚期融合或轻量中期融合 |
| 解释 | 相关证据在哪里 | 中期融合 |
| 生成 | 每个输出词是否能访问图像证据 | 中期融合 |
| 深度联合推理 | 图文是否从底层共同建模 | 早期融合 |

可以把选择原则压缩成一句经验法则：

先问任务主要是“匹配”还是“生成”。  
匹配优先晚期融合，生成优先中期融合。  
只有当你明确需要从最底层就学习图文联合结构，而且样本规模、算力、上下文长度都可控时，才考虑更激进的早期融合。

---

## 参考资料

- [CLIP 论文：Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)  
  作用：双塔对比学习的代表性一手论文，理解晚期融合的最佳入口。

- [ViT 论文：An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)  
  作用：说明视觉 token 的来源，即图像如何被切成 patch token。

- [ALIGN 论文页面：Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918)  
  作用：理解“大规模噪声图文对 + 双塔对比学习”为何仍然有效。

- [BLIP-2 论文：Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)  
  作用：理解 query bridge / Q-Former 这类中期融合桥接器。

- [Flamingo 论文：Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)  
  作用：理解“冻结视觉模型 + 冻结语言模型 + cross-attn 连接层”的工程路线。

- [OpenAI GPT-4V System Card](https://openai.com/index/gpt-4v-system-card/)  
  作用：确认 GPT-4V 是公开发布的视觉能力系统；但不公开完整内部架构细节。

- [OpenAI GPT-4o System Card](https://openai.com/index/gpt-4o-system-card/)  
  作用：对比“拼接式多模态”与“端到端原生多模态”路线。

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
  作用：理解 self-attention / cross-attention 的数学基础，适合回到原始 Transformer 机制。

- [Meta Intelligence 多模态综述](https://www.meta-intelligence.tech/en/insight-multimodal-ai)  
  作用：适合先建立全景图，再回到论文看细节。

- [EmergentMind: GPT-4V](https://www.emergentmind.com/topics/gpt-4v)  
  作用：二手整理材料，适合快速回顾术语，但应以论文和官方文档为准。
