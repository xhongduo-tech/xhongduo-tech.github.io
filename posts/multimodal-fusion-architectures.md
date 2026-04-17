## 核心结论

多模态融合的核心问题，不是“要不要让图像和文本一起工作”，而是“**在什么时候让它们发生交互**”。

早期融合、晚期融合、跨模态注意力，可以先用一句话区分：

| 策略 | 交互发生时间 | 直观理解 | 典型优势 | 主要代价 |
|---|---|---|---|---|
| 早期融合 | 输入层 | 图像 token 和文本 token 一开始就进入同一个 Transformer | 细粒度对齐强 | 计算量高，显存压力大 |
| 晚期融合 | 输出层 | 各自编码完，再比较最终语义向量 | 训练和部署灵活，适合检索 | 对齐较粗，局部推理弱 |
| 跨模态注意力 | 中间层 | 先各自处理，再在若干层交换信息 | 在效果和成本之间折中 | 结构更复杂，训练更难调 |

“token”第一次出现时，可以简单理解为：**模型处理信息时使用的最小离散单元**。  
文本里常见的是词片或子词，图像里常见的是 patch，也就是把图片切成小块后得到的视觉片段。

先看一个最小例子：图片里有一只黑狗，文本是“a black dog running”。

- 早期融合：图像 patch 和文字 token 一起排成一个序列，从第一层开始互相可见。
- 晚期融合：图像先变成一个向量，文本也先变成一个向量，最后只比较两个向量是否相似。
- 跨模态注意力：图像和文本先分别理解，再在中间交换关键信息，比如“black”是否对应到狗身上的深色区域。

三类结构可以先写成三个抽象公式：

$$
h=\mathrm{Transformer}(\mathrm{Concat}(E_{\text{img}}, E_{\text{text}}))
$$

$$
P=\mathrm{Softmax}\left(\frac{\cos(z_{\text{img}}, z_{\text{text}})}{\tau}\right)
$$

$$
h^{l+1}_{\text{img}}=\mathrm{Attn}(h^l_{\text{img}}, h^l_{\text{text}})
$$

第一式对应早期融合，第二式对应晚期融合，第三式对应中层跨模态交互。

这三个式子分别表达的是：

| 公式 | 表达的含义 | 对应策略 |
|---|---|---|
| $\mathrm{Concat}(E_{\text{img}}, E_{\text{text}})$ 后送入统一 Transformer | 两种模态从最开始就在同一计算图里交互 | 早期融合 |
| $\cos(z_{\text{img}}, z_{\text{text}})$ | 两种模态先各自压缩成全局语义向量，再比较是否接近 | 晚期融合 |
| $\mathrm{Attn}(h^l_{\text{img}}, h^l_{\text{text}})$ | 图像表示和文本表示在中间层彼此查询 | 跨模态注意力 |

结论先给清楚：

1. 需要细粒度对齐，比如“图中左上角的红灯是不是亮着”，优先考虑早期融合或中层跨模态注意力。
2. 需要大规模图文检索、双塔召回、独立扩展编码器，优先考虑晚期融合。
3. 需要视觉问答、视觉推理、图文生成，通常采用中层跨模态交互，因为它比纯晚期融合更强，又比完全早期融合更可控。

再补一句工程判断标准。真正选型时，通常不是先问“哪篇论文最好”，而是先问三个问题：

| 判断问题 | 如果回答是“是” | 更偏向 |
|---|---|---|
| 是否要做海量检索或缓存特征 | 图像库要离线编码，查询时只编码文本 | 晚期融合 |
| 是否要回答局部细节问题 | 需要词语和区域一一对应 | 早期融合 / 中层融合 |
| 是否要多轮生成或链式推理 | 需要持续交换模态信息 | 中层融合 |

---

## 问题定义与边界

多模态融合讨论的是：**图像、文本、音频等不同模态的信息，在模型内部以什么结构相互连接**。

“模态”第一次出现时，可以理解为：**信息的来源类型**。图像是一种模态，文本也是一种模态。融合就是让这些不同来源的信息共同参与决策。

这里聚焦视觉-语言，也就是图像和文本。问题边界主要有四个：

| 边界维度 | 关注点 | 为什么重要 |
|---|---|---|
| token 数量 | 图像 token 往往远多于文本 | 直接决定 attention 开销和显存占用 |
| 对齐粒度 | 是只做整图-整句匹配，还是做区域-词语对齐 | 决定是否必须深层交互 |
| 训练方式 | 是否端到端联合训练，是否允许冻结部分模块 | 决定训练稳定性和数据需求 |
| 部署方式 | 是否要独立缓存图像或文本特征 | 决定线上延迟、吞吐和成本 |

新手可以把它理解成“握手时机”问题：

- 早期融合：一开始就握手。
- 中层融合：走到半路再握手。
- 晚期融合：到了终点才握手。

这个比喻只用于帮助建立第一层直觉，真正差异仍然体现在**计算图结构**上：  
谁先编码，谁后交互，交互发生几次，是否全量互看，是否可以单独缓存特征，这些都会改变模型能力和工程形态。

再看一个更具体的玩具例子。输入是一张商品图，文本是“黑色双肩包，带侧袋”。

- 如果目标是检索“这张图和这句话是否匹配”，晚期融合通常就足够。
- 如果目标是回答“侧袋在左边还是右边”，晚期融合通常不够，因为它只保留了全局语义，缺少局部对应关系。
- 如果目标是根据图片和问题连续生成回答，通常需要中层或更深的跨模态交互，因为生成过程本身就需要持续访问视觉信息。

可以把不同任务对融合粒度的要求，压缩成一张表：

| 任务 | 需要的对齐粒度 | 典型输出 | 常见选择 |
|---|---|---|---|
| 图文检索 | 粗粒度 | 相似度分数、Top-K 结果 | 晚期融合 |
| 零样本分类 | 粗到中粒度 | 类别概率 | 晚期融合 |
| 视觉问答 | 中到细粒度 | 自然语言答案 | 中层融合 |
| 区域级判断 | 细粒度 | 局部属性或位置结论 | 早期融合 / 中层融合 |
| 图文生成 | 中到细粒度 | 连续文本 | 中层融合 |

复杂度也直接限制了可行性。若图像被切成 $N_{\text{img}}$ 个 token，文本有 $N_{\text{text}}$ 个 token，则：

- 早期融合 self-attention 复杂度近似为
  $$
  O((N_{\text{img}}+N_{\text{text}})^2)
  $$
- 晚期融合为
  $$
  O(N_{\text{img}}^2)+O(N_{\text{text}}^2)
  $$
  再加一个很小的向量对齐成本。

把平方项展开，早期融合的主要差别会更直观：

$$
(N_{\text{img}}+N_{\text{text}})^2
=
N_{\text{img}}^2
+
2N_{\text{img}}N_{\text{text}}
+
N_{\text{text}}^2
$$

中间那一项

$$
2N_{\text{img}}N_{\text{text}}
$$

就是额外的跨模态全连接交互成本。它说明一件事：  
早期融合贵，不只是因为“token 多”，更因为**每个图像 token 都要直接看每个文本 token**。

代入一个常见数字：图像 256 个 token，文本 32 个 token。

$$
(256+32)^2 = 288^2 = 82944
$$

而分开计算时：

$$
256^2 + 32^2 = 65536 + 1024 = 66560
$$

差值是：

$$
82944 - 66560 = 16384
$$

这个差值恰好就是：

$$
2 \times 256 \times 32 = 16384
$$

也就是新增的跨模态全量交互项。

如果图像 token 再上升到 576，而文本仍是 32，则：

$$
(576+32)^2 = 608^2 = 369664
$$

$$
576^2 + 32^2 = 331776 + 1024 = 332800
$$

额外成本变成：

$$
369664 - 332800 = 36864
$$

这说明 token 规模一大，早期融合的成本问题是结构性的，不是简单“加机器”就能完全解决。

---

## 核心机制与推导

先看早期融合。它的基本做法，是把图像嵌入和文本嵌入直接拼接：

$$
X_0 = \mathrm{Concat}(E_{\text{img}}, E_{\text{text}})
$$

然后送入统一 Transformer：

$$
H^{l+1} = \mathrm{SelfAttn}(H^l) + \mathrm{FFN}(H^l)
$$

“嵌入”第一次出现时，可以理解为：**把原始输入映射成模型可计算的向量表示**。  
例如，一张图片先经过视觉编码器被切成 patch embedding，一段文本先经过词表和位置编码变成 token embedding。

以标准 attention 为例：

$$
\mathrm{Attn}(Q,K,V)=\mathrm{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

这里：

| 符号 | 含义 |
|---|---|
| $Q$ | Query，表示“我想查询什么” |
| $K$ | Key，表示“我能提供什么索引” |
| $V$ | Value，表示“我真正携带的信息” |
| $d$ | 向量维度，用于缩放内积，避免数值过大 |

拼接后，$Q,K,V$ 都来自同一个联合序列，所以跨模态联系天然存在。  
这意味着“black”这个文本 token 可以在第一层就去关注深色视觉 patch；某个图像区域也可以直接去关注“dog”这个词。

如果把联合序列按模态分块，联合 attention 分数矩阵可以写成：

$$
S=
\frac{QK^\top}{\sqrt d}
=
\begin{bmatrix}
S_{\text{img}\rightarrow \text{img}} & S_{\text{img}\rightarrow \text{text}} \\
S_{\text{text}\rightarrow \text{img}} & S_{\text{text}\rightarrow \text{text}}
\end{bmatrix}
$$

这张分块矩阵很重要。它告诉我们：

- 左上角：图像内部自注意力。
- 右下角：文本内部自注意力。
- 右上角和左下角：跨模态注意力。

早期融合的强项正来自这两个非对角块，它们从第一层就存在。

再看晚期融合。它通常是双编码器结构：

$$
z_{\text{img}} = f_{\text{img}}(x_{\text{img}})
$$

$$
z_{\text{text}} = f_{\text{text}}(x_{\text{text}})
$$

其中 $f_{\text{img}}$ 和 $f_{\text{text}}$ 可以完全独立，只要最后把输出映射到同一语义空间即可。  
然后在同一语义空间比较相似度：

$$
s = \frac{z_{\text{img}}^\top z_{\text{text}}}{\|z_{\text{img}}\|\|z_{\text{text}}\|}
$$

这就是余弦相似度。它的取值范围通常在 $[-1,1]$，值越大代表两个向量方向越接近。

再经温度参数 $\tau$ 缩放：

$$
P=\mathrm{Softmax}\left(\frac{s}{\tau}\right)
$$

“温度参数”可以白话理解为：**控制分数区分度的缩放系数**。  
$\tau$ 越小，分布越尖锐，模型越强调高分样本和低分样本的差距；$\tau$ 越大，分布越平缓。

训练时常用对比学习损失。对一批图文对，图像 $i$ 对应文本 $i$，则图像到文本的一侧损失可写成：

$$
\ell_i=-\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)}
$$

文本到图像的一侧也可写成：

$$
\ell_i'=-\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ji}/\tau)}
$$

最终常取对称形式：

$$
\mathcal{L}=\frac{1}{2B}\sum_{i=1}^{B}(\ell_i+\ell_i')
$$

其中 $B$ 是 batch size。

这套目标的含义很直接：

- 正样本图文对要更接近。
- 同一批次里的其他图文对被当作负样本。
- 批次越大，负样本越多，对比学习通常越有效。

这就是 CLIP、ALIGN 一类方法的基础思路。它非常适合检索，因为图像向量和文本向量可以预先算好，再做近似最近邻搜索。

第三类是中层跨模态注意力。它不是一开始就全量拼接，也不是最后才比较向量，而是在中间若干层建立信息交换。一个常见形式是：

$$
h^{l+1}_{\text{img}}=\mathrm{Attn}(Q=h^l_{\text{img}}, K=h^l_{\text{text}}, V=h^l_{\text{text}})
$$

$$
h^{l+1}_{\text{text}}=\mathrm{Attn}(Q=h^l_{\text{text}}, K=h^l_{\text{img}}, V=h^l_{\text{img}})
$$

这个结构可以理解为：

- 图像 token 用文本表示做查询对象，更新自己的表示。
- 文本 token 也反过来读取图像表示。
- 两种模态先保留各自内部结构，再选择性地交换信息。

如果把一层中层融合拆开，它通常包含两部分：

$$
\tilde{h}^{l+1}_{\text{img}}=\mathrm{SelfAttn}(h^l_{\text{img}})
$$

$$
h^{l+1}_{\text{img}}=\mathrm{CrossAttn}(\tilde{h}^{l+1}_{\text{img}}, h^l_{\text{text}})
$$

文本侧同理。  
这说明中层融合并不是“只有跨模态注意力”，而是“**自模态建模 + 跨模态交换**”的组合。

也可以使用 bottleneck tokens，也就是“**专门负责传话的少量中介 token**”。设中介 token 数量为 $B$，则可先让图像和文本分别与 bottleneck 交互：

$$
h^{l+1}_{b}=\mathrm{Attn}(h^l_b, h^l_{\text{img}}\cup h^l_{\text{text}})
$$

$$
h^{l+1}_{\text{img}}=\mathrm{Attn}(h^l_{\text{img}}, h^{l+1}_{b})
\quad,\quad
h^{l+1}_{\text{text}}=\mathrm{Attn}(h^l_{\text{text}}, h^{l+1}_{b})
$$

这样做的重点不是“加几个特殊 token”，而是把原本图像和文本的全量两两交互，改成通过一个窄通道传递关键信息，从而压缩计算量。

玩具例子可以这样理解：

- 早期融合：图像和文本从第一天起在同一个办公室办公。
- 晚期融合：两个团队独立干活，只在最终汇报会上交换结果。
- 中层融合：前期分开，中期安排接口人同步，再各自继续推进。

如果把三者做成一张更适合新手的对照表：

| 维度 | 早期融合 | 晚期融合 | 中层融合 |
|---|---|---|---|
| 交互开始时间 | 第一层前 | 编码结束后 | 编码若干层后 |
| 是否保留局部对齐 | 强 | 弱 | 中到强 |
| 是否便于缓存特征 | 不便 | 非常方便 | 一般 |
| 是否适合生成 | 一般，可做但重 | 通常不够 | 适合 |
| 训练与实现复杂度 | 中 | 低到中 | 高 |

真实工程例子：

- CLIP、ALIGN 更接近“独立编码后对齐”，适合大规模图文匹配与零样本分类。
- Multimodal Bottleneck Transformer 代表的是“中间引入受控信息通道”的思路。
- GPT-4V 一类视觉语言生成系统更依赖中间层交互，因为它不仅要“像不像”，还要“根据图回答、推理、继续生成”。

---

## 代码实现

下面用一个**最小可运行**的 Python 例子演示三种思路。代码不依赖第三方库，只用标准库完成一个简化版 attention、相似度计算和中层交互。它不是训练框架，但每个函数都可以直接运行，足够对应前面的结构公式。

```python
import math
from typing import List, Tuple

Vector = List[float]
Matrix = List[Vector]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine(a: Vector, b: Vector) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0.0 or nb == 0.0:
        raise ValueError("zero vector is not allowed in cosine similarity")
    return dot(a, b) / (na * nb)


def softmax(xs: Vector) -> Vector:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]


def zeros(dim: int) -> Vector:
    return [0.0] * dim


def mean_pool(tokens: Matrix) -> Vector:
    if not tokens:
        raise ValueError("tokens must not be empty")
    dim = len(tokens[0])
    out = zeros(dim)
    for token in tokens:
        if len(token) != dim:
            raise ValueError("all tokens must have the same dimension")
        for i, value in enumerate(token):
            out[i] += value
    return [value / len(tokens) for value in out]


def transpose(m: Matrix) -> Matrix:
    if not m:
        return []
    return [list(col) for col in zip(*m)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b:
        raise ValueError("matrices must not be empty")
    b_t = transpose(b)
    return [[dot(row, col) for col in b_t] for row in a]


def scale_matrix(m: Matrix, factor: float) -> Matrix:
    return [[x * factor for x in row] for row in m]


def row_softmax(m: Matrix) -> Matrix:
    return [softmax(row) for row in m]


def weighted_sum(weights: Matrix, values: Matrix) -> Matrix:
    out = []
    for row in weights:
        merged = []
        for j in range(len(values[0])):
            merged.append(sum(row[i] * values[i][j] for i in range(len(values))))
        out.append(merged)
    return out


def attention(query: Matrix, key: Matrix, value: Matrix) -> Matrix:
    dim = len(query[0])
    scores = matmul(query, transpose(key))
    scores = scale_matrix(scores, 1.0 / math.sqrt(dim))
    weights = row_softmax(scores)
    return weighted_sum(weights, value)


def add(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]


def add_matrix(a: Matrix, b: Matrix) -> Matrix:
    return [add(x, y) for x, y in zip(a, b)]


def early_fusion(img_tokens: Matrix, text_tokens: Matrix) -> Tuple[Matrix, Vector]:
    # 早期融合：直接拼接后做一次联合 self-attention
    joint = img_tokens + text_tokens
    attended = attention(joint, joint, joint)
    pooled = mean_pool(attended)
    return attended, pooled


def late_fusion(img_tokens: Matrix, text_tokens: Matrix, tau: float = 0.1) -> Tuple[Vector, Vector, float, Vector]:
    # 晚期融合：各自池化成全局向量，再比较余弦相似度
    z_img = mean_pool(img_tokens)
    z_text = mean_pool(text_tokens)
    score = cosine(z_img, z_text)
    logits = [score / tau, 0.0]  # “匹配” vs “非匹配”的最小示意
    probs = softmax(logits)
    return z_img, z_text, score, probs


def cross_modal_fusion(img_tokens: Matrix, text_tokens: Matrix) -> Tuple[Matrix, Matrix, Vector, Vector]:
    # 中层融合：先各自内部建模，再做双向 cross-attention
    img_self = attention(img_tokens, img_tokens, img_tokens)
    text_self = attention(text_tokens, text_tokens, text_tokens)

    img_cross = attention(img_self, text_self, text_self)
    text_cross = attention(text_self, img_self, img_self)

    img_out = add_matrix(img_self, img_cross)
    text_out = add_matrix(text_self, text_cross)

    return img_out, text_out, mean_pool(img_out), mean_pool(text_out)


def main() -> None:
    # 3 维 toy embedding，只为演示“颜色 / 动物 / 动作”三个方向
    img_tokens = [
        [1.0, 0.0, 0.0],   # 深色区域
        [0.9, 0.1, 0.0],   # 狗身体
        [0.1, 0.0, 0.9],   # 跑动姿态
    ]

    text_tokens = [
        [1.0, 0.0, 0.0],   # black
        [0.0, 1.0, 0.0],   # dog
        [0.0, 0.0, 1.0],   # running
    ]

    early_tokens, early_vec = early_fusion(img_tokens, text_tokens)
    z_img, z_text, late_score, late_probs = late_fusion(img_tokens, text_tokens)
    cross_img_tokens, cross_text_tokens, cross_img_vec, cross_text_vec = cross_modal_fusion(
        img_tokens, text_tokens
    )

    assert len(early_tokens) == len(img_tokens) + len(text_tokens)
    assert len(early_vec) == 3
    assert len(z_img) == 3 and len(z_text) == 3
    assert -1.0 <= late_score <= 1.0
    assert abs(sum(late_probs) - 1.0) < 1e-9
    assert len(cross_img_tokens) == len(img_tokens)
    assert len(cross_text_tokens) == len(text_tokens)

    print("Early fusion pooled vector:", [round(x, 4) for x in early_vec])
    print("Late fusion cosine score:", round(late_score, 4))
    print("Late fusion match prob:", round(late_probs[0], 4))
    print("Cross fusion pooled image vector:", [round(x, 4) for x in cross_img_vec])
    print("Cross fusion pooled text vector:", [round(x, 4) for x in cross_text_vec])


if __name__ == "__main__":
    main()
```

这段代码对应的结构分别是：

| 函数 | 对应策略 | 它做了什么 |
|---|---|---|
| `early_fusion` | 早期融合 | 先拼接图像和文本 token，再做联合 self-attention |
| `late_fusion` | 晚期融合 | 先各自池化成向量，再计算余弦相似度 |
| `cross_modal_fusion` | 中层融合 | 先各自 self-attention，再做双向 cross-attention |

如果用更抽象的伪代码描述成模型接口，可以写成：

```python
# early fusion
tokens = concat(img_tokens, text_tokens)
h = transformer(tokens)

# late fusion
z_img = img_encoder(img_tokens)
z_text = text_encoder(text_tokens)
loss = contrastive_loss(z_img, z_text)

# mid fusion
h_img = img_encoder_front(img_tokens)
h_text = text_encoder_front(text_tokens)
h_img = cross_attention(h_img, h_text)
h_text = cross_attention(h_text, h_img)
y = decoder(h_img, h_text)
```

运行上面的完整示例时，可以这样理解输出：

- `Early fusion pooled vector`：联合建模后的整体表征。
- `Late fusion cosine score`：图像和文本全局语义是否匹配。
- `Late fusion match prob`：把相似度经过温度缩放和 softmax 后得到的示意概率。
- `Cross fusion pooled ... vector`：中层信息交换后，两种模态各自更新过的表示。

这段代码虽然简化，但有两个关键价值。

第一，它把“结构差异”而不是“模型规模”凸显出来。  
你可以直接看到三者的分界线不在于网络层数，而在于**何时交互**。

第二，它比只写 `mean_pool` 更接近真实模型。  
这里的 `attention(query, key, value)` 已经体现了“谁查询谁”的关系，因此更容易把代码和前面的公式一一对应。

真实工程例子也可以放进这个实现框架里理解：

- 做图文检索系统时，经常离线编码图像库，在线只编码查询文本，再做向量召回。这天然偏向晚期融合。
- 做视觉问答系统时，图片内容要和问题词语在多层中持续交互，例如“图中第二个人手里拿的是什么”，这更适合中层融合或带视觉 token 的生成式结构。
- 做高精度重排序时，也常见“先双塔召回，再交叉编码”的两阶段方案。第一阶段偏晚期融合，第二阶段偏中层或早期融合。

---

## 工程权衡与常见坑

真正落地时，决定架构的通常不是“哪种理论最好”，而是“哪种在约束下最能跑通”。

| 策略 | 优点 | 限制 | 缓解措施 |
|---|---|---|---|
| 早期融合 | 对齐细，适合局部推理 | attention 爆炸，显存压力大 | 限制分辨率、减少 patch、稀疏 attention、局部窗口 |
| 晚期融合 | 可独立训练，可缓存特征，检索友好 | 缺细粒度对齐，难回答局部问题 | 加强负样本、区域级对比、重排序模块 |
| 跨模态注意力 | 性能和成本折中，适合生成任务 | 结构复杂，训练不稳定性更高 | 使用 bottleneck tokens、分层融合、冻结部分模块 |

最典型的坑有五个。

第一，误把“效果差”归因于模型小，实际是融合时机不对。  
如果任务要求区域级对齐，而你只做双塔相似度，模型再大也不一定补得回来。因为信息路径本身就不支持细粒度匹配。

第二，图像 token 数失控。  
图像分辨率提高后，patch 数会显著增加。若 patch 大小固定，token 数近似与面积成正比，而早期融合成本近似随总 token 平方增长。一个从 224 到 448 的分辨率提升，不是“多一点”，而是 patch 数接近变成 4 倍，attention 成本可能进一步放大。

如果 patch 大小是 $p$，输入分辨率是 $H \times W$，则图像 token 数近似为：

$$
N_{\text{img}} \approx \frac{H}{p}\times\frac{W}{p}
\;=\;
\frac{HW}{p^2}
$$

当分辨率从 $224\times224$ 提升到 $448\times448$，在 patch 大小不变时：

$$
\frac{448\times448}{224\times224}=4
$$

也就是图像 token 数约为原来的 4 倍。  
如果又使用早期融合，那么联合 self-attention 的平方项会让压力继续扩大。

第三，晚期融合在训练集上分数很好，线上问答却不行。  
原因通常是目标错配。对比学习擅长“配不配”，不擅长“为什么”“在哪儿”“哪个区域”。训练指标和线上任务不一致，模型自然会在真实使用场景里暴露短板。

第四，中层融合没控制交互带宽。  
如果每层都让所有视觉 token 和所有文本 token 完全互看，最终还是回到高成本问题。bottleneck token 的意义就在这里，它像一个限流中介，只传最关键的信息。

第五，忽略部署链路。  
CLIP 这类结构的一个巨大优势，不只是论文指标，而是工程形态非常好：图像库可离线编码，文本查询在线编码即可。对于检索系统，这是非常硬的优势。很多团队在原型验证时只看离线效果，忽略了线上延迟、吞吐、特征缓存和索引更新，最后架构根本落不了地。

再补几个常见但容易被忽略的工程点：

| 常见问题 | 表现 | 本质原因 | 常见处理 |
|---|---|---|---|
| 负样本太弱 | 检索时容易把相似商品混淆 | 对比学习没学会细区分 | 用难负样本、同类细分负样本 |
| 图像裁剪策略不当 | 局部目标丢失 | 视觉 token 覆盖不足 | 多尺度输入、区域编码 |
| 文本过短 | 语义约束不够 | 文本侧信息量不足 | 加属性词、模板增强 |
| 冻结过多模块 | 迁移效果差 | 跨模态接口没适配新任务 | 解冻部分 cross-attention 或投影层 |
| 只看 Top-1 指标 | 线上体验与离线评测不一致 | 任务目标过于单一 | 增加召回率、延迟、稳定性评估 |

再看一个数值直觉。假设图像 256 token，文本 32 token：

- 早期融合：$O((256+32)^2)=82944$
- 晚期融合：$O(256^2)+O(32^2)=66560$

如果中层融合只引入 $B=8$ 个 bottleneck token，则额外交互规模可以近似理解为：

$$
O(B(N_{\text{img}}+N_{\text{text}}))
$$

代入数字就是：

$$
8 \times (256 + 32) = 2304
$$

这不是严格统一的实现公式，因为不同论文结构不同，但它足以说明一个核心思想：  
**通过窄通道交换信息，比让所有图像 token 和所有文本 token 全量两两相连更便宜。**

把三者放到同一个工程视角下看：

| 方案 | 离线编码 | 在线延迟 | 细粒度能力 | 系统复杂度 |
|---|---|---|---|---|
| 早期融合 | 差 | 高 | 强 | 中 |
| 晚期融合 | 强 | 低 | 弱 | 低 |
| 中层融合 | 一般 | 中 | 中到强 | 高 |

---

## 替代方案与适用边界

如果把三类方法看成“握手时间”，那么工程选择可以直接按任务目标划分。

| 方法 | 适用任务 | 不适合的场景 | 典型代表 |
|---|---|---|---|
| 早期融合 | 小规模、多模态细粒度对齐、局部推理 | 超大分辨率、超长文本、低资源部署 | 一些紧耦合视觉语言 Transformer |
| 晚期融合 | 图文检索、零样本分类、双塔召回 | 复杂问答、区域级解释、生成任务 | CLIP、ALIGN |
| 中层跨模态融合 | 视觉问答、图文推理、生成式多模态对话 | 极端低延迟且只能双塔部署的检索链路 | GPT-4V 类系统、MBT 类思路 |

但真实系统里，三种方案并不一定互斥。很多工业系统使用的是**分阶段架构**，而不是单一架构。

假设要做一个电商搜索系统，用户输入“白色运动鞋”，系统要从千万商品图里找结果。

- 第一阶段召回：适合晚期融合。原因是图像向量能离线建索引，查询快。
- 第二阶段重排：可以加入更深的跨模态交互，对 top-k 结果做更细判断。
- 如果需要回答“为什么这双鞋更符合‘复古跑鞋’”，还可能在后面再挂一个生成式模块。

如果你一开始就用全量早期融合跑全库匹配，成本通常无法接受。  
如果你全流程只用晚期融合，又很难处理细粒度属性和复杂问答。

所以更准确的说法不是“哪种融合最好”，而是“哪种融合放在**哪一层系统**最合适”。

一个常见的工业分层方案可以写成：

1. 召回阶段用晚期融合。
2. 重排阶段用中层交互。
3. 极少数高价值场景再用更重的早期或深度交叉编码。

这个思路和传统搜索系统很像：先用便宜模型缩小候选集，再用昂贵模型做精判。  
多模态系统的特殊之处只在于，候选筛选和精判之间的结构差异，来自跨模态交互时机不同。

再给一个任务到架构的简表，方便新手直接判断：

| 如果你的主要目标是…… | 优先考虑 |
|---|---|
| 大规模检索、零样本分类、向量召回 | 晚期融合 |
| 问答、推理、描述生成 | 中层跨模态交互 |
| 区域级判断、局部属性验证、强细节对齐 | 早期融合或较深的交叉编码 |
| 既要大规模检索又要高精度理解 | 两阶段或三阶段混合架构 |

所以最后的判断标准可以压缩成三句：

- 检索优先，选晚期融合。
- 推理与生成优先，选中层跨模态交互。
- token 总量可控且需要强细节对齐，才优先考虑早期融合。

---

## 参考资料

- Radford et al., “Learning Transferable Visual Models From Natural Language Supervision”, https://arxiv.org/abs/2103.00020
- Jia et al., “Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision”, https://arxiv.org/abs/2102.05918
- Nagrani et al., “Attention Bottlenecks for Multimodal Fusion”, https://arxiv.org/abs/2107.00135
- OpenAI, “GPT-4 Technical Report”, https://arxiv.org/abs/2303.08774
- OpenAI, “GPT-4V(ision) system card / related multimodal materials”, https://openai.com/research
- Meta-Intelligence, “Insight: Multimodal AI”, https://www.meta-intelligence.tech/en/insight-multimodal-ai
- Google Research / ALIGN related materials, https://research.google/pubs/scaling-up-visual-and-vision-language-representation-learning-with-noisy-text-supervision/
- InfoQ, “Google Introduces ALIGN, a New Approach to Vision-Language Pre-Training”, https://www.infoq.com/news/2021/07/google-vision-language-ai/
- Emergent Mind, “GPT-4V topic overview”, https://www.emergentmind.com/topics/gpt-4v
