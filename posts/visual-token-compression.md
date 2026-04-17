## 核心结论

视觉 Token 压缩的目标，是把图像编码器输出的长序列缩短到一个可控预算内。这里的“视觉 Token”可以先理解成“图像被切成很多小块后，每个小块对应的一段向量表示”。如果一张高分辨率图片被切成 576、1024 甚至更多个 patch，那么后续 Transformer 处理它们时，计算量和显存占用都会迅速膨胀。

这件事重要，不是因为“压缩本身很高级”，而是因为 VLM 的瓶颈常常不在语言模型，而在视觉序列太长。对于自注意力，序列长度从 $N$ 增加到 $2N$，核心计算通常接近增加到 $4$ 倍。因此，把视觉 Token 从 576 降到 144，不只是“少了 432 个 token”，而是把后续很多层的注意力成本一起压下去。

常见方法可以分成四类：

| 方法 | 核心动作 | 优点 | 风险 |
| --- | --- | --- | --- |
| 平均池化 | 邻近 token 直接取平均 | 简单、稳定、快 | 容易抹掉细节 |
| 注意力加权池化 | 按重要性分配权重后聚合 | 比平均更保语义 | 依赖打分质量 |
| 可学习压缩 | 用小投影器学习“怎么压” | 精度通常更好 | 需要训练或适配 |
| Token Merging（ToMe） | 把相似 token 合并 | 推理前后都容易插入 | 相似度定义不当会误合并 |

一个新手可理解的玩具例子是：把图像看成 $8\times8=64$ 个 patch，因此有 64 个视觉 token。先做一次 $2\times2$ 平均池化，每 4 个 token 变成 1 个，于是剩 16 个。再做一次 ToMe，把最相似的 token 两两合并，压缩比设为 0.5，于是剩 8 个。序列长度从 64 变成 8，注意力里 token 间两两比较的规模也同步明显下降。

真实工程里，压缩不是“越狠越好”，而是一个 Pareto 权衡。所谓 Pareto 权衡，可以直白理解为“同一时间不能同时把延迟、显存、精度都做到最优，只能找平衡点”。例如 DyMU 给出的多级压缩结果就说明，token 数减少后 FLOPs 大幅下降，但精度不会严格不变：

| Method | Token Count | FLOPs (Self-Attn) | VQA avg (%) |
| --- | --- | --- | --- |
| Full | 576 | 1.36 GFLOPs | 55.8 |
| DyMU-low | 89±27 | 72 MFLOPs | 54.5 (97.7% baseline) |
| DyMU-mid | 195±47 | 358 MFLOPs | 55.3 (99.1%) |
| DyMU-high | 394±57 | 1.27 GFLOPs | 56.0 (100.4%) |

结论可以压缩成一句话：视觉 Token 压缩不是附属优化，而是多模态推理系统的主路径优化，尤其是在边缘设备、长上下文、多图输入场景中。

---

## 问题定义与边界

问题先要说清楚。这里讨论的不是“把原始图片分辨率降低”这么简单，而是“图像已经过 ViT 或 CLIP 编码后，如何压缩中间的 token 序列”。也就是说，压缩对象是视觉编码器输出的表示，不是 JPEG 像素本身。

边界主要有三个。

第一，压缩目标是控制 token budget。所谓“token budget”，可以先理解成“系统允许视觉部分最多占多少个 token 名额”。因为在 VLM 里，视觉 token 和文本 token 通常共享后续上下文窗口。图像太长，会直接挤压文本上下文，或者把推理延迟拉高。

第二，压缩要保任务相关信息，不要求保全部信息。做 OCR、图表问答、细粒度定位时，图像边角的小文字可能很重要；做粗粒度图像描述时，大量背景 patch 可以丢掉。因此压缩不是无条件保真，而是保留对下游任务 $Y$ 有用的信息。

第三，压缩要看插入位置。你可以在视觉编码器后做一次固定压缩，也可以在多层 Transformer 内动态压缩。越靠前，省的算力越多；越靠后，通常越容易根据语义判断哪些 token 该保留。

这个问题可以用信息瓶颈来描述。信息瓶颈可以白话理解为“让表示尽量短，但别把任务必需的信息一起丢掉”。常见写法是：

$$
\min_{p(z|x)}\; I(X;Z) - \beta\, I(Z;Y)
$$

其中，$X$ 是原始视觉输入，$Z$ 是压缩后的 token 表示，$Y$ 是下游任务目标。$I(X;Z)$ 越小，表示越紧凑；$I(Z;Y)$ 越大，表示越有用。也可以写成约束形式：

$$
\min_{p(z|x)} I(X;Z),\quad \text{s.t. }I(Z;Y)\ge S_0,\ C(Z)\le C_0
$$

这里 $C(Z)$ 可以直接理解为“压缩后序列的成本”，比如 token 数、FLOPs、延迟或显存。

用一个具体数字更容易理解。假设 LLaVA-1.5 风格的视觉输入产生 576 个 token。如果不压缩，光 self-attention 的核心成本就已经不低；如果继续增加分辨率，token 数可能翻到上千，计算量会接近平方增长。DyMU-low 把 token 压到大约 89，FLOPs 降到原本的一个很小比例，但精度还保留在基线的 97.7% 左右。这说明边界不是“能不能压”，而是“压到哪里开始明显伤精度”。

所以，问题定义可以归结为：

1. 给定原始视觉 token 序列长度 $N_0$。
2. 选择压缩器，把它变成长度 $N_1$ 的新序列。
3. 在满足任务精度约束下，让 $N_1 \ll N_0$。
4. 最终优化的不是单一指标，而是速度、显存、精度三者的联合结果。

---

## 核心机制与推导

最基础的压缩机制是池化。池化可以理解成“把邻近区域做汇总”。如果原始 patch 网格是 $H\times W$，使用 $k\times k$ 的池化核，且不重叠，那么 token 数大致从

$$
N_0 = H\times W
$$

变成

$$
N_1 = \frac{H}{k}\times\frac{W}{k} = \frac{N_0}{k^2}
$$

例如 $24\times24=576$ 个 token，做一次 $2\times2$ 池化后就变成 $12\times12=144$ 个。这也是很多学习式投影器先做的第一步，因为它直接砍掉 75% 的长度，且实现很便宜。

但平均池化的问题是“一视同仁”。前景目标和无关背景会被同样处理，细节区域容易被抹平。所以第二类方法引入权重。注意力加权池化的直观做法是：先给每个 token 一个重要性分数 $a_i$，再做加权平均：

$$
z = \sum_{i=1}^{N_0} \alpha_i x_i,\quad
\alpha_i = \frac{\exp(a_i)}{\sum_j \exp(a_j)}
$$

这里 $x_i$ 是第 $i$ 个 token，$\alpha_i$ 是归一化后的权重。这样做的意义是，模型不再机械地保留空间邻近关系，而是按“哪些 token 对任务更重要”来聚合。

第三类是可学习压缩。可学习压缩可以直白理解成“不是人工规定怎么平均，而是让一个小网络自己学会怎么缩”。以 LDPv2 的简化形式为例，它先做通道投影，再做空间池化，再用 depthwise 卷积补局部结构：

$$
f_0 = \text{PW}_2(\text{GELU}(\text{PW}_1(f_v))),\quad
f_1 = \text{AvgPool}_{2\times2}(f_0),\quad
H_v = f_1 + \text{DW}(f_1)
$$

其中 `PW` 是 pointwise 线性投影，`DW` 是 depthwise 卷积。直白地说，这个模块先把特征变换到更适合 LLM 的空间，再做降采样，最后补一点局部模式，避免纯池化把结构信息丢太多。它把“压缩”和“模态对齐”合成了一个模块。

第四类是 Token Merging。Merging 不是删 token，而是把相似 token 合成一个新 token。设两个 token 相似度为 $S_{ij}$，可以写成点积或余弦相似度。若 $S_{ij}$ 高于阈值，就合并它们：

$$
x_{ij}' = \frac{w_i x_i + w_j x_j}{w_i + w_j}
$$

这里 $w_i,w_j$ 可以理解成各自的重要性或覆盖面积。和剪枝相比，合并的好处是信息没有被硬删除，而是被折叠进更短的序列里。

继续用前面的玩具例子。64 个 token 来自 $8\times8$ patch 网格。

1. 先做 $2\times2$ 平均池化，64 变成 16。
2. 对 16 个 token 计算相似度矩阵。
3. 把最相似的若干对 token 两两合并。
4. 若合并比例为 0.5，则 16 变成 8。

长度从 64 变成 8，压缩率是 87.5%。这时风险在于，合并后位置关系会变模糊。于是像 DyMU 这样的方案会维护“每个压缩 token 对应原来哪些位置”，并在注意力计算或后续重建时把这部分结构补回来。

真实工程例子可以看 MobileVLM V2。它面对的问题不是“学术上能不能压”，而是“Jetson Orin 这类设备能不能跑得动”。若视觉侧仍保留 576 个 token，跨模态推理的延迟和吞吐会很难看。LDPv2 把它压到 144，再叠加量化，系统吞吐才达到实际可部署的区间。这说明在边缘设备里，压缩不是锦上添花，而是上线条件。

---

## 代码实现

下面给一个可运行的 Python 玩具实现，演示两步压缩：先做 $2\times2$ 平均池化，再做一次简化版 ToMe 合并。代码不依赖深度学习框架，只用 `numpy`，方便把机制看清楚。

```python
import numpy as np

def avg_pool_tokens(tokens, grid_h, grid_w, kernel=2):
    """
    tokens: [N, D], where N = grid_h * grid_w
    """
    assert tokens.shape[0] == grid_h * grid_w
    assert grid_h % kernel == 0 and grid_w % kernel == 0

    d = tokens.shape[1]
    grid = tokens.reshape(grid_h, grid_w, d)

    out = []
    for i in range(0, grid_h, kernel):
        for j in range(0, grid_w, kernel):
            block = grid[i:i+kernel, j:j+kernel]
            out.append(block.mean(axis=(0, 1)))
    return np.stack(out, axis=0)

def cosine_similarity_matrix(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    x_norm = x / norm
    return x_norm @ x_norm.T

def tome_merge(tokens, keep_ratio=0.5):
    """
    Simplified ToMe:
    repeatedly merge the most similar pair until target length is reached.
    """
    assert 0 < keep_ratio <= 1
    x = tokens.copy()
    target = max(1, int(round(len(x) * keep_ratio)))

    while len(x) > target:
        sim = cosine_similarity_matrix(x)
        np.fill_diagonal(sim, -np.inf)
        i, j = np.unravel_index(np.argmax(sim), sim.shape)

        if i > j:
            i, j = j, i

        merged = (x[i] + x[j]) / 2.0
        keep = [k for k in range(len(x)) if k not in (i, j)]
        x = np.concatenate([x[keep], merged[None, :]], axis=0)

    return x

# Toy example: 8x8 patches => 64 visual tokens, dim = 4
np.random.seed(0)
tokens = np.random.randn(64, 4)

pooled = avg_pool_tokens(tokens, grid_h=8, grid_w=8, kernel=2)
merged = tome_merge(pooled, keep_ratio=0.5)

assert tokens.shape == (64, 4)
assert pooled.shape == (16, 4)   # 64 -> 16
assert merged.shape == (8, 4)    # 16 -> 8

# Attention cost proxy: sequence length squared
full_cost = tokens.shape[0] ** 2
compressed_cost = merged.shape[0] ** 2
assert full_cost == 4096
assert compressed_cost == 64
assert compressed_cost / full_cost == 1 / 64

print("original:", tokens.shape)
print("after pool:", pooled.shape)
print("after merge:", merged.shape)
print("attn cost ratio:", compressed_cost / full_cost)
```

这段代码表达了三个工程上很重要的点。

第一，池化和合并是两类不同操作。池化依赖空间网格，把局部邻近 patch 聚合；ToMe 依赖特征相似度，把语义相近的 token 合并。前者偏“几何降采样”，后者偏“表示压缩”。

第二，注意力成本近似和长度平方相关。上面的玩具例子里，64 个 token 的成本代理值是 $64^2=4096$，压到 8 个后是 $8^2=64$。真实模型里不会完全等于这个数字，因为还有投影、MLP、KV cache 等成本，但“长度下降会显著影响注意力成本”这个方向是对的。

第三，真实系统不会这么粗糙地全局找最相似对。DyMU 之类方法会在每层内维护 token 到原位置集合 $P_i$ 的映射，并引入阈值、预算和重建机制，避免把不该合并的结构性区域混在一起。其简化逻辑可以写成：

```text
x_1 = ViT_patch_projection(I)
P_1[t] = {t}
for i in 1..L:
    x_i, k_i = TransformerBlock(x_i, P_i)
    split tokens into A and B
    for t in A:
        t_B = argmax_{n in B} <k_i[t], k_i[n]>
        if similarity(t, t_B) > tau[i]:
            merge t into t_B
            P_i[t_B] = P_i[t_B] ∪ P_i[t]
            drop t
    x_{i+1} = compressed tokens
return x_{L+1}, P_{L+1}
```

在真实工程中，还常见一个 VTU 式重建步骤。它的作用可以白话解释为：“虽然中间为了省算力把 token 压短了，但在需要与原位置对齐时，再通过稀疏映射把它们投回近似原长度的结构上”。这样做能减少位置编码和后续语言层交互的错位。

---

## 工程权衡与常见坑

压缩系统最常见的误区，是只盯着 token 数，不看任务类型。视觉问答、文档理解、图表解析、细粒度 grounding，对局部细节的依赖差异很大。同样从 576 压到 144，图像描述任务可能几乎没感觉，OCR 任务却可能明显掉点。

第二个常见坑，是把 attention 分数直接当重要性真值。attention 可以先理解成“模型内部一层对另一层分配的关注权重”，但它不等于“这个 token 真有用”。在一些模型里，原始 attention 会对序列末端、padding 或某些固定位置有偏置。如果直接按分数裁剪，就可能保留大量无效 patch，反而丢掉主体区域。AdaTP 一类方法会先做去偏差，例如：

$$
\mathcal{A}_{\rm rel}(i)=\frac{\mathcal{A}_{\rm ori}(i)}{\mathcal{A}_{\rm bias}(i)+\varepsilon}
$$

它的意思很直接：先把“原始高分”除以“先天偏置”，得到相对更可靠的重要性。

第三个坑，是忽略结构多样性。图像里经常出现大片相似背景，比如天空、墙面、桌面。如果只看相似度，你可能一直保留同一类区域，造成表示塌缩。解决思路通常是把“语义相似”和“空间分散”一起考虑，而不是只按一个指标排序。

第四个坑，是压缩后位置失配。位置失配可以白话理解为“token 变少后，模型不知道它们原来在图上哪里”。对于依赖 RoPE 或其他位置编码的系统，这会影响跨模态对齐。DyMU 的 VTU 思路，本质就是给压缩后的 token 保留一条位置重建路径。

下面用表格汇总常见问题与规避方案：

| 问题 | 规避 |
| --- | --- |
| Attention 评分偏向 sequence end 或 padding | 先做 attention debias，再排序或聚合 |
| 结构冗余导致只保留重复 patch | 联合空间邻接与语义相似，保证多样性 |
| 压缩比设得过高，细节任务掉点严重 | 按任务设 token budget，别全场景同一比例 |
| 压缩后位置失配 | 保留位置映射，必要时做 VTU 式重建 |
| 只测平均精度，不测延迟和显存 | 同时看吞吐、峰值显存、精度三项 |

真实工程例子里，这些坑尤其明显。比如移动端问答系统若默认走 `DyMU-low`，可能在一般场景下看起来很划算，但一旦用户上传的是截图、表格、票据，细粒度文本区域就更容易被过度压缩。这就是为什么成熟系统常做“动态预算”：先快速判断图像复杂度，再决定用低、中、高三档 token 数，而不是固定一刀切。

---

## 替代方案与适用边界

ToMe 是最容易落地的替代方案之一。它的优点是接入成本低，适合“我不想重训模型，但希望先把推理速度提起来”的场景。对于 Stable Diffusion、ViT、部分 VLM 结构，设置 `ratio=0.5` 往往就能获得明显加速。它的边界是：当任务非常依赖局部精确结构时，过高的 merge ratio 会更容易伤质量。

PatchMerger、PiToMe、ATM 等方法，则可以理解成“不同风格的压缩器”。有的更强调训练后稳定性，有的更强调零训练插入，有的更强调严格控制精度损失。没有一种方法能在所有场景最好，所以要按部署条件选。

一个简洁对比如下：

| 方法 | FLOPs/Runtime ↓ | Accuracy Drop |
| --- | --- | --- |
| PatchMerger | 49–53% | ≤0.5% |
| ToMe | up to 2× speed | 0.2–0.4% |
| PiToMe | 40–60% | 0.5–0.7% |
| ATM | 30–40% | 0% |

适用边界大致可以这样判断：

1. 如果你要快速验证推理加速，优先考虑 ToMe 这类训练前可插拔方法。
2. 如果你要在边缘设备长期部署，且能做少量适配训练，可学习投影器如 LDPv2 往往更稳。
3. 如果输入复杂度波动很大，例如有时是自然图像，有时是文档截图，动态预算方法更合适。
4. 如果任务高度依赖空间结构恢复，必须考虑位置映射和重建，而不是只做纯粹删减。

一个真实工程判断标准是：如果系统瓶颈主要来自视觉序列太长，而且目标平台是手机、嵌入式设备、边缘 GPU，那么压缩几乎是必选项；如果你的系统本身图像分辨率不高、图像输入少、GPU 资源充足，压缩的收益可能没有那么大，甚至不值得引入额外复杂度。

---

## 参考资料

- Visual Token Compression: Enhancing Efficiency (emergentmind)  
- Visual Token Technology / 信息瓶颈公式 (emergentmind)  
- MobileVLM V2 architecture & LDPv2 (emergentmind)  
- DyMU: Dynamic Key Merging & VTU pseudocode (emergentmind)  
- AdaTP: Attention-Debiased Pruning (emergentmind)  
- Token merging (ToMe) deployment notes (Hugging Face Diffusers)  
- Token Merging for 3D Vision benchmarks (emergentmind)
