## 核心结论

LePE，Locally Enhanced Position Encoding，中文可理解为“局部增强位置编码”。它的核心不是给每个 token 额外查一张位置表，而是在注意力输出之外，针对 Value 分支再做一次逐通道深度卷积。深度卷积，意思是“每个通道单独卷，不混通道”，因此它天然适合把“邻居关系”作为局部位置偏置信号加回去。

标准自注意力擅长建模远距离依赖，也就是“谁和谁相关”。LePE 解决的是另一个问题：相关不等于相邻。两个 token 即使语义相似，也未必告诉模型它们在局部空间里是否紧挨着。于是 LePE 把局部卷积得到的邻域信息与全局注意力结果相加，让每个 token 同时带有两类信息：

| 组成部分 | 主要回答的问题 | 擅长的信息 |
| --- | --- | --- |
| Global Attention | 谁与我全局相关 | 长距离依赖、跨区域关联 |
| LePE 局部补偿 | 我的邻居是谁 | 边缘、纹理、局部结构 |
| 合并结果 | 谁相关且谁靠近我 | 全局语义 + 局部几何 |

玩具例子可以这样理解：注意力已经告诉模型“远处那个 token 很重要”，LePE 再补一句“你左右两边这两个 token 也很关键，因为它们构成了局部结构”。对图像来说，这类结构可能是边缘、轮廓、角点；对序列来说，这类结构可能是局部模式或短程依赖。

真实工程里，CSWin Transformer 把 LePE 放进 cross-shaped window attention 之后，使模型既保留窗口内外的全局关系，又不丢局部空间结构。在 ImageNet、COCO、ADE20K 这类视觉任务上，这种设计被证明有效，说明 LePE 不是概念补丁，而是能稳定带来收益的结构组件。

---

## 问题定义与边界

先定义问题。标准自注意力通常写成：

$$
\text{Attn}(Q,K,V)=\text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

其中 $Q,K,V$ 分别是 Query、Key、Value，可以理解为“我在找什么”“别人有什么特征”“最终取回什么内容”。问题不在于这个公式不会建模关系，而在于它对“邻接位置”没有直接偏好。Softmax 只看匹配分数，谁分高就关注谁，但“离得近”本身不会自动变成强约束。

这在高分辨率图像里尤其明显。图像不是一堆无序 token，而是有空间拓扑的。相邻 patch 往往构成边缘和纹理。如果模型只看全局相似度，就可能把局部空间上的连续性弱化掉。结果是：全局语义够强，但细节边界、局部纹理、邻近结构变差。

边界要说清楚。LePE 不是替代 attention，而是补充 attention。它不改 Softmax 的全局分配规则，不直接干预 $QK^\top$ 的相关性矩阵，而是在 Value 路径上加局部位置偏置。也就是说，LePE 的职责是“补局部”，不是“重写全局”。

| 问题 | 失败后果 | LePE 责任 |
| --- | --- | --- |
| 只靠全局相似度 | 邻居关系被弱化 | 给 Value 注入局部偏置 |
| 高分辨率细节多 | 边缘、纹理容易变糊 | 保留短程结构 |
| 不想破坏全局注意力 | 全局上下文可能被局部规则压制 | 在 attention 之外做加法补偿 |
| 分辨率变化大 | 固定位置表泛化受限 | 用卷积做相对稳定的局部建模 |

新手版例子：假设有 3 个连续像素 A、B、C。标准注意力会问“B 和谁最像”，但不会天然认为 A、C 因为挨着 B 就更重要。LePE 会对 Value 再扫一遍邻域，把 A 和 C 对 B 的局部影响显式加进去。这样模型不仅知道“谁像我”，还知道“谁在我旁边”。

因此，LePE 的适用边界也很明确：它最适合有明确局部拓扑的 token 排列，比如图像 patch、视频局部块、或能定义邻接关系的序列。如果 token 本身没有稳定邻接结构，卷积的局部归纳偏置就未必成立。

---

## 核心机制与推导

LePE 的核心公式可以直接写成：

$$
\text{Output}=\text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V+\text{DWConv}(V)
$$

这里的 DWConv 是 depthwise convolution，中文叫“逐通道深度卷积”。白话解释是：每个通道自己看自己的邻居，不和别的通道混在一起。这样做有两个好处。第一，参数量低；第二，它更像在每个通道上单独加一个局部位置滤波器，而不是重新做一次复杂的特征融合。

为什么是对 Value 做卷积，而不是对 Query 或 Key 做卷积？因为 Query 和 Key 主要负责算“相关性”，Value 负责承载“被取回的内容”。LePE 的思路不是改相关性矩阵，而是给最终取回的内容补上局部结构。因此把卷积放在 Value 路上更符合职责分离。

玩具例子可以用一维序列说明。设 3 个 token 的单通道 value 是：

$$
V=[1,0,1]
$$

使用卷积核 $[0.5,0,0.5]$，并做 same padding，那么中间位置会收到左右邻居的平均信息，DWConv 输出是：

$$
\text{DWConv}(V)=[0,1,0]
$$

如果 attention 输出是：

$$
\text{AttnOut}=[0.8,0.6,0.4]
$$

两者相加后得到：

$$
[0.8,1.6,0.4]
$$

这表示第二个 token 不仅保留了全局注意力汇聚来的结果，还因为它左右两边都有信号而得到局部增强。直观上，这就是“全局看关系，局部看邻居”。

如果把例子扩展到题目给出的三组 value 向量 $[1,2]$、$[0,1]$、$[1,0]$，逐通道卷积本质上是在每个特征维度上分别执行同样的邻域聚合，再把得到的位置偏置加回 attention 输出。这里“位置偏置”不是绝对坐标，而是由邻域结构诱导出来的相对局部信息。

LePE 和 GLA 的关系也可以统一理解。GLA，Gated Linear Attention，中文可理解为“带门控的线性注意力”，门控就是“让模型决定保留多少旧信息、接收多少新信息”。其简化递推常写为：

$$
S_t = G_t \odot S_{t-1} + k_t^\top v_t,\qquad o_t=q_t S_t
$$

其中 $G_t$ 控制全局记忆的遗忘与保留，$v_t$ 提供当前输入的内容。如果在 $v_t$ 进入记忆更新前，先注入 $\text{DWConv}(v_t)$ 形式的局部信号，那么递推就同时拥有“全局记忆更新”和“局部结构增强”。前者负责长程依赖，后者负责短程结构。两者不是冲突关系，而是分工关系。

真实工程例子是视觉骨干网络。CSWin 的 cross-shaped window attention 能覆盖横向和纵向更长的视野，但窗口注意力本身仍可能稀释邻近 patch 的精细关系。LePE 在 attention 后对 Value 做局部卷积，相当于在跨窗口全局建模之外，再补一条不依赖窗口相似度的局部通路。这就是它能在分类、检测、分割三个任务上同时受益的原因。

---

## 代码实现

下面给一个最小可运行的 Python 版本。它不依赖深度学习框架，只用列表实现单头 attention 和单通道 depthwise 1D 卷积，目的是把公式和数据流讲清楚。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def matmul_vec(weights, values):
    # weights: [n], values: [n]
    return sum(w * v for w, v in zip(weights, values))

def attention_1d(q, k, v):
    # 单头、单查询示例：输出一个标量
    scores = [qi * ki / math.sqrt(1.0) for qi, ki in zip([q] * len(k), k)]
    probs = softmax(scores)
    out = matmul_vec(probs, v)
    return out, probs

def depthwise_conv1d_same(v, kernel):
    # 单通道 same padding
    pad = len(kernel) // 2
    padded = [0.0] * pad + v + [0.0] * pad
    out = []
    for i in range(len(v)):
        s = 0.0
        for j, w in enumerate(kernel):
            s += padded[i + j] * w
        out.append(s)
    return out

def lepe_merge(attn_outs, v, kernel, scale=1.0):
    pos_bias = depthwise_conv1d_same(v, kernel)
    merged = [a + scale * p for a, p in zip(attn_outs, pos_bias)]
    return merged, pos_bias

# 玩具例子
v = [1.0, 0.0, 1.0]
kernel = [0.5, 0.0, 0.5]

# 假设每个位置各自得到一个 attention 输出
attn_outs = [0.8, 0.6, 0.4]
merged, pos_bias = lepe_merge(attn_outs, v, kernel, scale=1.0)

assert pos_bias == [0.0, 1.0, 0.0]
assert merged == [0.8, 1.6, 0.4]

# 再验证一个注意力权重例子
out, probs = attention_1d(q=1.0, k=[1.0, 0.0, 1.0], v=v)
assert abs(sum(probs) - 1.0) < 1e-9
assert out > 0.5

print("pos_bias =", pos_bias)
print("merged =", merged)
print("attention_probs =", probs)
print("attention_out =", out)
```

如果写成更接近深度学习框架的伪代码，结构通常是这样：

```python
attn = softmax(Q @ K.transpose(-2, -1) / sqrt(d)) @ V
pos_bias = depthwise_conv1d(V)   # 或按 2D patch 做 depthwise_conv2d
output = attn + scale * pos_bias
```

这里 `scale` 是可学习缩放参数，白话解释就是“控制局部补偿有多大”。很多工程实现里会把它设计成标量、逐通道参数，或者门控值 `sigmoid(gate)`。原因很简单：attention 与卷积来自两条不同统计性质的分支，直接裸加，比例未必稳定。

实现时有几个关键点。

1. 通道独立。LePE 的卷积应保持 depthwise 语义，否则就从“位置补偿”变成“重新混合特征”。
2. padding 对齐。输出长度必须与 token 长度一致，否则无法和 attention 输出逐元素相加。
3. 维度匹配。图像任务常在 $B \times H \times W \times C$ 或 $B \times N \times C$ 两种布局间切换，卷积前后要明确 reshape 逻辑。
4. 比例控制。实践中常加入 `scale`、`gate` 或归一化层，避免局部偏置信号过强。

---

## 工程权衡与常见坑

LePE 便宜、直接、可插拔，但它不是“加一层卷积就一定更好”。最大的问题是比例失衡。如果 attention 分支的数值范围较小，而 DWConv 输出较大，那么相加之后局部信号会压过全局信号，模型就会变成“伪卷积网络”。

| 典型坑 | 触发条件 | 缓解策略 |
| --- | --- | --- |
| 局部信号过强 | 直接裸加，未做缩放 | 加 `alpha`、门控或归一化 |
| 过度平滑 | kernel 太大、层数太深 | 减小 kernel，分阶段使用 |
| 长度错位 | padding 或 reshape 错误 | 强制 same padding，检查维度 |
| 失去通道独立性 | 误用普通卷积 | 明确使用 depthwise |
| 分辨率迁移不稳 | 窗口长度与卷积感受野不匹配 | 联合调 window size 与 kernel |

新手容易忽略的一点是，局部增强不等于局部越强越好。假设在线性注意力或门控注意力里直接做：

```python
output = linear_attn_out + dwconv(V)
```

如果 `dwconv(V)` 的方差明显高于 `linear_attn_out`，训练可能立刻偏向局部模式，导致长程依赖退化。很多改进工作之所以引入门控、小波域增强、可学习缩放，本质上都是在解决这个比例控制问题。更稳妥的形式通常是：

```python
output = linear_attn_out + sigmoid(gate) * dwconv(V)
```

这里的 `gate` 可以理解为“让模型自己决定这一层要不要更相信局部信号”。

另一个常见坑是把 LePE 想成万能位置编码。它不是绝对位置编码，不能单独告诉模型“这是第 37 个 token”。它更像局部相对结构增强器，擅长表达“你周围发生了什么”。所以在某些任务里，LePE 仍需要与绝对位置、相对位置、旋转位置编码等方案配合，而不是替代一切位置机制。

---

## 替代方案与适用边界

如果目标只是补局部信息，LePE 不是唯一方案。常见替代方法包括相对位置偏置、局部窗口注意力、卷积式相对位置编码等。

| 方案 | 局部能力 | 额外参数/开销 | 适合场景 |
| --- | --- | --- | --- |
| Relative Position Bias | 中等 | 需要位置偏置表 | 标准 Transformer |
| Window Attention | 强 | 需要窗口划分 | 高分辨率视觉任务 |
| Local Attention | 强 | 计算局部邻域注意力 | 长序列降算力场景 |
| LePE | 强 | 一层 depthwise conv | 需要全局 + 局部并存的模型 |

和滑动窗口注意力相比，LePE 的优势是插入成本低。它不要求你把 attention 机制重写成局部窗口版本，也不要求维护复杂的位置查表。只要已有 attention 输出和 Value 分支，就能把 depthwise convolution 插进去。对于已有架构，这种“低侵入性”非常有工程价值。

但 LePE 也有适用边界。第一，它默认 token 存在合理邻接关系，所以在图像和视频里天然适合；在文本里是否有效，要看 token 排列能否从卷积中受益。第二，它的局部感受野固定，无法像全局注意力那样自适应覆盖远距离位置。第三，如果任务本身主要依赖全局顺序而不是局部几何，LePE 的收益可能有限。

可以把几种方案的分工这样记：

1. 如果你缺的是“全局谁重要”，优先 attention。
2. 如果你缺的是“邻居如何组成局部结构”，LePE 很合适。
3. 如果你需要严格局部稀疏计算，window/local attention 更直接。
4. 如果你需要明确的位置索引信息，相对或绝对位置编码仍然必要。

因此，LePE 最适合的场景不是“只做局部”或“只做全局”，而是两者都需要、且希望结构尽量简单的时候。

---

## 参考资料

1. CSWin Transformer. CVPR 2022. 论文页面标题常见为 “CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows”。介绍了 LePE 在 cross-shaped window block 中的实现方式及在 ImageNet、COCO、ADE20K 上的结果。  
2. Global-Local Self-Attention Transformer. Applied Sciences, 2023. 讨论了 attention 后结合深度卷积提取局部位置信息的思路，适合理解全局与局部协同建模。  
3. Emergent Mind 对 Gated Linear Attention 的综述页面，标题为 “Gated Linear Attention (GLA)”。适合理解门控记忆更新与局部增强模块如何组合。  
4. ReadKong 上的 CSWin Transformer 论文镜像，可用于查看 LePE 公式与结构图。链接：https://www.readkong.com/page/cswin-transformer-a-general-vision-transformer-backbone-3717962  
5. MDPI Applied Sciences 页面，可补充阅读 global-local self-attention 的实现背景。链接：https://www.mdpi.com/2076-3417/12/19/10154  
6. MDPI Electronics 中关于局部增强与融合策略的讨论，可用于理解为何工程上常需要门控或缩放。链接：https://www.mdpi.com/2079-9292/14/7/1246  
7. Emergent Mind GLA 综述链接：https://www.emergentmind.com/articles/gated-linear-attention-gla
