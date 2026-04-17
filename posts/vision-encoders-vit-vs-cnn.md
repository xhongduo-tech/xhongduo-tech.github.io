## 核心结论

视觉编码器的任务，是把二维像素转换成后续模型可用的语义表示。这里的“语义表示”指的是：已经压缩掉大量像素级冗余、能够更稳定表达对象、部件、位置和上下文关系的向量或 token。

CNN，卷积神经网络，核心方法是用局部卷积核在图像上做参数共享计算，先提取边缘、纹理、角点等低层模式，再通过多层堆叠把局部模式组合成更高层语义。它的强项是局部特征提取稳定、参数效率高、归纳偏置强，且在小数据、低延迟和端侧部署场景中通常更稳。

ViT，Vision Transformer，核心方法是先把图像切成固定大小的 patch，再把每个 patch 映射成 token，随后用自注意力让所有 token 在每一层都能彼此交互。它的强项是全局关系建模、与文本 token 形式对齐、模型规模继续增大时通常更容易吃到数据和算力红利。

结论可以直接概括成下面这张表：

| 维度 | CNN | ViT |
| --- | --- | --- |
| 基本建模方式 | 局部卷积，逐层扩张感受野 | patch token + 全局自注意力 |
| 主要优势 | 局部纹理建模稳定，小数据更稳，推理高效 | 全局关系建模强，易扩展，便于多模态对齐 |
| 归纳偏置 | 强，默认局部邻域重要 | 弱，需要靠数据学习空间规律 |
| 小数据表现 | 通常更可靠 | 容易过拟合或训练不稳定 |
| 大模型/大数据扩展 | 仍然很强，但扩展曲线常更早变缓 | 通常更容易随数据和参数继续提升 |
| 多模态适配 | 需要额外桥接设计 | token 形式天然接近文本侧 |
| 端侧部署 | 通常更友好 | 需精细压缩或改造才能落地 |

如果只记一句工程判断，那么是：

- 数据少、算力紧、目标偏局部纹理或实时部署，优先从 CNN 做基线。
- 数据大、预训练充分、任务依赖全局关系或多模态对齐，优先考虑 ViT。
- 两类优势都要，通常落到混合结构，而不是纯粹二选一。

在多模态任务里，这个差异更明显。图像要和文本一起建模时，ViT 输出的是 patch token 序列，文本模型输出的是词 token 序列，二者在接口层更容易直接拼接、对齐或做 cross-attention。因此 CLIP、Flamingo、BLIP、GIT 一类系统通常更偏向 ViT 或其变体作为视觉骨干。

---

## 问题定义与边界

本文讨论的问题，不是“谁绝对更先进”，而是下面三个更可落地的问题：

1. 视觉编码器如何把像素映射成语义 token。
2. CNN 与 ViT 在局部建模、全局建模和表示组织方式上的机制差异是什么。
3. 在数据量、算力、延迟、多模态需求不同的情况下，应当怎么选。

先把边界说清楚。视觉编码器会服务于分类、检测、分割、检索、多模态理解、图像生成条件编码等任务，但这些任务对模型的要求并不相同。分类更看重全局判别表示，检测和分割更依赖空间细节，检索更关心语义对齐，多模态理解还要求视觉表示能和文本表示在结构上对接。

| 边界条件 | 更偏 CNN 的场景 | 更偏 ViT 的场景 |
| --- | --- | --- |
| 数据量 | 小数据、标注少、增广有限 | 海量预训练数据或强自监督预训练 |
| 算力预算 | 端侧、低延迟、低功耗 | 训练资源充足，可接受更高吞吐成本 |
| 任务特点 | 纹理主导、局部模式明显、时延敏感 | 依赖长距离关系、复杂上下文、全局配准 |
| 多模态需求 | 较弱，图像单模态为主 | 很强，需要与文本深度对齐 |
| 工程目标 | 稳定、简单、可控、便宜 | 上限高、扩展性强、可与大模型协同 |

对新手来说，可以把视觉编码器理解成一个“表示转换器”：

- 输入端是像素矩阵，形状通常是 `H x W x C`。
- 输出端不是最终答案，而是一组中间表示。
- 后续分类头、检测头、分割头或语言模型，会继续消费这些表示。

因此本文不比较所有视觉模型，也不展开 Mamba、State Space、视觉扩散编码器等其他路线，而是集中比较两条主路线：

- CNN：层级局部聚合。
- ViT：patch 序列上的全局交互。

---

## 核心机制与推导

### 1. CNN：局部卷积到层级感受野

卷积的本质，是一个局部线性算子在空间维度上滑动。设输入特征图为 $x \in \mathbb{R}^{H \times W \times C_{in}}$，卷积核大小为 $k \times k$，输出通道数为 $C_{out}$，则单个输出位置只依赖于输入上的一个局部邻域。

简化写法为：

$$
h^{(l)} = \sigma\left(W^{(l)} * h^{(l-1)} + b^{(l)}\right)
$$

其中：

| 符号 | 含义 |
| --- | --- |
| $h^{(l)}$ | 第 $l$ 层输出特征图 |
| $W^{(l)}$ | 第 $l$ 层卷积核参数 |
| $*$ | 卷积运算 |
| $b^{(l)}$ | 偏置项 |
| $\sigma$ | 非线性激活，如 ReLU、GELU |

CNN 的两个关键性质是：

1. 参数共享  
同一个卷积核会在整张图上复用，因此模型不需要为每个位置单独学一套参数。

2. 局部连接  
每个输出位置只看输入的局部邻域，因此天然更关注边缘、纹理和局部结构。

新手最容易困惑的点是：既然卷积每次只看局部，为什么最后还能识别整张图里的对象？

答案是感受野。感受野指的是：某个高层神经元在原图上能“间接看到”的区域大小。卷积层一层层堆叠后，单个特征点覆盖到的原图范围会逐渐扩大。

若所有卷积核大小都为 $k=3$、步长为 1，不考虑空洞卷积，则感受野递推可以写成：

$$
R_l = R_{l-1} + (k-1), \quad R_0 = 1
$$

所以连续堆叠 3 层 $3\times3$ 卷积后，感受野大小为：

$$
R_3 = 1 + 3 \times (3-1) = 7
$$

这意味着第 3 层的一个位置，已经能汇聚原图上 $7 \times 7$ 区域的信息。若再叠加下采样、池化或步长卷积，感受野会增长得更快。

一个最小数值例子：

| 层数 | 操作 | 理论感受野 |
| --- | --- | --- |
| 0 | 输入像素 | $1 \times 1$ |
| 1 | `3x3 conv` | $3 \times 3$ |
| 2 | `3x3 conv` | $5 \times 5$ |
| 3 | `3x3 conv` | $7 \times 7$ |

因此 CNN 的逻辑不是“一上来就全局理解”，而是“先从局部统计稳定模式，再逐层向上聚合”。

### 2. ViT：patch 切分到全局注意力

ViT 不直接在像素网格上做深层卷积，而是先把图像切成固定大小的 patch。设输入图像大小为 $H \times W \times C$，patch 边长为 $P$，则 patch 数量为：

$$
N = \frac{H}{P} \cdot \frac{W}{P}
$$

每个 patch $x_i \in \mathbb{R}^{P \times P \times C}$ 展平后，得到向量：

$$
\text{Flatten}(x_i) \in \mathbb{R}^{P^2 C}
$$

再乘一个可学习线性投影矩阵 $E \in \mathbb{R}^{P^2C \times D}$，映射成隐藏维度为 $D$ 的 token：

$$
e_i = \text{Flatten}(x_i) E
$$

如果没有位置编码，Transformer 只能看到一串 token，却不知道“哪个 token 在左上角，哪个 token 在右下角”。因此还要加位置编码 $p_i$：

$$
z_i^{(0)} = e_i + p_i
$$

分类任务里，通常还会额外引入一个 `CLS` token 作为全局汇聚位：

$$
T^{(0)} = [\text{CLS}; z_1^{(0)}; z_2^{(0)}; \dots; z_N^{(0)}]
$$

它和 CNN 的差别可以直接总结为：

- CNN 的基本单元是空间局部算子。
- ViT 的基本单元是序列 token。
- CNN 先保留局部结构，再逐层扩大全局视野。
- ViT 从第一层开始就把输入组织成可全局交互的序列。

### 3. 自注意力为什么能建模全局关系

ViT 的核心不是 patch 本身，而是 patch 之间的自注意力。对输入 token 矩阵 $X \in \mathbb{R}^{N \times D}$，模型会线性映射出三组向量：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

其中：

| 符号 | 直白解释 |
| --- | --- |
| $Q$ | 当前 token 想查询什么信息 |
| $K$ | 当前 token 可以被怎样匹配 |
| $V$ | 当前 token 真正携带的内容 |

缩放点积注意力公式为：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

这个公式可以拆成三步理解。

第一步，计算相关性：

$$
S = \frac{QK^\top}{\sqrt{d_k}}
$$

矩阵 $S_{ij}$ 表示第 $i$ 个 token 与第 $j$ 个 token 的匹配程度。

第二步，做行方向 softmax：

$$
A = \text{softmax}(S)
$$

于是第 $i$ 个 token 会得到一组和为 1 的权重，表示它应当从其他 token 取回多少信息。

第三步，做加权汇总：

$$
Y = AV
$$

这样每个 token 的新表示，都来自“自己”和“其他所有 token”的内容融合。

这就是 ViT 能较早建模长距离关系的根本原因。若图像左侧是人、右侧是狗，中间隔着草地，那么：

- CNN 在浅层主要还是局部局部地看，跨区域关系往往要靠更深层传播。
- ViT 在第一层就允许“人对应的 token”和“狗对应的 token”直接交互。

这并不代表 ViT 在任何情况下都一定更好。它只是说明：ViT 的计算图从结构上允许更早的全局依赖，而 CNN 的全局关系通常来自多层局部传播后的间接形成。

### 4. ViT-B/16 的最小数值例子

以最常见的 `ViT-B/16` 为例，输入图像大小是 `224 x 224 x 3`，patch 大小是 `16 x 16`。

patch 数量为：

$$
N = (224/16)^2 = 14^2 = 196
$$

每个 patch 展平后的维度是：

$$
16 \times 16 \times 3 = 768
$$

若嵌入维度设为 $D=768$，那么 patch embedding 层做的事情就是把每个长度 768 的 patch 向量，再映射到长度 768 的隐藏向量。这里“768”相同只是该配置下的巧合，不是必须相等。

因此 `ViT-B/16` 在输入 Transformer 编码器时，序列长度通常是：

$$
196 + 1 = 197
$$

其中：

- `196` 是 patch token 数量。
- `1` 是 `CLS` token。

为什么分辨率和 patch 大小会直接影响成本？因为注意力矩阵大小与 token 数平方相关。

若图像从 `224 x 224` 变成 `448 x 448`，patch 仍为 `16 x 16`，则 token 数变成：

$$
(448/16)^2 = 28^2 = 784
$$

注意力矩阵从大约 $196^2$ 增长到 $784^2$，增加了 16 倍。这里正是 ViT 在高分辨率场景下显存和计算快速膨胀的原因。

---

## 代码实现

下面给两个最小可运行例子：

1. `patch embedding`，演示图像如何变成 token。
2. `single-head self-attention`，演示 token 如何彼此交互。
3. 一个最小 CNN 卷积例子，帮助对比“局部滑动”和“全局交互”的差别。

代码只依赖 `numpy`，可以直接运行。

```python
import numpy as np


def patchify(image, patch_size):
    """
    image: (H, W, C)
    return: (N, patch_size * patch_size * C)
    """
    if image.ndim != 3:
        raise ValueError("image must have shape (H, W, C)")

    H, W, C = image.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError("H and W must be divisible by patch_size")

    patches = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size, :]
            patches.append(patch.reshape(-1))
    return np.stack(patches, axis=0)


def patch_embedding(image, patch_size, embed_dim, seed=0):
    """
    return:
      tokens: (N, embed_dim)
      proj:   (patch_dim, embed_dim)
      pos:    (N, embed_dim)
    """
    patches = patchify(image, patch_size)
    N, patch_dim = patches.shape

    rng = np.random.default_rng(seed)
    proj = rng.normal(loc=0.0, scale=0.02, size=(patch_dim, embed_dim))
    pos = rng.normal(loc=0.0, scale=0.02, size=(N, embed_dim))

    tokens = patches @ proj + pos
    return tokens, proj, pos


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def self_attention(tokens, seed=1):
    """
    tokens: (N, D)
    return:
      out:     (N, D)
      weights: (N, N)
    """
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (N, D)")

    N, D = tokens.shape
    rng = np.random.default_rng(seed)

    Wq = rng.normal(loc=0.0, scale=0.02, size=(D, D))
    Wk = rng.normal(loc=0.0, scale=0.02, size=(D, D))
    Wv = rng.normal(loc=0.0, scale=0.02, size=(D, D))

    Q = tokens @ Wq
    K = tokens @ Wk
    V = tokens @ Wv

    scores = Q @ K.T / np.sqrt(D)
    weights = softmax(scores, axis=1)
    out = weights @ V
    return out, weights


def conv2d_single_channel(image, kernel, stride=1):
    """
    image:  (H, W)
    kernel: (kH, kW)
    return: (out_H, out_W)
    """
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must be 2D arrays")

    H, W = image.shape
    kH, kW = kernel.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1

    out = np.zeros((out_H, out_W), dtype=np.float32)
    for i in range(out_H):
        for j in range(out_W):
            h0 = i * stride
            w0 = j * stride
            region = image[h0:h0 + kH, w0:w0 + kW]
            out[i, j] = np.sum(region * kernel)
    return out


def main():
    # 8x8 RGB toy image
    image = np.arange(8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)

    # ViT-style patch embedding
    tokens, proj, pos = patch_embedding(image, patch_size=4, embed_dim=16, seed=0)
    out, weights = self_attention(tokens, seed=1)

    assert tokens.shape == (4, 16)
    assert out.shape == (4, 16)
    assert weights.shape == (4, 4)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)

    print("Patch tokens shape:", tokens.shape)
    print("Attention output shape:", out.shape)
    print("Attention weights shape:", weights.shape)
    print("Row sums of attention weights:", weights.sum(axis=1))

    # CNN-style local convolution on one grayscale channel
    gray = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    edge_kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ], dtype=np.float32)

    conv_out = conv2d_single_channel(gray, edge_kernel, stride=1)
    assert conv_out.shape == (3, 3)

    print("Conv output shape:", conv_out.shape)
    print("Conv output:")
    print(conv_out)


if __name__ == "__main__":
    main()
```

如果运行这段代码，可以得到三个直观结论：

1. `patchify` 会把二维图像切成若干个固定大小的 patch，再展开成一组向量。
2. 自注意力的权重矩阵形状是 `(N, N)`，说明每个 token 都会考虑所有 token。
3. 卷积输出只来自局部滑动窗口，说明 CNN 的基本计算单元确实是局部算子。

把这个过程映射到真实模型，可以得到两条典型流水线。

CNN 流水线：

1. 输入 `H x W x C`
2. 多层卷积、归一化、激活、下采样
3. 得到多尺度特征图
4. 分类头或检测头消费特征图

ViT 流水线：

1. 输入 `H x W x C`
2. 切成 `P x P` patch
3. 线性映射成 token
4. 加位置编码
5. 输入多层 Transformer block
6. 取 `CLS` token 或所有 patch token 作为表示

在图文模型里，ViT 的一个明显好处是输出天然已经是 token 序列。比如在 CLIP 里，视觉编码器输出图像表示，文本编码器输出文本表示，再通过对比学习把匹配图文拉近；在 Flamingo、BLIP-2 这类系统里，还会进一步让文本 token 通过 cross-attention 查询视觉 token。

---

## 工程权衡与常见坑

工程里最常见的问题，不是“听说 ViT 更强”，而是“为什么在自己的数据和硬件上并不强”。

| 常见坑 | 发生原因 | 规避方式 |
| --- | --- | --- |
| 小数据直接训练 ViT 效果差 | 归纳偏置弱，模型容易记忆训练集噪声 | 用 ImageNet 预训练权重，或先做 MAE / DINO 这类自监督预训练 |
| ViT 显存高、训练慢 | 注意力复杂度随 token 数近似按 $O(N^2)$ 增长 | 增大 patch、降低输入分辨率、使用窗口注意力或分层 ViT |
| CNN 难捕捉长距离关系 | 局部卷积需要多层传播才能形成全局依赖 | 加全局池化、空洞卷积、注意力模块，或直接混合 Transformer |
| ViT 对细粒度定位不稳 | patch 太大时，小目标和边缘细节会被过早合并 | 采用更小 patch、分层结构、卷积 stem、多尺度特征 |
| 端侧部署效果不理想 | Transformer 访存模式和大矩阵乘开销对设备不友好 | 优先轻量 CNN，或选 MobileViT、EfficientFormer 一类轻量结构 |
| 迁移学习不稳定 | 学习率、正则化、数据增广设置不匹配 | 使用论文推荐超参，先复现成熟 baseline，再做结构改动 |

这里有两个判断必须明确。

第一，ViT 的“弱归纳偏置”既是优势，也是成本。  
优势在于，它不被固定局部结构限制，模型容量足够、数据足够时，能学到更灵活的空间关系。  
成本在于，小数据时这种自由度容易变成过拟合和训练不稳定。

第二，ViT 的全局注意力不是免费能力。  
若 token 数为 $N$，则单头注意力矩阵大小通常是 $N \times N$，时间和显存成本都会随着分辨率快速上升。设输入分辨率是 `224x224`、patch 为 `16x16`，则 $N=196$；若改成 patch 为 `8x8`，则：

$$
N = (224/8)^2 = 28^2 = 784
$$

注意力矩阵规模从：

$$
196^2 = 38{,}416
$$

上升到：

$$
784^2 = 614{,}656
$$

增长约 16 倍。这还没把多头、batch 和反向传播算进去。

对新手来说，一个实用的选择规则是：

| 条件 | 更稳的起点 |
| --- | --- |
| 数据量不到几万张，且无强预训练 | CNN |
| 已有高质量预训练权重，可接受较高训练成本 | ViT |
| 任务依赖细粒度纹理，如工业缺陷、医学局部结构 | 先 CNN，再考虑混合结构 |
| 任务依赖图文对齐、区域关系、复杂上下文 | ViT 或分层 ViT |
| 设备是手机、边缘盒子、摄像头 SoC | 轻量 CNN 优先 |

实践里常见流程通常不是“从零训练一个巨大 ViT”，而是：

1. 先选成熟预训练骨干。
2. 在自己任务上微调。
3. 再根据瓶颈决定是否改 patch、改输入分辨率、加多尺度结构，或者换回 CNN / Hybrid。

---

## 替代方案与适用边界

真实工程里很少只在“纯 CNN”和“纯 ViT”之间二选一，更常见的是三类路线并存。

| 路线 | 代表思路 | 适合场景 | 局限 |
| --- | --- | --- | --- |
| CNN-only | ResNet、ConvNeXt、EfficientNet | 小中型数据、边缘部署、局部纹理任务 | 全局关系建模通常需要额外设计 |
| ViT-only | ViT、DeiT、MAE 预训练 ViT | 大规模预训练、多模态、高上限任务 | 对数据、算力、超参更敏感 |
| Hybrid | CNN stem + Transformer，或分层混合架构 | 同时想要局部稳定性和全局建模能力 | 结构复杂，调参与部署成本更高 |

这里有一个常见误解：多模态系统偏向 ViT，不等于卷积不能做多模态。

更准确的说法是：

- CNN 也能作为视觉编码器接入文本模型。
- 但 ViT 输出的是 patch token 序列，和文本 token 的接口形式更一致。
- 当系统要做 cross-attention 时，视觉侧若已经是 token 序列，桥接成本通常更低。

例如图像问答里，问题可能是“左上角的红色标志旁边是什么物体”。这类问题不仅要求识别对象，还要求保留局部位置关系，并让文本侧按需访问对应视觉区域。ViT 或分层 Transformer 在这种任务里往往更自然。

但低资源场景依然是 CNN 的强势区域。移动端识图、工业视觉、实时视频分析，通常更看重：

- 延迟
- 吞吐
- 功耗
- 部署稳定性
- 算法栈成熟度

这些指标下，轻量 CNN 或混合结构往往比大 ViT 更现实。

因此选择边界可以概括成下面三条：

- 数据大、预训练强、任务需要全局关系和跨模态对齐，优先 ViT。
- 数据小、算力紧、部署要求高、目标依赖局部纹理，优先 CNN。
- 局部归纳偏置和全局建模能力都想保留，优先考虑 Hybrid，而不是执着于纯路线。

最后给一个更工程化的选型表：

| 任务/约束 | 更推荐的起点 |
| --- | --- |
| 图像分类基线、数据中小、快速落地 | ResNet / ConvNeXt |
| 图文检索、零样本分类 | ViT + 对比学习框架 |
| 文档理解、图表问答、图像问答 | ViT 或分层 ViT |
| 工业缺陷检测、边缘设备部署 | 轻量 CNN |
| 既要检测细节，又要建模全局关系 | CNN + Transformer 混合结构 |

---

## 参考资料

1. Dosovitskiy et al., 2021, *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*  
核心贡献：提出标准 ViT，把图像切成 patch token 后直接套用 Transformer 编码器。  
建议阅读方式：先看 patch embedding、位置编码和 `ViT-B/16` 配置，再看数据规模对性能的影响。  
链接：https://arxiv.org/abs/2010.11929

2. He et al., 2016, *Deep Residual Learning for Image Recognition*  
核心贡献：ResNet 证明了深层 CNN 可以稳定训练，是理解现代 CNN 主线的重要起点。  
建议阅读方式：重点看残差连接如何解决深层网络优化问题，不必先陷入全部实验细节。  
链接：https://arxiv.org/abs/1512.03385

3. Touvron et al., 2021, *Training data-efficient image transformers & distillation through attention*  
核心贡献：DeiT 说明 ViT 在更有限数据条件下也能被训练好，关键在于蒸馏和训练策略。  
建议阅读方式：把它和原始 ViT 对照看，理解“为什么 ViT 并非只能依赖超大数据集”。  
链接：https://arxiv.org/abs/2012.12877

4. Liu et al., 2021, *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*  
核心贡献：用分层结构和窗口注意力降低成本，并增强多尺度视觉任务适配性。  
建议阅读方式：重点看“窗口注意力 + shifted window”如何在效率与全局信息之间折中。  
链接：https://arxiv.org/abs/2103.14030

5. He et al., 2022, *Masked Autoencoders Are Scalable Vision Learners*  
核心贡献：说明 ViT 在自监督预训练下能显著受益，这也是今天很多视觉大模型的关键训练路线。  
建议阅读方式：先看掩码重建任务定义，再看为什么高掩码率反而有效。  
链接：https://arxiv.org/abs/2111.06377

6. Radford et al., 2021, *Learning Transferable Visual Models From Natural Language Supervision*  
核心贡献：CLIP 证明了 ViT 在图文对齐、多模态迁移和零样本识别中的工程价值。  
建议阅读方式：重点看双塔结构、对比损失和零样本分类构造方式。  
链接：https://arxiv.org/abs/2103.00020
