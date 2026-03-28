## 核心结论

2D 位置编码的任务，是把“这个 patch 在二维平面上的哪里”注入到 Transformer。patch 可以理解成“把图像切成很多小方块后，每个小方块对应的 token”。如果没有位置编码，注意力只看到一堆向量，不知道左上角和右下角的区别。

对二维输入，主流方案有四类：

| 方案 | 核心做法 | 优点 | 短板 | 典型场景 |
| --- | --- | --- | --- | --- |
| 可学习 2D 绝对位置嵌入 | 每个网格位置学一个独立向量 | 简单直接 | 分辨率变化时外推差 | 固定分辨率 ViT |
| 分解式 2D RoPE | x 轴和 y 轴各占一半维度，分别做旋转编码 | 相对位置一致，扩展到新分辨率更稳 | 实现比 absolute 稍复杂 | ViT、多模态模型 |
| 2D ALiBi | 在 attention logit 上加与行列距离成线性的偏置 | 参数少，易接入 | 方向设计不当会退化 | 文档、表格、版面理解 |
| 动态分辨率 + 1D/2D RoPE | 多个 tile 展平成序列，再统一编码 | 工程上最灵活 | token 数暴涨 | 高分辨率 VisionLLM |

最关键的判断标准不是“哪种更先进”，而是“输入分辨率和纵横比会不会变”。如果输入长期固定在 `224×224`，绝对位置嵌入仍然可用；如果输入会从 `14×14 patch` 变成 `22×44 patch`，或者需要把多张子图拼成一个序列，RoPE 或 ALiBi 更稳。

2D RoPE 的核心性质可以写成：

$$
R_{(i,j)}^\top R_{(k,l)} = R_{(k-i,l-j)}
$$

意思是：两个位置之间的相互作用，只依赖相对偏移 $(\Delta x,\Delta y)$，而不是依赖它们各自的绝对编号。这正是它适合多分辨率泛化的原因。

---

## 问题定义与边界

二维输入的问题，不是“有没有顺序”，而是“有没有平面坐标”。文本只有前后顺序，图像、视频帧、表格单元格、文档块都有横向和纵向两个方向。模型必须区分：

1. 两个 token 是否在同一行
2. 是否在同一列
3. 谁在左，谁在右
4. 谁在上，谁在下
5. 距离有多远

如果只把二维网格按 row-major，也就是“按行展开”的方式拉成一维序列，模型虽然拿到了顺序，但没有天然拿到完整的二维几何关系。比如序列上相邻的两个 token，可能真的是左右相邻，也可能是一行结尾和下一行开头。

另一个边界是成本。设图像大小为 $H\times W$，patch 大小为 $P\times P$，那么 token 数是：

$$
N=\frac{H}{P}\cdot\frac{W}{P}
$$

注意力矩阵规模近似是 $N^2$。这意味着分辨率翻倍时，token 数先变成 4 倍，attention 的核心计算和激活内存又会进一步放大。

下面这个表最直观：

| 输入分辨率 | patch 大小 | patch 网格 | token 数 | 相对 `224×224` 的 token 倍数 |
| --- | --- | --- | --- | --- |
| `224×224` | `16×16` | `14×14` | 196 | 1x |
| `448×448` | `16×16` | `28×28` | 784 | 4x |
| `352×704` | `16×16` | `22×44` | 968 | 4.94x |

玩具例子：一张 `352×704` 的长图切成 `22×44 patch`。如果你用固定 `14×14` 学出来的绝对位置表，新增出来的大量位置并没有稳定语义，通常只能靠插值“猜”一个向量；但如果你用 2D RoPE，`(3,5)` 到 `(3,6)` 的“向右一步”和 `(10,20)` 到 `(10,21)` 的“向右一步”仍然对应同一类相对偏移模式，注意力更容易保持一致。

真实工程例子：ViT-L 从 `224×224` 升到 `448×448` 时，patch 数从 196 变成 784，光 token 数就是 4 倍。实际显存和延迟通常也接近 4 倍甚至更高，所以高分辨率系统往往不能只“把图放大”，还要配套 tile、稀疏注意力或 token 压缩。

---

## 核心机制与推导

### 1. 分解式 2D RoPE

RoPE 是“用旋转代替相加的位置编码”。旋转的意思是：不往向量里加一个位置向量，而是按位置对 query 和 key 的二维子空间做角度变换。

在 2D 场景里，最常见的做法是把 embedding 维度 $d$ 拆成两半：

- 前 `d/2` 维负责 x 方向
- 后 `d/2` 维负责 y 方向

每两个通道构成一个可旋转的 2 维小块。于是位置 $(x,y)$ 上的编码可写成两个独立旋转块的直和。白话说，就是“横坐标转一半维度，纵坐标再转另一半维度”。

简化图示：

| 维度块 | 编码什么 | 操作 |
| --- | --- | --- |
| `0 ~ d/2-1` | x 轴位置 | 按 `x` 生成 `sin/cos` 并旋转 |
| `d/2 ~ d-1` | y 轴位置 | 按 `y` 生成 `sin/cos` 并旋转 |

于是 attention 的相互作用从原来的

$$
\frac{q^\top k}{\sqrt d}
$$

变成

$$
\frac{(R_{(i,j)}q)^\top(R_{(k,l)}k)}{\sqrt d}
=
\frac{q^\top R_{(i,j)}^\top R_{(k,l)}k}{\sqrt d}
=
\frac{q^\top R_{(k-i,l-j)}k}{\sqrt d}
$$

所以它天然只依赖相对偏移。这里的“相对偏移”就是“横向差几格、纵向差几格”。

更抽象一点说，2D RoPE 的旋转生成元可以看成 `so(4)` 里可交换的 block。对工程实现来说，不需要理解 Lie 代数细节，只要抓住两个事实：

1. x、y 可以分别控制
2. 点积只依赖相对位移

### 2. 2D ALiBi

ALiBi 的思路更直接：不旋转向量，直接改 attention 分数。它在 logit 上加一个“距离越远，惩罚越大”的偏置。

1D 形式是：

$$
a_{ij}=\frac{q_i^\top k_j}{\sqrt d}-m_h\cdot |i-j|
$$

扩展到 2D 后，可以把距离拆成行和列两个部分。一个常见写法是：

$$
a_{(i,j),(k,l)}=
\frac{q_{(i,j)}^\top k_{(k,l)}}{\sqrt d}
-\alpha_h\cdot |i-k|
-\beta_h\cdot |j-l|
$$

如果还要显式区分方向，可以继续分成：

- 向左 / 向上：`r_before`
- 向右 / 向下：`r_after`

这在文档和表格任务里很重要。因为“标题在正文上方”和“脚注在正文下方”不是同一种关系，只用一个无方向斜率会把它们混在一起。

玩具例子：在表格里，单元格 `(5,3)` 对 `(5,4)` 是“同一行向右一格”，对 `(6,3)` 是“同一列向下一格”。2D ALiBi 可以给这两种关系不同偏置，迫使某些头更偏好“横向读表”，另一些头更偏好“纵向聚合”。

---

## 代码实现

下面给出一个可运行的最小实现。它不依赖深度学习框架，只展示 2D RoPE 和 2D ALiBi 的核心计算。

```python
import math
import numpy as np


def token_count(h, w, patch):
    assert h % patch == 0 and w % patch == 0
    return (h // patch) * (w // patch)


def build_2d_coords(h, w):
    coords = []
    for y in range(h):
        for x in range(w):
            coords.append((x, y))
    return np.array(coords, dtype=np.int32)


def rotate_half(x):
    # x shape: (..., dim), dim 必须是偶数
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = -x2
    out[..., 1::2] = x1
    return out


def apply_1d_rope(x, pos, base=10000.0):
    # x: [n, dim], pos: [n]
    n, dim = x.shape
    assert dim % 2 == 0
    half = dim // 2
    freq = 1.0 / (base ** (np.arange(0, half, 1) / half))
    angles = pos[:, None] * freq[None, :]
    cos = np.repeat(np.cos(angles), 2, axis=1)
    sin = np.repeat(np.sin(angles), 2, axis=1)
    return x * cos + rotate_half(x) * sin


def apply_2d_rope(x, coords):
    # x: [n, dim], coords: [n, 2], coords[:,0]=x, coords[:,1]=y
    n, dim = x.shape
    assert dim % 4 == 0
    half = dim // 2
    x_part = apply_1d_rope(x[:, :half], coords[:, 0])
    y_part = apply_1d_rope(x[:, half:], coords[:, 1])
    return np.concatenate([x_part, y_part], axis=-1)


def build_2d_alibi_bias(coords, alpha=1.0, beta=1.0):
    # bias[i, j] = -alpha*|dy| - beta*|dx|
    dx = np.abs(coords[:, None, 0] - coords[None, :, 0])
    dy = np.abs(coords[:, None, 1] - coords[None, :, 1])
    return -(alpha * dy + beta * dx)


# 基本断言
assert token_count(224, 224, 16) == 196
assert token_count(448, 448, 16) == 784

coords = build_2d_coords(2, 3)  # 2 行 3 列，共 6 个位置
x = np.random.randn(6, 8).astype(np.float64)

x_rope = apply_2d_rope(x, coords)
assert x_rope.shape == x.shape

bias = build_2d_alibi_bias(coords, alpha=2.0, beta=1.0)
assert bias.shape == (6, 6)
assert bias[0, 0] == 0.0
assert bias[0, 1] == -1.0   # 同行右移一格
assert bias[0, 3] == -2.0   # 同列下移一格
```

如果换成 PyTorch，接入点通常只有两个：

1. RoPE：在 `q, k` 投影后、attention 前做旋转
2. ALiBi：在 `q @ k^T / sqrt(d)` 之后加 bias matrix

真实工程例子：InternVL-1.5 的动态分辨率策略会先把输入图像按分辨率和纵横比分成多个 `448×448` tile。训练时通常是 `1` 到 `12` 个 tile，测试时可以扩到 `40` 个 tile，且每个 `448×448` tile 经过像素重排后对应 256 个视觉 token。工程上的重点不是“每种分辨率写一套模型”，而是“把所有 tile 变成统一 token 序列，再复用同一套位置编码逻辑”。这就是为什么相对位置方案在 VisionLLM 里更自然。

---

## 工程权衡与常见坑

第一类坑是把 absolute embedding 硬拉伸。很多初学者会把训练好的 `14×14` 位置表，直接 `repeat` 到 `28×28`。这几乎总会出问题，因为复制出来的新位置没有真实几何含义。正确做法至少要做 2D 插值，或者直接改用相对位置方案。

第二类坑是误以为“RoPE 免费”。RoPE 本身参数少，但它并不能解决 token 爆炸。它解决的是“位置泛化”，不是“计算复杂度”。`448×448` 比 `224×224` 多 4 倍 token，这个成本依然在。

第三类坑是 2D ALiBi 只设一组斜率。这样会让“向左一格”和“向下一格”被同样惩罚，方向感会变弱。做文档版面分析、表格理解时，最好区分行列，必要时再区分 before/after。

下面是工程上常见的对比：

| 方案 | 额外参数 | 分辨率外推 | 方向感 | 计算开销 | 常见坑 |
| --- | --- | --- | --- | --- | --- |
| 绝对位置嵌入 | 有 | 弱 | 中 | 低 | 直接拉伸到新网格 |
| 2D RoPE | 无或极少 | 强 | 中到强 | 低 | 维度拆分写错，x/y 对不上 |
| 2D ALiBi | 极少 | 强 | 强，但依赖设计 | 低 | 只用单一斜率导致方向退化 |

新手警告：如果你把 `196` 个绝对位置向量简单复制到更高分辨率，会经常看到 attention map 异常集中在少数固定位置。这不是模型“更自信”，而是位置结构已经退化了。

---

## 替代方案与适用边界

如果任务长期固定输入，比如工业质检永远是 `224×224`，并且目标主要是分类而不是复杂空间推理，那么可学习 2D 绝对位置嵌入仍然是简单有效的基线。

如果任务经常变分辨率、变纵横比，或者要处理大图切片，多数情况下 2D RoPE 更合适。因为它和 LLM 里的 RoPE 思路一致，便于多模态系统统一实现。

如果任务特别依赖“方向性偏好”，比如文档版面分析、表格单元格关系建模、OCR 区块阅读顺序，那么 2D ALiBi 很有吸引力。它直接在 logit 上塑形，调试也更可解释。

可以这样选：

| 场景 | 更适合的方案 | 原因 |
| --- | --- | --- |
| 固定 `224×224` 分类 | 绝对位置嵌入 | 实现最简单，足够用 |
| 多分辨率 ViT 训练 / 推理 | 2D RoPE | 相对偏移一致，外推更稳 |
| 文档、表格、OCR 版面 | 2D ALiBi 或 2D RoPE+偏置 | 方向关系重要 |
| 多 tile 4K VisionLLM | 动态分辨率 + RoPE | 易和长序列架构兼容 |

边界也要讲清楚。RoPE 不是所有视觉任务的唯一答案。对于强局部先验任务，卷积、窗口注意力、层级结构仍然重要。位置编码只解决“坐标怎么表示”，不解决“计算预算怎么压缩”。

---

## 参考资料

- [Rotary Position Embedding for Vision Transformer](https://arxiv.org/abs/2403.13298)：Heo 等人在 2024 年提出将 RoPE 系统化用于 ViT，重点讨论视觉分辨率外推。
- [2D Rotary Position Embedding (RoPE) - Emergent Mind](https://www.emergentmind.com/topics/2d-rotary-position-embedding-rope)：总结 2D RoPE 的轴向分解、代数性质与视觉应用。
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy)：ViT 原始论文，固定 patch 网格和可学习位置嵌入的代表。
- [How To Compute The Token Consumption Of Vision Transformers?](https://ml-digest.com/computing-vision-transformer-tokens/)：给出 `224×224 -> 196 token` 与 `448×448 -> 784 token` 的直观计算。
- [InternVL 1.5: How Far Are We to GPT-4V?](https://internvl.github.io/blog/2024-04-30-InternVL-1.5/)：2024 年 4 月 30 日官方博客，说明动态分辨率、`448×448` tile、训练最多 12 tile、测试最多 40 tile，以及每 tile 256 visual tokens。
- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://openreview.net/forum?id=R8sQPpGCv0)：ALiBi 原始思想来源，核心是在线性距离上给 attention 加 bias。
- [Positional Embeddings in Transformer Models: Evolution from Text to Vision Domains](https://openreview.net/forum?id=Y0z5fIOk7z)：ICLR 2025 Blogpost Track，对 absolute、RoPE、ALiBi 及其视觉扩展做了统一综述。
