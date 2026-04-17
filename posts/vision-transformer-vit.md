## 核心结论

ViT（Vision Transformer）的关键改动，不是重新设计了一套新的 Transformer，而是先把二维图像改写成一维 token 序列，再把标准 Transformer Encoder 直接用于这些 token。这个改写过程就是 patch embedding。

具体流程分成四步：

1. 把输入图像切成固定大小的 patch。
2. 把每个 patch 展平为一个向量。
3. 用同一个可学习线性映射把每个 patch 投影到统一维度的 embedding。
4. 在序列最前面拼接一个可学习的 `[CLS]` token，并叠加位置编码。

如果输入是一张 $224\times224\times3$ 的彩色图像，patch 边长取 $P=16$，那么 patch 数量为：

$$
N=\frac{H}{P}\cdot\frac{W}{P}=\frac{224}{16}\cdot\frac{224}{16}=14\cdot14=196
$$

等价地，也可以写成：

$$
N=\frac{HW}{P^2}=\frac{224\times224}{16^2}=196
$$

其中，$H,W$ 分别表示图像高和宽，$P$ 表示 patch 边长。

每个 patch 的原始长度是：

$$
P^2C=16\times16\times3=768
$$

如果 embedding 维度取 $D=768$，那么每个 patch 会被映射成一个 $768$ 维 token。再加上一个 `[CLS]` token 后，序列长度变成：

$$
N+1=197
$$

因此位置编码参数的形状是：

$$
(N+1)\times D = 197\times768
$$

这个数值例子非常重要，因为它把 ViT 的输入结构说清楚了：模型不再直接处理像素网格，而是在处理长度为 197 的 token 序列，其中 196 个 token 来自图像 patch，1 个 token 用于汇总全局信息。

一个最小玩具例子更容易理解。假设输入是一张 $4\times4$ 的单通道图像，取 patch 边长 $P=2$，那么它会被切成 4 个 patch。原来“按二维网格排列的像素”，会变成“按一维顺序排列的 4 个图像词”。这就是 ViT 的本质：先离散成 patch，再像处理句子一样处理图像。

---

## 问题定义与边界

问题本身很直接：Transformer 原生处理的是序列，而图像原生是二维网格。要让 Transformer 处理图像，第一步不是修改注意力公式，而是定义“什么东西算一个 token”。ViT 的答案是：把局部图像块作为 token，也就是 patch。

这里的 “embedding” 可以先做一个工程化定义：它是把原始输入映射到固定维度向量空间的过程，使后续网络能够用统一的张量形状处理所有输入单元。对 ViT 来说，embedding 的对象不是整张图，也不是单个像素，而是一个 patch。

ViT 的输入边界主要由三个量决定：

| 量 | 含义 | 直接影响 |
|---|---|---|
| $P$ | patch 边长 | 决定一个 token 覆盖多大局部区域 |
| $N=\frac{HW}{P^2}$ | patch 数，也就是 token 数 | 决定注意力矩阵规模与序列长度 |
| $D$ | embedding 维度 | 决定每个 token 的表示容量和参数量 |

这三个量之间存在直接耦合关系。对固定分辨率的图像来说，$P$ 越小，$N$ 越大；而 $N$ 越大，自注意力的计算和显存成本越高。标准全局自注意力的主要开销近似随 token 数平方增长，即：

$$
\text{Attention Cost} \approx O(N^2D)
$$

更准确一点地说，若忽略 batch 和多头拆分细节，注意力矩阵本身的形状是 $N\times N$，因此序列长度增加会带来平方级扩张。

为什么 patch 大小是 ViT 的核心边界？因为它同时影响“看得多细”和“算得多贵”。

以 $1024\times1024$ 的图像为例：

| patch 边长 $P$ | token 数 $N=(1024/P)^2$ | 注意力矩阵规模 |
|---|---:|---:|
| 32 | 1024 | $1024^2 \approx 1.05\times10^6$ |
| 16 | 4096 | $4096^2 \approx 1.68\times10^7$ |
| 8 | 16384 | $16384^2 \approx 2.68\times10^8$ |

这个增长不是“小一点 patch，贵一点算力”，而是序列长度翻倍后，注意力矩阵面积接近变成四倍。对于高分辨率图像，这会很快变成显存瓶颈。

ViT 与 ResNet 的根本差异，也不在于“一个是新模型，一个是旧模型”，而在于归纳偏置不同。归纳偏置指的是：模型在训练前，就更容易学习到哪些结构。

卷积网络天然具备两个偏置：

1. 局部连接：卷积核只看局部邻域。
2. 权重共享：同一个卷积核在整张图上滑动。

这使 CNN 更容易学习边缘、纹理、局部模式和平移相关结构。而 ViT 从 patch 开始建模，没有卷积那种显式局部先验，因此更依赖数据规模、预训练和训练策略。

| 维度 | ResNet | ViT |
|---|---|---|
| 局部感受野 | 天然存在 | 不天然存在，需要数据学习 |
| 平移等变/不变性 | 较强 | 较弱 |
| 全局关系建模 | 需靠深层堆叠或额外模块 | 自注意力天然支持 |
| 小数据集稳定性 | 通常更稳 | 更依赖预训练 |
| 大规模预训练扩展性 | 强 | 往往更强 |

因此，ViT 不是“永远优于 CNN”，而是“在足够数据和合适训练条件下，能把 Transformer 的全局建模能力释放出来”。原始 ViT 论文中，优势通常出现在大规模预训练场景，例如 JFT-300M；在较小数据集上，往往需要 DeiT 这类训练补丁来补足。

对新手来说，可以先把问题边界记成一句话：

- ViT 要先解决“怎么把图像变成 token 序列”。
- patch embedding 就是这个序列化接口。
- patch 选得太大，会丢局部细节。
- patch 选得太小，会让注意力成本爆炸。

---

## 核心机制与推导

ViT 的 patch embedding 可以拆成三步：切块、展平、投影；随后再补上 `[CLS]` token 和位置编码。

先设输入图像为：

$$
x\in\mathbb{R}^{H\times W\times C}
$$

其中 $H$ 是高度，$W$ 是宽度，$C$ 是通道数。把图像切成不重叠 patch，patch 边长为 $P$。若 $H,W$ 都能被 $P$ 整除，则 patch 数量为：

$$
N=\frac{HW}{P^2}
$$

第 $i$ 个 patch 的原始形状是：

$$
P\times P\times C
$$

展平后变为：

$$
x_p^i\in\mathbb{R}^{P^2C}, \quad i=1,\dots,N
$$

这一步只是重排，不涉及参数。它的作用是把二维局部块改写成一维向量，方便接上线性层。

然后引入可学习投影矩阵：

$$
E\in\mathbb{R}^{P^2C\times D}
$$

每个 patch token 的 embedding 是：

$$
e^i=x_p^iE \in \mathbb{R}^{D}
$$

把所有 patch 拼起来，可以写成矩阵形式：

$$
X_p\in\mathbb{R}^{N\times(P^2C)}
$$

$$
X_e=X_pE\in\mathbb{R}^{N\times D}
$$

这一步可以理解成：用同一个共享权重矩阵，把所有 patch 压到同一个语义空间。共享权重很重要，因为模型需要对任意位置的 patch 使用同一套“读取规则”，否则不同位置会拥有不同的输入投影标准。

接下来在序列最前面加入一个可学习的分类 token：

$$
x_{\text{class}}\in\mathbb{R}^{D}
$$

然后拼接得到：

$$
[x_{\text{class}};X_e]\in\mathbb{R}^{(N+1)\times D}
$$

最后叠加位置编码：

$$
E_{\text{pos}}\in\mathbb{R}^{(N+1)\times D}
$$

ViT 的初始输入序列是：

$$
z_0=[x_{\text{class}}; x_p^1E;\dots;x_p^NE]+E_{\text{pos}}
$$

这一式子里，五个对象各自负责不同事情：

| 对象 | 作用 |
|---|---|
| patch 切分 | 定义 token 的基本单位 |
| flatten | 把局部块变成向量 |
| 线性投影 $E$ | 把不同 patch 统一映射到 $D$ 维空间 |
| `[CLS]` token | 提供一个聚合全局信息的专用位置 |
| 位置编码 $E_{\text{pos}}$ | 告诉模型每个 patch 原来在图像中的空间位置 |

为什么位置编码必须存在？因为自注意力本身对输入顺序不敏感。若只给模型一组 patch embedding，而不告诉它它们原来的空间位置，那么左上角 patch 和右下角 patch 的语义顺序在输入层是不可区分的。位置编码的作用，就是把空间顺序重新注入到 token 序列里。

以标准配置 $224\times224\times3,\ P=16,\ D=768$ 为例，整个形状推导如下：

| 步骤 | 张量形状 | 含义 |
|---|---|---|
| 输入图像 | $224\times224\times3$ | 原始 RGB 图像 |
| 切 patch | $14\times14$ 个 patch | 共 196 个 patch |
| 单个 patch 展平 | $16\times16\times3=768$ | 每个 patch 变成 768 维 |
| 线性投影后 | $196\times768$ | 196 个 token，每个 768 维 |
| 拼接 `[CLS]` | $197\times768$ | 序列长度加 1 |
| 加位置编码 | $197\times768$ | 送入 Transformer Encoder |

可以把流程压缩成一条工程记忆链：

```text
patch -> flatten -> linear projection -> concat [CLS] -> add position embedding -> Transformer Encoder
```

下面用一个最小玩具例子把“切块和展平”看清楚。输入是一张 $4\times4$ 的单通道图像：

| 1 | 2 | 3 | 4 |
|---|---|---|---|
| 5 | 6 | 7 | 8 |
| 9 | 10 | 11 | 12 |
| 13 | 14 | 15 | 16 |

取 $P=2$，则图像会被切成 4 个 patch：

| patch 编号 | 对应区域 | 展平结果 |
|---|---|---|
| patch 1 | 左上角 $2\times2$ | `[1, 2, 5, 6]` |
| patch 2 | 右上角 $2\times2$ | `[3, 4, 7, 8]` |
| patch 3 | 左下角 $2\times2$ | `[9, 10, 13, 14]` |
| patch 4 | 右下角 $2\times2$ | `[11, 12, 15, 16]` |

这一步之后，模型输入不再是二维像素网格，而是长度为 4 的 patch 序列。也就是说，ViT 的“词表单位”不是像素，而是局部图像块。

如果继续引入一个假设投影矩阵 $E\in\mathbb{R}^{4\times3}$，那么每个长度为 4 的 patch 向量会被映射成长度为 3 的 token。这样，4 个 patch 最终得到形状为 $4\times3$ 的 token 矩阵；加上 `[CLS]` 后，就是 $5\times3$。这个最小例子和真实 ViT 完全同构，只是尺寸更小。

---

## 代码实现

工程上，patch embedding 常见有两种写法：

1. 用 `nn.Linear(P*P*C, D)`：先手动切 patch、展平，再做线性投影。
2. 用 `nn.Conv2d(C, D, kernel_size=P, stride=P)`：一步完成“切块 + 线性投影”。

这两种写法在数学上是等价思路。第二种更常见，因为它更高效，张量布局也更适合深度学习框架。

先给一个不依赖深度学习框架的 NumPy 版本。它能完整展示 patchify、线性投影、`[CLS]` 拼接和位置编码叠加，并且代码可以直接运行。

```python
import numpy as np

def patchify(image, patch_size):
    """
    image: (H, W, C)
    return: (N, patch_size * patch_size * C)
    """
    H, W, C = image.shape
    assert H % patch_size == 0 and W % patch_size == 0, "H and W must be divisible by patch_size"

    h_blocks = H // patch_size
    w_blocks = W // patch_size

    patches = (
        image.reshape(h_blocks, patch_size, w_blocks, patch_size, C)
             .transpose(0, 2, 1, 3, 4)
             .reshape(h_blocks * w_blocks, patch_size * patch_size * C)
    )
    return patches

def patch_embed_numpy(image, patch_size, embed_dim, seed=0):
    """
    image: (H, W, C)
    return:
      tokens: (N+1, D)
      patches: (N, P*P*C)
    """
    rng = np.random.default_rng(seed)

    patches = patchify(image, patch_size)                # (N, P*P*C)
    patch_dim = patches.shape[1]
    num_patches = patches.shape[0]

    # 共享线性投影矩阵
    E = rng.standard_normal((patch_dim, embed_dim))
    patch_tokens = patches @ E                           # (N, D)

    # [CLS] token 和位置编码
    cls_token = rng.standard_normal((1, embed_dim))      # (1, D)
    pos_embed = rng.standard_normal((num_patches + 1, embed_dim))

    tokens = np.vstack([cls_token, patch_tokens])        # (N+1, D)
    tokens = tokens + pos_embed                          # (N+1, D)
    return tokens, patches

def main():
    # 4x4 单通道图像
    img = np.arange(1, 17, dtype=np.float32).reshape(4, 4, 1)

    patches = patchify(img, patch_size=2)
    assert patches.shape == (4, 4)
    assert patches[0].tolist() == [1.0, 2.0, 5.0, 6.0]
    assert patches[1].tolist() == [3.0, 4.0, 7.0, 8.0]
    assert patches[2].tolist() == [9.0, 10.0, 13.0, 14.0]
    assert patches[3].tolist() == [11.0, 12.0, 15.0, 16.0]

    tokens, raw_patches = patch_embed_numpy(img, patch_size=2, embed_dim=3, seed=42)
    assert raw_patches.shape == (4, 4)
    assert tokens.shape == (5, 3)   # 4 个 patch token + 1 个 [CLS]

    print("raw patches:")
    print(raw_patches)
    print("\ntokens shape:", tokens.shape)
    print(tokens)

if __name__ == "__main__":
    main()
```

这个版本有三个用途：

1. 看清 patch 是怎么从图像中重排出来的。
2. 看清线性投影如何把 `(N, P^2C)` 变成 `(N, D)`。
3. 看清 `[CLS]` 和位置编码叠加后，序列长度为什么是 `N+1`。

下面给出更接近真实训练代码的 PyTorch 实现。这个版本也可直接运行。

```python
import torch
import torch.nn as nn

class PatchEmbedLinear(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_chans

        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

    def patchify(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        P = self.patch_size
        assert H == self.img_size and W == self.img_size
        assert C == self.in_chans

        x = x.reshape(B, C, H // P, P, W // P, P)        # [B, C, H/P, P, W/P, P]
        x = x.permute(0, 2, 4, 3, 5, 1)                  # [B, H/P, W/P, P, P, C]
        x = x.reshape(B, self.num_patches, self.patch_dim)
        return x

    def forward(self, x):
        x = self.patchify(x)                             # [B, N, P*P*C]
        x = self.proj(x)                                 # [B, N, D]

        cls = self.cls_token.expand(x.size(0), -1, -1)   # [B, 1, D]
        x = torch.cat([cls, x], dim=1)                   # [B, N+1, D]
        x = x + self.pos_embed                           # [B, N+1, D]
        return x


class PatchEmbedConv(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Conv2d 的每个卷积核覆盖一个 patch，并且 stride=P，因此不会重叠
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        x = self.proj(x)                                 # [B, D, H/P, W/P]
        x = x.flatten(2)                                 # [B, D, N]
        x = x.transpose(1, 2)                            # [B, N, D]

        cls = self.cls_token.expand(B, -1, -1)           # [B, 1, D]
        x = torch.cat([cls, x], dim=1)                   # [B, N+1, D]
        x = x + self.pos_embed                           # [B, N+1, D]
        return x


def main():
    dummy = torch.randn(2, 3, 224, 224)

    m1 = PatchEmbedLinear(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
    out1 = m1(dummy)
    assert out1.shape == (2, 197, 768)

    m2 = PatchEmbedConv(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
    out2 = m2(dummy)
    assert out2.shape == (2, 197, 768)

    print("Linear version:", out1.shape)
    print("Conv version:  ", out2.shape)

if __name__ == "__main__":
    main()
```

为什么 `Conv2d` 版本与 `Linear` 版本等价？因为当 `kernel_size=P, stride=P` 时，每个卷积核恰好覆盖一个不重叠 patch；每个输出通道都对应一组长度为 $P^2C$ 的权重。对单个 patch 来说，这其实就是一次线性映射。

可以把两者对照成下面这张表：

| 写法 | 输入视角 | 参数形状 | 输出结果 |
|---|---|---|---|
| `nn.Linear(P^2C, D)` | 先手动拿到每个 patch | $(P^2C)\times D$ | 每个 patch 变成一个 $D$ 维 token |
| `nn.Conv2d(C, D, kernel=P, stride=P)` | 直接在图像上滑动不重叠窗口 | $D\times C\times P\times P$ | 每个 patch 位置得到一个 $D$ 维输出 |

真实工程里还有一个经常被忽略的问题：输入尺寸变化后，位置编码长度会不匹配。

例如，预训练使用 $224\times224$ 且 $P=16$，则 patch 数为 $14\times14=196$；微调改成 $384\times384$，则 patch 数变为：

$$
\left(\frac{384}{16}\right)^2 = 24^2 = 576
$$

这时原来的位置编码是 `(1, 197, D)`，新的输入却需要 `(1, 577, D)`。处理方法通常不是随便截断或补零，而是：

1. 单独保留 `[CLS]` 位置编码。
2. 把剩余 patch 位置编码从原来的二维网格形状恢复出来。
3. 用双线性插值或双三次插值调整到新的网格大小。
4. 再展平成新的 patch 序列并拼回 `[CLS]`。

初学者只要记住一个原则就够了：位置编码跟 patch 网格绑定，不是跟“图像文件”绑定。

---

## 工程权衡与常见坑

ViT 的第一类工程问题，是数据不够时训练不稳定。原因不是 Transformer 不能做视觉任务，而是它缺少 CNN 那种强局部归纳偏置。卷积模型天生更容易抓住边缘、纹理和局部重复结构；ViT 要从 patch token 和注意力里自己学这些模式，因此更依赖预训练和数据规模。

这在小样本任务里尤其明显。比如工业缺陷检测、医学影像、小规模遥感分类，数据往往不大，而且判别信息可能集中在非常局部的区域。如果 patch 太大，小缺陷会被淹没到大块背景里。

以工业缺陷检测为例。假设一张图像尺寸是 $512\times512$，而缺陷面积只有约 $8\times8$。若 patch 边长是 32，那么一个 patch 的覆盖面积为：

$$
32\times32=1024
$$

而缺陷面积为：

$$
8\times8=64
$$

缺陷只占这个 patch 的：

$$
\frac{64}{1024}=6.25\%
$$

这意味着，patch token 的大部分输入来自背景，异常信号会被明显稀释。此时即便全局建模能力很强，输入粒度本身也已经不够细。

第二类问题是位置编码长度 mismatch。因为位置编码形状固定为 $(N+1)\times D$，只要输入分辨率变了、patch 大小变了，或者裁剪策略变了，$N$ 就会变化。常见处理方式如下：

| 问题 | 原因 | 常见对策 |
|---|---|---|
| 小数据集效果差 | 缺少局部归纳偏置 | 大规模预训练、强数据增强、蒸馏 |
| 位置编码不匹配 | patch 数变化 | 对 patch 位置编码做二维插值 |
| patch 太大漏细节 | 局部信息被粗化 | 减小 patch，或改用多尺度结构 |
| patch 太小算力爆炸 | $N$ 增大，自注意力近似 $O(N^2)$ | 增大 patch、窗口注意力、层次化设计 |
| `[CLS]` 表现不稳 | 全局分类压在单 token 上 | 试验 mean pooling 或改进训练策略 |

第三类问题是形状处理错误。这是新手最常见、也最容易浪费时间的问题。典型错误包括：

1. 把图像张量写成 `[B, H, W, C]`，但 PyTorch 默认要求 `[B, C, H, W]`。
2. `flatten` 的轴不对，导致 token 维度和 embedding 维度混淆。
3. `[CLS]` token 拼接在序列末尾，而不是开头。
4. 位置编码长度写成 `N`，漏掉 `[CLS]` 的那一位。
5. `img_size` 不能被 `patch_size` 整除，却没有提前断言。

第四类问题是只记结论，不理解训练补丁为什么有效。DeiT 是一个典型例子。它并没有推翻 ViT 的主体结构，而是改进训练方式，让 ViT 在中等规模数据集上也更实用。

DeiT 的关键做法有两点：

1. 使用更强的数据增强、正则化和训练配方。
2. 引入 distillation token，让模型在真实标签之外，还模仿教师网络的输出分布。

这里的 distillation token 可以理解成：除了 `[CLS]` 负责对真实任务标签建模之外，模型又多了一个专门接收教师知识的 token。教师通常是性能稳定的 CNN，例如 RegNet。这样做的意义，不是把 ViT 变成 CNN，而是借助 CNN 的归纳偏置，降低 ViT 从零学局部结构的难度。

如果把工程经验压缩成一个判断表，可以记成下面这样：

| 任务现象 | 可能原因 | 优先检查 |
|---|---|---|
| 训练精度升得慢 | patch 太粗或训练配方太弱 | patch size、学习率、增强策略 |
| 微调时报 shape error | 位置编码长度不匹配 | 输入分辨率、patch 数、位置编码插值 |
| 小目标识别差 | patch 覆盖范围过大 | 是否需要更小 patch 或多尺度模型 |
| 显存爆掉 | token 数过多 | 分辨率、patch size、是否需要窗口注意力 |
| 分类头不稳定 | `[CLS]` 聚合不足 | 尝试 mean pooling、蒸馏或更长训练 |

对初学者最实用的建议不是背论文，而是先固定检查这四个形状：

```text
输入图像:      [B, C, H, W]
Conv输出:      [B, D, H/P, W/P]
展平转置后:    [B, N, D]
加CLS后:       [B, N+1, D]
```

只要这四步形状是对的，patch embedding 基本就不会出大错。

---

## 替代方案与适用边界

如果任务数据量不大，或者局部结构特别重要，直接使用纯 ViT 往往不是最稳的选择。更常见的替代路线有四类：DeiT、Swin Transformer、Hybrid CNN-Transformer、传统 CNN。

| 模型 | 数据需求 | 局部建模 | 全局建模 | 适用场景 |
|---|---|---|---|---|
| ViT | 高 | 弱，主要靠学习 | 强 | 大规模预训练、分类、基础视觉表征 |
| DeiT | 中 | 通过蒸馏间接增强 | 强 | ImageNet 级资源，希望保留 ViT 主体 |
| Swin Transformer | 中 | 强，窗口机制更友好 | 中到强 | 检测、分割、高分辨率视觉任务 |
| Hybrid CNN-Transformer | 低到中 | 强 | 中到强 | 小样本、纹理敏感任务 |
| 传统 CNN | 低到中 | 很强 | 中 | 数据较少、局部模式主导的任务 |

这些路线的区别，不只是“谁精度更高”，而是“谁更匹配任务的结构约束”。

### 1. DeiT：保留 ViT 主体，但补训练方法

DeiT 适合这样一类场景：你仍然想使用 ViT 结构，但又没有 JFT 这类超大规模数据。它的核心贡献在于训练配方和蒸馏，而不是重新设计 patch embedding。本质上，它是在告诉你：纯 ViT 不是不能用小数据，而是要靠更强训练策略来弥补先验不足。

### 2. Swin Transformer：控制 token 成本，引入层次结构

Swin Transformer 的关键点，是把全局注意力改成局部窗口注意力，并通过移位窗口让不同区域发生交互。它更适合高分辨率视觉任务，因为它不需要在一开始就对所有 patch 做全局两两注意力。对于检测、分割、遥感等任务，这种层次化结构通常比纯 ViT 更实际。

### 3. Hybrid CNN-Transformer：让 CNN 看局部，让 Transformer 汇总全局

Hybrid 路线很适合小样本场景。做法通常是先用 CNN 提取局部特征图，再把特征图切成 token 送入 Transformer。这样可以把两个模型的优势拆开使用：

- CNN 负责局部纹理、边缘和细节。
- Transformer 负责长程依赖和全局关系。

这种分工对缺陷检测、医学图像、细粒度分类都比较常见。

### 4. 传统 CNN：不是落后，而是任务匹配

如果数据量不大、任务非常依赖局部纹理、硬件资源有限，那么传统 CNN 仍然是很强的基线。很多工程失败，不是因为 CNN 不够先进，而是因为问题本身并不需要全局 token 交互。

可以把几条路线用更直白的话记成下面这样：

| 任务条件 | 更稳的起点 |
|---|---|
| 数据大、预训练强、重视全局语义 | ViT |
| 数据中等、希望使用 ViT 但训练更稳 | DeiT |
| 分辨率高、检测或分割任务 | Swin Transformer |
| 数据少、局部纹理关键 | Hybrid CNN-Transformer 或 CNN |

玩具级理解可以记成两句话：

- CNN 更像“先看邻居，再逐层扩大视野”。
- ViT 更像“先切成块，再让所有块互相看”。

真实工程里，可以用下面几个例子判断边界：

1. 对只有几千张图的工业缺陷库，局部毛刺、裂纹和孔洞往往比全局语义更重要，Hybrid 或 CNN 常常更稳。
2. 对高分辨率遥感图，纯 ViT 的全局注意力成本太高，Swin 这类窗口化、分层化结构通常更现实。
3. 对大规模分类预训练、跨任务迁移、图文联合建模这类场景，纯 ViT 依然是非常有竞争力的底座。

所以适用边界可以压缩成一句话：ViT 更适合“数据大、预训练强、全局关系重要”的问题；当“数据少、局部纹理关键、分辨率超高”时，DeiT、Swin 或 CNN-Transformer 混合结构通常更稳。

---

## 参考资料

1. Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.  
   用途：ViT 原始论文，patch embedding、`[CLS]` token、位置编码、预训练规模结论的核心出处。

2. Touvron et al., *Training data-efficient image transformers & distillation through attention*, ICML 2021.  
   用途：DeiT 论文，解释 distillation token、教师网络蒸馏和中等数据规模下的训练改进。

3. Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*, ICCV 2021.  
   用途：说明为什么高分辨率任务常采用窗口注意力和层次化结构，而不是直接扩大纯 ViT 的全局注意力。

4. PyTorch 官方文档：`torch.nn.Conv2d` 与 `torch.nn.Linear`。  
   用途：理解为什么 `Conv2d(kernel_size=P, stride=P)` 可以等价实现 patch embedding。

5. Google Research / ViT 官方代码实现。  
   用途：核对 patch embedding、位置编码参数形状、输入尺寸变化时的实现细节。

6. Ross Wightman 的 `timm` 中 ViT 实现。  
   用途：查看工业常用写法，包括 `PatchEmbed`、位置编码插值和不同 ViT 变体的工程实现。

7. 任何可靠的张量形状推导笔记或教程。  
   用途：重点不是“背概念”，而是核对 `[B, C, H, W] -> [B, N, D] -> [B, N+1, D]` 这条形状链是否正确。
