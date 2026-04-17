## 核心结论

ViT 的图像分块嵌入，本质上是把二维图像改写成一串一维 token。这里的 token 指 Transformer 处理的最小输入单元；对 ViT 来说，一个 token 通常对应图像中的一个 patch。没有分块、线性投影和位置编码，Transformer 看到的只是一组数值向量，无法判断哪些像素原本属于同一块，也无法恢复这些块在图像中的空间顺序。

具体过程可以压缩为三步：

1. 把输入图像按固定大小 $P \times P$ 切成若干 patch。
2. 把每个 patch 展平成长度为 $P^2C$ 的向量，再通过线性投影映射到统一维度 $D$。
3. 给每个 patch token 加上位置编码，再和 `[CLS]` 等特殊 token 拼接，送入标准 Transformer。

它为什么成立，关键不在“切块”这个动作本身，而在“统一表示空间”。线性投影把原始像素块转换成模型内部可比较、可加权的向量；位置编码把“这个块位于哪里”补回去。这样，自注意力层才能同时利用内容信息和位置信息，在所有 patch 之间建立全局关系。

对新手来说，最容易混淆的点是：分块后送进 Transformer 的不再是像素网格，而是 token 序列。也就是说，ViT 后续处理的对象不是“第 37 行第 52 列像素”，而是“第 12 个 patch token、第 87 个 patch token”。二维图像先被改写成序列接口，再交给 Transformer 主干。

一个最小玩具例子是：把一张 $4 \times 4$ 的灰度图按 $2 \times 2$ 切块，会得到 4 个 patch；每个 patch 展平后是 4 维向量，再映射到例如 8 维 embedding。模型真正接收的输入，是长度为 4 的 token 序列，而不是原始的 $4 \times 4$ 像素表。

一个真实工程例子是：在多模态模型里，图像经过 `PatchEmbed` 后得到一串视觉 token，文本经过 tokenizer 和词嵌入后得到一串文本 token。两者只要都被投影到统一的向量空间，就可以拼成一个联合序列，让同一个 Transformer 同时处理“看图”和“读字”。

| 阶段 | 形状变化 | 含义 |
|---|---|---|
| 输入图像 | $H \times W \times C$ | 原始像素张量 |
| 分块 | $N \times P \times P \times C$ | 切成 $N$ 个 patch |
| 展平 | $N \times (P^2C)$ | 每个 patch 变成一维向量 |
| 线性投影 | $N \times D$ | 映射到统一 token 维度 |
| 加位置编码 | $N \times D$ | 注入空间位置信息 |
| 拼接 `[CLS]` | $(N+1) \times D$ | 形成 Transformer 输入序列 |

---

## 问题定义与边界

问题定义很直接：给定一张大小为 $H \times W \times C$ 的图像，希望构造出一组长度固定的 token，供 Transformer 作为输入。这里“固定”指的是每个 token 的向量维度固定为 $D$，而不是 token 数量固定。token 数量由图像分辨率和 patch 大小共同决定。

如果 patch 大小为 $P \times P$，且图像尺寸能被 $P$ 整除，那么 patch 数量为：

$$
N = \frac{HW}{P^2} = \frac{H}{P}\cdot\frac{W}{P}
$$

每个 patch 的原始形状是 $P \times P \times C$，展平后长度为：

$$
P \times P \times C \Rightarrow P^2C
$$

例如常见配置中，输入图像为 $224 \times 224 \times 3$，取 $P=16$，则：

$$
N = \left(\frac{224}{16}\right)^2 = 14^2 = 196
$$

每个 patch 展平后的长度为：

$$
16 \times 16 \times 3 = 768
$$

如果再投影到 $D=768$，数值维度看起来没有变化，但语义已经变了。投影前的 768 维只是像素按顺序拼接；投影后的 768 维是模型可学习的特征坐标，后续注意力和前馈层都在这个特征空间中工作。

为了避免抽象，可以把这个过程理解成一张“接口转换表”：

| 变量 | 数学含义 | 直观解释 |
|---|---|---|
| $H, W$ | 图像高和宽 | 图像有多少行、多少列 |
| $C$ | 通道数 | 灰度图一般是 1，RGB 图一般是 3 |
| $P$ | patch 边长 | 每个小块切多大 |
| $N$ | patch 数量 | 最终会产生多少个视觉 token |
| $D$ | embedding 维度 | 每个 token 进入 Transformer 时的特征长度 |

边界主要有四类。

第一，patch 大小决定信息颗粒度。颗粒度指模型感知细节的精细程度。$P$ 越小，token 越多，细节保留越充分；但标准自注意力复杂度接近 $O(N^2)$，token 数一旦增加，计算和显存都会迅速上升。$P$ 越大，序列更短、训练更省，但边缘、小目标、细纹理更容易在 patch 内被平均掉。

第二，位置编码必须与 token 序列严格对齐。Transformer 本身不理解“左上角”“右下角”这些二维位置概念，所以必须显式提供位置信号。如果训练时的 patch 网格是 $14 \times 14$，推理时改成别的分辨率，位置编码往往就会出现长度不匹配，需要做二维插值或重新初始化。

第三，固定大小、非重叠切块会破坏局部连续性。卷积天然保留局部邻域，而 patch 切块会在边界处人为断开。对于分类任务，这种断裂通常可以靠后续层补回来；但对 OCR、小物体检测、医学影像这类细粒度任务，块边界可能直接损伤输入表示。

第四，分块嵌入只回答“怎么把图像改写成 Transformer 可接收的序列”，并不自动保证“模型一定能学到好的视觉模式”。ViT 缺少 CNN 那种强局部归纳偏置。归纳偏置可以理解为模型在训练开始前就更偏向于学习某类结构假设，例如局部邻域、平移等价性和层级特征。

因此，ViT 图像分块嵌入的适用边界不是“图像任务都可以直接切块处理”，而是“当全局关系建模收益足够大，并且训练数据、分辨率策略、位置编码处理都设计合理时，它是一个统一且高效的视觉入口”。

---

## 核心机制与推导

设输入图像为：

$$
I \in \mathbb{R}^{H \times W \times C}
$$

把它划分为 $N$ 个不重叠 patch。第 $i$ 个 patch 记为 $\text{patch}_i$，则：

$$
\text{patch}_i \in \mathbb{R}^{P \times P \times C}
$$

先对每个 patch 做展平：

$$
\text{flatten}(\text{patch}_i) \in \mathbb{R}^{P^2C}
$$

再乘上可学习的投影矩阵：

$$
E \in \mathbb{R}^{(P^2C)\times D}
$$

得到 patch embedding：

$$
x_i = \text{flatten}(\text{patch}_i)\,E
$$

如果把所有 patch 一次性写成矩阵形式，更容易看清整体形状变化。设

$$
X_p \in \mathbb{R}^{N \times (P^2C)}
$$

表示所有展平后的 patch 组成的矩阵，则：

$$
X_e = X_p E,\quad X_e \in \mathbb{R}^{N \times D}
$$

这里的线性投影不是“把维度机械压缩一下”，而是在学习一个映射：哪些像素组合应该在 embedding 空间里更靠近，哪些组合应该更容易被后续注意力区分。也正因为这个投影是可学习的，所以 `PatchEmbed` 不是纯预处理，而是模型参数的一部分。

之后加入位置编码 $pos_i$：

$$
z_i = x_i + pos_i,\quad pos_i \in \mathbb{R}^{D}
$$

再把分类 token `[CLS]` 拼在前面，形成最终输入序列：

$$
X = [\text{CLS}; z_1; z_2; \dots; z_N]
$$

也可以展开写成：

$$
X = [\text{CLS}; x_1 + pos_1; x_2 + pos_2; \dots; x_N + pos_N]
$$

如果按批次写，则常见的输入张量形状为：

$$
X \in \mathbb{R}^{B \times (N+1) \times D}
$$

其中 $B$ 是 batch size。

为什么必须加位置编码？因为自注意力的基本计算是内容相似度。设两个 token 为 $a,b$，注意力分数核心项是：

$$
\text{score}(a,b) = \frac{Q(a)K(b)^\top}{\sqrt{d_k}}
$$

这个计算天然只看 token 的表示内容，不知道它们原本位于图像的哪个位置。如果两个 patch 内容相似，但一个来自左上角、一个来自右下角，纯内容向量会让模型倾向于把它们当成相似项。位置编码提供的是“内容之外的坐标信号”。

一个 $4 \times 4$ 灰度图的玩具例子可以把这个过程看得更具体。设图像为：

$$
\begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16
\end{bmatrix}
$$

按 $2 \times 2$ 切块后，得到 4 个 patch：

| patch 位置 | 像素块 | 展平结果 |
|---|---|---|
| 左上 | $\begin{bmatrix}1 & 2 \\ 5 & 6\end{bmatrix}$ | $[1,2,5,6]$ |
| 右上 | $\begin{bmatrix}3 & 4 \\ 7 & 8\end{bmatrix}$ | $[3,4,7,8]$ |
| 左下 | $\begin{bmatrix}9 & 10 \\ 13 & 14\end{bmatrix}$ | $[9,10,13,14]$ |
| 右下 | $\begin{bmatrix}11 & 12 \\ 15 & 16\end{bmatrix}$ | $[11,12,15,16]$ |

如果不加位置编码，模型只知道序列里有 4 个向量，不知道哪个来自左上、哪个来自右下。位置编码的作用，就是让“内容相同但位置不同”的 patch 在表示上仍然可区分。

真实工程里，`切块 + 展平 + Linear` 通常会等价实现为一个卷积层：

$$
\text{Conv2d}(\text{kernel}=P,\ \text{stride}=P,\ \text{out\_channels}=D)
$$

为什么二者等价？因为当卷积核大小和步长都等于 patch 大小时，每次卷积恰好覆盖一个不重叠 patch；每个输出通道本质上就是对 patch 内所有像素做一次加权求和，这与线性投影的形式一致。区别只在工程实现更高效，底层库已经为卷积路径做了优化。

多模态模型中的意义可以再明确一步。视觉侧输出的是视觉 token 序列，文本侧输出的是词 token 序列。它们之所以能拼接到同一个 Transformer 主干，不是因为图像“变成了文字”，而是因为两者都被改写成同一种接口：

$$
\text{sequence length} \times \text{embedding dimension}
$$

统一序列接口，才是多模态模型能够共享注意力机制的根本原因。

---

## 代码实现

下面先给一个最小可运行的 Python 版本，只依赖 `numpy`。这段代码完整展示四个动作：切块、展平、线性投影、加入位置编码。所有 `assert` 都是为了把前面的公式和代码形状一一对上。

```python
import numpy as np

def patchify(image, patch_size):
    """
    image: (H, W, C)
    return:
        patches: (N, P, P, C)
        patches_flat: (N, P*P*C)
    """
    H, W, C = image.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, "image size must be divisible by patch size"

    n_h = H // P
    n_w = W // P

    patches = (
        image.reshape(n_h, P, n_w, P, C)
             .transpose(0, 2, 1, 3, 4)
             .reshape(n_h * n_w, P, P, C)
    )
    patches_flat = patches.reshape(n_h * n_w, P * P * C)
    return patches, patches_flat

def patch_embed(image, patch_size, embed_dim, add_cls=True, seed=0):
    """
    image: (H, W, C)
    return:
        patches_flat: (N, P*P*C)
        patch_tokens: (N, D)
        tokens_with_pos: (N or N+1, D)
    """
    H, W, C = image.shape
    _, patches_flat = patchify(image, patch_size)

    rng = np.random.default_rng(seed)
    proj = rng.normal(loc=0.0, scale=0.02, size=(patch_size * patch_size * C, embed_dim))
    patch_tokens = patches_flat @ proj  # (N, D)

    num_patches = patch_tokens.shape[0]
    pos_embed = rng.normal(loc=0.0, scale=0.02, size=(num_patches, embed_dim))
    patch_tokens_with_pos = patch_tokens + pos_embed

    if add_cls:
        cls_token = rng.normal(loc=0.0, scale=0.02, size=(1, embed_dim))
        cls_pos = rng.normal(loc=0.0, scale=0.02, size=(1, embed_dim))
        tokens = np.concatenate([cls_token + cls_pos, patch_tokens_with_pos], axis=0)
    else:
        tokens = patch_tokens_with_pos

    return patches_flat, patch_tokens, tokens

if __name__ == "__main__":
    # 4x4 单通道图像
    img = np.arange(1, 17, dtype=np.float32).reshape(4, 4, 1)

    patches_flat, patch_tokens, tokens = patch_embed(
        image=img,
        patch_size=2,
        embed_dim=8,
        add_cls=True,
        seed=42,
    )

    assert patches_flat.shape == (4, 4)   # N=4, patch_dim=4
    assert patch_tokens.shape == (4, 8)   # N=4, D=8
    assert tokens.shape == (5, 8)         # [CLS] + 4 patch tokens

    expected = np.array([
        [1, 2, 5, 6],
        [3, 4, 7, 8],
        [9, 10, 13, 14],
        [11, 12, 15, 16],
    ], dtype=np.float32)
    assert np.array_equal(patches_flat, expected)

    print("patches_flat:")
    print(patches_flat)
    print("patch_tokens shape:", patch_tokens.shape)
    print("tokens shape after [CLS] + position:", tokens.shape)
```

这段代码里最关键的验证有三个。

| 验证项 | 结果 | 对应概念 |
|---|---|---|
| `patches_flat.shape == (4, 4)` | 成立 | 一共 4 个 patch，每个 patch 展平后长度是 4 |
| `patches_flat[0] == [1,2,5,6]` | 成立 | patch 顺序按从上到下、从左到右扫描 |
| `tokens.shape == (5, 8)` | 成立 | 加入 `[CLS]` 后，序列长度从 4 变成 5 |

如果换成工程实现，最常见的是 PyTorch 版本。下面给一个可直接运行的简化 `PatchEmbed` 模块，它不再是伪代码，而是完整可执行的最小实现。

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # 等价于“每个 patch 做一次线性投影”
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, N+1, D]
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "input size mismatch"

        x = self.proj(x)                   # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)   # [B, N, D]

        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)          # [B, N+1, D]
        x = x + self.pos_embed
        return x

if __name__ == "__main__":
    model = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)

    assert out.shape == (2, 197, 768)
    print("output shape:", out.shape)
```

这个实现里，形状变化可以直接对照：

$$
[B, C, H, W]
\rightarrow
[B, D, H/P, W/P]
\rightarrow
[B, N, D]
\rightarrow
[B, N+1, D]
$$

以 `224 x 224` 输入、`patch_size=16` 为例：

$$
N = 14 \times 14 = 196,\quad N+1 = 197
$$

所以输出形状是：

$$
[B, 197, 768]
$$

真实工程里还会多处理两个问题。

第一，输入分辨率变化时的位置编码插值。训练用 `224 x 224`，推理改成 `384 x 384`，patch 数从 $196$ 变成：

$$
\left(\frac{384}{16}\right)^2 = 24^2 = 576
$$

这时原来的位置编码长度不够，必须插值或重建。

第二，多模态序列的拼接规则。视觉 token 通常不是直接裸拼，而是要结合 `<image>`、`<bos>`、query token、attention mask 等机制一起组织。视觉入口统一成 token 序列，只是多模态系统的第一步，不是最后一步。

---

## 工程权衡与常见坑

最常见的工程权衡，是 patch 大小和序列长度之间的矛盾。patch 越小，细节越完整；patch 越小，token 也越多。由于标准自注意力代价接近序列长度平方，所以这不是线性增长，而是很快变贵。

| patch 大小 | token 数量 | 细节保留 | 计算开销 | 典型风险 |
|---|---|---|---|---|
| 大 | 少 | 弱 | 低 | 小目标丢失、边界模糊 |
| 小 | 多 | 强 | 高 | 显存压力大、训练慢 |
| 重叠 patch | 更多 | 更强 | 更高 | 推理成本上升 |
| 动态分辨率 | 可变 | 更灵活 | 实现复杂 | 位置编码处理更难 |

一个最容易犯的误区是“patch 越小越好”。这不成立。以 $224 \times 224$ 图像为例：

| $P$ | patch 网格 | token 数 $N$ |
|---|---|---|
| 32 | $7 \times 7$ | 49 |
| 16 | $14 \times 14$ | 196 |
| 8 | $28 \times 28$ | 784 |

从 $P=16$ 变成 $P=8$，token 数从 196 变成 784，是 4 倍；如果仍用全局注意力，注意力矩阵规模会接近放大到 16 倍。对分类任务，这笔额外成本未必能换来等比例收益。

第二个坑是输入分辨率变化带来的位置编码失配。训练时如果使用固定长度的绝对位置编码，推理时直接更换输入尺寸，位置表长度会不一致。工程上常见做法是把位置编码恢复成二维网格后插值，再展平回一维序列。这个方法可用，但只是近似，不是严格等价。对于空间结构高度敏感的任务，例如文档版面分析、遥感、显微图像，插值误差可能更明显。

第三个坑是分块边界断裂。非重叠切块会让原本连续的边缘、笔画、细纹理被拆到不同 patch 中。后续注意力层理论上可以重新聚合这些信息，但这意味着模型要先承受“打碎”，再学习“重组”。在数据不足时，这条学习路径往往比 CNN 的局部连续建模更难。

第四个坑是把 `PatchEmbed` 误认为纯数据变形层。它实际上是一个可学习投影层，会直接决定视觉 token 的初始质量。如果训练数据主要是自然图像，而部署场景变成工业缺陷、医学切片、夜间监控或文档扫描，patch 级表示可能一开始就不稳定。后面层再强，也只能在一个已经偏移的输入空间上继续工作。

第五个坑是图像增强和位置编码的相互作用。已有研究指出，ViT 在某些对比度变化场景下可能比 CNN 更敏感，原因与 patch embedding 和位置编码的耦合有关。某些变体会在 patch embedding 附近引入 Pre-LayerNorm，以改善数值稳定性和分布偏移下的一致性。这类结论不是“所有 ViT 都一定脆弱”，而是说明视觉入口设计会影响鲁棒性表现。

| 结构 | 对 contrast 变化的稳定性 | 训练稳定性 | 说明 |
|---|---|---|---|
| Post-LayerNorm 风格入口 | 较弱 | 一般 | 分布偏移时更容易波动 |
| Pre-LayerNorm 风格入口 | 较强 | 更好 | 有助于稳定特征尺度与位置编码配合 |

真实工程里，这些问题会直接表现为业务错误。OCR 是典型例子。如果一行小字恰好跨越 patch 边界，且 patch 足够大，那么多个字符会被混在同一个 token 内，后续视觉文本对齐就会变差。表面上像是“语言模型读错了图”，实质上是视觉入口在切块阶段已经损失了可分辨结构。

---

## 替代方案与适用边界

ViT 的 patch embedding 不是唯一方案，它只是最标准、最统一、最容易接入 Transformer 主干的方案。不同任务下，替代路线可能更合适。

第一类替代方案是 `Conv + Flatten`。先用卷积提取局部特征，再把卷积特征图展平为 token。这样做保留了卷积的局部归纳偏置，更适合小样本、纹理敏感或边缘结构重要的任务。

第二类是 Hybrid CNN+Transformer。前几层用 CNN 做局部编码和降采样，后面再交给 Transformer 处理长程依赖。这类方案常见于检测、分割和高分辨率输入，因为直接把原图切成细粒度 patch 的成本太高。

第三类是 overlapping patch。相邻 patch 之间保留重叠区域，可以缓解块边界断裂问题，但会增加 token 数和计算量。OCR、细粒度识别、版面理解等任务更容易从这种设计中获益。

第四类是 local attention 或 window attention。代表性方案是 Swin Transformer。它不是让每个 token 与所有 token 做全局交互，而是先在局部窗口内建模，再通过跨窗口机制逐步传播全局信息。这种设计更适合超高分辨率图像，因为复杂度扩展性更好。

第五类是层级式视觉编码。它会在网络不同深度维护不同分辨率的特征表示，更接近 CNN 的特征金字塔思想。需要多尺度特征的任务，例如检测、分割、遥感、视频理解，往往更依赖这种结构。

| 方案 | 优势 | 劣势 | 更适合的场景 |
|---|---|---|---|
| ViT Patch Embed | 结构统一、易接入大模型 | 对数据和算力要求高 | 大规模预训练、分类、多模态 |
| Conv + Flatten | 局部偏置强、训练更稳 | 全局建模起点较弱 | 小数据、纹理任务、工业视觉 |
| Hybrid CNN+Transformer | 局部与全局折中 | 结构更复杂 | 高分辨率、检测、分割 |
| Overlapping Patch | 边界信息更连续 | token 增多 | 细粒度识别、OCR |
| Local/Window Attention | 更易扩展到大图 | 全局传播更慢 | 文档、遥感、超高分辨率输入 |
| 层级式视觉编码 | 多尺度表示自然 | 实现复杂度更高 | 检测、分割、视频、多尺度任务 |

新手可以用一个简单标准判断边界：如果任务的主要收益来自全局关系建模，且有足够预训练数据或可复用大模型权重，那么标准 patch embedding 很合适；如果任务更依赖局部纹理、小样本学习、细小目标或边界连续性，那么纯非重叠 patch 入口往往不是最稳妥的起点。

一个更工程化的判断标准是：先看目标对象占整图的比例。如果目标经常只占很小区域，例如二维码、裂纹、缺陷点、细小文字、病灶边缘，那么固定大 patch 很可能天然吃亏。此时先用卷积做局部编码，或采用重叠 patch、窗口注意力、层级结构，通常比“一步切成大块再全局注意力”更稳。

---

## 参考资料

| 分类 | 资料 | 重点贡献 |
|---|---|---|
| 原始论文 | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | 首次系统化给出 ViT 的 patchify、线性投影、位置编码和 `[CLS]` 设计 |
| 官方实现 | [Torchvision VisionTransformer 源码](https://docs.pytorch.org/vision/0.26/_modules/torchvision/models/vision_transformer.html) | 展示 `Conv2d(kernel_size=P, stride=P)` 如何等价实现 patch embedding |
| 多模态工程 | [Hugging Face BLIP-2 文档](https://huggingface.co/docs/transformers/model_doc/blip-2) | 展示视觉 token 如何作为视觉编码器输出接入多模态堆栈 |
| 多模态源码 | [transformers/models/blip_2/modeling_blip_2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py) | 可直接查看视觉 embedding、位置编码和后续模块的实际接口 |
| 鲁棒性研究 | [Improved robustness of vision transformers via prelayernorm in patch embedding](https://doi.org/10.1016/j.patcog.2023.109659) | 分析对比度变化下 patch embedding 与位置编码的耦合问题，以及 Pre-LayerNorm 的缓解作用 |
| 替代结构 | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | 给出窗口注意力和层级式视觉编码，说明标准全局 patch 序列并非唯一入口 |

建议阅读顺序可以按“先概念、再实现、再边界”的方式展开：

| 阅读顺序 | 类型 | 作用 |
|---|---|---|
| 1 | 原始论文 | 建立最标准的 ViT patch embedding 定义和符号体系 |
| 2 | 官方实现 | 把公式和真实代码一一对齐，理解 `Conv2d` 等价性 |
| 3 | 多模态工程 | 理解视觉 token 为什么能与文本 token 共享 Transformer 接口 |
| 4 | 鲁棒性研究 | 理解 patch embedding 在真实分布偏移下为什么可能失效 |
| 5 | 替代结构 | 理解什么时候该继续用标准 patch，什么时候该换窗口或层级方案 |
