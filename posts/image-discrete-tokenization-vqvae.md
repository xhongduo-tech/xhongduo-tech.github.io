## 核心结论

图像离散 tokenization 的核心作用，是把“图像压缩”与“图像生成”拆成两个相对独立的问题。前半段由 `编码器 -> 码本量化 -> 解码器` 负责，把连续像素压成离散 token；后半段由自回归模型负责，把 token 当成“视觉词表”逐个预测。这样做的价值是，生成模型不再直接面对 $256 \times 256 \times 3$ 个像素，而是面对长度更短、结构更规则的离散序列。

VQ-VAE 是这个路线的基础版本。它先把图像编码成连续特征，再用最近邻查表把特征映射到码本中的离散向量，最后解码回图像。其基本训练目标可写为：

$$
\mathcal{L}=\mathcal{L}_{rec}+\mathcal{L}_{vq}+\beta \mathcal{L}_{commit}
$$

这里的 reconstruction loss 是重建误差，白话解释就是“还原出来的图和原图差多远”；commitment loss 是承诺损失，白话解释就是“要求编码器别老在码本边界附近乱晃，要稳定贴近某个码字”。

VQGAN 在 VQ-VAE 的基础上加了感知损失和对抗损失。感知损失关注高层特征相似性，不只看像素差；对抗损失通过判别器逼迫重建结果更像真实图像。结果通常是：VQ-VAE 更偏“压缩可用”，VQGAN 更偏“纹理真实”。

视觉自回归模型如 LlamaGen，把这些离散 token 当成语言模型里的词，按顺序预测下一个 token。于是整体流程变成：

```text
图像 -> 编码器 -> 离散 token -> 解码器
                     |
                     v
              自回归模型预测 token 序列
```

玩具例子可以这样理解：把一张 `256x256` 彩图压成 `16x16` 的 token 网格，一共 256 个 token。每个 token 不是一个像素，而是一个“局部视觉片段编号”。后续生成器只要学会“第 1 个编号后面最可能接哪个编号”，就能像语言模型续写句子一样续写图像。

---

## 问题定义与边界

这个问题的正式定义是：用有限大小的离散码本，近似连续图像分布，同时保留足够的结构信息和纹理信息，便于后续生成模型建模。

边界首先在压缩阶段。图像是连续值，码本是离散值，量化一定带来信息损失。压得太狠，token 序列会变短，但细节会丢；压得太轻，重建更清楚，但 token 序列会过长，自回归建模成本会上升。

如果输入图像大小是 $H \times W$，编码器总下采样倍率是 $f$，那么 token 数量通常是：

$$
N = \frac{H}{f} \times \frac{W}{f}
$$

如果用 patch size $p$ 直接切块，并且每个 patch 只输出一个离散索引，那么也是：

$$
N = \frac{H}{p} \times \frac{W}{p}
$$

注意这里通常不再乘 RGB channel，因为颜色通道已经被编码进连续特征，再由一个码本向量表示。

下面这个表格能直接看出边界在哪里：

| 方案 | 输入图像 | 下采样/patch | token 数 | 优点 | 代价 |
|---|---:|---:|---:|---|---|
| 粗压缩 | 256x256 | 16 | 16x16=256 | 序列短，AR 好训练 | 细节损失更明显 |
| 中等压缩 | 256x256 | 8 | 32x32=1024 | 结构和纹理更平衡 | 序列明显变长 |
| 细压缩 | 256x256 | 4 | 64x64=4096 | 重建质量更高 | AR 计算成本很高 |

玩具例子：一张 `32x32` 图像，若按 `16x16` 切块，就只有 4 个 patch。某个 patch 经编码器得到连续特征 $z_e=[1.2,-0.6]$，再和 512 个码本向量比距离，最近的是第 101 个码字，于是这个 patch 被表示成 token `101`。这样后续模型预测的是“下一个 token 是多少”，而不是“下一个像素值是多少”。

真实工程例子：LlamaGen 这类模型通常先训练一个图像 tokenizer，把大规模图像数据变成离散 token，再训练 Llama 风格的自回归模型预测这些 token。这样训练目标统一成 next-token prediction，工程上可以复用大量语言模型训练经验，但代价是图像 token 序列往往比文本长得多。

---

## 核心机制与推导

VQ-VAE 的关键变量有三个。编码器输出 $z_e(x)$，这是连续 latent；码本是一个可训练向量集合 $\{e_k\}_{k=1}^K$；量化结果 $z_q(x)$ 是离 $z_e(x)$ 最近的码本向量。

最近邻量化可写为：

$$
k^\* = \arg\min_k \|z_e(x)-e_k\|_2^2,\qquad z_q(x)=e_{k^\*}
$$

白话解释：编码器先说“我想表达成这个连续向量”，量化器再回答“词表里最接近的是这个离散码字”。

总损失的常见写法是：

$$
\mathcal{L}=
\underbrace{\|x-\hat{x}\|}_{\mathcal{L}_{rec}}
+
\underbrace{\|sg[z_e(x)]-e_{k^\*}\|_2^2}_{\mathcal{L}_{vq}}
+
\beta\underbrace{\|z_e(x)-sg[e_{k^\*}]\|_2^2}_{\mathcal{L}_{commit}}
$$

其中 $sg[\cdot]$ 是 stop-gradient，白话解释就是“前向时当正常值用，反向时不传梯度”。

它的作用可以分开看：

| 项 | 更新谁 | 作用 |
|---|---|---|
| $\mathcal{L}_{rec}$ | 编码器、解码器 | 让重建图接近原图 |
| $\mathcal{L}_{vq}$ | 码本 | 让选中的码字靠近编码器输出 |
| $\mathcal{L}_{commit}$ | 编码器 | 让编码器输出别频繁漂移，稳定贴近码字 |

玩具例子：设 $z_e=[0.7,1.2]$，码本中有 $e_0=[0.5,1.0]$，$e_1=[1.5,0.5]$。  
距离分别是：

$$
\|z_e-e_0\|_2^2=0.2^2+0.2^2=0.08
$$

$$
\|z_e-e_1\|_2^2=0.8^2+(-0.7)^2=1.13
$$

所以选中 $e_0$。之后 $\mathcal{L}_{vq}$ 把 $e_0$ 拉向 $z_e$，$\mathcal{L}_{commit}$ 把 $z_e$ 拉向 $e_0$。两边一起收敛，量化误差会更稳定。

VQGAN 的扩展点在于，单纯像素级重建常常得到“平均化”的结果，结构没错，但高频纹理发糊。所以它把训练目标扩成：

$$
\mathcal{L}_{VQGAN}
=
\mathcal{L}_{rec}
+\lambda_{perc}\mathcal{L}_{perc}
+\lambda_{gan}\mathcal{L}_{gan}
+\mathcal{L}_{vq}
+\beta \mathcal{L}_{commit}
$$

感知损失 $\mathcal{L}_{perc}$ 通常比较预训练网络的特征差异，白话解释就是“不是逐像素对齐，而是看内容和纹理是否像”；对抗损失 $\mathcal{L}_{gan}$ 由判别器提供，白话解释就是“逼解码器骗过一个挑细节的审核器”。

VQ-VAE 和 VQGAN 的差异可以压缩成下面这张表：

| 模型 | 重建误差 | 感知损失 | GAN loss | 常见结果 |
|---|---|---|---|---|
| VQ-VAE | 有 | 无 | 无 | 结构保留较好，但细节偏糊 |
| VQGAN | 有 | 有 | 有 | 纹理更真实，但训练更难稳定 |

当离散 token 质量足够高后，自回归模型就可以接手。做法与语言模型一致：给定前面的 token，预测下一个 token 的概率分布：

$$
p(t_1,\dots,t_N)=\prod_{i=1}^{N} p(t_i \mid t_{<i})
$$

这就是 LlamaGen 这类模型的核心：先有好 tokenizer，再把图像当序列建模。

---

## 代码实现

下面给一个可运行的最小 Python 例子，只演示“量化 + loss 计算”。这里不依赖深度学习框架，目的是把机制讲清楚。

```python
import math

def squared_l2(a, b):
    assert len(a) == len(b)
    return sum((x - y) ** 2 for x, y in zip(a, b))

def quantize(z_e, codebook):
    assert len(codebook) > 0
    distances = [squared_l2(z_e, e) for e in codebook]
    k = min(range(len(distances)), key=lambda i: distances[i])
    return k, codebook[k], distances

def vq_losses(z_e, e_q, x, x_hat, beta=0.25):
    rec = squared_l2(x, x_hat)
    loss_vq = squared_l2(z_e, e_q)
    loss_commit = beta * squared_l2(z_e, e_q)
    total = rec + loss_vq + loss_commit
    return {
        "rec": rec,
        "vq": loss_vq,
        "commit": loss_commit,
        "total": total,
    }

# 玩具例子
z_e = [0.7, 1.2]
codebook = [
    [0.5, 1.0],
    [1.5, 0.5],
    [-0.2, 0.1],
]
k, e_q, distances = quantize(z_e, codebook)

assert k == 0
assert abs(distances[0] - 0.08) < 1e-9
assert distances[0] < distances[1]

# 假设原图和重建图各只有两个像素值
x = [0.0, 1.0]
x_hat = [0.1, 0.9]

losses = vq_losses(z_e, e_q, x, x_hat, beta=0.25)
assert losses["rec"] == 0.02
assert abs(losses["vq"] - 0.08) < 1e-9
assert abs(losses["commit"] - 0.02) < 1e-9
assert abs(losses["total"] - 0.12) < 1e-9

print("token_id =", k)
print(losses)
```

如果换成真实训练代码，结构通常如下：

```python
z_e = encoder(x)                   # [B, H', W', D]
token_id = argmin_distance(z_e, codebook)
z_q = codebook[token_id]

loss_rec = recon_loss(decoder(z_q), x)
loss_vq = ((sg(z_e) - z_q) ** 2).mean()
loss_commit = beta * ((z_e - sg(z_q)) ** 2).mean()

loss = loss_rec + loss_vq + loss_commit
```

组件职责可以直接记成这张表：

| 组件 | 输入 | 输出 | 职责 |
|---|---|---|---|
| Encoder | 图像 | 连续 latent | 压缩局部视觉信息 |
| Quantizer | 连续 latent | 离散 token / 量化向量 | 最近邻查表 |
| Decoder | 量化向量 | 重建图像 | 从离散表示恢复图像 |
| AR Transformer | token 序列 | 下一个 token 分布 | 学习视觉序列概率 |

真实工程例子：在 LlamaGen 风格系统里，训练分两阶段。第一阶段只训练 tokenizer，目标是让 `图像 -> token -> 图像` 尽量保真；第二阶段冻结 tokenizer，把所有训练图像转成 token 网格，再按 raster scan，也就是从左到右、从上到下的顺序拉平成序列，交给 Transformer 做 next-token prediction。

---

## 工程权衡与常见坑

最常见的问题是 codebook collapse，也就是码本坍塌。白话解释是：虽然码本有很多槽位，但训练后大多数输入只会落到少数几个码字上，等于词表白建了。结果是表达能力下降，生成多样性变差。

可以通过统计 token 频率观察这个问题：

| 现象 | 可能原因 | 对策 |
|---|---|---|
| 少数 token 占比极高 | 编码器输出集中、码本更新不稳定 | 用 EMA 更新码本，调低学习率 |
| 重建发糊 | 只优化像素重建 | 加 perceptual loss，必要时用 VQGAN |
| 判别器过强导致训练震荡 | GAN 部分失衡 | 延迟启用 GAN，减小 $\lambda_{gan}$ |
| token 太多，AR 太慢 | patch 太小或压缩率太低 | 提高下采样倍率，或改多尺度方案 |

EMA 更新是常见稳定技巧。白话解释是：不是让码本完全跟着单个 batch 乱跳，而是做指数滑动平均，更新更平滑。

```python
# 伪代码
ema_count[k] = decay * ema_count[k] + (1 - decay) * assign_count[k]
ema_weight[k] = decay * ema_weight[k] + (1 - decay) * sum_assigned_z[k]
codebook[k] = ema_weight[k] / (ema_count[k] + 1e-5)
```

真实工程里还有一个常被低估的权衡：tokenizer 质量直接限制 AR 上限。如果 tokenizer 已经把细节压没了，后面的 Llama 再大也“预测不出不存在的信息”。因此很多项目先花大量时间把 VQGAN 训稳，再开始训练 AR 模型。

另一个坑是序列长度爆炸。`256x256` 图像在下采样 16 倍时有 256 个 token，还算可接受；若下采样只有 8 倍，就变成 1024 个 token。自回归注意力复杂度通常随序列长度快速上涨，训练成本和采样延迟都会显著增加。

---

## 替代方案与适用边界

如果目标是“像语言一样离散建模图像”，VQ-VAE/VQGAN 加自回归是自然路线。但它并不是唯一选择。

| 方案 | latent 类型 | 采样方式 | 优点 | 局限 |
|---|---|---|---|---|
| VQ-VAE / VQGAN + AR | 离散 | 逐 token 生成 | 统一到 next-token 范式，易接语言模型 | 序列长，采样慢 |
| Latent Diffusion | 连续 | 多步去噪 | 质量高，已成主流工程方案 | 采样步数多，训练链路更长 |
| 经典 GAN | 连续 | 一次前向 | 采样快 | 训练不稳定，模式覆盖常不足 |

适用边界可以这样理解：

1. 如果你要做“统一文本和图像 token”的体系，或者希望直接复用 Llama 类架构，离散 tokenization 很合适。
2. 如果你更在意最终画质和工业成熟度，latent diffusion 往往更稳。
3. 如果你要求极快采样且分辨率不高，GAN 仍然有价值，但训练调参门槛高。

再看一个具体例子。若项目需要生成商品主图，边缘清晰、纹理细节重要，可以把 patch 从 `16x16` 缩到 `8x8`，token 数翻 4 倍，但重建和编辑质量通常更好。若项目只是做低成本草图预览，或者重在语义而非细节，较大的 patch 和更短的 token 序列往往更划算。

因此，离散化不是“更先进”的通用答案，而是“为了让视觉生成进入 token 范式”做的一次明确取舍。

---

## 参考资料

建议阅读顺序是：先看 VQ-VAE 理解量化与损失，再看 VQGAN 理解感知与对抗训练，最后看 LlamaGen 理解“tokenizer + 自回归生成”的完整工程链路。

| 资料 | 链接 | 用途 |
|---|---|---|
| Neural Discrete Representation Learning | https://arxiv.org/abs/1711.00937 | VQ-VAE 原始论文，适合理解码本量化与 commitment loss |
| Taming Transformers for High-Resolution Image Synthesis | https://arxiv.org/abs/2012.09841 | VQGAN 原始论文，重点看感知损失、PatchGAN、离散视觉 token |
| CompVis Taming Transformers 项目页 | https://compvis.github.io/taming-transformers/ | 看整体流程图和高分辨率生成示意最直观 |
| LlamaGen GitHub | https://github.com/FoundationVision/LlamaGen | 看 tokenizer、AR 训练和采样代码的工程组织 |
| Large World Model (LWM) | https://github.com/LargeWorldModel/LWM | 看多模态自回归如何把图像/视频 token 接入长上下文模型 |
| Emergent Mind: VQ-VAE | https://www.emergentmind.com/topics/vector-quantised-v-vae-vq-vae | 适合快速复习公式与概念，但应配合原论文阅读 |

推荐阅读列表：

1. 先读 VQ-VAE 论文中的量化公式和 stop-gradient 设计。
2. 再读 VQGAN，重点看为什么只做重建会模糊，以及感知损失怎么补高频细节。
3. 最后看 LlamaGen 仓库，重点关注 tokenizer 配置、token 序列化方式、next-token 训练 loop。
4. 如果想扩展到图像之外，再看 LWM，理解“视觉 token + 长上下文自回归”如何扩展到视频和多模态场景。
