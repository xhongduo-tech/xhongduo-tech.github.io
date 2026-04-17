## 核心结论

DiT（Diffusion Transformer）的核心变化只有一句话：它把扩散模型里负责“去噪”的主干网络，从 U-Net 换成了纯 Transformer。

这里的“去噪网络”可以先理解成扩散模型每一步都要调用的预测器，它接收带噪声的图像表示，输出当前步应当去掉多少噪声。DiT 没有沿用 U-Net 的编码器、解码器和跳连结构，而是把图像先变成一串 token，再让这些 token 一起进入标准 Transformer block 做全局自注意力。

它成立的前提有三个：

| 关键部件 | 白话解释 | 作用 |
|---|---|---|
| Patchified latent tokens | 先把 VAE latent 切成小块，再把每块当成一个 token | 让图像可以像文本一样进入 Transformer |
| adaLN-Zero | 一种带条件控制的归一化与残差门控方式 | 把 timestep 和类别等条件稳定注入每一层 |
| Standard ViT block | 标准视觉 Transformer 块 | 用统一的自注意力替代 U-Net 的多分支结构 |

最重要的经验结论不是“Transformer 也能做扩散”，而是“Transformer 在扩散里同样表现出 scaling law”。“Scaling law”直白说，就是模型算力、参数量、token 数增加后，性能会按稳定趋势提升。在 DiT 论文里，这个性能常用 FID 衡量。FID 可以先理解成“生成图像和真实图像分布差距的指标，越低越好”。

一个最直观的玩具例子是：把一个 `32×32×4` 的 latent 切成 `4×4` patch，那么会得到 `8×8=64` 个 token。再把 timestep、类别条件编码进网络，每层让这 64 个 token 一起做 self-attention。你可以把它理解为：原来 U-Net 依赖“上下采样路径”聚合信息，现在 DiT 依赖“所有 token 同时全局通信”。

DiT 的价值不只是结构更整齐，还在于它给后续 SD3、Flux 这类新一代架构提供了统一模板：图像 token、文本 token、时间步条件都作为序列处理，条件融合不再依赖 U-Net 中零散的 cross-attention 补丁式设计。

---

## 问题定义与边界

DiT 解决的问题，不是“怎么发明一种全新的生成范式”，而是更具体的一件事：在 Latent Diffusion 这个框架里，能不能不用 U-Net，而改用纯 Transformer 作为去噪网络，同时还保留条件控制能力和高质量生成效果。

“Latent Diffusion”可以先理解成“不是直接在像素空间做扩散，而是在 VAE 压缩后的 latent 空间做扩散”。这样做的原因很现实：像素太大，计算太贵；latent 更小，训练和采样都更可控。

问题的边界也很明确。

第一，DiT 处理的不是原始图像，而是 latent。假设输入图像是 `256×256`，经过 VAE 编码后可能得到 `32×32×4` 的 latent 张量。DiT 不直接看像素，而是看这个 latent。

第二，Transformer 不能直接吃二维张量，所以必须 patchify，也就是切块。若 patch size 为 $p$，latent 高宽为 $H, W$，则 token 数量为：

$$
N = \frac{H}{p} \times \frac{W}{p}
$$

如果是正方形 latent，且 $H=W=L$，则可写成：

$$
N = \left(\frac{L}{p}\right)^2
$$

若再拼接条件 token，输入序列长度就是：

$$
S = N + C
$$

其中 $C$ 表示条件 token 数，例如 timestep token、class token，或者更复杂场景下的 text token。

第三，去噪网络不能只接收图像 token，还必须知道当前扩散步。这是因为扩散的每一步噪声强度不同，模型如果不知道当前是第几步，就无法判断应当去掉多少噪声。这个“当前扩散步”通常记作 timestep。

第四，Transformer 堆深之后训练稳定性是硬约束。U-Net 的结构天然有局部归纳偏置，也就是它天然偏向处理局部空间结构；Transformer 没有这个优势，序列又更长，所以残差路径必须设计得更稳。DiT 里这件事主要靠 adaLN-Zero 完成。

一个简化例子如下。

假设我们有一张 `256×256` 图像，经 VAE 压缩为 `32×32×4` latent。若 patch size 取 `4`，则会得到 `8×8=64` 个 patch。每个 patch 的体积是 `4×4×4=64`，把它拉平成向量，再映射到隐藏维度，例如 768 维。于是输入序列就是 64 个图像 token。然后把 timestep embedding 和类别 embedding 作为条件输入，送进每一层的 adaLN-Zero 模块，让每一层都知道“当前是第几步、想生成什么类”。

这里要强调一个边界：DiT 不是天然更适合所有图像任务。它主要适合“可以接受较大自注意力成本，并希望模型随着规模扩展持续获益”的场景。如果分辨率极高、显存很紧，传统 U-Net 仍然可能更务实。

---

## 核心机制与推导

DiT 的核心机制可以拆成两层：输入如何序列化，条件如何稳定注入。

先看输入序列化。

传统 U-Net 的思路是二维特征图逐层下采样、再上采样，依赖不同分辨率的卷积特征完成局部到全局的信息融合。DiT 不这样做。它把图像 latent 切成 token，之后所有 token 进入同一套 Transformer block。于是模型不再显式区分“编码路径”和“解码路径”，而是通过自注意力让任意两个 token 直接通信。

可以画成一个很粗的草图：

```text
latent -> patchify -> token embeddings -> Transformer blocks -> unpatchify -> noise prediction
```

如果把条件也考虑进去，更准确的图是：

```text
image latent tokens ----\
timestep embedding ------> adaLN-Zero in each block -> residual stack -> output tokens
label/text embedding ----/
```

再看真正让 DiT 成立的关键：adaLN-Zero。

“LayerNorm”可以先理解成“把每个 token 的特征做标准化，减少训练时数值漂移”。普通 LayerNorm 只有固定参数，而 adaLN 是 adaptive LayerNorm，也就是“归一化后的缩放和偏移，不再是常量，而是由条件动态生成”。这些条件包括 timestep、类别、文本等。

DiT 中常见的残差形式可以写成：

$$
y = x + \alpha \cdot F(\text{LN}(x))
$$

其中：

- $x$ 是输入 token；
- $\text{LN}(x)$ 是归一化后的 token；
- $F(\cdot)$ 是 attention 或 MLP 子层；
- $\alpha$ 是由条件生成的门控系数。

在 adaLN-Zero 里，条件向量会回归出若干参数，常记为 $\gamma, \beta, \alpha$。其中 $\gamma, \beta$ 作用在归一化输出上，相当于动态缩放与平移；$\alpha$ 作用在残差分支前，决定这一层当前应该“放出多少改动”。

更完整地写，某一层可以理解为：

$$
\hat{x} = \gamma(c) \odot \text{LN}(x) + \beta(c)
$$

$$
y = x + \alpha(c) \odot F(\hat{x})
$$

这里的 $c$ 是条件向量，比如 timestep embedding 和 class embedding 的合成结果；$\odot$ 表示逐元素乘法。

最关键的一点是：$\alpha$ 采用零初始化。也就是说，在训练刚开始时，

$$
\alpha(c) \approx 0
$$

于是整层近似退化成：

$$
y \approx x
$$

这件事的意义非常大。因为它让每个 block 初始行为接近恒等映射，也就是“先别乱改输入”。对于很深的 Transformer，这相当于给优化过程加了一个稳定器：梯度不会因为残差分支初始过强而爆炸，网络也不会一开始就把随机噪声特征放大到失控。

可以把它理解成一个玩具例子。假设某层 block 的主分支输出幅值大约在 1 左右。如果没有零初始化门控，残差会直接把这份随机输出加回主路径，很多层叠加后，表示会迅速漂移。若 $\alpha=0$ 起步，那么第一步训练时这层几乎就是“旁观者”，等模型逐渐学会如何利用条件后，再慢慢把残差通道打开。

这也是 DiT 和“把 ViT 直接套到扩散上”之间的真正差别。难点不是“能不能把图像切成 token”，而是“如何让深层 Transformer 在扩散训练里稳定工作，并让时间步和条件信息逐层有效注入”。

DiT 的 scaling law 也可以在这个机制上理解。Transformer 的表达能力主要来自三个可扩展维度：宽度、深度、token 数。对于 DiT 来说，token 数又直接受 patch size 影响。patch 越小，token 越多，注意力成本越高，但图像细节和空间关系表达通常也更充分。因此论文中能观察到：GFlops 增大时，FID 往往下降。

下面这张表反映的是趋势，而不是要求记住每个精确数字：

| 模型 | 大致参数量 | 大致 GFlops | FID 趋势 |
|---|---:|---:|---:|
| DiT-S/8 | 33M | 0.36 | 高 |
| DiT-B/4 | 更大 | 更高 | 更低 |
| DiT-L/4 | 更大 | 更高 | 继续下降 |
| DiT-XL/2 | 675M | 118.6 | 约 19.5，显著更低 |

这张表要表达的核心不是“XL 一定最好”，而是“当主干换成 Transformer 后，扩大模型规模依然能稳定换来生成质量收益”。这正是 DiT 被广泛关注的原因。

---

## 代码实现

下面先给一个最小玩具实现，用 Python 演示三件事：

1. 如何把 latent 切成 patch token；
2. 如何计算 token 数量；
3. 为什么 zero-init 的门控会让残差层初始近似恒等。

```python
import numpy as np

def patchify(latent, patch_size):
    """
    latent: (H, W, C)
    return: (num_tokens, patch_dim)
    """
    H, W, C = latent.shape
    assert H % patch_size == 0
    assert W % patch_size == 0

    tokens = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = latent[i:i+patch_size, j:j+patch_size, :]
            tokens.append(patch.reshape(-1))
    return np.stack(tokens, axis=0)

def adaLN_zero_residual(x, block_out, alpha):
    """
    x: residual input
    block_out: transformed branch output
    alpha: gate scalar or vector
    """
    return x + alpha * block_out

# toy latent: 32x32x4
latent = np.zeros((32, 32, 4), dtype=np.float32)
tokens = patchify(latent, patch_size=4)

# 32x32 split by 4 => 8x8 = 64 tokens
assert tokens.shape == (64, 4 * 4 * 4)

# zero-init alpha means near-identity block
x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
block_out = np.array([10.0, -5.0, 2.0], dtype=np.float32)

y0 = adaLN_zero_residual(x, block_out, alpha=0.0)
assert np.allclose(y0, x)

# if alpha opens, the block begins to modify representation
y1 = adaLN_zero_residual(x, block_out, alpha=0.1)
assert np.allclose(y1, np.array([2.0, 1.5, 3.2], dtype=np.float32))

print("patchify and adaLN-Zero toy example passed")
```

上面这段代码不是完整 DiT，只是把最重要的机制抽了出来。对初学者来说，先看懂这两点就够了：

- patchify 让二维 latent 变成序列；
- zero-init gate 让残差层起步时近似恒等。

如果写成更接近真实框架的伪代码，流程通常如下：

```text
latent x_t
  -> patchify
  -> linear projection to hidden_dim
  -> add positional embedding
  -> for each transformer block:
       cond = timestep_embed(t) + class_or_text_embed(c)
       h = adaLN_zero_attention_block(h, cond)
       h = adaLN_zero_mlp_block(h, cond)
  -> final layer
  -> unpatchify
  -> predict noise epsilon or velocity v
```

真实工程例子可以看 Stable Diffusion 3 这一类模型。它已经不是“图像 token + 一个类别 embedding”这么简单，而是会把多个文本编码器的输出一起引入，例如 CLIP 系列和 T5 系列文本特征。此时“条件输入”不只是一个向量，而是整串文本 token。MMDiT 的做法是让图像 token 和文本 token 在同一个 Transformer 系统里统一处理，或者在双流结构中交互。它继承的正是 DiT 的核心思想：图像表示被 token 化，条件控制通过 Transformer 级别统一建模。

工程上还要补几项实现细节。

第一，位置编码不能省。因为 patchify 后 token 顺序本身不携带二维空间含义，必须明确告诉模型“这个 token 来自左上角还是右下角”。

第二，输出头通常要把 token 重新还原成 patch，再拼回 latent 形状。这个过程叫 unpatchify。

第三，损失函数仍然是扩散训练熟悉的目标，例如预测噪声 $\epsilon$ 的 L2 损失：

$$
\mathcal{L} = \mathbb{E}_{x_0,\epsilon,t}\left[\|\epsilon - \epsilon_\theta(x_t, t, c)\|_2^2\right]
$$

这里的 $\epsilon_\theta$ 就是 DiT 预测器本身。

---

## 工程权衡与常见坑

DiT 的工程难点不在“能不能跑起来”，而在“能不能稳定且高效地扩展”。

先看最常见的坑：

| 问题 | 后果 | 处理方式 |
|---|---|---|
| 去掉 adaLN 的 zero init | 初期训练不稳定，深层残差输出过强 | 保留 $\alpha$ 零初始化，检查初始化代码是否真的生效 |
| patch 太大，token 太少 | 细节和局部结构表达变弱，FID 不一定下降 | 优先用更小 patch，再评估显存和吞吐 |
| 条件输入顺序错乱 | timestep 或文本条件注入失效 | 固定 embedding 组装顺序，训练推理一致 |
| 不加位置编码 | 模型难以恢复空间结构 | 显式加入 2D 位置编码或等价机制 |
| 直接照搬 ViT 超参 | 训练速度慢或收敛差 | 根据扩散任务单独调 block 深度、hidden dim、patch size |

第一个真实工程场景是：有人为了“简化代码”删掉了 zero-init 的门控初始化。结果模型不是完全训不动，而是出现更隐蔽的问题：loss 曲线前期震荡很大，梯度裁剪频繁触发，最终样本质量明显落后。原因不是 Transformer 本身有问题，而是扩散训练要求网络在很多 timestep 上保持稳定预测；如果残差层一开始就过于激进，会放大随机表示，优化器要花更多步骤把网络拉回可训练区间。

第二个真实工程场景是：把 latent patch size 从 `4×4` 提大到 `8×8`，因为这样 token 数更少、显存更省。短期看吞吐提升了，但生成质量可能不升反降。原因很直接。若 latent 还是 `32×32`，那么 patch size 从 4 变成 8，token 数会从 64 变成 16：

$$
64 = \left(\frac{32}{4}\right)^2,\quad 16 = \left(\frac{32}{8}\right)^2
$$

token 数直接变成原来的四分之一。自注意力的通信对象变少了，细节建模也更粗了。对分类任务，粗粒度 patch 有时还能接受；但对生成任务，尤其是扩散去噪这种逐步修复细节的任务，信息损失更明显。

实际排查时，建议直接做以下检查点：

- `adaLN-Zero` 的门控参数是否真的零初始化，而不是只在论文里写了、代码里忘了。
- `patch_size` 改动后，token 数是否仍在合理范围内。
- `timestep embedding`、`class/text embedding` 是否训练和推理都按同一顺序送入。
- 位置编码是否和 patch 网格一一对应。
- 输出 `unpatchify` 时是否严格对齐原始 latent 维度。

DiT 还有一个很实际的权衡：attention 成本。自注意力的复杂度对序列长度敏感，token 一多，显存和计算会快速上涨。这意味着 DiT 的优势往往建立在 latent 空间而不是像素空间中。如果你直接在高分辨率像素上做同样事情，代价通常不可接受。

---

## 替代方案与适用边界

DiT 不是扩散模型唯一可行的骨干网络，而是一条在大模型时代非常顺滑的路线。

最直接的替代方案仍然是 U-Net。U-Net 的优点是局部归纳偏置强，也就是它天然适合处理图像局部结构；在高分辨率和有限资源下，它常常更省。它的代价是条件融合、结构扩展和跨模态统一处理通常更复杂，经常需要在不同分辨率层里插入 cross-attention 或额外模块。

DiT 之后，更重要的演进方向是 MMDiT。这里的“MMDiT”可以先理解成“多模态的 DiT”，即不仅处理图像 token，也处理文本 token，甚至让不同模态共享或协同使用 Transformer block。

下面用表格看 DiT、SD3、Flux 的关系：

| 模型 | token 类型 | 条件方式 | 采样/训练特点 | 适用场景 |
|---|---|---|---|---|
| DiT | 图像 latent patch token | timestep + class/condition，经 adaLN-Zero 注入 | 标准扩散训练 | 纯图像生成、类别条件生成 |
| SD3 | 图像 token + 多路文本 token | MMDiT 统一处理文本与图像 | 引入 Rectified Flow 等改进 | 文本到图像，多模态统一建模 |
| Flux | 图像 token + 文本 token | 继承 MMDiT 思路并做工程强化 | guidance distillation，少步采样更强 | 追求更快推理和更强文本对齐 |

可以再画一个简单流程对比：

```text
DiT:
image latent patches -> Transformer -> noise prediction

MMDiT (SD3 / Flux):
text tokens + image latent patches -> joint/dual Transformer -> flow/noise prediction
```

对初学者，一个好理解的说法是：DiT 先证明了“图像扩散的去噪网络可以被 token 化并交给 Transformer”；SD3 和 Flux 则进一步证明“文本和图像也可以在 token 层面更深地统一建模”。

适用边界也要说清楚。

如果你的目标是研究大规模生成模型、希望参数继续加大后性能还能稳步提升，DiT 路线通常更自然，因为它直接继承了 Transformer 生态里的扩展经验。

如果你的资源有限，或者任务分辨率极高、推理速度要求苛刻，那么 U-Net 路线依然很有现实价值。原因不是 DiT 不先进，而是注意力成本不会因为理论优雅而自动消失。

如果你的任务已经进入多模态条件生成，例如复杂文本到图像、图像和文本联合编辑，那么 MMDiT 往往比“U-Net 加很多 cross-attention 补丁”更统一。但它的工程复杂度也更高，对 token 组织、掩码策略、条件顺序和推理路径的要求更严格。

---

## 参考资料

1. 《Scalable Diffusion Models with Transformers》
   说明：DiT 原始论文，核心看点是 patchified latent、adaLN-Zero、模型规模与 FID 的 scaling law。
   链接提示：可查论文全文或其公开镜像版本。

2. Hugging Face Diffusers `SD3Transformer2DModel` 文档
   说明：适合看 SD3 中 Transformer 去噪主干的接口设计，尤其是文本和图像 token 如何统一进入模型。
   链接提示：查 `SD3Transformer2DModel`、`MMDiT`、`transformer_2d` 相关文档页。

3. Stable Diffusion 3 / Flux 架构分析文章
   说明：适合补工程上下文，理解 MMDiT、Rectified Flow、guidance distillation 这些 DiT 后继设计如何落地。
   链接提示：优先查带结构图和模块拆解的技术文档。

4. Diffusion Model 基础资料
   说明：如果对 $\epsilon$-prediction、timestep、scheduler 这些术语还不熟，先补扩散训练目标，再回来看 DiT 会更顺。
   链接提示：查 DDPM、Latent Diffusion、classifier-free guidance。
