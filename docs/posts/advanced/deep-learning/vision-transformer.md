---
title: Vision Transformer（ViT）：图像分块嵌入
date: 2026-08-07
---

# Vision Transformer（ViT）：图像分块嵌入

<div class="epigraph">
<p>如果注意力能理解文字，它也该能理解像素——只要喂法正确。</p>
<footer>—— 依据 ViT 思想的通俗表达</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ Dosovitskiy 等《ViT》（2021） ｜ 2026-08-07</p>
</div>

## 为什么从 Vision Transformer 开始

Transformer 统治了 NLP，但视觉一直被 CNN 统治——直到 **Vision Transformer（ViT）**（Dosovitskiy 等, 2021）证明：**把图像切成块（patch）、当作「词」喂给标准 Transformer，在大规模数据上训练，效果能超过最好的 CNN**。ViT 的「图像 = 词序列」的翻译，让「同一套 Transformer 处理文本与图像」成为可能——这是「多模态大一统」的技术前提。

**ViT 的核心洞见**：CNN 的「卷积先验」（局部性、平移不变）在「大数据」下可以被「纯注意力」学会——**「数据足够多时，先验不那么必要，注意力自己会学」**。ViT 用「图像分块 → 块嵌入 → Transformer 编码器」的三步，把「视觉」搬进了「Transformer 的语法」里。本节把「分块嵌入」「位置编码」「CLS token」「与 CNN 的对比」讲透——它是现代视觉架构（Swin、MAE、CLIP 的视觉侧）的基座。<span class="marginnote">「ViT 的『翻译』」：图像 → 16 个块的序列 =「句子」；每个块嵌入 =「词向量」；位置编码 =「词序」；Transformer 编码器 =「理解句子」。<strong>「把一个『像素网格』翻译成『token 序列』，一切 NLP 的 Transformer 技术原样可用」</strong>——这个「序列化」是 ViT 最根本的方法论：把「非序列数据」切成 token，让 Transformer 处理。</span>

## 1 图像分块：从像素网格到 token 序列

**ViT 的第一步**：把图像 $224\times224\times3$ 切成 $16\times16$ 的**块（patch）**，得到 $\frac{224}{16}\times\frac{224}{16} = 14\times14 = 196$ 个块。

**每个块被「展平 + 线性投影」成一个向量（块嵌入，patch embedding）**：

$$
\boldsymbol{E}_i = \text{Linear}(\text{flatten}(\boldsymbol{x}_i)) \in \mathbb{R}^d
$$

其中 $\boldsymbol{x}_i$ 是第 $i$ 个 $16\times16\times3$ 的块（展平成 $768$ 维），线性投影到 $d$ 维（如 768）。**「每个块 = 一个 token」**——196 个 token 组成「图像句子」送入 Transformer。

**为什么「分块」而不是「逐像素」？** 逐像素的话，$224\times224=50176$ 个 token——**序列太长，注意力的 $O(n^2)$ 扛不住**。分块到 196 个 token，序列短、计算可行——**「块大小 = 序列长度与粒度（信息保真）的权衡」**。<span class="marginnote">「块大小 vs 序列长度」：$16\times16$ 块 → 196 token；$8\times8$ 块 → 784 token（更细但 $O(n^2)$ 贵 16 倍）；$32\times32$ 块 → 49 token（更粗但便宜）。「<strong>块大小是 ViT 的『分辨率旋钮』：块小 = 细节多但慢，块大 = 抽象但快</strong>」——Swin Transformer 用「层级分块」在不同层用不同粒度来平衡（下一篇）。</span>

**易错点：** 块嵌入的「展平」**破坏块内的空间结构**——但 Transformer 用「注意力」在块间学关系，「块内结构」被「线性投影」粗糙编码。「<strong>ViT 的粒度是「块级」不是「像素级」</strong>」——这是它与 CNN（像素级卷积）的根本差异。

## 2 位置编码与 CLS token

**ViT 的两个「NLP 移植」细节**：

**位置编码**：块序列需要「位置信息」（Transformer 置换不变）——ViT 用**可学习的 1D 位置编码**（每个位置一个可学向量，加到块嵌入上）：

$$
\boldsymbol{z}_0 = [\boldsymbol{E}_{\text{cls}}; \boldsymbol{E}_1 + \boldsymbol{p}_1; \dots; \boldsymbol{E}_n + \boldsymbol{p}_n]
$$

**「1D 位置编码」看似丢掉了 2D 空间信息，但实验表明足够**——Transformer 能从「注意力模式」里自己恢复 2D 关系。

**CLS token**：在序列开头加一个**分类标记（class token）**$\boldsymbol{E}_{\text{cls}}$（可学习向量）——经过 Transformer 后，**CLS 的最终表示作为「整幅图像的表示」**用于分类（与 BERT 的 `[CLS]` token 完全同构）。

**「CLS token = 图像的『摘要向量』」**——它通过与所有块的注意力，聚合「全局信息」，充当「图像级表示」。<span class="marginnote">「为什么用 CLS token 而不是『平均所有块』」：CLS 是「可学习的聚合器」——它通过注意力「学会」怎么从各块「收集」对分类有用的信息（对比「平均池化」的固定加权）。「<strong>CLS 让『聚合方式』也可学习</strong>」——这是 BERT/ViT 的共同设计（虽然「平均池化」在某些任务上也不差，CLS 是「标准做法」）。</span>

**易错点：** ViT 的位置编码是「**1D**」的（块在「展平后的序列」里的位置）——它不显式编码「2D 坐标」。**「1D 位置 + 注意力自恢复 2D」**是 ViT 的「隐式空间建模」——这个「看似偷懒」的设计在实践中够用，但也让 ViT 对「旋转不变性」等 2D 性质弱于 CNN（Swin 等变体的一部分动机）。

## 3 ViT 的整体架构

**ViT 的完整流程**：

```text
图像 → 切成 16×16 的 patch → 展平 + 线性嵌入 → 拼上 CLS token、加位置编码
      → Transformer Encoder → 取 CLS 的最终表示 → 分类
```

**「图像 → 分块 → 嵌入 → Transformer → 表示」**——与「句子 → 分词 → 嵌入 → Transformer → 表示」**完全同构**。ViT 就是「把图像翻译成 Transformer 的语言」的模型。

**ViT 的变体**：ViT-Base（12 层、768 维）、ViT-Large（24 层、1024 维）、ViT-Huge（32 层、1280 维）——「<strong>规模缩放直接复用 NLP 的 Transformer 配方</strong>」。<span class="marginnote">「ViT 的『规模依赖』」：ViT 论文发现「小数据下 ViT 不如 CNN」（没有卷积先验，学得慢），但「大数据（ImageNet-21k / JFT-300M）下 ViT 反超 CNN」——「<strong>先验的『必要程度』随数据规模下降</strong>」。这个「大数据 = 弱先验」的规律，是 ViT 的核心发现，也解释了「为什么 ViT 适合做『大规模预训练 + 下游微调』」。</span>

**易错点：** ViT 没有「卷积」——它的「归纳偏置」只有「注意力 + 位置编码」。**「ViT 的『偏置少』既是优点（大数据下更灵活）也是缺点（小数据下学得慢）」**——「先验与数据的权衡」是选架构的核心。

## 4 公式解析：ViT 的注意力如何「看」图像

把 ViT 的「块间注意力」写出来，看它如何建模图像关系。设块嵌入矩阵 $\boldsymbol{Z}\in\mathbb{R}^{n\times d}$（$n=196$ 个块），Transformer 的自注意力：

$$
\text{Attention}(\boldsymbol{Z}) = \text{softmax}\Big(\frac{\boldsymbol{Z}\boldsymbol{W}^Q\boldsymbol{W}^{K\top}\boldsymbol{Z}^{\top}}{\sqrt{d_k}}\Big)\boldsymbol{Z}\boldsymbol{W}^V
$$

- **第一步，看「块间交互」**：注意力权重矩阵是 $196\times196$——每对「块-块」都有权重——**「任意两个图像块可以直接交互」**（对比 CNN 的「局部窗口」）。
- **第二步，看「全局感受野」**：第一层注意力就让每个块「看到全图」——**ViT 的第一层就是全局感受野**（对比 CNN 要堆叠才有全局）。
- **第三步，看「动态」**：权重由内容决定——「这个块该看哪些块」由数据学——**「ViT 的『感受野』是动态的，CNN 是静态的」**。<span class="marginnote">「ViT 的『全局注意力』 vs CNN 的『局部卷积』」：CNN 第一层只看 $3\times3$（局部先验），ViT 第一层看全图（无局部先验）——「<strong>ViT 牺牲『局部先验』，换取『全局灵活性』</strong>」。在小数据下，CNN 的「局部先验」是优势（学得快）；大数据下，ViT 的「全局灵活性」是优势（不预设「局部性」）——「<strong>先验与数据规模的博弈</strong>」是视觉架构史的主线。</span>

## 5 ViT vs CNN：一张对比表

| | CNN（ResNet） | ViT |
| --- | --- | --- |
| 基本操作 | 卷积（局部 + 共享） | 注意力（全局 + 动态） |
| 归纳偏置 | 局部性、平移不变 | 极少（只有位置） |
| 感受野 | 逐层扩大 | **第一层全局** |
| 小数据 | 好（先验帮助） | 差（先验少） |
| 大数据 | 好 | **更好**（灵活性） |
| 计算 | 高效（局部） | $O(n^2)$（全局） |

**「ViT 不是『取代』 CNN，而是『补充』一种大数据下的选择」**——小数据用 CNN（先验帮助）、大数据用 ViT（灵活取胜）、以及「混合」（Swin 的层级窗口注意力）——「<strong>架构选择 = 数据规模 + 计算预算的函数</strong>」。<span class="marginnote">「ViT 的『计算复杂度』」：注意力的 $O(n^2)$（$n=196$ 个块）在「图像分辨率高」时爆炸（$n$ 随分辨率平方增长）——「<strong>ViT 的高分辨率处理是难点</strong>」（Swin 用窗口注意力、金字塔结构解决——下一篇）。「$O(n^2)$ 与高分辨率的矛盾」是 ViT 落地高分辨率任务的工程瓶颈。</span>

**易错点：** ViT 的「CLS token」在「分类」外还用于「预训练」（MAE 掩码重建、CLIP 图文对比都用 ViT）——**「ViT 是『视觉骨干』，不限于分类」**。它成为「现代视觉预训练」的通用编码器（MAE、CLIP 的视觉侧都是 ViT）。

## 6 数值算例：patch 大小与序列长度

把「图像分块」的量算出来，理解 ViT 的序列长度由什么决定。设输入 $224\times224$ 图像，patch 大小 $P\times P$：

$$
\text{序列长度} = \frac{224}{P} \times \frac{224}{P} \ \ (+1 \ \text{CLS token})
$$

| patch 大小 $P$ | 序列长度 | 每块嵌入维（$16^2\times3$） | 计算特征 |
| --- | --- | --- | --- |
| $8\times8$ | $28\times28+1=785$ | 192 | 序列长、注意力贵（$O(L^2)$） |
| $16\times16$ | $14\times14+1=197$ | 768 | **ViT-Base 默认** |
| $32\times32$ | $7\times7+1=50$ | 3072 | 序列短、信息被压得粗 |

**patch 越大，序列越短、注意力越便宜，但每块的「局部信息」越粗**——这是一个「序列长度 vs 块内分辨率」的权衡。ViT 选 $16\times16$ 是把「注意力能处理的长度」与「块内保留的细节」折中。<span class="marginnote">关键对比：<strong>CNN 靠「卷积的局部归纳偏置」在像素级滑行，ViT 靠「先 patch 化、再全局注意力」在块级交互</strong>。$16\times16$ patch 意味着 ViT 一开始就「放弃」块内的像素级局部性（交给 MLP 处理），而把注意力预算花在「块与块的全局关系」上。这也是 ViT 需要更大数据预训练的原因——它没有 CNN 的平移/局部先验，得靠数据自己学出来。</span>

**易错点：CLS token 不是「万能池化」。** 分类用「CLS token 的输出」而非常见 CNN 的全局平均池化，是 ViT 的惯例，但它不是唯一选择——把「所有 patch 输出做平均/加权池化」同样可行且常常不差。**CLS token 的角色更像「一个自己会去聚合全局信息的可学习占位符」，而非「天然代表全图」**。现代视觉 Transformer（如 Swin）干脆不用 CLS token、直接池化，见《Swin Transformer》。

## 7 小结

- **ViT**：图像分块（patch）→ 块嵌入 → 位置编码 + CLS token → Transformer 编码器——「图像翻译成 Transformer 的语言」。
- **分块**：$16\times16$ 块 = token；块大小 = 「粒度 vs 计算」的权衡（$O(n^2)$）。
- **CLS token**：可学习的「图像摘要向量」，充当「图像级表示」。
- **全局注意力**：第一层就是全局感受野（对比 CNN 的逐层扩大）——「动态 vs 静态」感受野。
- **数据依赖**：小数据 CNN 好（先验帮助）、大数据 ViT 好（灵活取胜）——「先验与数据的博弈」。
- ViT 是现代视觉预训练（MAE、CLIP）的通用骨干——「视觉的 Transformer 时代」。

在下一节，我们看 ViT 的「层级改良」——用窗口注意力兼顾「全局」与「效率」，这就是 **Swin Transformer：移位窗口与层级结构**。
