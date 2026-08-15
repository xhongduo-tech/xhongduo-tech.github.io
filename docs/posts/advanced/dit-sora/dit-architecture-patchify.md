---
title: DiT 架构与 Patch 化
date: 2026-08-07
---

# DiT 架构与 Patch 化

<div class="epigraph">
<p>一切应当尽可能简单，但不能更简单（Everything should be made as simple as possible, but not simpler）。</p>
<footer>—— 爱因斯坦（Albert Einstein，流传名言）</footer>
</div>

<div class="article-byline">
<p>第四级 · DiT / Sora（论文解析） ｜ Peebles & Xie, Scalable Diffusion Models with Transformers (2023) §3 ｜ 2026-08-07</p>
</div>

## 为什么从「换掉 U-Net」开始

上一篇我们拿到了扩散引擎 DDPM，但引擎的「缸体」还没定型：去噪器 $\epsilon_\theta$ 到底用什么网络？在 DiT 出现之前，主流答案几乎是唯一的——**U-Net**。它凭跳跃连接和尺度金字塔统治了扩散生成（Stable Diffusion 的骨干就是它）。2023 年，William Peebles 与 Saining Xie 在论文 *Scalable Diffusion Models with Transformers\* 中提出了一个尖锐的问题：**如果扩散模型的去噪器不是 U-Net，而是 Transformer，会怎样？**

结果出乎很多人意料：更好，而且**更可缩放**。DiT（Diffusion Transformer，扩散 Transformer）把 ViT 的 token 化思想搬进扩散生成，在 ImageNet 上刷新了当时的 SOTA，也为一年后的 Sora 铺平了道路。本文是 DiT 的上篇：讲清它的骨架——**Patch 化（patchify）**。

## 1 从 U-Net 到 Transformer：一次「降维」式的简化

换个角度看这次替换：U-Net 的跳跃连接在「编码器与解码器之间传信息」，而 Transformer 的注意力在整个序列上传信息；前者是图像的专属语法，后者是通用的序列语法。DiT 赌的是——当数据与算力足够大时，通用语法比专属语法更值得投入。这个赌注在图像上赢了，在视频上（Sora）也赢了，因为它换来了一样 U-Net 给不了的东西：**scaling 的可预测性**——这正是本专题第四篇的主题。

U-Net 的复杂来自它对图像先验的执着：下采样编码、上采样解码、每层都有跳跃连接——它把「图像是空间结构」这一假设焊死在网络里。Transformer 恰好相反：它**拒绝任何图像先验**，只接受一个「token 序列」，把空间关系全部交给注意力机制去学。<span class="marginnote">这正是 ViT（Dosovitskiy 等，2021）在图像分类上的思路：把图像切成 patch 当词喂给 Transformer。DiT 把 ViT 的这步「token 化」原样移植进扩散去噪器——详见第三级《深度学习基础》中关于 Transformer 与注意力机制的章节。</span>

DiT 的论文写得非常坦诚：它不发明新组件，而是把 Latent Diffusion（LDM，Stable Diffusion 的底座）里那个 U-Net 整个换成标准 ViT 式的 Transformer，然后证明——**在同样数据、同样训练量下，Transformer 拿下了更好的 FID，且模型越大优势越明显。**

U-Net 的代价常被低估：它在 ImageNet 上刷出好成绩时，靠的是「针对图像定制的结构」，而不是「一套通用骨架」。同一套 U-Net 换到视频、换到三维医学图像，骨干几乎要重新设计；而 Transformer 的 token 化抽象让「图像 → 视频 → 多模态」只需换输入表示、骨干原封不动。所以 DiT 的价值不只是「FID 更低」，更是**把「骨干」变成了一件可跨任务复用的资产**——这正是后来 Sora 能直接继承 DiT 组件的前提。

这次替换还带来一份「训练稳定性」的意外红利：U-Net 深层的跳跃连接与注意力通常需要仔细的初始化与学习率调度，而标准 Transformer 块配合条件机制，训练曲线更平稳。DiT 论文专门报告「无需学习率预热与强正则化也能稳定收敛」——对依赖玄学的扩散训练来说，这本身就是可观的工程收益。

DiT 论文里 U-Net 与 Transformer 的对比不是「单点刷分」，而是「同一训练协议下的系统比较」：两边都在 ImageNet 256×256、同样的步数与批大小下训练，公平地比出 Transformer 的优势。这种「同协议、只改一个变量」的实验设计，是论文结论可信的根基，也值得我们在读任何对比实验时留意。

## 2 Patch 化：把 latent 切成 token

DiT 的输入不是像素，而是上一篇末尾提到的 latent：一张 $256 \times 256$ 图像经 VAE 编码后是 $32 \times 32 \times 4$ 的张量 $z$。**Patch 化**的第一步，就是把这张 $32 \times 32$ 的「特征图」切成 $p \times p$ 的小块；每个块被一个线性层压成一个向量（相当于逐块做一次矩阵乘法），再与位置编码相加，得到一条 token 序列：

$$
N = \left(\frac{h}{p}\right)^2 \quad\Longrightarrow\quad h = 32,\ p = 2 \Rightarrow N = 256 \text{ 个 token}
$$

DiT 考察了 $p \in \{8, 4, 2\}$ 三种块大小，并用「/p」标注配置：DiT-XL/2 就是 patch size 为 2 的 XL 版。**块越小，token 越多**：$p=8$ 只有 16 个 token，$p=2$ 则有 256 个——计算量与表达能力在此权衡。

把一个具体的图像走一遍：$256 \times 256$ 的 RGB 图像先经 LDM 的 VAE 编码，变成 $32 \times 32 \times 4$ 的 latent；若 patch size 取 2，则得到 $(32/2)^2 = 256$ 个 $2 \times 2 \times 4$ 的 patch，每个被线性层压成一个 $d$ 维向量。论文用「/2」「/4」「/8」标注配置，DiT-XL/2 便是在此基础上堆 28 层块、宽 1152 维的「大号」模型——这些规格在第四篇的缩放表格里会一一列出。

这条 token 序列就是 Transformer 的全部输入。它和 GPT 里的文本 token、ViT 里的图像 patch 是同一种东西：**一个语义碎片、一个位置编号、一条向量**。位置编号这一步不能省：token 序列被丢进注意力时，每个 token 没有任何空间坐标感——如果交换两块 patch 的位置，自注意力根本察觉不到。DiT 沿用 ViT 的可学习位置嵌入，让网络知道「这个 token 在图像左上还是右下」。值得注意的是，这种「位置全靠学」的设计正是 Transformer 拒绝图像先验的体现：它不写死平移不变性，而是让模型自己发现空间关系。此外 DiT 还像 ViT 一样额外拼接一个可学习的类别 token（类似 BERT 的 `[CLS]`），用于承载类别信息——这一点在下一篇《条件控制：AdaLN-Zero》里会展开。

Patch 嵌入的细节也值得拆一下：每个 $p \times p \times c$ 的 patch 被展开成长度 $p^2 c$ 的向量，再乘一个 $p^2 c \times d$ 的可学习矩阵，映射到 $d$ 维空间。这一步本质上是**把「局部像素/通道的排列」压缩成「一个语义向量」**，与词嵌入把「离散 token 映射成向量」在功能上完全平行。DiT 论文里 patch 嵌入矩阵就是「随机初始化、随训练更新」，没有花哨的技巧——但正是这一步，把「图像先验」彻底挡在门外：进入 Transformer 之前，网络眼里只有向量，没有「上」「下」「左」「右」。

「patch 大小」因此成了 DiT 手上一个横跨「表示」与「算力」的旋钮：$p=2$ 时序列长、每 token 细；$p=8$ 时序列短、每 token 粗。Transformer 对「长而细」与「短而粗」两种输入都能处理，但注意力的代价对前者更敏感——这也解释了为什么 DiT 论文用「/p」来命名模型：patch 是它区别于普通 ViT 的第一个标识。

## 3 DiT 块与四种条件注入方式

Patch 化之后，主体是 **L 个堆叠的 DiT 块**。每个块就是标准 Transformer 块：一个自注意力层加一个 MLP，外面套残差连接。真正有讲究的是**条件怎么进块**。扩散模型的去噪器必须知道「当前是第几步」，还要能接收类别、文本等条件，于是 DiT 比较了四种注入方式：

| 方式 | 做法 | DiT-XL/2 FID（越低越好） |
| --- | --- | --- |
| In-context | 把时间步与类别嵌入拼进 token 序列 | 2.55 |
| Cross-attention | 用交叉注意力让 token 去查条件 | 2.57 |
| AdaLN | 用条件回归 LayerNorm 的缩放与偏移 | 2.34 |
| **AdaLN-Zero** | AdaLN 之上再加残差门控，零初始化 | **2.27** |

先看块内的常规部件：自注意力与标准 Transformer 完全一致——每个 token 对整条序列做 QKV 注意力，输出再经残差与 LayerNorm 归位；MLP 提供逐 token 的非线性变换。真正让 DiT 块区别于 ViT 块的，是条件注入的接口——下一篇展开的 AdaLN-Zero 就藏在「残差、归一化、条件调制」这三件事的咬合处。

结论很清楚：**AdaLN-Zero 最优**，而且它只比普通 AdaLN 多一个线性层、几乎不增加参数。这也是 DiT 论文最重要的工程结论之一——我们把它的原理留给下一篇详细拆解。<span class="marginnote">FID（Fréchet Inception Distance）是生成质量的标准指标，数值越低表示生成分布与真实分布越接近；其数学定义见第三级《生成模型》专题。</span>

读这张表时注意一个前提：四种方案都在 DiT-XL/2 同一规模下比较，差距主要来自「条件信息进网络的深度与位置」。In-context 与 Cross-attention 都把条件当作「序列的一部分」或「独立的查询来源」，代价是注意力预算被条件占掉一部分；AdaLN 系列则让条件「绕过注意力」、直接调制每一层的统计量，几乎不占注意力预算——**这是 AdaLN 胜出的结构性原因，而不只是巧合**。它给我们一个可迁移的默认选择：在注意力越来越贵的时代，「把条件放进归一化，而不是放进序列」，通常更划算。

## 4 公式解析：token 数与注意力的二次代价

**Patch 化这一步看似平凡，却决定了整个模型的算力账单。** 拆解 token 数公式：

$$
N = \left(\frac{h}{p}\right)^2, \qquad
\text{注意力成本} = O\left(N^2 d\right)
$$

- **第一步，看 $N$ 随 $p$ 的变化**：latent 边长 $h$ 固定时，$p$ 每减半，token 数变成原来的 4 倍（$p=8 \to N=16$，$p=2 \to N=256$）。
- **第二步，看注意力对 $N$ 的依赖**：每个 token 要与序列里所有其他 token 计算注意力分数，所以注意力层的计算量正比于 $N^2$（再乘向量维数 $d$）。
- **第三步，把两步合起来**：$p$ 从 8 减到 2，token 数 ×16，注意力计算量 ×256。这就是为什么 DiT 论文里 **$p=8$ 是「每单位计算量最划算」的选择**——少切块、少 token，同等算力下能塞进更大的模型。

**选择 patch size，本质是在「表达的粒度」与「算力的预算」之间做权衡。** 块越小信息越精细，但注意力二次膨胀；块越大越省算力，却损失空间细节。

再追问一步：token 数减少这么多，信息真的没丢吗？注意力的二次代价省下来了，但每个 token 必须承载更大的空间——$p=8$ 时一个 patch 覆盖 $8 \times 8 \times 4$ 的原始 latent 区域，细节全靠高维向量和网络容量去弥补。所以「选哪个 $p$」没有绝对答案，它依赖数据规模与算力预算；DiT 的经验是，在 ImageNet 这类中等规模数据上，$p=8$ 的单位算力性价比最高，而更大的模型会更倾向较小的 patch 以换取细节。

一个直观的算力对比：假设 $d=384$、$L=12$，$p=8$ 时 16 个 token 的注意力计算量约在百万级浮点，$p=2$ 时 256 个 token 则升到亿级——差了约两个数量级。这正是论文里 DiT-S/2 与 DiT-S/8 前向 Gflops 相差巨大的原因，也呼应第四篇「用 Gflops 度量模型」的选择：**只有用计算量说话，patch 大小造成的量级差距才不会被参数量掩盖。**

从「patch 是旋钮」再往前推一步，就能看到 DiT 缩放实验的第一个结论为什么成立：既然 $p$ 同时调节序列长度与 token 粒度，那么在固定算力下存在一个「最优 $p$」——$p$ 太大丢细节、$p$ 太小注意力爆表，中间的甜点随模型规模移动。DiT 的缩放表格里同一模型用不同 $p$ 的 FID 差异，正是这个「最优 $p$」随规模漂移的证据；这也是第四篇「patch size 是缩放策略的一部分」这句话的来历。

## 5 从图像到视频的伏笔

DiT 的图像版已经走完了「Patch 化 → Transformer → 去噪」的闭环，它的整条流水线是这样的：

![DiT 图像生成流水线](/images/dit-sora/dit-architecture-patchify-1.svg)

解码端也值得一提：堆叠的块之后接一个最终的 LayerNorm 与线性层，把每个 token 的 $d$ 维向量映射回 $p^2 c$ 维（一个 patch 的原始通道数），再按 patch 网格拼回 $32 \times 32 \times 4$ 的 latent——这一步叫 **unpatchify**，是 patchify 的逆操作。整个网络从 latent 进、latent 出，像素编解码完全交给 VAE，这正是「latent 扩散」的设计精髓：**扩散模型只负责在抽象空间里去噪，具体的像素现实由另一套组件兜底。**

还有一处容易误读：unpatchify 不是简单地把向量「拼回」原 latent——它在每个 token 上做线性映射，把 $d$ 维还原成 $p^2 c$ 维的 patch 内容，再按网格拼合。这意味着解码端也有可学习参数，「patch 化的逆操作」并非严格无损的拼图，而是「从语义向量再生成局部内容」的一步。理解这点，就不会把「patch 化」错当成「压缩」：压缩在 VAE，patch 化只是「重新组织表示」。

值得注意的伏笔：**这套「切 patch 喂 Transformer」的逻辑天然可以推广到视频。** 图像是 $h \times w$ 的二维网格，视频则是 $t \times h \times w$ 的三维网格——只要把「切块」从 2D 变成 3D，视频就变成了一条更长的 token 序列。这正是 Sora 的时空 patch（本专题第六篇）做的第一件事。DiT 的另一个伏笔是条件机制：它证明了「条件的注入方式」可以成为可实验、可优化的设计维度，AdaLN-Zero 则为 Sora 的视频扩散 Transformer 提供了现成的方案。

回到 Sora 的视角，这条流水线的每个环节都能平移：patchify 变成时空 patchify，块堆叠变成 L 个 DiT 块处理三维 token，unpatchify 变成从视频 latent 还原。甚至「无条件→条件」的路线也一致：图像 DiT 的类别条件，在 Sora 里扩展成文本条件。可以说 **Sora 的架构蓝图，就写在这篇 DiT 论文里**——剩下要补的，只有「如何把二维 patch 变成三维时空 patch」这一块拼图。

再补充一处图像与视频的工程差异：图像 DiT 的 latent 是固定的 $32 \times 32$，patch 数与输入尺寸一一绑定；视频里 latent 的三维网格随时长与分辨率浮动，patch 数随之变化。这意味着视频 DiT 无法像图像那样「预先算好注意力规模」，而必须处理变长序列——这也是 Sora 把分块、序列并行等训练策略写进工程文档的原因。**「表示统一」让架构可以复用，「序列变长」让工程必须跟进**——读 DiT 时记住这两句，读 Sora 时就不会被工程细节绕晕。

## 6 小结

- DiT 的核心决定：把扩散去噪器的骨干从 U-Net 换成**标准 Transformer**，拒绝图像先验、拥抱 token。
- **Patch 化**把 latent 切成 $p \times p$ 块并线性嵌入，token 数 $N = (h/p)^2$；DiT-XL/2 即 patch size 2。
- 条件注入有四种方案，**AdaLN-Zero 在 FID 上全面胜出**（2.27），且几乎不增加参数。
- token 数的选择是「粒度 vs 算力」的权衡：注意力成本 $O(N^2 d)$，$p$ 每减半计算量 ×4。
- 图像 patch 化自然延伸到视频：三维网格切 3D patch，就是 Sora 的表示基础。
- 四种条件注入的比较说明：条件「占注意力预算」还是「调归一化」，是决定 FID 的结构性因素；AdaLN-Zero 的「零初始化 + 门控」让训练从恒等映射起步。
- Transformer 的 token 化把「骨干」变成可跨任务复用资产：图像、视频、多模态只换输入表示、不动骨架——这是 Sora 能继承 DiT 的结构性原因。
- 「同协议、只改一个变量」的实验设计是 DiT 结论可信的根基：U-Net 与 Transformer 在同一训练协议下公平对比，才让「Transformer 更好」成为可复现的事实。

在下一节，我们将深入那个真正让 DiT 发光的组件：**AdaLN-Zero——一种零初始化的自适应层归一化条件机制。**
