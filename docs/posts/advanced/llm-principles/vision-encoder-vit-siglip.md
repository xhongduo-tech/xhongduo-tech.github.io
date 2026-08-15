---
title: 视觉编码器：ViT 与 SigLIP
date: 2026-08-07
---

# 视觉编码器：ViT 与 SigLIP

<div class="epigraph">
<p>模型的「眼睛」，是一堆 patch 在 Transformer 里的互相关注。</p>
<footer>—— 视觉模型谚语（化用）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型原理 ｜ Dosovitskiy et al. 2020《ViT》 / Zhai et al. 2023《SigLIP》 ｜ 2026-08-07</p>
</div>

## 为什么「视觉编码器」决定模型的「眼神」

三段式的第一段——视觉编码器——把图像变成「视觉 token」。它的质量直接决定模型「能看见多细」：patch 太小算力爆炸，patch 太大看不清细节；特征「语义化」程度决定「读懂」还是「看懂」。主流视觉编码器是 **ViT**（Vision Transformer），而 VLM 里常用 **CLIP/SigLIP 预训练的 ViT**——因为它们的特征「语义对齐」更好。<span class="marginnote">「视觉编码器」这个名字揭示了它的角色：<strong>编码（compress）视觉信息为「语义向量」</strong>。ViT 把「图像」当作「patch 序列」用 Transformer 处理——与语言模型的「token 序列」异曲同工。这使 ViT 与 LLM 的「序列世界观」天然契合——也是它能当「VLM 眼睛」的结构原因。</span>

## 1 ViT：把图像当「句子」读

**ViT（Vision Transformer）** 的核心思想：把图像切成 patch，每个 patch 相当于「一个视觉词」，然后像处理文本一样用 Transformer 处理 patch 序列。

**流程**：

1. **切 patch**：图像 $I \in \mathbb{R}^{H \times W \times 3}$ 切成 $P \times P$ 的 patch。224×224 图像、patch=16 → 14×14 = 196 个 patch。
2. **展平 + 投影**：每个 patch 展平成 $P^2 \times 3$ 维向量，线性投影到 $d$ 维——得到「patch 嵌入」。
3. **加位置编码**：patch 有空间位置，加可学习位置编码。
4. **Transformer 编码**：patch 序列过标准 Transformer——patch 之间互相「关注」，融合全局信息。
5. **输出**：每 patch 的编码向量 → 视觉 token 序列。

**关键点**：ViT 用「自注意力」让 patch 之间交互——「这个 patch 的语义」由「它与其他 patch 的关系」决定。这与语言模型「词义由上下文决定」完全同构。<span class="marginnote">ViT 的「革命性」在于「抛弃卷积」：它把图像「翻译」成了「序列」，直接用 Transformer 处理。这带来一个好处——<strong>ViT 与语言模型「同构」，可以无缝地当 VLM 的眼睛</strong>。代价是「局部归纳偏置」没了（卷积天然擅长局部），需要大量数据弥补——这也是 ViT 需要「大数据预训练」的原因。</span>

## 2 CLIP：让视觉特征「懂语义」

**CLIP（Contrastive Language-Image Pre-training）** 的贡献：让视觉编码器的特征**语义对齐**——「图像特征」与「文本特征」在同一个空间里可比。

- 训练：图像与「描述它的文本」配成对（如图文对「猫的照片」+「a cat」）。
- 目标：**对比学习**——让「匹配的图文对」特征相似度高，「不匹配的」相似度低。
- 结果：图像编码器学到的特征「带有语义」——「猫的 patch 特征」与「文本『cat』的嵌入」相近。

**为什么 VLM 需要 CLIP 式的编码器**：VLM 要把视觉特征「翻译」给 LLM——如果视觉特征本身「语义化」（与文本空间近），翻译就轻松；如果只是「低级视觉特征」（颜色、边缘），LLM 无从理解。**CLIP 预训练让视觉特征「接近语义」，是「对齐」的加速器**。

## 3 SigLIP：CLIP 的简化与增强

**SigLIP（Sigmoid Loss for Language Image Pre-training）** 是 CLIP 的改进版：

**对比学习的目标差异**：

- **CLIP**：用 **softmax 对比损失**——在一个 batch 内「归一化」，图文对「互相竞争」。
- **SigLIP**：用 **sigmoid 逐对损失**——每对图文独立计算「匹配/不匹配」的二分类损失。

$$
\mathcal{L}_{\text{SigLIP}} = -\log \sigma(t \cdot \langle f_I, f_T \rangle - b) - \sum_{j \neq i} \log \sigma(b - t \cdot \langle f_I, f_T \rangle)
$$

**SigLIP 的优势**：

- **不需要大 batch**：softmax 对比损失依赖「大 batch」提供负样本；sigmoid 逐对损失每对独立——**batch 大小不影响负样本质量**。
- **训练更稳**：损失形态更简单，收敛更好。
- **效果**：在同样数据下，SigLIP 的视觉特征质量与 CLIP 相当或更好。

**SigLIP 成为 VLM 标配**：LLaVA-1.6、Qwen-VL、Gemini 等都用 SigLIP 的 ViT 作为视觉编码器。<span class="marginnote">SigLIP 的「每对独立」是一个工程细节的胜利：CLIP 的对比损失需要「一个 batch 里所有图文对」一起算（显存与 batch 强相关）；SigLIP 把损失拆成「逐对」，<strong>训练不受 batch 限制、实现更简单</strong>。这体现了「损失函数的形态」对「可训练性」的影响——和「混合精度」是同类问题。</span>

## 4 公式解析：CLIP 的对比损失

CLIP 的对比学习核心：给定一个 batch 的图文对 $(I_i, T_i)$，最大化「匹配对」的相似度、最小化「不匹配对」。归一化后，图像与文本特征的相似度矩阵 $S_{ij} = \langle \hat{f}_{I_i}, \hat{f}_{T_j} \rangle$：

$$
\mathcal{L}_{\text{CLIP}} = -\frac{1}{2}\left(\log \frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}} + \log \frac{e^{S_{ii}}}{\sum_j e^{S_{ji}}}\right)
$$

对这条式子做三步拆解：

- **第一步，读懂 $S_{ij}$**：第 $i$ 张图与第 $j$ 段文本的余弦相似度。对角线 $S_{ii}$ 是「匹配对」，非对角线是「不匹配对」。
- **第二步，读懂 softmax**：对第 $i$ 行做 softmax——「第 $i$ 张图最该配哪段文本」。让「正确的那段」（$S_{ii}$）概率最高——**这行 softmax 让图像「找到自己的文本」**。
- **第三步，读懂双向**：两个 log 项分别从「图像视角」（行）与「文本视角」（列）算——**图文互找**。对比学习让「匹配对」在特征空间「靠近」——这就是「语义对齐」的机制。

**辨析｜易错点：** CLIP/SigLIP 的「文本编码器」也是一个 Transformer——VLM 里通常**不用** CLIP 的文本编码器（LLM 自己有更强的文本能力），只用它的**视觉编码器**。别把「CLIP 模型」与「CLIP 的 ViT 部分」混为一谈——VLM 借的是「眼睛」，不是「全文」。

## 5 视觉编码器在 VLM 里的「冻结与否」

- **冻结**：早期 VLM 冻结视觉编码器（省算力），但「眼神」定型——看不到训练数据里的「新细节」。
- **部分微调**：LoRA 视觉编码器——主流折中。
- **全量微调**：让视觉编码器「适应 VLM 的任务」——最强但最贵（且可能损坏预训练特征）。

**趋势**：从「完全冻结」到「部分微调」——因为「视觉-语言对齐」需要视觉编码器「学到语言相关的细节」（如「看数字」「看箭头」），冻结的编码器可能不够。

## 6 术语速查表

| 术语 | 英文 | 一句话定义 |
| --- | --- | --- |
| ViT | vision transformer | 把图像当 patch 序列处理的 Transformer |
| patch | patch | 图像切成的方块单元 |
| CLIP | CLIP | 对比图文预训练方法 |
| SigLIP | SigLIP | sigmoid 损失的图文预训练方法 |
| 视觉 token | vision token | patch 编码出的特征向量 |
| 冻结 | freezing | 训练时不更新视觉编码器参数 |

## 7 数值算例：分辨率与视觉 token 数

视觉 token 数 = patch 网格数 = $(H/P) \times (W/P)$：

| 图像尺寸 | patch 大小 | 视觉 token 数 |
| --- | --- | --- |
| 224×224 | 16 | 14×14 = 196 |
| 224×224 | 14 | 16×16 = 256 |
| 448×448 | 14 | 32×32 = 1024 |
| 672×672 | 14 | 48×48 = 2304 |

**读这张表**：patch 越小、图像越大，视觉 token 越多——LLM 的输入序列就越长。视觉 token 是 LLM 计算的「大头」（每个 token 都要过全部 Transformer 层），所以 VLM 要在「看得清」（token 多）与「算得起」（token 少）之间权衡。这也解释了为什么「动态分辨率」与「视觉下采样」成为 VLM 的标配组件。

**辨析｜易错点：** patch 越小并不总是越好。patch=8 时 token 数翻 4 倍，算力爆炸；patch 过大（如 32）又看不清细节。**「分辨率」是图像本身、patch 是切法**——提高有效分辨率也可以靠「动态切块」（第十二篇 Qwen-VL），而不必全局降 patch。

## 8 视觉编码器选型速查

| 编码器 | 特点 | 代表 VLM |
| --- | --- | --- |
| CLIP-ViT-L/14 | 经典语义对齐 | LLaVA-1.5 |
| SigLIP-ViT-SO400M | 免大 batch、稳定 | LLaVA-1.6、Qwen-VL |
| 高分辨率 SigLIP | 细节保真 | 文档/图表类模型 |
| 原生多模态编码器 | 统一理解生成 | 原生多模态（末篇） |

选型三原则：**语义对齐质量**（决定「看懂」）、**分辨率策略**（决定「看清」）、**冻结策略**（决定「能不能学新细节」）。视觉编码器是 VLM 的「眼睛」，选错眼睛，后面一切翻译都白搭。

## 9 小结

- **ViT**：把图像切成 patch、当「序列」用 Transformer 处理——与 LLM 同构。
- **CLIP**：对比学习让视觉特征「语义对齐」——「猫的 patch」与「文本 cat」相近。
- **SigLIP**：sigmoid 逐对损失，不依赖大 batch、训练更稳——VLM 标配。
- 对比损失是「图文互找」的 softmax 分类——对齐的机制。
- 视觉编码器「冻结与否」是训练权衡——趋势是「部分微调」。

在下一节，我们讲「对齐」的预训练基础——**视觉-语言对齐：CLIP 的对比学习目标**（从编码器视角补全对齐的完整画面）。
