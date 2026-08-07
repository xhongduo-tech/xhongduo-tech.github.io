---
title: GoogLeNet：Inception 并行结构
date: 2026-08-07
---

# GoogLeNet：Inception 并行结构

<div class="epigraph">
<p>与其纠结选哪条路，不如四条路都走，再决定谁更重要。</p>
<footer>—— 依据 Inception 思想的通俗表达</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 李沐《动手学深度学习》§6.10 ｜ 2026-08-07</p>
</div>

## 为什么从 GoogLeNet 开始

VGG 用「小核深堆」把深度推向 19 层，但计算量巨大。**GoogLeNet**（Szegedy 等, 2014）走的是一条不同的路：**在「宽度」上做文章**——它发明了 **Inception 块**：在同一层**并行**运行多个不同尺度的卷积（$1\times1$、$3\times3$、$5\times5$）+ 池化，再把结果在通道维**拼接**。这个「多尺度并行」的块，让网络在**不加深太多**的情况下获得更强的多尺度特征表达。

GoogLeNet（又名 Inception-v1）用 22 层结构拿下 2014 年 ImageNet 冠军（top-5 错误率 6.7%，超过 VGG），且**计算量只有 VGG 的约 1/12**——「同样精度，便宜一个数量级」。它的两个关键技巧——**1×1 卷积降维（bottleneck）**与**辅助分类器**——都是「工程智慧」的典范。本节把 Inception 块的并行结构、降维技巧与训练技巧讲透。<span class="marginnote">「GoogLeNet」的命名致敬 LeNet（向 LeCun 的致敬），而「Inception」来自电影《盗梦空间》里的名句「We need to go deeper」——这个名字本身就反映了「更深、更巧」的追求。Inception 系列（v1→v4 + Inception-ResNet）是整个 2014–2016 年代最重要的架构家族之一。</span>

## 1 Inception 块：多尺度并行

**Inception 块**的经典结构（v1）有**四条并行分支**：

| 分支 | 操作 | 作用 |
| --- | --- | --- |
| 1 | $1\times1$ 卷积 | 通道混合、捕捉「点特征」 |
| 2 | $1\times1$ 卷积 + $3\times3$ 卷积 | 降维后捕捉「小尺度局部特征」 |
| 3 | $1\times1$ 卷积 + $5\times5$ 卷积 | 降维后捕捉「大尺度局部特征」 |
| 4 | $3\times3$ 最大池化 + $1\times1$ 卷积 | 池化后做通道混合 |

四条分支的输出在**通道维拼接**，成为下一层的输入。

**设计哲学：与其选「用 3×3 还是 5×5」，不如都用，让网络自己学「哪个尺度更重要」**。每个分支的输出通道数由训练决定——网络会按需分配「权重」给不同尺度。这打破了「每层必须选一种核大小」的硬约束。<span class="marginnote">Inception 的「多尺度并行」与 VGG 的「单一尺度深堆」是两种互补的策略：VGG 赌「小核深堆最优」，Inception 赌「多尺度混合最优」。后来的 ResNet 证明「深度 + 残差」更稳，但 Inception 的「并行多尺度」思想在「多尺度特征融合」（FPN、HRNet）里延续至今。</span>

**易错点：** Inception 的拼接要求各分支输出**空间尺寸一致**——所以所有分支的卷积都用「same 填充」（$1\times1$ 不改变尺寸，$3\times3$ 配 pad 1、$5\times5$ 配 pad 2）。**「尺寸对齐」是并行拼接结构的第一约束**。

## 2 1×1 瓶颈：用降维省计算

Inception 的每条大核分支都先接一个 $1\times1$ 卷积——这是 **1×1 bottleneck（瓶颈）**：先把通道从 $C_{\text{in}}$ 压到一个小数（如 $\frac{C_{\text{in}}}{4}$），做完 $3\times3/5\times5$ 卷积后再（在拼接时）恢复总通道。

**为什么瓶颈能省算力？** 设 $C_{\text{in}}=C_{\text{out}}=C$：

- **直接 $5\times5$ 卷积**：计算量 $\approx 25 C^2 HW$。
- **$1\times1$ 降到 $C/4$ + $5\times5$**：计算量 $\approx C\times\frac{C}{4}HW + 25\times\frac{C}{4}\times\frac{C}{4}HW \approx 1.8 C^2HW$——**省约 14 倍**。

**「先降维、再计算、最后升维」**是深度学习最通用的省算力模式（ResNet 的瓶颈块也用它）。它牺牲一点表达（降维是信息压缩），换取巨大的计算节省。<span class="marginnote">瓶颈的「信息论直觉」：通道维往往高度冗余（相邻通道特征相似），$1\times1$ 降维先做「有损压缩」，把冗余去掉再计算，计算量大幅下降而精度损失很小。这个「先压缩再计算」的模式，在注意力机制的「多头降维投影」里也反复出现（Transformer 的 $d_{\text{model}}\to d_k$）。</span>

**易错点：** 瓶颈的降维比例（压缩到多少）是超参数。压得太狠（如 $C/16$）信息损失过大、精度下降；压得太松（如 $C/2$）省算力有限。Inception 常用 $1/4$ 左右——**「压缩比」要在算力与精度间权衡，由验证集裁决**。

## 3 GoogLeNet 的整体结构：深而巧

GoogLeNet 由 9 个 Inception 块 + 若干辅助结构组成：

```
输入(224x224x3)
→ Stem：卷积(64) + 卷积(192) + 池化
→ Inception(3a) → Inception(3b) → 池化
→ Inception(4a) → Inception(4b) → Inception(4c) → Inception(4d) → Inception(4e) → 池化
→ Inception(5a) → Inception(5b)
→ GAP → Dropout → Softmax(1000)
```

**深度约 22 层，参数约 700 万**（远少于 VGG 的 1.4 亿），计算量约 1.5 GFLOPs（VGG 的约 1/10）。**GoogLeNet 证明「巧设计」可以替代「蛮力加深」**。<span class="marginnote">GoogLeNet 的 9 个 Inception 块「逐渐增宽」：早期块输出 128–256 通道，后期 512–832 通道。这个「由窄到宽」的通道调度，配合「空间逐级减半」，是「空间减半、通道翻倍」法则的又一实现——只是用「并行块」而非「串行小核」来增宽。</span>

**辅助分类器（auxiliary classifiers）**：在网络中部（Inception 4a、4d）各接一个「旁路分类器」（GAP → FC → Softmax）。训练时，主损失 + 两个辅助损失**加权相加**，辅助分类器的梯度帮助「中层」也能收到强监督信号——**缓解梯度消失、加速中浅层收敛**。推断时辅助分类器被丢弃，不影响最终结构。

**易错点：** 辅助分类器是**训练技巧**，不是**网络结构**——它只在训练时存在，推理时删掉。它的作用是「给中层梯度」，而非「参与最终预测」。现代网络（ResNet 之后）用残差连接解决梯度问题，辅助分类器便不再必要。

## 4 公式解析：Inception 块的计算量对比

把「并行多分支」与「串行大核」的计算量精确对比。设输入通道 $C$、输出总通道 $C_{\text{out}}$、特征图 $H\times W$，Inception 块的四个分支各自输出 $C_{\text{out}}/4$ 通道：

- **分支 2（1×1 + 3×3）**：$1\times1$ 把 $C\to C/4$，$3\times3$ 把 $C/4\to C_{\text{out}}/4$。FLOPs $\approx C\times\frac{C}{4}HW + 9\times\frac{C}{4}\times\frac{C_{\text{out}}}{4}HW$。
- **对比单层 5×5 直接卷积**（$C\to C_{\text{out}}$）：FLOPs $\approx 25\times C\times C_{\text{out}}\times HW$。

- **第一步，代入数值**：设 $C=C_{\text{out}}=512$、$HW=56^2$。Inception 分支 2 的 FLOPs 约 1.03 GFLOPs；单层 $5\times5$ 约 20.1 GFLOPs。
- **第二步，看差距**：Inception 用一个 $1\times1$ 的「前置压缩」，把 5×5 分支的计算量降了一个数量级。
- **第三步，读整体**：四条并行分支的总计算量仍远小于「每层一个 5×5」的串行方案——**「并行 + 降维」是 GoogLeNet 省算力的双重来源**。<span class="marginnote">「计算量预算」是架构设计的隐形约束：同一精度下，GoogLeNet 用 1.5 GFLOPs、VGG 用 15+ GFLOPs——这意味着 GoogLeNet 可以在更便宜的硬件上实时运行。Inception 系列后续版本（v2/v3）的改进（BatchNorm、因子分解）也大多在「更省算力的同时更准」这条线上。</span>

## 5 从 Inception 到现代多尺度设计

Inception 的「多尺度并行」思想在后续架构中以不同形态延续：

- **Inception v2/v3**：把 $5\times5$ 分解成两个 $3\times3$、把 $n\times n$ 分解成 $1\times n + n\times 1$——「大核分解」进一步省算力；引入 BatchNorm。
- **FPN（特征金字塔）**：在检测网络里做「多尺度特征融合」——不同层的特征图并行/融合，是「多尺度」思想在「跨层」上的延续。
- **HRNet**：全程并行多分辨率分支——Inception 的「并行」哲学的最彻底实现。

**「Inception 教会我们的不是某个具体块，而是『并行多尺度』这个设计维度」**——当一层不知道该用什么尺度时，并行地都用、让数据决定权重。<span class="marginnote">「并行让数据选择」的思想甚至超越了卷积：Transformer 的多头注意力本质上也是「并行多个注意力子空间，让网络自己分配重要性」——「并行 + 融合」是现代神经网络最通用的「选择机制」。从这个角度看，Inception 与多头注意力是同一设计哲学在不同时代的两个实例。</span>

## 6 小结

- **Inception 块**：四条并行分支（1×1 / 1×1+3×3 / 1×1+5×5 / 池化+1×1），通道维拼接——多尺度并行、让数据选尺度。
- **1×1 瓶颈**：先降通道再计算，省算力约一个数量级；「先压缩再计算」的通用模式。
- GoogLeNet：22 层、700 万参数、1.5 GFLOPs——「巧设计」替代「蛮力加深」。
- **辅助分类器**：训练时给中层额外梯度，推理时丢弃——纯训练技巧。
- 并行要求分支**空间尺寸对齐**（same 填充）。
- 遗产：并行多尺度 → FPN/HRNet/多头注意力的设计先声。

在下一节，我们进入 CNN 史上最重要的一次架构革命——它用一个「跳跃连接」把可训练深度从 19 层推到 152 层，这就是 **ResNet：残差连接与恒等映射**。
