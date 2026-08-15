---
title: RMSNorm 与 SwiGLU
date: 2026-08-07
---

# RMSNorm 与 SwiGLU

<div class="epigraph">
<p>我们以 RMSNorm 替代 LayerNorm，以 SwiGLU 激活函数替代 ReLU——用更少的计算获得同等的表达能力。</p>
<footer>—— Hugo Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023) §2.2</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型预训练 ｜ LLaMA 2023 §2.2; Zhang & Sennrich 2019; Shazeer 2020 ｜ 2026-08-07</p>
</div>

## 为什么从 RMSNorm 与 SwiGLU 开始

构建一个深层 Transformer，每个子层都要做两件事：**归一化**（让训练稳定）与**非线性变换**（给网络表达能力）。BERT/GPT-1 时代的标准配方是 LayerNorm + GELU。但预训练一放大到 70B 参数，两件小事变得无比昂贵：层归一化里**减均值算均值**在每个 token 上都要算一次，激活函数的选择直接影响参数与 FLOPs。<span class="marginnote">LLaMA 的哲学是「省下来的每一分算力都值得」——一个看似微小的激活函数改动，在千亿级 token 的训练里可能省下整个集群一周的 GPU 时间。</span>

**RMSNorm（Root Mean Square Layer Normalization）**：LayerNorm 的简化变体，只做缩放、去掉均值中心化，用均方根（RMS）做归一化因子。**SwiGLU 激活函数**：把 GLU（门控线性单元）与 SiLU（Swish）结合的 gated 激活，用「门控」让网络能更灵活地控制信息流。两者共同构成 LLaMA 风格 Transformer 的标准件。

## 1 从 LayerNorm 到 RMSNorm

标准 LayerNorm 对一个向量 $x$ 做的是：

$$ \mathrm{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta, \qquad \mu = \frac{1}{d}\sum_i x_i, \quad \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2 $$

而 RMSNorm 去掉均值中心化，直接：

$$ \mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\mathrm{RMS}(x) + \epsilon}} \cdot \gamma, \qquad \mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2} $$

**RMSNorm 去掉了 $\mu$（减均值）和 $\beta$（bias）**，只剩缩放因子 $\gamma$。为什么这么做仍然有效？<span class="marginnote">Zhang & Sennrich (2019) 的观测：LN 的平移不变性（均值）在 Transformer 里对残差流的作用有限，而<strong>缩放不变性</strong>才是稳定训练的关键——RMSNorm 保留了后者、舍弃了前者。</span>

### 公式解析：RMSNorm 为何能「省」

- **第一步，看计算量**：LayerNorm 要算 $\mu$ 和 $\sigma^2$ 两次求和；RMSNorm 只算 $\sum x_i^2$ 一次。且**省去了均值中心的减法**，反向传播少了一整条路径。
- **第二步，看信息损失**：$\mu$ 在残差连接中可以被后续层学回来——残差流本身会携带均值信息，所以去掉它损失不大。
- **第三步，工程落地**：现代实现（如 Triton kernel）把归一化「融合」进前向，RMSNorm 的 kernel 更简单、显存占用更小。**收益不来自数学奇迹，而来自「少做一件不那么必要的事」**。

**数值算例**：设某个 token 的向量是 $x = (1.0, 2.0, 3.0)$，$d = 3$。

- LayerNorm 先算 $\mu = (1+2+3)/3 = 2.0$，再算 $\sigma^2 = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3$，于是中心化后的向量是 $(-1, 0, 1)$，再除以 $\sqrt{2/3} \approx 0.816$ 得到 $\approx (-1.22, 0, 1.22)$。
- RMSNorm 不算均值，直接算 $\mathrm{RMS} = \sqrt{(1+4+9)/3} = \sqrt{14/3} \approx 2.16$，归一化后是 $(0.46, 0.93, 1.39)$。

两者输出的**形状相似**——都让向量「回到单位量级」，只是 RMSNorm 不做去均值。在残差流里，均值信息本来就会被后面学回来，所以这一步省得理直气壮。

## 2 从 ReLU 到 SwiGLU

激活函数给线性变换带来非线性。LLaMA 用 SwiGLU 替代了 GPT 系列的 GELU 前馈网络。

先看 **SiLU（Swish）**：$\mathrm{SiLU}(x) = x \cdot \sigma(x)$，其中 $\sigma$ 是 sigmoid。它是一条平滑的「先缓后陡」曲线，没有 ReLU 在 0 点的折角，梯度更温和。

再叠上 **GLU（门控线性单元）**：GLU 把输入拆成两支，一支做线性变换、一支过 sigmoid 当「门」：

$$ \mathrm{GLU}(x, W, V, b) = \sigma(xW + b) \otimes (xV) $$

$\otimes$ 是逐元素乘积，$\sigma(xW+b)$ 充当「开关」：门值接近 1 时放行信息、接近 0 时屏蔽。<span class="marginnote">门控的直觉来自 LSTM 的遗忘门：网络可以学会「某些维度此刻该关闭」。GLU 把这种「可学习的开关」内建进前馈层，表达能力比普通逐元素激活更强。</span>

**SwiGLU** 就是把门从 sigmoid 换成 SiLU（两者形状相近，但 SiLU 对负值有「软渗透」）：

$$ \mathrm{SwiGLU}(x, W, V, b) = \mathrm{SiLU}(xW + b) \otimes (xV) = (xW + b) \cdot \sigma(xW + b) \otimes (xV) $$

### 公式解析：SwiGLU 为什么「更强且更省」

- **第一步，拆解结构**：输入 $x$ 走两条支路——支路一 $xW$ 过 SiLU 变「门控权重」，支路二 $xV$ 是「内容」；两者逐元素相乘。
- **第二步，表达能力**：双支路让网络在每一维上都可以独立决定「放行多少」，相当于把 ReLU 的「硬开关」升级成「连续旋钮」。Shazeer (2020) 的实验证明，相同参数量下 GLU 族激活的困惑度低于 ReLU 族。
- **第三步，省算力的代价**：SwiGLU 需要两个矩阵 $W, V$，参数量比单线性层多 $\frac{2}{3}$。LLaMA 的做法是**把中间维度从 $4d$ 压到 $\frac{8}{3}d$**，让总参数量回到与 ReLU 版本持平——于是**同等参数下表达能力更强**。这正是「用结构换智能、不换预算」的典范。<span class="marginnote">参数是多了还是少了，取决于你如何对齐：如果直接比较「同维度」的 ReLU 与 SwiGLU，SwiGLU 参数更多但效果也更好；如果比较「同参数」，SwiGLU 用更小的中间维换来了更好的效果。业界统一用后者。</span>

主流激活函数放一张表对比：

| 激活函数 | 公式 | 特点 | 常见搭配 |
| --- | --- | --- | --- |
| ReLU | $\max(0, x)$ | 简单、稀疏，0 点有折角 | CNN、早期 Transformer |
| GELU | $x\Phi(x)$ | 平滑近似，GPT 系列早期默认 | BERT、GPT-1/2 |
| SiLU（Swish） | $x\sigma(x)$ | 平滑、负值软渗透 | 可作门控 |
| SwiGLU | $\mathrm{SiLU}(xW+b) \otimes (xV)$ | 门控、同参数更强 | LLaMA、GPT-4 |

从这张表能看到一条清晰的演化线：激活函数从「简单、稀疏」（ReLU）走向「平滑、可门控」（SwiGLU），本质是让网络在每一维上拥有更细腻的信息流控制——这条演化线的终点，是当代大模型里几乎清一色的 gated 激活。它背后还有一个工程动机：门控的双支路结构天然更适合与注意力头的「多头」并行，也更容易被张量并行切分（《高效训练与并行》），这让 SwiGLU 在「效果与工程」两头都占了先。

## 3 归一化位置：Pre-Norm 还是 Post-Norm？

RMSNorm 放在哪里，是另一个直接影响训练稳定性的选择。LLaMA 采用 **Pre-Norm**（把归一化放在子层之前）：

$$ x_{l+1} = x_l + \mathrm{SubLayer}(\mathrm{RMSNorm}(x_l)) $$

**辨析｜易错点：** Transformer 原文用的是 Post-Norm（先子层后归一化），但 Pre-Norm 在现代大模型中几乎成为默认。原因有二：

- **梯度更稳**：Pre-Norm 的残差路径直接叠加 $x_l$，即使深层也有一条「恒等捷径」，深层网络的梯度消失被进一步缓解。
- **代价是表达力略降**：Post-Norm 的理论表达更强，但训练不稳。实践上大模型普遍「用 Pre-Norm 换稳定」，这就是为什么《训练稳定性与损失尖峰》那一节的「稳定优先」哲学在这里已经有铺垫。<span class="marginnote">注意 LLaMA 还做了个小细节：把 `RMSNorm` 也加在了<strong>最后一层输出之前</strong>（final norm），因为嵌入权重是共享的，输出端也要归一化。</span>

归一化策略与激活函数是「组件的组件」——单独看都不起眼，组合起来却决定了几十亿参数网络能不能被稳定训练起来。这正是预训练工程「细节决定成败」的地方：**每一个「少做一步」的设计，都是在为《高效训练与并行》里的显存与算力预算腾地方**。

## 4 RMSNorm 与 SwiGLU 的组合效果

RMSNorm 与 SwiGLU 单独看是「省一点」「强一点」，组合起来却在多个层面改变了预训练：

- **收敛更快**：更少的归一化计算 + 更平滑的激活，让损失下降的整体曲线更顺；
- **显存更低**：RMSNorm 省去均值路径的中间量、SwiGLU 的中间维度压缩减少激活显存，二者叠加直接缓解《高效训练与并行》里的显存压力；
- **数值更稳**：去掉 bias 的归一化 + 软渗透的门控，让极端激活更难出现，间接服务《训练稳定性与损失尖峰》。

**数值算例**：设隐藏维 $d = 4096$。ReLU 风格前馈的中间维是 $4d = 16384$，两块权重合计约 $2 \times d \times 4d = 8d^2 \approx 1.34 \times 10^8$ 参数；SwiGLU 把中间维压到 $\frac{8}{3}d \approx 10922$，两块门控/内容权重合计约 $2 \times 2 \times d \times \frac{8}{3}d \approx 10.7d^2 \approx 1.79 \times 10^8$——**门控带来的表达力提升，只付出少量额外参数的代价**，而 LLaMA 通过维度压到 $\frac{8}{3}d$ 让总预算与 ReLU 版本相当。

**辨析｜易错点：** 一个常见误解是「SwiGLU 一定比 ReLU 参数少」。如果直接比较「同中间维」，SwiGLU 因为多一个矩阵，参数反而更多；业界说的是「同参数量下的效果对比」。LLaMA 用 $\frac{8}{3}d$ 的目的，正是把「多出来的门控参数」压回与 ReLU 相当的量级，再在同一个预算下比较表达力。

这套「归一化 + 激活」的选择还有一个宏观意义：**它标志着预训练架构从「通用配方」走向「按算力定制」**。LayerNorm + ReLU 是 CNN/小模型的遗产，而 RMSNorm + SwiGLU + Pre-Norm 是「千亿参数、万亿 token」时代的特化——组件的选择本身，就是规模定律在架构层面的回响。

**RMSNorm 的「效果不减」还有实证证据**：Zhang & Sennrich 在 LSTM 与 Transformer 上都验证过，RMSNorm 与 LayerNorm 的收敛曲线几乎重合——这从实验上支持了「均值中心化在残差架构里是冗余的」这一论断。

**Pre-Norm 与 Post-Norm 还有一个「帽子戏法」**：Post-Norm 在层数少时表现更好、层数多时训练崩坏；Pre-Norm 则在深层更稳，早期层数少时略逊。大模型动辄几十上百层，因此 Pre-Norm 成了「深层的刚需」——这也是为什么从 BERT 到 LLaMA，架构越深、Pre-Norm 越普遍。

激活函数的选择还有「推理端」的考量：SiLU 是光滑函数，量化（如 INT8）时比 ReLU 的折点更「友好」；SwiGLU 的门控输出天然夹在门值与内容之间，动态范围可控——这让它在部署友好的现代模型里也占优。

## 5 小结

- **RMSNorm** 去掉均值和 bias，只做 RMS 缩放，**算力省、效果不减**——来自「缩放不变性才是关键」的洞察。
- **SwiGLU** = SiLU 门控 × 线性内容，把硬开关升级成连续旋钮；配合中间维度压到 $\frac{8}{3}d$，**同参数下表达力更强**。
- **Pre-Norm**（先归一化后子层）在现代大模型中默认，因为梯度更稳、深层更可训练。
- 三个选择共享同一哲学：**用结构上的「少」换训练上的「稳」和「省」**——这是预训练工程的普遍美学。

在下一节，我们将把视角从「网络结构」转向「怎么把它跑起来」——**高效训练与并行**：数据并行、张量并行、流水线并行与它们如何协作。
