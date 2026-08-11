---
title: RMSNorm 与 SwiGLU
date: 2026-08-11
---

# RMSNorm 与 SwiGLU

<div class="epigraph">
<p>我们以 RMSNorm 替代 LayerNorm，以 SwiGLU 激活函数替代 ReLU——用更少的计算获得同等的表达能力。</p>
<footer>—— Hugo Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023) §2.2</footer>
</div>

<div class="article-byline">
<p>第四级 · 高阶专题 · 大模型预训练 ｜ 对标教材：LLaMA 2023 §2.2; Zhang & Sennrich 2019; Shazeer 2020 ｜ 2026-08-11</p>
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

## 3 归一化位置：Pre-Norm 还是 Post-Norm？

RMSNorm 放在哪里，是另一个直接影响训练稳定性的选择。LLaMA 采用 **Pre-Norm**（把归一化放在子层之前）：

$$ x_{l+1} = x_l + \mathrm{SubLayer}(\mathrm{RMSNorm}(x_l)) $$

**辨析｜易错点：** Transformer 原文用的是 Post-Norm（先子层后归一化），但 Pre-Norm 在现代大模型中几乎成为默认。原因有二：

- **梯度更稳**：Pre-Norm 的残差路径直接叠加 $x_l$，即使深层也有一条「恒等捷径」，深层网络的梯度消失被进一步缓解。
- **代价是表达力略降**：Post-Norm 的理论表达更强，但训练不稳。实践上大模型普遍「用 Pre-Norm 换稳定」，这就是为什么《训练稳定性与损失尖峰》那一节的「稳定优先」哲学在这里已经有铺垫。<span class="marginnote">注意 LLaMA 还做了个小细节：把 `RMSNorm` 也加在了<strong>最后一层输出之前</strong>（final norm），因为嵌入权重是共享的，输出端也要归一化。</span>

## 4 小结

- **RMSNorm** 去掉均值和 bias，只做 RMS 缩放，**算力省、效果不减**——来自「缩放不变性才是关键」的洞察。
- **SwiGLU** = SiLU 门控 × 线性内容，把硬开关升级成连续旋钮；配合中间维度压到 $\frac{8}{3}d$，**同参数下表达力更强**。
- **Pre-Norm**（先归一化后子层）在现代大模型中默认，因为梯度更稳、深层更可训练。
- 三个选择共享同一哲学：**用结构上的「少」换训练上的「稳」和「省」**——这是预训练工程的普遍美学。

在下一节，我们将把视角从「网络结构」转向「怎么把它跑起来」——**高效训练与并行**：数据并行、张量并行、流水线并行与它们如何协作。
