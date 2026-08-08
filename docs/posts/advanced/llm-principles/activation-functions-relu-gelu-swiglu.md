---
title: 激活函数演进：ReLU、GELU 到 SwiGLU
date: 2026-08-07
---

# 激活函数演进：ReLU、GELU 到 SwiGLU

<div class="epigraph">
<p>非线性是深度学习的灵魂。</p>
<footer>—— 机器学习社区谚语（化用）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型原理 ｜ Hendrycks & Gimpel 2016《GELU》 / Shazeer 2020《GLU Variants》 ｜ 2026-08-07</p>
</div>

## 为什么激活函数是"每个 FFN 都有的小改动"

激活函数是 FFN 中间那个不起眼的非线性。但就是这个小改动，主导了从 ReLU（GPT-2）到 GELU（GPT-3）再到 SwiGLU（LLaMA）的演进——每一次替换都带来了可测的效果提升。<span class="marginnote">激活函数选择的微妙之处在于：它不改变网络的「结构容量」（层数、宽度不变），只改变「非线性形态」——而就这么一点形态差异，在万亿 token 的预训练尺度上会被放大成明显的 loss 差距。小改动、大杠杆。</span>这篇沿时间线讲清三代激活的动机与形态。

## 1 ReLU：简单但不平滑

**ReLU（Rectified Linear Unit）**：

$$
\text{ReLU}(x) = \max(0, x)
$$

它是 Transformer 早期（以及一切深层网络）的默认选择。优势无可辩驳：

- **计算极简**：一次比较，无指数、无除法。
- **梯度稀疏但方向稳定**：正半轴导数恒 1，不存在 sigmoid 的梯度饱和。
- **实现友好**：GPU/CPU 上一条指令，融合内核容易。

但 ReLU 有**两个毛病**：

- **不可微于 0**：在 $x=0$ 处导数不存在（工程上取 0 或 1），梯度不平滑。
- **死亡 ReLU**：负半轴梯度恒 0，一旦神经元权重更新到「永远输出负值」，它就永久失活。Transformer 里的 FFN 靠大维度稀释了这个风险，但依然存在。

**关键短板是「不平滑」**：ReLU 在 0 处有尖角，且在负区间完全平坦——这让梯度在 0 附近剧烈变化，优化轨迹不够顺滑。GELU 正是为「平滑」而来。<span class="marginnote">ReLU 的「尖角」在视觉里很直观：它在 $x=0$ 处折了一个 90 度的弯。深度学习优化理论里，目标函数的平滑性（Lipschitz 梯度）越好，梯度下降越稳。GELU/SiLU 的平滑让优化「少磕碰」。</span>

## 2 GELU：用概率门控的平滑 ReLU

**GELU（Gaussian Error Linear Unit）**：

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数（CDF）。直觉是：**用「$x$ 有多大可能为正」来门控 $x$ 本身**——$x$ 越大，通过的比例越高。

它有几个重要性质：

- **平滑**：$\Phi(x)$ 光滑单调，整个函数处处可微，无尖角。
- **近似 ReLU**：$\Phi(x)$ 的形状近似「软的阶跃」，因此 GELU 曲线近似「圆滑版的 ReLU」——大正数 ≈ 通过，大负数 ≈ 截止，过渡区平滑衔接。
- **负区间允许微弱输出**：不像 ReLU 硬性归零，GELU 在负半轴保留很小的值，携带了「微弱但可能有用」的负信号。

常用近似（避免每次算 $\Phi$ 的高斯积分）：

$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{2/\pi}\,(x + 0.044715 x^3)\right)\right)
$$

GPT-3、BERT 都默认 GELU，因为它用「平滑门控」换来了更稳的优化与更好的泛化。<span class="marginnote">GELU 的思想可以追溯到「随机正则」：$\Phi(x)$ 可以解释成「以概率 $\Phi(x)$ 把 $x$ 原样通过、以 $1-\Phi(x)$ 置零」的期望。所以 GELU 自带一点点「dropout 味」——这也是它泛化更好的一个解释。</span>

## 3 SiLU 与 SwiGLU：门控家族的崛起

**SiLU（Sigmoid Linear Unit，也叫 Swish）**：

$$
\text{SiLU}(x) = x \cdot \sigma(x)
$$

与 GELU 同族（都是「$x$ 乘一个 0–1 之间的平滑门」），但用 sigmoid 替代高斯 CDF，计算更便宜。它非单调、有下界无上界，在接近 0 处有一个「凹陷」，实证上优于 ReLU。

**GLU（Gated Linear Unit）** 更进一步：不是「一个输入自己门控」，而是**两个线性投影互相门控**：

$$
\text{GLU}(a, b) = a \odot \sigma(b)
$$

其中 $a$ 是「内容」，$b$ 是「门」。**SwiGLU** 把门换成 Swish：

$$
\text{SwiGLU}(x) = (xW_1) \odot \text{swish}(xW_g) \cdot W_2
$$

即 FFN 里不再是「一层线性 + 激活」，而是**两条平行线性分支，一条作内容、一条作门，逐元素相乘后再投影输出**。<span class="marginnote">门控的核心价值是「选择性」：模型可以让门分支学习「哪些中间特征应该被放到输出里」，像一道可学习的闸门。相比 GELU 的「用 $x$ 自己门控」，GLU 的「用另一个投影门控」给了模型更大的自由度——代价是多一组参数。</span>

## 4 公式解析：SwiGLU 的参数与计算代价

SwiGLU-FFN 的完整前向：

$$
\text{FFN}_{\text{SwiGLU}}(x) = \underbrace{\big((xW_1) \odot \text{swish}(xW_g)\big)}_{h \in \mathbb{R}^{m}} W_2
$$

其中 $W_1, W_g \in \mathbb{R}^{d \times m}$，$W_2 \in \mathbb{R}^{m \times d}$，$m$ 是中间维度。

对这条式子做四步拆解：

- **第一步，读懂三个投影**：$W_1$ 生成「内容」，$W_g$ 生成「门」，$W_2$ 把门控后的结果投影回 $d$ 维。与普通 FFN 的「升维→激活→降维」相比，这里升维被拆成两路（内容 + 门）。
- **第二步，读懂 $\odot$ 与 swish**：$\odot$ 是逐元素相乘。门值在 $(0, x)$ 之间（swish 有界于 $(-0.28, \infty)$），相乘相当于「按特征软开关」。
- **第三步，算参数**：三个投影共 $3md$ 个参数，比普通 FFN 的 $2md$ 多了 50%。**但 LLaMA 用 $3md$ 的 2/3 来补偿**——即把 $m$ 调小，使总参数与普通 FFN 持平的同时，效果更好。
- **第四步，读出为什么更强**：门控让模型能「同时记住『该输出什么』与『该抑制什么』」——表达能力从「一个非线性变换」升级为「两个互补通道的决策」。实证上 SwiGLU 比 GELU 在同等参数下 loss 更低。

**辨析｜易错点：** SwiGLU 的「门」是**逐元素**的软门，不是 token 级的硬路由。别把它和 MoE 的路由混淆：SwiGLU 每个 token 都完整经过所有中间特征（只是特征被软缩放）；MoE 是「每个 token 只选少数专家」。**门控是软的、路由是硬的**。

## 5 选型对照表

| 激活 | 形式 | 平滑 | 参数开销 | 代表模型 |
| --- | --- | --- | --- | --- |
| ReLU | $\max(0,x)$ | 否 | 无额外 | 早期 Transformer |
| GELU | $x\Phi(x)$ | 是 | 无额外 | GPT-2/3、BERT |
| SiLU/Swish | $x\sigma(x)$ | 是 | 无额外 | 部分 ViT、MoE 门控 |
| SwiGLU | $(xW_1)\odot \text{swish}(xW_g)W_2$ | 是 | +50%（可压缩补偿） | LLaMA、Mistral、Qwen、Gemma |

**趋势总结**：从「硬截止」到「软门控」，从「无参数」到「有参数门控」——激活函数越来越「有信息量」。代价是计算与参数增多，但在现代硬件上这点开销换来的是更稳的训练与更低的 loss，是划算的。

## 6 小结

- **ReLU**：极简但**不平滑、有死亡神经元**；是早期默认。
- **GELU**：用高斯 CDF **平滑门控**，处处可微、近似圆滑 ReLU，GPT-3 默认。
- **SiLU**：sigmoid 版门控，更便宜；**GLU** 用双分支互相门控。
- **SwiGLU**：内容分支 + Swish 门分支逐元素相乘，**LLaMA 系默认**；参数多 50%，用缩小中间维度补偿。
- 演进主线：**硬截止 → 平滑 → 软门控**，表达能力逐级增强。

在下一节，我们补上归一化与激活之外的「进阶技巧」——**QK-Norm、Sandwich-LN 与注意力 logits 软裁剪**，这些是大模型训练稳定性的最后一公里。
