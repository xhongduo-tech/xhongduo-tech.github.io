## 核心结论

DeepSeek-V3 的长上下文扩展，不是“重新训练一个 128K 模型”，而是把一个原本主要在 4K 上下文上学会语言建模的 base 模型，用两阶段 YaRN 微调，逐步推到 32K，再推到 128K。

这里先解释两个术语。

RoPE 是旋转位置编码，白话说，它不是给 token 直接加一个位置编号，而是让注意力里的向量按位置发生“旋转”，这样模型能感知“谁在前、谁在后、相距多远”。

MLA 是 Multi-head Latent Attention，白话说，它先把 Key/Value 压缩到更小的隐空间，再做缓存和计算，用更少显存保留长历史信息。

DeepSeek-V3 的关键点不只是“用了 YaRN”，而是“YaRN 只作用在 MLA 中解耦出来的 RoPE 维度 $d_r=64$ 上，不改压缩维度”。这意味着：

| 结论 | 含义 | 工程价值 |
| --- | --- | --- |
| 两阶段扩展：4K $\rightarrow$ 32K $\rightarrow$ 128K | 不是一步拉到 128K，而是分两次适配 | 训练更稳定 |
| 只改解耦的 RoPE 维度 | 只动位置频率相关部分 | 不破坏 MLA 压缩缓存结构 |
| 压缩维度保持不动 | 历史 KV 压缩方式不重建 | 成本低、兼容原架构 |
| softmax_scale 同步放大 | 长距离注意力不会被“抹平” | perplexity 更稳 |

对初学者可以这样理解：模型原来只会在 4K 长度里“看图识路”，YaRN 不是给它换眼睛，而是把它识别距离的“刻度尺”拉长；DeepSeek-V3 又进一步限定，只拉长负责位置感知的那一小段刻度，不去碰 MLA 的压缩主干，因此代价小、风险低。

---

## 问题定义与边界

问题定义很明确：如何让一个原本按 4K 上下文训练出来的模型，在不推翻原注意力架构的前提下，可靠支持 32K 甚至 128K 上下文。

这里的“可靠”至少包含三层要求：

| 要求 | 具体含义 |
| --- | --- |
| 可训练 | 扩展后不会立刻数值发散 |
| 可泛化 | 长距离位置关系仍能被模型识别 |
| 可部署 | KV 缓存和显存成本不能失控 |

DeepSeek-V3 给出的边界条件是：

1. 不从头做一次 128K 预训练。
2. 不扩大 MLA 的压缩维度。
3. 只对解耦出来的 RoPE 分量做频率调整。
4. 训练时同步调整 softmax_scale，否则长上下文注意力会退化。

为什么不能直接把上下文长度配置改成 128K 就结束？因为训练过的 RoPE 频率，实际上只在原始训练长度附近是“熟悉区间”。如果把长度一下拉到 128K，模型虽然形式上能处理更长输入，但高频位置分量会在远距离区域失真，表现为远处 token 的相对关系不再稳定，检索、跨段推理和长代码依赖都会出问题。

这可以用一个玩具例子说明。

假设一个模型原来只在 4K 内训练。现在把一篇 50K token 的文档喂进去，文档开头定义了变量 `risk_budget = 0.3`，尾部第 48K token 处需要用到它。如果没有长上下文适配，模型往往不是“完全看不到前文”，而是“位置关系变形后，注意力难以稳定落在正确位置”。结果就是，它会把远处信息当成噪声。

DeepSeek-V3 的边界也很清楚：这种方法适用于“已有 RoPE 且位置维度能被明确分离”的模型。对于没有 RoPE、或者位置编码已经深度耦合进主计算路径的架构，这条路不一定成立。

---

## 核心机制与推导

YaRN 的核心思想，是把原始 RoPE 的频率分布平滑改写成更适合长上下文的频率分布，而不是粗暴替换。

先看原始 RoPE 的基本形式。对第 $i$ 对偶维，角频率通常写成：

$$
\omega_i = \theta^{-2i/d}
$$

其中 $\theta$ 是基频常数，常见取值是 $10^4$，$d$ 是参与旋转的位置维度。

位置为 $m$ 的 token，会在该频率下产生角度：

$$
\phi_{m,i} = m \cdot \omega_i
$$

问题在于，当 $m$ 从 4K 拉到 128K 时，原本训练好的频率分布会让高频部分变化过快，远距离区域的相位关系变得不稳定。YaRN 的做法是：对一部分频率分量做缩放，并在“原频率”和“缩放后频率”之间做平滑过渡。

可以把它写成一个混合形式：

$$
\omega_i' = (1-r_i)\cdot \frac{\omega_i}{s} + r_i \cdot \omega_i
$$

其中：

- $s$ 是缩放因子，白话说，就是把“位置时钟”放慢多少倍。
- $r_i \in [0,1]$ 是 ramp 系数，白话说，就是这一维更接近旧频率还是新频率。

DeepSeek-V3 的公开资料里给出的典型配置是 $s=40,\alpha=1,\beta=32$，并结合线性 ramp 使用。直观上看，低频部分原本就更适合长距离建模，可以少改；高频部分更容易在长距离失真，需要更强的缩放；中间区域则做平滑混合。

可以把这种机制理解成“旧地图到新地图的渐进换算”。不是所有比例尺一起乘 40，而是有选择地调整不同频段，再用 ramp 过渡，避免模型在参数空间里遭遇突然跳变。

为什么 DeepSeek-V3 特别强调只改 $d_r=64$ 的解耦 RoPE 维度？因为在 MLA 中，位置相关分量和压缩缓存主分量已经被分开。于是可以把公式只施加到那部分：

$$
k = [k^{C}; k^{R}]
$$

这里：

- $k^{C}$ 是压缩后的内容分量，白话说，主要承载语义内容并参与低成本缓存。
- $k^{R}$ 是单独保留的 RoPE 分量，白话说，专门负责位置旋转。

DeepSeek-V3 的长上下文适配，本质上就是只改 $k^R$ 的旋转频率，不改 $k^C$ 的压缩表达。于是缓存结构、低秩投影和大部分推理路径都能保持原样。

这就是它比“全量改所有 Key/Query 维度”更工程化的地方。

还要处理一个经常被忽略的问题：softmax_scale。

注意力分数一般写成：

$$
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}} \cdot \gamma \right)
$$

其中 $\gamma$ 可以理解为额外的缩放系数。上下文变长后，位置分布和相似度统计会变化，如果仍然用原来的 scale，注意力可能变得过于平滑，远距离有效峰值不够明显。DeepSeek-V3 在扩展时同步提高 softmax_scale，本质上是在补偿长上下文导致的注意力熵变化。

从直觉上说，YaRN 负责“把尺子拉长”，softmax_scale 负责“把焦点重新对准”。只做前者而不做后者，模型能看远，但看不清。

训练阶段可以概括成下面这张表：

| 阶段 | 训练上下文 | 目标作用 | 典型 batch | 步数 |
| --- | --- | --- | --- | --- |
| 阶段一 | 4K $\rightarrow$ 32K | 先把模型适配到中长上下文 | 1920 | 1000 |
| 阶段二 | 32K $\rightarrow$ 128K | 再把位置能力推到超长上下文 | 480 | 1000 |

这里的逻辑不是“32K 够了再多训一点”，而是“先进入较长区间，再进入超长区间”。因为 4K 到 128K 的跨度是 32 倍，直接一步跨过去，参数更新很容易不稳定；拆成两次，模型有一个中间着陆点。

真实工程例子可以看长代码库理解。一个大型仓库可能有几十万行代码，单个 bug 的根因横跨 README、配置文件、接口定义、调用方和测试。4K 或 8K 上下文往往只能看局部；32K 能覆盖一个中型模块；128K 才有机会在一次推理里把“接口定义在前、异常触发在中、补丁建议在后”串起来。DeepSeek-V3 的这种两阶段扩展，就是为了让模型在这种任务上保留更稳定的远距检索能力。

---

## 代码实现

下面给一个可运行的玩具实现。它不复现 DeepSeek-V3 的全部训练细节，只演示三个关键点：

1. 只对解耦的 RoPE 维度做 YaRN 缩放。
2. 用 ramp 在旧频率和新频率之间平滑混合。
3. 上下文扩展时同步调整 softmax_scale。

```python
import math

def linear_ramp(i, start, end):
    if i <= start:
        return 1.0
    if i >= end:
        return 0.0
    return 1.0 - (i - start) / (end - start)

def yarn_frequencies(dim, base_theta=10000.0, scale=40.0, beta_slow=1, beta_fast=32):
    assert dim % 2 == 0
    half = dim // 2
    freqs = []
    for i in range(half):
        # 原始 RoPE 频率
        omega = base_theta ** (-2.0 * i / dim)
        # 线性 ramp：低索引更保留原频率，高索引更偏向缩放后频率
        r = linear_ramp(i, beta_slow, beta_fast)
        omega_scaled = omega / scale
        omega_mixed = r * omega + (1.0 - r) * omega_scaled
        freqs.append(omega_mixed)
    return freqs

def softmax_scale_multiplier(scale):
    # 近似演示：资料里常见写法会把 mscale 与 ln(scale) 关联
    # 这里只保留“scale 增大时，softmax_scale 也增大”的性质
    return 1.0 + 0.1 * math.log(scale)

def apply_stage(base_softmax_scale, rope_dim, target_ctx):
    if target_ctx == 32_000:
        scale = 8.0   # 相对 4K 的近似扩展倍数
    elif target_ctx == 128_000:
        scale = 32.0  # 相对 4K 的近似扩展倍数
    else:
        raise ValueError("unsupported target_ctx")

    freqs = yarn_frequencies(dim=rope_dim, scale=40.0, beta_slow=1, beta_fast=32)
    new_softmax_scale = base_softmax_scale * softmax_scale_multiplier(scale)
    return freqs, new_softmax_scale

# 只对解耦的 RoPE 维度生效，DeepSeek-V3 报告中的典型值是 d_r=64
rope_dim = 64
base_softmax_scale = 1.0

freqs_32k, scale_32k = apply_stage(base_softmax_scale, rope_dim, 32_000)
freqs_128k, scale_128k = apply_stage(base_softmax_scale, rope_dim, 128_000)

assert len(freqs_32k) == rope_dim // 2
assert scale_32k > base_softmax_scale
assert scale_128k > scale_32k

# 高频端应更接近缩放后的慢频率，因而通常小于低维原始频率
assert freqs_32k[-1] < freqs_32k[0]

print("32K softmax scale:", round(scale_32k, 4))
print("128K softmax scale:", round(scale_128k, 4))
```

上面这个玩具代码对应的真实实现思路是：

| 组件 | 是否修改 | 原因 |
| --- | --- | --- |
| MLA 压缩维度 $d_k,d_v$ | 否 | 保持缓存结构稳定 |
| 解耦 RoPE 维度 $d_r=64$ | 是 | 长上下文能力主要在这里适配 |
| softmax_scale | 是 | 补偿长上下文下的注意力平滑 |
| 学习率 | 基本不变 | 避免大步更新导致崩溃 |

如果写成训练伪代码，结构更接近工程实践：

```python
base_model = load_pretrained_4k_model()

for stage in [
    {"target_ctx": 32000, "steps": 1000, "batch": 1920, "lr": 7.3e-6},
    {"target_ctx": 128000, "steps": 1000, "batch": 480, "lr": 7.3e-6},
]:
    model = base_model

    # 1. 只替换解耦 RoPE 分量的频率表
    model.mla.rope_freqs = build_yarn_freqs(
        rope_dim=64,
        scale=40.0,
        alpha=1.0,
        beta=32.0,
        target_ctx=stage["target_ctx"],
    )

    # 2. 同步调整 attention 的 softmax_scale
    model.attn.softmax_scale *= mscale(stage["target_ctx"] / 4000)

    # 3. 用对应长度的数据继续训练
    for step in range(stage["steps"]):
        batch = next_long_context_batch(
            seq_len=stage["target_ctx"],
            global_batch=stage["batch"],
        )
        loss = model(batch)
        loss.backward()
        optimizer.step(lr=stage["lr"])
        optimizer.zero_grad()

    base_model = model
```

初学者要抓住一点：这里不是“先训练 MLA，再训练 YaRN”，而是“在保留 MLA 主体不变的前提下，针对 RoPE 子空间做长上下文适配训练”。

---

## 工程权衡与常见坑

工程上最重要的权衡，是“最小改动”与“最长上下文”之间的平衡。

DeepSeek-V3 选择了偏工程化的路线：不追求最彻底的重构，而是在已有架构上找到最小可改面。优点是成本低、迁移快、缓存兼容；代价是这种方法高度依赖架构本身已经把位置分量解耦出来。

常见坑可以直接列出来：

| 常见坑 | 现象 | 原因 | 规避方式 |
| --- | --- | --- | --- |
| 把 YaRN 用到压缩 KV 主维度 | 推理缓存错位，效果明显掉 | 破坏 MLA 原始压缩结构 | 只改 $k^R$ 与对应 RoPE 分量 |
| 只改 RoPE，不改 softmax_scale | 长距离检索能力弱 | 注意力在长序列上过于平滑 | 随扩展比例同步调大 scale |
| 32K 和 128K 直接混训 | 训练震荡，loss 不稳 | 长度跨度过大，数据分布切换剧烈 | 先 32K，再 128K，两阶段过渡 |
| 仍沿用短序列 batch | 显存爆炸或吞吐大降 | 序列长度上去后 token 数急剧增加 | 32K/128K 分别下调 batch |
| 学习率拉高 | 原模型能力被破坏 | 这是结构适配，不是全新学习 | 保持弱学习率、小步更新 |

再看一个更接近真实系统的例子。

假设你在做一个“长文档问答”系统，单次输入要容纳产品需求文档、接口文档、日志摘要和监控说明，总长度接近 90K token。模型如果只是形式支持 128K，但没有真正做过长上下文适配，常见故障是：

1. 开头提到的术语定义，到后文引用时失效。
2. 中间段落里的约束条件被忽略。
3. 文档后半段的日志证据无法稳定回指到前半段配置项。

如果你又错误地把 YaRN 施加到 MLA 的压缩主维度，推理阶段还会进一步出现 KV 缓存不一致，表现为“同样输入，多轮结果抖动变大”。

这里可以给一个初学者版本的直观比喻，但只保留定义作用：RoPE 像位置时钟，MLA 压缩维度像仓库货架。DeepSeek-V3 的做法是调时钟，不拆货架。你如果连货架一起重刷编号，库存系统就先乱了。

---

## 替代方案与适用边界

YaRN 不是唯一方案，它只是“在已有 RoPE 模型上做低成本长上下文扩展”时非常合适的一条路。

先和从头预训练做对比：

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| YaRN 两阶段微调 | 成本低，迁移快，能复用原模型 | 受原架构限制，能力上限依赖基座 | 已有强 base 模型，想快速扩长 |
| 从头做 32K/128K 预训练 | 理论上最一致 | 成本极高，训练周期长 | 资金和算力都充足的新模型 |
| 一步从 4K 拉到 128K | 配置简单 | 稳定性差，容易退化 | 不建议用于正式工程 |
| 保持短上下文，用 RAG 补足 | 推理成本可控 | 跨段全局推理仍受限 | 检索强、推理跨度较小的任务 |

再和其他位置方案做对比：

| 位置方案 | 核心思想 | 优势 | 边界 |
| --- | --- | --- | --- |
| RoPE + YaRN | 通过频率缩放和平滑插值扩展上下文 | 对已有 RoPE 模型友好 | 需要可调整的 RoPE 结构 |
| ALiBi | 给注意力分数加距离偏置 | 实现简单，外推性强 | 与既有 RoPE 模型不一定兼容 |
| 全量 RoPE 插值 | 所有位置维统一改写 | 通用性更强 | 若架构有解耦子空间，可能不如局部改动稳 |
| 重新设计相对位置编码 | 改模型定义本身 | 上限高 | 改动大，迁移成本高 |

DeepSeek-V3 的做法特别适合下面这个边界条件：模型已经有强大的短中程能力，且架构上能明确区分“内容压缩分量”和“位置旋转分量”。这时只改后者，收益很高。

但如果模型根本没有解耦的 RoPE 子空间，或者整个注意力实现把位置编码深度揉进了主投影，那就不能简单照搬“只改 $d_r=64$”这条路线。此时要么改成更通用的 RoPE 插值，要么考虑 ALiBi 一类替代方案，要么接受重新训练更长上下文基座的成本。

一句话总结适用边界：YaRN 像“原地加长跑道”，不是“重修机场”。机场结构本身支持扩建时，它很划算；结构不支持时，硬拉只会引入新的不稳定点。

---

## 参考资料

1. DeepSeek-V3 Technical Report：覆盖两阶段 YaRN、32K/128K 扩展流程、训练超参与长上下文评测。  
2. DeepSeek-V3 Base 系列解读文章：覆盖 RoPE 插值直觉、两阶段扩展示例和训练配置的工程解释。  
3. MLA 深度解析文章：覆盖解耦 RoPE 维度、$d_r=64$ 的角色，以及为什么只改这一部分。  
4. YaRN 相关二次解读：覆盖 $s=40,\alpha=1,\beta=32$、ramp 混合和 softmax_scale 补偿思路。  
5. 长上下文工程案例分析：覆盖长文检索、代码库理解、Needle-in-a-Haystack 一类任务中的适配价值。
