## 核心结论

Flamingo 的核心做法不是重训练整个多模态大模型，而是把一个已经很强的视觉编码器和一个已经很强的大语言模型冻结住，只在语言模型的若干层之间插入少量可训练模块。这里的“冻结”是指主干参数不更新，只训练新增的小模块；这样做的直接好处是训练成本更低，也更容易保留原有语言能力。

这组新增模块主要有两个：

1. Perceiver Resampler。它的作用可以理解为“把很长的视觉特征压缩成固定长度的摘要”。
2. Gated Cross-Attention。它的作用可以理解为“给语言层加一个可学习的视觉入口，而且这个入口有门控开关”。

因此，Flamingo 不是把图像直接拼成很多 token 丢给 LLM，而是先把图像编码成一组固定数量的 latent，再让语言模型在若干层通过交叉注意力去读取这些 latent。门控参数初始接近“关门”状态，训练后再学会在需要视觉信息时把门打开。

如果把结论压缩成一个公式，可以写成：

$$
h' = h + \tanh(g_{\text{attn}})\cdot c + \tanh(g_{\text{ff}})\cdot f
$$

其中 $h$ 是原始文本隐藏态，白话说就是“语言模型当前这一层对文本的内部表示”；$c$ 是跨模态注意力输出，白话说就是“从图像里读出来的补充信息”；$f$ 是额外前馈层输出；$g_{\text{attn}}$ 和 $g_{\text{ff}}$ 是两个可学习门控。这个结构说明：Flamingo 不是替换 LLM，而是在 LLM 内部按层插入受控的视觉增量。

这也是 Flamingo 能做多模态 few-shot in-context learning 的根本原因。所谓 in-context learning，白话说就是“模型不改参数，只靠提示里的几个例子现场学会任务格式”。Flamingo 训练后可以处理交错的图文序列，所以提示里只要按“图像-问题-答案-图像-问题-答案”的格式放几个样例，再给出新图和新问题，它就有机会直接生成答案，而不需要为每个任务再做一次专门微调。

---

## 问题定义与边界

Flamingo 要解决的问题很具体：如何把预训练视觉模型和预训练语言模型接起来，让模型在看到少量图文示例后，直接完成视觉问答、图像描述、多轮图文对话等任务。

这里有三个边界需要先讲清楚。

| 维度 | Flamingo 的目标 | 不负责解决的部分 |
| --- | --- | --- |
| 训练目标 | 让冻结主干具备多模态 few-shot 能力 | 不追求端到端全部参数都更新 |
| 输入形式 | 支持交错图文序列 | 不是只做单图单问的窄任务模型 |
| 工程重点 | 降低新增训练参数，保留 LLM 能力 | 不保证是最省推理成本的结构 |

对初学者来说，最容易混淆的是“few-shot”与“fine-tuning”。few-shot 是“在提示里给几个例子”；fine-tuning 是“拿数据继续更新模型参数”。Flamingo 的强项主要在前者：任务切换时尽量不改模型参数，而是通过提示里的交错图文样例来适配。

一个玩具例子可以说明它的边界。假设提示是：

- 图 1：两只猫在沙发上
- 问：图中有几只猫？
- 答：2
- 图 2：一只狗在草地上
- 问：图中是什么动物？
- 答：狗
- 图 3：新图
- 问：图中有几只猫？
- 答：

Flamingo 的目标是让模型从前两个图文样例中学会“这是视觉问答任务，而且回答要简短”，然后在第三个样例上直接输出答案。这里模型依赖的是提示结构和跨模态读取能力，而不是任务时再训练一遍。

真实工程里，一个典型场景是电商质检或客服辅助。比如系统收到用户上传的商品图，提示中给两三个“图像 + 问题 + 标准回答”的样例，再问“这张图里的包装是否破损？”或者“主色是否为黑色？”。如果模型能稳定利用图像信息并跟随提示格式，就可以快速接到新任务上。

但 Flamingo 也有明显边界：

| 场景 | 是否适合 Flamingo | 原因 |
| --- | --- | --- |
| 多轮图文对话 | 适合 | 图文交错输入是原生设计目标 |
| few-shot VQA | 适合 | 可通过上下文样例适配 |
| 大规模任务专门优化 | 视情况而定 | 端到端微调模型可能更强 |
| 强视觉定位、检测框输出 | 不理想 | Flamingo 的核心不是检测头 |
| 极低延迟在线推理 | 需谨慎 | 跨模态插层会增加推理开销 |

所以，Flamingo 更像“通用多模态语言模型桥接方案”，而不是“针对某个视觉任务做到极致”的专用网络。

---

## 核心机制与推导

Flamingo 的机制可以拆成三步：视觉编码、视觉压缩、语言层插入跨模态读取。

第一步是视觉编码。通常先用预训练视觉编码器，例如 CLIP ViT，把图像变成视觉 token 序列。这里的“视觉 token”可以理解为“图像被切成小块后，每块对应的特征向量”。

第二步是 Perceiver Resampler。因为视觉 token 可能很多，直接让 LLM 每层都去关注它们，计算会很重。于是 Flamingo 用一组固定数量的 learnable latents 去“读”原始视觉 token，再输出固定长度结果。白话讲，就是把可变长图像特征压缩成固定长度的视觉摘要。

如果原始视觉特征长度是 $N$，压缩后 latent 数是 $L$，通常有 $L \ll N$。例如：

- 原始视觉 token：256 个
- Resampler 输出 latent：64 个

这样后续跨模态注意力的 Key/Value 长度从 256 变成 64，计算明显下降。

第三步是语言模型插层。Flamingo 不在每一层都插 cross-attention，而是按 `cross_attn_every_n_layers` 的节奏插入。例如每 2 层插 1 次，或者每 4 层插 1 次。这样可以在“视觉融合能力”和“计算成本”之间折中。

跨模态注意力本身仍然服从标准公式：

$$
\text{CrossAttn}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

这里：

- $Q$ 来自文本隐藏态，意思是“当前文本 token 想问什么”
- $K,V$ 来自视觉 latent，意思是“图像里有哪些可查的信息”

也就是说，文本 token 不再只看文本上下文，还能去查询图像摘要。

Flamingo 的特别之处在于门控。不是把 cross-attention 输出直接强加到语言层，而是先乘以 $\tanh(g_{\text{attn}})$。这样做有两个作用：

1. 初始化稳定。门控初始为 0 时，$\tanh(0)=0$，新增视觉路径一开始几乎不扰动原始 LLM。
2. 可学习控制。训练后模型可以学会在哪些层、哪些任务上更依赖视觉输入。

继续看一个最小数值例子。假设：

- Resampler 把 256 个视觉 token 压成 64 个 latent
- 当前层得到 cross-attention 输出 $c$
- FFN 输出为 $f$
- $g_{\text{attn}}=0.6,\ g_{\text{ff}}=0.1$

则：

$$
\tanh(0.6)\approx 0.537,\quad \tanh(0.1)\approx 0.100
$$

于是该层输出近似为：

$$
h' \approx h + 0.54c + 0.10f
$$

这意味着本层主要加入了视觉补充信息，而额外 FFN 修正较小。若当前文本 token 是“几只猫？”，那么 $c$ 就会把与“猫数量”相关的图像线索送入文本流。

这里还要补一个关键点：Flamingo 处理的不是“整段文本都能看所有图片”的简单情形，而是交错图文序列。实现上会用 `media_token` 标记图像在文本中的位置，并通过 mask 控制某个文本位置能关注哪些图像。白话说，就是“让文本只去看它该看的图”，防止注意力越界。

---

## 代码实现

如果把 Flamingo 简化成教学版本，核心逻辑可以概括为三层：

1. 图像经过视觉编码器得到 `visual_features`
2. `visual_features` 经过 `PerceiverResampler` 变成固定长度 `latents`
3. 文本隐藏态在若干层通过 `GatedCrossAttentionBlock` 读取这些 `latents`

下面这个 Python 代码块不是完整论文实现，但保留了最核心的数学路径，而且可以直接运行：

```python
import math

def tanh(x: float) -> float:
    e2x = math.exp(2 * x)
    return (e2x - 1) / (e2x + 1)

def flamingo_layer(hidden, cross_out, ffn_out, attn_gate, ff_gate):
    assert len(hidden) == len(cross_out) == len(ffn_out)
    a = tanh(attn_gate)
    b = tanh(ff_gate)
    return [h + a * c + b * f for h, c, f in zip(hidden, cross_out, ffn_out)]

# 玩具例子：3 维隐藏态
hidden = [1.0, 2.0, 3.0]
cross_out = [0.5, -0.5, 1.0]
ffn_out = [0.2, 0.2, 0.2]

out = flamingo_layer(hidden, cross_out, ffn_out, attn_gate=0.6, ff_gate=0.1)

# 结果应该体现视觉分支被“部分打开”
assert len(out) == 3
assert out[0] > hidden[0]
assert out[2] > hidden[2]

# 当 gate 为 0 时，增量应接近 0
out_closed = flamingo_layer(hidden, cross_out, ffn_out, attn_gate=0.0, ff_gate=0.0)
for x, y in zip(out_closed, hidden):
    assert abs(x - y) < 1e-9
```

这个例子体现了两个事实：

- 门控为 0 时，层几乎退化成原始路径
- 门控打开后，视觉分支和额外 FFN 才真正影响输出

更贴近工程的伪代码如下：

```python
perceiver_latents = perceiver_resampler(visual_features)

for i, lm_layer in enumerate(language_model_layers):
    if i % cross_attn_every_n_layers == 0:
        text_hidden = text_hidden + tanh(attn_gate[i]) * cross_attention(
            query=text_hidden,
            key=perceiver_latents,
            value=perceiver_latents,
            media_locations=media_locations,
        )
        text_hidden = text_hidden + tanh(ff_gate[i]) * ff_block(text_hidden)

    text_hidden = lm_layer(text_hidden)
```

真实工程例子可以用“图文客服问答”来理解。假设系统 prompt 中有两段示例：

- `<image> 这是什么商品？ -> 运动鞋`
- `<image> 包装是否破损？ -> 否`

新输入是：

- `<image> 鞋带是否缺失？ ->`

推理时的执行顺序大致是：

| 步骤 | 模块 | 作用 |
| --- | --- | --- |
| 1 | 视觉编码器 | 把三张图变成视觉特征 |
| 2 | Resampler | 每张图压成固定长度 latent |
| 3 | `media_token` 检测 | 记录文本里每个 `<image>` 位置 |
| 4 | Gated Cross-Attention | 让后续文本 token 读取对应图像 |
| 5 | LLM 解码层 | 基于文本上下文和视觉线索生成答案 |

OpenFlamingo 一类实现通常还会加入缓存逻辑。原因是生成阶段常常一次只生成一个 token，如果每次都重复编码图像会非常浪费，所以视觉特征通常会先缓存，再在后续 token 解码时复用。

---

## 工程权衡与常见坑

Flamingo 在论文层面很优雅，但工程里有几个坑非常高频。

第一个坑是 gate 长时间打不开。因为门控初始化为 0，模型一开始基本沿用原始 LLM 行为。如果训练数据里图像和答案的对应关系不够强，或者训练强度不够，模型就会学成“只看文本，不看图像”。表现上往往是：即使输入换图，回答也几乎不变。

第二个坑是 `media_token` 对不齐。交错图文序列的能力建立在严格的位置约束上。如果文本里插入图像标记的位置和实际缓存图像顺序不一致，cross-attention 可能会读错图。这个问题一旦出现，输出通常不是完全崩掉，而是“偶尔答对、整体不稳定”，所以更难排查。

第三个坑是 `cross_attn_every_n_layers` 选择不当。插得太密，推理成本上升，而且训练时容易让模型过度依赖视觉增量；插得太稀，语言模型拿不到足够视觉信号，尤其在长文本推理里更明显。

第四个坑是视觉压缩过强。Resampler 把视觉 token 压成固定 latent 数，latent 太少时，细粒度信息容易丢失。对于“图中有几只猫”这类粗任务可能还行，但对“左上角标签写的是什么”这类细节任务就不够。

常见问题和对策可以整理成表：

| 常见坑 | 现象 | 对策 |
| --- | --- | --- |
| gate 初始化为 0 后长期不开 | 输出像纯文本模型 | 增加高质量图文训练样本，检查视觉梯度是否有效 |
| `media_token` 错位 | 同一问题对不同图回答混乱 | 严格校验图像顺序、token 位置、缓存逻辑 |
| 跨注意力插层过密 | 显存和延迟明显上升 | 调大 `cross_attn_every_n_layers` |
| Resampler latent 太少 | 细节任务表现差 | 增加 latent 数或提高视觉分辨率 |
| 生成阶段未缓存视觉特征 | 推理很慢 | 预编码图像并复用视觉缓存 |

还有一个容易忽略的点：`only_attend_immediate_media` 这类掩码策略。它的意思可以白话理解为“一个文本位置只看最近那张图，还是看之前所有图”。如果做多轮图文对话，这个开关会直接影响信息范围。设错了，模型就可能把前一轮图片的信息错误带到当前回答里。

---

## 替代方案与适用边界

Flamingo 经常被拿来和 Q-Former 对比。Q-Former 可以理解为“用一小组可学习查询 token 去从图像里提取信息，再把这些查询结果交给语言模型”。它最典型地出现在 BLIP-2 一类方法中。

两者的核心差异不在于“有没有 cross-attention”，而在于“跨模态桥接发生在哪”。

| 维度 | Flamingo | Q-Former |
| --- | --- | --- |
| 视觉压缩方式 | Perceiver Resampler 生成固定 latent | Learnable query tokens 主动查询图像 |
| 与 LLM 的连接方式 | 在 LLM 层间插入 cross-attn | 常先形成紧凑视觉查询表示，再送入 LLM |
| 是否改造 LLM 内部 | 是，插层 | 往往较少改动 LLM 内部结构 |
| 适合输入形式 | 交错图文序列、多轮上下文 | 图像先抽象，再与文本对接 |
| few-shot 图文 in-context | 很自然 | 可以做，但交错图文原生感略弱 |

对初学者可以这样记：

- Flamingo 更像“在语言流里周期性打开视觉接口”
- Q-Former 更像“先派几个查询 token 去读图，再把读出来的结果交给语言模型”

什么时候更适合 Flamingo？

- 需要处理交错图文提示
- 需要多轮图文上下文
- 希望尽量保留冻结 LLM 的原始语言能力
- 希望通过少量新模块实现多模态扩展

什么时候更适合 Q-Former 类方案？

- 希望视觉侧先得到更独立、更紧凑的表示
- 不想深度改造 LLM 内部层结构
- 需要一个更清晰的“视觉查询瓶颈”作为接口

但也要明确：Flamingo 不是所有多模态任务的通用最优解。如果任务非常依赖精细视觉定位、区域检测、OCR 级文字读取，单纯靠 Resampler + 插层 cross-attn 往往不够，还需要更强视觉前端或专门任务头。

---

## 参考资料

- Alayrac, Jean-Baptiste, et al. “Flamingo: a Visual Language Model for Few-Shot Learning.” arXiv:2204.14198, 2022. https://arxiv.org/abs/2204.14198
- OpenFlamingo 文档，Cross-Attention Mechanisms。https://deepwiki.com/mlfoundations/open_flamingo/2.3-cross-attention-mechanisms
- Li, Junnan, et al. “BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.” arXiv:2301.12597, 2023. https://arxiv.org/abs/2301.12597
