## 核心结论

Gemini 的“原生多模态训练范式”，指的是模型在预训练阶段就把文本、图像、音频、视频统一变成同一条 token 序列。token 可以理解为“模型内部处理的最小离散单位”。这和 LLaVA 一类“先有语言模型，再外挂视觉适配器”的方案不同：Gemini 不是在最后一步把视觉信息塞给语言模型，而是从一开始就让不同模态在同一个 Transformer 里共同训练、共同参与每一层注意力计算。

这件事的直接意义有两点。第一，跨模态信息交换发生在网络深处的每一层，而不是只发生在输入拼接点，所以“文字指向图像区域”“音频节奏约束字幕含义”“视频帧补充文本上下文”都可以在长链路推理中持续保留。第二，统一训练目标让模型不必维护“语言头一套规则、视觉头一套规则”，而是用同一套自回归目标预测下一步 token，因此在复杂图文、UI、视频理解任务里更容易出现稳定的综合能力。

对初学者，一个足够准确的直觉是：Gemini 不是“先看图，再把图的摘要交给语言模型”，而是“看图、读字、听音频”这三件事从第一层开始就在同一个思考过程里发生。

---

## 问题定义与边界

问题的核心不是“模型能不能处理图片”，而是“能不能把不同模态放进同一种计算语言里”。模态可以理解为“信息的表现形式”，例如文本、图像、音频、视频。原生多模态训练要解决三个子问题：

| 问题 | 具体含义 | Gemini 式做法 |
| --- | --- | --- |
| 统一表示 | 不同模态原本结构不同，文本是词序列，图像是二维像素，音频是时间波形 | 先各自切分，再映射成统一 token 流 |
| 统一建模 | 文字和图像如何在同一个网络里交互 | 交错输入同一个 decoder-only Transformer |
| 统一训练 | 多模态输出如何共同优化 | 用统一自回归交叉熵损失训练 |

这里的边界也很重要。Gemini 并不是“原始像素直接喂给 Transformer”这么简单。它仍然需要模态前端，把图像切成 patch，把音频切成帧，把视频切成时空块，再转成可训练的 embedding。embedding 可以理解为“token 对应的连续向量表示”。所谓“统一”，不是说所有模态使用完全相同的分词器，而是说它们最后被投影到同一计算图里，并共享后续的注意力与训练目标。

玩具例子可以这样理解。假设输入是：

1. 文本 token：`[这个, 页面, 有, 红色, 按钮]`
2. 图像 patch token：`[左上区域, 中间区域, 红色按钮区域]`
3. 音频 token：`[提示音起点, 提示音峰值]`

如果这些 token 被排进一条序列，模型就可以在处理“红色按钮”时，直接关注“红色按钮区域”那个图像 patch，也可以顺便利用“提示音峰值”判断用户是否刚刚触发了某个操作。这种“同一序列、同一注意力”的定义，就是原生多模态的边界。

训练边界则在数据配比。公开解读通常把 Gemini 的训练描述为分阶段课程学习，也就是 curriculum。课程学习可以理解为“先学简单分布，再逐步加入复杂分布”。原因很现实：文本样本极多，图像很多，视频和音频相对更稀疏、成本更高。如果一开始就把所有模态混在一起，低频模态往往学不稳，高频模态又会主导梯度。

| 阶段 | 主体数据 | 目标 | 风险控制 |
| --- | --- | --- | --- |
| 阶段 1 | 大规模文本 | 建立语言与推理骨架 | 保证基础自回归能力 |
| 阶段 2 | 文本 + 图像/图文对 | 学会图文对齐 | 防止视觉信号只停留在浅层 |
| 阶段 3 | 文本 + 图像 + 视频/音频 | 建立时序与跨模态推理 | 控制长序列成本与模态失衡 |

所以，原生多模态不是“多接几个输入口”这么简单，而是一个统一表示、统一网络、统一损失、分阶段训练的完整范式。

---

## 核心机制与推导

Gemini 这类架构最关键的地方，是把所有模态 token 放进同一个自回归目标里。自回归可以理解为“每次根据前面内容预测下一个 token”。其统一损失可以写成：

$$
\mathcal{L}=-\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}, \text{all modalities})
$$

这里的 $x_t$ 不需要限定是文字 token、图像 token 还是音频 token。只要它被编码进统一序列，它就属于同一损失函数的一部分。这个公式的含义很直接：模型被要求在混合上下文中持续预测正确的下一步。

玩具例子最容易说明这一点。假设当前真实序列只有 3 个目标 token，模型对正确 token 的预测概率分别是：

- `text1` 的正确概率：$0.6$
- `text2` 的正确概率：$0.3$
- `image\_patch1` 的正确概率：$0.9$

那么总损失就是：

$$
-\log 0.6 - \log 0.3 - \log 0.9 \approx 0.51 + 1.20 + 0.11 = 1.82
$$

重点不在数值本身，而在“图像 token 和文本 token 使用同一套目标函数”。这和 late fusion 方案不同。late fusion 可以理解为“先各自编码，最后再合并”。那种方案里，视觉编码器和语言模型往往各自优化，真正的融合点很晚，因此跨模态约束比较弱。

第二个核心机制是跨模态注意力。注意力可以理解为“当前 token 从历史 token 里选择该看谁、看多少”。它的基础形式并不因为模态不同而改变：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中 $Q,K,V$ 分别是 query、key、value，也就是“我要找什么”“别人身上有什么索引”“别人真正提供什么内容”。原生多模态的关键不是发明一个新的注意力公式，而是让 $Q,K,V$ 都来自交错后的多模态 token 序列。于是：

- 文本 token 的 query 可以去匹配图像 patch 的 key
- 图像 token 的 query 也可以反过来匹配文本说明
- 视频帧 token 可以同时参考前文文字与前一帧视觉状态

伪代码大致如下：

```python
# x 是交错后的多模态 token embedding
for layer in transformer_layers:
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv
    scores = (Q @ K.T) / sqrt(d_k)
    weights = softmax(mask(scores))
    x = weights @ V
    x = ffn(x)
```

这里没有“专门给图像一套注意力、给文本另一套注意力”。差异只体现在 token 来源和位置编码。位置编码可以理解为“告诉模型这个 token 在序列里的位置，或在图像/时间轴上的相对位置”。模态标记则告诉模型“这个 token 属于哪一类输入”。

真实工程例子是 UI 理解。假设用户输入一句话：“点右上角设置按钮，然后读出弹窗里的版本号。”如果模型只是把截图先压缩成一段文字摘要，再交给语言模型，那么“右上角”“设置按钮”“弹窗”“版本号”这些引用关系很容易在多步推理中丢失。原生多模态模型则能让“右上角”与空间 patch 对齐，让“弹窗”与后续区域变化对齐，让“版本号”与 OCR 文本对齐，这种逐层保留的对齐关系更适合长链推理。

---

## 代码实现

如果只做一个教学版实验，最小实现并不复杂。核心流程是：

1. 各模态分别切分成 token
2. 通过各自前端转成 embedding
3. 加上位置编码和模态标记
4. 交错拼接成单一序列
5. 送入标准 Transformer decoder
6. 用统一交叉熵优化

下面这个 Python 例子不追求工业级性能，只用来说明“统一序列 + 统一损失”的计算方式。代码可直接运行：

```python
import math

def cross_entropy_of_correct_probs(probs):
    assert len(probs) > 0
    total = 0.0
    for p in probs:
        assert 0.0 < p <= 1.0
        total += -math.log(p)
    return total

def build_interleaved_tokens():
    text_tokens = ["<txt>", "red", "button"]
    image_tokens = ["<img>", "patch_12", "patch_13"]
    audio_tokens = ["<aud>", "beep_start"]

    # 一个玩具级交错序列：文本描述 -> 图像区域 -> 音频事件
    stream = [
        text_tokens[0], text_tokens[1],
        image_tokens[0], image_tokens[1],
        text_tokens[2],
        image_tokens[2],
        audio_tokens[0], audio_tokens[1],
    ]
    return stream

def modality_id(token):
    if token.startswith("<txt>") or token in {"red", "button"}:
        return "text"
    if token.startswith("<img>") or token.startswith("patch_"):
        return "image"
    if token.startswith("<aud>") or token.startswith("beep_"):
        return "audio"
    raise ValueError(f"unknown token: {token}")

stream = build_interleaved_tokens()
mods = [modality_id(tok) for tok in stream]

assert stream[0] == "<txt>"
assert "image" in mods
assert "audio" in mods

# 假设模型对真实下一 token 的预测概率
correct_probs = [0.6, 0.3, 0.9]
loss = cross_entropy_of_correct_probs(correct_probs)

assert round(loss, 2) == 1.82
print("stream:", stream)
print("modalities:", mods)
print("loss:", round(loss, 2))
```

这个例子省略了真实模型里的矩阵运算，但保留了三个关键事实：

- 多模态 token 被组织到同一条流里
- 不同模态仍然可以保留自己的标记
- 损失函数并不区分“这是文字预测还是图像预测”

如果再往前走一步，教学版的 attention 实现会长这样：

```python
import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.5, 0.5]])
V = np.array([[2.0, 0.0], [0.0, 4.0]])

out = attention(Q, K, V)
assert out.shape == (2, 2)
```

真实工程里，图像 token 通常来自 patch encoder，音频 token 来自时间帧编码器，视频 token 还要额外处理时间维。但只要它们最终进入同一个 decoder，范式上就仍然属于原生多模态。

---

## 工程权衡与常见坑

原生多模态的最大优势，是融合深；最大代价，也是融合深。因为一旦所有模态都在主干网络里共同训练，数据、算力、序列长度、采样策略就会互相影响。

先看一个公开解读中经常引用的对比方向：

| 架构 | 融合位置 | 复杂 UI/图文推理 | 延迟与成本 | 数据要求 |
| --- | --- | --- | --- | --- |
| 原生多模态 | 主干网络每一层 | 高 | 高 | 很高 |
| 拼接式/适配器式 | 输入端或少数接口层 | 中到低 | 较低 | 中 |
| 纯文本 + 预处理摘要 | 模型外部 | 低 | 最低 | 低 |

在一些面向 UI 理解的公开 benchmark 解读中，Gemini 在 ScreenSpot-Pro 这类任务上的 Deep Think 模式表现，相比传统 stitched 模型能拉开 30% 以上准确率差距；文中常引用的一组数字是 72.7% 对 36.2%。这里最值得关注的不是单个榜单，而是差距出现的位置：不是简单图片分类，而是“文字说明 + 图标语义 + 页面布局 + 多步动作”的联合推理任务。这个场景恰好放大了 native 多模态的优势。

常见坑主要有五个。

第一，模态比例失衡。文本 token 数量天然更大，视频 token 增长更快，音频 token 又容易带来高冗余。如果采样策略设计不好，模型会变成“主要学文本，顺便看点图”，或者相反，“长视频把注意力预算吃光”。这就是为什么课程学习和配比控制是核心工程问题。

第二，序列长度爆炸。单张图像切 patch、视频切多帧、再叠加文本，很快就会把上下文窗口撑满。窗口可以理解为“模型一次能同时看到的 token 上限”。窗口一满，注意力成本通常按平方增长，训练和推理都会变贵。

第三，对齐信号不足。很多团队以为“把图和文放在一起训练”就够了，但如果缺少高质量图文、视频文本、音频文本对齐数据，模型学到的只是共现，而不是可推理的对应关系。

第四，先分开训再硬拼。这个坑最常见。假设你先把视觉编码器训好，再把语言模型训好，最后只在顶部接一个 connector。connector 可以理解为“负责把两个系统粘起来的小模块”。这种方案在单图问答上常常够用，但一到多步引用、多区域定位、跨时间推理，信息瓶颈就会暴露出来。

第五，评测误导。如果只看 VQA 或图文检索这类短链任务，late fusion 和 native multimodal 的差距未必大。但在 UI 操作、长视频理解、跨模态规划、连续多轮引用这类任务中，差距通常才明显。评测集选错，会导致架构判断失真。

---

## 替代方案与适用边界

不是所有项目都该追求 Gemini 式原生多模态。工程上更合理的问题是：你的任务真的需要“深层跨模态推理”吗？

| 方案 | 适合场景 | 优点 | 局限 |
| --- | --- | --- | --- |
| 原生多模态 | 长链图文推理、UI 代理、视频理解、复杂助手 | 融合深，能力上限高 | 成本高，训练复杂 |
| Late Fusion + Adapter | 单图问答、文档理解、快速产品化 | 复用现有 LLM，开发快 | 跨模态信息瓶颈明显 |
| CLIP 式对齐 + 文本解码 | 检索、分类、轻量多模态输入 | 部署轻，移动端友好 | 细粒度推理弱 |

对新手，一个简化但不失真的理解是：

- late fusion：先各看各的，再交换摘要
- native multimodal：从一开始就在同一个思考过程中共同计算

如果你的任务只是“上传一张商品图，生成一句描述”，那原生多模态带来的收益未必覆盖成本。因为这里的跨模态依赖很短，预训练好的视觉编码器加一个语言头就足够了。

如果你的任务是“看一段产品演示视频，读配音文字，理解页面状态变化，再给出下一步操作建议”，那 late fusion 很容易在中间丢失引用链。此时原生多模态更有意义，因为它让“页面区域”“字幕内容”“动作时序”在多层注意力里持续共存。

真实工程里还要考虑部署边界。移动端、边缘端、低延迟客服系统，往往更偏向轻量架构；云端代理、研究模型、复杂自动化系统，才更值得为原生多模态投入数据和算力。架构不是越先进越好，而是要和任务深度、预算、延迟约束匹配。

---

## 参考资料

- Learnia：《Gemini 2.0 Native Multimodal: Beyond Text and Images》
- EmergentMind：《Gemini: Multimodal Transformer Innovation》
- EmergentMind：《Gemini 2.0: Advanced Multimodal Model》
- EmergentMind：《Gemini 3.0 Pro: Advanced Multimodal Transformer》
- AGI House：《Native Multimodal Architectures: Why Cross-Modal Fusion Defines the Next Defensible Moat》
- ShShell：《Multimodal Capabilities: Seeing, Hearing, and Reasoning with Gemini》
