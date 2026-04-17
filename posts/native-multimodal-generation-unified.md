## 核心结论

原生多模态生成，指的是把文本、图像、视频都变成同一种“离散 token 序列”。离散 token 可以理解为“把连续内容切成可编号的小单位”。这样，一个 decoder-only Transformer 就能像写下一句话那样，统一做“看图”“写文”“生成图”“续视频”。

这条路线的价值不只是“结构更统一”，而是训练目标也统一了。传统方案常见做法是“视觉编码器负责理解，扩散模型负责生成，LLM 负责语言”，每一段都是不同模块、不同 loss、不同推理流程。原生多模态则尝试把任务收敛成同一个目标：预测下一个 token。Emu3 给出的证据是，这个简单目标在图像、视频、文本上都能工作，而且可以和专用系统竞争。Chameleon 更早证明了另一件事：只要图文能被串成一个混合序列，单模型就能同时保留较强文本能力和非平凡图像生成能力。

给初学者的直观版，可以把它理解成“写小说和画插图用同一支笔”。以前是三个人接力：一个人读图，一个人写字，一个人画图。现在是把文字 token 和图像 token 排成一长串，模型每次只看前面内容，继续补下一个 token，直到整段图文故事写完。

| 特征 | 组合式流水线 | 原生多模态统一 token |
|---|---|---|
| 模型数量 | 视觉 encoder + LLM + diffusion | 单个 decoder 为主 |
| 训练目标 | 多阶段、异构 loss | 统一下一个 token 交叉熵 |
| 生成速度 | 常依赖多步采样 | 自回归逐 token 解码 |
| 理解/生成耦合 | 模态分离 | 同一注意力机制内协同 |
| 系统复杂度 | 高，接口多 | 低一些，但上下文压力大 |

---

## 问题定义与边界

这篇文章讨论的“原生多模态生成”，边界很明确：不是“LLM 外挂一个画图模型”，而是把多种模态先离散化，再交给同一个序列模型处理。序列模型就是按顺序读输入的模型，这里主要指 Transformer decoder。

如果把文本 token 记作 $T=[t_1,\dots,t_m]$，图像 token 记作 $V=[v_1,\dots,v_n]$，那么模型真正看到的是一个总序列：

$$
S = [t_1,\dots,t_m, v_1,\dots,v_n]
$$

或者更一般一些，是图文交错的混合序列。它不区分“这是语言阶段”还是“这是画图阶段”，只区分“当前位置下一个 token 是什么”。

边界也很现实。第一，视觉 token 数量很大，一张图远比一句话长。Emu3 的论文里，512×512 图像可被压成 4096 个视觉 token，再加上文本 token，长度很快逼近上下文上限。于是上下文长度近似是：

$$
L = n_{\text{vision}} + m_{\text{text}}
$$

如果取论文中的示意值：

$$
L = 4096 + 64 = 4160
$$

第二，tokenizer 的压缩质量直接决定成败。tokenizer 可以理解为“把图像翻译成离散编号表”的模块。压缩太狠，空间结构丢失，图像发虚；压缩不够，序列过长，Transformer 算不动。

玩具例子：想象你把一张猫的图片切成很多“图像字”，比如“耳朵”“眼睛”“胡须”“背景沙发”。真实系统当然不是这么人工分块，但直觉上相似。模型读到“橘猫、沙发、阳光”后，继续预测下一段视觉 token，就像在补全文字。

真实工程例子：一个多模态助手要完成“看商品图，回答用户问题，再生成一张营销海报草图”。组合式流水线需要视觉问答模型、文本 LLM、图像生成模型三套系统；原生多模态方案试图在一个上下文里把“商品图 token + 用户问题 token + 海报 token”直接串起来处理。

---

## 核心机制与推导

原生多模态的训练目标，本质上和语言模型一致：

$$
P(s_{k+1}\mid s_{\le k})
$$

也就是“给定前面所有 token，预测下一个 token 的概率”。这里的 $s_k$ 不再只是词，也可能是图像 token、视频 token，甚至动作 token。

对应的训练损失是交叉熵：

$$
\mathcal{L}(\theta) = -\sum_k \log P_{\theta}(s_{k+1}\mid s_{\le k})
$$

交叉熵可以理解为“如果模型给正确答案的概率越高，损失越低”。关键在于，这个公式对文本和视觉 token 一视同仁。统一，不是口号，而是 loss 函数层面的统一。

为什么这件事难？因为语言结构和视觉结构差别很大。语言主要是时间顺序，图像主要是空间关系，视频同时有时间和空间。把它们强行放进一条序列，相当于要求同一套注意力机制同时学会“句法依赖”和“空间布局”。这也是原生多模态是否成立的核心问题。

Emu3 的做法，是先用统一视觉 tokenizer 把图像和视频压成离散 token，再让 decoder-only 模型直接学习这些 token 的联合分布。论文中给出的视觉 tokenizer可把 4 帧 512×512 视频片段，或一张 512×512 图像，编码成 4096 个离散 token，来自 32768 大小的 codebook。codebook 可以理解为“图像小块的词表”。

这带来两个重要推论。

第一，理解和生成可以共用一套表示。如果模型已经学会“这组视觉 token 代表一只狗在草地上”，那么问答时它能读，生成时它也能写，不需要再切换到扩散采样器。

第二，图文交错成为自然能力，而不是后处理技巧。Chameleon 的意义就在这里。它证明了早融合，也就是文本和图像 token 从输入层面就混在一起，而不是先各算各的再拼接，这种方式可以支持任意顺序的图文输入输出。

玩具例子：输入是“请画一棵秋天的树，并在图下写一句说明”。统一序列可能先生成一串表示树叶、树干、地面的视觉 token，再接一句文本 token：“秋风让颜色先于温度抵达。”对模型来说，前半段和后半段只是同一条序列中的不同片段。

---

## 代码实现

下面用一个可运行的玩具实现，说明“统一 token 训练”到底在程序里长什么样。它不是真的图像生成器，只是把“文本 token”和“图像 token”都当成整数序列，训练一个最简单的 next-token 统计模型。重点不是效果，而是流程。

```python
from collections import defaultdict, Counter

BOS = 0

def build_bigram_model(sequences):
    counts = defaultdict(Counter)
    for seq in sequences:
        full = [BOS] + seq
        for a, b in zip(full[:-1], full[1:]):
            counts[a][b] += 1
    return counts

def predict_next(counts, prev_token):
    if prev_token not in counts or not counts[prev_token]:
        return None
    return counts[prev_token].most_common(1)[0][0]

# 1-99 看作文本 token，100-199 看作图像 token
toy_dataset = [
    [1, 2, 3, 100, 101, 102],   # "红色 苹果" -> 一组图像 token
    [1, 2, 4, 100, 101, 103],   # "红色 梨子" -> 一组图像 token
    [5, 6, 7, 110, 111, 112],   # 另一类图文样本
]

model = build_bigram_model(toy_dataset)

# 前文 token=1 后，最常见下一个 token 是 2
assert predict_next(model, 1) == 2

# 图像 token 100 后，最常见下一个 token 是 101
assert predict_next(model, 100) == 101

# 说明：同一个 next-token 机制同时作用于“文本区间”和“图像区间”
print("ok")
```

这个例子省略了神经网络，但保留了最核心的统一思想：训练阶段不关心 token 来自文本还是图像，只关心“前缀之后最可能出现什么”。

如果把它升级成真实系统，流程大致如下：

```python
def make_sequence(sample, vision_tokenizer, text_tokenizer):
    seq = []
    if "prompt" in sample:
        seq += text_tokenizer(sample["prompt"])
    if "image" in sample:
        seq += vision_tokenizer(sample["image"])
    if "caption" in sample:
        seq += text_tokenizer(sample["caption"])
    return seq

def training_step(sample, decoder, vision_tokenizer, text_tokenizer):
    tokens = make_sequence(sample, vision_tokenizer, text_tokenizer)
    logits = decoder(tokens[:-1])
    target = tokens[1:]
    loss = cross_entropy(logits, target)
    return loss
```

真实工程例子可以更具体一些。假设你在做电商内容生产系统，输入包括商品图、标题、卖点文案，输出包括详情页文案和广告配图。统一架构的好处是训练样本天然能写成：

`[商品图 token][标题 token][卖点 token][广告图 token][说明文案 token]`

模型学到的是整个业务流程中的联合分布，而不是若干模块之间的人为接口。

---

## 工程权衡与常见坑

第一类权衡，是“统一”与“冲突”的权衡。Janus 提出的核心观察是：理解任务和生成任务对视觉表示的要求并不完全相同。理解更需要细粒度、判别性强的特征；生成更需要适合离散重建和顺滑续写的表示。如果强行共享一套视觉编码路径，可能会互相牵制，尤其拖累理解效果。它的解决思路不是放弃统一，而是解耦视觉编码，再共享后端 Transformer。

| 设计 | 理解效果 | 生成流畅性 | 系统复杂度 |
|---|---|---|---|
| 共享视觉编码路径 | 可能受冲突影响 | 可以统一 | 中等 |
| 解耦视觉编码 + 统一 decoder | 更稳 | 也可保留 | 更高 |

第二类权衡，是序列长度与质量。图像和视频一旦离散化，token 数会暴涨。文本 100 个 token 很短，图像 4096 个 token 已经很长，视频更长。序列越长，显存、吞吐、训练稳定性都会变差。Emu3 为了支持视频，使用了很长的上下文，并依赖并行训练与序列 packing。packing 可以理解为“把不同样本尽量装满上下文窗口”，提高硬件利用率。

第三类常见坑，是误以为“统一 decoder 就等于实现简单”。事实并非如此。前端 tokenizer、特殊分隔 token、图文顺序设计、loss 权重平衡、推理时的采样策略，都会显著影响结果。Emu3 在论文中提到，需要对视觉 token 适当降权，否则视觉 token 数量太大，会淹没文本优化信号。

第四类坑，是把生成质量问题全归因于大模型本体。很多时候瓶颈其实在 tokenizer。若视觉 tokenizer 还原能力差，后面的 decoder 再强，也只是在预测“质量受损后的视觉字典”。

---

## 替代方案与适用边界

原生多模态不是对所有场景都更优。它更像一种“统一操作系统”式方案，适合任务经常跨模态切换、输入输出交错、长期目标是做通用多模态助手的团队。

如果任务只是感知，不涉及图像生成，比如 OCR、图像分类、图像问答，那么组合式方案通常更便宜。原因很直接：专用视觉编码器在理解任务上已经很成熟，不必为了“统一架构”承担长序列成本。

如果任务主要是高质量图片生成，且不要求和文本理解深度耦合，那么“LLM 负责提示词，扩散模型负责出图”的组合在工业上仍然有效。扩散模型指通过逐步去噪来生成图像的模型，它在高保真生成上仍有竞争力。

可以把选择逻辑写成一个简单决策表：

| 场景 | 更合适的方案 |
|---|---|
| 只做视觉理解 | 专用视觉模型或视觉编码器 + LLM |
| 只做高质量出图 | LLM + diffusion |
| 图文交错生成 | 原生多模态统一 token |
| 既要强理解又要统一生成 | Janus 式解耦编码 + 统一 decoder |

对应的伪代码是：

```python
def choose_system(task):
    if task in {"ocr", "caption", "vqa"}:
        return "perception_pipeline"
    if task in {"text_to_image", "marketing_poster"}:
        return "llm_plus_diffusion"
    return "native_multimodal_decoder"
```

玩具例子：如果你的应用只是“上传一张发票图片，输出文字”，那没必要上原生多模态。真实工程例子：如果你要做“会议助手”，它既要读 PPT 截图、理解表格、根据上下文生成补充示意图、再续写会议纪要，那么原生多模态的统一上下文会更有价值。

未来更可能出现的，不是某一派彻底胜出，而是两条路线并存。原生多模态适合追求统一能力上限；模态专用系统集成适合追求成本和局部最优。对工程团队来说，关键不是追新，而是先回答一个问题：你的产品到底需要“统一地看和写”，还是只需要“把几个成熟模块接起来”。

---

## 参考资料

- Emu3: *Multimodal learning with next-token prediction for large multimodal models*. Nature 650, 327-333 (2026). https://www.nature.com/articles/s41586-025-10041-x
- Chameleon Team: *Chameleon: Mixed-Modal Early-Fusion Foundation Models* (arXiv:2405.09818, 2024). https://arxiv.org/abs/2405.09818
- Janus: *Decoupling Visual Encoding for Unified Multimodal Understanding and Generation* (arXiv:2410.13848, 2024). https://arxiv.org/abs/2410.13848
