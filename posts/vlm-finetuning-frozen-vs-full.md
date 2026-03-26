## 核心结论

VLM（Vision-Language Model，视觉语言模型，指同时处理图像和文本的模型）微调时，是否解冻视觉编码器，不是“性能越高越好”的单选题，而是由数据量、领域偏移和显存共同决定的工程选择。

先给结论：

| 策略 | 适合数据量 | 领域偏移 | 显存压力 | 过拟合风险 | 典型做法 |
| --- | --- | --- | --- | --- | --- |
| 冻结视觉编码器，只训 Projector | 小数据 | 弱偏移 | 低 | 低 | 冻结 ViT，训练投影器，必要时微调 LLM |
| 冻结 ViT，给 LLM/ViT 加 LoRA | 中小数据 | 中等偏移 | 中低 | 中 | 只训练低秩适配器 |
| 逐层解冻 ViT + 同步调 LLM | 中大数据 | 强偏移 | 高 | 中高 | 分阶段解冻，ViT 用更小 lr |
| 全参数联合微调 | 大数据 | 强偏移 | 很高 | 高 | 多卡、梯度检查点、严格数值控制 |

核心规律是：少数据、弱域迁移时，冻结视觉编码器通常更稳，因为预训练 ViT（Vision Transformer，视觉 Transformer，用来把图像编码成特征）已经学到了通用视觉表示，只需要让投影器把视觉特征映射到 LLM（Large Language Model，大语言模型）可理解的空间。大数据、强领域迁移时，只训投影器不够，因为视觉特征本身没有贴近任务分布，这时要逐步解冻 ViT，让视觉表征发生适配。

面向新手，可以把训练流程理解成两步：

1. 先只训练投影器，让“图像特征”和“语言特征”先对齐。
2. 再逐层解冻 ViT，并继续更新 LLM，但给 ViT 更小学习率，避免梯度波动把预训练能力冲坏。

这个流程比一开始就全参数训练更稳，也比永远冻结更有上限。

---

## 问题定义与边界

问题的本质是：在有限资源下，如何平衡“稳定性”和“表征能力”。

这里的稳定性，指训练不容易发散、不容易过拟合、不容易因为小数据把预训练知识洗掉。表征能力，指模型能不能把视觉特征真正改造成适合当前任务的数据分布。

边界主要有三条。

第一条是数据量边界。  
如果只有几千张图，甚至几百张图，全参数更新通常不划算。因为可训练参数远多于有效监督信号，模型容易记住训练集而不是学到规律。此时冻结 ViT，只训练 Projector（投影器，把视觉特征转成 LLM 输入空间的小模块）是常见首选。

第二条是领域偏移边界。  
如果任务图像和预训练数据很像，比如普通商品图、自然场景问答，冻结视觉编码器往往足够。  
如果任务图像和预训练数据差异很大，比如遥感、多光谱、工业缺陷、医学影像，视觉特征本身可能就不适配，此时只训投影器很难补救，需要逐层解冻甚至联合微调 ViT。

第三条是显存边界。  
VLM 的显存瓶颈通常不在视觉侧，而在语言侧。7B 级 LLM 的参数量远大于 ViT，所以全参数训练时，权重、梯度、优化器状态的主要开销集中在 LLM。显存不足时，先考虑冻结大部分主干，只训练 Projector 或 LoRA，而不是直接全参数硬上。

玩具例子：  
假设你有一个“看图说一句话”的小任务，只有 3000 张猫狗图片，每张配一句描述。这里视觉模式很简单，预训练 ViT 已经能稳定区分猫狗。你真正缺的是“把图像特征正确接到语言模型上”的能力，所以只训练投影器就够了。  
如果你把整个 ViT 都解冻，模型可能会记住这 3000 张训练图的纹理细节，验证集反而变差。

真实工程例子：  
遥感少样本目标检测或遥感问答中，卫星图像的尺度、纹理、拍摄角度都和自然图片差异很大。若直接冻结 ViT，只训投影器，语言侧可能能读懂“有图像输入”，但视觉侧给出的仍是偏自然图像分布的特征，导致细粒度地物识别效果差。这类场景适合“两阶段流程”：先冻结 ViT/LLM 训练投影器，再逐步放开 ViT 后几层，必要时再联动更新 LLM。

可以把流程压缩成一个简洁图：

```text
阶段1：Freeze ViT + Freeze/Light Tune LLM
图像 -> ViT(冻结) -> Projector(训练) -> LLM(冻结或轻调)

阶段2：Unfreeze Top-N ViT Layers + Tune LLM
图像 -> ViT(后N层解冻, 小lr) -> Projector(继续训练) -> LLM(继续训练)
```

---

## 核心机制与推导

先看最基础的结构。设视觉编码器输出为 $h_v$，投影器为 $P$，语言侧输入嵌入空间为 $h_t$，则最简对齐过程可以写成：

$$
z = P(h_v)
$$

其中 $z$ 会被送入 LLM，与文本 token 的表示拼接或融合。如果 ViT 冻结，那么训练时真正变化的是 $P$ 和部分 LLM 参数，优化目标是在固定视觉特征上找到一个更好的语言接口。

这类方法稳定，是因为它限制了可训练自由度。直白地说：不让模型“大改底座”，就不容易把预训练能力改坏。

但它的上限也明显。如果任务所需的视觉判别边界和原始 ViT 的预训练分布差太多，那么无论投影器怎么学，输入的 $h_v$ 本身就不够好。此时需要解冻 ViT，让视觉特征自己迁移到新任务分布。

LoRA（Low-Rank Adaptation，低秩适配，一种只训练小矩阵增量而不直接改大矩阵的方法）就是两者之间的折中。  
设某层原始权重为：

$$
W_0 \in \mathbb{R}^{d \times k}
$$

LoRA 不直接更新 $W_0$，而是学习一个低秩增量：

$$
\Delta W = AB
$$

其中：

$$
A \in \mathbb{R}^{d \times r}, \quad B \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k)
$$

于是前向计算变成：

$$
W = W_0 + \Delta W = W_0 + AB
$$

如果输入是 $x$，输出为：

$$
y = (W_0 + AB)x
$$

训练时梯度主要流向 $A,B$，而不是整块 $W_0$。这意味着：

| 方法 | 梯度流向 | 可训练自由度 | 显存开销 | 适合场景 |
| --- | --- | --- | --- | --- |
| 冻结 + 投影器 | 只到 Projector | 最低 | 最低 | 少数据、弱偏移 |
| LoRA | 到低秩矩阵 $A,B$ | 中等 | 中低 | 标准 GPU、要一定适配 |
| 全参数 | 到全部权重 | 最高 | 最高 | 大数据、强偏移 |

一个简化推导可以说明 LoRA 为什么适合 VLM。  
假设 ViT 的某个 attention 线性层原始权重是 $W_0$，如果全参数训练，梯度要更新整个 $W_0$。若这个层维度是 $4096 \times 4096$，参数量就是千万级。  
如果采用 rank 为 $r=16$ 的 LoRA，则新增参数量近似是：

$$
4096 \times 16 + 16 \times 4096 = 131072
$$

相比原矩阵的 $16777216$ 个参数，小了两个数量级以上。  
这就是“只在关键方向上调整”的含义：不是完全不改模型，而是只让它在低维子空间里移动。

为什么全参数训练更容易出现梯度问题？  
因为 VLM 的视觉侧和语言侧参数规模差异很大。LLM 通常占绝大多数参数，导致优化时不同模块梯度量级不均衡。工程上常见现象是：LLM 梯度主导更新，ViT 梯度较小但对视觉分布又非常敏感。如果统一设置学习率，可能出现两种坏情况：

1. 学习率偏大，ViT 被冲坏，视觉特征崩掉。
2. 学习率偏小，LLM 学不动，整体收敛太慢。

所以联合微调时通常采用分组学习率：ViT 更小，Projector 更大，LLM 介于两者之间，并倾向使用 `float32` 保持数值稳定。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不依赖深度学习框架，只演示“冻结、逐层解冻、分组学习率、LoRA 参数量计算”这些核心控制逻辑。

```python
from dataclasses import dataclass, field

@dataclass
class Param:
    name: str
    size: int
    requires_grad: bool = True

@dataclass
class Module:
    name: str
    params: list[Param] = field(default_factory=list)

    def requires_grad_(self, flag: bool):
        for p in self.params:
            p.requires_grad = flag

    def trainable_params(self):
        return [p for p in self.params if p.requires_grad]

@dataclass
class VLMModel:
    vit_layers: list[Module]
    projector: Module
    llm: Module

def freeze_backbone(model: VLMModel):
    for layer in model.vit_layers:
        layer.requires_grad_(False)
    model.llm.requires_grad_(False)
    model.projector.requires_grad_(True)

def unfreeze_top_vit_layers(model: VLMModel, n: int):
    for layer in model.vit_layers[-n:]:
        layer.requires_grad_(True)

def count_trainable(model: VLMModel):
    total = 0
    for layer in model.vit_layers:
        total += sum(p.size for p in layer.trainable_params())
    total += sum(p.size for p in model.projector.trainable_params())
    total += sum(p.size for p in model.llm.trainable_params())
    return total

def lora_param_count(d: int, k: int, r: int) -> int:
    return d * r + r * k

# toy model
vit = [Module(f"vit_{i}", [Param(f"vit_{i}_w", 1_000_000)]) for i in range(12)]
projector = Module("projector", [Param("proj_w", 200_000)])
llm = Module("llm", [Param("llm_w", 7_000_000_000)])
model = VLMModel(vit, projector, llm)

# stage 1: freeze vit + llm, train projector
freeze_backbone(model)
stage1_trainable = count_trainable(model)
assert stage1_trainable == 200_000

# stage 2: unfreeze top-2 vit layers
unfreeze_top_vit_layers(model, 2)
stage2_trainable = count_trainable(model)
assert stage2_trainable == 2_200_000

# LoRA param comparison
full_matrix = 4096 * 4096
lora_matrix = lora_param_count(4096, 4096, 16)
assert lora_matrix < full_matrix
assert lora_matrix == 131072

print("stage1 trainable:", stage1_trainable)
print("stage2 trainable:", stage2_trainable)
print("full matrix params:", full_matrix)
print("lora params:", lora_matrix)
```

上面的断言体现了三个事实：

1. 阶段 1 只训练投影器，可训练参数最少。
2. 阶段 2 解冻顶层 ViT 后，可训练参数增加，但仍可控。
3. LoRA 的参数量远小于全参数更新。

如果换成 PyTorch，训练控制通常长这样：

```python
import torch
from torch.optim import AdamW

torch.set_default_dtype(torch.float32)

def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True

# stage 1
freeze_module(model.vit)
freeze_module(model.llm)
unfreeze_module(model.projector)

optimizer = AdamW(
    [{"params": model.projector.parameters(), "lr": 1e-4}],
    betas=(0.9, 0.95),
    weight_decay=0.01
)

# stage 2
for layer in model.vit.layers[-4:]:
    unfreeze_module(layer)

for p in model.llm.parameters():
    p.requires_grad = True

optimizer = AdamW([
    {"params": model.projector.parameters(), "lr": 1e-4},
    {"params": [p for l in model.vit.layers[-4:] for p in l.parameters()], "lr": 1e-5},
    {"params": model.llm.parameters(), "lr": 5e-6},
])
```

一个常见的工程配置如下：

| 模块 | 是否建议先冻结 | 常见学习率 | 说明 |
| --- | --- | --- | --- |
| Projector | 否 | $1e^{-4}$ 到 $5e^{-4}$ | 负责跨模态对齐，通常最先训练 |
| ViT 后几层 | 是，第二阶段再放开 | $1e^{-5}$ 到 $5e^{-6}$ | 容易破坏预训练视觉特征 |
| LLM | 视资源而定 | $5e^{-6}$ 到 $2e^{-5}$ | 参数最多，显存压力最大 |
| LoRA 参数 | 否 | $1e^{-4}$ 左右 | 参数少，通常可以给更大学习率 |

真实工程例子可以这样落地：  
做遥感问答时，第一周先冻结 ViT 和 LLM，只训练投影器验证数据链路、token 拼接、loss 下降是否正常。确认基础可训后，再解冻 ViT 最后 2 到 4 层，并在 LLM 上仅启用 LoRA。这样能在单机显存受限时获得比“只训投影器”更强的跨域适配能力，同时避免全参数训练直接爆显存。

---

## 工程权衡与常见坑

全参数微调不是“更高级的版本”，而是更昂贵、更容易出错的版本。它只有在数据和资源都足够时才值得。

先看显存。以 7B LLM 为例，若使用 FP16 存权重，单是权重就大约需要 14GB；训练时梯度通常再来一份，约 14GB；优化器状态还会继续放大占用。即使视觉侧只有几亿参数，主瓶颈仍然在语言侧。这也是为什么很多 VLM 工程项目最终都选择 LoRA、QLoRA 或冻结大部分主干。

常见坑可以直接列成表：

| 问题 | 典型现象 | 根因 | 规避方法 |
| --- | --- | --- | --- |
| 投影器过拟合 | 训练集快降，验证集不升反降 | 小数据下投影器记住样本映射 | 继续冻结 ViT/LLM，减小投影器容量，增加正则 |
| 解冻 ViT 后发散 | loss 突然爆炸，输出乱码 | 学习率过大，数值精度不足 | ViT 用更小 lr，关键更新用 float32 |
| 显存爆满 | batch size 只能到 1 或直接 OOM | LLM 参数和梯度占用过大 | LoRA/QLoRA、梯度检查点、分布式训练 |
| 梯度不均衡 | LLM 学得快，视觉侧无改善 | 模块规模差异太大 | 分组学习率、分阶段训练、梯度裁剪 |
| 训练稳定但效果上不去 | loss 正常下降，指标长期平台期 | 冻结过多，视觉表征未适配 | 解冻 ViT 顶层，或给 ViT 加 LoRA |
| 多模态接口错配 | 文本能生成，但图像信息几乎没被利用 | Projector 未对齐，拼接方式有问题 | 先单独验证 projector 对齐能力 |

这里最容易被忽略的是“梯度不均衡”。  
新手常把所有参数丢进同一个优化器组，用同一个学习率。这样做在纯 LLM 微调中有时还能勉强工作，在 VLM 上风险更高，因为视觉侧和语言侧不仅参数规模不同，训练敏感性也不同。ViT 往往需要更保守的更新步长，否则视觉表征会在前几千 step 就被破坏。

另一个常见误区是“只要全参数就一定更强”。这不成立。  
如果数据规模不足，全参数训练只是把模型容量暴露出来，不会自动变成更好的泛化能力。很多场景里，冻结 ViT + 训练投影器，或者 ViT/LLM 各自挂 LoRA，反而是更优解，因为它们在约束模型只做必要改动。

---

## 替代方案与适用边界

如果资源有限，最现实的替代方案不是“放弃微调”，而是改用参数高效微调。

QLoRA（Quantized LoRA，量化后的 LoRA，用更低位宽存底座权重，再训练低秩适配器）适合标准 GPU。直白理解是：底座模型尽量省显存保存，只训练很小的 LoRA 参数，因此能在较弱硬件上完成本来做不了的任务。

下面是常见方案对比：

| 方案 | 显存需求 | 训练复杂度 | 泛化与适配能力 | 适用边界 |
| --- | --- | --- | --- | --- |
| 只训 Projector | 最低 | 最低 | 低到中 | 数据少、偏移弱、先验证链路 |
| LoRA | 低 | 中 | 中 | 中小数据、单机 GPU 常用 |
| QLoRA | 更低 | 中高 | 中 | 显存更紧、接受量化复杂度 |
| 逐层解冻 | 中高 | 中高 | 中高 | 跨域明显，但资源还有限 |
| 全参数 | 最高 | 最高 | 上限最高 | 大数据、多卡、强领域迁移 |

新手版建议可以压缩成一句话：  
如果你只有标准 GPU，不要一开始追求全参数。先给 ViT 和 LLM 分别加 LoRA 或 QLoRA，再保留一个可训练的 Projector，通常已经能在显存、稳定性和效果之间取得更平衡的结果。

从策略上看，可以把选择分成三档：

1. 数据少且和原始预训练分布接近：冻结 ViT，训练 Projector，必要时轻调 LLM。
2. 数据中等或跨域明显：两阶段训练，先对齐，再逐层解冻 ViT 顶层，LLM 用 LoRA。
3. 数据大、跨域强、算力足：再考虑全参数联合微调。

这也是为什么“两阶段流程”经常成为默认折中方案。它不是理论上最优，而是工程上最稳：  
先用低风险方式建立跨模态对齐，再在确实需要时释放更多可训练自由度。

---

## 参考资料

1. Ridge VLM 训练策略解析（2025-11-26）  
   用途：解释冻结视觉编码器、先训投影器、再逐层解冻 ViT 的阶段化思路，以及小学习率与 `float32` 的稳定性要求。  
   链接：https://iblog.ridge-i.com/entry/2025/11/26/184501?utm_source=openai

2. EmergentMind: Fine-Tuning Vision-Language Models  
   用途：给出 LoRA/PEFT 在 VLM 中的核心机制，尤其是 $\Delta W = AB$ 的低秩更新形式，以及在 ViT/LLM 的 attention、MLP 中插入适配器的思路。  
   链接：https://www.emergentmind.com/topics/fine-tuning-vision-language-models-vlms?utm_source=openai

3. DevTechTools: LoRA vs QLoRA VRAM Efficient LLM Fine-Tuning  
   用途：说明 7B 级模型在全参数训练时的显存组成，帮助理解为什么 VLM 全参数训练的主要瓶颈通常在语言侧。  
   链接：https://devtechtools.org/en/blog/lora-vs-qlora-vram-efficient-llm-fine-tuning-production?utm_source=openai

4. MDPI 遥感相关研究（Remote Sensing）  
   用途：提供遥感少样本、强领域偏移条件下，多阶段微调与参数高效适配的现实场景参考。  
   链接：https://www.mdpi.com/2072-4292/18/2/266?utm_source=openai

5. 补充提醒  
   硬件显存、量化实现、具体 LLM/ViT 结构更新很快。实际项目应再核对当前框架版本、模型 license、量化内核支持和最新显存测算，不要直接套用旧配置。
