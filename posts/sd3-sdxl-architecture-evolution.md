## 核心结论

SDXL 和 SD3 的差异，不是“参数更多”这么简单，而是**生成主干是否还是 U-Net**这个层面的架构切换。

SDXL 仍然属于经典 latent diffusion 路线。所谓 latent diffusion，就是先把图片压缩到一个更小、更容易建模的潜空间 latent，再在 latent 上做逐步去噪，最后再解码回像素图。它的核心做法是把生成过程拆成两段：`Base` 负责高噪声阶段的主体结构，`Refiner` 负责低噪声阶段的细节修复，再配合更强的 VAE 提升高分辨率图像的还原质量。对新手可以先记成一句话：**先出草稿，再修细节**。

SD3 则换成了 MMDiT。MMDiT 是 Multimodal Diffusion Transformer，意思是“多模态扩散 Transformer”。它把图像 token 和文本 token 一起放进 Transformer 中做注意力计算，不再以 U-Net 作为主干。这里的 token 可以理解成“模型内部处理信息时使用的一段离散位置表示”，类似“把整张图和整段文字切成很多小块，再让这些小块彼此通信”。对新手可以这样理解：**不再是一个只看图像 latent 的去噪器额外接收文本提示，而是图和文从一开始就在同一个注意力系统里交互**。

两者最重要的工程区别可以先看表：

| 维度 | SDXL | SD3 |
|---|---|---|
| 主干架构 | 双阶段 U-Net（Base + Refiner） | MMDiT |
| 生成空间 | VAE latent 空间 | 仍依赖 latent/VAE，但主干对 latent 的处理方式变为 Transformer token 流 |
| 文本条件 | 文本编码结果条件化 U-Net | 三编码器文本表示直接参与多模态注意力 |
| 图像生成流程 | 先高噪声粗生成，再低噪声细化 | 单一 Transformer 主干持续去噪 |
| 重点能力 | 高分辨率细节恢复、兼容传统生态 | 更强文本理解、多模态交互 |
| 调参重心 | 分阶段调度、Refiner 接入时机 | 编码器齐全、条件结构匹配、显存管理 |
| 生态适配 | 延续 SD1.x/SD2.x/SDXL 的大量习惯 | 接口更严格，ControlNet 和 VAE 适配要求更高 |

玩具例子：输入提示词 `a red mug on a wooden desk, morning light`。  
SDXL 的常见理解方式是：Base 先画出“红色杯子在桌上”这个大结构，Refiner 再把木纹、反光、边缘细节补上。  
SD3 的理解方式更接近：文字里的 `red`、`wooden desk`、`morning light`、`mug` 这些语义片段，会在 Transformer 内持续参与图像 token 的更新，因此在复杂描述、属性绑定、局部关系上更有潜力。

如果只保留一句结论，那就是：

> **SDXL 是“分段式 U-Net 去噪流水线”，SD3 是“统一式多模态 Transformer 去噪主干”。**

---

## 问题定义与边界

这篇文章讨论的是**Stable Diffusion 从 SDXL 到 SD3 的架构演进**，重点不是出图审美，也不是各模型排行榜，而是三件事：

1. SDXL 为什么仍然属于 U-Net 体系，只是做成了 Base/Refiner 分段流水线。
2. SD3 为什么被认为是一次主干切换，即从 U-Net 过渡到多模态 Transformer。
3. 这种切换会怎样影响工程接口，尤其是 ComfyUI、ControlNet、VAE 和文本编码器的接法。

边界需要先说清楚。

第一，SDXL 不是“两个模型随便串联”。它依赖**去噪区间切分**。扩散模型每一步都在从噪声还原图像，噪声越高，模型更关注整体构图；噪声越低，模型更关注纹理和局部细节。官方常见做法是让 Base 在 `denoising_end=0.7` 结束，再把 latent 交给 Refiner，并让 Refiner 从 `denoising_start=0.7` 接手。这里的 `0.7` 不是魔法数字，而是一个阶段分界。你也可以把它改成 `0.6`、`0.8`，但切分点不同，粗结构和细节的责任边界就会改变。

第二，SD3 讨论时不能再沿用“U-Net 接文本 embedding”的老心智模型。它的文本条件不是单一编码器，而是三路：CLIP-L、CLIP-G、T5-XXL。  
CLIP 可以理解成“把文字和图片映射到共同语义空间的编码器”，更偏向图文对齐；T5 更像“擅长处理语言结构和长文本关系的文本编码器”。三者提供的不是同一种表示：有 pooled 向量，也有 sequence 序列。  
pooled 可以理解成“整句话压缩后的全局摘要”；sequence 可以理解成“逐 token 的详细表示”。前者像一句话的总意思，后者像句子里每个词分别留下的痕迹。

第三，工程上不能把 SD1.5 或 SDXL 的 ControlNet 节点直接视为 SD3 可复用组件。因为 SD3/Flux 类工作流对 VAE 端口、token 结构、条件输入类型要求更严格，尤其涉及 16 通道 VAE 或专用条件管线时，旧节点可能根本接不上。

下面这个表把边界列清楚：

| 模块 | SDXL 输入 | SD3 输入 | 噪声阶段 | 文本条件形式 |
|---|---|---|---|---|
| 生成主干 | latent + prompt 条件 | image tokens + text tokens | 全程迭代，但 SDXL 常切阶段 | SDXL 以编码条件注入；SD3 以 token 参与注意力 |
| 第二阶段细化 | Refiner 接 Base latent | 无单独 Refiner 必需结构 | 低噪声段 | 继续复用文本 token |
| VAE | 负责 latent 与像素空间转换 | 同样需要，但接口要求更严格 | 解码前后 | 与主干分离 |
| ControlNet | 传统接口较成熟 | 需 SD3/Flux 适配版本 | 与主干联动 | 需要匹配新条件结构 |
| 文本编码器 | 通常是 SDXL 约定的双编码器方案 | CLIP-L + CLIP-G + T5-XXL | 全程生效 | pooled + sequence 并存 |

新手版边界判断很简单：

- 如果你的流程图核心还是“Base 出 latent，Refiner 精修 latent”，那你仍在 SDXL 世界。
- 如果你的流程要求三编码器文本表示、SD3 兼容 ControlNet、特定 VAE 端口，那你已经进入 SD3/Flux 世界。
- 如果你看到的问题是“节点类型对不上”“某个条件输入为空”，优先怀疑的是**工作流接口不兼容**，而不是提示词本身。

为了避免概念混淆，再给一个常见误区表：

| 常见说法 | 是否准确 | 更准确的说法 |
|---|---|---|
| SDXL 有两个模型，所以它和 SD3 一样都是多阶段 | 不准确 | SDXL 是分阶段 U-Net；SD3 是单主干 Transformer |
| SD3 只是更大的 SDXL | 不准确 | SD3 的核心变化是主干架构和条件交互方式 |
| VAE 只是解码器，工程上不用太关心 | 不准确 | VAE 决定 latent 形状、通道和接口，接错会直接报错 |
| 权重能加载就说明流程兼容 | 不准确 | 节点类型、文本编码器、ControlNet、VAE 都要一起匹配 |

---

## 核心机制与推导

先看 SDXL。

SDXL 的关键不是“有两个 U-Net”，而是**两个 U-Net 在不同噪声区间做不同工作**。设总去噪步数归一化为区间 $[0,1]$，则一个常见划分是：

$$
\text{Base}: t \in [1, 0.7], \quad \text{Refiner}: t \in [0.7, 0]
$$

这里 $t$ 越大表示噪声越重。Base 在高噪声阶段更擅长建立轮廓、布局、主体关系；Refiner 在低噪声阶段更擅长纹理、边缘、局部材质。VAE 则负责潜空间与像素空间的转换，VAE 可以理解成“把大图片压缩成更容易建模的紧凑表示，再在最后还原回来”。

如果把扩散过程写得再抽象一点，可以写成：

$$
x_{t-\Delta t} = f_\theta(x_t, c, t)
$$

其中：

- $x_t$ 表示时刻 $t$ 的 noisy latent
- $c$ 表示文本条件
- $f_\theta$ 表示去噪网络

在 SDXL 中，更准确地说是：

$$
x_{t-\Delta t} =
\begin{cases}
f_{\theta_{\text{base}}}(x_t, c, t), & t \ge 0.7 \\
f_{\theta_{\text{refiner}}}(x_t, c, t), & t < 0.7
\end{cases}
$$

这说明 Base 和 Refiner 并不是“先后各跑一遍同样的事”，而是在**不同噪声区间承担不同职责**。

为什么这种分工有效？因为高噪声时，latent 里几乎没有清晰局部信息，模型很难精修纹理，这时更应该先确定“主体是谁、物体在哪、构图怎么摆”。低噪声时，主体轮廓已经稳定，模型才有条件把边缘、表面、光照细节补出来。

下面这个表把“噪声阶段”和“模型关注点”对齐：

| 噪声区间 | latent 状态 | 模型更关心什么 | 典型错误 |
|---|---|---|---|
| 高噪声 | 接近随机噪声 | 构图、主体、空间关系 | 主体缺失、物体位置错乱 |
| 中噪声 | 结构逐渐成形 | 形体、轮廓、局部关系 | 手部畸形、比例问题 |
| 低噪声 | 细节开始稳定 | 纹理、边缘、材质、光影 | 发糊、脏纹理、局部假细节 |

这就是为什么 SDXL 在高分辨率场景中表现稳定。因为高分辨率图像的问题不只是“像素更多”，而是细节恢复更容易出错。把粗结构和细节恢复拆开，工程可控性更强。

再看 SD3。

SD3 的文本条件通常可以表示为三路编码器输出。若 CLIP-L、CLIP-G 分别输出 pooled 向量 $p_l, p_g$ 和序列表示 $s_l, s_g$，T5-XXL 输出序列 $s_t$，则可以写成：

$$
P = [p_l; p_g]
$$

$$
\mathrm{Seq} = [s_l; s_g; s_t]
$$

其中 $[a;b]$ 表示拼接。$P$ 是全局摘要向量，$\mathrm{Seq}$ 是逐 token 的详细文本表示。MMDiT 的核心不是“把文本喂给图像网络”，而是把图像 token $I$ 与文本序列一起放入注意力层中：

$$
X = [I; \mathrm{Seq}]
$$

然后在 Transformer 的注意力层里更新 $X$。如果把单层注意力写成标准形式，就是：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

这里最关键的变化不是公式本身，而是 **$Q,K,V$ 都可以同时来自图像 token 和文本 token**。这意味着图像中的某个局部区域，可以直接“看见”文本里的某些词；文本里的某些词，也会影响图像 token 的更新强度。

对新手来说，可以这样理解：

- 在 U-Net 条件化里，文本更像“给图像网络一个提示”。
- 在 MMDiT 里，文本和图像更像“进入同一个会议室一起讨论”。

这带来两个直接结果。

第一，长提示词和复杂语义关系更容易被保留。比如“左侧玻璃杯折射出窗外冷色天空，右侧木桌边缘有暖色反光”，这种包含空间关系、材质、颜色和局部细节的描述，Transformer 结构通常比传统 U-Net 条件化更容易组织。

第二，控制模块必须适应这种更复杂的条件结构。因为现在条件不只是一个 embedding，而是一组多路文本表示，加上图像 token 也在同一主干里流动。

下面用表格压缩对比机制：

| Stage | 输入 token / 表示 | 主要作用 |
|---|---|---|
| SDXL Base | latent + 文本条件 | 在高噪声阶段确定构图、主体和大关系 |
| SDXL Refiner | Base 输出 latent + 文本条件 | 在低噪声阶段恢复纹理、边缘和高频细节 |
| SD3 MMDiT | image tokens + `Seq` + pooled 条件 | 在统一 Transformer 中完成图文交互式去噪 |

再补一个更细的对比表，帮助新手理解“条件化”到底变在哪里：

| 问题 | SDXL | SD3 |
|---|---|---|
| 文本信息什么时候进入主干 | 作为条件注入 U-Net | 从 token 级别进入统一注意力 |
| 图像和文本的关系 | 图像主干为主，文本提供条件 | 图文在同一 token 流里交互 |
| 长 prompt 的处理压力 | 更依赖编码和条件注入效果 | 更依赖注意力容量和编码器质量 |
| 局部语义绑定 | 能做，但经常依赖调参经验 | 理论上更自然 |

玩具例子可以更具体一点。

假设 prompt 是“蓝色陶瓷杯，杯口有蒸汽，桌面有一本翻开的书”。  
在 SDXL 中，Base 更可能先把“杯子、书、桌面”三者的位置关系搭好，Refiner 再把“陶瓷质感、蒸汽边缘、纸张纹理”补细。  
在 SD3 中，`blue`、`ceramic`、`steam`、`open book` 这些文本 token 在注意力中与图像 token 持续交互，因此“蒸汽应该出现在杯口而不是书页上”这种局部关系，理论上更容易被结构性表达。

如果把这个差异继续压缩成一句话：

> **SDXL 擅长把去噪过程分工清楚；SD3 擅长让图和文在去噪过程中持续对话。**

---

## 代码实现

先给一个最小可运行的 Python 玩具实现，用来说明 SDXL 的“阶段切分”和 SD3 的“token 拼接”不是玄学，而是明确的数据流。下面这段代码只依赖 Python 3.10+ 标准库，可以直接运行。

```python
from dataclasses import dataclass
from math import ceil
from typing import List, Sequence


@dataclass(frozen=True)
class SDXLStageConfig:
    name: str
    steps: int
    denoising_start: float | None = None
    denoising_end: float | None = None


def split_sdxl_schedule(total_steps: int, split: float = 0.7) -> tuple[SDXLStageConfig, SDXLStageConfig]:
    if total_steps <= 1:
        raise ValueError("total_steps must be > 1")
    if not (0.0 < split < 1.0):
        raise ValueError("split must be in (0, 1)")

    base_steps = ceil(total_steps * split)
    refiner_steps = total_steps - base_steps
    if refiner_steps <= 0:
        raise ValueError("split leaves no steps for refiner")

    base = SDXLStageConfig(
        name="base",
        steps=base_steps,
        denoising_end=split,
    )
    refiner = SDXLStageConfig(
        name="refiner",
        steps=refiner_steps,
        denoising_start=split,
    )
    return base, refiner


def build_sd3_text_condition(
    clip_l_seq: Sequence[str],
    clip_g_seq: Sequence[str],
    t5_seq: Sequence[str],
    clip_l_pooled: Sequence[float],
    clip_g_pooled: Sequence[float],
) -> tuple[List[float], List[str]]:
    seq = list(clip_l_seq) + list(clip_g_seq) + list(t5_seq)
    pooled = list(clip_l_pooled) + list(clip_g_pooled)
    return pooled, seq


def main() -> None:
    base_cfg, refiner_cfg = split_sdxl_schedule(total_steps=30, split=0.7)
    print("SDXL schedule")
    print(base_cfg)
    print(refiner_cfg)

    pooled, seq = build_sd3_text_condition(
        clip_l_seq=["blue", "ceramic"],
        clip_g_seq=["cup", "on", "desk"],
        t5_seq=["with", "visible", "steam"],
        clip_l_pooled=[0.11, 0.22],
        clip_g_pooled=[0.33, 0.44],
    )
    print("\nSD3 text condition")
    print("pooled:", pooled)
    print("seq:", seq)


if __name__ == "__main__":
    main()
```

运行结果会清楚展示两件事：

1. SDXL 的 Base/Refiner 是按噪声阶段分工。
2. SD3 的文本条件至少可以抽象为 `pooled + sequence` 两类表示，其中 sequence 由多编码器拼接而来。

再给一个“带伪去噪循环”的版本，帮助新手把“阶段切换”和“单主干持续更新”看得更直观。这个示例同样可以直接运行：

```python
from dataclasses import dataclass


@dataclass
class Latent:
    value: float


def base_denoise(latent: Latent, step: int) -> Latent:
    # 高噪声阶段：更大步长地确定整体结构
    return Latent(value=latent.value * 0.85)


def refiner_denoise(latent: Latent, step: int) -> Latent:
    # 低噪声阶段：更小步长地补细节
    return Latent(value=latent.value * 0.95)


def mmdit_denoise(latent: Latent, text_tokens: list[str], step: int) -> Latent:
    # 文本 token 数量越多，代表条件越丰富；这里只是玩具表达
    strength = 0.90 - min(len(text_tokens), 10) * 0.002
    return Latent(value=latent.value * strength)


def run_sdxl(total_steps: int = 10, split: float = 0.7) -> Latent:
    cut = int(total_steps * split)
    x = Latent(value=1.0)
    for step in range(total_steps):
        if step < cut:
            x = base_denoise(x, step)
        else:
            x = refiner_denoise(x, step)
    return x


def run_sd3(total_steps: int = 10) -> Latent:
    x = Latent(value=1.0)
    text_tokens = ["blue", "ceramic", "cup", "with", "steam"]
    for step in range(total_steps):
        x = mmdit_denoise(x, text_tokens, step)
    return x


print("SDXL final latent:", run_sdxl())
print("SD3 final latent:", run_sd3())
```

这段代码当然不是实际模型，但它把两条数据流表达清楚了：

- SDXL：`高噪声阶段 -> Base`，`低噪声阶段 -> Refiner`
- SD3：`整个过程 -> 同一个 MMDiT 主干`

如果用 `diffusers` 写 SDXL 的典型调用，逻辑接近下面这样。这个示例依赖 `diffusers`、`torch` 和可用 GPU，属于“真实接口示意”，不是本文前面那种零依赖玩具代码。

```python
import torch
from diffusers import DiffusionPipeline

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

prompt = "a red mug on a wooden desk, morning light, realistic"

latent = base(
    prompt=prompt,
    num_inference_steps=75,
    denoising_end=0.7,
    output_type="latent",
).images

image = refiner(
    prompt=prompt,
    num_inference_steps=75,
    denoising_start=0.7,
    image=latent,
).images[0]

image.save("sdxl_result.png")
```

这里最关键的不是 API 名字，而是两点：

1. Base 输出的是 latent，而不是最终像素图。
2. Refiner 的输入是这个 latent，并且去噪起点要和 Base 的结束点对齐。

很多新手第一次写这段代码时容易犯三个错误：

| 错误 | 现象 | 原因 |
|---|---|---|
| Base 没有 `output_type="latent"` | Refiner 接不上 | Refiner 需要 latent，不是像素图 |
| `denoising_end` 和 `denoising_start` 不对齐 | 图像发糊或收益很低 | 两阶段噪声区间断裂或重叠 |
| Refiner 没复用合适的编码器/VAE | 显存更高、结果不稳定 | 两阶段接口没有正确共享 |

再看 SD3 风格的数据流伪代码：

```python
from dataclasses import dataclass
from typing import List


@dataclass
class EncoderOutput:
    pooled: List[float]
    seq: List[str]


def concat_text_features(clipl: EncoderOutput, clipg: EncoderOutput, t5xxl: EncoderOutput):
    pooled = clipl.pooled + clipg.pooled
    seq_prompt = clipl.seq + clipg.seq + t5xxl.seq
    return pooled, seq_prompt


def fake_vae_patchify(latent_patches: List[str]) -> List[str]:
    return [f"img:{patch}" for patch in latent_patches]


def fake_mmdit(tokens: List[str], pooled_text: List[float]) -> dict:
    return {
        "token_count": len(tokens),
        "pooled_dim": len(pooled_text),
        "summary": "image tokens and text tokens processed together",
    }


clipl = EncoderOutput(pooled=[0.1, 0.2], seq=["blue", "cup"])
clipg = EncoderOutput(pooled=[0.3, 0.4], seq=["on", "desk"])
t5xxl = EncoderOutput(pooled=[], seq=["with", "visible", "steam"])

pooled, seq_prompt = concat_text_features(clipl, clipg, t5xxl)
image_tokens = fake_vae_patchify(["patch1", "patch2", "patch3"])
tokens = image_tokens + seq_prompt
pred = fake_mmdit(tokens=tokens, pooled_text=pooled)

print("pooled:", pooled)
print("tokens:", tokens)
print("pred:", pred)
```

这段伪代码虽然不生成图片，但它把 SD3 的关键数据流说清楚了：

1. 三路文本编码器先产生不同形式的文本特征。
2. 图像 latent 会进一步变成 image tokens。
3. image tokens 和 text tokens 在同一主干中一起计算。

真实工程例子是 ComfyUI 里的 SD3.5 Large ControlNet 工作流。比如你要做“草图约束 + 深度约束”的商品图重绘，在 SDXL 时代通常是 Base/Refiner 配合已有 ControlNet 节点；到了 SD3.5 Large ControlNet，你需要确认：

1. 工作流节点是 SD3/Flux 兼容版本。
2. 文本编码链路提供 CLIP-L、CLIP-G 和 T5-XXL 对应输出。
3. VAE 类型和通道配置匹配模型要求。
4. Apply ControlNet 节点版本与当前工作流接口一致。

否则即使权重文件加载成功，图也不一定能跑通。

最后用一个表把两段代码背后的差异压缩一下：

| 项目 | SDXL 示例 | SD3 示例 |
|---|---|---|
| 代码重心 | 调度切分 | 条件拼接 |
| 核心数据对象 | latent | image tokens + text tokens |
| 关键接口 | `denoising_end/start` | 编码器输出结构是否齐全 |
| 新手最该检查什么 | Refiner 接入点是否对齐 | 文本编码器和节点类型是否匹配 |

---

## 工程权衡与常见坑

SDXL 的优势是**熟悉、稳定、可控**。你能明确调 `denoising_end` 和 `denoising_start`，能在 Base-only、Base+Refiner 之间权衡速度与质量，也更容易复用已有高分辨率工作流。但它的代价是流程更像“分段管线”，不是单一主干；当提示词特别长、局部语义约束很复杂时，它的表达能力更依赖经验调参。

SD3 的优势是**统一主干下的多模态交互更强**。长文本、复杂语义、图文协同控制更有潜力。但它的代价也很直接：显存压力更高，接口更复杂，生态兼容期更痛苦。

如果用工程语言归纳，两者的权衡可以写成下面这张表：

| 维度 | SDXL | SD3 |
|---|---|---|
| 训练与推理主干 | U-Net 家族，社区经验多 | Transformer 主干，接口新 |
| 显存压力 | 相对友好 | 通常更高 |
| 工作流可解释性 | 强，分段清楚 | 更依赖整体链路理解 |
| 生态成熟度 | 高 | 仍在适配和演化 |
| 长 prompt 处理 | 可用，但更吃经验 | 更有结构优势 |
| 迁移成本 | 低 | 高 |

最常见的坑如下：

| 坑 | 典型症状 | 解决方案 |
|---|---|---|
| 把 SDXL ControlNet 直接接到 SD3 | 节点报错、条件为空、`y is None` | 使用 SD3/Flux 兼容 ControlNet 与对应工作流 |
| VAE 通道不匹配 | 推理时报类型或 shape 错误 | 使用模型要求的 VAE，确认端口类型一致 |
| 只加载部分文本编码器 | 图能跑，但语义跟随明显变差 | 确认 CLIP-L、CLIP-G、T5-XXL 都已正确接入 |
| SDXL Refiner 接错阶段 | 图像发糊、细节怪异或收益极低 | 对齐 `denoising_end` 与 `denoising_start` |
| 误把 Base-only 结果当成 SDXL 全流程质量 | 觉得 SDXL 细节差 | 区分 Base-only 与 Base+Refiner 的输出预期 |
| 误把“权重兼容”当成“工作流兼容” | 文件能加载，但流程跑不通 | 同时检查节点类型、编码器、VAE、ControlNet |
| 长 prompt 只堆词不控结构 | 局部属性错绑 | 适当分句、减少冲突描述、检查编码器支持 |

新手最容易犯的错误，是把“模型权重兼容”理解成“工作流兼容”。这是错的。权重能下载，不代表节点接口、VAE 类型、文本编码器输出结构能自动对齐。

再给一个真实工程判断标准。  
如果你在 ComfyUI 中把旧版 SD1.5/SDXL 的 ControlNet 节点拖到 SD3 工作流里，然后报输入类型错误，问题通常不在“提示词写错了”，而在于**工作流的条件结构已经变了**。正确做法一般不是反复改 prompt，而是检查节点版本、VAE 接法和 ControlNet 适配版本。

还可以把排错顺序写成一个更实用的检查单：

1. 先看主干是不是同一代架构。SDXL 节点不一定能进 SD3。
2. 再看文本编码器是不是齐全。缺一路，结果都可能劣化。
3. 再看 VAE 是否匹配。这里最容易出现 shape 和通道错误。
4. 最后才看 prompt 和采样器参数。很多时候真正的问题根本不在提示词。

一个很典型的误判是：  
“图出不来，说明 prompt 写得差。”  
更常见的真实原因是：  
“图出不来，是因为节点输入类型、VAE 或 ControlNet 结构不兼容。”

---

## 替代方案与适用边界

如果你的目标是**高分辨率出图、工作流稳定、复用现有生态**，SDXL 仍然是非常合理的选择。尤其是已有 LoRA、ControlNet、放大、修脸、后处理流水线都围绕 U-Net 体系构建时，继续使用 SDXL 的迁移成本最低。

如果你的目标是**更强的文本理解、更复杂的控制关系、愿意接受新接口和更高资源需求**，SD3 更值得投入。它不是简单升级版 SDXL，而是另一套主干哲学。

下面这个表可以直接作为选型规则：

| 场景 | 更适合 SDXL | 更适合 SD3 |
|---|---|---|
| 资源需求 | 显存更友好，旧设备更容易跑 | 通常更重，对显存和链路要求更高 |
| 提示长度 | 中短 prompt 足够稳定 | 长 prompt、复杂语义更有优势 |
| 控制精度 | 传统生态成熟 | 新一代控制方式潜力更强，但更挑接口 |
| 兼容生态 | 与既有 U-Net 工作流兼容度高 | 需要适配新的节点和模型结构 |
| 上手成本 | 低 | 较高 |

可以把选择策略压缩成三句话。

第一，只想快速生成高质量图片，并且你已经有 SDXL 工作流，继续用 SDXL，甚至只用 Base-only 都可能足够。  
第二，如果你确实需要更强的复杂提示理解，再考虑 Base+Refiner 全流程。  
第三，如果你需要的是“更深的图文联合建模”，例如复杂草图、深度图、长文本语义同时参与控制，再考虑 SD3 或 SD3.5 生态。

为了让选择边界更清楚，再补一个“任务类型”表：

| 任务类型 | 更稳妥的选择 | 原因 |
|---|---|---|
| 博客封面、插画草图、头像生成 | SDXL | 成本低、生态成熟、速度更友好 |
| 海报概念图、产品渲染、写实人像 | SDXL 或 SD3，取决于控制复杂度 | 如果控制关系不复杂，SDXL 已够用 |
| 长 prompt 场景合成 | SD3 | 语义绑定更有优势 |
| 多条件联合控制 | SD3/Flux | 条件表达潜力更高，但链路更复杂 |
| 老工作流维护 | SDXL | 迁移风险最小 |
| 新项目探索 | SD3 | 值得围绕新接口设计流程 |

玩具例子：你只是要给博客封面生成“蓝色芯片插画，白底，简洁科技风”，SDXL 就够。  
真实工程例子：你要做电商主图重绘，输入包括产品草图、深度图、长提示词、材质约束和局部文字语义，这时 SD3/Flux 路线更值得尝试。

最后给一个最简单的决策口诀：

- 已有成熟工作流，优先 SDXL。
- 需要复杂语义和联合控制，再看 SD3。
- 先判断接口成本，再判断模型能力，不要反过来。

---

## 参考资料

下面列的不是“随手搜到的二手解读”，而是按用途划分的阅读入口。看资料时，优先读官方文档和原始论文，再读工程侧解析。

| 来源 | 关注点 | 用途 |
|---|---|---|
| Hugging Face Diffusers SDXL 文档 | Base/Refiner 流程、`denoising_end/start`、SDXL 使用方式 | 查 SDXL 参数和标准调用方式 |
| Stability AI Stable Diffusion 3 技术报告 / 研究稿 | MMDiT、三编码器、多模态注意力 | 理解 SD3 的架构定义 |
| Stability AI 发布说明与模型卡 | 模型组件、编码器、使用边界 | 对照实际模型要求 |
| Hugging Face Diffusers SD3 文档与示例 | 推理接口、编码器接法、pipeline 约定 | 理解工程落地方式 |
| ComfyUI 官方文档与节点说明 | 节点端口、模型加载、工作流迁移 | 排查工作流适配问题 |
| Comfy Blog: SD3.5 Large ControlNet | SD3.5 Large ControlNet 工作流和应用场景 | 看真实工程接法 |
| 社区架构解析文章 | 三编码器组合、pooled 与 sequence 形式、工程解释 | 帮助建立整体心智模型 |
| 模型卡中的 VAE / Text Encoder 配置说明 | 通道数、编码器类型、依赖关系 | 定位 shape 和接口错误 |
| ControlNet 适配说明或仓库 README | 控制分支是否兼容 SD3/Flux | 避免旧版节点直接复用 |

如果你只打算精读三类资料，建议顺序是：

1. 先读 SDXL 官方或 Diffusers 文档，理解 Base/Refiner 的阶段切分。
2. 再读 SD3 技术报告，理解 MMDiT 和三编码器结构。
3. 最后看 ComfyUI / ControlNet 的工程文档，把“论文里的结构”映射到“实际节点接口”。

这样阅读最不容易混淆“模型原理”和“工作流实现”。
