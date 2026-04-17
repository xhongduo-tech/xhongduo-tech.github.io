## 核心结论

SDXL 可以先记成一句定义：

`SDXL = high-resolution latent diffusion + dual text encoders + two-stage denoising`

这句话拆开看有三层意思。第一，它仍然是潜空间扩散模型。潜空间的意思是：模型不直接在原始像素上反复去噪，而是在压缩后的特征空间里工作，计算量更低。第二，它用了双文本编码器。文本编码器就是把提示词变成向量的模块，便于图像模型理解语义；SDXL 同时使用 OpenCLIP-ViT-G 和 CLIP-ViT-L，两套编码结果共同作为条件输入。第三，它把生成过程拆成两个阶段：`base` 负责主体、布局、画幅结构，`refiner` 负责边缘、材质、局部纹理。

SDXL 的核心价值不是“参数更大”，而是“高分辨率下更稳”。这里的“稳”主要指三件事：高分辨率时主体不容易散、提示词语义更容易对齐、局部细节不会在放大后明显发虚。默认目标分辨率是 `1024×1024`，这意味着它不是为了低成本草图而设计，而是为了更接近交付质量的图像输出而设计。

可以把整体流程压成一张总览图：

| 输入到输出 | 含义 |
|---|---|
| `prompt -> base -> refiner -> final image` | 先定构图，再做精修 |

玩具例子先看最直观版本。提示词是“玻璃杯里的柠檬水，木桌，窗边自然光”。如果直接把全部任务压给单阶段模型，它可能先把玻璃反光和液体高光画出来，但桌面透视、杯子位置、背景窗框不一定稳定。SDXL 的思路相反：先让 `base` 确定“这是一张近景静物图，主体居中，背景是窗边环境”，再让 `refiner` 去把玻璃边缘、液体透明感和木纹细节补出来。

真实工程例子更能看出价值。电商海报常见要求是：商品主体必须稳定、留白要能放文案、材质要清楚、输出尺寸要够大。用 SDXL 时，`base` 先把香水瓶位置、背景石材结构、主光方向和留白区域定住；`refiner` 再去补玻璃折射、金属喷头高光、瓶身边缘锐度。这样比单次长步数采样更容易稳定复现。

简化职责表如下：

| 模块 | 主要职责 | 新手理解 |
|---|---|---|
| `base` | 定主体、定布局、定大色块 | 先打草稿 |
| `refiner` | 补边缘、补材质、补纹理 | 再精修 |

---

## 问题定义与边界

SDXL 解决的问题不是“文本能不能变成图片”，而是“在较高分辨率下，如何同时保住构图、语义和细节”。这三个目标经常互相冲突。分辨率一高，模型更容易出现局部很认真、整体很松散的情况；提示词写得越长，局部对象越多，画面越容易失衡；想把细节拉满，往往会破坏整体结构。

可以先用一个问题定义表把差别说清楚：

| 问题 | 传统单阶段模型 | SDXL |
|---|---|---|
| 高分辨率构图 | 容易漂移 | 更稳定 |
| 细节层次 | 易糊 | 更细 |
| 提示词对齐 | 中等 | 更强 |

这里的“构图漂移”可以白话解释成：你想要一个主体在左侧、右侧留白的海报，模型最后却把主体画在中间，或者把背景元素塞满整个画面。“提示词对齐”则是指：文本里写的对象、关系、材质和风格，模型能否准确反映出来。

边界也要说清楚。SDXL 不是所有场景都更优。

| 场景 | 是否优先用 SDXL | 原因 |
|---|---|---|
| `512×512` 快速草图 | 否 | 速度和资源效率不划算 |
| `1024×1024` 插画或海报 | 是 | 结构和细节优势明显 |
| 商品主图、封面图 | 是 | 交付质量要求高 |
| 极低显存环境 | 否 | 模型体量和推理开销较高 |

什么时候只用 `base`？当你需要快速试构图、批量看方向、对最终锐度要求不高时，只跑 `base` 往往已经够用。什么时候用 `base + refiner`？当你准备输出最终图，或者材质、边缘、皮肤、玻璃、金属这些局部细节很关键时，两阶段更合适。什么时候不必用 `refiner`？低分辨率、强风格化草图、只做构思验证时，`refiner` 带来的收益通常不够覆盖增加的推理时间。

一个对比性的玩具例子是横版海报。提示词写“竖版电影海报，角色站左侧，右侧留出标题空间”。低分辨率单阶段模型可能先把角色脸和衣服细节做得很满，但右侧留白被背景元素侵占。SDXL 更倾向先保证“左主体右留白”的大结构，再逐步补充细节。这就是它的问题边界：它优先保证高分辨率输出的整体正确性。

---

## 核心机制与推导

SDXL 的关键设计是按噪声阶段拆分职责。噪声可以先理解为“离最终图像还有多远的混乱程度”。高噪声阶段图像几乎是随机的，这时最重要的是确定主体、布局、透视和大色块；低噪声阶段已经接近成图，这时最适合处理边缘清晰度、表面纹理、局部光影和材质反射。

因此，SDXL 不是把一个模型简单放大，而是把“去噪职责”拆开。论文和工程实现里常把这个思路描述为 `ensemble of expert denoisers`，意思是“不同阶段由擅长该阶段的去噪器接力”。

其简化写法可以表示为：

$$
z_{\alpha T} = Base(p, T, \alpha)
$$

$$
x = Refiner(p, z_{\alpha T}, T, \alpha)
$$

其中，`p` 是提示词，`T` 是总步数，$\alpha$ 是阶段切分比例。`Base` 从高噪声走到中间状态 $z_{\alpha T}$，`Refiner` 再从这个中间状态走到最终图像 $x$。如果 `num_inference_steps=40`，并设 $\alpha=0.8$，那就是 `base` 先跑前 `32` 步，`refiner` 接后 `8` 步。

条件输入也不是只有文本。SDXL 会额外接收尺寸相关条件，可以写成：

$$
c = [e_{\text{openclip}}(p), e_{\text{clip}}(p), \text{original\_size}, \text{target\_size}, \text{crop}]
$$

这里：
- `e_openclip(p)` 和 `e_clip(p)` 是两套文本编码结果。
- `original_size` 表示原始图像尺寸信息。
- `target_size` 表示目标构图尺寸。
- `crop` 表示裁剪左上角位置。

这些额外条件的意义很直接：模型不只知道“你想画什么”，还知道“你打算用什么画幅画”。这对非正方形图像很关键。比如你要做 `768×1344` 的竖版海报，如果不给尺寸条件，模型更容易按接近正方形的经验去组织主体，导致人物比例和留白失衡。

机制表可以压成下面这样：

| 阶段 | 噪声水平 | 任务 | 产出 |
|---|---|---|---|
| `base` | 高 | 定结构、定主体、定布局 | latent 中间结果 |
| `refiner` | 低 | 补边缘、补纹理、提清晰度 | 最终图像 |

玩具例子：生成“一只红色马克杯放在白桌上，右侧有一本书”。高噪声阶段真正重要的不是杯把手的高光，而是“杯子在左、书在右、桌面是俯视还是平视”。如果这个阶段结构错了，后面细节补得再漂亮也没用。低噪声阶段则适合补杯壁反光、书页边缘和桌面纹理。

真实工程例子：电商香水瓶主图。业务要求是瓶身居中偏下、上方保留品牌文案空间、背景是浅色石材、玻璃和金属材质清楚。高噪声阶段需要优先确保瓶子位置、背景透视、主副光关系与文案留白；低噪声阶段再补玻璃厚度感、液体折射、金属喷头高光。这正是职责拆分比“一次到底”更稳的原因。

---

## 代码实现

工程实现层面，最重要的是两点：第一，`base` 和 `refiner` 是接力关系，不是两个独立文本生图器；第二，尺寸参数要显式传入，尤其是非正方形图像。

先看一个最小思维模型。`base` 负责先把中间结果做出来，`refiner` 从该结果继续去噪，而不是重新从纯噪声开始。

```python
def split_steps(num_inference_steps: int, alpha: float) -> tuple[int, int]:
    assert num_inference_steps > 0
    assert 0.0 < alpha < 1.0
    base_steps = int(num_inference_steps * alpha)
    refiner_steps = num_inference_steps - base_steps
    assert base_steps > 0
    assert refiner_steps > 0
    return base_steps, refiner_steps

def should_use_refiner(width: int, height: int, final_delivery: bool) -> bool:
    assert width > 0 and height > 0
    megapixel = width * height / 1_000_000
    return final_delivery and megapixel >= 1.0

base_steps, refiner_steps = split_steps(40, 0.8)
assert (base_steps, refiner_steps) == (32, 8)
assert should_use_refiner(1024, 1024, True) is True
assert should_use_refiner(512, 512, True) is False
assert should_use_refiner(1024, 1024, False) is False
```

上面这段代码虽然不直接生成图片，但它准确表达了两阶段推理的核心逻辑：先切步数，再决定什么场景值得启用 `refiner`。

再看接近实际调用的最小框架：

```python
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

prompt = "a product hero shot of a perfume bottle on reflective stone, studio lighting"
size = (1024, 1024)

image = base(
    prompt=prompt,
    num_inference_steps=40,
    denoising_end=0.8,
    original_size=size,
    target_size=size,
).images[0]

final = refiner(
    prompt=prompt,
    image=image,
    num_inference_steps=40,
    denoising_start=0.8,
    original_size=size,
    target_size=size,
).images[0]
```

这个流程里，`denoising_end=0.8` 的意思是 `base` 在 `80%` 处停下，把后 `20%` 留给 `refiner`。对应地，`denoising_start=0.8` 表示 `refiner` 从同一位置接手。如果你把两者设错，常见结果要么是细节补不起来，要么是前后阶段职责重叠，浪费算力还不稳定。

参数说明表如下：

| 参数 | 作用 | 新手理解 |
|---|---|---|
| `denoising_end` | `base` 何时停 | 草稿做到哪一步停 |
| `denoising_start` | `refiner` 从哪一步接手 | 精修从哪一步开始 |
| `original_size` | 原始构图尺寸 | 这张图原本想画多大 |
| `target_size` | 目标输出尺寸 | 最终要交付多大 |
| `crop_coords_top_left` | 裁剪起点 | 构图从哪一块取景 |

如果要做竖版海报，建议显式写成类似：

```python
original_size = (768, 1344)
target_size = (768, 1344)
crop_coords_top_left = (0, 0)
```

不要偷懒继续用正方形默认值。因为模型会把这些尺寸条件当作构图约束的一部分，而不是纯粹的输出尺寸说明。

---

## 工程权衡与常见坑

SDXL 的优势来自职责拆分，但代价也很明确：推理更慢、显存更吃紧、参数更多、调参链路更长。单阶段模型常见的优化点主要是提示词、步数和采样器；SDXL 还额外引入 `base/refiner` 切分、尺寸条件和阶段衔接问题。

建议先用一个保守默认值开始：
- `base`：`70%~85%`
- `refiner`：`15%~30%`

`80/20` 是常见起点。原因很简单：结构通常需要更多步骤稳定下来，细节精修所需步数相对更少。如果你把 `refiner` 占比拉得太高，`base` 没把骨架搭稳，后面就会出现“边缘很认真，整体很别扭”的现象。反过来，如果 `refiner` 太少，图像会看起来完成了，但局部缺乏锐度和材质层次。

常见坑可以整理成表：

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| `refiner` 单独使用 | 生成能力弱、结果不稳 | 只把它当低噪声精修器 |
| 忽略尺寸参数 | 构图偏移 | 显式传 `original_size` / `target_size` |
| 步数切分不合理 | 结构不稳或细节不足 | 先用 `80/20`，再微调 |
| 低分辨率硬用 SDXL | 质感下降且速度慢 | 优先在 `1024×1024` 目标下使用 |

新手最容易犯的错误，是把 `refiner` 理解成“更强的第二个模型”。这不准确。它更像“只会在成图后半段工作的精修器”。如果只开 `refiner`、不给 `base` 结果，就相当于让修图师从空白画布开始作画，这不是它的职责区间。

真实工程里，尺寸参数被忽略的代价尤其高。比如你做竖版海报，最终交付是 `768×1344`，但推理时仍然沿用 `1024×1024` 相关默认参数，模型经常会把主体安排在不合适的位置，或者把留白分配错。表面看是“提示词没写好”，实质上往往是尺寸条件缺失。

还有一个现实权衡是吞吐量。批量生成时，`base + refiner` 的总成本高于只跑 `base`。如果任务是一次性探索 100 个构图方向，先只跑 `base` 更合理；如果任务是从中筛出 5 张交付图，再接 `refiner` 精修，整体效率更高。这种分层工作流通常比“每一张都全流程拉满”更符合工程成本。

---

## 替代方案与适用边界

SDXL 不是默认答案，而是一种面向高分辨率稳定输出的选择。是否使用它，取决于目标是“快”，还是“稳且细”。

选择表可以直接记成下面这样：

| 任务 | 更适合的方案 | 原因 |
|---|---|---|
| 快速草图 | 单阶段模型 | 更快、更省资源 |
| 高分辨率插画 | SDXL | 构图和细节更稳 |
| 高质量商品图 | SDXL `base + refiner` | 结构与质感兼顾 |
| 低显存设备 | 轻量模型 | 资源压力更低 |

如果目标只是验证概念，比如“这个角色穿长袍还是短夹克”“海报是冷色还是暖色”，单阶段模型更合适，因为单位时间能试更多方向。如果目标是电商主图、书籍封面、宣传海报这类最终交付图，SDXL 的优势才会真正转化为业务价值。因为最终交付关心的不只是“能看”，而是“构图稳定、材质可信、放大后不垮”。

一个简单判断规则：

- 如果目标是“快”，优先单阶段方案。
- 如果目标是“稳且细”，优先 SDXL。
- 如果目标是“高分辨率交付”，优先 `base + refiner`。

还要补一个边界：SDXL 很强，但它不是后处理超分模型，也不是万能修图器。它擅长的是在生成链路里，把高噪声结构建模和低噪声细节建模拆开，而不是替代所有上采样、修复、重绘流程。实际工程中，常见组合是“SDXL 出主图 + 专门超分或修复工具做后处理”，而不是指望单一模型包办全部质量问题。

---

## 参考资料

论文用于原理，文档用于参数，模型卡用于部署建议。

- [SDXL 论文页（Hugging Face Papers，arXiv:2307.01952）](https://huggingface.co/papers/2307.01952)
- [Stability AI 官方仓库 `generative-models` README](https://github.com/Stability-AI/generative-models)
- [Hugging Face Diffusers: Stable Diffusion XL 指南](https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl)
- [Hugging Face 模型卡：`stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Hugging Face 模型卡：`stable-diffusion-xl-refiner-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)
