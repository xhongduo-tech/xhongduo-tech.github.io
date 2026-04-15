## 核心结论

SVD，完整名称是 Stable Video Diffusion，是 Stability AI 提出的图像到视频模型。图像到视频的意思是：输入一张静态图片，输出一段短视频，而不是从零开始生成整段故事视频。它的重点不是做很长、很复杂、很可控的视频，而是让一张图在短时间内稳定地“动起来”。

它的核心价值在于“稳定”。这里的稳定，不是说每一帧都几乎一样，而是说三个性质同时成立：

1. 主体身份尽量保持一致，比如人脸、产品外形、文字区域不要每帧都重画。
2. 相邻帧变化要连续，比如头发摆动、镜头推进、光影变化不能突兀跳变。
3. 整段视频的结构要闭合在一个短时间窗口内，不要前几帧正常、后几帧突然崩掉。

对零基础读者来说，可以把它理解成一句话：不是重新画 14 或 25 张互不相关的图片，而是让同一张图沿着时间轴展开成一小段连续画面。

一个典型场景是：上传一张产品主图，输出 2 到 4 秒的轻动画预览。比如商品包装轻微旋转、背景有轻微流动、人物头发轻摆、镜头缓慢推进。这个场景里，SVD 往往比“通用视频大模型”更容易得到可用结果，因为任务更窄，模型目标也更明确。

下面先用一张总览流程图把它的工作方式定住：

```text
输入图像 I
   ->
VAE 编码到 latent 空间
   ->
复制/对齐到时间轴并与噪声视频 latent 结合
   ->
时空去噪网络（U-Net + temporal layers）
   ->
逐步去噪得到视频 latent 序列 Z1:T
   ->
VAE 解码
   ->
输出视频帧 x1:T
```

结论可以先压成一张表：

| 维度 | 适合什么 | 不适合什么 |
|---|---|---|
| 输入形式 | 单图驱动短视频 | 长脚本、多镜头叙事 |
| 输出时长 | 2 到 4 秒级短片段 | 20 秒以上长时序视频 |
| 目标 | 外观保持、时间连续、轻运动 | 强动作编排、复杂剧情 |
| 控制方式 | 图像条件为主，弱文本/弱条件控制 | 精细文本导演式控制 |
| 典型用途 | 商品图动效、封面动画、人物轻微动态 | 电影镜头设计、长剧情生成 |

如果再压缩成一句结论：SVD 不是“什么视频都能生成”，而是“把一张图稳定地变成一小段视频”。

---

## 问题定义与边界

SVD 要解决的问题可以写得很明确：给定一张输入图像 $I$，生成一个短视频帧序列 $x_{1:T}$，其中 $T$ 表示帧数。帧数就是视频由多少张连续画面组成。模型要同时满足两件事：

1. 主体外观尽量保持一致。
2. 相邻帧之间的运动尽量连续。

如果只看数学形式，它解决的是：

$$
I \rightarrow x_{1:T}
$$

其中输出不是单帧图片，而是一串按时间顺序排列的画面。

更准确一点，可以写成条件生成问题：

$$
x_{1:T} \sim p_\theta(x_{1:T}\mid I, c)
$$

其中 $c$ 表示额外条件，例如帧率 `fps`、运动强度条件 `motion bucket id`、噪声增强强度 `noise_aug_strength` 等。这个式子的意思是：模型不是无条件乱生成，而是在“给定输入图和一组条件”的前提下，采样出一段视频。

这个问题的边界必须先说清，否则很容易误用模型。SVD 不是：

- 通用长视频生成器。
- 以文本为核心控制入口的复杂动作模型。
- 专门用于人物舞蹈编排、长时动作一致性的系统。
- 能自动补全完整故事分镜的“导演模型”。

它更接近“短片动效器”，而不是“完整短视频生产线”。

举两个对比：

- 适用例子：让人物照片轻微转头、头发飘动、镜头推进 3 秒。
- 超界例子：把一张人物照片直接变成 25 秒舞蹈视频，还要求动作准确、镜头切换丰富、服装不漂移。

前者是单图驱动、短时长、轻动作，符合 SVD 的设计目标。后者要求长时序、复杂人体动作和强可控性，已经超出它的边界。

边界可以压成一张表：

| 维度 | SVD 的典型边界 |
|---|---|
| 输入类型 | 单张图像为主，属于单图驱动 |
| 输出时长 | 短视频，常见是几秒内 |
| 可控性 | 弱文本控制，主要靠输入图和少量微条件 |
| 运动幅度 | 适合轻运动、中低幅度变化 |
| 适用场景 | 商品动效、头像轻动画、封面短片、镜头轻推拉 |
| 不适用场景 | 长视频叙事、强动作编排、复杂镜头调度 |

这里给一个“玩具例子”。假设输入是一张咖啡杯图片，杯子位于画面中央，背景是静态桌面。SVD 更擅长生成的结果是：镜头轻微前推、杯口热气缓慢上升、背景有一点景深漂移。它不擅长的是：让咖啡杯先跳起来，再落下，再绕桌子旋转一圈。原因不在“它不会画”，而在于它的训练目标不是长时间复杂动作控制。

再看一个新手常见误解。很多人会把“输入一张图”理解成“模型会自动脑补完整三维世界并任意拍摄”。这不对。SVD 可以表现出一定的视角变化和运动先验，但它并没有获得无限制的几何真值。输入图里本来就看不见的结构，模型只能依据训练中学到的统计规律去猜，因此猜得越多，风险越大。

所以边界判断可以用一句话完成：如果任务的核心是“保持输入图并做短时平滑变化”，SVD 通常合适；如果任务的核心是“编排复杂动作和长时间叙事”，SVD 通常不合适。

---

## 核心机制与推导

SVD 的关键机制不是“逐帧生成再拼接”，而是“把多帧一起放进 latent 空间后统一建模”。`latent` 可以理解成压缩后的特征表示，也就是先把大图片压成更小、更容易计算的内部编码，再在这个编码空间里做视频生成。

记视频帧为 $x_1, x_2, \dots, x_T$，VAE 编码器记作 $E$。则每一帧都会先被编码成：

$$
z_t = E(x_t)
$$

然后把所有帧的 latent 沿时间维堆叠成：

$$
Z = [z_1, z_2, \dots, z_T]
$$

在扩散模型里，训练目标不是直接预测最终帧，而是先把 latent 加噪，再让网络学会把噪声还原。若噪声记作 $\epsilon$，加噪后的 latent 记作 $Z_\tau$，则常见训练形式可写成：

$$
Z_\tau = \alpha_\tau Z + \sigma_\tau \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

这里：

- $\tau$ 是扩散时间步。
- $\alpha_\tau$ 控制信号保留多少。
- $\sigma_\tau$ 控制噪声注入多少。
- $\epsilon$ 是高斯噪声。

去噪网络记作 $f_\theta$。SVD 的关键在于：它不只看单帧，而是看整段视频 latent 以及条件输入，因此可以写成：

$$
\hat{\epsilon} = f_\theta(Z_\tau, I, r, m, s)
$$

其中：

- $I$ 是输入图像条件。
- $r$ 是帧率条件，`frame rate`。
- $m$ 是 `motion bucket`，表示运动强度的离散条件桶。
- $s$ 可以表示其他微条件，例如噪声增强强度。
- $\hat{\epsilon}$ 是模型预测的噪声，用来在扩散过程中逐步恢复视频。

如果新手第一次看到这个式子，重点只需要抓住两件事：

1. 模型不是一次把视频“画完”，而是反复去噪。
2. 模型不是孤立看某一帧，而是一次看整段时间。

机制图可以画成这样：

```text
视频帧 x1 ... xT
   ->
VAE 编码：z1 = E(x1), ..., zT = E(xT)
   ->
时间维堆叠：Z = [z1, z2, ..., zT]
   ->
加入噪声得到 Z_tau
   ->
时空去噪网络（spatial layers + temporal layers）
   ->
结合条件输入：输入图 I、帧率 r、motion bucket m
   ->
预测噪声并迭代去噪
   ->
解码回视频帧
```

为什么这里经常会看到 `3D U-Net`、`temporal layers`、`spatio-temporal attention` 这些词？意思不是它在输出三维模型，而是网络在处理数据时，不只处理二维空间 $(H, W)$，还显式处理时间维 $T$。对视频来说，张量形状常常可以写成：

$$
Z \in \mathbb{R}^{T \times C \times H \times W}
$$

其中：

- $T$ 是帧数。
- $C$ 是通道数。
- $H, W$ 是 latent 空间下的高和宽。

如果只做 2D 图像建模，网络每次只看某一帧的 $(C, H, W)$。如果做视频建模，网络还要看帧与帧之间如何变化，也就是时间维 $T$。

从 2D latent diffusion 扩展到 video latent diffusion，核心是两步：

1. 先沿用成熟的图像 latent diffusion 架构，保留空间建模能力。
2. 再插入 temporal layers，让网络学会时间连续性。

论文里的重要结论之一是：单靠架构还不够，数据质量同样关键。Stable Video Diffusion 论文强调了三阶段训练路线：

| 阶段 | 目标 | 作用 |
|---|---|---|
| Stage I | 图像预训练 | 先学清楚“物体长什么样” |
| Stage II | 大规模视频预训练 | 学会“物体如何随时间变化” |
| Stage III | 高质量视频微调 | 把分辨率、质感和时间稳定性拉起来 |

这个设计的工程意义很直接：可以复用图像模型已经学到的大量空间先验。空间先验就是“物体应该长成什么样”的已有知识。视频训练只需要重点补“时间如何变化”这一层，而不是从零开始学习所有图像结构。

论文还讨论了大规模视频数据清洗，尤其是切镜头检测、光流过滤、OCR 过滤、审美分数筛选等。这些步骤听起来像“数据工程”，但它直接影响结果。因为如果训练数据里充满静止画面、突兀转场、字幕覆盖和低质量运动，模型学到的时间连续性就会很差。

这里再给一个玩具推导。假设分辨率是 `576×1024`，VAE 采用常见的 `8×` 压缩，那么每帧 latent 大小可粗略估算为：

$$
H' = \lceil 576/8 \rceil = 72,\quad W' = \lceil 1024/8 \rceil = 128
$$

因此每帧 latent 的空间大小约为：

$$
72 \times 128
$$

如果视频有 14 帧，那么一段视频 latent 的时空体积与

$$
14 \times 72 \times 128
$$

成正比。如果视频有 25 帧，则体积与

$$
25 \times 72 \times 128
$$

成正比。两者时间长度之比为：

$$
\frac{25}{14} \approx 1.79
$$

这意味着模型在时间维上要处理更长的依赖，显存、计算量和稳定性压力都会上升。这也是为什么视频模型里“帧数更多”不天然等于“结果更好”。

再把这个结论翻译成最朴素的话：14 帧时，模型只需要把短时间窗口照顾好；25 帧时，它要在更长的时间跨度里持续保持脸不变、产品不扭曲、背景不乱漂，这显然更难。

真实工程例子是电商主图动效。输入一张饮料瓶商品图，团队希望自动产出首页卡片短视频：瓶身轻微转动、镜头缓推、背景水波微动。这个任务要求主体不变形、品牌字样不漂移、动作幅度克制，正好和 SVD 的设计目标吻合。如果换成“根据文本让饮料瓶完成追逐、碰撞、爆炸再转场”，那就已经不是它擅长的区域。

---

## 代码实现

代码层面，新手最需要搞清楚的是推理流程，不是训练细节。推理就是：已经有模型权重的前提下，怎样把一张图送进去，设置条件，再把输出帧保存成视频。

最小流程可以写成四步：

1. 读取输入图像并做分辨率适配。
2. 设置生成条件，比如 `num_frames`、`fps`、`motion_bucket_id`、`seed`。
3. 调用 SVD 模型得到帧序列。
4. 把帧序列编码成 `mp4` 或 `gif`。

先给一段真正可运行、并且接近官方生态的 Python 示例。它基于 Hugging Face Diffusers 的 `StableVideoDiffusionPipeline`，适合新手先跑通“从图片到视频”的闭环。

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
INPUT_PATH = "input.png"
OUTPUT_PATH = "output.mp4"

def main():
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # 显存不够时优先打开 CPU offload
    pipe.enable_model_cpu_offload()

    image = load_image(INPUT_PATH).resize((1024, 576))

    generator = torch.Generator(device="cpu").manual_seed(42)

    result = pipe(
        image,
        num_frames=25,
        fps=6,
        motion_bucket_id=127,
        noise_aug_strength=0.02,
        decode_chunk_size=8,
        generator=generator,
    )

    frames = result.frames[0]
    export_to_video(frames, OUTPUT_PATH, fps=6)
    print(f"saved to {OUTPUT_PATH}, total frames = {len(frames)}")

if __name__ == "__main__":
    main()
```

对应安装命令通常是：

```bash
pip install -U diffusers transformers accelerate imageio[ffmpeg]
```

这段代码为什么算“可运行”而不是伪代码？因为它具备完整的输入、模型加载、参数设置和视频导出步骤。你只要准备好：

- 一张 `input.png`
- 可用的 PyTorch 环境
- 足够的显存，或启用 CPU offload

就可以直接跑。

再给一段更适合理解工程量级的“玩具代码”。它不是推理器，而是帮助你在跑模型前先把尺寸、时长、时间成本估算清楚。

```python
from math import ceil

def estimate_latent_hw(height: int, width: int, vae_downsample: int = 8):
    assert height > 0 and width > 0
    assert vae_downsample > 0
    return ceil(height / vae_downsample), ceil(width / vae_downsample)

def estimate_video_seconds(num_frames: int, fps: int) -> float:
    assert num_frames > 0 and fps > 0
    return num_frames / fps

def estimate_relative_temporal_cost(base_frames: int, new_frames: int) -> float:
    assert base_frames > 0 and new_frames > 0
    return new_frames / base_frames

def recommend_profile(num_frames: int, fps: int, motion_bucket_id: int):
    assert 0 <= motion_bucket_id <= 255
    seconds = estimate_video_seconds(num_frames, fps)
    if seconds <= 4 and 96 <= motion_bucket_id <= 160:
        return "short_stable_i2v"
    return "risky_or_out_of_default_zone"

h, w = estimate_latent_hw(576, 1024)
assert (h, w) == (72, 128)

seconds_14 = estimate_video_seconds(14, 6)
assert abs(seconds_14 - (14 / 6)) < 1e-9

ratio = estimate_relative_temporal_cost(14, 25)
assert abs(ratio - (25 / 14)) < 1e-9

profile = recommend_profile(num_frames=25, fps=6, motion_bucket_id=127)
assert profile == "short_stable_i2v"

print("latent:", h, w)
print("duration:", round(seconds_14, 2), "seconds for 14 frames @ 6 fps")
print("25f vs 14f temporal ratio:", round(ratio, 2))
print("profile:", profile)
```

输出大致会是：

```text
latent: 72 128
duration: 2.33 seconds for 14 frames @ 6 fps
25f vs 14f temporal ratio: 1.79
profile: short_stable_i2v
```

如果把它映射回真实推理接口，流程通常长这样：

```python
image = load_image("input.png").resize((1024, 576))

frames = pipe(
    image,
    num_frames=25,
    fps=6,
    motion_bucket_id=127,
    noise_aug_strength=0.02,
    generator=torch.Generator(device="cpu").manual_seed(42),
).frames[0]

export_to_video(frames, "output.mp4", fps=6)
```

参数表建议先背熟：

| 参数 | 含义 | 典型作用 | 新手建议 |
|---|---|---|---|
| `num_frames` | 输出总帧数 | 决定视频长度与时间建模压力 | 先用 14 或 25 |
| `fps` | 每秒帧数 | 决定播放速度与时间感受 | 先靠近 6 |
| `motion_bucket_id` | 运动强度条件 | 影响动作幅度和风格倾向 | 先从 127 附近试 |
| `noise_aug_strength` | 输入图加噪强度 | 值越大，越不像原图，但运动可能更强 | 从很小值开始 |
| `decode_chunk_size` | 分块解码帧数 | 降低解码阶段显存占用 | 显存紧张就调小 |
| `resolution` | 输出分辨率 | 影响画质与显存占用 | 先用 576×1024 |
| `seed` | 随机种子 | 影响复现性 | 固定一个便于对比 |

有两个容易混淆的概念需要单独说明。

第一，`num_frames` 决定“生成多少帧”，`fps` 决定“这些帧如何播放”。例如：

- `14 帧 @ 6 fps`，时长约为 $14/6 \approx 2.33$ 秒。
- `25 帧 @ 6 fps`，时长约为 $25/6 \approx 4.17$ 秒。

第二，训练时固定条件和推理时播放设置不是一回事。SVD 1.1 模型卡写得很清楚：该版本是在固定 `6 FPS` 和 `Motion Bucket Id 127` 的条件上做微调，用来提高一致性。这意味着这组条件附近通常更稳，不意味着你只能用这一组参数。

因此新手最稳妥的实践不是一开始就乱改参数，而是先固定：

- `num_frames=25` 或 `14`
- `fps=6`
- `motion_bucket_id=127`
- `resolution=576×1024` 或接近官方推荐配置
- `noise_aug_strength` 用较小值

先拿到稳定结果，再做偏移实验。

下面给一个非常实用的调参顺序表：

| 目标 | 优先改哪个参数 | 原因 |
|---|---|---|
| 想让视频更长 | 先改 `num_frames` | 这是直接控制时长的参数 |
| 想让动作更明显 | 先小幅改 `motion_bucket_id` | 它直接影响运动倾向 |
| 想让结果更像输入图 | 降低 `noise_aug_strength` | 噪声越小，保真度通常越高 |
| 显存不够 | 降低 `decode_chunk_size` 或分辨率 | 先保跑通 |
| 想稳定复现 | 固定 `seed` | 方便做 A/B 对比 |

新手调参时最好的习惯不是“同时改五个参数”，而是“一次只改一个量”。否则你看到结果变化，也不知道到底是哪一个参数在起作用。

---

## 工程权衡与常见坑

SVD 的工程权衡主要发生在两个维度：空间维和时间维。空间维就是分辨率，时间维就是帧数。两者都会直接推高显存占用和计算量。

可以先看资源权衡表：

| 帧数 | 分辨率 | 显存压力 | 稳定性风险 | 适合场景 |
|---|---|---|---|---|
| 14 | 中等 | 较低 | 较低 | 快速预览、商品动效 |
| 14 | 高 | 中等 | 中等 | 高质感短片 |
| 25 | 中等 | 中高 | 中高 | 更长轻动画 |
| 25 | 高 | 高 | 高 | 实验用途，不宜默认 |

为什么 25 帧风险明显上升？因为相对 14 帧，时间长度变成约 $1.79\times$。这不是只多 11 张图这么简单，而是模型需要在更长的时间窗口里持续保持结构一致性。时间越长，抖动、漂移、局部错位就越容易积累。

可以把主要权衡压成一个二维公式感知：

$$
\text{成本} \uparrow \quad \text{通常随} \quad T \times H' \times W' \quad \text{增长}
$$

这里不是严格的显存公式，而是一个工程直觉：帧数变多、latent 分辨率变大，成本通常都会上升。

常见坑可以总结成下面这张表：

| 误区 | 后果 | 规避方式 |
|---|---|---|
| 把 SVD 当成长视频模型 | 长时序漂移、主体变形、后段质量下降 | 把任务拆成短片段，按镜头生成 |
| 只加帧数，不管分辨率 | 显存暴涨，推理失败或明显不稳 | 先定平台用途，再平衡帧数与分辨率 |
| 忽略 `fps` | 播放节奏怪异，运动观感不自然 | 优先使用默认或接近默认值 |
| 乱改 `motion_bucket_id` | 动作幅度失控，画面抖动增加 | 从 `127` 附近小范围试验 |
| 把 `noise_aug_strength` 拉太高 | 越来越不像原图，身份和结构更容易漂 | 从小值起步 |
| 输入图像本身结构混乱 | 模型难以判断主体与背景关系 | 选主体清晰、构图稳定的输入图 |
| 期待强文本控制 | 结果和提示词不一致 | 把它当 I2V，而不是强 T2V |

还有一个很现实的工程问题：输入图像质量会直接决定上限。SVD 不是魔法，它更像“把现有图像沿时间展开”。如果输入图本身已经有透视错误、边缘粘连、主体遮挡不清，生成视频时这些问题通常不会消失，反而会被时间维放大成连续伪影。

真实工程里，一个常见流程是先做输入图筛选，再跑视频生成。例如电商团队可能先要求：

- 主体完整，不被裁切。
- 背景层次清楚。
- 品牌字样清晰。
- 反光和透明区域不过度复杂。
- 主体周围没有大面积粘连阴影或脏边。

这样做的原因是：SVD 的目标是“稳定地动起来”，不是“重建一切视觉错误”。

再给一个具体例子。假设你要把一张香水瓶商品图做成首页动效：

- 如果原图是纯净背景、瓶身轮廓清楚、标签文字清晰，SVD 往往更容易得到“瓶身轻旋转 + 镜头慢推 + 高光轻微流动”的稳定结果。
- 如果原图本身就有复杂玻璃反射、背景霓虹、高对比阴影和半透明液体，模型会更容易在边缘、文字和反光区域出现跳动。

因此，工程上经常是“先把图修好，再生成视频”，而不是“指望视频模型顺便把图修好”。

还有一个常见误会是“帧插值”和“SVD 生成”混为一谈。SVD 负责创作一段短视频；帧插值负责在已有帧之间补帧，让播放更流畅。二者可以串联，但不是一回事。前者解决“生成什么”，后者解决“怎么更顺”。

---

## 替代方案与适用边界

SVD 适合的是“图像驱动的短视频生成”。如果需求变成“文本驱动的复杂动作编排”，通常就该考虑其他方案。这里最容易和它混淆的是 AnimateDiff。

AnimateDiff 的定位更像给现有文本到图像模型外挂一个 motion module。`motion module` 可以理解成一个专门补充运动信息的模块，让原本更擅长生成单张图的系统具备视频动画能力。它和 SVD 的根本区别是：SVD 是专门训练的图像到视频 latent video model，而 AnimateDiff 更接近在现有 T2I 体系上加动画能力。

对比表如下：

| 方案 | 输入类型 | 适用场景 | 控制方式 | 时长上限 |
|---|---|---|---|---|
| SVD | 单图驱动 | 商品图动效、封面动画、人物轻运动 | 图像条件为主，弱文本控制 | 短时长 |
| AnimateDiff | 文本或图像配合现有 T2I | 风格动画、社区工作流、可插拔运动模块 | 提示词、LoRA、控制模块组合 | 中短时长 |
| Text-to-Video 模型 | 文本为主 | 从文本生成复杂视频片段 | 文本控制较强 | 依模型而定 |
| 帧插值/补帧 | 已有视频帧序列 | 提升流畅度、补中间帧 | 不负责创作，只负责插值 | 依赖原视频长度 |

如果需求是以下三类，可以这样选：

- 想做“商品图转短动画预览”，选 SVD。
- 想做“文本提示驱动的风格动画视频”，更可能选 AnimateDiff。
- 想做“长时间叙事视频”，SVD 不合适，应换长时序视频模型，或采用分段生成再剪辑拼接。
- 想把已经生成好的视频变得更顺滑，选帧插值，不要用 SVD 重新生成。

再把差异说得更直白一点。

SVD 的强项是“保输入图”。如果你已经有一张满意的静态图，希望它轻微动起来，SVD 的问题定义和这个需求天然一致。

AnimateDiff 的强项是“复用文本到图像生态”。如果你已经在 Stable Diffusion 社区工作流里大量使用 prompt、LoRA、ControlNet、角色模型，那么 AnimateDiff 往往更容易接入。

而通用 Text-to-Video 的强项是“从文本直接生成片段”。它更适合“一个人在雪地里奔跑，镜头从远景推到近景”这类以文字描述为主的需求，但它未必擅长严格保持某一张输入图中的身份、构图和产品细节。

所以一句话总结适用边界：SVD 负责把一张图稳定地动起来，不负责完整电影级镜头编排。

---

## 参考资料

| 标题 | 类型 | 可引用内容 | 用途 |
|---|---|---|---|
| [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://stability.ai/research/stable-video-diffusion-scaling-latent-video-diffusion-models-to-large-datasets) | 论文页 | 三阶段训练路线、数据清洗思路、temporal layers、I2V/T2V 统一背景 | 解释为什么要从图像模型扩展到视频 latent diffusion |
| [Stable Video Diffusion 论文 PDF](https://stability.ai/s/stable_video_diffusion.pdf) | 论文 | Stage I/II/III、LVD/LVD-F 数据、时间层插入方式、数据筛选结论 | 作为核心技术来源 |
| [Stable Video Diffusion Image-to-Video Model Card](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) | 模型卡 | 14 帧版本、576×1024 分辨率、I2V 任务定位 | 确认基本输入输出边界 |
| [Stable Video Diffusion Image-to-Video XT Model Card](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) | 模型卡 | 25 帧版本、时长扩展 | 说明 14 帧与 25 帧差异 |
| [Stable Video Diffusion 1.1 Image-to-Video Model Card](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) | 模型卡 | 固定 `6 FPS` 与 `Motion Bucket Id 127` 微调、1.1 的稳定性定位 | 确认工程默认参数 |
| [Introducing Stable Video Diffusion](https://stability.ai/news-updates/stable-video-diffusion-open-ai-video-model) | 官方介绍 | 产品定位、应用方向、代码与权重发布入口 | 解释官方目标与落地场景 |
| [Diffusers: Stable Video Diffusion Guide](https://huggingface.co/docs/diffusers/main/using-diffusers/svd) | 官方文档 | `StableVideoDiffusionPipeline` 用法、`decode_chunk_size`、`motion_bucket_id`、`noise_aug_strength` | 提供可运行推理示例 |
| [AnimateDiff 官方项目页](https://animatediff.github.io/) | 项目页 | motion module 的方法定位、和 T2I 生态的关系 | 用于和 SVD 做方法对比 |
| [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models) | 官方仓库 | SVD 研究代码与采样脚本入口 | 作为实现参考 |

参考资料里最重要的三类来源分别是：论文、模型卡、官方推理文档。论文回答“为什么这样设计”，模型卡回答“这个版本能做什么”，推理文档回答“代码怎么跑起来”。这三类资料结合起来，才足够支撑工程判断。
