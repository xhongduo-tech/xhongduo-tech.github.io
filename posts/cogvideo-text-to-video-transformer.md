## 核心结论

CogVideo 这条技术线的价值，不是“把图像模型改成一次输出很多帧”，而是把已经在文本生成、图像生成中验证过的 Transformer 路线迁移到视频生成，并专门处理时间维带来的序列长度爆炸、时序一致性和推理成本问题。这里必须先区分两个名字相近、但生成机制不同的版本：

| 维度 | 原版 CogVideo | CogVideoX |
|---|---|---|
| 生成范式 | 自回归生成 | 扩散生成 |
| 表示空间 | 直接面向视频 token / 帧序列 | 先压缩到 latent，再生成 |
| 核心模块 | 预训练 Transformer、分层帧率、递归插帧、局部注意力 | 3D VAE、Diffusion Transformer、expert transformer |
| 训练目标 | 从文本条件生成连续帧序列 | 在压缩视频表示上做条件扩散去噪 |
| 推理特征 | 先粗后细，先关键帧再插帧 | 先采样 latent，再统一解码成视频 |
| 主要工程压力 | 长序列自回归慢、注意力成本高 | 扩散采样慢、VAE 压缩质量决定上限 |

“自回归”可以理解为：像写句子一样，一个 token 接一个 token 地往后生成；前面已经生成的内容会作为后面的条件。“扩散”可以理解为：先从噪声开始，再经过多步去噪逐渐逼近目标视频。这两个目标函数不同、训练方式不同、推理路径也不同，所以不能把原版 CogVideo 和 CogVideoX 当成同一个架构。

如果只记一句话，可以记成：

- 原版 CogVideo 解决的是：如何把图像生成里的 Transformer 路线扩展到短视频生成。
- CogVideoX 解决的是：如何在更短、更稳定、更省算力的 latent 时空表示里生成更高质量视频。

从整体链路看，两者都在做同一件事：在文本条件 $c$ 下生成视频 $x$。只是中间表示不同。统一写法可以表示为：

$$
x \xrightarrow{E} z,\quad z \sim G(z \mid c),\quad \hat{x} = D(z)
$$

其中：

- $E$ 是编码器，把原始视频压缩成更短的内部表示。
- $G$ 是生成模型，在文本条件下生成内部表示。
- $D$ 是解码器，把内部表示还原回像素视频。

原版 CogVideo 更接近直接在视频序列上建模；CogVideoX 则明确采用“先压缩、再生成、再解码”的路线。

玩具例子：prompt 是“一个红球从左滚到右”。单帧图像生成只需要画出一个红球；视频生成还要保证红球颜色不漂、位置连续、背景不闪烁、运动速度合理。原版 CogVideo 更像“先给出几个关键状态，再补中间过程”；CogVideoX 更像“先在压缩后的时空表示里把整个运动过程建好，再统一解码”。

真实工程例子：做一个 5 秒产品开箱短视频，目标不是“每一帧都像海报”，而是“手部动作连续、产品外观稳定、镜头运动平滑、显存成本可控”。CogVideo 家族真正处理的是这个问题。

---

## 问题定义与边界

本文讨论的问题是：给定文本提示词，生成一段连续视频，而不是一张静态图片。形式化写法是，输入文本条件 $c$，输出视频张量：

$$
x \in \mathbb{R}^{T \times H \times W \times C}
$$

其中：

- $T$ 是帧数，也就是视频里一共有多少张连续图片。
- $H, W$ 是每一帧的高和宽。
- $C$ 是通道数，通常 RGB 图像里 $C=3$。

图像生成只需要建模 $H \times W \times C$；视频生成还要额外乘一个时间长度 $T$。如果一张图是 `512 × 512 × 3`，80 帧视频的原始像素规模就是单张图的 80 倍。难点不只是“多了 80 张图”，而是这些图之间必须满足连续关系。

真正难的是下面四件事要同时成立：

| 项目 | 含义 | 为什么难 |
|---|---|---|
| 语义对齐 | 画面要符合 prompt | 文本约束必须跨多帧保持一致 |
| 时序一致 | 相邻帧动作要连续 | 每帧独立生成会抖动、漂移、闪烁 |
| 节奏控制 | fps、时长、镜头运动要合理 | 同样的帧数在不同 fps 下观感完全不同 |
| 计算可承受 | 显存、算力、训练稳定性可接受 | 时空序列过长，自注意力成本迅速上升 |

“时序一致”可以直接理解为：下一帧不能像重新画了一张不相关的图。  
“fps”是每秒帧数。例如：

$$
\text{duration} \approx \frac{T - 1}{\text{fps}}
$$

如果 $T=81$、fps 为 16，那么视频时长约为：

$$
\frac{81 - 1}{16} = 5 \text{ 秒}
$$

这也是为什么视频生成里不能只讨论“帧数”，而必须把帧数、fps 和时长一起看。

本文边界如下：

| 维度 | 本文覆盖 | 本文不展开 |
|---|---|---|
| 输入 | 文本 prompt | 视频编辑、图生视频、逐帧控制 |
| 输出 | 短时文本到视频 | 超长叙事视频 |
| 目标 | 理解 CogVideo 与 CogVideoX 的架构逻辑 | 全部视频模型横向综述 |
| 重点 | 分层生成、注意力、latent 压缩、训练策略 | 商业产品效果排名 |

新手最常见的误解是：“视频生成不就是多生成几张图吗？”这不准确。  
如果每一帧都独立生成“狗在跑”，你得到的通常不是视频，而是一组主题相近、但姿态、背景、颜色不断跳变的图片。CogVideo 家族真正要解决的是跨帧一致性，而不是单帧好不好看。

---

## 核心机制与推导

先看原版 CogVideo。它的重要思路可以概括成三件事：

1. 把视频看成一个更长的序列。
2. 不直接一次生成完整高帧率视频，而是分层生成。
3. 用局部注意力而不是全局两两连接，控制复杂度。

### 1. 原版 CogVideo：分层帧率 + 顺序生成 + 递归插帧

“分层帧率”可以理解为：先在粗时间网格上生成关键内容，再补中间帧。原因很直接，如果一开始就要求模型在高帧率下同时决定几十上百帧，序列长度会非常长，训练和推理都会变重。

可以形式化写成：

$$
y_{\text{coarse}} = G_1(c, r)
$$

$$
y_{\text{fine}} = G_2(y_{\text{coarse}}, r/2)
$$

其中：

- $c$ 是文本条件。
- $r$ 是较粗的采样帧率。
- $G_1$ 先生成低帧率关键帧序列。
- $G_2$ 以关键帧为条件，继续插入中间帧。

白话讲，第一阶段先回答“这段视频大概发生了什么”，第二阶段再回答“中间如何平滑过渡”。

玩具例子：要生成“小球从桌子左边滚到右边”。

- 如果先生成第 1、5、9 帧这些关键位置，再补 2、3、4、6、7、8 帧，轨迹更容易连续。
- 如果直接要求模型独立决定 1 到 9 帧，小球位置就容易忽左忽右，甚至大小和颜色都漂。

这就是“粗骨架 + 细补全”的思路。它不是把问题变简单了，而是把一个难问题拆成两个更容易训练的子问题。

### 2. 为什么注意力会成为瓶颈

Transformer 的核心是注意力机制。简单说，每个位置都要决定“我应该关注序列里的哪些其他位置”。如果采用全局自注意力，长度为 $L$ 的序列复杂度近似为：

$$
O(L^2)
$$

在视频里，$L$ 会非常大。因为视频 token 数量大致来自：

$$
L \propto T \times \frac{H}{p} \times \frac{W}{p}
$$

其中 $p$ 是 patch 尺寸。时间变长、分辨率变高，$L$ 都会迅速增长。  
一旦 $L$ 增大，全局注意力中的“任意两个 token 都两两交互”就会带来平方级成本。

因此原版 CogVideo 会采用局部或滑动窗口注意力。假设每个 token 只看窗口大小为 $w$ 的邻域，那么复杂度可近似写成：

$$
O(L \cdot w)
$$

这里的关键不是“模型只理解局部”，而是“主要计算集中在局部邻域，避免全局两两交互”。这是一种工程降复杂度手段，不应被误读成绝对的可解释性结论。

### 3. CogVideoX：3D VAE 压缩 + latent 扩散 + expert transformer

CogVideoX 的路线发生了明显变化。它不再直接在像素级长序列上建模，而是先把视频压缩到 latent 空间，再在 latent 上做扩散生成。

第一步是 3D VAE 编码：

$$
z = E(x) \in \mathbb{R}^{T' \times H' \times W' \times d}
$$

其中：

- $T' < T$，时间维被压缩。
- $H' < H, W' < W$，空间维也被压缩。
- $d$ 是 latent 通道数。

把它展平成序列后，长度变成：

$$
L = T' \cdot H' \cdot W'
$$

由于 $T', H', W'$ 都比原始视频小得多，模型处理的序列长度显著下降。这一步是 CogVideoX 工程可行性的基础。

扩散过程可以写成更标准的形式。训练时，对 latent $z_0$ 加噪得到：

$$
z_t = \sqrt{\alpha_t} z_0 + \sqrt{1-\alpha_t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

模型学习在文本条件 $c$ 下预测噪声或等价目标：

$$
\mathcal{L} = \mathbb{E}_{z_0,\epsilon,t,c}\left[\left\|\epsilon - \epsilon_\theta(z_t, t, c)\right\|_2^2\right]
$$

它的含义并不复杂：  
训练时故意把真实视频的 latent 搞脏，再让模型学会“在知道文本条件的前提下，怎么把噪声方向估计回来”。推理时从纯噪声开始，反复去噪，最后得到可解码的 latent 视频。

“expert transformer”可以理解为：仍然保留 Transformer 主体，但通过更适合时空建模的结构安排，提高容量利用率和训练效率。重点不是换一个完全不同的数学体系，而是在 Transformer 框架里针对视频任务做专门优化。

真实工程例子：生成“无人机绕着一辆银色汽车环拍 5 秒”。

- 如果直接在像素空间逐帧生成，模型要同时记住车体颜色、金属反光、背景透视变化、镜头位移、轮胎细节，难度很高。
- 如果先压到 latent 空间，模型更容易先学习“同一辆银色车 + 环绕镜头 + 连续背景变化”的整体结构，再由解码器还原局部细节。

### 4. 两条技术线的机制差异

| 维度 | 原版 CogVideo | CogVideoX |
|---|---|---|
| 生成空间 | 视频 token / 帧序列 | 压缩后的 latent 序列 |
| 时间建模 | 多帧率递归插帧 | 3D latent 上统一时空建模 |
| 训练目标 | 自回归预测下一个 token / 帧 | 扩散去噪 |
| 注意力策略 | 局部/滑窗/Swin 类思路 | 在更短 latent 序列上做 Transformer |
| 主要瓶颈 | 长序列自回归慢、显存重 | VAE 压缩质量、扩散采样成本 |
| 迁移路径 | 从图像 Transformer 扩展到视频 | 从视频压缩表示出发重建视频生成体系 |

如果把两者压缩成一句技术判断：

- 原版 CogVideo 的核心贡献是“把 Transformer 视频生成这条路走通”。
- CogVideoX 的核心贡献是“把视频生成从直接长序列建模，转向更现代的 latent diffusion 视频建模”。

---

## 代码实现

实现层不要先记库名，先把主流程看清楚。无论是原版 CogVideo 还是 CogVideoX，工程链路都可以拆成四步：

1. 输入预处理。
2. 序列或 latent 表示构造。
3. 模型推理。
4. 视频解码与导出。

对于 CogVideoX 风格流程，可以写成：

1. 读取 prompt，并做必要的 prompt 重写。
2. 根据模型约束对齐帧数，例如 `16N+1`。
3. 文本编码得到条件表示。
4. 在 latent 空间做扩散采样。
5. 用 VAE 解码得到视频帧。
6. 写出 mp4 或帧序列。

这里的 `16N+1` 不是随手定的数字，而是很多视频模型在训练时采用的时间网格约束。  
例如：

$$
81 = 16 \times 5 + 1
$$

表示 81 帧恰好落在该时间网格上。对于目标 5 秒、16 fps 的视频，81 帧是常见选择，因为：

$$
\frac{81 - 1}{16} = 5
$$

### 1. 可运行的最小 Python 示例

下面这段代码只依赖 Python 标准库，可以直接运行。它不复刻真实模型，而是把“帧数对齐、时长计算、参数校验、阶段规划”写清楚。

```python
from dataclasses import dataclass, asdict
import json


@dataclass
class VideoRequest:
    prompt: str
    seconds: float
    fps: int
    frame_rule_stride: int = 16  # 常见时间网格之一：16N + 1


def align_num_frames(seconds: float, fps: int, stride: int = 16) -> int:
    if seconds <= 0:
        raise ValueError("seconds must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    raw_frames = int(round(seconds * fps))
    raw_frames = max(raw_frames, 1)

    if raw_frames == 1:
        return 1

    n = (raw_frames - 1 + stride - 1) // stride
    return n * stride + 1


def validate_num_frames(num_frames: int, stride: int = 16) -> bool:
    return num_frames >= 1 and (num_frames - 1) % stride == 0


def estimate_duration(num_frames: int, fps: int) -> float:
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    return (num_frames - 1) / fps


def build_generation_plan(req: VideoRequest) -> dict:
    if not req.prompt.strip():
        raise ValueError("prompt must not be empty")

    num_frames = align_num_frames(req.seconds, req.fps, req.frame_rule_stride)
    duration = estimate_duration(num_frames, req.fps)

    return {
        "request": asdict(req),
        "num_frames": num_frames,
        "estimated_duration": duration,
        "valid_frame_rule": validate_num_frames(num_frames, req.frame_rule_stride),
        "pipeline": [
            "prompt_rewrite",
            "text_encoder",
            "diffusion_transformer_sampler",
            "vae_decoder",
            "video_writer",
        ],
    }


def main() -> None:
    req = VideoRequest(
        prompt="A silver car rotating on a turntable, studio lighting, smooth camera motion",
        seconds=5,
        fps=16,
    )
    plan = build_generation_plan(req)

    assert plan["num_frames"] == 81
    assert abs(plan["estimated_duration"] - 5.0) < 1e-9
    assert plan["valid_frame_rule"] is True

    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

这段代码解决的是三个新手最容易犯错的问题：

| 问题 | 错误写法 | 结果 |
|---|---|---|
| 帧数不对齐 | 直接写 `80` 帧 | 可能触发模型内部 shape 或 pack 错误 |
| 时长理解错误 | 以为 `num_frames / fps` 就一定是最终时长 | 忽略端点组织方式 |
| 输入未校验 | prompt 为空、fps 为 0 也继续调用 | 问题在模型内部才暴露，难排查 |

### 2. 接近真实推理流程的伪代码

下面的伪代码更接近 CogVideoX 类模型的实际使用方式：

```python
prompt = """
Close-up shot of a robot opening a product box on a wooden desk,
camera slowly pushes in, product remains centered, soft studio lighting
""".strip()

num_frames = align_num_frames(seconds=5, fps=16, stride=16)
assert validate_num_frames(num_frames, 16)

text_cond = text_encoder(prompt)

latent_video = diffusion_transformer.sample(
    text_cond=text_cond,
    num_frames=num_frames,
    guidance_scale=6.0,
    num_inference_steps=50,
)

video = vae_decoder.decode(latent_video)
save_video(video, path="output.mp4", fps=16)
```

虽然这段代码是伪代码，但主流程是真实的：

- `text_encoder` 把 prompt 变成条件向量。
- `diffusion_transformer.sample` 在 latent 空间从噪声开始逐步采样。
- `vae_decoder.decode` 把 latent 视频还原到像素空间。
- `save_video` 负责导出。

### 3. Prompt 为什么会直接影响视频质量

很多新手会把 prompt 写得过短，例如只写：

```text
a cat
```

这通常不足以约束视频。因为视频不是只决定“画什么”，还要决定“怎么动、镜头怎么走、背景怎么保持一致”。更合理的写法是把主体、动作、镜头、场景一起写清楚：

| 维度 | 差 prompt | 较好 prompt |
|---|---|---|
| 主体 | `a phone` | `a black smartphone with metallic edges` |
| 动作 | 无 | `hands open the box and lift the phone slowly` |
| 镜头 | 无 | `close-up shot, camera slowly pushes in` |
| 构图 | 无 | `product remains centered in frame` |
| 光线 | 无 | `soft studio lighting, clean background` |

真实工程例子：电商团队做“产品开箱 5 秒演示”，如果只写“show product”，模型大概率会生成静止镜头或者动作不明确的视频。可用 prompt 通常要同时写清：

- 主体是什么
- 动作怎么发生
- 镜头如何移动
- 背景和光线是什么
- 哪些对象必须保持稳定

### 4. 原版 CogVideo 与 CogVideoX 的实现关注点不同

| 模块 | 原版 CogVideo 更关注 | CogVideoX 更关注 |
|---|---|---|
| 文本条件 | 从预训练文本/图像模型迁移能力 | caption 质量与 prompt 对齐 |
| 时序建模 | 递归插帧、局部注意力 | latent 扩散、时空 token 排布 |
| 帧数控制 | 分阶段生成的时间层级 | `16N+1` 等时间网格 |
| 推理瓶颈 | 自回归速度慢 | 扩散步数、VAE 解码耗时 |
| 部署优化 | local attention kernel | 量化、低显存采样、解码优化 |

---

## 工程权衡与常见坑

视频生成里最常见的错误，不是模型结构理解错，而是参数设定和工程边界判断错。尤其是帧率、帧数、时长三者必须联动考虑：

$$
\text{duration} \approx \frac{\text{num\_frames} - 1}{\text{fps}}
$$

例如 81 帧、16 fps 对应约 5 秒；161 帧、16 fps 对应约 10 秒：

$$
\frac{161 - 1}{16} = 10
$$

这不是巧合，而是训练时的时间网格设计在推理阶段留下的约束。

### 1. 常见坑

| 常见坑 | 表现 | 原因 | 规避方式 |
|---|---|---|---|
| 帧数乱设 | 直接报错或输出异常 | 不满足 `16N+1` / `8N+1` 约束 | 推理前先做对齐检查 |
| 只看 fps 不看总帧数 | 画面节奏不对 | 时长和训练网格不匹配 | 同时固定 fps 与 duration |
| prompt 太短 | 动作重复、镜头静止 | 语义条件不足 | 明确主体、动作、镜头、背景 |
| caption 数据脏 | 语义漂移、主体不稳 | 训练条件本身不一致 | 清洗训练文本、做 prompt 重写 |
| 误读局部注意力 | 以为模型“只能看附近几帧” | 把降复杂度手段当成完整解释 | 把它理解成成本控制策略 |
| kernel 未启用 | 速度慢、OOM、退化 | 高效注意力算子未生效 | 部署前做环境验证 |
| VAE 压缩太强 | 细节糊、文字难读 | latent 丢失高频信息 | 控制压缩率，接受质量上限 |
| 扩散步数太少 | 画面粗糙、抖动明显 | 去噪不充分 | 在速度和质量间调参 |

### 2. 新手版例子

你要生成“猫从沙发跳到地上”。

如果 prompt 只写：

```text
a cat
```

模型得到的条件只有“主体是猫”。它不知道猫要不要动、往哪动、镜头是否跟拍、背景是否固定。所以输出往往会出现：

- 猫几乎不动
- 背景不断变化
- 猫的花色和体型在中间帧漂移

如果 prompt 写成：

```text
A ginger cat jumps from a sofa to the floor, side view, indoor living room,
camera remains steady, warm daylight, continuous motion
```

约束就清楚得多。  
这不是“提示词技巧”，而是视频生成任务本身需要更完整的条件描述。

### 3. 真实工程例子

短视频团队做“手机开箱演示”，目标是 10 秒、16 fps。若模型常见时间网格是 `16N+1`，自然选择就是：

$$
161 = 16 \times 10 + 1
$$

如果误填成 160 帧，可能出现：

- 时间维 pack 逻辑不匹配
- 模型内部断言失败
- 自动补齐导致时长不准
- 解码阶段 shape 不一致

工程上应该先把业务需求映射到模型允许的帧数网格，再决定分辨率和采样参数，而不是先拍脑袋填一个“看起来差不多”的帧数。

### 4. 推理前应该显式做参数检查

下面这段代码可以直接运行，用来挡掉大量低级错误：

```python
def validate_generation_args(seconds: float, fps: int, stride: int = 16) -> dict:
    num_frames = align_num_frames(seconds, fps, stride)
    return {
        "seconds": seconds,
        "fps": fps,
        "num_frames": num_frames,
        "valid": validate_num_frames(num_frames, stride),
        "estimated_duration": estimate_duration(num_frames, fps),
    }


check_5s = validate_generation_args(5, 16, 16)
check_10s = validate_generation_args(10, 16, 16)

assert check_5s["num_frames"] == 81
assert check_10s["num_frames"] == 161
assert check_5s["valid"] is True
assert check_10s["valid"] is True

print(check_5s)
print(check_10s)
```

这类检查非常简单，但它能把错误留在外层，而不是等模型内部用一个很难读的 shape 报错把你拦住。

---

## 替代方案与适用边界

CogVideo 家族的重要性，主要在于它展示了“Transformer 如何进入视频生成”以及“视频生成如何从直接长序列建模转向 latent diffusion”这条技术演进线。它适合拿来理解架构演化，但不应被理解成所有场景下的唯一答案。

如果从方案选择角度看：

| 方案 | 优点 | 局限 | 适用任务 | 不适用场景 |
|---|---|---|---|---|
| 原版 CogVideo | 教学价值高，能清楚展示自回归视频生成逻辑 | 长序列生成慢，扩展性有限 | 理解早期视频 Transformer 机制 | 高分辨率、长视频、大规模生产 |
| CogVideoX | latent 扩散更稳，质量与效率更平衡 | 仍有采样成本，对 caption / prompt 质量敏感 | 5 到 10 秒文本到视频生成 | 强实时、低延迟交互 |
| 其它视频扩散模型 | 工业生态更丰富，控制模块更多 | 体系差异大，学习成本高 | 实际业务部署、可控生成 | 想追踪 CogVideo 技术脉络时 |

这里的边界可以直接记成两句：

- 如果目标是理解“视频生成是如何从图像 Transformer 扩展而来”，原版 CogVideo 很有代表性。
- 如果目标是理解“现代视频生成为什么普遍走向 latent diffusion”，CogVideoX 更有代表性。

复杂度也解释了为什么长视频越来越依赖 latent 压缩和局部建模。设序列长度为 $L$：

- 全局自注意力复杂度是 $O(L^2)$
- 局部窗口注意力复杂度近似是 $O(L \cdot w)$

如果再先做时空压缩，相当于先把 $L$ 本身减小。这就是 CogVideoX 的工程合理性：  
不是因为它“理论上更优雅”，而是因为在视频任务里，不先压缩、不控制注意力范围，系统往往根本训不动、跑不动。

所以选择标准可以很简单：

- 原版 CogVideo 更适合学原理。
- CogVideoX 更适合看现代实现。
- 如果任务要求超长视频、镜头级精控、实时交互，通常还要看额外控制模块或别的系统，而不是只靠 CogVideo 主体。

---

## 参考资料

下面按“来源、能证明什么、适合先看哪里”整理。建议阅读顺序是：先看论文摘要或 README 建立全局图，再看 PDF 核对机制细节，最后再回到代码仓库看工程实现。

| 来源 | 能证明的结论 | 建议先看 |
|---|---|---|
| CogVideo 论文页 / OpenReview 页面 | 原版 CogVideo 是早期大规模文本到视频 Transformer 路线，核心是从预训练生成模型迁移到视频生成 | 摘要、方法概览 |
| CogVideo 论文 PDF | 分层帧率训练、顺序生成、递归插帧、局部注意力等细节 | 方法章节、训练策略章节 |
| CogVideo 官方仓库 README | 原版工程实现、阶段化模型、local attention 的工程要求 | 模型说明、环境依赖、推理说明 |
| CogVideoX 论文页 / OpenReview 页面 | CogVideoX 采用 3D VAE、expert transformer、扩散式视频生成 | 摘要、贡献列表 |
| CogVideoX 论文 PDF | 3D causal VAE、caption pipeline、frame packing、时空 latent 建模等机制 | 方法与实验章节 |
| CogVideoX 官方仓库 README | 现代工程链路、推理参数、帧数约束、部署要点 | 推理示例、参数说明、显存要求 |

为了防止读混，阅读时最好按下面三个问题去核对：

| 问题 | 原版 CogVideo 的答案 | CogVideoX 的答案 |
|---|---|---|
| 它到底是自回归还是扩散？ | 自回归 | 扩散 |
| 它是在像素 / token 空间生成，还是在 latent 空间生成？ | 更接近直接视频序列建模 | 明确在 latent 空间生成 |
| 它如何处理长序列复杂度？ | 分层帧率、递归插帧、局部注意力 | 3D VAE 压缩 + latent Transformer |

如果只想抓住最重要的阅读路线，可以按这个顺序：

1. 先确认生成范式：自回归还是扩散。
2. 再确认表示空间：直接视频序列还是 latent。
3. 最后确认复杂度控制手段：分层生成、局部注意力，还是先压缩再扩散。

这样最不容易把原版 CogVideo 和 CogVideoX 的论证链混在一起。
