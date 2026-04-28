## 核心结论

多模态推理优化，最常见的主瓶颈不是语言模型逐 token 解码，而是图像进入模型前后的两段：`视觉编码`和`跨模态融合`。这里的“视觉编码”就是把图片切块、映射成向量并送进视觉塔；“跨模态融合”就是把视觉 token 和文本 token 接到同一个推理链路里。

`TTFT`（Time To First Token，首 token 时间）是用户提交请求后，模型吐出第一个输出 token 的时间。对在线服务来说，它通常比平均吞吐更直接影响体感。多模态场景里，`TTFT` 往往先被图像处理阶段拉长，因为文本解码必须等视觉侧准备完才能开始真正回答。

先看结论表：

| 瓶颈 | 主要影响 | 典型表现 |
|---|---|---|
| 视觉编码 | 计算与显存 | `TTFT` 变长 |
| 跨模态融合 | 计算 | prefill 变慢 |
| 语言解码 | 相对次要 | 对总延迟贡献较小 |

如果只记一个判断标准，就是：先看图像分辨率和视觉 token 数，再看 LLM 解码速度。很多系统把大部分时间花在“把图送进来”这一步，而不是“把字吐出来”这一步。

---

## 问题定义与边界

本文讨论的是一类典型链路：输入里包含图像或视频帧，系统先经过视觉编码器得到视觉 token，再把它们与文本 token 一起交给大语言模型继续推理。这里的“token”可以先理解成模型内部处理的最小离散单元，文本 token 对应子词，视觉 token 对应图像 patch 的向量表示。

范围边界如下：

| 范围内 | 范围外 |
|---|---|
| 图像/视频编码 | 纯文本 LLM |
| 在线推理 | 训练阶段优化 |
| 显存、`TTFT`、吞吐 | 数据标注、模型对齐 |

本文重点关注三个工程问题。

第一，延迟。用户上传图片后，为什么首 token 很慢。

第二，显存。为什么参数明明放得下，但一上并发就 OOM。

第三，吞吐。为什么单请求还能跑，多请求一起进来就抖动明显。

常用变量先统一：

- `H×W`：输入分辨率，高和宽。
- `p`：patch size，即每个视觉 patch 的边长。
- `N_v`：视觉 token 数。
- `N_t`：文本 token 数。
- `d`：hidden size，模型内部向量宽度。
- `b`：位宽，例如 fp16、bf16、int8。

一个典型场景是批量文档问答服务。每个请求带 1 到 3 张扫描件，图像本身不一定复杂，但分辨率往往很高，因为用户希望 OCR 和版面信息尽量保留。此时真正的问题不是模型“能不能答对”，而是高并发时服务“能不能稳定接住请求”。

---

## 核心机制与推导

最核心的量是视觉 token 数。对 ViT 系视觉编码器，一张图先被切成固定大小的 patch，再把每个 patch 映射为一个 token。因此可以近似写成：

$$
N_v \approx \left(\frac{H}{p}\right)\times\left(\frac{W}{p}\right)+c
$$

其中 `c` 通常是一个额外特殊 token，比如分类 token。对工程估算来说，`c` 一般不是主要矛盾，主导项还是分辨率和 patch size。

这条公式直接说明一件事：如果 `p` 固定，图像面积增大，`N_v` 会近似按面积增长。分辨率从 `224×224` 到 `448×448`，高和宽都翻倍，但面积变成 4 倍，所以视觉 token 数也大约变成 4 倍。

玩具例子先算清楚：

- `224×224, p=14` 时，`N_v = 16×16 = 256`
- `448×448, p=14` 时，`N_v = 32×32 = 1024`

这不是“稍微变大一点”，而是视觉 token 数直接变成 4 倍。

接下来是为什么代价会放大得更厉害。视觉塔常见的核心算子是自注意力。这里的“自注意力”可以先理解成：每个 token 都要和别的 token 交互，所以 token 越多，配对关系增长越快。它的主要复杂度通常近似为：

$$
O(N_v^2 \cdot d)
$$

因此 `N_v` 从 `256` 增加到 `1024`，是 4 倍；但平方项意味着视觉注意力理论计算量会接近增加到 16 倍。

多模态系统还要把视觉 token 和文本 token 接起来做融合。无论是 cross-attention、拼接后统一 prefill，还是 projector 后再输入 LLM，本质上都要处理视觉序列和文本序列的共同开销，常见估算形式为：

$$
O(N_v \cdot N_t \cdot d)
$$

所以只要 `N_v` 大，哪怕文本长度 `N_t` 不夸张，prefill 阶段也会明显变慢。

这就得到一条很实用的推导链：

1. 分辨率升高  
2. `N_v` 增加  
3. 视觉注意力和跨模态融合成本上升  
4. `TTFT` 变差，同时显存压力升高  

`TTFT` 可以粗略拆成：

$$
TTFT \approx T_{vis} + T_{proj} + T_{fuse} + T_{prefill}
$$

- `T_vis`：视觉编码时间
- `T_proj`：视觉特征投影到 LLM 空间的时间
- `T_fuse`：跨模态融合时间
- `T_prefill`：LLM 在完整上下文上跑首轮前向的时间

这条式子背后的意思很直接：用户看到的“慢”，并不只来自 LLM。只要视觉塔和投影层在前面串行执行，语言解码连起跑线都还没到。

显存也不能只看模型参数。粗略写法是：

$$
M \propto b \times (\text{参数} + \text{激活} + \text{cache})
$$

这里的“激活”就是前向过程中中间层产生的临时张量，“cache”包括 KV cache 和多模态特征缓存。位宽 `b` 降低，参数显存会下降，但真正在线上击穿内存预算的，常常是激活峰值和 cache 复制。

真实工程例子更能说明问题。假设一个文档问答服务：

- 每请求 2 张 `1536×1024` 扫描件
- 每张图经过裁剪后仍产生较多视觉 token
- 用户问题只有 40 个文本 token
- 回答平均只输出 120 个 token

这时语言解码并不长，但因为图片先占掉大量视觉编码和 prefill 时间，系统可能在“开始回答前”就已经用掉了大部分延迟预算。并发一高，`encoder + projector + prefill` 还会同时争抢显存，导致 `P95/P99 TTFT` 明显抬升。

---

## 代码实现

实现重点不是“把模型完整跑起来”，而是把推理链路拆开，并对最敏感的变量设硬限制。最小流程通常像这样：

```python
image = load_image(path)
image = resize_with_max_pixels(image, max_pixels=...)
visual_tokens = vision_encoder(image)
projected_tokens = projector(visual_tokens)
output = llm.prefill(text_tokens, projected_tokens)
answer = llm.decode()
```

关键不是这几行本身，而是要把每一步单独统计。

下面给一个可运行的简化 Python 例子，用来估算视觉 token 数、注意力相对成本和是否超过像素预算：

```python
from math import ceil

def visual_tokens(height: int, width: int, patch: int, extra_tokens: int = 1) -> int:
    assert height > 0 and width > 0 and patch > 0
    return ceil(height / patch) * ceil(width / patch) + extra_tokens

def relative_attention_cost(nv_a: int, nv_b: int) -> float:
    assert nv_a > 0 and nv_b > 0
    return (nv_a * nv_a) / (nv_b * nv_b)

def clamp_resolution(height: int, width: int, max_pixels: int) -> tuple[int, int]:
    assert height > 0 and width > 0 and max_pixels > 0
    pixels = height * width
    if pixels <= max_pixels:
        return height, width

    scale = (max_pixels / pixels) ** 0.5
    new_h = max(1, int(height * scale))
    new_w = max(1, int(width * scale))
    assert new_h * new_w <= max_pixels
    return new_h, new_w

# 玩具例子：224 -> 448，视觉 token 约 4 倍
nv_224 = visual_tokens(224, 224, 14, extra_tokens=0)
nv_448 = visual_tokens(448, 448, 14, extra_tokens=0)
assert nv_224 == 256
assert nv_448 == 1024
assert nv_448 / nv_224 == 4

# 理论自注意力成本约 16 倍
ratio = relative_attention_cost(nv_448, nv_224)
assert ratio == 16

# 工程例子：限制最大像素数
h, w = clamp_resolution(1536, 1024, max_pixels=512 * 512)
assert h * w <= 512 * 512
```

如果要把它映射到服务配置，通常需要几个硬阈值：

```yaml
max_pixels: 262144
num_crops: 1
num_frames: 8
multimodal_cache_limit_gb: 4
encoder_tp: 2
```

这里每一项都对应一个明确目标：

| 优化点 | 作用 | 代价 |
|---|---|---|
| 限制 `max_pixels` | 降低 `N_v` | 可能损失细节 |
| 分离 encoder 和 LLM | 降低串行阻塞 | 系统复杂度上升 |
| 单独控制 multimodal cache | 降低显存峰值 | 需要额外监控 |

工程实现时，建议至少做三件事。

第一，输入预处理前置。图片一进入系统就做分辨率裁剪或像素上限控制，不要等进入模型后才发现 token 爆了。

第二，分阶段计时。把 `T_vis`、`T_proj`、`T_fuse`、`T_prefill`、`T_decode` 分别打点，否则你只能看到一个总延迟，无法知道该优化哪一段。

第三，显存预算分桶。不要只看参数占用，把视觉缓存、文本 KV cache、激活峰值分开监控。多模态系统里，“图片相关缓存”往往是额外增长项，不会自动和纯文本经验一致。

---

## 工程权衡与常见坑

多模态优化最大的误区，是用纯文本 LLM 的思维去看系统性能。纯文本里，解码阶段经常是主要关注点；但在图文推理里，prefill 前半段的视觉路径可能才是主矛盾。

常见坑如下：

| 常见坑 | 后果 | 规避方法 |
|---|---|---|
| 只测 decode | 误判系统性能 | 拆分 `T_vis` / `T_fuse` / `TTFT` |
| 分辨率过大 | `N_v` 暴涨 | `max_pixels` 硬限制 |
| 只看参数显存 | 低估实际占用 | 同时看激活和 cache |
| 视觉塔与 LLM 串行跑 | 尾延迟高 | encoder disaggregation |
| 忽略 multimodal cache 复制 | 显存被重复占用 | 独立预算和监控 |

指标优先级通常也要重排：

| 优先级 | 指标 |
|---|---|
| 1 | `P95/P99 TTFT` |
| 2 | 显存峰值 |
| 3 | 吞吐 |

为什么不是先看吞吐？因为在线服务先死在尾延迟，再死在吞吐。用户更容易感知的是“为什么第一下这么慢”，而不是“每秒能处理多少 token”。

再看一个真实工程坑。多图文档问答里，开发者常常把所有图片都按原分辨率送入视觉塔，觉得“先别丢信息”。这在离线评测里可能没问题，因为只跑单请求；但线上一旦 8 到 16 个请求并发，每个请求 2 张图，视觉特征、中间激活和 projector 输出会同时堆起来。结果不是平均延迟变差一点，而是显存峰值突然穿顶，随后出现抖动、回退甚至 OOM。

另一个常见错误是只做参数量化，不看算子支持。比如从 fp16 压到 int8，理论上参数显存能下降不少，但如果视觉塔某些路径仍以高精度激活运行，或者 projector 和 cache 没同步优化，那么线上收益会明显低于纸面收益。量化是有效手段，但不是“一键减半显存”的保证。

---

## 替代方案与适用边界

多模态优化没有单一万能解。你想省显存、降延迟、提并发，往往不能同时做到最优，只能按目标选方案。

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| 限制 `max_pixels` | 通用在线服务 | 简单有效 | 可能损失细节 |
| `num_crops` 控制 | 长图/文档 | 保留局部信息 | 计算更复杂 |
| encoder disaggregation | 高并发服务 | 降低串行阻塞 | 系统改造大 |
| 量化到 INT8 | 显存敏感 | 显存下降明显 | 可能影响精度 |
| 独立 multimodal cache | 多模态并发 | 控制峰值 | 需要更复杂的资源管理 |

几个典型边界可以直接判断。

如果追求最小延迟，优先做分辨率控制、视觉侧并行化、分阶段调度。目标是让 `T_vis + T_proj + T_fuse` 尽量短。

如果追求最高精度，比如高精度 OCR、细粒度图表理解、复杂页面问答，就不能激进压缩分辨率。此时更适合控制 `num_crops`、优化服务调度，或增加资源预算，而不是简单暴力地下采样。

如果追求最低成本，最直接的是限制 `max_pixels`、限制帧数、限制多图数量，再配合量化。但要接受效果上限下降，因为你本质上在减少输入信息和计算预算。

低分辨率问答场景是一个正面例子。比如商品图粗分类、简单截图问答，图像信息密度低，先压分辨率再推理，通常比保留原图更划算。反过来，在扫描件 OCR 或表格理解场景，过度压图会直接伤害识别结果，因为小字、线条和局部结构本来就脆弱。

所以边界可以总结成一句话：追求最小延迟、最高精度、最低成本，这三者通常不能同时最优。多模态推理优化不是把某个开关开到最大，而是先确定你最在乎哪一个目标。

---

## 参考资料

| 类型 | 资料 |
|---|---|
| 论文 | [ViT: An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) |
| 论文 | [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) |
| 论文 | [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191) |
| 论文 | [SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant](https://openreview.net/forum?id=UwFk7EvZjK) |
| 工程 | [vLLM: Encoder Disaggregation for Scalable Multimodal Model Serving](https://vllm.ai/blog/vllm-epd) |
| 文档 | [vLLM docs: multimodal config / encoder TP / mm profiling](https://docs.vllm.ai/en/latest/cli/bench/latency/) |

1. [ViT: An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
2. [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
3. [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)
4. [SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant](https://openreview.net/forum?id=UwFk7EvZjK)
5. [vLLM: Encoder Disaggregation for Scalable Multimodal Model Serving](https://vllm.ai/blog/vllm-epd)
6. [vLLM Documentation](https://docs.vllm.ai/)
