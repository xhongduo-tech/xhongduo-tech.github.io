## 核心结论

视觉编码器的分辨率适应，核心不是“把图喂得更大”，而是“把有限 token 预算优先花在重要细节上”。这里的 token 可以先理解成图像切成小块后得到的离散表示，类似把一张大图切成很多小格子再分别编码。对多模态系统来说，真正稀缺的资源往往不是输入像素，而是后续 Transformer 能承受的视觉 token 数、注意力计算量和 KV cache。

如果直接把输入分辨率从 224 提到 896，在 patch 大小不变时，token 数会从 256 增加到 4096，自注意力开销近似按平方增长，计算量不是 16 倍而是 256 倍。这就是为什么“高分辨率天然更好”在工程上不成立。模型可能看到了更多像素，但推理延迟、显存占用、吞吐和视频帧率会先崩。

分辨率适应的有效做法，是同时保留一个低分辨率全局视图，再对关键区域做高分辨率采样，最后把这些可变长度的视觉 token 重新整理成统一接口，交给语言模型或多模态 Transformer。可以把它理解成看地图时，不是把整张中国地图缩到手机屏幕大小，而是先看全国概览，再把“市中心”单独放大成高精度瓦片；模型的输入维度仍受控，但关键区域的 token 密度更高。

下面这个最小对比能直接说明问题：

| 输入分辨率 | patch 大小 | patch 网格 | token 数 $N$ | 注意力相对开销 $N^2$ |
|---|---:|---:|---:|---:|
| 224×224 | 14 | 16×16 | 256 | 1× |
| 448×448 | 14 | 32×32 | 1024 | 16× |
| 896×896 | 14 | 64×64 | 4096 | 256× |

所以，分辨率适应成立的前提不是“模型足够大”，而是“通过动态区域采样、多尺度视图和位置对齐，让高细节区域的 token 密度上升，同时让全局 token 总量保持在可控范围内”。

---

## 问题定义与边界

问题定义可以写得非常直接：给定一张原始图像，如何在不让视觉 token 数失控的前提下，保留对小目标、远景目标、细文字和局部结构的辨识能力。

这里有两个边界必须先说清。

第一，视觉编码器看到的不是“整张图”，而是 patch。patch 是固定大小的小图块，比如 14×14 像素。图像被切成 patch 后，每个 patch 变成一个 token。token 数量由分辨率和 patch 大小共同决定：

$$
N=\left\lceil\frac{h}{p}\right\rceil\times\left\lceil\frac{w}{p}\right\rceil
$$

其中 $h,w$ 是图像高宽，$p$ 是 patch 边长。对 ViT 类视觉编码器，后续自注意力成本近似为：

$$
O(N^2\cdot d)
$$

其中 $d$ 是隐藏维度。白话说，token 变多一点，注意力成本会放大得更快，因为每个 token 都要和很多别的 token 建立关系。

第二，系统必须在 detail 和 latency 之间设边界。detail 指图像细节保真度，比如远处交通标志上的小字是否还能读出来；latency 指单次推理延迟，视频任务里通常进一步体现在 FPS。真实系统不会无限追求细节，因为多模态模型的视觉前端、跨模态连接层和语言模块都要承担后续成本。

用最直接的数字比较：

- 224×224，patch=14，得到 $16\times16=256$ 个 token
- 896×896，patch=14，得到 $64\times64=4096$ 个 token
- token 数是 16 倍
- 注意力开销近似是 $4096^2/256^2=256$ 倍

这还是只看单层注意力的量级。如果后面还接语言模型，KV cache 也会随视觉 token 序列增长。KV cache 可以理解成推理时为每层保存历史 key/value 的内存区，token 越长，占用越大。于是一个“朴素升分辨率”的方案，很容易在离线 benchmark 上有提升，但在在线服务、机器人控制或视频理解中立刻失去可用性。

玩具例子最容易说明这个边界。假设任务是识别一张城市地图里的医院图标。如果把整张地图统一缩到 224×224，医院图标可能只剩几个像素，直接消失；如果强行把整张图提到 896×896，医院图标确实回来了，但整个编码器要处理 4096 个 token。更合理的方案是：全图用低分辨率保留道路和城区关系，市中心区域单独切高分辨率瓦片，医院图标所在区域获得更高 token 密度。

因此，问题的边界不是“能不能支持任意大图”，而是“是否能在受限 token 预算里，让重要区域的有效分辨率上升，同时不破坏全局上下文”。

---

## 核心机制与推导

分辨率适应通常由三部分组成：全局低分辨率视图、局部高分辨率区域、可变长度 token 到统一表示的重映射。

先看核心思路。以 TEVA 这类方法为例，流程不是把整张高分辨率图全部切细 patch，而是先用较便宜的方式找“值得放大”的区域。这个“区域提案”可以理解成候选框筛选，目标是找到文字密集区、小目标聚集区、操作关键区域或显著物体。然后：

1. 对整张图生成一个低分辨率全局视图
2. 对候选区域使用更细 patch 或更高采样倍率
3. 把全局 token 和局部 token 拼接成一条统一 token 流
4. 通过位置重映射，让模型知道“这些高分辨率 token 在原图的哪里”

这样做的关键不在于“裁剪”，而在于“控制 token 预算”。设全局低分辨率视图产生 $N_g$ 个 token，第 $i$ 个高分辨率区域产生 $N_i$ 个 token，总预算为 $B$，则目标是：

$$
N_g+\sum_{i=1}^{k}N_i \le B
$$

如果直接整图细粒度编码，预算会变成：

$$
N_{\text{full-high}}=\left\lceil\frac{H}{p_s}\right\rceil\left\lceil\frac{W}{p_s}\right\rceil
$$

其中 $p_s$ 是更小的细粒度 patch，大多数情况下远大于 $B$。分辨率适应的本质，就是让“高密度 patch 只出现在少数有价值区域”，于是：

$$
N_{\text{adaptive}} = N_g + \sum_{i=1}^{k}\left\lceil\frac{h_i}{p_f}\right\rceil\left\lceil\frac{w_i}{p_f}\right\rceil
$$

只要区域面积 $\sum h_iw_i$ 远小于整图面积 $HW$，就能用较低额外成本保住关键细节。

这里的直觉可以用地图例子进一步展开。全局低分辨率视图回答“这是什么场景，目标大致在什么位置”；局部高分辨率区域回答“这个标志牌上的字是什么，这个按钮当前是什么状态”。两类信息职责不同，不应该用同一种采样密度处理。

还要解决一个常被忽略的问题：可变长度 token 如何交给固定结构的语言模型。许多多模态系统会在视觉 front-end 后接一个适配层，把不同分辨率、不同区域数产生的 token 重新映射到统一维度和统一顺序。ID-Align 这类方法的作用，可以白话理解为“给每个局部 token 一个能回到原图坐标系的位置身份”，防止模型只看到一堆局部块，却不知道它们在整图中的相对关系。

如果没有这一步，会发生两个典型错误。

第一，上下文割裂。比如文档阅读中，一行字跨越两个瓦片；若只做非重叠切块，模型可能分别识别出半句话，但无法恢复整句关系。

第二，位置失配。很多视觉 Transformer 依赖二维位置编码或 RoPE。不同尺度、不同裁剪窗口的 token 拼接后，如果位置编码仍按局部坐标独立计算，模型会把“局部左上角”误当成“全局左上角”。结果是高分辨率 token 虽然更细，但空间关系反而更乱。

真实工程例子可以看自动化挖掘机场景。摄像头既要看近处铲斗和地面接触细节，又要保留更大范围的施工上下文。Qwen2-VL 的 Naive Dynamic Resolution 思路，本质上是允许不同图像输入产生不同数量的视觉 token，再通过统一接口送给语言模块。这样，地面碎石、沟槽边界、机械臂相对位置等局部细节不需要永远牺牲给固定 224×224 的缩放策略，但整套系统又不必为每帧整图高分辨率编码付出极高成本。

所以，核心机制可以概括成一句话：先用便宜的全局视图建立场景坐标系，再把高 token 密度留给真正影响决策的局部区域，最后通过位置对齐把这些异构 token 重新组织成统一视觉语义空间。

---

## 代码实现

下面给一个可运行的玩具实现，目标不是复现论文，而是把 pipeline 的关键步骤讲清楚：选择原生分辨率、生成全局低分辨率 token、对感兴趣区域做高分辨率采样、再把 token 预算裁到固定上限。

```python
from math import ceil
from typing import List, Tuple, Dict

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2

def patch_tokens(h: int, w: int, patch: int) -> int:
    return ceil(h / patch) * ceil(w / patch)

def choose_native_resolution(h: int, w: int, max_side: int = 896) -> Tuple[int, int]:
    scale = min(max_side / max(h, w), 1.0)
    return int(round(h * scale)), int(round(w * scale))

def sample_regions(image_hw: Tuple[int, int], proposals: List[Box], patch_fine: int) -> List[Dict]:
    regions = []
    H, W = image_hw
    for idx, (x1, y1, x2, y2) in enumerate(proposals):
        x1 = max(0, min(x1, W))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H))
        y2 = max(0, min(y2, H))
        if x2 <= x1 or y2 <= y1:
            continue
        h = y2 - y1
        w = x2 - x1
        regions.append({
            "id": idx,
            "box": (x1, y1, x2, y2),
            "tokens": patch_tokens(h, w, patch_fine),
        })
    return regions

def concat_global_lowres(image_hw: Tuple[int, int], patch_global: int) -> Dict:
    H, W = image_hw
    return {
        "type": "global",
        "tokens": patch_tokens(H, W, patch_global),
        "coord_space": (0, 0, W, H),
    }

def pad_to_fixed_dim(global_view: Dict, regions: List[Dict], token_budget: int) -> Dict:
    total = global_view["tokens"]
    chosen = []
    for r in sorted(regions, key=lambda x: x["tokens"]):
        if total + r["tokens"] <= token_budget:
            chosen.append(r)
            total += r["tokens"]
    return {
        "global_tokens": global_view["tokens"],
        "region_tokens": sum(r["tokens"] for r in chosen),
        "total_tokens": total,
        "selected_regions": [r["box"] for r in chosen],
    }

def build_pipeline(h: int, w: int, proposals: List[Box]) -> Dict:
    native_h, native_w = choose_native_resolution(h, w, max_side=896)
    global_view = concat_global_lowres((native_h, native_w), patch_global=28)
    highres_regions = sample_regions((native_h, native_w), proposals, patch_fine=14)
    packed = pad_to_fixed_dim(global_view, highres_regions, token_budget=900)
    packed["native_resolution"] = (native_h, native_w)
    return packed

result = build_pipeline(
    1200, 1600,
    proposals=[(100, 100, 500, 500), (900, 200, 1300, 700), (50, 900, 300, 1150)]
)

assert result["native_resolution"][1] <= 896
assert result["total_tokens"] <= 900
assert result["global_tokens"] > 0
assert isinstance(result["selected_regions"], list)
```

这个代码里有四个关键点。

第一，`choose_native_resolution` 不是盲目缩到正方形，而是保留长宽比，只约束最大边。native resolution 可以理解成“尽量保留原图几何比例的输入尺度”。

第二，`concat_global_lowres` 用较大的 patch 生成全局视图。这里的全局视图不是为了读细字，而是为了提供场景级上下文。

第三，`sample_regions` 对候选框用更小 patch 采样，等价于更高 token 密度。真实模型里这一步可能由显著性检测、文字检测、目标提案或上游策略网络产生，而不是手工给框。

第四，`pad_to_fixed_dim` 把总 token 控制在预算内。真实系统会更复杂，可能按区域重要性排序，而不是简单按 token 少优先，但预算控制是必须的。

如果把它翻成接近工程实现的伪代码，可以写成：

```python
def encode_image(image):
    image = choose_native_resolution(image, keep_aspect_ratio=True)
    global_tokens = patchify(image.downsample(), patch_size=28)

    proposals = detect_regions(image)
    highres_patch_list = []
    for box in proposals:
        crop = crop_region(image, box)
        patch_tokens = patchify(crop, patch_size=14)
        patch_tokens = remap_position_with_id_align(patch_tokens, box, image.shape)
        highres_patch_list.append(patch_tokens)

    visual_tokens = concat_global_lowres(global_tokens, highres_patch_list)
    visual_tokens = pad_to_fixed_dim(visual_tokens, budget=900)
    return visual_tokens
```

玩具例子可以这样理解。一张 1600×1200 的地图，整图先压到最大边 896，得到全局 token；然后只对“市中心商业区”和“医院所在街区”切高分辨率块。最终语言模型看到的不是“一张被暴力缩小的整图”，而是一份“整图概览 + 局部放大块”的组合表示。

真实工程例子则更接近机器人与文档场景。机器人控制时，低分辨率全局图告诉系统“铲斗在画面左侧，地平线在上部”；高分辨率局部块告诉系统“铲斗齿尖距离地面还有几厘米”。文档阅读时，全局视图提供版面结构，局部高分辨率块保证小字号、脚注和表格线不丢失。

---

## 工程权衡与常见坑

分辨率适应不是免费午餐。它解决的是“整图升分辨率太贵”的问题，但会引入新的工程复杂度。下面这张表是最常见的坑和对应策略。

| 问题 | 现象 | 根因 | 常见规避策略 |
|---|---|---|---|
| token 爆炸 | 显存占用暴涨，视频 FPS 下降 | 直接提高整图分辨率，$N$ 急剧增大 | 动态采样、全局低分辨率视图、区域预算上限 |
| 上下文割裂 | 跨块目标关系判断错误 | 瓦片化后缺少全局关联 | 全局 token 融合、重叠裁剪、跨块 sparse attention |
| 位置失配 | 高分辨率块拼接后空间关系混乱 | 局部坐标和全局坐标未对齐 | ID-Align、二维位置重映射、尺度标记 |
| 提案偏差 | 关键区域没被放大 | 区域提案器漏检 | 提案冗余、回退全局策略、多尺度候选 |
| 延迟抖动 | 同样模型，不同输入延迟差异大 | 可变 token 长度导致算子负载不稳定 | token 分桶、预算离散化、静态 batch 档位 |
| 训练推理不一致 | 训练有效，线上退化 | 训练没覆盖可变分辨率分布 | 训练期引入多分辨率与动态裁剪增强 |

先看最容易踩的坑：直接升分辨率。很多系统在离线实验里把 224 提到 448 或 896，指标会上涨，于是误以为问题已解决。但一旦进入真实推理链路，视觉 token 更长，跨模态连接层更重，语言模块处理前缀更慢，KV cache 明显增大。最后的症状通常不是“略慢”，而是端到端 FPS 崩掉。

第二个坑是瓦片化割裂上下文。非重叠裁块很省事，但对象可能跨块出现。比如一段文本横跨两个 crop，或者一个交通标志的轮廓在低分辨率图里有、文字在局部块里有，如果没有全局融合，模型很容易把两者当作无关信息。

第三个坑是位置编码失配。RoPE 可以理解成一种把位置信息编码进向量旋转关系的方法。它在规则序列上很好用，但当视觉 token 来自不同尺度和不同局部裁剪时，若仍按原始序列方式编码，模型会错误理解“距离”和“邻接”。这时需要 ID-Align 或等价的位置重映射，把局部 token 放回原图坐标系。

Qwen2-VL 在自动化挖掘机任务中的例子，说明了为什么工程上必须允许“任意分辨率对应不同视觉 token 数”。施工现场的输入并不整齐，地面材质、阴影、机械臂姿态和远处背景变化都很大。如果坚持单一固定输入尺寸，近处细节和全局态势总有一边被压坏。动态分辨率接口的价值，不是追求论文里的漂亮概念，而是让视觉前端在统一协议下对不同图像复杂度做不同预算分配。

还有一个常被忽略的问题是调度复杂度。可变 token 长度意味着 batch 内部样本形状更不一致，GPU 利用率可能下降。工程上通常不会完全自由，而是把预算离散到几个档位，例如 256、512、768、1024 token。这样虽然牺牲了一些灵活性，但可以换来更稳定的吞吐和更简单的部署。

---

## 替代方案与适用边界

分辨率适应不是唯一方案，它只是“在细节和算力之间重新分配预算”的一类方法。另一类常见路线，是维持较低输入分辨率，但在注意力结构、下采样策略或任务设计上做优化。

最典型的对比是下面两种策略：

| 策略 | 核心思路 | 优势 | 局限 | 典型场景 |
|---|---|---|---|---|
| 固定 224×224 + window attention | 输入固定，小窗口内做注意力 | 部署简单、吞吐稳定、延迟可控 | 小目标和细文字容易丢失，全局与局部都受压缩 | 轻量端部署、强实时视频分类 |
| 动态 native resolution + region proposal | 保留长宽比，全局低清 + 局部高精 | 细节保真更好，适应多尺度目标 | 系统更复杂，提案与位置对齐要额外设计 | 文档阅读、机器人控制、复杂场景理解 |

固定低分辨率加稀疏注意力的路线，适合 FPS 极敏感任务。比如边缘设备上的快速筛查、安防视频中的粗粒度事件检测、移动端实时交互。这类任务的首要目标往往是稳定吞吐，而不是读小字或识别远处细节。

动态分辨率加区域提案，更适合细粒度信息真的影响决策的场景。文档 OCR、图表理解、遥感小目标检测、机器人精细操作，都会明显受益。因为这些任务的失败，往往不是“没看到大轮廓”，而是“关键细节被统一缩放压没了”。

还有一条边界需要说清：如果目标本身就依赖全局一致性，过度局部采样反而有风险。比如审美评分、整体布局理解、密集场景关系推理。如果区域提案过强，只盯着局部热点，模型可能忽略全局风格和大尺度结构。这种任务往往要提高全局视图质量，或者限制局部块数量，避免系统被局部高频细节牵着走。

因此，适用边界可以简单总结为：

- 任务主要看“局部关键细节”时，优先考虑分辨率适应
- 任务主要看“稳定实时吞吐”时，优先考虑固定低分辨率
- 任务同时需要“全局关系 + 局部细节”时，必须保留全局低分辨率视图，不能只做纯裁块

---

## 参考资料

1. Michael Brenndoerfer. *Vision Encoders for VLMs: SigLIP, Resolution, and Architecture*. 2025. 主要贡献：系统解释视觉编码器中的 native resolution、patch token、分辨率与注意力成本关系，以及多模态视觉前端如何处理可变分辨率输入。

2. Yitong Jiang, et al. *Token-Efficient VLM High-Resolution Image Understanding via Dynamic Region Proposal (TEVA)*. ICCV 2025. 主要贡献：提出动态区域提案与多尺度 patch 抽样，在受控 token 预算下提升高分辨率图像理解能力。

3. *Resource-efficient fine-tuning of vision-language models for autonomous excavators*. Frontiers in AI, 2025. 主要贡献：报告 Qwen2-VL 在自动化挖掘机场景中的工程实现，说明 Naive Dynamic Resolution 如何在统一接口下兼顾地面细节和实时视频流处理。
