## 核心结论

Qwen2-VL 的动态分辨率，本质上是在回答一个老问题：图像大小不固定时，视觉编码器怎么既保留细节，又不把序列长度炸穿。它的做法不是把所有图都硬压到 `224×224`，而是先把输入高宽调整到 `patch_size × merge_size` 的倍数，在 Qwen2-VL 里通常是 $14 \times 2 = 28$，然后再切成 patch，最后把局部 patch 通过 MLP 合并成更少的视觉 token。

对白话解释一次：`patch` 就是“把图像切成很多固定大小的小方格”；`merge` 就是“把相邻几个小方格再打包成一个更大的语义单元”；`token` 就是“送给语言模型读的一条视觉摘要”。这样做的结果是，图像越大，token 一般越多，但增长被控制住，不会无限膨胀。

玩具例子可以这样理解：不要把一张大地图先糊成一张小缩略图，再去找街道名；而是先按规则切成小块，再按局部合并，保留局部高分辨率信息，同时控制总块数。真实工程里，这种机制特别适合截图理解、网页 Agent、OCR 较多的多模态问答，因为固定缩放最容易先丢掉的，恰好就是字体、小图标和边缘细节。

一个简化数据流如下：

```text
原图
  ↓ resize 到 28 的倍数
规则化尺寸图
  ↓ 按 14×14 切 patch
patch 序列
  ↓ 加 2D-RoPE 位置编码
带空间位置信息的 patch
  ↓ 按 2×2 局部组块，用 MLP 合并
视觉 token
  ↓ 送入 LLM
文本推理结果
```

下表先给出最重要的参数关系：

| 参数 | 含义 | Qwen2-VL 常见值 | 直接影响 |
|---|---|---:|---|
| `patch_size` | 每个视觉小块边长 | 14 | patch 切分粒度 |
| `merge_size` | 合并 patch 的局部分组边长 | 2 | token 压缩强度 |
| `factor` | 输入尺寸对齐因子 | 28 | 高宽必须对齐到的倍数 |
| `min_pixels` | 最小像素约束 | 3136 | 防止小图过度压缩 |
| `max_pixels` | 最大像素约束 | 12,845,056 | 防止大图显存失控 |
| `token_limit` | 视觉序列预算上限 | 16384 | 控制输入到 LLM 的长度 |

---

## 问题定义与边界

问题定义很具体：给定任意分辨率图像，甚至百万像素级输入，如何让 Vision Transformer 仍然可以稳定处理，并把结果交给语言模型，而不因为图太大导致显存、时延、上下文长度全部失控。

这里有两个边界必须同时满足。

第一，不能过早下采样。`downsample` 就是“先把图整体缩小再处理”。这在自然图像分类里常常足够，但在多模态系统里问题很大，因为文字、表格、控件边框、地图标注、图标角标都属于高频细节，一旦先缩糊，后面再强行识别基本救不回来。

第二，不能无限保留原图。因为 ViT 的复杂度和 token 数高度相关，图像面积越大，patch 越多，后续注意力和跨模态拼接成本越高。没有约束时，大图不是“效果更好”，而是“系统先崩”。

所以动态分辨率不是“原图直喂”，而是“在像素和 token 预算内尽量保留原图结构”。这就引入了两个数值边界：

| 约束 | 数值 | 作用 | 设错的风险 |
|---|---:|---|---|
| `min_pixels` | 3136 | 防止图像被压得过小 | 太高会把小图放大到训练分布外 |
| `max_pixels` | 12,845,056 | 限制最大输入面积 | 太高会导致显存和时延暴涨 |
| 最大 token | 16384 | 限制视觉序列总长度 | 超过后无法稳定送入 LLM |
| 最大宽度 | 常见工程值 1920 | 对超宽或 4K 图额外限流 | 不设时大屏截图最危险 |

玩具例子：一张 `93×41` 的小图，如果直接按 `14×14` patch 切，会出现边缘不整齐和 token 稀碎的问题。动态分辨率会先把它调整到接近且满足约束的 `28` 倍数，比如 `84×56` 或 `112×56` 这一类规则尺寸，再切 patch。

真实工程例子：浏览器 Agent 连续读取网页截图。网页可能是 `1440×900`，也可能是滚动后拼出的长图，甚至是 `3840×2160` 的桌面截屏。如果全部按固定 `224×224`，按钮和小字会消失；如果全部按原图切 patch，序列长度会迅速爆炸。动态分辨率的价值，就在这两个坏结果之间找一条可运行的中线。

---

## 核心机制与推导

核心机制可以压缩成四步：尺寸对齐、patch 切分、位置编码、局部合并。

先定义：

$$
factor = patch\_size \times merge\_size
$$

在 Qwen2-VL 里通常有：

$$
patch\_size = 14,\quad merge\_size = 2,\quad factor = 28
$$

输入图像高宽先对齐到 `factor` 的倍数：

$$
h' = round(h / factor) \times factor
$$

$$
w' = round(w / factor) \times factor
$$

同时满足：

$$
min\_pixels \le h' \times w' \le max\_pixels
$$

然后按 `patch_size` 切分，初始 patch 数为：

$$
patch\_count = (h' / patch\_size)\times(w' / patch\_size)
$$

因为 `merge_size=2`，可以把相邻 `2×2` 的 patch 视作一个合并单元，所以送入后续语言模型前的视觉 token 数可近似写成：

$$
visual\_tokens \approx \frac{patch\_count}{merge\_size^2}
$$

即：

$$
visual\_tokens \approx \frac{(h'/14)\times(w'/14)}{4}
$$

再加上预算约束：

$$
visual\_tokens \le token\_limit = 16384
$$

这里的“近似”很重要，因为真实系统里还会受具体实现、边界处理、视频帧拼接方式影响，但工程判断上这个式子已经足够有用。

看一个玩具例子。输入图像是 `512×512`：

1. `512` 本身不是 `28` 的倍数，最接近的规则尺寸可取 `504` 或 `532`，工程上通常会选满足预算且接近原始比例的尺寸。
2. 若取 `504×504`，则 patch 数约为 $(504/14)^2 = 36^2 = 1296$。
3. 按 `2×2` 合并后，视觉 token 约为 $1296/4 = 324$。

这和固定 `224×224` 的差异在于，后者会强制丢细节；而这里虽然 token 变多，但仍处在可控预算内。

再看真实工程例子。假设地图截图为 `1920×1080`：

- patch 数约为 $(1920/14)\times(1064/14)$ 这一量级；
- 合并后 token 仍然远高于小图，但仍可以通过总预算控制在单帧可接受范围；
- 如果是视频或多步截图序列，就必须把每帧 token 和帧数一起管理，否则不是单帧爆，而是总序列爆。

下面这个表格把常见输入和 token 规模直观列出来：

| 输入尺寸 | 对齐后尺寸示意 | 初始 patch 数近似 | merge 后视觉 token 近似 |
|---|---|---:|---:|
| `112×56` | `112×56` | $(8×4)=32$ | `8` |
| `512×512` | `504×504` 或相近尺寸 | `1296` 左右 | `324` 左右 |
| `1280×720` | 接近 28 倍数的尺寸 | `4700+` | `1170+` |
| `1920×1080` | 接近 28 倍数的尺寸 | `10000+` | `2500+` |

这就是它为什么成立：视觉 token 数随图像面积增长，但增长率被 patch 切分和 merge 压缩共同钳住，不再是“原图越大，输入越不可用”的线性灾难。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不包含真实的 ViT、2D-RoPE 或 MLP 权重，但把动态分辨率的尺寸约束和 token 估算逻辑完整跑通了。

```python
from math import ceil

PATCH_SIZE = 14
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE
MIN_PIXELS = 3136
MAX_PIXELS = 12_845_056
TOKEN_LIMIT = 16384


def round_to_factor(x: int, factor: int) -> int:
    # 把尺寸对齐到最接近的 factor 倍数
    return max(factor, round(x / factor) * factor)


def adjust_resolution(h: int, w: int) -> tuple[int, int]:
    # 第一步：先把高宽对齐到 factor 的倍数
    h2 = round_to_factor(h, FACTOR)
    w2 = round_to_factor(w, FACTOR)

    # 第二步：如果像素过小，就按比例放大
    while h2 * w2 < MIN_PIXELS:
        h2 += FACTOR
        w2 += FACTOR

    # 第三步：如果像素过大，就按比例缩小
    while h2 * w2 > MAX_PIXELS:
        h2 -= FACTOR
        w2 -= FACTOR
        if h2 < FACTOR or w2 < FACTOR:
            raise ValueError("image too extreme after constraint adjustment")

    return h2, w2


def estimate_visual_tokens(h: int, w: int) -> int:
    # patch 总数
    patch_count = (h // PATCH_SIZE) * (w // PATCH_SIZE)
    # 2x2 merge 后的视觉 token 数
    visual_tokens = ceil(patch_count / (MERGE_SIZE ** 2))
    return visual_tokens


def pipeline_shape(h: int, w: int) -> dict:
    h2, w2 = adjust_resolution(h, w)
    tokens = estimate_visual_tokens(h2, w2)
    return {
        "input": (h, w),
        "adjusted": (h2, w2),
        "patch_count": (h2 // PATCH_SIZE) * (w2 // PATCH_SIZE),
        "visual_tokens": tokens,
        "within_limit": tokens <= TOKEN_LIMIT,
    }


toy = pipeline_shape(93, 41)
assert toy["adjusted"][0] % FACTOR == 0
assert toy["adjusted"][1] % FACTOR == 0

case_512 = pipeline_shape(512, 512)
assert case_512["within_limit"] is True
assert case_512["visual_tokens"] > 0

case_hd = pipeline_shape(1920, 1080)
assert case_hd["adjusted"][0] % FACTOR == 0
assert case_hd["adjusted"][1] % FACTOR == 0

print(toy)
print(case_512)
print(case_hd)
```

如果只看流程，伪代码可以写成这样：

```python
# 1. 计算输入对齐因子
factor = patch_size * merge_size

# 2. 把图像尺寸调整到 factor 的倍数，并满足 min/max pixels
h_adj = round(h / factor) * factor
w_adj = round(w / factor) * factor

# 3. 从规则化后的图像中提取 patch
patches = extract_patches(image, patch_size)

# 4. 给每个 patch 加二维位置编码，保留空间关系
positioned = add_2d_rope(patches)

# 5. 对局部 patch 做 merge，用 MLP 压缩 token 数
visual_tokens = mlp_merge(positioned, merge_size)

# 6. 把视觉 token 接到文本 token 前后，送入 LLM
output = llm(visual_tokens, text_tokens)
```

每一步各有目的：

| 步骤 | 作用 | 不做会怎样 |
|---|---|---|
| `resize to factor` | 保证 patch 与 merge 对齐 | 边界 patch 难处理，token 形状不稳定 |
| `extract_patches` | 把图像转成 ViT 可读序列 | 无法进入 Transformer |
| `add_2d_rope` | 保留“左上右下”的空间位置 | 模型知道内容，不知道布局 |
| `mlp_merge` | 压缩局部冗余，控制 token 数 | 大图序列过长，成本过高 |

关键不是“可变长度”本身，而是“长度跟图像内容规模相关，但仍受预算约束”。这比固定 `224×224` 更贴近真实任务，也比完全原图直送更工程化。

---

## 工程权衡与常见坑

第一个坑是把 `min_pixels` 设得过高。表面看这像是在“给模型更多像素”，实际可能是在把小图强行放大。对白话解释：模型训练时见过的是正常大小的字符和图标，你现在把一张本来很小的票据、图标或头像放大很多倍，像素更多了，但统计分布变了，尤其 OCR 和小目标任务可能反而退化。

第二个坑是低估大图风险。`3840×2160` 这类 4K 输入，即使经过 merge，token 数也依然不小。如果再叠加视频帧、页面操作历史、多轮对话，很快会把视觉预算和总上下文预算同时打满。因此工程里常见的做法是再加一层“最大宽度”保护，比如把宽度截到 `1920` 左右，或者直接降低 `max_pixels`。

第三个坑是只管单帧，不管全序列。多模态 Agent 常常不是看一张图，而是看一个操作轨迹。此时除了单帧动态分辨率，还要配合 `vision_start` / `vision_end` 这样的分段标记，让模型知道哪些 token 属于第几帧、哪一段视觉上下文已经结束。否则即使单张图安全，多帧拼起来也会超限。

下面这个表格是更实用的工程配置视图：

| 设置项 | 设太小的后果 | 设太大的后果 | 常见建议 |
|---|---|---|---|
| `min_pixels` | 小图信息不足 | 小图被放大到分布外 | 从训练和任务输入尺度反推 |
| `max_pixels` | 大图被压得太狠 | 显存和时延暴涨 | 结合 GPU 预算设上限 |
| 最大宽度 | 超宽图仍可能过长 | 过度裁弱横向细节 | 对桌面截图常设 `1920` 左右 |
| 帧率 / 帧数 | 丢失操作过程 | 序列 token 爆炸 | Agent 场景按任务抽帧 |

真实工程例子：地图 Agent 以 `2 fps` 读取屏幕，在地图上搜索目标门店。如果每帧都保留高分辨率且不限制宽度，几秒钟就会累积大量视觉 token。合理做法不是退回 `224×224`，而是同时做三件事：单帧动态分辨率、总帧数控制、按段落插入视觉边界标记。这样保住街道名、图标和弹窗细节，同时不让上下文失控。

---

## 替代方案与适用边界

最直接的替代方案是固定下采样，比如全部压到 `224×224` 或 `336×336`。这种方案的优点是实现简单、显存稳定、吞吐容易预估；缺点是高频细节损失严重，尤其不适合 OCR、UI 理解、图表和地图类任务。

另一个替代方案是裁块推理。也就是先把大图裁成多个局部窗口，再分别编码。这能保留局部清晰度，但跨窗口关系容易断裂，后处理也更复杂。对需要全局布局理解的页面或长图，这种方法不总是比动态分辨率更好。

对比可以放在一张表里：

| 方案 | token 数稳定性 | 细节保留 | 显存波动 | 适用场景 |
|---|---|---|---|---|
| 动态分辨率 | 中等，可控 | 高 | 中等 | UI、OCR、图表、地图、多帧截图 |
| 固定 `224×224` | 高 | 低 | 低 | 简单分类、粗粒度问答 |
| 裁块推理 | 中等 | 高 | 高 | 超大图局部分析、离线处理 |

它的适用边界也要说清楚。动态分辨率不是“永远优于固定缩放”。如果任务本身只需要粗粒度语义，比如判断一张图是风景还是室内，固定缩放更便宜。如果任务输入天然很小，比如身份证裁剪图、单个按钮截图、低分辨率监控小图，把 `min_pixels` 设太高反而可能产生不必要放大，带来分布偏移。

所以决策原则很简单：

1. 要保细节，就优先考虑动态分辨率。
2. 要保吞吐、任务又不依赖细节，就用固定缩放。
3. 图像极大时，即使用动态分辨率，也继续限制最大宽度或回退更激进的 downsample。
4. 多帧输入时，单帧分辨率和总帧数要一起调，不能分开看。

---

## 参考资料

1. Qwen2-VL 论文，重点看动态分辨率、2D-RoPE、视觉 token 预算设计。  
   https://ar5iv.org/abs/2409.12191

2. EmergentMind 的 Qwen2.5-VL 概览，适合先建立直觉，再回到论文核对术语和流程。  
   https://www.emergentmind.com/topics/qwen2-5-vl

3. CSDN 博客“LLM: 多模态LLM动态分辨率”，适合快速抓实现参数、`factor`、`min_pixels/max_pixels` 与 pipeline 细节。  
   https://blog.csdn.net/WiSirius/article/details/145545334
