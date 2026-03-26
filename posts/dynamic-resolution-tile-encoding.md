## 核心结论

动态分辨率与瓦片编码，是高分辨率视觉语言模型处理大图时最常见的一组工程策略。动态分辨率的意思是：输入图像不再强制缩放到统一尺寸，而是根据原始宽高比和像素规模，动态决定要分成多少块；瓦片编码的意思是：把大图切成多个固定大小的小块，分别送入视觉编码器，再把结果按原始位置组织回来。

这组策略解决的核心问题很直接：固定分辨率 ViT 会把 4K 图像压缩到 224×224 或 448×448，小目标、细字、边界线和局部布局会被平均掉。对于文档理解、表格问答、GUI 识别、多图对比、长图推理，这种损失通常是不可接受的。

玩具例子可以这样理解：一张 4K 图像如果直接缩到 224×224，像把一张大地图拍成邮票；而瓦片编码更像把地图切成多个 448×448 的拼图块，每块单独看清细节，再保留它原来的空间位置。这样做不能完全等价于“同时看整张原图”，但在显存和算力有限时，是目前最实用的折中方案。

公式上，若 patch size 为 14，则视觉 token 数可写成：

$$
N(H,W)=\left\lfloor\frac{H}{14}\right\rfloor\cdot \left\lfloor\frac{W}{14}\right\rfloor
$$

这说明图像越大，token 数越快增长。动态分辨率控制的是“保留多少原始像素”，瓦片编码控制的是“如何把这些像素拆成模型能吃下的批次”。

---

## 问题定义与边界

问题定义：给定一张高分辨率图像 $H\times W$，希望既保留细节，又不让视觉编码器的计算和显存成本失控。这里的高分辨率不只指 4K，也包括超长截图、多页拼接图、多图输入和视频帧序列。

术语先解释一遍。ViT 是 Vision Transformer，白话讲就是“把图像切成很多小块，再把这些小块当作序列送进 Transformer”。tile 是瓦片，白话讲就是“更大一级的图像子图块”，例如 448×448 的裁剪块。window attention 是窗口注意力，白话讲就是“每个 token 只和附近一小片区域做注意力，不和全图所有 token 两两交互”。

如果把 3360×1890 的图直接缩到 224×224，像素总量从约 635 万下降到约 5 万，很多局部结构会直接消失。建筑图里的细线、文档中的角标、表格中的小字、网页截图里的按钮标签，都可能在缩放后不再可分辨。这不是模型“理解不好”，而是输入阶段已经丢掉了信息。

下面用表格看固定 resize 和动态 tile 的边界差异：

| 方案 | 输入方式 | 细节保留 | token 数 | 计算压力 | 适合场景 | 风险 |
|---|---|---:|---:|---:|---|---|
| 固定 224² | 全图缩放到 224×224 | 低 | 很低 | 低 | 粗粒度分类、资源极紧 | 小目标和细字丢失 |
| 固定 448² | 全图缩放到 448×448 | 中 | 中 | 中 | 普通图文问答 | 长图和超大图仍丢细节 |
| 瓦片编码 | 切成多个 448×448 tile | 高 | 高 | 可控 | 文档、GUI、多图 | tile 过多会爆预算 |
| 动态分辨率 + 瓦片 | 按宽高比和预算决定 tile | 高 | 按预算动态 | 可控且更稳 | 4K、长图、视频帧 | 实现复杂，需要位置编码配合 |

边界也要明确。它并不能免费获得“全局无损理解”。如果 tile 之间只看局部，不做全局信息交换，模型仍可能遗漏跨区域关系。所以这类方案的目标不是完美恢复原图，而是在可接受算力内尽量保留细节与空间结构。

---

## 核心机制与推导

核心机制分三层：先决定分多少 tile，再把每个 tile 编成 token，最后通过位置编码和局部注意力保留空间结构。

第一层是 token 增长规律。若 patch size 为 14，则：

$$
N(H,W)=\left\lfloor\frac{H}{14}\right\rfloor\cdot \left\lfloor\frac{W}{14}\right\rfloor
$$

以 3360×1890 为例：

$$
\left\lfloor\frac{3360}{14}\right\rfloor=240,\quad
\left\lfloor\frac{1890}{14}\right\rfloor=135
$$

所以总 token 数为：

$$
N=240\times135=32400
$$

32,400 个视觉 token 如果直接做全局自注意力，复杂度近似是 $O(N^2D)$。这里的 $D$ 是隐藏维度，白话讲就是“每个 token 的向量长度”。当 $N$ 很大时，$N^2$ 会迅速失控。

第二层是窗口注意力。若每个窗口大小为 $8\times8$，则每个 token 只在局部窗口内做注意力，复杂度可近似写为：

$$
O(N\cdot W^2\cdot D)
$$

其中 $W=8$ 时，单层注意力规模从全局的 $N^2$ 降到 $N\cdot64$。对 32,400 个 token 来说，这个差距是数量级级别的。

第三层是位置恢复。图像切成 tile 后，模型必须知道“这个 tile 来自哪里”。常见做法是给每个 token 注入绝对坐标，或者使用 2D RoPE。RoPE 是旋转位置编码，白话讲就是“把位置信息直接写进向量旋转关系里”。2D RoPE 则把横向和纵向位置分别编码，让模型知道 token 的二维位置，而不是只知道一维顺序。

简化流程图如下：

```text
原图 H×W
  ↓
按宽高比和预算决定 tile 网格
  ↓
切成多个 448×448 子图
  ↓
每个 tile → patch embedding → 局部 window attention
  ↓
注入 2D RoPE / 绝对坐标
  ↓
按原始空间顺序拼接 token
  ↓
送入多模态语言模型
```

玩具例子：一张手机截图很长，宽 1080、高 4800。如果固定缩放，导航栏、输入框、商品价格会挤在一起。动态分辨率会优先沿长边增加 tile 数，比如切成 1×4 或 1×5 的 tile 网格，每块仍保持足够可读性。

真实工程例子：文档问答里，一页扫描 PDF 可能包含页眉、正文、表格、角注和签章。全图缩放后，正文还能看，角注通常已经糊掉。Qwen2.5-VL、InternVL 这类模型采用动态高分辨率策略，本质上就是为这种输入保留局部细节，同时让模型还能理解页面整体布局。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现，用来说明 tile 分配、28 倍数对齐和 token 预算检查。它不是完整训练代码，但逻辑和工程实现是一致的。

```python
import math

PATCH = 14
TILE_SIZE = 448
ALIGN = 28

def ceil_to_multiple(x: int, base: int) -> int:
    return ((x + base - 1) // base) * base

def dynamic_tile_plan(width: int, height: int, max_tiles: int = 12):
    # 先对齐到 backbone 常见的 28 倍数，避免后续网格错位
    width_a = ceil_to_multiple(width, ALIGN)
    height_a = ceil_to_multiple(height, ALIGN)

    aspect = width_a / height_a
    if aspect >= 1:
        cols = min(max_tiles, max(1, math.ceil(math.sqrt(max_tiles * aspect))))
        rows = max(1, math.ceil(max_tiles / cols))
    else:
        rows = min(max_tiles, max(1, math.ceil(math.sqrt(max_tiles / aspect))))
        cols = max(1, math.ceil(max_tiles / rows))

    while rows * cols > max_tiles:
        if cols >= rows:
            cols -= 1
        else:
            rows -= 1

    resized_w = cols * TILE_SIZE
    resized_h = rows * TILE_SIZE

    token_h = resized_h // PATCH
    token_w = resized_w // PATCH
    total_tokens = token_h * token_w

    return {
        "aligned_size": (width_a, height_a),
        "grid": (rows, cols),
        "resized_size": (resized_w, resized_h),
        "tokens": total_tokens,
    }

plan = dynamic_tile_plan(3360, 1890, max_tiles=12)

assert plan["aligned_size"][0] % 28 == 0
assert plan["aligned_size"][1] % 28 == 0
assert plan["grid"][0] * plan["grid"][1] <= 12
assert plan["tokens"] == (plan["resized_size"][0] // 14) * (plan["resized_size"][1] // 14)

print(plan)
```

这段代码做了三件事：

| 步骤 | 作用 | 为什么要做 |
|---|---|---|
| 对齐到 28 倍数 | 修正输入尺寸 | 避免 patch/grid 对不上 |
| 按宽高比分配 rows/cols | 决定 tile 网格 | 长图和宽图需要不同切法 |
| 计算 token 预算 | 约束显存和算力 | 防止单图占满 batch |

如果要更接近真实模型，前向流程通常类似下面的伪代码：

```python
def encode_image(image, max_tiles):
    tiles, coords = split_into_tiles(image, max_tiles=max_tiles, tile_size=448)
    tile_tokens = []
    for tile, coord in zip(tiles, coords):
        x = patch_embed(tile)
        x = add_2d_rope(x, coord)   # 记录绝对空间位置
        x = local_window_attention(x, window_size=8)
        tile_tokens.append(x)
    return merge_tokens(tile_tokens, coords)
```

这里的 `coords` 是关键。它记录每个 tile 在原图中的绝对位置，例如左上角坐标、缩放比例、tile 索引。如果忽略这些信息，模型虽然能看清局部，但很难判断“这个表头在页面顶部”还是“这个按钮在截图底部”。

一个输入尺寸到 tile 数和 token 预算的映射，可以粗略写成：

| 输入尺寸 | 可能网格 | tile 数 | 每 tile token | 总 token 量级 |
|---|---:|---:|---:|---:|
| 512×512 | 1×1 | 1 | 32×32=1024 | 1K |
| 1080×1920 | 2×3 | 6 | 1024 | 6K |
| 1890×3360 | 3×4 | 12 | 1024 | 12K |
| 多图 4 张 1080p | 每张 2×2 | 16 | 1024 | 16K |

这个表说明一个事实：动态分辨率不是“分得越细越好”，而是“在总 token 预算内尽量保留更多有效像素”。

---

## 工程权衡与常见坑

工程上最常见的问题，不是模型结构本身，而是输入约束没有守住。

第一类坑是尺寸不对齐。很多视觉 backbone 的下采样步长最终与 28 相关，如果输入尺寸不是 28 的整数倍，patch 网格、位置编码和后续特征图映射都可能出现偏移。表现出来就是框位置不稳、表格边界错位、局部区域语义漂移。

第二类坑是 tile 数超过训练上限。训练时如果模型只见过最多 12 或 40 个 tile，部署时直接喂 80 个 tile，模型未必崩，但分布已经变了，性能通常不稳定。正确做法不是盲目加 tile，而是建立明确的 token budget 监控。

第三类坑是只切块，不保留绝对位置。这样模型只能知道“某块里有文字”，不知道“这块文字在左栏顶部”。对于文档、图表和 GUI，这会直接影响推理结果。

第四类坑是长宽比处理粗暴。比如把超长截图直接压成正方形，虽然 tile 数减少，但空间关系已经被扭曲。动态分辨率的价值就在于尽量保留原始纵横比，而不是仅仅多切几块。

常见坑与规避策略如下：

| 坑点 | 后果 | 规避策略 |
|---|---|---|
| 非 28 倍数输入 | 网格错位、位置漂移 | resize 或 pad 到 28 的整数倍 |
| tile 总数超过 `n_max` | 显存爆炸或分布偏移 | 动态裁剪，先控预算再编码 |
| 忽略绝对坐标 | 空间尺度丢失 | 注入 2D RoPE 或显式坐标 |
| 长图强压正方形 | 布局关系扭曲 | 按宽高比决定 rows/cols |
| 多图 token 不均衡 | batch 利用率差 | 按图像预算分桶或截断 |
| 只做局部注意力 | 跨 tile 关系弱 | 周期性全局层或跨块聚合 |

下面是一个简单的预算检查片段：

```python
def check_token_budget(num_tiles: int, tile_size: int = 448, patch: int = 14, max_tokens: int = 16384):
    tokens_per_tile = (tile_size // patch) ** 2
    total = num_tiles * tokens_per_tile
    if total > max_tokens:
        raise ValueError(f"token budget exceeded: {total} > {max_tokens}")
    return total

assert check_token_budget(8) == 8 * (448 // 14) ** 2
```

真实工程例子：多页文档系统里，一次请求可能同时上传 6 张扫描页。如果每页都按最高分辨率切满 tile，总 token 很快超过模型上下文。此时通常要先做页面排序，再对正文页和附录页采用不同 tile 上限，而不是平均分配。

---

## 替代方案与适用边界

不是所有任务都需要动态分辨率与瓦片编码。如果输入本身分辨率不高，或者任务只关心全局语义，固定 resize 仍然是最省事的方案。

可以把方案对比成下面这样：

| 方案 | 分辨率保留 | 计算复杂度 | 并行能力 | 细节粒度 | 适用边界 |
|---|---:|---:|---:|---:|---|
| 全图固定 resize | 低 | 低 | 高 | 低 | 分类、粗粒度问答 |
| 滑动窗口 | 中到高 | 高 | 中 | 高 | 小目标扫描、局部检索 |
| 多尺度金字塔 | 中到高 | 中到高 | 中 | 中到高 | 检测、分割、结构理解 |
| 动态分辨率 + tile | 高 | 可控 | 高 | 高 | 文档、GUI、长图、多图 |
| 全局高分辨率 attention | 最高 | 极高 | 低 | 最高 | 小规模研究验证 |

一个简单判断规则是：

```text
低分辨率、单图、只看全局语义
  → 固定 resize 足够

高分辨率、细字、小目标、长图、多图
  → 动态分辨率 + tile 更合适
```

玩具例子：只有一张 512×512 商品图，要问“这是什么类别”，全图缩到 448×448 通常足够。  
真实工程例子：一张 4K 财报页面里，要问“第三列第二行的毛利率是多少”，如果还用固定 resize，模型输入里很可能已经没有可读数字，此时必须用 tile 策略。

要注意，这套方法也有适用边界。如果任务主要依赖跨大范围区域的细粒度关系，例如超长流程图中的远距离连接关系，单纯局部窗口注意力仍然可能不够，往往要叠加全局汇聚层、跨 tile 交互层，或者后续检索式读取机制。

---

## 参考资料

- Qwen2.5-VL Backbone: Dynamic-Resolution ViT。重点看原生动态分辨率、局部窗口注意力、视觉 token 组织方式。
- Qwen2.5-VL 相关技术解读。重点看高分辨率视觉输入如何与语言模型上下文对接，以及原始坐标表示的设计。
- InternVL 1.5 博客。重点看按宽高比将图像切成 1 到 40 个 448×448 tile 的策略。
- InternVL 官方文档。重点看动态高分辨率训练、图像和视频帧的 tile 分配规则。
- Vision Transformer 原始论文。重点看 patch embedding 与标准全局自注意力的计算特点。
- RoPE 与 2D 位置编码相关资料。重点看二维空间位置如何注入视觉 token。
