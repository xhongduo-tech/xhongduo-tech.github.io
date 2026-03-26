## 核心结论

InternVL2 和 InternVL3 的系统设计，核心不是“再堆一个更大的多模态模型”，而是先解决一个更基础的问题：**高分辨率图像里的细节，怎样在可控 token 预算内送进语言模型**。这里的 token 可以先理解为“模型看到的一小块输入单位”。

InternVL2 的关键组合是：**InternViT-6B 视觉编码器 + Dynamic Tiling 动态切图 + pixel unshuffle 像素重排降 token**。白话说，就是先把任意大图切成若干个 `448×448` 的瓦片，再把每块图像压成更少的视觉 token，这样模型既能看清小字、表格线、公式符号，又不至于因为输入太大把上下文撑爆。

InternVL3 延续这条路线，但把重点从“能看清”推进到“能在更长视觉上下文里持续理解”。它新增了 **V2PE**，即 Variable Visual Position Encoding，可变视觉位置编码。白话说，这是让模型更稳定地区分“这些视觉 token 在长序列里的相对位置”；同时它使用 **Native Multimodal Pre-Training**，即原生多模态预训练，把文本、图像、视频在同一个预训练阶段联合学习，不再完全依赖“先训语言模型，再外挂视觉模块”的串联流程。

一个适合新手记忆的版本是：**InternVL2/3 的方法不是把整张 4K 图硬塞进去，而是先切块，再压 token，再按顺序让模型读图。**这就是它们在 OCR、文档理解、图表问答上能和 GPT-4o、Claude 3.5 Vision 进入同一赛道的主要原因。

下表先给出结论对照：

| 维度 | InternVL2 | InternVL3 |
| --- | --- | --- |
| 核心目标 | 高分辨率图像与文档理解 | 长上下文、多图、视频理解 |
| 视觉主干 | InternViT-6B / 300M 系列 | InternViT-6B-V2_5 / 300M-V2_5 |
| 高分辨率策略 | Dynamic Tiling + pixel unshuffle | 延续 InternVL2 策略 |
| 新增能力 | 多图、视频训练数据已纳入 2.0 | V2PE、原生多模态预训练、长上下文更强 |
| 优势场景 | OCR、DocVQA、表格/扫描件 | 多图推理、视频、多步视觉上下文 |
| 与闭源模型关系 | 在部分 OCR/文档指标上已很接近，局部超过 | 继续向更广泛多模态推理逼近 |

---

## 问题定义与边界

问题定义可以写得非常直接：**如果目标是做一个开源 VLM，既要读高分辨率文档，又要回答语言问题，那么“视觉细节保留”和“token 成本可控”必须同时成立。**  
VLM 是 Vision-Language Model，白话说，就是既能看图又能生成文本的模型。

这类系统面对的典型输入不是普通自拍，而是：

| 任务 | 输入特点 | 为什么难 | 常见成功指标 |
| --- | --- | --- | --- |
| OCR 问答 | 小字密集、版式复杂 | 缩放后文字会糊 | OCRBench、TextVQA |
| 文档理解 | PDF 扫描件、表格、票据 | 既要识别字，也要理解结构 | DocVQA、InfoVQA |
| 图表理解 | 坐标轴、图例、数据标签 | 视觉解析和文本推理都要做 | ChartQA |
| 多图推理 | 多张图共同组成上下文 | 跨图位置与顺序易乱 | 多图 QA、自定义任务 |
| 视频理解 | 多帧连续变化 | 帧间关系和时间顺序更难 | 视频 QA、时序推理 |

边界也必须说清楚。

第一，InternVL2/3 不是“无限上下文、无限分辨率”的免费午餐。图像切得越细，视觉 token 越多，显存、推理延迟、批处理吞吐都会受影响。  
第二，它们特别擅长的是**高分辨率视觉输入下的理解**，尤其是文档、图表、OCR，而不是所有任务都全面压制商业模型。  
第三，和 GPT-4o、Claude 3.5 Vision 的比较必须建立在**统一评测工具和统一设置**上，否则同一个模型在不同工具链下分数会有偏差。

可以把需求总结成一句话：**把高分辨率 PDF 或长图拆成“模型可读块”，同时保留原始顺序和关键细节，让 LLM 最终还能输出结构化答案。**

---

## 核心机制与推导

先看最核心的 Dynamic Tiling。它的本质是**按输入图像大小动态决定要切多少个 `448×448` 的块**。  
tile 可以理解为“瓦片”，也就是切出来的一小块图像。

若输入尺寸为 $W \times H$，最直观的切块数近似可以写成：

$$
T \approx \max(1,\lceil W / 448 \rceil) \times \lceil H / 448 \rceil
$$

这个公式表达的不是精确工程实现细节，而是一个足够有用的近似：图越大，块越多；图越长，某一维上的块数就越多。

但只切块还不够，因为每个 tile 进入视觉编码器后仍会生成大量 token。InternVL 的第二步是 **pixel unshuffle**。  
pixel unshuffle 可以先理解成“把局部像素重新排布到通道里，用空间换通道，从而减少后续空间 token 数”。

如果 patch size 取 16，那么单个 `448×448` tile 的原始 patch token 数大致是：

$$
(448 / 16)^2 = 28^2 = 784
$$

经过 `2×2` 的 pixel unshuffle 后，空间边长等效减半，因此 token 数近似变成：

$$
(448 / 32)^2 = 14^2 = 196
$$

但 InternVL 官方在实际实现里把单张 `448×448` 图像表示为 **256 个视觉 token**。这说明工程上并不是简单套一个教科书公式，而是和具体视觉主干、下采样、投影设计绑定。对初学者更重要的结论是：**pixel unshuffle 的作用就是把视觉 token 压到原来的约四分之一量级**，让高分辨率输入变得可算。

因此总视觉 token 的量级可以近似理解为：

$$
N_{\text{visual}} \propto T \times N_{\text{tile}}
$$

其中 $N_{\text{tile}}$ 是每个 tile 压缩后的 token 数。在 InternVL 文档给出的设定里，`448×448 -> 256 token`，所以：

$$
N_{\text{visual}} \approx 256 \times T
$$

### 玩具例子

假设有一张 `896×448` 的长条图。

1. 宽度方向需要 `896 / 448 = 2` 个 tile。
2. 高度方向需要 `448 / 448 = 1` 个 tile。
3. 总 tile 数就是 `2 × 1 = 2`。
4. 若每 tile 近似为 256 token，总视觉 token 约为 `512`。

这个例子很小，但已经说明 Dynamic Tiling 的价值：**模型不是把整张图缩成一张模糊缩略图，而是保留局部清晰块。**

### 4K 数值例子

如果输入是 `4096×4096` 的图像，按近似切法会得到很多 tile。InternVL 官方文档给出的关键约束是：**测试时可 zero-shot 扩展到最多 40 个 `448×448` tile，支持 4K 输入**。如果按每 tile 256 个视觉 token 估算，那么上限视觉 token 约是：

$$
40 \times 256 = 10240
$$

这就是文章开头那句总结的真正含义：**先切，再压，再送入视觉编码器。**

下面用简化示意图表示流程：

```text
原始图像 W×H
   |
   v
按分辨率和长宽比切成多个 448×448 tiles
   |
   v
对每个 tile 做 pixel unshuffle
   |
   v
每个 tile 变成更少的视觉 tokens
   |
   v
InternViT-6B 编码
   |
   v
MLP projector 对齐到 LLM 词向量空间
   |
   v
LLM 做问答、摘要、表格抽取、跨图推理
```

InternVL3 在这条链路上加入的关键不是“再切得更细”，而是**让位置编码适配更长的视觉序列**。  
位置编码可以理解为“告诉模型每个 token 在哪里”的编号方法。

V2PE 的意义在于：多图、长图、视频帧混在一起时，视觉 token 的位置跨度更长、更复杂。如果位置编码仍按固定、粗粒度方式处理，模型容易在长上下文里丢失空间关系。InternVL3 通过更灵活的视觉位置增量，使多图和视频输入下的长上下文理解更稳定。

---

## 代码实现

实现上可以拆成三层：`preprocess -> vision encoder -> multimodal head`。

| 模块 | 作用 | 典型参数 |
| --- | --- | --- |
| preprocess | 切 tile、记录顺序、生成位置 | `tile_size=448`、`max_tiles=40` |
| vision encoder | 把 tile 变成视觉特征 | InternViT-6B / 300M |
| projector + LLM | 把视觉特征接到语言模型 | MLP projector、Qwen/InternLM 系列 |
| V2PE / frame pos | 多图和视频的位置建模 | 视觉位置步长、帧索引 |

最小伪代码如下：

```python
tiles = split_image(image, tile_size=448)
tokens = [pixel_unshuffle(tile) for tile in tiles]
visual_inputs = stack(tokens)
visual_feats = vision_encoder(visual_inputs)
llm_inputs = projector(visual_feats)
answer = llm.generate(llm_inputs, prompt)
```

下面给一个可运行的 Python 玩具实现，只模拟“切块数量”和“token 预算估算”，不依赖深度学习框架：

```python
import math

def estimate_tiles(width, height, tile_size=448, max_tiles=40):
    tiles_w = max(1, math.ceil(width / tile_size))
    tiles_h = max(1, math.ceil(height / tile_size))
    total = tiles_w * tiles_h
    return min(total, max_tiles)

def estimate_visual_tokens(width, height, tile_size=448, tokens_per_tile=256, max_tiles=40):
    tiles = estimate_tiles(width, height, tile_size=tile_size, max_tiles=max_tiles)
    return tiles * tokens_per_tile

def split_order(width, height, tile_size=448):
    tiles_w = max(1, math.ceil(width / tile_size))
    tiles_h = max(1, math.ceil(height / tile_size))
    order = []
    for r in range(tiles_h):
        for c in range(tiles_w):
            order.append((r, c))
    return order

# 玩具例子：896x448 -> 2 个 tile
assert estimate_tiles(896, 448) == 2
assert estimate_visual_tokens(896, 448) == 512

# 官方常见说明：4K 输入可扩展到最多 40 tiles
assert estimate_tiles(4096, 4096) == 40
assert estimate_visual_tokens(4096, 4096) == 10240

# OCR 顺序必须稳定，先行后列
assert split_order(896, 896) == [(0, 0), (0, 1), (1, 0), (1, 1)]

print("all assertions passed")
```

真实工程里，预处理不能只返回 tile，还要返回三类元数据：

1. `tile_index`：第几个 tile。
2. `grid_pos`：它在原图网格中的行列位置。
3. `source_id/frame_id`：它属于哪张图、哪一帧。

如果没有这三类信息，多图或视频任务很容易出现“内容看到了，但顺序错了”的问题。InternVL3 的 V2PE，本质上就是把这类长视觉上下文的位置关系建模得更自然。

### 真实工程例子

企业文档流水线里，一个常见任务是处理扫描 PDF：合同、报销单、医疗文档、银行回单。输入通常是高清长页，包含小字、表格、章印、手写内容。一个更稳的流程不是直接让模型“看整页回答”，而是：

1. 按页渲染为高分辨率图。
2. 动态切成 tiles，并记录阅读顺序。
3. 视觉编码后送入统一的问答模板。
4. 输出固定 JSON 或 Markdown 表格。
5. 再做字段校验和人工抽检。

这时 InternVL2 的价值主要体现在“识别与理解都在同一模型内完成”，不必先 OCR 再规则拼接；InternVL3 的价值则进一步体现在“多页、多图、多帧共同推理”时更稳。

---

## 工程权衡与常见坑

最关键的工程权衡是：**高分辨率保真和计算成本一定互相拉扯。**

| 挑战 | 具体风险 | 规避措施 |
| --- | --- | --- |
| tile 太多 | 延迟高、显存涨、吞吐下降 | 设 `max_tiles`，按任务分级输入 |
| tile 顺序丢失 | OCR 文字顺序错、表格列错位 | 显式保存 `(page, row, col)` |
| 评测工具不一致 | 分数对不上，误判模型能力 | 固定 InternVL / VLMEvalKit 版本 |
| 只看单项高分 | 真实业务效果不稳定 | 增加端到端字段抽取评估 |
| 多图输入混乱 | 图 A 与图 B 内容串台 | 给每张图单独 source id |
| 视频帧过密 | 成本大但收益不明显 | 做关键帧采样与帧间压缩 |

这里有一个常见误解：**开源模型就一定全面落后于闭源模型。**  
从 InternVL2 官方 benchmark 看，这种判断并不成立。以官方页面列出的分数为例，InternVL2-40B 在 OCRBench 上是 **837**，高于表中的 Claude 3.5 Sonnet 的 **788**；在 DocVQA 上，InternVL2-40B 是 **93.9**，已经接近 GPT-4o 的 **92.8** 和 Claude 3.5 Sonnet 的 **95.2**。这说明至少在文档/OCR 这一类任务上，InternVL 已经不是“只能做开源替代品”的位置。

但另一个坑也要避免：**不能把这些分数直接理解成“全面超过商业模型”。**  
原因有三点：

1. 指标只覆盖部分能力，不覆盖稳定性、工具调用、复杂代理流程。
2. 同一模型在不同测试工具下可能有轻微差异。
3. 工程系统最终看的是业务指标，不是 leaderboard 截图。

因此更稳的做法是把评估拆成两层：

| 层级 | 看什么 |
| --- | --- |
| 基准评测 | OCRBench、DocVQA、ChartQA、TextVQA |
| 业务评测 | 字段抽取准确率、页级召回率、人工复核率、平均延迟 |

对企业流水线而言，真正的坑往往不是“模型不够强”，而是“切图和回拼不够严谨”。例如一份扫描合同被切成 12 个 tiles，如果你没有把 tile 顺序和页码回传给后处理模块，那么最终生成的表格可能会把页脚金额拼到页眉字段里。这种错误和模型智力无关，纯粹是系统设计失误。

---

## 替代方案与适用边界

InternVL2/3 很强，但不意味着所有视觉任务都该上这条路线。选型应该先看输入分辨率、结构复杂度、是否需要跨图推理。

| 方案 | 适合场景 | 局限 |
| --- | --- | --- |
| InternVL2 | 高分辨率文档、OCR、图表问答 | 多图/视频长上下文不如新版本 |
| InternVL3 | 多图、视频、长上下文视觉推理 | 成本更高，系统复杂度更高 |
| 标准 ViT/CLIP + OCR | 截图检索、简单文本识别 | 全局理解和问答弱 |
| OCR 引擎 + 规则系统 | 固定模板票据、表单 | 模板变化后脆弱 |
| 商业闭源 VLM API | 快速上线、工程门槛低 | 成本、数据合规、可控性受限 |
| 专用 video-LLM | 视频问答、时序事件识别 | 文档 OCR 不一定占优 |

可以用一句很实用的话做边界判断：

**需要全景文档理解、复杂版式、跨页或多图推理时，用 InternVL2/3。只需要少量截图上的文字搜索或关键词抽取时，用传统 OCR + Q&A 往往更便宜。**

再给一个切换边界：

1. 如果输入通常低于 `1080p`，且主要是自然图像问答，Dynamic Tiling 的优势不一定明显。
2. 如果任务核心是固定字段提取，如“发票号码、税额、日期”，传统 OCR + 模板规则可能性价比更高。
3. 如果任务核心是视频时序理解，如“第几秒发生了什么”，应优先考虑 InternVL3 这类支持视频输入的方案，或更专门的 video-LLM。
4. 如果业务要求强隐私、低延迟、可离线部署，开源方案会比纯 API 调用更合适。

所以，InternVL2/3 的真正适用边界不是“它是不是最强”，而是：**你的问题是否真的需要在高分辨率、多图、多帧条件下保留细粒度视觉信息，并把这些信息直接接入语言推理。**

---

## 参考资料

| 标题 | URL | 说明 |
| --- | --- | --- |
| Introduction of InternVL2 Series | https://internvl.readthedocs.io/en/latest/internvl2.0/introduction.html | 官方 2.0 架构、Dynamic Tiling、benchmark |
| Introduction of InternVL3.0 Series | https://internvl.readthedocs.io/en/latest/internvl3.0/introduction.html | 官方 3.0 架构、V2PE、原生多模态预训练 |
| Introduction of InternVL 1.5 Series | https://internvl.readthedocs.io/en/latest/internvl1.5/introduction.html | 高分辨率动态切图与 pixel unshuffle 的早期公开说明 |
| InternVL GitHub Repository | https://github.com/OpenGVLab/InternVL | 官方代码仓库，含部署与评测入口 |
| InternVL3 Paper (arXiv) | https://arxiv.org/abs/2504.10479 | InternVL3 论文入口，适合继续追训练细节 |
