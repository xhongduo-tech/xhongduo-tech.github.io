## 核心结论

Claude 3 的多模态能力，公开资料能确认的核心事实有两点。

第一，图像不是作为“外挂功能”单独处理，而是被转换成会占用上下文预算的视觉 token，再和文本一起参与同一次推理。这里的 token 可以先理解成“模型内部处理信息的最小片段”。Anthropic 官方文档直接给出了图像 token 的估算公式，并说明图像会影响 time-to-first-token，这说明图像输入已经进入主推理流水线，而不是先离线做一段固定摘要再喂给语言模型。

第二，Claude 3 Opus 在 MMMU 上的公开成绩是 59.4%。MMMU 是一个跨学科多模态理解基准，可以把它理解成“带图的大学级综合题库”。这个分数不能单独证明内部层级结构，但它说明 Claude 3 的视觉理解不是停留在 OCR 或图片分类，而具备跨图文联合推理能力。

把机制抽象成工程图，最稳妥的理解是：

`图像 -> 分片/编码 -> 视觉 token -> 与文本 token 共同进入统一推理上下文`

这里常见的业内实现会使用 ViT-like 编码器，也就是“把图片切成很多小块再编码的视觉 Transformer 前端”。但要注意，Anthropic 并没有完整公开 Claude 3 的逐层内部架构，因此“ViT 变体 + 投影对齐 + 共享 Transformer”更适合作为工程抽象，而不是官方逐层电路图。

| 模型 | 基准 | 分数 | 说明 |
| --- | --- | ---: | --- |
| Claude 3 Opus | MMMU | 59.4 | 公开多模态综合成绩 |
| Claude 3 Sonnet | MMMU | 53.1 | 同代较小模型 |
| Claude 3 Haiku | MMMU | 50.2 | 同代轻量模型 |

---

## 问题定义与边界

这篇文章讨论的不是“Claude 3 会不会看图”，而是两个更具体的问题：

1. 图像怎样变成能和文本一起推理的输入。
2. 在工程上，图像尺寸、token 预算、请求大小会如何限制系统设计。

先看边界。Anthropic Vision 文档给出的关键约束如下。

| 约束项 | 数值 | 工程含义 |
| --- | ---: | --- |
| 单次请求图片数 | API 最多 100 张 | 多图对比可做，但很容易撞请求上限 |
| `claude.ai` 单次图片数 | 最多 20 张 | 产品端更保守 |
| 标准端点请求体 | 32MB | 图片多时先卡带宽和上传大小 |
| 单图拒绝阈值 | 超过 8000×8000 px 直接拒绝 | 需要前置校验 |
| 超过 20 张图片时单图限制 | 2000×2000 px | 多图批处理要主动降分辨率 |
| 推荐分辨率 | 不超过 1.15 MP | 降低首 token 延迟 |
| 推荐边长 | 两边不超过 1568 px | 超过会触发自动缩放 |
| 小图风险 | 任一边小于 200 px 可能降效果 | 缩略图常常不可靠 |

如果图像不需要重采样，官方给出的估算公式是：

$$
\text{image\_tokens} \approx \frac{width \times height}{750}
$$

玩具例子：一张 1000×1000 的截图，大约会占

$$
\frac{1000 \times 1000}{750} \approx 1333
$$

也就是约 1334 个视觉 token。

这对新手最重要的一点是：上传图片不是“免费附件”，而是实打实消耗上下文和成本的输入。比如你做一个 PPT 分析工具，用户上传 12 张高分辨率页图时，真正先爆掉的往往不是模型能力，而是请求体大小、自动缩放延迟和图像 token 预算。

---

## 核心机制与推导

公开资料没有给出 Claude 3 的完整层级图，但从 Vision API 的行为可以推导出一个可靠的最小机制模型。

1. 图像先被切分并编码。
2. 编码结果被表示成一串视觉 token。
3. 这些 token 和文本 token 一起进入同一轮上下文推理。
4. 模型输出文本答案。

“视觉编码器”可以先理解成“把像素压缩成语义向量的前端网络”。“投影对齐”可以理解成“把视觉向量映射到语言模型能直接接收的维度空间”。如果用业内最常见的实现范式描述，就是：

`patchify -> vision encoder -> projection -> concatenate with text tokens -> shared attention`

共享 attention 的意思是：文本问题可以直接去“看”图像 token，图像中的局部区域也会在上下文里影响文本推理。它不是先得到一段固定图片摘要，再把摘要扔给语言模型，而是让模型在回答时继续访问视觉表示。

玩具例子：把一张 1092×1092 的图想成被切成约 1600 个小块。模型不是只记住“这是一张图表”，而是保留很多局部块的表示，例如标题区域、坐标轴区域、图例区域、柱状条区域。这样当用户问“第三季度哪个品类增长最快”时，文本 token 可以在注意力里去查询与“第三季度”“增长”相关的视觉区域。

真实工程例子：做发票审核时，系统同时输入“请提取发票号、税额，并判断总额是否等于各项求和”和发票图片。此时问题文本会驱动模型在图片里关注编号区、金额区、表格区。真正有价值的不是 OCR 出文本本身，而是图文联合约束下的结构推理：字段在哪里、字段之间是否一致、表格合计是否正确。

从尺寸限制也能反推它为什么需要前置压缩。若单图被控制在约 1600 个视觉 token，模型在多图场景下的总注意力开销才不会失控。因为自注意力的计算成本通常随序列长度近似按 $O(n^2)$ 增长。这里的 $n$ 可以粗略理解成“文本 token + 所有视觉 token 的总和”。图越多、越大，延迟越高，这是统一上下文架构必须面对的物理成本。

---

## 代码实现

下面的代码演示一个最小可运行方案：先估算图像 token，再按 1568 边长和 1.15MP 约束缩放，最后组装请求载荷。代码本身不依赖 Anthropic SDK，重点在前置预算逻辑。

```python
import math

MAX_EDGE = 1568
MAX_PIXELS = 1_150_000
REJECT_EDGE = 8000

def estimate_image_tokens(width: int, height: int) -> int:
    assert width > 0 and height > 0
    return math.ceil(width * height / 750)

def resize_keep_ratio(width: int, height: int):
    assert width <= REJECT_EDGE and height <= REJECT_EDGE

    scale = 1.0

    if max(width, height) > MAX_EDGE:
        scale = min(scale, MAX_EDGE / max(width, height))

    if width * height * scale * scale > MAX_PIXELS:
        scale = min(scale, math.sqrt(MAX_PIXELS / (width * height)))

    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return new_w, new_h

def preprocess_image_meta(width: int, height: int):
    new_w, new_h = resize_keep_ratio(width, height)
    tokens = estimate_image_tokens(new_w, new_h)
    return {
        "original": (width, height),
        "resized": (new_w, new_h),
        "tokens": tokens,
        "needs_resize": (new_w, new_h) != (width, height),
    }

toy = preprocess_image_meta(1092, 1092)
assert toy["tokens"] in range(1580, 1605)

ppt = preprocess_image_meta(2400, 1800)
assert ppt["resized"][0] <= 1568
assert ppt["resized"][1] <= 1568
assert ppt["resized"][0] * ppt["resized"][1] <= MAX_PIXELS

print("toy:", toy)
print("ppt:", ppt)
```

如果你在客户端接 API，请把检查逻辑放在上传前，而不是报错后重试。伪代码如下：

```python
payload = {
    "model": "claude-3-opus-20240229",
    "max_tokens": 1024,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": "..."},
                {"type": "text", "text": "请比较这张图里的趋势，并解释异常点。"},
            ],
        }
    ],
}
```

真实工程例子：做文档分析平台时，建议前端就做三层限制。

| 检查层 | 目的 | 动作 |
| --- | --- | --- |
| 尺寸检查 | 防止 8000 px 拒绝 | 上传前拦截 |
| 分辨率检查 | 防止自动缩放拖慢延迟 | 先本地压缩到 1.15MP 左右 |
| 请求体检查 | 防止 32MB 超限 | 多图时改传 URL 或分批请求 |

---

## 工程权衡与常见坑

统一图文推理的优势是语义一致，代价是输入预算更贵、延迟更敏感。

最常见的坑不是“模型看不懂”，而是“你把输入喂坏了”。

| 常见坑 | 症状 | 原因 | 缓解策略 |
| --- | --- | --- | --- |
| 长边超过 1568 px | 首 token 变慢 | 服务端先重采样 | 客户端预缩放 |
| 图片超过约 1600 token | 费用和延迟上升 | 图像序列过长 | 上传前估算 token |
| 超过 8000×8000 px | 直接拒绝 | 超出硬限制 | UI 层前置校验 |
| 任一边小于 200 px | 回答模糊 | 细节不可读 | 不要传缩略图 |
| 多张高清图一起传 | 触发 32MB 限制 | 请求体过大 | 压缩、分批、URL 化 |
| 图片放在文本后面 | 效果不稳定 | 提示结构较弱 | 优先 image-then-text |

一个实用经验是，不要把“模型能收 100 张图”理解成“系统就应该一次传 100 张图”。官方说的是 API 支持，不等于这是高质量和低延迟的默认工作点。对零基础到初级工程师来说，更稳的产品策略通常是：单次 4 到 12 张图，分辨率控制在 1.15MP 内，只在确实需要跨图比较时才做多图同轮推理。

---

## 替代方案与适用边界

Claude 3 这类统一图文流水线，适合“图像内容必须和文本指令一起推理”的任务。但不是所有视觉任务都该用它。

| 方案 | 适合场景 | 优势 | 劣势 |
| --- | --- | --- | --- |
| Claude 3 统一多模态 | 图文联合问答、图表解释、文档审核 | 语义一致，交互自然 | 输入贵，延迟更高 |
| Dual-encoder/两段式 | 先图像理解，再文本推理 | 可拆分优化，缓存方便 | 视觉信息可能在摘要阶段丢失 |
| Vision-only 模型 | 分类、检测、简单 OCR | 快、便宜、吞吐高 | 长文本交互弱 |
| OCR + LLM | 表单抽取、规则化文档 | 可控、可审计 | 对版面和图像关系理解弱 |

新手可以这样判断：

如果任务是“这张图里有什么”，Vision-only 或 OCR 往往更经济。  
如果任务是“结合图和我这段要求做判断”，统一多模态更合适。  
如果任务是“几百页文档先粗提取，再少量复杂问答”，两段式通常更稳。

也就是说，Claude 3 的价值不在于替代所有视觉模型，而在于当“图片本身是推理上下文的一部分”时，统一流水线更自然，也更少中间信息损失。

---

## 参考资料

- 官方文档：Anthropic Vision 指南，说明图片数量限制、32MB 请求体、8000px 拒绝阈值、1568px 推荐边界、1.15MP 建议值，以及 `tokens = (width * height) / 750` 的估算公式  
  https://platform.claude.com/docs/en/build-with-claude/vision

- 官方资料：Anthropic Claude 3 系统卡，包含 Claude 3 Opus / Sonnet / Haiku 在 MMMU 上的公开成绩，其中 Opus 为 59.4  
  https://www.anthropic.com/system-cards

- 基准主页：MMMU 数据集与迷你排行榜，可交叉核对 Claude 3 Opus 的 59.4 分数  
  https://huggingface.co/datasets/MMMU/MMMU

- 会议论文：MMMU 论文与相关评测背景，适合深入理解这个基准测什么、为什么难  
  https://openreview.net/forum?id=XzBb6-0y_F

- 补充阅读：Anthropic PDF 支持文档，说明 PDF 视觉理解复用了 vision 能力，因此会继承相同的输入限制与预算约束  
  https://docs.anthropic.com/en/docs/build-with-claude/pdf-support
