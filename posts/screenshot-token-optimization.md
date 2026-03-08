## 核心结论

屏幕截图的 token 优化，本质上是在模型“看图”之前先控制输入面积、细节等级和更新频率。**token** 可以理解为模型处理信息时消耗的“计费单位”；截图越大、细节越高、发送越频繁，消耗越快。对 Computer Use 这类要连续观察屏幕的 agent 来说，截图通常不是小开销，而是主要成本项。

最重要的判断只有一句：**不要把“整张屏幕的每一帧高细节截图”当成默认方案**。更可行的组合通常是：

| 策略 | 作用 | 典型效果 | 适合场景 |
|---|---:|---:|---|
| 裁剪 ROI | 只保留关注区域 | token 降 40% 到 70% | 按钮、表单、弹窗操作 |
| 降低分辨率 | 减少平铺块数 | token 再降 20% 到 50% | 页面结构已知 |
| `low detail` | 只保留全局轮廓 | 单张可低到约 85 token | 导航、状态巡检 |
| 差分截图 | 只在有变化时上传 | 多轮累计省 60% 到 80% | 连续交互、轮询任务 |
| 关键帧 | 只保留重要时刻高细节图 | 大幅降低上下文占用 | 多步骤自动化 |

一个新手最容易感受到的例子是：同样是“看网页上的一个按钮”，1280×800 的整图高细节输入，可能要先缩放再切成多个小块，一张就是千级 token；如果只裁剪到按钮附近 512×320，再用低细节模式，消耗会立刻掉到几百甚至更低。区别不是“略微优化”，而是“能不能长期运行”。

先给一个实用决策表，便于把结论直接落到工程上：

| 任务目标 | 默认截图方案 | 什么时候升级到高细节 |
|---|---|---|
| 判断页面是否跳转成功 | 全局 `low detail` | 页面结构异常、状态不明确 |
| 点击已知位置按钮 | 按钮附近 ROI | 按钮文字太小、状态不可辨认 |
| 填表与校验 | 当前输入区域 ROI | 需要确认校验提示、错误信息 |
| 等待异步加载 | 全局 `low detail` + 差分 | 加载完成后抓一张关键帧 |
| OCR 读细小文本 | 局部高细节 | 文本跨区域、需要整段上下文 |

如果只记一条工程规则，可以记成：

$$
\text{默认传输} = \text{低细节全局感知} + \text{局部高细节关键帧}
$$

而不是：

$$
\text{默认传输} = \text{整图高细节连续上传}
$$

---

## 问题定义与边界

这里讨论的问题，不是 JPEG 文件能不能更小，也不是 base64 文本能不能更短，而是**视觉 token** 的控制。视觉 token 可以理解为模型内部为了“读懂图片”而分配的表示预算，很多时候它和图片文件字节数不是一回事。

问题边界要先说清楚：

1. 我们关心的是**多轮连续截图**。
2. 我们关心的是**模型上下文和调用成本**。
3. 我们关心的是**在不丢失关键界面信息前提下减少输入量**。

不在本文范围内的情况包括：单张离线精读图片、OCR 精确识别整页文档、医学影像这类宁可贵也不能漏的任务。这些任务更适合保留高细节，而不是优先压 token。

可以把问题抽象成一个简单关系：

$$
\text{总成本} \approx \text{单张截图 token} \times \text{截图次数}
$$

再展开一点：

$$
\text{单张截图 token} \propto \text{面积} \times \text{detail 等级} \times \text{是否整图上传}
$$

如果任务是多轮交互，更准确的工程表达是：

$$
\text{总 token} = \sum_{i=1}^{N} \text{token}(\text{frame}_i)
$$

而当系统采用“全局低细节 + 局部关键帧”后，它会变成：

$$
\text{总 token} \approx N_{low}\cdot T_{low} + \sum_{j=1}^{K} T_{high}^{(j)}
$$

其中：

- $N_{low}$ 是低细节全局图的张数；
- $T_{low}$ 是每张低细节图的固定或近固定成本；
- $K$ 是高细节关键帧数量；
- $T_{high}^{(j)}$ 是第 $j$ 张局部高细节图的 token。

这个公式比“每张都高细节”更贴近真实系统，因为绝大多数时刻，模型并不需要重新阅读整个屏幕。

所以连续截图为什么容易出问题，不是因为某一张特别贵，而是因为它会线性累积。比如一个 agent 每 5 秒截一张图，2 分钟就是 24 张。如果每张都在 2,000 token 左右，那么光视觉输入就接近 48K token，后面的推理、工具调用结果、用户指令都要和这 48K 竞争上下文空间。

一个“玩具例子”足够说明边界：假设你的任务只是点击“提交”按钮。真正重要的信息只有按钮附近约 500×300 的区域，但如果你每次都上传整张 1280×800 屏幕，那么浏览器边框、侧栏、空白边距、状态栏都会被计入视觉输入。它们对点击“提交”几乎没帮助，却真实占掉 token。

一个“真实工程例子”是远程桌面自动化。agent 需要登录后台、展开菜单、填写表单、等待异步加载、检查提交结果。这个过程通常有十几到几十步。如果没有截图压缩策略，系统很容易出现两类问题：一类是成本飙升，另一类是上下文过长后，模型开始遗忘前几步的界面状态。

新手还需要区分三个容易混淆的概念：

| 概念 | 它优化的是什么 | 是否直接影响视觉 token |
|---|---|---|
| PNG/JPEG 压缩率 | 网络传输字节数 | 不一定 |
| base64 长度 | 请求载荷文本长度 | 通常不是核心因素 |
| ROI / detail / tile 数 | 模型实际读取的视觉预算 | 是 |

因此，本文真正讨论的是第三类优化。

---

## 核心机制与推导

先看最常被引用的一类计费逻辑。以 OpenAI 文档中的 GPT-4o 高细节模式为代表，核心思想是：**图片先缩放，再按固定大小平铺，token 与平铺块数成正比**。这里的 **tile** 就是“把大图切成若干 512×512 小块后，每一块单独计算成本”。

常见近似公式可写成：

$$
\text{token}_{high} = 85 + 170 \times \text{tiles}
$$

其中：

$$
\text{tiles} = \left\lceil \frac{width}{512} \right\rceil \times \left\lceil \frac{height}{512} \right\rceil
$$

如果短边先被缩到 768，那么宽高应先按比例缩放后再代入上式。对 `low detail`，OpenAI 文档给出的典型口径是固定成本，GPT-4o 约为 85 token。

不同平台细节不同，但共性很强：**面积越大，token 越多；整图越完整，成本越高**。

| 平台/模式 | 近似规则 | 直观含义 |
|---|---|---|
| OpenAI `high detail` | $85 + 170 \times tiles$ | 平铺块越多越贵 |
| OpenAI `low detail` | 约 85 token | 用粗粒度全局表示看图 |
| Claude 视觉输入 | 近似按像素面积估算 | 总像素越多越贵 |
| Gemini 图像输入 | 分辨率越高，输入 token 越高 | 更大图片消耗更多输入预算 |

现在做一个必须掌握的推导。

原图：1280×800。  
高细节模式下，先把短边从 800 缩到 768。缩放比例为：

$$
r = \frac{768}{800} = 0.96
$$

那么新宽度约为：

$$
1280 \times 0.96 \approx 1229
$$

于是平铺块数：

$$
\text{tiles} = \left\lceil \frac{1229}{512} \right\rceil \times \left\lceil \frac{768}{512} \right\rceil = 3 \times 2 = 6
$$

最终 token：

$$
85 + 170 \times 6 = 1105
$$

这就是“单张 1280×800 高细节截图约 1105 token”的来源。

再看裁剪后的玩具例子。假设你只保留一个 512×320 的区域：

$$
\text{tiles} = \left\lceil \frac{512}{512} \right\rceil \times \left\lceil \frac{320}{512} \right\rceil = 1 \times 1 = 1
$$

则高细节 token 为：

$$
85 + 170 \times 1 = 255
$$

从 1105 到 255，节省约：

$$
1 - \frac{255}{1105} \approx 76.9\%
$$

这个数字比“减少 60%”还激进，但它很合理，因为裁剪不仅减少像素，还直接减少了 tile 数。对视觉模型来说，tile 数往往比“文件压缩率”更关键。

再看一个很多人会忽略的反例。假设你把 1280×800 整图缩小到 1100×688，但缩完后仍然需要：

$$
\left\lceil \frac{1100}{512} \right\rceil \times \left\lceil \frac{688}{512} \right\rceil = 3 \times 2 = 6
$$

那么 token 仍然接近原值。也就是说：

**分辨率下降，不等于 token 一定下降；只有 tile 数下降，成本才会明显下降。**

因此可以得到三个工程上非常实用的推论：

1. 裁剪 ROI 最直接，因为它立刻减少 tile。
2. 降低分辨率只有在能减少 tile 时才真正省钱。
3. 差分截图最强，因为它不是优化单张，而是减少“发送次数 × 发送面积”。

还可以再补一个推论：

4. `low detail` 适合回答“界面现在大概是什么状态”，不适合回答“这个小字到底写了什么”。

把这些推论映射到任务类型，会更容易理解：

| 任务问题 | 需要的信息粒度 | 推荐 detail |
|---|---|---|
| “页面是不是已经跳到订单详情了？” | 低 | `low detail` |
| “这个按钮是禁用还是可点击？” | 中 | 小 ROI `high detail` |
| “错误提示具体是哪一行文案？” | 高 | 更大 ROI `high detail` |
| “表格第 7 行第 3 列写的是什么？” | 高 | 局部高分辨率，必要时 OCR |

---

## 代码实现

实现上不要把它看成“截图后直接上传”，而应看成一条预处理流水线：

`截屏 -> 检测 ROI -> 裁剪 -> 缩放 -> 估算 token -> 判断是否差分发送 -> 选择 detail -> 上传`

下面给一个**可直接运行**的 Python 示例。它不依赖真实截图库，只模拟截图元数据、变化区域和策略选择，重点是把决策逻辑写清楚，并且可以用 `python demo.py` 直接验证公式与输出。

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class Frame:
    width: int
    height: int
    changed_regions: Tuple[Region, ...] = ()

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class UploadPlan:
    detail: str
    region: Region
    tokens: int
    reason: str


FULL_FRAME_ORIGIN = Region(0, 0, 0, 0)


def full_region(frame: Frame) -> Region:
    return Region(0, 0, frame.width, frame.height)


def scale_to_short_edge(width: int, height: int, target_short: int = 768) -> Tuple[int, int]:
    short_edge = min(width, height)
    if short_edge <= target_short:
        return width, height
    ratio = target_short / short_edge
    return round(width * ratio), round(height * ratio)


def gpt4o_high_tokens(width: int, height: int, target_short: int = 768) -> int:
    scaled_w, scaled_h = scale_to_short_edge(width, height, target_short)
    tiles = math.ceil(scaled_w / 512) * math.ceil(scaled_h / 512)
    return 85 + 170 * tiles


def gpt4o_low_tokens() -> int:
    return 85


def union_bbox(regions: Tuple[Region, ...]) -> Optional[Region]:
    if not regions:
        return None
    left = min(r.x for r in regions)
    top = min(r.y for r in regions)
    right = max(r.x + r.width for r in regions)
    bottom = max(r.y + r.height for r in regions)
    return Region(left, top, right - left, bottom - top)


def expand_region(region: Region, frame: Frame, padding: int = 24) -> Region:
    x1 = max(0, region.x - padding)
    y1 = max(0, region.y - padding)
    x2 = min(frame.width, region.x + region.width + padding)
    y2 = min(frame.height, region.y + region.height + padding)
    return Region(x1, y1, x2 - x1, y2 - y1)


def changed_pixel_ratio(frame: Frame) -> float:
    changed_pixels = sum(r.area for r in frame.changed_regions)
    return changed_pixels / frame.area if frame.area else 0.0


def detect_focus_region(frame: Frame, fallback: Optional[Region] = None) -> Region:
    bbox = union_bbox(frame.changed_regions)
    if bbox is not None:
        return expand_region(bbox, frame, padding=24)
    if fallback is not None:
        return fallback
    return full_region(frame)


def choose_upload_plan(
    frame: Frame,
    last_action_roi: Optional[Region] = None,
    diff_threshold: float = 0.10,
) -> List[UploadPlan]:
    plans: List[UploadPlan] = []

    plans.append(
        UploadPlan(
            detail="low",
            region=full_region(frame),
            tokens=gpt4o_low_tokens(),
            reason="保留全局状态感知，成本固定",
        )
    )

    roi = detect_focus_region(frame, fallback=last_action_roi)
    ratio = changed_pixel_ratio(frame)

    # 如果变化很小，就只发低细节全局图
    if ratio < diff_threshold:
        return plans

    plans.append(
        UploadPlan(
            detail="high",
            region=roi,
            tokens=gpt4o_high_tokens(roi.width, roi.height),
            reason=f"变化比例 {ratio:.1%} 超过阈值 {diff_threshold:.0%}，发送局部高细节关键帧",
        )
    )
    return plans


def estimate_session_tokens(plans_per_step: List[List[UploadPlan]]) -> int:
    return sum(plan.tokens for step in plans_per_step for plan in step)


def main() -> None:
    # 1) 验证文中的公式
    assert gpt4o_high_tokens(1280, 800) == 1105
    assert gpt4o_high_tokens(512, 320) == 255
    assert gpt4o_low_tokens() == 85

    # 2) 模拟一个“列表页 -> 详情页 -> 提交结果”的三步任务
    last_action_roi = Region(300, 200, 512, 320)

    step1 = Frame(
        width=1280,
        height=800,
        changed_regions=(),  # 几乎没有变化，只需要低细节巡检
    )

    step2 = Frame(
        width=1280,
        height=800,
        changed_regions=(
            Region(320, 180, 460, 300),  # 中间详情面板有明显变化
        ),
    )

    step3 = Frame(
        width=1280,
        height=800,
        changed_regions=(
            Region(880, 96, 180, 52),   # 顶部 toast
            Region(640, 690, 140, 50),  # 提交按钮状态变化
        ),
    )

    steps = [step1, step2, step3]
    all_plans: List[List[UploadPlan]] = []

    for index, frame in enumerate(steps, start=1):
        plans = choose_upload_plan(frame, last_action_roi=last_action_roi, diff_threshold=0.10)
        all_plans.append(plans)
        print(f"Step {index}")
        for plan in plans:
            r = plan.region
            print(
                f"  detail={plan.detail:<4} "
                f"region=({r.x},{r.y},{r.width},{r.height}) "
                f"tokens={plan.tokens:<4} "
                f"reason={plan.reason}"
            )

    total = estimate_session_tokens(all_plans)
    naive = len(steps) * gpt4o_high_tokens(1280, 800)

    print("\nSummary")
    print(f"  optimized tokens = {total}")
    print(f"  naive full-high  = {naive}")
    print(f"  saved            = {1 - total / naive:.1%}")


if __name__ == "__main__":
    main()
```

这段代码表达的是一个很实用的模式：

1. 每一轮先发一张 `low detail` 全局图，成本固定、便于模型保持场景感知。
2. 只在变化比例超过阈值时，再发 ROI 的高细节关键帧。
3. 高细节图尽量不是整图，而是与当前动作相关的局部区域。

如果运行上面的示例，逻辑上会得到类似结果：

| 步骤 | 上传内容 | 目的 |
|---|---|---|
| Step 1 | 全局 `low detail` | 先确认整体页面仍在预期状态 |
| Step 2 | 全局 `low detail` + 详情面板高细节 ROI | 页面发生主要内容变化，抓关键帧 |
| Step 3 | 全局 `low detail` + toast/按钮附近高细节 ROI | 只确认提交结果与按钮状态 |

如果把它翻译成更接近生产系统的伪代码，大致如下：

```python
capture = screenshot()

roi = detect_focus_region(capture, last_action)
crop = crop_image(capture, roi)
scaled = resize_short_edge(crop, 768)

send(low_detail(capture))  # 全局只做导航和状态感知

if pixel_change_ratio(crop, prev_crop) > 0.10:
    send(high_detail(scaled))  # 只在变化显著时发送关键帧
else:
    send_text("界面结构未发生关键变化，继续执行上一步计划")
```

真实工程里，`detect_focus_region` 往往来自三种信息源：

| 信息源 | 说明 | 优点 | 风险 |
|---|---|---|---|
| 上一步动作位置 | 如上次点击坐标附近 | 简单直接 | 页面跳转后可能失效 |
| DOM/可访问性树 | 从浏览器或应用层拿结构化节点 | 精度高 | 不一定可用 |
| 视觉检测 | 用轻量模型找按钮、输入框、弹窗 | 通用 | 额外增加推理开销 |

一个真实工程例子是：后台运营系统里批量审核内容。agent 连续执行“打开列表页 -> 进入详情 -> 查看状态 -> 点击通过/拒绝 -> 返回列表”。这类流程高度重复，界面变化局部而非全屏。最省的实现不是每步传整图，而是：

1. 列表页只传低细节全局图；
2. 点击某一行后，只上传详情面板 ROI；
3. 提交后仅在 toast、状态栏、按钮禁用状态附近做差分；
4. 每完成一单，保留一张关键帧文本摘要，丢弃中间无变化截图。

进一步看，生产系统通常还会把“截图策略”做成状态机，而不是单条 if-else：

| 系统状态 | 默认策略 |
|---|---|
| 导航中 | 全局 `low detail` |
| 操作中 | 当前交互区域 ROI |
| 等待异步结果 | 差分检测 + 低频采样 |
| 验证结果 | 高细节关键帧 |
| 归档完成 | 只保留文本摘要和必要关键帧 |

这样做的原因很简单：**截图优化本质上是调度问题，不只是图片压缩问题。**

---

## 工程权衡与常见坑

截图优化不是“压得越狠越好”，而是在信息完整性和成本之间找稳定点。

先看一个风险-收益表：

| 策略 | token 降幅 | 信息丢失风险 | 一致性影响 | 常见问题 |
|---|---:|---:|---:|---|
| 裁剪 ROI | 高 | 中 | 中 | 裁太小，漏掉上下文 |
| 降分辨率 | 中 | 中 | 低 | 小字、图标变模糊 |
| `low detail` | 极高 | 高 | 低 | 无法读细小文本 |
| 差分截图 | 极高 | 中 | 高 | 阈值不稳，漏掉关键变化 |
| 关键帧 | 高 | 低到中 | 低 | 关键时刻选择错误 |

几个最常见的坑，需要单独指出。

**第一，误以为 base64 压缩能显著省 token。**  
这通常是概念混淆。API 传输时图片可能被编码成 base64，但模型的视觉计费看的是图像解析后的表示，不是 base64 字符串有多短。也就是说，网络带宽和视觉 token 不是同一个优化目标。

**第二，每一帧都发高细节整图。**  
这是最昂贵也最没必要的默认配置。连续 24 张、每张 2,000 token，就是 48K token。很多 agent 不是推理能力不够，而是被自己的视觉输入挤爆了。

**第三，不裁剪无关区域。**  
浏览器标签栏、系统菜单栏、左右空白、固定导航栏，经常在几十轮任务里完全不变，却持续占用 tile。ROI 裁剪不是可选小技巧，而是第一优先级。

**第四，只做差分，不做关键帧。**  
差分截图可以理解为“只在变化时更新”，但如果一直没有完整关键帧，模型可能逐步丢失全局状态。稳妥做法通常是“低细节全局 + 局部高细节关键帧 + 差分触发”。

**第五，把 OCR 需求和导航需求混在一起。**  
导航只需要知道“按钮在哪、页面到了哪一步”；OCR 则需要读清楚精细文本。前者适合低细节和裁剪，后者则需要更高分辨率甚至专门的文档理解链路。两者不要共用同一张昂贵整图。

**第六，只看单张成本，不看整段任务成本。**  
有些团队会纠结“这一张从 255 token 优化到 170 token 值不值”，却忽略了任务本身可能要循环 100 次。单张的小差异，在长任务里会被放大成总账。

**第七，阈值只按像素变化，不按语义变化。**  
例如 loading spinner 一直在转，像素变化很大，但语义上没变化；而一个“提交成功”提示只变了一小块区域，像素变化很小，却是关键信号。因此差分阈值最好和 UI 语义一起使用。

把这类问题整理成“症状 -> 原因 -> 修正方式”，更容易排查：

| 症状 | 常见原因 | 修正方式 |
|---|---|---|
| token 很高但模型没变聪明 | 上传了大量无关区域 | 先做 ROI 裁剪 |
| 模型总说“看不清按钮文字” | 过度依赖 `low detail` | 对按钮局部升级高细节 |
| 任务后半段开始混乱 | 上下文被截图挤满 | 保留关键帧，删除中间冗余图 |
| 明明界面变了却没触发上传 | 差分阈值过高 | 降阈值，或补充语义规则 |
| 一直上传很多图 | 轮询频率太高 | 改为事件驱动或指数退避 |

工程上，一个常见经验法则是：

$$
\text{总 token} = \text{低细节保底} + \text{少量关键帧} + \text{必要时差分增量}
$$

而不是：

$$
\text{总 token} = \text{每一帧整图高细节}
$$

前一种更像“事件驱动”，后一种更像“无脑录像”。

还有两个很实用的阈值经验，可以给新手作为起点：

| 参数 | 推荐起点 | 说明 |
|---|---:|---|
| 差分触发阈值 | 5% 到 15% | 页面变化越局部，阈值应越低 |
| 全局低细节采样间隔 | 2 到 10 秒 | 与任务响应要求有关 |
| 强制关键帧间隔 | 5 到 20 步 | 防止模型长期失去全局状态 |

这些不是固定真理，但足够作为第一版系统的默认值。

---

## 替代方案与适用边界

并不是所有场景都值得自己实现复杂的截图压缩流水线。要看任务形态。

下面给一个适用矩阵：

| 场景 | 推荐策略 | 优先级 | 说明 |
|---|---|---|---|
| 实时桌面操作 | ROI + low detail + 差分 + 关键帧 | 高 | 最需要控 token |
| 后台表单自动化 | ROI + 关键帧 | 高 | 结构稳定、局部操作多 |
| 长时间监控看板 | low detail + 变化触发 | 高 | 大部分时间无变化 |
| 文档/报表定时归档 | 周期性 full detail | 中 | 非实时，重精度 |
| OCR 密集任务 | 高细节或专门文档模型 | 高 | 不能盲目降采样 |
| 离线审计 | 批量高细节 + 异步处理 | 中 | 不强调交互延迟 |

替代方案主要有三类。

**第一，用便宜视觉模型先做摘要。**  
也就是让一个低成本模型负责“看直播”，只输出结构化文本状态，如“页面已进入订单详情，主按钮可点击，状态为待审核”。主模型不直接看每一帧，而是只在关键时刻看关键帧。这相当于把视频流先压成事件流。

这类链路可以写成：

`截图流 -> 轻量视觉摘要 -> 状态机 -> 主模型只读摘要与关键帧`

优点是总成本低，缺点是摘要质量本身会影响主模型判断。

**第二，把完整截图交给专门链路。**  
比如 OCR 服务、文档理解模型、浏览器 DOM 提取器。这样主 agent 不再承担所有视觉解析任务，而是只消费提取后的结构化结果。

适合拆分的情况包括：

| 子任务 | 更适合的链路 |
|---|---|
| 读表格、票据、报表 | OCR / 文档模型 |
| 找按钮、输入框、弹窗 | UI 检测器 / accessibility tree |
| 判断页面路由和状态 | 轻量视觉模型或 DOM |
| 执行动作并回读结果 | 主 agent |

**第三，完全避免截图驱动。**  
如果环境允许，优先使用 DOM、Accessibility Tree、控件树、原生应用自动化接口。这些结构化信号通常比截图更便宜、更稳定。截图应当作为兜底视觉通道，而不是唯一数据源。

一个新手容易理解的对比例子是：  
如果你只是每小时收集一张报表截图做归档，那么直接发高细节整图即可，没必要为了省那几百 token 引入复杂差分逻辑。  
但如果你在做多轮远程协作，agent 每几秒就观察一次屏幕，那么差分和关键帧就不是“高级优化”，而是系统是否可持续运行的前提。

所以适用边界可以概括为：

1. **实时交互优先压 token。**
2. **离线精读优先保真度。**
3. **变化频繁的界面优先差分。**
4. **静态内容优先直接关键帧。**

还可以再补一条更工程化的判断标准：

5. **只要存在结构化信号，就不要默认依赖截图。**

因为截图最贵的地方，不是“它是一张图”，而是“模型必须从像素里重新推断一遍原本可以直接读取的结构”。

---

## 参考资料

- OpenAI API Docs: *Images and Vision*  
  https://platform.openai.com/docs/guides/images-vision

- OpenAI API Docs: *Pricing*  
  https://platform.openai.com/docs/pricing/

- Anthropic Docs: *Vision*  
  https://docs.anthropic.com/en/docs/build-with-claude/vision

- Google Cloud: *Vertex AI Generative AI Pricing*  
  https://cloud.google.com/vertex-ai/generative-ai/pricing

- Arun Baby: *Screenshot Understanding Agents*  
  https://www.arunbaby.com/ai-agents/0022-screenshot-understanding-agents/

- Emergent Mind: *Keyframe-oriented Vision Token Pruning*  
  https://www.emergentmind.com/topics/keyframe-oriented-vision-token-pruning
