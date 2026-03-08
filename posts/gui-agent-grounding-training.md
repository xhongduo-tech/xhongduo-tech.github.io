## 核心结论

GUI Grounding 是把自然语言指令映射到屏幕坐标或边界框的能力。更准确地说，输入是一张界面截图 $I$ 和一段操作指令 $T$，输出是目标控件的位置 $B$。对白话理解来说，它回答的问题就是：“这句话说的那个按钮，在图上到底是哪一块。”

它不是普通图文匹配任务的直接延伸。通用视觉语言模型通常能判断“这是聊天页面”“这是商品列表页”，但这类语义理解不足以支撑精确点击。GUI 场景的问题在于：字体小、图标小、布局重复、控件密集，而且错误不是“理解偏一点”，而是直接点错地方。对 grounding 来说，定位误差往往比分类误差更致命。

当前有效路线已经比较清晰，可以概括成两条主线：

| 路线 | 解决的问题 | 代表方法 | 核心收益 |
|---|---|---|---|
| GUI 专属对齐数据 | 教模型学会“指令 -> 控件”的监督关系 | SeeClick、UGround | 减少把自然语言映射到错误控件的情况 |
| 高分辨率视觉建模 | 让模型看清小字、细边框、小图标 | CogAgent | 提升小目标、密集界面的定位精度 |

两条路线不是替代关系，而是互补关系。只有把“看得清”与“对得准”同时解决，grounding 才能从“知道页面在说什么”推进到“知道该点哪里”。

一个最容易理解的玩具例子是：用户说“点击发送按钮”，输入是一张 $1120 \times 1120$ 的聊天窗口截图，模型输出按钮边界框 $(x_1, y_1, x_2, y_2)$，或者输出中心点 $(x, y)$。对初学者来说，可以把它理解成“模型先在截图里把按钮圈出来，再给出实际可点击的位置”。

下面这张总览表可以先把三类代表方法的差异看清楚：

| 方法 | 主要改进点 | 训练数据策略 | 代表精度特征 |
|---|---|---|---|
| CogAgent 18B | 双分辨率 cross-attention，看高分辨率 UI 细节 | GUI 专属图文与 REC/REG 数据，阶段式冻结再解冻 | 对文本按钮、小图标、小控件定位更稳定 |
| SeeClick | 多源 GUI 数据构建，强调 instruction-bbox 对齐 | RICO Widget Caption、RICOSCA、网页元素清洗数据 | 文本类点击明显优于早期通用模型 |
| UGround | 大规模跨平台 synthetic grounding | Web + Android 约千万级元素数据 | 平台迁移更强，平均准确率提升更明显 |

如果只看“能不能支撑 click/scroll planner”，答案已经接近“能”。但前提很明确：任务边界要清楚，截图质量要足够，数据清洗要严格，评测指标要针对真实点击误差，而不是只看模糊语义相似度。

---

## 问题定义与边界

GUI Grounding 的标准形式可以写成：

$$
f(I, T) \rightarrow B
$$

其中：

- $I$ 是界面截图。
- $T$ 是自然语言指令，例如“点击发送按钮”“打开右上角设置”。
- $B$ 是目标区域，通常表示为边界框或点击中心点。

若输出边界框，则常见形式为：

$$
B = (x_1, y_1, x_2, y_2), \quad x_2 > x_1,\; y_2 > y_1
$$

若输出点击点，则通常取：

$$
p = (x, y), \quad x = \frac{x_1+x_2}{2},\; y = \frac{y_1+y_2}{2}
$$

很多初学者会把 GUI Grounding 和 OCR、目标检测、网页自动化混在一起，但它们关注的问题不同：

| 任务 | 输入 | 输出 | 重点 |
|---|---|---|---|
| OCR | 图片 | 文本 | 把字识别出来 |
| 目标检测 | 图片 | 类别 + 框 | 找出属于某类的物体 |
| GUI Grounding | 截图 + 指令 | 控件框/坐标 | 把语言和界面元素精确对齐 |
| 自动化执行 | 指令 + 坐标 | click/scroll/type | 真正向系统发动作 |

它们之间的关系可以这样理解：

```text
OCR 负责“看清字”
目标检测负责“圈出候选区域”
GUI Grounding负责“根据指令选中正确控件”
自动化执行负责“把动作真正发出去”
```

边界主要体现在三个平台维度上，不同平台的困难点并不一样：

| 平台 | 典型元素 | 难点 | 例子 |
|---|---|---|---|
| Web | 链接、按钮、输入框、下拉菜单 | DOM 噪声多，重复文案多，隐藏元素多 | “点击顶部的登录链接” |
| Mobile | tab 图标、悬浮按钮、卡片、开关 | 图标小，控件密集，文字少 | “点右下角发布图标” |
| Desktop | 菜单栏、窗口控件、工具栏、侧边栏 | 分辨率更高，样式不统一，缩放比复杂 | “点击导出菜单” |

再看目标类型，也能更清楚地理解为什么有些任务更难：

| 目标类型 | 是否依赖 OCR | 是否依赖上下文 | 难点 |
|---|---|---|---|
| 文本按钮 | 高 | 中 | 文字必须看清，容易受分辨率影响 |
| 图标按钮 | 低 | 高 | 没有文字，只能靠形状和位置判断 |
| 重复按钮 | 中 | 高 | 语义相同，必须靠相对位置消歧 |
| 小型控件 | 中 | 中 | 很容易因为像素偏差造成误点 |

以常见 benchmark 的任务分布为例，web 场景中的“点击链接”往往依赖文本可读性；mobile 场景中的“点击图标”则经常没有文字，只能靠视觉形状、周围控件和位置关系联合判断。这也是为什么文本控件一般比纯图标控件更容易，web-only 方法也很难直接迁移到 mobile。

真实工程里，GUI Grounding 通常处在 agent 系统的中间层。以“在客服后台找到订单详情页，然后点击退款申请”为例，执行链路通常是：

```text
用户目标
-> planner 生成当前子任务：点击“退款申请”
-> grounding 模型把子任务映射为具体坐标
-> automation 层执行 click
-> 环境返回新截图
-> planner 再决定下一步
```

这里最脆弱的一环往往不是 planner，而是 grounding。因为 planner 只要知道“下一步该点退款申请”，而 grounding 必须在多个相似按钮中找出唯一正确目标。前者是语义规划问题，后者是空间对齐问题。

---

## 核心机制与推导

CogAgent 的关键不在于“模型更大”，而在于“看 GUI 的方式更适合 GUI”。它采用双分辨率 cross-attention，可以理解为先用低成本分支看全局布局，再用高分辨率分支补局部细节。

一个简化示意如下：

```text
指令文本: "点击发送按钮"
        |
        v
文本编码 --------------------------+
                                  |
低分辨率图像 224x224 -> 少量 tokens ---- cross-attn ---- 粗定位
                                  |
高分辨率图像 1120x1120 -> 大量 tokens --- cross-attn ---- 细定位
                                  |
                                  v
                         grounding head 输出 bbox / point
```

这个结构背后的直觉很直接：

- 低分辨率分支负责回答“目标大致在哪一片区域”。
- 高分辨率分支负责回答“那片区域里具体是哪一个像素级控件”。
- 文本 token 通过 cross-attention 去查询图像 token，本质上是在做“指令条件下的区域检索”。

如果只用低分辨率图像，小字和小图标会糊掉；如果一开始就对整张高分辨率图像做全局建模，计算成本又会显著上升。因此，双分辨率不是装饰，而是成本和精度之间的折中。

可以用一个更明确的代价近似式表示：

$$
\text{Cost Ratio} = \frac{L_{I\_hi}+L_T}{L_{I\_lo}+L_T}
$$

其中：

- $L_{I\_hi}$ 表示高分辨率图像 token 数
- $L_{I\_lo}$ 表示低分辨率图像 token 数
- $L_T$ 表示文本 token 数

若近似取：

$$
L_{I\_hi}=6400,\quad L_{I\_lo}=256,\quad L_T \ll L_{I\_hi}
$$

则有：

$$
\text{Cost Ratio} \approx \frac{6400}{256} = 25
$$

这意味着，单纯把整张 GUI 全部替换成高分辨率 token，代价可能是低分辨率方案的数十倍。对训练和推理来说，这个成本差异已经足以决定方案能否落地。

如果再把误差写成更接近工程视角的形式，可以得到一个直观结论。设真实框为 $B^\*$，预测框为 $\hat{B}$，则定位误差可以简写为：

$$
\mathcal{L}_{loc} = \|\hat{B} - B^\*\|_1
$$

当控件很小时，同样的像素偏差会占据更高比例。假设一个按钮宽度只有 20 像素，横向偏 5 像素就已经是 25% 的相对误差；但若按钮宽 200 像素，偏 5 像素只相当于 2.5%。这就是为什么 GUI grounding 对高分辨率输入更敏感，而自然图像任务未必如此。

训练策略也通常不是“一次性全开”，而是分阶段：

```text
阶段 1: 冻结原有主干，仅训练 high-res cross module
阶段 2: 逐步解冻相关视觉层，让高分辨率特征接入稳定
阶段 3: 联合训练，统一优化文本、低分辨率、高分辨率表征
```

这类训练方式的工程含义是：

| 阶段 | 训练目标 | 原因 |
|---|---|---|
| 阶段 1 | 先学会接入高分辨率信息 | 避免一开始全模型震荡过大 |
| 阶段 2 | 让视觉分支适应 GUI 细节 | 提升小字、小图标建模能力 |
| 阶段 3 | 联合对齐所有表征 | 让粗定位与细定位协同工作 |

SeeClick 和 UGround 的贡献则更偏向数据层。SeeClick 证明了一件事：GUI grounding 的瓶颈不只是模型结构，更是缺 instruction-bbox 对齐样本。UGround 再进一步，把 Web 和 Android 的大量元素构造成 synthetic grounding 数据，解决了平台覆盖不足、标注成本过高和真实数据稀缺的问题。

因此，这条路线的完整逻辑可以写成：

1. GUI grounding 不是纯语义理解，而是像素级或框级对齐问题。
2. 小控件、小图标、细文本要求高分辨率视觉输入。
3. 语言到控件的映射关系需要 GUI 专属监督，而不是普通图文对。
4. 跨平台泛化要求数据覆盖 Web、Mobile 乃至 Desktop 的不同布局模式。
5. 因此，最有效的方案通常是“高分辨率架构 + GUI 专属数据”。

对初学者来说，可以把上面五步压缩成一句话：模型既要看清楚，也要学过这种任务。

---

## 代码实现

如果把训练样本抽象成统一格式，本质上就是生成下面这种结构：

```json
{
  "image": "screenshot.png",
  "instruction": "点击发送按钮",
  "bbox": [860, 980, 940, 1030]
}
```

这里的 `bbox` 表示目标控件在截图中的矩形区域。很多真实系统最后并不是直接点击整个框，而是先把框转换成一个可执行的中心点。

先看一个可直接运行的最小例子。下面的代码定义了边界框、中心点计算、合法性检查和越界检查：

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def validate(self) -> None:
        if not (self.x2 > self.x1 and self.y2 > self.y1):
            raise ValueError(f"invalid bbox: {self}")

    def center(self) -> tuple[int, int]:
        self.validate()
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def contains(self, x: int, y: int) -> bool:
        self.validate()
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def clip_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    x = min(max(x, 0), width - 1)
    y = min(max(y, 0), height - 1)
    return x, y


if __name__ == "__main__":
    image_width, image_height = 1120, 1120
    send_button = BBox(860, 980, 940, 1030)

    cx, cy = send_button.center()
    cx, cy = clip_point(cx, cy, image_width, image_height)

    assert (cx, cy) == (900, 1005)
    assert send_button.contains(cx, cy)
    print({"click_point": (cx, cy)})
```

这段代码虽然简单，但已经覆盖了真实系统里的三个基本要求：

| 步骤 | 作用 | 为什么需要 |
|---|---|---|
| `validate()` | 检查框是否合法 | 避免训练或推理阶段出现坏标注 |
| `center()` | 计算点击点 | 很多执行器最终只接受点坐标 |
| `clip_point()` | 防止坐标越界 | 截图缩放、标注误差都可能导致越界 |

如果要做网页数据构建，常见流程是“读取页面元素 -> 清洗噪声属性 -> 生成 instruction + bbox -> 过滤坏样本”。下面给出一个可以直接运行的简化版本，而不是只写伪代码：

```python
from dataclasses import dataclass


BAD_TOKENS = ("javascript:", "http://", "https://", "undefined", "null")


@dataclass
class Element:
    label: str
    bbox: list[int]
    visible: bool = True
    clickable: bool = True


def clean_dom_text(text: str) -> str:
    text = text.strip()
    for token in BAD_TOKENS:
        text = text.replace(token, "")
    return " ".join(text.split())


def valid_bbox(bbox: list[int], image_w: int, image_h: int) -> bool:
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    if not (x2 > x1 and y2 > y1):
        return False
    if x1 < 0 or y1 < 0 or x2 > image_w or y2 > image_h:
        return False
    return True


def build_sample(element: Element, image_w: int, image_h: int) -> dict | None:
    if not element.visible or not element.clickable:
        return None

    label = clean_dom_text(element.label)
    if not label:
        return None

    if not valid_bbox(element.bbox, image_w, image_h):
        return None

    instruction = f"点击{label}"
    return {
        "instruction": instruction,
        "bbox": element.bbox,
    }


if __name__ == "__main__":
    image_w, image_h = 1120, 1120
    element = Element(label="  发送  ", bbox=[860, 980, 940, 1030])

    sample = build_sample(element, image_w, image_h)
    assert sample is not None
    assert sample["instruction"] == "点击发送"
    assert sample["bbox"] == [860, 980, 940, 1030]
    print(sample)
```

上面这段代码已经体现了 GUI 数据构建中的几个关键事实：

- 样本不是“抓到一个元素就能用”，而是要先过滤不可见、不可点击、空文本、坏框。
- `instruction` 的质量直接影响模型学到的语言对齐关系。
- `bbox` 的质量直接影响模型学到的空间监督关系。

把它再往前扩一层，就能得到更接近真实训练集构建的流程：

```text
页面截图 + DOM/OCR 元素
-> 过滤隐藏元素
-> 清洗 label
-> 过滤无效 bbox
-> 判断是否可点击
-> 生成 instruction-bbox
-> 写入训练样本
```

如果对应到 CogAgent 风格的结构，前向过程可以抽象成下面的伪代码：

```python
def forward(image_low, image_high, instruction):
    low_tokens = encode_image(image_low)
    high_tokens = encode_image(image_high)
    text_tokens = encode_text(instruction)

    coarse_features = cross_attention(text_tokens, low_tokens)
    fine_features = cross_attention(coarse_features, high_tokens)

    pred_bbox = grounding_head(fine_features)
    return pred_bbox
```

如果进一步写出训练损失，最常见的简化形式是：

```python
def training_step(model, batch):
    pred_bbox = model(batch["image_low"], batch["image_high"], batch["instruction"])
    target_bbox = batch["bbox"]

    loss_l1 = l1_loss(pred_bbox, target_bbox)
    loss_iou = iou_loss(pred_bbox, target_bbox)
    loss = loss_l1 + loss_iou
    return loss
```

把这个流程对应到模块配置，可以压缩成一张更清楚的表：

| 模块 | 作用 | 典型配置 |
|---|---|---|
| 低分辨率视觉分支 | 保留页面全局布局 | 224x224，少量视觉 tokens |
| 高分辨率视觉分支 | 捕捉细粒度控件细节 | 1120x1120，大量视觉 tokens |
| Cross-attention | 让文本条件作用于图像区域 | 初期可单独训练 |
| Grounding head | 输出框或点击点 | MLP / box regression head |
| 数据清洗模块 | 去噪、过滤坏框、构造指令 | DOM/OCR 清洗规则 |

真实工程里，SeeClick/UGround 风格的数据管线往往比模型本身更费时间。原因不神秘：网络结构大多可以复用，但 GUI 数据天然脏，而且脏的方式很多。

| 数据问题 | 表现 | 对训练的破坏 |
|---|---|---|
| 文本脏 | URL、脚本、空字符串混入 label | 指令语义污染 |
| 坏框 | 越界框、零面积框、重叠错误框 | 监督信号失真 |
| 目标歧义 | 同名按钮多个 | 模型无法学到唯一对齐关系 |
| 视图不一致 | DOM 来自一个状态，截图来自另一个状态 | 指令和图像不对应 |

对初学者来说，最容易忽视的一点是：grounding 训练集不是“图 + 字 + 框”三个字段齐了就可以，而是三者必须严格同一时刻、同一页面状态、同一控件定义。

---

## 工程权衡与常见坑

最常见误区是“拿一个通用多模态模型微调一下就够了”。通常不够。原因不在于通用模型完全没用，而在于 GUI 不是普通自然图像。很多目标没有明确视觉类别，只有交互意义。例如一个齿轮图标表示“设置”，一个纸飞机图标表示“发送”，它们的语义并不来自物体类别，而来自界面约定和上下文位置。

下面这张表更适合工程决策：

| 方案 | 数据量 | 清洗成本 | 对小图标效果 | 跨平台泛化 |
|---|---|---|---|---|
| 通用图文数据微调 | 低到中 | 低 | 弱 | 弱 |
| SeeClick 式多源 GUI 数据 | 中 | 中到高 | 中 | 中 |
| UGround 式大规模 synthetic GUI | 高 | 高 | 强 | 强 |
| 高分辨率模型 + GUI 数据 | 高 | 高 | 最强 | 强 |

很多失败案例不是“模型太差”，而是数据前处理出了问题。最典型的是 DOM 未清洗：

| 情况 | instruction 生成结果 | 后果 |
|---|---|---|
| 未清洗 | 点击 javascript:void(0) send_btn primary | 语言噪声大，监督关系变差 |
| 部分清洗 | 点击 send_btn | 还能训练，但语义不自然 |
| 清洗后 | 点击发送按钮 | 语义直接，框更稳定 |

可以把网页清洗流程理解成下面这条线：

```text
原始 DOM / OCR / bbox
-> 去隐藏元素
-> 去空文本与脚本垃圾
-> 过滤无效 URL
-> 合并重复标签
-> 检查 bbox 是否越界
-> 处理遮挡与重叠
-> 生成 instruction-bbox 样本
```

还有两个常见坑值得单独展开。

第一类坑是“文本框强、图标框弱”。原因不是模型偏心，而是任务信息量不同。文本按钮自带可读语义，图标按钮通常只有形状线索。前者更像“先 OCR，再对齐”；后者更像“在上下文里做视觉消歧”。

| 目标 | 主要依赖 | 常见失误 |
|---|---|---|
| 文本按钮 | OCR + 文本对齐 | 文字看不清、重复文案混淆 |
| 图标按钮 | 形状 + 相对位置 | 图标相似、上下文理解不足 |

第二类坑是“大框准确率高、小框准确率低”。这是一个几乎必然出现的现象。因为点击误差通常按像素统计，而小框对像素误差更敏感。若框宽高分别为 $w, h$，中心点偏移为 $\Delta x, \Delta y$，则相对误差可以简写为：

$$
\epsilon = \max\left(\frac{|\Delta x|}{w}, \frac{|\Delta y|}{h}\right)
$$

当 $w, h$ 很小时，即使 $\Delta x, \Delta y$ 不大，$\epsilon$ 也会迅速变大。这就是为什么高分辨率分支对小控件价值更高，而对大按钮的收益相对有限。

如果你在训练中看到“文字按钮很好，齿轮图标很差”，先不要急着调优化器，优先检查这三件事：

1. 输入分辨率是否足够支撑小目标识别。
2. 图标样本比例是否过低，导致模型几乎没学到图标分布。
3. 标注框是否过松、过紧或与截图状态不一致。

再补一个工程上很常见但论文里不一定展开的坑：截图缩放比例。很多自动化链路里，标注框基于原始分辨率生成，但模型输入经过缩放，执行器又在另一套坐标系点击。只要坐标变换链路没对齐，再好的模型也会看起来像“定位漂移”。

| 坐标系 | 来源 | 常见问题 |
|---|---|---|
| 原始截图坐标 | 标注工具或采集器 | 与模型输入尺寸不一致 |
| 模型输入坐标 | resize 后图像 | 预测框需要反变换 |
| 执行器坐标 | 浏览器窗口或设备屏幕 | 还可能叠加滚动偏移、状态栏偏移 |

所以很多“训练问题”，最后排查出来其实是“坐标系问题”。

---

## 替代方案与适用边界

不是所有场景都要上 1120 分辨率和大规模 synthetic 数据。Grounding 系统的设计要看业务代价，而不是只看论文最优结果。

如果资源有限，可以先用低分辨率 grounding 模型配合 planner，只处理大控件和文本控件。这个方案适合命令式自动化，例如“点击提交”“打开设置”“搜索订单”，因为这类目标通常尺寸较大、文字较明显、布局相对稳定。

如果是 web-only 场景，SeeClick 加网页元素清洗，往往已经能覆盖大部分点击需求。原因是 Web 页面更容易拿到 DOM 结构，文本密度也更高，天然比纯视觉 mobile 场景更适合构建监督数据。

如果要做 web + mobile + desktop，跨平台数据基本是硬要求。否则模型很容易学到强平台偏置，例如把“右上角齿轮”过拟合为 Web 设置入口，却无法迁移到移动端的底部 tab 图标或桌面应用的工具栏按钮。

可以用一个简单决策流表示：

```text
是否只做 Web?
-> 是: 优先网页数据清洗 + 中等分辨率 grounding
-> 否: 继续看

是否经常点击小图标/细文本?
-> 是: 需要高分辨率分支
-> 否: 低分辨率方案可先落地

是否要求跨平台?
-> 是: 需要 synthetic + real GUI 混合数据
-> 否: 单平台专项数据即可
```

把不同方案放到同一张表里，会更容易判断：

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| 低分辨率 grounding | 企业后台、大按钮、文本控件为主 | 便宜、部署简单 | 小图标和密集布局效果差 |
| Web-only GUI 数据训练 | 浏览器自动化、表单操作 | 数据好构建，文本目标强 | 移动端迁移弱 |
| Synthetic Web + Android | 跨平台 agent | 覆盖广，泛化更强 | 数据管线和训练成本高 |
| 高分辨率 + GUI 专属数据 | 高精度点击、复杂 UI | 小目标效果最好 | 成本最高 |

一个更具体的对比是：

- `256 token 低分辨率方案`：适合页面按钮较大、布局稳定、误点代价不高的企业后台。
- `中等分辨率 + Web 清洗数据`：适合只做浏览器操作、希望尽快落地的系统。
- `UGround 风格 synthetic Web + Android`：适合要覆盖网页和移动端的 agent 系统。
- `高分辨率架构 + 混合 GUI 数据`：适合误点代价高、图标多、控件小、平台杂的复杂场景。

所以替代方案不是“哪个更先进”，而是“你的误点代价有多大、控件有多小、平台有多杂、可接受成本有多高”。

---

## 参考资料

下面这些资料最值得优先看，因为它们分别对应架构、数据和评测三个层面：

| Source | 内容 | 可复用资源 |
|---|---|---|
| CogAgent 论文 | 双分辨率 cross-attention、训练阶段、GUI 数据组成 | 架构图、训练策略、消融结论 |
| SeeClick 论文/项目页 | RICO、RICOSCA、网页数据构建方式 | 数据清洗思路、样本组织方法 |
| UGround 论文/项目页 | 跨平台 synthetic grounding、Web 和 Android 数据规模 | 跨平台数据设计、系统整合思路 |
| ScreenSpot benchmark / 相关论文 | web/mobile/desktop、text/icon 任务上的对比结果 | 评测维度、误差分析框架 |

如果你要复现实验，阅读顺序建议如下：

1. 先看 benchmark 和任务定义，确认评测到底是点级准确率、框级准确率，还是更宽松的匹配标准。
2. 再看数据说明，理解样本如何构造、过滤了哪些脏数据、平台是否平衡。
3. 最后再看模型结构和训练配置，因为 GUI grounding 的问题里，数据质量往往先于模型复杂度决定上限。

对初学者来说，一个很实用的判断标准是：如果一篇资料没有讲清楚“样本从哪里来、bbox 怎么校验、文本怎么清洗、不同平台怎么平衡”，那它对复现的帮助通常有限。因为 grounding 的难点并不只是模型怎么写，更是监督信号怎么变得可信。

如果只选三份资料优先读，可以这样分工：

| 优先级 | 资料 | 建议关注点 |
|---|---|---|
| 1 | SeeClick | GUI 数据从哪里来，instruction-bbox 怎么构造 |
| 2 | CogAgent | 高分辨率分支为什么有效，训练阶段怎么设计 |
| 3 | UGround | 为什么 synthetic 数据能改善跨平台泛化 |
