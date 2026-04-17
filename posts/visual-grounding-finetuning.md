## 核心结论

视觉 Grounding 的目标，是把一句自然语言描述稳定地翻译成图像里的可执行位置。这里的“可执行”指输出不是普通文本解释，而是产品层能直接消费的定位结果，例如边界框 `bbox`、点击点 `(x, y)`，或分割掩码 `mask`。对于零基础读者，可以把它理解成一句话版目标检测：不是“图里有没有猫”，而是“把‘左边戴帽子的那只猫’准确找出来”。

在工程上，Referring Expression Comprehension，简称 REC，意思是“根据一句参照描述去找目标区域”，是最典型的 Grounding 任务。常见训练数据包括 RefCOCO、RefCOCO+、RefCOCOg 和 GRIT。它们共同提供了“文本描述 + 图像区域”的监督信号，但粒度不同：有的偏短描述和单目标框，有的覆盖更长表达，有的更适合大规模泛化训练。

当前主流输出接口可以分成三类：

| 输出类型 | 典型字段 | 适合场景 | 精度特点 |
|---|---|---|---|
| 边界框 bbox | `{"bbox":[x1,y1,x2,y2]}` 或 `xywh` | 检测、点击、拖拽、区域裁剪 | 中高，接口最通用 |
| 点击点 point | `{"point":[x,y]}` | UI 自动化、按钮点击、地图点选 | 高效，但不表达范围 |
| 分割掩码 mask | `{"mask_tokens":[t1,t2]}` 或解码后 mask | 精细选区、像素级交互、复杂形状对象 | 最高，但链路最复杂 |

结论先给清楚：

1. 如果下游动作只是“点哪里”“框哪里”，优先用 bbox 或点，训练简单、推理便宜、接口稳定。
2. 如果下游动作要求“选中不规则区域”，例如抠图、复杂选区、多区域交互，必须上 mask。
3. Qwen2.5-VL 一类模型已经把 Grounding 输出做成稳定 JSON 风格接口，适合直接接产品逻辑。
4. 基于 VLM-R1 的 GRPO 强化学习，本质是在强化“语言到坐标”的翻译能力，尤其对跨域泛化更有价值。
5. SAMTok 这类方案把 mask 压成离散 token，本质是在把像素级问题改写成语言模型能直接预测的“词”。

玩具例子先看一个最小版本。给模型一张网页截图和一句话“点击提交按钮”，模型输出：

```json
{"bbox":[0.41,0.82,0.55,0.88]}
```

这四个数是归一化坐标，意思是按钮位于整张图宽高的相对位置。产品层拿到它以后，可以直接算出点击中心点并执行自动化流程。

---

## 问题定义与边界

REC 的正式定义是：给定一张图和一段自然语言参照表达，预测该表达对应的图像区域。这里“参照表达”就是“用语言指某个东西”，例如“左下角的提交按钮”“第二排穿红衣服的人”“靠近门把手的标签”。

这个任务的边界要先说清楚，否则很容易把几类问题混在一起：

| 任务 | 输入 | 输出 | 评价指标 |
|---|---|---|---|
| REC | 图像 + 文本描述 | 单个或少量目标框/点/区域 | Accuracy、IoU 命中 |
| 开放词汇检测 | 图像 + 类别词 | 多个候选目标框 | mAP、Recall |
| 指代分割 | 图像 + 文本描述 | 像素级 mask | mIoU、Jaccard |
| OCR Grounding | 图像 + 文本字段 | 文本框位置 | 定位准确率 |

在 RefCOCO 系列里，常见评价方式是看预测框和真值框的 IoU，IoU 的白话解释是“两个框重叠得有多像”。如果预测框与真值框的交并比大于阈值，通常记作预测成功。数学上：

$$
IoU = \frac{|B_{pred} \cap B_{gt}|}{|B_{pred} \cup B_{gt}|}
$$

其中 $B_{pred}$ 是预测框，$B_{gt}$ 是标注框。

几个常见数据集可以这样理解：

| 数据集 | 特点 | 常见输出 | 常见用途 |
|---|---|---|---|
| RefCOCO | 经典 REC 基准，表达较自然 | bbox | 单目标参照定位 |
| RefCOCO+ | 更强调外观描述，弱化绝对位置词 | bbox | 检验视觉属性理解 |
| RefCOCOg | 表达更长、更完整 | bbox | 检验长文本 grounding |
| GRIT | 大规模 grounded image-text 对 | bbox/短语对齐 | 预训练与泛化提升 |

这里有一个很重要的边界：语言可能是模糊的。比如“左边那个按钮”在两列布局里可能有多个候选；“最大的蓝色框”可能依赖相对比较；“提交按钮旁边的取消按钮”还需要关系推理。也就是说，Grounding 不是纯视觉任务，而是“语言解析 + 视觉对齐 + 空间决策”的组合任务。

新手最容易忽略的第二个边界，是输出坐标通常要归一化到 `[0,1]`。白话解释：不用图片真实像素值，而用相对位置表示，这样同一套接口能处理不同分辨率图像。比如一句“左下角的提交按钮”，模型可以输出：

`x=0.453, y=0.872, w=0.122, h=0.053`

意思不是图片有这么多像素，而是目标框左上角大约在宽度的 45.3%、高度的 87.2% 处，宽占 12.2%，高占 5.3%。

---

## 核心机制与推导

Grounding 微调最核心的设计，不是“让模型看懂图”，而是“让模型把理解压缩成一种稳定输出协议”。这个协议可以是框、点，或 mask token。

最常见的是归一化 `xywh`。如果原图宽高分别是 $W$ 和 $H$，标注框左上角是 $(x_{min}, y_{min})$，右下角是 $(x_{max}, y_{max})$，那么：

$$
x = \frac{x_{min}}{W}, \qquad
y = \frac{y_{min}}{H}, \qquad
w = \frac{x_{max} - x_{min}}{W}, \qquad
h = \frac{y_{max} - y_{min}}{H}
$$

如果产品侧要的是 `x1,y1,x2,y2`，则直接输出：

$$
x_1 = \frac{x_{min}}{W}, \quad
y_1 = \frac{y_{min}}{H}, \quad
x_2 = \frac{x_{max}}{W}, \quad
y_2 = \frac{y_{max}}{H}
$$

这两种格式本质等价，只是消费接口不同。

一个玩具例子。假设截图宽 1000 像素，高 800 像素，按钮框是 `(400, 640, 560, 704)`。则：

- `x = 400 / 1000 = 0.4`
- `y = 640 / 800 = 0.8`
- `w = 160 / 1000 = 0.16`
- `h = 64 / 800 = 0.08`

也就是：

```json
{"xywh":[0.4,0.8,0.16,0.08]}
```

如果产品侧更喜欢点击点，那么中心点是：

$$
c_x = \frac{x_{min} + x_{max}}{2W}, \qquad
c_y = \frac{y_{min} + y_{max}}{2H}
$$

这个例子里中心点就是 `(0.48, 0.84)`。

从训练角度看，Grounding 微调至少包含三段逻辑：

```text
自然语言表达
    ↓
语言解析：目标词、属性词、关系词、位置词
    ↓
视觉对齐：在图像特征里找候选区域
    ↓
输出头或文本解码：bbox / point / mask token
    ↓
结构化结果：JSON、坐标串、token 序列
```

Qwen2.5-VL 的优势，在于它能把 Grounding 也纳入统一生成式接口里。也就是说，它不是单独挂一个检测头，而是能直接生成结构化文本，例如 bbox JSON 或点坐标。这让它更适合直接做视觉 Agent，尤其在“看图后执行动作”的链路里。

再往前一步，VLM-R1 用 GRPO 做强化学习。GRPO 可以理解成“用规则奖励去强化正确输出格式和正确空间位置”。REC 很适合做这种训练，因为真值答案通常明确：框对不对、点对不对、IoU 高不高，都能直接算奖励。对模型来说，这不是在学新的视觉编码器，而是在强化“如何把语言意图翻译成空间输出”。

真实工程例子更直观。给模型一张后台管理系统截图，指令是“点击左下角的提交按钮”。如果模型只会普通问答，它可能回答“提交按钮位于页面下方”。这对系统没有帮助。Grounding 微调后的模型则输出：

```json
{"bbox":[0.438,0.846,0.562,0.901],"label":"submit_button"}
```

这时产品系统可以继续做三件事：

1. 计算中心点并点击。
2. 在前端高亮该框，给用户做可视化确认。
3. 记录该框与指令的匹配日志，形成回放数据。

这就是“理解”变成“动作”的关键差别。

对于 mask 路线，SAMTok 的核心机制是把任意分割掩码压成两个离散 token。白话解释：原来 mask 是一大片二维像素，现在把它编码成两个像“词”一样的离散标记，语言模型只要学会预测这两个 token，就等价于学会输出区域。这样做的意义是，像素级任务不再需要另挂复杂分割头，而是能复用语言模型现有的 next-token 训练范式。

---

## 代码实现

下面先给一个最小可运行版本：把像素框转成归一化 JSON，并生成点击点。这个例子不依赖具体模型，目的是先把输出协议讲清楚。

```python
from dataclasses import dataclass

@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

def to_xywh_norm(box: BBox, width: int, height: int):
    assert width > 0 and height > 0
    assert 0 <= box.x1 < box.x2 <= width
    assert 0 <= box.y1 < box.y2 <= height

    x = box.x1 / width
    y = box.y1 / height
    w = (box.x2 - box.x1) / width
    h = (box.y2 - box.y1) / height
    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]

def to_click_point_norm(box: BBox, width: int, height: int):
    cx = (box.x1 + box.x2) / 2 / width
    cy = (box.y1 + box.y2) / 2 / height
    return [round(cx, 4), round(cy, 4)]

def to_json_payload(box: BBox, width: int, height: int):
    x1 = round(box.x1 / width, 4)
    y1 = round(box.y1 / height, 4)
    x2 = round(box.x2 / width, 4)
    y2 = round(box.y2 / height, 4)
    return {"bbox": [x1, y1, x2, y2]}

box = BBox(400, 640, 560, 704)
assert to_xywh_norm(box, 1000, 800) == [0.4, 0.8, 0.16, 0.08]
assert to_click_point_norm(box, 1000, 800) == [0.48, 0.84]
assert to_json_payload(box, 1000, 800) == {"bbox": [0.4, 0.8, 0.56, 0.88]}

print(to_json_payload(box, 1000, 800))
print({"point": to_click_point_norm(box, 1000, 800)})
```

如果接的是生成式多模态模型，推理层通常长这样：

```python
def grounding_infer(model, image, text):
    # 模型返回的可以是文本，也可以是结构化对象
    raw = model.predict(image=image, text=text)

    # 这里假设模型直接给出归一化框
    bbox = raw["bbox"]
    assert len(bbox) == 4
    assert all(0.0 <= v <= 1.0 for v in bbox)

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return {
        "bbox": [x1, y1, x2, y2],
        "click_point": [round(cx, 4), round(cy, 4)]
    }

# 伪返回值：适合“点击提交按钮”这种 UI grounding 任务
mock = {"bbox": [0.2, 0.3, 0.4, 0.5]}
```

如果是 mask token 路线，逻辑会变成：

1. 模型输出 `mask_tokens=[t1, t2]`
2. 送进 mask decoder 或 tokenizer decoder
3. 恢复为二值 mask
4. 如有需要，再从 mask 外接矩形得到 bbox

这说明 bbox、点、mask 其实不是互斥关系，而是不同精度层级的接口。工程上很常见的做法是：训练时保留更细粒度监督，在线服务时根据场景降级成点或框。

一个真实工程实现可以这样设计：

| 模块 | 输入 | 输出 | 说明 |
|---|---|---|---|
| 多模态模型 | 图像 + 指令文本 | bbox/point/mask token | 负责 Grounding 推理 |
| 输出解析器 | 模型原始字符串 | 标准 JSON | 负责校验格式 |
| 动作执行器 | JSON 坐标 | 点击/拖拽/高亮 | 负责产品动作 |
| 日志回放器 | 指令 + 输出 + 截图 | 样本记录 | 负责评估与复训 |

这种拆法的关键收益是：模型只负责“找”，产品系统负责“做”。接口一旦稳定，后面替换 Qwen2.5-VL、InternVL、或自定义微调版本都比较容易。

---

## 工程权衡与常见坑

Grounding 微调最常见的问题，不是模型完全不会找，而是“能找大概，但接口不稳”“框能出，但不好用”。

| 问题 | 原因 | 解决策略 |
|---|---|---|
| 输出格式不稳定 | 生成式模型容易多说解释文本 | 用严格模板、JSON schema、格式奖励 |
| 框位置飘 | 训练集分辨率、坐标协议不统一 | 统一归一化方式，训练前做坐标清洗 |
| 点能点到，框却很差 | 只学了中心偏置，没学边界 | 增加 bbox 监督或 IoU 奖励 |
| mask 很粗糙 | token 表达能力不够或数据噪声大 | 提高 mask token 质量，加入像素级验证 |
| 跨域泛化差 | 训练全是自然图，线上却是 UI 截图 | 混合训练 UI 数据或做领域续训 |
| 同义表达不稳 | 语言覆盖不足 | 扩充指令改写、属性词和关系词数据 |
| 模型会“编坐标” | 奖励只约束格式，没约束位置正确性 | 奖励函数加入 IoU/点击命中 |

这里有两个坑尤其值得单独说。

第一个坑是 mask 监督被低估。很多团队一开始只想做点击，就直接把任务压成点坐标。这样短期上线快，但一旦遇到复杂目标，比如不规则图标、多个靠很近的选区、需要拖拽边界的元素，模型就没有足够细粒度的空间概念。SAMTok 这类方案的价值就在这里：哪怕最终线上只用 bbox，mask 级训练也能提升边界感知。

第二个坑是数据阶段质量。InternVL 的经典三阶段训练是 contrastive、generative、supervised。白话解释：先对齐图文表示，再让模型学会生成式理解，最后用高质量任务数据做定向收口。这个流程的含义不是“阶段越多越高级”，而是每一段的数据噪声都会向下游传递。如果中间 generative 数据里位置描述混乱、框协议不统一、截图噪声大，最后做 Grounding 微调时就会表现成目标错位、关系词理解差、位置词失真。实际工程里，继续训练一个干净 checkpoint，往往比从脏数据全流程重训更划算。

再给一个新手容易踩的例子。训练集里“提交按钮”的标注框如果时大时小，有的只框文字，有的框整个按钮背景，那么模型上线后就会出现两种症状：

1. 点击中心偏到文字上，刚好还能点中，但拖拽失败。
2. 高亮框看起来总比真实按钮小一圈，用户会觉得模型“不准”。

这类问题不是模型架构先出错，而是标注协议没有统一。

---

## 替代方案与适用边界

如果把 Grounding 方案按输出能力分层，可以得到一个更实用的决策矩阵：

| 方案 | 输出类型 | 适用任务 | 响应粒度 |
|---|---|---|---|
| 纯 bbox 微调 | 框 | 目标定位、裁剪、区域高亮 | 区域级 |
| 点坐标微调 | 点 | UI 点击、地图点选、关键点动作 | 点级 |
| bbox + 点联合 | 框 + 点 | 同时要可视化和执行动作 | 区域级 + 点级 |
| SAMTok/指代分割 | mask token / mask | 精细选区、复杂轮廓、抠图 | 像素级 |
| 外接检测器 + 文本重排 | detector + rank | 工业稳定优先、解释性优先 | 依赖检测器 |

如果你的目标是“让 Agent 点击按钮、勾选复选框、拖拽滑块”，那么 bbox 或点通常已经足够。理由很简单：动作执行最终只需要一个落点，或者一个矩形区域。Qwen2.5-VL 这类模型的结构化输出很适合这条路线，接口轻、推理快、部署简单。

如果你的目标是“框出头发区域”“选择图中破损边缘”“把某个不规则图标完整选中”，那么 bbox 不够，点更不够，必须用 mask。SAMTok 的适用边界就在这里。它不是为了替代所有 bbox，而是为了把像素级区域也纳入语言模型统一训练范式。

还有一类替代方案是传统“两阶段”流水线：先目标检测，再用文本匹配排序。它的优点是调试容易、每一层职责清晰；缺点是开放表达能力弱，面对“穿红衣服且站在门左边的人”这种复杂关系描述，常常需要额外关系模块。对于零基础到初级工程师，判断标准很直接：

1. 只要动作，不追求边界细节：点或 bbox。
2. 要精细区域：mask。
3. 要最高可控性、最高可解释性：检测器加排序器。
4. 要统一成一个多模态 Agent：生成式 Grounding。

---

## 参考资料

- [Qwen2.5-VL 官方博客](https://qwenlm.github.io/blog/qwen2.5-vl/)：确认 Qwen2.5-VL 支持 bbox、point 和稳定 JSON 风格输出，用于说明产品接口形态。
- [VLM Run Visual Grounding 文档](https://docs.vlm.run/capabilities/visual-grounding)：给出归一化 `xywh` 的定义与接口解释，用于说明坐标协议。
- [FlagEval RefCOCO / RefCOCO+ / RefCOCOg 数据页](https://flageval.baai.ac.cn/docs/en/multimodal/Visual-Grounding/RefCOCO_RefCOCO%2B_RefCOCOg-Visual_Grounding.html)：用于定义 REC 任务和数据集边界。
- [VLM-R1 REC 模型卡](https://huggingface.co/omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps)：说明 Qwen2.5-VL 在 REC 上使用 GRPO/RL 与 RefCOCO 系列训练。
- [GRIT 数据集卡](https://huggingface.co/datasets/zzliang/GRIT)：说明 GRIT 是 grounded image-text 数据集，可用于 phrase grounding、REC 与预训练。
- [InternVL 官方站点](https://internvl.github.io/)：确认 InternVL 的三阶段训练范式，包括 contrastive、generative、supervised。
- [InternVL 1.0 博客](https://internvl.github.io/blog/2023-12-12-InternVL-1.0/)：用于引用三阶段训练的具体文字描述。
- [SAMTok 论文页](https://huggingface.co/papers/2601.16093)：说明“任意 mask 编成两个 token”的核心机制及其在 QwenVL 系列上的像素级扩展能力。
