## 核心结论

Visual Grounding 的任务，不是判断一句话“是否和图像相关”，而是把这句话落到图像中的一个可验证区域上。模型必须给出空间证据，说明“你描述的是图里的哪一块”。这一步把多模态理解从抽象语义匹配，推进到可检查、可评估、可用于交互的区域级理解。

在这个任务里，Referring Expression Comprehension，简称 REC，表示“根据描述找到目标框”；Referring Expression Segmentation，简称 RES，表示“根据描述找到目标像素”。REC 回答的是“目标大致在哪里”，RES 回答的是“目标具体长什么样、边界到哪里结束”。两者不是二选一关系，而是两个粒度不同的输出层：REC 更适合检索、粗定位、候选筛选；RES 更适合抓取、编辑、避障、精细交互。

一个直观例子是：“请标记桌面上穿白衬衫的人手里的红杯子。”  
如果只做 REC，系统可能返回一个覆盖“杯子加手部”的矩形框；如果继续做 RES，系统才会把杯子本身的像素轮廓分离出来。前者能证明模型找到了正确位置，后者能证明模型理解了目标的真实几何边界。

评价时最常见指标是 IoU，Intersection over Union，表示预测区域与真实区域的重叠程度：

$$
IoU = \frac{|P \cap G|}{|P \cup G|}
$$

其中 $P$ 是预测区域，$G$ 是真实标注区域。  
在 REC 中，常用 “$IoU \ge 0.5$ 视为命中” 这样的阈值判断是否定位正确；在 RES 中，更常看 mask IoU、mIoU 或 Dice，因为分割任务关心的是每个像素是否对齐，而不是只看一个粗框是否覆盖住目标。

如果把这件事压缩成一句话：Visual Grounding 解决的是“语言如何精确约束图像中的空间实体”，而 REC 与 RES 分别对应“位置正确”与“形状正确”两个层次的检验。

---

## 问题定义与边界

形式化地看，输入是图像 $I$ 和文本表达 $q$，目标是学习映射：

$$
f(I, q) \rightarrow y
$$

其中输出 $y$ 可以是：

- 一个边界框 $b = (x_1, y_1, x_2, y_2)$
- 一张二值掩码 $M \in \{0,1\}^{H \times W}$
- 在复杂设置下，也可以是多个区域组成的集合 $\{r_1, r_2, \dots, r_n\}$

这一定义听起来简单，但边界必须先划清，否则很容易把它和其他视觉语言任务混淆。

第一，Visual Grounding 不是普通图像分类。  
图像分类回答的是“这张图里有没有猫”；Grounding 回答的是“你说的那只猫具体在哪里”。

第二，它也不是图像描述生成。  
描述任务是从图像生成语言；Grounding 是反过来让语言约束图像中的区域选择。前者输出一句话，后者输出一个区域。

第三，REC 和 RES 的监督对象不同。  
REC 学习几何定位，通常输出框坐标；RES 学习像素归属，通常输出逐像素 mask。二者可共享编码器，但损失函数、评价方式、工程用途都不同。

下面用一个玩具例子说明。  
假设图中有一个蓝盒子、一个红球、两个小人，表达是：“红色球在蓝色盒子上面的那个小人。”

这个表达包含三类信息：

| 约束类型 | 例子 | 模型需要做什么 |
| --- | --- | --- |
| 属性约束 | “红色”“蓝色” | 识别颜色、外观等局部视觉属性 |
| 空间约束 | “在……上面” | 建立实体间相对位置关系 |
| 指代约束 | “那个小人” | 在多个候选对象中选出唯一目标 |

REC 的目标，是找出“那个小人”的矩形区域。  
RES 的目标，是进一步把该小人的像素精确分离出来，排除背景、盒子边缘、球体遮挡等无关部分。

因此，REC 和 RES 的差异可以直接对照：

| 任务 | 输出形式 | 主要回答的问题 | 常用指标 | 典型场景 |
| --- | --- | --- | --- | --- |
| REC | Bounding box | 目标大致在哪 | IoU、Acc@0.5、AP | 检索、粗定位、候选过滤 |
| RES | Segmentation mask | 目标轮廓在哪里结束 | mask IoU、mIoU、Dice | 抓取、编辑、抠图、避障 |

对新手来说，最容易记的一句话是：

- REC 关注“找对位置”
- RES 关注“找对形状”

真实系统里，二者经常串联使用：先用 REC 缩小搜索范围，再用 RES 做细化。

---

## 核心机制与推导

现代区域级视觉理解通常不是把“图像特征”和“文本特征”机械拼接，而是把视觉 token 与文本 token 放进统一的对齐框架中，通过跨模态注意力建立对应关系。

这里先把几个术语压缩成最小定义：

| 术语 | 最小定义 | 在 Grounding 中的作用 |
| --- | --- | --- |
| Token | 模型处理的最小离散单元 | 文本 token 表示词片段，视觉 token 表示图像块或区域特征 |
| Encoder | 把输入变成特征表示的模块 | 图像编码成区域向量，文本编码成语义向量 |
| Attention | 按相关性分配权重的机制 | 决定某个词该关注哪些图像位置 |
| Cross-attention | 一种跨模态 attention | 让文本查询图像，或让图像查询文本 |

当模型读到“红杯子”时，它会提升与红色区域、杯状形状相关的视觉位置权重；  
当模型继续读到“穿白衬衫的人手中的”时，它还要把“人”“衣着属性”“手持关系”“杯子”串成一条关系链。  
因此，Grounding 本质上不是“名词匹配”，而是“复合约束求交”。

可以把这一过程抽象成三步：

$$
V = E_{img}(I), \quad T = E_{txt}(q)
$$

其中 $V$ 是图像特征，$T$ 是文本特征。然后通过跨模态融合：

$$
F = \mathrm{Fuse}(V, T)
$$

最后由任务头输出框或掩码：

$$
\hat{b}, \hat{M} = H(F)
$$

如果是联合建模 REC 和 RES，总损失通常写成：

$$
L_{total} = \lambda_{box} L_{box} + \lambda_{cls} L_{cls} + \lambda_{mask} L_{mask}
$$

更具体一点，常见展开形式是：

$$
L_{box} = L_{L1} + \alpha L_{IoU}
$$

$$
L_{mask} = \beta L_{BCE} + \gamma L_{Dice}
$$

于是总损失可以写成：

$$
L_{total} = \lambda_1 (L_{L1} + \alpha L_{IoU}) + \lambda_2 L_{CE} + \lambda_3 (\beta L_{BCE} + \gamma L_{Dice})
$$

其中：

- $L_{L1}$ 约束预测框和真实框的坐标偏差
- $L_{IoU}$ 直接优化框的重叠质量
- $L_{CE}$ 或其他分类损失，约束类别或离散 token 预测
- $L_{BCE}$ 约束每个像素是否属于目标
- $L_{Dice}$ 强调形状重叠，尤其对小目标和细长边界更敏感

Dice 系数定义为：

$$
Dice(P, G) = \frac{2|P \cap G|}{|P| + |G|}
$$

对应的 Dice Loss 通常写作：

$$
L_{Dice} = 1 - Dice(P, G)
$$

它和 IoU 类似，但对小目标更稳定。原因很简单：小目标的像素数本来就少，稍微错一点，IoU 可能大幅波动；Dice 在这类情形下通常更平滑。

下面用一个数值例子把指标算清楚。

假设预测框 $A=(10,10,110,110)$，真实框 $B=(15,15,95,95)$。

- 预测框面积：$100 \times 100 = 10000$
- 真实框面积：$80 \times 80 = 6400$
- 交集框恰好就是真实框，面积为 $6400$
- 并集面积：$10000 + 6400 - 6400 = 10000$

所以：

$$
IoU = \frac{6400}{10000} = 0.64
$$

这说明 REC 在常见的 $0.5$ 阈值下可判为正确。

再看 mask。  
若预测 mask 与真实 mask 的交集像素为 $3$，并集像素为 $4$，则：

$$
IoU_{mask} = \frac{3}{4} = 0.75
$$

Dice 为：

$$
Dice = \frac{2 \times 3}{3 + 4} = \frac{6}{7} \approx 0.857
$$

这两个数一起看，比只看框更可靠。因为一个框完全可能把目标“包住了”，但内部混入大量背景，尤其是在细长、弯曲、被遮挡或密集相邻目标场景中。

真实工程里，这一点尤其明显：

| 场景 | 只做 REC 的问题 | 为什么还要 RES |
| --- | --- | --- |
| 遥感图像 | 相邻建筑可能落在相似矩形里 | 需要分清真实边界、屋顶形状、遮挡关系 |
| 桌面抓取 | 框可能同时覆盖目标和手部/邻近物体 | 机械臂需要知道可抓取边界 |
| 医学图像 | 粗框能定位病灶大概区域 | 诊断和测量常需要精确轮廓 |
| 图像编辑 | 框只能圈出对象附近区域 | 抠图、替换、局部重绘需要像素级 mask |

所以，核心机制可以概括成一句话：模型必须把语言中的属性、关系、指代链，压缩进一个区域选择过程，而 REC 与 RES 分别是在粗粒度和细粒度上验证这个过程是否真的执行正确。

---

## 代码实现

下面给出一个最小可运行示例，用纯 Python 演示 box IoU、mask IoU 和 Dice 的计算。示例只依赖标准库，可以直接运行。

```python
from typing import List, Sequence, Tuple

Box = Tuple[int, int, int, int]
Mask = List[List[int]]


def box_area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = box_area(a) + box_area(b) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def validate_same_shape(pred: Sequence[Sequence[int]], gt: Sequence[Sequence[int]]) -> None:
    if len(pred) != len(gt):
        raise ValueError("pred and gt must have the same number of rows")
    for row_p, row_g in zip(pred, gt):
        if len(row_p) != len(row_g):
            raise ValueError("pred and gt must have the same shape")


def mask_iou(pred: Mask, gt: Mask) -> float:
    validate_same_shape(pred, gt)

    inter = 0
    union = 0
    for row_p, row_g in zip(pred, gt):
        for p, g in zip(row_p, row_g):
            p = 1 if p else 0
            g = 1 if g else 0
            if p == 1 and g == 1:
                inter += 1
            if p == 1 or g == 1:
                union += 1

    return inter / union if union > 0 else 0.0


def dice_score(pred: Mask, gt: Mask) -> float:
    validate_same_shape(pred, gt)

    inter = 0
    pred_sum = 0
    gt_sum = 0
    for row_p, row_g in zip(pred, gt):
        for p, g in zip(row_p, row_g):
            p = 1 if p else 0
            g = 1 if g else 0
            if p == 1 and g == 1:
                inter += 1
            pred_sum += p
            gt_sum += g

    denom = pred_sum + gt_sum
    return (2 * inter) / denom if denom > 0 else 1.0


if __name__ == "__main__":
    pred_box = (10, 10, 110, 110)
    gt_box = (15, 15, 95, 95)

    pred_mask = [
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
    ]
    gt_mask = [
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]

    iou_box = box_iou(pred_box, gt_box)
    iou_mask = mask_iou(pred_mask, gt_mask)
    dice = dice_score(pred_mask, gt_mask)

    assert round(iou_box, 2) == 0.64
    assert round(iou_mask, 2) == 0.75
    assert round(dice, 3) == 0.857

    print(f"box IoU  = {iou_box:.2f}")
    print(f"mask IoU = {iou_mask:.2f}")
    print(f"Dice     = {dice:.3f}")
```

这段代码要点很少，但对新手很关键：

| 函数 | 作用 | 常见错误 |
| --- | --- | --- |
| `box_iou` | 计算两个框的重叠程度 | 把坐标写成中心点格式却按角点格式计算 |
| `mask_iou` | 计算两个 mask 的并交比 | 两个 mask 尺寸不一致时仍强行 zip |
| `dice_score` | 衡量形状重叠 | 把多类别 mask 当成二值 mask 直接计算 |

如果运行这段程序，输出应为：

```text
box IoU  = 0.64
mask IoU = 0.75
Dice     = 0.857
```

这说明：

- 框层面已经达到常见的 REC 命中阈值
- mask 层面与真实轮廓也较接近
- Dice 比 IoU 更高，符合其对重叠更“宽容”的特性

如果把它放回训练流程，一个不依赖具体框架的最小伪代码如下：

```python
for image, expr, gt_box, gt_mask in data_loader:
    img_feats = vision_encoder(image)
    txt_feats = text_encoder(expr)

    fused_feats = cross_attention(img_feats, txt_feats)

    pred_box = box_head(fused_feats)
    pred_mask = mask_head(fused_feats)

    loss_box = l1_loss(pred_box, gt_box) + iou_loss(pred_box, gt_box)
    loss_mask = bce_loss(pred_mask, gt_mask) + dice_loss(pred_mask, gt_mask)
    loss = loss_box + loss_mask

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

这里有三个实现点值得特别强调。

第一，共享特征底座。  
也就是框和 mask 不完全分家，而是从同一套跨模态融合特征出发。这样“这句话说的是谁”和“这个目标的边界在哪里”能够互相约束。

第二，cross-attention 是关键，不是装饰。  
它的作用不是把特征拼在一起，而是让文本动态决定图像中哪些区域更值得看。表达越依赖关系词和限定词，这一步越重要。例如“左边第二个杯子”和“靠近键盘的那个杯子”，没有跨模态对齐，模型很容易退化成“看到杯子就行”。

第三，训练和评估不要只看一个数字。  
只看框 IoU，可能把“框对了、轮廓错得很厉害”的情况掩盖掉；只看 mask IoU，又可能忽略模型是否先找到了正确候选区域。工程上通常会同时记录 box IoU、mask IoU、Dice，必要时再分短语类型做误差分析，例如属性错误、空间关系错误、否定词错误。

再看一个面向新手的具体例子。  
表达是：“左边穿白衣的人手中的红杯子。”

一个合理的推理链应当是：

1. 先在图中找出所有“人”
2. 用“左边”“穿白衣”筛掉错误候选
3. 在正确的人附近找“手中的物体”
4. 在这些物体中再筛选“红杯子”
5. 输出框或 mask

如果模型直接从整句话跳到“所有红色物体”，那它并没有真正执行语言约束，只是在做模糊相似度匹配。这正是很多 Grounding 系统在复杂表达下失败的根源。

---

## 工程权衡与常见坑

真实系统里，Grounding 难的不是“能不能输出一个框”，而是“语言里的每个约束是否真的被执行了”。模型常见失败并不是视觉看不见，而是语言没被严格落实。

最常见的坑，是空间关系和否定表达。

例如：“not the cat next to the lamp.”  
如果模型主要依赖视觉先验，它可能一看到猫就给出其中一只，而忽略 `not`。这不是目标检测失败，而是语言执行失败。对于 Grounding，这类错误比普通分类错误更严重，因为它说明系统没有按表达进行约束推理。

第二个坑，是链式关系太长。  
例如：“站在窗边、穿白衬衫、手里拿红杯子的人左边的黑包。”  
这里至少有三层条件：

- 锁定人
- 用属性和位置确定是哪一个人
- 再根据该人去找左边的黑包

如果模型只抓住“黑包”两个字，最终结果看起来可能“像是相关物体”，但实际上不满足指代链。

第三个坑，是训练提示本身带偏模型。  
有些方法直接在图像上叠加高亮框，希望模型更关注候选区域。但如果设计不当，模型可能学到“被高亮的区域更像答案”这种伪规律，而不是学会真正根据语言判断区域是否正确。

第四个坑，是小目标、细长目标和遮挡目标。  
REC 在这类场景里常常看起来“还行”，因为一个大框很容易把目标包进去；但一到 RES 就暴露问题，因为真实轮廓非常难分，尤其在低分辨率和复杂背景下。

第五个坑，是多实例歧义。  
一句话里如果出现“左边第二个杯子”“后排中间的人”“离键盘最近的那支笔”，模型不仅要识别类别，还要做排序、比较和相对位置推理。这比单对象识别难得多。

工程上常见问题和规避方式可总结如下：

| 陷阱 | 典型表现 | 根因 | 规避方式 |
| --- | --- | --- | --- |
| 空间语言歧义 | “左边”“前面”“靠近”表现不稳 | 相对位置建模不足 | 显式加入关系建模或结构解析 |
| 否定表达失败 | “不是那只猫”仍返回猫 | 模型忽略否定词 | 增加 hard negative 与否定样本 |
| 链式关系断裂 | 只抓住最后一个名词 | 长依赖推理弱 | 用分步推理或层次关系建模 |
| 高亮提示偏置 | 模型总偏向被提示区域 | 学到伪相关模式 | 使用对比式区域指导而非直接 overlay |
| 小目标分割差 | 框还行，mask 很差 | 分辨率不足、边界监督弱 | 提高输入分辨率，增加 Dice/边界损失 |
| 多实例混淆 | 同类对象中选错实例 | 排序与参照建模不足 | 强化实例级关系和排序学习 |

再看一个工程例子。  
桌面机器人接到指令：“把显示器前面、白衬衫的人手边那个红杯子拿过来。”

一个真正可用的系统，通常不会一步到位地产生动作，而是至少包含三层处理：

1. 语言解析：拆出“显示器前面”“白衬衫的人”“手边”“红杯子”等约束
2. REC：先把候选杯子范围缩到 1 到 2 个实例
3. RES：输出可抓取的精确轮廓，必要时再接关键点或抓取姿态模块

如果缺少第二步，系统可能在所有红杯子中乱选；  
如果缺少第三步，系统可能虽然找对杯子，却抓在杯口、把手外侧或与手部遮挡区域发生冲突。

所以 Grounding 的工程质量，不取决于模型会不会“说得像懂了”，而取决于它能不能把每个语言约束落实成可检验的区域选择。

---

## 替代方案与适用边界

并不是每个团队都适合直接训练一个高质量的 REC+RES 一体模型。原因很现实：区域级标注贵，表达级标注更贵，分割级标注更贵，训练和迭代成本也高。因此，方案要按资源和目标来选。

第一类替代方案，是用生成式 VLM 合成 referring expression，再配合已有检测框、分割结果或少量人工校验扩充语料。  
它的核心思路不是直接拿生成模型替代 Grounding，而是让生成模型先成为“数据放大器”。对于图像库存多、人工描述少的团队，这种方法能快速补足“描述多样性”与“关系表达覆盖率”。

第二类方案，是 gaze-assisted grounding。  
这里的 `gaze` 指人的注视轨迹或注视热区。它适合人机协作明确、能够采集眼动或注视行为的场景，例如装配辅助、桌面抓取、驾驶辅助、可穿戴交互。优点是当语言本身含糊时，gaze 能提供一个强先验；缺点是额外硬件、噪声处理和跨用户泛化都会增加系统复杂度。

第三类方案，是 CRG 一类的区域对比指导方法。  
它们的共同点是：不一定重训主模型，而是在训练或推理阶段通过原图与遮挡图、目标区域与非目标区域的对比，检查模型的答案是否真正依赖目标区域。这类方法适合已经有现成 VLM 或 Grounding 管线，但希望以较低成本降低区域偏置的团队。

第四类方案，是只做 REC，不做 RES。  
这不是“落后方案”，而是典型的成本收益权衡。如果业务目标只是“给用户圈出来”“给下游模块一个候选框”“在页面上做高亮跳转”，那么只做 REC 往往已经足够。

第五类方案，是 REC + RES 联合建模。  
这是精度最高、信息最完整的方案，但也要求更高的数据、算力和工程维护成本。只有当业务真的依赖像素级边界时，这个投入才划算。

可以按场景做一个压缩选择：

| 方案 | 适用情况 | 优点 | 限制 |
| --- | --- | --- | --- |
| Generative VLM 合成表达 | 图像多、描述少、想快速扩数据 | 低成本放大语料覆盖 | 生成描述质量需要抽检 |
| Gaze-assisted grounding | 人机协作强、表达可能含糊 | 额外先验能缩小候选范围 | 依赖 gaze 采集与对齐 |
| CRG / 区域对比指导 | 已有模型，不想大改训练 | 可在较低成本下缓解偏置 | 依赖候选区域质量 |
| 仅用 REC | 只需粗定位或前置筛选 | 标注和部署简单 | 无法提供精确边界 |
| REC + RES 联合 | 抓取、编辑、抠图、避障 | 同时验证位置与形状 | 标注和训练成本最高 |

适用边界也要说清楚。

如果任务是“把图片里提到的物体圈出来给人看”，REC 往往够用。  
如果任务是“让机器人抓住杯柄而不是杯身”，REC 往往不够，至少需要 RES，很多时候还要叠加关键点检测、6D 姿态估计或可抓取区域预测。  
如果表达中经常出现复杂空间关系、否定、比较级和多实例排序，单纯依赖视觉相似度匹配通常会失效，此时必须增加语言结构解析、关系推理或对比式验证模块。

对新手而言，最实用的判断标准只有一个：  
先问自己，下游动作到底需要“框”，还是需要“边界”。如果答案是后者，就不要假设 REC 能自然替代 RES。

---

## 参考资料

- Georgios Pantazopoulos, Eda B. Ozyigit. *Towards Understanding Visual Grounding in VLMs*. arXiv, 2025.  
  https://www.emergentmind.com/papers/2509.10345?utm_source=openai

- NeurIPS 2024 tutorial / slides on Referring Expression tasks (REC vs. RES).  
  https://neurips.cc/media/neurips-2024/Slides/93378_ROahXfO.pdf?utm_source=openai

- Ho et al. *RSSep: Sequence-to-Sequence Model for Simultaneous Referring Remote Sensing Segmentation and Detection*. ACCV 2024 Workshop.  
  https://openaccess.thecvf.com/content/ACCV2024W/LAMM/papers/Ho_RSSep_Sequence-to-Sequence_Model_for_Simultaneous_Referring_Remote_Sensing_Segmentation_and_ACCVW_2024_paper.pdf?utm_source=openai

- *Referring Object Grasping with Gaze-Assisted Visual Grounding*. Engineering Applications of Artificial Intelligence, 2024.  
  https://www.sciencedirect.com/science/article/pii/S0952197624006511?utm_source=openai

- *Contrastive Region Guidance*. ECCV 2024.  
  https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10172.pdf?utm_source=openai

- *Spatial Language Grounding Failures*. IJCNLP 2025.  
  https://aclanthology.org/2025.ijcnlp-long.183/?utm_source=openai

- Wang et al. *Learning Visual Grounding from Generative Vision and Language Model*. WACV 2025.  
  https://openaccess.thecvf.com/content/WACV2025/papers/Wang_Learning_Visual_Grounding_from_Generative_Vision_and_Language_Model_WACV_2025_paper.pdf?utm_source=openai
