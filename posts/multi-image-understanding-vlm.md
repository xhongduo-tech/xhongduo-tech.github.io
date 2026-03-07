## 核心结论

多图理解的核心，不是“把多张图一起塞进模型”，而是同时解决三个彼此独立但会互相耦合的问题：输入怎样组织，跨图对象怎样对齐，上下文预算不够时怎样压缩。

第一，`interleaved` 输入是目前最实用的统一模板。`interleaved` 的定义是：图像 token 与文本 token 按语义顺序交错排列。哪段文字解释哪张图，就让那张图出现在那段文字附近。这样单图、多图、多页 PDF、多帧视频、多视角样本都能复用同一条推理链路，训练和部署不必为每种输入形态单独设计模板。

第二，多图任务真正困难的部分，通常不是识别单张图里有什么，而是建模图与图之间的关系。比较、排序、跨图指代、故事生成、本质上都要求模型回答一个更细的问题：图 1 中的某个实体，和图 2 中的某个区域，是否在语义上对应。像 ClawMachine 这类方法会把一个区域覆盖的视觉 token 组织成 `token collective`。它的作用可以直接理解为“把属于同一实体的视觉碎片打包”，后续语言解码器不再面对整张图的模糊表示，而是面对一个可引用、可比较的实体级表示。

第三，上下文长度是多图系统的主要工程瓶颈。假设每张图都被切成上千个 patch token，那么 5 到 8 张图就会迅速耗尽上下文窗口，导致文本指令和中间推理空间被挤压。PVC 这类压缩方法把每张图视为“静态视频”，只保留逐帧新增信息，再把每帧压缩为固定数量的 token，例如 64 个 token。这样多图输入仍能保留全局语义和局部增量，而不是把所有 patch 原样塞进解码器。

| 场景 | 输入形态 | 主要难点 | 常见任务 | 例子 |
|---|---|---|---|---|
| 单张图片 | 1 张图 + 指令 | 单图识别 | 描述、OCR、问答 | 识别发票金额 |
| 多图 | 多张独立图片 + 指令 | 跨图比较与指代 | 比较、排序、汇总 | 比较两款冲锋衣 |
| 多帧 | 连续帧 + 指令 | 时序变化建模 | 动作识别、事件定位 | 分析监控片段 |
| 多视角 | 同一对象不同视角 | 视角对齐 | 3D 理解、结构定位 | 识别机械臂部件 |
| 多页文档 | 页面截图 + 问题 | 页间证据整合 | 文档问答、摘要 | 从财报多页找利润变化原因 |

玩具例子：把五页 PPT 截图依次输入，再给出指令“总结每页重点并比较两套方案差异”。如果采用 `interleaved`，模型更容易把“第 3 页这段说明”与“第 3 页图表内容”绑定在一起；如果采用纯并列堆叠，模型仍可能读懂每页内容，但更容易在引用页码、比较方案时发生混淆。

---

## 问题定义与边界

多图理解，指的是模型在同一个上下文窗口中同时接收多张图像与文字，并输出一个统一回答。这里的“统一”很关键，因为系统不是先分别处理每张图，再用业务代码做后处理拼接，而是在同一段 token 序列里直接建模全部图文关系。

最常见的两种输入组织方式如下。

| 格式 | token 顺序 | 优点 | 缺点 | 更适合的任务 |
|---|---|---|---|---|
| `in-front` | 图1 → 图2 → 图3 → 指令 | 实现简单，训练数据容易构造 | 指令与图像距离远，跨图引用容易漂移 | 简单问答、逐图描述 |
| `interleaved` | 图1 → 说明1 → 图2 → 说明2 → 指令 | 图文对应关系明确，局部注意力更稳定 | 数据组织和标注成本更高 | 比较、排序、跨图推理 |
| `grouped-interleaved` | 图1 → 说明1 → 图2 → 说明2 → 小结 → 指令 | 可在中间插入阶段性摘要 | 模板更复杂，训练分布要求更高 | 多页文档、多阶段摘要 |

边界也需要说清楚。多图理解不等于“任意多图都能无损处理”，它至少受三类约束。

### 1. 上下文长度约束

图像不会以像素形式直接进入语言模型，而是先被视觉编码器切分并映射为视觉 token。token 可以理解为模型真正消费的离散计算单位。高分辨率、多页、长视频都会迅速吃掉上下文预算。上下文预算一旦被视觉 token 占满，文本指令、历史对话和中间推理都会被压缩。

### 2. 指代约束

当用户说“左边那张图”“第二张图里的绿色外套”“第 4 页右上角表格”，模型需要把自然语言中的指代表达稳定绑定到具体图像、具体区域、甚至具体对象。如果绑定不稳定，回答会出现典型漂移：页码错位、颜色错位、对象错位。

### 3. 任务约束

不是所有多图任务都需要复杂的关系建模。如果任务只是“分别描述这 3 张图”，并列输入通常就够用；但如果任务是“哪张图最适合城市通勤”“这 6 页里哪一页首次提到利润率下降”“图 2 中的设备和图 5 中的是不是同一型号”，那么系统就必须显式建模跨图关系。

### 4. 分辨率与细节约束

多图系统最容易在压缩阶段损失细节。对于 OCR、小目标检测、医学影像、工业缺陷定位这类任务，决定结果的往往不是整图语义，而是很小的局部文字、纹理或边缘。此时“先压缩再推理”不一定成立，必须保留更高的局部 token 密度。

一个真实工程例子是电商内容审核。系统一次接收商品主图、模特图、细节图、尺码图，再问：“这组素材是否存在主图与细节图信息不一致？”这个问题不能拆成四次单图分类，因为判定条件本身就是跨图一致性，它是标准的多图理解任务。

为了让边界更清晰，可以把任务拆成下面这张表。

| 任务类型 | 是否需要跨图关系 | 是否需要实体级对齐 | 是否容易受上下文限制 |
|---|---|---|---|
| 分别描述每张图 | 低 | 低 | 中 |
| 比较两张图差异 | 高 | 中 | 中 |
| 多页文档问答 | 高 | 中 | 高 |
| 跨图找同一对象 | 高 | 高 | 中 |
| 多帧动作理解 | 高 | 高 | 高 |

---

## 核心机制与推导

先看 token 数为什么会爆炸。假设一张 $1024 \times 1024$ 的图像被切成 $32 \times 32$ 的 patch，那么每个方向上有

$$
\frac{1024}{32}=32
$$

个 patch，总 patch 数为

$$
N = 32 \times 32 = 1024
$$

如果输入 6 张图，仅视觉 token 就有

$$
6 \times 1024 = 6144
$$

个。再加上用户问题、系统提示、输出空间，很容易逼近甚至超过常见上下文窗口。

为了更直观看到增长速度，可用一般式表示：

$$
N_{\text{patch}}(H, W, P)=\frac{H}{P}\times \frac{W}{P}
$$

其中 $H, W$ 是图像分辨率，$P$ 是 patch 边长。若有 $M$ 张图，则总视觉 token 数为

$$
N_{\text{vision}} = M \cdot N_{\text{patch}}(H, W, P)
$$

这说明多图问题首先是一个预算问题，其次才是一个推理问题。

### 1. 输入排列：`interleaved` 为什么更稳

设图像序列为 $S_{img}=[I_1, I_2, \dots, I_n]$，文本片段为 $T_1, T_2, \dots, T_n$，最终查询为 $T_q$。

`in-front` 可写为：

$$
S_{\text{front}} = [I_1^{tok}, I_2^{tok}, \dots, I_n^{tok}, T_q]
$$

`interleaved` 可写为：

$$
S_{\text{inter}} = [I_1^{tok}, T_1, I_2^{tok}, T_2, \dots, I_n^{tok}, T_n, T_q]
$$

两者的差异，不只是“排列不同”，而是注意力路径长度不同。若把文本 token $x$ 对图像 token $y$ 的可达难度粗略近似为位置距离 $d(x,y)$ 的函数，则局部对齐更稳定可以写成：

$$
\mathrm{AlignScore}(x,y) \propto \frac{1}{d(x,y)+\epsilon}
$$

这不是严格训练目标，而是一个工程上常用的直观近似：相关 token 越近，模型越容易在有限层数和有限上下文中建立稳定关联。`interleaved` 的优势，正来自于它系统性缩短了说明文字和对应图像之间的距离。

看一个新手更容易理解的例子。

- `in-front`：先放 5 张图，最后问“第 4 张图里的表格和第 2 张图里的结论是否冲突？”
- `interleaved`：图 1 + “封面”，图 2 + “结论页”，图 3 + “方法页”，图 4 + “数据表格页”，最后提问。

第二种形式不一定提升所有任务，但在“页号 + 内容 + 问题”强相关的任务里，模型更容易把“第 4 张”与“数据表格页”绑定起来。

### 2. 跨图关系：`token collective` 怎样做指代

单张图识别依赖的是局部视觉模式，多图推理依赖的是对象级对齐。也就是说，模型不能只知道“图 1 大概有外套、图 2 大概有外套”，还要知道“图 1 中这个绿色外套”和“图 2 中那个蓝色夹克”分别对应哪些区域，后续才能比较材质、颜色、口袋设计或品牌标识。

设图像编码器输出视觉 token 集合 $V=\{v_1, v_2, \dots, v_m\}$。若一个目标框 $bbox_j$ 覆盖若干 patch，则该实体的 `token collective` 可定义为：

$$
C_j = \mathrm{sort}\big(\{v_k \mid center(v_k)\in bbox_j\}\big)
$$

其中：

- $v_k$ 是第 $k$ 个视觉 token
- $center(v_k)$ 表示该 token 对应 patch 的中心坐标
- `sort` 用于给集合施加稳定顺序，避免同一实体每次编码后的内部顺序不一致

为了得到实体级表示，常见做法是再对集合做池化：

$$
e_j = \mathrm{Pool}(C_j)
$$

其中 $\mathrm{Pool}$ 可以是平均池化、注意力池化或轻量聚合器。这样，后续语言解码器面对的不再是“整张图的视觉海洋”，而是若干个可引用的实体表示 $e_1, e_2, \dots$。

如果要比较图 1 的实体 $a$ 与图 2 的实体 $b$，可以进一步定义相似度：

$$
s(a,b)=\frac{e_a^\top e_b}{\|e_a\|\|e_b\|}
$$

当用户问“图 2 右上角的设备和图 4 左下角的是不是同一类型”，系统就不必把整个问题交给隐式注意力猜测，而可以在实体级表示之间做更直接的对齐。

一个简单例子：

- 图 1：白底商品图，绿色冲锋衣
- 图 2：模特上身图，蓝色夹克
- 问题：“哪一件更适合雨天徒步？”

若没有实体级表示，模型只能依赖整图语义作答；若有 `token collective`，系统至少可以把“衣服主体区域”“帽檐区域”“拉链区域”抽出来做比较，再结合文字描述回答防水性和使用场景。

### 3. 上下文压缩：PVC 的核心思想

PVC 的核心，不是简单下采样，而是把一张图看成“逐帧揭示信息的静态视频”。第一帧保留全局轮廓，后续帧逐步补充新信息，只对增量内容分配计算与 token 预算。

其简化形式可写为：

$$
F_t = f_{\text{attn}}(F_{t-1}) + \Delta_t
$$

其中：

- $F_t$ 表示第 $t$ 帧的特征表示
- $f_{\text{attn}}$ 表示基于前一帧状态的注意力更新
- $\Delta_t$ 表示第 $t$ 帧带来的新增信息

如果再引入 AdaLN，自适应地按时间步或帧条件调节归一化参数，则可写为：

$$
\Delta_t' = \mathrm{AdaLN}(\Delta_t, t)
$$

AdaLN 可以直接理解为：归一化不再使用一套固定参数，而是根据“当前是第几帧、当前保留了什么信息”动态调整，使不同帧承担不同语义角色。

最终，把每帧高维 patch 特征压缩为固定数量 token。例如每张图切成 4 帧，每帧保留 64 个 token，则单张图压缩后 token 数为

$$
N_{\text{PVC}} = 4 \times 64 = 256
$$

若原图是 1024 个 patch token，则压缩率为

$$
r = 1-\frac{256}{1024}=75\%
$$

若输入 2 张图，则总 token 数从

$$
2 \times 1024 = 2048
$$

变为

$$
2 \times 4 \times 64 = 512
$$

这意味着相同上下文预算下，系统可以容纳更多页面、更长文本，或者把节省下来的 token 用于保留关键区域的高分辨率细节。

PVC 适合什么场景，也可以明确成表。

| 场景 | 信息冗余度 | 是否适合 PVC | 原因 |
|---|---|---|---|
| 多页 PDF | 高 | 适合 | 页面结构重复，增量信息明显 |
| 相邻视频帧 | 高 | 适合 | 帧间内容变化通常局部 |
| 同一对象多视角 | 中到高 | 较适合 | 全局对象一致，局部视角变化 |
| 商品图混合集 | 中 | 视情况而定 | 类别相似时可压缩，不相似时收益下降 |
| 密集 OCR | 低 | 不适合激进压缩 | 小字细节容易丢失 |
| 医学影像判读 | 低 | 谨慎使用 | 局部纹理可能决定诊断结果 |

玩具例子：一张流程图的阅读可以拆成 4 帧。
第一帧保留版面结构，第二帧保留标题与框名，第三帧保留箭头方向，第四帧保留小字说明。语言模型看到的是 4 组压缩后的关键 token，而不是 1024 个未经筛选的 patch。

---

## 代码实现

下面给出一个可运行的简化示例，演示四件事：

1. 如何构造 `interleaved` 序列  
2. 如何估算原始 patch token 数  
3. 如何估算 PVC 压缩后的 token 数  
4. 如何把区域映射成一个简化版 `token collective`

代码只依赖 Python 标准库，直接可运行。

```python
from dataclasses import dataclass
from math import isclose
from typing import List, Tuple


@dataclass
class ImageItem:
    name: str
    width: int
    height: int
    patch_size: int
    text_chunk: str


@dataclass
class PatchToken:
    token_id: int
    center_x: float
    center_y: float


BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


def patch_tokens_per_image(width: int, height: int, patch_size: int) -> int:
    if width % patch_size != 0 or height % patch_size != 0:
        raise ValueError("width and height must be divisible by patch_size")
    return (width // patch_size) * (height // patch_size)


def build_interleaved_sequence(images: List[ImageItem], query: str) -> List[str]:
    sequence = []
    for image in images:
        token_count = patch_tokens_per_image(image.width, image.height, image.patch_size)
        sequence.append(f"<image:{image.name}:{token_count}tok>")
        sequence.append(image.text_chunk)
    sequence.append(query)
    return sequence


def raw_visual_tokens(images: List[ImageItem]) -> int:
    return sum(
        patch_tokens_per_image(image.width, image.height, image.patch_size)
        for image in images
    )


def pvc_compressed_tokens(
    num_images: int,
    frames_per_image: int = 4,
    tokens_per_frame: int = 64,
) -> int:
    return num_images * frames_per_image * tokens_per_frame


def compression_ratio(raw_tokens: int, compressed_tokens: int) -> float:
    if raw_tokens <= 0:
        raise ValueError("raw_tokens must be positive")
    return 1.0 - (compressed_tokens / raw_tokens)


def make_patch_grid(width: int, height: int, patch_size: int) -> List[PatchToken]:
    cols = width // patch_size
    rows = height // patch_size
    tokens = []
    token_id = 0
    for row in range(rows):
        for col in range(cols):
            center_x = col * patch_size + patch_size / 2
            center_y = row * patch_size + patch_size / 2
            tokens.append(PatchToken(token_id, center_x, center_y))
            token_id += 1
    return tokens


def token_collective(tokens: List[PatchToken], bbox: BBox) -> List[int]:
    x1, y1, x2, y2 = bbox
    selected = [
        token.token_id
        for token in tokens
        if x1 <= token.center_x <= x2 and y1 <= token.center_y <= y2
    ]
    return sorted(selected)


def main() -> None:
    images = [
        ImageItem(
            name="page1",
            width=1024,
            height=1024,
            patch_size=32,
            text_chunk="第1页：系统架构总览。",
        ),
        ImageItem(
            name="page2",
            width=1024,
            height=1024,
            patch_size=32,
            text_chunk="第2页：成本与性能对比。",
        ),
    ]

    query = "请总结每页重点，并指出哪一页首次出现成本下降结论。"

    sequence = build_interleaved_sequence(images, query)
    raw_tokens = raw_visual_tokens(images)
    compressed_tokens = pvc_compressed_tokens(
        num_images=len(images),
        frames_per_image=4,
        tokens_per_frame=64,
    )
    ratio = compression_ratio(raw_tokens, compressed_tokens)

    grid = make_patch_grid(width=128, height=128, patch_size=32)
    bbox = (16, 16, 80, 80)
    collective = token_collective(grid, bbox)

    assert len(sequence) == 5
    assert raw_tokens == 2048
    assert compressed_tokens == 512
    assert isclose(ratio, 0.75)
    assert collective == [0, 1, 4, 5]

    print("Interleaved sequence:")
    for item in sequence:
        print("  ", item)

    print("\nToken statistics:")
    print(
        {
            "raw_tokens": raw_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": round(ratio, 2),
        }
    )

    print("\nExample token collective:")
    print({"bbox": bbox, "token_ids": collective})


if __name__ == "__main__":
    main()
```

这段代码虽然是玩具实现，但逻辑上对应真实系统的关键环节：

1. 视觉编码器把图像转成 patch token  
2. 数据管线把样本组织成 `in-front` 或 `interleaved`  
3. 若 token 超预算，进入压缩模块，例如 PVC  
4. 若任务依赖实体对齐，则基于区域或检测框构造 `token collective`  
5. 最终把视觉 token 与文本 token 一并送入语言解码器

如果把这个流程代入真实工程，例如多页 PDF 问答，输入通常会被组织成下面这种形态：

- 图 1 + “封面页”
- 图 2 + “目录页”
- 图 3 + “利润表”
- 图 4 + “现金流量表”
- 用户问题：“利润率下降的主要原因是什么，首次出现在哪一页？”

这类任务如果只把所有图片堆在最前面，模型可能知道第 3 页和第 4 页各讲什么，但在回答“首次出现在哪一页”时容易页码错位；`interleaved` 则更容易保留“页内容”和“页标签”的对应关系。

再给一个面向新手的输入输出示例。假设要比较两张商品图：

- 图 1：防水冲锋衣商品页
- 图 2：城市通勤夹克商品页
- 问题：“哪一件更适合连续小雨天气下的通勤？”

系统不是先各自生成两段描述再人工拼接，而是在同一个 token 序列里同时建模两张图和问题，因此可以直接输出“图 1 更适合，因为帽檐、压胶拉链和面料描述更偏向防水场景”。

---

## 工程权衡与常见坑

多图系统最常见的问题，不是模型参数不够，而是训练格式、推理格式、压缩策略和任务目标之间没有对齐。

| 训练格式 | 推理格式 | 典型表现 | 风险 |
|---|---|---|---|
| 只用 `in-front` | 也用 `in-front` | 通常稳定 | 跨图引用能力上限较低 |
| 只用 `in-front` | 改成 `interleaved` | 常见掉点 | 模型未学过交错式图文对齐 |
| 混合训练 | `in-front` 或 `interleaved` | 更稳健 | 数据构造和清洗成本更高 |
| 强压缩训练 | 轻压缩推理 | 细节可能恢复不足 | 训练分布偏向低信息密度 |
| 轻压缩训练 | 强压缩推理 | 常见性能骤降 | 推理时信息损失超出训练经验 |

### 常见问题 1：多图顺序感丢失

如果 prompt 是“先给两张图，最后再问哪个更适合城市通勤”，模型可能知道两张图分别是什么，但不稳定地记住“第一张”和“第二张”的角色。症状通常表现为：

- 回答内容正确，但图号引用错位
- 比较结论稳定，证据出处不稳定
- 复述时把图 1 与图 2 的描述串台

解决方法通常有三类：

- 使用 `interleaved`，让图像与说明邻接
- 在文本中显式加入页号、图号、顺序标签
- 训练时混合 `in-front` 与 `interleaved`，避免模型只学到单一模板

### 常见问题 2：压缩后细节丢失

PVC 不是免费午餐。它适合信息高度冗余的场景，例如多页文档、相邻视频帧、同一对象多视角。如果图像之间差异很大，或者任务高度依赖局部细节，比如小字 OCR、票据金额识别、精细医学影像判读，那么固定 64 个 token 可能远远不够。

一个实用判断是：

$$
\text{需要的 token 配额} \propto \text{局部细节密度} \times \text{任务容错率}^{-1}
$$

这不是论文中的标准公式，而是工程上非常实用的经验表达。局部细节越密、任务容错率越低，可压缩空间就越小。

### 常见问题 3：跨图指代不稳定

用户问“图 2 右上角的红框”和“图 4 里的同一设备”时，如果系统没有显式实体表示，模型只能依赖隐式注意力做猜测，容易答偏。常见修复手段包括：

- 给每张图添加顺序标签、页号标签或语义标签
- 对关键区域构造 `bbox -> token collective`
- 把区域描述写入 prompt，而不是完全依赖“它”“那个设备”这类代词
- 在训练集中加入跨图比较、跨图找同一对象这类样本

### 常见问题 4：图像数量一多，文本空间被挤压

很多系统在多图场景下只关注“能不能塞下图像”，却忽略了文本也需要预算。用户问题、系统提示、检索到的额外文本、模型输出都要占上下文。若视觉 token 吃掉全部窗口，模型即使“看到了图”，也没有足够空间进行链式比较和证据整合。

因此工程上往往要预留一个明确预算：

| 预算项 | 建议思路 |
|---|---|
| 视觉 token | 先估算总量，再决定是否压缩 |
| 用户问题 | 保留完整，不要过度截断 |
| 系统提示 | 尽量短，避免泛泛说明 |
| 输出空间 | 为比较、解释、引用页码预留足够 token |
| 中间检索文本 | 只保留与当前问题直接相关的内容 |

### 常见问题 5：训练样本看起来是多图，实际上不是多图任务

有些数据集只是把多张图拼在一起，但监督信号仍是单图标签，例如“这组图里有没有鞋子”。这类样本会让模型学会“看见多张图”，却未必学会“比较多张图”。如果目标任务是跨图关系推理，那么训练数据必须显式包含：

- 图间差异判断
- 图间因果或时间顺序
- 跨图指代
- 多页证据聚合

---

## 替代方案与适用边界

多图理解不只有一条路线。工程上至少有三种常见方案，它们分别解决不同瓶颈。

| 方案 | token 使用 | 跨图指代能力 | 实现复杂度 | 适用场景 | 主要问题 |
|---|---|---|---|---|---|
| 纯 `in-front` | 高 | 中 | 低 | 简单多图问答 | 图文距离远，复杂引用易错 |
| `interleaved` + `token collective` | 中 | 高 | 中到高 | 比较、排序、跨图实体推理 | 数据和区域标注成本较高 |
| PVC 逐帧压缩 | 低 | 中到高 | 高 | 多页文档、长视频、多视角 | 压缩策略不当会损失细节 |
| 外接 grounding module | 中到高 | 高 | 高 | 精确定位、区域问答 | 系统链路更长，部署更重 |
| 检索式两阶段方案 | 低到中 | 中 | 中 | 图很多但问题稀疏 | 前检索错误会影响后续推理 |

### 1. 纯 `in-front`

优点是简单，适合快速上线。视觉编码、拼接、解码几乎都能沿用单图路径。缺点也同样直接：一旦问题涉及“第几张图”“哪张图更像”“图 3 与图 5 是否一致”，模型就要在长距离 token 上建立复杂对齐，错误率通常上升。

### 2. `interleaved` + `token collective`

这是目前最均衡的路线。`interleaved` 解决图文邻接问题，`token collective` 解决实体级引用问题。它特别适合：

- 商品多图比较
- 多页 PDF 问答
- 跨图找同一对象
- 多步图文汇总

代价是数据构造更复杂。你不仅要知道“这张图是什么”，还要知道“这段话解释哪张图”“这个框覆盖哪个实体”。

### 3. PVC 逐帧压缩

PVC 的价值在于先解决预算问题，再谈复杂推理。对于 20 页文档、长视频、多视角扫描，这类场景如果不先压缩，后面的关系建模往往根本进不了上下文。它不是替代 `interleaved`，更像是放在前面的预算控制器。

### 4. 外接 grounding module

`grounding module` 可以理解为一个专门负责“把语言短语对到视觉区域”的附加模块。它适合需要精确定位的场景，例如：

- “第 2 张图里左上角的部件”
- “图 4 中与图 1 同型号的设备”
- “哪张图出现了和说明书一致的按钮布局”

它能提升精确对齐能力，但会让系统从“一个端到端 VLM”变成“VLM + 区域模块 + 后处理”的多组件架构，训练、部署、调试都更复杂。

### 5. 检索式两阶段方案

如果图像非常多，例如上百页扫描件，完全端到端一起输入通常不现实。工程上常见做法是：

1. 先用轻量模型或检索器从全部图片中召回最相关的若干页  
2. 再把这些页以 `interleaved` 方式送入主模型做精推理

这条路线牺牲了“全局一次建模”，换来可扩展性。缺点是前段召回一旦漏掉关键页，后段模型再强也无法补救。

因此，可以给出一个更实用的选型判断：

- 图少、问题简单：先用 `in-front`
- 图不多，但问题依赖逐图对应关系：优先 `interleaved`
- 图很多、上下文明显不够：在 `interleaved` 基础上接 PVC
- 问题依赖精确实体对照：增加 `token collective` 或 grounding 机制
- 图像总量远超上下文能力：采用检索式两阶段方案

一句话概括边界：多图系统的上限，不只由模型大小决定，更由“输入模板、对齐粒度、压缩策略”三者是否匹配任务决定。

---

## 参考资料

1. **LLaVA-NeXT Interleave Blog**  
   链接：https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave  
   核心贡献：说明了 `interleaved` 图文输入为什么能统一单图、多图、视频等输入形式，以及这种模板如何改善多模态上下文组织。  
   建议阅读方式：先看输入样例和任务案例，再看为什么交错输入比简单堆叠更适合复杂推理。

2. **Yang et al., PVC: Progressive Visual Token Compression for Unified Image and Video Understanding, CVPR 2025**  
   链接：https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_PVC_Progressive_Visual_Token_Compression_for_Unified_Image_and_Video_CVPR_2025_paper.pdf  
   核心贡献：提出逐步压缩视觉 token 的框架，把图像和视频统一到“帧增量压缩”的视角中，重点解决长上下文场景下视觉 token 过多的问题。  
   建议阅读方式：先看方法图和压缩流程，再重点看实验部分里不同 token 预算下的效果变化。

3. **ClawMachine, ICLR 2025**  
   链接：https://proceedings.iclr.cc/paper_files/paper/2025/hash/b1abd42eb5aace7f0ad9ba9cfb029f54-Abstract-Conference.html  
   核心贡献：强调在多图或复杂视觉场景中，单靠整图表示不足以支持稳定指代，需要把对象级视觉表示显式组织起来，这正是 `token collective` 思路的价值所在。  
   建议阅读方式：先理解它为什么要从“整图表示”转向“实体级表示”，再看它如何支持跨图引用与比较任务。

4. **建议配套阅读顺序**  
   如果是第一次接触多图理解，推荐按下面顺序阅读：  
   先看 LLaVA-NeXT Interleave，理解输入模板；再看 PVC，理解预算与压缩；最后看 ClawMachine，理解跨图实体对齐。这样能把“怎么摆输入”“怎么省 token”“怎么做指代”三件事拆开理解。
