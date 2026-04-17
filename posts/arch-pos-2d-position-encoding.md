## 核心结论

2D 位置编码的作用，是让 Transformer 在处理图像、视频帧、表格版面这类二维网格输入时，显式知道“谁在左边、谁在上面、谁离得更远”。如果没有这层信息，模型只看到一串 token，无法稳定区分“左右相邻”和“上下相邻”。

二维场景里常见的做法有四类：

| 方法 | 核心思路 | 是否保留二维方向信息 | 分辨率外推 | 典型场景 |
| --- | --- | --- | --- | --- |
| 可学习 2D 位置嵌入 | 给每个网格位置一个独立向量 | 强 | 弱，超出训练分辨率要插值 | 原始 ViT |
| 分解式 2D RoPE | 行、列各分一半维度，分别做旋转 | 强 | 较强 | ViT、视觉 Transformer |
| 2D ALiBi | 在 attention 分数上按二维距离加线性偏置 | 中，偏重相对距离 | 强 | 遥感、文档版面 |
| 拼块后 1D RoPE | 先把二维 patch 按固定顺序排成序列，再做 1D 编码 | 间接保留 | 强，但依赖顺序一致 | VisionLLM、InternVL |

对初学者来说，可以先记住一句话：2D RoPE 是“旋转向量让相对位移进入点积”，2D ALiBi 是“给远距离 token 额外扣分”。二者都在解决同一个问题：让注意力分数不再只依赖内容，还依赖二维空间关系。

一个玩具例子：把一张图片切成 $14\times14$ 个 patch。左上角 patch 与它右边的 patch 只相差一列；左上角 patch 与右下角 patch 则相差很多行和列。加入 2D 位置编码后，模型会自然学到前者更“近”，后者更“远”，注意力不再把它们当成同样普通的两个 token。

---

## 问题定义与边界

问题先定义清楚。图像、表格、版面天然是二维网格，每个 token 都有坐标 $(x,y)$。但标准 Transformer 的 self-attention 只对一维序列定义，原始公式里没有“横向”和“纵向”的概念。也就是说，如果只把 patch 展平成序列，模型看到的是第 37 个 token、第 38 个 token，却不知道它们到底是左右相邻还是换行后的上下相邻。

这里的“位置编码”，白话说就是给 token 加一份“坐标感”。目标不是简单记住绝对编号，而是让注意力能反映二维空间关系，尤其是相对位移。理想情况下，token $i$ 和 token $j$ 的相关性应当与 $(x_i-x_j, y_i-y_j)$ 有稳定关系，而不是仅与一维下标 $i-j$ 有关系。

这件事有三个边界：

1. 需要处理的是二维网格，不是纯文本序列。
2. 希望模型能适配不同分辨率，而不是只记住训练时那一张固定大小的网格。
3. 位置编码不能把计算代价抬得过高，否则高分辨率输入会直接拖垮 attention。

下面这张表可以看出几种方案解决问题的角度不同：

| 方案 | 输入坐标 | 输出作用位置 | 覆盖的空间关系 | 主要限制 |
| --- | --- | --- | --- | --- |
| 1D RoPE | 一维序列位置 $i$ | Q/K 向量 | 只覆盖序列相对位移 | 二维结构被展平后会失真 |
| 2D RoPE | 二维坐标 $(x,y)$ | Q/K 向量 | 同时覆盖行差与列差 | 维度切分必须严格正确 |
| 2D ALiBi | 二维坐标 $(x,y)$ | attention logits | 直接编码距离衰减 | 只表达相对距离，不保绝对坐标 |

新手版理解可以这样想：两个 patch 如果在图上左右相邻或上下相邻，模型应该觉得它们更容易互相参考；如果隔着很远的区域，attention 分数应该更低。二维位置编码本质上就是把这种“近邻优先”的结构性知识显式写进模型。

---

## 核心机制与推导

### 1. 2D RoPE：把二维坐标变成向量旋转

RoPE 的全称是 Rotary Position Embedding，白话说是“用旋转角度表达位置”。一维 RoPE 的关键性质是：对 query 和 key 同时施加与位置相关的旋转后，它们的点积只依赖相对位置差，而不是绝对位置本身。

二维版做法通常是把 embedding 维度分成两半：

- 前 $d/2$ 维给行坐标 $x$
- 后 $d/2$ 维给列坐标 $y$

每两个维度组成一个二维小块，对这个小块应用旋转矩阵：

$$
R(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

如果 token 位于坐标 $(x,y)$，那么它的 query 和 key 分别被行角度 $\theta_x$、列角度 $\theta_y$ 旋转。于是，二维位置编码后的点积可以写成“行方向相对位移贡献 + 列方向相对位移贡献”的组合。直观上，模型不再只知道“你是第几个 token”，而是知道“你和我差几行、差几列”。

玩具例子：patch A 在 $(1,2)$，patch B 在 $(2,3)$。A 和 B 的相对位移是 $(-1,-1)$。2D RoPE 并不直接把这个差值写成一个数字塞进模型，而是通过对 Q/K 的旋转，让最终 attention score 只依赖这组差值。这样，位于别的位置、但同样相差一行一列的 patch 对，也会产生相似的空间关系响应。这就是“平移不变性”，白话说就是整体平移后，相对关系不变。

### 2. 2D ALiBi：把二维距离直接写进分数

ALiBi 的全称是 Attention with Linear Biases，白话说是“给注意力分数加一个按距离变化的偏置”。它不去旋转向量，而是直接在 logits 上动手：

$$
\text{score}_{ij}
=
\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d}}
-
\text{dist}((x_i,y_i),(x_j,y_j)) \cdot m_h
$$

其中：

- $\text{dist}$ 是二维距离，可以是欧氏距离，也可以是曼哈顿距离
- $m_h$ 是第 $h$ 个注意力头的斜率
- 斜率常按几何级数分配，让不同头关注不同距离尺度

这条公式的意思很直接：内容相似度先算出来，再对远距离 token 扣分。离得越远，扣得越多。于是有的头更偏向局部，有的头可以看更远。

如果还是看 A$(1,2)$ 和 B$(2,3)$，两者欧氏距离是：

$$
\sqrt{(1-2)^2 + (2-3)^2} = \sqrt{2}
$$

若另一个 patch C 在 $(10,10)$，那么 A 和 C 的距离远大于 $\sqrt{2}$，它们的分数会被减去更大的偏置，因此更难产生高注意力。

### 3. 为什么这两类方法都有效

从机制上看，2D RoPE 和 2D ALiBi 分别把位置注入到两个不同地方：

- 2D RoPE 注入 Q/K 表示空间，是“先改向量，再算分数”
- 2D ALiBi 注入 logits 表示空间，是“先算内容，再按距离修正”

二者都在逼近同一个目标：让 attention 分数与二维相对位移相关。差别在于，RoPE 更像把几何关系编码进表示空间；ALiBi 更像在决策层面加规则。

一个真实工程例子是文档版面分析。表格单元格、标题、页眉、页脚常常要求模型理解“上下对齐”和“左右邻接”。这类任务中，2D ALiBi 很实用，因为它直接把二维距离偏置加到 attention 上，不需要为每个分辨率都重新学一套绝对位置向量。

---

## 代码实现

下面先给一个最小可运行的 Python 示例，演示两件事：

1. 如何为二维网格生成 2D RoPE 角度并做旋转
2. 如何为 2D ALiBi 生成距离偏置矩阵

```python
import math
import numpy as np

def rope_angles(length, dim_half):
    assert dim_half % 2 == 0
    freqs = 1.0 / (10000 ** (np.arange(0, dim_half, 2) / dim_half))
    pos = np.arange(length)[:, None]
    angles = pos * freqs[None, :]
    return angles  # [length, dim_half/2]

def rotate_half(x):
    assert x.shape[-1] % 2 == 0
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return np.stack([-x2, x1], axis=-1).reshape(x.shape)

def apply_1d_rope(x, angles):
    # x: [dim_half], angles: [dim_half/2]
    cos = np.cos(angles).repeat(2, axis=-1)
    sin = np.sin(angles).repeat(2, axis=-1)
    return x * cos + rotate_half(x) * sin

def apply_2d_rope(vec, row, col, row_angles, col_angles):
    d = vec.shape[-1]
    assert d % 2 == 0
    half = d // 2
    row_part = apply_1d_rope(vec[:half], row_angles[row])
    col_part = apply_1d_rope(vec[half:], col_angles[col])
    return np.concatenate([row_part, col_part], axis=0)

def alibi_bias(coords, slope):
    n = len(coords)
    bias = np.zeros((n, n), dtype=np.float32)
    for i, (x1, y1) in enumerate(coords):
        for j, (x2, y2) in enumerate(coords):
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            bias[i, j] = -dist * slope
    return bias

# 2x2 网格，embedding 维度 8，其中 4 维给行，4 维给列
coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
row_angles = rope_angles(length=2, dim_half=4)
col_angles = rope_angles(length=2, dim_half=4)

v = np.array([1., 2., 3., 4., 5., 6., 7., 8.], dtype=np.float32)
v00 = apply_2d_rope(v, 0, 0, row_angles, col_angles)
v11 = apply_2d_rope(v, 1, 1, row_angles, col_angles)

assert v00.shape == (8,)
assert v11.shape == (8,)
assert not np.allclose(v00, v11)

bias = alibi_bias(coords, slope=0.5)
assert bias.shape == (4, 4)
assert bias[0, 0] == 0.0
assert bias[0, 3] < bias[0, 1]  # 更远的点惩罚更大

print("2D RoPE and 2D ALiBi toy example passed.")
```

这个例子里，`apply_2d_rope` 的核心就是“前半维看行，后半维看列”。如果这里分错了，比如把某些行维度和列维度混在一起，模型得到的就不是稳定的二维相对关系，而是一种扭曲的位置映射。

如果写成伪代码，结构更直观：

```python
for row in range(H):
    for col in range(W):
        q[row, col] = rotate_row(q[row, col][:d//2], theta_x[row]) \
                    + rotate_col(q[row, col][d//2:], theta_y[col])
```

2D ALiBi 的实现重点则是预计算所有 patch 对之间的距离：

```python
for i, (x1, y1) in enumerate(coords):
    for j, (x2, y2) in enumerate(coords):
        dx = x1 - x2
        dy = y1 - y2
        bias[h, i, j] = -math.sqrt(dx * dx + dy * dy) * slope[h]
```

真实工程里，常见做法不是手写双重循环，而是一次性用张量广播生成 $N \times N$ 的距离矩阵，再按头数扩展成 `num_heads x N x N`。否则高分辨率下会很慢。

再看一个真实工程例子。VisionLLM 或 InternVL 一类系统处理高分辨率图片时，通常先把大图切成多个子图，再把每个子图切成 patch，最后按 row-major，也就是“行主序”拼成一条长序列。此时它不一定显式做 2D RoPE，而是把“二维切块顺序”固定好，再沿这条一维序列应用 1D RoPE。这个方案本质上是用严格的拼接规则，间接保留二维邻接关系。

---

## 工程权衡与常见坑

第一个权衡是表达能力与实现复杂度。可学习 2D 位置嵌入最直观，每个位置一个向量，但分辨率变了就麻烦，往往需要插值。2D RoPE 的外推性更好，因为它依赖坐标生成旋转角，不要求为每个新位置重新学习参数。2D ALiBi 更轻，因为它只是加偏置，但它表达的是相对距离，不直接携带绝对位置标签。

第二个权衡是显存。注意力复杂度和序列长度平方相关，即 $O(N^2)$。图像分辨率翻倍后，patch 数不是翻倍，而是通常变成四倍。比如 ViT-L 使用 $16\times16$ patch：

| 输入分辨率 | 网格大小 | patch 数 $N$ | attention 复杂度近似 | 相对 224 的 attention 存储倍数 |
| --- | --- | --- | --- | --- |
| 224×224 | 14×14 | 196 | $196^2$ | 1× |
| 448×448 | 28×28 | 784 | $784^2$ | 16× |

这里要分清两个量：

- patch 数从 196 变到 784，是 4 倍
- attention 矩阵大小从 $196^2$ 变到 $784^2$，是 16 倍

这就是为什么高分辨率视觉模型很容易爆显存。很多初学者只盯着“token 变 4 倍”，没意识到 attention 是平方增长。

第三个常见坑是顺序一致性。无论是显式 2D 编码，还是拼块后做 1D RoPE，只要你定义了某种空间顺序，训练和推理就必须一致。比如把 $2\times2$ 图块按下面顺序拼：

- 正确：左上 → 右上 → 左下 → 右下
- 错误：左下 → 右下 → 左上 → 右上

如果顺序搞反，1D RoPE 看到的相对距离就变了。模型原本以为相邻的 token，现在可能来自完全不同的图像区域，空间一致性直接被破坏。

第四个坑是 2D RoPE 的维度切分。它要求每个轴各拿一半维度，而且每一半内部仍按成对维度旋转。如果 `d/2` 不是偶数，或者实现里把行角度误用到了列向量上，理论上的相对性就不成立。表现上往往不是立刻报错，而是训练效果差、外推不稳定、换分辨率后退化明显，这类问题最难排查。

---

## 替代方案与适用边界

如果任务主要依赖固定分辨率，比如经典图像分类，可学习 2D 位置嵌入仍然是简单有效的基线。原始 ViT 就是这样做的：224×224 图像配合 patch size 16，会得到 $14\times14=196$ 个 patch，于是直接学习 196 个位置向量。这种方案训练简单，但超出训练分辨率时需要插值，泛化不如 RoPE/ALiBi 稳定。

如果任务强调跨分辨率推理，例如遥感大图、文档长页、多页拼接，2D ALiBi 往往更合适。因为它只依赖距离偏置，不需要额外学习新的绝对位置参数。代价是它不能恢复绝对坐标。白话说，它知道“你离我多远”，但不天然知道“你在页面左上角还是右下角”。因此做版面理解时，常要额外加入 layout token、bbox 信息或显式坐标特征。

如果系统最终要接到 LLM 上，常见折中是“动态拼块 + 1D RoPE”。也就是先把二维图像切成多个子图，再把 token 展开为一维序列送进语言模型。这种做法的好处是工程复用高，可以直接复用 LLM 的位置编码和上下文窗口机制；坏处是空间关系依赖拼接顺序，必须统一 raster scan，也就是固定行主序扫描。

下面这张表总结适用边界：

| 方案 | 是否保留绝对位置 | 是否突出相对距离 | 显存负担 | 分辨率扩展性 | 适用边界 |
| --- | --- | --- | --- | --- | --- |
| 2D RoPE | 弱，重点不在绝对坐标 | 强 | 中 | 强 | 通用视觉 Transformer |
| 2D ALiBi | 弱 | 很强 | 较低 | 很强 | 遥感、文档、长距离外推 |
| 1D RoPE（拼块后） | 依赖拼接顺序间接保留 | 中 | 中 | 强 | VisionLLM、多模态接 LLM |
| 可学习 2D 嵌入 | 强 | 弱 | 中 | 弱 | 固定分辨率分类/预训练 |

一个新手容易忽略的点是：没有一种方案在所有任务上都最优。若任务要求“知道版心、页边距、页码区域这些绝对版面结构”，只靠 ALiBi 通常不够；若任务要求高分辨率外推，可学习绝对嵌入又会比较吃亏。工程里通常不是争论谁最先进，而是先看你的输入是否会变分辨率、是否需要强几何外推、是否受限于现有 LLM 框架。

---

## 参考资料

| 来源 | 内容摘要 | 用途 |
| --- | --- | --- |
| [EmergentMind: RoPE-2D](https://www.emergentmind.com/topics/rope-2d?utm_source=openai) | 解释 2D RoPE 的轴向拆分、旋转原理与相对位置性质 | 理解 2D RoPE 数学机制 |
| [NeurIPS 2023 CROMA Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf) | 给出 2D-ALiBi/X-ALiBi，并展示高分辨率外推结果 | 理解 2D ALiBi 与遥感应用 |
| [Michael Brenndoerfer: Vision Encoders for VLMs](https://mbrenndoerfer.com/writing/vision-encoders-vlms-siglip-resolution-architecture) | 说明 ViT 分辨率变化、patch 数增长与显存代价 | 理解 224→448 的复杂度变化 |
| [InternVL 1.5 Introduction](https://internvl.readthedocs.io/en/latest/internvl1.5/introduction.html?utm_source=openai) | 介绍动态分辨率、子图切分、token 压缩与 LLM 接口 | 理解高分辨率视觉输入的工程落地 |
| [REOrder 项目说明](https://d3tk.github.io/REOrder/?utm_source=openai) | 讨论 token 顺序对视觉建模的影响 | 理解拼块顺序错误带来的位置失真 |
