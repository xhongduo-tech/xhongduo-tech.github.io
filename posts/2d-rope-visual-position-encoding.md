## 核心结论

2D RoPE 的核心作用，是把图像 token 的位置从“一个序号”改成“行坐标 + 列坐标”分别编码。RoPE 是 Rotary Position Embedding，白话解释就是“把向量按位置对应的角度旋转，让注意力天然感知相对距离”。对图像来说，标准 1D RoPE 只能看到 token 在展平序列里的先后顺序，无法直接区分“这是上下相邻”还是“这是左右相邻”。2D RoPE 通过把位置写成 $(x, y)$，分别对两个轴做旋转，使注意力分数只依赖相对行差和列差。

M-RoPE 可以看成 2D RoPE 的多模态扩展。这里的 M 指 multi-dimensional 或 multimodal，白话解释就是“把多个坐标轴统一放进同一套旋转框架里”。在图像的高、宽之外，再加入视频的时间轴 $t$，于是位置从 $(x, y)$ 变成 $(t, x, y)$。这样模型不仅知道两个 patch 是否左右相邻、上下相邻，还知道它们是否来自不同时间点。

一个最小玩具例子能直接说明问题。设图像被切成 $2 \times 2$ 的 patch 网格，四个位置分别是 $(0,0)$、$(0,1)$、$(1,0)$、$(1,1)$，并令 $\theta_x = \theta_y = \pi/8$。那么 $(0,0)$ 与 $(1,0)$ 在 $x$ 轴上相差 1、在 $y$ 轴上相差 0，对应旋转后的点积会出现 $\cos(\pi/8)$ 这一项；这说明模型感知到“行方向差了一个单位”。同理，$(0,0)$ 与 $(0,1)$ 的差异主要来自列方向。这个性质不是靠额外规则硬编码，而是直接来自旋转点积的数学结构。

下面这张表先给出三种常见方案的整体差异。

| 方法 | 位置表示方式 | 能否区分行/列/时间 | 相对位置建模 | 高分辨率/长视频稳定性 |
| --- | --- | --- | --- | --- |
| 1D VPE | 按展平后的单序号编码 | 弱 | 弱，更多依赖绝对顺序 | 容易退化 |
| 2D RoPE | 对 $(x, y)$ 两轴分别旋转 | 能区分行列 | 强，直接依赖坐标差 | 较好 |
| M-RoPE | 对 $(t, x, y)$ 三轴分别旋转 | 能区分时间、高、宽 | 强，直接依赖三轴差值 | 更适合视频与多模态 |

---

## 问题定义与边界

问题的本质很简单：图像和视频不是一维序列，而是二维或三维网格。图像 token 的自然位置是 $(x, y)$，视频 token 的自然位置是 $(t, x, y)$。如果位置表示错了，模型就会把空间关系和时间关系理解错。

对白话一点的理解是：把一张图片强行展平成一句话，模型只知道“哪个 patch 排在前面，哪个排在后面”，但它不知道两个 patch 是横向邻居还是纵向邻居。对于文本，这种顺序通常已经够用；但对图像定位、表格理解、UI grounding、视频时序分析，这远远不够。

设二维网格位置为：

$$
\text{position} = (x, y)
$$

如果把它展平为单个序号，例如宽度为 $W$ 时：

$$
p = x \cdot W + y
$$

那么 1D 编码感知的是 $p_m - p_n$，而不是分别感知 $(x_m - x_n)$ 与 $(y_m - y_n)$。这会带来一个直接问题：很多不同的二维关系会映射成相似的一维差值，模型难以稳定学习“上下”和“左右”是不同结构。

举一个新手很容易理解的例子。假设有两对 patch：

- A: $(0,0)$ 和 $(0,1)$，它们是左右相邻
- B: $(0,0)$ 和 $(1,0)$，它们是上下相邻

如果只看一维展平后的编号，这两种关系只是“差了一个或若干序号”；但对于视觉任务，这两种邻接关系经常意味着完全不同的语义。比如 OCR 中左右可能是同一行文本，垂直则可能是上下两行；UI 中左右可能是同一工具栏，垂直则可能是不同模块。

这里也要明确边界。2D RoPE 和 M-RoPE 解决的是“坐标表达”问题，不是“视觉理解”本身。它们不能替代 patch 划分、不能替代足够的数据训练，也不能自动解决遮挡、尺度变化、跨视角几何变换等更复杂的问题。它们只是让模型在注意力层里拥有更合理的位置几何。

真实工程里，这个问题在高分辨率场景尤其明显。比如一个网页截图被切成上千个视觉 token，如果仍然只用 1D 顺序编码，模型做坐标预测时容易依赖训练集里常见的绝对位置先验，而不是依赖真实几何关系。一旦截图分辨率变化、布局比例变化，这种先验就会崩掉。

---

## 核心机制与推导

RoPE 的基本思想是：把 embedding 的每一对维度看成一个二维平面，然后按位置对应的角度做旋转。所谓“每一对维度”，白话解释就是“把向量拆成很多个长度为 2 的小片段，每个小片段都能像平面坐标一样旋转”。

对一维位置 $p$，某一对维度上的旋转可以写成：

$$
R(\theta p)=
\begin{bmatrix}
\cos(\theta p) & -\sin(\theta p) \\
\sin(\theta p) & \cos(\theta p)
\end{bmatrix}
$$

如果把二维向量写成复数形式，旋转也可以写成：

$$
z \mapsto z \, e^{i\theta p}
$$

这里的复数形式只是记法更紧凑，白话解释就是“旋转角度用复指数表示”。

对于 1D RoPE，查询和键分别旋转后，点积只依赖位置差：

$$
\langle R(\theta p_m) q, R(\theta p_n) k \rangle
\propto \cos(\theta(p_m-p_n))
$$

这正是它适合文本的原因，因为文本天然是一维顺序。

2D RoPE 把位置改成两个轴，分别编码：

$$
\text{position}=(x,y)
$$

并让一部分维度负责 $x$ 轴，一部分维度负责 $y$ 轴。记作：

$$
z_x \mapsto z_x e^{i\theta_x x}, \qquad
z_y \mapsto z_y e^{i\theta_y y}
$$

于是旋转后的查询键点积会变成依赖两个相对差：

$$
\langle q_m, k_n \rangle_{\text{rope}}
\Rightarrow f(x_m-x_n,\, y_m-y_n)
$$

更具体地说，可以写成若干频率分量的和：

$$
\sum_j \Re\left(q_{x,j}\overline{k_{x,j}} e^{i\theta_{x,j}(x_m-x_n)}\right)
+
\sum_j \Re\left(q_{y,j}\overline{k_{y,j}} e^{i\theta_{y,j}(y_m-y_n)}\right)
$$

它的关键结论是：注意力不再依赖绝对编号，而主要依赖相对坐标差。

继续用前面的 $2 \times 2$ 玩具例子。设 $(0,0)$ 与 $(1,0)$ 比较，只看 $x$ 轴一个频率分量，并假设基向量已归一化，那么点积贡献是：

$$
\Re\left(e^{i\theta_x \cdot 0} e^{-i\theta_x \cdot 1}\right)
= \cos(\theta_x)
$$

若 $\theta_x=\pi/8$，则得到：

$$
\cos(\pi/8)
$$

这说明相差一个行单位时，注意力的变化由固定角频率决定。对应列方向同理。

M-RoPE 再往前一步，把视频时间轴加入：

$$
\text{position}=(t,x,y)
$$

即：

$$
z_t \mapsto z_t e^{i\theta_t t}, \quad
z_x \mapsto z_x e^{i\theta_x x}, \quad
z_y \mapsto z_y e^{i\theta_y y}
$$

于是点积依赖三轴相对差：

$$
\langle q_m, k_n \rangle_{\text{m-rope}}
\Rightarrow g(t_m-t_n,\, x_m-x_n,\, y_m-y_n)
$$

这个形式的重要意义是：同一空间位置但不同时间帧的 token，会因为 $t_m-t_n$ 不同而被分开；同一时间点但不同空间位置的 token，也会因为 $(x,y)$ 差异而被分开。视频理解、时序 grounding、本体跟踪都依赖这个能力。

一个真实工程例子是视频问答。假设模型需要回答“第 3 秒按钮从左上移动到了哪里”。如果时间编码只是简单帧号，那么 15fps 和 30fps 的视频会对同一真实时间产生不同编号，模型会混淆“同样 3 秒”与“不同采样密度”。M-RoPE 如果按真实秒数编码，就能让时间位置跨帧率对齐。

---

## 代码实现

下面给一个可运行的简化版 `python` 实现，展示 2D RoPE 如何对行列分别编码。这个实现不是完整训练代码，但足够说明核心机制，并且能验证“注意力只依赖相对坐标差”。

```python
import numpy as np

def rotate_pairs(x, angles):
    # x: [N, D], D 必须为偶数
    # angles: [N, D/2]
    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    c = np.cos(angles)
    s = np.sin(angles)
    out_even = x_even * c - x_odd * s
    out_odd = x_even * s + x_odd * c

    out = np.empty_like(x)
    out[:, 0::2] = out_even
    out[:, 1::2] = out_odd
    return out

def apply_rope_2d(x, row_ids, col_ids, theta_row, theta_col):
    # x: [N, D]
    # 前一半维度给 row，后一半维度给 col
    n, d = x.shape
    assert d % 4 == 0

    half = d // 2
    row_part = x[:, :half]
    col_part = x[:, half:]

    row_pairs = half // 2
    col_pairs = half // 2

    row_angles = np.outer(row_ids, theta_row[:row_pairs])
    col_angles = np.outer(col_ids, theta_col[:col_pairs])

    row_rot = rotate_pairs(row_part, row_angles)
    col_rot = rotate_pairs(col_part, col_angles)
    return np.concatenate([row_rot, col_rot], axis=1)

# 2x2 patch 网格
coords = np.array([
    [0, 0],  # (row, col)
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=np.float64)

row_ids = coords[:, 0]
col_ids = coords[:, 1]

# 构造一个简单 embedding
x = np.array([
    [1, 0,  1, 0,  1, 0,  1, 0],
    [1, 0,  1, 0,  1, 0,  1, 0],
    [1, 0,  1, 0,  1, 0,  1, 0],
    [1, 0,  1, 0,  1, 0,  1, 0],
], dtype=np.float64)

theta_row = np.array([np.pi / 8, np.pi / 16])
theta_col = np.array([np.pi / 8, np.pi / 16])

y = apply_rope_2d(x, row_ids, col_ids, theta_row, theta_col)

# 比较 (0,0) 与 (1,0): 行差为 1，列差为 0
dot_00_10 = np.dot(y[0], y[2])

# 比较 (0,0) 与 (0,1): 行差为 0，列差为 1
dot_00_01 = np.dot(y[0], y[1])

assert y.shape == x.shape
assert dot_00_10 < np.dot(y[0], y[0])  # 相对位置差导致点积下降
assert dot_00_01 < np.dot(y[0], y[0])
```

如果放到 ViT 这类视觉 Transformer 中，实际实现通常不是直接“给 embedding 加两个向量”，而是在注意力里的 `q`、`k` 上做旋转。常见流程是：

1. 先根据 patch 网格生成 `row_id` 和 `col_id`
2. 依据频率表生成 `row_angles`、`col_angles`
3. 对 `q`、`k` 的不同维度分块旋转
4. 再进入 `q @ k^T`

伪代码可以写成：

```python
def rope_2d_for_attention(q, k, row_ids, col_ids, inv_freq_row, inv_freq_col):
    row_angles = row_ids[:, None] * inv_freq_row[None, :]
    col_angles = col_ids[:, None] * inv_freq_col[None, :]

    q = apply_rope_2d(q, row_ids, col_ids, inv_freq_row, inv_freq_col)
    k = apply_rope_2d(k, row_ids, col_ids, inv_freq_row, inv_freq_col)

    scores = np.einsum("id,jd->ij", q, k)
    return scores
```

真实工程例子是多模态视频模型。假设输入视频先按帧切块，再把每帧切成 patch，那么每个 token 都带有三元组 `(frame_time, row, col)`。如果视频帧率不固定，时间轴不应简单用 `frame_idx`，而应使用真实秒数：

$$
\text{seconds} = \frac{\text{frame\_idx}}{\text{fps}}
$$

此时时间角度为：

$$
\phi_t = \theta_t \cdot \text{seconds}
$$

这样 15fps 和 30fps 的视频，即使帧编号不同，只要物理时间相同，时间位置编码就能对齐。这正是视频场景比图像场景更容易踩坑的地方。

---

## 工程权衡与常见坑

2D RoPE 和 M-RoPE 的理论结构很干净，但工程上有几个问题必须提前处理。

第一类问题是频率分配。频率就是不同维度使用的旋转快慢，白话解释就是“有的维度负责粗粒度位置，有的维度负责细粒度位置”。如果高频全部塞到某一个轴，比如全部给高度轴 $y$，那么宽度轴 $x$ 的分辨能力会明显不足。结果是模型对纵向细节很敏感，但对横向定位发虚。

第二类问题是时间轴对齐。M-RoPE 的时间 ID 如果直接取帧号，而不是取真实秒数，那么不同帧率的视频会产生错位。训练集中 30fps 为主时，模型会把“第 90 帧”当成“第 3 秒”；但在 15fps 视频里，第 90 帧其实是第 6 秒。这会直接污染时间理解。

第三类问题是把 VPE 当成足够强的视觉位置方案。VPE 是 Visual Positional Embedding，白话解释就是“给视觉 token 一个位置向量，通常按序号查表”。这在低分辨率分类任务可能还能工作，但在需要坐标生成、细粒度布局理解、跨分辨率泛化的任务里，往往会暴露明显偏置。

下面用表格归纳常见坑和处理方式。

| 常见问题 | 现象 | 原因 | 缓解策略 |
| --- | --- | --- | --- |
| 频谱不均衡 | 一条轴定位准，另一条轴定位差 | 高频集中在单轴 | 交错分配频率，或对多轴做混合调度 |
| 高分辨率退化 | 坐标输出偏向少数常见位置 | 1D VPE 学到数据先验而非几何关系 | 改用 2D RoPE，并做跨分辨率验证 |
| 帧率不一致 | 同一秒的动作对不齐 | 时间 ID 用帧号而非秒数 | 用 `frame_idx / fps` 或等价真实时间 |
| 训练正常、推理失真 | 新布局下 attention 异常 | 位置编码依赖训练分布 | 加强分辨率扰动、布局扰动测试 |
| 空间轴含义错位 | 行列效果颠倒 | `row_id` / `col_id` 生成顺序写错 | 固定坐标约定，并写单元测试 |

一个典型坑例是 GUI grounding。模型需要输出“按钮位于屏幕右上区域”。如果你把高频都给了纵轴，模型可能能学会顶部/底部，但对左/右的判定发虚，最终输出的横坐标会向中间数字收缩。这不是语言能力问题，而是位置频谱已经退化。

还有一个容易被忽略的工程点：窗口注意力与全局位置编码的配合。很多视觉模型为了省算力，会在局部窗口里做注意力。如果位置编码只在窗口内重新编号，而没有保留全局坐标语义，那么跨窗口关系仍然会丢。实际系统通常需要同时考虑局部计算效率和全局位置一致性。

---

## 替代方案与适用边界

不是所有任务都必须上 2D RoPE 或 M-RoPE。位置编码的选择，取决于任务是否真的需要精细坐标关系。

最简单的替代方案是 1D VPE 或 learned positional embedding。learned embedding 的意思是“位置向量直接作为参数学习出来”，白话解释就是“模型自己背下每个位置该长什么样”。它在低分辨率、固定输入尺寸、只做分类时通常足够。比如 64×64 图像分类，模型只需要知道整体类别，不需要输出精确框坐标，这时 1D VPE 往往就能满足需求。

一个低分辨率场景的例子是猫狗分类。输入图像很小，且任务目标只是输出“猫”或“狗”。模型不需要回答“耳朵在第几行第几列”，也不需要跨分辨率做定位迁移。此时简单位置编码就够，继续上更复杂的 2D/M-RoPE，收益可能不明显。

但如果任务是文档理解、屏幕理解、检测式问答、长视频时序检索，那么适用边界就变了。因为这些任务不是只看“有没有”，而是要看“在哪里”和“什么时候”。

另一类替代思路是频率调度优化，而不是放弃 RoPE。本质上仍然用 2D RoPE/M-RoPE，但对不同轴采用不同 spacing、不同混合策略，让频谱分布更均衡。这种方法常见于 GUI grounding、坐标生成类任务，因为它们对空间敏感度特别高。

下面给出一个简化对比。

| 方法 | 参数/实现复杂度 | 坐标精度 | 分辨率泛化 | 适用场景 |
| --- | --- | --- | --- | --- |
| 1D VPE | 低 | 低 | 弱 | 小图分类、粗粒度理解 |
| Learned Positional Embedding | 中 | 中 | 通常弱 | 固定尺寸输入、训练分布稳定 |
| 2D RoPE | 中 | 高 | 较强 | 图像理解、文档、UI、定位任务 |
| M-RoPE | 较高 | 高 | 对视频更强 | 视频问答、时序 grounding、多模态统一建模 |

因此可以用一句话概括边界：如果任务只关心“内容是什么”，简单位置编码可能够用；如果任务关心“内容在哪里、何时出现、跨尺寸是否稳定”，2D RoPE 或 M-RoPE 更合适。

---

## 参考资料

| 资料 | 内容侧重点 | 链接 |
| --- | --- | --- |
| EmergentMind 2D Rotary Position Embedding Overview | 2D RoPE 的数学形式、相对位置性质、二维扩展思路 | https://www.emergentmind.com/topics/2d-rotary-position-embedding-rope |
| Qwen2.5-VL Blog | M-RoPE 在图像/视频中的工程实现，时间轴按真实秒对齐 | https://qwenlm.github.io/blog/qwen2.5-vl/ |
| Mitigating Coordinate Prediction Bias | 分析 1D VPE 在坐标预测中的偏置与退化现象 | https://chatpaper.com/paper/203898 |
| GUI grounding 相关工作 | 关注多轴频谱分配不均衡、分辨率变化下的位置泛化 | 可沿相关论文引用继续追踪 |

1. EmergentMind 的 2D RoPE 综述可以作为核心理论入口，因为它把“旋转后点积只依赖相对坐标差”讲得最清楚。
2. Qwen2.5-VL 官方博客更适合理解 M-RoPE 的工程细节，尤其是时间 ID 与真实秒数对齐这一点。
3. 坐标预测偏差相关分析适合解释为什么很多视觉任务里，1D VPE 会在高分辨率或新布局上突然失效。
