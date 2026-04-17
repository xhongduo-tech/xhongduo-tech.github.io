## 核心结论

Gemini 的视觉编码器可以理解成“为多模态大模型定制的 ViT 路线”。ViT 是 Vision Transformer，白话说就是“把图像切成很多小块，再像处理词一样处理这些图像块”。它的关键不是把图像单独编码完再交给语言模型，而是尽早把视觉信息整理成一串可与文本 token 对齐的统一表征。

公开资料没有完整披露 Gemini 视觉塔的全部结构细节，但从 Gemini 的分辨率控制接口、视觉 token 预算设计，以及同类前沿视觉语言模型的公开技术路径看，一个稳定的工程范式已经很清楚：先按 patch 切图，再用带二维位置信息的 Transformer 编码，在高分辨率下用局部注意力控制复杂度，最后把视觉 token 压缩并投影到与语言模型一致的隐藏维度，让视觉序列和文本序列进入同一推理堆栈。

这条路线的核心价值有三点：

| 模块 | 作用 | 直接收益 |
| --- | --- | --- |
| Patch embedding | 把像素块变成向量 | 图像可被 Transformer 处理 |
| 2D 位置编码 | 让模型知道 token 在第几行第几列 | 保住空间关系与坐标感 |
| Token 压缩与对齐 | 把视觉序列缩短并投影到文本维度 | 视觉和语言可直接融合 |

以 $896 \times 896$ 图像为例，若 patch 大小为 $14 \times 14$，则一共得到
$$
N=\frac{896}{14}\times\frac{896}{14}=64\times 64=4096
$$
个 patch token。若再做一次 $2\times 2$ 聚合，token 数会降到 $1024$。这时视觉序列长度和后续语言堆栈更容易匹配，计算成本也明显下降。

---

## 问题定义与边界

问题不是“怎么把图片喂给模型”，而是“怎么把任意分辨率图像稳定地变成一串既保留空间结构、又不会把算力打爆的 token”。

这里有三个边界必须先说清楚。

第一，输入是任意 $H\times W$ 图像，不是固定的 $224\times224$。这意味着模型不能只靠固定分辨率训练时学到的位置模式，而要在原生分辨率上处理 patch 序列。原生分辨率，白话说就是“不先粗暴缩到统一尺寸，而是尽量保留原图尺度”。

第二，patch 数会随分辨率上升而快速增长。若 patch 大小是 $p=14$，那么 patch 数近似为
$$
N=\left\lfloor \frac{H}{14} \right\rfloor \cdot \left\lfloor \frac{W}{14} \right\rfloor
$$
当 $H,W$ 变大时，$N$ 很快就会到几千甚至上万。

第三，标准全局注意力的复杂度是 $O(N^2)$。注意力，白话说就是“每个 token 都去看其他 token”。如果 4096 个 token 两两计算，成本会非常高，因此高分辨率视觉编码不能长期依赖纯全局注意力。

一个玩具例子可以直接看出这个边界。

假设有一张 $28\times28$ 的灰度图，patch 大小为 $14$。那么图像被切成 4 个 patch，它们分别对应左上、右上、左下、右下四个位置。此时全局注意力完全没问题，因为 token 太少了。

但如果换成 $896\times896$，patch 数变成 4096。若每层都做全局注意力，成本会从“4 个点互相看”变成“4096 个点互相看”，这已经不是同一个工程量级。

所以视觉编码器的边界可以概括成一句话：必须同时解决“空间保真”和“复杂度可控”两个目标，缺一不可。

---

## 核心机制与推导

这类视觉编码器通常按四步工作：patchify、位置编码、注意力编码、token 压缩。

### 1. Patch embedding：从像素到 token

Patch embedding 的意思是“把每个小图块拉直，再映射成固定维度向量”。例如一个 RGB patch 大小为 $14\times14$，它的原始维度就是 $14\times14\times3=588$。经过线性层后，可以变成维度为 $d$ 的向量，比如 1024 或 1280。

如果输入是 $H\times W$，则 patch 网格大小近似是：
$$
h=\left\lfloor \frac{H}{14} \right\rfloor,\quad
w=\left\lfloor \frac{W}{14} \right\rfloor
$$
总 token 数是 $N=h\cdot w$。

### 2. 2D RoPE：给 token 注入二维位置

RoPE 是 Rotary Position Embedding，白话说就是“用旋转角度把位置信息写进向量里”。普通一维 RoPE 适合文本，因为文本是单轴序列；图像是二维网格，所以需要把“行位置”和“列位置”分别编码。

可把一个 token 的查询向量 $q$ 划分成两部分，一部分承载行坐标，一部分承载列坐标。形式上可写成：
$$
q^{(row)}_i = R(\theta_{r_i}) q_i^{(row)}, \quad
q^{(col)}_i = R(\theta_{c_i}) q_i^{(col)}
$$
其中 $r_i, c_i$ 分别是第 $i$ 个 patch 的行列坐标，$R(\theta)$ 是二维旋转矩阵：
$$
R(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}
$$

这样做的作用是：相邻 patch 在注意力里不仅“内容接近”，而且“位置关系也可比较”。这对文档理解尤其重要，因为表格单元格的相对位置本身就是语义的一部分。

### 3. 局部窗口注意力 + 少量全局层

窗口注意力，白话说就是“多数时候只看附近邻居，不看整张图”。如果窗口边长是 $8$，那么每个 token 主要在本地 $8\times8$ 范围内计算。

此时单层复杂度近似为：
$$
O(N\cdot w^2 \cdot d)
$$
其中这里的 $w^2$ 指窗口内 token 数，不再是全局的 $N$。

如果总共有 32 层，其中 28 层是窗口注意力，4 层是全局注意力，那么总预算可以近似写成：
$$
\text{Cost} \approx \frac{28}{32}O(N\cdot 8^2\cdot d)+\frac{4}{32}O(N^2\cdot d)
$$
这不是严格数学证明，而是工程级的复杂度估算。结论很直接：绝大多数层保持近线性扩展，少数全局层负责跨区域信息汇总。

### 4. 2×2 聚合：从视觉序列走向统一表征

统一表征的意思是“视觉 token 和文本 token 使用同一类隐藏空间表示”。常见做法是把空间上相邻的 $2\times2$ patch 合并，再经过 MLP 投影到语言模型维度。

如果原始视觉 token 数是 $N$，做一次 $2\times2$ 聚合后大约变成：
$$
N'=\frac{N}{4}
$$

仍以 $896\times896$ 为例：

| 图像尺寸 | patch 大小 | 初始 token 数 | 聚合方式 | 聚合后 token 数 |
| --- | --- | --- | --- | --- |
| 896×896 | 14×14 | 4096 | 2×2 | 1024 |
| 1792×896 | 14×14 | 8192 | 2×2 | 2048 |

这就是“从 ViT 到统一表征”的关键一步。ViT 负责把图像编码成高质量视觉特征，统一表征负责把这些特征整理成语言模型愿意接收的 token 序列。

一个真实工程例子是复杂表格解析。表头、单元格边界、脚注位置往往非常细，若先缩图，细节会直接消失；若全程不压缩 token，成本又太高。于是系统通常先在较高分辨率下保住局部结构，再在进入语言堆栈前做 token 聚合，这是目前较稳的平衡点。

---

## 代码实现

下面用一个可运行的 Python 玩具实现，演示三个事实：

1. patch 数如何随分辨率变化；
2. $2\times2$ 聚合如何把 token 数缩成四分之一；
3. 这种压缩与统一表征的工程目标一致。

```python
from math import floor

def patch_grid(height: int, width: int, patch: int = 14):
    return floor(height / patch), floor(width / patch)

def patch_count(height: int, width: int, patch: int = 14):
    gh, gw = patch_grid(height, width, patch)
    return gh * gw

def compress_2x2_tokens(token_count: int, grid_h: int, grid_w: int):
    assert grid_h % 2 == 0 and grid_w % 2 == 0
    assert token_count == grid_h * grid_w
    return (grid_h // 2) * (grid_w // 2)

# 玩具例子：28x28 图像
gh, gw = patch_grid(28, 28, 14)
n = patch_count(28, 28, 14)
assert (gh, gw) == (2, 2)
assert n == 4
assert compress_2x2_tokens(n, gh, gw) == 1

# 真实工程尺度例子：896x896 图像
gh, gw = patch_grid(896, 896, 14)
n = patch_count(896, 896, 14)
assert (gh, gw) == (64, 64)
assert n == 4096
assert compress_2x2_tokens(n, gh, gw) == 1024

# 横向长图：1792x896
gh, gw = patch_grid(1792, 896, 14)
n = patch_count(1792, 896, 14)
assert (gh, gw) == (128, 64)
assert n == 8192
assert compress_2x2_tokens(n, gh, gw) == 2048

print("all assertions passed")
```

如果把这段逻辑翻译成视觉编码主流程，结构大致如下：

```python
def vision_encoder(image):
    patches = image_to_patches(image, patch_size=14)
    x = linear_patch_embedding(patches)
    x = apply_2d_rope(x)              # 给每个 token 注入行列位置信息

    for layer_id in range(32):
        if layer_id in {7, 15, 23, 31}:
            x = global_attention(x)
        else:
            x = window_attention(x, window_size=8)

    x = fuse_2x2_neighbors_with_mlp(x)
    x = project_to_llm_hidden_size(x)
    return x
```

这里最重要的不是函数名，而是数据形态变化：

`像素 -> patch 向量 -> 带二维位置的视觉 token -> 压缩后的视觉序列 -> 与文本维度一致的统一 token`

这也是为什么很多多模态系统不再把视觉塔看成“独立前端”，而是看成语言模型的一个序列生成器。

---

## 工程权衡与常见坑

第一类权衡是分辨率与成本。分辨率越高，patch 越多，细节越完整，但 token 预算、显存占用和延迟都会上升。对于复杂表格、长文档、UI 截图，高分辨率通常值得；对于普通自然图像分类或粗粒度问答，过高分辨率常常是浪费。

第二类权衡是局部感受野与全局建模。窗口注意力能省成本，但如果全局层太少，模型会丢掉远距离依赖，例如表头与跨页脚注、图例与主图之间的关联。全局层不是越多越好，而是要用来“打通局部块之间的信息孤岛”。

第三类权衡是 token 压缩时机。压缩太早，细节还没抽取出来就被合并；压缩太晚，序列已经太长，成本很难控制。实践中常见策略是先让若干层在高分辨率空间里提取局部模式，再做邻域聚合。

常见坑主要有四个：

| 常见坑 | 本质问题 | 后果 |
| --- | --- | --- |
| 先统一缩到小尺寸 | 原始坐标被破坏 | OCR、检测、表格理解下降 |
| 全程全局注意力 | 复杂度 $O(N^2)$ 过高 | 延迟和显存失控 |
| 过早压缩 token | 细节尚未编码完成 | 小字、边框、局部纹理丢失 |
| 视觉维度与文本维度不对齐 | 融合接口不稳定 | 跨模态对齐成本上升 |

真实工程里，Gemini 的 `media_resolution` 就是在暴露这种权衡。它不是一个“画质开关”，而是一个“视觉 token 预算开关”。复杂表格、合同扫描件、图中嵌字界面，通常应选更高分辨率；普通照片问答、粗粒度场景理解，可以选较低设置，换取更低延迟和更稳定的成本。

---

## 替代方案与适用边界

第一种替代方案是经典固定分辨率 ViT。它适合输入尺寸稳定、任务目标单一的场景，例如固定大小分类、轻量级特征提取。优点是实现简单，训练与推理路径稳定；缺点是对大图和文档不友好，因为缩放会直接破坏像素级空间关系。

第二种替代方案是纯全局注意力视觉 Transformer。它在中低分辨率上表达能力强，也容易统一建模，但当 token 数进入几千量级时，复杂度问题会迅速暴露。

第三种替代方案是 CNN 或 CNN+Transformer 混合结构。CNN 是卷积神经网络，白话说就是“用局部卷积核逐层提取图像模式”。它在局部纹理抽取上很强，但要把输出自然接入大语言模型的 token 流，通常还需要额外的序列化与投影设计。

不同方案的边界可以直接比较：

| 方案 | 分辨率适应性 | 坐标保真 | 高分辨率成本 | 适合任务 |
| --- | --- | --- | --- | --- |
| 固定分辨率 ViT | 弱 | 中 | 中 | 分类、检索、标准图像任务 |
| 全局注意力 ViT | 中 | 高 | 高 | 中等分辨率跨区域理解 |
| 动态分辨率 ViT + 压缩 | 强 | 高 | 可控 | 文档、表格、UI、视频、多模态对话 |
| CNN 主导结构 | 中 | 中 | 低到中 | 检测、分割、边缘设备 |

所以“Gemini 的视觉编码器为什么重要”，答案不是它用了 ViT，而是它把 ViT 改造成了适合多模态统一推理的形态：输入分辨率可变、空间关系可保留、token 数可压缩、输出表示可直接接入语言模型。

如果任务只是 $224\times224$ 的固定分类，经典 ViT 已经足够；如果任务是长文档问答、复杂表格解析、跨图文推理，那么动态分辨率与统一表征就不是锦上添花，而是必要条件。

---

## 参考资料

- Qwen2.5-VL 技术报告（动态分辨率 ViT、14×14 patch、2D/3D RoPE、MLP 压缩）：https://www.52nlp.cn/wp-content/uploads/2025/02/Qwen2.5-VL%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A.pdf
- Qwen2.5-VL Backbone: Dynamic-Resolution ViT：https://www.emergentmind.com/topics/qwen2-5-vl-backbone
- Qwen2.5-VL: Advanced Vision-Language Model：https://www.emergentmind.com/topics/qwen2-5-vl-model
- Gemini 开发文档，视觉输入与 `media_resolution` 控制：https://ai.google.dev/
- Google Developer Blog，Gemini 视觉能力与工程使用建议：https://blog.google/technology/developers/
