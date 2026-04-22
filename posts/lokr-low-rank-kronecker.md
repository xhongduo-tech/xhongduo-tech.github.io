## 核心结论

LoKr（Low-Rank Kronecker Product，低秩克罗内克积适应）是一种参数高效微调方法。参数高效微调的意思是：不重新训练模型里所有权重，只额外训练少量适配参数，让预训练模型适应新任务。

LoKr 的核心公式是：

$$
W = W_0 + \Delta W,\quad \Delta W = \gamma \cdot (A \otimes B)
$$

其中 \(W_0\) 是原始预训练权重，\(\Delta W\) 是微调时学到的权重更新，\(\gamma\) 是缩放系数，\(A \otimes B\) 是克罗内克积。克罗内克积的白话解释是：用一个小矩阵里的每个数去放大另一个小矩阵，再按块拼成一个更大的矩阵。

LoKr 不直接训练完整大矩阵，而是训练两个更小的矩阵 \(A\) 和 \(B\)，再通过克罗内克积生成大矩阵更新量。全量微调像是重画整张地图，LoKr 像是先画两张小模板，再按固定规则拼成一张大更新图。这个比喻只用于建立直觉，真正机制仍然是矩阵分解和结构化参数共享。

| 方法 | 训练对象 | 参数量 | 结构约束 | 典型用途 |
|---|---:|---:|---|---|
| 全量微调 | 原始权重 \(W\) | 最高 | 几乎无 | 数据充足、资源充足、任务变化大 |
| LoRA | 低秩矩阵 \(A,B\) | 低 | 低秩约束 | 快速适配语言模型或扩散模型 |
| LoKr | 克罗内克因子 \(A,B\) | 低到中 | 块结构约束 | 需要更强结构表达的垂直领域适配 |

核心结论是：LoKr 不是“更省参数的万能 LoRA”，而是“用克罗内克积表达结构化更新”的适配方法。它适合大型模型中形状规则、结构信息明显的层，尤其常见于 Stable Diffusion / SDXL 等文本到图像模型的风格、角色、领域微调。

---

## 问题定义与边界

LoKr 解决的问题是：大型模型微调成本太高。一个线性层如果权重形状是 \(4096 \times 4096\)，完整权重有：

$$
4096 \times 4096 = 16,777,216
$$

也就是 1600 多万个参数。全量微调要对这些参数计算梯度、保存优化器状态，并在每个任务上保存一份完整或接近完整的权重变化。对零基础到初级工程师来说，可以先记住一句话：参数越多，训练显存、存储和调参成本通常越高。

LoKr 的输入、输出和约束可以这样定义：

| 项目 | 含义 |
|---|---|
| 输入 | 预训练权重 \(W_0\)，例如某个线性层或卷积层的权重 |
| 输出 | 更新后的权重 \(W = W_0 + \Delta W\) |
| 训练参数 | 通常只训练 \(A,B\)，有些实现还会引入额外分解项或控制项 |
| 约束 | \(\Delta W\) 必须能由克罗内克结构生成，不是完全自由矩阵 |
| 目标 | 降低训练参数量、显存占用和适配器文件大小 |

LoKr 的边界同样重要。它不是通用压缩算法，也不是把任何层套进去都能变好。它的优势来自结构化假设：大更新矩阵可以由小矩阵按块组合出来。如果某个任务需要非常自由、非常细粒度的权重变化，这种结构约束可能会限制表达能力。

玩具例子：一个 \(4 \times 4\) 更新矩阵有 16 个数。LoKr 可以用两个 \(2 \times 2\) 矩阵生成它，只训练 8 个数。这个例子说明了参数节省，但也说明了限制：生成出来的 16 个数不是彼此独立的，而是受 \(A\) 和 \(B\) 的块状规则控制。

真实工程例子：在 SDXL 上做某个产品图风格微调时，团队可能只想让模型学会固定材质、光照和构图倾向，而不想完整重训 UNet 和 text encoder。此时可以把 LoKr 挂到 attention 的 `q_proj/k_proj/v_proj/out_proj`，以及部分卷积模块上。目标不是让模型学会一切，而是在小显存和小适配器文件下完成垂直领域适配。

适合 LoKr 的情况包括：大模型微调、显存敏感、需要保存多个领域适配器、模型层形状规则、希望保留一定块结构信息。不适合的情况包括：极小模型、层形状不规则、任务必须完全自由更新、数据量太小但目标变化很大。

---

## 核心机制与推导

LoKr 的基础来自克罗内克积。设：

$$
A \in \mathbb{R}^{m_1 \times n_1},\quad B \in \mathbb{R}^{m_2 \times n_2}
$$

那么：

$$
A \otimes B \in \mathbb{R}^{m_1m_2 \times n_1n_2}
$$

也就是说，两个小矩阵可以组合成一个更大的矩阵。它不是普通矩阵乘法。普通矩阵乘法会把两个矩阵相乘后压成一个结果；克罗内克积会把 \(A\) 中每个元素变成一个由 \(B\) 放大得到的块。

主公式是：

$$
\Delta W = \gamma \cdot (A \otimes B)
$$

实际实现中还可能使用扩展形式：

$$
\Delta W = \gamma \cdot (A \otimes B) \odot C
$$

其中 \(\odot\) 表示逐元素乘。逐元素乘的白话解释是：两个形状相同的矩阵，对应位置一一相乘。这里的 \(C\) 可以理解为额外控制项，用来提高更新的灵活性。不同库的具体实现会有差异，所以工程上应以当前使用库的文档和源码为准。

看一个最小数值例子。设：

$$
A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
B=\begin{bmatrix}0.5&1\\1.5&2\end{bmatrix},\quad
\gamma=0.1
$$

则：

$$
A\otimes B=
\begin{bmatrix}
0.5&1&1&2\\
1.5&2&3&4\\
1.5&3&2&4\\
4.5&6&6&8
\end{bmatrix}
$$

所以：

$$
\Delta W=0.1(A\otimes B)=
\begin{bmatrix}
0.05&0.10&0.10&0.20\\
0.15&0.20&0.30&0.40\\
0.15&0.30&0.20&0.40\\
0.45&0.60&0.60&0.80
\end{bmatrix}
$$

如果 \(W_0=0\)，那么 \(W=\Delta W\)。这里完整 \(4\times4\) 更新矩阵需要 16 个参数，而 LoKr 只训练 \(A\) 和 \(B\) 的 8 个参数。LoKr 学到的不是整张更新表，而是两张小表拼出来的更新规则。

参数量可以按下面的方式理解：

| 更新方式 | 目标矩阵形状 | 可训练参数量 | 参数节省比例 |
|---|---:|---:|---:|
| 完整矩阵 | \(4 \times 4\) | 16 | 0% |
| LoKr：\(2\times2\) 与 \(2\times2\) | \(4 \times 4\) | 8 | 50% |
| 完整矩阵 | \(4096 \times 4096\) | 16,777,216 | 0% |
| LoKr 示例分解 | 依赖具体 factor/rank | 远低于完整矩阵 | 依赖配置 |

这里不能简单说 LoKr 一定比 LoRA 更强。LoRA 的核心是低秩加法更新，LoKr 的核心是克罗内克结构更新。LoRA 更像单一方向上的简洁改写，LoKr 更像带结构的块状改写。两者的差异不是“谁永远更好”，而是“表达假设不同”。

---

## 代码实现

下面先用一个可运行的 Python 玩具实现说明克罗内克积如何生成更新矩阵。这个代码不依赖深度学习框架，只用于验证机制。

```python
import numpy as np

def lokr_delta(A, B, gamma=1.0):
    return gamma * np.kron(A, B)

A = np.array([[1, 2],
              [3, 4]], dtype=float)

B = np.array([[0.5, 1.0],
              [1.5, 2.0]], dtype=float)

delta = lokr_delta(A, B, gamma=0.1)

expected = np.array([
    [0.05, 0.10, 0.10, 0.20],
    [0.15, 0.20, 0.30, 0.40],
    [0.15, 0.30, 0.20, 0.40],
    [0.45, 0.60, 0.60, 0.80],
])

assert delta.shape == (4, 4)
assert np.allclose(delta, expected)
assert A.size + B.size == 8
assert delta.size == 16
```

工程里通常不会自己手写矩阵展开再塞回模型，而是使用 PEFT 或 LyCORIS 这类现成实现。原因很直接：真实模型里有线性层、卷积层、权重合并、保存加载、混合精度、分布式训练等问题，手搓实现容易在边界上出错。

一个简化版伪代码如下：

```python
# 伪代码：表达训练流程，不代表某个库的完整 API

base_model = load_pretrained_model("some-large-model")

freeze(base_model)  # 冻结原始权重 W0

target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
]

lokr_config = {
    "r": 8,
    "alpha": 8,
    "target_modules": target_modules,
    "decompose_factor": -1,
    "use_effective_conv2d": False,
}

model = attach_lokr_adapter(base_model, lokr_config)

for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

save_adapter(model, "domain-lokr-adapter")
```

使用 PEFT 时，关键不是记住每个参数名，而是理解配置在控制什么。`target_modules` 决定替换哪些层，`r` 控制 LoKr rank，`alpha` 控制缩放，`use_effective_conv2d` 与卷积层处理有关。不同版本可能存在参数名变化，例如旧示例里可能出现 `lora_alpha`，而当前文档中 LoKrConfig 使用的是 `alpha`。工程上应检查你安装版本对应的文档，而不是复制任意博客代码。

一个更接近真实场景的配置片段如下：

```python
from peft import LoKrConfig, LoKrModel

text_encoder_config = LoKrConfig(
    r=8,
    alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
)

unet_config = LoKrConfig(
    r=8,
    alpha=8,
    target_modules=[
        "proj_in",
        "proj_out",
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
    use_effective_conv2d=True,
)

# pipeline = StableDiffusionPipeline.from_pretrained(...)
# pipeline.text_encoder = LoKrModel(pipeline.text_encoder, text_encoder_config, "default")
# pipeline.unet = LoKrModel(pipeline.unet, unet_config, "default")
```

线性层和卷积层不能完全按同一种方式处理。线性层权重天然是二维矩阵，比较适合直接讨论 \(W \in \mathbb{R}^{out \times in}\)。卷积层权重通常是四维张量，例如：

| 层类型 | 常见权重形状 | LoKr 使用注意点 |
|---|---|---|
| 线性层 | `[out_features, in_features]` | 适合按矩阵分解理解 |
| 1D 卷积 | `[out_channels, in_channels, kernel]` | 需要实现处理核维度 |
| 2D 卷积 | `[out_channels, in_channels, kernel_h, kernel_w]` | 常需使用库提供的有效卷积分解选项 |

推理时有两种常见方式：一种是保持 adapter 形式，加载基础模型后再加载 LoKr adapter；另一种是把 LoKr 更新合并回主权重。前者便于切换多个风格或任务，后者便于部署，但合并后要注意精度和版本兼容。

---

## 工程权衡与常见坑

LoKr 的效果强依赖结构假设。参数不是越多越好，也不是越少越好。参数过少时表达能力不足，参数过多时训练成本上升，还可能接近全量微调却没有全量微调的自由度。真正要调的是：目标层是否合适、分解形状是否合理、rank/factor 是否匹配任务难度。

常见坑如下：

| 常见坑 | 表现 | 原因 | 规避方式 |
|---|---|---|---|
| factor 或 rank 过小 | 训练 loss 降不动，效果弱 | 更新空间太窄 | 从库默认值或小配置起步，再逐步增大 |
| factor 或 rank 过大 | 显存上升，过拟合，适配器变大 | 参数量增加且约束仍存在 | 比较验证集，不只看训练集 |
| 所有层都强行套 LoKr | 训练不稳定，收益不明显 | 不同层对任务贡献不同 | 先选 attention 和关键卷积层 |
| 卷积层按线性层粗暴处理 | 形状错误或效果异常 | 卷积核维度需要特殊处理 | 使用库里的卷积相关选项 |
| 混淆库参数名 | 代码报错或配置无效 | PEFT、LyCORIS、不同版本 API 不同 | 以当前安装版本文档为准 |
| 只看训练损失 | loss 下降但生成质量变差 | loss 与真实体验不完全一致 | 固定 prompt 集做回归测试 |
| 维度划分不合理 | 参数不少但表达差 | 克罗内克块结构不匹配层形状 | 检查目标层尺寸和分解因子 |

新手版错误案例：把模型里所有线性层和卷积层都套 LoKr，直接开较大的 rank 训练。结果训练损失下降很快，但生成图像出现风格污染，原模型能力下降，适配器文件也明显变大。更合理的做法是：先只选 attention 里的 `q_proj/k_proj/v_proj/out_proj`，用较小配置跑一个短实验，再根据固定验证 prompt 的结果决定是否加入卷积层或前馈层。

排查流程可以按下面顺序做：

| 步骤 | 检查项 | 判断标准 |
|---|---|---|
| 1 | 目标层类型 | 是否真的是线性层或库支持的卷积层 |
| 2 | 目标层名称 | `target_modules` 是否匹配实际模块名 |
| 3 | 分解配置 | rank/factor 是否导致参数量异常 |
| 4 | 训练稳定性 | loss 是否爆炸、梯度是否异常 |
| 5 | 验证质量 | 固定样例输出是否变好 |
| 6 | 部署方式 | adapter 加载或合并后结果是否一致 |

真实工程中，LoKr 常用于多个领域 adapter 的管理。例如一个图像生成服务同时支持“电商白底图”“写实人像”“室内设计”“二次元角色”等风格。全量微调每个模型都保存一份权重成本很高；LoKr 可以让每个领域保存一个较小 adapter。代价是每个 adapter 都要验证目标层、配置和生成质量，不能只根据文件大小判断好坏。

---

## 替代方案与适用边界

LoKr 应该放在参数高效微调方法族里理解，而不是单独神化。不同方法代表不同权衡。

| 方法 | 参数量 | 结构约束 | 训练稳定性 | 适用场景 | 代价 |
|---|---:|---|---|---|---|
| 全量微调 | 最高 | 最少 | 依赖数据和训练配置 | 任务变化大、资源充足 | 显存、存储、调参成本最高 |
| LoRA | 低 | 低秩约束 | 通常较稳 | 快速试验、语言模型适配、通用微调 | 表达能力受 rank 限制 |
| LoKr | 低到中 | 克罗内克块结构 | 依赖层形状和配置 | 图像模型风格/角色/领域适配 | 配置理解成本更高 |
| LoHa | 低到中 | Hadamard 相关结构 | 依赖实现 | LyCORIS 系列实验 | 需要更多对比验证 |
| 其他 LyCORIS 变体 | 不固定 | 不固定 | 不固定 | 针对特定模型或任务探索 | 文档和版本差异更明显 |

场景选择可以简单记：

| 场景 | 优先选择 |
|---|---|
| 任务 A：资源极少，只想快速跑通 | LoRA |
| 任务 B：Stable Diffusion 风格或角色微调，希望更强结构适配 | LoKr |
| 任务 C：业务目标变化很大，需要最大自由度 | 全量微调 |
| 任务 D：还在探索不同 adapter 表达形式 | LoRA、LoKr、LoHa 都做小规模对比 |

LoRA 更适合作为默认起点，因为它简单、资料多、实现成熟。LoKr 更适合在你已经确认 LoRA 表达不足，或者任务本身有明显结构特征时使用。比如图像模型中的风格、纹理、角色细节、构图偏好，往往比纯文本分类更依赖空间和块状结构，LoKr 的结构化更新可能更有价值。

不建议在以下情况优先使用 LoKr：模型很小、数据很少且没有稳定验证集、团队还不熟悉 target module 命名、训练框架版本混乱、只追求最短实现路径。此时 LoRA 更容易建立 baseline。baseline 的白话解释是：先做一个可靠的最低基准方案，用它判断后续复杂方法是否真的带来收益。

最终选择标准不是“LoKr 是否先进”，而是“在同样显存、训练时间、验证集质量和部署复杂度下，它是否比 LoRA 或全量微调更合适”。

---

## 参考资料

| 阅读目标 | 建议资料 |
|---|---|
| 入门使用 | Hugging Face PEFT 当前 LoKr 文档 |
| 理解实现 | LyCORIS 仓库 README 与源码 |
| 深入机制 | LyCORIS 相关论文与 FedPara 相关工作 |
| 查兼容性 | PEFT 旧版文档和当前安装版本文档对照 |

1. [Hugging Face PEFT LoKr 文档](https://huggingface.co/docs/peft/package_reference/lokr)
2. [LyCORIS 官方仓库](https://github.com/KohakuBlueleaf/LyCORIS)
3. [Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning to Model Evaluation](https://huggingface.co/papers/2309.14859)
4. [PEFT v0.12.0 LoKr 文档](https://huggingface.co/docs/peft/v0.12.0/package_reference/lokr)
5. [FedPara: Low-Rank Hadamard Product for Communication-Efficient Federated Learning](https://huggingface.co/papers/2108.06098)
