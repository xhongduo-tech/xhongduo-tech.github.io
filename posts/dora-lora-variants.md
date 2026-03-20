## 核心结论

DoRA、VeRA、LongLoRA 都属于 PEFT，即“参数高效微调”，意思是只训练很少一部分参数，让大模型适配新任务，而不是把全部权重都重新训练一遍。三者的目标不同：

| 方法 | 解决的核心问题 | 主要做法 | 推理阶段是否改模型结构 |
| --- | --- | --- | --- |
| LoRA | 全量微调成本太高 | 给权重加一个低秩增量 $BA$ | 否，可合并 |
| DoRA | LoRA 把“缩放”和“转向”混在一起 | 把权重拆成幅度与方向分别更新 | 否，可合并 |
| VeRA | LoRA 每层都要存一套 $A,B$，适配器太多时占空间 | 所有层共享随机矩阵，只学缩放向量 | 否，可合并 |
| LongLoRA | 长上下文训练显存和算力压力过大 | 训练时用 S²-Attn 稀疏注意力近似 | 否，推理仍用原始注意力 |

先给结论：

1. DoRA 适合“LoRA 已经能用，但你想更接近全量微调效果”的场景。它把权重的“长度”和“方向”拆开学，训练更稳定，常见于指令微调、专业问答、低比特量化后的补偿微调。
2. VeRA 适合“适配器非常多，存储比训练更贵”的场景。它不再为每层保存完整的低秩矩阵，而是复用一套随机基底，只保存每层的缩放向量。
3. LongLoRA 不是单纯替代 LoRA，而是解决“上下文长度扩展”问题。它通常和 LoRA 系方法一起用，让 32k、64k、100k 级上下文训练变得可行。

DoRA 最重要的公式是：

$$
W' = m \cdot \frac{V + \Delta V}{\|V + \Delta V\|_{(c)} + \tau}, \quad \Delta V = BA
$$

这里 $m$ 是幅度向量，白话讲就是每一列权重的“长度”；$V$ 是方向矩阵，白话讲就是单位化后的“朝向”；$\tau$ 是避免除零的小常数。DoRA 只训练 $m$ 和低秩增量 $BA$，原始方向基底来自预训练权重。这样做的直接收益是：缩放变化和方向变化不再挤在同一条梯度路径里。

---

## 问题定义与边界

先定义问题。假设某层线性层原始权重为 $W_0 \in \mathbb{R}^{d_{out}\times d_{in}}$。

LoRA 的写法是：

$$
W' = W_0 + BA
$$

其中 $A \in \mathbb{R}^{r \times d_{in}}$、$B \in \mathbb{R}^{d_{out} \times r}$，$r$ 是 rank，也就是低秩维度。低秩的意思是：不用直接学一个完整大矩阵，而是学两个更小的矩阵相乘。

LoRA 的优势很直接：训练参数少、部署简单、可以合并回原模型。但它有两个边界问题。

第一，LoRA 更新把幅度和方向绑在一起。某一列权重如果既需要“变长”又需要“转向”，LoRA 只能通过同一个增量去同时表达这两件事。训练时，这会让梯度既承担尺度调节，又承担几何方向修正。

第二，适配器数量一多，存储增长明显。假设一个模型有很多线性层，每层都要保存自己的 $A,B$。如果你做多任务、多客户、多版本适配器管理，累计体积并不小。

这也是 DoRA 和 VeRA 出现的直接背景：

| 问题 | LoRA 的表现 | 对应变体 |
| --- | --- | --- |
| 权重缩放和方向耦合 | 统一由 $BA$ 表达 | DoRA |
| 多层、多任务下适配器存储变大 | 每层独立保存 $A,B$ | VeRA |
| 长上下文训练显存压力高 | 注意力开销随序列长度快速上升 | LongLoRA |

一个玩具例子足够说明 DoRA 的直觉。

设某列原始权重为：

$$
w_0 = \begin{bmatrix} 3 \\ 4 \end{bmatrix}
$$

它的长度是 $\|w_0\|_2 = 5$，单位方向是：

$$
v = \frac{w_0}{\|w_0\|_2} = \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix}
$$

如果任务需要“整体强一点，但方向只小改”，DoRA 可以单独把长度改成 $5.2$，再让方向做很小修正。LoRA 也能做到近似效果，但它不能显式表示“哪部分是长度变化，哪部分是方向变化”。

边界也要说明清楚：

1. DoRA 不是对所有任务都必然优于 LoRA。小模型、小数据、低 rank、训练轮数短时，收益可能不明显。
2. VeRA 不是“更强的 LoRA”，而是“更省参数的 LoRA 家族方案”。它核心优化的是存储与扩展性，不是无条件提高精度。
3. LongLoRA 解决的是长上下文训练成本，不直接解决低秩表示能力问题。它经常与 LoRA/DoRA 同时使用，而不是二选一。

---

## 核心机制与推导

### 1. DoRA：把权重拆成幅度和方向

DoRA 的核心是列分解。对权重矩阵 $W_0$ 的每一列做：

$$
W_0[:,j] = m_j \cdot v_j, \quad \|v_j\|_2 = 1
$$

其中 $m_j$ 是第 $j$ 列的幅度，$v_j$ 是第 $j$ 列的方向。矩阵形式可以写成：

$$
W_0 = M \odot V
$$

更常见的实现里，不直接训练完整的 $V$，而是在预训练方向上加 LoRA 风格的低秩更新：

$$
V' = V + BA
$$

再按列归一化：

$$
W' = m' \cdot \frac{V'}{\|V'\|_{(c)} + \tau}
$$

这里 $\|V'\|_{(c)}$ 表示按列求范数。白话解释是：每一列先“转向”，再重新拉回单位长度，最后乘上新的幅度。

为什么这样更合理？因为方向更新会天然落在单位球的切空间附近，也就是“只改朝向，不直接放大长度”。从梯度形状上看，方向项会带一个投影结构：

$$
\nabla_{V'}L \propto \left(I - \frac{V'V'^\top}{\|V'\|^2}\right)\nabla_{W'}L
$$

这个投影项的含义很直白：把沿当前方向的分量剔掉，只保留正交方向上的修正。于是：

- 幅度参数负责“拉长或缩短箭”
- 低秩方向更新负责“转动箭头”

这比 LoRA 用一个增量同时做两件事更接近全量微调的学习模式。

### 2. VeRA：共享随机基底，只学缩放

VeRA 的全称可以理解为“向量化随机适配”。它保留低秩思想，但不再为每层学习完整的 $A,B$。做法是：

- 先生成一组共享随机矩阵 $A,B$
- 每一层只学习两个缩放向量

形式可写为：

$$
\Delta W^{(\ell)} = \operatorname{diag}(b^{(\ell)}) \, B \, \operatorname{diag}(d^{(\ell)}) \, A
$$

其中 $\operatorname{diag}(\cdot)$ 是把向量放到对角线上形成对角矩阵。白话讲，就是“底座模板全层共用，每层只调几个旋钮”。

参数规模因此明显变化。若忽略偏置和少量实现细节：

| 方法 | 每层新增参数规模 | 总体增长趋势 |
| --- | --- | --- |
| LoRA | $2r(d_{in}+d_{out})$ | 随层数和维度一起增长 |
| VeRA | 近似 $r + d_{out}$ 或同量级缩放向量 | 共享基底后更接近按层线性增长 |

当你需要管理很多 adapter 时，VeRA 的优势会很明显，因为共享矩阵只需保存一次，甚至在某些实现里可以靠随机种子重建。

### 3. LongLoRA：训练时稀疏，推理时回到原注意力

LongLoRA 的核心不是换掉 LoRA，而是引入 S²-Attn。S²-Attn 可以理解为“Shifted Sparse Attention”，即“带偏移的稀疏注意力”。

长序列的标准注意力成本近似是 $O(n^2)$。当序列长度从 4k 增到 100k 时，训练压力会非常大。LongLoRA 的做法是：

1. 把序列切成若干窗口
2. 窗口内做稠密 attention
3. 让部分头做半窗口偏移，形成相邻窗口间的信息流

这样训练时不用对全局所有 token 两两计算注意力，但又不是完全局部封闭。推理阶段仍可以使用原始全 attention 结构，因此模型部署形式不必重写。

真实工程例子：如果你要把一个 7B 模型从 4k 上下文扩到 100k，用于“整篇论文+附录+用户注释”联合问答，直接全量训练会非常贵。LongLoRA 会在训练期用 S²-Attn 降低成本，同时常配合 LoRA+ 或可训练 embedding/norm，保证长上下文适配不只是“记住位置”，而是真能利用更长上下文。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现。它不依赖深度学习框架，只演示 DoRA 的“幅度/方向分离”与 LoRA 的“直接增量”差异。

```python
import numpy as np

def normalize_columns(mat, eps=1e-12):
    norms = np.linalg.norm(mat, axis=0, keepdims=True)
    return mat / (norms + eps), norms

def dora_recompose(W0, delta, new_magnitude=None, tau=1e-12):
    direction, base_norms = normalize_columns(W0)
    V_prime = direction + delta
    V_hat, _ = normalize_columns(V_prime, eps=tau)
    if new_magnitude is None:
        new_magnitude = base_norms
    return V_hat * new_magnitude

def lora_recompose(W0, delta):
    return W0 + delta

# 玩具例子：两列权重
W0 = np.array([
    [3.0,  1.0],
    [4.0, -2.0],
])

# 低秩更新在这个玩具里直接写成 delta
delta = np.array([
    [0.1,  0.2],
    [-0.2, 0.1],
])

# DoRA：第一列长度从 5 调到 5.2，第二列保持不变
base_norms = np.linalg.norm(W0, axis=0, keepdims=True)
new_m = base_norms.copy()
new_m[0, 0] = 5.2

Wd = dora_recompose(W0, delta, new_m)
Wl = lora_recompose(W0, delta)

# 断言 1：DoRA 第一列长度接近 5.2
assert abs(np.linalg.norm(Wd[:, 0]) - 5.2) < 1e-6

# 断言 2：LoRA 的第一列长度不是显式控制的
assert abs(np.linalg.norm(Wl[:, 0]) - 5.2) > 1e-3

# 断言 3：DoRA 第二列长度保持原始长度
assert abs(np.linalg.norm(Wd[:, 1]) - np.linalg.norm(W0[:, 1])) < 1e-6

print("W0 =\n", W0)
print("DoRA recomposed =\n", np.round(Wd, 4))
print("LoRA recomposed =\n", np.round(Wl, 4))
```

这个例子说明一点：DoRA 能显式保证某列最终长度是多少，而 LoRA 没有这个单独控制接口。

如果进入实际工程，常见写法是直接使用 PEFT 配置。

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, VeraConfig, get_peft_model

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

dora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    use_dora=True,
)

dora_model = get_peft_model(base_model, dora_config)

vera_config = VeraConfig(
    r=256,
    target_modules="all-linear",
    projection_prng_key=42,
    save_projection=True,
)

vera_model = get_peft_model(base_model, vera_config)
```

这里有三个实现重点：

1. `use_dora=True` 表示仍走 LoRA 配置入口，但内部采用 DoRA 的权重分解逻辑。
2. `VeraConfig` 会构造共享投影矩阵，再为各层挂缩放向量。
3. 训练结束后通常会执行 `merge_and_unload()`，把适配器合并回基础权重，推理路径与普通模型一致。

实际落地时，DoRA 最适合替换现有 LoRA pipeline，而不是重写训练框架。也就是说，如果你已经有一套成熟的 LoRA 指令微调代码，DoRA 往往只是配置级改动。

---

## 工程权衡与常见坑

### 1. DoRA 的归一化不是免费的

DoRA 引入按列归一化，训练时会多一条范数相关的计算路径。如果把范数完整纳入反向传播，显存和算力开销会上升。工程上常见做法是把范数项视为常量近似，或者在实现里对归一化路径做 `detach` 式处理，减少反向图复杂度。

| 风险 | 具体表现 | 规避方式 |
| --- | --- | --- |
| DoRA 归一化参与完整反传 | 显存上涨、训练变慢 | 对范数路径做近似或截断梯度 |
| 幅度参数初始化不稳 | 初期 loss 抖动 | 从原始列范数初始化 $m$ |
| rank 太低 | 方向修正能力不足 | 提高 $r$ 或扩大 target modules |

### 2. VeRA 的重现性依赖随机基底

VeRA 的“省存储”依赖共享随机矩阵。但如果你选择不保存投影矩阵，而只保存随机种子，那么不同硬件、不同库版本、不同随机实现都可能让重建结果不一致。

因此，工程上要明确两种模式：

- 追求最小 checkpoint：保存 seed，不保存投影矩阵
- 追求绝对可重现：固定 `projection_prng_key`，并直接保存 projection

如果是生产环境或多人协作，后者更稳。

### 3. LongLoRA 不是“打开开关就变长”

LongLoRA 的关键点在训练流程，而不是模型参数名字。最常见误解是：只要把 attention 改成稀疏形式，就能获得长上下文能力。实际上不够。原因有两个：

1. 位置相关能力需要重新适配
2. embedding 和 norm 常常也要一起训练，否则只靠低秩更新，长距离依赖很难补齐

真实工程例子：做企业知识库问答时，用户一次输入可能包含几十页 PDF。若你把 7B 模型上下文从 4k 扩到 64k，但仍冻结 embedding 和 norm，只训练很小的 LoRA，模型往往能“吃进去”更长文本，却不一定能稳定利用远距离信息，表现为前半段内容答得准、后半段内容频繁遗忘。

### 4. 合并与量化顺序要注意

DoRA、LoRA、VeRA 最终都常要合并回基础权重。如果你的模型还带量化，比如 4-bit 或 8-bit，常见坑是：

- 先量化再合并，可能有额外误差
- 合并后再导出，不同框架支持路径不同
- 某些层支持 LoRA，不代表所有层都适合 DoRA/VeRA

因此需要先确认训练框架对目标模块的支持范围，再决定是“训练时量化”还是“合并后量化”。

---

## 替代方案与适用边界

DoRA、VeRA、LongLoRA 各自有明确边界，不应混成一个概念。

### 1. 什么时候优先 DoRA

如果你的核心目标是“在 LoRA 成本附近，尽量逼近全量微调效果”，优先看 DoRA。典型场景：

- 领域问答，尤其术语密集、答案风格稳定的任务
- 量化后微调，希望补偿表达能力损失
- 对训练稳定性比极限省参数更敏感的场景

原因是 DoRA 把尺度和方向拆开后，优化路径更清晰。

### 2. 什么时候优先 VeRA

如果你的核心目标是“海量 adapter 管理”，优先看 VeRA。典型场景：

- 一个底模服务很多客户，每个客户一个 adapter
- 需要在边缘设备或对象存储中长期保存大量适配器
- 任务差异中等，但更在意部署与存储成本

它的本质优势不是单任务极限精度，而是参数库管理效率。

### 3. 什么时候必须考虑 LongLoRA

只要任务目标包含“显著扩展上下文长度”，LongLoRA 就不是可选项，而是候选主方案。典型场景：

- 长文档问答
- 多轮法律或医疗材料分析
- 代码库级检索增强问答
- 论文、财报、审计报告的整文处理

### 4. 其他替代方案

| 方法 | 关键思路 | 更适合什么场景 |
| --- | --- | --- |
| AdaLoRA | 动态调整各层 rank 分配 | 参数预算严格、希望把 rank 用在更重要层 |
| 正交/流形约束 LoRA | 对方向空间加约束 | 更强调几何稳定性与可解释性 |
| Prefix/Prompt Tuning | 不改主干权重，只学前缀或提示向量 | 生成任务、接入轻量、但表达能力有限 |
| 全量微调 | 所有参数都训练 | 数据多、预算足、追求上限性能 |

可以用一个简单决策表收尾：

| 你的主要约束 | 更合适的方法 |
| --- | --- |
| 想在 LoRA 成本下追求更稳和更强 | DoRA |
| 适配器数量巨大，存储压力高 | VeRA |
| 主要问题是上下文长度不够 | LongLoRA |
| 预算充足，追求绝对性能上限 | 全量微调 |

---

## 参考资料

- DoRA: Weight-Decomposed Low-Rank Adaptation，NVIDIA Research，ICML 2024 相关页面  
- VeRA: Vector-based Random Matrix Adaptation，ICLR 2024 相关论文与 Hugging Face PEFT 文档  
- LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models，S²-Attn 与 LoRA+ 方案  
- Hugging Face PEFT 官方文档中 `LoraConfig(use_dora=True)` 与 `VeraConfig` 的实现说明  
- DoRA、VeRA、LongLoRA 的公开实现示例与技术解读材料
