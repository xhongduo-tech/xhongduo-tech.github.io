## 核心结论

LoHa，全称 Low-Rank Hadamard Adaptation，意思是“低秩哈达玛适应”。它是一种参数高效微调方法：冻结原始模型权重 $W_0$，只训练一小组额外参数，用这些参数生成权重更新 $\Delta W$。

LoHa 的核心公式是：

$$
\Delta W = (B_1A_1)\odot(B_2A_2),\quad
y = (W_0 + \alpha \Delta W)x
$$

其中，$\odot$ 表示哈达玛积，也就是两个形状相同的矩阵逐元素相乘。白话说，矩阵中同一位置的两个数相乘，得到新矩阵同一位置的数。

LoRA 的更新通常写成：

$$
\Delta W = BA
$$

LoHa 不只生成一个低秩更新矩阵，而是先生成两个低秩更新矩阵，再把它们逐元素相乘。新手版理解是：LoRA 是两块积木拼成一个更新；LoHa 是先拼出两块更新图，再把两张图逐格叠加成最终更新图。

| LoRA | LoHa | 区别 | 直觉 |
|---|---|---|---|
| $\Delta W=BA$ | $\Delta W=(B_1A_1)\odot(B_2A_2)$ | LoRA 用一组低秩分解，LoHa 用两组低秩分解再逐元素相乘 | LoRA 画一张更新图，LoHa 画两张更新图再逐格融合 |
| 2 个小矩阵 | 4 个小矩阵 | LoHa 参数结构更复杂 | 用更多结构换更强表达力 |
| 主要依赖矩阵乘法生成更新 | 最终依赖 Hadamard 积组合更新 | 组合方式不同 | LoHa 的更新形状更细粒度 |

LoHa 的价值不在“公式更复杂”，而在“同样参数预算下表达力更强”。它适合参数预算敏感、但又希望适配器能表达更灵活权重变化的微调场景。

---

## 问题定义与边界

大模型微调的基本问题是：已经有一个预训练好的模型，但新任务的数据分布和原任务不同，需要让模型适配新任务。全量微调会更新所有参数，效果直接，但训练成本、显存占用、存储成本和部署管理成本都高。

参数高效微调，英文常写作 PEFT，是一类只训练少量额外参数的方法。它的目标不是重新训练整个模型，而是在冻结基座模型的基础上，为部分层注入小型可训练模块。

统一写法是：

$$
W = W_0 + \Delta W
$$

其中 $W_0$ 是冻结的原始权重，$\Delta W$ 是微调阶段学习出来的更新量。模型推理时实际使用的是更新后的权重 $W$。

玩具例子：一个线性层把二维输入映射到二维输出。原始矩阵 $W_0$ 已经能处理通用任务，但现在要处理一个很小的文本分类任务。如果全量训练 $W_0$，只有 4 个参数还不明显；但真实模型里一个投影层可能有几百万参数，全量更新会很重。LoHa 的做法是：不直接改 $W_0$，而是学习一个结构化的 $\Delta W$。

真实工程例子：在文本分类微调中，工程师通常不会更新整个大语言模型。更常见的做法是只对注意力层里的 `q_proj`、`v_proj` 等线性投影层注入适配器，让模型在保留原有语言能力的同时适配分类标签。

LoHa 解决的是“参数效率与表达力”的平衡问题，不是专门为推理加速设计的方法。它也不是所有层都默认适用，尤其是 embedding 层、卷积层、特殊投影层是否支持，需要看具体框架实现。

| 范围 | 说明 |
|---|---|
| 适用对象 | 线性层、部分注意力投影层、部分扩散模型权重层 |
| 主要目标 | 用较少可训练参数表达有效权重更新 |
| 不适合 | 必须重写大量原生权重的任务，或适配器表达力明显不足的任务 |
| 前提 | 训练框架支持 LoHa 注入、保存、加载和合并 |
| 非目标 | LoHa 本身不是推理加速算法，也不保证一定优于 LoRA |

---

## 核心机制与推导

低秩分解，白话说，就是用两个较小矩阵相乘来近似一个较大矩阵。假设原始线性层权重 $W_0\in\mathbb{R}^{m\times n}$，输入 $x\in\mathbb{R}^{n}$，输出 $y\in\mathbb{R}^{m}$。

LoRA 的思路是用：

$$
A\in \mathbb{R}^{r\times n},\quad B\in \mathbb{R}^{m\times r}
$$

生成：

$$
\Delta W=BA
$$

这里 $r$ 是秩，通常远小于 $m$ 和 $n$。秩可以理解为“中间压缩维度”，它控制可训练参数量和表达能力。

LoHa 使用两组这样的低秩分支：

$$
A_i\in \mathbb{R}^{r\times n},\quad B_i\in \mathbb{R}^{m\times r},\quad i\in\{1,2\}
$$

先得到两个矩阵：

$$
M_1=B_1A_1,\quad M_2=B_2A_2
$$

再逐元素相乘：

$$
\Delta W=M_1\odot M_2=(B_1A_1)\odot(B_2A_2)
$$

这一步很关键：$\odot$ 不是矩阵乘法。矩阵乘法会做行列内积，Hadamard 积只处理同一位置的元素。两个矩阵必须有相同形状。

最小数值例子：

$$
M_1=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
M_2=\begin{bmatrix}2&1\\1&2\end{bmatrix}
$$

逐元素相乘得到：

$$
\Delta W = M_1\odot M_2
= \begin{bmatrix}2&2\\3&8\end{bmatrix}
$$

若输入为：

$$
x=\begin{bmatrix}1\\1\end{bmatrix}
$$

则：

$$
\Delta Wx=\begin{bmatrix}4\\11\end{bmatrix}
$$

这个例子说明，LoHa 的重点不是把 $M_1$ 和 $M_2$ 再做一次矩阵乘法，而是让两个低秩分支在每个权重位置上产生交互。低秩分支提供参数压缩，Hadamard 积提供更细粒度的组合方式。

| 对比项 | LoRA | LoHa |
|---|---|---|
| 更新形式 | $\Delta W=BA$ | $\Delta W=(B_1A_1)\odot(B_2A_2)$ |
| 参数结构 | 一组 $A,B$ | 两组 $A_1,B_1,A_2,B_2$ |
| 组合方式 | 矩阵乘法生成一个低秩更新 | 两个低秩更新逐元素相乘 |
| 表达特点 | 更新受单个低秩子空间约束 | 通过元素级交互获得更灵活形状 |
| 主要代价 | 参数少，实现简单 | 参数略多，实现和框架支持更复杂 |

从表达力角度看，LoHa 不只是让更新矩阵落在一个简单低秩结构里，而是让两个低秩结构发生逐元素交互。因此，在相同或接近的参数预算下，它可能表示更复杂的权重变化。

---

## 代码实现

代码层面要分成三件事：注入适配器、训练适配器、合并权重用于推理。只会挂适配器不够，工程上还必须知道保存、加载、merge 和 unmerge 的行为。

下面先给一个完全可运行的 NumPy 玩具实现，用来验证 Hadamard 组合本身：

```python
import numpy as np

M1 = np.array([[1, 2], [3, 4]])
M2 = np.array([[2, 1], [1, 2]])
delta_w = M1 * M2  # Hadamard product, not matrix multiplication

x = np.array([[1], [1]])
out = delta_w @ x

assert delta_w.tolist() == [[2, 2], [3, 8]]
assert out.tolist() == [[4], [11]]
```

Hugging Face PEFT 风格的最小结构如下。注意：不同 PEFT 版本的 LoHa 配置类和参数名可能不同，实际项目应以当前版本文档为准。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model

model_name = "your-base-model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 示例写法仅用于说明结构，实际 LoHa 配置项以 PEFT 版本文档为准
config = {
    "peft_type": "LOHA",
    "r": 8,
    "alpha": 16,
    "target_modules": ["q_proj", "v_proj"]
}

model = get_peft_model(model, config)

# train(model, tokenizer, dataset)
model.save_pretrained("./loha-adapter")

# 训练后推理前，确认是否需要 merge
model = model.merge_and_unload()
```

更完整的最小训练流程可以写成：

```python
def load_model(model_name):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_name)

def apply_adapter(model, loha_config):
    from peft import get_peft_model
    return get_peft_model(model, loha_config)

def train(model, dataloader, optimizer):
    model.train()
    for batch in dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def save(model, path):
    model.save_pretrained(path)

def merge(model):
    return model.merge_and_unload()

# load_model -> apply_adapter -> train -> save -> merge
```

目标层选择会直接影响效果和稳定性。对文本模型来说，常见优先级如下：

| 层名 | 是否常用 | 备注 |
|---|---|---|
| `q_proj` | 常用 | 注意力查询投影，常见适配目标 |
| `v_proj` | 常用 | 注意力值投影，LoRA/LoHa 中都常见 |
| `k_proj` | 视任务而定 | 有时加入会提升效果，也可能增加成本 |
| `o_proj` | 视任务而定 | 注意力输出投影，适合更强适配需求 |
| `gate_proj` / `up_proj` / `down_proj` | 任务相关 | MLP 层适配会增加参数和训练成本 |
| `embed_tokens` | 谨慎 | 不同框架支持不一致，需先确认 |
| `lm_head` | 谨慎 | 可能影响输出分布，部署前要验证 |

---

## 工程权衡与常见坑

第一个常见错误是把 $\odot$ 误读成矩阵乘法。这样会直接导致形状推导、代码实现和训练结果都错。LoHa 的两个分支先各自生成 $m\times n$ 的矩阵，然后同位置相乘，最终仍然得到 $m\times n$ 的 $\Delta W$。

第二个问题是 rank 选择。`r` 太小，适配器容量不够，模型可能学不动；`r` 太大，训练参数、显存占用和存储成本都会上升，LoHa 的参数效率优势会被削弱。

例如，同样做一个分类任务，`r=4` 可能无法表达任务所需变化，验证集指标一直上不去；`r=32` 可能效果提升，但训练成本明显增加，并且开始接近“用更多参数硬学”的思路。合理做法是从较小 rank 起步，用验证集指标、训练稳定性和适配器大小共同决定是否增大。

| 问题 | 后果 | 规避方式 |
|---|---|---|
| 把 $\odot$ 当成矩阵乘法 | 形状错误或实现逻辑错误 | 明确 Hadamard 积是逐元素乘法 |
| `r` 过小 | 欠拟合，任务收益不明显 | 从小 rank 试起，根据验证集调大 |
| `r` 过大 | 参数效率下降，显存和存储增加 | 设定参数预算上限 |
| 目标层选择不当 | 效果差或训练不稳定 | 优先选择成熟支持的投影层 |
| 忽略框架版本 | 配置项不可用或行为变化 | 固定版本并查 API 文档 |
| 重复 merge 或重复加载适配器 | 推理输出异常 | 明确保存、加载、merge 状态 |

`merge/unmerge` 必须单独检查。merge 是把适配器学到的更新合并进基座权重，便于推理时使用普通模型结构。如果已经 merge 过，又重复叠加适配器，等价于把 $\Delta W$ 加了两次，输出会直接改变。工程上要记录模型当前状态：是“基座模型 + 适配器”，还是“已合并权重模型”。

---

## 替代方案与适用边界

LoHa 不是普适最优。它应该和 LoRA、LoKr、全量微调放在同一坐标系里比较。

LoKr，通常指 Low-Rank Kronecker Product，是使用 Kronecker 积结构的参数高效方法。Kronecker 积白话说，就是用两个小矩阵按块扩展成更大的矩阵。它和 LoHa 一样都在尝试用结构化参数提升表达效率，但组合方式不同。

| 方法 | 参数量 | 表达力 | 实现复杂度 | 适用场景 |
|---|---:|---|---|---|
| LoRA | 低 | 中等 | 低 | 快速试验、框架支持广、默认首选 |
| LoHa | 低到中 | 较强 | 中 | 参数敏感但需要更灵活更新形状 |
| LoKr | 低到中 | 较强 | 中到高 | 适合工具链明确支持 Kronecker 结构的场景 |
| 全量微调 | 高 | 强 | 中 | 数据充足、算力充足、需要最大可控性 |
| Prompt tuning | 很低 | 较弱到中等 | 低 | 任务简单、只需轻量行为调整 |

选型规则可以很明确：默认从 LoRA 起步，只有在任务收益明确、框架支持成熟、并且参数表达力成为瓶颈时再上 LoHa。

一个真实选择场景是：如果团队正在做大语言模型的文本分类微调，目标是快速上线并降低工程风险，LoRA 往往更稳，因为生态成熟、示例多、排错成本低。如果团队在做扩散模型风格定制，已经使用 LyCORIS 或 PEFT 中支持 LoHa 的版本，并且发现 LoRA 在相同参数预算下表达不足，那么 LoHa 就值得尝试。

LoHa 的边界也要说清楚。它不能保证小数据集上一定更好，也不能自动解决灾难性遗忘、数据质量差、标签噪声高等问题。适配器方法改变的是参数更新方式，不替代数据清洗、评估集设计和训练过程控制。

---

## 参考资料

1. [FedPara: Low-Rank Hadamard Product for Communication-Efficient Federated Learning](https://openreview.net/forum?id=d71n4ftoCBy)  
论文资料，用于理解 Low-Rank Hadamard Product 的提出背景，以及它在联邦学习中降低通信成本的动机。

2. [PEFT conceptual guide - Low-Rank Hadamard Product LoHa](https://huggingface.co/docs/peft/v0.18.0.rc0/conceptual_guides/adapter#low-rank-hadamard-product-loha)  
官方概念文档，用于确认 LoHa 和 LoRA 的核心差异，以及 LoHa 使用 4 个小矩阵的参数化形式。

3. [PEFT LoHa API reference](https://huggingface.co/docs/peft/package_reference/loha)  
官方 API 文档，用于确认具体版本中 LoHa 配置、目标层支持和调用方式。

4. [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)  
实现仓库，用于查看 LoHa、LoKr 等适配器方法在扩散模型和实际工具链中的实现细节。
