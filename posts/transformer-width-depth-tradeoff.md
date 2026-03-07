## 核心结论

Transformer 的“宽度”通常指隐藏维度 $d_{\text{model}}$，也就是每一层向量表示的长度；“深度”通常指层数 $L$，也就是输入要经过多少个 Transformer block。两者都能提升模型能力，但在固定参数预算下，它们会直接竞争同一份资源，因此不能独立增加。

一个常用的参数近似公式是：

$$
P \approx V \cdot d_{\text{model}} + 12 \cdot L \cdot d_{\text{model}}^2
$$

其中：

- $P$：总参数量
- $V$：词表大小
- $d_{\text{model}}$：隐藏维度
- $L$：层数

第一项主要对应 token embedding，第二项主要对应每个 Transformer block 内的注意力投影和前馈网络。这个式子不是逐项精确计数，而是用于架构搜索时的数量级估算。

如果模型不小，通常 block 参数占主导，可以忽略 embedding 项，得到：

$$
P \approx 12Ld_{\text{model}}^2
\Rightarrow d_{\text{model}} \approx \sqrt{\frac{P}{12L}}
$$

这说明，在固定参数预算下，层数每增加，隐藏维度就必须按大约 $1/\sqrt{L}$ 的速度缩小。层数翻 4 倍，宽度大约只能保留一半。这就是宽度与深度的核心耦合关系。

对新手，可以把它理解成两个不同方向的投入：

- 宽度：提升单层的表示容量，让每一步变换更“粗壮”
- 深度：增加连续变换的步数，让信息能被逐层重组和提炼

因此，少层多宽更像“每一步都做很重的加工”，多层窄更像“把复杂过程拆成更多小步骤”。前者通常更有利于表示、压缩、分类；后者通常更有利于组合、多步推理、长链依赖传播。

实际工程中，几乎不会只押注一边。更常见的做法是：先根据任务类型确定偏向宽还是偏向深，再根据显存、吞吐、延迟和训练稳定性做折中，必要时再引入稀疏专家、局部注意力、分层结构等额外偏置。

---

## 问题定义与边界

问题不是“宽好还是深好”，而是：在固定预算下，参数应该优先放在更大的单层表示空间，还是优先放在更多层的层级组合上。

更严格地说，宽度与深度的比较至少受四个边界条件限制：

1. 参数预算是否固定  
如果参数预算不固定，那么“又宽又深”当然通常更强。但真正的架构搜索通常是在预算固定、算力固定或延迟固定的条件下进行的。

2. 训练计算是否固定  
同样参数量的两个模型，训练 FLOPs、激活开销、通信成本和并行效率不一定相同。参数相同，不等于训练代价相同。

3. 推理目标是否固定  
有些系统关心离线验证集指标，有些系统关心线上 P99 延迟，有些系统关心单位成本吞吐。同一个架构在这三种目标下的优先级可能完全不同。

4. 任务需要的是“表示”还是“组合”  
如果任务主要依赖高质量向量表示，那么宽度往往更划算；如果任务主要依赖多步状态变换，那么深度通常更关键。

可以先用一个粗粒度表格建立直觉。

| 方案 | 参数分配倾向 | 核心收益 | 典型代价 | 更适合的任务 | 不适合的场景 |
|---|---|---|---|---|---|
| 少层多宽 | 更多参数进入 embedding、attention 投影、FFN | 单层表示强，吞吐通常更友好 | 单层算子更重，组合步数少 | 分类、检索、短文本建模、召回重排 | 长链推理、复杂规划、图传播 |
| 多层窄 | 更多参数进入 block 堆叠次数 | 多步变换强，层级组合更充分 | 串行路径更长，优化更难 | 多跳推理、代码规划、长链关系建模 | 强延迟约束、弱算力部署 |
| 宽深折中 | 表示与组合都保留 | 泛化较稳，适合通用任务 | 调参空间更大 | 通用语言模型、理解与生成混合任务 | 极小模型预算、极端硬件限制 |

再看一个具体预算例子。假设总参数预算约为 7000 万，词表大小固定在 5 万：

- 方案 A：4 层、768 维
- 方案 B：64 层、256 维

两者参数量可做得接近，但行为不同：

- 方案 A：每层更宽，单步变换更强，适合把输入尽快压成高质量表示
- 方案 B：每层更窄，但有更多步骤，适合把复杂关系拆成多次更新

如果任务是情感分类、FAQ 匹配、短文本 rerank，A 往往已经足够强且更实用；如果任务是规则推理、程序执行轨迹预测、多跳关系组合，B 更有机会受益。

这里还需要补一个常被忽略的边界：宽和深不是唯一杠杆。很多时候真正决定收益的不是“再加几层”或“再加几点宽度”，而是有没有结构偏置，例如：

- 局部注意力：缩短长序列中的无效全连接
- 稀疏专家：增加总容量但不线性增加每步计算
- 分层路由：让不同 token 走不同路径
- 更好的归一化和残差设计：缓解深层训练退化

所以，本文讨论的是“在标准 Transformer 骨架下，宽与深如何分配”这个问题，而不是声称所有模型都只能在这两者之间二选一。

---

## 核心机制与推导

先把参数来源拆开。一个标准 Transformer block 主要包含两大部分：

1. 多头注意力的线性投影  
也就是输入经过 $W_Q, W_K, W_V, W_O$ 四组映射，把表示投影到 query、key、value 和输出空间。

2. 前馈网络 FFN  
通常是两层 MLP，例如从 $d_{\text{model}}$ 扩到 $4d_{\text{model}}$，再投回 $d_{\text{model}}$。这部分往往比注意力更吃参数。

如果忽略 bias、LayerNorm 等小项，单层参数可以写成：

$$
P_{\text{layer}} \approx P_{\text{attn}} + P_{\text{ffn}}
$$

对于标准配置：

$$
P_{\text{attn}} \approx 4d_{\text{model}}^2
$$

原因是 $Q,K,V,O$ 四个投影矩阵，每个大约都是 $d_{\text{model}} \times d_{\text{model}}$。

FFN 若采用扩展比 $r$，单层参数近似为：

$$
P_{\text{ffn}} \approx 2rd_{\text{model}}^2
$$

因为第一层是 $d_{\text{model}} \to rd_{\text{model}}$，第二层是 $rd_{\text{model}} \to d_{\text{model}}$。

当 $r=4$ 时：

$$
P_{\text{ffn}} \approx 8d_{\text{model}}^2
$$

于是单层 block 参数近似为：

$$
P_{\text{layer}} \approx 4d_{\text{model}}^2 + 8d_{\text{model}}^2 = 12d_{\text{model}}^2
$$

整个模型近似为：

$$
P \approx Vd_{\text{model}} + 12Ld_{\text{model}}^2
$$

这就是前面公式中常数 12 的来源。它不是数学常数，而是来自“标准 attention + FFN 扩展比为 4”的实现假设。如果采用 GQA、MQA、GLU、MoE 或不同 FFN 扩展比，这个常数会变化。

### 1. 固定参数预算下的宽深关系

若忽略 embedding 项：

$$
P \approx 12Ld_{\text{model}}^2
$$

可得：

$$
d_{\text{model}} \approx \sqrt{\frac{P}{12L}}
$$

因此：

- 深度增加 2 倍，宽度约缩小到原来的 $1/\sqrt{2}$
- 深度增加 4 倍，宽度约缩小到原来的 $1/2$
- 深度增加 16 倍，宽度约缩小到原来的 $1/4$

这说明“多堆几层”不是免费的。你得到的是更多变换步骤，但每一步的表示空间在变小。

### 2. 为什么不同任务对宽深的偏好不同

从计算图角度看，宽度和深度提供的是两种不同能力。

宽度更偏向：

- 提高单层可容纳的信息维度
- 增强单步非线性变换的表达力
- 让注意力和 FFN 在一次映射中容纳更多特征子空间

深度更偏向：

- 增加逐层重写状态的次数
- 提供更长的组合链条
- 让模型把复杂计算拆成多步完成

这两类能力不对等。某些任务更像“提取高质量特征”，某些任务更像“反复迭代中间状态”。

下面是一个更细的对照表。

| 任务特征 | 更依赖宽度还是深度 | 原因 |
|---|---|---|
| 分类、相似度判断、检索 | 偏宽 | 关键在于把输入映射成高质量表示 |
| 语言建模 | 折中 | 既需要表示容量，也需要层级组合 |
| 多步推理、代码执行规划 | 偏深 | 需要中间状态逐层更新 |
| 图结构传播、关系链组合 | 偏深 | 信息要沿结构多次传递 |
| 短上下文、高吞吐服务 | 偏宽 | 单层算子大但并行更友好 |
| 极低延迟场景 | 常偏宽浅 | 层数少可减少串行路径 |

### 3. 为什么深度收益会递减

经验上，深度增加后的收益通常不是线性增长。很多实验会观察到：

- 前几层增加收益明显
- 中后期继续加层，收益开始放缓
- 如果训练策略和结构不改，后续层容易学成相似变换

一种简化写法是：

$$
\Delta \mathcal{L}(L) \propto \frac{1}{L}
$$

这里 $\Delta \mathcal{L}$ 表示增加层数带来的损失改善量。这个式子不是精确定律，而是表达一个经验事实：越往后加层，边际收益越低。

深度收益递减主要有三类原因：

1. 层间表示趋同  
后面的层如果没有新的结构约束，很容易重复前面的变换。

2. 优化难度上升  
深层网络更依赖良好的初始化、残差路径、归一化和学习率设置。

3. 任务本身不需要那么多步骤  
如果任务不需要复杂组合，额外深度就没有足够的可学习信号。

可以把这件事理解成“流水线的工位设计”：

- 宽度像给每个工位更多工具
- 深度像增加工位数量

如果产品本身只需要 5 道工序，你把工位增加到 30 个，后面的工位很可能只是重复前面已经做过的事。

### 4. 为什么纯加宽也会遇到瓶颈

纯加宽的问题不是“没用”，而是成本上升很快，因为 block 参数按 $d_{\text{model}}^2$ 增长。

例如把宽度翻倍：

$$
d_{\text{model}} \to 2d_{\text{model}}
$$

则 block 参数近似变为：

$$
12L(2d_{\text{model}})^2 = 48Ld_{\text{model}}^2
$$

也就是在层数不变时，block 参数直接放大 4 倍。与此同时，激活显存、矩阵乘法成本、KV cache 大小也会随之增大。对部署来说，这常常比参数量本身更敏感。

### 5. 用数值例子建立直觉

假设：

- 参数预算 $P=70\,000\,000$
- 词表大小 $V=50\,000$

若取 $L=4$，则解近似方程可得到较大的 $d_{\text{model}}$；若取 $L=64$，为了不超预算，$d_{\text{model}}$ 必须显著减小。

把这种关系列成表格更直观：

| 层数 $L$ | 允许的典型宽度趋势 | 架构直觉 |
|---|---|---|
| 4 | 宽度可以很大 | 少数几次重加工 |
| 8 | 仍可保持较宽 | 比较常见的小中型实用配置 |
| 12 | 宽深开始平衡 | 通用模型常见区域 |
| 24 | 宽度明显受限 | 更偏向层级组合 |
| 48+ | 宽度必须进一步缩小 | 明显偏深，优化和延迟更敏感 |

### 6. 真实模型家族体现的不是“单押”，而是折中

公开模型家族很少只加宽或只加深，常见做法是共同增加，但幅度不同：

| 模型 | 代表配置 | 现象 |
|---|---|---|
| GPT-2 | 12×768 到 48×1600 | 深度和宽度都提升，面向更强语言建模 |
| BERT | Base 12×768，Large 24×1024 | 深度和宽度共同增加，强化上下文建模 |
| T5 | 多种 encoder-decoder 配置 | 更强调编解码平衡，而不是单纯堆单侧 |

这些例子说明，工程上真正常见的不是“宽 vs 深”的绝对选择，而是“在目标预算内，把宽和深放在什么比例”。

---

## 代码实现

下面给一个可直接运行的 Python 脚本，用来根据参数预算粗略搜索不同的宽深组合。它做三件事：

1. 用近似公式估算参数量
2. 在给定层数时求可行的最大隐藏维度
3. 输出一批候选配置，方便比较宽而浅、窄而深和折中方案

```python
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable, List


@dataclass(frozen=True)
class ArchConfig:
    vocab_size: int
    param_budget: int
    num_layers: int
    model_dim: int
    ffn_dim: int
    num_heads: int
    estimated_params: int
    block_params: int
    embedding_params: int


def estimate_params(vocab_size: int, model_dim: int, num_layers: int, ffn_ratio: int = 4) -> int:
    """
    近似公式:
    - embedding: V * d
    - attention: 4 * L * d^2
    - FFN: 2 * ffn_ratio * L * d^2
    当 ffn_ratio = 4 时，总计约为 V * d + 12 * L * d^2
    """
    embedding_params = vocab_size * model_dim
    attention_params = 4 * num_layers * model_dim * model_dim
    ffn_params = 2 * ffn_ratio * num_layers * model_dim * model_dim
    return embedding_params + attention_params + ffn_params


def solve_model_dim(param_budget: int, vocab_size: int, num_layers: int, ffn_ratio: int = 4) -> int:
    """
    解不等式:
        V * d + (4 + 2 * ffn_ratio) * L * d^2 <= P
    取满足预算的最大正整数 d。
    """
    if param_budget <= 0:
        raise ValueError("param_budget must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    a = (4 + 2 * ffn_ratio) * num_layers
    b = vocab_size
    c = -param_budget

    disc = b * b - 4 * a * c
    if disc < 0:
        raise ValueError("no real solution for the given budget")

    root = (-b + math.sqrt(disc)) / (2 * a)
    model_dim = max(1, int(root))

    # 向下修正，确保一定不超预算
    while estimate_params(vocab_size, model_dim, num_layers, ffn_ratio) > param_budget:
        model_dim -= 1

    return model_dim


def choose_num_heads(model_dim: int, head_dim: int = 64) -> int:
    """
    让 head 数尽量满足 model_dim = num_heads * head_dim。
    若无法整除，则退化为能整除 model_dim 的较小 head 数。
    """
    if model_dim < head_dim:
        return 1

    candidate = model_dim // head_dim
    while candidate > 1 and model_dim % candidate != 0:
        candidate -= 1
    return max(1, candidate)


def make_config(param_budget: int, vocab_size: int, num_layers: int, ffn_ratio: int = 4) -> ArchConfig:
    model_dim = solve_model_dim(param_budget, vocab_size, num_layers, ffn_ratio)
    ffn_dim = ffn_ratio * model_dim
    num_heads = choose_num_heads(model_dim)

    embedding_params = vocab_size * model_dim
    estimated = estimate_params(vocab_size, model_dim, num_layers, ffn_ratio)
    block_params = estimated - embedding_params

    return ArchConfig(
        vocab_size=vocab_size,
        param_budget=param_budget,
        num_layers=num_layers,
        model_dim=model_dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        estimated_params=estimated,
        block_params=block_params,
        embedding_params=embedding_params,
    )


def search_configs(
    param_budget: int,
    vocab_size: int,
    candidate_layers: Iterable[int],
    ffn_ratio: int = 4,
) -> List[ArchConfig]:
    configs = []
    for num_layers in candidate_layers:
        cfg = make_config(param_budget, vocab_size, num_layers, ffn_ratio)
        configs.append(cfg)
    return configs


def pretty_print(configs: List[ArchConfig]) -> None:
    print(
        f"{'layers':>6} {'d_model':>8} {'heads':>6} {'ffn_dim':>8} "
        f"{'params(M)':>10} {'embed(M)':>10} {'block(M)':>10}"
    )
    for cfg in configs:
        print(
            f"{cfg.num_layers:>6} {cfg.model_dim:>8} {cfg.num_heads:>6} {cfg.ffn_dim:>8} "
            f"{cfg.estimated_params / 1e6:>10.2f} {cfg.embedding_params / 1e6:>10.2f} "
            f"{cfg.block_params / 1e6:>10.2f}"
        )


if __name__ == "__main__":
    PARAM_BUDGET = 70_000_000
    VOCAB_SIZE = 50_000
    CANDIDATE_LAYERS = [4, 8, 12, 24, 48, 64]

    configs = search_configs(
        param_budget=PARAM_BUDGET,
        vocab_size=VOCAB_SIZE,
        candidate_layers=CANDIDATE_LAYERS,
        ffn_ratio=4,
    )

    # 基本正确性检查
    assert all(cfg.estimated_params <= PARAM_BUDGET for cfg in configs)
    assert configs[0].model_dim > configs[-1].model_dim
    assert configs[0].num_layers < configs[-1].num_layers

    pretty_print(configs)

    print("\nExample shallow-wide config:")
    print(asdict(configs[0]))

    print("\nExample deep-narrow config:")
    print(asdict(configs[-1]))
```

这段代码可以直接运行，且每一步都对应明确的工程含义：

| 字段 | 含义 | 为什么重要 |
|---|---|---|
| `model_dim` | 隐藏维度，即宽度 | 决定单层表示容量 |
| `num_layers` | 层数，即深度 | 决定多步变换次数 |
| `ffn_dim` | FFN 中间层维度 | 常见取 $4d_{\text{model}}$ |
| `num_heads` | 注意力头数 | 影响每层注意力的分解方式 |
| `embedding_params` | embedding 参数量 | 小模型里占比可能不低 |
| `block_params` | block 参数量 | 大模型里通常是主要部分 |

如果你运行这段脚本，通常会看到一种稳定趋势：

- 层数越少，可行的 `model_dim` 越大
- 层数越多，为了不超预算，`model_dim` 必须变小
- 很深的模型里，embedding 占比会下降，block 堆叠占比会上升

这正是前面公式的数值体现。

### 一个更接近工程决策的选择器

下面再给一个更偏工程的玩具函数。它不是为了最优，而是为了把“任务类型 + 部署目标”如何影响架构选择表达清楚。

```python
def choose_arch(task_type: str, latency_sensitive: bool, long_context: bool) -> dict:
    """
    一个简单的启发式示例：
    - 表示型任务：偏宽浅
    - 推理型任务：偏深
    - 长上下文：深度保留，同时需要结构化注意力配合
    """
    if task_type == "reasoning":
        return {
            "model_dim": 512,
            "num_layers": 24,
            "note": "偏深，适合多步组合；通常还需要较稳的优化配置"
        }

    if task_type == "retrieval" and latency_sensitive:
        return {
            "model_dim": 1024,
            "num_layers": 8,
            "note": "偏宽浅，适合低延迟高吞吐表示任务"
        }

    if long_context:
        return {
            "model_dim": 768,
            "num_layers": 16,
            "note": "宽深折中，但仅靠加深通常不够，需配合局部/分层注意力"
        }

    return {
        "model_dim": 768,
        "num_layers": 12,
        "note": "通用折中配置"
    }


cfg1 = choose_arch(task_type="reasoning", latency_sensitive=False, long_context=False)
cfg2 = choose_arch(task_type="retrieval", latency_sensitive=True, long_context=False)

assert cfg1["num_layers"] > cfg2["num_layers"]
assert cfg2["model_dim"] > cfg1["model_dim"]

print(cfg1)
print(cfg2)
```

这个例子想说明的不是“512×24 一定优于 1024×8”，而是：

- 推理型任务常更值得优先保留深度
- 表示型任务常更值得优先保留宽度
- 长上下文问题仅靠调宽深通常不够，还要改注意力结构

如果把它放到具体业务中：

- FAQ 检索重排器：通常更在意低延迟和高吞吐，偏宽浅架构更常见
- 代码规划、Agent 工具调用、多步规则推理：通常更值得优先保留足够深度
- 长文档问答：如果只加宽，常常治标不治本；需要深度与结构化注意力同时设计

---

## 工程权衡与常见坑

实际工程里，宽与深的争论通常不是抽象理论问题，而是稳定性、成本和部署目标的联合问题。

### 1. 盲目加深

最常见的问题是：层数加上去了，但后面层没有学到新的功能，训练成本和推理延迟却实打实增加了。

典型表现：

- 训练 loss 继续下降很慢
- 验证集提升小于预期
- 层间表示相似度偏高
- 推理延迟明显恶化

原因通常包括：

- 残差路径和归一化设计不足
- 学习率、warmup、初始化不适合更深结构
- 任务本身不需要这么多层级组合
- 缺少额外结构偏置，导致后续层重复前面计算

### 2. 盲目加宽

宽度的风险在于参数、激活和算子成本都会迅速膨胀。很多新手只盯着参数量，却忽略了单层矩阵乘法和中间激活的代价。

典型表现：

- 显存压力大幅上升
- batch size 被迫缩小
- 吞吐下降
- 训练不稳定时，以为“再加点宽度就能解决”

如果任务本身需要的是多步计算，那么纯加宽往往是在用更贵的单步表示去补偿缺失的组合链条，效率通常不高。

### 3. 只看参数量，不看训练和推理代价

同样都是 7000 万参数，两个模型的真实代价可能差很多。因为除了参数量，还要看：

- FLOPs
- 激活显存
- 序列长度下的注意力成本
- KV cache 大小
- 并行效率
- 通信开销

下面这个表更接近工程视角。

| 比较项 | 宽而浅 | 窄而深 |
|---|---|---|
| 参数量 | 可做相近 | 可做相近 |
| 单层计算 | 更重 | 更轻 |
| 串行路径 | 更短 | 更长 |
| GPU 吞吐 | 通常更友好 | 可能更差 |
| 推理延迟 | 常更低 | 常更高 |
| 多步组合能力 | 较弱 | 较强 |
| 训练稳定性 | 常较稳 | 更依赖优化细节 |

### 4. 忽略任务类型

“任务需要什么”比“论文里什么更强”更重要。一个检索 reranker 和一个代码规划器，对宽深的偏好可能完全不同。

下面给一个常见误区对照表。

| 坑 | 直接原因 | 常见表现 | 更合理的做法 |
|---|---|---|---|
| 盲目加深 | 以为层数越多越强 | 收益递减、延迟升高 | 先验证任务是否真依赖多步组合 |
| 盲目加宽 | 以为更大表示必然更强 | 显存吃紧、吞吐下降 | 先确认瓶颈是表示不足还是步骤不足 |
| 只看参数量 | 忽略 FLOPs 和串行深度 | 离线估计与线上体验不一致 | 同时比较延迟、吞吐、KV cache |
| 忽略结构偏置 | 只在宽深上做线性扩展 | 加很多资源但收益有限 | 必要时引入 MoE、局部注意力等 |

### 5. 为什么结构偏置能改变宽深权衡

结构偏置的本质是：不是让所有层做同样的事，而是明确引导不同层、不同专家或不同路径处理不同类型的信息。

例如：

- 稀疏专家 MoE：提高总容量，但每次只激活部分参数
- 局部注意力：限制交互范围，提高长序列效率
- 分层注意力：让不同层关注不同尺度的信息
- 跳连与路由：让信息不必机械地穿过每一层

这类方法的意义是增加第三个杠杆：条件计算。到这一步，问题就不再只是“宽还是深”，而是“是否应该改变计算路径本身”。

---

## 替代方案与适用边界

当纯粹调宽度和深度已经接近瓶颈时，常见替代方向至少有三类。

### 1. 稀疏专家 MoE

MoE 的核心不是简单增加层数或宽度，而是增加“总容量”和“路径分化”。它允许模型总参数很大，但每个 token 只激活少数专家，从而避免每一步都支付全部计算成本。

适用场景：

- 想扩总参数，但不能线性增加单步计算
- 数据分布复杂，希望不同模式走不同子网络
- 训练算力足够，但部署能接受路由机制

边界也很明确：

- 路由设计和负载均衡会增加训练复杂度
- 部署系统更复杂
- 不一定适合极低延迟、小规模服务

### 2. 结构化注意力或局部偏置

如果任务瓶颈来自长上下文、图结构或局部模式，单纯增加宽深往往不够，因为问题不在容量，而在路径设计不对。

常见方式包括：

- 局部注意力
- 滑动窗口注意力
- 分层注意力
- 图结构偏置
- 检索增强

这些方法更像是在改“信息怎么流动”，而不只是改“模型有多大”。

### 3. 任务特化架构

有些任务其实不需要一个通用大 Transformer。例如：

- 检索任务可能更适合双塔或轻量 cross-encoder
- 长文档任务可能需要检索增强或分块机制
- 代码任务可能更受益于更深层级组合和更长训练上下文

这意味着，很多时候“宽深怎么调”只是第二层问题，第一层问题是“骨架是否对题”。

下面给一个任务到策略的映射表。

| 任务类型 | 推荐策略 | 原因 |
|---|---|---|
| 分类、检索、短文本理解 | 少层、适度更宽 | 更看重表示质量与延迟 |
| 通用语言建模 | 宽深折中 | 同时需要表示和组合 |
| 多步推理、代码规划、图传播 | 先保深度，再补宽度 | 中间状态更新次数更重要 |
| 超大容量但单步算力受限 | MoE / 条件计算 | 扩总容量但控制每步成本 |
| 长上下文建模 | 深度 + 结构化注意力 | 仅加宽难以改善长程依赖路径 |

所以，宽与深的判断可以压缩成三步：

1. 先判断任务更偏“表示”还是“组合”
2. 再判断系统更偏“吞吐”还是“效果”
3. 最后判断是否需要引入第三个杠杆，例如 MoE、局部注意力或检索增强

如果任务明显偏推理，多数情况下应先保证足够深度，再考虑增加宽度；如果任务主要是表示、分类、召回，较少层但更宽的架构通常更划算，尤其是在低延迟场景。

---

## 参考资料

1. *The Optimal Architecture for Small Language Models*  
这类工作关注小模型的架构搜索，核心价值不是给出唯一最优答案，而是说明在固定参数预算下，宽度和深度会通过近似公式耦合，不能分开讨论。文中常见的近似关系正是：
$$
P \approx Vd_{\text{model}} + 12Ld_{\text{model}}^2
$$
它适合用来做第一轮候选架构筛选。

2. *Inverse Depth Scaling in LLMs*  
这类研究讨论“继续加深为什么不总是有效”。核心观察是：若结构、优化和训练目标不变化，后续层的边际收益会变小，甚至出现层间功能趋同。它提醒读者，深度不是免费午餐。

3. *Expressiveness of Transformers*  
这一方向从表达能力角度分析宽度与深度的差异。重要结论不是“深一定强于宽”或“宽一定强于深”，而是不同任务依赖的计算类型不同。组合式、算法式任务通常更依赖深度，表示式任务往往能从宽度中更快受益。

4. GPT-2、BERT、T5 的公开架构资料  
这些模型族最有价值的地方在于展示真实工程实践：主流模型扩展时通常同时加宽和加深，但比例不同。它们共同说明，工业界更关心“在预算内如何联合设计”，而不是只押某一个方向。

5. 关于 MoE、局部注意力和条件计算的公开工作  
这类资料说明，当纯宽深扩展逼近瓶颈时，改变计算路径往往比继续线性加参数更有效。它们提供的是第三个杠杆：不是只问“模型更宽还是更深”，而是问“是否应该让不同输入走不同子结构”。
