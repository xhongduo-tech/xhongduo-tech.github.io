## 核心结论

位置编码外推，指的是**修改 RoPE 的位置映射规则，让短上下文训练的模型能推到更长上下文**。它解决的重点不是“重新发明位置编码”，而是让一个已经在短序列上训练好的模型，尽量平滑地适配更长输入。

最直观的场景是：模型训练时只见过 $L_{train}=2048$，现在推理要处理 $L_{test}=8192$，扩展倍率为

$$
s=\frac{L_{test}}{L_{train}}=4
$$

这时问题不在于“输入变长了”这句空话，而在于模型以前只学过前 2048 个位置对应的旋转规律，现在突然要处理第 3000、5000、7000 个位置，RoPE 产生的相位已经跑到训练分布之外，注意力会开始失真。

三种主流方法可以先压成三句话：

| 方法 | 核心思路 | 优点 | 代价 |
|---|---|---|---|
| PI | 把位置整体压缩回训练区间 | 最简单，改动少 | 压得太狠会伤短距离分辨率 |
| NTK-aware scaling | 不同频段采用不同缩放速度 | 比一刀切更稳 | 理解和实现更复杂 |
| YaRN | 分段混合缩放，并补偿 attention 尺度 | 长短上下文兼顾通常更好 | 参数更多，需要校准 |

玩具例子可以直接这么看：模型只熟悉 2k 的“跑步节奏”，现在突然让它跑 8k。外推方法做的事，就是把超出训练范围的位置信号重新映射，让模型还能大致认出“谁离谁近，谁离谁远”。

一句话判断优先级：**PI 是整体压缩位置，NTK-aware 是按频段差异化压缩，YaRN 是分段混合并补偿注意力尺度。**

---

## 问题定义与边界

RoPE，旋转位置编码，白话解释就是：**它不直接给 token 加一个编号向量，而是让查询和键在不同频段上按位置做旋转**。常见写法是

$$
\omega_i=b^{-2i/d}, \quad \phi_i(m)=m\cdot \omega_i
$$

其中：

- $d$ 是隐藏维度里用于 RoPE 的维数。
- $b$ 是频率基底，常见取值如 10000。
- $\omega_i$ 是第 $i$ 个频段的角速度。
- $\phi_i(m)$ 是位置 $m$ 在该频段上的旋转角。

问题本质不是“上下文长度变大”，而是**训练时学到的位置分布，到了更长序列后失配**。模型在训练中只看过 $m \le L_{train}$ 的旋转角，测试时如果直接喂入 $m > L_{train}$，相位会进入没有见过的区域。

设训练长度 $L_{train}=2048$，测试长度 $L_{test}=8192$，则 $s=4$。对新手来说，可以把它理解成：模型以前只知道“前 2048 个位置怎么排队”，现在你突然让它理解第 5000 个位置，它不是完全不会算，而是**算出来的旋转模式不再符合训练时见过的数据分布**。

这类方法的边界也要说清：

| 适用场景 | 是否适合外推 | 原因 |
|---|---|---|
| 长文 QA | 适合 | 需要跨大范围读取和定位 |
| 检索与摘要 | 适合 | 需要一次容纳更长文档 |
| 代码定位 | 适合 | 文件长，跨函数依赖远 |
| 短文本分类 | 不适合 | 根本不需要扩展长度 |
| 小上下文生成 | 不适合 | 收益很小，反而可能掉短文本质量 |
| 完全不需要长输入的任务 | 不适合 | 只增加复杂度，没有收益 |

必须明确一句：**外推方法不是替代长上下文微调，而是“先让模型能用，再尽量稳”**。如果任务本身要求强长距离推理，只改位置编码通常不够，后面仍然需要长上下文数据继续校准或微调。

---

## 核心机制与推导

先看根问题。RoPE 的关键是把位置 $m$ 映射成旋转角 $\phi_i(m)=m\omega_i$。当 $m$ 变得很大时，不同频段的角度都会继续增长，但它们增长得并不一致。高频段变化快，低频段变化慢。模型训练时学到的是某个长度范围内的统计规律，一旦超出这个范围，注意力对“相对远近”的判断就会变差。

### 1. PI：直接压位置

PI，位置插值，白话解释就是：**把长位置重新缩回短位置再算 RoPE**。公式最简单：

$$
m'=\frac{m}{s}
$$

如果 $L_{train}=2048, L_{test}=8192, s=4$，那么位置 $m=3000$ 会被当成

$$
m'=\frac{3000}{4}=750
$$

来看。也就是“3000 号位置，按 750 号位置的旋转去算”。它的优点是非常直接，几乎所有 RoPE 模型都能很容易接进去；缺点是一刀切，所有频段都按同样倍率压缩，可能损失局部细节。

### 2. NTK-aware：按频段缩放

NTK-aware scaling 的思路是：**不要对全部频段用同一个缩放倍率，而是让不同频段缩放不同**。常见写法是先改基底：

$$
b' = b \cdot s^{d/(d-2)}
$$

进一步可写成新频率：

$$
\omega'_i=\omega_i \cdot s^{-2i/(d-2)}
$$

这里的直观含义是：不同 $i$ 的频段会得到不同程度的压缩。不是所有角速度都等比例缩小，因此比 PI 更接近“保留局部、扩展全局”的目标。

在 $d=64, s=4$ 的例子里，低频和高频的变化不会一样。新手可以先记一句结论：**NTK-aware 不是把 3000 统一当成 750，而是在不同频段上，把“3000 应该像什么位置”这件事做成差异化处理。**

### 3. YaRN：分段混合并补偿尺度

YaRN 可以理解成更工程化的折中。它不是简单选“原频率”或“缩放频率”，而是在频段上做混合：

$$
\omega_i^y=(1-r_i)\cdot \omega_i + r_i \cdot \left(\frac{\omega_i}{s}\right)
$$

其中 $r_i \in [0,1]$ 是一个 ramp，白话解释就是：**一个从 0 平滑过渡到 1 的分段权重**。它通常由 `beta_fast` 和 `beta_slow` 定义边界。高频段 $r_i$ 接近 0，几乎不动；低频段 $r_i$ 接近 1，更接近按 $1/s$ 缩放；中间频段平滑过渡。

很多实现里还会引入 `attention_factor`，对白话解释就是：**修正注意力分数的温度，避免缩放后 logits 过大或过小**。

可以用一个简化示意图理解三者差别：

| 频段位置 | PI 缩放比例 | NTK-aware 缩放比例 | YaRN 缩放比例 |
|---|---|---|---|
| 高频 | 固定 | 缩放较小 | 接近不变 |
| 中频 | 固定 | 中等缩放 | 平滑过渡 |
| 低频 | 固定 | 缩放较大 | 接近 $1/s$ |

如果把横轴看成“频段”，纵轴看成“缩放比例”，那么：

- PI 是一条水平直线。
- NTK-aware 是一条连续变化曲线。
- YaRN 是“高频近原样，低频更强压缩，中间平滑连接”的分段曲线。

所以三种方法的共同目标，其实都是一件事：**控制“位置到相位”的变化速度，避免长位置进入训练外的失真区域。**

---

## 代码实现

工程里通常不是手写整套公式，而是直接改模型配置里的 `rope_scaling`。如果你用 Hugging Face 加载 Llama、Qwen 一类模型，常见做法是：先把训练时的最大长度告诉配置，再指定扩展倍率和方法，然后做一次短程继续微调或校准。

下面先给一个最小可运行的玩具实现，用来验证三种思路的核心计算。

```python
import math

def pi_position(m: float, factor: float) -> float:
    return m / factor

def ntk_new_base(base: float, factor: float, dim: int) -> float:
    return base * (factor ** (dim / (dim - 2)))

def ntk_scaled_freq(freq: float, index: int, factor: float, dim: int) -> float:
    return freq * (factor ** (-2 * index / (dim - 2)))

def yarn_freq(freq: float, factor: float, r: float) -> float:
    assert 0.0 <= r <= 1.0
    return (1 - r) * freq + r * (freq / factor)

L_train = 2048
L_test = 8192
factor = L_test / L_train
dim = 64
base = 10000.0

assert factor == 4.0
assert pi_position(3000, factor) == 750.0

new_base = ntk_new_base(base, factor, dim)
assert new_base > base

freq0 = base ** (-2 * 0 / dim)
freq_last = base ** (-2 * 30 / dim)

scaled0 = ntk_scaled_freq(freq0, 0, factor, dim)
scaled_last = ntk_scaled_freq(freq_last, 30, factor, dim)

assert math.isclose(scaled0, freq0)
assert scaled_last < freq_last

mixed = yarn_freq(freq_last, factor, r=0.5)
assert freq_last / factor < mixed < freq_last
```

上面这段代码体现了三个最重要的工程事实：

- `linear` 类方法主要改位置或等价缩放。
- `dynamic` 或 NTK-aware 类方法主要改 base/frequency。
- `yarn` 还会多出分段参数和 attention 补偿参数。

配置层面的常见写法可以概括成下面这样：

```python
# linear / PI 风格
config.rope_scaling = {
    "type": "linear",
    "factor": 4.0
}

# dynamic / NTK-aware 风格
config.rope_scaling = {
    "type": "dynamic",
    "factor": 4.0
}

# yarn 风格
config.rope_scaling = {
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 2048,
    "beta_fast": 32,
    "beta_slow": 1,
    "attention_factor": 1.0
}
```

如果从实现角度写伪代码，核心路径通常是：

```python
# 伪代码
read original_base, original_max_position_embeddings, dim
read rope_scaling.type and factor

if type == "linear":
    new_position = position / factor

elif type == "dynamic":
    new_base = original_base * factor ** (dim / (dim - 2))
    new_freq = build_inv_freq(new_base, dim)

elif type == "yarn":
    ramp = build_ramp(beta_fast, beta_slow, dim)
    new_freq = mix(original_freq, original_freq / factor, ramp)
    logits = logits * attention_factor
```

真实工程例子是：你要把一个 2k 训练的模型扩到 8k 做长文问答。常见路线不是“直接上线”，而是：

1. 配置 `rope_scaling`，把 `factor=4`，并写对 `original_max_position_embeddings=2048`。
2. 用少量 4k 到 8k 的真实任务数据做短程继续微调或校准。
3. 同时回归短文本任务，确认没有明显退化。

还可以用一张表快速检查改动点：

| 方法 | 需要改的参数 | 是否通常需要继续微调 | 风险点 |
|---|---|---|---|
| PI / linear | `factor` | 建议需要 | 过度压缩导致短距离分辨率下降 |
| dynamic / NTK-aware | `factor` | 建议需要 | 不同实现细节不完全一致 |
| YaRN | `factor`、`original_max_position_embeddings`、`beta_fast`、`beta_slow`、`attention_factor` | 更建议需要 | 分段边界和尺度补偿配错 |

配置校验清单必须做：

| 检查项 | 说明 |
|---|---|
| 目标长度是否一致 | 训练、推理、评估都要按同一个目标长度设计 |
| 训练长度是否写错 | `original_max_position_embeddings` 写错会让外推公式失真 |
| 推理代码是否读取新配置 | 有些部署链路会忽略配置字段 |
| 校准数据是否覆盖真实任务 | 只用通用文本校准，未必适合你的长文 QA 或代码检索 |

---

## 工程权衡与常见坑

外推不是“参数一改就稳定”。它对配置、数据、评估方式都很敏感。

最常见的误区是把“模型能吃下更长输入”和“模型真的会用长上下文”混为一谈。前者是长度适配，后者是任务效果。两者不是一回事。

一个典型坑是 `factor` 设得过大。比如你把 2k 直接拉到 32k，模型可能技术上能跑，但短文本质量明显下降。原因不复杂：位置压缩过猛后，原本近距离的差别被挤在更小区间里，局部顺序感会变差。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 只改推理参数，不做长上下文校准 | 长文性能抖动大 | 至少做短程继续微调 |
| `factor` 过大 | 短文本退化 | 小步长 sweep，比如 2x、4x、8x 逐步试 |
| YaRN 边界参数设置不当 | 分段过渡异常 | 校对 `beta_fast`、`beta_slow` 与实现 |
| 只测困惑度 | 误判长上下文能力 | 增加定位、检索、问答评测 |

评估项也不能只看一个数字。建议最少覆盖：

| 评估项 | 用途 |
|---|---|
| 短文本回归集 | 防止基础能力退化 |
| 长文档检索集 | 检查远距离定位能力 |
| needle retrieval | 检查“长文中找一根针”的能力 |
| 长文 QA | 检查跨段阅读与归纳 |
| 代码定位 | 检查长文件中的工程实用性 |

短结论很直接：

- 如果目标是快速上线，优先保证“能跑起来”，PI 或 dynamic NTK 往往够用。
- 如果目标是稳定可用，必须加入长短上下文双回归集，不能只看单一指标。

---

## 替代方案与适用边界

PI、NTK-aware、YaRN 不是唯一方案。选型要看你更在乎“部署速度”还是“质量稳定性”。

一个实用对比是：

| 方案 | 优点 | 缺点 | 是否需要微调 | 适合场景 |
|---|---|---|---|---|
| PI | 最快、最容易接入 | 一刀切压缩，退化风险更明显 | 建议做 | 快速把 7k、8k 能力上线 |
| dynamic / NTK-aware | 比 PI 更细 | 实现理解成本更高 | 建议做 | 想更稳地扩长度 |
| YaRN | 长短上下文兼顾通常更好 | 参数更多，调试更复杂 | 更建议做 | 线上长期使用、质量要求更高 |

真实场景可以这样判断：

- 要快速把 7k、8k 能力上线，先用 PI 或 dynamic NTK。
- 要尽量保住短文本质量，同时提升长上下文稳定性，更适合 YaRN。

除了直接做位置外推，还有两类常见替代思路。

第一类是**只做短程继续微调**。白话解释就是：不改太多位置公式，而是直接用更长样本把模型再训一小段。这种方法对算力要求更高，但有时比纯外推更稳。

第二类是**结合检索增强**。检索增强，白话解释就是：先从长文里找出相关片段，再让模型只读局部内容。它解决的是任务拆解，而不是位置编码本身。比如一个 50k 文档问答系统，可以先检索出 3 段最相关内容，每段 2k，再交给模型总结。

边界一定要写清：

- 外推是“长度适配”。
- 检索增强是“任务拆解”。
- 这两类方案可以组合，不是互斥关系。

工程上常见的组合拳是：**先用 YaRN 或 dynamic NTK 把窗口扩到 8k 或 16k，再配合检索，把超长任务拆成“召回 + 局部阅读”**。这样通常比单纯硬拉到超长窗口更稳。

---

## 参考资料

本文公式与实现对照来源，建议按“论文先理解原理，再看实现确认参数名”的顺序阅读。推荐顺序是：先读 PI，理解最简单的压缩思路；再读 NTK-aware 相关实现，理解按频段缩放；最后读 YaRN 和实现代码，理解工程落地。

1. [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)
2. [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
3. [Hugging Face Utilities for Rotary Embedding](https://huggingface.co/docs/transformers/internal/rope_utils)
4. [Hugging Face `modeling_rope_utils.py` 实现示例](https://huggingface.co/internlm/Intern-S1-Pro/blob/0d382e35c6e21c6ec9b1763c2a641c16f50647c3/modeling_rope_utils.py)
5. [YaRN 官方代码仓库](https://github.com/jquesnelle/yarn)
