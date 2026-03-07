## 核心结论

YaRN 和 LongRoPE 解决的是同一个核心问题：**RoPE 在训练长度之外直接外推时，位置角频率会脱离训练分布，进而让注意力分数失稳**。RoPE 是“旋转位置编码”，本质是把位置写进查询向量和键向量的相位里；相位关系一旦偏离训练期，模型对“哪些 token 应该互相关注”的判断就会变形。

两者的共同思路不是重写 Transformer 主体，而是**只改 RoPE 的频率分配方式**。关键变化是：不同维度不再统一按同一个倍率缩放，而是按频率分层处理。低频维度承担长距离关系，高频维度承担局部顺序和细节。YaRN 用非均匀插值加平滑过渡，LongRoPE 则进一步把“每一段维度该缩多少”提升为搜索或优化问题。

工程上可以先记住三点：

| 方案 | 核心动作 | 优点 | 代价 |
|---|---|---|---|
| 线性位置插值 | 所有频率统一缩放 | 最简单，改动最小 | 高频细节损失明显，短文本精度容易掉 |
| YaRN | 低频强缩放，高频少改，中间用 ramp 过渡 | 兼顾短文本与长文本，部署门槛低 | 需要调 `beta_fast`、`beta_slow`、温度等参数 |
| LongRoPE | 维度级非均匀缩放，可两阶段扩窗 | 可扩到 256k、1M、2M 级别 | 实现复杂，通常还需要继续训练或蒸馏适配 |

一个常见误解是“只要把最大长度从 4k 改到 16k 就行”。这不对。若训练时最大长度是 $L_{train}=4096$，推理时直接跑到 $L_{infer}=16384$，模型看到的相位模式已经进入未训练区域。真正要控制的是**频率如何随位置增长**，而不是只改配置文件里的长度上限。

---

## 问题定义与边界

先定义问题。标准 RoPE 会给每对偶数/奇数维分配一个角频率。对位置 $p$ 和第 $i$ 个频率分量，旋转角可写成：

$$
\theta_i(p)=p \cdot \text{inv\_freq}_i
$$

常见实现中，

$$
\text{inv\_freq}_i = \frac{1}{\text{base}^{\,2i/d}}
$$

其中：

| 符号 | 含义 |
|---|---|
| $p$ | token 位置，从 $0,1,2,\dots$ 开始 |
| $d$ | rotary 维度 |
| $i$ | 第 $i$ 个频率分量 |
| `base` | RoPE 基底，常见为 10000 |
| $\text{inv\_freq}_i$ | 第 $i$ 个倒频率，越大表示旋转越快 |

白话说：**高频维度转得快，适合表示局部顺序；低频维度转得慢，适合表示远距离关系。**

RoPE 真正起作用的不是绝对角度本身，而是查询与键之间的**相对相位差**。若两个位置分别是 $p$ 和 $q$，则相关项会依赖：

$$
\Delta \theta_i = (p-q)\cdot \text{inv\_freq}_i
$$

这也是为什么 RoPE 对相对位置敏感。问题在于，一旦 $|p-q|$ 远超训练分布，某些高频分量会快速绕圈，导致相位差进入模型从未见过的区域。

假设模型训练于 4k，上线后想支持更长输入：

| 训练窗口 | 推理窗口 | 扩展倍率 $s=L_{infer}/L_{train}$ | 风险 |
|---|---:|---:|---|
| 4k | 16k | 4x | 多数模型可尝试，但需要位置调整 |
| 4k | 32k | 8x | 只做线性插值通常已明显退化 |
| 4k | 256k | 64x | 基本必须做非均匀缩放，并配合继续训练 |
| 4k | 2048k | 512x | 需要分阶段扩窗与专门优化 |

这里要明确边界：

1. 讨论对象是 **RoPE 模型的上下文扩展**，不是 RAG、分块摘要、外部记忆库等系统层补偿方案。
2. 重点是 **推理时如何改位置编码**，但真实工程里通常还要配合少量长文本继续训练。
3. 不是所有任务都需要极长上下文。若业务主要在 2k 到 8k，激进扩窗很可能收益有限，风险却真实存在。

一个玩具例子：训练窗口是 4k，你把一篇 12k 技术文档直接喂给原始模型。模型在比较文首和文尾时，部分高频维度已经旋转了大量圈数，相邻 token 的相位差模式与训练期完全不同。结果不是“均匀变差”，而往往是**某些注意力头直接失去稳定模式**，表现为摘要跳段、引用错位、代码解释前后矛盾。

一个更接近生产的例子：法律卷宗、审计材料、超长代码仓库说明文档，长度常常在 80k 到 200k token。若只靠原始 4k RoPE 外推，模型在跨章节引用、附录编号映射、判例与正文对应上会明显失真。YaRN 和 LongRoPE 的价值就在于：**尽量保住 4k/8k 近场能力，同时把更长距离的位置信息重新映射到模型可处理的范围内。**

---

## 核心机制与推导

先从最基础的线性插值说起。若想从 4k 扩到 16k，最直接的想法是把推理位置压回训练范围：

$$
p'=\frac{p}{s}, \qquad s=\frac{L_{infer}}{L_{train}}
$$

这等价于把所有倒频率统一缩放为：

$$
\text{inv\_freq}^{interp}_i=\frac{\text{inv\_freq}^{orig}_i}{s}
$$

于是旋转角变成：

$$
\theta_i^{interp}(p)=p\cdot \frac{\text{inv\_freq}^{orig}_i}{s}
$$

它的优点很直接：长位置被压缩到训练期更熟悉的相位区间。缺点同样直接：**所有频率都变慢了**。低频变慢是我们想要的，因为它让模型能表达更长距离；但高频也一起变慢，就会伤到局部顺序、邻近 token 区分和短文本性能。

可以把这个问题理解成一个频率资源分配问题：

| 频率类型 | 原始职责 | 统一缩放后的问题 |
|---|---|---|
| 低频 | 编码长距离结构 | 变化不大，反而有助于扩窗 |
| 中频 | 兼顾局部与跨段关系 | 容易变得模糊 |
| 高频 | 编码邻近细节和局部顺序 | 最容易失真，短文本先受伤 |

YaRN 的关键改进就是：**不要所有维度一起等比例缩放。**

它把频率区间分成三段：

| 频率区间 | 作用 | 缩放方式 |
|---|---|---|
| 低频 | 长距离关系 | 强插值，通常按 $1/s$ 缩放 |
| 中间频率 | 连接长距与局部 | 用 ramp 做平滑过渡 |
| 高频 | 局部顺序与细节 | 尽量保留原频率或只做轻调 |

写成统一形式，可以表示为：

$$
\text{inv\_freq}_i^{new}
=
(1-r_i)\cdot \text{inv\_freq}_i^{interp}
+
r_i\cdot \text{inv\_freq}_i^{orig}
$$

其中：

$$
\text{inv\_freq}_i^{interp}=\frac{\text{inv\_freq}_i^{orig}}{s}
$$

$$
r_i \in [0,1]
$$

这里的 $r_i$ 是随维度变化的 ramp 系数：

- $r_i=0$：完全使用插值频率，更偏向长上下文
- $r_i=1$：完全保留原始频率，更偏向短文本
- $0<r_i<1$：处于过渡区

一个常见的分段定义是：

$$
r_i=
\begin{cases}
0, & i \le \beta_{slow} \\
\frac{i-\beta_{slow}}{\beta_{fast}-\beta_{slow}}, & \beta_{slow}<i<\beta_{fast} \\
1, & i \ge \beta_{fast}
\end{cases}
$$

其中 $\beta_{slow}$ 和 $\beta_{fast}$ 控制过渡区宽度。直觉上：

- $\beta_{slow}$ 越小，越多维度会被强插值
- $\beta_{fast}$ 越大，保留原始高频的区域越靠后
- 两者间隔太小会像硬切换，间隔太大则高频保护不足

一个 4k 到 16k 的玩具分段可以这样理解：

| 维度区间 | 处理方式 | 目的 |
|---|---|---|
| 低频前段 | 直接除以 4 | 给长距离关系腾出可用分辨率 |
| 中间一段 | 用 ramp 混合 | 避免频率断层 |
| 高频后段 | 保持原值或轻调 | 尽量保住近场和局部顺序能力 |

为什么要有 ramp，而不是硬切分？因为硬切分会造成频率分布不连续。RoPE 并不是单维起作用，而是多个维度共同决定相位模式；一旦某一段维度突然从“除以 4”跳到“完全不变”，就可能导致某些注意力头在特定长度区间出现异常峰值或衰减。

NTK-aware 插值比 YaRN 进一步，它不只是“保留哪些频率”，而是试图让扩窗后的模型在核意义上更接近原模型。工程上可以把它理解为：**除了位置缩放，还要调 RoPE 的基底或相位分布，让不同长度下的注意力几何关系更平滑。**它常作为中间路线，复杂度高于纯线性插值，但通常低于 LongRoPE。

LongRoPE 则把“每段维度该缩多少”从手工规则推进到搜索或优化问题。它通常不是只学一个统一倍率，而是学习一组**分维度或分频段缩放因子**。代表性流程通常是：

1. 先从较短窗口扩到中长窗口，如 4k 到 128k 或 256k，得到一组非均匀缩放因子。
2. 用少量长文本继续训练，让模型适应新相位分布。
3. 再基于适配后的模型继续扩到更长，如 1M 或 2M。
4. 对短窗口如 4k、8k 重新搜索兼容配置，减少短文本退化。

这也是 LongRoPE 与“普通插值”的根本区别：它不是一个单公式技巧，而是一套**频率重参数化 + 分阶段扩窗 + 再适配**的工程体系。

还有一个经常被忽略的量是 attention temperature。注意力分数通常形如：

$$
A=\text{softmax}\left(\frac{QK^\top}{\sqrt{d}} \cdot \tau \right)
$$

其中 $\tau$ 可以视为温度缩放项。扩窗后，即使位置频率分配更合理，注意力分数的尺度也可能变化，导致：

- 分数过尖：模型只盯极少数 token
- 分数过平：模型不敢聚焦有效远距离信息

因此很多工程实现会把**频率调整**和**温度调整**一起看，而不是孤立处理。

---

## 代码实现

下面给一个可运行的最小 Python 示例，目标不是复刻某个框架源码，而是把 YaRN 的核心动作拆清楚：

1. 构造原始 `inv_freq`
2. 为不同维度生成 ramp
3. 混合“插值频率”和“原始频率”
4. 构造 `cos/sin` cache
5. 用一个最小 RoPE 旋转函数验证形状与数值逻辑

```python
import math
from typing import List, Tuple


def build_inv_freq(dim: int, base: float = 10000.0) -> List[float]:
    """
    标准 RoPE 的倒频率。
    返回长度为 dim // 2 的列表，每个元素对应一对旋转维度。
    """
    if dim % 2 != 0:
        raise ValueError("dim must be even")

    return [1.0 / (base ** (2.0 * i / dim)) for i in range(dim // 2)]


def linear_ramp(index: int, low: int, high: int) -> float:
    """
    线性 ramp:
    - index <= low  -> 0.0
    - index >= high -> 1.0
    - 中间线性过渡
    """
    if low > high:
        raise ValueError("low must be <= high")

    if index <= low:
        return 0.0
    if index >= high:
        return 1.0
    if high == low:
        return 1.0

    return (index - low) / float(high - low)


def yarn_inv_freq(
    dim: int,
    scale: float,
    beta_slow: int,
    beta_fast: int,
    base: float = 10000.0,
) -> Tuple[List[float], List[float], List[float]]:
    """
    简化版 YaRN:
    - 低频维度更多使用 inv_freq / scale
    - 高频维度更多保留原始 inv_freq
    - 中间线性过渡
    返回:
    - orig: 原始倒频率
    - mixed: 新倒频率
    - ramps: 每个频率分量对应的 ramp 系数
    """
    if scale < 1.0:
        raise ValueError("scale should be >= 1.0 for context extension")

    orig = build_inv_freq(dim, base=base)
    mixed = []
    ramps = []

    for idx, inv in enumerate(orig):
        ramp = linear_ramp(idx, beta_slow, beta_fast)
        inv_interp = inv / scale
        inv_extrap = inv
        mixed_inv = inv_interp * (1.0 - ramp) + inv_extrap * ramp
        mixed.append(mixed_inv)
        ramps.append(ramp)

    return orig, mixed, ramps


def rope_angles(max_position: int, inv_freq: List[float]) -> List[List[float]]:
    """
    构造 [max_position, dim//2] 的角度矩阵
    """
    angles = []
    for p in range(max_position):
        row = [p * f for f in inv_freq]
        angles.append(row)
    return angles


def build_cos_sin_cache(max_position: int, inv_freq: List[float]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    构造 cos/sin cache，形状都是 [max_position, dim//2]
    """
    angles = rope_angles(max_position, inv_freq)
    cos_cache = [[math.cos(x) for x in row] for row in angles]
    sin_cache = [[math.sin(x) for x in row] for row in angles]
    return cos_cache, sin_cache


def apply_rope_to_vector(x: List[float], cos_row: List[float], sin_row: List[float]) -> List[float]:
    """
    对单个向量做 RoPE 旋转。
    x 的长度必须为偶数；每两维共享一个频率分量。
    """
    if len(x) % 2 != 0:
        raise ValueError("vector length must be even")
    if len(cos_row) * 2 != len(x):
        raise ValueError("cache dimension mismatch")

    out = [0.0] * len(x)
    for i in range(0, len(x), 2):
        pair_idx = i // 2
        x_even = x[i]
        x_odd = x[i + 1]
        c = cos_row[pair_idx]
        s = sin_row[pair_idx]

        out[i] = x_even * c - x_odd * s
        out[i + 1] = x_even * s + x_odd * c

    return out


def demo() -> None:
    dim = 16
    train_len = 4096
    target_len = 16384
    scale = target_len / train_len  # 4.0

    orig, mixed, ramps = yarn_inv_freq(
        dim=dim,
        scale=scale,
        beta_slow=1,
        beta_fast=5,
    )

    # 低频分量更接近插值结果
    assert abs(mixed[0] - orig[0] / scale) < 1e-12

    # 高频分量更接近原始结果
    assert abs(mixed[-1] - orig[-1]) < 1e-12

    # ramp 单调不减
    for i in range(len(ramps) - 1):
        assert ramps[i] <= ramps[i + 1]

    # 在长位置上，低频相位增长变慢
    p = 12000
    assert p * mixed[0] < p * orig[0]

    # 构造 cache 并做一次实际旋转
    cos_cache, sin_cache = build_cos_sin_cache(max_position=8, inv_freq=mixed)

    q = [1.0, 0.0] * (dim // 2)
    q_rot = apply_rope_to_vector(q, cos_cache[3], sin_cache[3])

    assert len(q_rot) == dim
    assert all(isinstance(v, float) for v in q_rot)

    print("YaRN toy example passed.")
    print("Original inv_freq:", [round(x, 6) for x in orig])
    print("Mixed inv_freq:   ", [round(x, 6) for x in mixed])
    print("Ramp values:      ", [round(x, 3) for x in ramps])


if __name__ == "__main__":
    demo()
```

这段代码有几个关键点：

| 参数 | 含义 | 工程解释 |
|---|---|---|
| `scale` | 扩展倍率 | 4k 到 16k 时就是 4 |
| `beta_slow` | 开始过渡的位置 | 更早进入混合区，更多维度偏向长上下文 |
| `beta_fast` | 结束过渡的位置 | 更晚完全回到原频率，短文本保护更强 |
| `base` | RoPE 基底 | 决定原始频率分布疏密 |

对于新手，最容易混淆的点有两个：

1. `idx` 不是 token 位置，而是**频率分量编号**
2. `inv_freq` 不是 attention 分数，而是用来生成旋转角的参数

如果把它接到真实模型里，通常改动位置不在 attention 主流程，而是在“构造 `inv_freq`”或“生成 `cos/sin` cache”的那一层。伪代码可以写成：

```python
def load_rope_frequencies(model_config, longrope_scales=None):
    inv_freq = build_default_inv_freq(
        dim=model_config.rotary_dim,
        base=model_config.rope_base,
    )

    if longrope_scales is not None:
        # LongRoPE: 使用离线搜索得到的分维度缩放因子
        inv_freq = [
            f / s for f, s in zip(inv_freq, longrope_scales)
        ]
    else:
        # YaRN / NTK-aware: 按目标长度运行时生成
        _, inv_freq, _ = yarn_inv_freq(
            dim=model_config.rotary_dim,
            scale=model_config.target_len / model_config.train_len,
            beta_slow=model_config.beta_slow,
            beta_fast=model_config.beta_fast,
            base=model_config.rope_base,
        )

    cos_cache, sin_cache = build_cos_sin_cache(
        max_position=model_config.target_len,
        inv_freq=inv_freq,
    )
    return cos_cache, sin_cache
```

一个实际部署方式通常是这样的：

| 请求长度 | 策略 |
|---|---|
| `<= 4k` | 原始 RoPE |
| `4k ~ 32k` | YaRN 或 NTK-aware |
| `> 32k` | 需要额外评测，必要时启用更激进配置或专门模型 |

这种做法的好处是风险隔离清晰。你可以先把上下文扩展作为**推理策略**上线，而不是立刻改模型权重版本。若发现 2k 到 4k 的代码补全或对话性能下降，也可以快速回退。

LongRoPE 的代码实现通常不会只有一个小函数，因为它往往还包括：

1. 缩放因子的离线搜索
2. 多阶段扩窗配置
3. 短窗口回补优化
4. 与继续训练数据管线的配合

因此工程上常见的分层是：

| 层级 | 负责内容 |
|---|---|
| 在线推理层 | 根据目标长度加载对应 `inv_freq`/cache |
| 配置层 | 存储不同窗口下的缩放因子 |
| 训练适配层 | 用长文本继续训练或蒸馏 |
| 评测层 | 同时看长窗口提升与短窗口回退 |

---

## 工程权衡与常见坑

最常见的错误是：**把“能塞进更长输入”误认为“能稳定理解更长输入”。** 模型能接受 64k token，不代表它在 64k 上还能保持可靠推理。

典型坑位如下：

| 坑 | 现象 | 规避方式 |
|---|---|---|
| 全局线性插值 | 短文本 perplexity 上升，局部语义变钝 | 改用 YaRN 或 NTK-aware |
| 没有 ramp 过渡 | 某些头在 8k/16k 处出现注意力断层 | 使用平滑分段而非硬切换 |
| 只改 RoPE 不做长文本适配 | 32k 以上回答表面连贯但定位错误 | 补少量长文本继续训练 |
| 扩展倍率过大 | 64x、128x 后稳定性明显崩溃 | 采用分阶段扩窗 |
| 忽略短窗口回退 | 4k 任务质量下降 | 单独评测短窗并保留兼容配置 |
| 只看通用基准 | 长文检索和跨段引用仍失真 | 增加结构化长距测试 |
| 温度未联动调节 | 注意力过尖或过平 | 与 RoPE 改动一起调 temperature |

其中最值得单独展开的是两个参数组。

第一组是 `beta_fast` / `beta_slow`。

可以把它们理解成“频率混合的缓冲带”：

| 配置情况 | 结果 |
|---|---|
| 过渡区太窄 | 接近硬切换，容易引发局部不连续 |
| 过渡区太宽 | 高频保护不足，短文本更容易退化 |
| `beta_slow` 太小 | 太多维度被强插值，近场能力受伤 |
| `beta_fast` 太小 | 还没到高频就已恢复原频率，长上下文提升有限 |

第二组是 attention temperature。

扩窗后即使相位处理合理，分数分布也可能变化。工程上常见症状有：

| 症状 | 可能原因 |
|---|---|
| 模型总盯少数远处 token | 分数过尖，温度偏高或频率失配 |
| 模型几乎不看远处 token | 分数过平，温度偏低或长距表示不足 |
| 远距检索偶尔命中但不稳定 | 位置映射和温度共同不稳 |

一个更可靠的判断标准是：**不要只看“模型有没有答出来”，要看它是不是准确利用了远距离证据。**

建议至少做四类测试：

| 测试类型 | 目的 |
|---|---|
| 长距检索 | 文首埋点，文尾提问，测定位准确率 |
| 跨章节问答 | 测模型能否连接多个远距离段落 |
| 引用一致性 | 测附录、编号、注释与正文能否对齐 |
| 短窗回归测试 | 测 2k/4k 任务是否退化 |

例如，你把 4k 模型直接拉到 256k，样例看起来“能回答”，这并不等于扩窗成功。它可能只是利用了问题附近的局部线索，根本没有稳定读取 100k 之外的信息。没有专门的长距测试，这类假象很常见。

---

## 替代方案与适用边界

YaRN 和 LongRoPE 不是唯一方案，也不总是最优方案。是否值得用它们，取决于任务是否真的需要“把整段原文放进模型并保持跨远距离推理”。

先看一个总览：

| 方案 | 适用长度 | 训练成本 | 对短文本影响 | 适合场景 |
|---|---|---:|---|---|
| 重新训练更长上下文模型 | 中到超长 | 高 | 可控 | 有算力、长期维护基础模型 |
| 线性插值 | 轻度扩窗 | 很低 | 中等到明显 | 临时验证、低要求任务 |
| YaRN / NTK-aware | 16k-128k 常见 | 低到中 | 较小 | 希望尽量保住原模型能力 |
| LongRoPE | 256k-2M | 中到高 | 可通过再搜索缓解 | 超长文档分析 |
| 滑动窗口 / chunking | 任意 | 低 | 几乎无 | 只需局部上下文 |
| RAG / 外部记忆库 | 任意 | 中 | 无直接影响 | 以知识查找为主 |

一个实用判断流程可以这样用：

1. 文档是否必须整体进入模型？
2. 如果不必须，优先考虑分块、滑窗或 RAG。
3. 如果必须整体进入，目标是 16k/32k，还是 256k 以上？
4. 若是 16k 到 32k，YaRN 或 NTK-aware 通常是性价比更高的方案。
5. 若是 256k、1M、2M，并且允许继续训练，LongRoPE 更合适。
6. 若业务大量依赖 2k 到 4k 的近场能力，必须把短文本回退纳入主评测，而不是附带评测。

可以用具体任务来理解：

| 任务 | 更合适的方案 | 原因 |
|---|---|---|
| 客服对话总结，长度 4k-8k | 原模型或轻度插值 | 不需要激进扩窗 |
| 长论文阅读，长度 16k-64k | YaRN / NTK-aware | 既要长文，也要保住一般能力 |
| 法律卷宗、年报、整本书分析 | LongRoPE | 需要稳定跨章节关系 |
| 代码补全、逐行翻译 | 谨慎扩窗 | 近场局部顺序比超长窗口更重要 |
| 知识问答 | RAG 往往优先 | 问题核心是取回信息，而非全文原地推理 |

需要特别强调的一点是：**更长上下文不是免费午餐。**

- 显存和 KV cache 成本会增加
- 推理延迟会增加
- 评测成本会增加
- 模型短窗能力可能回退

因此工程上的正确问题不是“能不能扩到 128k”，而是“业务是否真的需要 128k，以及这份收益是否值得相应成本”。

---

## 参考资料

下面按“论文原文、实现/复现、工程解读”三类整理。读法建议是：先看论文确定边界，再看实现确认参数，再看工程解读理解踩坑点。

| 来源 | 类型 | 侧重点 |
|---|---|---|
| YaRN 论文（arXiv:2309.00071） | 论文 | 非均匀频率缩放、ramp 过渡、扩窗动机 |
| LongRoPE 论文（arXiv:2402.13753） | 论文 | 分维度缩放、两阶段扩窗、超长上下文实验 |
| 相关开源实现中的 RoPE/YaRN 配置代码 | 实现 | `inv_freq` 构造、cache 生成、参数命名 |
| Papers With Code 上的 YaRN/LongRoPE 页面 | 复现入口 | 论文、代码、基准结果的聚合入口 |
| DeepWiki 或类似工程笔记中的 RoPE 变体总结 | 工程解读 | 各种 RoPE 变体的实现差异 |
| 面向实践的 YaRN / LongRoPE 拆解文章 | 工程解读 | 参数直觉、部署方式、常见误区 |

如果按阅读顺序排，可以这样看：

| 阅读顺序 | 目的 |
|---|---|
| 先读 YaRN 论文 | 先理解为什么“统一缩放”不够 |
| 再读 LongRoPE 论文 | 理解为什么要把缩放因子做成搜索问题 |
| 然后看开源实现 | 对上参数名、缓存构造和推理接入点 |
| 最后看工程解读 | 理解温度、短窗回退、评测设计等实务问题 |

还可以把这些资料分别回答不同问题：

| 你要确认的问题 | 最适合看什么 |
|---|---|
| 数学定义和方法边界 | 论文原文 |
| 参数怎么落到代码里 | 开源实现 |
| 为什么线上效果不稳定 | 工程解读与复现实验 |
| 如何比较 YaRN 和 LongRoPE | 论文实验部分 + 复现页面 |

最后给一个压缩判断：

- 只想从 4k 轻扩到 16k/32k，先看 YaRN
- 想把窗口做到 256k 以上，重点看 LongRoPE
- 想知道为什么改完“能跑但不稳”，重点看工程解读和评测设计
