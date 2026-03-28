## 核心结论

YaRN 是一种把 RoPE 做“按频率分段处理”的长度外推方法。RoPE 是旋转位置编码，它不是给每个 token 直接加一个位置向量，而是把位置信息写进 query、key 的旋转角度里。这样一来，注意力分数天然带有相对位置信息。

YaRN 的关键点不在于“把所有维度统一拉长”，而在于承认不同频率的维度负责的信息不同，再按频率分别处理：

| 频段 | 负责的信息 | 处理方式 | 目的 |
|---|---|---|---|
| 低频 | 长距离结构 | 按 $1/s$ 缩放 | 让模型看得更远 |
| 中频 | 过渡信息 | 原始角度与缩放角度线性插值 | 避免频段切换过硬 |
| 高频 | 局部细节 | 保持不变 | 保住短程精度 |

其中 $s=L'/L$ 是扩展比例，表示目标上下文长度 $L'$ 相对原始训练长度 $L$ 放大了多少倍。

YaRN 相比“所有频率统一缩放”的做法，多了两个决定性设计：

1. 频率分组插值：低频缩放，中频平滑过渡，高频保留原样。
2. 注意力温度缩放：把 softmax 的有效锐度按 $0.1\ln(s)+1$ 做补偿，抵消长序列下注意力容易变平的问题。

一个新手版例子：原模型训练到 4K，目标推理到 64K，那么 $s=16$。此时不是把全部角度统一除以 16，而是只让低频部分完整除以 16，中频部分介于“除以 16”和“保持不变”之间，高频部分保持原样；同时再对注意力 logits 乘一个和 $\ln(16)$ 相关的修正项，避免长上下文下注意力过散。

简化写法可以记成：

$$
\theta'=(1-\gamma(r))\cdot \frac{\theta}{s}+\gamma(r)\cdot \theta
$$

以及

$$
\sqrt{\frac{1}{t}}\approx 0.1\ln s+1
\quad\Rightarrow\quad
t\approx \frac{1}{(0.1\ln s+1)^2}
$$

这里：

- $\theta$ 是原始旋转角度
- $\theta'$ 是外推后的角度
- $\gamma(r)$ 是频率区间上的插值权重
- $t$ 是 softmax 中的温度参数

一句话概括：YaRN 不是“把位置编码整体压扁”，而是“只让适合负责远距离的那部分频率去负责外推”。

---

## 问题定义与边界

问题很明确：一个只在 4K 长度上训练过的 Transformer，能不能在不重新做大规模长序列预训练的前提下，稳定跑到 32K、64K，甚至更长？

难点在于，RoPE 里的不同频率承担的职责不同。高频维度更偏局部细节，擅长区分近邻 token 的相对位置；低频维度更偏全局结构，适合表达较远距离上的依赖。统一缩放会同时伤害这两类能力。

先把文章里会用到的量定义清楚：

| 变量 | 含义 | 作用 |
|---|---|---|
| $L$ | 原始训练长度 | 模型原本学到的位置范围 |
| $L'$ | 目标推理长度 | 希望扩展到的范围 |
| $s=L'/L$ | 伸长比例 | 上下文扩大多少倍 |
| $\lambda$ | 对应频率的波长 | 越大越低频 |
| $r=L/\lambda$ | 频率比 | 用来判断该频率落在哪个区间 |
| $\theta=p\omega$ | 第 $p$ 个位置对应的旋转角 | RoPE 真正进入旋转矩阵的量 |

这里最容易卡住的新手点是：为什么要引入“波长” $\lambda$？

因为 RoPE 的每个二维子空间都对应一个角频率 $\omega$。频率越高，角度随位置变化得越快；频率越低，变化越慢。把它换成更直观的量就是波长：

$$
\lambda=\frac{2\pi}{\omega}
$$

- $\lambda$ 大：低频，适合表达慢变化、长距离结构
- $\lambda$ 小：高频，适合表达快变化、局部相对关系

所以 YaRN 实际上是在回答一个具体问题：哪些频率应该被“拉长”，哪些不应该？

YaRN 的边界也要说清楚：

1. 它解决的是“长度外推”，不是让模型凭空获得新知识。
2. 它通常需要少量继续训练或长上下文微调，不能理解成纯推理时零成本魔法。
3. 它常用于把上下文扩到原长度的数倍到几十倍；伸长比越大，越需要额外调参、数据和训练步骤。
4. 它只能缓解位置编码失配，不能解决注意力计算本身的二次复杂度，也不能自动提升检索、规划、记忆质量。

玩具例子：

- 一个维度主要编码“段首和段尾是否呼应”，这是低频。
- 另一个维度主要编码“两个相邻 token 的顺序差异”，这是高频。

如果这两类维度都统一除以 16，确实能让模型在角度空间里覆盖更远的位置，但高频维度原本负责的近邻区分会被压平。实际结果往往是：

- 短程语法边界更模糊
- 代码符号、标点、子词组合更容易错
- 长文虽然“看起来能放进去”，但局部精度先掉

YaRN 的思路就是承认：长距离结构和局部细节不是同一类任务，不该用同一个缩放公式粗暴处理。

---

## 核心机制与推导

YaRN 的机制可以分成两部分：RoPE 角度插值和注意力温度校正。

### 1. RoPE 角度插值

RoPE 的核心操作是把 query、key 的偶数维和奇数维两两配对，在二维平面中按位置旋转。对第 $i$ 个二维子空间，位置 $p$ 的旋转角可以写成：

$$
\theta_i(p)=p\omega_i
$$

其中 $\omega_i$ 是该维度对应的角频率。

如果直接做统一缩放，那么新角度就是：

$$
\theta_i'(p)=\frac{\theta_i(p)}{s}
$$

这等价于把所有频率都按同一比例降低。它的好处是简单，但问题也很直接：低频和高频被一视同仁。

YaRN 改成按频率比 $r=L/\lambda$ 分段：

- 当 $r<\alpha$，视为低频，完全采用缩放版本
- 当 $\alpha\le r\le \beta$，视为中频，原始角度和缩放角度线性混合
- 当 $r>\beta$，视为高频，保持原角度不变

经验上常见的阈值是 $\alpha=1,\beta=32$。插值函数写成：

$$
\gamma(r)=
\begin{cases}
0, & r<\alpha \\
\frac{r-\alpha}{\beta-\alpha}, & \alpha\le r\le \beta \\
1, & r>\beta
\end{cases}
$$

于是最终角度为：

$$
\theta'=(1-\gamma(r))\cdot \frac{\theta}{s}+\gamma(r)\cdot \theta
$$

这个式子很好读：

- 低频：$\gamma=0$，所以 $\theta'=\theta/s$
- 高频：$\gamma=1$，所以 $\theta'=\theta$
- 中频：介于两者之间，平滑过渡

为什么这样合理？因为长度外推真正要扩的是“模型可区分的长距离范围”，这本来就更应该由低频承担；而高频本来就负责局部精度，如果也一起拉长，最先坏掉的就是短程建模能力。

### 2. 为什么用 $r=L/\lambda$

这个比值的含义是：在原始训练长度 $L$ 内，一个波长 $\lambda$ 会重复多少次。

$$
r=\frac{L}{\lambda}
$$

- 若 $r\ll 1$：说明在训练窗口内这个频率变化很慢，属于低频
- 若 $r\gg 1$：说明它在训练窗口内已经绕了很多圈，属于高频

所以 $r$ 天然就是判断“该不该缩放”的指标。

用这个量来分段，比直接按维度编号分段更合理，因为真正决定语义角色的是频率，不是维度下标本身。

### 3. 注意力温度校正

只改位置编码还不够。上下文变长后，参与竞争的位置变多，注意力分布更容易变平。注意力一旦过平，模型虽然“能看见”很多 token，但不愿意把权重集中到真正相关的位置。

YaRN 的做法是对 softmax 前的 logits 做一个额外校正。论文里给出经验近似：

$$
\sqrt{\frac{1}{t}}\approx 0.1\ln s+1
$$

等价地：

$$
t\approx \frac{1}{(0.1\ln s+1)^2}
$$

放进注意力里可以写成：

$$
\text{softmax}\left(\frac{QK^\top}{t\sqrt{d}}\right)
$$

因为当 $s>1$ 时，通常有 $t<1$，所以分母变小，logits 的有效幅度会变大，softmax 会更尖一些。这一步的作用不是“让模型更激进”，而是补偿长上下文导致的熵增。

这里给一个直觉解释：

- 不做温度校正：候选位置变多，注意力更平均
- 做温度校正：把 logits 拉开一点，让真正相关的位置重新凸显出来

### 4. 代入数字看一遍

继续用例子：原长度 $L=4096$，目标长度 $L'=65536$，所以

$$
s=\frac{65536}{4096}=16
$$

再看三种频率：

1. 若某维度 $r=0.5<1$，属于低频，则 $\gamma=0$：

$$
\theta'=\frac{\theta}{16}
$$

2. 若某维度 $r=40>32$，属于高频，则 $\gamma=1$：

$$
\theta'=\theta
$$

3. 若某维度 $r=4$，属于中频，则

$$
\gamma=\frac{4-1}{32-1}=\frac{3}{31}\approx 0.097
$$

于是

$$
\theta'\approx 0.903\cdot \frac{\theta}{16}+0.097\cdot \theta
$$

再看温度项：

$$
0.1\ln(16)+1 \approx 1.277
$$

所以

$$
t\approx \frac{1}{1.277^2}\approx 0.613
$$

这意味着注意力中的有效缩放大约会额外乘上：

$$
\frac{1}{t}\approx 1.63
$$

这就是 YaRN 的完整直觉：

- 低频负责把视野拉远
- 高频负责把局部细节保住
- 中频负责避免硬切换
- 温度校正负责避免长上下文下注意力塌成一片“平均分配”

### 5. 和 NTK-aware 的差别

把 YaRN 只理解成“另一种插值公式”是不够的。更准确的说法是：YaRN 在 NTK-aware/PI 这类位置缩放思路上，又补了两层工程上真正重要的修正。

| 方案 | 对频率的处理 | 是否做过渡区 | 是否补偿长上下文熵增 |
|---|---|---|---|
| 统一 PI / 简单缩放 | 全部同一比例缩放 | 否 | 否 |
| NTK-aware 风格修正 | 主要是整体频率重标定 | 弱或无 | 通常不作为核心部分 |
| YaRN | 低频缩放、高频保留、中频插值 | 是 | 是 |

因此 YaRN 的优势不是“它更复杂”，而是它把两个实际问题都单独处理了：

1. 统一缩放会破坏高频局部信息。
2. 长序列会让注意力竞争格局变化，单改位置编码不够。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不依赖第三方库，直接演示 YaRN 的两个核心步骤：

1. 按 RoPE 频率计算每个维度的波长和频率比 $r$
2. 生成 YaRN 修正后的角度
3. 计算温度修正项
4. 用一个最小示例打印结果

```python
import math
from typing import List, Tuple


def build_inv_freq(head_dim: int, base: float = 10000.0) -> List[float]:
    """
    RoPE 常见写法中的 inverse frequency。
    只为每个二维子空间生成一个频率，因此长度是 head_dim // 2。
    """
    assert head_dim % 2 == 0, "head_dim must be even"
    return [1.0 / (base ** (2.0 * i / head_dim)) for i in range(head_dim // 2)]


def wavelength_from_inv_freq(inv_freq: float) -> float:
    """
    若角频率 omega = inv_freq，则波长 lambda = 2*pi / omega。
    """
    return 2.0 * math.pi / inv_freq


def gamma_of_r(r: float, alpha: float = 1.0, beta: float = 32.0) -> float:
    if r < alpha:
        return 0.0
    if r > beta:
        return 1.0
    return (r - alpha) / (beta - alpha)


def yarn_theta(theta: float, s: float, r: float, alpha: float = 1.0, beta: float = 32.0) -> float:
    gamma = gamma_of_r(r, alpha=alpha, beta=beta)
    return (1.0 - gamma) * (theta / s) + gamma * theta


def yarn_temperature_gain(s: float) -> float:
    """
    对应 sqrt(1/t) ≈ 0.1 ln(s) + 1
    这是直接乘到 logits 上更直观的形式。
    """
    if s <= 0:
        raise ValueError("s must be positive")
    return 0.1 * math.log(s) + 1.0


def yarn_temperature_t(s: float) -> float:
    gain = yarn_temperature_gain(s)
    return 1.0 / (gain ** 2)


def rope_theta(position: int, inv_freq: float) -> float:
    return position * inv_freq


def yarn_angles_for_position(
    position: int,
    head_dim: int,
    train_len: int,
    target_len: int,
    base: float = 10000.0,
    alpha: float = 1.0,
    beta: float = 32.0,
) -> List[Tuple[int, float, float, float, float]]:
    """
    返回每个二维子空间的:
    (index, lambda, r, theta_original, theta_yarn)
    """
    inv_freqs = build_inv_freq(head_dim=head_dim, base=base)
    s = target_len / train_len
    result = []

    for i, inv_freq in enumerate(inv_freqs):
        wavelength = wavelength_from_inv_freq(inv_freq)
        r = train_len / wavelength
        theta = rope_theta(position=position, inv_freq=inv_freq)
        theta_new = yarn_theta(theta=theta, s=s, r=r, alpha=alpha, beta=beta)
        result.append((i, wavelength, r, theta, theta_new))

    return result


def demo():
    # 例子: 4K -> 64K
    train_len = 4096
    target_len = 65536
    s = target_len / train_len
    assert s == 16

    head_dim = 16
    position = 6000  # 故意超过原始 4K，模拟长上下文位置

    rows = yarn_angles_for_position(
        position=position,
        head_dim=head_dim,
        train_len=train_len,
        target_len=target_len,
    )

    print(f"s = {s}")
    print(f"logit gain ~= {yarn_temperature_gain(s):.6f}")
    print(f"t ~= {yarn_temperature_t(s):.6f}")
    print()

    print("idx | wavelength | r=L/lambda | theta_orig | theta_yarn | band")
    print("-" * 78)
    for idx, wavelength, r, theta, theta_new in rows:
        if r < 1.0:
            band = "low"
        elif r > 32.0:
            band = "high"
        else:
            band = "mid"
        print(
            f"{idx:>3} | "
            f"{wavelength:>10.3f} | "
            f"{r:>10.3f} | "
            f"{theta:>10.6f} | "
            f"{theta_new:>10.6f} | "
            f"{band}"
        )

    # 基本正确性检查
    theta = 2.0
    low = yarn_theta(theta, s=s, r=0.5)
    mid = yarn_theta(theta, s=s, r=4.0)
    high = yarn_theta(theta, s=s, r=40.0)

    assert abs(low - theta / 16.0) < 1e-12
    assert theta / 16.0 < mid < theta
    assert abs(high - theta) < 1e-12
    assert yarn_temperature_gain(s) > 1.0
    assert yarn_temperature_t(s) < 1.0

    print()
    print("sanity checks passed")


if __name__ == "__main__":
    demo()
```

这段代码可以直接运行，输出里你会看到一个很直观的现象：

- 某些维度被标成 `low`，其角度明显按 $1/s$ 缩小
- 某些维度被标成 `high`，其角度几乎不动
- 中间那一段维度处于 `mid`，被平滑过渡

这比只写一个 `theta / s` 更贴近真实实现，因为它把“先算频率，再按频率决定缩放方式”的过程完整展示出来了。

如果放到真实模型里，改动点通常在两处：

| 模块 | 改动内容 | 目的 |
|---|---|---|
| RoPE 角度生成 | 按 $r$ 计算 $\gamma$，生成 $\theta'$ 或等价的 `inv_freq'` | 做分频外推 |
| Attention logits | 用 $t$ 或等价缩放修正 $QK^\top$ | 抵消长序列熵增 |

伪代码如下：

```python
# 1. 位置编码侧
gamma = clip((r - alpha) / (beta - alpha), 0.0, 1.0)
theta_scaled = theta / s
theta_new = (1.0 - gamma) * theta_scaled + gamma * theta

# 2. 注意力侧
gain = 0.1 * math.log(s) + 1.0   # 对应 sqrt(1/t)
t = 1.0 / (gain ** 2)

scores = (Q @ K.T) / (math.sqrt(d) * t)
attn = softmax(scores, dim=-1)
```

工程上还要补三个实现细节：

1. 很多框架不是显式改 $\theta$，而是改 `inv_freq` 或预先缓存的 `cos/sin`。本质等价，关键是最终旋转进去的角度满足上面的分段逻辑。
2. 不同实现对“温度校正”写法不同，有的改 `t`，有的直接乘一个 gain 到 logits。只要数学上等价即可。
3. 真正部署时要保证训练和推理对 RoPE 的实现一致，否则很容易出现“离线验证正常，线上长文本退化”的错配。

---

## 工程权衡与常见坑

YaRN 不是“把上下文长度配置项改大”这么简单。真正容易踩坑的地方，基本都出在“理论公式落到工程实现”这一层。

| 常见坑 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| 统一做 $\theta/s$ | 局部语法、拼写、代码 token 精度下降 | 高频细节被破坏 | 必须分频处理 |
| 不做温度校正 | 长上下文注意力发散、输出重复、主题漂移 | softmax 熵增 | 按 $\ln s$ 做 logits 补偿 |
| 只改推理不做继续训练 | 能跑长输入，但结果不稳定 | 模型没适应新的位置分布 | 做少量长文本继续训练 |
| $\alpha,\beta$ 机械照搬 | 某些模型效果波动大 | 频段切分和模型头维度、训练长度相关 | 结合验证集调参 |
| RoPE cache 实现不一致 | 训练正常，推理错位 | 角度缓存和实际公式不匹配 | 检查 `inv_freq/cos/sin` 全链路 |
| 只看 loss 不看注意力统计 | 线上长文本突然退化 | 问题先出现在机制层而非最终 loss | 同时看熵、logit 幅度、长短上下文分桶指标 |

一个典型错误是只做位置缩放，不做温度校正。模型表面上可以接收更长输入，但实际生成会出现这些症状：

- 重复段落
- 摘要时漏掉远端关键信息
- 代码补全中段还能对齐，尾部开始发散
- 长对话里更依赖最近几轮，较早上下文召回差

这类问题本质上不是“模型没看到远处 token”，而是“它看到了，但没有把足够的注意力分过去”。

真实工程里建议至少监控三类指标：

| 指标 | 看什么 | 作用 |
|---|---|---|
| PPL / loss | 长度变长后整体语言建模能力是否恶化 | 结果层监控 |
| Attention entropy | 注意力是否过平或过尖 | 机制层监控 |
| Gradient norm / update norm | 微调时训练是否失稳 | 优化层监控 |

如果要做得更稳，可以再补两类对照实验：

| 对照实验 | 目的 |
|---|---|
| 短上下文回归测试 | 确保扩窗后 4K 内性能没有明显回退 |
| 长短混合验证集 | 看模型是不是只会在长文本上“勉强能跑”，而非真正稳定 |

一个务实的工程判断标准不是“能不能塞进 64K”，而是下面三个问题能否同时回答“是”：

1. 短上下文性能没有明显退化。
2. 长上下文指标不是偶尔好、偶尔崩。
3. 输出质量提升来自真正使用远处信息，而不是随机运气。

如果 $s$ 很大，比如远超 32，YaRN 仍然可能有效，但不要把默认阈值和默认温度公式当成定理。伸长比越大，它越像“一个强基线”，而不是“最终答案”。

---

## 替代方案与适用边界

YaRN 的主要优势是便宜、直接、容易落地。它适合这种场景：已经有一个中等规模模型，原始窗口较短，现在希望在不重做完整长序列预训练的前提下，把上下文扩到 32K 或 64K，并愿意接受少量继续训练。

和常见替代方案对比更清楚：

| 方案 | 核心思路 | 复杂度 | 适合的伸长范围 | 代价 |
|---|---|---|---|---|
| YaRN | 分频插值 + 温度校正 | 中 | 中等倍数扩窗更常见 | 低到中 |
| LongRoPE | 对不同维度做非均匀搜索和渐进外推 | 高 | 更激进的超长外推 | 中到高 |
| 重新预训练 | 从训练阶段直接支持长上下文 | 最高 | 理论最稳 | 极高 |

LongRoPE 的强项是更细粒度。它不是只分三段，而是对不同维度寻找更合适的非均匀伸缩系数，再配合渐进式扩窗。因此在极端长上下文上更有潜力，但工程成本也更高：

- 搜索空间更大
- 验证流程更复杂
- 调参与复现成本更高

怎么选更实际：

1. 从 4K 扩到 32K 或 64K，资源有限，优先 YaRN。
2. 想从 4K 直接冲到 128K、200K 甚至更高，YaRN 往往不是终点，开始考虑 LongRoPE 或其他多尺度位置方案。
3. 如果长上下文是模型的核心卖点，而且训练数据、预算、时间都充足，重新预训练仍然是最稳的路线。

也要明确 YaRN 的非适用边界：

| 场景 | 为什么 YaRN 不是核心解法 |
|---|---|
| 想降低长上下文推理显存和算力 | YaRN 不改变注意力复杂度 |
| 想提升模型知识量 | 它只改位置泛化，不补知识 |
| 想让模型更会检索、规划、工具使用 | 这些更多取决于数据和训练目标 |
| 超长上下文下要求极高稳定性 | 往往需要更强的位置方案和更系统的长序列训练 |

一句话概括边界：YaRN 更像“高性价比扩容”，不是“无限外推许可证”。

---

## 参考资料

下面这部分建议分成“先看论文”和“再看工程实现”两层。

| 资料 | 年份 | 侧重点 |
|---|---|---|
| YaRN 原始论文《YaRN: Efficient Context Window Extension of Large Language Models》 | 2023 | 方法定义、公式、实验结果 |
| LongRoPE 论文《LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens》 | 2024 | 极长上下文下的非均匀插值与渐进扩窗 |
| jquesnelle/yarn 开源仓库与模型发布 | 2023-2024 | 社区实现、参数设置、LLaMA 扩窗实践 |
| 各类 RoPE / PI / NTK-aware 技术解读文章 | 2023-2025 | 建立直觉，帮助理解 YaRN 为什么要分频处理 |

如果你的目标是“把公式搞明白”，阅读顺序建议如下：

1. 先看 YaRN 原论文，弄清楚分频插值和温度修正各自解决什么问题。
2. 再看 LongRoPE，理解为什么极长上下文下需要更细粒度的非均匀缩放。
3. 最后看社区实现，确认这些公式在工程上到底落在 `inv_freq`、RoPE cache、还是 attention logits 上。

如果你的目标是“把代码改对”，重点检查这三项：

| 检查项 | 说明 |
|---|---|
| 角度生成位置 | 是直接改 $\theta$，还是改 `inv_freq` / `cos,sin cache` |
| 训练推理一致性 | 长短上下文下是否走同一套 RoPE 逻辑 |
| 长文本验证集 | 是否真的使用远处信息，而不是仅仅支持更长输入长度 |

几个可直接查找的资料名：

- YaRN: Efficient Context Window Extension of Large Language Models
- LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens
- `jquesnelle/yarn`
- “RoPE positional interpolation”
- “NTK-aware scaled RoPE”

如果只看一篇，优先 YaRN 原始论文；如果要判断它在你的场景里是不是够用，则必须把 LongRoPE 一起看，因为两者对应的是不同强度的扩窗需求。
