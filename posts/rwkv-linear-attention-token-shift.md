## 核心结论

RWKV 的核心改动，是把标准 Transformer 里“当前 token 对所有历史 token 做一次 softmax 注意力”的过程，改写成一个可递归更新的 WKV 算子。WKV 可以理解成“带指数遗忘的加权平均”：越久远的信息，默认权重越小，但不会被硬截断。

它的定义是：

$$
\operatorname{wkv}_t=\frac{\sum_{i\le t} e^{-(t-i)w+k_i}\cdot v_i}{\sum_{i\le t} e^{-(t-i)w+k_i}}
$$

其中，$k_i$ 是 key，表示“这个位置值不值得被记住”；$v_i$ 是 value，表示“真正要被记住的内容”；$w$ 是衰减率，表示“历史记忆每前进一步要衰减多少”。术语白话解释：衰减率就是“忘得快还是忘得慢”的参数。

这件事的重要性在于，RWKV 不再需要为每个新 token 回看整段历史并做 $O(T^2)$ 级别的注意力计算，而是只维护分子和分母两个递归状态，就能在 $O(1)$ 状态更新下继续表达长依赖。这里的 $O(1)$ 指“每一步更新只依赖固定大小状态，而不依赖历史长度”。

Token Shift 进一步补上了一个关键短板。单纯做递归记忆，容易让模型对“上一个 token 的局部变化”反应不够直接。Token Shift 的做法是，在进入线性层之前，先把当前 token 和前一个 token 做一次线性混合：

$$
r_t=\mu\odot x_t+(1-\mu)\odot x_{t-1}
$$

其中 $\mu$ 是混合系数，白话说就是“当前信息占多少，上一时刻信息占多少”。这样每一层在最开始就能看见一点局部上下文，而不是完全依赖更深层去慢慢形成短程关系。

一个最小玩具例子可以直接看出 WKV 的递归思想。设 $w=0.5$，第一个 token 有 $k_1=0,v_1=1$，第二个 token 有 $k_2=0.3,v_2=2$。到第 2 步时：

$$
\text{分子}=e^{-0.5+0}\cdot 1 + e^{0.3}\cdot 2 \approx 0.6065+2.6998=3.3063
$$

分母为：

$$
\text{分母}=e^{-0.5}+e^{0.3}\approx 0.6065+1.3499=1.9564
$$

所以：

$$
\operatorname{wkv}_2\approx \frac{3.3063}{1.9564}\approx 1.6900
$$

重点不在具体小数点，而在于第 2 步并不需要重新遍历全部历史，只要保留“上一步已经累好的状态”，再叠加当前项即可。

为了避免把 RWKV 想成“任何注意力都能压缩成两个状态”，先看两者的结构差异：

| 机制 | 标准 softmax attention | RWKV 的 WKV |
|---|---|---|
| 当前步依赖 | 当前 query 与全部历史 key/value | 上一步递归状态 + 当前 $k_t,v_t$ |
| 历史存储 | 整段 KV cache | 分子/分母等固定状态 |
| 序列计算代价 | 训练常见为平方级 | 递归更新更接近线性 |
| 长程记忆方式 | 显式对全部历史做归一化 | 指数衰减记忆 |
| 局部上下文 | 由 attention 自行学习 | 额外由 Token Shift 提供 |

一句话概括：RWKV 的价值不在于“完全等价替代 attention”，而在于“用更适合流式推理的状态递推结构，保留足够强的历史记忆能力”。

---

## 问题定义与边界

RWKV 要解决的问题，不是“完全复刻 Transformer 的全部行为”，而是“在长序列建模里保留足够强的历史表达能力，同时把计算和缓存结构改造成更适合流式推理的形式”。

更具体地说，目标有两个：

1. 用常数大小状态承接历史，而不是把所有历史 token 的 key/value 都存下来。
2. 让不同通道拥有不同的记忆节奏，避免所有特征都以同一种速度遗忘。

“通道”这个词，白话解释就是隐藏向量中的每一维或一组维度。不同通道通常承担不同类型的语义，比如有的更适合保留主题，有的更适合跟踪最近标点或局部语法。如果只用一个全局标量 $w$，所有通道都会按同样速度遗忘，这会把不同时间尺度的需求混在一起。

设序列输入为 $x_1,x_2,\dots,x_t$，系统希望在第 $t$ 步输出时，只使用固定大小状态。那它至少要维护两类量：

- 分子累计：历史值的“加权和”
- 分母累计：历史权重的“总和”

也就是：

$$
m_t=\sum_{i\le t} e^{-(t-i)w+k_i}v_i,\quad
n_t=\sum_{i\le t} e^{-(t-i)w+k_i}
$$

于是有：

$$
\operatorname{wkv}_t=\frac{m_t}{n_t}
$$

这已经把“注意力输出”变成了“两个状态的比值”。问题边界也由此清楚了：RWKV 擅长的是“随时间递推的加权记忆”，而不是任意 token 两两之间的自由对齐。后者是标准 attention 更自然的能力。

再看 Token Shift 的边界。它不是替代 WKV 的主记忆机制，而是补充一个“最邻近的局部上下文入口”。因为如果只靠递归状态，模型虽然能记很长，但对“刚刚前一个 token”的信息利用可能不够直接，尤其在浅层更明显。

| 组件 | 输入 | 状态 | 输出 | 主要依赖 |
|---|---|---|---|---|
| WKV | 当前 $k_t,v_t$ | $m_{t-1},n_{t-1}$ | $\operatorname{wkv}_t$ | 每通道衰减 $w$ |
| Token Shift | $x_t,x_{t-1}$ | 前一 token | 混合后的表示 | 混合系数 $\mu$ |
| RWKV-6 动态衰减 | 当前表示 $x_t$ | base decay | 动态 $w_t$ | 输入依赖偏移 |

把这个边界放到实际任务里更容易理解。假设模型在写技术文档，前半段已经定义了变量名 `cache_state`，后半段仍要保持一致。主题相关通道应该慢慢忘，才能维持长距离一致性；句法相关通道应该更快忘，才能跟上最近的标点、缩进和函数调用模式。如果所有通道共用同一个 $w$，这两类需求会直接冲突。

还可以把 RWKV 的能力边界总结成下面这张表：

| 问题 | RWKV 是否擅长 | 原因 |
|---|---|---|
| 维持长程主题一致性 | 是 | 递归状态可以持续保留历史摘要 |
| 流式逐 token 推理 | 是 | 每步只更新固定状态 |
| 直接检索历史中某个精确位置 | 一般 | 没有显式的任意位置对齐 |
| 最近一两个 token 的局部关系 | 需要补充 | 主要靠 Token Shift 和层间堆叠 |
| 通道级多时间尺度建模 | 是 | 每通道可拥有不同衰减率 |

所以，RWKV 的设计目标从一开始就很工程化：不是追求最自由的对齐，而是追求“状态小、更新快、长记忆仍够用”。

---

## 核心机制与推导

WKV 的关键，是把原始定义改写成递归式。

先看分子：

$$
m_t=\sum_{i\le t} e^{-(t-i)w+k_i}v_i
$$

把最后一项单独拿出来：

$$
m_t=\sum_{i\le t-1} e^{-(t-i)w+k_i}v_i + e^{k_t}v_t
$$

又因为对前 $t-1$ 项有：

$$
e^{-(t-i)w}=e^{-w}\cdot e^{-((t-1)-i)w}
$$

所以：

$$
m_t=e^{-w}m_{t-1}+e^{k_t}v_t
$$

同理，分母满足：

$$
n_t=e^{-w}n_{t-1}+e^{k_t}
$$

因此：

$$
\operatorname{wkv}_t=\frac{m_t}{n_t}
$$

这就是 RWKV 线性 attention 的核心。所谓“线性”，白话解释就是“每一步更新成本与历史长度无关，按步递推即可”。

如果把每个时间步拆开看，递推顺序很明确：

$$
(m_{t-1},n_{t-1}) \xrightarrow[\text{衰减}]{e^{-w}}
(e^{-w}m_{t-1},\,e^{-w}n_{t-1})
\xrightarrow[\text{加入当前 token}]{e^{k_t},v_t}
(m_t,n_t)
\xrightarrow[\text{归一化}]{/}
\operatorname{wkv}_t
$$

继续用玩具例子。设初始 $m_0=n_0=0$，$w=0.5$。

第 1 步：

$$
m_1=e^{-0.5}\cdot 0+e^0\cdot 1=1
$$

$$
n_1=e^{-0.5}\cdot 0+e^0=1
$$

所以：

$$
\operatorname{wkv}_1=1
$$

第 2 步：

$$
m_2=e^{-0.5}m_1+e^{0.3}\cdot 2
$$

若取 $e^{-0.5}\approx 0.6065,e^{0.3}\approx1.3499$，则：

$$
m_2\approx 0.6065+2.6998=3.3063
$$

$$
n_2=e^{-0.5}n_1+e^{0.3}\approx 0.6065+1.3499=1.9564
$$

$$
\operatorname{wkv}_2\approx \frac{3.3063}{1.9564}\approx1.6900
$$

这个例子只说明一件事：递归更新是严格成立的。真正训练时，模型会自己学习什么样的 $k_t$、$v_t$、$w$ 更有利于任务目标。

Token Shift 的推导更直接。设输入为 $x_t$，则某一路投影前先做：

$$
x_t^{(\text{mix})}=\mu\odot x_t+(1-\mu)\odot x_{t-1}
$$

然后再送入线性层：

$$
r_t=W_r x_t^{(\text{mix})},\quad
k_t=W_k x_t^{(\text{mix})},\quad
v_t=W_v x_t^{(\text{mix})}
$$

这里 $\mu$ 可以是向量而不是标量。向量的意思是“每个通道自己决定当前/前一 token 各占多少”。这比固定一个常数更灵活。

如果设 $\mu=0.6$，$x_1=0.2,x_2=0.5$，那么第 2 步混合结果是：

$$
x_2^{(\text{mix})}=0.6\times 0.5+0.4\times 0.2=0.38
$$

这表示第 2 个位置在进入线性层前，已经显式带入了一部分第 1 个位置的信息。

把两者放在一起，可以得到更完整的一层计算顺序：

| 步骤 | 作用 | 公式 |
|---|---|---|
| 1 | 局部混合 | $x_t^{(\text{mix})}=\mu\odot x_t+(1-\mu)\odot x_{t-1}$ |
| 2 | 线性投影 | $r_t=W_r x_t^{(\text{mix})},\;k_t=W_k x_t^{(\text{mix})},\;v_t=W_v x_t^{(\text{mix})}$ |
| 3 | 状态衰减并注入当前信息 | $m_t=e^{-w}m_{t-1}+e^{k_t}v_t,\;n_t=e^{-w}n_{t-1}+e^{k_t}$ |
| 4 | 归一化输出 | $\operatorname{wkv}_t=m_t/n_t$ |
| 5 | 后续门控或残差融合 | 由具体 RWKV 版本决定 |

RWKV-6 在此基础上继续放宽，把衰减率从固定参数扩展成输入相关参数：

$$
w_t=\text{base\_decay}+\Delta w(x_t)
$$

常见实现中，$\Delta w(x_t)$ 由小型低秩网络给出，可写成：

$$
w_t=\text{base\_decay}+\operatorname{LoRA}(x_t)
$$

“低秩”白话解释就是“用更少参数近似一个较大的线性变换”。这样做的结果是，模型可以在不同上下文里动态改变遗忘速度。比如新段落开始时提高衰减，尽快淡化旧主题；遇到同一段中的持续叙述时降低衰减，保留更长上下文。

这里有一个对新手很重要的理解：动态衰减不是把历史全部丢掉，而是在说“当前这个 token 到来时，哪些通道应该忘快一点，哪些应该忘慢一点”。它调的是记忆节奏，不是把状态机制推翻重来。

| 变量 | 作用 | 每步是否更新 | 典型形状 |
|---|---|---|---|
| $m_t$ | 分子累计状态 | 是 | `(batch, channel)` |
| $n_t$ | 分母累计状态 | 是 | `(batch, channel)` |
| $w$ 或 $w_t$ | 衰减率 | 固定或动态 | `(channel)` 或 `(batch, channel)` |
| $\mu$ | Token Shift 混合系数 | 固定或可学习 | `(channel)` |
| $x_t$ | 当前输入表示 | 是 | `(batch, channel)` |

---

## 代码实现

下面给出一个最小可运行实现，目的是说明状态更新顺序，而不是复刻完整 RWKV 训练框架。代码里每个通道独立维护 $m,n$，这正是“per-channel”实现的基本形式。代码可以直接保存为 `rwkv_minimal.py` 并运行。

```python
import math
from typing import Iterable, List, Sequence, Tuple


def token_shift(x_prev: Sequence[float], x_cur: Sequence[float], mu: Sequence[float]) -> List[float]:
    """
    x_prev, x_cur, mu 的长度都等于 channel 数。
    mu[c] 表示第 c 个通道中，当前 token 占比多少。
    """
    if not (len(x_prev) == len(x_cur) == len(mu)):
        raise ValueError("x_prev, x_cur, mu must have the same length")
    return [m * xc + (1.0 - m) * xp for xp, xc, m in zip(x_prev, x_cur, mu)]


def wkv_step(
    m: Sequence[float],
    n: Sequence[float],
    w: Sequence[float],
    k: Sequence[float],
    v: Sequence[float],
) -> Tuple[List[float], List[float], List[float]]:
    """
    单步 WKV 递推。
    m: 上一步分子状态
    n: 上一步分母状态
    w: 每通道衰减率
    k: 当前 key
    v: 当前 value

    返回:
    out: 当前步 wkv 输出
    new_m: 更新后的分子状态
    new_n: 更新后的分母状态
    """
    if not (len(m) == len(n) == len(w) == len(k) == len(v)):
        raise ValueError("all inputs must have the same length")

    out = []
    new_m = []
    new_n = []

    for mi, ni, wi, ki, vi in zip(m, n, w, k, v):
        decay = math.exp(-wi)
        ek = math.exp(ki)

        mi_new = decay * mi + ek * vi
        ni_new = decay * ni + ek
        yi = mi_new / ni_new

        new_m.append(mi_new)
        new_n.append(ni_new)
        out.append(yi)

    return out, new_m, new_n


def run_sequence(
    w: Sequence[float],
    ks: Iterable[Sequence[float]],
    vs: Iterable[Sequence[float]],
) -> List[List[float]]:
    """
    用一串 k_t, v_t 运行整个序列，返回每一步的 wkv 输出。
    """
    channels = len(w)
    m = [0.0] * channels
    n = [0.0] * channels
    outputs: List[List[float]] = []

    for k_t, v_t in zip(ks, vs):
        out, m, n = wkv_step(m, n, w, k_t, v_t)
        outputs.append(out)

    return outputs


def main() -> None:
    # 单通道玩具例子
    w = [0.5]
    ks = [[0.0], [0.3]]
    vs = [[1.0], [2.0]]
    outputs = run_sequence(w, ks, vs)

    # token shift 例子
    x1 = [0.2]
    x2 = [0.5]
    mu = [0.6]
    mix2 = token_shift(x1, x2, mu)

    print("wkv outputs:", outputs)
    print("mixed x2:", mix2)

    # 断言用于验证实现与公式一致
    assert abs(outputs[0][0] - 1.0) < 1e-12
    assert abs(outputs[1][0] - 1.6900386997) < 1e-9
    assert abs(mix2[0] - 0.38) < 1e-12


if __name__ == "__main__":
    main()
```

运行后应得到近似输出：

```text
wkv outputs: [[1.0], [1.6900386997...]]
mixed x2: [0.38]
```

上面 `assert` 的作用，是在最小例子中验证递归更新没有写错。对于初学者，最容易错的地方有三个：

1. 先更新 `m,n`，再计算 `out`
2. `w` 必须按通道广播，不能偷懒写成单个全局标量
3. `k` 进入指数时要和 `v` 同步对应到当前步，不能错位

如果换成张量框架，伪代码通常就是：

```python
decay = torch.exp(-w)                # shape: (channel,)
ek = torch.exp(k)                    # shape: (batch, channel)
m = decay * m + ek * v               # shape: (batch, channel)
n = decay * n + ek                   # shape: (batch, channel)
wkv = m / n
```

这里的维度关系建议直接记成表：

| 变量 | 推荐维度 | 说明 |
|---|---|---|
| `x` | `(batch, channel)` | 当前层输入表示 |
| `k, v` | `(batch, channel)` | 当前步投影结果 |
| `w` | `(channel,)` | 每通道衰减率 |
| `m, n` | `(batch, channel)` | 递归状态 |
| `mu` | `(channel,)` 或 `(batch, channel)` | Token Shift 混合系数 |

真实工程里还要处理数值稳定性。因为 `exp(k)` 可能非常大，工程实现通常会用“最大值平移”或 log-sum-exp 风格的重参数化，避免上溢。常见思路是额外维护一个缩放基准，让分子分母都在可控范围内更新。原理不变，只是把“数学上等价”的式子改写成“浮点数上更安全”的式子。

下面给一个简化后的稳定化思路，帮助理解为什么需要这一步：

$$
\frac{e^{a_1}b_1 + e^{a_2}b_2}{e^{a_1}+e^{a_2}}
=
\frac{e^{a_1-c}b_1 + e^{a_2-c}b_2}{e^{a_1-c}+e^{a_2-c}}
\quad\text{其中 } c=\max(a_1,a_2)
$$

两边数学上完全相等，但右边更不容易溢出。RWKV 的稳定实现本质上也在做类似处理。

再看一个更接近真实部署的场景。假设你在做在线代码补全服务，用户每输入一个 token，服务端都要继续生成。标准 attention 需要保留完整 KV cache；RWKV 只需保留每层的 `m,n` 状态，以及 Token Shift 需要的上一个 token 表示。这样会直接影响三件事：

| 工程指标 | 标准 attention | RWKV |
|---|---|---|
| 单会话历史缓存 | 随序列增长 | 固定大小状态 |
| 长会话显存压力 | 越写越大 | 更稳定 |
| 流式逐步更新 | 需要维护 cache 对齐 | 天然按步递推 |

这也是为什么 RWKV 经常被讨论为“更适合流式推理”的结构，而不只是一个公式小改动。

---

## 工程权衡与常见坑

第一类坑，是把 $w$ 写成全局标量。这样做代码最省事，但会破坏 RWKV 最重要的时间尺度分工。

想象两个通道：

- 通道 A 负责段落主题，需要低衰减，也就是忘得慢
- 通道 B 负责最近标点和局部语法，需要高衰减，也就是忘得快

如果两者共享同一个 $w$，那就只能在“都忘得快”或“都忘得慢”之间折中。结果通常是：主题跟不住，或者局部噪声残留太久。RWKV 的实际价值，很大程度上就来自这种通道级时间尺度分化。

第二类坑，是把 Token Shift 的 $\mu$ 固定成常数而不训练。固定常数确实能让模型看到上一 token，但表达能力有限。更合理的做法是：

- 把 $\mu$ 设成可学习参数
- 或让 $\mu$ 依赖当前/前一 token 的内容
- 在 RWKV-6 一类变体里，用更动态的混合方式

第三类坑，是错误理解 WKV 与标准 attention 的关系。WKV 不是“把 attention 原样压缩成两个标量状态”，而是“限制注意力结构为指数衰减记忆后的递归实现”。它保留了很强的序列建模能力，但不代表任何 attention 图样都能等价映射过去。

第四类坑，是状态更新顺序写反。若先拿旧 `m,n` 输出，再把当前 token 累进去，就会产生 off-by-one 错位，也就是“当前步输出实际上少看了当前 token”。

第五类坑，是忽略初始化和边界 token。第一个 token 没有前一个位置，Token Shift 一般会把 $x_0$ 视作零向量，或直接让首 token 只使用自身输入。这里若实现不一致，训练和推理可能出现首位偏移。

第六类坑，是只看公式，不看硬件代价。RWKV 的理论状态是固定大小，但实际吞吐量还取决于 kernel 实现、内存访问模式、张量布局以及是否做了 fused kernel。工程里“状态小”不自动等于“所有硬件上都更快”。

| 常见坑 | 结果 | 规避方式 |
|---|---|---|
| `$w$` 用全局标量 | 所有通道记忆节奏相同 | 使用 per-channel 向量 |
| `$\mu$` 固定常数 | 局部上下文能力弱 | 设为可学习或输入依赖 |
| 忘记数值稳定 | `exp(k)` 溢出 | 使用 log-sum-exp 或平移技巧 |
| 更新顺序错误 | 当前步少用当前 token | 先更新状态再求输出 |
| 首 token 处理不一致 | 序列起点行为异常 | 明确 `x_0` 或首位规则 |
| 把 WKV 当成任意 attention 等价物 | 任务预期过高 | 明确其是衰减记忆结构 |

在实际部署中，RWKV-6 的动态衰减是常见补救。它允许某些输入触发“快速遗忘”。比如文档生成时遇到新标题、空行或章节切换，模型可以让部分通道瞬间加大衰减，避免上一段主题继续污染当前段落。这种行为若只靠固定 $w$，很难学得足够自然。

如果实现动态 $\mu$，通常也要注意更新位置。一个常见做法是先根据 $x_t,x_{t-1}$ 算出新的混合系数，再进行 time mix；不要把 $\mu$ 的更新拖到 WKV 之后，否则会让局部上下文入口滞后一步。

对新手来说，可以把工程检查清单记成下面 6 项：

| 检查项 | 为什么重要 |
|---|---|
| 每通道是否有独立衰减 | 决定多时间尺度能力 |
| 当前 token 是否先写入状态 | 决定是否 off-by-one |
| 首 token 如何做 Token Shift | 决定边界行为 |
| `exp(k)` 是否稳定 | 决定数值是否爆炸 |
| 训练和推理状态更新是否一致 | 决定线上线下是否偏移 |
| 是否真的减少了缓存占用 | 决定 RWKV 的工程价值有没有落地 |

---

## 替代方案与适用边界

最直接的替代方案，当然是标准 Transformer attention。它的优点是表达最自由，每个位置都可以显式对齐历史任意位置；缺点是训练和长上下文推理成本更高，缓存也更重。

另一个替代方向，是各种线性 attention 或状态空间模型。它们和 RWKV 的共同点，都是试图用递归状态替代完整注意力图；不同点在于权重核、状态设计和训练稳定性。

RWKV 适合的场景很明确：

- 流式生成
- 长文本续写
- 低延迟部署
- 受显存约束的在线推理

不那么占优的场景也很明确：

- 需要精确全局对齐的任务
- 多模态里强空间关系建模
- 明显依赖 token 间成对匹配的结构

比如 Vision Transformer 处理图像块关系时，经典 attention 更自然，因为图像中的远距离块之间常常需要显式两两交互；RWKV 的时间递归记忆更适合“一维顺序展开”的建模逻辑。

| 场景 | 标准 attention | RWKV |
|---|---|---|
| 流式文本生成 | 可做，但缓存更重 | 更自然，状态小 |
| 超长上下文部署 | 成本高，KV cache 大 | 状态固定，更易控 |
| 精确 token 对齐 | 更强 | 较弱 |
| 低延迟在线服务 | 取决于缓存与实现 | 通常更有优势 |
| 视觉全局关系 | 常为首选 | 一般不是首选 |

再看工程维度：

| 维度 | 标准 attention | RWKV |
|---|---|---|
| batch 扩展 | 成熟，但显存随 cache 增长 | 状态固定，长序列更友好 |
| caching 难度 | 需要维护 KV cache | 只维护递归状态 |
| chunking | 常需额外处理边界 | 天然适配按步递推 |
| 精度敏感点 | softmax、mask、cache 对齐 | 指数衰减与状态稳定性 |

如果再把常见替代路线放进来，区别会更清楚：

| 路线 | 核心思想 | 优势 | 代价 |
|---|---|---|---|
| 标准 attention | 显式两两对齐 | 表达最自由 | 缓存和计算开销大 |
| 线性 attention | 用核技巧改写注意力 | 可降复杂度 | 表达形式受限 |
| 状态空间模型 | 用连续或离散状态递推 | 长序列友好 | 训练和实现有门槛 |
| RWKV | 指数衰减记忆 + Token Shift | 流式推理自然、状态小 | 精确对齐能力不如全注意力 |

因此，RWKV 不是“取代一切 attention”，而是在“长序列、流式、低延迟”这条线上给出一个非常工程化的答案。若任务核心需求是“记住历史并持续生成”，它很有吸引力；若任务核心需求是“精确找到历史中的某个特定位置并做显式对齐”，标准 attention 往往更稳妥。

一个简单判断标准是：

- 如果你最关心的是“历史能不能一直带着走，而且缓存不要越涨越大”，RWKV 值得优先考虑。
- 如果你最关心的是“当前 token 能不能精确匹配到历史中任意一个关键位置”，标准 attention 通常更直接。

---

## 参考资料

1. **Peng Bo, et al.《RWKV: Reinventing RNNs for the Transformer Era》及相关公开材料**  
   用途：理解 RWKV 的总体动机、WKV 的状态化思想、RNN 模式与 Transformer 模式的统一。  
   建议阅读重点：WKV 递推、time mixing、推理状态设计。

2. **Sergiu Dumitrescu, The Evolution of RWKV Part 2**  
   用途：解释 Token Shift、时间混合、RWKV 早期结构演化。  
   URL: https://sergiudm.github.io/p/the-evolution-of-rwkv-part-2/

3. **Sergiu Dumitrescu, The Evolution of RWKV Part 3**  
   用途：解释 RWKV-6、动态衰减、工程实现上的变化。  
   URL: https://sergiudm.github.io/p/the-evolution-of-rwkv-part-3/

4. **RWKV-LM 官方或社区实现仓库**  
   用途：查看不同版本对 WKV、time mix、数值稳定、CUDA kernel 的具体落地方式。  
   检索关键词：`BlinkDL RWKV-LM`, `RWKV v4`, `RWKV v5`, `RWKV v6`

5. **线性 attention / 状态空间模型综述材料**  
   用途：把 RWKV 放回更大的“长序列建模”背景里理解，避免把它误解成孤立技巧。  
   建议阅读重点：递归状态、核化注意力、长程依赖建模的不同路线。

6. **本文建议的阅读顺序**  
   第一步：先看 WKV 公式和递推推导，确认“为什么能只维护两个状态”。  
   第二步：再看 Token Shift，理解“为什么还需要补一个局部入口”。  
   第三步：最后看 RWKV-6 的动态衰减，理解“为什么固定遗忘速度不够”。
