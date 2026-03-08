## 核心结论

GLA，Gated Linear Attention，直译是“带门控的线性注意力”。它不是重新定义注意力，而是在**线性注意力的递推状态**上加入一个可学习的遗忘门，让模型在保持线性复杂度的同时，获得“该记什么、该忘什么”的能力。

它的最小形式可以写成：

$$
S_t = g_t S_{t-1} + k_t v_t^\top,\qquad o_t = q_t S_t
$$

其中：

- $S_t$ 是第 $t$ 步的状态，可以理解成“到当前为止的压缩记忆”。
- $k_t v_t^\top$ 是一次 rank-1 写入，也就是把当前 token 产生的信息写进状态。
- $q_t$ 是读头，用来从状态里读出当前 token 需要的上下文。
- $g_t \in (0,1)$ 是遗忘门，用来决定旧状态保留多少。

很多实现里，门控并不是固定超参数，而是由当前输入动态计算：

$$
g_t=\sigma(f(q_t,k_t))
$$

这里 $\sigma$ 是 sigmoid，把任意实数压到 $(0,1)$；$f$ 一般是轻量线性层或小 MLP。它的作用不是增加很多参数，而是给状态更新加一根“可学习的控制杆”。

如果只用一句话概括 GLA，可以写成：

> GLA 在线性注意力的递推记忆上加入显式遗忘门，让模型在长序列里既能保留线性时间和小缓存优势，又比普通线性注意力更会管理历史信息。

它的重要性主要体现在下面四点：

| 维度 | 普通 Softmax 注意力 | 普通线性注意力 | GLA |
|---|---|---|---|
| 训练主复杂度 | 通常随序列长度近似二次增长 | 线性递推，可做线性复杂度实现 | 保持线性递推 |
| 是否有显式“遗忘” | 没有单独门控 | 通常没有 | 有，$g_t$ 可学习 |
| 长上下文状态携带 | 依赖全量注意力矩阵 | 依赖累计状态 | 依赖累计状态且可控衰减 |
| GPU I/O 优化空间 | 很高，FlashAttention 系列已成熟 | 早期实现常被 I/O 拖慢 | 可借助 chunk-wise 与 fused kernel 提升 |
| 推理缓存形式 | KV cache | 递推状态 | 带门控的递推状态 |

这里“更有表达力”的意思需要说清楚。普通线性注意力通常把历史不断累加到状态里，旧信息是否继续影响未来，只能靠投影和数值比例间接决定；GLA 则多了一个显式门控，模型可以在某些位置保留长期记忆，在另一些位置主动衰减无关历史。这种能力对长上下文尤其重要，因为长序列里的噪声、主题切换、段落边界都很多。

一个直观例子是文档问答。假设前 1000 个 token 在讨论数据库事务，后 1000 个 token 转到 GPU kernel 调优。普通线性注意力可能把前半段内容一直累积在状态里；GLA 则可以在主题切换处把门控调低，减少“旧主题残留”污染当前读出。

所以结论不是“GLA 一定全面替代 Softmax 注意力”，而是：

1. 它给线性注意力补上了“选择性遗忘”。
2. 它仍然保留递推状态和线性复杂度。
3. 它在长上下文、受 I/O 限制的训练和推理中更有工程价值。

---

## 问题定义与边界

GLA 要解决的问题很具体：

> 如何在不回到 $O(n^2)$ 注意力成本的前提下，让模型对长上下文有更强的记忆管理能力。

这个问题有两个关键词。

第一是“**不回到 $O(n^2)$**”。  
标准自注意力需要计算大量 token 两两交互，序列越长，代价越高。即使有 FlashAttention 这类高效实现，底层仍然是在处理完整注意力语义，只是把访存做得更高效。

第二是“**记忆管理能力**”。  
很多线性注意力方法的问题不在于不能压缩历史，而在于压缩后缺少显式控制。它能记，但不一定会忘；能累加，但不一定会筛选。

这里讨论的边界也要说清楚。本文默认场景是：

1. 因果语言模型里的**单向注意力层**。
2. 序列长度通常在 **2K token 以上**。
3. 目标是把标准注意力替换成**线性递推层**。
4. 硬件主要是 GPU，且性能瓶颈常常来自显存读写，而不是纯算力。
5. 实现上愿意采用 Triton、CUDA 或类似方式做定制 kernel。

这意味着，GLA 不是在所有地方都值得上。比如：

| 场景 | 是否适合 GLA | 原因 |
|---|---|---|
| 256 或 512 长度的小模型训练 | 通常不明显 | 短序列下二次复杂度还没成为主要瓶颈 |
| 需要严格保留标准注意力语义的复现实验 | 不一定 | GLA 本质上是替代层，不是 Softmax 的无损等价物 |
| 流式推理、长上下文生成 | 适合 | 递推状态缓存小，门控有助于长期稳定 |
| 长文档建模、长代码补全 | 适合 | 既要省缓存，又要能控制历史污染 |
| 团队不维护定制 kernel | 边际收益有限 | 纸面线性复杂度不自动等于真实吞吐优势 |

对初学者来说，最容易误解的一点是：**理论复杂度线性，不代表 GPU 一定更快。**

原因在于 GPU 上真正贵的不只是算术运算，还有 I/O。更准确地说，是 HBM（显存）和片上 SRAM / register 之间的数据搬运。如果一个方法虽然公式上是 $O(n)$，但每一步都把中间状态写回显存，再从显存读出来继续算，那么实际速度可能比高质量实现的 Softmax 注意力还差。

因此，GLA 的实际问题定义应当写成“三目标同时成立”：

| 目标 | 要求 |
|---|---|
| 表达力 | 比普通线性注意力更会保留或遗忘历史 |
| 复杂度 | 仍然保持递推式的线性时间与紧凑缓存 |
| 硬件效率 | 减少 global memory materialization，尽量在片上完成更新与读出 |

这里的 `materialization` 可以直接理解为：  
**是否把中间状态真的写到全局显存。**

举一个具体工程设定。假设序列长度为 8192，chunk size 取 2048，那么整条序列分成 4 个块。块内可以并行算，块间通过递推状态 $S$ 连接。如果你把块内每个时间步的状态都落到显存，I/O 会非常重；如果你只保留块边界状态，或只保留反向传播需要的少量锚点，吞吐量通常会好很多。

所以 GLA 的关键不是“门控公式写出来了没有”，而是下面三件事能否同时做好：

1. 用门控提升状态表达力。
2. 用递推形式维持线性时间和小缓存。
3. 用 chunk + fused kernel 避免 I/O 把理论优势吃掉。

---

## 核心机制与推导

先从最小递推式开始。对第 $t$ 个 token，GLA 做三件事：

1. 生成写入项 $k_t v_t^\top$
2. 计算遗忘门 $g_t$
3. 从新状态读出输出

对应公式是：

$$
S_t = g_t S_{t-1} + k_t v_t^\top
$$

$$
o_t = q_t S_t
$$

$$
g_t=\sigma(f(q_t,k_t))
$$

先解释符号，避免术语堆叠：

| 符号 | 含义 | 初学者可理解成 |
|---|---|---|
| $q_t$ | query | 当前 token 用来“提问”的向量 |
| $k_t$ | key | 当前 token 写入状态时使用的“地址方向” |
| $v_t$ | value | 当前 token 要写入的内容 |
| $S_t$ | state | 截止到第 $t$ 步的压缩记忆 |
| $g_t$ | gate | 旧记忆保留比例 |
| $o_t$ | output | 当前步从状态中读出的结果 |

### 1. 为什么写入项是 $k_t v_t^\top$

这是一个外积。若 $k_t \in \mathbb{R}^{d_k}$，$v_t \in \mathbb{R}^{d_v}$，那么：

$$
k_t v_t^\top \in \mathbb{R}^{d_k \times d_v}
$$

它是一个矩阵，但不是任意矩阵，而是 rank-1 矩阵。rank-1 的意思是：这次写入非常“薄”，成本低，结构也简单。你可以把它理解成：

- $k_t$ 决定往哪一类方向写。
- $v_t$ 决定写进去的内容是什么。

如果每步都做一次这种写入，状态矩阵 $S_t$ 就像一块不断被更新的记忆板。

### 2. 如果没有门控，会发生什么

普通线性注意力的递推一般近似成：

$$
S_t=S_{t-1}+k_t v_t^\top
$$

也就是**只累加，不遗忘**。

这会带来两个直接问题：

1. 旧信息会长期残留，历史越长，状态越容易被“写满”。
2. 当上下文主题频繁切换时，旧内容可能持续干扰当前读出。

GLA 把它改成：

$$
S_t = g_t S_{t-1} + k_t v_t^\top
$$

于是：

- 当 $g_t \approx 1$，旧记忆几乎完整保留。
- 当 $g_t \approx 0$，旧记忆大幅衰减。
- 当 $g_t$ 由输入动态决定，模型就能学“什么时候忘、忘多少”。

### 3. 一个最小数值例子

假设上一时刻状态是：

$$
S_{t-1}=0.8
$$

当前门控：

$$
g_t=0.5
$$

当前写入：

$$
k_t v_t^\top=0.3
$$

那么新状态就是：

$$
S_t=0.5\times 0.8+0.3=0.7
$$

若当前查询是：

$$
q_t=1.2
$$

则输出为：

$$
o_t=1.2\times 0.7=0.84
$$

这个例子虽然是标量，但已经把 GLA 的三步表达完整了：

1. 先保留一部分旧状态。
2. 再写入当前信息。
3. 最后用查询去读。

### 4. 为什么这仍然是“线性注意力”

标准 Softmax 注意力通常写成：

$$
\text{softmax}(QK^\top)V
$$

这里本质上要处理大量 token 对之间的关系。GLA 不这样做。它把历史压缩进状态 $S_t$，每来一个 token 就递推一次，因此时间和缓存都更接近线性。

但这种“压缩历史”的代价是：  
状态并不保留每个 token 与每个 token 的完整交互细节。换句话说，它用结构性压缩换掉了全量注意力矩阵。

GLA 的补偿方式就是门控。它做不了完整注意力的所有事情，但它比“纯累加式线性注意力”更有选择性，因此在很多长序列任务上效果更稳。

### 5. 把门控写得更具体

一种常见写法是：

$$
g_t=\sigma(W_g[q_t;k_t]+b_g)
$$

其中 $[q_t;k_t]$ 表示拼接。若按 head 独立计算，$g_t$ 还可以是向量而不是标量，例如：

$$
g_t \in (0,1)^{d_k}
$$

这表示不同通道有不同遗忘率。这样更灵活，但实现和数值控制也更复杂。

门控网络不宜过重，原因很简单：

| 设计 | 结果 |
|---|---|
| 门控很轻 | 保留 GLA 的递推和 I/O 优势 |
| 门控很重 | 额外算力与访存会吃掉收益 |
| 门控过深 | 训练和 kernel 融合都更麻烦 |

所以工程上常见原则不是“门控越复杂越强”，而是：

> 门控要足够表达，但不能重到破坏线性注意力的硬件优势。

### 6. 从记忆角度再看一遍

如果把状态 $S_t$ 当作一块白板，那么每一步发生的是：

- 旧白板先按比例擦掉一部分：$g_t S_{t-1}$
- 当前 token 再写入一笔：$k_t v_t^\top$
- 当前查询从白板上读出需要的信息：$q_t S_t$

这种解释对初学者很有效，因为它抓住了 GLA 的本质：  
**它不是让每个 token 回头看全历史，而是让历史先被压缩成一个可控的状态，再从状态里读。**

---

## 代码实现

下面先给一个**可以直接运行**的 Python 玩具实现。它不追求性能，只用来说明 GLA 的递推逻辑、张量形状和最小验证方式。

### 1. 标量版：先把公式跑通

```python
import math
from typing import List, Tuple


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def gla_scalar_sequence(
    qs: List[float],
    ks: List[float],
    vs: List[float],
) -> Tuple[List[float], List[float], float]:
    """
    标量版 GLA:
        S_t = g_t * S_{t-1} + k_t * v_t
        o_t = q_t * S_t
        g_t = sigmoid(q_t + k_t)

    返回:
        outputs: 每一步输出
        gates:   每一步 gate
        state:   最终状态
    """
    if not (len(qs) == len(ks) == len(vs)):
        raise ValueError("qs, ks, vs must have the same length")

    state = 0.0
    outputs = []
    gates = []

    for q, k, v in zip(qs, ks, vs):
        g = sigmoid(q + k)
        state = g * state + k * v
        out = q * state
        gates.append(g)
        outputs.append(out)

    return outputs, gates, state


def main() -> None:
    # 手工例子验证
    prev_state = 0.8
    g = 0.5
    write = 0.3
    state = g * prev_state + write
    out = 1.2 * state

    assert abs(state - 0.7) < 1e-9
    assert abs(out - 0.84) < 1e-9

    # 序列验证
    outputs, gates, final_state = gla_scalar_sequence(
        qs=[0.2, 1.2, -0.1],
        ks=[0.1, 0.5, 0.3],
        vs=[1.0, 0.6, -0.4],
    )

    assert len(outputs) == 3
    assert all(0.0 < gate < 1.0 for gate in gates)
    assert isinstance(final_state, float)

    print("outputs =", outputs)
    print("gates   =", gates)
    print("state   =", final_state)


if __name__ == "__main__":
    main()
```

这个版本的意义只有一个：把“写入、遗忘、读出”三步拆开，让你看清公式是怎么落成代码的。

### 2. 矩阵版：更接近真实模型

真实模型里，$q_t,k_t,v_t$ 都是向量，$S_t$ 是矩阵。下面给一个依赖 `numpy` 的最小可运行版本：

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gla_numpy(q, k, v, wg, bg, state0=None):
    """
    q: [T, Dk]
    k: [T, Dk]
    v: [T, Dv]
    wg: [2*Dk, 1]   简化为标量 gate
    bg: [1]
    state0: [Dk, Dv]

    returns:
        outputs: [T, Dv]
        gates:   [T]
        state:   [Dk, Dv]
    """
    T, Dk = q.shape
    Tv, Dv = v.shape
    if T != Tv or k.shape != (T, Dk):
        raise ValueError("shape mismatch")

    if state0 is None:
        state = np.zeros((Dk, Dv), dtype=np.float64)
    else:
        state = state0.astype(np.float64).copy()

    outputs = []
    gates = []

    for t in range(T):
        gate_in = np.concatenate([q[t], k[t]], axis=0)   # [2*Dk]
        g = float(sigmoid(gate_in @ wg[:, 0] + bg[0]))   # 标量 gate

        write = np.outer(k[t], v[t])                     # [Dk, Dv]
        state = g * state + write
        out = q[t] @ state                               # [Dv]

        outputs.append(out)
        gates.append(g)

    return np.stack(outputs, axis=0), np.array(gates), state


def demo():
    T, Dk, Dv = 4, 3, 2
    rng = np.random.default_rng(0)

    q = rng.normal(size=(T, Dk))
    k = rng.normal(size=(T, Dk))
    v = rng.normal(size=(T, Dv))
    wg = rng.normal(size=(2 * Dk, 1))
    bg = np.zeros((1,))

    outputs, gates, state = gla_numpy(q, k, v, wg, bg)

    assert outputs.shape == (T, Dv)
    assert gates.shape == (T,)
    assert state.shape == (Dk, Dv)
    assert np.all(gates > 0.0) and np.all(gates < 1.0)

    print("outputs shape:", outputs.shape)
    print("gates:", gates)
    print("final state shape:", state.shape)


if __name__ == "__main__":
    demo()
```

这个实现已经包含了 GLA 的核心结构：

- `np.outer(k[t], v[t])` 对应 $k_t v_t^\top$
- `state = g * state + write` 对应递推状态更新
- `out = q[t] @ state` 对应读出

### 3. 为什么真实实现不会逐 token 用 Python for 循环

上面代码适合理解，不适合训练。原因不是公式错，而是执行方式太慢。

真实 GPU 实现通常不会让 Python 在时间维上逐 token 驱动，而是采用：

1. **块内并行**
2. **块间递推**
3. **状态更新与读出融合**
4. **尽量不把中间状态写回显存**

伪代码可以写成：

```python
def gla_chunk_forward(q, k, v, chunk_size, state0):
    state = state0
    outputs = []

    for chunk_start in range(0, len(q), chunk_size):
        qc = q[chunk_start:chunk_start + chunk_size]
        kc = k[chunk_start:chunk_start + chunk_size]
        vc = v[chunk_start:chunk_start + chunk_size]

        gc = gate_net(qc, kc)

        chunk_out, state = fused_chunk_kernel(
            q=qc,
            k=kc,
            v=vc,
            g=gc,
            state=state,
            materialize_intermediate=False,
        )

        outputs.append(chunk_out)

    return concat(outputs), state
```

这段伪代码里最重要的不是语法，而是三个工程点。

### 4. 工程点一：为什么要 fused kernel

如果你这样做：

1. 先更新状态
2. 把状态写回显存
3. 再开一个 kernel 读状态
4. 再计算输出

那么一次前向就会发生多次不必要的 HBM 往返。  
所以更好的做法是把下面两步融合：

- `state = g * state + outer(k, v)`
- `out = q @ state`

也就是说，在寄存器或 shared memory 里更新状态后，立刻完成读出，而不是先落盘再读。

### 5. 工程点二：什么叫 materialization

`materialize_intermediate=False` 的含义是：  
**不把每个时间步的中间状态都写到全局显存。**

常见策略如下：

| 策略 | 做法 | 优点 | 代价 |
|---|---|---|---|
| 不物化中间态 | 只保留最终块状态 | I/O 最省 | 反向传播需更多重算 |
| 只物化块边界 | 每个 chunk 结束存一次 | 训练时常见折中 | 仍需少量状态存储 |
| 全物化块内状态 | 每步都存 | 调试简单 | I/O 很重，通常最慢 |

对训练来说，常见折中是“只存块边界 + 必要重算”。  
对推理来说，更常见的是“只维护当前递推状态”。

### 6. 工程点三：chunk size 怎么看

chunk 不是越大越好，也不是越小越好。

| chunk 太小 | chunk 太大 |
|---|---|
| kernel 启动开销占比高 | 寄存器压力和块内计算压力变大 |
| I/O 优化难体现 | occupancy 可能下降 |
| GPU 利用率不够 | 内核调优更困难 |

实践中常从 `64/128/256` 一类值开始试。  
很多实现里，`128` 或 `256` 比 `64` 更容易跑出稳定吞吐；但最终值取决于 head 维度、状态布局、GPU 架构和反向实现方式。

### 7. 一个更贴近训练的思路

假设你在训练一个 4K context 的语言模型，并将一层标准注意力替换为 GLA，那么一层前向通常会这样组织：

1. 先从输入投影得到 $Q,K,V$
2. 从 $Q,K$ 计算 gate logits，再经 sigmoid 得到 $g$
3. 按 chunk 扫过序列
4. 块内做状态更新和输出读出
5. 块间只传递紧凑的状态矩阵，而不是整段 KV cache

好处是：

- 缓存从“按 token 存 KV”变成“存递推状态”
- 长度增大时，内存压力上升更慢
- 如果 kernel 融合做得好，I/O 可显著下降

这也是为什么 GLA 常与 `fused recurrent`、`fused chunk` 这类术语一起出现。它们讨论的不是公式本身，而是**怎样把公式映射到 GPU 友好的执行路径**。

---

## 工程权衡与常见坑

GLA 的难点不在公式，而在工程实现。很多失败案例不是“理论不成立”，而是“实现把优势做没了”。

先看最常见的坑：

| 坑 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| gate 长期偏高 | 旧上下文几乎不衰减，状态越来越黏 | 模型学成近似恒等保留 | 调整 gate 初始化，监控分布 |
| gate 长期偏低 | 模型像短记忆滤波器，长程依赖丢失 | 过度遗忘 | 检查 gate bias，避免初始过小 |
| chunk 太小 | Triton kernel 吞吐下降 | 启动开销和 I/O 比例过高 | 从 128 或 256 起调 |
| 频繁 materialization | 理论线性但实际很慢 | HBM 写回过多 | 只物化块边界或必要锚点 |
| gate 网络过重 | MLP 吃掉收益 | 计算与访存膨胀 | 保持门控轻量 |
| 状态尺度不稳 | 输出波动大，训练不稳 | 状态累积数值范围失控 | 配合归一化、缩放和稳定初始化 |

下面把几个最关键的问题展开。

### 1. gate 分布失控是最常见问题

很多人以为“门控交给模型自己学就行”，但训练初期如果 bias 或尺度不合适，门控很容易塌到极端区域。

若 sigmoid 输入过大：

$$
g_t \approx 1
$$

则模型几乎不忘记，GLA 会退化成“黏住历史的累加器”。

若 sigmoid 输入过小：

$$
g_t \approx 0
$$

则模型几乎每步都在重置状态，GLA 会退化成“只有短期记忆的滤波器”。

因此训练时建议至少监控：

| 监控项 | 含义 |
|---|---|
| gate 的均值 | 整体更偏保留还是更偏遗忘 |
| gate 的分位数 | 是否大量挤在 0 或 1 附近 |
| 不同层的 gate 分布 | 浅层和深层是否出现职责分化 |
| 长序列上的 gate 漂移 | 序列后段是否系统性失衡 |

对初学者来说，可以把这件事理解成：  
**GLA 能不能工作，很大程度上取决于门控有没有真正学出“动态遗忘”，而不是一直开着或一直关着。**

### 2. 数值稳定性不能只看公式

递推状态长期累积后，尺度容易出问题。因为：

$$
S_t = g_t S_{t-1} + k_t v_t^\top
$$

即使 $g_t \in (0,1)$，如果 $k_t$ 和 $v_t$ 的尺度控制不好，状态仍可能偏大或偏小。

常见缓解方式包括：

1. 对 $q,k,v$ 做合适缩放。
2. 对状态读出前后配合 normalization。
3. 让 gate 的初始分布落在中间区，而不是一开始就饱和。
4. 在 kernel 实现里注意累积精度，比如 FP16/BF16 下的中间计算策略。

如果只从论文公式抄实现，但不处理尺度问题，训练常会表现为：

- loss 抖动大
- 深层状态异常放大
- 某些 head 几乎失活
- 长长度外推变差

### 3. chunk 是硬件参数，不是数学参数

很多人第一次接触 chunk，会觉得它只是把长序列切段。这个理解不够。

更准确地说，chunk 是在平衡下面这个近似关系：

$$
\text{总时间} \approx \text{计算时间} + \text{HBM 访存时间} + \text{kernel 调度开销}
$$

- chunk 太小：调度开销高，GPU 吃不满。
- chunk 太大：寄存器压力和块内计算量增大。
- materialization 太多：HBM 访存时间占主导。

所以调 chunk 的目的不是改变模型语义，而是让实现落在更好的硬件工作点。

### 4. 反向传播常常比前向更难做对

前向看起来只有一条递推链，反向却涉及：

- 状态如何回传梯度
- 哪些中间量要缓存
- 哪些中间量可以重算
- 哪些状态必须在块边界保留

如果训练时把每一步状态全保存，显存会很难看；如果一项都不存，反向重算成本又可能太高。  
因此很多高质量实现采用“块边界保存 + 块内重算”的折中，这和 FlashAttention 系列通过重算换显存的思路是类似的。

### 5. GLA 不是一换就更快

这一点必须单独强调。若场景是：

- 序列不长
- batch 不大
- GPU 对 FlashAttention 已高度优化
- 工程团队没有定制 kernel

那么标准 Softmax 注意力可能依然是更好的选择。

GLA 真正占优的区域通常是：

1. 上下文足够长。
2. I/O 已经成为主要瓶颈。
3. 你愿意投入 kernel 调优。
4. 你既想要线性递推缓存，又不想失去太多表达力。

---

## 替代方案与适用边界

如果目标只是“更快的注意力”或“更小的缓存”，GLA 不是唯一答案。它的特点是平衡得比较均匀，但不是所有维度都最强。

先放在同一张表里比较：

| 方案 | 吞吐潜力 | 表达力 | 缓存需求 | 适用边界 |
|---|---|---|---|---|
| GLA | 高，尤其长上下文 | 高于普通线性注意力 | 存递推状态 | 2K+ 长度，重视长程泛化与 I/O |
| FlashAttention-2 | 很高，短中长序列都成熟 | 强 | KV cache | 标准 Transformer 训练推理 |
| 传统 Softmax 注意力 | 依赖实现，长序列成本高 | 强 | KV cache 大 | 序列不长或必须保留标准注意力语义 |
| 普通线性注意力 | 高 | 中等，缺少显式遗忘 | 存递推状态 | 更在意线性复杂度，能接受表达力折中 |
| RWKV / fused recurrent 风格 | 很高，递推友好 | 取决于具体门控结构 | 很小 | 流式推理、小缓存场景 |

下面按“你真正关心的问题”来选。

### 1. 如果你最在意成熟生态

优先看 FlashAttention-2 配标准 Transformer。

原因是：

- 工程生态成熟
- 复现路径清晰
- 训练行为更容易与已有模型对齐
- 不需要重新适应新的注意力语义

代价是长上下文下，KV cache 和二次交互成本仍然更重。

### 2. 如果你最在意线性复杂度，但想先走最简单路线

普通线性注意力是更低门槛的起点。  
它的优势是结构简单、缓存小；问题是没有显式遗忘，长期状态管理往往不如 GLA。

可以把两者关系理解成：

- 普通线性注意力：先把“线性递推”做出来
- GLA：再给这个递推加上“选择性遗忘”

### 3. 如果你最在意流式推理和极小缓存

RWKV 或其他 fused recurrent 风格方法值得看。  
它们通常对流式推理很友好，缓存非常小，但模型结构和 Transformer 家族并不完全同构，迁移成本更高。

### 4. GLA 的适用边界

GLA 适合下面两类项目：

1. 你要在单 GPU 或有限 GPU 资源下处理 2K+ token，且不希望 KV cache 继续按长度线性膨胀。
2. 你希望比普通线性注意力更会管理长期历史，同时愿意为此维护一定的 kernel 工程复杂度。

它不太适合下面两类项目：

1. 序列很短，标准注意力已经足够便宜。
2. 团队不准备维护 Triton/CUDA 定制实现，只想依赖现成通用算子。

因此，GLA 更像一个**长上下文工程选项**，不是“所有模型都应该默认替换成 GLA”的普适答案。

---

## 参考资料

1. Yang, Wang, Shen, Panda, Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training*. PMLR 235, 2024. https://proceedings.mlr.press/v235/yang24ab.html  
用途：核心定义、状态更新公式、硬件友好训练思路、长长度外推实验结论。适合对应本文“核心结论”“核心机制与推导”“工程权衡与常见坑”。

2. Emergent Mind, *Gated Linear Attention*. https://www.emergentmind.com/topics/gated-linear-attention  
用途：整理 GLA 的递推形式、chunkwise 并行、materialization、硬件执行路径等概念。适合对应本文“问题定义与边界”“代码实现”“工程权衡与常见坑”。

3. `rwkv-fla` PyPI 文档. https://pypi.org/project/rwkv-fla/0.1.202411240422/  
用途：帮助理解 `FusedRecurrent`、`FusedChunk`、`ParallelChunk` 等工程实现术语，以及是否把中间态 materialize 到 global memory 的取舍。适合对应本文“代码实现”“工程权衡与常见坑”。

4. 补充阅读建议：如果要把 GLA 放回更大的技术背景中，可以对照阅读 FlashAttention 系列论文或实现说明。  
用途：理解为什么“理论复杂度更低”不等于“GPU 一定更快”，以及为什么 I/O-aware kernel 设计会直接决定长序列训练吞吐。
