## 核心结论

RetNet 的核心不是“把 Attention 改快一点”，而是把“序列历史”改写成一个可以递推维护的状态。这个状态通常记作 $S_n$，可以理解为“到第 $n$ 个 token 为止，模型保留下来的记忆摘要”。

它的基本递推式是：

$$
S_n = \gamma S_{n-1} + K_n^\top V_n,\quad \gamma \in (0,1)
$$

对应输出是：

$$
O_n = Q_n S_n
$$

这里：

- $Q,K,V$ 分别是 query、key、value，也就是“当前查询”“当前内容索引”“当前内容值”。
- $\gamma$ 是衰减系数，也就是“旧记忆每走一步还保留多少”。
- $K_n^\top V_n$ 是本步新增的信息，可以理解为“当前 token 对状态的增量写入”。

RetNet 的关键价值有三点：

1. 训练时可以并行展开整个序列，保留类似 Transformer 的高吞吐训练方式。
2. 推理时可以只维护状态 $S$，逐 token 更新，单步额外状态复杂度接近 $O(1)$。
3. 不同注意力头使用不同的 $\gamma$，形成多尺度记忆。小 $\gamma$ 偏短期，大 $\gamma$ 偏长期。

一个最直观的玩具例子是三个头分别取 $\gamma=\{0.5,0.8,0.98\}$。同样看到一段历史后：

- $\gamma=0.5$ 的头很快忘掉旧信息，适合局部模式。
- $\gamma=0.8$ 的头保留中等长度上下文。
- $\gamma=0.98$ 的头几乎持续携带旧状态，适合长距离依赖。

这就是 RetNet 所说的 multi-scale retention，多尺度保持机制。

---

## 问题定义与边界

RetNet 要解决的问题很具体：能不能在不依赖 softmax 注意力的前提下，同时做到“训练可并行”和“推理可递推”。

标准 softmax attention 的核心问题不是表达能力不够，而是推理时每个新 token 都要回看全部历史缓存。序列越长，缓存越大，访存和延迟越重。RetNet 试图把“回看历史矩阵”改成“更新一个状态”。

因此，本文讨论的边界不是“RetNet 是否全面取代 Transformer”，而是下面这个更窄的问题：

> 当任务需要长序列建模，而且部署阶段非常依赖逐 token 推理时，RetNet 的指数衰减递推是否能提供更统一的训练/推理路径？

这里有几个边界必须先说清：

| 模式 | 核心做法 | 典型复杂度 | 并行度 | 是否依赖前一时刻状态 |
|---|---|---:|---|---|
| Parallel | 整段序列一次性展开计算 | $O(N^2)$ | 高 | 否 |
| Chunk-wise | 块内并行，块间递推 | 近似 $O(C^2)$ 块内 + 块间状态传递 | 中 | 是 |
| Recursive | 逐 token 更新 $S_n$ | 单步额外状态更新接近 $O(1)$ | 低 | 是 |

表里有两个容易混淆的点：

- `Parallel` 不是说 RetNet 本质回到了普通 attention，而是说它存在一个与递推形式等价的并行展开写法。
- `Recursive` 的 $O(1)$ 指的是“相对历史长度的增量状态成本”，不是整个网络真实 FLOPs 变成常数。

对白话一点的理解可以这样说：

- $K_n^\top V_n$ 是“每一步新写入的记忆片段”。
- $\gamma$ 是“旧记忆每过一步要衰减多少”。
- RetNet 要做的是：让训练时能批量算，让推理时能只维护状态，而且两者结果尽量一致。

因此，RetNet 的成败不只在公式是否优雅，更在于三种计算范式能否在工程上保持一致。

---

## 核心机制与推导

先看最小递推。

设单个头在第 $n$ 步的状态是 $S_n$，初始 $S_0=0$。那么：

$$
S_n = \gamma S_{n-1} + K_n^\top V_n
$$

把它展开：

$$
S_n = \sum_{i=1}^{n} \gamma^{n-i} K_i^\top V_i
$$

这说明第 $i$ 步写入的信息，在第 $n$ 步的权重是 $\gamma^{n-i}$。也就是说，离当前越远，权重越小，而且是指数衰减。

这和 softmax attention 的区别在于：

- softmax 是“对当前 query 动态归一化地选择历史”。
- retention 是“先把历史按固定衰减压进状态，再由 query 去读这个状态”。

这里的“状态”不是抽象概念，而是一个真实张量。它像一个被持续更新的记忆缓存，但这个缓存不是保存全部 token，而是保存加权后的汇总。

### 玩具例子

取 $\gamma=0.8$，并沿用一个最小标量例子。注意，这里只是为了看清递推，不代表真实模型维度。

假设：

- $S_0=0$
- 第一步写入 $K_1^\top V_1 = 2$
- 第二步写入 $K_2^\top V_2 = 3$

则：

$$
S_1 = 0.8 \cdot 0 + 2 = 2
$$

如果某些实现把衰减先作用到当前等价展开里，也可能写成文献里常见的变体记号；本文统一使用上面的标准递推。继续第二步：

$$
S_2 = 0.8 \cdot 2 + 3 = 4.6
$$

若当前查询是标量 $Q_2=0.5$，则输出：

$$
O_2 = Q_2 S_2 = 0.5 \times 4.6 = 2.3
$$

这个例子说明两件事：

1. 旧信息没有消失，而是按指数衰减继续存在。
2. 输出不是直接取最近 token，而是从累计状态中读取。

如果换成多个头，每个头有不同的 $\gamma_h$，那么：

$$
S_n^{(h)} = \gamma_h S_{n-1}^{(h)} + K_n^{(h)\top}V_n^{(h)}
$$

这就是多尺度来源。不同头的时间常数不同，模型会自然分工：

- 短程头跟踪局部模式，比如最近几个 token 的搭配。
- 长程头保留远距离依赖，比如较早的定义、变量约束、章节主题。

从信号处理角度看，这很像一组不同时间常数的滤波器。

### 为什么三种模式能等价

关键在于上面的展开式：

$$
S_n = \sum_{i=1}^{n} \gamma^{n-i} K_i^\top V_i
$$

如果你一次性知道整段序列，就能把所有项并行算出来，这就是 parallel。

如果你把序列分块，每块内部展开并行，块与块之间只传递边界状态，这就是 chunk-wise。

如果你逐 token 到来，就按递推式更新一次，这就是 recursive。

三者的数学对象相同，只是计算顺序不同。

可以把它画成一条状态链：

$$
S_0 \xrightarrow{\gamma,+K_1^\top V_1} S_1 \xrightarrow{\gamma,+K_2^\top V_2} S_2 \xrightarrow{\gamma,+K_3^\top V_3} \cdots \xrightarrow{} S_n
$$

每走一步：

- 先把旧状态乘上 $\gamma$
- 再把当前写入项加进去

### 与 RoPE 结合的直觉

RoPE 是 Rotary Position Embedding，白话解释就是“用旋转的方式把位置信息编码进向量”。它的好处是位置关系可以通过相对相位表达。

RetNet 与 RoPE 结合时，可以把简单实数衰减扩展到带旋转相位的特征空间。直觉上，相当于：

- 模长部分由 $\gamma$ 控制“记忆保留程度”
- 相位部分由旋转编码控制“位置信息演化”

文献里常把这种形式写成更一般的特征值视角：状态更新不只是实数缩放，也可以是带复数结构的线性变换。对初学者而言，不必先抓复数形式，先抓住“衰减决定记忆长度，旋转决定位置表达”就够了。

---

## 代码实现

下面给一个可运行的最小 Python 实现。它不是完整 RetNet，而是把“parallel 展开”和“recursive 递推”放在同一套标量/小矩阵语义下，验证两者一致。

```python
import numpy as np

def recursive_retention(Q, K, V, gamma):
    """
    Q: [N, d]
    K: [N, d]
    V: [N, m]
    return: [N, m]
    """
    N, d = Q.shape
    m = V.shape[1]
    S = np.zeros((d, m), dtype=np.float64)
    outputs = []

    for n in range(N):
        kv = np.outer(K[n], V[n])   # K_n^T V_n
        S = gamma * S + kv
        O = Q[n] @ S
        outputs.append(O)

    return np.stack(outputs, axis=0)

def parallel_retention(Q, K, V, gamma):
    N, d = Q.shape
    m = V.shape[1]
    outputs = np.zeros((N, m), dtype=np.float64)

    for n in range(N):
        S_n = np.zeros((d, m), dtype=np.float64)
        for i in range(n + 1):
            S_n += (gamma ** (n - i)) * np.outer(K[i], V[i])
        outputs[n] = Q[n] @ S_n

    return outputs

def chunk_retention(Q, K, V, gamma, chunk_size):
    N, d = Q.shape
    m = V.shape[1]
    S = np.zeros((d, m), dtype=np.float64)
    outputs = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)

        # 块内每个位置都从块外传入的同一个初始状态开始
        for n in range(start, end):
            kv = np.outer(K[n], V[n])
            S = gamma * S + kv
            outputs.append(Q[n] @ S)

    return np.stack(outputs, axis=0)

# 玩具输入
Q = np.array([[1.0, 0.0],
              [0.5, 1.0],
              [1.0, -1.0]])

K = np.array([[1.0, 2.0],
              [0.0, 1.0],
              [1.0, 1.0]])

V = np.array([[2.0, 1.0],
              [3.0, 0.0],
              [1.0, 4.0]])

gamma = 0.8

out_rec = recursive_retention(Q, K, V, gamma)
out_par = parallel_retention(Q, K, V, gamma)
out_chk = chunk_retention(Q, K, V, gamma, chunk_size=2)

assert np.allclose(out_rec, out_par, atol=1e-10)
assert np.allclose(out_rec, out_chk, atol=1e-10)

print("recursive output:\n", out_rec)
print("parallel output:\n", out_par)
print("chunk output:\n", out_chk)
```

这段代码验证了一个核心事实：只要定义一致、归一化一致、分块边界处理一致，三种模式应该得到相同结果。

如果把它映射到真实工程中的 `MultiScaleRetention` 模块，最常见结构是：

```python
# 伪代码
states = [zeros_for_head(h) for h in heads]

for chunk in sequence_chunks:
    for h in heads:
        gamma = gamma_per_head[h]
        for t in chunk:
            states[h] = gamma * states[h] + outer(K[h, t], V[h, t])
            O[h, t] = Q[h, t] @ states[h]
```

真实实现会再加上：

- 多头拆分与拼接
- 线性投影层
- 归一化
- 位置编码
- 数值稳定处理

### 真实工程例子

假设你在做一个面向长上下文日志分析的生成模型。输入是持续追加的系统日志，部署侧要求：

- 在线推理
- 每来一条日志就增量更新
- 显存预算固定
- 历史长度可能达到几十万 token

这时，标准 softmax attention 的 KV cache 会持续增长，推理延迟和显存占用都变差。RetNet 的递推状态更像“压缩后的记忆寄存器”：

- 训练时可用 parallel 形式整段训练，保持吞吐。
- 线上可切到 recursive 形式，每来一条日志只更新一次状态。
- 如果在线批处理多个小窗口，还可用 chunk-wise，在吞吐和延迟之间折中。

这类场景就是 RetNet 的典型发力点。

---

## 工程权衡与常见坑

RetNet 的难点不在“公式看不懂”，而在“工程上很容易看起来等价，实际却跑偏”。

### 常见坑一：三种模式的归一化不一致

有些实现会加入额外 normalization。归一化的白话解释就是“把数值尺度控制在稳定范围内，避免某些头爆掉或塌掉”。

问题在于：

- 某些 normalization 方案只适合 parallel 展开
- 换到 recurrent 或 chunk 时，不一定严格保持同一结果

这会导致训练时指标正常，部署切到递推后输出分布漂移。

### 常见坑二：$\gamma$ 配置训练和部署不一致

多尺度 RetNet 的能力高度依赖每个头的 $\gamma$。如果训练时和推理时代码路径不同，或者 chunk 模式下用了不同衰减广播规则，就会直接破坏一致性。

尤其在多头实现里，常见错误包括：

- 头顺序错位
- 广播维度错误
- chunk 边界处重复衰减或漏衰减

### 常见坑三：chunk 数过多带来的数值误差

理论上等价，不代表浮点上严格相等。因为：

- 不同计算顺序会改变浮点舍入路径
- $\gamma$ 接近 1 时，长序列累计误差更明显
- 半精度训练/推理下更容易放大差异

因此，`torch.allclose` 通过不是礼节，而是必要回归项。

### 常见坑四：把 RetNet 当成“无脑替代 Attention”

RetNet 解决的是“状态递推与部署效率”问题，不是自动提升全部任务精度。若任务高度依赖动态精细选择，softmax attention 仍然更直接。

下面用表格归纳：

| 风险点 | 现象 | 根因 | 推荐做法 |
|---|---|---|---|
| normalization 不一致 | parallel 好，recurrent 漂移 | 数学形式不完全等价 | 部署前固定同一归一化配置做一致性测试 |
| $\gamma$ 广播错误 | 多头结果异常 | 头维、块维处理错位 | 对每个 head 单独打印/校验 $\gamma$ |
| chunk 过细 | chunk 与 recursive 不稳定 | 浮点误差累积 | 用目标 chunk 大小做回归，不只测全序列 |
| $\gamma$ 取值过极端 | 头几乎失忆或几乎冻结 | 时间常数失衡 | 用覆盖短中长依赖的分布初始化 |
| 位置编码耦合错误 | 长程建模异常 | RoPE 或相对位置实现错配 | 先在无 RoPE 版本验证三模式一致，再引入位置编码 |

实际工程中，建议把下面这件事做成单元测试：

1. 同一批输入
2. 同一组参数
3. 跑 parallel、chunk、recursive
4. 比较输出与中间状态是否 `allclose`

不做这一步，部署问题通常会在最晚的时候暴露。

---

## 替代方案与适用边界

RetNet 不是唯一的长序列方案。是否适合，取决于你到底想优化什么。

| 方案 | 核心机制 | 训练复杂度直觉 | 推理友好度 | 控制粒度 |
|---|---|---|---|---|
| RetNet | 指数衰减状态递推 + 多尺度头 | 可并行展开 | 高，支持递推状态 | 通过 $\gamma$ 控制记忆尺度 |
| Softmax Attention | 显式对全历史做归一化选择 | $O(N^2)$ | 中，依赖 KV cache | 直接、表达强 |
| Performer 等线性注意力 | 核技巧近似 attention | 近线性 | 通常较好，但训练/推理路径未必像 RetNet 这样统一 | 近似误差依赖特征映射 |

可以这样理解适用边界：

- 如果你最在意的是生成式部署效率，特别是长序列逐 token 推理，RetNet 值得优先考虑。
- 如果你最在意的是成熟度、生态和训练稳定性，标准 softmax attention 仍然是更稳妥的基线。
- 如果你想要线性复杂度，但不一定要求“并行训练和递推推理共用同一套状态语义”，Performer 一类方法也可以考虑。

再给一个更直接的判断规则：

- 任务主要在训练端，推理不是瓶颈：先用 softmax。
- 任务是长上下文在线生成，缓存成本敏感：优先看 RetNet。
- 任务能接受近似注意力，但不强求严格递推形式：可以比较 Performer 等线性 attention。

RetNet 自身也有边界。即使有多尺度 $\gamma$，它仍然需要调参：

- 头数怎么分给短程和长程
- $\gamma$ 初值怎么设
- 是否引入额外 normalization
- chunk 大小怎么定

所以它不是“零成本更快的 Transformer”，而是“为递推部署做了结构性重写的序列模型”。

---

## 参考资料

| 来源 | 核心贡献 | 值得再读的内容 |
|---|---|---|
| Retentive Network 原论文（arXiv 2307.08621） | 给出 retention 定义、多尺度设计、parallel/chunk/recurrent 三种计算范式 | 三种等价计算模式与复杂度讨论 |
| DeepWiki 的 Retention / State Space 讲解 | 用工程视角解释 retention 层、多头状态与状态空间关系 | 多 $\gamma$ 状态维护和实现说明 |
| `myscience/retnet-pytorch` | 给出 PyTorch 参考实现，并展示 parallel/chunk/recurrent 的一致性测试思路 | `torch.allclose` 一致性验证与配置注意事项 |
| CSDN 对线性注意力与 RetNet 的分析 | 补充指数衰减直觉，以及与 RoPE/复数特征值扩展的联系 | 用直观方式理解衰减和位置编码耦合 |

参考链接：

| 来源 | 链接 |
|---|---|
| arXiv 论文概述 | https://summarizepaper.com/en/tree/2307.08621v1/?utm_source=openai |
| DeepWiki | https://deepwiki.com/fla-org/flash-linear-attention/2.6-retention-and-state-space-models?utm_source=openai |
| PyTorch 实现仓库 | https://github.com/myscience/retnet-pytorch?utm_source=openai |
| CSDN 解析 | https://blog.csdn.net/tMb8Z9Vdm66wH68VX1/article/details/150034663?utm_source=openai |
