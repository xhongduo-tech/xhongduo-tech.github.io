## 核心结论

TransNormerLLM 的核心不是“把 Transformer 全部推翻重来”，而是把最贵、最难扩展的那一部分换掉：把标准 softmax 注意力替换成 **NormAttention**，再用 **GLA** 和 **SGLU** 补回线性注意力常见的表达能力损失。

这里先把术语说白话：

- **softmax 注意力**：每个 token 都和所有 token 两两打分，再做归一化，所以算力和显存通常随序列长度按平方增长。
- **NormAttention**：一种简单线性注意力，把“先算完整注意力矩阵再加权”改成“先累计上下文统计量，再线性读取”，因此能按序列长度线性扩展。
- **GLA（Gated Linear Attention）**：带门控的线性注意力，白话说就是“不是所有 token 都无条件写入上下文，而是先过一个门，决定写多少”。
- **SGLU（Simple Gated Linear Unit）**：简单门控单元，白话说就是“用两个线性投影做逐元素相乘，让通道方向也有非线性筛选能力”。

可以把它理解成一句话：

> Transformer 的平方注意力换成 NormAttention + GLA + SGLU，就像把原来必须先算完整“关系表”的流程，改成一遍扫描序列、边读边累计上下文，再用门控控制哪些信息该保留、该放大。

标准注意力常写成：

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}(QK^\top)V
$$

而线性注意力的典型改写思路是把它变成可重排的形式：

$$
\mathrm{Attn}(Q,K,V)\approx \phi(Q)\big(\phi(K)^\top V\big)
$$

其中 $\phi(\cdot)$ 是某种特征映射。这样就不必显式构造 $n \times n$ 的注意力矩阵，长度为 $n$ 时，复杂度可从 $O(n^2)$ 降到 $O(n)$ 或 $O(nd^2)$ 这一类线性形式。

TransNormerLLM 的价值不只是“更快”，而是：它试图在**线性复杂度**、**全局上下文融合**、**训练稳定性**之间找到一个工程上可用的平衡点。

---

## 问题定义与边界

问题先定义清楚：为什么标准 Transformer 在长序列下越来越难用？

原因很直接。设序列长度为 $n$，隐藏维度先忽略常数项。softmax 注意力要计算 $QK^\top$，得到一个 $n \times n$ 的矩阵。于是：

- 计算量近似随 $n^2$ 增长
- 注意力矩阵显存也近似随 $n^2$ 增长
- 训练时反向传播还会把这个成本进一步放大
- 自回归推理时，序列越长，历史缓存越大

如果从 4K token 扩到 64K token，长度放大 16 倍，二次项会放大到 $16^2=256$ 倍。这就是长上下文模型一旦继续扩窗，成本突然不可控的根本原因。

但只把二次复杂度改成线性，也不自动等于“模型就好用”。很多早期线性注意力方案的问题是：

- 归一化不足，数值容易漂
- 缺少足够的门控，所有 token 都被平均式地混进上下文
- 通道方向非线性不够，表达能力下降
- 长距离依赖虽然形式上保留，但实际效果变弱

下面这个表可以概括边界：

| 方案 | 时间/显存随序列长度 | 是否显式构造注意力矩阵 | 长上下文扩展性 | 训练稳定性 |
|---|---|---:|---|---|
| softmax Attention | $O(n^2)$ | 是 | 一般 | 通常较稳 |
| 纯线性投影+累加 | 近似 $O(n)$ | 否 | 强 | 可能掉表达能力 |
| TransNormerLLM 的 NormAttention + GLA + SGLU | 近似 $O(n)$ | 否 | 强 | 通过门控与归一化改善 |

所以 TransNormerLLM 解决的问题不是“如何证明线性注意力存在”，而是更具体的工程问题：

1. 如何把注意力改成线性复杂度。
2. 如何避免线性化后模型变成“只有快，没有效果”。
3. 如何让它在真实 LLM 训练和推理中可落地。

它的边界也要说清楚：它不是在所有场景都优于标准 Transformer。短序列、小模型、对成熟训练配方依赖强的场景，softmax 结构仍然常常更稳。

---

## 核心机制与推导

TransNormerLLM 可以拆成两条主线：

1. **TokenMixing**：token 之间怎么交互，由 GLA 和 NormAttention 负责。
2. **ChannelMixing**：每个 token 内部通道怎么变换，由 SGLU 负责。

### 1. GLA：先决定哪些 token 值得写入上下文

可以把 GLA 理解成“给线性注意力加一个动态阀门”。论文中的思路可抽象写成：

$$
g_t = W_g x_t \tag{3}
$$

$$
\tilde{v}_t = \mathrm{swish}(g_t)\odot (W_v x_t) \tag{4}
$$

其中：

- $x_t$ 是第 $t$ 个 token 的输入
- $W_g, W_v$ 是线性投影
- $\mathrm{swish}(z)=z\cdot \sigma(z)$
- $\odot$ 是逐元素乘法

白话解释：token 不再直接把自己的值向量写进全局上下文，而是先通过一个 gate。gate 大，这个 token 贡献更大；gate 小，它对后续 token 的影响就被抑制。

这解决了“线性累加过于平均”的问题。

### 2. SGLU：通道方向用乘法产生非线性

SGLU 可以写成：

$$
\mathrm{SGLU}(x) = (W_a x)\odot(W_b x) \tag{5}
$$

和标准 FFN 相比，它非常简单，但逐元素乘法本身就是强约束的非线性。这里给一个玩具例子。

设输入向量：

$$
x=[1,2]
$$

若两个投影都取单位映射，即：

$$
W_a x = [1,2], \quad W_b x=[1,2]
$$

则：

$$
\mathrm{SGLU}(x)=[1,2]\odot[1,2]=[1,4]
$$

这个例子很小，但足够说明问题：即使不再显式加 ReLU/GELU，这种“两个线性结果相乘”的结构也会改变表示形状，不是单纯线性层能做到的。

### 3. NormAttention：把平方级关系表改成线性累计

NormAttention 的核心是去掉 softmax，用可重排的形式做上下文聚合。可写成：

$$
Y = D^{-1}\phi(Q)\big(\phi(K)^\top V\big) \tag{6}
$$

其中 $D$ 是归一化项，常见写法与 $\phi(Q)\phi(K)^\top \mathbf{1}$ 有关，用来避免数值无界增长。进一步可把右侧理解成：

$$
S=\phi(K)^\top V,\quad Z=\phi(K)^\top \mathbf{1},\quad
Y_i=\frac{\phi(q_i)S}{\phi(q_i)Z} \tag{7}
$$

这一步为什么关键？因为 $S$ 和 $Z$ 可以沿序列扫描时递推累计。于是流程从“每个位置都和所有位置单独算一遍”变成：

1. 读取一个 token
2. 更新累计状态 $S, Z$
3. 当前 query 直接读取累计状态
4. 继续向后扫描

这就是“右乘形式”的意义。它让全局上下文不必通过完整注意力矩阵表达，而是通过累计统计量表达。

### 4. 一个从机制到直觉的完整图景

可以按下面顺序理解整个块：

1. 输入 token 先做线性投影，得到 query、key、value。
2. GLA 根据 gate 控制 value 实际写入多少。
3. NormAttention 不构造完整注意力矩阵，而是累计 key-value 统计量。
4. 当前 query 读取累计上下文，完成 token mixing。
5. 得到的新表示再进入 SGLU，在通道方向做非线性筛选。

所以它不是“单一线性注意力模块”，而是一整套互补设计：

- NormAttention 负责复杂度
- GLA 负责信息流控制
- SGLU 负责通道非线性

---

## 代码实现

下面给一个可运行的简化 Python 版本，重点不是完全复现论文细节，而是把三件事拆开：NormAttention、GLA、SGLU。这样最容易看懂，也方便把原有 Transformer 模块逐步替换掉。

```python
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def swish(x):
    return x * sigmoid(x)

def sglu(x):
    # 玩具版：两个投影都设为恒等映射
    a = x[:]
    b = x[:]
    return [ai * bi for ai, bi in zip(a, b)]

def gla_gate(v, g):
    # 用 swish(g) 控制 value 写入强度
    return [swish(gi) * vi for gi, vi in zip(g, v)]

def norm_attention(queries, keys, values, eps=1e-8):
    """
    因果版简化实现：
    对每个位置 i，累计 0..i 的 key/value 统计量，再由 q_i 读取。
    这里假设 q/k 已经是非负映射后的结果，方便演示。
    """
    dim = len(keys[0])
    s = [0.0] * dim   # 近似对应 sum_t k_t * v_t（逐维玩具版）
    z = [0.0] * dim   # 近似对应 sum_t k_t
    outputs = []

    for q, k, v in zip(queries, keys, values):
        for d in range(dim):
            s[d] += k[d] * v[d]
            z[d] += k[d]

        out = []
        for d in range(dim):
            numer = q[d] * s[d]
            denom = q[d] * z[d] + eps
            out.append(numer / denom)
        outputs.append(out)

    return outputs

# 玩具例子
x = [1.0, 2.0]
assert sglu(x) == [1.0, 4.0]

queries = [[1.0, 1.0], [1.0, 2.0]]
keys    = [[1.0, 1.0], [2.0, 1.0]]
values  = [[3.0, 4.0], [5.0, 6.0]]
gates   = [[1.0, 0.0], [0.5, 1.0]]

gated_values = [gla_gate(v, g) for v, g in zip(values, gates)]
outs = norm_attention(queries, keys, gated_values)

assert len(outs) == 2
assert len(outs[0]) == 2
assert outs[0][0] > 0
assert outs[1][1] > outs[0][1]
```

如果写成 PyTorch 风格的前向过程，结构上通常就是：

```python
def forward(x):
    q = q_proj(x)
    k = k_proj(x)
    v = v_proj(x)

    gate = gate_proj(x)
    v = swish(gate) * v           # GLA

    x = norm_attention(q, k, v)   # TokenMixing
    x = out_proj(x)

    c1 = up_proj(x)
    c2 = gate2_proj(x)
    x = c1 * c2                   # SGLU, ChannelMixing

    return x
```

真实工程例子是 OpenNLPLab 的实现路线：把 Attention、SGLU、GLA 拆成独立模块，再配合 Lightning Attention、FSDP、Triton、BFloat16 等训练推理组件。这种拆分方式的好处是：

- 可以只替换注意力，不动外层训练框架
- 可以单独调 gate 维度、归一化方式、块化策略
- 可以逐步验证“速度收益”和“精度损失”分别来自哪里

---

## 工程权衡与常见坑

线性注意力最大的误区是：公式写成 $O(n)$，工程上就一定更快。实际并不是这样。

第一类坑是**数值稳定性**。例如 LRPE-d 一类位置编码或衰减项里，如果把 temperature 设成可学习参数，训练前期很可能出现梯度突然放大，随后 loss 直接变成 NaN。原因通常不是“模型学不会”，而是某个指数或归一化因子在早期被推到危险区间。

第二类坑是**内存层级没有处理好**。Lightning Attention 一类实现往往依赖块化加载，把大块数据分层放到 HBM/SRAM 友好的访问模式里。如果只是把公式从 softmax 改成 cumsum 或累计状态，但没有做块化，实际内核可能反而更慢。

第三类坑是**门控维度太省**。SGLU 理论上很简单，但如果 gate 投影维度压得过小，逐元素乘法就会退化成过强的信息裁剪，表现成容量不足、训练后期上不去。

| 常见坑 | 典型现象 | 规避措施 |
|---|---|---|
| 可学习 temperature 导致不稳定 | 前几千步后 loss 突然 NaN | 先固定 decay/temperature，再逐步开放 |
| 未做块化加载 | 线性注意力理论更优，实际 wall-clock 更慢 | 使用分块 kernel 与层级缓存设计 |
| gate 维度过小 | 验证集精度明显低于预期 | 适当增大 gate/hidden ratio |
| 归一化项过弱 | 长序列输出幅值漂移 | 显式维护分母项并加稳定项 `eps` |
| 直接照搬 softmax 超参 | 收敛慢或不收敛 | 单独重调学习率、warmup、初始化 |

这里给一个真实工程判断标准：如果你把 8K 上表现不错的 softmax 模型改成线性注意力模型，结果 8K 更慢、32K 没明显收益，那通常不是“线性注意力没用”，而是内核实现、块化策略、缓存读写模式还没做对。

---

## 替代方案与适用边界

可以把选择逻辑压缩成一个简单表格：

| 场景 | 更合适的方案 |
|---|---|
| 4K 左右短序列、训练配方成熟优先 | 标准 Transformer |
| 需要 32K、64K 甚至更长上下文 | TransNormerLLM 一类线性注意力方案 |
| 小数据集、极度追求稳定收敛 | softmax + 常规 FFN |
| 算力受限但要保留全局上下文 | NormAttention + GLA + SGLU |

经验上可以这样理解：

- **常规 Transformer 更适合 4K token 左右**：生态成熟、实现稳定、调参经验多。
- **TransNormer 类方案更适合 64K token 一类长上下文任务**：例如长文档问答、代码仓库检索、超长日志分析、多轮历史会话压缩建模。

但也不要过度泛化。若任务满足下面三个条件：

1. 序列不长
2. 数据不多
3. 精度波动容忍度很低

那么 softmax attention + 常规 FeedForward 往往仍是更稳的选项。因为在这些条件下，二次复杂度还没有成为主要矛盾，而成熟训练行为才是主要矛盾。

TransNormerLLM 的适用边界可以总结成一句话：**当长度扩展已经成为系统瓶颈时，它值得优先考虑；当长度不是瓶颈时，标准 Transformer 仍然是强基线。**

---

## 参考资料

- `arXiv 2307.14995, Scaling TransNormer to 175 Billion Parameters`
  - 作用：给出 TransNormerLLM 的核心公式、NormAttention/GLA/SGLU 设计与扩展到大模型的论证。
- `OpenNLPLab / TransNormerLLM GitHub 项目`
  - 作用：提供工程实现、训练配置、Lightning Attention、FSDP、Triton、BFloat16 等落地细节。
- `EmergentMind 论文速览/项目解读`
  - 作用：提供较短路径的结构概览，适合先建立整体图景，再回看原论文公式。
