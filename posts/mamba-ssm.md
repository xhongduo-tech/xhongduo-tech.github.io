## 核心结论

Mamba 的核心不是“把 Attention 换成 RNN”，而是把传统状态空间模型（SSM，指用一个隐藏状态递推整段序列的模型）改造成**选择性状态空间模型**：每个 token 都能生成自己的状态更新规则。

标准离散 SSM 写成：

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t,\qquad y_t = \bar{C} h_t
$$

这里 $\bar{A},\bar{B},\bar{C}$ 固定不变。Mamba 改成：

$$
h_t=\bar{A}_t h_{t-1}+\bar{B}_t x_t,\qquad y_t=\bar{C}_t h_t
$$

其中 $\bar{A}_t,\bar{B}_t,\bar{C}_t$ 都依赖当前输入 $x_t$。白话说，模型不再用一套固定规则处理所有 token，而是让每个 token 决定“该记住什么、该忘掉什么”。

这解决了经典 SSM 的核心短板：**固定动态系统缺少内容感知能力**。Transformer 的强项是内容相关的选择，Mamba 则把这种“选择性”放进状态更新里，同时保留线性复杂度的长序列优势。

| 维度 | 静态 SSM | Mamba |
|---|---|---|
| 状态参数 | 固定 $\bar{A},\bar{B},\bar{C}$ | 每个 token 生成 $\bar{A}_t,\bar{B}_t,\bar{C}_t$ |
| 是否内容感知 | 弱 | 强 |
| 训练并行方式 | 常可转卷积 | 用 selective scan 并行扫描 |
| 推理方式 | 递推 | 递推 |
| 序列复杂度 | 线性 | 线性 |
| 长上下文推理 | 快 | 更适合内容敏感长上下文 |

玩具例子先看最小数值。若 $h_{t-1}=1$，某个 token 生成：

$$
\bar{A}_t=0.9,\ \bar{B}_t=0.5,\ \bar{C}_t=1.2,\ x_t=2
$$

则：

$$
h_t=0.9\cdot1+0.5\cdot2=1.9,\qquad y_t=1.2\cdot1.9=2.28
$$

如果下一个 token 让 $\bar{A}_{t+1}=0.1$，表示旧状态几乎被清空，新输入主导更新。这就是“选择性保留/忘记”。

---

## 问题定义与边界

问题很具体：**能不能在长序列上保持线性复杂度，同时又像注意力一样对内容敏感**？

Transformer 的自注意力（attention，指每个 token 与所有 token 两两计算相关性）在长度为 $n$ 的序列上，时间和显存主项通常是 $O(n^2)$。当上下文从 4k 扩到 1M 时，成对交互数量从约 $1.6\times10^7$ 增长到 $10^{12}$ 量级，增长过快，训练和推理都很难承受。

Mamba 试图解决的不是“所有序列问题”，而是下面这类问题：

| 目标 | 说明 |
|---|---|
| 长上下文 | 序列很长，不能接受 $O(n^2)$ |
| 内容敏感记忆 | 不是所有历史都要平均保留 |
| 可并行训练 | 不能退回纯串行 RNN 训练 |
| 高效推理 | 每步最好是常数时间更新 |

它的边界也很明确。

第一，Mamba 的并行化依赖**可组合的仿射递推**。仿射，意思是“线性变换加偏置”这一类形式，便于做前缀扫描。  
第二，虽然它是线性复杂度，但不是“零代价”。动态生成 $\bar{A}_t,\bar{B}_t,\bar{C}_t$ 会带来额外投影和 kernel 设计成本。  
第三，Mamba 不是通用替代一切注意力结构。在需要显式 token-token 对齐、检索式复制、复杂跨位置路由时，注意力依然有优势。

真实工程例子：假设要做 100 万 token 的日志异常检测。Transformer 若保留全局注意力，成本会被上下文长度平方放大；Mamba 只需顺序维护状态，训练时用扫描并行化，推理时按步更新隐藏状态，更容易把吞吐量维持在可部署范围内。

---

## 核心机制与推导

先看普通 SSM。它本质上是一个递推系统：

$$
h_t = \bar{A}h_{t-1} + \bar{B}x_t
$$

这里隐藏状态 $h_t$ 是“压缩后的历史摘要”。问题在于 $\bar{A},\bar{B},\bar{C}$ 固定后，面对不同 token，系统没有足够自由度去决定哪些信息重要。

Mamba 的做法是把这些参数变成输入的函数。常见写法可以理解为：

$$
\Delta_t = \mathrm{softplus}(W_\Delta x_t),\quad
\bar{B}_t = W_B \,\mathrm{SiLU}(x_t),\quad
\bar{C}_t = W_C \,\mathrm{SiLU}(x_t)
$$

再结合连续到离散的参数化，得到依赖 $x_t$ 的 $\bar{A}_t$。SiLU 是一种平滑激活函数，可以理解为“保留幅值信息的软门控”。

于是每一步都成了一个仿射变换：

$$
h_t = a_t h_{t-1} + b_t
$$

其中把 $b_t$ 简写为 $\bar{B}_t x_t$。关键在于，这种变换可以组合：

$$
h_2 = a_2(a_1 h_0 + b_1) + b_2 = (a_2 a_1) h_0 + (a_2 b_1 + b_2)
$$

所以两个步骤 $(a_1,b_1)$ 和 $(a_2,b_2)$ 能合并成一个更大的步骤：

$$
(a_2,b_2)\otimes(a_1,b_1)=(a_2a_1,\ a_2b_1+b_2)
$$

只要这个组合满足结合律：

$$
(u\otimes v)\otimes w = u\otimes(v\otimes w)
$$

就能做 parallel scan。结合律的白话意思是：先合并左边还是先合并右边，结果一样。Blelloch scan 正是利用这点，把看似必须串行的前缀递推改成树形归约和下发。

可把训练中的 selective scan 理解成两段：

1. 上扫（up-sweep）：先把局部片段两两合并，逐层算出大块前缀摘要。
2. 下扫（down-sweep）：再把前缀结果分发回每个位置，得到每一步对应的状态。

伪代码结构如下：

```text
pairs[t] = (a_t, b_t)

# up-sweep
for level in tree_levels:
    combine neighboring pairs with ⊗

# down-sweep
for level in reversed(tree_levels):
    propagate prefix summaries to children

h_t = prefix_pair[t] applied to h_0
```

玩具例子可以直接看“选择性”如何发生。

设初始状态 $h_0=1$，两个 token 的参数分别为：

| token | $\bar{A}_t$ | $\bar{B}_t$ | $x_t$ | 含义 |
|---|---:|---:|---:|---|
| “重要错误码” | 0.95 | 0.80 | 2.0 | 强保留旧状态，强写入新信息 |
| “无关分隔符” | 0.10 | 0.05 | 2.0 | 迅速遗忘，几乎不写入 |

第一步：

$$
h_1=0.95\cdot1+0.8\cdot2=2.55
$$

第二步若是无关 token：

$$
h_2=0.10\cdot2.55+0.05\cdot2=0.355
$$

同样的输入值，仅仅因为 token 类型不同，系统就选择了完全不同的记忆策略。这就是 Mamba 相比静态 SSM 最核心的表达能力提升。

---

## 代码实现

下面先用一个最小可运行 Python 例子，模拟“静态 SSM”和“选择性 SSM”的差异。代码里故意只保留标量状态，方便验证。

```python
from math import isclose

def static_ssm(xs, A=0.8, B=0.3, C=1.0, h0=0.0):
    h = h0
    ys = []
    for x in xs:
        h = A * h + B * x
        ys.append(C * h)
    return ys

def selective_params(token):
    # token-dependent 参数：不同 token 决定记忆/遗忘强度
    if token == "important":
        return 0.95, 0.80, 1.20
    if token == "separator":
        return 0.10, 0.05, 1.00
    return 0.60, 0.30, 1.00

def selective_ssm(tokens, xs, h0=0.0):
    h = h0
    ys = []
    for token, x in zip(tokens, xs):
        A_t, B_t, C_t = selective_params(token)
        h = A_t * h + B_t * x
        ys.append(C_t * h)
    return ys

# 最小数值例子
h_prev = 1.0
A_t, B_t, C_t, x_t = 0.9, 0.5, 1.2, 2.0
h_t = A_t * h_prev + B_t * x_t
y_t = C_t * h_t
assert isclose(h_t, 1.9)
assert isclose(y_t, 2.28)

# 静态 SSM 无法按 token 改写规则
ys_static = static_ssm([2.0, 2.0], A=0.9, B=0.5, C=1.2, h0=1.0)
assert len(ys_static) == 2

# 选择性 SSM：第二个 token 主动忘记
ys_selective = selective_ssm(["important", "separator"], [2.0, 2.0], h0=1.0)
assert ys_selective[0] > ys_selective[1]
assert isclose(ys_selective[0], 1.2 * (0.95 * 1.0 + 0.80 * 2.0))
```

如果把它映射到真实实现，训练和推理通常分开设计。

训练阶段伪代码：

```python
def train_block(x_seq):
    # 1. 输入投影，生成 token-dependent 参数
    delta = softplus(linear_delta(x_seq))
    B_t = linear_B(silu(x_seq))
    C_t = linear_C(silu(x_seq))
    A_t = discretize(base_A, delta)

    # 2. 组织成可 scan 的仿射对
    pairs = [(A_t[i], B_t[i] * x_seq[i]) for i in range(len(x_seq))]

    # 3. selective scan 并行求前缀
    prefix_pairs = blelloch_scan(pairs, combine_affine)

    # 4. 还原每个位置状态并读出
    h_seq = [apply_affine(p, h0=0.0) for p in prefix_pairs]
    y_seq = [C_t[i] * h_seq[i] for i in range(len(x_seq))]
    return y_seq
```

推理阶段伪代码：

```python
def infer_step(x_t, h_prev):
    delta_t = softplus(linear_delta(x_t))
    B_t = linear_B(silu(x_t))
    C_t = linear_C(silu(x_t))
    A_t = discretize(base_A, delta_t)

    h_t = A_t * h_prev + B_t * x_t
    y_t = C_t * h_t
    return y_t, h_t
```

流程上可以概括成：

| 阶段 | 做什么 | 复杂度直觉 |
|---|---|---|
| 训练 | 投影参数 + selective scan | 对长度近线性扩展，能并行 |
| 推理 | 逐 token 递推 | 每步常数状态更新 |

---

## 工程权衡与常见坑

Mamba 难点不在公式，而在**算子设计是否真的适合硬件**。

第一类坑是结合律。  
只有当递推能写成可组合的仿射形式时，scan 才正确。若你在组合中间插入了不满足结合律的逐步非线性，例如先把两个前缀结果相加再过一次任意激活，scan 就不再等价于原始串行递推，训练结果会错。

第二类坑是内存搬运。  
$\bar{A}_t,\bar{B}_t,\bar{C}_t$ 是动态的，不能像静态卷积核一样提前展开成长滤波器。若实现上把每步中间张量频繁写回 DRAM，再读回 SRAM，理论线性复杂度会被真实带宽瓶颈拖垮，所以需要 fused kernel 和适度重计算。

第三类坑是推理模式切换。  
训练时用 selective scan 是为了并行；推理时若还坚持整段扫描，就主动放弃了递推模型每步 $O(1)$ 更新的优势。部署必须切回缓存隐藏状态的 RNN 样式接口。

| 常见坑 | 为什么出问题 | 规避方式 |
|---|---|---|
| 非结合算子参与 scan | 前缀合并顺序改变结果 | 只在仿射组合层做 scan |
| 动态矩阵预展开 | 张量规模爆炸，失去效率 | 用 fused kernel 即时计算 |
| 中间状态频繁落盘 | 带宽而不是算力成瓶颈 | kernel fusion + recomputation |
| 推理继续用全局 scan | 失去常数时间/步优势 | 推理改为递推接口 |
| 把 Mamba 当成注意力等价替身 | 某些显式检索任务未必占优 | 按任务评估，不做口号式替换 |

真实工程例子：做在线代码补全时，推理请求通常是一 token 一 token 地追加。此时最合理的实现不是重复扫描整段上下文，而是维护每层状态缓存 `h_prev`，新 token 到来后只执行一次 `h_t = A_t h_prev + B_t x_t`。这正是 Mamba 在长上下文流式推理里的优势来源。

---

## 替代方案与适用边界

如果只比较“能不能做长序列”，Mamba 不是唯一答案。真正要比较的是：复杂度、内容感知能力、训练并行性、推理硬件友好性。

| 方案 | 主复杂度 | 内容感知记忆 | 长序列表现 | 推理特性 | 适用边界 |
|---|---|---|---|---|---|
| Transformer | $O(n^2)$ | 很强 | 长序列成本高 | KV cache 成本随上下文增大 | 短到中等上下文，显式对齐任务 |
| S4/静态 SSM | 近线性 | 弱于注意力 | 强 | 递推快 | 连续信号、时序建模、固定动态更自然 |
| Mamba | 线性 | 强于静态 SSM | 很强 | 常数时间/步递推 | 超长上下文、流式推理、硬件敏感部署 |

S4 是 Mamba 的重要前身。它擅长把结构化状态空间模型做得稳定、高效，但参数在序列中通常是静态或弱动态的，所以对离散 token 的“按内容选择”不如 Mamba。

Transformer 仍然是最稳的基线。理由很简单：生态成熟、训练技巧丰富、很多任务天然依赖显式 token 间交互。若上下文长度只有 2k 到 8k，且主要瓶颈不是注意力，那么 Mamba 未必带来决定性收益。

Mamba 更适合的边界是：

1. 上下文很长，最好到数十万甚至百万 token。
2. 推理以流式追加为主，希望每步更新固定成本。
3. 硬件对带宽敏感，希望减少注意力矩阵和大 KV cache 压力。
4. 任务需要内容感知记忆，但不一定需要显式全对全匹配。

---

## 参考资料

- Gu, Albert; Dao, Tri. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. 论文原文，负责“核心定义、选择性 SSM 公式、5× 推理吞吐、百万长度序列”这几部分。https://arxiv.org/abs/2312.00752
- IBM Think. *What Is A Mamba Model?* 适合解释“为什么固定 SSM 不够、为什么需要 selective scan、训练和推理为何分开实现”。https://www.ibm.com/think/topics/mamba-model
- Emergent Mind. *Mamba-Based Selective State Space Model*. 适合做“新手版机制概览”，包括输入相关的 $A/B/C$、Selective SSM 模块结构。https://www.emergentmind.com/topics/mamba-based-selective-state-space-model
- Harris, Sengupta, Owens. *Parallel Prefix Sum (Scan) with CUDA*, GPU Gems 3. 不是 Mamba 专文，但最适合理解 Blelloch-style scan 的 up-sweep / down-sweep 结构，对“核心机制与推导”一节有直接帮助。https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
- Gu, Goel, Re. *Efficiently Modeling Long Sequences with Structured State Spaces*. S4 论文，用于理解 Mamba 之前的静态结构化 SSM 基础，以及为什么 Mamba 要进一步引入选择性。https://arxiv.org/abs/2111.00396
