## 核心结论

MoE 和 MoA/MoH 可以放进同一个统一框架里理解：它们本质上都是**条件计算**。条件计算的直白解释是，不是每次都把整层都算一遍，而是先让一个 router 决定“这次该激活谁”。

统一写法是：

$$
y=\sum_{i \in G(x)} w_i \cdot E_i(x), \quad
G(x)=\operatorname{TopK}(\operatorname{Softmax}(W_g x))
$$

这里 $E_i$ 是“专家”。在 MoE 里，专家通常是 FFN；在 MoA 或 MoH 里，专家可以是 attention head。差别不在公式，而在**专家粒度**。

这就得到一条设计谱系：

| 设计粒度 | 专家是什么 | 典型做法 | 主要收益 | 主要代价 |
| --- | --- | --- | --- | --- |
| 层级 | 整层或整块子网络 | 跳过整层、路由整块 | 调度简单 | 表达力粗 |
| 子层级 | FFN 或 Attention 子层 | 典型 MoE | 参数容量大 | 专家负载不均 |
| 头级 | 单个 attention head | MoA / MoH | token 适配更细 | 路由频繁、缓存复杂 |

核心判断很直接：**粒度越细，表达力通常越强；但 router、调度、缓存和负载均衡的成本也越高。**  
因此，MoE 与 MoA 不是对立方案，而是“同一套稀疏计算语言”在不同粒度上的落点。

先把这句话再说得更具体一些。MoE 更像“为 token 选择哪几个 FFN 模块”；MoA/MoH 更像“为 token 选择哪几个注意力头”。两者都在做同一件事：**让不同 token 走不同计算路径**。如果只记一个结论，记住这一句就够了。

---

## 问题定义与边界

先把问题说清楚。我们关心的不是“模型里有没有专家”这么宽泛的问题，而是两个更具体的问题：

1. 一个 token 到底激活哪些计算单元？
2. 被激活的单元输出如何加权合并？

这两个问题一旦固定，MoE、MoA、MoH 就能用同一套语言比较。

术语上：

| 术语 | 严格含义 | 直白解释 |
| --- | --- | --- |
| router | 根据输入给专家打分的模块 | 决定“谁上场” |
| logits | router 输出的未归一化分数 | 还没变成概率的原始分数 |
| Softmax | 把分数变成概率分布 | 让所有分数加起来等于 1 |
| Top-k | 只保留得分最高的 $k$ 个专家 | 只让前几名工作 |
| load balancing | 让专家使用率尽量均衡 | 别让少数专家一直忙 |
| capacity factor | 单个专家可接收 token 的上限系数 | 给专家设置“最大接单量” |

从边界看，这篇文章只讨论**稀疏激活的条件计算**，不讨论 dense attention 的各种重参数化，也不讨论 KV cache 压缩本身。统一框架里只比较一件事：专家粒度变细以后，收益和成本怎么变。

一个玩具例子最容易说明边界。假设某层有 8 个注意力头，router 对某个 token 输出概率：

$$
p=[0.35,0.25,0.15,0.10,0.05,0.04,0.03,0.03]
$$

如果用 Top-2，只保留前两个头。归一化后：

$$
w_1=\frac{0.35}{0.35+0.25}\approx 0.58,\quad
w_2=\frac{0.25}{0.35+0.25}\approx 0.42
$$

于是该 token 的输出变成：

$$
y = 0.58 E_1(x) + 0.42 E_2(x)
$$

其余 6 个头完全不算。对新手来说，这可以理解成：原来每个 token 都要让 8 个头都工作；现在只让最相关的 2 个头工作。

但这不等于“成本一定降到 $2/8$”。原因至少有四个：

| 理论上省掉的部分 | 工程里新增的部分 |
| --- | --- |
| 未被选中的专家前向 | router 打分 |
| 部分矩阵乘法 | Top-k 选择 |
| 部分激活存储 | token 到专家的分发与回收 |
| 部分参数访问 | 动态 kernel 调度、缓存整理 |

所以“稀疏”不等于“更快”，而是“有机会更省算”。这个机会能不能兑现，取决于实现方式、硬件、batch 形态和推理框架。

---

## 核心机制与推导

统一机制可以分成四步。

1. 计算 router 分数  
   对输入表示 $x$ 做线性变换，得到 logits：
   $$
   z=W_g x
   $$

2. 转成概率  
   用 Softmax 得到每个专家被选中的相对权重：
   $$
   p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
   $$

3. 选 Top-k  
   只保留最大的 $k$ 个分量，记为集合 $G(x)$。

4. 重新归一化并聚合  
   $$
   w_i=\frac{p_i}{\sum_{j \in G(x)} p_j}, \quad i \in G(x)
   $$
   最终输出：
   $$
   y=\sum_{i \in G(x)} w_i E_i(x)
   $$

关键点在于，公式没变，但 $E_i$ 的定义变了。

| 方案 | $E_i(x)$ 的含义 | 路由频率 | 典型瓶颈 |
| --- | --- | --- | --- |
| MoE | 第 $i$ 个 FFN 专家 | 常见是每层、每 token | 专家分配不均、跨设备通信 |
| MoA | 第 $i$ 个 attention head | 每个 token、每个注意力层 | 头级调度、动态聚合 |
| MoH | 头作为专家并做加权和 | 每个 token、每个注意力层 | 头选择与 KV/cache 组织 |

如果把视角再压缩一步，可以得到一个更通用的形式：

$$
E_i : \mathbb{R}^{d_{\text{model}}} \rightarrow \mathbb{R}^{d_{\text{out}}}
$$

也就是说，专家本质上只是“一个把输入映射到输出空间的函数”。只要多个候选函数里，router 能替你挑一部分出来并做加权合并，这就是统一框架里的条件计算。

### 为什么 MoE 容易统一到 FFN？

以标准 Transformer 的 FFN 为例，单层通常写成：

$$
\operatorname{FFN}(x)=W_2 \sigma(W_1 x)
$$

MoE 的做法，是把原来单一的 FFN 替换成多个专家 FFN：

$$
E_i(x)=W_{2,i}\sigma(W_{1,i}x)
$$

然后只激活其中少数几个：

$$
y=\sum_{i\in G(x)} w_i E_i(x)
$$

这样做的直接含义是：**参数容量增大了，但单个 token 实际使用的计算量不一定按专家总数线性增长**。这也是 MoE 能在大模型里扩大参数规模的核心原因。

### 为什么 MoA/MoH 也能写成同一式子？

标准多头注意力可写成多个 head 输出的组合。若第 $i$ 个头输出为 $h_i(x)$，那么传统 MHA 在抽象上可以看成：

$$
y=\sum_{i=1}^{H} h_i(x)
$$

或在更常见实现里先拼接再经输出投影。MoA/MoH 的关键不是“改了注意力公式本身”，而是把“所有头都参与”改成“只让部分头参与，并允许权重由输入决定”：

$$
y=\sum_{i\in G(x)} w_i h_i(x)
$$

这时，head 就成了专家。  
因此，MoE 与 MoA/MoH 的差异，不在“一个是专家，一个不是专家”，而在“专家到底是 FFN，还是 head”。

### 为什么粒度越细通常越强？

因为不同 token 的统计结构真的不同。举三个直观例子：

| token 类型 | 更可能需要的能力 | 细粒度路由的价值 |
| --- | --- | --- |
| 实体名、术语 | 局部语义区分、指代跟踪 | 可以优先选擅长语义聚焦的头 |
| 代码 token | 长距离括号匹配、缩进结构 | 可以优先选擅长结构对齐的头 |
| 数学符号 | 层级关系、公式作用域 | 可以优先选擅长远程依赖的头 |

粗粒度路由只能说“这个 token 走哪一个大模块”。细粒度路由还能说“这个 token 在注意力内部到底该让哪些头工作”。这会让模型的计算路径更贴近 token 的具体需求。

### 为什么粒度越细通常越难做快？

因为要多付三类账：

| 开销类型 | 粗粒度表现 | 细粒度表现 |
| --- | --- | --- |
| Router 计算 | 次数少 | 次数多，几乎每层每 token 都要做 |
| 调度开销 | 批量规则整齐 | 动态分支多，kernel 利用率下降 |
| 状态管理 | 参数和激活较稳定 | 头选择、KV cache、mask 更复杂 |

还可以把这件事写成一个简单的成本分解：

$$
T_{\text{real}}
\approx
T_{\text{selected experts}}
+
T_{\text{router}}
+
T_{\text{dispatch}}
+
T_{\text{memory/cache}}
$$

其中真正让人误判的，是后面三项。很多论文里“理论 FLOPs 下降”主要对应第一项，但真实延迟往往被第二到第四项抬高。

所以统一推导的真正价值不是“证明它们一样”，而是把所有设计都压缩成同一个判断问题：**你愿意为了更细的 token 级适配，支付多少路由与调度成本？**

真实工程例子可以看 MoH。论文摘要给出的结果是：MoH 在 ViT、DiT、LLM 上，使用约 50% 到 90% 的 attention heads 仍能达到更好结果；对继续调优得到的 MoH-LLaMA3-8B，14 个基准平均准确率为 64.0%，比原始 LLaMA3-8B 高 2.4%，同时只使用 75% 的 attention heads。这里说明的不是“头越少越好”，而是**头级稀疏确实能把表达能力和计算路径绑在一起优化**。

---

## 代码实现

下面给一个**最小可运行**的 Python 版本。它不依赖深度学习框架，只演示统一路由逻辑。你可以把 `experts` 理解成 FFN 专家，也可以理解成 attention head 的简化代理。

这次代码不再用标量，而是用向量输入和矩阵专家，这样更接近真实神经网络。

```python
import math
from typing import Callable, List, Sequence, Tuple

Vector = List[float]
Matrix = List[List[float]]
Expert = Callable[[Vector], Vector]


def softmax(xs: Sequence[float]) -> List[float]:
    if not xs:
        raise ValueError("softmax input must not be empty")
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]


def topk_indices(xs: Sequence[float], k: int) -> List[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(xs):
        raise ValueError("k must be <= number of experts")
    pairs = sorted(enumerate(xs), key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in pairs[:k]]


def matvec(mat: Matrix, vec: Vector) -> Vector:
    if not mat or not vec:
        raise ValueError("matrix and vector must not be empty")
    if any(len(row) != len(vec) for row in mat):
        raise ValueError("matrix width must match vector length")
    return [sum(a * b for a, b in zip(row, vec)) for row in mat]


def vec_add(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vector sizes do not match")
    return [x + y for x, y in zip(a, b)]


def vec_scale(a: Vector, s: float) -> Vector:
    return [x * s for x in a]


def route_and_mix(
    x: Vector,
    gate_matrix: Matrix,
    experts: Sequence[Expert],
    k: int = 2,
) -> Tuple[Vector, List[float], List[int]]:
    if len(gate_matrix) != len(experts):
        raise ValueError("gate_matrix rows must equal number of experts")

    logits = matvec(gate_matrix, x)
    probs = softmax(logits)
    picked = topk_indices(probs, k)

    denom = sum(probs[i] for i in picked)
    weights = [0.0 for _ in experts]
    for i in picked:
        weights[i] = probs[i] / denom

    output_dim = len(experts[0](x))
    y = [0.0] * output_dim
    for i in picked:
        expert_out = experts[i](x)
        if len(expert_out) != output_dim:
            raise ValueError("all experts must return vectors of same size")
        y = vec_add(y, vec_scale(expert_out, weights[i]))

    return y, weights, picked


def make_linear_expert(weight: Matrix, bias: Vector) -> Expert:
    def expert(x: Vector) -> Vector:
        return vec_add(matvec(weight, x), bias)
    return expert


# 4 个“专家”，都把 3 维输入映射到 2 维输出
experts = [
    make_linear_expert(
        weight=[[1.0, 0.0, 0.5], [0.2, 0.3, 0.1]],
        bias=[0.1, -0.2],
    ),
    make_linear_expert(
        weight=[[0.4, 0.8, -0.3], [0.0, 1.0, 0.5]],
        bias=[0.0, 0.3],
    ),
    make_linear_expert(
        weight=[[-0.6, 0.2, 0.9], [0.7, -0.1, 0.4]],
        bias=[-0.2, 0.0],
    ),
    make_linear_expert(
        weight=[[0.3, -0.4, 0.7], [0.5, 0.5, 0.5]],
        bias=[0.2, 0.2],
    ),
]

# router 权重：4 个专家，每个专家对 3 维输入给一个打分
gate_matrix = [
    [1.2, 0.1, 0.0],
    [0.8, 0.6, -0.2],
    [-0.5, 0.2, 0.4],
    [-1.0, 0.3, 0.1],
]

x = [2.0, 1.0, -1.0]

y, weights, picked = route_and_mix(x, gate_matrix, experts, k=2)

assert len(picked) == 2
assert abs(sum(weights) - 1.0) < 1e-9
assert all(weights[i] == 0.0 for i in range(len(weights)) if i not in picked)

manual = [0.0, 0.0]
for i in picked:
    out_i = experts[i](x)
    manual = vec_add(manual, vec_scale(out_i, weights[i]))

assert all(abs(a - b) < 1e-9 for a, b in zip(y, manual))

print("picked experts:", picked)
print("weights:", [round(w, 4) for w in weights])
print("output:", [round(v, 4) for v in y])
```

这段代码能直接运行，输出里会告诉你三件事：

1. 选中了哪些专家；
2. 每个被选中专家的权重是多少；
3. 最终混合后的输出向量是什么。

如果从教学角度理解，这段代码对应的是下面这张映射表：

| 代码对象 | 数学对象 | 含义 |
| --- | --- | --- |
| `gate_matrix` | $W_g$ | router 参数 |
| `logits` | $z$ | 专家原始分数 |
| `probs` | $p$ | Softmax 后的概率 |
| `picked` | $G(x)$ | Top-k 选中的专家集合 |
| `weights` | $w_i$ | 重新归一化后的混合权重 |
| `experts[i](x)` | $E_i(x)$ | 第 $i$ 个专家对输入的变换 |

实现上最重要的不是这几行公式，而是“专家是否真的独立”。如果几个所谓的专家共享同一套参数，只是输出时做了不同缩放，那通常不能算真正的 MoE 式专家设计。

把这个伪代码映射到 Transformer 时，有三种常见落点：

| 落点 | `expert_forward` 对应什么 | 难点 |
| --- | --- | --- |
| FFN-MoE | 不同 FFN 模块 | token 分发到不同专家 |
| Head-MoA | 不同 attention head | 每个 token 的头选择不同 |
| 联合 MoE+MoA | FFN 和 head 都做稀疏 | 训练稳定性与推理栈复杂度同时上升 |

如果是教学或实验代码，先做单层 Top-2 FFN-MoE 最稳。要做头级路由，必须提前设计好张量布局，否则性能很容易被 gather/scatter 吃掉。

再往前走一步，工程里还常加两个补件。

### 1. 负载均衡损失

如果 router 总是把 token 派给少数专家，模型虽然“有很多专家”，但真正起作用的只有几个。于是通常会加 auxiliary loss，把专家使用率往均衡方向拉。常见写法的思想是同时考虑：

- router 给出的平均概率；
- 专家实际被选中的频率。

目标不是强行完全平均，而是防止极端偏斜。

### 2. 容量约束

设 batch 里共有 $B\times T$ 个 token，专家数为 $N$，每个 token 选 $k$ 个专家，则单专家容量常写成：

$$
\text{Capacity}=\text{CF}\times \frac{B\times T\times k}{N}
$$

其中 CF 是 capacity factor。  
如果某个专家收到的 token 超过上限，多出来的 token 就要被丢弃、回退，或走别的处理逻辑。这能防止“一个专家爆仓”，但也会带来训练不连续、实现变复杂等问题。

---

## 工程权衡与常见坑

工程里最常见的误判是：看到理论 FLOPs 降了，就默认吞吐一定升。这个判断经常错。

Huang 等人在 2025 年的实验结果给出的信号很直接：在其字符级 Transformer 设置下，MoE 变体虽然验证性能接近 baseline，但训练时间增加约 50% 到 60.9%，推理速度下降约 43.1% 到 55.8%。这说明 router 和调度开销足以吞掉稀疏计算的理论收益。

| 指标 | Baseline FFN | MoE Base | MoE Top-2 | MoE Capacity 1.0 |
| --- | --- | --- | --- | --- |
| 最优验证损失 | 1.4739 | 1.4764 | 1.4718 | 1.4718 |
| 训练时间 | 287.9s | 434.0s | 460.5s | 463.3s |
| 相对训练变化 | 基线 | +50.7% | +60.0% | +60.9% |
| 推理速度 | 441.3 tok/s | 250.9 tok/s | 224.7 tok/s | 195.0 tok/s |
| 相对推理变化 | 基线 | -43.1% | -49.1% | -55.8% |

这个例子特别适合给新手一个正确预期：**稀疏架构不是“自动加速器”，而是“需要被实现得足够好，才有机会加速”的架构选择。**

常见坑主要有四个。

### 第一，负载不均

少数专家被反复选中，其他专家基本闲置。结果是参数容量虽然大了，但有效容量没起来。常见补救是 auxiliary load-balance loss，或者 capacity factor。

一个直观判断标准是看专家使用率直方图。如果长期出现“前两个专家占掉大部分 token”，就说明 router 正在塌缩。

### 第二，capacity mask 不是免费午餐

capacity factor 的直白解释是“每个专家最多接多少 token”。它能限制爆仓，但超过容量的 token 可能被 mask 掉，训练行为会更复杂。

常见副作用有：

| 现象 | 后果 |
| --- | --- |
| token 被丢弃或回退 | 梯度路径变复杂 |
| 专家容量太小 | 学不到稳定分工 |
| 专家容量太大 | 负载均衡效果变弱 |

所以 capacity 不是越严格越好，而是一个系统参数，要和 batch 大小、专家数、路由策略一起调。

### 第三，头级路由的缓存比 FFN-MoE 更难管

FFN 输出是一轮前向就结束，attention head 还牵涉 KV cache、head packing、解码阶段的动态选择。MoH 论文里“只用 75% 的头”是模型层面的结果，不代表你在现有推理框架里能无成本拿到同样吞吐。

特别是在自回归解码里，问题会更集中：

| 环节 | FFN-MoE | 头级 MoA/MoH |
| --- | --- | --- |
| token 分发 | 有 | 有 |
| KV cache 维护 | 基本无新增复杂度 | 复杂度明显上升 |
| 动态 shape 处理 | 中等 | 更高 |
| 推理框架兼容性 | 相对成熟 | 通常更脆弱 |

### 第四，细粒度稀疏会放大实现质量差异

同样的数学形式，在 eager 模式、低效 kernel、错误的 batch 聚合下，速度可能比 dense 还差很多。实践里常见三种情况：

1. 数学上省了算，但 kernel 启动次数暴涨；
2. 参数访问变碎，显存带宽成为瓶颈；
3. 每层都做小而散的动态操作，GPU 利用率明显下降。

因此，工程上真正该问的问题不是“这个方法理不理论优”，而是：

| 检查项 | 该问什么 |
| --- | --- |
| 训练 | 是否比 dense 收敛更稳，还是更难训？ |
| 推理 | 延迟和吞吐是否真的改善，而不是只降 FLOPs？ |
| 基础设施 | 现有框架是否支持高效 dispatch / combine？ |
| 部署 | 动态路由是否会破坏缓存与批处理效率？ |

---

## 替代方案与适用边界

不是所有场景都该上固定 Top-k 的 MoE 或 MoA。

如果模型规模不大、卡数不多、batch 小，router 的开销和负载不均问题通常更明显。这种情况下，粗粒度方案往往更划算。反过来，在大模型、多卡并行、长序列任务里，稀疏激活更有机会把增加的参数容量转成真实收益。

一种重要替代是**动态容量 MoE**。动态容量的直白解释是，不强制每个 token 都固定选 2 个专家，而是让平均激活专家数 $\bar{t}$ 随任务和容量约束变化。

如果固定 Top-2 的平均收益不足，动态容量可能更合适。假设某任务统计下来，每个 token 真正需要的平均专家数是：

$$
\bar{t}=1.76
$$

而硬件预算只能承受 1.5 个专家的平均激活量，那么固定 Top-2 就在系统层面过载；动态容量策略会通过阈值、温度或额外损失，把一部分 token 回退到 Top-1，从而把平均激活数压回预算内。

这类策略常见的目标可以写成：

$$
\mathbb{E}[|G(x)|] \le C
$$

其中 $C$ 是系统预算允许的平均激活专家数。  
这比固定 Top-k 更灵活，但实现也更复杂，因为你不仅要决定“选谁”，还要决定“选几个”。

| 方案 | 平均激活专家数 | 表达力 | 调度复杂度 | 适用场景 |
| --- | --- | --- | --- | --- |
| 固定 Top-2 | 稳定为 2 | 较强 | 中等 | 吞吐可预测、实现简单 |
| 动态容量 | 随 token 波动 | 更灵活 | 更高 | 预算严格、输入难度差异大 |
| 回退粗粒度 MoE | 常较低 | 较稳 | 较低 | 小模型、推理栈简单优先 |
| Dense Attention / FFN | 全激活 | 基线 | 最低 | 工程稳定性优先 |

还可以把“什么时候不该用细粒度路由”说得更明确一点：

| 场景 | 更合适的选择 | 原因 |
| --- | --- | --- |
| 小模型、单卡实验 | Dense 或粗粒度 MoE | 稀疏开销容易盖过收益 |
| 推理栈固定、无法改 kernel | Dense 或 FFN-MoE | 头级路由对栈要求更高 |
| 延迟比精度更敏感 | 粗粒度稀疏或不做稀疏 | 动态路由会放大尾延迟 |
| 训练资源充足、追求容量扩展 | MoE | 工程成熟度通常更高 |
| 追求 token 级精细适配 | MoA/MoH | 头级粒度更细，表达更强 |

MoE 与 MoA 的联合使用也有边界。理论上，FFN 做 MoE、attention 头做 MoA，可以把“参数容量”和“token 级路径适配”同时做强；但工程上这意味着两个 router、两套负载问题、两种缓存路径。只有在你能控制训练基础设施和推理栈时，这种联合设计才值得推进。否则，先在单一维度上做出稳定收益更现实。

最后把选择逻辑压成一句话：

- 如果你的首要目标是**扩大参数容量**，优先看 FFN-MoE。
- 如果你的首要目标是**让 token 在注意力内部走更细的路径**，优先看 MoA/MoH。
- 如果你的首要目标是**稳定部署**，dense 方案往往仍是默认基线。

---

## 参考资料

1. Zhang, Xiaofeng, Yikang Shen, Zeyu Huang, Jie Zhou, Wenge Rong, and Zhang Xiong. *Mixture of Attention Heads: Selecting Attention Heads Per Token*. EMNLP 2022. ACL Anthology. https://aclanthology.org/2022.emnlp-main.278/  
   用途：给出 MoA 的标准定义，即“按 token 选择一部分 attention heads”。

2. Jin, Peng, Bo Zhu, Li Yuan, and Shuicheng Yan. *MoH: Multi-Head Attention as Mixture-of-Head Attention*. ICML 2025. PMLR 267:28233-28255. https://proceedings.mlr.press/v267/jin25l.html  
   用途：给出“把头视为专家”的现代实现，并报告在 ViT、DiT、LLM 上使用 50% 到 90% 的 heads 仍可提升结果；MoH-LLaMA3-8B 在 14 个基准上平均准确率 64.0%，较 LLaMA3-8B 提高 2.4%，使用 75% heads。

3. Huang, Zhigao, Musheng Chen, and Shiyan Zheng. *Dynamic Mixture of Experts for Adaptive Computation in Character-Level Transformers*. Information 2025, 16(6):483. https://www.mdpi.com/2078-2489/16/6/483  
   用途：给出一个很有代表性的反例，说明在较小设置下，MoE 即使验证性能接近 baseline，也可能带来约 50% 的训练时间增长和最高约 56% 的推理速度下降。
