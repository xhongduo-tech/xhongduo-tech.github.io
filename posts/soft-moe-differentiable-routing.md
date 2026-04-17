## 核心结论

Soft MoE 是一种用连续权重替代硬路由的 Mixture of Experts。Mixture of Experts，简称 MoE，可以理解为“多个子网络分工处理输入，再把结果合成”的模型结构。它的关键不在于“每个 token 只选一个专家”，而在于“每个 token 以不同权重参与多个专家的输入构造与输出组合”。

在最常见的 Soft MoE 写法里，路由会出现两次 softmax。第一次决定“每个专家从所有 token 中各取多少信息”，第二次决定“每个 token 从所有专家结果中各取多少信息”。因此，Soft MoE 不是把 token 硬塞进某个专家，而是先做一次软聚合，再做一次软回写。

这样做的直接结果可以概括为：

| 维度 | Soft MoE | 稀疏 / Top-k MoE |
|---|---|---|
| 梯度来源 | 每个 token 对多个专家都有梯度路径 | 主要只有被选中的少数专家收到梯度 |
| 信息连通 | 全局连通、端到端可微 | 路由处包含离散选择 |
| 负载均衡 | 更平滑，较少死专家 | 常依赖额外负载均衡损失 |
| token dropping | 通常不需要靠丢 token 控容量 | 常见容量限制与丢 token 问题 |
| 推理开销 | 路由更密，矩阵运算更多 | 单 token 只激活少数专家，更省在线算力 |
| 训练稳定性 | 通常更稳，路由抖动更小 | 更容易出现路由不稳或专家利用失衡 |

对初学者，一个足够准确的直观图像是：每个 token 不再“排队去唯一柜台”，而是先把自己的表示按比例发送到多个专家形成若干“混合请求”，再把多个专家的回答按比例取回。这样几乎不会出现“某个专家长期接不到样本”或“某个 token 因没被选中而在该层失去大部分梯度”的情况。

---

## 问题定义与边界

Soft MoE 要解决的问题很具体：硬路由把“该 token 送给哪个专家”做成离散决策，训练时容易出现梯度稀疏、token dropping、专家利用不均、路由震荡，以及在多层堆叠时信息传播不连续。

这里先把两个常见术语说清：

| 术语 | 含义 | 为什么麻烦 |
|---|---|---|
| 硬路由 | token 只去 1 个或 k 个专家 | 未被选中的专家拿不到该 token 的训练信号 |
| token dropping | 因专家容量上限或路由冲突，一些 token 被丢弃或回退 | 有效信息没被目标专家处理，训练和推理都可能退化 |

Soft MoE 的边界也必须明确。它不是“任何情况下都优于稀疏 MoE”，而是在下面这些目标下更有优势：

1. 目标是稳定训练，而不是极限压低单次推理延迟。
2. 需要很多层专家模块堆叠，且希望信息在层间持续流动。
3. 希望减少死专家、路由抖动和复杂的辅助损失设计。
4. 更关心总训练效率、可调性和扩展性，而不是单 token 的最省算路径。

反过来，如果系统目标是在线服务的最低延迟、最少带宽或最少 all-to-all 通信，那么稀疏 / Top-k MoE 往往更实用。原因很简单：Soft MoE 的路由是稠密的，虽然训练更平滑，但也意味着更多矩阵计算和更高的通信压力。

理论文献还给了一个重要提醒：Softmax 门控的 MoE 是否容易学，和专家函数是否“强可辨认”有关。强可辨认性可以先理解为：不同专家参数真的会导致可区分的函数行为，而不是很多组参数看起来几乎等价。如果这个条件成立，参数估计和函数估计会快很多；如果不成立，例如专家只是线性层或低阶多项式，收敛会明显变慢。

常见文献给出的量级结论可概括为：

$$
\text{强可辨认、设定较好时：}\quad O_P\!\left(\sqrt{\frac{\log n}{n}}\right)
$$

$$
\text{过参数或可辨认性变弱时：}\quad O_P\!\left(\left(\frac{\log n}{n}\right)^{1/4}\right)
$$

这里的 $n$ 表示样本规模。对初学者，重点不是背这个式子，而是理解它传达的工程含义：专家结构不是实现细节，专家能否形成可区分分工，会直接影响模型是否容易学、能否稳定学。

还要补一层边界：上面的统计结论主要来自“softmax-gated MoE”的一般理论分析，并不等于“视觉 Soft MoE 论文中的具体层实现可以逐条直接套用这些误差界”。但它足以支持一个稳定判断：非线性、可区分的专家，比线性、近似可交换的专家更适合 Soft MoE。

---

## 核心机制与推导

设输入 token 矩阵为 $X\in\mathbb{R}^{m\times d}$，其中 $m$ 是 token 数，$d$ 是隐藏维度，专家数为 $n$。Soft MoE 的一个简化写法是：

\[
z_j=\sum_{i=1}^{m} D_{ij}x_i,\qquad y_j=f_j(z_j),\qquad \hat y_i=\sum_{j=1}^{n} C_{ij}y_j
\]

其中：

- $D\in\mathbb{R}^{m\times n}$：token 到专家输入槽位的分配权重。
- $C\in\mathbb{R}^{m\times n}$：专家输出回写到 token 的组合权重。
- $f_j$：第 $j$ 个专家，通常是一个小型 MLP。
- $z_j$：专家 $j$ 看到的聚合输入。
- $y_j$：专家 $j$ 的输出。
- $\hat y_i$：第 $i$ 个 token 在该层得到的最终输出。

这三个式子可以按“先聚合、后处理、再分发”来读：

| 步骤 | 数学对象 | 白话解释 |
|---|---|---|
| 软分配到专家 | $D$ | 每个专家从所有 token 中按权重取信息 |
| 专家内部变换 | $f_j$ | 每个专家对自己收到的混合表示做非线性处理 |
| 软回写到 token | $C$ | 每个 token 再从所有专家结果中按权重取回信息 |

若 $D$ 的每一列由 token 维做 softmax 得到，那么每个专家接收到的是所有 token 的凸组合。所谓凸组合，就是“权重非负，且总和为 1 的加权平均”。若 $C$ 的每一行由专家维做 softmax 得到，那么每个 token 最终拿到的是所有专家输出的凸组合。

这和 Top-k MoE 的差别在于：Top-k 先做离散选择，再只让少数专家处理该 token；Soft MoE 则保留连续权重，让每个 token 和多个专家之间始终存在信息路径。

一个最小玩具例子最容易把这个过程看清。

设有两个 token、两个专家：

\[
D=
\begin{bmatrix}
0.7 & 0.3\\
0.2 & 0.8
\end{bmatrix},
\qquad
C=
\begin{bmatrix}
0.6 & 0.4\\
0.4 & 0.6
\end{bmatrix}
\]

输入为标量 token：$x_1=1,\;x_2=3$。那么两个专家看到的输入分别是：

\[
z_1=0.7x_1+0.2x_2=0.7+0.6=1.3
\]

\[
z_2=0.3x_1+0.8x_2=0.3+2.4=2.7
\]

如果两个专家分别执行：

\[
f_1(z)=2z,\qquad f_2(z)=z+1
\]

那么专家输出是：

\[
y_1=f_1(1.3)=2.6,\qquad y_2=f_2(2.7)=3.7
\]

再通过回写矩阵 $C$ 把结果发回两个 token：

\[
\hat y_1=0.6\cdot 2.6+0.4\cdot 3.7=3.04
\]

\[
\hat y_2=0.4\cdot 2.6+0.6\cdot 3.7=3.26
\]

这个例子最值得观察的不是数值本身，而是路径结构：

- $x_1$ 和 $x_2$ 都进入了两个专家。
- 两个专家的输出都回流到了两个 token。
- 整层没有“只留一个、其余丢掉”的离散切断。

如果你把它画成图，Soft MoE 更像一张带权双向连通图；Top-k MoE 更像一次离散分流。

从更一般的函数形式看，Soft MoE 可写成输入相关的专家加权和：

\[
f(x)=\sum_{j=1}^{k} \pi_j(x)\, h(x;\eta_j),\qquad
\pi_j(x)=\frac{e^{g_j(x)}}{\sum_{\ell=1}^{k}e^{g_\ell(x)}}
\]

这里：

- $h(x;\eta_j)$ 是第 $j$ 个专家函数。
- $g_j(x)$ 是路由器给专家 $j$ 的打分。
- $\pi_j(x)$ 是 softmax 后的门控权重，表示输入 $x$ 对专家 $j$ 的依赖程度。

这一定义解释了 Soft MoE 为什么既比单一 MLP 更容易形成分工，又比硬路由更稳定：它既允许不同输入偏向不同专家，又不把路由写成不可导的硬选择。

但也要看到退化条件。如果所有专家都是线性的，例如都只是 `Linear(d, d)`，那么

\[
h(x;\eta_j)=W_jx+b_j
\]

代回去后，整体仍可能近似为一个大的输入相关线性组合，专家之间很难形成真正有区分度的功能分工。于是你会得到一个参数很多、但行为上像“多组线性变换求平均”的结构，表达收益和训练收益都会被削弱。

这也是为什么实际实现里，Soft MoE 的专家一般不会只用单层线性层，而是至少使用带激活函数的两层 MLP。

---

## 代码实现

下面给出一个可运行的最小 Python 实现。它只依赖 `numpy`，完整演示两次 softmax、专家聚合、专家计算和结果回写。代码可以直接复制运行。

```python
import numpy as np


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def make_mlp(w1, b1, w2, b2):
    def expert(z):
        hidden = np.tanh(z @ w1 + b1)
        return hidden @ w2 + b2
    return expert


def soft_moe(tokens, gate_logits_in, gate_logits_out, experts):
    """
    tokens: [m, d]
    gate_logits_in: [m, n]   token -> expert logits
    gate_logits_out: [m, n]  expert -> token logits
    experts: list of callables, each maps [d] -> [d_out]
    """
    num_experts = len(experts)
    if gate_logits_in.shape[1] != num_experts:
        raise ValueError("gate_logits_in.shape[1] must match number of experts")
    if gate_logits_out.shape[1] != num_experts:
        raise ValueError("gate_logits_out.shape[1] must match number of experts")

    # D: column-wise softmax
    # 每一列对应一个专家，列和为 1，表示该专家从所有 token 中取多少信息
    D = softmax(gate_logits_in, axis=0)  # [m, n]

    # 每个专家得到一个聚合后的输入向量
    Z = D.T @ tokens  # [n, d]

    # 每个专家独立处理自己的输入
    Y = np.stack([experts[j](Z[j]) for j in range(num_experts)], axis=0)  # [n, d_out]

    # C: row-wise softmax
    # 每一行对应一个 token，行和为 1，表示该 token 从所有专家输出中取多少
    C = softmax(gate_logits_out, axis=1)  # [m, n]

    # 将专家输出按权重回写到 token
    O = C @ Y  # [m, d_out]
    return D, Z, Y, C, O


def main():
    # 两个标量 token，写成 [m, d] 形式
    tokens = np.array([[1.0], [3.0]], dtype=np.float64)

    # 直接用 log(prob) 构造 logits，softmax 后能还原为指定权重
    gate_logits_in = np.log(
        np.array(
            [
                [0.7, 0.3],
                [0.2, 0.8],
            ],
            dtype=np.float64,
        )
    )

    gate_logits_out = np.log(
        np.array(
            [
                [0.6, 0.4],
                [0.4, 0.6],
            ],
            dtype=np.float64,
        )
    )

    # 为了和上文公式精确对应，这里仍使用两个简单专家：
    # f1(z)=2z, f2(z)=z+1
    experts = [
        lambda z: 2.0 * z,
        lambda z: z + 1.0,
    ]

    D, Z, Y, C, O = soft_moe(tokens, gate_logits_in, gate_logits_out, experts)

    print("D =\n", D)
    print("Z =\n", Z)
    print("Y =\n", Y)
    print("C =\n", C)
    print("O =\n", O)

    # 基本正确性检查
    assert np.allclose(D.sum(axis=0), np.ones(2))
    assert np.allclose(C.sum(axis=1), np.ones(2))
    assert np.allclose(Z[:, 0], np.array([1.3, 2.7]))
    assert np.allclose(Y[:, 0], np.array([2.6, 3.7]))
    assert np.allclose(O[:, 0], np.array([3.04, 3.26]))

    # 再给一个稍微更接近真实网络的非线性专家例子
    mlp_experts = [
        make_mlp(
            w1=np.array([[1.5, -0.5]]),
            b1=np.array([0.1, -0.2]),
            w2=np.array([[1.0], [0.5]]),
            b2=np.array([0.0]),
        ),
        make_mlp(
            w1=np.array([[0.7, 1.2]]),
            b1=np.array([-0.1, 0.3]),
            w2=np.array([[0.8], [-0.6]]),
            b2=np.array([0.2]),
        ),
    ]

    _, _, _, _, O_nonlinear = soft_moe(tokens, gate_logits_in, gate_logits_out, mlp_experts)
    print("nonlinear output =\n", O_nonlinear)


if __name__ == "__main__":
    main()
```

如果运行成功，第一部分输出应与上面的手算一致，核心结果是：

```python
O[:, 0] == [3.04, 3.26]
```

这段代码有三个初学者最容易写错的地方。

第一，两个 softmax 的轴不同。  
`D = softmax(gate_logits_in, axis=0)` 表示“对每个专家，在 token 维上归一化”；`C = softmax(gate_logits_out, axis=1)` 表示“对每个 token，在专家维上归一化”。如果两个都写成同一个 `axis`，代码仍然能跑，但语义已经错了。

第二，`D.T @ tokens` 的形状要对上。  
如果 `tokens` 是 `[m, d]`，那么 `D` 必须是 `[m, n]`，转置后才是 `[n, m]`，乘完得到 `[n, d]`，也就是每个专家一个聚合向量。

第三，专家最好是非线性的。  
上面的线性专家只是为了让手算例子能对齐。在真实模型里，专家通常是 FFN/MLP。否则 Soft MoE 很容易退化成“聚合后做近似线性变换，再聚合回来”，参数虽然多，分工却不明显。

再补一个工程上常被忽略的点：Soft MoE 在论文实现里往往不是“每个专家直接看所有原始 token”，而是先把 token 压到少量 slot，再让专家处理这些聚合后的槽位表示。这样能显著降低专家实际处理的序列长度，也是为什么它可以在参数量大幅增加时，推理时间只小幅上升。

一个代表性工程结果来自 Google DeepMind 的 `From Sparse to Soft Mixture of Experts`：在视觉 Transformer 中，用 Soft MoE 替换若干 MLP 块后，`Soft MoE Huge/14` 在 128 个专家、16 个 MoE 层的设置下，参数量超过 ViT Huge/14 的 40 倍，但推理时间只增加约 2%，同时精度更高。这个结果说明，Soft MoE 的关键不只是“软路由”，还包括“专家只处理聚合后的少量表示，而不是把所有 token 都完整复制给所有专家”。

---

## 工程权衡与常见坑

Soft MoE 的代价不是“不能部署”，而是“路由更密，实现和硬件匹配更重要”。如果你只看单 token 激活专家数，稀疏 MoE 更便宜；如果你看训练稳定性、可微性和大规模扩展难度，Soft MoE 通常更省心。

| 常见坑 | 现象 | 原因 | 规避措施 |
|---|---|---|---|
| 专家只用线性层 | loss 能降，但专家分工很弱 | 可辨认性差，整体退化为近似线性混合 | 改为两层或更深的非线性 MLP |
| 两个 softmax 轴写反 | 训练能跑，但效果显著偏差 | 路由语义错误 | 明确 `D` 按 token 维归一化、`C` 按专家维归一化 |
| router 温度过低 | 软路由越来越像硬路由 | softmax 过尖锐，权重接近 one-hot | 调温度、加熵正则、延后降温 |
| 专家过多、数据不足 | 参数暴涨但收益不明显 | 每个专家分到的有效样本不够 | 先增加专家宽度，再增加专家数量 |
| 只看总 loss | 表面正常，实际部分专家几乎闲置 | 专家利用率信息被平均掉 | 监控每层专家使用直方图、门控熵 |
| 忽视通信代价 | 理论 FLOPs 不高，实测却慢 | 稠密路由带来额外带宽和缓存压力 | 先分析硬件瓶颈是算力、显存还是带宽 |
| 直接替换稀疏 MoE | 延迟不降反升 | 两者计算图和通信模式不同 | 单独做 profiling，不要假设可等价替换 |

这里最容易被误解的一点是：Soft MoE 不等于“绝不会塌陷”。它只是比 Top-k 更平滑，更不容易出现彻底的死专家；但如果 router logits 过于尖锐，softmax 仍会非常接近硬选择。

工程上通常会配合这些手段：

1. 对 gate 分布增加熵正则，避免早期过尖锐。
2. 给 router 加 dropout、噪声或轻微扰动，增加探索。
3. 让专家容量和数据复杂度匹配，不要无意义地堆专家。
4. 监控专家使用率、门控熵、每层负载分布，而不只看训练损失。
5. 先把单层跑通，再逐层堆叠，因为多层 Soft MoE 的路由耦合更强。

复杂度上，若简化为 $m$ 个 token、$n$ 个专家、隐藏维 $d$，两次稠密路由会带来大致 $O(mnd)$ 量级的矩阵代价；而 Top-k MoE 更接近 $O(mkd)$，其中 $k\ll n$。这就是经典权衡：

| 目标 | 更优选择 |
|---|---|
| 最低在线延迟 | 稀疏 / Top-k MoE |
| 更稳定训练 | Soft MoE |
| 降低 token dropping | Soft MoE |
| 极大专家池但只激活少数专家 | 稀疏 / Top-k MoE |
| 少调辅助损失、少处理离散路由问题 | Soft MoE |

因此，Soft MoE 的真正工程问题不是“能不能用”，而是“当前瓶颈是优化稳定性，还是在线成本”。这两个目标常常不是同一个最优点。

---

## 替代方案与适用边界

Soft MoE 不是唯一选择。它更像 MoE 设计空间里“训练友好、连通性强、可微到底”的一端；稀疏 / Top-k MoE 则更靠近“在线便宜、条件计算更极致”的另一端。

| 方案 | 适用条件 | 优点 | 限制 |
|---|---|---|---|
| Soft MoE | 大模型训练、希望稳定收敛、减少 token dropping | 全可微、梯度更密、专家利用更平滑 | 路由更密，推理与通信成本更高 |
| 稀疏 / Top-k MoE | 在线服务更看重延迟和吞吐 | 单 token 只激活少数专家，更省在线算力 | 训练更不稳定，更依赖负载均衡与容量设计 |
| MoE-lite / 低秩专家 | 显存紧张、参数预算有限 | 结构轻、部署容易 | 分工上限和容量增益有限 |
| 共享底座 + 专家 Adapter | 想保留主干不动，只局部加专家 | 参数效率较高，迁移方便 | 专家表达能力受插入位置限制 |
| 稠密 Transformer | 中小模型、简单可靠优先 | 实现成熟，调参经验最多 | 条件计算能力有限，参数扩展不如 MoE 灵活 |

对初学者，可以把几类方案这样区分：

- 只叫一个或少数几个专家回答，最省钱，但容易出现路由不稳，这像稀疏 / Top-k MoE。
- 邀请所有专家都参与，再按权重汇总，训练更稳，但更贵，这像 Soft MoE。
- 根本不分专家，所有 token 都走同一条路径，这像普通稠密 Transformer。

更具体地说，Soft MoE 适合这些场景：

1. 你在训练大视觉模型、多模态模型，主要痛点是训练不稳、token dropping、专家扩展困难。
2. 你更关心参数容量增长和收敛质量，而不是每次推理的最低成本。
3. 你希望减少大量针对硬路由的技巧性补丁，例如复杂负载均衡、容量裁剪和 token 回退策略。

它不那么适合这些场景：

1. 你的目标是在线推理延迟最小化。
2. 硬件对稠密矩阵乘或专家间通信不友好。
3. 数据规模不大，模型本身也不大，此时直接用稠密 Transformer 往往更简单、更确定。
4. 你真正需要的是“每个输入只激活极少数专家”的条件计算收益，而不是更稳的训练路径。

一个实用判断标准是：  
如果你的问题首先是“模型学不稳、专家不工作、路由不好调”，Soft MoE 值得优先考虑；如果你的问题首先是“线上慢、带宽紧、GPU 成本高”，那就应优先考虑稀疏 / Top-k MoE 或更轻量的专家方案。

---

## 参考资料

| 来源 | 内容摘要 | 贡献点 |
|---|---|---|
| [From Sparse to Soft Mixture of Experts (Google DeepMind)](https://deepmind.google/research/publications/from-sparse-to-soft-mixture-of-experts/) | Soft MoE 的代表性工程论文，给出两次 softmax 的软分配思想，并报告视觉 Transformer 上的大规模实验结果 | 适合支撑工程层：为什么比稀疏 MoE 更稳定、为何参数量可大幅增加而推理时间仅小幅上升 |
| [On Least Square Estimation in Softmax Gating Mixture of Experts (ICML 2024, PMLR)](https://proceedings.mlr.press/v235/nguyen24f.html) | 在回归设定下分析 softmax-gated MoE 的最小二乘估计，提出 strong identifiability，并比较非线性专家与多项式专家的收敛行为 | 适合支撑理论层：为什么“专家要可区分、最好非线性”不是经验主义，而是与收敛速度直接相关 |
| [A General Theory for Softmax Gating Multinomial Logistic Mixture of Experts (ICML 2024, PMLR)](https://proceedings.mlr.press/v235/nguyen24b.html) | 在分类设定下分析 softmax 门控 MoE，指出当专家参数退化或相互耦合时，参数估计会显著变慢，并讨论改造门控函数后的改善 | 适合补充边界：Softmax 门控虽可微，但并不自动保证样本效率，router 设计仍重要 |
| [Demystifying Softmax Gating Function in Gaussian Mixture of Experts (NeurIPS 2023)](https://trung-tinnguyen.github.io/publication/nguyen2023demystifying/) | 讨论 softmax 门控 Gaussian MoE 的可辨认性、参数平移不变性和门控-专家耦合问题 | 适合补底层背景：为什么 softmax 门控的理论分析长期困难，哪些病理情况会拖慢估计 |
| [Soft Mixture-of-Experts (Emergent Mind)](https://www.emergentmind.com/topics/soft-mixture-of-experts-soft-moe) | 对 Soft MoE 的形式化定义、两次 softmax 机制、工程变体和理论结果做了聚合整理 | 适合快速建立全局图景，但应优先与原论文配合阅读 |
