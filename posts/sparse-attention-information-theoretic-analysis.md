## 核心结论

稀疏注意力模式可以用信息论统一描述，但分析时必须拆开三件事，否则很容易把“分布集中”误判成“可以安全裁剪”。

第一，注意力熵
$$
H(A)=-\sum_i \alpha_i \log \alpha_i
$$
衡量的是注意力分布的离散程度。直白地说，它回答的是：一个 query 的权重是平均分给很多位置，还是主要压在少数几个位置。熵越低，分布越尖锐，说明少量 key 已经拿走了大部分概率质量，稀疏化通常更有希望。SparseLinTab 在表格任务上的实验给出了一组具有代表性的现象：第 1 层平均 attention entropy 为 2.15，第 4 层降到 1.82，同时 sparsity ratio 从 0.28 降到 0.15。这说明深层注意力往往更集中，因此可保留的有效连接更少。

第二，Top-k 稀疏化本质上是在用截断后的近似分布 $\hat P$ 逼近完整分布 $P$。若保留集合的总质量为 $1-\tau$，则尾部被删掉的概率质量为 $\tau$。对 Top-k 截断这个特定构造，有一个非常干净的结论：
$$
\mathrm{TV}(P,\hat P)=\tau
$$
其中 TV 是总变差距离。若 $\hat P$ 是保留头部后重新归一化得到的分布，则进一步有
$$
\mathrm{KL}(\hat P\Vert P)= -\log(1-\tau),\qquad
\mathrm{TV}(P,\hat P)=1-e^{-\mathrm{KL}(\hat P\Vert P)}
$$
这里的 KL 散度可以理解为：近似分布相对原分布损失了多少信息。要特别注意方向，这里成立的是 $\mathrm{KL}(\hat P\Vert P)$，不是任意方向都成立。

第三，输出误差不只由“删掉了多少概率”决定，还取决于尾部 value 与头部 value 在表示空间里相差多远。对应公式是
$$
\|\mathrm{Attn}(P,V)-\mathrm{Attn}(\hat P,V)\|_2
=
\tau \cdot \|\mu_{\text{tail}}-\mu_{\text{head}}\|_2
$$
这说明尾部质量小只是必要条件，不是充分条件。某些尾部 token 虽然概率低，但如果它们携带少见实体、关键约束或长程依赖，直接裁剪仍可能引入大误差。vAttention 的价值就在这里：它不是只做确定性 Top-k，而是把“必须保留的 heavy hitters”和“尾部的随机抽样估计”结合起来，并给出每层每头的 $(\varepsilon,\delta)$ 误差保证。

---

## 问题定义与边界

我们只讨论单个 query 对一组 key 的注意力分布，不混入多头、多层和训练细节。先把对象定义清楚。

设完整 softmax 分布为
$$
P=(p_1,\dots,p_N),\quad p_i\ge 0,\quad \sum_{i=1}^N p_i=1
$$
对应的 attention 输出为
$$
o=\sum_{i=1}^N p_i v_i
$$
其中 $v_i$ 是第 $i$ 个 value 向量。value 向量就是最终被加权汇总的内容；注意力权重决定“看谁”，value 决定“拿什么回来”。

如果只保留 Top-k 位置，记保留集合为 $S_k$，尾部质量为
$$
\tau=\sum_{i\notin S_k} p_i
$$
则截断并重新归一化后的分布为
$$
\hat p_i=
\begin{cases}
\frac{p_i}{1-\tau}, & i\in S_k \\
0, & i\notin S_k
\end{cases}
$$

下面几个量分别对应不同问题，不能混为一谈。

| 量 | 公式 | 它回答什么问题 | 常见误读 |
|---|---|---|---|
| 熵 $H(P)$ | $-\sum_i p_i\log p_i$ | 分布是否集中，是否存在稀疏潜力 | 低熵就一定安全 |
| 尾部质量 $\tau$ | $\sum_{i\notin S_k}p_i$ | 删掉了多少概率质量 | 小 $\tau$ 就等于小输出误差 |
| 总变差 TV | $\frac12\sum_i |p_i-\hat p_i|$ | 截断后分布偏离了多少 | TV 小就一定不会影响语义 |
| KL 散度 | $\sum_i \hat p_i\log\frac{\hat p_i}{p_i}$ | 近似分布相对完整分布的信息损失 | 忘记 KL 有方向 |
| $(\varepsilon,\delta)$ 保证 | $\Pr(\|\hat o-o\|>\varepsilon\|o\|)\le\delta$ | 输出超出误差预算的概率多大 | 以为这是分布层面的指标 |

对新手最重要的边界有两个。

第一，低熵不等于一定能安全稀疏。熵只看权重分布，不看被聚合的 value 内容。一个分布可以非常尖锐，但尾部若恰好包含与头部方向差异很大的 value，输出偏差仍可能放大。

第二，Top-k 不等于复杂度就会显著下降。很多实现是先算出全部 query-key 相似度，再从中选 Top-k。这样只把后半段加权求和做稀疏了，前半段“谁重要”的搜索成本仍然接近全量。LLSA 这类方法的重点不只是截断，而是把候选搜索本身也变成分层近似过程。

用一个最小例子先建立量感。设
$$
P=[0.8,0.1,0.05,0.05]
$$
保留 top-2 后，尾部质量为
$$
\tau=0.05+0.05=0.1
$$
归一化分布变为
$$
\hat P=\left[\frac{0.8}{0.9},\frac{0.1}{0.9},0,0\right]
=[0.888\ldots,0.111\ldots,0,0]
$$
此时
$$
H(P)\approx 0.708\ \text{nat},\qquad H(\hat P)\approx 0.349\ \text{nat}
$$
并且
$$
\mathrm{KL}(\hat P\Vert P)= -\log 0.9 \approx 0.105\ \text{nat}
$$

这组数应这样理解：

| 现象 | 数值 | 含义 |
|---|---|---|
| 熵从 0.708 降到 0.349 | 分布更尖 | 头部信息被进一步集中 |
| 尾部质量为 0.1 | 删掉 10% 概率质量 | 这是分布层面的直接损失 |
| KL 约为 0.105 nat | 信息损失不大但非零 | 不能把截断近似当成“无损” |

如果再给每个位置配上 value，例如
$$
V=[1,2,10,12]
$$
则完整输出与稀疏输出并不会只由 $\tau=0.1$ 唯一决定，因为关键还取决于尾部两个大 value 是否与头部均值差异明显。后文会把这件事写成精确公式。

---

## 核心机制与推导

### 1. 头部与尾部的误差分解

把完整输出按保留集合和尾部集合拆开：
$$
o=\sum_{i\in S_k}p_i v_i+\sum_{i\notin S_k}p_i v_i
$$
定义头部条件均值和尾部条件均值
$$
\mu_{\text{head}}=\sum_{i\in S_k}\frac{p_i}{1-\tau}v_i,\qquad
\mu_{\text{tail}}=\sum_{i\notin S_k}\frac{p_i}{\tau}v_i
$$
则完整输出可写为
$$
o=(1-\tau)\mu_{\text{head}}+\tau\mu_{\text{tail}}
$$
而 Top-k 截断后重新归一化得到的输出正好是
$$
\hat o=\mu_{\text{head}}
$$
两式相减：
$$
o-\hat o
=
(1-\tau)\mu_{\text{head}}+\tau\mu_{\text{tail}}-\mu_{\text{head}}
=
\tau(\mu_{\text{tail}}-\mu_{\text{head}})
$$
因此
$$
\|o-\hat o\|_2=\tau\cdot\|\mu_{\text{tail}}-\mu_{\text{head}}\|_2
$$

这个等式非常关键，因为它说明输出误差由两个独立因子共同决定：

| 因子 | 数学量 | 含义 |
|---|---|---|
| 统计因素 | $\tau$ | 你删掉了多少概率质量 |
| 表示因素 | $\|\mu_{\text{tail}}-\mu_{\text{head}}\|_2$ | 被删内容与保留内容在语义空间里差多远 |

因此，稀疏注意力的设计目标不应写成“固定 $k$ 尽量小”，而应写成：

1. 让 $\tau$ 尽量小。
2. 避免把与头部差异很大的 value 全部放进尾部。
3. 在候选搜索本身也昂贵时，把搜索开销一起稀疏化。

### 2. 为什么 TV 与 KL 能写成闭式

对 Top-k 截断后的 $\hat P$，尾部位置全部被置零，头部按 $1-\tau$ 重新归一化。于是对头部位置 $i\in S_k$，
$$
\hat p_i-p_i=\frac{p_i}{1-\tau}-p_i
= p_i\frac{\tau}{1-\tau}
$$
对尾部位置 $i\notin S_k$，
$$
\hat p_i-p_i=-p_i
$$
因此
$$
\sum_{i\in S_k}|\hat p_i-p_i|
=
\sum_{i\in S_k}p_i\frac{\tau}{1-\tau}
=
(1-\tau)\frac{\tau}{1-\tau}
=
\tau
$$
而尾部部分
$$
\sum_{i\notin S_k}|\hat p_i-p_i|
=
\sum_{i\notin S_k}p_i
=
\tau
$$
所以
$$
\mathrm{TV}(P,\hat P)
=
\frac12(\tau+\tau)=\tau
$$

再看 KL。因为只有头部有非零 $\hat p_i$，
$$
\mathrm{KL}(\hat P\Vert P)
=
\sum_{i\in S_k}\hat p_i\log\frac{\hat p_i}{p_i}
=
\sum_{i\in S_k}\hat p_i\log\frac{1}{1-\tau}
\=
\log\frac{1}{1-\tau}\sum_{i\in S_k}\hat p_i
=
-\log(1-\tau)
$$
于是立刻得到
$$
1-\tau=e^{-\mathrm{KL}(\hat P\Vert P)}
\quad\Rightarrow\quad
\mathrm{TV}(P,\hat P)=1-e^{-\mathrm{KL}(\hat P\Vert P)}
$$

这个关系成立的前提是：$\hat P$ 正是由 Top-k 截断再归一化得到的特殊分布。换成一般近似分布，这个闭式关系通常不存在。

### 3. 一个“同样的 $\tau$，误差完全不同”的例子

下面两个例子尾部质量相同，都是 $\tau=0.1$，但输出误差差很多。

设
$$
P=[0.8,0.1,0.05,0.05],\quad S_k=\{1,2\}
$$

情况 A：
$$
V_A=[1,2,1.1,1.2]
$$
此时头部和尾部非常接近，
$$
\mu_{\text{head}}\approx 1.111,\qquad
\mu_{\text{tail}}=1.15
$$
所以
$$
\|o-\hat o\|\approx 0.1\times 0.039=0.0039
$$

情况 B：
$$
V_B=[1,2,10,12]
$$
此时尾部与头部差异很大，
$$
\mu_{\text{head}}\approx 1.111,\qquad
\mu_{\text{tail}}=11
$$
所以
$$
\|o-\hat o\|\approx 0.1\times 9.889=0.9889
$$

结论很直接：同一个尾部质量，不同的 value 几何结构，会导致完全不同的输出误差。这也是为什么只看熵或只看 $\tau$ 都不够。

### 4. vAttention 解决的是什么问题

vAttention 走的是验证式近似路线。它不是假设尾部一定不重要，而是把尾部视为需要统计估计的对象。典型过程可以概括为：

1. 先用确定性规则保留 heavy hitters。
2. 再对剩余尾部做随机采样。
3. 对分子和分母分别做误差分析。
4. 组合成最终的输出误差界。

这里 heavy hitter 的意思是：几乎肯定不能删掉的高权重位置，例如 sink token、局部窗口内的重要位置、或者显式 Top-k 命中的位置。

vAttention 给出的目标形式是
$$
\Pr\bigl(\|\hat o-o\|_2>\varepsilon\|o\|_2\bigr)\le\delta
$$
其中：

| 参数 | 含义 |
|---|---|
| $\varepsilon$ | 允许的相对误差大小 |
| $\delta$ | 误差超出预算的失败概率 |
| smaller $\varepsilon$ | 更高精度，但需要更多保留或更多采样 |
| smaller $\delta$ | 更稳健，但预算更保守 |

它与纯 Top-k 的核心区别不是“也做稀疏”，而是“把稀疏误差转成可量化的统计保证”。

### 5. LLSA 解决的是什么问题

LLSA 解决的是另一个层面的问题：如何避免先做完整的二次复杂度匹配，再谈稀疏。

它的核心思想是多层压缩与递归筛选。设块大小为 $B$，序列长度为 $N$。先把原始 token 按块池化成较粗粒度表示，再继续向上压缩，得到大约 $\log_B N$ 层层级结构。对每个 query，不是在原始 $N$ 个位置上直接做全量排序，而是：

1. 在最粗层只比较少量块。
2. 选出 top-k 候选块。
3. 对这些候选块向下一层展开。
4. 重复直到回到 token 级别。

如果每层扩展的候选数控制在常数级，那么每个 query 的候选访问数近似为
$$
O(k\log N)
$$
全局复杂度写成
$$
O(Nk\log N)
$$
当 $k$ 视为常数时，就是
$$
O(N\log N)
$$

一个 8-token 例子最直观。设 $N=8,B=2,K=1$：

| 层级 | 对象数 | 操作 |
|---|---|---|
| token 层 | 8 | 原始位置 |
| 第 1 层块 | 4 | 每 2 个 token 合并成 1 块 |
| 第 2 层块 | 2 | 再把相邻两块合并 |
| 搜索过程 | 从 2 个粗块中选 1 个，再逐层下钻 | 逐步缩小搜索范围 |

这不是“一次性删掉大多数位置”，而是“先粗筛，再细化”，把“谁值得精看”这一步本身做成层级近似。

---

## 代码实现

下面给出一个可以直接运行的 Python 示例。它完成四件事：

1. 计算熵、尾部质量、TV、KL。
2. 计算完整 attention 输出与 Top-k 稀疏输出。
3. 验证误差分解公式。
4. 演示“同样的 $\tau$，不同 value 导致不同误差”。

```python
import math
from typing import List, Sequence, Tuple


def entropy_nat(p: Sequence[float]) -> float:
    if abs(sum(p) - 1.0) > 1e-12:
        raise ValueError("distribution must sum to 1")
    if any(x < 0 for x in p):
        raise ValueError("distribution must be non-negative")
    return -sum(x * math.log(x) for x in p if x > 0.0)


def topk_sparse(p: Sequence[float], k: int) -> Tuple[List[float], List[int], float]:
    if not 1 <= k <= len(p):
        raise ValueError("k must be in [1, len(p)]")
    order = sorted(range(len(p)), key=lambda i: p[i], reverse=True)
    keep_idx = sorted(order[:k])
    keep_mass = sum(p[i] for i in keep_idx)
    if keep_mass <= 0:
        raise ValueError("keep_mass must be positive")

    q = [0.0] * len(p)
    for i in keep_idx:
        q[i] = p[i] / keep_mass

    tau = 1.0 - keep_mass
    return q, keep_idx, tau


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))


def kl_q_to_p(q: Sequence[float], p: Sequence[float]) -> float:
    total = 0.0
    for qi, pi in zip(q, p):
        if qi > 0.0:
            if pi <= 0.0:
                return math.inf
            total += qi * math.log(qi / pi)
    return total


def weighted_sum(weights: Sequence[float], values: Sequence[float]) -> float:
    return sum(w * v for w, v in zip(weights, values))


def conditional_mean(p: Sequence[float], values: Sequence[float], idx: Sequence[int]) -> float:
    mass = sum(p[i] for i in idx)
    if mass <= 0:
        raise ValueError("conditional set must have positive mass")
    return sum((p[i] / mass) * values[i] for i in idx)


def analyze_case(p: Sequence[float], values: Sequence[float], k: int) -> None:
    q, keep_idx, tau = topk_sparse(p, k)
    tail_idx = [i for i in range(len(p)) if i not in keep_idx]

    full_out = weighted_sum(p, values)
    sparse_out = weighted_sum(q, values)

    mu_head = conditional_mean(p, values, keep_idx)
    mu_tail = conditional_mean(p, values, tail_idx) if tau > 0 else 0.0

    lhs = abs(full_out - sparse_out)
    rhs = tau * abs(mu_tail - mu_head) if tau > 0 else 0.0

    H_full = entropy_nat(p)
    H_sparse = entropy_nat(q)
    TV = total_variation(p, q)
    KL = kl_q_to_p(q, p)

    print("p =", list(p))
    print("q =", [round(x, 6) for x in q])
    print("keep_idx =", keep_idx)
    print("values =", list(values))
    print("H(p) =", round(H_full, 6))
    print("H(q) =", round(H_sparse, 6))
    print("tau =", round(tau, 6))
    print("TV(p, q) =", round(TV, 6))
    print("KL(q || p) =", round(KL, 6))
    print("full_output =", round(full_out, 6))
    print("sparse_output =", round(sparse_out, 6))
    print("mu_head =", round(mu_head, 6))
    print("mu_tail =", round(mu_tail, 6))
    print("error_lhs =", round(lhs, 6))
    print("error_rhs =", round(rhs, 6))
    print()

    assert abs(sum(q) - 1.0) < 1e-12
    assert abs(TV - tau) < 1e-12
    assert abs(KL - (-math.log(1.0 - tau))) < 1e-12
    assert abs(lhs - rhs) < 1e-12


if __name__ == "__main__":
    p = [0.8, 0.1, 0.05, 0.05]

    print("Case A: tail values close to head")
    analyze_case(p, [1.0, 2.0, 1.1, 1.2], k=2)

    print("Case B: tail values far from head")
    analyze_case(p, [1.0, 2.0, 10.0, 12.0], k=2)
```

这段代码在本地直接运行即可，输出会验证三个关键事实：

| 验证项 | 预期结果 |
|---|---|
| `TV(p, q)` 与 `tau` | 完全相等 |
| `KL(q || p)` 与 `-log(1-tau)` | 完全相等 |
| `|full_output - sparse_output|` 与 `tau * |mu_tail - mu_head|` | 完全相等 |

如果把上面的玩具实现翻译成工程里的稀疏注意力流程，通常会抽象成下面三步：

```python
selected = deterministic_filter(scores, k, sink_tokens, local_window)
residual = exclude_all_tokens(selected)
sampled = sample_residual(residual, budget, mode="clt")  # 或 "hoeffding"
approx = combine_with_importance_weight(selected, sampled)
```

它们的职责分工如下。

| 模块 | 作用 | 典型方法 | 为什么需要它 |
|---|---|---|---|
| `deterministic_filter` | 保住主质量路径 | sink、local window、top-k | 防止大权重位置被随机漏掉 |
| `sample_residual` | 估计尾部贡献 | uniform / importance sampling | 尾部不能简单当成零 |
| `combine_with_importance_weight` | 组合头部与尾部估计 | importance weighting | 保持估计无偏或低偏 |
| `denominator_control` | 控制 softmax 分母误差 | 单独校验或联合界 | 只控分子通常不够 |

这里最容易被忽略的是分母。softmax attention 不是普通加权平均，而是“分子除以分母”的归一化结果。若只近似了分子
$$
\sum_i e^{s_i}v_i
$$
却没有控制分母
$$
\sum_i e^{s_i}
$$
则最终输出会因为归一化失真而漂移。很多近似方法在小实验里看起来有效，问题就出在这里：它们保住了部分大权重项，但没有把分母的偏差纳入误差预算。

---

## 工程权衡与常见坑

最常见的误解是“attention 分布看起来很尖，就直接固定 Top-k”。这在不少 head 上确实有效，但它只是经验，不是定理。工程上真正该优先监控的往往不是固定的 $k$，而是每层每头的尾部质量 $\tau$、分布熵和输出误差统计。

下面这张表可直接作为选型参考。

| 方法 | 误差控制 | 额外成本 | 复杂度特征 | 典型坑 |
|---|---|---|---|---|
| Top-k only | 弱，通常只有经验保证 | 低 | 常常仍受全量搜索影响 | 平坦分布 head 失真明显 |
| vAttention verified | 强，可设 $(\varepsilon,\delta)$ | 中，需要采样与预算估计 | 推理期更可控 | 若分母未单独控制会漂移 |
| LLSA 分层 | 中，依赖结构保真 | 中到高，需要多层索引与 kernel | 可做到 $O(N\log N)$ | 实现复杂，训练与推理都要改 |

几个常见坑值得单独展开。

第一，错误使用 KL 的方向。  
对 Top-k 重新归一化分布，容易得到闭式关系的是
$$
\mathrm{KL}(\hat P\Vert P)
$$
不是
$$
\mathrm{KL}(P\Vert \hat P)
$$
原因很简单：$\hat P$ 在尾部为 0，而 $P$ 在尾部通常大于 0，所以
$$
\mathrm{KL}(P\Vert \hat P)
=
\sum_i p_i\log\frac{p_i}{\hat p_i}
$$
在尾部会出现除以 0，通常发散到无穷大。

第二，把低熵误读成低风险。  
熵只描述权重分布是否集中，不描述 value 承载的信息是否关键。一个尾部 token 权重可能只有 0.02，但如果它携带少见实体名、代码符号、长程约束，删掉后仍可能导致输出明显错误。

第三，忽略候选搜索成本。  
很多所谓稀疏实现，只是把最后的加权聚合做稀疏，但 Top-k 候选仍是从全量分数矩阵中找出来的。这样 kernel 看起来更稀疏，端到端复杂度却未必真正下降。对于长上下文，必须把“找谁重要”这件事本身做成近似或分层流程。

第四，统一预算掩盖坏头。  
工程里不同 layer、不同 head 的分布形态差异很大。同样的 $k=16$，某些 head 可能已覆盖 99% 质量，另一些 head 可能只覆盖 70%。统一阈值经常会让少数平坦 head 成为主要误差来源。

第五，只看平均值，不看尾部案例。  
许多实验报告只给平均 perplexity、平均延迟、平均覆盖率，但真正出错的往往是少数 query。若应用场景是代码生成、长文档问答或检索增强生成，坏案例比均值更重要。

工程上更实用的监控面板通常应包含下列指标：

| 指标 | 为什么看它 |
|---|---|
| 每层每头的熵分布 | 判断分布是否天然尖锐 |
| 每层每头的 Top-k 覆盖质量 $1-\tau$ | 比固定 $k$ 更有解释力 |
| 稀疏输出与全量输出的相对误差 | 直接对应功能退化风险 |
| 候选搜索时间占比 | 判断是否真的降了复杂度 |
| 极端 query 的误差分位数 | 防止平均值掩盖坏样本 |

---

## 替代方案与适用边界

如果目标只是短文本推理，且上下文长度还没有把 attention 变成主瓶颈，那么最简单的 Top-k 或窗口注意力通常就够了。优点是实现简单、改动小；缺点是误差通常不可验证，更多依赖经验调参。

如果目标是给已经训练好的大模型做长上下文推理，尤其是需要把 KV cache 放在 CPU、主存或分层存储中，那么 vAttention 更合适。它不要求重训模型，重点是在推理期给出每层每头的 $(\varepsilon,\delta)$ 误差控制，适合“必须知道自己大概会错多少”的场景。

如果目标是从结构上把复杂度压到 $O(N\log N)$，LLSA 更合适。它关注的是架构级改造，适用于超长上下文、图像、视频等 token 数极多、并且可以接受训练和 kernel 重写成本的任务。

| 方案 | 适合阶段 | 上下文长度 | 误差预算 | 复杂度级别 |
|---|---|---|---|---|
| Top-k / Window | 轻量推理改造 | 短到中等 | 经验阈值 | 常见实现仍接近二次 |
| vAttention | 预训练模型推理 | 中长到超长 | 显式 $(\varepsilon,\delta)$ | 取决于候选筛选方式 |
| LLSA | 训练与结构改造 | 超长 | 结构保真，不是逐头验证 | $O(N\log N)$ |
| LLSA + vAttention | 结构改造 + 推理验证 | 超长 | 结构稀疏 + 统计验证 | 更稳，但工程最复杂 |

可以用一个简单问题做选择：

| 你真正要解决的问题 | 更匹配的方法 |
|---|---|
| 先尽快把推理跑快 | Top-k / Window |
| 不重训模型，还要控制近似误差 | vAttention |
| 从架构层面降低长序列复杂度 | LLSA |
| 既要结构级降复杂度，又要推理期统计保证 | LLSA + vAttention |

实务上，最关键的判断不是“哪种方法更先进”，而是“你要的是更快，还是可证明地更快且误差受控”。前者通常靠启发式规则已经够用；后者则必须把信息论指标、统计预算和 kernel 设计放在同一个框架里评估。

---

## 参考资料

1. Claude Shannon, “A Mathematical Theory of Communication”, 1948. 熵定义的经典来源。  
2. Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning, “What Does BERT Look at? An Analysis of BERT’s Attention”, ACL BlackboxNLP 2019. https://aclanthology.org/W19-4828/  
3. “SparseLinTab: Sparse linear self-attention for efficient feature interaction in tabular data”, 2025. 文中 Table 7 给出第 1 层与第 4 层的 entropy / sparsity 对比。 https://www.sciencedirect.com/science/article/pii/S2667295225000832  
4. “vAttention: Verified Sparse Attention for LLMs”, 2025. 给出 deterministic selection + residual sampling + $(\varepsilon,\delta)$ 保证。 https://www.emergentmind.com/papers/2510.05688  
5. “A Mathematical Theory of Top-k Sparse Attention via Total Variation Distance”, 2025. 给出 Top-k 截断下 TV、KL 与 head-tail 误差分解。 https://www.emergentmind.com/papers/2512.07647  
6. Yifan Zhou, “Log-linear Sparse Attention: The First High-Performance Sparse Attention with O(N log N) Complexity”, 2025. 解释 LLSA 的层级压缩与复杂度。 https://zhouyifan.net/blog-en/2025/12/19/20251211-llsa-1/  
7. Wikipedia, “Entropy (information theory)”. 便于快速查公式。 https://en.wikipedia.org/wiki/Entropy_%28information_theory%29  
8. Wikipedia, “Kullback-Leibler divergence”. 便于快速查 KL 基本定义。 https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
