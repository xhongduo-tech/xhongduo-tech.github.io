## 核心结论

Mixture of Attention Heads 的核心思想是：把 MoE 的“按输入只激活少数专家”搬到多头注意力里。标准 multi-head attention（MHA）要求每个 token 在每一层都经过全部 attention head；而 Mixture of Attention Heads 会先用一个轻量 router 判断“这个 token 当前更需要哪些 head”，然后只激活少数几个 head，例如总共 8 个 head，只运行其中 2 个，其余 head 直接跳过。

这件事的重要性在于，注意力层的成本不只来自参数量，更来自“每个 token 要和多少个 head、多少个 token 交互”。如果每个 token 只激活 $k$ 个 head，而总 head 数是 $H$，那么在 head 这一维上的计算量会从 $H$ 降到 $k$，近似得到：

$$
\text{Head-side compute ratio} \approx \frac{k}{H}
$$

最直白的玩具例子是：一层有 8 个 head，每个 token 只用 2 个 head，那么与 head 数量直接相关的 attention 计算可以近似缩减为原来的

$$
\frac{2}{8}=25\%
$$

也就是理论上约 4 倍压缩。这里的关键词是“理论上”，因为真实速度还受 GPU kernel、内存访问、调度方式影响，后文会专门讲这个边界。

SwitchHead 这类工作说明，这种“按 token 动态选 head”的做法不只是概念成立，而且已经能落到工程实现上。根据其 NeurIPS 2024 论文，在 262M 级别 Transformer 上，它在保持语言建模质量接近基线的前提下，把 attention 相关计算压到约 44%，显存相关占用压到约 27%。这说明它不是简单把模型砍小，而是把“哪些 token 需要哪些 head”这个决策显式交给路由器。

| 配置 | 总 head 数 $H$ | 活跃 head 数 $k$ | 相对 attention 计算比例 |
|---|---:|---:|---:|
| 标准 MHA | 8 | 8 | 100% |
| 稀疏路由 | 8 | 4 | 50% |
| 稀疏路由 | 8 | 2 | 25% |
| 更激进稀疏 | 16 | 2 | 12.5% |

结论可以压缩成一句话：Mixture of Attention Heads 不是减少 head 的总数量，而是让 head 的使用从“所有 token 一视同仁地全部执行”变成“按 token 动态分配”，因此能在尽量保留表达能力的同时，减少大量本来没有必要发生的 attention 计算。

---

## 问题定义与边界

先把问题定义清楚。

在标准 Transformer 里，多头注意力可以写成：

$$
\operatorname{MHA}(X)=\operatorname{Concat}(A^{(1)},A^{(2)},\dots,A^{(H)})W_O
$$

其中第 $h$ 个 head 的输出是：

$$
A^{(h)}=\operatorname{Softmax}\left(\frac{Q^{(h)}K^{(h)\top}}{\sqrt{d_h}}\right)V^{(h)}
$$

这里：

- $X \in \mathbb{R}^{T \times d}$ 是一段长度为 $T$ 的序列表示。
- $H$ 是 head 数。
- $d_h=d/H$ 是单个 head 的维度。
- 每个 head 学的是不同的投影空间和不同的注意力模式。

对新手最重要的一点是：标准 MHA 的“多头”不等于“模型自己决定是否用这些头”。标准做法里，所有 head 默认都会参与计算。也就是说，不管一个 token 当前只需要局部位置信息，还是需要长程依赖、实体对齐、语法线索，它都要把全部 head 跑一遍。

如果序列长度是 $T$，head 数是 $H$，那么粗略地看，head 参与规模可以写成：

$$
\text{Head-wise work} \propto T \times H
$$

而动态稀疏路由的目标，是把它改成：

$$
\text{Sparse head-wise work} \propto T \times k,\quad k \ll H
$$

这不意味着总复杂度自动从 $O(T^2)$ 变成线性。原因很简单：即使只激活了少数几个 head，每个活跃 head 内部仍然可能要对整段序列做 token 两两交互。因此：

- head 稀疏解决的是“每个 token 要跑多少个 head”
- token 稀疏解决的是“每个活跃 head 要看多少个 token”
- 这两件事不是同一个问题

所以 Mixture of Attention Heads 的准确定位是：它减少的是 head 维度上的冗余，不是直接把注意力整体变成线性复杂度。

可以用一个低风险比喻帮助理解，但不能让比喻替代定义。标准 MHA 像是每次处理一个词，都把 8 本不同主题的参考书全部翻一遍；动态路由则是先判断这个词更需要“位置”“实体”“句法”还是“长距离依赖”，只翻最相关的 2 本。比喻的边界在于：书翻少了，不代表答案自动正确，所以 router 的判断质量决定了你省下来的计算是否真的值。

| 方案 | 每个 token 使用的 head 数 | head 维度计算规模 |
|---|---:|---:|
| 标准 MHA | $H$ | $T \times H$ |
| Mixture of Heads | $k$ | $T \times k$ |
| 极端稀疏 | $1$ | $T$ |

真实工程里还要再补两个边界。

第一，数学上省 FLOP，不等于 GPU 上一定更快。GPU 喜欢规则的大矩阵乘，不喜欢大量细碎的 gather、scatter 和动态分支。  
第二，router 本身也有成本。虽然它通常远小于完整 attention，但如果你只看“attention 省了多少”，不看“路由和调度花了多少”，判断就会失真。

因此评估这类方法时，至少要同时看四个指标：

| 指标 | 看什么 | 为什么不能只看 FLOP |
|---|---|---|
| 理论 FLOP | attention 算法本身省了多少乘加 | 不能反映 kernel 和访存开销 |
| Wall-clock time | 真实训练/推理耗时 | 直接反映上线收益 |
| 显存 / 内存 | 激活、KV cache、临时缓冲区 | 长上下文下经常比 FLOP 更先爆 |
| 吞吐 | token/s 或样本/s | 更接近系统侧的最终指标 |

---

## 核心机制与推导

核心机制可以拆成四步：计算路由得分、归一化、选 top-$k$、只执行活跃 head 的 attention。

### 1. 路由打分

设第 $i$ 个 token 对第 $h$ 个 head 的路由打分是 $r_i^{(h)}$。router 通常读取 query 向量或 token 隐状态，因为 query 最接近“当前 token 想获取什么信息”。

最简单的做法是线性打分：

$$
r_i = W_r q_i + b_r,\quad r_i \in \mathbb{R}^{H}
$$

也就是给每个 token 产出一个长度为 $H$ 的分数向量，表示它对每个 head 的偏好。

### 2. 归一化为路由权重

然后用 softmax 变成概率分布：

$$
g_i^{(h)}=\frac{\exp(r_i^{(h)})}{\sum_{\ell=1}^{H}\exp(r_i^{(\ell)})}
$$

这里的 $g_i^{(h)}$ 可以理解为：第 $i$ 个 token 愿意把多少“head 预算”分给第 $h$ 个 head。

### 3. 做 top-$k$ 截断

只做 softmax 还不够，因为 softmax 只是把权重拉开，并没有真正减少计算。要省计算，必须显式截断。

定义一个 top-$k$ 掩码：

$$
m_i^{(h)}=
\begin{cases}
1, & h \in \operatorname{TopK}(g_i, k) \\
0, & \text{otherwise}
\end{cases}
$$

然后只在被选中的 head 上重新归一化：

$$
\tilde g_i^{(h)}=\frac{m_i^{(h)} g_i^{(h)}}{\sum_{\ell=1}^{H}m_i^{(\ell)} g_i^{(\ell)}}
$$

这一步非常关键。它说明：

- softmax 是“连续偏好”
- top-$k$ 是“离散调度”
- 真正的计算节省来自后者，而不是前者

### 4. 只执行活跃 head

最终某个 token 的输出可以写成：

$$
y_i=\sum_{h=1}^{H} m_i^{(h)} \cdot \tilde g_i^{(h)} \cdot \operatorname{Attn}^{(h)}(q_i, K^{(h)}, V^{(h)})
$$

未被选中的 head 因为 $m_i^{(h)}=0$，不只是“输出置零”，而是更理想地“根本不执行”。

这句话对新手尤其重要：  
动态 head 路由不是“先把 8 个 head 都算完，再把其中 6 个删掉”；真正有价值的实现必须是“调度层面只运行那 2 个 head”。

### 玩具例子：8 个 head 里选 2 个

假设某个 token 的 router 分数是：

| head | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| $r_i^{(h)}$ | 0.2 | 1.6 | -0.3 | 0.9 | 2.1 | -1.0 | 0.4 | 1.2 |

softmax 后，权重最大的两个 head 是 head 5 和 head 2。若设 $k=2$，则：

$$
m_i=[0,1,0,0,1,0,0,0]
$$

如果只看 head 2 和 head 5 的归一化结果，那么该 token 的输出就是：

$$
y_i=\tilde g_i^{(2)} A_i^{(2)}+\tilde g_i^{(5)} A_i^{(5)}
$$

其余 6 个 head 对这个 token 不参与前向。

### 复杂度推导：为什么能省

先看标准 MHA。若每个 token 都过全部 $H$ 个 head，则与 head 数直接相关的工作量可写成：

$$
C_{\text{dense}} \propto T \times H
$$

若每个 token 只过 $k$ 个 head，则：

$$
C_{\text{sparse-head}} \propto T \times k
$$

所以 head 维上的相对开销是：

$$
\frac{C_{\text{sparse-head}}}{C_{\text{dense}}}\approx \frac{k}{H}
$$

这就是“8 个 head 只激活 2 个，head 侧计算约为原来 25%”的来源。

但这里必须再强调一次：如果单个活跃 head 依然看全序列，那么每个活跃 head 里仍有 $T^2$ 级 token 交互。于是更完整的粗略写法是：

$$
C_{\text{dense}} \propto H \cdot T^2
$$

$$
C_{\text{head-sparse}} \propto k \cdot T^2
$$

因此 head 稀疏主要是把前面的系数从 $H$ 降成 $k$。

如果进一步像 MoSA 那样，对每个 head 只选择少量 token 参与注意力，那么单个活跃 head 的 token 交互也会下降，文献中会出现近似：

$$
O(k^2 + T)
$$

这里的直觉是：

- 被选中的少量 token 子集内部做较密集计算
- 未被选中的大部分 token 走轻量路径或共享路径
- 所以单 head 的复杂度不再是全局 $T^2$

这说明“head 稀疏”和“token 稀疏”是两层优化，不应混写成同一件事。

---

## 代码实现

实现时最小可用结构通常有三个模块：router、sparse head dispatch、attention aggregation。

- `router` 负责给每个 token 产生 head 偏好分数
- `dispatch` 负责把 token 分发到各自被选中的 head
- `aggregation` 负责把各 head 输出按原 token 位置聚合回来

下面先给一个完全可运行的纯 Python 玩具实现，目的是把逻辑讲透。它不追求高性能，只追求新手能直接跑通并看懂。

### 1. 纯 Python：单个 token 的 top-$k$ head 路由

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def topk_mask(weights, k):
    assert 1 <= k <= len(weights)
    idx = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:k]
    mask = [0] * len(weights)
    for i in idx:
        mask[i] = 1
    return mask, idx

def renorm_topk(weights, mask):
    picked = [w if m else 0.0 for w, m in zip(weights, mask)]
    s = sum(picked)
    if s == 0.0:
        return [0.0] * len(weights)
    return [w / s for w in picked]

def route_one_token(logits, k):
    weights = softmax(logits)
    mask, idx = topk_mask(weights, k)
    sparse_weights = renorm_topk(weights, mask)
    return {
        "dense_weights": weights,
        "mask": mask,
        "sparse_weights": sparse_weights,
        "active_idx": idx,
    }

if __name__ == "__main__":
    # 8 个 head，只激活 2 个
    logits = [0.2, 1.6, -0.3, 0.9, 2.1, -1.0, 0.4, 1.2]
    result = route_one_token(logits, k=2)

    print("dense_weights =", [round(x, 4) for x in result["dense_weights"]])
    print("mask          =", result["mask"])
    print("sparse_weights=", [round(x, 4) for x in result["sparse_weights"]])
    print("active_idx    =", result["active_idx"])

    assert len(result["dense_weights"]) == 8
    assert sum(result["mask"]) == 2
    assert set(result["active_idx"]) == {1, 4}  # 第 2、5 个 head
    assert abs(sum(result["sparse_weights"]) - 1.0) < 1e-9
```

这段代码体现了三个关键动作：

1. 先把 router logits 变成连续概率  
2. 再做 top-$k$ 截断  
3. 最后只对保留下来的 head 重新归一化

如果你运行它，会看到最终只有第 2 和第 5 个 head 非零。

### 2. 纯 Python：把“路由”接到一个极简 attention 上

上面的代码还没真正做 attention。下面补一个最小版本，把“只运行活跃 head”这件事走通。

```python
import math

def dot(xs, ys):
    return sum(x * y for x, y in zip(xs, ys))

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def topk_indices(xs, k):
    return sorted(range(len(xs)), key=lambda i: xs[i], reverse=True)[:k]

def attention_one_query(q, K, V):
    # q: [Dh]
    # K: [T, Dh]
    # V: [T, Dh]
    scores = [dot(q, k) / math.sqrt(len(q)) for k in K]
    probs = softmax(scores)

    out = [0.0] * len(q)
    for p, v in zip(probs, V):
        for j in range(len(q)):
            out[j] += p * v[j]
    return out

def mixture_of_heads_one_token(router_logits, q_heads, K_heads, V_heads, k_active):
    # router_logits: [H]
    # q_heads: [H, Dh]
    # K_heads: [H, T, Dh]
    # V_heads: [H, T, Dh]
    H = len(router_logits)

    gate = softmax(router_logits)
    active = topk_indices(gate, k_active)

    sparse_gate = [0.0] * H
    denom = sum(gate[i] for i in active)
    for i in active:
        sparse_gate[i] = gate[i] / denom

    out = [0.0] * len(q_heads[0])
    for h in active:
        head_out = attention_one_query(q_heads[h], K_heads[h], V_heads[h])
        for j in range(len(out)):
            out[j] += sparse_gate[h] * head_out[j]

    return out, gate, sparse_gate, active

if __name__ == "__main__":
    H, T, Dh = 4, 3, 2

    router_logits = [0.1, 1.8, -0.4, 1.2]
    q_heads = [
        [0.2, 0.5],
        [0.8, 0.1],
        [0.3, 0.3],
        [0.7, 0.6],
    ]
    K_heads = [
        [[0.1, 0.0], [0.3, 0.2], [0.0, 0.6]],
        [[0.7, 0.2], [0.6, 0.1], [0.2, 0.8]],
        [[0.2, 0.9], [0.1, 0.4], [0.3, 0.3]],
        [[0.5, 0.4], [0.4, 0.7], [0.8, 0.1]],
    ]
    V_heads = [
        [[1.0, 0.0], [0.5, 0.2], [0.1, 0.8]],
        [[0.3, 1.2], [0.4, 0.9], [0.8, 0.5]],
        [[1.1, 0.1], [0.6, 0.7], [0.2, 0.4]],
        [[0.9, 0.3], [0.1, 1.0], [0.5, 0.6]],
    ]

    out, gate, sparse_gate, active = mixture_of_heads_one_token(
        router_logits, q_heads, K_heads, V_heads, k_active=2
    )

    print("gate       =", [round(x, 4) for x in gate])
    print("sparse_gate=", [round(x, 4) for x in sparse_gate])
    print("active     =", active)
    print("output     =", [round(x, 4) for x in out])
```

这段代码已经体现出完整链路：

- 路由器先选 head
- 只对被选中的 head 调用 `attention_one_query`
- 最后按稀疏 gate 加权求和

### 3. PyTorch 伪代码：更接近真实模型

真实模型里，通常不会用 Python 循环一个 token 一个 token 地跑，否则肯定慢。更接近工程实现的伪代码如下：

```python
# x: [B, T, D]
# router(x): [B, T, H]
# project_qkv(x): q, k, v with shapes [B, T, H, Dh]

scores = router(x) / temperature                    # [B, T, H]
gates = scores.softmax(dim=-1)                      # [B, T, H]

topk_val, topk_idx = gates.topk(k, dim=-1)          # [B, T, k]
sparse_gates = gates.new_zeros(gates.shape)
sparse_gates.scatter_(-1, topk_idx, topk_val)
sparse_gates = sparse_gates / sparse_gates.sum(dim=-1, keepdim=True)

# 实际工程不会直接三重 for，而是：
# 1. 按 head 把 token 打包
# 2. 每个 head 对自己收到的 token 批量计算
# 3. 再 scatter 回原顺序

output = x.new_zeros(B, T, D)

for h in range(H):
    token_mask = (topk_idx == h).any(dim=-1)        # [B, T]
    if token_mask.sum() == 0:
        continue

    q_h = select_tokens(q[:, :, h, :], token_mask)  # [N_h, Dh]
    k_h = k[:, :, h, :]                             # [B, T, Dh]
    v_h = v[:, :, h, :]                             # [B, T, Dh]

    o_h = attention(q_h, k_h, v_h)                  # 稀疏 token 对应的输出
    output = scatter_back(output, o_h, token_mask, sparse_gates[..., h])
```

这里最值得注意的不是公式，而是执行路径发生了变化：

| 模块 | 标准 MHA | Mixture of Attention Heads |
|---|---|---|
| Q/K/V 投影 | 默认对所有 head 计算 | 可全算，也可只算活跃 head |
| head 选择 | 没有选择，全部执行 | router 先给每个 token 选 top-$k$ |
| attention 计算 | 每个 token 跑全部 head | 只让活跃 head 计算 |
| 输出融合 | 所有 head 拼接后线性投影 | 只聚合活跃 head，未激活 head 为 0 |

### 4. 训练时为什么要加负载均衡损失

如果没有额外约束，router 很容易学成“永远偏爱少数几个 head”。这会导致模型形式上是多头，实际上却只有少数 head 在工作。

因此训练时常会加一个负载均衡项。一个常见写法是：

$$
L_{\text{balance}} = H \sum_{h=1}^{H} \bar p_h \cdot \bar f_h
$$

其中：

- $\bar p_h$ 表示第 $h$ 个 head 的平均路由概率
- $\bar f_h$ 表示第 $h$ 个 head 被真实选中的频率

目标不是强行让所有 head 完全一样，而是防止极端塌缩，让路由分布不要过早集中到极少数 head。

### 5. 推理时为什么能省 KV cache

标准 MHA 下，每个 token 在每个 head 上都要保留 K/V 状态，因此 KV cache 大小与 head 数直接相关。若 token 只在部分 head 上活跃，那么理论上可以只为这些活跃 head 保留状态，KV cache 也会随之下降。

把单层 KV cache 粗略写成：

$$
\text{KV cache size} \propto T \times H \times d_h
$$

若平均只保留 $k$ 个活跃 head，则有机会下降到：

$$
\text{Sparse KV cache size} \propto T \times k \times d_h
$$

实际工程是否能完全吃到这部分收益，要看缓存布局是否支持稀疏存储，以及是否有额外索引开销。

---

## 工程权衡与常见坑

Mixture of Attention Heads 真正难的地方，不是公式，而是工程落地。

### 1. 路由塌缩

最常见的问题是路由塌缩。也就是 router 总是选那几个 head，剩余 head 长期闲置。

典型现象包括：

- 某 1 到 2 个 head 获得绝大多数 token
- 其余 head 的平均选中频率接近 0
- 训练初期 loss 看起来下降，但中后期效果不稳

后果是：

- 名义上有很多 head，实际有效容量很小
- 模型 specialization 不够
- 泛化可能下降

常见规避方式：

- 加 load-balance loss
- 训练前期提高 router temperature
- 在早期使用更软的 gate，再逐步退火到更硬的 top-$k$

### 2. 容量拥塞

第二类问题是 capacity，也就是单个 head 在一个 batch 内能接收多少 token。

若某个 head 被太多 token 选中，而实现里没有容量上限，就容易出现：

- 某些 head 特别忙
- 中间激活尺寸不稳定
- 显存峰值波动明显
- 单步延迟抖动

一个常见的容量上限写法是：

$$
C = \alpha \cdot \frac{B \times T \times k}{H}
$$

其中：

- $B$ 是 batch size
- $T$ 是序列长度
- $k$ 是每个 token 激活的 head 数
- $H$ 是总 head 数
- $\alpha$ 是 capacity factor，通常大于 1

它表示：在理想均匀分配下，每个 head 平均会收到 $\frac{B \times T \times k}{H}$ 个 token，再乘一个安全系数做缓冲。

### 3. 稀疏不等于快

这是最容易误判的一点。很多人第一次实现这类方法，都会看到 profiler 里 attention FLOP 降了，但训练吞吐没升，甚至变慢。

原因通常不是理论错，而是实现方式错。GPU 擅长：

- 大批量
- 规则形状
- 连续内存
- 少分支

GPU 不擅长：

- 大量小 tensor
- 不规则 gather / scatter
- 高频动态分支
- Python 级循环调度

所以如果你的实现是“对每个 token 判断一次，再在 Python 里一个 head 一个 head 地跳着跑”，结果往往会比密集 attention 更慢。

真正更靠谱的实现思路是：

1. 先把选择了同一 head 的 token 打包  
2. 对每个 head 批量运行  
3. 最后一次性 scatter 回原位置  
4. 尽量减少小 kernel 数量  
5. 能 fuse 的地方尽量 fuse

### 4. 路由过尖，训练不稳定

若训练一开始就让 top-$k$ 很硬、temperature 很低，router 会过早做出高置信度但低质量的决策，梯度信号容易很差。

常见症状包括：

- 训练早期 loss 抖动明显
- 不同 seed 方差很大
- 某些 head 迅速独占流量

常见处理办法：

- 训练初期使用较高 temperature
- 先用 soft routing，后期再硬化
- 用 straight-through 近似或 soft top-$k$
- 对 router 单独设较小学习率

### 5. KV cache 和索引管理复杂

理论上 head 稀疏可以省 KV cache，但实际会把缓存结构从“规则矩阵”变成“带索引的动态结构”。

这会带来几个额外问题：

- 某个 token 在哪些 head 上有缓存，要额外记录
- 不同 head 的有效 token 数不一致，布局变复杂
- 解码阶段要在“省内存”和“索引开销”之间平衡

这也是为什么很多论文能显著降低理论 cache，但工业实现里往往需要重新设计缓存布局，才能把收益稳定拿出来。

### 6. 可解释性不能被夸大

路由统计确实比标准 MHA 更容易观察，因为你能直接看到“某类 token 常激活哪些 head”。但这不等于这些 head 一定有稳定的人类可读语义。

更稳妥的说法是：

- 路由日志是很好的诊断信号
- 它可以帮助你发现塌缩、拥塞、偏置
- 但它不是自动生成的机制解释

| 常见坑 | 现象 | 后果 | 规避方式 |
|---|---|---|---|
| 路由塌缩 | 少数 head 获得绝大多数流量 | 容量浪费，泛化下降 | load-balance loss、温度调度 |
| 容量溢出 | 某 head 分到过多 token | 显存峰值和延迟抖动 | 设置 capacity factor |
| 稀疏执行不规则 | FLOP 降了但速度没升 | GPU 利用率低 | token 打包、减少小 kernel、fused 实现 |
| 路由过尖 | 训练早期 top-$k$ 太硬 | 梯度不稳定 | soft routing、退火、straight-through |
| 缓存管理复杂 | KV cache 变成动态结构 | 推理代码复杂化 | 统一索引布局、预分配缓冲区 |
| 解释过度 | 看到 head 被选中就强行赋语义 | 误判机制 | 把路由统计当诊断，而非证明 |

---

## 替代方案与适用边界

理解这类方法时，最有效的分类方式是：稀疏发生在哪一维。

### 1. SwitchHead：主要在 head 维做稀疏

SwitchHead 的重点是：每个 token 不再经过全部 head，而是只经过少数被 router 选中的 head。

它的主要收益是：

- 减少 attention 相关 FLOP
- 减少部分 memory / KV cache 开销
- 对现有 Transformer 改造相对直接

它适合的场景是：

- 你已经有标准 Transformer
- 主要瓶颈在 attention，而不是 FFN
- 你希望较低侵入地做稀疏加速

### 2. MoSA：同时在 head 和 token 两维做稀疏

MoSA 更进一步。它不仅问“哪个 head 应该工作”，还问“工作中的 head 应该关注哪些 token”。

因此它更适合长上下文场景，因为：

- 不只是减少工作 head 的数量
- 还减少每个活跃 head 内部的 token 交互范围

这也是它比单纯 head 稀疏更激进的地方，但代价也更明显：

- 稀疏模式更复杂
- kernel 实现要求更高
- 调试难度更大

### 3. MoH：把多头注意力整体重写为 mixture

MoH 的重点不是只追求最极端的稀疏，而是把多头注意力整体改写成 mixture 结构，让每个 token 选择更合适的 head 组合，并用加权求和替代标准的等权汇总。

它的含义更接近：

- 多头不是固定并联模块
- 多头可以被看作一组可路由专家
- 不同 token 可以使用不同的 head 子集和权重

这类方法通常更强调“表达力与效率的平衡”，而不是单一维度上的最低 FLOP。

| 方法 | 稀疏位置 | 主要收益 | 适合场景 | 主要代价 |
|---|---|---|---|---|
| SwitchHead | head 维度 | attention FLOP、KV cache 下降 | 现有 Transformer 加速、推理省显存 | 动态调度实现复杂 |
| MoSA | head + token 维度 | 长序列复杂度进一步下降 | 长上下文预训练、长文档建模 | 稀疏模式更复杂，kernel 要求更高 |
| MoH | mixture 式 head 路由 | 在较少活跃 head 下保持或提升性能 | 追求表达力与效率平衡 | 训练策略更敏感，设计空间更大 |

### 4. 什么场景下不值得做

这类方法并不是通用答案，以下几种情况收益可能并不高。

第一，序列本来就不长。  
如果你的任务只有几百 token，attention 本身不是瓶颈，那么 head 稀疏节省的空间可能远小于系统复杂度增加的成本。

第二，瓶颈主要在 FFN。  
大模型里 FFN 往往占很大一部分计算。如果 attention 不是主瓶颈，那么优先做 MoE-FFN、量化、蒸馏，通常更直接。

第三，部署环境不适合动态稀疏。  
如果你的 GPU kernel、推理引擎、缓存系统都围绕规则 dense tensor 优化，那么理论省下来的 FLOP 可能换不成真实速度。

第四，你追求的是超长上下文极限复杂度。  
如果目标是把长序列成本大幅压下来，单纯 head 稀疏通常不够，因为活跃 head 内部可能仍是 $T^2$。这时更该看：

- token 稀疏
- 局部窗口
- chunk attention
- retrieval attention
- 低秩近似

所以适用边界可以压缩成一句话：  
Mixture of Attention Heads 更适合“attention head 存在明显冗余、且系统能支持动态路由”的场景；如果你真正要解决的是超长上下文的 token 两两交互成本，仅靠 head 稀疏通常不够。

---

## 参考资料

1. SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention，NeurIPS 2024。核心贡献是把 MoE 式路由引入 attention head 选择，并在 262M Transformer 上报告了约 44% 的计算占比和约 27% 的内存占比，同时保持接近基线的语言建模效果。论文页面：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/87be61bf9338389702712f5e9754a986-Abstract-Conference.html>

2. Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing，arXiv:2505.00315，2025-05-01。核心贡献是把动态稀疏推进到 token 选择层面，使每个 attention head 只处理被路由到的 token 子集，并给出从 $O(T^2)$ 到近似 $O(k^2 + T)$ 的复杂度讨论。索引页：<https://huggingface.co/papers/2505.00315>

3. MoH: Multi-Head Attention as Mixture-of-Head Attention，arXiv:2410.11842，2024-10-15。核心贡献是把多头注意力重新表达为 mixture 结构，使 token 可以选择更合适的 head 组合，并通过加权汇聚替代标准等权汇聚。项目仓库：<https://github.com/SkyworkAI/MoH>

4. NeurIPS 2024 SwitchHead 会议页面。适合先看摘要与主结果，再进入论文正文核对实验设定和指标定义。页面：<https://neurips.cc/virtual/2024/poster/96404>

5. 如果只是入门索引，可以看论文聚合页帮助快速定位相关工作；但涉及公式、复杂度、实验数字时，应回到原论文或官方项目页核对。MoH 索引：<https://www.emergentmind.com/articles/2410.11842>
