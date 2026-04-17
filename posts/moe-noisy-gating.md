## 核心结论

MoE（Mixture of Experts，专家混合；白话说，就是“很多子模型里只挑少数几个来算”）里的路由器决定每个 token 送给哪些专家。纯 Top-K 路由只看当前分数最高的 $k$ 个专家，训练很容易出现“专家坍缩”：少数专家越来越忙，其他专家长期拿不到样本，最后虽然参数很多，但真正起作用的专家很少。

Noisy Top-K Gating 的核心做法是在路由 logits 上加可学习噪声：

$$
H(x)=W_gx+\operatorname{softplus}(W_{noise}x)\cdot \epsilon,\quad \epsilon\sim \mathcal N(0,1)
$$

这里的 logits 可以理解为“每个专家的原始打分”；`softplus` 是把任意实数变成正数的平滑函数；$\epsilon$ 是标准高斯噪声。训练时加噪，推理时关噪。这样做的效果不是“让路由更乱”，而是让边缘样本在训练早期有机会探索冷门专家，打破“越被选中越强、越强越被选中”的自增强循环。

结论可以压缩成三点：

| 结论 | 作用 | 代价 |
|---|---|---|
| 给路由 logits 加可调噪声 | 提升专家探索，缓解坍缩 | 训练期路由方差增大 |
| 噪声幅度由 $W_{noise}$ 学习 | 不同输入可以有不同探索强度 | 需要额外参数和调参 |
| 训练开噪、推理关噪 | 兼顾训练探索和部署稳定性 | 训练与推理行为不完全一致 |

一个玩具例子足够说明直觉。假设某个 token 对两个专家的 clean logits 是 $[1.0, 0.0]$。如果不用噪声，softmax 后第二个专家概率约为：

$$
\frac{e^0}{e^1+e^0}\approx 0.269
$$

如果再经过 Top-1，第二个专家直接变成 0，完全没有机会。加入噪声后，第二个专家的分数可能偶尔抬高，虽然大多数时候仍然落选，但它终于能看到一部分训练样本，这就是探索价值。

---

## 问题定义与边界

问题先说清楚。MoE 路由的目标不是“把 token 平均分给所有专家”，而是在两个约束下尽量做对：

1. 稀疏性：每个 token 只激活很少几个专家，通常是 Top-1 或 Top-2。
2. 可训练性：路由器和专家都要能稳定学到东西。

如果只有第一个目标，最简单的方法就是纯 Top-K：先算分，再选前 $k$ 个专家。但这会带来一个非常典型的问题。某些专家在随机初始化后稍微占优，就会更常被选中；更常被选中就会得到更多梯度；得到更多梯度后又更强；最后路由越来越偏。这种现象叫路由坍缩，也可以理解为“资源分配被少数热门专家垄断”。

用非技术语言描述，就是一个“越热门越热门”的过程。系统原本想让很多专家分工，但实际训练中变成“班上总是那几个同学被点名回答问题”，其他同学因为一直没有练手机会，看起来就更不擅长，于是以后更不被点名。

Noisy Top-K 的边界也要讲清楚：

| 维度 | 纯 Top-K | Noisy Top-K |
|---|---|---|
| 路由是否确定 | 是 | 训练期否，推理期是 |
| 是否鼓励探索 | 弱 | 强 |
| 是否容易坍缩 | 较高 | 较低 |
| 训练稳定性 | 在小规模场景可接受 | 大规模场景通常更稳 |
| 推理额外开销 | 低 | 与纯 Top-K 基本一致 |

它解决的是“训练阶段的路由探索不足”，不是所有负载均衡问题的万能解。比如通信瓶颈、专家容量上限、跨机调度失败，这些不是单靠噪声能解决的。再比如专家数量很少、数据分布简单时，纯 Top-K 也可能已经够用，此时强行加噪声反而会拖慢收敛。

---

## 核心机制与推导

Noisy Top-K 可以拆成三步看。

第一步，计算 clean logits：

$$
g=W_gx
$$

这里 $x$ 是 token 表示，$W_g$ 是路由权重。$g_i$ 越大，表示第 $i$ 个专家越适合这个 token。

第二步，计算噪声尺度：

$$
s=\operatorname{softplus}(W_{noise}x)
$$

噪声尺度就是“这次允许打分抖动多大”。为什么要用 `softplus`？因为噪声标准差必须非负，而 `softplus(z)=\log(1+e^z)$ 永远大于 0。

第三步，加噪、取 Top-K、再 softmax：

$$
l=g+s\odot \epsilon,\quad \epsilon\sim\mathcal N(0,1)
$$

$$
gate=\operatorname{softmax}(\operatorname{keep\_top\_k}(l,k))
$$

`keep_top_k` 的意思是：只保留前 $k$ 个 logits，其他位置设成 $-\infty$。这样 softmax 后，未入选专家的权重就是 0，仍然保持稀疏路由。

看一个具体推导。设 clean logits 为 $[3,1]$，noise scale 为 $[0.2,1.5]$，采样噪声为 $[0.1,-0.5]$。那么 noisy logits 是：

$$
l=[3+0.2\times 0.1,\ 1+1.5\times(-0.5)]=[3.02,\ 0.25]
$$

如果 $k=2$，softmax 后两个专家都有权重：

$$
p_1=\frac{e^{3.02}}{e^{3.02}+e^{0.25}},\quad
p_2=\frac{e^{0.25}}{e^{3.02}+e^{0.25}}
$$

第二个专家虽然弱，但不是完全没机会。如果 $k=1$，这一次仍然是专家 1 获胜；但关键在于下一次噪声重新采样，某些边界 token 可能转而落到专家 2。训练进行多轮后，冷门专家不再完全“饿死”。

这里有一个常被忽略的点：噪声不是固定超参数，而是输入相关的。也就是说，不同 token 的探索强度不同。某些 token 路由非常明确，$W_{noise}x$ 学出来会很小；某些 token 本身处于多个专家边界附近，模型会学出更大的噪声尺度，让它们更充分地试探不同专家。

从优化角度看，Top-K 是离散选择，梯度传播并不天然平滑。因此工程实现里常配合辅助负载损失，例如 importance loss 和 load loss。importance 可以理解为“平均门控权重是否过度集中”，load 可以理解为“实际被分配 token 的数量是否偏斜”。它们通常不是替代噪声，而是和噪声一起使用。噪声负责提供探索机会，辅助损失负责把这种探索进一步收敛成更均衡的长期分工。

真实工程例子是 Switch Transformer 一类稀疏模型。它们往往使用 Top-1 或 Top-2 路由，并非常关注专家负载是否均匀。因为一旦某个专家在分布式系统里过载，不只是精度受影响，还会直接拖慢整个训练 step，甚至触发 capacity overflow，也就是“专家收件箱满了，后来的 token 只能被丢弃或降级处理”。

---

## 代码实现

下面给一个可以直接运行的 `numpy` 版本，演示 Noisy Top-K 路由。它不是完整训练代码，但足够说明公式如何落地。

```python
import numpy as np

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def noisy_topk_gating(X, W_g, W_noise, k=2, training=True, eps=None):
    """
    X: [batch, d_model]
    W_g: [d_model, n_experts]
    W_noise: [d_model, n_experts]
    """
    clean_logits = X @ W_g
    noise_scale = softplus(X @ W_noise)

    if training:
        if eps is None:
            eps = np.random.randn(*clean_logits.shape)
        noisy_logits = clean_logits + noise_scale * eps
    else:
        noisy_logits = clean_logits

    topk_idx = np.argsort(noisy_logits, axis=-1)[:, -k:]
    masked = np.full_like(noisy_logits, -1e9)

    for i in range(X.shape[0]):
        masked[i, topk_idx[i]] = noisy_logits[i, topk_idx[i]]

    gates = softmax(masked)
    return clean_logits, noise_scale, noisy_logits, gates

# 玩具例子：2 维 token，2 个专家
X = np.array([[1.0, 2.0]])
W_g = np.array([[1.0, 0.0],
                [0.0, 1.0]])
W_noise = np.array([[0.5, 0.5],
                    [0.0, 0.0]])

# 手工指定噪声，保证结果可复现
eps = np.array([[1.0, -1.0]])

clean, scale, noisy, gates = noisy_topk_gating(
    X, W_g, W_noise, k=2, training=True, eps=eps
)

# clean logits = [1, 2]
assert np.allclose(clean, np.array([[1.0, 2.0]]))

# scale 第一列和第二列相同，因为 W_noise 两列相同
expected_scale = softplus(np.array([[0.5, 0.5]]))
assert np.allclose(scale, expected_scale)

# gates 概率和为 1
assert np.allclose(np.sum(gates, axis=-1), np.array([1.0]))

# 推理阶段关噪
_, _, noisy_eval, gates_eval = noisy_topk_gating(
    X, W_g, W_noise, k=2, training=False
)
assert np.allclose(noisy_eval, clean)
assert np.allclose(np.sum(gates_eval, axis=-1), np.array([1.0]))

print("clean logits:", clean)
print("noise scale:", scale)
print("noisy logits:", noisy)
print("train gates:", gates)
print("eval gates:", gates_eval)
```

这段实现里有几个关键点：

| 实现点 | 含义 |
|---|---|
| `clean_logits = X @ W_g` | 计算每个专家的基础打分 |
| `noise_scale = softplus(X @ W_noise)` | 为每个 token、每个专家生成非负噪声尺度 |
| `noisy_logits = clean + scale * eps` | 训练时做探索 |
| `masked = -1e9` | 近似实现未入选专家的 $-\infty$ |
| `softmax(masked)` | 对保留的 Top-K 专家重新归一化 |

如果要写成真实工程代码，通常还会加三类逻辑：

1. capacity 约束：每个专家最多接收多少 token。
2. auxiliary loss：显式约束负载均衡。
3. dispatch/combine：把 token 真正发给专家，再把结果按 gate 权重合并回来。

---

## 工程权衡与常见坑

Noisy Top-K 最常见的误区，是把“加噪声”理解成简单调大学习随机性。实际上，工程上要权衡的是探索与收敛。

| 设置 | 好处 | 风险 |
|---|---|---|
| 噪声很小 | 路由稳定，专家容易收敛 | 打不破坍缩，冷门专家仍然没样本 |
| 噪声适中 | 能探索，又不至于完全随机 | 需要配合负载损失一起调 |
| 噪声很大 | 负载更均匀 | 路由接近随机分发，专家难以专精 |

第一类坑是训练期和推理期没分开。正确做法通常是训练时采样 $\epsilon$，推理时令 $\epsilon=0$。原因很直接：训练要探索，部署要稳定。如果线上推理还保留随机噪声，同一个输入可能被送到不同专家，输出会抖动，这通常不可接受。

第二类坑是只加噪声，不做负载约束。噪声能缓解坍缩，但不保证严格均衡。大规模 MoE 训练里，常见做法是“噪声 + auxiliary load loss + capacity control”一起上。前者负责打破局部最优，后两者负责把整体系统拉回可控状态。

第三类坑是误判噪声来源。Noisy Top-K 不是对专家输出加噪，而是对路由 logits 加噪。这两者效果完全不同。前者会污染专家计算结果，后者只影响“把 token 送给谁”，目标更明确，推理时也更容易关闭。

第四类坑出现在分布式训练。真实工程里，一个专家可能部署在独立设备上。假设某个热门专家突然吸收了大量 token，就会造成跨卡通信拥塞和局部 OOM。Switch Transformer 这类系统之所以重视可训练噪声，不只是为了最终精度，还因为负载均衡本身决定了训练吞吐和稳定性。对于上百到上千专家的系统，如果路由不均衡，问题会先表现为系统级故障，再表现为模型效果问题。

---

## 替代方案与适用边界

Noisy Top-K 不是唯一方案，但它在大规模稀疏路由里很常见，因为它对原有 Top-K 框架改动小，推理阶段几乎不加成本。

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| 纯 Top-K | 专家数少、训练简单 | 实现最直接，推理完全确定 | 易坍缩 |
| Noisy Top-K | 中大规模稀疏 MoE | 探索能力强，部署仍可确定 | 训练调参更复杂 |
| Top-K + Load Loss | 已有稳定路由框架 | 显式控制均衡 | 单独使用时探索不足 |
| Capacity-based Routing | 专家易过载的分布式系统 | 防止单专家爆仓 | 可能丢 token 或做回退 |
| Dense Soft Routing | 小模型、可接受高算力 | 可微、训练平滑 | 推理成本高，不再稀疏 |

可以用两个对比场景理解边界。

玩具场景：只有 16 个专家、单机训练、数据分布不复杂。如果观察到负载已经比较均匀，纯 Top-K 加少量 load loss 往往就够了，Noisy Top-K 不一定带来明显收益。

真实工程场景：上百到上千专家、跨机训练、token 数极大。这时早期训练里任何轻微偏置都会被迅速放大。Noisy Top-K 往往是更安全的默认选项，因为它先解决“冷门专家没有生存空间”的问题，再让辅助损失去优化均衡。

因此选择标准可以简单写成一句话：如果系统的主要矛盾是“探索不足导致坍缩”，优先考虑 Noisy Top-K；如果主要矛盾是“容量控制、通信、时延”，就要把 capacity、负载损失和系统调度一起纳入设计。

---

## 参考资料

1. Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*.
2. ApX Machine Learning, Noisy Top-K Gating 课程页：<https://apxml.com/courses/mixture-of-experts-advanced-implementation/chapter-2-advanced-routing-mechanisms/noisy-top-k-gating>
3. Medium, *Demystifying Mixture of Experts (MoE) Layers*：<https://medium.com/%40dipankar0705018/demystifying-mixture-of-experts-moe-layers-6ba49d9ee62b>
4. Hypercoast MoE VAE 路由实现说明：<https://hypercoast.org/moe_vae/model/>
5. Saguaro MoE 复现笔记：<https://shuqihere.vercel.app/archive/re-implementation/moe>
6. Next.gr 关于稀疏 MoE 路由与负载均衡的说明：<https://next.gr/ai/large-language-models/sparse-mixture-of-experts-at-scale>
7. Next.gr 关于动态路由实现示例：<https://www.next.gr/ai/object-detection/dynamic-token-routing-in-moe-transformers>
