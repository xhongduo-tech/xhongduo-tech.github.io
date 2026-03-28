## 核心结论

Top-K 路由策略是 MoE（Mixture of Experts，专家混合，意思是“很多子网络里只激活少数几个”）的核心入口。它先用一个路由器把每个 token 映射成对各个专家的概率分布，再只保留概率最高的前 $k$ 个专家，其余专家直接置零。这样做的目标不是“让所有专家都参与”，而是“让每个 token 只走少量路径”，从而在参数规模很大时仍把单次计算量控制在可接受范围内。

它的标准形式可以写成：

$$
p(x)=\text{softmax}(W_r x)
$$

其中 $x$ 是 token 表示，$W_r$ 是路由器参数，$p(x)$ 是该 token 对所有专家的概率。随后执行 Top-K：

$$
S_k(x)=\text{TopK}(p(x), k)
$$

只把 token 分发给集合 $S_k(x)$ 中的专家，并按对应权重聚合输出。

问题不在“能不能路由”，而在“会不会塌缩”。如果路由器长期偏爱少数专家，就会出现 expert collapse（专家塌缩，意思是“理论上很多专家存在，实际上总是同几个在干活”）。这时有效专家数会退化，极端时接近 1，MoE 失去意义。

所以 Top-K 路由几乎总要配负载均衡项。Switch Transformer 中常见辅助损失是：

$$
L_{\text{aux}}=\alpha N \sum_{i=1}^{N} f_i P_i
$$

其中 $f_i$ 是专家 $i$ 的实际负载比，$P_i$ 是路由器给专家 $i$ 的平均概率，$N$ 是专家数，实验里常见 $\alpha \approx 0.01$。它的作用不是直接规定“每个专家必须一样忙”，而是持续惩罚“概率高且已经过载”的专家。

| 路由步骤 | 做什么 | 激活专家数 | 计算/通信影响 |
|---|---|---:|---|
| 线性投影 | $W_r x$ 产生 logits | 全部专家可见 | 开销小 |
| softmax | 得到概率分布 $p(x)$ | 全部专家有概率 | 仍是稠密分布 |
| Top-K 筛选 | 只保留前 $k$ 个 | 每个 token 只激活 $k$ 个 | 计算显著下降 |
| 掩码分发 | 把 token 送入选中专家 | 稀疏激活 | 跨设备通信受路由分布影响 |

一个玩具例子：8 个专家、$k=2$。某个 token 的路由概率是 $[0.02,0.05,0.60,0.30,0.01,0.01,0.00,0.01]$，那么只选专家 3 和 4，权重分别是 0.60 和 0.30。其余 6 个专家在这个 token 上完全静默。$k$ 越小，单 token 计算越省；但 $k$ 太小也更容易导致路由过度集中。

---

## 问题定义与边界

Top-K 路由要解决的问题很具体：在专家总数很多时，让每个 token 只使用极少数专家，从而把总参数量和单次 FLOPs 解耦。这里的 FLOPs 是浮点运算次数，白话说就是“模型实际干了多少数学计算”。

假设一个稠密前馈层有 1 万亿参数，那么每个 token 都会走完整条路径，计算成本极高。MoE 的思路是把这个大层拆成很多专家，每次只激活 $k$ 个。这样总参数量可以继续增大，但每个 token 的实际计算接近“只用了几个专家”的成本。

它的边界也很清楚：

1. Top-K 只解决“稀疏激活”，不自动解决“均匀利用”。
2. 路由概率是可学出来的，因此会偏。
3. 一旦偏到少数专家，训练速度、显存、通信、吞吐都会恶化。

一个新手能看懂的例子：100 个 token、4 个专家、$k=1$。理想情况是每个专家各处理 25 个 token。但如果路由器总偏向专家 1，就可能出现分配 $[70,10,10,10]$。这时虽然模型“名义上”有 4 个专家，但系统资源会集中耗在一个专家上，其他专家长期学不到有用参数。

可以用两个指标描述这种偏差：

- $f_i$：实际负载比，意思是“这一批 token 里，最终有多少比例真的被分给专家 $i$”。
- $P_i$：概率均值，意思是“路由器平均有多想把 token 送给专家 $i$”。

理想与偏置负载可以直接对比：

| 场景 | 专家1 | 专家2 | 专家3 | 专家4 | 说明 |
|---|---:|---:|---:|---:|---|
| 理想 $f_i$ | 0.25 | 0.25 | 0.25 | 0.25 | 实际负载均匀 |
| 理想 $P_i$ | 0.25 | 0.25 | 0.25 | 0.25 | 路由偏好均匀 |
| 偏置 $f_i$ | 0.70 | 0.10 | 0.10 | 0.10 | 专家1过载 |
| 偏置 $P_i$ | 0.40 | 0.20 | 0.20 | 0.20 | 路由器已明显偏心 |

这里要注意一个边界条件：即使 $P_i$ 看起来没那么极端，只要 argmax 或 top-k 结果长期集中，$f_i$ 仍会快速失衡。也就是说，训练里要同时监控“概率分布”和“真实落点”，只看一个不够。

---

## 核心机制与推导

Top-K 路由的核心机制可以拆成四步：

1. 对 token 做线性投影，得到每个专家的打分。
2. 对打分做 softmax，得到概率分布。
3. 取前 $k$ 个最大概率的专家。
4. 只把 token 送给这 $k$ 个专家，并按权重聚合输出。

形式化地写：

$$
z(x)=W_r x
$$

$$
p_i(x)=\frac{e^{z_i(x)}}{\sum_{j=1}^{N} e^{z_j(x)}}
$$

再对 $p(x)$ 取 Top-K。若选中专家集合为 $S_k(x)$，则常见聚合方式是：

$$
y(x)=\sum_{i \in S_k(x)} \tilde{p}_i(x)\,E_i(x)
$$

其中 $E_i(x)$ 是第 $i$ 个专家输出，$\tilde{p}_i(x)$ 是保留后的权重，常见实现会在 Top-K 后再归一化一次。

为什么会塌缩？因为主任务损失只关心“最终预测对不对”，不关心“是不是所有专家都被公平使用”。如果专家 1 在当前阶段最容易帮模型降损失，梯度就会推动更多 token 继续走专家 1，这会形成正反馈。

辅助损失正是用来打断这个正反馈。定义如下：

$$
f_i=\frac{1}{T}\sum_{x \in B}\mathbf{1}(\text{expert}(x)=i)
$$

$$
P_i=\frac{1}{T}\sum_{x \in B} p_i(x)
$$

$$
L_{\text{aux}}=\alpha N\sum_{i=1}^{N} f_i P_i
$$

这里：

- $T$ 是 batch 内 token 数。
- $\mathbf{1}$ 是指示函数，白话说就是“如果条件成立就记 1，否则记 0”。
- $f_i$ 表示真实负载。
- $P_i$ 表示平均偏好。

为什么这个式子能推动均衡？因为在均匀分布时，$f_i=P_i=1/N$，于是：

$$
\sum_i f_iP_i = N \cdot \frac{1}{N}\cdot\frac{1}{N}=\frac{1}{N}
$$

这是希望达到的低值区域。若某个专家同时“被选得多”且“被偏爱得多”，它对应的 $f_iP_i$ 会变大，整体损失上升，反向传播就会压低这个专家的路由概率。

还是看一个玩具例子。4 个专家时，若均匀分布：

$$
f_i=P_i=0.25
$$

则：

$$
\sum_i f_iP_i = 4 \times 0.25 \times 0.25 = 0.25
$$

若专家 1 过载，出现 $f_1=0.7,\ P_1=0.4$，其单项贡献就是：

$$
f_1P_1=0.28
$$

这一个专家的贡献就已经比均匀情况下单专家的 $0.0625$ 大很多。若其他专家再补上，总和会显著高于 0.25。于是优化器会学到一个趋势：别再把那么多概率堆给已经拥挤的专家。

这不是硬约束，而是软惩罚。软惩罚的意思是“倾向于均衡，但在必要时允许短期偏离”。这很重要，因为某些 token 确实更适合某类专家，完全强制平均反而会伤害主任务效果。

---

## 代码实现

下面用一个最小可运行的 Python 例子演示 Top-K 路由和辅助损失。为了便于理解，这里不依赖深度学习框架，只用 `numpy` 展示核心逻辑。

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def topk_route(tokens, W_r, k=2, alpha=0.01):
    """
    tokens: [T, d_model]
    W_r:    [num_experts, d_model]
    """
    logits = tokens @ W_r.T                     # [T, N]
    probs = softmax(logits, axis=1)             # [T, N]

    topk_idx = np.argsort(-probs, axis=1)[:, :k]
    topk_prob = np.take_along_axis(probs, topk_idx, axis=1)

    # 常见实现会对保留下来的 top-k 权重再归一化
    topk_prob = topk_prob / np.sum(topk_prob, axis=1, keepdims=True)

    T, N = probs.shape
    dispatch_mask = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        dispatch_mask[t, topk_idx[t]] = topk_prob[t]

    # 这里用 top-1 落点统计实际负载比 f_i，贴近 Switch 风格
    assigned = topk_idx[:, 0]
    counts = np.bincount(assigned, minlength=N)
    f = counts / T
    P = probs.mean(axis=0)

    aux_loss = alpha * N * np.sum(f * P)
    return probs, topk_idx, dispatch_mask, f, P, aux_loss

# 玩具数据
tokens = np.array([
    [1.0, 0.0],
    [0.9, 0.1],
    [0.0, 1.0],
    [0.2, 0.8],
], dtype=np.float32)

W_r = np.array([
    [2.0, 0.0],   # expert 0 偏好第一维
    [0.0, 2.0],   # expert 1 偏好第二维
    [1.0, 1.0],   # expert 2 比较平均
], dtype=np.float32)

probs, topk_idx, mask, f, P, aux_loss = topk_route(tokens, W_r, k=2, alpha=0.01)

assert probs.shape == (4, 3)
assert topk_idx.shape == (4, 2)
assert mask.shape == (4, 3)
assert np.allclose(mask.sum(axis=1), 1.0)  # top-k 归一化后每行和为 1
assert np.isclose(f.sum(), 1.0)
assert aux_loss > 0

print("router probs:\n", probs)
print("top-k index:\n", topk_idx)
print("load ratio f:", f)
print("mean prob P:", P)
print("aux loss:", aux_loss)
```

如果换成 PyTorch，核心逻辑通常是：

1. `router_logits = router(x)`
2. `router_probs = softmax(router_logits, dim=-1)`
3. `topk_val, topk_idx = torch.topk(router_probs, k, dim=-1)`
4. 用 `scatter` 或 `scatter_add` 生成 dispatch mask
5. 用 `torch.bincount` 统计各专家收到的 token 数，计算 $f_i$
6. 用 `router_probs.mean(dim=0)` 计算 $P_i$
7. `aux_loss = alpha * num_experts * torch.sum(f * P)`

真实工程例子是在分布式训练中的 Switch 或 Mixtral 类 MoE 层。假设 64 个专家分布在 8 张卡上，每张卡承载 8 个专家。路由器先决定 token 去哪些专家，然后系统需要把 token 跨卡发送到对应设备执行前馈层，再把结果取回。如果某几个专家长期拥挤，不只是“学习不均匀”，而是会直接带来跨卡拥塞、单卡 OOM、尾延迟上升。

这里还要引入 capacity factor（容量系数，意思是“每个专家允许接收的 token 上限相对平均值放大多少”）。一个常见近似是：

$$
\text{capacity} = \left\lceil \text{capacity\_factor} \times \frac{T \cdot k}{N} \right\rceil
$$

如果某专家收到的 token 超过上限，超出的 token 可能被丢弃、回退或者重路由。容量太小会掉 token，容量太大又会浪费内存并削弱均衡约束。

---

## 工程权衡与常见坑

第一类坑是辅助损失太弱。$\alpha$ 太小、没加、或者实现错了，都会让 expert collapse 很快出现。Switch Transformer 经验里常见 $\alpha=0.01$，不是理论常数，但它说明这个量级足以对路由器产生稳定约束，又不至于压过主任务损失。

第二类坑是只监控最终 loss，不监控路由指标。MoE 训练早期常见现象是主任务 loss 正常下降，但某层专家利用率已经高度失衡。等到吞吐下降或显存爆炸再看，问题通常已经积累很久。

建议至少监控这些量：

| 指标 | 含义 | 异常信号 | 常见处理 |
|---|---|---|---|
| $f_i$ 方差 | 实际负载是否集中 | 方差持续升高 | 增大 $\alpha$，调 capacity |
| $P_i$ 方差 | 路由偏好是否集中 | 少数专家概率长期偏高 | 检查 router 初始化与学习率 |
| `aux_loss` | 均衡惩罚强度 | 长期过高或接近 0 | 过高说明失衡，过低可能约束失效 |
| dropped token ratio | 因容量不足被丢弃比例 | 上升明显 | 调大 capacity factor |
| per-expert token count | 每专家实际 token 数 | 头部专家远超均值 | 检查通信与负载均衡策略 |

第三类坑是路由精度。很多实现会让 router 用 fp32，即使主干网络在 bf16 或 fp16。原因很直接：softmax 对数值精度敏感，路由 logits 如果抖动过大，top-k 结果会不稳定，进而放大负载偏斜。

第四类坑是把“均匀”误解成“越平均越好”。实际上，完全平均不一定最优。某些任务里，专家确实会形成功能分工。如果你把均衡损失压得过重，模型可能学不出清晰专家专长。工程上追求的是“避免塌缩”，不是“消灭差异”。

第五类坑发生在分布式环境。单机上看每个专家负载似乎正常，但如果专家跨设备部署，真正要看的不是本地统计，而是全局统计。也就是说，`torch.bincount` 得到的 expert count 往往需要先做 all-reduce，再算全局 $f_i$。否则你以为系统平衡，实际上只是每张卡各自看起来平衡。

---

## 替代方案与适用边界

Top-K + 辅助损失不是唯一选择，它只是当前工程上最常见、最稳定的一类。

一种替代思路是 auxiliary-loss-free load balancing，比如 ALF-LB。这类方法不依赖额外损失项，而是在路由阶段直接给专家打分加动态 bias。白话说，就是“检测谁太忙，就在打分时先扣一点；谁太闲，就先补一点”，把流量往空闲专家推。

另一类替代是动态路由。它不固定每个 token 激活 $k$ 个专家，而是按 token 难度决定激活几个。简单 token 激活 1 个专家，复杂 token 激活更多。好处是更灵活，坏处是更难训，超参数也更多。

几种方法可以对比看：

| 方法 | 是否需要 aux loss | 是否依赖梯度均衡 | 适用规模 | 特点 |
|---|---|---|---|---|
| Top-K + aux | 是 | 是 | 中到超大规模 | 实现成熟，最常见 |
| Top-1 Switch + aux | 是 | 是 | 超大规模 | 通信更省，路由更简单 |
| ALF-LB 类偏置法 | 否 | 弱依赖或不依赖 | 大规模 | 直接在路由时纠偏 |
| Threshold / Dynamic K | 视实现而定 | 通常需要 | 中大规模 | 表达力强，但更复杂 |

什么时候仍然优先用 Top-K + aux？

1. 你需要一个成熟、可解释、论文与工业实践都很多的方案。
2. 你希望先把系统跑稳，再逐步优化吞吐。
3. 你能接受引入一个额外损失项，并有监控基础设施。

什么时候它不一定最好？

1. 你对通信极度敏感，想进一步减少跨设备交换，这时 Top-1 可能更合适。
2. 你发现 aux loss 很难调，且负载失衡主要来自系统层而非学习层，可以考虑偏置式均衡。
3. 你希望按 token 难度自适应分配计算量，可以考虑动态路由，但训练复杂度会明显上升。

---

## 参考资料

| 标题 | 来源 | 关注点 |
|---|---|---|
| Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | arXiv / 论文解读 | Top-1 routing、负载均衡损失、超大规模训练 |
| Switch Transformer 笔记 | DOCSAID | $f_i$、$P_i$、$L_{\text{aux}}$ 的公式解释 |
| Switch Transformer Top-1 Routing 解析 | 技术博客 | 路由实现、capacity factor、工程细节 |
| Auxiliary-Loss-Free Load Balancing | 技术综述 | 不用辅助损失的负载均衡思路 |
| MoE 负载均衡实战文章 | 工程博客 | expert collapse、监控指标、调参经验 |

- Switch Transformer 原论文与解读：Top-1/Top-K 路由为什么能把参数规模和计算规模拆开。
- DOCSAID 等笔记：辅助损失公式、均衡目标、训练行为解释。
- 工程实践文章：capacity factor、全局计数、跨设备路由、router 精度等实现细节。
- ALF-LB 相关综述：理解为什么“均衡”不一定非要通过额外 loss 来做。
