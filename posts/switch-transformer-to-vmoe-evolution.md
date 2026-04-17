## 核心结论

Switch Transformer 和 V-MoE 都属于 MoE，中文常译为“专家混合模型”，白话讲就是“同一层里放很多个前馈网络，但每个 token 只激活其中少数几个”。

这条演进线的主问题不是“怎样把参数堆得更大”，而是“怎样在通信、计算、负载平衡之间做更便宜的稀疏激活”。

GShard 在 2020 年把大规模 MoE 做到了可训练，典型做法是 Top-2 路由，即每个 token 交给两个专家。Switch Transformer 在 2021 年把它进一步简化成 Top-1，只保留最可能的那个专家，直接减少一次专家计算和一部分跨设备通信。V-MoE 在 2021 年 NeurIPS 论文、2022 年 1 月 Google Research 博客中把 MoE 放进 ViT，并引入 Batch Prioritized Routing，简称 BPR，白话讲就是“先按 token 重要性排队，再把有限专家容量优先留给重要 token”。

最关键的准确表述是：V-MoE 的创新不等于“把 Switch 的 Top-1 换成优先路由”。BPR 主要解决的是容量受限时“哪些 token 应该先被处理”，它和 token 到专家的 Top-k 选择是两个层面。原始 V-MoE 常见设定仍然是 $K=2$，只是当容量不足时，不再随机或按原顺序丢 token，而是按重要性优先保留。

| 模型 | token 到专家 | 容量不足时怎么处理 | 计算占用 | 主要难点 |
|------|--------------|-------------------|----------|----------|
| GShard | Top-2 | 超出容量则丢弃或回退 | 2 个专家 | 通信和实现复杂 |
| Switch Transformer | Top-1 | 超出容量则丢弃 | 1 个专家 | 路由过硬，易失衡 |
| V-MoE | 常见为 Top-2，再叠加 BPR | 先保留高优先级 token | capacity-aware | 排序、缓冲和 drop 策略 |

玩具例子可以这样记：Switch 像“每个学生只允许报一门最想上的课”；V-MoE 像“先看每个学生对这门课的意愿强度，再把座位优先给最需要的人”。

---

## 问题定义与边界

MoE 的核心收益来自“参数总量可以很大，但单个 token 实际只走少量路径”，因此单次前向的 FLOPs 不必按总参数线性增长。

设一层里有 $E$ 个专家，token 表示为 $x_t$，路由器输出为 $g(x_t)\in\mathbb{R}^E$。经过 softmax 后得到每个专家的概率。Switch 的 Top-1 路由是：

$$
e_t=\arg\max_i g(x_t)_i
$$

白话讲，就是“每个 token 只认一个最有把握的专家”。

如果一个 batch 有 $T$ 个 token，每个专家容量上限记为 $C$，常见近似写法是：

$$
C \approx \frac{T}{E}\cdot \text{capacity\_factor}
$$

这里的 capacity factor 就是“容量系数”，白话讲是“平均每个专家该分到多少 token，再乘一个保险倍数”。

V-MoE 的 BPR 先给每个 token 一个优先级分数，例如：

$$
s_t=\max_i g(x_t)_i
$$

或基于所选专家权重的其他分数。然后把整个 batch 的 token 按 $s_t$ 从大到小排序，优先把高分 token 填入专家缓冲区。超过容量的 token 会在该层被 drop，也就是“这一层的专家 FFN 不处理它”。

新手容易混淆的边界有两个：

1. Top-k 决定“一个 token 想去哪些专家”。
2. BPR 决定“容量不够时，哪些 token 值得先处理”。

看电影排队的例子最直观。Switch 是每个人只选一个最想去的厅，然后直接去排队；V-MoE 是先按“想看程度”给所有人排序，再根据每个厅的座位数决定谁能进去。后者多了一层“何时放弃”的显式机制。

---

## 核心机制与推导

先看 Switch。它把 GShard 的双专家路径压成单路径，前向过程可以写成：

1. 计算 router logits：$z_t=W_r x_t$
2. 计算概率：$p_t=\text{softmax}(z_t)$
3. 选择专家：$e_t=\arg\max_i p_{t,i}$
4. 只把 $x_t$ 发给专家 $e_t$
5. 输出乘对应门控权重后回写

这样做的好处是计算和通信都下降，因为每个 token 只需一次专家前向。代价是路由变硬，logits 的微小波动也可能让 token 从专家 A 突然跳到专家 B。原始 Switch 的核心稳定手段是负载均衡辅助损失和对路由计算使用更高精度；工程实践里又常结合 router z-loss 一类手段抑制 logits 过大，减少 softmax 过尖。

常见负载均衡项写成：

$$
L_{\text{aux}}=\alpha E \sum_{i=1}^{E} f_i P_i
$$

其中 $f_i$ 是“实际被分到专家 $i$ 的 token 比例”，$P_i$ 是“路由器分给专家 $i$ 的平均概率质量”。白话讲，这个损失鼓励“说要去”和“真的去了”都更均匀，防止某个专家爆满。

再看 V-MoE。它通常把 ViT 中若干 FFN 层替换成 MoE 层。ViT 里的 token 是图像 patch，白话讲就是“把一张图切成很多小块后得到的一串向量”。视觉任务里 patch token 数量多，而且重要性差异明显，比如目标边缘、主体区域往往比大片背景更关键，因此“谁先占用容量”会直接影响效果。

BPR 的过程可以分成三步：

1. 先做普通路由，得到每个 token 倾向的专家。
2. 计算每个 token 的优先级分数 $s_t$。
3. 在整个 batch 上按 $s_t$ 排序，依次尝试写入目标专家的 buffer；如果该专家已满，则该 token 在该层被跳过。

形式化一点，若 token $t$ 想去专家 $e_t$，则它被保留的条件可写成：

$$
\text{rank}_{e_t}(t) \le C
$$

这里 $\text{rank}_{e_t}(t)$ 表示“在所有同样想去专家 $e_t$ 的 token 中，按优先级排序后的名次”。若名次超过 $C$，则 drop。

一个最小数值例子：假设一层处理 $T=784$ 个 patch token，专家数 $E=32$，capacity factor 取 $0.5$，则每个专家大约只有

$$
C=\frac{784}{32}\times 0.5 \approx 12
$$

个槽位。总槽位大约是 $32\times 12=384$。这意味着最多只有约一半 token 会在该层专家 FFN 中被处理。BPR 的目标不是“所有 token 都公平处理”，而是“把算力留给更关键的 token”。

真实工程例子是 Google Research 公开的 V-MoE-H/14。论文与官方博客给出的结果是：15B 参数规模下，ImageNet 微调可达 90.35%，并且在不少设置下可用大约一半于同等级 dense ViT 的推理计算得到接近或更好的表现。这里的关键不是参数更大本身，而是稀疏路由把计算集中到了更有信息密度的 patch 上。

---

## 代码实现

下面的代码不是完整训练版，而是一个可以直接运行的玩具实现，用来对比 Switch 的 Top-1 和 V-MoE 风格的“按优先级填容量”。

```python
from math import exp

def softmax(xs):
    m = max(xs)
    exps = [exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def switch_route(router_logits):
    probs = [softmax(x) for x in router_logits]
    expert_ids = [max(range(len(p)), key=lambda i: p[i]) for p in probs]
    weights = [p[e] for p, e in zip(probs, expert_ids)]
    return expert_ids, weights

def bpr_route(router_logits, capacity):
    probs = [softmax(x) for x in router_logits]
    token_choice = [max(range(len(p)), key=lambda i: p[i]) for p in probs]
    priority = [max(p) for p in probs]

    order = sorted(range(len(probs)), key=lambda t: priority[t], reverse=True)
    used = [0] * len(probs[0])
    kept = [False] * len(probs)

    for t in order:
        e = token_choice[t]
        if used[e] < capacity:
            used[e] += 1
            kept[t] = True

    return token_choice, priority, kept, used

# 5 个 token，2 个专家
router_logits = [
    [3.2, 0.1],  # token0 强烈偏向 expert0
    [2.8, 0.2],  # token1 强烈偏向 expert0
    [2.1, 0.3],  # token2 偏向 expert0
    [0.4, 2.5],  # token3 偏向 expert1
    [0.2, 1.4],  # token4 偏向 expert1
]

switch_experts, switch_weights = switch_route(router_logits)
assert switch_experts == [0, 0, 0, 1, 1]
assert len(switch_weights) == 5

choice, priority, kept, used = bpr_route(router_logits, capacity=2)

# expert0 只有 2 个槽位，因此去 expert0 的 3 个 token 中会丢 1 个优先级最低的
assert choice == [0, 0, 0, 1, 1]
assert sum(kept) == 4
assert used == [2, 2]

print("switch experts:", switch_experts)
print("priority:", [round(x, 4) for x in priority])
print("kept mask:", kept)
print("used slots:", used)
```

这段代码对应的流程就是：

- Switch：`softmax -> argmax -> 单专家派发`
- V-MoE/BPR：`softmax -> 选目标专家 -> 算 priority -> 全 batch 排序 -> 按 capacity 填槽 -> 生成 drop mask`

如果写成伪代码，更接近工程实现：

```python
scores = router(x)                  # [T, E]
probs = softmax(scores, dim=-1)
expert = probs.argmax(dim=-1)       # Switch / token-choice
priority = probs.max(dim=-1).values # BPR score
order = priority.argsort(descending=True)

for t in order:
    e = expert[t]
    if slots[e] < C:
        dispatch(t, e)
        slots[e] += 1
    else:
        drop_mask[t] = True
```

真实训练里还要补三件事：

1. dispatcher buffer，把不同 token 打包发往不同专家。
2. auxiliary loss，避免专家冷热不均。
3. 反向传播细节，确保 gate、mask、combine weight 的梯度路径稳定。

---

## 工程权衡与常见坑

Switch 的最大优势是简单，但“简单”本身会带来硬边界。每个 token 只有一个专家，一旦 router logits 很尖，路由会非常敏感。表现上常见为某几个专家过热，别的专家长期吃不到足够 token，最终出现训练震荡或专家退化。

V-MoE/BPR 的主要坑则是容量设得太小。因为优先级分数在训练早期并不可靠，如果这时就大量 drop 低分 token，模型可能把“暂时学得不好”误当成“永远不重要”。在视觉任务里，这会导致背景、小目标或边角区域长期得不到专家处理。

| 优化点 | 触发条件 | 作用 |
|------|----------|------|
| 负载均衡 loss | 某专家 token 占比长期过高 | 拉平专家使用率 |
| router 高精度计算 | 低精度训练时路由不稳 | 减少数值误差 |
| z-loss 或类似 logit 正则 | softmax 过尖、频繁跳专家 | 压制 logit 爆炸 |
| 调大 capacity factor | token drop 过多 | 提高保留率，换更多算力 |
| 逐层学习 priority | 视觉 token 重要性差异大 | 让算力更集中于关键信息 |

面向初学者，一个很实用的判断标准是：

- 如果你看到“专家利用率极不均匀”，先查 router 分布和辅助损失。
- 如果你看到“精度突然掉很多且 token drop 很高”，先查 capacity。
- 如果你看到“训练能跑但收益不明显”，先查通信开销是否吃掉了稀疏激活的理论收益。

---

## 替代方案与适用边界

不是所有场景都该上 Switch 或 V-MoE。

GShard 的 Top-2 更像“给每个 token 留一个备胎专家”，鲁棒性往往更好，但通信和实现更重。Switch 适合“先把 MoE 跑起来并压低成本”的阶段。V-MoE 适合“token 数多、重要性差异大、能接受排序和 buffer 管理开销”的视觉场景。

Dense ViT 仍有明确边界优势。若 token 数很少，例如图像分辨率不高、patch 数不到 200，或者你的推理延迟要求极严，BPR 的排序、打包、drop 控制本身就可能抵消收益。此时 dense ViT 往往更直接。

| 方案 | 更适合的 token 规模 | 精度/计算权衡 | 实现复杂度 | 适用边界 |
|------|--------------------|---------------|------------|----------|
| Dense ViT | 小到中等 | 稳定但无法按需跳算 | 低 | 低延迟、小模型 |
| GShard Top-2 | 中到大 | 精度更稳，计算更高 | 高 | 通信资源充足 |
| Switch Transformer | 中到超大 | 成本低，路由更硬 | 中 | 语言类大规模预训练 |
| V-MoE + BPR | 大量视觉 token | 能按 token 重要性省算力 | 高 | 高分辨率视觉任务 |

一个具体判断：如果是小型数据集、低分辨率分类、batch 很小，优先考虑 dense ViT；如果是高分辨率图像、patch 很多、推理计算预算紧，但又想保留大模型容量，V-MoE 的收益会更明显。

---

## 参考资料

1. Switch Transformer 论文，Fedus, Zoph, Shazeer，JMLR 2022: https://jmlr.org/papers/v23/21-0998.html
2. Switch Transformer arXiv 版本，2021-01-11: https://arxiv.org/abs/2101.03961
3. GShard 论文，Lepikhin 等，2020: https://arxiv.org/abs/2006.16668
4. V-MoE 论文页，Google Research，NeurIPS 2021: https://research.google/pubs/scaling-vision-with-sparse-mixture-of-experts/
5. V-MoE 官方博客，Google Research，2022-01-13: https://research.google/blog/scaling-vision-with-sparse-mixture-of-experts/
6. V-MoE NeurIPS 论文 PDF: https://proceedings.neurips.cc/paper_files/paper/2021/file/48237d9f2dea8c74c2a72126cf63d933-Paper.pdf
7. IBM 对 MoE 的综述性介绍，便于补背景: https://www.ibm.com/think/topics/mixture-of-experts
8. 作为后续演进参照，可继续读 Soft MoE，Google DeepMind，2024: https://deepmind.google/research/publications/from-sparse-to-soft-mixture-of-experts/
