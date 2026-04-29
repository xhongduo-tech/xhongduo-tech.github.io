## 核心结论

注意力头剪枝与重分配，讨论的是一个很具体的问题：多头注意力里，每个 head 是否都值得保留。这里的 head 可以先理解成“并行处理同一段输入的一组小计算通道”。

结论先给出：

1. 多头注意力不是 head 越多越好。很多 head 的功能重复，真正稳定贡献效果的，往往只是少数几个专门化 head。
2. 剪枝不是把某些 head 临时置零，而是把低贡献 head 从 `Q/K/V/O` 参数结构里真正删掉，这样才可能减少参数、显存和推理开销。
3. 重分配比“盲目删头”更重要。省下来的预算可以转成更少的 KV 组、更适合解码的 `GQA/MQA` 结构，或者通过少量 uptraining 恢复精度。
4. 最安全的工程路径通常不是“每层统一裁掉 25%”，而是先在验证集上做单头消融，计算  
   $$
   s_i = M(\text{full}) - M(\text{prune } i)
   $$
   其中 $M$ 是验证指标。$s_i$ 越小，说明删掉第 $i$ 个 head 影响越小，越适合优先剪掉。

一个直接的玩具例子：8 个 head 的模型里，如果验证集实验显示其中 2 个 head 被删后指标几乎不变，那么先剪这 2 个 head，通常比“所有层平均裁掉 25%”更稳。因为真实模型里，不同层、不同头的重要度差异很大。

“剪枝”和“重分配”不是一回事：

| 方案 | 目标 | 是否改结构 | 主要收益 | 主要风险 |
| --- | --- | --- | --- | --- |
| head 剪枝 | 删除低贡献 head | 是 | 降参数、降部分计算 | 误删关键 head 导致掉点 |
| head mask | 前向时置零 | 否 | 便于实验 | 通常没有真实延迟收益 |
| 资源重分配 | 把预算迁移到更高效结构 | 是 | 更明显降 KV cache 和带宽 | 需要重构和再训练 |
| uptraining | 少量继续训练修复新结构 | 否/配合结构改动 | 恢复精度 | 额外训练成本 |

---

## 问题定义与边界

先把几个术语说清楚。

| 术语 | 定义 | 白话解释 | 本文关注点 |
| --- | --- | --- | --- |
| 剪枝 | 把低贡献 head 对应的参数从模型里删除 | 撤掉长期空转的工位 | 是 |
| mask | 运行时把某些 head 输出乘 0 | 工位还在，只是不让它干活 | 只作为评估手段 |
| 重分配 | 把省下的 head 预算转成别的注意力结构 | 工位数量变少，但布局更高效 | 是 |
| uptraining | 从已有 checkpoint 出发做少量继续训练 | 小修正，不是从零重训 | 是 |

可以把多头注意力理解成“多个工位同时处理同一批输入”。剪枝，就是撤掉长期贡献很小的工位。重分配，则是把这些工位改造成更少但更高效的配置，比如多个 query head 共享一组 `K/V`。

本文边界很明确：

1. 重点讨论 Transformer 的多头注意力，尤其是 decoder self-attention。
2. 重点是推理阶段优化，因为实际部署里，KV cache、显存带宽和单 token 延迟通常比训练时的理论 FLOPs 更敏感。
3. 不讨论所有压缩方法，只讨论“head 是否值得保留”以及“删完后预算如何重新组织”。

这里要特别强调一个边界：如果你只是为了分析模型行为，那么 mask 就够了；但如果你想降低线上成本，必须做结构删除或结构重组。因为 GPU kernel、矩阵形状、KV cache 布局都依赖真实张量维度，不会因为“逻辑上不用这个 head”而自动变快。

---

## 核心机制与推导

标准多头注意力可以写成下面三步。设输入为 $X \in \mathbb{R}^{L \times d_{model}}$，序列长度为 $L$，总头数为 $H$，单头维度为 $d_h = d_{model}/H$。

第 $i$ 个 head 的投影是：

$$
Q_i = XW_Q^{(i)}, \quad K_i = XW_K^{(i)}, \quad V_i = XW_V^{(i)}
$$

这里的投影矩阵可以理解成“每个工位看输入时用的专属视角”。

单个 head 的注意力输出是：

$$
A_i = \operatorname{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_h}}\right)V_i
$$

最后把所有 head 拼接，再过输出投影：

$$
Y = \operatorname{Concat}(A_1, \ldots, A_H)W_O
$$

为什么 head 可以被剪？因为这些并行通道并不一定都学到独特功能。实际训练后，常见情况是：

1. 有些 head 专门做局部对齐、位置偏置或句法关系。
2. 有些 head 与别的 head 功能高度重复。
3. 有些 head 的输出长期接近冗余噪声。

所以最直接的重要度定义就是单头消融。设完整模型指标为 $M(\text{full})$，删掉第 $i$ 个 head 后的指标为 $M(\text{prune } i)$，那么：

$$
s_i = M(\text{full}) - M(\text{prune } i)
$$

如果 $s_i$ 很小，说明删掉这个 head 几乎不影响效果；如果 $s_i$ 很大，说明它很关键。

### 玩具例子：4 个 head 的 KV cache 代价

设 `d_model = 8, H = 4, d_h = 2`。对 decoder self-attention，每生成 1 个 token，都要把当前 token 的 `K` 和 `V` 存进 KV cache。

每个 head 的 `K` 和 `V` 各有 `d_h = 2` 个标量，所以每 token 每层缓存量是：

$$
\text{KV per token} = H \times d_h \times 2
$$

代入数值：

$$
4 \times 2 \times 2 = 16
$$

如果剪掉 1 个低重要度 head，变成 3 个 head：

$$
3 \times 2 \times 2 = 12
$$

节省比例：

$$
\frac{16 - 12}{16} = 25\%
$$

如果不是直接剪，而是改成 `G = 2` 的 GQA，意思是 `H_q = 4` 个 query head 共享 `G = 2` 组 `K/V`，那么每 token 的 KV cache 近似变成：

$$
\text{KV per token in GQA} = G \times d_h \times 2 = 2 \times 2 \times 2 = 8
$$

相对原始 MHA 节省：

$$
\frac{16 - 8}{16} = 50\%
$$

一般地，若 query head 数为 $H_q$，KV group 数为 $G$，并保持每组 KV 维度不变，则 KV cache 规模大致按

$$
\frac{G}{H_q}
$$

缩小。这个比例解释了为什么在长上下文解码里，GQA/MQA 往往比“只删少量 head”更直接有效。

### 真实工程例子：长上下文生成服务

假设你维护一个 7B 级别的对话模型，主要成本来自长上下文自回归生成。线上 profile 发现：

1. 预填充阶段不是主要瓶颈。
2. 解码阶段受 KV cache 占用和显存带宽限制明显。
3. 注意力层里并不是所有 head 都有稳定贡献。

这时只做传统 head 剪枝，收益可能有限；更常见的工程路径是：

1. 先做 head ablation，识别低贡献 head。
2. 再把 decoder self-attention 从标准 MHA 改成 GQA。
3. 用少量 uptraining 把旧 checkpoint 迁移到新结构。
4. 最后在目标硬件上重新测吞吐、显存和质量。

这里的核心不是“理论上删了多少 FLOPs”，而是“是否真正减少了 KV cache 写入、显存读取和 kernel 调度压力”。

---

## 代码实现

工程上通常分三步：测重要度、选头、改结构并微调。只停留在“打分”阶段，没有部署价值。

先给一个最小可运行的玩具实现，用来说明如何计算单头消融分数。这里不用深度学习框架，直接模拟“删掉某个 head 后指标变化”。

```python
from dataclasses import dataclass

@dataclass
class Head:
    name: str
    contribution: float  # 对验证指标的近似贡献

def evaluate(heads):
    # 玩具指标：基础分 + 所有保留 head 贡献之和
    return 0.80 + sum(h.contribution for h in heads)

def ablation_scores(heads):
    full = evaluate(heads)
    scores = {}
    for i, h in enumerate(heads):
        pruned = heads[:i] + heads[i+1:]
        scores[h.name] = full - evaluate(pruned)
    return full, scores

heads = [
    Head("h0", 0.030),
    Head("h1", 0.002),
    Head("h2", 0.015),
    Head("h3", 0.001),
]

full_score, scores = ablation_scores(heads)

assert round(full_score, 3) == 0.848
assert round(scores["h1"], 3) == 0.002
assert round(scores["h3"], 3) == 0.001

# 低分数 head 优先剪
ranked = sorted(scores.items(), key=lambda x: x[1])
assert ranked[0][0] == "h3"
assert ranked[1][0] == "h1"

print(full_score)
print(ranked)
```

这个例子展示的是方法，不是训练真实模型。真实流程通常如下：

```python
# 伪代码
scores = {}
full_metric = evaluate(model, val_loader)

for layer in model.layers:
    for head_id in range(layer.num_heads):
        mask_head_temporarily(layer, head_id)   # 只用于评估
        metric = evaluate(model, val_loader)
        unmask_head(layer, head_id)
        scores[(layer.id, head_id)] = full_metric - metric

selected = select_heads_by_threshold(scores, threshold=0.001)
```

选完后，要做“真正删除”，不是继续保留原形状。

如果你用 Hugging Face 模型，很多架构已经提供 `prune_heads` 接口，最小流程通常类似这样：

```python
# 伪代码，示意接口用法
heads_to_prune = {
    0: [1, 3],   # 第0层剪第1和第3个head
    4: [7],      # 第4层剪第7个head
}
model.prune_heads(heads_to_prune)

# 少量数据做 uptraining
for batch in train_loader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

如果是自定义实现，本质上要同步修改 `W_Q/W_K/W_V/W_O` 的维度。假设某层原来有 `H=8` 个 head，每个 head 维度为 `d_h=64`，你删除其中 2 个 head，那么：

1. `W_Q/W_K/W_V` 中对应这 2 个 head 的输出通道要删掉。
2. `W_O` 的输入通道也要删掉，因为 `Concat(A_1,...,A_H)` 的总宽度变了。
3. 后续张量 reshape 逻辑要从 `8 x 64` 改成 `6 x 64`。

示意代码如下：

```python
# 伪代码：保留 keep_heads
keep_heads = [0, 2, 3, 5, 6, 7]

def slice_qkv_weight(W, d_h, keep_heads):
    # W shape: [d_model, H * d_h] 或 [H * d_h, d_model]，按你的实现而定
    cols = []
    for h in keep_heads:
        start = h * d_h
        end = (h + 1) * d_h
        cols.extend(range(start, end))
    return W[:, cols]

def slice_output_weight(Wo, d_h, keep_heads):
    cols = []
    for h in keep_heads:
        start = h * d_h
        end = (h + 1) * d_h
        cols.extend(range(start, end))
    return Wo[cols, :]
```

如果目标不是保留少量 MHA head，而是迁移到 GQA，那么结构改动会更大。此时不是简单删 query head，而是减少 `K/V` 组数，让多个 query head 共享同一组 `K/V`。这类重分配通常需要 uptraining，因为模型的参数语义已经变了。

---

## 工程权衡与常见坑

最常见的误区，是把“分析工具”和“部署优化”混为一谈。

下面这张表是最容易踩的坑：

| 坑点 | 为什么错 | 结果 |
| --- | --- | --- |
| 只看 attention heatmap | 热图好看不等于重要 | 容易错保留、错删除 |
| 只做 mask 不做结构删除 | 张量维度没变 | 实际延迟常常不变 |
| 不微调直接剪 | 结构突变后分布漂移 | 精度波动大 |
| 层间统一剪枝率 | 不同层敏感度不同 | 关键层容易被剪坏 |
| 不做硬件 profile | 理论 FLOPs 不等于吞吐 | 优化方向可能完全错 |

这里有一个工程判断准则：训练 FLOPs、参数量、激活显存、KV cache、真实吞吐，不是同一个指标。

1. 参数量下降，未必能降低解码延迟。
2. 理论 FLOPs 下降，未必能提升 tokens/s。
3. KV cache 下降，通常更直接影响长上下文解码成本。
4. 真正的线上收益，必须在目标硬件、目标 batch size、目标上下文长度下测。

举一个典型工程例子：你在 A100 上把某模型 attention 相关理论 FLOPs 降低了 20%，但线上吞吐几乎不变。原因可能是：

1. 瓶颈实际在 FFN，不在 attention。
2. batch 太小，kernel 启动和调度开销主导。
3. 你只是 mask 了 head，内核形状没变。
4. 量化、paged KV cache、张量并行策略比 head 数更影响性能。

所以真正可靠的流程应该是：

1. 先 profile，确认瓶颈到底是不是 attention。
2. 再做分层、分头的重要度测量。
3. 真正改结构，而不是只改前向逻辑。
4. 做少量 uptraining。
5. 在目标部署环境复测质量和吞吐。

---

## 替代方案与适用边界

如果目标是降低推理成本，head 剪枝只是方案之一，而且不一定是最优先的。

| 方案 | 适合目标 | 主要收益 | 主要代价 | 适用边界 |
| --- | --- | --- | --- | --- |
| head 剪枝 | 去掉冗余头 | 降部分参数和计算 | 需要逐头评估 | 已知 head 冗余明显 |
| GQA/MQA | 降 KV cache、提解码效率 | 长上下文收益明显 | 要改结构并迁移 | 解码瓶颈明显 |
| 蒸馏 | 保精度压缩模型 | 端到端效果稳 | 训练成本较高 | 可接受教师学生流程 |
| 量化 | 降显存和带宽 | 部署收益直接 | 需处理精度回退 | 硬件支持好时优先 |
| 不改结构只调参 | 快速试验 | 风险低 | 收益有限 | 先验证方向 |

适用边界可以按场景区分。

训练阶段优化：
如果你关注训练成本，head 剪枝可以减少后续训练负担，但通常要配合重新训练或蒸馏，否则收益有限。

推理阶段优化：
如果你关注线上成本，必须优先看结构改动是否真正影响 kernel 形状和 KV cache 布局。否则“剪了”可能只是数学意义上的剪。

长上下文场景：
如果主要瓶颈是 KV cache 占用和显存带宽，GQA/MQA 往往比单纯删除少量 head 更直接。因为它直接减少 `K/V` 存储和读取。

低延迟场景：
如果目标是单请求低延迟，除了 attention 结构，还要同时看 batch 策略、KV cache 管理、量化、算子融合。只盯着 head 数，通常不够。

还有一种情况不适合强行剪枝：模型本来就很小，或者任务对质量极敏感，比如高精度代码生成、法律问答、医疗文本等。在这种场景，剪掉少量 head 带来的成本收益，可能小于风险。此时蒸馏、量化、推理缓存优化常常更稳。

可以把选择逻辑概括成一句话：如果你确认冗余 head 很多，先做剪枝；如果你确认瓶颈在长上下文解码，优先考虑 GQA/MQA；如果你更在乎质量稳定，优先考虑蒸馏或温和量化。

---

## 参考资料

1. [Michel et al., Are Sixteen Heads Really Better than One?](https://papers.nips.cc/paper/2019/file/2c601ad9d2ff9bc8b282670cdd54f69f-Paper.pdf)
2. [Voita et al., Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://arxiv.org/pdf/1905.09418.pdf)
3. [Shazeer, Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/pdf/1911.02150.pdf)
4. [Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245.pdf)
5. [PyTorch `nn.MultiheadAttention` 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html)
6. [Hugging Face `prune_heads` 实现](https://huggingface.co/transformers/v4.12.5/_modules/transformers/modeling_utils.html)
