## 核心结论

Medusa 的核心不是“让 attention 更便宜”，而是“让一次解码尽量确认更多 token”。传统自回归解码每轮只能生成 1 个 token，Medusa 则在基座模型最后一层外挂多个轻量解码头，让这些头并行预测未来多个位置的候选 token，再用树状验证机制一次性检查哪些候选前缀能被原模型接受。只要平均每轮能多接受几个 token，总解码轮数就会明显下降。

“解码头”可以先理解成额外加在模型输出层上的小预测器；它不替代原模型，只负责猜测未来位置的候选。“树状验证”可以先理解成把多头给出的候选路径组织成一棵树，再让原模型并行检查这棵树里哪些分支可信。这样做的收益来自减少 sequential dependency，也就是“后一个 token 必须等前一个 token 算完”的串行依赖。

从工程角度看，Medusa 适合这样一类场景：你已经有一个主模型，希望在不额外维护 draft model 的前提下提升单请求、低 batch 推理速度。它常见能带来约 $1.5\times$ 到 $2.5\times$ 的 wall-time 加速，在论文里的部分 7B/13B 实验中甚至更高；但前提是候选前缀的接受率足够高。如果接受率低，多头预测和验证树的额外开销会把收益吃掉。

下表先给一个直观印象。不同实现、硬件、采样设置会影响绝对值，但趋势是稳定的：Medusa-1 已经有效，Medusa-2 通常更快。

| 模型 | 方案 | 相对速度 | 直观含义 |
|---|---:|---:|---|
| Vicuna-7B | Baseline | 1.0x | 每轮只确认 1 个 token |
| Vicuna-7B | Medusa-1 | 约 2.2x | 多头预测 + 树状验证 |
| Vicuna-7B | Medusa-2 | 约 2.8x | 训练更充分，接受率更高 |
| Vicuna-13B | Baseline | 1.0x | 参数更大，逐 token 成本更高 |
| Vicuna-13B | Medusa-1/2 | 约 2x 级别 | 仍然能显著减少轮数 |

玩具例子先看最小版本。假设当前前缀是“北京是中国的”，原始解码会先算“首”，再算“都”。Medusa 会让一个头猜第 1 个未来 token 的候选，比如 `首/一`，另一个头猜第 2 个未来 token 的候选，比如 `都/线/城`。然后系统不是只试一条路，而是把 `首都`、`首线`、`首城`、`一都`、`一线`、`一城` 这些候选路径一起验证。若原模型认为“首都”这两个 token 都合理，就能一轮直接接受两个 token，而不是分两轮慢慢生成。

---

## 问题定义与边界

问题定义很明确：大语言模型推理慢，瓶颈主要在解码阶段的串行性，而不是训练时常说的理论 FLOPs。每生成一个新 token，都要重新经过整套网络，把当前上下文送入模型，再得到下一个 token 的分布。对于大模型，这意味着显存带宽和权重读取会在每一轮反复发生。

“显存带宽”可以先理解成 GPU 在单位时间里搬运参数和激活值的能力。对大模型解码而言，很多时候不是算术单元不够，而是每次都得把大量权重重新参与计算。于是问题变成：能不能让“一次完整前向”不只确认 1 个 token，而是确认一段前缀？

这里需要划清边界。

第一，Medusa不改变 Transformer 自注意力的渐近复杂度。它不是 FlashAttention 一类算子优化，也不是 KV Cache 压缩方案。它解决的是“轮数”，不是“单轮 attention 复杂度”。

第二，Medusa不等于传统 speculative decoding。后者通常需要两个模型：一个小 draft model 先草拟若干 token，大模型再验证。Medusa 则把“草拟未来 token”的能力内嵌到主模型里，通过多个轻量头完成，因此部署时通常只维护一个模型体系，系统复杂度更低。

第三，Medusa 的收益最明显的场景通常是 batch=1 或小 batch 的在线生成。原因很简单：当 batch 已经很大时，GPU 往往更容易被算满，单请求延迟不再主要由逐 token 串行决定，此时 Medusa 的收益未必像单路对话那样明显。

可以用一张表看自回归和 Medusa 的差异：

| 维度 | 传统自回归 | Medusa |
|---|---|---|
| 每轮确认 token 数 | 1 | 可能大于 1 |
| 轮数 | 与输出长度近似相同 | 与“接受的前缀块数”相关 |
| 是否需要额外小模型 | 通常不需要 | 不需要独立 draft model |
| 额外结构 | 无 | 多个轻量解码头 + 树状验证 |
| 收益来源 | 无 | 减少解码轮次 |
| 主要风险 | 纯串行慢 | 接受率低时额外开销反噬 |

真实工程里，一个直接的例子是本地单卡问答服务。假设你在一张 3090 上部署 7B 级模型，为了降低显存通常还会开 8bit 或 4bit 量化。此时单次生成 200 个 token，如果每轮只能吐 1 个 token，就要走 200 轮完整解码；如果 Medusa 把平均每轮接受长度提高到 2 到 3 个 token，那么总轮数可以下降到大约原来的 $1/2$ 到 $1/3$。这也是它常能带来接近倍数级加速的根本原因。

---

## 核心机制与推导

Medusa 的机制可以拆成三步：多头预测、候选成树、原模型验证。

先看多头预测。假设挂了 $K$ 个 Medusa head，第 $k$ 个 head 负责预测未来第 $k$ 个位置的 token 候选，并保留 top-$s_k$ 个候选。这里的 top-$k$ 可以先理解成“概率最高的前几个词元”。如果 head1 保留 2 个候选，head2 保留 3 个候选，那么未来两步的候选路径就不止 5 个，而是要按前缀展开成树。

候选总路径数可写成：

$$
N_{\text{cand}}=\sum_{k=1}^{K}\prod_{i=1}^{k}s_i
$$

这个式子的意思很直接。长度为 1 的前缀有 $s_1$ 条；长度为 2 的前缀有 $s_1 s_2$ 条；一直累加到长度为 $K$。如果 $K=2, s_1=2, s_2=3$，则总候选节点数是：

$$
2 + 2\times 3 = 8
$$

这里的“节点数”与“完整路径数”要区分。完整长度为 2 的路径有 6 条，但树里还包含长度为 1 的中间前缀节点，所以总验证节点数是 8。

玩具例子可以具体写出来。假设当前前缀后，head1 给出 `首/一`，head2 给出 `都/线/城`。则二层树如下：

| 层级 | 候选 |
|---|---|
| 第 1 层 | `首`，`一` |
| 第 2 层 | `首都`，`首线`，`首城`，`一都`，`一线`，`一城` |

原模型接下来不是按普通自回归一条条跑，而是借助树状 attention 一次性验证这些候选前缀。所谓“树状 attention”，可以先理解成一种让不同候选共享前缀计算、同时保持因果约束的注意力组织方式。它不是把每条路径完全独立地重新算一遍，否则成本会太高；它会尽量让共享部分复用。

验证时并不是只看某个 head 自己说“这个 token 很可能”，而是看原始模型在对应上下文下是否认可这个 token。论文里常用的是 typical acceptance，也就是“典型接受”。“熵”可以先理解成“分布有多不确定”；熵越高，说明模型对下一个 token 越拿不准。其接受准则可以写成：

$$
p_{\text{orig}}(x_{n+k}\mid x_{\le n+k-1})
>
\min\left(\epsilon,\ \delta\cdot e^{-H\left(p_{\text{orig}}(\cdot\mid x_{\le n+k-1})\right)}\right)
$$

其中 $H(p)$ 是熵：

$$
H(p) = -\sum_i p_i \log p_i
$$

这条规则表达的是：如果原模型本来就很确定，那么阈值会更严格；如果原模型本来就不确定，阈值会随熵做动态调整，不至于因为分布太平而把所有候选都拒掉。最终系统会从候选树里找到“最长的可接受前缀”，然后一次性把这段前缀加入输出，再进入下一轮。

这个“最长前缀”是收益关键。设输出总长度是 $T$，每轮平均接受长度是 $\bar{L}$，那么解码轮数大致从 $T$ 降到：

$$
\frac{T}{\bar{L}}
$$

如果 $\bar{L}=2.2$，理论上轮数就是原来的约 $45\%$。当然真实 wall-time 还要扣除多头计算与树验证开销，所以最终加速通常低于纯轮数缩减比例，但趋势一致。

---

## 代码实现

如果你只是想跑起来，最简单的方法不是自己改 Transformer 源码，而是直接用官方 CLI。下面是一个最小工程例子：在单卡上加载 Medusa 权重，开启量化，体验多头解码。

```bash
CUDA_VISIBLE_DEVICES=0 python -m medusa.inference.cli \
  --model FasterDecoding/medusa-vicuna-7b-v1.3 \
  --load-in-8bit
```

如果你已经有熟悉的底座模型，也可以显式指定 `--base-model`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m medusa.inference.cli \
  --model FasterDecoding/medusa-vicuna-7b-v1.3 \
  --base-model lmsys/vicuna-7b-v1.3 \
  --load-in-4bit
```

这里的 `--load-in-8bit/4bit` 是量化加载开关，作用是降低显存占用；`--base-model` 的作用是告诉系统用哪个底座权重做实际推理。对初学者来说，最重要的认知是：Medusa 不是一个独立模型家族，而是“基座模型 + 多头解码增强”。

下面给一个可运行的 Python 玩具程序，用来计算候选节点数、估算不同接受长度下的理论轮数收益。它不是官方实现，但能帮助你把公式和工程直觉对上。

```python
from math import prod, ceil

def medusa_candidate_nodes(topks):
    total = 0
    for k in range(1, len(topks) + 1):
        total += prod(topks[:k])
    return total

def decoding_rounds(output_tokens, accepted_prefix_avg):
    assert output_tokens > 0
    assert accepted_prefix_avg >= 1.0
    return ceil(output_tokens / accepted_prefix_avg)

# 玩具例子：2 个头，分别保留 top-2 和 top-3
topks = [2, 3]
nodes = medusa_candidate_nodes(topks)
assert nodes == 8  # 2 + 2*3

# 真实工程近似：输出 200 个 token
baseline_rounds = decoding_rounds(200, 1.0)
medusa_rounds = decoding_rounds(200, 2.2)

assert baseline_rounds == 200
assert medusa_rounds == 91
assert medusa_rounds < baseline_rounds

speedup_ideal = baseline_rounds / medusa_rounds
assert speedup_ideal > 2.0

print("candidate_nodes =", nodes)
print("baseline_rounds =", baseline_rounds)
print("medusa_rounds =", medusa_rounds)
print("ideal_round_speedup =", round(speedup_ideal, 2))
```

真实工程例子可以这样理解。你做一个 RAG 问答服务，输入很长，回答通常在 100 到 300 token。传统方式里，尾部生成阶段最慢，因为每个 token 都要完整过一遍模型。接入 Medusa 后，你不需要再维护一个小 draft model，也不必处理双模型一致性，只需要在原模型推理栈中支持多头输出和树状验证。这就是它在“单模型、低 batch、工程简化优先”的部署环境里很有吸引力的原因。

---

## 工程权衡与常见坑

Medusa 的第一大权衡是：候选树越大，不代表越快。很多初学者看到“多几个头、多保留几个 top-k”会直觉以为收益线性上涨，实际上候选规模是乘法增长，很容易把验证成本推高。

例如 4 个头、每个头保留 5 个候选时，候选节点数是：

$$
5 + 25 + 125 + 625 = 780
$$

这已经不是“小修小补”的额外开销，而是一棵很大的验证树。如果你的接受率只有 20%，等于大部分候选都白算了。于是看起来减少了轮数，实际上每轮成本暴涨，最终 wall-time 反而变差。

第二个坑是把“head 准确率”和“最终收益”混为一谈。头本身预测得不错，不代表最终一定快。真正决定收益的是原模型愿意接受多长的前缀，也就是 accepted prefix length。如果头常常能猜对第一个 token，但第二、第三个 token 总被拒绝，那么加速可能只比 baseline 好一点。

第三个坑是阈值调得过死或过松。$\epsilon$、$\delta$ 太保守，系统就会频繁只接受 1 个 token，退化回普通自回归；太激进，则可能放进不稳定候选，出现质量回退或分布偏移。典型接受之所以要引入熵，就是为了让阈值跟模型当前不确定性联动，而不是写死一个固定概率门槛。

下面这张表可以作为调参直觉：

| 配置 | 候选节点趋势 | 常见结果 |
|---|---|---|
| 头数少，top-k 小 | 低 | 额外成本可控，但加速上限有限 |
| 头数少，top-k 中等 | 中 | 往往是最稳妥区间 |
| 头数多，top-k 大 | 高 | 易出现树验证吞掉收益 |
| 阈值过严 | 低接受率 | 速度接近 baseline |
| 阈值过松 | 高接受率表面提升 | 可能影响生成稳定性 |

第四个坑是忽视场景差异。代码补全、结构化输出、数学推导，这几类任务的 token 分布差异很大。对于分布更尖锐、可预测性更强的场景，Medusa 通常更容易接受更长前缀；对于开放式发散生成，后续 token 的不确定性更高，接受率可能下降。

第五个坑是只看 token/s，不看端到端延迟。工程上用户关心的常常是 TTFT 和整体响应时间，而不是某个局部阶段的峰值吞吐。Medusa 主要改善的是持续生成阶段，不一定显著降低首 token 延迟。如果你的产品瓶颈在检索、网络、模板拼接或首 token 计算，Medusa 的体感提升会被摊薄。

---

## 替代方案与适用边界

最常见的替代路线是传统 speculative decoding。它的思路是用一个小 draft model 先生成若干 token，再由大模型统一验证。优点是上限可能更高，尤其当小模型足够便宜、接受率又够高时；缺点是系统复杂度明显增加，你要维护两套模型、两套缓存逻辑，还要处理它们之间的分布对齐。

Hydra、Eagle 一类后续方法也是在“减少解码轮数”这个方向上继续优化。它们在某些场景下可以跑出比 Medusa 更高的速度，但代价通常是更复杂的训练方式、推理流水线或额外模型结构。对追求极致吞吐的多卡集群，这可能值得；对单卡、本地部署、想尽量少改系统的人，Medusa 往往更均衡。

可以这样理解它们的适用边界：

| 方案 | 额外模型 | 部署复杂度 | 常见收益 | 适合场景 |
|---|---|---:|---:|---|
| Baseline 自回归 | 否 | 低 | 1.0x | 简单、稳定、无额外工程 |
| Medusa-1 | 否 | 中 | 约 1.5x-2.2x | 单模型部署，优先简化系统 |
| Medusa-2 | 否 | 中 | 约 2x-2.8x | 接受率更高，追求更强加速 |
| 传统 Speculative Decoding | 是 | 高 | 可能较高 | 有能力维护双模型体系 |
| Hydra / Eagle | 常常需要额外机制 | 高 | 某些场景更高 | 多卡、专用推理平台、吞吐优先 |

所以结论不是“Medusa 一定最好”，而是“Medusa 在单模型部署成本和可观加速之间给出了一个很好的工程平衡点”。如果你的目标是本地单卡聊天、在线低 batch 服务、希望在现有主模型上快速加速，它很合适；如果你的目标是多卡大规模服务、极限吞吐、可以接受更复杂的训练和调度，那么可以进一步评估 Hydra、Eagle 或双模型 speculative decoding。

---

## 参考资料

1. Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, ICML 2024. [https://arxiv.org/pdf/2401.10774](https://arxiv.org/pdf/2401.10774)
2. OpenReview 页面与实验摘要。 [https://openreview.net/pdf?id=PEpbUobfJv](https://openreview.net/pdf?id=PEpbUobfJv)
3. FasterDecoding/Medusa 官方仓库与 CLI 示例。 [https://github.com/FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa)
4. Hugging Face 论文索引页。 [https://huggingface.co/papers/2401.10774](https://huggingface.co/papers/2401.10774)
5. 关于 Medusa、Hydra、Eagle 等后续路线的对比讨论。 [https://aclanthology.org/2025.emnlp-main.986.pdf](https://aclanthology.org/2025.emnlp-main.986.pdf)
