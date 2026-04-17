## 核心结论

多任务微调不是“把多个数据集拼起来一起训”这么简单。真正决定效果的，通常是三件事一起工作：

1. 任务混合比例要受控。大任务样本多，如果直接按原始数据量采样，会长期主导参数更新，低资源任务几乎学不到。
2. 损失权重要可调。即使采样比例合适，不同任务的 loss 尺度也可能不同，仍然会让某些任务在反向传播时声音更大。
3. 梯度冲突要监控。梯度就是“参数该往哪里走”的方向向量；如果两个任务的梯度方向相反，就会出现负迁移，也就是一个任务帮忙，另一个任务拆台。

一个常用的起点是：先做限额加温度采样，再监控梯度余弦相似度，必要时引入 PCGrad 或 GradNorm。Flan 系列进一步说明，除了“训哪些任务”，还要管“同一任务以什么 prompt 形式出现”，把零样本、少样本、CoT 模板混合进去，模型对指令风格的泛化会更稳。

玩具例子：任务 A 有 1,000,000 条样本，任务 B 有 50,000 条。若设置采样上限 $K=65536$，则先截断任务规模：

$$
r_m=\frac{\min(e_m,K)}{\sum_n \min(e_n,K)}
$$

此时

$$
r_A=\frac{65536}{65536+50000}\approx0.567,\quad r_B\approx0.433
$$

再做温度平滑：

$$
r'_m=\frac{r_m^{1/T}}{\sum_n r_n^{1/T}}
$$

当 $T=2$ 时，得到 $r'_A\approx0.534,\ r'_B\approx0.466$。结论很直观：高资源任务比例下降，低资源任务被抬升，但又没有被强行改成 50:50。

| 方案 | A 采样占比 | B 采样占比 | 结果直观解释 |
|---|---:|---:|---|
| 原始按数据量 | 95.24% | 4.76% | B 基本被淹没 |
| 限额 $K=65536$ | 56.7% | 43.3% | 先抑制极端资源差 |
| 限额 + 温度 $T=2$ | 53.4% | 46.6% | 再进一步拉平 |

---

## 问题定义与边界

多任务微调的目标，是让一个共享模型在多个异质任务上找到兼容解。这里的“异质”，指任务可能在输入格式、输出长度、难度、loss 尺度、标签空间上都不同。

要先划清边界。本文讨论的是共享主干参数的多任务 SFT 或 instruction tuning，不讨论 MoE 路由、任务专属 Adapter、大规模 RLHF 混训等更重的结构改造。

可控变量主要有四类：

| 变量 | 含义 | 解决的问题 |
|---|---|---|
| $K$ | 采样上限 | 防止超大任务无限放大存在感 |
| $T$ | 温度 | 控制任务分布被拉平的强度 |
| $w_i$ | 第 $i$ 个任务 loss 权重 | 修正不同任务 loss/梯度尺度 |
| $\cos(g_i,g_j)$ | 任务梯度余弦 | 诊断任务是否互相冲突 |

对初学者，一个可记住的白话版本是：每个任务像一个杯子，数据量是杯子的容量。$K$ 表示“再大的杯子也先只倒这么多”，$T$ 表示“是否再把各杯水位拉近一些”。如果不做这两步，大杯子会一直抢走水流。

这里还有两个常见误区：

1. “均匀采样最公平。”不一定。低资源任务会更频繁重复同一批样本，容易过拟合。
2. “采样调好了就够了。”不一定。即使 batch 里各任务出现频率合理，梯度方向仍可能互相抵消。

---

## 核心机制与推导

整个训练流程可以抽象成：

采样 -> 前向计算各任务 loss -> 按权重聚合 -> 检测梯度冲突 -> 做梯度外科手术或动态调权 -> 参数更新

### 1. 任务混合比例

设第 $m$ 个任务样本量为 $e_m$。先用限额截断：

$$
r_m=\frac{\min(e_m,K)}{\sum_n \min(e_n,K)}
$$

这一步只解决“超大任务是否碾压别人”。然后再用温度变换：

$$
r'_m=\frac{r_m^{1/T}}{\sum_n r_n^{1/T}}
$$

当 $T>1$，分布更平；当 $T=1$，不变；当 $T<1$，强者更强。工程上常先固定 $K$，再把 $T$ 从 1 到 2 之间扫描。

### 2. 梯度冲突诊断

梯度余弦相似度衡量两个任务更新方向是否一致：

$$
\cos(g_i,g_j)=\frac{g_i\cdot g_j}{\|g_i\|\|g_j\|}
$$

这里的余弦相似度，就是“两个箭头夹角有多小”。若结果为正，方向大致一致；接近 0，说明基本无关；小于 0，说明存在冲突。

玩具例子：若 $g_A=[2,1]$，$g_B=[-1,0]$，则点积为 $-2$，余弦为负。表示 A 希望参数向右上，B 希望向左，两个任务在共享参数上互相拉扯。

### 3. PCGrad：只去掉冲突分量

PCGrad 的核心思想不是“压掉另一个任务”，而是“从当前任务梯度里，删掉与别的任务冲突的那一段”。若 $g_i\cdot g_j<0$，则

$$
g_i^{(PC)}=g_i-\frac{g_i\cdot g_j}{\|g_j\|^2}g_j
$$

这一步等于把 $g_i$ 投影到与 $g_j$ 正交的平面上。直观理解：保留能推进自己任务的部分，去掉会直接伤害别的任务的部分。

### 4. GradNorm：让训练速度重新对齐

GradNorm 不直接改方向，而是调每个任务的 loss 权重，让各任务的梯度范数更接近目标值。定义相对逆训练速率 $r_i$ 后，目标梯度范数为：

$$
G_i^{\text{target}}=\bar G_W\cdot [r_i]^\alpha
$$

其中 $\bar G_W$ 是共享层梯度范数均值，$\alpha$ 控制“纠偏力度”。再定义梯度平衡损失：

$$
L_{\text{grad}}=\sum_i \left|G_i-\bar G_W[r_i]^\alpha\right|
$$

如果某任务学得太慢，GradNorm 会增大它的有效权重；如果某任务学得太快，就会压一压它。

### 5. Flan 的启发：任务混合之外，还要 prompt 混合

真实 instruction tuning 里，同一任务也不该只用一种模板。Flan 系列的经验是，把零样本、少样本、CoT prompt 一起放进训练，模型不只学“答案是什么”，还学“用户可能怎么提问”。

真实工程例子：一个企业内的统一助手，可能同时做 SQL 生成、工单摘要、分类打标、问答检索改写。如果训练时只有“请回答问题”这一种模板，模型遇到“根据表结构写 SQL”或“先分析再总结”时，风格切换会明显变差。把任务采样、loss 加权、梯度协调和 prompt 多样性一起做，才更接近线上真实流量。

---

## 代码实现

下面给一个最小可运行的 Python 版本，演示采样比例、梯度余弦和 PCGrad 投影。它不是训练框架，但逻辑可直接迁移到 PyTorch 训练循环里。

```python
import math

def mix_rates(task_sizes, K=65536, T=2.0):
    capped = {k: min(v, K) for k, v in task_sizes.items()}
    base_sum = sum(capped.values())
    base = {k: v / base_sum for k, v in capped.items()}

    temp_raw = {k: (p ** (1.0 / T)) for k, p in base.items()}
    temp_sum = sum(temp_raw.values())
    mixed = {k: v / temp_sum for k, v in temp_raw.items()}
    return base, mixed

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def norm(a):
    return math.sqrt(dot(a, a))

def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def pcgrad(gi, gj):
    d = dot(gi, gj)
    if d < 0:
        scale = d / dot(gj, gj)
        gi = [x - scale * y for x, y in zip(gi, gj)]
    return gi

sizes = {"A": 1_000_000, "B": 50_000}
base, mixed = mix_rates(sizes, K=65536, T=2.0)

assert round(base["A"], 3) == 0.567
assert round(base["B"], 3) == 0.433
assert round(mixed["A"], 3) == 0.534
assert round(mixed["B"], 3) == 0.466

g1 = [2.0, 1.0]
g2 = [-1.0, 0.0]
assert cosine(g1, g2) < 0

g1_pc = pcgrad(g1, g2)
assert dot(g1_pc, g2) >= -1e-9  # 投影后不再与 g2 冲突
print(base, mixed, g1_pc)
```

变量含义如下：

| 变量 | 含义 |
|---|---|
| `K` | 单任务样本数上限 |
| `T` | 温度，越大越拉平 |
| `w_i` | 第 $i$ 个任务 loss 权重 |
| `g_i` | 第 $i$ 个任务在共享参数上的梯度 |
| `cosine(g_i, g_j)` | 两任务梯度方向一致程度 |

训练伪代码可以写成：

```python
for step in training_steps:
    task = sample_task_by_rate(r_prime)
    loss_dict = forward_each_task(...)
    weighted_loss = sum(w[i] * loss_dict[i] for i in tasks)

    grads = get_per_task_shared_grads(loss_dict)
    if conflict_detected(grads):
        grads = apply_pcgrad(grads)

    if use_gradnorm:
        w = update_loss_weights_by_gradnorm(loss_dict, grads, w)

    optimizer.step()
```

---

## 工程权衡与常见坑

先说结论：多任务训练里最常见的失败，不是模型不够大，而是调度太粗糙。

| 常见坑 | 现象 | 规避策略 |
|---|---|---|
| 直接按原始数据量采样 | 大任务独占训练 | 先设 $K$，再调 $T$ |
| 直接均匀采样 | 小任务重复过多，过拟合 | 对低资源任务加去重或降低重复率 |
| 只看总 loss | 某个子任务悄悄退化 | 分任务记录 loss、指标、采样占比 |
| 不看梯度冲突 | 任务互相拖累但日志看不出 | 定期统计梯度余弦分布 |
| 只加 PCGrad 不调采样 | 冲突减少但资源仍失衡 | 采样和梯度协调要配合 |
| 只调 loss 权重不看 prompt | 指令风格泛化差 | 混入 zero-shot/few-shot/CoT 模板 |

一个实用监控规则是：若一段时间内 $\cos(g_i,g_j)<0$ 的比例持续升高，同时某任务验证集不升反降，就该怀疑负迁移。此时优先级通常是：

1. 先检查采样比例是否失衡。
2. 再检查某任务 loss 是否天然更大。
3. 最后再决定是否上 PCGrad 或 GradNorm。

经验上，$T\approx2$ 常是可用起点，但不是定律。任务差异很大时，$T$ 太高会让低资源任务被过度重复；任务差异很小时，过强的梯度手术反而可能压掉有益共享。

---

## 替代方案与适用边界

PCGrad 和 GradNorm 都不是银弹，它们解决的是“共享参数下的优化冲突”，不是所有多任务问题。

| 方法 | 资源偏差处理 | 梯度冲突处理 | 实现复杂度 | 适用边界 |
|---|---|---|---|---|
| 限额 + 温度采样 | 强 | 弱 | 低 | 首选基线 |
| 静态 loss scaling | 中 | 弱 | 低 | 任务差异不大时够用 |
| GradNorm | 中 | 中 | 中 | loss 尺度差异明显时有效 |
| PCGrad | 弱 | 强 | 中 | 冲突频繁、共享层明显时有效 |
| GradDrop / CAGrad / IMTL | 中 | 强 | 中到高 | 需要更细粒度协调时 |
| 任务专属 Adapter / LoRA 分支 | 强 | 强 | 高 | 任务差异很大、共享困难时 |

对初学者，一个好理解的对照是：

- 简单 loss scaling：像手工给不同任务调音量，便宜但粗糙。
- GradNorm：不是手工调音量，而是根据“谁学得慢”自动回调。
- PCGrad：不改音量，直接删掉彼此打架的方向分量。

如果任务之间高度相关，比如多个近似分类任务，采样调度加轻量 loss 权重常常就够了。若任务输出形态差异很大，比如生成 SQL、抽取字段、长摘要混在一起，仅靠共享主干和简单加权往往不稳，这时应考虑结构化拆分，例如共享编码器加任务头，或共享基座加任务专属 LoRA。

---

## 参考资料

| 资料 | 聚焦点 | 链接 |
|---|---|---|
| Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* | T5 统一 text-to-text 框架，多任务采样与任务混合思想 | [JMLR](https://www.jmlr.org/papers/v21/20-074.html) |
| Yu et al., *Gradient Surgery for Multi-Task Learning* | PCGrad 的冲突检测与投影公式 | [arXiv / 代码入口](https://github.com/tianheyu927/PCGrad) |
| Chen et al., *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks* | 动态 loss 权重与目标梯度范数 | [PMLR](https://proceedings.mlr.press/v80/chen18a.html) |
| Longpre et al., *The Flan Collection: Designing Data and Methods for Effective Instruction Tuning* | task balancing、zero/few/CoT 混合 prompt | [PMLR](https://proceedings.mlr.press/v202/longpre23a.html) |
| Wang et al., *Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models* | 梯度相似度作为多任务诊断信号 | [Google Research](https://research.google/pubs/gradient-vaccine-investigating-and-improving-multi-task-optimization-in-massively-multilingual-models/) |
