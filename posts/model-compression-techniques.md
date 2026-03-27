## 核心结论

模型压缩的目标不是“把模型变小”这么简单，而是**在精度损失可控的前提下，同时降低参数量、显存占用、磁盘体积和推理延迟**。工程上最常见的三类方法是剪枝、知识蒸馏和神经架构搜索（NAS，白话讲就是“自动找更省资源的网络结构”）。

它们解决的问题并不相同。剪枝直接删掉冗余参数或通道；知识蒸馏把大模型学到的决策模式传给小模型；NAS从结构层面寻找更适合目标硬件的网络。三者可以单独使用，也可以组合使用，常见顺序是“先选结构，再蒸馏，再做部署友好的剪枝”。

一个最直观的玩具例子：某图像分类模型原始大小为 120MB，压缩后为 30MB，单次推理从 20ms 降到 8ms，准确率从 92% 降到 91%。则：

$$
C=\frac{120}{30}=4,\quad
\delta=\frac{20}{8}=2.5,\quad
\Delta_{\text{acc}}=92\%-91\%=1\%
$$

这里 $C$ 是压缩率，表示体积缩小了多少倍；$\delta$ 是加速比，表示推理速度提升了多少；$\Delta_{\text{acc}}$ 是精度损失，表示性能下降了多少。

| 指标 | 定义 | 例子 |
|---|---|---|
| 压缩率 $C$ | 原始模型大小 / 压缩后大小 | $120/30=4$ |
| 加速比 $\delta$ | 原始延迟 / 压缩后延迟 | $20/8=2.5$ |
| 精度损失 $\Delta_{\text{acc}}$ | 原始准确率 - 压缩后准确率 | $92\%-91\%=1\%$ |

如果你的部署平台真正关心的是延迟和吞吐，而不是单纯的参数个数，那么“能删多少”往往不如“删完后硬件能不能跑得更快”重要。这也是为什么结构化剪枝通常比非结构化剪枝更容易在真实系统里拿到收益。

---

## 问题定义与边界

模型压缩的标准问题可以写成：在精度下降不超过阈值 $\epsilon$ 的条件下，最小化模型大小、显存占用和延迟。形式化写法是：

$$
\min_{\theta'} \ \text{Size}(\theta'),\ \text{Latency}(\theta')
\quad \text{s.t.} \quad
\text{Acc}(\theta)-\text{Acc}(\theta') \le \epsilon
$$

这里 $\theta$ 是原始模型参数，$\theta'$ 是压缩后的模型参数。

边界主要来自四类约束。

| 约束维度 | 典型问题 | 工程含义 |
|---|---|---|
| 硬件资源 | 显存只有 2GB，CPU 没有向量化优化 | 决定能否直接上大模型 |
| 延迟目标 | 在线接口必须低于 50ms | 决定优先做结构优化还是离线压缩 |
| 精度容忍度 | 掉点不能超过 0.5% | 决定压缩强度上限 |
| 算子支持 | 设备不支持稀疏矩阵加速 | 决定非结构化剪枝是否有意义 |

真实工程例子：边缘设备做人脸识别，设备显存只有 2GB，原模型虽然准确，但只能跑到 5 FPS。需求是至少 20 FPS，且识别精度下降不能超过 0.5%。这时问题不是“有没有更小的模型”，而是“在设备算子支持范围内，怎么把延迟压下去”。如果目标芯片不支持稀疏乘法，那么大比例非结构化剪枝即使把参数删掉，也可能拿不到足够的 FPS，反而应该优先考虑通道剪枝、轻量骨干网络和蒸馏。

因此，模型压缩从来不是脱离部署环境单独讨论的算法题，而是一个和硬件、框架、业务指标强绑定的系统问题。

---

## 核心机制与推导

### 1. 剪枝

剪枝就是删除“不重要”的参数。这里“不重要”通常指删掉后损失变化小。最简单的判据是幅度，即权重绝对值越小，越可能不重要，这叫基于幅度的剪枝。更精细的做法会结合梯度或 Hessian 近似，估计某个参数被删掉后损失增加多少，这叫敏感度评估。

剪枝有两种主流形态：

| 类型 | 删什么 | 优点 | 缺点 |
|---|---|---|---|
| 非结构化剪枝 | 单个权重 | 压缩率高 | 硬件通常难加速 |
| 结构化剪枝 | 通道、卷积核、层 | 易部署、易加速 | 精度更敏感 |

玩具例子：一个全连接层有 8 个权重，绝对值最小的 4 个接近 0。删掉这 4 个权重，参数量减半，但矩阵维度没变，所以在通用 GPU 上未必更快。若改为直接删除一整列或一整个输出通道，矩阵形状变小，实际推理更容易加速。

迭代剪枝常比一次性剪掉更稳。因为每删一部分参数，就重新训练一次，让模型适应新的结构。典型流程是：训练原模型 $\rightarrow$ 剪掉 10% $\rightarrow$ 微调 $\rightarrow$ 再剪 10% $\rightarrow$ 微调。

### 2. 知识蒸馏

知识蒸馏中的教师模型，就是性能更强但更重的模型；学生模型，就是更小、更快、目标是接近教师效果的模型。学生不仅学习真实标签，还学习教师输出的“软标签”，也就是一组带有类别相似度信息的概率分布。

温度 $T$ 的作用，是把教师输出的概率分布变得更平滑：

$$
p_i^{(T)}=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

这里 $z_i$ 是 logits，也就是 softmax 之前的分数。$T$ 越高，分布越平；$T$ 越低，分布越尖锐。蒸馏损失常写成：

$$
\mathcal{L}= \alpha \mathcal{L}_{CE}+(1-\alpha)T^2 \mathcal{L}_{KD}
$$

其中 $\mathcal{L}_{CE}$ 是学生对真实标签的交叉熵，$\mathcal{L}_{KD}$ 是学生与教师软标签之间的 KL 散度或交叉熵。$T^2$ 是尺度补偿项，避免高温度下梯度过小。

白话理解：硬标签只告诉学生“答案是猫”，软标签还告诉学生“像猫，也有一点像狐狸，不太像汽车”。这类相对关系对小模型很有价值。

### 3. 神经架构搜索

NAS不是直接压现有模型，而是在给定搜索空间里自动找结构。搜索目标通常不是单一准确率，而是多目标优化：

$$
\max_{a \in \mathcal{A}} \ \text{Acc}(a)-\lambda_1 \text{Latency}(a)-\lambda_2 \text{FLOPs}(a)-\lambda_3 \text{Size}(a)
$$

这里 $a$ 是候选结构，$\mathcal{A}$ 是搜索空间。核心区别在于：剪枝和蒸馏更像“优化已有模型”，NAS更像“从源头上找更省资源的模型”。

如果没有把真实硬件延迟纳入目标函数，NAS很可能搜出 FLOPs 很低、但在目标设备上并不快的结构。这是初学者最容易忽略的地方。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现，演示幅度剪枝的核心逻辑。它不依赖深度学习框架，但能清楚说明“按阈值生成 mask，再应用到参数上”的过程。

```python
import numpy as np

def magnitude_prune(weights, prune_ratio):
    assert 0.0 <= prune_ratio < 1.0
    flat = np.abs(weights).reshape(-1)
    k = int(len(flat) * prune_ratio)
    if k == 0:
        mask = np.ones_like(weights, dtype=np.float32)
        return weights * mask, mask

    threshold = np.partition(flat, k - 1)[k - 1]
    mask = (np.abs(weights) > threshold).astype(np.float32)
    pruned = weights * mask
    return pruned, mask

w = np.array([0.01, -0.8, 0.03, 1.2, -0.02, 0.5], dtype=np.float32)
pruned_w, mask = magnitude_prune(w, prune_ratio=0.5)

assert pruned_w.shape == w.shape
assert int(mask.sum()) <= len(w)
assert np.allclose(pruned_w[mask == 0], 0.0)
assert np.count_nonzero(pruned_w) <= np.count_nonzero(w)
```

如果换成 PyTorch，训练中的典型流程如下。这里的 `mask` 是布尔掩码，表示哪些权重保留。

```python
import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, targets, alpha=0.5, T=4.0):
    ce = F.cross_entropy(student_logits, targets)
    s_log_prob = F.log_softmax(student_logits / T, dim=1)
    t_prob = F.softmax(teacher_logits / T, dim=1)
    kd = F.kl_div(s_log_prob, t_prob, reduction="batchmean")
    return alpha * ce + (1 - alpha) * (T * T) * kd

# 伪代码
# 1. 训练 teacher
# 2. 初始化 student
# 3. 计算权重幅度，生成 mask
# 4. 每次 optimizer.step() 后重新应用 mask
for x, y in dataloader:
    with torch.no_grad():
        teacher_logits = teacher(x)

    student_logits = student(x)
    loss = kd_loss(student_logits, teacher_logits, y, alpha=0.3, T=4.0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 重新应用剪枝 mask，防止被更新回非零
    for p, m in zip(student.parameters(), masks):
        p.data.mul_(m)
```

真实工程里常见的组合是：

1. 先选一个轻量学生模型，如 MobileNet、ShuffleNet 或小型 Transformer。
2. 用教师模型做蒸馏，先把学生精度拉起来。
3. 对学生做结构化剪枝，例如删掉注意力头、MLP 通道或卷积通道。
4. 微调恢复精度。
5. 在目标硬件上重新测延迟，而不是只看参数量或 FLOPs。

这个顺序比“先把大模型剪得很稀，再想办法部署”通常更稳，因为结构先天轻量的模型更容易得到稳定收益。

---

## 工程权衡与常见坑

模型压缩最常见的误区，是把“参数少”误认为“速度快”。两者经常相关，但并不等价。

| 常见坑 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| 非结构化剪枝后不加速 | 参数少了，延迟没降 | 硬件不支持稀疏加速 | 优先结构化剪枝 |
| 蒸馏温度过高 | 学生学不到清晰决策边界 | 软标签过于平滑 | 在 2 到 6 之间做验证 |
| 蒸馏温度过低 | 退化成硬标签训练 | 类间关系信息不足 | 联合调节 $T$ 与 $\alpha$ |
| NAS 结果难部署 | 搜到的结构实验室里好看，线上不快 | 目标函数没有真实硬件延迟 | 做硬件感知搜索 |
| 只看 FLOPs | FLOPs 下降明显，QPS 变化很小 | 内存访问和算子调度才是瓶颈 | 以真实 benchmark 为准 |

一个真实工程例子：某推荐模型做了 70% 非结构化剪枝，离线统计参数量下降很多，但在线 GPU 推理几乎没有加速。排查后发现，部署框架仍然执行稠密矩阵乘法，稀疏模式没有真正启用。后来改为按通道剪枝，再配合 TensorRT 重新导出，虽然压缩率不如之前激进，但端到端延迟稳定下降了接近 2 倍。

还要注意一个细节：蒸馏并不是“教师越强越好”。如果教师和学生容量差距极大，或者任务标签本身噪声高，学生可能学到的是教师的偏差。对于数据分布漂移明显的业务，蒸馏效果也会明显下降。

---

## 替代方案与适用边界

剪枝、蒸馏和 NAS 不是唯一方案。很多场景下，量化、低秩分解和混合精度更直接。

| 方法 | 主要作用 | 压缩收益 | 部署难度 | 适用边界 |
|---|---|---|---|---|
| 量化 | 用更低位宽表示参数和激活 | 高 | 中 | 设备支持 int8/fp16 时很有效 |
| 低秩分解 | 用更小矩阵近似大矩阵 | 中 | 中 | 线性层、卷积层冗余明显时 |
| 混合精度 | 关键层保高精度，其余降精度 | 中 | 低到中 | 追求低改造成本时 |
| 轻量骨干替换 | 直接换更高效网络 | 高 | 中 | 新项目或可改结构时 |

一个典型场景是移动端 SDK 集成。第三方接口不允许你改网络结构，但允许你导出 int8 模型，那么量化常比剪枝更实用。再比如大语言模型部署，如果显存是主要瓶颈，4bit/8bit 量化通常比单纯剪枝更有效；如果在线延迟是主要瓶颈，则要进一步看 KV cache、算子融合和 batch 策略，而不是只盯着参数量。

因此，适用边界可以概括为：

- 需要直接降低结构计算量，优先考虑结构化剪枝或轻量结构设计。
- 需要在小模型上尽量保住精度，优先考虑知识蒸馏。
- 需要系统级搜索最优结构，且有足够训练预算，考虑 NAS。
- 不能改结构、但能改数值表示时，量化往往是成本最低的方案。

工程上最稳的路线通常不是单一方法，而是“轻量结构 + 蒸馏 + 量化”，剪枝作为补充手段按硬件支持情况选择。

---

## 参考资料

1. SystemOverflow, *Structured vs Unstructured Pruning*  
   重点看结构化与非结构化剪枝在硬件友好性上的差异，适合理解“为什么删了参数不一定变快”。  
   https://www.systemoverflow.com/learn/ml-model-optimization/model-pruning/structured-vs-unstructured-pruning-core-differences

2. ScienceDirect, *Knowledge Distillation: A Survey*  
   系统整理了软标签蒸馏、特征蒸馏、关系蒸馏等分支，适合建立完整知识图谱。  
   https://www.sciencedirect.com/

3. TomorrowDesk, *Model Compression*  
   对压缩率、加速比、精度损失等评估指标讲得较清楚，适合作为入门度量参考。  
   https://tomorrowdesk.com/info/model-compression

4. Red Hat Developers, *LLM Compressor: Optimize LLMs for low-latency deployments*  
   更偏工程实践，适合理解大模型在低延迟部署中的压缩、量化与系统协同。  
   https://developers.redhat.com/articles/2025/05/09/llm-compressor-optimize-llms-low-latency-deployments

5. Emergent Mind, *Model Compression Techniques*  
   适合快速浏览模型压缩主流技术版图，建立剪枝、蒸馏、搜索、量化之间的关系。  
   https://www.emergentmind.com/topics/model-compression-techniques
