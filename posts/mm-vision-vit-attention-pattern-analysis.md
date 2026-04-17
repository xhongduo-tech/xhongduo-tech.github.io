## 核心结论

ViT 的注意力模式分析，本质上是在回答一个具体问题：一层里的每个 attention head，到底更依赖 `CLS token` 还是普通 `patch token`，更偏向局部邻域还是全局汇聚，以及它是否沿某种空间方向稳定分布。`CLS token` 可以理解为“整张图的汇总槽位”，模型最终常用它做分类；`patch token` 就是图像切块后的单元表示。

把图像看成拼图是有帮助的，但这里不能停在比喻。严格地说，ViT 把图像切成网格 patch，每个 head 都在构造一个 $N \times N$ 的注意力矩阵，其中 $N$ 是 token 数。这个矩阵的每一行表示“当前 token 在看谁”，每一列表示“谁被看得更多”。分析这些矩阵后，通常能看到几类稳定模式：

| 头类型 | 空间模式 | 常见作用 | 可视化判断依据 |
|---|---|---|---|
| CLS 聚合头 | 大量权重流向或流出 CLS | 汇总全局语义，服务分类 | CLS 行或 CLS 列明显高亮 |
| 局部头 | 权重集中在邻近 patch | 提取纹理、边缘、局部一致性 | 主对角线附近亮，离得远迅速变暗 |
| 全局头 | 权重分散到大范围 patch | 建立长距离依赖 | 热力图覆盖面广，不只盯近邻 |
| 垂直方向头 | 沿上下方向有条纹结构 | 建模竖直排列物体或文本 | 列方向模式更稳定 |
| 水平方向头 | 沿左右方向有条纹结构 | 建模横向结构或版面 | 行方向模式更稳定 |

对真实系统来说，这种分析不是“看图说话”，而是帮助判断模型到底靠什么在工作：它是在抓主体轮廓，还是在死记局部纹理；是在稳健地聚合全局信息，还是被某几个脆弱 head 主导。

---

## 问题定义与边界

问题可以定义为：给定 ViT 某层某个 head 的注意力矩阵，如何量化它对 CLS、邻域、远距离 token 和空间方向的偏好，并据此解释它对下游任务的贡献。

先把对象说清楚。ViT 输入图像后，会先切成固定大小的 patch，再映射成向量，这一步叫 `patch embedding`，也就是“把像素块变成模型能运算的 token 表示”。假设图像被切成 $14 \times 14$ 个 patch，那么共有 $196$ 个 patch token；再加 1 个 CLS token，序列长度通常是 $N=197$。每个 head 的注意力计算是：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中：

$$
Q = XW_q,\quad K = XW_k,\quad V = XW_v
$$

$X$ 是输入 token 序列，$W_q, W_k, W_v$ 是可学习矩阵。白话说，$Q$ 决定“我想找什么”，$K$ 决定“我能提供什么线索”，$V$ 是“真正被取走的信息”。

这里的边界也要说清楚：

| 分析对象 | 能回答的问题 | 不能单独回答的问题 |
|---|---|---|
| 单层单头热力图 | 这个 head 在盯谁 | 整个模型最终为何预测正确 |
| 多层统计结果 | 早层偏局部还是晚层偏全局 | 训练数据偏差的根本来源 |
| CLS/patch 权重比 | 分类是否依赖全局汇总 | 定位质量一定更好 |
| 方向性模式 | 是否存在版面或结构先验 | 语义概念是否真正对齐人类解释 |

一个新手容易误解的点是：注意力高，不等于因果贡献一定高。它只能说明“信息路由倾向”，不能单独证明“这个 token 决定了预测结果”。因此注意力模式分析更适合做机制解释、错误排查和工程调试，而不是直接替代完整归因方法。

玩具例子：假设只有 4 个 patch 加 1 个 CLS，共 5 个 token。如果某个 head 的 CLS 行对所有 patch 都给出接近均匀的高权重，这说明它在做全局收集；如果另一个 head 只让 patch 2 强烈关注 patch 1 和 patch 3，它更像局部边缘检测器。哪怕这是最小例子，也已经包含了 ViT 分工的基本形式。

---

## 核心机制与推导

ViT 注意力模式分析通常分三步：先取出注意力矩阵，再定义强度指标，最后做模式聚类。`强度指标` 就是“一个 head 在某类关系上有多活跃”；`聚类` 就是“把长得像的 head 放到一组里”。

第一步是矩阵级观察。设某层某个 head 的注意力矩阵为 $A \in \mathbb{R}^{N \times N}$，其中：

$$
A_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{t=1}^{N}\exp(q_i \cdot k_t / \sqrt{d_k})}
$$

那么每一行和为 1。若第 0 个 token 是 CLS，则可以定义 CLS 强度：

$$
s_{\text{cls}} = \frac{1}{N-1}\sum_{i=1}^{N-1} A_{i,0}
$$

它表示普通 patch 平均有多愿意把信息交给 CLS。再定义局部强度。若 token $i$ 在二维网格上的邻域为 $\mathcal{N}_k(i)$，其中 $k$ 表示 hop 半径，那么：

$$
s_{\text{local}} = \frac{1}{N_p}\sum_{i=1}^{N_p}\frac{1}{|\mathcal{N}_k(i)|}\sum_{j \in \mathcal{N}_k(i)} A_{ij}
$$

这里 $N_p$ 是 patch token 数。它衡量“平均看邻居”的程度。对应地，全局比值可定义为远距离区域平均权重与近邻平均权重之比。

可以把流程压缩成一条简单链路：

$$
\text{attention matrix} \rightarrow \text{k-hop average} \rightarrow \text{pattern features} \rightarrow \text{clustering}
$$

含义是：先从矩阵中提取邻域平均、CLS 比例、方向统计等特征，再把 head 聚成“CLS 型、局部型、全局型、方向型”。

一个 6 个 head 的示意例子很典型：
1. 头 1、头 2 高亮 CLS，说明它们更像语义汇聚器。
2. 头 3、头 4 主要亮在对角线附近，说明它们偏局部纹理。
3. 头 5 出现竖向条纹，可能在建模文本列或立柱结构。
4. 头 6 覆盖整幅图，常见于后层全局上下文建模。

这类分工解释了为什么 ViT 能同时保留空间结构和全局语义。它不是靠一个 head 完成所有工作，而是通过多头并行，把“看中心、看邻居、看远处、看方向”分开建模。多模态系统里，这一点更重要，因为视觉 token 还可能和文本或其他模态的 token 对齐，若没有分工，语义会被过度混合，导致表示塌缩。

---

## 代码实现

下面给一个可运行的最小 Python 例子。它不依赖具体 ViT 框架，只演示如何从注意力矩阵计算 `CLS 平均权重` 和 `局部/全局比`。这里把“局部”简化为距离 1 的邻接关系，足够说明方法。

```python
import math

def row_softmax(row):
    exps = [math.exp(x) for x in row]
    s = sum(exps)
    return [x / s for x in exps]

def normalize_attention(scores):
    return [row_softmax(row) for row in scores]

def analyze_attention(attn, cls_idx=0):
    n = len(attn)
    cls_weight = sum(attn[i][cls_idx] for i in range(1, n)) / (n - 1)

    local_sum = 0.0
    local_cnt = 0
    global_sum = 0.0
    global_cnt = 0

    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            if abs(i - j) == 1:
                local_sum += attn[i][j]
                local_cnt += 1
            elif abs(i - j) >= 2:
                global_sum += attn[i][j]
                global_cnt += 1

    local_avg = local_sum / local_cnt
    global_avg = global_sum / global_cnt
    return cls_weight, local_avg / global_avg

scores = [
    [3.0, 1.0, 1.0, 1.0, 1.0],  # CLS 看所有 token
    [2.5, 3.0, 2.0, 0.2, 0.1],  # patch1 偏 CLS 和近邻
    [2.0, 2.2, 3.0, 2.1, 0.2],
    [1.8, 0.1, 2.0, 3.0, 2.2],
    [1.9, 0.1, 0.2, 2.0, 3.0],
]

attn = normalize_attention(scores)
cls_weight, local_global_ratio = analyze_attention(attn)

assert len(attn) == 5
assert 0.0 < cls_weight < 1.0
assert local_global_ratio > 1.0

print(round(cls_weight, 4), round(local_global_ratio, 4))
```

这个例子的输入是未归一化的 attention score，输出是两个调试指标：
1. `cls_weight`：patch 平均给 CLS 多少权重。
2. `local_global_ratio`：局部关注是否强于远距离关注。

如果接入真实 ViT，流程一般是：
1. 前向时拿到每层每个 head 的 attention weight。
2. 对每个 head 计算 CLS 权重、局部比、方向性统计。
3. 输出一张表辅助排查。

例如可以生成这样的调试表：

| layer | head | avg_cls_weight | local/global_ratio | pattern |
|---|---|---:|---:|---|
| 3 | 1 | 0.31 | 0.84 | 全局/CLS |
| 3 | 2 | 0.09 | 2.77 | 局部 |
| 7 | 5 | 0.22 | 1.15 | 垂直方向 |
| 11 | 0 | 0.35 | 0.91 | CLS 聚合 |

真实工程例子是电力柜文本识别中的 SA-ViT。那类任务里，某些 head 会稳定跟踪码值串，另一些 head 更依赖标签和值之间的栅格一致性。只看最终准确率，你只能知道“模型坏了”；但看 attention pattern，你会发现极端透视角下，本来该沿文本方向传播的 head 开始漂移，定位和识别一起退化。

---

## 工程权衡与常见坑

注意力模式能解释行为，但工程里更关心“什么时候会失效”。

第一类问题是软错误和量化误差。`软错误` 可以理解为硬件瞬时翻转或数值扰动，不是代码逻辑错，而是运行时比特或激活值出了偏差。ViT 早层的 Q/K 一旦抖动，后续 softmax 会把差异非线性放大，导致某些 head 从“局部头”突然变成“噪声扩散头”。这在航天、车载、边缘设备里尤其危险。

第二类问题是二次复杂度。注意力矩阵大小是 $N^2$，分辨率一上去就爆。若 patch 数从 $196$ 增到 $784$，矩阵元素会从 $3.8 \times 10^4$ 变成 $6.1 \times 10^5$，不只是显存变大，分析本身也更难做。

第三类问题是多模态过聚合。视觉和文本一起输入时，若某些 head 过度偏向全局混合，模型可能把本该区分的局部视觉信号提前平均掉，最后出现“都差不多”的语义偏置。

| 常见坑 | 典型表现 | 根因 | 规避策略 |
|---|---|---|---|
| 软错误放大 | 早层小扰动导致后层热力图失真 | Q/K 对 softmax 敏感 | ECC、激活校验、关键层冗余 |
| 二次复杂度过高 | 高分辨率下显存和延迟失控 | attention 是 $O(N^2)$ | window attention、TokenLearner、token merging |
| 多模态过聚合 | 不同表情或对象被混成同类 | 全局头过强，局部差异被抹平 | 语义权重、uncertainty weighting |
| 只看注意力图下结论 | 可视化好看但解释失真 | 注意力不等于因果贡献 | 结合遮挡实验、归因或误差分析 |
| 方向头失配 | 版面变化后识别突降 | 训练分布中的空间先验过强 | 数据增强、透视鲁棒训练 |

SA-ViT 的失效就是典型例子。在电力柜文本识别里，当视角极端倾斜、柜型又超出训练分布时，原本沿码值串传播的注意力头会发生 pattern mismatch，也就是“该看一条线时没有看成一条线”。这类错误会带来约 15% 到 20% 的准确率下降，而且往往不是单一 token 问题，而是多个关键 head 同时偏移。

---

## 替代方案与适用边界

不是所有场景都适合 full attention。若输入分辨率高、设备内存紧、实时性要求强，直接保留全局 $N \times N$ 注意力往往不现实。

先看一个简化对比。设 patch 数为 $N=196$，通道维度为 $d=64$。full attention 的核心开销近似与 $N^2d$ 成正比；局部窗口注意力若窗口边长为 $k=7$，则只在局部窗口内做关系建模，开销明显下降。

| 方案 | 计算成本 | 模式保留程度 | 适用场景 |
|---|---|---|---|
| Full Attention | 高，约随 $N^2$ 增长 | 最完整，能保留长程依赖 | 离线分析、高精度分类、多模态对齐 |
| Window Attention | 中等，局部窗口内计算 | 保留局部结构，丢失部分远程关系 | 实时视觉、移动端、高分辨率输入 |
| Sparse Attention | 中到低，按规则稀疏连接 | 保留关键远程连接，依赖设计质量 | 大图建模、结构明显的场景 |
| TokenLearner / Token Merging | 低，通过压缩 token 降成本 | 可能损失细粒度局部模式 | 资源受限部署、视频或多帧输入 |

新手可以这样理解：
1. `Full attention` 是所有 token 都互相开会，信息最全，但会很贵。
2. `Window attention` 是先分小组讨论，只看邻近成员，省钱但可能漏掉远距离关系。
3. `Token merging` 是先把相似成员合并，再开会，更省，但容易把细节合丢。

因此适用边界很明确：
1. 如果任务依赖全局语义分类，且 token 数不大，full attention 值得保留。
2. 如果任务更偏局部结构，如文本行、格栅、部件边界，window attention 往往已经够用。
3. 如果是多模态融合，不能只追求压缩，还要监控是否出现“跨模态过早平均”。
4. 如果你做的是解释分析，压缩前后都要看 attention pattern，否则无法判断性能变化来自计算裁剪还是机制改变。

---

## 参考资料

1. [PMC10290521] Vision Transformer visual analytics，2023。贡献：系统讨论 head importance、strength、pattern 的可视化流程，支撑本文对 CLS/局部/方向型头的划分。
2. Fine-Grained Fault Sensitivity in Vision Transformers，2025。贡献：分析不同层和不同 head 对软错误的脆弱性，支撑本文关于早层 Q/K 扰动会被放大的工程判断。
3. SA-ViT for text recognition in electrical cabinets，2025。贡献：给出真实工程里的 attention 失配案例，支撑本文关于透视偏差和训练外分布导致准确率下降的讨论。
4. Visual Perception Through Vision Transformers，博客综述。贡献：用较直观方式解释 patch embedding、Q/K/V 和 attention matrix，适合作为入门配套阅读。
5. Vision Transformers Explained，工程综述。贡献：给出 patch 数、计算量和局部窗口注意力的简化估算，支撑本文对 $O(N^2)$ 成本的说明。
