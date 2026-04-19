## 核心结论

多任务梯度冲突处理，是在多个任务共用同一组共享参数时，通过调整任务权重或修正梯度方向，减少任务之间的负干扰。

它的目标不是“让所有任务都更重要”，而是让共享参数 $\theta_s$ 上的更新方向更一致，避免一个任务刚把参数推向有利方向，另一个任务又把它拉回来。

统一写法是：

$$
J = \sum_{i=1}^{K} w_i L_i
$$

其中 $K$ 是任务数，$L_i$ 是任务 $i$ 的损失，$w_i$ 是任务权重，$J$ 是最终用于反向传播的总损失。

新手版本可以这样理解：两个任务共用一个骨干网络。任务 A 想把参数往东推，任务 B 想把参数往西拉。直接把梯度相加后，更新可能接近 0，结果是两个任务都学得慢。梯度冲突处理就是想办法让它们少互相抵消。

常见方法可以粗分成两类：

| 类型 | 代表方法 | 改什么 | 解决的问题 |
|---|---|---|---|
| 权重法 | `GradNorm`、`DWA` | 改 $w_i$ | 哪个任务该多学一点，哪个任务该少学一点 |
| 方向修正法 | `PCGrad` | 改梯度方向 | 任务梯度方向互相冲突，更新互相抵消 |

`GradNorm` 和 `DWA` 主要回答“谁学得更多”；`PCGrad` 主要回答“梯度怎么别打架”。这两个问题相关，但不是同一个问题。

---

## 问题定义与边界

多任务学习，是用一个模型同时学习多个目标。常见结构是共享一个 backbone，也就是公共特征提取网络，再给每个任务接一个独立 head，也就是任务专用输出层。

这里讨论的问题只发生在共享参数 $\theta_s$ 上。每个任务都有自己的损失 $L_i$，每个损失都会对共享层产生梯度：

$$
g_i = \nabla_{\theta_s} L_i
$$

如果两个任务的梯度方向接近，说明它们对共享层的更新要求一致；如果方向相反，说明一个任务希望参数往某个方向走，另一个任务希望参数往反方向走。常用判断是点积：

$$
g_i \cdot g_j < 0
$$

点积为负，表示两个梯度夹角大于 90 度，存在方向冲突。

一个图像任务例子：同一张图像里同时做语义分割和深度估计。语义分割要判断每个像素属于哪类物体，通常很依赖边界；深度估计要预测每个像素离相机多远，通常更强调空间连续性。两者都依赖共享 encoder，但对特征的要求不同。如果不处理冲突，训练中经常出现一个任务明显占优，另一个任务长期掉队。

统一记号如下：

| 记号 | 含义 |
|---|---|
| $K$ | 任务数 |
| $\theta_s$ | 共享参数 |
| $L_i$ | 任务 $i$ 的损失 |
| $w_i$ | 任务 $i$ 的权重 |
| $g_i$ | 任务 $i$ 在共享参数上的梯度 |
| $J = \sum_i w_i L_i$ | 总损失 |

边界也要说清楚：

| 场景 | 是否适用 | 原因 |
|---|---:|---|
| 共享 backbone + 多任务 head | 适用 | 多个任务会同时影响共享参数 |
| 推荐系统中 CTR、CVR、停留时长联合建模 | 适用 | 多个目标可能竞争同一套用户和物品表示 |
| 自动驾驶中检测、分割、深度估计联合训练 | 适用 | 多个视觉目标对共享 encoder 的要求不完全一致 |
| 任务完全独立，各自一套模型 | 不适用 | 没有共享参数，也就没有共享梯度冲突 |
| 单任务训练 | 不适用 | 不存在任务之间的梯度竞争 |
| 只是推理阶段多输出 | 不适用 | 训练阶段没有多任务损失共同优化 |

真实工程例子：推荐系统里常见一个共享用户表示和物品表示，同时预测点击率、转化率、观看时长。点击率任务可能偏向短期兴趣，转化率任务可能偏向强意图，观看时长任务可能偏向内容质量。三个任务都更新同一套 embedding 和底层网络，如果直接把 loss 相加，模型可能主要照顾样本更多、梯度更大的任务。

---

## 核心机制与推导

`GradNorm` 的核心是调整任务权重，让不同任务的训练速度更平衡。它关注的是每个任务对共享参数贡献的梯度规模是否合理，不是直接修改梯度方向。

定义任务 $i$ 的加权梯度范数：

$$
G_i(t) = ||\nabla_{\theta_s}[w_i(t)L_i(t)]||_2
$$

梯度范数是梯度向量的长度，可以理解为这个任务当前推动共享层更新的强度。

再定义相对损失下降比例：

$$
\tilde L_i(t) = \frac{L_i(t)}{L_i(0)}
$$

如果 $\tilde L_i(t)$ 大，说明这个任务相对初始状态下降慢。再计算相对训练速度：

$$
r_i(t) = \frac{\tilde L_i(t)}{\frac{1}{K}\sum_j \tilde L_j(t)}
$$

然后给每个任务设置目标梯度范数：

$$
G_i^*(t) = \bar G(t)[r_i(t)]^\alpha
$$

其中 $\bar G(t)$ 是所有任务梯度范数的平均值，$\alpha$ 控制对训练速度差异的敏感程度。最后用下面的辅助损失调整权重：

$$
L_{grad} = \sum_i |G_i(t) - G_i^*(t)|
$$

直白地说，学得慢的任务会被要求有更大的梯度贡献，学得快的任务会被压低一点。

`PCGrad` 的核心是直接处理方向冲突。它检查两个任务梯度的点积。如果点积为负，就把一个梯度中与另一个梯度冲突的分量投影掉：

$$
\text{若 } g_i \cdot g_j < 0,\quad
g_i \leftarrow g_i - \frac{g_i \cdot g_j}{||g_j||^2}g_j
$$

投影是把一个向量拆成“沿着另一个向量的部分”和“垂直于另一个向量的部分”。`PCGrad` 删除的是会和另一个任务相互抵消的那部分。

玩具例子：设两个任务在共享层上的梯度为：

$$
g_1 = (1, 0),\quad g_2 = (-1, 1)
$$

点积为：

$$
g_1 \cdot g_2 = -1 < 0
$$

说明冲突。对 $g_1$ 做投影修正：

$$
g_1' = (1,0) - \frac{-1}{2}(-1,1) = (0.5, 0.5)
$$

原来 $g_1$ 有一部分正好和 $g_2$ 对着干，修正后冲突分量被去掉。

`DWA` 是 Dynamic Weight Average，意思是动态权重平均。它根据最近几轮 loss 的下降速度调整任务权重。先计算：

$$
q_i(t-1) = \frac{L_i(t-1)}{L_i(t-2)}
$$

如果 $q_i$ 大，表示这个任务下降慢。再用 softmax 得到权重：

$$
\lambda_i(t) = \frac{K \exp(q_i/T)}{\sum_j \exp(q_j/T)}
$$

$T$ 是温度参数，用来控制权重变化的平滑程度。$T$ 越大，权重越接近平均；$T$ 越小，权重差异越明显。

三个方法的直观区别如下：

| 方法 | 输入 | 处理对象 | 典型目标 | 新手理解 |
|---|---|---|---|---|
| `GradNorm` | loss 和共享层梯度范数 | 任务权重 | 平衡训练速度 | 成绩落后的课多分点复习时间 |
| `DWA` | 历史 loss | 任务权重 | 轻量级动态加权 | 根据最近几次作业进步情况调时间 |
| `PCGrad` | 每个任务的梯度 | 梯度方向 | 减少方向冲突 | 两个人推桌子，只保留不冲突的力量 |

---

## 代码实现

实现上通常分三步：先前向计算每个任务的损失，再拿到共享层梯度，最后根据 `GradNorm`、`DWA` 或 `PCGrad` 更新参数。

关键点是不要一开始就把所有 loss 直接 `sum()`。训练时先保留 `L1、L2、L3`，看每个任务的梯度或历史损失，再决定是调权重还是投影冲突梯度。这样就像先看每个人的意见，再决定怎么汇总，而不是直接拍板。

建议代码结构如下：

| 函数 | 职责 |
|---|---|
| `compute_task_losses()` | 返回各任务损失 |
| `compute_shared_gradients()` | 获取各任务在共享参数上的梯度 |
| `apply_gradnorm()` | 根据梯度范数更新任务权重 |
| `apply_pcgrad()` | 对冲突梯度做投影 |
| `apply_dwa()` | 根据历史损失更新任务权重 |

伪代码结构：

```python
losses = [L1, L2, ..., LK]

if method == "DWA":
    weights = compute_dwa(history_losses)

weighted_loss = sum(w_i * L_i for w_i, L_i in zip(weights, losses))
weighted_loss.backward()

if method == "PCGrad":
    grads = get_task_grads(shared_params, losses)
    grads = project_conflicting_grads(grads)
    assign_shared_grads(shared_params, grads)

optimizer.step()
```

下面是一个可运行的 Python 玩具实现，演示 `PCGrad` 如何去掉冲突分量：

```python
import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def norm_sq(a):
    return dot(a, a)

def pcgrad_pair(g_i, g_j):
    product = dot(g_i, g_j)
    if product >= 0:
        return g_i[:]
    scale = product / norm_sq(g_j)
    return [x - scale * y for x, y in zip(g_i, g_j)]

def add(a, b):
    return [x + y for x, y in zip(a, b)]

g1 = [1.0, 0.0]
g2 = [-1.0, 1.0]

assert dot(g1, g2) < 0

g1_fixed = pcgrad_pair(g1, g2)
assert all(abs(a - b) < 1e-9 for a, b in zip(g1_fixed, [0.5, 0.5]))

merged_before = add(g1, g2)
merged_after = add(g1_fixed, g2)

assert merged_before == [0.0, 1.0]
assert merged_after == [-0.5, 1.5]
assert dot(g1_fixed, g2) == 0.0

def dwa_weights(q_values, temperature=2.0):
    exps = [math.exp(q / temperature) for q in q_values]
    total = sum(exps)
    k = len(q_values)
    return [k * x / total for x in exps]

weights = dwa_weights([0.8, 1.2], temperature=2.0)
assert len(weights) == 2
assert abs(sum(weights) - 2.0) < 1e-9
assert weights[1] > weights[0]
```

实现注意点：

| 注意点 | 原因 |
|---|---|
| 需要保留每个任务单独 loss | 否则无法计算任务级梯度或历史下降速度 |
| 需要明确共享参数范围 | 冲突处理只应该作用在共享层，不应误伤任务 head |
| 需要记录历史损失 | `DWA` 依赖最近几轮 loss |
| 需要记录梯度范数 | `GradNorm` 依赖共享参数上的梯度强度 |
| 需要控制权重归一化 | 避免总 loss 尺度变化导致学习率漂移 |

在 PyTorch 里，`PCGrad` 通常需要对每个任务 loss 分别调用 `autograd.grad`，拿到共享参数梯度后再做投影，最后手动写回参数的 `.grad`。`GradNorm` 则通常需要把任务权重设成可学习变量，并用额外的 `L_grad` 更新这些权重。

---

## 工程权衡与常见坑

这些方法解决的问题不同，不能把 `GradNorm`、`DWA`、`PCGrad` 当成同一种工具。权重法更适合处理任务量级不平衡，方向法更适合处理梯度冲突。

常见误判是只看总 loss。总 loss 下降不代表每个任务都正常。一个任务 loss 大幅下降，另一个任务指标崩掉，总 loss 仍然可能看起来很好。多任务训练必须单独看每个任务的 loss 和主指标。

常见坑如下：

| 常见坑 | 后果 |
|---|---|
| 只看总 loss，不看各任务指标 | 某个任务掉队时不容易发现 |
| `GradNorm` 共享层选得过宽或过窄 | 梯度范数不能代表真正的共享学习状态 |
| 不做权重归一化 | 总梯度尺度变化，等价于学习率漂移 |
| `DWA` 过早启用 | 早期 loss 波动大，权重会被噪声带偏 |
| `PCGrad` 只处理方向，不处理量级 | 梯度不打架了，但大任务仍然可能压制小任务 |
| 随机顺序导致结果波动 | `PCGrad` 对任务遍历顺序敏感，实验结果可能不稳定 |

规避建议如下：

| 建议 | 做法 |
|---|---|
| 每个任务单独画曲线 | 分别记录 loss、AUC、mIoU、RMSE 等指标 |
| 只处理共享 trunk 的一段参数 | 例如 backbone 最后一段共享层 |
| 让 $\sum_i w_i = K$ | 保持整体 loss 尺度接近等权训练 |
| 前 1-2 个 epoch 先等权 | 等历史 loss 稳定后再启用 `DWA` |
| 固定随机种子 | 降低任务顺序和初始化带来的波动 |
| 多次实验取均值 | 避免单次运行误判方法优劣 |

真实工程例子：在推荐系统里，点击率任务样本多、反馈密集，转化率任务样本少、反馈稀疏。如果直接等权相加，点击率可能主导共享 embedding 的更新。`DWA` 可以根据近期 loss 下降速度给转化率更多权重；`GradNorm` 可以根据共享层梯度范数更精细地平衡任务贡献；如果两个任务在某些 batch 上梯度方向相反，`PCGrad` 可以进一步减少抵消。

但 `PCGrad` 只负责“别打架”，不负责“谁更重要”。如果业务上明确要求转化率优先，仍然需要业务权重、采样策略或评价指标约束。算法不能替代目标定义。

---

## 替代方案与适用边界

`GradNorm` 适合任务训练速度差异明显的情况，尤其是某些任务收敛太快、另一些任务长期落后时。它比简单手调权重更自动，但实现复杂度更高，因为它要计算共享层梯度范数，还要维护可学习的任务权重。

`DWA` 适合做轻量级动态加权。它只依赖历史 loss，不需要额外计算每个任务的共享梯度，所以接入成本低。缺点是它只看 loss 下降速度，不直接知道梯度是否冲突。

`PCGrad` 适合冲突明确、方向矛盾明显的场景。它直接处理梯度方向，但不保证任务权重合理，也不保证主任务优先级。

方法选择可以按下面的表来做：

| 方法 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| `DWA` | 实现简单，依赖历史损失 | 对 loss 噪声敏感，不直接看梯度 | 想快速接入动态加权 |
| `GradNorm` | 更精细，能平衡训练速度 | 需要梯度范数，实现更复杂 | 任务学习速度差异明显 |
| `PCGrad` | 直接处理冲突方向 | 不处理任务重要性和量级 | 梯度方向冲突明显 |
| 权重法 + 方向法 | 同时处理量级和方向 | 调参和实现成本更高 | 任务既不平衡又互相冲突 |

适用边界如下：

| 条件 | 说明 |
|---|---|
| 任务高度相关但不完全一致 | 完全无关时，共享本身可能就不合理 |
| 共享层明显存在梯度冲突 | 可以通过点积、余弦相似度观察 |
| 需要兼顾主任务和辅助任务 | 辅助任务不能拖垮主任务 |
| 训练指标出现任务掉队 | 某些任务长期不收敛或明显变差 |

新手可以按这个顺序判断：如果两个任务主要问题是“一个学得太快，一个学得太慢”，先考虑 `GradNorm` 或 `DWA`。如果两个任务主要问题是“更新方向互相打架”，先考虑 `PCGrad`。如果两种问题都存在，可以先做权重平衡，再做方向修正。

实际落地时，还要先确认多任务共享是否值得。若任务关系很弱，强行共享 backbone 会让梯度冲突变成结构性问题。这时与其在优化器上补救，不如考虑减少共享层、使用任务专用 adapter，或者把任务拆成独立模型。

---

## 参考资料

参考资料不是为了堆论文名，而是为了确认三件事：方法到底在改权重还是改方向、公式怎么写、代码怎么落地。

1. [Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://arxiv.org/abs/1711.02257)
2. [Gradient Surgery for Multi-Task Learning](https://papers.nips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
3. [End-To-End Multi-Task Learning With Attention](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.html)
4. [lorenmt/mtan: Multi-Task Attention Network](https://github.com/lorenmt/mtan)
5. [tianheyu927/PCGrad](https://github.com/tianheyu927/PCGrad)
