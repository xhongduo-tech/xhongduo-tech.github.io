## 核心结论

差分隐私（Differential Privacy, DP，白话说就是“让单条数据的有无几乎不改变最终输出”）在训练阶段最常见的落地方法是 DP-SGD。它不是修改数据集，也不是只在发布结果时做脱敏，而是在每一步参数更新里限制单样本影响。

DP-SGD 有两个不可少的动作：

1. 对每个样本的梯度做裁剪。裁剪的意思是“把单个样本对参数更新的最大推动力封顶”。
2. 对聚合后的梯度加高斯噪声。高斯噪声就是“从正态分布随机采样的一段扰动”，让更新结果从确定值变成随机值。

因此，普通 SGD 的更新是 deterministic，也就是同样的数据、初始化、顺序下会得到同样的梯度更新；DP-SGD 的更新是 probabilistic，也就是每一步都会带随机扰动。攻击者即使知道模型结构、训练流程，观察最终模型，也更难判断某个样本是否参加过训练。

下表先给出最重要的隐私-效果关系：

| 参数/现象 | 变小/变大时的影响 | 直观含义 |
| --- | --- | --- |
| $\varepsilon$ 变小 | 隐私更强，通常需要更大噪声，精度更容易下降 | 允许输出分布差异更小 |
| $\delta$ 变小 | 失败概率更低，要求更严格，预算更难满足 | “极少数坏事件”的上界更小 |
| 裁剪阈值 $C$ 变小 | 单样本影响更受限，但偏差更大 | 很多真实梯度会被压扁 |
| 噪声系数 $\sigma$ 变大 | 隐私更强，但收敛更慢 | 每步更新更模糊 |
| 训练步数变多 | 隐私预算持续累积 | 不能只看单步 |

一个最小玩具例子可以直接说明机制。假设一个 batch 里只有两个样本，普通 SGD 的梯度分别是 $g_1=(6,8)$、$g_2=(1,1)$。若裁剪阈值 $C=5$，则 $\|g_1\|=10$，会被缩放成 $\tilde g_1=(3,4)$，而 $g_2$ 不变。此时单样本再大，也只能把贡献推到半径 5 的球面内。随后再给均值梯度加一个高斯噪声向量，模型更新就不再精确暴露“是否出现了那个大梯度样本”。

---

## 问题定义与边界

差分隐私的标准定义如下。设 $D$ 与 $D'$ 是相邻数据集，意思是它们只相差一个样本；设 $\mathcal M$ 是一个随机机制，意思是“输入数据后会带随机性地产生输出”的过程。那么 $\mathcal M$ 满足 $(\varepsilon,\delta)$-DP，当且仅当对任意事件集合 $O$ 都有：

$$
\Pr[\mathcal M(D)\in O] \le e^\varepsilon \Pr[\mathcal M(D')\in O] + \delta
$$

这句话的白话解释是：把某个人的数据放进训练集，或者拿掉，最终输出分布只允许有有限差异。$\varepsilon$ 控制这个差异上限，$\delta$ 控制极小概率下允许失效的空间。

这里的边界必须说清楚：

| 能保护什么 | 不能自动保护什么 |
| --- | --- |
| 降低成员推断攻击成功率 | 不保证模型完全无信息泄露 |
| 限制单条训练记录对参数的影响 | 不解决数据投毒、标签错误、版权问题 |
| 给出可组合的数学隐私账本 | 不等于所有中间日志、缓存、检查点都安全 |

训练阶段的核心攻击面是成员推断（membership inference，白话说就是“猜某条数据是否在训练集中”）。攻击者可能是黑盒，也就是只能查询模型输出；也可能是白盒，也就是知道模型结构、参数甚至训练流程。DP 的价值不在于“让模型什么都学不会”，而在于“让加入或删除一个样本后，输出变化受控到不足以稳定区分”。

可以用一个接近真实场景的 CIFAR-10 mini 例子理解。设有两个训练集：

- $D$：包含 1000 张图像
- $D'$：和 $D$ 完全相同，只多了一张用户的私人图片 $x^\*$

如果用普通 SGD 训练，高置信度输出、损失曲线细节、甚至最终参数都可能因为这张图被可检测地改变。若该样本罕见、容易被记忆，攻击者就更容易判断它是否存在。DP-SGD 的目标不是让 $D$ 和 $D'$ 训练出的模型完全一样，而是让它们的输出分布足够接近，接近到攻击者无法高把握地区分。

---

## 核心机制与推导

DP-SGD 的第一步是按样本计算梯度。这里的 per-sample gradient，白话说就是“每条样本各自对参数提出一次更新意见”。

对第 $i$ 个样本的原始梯度 $g_i$，做 $\ell_2$ 裁剪：

$$
\tilde g_i = \frac{g_i}{\max(1, \|g_i\|_2 / C)}
$$

它的作用是把敏感度界定在 $C$。敏感度可以理解为“换掉一条记录后，聚合结果最多能变多少”。

随后对一个 batch 大小为 $B$ 的批次，做聚合并加噪。常见写法是：

$$
\theta_{t+1}=\theta_t-\eta\left(\frac{1}{B}\sum_{i=1}^{B}\tilde g_i+\mathcal N(0,\sigma^2 C^2 I)\right)
$$

其中：

- $\theta_t$：第 $t$ 步参数
- $\eta$：学习率
- $C$：裁剪阈值
- $\sigma$：噪声系数
- $I$：单位矩阵，表示各维度独立加噪

单步流程可以压缩成一张表：

| 步骤 | 数学对象 | 目的 |
| --- | --- | --- |
| 裁剪 | $g_i \rightarrow \tilde g_i$ | 限制单样本最大影响 |
| 聚合 | $\frac{1}{B}\sum_i \tilde g_i$ | 得到批次更新方向 |
| 加噪 | $+\mathcal N(0,\sigma^2 C^2 I)$ | 隐藏单样本痕迹 |
| 更新 | $\theta_{t+1}=\theta_t-\eta(\cdot)$ | 训练模型 |

这里最容易误解的一点是：噪声不是直接加到 loss 上，而是加到梯度更新上。所以它和 L2 正则化不同。L2 正则化是在目标函数里加惩罚项，仍然是确定性的；DP-SGD 的噪声是随机注入到更新过程，直接改变优化轨迹。

隐私预算为什么会累积？因为训练不是只做一步，而是做很多步。每一步都泄露一点点，很多步叠加后总泄露会上升。早期常用 Moments Accountant，后续工程上大量采用 RDP（Rényi Differential Privacy，白话说就是“用一族更容易组合的隐私度量先记账，再换算回 $(\varepsilon,\delta)$”）来跟踪总预算。实践里常见流程是：

1. 设定目标 $\delta$，通常取接近 $1/N$ 量级，$N$ 为训练集大小。
2. 固定采样率 $q=B/N$、步数 $T$、噪声系数 $\sigma$。
3. 用 Moments Accountant 或 RDP accountant 计算最终 $\varepsilon$。
4. 若 $\varepsilon$ 太大，则增大 $\sigma$、减小步数、减小采样率，或扩大数据量。

注意，隐私预算随训练轮数增加而上升，但不是简单的“每个 epoch 线性加一个固定 $\varepsilon$”。真实计算依赖采样策略和会计方法，必须让账本工具来算。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现，只演示“按样本裁剪 + 聚合 + 加噪”的核心逻辑，不依赖深度学习框架：

```python
import math
import random

def l2_norm(vec):
    return math.sqrt(sum(x * x for x in vec))

def clip_grad(g, C):
    norm = l2_norm(g)
    scale = max(1.0, norm / C)
    return [x / scale for x in g]

def dp_sgd_step(per_sample_grads, lr, C, sigma, seed=0):
    random.seed(seed)

    clipped = [clip_grad(g, C) for g in per_sample_grads]
    dim = len(clipped[0])
    batch_size = len(clipped)

    avg_grad = []
    for j in range(dim):
        avg_grad.append(sum(g[j] for g in clipped) / batch_size)

    # 高斯噪声：标准差为 sigma * C / B
    noise = [random.gauss(0.0, sigma * C / batch_size) for _ in range(dim)]
    noisy_grad = [avg_grad[j] + noise[j] for j in range(dim)]

    theta = [0.0] * dim
    new_theta = [theta[j] - lr * noisy_grad[j] for j in range(dim)]
    return clipped, avg_grad, noise, new_theta

grads = [
    [6.0, 8.0],  # 范数 10，会被裁剪到范数 5
    [1.0, 1.0],  # 不会被裁剪
]

clipped, avg_grad, noise, new_theta = dp_sgd_step(
    per_sample_grads=grads,
    lr=0.1,
    C=5.0,
    sigma=1.0,
    seed=42,
)

assert round(l2_norm(clipped[0]), 6) <= 5.0
assert clipped[1] == [1.0, 1.0]
assert len(new_theta) == 2
assert new_theta != [-0.1 * x for x in avg_grad]  # 加噪后结果不等于纯均值更新
```

这个例子对应的学习点很直接：

- 第一个样本梯度太大，被压到范数 5。
- 第二个样本保持不变。
- 最终更新不是简单平均梯度，而是“平均梯度 + 噪声”。

真实工程里通常不会手写梯度，而是依赖 PyTorch + Opacus 或 TensorFlow Privacy。高层伪代码如下：

```python
for batch in dataloader:
    per_sample_grads = compute_per_sample_grads(model, batch)

    clipped_grads = []
    for g in per_sample_grads:
        clipped_grads.append(clip_by_l2_norm(g, C))

    grad = average(clipped_grads)
    grad = grad + gaussian_noise(std=sigma * C / batch_size)

    optimizer.apply(grad)
    privacy_accountant.step(sample_rate=batch_size / dataset_size)
```

关键参数建议如下：

| 参数 | 含义 | 调参建议 |
| --- | --- | --- |
| $C$ | 裁剪阈值 | 先统计梯度范数分布，再取中位数或分位数附近做初值 |
| $\sigma$ | 噪声系数 | 先由目标 $\varepsilon,\delta$ 反推，再微调 |
| $B$ | batch size | 越大越稳，但采样率和显存也受影响 |
| $\varepsilon$ | 总隐私预算 | 生产中常在 1 到 10 之间权衡 |
| $\delta$ | 失效概率 | 常取 $10^{-5}$ 或接近 $1/N$ |

一个真实工程例子是大模型训练。FlashDP 的思路不是改变 DP 定义，而是优化 per-layer DP-SGD 的执行方式：把逐层梯度计算、裁剪、聚合尽量融合，减少显存搬运和重复计算。公开结果显示，它在 4 张 A100 上预训练 Llama-13B 时，不增加额外显存，吞吐可维持在非 DP 训练的大约 90%。这说明 DP-SGD 的瓶颈常常不是“理论不可做”，而是“朴素实现太慢太耗显存”。

---

## 工程权衡与常见坑

DP-SGD 的第一类成本是效用成本，也就是精度损失。CIFAR-10 这类标准任务上，非隐私训练可能到 95% 以上，而 $\varepsilon=1,\delta\approx10^{-5}$ 的强隐私配置下，准确率常掉到 60% 到 70%。这不是参数没调好就一定能消失的差距，而是隐私约束真实存在的代价。

第二类成本是系统成本。要做 per-sample gradient，显存和计算都会上涨；模型越大，这部分越痛。普通训练只需要 batch 平均梯度，DP-SGD 却要先知道每个样本的梯度，再裁剪、再聚合。

常见坑可以直接列出来：

| 坑 | 现象 | 规避方式 |
| --- | --- | --- |
| $C$ 设太小 | 梯度长期被压扁，欠拟合明显 | 先观测梯度范数分布，必要时做自适应裁剪 |
| $C$ 设太大 | 单样本敏感度大，必须加更多噪声 | 不要把裁剪当形式步骤 |
| $\sigma$ 太大 | loss 抖动大，收敛极慢 | 结合更大数据量、更长 warmup、更稳优化器 |
| 只看单步隐私 | 训练后发现总 $\varepsilon$ 爆掉 | 全程接入 accountant 实时记账 |
| 把 DP 当正则化替代品 | 误以为“有噪声就等于泛化更好” | 明确 DP 的目标是隐私，不是泛化提升 |
| 忽略采样策略 | 账本与实现不匹配 | 明确是否是 Poisson sampling 或固定 batch sampling |

还要区分“噪声像正则化”与“噪声就是正则化”。前者是现象层面相似，后者在机制上不对。L1/L2 正则通过修改损失函数塑造参数偏好；DP-SGD 通过约束单样本敏感度并注入随机机制获得隐私保证。它们的目标、分析工具、预算约束都不同。

如果把视角放到大模型，工程策略通常有三条：

1. 做 per-layer 或分组裁剪，降低全量 per-sample 梯度存储压力。
2. 融合裁剪与反向计算，减少显存读写。
3. 用 RDP/PLD 账本持续跟踪预算，而不是训练完再补算。

---

## 替代方案与适用边界

DP-SGD 不是唯一隐私方案，只是“训练阶段保护单样本贡献”最标准的一种。

| 方法 | 核心思路 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- | --- |
| DP-SGD | 裁剪 + 加噪训练 | 定义严格、工具成熟 | 精度和算力代价高 | 直接训练私有模型 |
| PATE | 多教师投票 + 私有聚合 + 蒸馏 | 对学生模型训练友好 | 需要教师划分和公共无标注数据 | 分类任务、可拆分教师数据 |
| 加密推理 | 在推理时对输入/计算做加密 | 保护查询数据 | 不保护训练集成员隐私 | 高敏感在线推理 |
| 输出后处理 | 只在发布结果时加噪 | 成本低 | 不能替代训练时 DP | 聚合统计或报表发布 |

PATE（Private Aggregation of Teacher Ensembles，白话说就是“让多个教师模型分别学私有子集，再把投票结果加噪后教给学生模型”）和 DP-SGD 的差别很大。PATE 的隐私主要发生在教师投票和知识蒸馏阶段，更适合分类、且最好有可用公共数据；DP-SGD 则直接作用于优化器，更通用，但更吃算力。

DP-SGD 的适用边界也很明确：

适用条件：

- 数据量足够大，因为更多样本能摊薄噪声影响
- 任务能接受一定精度下降
- 梯度范数分布相对稳定，便于选择 $C$
- 系统有余量承担 per-sample gradient 计算

不太适用的条件：

- 小数据集且精度要求极高
- 训练预算非常紧，无法承受明显吞吐下降
- 梯度极不稳定，固定裁剪阈值长期失真
- 需要保护的是推理请求，而不是训练样本

对于大模型，FlashDP 这类实现优化扩展了“可训练”的边界，但没有改变 DP-SGD 的基本代价结构。它解决的是显存和吞吐问题，不是免费消除隐私-效用折中。

---

## 参考资料

[1] Scientific Reports, *Tighter privacy auditing of differentially private stochastic gradient descent in the hidden state threat model*  
页面：https://www.nature.com/articles/s41598-026-38537-0  
说明：给出 $(\varepsilon,\delta)$-DP 定义、DP-SGD 公式与威胁模型表述。

[2] Emergent Mind, *DP-SGD: Differential Privacy in ML*  
页面：https://www.emergentmind.com/topics/differential-privacy-dp-sgd  
说明：总结 DP-SGD 的裁剪、加噪、采样放大、RDP/PLD 等机制。

[3] Emergent Mind, *Moments Accountant in Differential Privacy*  
页面：https://www.emergentmind.com/topics/moments-accountant  
说明：解释 Moments Accountant 的记账思想及其在 DP-SGD 中的作用。

[4] System Overflow, *Training ML Models with Differential Privacy (DP-SGD)*  
页面：https://www.systemoverflow.com/learn/ml-privacy-fairness/differential-privacy/training-ml-models-with-differential-privacy-dp-sgd  
说明：给出面向初学者的 DP-SGD 直观解释，以及 CIFAR-10 上隐私-精度折中示例。

[5] OpenReview, *Private Training Large-scale Models with Efficient DP-SGD*  
页面：https://openreview.net/forum?id=b6SWqFEOSF  
说明：FlashDP 的工程结果，覆盖 per-layer DP-SGD、显存与吞吐优化、Llama-13B 实验。

[6] Springer Artificial Intelligence Review, *Defending against attacks in deep learning with differential privacy: a survey*  
页面：https://link.springer.com/article/10.1007/s10462-025-11350-3  
说明：综述 DP 在深度学习中的攻击面、防御方式、动态裁剪和常见工程问题。
