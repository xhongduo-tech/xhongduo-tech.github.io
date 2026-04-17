## 核心结论

学习率调度策略，本质上是定义训练过程中学习率 $\eta_t$ 随时间 $t$ 变化的函数。白话说，它决定优化器每一步“走多大”。这个函数不是装饰项，而是训练能否稳定、能否收敛到更好结果的核心控制杆。

第一，学习率调度解决的是“前期要快、后期要稳”的矛盾。固定学习率经常只能在两者之间二选一：设大了，前期下降快但容易震荡；设小了，训练稳定但收敛太慢。调度把这两个阶段拆开处理，让前期大步靠近可行区域，后期小步精细搜索。

第二，Warmup–Stable–Decay 是现在大模型训练里非常实用的三段式曲线。warmup 是热身，白话说就是先别一下踩满油门；stable 是稳定高位阶段，让模型在较大步长下高效学习；decay 是衰减阶段，逐渐减小步长以降低震荡并榨出最终性能。

第三，常见调度策略里，`constant` 最简单但通常不够稳，`step decay` 易实现但不平滑，`exponential decay` 连续但衰减速度不易直观控制，`cosine annealing` 在工程里最常见，因为它平滑，通常比生硬的阶梯下降更不容易引入 loss 抖动。大模型尤其常见的是 `warmup + cosine` 或 `warmup + stable + cosine`。

一个新手可落地的例子：训练 ResNet 做图像分类，基础学习率起点设为 $10^{-2}$，前 5 个 epoch 线性 warmup 到 $10^{-1}$，然后进入 cosine 衰减直到 $10^{-4}$。通常会看到训练 loss 在前期快速下降，到了中后期逐渐变稳，验证集指标也更容易继续上涨，而不是反复抖动。

下表先给出不同策略在训练早期和后期的节奏差异：

| 策略 | 早期节奏 | 后期节奏 | 平滑性 | 典型用途 |
|---|---|---|---|---|
| Constant | 一直固定 | 一直固定 | 高，但无阶段控制 | 小模型、基线实验 |
| Step Decay | 前期不变，到节点突降 | 后期分段变小 | 低，拐点生硬 | 传统 CNN、资源紧张场景 |
| Exponential Decay | 持续缓慢下降 | 后期越来越小 | 中 | 想要连续下降但实现简单 |
| Cosine Annealing | 前期下降较慢 | 后期平滑贴近最小值 | 高 | 通用深度学习训练 |
| Warmup + Cosine | 先上升，再平滑下降 | 后期平稳细化 | 很高 | Transformer、较大 batch |
| Transformer 公式 | 先快速升温 | 反平方根下降，尾部较慢 | 高 | 原始 Transformer 训练 |

---

## 问题定义与边界

学习率调度的数学定义可以写成：

$$
\eta_t = f(t; \theta)
$$

其中 $t$ 表示训练时间，可以是 `step`，也可以是 `epoch`；$\theta$ 表示调度参数，比如 warmup 长度、衰减系数、最小学习率等。白话说，就是“给训练步数输入一个数字，输出当前该用多大的学习率”。

这件事的目标不是让曲线好看，而是解决两个具体问题：

1. 训练初期参数还没进入稳定区域，如果学习率过大，容易出现梯度爆炸。梯度爆炸，白话说就是参数更新幅度大到失控，loss 会突然变成很大，甚至直接 `nan`。
2. 训练后期如果学习率还维持很大，模型会在较优区域附近来回跳，收敛不了；如果太早把学习率降很低，又会过早进入“走不动”的状态。

所以一个完整调度函数至少要明确几个边界条件：

| 边界条件 | 含义 | 常见设置方式 |
|---|---|---|
| $t_{\text{warmup}}$ | 热身区间长度 | 总 step 的 1% 到 10% |
| $t_{\text{stable}}$ | 稳定高位阶段长度 | 大模型常设一个 plateau |
| $\eta_{\max}$ | 峰值学习率 | 由 batch size、模型规模决定 |
| $\eta_{\min}$ | 末端学习率 | 常设为 $\eta_{\max}$ 的 1% 到 10% |
| $T$ | 总衰减长度 | 通常等于总训练步数减去 warmup/stable |

一个常见误区是把学习率调度当成“锦上添花”。对 GPT-like 预训练来说，这不是锦上添花，而是基础设施。假设你直接用固定大学习率启动训练，比如 AdamW 配上较大 batch，然后一开始就用目标峰值学习率。对新手可以这样理解：这像汽车刚从停车场起步，还没进入车道，就直接切到高速巡航。模型还没有形成稳定激活和梯度尺度，更新会非常剧烈，于是出现 loss 尖峰、梯度范数飙升、混合精度溢出，严重时训练直接中断。

因此本文讨论的边界是：以监督训练和预训练中最常见的按 step 调度为主，重点覆盖 constant、step、exponential、cosine、warmup + cosine，以及 Transformer 原始公式。像 ReduceLROnPlateau 这种依赖验证集反馈的策略，本文只在替代方案里简要提及，不作为主线。

---

## 核心机制与推导

先统一记号。设初始基准学习率为 $\eta_0$，当前步数为 $t$。

### 1. Constant

$$
\eta_t = \eta_0
$$

这是最简单的策略，每一步都一样。优点是实现成本最低，缺点是无法同时兼顾快和稳。

### 2. Step Decay

$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}
$$

其中 $s$ 是衰减间隔，$\gamma < 1$ 是每次衰减倍率。白话说，就是每隔固定步数“砍一刀”。它的缺点是拐点不连续，loss 曲线有时会在节点附近出现可见抖动。

### 3. Exponential Decay

$$
\eta_t = \eta_0 \cdot e^{-kt}
$$

这里 $k$ 控制衰减速度。它比 step 更平滑，但工程上不如 cosine 直观，因为很难直接从公式看出“训练到末尾还剩多少学习率”。

### 4. Cosine Annealing

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \frac{t}{T}\right)\right)
$$

余弦退火，白话说是让学习率沿半个余弦曲线平滑下降。它的一个重要特点是前期下降较慢，中后期下降更明显，所以常常兼顾训练速度和后期精调。

### 5. Warmup

单独看 warmup，常见线性形式为：

$$
\eta_t = \eta_{\max}\cdot \frac{t}{t_{\text{warmup}}}, \quad 1 \le t \le t_{\text{warmup}}
$$

它不是完整调度，而是完整调度的前置阶段。目的不是提升最终上限，而是避免训练一开始就使用过大的更新幅度。

### 6. Warmup + Cosine

这是最常见的实用组合之一：

$$
\eta_t=
\begin{cases}
\eta_{\max}\cdot \frac{t}{t_{\text{warmup}}}, & t \le t_{\text{warmup}} \\
\eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\left(\pi\frac{t-t_{\text{warmup}}}{T-t_{\text{warmup}}}\right)\right), & t > t_{\text{warmup}}
\end{cases}
$$

玩具例子：设 $\eta_{\max}=3\times10^{-4}$，$\eta_{\min}=1\times10^{-4}$，$t_{\text{warmup}}=5$，总步数 $T=20$。那么前 5 步学习率分别是 $0.6, 1.2, 1.8, 2.4, 3.0 \times 10^{-4}$；第 6 步开始，不再突然下砍，而是平滑过渡到余弦下降。这个“先升后降”的过程，就是很多训练日志里稳定 loss 曲线的来源。

### 7. Transformer 原始公式

原始 Transformer 使用的是尺度感知调度：

$$
\eta_t = d_{\text{model}}^{-0.5}\cdot \min(t^{-0.5},\ t\cdot t_{\text{warmup}}^{-1.5})
$$

这里 $d_{\text{model}}$ 是模型隐藏维度。尺度感知，白话说就是学习率会根据模型宽度自动缩放。这个公式的含义分两段：

1. 在 warmup 阶段，$\eta_t \propto t$，线性上升。
2. 过了 warmup，$\eta_t \propto t^{-0.5}$，按反平方根衰减。

它和普通 warmup + cosine 的区别在于：后半段不是余弦，而是反平方根；并且全局乘了 $d_{\text{model}}^{-0.5}$，让不同宽度模型的更新量更可比。

下面用同一组步数做一个直观对比。设 $\eta_0=1.0$，step 衰减参数为 $\gamma=0.5, s=5$，指数衰减参数为 $k=0.1$，cosine 的 $\eta_{\min}=0.1,\eta_{\max}=1.0,T=20$，warmup 长度为 5。

| step | Constant | Step | Exponential | Cosine | Warmup+Cosine |
|---|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 0.905 | 0.994 | 0.200 |
| 5 | 1.000 | 0.500 | 0.607 | 0.868 | 1.000 |
| 10 | 1.000 | 0.250 | 0.368 | 0.550 | 0.775 |
| 15 | 1.000 | 0.125 | 0.223 | 0.232 | 0.325 |
| 20 | 1.000 | 0.062 | 0.135 | 0.100 | 0.100 |

这个表能看出两个关键差异。第一，warmup + cosine 在最开始并不激进，它先慢慢爬到峰值；第二，cosine 类曲线后期更平滑，不像 step 那样突然掉台阶。

真实工程例子：预训练一个 7B 级别 Transformer，常见做法不是“从第 1 步就用最大学习率”，而是前 1% 到 3% step 做 warmup，中间保留高位 plateau，然后把剩余大部分训练预算交给 cosine 或线性 decay。原因很直接：大 batch、混合精度、长序列三者叠加时，训练初期的数值稳定性最差，而调度正是在这里承担“限速器”的角色。

---

## 代码实现

工程上最常见的实现思路是把学习率写成一个按 `step` 输入的函数，再交给调度器调用。PyTorch 中常用 `LambdaLR` 完成这件事。

先给一个可运行的纯 Python 版本，便于理解阶段逻辑：

```python
import math

def warmup_cosine_lr(step, warmup, total_steps, max_lr, min_lr):
    assert total_steps > warmup >= 1
    assert max_lr >= min_lr >= 0.0
    assert 1 <= step <= total_steps

    if step <= warmup:
        return max_lr * step / warmup

    progress = (step - warmup) / (total_steps - warmup)
    cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cos_decay

warmup = 5
total_steps = 20
max_lr = 3e-4
min_lr = 1e-4

lrs = [warmup_cosine_lr(s, warmup, total_steps, max_lr, min_lr) for s in range(1, total_steps + 1)]

assert abs(lrs[0] - 6e-5) < 1e-12
assert abs(lrs[4] - 3e-4) < 1e-12
assert lrs[5] < lrs[4]
assert abs(lrs[-1] - min_lr) < 1e-12
assert max(lrs) == lrs[4]
```

这段代码体现了两个实现原则：

1. 调度最好以 `step` 为单位，而不是以 `epoch` 为单位。因为现代训练里梯度累积、分布式并行、动态长度 batch 很常见，真正稳定的时钟往往是 optimizer step。
2. warmup、stable、decay 最好拆成阶段函数，而不是把所有逻辑揉成一团。这样便于做实验，也便于排查错误。

下面给一个更接近训练脚本的写法：

```python
import math

def lr_fn(step, warmup, stable, total_steps, max_lr, min_lr):
    if step <= warmup:
        return max_lr * step / warmup

    if step <= warmup + stable:
        return max_lr

    decay_steps = total_steps - warmup - stable
    decay_progress = (step - warmup - stable) / decay_steps
    cos_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
    return min_lr + (max_lr - min_lr) * cos_decay
```

如果要和 PyTorch 绑定，`LambdaLR` 需要的是一个“倍率函数”，通常写法是返回相对于优化器初始学习率的比例，而不是直接返回绝对学习率。因此常见实现是：

```python
import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

model = torch.nn.Linear(8, 2)
base_lr = 3e-4
optimizer = AdamW(model.parameters(), lr=base_lr)

warmup = 1000
total_steps = 10000
min_ratio = 0.1

def lr_lambda(step):
    step = max(step, 1)
    if step <= warmup:
        return step / warmup

    progress = (step - warmup) / (total_steps - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_ratio + (1 - min_ratio) * cosine

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
```

如果要复现 Transformer 原始方案，可以写成：

```python
def transformer_lambda(step, warmup, d_model):
    step = max(step, 1)
    scale = d_model ** (-0.5)
    return scale * min(step ** (-0.5), step * (warmup ** -1.5))
```

这里要注意一个实现细节：`LambdaLR` 的输入 `step` 通常从 0 开始，所以代码里常用 `max(step, 1)`，否则会出现 $0^{-0.5}$ 这种非法计算。

真实工程里还要处理两件事。第一，分布式训练下只应在真正执行 `optimizer.step()` 后更新调度器，不能按 dataloader iteration 盲目前进，否则 warmup 长度会被梯度累积放大。第二，断点恢复时必须同时恢复 optimizer 和 scheduler 状态，否则训练会在错误的学习率位置继续跑。

---

## 工程权衡与常见坑

学习率调度不是“选一个流行曲线”就结束，真正的难点在于曲线与训练系统其他部分耦合得很深，比如 batch size、优化器、梯度裁剪、混合精度和总训练步数。

先看一个典型工程现象。假设你设计了 `warmup 5 步 -> stable 50 步 -> cosine 衰减到 0.5\eta_{\max}`。这时训练曲线通常比较平滑，因为模型在第 6 到第 55 步之间有足够时间在峰值学习率附近吸收信息。如果你把 plateau 缩短到 10 步，衰减会过早发生，模型还没充分利用高学习率的探索能力，就被迫进入保守更新区间。结果往往是衰减初期 loss 波动增大，或者验证指标进入平台期过早。

下表列出常见坑和对应对策：

| 问题 | 典型现象 | 原因 | 对策 |
|---|---|---|---|
| 缺少 warmup | 训练前几百步 loss 爆炸、`nan` | 初期更新过猛 | 增加 warmup，并配合 gradient clipping |
| step 衰减不平滑 | 衰减节点前后 loss 抖动 | 学习率跳变过大 | 改用 cosine 或更密集的小步衰减 |
| stable 期太短 | 很快进入保守更新，收敛慢 | 高学习率利用不足 | 延长 plateau，尤其是大数据训练 |
| 衰减太早 | 训练后半程几乎不学习 | 总步数估计错误 | 按真实 optimizer step 重算 schedule |
| 末端 LR 太高 | 验证集指标停滞，无法精修 | 尾部步长过大 | 降低 $\eta_{\min}$ |
| 末端 LR 太低 | 训练后期过慢 | 学习率提前冻结 | 提高 final ratio 或缩短 decay |
| 断点恢复后曲线异常 | 恢复训练后 loss 突变 | scheduler 状态没同步 | 保存并恢复 scheduler state_dict |

还有两个初学者很容易踩的坑。

第一，把 `epoch` 当作唯一时间单位。对于小数据集问题不大，但在大规模训练里，真正决定参数更新的是 optimizer step。比如你用了梯度累积 8 次才更新一次参数，如果 warmup 仍按 iteration 算，就等于把设计的 warmup 放大了 8 倍。

第二，把优化器自适应能力当成调度的替代。AdamW 确实会根据一阶、二阶统计量调整每个参数的有效更新，但它并不会替你决定整个训练过程该在什么时候大胆、什么时候保守。也就是说，自适应优化器解决的是“不同参数该怎么走”，学习率调度解决的是“整个系统当前该走多快”，两者不是互斥关系。

---

## 替代方案与适用边界

不是所有任务都要用同一条曲线。学习率调度的选择应当跟数据规模、模型大小、总训练预算一起看。

先给一个场景对照表：

| 适用场景 | 推荐策略 |
|---|---|
| 小模型、快速基线 | Constant 或 Step Decay |
| 传统 CNN 图像分类 | Step Decay 或 Cosine |
| 中大型 Transformer 微调 | Warmup + Linear/Cosine Decay |
| 超大模型预训练 | Warmup–Stable–Decay + 较小 final LR |
| 数据少、epoch 多、易过拟合 | Cosine Annealing with Restarts |
| 资源紧张、实现要求极简 | Step Decay |
| 指标波动大、需要平滑收敛 | Warmup + Cosine |

对于小数据集，尤其是会训练很多 epoch 的场景，周期性 cosine 或余弦重启常常比单调衰减更合适。重启，白话说就是学习率下降到较低位置后，再周期性抬高一点，让模型重新探索新的区域。这对容易陷入局部最优的小数据任务很有帮助。

微调 BERT 是一个典型新手场景。数据量不大、训练步数有限，这时通常用 `warmup + linear decay` 就够了，原因是实现简单、可控性强、社区经验也足够多。相反，如果是大规模事实 QA 预训练，数据极多、总 step 很长，用 warmup–stable–decay 往往更合理，因为你需要一个明显的高学习率平台来提升整体训练效率。

再看边界。下面这些情况不适合盲目套用复杂调度：

1. 训练总步数极少。比如只训练几百步，这时复杂的 warmup、plateau、cosine 三段式可能还没展开就结束了，不如直接 warmup + 短线性衰减。
2. 模型本身很小、任务也不难。如果 constant 已经稳定收敛，继续堆复杂 schedule 的边际收益可能很低。
3. 你没有可靠的总步数估计。cosine、linear 这类依赖训练总长度的策略，如果总步数经常变化，调度曲线就会失真。这时要么动态重算，要么选对总长度不那么敏感的方案。

最后要强调一点：没有“全场最佳”的单一学习率曲线。更准确的说法是，不同任务、不同训练阶段、不同系统约束下，有不同的最优折中。工程实践里，最稳妥的起点通常是：

1. Transformer 微调：`warmup + linear decay`
2. 大模型预训练：`warmup + stable + cosine/linear decay`
3. 传统视觉任务：`cosine` 或里程碑式 `step decay`

---

## 参考资料

- Learning Rate Scheduling Strategies（artificial-intelligence-wiki）：学习率调度作用、常见曲线与基本公式综述。  
  https://artificial-intelligence-wiki.com/ai-development/training-and-fine-tuning/learning-rate-scheduling-strategies/

- Annotated Transformer（deepwiki）：Transformer 原始学习率公式及 `LambdaLR` 风格实现说明。  
  https://deepwiki.com/harvardnlp/annotated-transformer/3.2-learning-rate-scheduling

- Why Cosine Annealing + Warmup Stabilizes Training（gaohongnan）：从分阶段公式和数值例子解释 warmup + cosine 的稳定性来源。  
  https://www.gaohongnan.com/playbook/training/why_cosine_annealing_warmup_stabilize_training.html

- Warmup-Stable-Decay Pattern（emergentmind）：大模型训练中三阶段调度的经验模式与适用场景。  
  https://www.emergentmind.com/topics/warmup-stable-decay-wsd-pattern
