## 核心结论

微调阶段的学习率调度，核心目标不是“让学习率变化得更花哨”，而是在不破坏预训练权重的前提下，把有限训练步数用在最有效的位置。对大多数语言模型微调任务，`warmup + cosine decay` 是默认优先方案：前期用 warmup 线性升高学习率，避免一开始更新过猛；中后期用余弦退火平滑下降学习率，让参数从“快速靠近可行区域”过渡到“细致收敛”。

warmup 的必要性在于：训练最开始几百步内，梯度方向通常不稳定。梯度是“参数应该往哪个方向改”的信号，如果此时直接使用较大学习率，更新会把预训练得到的表示结构迅速冲坏，尤其是在数据量小、分布偏移明显、只训练少量步数的微调任务里更明显。

余弦退火的价值在于：它不是突然降速，而是平滑减速。学习率越到后期越小，等价于优化器从“先大步找方向”切到“后小步找细节”。如果训练步数较长，还可以用 cycle 版本，也就是 SGDR，把余弦曲线周期性重置，在多个阶段重新探索更好的局部区域。

LR Range Test 的作用是先做一次低成本试验，快速估计“什么学习率区间值得用”。它不是正式训练，而是让学习率从极小值逐步增大，同时记录 loss 变化，再根据 loss 明显下降、震荡、反弹的位置推测峰值学习率和下界。

一个适合初学者的玩具理解是：模型像刚起跑的跑者。warmup 是热身，不先热身直接全速冲，肌肉容易拉伤；余弦退火是后程逐渐收步，不是突然刹停；LR Range Test 是赛前试跑，找哪种配速最稳。这个比喻只帮助理解直觉，真正决定效果的仍然是参数更新稳定性。

| 阶段 | 目标 | 典型步数范围 | 学习率变化 |
| --- | --- | --- | --- |
| Warmup | 保护预训练权重，稳定初始梯度 | 总步数的 1% 到 10% | 从 $L_0$ 线性升到 $L_{\max}$ |
| Decay | 提高后期收敛精度，减少震荡 | 剩余大部分训练步数 | 从 $L_{\max}$ 平滑降到 $L_{\min}$ |
| Cycle Cosine | 多轮探索与收敛 | 长训练或多阶段训练 | 每个周期重复“高到低”的余弦曲线 |

---

## 问题定义与边界

本文讨论的是“微调时如何设计学习率调度”，重点放在 LoRA、adapter、部分层解冻这类参数高效微调场景。LoRA 是“只训练少量低秩附加参数”的方法，白话说，就是不给整个大模型大改，只在旁边加一组更小、可训练的补丁矩阵。因为可训练参数和冻结参数的角色不同，所以学习率不一定应该相同。

问题的本质是平衡两个目标：

1. 稳定性：不要把预训练权重推离已有的好区域。
2. 速度：在有限 epoch 和有限 token 下尽快降低任务损失。

如果只追求速度，直接上大学习率往往会让 loss 一开始就剧烈波动；如果只追求稳定，学习率过小又可能几乎学不到任务特征。微调调度要解决的就是这个矛盾。

为什么“直接用大 LR”有风险？因为参数更新近似满足
$$
\theta_{t+1}=\theta_t-\eta_t g_t
$$
其中 $\theta_t$ 是参数，$g_t$ 是梯度，$\eta_t$ 是学习率。学习率可以理解为“每次沿梯度走多远”。在训练初期，$g_t$ 的方向还没有稳定下来，如果 $\eta_t$ 太大，更新步长就会远超当前局部几何结构允许的范围，导致模型偏离原本良好的预训练解。

这也是为什么很多人会感觉“大 LR 像把精致陶瓷猛地一晃”。这个说法只是帮助理解：真正含义是预训练权重已经编码了大量语言结构，初期粗暴更新会破坏这些结构，而不是在其上做受控调整。

LR Range Test 的基本想法可以写成：
$$
\eta_t=\eta_{\min}\left(\frac{\eta_{\max}}{\eta_{\min}}\right)^{t/N}
$$
其中 $N$ 是测试步数。也就是让学习率按指数形式持续增大，同时记录每一步的 loss。观察上通常会出现三段：

| 区间 | 现象 | 含义 |
| --- | --- | --- |
| 极小 LR 区间 | loss 几乎不动 | 学习率太小，更新无效 |
| 有效 LR 区间 | loss 持续下降 | 适合作为候选训练区间 |
| 过大 LR 区间 | loss 剧烈震荡或反弹 | 学习率过大，不稳定 |

本文不讨论完整预训练、不讨论复杂自适应 schedule 搜索，也不展开讨论 reinforcement learning 阶段的学习率控制，因为那些任务的梯度统计性质与标准监督微调不同。

---

## 核心机制与推导

warmup 最常见的是线性 warmup。设总步数为 $T$，warmup 步数为 $W$，起始学习率为 $L_0$，峰值学习率为 $L_{\max}$，则在 $t < W$ 时：
$$
\text{lr}(t)=L_0+(L_{\max}-L_0)\frac{t}{W}
$$

这条式子的意义很直接：每一步只增加一点点学习率，让优化器逐渐进入工作状态。这里的“逐渐”不是审美问题，而是控制参数更新范数的手段。因为真实更新量大约与 $\eta_t \|g_t\|$ 成正比，所以 warmup 实际上是在约束初始几步的扰动强度。

warmup 结束后，常用余弦退火：
$$
\text{lr}(t)=L_{\min}+\frac{L_{\max}-L_{\min}}{2}\left(1+\cos\frac{\pi(t-W)}{T-W}\right)
$$

这个公式有两个特点。第一，开始时接近 $L_{\max}$，结束时接近 $L_{\min}$。第二，下降过程平滑，没有线性衰减在分段点上那种“恒定速度下滑”的硬感。平滑的好处是后期 loss 接近平台时，不会因为学习率下降过于生硬而突然失去有效更新。

一个玩具例子：总共训练 1000 步，前 100 步 warmup，把学习率从 $10^{-6}$ 升到 $2\times10^{-4}$；剩余 900 步用余弦退火把它降回 $10^{-6}$。这意味着模型先用很小步幅试探，再在中段高效率学习，最后细致收尾。如果把这 1000 步都固定在 $2\times10^{-4}$，训练前段更容易出现 loss 抖动，训练后段又不够细。

循环余弦，也就是 SGDR，可以写成“把当前周期步数 $T_{\text{cur}}$ 代入余弦公式，并在每次重启时清零”。白话说，就是让学习率在多个周期里反复从高到低变化。它适合更长训练，或者你希望模型多次跳出当前收敛盆地时使用。

| 方案 | 关键参数 | 行为特征 | 适用场景 |
| --- | --- | --- | --- |
| 常规余弦 | $L_{\max},L_{\min},T,W$ | 只下降一次 | 常见微调、步数固定 |
| SGDR | $L_{\max},L_{\min},T_0,T_{\text{mult}}$ | 多个余弦周期，周期可变长 | 长训练、多阶段探索 |
| 线性衰减 | 起点、终点、总步数 | 线性下降 | 简单、可解释性强 |

LR Range Test 则提供“峰值学习率怎么定”的经验依据。常见做法不是选 loss 最低点本身，而是选“loss 开始明显恶化前”的学习率作为候选 $L_{\max}$。然后再把这个值除以 3 到 4，得到更保守的长期训练学习率下界或稳定起点。这不是严格定理，而是工程经验：范围测试的目标是找到边界，不是直接复用整个曲线。

真实工程例子是 LoRA 微调 7B 级模型。base model 是“原始主干参数”，adapter 是“额外插入的小模块参数”。很多团队会让 base 使用 $2\times10^{-5}$，adapter 使用 $2\times10^{-4}$ 左右，也就是高约 10 倍。原因是 adapter 从随机初始化或近随机状态开始，需要更快学习；base 已经有较强表示能力，只需轻微偏移。如果两者共用同一个很小 LR，adapter 学得太慢；如果共用同一个很大 LR，base 又容易不稳。

---

## 代码实现

下面先给一个可运行的玩具实现，展示“warmup 后接 cosine”的调度函数。它不依赖 PyTorch，只验证数学行为是否正确。

```python
import math

def warmup_cosine_lr(step, total_steps, warmup_steps, lr_init, lr_max, lr_min):
    assert total_steps > 0
    assert 0 <= warmup_steps < total_steps
    assert 0 <= step < total_steps
    assert lr_init <= lr_max
    assert lr_min <= lr_max

    if step < warmup_steps:
        return lr_init + (lr_max - lr_init) * step / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cosine

# 玩具例子：1000 步训练，100 步 warmup
total_steps = 1000
warmup_steps = 100
lr0 = 1e-6
lr_max = 2e-4
lr_min = 1e-6

assert abs(warmup_cosine_lr(0, total_steps, warmup_steps, lr0, lr_max, lr_min) - 1e-6) < 1e-12
assert abs(warmup_cosine_lr(100, total_steps, warmup_steps, lr0, lr_max, lr_min) - 2e-4) < 1e-12
assert warmup_cosine_lr(999, total_steps, warmup_steps, lr0, lr_max, lr_min) > lr_min
assert warmup_cosine_lr(999, total_steps, warmup_steps, lr0, lr_max, lr_min) < 1.1e-6
```

如果在 PyTorch 中实现，更常见的是把 warmup 和 cosine 合成一个 `LambdaLR`，或者先用 warmup scheduler，再切换到 `CosineAnnealingLR`。逻辑重点是：`t < W` 时走 warmup，之后进入 cosine。

```python
import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

model = torch.nn.Linear(8, 2)
optimizer = AdamW(model.parameters(), lr=2e-4)

total_steps = 1000
warmup_steps = 100
min_lr_ratio = 1e-6 / 2e-4  # L_min / L_max

def lr_lambda(current_step: int):
    if current_step < warmup_steps:
        return max(1e-6 / 2e-4, float(current_step) / float(max(1, warmup_steps)))

    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

scheduler = LambdaLR(optimizer, lr_lambda)

for step in range(total_steps):
    loss = model(torch.randn(4, 8)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

LR Range Test 的核心循环也很简单：每个 batch 后把学习率按指数增大，并记录 loss。

```python
import math

def lr_range(step, num_steps, lr_start, lr_end):
    ratio = lr_end / lr_start
    return lr_start * (ratio ** (step / num_steps))

loss_history = []
lr_history = []

for step, batch in enumerate(train_loader):
    lr = lr_range(step, num_steps=200, lr_start=1e-6, lr_end=1e-2)
    for group in optimizer.param_groups:
        group["lr"] = lr

    loss = train_one_batch(batch)
    loss_history.append(float(loss))
    lr_history.append(lr)

# 之后看 loss 随 lr 的曲线：
# 1. 明显下降区间 -> 候选有效区间
# 2. 剧烈反弹前 -> 候选 lr_max
# 3. 经验上可把峰值再除 3~4，得到更稳的训练值
```

LoRA 的差异化学习率通常通过参数组完成：

```python
optimizer = AdamW([
    {"params": base_params, "lr": 2e-5},
    {"params": lora_params, "lr": 2e-4},
], weight_decay=0.01)
```

这里的原则不是“LoRA 永远 10 倍”，而是“新增参数通常需要更激进，主干参数通常需要更保守”。

---

## 工程权衡与常见坑

第一类坑是跳过 warmup。很多失败案例不是模型架构不行，而是训练前几百步就把主干参数冲坏了。特别是在小数据集、高峰值学习率、混合精度和较大 batch 一起出现时，这个问题更明显。

第二类坑是 LoRA 和 base 共用同一个学习率。对 base 来说这可能过大，对 adapter 来说又可能过小。一个常见且实用的起点是：base 设为 `2e-5`，adapter 设为 `1e-4` 到 `2e-4`。如果发现 loss 降得慢、验证集不涨，先检查 adapter LR 是否太保守，而不是只怀疑数据或模型。

第三类坑是把 scheduler 当成独立旋钮。实际上 batch size、weight decay、gradient clipping 会共同影响稳定性。batch size 增大后，梯度方差通常减小，常让人误以为可以直接抬高峰值 LR，但微调数据分布窄时，这种放大并不总是安全。gradient clipping 是“限制梯度过大时直接截断”的手段，它能缓解异常 batch，但不能替代 warmup。

| 常见坑 | 典型现象 | 原因 | 规避措施 |
| --- | --- | --- | --- |
| 跳过 warmup | 前几百步 loss 抖动、发散 | 初始更新过猛 | 预留 1% 到 10% 步数做线性 warmup |
| adapter LR 过低 | 训练很稳但几乎没提升 | 新增参数学习太慢 | adapter 设为 base 的 5 到 10 倍 |
| base LR 过高 | 验证集快速恶化 | 预训练表示被破坏 | base 使用更保守 LR，必要时减少解冻层 |
| 衰减过快 | 后期 loss 早早停住 | 学习率太早接近 0 | 提高 $L_{\min}$ 或缩短 warmup 占比 |
| 只看训练 loss | 训练集下降，验证集无提升 | 过拟合或任务迁移失败 | 同时看验证曲线和梯度稳定性 |
| batch 太大直接抬 LR | 训练不稳定 | 微调场景不满足线性缩放假设 | 先做 LR range test，再逐步放大 |

一个真实工程上的经验是：如果你做指令微调，数据只有几十万到几百万 token，训练步数又不长，那么“温和 warmup + 保守 base LR + 稍高 adapter LR”往往比激进地追求大峰值更可靠。很多时候稳定收敛比理论上的最快收敛更重要，因为微调的预算主要浪费在失败重跑上。

---

## 替代方案与适用边界

余弦退火不是唯一方案。线性衰减、指数衰减、OneCycle 也都常见。选择标准主要看三件事：总训练步数是否很短、是否需要强探索、以及模型参数是否大部分冻结。

OneCycle 可以理解为“先升得更高，再降得更低”的单周期策略。它和 warmup+cosine 的区别在于：warmup+cosine 通常把 warmup 当成稳定阶段，峰值后开始持续衰减；OneCycle 则更强调中前期快速推到高学习率，再用长尾衰减完成收敛，常用于希望利用较大学习率获得更强正则化效应的场景。

| 调度方案 | 特点 | 适用场景 | 主要参数 |
| --- | --- | --- | --- |
| Warmup + Cosine | 稳定、平滑、通用 | 大多数 LLM 微调 | warmup 比例、$L_{\max}$、$L_{\min}$ |
| Warmup + Linear | 简单、易解释 | 短训练、基线实验 | warmup 步数、终点 LR |
| Exponential Decay | 衰减更固定 | 传统模型或旧系统兼容 | decay rate、decay steps |
| OneCycle | 先上升再下降，探索更强 | 中等步数、希望更快找到好区域 | max LR、pct_start |
| SGDR | 多轮周期重启 | 长训练、多阶段优化 | $T_0$、$T_{\text{mult}}$ |

适用边界也要明确：

1. 如果只训练几百步，复杂 cycle 往往不值得，简单 warmup + linear/cosine 就够了。
2. 如果 base 完全冻结，只训练 LoRA，小幅 warmup 仍然有意义，但核心应放在 adapter LR 选择。
3. 如果任务非常接近预训练分布，有时固定较小 LR 也能工作，但这通常是“任务足够简单”的结果，不是更优原则。
4. 对超大模型、长上下文、梯度噪声大的场景，可以考虑更慢的 ramp-up，或者在 plateau 检测后再额外降低学习率。

因此，实践中的默认顺序通常是：先做 LR Range Test 估计区间，再用 warmup + cosine 跑基线，最后只在“训练很长”或“明显需要更强探索”时考虑 SGDR 或 OneCycle。

---

## 参考资料

- Nano Language Models 的学习率调度文章：解释了 warmup、decay、微调稳定性的关系，适合作为整体直觉入口。链接：`https://nanolanguagemodels.com/2025/11/29/learning-rate-schedules-warmup-decay-why-they-matter/`
- PyTorch `CosineAnnealingLR` 文档：直接看 API 和参数定义最有效，适合落地实现。链接：`https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html`
- PyTorch `LambdaLR` 文档：适合把 warmup 和 cosine 拼成一个自定义调度器。链接：`https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html`
- Kobiso 关于 super-convergence 与 LR range test 的整理：有助于理解范围测试和 OneCycle 的背景。链接：`https://kobiso.github.io/research/research-super-convergence/`
- “LoRA Without Regret” 实践指南：关注 LoRA 微调时 base 与 adapter 的学习率差异。链接：`https://www.ikangai.com/lora-without-regret-a-practitioners-guide-to-reliable-fine-tuning/`
- Dataa.dev 关于 LoRA+ 的说明：讨论对不同参数组使用不同学习率的工程经验。链接：`https://dataa.dev/`
- Zhao Weiguo 的《Build LLM from Scratch》相关笔记：给出了 warmup、余弦调度和 LR 范围测试的公式化描述。链接：`https://knowledge.zhaoweiguo.com/build/html/x-learning/books/ais/2024/build_llm_from_scratch`
