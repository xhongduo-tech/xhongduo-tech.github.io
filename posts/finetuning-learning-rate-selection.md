## 核心结论

微调学习率是控制参数更新步长的超参数。它的目标不是单纯让训练更快，而是在“适应新任务”和“保留预训练知识”之间取得平衡。

全参数微调通常从 `1e-5 ~ 5e-5` 起步。LoRA 通常可以从 `1e-4 ~ 1e-3` 之间试起，但更推荐先从 `1e-4` 或 `2e-4` 这类温和数值开始。这里的范围是经验起点，不是硬规则，最终要看验证集损失、任务指标和人工检查结果。

同样是微调 7B 指令模型，如果做全参数微调，可以先试 `2e-5`；如果做 LoRA，可以先试 `2e-4`。前者更保守，因为所有模型参数都会被更新；后者更激进，因为只训练少量适配器参数。两者都不能只看训练集 loss，必须用验证集判断是否真的有效。

| 训练方式 | 典型学习率 | 风险 | 适用场景 |
|---|---:|---|---|
| 全参数微调 | `1e-5 ~ 5e-5` | 学习率过大会破坏预训练能力 | 数据质量高、算力充足、任务差异较大 |
| LoRA | `1e-4 ~ 1e-3` | 小数据或噪声数据上容易过拟合 | 常规指令微调、领域适配、低成本实验 |
| LoRA 保守配置 | `1e-4 ~ 2e-4` | 收敛可能稍慢 | 新手首轮实验、小数据集 |
| LoRA 激进配置 | `5e-4 ~ 1e-3` | 前期震荡、验证集变差 | 数据较多、较干净、验证机制可靠 |

---

## 问题定义与边界

本文讨论的“微调学习率”，指训练阶段控制参数更新幅度的超参数，通常记作 $\eta$ 或 `lr`。它不讨论预训练阶段的大规模优化策略，也不讨论推理阶段的 `temperature`、`top_p`、`top_k` 等采样参数。

微调是指在已有预训练模型基础上，用新数据继续训练，使模型适应某个任务或领域。学习率决定每一步参数变化有多大。步子太大，模型可能快速偏离原有能力；步子太小，训练成本增加，甚至学不到新任务信号。

需要先区分三类场景：

| 场景 | 本文是否覆盖 | 说明 |
|---|---|---|
| 全参数微调 | 覆盖 | 所有或大部分参数参与训练 |
| LoRA | 覆盖 | 冻结底座模型，只训练低秩适配器 |
| LoRA+ 等分组学习率方法 | 覆盖基本判断 | 不展开论文级推导 |
| 预训练优化策略 | 不覆盖 | 训练规模、目标和稳定性问题不同 |
| 分布式训练细节 | 不覆盖 | 如 ZeRO、FSDP、张量并行 |
| 优化器完整原理 | 不展开 | 只说明和学习率选择直接相关的部分 |

一个玩具例子：全参数微调可以直观理解为整辆车都在修，动作要轻；LoRA 像是在车上加一个小外挂模块，允许改动快一点。这只是帮助新手建立直觉，不是严格定义。真正判断仍然要看验证集。如果验证集变差，再好看的直觉也无效。

一个真实工程例子：团队要把 7B 指令模型微调成客服 FAQ 助手，数据有 3 万条，包含标准问答、历史工单摘要和拒答样例。全参数微调成本高、风险大，可以先用 `2e-5`；LoRA 可以先用 `2e-4`。如果前 200 step 验证损失明显抖动，优先降低学习率或增加 warmup，而不是直接增加 epoch。

---

## 核心机制与推导

全参数微调的基本更新公式是：

$$
\theta_{t+1} = \theta_t - \eta_t \nabla_\theta L(\theta_t)
$$

其中，$\theta_t$ 表示第 $t$ 步的模型参数，$L$ 表示损失函数，$\nabla_\theta L(\theta_t)$ 表示损失对参数的梯度，$\eta_t$ 表示第 $t$ 步学习率。梯度是告诉参数“往哪个方向改能让损失下降”的量，学习率则决定“沿这个方向走多远”。

如果 $\eta_t$ 太大，参数更新幅度过大，训练可能震荡，甚至 loss 爆炸。如果 $\eta_t$ 太小，模型变化太慢，训练集 loss 降不下来，或者要花很多 step 才能看到效果。

LoRA 的核心公式是：

$$
W = W_0 + \Delta W
$$

$$
\Delta W = BA
$$

其中，$W_0$ 是冻结的预训练权重，$\Delta W$ 是训练出来的增量权重，$A$ 和 $B$ 是低秩矩阵。低秩的意思是用更小的中间维度 $r$ 表达一个较大的矩阵变化，通常 $r \ll d$。训练时只更新 $A$ 和 $B$，不直接改动 $W_0$。

这就是 LoRA 可以使用更高学习率的主要原因：它更新的不是整个底座模型，而是少量适配器参数。即使适配器变化较快，对预训练权重本身的破坏也更间接。但这不表示 LoRA 可以无脑使用 `1e-3`。如果数据少、噪声大、rank 很低，过高学习率仍然会让适配器学到偏置和噪声。

常用学习率策略不是固定值，而是 warmup + cosine decay。Warmup 是预热，意思是在训练前期逐步把学习率从 0 增大到峰值，避免刚开始梯度不稳定时步子过大。Cosine decay 是余弦衰减，意思是在预热结束后让学习率平滑下降。

公式可以写成：

$$
\eta_t = \eta_{\max} \cdot \frac{t}{T_w}, \quad t \le T_w
$$

$$
\eta_t = \eta_{\max} \cdot 0.5 \cdot \left(1 + \cos\left(\pi \cdot \frac{t - T_w}{T - T_w}\right)\right), \quad t > T_w
$$

其中，$T$ 是总训练步数，$T_w$ 是 warmup 步数，$\eta_{\max}$ 是峰值学习率。

设总步数 $T=1000$，warmup 步数 $T_w=100$，LoRA 峰值学习率 $\eta_{\max}=2e-4$：

| 训练步数 | 阶段 | 学习率变化 |
|---:|---|---:|
| `t=0` | warmup 起点 | `0` |
| `t=50` | warmup 中间 | `1e-4` |
| `t=100` | 到达峰值 | `2e-4` |
| `t=550` | cosine 中段 | 约 `1e-4` |
| `t=1000` | 训练结束 | 约 `0` |

如果把同样的策略用于全参数微调，只需要把峰值学习率从 `2e-4` 换成 `2e-5`。曲线形状不变，但所有更新步长都缩小 10 倍。

---

## 代码实现

代码里不要只写一个 `learning_rate`。实际训练至少要同时配置优化器、scheduler、warmup 和总训练步数。优化器负责根据梯度更新参数，scheduler 负责在不同 step 改变学习率。

下面是一个可运行的 Python 玩具例子，用来验证 warmup + cosine 的学习率计算：

```python
import math

def warmup_cosine_lr(step, total_steps, warmup_steps, max_lr):
    assert total_steps > 0
    assert 0 <= warmup_steps < total_steps
    assert 0 <= step <= total_steps

    if step <= warmup_steps:
        return max_lr * step / warmup_steps if warmup_steps > 0 else max_lr

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))

T = 1000
Tw = 100
max_lr = 2e-4

assert warmup_cosine_lr(0, T, Tw, max_lr) == 0
assert abs(warmup_cosine_lr(50, T, Tw, max_lr) - 1e-4) < 1e-12
assert abs(warmup_cosine_lr(100, T, Tw, max_lr) - 2e-4) < 1e-12
assert abs(warmup_cosine_lr(550, T, Tw, max_lr) - 1e-4) < 1e-8
assert abs(warmup_cosine_lr(1000, T, Tw, max_lr) - 0.0) < 1e-12
```

Hugging Face 风格的全参数微调配置可以从下面这个起点开始：

```python
training_args_full = {
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
}
```

LoRA 配置通常把学习率提高一个数量级：

```python
training_args_lora = {
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
}

lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}
```

如果使用 `Trainer` 或 TRL 的 `SFTTrainer`，核心字段仍然是这几个：

| 配置项 | 全参微调起点 | LoRA 起点 | 作用 |
|---|---:|---:|---|
| `learning_rate` | `2e-5` | `2e-4` | 峰值学习率 |
| `lr_scheduler_type` | `cosine` | `cosine` | 学习率衰减方式 |
| `warmup_ratio` | `0.03 ~ 0.10` | `0.03 ~ 0.10` | 前期预热比例 |
| `max_grad_norm` | `1.0` | `1.0` | 梯度裁剪，降低异常更新 |
| `num_train_epochs` | `1 ~ 3` | `1 ~ 5` | 训练轮数，需看验证集 |

更进阶的 LoRA+ 会对 `A` 和 `B` 使用不同学习率。新手不需要第一轮就上 LoRA+，但要知道它不是“一个统一 lr 走到底”的思路。

---

## 工程权衡与常见坑

学习率不是孤立变量。它和 batch size、梯度累积、训练步数、warmup、decay、数据噪声、LoRA rank 都有关。只改 `learning_rate`，很容易把问题判断错。

Batch size 是每次参数更新看到的样本数量。batch size 越大，梯度通常越平滑，可以承受稍大的学习率；batch size 越小，梯度噪声更大，学习率过高时更容易震荡。梯度累积是把多次小 batch 的梯度累加后再更新一次参数，用来模拟更大的 batch size。

常见排查表如下：

| 现象 | 可能原因 | 优先处理手段 |
|---|---|---|
| 前 100 到 200 step loss 剧烈抖动 | 学习率过高或 warmup 太短 | 降低 `lr`，把 warmup 从 `3%` 提到 `5%~10%` |
| loss 爆炸或出现 `nan` | 学习率过大、梯度异常 | 降 `lr`、开启梯度裁剪、检查数据 |
| train loss 降，val loss 升 | 过拟合或学习率偏大 | 早停、降 `lr`、减少 epoch |
| train loss 几乎不降 | 学习率太小或数据格式有问题 | 先抽样检查数据，再适度增大 `lr` |
| LoRA 训练效果波动大 | rank 太低、学习率过高、数据噪声大 | 降 `lr`、提高 rank、清洗数据 |
| 全参微调后通用能力下降 | 破坏预训练知识 | 降 `lr`、减少训练步数、增加通用数据混合 |

一个真实工程排错流程：客服 FAQ 微调时，使用 LoRA、`lr=1e-3`、warmup `3%`。训练集 loss 很快下降，但验证集在 150 step 后开始上升，人工评测发现模型更爱编造答案。这时不要先加 epoch。更合理的顺序是：把 `lr` 降到 `2e-4`，warmup 提到 `5%` 或 `10%`，保留 cosine decay，然后重新看验证集。如果验证集仍然变差，再减少 epoch 或清洗高噪声样本。

另一个常见坑是把全参微调和 LoRA 的学习率经验值混用。全参直接用 `2e-4` 通常过大，尤其是数据量不大时，很容易让模型偏离原有能力。LoRA 直接用 `2e-5` 也可能太保守，训练很多 step 仍然学不到领域格式。

---

## 替代方案与适用边界

默认经验值不稳定时，可以考虑替代策略。它们不是为了显得复杂，而是解决不同模块、不同训练阶段对学习率敏感度不同的问题。

| 方案 | 适合场景 | 优点 | 代价 |
|---|---|---|---|
| 固定学习率 | 小型玩具实验 | 简单，容易复现 | 前期和后期都不够稳 |
| warmup + cosine | 常规微调首选 | 前期稳，后期平滑收敛 | 需要设置总步数和 warmup |
| layer-wise learning rate decay | 全参微调、模型层数较深 | 底层更保守，高层更灵活 | 配置更复杂 |
| 不同模块不同学习率 | 只想重点训练部分模块 | 控制更精细 | 需要理解模型结构 |
| LoRA+ 分组学习率 | LoRA 适配器训练不充分 | 可能提升收敛效率 | 多一个分组超参 |

Layer-wise learning rate decay 是分层学习率衰减，意思是靠近输入的底层用更小学习率，靠近输出的高层用更大学习率。直觉上，底层更偏通用表征，高层更贴近任务输出，所以底层改动要更保守。

适用边界可以简单判断：

| 数据情况 | 建议起点 |
|---|---|
| 几千条、噪声较高 | LoRA 从 `1e-4` 或 `2e-4` 开始，不要先冲 `1e-3` |
| 几万条、质量较好 | LoRA 可从 `2e-4` 开始，观察验证集 |
| 任务和底座差异很大 | 可适当增大学习率或训练步数，但必须监控验证集 |
| 输出格式要求严格 | 更重视验证集格式准确率，不只看 loss |
| 全参数微调 | 优先从 `1e-5 ~ 5e-5` 内选择 |

本文建议全参数微调先从 `1e-5 ~ 5e-5`，LoRA 先从 `1e-4` 量级试起，是工程经验起点，不是理论定律。数据越小、噪声越高、任务越敏感，越要保守；数据越多、越干净、任务差异越大，LoRA 适当提高学习率才更可能有效。

---

## 参考资料

1. [Hugging Face Transformers: Optimization and Schedules](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
2. [Hugging Face PEFT: LoRA Developer Guide](https://huggingface.co/docs/peft/main/en/developer_guides/lora)
3. [Hugging Face TRL: PEFT Integration](https://huggingface.co/docs/trl/main/peft_integration)
4. [LoRA: Low-Rank Adaptation of Large Language Models](https://huggingface.co/papers/2106.09685)
5. [LoRA+: Efficient Low Rank Adaptation of Large Models](https://huggingface.co/papers/2402.12354)
