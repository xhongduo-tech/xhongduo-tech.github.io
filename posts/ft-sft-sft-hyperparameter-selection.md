## 核心结论

指令微调，英文常写作 SFT（Supervised Fine-Tuning，意思是“拿人工整理好的问答对继续训练模型”），最核心的超参矛盾只有两个：一是模型要学得动，二是不要把底座模型原本的能力洗掉。

经验上，可以先记住一组保守起点：

| 方案 | 常见学习率起点 | 常见有效 batch | 常见 epoch | 典型目标 |
|---|---:|---:|---:|---|
| 全量 SFT | $1\times10^{-5}$ 到 $2\times10^{-5}$ | 64 到 256 | 1 到 3 | 在较大范围内改写模型行为 |
| LoRA / PEFT | $1\times10^{-4}$ 到 $3\times10^{-4}$ | 64 到 256 | 1 到 3 | 用较低成本学习风格、格式、领域偏好 |

这里的有效 batch，意思是“真正参与一次参数更新的样本总数”。如果显存不够，通常不直接把单卡 batch 开大，而是用梯度累积来模拟。

对初学者，最稳的起手式不是“找最优超参”，而是先用保守配置跑通，再看验证集。一个实用模板是：`micro batch=1`、`gradient accumulation=64`、全量 SFT 学习率从 `1e-5` 起、训练 `1~2 epoch`、warmup 设为较长的前期热身。如果每个 epoch 后验证损失稳定下降，说明方向对；如果验证损失抖动或反弹，先把学习率降 50%，再考虑别的改动。

高学习率最早暴露出来的问题，往往不是训练直接崩，而是输出风格变得僵硬、重复增多、拒答异常。epoch 太多则更典型：训练 loss 继续下降，但验证 loss 开始上升，说明模型正在记训练集细节，泛化能力反而变差。

---

## 问题定义与边界

这篇文章讨论的是“指令微调阶段的超参怎么选”，不是预训练，也不是 RLHF。目标很明确：在尽量保留原始模型能力的前提下，让模型学会一组新的回答偏好，比如更规范的客服话术、更稳定的输出格式、某个垂直领域的解释方式。

这里有三个边界要先说清楚。

第一，训练数据是“筛选过的指令-回复对”。这类数据不是随便抓来的文本，而是明确告诉模型“遇到这种输入，应该输出这种回答”的样本。换句话说，SFT 学的是偏好和映射，不是从零学语言。

第二，显存通常是硬约束。很多人看到推荐的 batch 是 64 或 128，会误以为单卡就要塞这么多样本。实际工程里更常见的是：

$$
\text{effective batch}=\text{per-device batch size}\times \text{gradient accumulation steps}\times \text{device count}
$$

如果只用 1 张卡、每步只能放 1 条样本，但你累积 64 步再更新一次，那么有效 batch 仍然是 64。

第三，判断是否“调好了”，主要看验证集，不看训练集。训练集 loss 降低只能说明模型在记住训练数据；验证集 loss 是否稳定，才说明它对没见过的数据还有泛化能力。

先看一个玩具例子。

假设你有 64 条“解释过拟合”的问答对。如果你的 `per_device_train_batch_size=16`，`gradient_accumulation_steps=4`，那么模型会先用 16 条做前向和反向，但不立即更新参数；连续做 4 次后，把累计到的梯度合起来再更新一次。这相当于“用 64 条样本的平均意见”去改参数，比拿 1 条或 2 条样本就立刻更新更稳定。

很多新手把问题理解成“怎么把 loss 压到最低”，这不对。SFT 更准确的目标是：让模型在新任务上更听话，同时别把原来的通用能力破坏掉。超参的作用，就是控制“改动幅度”和“改动速度”。

如果把验证曲线用文字描述，大致有三种典型形态：

| 曲线表现 | 含义 | 常见原因 |
|---|---|---|
| train loss 降，val loss 也缓慢降 | 训练正常 | 学习率和 batch 基本合理 |
| train loss 抖动大，val loss 同样不稳 | 更新过猛 | 学习率偏大、有效 batch 太小 |
| train loss 持续降，val loss 先降后升 | 过拟合 | epoch 太多、数据太少或太窄 |

---

## 核心机制与推导

先看学习率。学习率，白话说就是“每次更新参数时，步子迈多大”。步子太小，模型学不动；步子太大，模型会跳过稳定区域，把底座已经学好的东西冲掉。

在最简化的梯度下降里，参数更新可以写成：

$$
\theta_{t+1}=\theta_t-\eta \nabla L(\theta_t)
$$

其中 $\theta$ 是模型参数，$\eta$ 是学习率，$\nabla L(\theta_t)$ 是当前梯度。这个式子只表达一件事：梯度告诉你朝哪个方向改，学习率决定改多少。

为什么有效 batch 会影响学习率选择？因为小 batch 的梯度噪声更大。梯度噪声，意思是“这一步算出来的更新方向里，随机波动很多”。如果 batch 很小，又配一个很大的学习率，就容易朝着偶然噪声猛冲，表现成 loss 抖动、输出风格突然变怪。

所以通常有一个朴素规律：有效 batch 越大，梯度越平滑，学习率的可用范围就越稳定；有效 batch 太小，则需要更保守的学习率。

再看 epoch。epoch 的意思是“整个训练集被完整看过几遍”。SFT 常见设置是 1 到 3，不是因为不能跑更多，而是因为很多指令数据集相对窄，模型很快就会记住训练集格式。训练集 loss 会继续下降，但那并不代表模型更聪明，只代表它更会背答案。

过拟合可以用一个很简单的判据来观察：

$$
\text{如果 } epoch \uparrow \text{ 且 } val\_loss \uparrow,\text{ 则 generalization } \downarrow
$$

这里的 generalization，意思是“模型在没见过样本上的表现能力”。一旦验证集开始变差，再继续训练，通常是在消耗底座能力换训练集拟合。

可以把这件事理解成一个玩具例子。你教模型回答“如何重置密码”，训练集里所有样本都写成三步固定模板。模型在第 1 个 epoch 学会了任务；第 3 个 epoch 开始，它学到的可能不是“如何解释重置密码”，而是“任何类似问题都要照抄那种三段模板”。这时训练 loss 还会下降，但模型的表达弹性已经变差了。

真实工程里，这个现象常出现在客服、流程审批、结构化回复等窄任务上。因为数据风格高度统一，模型很容易学到“固定腔调”。如果学习率再偏高，就会进一步放大这种风格覆盖，最终表现成拒答率上升、重复句式变多、对开放问题的回答僵硬。

warmup 也重要。warmup，意思是“训练一开始先用很小的学习率，慢慢升到目标学习率”。原因很直接：刚开始梯度分布还不稳定，如果第一步就用目标学习率猛冲，最容易把参数带偏。对大模型 SFT，前期热身不是装饰，而是稳定性的组成部分。

---

## 代码实现

下面先用一个最小 Python 例子把“有效 batch”和“验证集早停”讲清楚。代码是可运行的，不依赖深度学习框架。

```python
def effective_batch(per_device_batch_size, grad_accum_steps, num_devices=1):
    return per_device_batch_size * grad_accum_steps * num_devices

def should_early_stop(val_losses, patience=1):
    """
    如果最近连续 `patience` 次没有刷新最优验证损失，则触发早停。
    """
    best = float("inf")
    bad_rounds = 0
    for loss in val_losses:
        if loss < best:
            best = loss
            bad_rounds = 0
        else:
            bad_rounds += 1
        if bad_rounds > patience:
            return True
    return False

# 玩具例子：单卡 micro batch=1，累积 64 步，相当于有效 batch 64
assert effective_batch(1, 64, 1) == 64

# 验证集先降后升，说明可能过拟合
val_losses = [1.20, 1.05, 0.98, 1.01, 1.06]
assert should_early_stop(val_losses, patience=1) is True

# 验证集持续改善，不应早停
better_val_losses = [1.20, 1.10, 1.02, 0.97, 0.95]
assert should_early_stop(better_val_losses, patience=1) is False
```

在真正训练里，可以把配置写成下面这样。这里用的是接近 Hugging Face `Trainer` 的伪代码风格，重点是参数关系，不是某个特定框架的完整脚本。

```python
train_config = {
    "learning_rate": 1e-5,
    "warmup_steps": 30000,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 64,
    "max_epochs": 2,
    "lr_scheduler_type": "constant",
    "weight_decay": 0.1,
    "logging_steps": 10,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
}

effective_batch = (
    train_config["per_device_train_batch_size"]
    * train_config["gradient_accumulation_steps"]
)

assert effective_batch == 64

for epoch in range(train_config["max_epochs"]):
    train_one_epoch()
    val_loss = evaluate_on_validation_set()
    refusal_rate = evaluate_refusal_rate()
    repetition_rate = evaluate_repetition_rate()

    print({
        "epoch": epoch + 1,
        "val_loss": val_loss,
        "refusal_rate": refusal_rate,
        "repetition_rate": repetition_rate,
    })

    if should_stop_by_val_loss(val_loss):
        break
```

如果做的是 LoRA，配置会多出一组“低秩适配”参数。LoRA，白话说就是“不改全部参数，只加一小块可训练增量”。它的优点是省显存、省算力，也更不容易把底座整体洗掉。

```python
lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

assert lora_config["r"] in (16, 32)
assert "q_proj" in lora_config["target_modules"]
```

参数可以先按下面这张表记：

| 参数 | 作用 | 新手保守起点 | 风险信号 |
|---|---|---|---|
| `learning_rate` | 控制每次更新幅度 | 全量 `1e-5`，LoRA `1e-4` | loss 抖动、重复、拒答变多 |
| `warmup_steps` | 前期慢启动 | 大模型可设较长热身 | 一开始就震荡 |
| `per_device_train_batch_size` | 单卡每步样本数 | 1 或 2 | 显存爆掉 |
| `gradient_accumulation_steps` | 累积多步再更新 | 补足到 64 或 128 | 太小会导致梯度噪声大 |
| `max_epochs` | 全量看数据几遍 | 1 到 2 起步 | 过多会过拟合 |
| `lr_scheduler_type` | 学习率变化策略 | `constant` 或带 warmup | 调度过复杂不利于排查 |

一个真实工程例子是客服助手微调。假设你要把通用模型改成电商售后机器人，目标是学会统一礼貌话术、赔付流程、退换货条件。这类任务通常不需要改底座的语言能力，只需要学“回复风格”和“业务规则映射”。这时更推荐从 LoRA 起手：`r=16` 或 `32`，学习率从 `1e-4` 左右试，epoch 先设 `2`，每个 epoch 后同时观察 `val_loss`、拒答率、重复率，而不是只盯着训练 loss。

---

## 工程权衡与常见坑

SFT 超参选择里，最常见的坑不是“完全学不会”，而是“表面上学会了，实际把模型带偏了”。

第一类坑是学习率过高。它的早期信号通常是：

| 现象 | 更可能的原因 | 处理策略 |
|---|---|---|
| 训练 loss 大幅抖动 | 学习率过高 | 先降 50% |
| 回答更模板化、更僵硬 | 更新过猛，底座被覆盖 | 降学习率，缩短 epoch |
| 重复输出增多 | 梯度不稳或过拟合 | 降学习率，检查数据重复 |
| 拒答率异常升高 | 模型偏到保守模式 | 降学习率，加验证监控 |

很多人会问，为什么高学习率会让拒答变多？因为模型不是只学“新任务”，而是在重新调整一部分已有分布。步子太大时，它可能把原本细腻的语言模式压扁，收缩成少数高概率、保守、重复的回答模板。

第二类坑是 epoch 太多。这是最典型、也最容易被误判的错误。训练 loss 一直降，很容易让人误以为模型越来越好；但验证集上升时，真正发生的是模型开始背训练集。尤其当数据集规模不大、风格很统一时，1 到 2 个 epoch 可能已经足够。

第三类坑是只看 loss，不看行为指标。对于指令微调，行为指标很重要，例如：

```python
metrics = {
    "val_loss": 0.92,
    "refusal_rate": 0.08,
    "repetition_rate": 0.04,
}

assert 0.0 <= metrics["refusal_rate"] <= 1.0
assert 0.0 <= metrics["repetition_rate"] <= 1.0
```

`refusal_rate` 可以理解成“明明应该回答，却拒绝回答的比例”；`repetition_rate` 可以理解成“输出中明显重复片段的比例”。这两个指标在高学习率或过拟合阶段，经常比 loss 更早报警。

第四类坑是把 gradient accumulation 当成“纯显存技巧”。它当然能省显存，但更重要的是稳定梯度。如果显存只够 `batch_size=1`，那就不要硬顶着小 batch 配大学习率。更稳的做法是把 `gradient_accumulation_steps` 拉高，让有效 batch 回到 64 或 128，再用保守学习率训练。

对新手，一个很实用的经验法则是：如果你发现验证损失开始跳、生成变僵、拒答率上升，不要同时改 5 个参数。先把学习率降一半。如果问题仍在，再考虑减少 epoch、增加 dropout、或检查数据质量。超参排查最怕多变量同时变动，因为你会失去因果判断。

---

## 替代方案与适用边界

不是所有需求都该用全量 SFT。很多场景，LoRA 或其他 PEFT（Parameter-Efficient Fine-Tuning，意思是“只训练少量参数的微调方法”）更合适。

先看一个对比。

| 方案 | 适用场景 | 常见超参起点 | 主要风险 |
|---|---|---|---|
| 全量 SFT | 需要较深改写模型行为，或任务迁移幅度大 | `lr=1e-5~2e-5`，`epoch=1~2` | 容易洗掉底座，成本高 |
| LoRA / PEFT | 主要改风格、格式、领域偏好 | `lr=1e-4~3e-4`，`r=16/32`，`epoch=1~3` | 改动深度有限，模块选错效果差 |

如果你的需求只是“把回复风格改成客服口吻”“把输出改成 JSON”“让模型学会几个固定业务流程”，通常优先选 LoRA。原因不是它永远更好，而是它的改动范围更受控，调参窗口也更宽。

一个新手友好的选择是：LoRA 用 `rank=16`，学习率从 `3e-4` 或更保守的 `1e-4` 起，epoch 设 `2`，先覆盖注意力主投影层。如果验证集稳定、业务指标也改善，就没必要急着上全量微调。

什么时候从 LoRA 切到全量 SFT？通常有三种信号：

| 信号 | 说明 |
|---|---|
| LoRA 已经稳定收敛，但目标能力仍学不进去 | 任务改动深度超过了少量增量参数的承载范围 |
| 需要系统性改变模型知识组织方式 | 比如较强的领域迁移或新语言适配 |
| 目标效果对底层表示改写要求更高 | 只改投影层不够，需要更多层一起调整 |

但即使决定做全量 SFT，也不意味着可以把学习率开大。恰恰相反，为了保护底座，全量训练通常要比 LoRA 更保守。一个常见的工程判断是：如果你非常在意保留原有能力，那就宁可从更低学习率、更少 epoch 起步，再根据验证集逐步放开。

最后再给一个真实工程判断例子。假设你做企业知识库问答，只要求模型“按公司术语解释问题”，并不要求它获得新的通用语言能力。这类任务优先 LoRA；如果你发现 LoRA 无法稳定学会专业术语之间的映射关系，或者回答虽然口吻对了，但核心内容还是错，那才考虑切到全量 SFT，并把学习率压到 `1e-5` 量级，用更严的验证集监控来保护底座能力。

---

## 参考资料

| 来源 | 内容焦点 |
|---|---|
| Kinda Technical, Advanced Generative AI SFT | 解释 SFT 的目标、标准训练方式，以及学习率、epoch、batch 的基本作用 |
| NVIDIA Nemotron SFT 文档 | 给出可直接参考的工程配置，如 global batch 64、warmup 30k、1 到 2 epoch |
| Unsloth LoRA Hyperparameters Guide | 总结 LoRA 常见学习率范围与调参经验 |
| AWS Nova SFT 文档 | 提供面向工程实践的 LoRA 起始值与训练建议 |
| Swiftorial 关于微调失败分析 | 总结高学习率、过拟合、输出退化等常见失败模式 |

1. Kinda Technical: Advanced Generative AI, Supervised Fine-Tuning: Turning Base Models into Assistants  
   https://kindatechnical.com/advanced-llm-topics/lesson-21-supervised-fine-tuning-turning-base-models-into-assistants.html

2. NVIDIA Nemotron Documentation, SFT  
   https://docs.nvidia.com/nemotron/nightly/nemotron/super3/sft.html

3. Unsloth Docs, LoRA Hyperparameters Guide  
   https://docs.unsloth.ai/get-started/fine-tuning-guide/lora-hyperparameters-guide

4. AWS Nova Documentation, Fine-tune with SFT  
   https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-sft-2-fine-tune.html

5. Swiftorial, Why Fine-Tuned Model Fails  
   https://www.swiftorial.com/articles/llm-fine-tuning/why-fine-tuned-model-fails/
