## 核心结论

微调是在已有大模型上继续训练，让模型更适合特定任务。核心分歧只有一个：是更新全部参数，还是只更新少量附加参数。

结论先给出：

1. 如果目标是“尽快做出可用版本”，优先选参数高效微调（PEFT，Parameter-Efficient Fine-Tuning，白话解释：只改一小部分参数，而不是重训整模型）。
2. 如果硬件只有单张消费级 GPU，例如 RTX 4090 24GB，实际可行方案通常是 LoRA 或 QLoRA，而不是全参数微调。
3. 如果任务要求极高、数据量大、预算足，且希望模型内部能力也发生明显改变，才考虑全参数微调。
4. 对大多数 7B 级模型，数据质量通常比“再调一次学习率”更重要。去重、统一格式、减少噪声，收益往往高于盲目堆数据。
5. LoRA/QLoRA 的常见质量可以达到全参数微调的大部分效果，但不是无损替代。工程上常见判断是：先用 PEFT 验证上限，不够再考虑更贵方案。

下面先用一张表把主流方案放在一起看：

| 策略 | 更新范围 | 典型显存需求 | 训练成本 | 质量上限 | 适合谁 |
|---|---:|---:|---:|---:|---|
| 全参数微调 | 全部权重 | 高，7B 常需多卡或 100GB 级资源 | 最高 | 最高 | 大团队、长周期项目 |
| LoRA | 少量低秩矩阵 | 中低，7B 可在十几到几十 GB | 低 | 较高 | 大多数团队 |
| QLoRA | 4-bit 基座 + LoRA | 更低，单卡 24GB 常可运行 | 最低 | 接近 LoRA | 单卡/低预算 |
| Adapter | 插入小模块 | 中 | 中 | 中高 | 需要模块化管理 |
| Prefix Tuning | 只学前缀向量 | 低 | 低 | 中 | 输入模式固定任务 |
| P-Tuning | 学连续提示向量 | 低 | 低 | 中 | 轻量试验场景 |

一个直观理解是：全参数微调像把整本书重写一遍；LoRA 像只在关键页贴便签；QLoRA 则是先把书压缩存放，再贴便签。这个比喻不精确，但足够帮助新手理解资源差异。

---

## 问题定义与边界

“微调”指的是：拿一个已经预训练好的模型，例如 7B 或 13B 语言模型，再用你的任务数据继续训练。

这里要先分清两个边界。

第一，更新哪些参数。

| 方式 | 会修改什么 | 不会修改什么 |
|---|---|---|
| 全参数微调 | 原模型全部权重 | 无 |
| PEFT | 新增的小参数模块 | 原模型主权重基本冻结 |

第二，解决什么问题。

微调适合把模型往“某种稳定输出风格或稳定任务能力”上推，例如客服回复、工单分类、结构化抽取、企业术语风格统一。它不适合替代检索系统，也不适合把模型“记住所有最新知识”。如果需求本质是“查最新文档并回答”，优先考虑 RAG（检索增强生成，白话解释：先查资料再回答）而不是先微调。

再看硬件边界。显存决定策略上限：

| 模型规模 | 全参数微调最低建议 | LoRA 最低建议 | QLoRA 最低建议 |
|---|---|---|---|
| 7B | 多卡或 80GB 级别以上 | 16GB+ | 12GB~24GB |
| 13B | 多卡 80GB 级别 | 24GB+ | 24GB~48GB |
| 35B | 多卡 H100/A100 | 多卡 | 多卡 |
| 70B+ | 多卡集群 | 多卡集群 | 多卡集群 |

玩具例子：你要训练一个“把用户评价改写成标准情感标签”的模型，数据只有 3000 条，输出空间很窄。这时全参数微调通常不划算，因为模型本体语言能力已经足够，你只需要把输出格式和任务偏好教给它。

真实工程例子：公司要做客服助手，输入是“用户问题 + 历史工单摘要”，输出要符合公司话术、退款规则、风险提示模板。这种任务常用 7B 模型配合 QLoRA，在单卡上训练一个业务 adapter，再按业务线切换多个 adapter 部署。

---

## 核心机制与推导

LoRA 的核心思想是：不直接更新原始权重矩阵 $W$，而是学习一个低秩更新量 $\Delta W$。

公式是：

$$
W' = W + \Delta W,\quad \Delta W = BA
$$

其中：

- $W \in \mathbb{R}^{d \times k}$ 是原始权重
- $B \in \mathbb{R}^{d \times r}$
- $A \in \mathbb{R}^{r \times k}$
- $r$ 是 rank，白话解释：压缩后的中间维度，通常远小于 $d$ 和 $k$

这样训练参数量从原本的 $d \times k$，变成：

$$
|B| + |A| = dr + rk = r(d+k)
$$

如果某层同时按两组矩阵计，工程上常写成近似：

$$
\text{PEFT参数量} \approx 2r(d+k)
$$

这就是 LoRA 为什么省显存。你不再为整个大矩阵保存完整梯度和优化器状态，只为小矩阵保存。

举一个玩具数值例子。假设某层权重大小是 $4096 \times 4096$：

- 全参数更新：约 $1677$ 万参数
- 如果 LoRA 取 $r=16$：
  $$
  2r(d+k)=2 \times 16 \times (4096+4096)=262144
  $$
  只有约 26 万参数

差距接近两个数量级。

QLoRA 再向前一步。它先把基础模型量化成 4-bit 权重，再在量化权重之上叠加 LoRA。量化，白话解释，就是把权重用更少比特存储，从而减少显存占用。QLoRA 的关键价值不是“数学上更强”，而是“同样的方法能在更小显存里跑起来”。

训练时还要理解一个公式：

$$
\text{Effective Batch Size} = \text{per\_device\_train\_batch\_size} \times \text{gradient\_accumulation\_steps}
$$

如果单卡显存只能放下 batch size = 2，你可以用梯度累积把有效 batch 做到 16：

$$
2 \times 8 = 16
$$

这会直接影响梯度稳定性。

LoRA 常见超参可以用下面的表理解：

| 超参 | 作用 | 取大了会怎样 | 取小了会怎样 | 常见起点 |
|---|---|---|---|---|
| `r` | 低秩维度 | 显存和训练参数上升 | 表达能力不足 | 8, 16, 32 |
| `alpha` | 缩放系数 | 更新幅度过强，训练不稳 | 更新太弱 | 常设为 `r` 或 `2r` |
| `dropout` | 防过拟合 | 太大可能欠拟合 | 太小可能记忆训练集 | 0 或 0.05 |
| `target_modules` | 作用层选择 | 覆盖过多更耗资源 | 覆盖过少效果有限 | `q_proj/v_proj` 起步 |

很多实践里会看到“`alpha = r` 或 `alpha = 2r`”。本质上它是控制 LoRA 更新量的放大倍数，不是越大越好。

再说推理延迟。Prefix Tuning、P-Tuning、Adapter 这类方案通常在推理图里保留额外结构，因此更容易带来额外延迟。LoRA 的优势之一是训练后可 merge，也就是把增量权重并回主模型。合并后推理路径和普通模型一致，额外延迟接近没有。

---

## 代码实现

下面给一个最小可运行的 Python 玩具实现，不依赖深度学习框架，只演示 LoRA 参数量为何更少，并验证有效 batch 的计算。

```python
def lora_param_count(d: int, k: int, r: int, two_mats: bool = True) -> int:
    base = r * (d + k)
    return 2 * base if two_mats else base

def full_param_count(d: int, k: int) -> int:
    return d * k

def effective_batch(batch_size: int, grad_accum: int) -> int:
    return batch_size * grad_accum

# 玩具例子：4096 x 4096 的线性层
d, k, r = 4096, 4096, 16
full = full_param_count(d, k)
lora = lora_param_count(d, k, r)

assert full == 4096 * 4096
assert lora == 2 * 16 * (4096 + 4096)
assert lora < full
assert effective_batch(2, 8) == 16

ratio = lora / full
print("full params:", full)
print("lora params:", lora)
print("ratio:", round(ratio, 4))
print("effective batch:", effective_batch(2, 8))
```

如果进入真实训练，主流代码结构大致是这样：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

model_name = "meta-llama/Llama-3.1-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=200,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
model.save_pretrained("./adapter_ckpt")
```

这段代码里的关键点只有三个：

1. `BitsAndBytesConfig` 把底座模型按 4-bit 加载，降低显存。
2. `LoraConfig` 指定 LoRA 的 rank、alpha、dropout 和作用层。
3. `per_device_train_batch_size=2` 与 `gradient_accumulation_steps=8` 组合，得到有效 batch 16。

真实工程例子可以写成下面这种数据格式。推荐统一成指令微调常见的三段式：

| instruction | input | output |
|---|---|---|
| 根据客服规范回答用户问题 | 用户说商品已签收但包装破损 | 很抱歉给您带来不便，麻烦您提供外包装照片... |
| 判断是否需要升级人工 | 用户连续三次要求退款并提到投诉 | 该问题涉及高风险投诉，建议立即转人工处理 |

保存成 JSONL 后，每一行一个样本，便于流式读取、去重和抽样。

训练完成后，一般有两种部署路径：

1. 保留 adapter：底座模型不变，按业务线加载不同 adapter。
2. 合并 adapter：使用 `merge_and_unload()` 把 LoRA 权重并入主模型，适合单一任务部署。

如果一个公司同时有“客服回复”“法务摘要”“商品标题标准化”三个任务，保留多个 adapter 往往比维护三套全参数模型更省存储和运维成本。

---

## 工程权衡与常见坑

真正影响效果的，通常不是“LoRA 还是 QLoRA”，而是“数据到底干不干净”。

先看数据侧常见问题：

| 问题 | 表现 | 后果 | 处理方式 |
|---|---|---|---|
| 重复样本过多 | 训练 loss 降很快 | 过拟合，验证效果差 | 去重、降采样 |
| 标注风格不一致 | 同类问题不同写法 | 模型输出摇摆 | 统一模板 |
| 输出过长过散 | 答案结构不稳定 | 难以收敛 | 收紧格式 |
| 噪声数据混入 | 事实错误或空样本 | 学到错误模式 | 过滤 |
| 类别不平衡 | 少数任务总答不好 | 偏向头部样本 | 重采样或补数 |

建议把数据先整理成固定结构，比如 `instruction/input/output`。这样做的价值不是“更学术”，而是让模型更容易分清任务说明、上下文和目标答案。

数据量没有统一阈值，但可以按工程经验理解：

| 阶段 | 样本量建议 | 目标 |
|---|---:|---|
| 可行性验证 | 500 ~ 2000 | 验证方向 |
| 小规模上线 | 2000 ~ 10000 | 稳定覆盖主场景 |
| 生产级优化 | 10000+ | 打磨长尾和鲁棒性 |

超参数上，新手最容易踩三个坑：

1. 学习率太高。LoRA 常见起点可以是 `2e-4`，不是所有任务都要更大。
2. 轮数太多。很多任务 1 到 3 轮已经够，再训只是在背答案。
3. 只看训练集 loss，不看验证集。训练 loss 降、验证 loss 升，往往说明已经过拟合。

下面这张表可直接当排查清单：

| 超参 | 常见起点 | 风险信号 | 调整方向 |
|---|---|---|---|
| Learning Rate | `2e-4` | loss 抖动大、发散 | 降到 `1e-4` 或更低 |
| Batch Size | 1~4 | 梯度噪声大 | 配合 grad accum |
| Grad Accum | 4~16 | 训练太慢 | 在显存允许下减少 |
| Epoch | 1~3 | 验证集变差 | 提前停止 |
| Max Length | 视任务而定 | 截断关键信息 | 检查样本长度分布 |

数据管道建议至少做这几步校验：

1. 去重：完全重复、近重复都要查。
2. 标注一致性检查：同类输入是否得到同类输出。
3. 长度分布检查：避免极少数超长样本拖慢训练。
4. 规则过滤：空样本、乱码、模板残留要清掉。
5. 难例补齐：对失败样本进行定向补数，而不是盲目扩容。
6. 合成数据谨慎使用：SMOTE 或模型生成样本只能补边角，不能替代人工高质量标注。
7. 评测集隔离：不要把训练样本混入验证或测试。

再说部署坑。Prefix、Adapter 方案虽然也省训练资源，但推理时可能保留额外模块；LoRA/QLoRA 的好处是训练完可 merge。如果上线形态是高并发 API，merge 后通常更稳、更简单。

---

## 替代方案与适用边界

PEFT 不是唯一方案，它只是“在资源受限时最常用”的那一类。

先把几个相关概念放在一起：

| 方法 | 解决什么问题 | 数据需求 | 算力需求 | 稳定性 | 适用场景 |
|---|---|---|---|---|---|
| SFT | 监督微调，教模型按样本作答 | 指令-答案对 | 中 | 高 | 基础任务适配 |
| PEFT | 用少量参数做 SFT 或任务适配 | 同 SFT | 低 | 高 | 低预算首选 |
| DPO | 直接偏好优化，学“哪个回答更好” | 偏好对数据 | 中 | 较高 | 对齐风格和偏好 |
| RLHF | 奖励模型 + 强化学习对齐 | 偏好数据更多 | 高 | 相对复杂 | 大规模对齐 |

SFT，白话解释，就是最基本的“给输入和标准答案，让模型照着学”。PEFT 通常是 SFT 的一种实现方式，不是互斥关系。你可以做“LoRA 版 SFT”。

DPO 和 RLHF 处理的是更后面的对齐问题。如果任务是“回答要更符合公司语气和人工偏好”，可以先用 SFT+LoRA 起步；如果发现模型会答对，但“答得不够像公司要的风格”，再考虑 DPO。RLHF 更重、更复杂，不是新手第一站。

什么时候 PEFT 不够？

1. 任务要求极高，并且小幅性能差都不能接受。
2. 数据量很大，希望模型底层表征也发生明显变化。
3. 需要持续学习多个阶段，且不能接受旧能力遗忘。
4. 目标模型很小，LoRA 增益空间有限，反而全参更直接。

一个现实决策路径通常是：

1. 先做 SFT + QLoRA，在单卡上验证是否有业务收益。
2. 如果收益明显，再扩数据、扩评测、调 rank 和 target modules。
3. 如果仍达不到质量要求，再评估全参数微调或后续 DPO。
4. 如果需求主要是知识更新，不先做微调，先补检索系统。

真实工程里，很多团队最终会保留“一个底座 + 多个 adapter”的结构。比如同一个 7B 模型上挂“客服 adapter”“摘要 adapter”“分类 adapter”，运行时按任务切换。这样比维护多套全参数模型更灵活，也更便于版本回滚。

---

## 参考资料

1. Introl, *Fine-Tuning Infrastructure: LoRA, QLoRA, and PEFT at Scale*，用于理解全参数微调、LoRA、QLoRA 的资源差异、部署方式和 merge 思路。
2. Zylos, *2026 LLM Fine-Tuning Techniques*，用于整理 2026 年常见微调方法、数据质量优先级、PEFT 与全参的适用边界。
3. Meta Intelligence, *Instruction Tuning Data Format Design*，用于指令数据格式设计，重点参考 `instruction/input/output` 的样本组织方式。
4. Unsloth Docs, *LoRA Hyperparameters Guide*，用于 LoRA 的 `r`、`alpha`、`dropout`、effective batch 等超参数经验。
5. Fireworks AI, *LLM Fine-Tuning Best Practices*，用于数据清洗、训练轮数、学习率和推理部署中的常见坑。
