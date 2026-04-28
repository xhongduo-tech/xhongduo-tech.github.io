## 核心结论

DeepSpeed ZeRO 是一种把优化器状态、梯度、参数按数据并行进程分片，而不是复制的显存优化方案。

它解决的核心问题不是“让训练计算更少”，而是“让训练时不必在每张 GPU 上重复存一整份训练状态”。对微调场景来说，最常见的第一瓶颈不是算力，而是显存。模型前向能算，不代表训练态能装下。

可以把传统数据并行理解成：4 张卡各自保留一整套教材。ZeRO 则是：4 张卡平均分管教材的不同章节，用到时再临时调回来。所以它节省的是存储冗余，不是把矩阵乘法变少。

| 方案 | 参数 `P` | 梯度 `G` | 优化器状态 `O` | 单卡显存特征 |
|---|---:|---:|---:|---|
| DP | 完整复制 | 完整复制 | 完整复制 | 最简单，但最耗显存 |
| ZeRO-1 | 完整复制 | 完整复制 | 分片 | 先省优化器状态 |
| ZeRO-2 | 完整复制 | 分片 | 分片 | 进一步省梯度 |
| ZeRO-3 | 分片 | 分片 | 分片 | 显存最省，但通信最复杂 |

如果目标是训练或微调超大基座模型，尤其是几十亿到上百亿参数模型，真正决定“能不能跑”的往往是 `ZeRO-3`，因为只有它连参数本身也分片。

---

## 问题定义与边界

先定义训练态显存。训练时显存不只装模型权重，还要装梯度和优化器状态。

- 参数 `P`：模型权重，也就是神经网络真正要学习的数值。
- 梯度 `G`：损失函数对参数的导数，白话说是“这一步该往哪个方向改参数”。
- 优化器状态 `O`：优化器为更新参数保存的历史量。以 Adam 为例，通常至少有一阶矩和二阶矩，显存占用常常比参数本身更大。

传统数据并行的问题是：每张卡都完整保存 `P + G + O`。这导致模型参数量一大，显存很快爆掉。

| 显存组成 | 作用 | 典型特点 |
|---|---|---|
| `P` 参数 | 前向与反向计算都要用 | 模型越大，占用越大 |
| `G` 梯度 | 反向传播后产生，用于更新 | 通常与参数规模同量级 |
| `O` 优化器状态 | 优化器保存的历史信息 | Adam 常明显大于参数本身 |

玩具例子：假设一个 `1B` 参数模型，使用 FP16 保存参数和梯度。粗略估算：

- 参数：`P = 2GB`
- 梯度：`G = 2GB`
- Adam 状态：`O = 8GB`

那么单卡训练态显存近似就是：

$$
M_0 = P + G + O = 2 + 2 + 8 = 12GB
$$

这还没算激活值、临时缓冲区、CUDA 内存碎片。也就是说，一个“看起来只有 1B 参数”的模型，在训练时远不止 2GB。

ZeRO 的边界也要讲清楚。它主要解决的是训练态显存，而不是所有系统瓶颈。

ZeRO 直接解决：
- 参数、梯度、优化器状态的重复存储
- 大模型微调时“单卡装不下训练态”的问题

ZeRO 不直接解决：
- GPU 算力不足
- 数据加载慢
- 网络带宽差
- 模型本身收敛不好
- 激活值过大导致的全部显存问题

所以 ZeRO 不是万能开关。它让模型“能放进去”，但不保证“训练一定更快”。

---

## 核心机制与推导

设数据并行卡数为 `N`。基线数据并行中，每张卡都持有完整的 `P + G + O`，所以：

$$
M_0 = P + G + O
$$

### ZeRO-1：只分片优化器状态

ZeRO-1 只把优化器状态分散到多张卡上，参数和梯度仍然每卡完整复制，因此：

$$
M_1 \approx P + G + O/N
$$

这一步的价值在于 Adam 状态通常很大，所以即使只分 `O`，收益也已经明显。

### ZeRO-2：再分片梯度

ZeRO-2 在 ZeRO-1 基础上继续分片梯度：

$$
M_2 \approx P + G/N + O/N
$$

此时参数仍然完整复制，所以如果模型太大，参数本身还是可能成为瓶颈。

### ZeRO-3：进一步分片参数

ZeRO-3 把参数也分片。每张卡平时只保留自己负责的那一部分参数；当前向或反向需要某一层时，再临时 all-gather，也就是“跨卡拼回完整参数”。

$$
M_3 \approx (P + G + O)/N + A_{peak}
$$

这里的 $A_{peak}$ 表示峰值额外开销，主要来自：
- 临时聚合参数时的缓存
- 激活值
- 通信过程中的额外缓冲

这就是 ZeRO-3 为什么最省显存，也最依赖通信。

| Stage | 分片对象 | 仍完整保留的对象 | 主要收益 | 额外代价 |
|---|---|---|---|---|
| ZeRO-1 | `O` | `P, G` | 先砍掉大块优化器显存 | 通信略增 |
| ZeRO-2 | `O, G` | `P` | 微调场景更常用 | 反向同步更复杂 |
| ZeRO-3 | `O, G, P` | 无完整常驻副本 | 最大幅度省显存 | all-gather 开销最高 |

玩具例子继续。假设 `N=4`，仍使用前面的 `1B` 参数模型：

- DP：`12GB`
- ZeRO-1：`2 + 2 + 8/4 = 6GB`
- ZeRO-2：`2 + 2/4 + 8/4 = 4.5GB`
- ZeRO-3：`(2 + 2 + 8)/4 = 3GB`，再加临时峰值开销

这组数字说明一个关键事实：显存下降的根源是“分摊持有”，不是“减少训练数学量”。

真实工程例子：`Llama-70B + LoRA + ZeRO-3`。这类场景里，LoRA 是低秩适配，白话说是“只额外训练一小组增量矩阵，而不全量更新基座权重”。LoRA 可训练参数很少，但基座模型很大，所以决定能否启动训练的，通常仍然是基座参数如何分片。此时 ZeRO-3 负责把大权重切开，LoRA 小参数则常保留本地，避免为很小的张量付出过多通信成本。

---

## 代码实现

ZeRO 在工程里通常不是重写训练算法，而是让 DeepSpeed 接管训练态管理。你重点要做三件事：

1. 指定 `zero_stage`
2. 在大模型初始化阶段避免先完整落到单卡
3. 明确 checkpoint 的保存与恢复方式

最小配置示意：

```json
{
  "zero_optimization": {
    "stage": 3,
    "zero3_init_flag": true,
    "zero3_save_16bit_model": true
  }
}
```

字段含义：
- `stage: 3`：启用 ZeRO-3，也就是参数、梯度、优化器状态都分片。
- `zero3_init_flag: true`：在模型构造阶段就按 ZeRO-3 方式处理，避免“模型还没分片就先 OOM”。
- `zero3_save_16bit_model: true`：需要导出普通 16-bit 权重时，按配置聚合保存，便于后续推理或发布。

最小接入流程：

| 步骤 | 要做什么 | 目的 |
|---|---|---|
| 加载模型 | 构造基础模型与 tokenizer | 建立训练对象 |
| 包装 DeepSpeed | 让 DeepSpeed 接管参数、梯度、优化器 | 开启 ZeRO |
| 启动训练 | 正常前向、反向、step | 不必自己手写分片逻辑 |
| 保存 checkpoint | 保存分片训练状态 | 用于继续训练 |
| 导出 16-bit 模型 | 聚合权重为常规模型文件 | 用于推理或分发 |

下面给一个可运行的 Python 玩具代码。它不依赖 DeepSpeed，只演示不同 Stage 的显存估算逻辑：

```python
def zero_memory(p_gb, g_gb, o_gb, n, stage, a_peak_gb=0.0):
    assert n >= 1
    assert stage in (0, 1, 2, 3)

    if stage == 0:
        m = p_gb + g_gb + o_gb
    elif stage == 1:
        m = p_gb + g_gb + o_gb / n
    elif stage == 2:
        m = p_gb + g_gb / n + o_gb / n
    else:
        m = (p_gb + g_gb + o_gb) / n + a_peak_gb
    return round(m, 4)

# 1B 参数模型的粗略估算
p, g, o, n = 2.0, 2.0, 8.0, 4

assert zero_memory(p, g, o, n, 0) == 12.0
assert zero_memory(p, g, o, n, 1) == 6.0
assert zero_memory(p, g, o, n, 2) == 4.5
assert zero_memory(p, g, o, n, 3, a_peak_gb=0.0) == 3.0

print("All assertions passed.")
```

如果接 Hugging Face + PEFT，训练代码的思路通常是：

```python
# 伪代码：Llama + LoRA + DeepSpeed ZeRO-3
model = load_base_model(...)
model = attach_lora_adapter(model, ...)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        deepspeed="ds_zero3_config.json",
        gradient_checkpointing=True,
        fp16=True,
        ...
    ),
    train_dataset=...,
)

trainer.train()
trainer.save_model()
```

重点不是训练循环写得多复杂，而是让配置和模型初始化路径正确。

---

## 工程权衡与常见坑

ZeRO 最容易出问题的地方通常不是公式，而是工程细节。

| 误区 | 后果 | 正确做法 |
|---|---|---|
| 以为 ZeRO-1/2 会分片参数 | 大模型仍然 OOM | 需要压参数显存时直接看 ZeRO-3 |
| 没开 `zero3_init_flag` | 模型构造阶段先爆显存 | 大模型初始化时优先开启 |
| 把分片 checkpoint 当普通模型文件读 | 恢复失败或权重不完整 | 按 DeepSpeed/HF 指南恢复或导出聚合模型 |
| 盲目启用 CPU/NVMe offload | 吞吐骤降，训练被带宽拖慢 | 先确认显存瓶颈，再评估带宽 |
| LoRA 小参数也强行过度分片 | 通信复杂度上升，收益很小 | 常见做法是只重点分片大基座权重 |

几个关键点单独展开。

第一，`ZeRO-1/2` 不分片参数。这是初学者最容易混淆的地方。若基座模型本身已经装不下，即使梯度和优化器状态省了很多，还是可能启动不了。

第二，`zero3_init_flag` 很重要。它解决的是“初始化时 OOM”，不是“训练中 OOM”。白话说，房子还没分配房间，你先把所有大件家具堆到一个房间里，自然会爆。

第三，checkpoint 要区分“继续训练”和“导出推理模型”。ZeRO-3 的训练 checkpoint 常常是分片状态，适合恢复训练；但如果你想拿到普通 FP16 模型做推理，需要按框架提供的流程把权重聚合导出。

第四，offload 不是免费午餐。offload 是把部分状态放到 CPU 或 NVMe，换取 GPU 显存空间。但总线和磁盘带宽远慢于 GPU 显存，所以它本质是“用时间换空间”。

第五，LoRA 与 ZeRO 结合时，不要默认“所有参数都分得越细越好”。LoRA 适配器本身参数量小，很多真实工程会保留它们本地化，只对基础模型的大权重做分片。原因很简单：几 MB 或几百 MB 的小参数，省不了多少显存，却可能带来额外同步成本。

---

## 替代方案与适用边界

不是所有微调都必须上 ZeRO-3。如果模型本来就能放进单卡或普通数据并行，直接上最复杂的方案通常不划算。

| 方案 | 主要省什么 | 代价 | 适合场景 |
|---|---|---|---|
| DP | 不省显存 | 实现最简单 | 中小模型、资源充足 |
| ZeRO-2 | 省梯度和优化器状态 | 通信增加 | 参数还能放下，但训练态偏大 |
| ZeRO-3 | 连参数也省 | 通信与 checkpoint 更复杂 | 大基座模型微调 |
| FSDP | 参数分片训练 | 框架配置复杂，策略选择多 | PyTorch 原生分片场景 |
| Gradient Checkpointing | 省激活值 | 增加重算时间 | 激活占用高的模型 |

几个常见判断准则：

- 如果中小模型单卡就能微调，先开混合精度和梯度检查点，通常更简单。
- 如果参数能放下，但 Adam 状态和梯度太大，`ZeRO-2` 往往已经够用。
- 如果基座模型参数本身就放不下，`ZeRO-3` 或 FSDP 才是主选项。
- 如果瓶颈主要是网络通信而不是显存，强行上 ZeRO-3 可能得不偿失。

FSDP 可以理解为 PyTorch 原生的全分片数据并行方案，目标和 ZeRO-3 类似，都是把参数训练态拆开分摊。选型时更多看团队栈、现有训练框架、checkpoint 生态和调试经验，而不是只看“理论上谁更省”。

结论可以压缩成一句话：当显存是主要约束时，ZeRO 尤其是 ZeRO-3 非常有价值；当显存不是主要约束时，复杂度就是它的成本。

---

## 参考资料

1. [DeepSpeed ZeRO 官方文档](https://deepspeed.readthedocs.io/en/stable/zero3.html)
2. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://www.microsoft.com/en-us/research/publication/zero-memory-optimizations-toward-training-trillion-parameter-models/)
3. [DeepSpeed 博客：训练 100B+ 参数模型](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
4. [LoRA 论文页](https://huggingface.co/papers/2106.09685)
5. [PEFT + DeepSpeed 官方指南](https://huggingface.co/docs/peft/accelerate/deepspeed)
