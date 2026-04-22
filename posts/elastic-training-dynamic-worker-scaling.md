## 核心结论

弹性训练是指训练过程中允许 `worker` 数量动态增减，并在成员变化后通过重新组网、恢复状态、继续训练来保持任务不断跑。`worker` 是参与训练的进程或设备，常见形式是一张 GPU 对应一个 worker。

对新手来说，可以先记住一句话：集群里有机器下线时，不必重头训练，系统会重新组网、从 checkpoint 继续跑。比如原来 8 张卡训练，Spot 实例被回收后只剩 4 张卡，弹性训练的目标不是让训练自动变快，而是让任务从最近状态恢复后继续，不回到第 0 步。

弹性训练的核心公式是：

$$
B = m \times a \times W
$$

其中 `B` 是有效全局 batch size，`m` 是每卡 micro batch size，`a` 是梯度累积步数，`W` 是当前 world size。`world size` 是当前参与分布式训练的 worker 总数。`W` 变化时，通常要同步调整 `m` 或 `a`，尽量让 `B` 不变，从而减少训练语义变化。

| 维度 | 固定训练 | 弹性训练 |
|---|---|---|
| worker 数量 | 启动后固定 | 运行时可增减 |
| 节点掉线 | 通常任务失败 | 可重新组网后继续 |
| 是否需要 rendezvous | 启动时一次 | 成员变化时重新执行 |
| checkpoint 作用 | 容错或恢复 | 训练连续性的核心依赖 |
| 复杂度 | 低 | 中到高 |

---

## 问题定义与边界

弹性训练解决的是“成员变化后的连续训练”，不是自动修复所有训练问题。它不能保证任何情况下都得到完全相同的训练轨迹，也不能替代正确的数据切分、状态保存和学习率配置。

先定义几个核心符号：

| 符号或术语 | 含义 |
|---|---|
| `W` | 当前 world size，即参与训练的 worker 数量 |
| `m` | 每个 worker 单次前向/反向处理的 micro batch size |
| `a` | gradient accumulation steps，即累积多少次梯度后更新一次参数 |
| `B` | 有效全局 batch size，公式为 `m × a × W` |
| `checkpoint` | 训练状态快照，至少包含模型、优化器、调度器和进度 |
| `rendezvous` | 重新确认成员列表并建立分布式通信组的过程 |

`gradient accumulation` 是“先算多次小 batch 的梯度，再合并成一次参数更新”的技术。它常用于显存不够但又希望保持较大全局 batch 的场景。

一个典型边界例子是 Spot 节点被回收。原来有 8 个 worker，运行中只剩 5 个。弹性训练不会把这个情况简单视为失败，而是重新确认当前可用 worker，得到新的 `W=5`，然后决定如何恢复训练、如何重建 dataloader、如何调整 batch 配置。

| 问题 | 是否属于弹性训练范围 | 说明 |
|---|---|---|
| 节点被抢占后继续训练 | 是 | 通过重新组网和 checkpoint 恢复处理 |
| 新 worker 加入后扩大训练规模 | 是 | 需要重新计算 `W` 和 batch 配置 |
| 显存不足导致 OOM | 部分相关 | 可以调 `m`，但不是弹性训练本身自动解决 |
| loss 爆炸或模型不收敛 | 否 | 主要是优化、数据、模型问题 |
| 数据本身有脏样本 | 否 | 需要数据清洗和校验 |

---

## 核心机制与推导

弹性训练的主流程可以拆成三步：

```text
成员变化
  ↓
rendezvous：重新确认 rank / world size / 通信组
  ↓
从最近 checkpoint 恢复 model / optimizer / scheduler / RNG / 进度
  ↓
按新的 W 重建 dataloader，继续训练
```

`rank` 是每个 worker 在分布式任务里的编号，用来区分不同进程负责的通信和数据分片。成员变化后，旧的 rank 映射可能失效，所以不能假设“原来的第 3 个 worker 永远还是第 3 个”。

batch 推导是弹性训练里最容易被忽略的部分。假设初始配置为：

```text
m = 4
a = 2
W = 8
B = 4 × 2 × 8 = 64
```

如果 Spot 回收导致 worker 变成 `W=4`，为了保持 `B=64`，可以把 `a` 调成 4：

```text
m = 4
a = 4
W = 4
B = 4 × 4 × 4 = 64
```

新手版解释是：卡少了一半，就让每张卡多累积几步梯度，保持一次参数更新看到的总样本数不变。

但这只是理想情况。实际工程里，`m` 受显存限制，`a` 受吞吐和训练时间影响，学习率还可能依赖 batch size。如果无法维持原来的 `B`，就要接受新的可行配置，并同步调整学习率或 warmup 计划。

| 可调整项 | 作用 | 风险 |
|---|---|---|
| `m` | 改变每卡显存占用和吞吐 | 太大容易 OOM |
| `a` | 保持全局 batch 的常用手段 | 太大可能降低更新频率 |
| `learning rate` | 适配 batch 变化 | 不调可能收敛漂移 |
| `checkpoint interval` | 控制故障后损失的进度 | 太频繁会增加 I/O 开销 |

数据侧还有一个关键问题：worker 数量变化后，样本分片必须稳定。常见做法是稳定分区或一致性 hash。一致性 hash 是一种“让对象尽量稳定映射到节点”的方法；节点数量变化时，只让一小部分样本重新分配，减少重复和遗漏风险。它不是 PyTorch Elastic 的强制要求，但在自定义数据管线中很常见。

玩具例子：有 12 个样本，固定写死成 `sample_id % 8 == rank`。当 worker 从 8 个变成 4 个时，旧规则不再对应当前 rank 范围，部分样本会没有 worker 处理，或者处理逻辑被迫改写。更稳妥的方式是按当前 `W` 重新分片，并记录已消费进度。

---

## 代码实现

实现弹性训练要拆成两部分：启动层负责弹性拉起和重新组网，训练脚本负责完整保存与恢复状态。只保存模型参数不够，因为 optimizer、scheduler、随机数状态和 dataloader 进度都会影响后续训练。

下面是一个可运行的 Python 玩具代码，演示 `W` 变化后如何重算梯度累积步数，并用 `assert` 校验全局 batch 不变：

```python
def choose_accumulation(target_global_batch, micro_batch, world_size):
    assert target_global_batch > 0
    assert micro_batch > 0
    assert world_size > 0
    denom = micro_batch * world_size
    assert target_global_batch % denom == 0, "需要能整除，否则要接受新的 batch 配置"
    return target_global_batch // denom

m = 4
old_w = 8
old_a = 2
target_b = m * old_a * old_w

new_w = 4
new_a = choose_accumulation(target_b, m, new_w)

assert target_b == 64
assert new_a == 4
assert m * new_a * new_w == target_b
```

真实 PyTorch 工程中，训练骨架通常长这样：

```python
import os
import random
import torch
import torch.distributed as dist

def init_distributed():
    # 1. 初始化分布式环境
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank

def save_checkpoint(path, model, optimizer, scheduler, step, dataloader_state):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
        "step": step,
        "dataloader_state": dataloader_state,
    }, path)

def load_checkpoint(path, model, optimizer, scheduler):
    # 2. 加载 checkpoint
    if not os.path.exists(path):
        return 0, None

    ckpt = torch.load(path, map_location="cpu")

    # 3. 恢复 model / optimizer / scheduler / RNG
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    torch.set_rng_state(ckpt["rng_cpu"])
    torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    random.setstate(ckpt["python_rng"])

    return ckpt["step"], ckpt["dataloader_state"]

def build_dataloader(dataset, world_size, rank, dataloader_state):
    # 4. 按当前 world size 重建 dataloader
    # 实际项目中应使用 DistributedSampler 或自定义可恢复 sampler。
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=4, sampler=sampler)

def train(model, optimizer, scheduler, dataset, checkpoint_path):
    rank, world_size, local_rank = init_distributed()

    step, dataloader_state = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler
    )

    dataloader = build_dataloader(dataset, world_size, rank, dataloader_state)

    # 5. 继续训练
    for batch in dataloader:
        loss = model(batch).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1

        if rank == 0 and step % 100 == 0:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                step,
                dataloader_state={"step": step},
            )
```

PyTorch Elastic 常见启动方式如下。`--nnodes=1:4` 表示允许 1 到 4 个节点参与，`--max-restarts` 表示成员变化或失败后的重启次数上限，`--rdzv-*` 是 rendezvous 配置：

```bash
torchrun \
  --nnodes=1:4 \
  --nproc-per-node=8 \
  --max-restarts=20 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=host0:29400 \
  --rdzv-id=pretrain-job-001 \
  train.py
```

DeepSpeed Elastic 的配置思想类似：声明最小和最大 worker 范围，并让框架在约束内搜索可行 batch 配置。真实配置会依赖 DeepSpeed 版本和集群启动器，核心是把“允许变化的节点范围”和“batch 约束”写清楚。

真实工程例子：大模型预训练运行在 AWS、GCP 或 Azure 的 Spot 节点上。某些节点被回收时，`torchrun` 或 DeepSpeed Elastic 触发重新 rendezvous；训练脚本从最近 checkpoint 恢复模型、优化器、LR scheduler 和随机数状态；数据侧使用可重分片 sampler 或稳定分区，避免丢进度和重复样本。

---

## 工程权衡与常见坑

弹性训练最大风险不是“训练中断”，而是“恢复后语义变了”。训练能继续跑，只说明系统活着；训练是否等价恢复，还取决于状态保存、数据分片和超参数联动。

| 常见坑 | 后果 | 规避动作 |
|---|---|---|
| `optimizer/scheduler/RNG` 没保存 | 恢复后更新方向、学习率、随机增强都可能变化 | checkpoint 保存完整训练状态 |
| 数据分片固定死 | worker 变化后重复采样或漏样本 | 使用 `DistributedSampler`、可恢复 sampler 或稳定分区 |
| batch 变了但 LR 没调 | 收敛速度和稳定性漂移 | 记录 `B`，按策略同步调整 LR 和 warmup |
| checkpoint 太稀疏 | 故障后回退很多步 | 按训练成本和 I/O 成本设置间隔 |
| 只测单机恢复 | 多机重组网时才暴露问题 | 在测试环境模拟 worker 增减 |

一个常见失败案例是把数据划分写成固定的 `rank % N`。原来按 8 份切数据，现在只剩 4 份，旧规则就不再对齐。结果可能是某些样本被多个 worker 重复读，某些样本完全没人读。对小模型来说这可能只是指标抖动；对大规模预训练来说，这会变成难以定位的数据语义问题。

checkpoint 间隔也要权衡。间隔太短会增加存储和网络 I/O，影响吞吐；间隔太长则每次故障都会损失更多进度。Spot 场景里，通常会根据实例回收频率、单步成本和 checkpoint 写入速度设置，而不是机械地每隔固定 epoch 保存。

---

## 替代方案与适用边界

弹性训练不是默认最优方案。稳定集群里，固定 worker 数训练更简单、可预测、便于复现实验。如果节点很少掉线，引入弹性训练反而会增加启动、状态管理、数据恢复和排障成本。

| 方案 | 适用场景 | 优点 | 缺点 | 维护成本 |
|---|---|---|---|---|
| 固定训练 | 单机、多机专有集群、节点稳定 | 简单、复现性好 | 节点失败通常导致任务失败 | 低 |
| PyTorch Elastic | PyTorch 分布式训练，节点可能变化 | 与 `torchrun` 集成，适合通用训练脚本 | 需要自己处理状态和数据恢复 | 中 |
| DeepSpeed Elastic | 大模型训练，已有 DeepSpeed 栈 | 可结合 ZeRO 和 batch 配置搜索 | 配置复杂，依赖 DeepSpeed 生态 | 中到高 |
| Horovod Elastic | 使用 Horovod 的多框架训练 | 支持弹性 worker 语义 | 生态选择取决于团队历史栈 | 中 |

场景 A：单机 8 卡或专有 GPU 集群，节点几乎不掉线，训练时长可控。固定训练更合适，因为它减少变量，实验更容易复现。

场景 B：训练跑在 AWS/GCP/Azure Spot 节点或共享集群上，机器经常被回收，排队和抢占不可控。弹性训练更合适，因为它用额外复杂度换取更高资源利用率和更低失败成本。

结论是：只有当节点变化是常态，弹性训练才明显值得。否则，先把固定训练、checkpoint 恢复和数据加载做稳，通常更符合工程收益。

---

## 参考资料

本文实现细节主要参考 PyTorch Elastic 的启动与训练脚本、Distributed Checkpoint、DeepSpeed Elastic Config、Horovod Elastic 文档。

1. [PyTorch torchrun Elastic Launch](https://docs.pytorch.org/docs/2.9/elastic/run.html)
2. [PyTorch Train Script for Elastic](https://docs.pytorch.org/docs/2.8/elastic/train_script.html)
3. [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/2.8/distributed.checkpoint.html)
4. [DeepSpeed Config JSON: Elastic Training Config](https://www.deepspeed.ai/docs/config-json/)
5. [Horovod Elastic Documentation](https://horovod.readthedocs.io/en/latest/elastic_include.html)
