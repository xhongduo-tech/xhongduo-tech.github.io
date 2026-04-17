## 核心结论

ZeRO Stage 3 的核心动作，是把**参数本身也纳入分片范围**。在普通 DDP 里，每张 GPU 都长期保存一整份参数、梯度和优化器状态；在 ZeRO-3 里，这三类状态都只保留本 rank 负责的那一段，真正计算某一层时，再把这一层需要的完整参数临时收集回来。

先把量纲说清楚。假设模型有 $P$ 个参数，采用 Adam 优化器和混合精度训练，常见持久状态包括：

| 状态 | 精度 | 每个参数对应字节数 | 总占用 |
| --- | --- | --- | --- |
| 参数 | FP16 / BF16 | 2 bytes | $2P$ |
| 梯度 | FP16 / BF16 | 2 bytes | $2P$ |
| Adam 主权重 | FP32 | 4 bytes | $4P$ |
| Adam 一阶矩 $m$ | FP32 | 4 bytes | $4P$ |
| Adam 二阶矩 $v$ | FP32 | 4 bytes | $4P$ |

所以训练态的持久状态总量近似为：

$$
2P + 2P + 4P + 4P + 4P = 16P \text{ bytes}
$$

为了书写方便，后文把 $P$ 记成归一化后的模型规模 $\Psi$，于是有：

$$
(2 + 2 + 12)\Psi = 16\Psi \text{ bytes}
$$

ZeRO-3 会把参数、梯度、优化器状态都按数据并行组内的 $N$ 张卡分片，因此单卡持久显存近似变成：

$$
M_{\text{ZeRO-3}} = \frac{16\Psi}{N}
$$

这就是 ZeRO-3 的根本价值：**单卡长期持有的模型状态，理论上随数据并行卡数近似按 $N$ 倍下降。**

代价是通信更多。因为参数不再完整常驻，每次层计算前都要把该层参数临时凑齐。用抽象通信模型描述，一步训练里最主要的通信可以近似写成：

$$
\Psi \;(\text{forward AllGather})
+ \Psi \;(\text{backward AllGather})
+ \Psi \;(\text{grad Reduce-Scatter})
= 3\Psi
$$

而普通 DDP 的梯度同步通常近似为：

$$
C_{\text{DDP}} \approx 2\Psi
$$

因此：

$$
\frac{C_{\text{ZeRO-3}}}{C_{\text{DDP}}}
= \frac{3\Psi}{2\Psi}
= 1.5
$$

也就是常说的：**ZeRO-3 在理想化模型下，通信量大约比 DDP 多 50%。**

给一个新手能立即看懂的玩具例子。假设模型训练态总共需要 16GB 持久状态，放在 4 张 GPU 上：

- DDP：每张卡都常驻完整 16GB
- ZeRO-3：每张卡长期只保留约 4GB 分片
- 某一层要计算时，再把该层参数临时收集成完整权重
- 这一层算完后，这份临时完整权重可以释放

所以 ZeRO-3 解决的不是“总状态消失了”，而是把“每张卡都存一整份”，改成“每张卡只长期存自己负责的那一份”。

| 方案 | 单卡持久状态 | 参数是否持久全量复制 | 单步主要通信 |
| --- | --- | --- | --- |
| DDP | $16\Psi$ | 是 | 约 $2\Psi$ |
| ZeRO-3 | $16\Psi / N$ | 否 | 约 $3\Psi$ |
| 直接收益 | 显存简单但压力大 | 实现直观 | 带宽压力较低 |
| 直接代价 | 显存随模型线性上涨 | 运行时无需取参 | 无法靠分片撑更大模型 |
| ZeRO-3 的交换条件 | 显存按卡数扩展 | 需要按层拉取参数 | 能训练更大模型，但更吃通信 |

---

## 问题定义与边界

问题先限定清楚：**ZeRO-3 主要解决的是模型状态装不下的问题，不是算力不够的问题。**

DDP 是最常见的数据并行训练方式。它的运行逻辑可以概括成两句话：

1. 每张卡放同一个模型副本
2. 每张卡处理不同 mini-batch，再把梯度同步起来

这套机制实现简单、生态成熟，但它有一个很硬的代价：**模型状态在每张卡上都是完整复制的。**

如果按 Adam + 混合精度训练，单卡持久模型状态近似为：

$$
M_{\text{DDP}} = 16\Psi
$$

ZeRO-3 并不减少全局总状态，它只是把总状态切开，分散到 $N$ 张卡上：

$$
M_{\text{ZeRO-3}} = \frac{16\Psi}{N}
$$

这决定了 ZeRO-3 的适用边界很明确：

1. 模型训练态单卡放不下，DDP 无法直接跑。
2. 你至少有足够多的数据并行卡，让分片后的单卡状态落回可运行区间。
3. 你的通信链路不能太差，否则参数收集会把吞吐拖垮。

举一个数量级明确的例子。假设某模型训练态持久状态约 256GB，你有 8 张 32GB GPU：

- DDP：每张卡都要放 256GB，完全不可行
- ZeRO-3：单卡持久状态约为 $256/8 = 32$ GB
- 从“完全放不下”，变成“接近可运行边界”

这里“接近可运行”非常重要，因为 ZeRO-3 降低的是**持久模型状态**，不是所有显存占用。训练时还有这些部分会占显存：

| 显存来源 | 是否被 ZeRO-3 直接按 $N$ 分摊 | 说明 |
| --- | --- | --- |
| 持久参数 / 梯度 / 优化器状态 | 是 | 这是 ZeRO-3 的主要收益来源 |
| 激活值 | 否 | 与 batch size、序列长度、层数强相关 |
| 临时工作区 / kernel workspace | 否 | 由具体算子和实现决定 |
| 通信 buffer | 部分相关 | bucket 越大，峰值越高 |
| CUDA allocator 碎片 | 否 | 工程里经常是隐藏 OOM 来源 |

所以工程上真正的判断标准不是“$\frac{16\Psi}{N}$ 小于显存容量就一定能跑”，而是：

$$
\text{持久状态} + \text{激活值} + \text{通信缓存} + \text{临时 buffer} + \text{碎片余量}
< \text{GPU 可用显存}
$$

这也是为什么实际训练中，ZeRO-3 往往要和这些手段配合使用：

- activation checkpointing
- 合理设置 micro-batch size
- 调整 allgather / reduce bucket
- CPU 或 NVMe offload
- 限制某些参数持久驻留

换句话说，**ZeRO-3 解决的是“状态复制过多”这个核心瓶颈，但它不是显存问题的唯一答案。**

---

## 核心机制与推导

ZeRO-3 可以拆成两个动作理解：**永久分片**和**按需收集**。这两个动作合在一起，才构成它和 ZeRO-1、ZeRO-2 的本质差异。

### 1. 永久分片

所谓永久分片，是指训练过程中长期保存在每张卡上的，不再是完整状态，而只是其中一部分。

如果数据并行组大小为 $N$，那么单卡长期持有的状态近似为：

- 参数：$\frac{2\Psi}{N}$
- 梯度：$\frac{2\Psi}{N}$
- Adam 状态：$\frac{12\Psi}{N}$

所以单卡持久状态总量就是：

$$
\frac{2\Psi}{N} + \frac{2\Psi}{N} + \frac{12\Psi}{N}
= \frac{16\Psi}{N}
$$

这一步只回答了一个问题：**为什么显存降了。**

### 2. 按需收集

但分片之后，新的问题立刻出现：某一层做矩阵乘法时，需要的是这层的完整权重，不是一小段分片。于是 ZeRO-3 的第二个动作是按需收集。

典型过程如下：

1. 前向走到某层
2. 每张卡拿出自己持有的参数分片
3. 通过 AllGather 把该层完整参数临时拼出来
4. 执行前向计算
5. 如果完整参数后面暂时用不到，可以释放
6. 反向走回这一层时，再次 AllGather
7. 计算梯度后，通过 Reduce-Scatter 把梯度重新分散回各卡

这个过程可以画成一个极简时序：

| 时刻 | GPU 本地长期保存 | 临时通信动作 | 结果 |
| --- | --- | --- | --- |
| 前向前 | 仅参数分片 | 无 | 显存低 |
| 某层前向开始 | 参数分片 | AllGather 该层参数 | 临时得到完整权重 |
| 某层前向结束 | 参数分片 | 释放完整权重 | 回到低占用 |
| 某层反向开始 | 参数分片 | 再次 AllGather | 再次得到完整权重 |
| 梯度计算后 | 梯度完整值短暂存在 | Reduce-Scatter | 每张卡只留梯度分片 |

于是，抽象通信模型下，ZeRO-3 单步训练最主要的通信量可以写成：

$$
C_{\text{ZeRO-3}}
\approx
\Psi + \Psi + \Psi = 3\Psi
$$

这里三项分别对应：

1. 前向参数 AllGather
2. 反向参数 AllGather
3. 梯度 Reduce-Scatter

而 DDP 通常只需要对梯度做 AllReduce。把一次 AllReduce 按发送和接收两部分近似，可写成：

$$
C_{\text{DDP}} \approx 2\Psi
$$

因此二者比值为：

$$
\frac{C_{\text{ZeRO-3}}}{C_{\text{DDP}}}
=
\frac{3\Psi}{2\Psi}
=
1.5
$$

这就是“通信增加约 50%”的数学来源。这里要强调两点：

- 这是**抽象量级模型**，忽略了启动延迟、拓扑差异、bucket 切分和重叠效果
- 真实训练里，吞吐下降未必刚好是 50%，因为通信可以和计算部分重叠

### 3. 一个按层理解的玩具例子

假设某层完整权重是 400MB，4 张卡训练。

在 DDP 中：

- 每张卡长期都有 400MB 完整权重
- 反向后做梯度同步

在 ZeRO-3 中：

- 每张卡平时只存 100MB 参数分片
- 前向到这一层时，4 张卡 AllGather，临时凑成 400MB
- 前向算完，这 400MB 完整权重可释放
- 反向时再 AllGather 一次
- 梯度算完，不保留完整梯度，而是 Reduce-Scatter 成每张卡 100MB 梯度分片

所以对新手最关键的一句话是：

**ZeRO-3 不是“参数不存在了”，而是“参数默认不完整常驻，只有在这层要算时才短暂拼齐”。**

### 4. 为什么跨节点更容易掉吞吐

单机内通常有 NVLink 或高带宽 PCIe，AllGather 成本相对还能接受；跨节点时，通信可能要穿过 InfiniBand，甚至普通以太网。此时 ZeRO-3 的参数收集会直接进入前后向关键路径，问题会明显放大。

可以把瓶颈粗略写成：

$$
T_{\text{step}}
\approx
T_{\text{compute}}
+
\max(0, T_{\text{comm}} - T_{\text{overlap}})
$$

其中：

- $T_{\text{compute}}$ 是算子计算时间
- $T_{\text{comm}}$ 是参数收集和梯度分发时间
- $T_{\text{overlap}}$ 是被计算掩盖掉的通信时间

如果网络快、重叠做得好，那么额外通信未必显著拖慢训练；如果网络慢、layer 很碎、bucket 很小、跨机频繁，那么 $T_{\text{comm}}$ 就会直接变成吞吐损失。

所以从机制上看，ZeRO-3 的收益和代价非常对称：

- 收益：把状态从“每卡完整复制”变成“每卡只留分片”
- 代价：把“完整参数常驻”改成“完整参数按层动态收集”

---

## 代码实现

这一节给两类代码：

1. 一个**可以直接运行**的 Python 小脚本，用来验证文中的显存与通信公式
2. 一个更接近真实训练的 DeepSpeed 配置与初始化示例

先给公式验证代码。下面的脚本不依赖 DeepSpeed，只依赖 Python 标准库，可以直接运行。

```python
from dataclasses import dataclass


BYTES_FP16 = 2
BYTES_FP32 = 4


@dataclass
class MemoryReport:
    param_count: int
    world_size: int
    ddp_persistent_bytes: int
    zero3_persistent_bytes: int
    ddp_comm_bytes_per_step: int
    zero3_comm_bytes_per_step: int
    comm_overhead_ratio: float


def format_gib(num_bytes: int) -> str:
    gib = num_bytes / (1024 ** 3)
    return f"{gib:.2f} GiB"


def zero3_memory_and_comm(param_count: int, world_size: int) -> MemoryReport:
    """
    用文章中的简化假设估算 DDP 与 ZeRO-3。

    假设：
    1. 参数为 FP16/BF16: 2 bytes / param
    2. 梯度为 FP16/BF16: 2 bytes / param
    3. Adam 状态包含：
       - FP32 master weight: 4 bytes / param
       - FP32 m: 4 bytes / param
       - FP32 v: 4 bytes / param
    4. 忽略激活值、碎片、kernel workspace、通信启动开销
    """
    if param_count <= 0:
        raise ValueError("param_count must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    param_bytes = BYTES_FP16 * param_count
    grad_bytes = BYTES_FP16 * param_count
    adam_state_bytes = 3 * BYTES_FP32 * param_count

    ddp_persistent_bytes = param_bytes + grad_bytes + adam_state_bytes
    zero3_persistent_bytes = ddp_persistent_bytes // world_size

    # 抽象通信模型：
    # DDP: 约 2 * Psi
    # ZeRO-3: forward AG + backward AG + grad RS = 3 * Psi
    ddp_comm_bytes_per_step = 2 * param_count
    zero3_comm_bytes_per_step = 3 * param_count

    return MemoryReport(
        param_count=param_count,
        world_size=world_size,
        ddp_persistent_bytes=ddp_persistent_bytes,
        zero3_persistent_bytes=zero3_persistent_bytes,
        ddp_comm_bytes_per_step=ddp_comm_bytes_per_step,
        zero3_comm_bytes_per_step=zero3_comm_bytes_per_step,
        comm_overhead_ratio=zero3_comm_bytes_per_step / ddp_comm_bytes_per_step,
    )


def main() -> None:
    report = zero3_memory_and_comm(param_count=1_000_000_000, world_size=8)

    print(f"parameters: {report.param_count:,}")
    print(f"world size: {report.world_size}")
    print(f"DDP persistent memory per GPU:    {format_gib(report.ddp_persistent_bytes)}")
    print(f"ZeRO-3 persistent memory per GPU: {format_gib(report.zero3_persistent_bytes)}")
    print(f"DDP communication units:          {report.ddp_comm_bytes_per_step / 1e9:.2f}")
    print(f"ZeRO-3 communication units:       {report.zero3_comm_bytes_per_step / 1e9:.2f}")
    print(f"communication ratio:              {report.comm_overhead_ratio:.2f}x")

    assert report.zero3_persistent_bytes * report.world_size == report.ddp_persistent_bytes
    assert abs(report.comm_overhead_ratio - 1.5) < 1e-12
    assert report.zero3_comm_bytes_per_step > report.ddp_comm_bytes_per_step


if __name__ == "__main__":
    main()
```

如果参数量是 $10^9$，这个脚本会得出：

- DDP 单卡持久状态约 14.90 GiB
- ZeRO-3 在 8 卡上约 1.86 GiB
- 抽象通信比值为 1.5x

这里 14.90 GiB 来自：

$$
16 \times 10^9 \text{ bytes} \div 1024^3 \approx 14.90 \text{ GiB}
$$

这个例子只验证两个核心结论：

1. 持久状态按 $N$ 分摊
2. 通信从 $2\Psi$ 增到 $3\Psi$

它**没有**模拟这些真实因素：

- activation memory
- 通信与计算重叠
- bucket 切分
- 拓扑结构
- 参数按层收集的时序细节

下面给一个更接近实战的 DeepSpeed 配置：

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "overlap_comm": true,
    "allgather_bucket_size": 200000000,
    "reduce_bucket_size": 200000000,
    "stage3_prefetch_bucket_size": 50000000,
    "stage3_param_persistence_threshold": 100000,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

这些配置项里，新手最需要先理解的是下面几个：

| 配置项 | 作用 | 调大后的倾向 | 调小后的倾向 |
| --- | --- | --- | --- |
| `stage` | 开启哪一档 ZeRO | `3` 表示连参数也分片 | 无 |
| `reduce_scatter` | 梯度同步后按分片落地 | 更符合 ZeRO-3 模式 | 关闭通常不划算 |
| `overlap_comm` | 通信和计算重叠 | 吞吐可能更高 | 行为更保守 |
| `allgather_bucket_size` | 参数收集的 bucket 大小 | 小包更少，但峰值显存更高 | 峰值更低，但通信碎片更多 |
| `reduce_bucket_size` | 梯度分发的 bucket 大小 | 同上 | 同上 |
| `stage3_prefetch_bucket_size` | 提前预取参数的大小 | 更积极隐藏延迟 | 预取不足，可能等通信 |
| `stage3_param_persistence_threshold` | 小参数是否尽量常驻 | 减少小参数反复 gather | 显存更省但取参更频繁 |
| `stage3_gather_16bit_weights_on_model_save` | 保存时是否收集完整 16-bit 权重 | 导出更方便 | 需要额外转换脚本 |

下面是一个最小训练循环示例。它假设你已有 `MyModel`、`dataloader`，并且环境里已经安装了 `deepspeed` 和 `torch`。

```python
import torch
import deepspeed


class MyModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, batch):
        x = batch["x"]
        y = self.net(x)
        return (y ** 2).mean()


def build_fake_dataloader(
    num_batches: int = 10,
    batch_size: int = 2,
    hidden_size: int = 1024,
    device: str = "cpu",
):
    for _ in range(num_batches):
        yield {"x": torch.randn(batch_size, hidden_size, device=device)}


def main():
    model = MyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config="ds_config_zero3.json",
    )

    dataloader = build_fake_dataloader(device=engine.device)

    engine.train()
    for step, batch in enumerate(dataloader, start=1):
        loss = engine(batch)
        engine.backward(loss)
        engine.step()

        if engine.global_rank == 0:
            print(f"step={step}, loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
```

启动命令通常类似：

```bash
deepspeed --num_gpus=8 train.py
```

如果你是多机训练，则需要额外提供 hostfile 或 launcher 参数，这部分和 ZeRO-3 本身不是同一个问题，但会直接影响通信效果。

### 为什么有时训练阶段没 OOM，初始化阶段却 OOM

这是 ZeRO-3 最常见的新手误区之一。

很多人理解为“我已经开了 Stage 3，所以模型会自动分片”，但如果模型对象在 `deepspeed.initialize()` 之前就以完整形式构建完成，那么**初始化那一刻**依然可能先把完整参数放上 GPU 或放进某个中间状态，从而提前 OOM。

这就是 `deepspeed.zero.Init()` 的意义。它让参数在构建阶段就按分片方式初始化，而不是“先完整创建，再分片”。

示意代码如下：

```python
import deepspeed
import torch


class HugeModel(torch.nn.Module):
    def __init__(self, hidden_size=16384):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.proj(x)


with deepspeed.zero.Init():
    model = HugeModel()
```

对超大模型，这不是可选优化，而是能否完成初始化的前提。

---

## 工程权衡与常见坑

ZeRO-3 不是“免费显存”。它做的是一笔非常明确的交换：

- 少占持久显存
- 多做参数通信
- 增加运行时调度复杂度

所以判断要不要上 Stage 3，不能只问“我想不想省显存”，而要问：

**当前训练的主矛盾，到底是显存容量，还是通信吞吐。**

下面这张表把常见坑和处理思路放在一起：

| 常见坑 | 典型现象 | 根因 | 优先处理方式 |
| --- | --- | --- | --- |
| 跨节点 AllGather 很慢 | GPU 利用率低，step time 抖动大 | 参数收集跨慢链路 | 开 `overlap_comm`，检查网络拓扑，减少跨机 gather |
| `allgather_bucket_size` 过大 | 峰值显存升高，偶发 OOM | 单次通信缓存太大 | 逐步下调 bucket |
| `allgather_bucket_size` 过小 | 吞吐变差 | 包太碎，延迟占主导 | 在不 OOM 的前提下增大 bucket |
| 小层很多，参数频繁拉取 | MFU 低，通信调用密 | 参数 gather 次数过多 | 提高小参数持久化阈值，减少碎片层 |
| 保存模型失败或只拿到分片权重 | 导出的 `state_dict` 不完整 | Stage 3 下权重天然是分片的 | 打开 `stage3_gather_16bit_weights_on_model_save` 或后处理转换 |
| 初始化阶段直接 OOM | 还没开始 train 就挂 | 模型先被完整构建 | 使用 `deepspeed.zero.Init()` |
| 自定义模块访问外部参数报错 | 某层 forward 异常 | 参数访问路径不在 ZeRO 管理范围内 | 使用 `register_external_parameter` |
| 开了 ZeRO-3 还是 OOM | 理论上应能装下，实际上还是爆 | 激活值、碎片、workspace 未控制 | 配合 checkpointing、减小 batch、调 bucket |

### 一个真实感很强的工程场景

单机 8 卡时，训练看起来正常；扩到 2 节点 16 卡后，吞吐突然掉很多。这里最常见的错误判断是“模型变大了，所以算子慢了”。实际上，很多时候慢的是通信，不是计算。

原因通常是：

1. 单机内参数收集走 NVLink，代价可接受
2. 多机后，部分 AllGather 开始跨节点
3. 参数收集被插进前向和反向路径
4. 如果重叠不充分，通信时间直接暴露出来

这类问题通常按下面顺序排查：

1. 先确认是不是网络瓶颈，而不是 dataloader 或 CPU 卡住
2. 检查 `overlap_comm` 是否开启
3. 调整 `allgather_bucket_size` 和 `reduce_bucket_size`
4. 检查模型是否有大量小层、MoE 路由、外部参数访问等导致 gather 频繁
5. 再考虑 `stage3_param_persistence_threshold`、ZeRO++、offload 等更重手段

一个经验判断是：

- 如果单机快、多机慢，优先怀疑跨节点通信
- 如果显存够但吞吐差，优先怀疑 Stage 3 的参数拉取代价
- 如果显存不够，那 Stage 3 可能仍然是必须的，只是需要继续做通信优化

---

## 替代方案与适用边界

不是所有显存问题都应该直接上 ZeRO-3。ZeRO-3 最强，但也最依赖通信条件。很多时候，更温和的方案反而更稳。

先看几种方案的对比：

| 方案 | 分片对象 | 单卡状态量级 | 通信特征 | 适用边界 |
| --- | --- | --- | --- | --- |
| DDP | 不分片 | $16\Psi$ | 约 $2\Psi$ | 模型能轻松放下，优先简单稳定 |
| ZeRO-1 | 只分优化器状态 | 约 $(2 + 2 + 12/N)\Psi$ | 接近 DDP | 显存略紧，先省 optimizer |
| ZeRO-2 | 分优化器状态和梯度 | 约 $(2 + 14/N)\Psi$ | 接近 DDP | 显存明显紧张，但还不想引入参数 gather |
| ZeRO-3 | 参数、梯度、优化器都分片 | $16\Psi/N$ | 约 $3\Psi$ | 必须靠参数分片才能训练 |
| ZeRO-3 + Offload | 再把部分状态放 CPU/NVMe | GPU 占用更低 | 延迟和带宽更复杂 | GPU 显存极紧，但主机资源充足 |

### 给新手的选择规则

如果只需要一个能落地的判断顺序，可以用下面这套：

1. 模型能稳定放下，优先 DDP。
2. 只差一点显存，先试 ZeRO-2。
3. 明显放不下，且参数复制就是主瓶颈，再上 ZeRO-3。
4. ZeRO-3 仍然不够，再加 CPU / NVMe offload。
5. 如果跨节点网络差，优先考虑 ZeRO-2 + offload，而不是直接冲 Stage 3。

### 为什么 ZeRO-2 经常是更稳的中间解

ZeRO-2 已经能分掉优化器状态和梯度，单卡状态从

$$
16\Psi
$$

降到

$$
(2 + 14/N)\Psi
$$

它保留了完整参数常驻，因此**不需要在前后向里频繁 AllGather 参数**。这意味着：

- 显存比 DDP 低很多
- 通信模式又没有 ZeRO-3 那么激进
- 对网络条件的要求更低

举个具体判断。假设你只有 4 张 24GB GPU，而且机器间带宽一般：

- 如果模型只是“略超显存”，ZeRO-2 + CPU offload 往往更稳
- 如果模型连参数完整常驻都做不到，那才必须上 ZeRO-3

所以真正实用的思路不是“ZeRO-3 最先进，所以优先用”，而是：

**只在参数复制已经成为决定性瓶颈时，才支付 ZeRO-3 的额外通信成本。**

---

## 参考资料

下面给出更完整的参考资料表。阅读顺序建议是：先看官方文档确认行为，再看论文理解阶段划分，最后看博客理解工程动机和规模案例。

| 来源 | 链接 | 主要用途 |
| --- | --- | --- |
| DeepSpeed 官方 ZeRO 文档 | [https://deepspeed.readthedocs.io/en/stable/zero3.html](https://deepspeed.readthedocs.io/en/stable/zero3.html) | 核实 Stage 3 会对参数分片，并在前后向中自动收集与再分片 |
| DeepSpeed Training API | [https://deepspeed.readthedocs.io/en/stable/training.html](https://deepspeed.readthedocs.io/en/stable/training.html) | 核实训练初始化、保存 16-bit 权重与相关接口 |
| ZeRO 官方论文 | [https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054) | 核实 ZeRO 的目标、三阶段划分和内存节省公式 |
| Microsoft Research 博客 | [https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) | 补充 ZeRO 的动机、线性扩展直觉和通信代价背景 |
| DeepSpeed ZeRO-3 Offload 博客 | [https://www.deepspeed.ai/2021/03/07/zero3-offload.html](https://www.deepspeed.ai/2021/03/07/zero3-offload.html) | 工程案例：显存进一步压缩到 CPU / NVMe 的做法 |
| DeepSpeed 配置 JSON 说明 | [https://www.deepspeed.ai/docs/config-json/](https://www.deepspeed.ai/docs/config-json/) | 查各类 bucket、offload、通信相关配置项的语义 |
| PyTorch Distributed 文档 | [https://pytorch.org/docs/stable/distributed.html](https://pytorch.org/docs/stable/distributed.html) | 补充理解 AllReduce、AllGather、Reduce-Scatter 等通信原语 |
| NCCL 文档 | [https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) | 理解多 GPU / 多机通信性能与拓扑影响 |

如果只保留一句总结，可以写成：

- DDP 的问题是每张卡都长期保存一整份状态。
- ZeRO-3 的解法是把参数、梯度、优化器状态都切开。
- 它把单卡持久状态从 $16\Psi$ 压到 $16\Psi/N$。
- 代价是完整参数不再常驻，而要在前后向中按层临时收集，因此通信从约 $2\Psi$ 增到约 $3\Psi$。
- 所以 ZeRO-3 适合“显存已经是决定性瓶颈”的场景，不适合网络很差、但模型其实还能放下的场景。
