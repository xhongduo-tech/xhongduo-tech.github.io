## 核心结论

ZeRO Stage 1 的本质，是把**优化器状态**按数据并行进程数切分，而不是把整个模型切分。这里先把三个最容易混淆的对象分开：

| 对象 | 是什么 | 训练时为什么要占显存 |
| --- | --- | --- |
| 参数（Parameters） | 模型真正要学习的权重 $w$ | 前向、反向、更新都要用 |
| 梯度（Gradients） | 损失函数对参数的导数 $g=\partial L/\partial w$ | 反向传播后用于更新参数 |
| 优化器状态（Optimizer States） | 优化器为了“记住历史”而保存的额外张量 | 例如 Adam 需要一阶矩 $m$ 和二阶矩 $v$ |

以 Adam 为例，优化器状态通常至少包含每个参数对应的两份历史统计量：

$$
m_t,\; v_t
$$

如果还使用混合精度训练，很多实现还会保留 FP32 master weight，所以优化器相关显存往往比新手直觉里更大。

在传统数据并行（Data Parallel, DP）里，每张卡都完整保存参数、梯度、优化器状态三份；在 ZeRO Stage 1 里，参数和梯度仍然完整保留在每张卡上，**只有优化器状态变成每卡只保留一部分**。

这带来的直接结果是：如果有 $D$ 张卡，每卡优化器状态的显存占用从 $SP$ 降到 $\frac{SP}{D}$。因此每卡显存公式从

$$
AP + GP + SP
$$

变成

$$
AP + GP + \frac{SP}{D}
$$

其中：

| 符号 | 含义 |
| --- | --- |
| $P$ | 参数总量 |
| $A$ | 每个参数本体的字节数系数 |
| $G$ | 每个梯度的字节数系数 |
| $S$ | 每个优化器状态的字节数系数 |
| $D$ | 数据并行卡数 |

它解决的是“**优化器状态冗余**”问题，不解决“参数本体太大”或“激活太大”问题。这里“激活”指前向传播过程中各层产生、并需要保留到反向传播使用的中间结果。很多 OOM 并不是参数爆掉，而是激活太多，这一点必须先区分。

所以判断是否该上 Stage 1，可以先用一句话概括：

- 如果你最先爆的是 Adam 状态，Stage 1 往往有效。
- 如果爆的是参数本体，Stage 1 不够，需要 Stage 3 或模型并行。
- 如果爆的是激活，Stage 1 基本无效，需要 activation checkpoint 等方法。

从训练行为看，ZeRO Stage 1 一般不改变数学上的优化目标，收敛速度和最终指标通常可以接近原始数据并行；但前提是分片顺序正确、通信桶大小合理、梯度缩放和混合精度配置稳定。它的代价不是重计算，而是更精细的通信调度。

---

## 问题定义与边界

先把边界说清楚。ZeRO Stage 1 不是模型并行。模型并行是“不同卡算模型的不同层或不同张量块”；ZeRO Stage 1 仍然是**数据并行**，只是把冗余保存的优化器状态拆开存。

这个区别非常重要，因为很多新手会把“切分”自动理解成“模型被拆到不同 GPU 上”。Stage 1 不是这样。它的计算形态仍然是：

1. 每张卡拿到一份完整模型参数。
2. 每张卡处理不同的小批数据。
3. 各卡之间同步训练结果。
4. 只是优化器状态不再人人一份。

传统数据并行的问题在于“每卡都保存一模一样的训练状态”。如果参数用 FP16 保存，通常每参数约 2B；梯度也常按 2B 计；而 Adam 的状态至少包含两个 FP32 张量 $m,v$，再加上常见实现里的 FP32 master weight，优化器相关状态往往达到 8B 到 12B 甚至更多。对大模型来说，真正最先顶满显存的，经常不是参数，而是优化器状态。

下面给一个更完整的估算表。假设混合精度训练时采用常见实现：

| 项目 | 常见精度 | 每参数占用 |
| --- | --- | --- |
| 参数副本 | FP16 | 2B |
| 梯度 | FP16 | 2B |
| Adam 一阶矩 $m$ | FP32 | 4B |
| Adam 二阶矩 $v$ | FP32 | 4B |
| FP32 master weight（很多实现会有） | FP32 | 4B |

如果把 master weight 也算在“优化器相关状态”里，那么每参数对应的优化器相关显存会变成：

$$
S \approx 4 + 4 + 4 = 12\text{B}
$$

如果某个实现不保存 master weight，则常见是：

$$
S \approx 4 + 4 = 8\text{B}
$$

这也是为什么工程里经常说“Adam 比 SGD 更吃显存”。不是因为参数变多了，而是因为它额外保存了更多历史状态。

看一个玩具例子。假设每个参数对应：

| 项目 | 每参数占用 |
| --- | --- |
| 参数 | 2B |
| 梯度 | 2B |
| Adam 状态 | 8B |

那么 4 卡训练时：

| 模式 | 每卡参数 | 每卡梯度 | 每卡优化器状态 | 每卡总计 |
| --- | --- | --- | --- | --- |
| 传统数据并行 | 2B | 2B | 8B | 12B |
| ZeRO Stage 1 | 2B | 2B | 2B | 6B |

这里最关键的观察是：Stage 1 只动优化器状态，所以参数和梯度完全没变。也正因此，它特别适合下面这类场景：

| 场景 | 是否适合 ZeRO-1 | 原因 |
| --- | --- | --- |
| Adam 状态先爆显存 | 适合 | 直接削减最大冗余项 |
| 梯度占用过大 | 一般 | Stage 1 不切梯度 |
| 参数本体放不下 | 不适合 | Stage 1 不切参数 |
| 激活爆显存 | 不适合 | Stage 1 不动激活 |

再把“边界”说得更工程化一点。假设某模型有 70 亿参数，参数和梯度采用 2B，Adam 状态按 8B 估算，则单卡数据并行的理论静态占用大致是：

$$
P \times (2 + 2 + 8) = 7\times10^9 \times 12\text{B} \approx 84\text{GB}
$$

这还**没算激活、临时通信缓冲、CUDA allocator 碎片**。如果用了 8 卡 ZeRO-1，则优化器状态项变为：

$$
7\times10^9 \times \frac{8}{8}\text{B} = 7\times10^9 \times 1\text{B?}
$$

更准确地写应是每参数 8B 的优化器状态被 8 卡均摊：

$$
7\times10^9 \times \frac{8\text{B}}{8} = 7\times10^9 \times 1\text{B}
$$

于是每卡静态部分接近：

$$
7\times10^9 \times (2 + 2 + 1)\text{B} = 35\text{GB}
$$

这说明 Stage 1 的收益可以非常大，但也说明另一件事：**参数和梯度仍在，所以模型再大一些，Stage 1 还是会失效**。

所以它的适用边界很明确：**中等规模模型、优化器状态主导显存、希望尽量少改训练代码**。这也是它在工程里常被作为第一步优化的原因。

---

## 核心机制与推导

ZeRO Stage 1 的一个容易误解之处是：既然只分片优化器状态，那每张卡到底有没有完整梯度？答案是，**反向计算时有；但进入优化器更新阶段后，不需要每张卡都长期保留全量状态来完成更新**。

把一个标准迭代拆开看，会更清楚：

1. 每张卡做完整前向和反向，得到本地完整梯度。
2. 数据并行组内对梯度做聚合，并把聚合结果按参数分片分发出去。
3. 每张卡只对自己负责的参数分片执行 optimizer step。
4. 更新后的参数分片再同步给其他卡，恢复下一轮前向所需的完整参数视图。

如果只看概念，常见描述是：

$$
\text{ReduceScatter(gradients)} \rightarrow \text{Local Optimizer Step} \rightarrow \text{AllGather(parameters)}
$$

这里几个通信原语可以先用白话理解：

| 通信原语 | 白话解释 | 在 ZeRO-1 里的作用 |
| --- | --- | --- |
| AllReduce | 大家把同一份张量相加/求平均后，每个人都拿到完整结果 | 传统 DP 常用 |
| ReduceScatter | 先规约，再把结果按切片分给不同卡 | 每卡拿到自己那一段聚合梯度 |
| AllGather | 每卡拿出自己那一段，拼回完整张量 | 把更新后的参数重新拼全 |

设第 $k$ 张卡负责的参数分片为 $w^{(k)}$，对应聚合后的梯度分片为 $\hat g^{(k)}$。对 Adam，有：

$$
m_t^{(k)} \leftarrow \beta_1 m_{t-1}^{(k)} + (1-\beta_1)\hat g_t^{(k)}
$$

$$
v_t^{(k)} \leftarrow \beta_2 v_{t-1}^{(k)} + (1-\beta_2)\big(\hat g_t^{(k)}\big)^2
$$

偏差修正后：

$$
\hat m_t^{(k)} = \frac{m_t^{(k)}}{1-\beta_1^t}, \quad
\hat v_t^{(k)} = \frac{v_t^{(k)}}{1-\beta_2^t}
$$

参数更新为：

$$
w_t^{(k)} \leftarrow w_{t-1}^{(k)} - \eta \cdot \frac{\hat m_t^{(k)}}{\sqrt{\hat v_t^{(k)}}+\epsilon}
$$

这套公式与普通 Adam 没有变化。变化的是：谁来保存 $m_t^{(k)},v_t^{(k)}$，谁来执行这段参数更新。

如果把全量参数向量记成：

$$
w = [w^{(0)}, w^{(1)}, \dots, w^{(D-1)}]
$$

则 ZeRO-1 只是把优化器状态也按相同方式划分：

$$
m = [m^{(0)}, m^{(1)}, \dots, m^{(D-1)}]
$$

$$
v = [v^{(0)}, v^{(1)}, \dots, v^{(D-1)}]
$$

并要求**参数分片、梯度分片、状态分片的映射完全一致**。这句话是 Stage 1 的正确性核心。只要这三者对应关系不乱，数学上更新结果就与普通数据并行一致或数值近似一致。

下面给出一个更适合新手理解的对照。

| 问题 | 普通数据并行 | ZeRO Stage 1 |
| --- | --- | --- |
| 每卡是否有完整参数 | 有 | 有 |
| 每卡是否有完整梯度 | 反向后有 | 反向后有 |
| 每卡是否有完整优化器状态 | 有 | 没有，只保留一部分 |
| 更新时谁负责参数块 $w^{(k)}$ | 每卡都更新同样一份 | 只有拥有该状态分片的卡更新 |
| 更新后参数如何恢复一致 | AllReduce 后天然一致 | 通过 AllGather 或等价同步恢复一致 |

为什么分片不会改变优化方向？因为真正参与更新的仍然是全局规约后的梯度。Stage 1 只是把“存储”和“执行更新”的责任按参数块分给不同 GPU。只要聚合正确、分片一致，它不是近似算法层面的变化，而是训练状态存储方式的变化。

这里的稳定性信号主要看三类。

| 信号 | 正常现象 | 异常现象 |
| --- | --- | --- |
| loss 曲线 | 与普通 DP 接近 | 周期性尖峰、突然发散 |
| grad norm | 变化平滑 | 间歇性爆炸或频繁 NaN |
| 吞吐 | 略降或持平 | bucket 配置不当时明显抖动 |

再看通信与显存的平衡。传统数据并行常用 AllReduce 聚合全梯度；Stage 1 更自然的实现是 `ReduceScatter(梯度) -> 本地更新 -> AllGather(参数)`。它没有通过重计算省显存，省的是优化器状态副本，所以通信并不会消失，只是从“全量梯度规约”转成“梯度分发 + 参数拼回”的模式。

从粗略通信量上看，在很多实现下，Stage 1 与普通 DP 处于同一量级，不是“免费午餐”。因此它常见的真实收益模式是：

- 先省下优化器状态显存。
- 再把节省出来的显存换成更大的 batch size 或更长序列长度。
- 最终训练吞吐可能提升，也可能不提升，取决于网络和桶大小配置。

---

## 代码实现

工程里最常见的实现是 DeepSpeed。最小配置通常只需要打开 Stage 1：

```json
{
  "train_batch_size": 64,
  "gradient_accumulation_steps": 4,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 1,
    "reduce_bucket_size": 500000000,
    "allgather_bucket_size": 500000000,
    "contiguous_gradients": true
  }
}
```

这里把 `allgather_bucket_size` 也写成 `500000000`，是为了让例子更稳妥。真实工程里它未必必须等于 `reduce_bucket_size`，但对初学者来说，先从同量级配置起步更容易排查问题。

几个关键超参数需要理解：

| 参数 | 白话解释 | 影响 |
| --- | --- | --- |
| `stage` | 用哪一级 ZeRO | `1` 代表只分片优化器状态 |
| `reduce_bucket_size` | 一次归并多少梯度 | 太小通信碎片多，太大峰值显存高 |
| `allgather_bucket_size` | 一次拼回多少参数 | 太小吞吐下降，太大容易卡显存 |
| `contiguous_gradients` | 是否把梯度尽量放连续内存 | 通常利于通信和内存碎片控制 |

如果用 Hugging Face `Trainer` 配合 DeepSpeed，一个最小可运行入口大致如下：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

texts = [
    "ZeRO Stage 1 shards optimizer states across data parallel ranks.",
    "Parameters remain replicated, optimizer states do not."
]

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    return_tensors="pt"
)

class ToyDataset:
    def __len__(self):
        return encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item

args = TrainingArguments(
    output_dir="./out",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=1,
    deepspeed="ds_zero1.json",
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ToyDataset()
)

trainer.train()
```

其中 `ds_zero1.json` 就是上面的配置文件。这个例子能跑起来，但要注意两点：

1. 单机单卡运行时，即使打开 ZeRO-1，也没有“跨卡切分”的实际收益。
2. 真正观察显存变化，至少需要多卡数据并行环境。

底层实现通常会先把参数按固定顺序扁平化，再切成与 rank 对应的分区。这样做的目的，是让梯度分片、优化器状态分片、参数分片三者在偏移上完全一致。若不一致，就会出现“第 0 张卡拿到了第 1 段梯度，却更新了第 2 段参数”的错位问题，结果通常不是轻微退化，而是直接训练发散。

下面给一个**可直接运行**的 Python 玩具实现，只演示“按分片更新 Adam 状态”的核心逻辑，不依赖分布式库。它模拟 2 个 rank，对完整梯度先做平均，再按分片执行更新，最后拼回完整参数。

```python
import math


def split_by_sizes(values, sizes):
    chunks = []
    offset = 0
    for size in sizes:
        chunks.append(values[offset: offset + size])
        offset += size
    return chunks


def flatten(chunks):
    out = []
    for chunk in chunks:
        out.extend(chunk)
    return out


def average_gradients(per_rank_grads):
    world_size = len(per_rank_grads)
    num_params = len(per_rank_grads[0])
    avg = []
    for i in range(num_params):
        avg.append(sum(grads[i] for grads in per_rank_grads) / world_size)
    return avg


def adam_zero_stage1_step(
    per_rank_grads,
    param_shards,
    state_shards,
    lr=1e-2,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    step=1,
):
    sizes = [len(shard) for shard in param_shards]

    # 1) 模拟数据并行梯度平均
    global_grad = average_gradients(per_rank_grads)

    # 2) 模拟 ReduceScatter：把聚合后的全局梯度按 shard 切开
    grad_shards = split_by_sizes(global_grad, sizes)

    updated_param_shards = []

    # 3) 每个 rank 只更新自己负责的参数片和优化器状态
    for shard, grad_shard, state in zip(param_shards, grad_shards, state_shards):
        m = state["m"]
        v = state["v"]

        if len(m) != len(shard) or len(v) != len(shard):
            raise ValueError("state shard size mismatch")

        new_shard = []
        for i, (w, g) in enumerate(zip(shard, grad_shard)):
            m[i] = beta1 * m[i] + (1.0 - beta1) * g
            v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g)

            m_hat = m[i] / (1.0 - beta1 ** step)
            v_hat = v[i] / (1.0 - beta2 ** step)

            new_w = w - lr * m_hat / (math.sqrt(v_hat) + eps)
            new_shard.append(new_w)

        updated_param_shards.append(new_shard)

    # 4) 模拟 AllGather：把各 rank 更新后的参数片重新拼回完整参数
    full_param = flatten(updated_param_shards)
    return full_param, updated_param_shards


def main():
    # 两个 rank，各自拥有一半参数
    param_shards = [
        [1.0, 2.0],
        [3.0, 4.0],
    ]

    # 每个 rank 只保存自己那一片参数对应的 Adam 状态
    state_shards = [
        {"m": [0.0, 0.0], "v": [0.0, 0.0]},
        {"m": [0.0, 0.0], "v": [0.0, 0.0]},
    ]

    # 两个 rank 的本地梯度，长度都是完整参数长度
    per_rank_grads = [
        [0.1, -0.2, 0.3, -0.4],
        [0.3, -0.4, 0.5, -0.6],
    ]

    full_param, updated_param_shards = adam_zero_stage1_step(
        per_rank_grads=per_rank_grads,
        param_shards=param_shards,
        state_shards=state_shards,
        lr=0.01,
        step=1,
    )

    print("updated_param_shards =", updated_param_shards)
    print("full_param =", full_param)
    print("state_shards =", state_shards)

    # 聚合后平均梯度为 [0.2, -0.3, 0.4, -0.5]
    # Adam 第一步在这里会沿梯度符号方向移动
    assert len(full_param) == 4
    assert full_param[0] < 1.0
    assert full_param[1] > 2.0
    assert full_param[2] < 3.0
    assert full_param[3] > 4.0

    # 只有对应 shard 的优化器状态被维护
    assert state_shards[0]["m"][0] != 0.0
    assert state_shards[1]["v"][1] > 0.0

    print("sanity check passed")


if __name__ == "__main__":
    main()
```

如果运行这个脚本，输出会显示两件事：

1. 更新后的完整参数确实被重新拼回来了。
2. 每个 rank 只维护自己那一段参数对应的 `m` 和 `v`。

这个例子展示了 Stage 1 的核心事实：虽然完整梯度来自整个模型，但每个 rank 只维护自己那一段参数对应的优化器状态。也就是说，**计算仍然是“看全局”，存储变成了“各管一段”**。

真实工程例子里，8 张 V100 训练 GPT 类模型时，不开 ZeRO 时 Adam 状态可能单卡就占十几 GB；开 Stage 1 后，优化器状态按 8 卡分片，单卡压力显著下降，模型代码几乎不用改，往往就能把原本放不下的 batch 或更大 hidden size 跑起来。

---

## 工程权衡与常见坑

Stage 1 的优点是简单，但它不是“白拿显存”。工程上至少有四个常见权衡。

第一，**显存省在优化器状态，不省在参数和激活**。如果你的 profile 显示激活占大头，只开 Stage 1 基本不会解决问题。很多新手看到“ZeRO 能省显存”，就默认任何 OOM 都能解决，这是最常见误判。

第二，**吞吐不一定提升**。显存腾出来后，你可以增大 batch，这可能让有效吞吐上升；但单步通信也会更依赖网络。特别是 `reduce_bucket_size` 和 `allgather_bucket_size` 过小，会导致很多小包通信，GPU 在等 NCCL；过大则会抬高瞬时显存峰值，甚至让 overlap 失效。

第三，**收敛通常不变，但错误配置会伪装成“算法不稳定”**。也就是说，很多所谓“ZeRO-1 不稳定”的案例，本质上不是算法问题，而是实现和配置问题。

第四，**checkpoint 与恢复比普通 DP 更依赖元信息一致性**。因为状态是分片保存的，所以恢复时不仅要把数值读回来，还要保证“这一段状态到底对应哪一段参数”没有变。

典型故障包括：

| 问题 | 现象 | 根因 |
| --- | --- | --- |
| 分片顺序与扁平化顺序不一致 | loss 很快发散 | 梯度、参数、状态错位 |
| 混合精度 loss scale 不稳 | 间歇性 NaN / inf | 梯度下溢或溢出 |
| checkpoint 元信息不一致 | 恢复后 loss 突变 | shard 映射变化 |
| 梯度裁剪位置错误 | 梯度统计异常 | 用局部分片裁剪代替全局裁剪 |
| bucket 太小 | 吞吐剧烈波动 | 通信碎片过多 |
| bucket 太大 | 峰值显存异常上升 | 通信缓冲过大 |

其中“梯度裁剪位置错误”是一个很容易被忽略的问题。以全局范数裁剪（global norm clipping）为例，正确形式应是先基于全局梯度求：

$$
\|g\|_2 = \sqrt{\sum_i g_i^2}
$$

然后再统一缩放：

$$
g \leftarrow g \cdot \min\left(1, \frac{\tau}{\|g\|_2}\right)
$$

如果误把每个 shard 的局部范数分别裁剪，实际效果就不再等价于标准全局裁剪，收敛行为可能明显变化。

下面这个表可以帮助判断是否该停留在 Stage 1：

| 项目 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
| --- | --- | --- | --- |
| 切分对象 | 只切优化器状态 | 再加梯度 | 再加参数 |
| 每卡显存下降 | 中等 | 更明显 | 最大 |
| 通信复杂度 | 低到中 | 中 | 高 |
| 代码侵入性 | 低 | 中 | 较高 |
| 常见使用时机 | Adam 状态先爆 | 梯度也开始吃紧 | 参数本体放不下 |

实操上，判断配置是否健康，可以盯下面这些指标：

| 指标 | 应怎么看 | 异常时通常意味着什么 |
| --- | --- | --- |
| 单步耗时 | 是否稳定在一个窄区间 | bucket 不合适、网络阻塞 |
| `grad_norm` | 是否与未启用 ZeRO 的历史区间接近 | 裁剪、缩放或同步逻辑有问题 |
| loss 曲线 | 是否大体延续原有趋势 | 数值稳定性问题或 shard 错位 |
| 验证集指标 | 同 batch size 下是否同量级 | 实现与普通 DP 不再等价 |
| 显存峰值 | 是否主要下降在 optimizer/state 部分 | 你的瓶颈也许不是优化器状态 |

如果这几项都正常，Stage 1 通常是“最省心的第一层显存优化”。

还有一个认知误区值得单独点出。有人会问：既然参数还是全量复制，为什么更新后还要 `AllGather`？答案取决于具体实现。有的实现会在一个优化步结束后把参数重新同步成完整副本；也有实现通过更细粒度的方式维护参数可见性。你在日志里看到的通信原语，不一定与教材图示完全一一对应。**判断 ZeRO-1 是否生效，应该看显存分布、状态切分和实际 trace，而不是只看日志里是否出现某个关键词。**

---

## 替代方案与适用边界

如果 Stage 1 不够，下一步不是盲目继续堆配置，而是先确认瓶颈在哪。可以先问三个问题：

1. 显存主要花在参数、梯度、优化器状态，还是激活？
2. 你的瓶颈发生在静态占用，还是某个前向/反向峰值？
3. 你更愿意接受通信开销，还是重计算开销？

不同方案解决的问题不同：

| 方案 | 解决什么问题 | 代价 |
| --- | --- | --- |
| ZeRO-1 | 优化器状态冗余 | 需要分片通信 |
| ZeRO-2 | 再解决梯度冗余 | 通信更复杂 |
| ZeRO-3 | 再解决参数冗余 | 运行时 gather/scatter 更频繁 |
| Activation Checkpoint | 解决激活占用 | 用重计算换显存 |
| Offload | 把状态挪到 CPU/NVMe | 带宽和延迟更差 |

三种 ZeRO 的显存公式可以并列看：

| Stage | 每卡显存 | 主要解决的瓶颈 |
| --- | --- | --- |
| ZeRO-1 | $AP + GP + \frac{SP}{D}$ | 优化器状态 |
| ZeRO-2 | $AP + \frac{GP}{D} + \frac{SP}{D}$ | 梯度 + 优化器状态 |
| ZeRO-3 | $\frac{AP + GP + SP}{D}$ | 参数 + 梯度 + 优化器状态 |

这个表的意义，不只是记公式，而是帮助你判断为什么 Stage 1 有时“看起来没什么用”。

举个简单判断例子。假设某次训练里每卡显存大致构成如下：

| 项目 | 占用 |
| --- | --- |
| 参数 | 10 GB |
| 梯度 | 10 GB |
| 优化器状态 | 20 GB |
| 激活 | 14 GB |

总计 54 GB。如果 8 卡启用 ZeRO-1，则优化器状态会变成：

$$
20 / 8 = 2.5 \text{ GB}
$$

每卡总占用理论上约变成：

$$
10 + 10 + 2.5 + 14 = 36.5 \text{ GB}
$$

这时收益很明显。

但如果原始构成是：

| 项目 | 占用 |
| --- | --- |
| 参数 | 10 GB |
| 梯度 | 10 GB |
| 优化器状态 | 8 GB |
| 激活 | 28 GB |

那么 8 卡 ZeRO-1 后，优化器状态只从 8 GB 降到 1 GB，总占用只是从 56 GB 降到 49 GB。你还是很可能 OOM，因为主因其实是激活。这个时候更有效的动作通常是 activation checkpoint、缩短序列长度、减小 micro-batch，或者进一步上更高阶段的 ZeRO。

因此适用边界很清楚：

- 如果模型参数和激活都还放得下，只是 Adam 状态太重，优先 Stage 1。
- 如果 batch 稍一增大就被梯度顶满，考虑 Stage 2。
- 如果连参数本体都放不进单卡，Stage 3 才是正解。
- 如果激活是大头，即使上 Stage 3，也常要叠加 activation checkpoint。
- 如果 GPU 显存极小但 CPU 内存充足，可考虑 offload，但吞吐通常会更差。

对零基础读者来说，可以把三阶段理解成：

| 阶段 | 直观理解 |
| --- | --- |
| Stage 1 | 先拆“训练历史记录” |
| Stage 2 | 再拆“梯度缓存” |
| Stage 3 | 最后连“模型本体”也拆 |

Stage 1 之所以常被先采用，不是因为它最强，而是因为它在“收益、复杂度、兼容性”三者之间最平衡。它通常是 ZeRO 体系里最容易上线、最容易验证、也最不容易把系统复杂度一下子抬高的一步。

---

## 参考资料

- DeepSpeed ZeRO 教程与配置指南: https://www.deepspeed.ai/tutorials/zero/
- DeepSpeed ZeRO 文档: https://deepspeed.readthedocs.io/en/stable/zero3.html
- DeepSpeed 配置文档: https://www.deepspeed.ai/docs/config-json/
- ZeRO 论文《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》: https://arxiv.org/abs/1910.02054
- Damek STAT 4830 笔记，ZeRO/ReduceScatter/AllGather 说明: https://damek.github.io/STAT-4830/section/12/notes.html
- EmergingMind，Zero Redundancy Optimizer 概述: https://www.emergentmind.com/topics/zero-redundancy-optimizer-zero
- DeepWiki，DeepSpeed Zero Optimizer 实现与 Stage 对比: https://deepwiki.com/deepspeedai/DeepSpeed/3.3-sequence-parallelism
- Lewis Won 对数据并行与 ZeRO 的讲解: https://dev.to/lewis_won/data-parallelism-4g3m
- PyTorch Distributed 文档，集合通信原语说明: https://pytorch.org/docs/stable/distributed.html
- DeepSpeed GPT/Megatron-LM 训练示例: https://www.deepspeed.ai/tutorials/zero/
