## 核心结论

ZeRO-3 和模型并行里的 Tensor Parallel，简称 TP，解决的是同一件事的不同侧面：前者主要解决“模型状态放不下”，后者主要解决“单层矩阵太大、单卡算不完”。

先把两个概念拆开：

- **ZeRO-3** 主要切的是训练状态，包括参数、梯度、优化器状态。
- **TP** 主要切的是层内计算，把同一个线性层或注意力投影拆到多张卡上并行算。

所以它们的第一性问题并不相同。

- 当你遇到的是“参数、梯度、Adam 状态加起来显存爆掉”，优先看 ZeRO-3。
- 当你遇到的是“某一层矩阵乘法太大，单卡算不动或太慢”，优先看 TP。
- 真正的大模型训练里，两者经常组合使用，但决策顺序通常不是一起上，而是先解决最硬的约束。

以 GPT-3 175B 为例，先看最容易算的一项：仅 FP16 参数本体就是约

$$
175\times10^9 \times 2 \text{ Byte} \approx 350 \text{ GB}
$$

这 350 GB 只是**参数本体**，不是完整训练状态。若做 8 卡纯 TP，每卡大约持有：

$$
350 / 8 = 43.75 \text{ GB}
$$

这里要先纠正一个常见误解：**43.75 GB 对应的是参数分片，不是完整优化器状态。**  
训练时真正吃显存的是“参数 + 梯度 + 优化器状态 + 可能的 master weight + 激活”，总量通常是 TB 级，而不是几百 GB。

一个常见的 Adam 训练口径可以这样估：

| 项目 | 常见精度/副本 | 近似倍数（相对参数本体） | 说明 |
|---|---|---:|---|
| 参数 | FP16/BF16 | 1x | 前向和反向直接使用 |
| 梯度 | FP16/BF16 | 1x | 反向得到 |
| Master weight | FP32 | 2x | 混合精度训练常见 |
| Adam 一阶矩 `m` | FP32 | 2x | 优化器状态 |
| Adam 二阶矩 `v` | FP32 | 2x | 优化器状态 |
| 合计 |  | 8x | 仅模型状态，不含激活 |

因此，若只看模型状态，175B 参数在这个口径下大约是：

$$
350\text{ GB} \times 8 = 2800\text{ GB} = 2.8\text{ TB}
$$

若采用另一种更保守或更优化的实现口径，也常看到约 **2.1 TB**。所以公开材料里常见“2.1 TB 到 2.8 TB”并不矛盾，它反映的是**优化器实现、精度副本、是否保留 master weight** 等差异。

ZeRO-3 的核心价值，是把参数、梯度、优化器状态都沿数据并行维度切片。若按常见的约 2.1 TB 口径估算，64 卡 ZeRO-3 后每卡平均约：

$$
2.1\text{ TB}/64 \approx 33\text{ GB}
$$

这时单卡需要长期持有的状态压力，明显低于只靠 TP。

代价是通信更多。因为 ZeRO-3 不是把完整参数永久留在每张卡上，而是**用到哪一层，就临时把这一层参数聚过来；算完，再把梯度切回去**。公开材料里常见结论是：**ZeRO-3 相比普通数据并行会多出约 50% 通信量**，而在很多训练配置下，它的跨卡通信压力也会高于同规模 TP。

一个实用判断是：  
在 A100 级硬件、$h=12288,\ seq=2048,\ mbs=1,\ DP=64,\ TP=8$ 这组常见 GPT-3 配置下，ZeRO-3 想把通信较好地隐藏在计算后面，带宽门槛大约是 **74 GB/s**；TP 的对应门槛大约是 **89 GB/s**。因此：

| 网络带宽区间 | 更容易保持吞吐的方案 | 主要原因 |
|---|---|---|
| `< 74 GB/s` | 两者都容易被通信卡住 | 网络明显跟不上计算，通信几乎一定暴露在关键路径上 |
| `74-89 GB/s` | TP 往往更稳 | TP 通信更少，更容易被计算覆盖 |
| `> 89 GB/s` | ZeRO-3 更有吸引力 | 内存优势开始兑现，吞吐损失更容易被接受 |

这张表不要读成“TP 永远更快”或“ZeRO-3 永远更省”。它表达的是更窄的判断：

- 在**同一模型规模**、**同一超参数**、**同一硬件代际**下，
- 当网络较弱时，通信更少的方案更容易保吞吐；
- 当网络足够强时，显存更省的方案更容易换来更大 batch、更深流水线或更稳定的可训练性。

玩具例子可以这样理解：8 卡 TP 像把一本厚书按页拆成 8 份，每个人一直只保管自己负责的页；ZeRO-3 则像把书、批注、草稿都拆成很多份，谁要看哪一章，就临时把那一章借过来。前者长期持有更少的“层内内容”，后者长期持有更少的“训练状态”，但借还次数更多。

---

## 问题定义与边界

本文只比较训练场景里的两类成本：

1. 显存占用  
显存占用，就是每块 GPU 上必须常驻或阶段性驻留的参数、梯度、优化器状态、激活等内容。

2. 通信开销  
通信开销，就是 GPU 之间交换参数、激活或梯度所花的时间。

这两个成本会共同决定训练能否成立：

- 显存不够，训练直接起不来。
- 显存够但通信太重，训练能跑，但吞吐会很差。

边界先讲清楚。

第一，ZeRO-3 主要影响“模型状态怎么放”；TP 主要影响“单层怎么切”。两者不是互斥关系，工程上经常混用。  
更准确地说：

| 维度 | ZeRO-3 | TP |
|---|---|---|
| 核心切分对象 | 参数/梯度/优化器状态 | 单层张量与层内计算 |
| 主要收益 | 降低状态常驻显存 | 让超大矩阵能分布式计算 |
| 主要代价 | 每层参数聚合与梯度切回 | 层内同步频繁 |
| 是否影响单层可计算性 | 弱 | 强 |

第二，本文聚焦预训练或大规模继续训练，不讨论推理时的 KV Cache 主导场景。KV Cache 是推理过程中保存历史 token 中间结果的缓存。  
推理时常见瓶颈是：

- 长上下文导致 KV Cache 变大；
- 单 token 解码导致算力利用率下降；
- 延迟而不是吞吐成为第一目标。

这些都不是本文要比较的重点。

第三，本文用“通信能否被计算覆盖”作为吞吐判断标准。若通信时间和计算时间的比值大于 1，说明通信已经站到关键路径上，训练速度会明显掉下来。

这个标准的含义对新手很重要。设：

$$
\rho = \frac{t_{\text{comm}}}{t_{\text{compute}}}
$$

那么：

| 比值 $\rho$ | 含义 | 训练表现 |
|---|---|---|
| $\rho \ll 1$ | 通信大多被计算覆盖 | 吞吐基本由算力决定 |
| $\rho \approx 1$ | 通信与计算接近 | 吞吐开始明显受网络影响 |
| $\rho > 1$ | 通信暴露在关键路径 | 增卡不增速，甚至降速 |

核心判断公式来自公开资料。

ZeRO-3 的层级通信覆盖条件可写成：

$$
\frac{t_{\text{comm}}^{\text{ZeRO}}}{t_{\text{compute}}}
=
\frac{1}{2\cdot seq \cdot mbs}\cdot \frac{DP-1}{DP}\cdot \frac{peak\_flops}{peak\_bw}
\le 1
$$

TP 的对应条件可写成：

$$
\frac{t_{\text{comm}}^{\text{TP}}}{t_{\text{compute}}}
=
\frac{TP-1}{2h}\cdot \frac{peak\_flops}{peak\_bw}
\le 1
$$

这两个式子告诉我们一个关键差别：

- ZeRO-3 是否划算，和 $seq,\ mbs,\ DP$ 强相关。
- TP 是否划算，主要和隐藏维度 $h$、TP 度以及带宽相关。

也就是说：

- **序列长、微批大**时，ZeRO-3 更容易把通信摊薄。
- **隐藏维度大**时，TP 更容易成立，因为每次同步相对计算量更小。
- 这也是为什么同样一套并行策略，换了模型尺寸或 batch 配置，结论会变。

---

## 核心机制与推导

先看 ZeRO-3。

ZeRO-3 的做法是：当前层计算前先把这一层参数 `AllGather` 到本卡，反向后再把梯度 `ReduceScatter` 回去。`AllGather` 就是各卡把各自分片拼成完整张量，`ReduceScatter` 就是先做求和再重新切片分发。这样单卡不必长期保存完整参数，但每层都要借参数、还梯度。

对新手来说，可以把一次训练 step 中的 ZeRO-3 想成下面这个时序：

| 阶段 | 本卡上有什么 | 发生什么通信 |
|---|---|---|
| 初始化后常驻 | 参数分片、梯度分片、优化器状态分片 | 无 |
| 某层前向前 | 还没有该层完整参数 | `AllGather` 聚齐这一层参数 |
| 某层前向/反向 | 该层完整参数短暂驻留 | 无或少量重叠通信 |
| 某层反向后 | 生成该层梯度 | `ReduceScatter` 把梯度切回去 |
| step 更新后 | 只保留分片状态 | 继续下一层或下一步 |

这个机制的核心收益是：**完整参数不是长期常驻，而是按层临时出现。**

公开推导里，单个 Transformer block 的参数规模常近似写成 $16h^2$。这个近似来自典型 GPT 类 block 的主项：

- 自注意力中的 `Q/K/V/O` 投影，量级为若干个 $h \times h$
- MLP 中两个大线性层，量级也主要是 $h^2$
- 省略 LayerNorm、bias、embedding 等低阶项后，可用 $16h^2$ 抓主量级

于是 ZeRO-3 的通信时间可近似写成：

$$
t_{\text{comm}}^{\text{ZeRO}}
=
16h^2 \cdot \frac{DP-1}{DP\cdot peak\_bw}
$$

直觉上，这个式子表达的是：

- 层越大，通信越大；
- DP 组越大，通信分摊越多，但并不完全线性；
- 带宽越高，通信时间越低。

对应一层前向计算时间近似为：

$$
t_{\text{compute}}^{\text{ZeRO}}
=
\frac{32\cdot seq \cdot mbs \cdot h^2}{peak\_flops}
$$

把两者相除，$h^2$ 被约掉，得到：

$$
\frac{t_{\text{comm}}^{\text{ZeRO}}}{t_{\text{compute}}}
=
\frac{1}{2\cdot seq \cdot mbs}\cdot \frac{DP-1}{DP}\cdot \frac{peak\_flops}{peak\_bw}
$$

这说明 ZeRO-3 的通信压力，和“每次通信要服务多少 token 计算量”强相关。  
因此：

- `seq` 越长，单位通信上能覆盖的计算越多；
- `mbs` 越大，同一次参数聚合能服务更多样本；
- `DP` 越大，分片更细，但每层借参数的通信范围也更大。

再看 TP。

TP 的思路不同。它把一个线性层横向切开，每卡只算部分矩阵乘法，中间再做 `AllReduce` 或 `AllGather`。`AllReduce` 就是先聚合再把结果发回所有参与者。它的好处是参数和激活都能按层切开，坏处是同步点直接插在层内，比较难完全隐藏。

可以把 TP 的一次层内计算理解成：

1. 每张卡持有某个线性层的一部分权重；
2. 每张卡独立算出局部输出；
3. 由于下一步往往需要完整结果，卡之间要同步；
4. 同步完成后，才能继续后续层。

所以 TP 的同步不像 ZeRO-3 那样主要发生在“层与层之间借参数”，而是更靠近“层内部的必经步骤”。这也是它对节点内高速互联更敏感的原因。

TP 的一个常见近似式是：

$$
t_{\text{comm}}^{\text{TP}}
=
\frac{seq \cdot mbs \cdot h}{TP}\cdot \frac{TP-1}{peak\_bw}
$$

而下一线性层计算时间近似为：

$$
t_{\text{compute}}^{\text{TP}}
=
\frac{2\cdot seq \cdot mbs \cdot h^2}{TP\cdot peak\_flops}
$$

两者相除后，$seq$ 和 $mbs$ 被约掉，只剩：

$$
\frac{t_{\text{comm}}^{\text{TP}}}{t_{\text{compute}}^{\text{TP}}}
=
\frac{TP-1}{2h}\cdot \frac{peak\_flops}{peak\_bw}
$$

这个式子比 ZeRO-3 更“稳定”，因为它不直接依赖微批大小。  
含义是：

- 隐藏维度 $h$ 越大，计算越厚，TP 越容易覆盖通信；
- TP 度越高，同步参与者越多，通信更重；
- 带宽不足时，TP 组不能盲目跨节点扩张。

这里有一个很重要的新手误区：  
**“TP 把参数切了，所以显存一定比 ZeRO-3 省。”**  
这句话不成立，因为 TP 切的是层内参数与计算，不会像 ZeRO-3 那样把**优化器状态和梯度**一起按 DP 维切到很细。训练里真正决定能否起跑的，经常不是“参数本体”，而是“全套状态”。

下面用同一组参数把这件事具体算出来。

真实工程例子：取 GPT-3 175B 常见配置 $h=12288,\ seq=2048,\ mbs=1,\ DP=64,\ TP=8$，A100 峰值算力约 312 TFLOPs。代入后：

- ZeRO-3 带宽门槛约为 74 GB/s
- TP 带宽门槛约为 89 GB/s

推导过程如下。

对于 ZeRO-3，令覆盖条件取等号：

$$
\frac{1}{2\cdot seq \cdot mbs}\cdot \frac{DP-1}{DP}\cdot \frac{peak\_flops}{peak\_bw}=1
$$

解得：

$$
peak\_bw
=
\frac{peak\_flops}{2\cdot seq\cdot mbs}\cdot \frac{DP-1}{DP}
$$

代入 $peak\_flops=312000\ \text{GB/s 等价口径},\ seq=2048,\ mbs=1,\ DP=64$：

$$
peak\_bw \approx \frac{312000}{4096}\times \frac{63}{64}\approx 74.8 \text{ GB/s}
$$

对于 TP，同理：

$$
\frac{TP-1}{2h}\cdot \frac{peak\_flops}{peak\_bw}=1
$$

解得：

$$
peak\_bw
=
peak\_flops}\cdot \frac{TP-1}{2h}
$$

代入 $h=12288,\ TP=8$：

$$
peak\_bw \approx 312000\times \frac{7}{24576}\approx 88.9 \text{ GB/s}
$$

所以在 80 GB/s 左右网络上，经常出现“ZeRO-3 更省显存，但 TP 吞吐更高”的交叉现象。  
原因不是 ZeRO-3 算法更差，而是这组超参数下，它更难把额外通信完全藏在计算后面。

为了避免把结论读死，可以再看一张变化趋势表：

| 配置变化 | 对 ZeRO-3 的影响 | 对 TP 的影响 |
|---|---|---|
| 增大 `seq` | 更有利 | 近似不变 |
| 增大 `mbs` | 更有利 | 近似不变 |
| 增大 `DP` | 分片更细，但通信范围更大 | 无直接影响 |
| 增大 `h` | 主公式里约掉，影响较弱 | 更有利 |
| 增大 `TP` | 无直接影响 | 通信更重 |
| 提升网络带宽 | 更有利 | 更有利 |

所以不要把“74 vs 89 GB/s”理解成普适常数。它只是这组 GPT-3 典型配置下的工程分界线。

---

## 代码实现

下面先给一个可运行的 Python 小脚本，用同一组超参数估算显存和带宽门槛。脚本只依赖 Python 标准库，直接 `python3 script.py` 即可运行。

```python
from dataclasses import dataclass


@dataclass
class TrainingStateEstimate:
    params_gb: float
    grads_gb: float
    master_weights_gb: float
    adam_m_gb: float
    adam_v_gb: float

    @property
    def total_gb(self) -> float:
        return (
            self.params_gb
            + self.grads_gb
            + self.master_weights_gb
            + self.adam_m_gb
            + self.adam_v_gb
        )


def fp16_param_gb(params_billion: float) -> float:
    # 1e9 parameters * 2 bytes / 1e9 -> GB
    return params_billion * 2.0


def adam_training_state_gb(
    params_billion: float,
    use_master_weights: bool = True,
) -> TrainingStateEstimate:
    params_gb = fp16_param_gb(params_billion)      # FP16/BF16 params
    grads_gb = params_gb                           # FP16/BF16 grads
    master_weights_gb = params_gb * 2 if use_master_weights else 0.0  # FP32
    adam_m_gb = params_gb * 2                      # FP32 first moment
    adam_v_gb = params_gb * 2                      # FP32 second moment
    return TrainingStateEstimate(
        params_gb=params_gb,
        grads_gb=grads_gb,
        master_weights_gb=master_weights_gb,
        adam_m_gb=adam_m_gb,
        adam_v_gb=adam_v_gb,
    )


def zero3_bw_threshold_gbps(
    seq: int,
    mbs: int,
    dp: int,
    peak_flops_tflops: float = 312.0,
) -> float:
    # Use the same unit convention as the common engineering derivation:
    # TFLOPs -> "GB/s equivalent" by multiplying 1000.
    peak_flops = peak_flops_tflops * 1000.0
    return peak_flops * ((dp - 1) / dp) / (2 * seq * mbs)


def tp_bw_threshold_gbps(
    h: int,
    tp: int,
    peak_flops_tflops: float = 312.0,
) -> float:
    peak_flops = peak_flops_tflops * 1000.0
    return peak_flops * ((tp - 1) / (2 * h))


def zero3_state_per_gpu_gb(total_state_gb: float, dp: int) -> float:
    return total_state_gb / dp


def tp_param_per_gpu_gb(total_param_gb: float, tp: int) -> float:
    return total_param_gb / tp


if __name__ == "__main__":
    params_billion = 175
    state = adam_training_state_gb(params_billion=params_billion, use_master_weights=True)

    gpt3_params_gb = fp16_param_gb(params_billion)
    tp8_param_per_gpu_gb = tp_param_per_gpu_gb(gpt3_params_gb, tp=8)

    # Two common reporting calibers in public material:
    zero3_state_per_gpu_21tb_gb = zero3_state_per_gpu_gb(total_state_gb=2100, dp=64)
    zero3_state_per_gpu_28tb_gb = zero3_state_per_gpu_gb(total_state_gb=2800, dp=64)

    z_bw = zero3_bw_threshold_gbps(seq=2048, mbs=1, dp=64)
    t_bw = tp_bw_threshold_gbps(h=12288, tp=8)

    assert round(gpt3_params_gb, 1) == 350.0
    assert round(tp8_param_per_gpu_gb, 2) == 43.75
    assert round(zero3_state_per_gpu_21tb_gb, 2) == 32.81
    assert round(zero3_state_per_gpu_28tb_gb, 2) == 43.75
    assert 74.0 < z_bw < 75.0
    assert 88.0 < t_bw < 90.0

    print("=== GPT-3 175B memory estimate ===")
    print(f"FP16 params only: {gpt3_params_gb:.2f} GB")
    print(f"TP=8 param shard per GPU: {tp8_param_per_gpu_gb:.2f} GB")
    print(f"Adam state total (with FP32 master weights): {state.total_gb:.2f} GB")
    print(f"ZeRO-3 per GPU if total state is 2.1 TB and DP=64: {zero3_state_per_gpu_21tb_gb:.2f} GB")
    print(f"ZeRO-3 per GPU if total state is 2.8 TB and DP=64: {zero3_state_per_gpu_28tb_gb:.2f} GB")
    print()
    print("=== Bandwidth threshold estimate ===")
    print(f"ZeRO-3 threshold: {z_bw:.2f} GB/s")
    print(f"TP threshold: {t_bw:.2f} GB/s")
```

这段脚本有三个作用：

1. 把“350 GB 只是参数，不是全部训练状态”明确算出来。
2. 把 2.1 TB 和 2.8 TB 两种常见口径同时列出来，避免数字看起来互相冲突。
3. 把 ZeRO-3 与 TP 的带宽门槛放在同一个脚本里，便于做参数敏感性实验。

如果想自己试变化，只需要改这几个值：

- `params_billion`
- `seq`
- `mbs`
- `dp`
- `h`
- `tp`
- `peak_flops_tflops`

例如：

- 把 `mbs=1` 改成 `mbs=4`，你会看到 ZeRO-3 的带宽门槛明显下降；
- 把 `tp=8` 改成 `tp=16`，你会看到 TP 的带宽门槛上升；
- 把 `h=12288` 改得更大，TP 更容易覆盖通信。

DeepSpeed 里开启 ZeRO-3 的最小配置通常长这样：

```json
{
  "train_batch_size": 256,
  "gradient_accumulation_steps": 16,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "none" },
    "offload_param": { "device": "none" },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true
  }
}
```

这份配置的含义可以顺手解释一下：

| 配置项 | 作用 | 新手常见误解 |
|---|---|---|
| `stage: 3` | 参数、梯度、优化器状态都切分 | 不是“只切优化器” |
| `overlap_comm: true` | 尝试把通信和计算重叠 | 开了不代表一定能完全隐藏 |
| `reduce_scatter: true` | 梯度回传时先归约再切片 | 不是额外功能，常是关键路径优化 |
| `offload_*: none` | 不把状态卸载到 CPU/NVMe | 节省 PCIe/存储往返延迟 |
| `contiguous_gradients: true` | 减少碎片、优化通信布局 | 不是数学层面的并行策略变化 |

Megatron 或 NeMo 里配置 TP/PP 时，关键参数通常是：

| 目标 | 典型配置项 | 含义 |
|---|---|---|
| 开 TP | `--tensor-model-parallel-size 4/8` | 把层内矩阵切到多卡 |
| 开 SP | `--sequence-parallel` | 把部分激活沿序列维切分，降低激活显存 |
| 开 PP | `--pipeline-model-parallel-size 4/8/16` | 把不同层段放到不同 GPU/节点 |
| 开 ZeRO-3/FSDP | `--data-parallel-sharding-strategy optim_grads_params` | 对完整训练状态分片 |

如果要混合使用，经验上通常是：**先用 ZeRO-3 保证状态放得下，再把 TP 限定在单机或少数节点内。**

原因不是“TP 不重要”，而是更基本：

- 没有 ZeRO-3/FSDP 时，很多模型根本起不来；
- TP 跨节点太深时，层内同步容易直接成为瓶颈；
- 因此更稳的做法是先保可训练性，再保扩展效率。

---

## 工程权衡与常见坑

最常见的权衡不是“哪个更先进”，而是“你的网络和显存先到哪条红线”。

先给一个决策视角：

| 你先撞到的限制 | 优先方案 | 原因 |
|---|---|---|
| 状态显存爆掉 | ZeRO-3/FSDP | 直接减少常驻状态 |
| 单层算不下或算太慢 | TP | 直接切层内矩阵 |
| 节点很多、层数也很深 | PP + 低度 TP/ZeRO-3 | 分摊层深和状态 |
| 网络较弱 | 低度 TP 或更保守的并行组合 | 减少高频跨卡同步 |

再看常见坑：

| 常见坑 | 现象 | 为什么会这样 | 补救方式 |
|---|---|---|---|
| 低带宽上直接开 ZeRO-3 | 显存够了，但 tokens/s 明显掉 | 每层都要借参数、还梯度，通信过密 | 降 DP、增 PP，或先回到 TP-only/低阶 sharding |
| TP 跨节点拉太深 | AllReduce 成为瓶颈 | 层内同步进入关键路径，且跨节点延迟高 | 把 TP 收敛到 4 到 8，优先放在 NVLink/NVSwitch 域内 |
| 只看参数大小，不看训练状态 | 以为 350 GB 就是全部成本 | 忽略梯度、master weight、Adam `m/v` | 单独核算参数、梯度、优化器状态 |
| TP 不开 sequence parallel | 激活显存偏高 | 参数切了，但序列维激活仍大 | 只要开 TP，通常就顺手开 SP |
| ZeRO-3 和 PP 叠加不当 | 微批太小，通信摊不薄 | 流水线切得深后，每卡计算块过小 | 提高 gradient accumulation，调大可承受微批 |
| 只追求更高并行度 | 卡数变多但吞吐不升反降 | 通信、流水线气泡、负载不均开始主导 | 先做 profiling，再决定是否继续扩并行度 |

这里再补一条很关键的工程经验：

- **ZeRO-3 解决的是“装不下”**
- **TP 解决的是“切不开”**
- **PP 解决的是“层太多，单机放不下或调度不平衡”**

如果你把问题判断错了，后面的调参会非常低效。  
例如：

- 模型状态已经明显超显存，却只尝试增 TP，往往会发现参数虽然分了，但优化器状态仍然压爆；
- 单层 GEMM 已经大到算不动，却只上 ZeRO-3，往往会发现状态能放下了，但层内计算和同步模式并没有根本改变。

一个对新手很有用的经验线是：  
**80 到 90 GB/s 可以当成 ZeRO-3 与 TP 的粗略分界带。**  
低于这条线，优先怀疑通信；高于这条线，才更有资格讨论 ZeRO-3 释放出来的激活空间能否换来更大微批和更高吞吐。

这条经验线的正确读法是：

- 它不是协议标准，也不是硬性阈值；
- 它只是对 A100 级硬件和 GPT-3 一类配置的实用工程近似；
- 真正上线前，仍然要结合 profiler 看 `AllGather`、`ReduceScatter`、`AllReduce` 的占比。

如果想快速做一轮排障，顺序通常是：

1. 先看单卡峰值显存，确认到底是状态爆掉还是激活爆掉。
2. 再看 NCCL 通信占比，确认是 ZeRO-3 的分层聚合重，还是 TP 的层内同步重。
3. 再看 TP 组是否跨节点、PP 是否导致微批过小。
4. 最后才决定是调并行度、调 batch，还是换网络拓扑。

---

## 替代方案与适用边界

并行策略不是二选一，而是组合题。

| 方案 | 适用场景 | 资源前提 | 主要缺点 |
|---|---|---|---|
| TP-only | 节点内高速互联强，模型层太宽 | NVLink/NVSwitch 强 | 跨节点后很容易被 AllReduce 卡住 |
| ZeRO-3 + DP | 状态放不下，但网络还可以 | 中高带宽网络 | 每层借参数，通信更频繁 |
| TP + PP | 超大模型、层数深 | 节点多，调度复杂度可接受 | 有流水线气泡，调参更难 |
| ZeRO-3 + TP | 既缺显存又缺单层容量 | 至少节点内高带宽 | 组网复杂，调优成本高 |
| ZeRO-3 + TP + PP | 100B 以上大模型常见终局 | 大规模集群 | 工程复杂度最高 |

为了避免这张表过于抽象，可以把几种方案理解为不同的“切分方向”：

| 并行方式 | 切什么 | 典型收益 | 典型代价 |
|---|---|---|---|
| DP | 切样本 | 扩吞吐简单 | 每卡保留完整状态 |
| ZeRO-3/FSDP | 切训练状态 | 显著降状态显存 | 参数借还更频繁 |
| TP | 切层内张量 | 让大矩阵可分布计算 | 层内同步强依赖高带宽 |
| PP | 切层段 | 分摊模型深度 | 有流水线气泡与调度复杂度 |
| SP | 切部分序列维激活 | 降激活显存 | 依赖具体实现与配套通信 |

实践里比较稳的顺序通常是：

1. 先用 ZeRO-3 或 FSDP 把模型状态压到可训练范围。
2. 再在单节点内加低度 TP，优先 4 或 8。
3. 模型再变大时，用 PP 分摊层深。
4. 网络不够时，先降低 TP 跨节点范围，而不是盲目继续加 TP 度。

原因很简单：显存不够时训练根本起不来；通信稍差时，训练只是变慢。先解决“能不能跑”，再解决“能不能快”，这是更稳的工程顺序。

还可以把这个顺序进一步写成一套判断题：

| 问题 | 若答案为“是” | 优先动作 |
|---|---|---|
| 模型状态是否已经超单卡显存？ | 是 | 先上 ZeRO-3/FSDP |
| 单层矩阵是否已经大到单卡不适合算？ | 是 | 再引入 TP |
| 模型层数是否深到单节点难以承载？ | 是 | 再考虑 PP |
| TP 是否已经跨节点且通信过重？ | 是 | 收缩 TP 组，优先节点内 TP |
| 微批是否太小导致覆盖失败？ | 是 | 提高梯度累积或调整 PP 深度 |

因此，本文的核心边界可以总结成一句话：

- **ZeRO-3 更偏“内存治理”**
- **TP 更偏“算子切分”**
- **工程上先判定主要瓶颈，再决定组合顺序**

---

## 参考资料

| 来源 | 主题 | 链接 |
|---|---|---|
| Microsoft Research Blog | ZeRO 三阶段、线性内存收益、ZeRO-3 约 50% 额外通信 | [ZeRO & DeepSpeed](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) |
| Hugging Face Nanotron Ultra-Scale Playbook | ZeRO-3、TP、PP 的通信/计算比公式 | [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/main/ultra_blog.md) |
| NVIDIA NeMo User Guide | GPT-3 175B 预定义训练配置，TP/PP 组合示例 | [Training with Predefined Configurations](https://docs.nvidia.com/nemo-framework/user-guide/24.07/llms/gpt/trainingpredefined.html) |
| NVIDIA Megatron Core Guide | TP、PP、FSDP/ZeRO-3 的组合建议 | [Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/user-guide/parallelism-guide.html) |
| HPS/Glenn Klockwood 材料 | GPT-3 175B 参数本体约 350GB、训练状态放大到 TB 级的工程直觉 | [The Guts of Large Language Models](https://hps.vi4io.org/_media/events/2024/hpciodc24-glenn.pdf) |
| DeepSpeed Documentation | ZeRO 配置项、stage 语义与工程开关 | [DeepSpeed ZeRO Docs](https://deepspeed.readthedocs.io/en/latest/zero3.html) |
| Megatron-LM 论文与实现资料 | TP、PP、SP 的基本设计 | [Megatron-LM](https://arxiv.org/abs/1909.08053) |
