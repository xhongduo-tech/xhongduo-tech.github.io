## 核心结论

交错式流水线（interleaved pipeline parallelism）是 Megatron-LM 对普通流水线并行的一次再切分：每个物理设备不只负责一段连续层，而是负责 $v$ 个更小的“虚拟阶段”（virtual pipeline stages, VPP）。白话定义是：一张卡原本拿到一大段层，现在再拆成几小段，调度器按更细的粒度穿插前向和反向。

它要解决的问题是 pipeline bubble。bubble 的严格定义是：在流水线启动和收尾阶段，部分设备没有可执行的 microbatch，只能等待上下游，于是产生空转时间。白话解释是：GPU 已经参与这次训练，但某些时间片里没有真正算活。

在常见的平衡假设下，普通 1F1B（one-forward-one-backward）流水线的 bubble 比例近似为：

$$
\text{bubble fraction}\approx\frac{p-1}{m}
$$

其中：

- $p$ 是物理流水线阶段数，也就是参与 PP 的设备数。
- $m$ 是一个全局 batch 内拆出的微批次数（microbatches）。

引入每卡 $v$ 个虚拟阶段后，bubble 比例近似变为：

$$
\text{bubble fraction}^{\text{interleaved}}
\approx
\frac{p-1}{m\cdot v}
$$

结论很直接：交错式流水线用更高的通信频率，换更低的空闲时间。代价也很直接：逻辑上的 stage 总数从 $p$ 变成 $p\cdot v$，激活值在相邻 rank 之间传递得更频繁，点对点通信次数通常近似按 $v$ 增长。

| 配置 | bubble 比例 | 通信量相对普通流水线 |
| --- | --- | --- |
| 非交错，$v=1$ | $\frac{p-1}{m}$ | $1\times$ |
| 交错，$v>1$ | $\frac{p-1}{m\cdot v}$ | 约 $v\times$ |
| 极限细分，$v\approx\frac{L}{p}$ | 接近 0 | 最高，接近理论上限 |

这里的 $L$ 是总层数。如果把 $v$ 一路加到每个物理 stage 只剩 1 层或极少层，bubble 可以继续下降，但这通常也是通信最重、调度最复杂、对链路最挑剔的一档配置。

需要先强调一个容易被误读的点：bubble 下降并不等于训练吞吐按同样倍数上升。交错式流水线减少的是空转，不减少 FLOPs，也不减少参数同步成本。真正的吞吐改善取决于以下条件是否同时成立：

- 普通 PP 的 bubble 原本就大。
- 相邻设备之间的 P2P 链路足够强。
- 每个虚拟阶段切分后，计算负载仍然足够均衡。
- 调度器能够把新增通信与计算较好地重叠。

如果这些条件不成立，公式上的收益会被现实中的通信和调度开销抵消。

---

## 问题定义与边界

本文只讨论 Megatron-LM 中的 pipeline parallelism，尤其是 interleaved 1F1B schedule，也就是“交错式一前一后调度”。它不改变模型的数学定义，不改变 loss，也不改变梯度本身；它改变的是同一个 batch 在多卡上的执行顺序。

问题边界先说清楚：

1. 这里的主要收益来自减少 pipeline bubble，不是减少总计算量。
2. 这里的主要代价来自更频繁的 P2P 通信，不是参数量增加。
3. 这里讨论的是 stage 切分与调度，不是 tensor parallel 或 data parallel 本身。
4. 只有当流水线原本就存在明显空转时，交错式流水线才有意义。若 $m$ 已经远大于 $p$，普通流水线的 bubble 本来就不高，再继续增大 $v$ 往往得不偿失。
5. 文中的公式默认各 stage 计算时间大致平衡，并且暂时忽略通信与计算的复杂重叠细节。工程上如果层不均衡、序列长度波动大、网络较慢，真实收益会偏离理论值。

先看一个最小例子。设：

$$
p=4,\quad m=8
$$

普通流水线的 bubble 比例为：

$$
\frac{p-1}{m}=\frac{3}{8}=37.5\%
$$

若每张卡切成 $v=2$ 个虚拟阶段，则：

$$
\frac{p-1}{m\cdot v}=\frac{3}{16}=18.75\%
$$

若进一步切到 $v=4$，则：

$$
\frac{3}{32}=9.375\%
$$

同一组 $p,m$ 参数下，bubble 的确在变小，但通信不会免费消失。更准确地说，虚拟阶段越多，激活需要穿过的逻辑边界越多，通信调用次数也越多。

| $p=4,m=8$ 时的配置 | bubble 比例 | 逻辑 stage 数 | 相对通信量 |
| --- | --- | --- | --- |
| $v=1$ | 37.5% | $4$ | $1\times$ |
| $v=2$ | 18.75% | $8$ | 约 $2\times$ |
| $v=4$ | 9.375% | $16$ | 约 $4\times$ |

这个表已经说明交错式流水线的适用边界：它不是“总是更快”，而是“当 bubble 足够大、通信链路也足够强时更快”。

对新手更实用的判断方式是先问两个问题：

1. 当前训练到底是算力瓶颈，还是流水线空转瓶颈？
2. 你的节点内和跨节点链路，能不能承受更多激活传输？

如果这两个问题没有明确答案，先做 profile，再讨论是否开启虚拟阶段，比直接调参数更有效。

---

## 核心机制与推导

先统一符号。后面的公式和代码都用这一套定义。

- $L$：模型总层数。
- $p$：物理流水线阶段数，也就是 PP world size。
- $m$：一个 batch 拆出的微批次数。
- $v$：每个物理阶段内部再切出的虚拟阶段数。
- $t_f,t_b$：某个物理 stage 处理一个微批次时，前向和反向的总时间。
- $t_{comm}$：相邻 stage 之间一次激活或梯度传输的时间。
- 1F1B：完成若干 warmup 前向后，调度进入“做一次前向，做一次反向”的稳态。

先看普通流水线。把一个批次拆成 $m$ 个 microbatches 后，每个 microbatch 都要依次经过 $p$ 个 stage。由于第一张卡最先开始算，最后一张卡最晚拿到数据，所以一开始会有启动空泡；同理，在反向收尾时会有尾部空泡。

在最理想的平衡模型下：

- 启动阶段少了 $p-1$ 个“完整 stage 时间”。
- 收尾阶段同样少了 $p-1$ 个“完整 stage 时间”。

于是 bubble 时间可写成：

$$
t_{pb}=(p-1)(t_f+t_b)
$$

而不考虑 bubble 时，处理完 $m$ 个微批次的理想有效计算时间为：

$$
t_{id}=m(t_f+t_b)
$$

两者相除，得到普通 1F1B 的 bubble 比例：

$$
\frac{t_{pb}}{t_{id}}=\frac{p-1}{m}
$$

这个推导成立的关键是假设每个 stage 的计算开销差不多。如果某一段层特别重，例如 attention 和 MLP 切分不均衡，那么真实空转会比这个公式更差，因为流水线会被最慢 stage 决定。

### 为什么交错后会多一个 $v$

交错式流水线做的事情不是增加设备，而是把每个物理 stage 再拆成 $v$ 个更小的模型块。设原来每张卡负责 8 层，现在切成两个虚拟阶段，那么一张卡内部就可能变成：

- 虚拟阶段 0：前 4 层
- 虚拟阶段 1：后 4 层

调度器不再把整张卡看成一个粗粒度 stage，而是把它看成两个交替参与调度的逻辑块。这样做的后果是：

1. 每次执行的计算块变小了。
2. 前向与反向可以更频繁地交错。
3. 启动和收尾阶段的“空档长度”按块大小缩短。

如果单个物理 stage 的前向和反向时间分别是 $t_f,t_b$，在平衡切分下，单个虚拟块的时间近似变成：

$$
\frac{t_f}{v},\quad \frac{t_b}{v}
$$

此时启动和收尾仍然要跨越 $p-1$ 个物理 stage，但每次空转对应的是更小的块，因此 bubble 时间缩短为：

$$
t_{pb}^{\text{int}}=\frac{(p-1)(t_f+t_b)}{v}
$$

再除以理想计算时间 $t_{id}=m(t_f+t_b)$，得到：

$$
\text{bubble fraction}^{\text{int}}
=
\frac{t_{pb}^{\text{int}}}{t_{id}}
=
\frac{p-1}{m\cdot v}
$$

这就是 Megatron-LM 中虚拟阶段降低 bubble 的核心数学原因。

### 一个更完整的总时间视角

只看 bubble 容易误判。更接近真实工程的总时间写法是：

$$
T_{\text{step}}
\approx
T_{\text{compute}}
+
T_{\text{bubble}}
+
T_{\text{comm}}
-
T_{\text{overlap}}
$$

其中：

- $T_{\text{compute}}$ 由总 FLOPs 决定，交错本身不会减少它。
- $T_{\text{bubble}}$ 会随着 $v$ 增大而下降。
- $T_{\text{comm}}$ 往往会随着 $v$ 增大而上升。
- $T_{\text{overlap}}$ 表示通信与计算真正重叠掉的那部分时间，它决定了“新增通信有多少是显性的”。

因此，交错式流水线成立的必要条件不是“bubble 能降”，而是：

$$
\Delta T_{\text{bubble}} > \Delta T_{\text{comm}} - \Delta T_{\text{overlap}}
$$

如果右边更大，那么即使公式上 bubble 更小，实际吞吐也可能更差。

### 用 ASCII 图看调度直觉

下面只画前向方向，`A0/A1` 表示 GPU0 上的两个虚拟阶段，`B0/B1` 表示 GPU1 上的两个虚拟阶段。

```text
普通 PP（v=1）:
GPU0: F0 ---- F1 ---- F2 ---- F3
GPU1:      F0 ---- F1 ---- F2 ---- F3
GPU2:           F0 ---- F1 ---- F2 ---- F3
GPU3:                F0 ---- F1 ---- F2 ---- F3

交错 PP（v=2）:
GPU0: A0-F0 A1-F0 A0-F1 A1-F1 A0-F2 A1-F2 ...
GPU1:      B0-F0 B1-F0 B0-F1 B1-F1 B0-F2 B1-F2 ...
GPU2:           C0-F0 C1-F0 C0-F1 C1-F1 C0-F2 C1-F2 ...
GPU3:                D0-F0 D1-F0 D0-F1 D1-F1 D0-F2 D1-F2 ...
```

这个图表达的本质是：原本粗粒度的大块计算，现在拆成了细粒度的小块，流水线的“缝隙”更容易被后续工作填上。

### 一个具体层切分例子

设模型共有 $L=32$ 层，$p=4$，那么普通流水线下每个物理 stage 负责：

$$
\frac{L}{p}=\frac{32}{4}=8\text{ 层}
$$

如果再设每个虚拟阶段放 4 层，那么每个物理 stage 内部会被拆成：

$$
v=\frac{32}{4\times 4}=2
$$

此时每张卡不再只对应“8 层的一段”，而是对应“两个 4 层的小段”。这就是命令行参数 `num_layers_per_virtual_pipeline_stage=4` 背后的实际含义。

---

## 代码实现

Megatron-LM 暴露的关键参数是 `num_layers_per_virtual_pipeline_stage`。它的含义不是直接给出 $v$，而是指定“每个虚拟阶段放几层”。

若总层数为 `num_layers`，物理流水线阶段数为 `pp_size`，每个虚拟阶段层数为 `layers_per_virtual_stage`，那么：

$$
v=
\frac{\text{num\_layers}}
{\text{pp\_size}\times \text{layers\_per\_virtual\_stage}}
$$

这要求上式能整除，否则虚拟阶段就无法均匀切分。

对新手来说，这里最容易混淆的有两个量：

- `pp_size` 决定“有几张卡参与流水线”。
- `layers_per_virtual_stage` 决定“每张卡内部再切多细”。

也就是说，`pp_size` 控制物理 stage 数，`layers_per_virtual_stage` 间接控制虚拟 stage 数 $v$。

下面给出一个可以直接运行的 Python 例子。它做三件事：

1. 计算理论 bubble 比例。
2. 根据层数和切分规则计算虚拟阶段数。
3. 检查配置是否满足 Megatron 交错调度常见约束。

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class InterleavedConfig:
    num_layers: int
    pp_size: int
    layers_per_virtual_stage: int
    num_microbatches: int


def bubble_fraction(pp_size: int, num_microbatches: int, vp_size: int = 1) -> float:
    if pp_size < 1:
        raise ValueError("pp_size must be >= 1")
    if num_microbatches < 1:
        raise ValueError("num_microbatches must be >= 1")
    if vp_size < 1:
        raise ValueError("vp_size must be >= 1")
    return (pp_size - 1) / (num_microbatches * vp_size)


def virtual_pipeline_size(
    num_layers: int,
    pp_size: int,
    layers_per_virtual_stage: int,
) -> int:
    if num_layers <= 0:
        raise ValueError("num_layers must be > 0")
    if pp_size <= 0:
        raise ValueError("pp_size must be > 0")
    if layers_per_virtual_stage <= 0:
        raise ValueError("layers_per_virtual_stage must be > 0")

    denom = pp_size * layers_per_virtual_stage
    if num_layers % denom != 0:
        raise ValueError(
            "num_layers must be divisible by "
            "pp_size * layers_per_virtual_stage"
        )

    return num_layers // denom


def validate_interleaved_config(cfg: InterleavedConfig) -> int:
    if cfg.pp_size <= 2:
        raise ValueError(
            "Megatron interleaved schedule commonly requires pp_size > 2"
        )

    vp_size = virtual_pipeline_size(
        num_layers=cfg.num_layers,
        pp_size=cfg.pp_size,
        layers_per_virtual_stage=cfg.layers_per_virtual_stage,
    )

    # Megatron 的交错调度会要求 microbatch 与调度表对齐。
    # 实际规则随实现版本和 schedule 细节而变化，但下面这个检查
    # 可以作为常见配置的最小可用近似。
    if cfg.num_microbatches % cfg.pp_size != 0:
        raise ValueError(
            "num_microbatches should usually be a multiple of pp_size "
            "for interleaved scheduling"
        )

    return vp_size


def relative_comm_factor(vp_size: int) -> int:
    if vp_size < 1:
        raise ValueError("vp_size must be >= 1")
    return vp_size


def demo() -> None:
    # 玩具例子: p=4, m=8
    print("Toy example:")
    for v in (1, 2, 4):
        bf = bubble_fraction(pp_size=4, num_microbatches=8, vp_size=v)
        print(f"  v={v}: bubble={bf:.6f}, comm~{relative_comm_factor(v)}x")

    # 更接近真实配置的例子: 32 层模型, 4 个 PP stage, 每个虚拟阶段 4 层
    cfg = InterleavedConfig(
        num_layers=32,
        pp_size=4,
        layers_per_virtual_stage=4,
        num_microbatches=8,
    )
    vp = validate_interleaved_config(cfg)
    bf = bubble_fraction(cfg.pp_size, cfg.num_microbatches, vp)

    print("\nPractical example:")
    print(f"  virtual pipeline size = {vp}")
    print(f"  bubble fraction       = {bf:.6f}")
    print(f"  communication factor  = ~{relative_comm_factor(vp)}x")


if __name__ == "__main__":
    demo()
```

这段代码直接保存为 `vpp_demo.py` 后可以运行：

```bash
python3 vpp_demo.py
```

预期输出大致如下：

```text
Toy example:
  v=1: bubble=0.375000, comm~1x
  v=2: bubble=0.187500, comm~2x
  v=4: bubble=0.093750, comm~4x

Practical example:
  virtual pipeline size = 2
  bubble fraction       = 0.187500
  communication factor  = ~2x
```

这个例子虽然是玩具版，但已经对应了 Megatron 里的两类真实约束：

1. 层数必须能被虚拟阶段配置整除。
2. 微批次数必须能和交错调度表对齐。

再把它映射回 Megatron 的命令行参数：

```bash
--pipeline-model-parallel-size 4
--num-layers 32
--num-layers-per-virtual-pipeline-stage 4
```

这个配置的含义是：

- 一共 4 个物理 pipeline stages。
- 模型总共有 32 层。
- 每个虚拟阶段放 4 层。
- 所以每个物理 stage 原本的 8 层会再切成两个 4 层块，即 $v=2$。

为了更直观看懂这个映射，可以直接列成表：

| 参数 | 数值 | 含义 |
| --- | --- | --- |
| `num_layers` | 32 | 模型总层数 |
| `pipeline-model-parallel-size` | 4 | 物理 PP stage 数 |
| 每个物理 stage 层数 | 8 | $32/4$ |
| `num-layers-per-virtual-pipeline-stage` | 4 | 每个虚拟阶段放 4 层 |
| `v` | 2 | 每个物理 stage 被切成 2 段 |

如果你调参时发现“bubble 公式没问题，但程序直接报错”，通常先检查的就是上面这几项是否整除、是否对齐，而不是先怀疑训练框架本身。

---

## 工程权衡与常见坑

真实工程里，交错式流水线的瓶颈通常不是公式，而是链路、负载均衡和调度约束。

先把最重要的经验结论写清楚。Megatron-LM 相关论文和 NVIDIA 的技术文章都强调过同一件事：interleaved schedule 的收益来自减少流水线空泡，但它会增加通信频率，因此收益高度依赖硬件拓扑和 batch 配置。换句话说，“开 VPP 之后更快”不是静态结论，而是一个需要实测的条件结论。

在现代多机训练里，常见的并行组合是：

- 节点内优先使用 tensor parallelism，利用 NVLink 或 NVSwitch。
- 节点间用 pipeline parallelism 扩展模型容量。
- 如果再开启 VPP，就尽量让更重的 P2P 通信发生在更强的链路上。

原因很简单。VPP 增加的是相邻 stage 之间的激活传输频率。如果这些传输落在跨节点的慢链路上，理论上少掉的 bubble 可能会被网络阻塞全部吃回去。

### 常见坑

| 常见坑 | 现象 | 原因 | 规避方式 |
| --- | --- | --- | --- |
| `m` 不是 `p` 的整数倍 | 调度难以对齐，直接报错或吞吐异常 | 交错 schedule 需要按整轮展开 | 调整 global batch、micro batch 或 data parallel size，使微批次数满足约束 |
| `v` 设得过大 | 通信显著上升，吞吐反而下降 | bubble 虽降，但 P2P 次数增太多 | 从 `v=2` 或 `v=4` 起测，不要直接拉满 |
| 层数分配不整齐 | 某些 rank 层数异常，甚至无法构图 | `num_layers` 无法整除切分规则 | 确保 `num_layers % (p * layers_per_virtual_stage) == 0` |
| stage 负载不均衡 | 某张卡始终最慢，其他卡等待 | 各段层的算子开销不同 | 尽量让每个 stage 的实际计算时间接近，而不是只看层数相等 |
| 跨节点带宽不足 | GPU 利用率不高，但网络接近打满 | 交错放大了激活传输频率 | 优先让高频通信留在节点内，谨慎跨节点开启 VPP |
| 激活缓存压力增大 | 显存紧张，重算比例上升 | 更细粒度调度会改变激活驻留方式 | 联合 activation checkpointing 一起评估 |
| 只看理论 bubble | 纸面收益很好，真实吞吐一般 | 忽略通信和 overlap | 用 profiler 观察 step time、P2P、SM 利用率和网络占用 |

### 一个工程上更有用的判断表

| 观察现象 | 更可能的瓶颈 | 是否适合先尝试 VPP |
| --- | --- | --- |
| GPU 利用率低，网络也不满 | pipeline bubble 或 stage 失衡 | 适合，先测 `v=2` |
| GPU 利用率低，但网络已很高 | 通信瓶颈 | 不适合直接增大 `v` |
| 增大 microbatches 后吞吐明显改善 | bubble 确实偏大 | 可以继续评估 VPP |
| 增大 microbatches 后吞吐几乎不变 | 可能已不是 bubble 主导 | VPP 收益可能有限 |
| 单个 stage 明显慢于其他 stage | stage 不均衡 | 先做负载均衡，再谈 VPP |

### 一个容易误判的点

看到公式里 bubble 变成 $\frac{1}{v}$ 缩放，很容易误以为吞吐也会接近按 $v$ 提升。这通常不成立。

更现实的总时间关系是：

$$
T \approx T_{\text{compute}} + T_{\text{bubble}} + T_{\text{comm}}
$$

若再写得更完整一些：

$$
T \approx T_{\text{compute}} + T_{\text{bubble}} + T_{\text{comm}} - T_{\text{overlap}}
$$

交错式流水线减少的是 $T_{\text{bubble}}$，但常常增大 $T_{\text{comm}}$。如果系统本来就是通信受限，VPP 甚至会让总时间更差。

因此，工程上判断 VPP 是否值得开，最少要回答三个问题：

1. 当前 step time 中，bubble 占比到底多大？
2. 新增的 P2P 通信是否能被 overlap 掉？
3. 虚拟阶段切分后，各段计算是否仍然均衡？

只有这三个问题的答案大致偏正面，VPP 才更可能带来净收益。

---

## 替代方案与适用边界

最简单的替代方案就是不用交错，也就是 $v=1$。这不是落后配置，而是很多通信受限场景下的常见最优点。尤其在以下情况里，$v=1$ 往往更稳：

- 跨节点链路明显慢于节点内计算。
- 微批次已经足够多，原始 bubble 不大。
- 模型层数不容易均匀切分。
- 调试稳定性优先于极限吞吐。
- 训练系统里已经叠加了 TP、DP、重算和 ZeRO，调度复杂度很高。

另一端是极限细分，也就是接近：

$$
v=\frac{L}{p}
$$

此时每个虚拟阶段只有 1 层或极少层，bubble 最小，但通信最多。这类配置更适合以下环境：

- 节点内带宽极强。
- P2P 通信路径已经过优化。
- stage 负载较均衡。
- 目标是追求极限吞吐，而不是最稳的训练配置。

工程上更常见的是中间值，例如 `v=2` 或 `v=4`。原因很简单：它通常已经能削掉一大块 bubble，但不会把通信推到最极端。

| 选择 | 适用场景 | 主要收益 | 主要代价 |
| --- | --- | --- | --- |
| $v=1$ | 通信受限、调试期、链路一般 | 调度简单，通信最少 | bubble 较大 |
| $v=2$ | 多数生产训练的第一档尝试 | 常能明显减少空泡 | 通信上升，但通常可控 |
| $v=4$ | bubble 仍偏大且链路较强 | 进一步压缩空泡 | 调度与通信成本更明显 |
| $v\approx\frac{L}{p}$ | 带宽富余、强优化、追求极限吞吐 | bubble 最小 | 通信和复杂度最高 |

如果把选择过程写成一个简单决策顺序，通常是：

1. 先确认普通 PP 的 bubble 是否真的是瓶颈。
2. 再确认链路是否有余量承接更多 P2P。
3. 再检查层数能否整齐切分。
4. 最后从小 `v` 开始做 profile，而不是直接冲最大值。

对新手最实用的建议不是“默认开 VPP”，而是“默认把 VPP 当作一项可验证的优化”。它不是架构正确性的必要条件，只是当空泡足够明显时，一种常见而有效的吞吐优化手段。

---

## 参考资料

- Deepak Narayanan, Mohammad Shoeybi, Jared Casper, et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*. SC 2021.  
  支持本文关于 interleaved pipeline schedule、pipeline bubble 推导、虚拟阶段降低空泡的核心结论。  
  https://cs.stanford.edu/people/matei/papers/2021/sc_megatron_lm.pdf

- NVIDIA Technical Blog. *Scaling Language Model Training to a Trillion Parameters Using Megatron*.  
  支持本文关于交错调度的工程解释、吞吐提升依赖通信优化、DGX A100 类硬件上的实践经验。  
  https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/

- NVIDIA Megatron Core Documentation. `pipeline_parallel` / `schedules`.  
  支持本文关于 Megatron 具有 non-interleaved 与 interleaved 两类流水线调度入口、参数与 schedule 实现分离的描述。  
  https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/pipeline_parallel.html  
  https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.pipeline_parallel.schedules.html

- NVIDIA/Megatron-LM Repository.  
  支持本文关于 `num_layers_per_virtual_pipeline_stage`、pipeline model parallel 参数入口及实现背景的说明。  
  https://github.com/NVIDIA/Megatron-LM

- NVIDIA Megatron Core API Guide, pipeline schedules source docs.  
  用于对照交错与非交错 schedule 的调度函数组织方式，以及微批次调度约束在实现中的落点。  
  https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.pipeline_parallel.schedules.html
