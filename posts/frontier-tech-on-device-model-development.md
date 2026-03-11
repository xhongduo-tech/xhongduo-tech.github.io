## 核心结论

端侧模型，是指直接运行在手机、PC、车机或 IoT 设备上的模型，而不是把请求发到云端。它的发展主线不是“把云端大模型原样搬到本地”，而是围绕端侧资源约束做系统性裁剪：小模型架构优化、4bit 或 2bit 量化、蒸馏、KV Cache 压缩，以及面向 NPU 的算子适配。

这条路线的目标很明确：让 7B 以下模型在本地可用，优先满足隐私、离线可用、低交互延迟。这里的“低交互延迟”主要指首 token 延迟，即用户提交输入后，模型多久能给出第一个可见输出；端侧的优势，往往首先体现在首 token 不再受网络往返和云端排队影响。

一个新手可以先记住最小结论：不是所有模型都适合端侧，真正可用的端侧方案通常都围绕特定任务裁剪，而不是追求通用排行榜分数。原因也直接，端侧设备的瓶颈不是参数量一个指标，而是总内存、内存带宽、热设计功耗和算子兼容性共同决定。

可以先看一个最简单的体积估算。假设一个 7B 模型按 FP32 存储，则权重大约需要：

$$
7\times 10^9 \times 4 \approx 28\text{GB}
$$

这还只是权重，不包含 KV Cache、运行时缓冲区和操作系统占用，手机显然装不下。若改成 4bit 量化，则理论权重空间接近：

$$
7\times 10^9 \times 0.5 \approx 3.5\text{GB}
$$

再配合蒸馏、GQA 或 MQA、KV Cache 压缩和更适合端侧的小模型结构，本地聊天、轻量搜索、文档问答才开始变得可行。

这里要补一句常被忽略的话：4bit 不等于“直接快 8 倍”。量化首先解决的是权重存储和带宽压力，是否真的更快，还取决于硬件是否支持对应低比特算子、运行时是否有合适的 kernel，以及数据是否频繁在 CPU、GPU、NPU 之间搬运。

| 路线 | 典型权重表示 | 目标模型规模 | 设备侧占用特征 | 首 token 延迟特征 |
|---|---:|---:|---|---|
| 原始 32bit | FP32 | 7B | 体积大，通常无法直接端侧部署 | 高，且多数直接超内存 |
| 4bit AWQ + 蒸馏 | INT4/4bit | 1B-7B | 权重显著下降，适合手机或轻薄本 | 中低，适合对话/搜索 |
| 混合 SSM + NPU 特化 | 4bit/2bit + 特化算子 | 1B-7B | 同时压权重和上下文开销 | 更低，长上下文更稳定 |

如果只记一条判断规则，可以记成下面这句：端侧模型追求的不是“理论最强”，而是“预算内最稳”。一个能持续离线运行、首 token 快、十分钟后也不降频卡顿的 3B 方案，通常比一个只能短时间演示跑通的 7B 方案更有工程价值。

---

## 问题定义与边界

端侧模型要解决的问题，不是“模型能不能跑起来”这么简单，而是“能不能在有限内存、有限带宽、有限功耗下，稳定完成具体任务”。这里的“稳定”至少包括四件事：首 token 延迟可接受、连续输出不中断、上下文长度够用、设备不过热。

问题边界可以先用一个总内存预算表示。设设备允许分给模型的预算是 $M_{\text{total}}$，则至少要满足：

$$
M_{\text{total}} = M_{\text{weights}} + M_{\text{KV}} + M_{\text{runtime\ overhead}}
$$

其中权重内存近似为：

$$
M_{\text{weights}} = \text{params} \times \frac{\text{bitwidth}}{8}
$$

运行时真正难估的是 KV Cache。若用更接近工程实现的写法，单条请求的 KV Cache 可粗略写为：

$$
M_{\text{KV}} \approx 2 \times L \times T \times n_{\text{kv}} \times d_{\text{head}} \times b
$$

其中：

- $L$ 是层数
- $T$ 是当前上下文 token 数
- $n_{\text{kv}}$ 是 KV 头数
- $d_{\text{head}}$ 是每个头的维度
- $b$ 是每个 K 或 V 元素的字节数
- 前面的 $2$ 表示同时保存 K 和 V 两份缓存

KV Cache 是推理时保存历史 token 中间状态的缓存。白话说，它相当于模型“记住前文”的临时记忆。上下文越长，这块临时记忆越大；如果用传统密集注意力，它通常会随 token 数近似线性增长。

这也是端侧部署最容易被低估的边界：很多人只看模型参数量，不看上下文长度。实际上，参数决定基础体积，上下文决定运行时膨胀速度。前者像“行李箱本体”，后者像“旅行时不断往里塞的东西”。端侧经常不是被模型本体压死，而是被运行时缓存压死。

一个具体例子：如果在一台 16GB RAM 的 Surface Laptop 7 上跑 3B 模型，系统、浏览器、编辑器和本地索引服务也要占内存，真正可持续留给模型的预算也许只有 2GB 到 4GB。此时如果 4bit 权重已经占了约 1.4GB，而你又把上下文长度开到 64k，KV Cache 可能再吃掉 1GB 到数 GB。问题就不是“慢一点”，而是直接不可持续，系统会开始交换内存、掉吞吐，甚至直接 OOM。

因此，端侧问题的边界通常包括四项：

| 约束项 | 白话解释 | 典型边界 |
|---|---|---|
| 权重内存 | 模型本体大小 | 常见要求控制在 1GB-3GB |
| KV Cache | 长上下文临时记忆 | 32k/64k 时可能接近或超过权重本体 |
| 带宽 | 芯片搬运数据的速度 | 手机/轻薄本常见几十到百余 GB/s |
| 热功耗 | 连续运行会不会发热降频 | 决定能否持续推理 |

还要补一个新手容易混淆的点：同样写着“16GB 内存”，手机、轻薄本、桌面机的可用性并不一样。因为端侧推理不是只比总容量，还要比内存带宽、统一内存架构、NPU/GPU 可见内存、系统保留开销，以及持续负载下的散热能力。

所以，端侧模型并不是“越大越好”，而是“在预算内完成任务最好”。如果任务只是离线摘要、问答、搜索，1B 到 3B 模型经过良好裁剪，往往比一个勉强塞进去的 7B 模型更实用。

---

## 核心机制与推导

端侧模型的核心机制可以拆成四层：量化、蒸馏、KV Cache 压缩、NPU 特化。四者缺一不可，只是不同产品侧重不同。

先看量化。量化是把原来用 16bit 或 32bit 表示的权重，改成 8bit、4bit，甚至 2bit。白话说，就是用更少的位数保存参数，从而直接减小模型体积。它的最直接收益来自公式：

$$
M_{\text{weights}} = \text{params} \times \frac{\text{bitwidth}}{8}
$$

例如一个 3B 模型，若按 FP16 计算，权重大约是：

$$
3\times 10^9 \times 2 \approx 6\text{GB}
$$

如果变成 4bit：

$$
3\times 10^9 \times 0.5 \approx 1.5\text{GB}
$$

这就是端侧部署最关键的一步，因为很多设备的可用预算本来就在 2GB 到 4GB 之间。

但量化不是免费午餐。位宽越低，精度通常越差，尤其是知识问答、复杂推理和少见领域任务更容易退化。所以第二层机制是蒸馏。蒸馏是让小模型学习大模型的输出行为。白话说，就是让一个更小的学生模型模仿老师模型，从而用更少参数保留更多能力。量化压缩的是“表示精度”，蒸馏补偿的是“行为损失”。

可以把两者的分工记成一句话：

- 量化解决“装不下”
- 蒸馏解决“装下以后还好不好用”

| 位宽与方法 | 权重大小变化 | 常见精度影响 | 工程判断 |
|---|---:|---:|---|
| FP16 | 基线 | 基线 | 精度好，但端侧体积偏大 |
| INT8/PTQ | 约降一半 | 轻微退化 | 适合作为保守起点 |
| INT4/AWQ | 约降到四分之一 | 常见掉点可控 | 端侧主流方案 |
| INT2 | 更小 | 退化风险高 | 只适合窄任务或专用硬件 |

这里的 PTQ 是后训练量化，意思是不重新训练，直接把已有模型压缩；AWQ 是一种权重感知量化方法，它会优先保护更重要的权重。对白话读者来说，可以理解成“不是每个参数都一样重要，压缩时要区别对待”。

第三层是 KV Cache 压缩。传统 Transformer 在长上下文下，KV Cache 随 token 数增长。所谓“增长”，不是抽象说法，而是你每多输入一段内容，模型都要多保存一份历史状态。端侧如果直接保留完整密集缓存，很快就会碰到内存上限。

这里常见的工程路线有三种：

| 方法 | 作用点 | 主要收益 | 代价 |
|---|---|---|---|
| GQA/MQA | 减少 KV 头数 | 直接降低 KV Cache 体积 | 结构需在模型设计或训练时考虑 |
| KV 量化 | 降低缓存精度 | 进一步压缩运行时内存 | 过低位宽可能影响长上下文质量 |
| 混合 SSM/窗口化机制 | 改写长历史表示方式 | 长上下文更稳，缓存增长更慢 | 需要模型结构和推理栈一起适配 |

例如，若从多头注意力切到 GQA，本质上是让多个查询头共享更少的 KV 头数。这样做不会改变“模型有多少层”，但会显著减少每层缓存的宽度。对端侧来说，这往往比单纯继续压权重更有效，因为很多设备的真正瓶颈发生在长上下文阶段，而不是加载权重阶段。

示意公式可以写成：

$$
M_{\text{KV}}^{\text{compressed}} = M_{\text{KV}}^{\text{dense}} \times r
$$

其中 $r$ 是压缩比例，可能来自 GQA、KV 量化、窗口化注意力或混合 SSM；它不是一个固定常数，而是方案结果。工程上常见的是把原始 KV Cache 压到原来的几分之一，而不是盲目追求一个统一倍率。

玩具例子如下。假设一个 3B 模型：

- 4bit 量化后权重约 1.4GB 到 1.5GB
- 若使用 GQA，64k 上下文下原始 KV Cache 约 7GB 左右
- 若进一步做 KV 压缩，压缩比例取 $0.125$，则 KV Cache 可降到约 0.9GB

则总占用近似为：

$$
M_{\text{total}} \approx 1.4 + 0.9 + \text{runtime overhead}
$$

如果运行时缓冲区和系统余量再预留 0.4GB 到 0.8GB，总预算就落在约 2.7GB 到 3.1GB 之间，这才进入部分轻薄本或高端移动设备可接受的区间。

第四层是 NPU 特化。NPU 是神经网络处理器，白话说就是专门跑 AI 算子的芯片单元。很多端侧设备并不是算力绝对不够，而是通用算子在 NPU 上跑不顺，数据来回搬运太慢，最后卡死在带宽上。所以要把模型算子改写成目标硬件擅长的形式，例如特定矩阵乘、查表量化核、融合算子路径、静态图导出和内存复用计划。

这里可以把端侧推理分成两层理解：

1. 模型层：你把模型压小、蒸馏好、缓存压下来了。
2. 系统层：你还要让这个模型真的走到目标 NPU/GPU/CPU 的高效执行路径上。

只做第一层，常见结果是“理论可部署”；两层都做好，才会变成“产品可交付”。

真实工程例子可以参考离线文档处理场景：一款文档 App 想在 Surface Laptop 7 或新一代手机上本地运行 7B 以内模型，用于文件问答和摘要。工程上不会直接部署一个通用原版模型，而是先做 4bit AWQ，再针对文档 QA 任务做蒸馏，然后用 GQA、KV 压缩或混合长上下文机制控制 32k 到 64k 上下文缓存，最后在 ExecuTorch 或厂商运行时上导出目标 NPU 可执行图。最终目标不是刷榜，而是让“打开文件后 1 到 2 秒给出首个回答”且离线可用。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不是真实量化器，也不模拟具体芯片，而是把端侧部署里的核心预算逻辑显式写出来，用来判断一个配置是否可能落在设备预算内。核心思想是：先算权重，再算 KV Cache，再把运行时冗余和带宽压力一起纳入估算。

这版代码修正了一个常见错误：KV Cache 不能直接用 `hidden_size` 粗暴代替全部 KV 宽度，否则在使用 GQA/MQA 的模型上会严重高估或低估。更接近实际的写法，是显式写出 `num_kv_heads` 和 `head_dim`。

```python
from dataclasses import dataclass, asdict
from typing import Dict


GiB = 1024 ** 3


@dataclass
class EdgeConfig:
    name: str
    params_billion: float
    weight_bitwidth: int
    num_layers: int
    context_length: int
    head_dim: int
    num_kv_heads: int
    kv_bytes_per_value: float
    kv_compression_ratio: float
    runtime_overhead_gb: float
    memory_budget_gb: float
    bandwidth_budget_gbps: float
    target_toks_per_sec: float


def weight_memory_gb(params_billion: float, bitwidth: int) -> float:
    params = params_billion * 1_000_000_000
    return params * (bitwidth / 8.0) / GiB


def kv_cache_memory_gb(
    context_length: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kv_bytes_per_value: float,
    kv_compression_ratio: float,
) -> float:
    # K 和 V 各一份，所以乘 2
    dense_bytes = (
        2
        * context_length
        * num_layers
        * num_kv_heads
        * head_dim
        * kv_bytes_per_value
    )
    return dense_bytes * kv_compression_ratio / GiB


def total_memory_gb(cfg: EdgeConfig) -> Dict[str, float]:
    weights = weight_memory_gb(cfg.params_billion, cfg.weight_bitwidth)
    kv = kv_cache_memory_gb(
        context_length=cfg.context_length,
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        kv_bytes_per_value=cfg.kv_bytes_per_value,
        kv_compression_ratio=cfg.kv_compression_ratio,
    )
    total = weights + kv + cfg.runtime_overhead_gb
    return {
        "weights_gb": weights,
        "kv_gb": kv,
        "runtime_overhead_gb": cfg.runtime_overhead_gb,
        "total_gb": total,
    }


def bandwidth_pressure(cfg: EdgeConfig, weights_gb: float) -> float:
    # 简化指标：解码阶段越依赖重复搬运权重，压力越高
    # 小于 1 表示通常仍在预算内；大于 1 表示大概率受带宽限制
    required_gbps = weights_gb * cfg.target_toks_per_sec
    return required_gbps / cfg.bandwidth_budget_gbps


def evaluate(cfg: EdgeConfig) -> Dict[str, float]:
    mem = total_memory_gb(cfg)
    pressure = bandwidth_pressure(cfg, mem["weights_gb"])
    feasible = mem["total_gb"] <= cfg.memory_budget_gb and pressure <= 1.0

    result = {
        **asdict(cfg),
        **{k: round(v, 3) for k, v in mem.items()},
        "bandwidth_pressure": round(pressure, 3),
        "feasible": feasible,
    }
    return result


def pretty_print(result: Dict[str, float]) -> None:
    print(f"== {result['name']} ==")
    print(f"weights_gb          : {result['weights_gb']}")
    print(f"kv_gb               : {result['kv_gb']}")
    print(f"runtime_overhead_gb : {result['runtime_overhead_gb']}")
    print(f"total_gb            : {result['total_gb']}")
    print(f"bandwidth_pressure  : {result['bandwidth_pressure']}")
    print(f"feasible            : {result['feasible']}")
    print()


if __name__ == "__main__":
    laptop_cfg = EdgeConfig(
        name="3B-4bit-64k-on-laptop",
        params_billion=3.0,
        weight_bitwidth=4,
        num_layers=28,
        context_length=64_000,
        head_dim=128,
        num_kv_heads=8,          # GQA: KV 头数少于查询头数
        kv_bytes_per_value=2.0,  # 这里先按 FP16 KV 估算
        kv_compression_ratio=0.125,
        runtime_overhead_gb=0.5,
        memory_budget_gb=3.0,
        bandwidth_budget_gbps=80.0,
        target_toks_per_sec=20.0,
    )

    phone_cfg = EdgeConfig(
        name="3B-4bit-32k-on-phone",
        params_billion=3.0,
        weight_bitwidth=4,
        num_layers=28,
        context_length=32_000,
        head_dim=128,
        num_kv_heads=8,
        kv_bytes_per_value=2.0,
        kv_compression_ratio=0.125,
        runtime_overhead_gb=0.4,
        memory_budget_gb=2.4,
        bandwidth_budget_gbps=60.0,
        target_toks_per_sec=12.0,
    )

    laptop_result = evaluate(laptop_cfg)
    phone_result = evaluate(phone_cfg)

    pretty_print(laptop_result)
    pretty_print(phone_result)

    # 基本自检，确保示例与文中结论一致
    assert 1.3 < laptop_result["weights_gb"] < 1.5
    assert 0.8 < laptop_result["kv_gb"] < 1.0
    assert 2.7 < laptop_result["total_gb"] < 3.0
    assert laptop_result["bandwidth_pressure"] < 0.5
```

这段代码的意义不在于精确模拟芯片，而在于把部署脚本里必须暴露的参数固定下来。真正工程里，至少应显式配置以下参数：

| 参数 | 含义 | 建议起点 |
|---|---|---|
| `context_length` | 最大上下文长度 | 8k、32k、64k 按任务选 |
| `weight_bitwidth` | 权重量化位宽 | 4bit 优先，2bit 仅限强约束场景 |
| `num_kv_heads` | KV 头数 | 由模型结构决定，GQA 时要单独写清 |
| `kv_compression_ratio` | KV Cache 压缩比例 | `0.125` 可作示意起点 |
| `runtime_overhead_gb` | 运行时缓冲和系统冗余 | 不要省略，常见 0.3GB-1GB |
| `bandwidth_budget_gbps` | 目标设备带宽预算 | 按手机/PC 实测填写 |
| `target_toks_per_sec` | 目标吞吐 | 交互型任务优先首 token 和稳定输出 |

如果把流程写成伪代码，大致如下：

```python
model = load_teacher_or_base_model()
student = shrink_model(model, target_params="1B-3B")
student = quantize(student, bit=4, method="AWQ")
student = distill(student, teacher=model, task_dataset="QA/search/chat")
student = optimize_attention(student, method="GQA+KV_compression", ratio=0.125)
artifact = export_runtime_graph(
    student,
    target="ArmNPU",
    context_length=64000,
    bandwidth_budget_gbps=80,
)
deploy(artifact, runtime="ExecuTorch")
```

这里每一步都不是装饰：

- `shrink_model` 决定基础规模，不做这一层，后面压缩空间有限。
- `quantize` 控制权重体积，是端侧可行性的第一道门槛。
- `distill` 用来追回量化和缩模带来的能力损失。
- `optimize_attention` 决定长上下文是否可用。
- `export_runtime_graph` 决定模型能否真正在目标 NPU 上高效执行。

对新手来说，最重要的不是背下所有缩写，而是理解一个顺序：先把模型变成“装得下”，再把它变成“跑得稳”，最后把它变成“对任务真的有用”。

---

## 工程权衡与常见坑

端侧模型最大的误区，是把它当成“一个更小的云模型”。实际上它是一个强约束系统，任何单项优化都可能被其他瓶颈抵消。

第一类坑是只做 PTQ，不做蒸馏和任务校验。这样做的结果通常是演示能跑，真实业务不稳。因为 PTQ 更像“压缩文件”，它不会自动补偿模型能力损失。很多团队在通用基准上看起来只掉了几个点，但在自己的知识问答、检索增强、长文摘要任务上会出现更明显的错误累积。

第二类坑是只盯权重，不盯 KV Cache。尤其是长上下文产品，很多人算出了“模型只有 1.4GB”，就以为手机可以跑，结果一到 32k 或 64k 输入直接崩掉。原因不是权重，而是上下文缓存吃掉了剩余预算。

第三类坑是忽略带宽和热设计。端侧推理的上限常常不是理论 FLOPS，而是内存搬运和持续发热。一个示意性的带宽约束可以写成：

$$
C_{\text{bandwidth}} \propto \frac{\text{bytes moved per token}}{\text{available bandwidth}}
$$

这个式子不是严格物理模型，但含义很重要：如果每生成一个 token 都要大量搬运权重和中间张量，那么即使芯片有足够峰值算力，也会被带宽卡住。对新手来说，可以把它理解成“算得快不够，还要喂得上”。

第四类坑是忽略算子兼容性。很多宣传资料会直接写“设备有 N TOPS”，但这不等于你的模型就一定能高效跑起来。因为模型是否真能用上 NPU，取决于：

- 量化格式是否被目标后端支持
- 算子是否能下沉到 NPU，而不是回退到 CPU
- 图是否能做融合，减少跨设备搬运
- 动态 shape、采样器、tokenizer 等外围环节是否也被优化

一个真实坑例子：某团队把一个 3B 模型直接做 PTQ 后部署到手机，实验室里可以跑，但一到真实用户场景，32k 上下文就开始不稳定，且连续问答 3 分钟后因发热降频，输出速度明显下降。后续修复不是继续压 bit，而是补做面向目标任务的蒸馏，降低默认上下文长度，引入 GQA/KV 压缩，并把带宽、功耗、温度和 tok/s 一起纳入预算表，才把产品稳定下来。

| 问题 | 原因 | 解决方式 |
|---|---|---|
| PTQ 后精度掉点明显 | 只压缩，没有行为补偿 | 增加蒸馏和任务集 QA 回路 |
| 32k/64k 不稳定 | 低估 KV Cache 占用 | 做 KV 压缩、GQA 或分级上下文 |
| NPU 跑不满 | 算子不匹配，数据搬运过多 | 做目标硬件算子特化和图融合 |
| 跑一会变慢 | 发热降频 | 控制 tok/s、批量、功耗模式 |
| 榜单分数高但产品体验差 | 模型过于通用，未围绕任务裁剪 | 按任务压缩和定制蒸馏 |

工程上最重要的权衡只有一句话：端侧模型不是追求“最强能力”，而是追求“预算内最稳能力”。如果一个 7B 模型只能短时间跑通，而一个 1.5B 或 3B 模型能离线稳定做搜索、摘要和表单抽取，后者往往才是可交付方案。

---

## 替代方案与适用边界

端侧模型不是云模型的替代品，而是一类有明确适用边界的部署模式。只要任务超出边界，就应该果断引入云端，而不是继续硬压本地模型。

最常见的边界有三类：

1. 需要超过 7B 级别的知识与推理能力。
2. 需要很长上下文并长期稳定运行，例如 128k 以上且不能明显掉速。
3. 需要频繁访问最新外部知识、复杂工具链或多步工作流。

这三类任务更适合云端，因为云侧更容易提供大显存、大带宽和弹性扩容。代价则是隐私、网络依赖和交互延迟。

对多数产品来说，更现实的是端云协同。所谓“端云协同”，就是轻任务本地完成，重任务上云。白话说，能在设备上几百毫秒到 1 秒内解决的事不要上云，只有本地模型明显超界时才切换到远端模型。

一个简单策略是：

- 端侧负责常见 1B 到 3B 任务：意图分类、短问答、本地搜索、文档摘要、离线助手。
- 云端负责重任务：跨文档复杂推理、长链工具调用、超长上下文综合分析、需要最新知识的问答。

新手可以把它理解为“先让本地解决高频、低复杂度任务，再把低频、高复杂度任务交给云”。这比一开始就试图把所有功能都塞进端侧要稳得多。

| 方案 | 延迟 | 隐私 | 长上下文能力 | 成本结构 | 适用场景 |
|---|---|---|---|---|---|
| 纯端侧模型 | 低，本地首 token 快 | 高，数据不出设备 | 弱到中，受内存限制 | 前期优化重，边际成本低 | 离线助手、本地搜索、轻摘要 |
| 端侧+云协同 | 中 | 中高，可按任务分流 | 中到强 | 架构复杂，但体验均衡 | 消费级 AI App、企业终端助手 |
| 纯云模型 | 中到高，受网络影响 | 低到中 | 强 | 按调用付费，扩展性高 | 复杂推理、超长上下文、最新知识问答 |

什么时候应该从端侧切到云端，可以用一条简单规则判断：如果任务要求的模型规模、上下文长度或工具复杂度，已经让本地方案必须明显牺牲准确率、时延或稳定性，那就切云。反过来，如果任务可以被压缩为分类、抽取、摘要、模板化问答、局部检索，那么 1B 到 3B 级端侧模型通常更合适。

因此，端侧模型的发展方向并不是和云“二选一”，而是把本地可完成的任务尽量本地化，把高复杂度任务保留给云。这种边界清晰的系统，比一味追求“全端侧”更符合工程现实。

---

## 参考资料

本文优先引用论文和官方文档，而不是难以复核的行业综述。原因很简单：端侧模型的工程路径变化很快，但“量化为什么有效”“GQA 为什么能省 KV Cache”“运行时如何落到设备后端”这些关键结论，最好还是用论文和官方部署文档来支撑。

| 标题 | 发布日期 | 覆盖主题 |
|---|---|---|
| [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) | 2023-06-02 | 4bit 权重量化、端侧推理、低比特压缩 |
| [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) | 2023-07-18 | 长上下文、GQA、KV Cache 工程权衡 |
| [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | 2023-12-02 | SSM、长序列建模、线性扩展路径 |
| [OpenELM: An Efficient Language Model Family with Open Training and Inference Framework](https://arxiv.org/abs/2404.14619) | 2024-04-23 | 小模型架构优化、GQA、端侧友好设计 |
| [ExecuTorch 文档：Deploying LLMs to ExecuTorch](https://docs.pytorch.org/executorch/stable/llm/getting-started.html) | 2026-03-12 访问 | 端侧 LLM 导出、运行时、硬件后端 |
| [ExecuTorch 文档：Run Llama 3 3B Instruct on Android with Qualcomm Backend](https://docs.pytorch.org/executorch/stable/llm/build-run-llama3-qualcomm-ai-engine-direct-backend.html) | 2026-03-12 访问 | Android 端侧部署、Qualcomm 后端、内存约束 |
| [Apple Developer: Apple Intelligence Resources / Foundation Models framework](https://developer.apple.com/apple-intelligence/resources/) | 2026-03-12 访问 | Apple 端侧基础模型、官方设备侧能力入口 |

以上结论可作为 2026 年初端侧模型路线的阶段性总结：小模型、低比特量化、蒸馏、KV Cache 压缩和 NPU 特化仍然是主轴；真正落地时，应先按任务预算做裁剪，再谈模型规模和基准分数。若只能记住一句话，那就是：端侧模型的本质不是把模型“做小”，而是把任务“做稳”。
