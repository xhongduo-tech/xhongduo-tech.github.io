## 核心结论

MoE，Mixture of Experts，意思是“一个 MoE 层里有很多个前馈子网络，但每个 token 只走其中少数几个”。对这类模型做推理优化，核心目标通常不是继续压 FLOPs，而是降低**专家权重常驻 GPU 显存**带来的内存压力，并把专家搬运造成的停顿尽量藏到计算重叠里。

以 Mixtral 8x7B 为代表的稀疏模型为例，论文给出的关键信息是：模型总参数约 47B，但每个 token 推理时只激活约 13B 参数，因为每层 8 个专家里只选 top-2。这个结构意味着“单次真正参与计算的参数量”远小于“模型总参数量”，因此理论上可以把大量不活跃专家留在 CPU 内存甚至 SSD，而不是全部常驻 GPU。

可行路径通常是三件事同时做：

1. 把非活跃专家卸载到 CPU 内存或 SSD，只在需要时回载到 GPU。
2. 用 LRU 缓存保留“刚刚被用过、短时间内大概率还会再用”的热专家。
3. 用路由预测提前预取“下一批 token 可能会用到”的专家，把 I/O 延迟尽量和当前计算重叠。

玩具例子：把 256 个专家看成 256 本教材，GPU 只能放 16 本。每节课只会抽 8 本真正用到。LRU 是“最近刚用过的教材先别收走”，预取是“根据下一节课的课表，提前把可能要用的几本先摆到桌上”。如果课程切换有规律，搬书次数会明显下降；如果课程切换完全随机，缓存和预取收益会快速衰减。

| 方案 | 显存占用 | 单次 token 的 I/O 压力 | 平均延迟特征 | 适用场景 |
| --- | --- | --- | --- | --- |
| 全部专家常驻 GPU | 最高 | 最低 | 最稳定，但硬件门槛最高 | 多卡或高显存服务器 |
| 只做专家卸载，不做缓存 | 低 | 最高 | 长尾明显，容易被 PCIe/SSD 拖慢 | 仅验证“能跑” |
| 卸载 + LRU 缓存 | 中低 | 中等 | 对局部重复路由有效 | 单卡、低批量 |
| 卸载 + LRU + 预取 | 中低 | 更可控 | 命中率高时延迟最好 | 在线解码、路由有规律 |

---

## 问题定义与边界

问题先定义清楚：MoE 推理里的真实瓶颈，经常不是算力，而是**权重搬运**。原因很直接。路由器只有在运行时才知道当前 token 该去哪个专家；如果目标专家不在 GPU，就必须把这一整块专家权重搬进来。专家通常是大张量，搬运链路往往是：

`SSD -> CPU 内存 -> PCIe/NVLink -> GPU 显存`

这里每一跳都比 GPU 直接读本地显存慢得多。

对新手最容易混淆的一点是：**“只激活少数专家”不等于“只占少数显存”**。  
如果你把所有专家长期放在 GPU，那么即使每次只算 top-2，显存占用依然接近“把整套模型都摊开部署”的量级。稀疏路由省掉的是“本次不算哪些专家的乘加”，不是“这些专家从物理上不存在”。

Mixtral 8x7B 给这个问题提供了一个很典型的样本：

| 指标 | 含义 | 对部署的启示 |
| --- | --- | --- |
| 总参数约 47B | 模型全部参数的总和 | 不能按“47B 全常驻 GPU”思路部署到消费卡 |
| 每 token 激活约 13B | 当前 token 实际参与计算的参数量 | 存在专家卸载和按需加载空间 |
| 每层 8 个专家、top-2 路由 | 一个 token 每层只走 2 个专家 | 路由结果稀疏，适合做缓存 |
| 不同 token 可走不同专家 | 路由在时间上变化 | 需要缓存和预取来对抗抖动 |

边界也必须说明，否则讨论会跑偏：

- 这里讨论的是**单机、显存紧张**场景，不是多卡 expert parallel。
- 重点是**推理**，不是训练。训练还要承担梯度、优化器状态、激活保存，问题更重。
- 重点是**解码阶段**和**小批量在线服务**。大 batch prefill 的路由分布更密，调度目标不同。
- 如果专家激活比例很高，或者路由近似随机，缓存和预取都会明显失效。
- 如果你的 GPU 足够大，能把所有专家稳定常驻，那 offloading 本身就不是优先项。

真实工程例子是 `dvmazur/mixtral-offloading`。它的基本路线不是“整层整层搬”，而是把专家拆开存储，运行时只把当前需要的专家搬到 GPU，其余放在 CPU，并配合混合量化降低尺寸。这样能把“根本放不下”变成“可以运行”，但平均延迟会高度依赖三件事：缓存命中率、CPU 到 GPU 带宽、以及是否能把搬运和计算重叠起来。

把资源约束写成表格更直观：

| 资源项 | MoE 推理需求 | 常见消费级设备约束 | 风险 |
| --- | --- | --- | --- |
| GPU 显存 | 容纳基础层 + 活跃专家 + 传输缓冲区 + KV cache | 16GB / 24GB 常见 | 专家无法全部常驻 |
| CPU 内存 | 存放卸载专家、页锁定缓冲区 | 32GB / 64GB 常见 | 容量常够，但延迟高于显存 |
| SSD 带宽 | 冷启动或极端 miss 时回源 | NVMe 与 SATA 差异大 | 长尾延迟差异极大 |
| PCIe 带宽 | CPU 到 GPU 的关键搬运通道 | 单卡桌面带宽有限 | 容易成为主瓶颈 |
| 路由稳定度 | 决定缓存和预测上限 | 由模型、任务、阶段决定 | 局部性差时收益下降 |

还有一个常用的显存估算式，方便判断缓存到底“放得下多少”：

$$
M_{gpu} \approx M_{base} + C_{hot}\cdot S_{expert} + B_{buf}\cdot S_{expert} + M_{kv}
$$

其中：

- $M_{base}$：非专家基础层常驻显存
- $C_{hot}$：GPU 热缓存可容纳的专家个数
- $S_{expert}$：单个专家的权重大小
- $B_{buf}$：传输缓冲区可同时容纳的专家个数
- $M_{kv}$：KV cache 占用

这个式子不求特别精确，但足够说明一个工程事实：**缓存容量不是白来的，它会直接挤占 KV cache 和其他常驻模块的空间。**

---

## 核心机制与推导

把系统抽象成三层存储最容易理解：

```text
[ GPU 热缓存 ]
      ^
      |  PCIe / NVLink
      v
[ CPU 冷缓存 ]
      ^
      |  SSD / 文件系统
      v
[ 持久化权重 ]
```

这三层各自负责不同的事：

| 层级 | 典型介质 | 访问速度 | 适合放什么 |
| --- | --- | --- | --- |
| 第 1 层 | GPU 显存 | 最快 | 当前步即将执行或刚刚执行过的热专家 |
| 第 2 层 | CPU 内存（最好是 pinned memory） | 中等 | 最近可能再次被访问的冷专家 |
| 第 3 层 | SSD / 文件系统 | 最慢 | 长时间不用的持久化权重 |

### 1. LRU 缓存为什么有效

LRU，Least Recently Used，意思是“最久没用过的先淘汰”。它依赖的不是数学魔法，而是一个工程假设：**专家路由存在时间局部性**。

时间局部性可以用一句话概括：如果专家 `E17` 刚刚在当前几个 token 被频繁调用，那么接下来短时间内它再次被调用的概率，通常高于一个完全随机的专家。对自然语言生成，这个假设往往成立，因为相邻 token 的句法结构、主题语义、局部上下文是连续的，路由分布不会每一步都完全洗牌。

一个初学者更容易理解的例子：

- 假设每一步需要 8 个专家
- GPU 最多缓存 16 个专家
- 最近 5 步里，平均有 5 到 6 个专家与上一步重叠

这意味着当前步真正需要新搬运的专家，平均不是 8 个，而可能只有 2 到 3 个。哪怕单次搬运很贵，只要 miss 数降低，平均延迟就会明显下降。

可以把命中和 miss 看成下面这个关系：

$$
\text{misses per step} = K - \left|E_t \cap \mathcal{C}_t\right|
$$

其中：

- $K$：当前步需要的专家数量，例如 top-2 路由在多层上汇总后的唯一专家数
- $E_t$：当前步实际需要的专家集合
- $\mathcal{C}_t$：当前步开始时 GPU 缓存中的专家集合

当交集越大，miss 越少，等待时间越低。

### 2. 预取为什么比“被动 miss 再加载”更重要

缓存解决的是“之前已经用过的专家还在不在”。  
预取解决的是“下一步可能要用的专家能不能提前到位”。

两者的差别很关键：

- 只有缓存，没有预取：一旦 miss，用户路径上仍然要同步等待加载。
- 有缓存也有预取：可以把“未来很可能发生的 miss”提前转成后台异步加载。

这就是为什么很多系统里，预取的价值不只是“少搬一点”，而是“**把等待从前台挪到后台**”。

对 decode 阶段，这一点尤其重要。因为 decode 是逐 token 串行的，单步等待会直接体现成用户看到的 token 间隔。如果当前 token 的计算时间能覆盖下一个 token 的部分专家加载时间，那么用户感知到的停顿会显著减小。

常见预测信号有三类：

- 当前 token 的路由结果  
  含义：如果当前 token 走了专家 `E5`、`E9`，下一步仍可能继续走它们或相邻模式的专家。
- 前几个 token 的专家轨迹  
  含义：观察最近几步的专家切换是否稳定，再决定预取窗口。
- 额外训练的轻量预测器  
  含义：用运行时特征、层索引、历史路由统计预测下几步最可能用到的专家。

对新手最实用的判断标准不是“预取算法是否高级”，而是下面这句：

> 预取只有在“预测够准”且“异步搬运真的和计算重叠”时才有价值。

如果预测不准，预取就会变成提前搬错货；如果虽然预测对了，但加载仍然挡在前台同步等待，那它也没有真正改善用户路径延迟。

### 3. 延迟公式如何理解

先给一个足够实用的平均延迟模型。设：

- $h$：缓存命中率
- $L_{cache}$：命中时直接在 GPU 读取专家的延迟
- $L_{prefetch}$：预取带来的调度和管理成本
- $L_{load}$：未命中时从 CPU/SSD 加载专家的成本

则平均专家访问延迟可写成：

$$
E[L] = h \cdot L_{cache} + (1-h)\cdot(L_{prefetch} + L_{load})
$$

这个式子本身不复杂，但足够解释几个常见现象：

- 当 $h$ 提升时，平均延迟通常会近似线性下降。
- 当预取足够及时，$L_{load}$ 的一部分可以与其他计算重叠，等效成本下降。
- 当预取不准时，$L_{prefetch}$ 会变大，因为你搬来了不会立刻使用的冷专家。

如果进一步考虑“计算与加载重叠”，更贴近工程实际的写法是：

$$
L_{miss,visible} \approx L_{sched} + \max(0,\; L_{io} - L_{overlap})
$$

其中：

- $L_{sched}$：调度开销
- $L_{io}$：实际搬运所需时间
- $L_{overlap}$：被当前计算遮住的那部分时间

这说明真正决定用户感知延迟的，不是总共搬了多少字节，而是**还有多少 I/O 没被遮住，最后不得不同步等待**。

下面这个表格能把三种路径区分清楚：

| 路径 | 是否命中 | 延迟组成 | 说明 |
| --- | --- | --- | --- |
| GPU 直接执行 | 是 | $L_{cache}$ | 理想路径 |
| 预取完成后执行 | 否，但提前搬到位 | $L_{sched}$ 或很小的等待 | 大部分 I/O 被隐藏 |
| miss 后同步加载 | 否 | $L_{sched} + L_{io}$ | 用户直接感知卡顿 |

如果从吞吐角度看，设每步需同步加载的专家数为 $m$，单专家平均搬运时间为 $t_{io}$，那么这一步的额外等待近似受下面约束：

$$
T_{stall} \propto m \cdot t_{io}
$$

因此缓存和预取的真正目标，不是把“总搬运量”做到绝对最小，而是让**同步等待的专家数**尽量接近 0。

---

## 代码实现

工程上通常会把专家按层、按编号拆成独立权重块，例如：

- `layer_12_expert_3.safetensors`
- `layer_12_expert_4.safetensors`

这样调度器拿到路由结果后，就能只搬当前需要的块，而不是整层一起搬。真实系统还会额外维护：

- GPU 常驻区
- CPU pinned memory 区
- 传输缓冲区
- 命中统计和淘汰顺序

下面给一个**可直接运行**的 Python 玩具实现。它不依赖深度学习框架，但会完整演示：

- LRU 热缓存
- 当前步专家加载
- 下一步小窗口预取
- prefill 与 decode 的不同预算
- 命中率、预取命中率、无效预取率统计

```python
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple


@dataclass
class CacheStats:
    demand_hits: int = 0
    demand_misses: int = 0
    prefetch_requests: int = 0
    useful_prefetches: int = 0
    wasted_prefetches: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.demand_hits + self.demand_misses
        return self.demand_hits / total if total else 0.0

    @property
    def prefetch_use_rate(self) -> float:
        return self.useful_prefetches / self.prefetch_requests if self.prefetch_requests else 0.0

    @property
    def prefetch_waste_rate(self) -> float:
        return self.wasted_prefetches / self.prefetch_requests if self.prefetch_requests else 0.0


@dataclass
class ExpertSlot:
    resident: str  # "gpu-demand" or "gpu-prefetch"


@dataclass
class ExpertCache:
    capacity: int
    slots: "OrderedDict[str, ExpertSlot]" = field(default_factory=OrderedDict)
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")

    def has(self, expert_id: str) -> bool:
        return expert_id in self.slots

    def _evict_if_needed(self) -> None:
        while len(self.slots) > self.capacity:
            evicted_id, evicted_slot = self.slots.popitem(last=False)
            if evicted_slot.resident == "gpu-prefetch":
                self.stats.wasted_prefetches += 1
            print(f"  evict   -> {evicted_id:>3} ({evicted_slot.resident})")

    def _insert(self, expert_id: str, resident: str) -> None:
        if expert_id in self.slots:
            self.slots.move_to_end(expert_id)
            current = self.slots[expert_id]
            # 如果该专家原本是预取进来的，而现在被真正使用，则升级为 demand
            if current.resident == "gpu-prefetch" and resident == "gpu-demand":
                current.resident = "gpu-demand"
                self.stats.useful_prefetches += 1
            return

        self.slots[expert_id] = ExpertSlot(resident=resident)
        self._evict_if_needed()

    def demand_load(self, expert_id: str) -> None:
        if expert_id in self.slots:
            self.stats.demand_hits += 1
            self.slots.move_to_end(expert_id)
            slot = self.slots[expert_id]
            if slot.resident == "gpu-prefetch":
                slot.resident = "gpu-demand"
                self.stats.useful_prefetches += 1
            print(f"  hit     -> {expert_id:>3}")
        else:
            self.stats.demand_misses += 1
            self.slots[expert_id] = ExpertSlot(resident="gpu-demand")
            print(f"  miss    -> {expert_id:>3} (load to GPU)")
            self._evict_if_needed()

    def prefetch(self, expert_id: str) -> bool:
        if expert_id in self.slots:
            self.slots.move_to_end(expert_id)
            return False
        self.stats.prefetch_requests += 1
        self.slots[expert_id] = ExpertSlot(resident="gpu-prefetch")
        print(f"  prefetch-> {expert_id:>3}")
        self._evict_if_needed()
        return True

    def snapshot(self) -> List[str]:
        return list(self.slots.keys())


def choose_prefetch_budget(phase: str, free_slots: int) -> int:
    if phase == "prefill":
        # prefill 路由更密，给主计算保留更多缓存空间
        return max(1, free_slots // 4)
    if phase == "decode":
        # decode 更稀疏，允许更积极一些的小窗口预取
        return max(1, free_slots // 2)
    raise ValueError(f"unknown phase: {phase}")


def schedule_step(
    cache: ExpertCache,
    current_experts: Sequence[str],
    predicted_next_experts: Sequence[str],
    phase: str,
) -> Set[str]:
    print(f"\nphase={phase}, current={list(current_experts)}, predicted_next={list(predicted_next_experts)}")

    # 1. 保证当前步需要的专家可用
    for expert_id in current_experts:
        cache.demand_load(expert_id)

    # 2. 为下一步做小窗口预取
    used_now = set(current_experts)
    candidates = [eid for eid in predicted_next_experts if eid not in used_now and not cache.has(eid)]
    free_slots = max(1, cache.capacity - len(cache.snapshot()))
    budget = choose_prefetch_budget(phase, free_slots)

    prefetched: Set[str] = set()
    for expert_id in candidates[:budget]:
        if cache.prefetch(expert_id):
            prefetched.add(expert_id)

    print(f"  cache   -> {cache.snapshot()}")
    return prefetched


def run_demo() -> None:
    # 假设相邻 token 的路由有局部重复：这是 LRU 能工作的前提
    trace: List[Tuple[str, List[str], List[str]]] = [
        ("prefill", ["e1", "e2", "e3"], ["e2", "e3", "e4"]),
        ("prefill", ["e2", "e3", "e4"], ["e3", "e4", "e5"]),
        ("decode",  ["e3", "e4"],       ["e4", "e5"]),
        ("decode",  ["e4", "e5"],       ["e5", "e6"]),
        ("decode",  ["e5", "e6"],       ["e6", "e7"]),
        ("decode",  ["e6", "e7"],       ["e7", "e8"]),
    ]

    cache = ExpertCache(capacity=5)

    for phase, current_experts, predicted_next in trace:
        schedule_step(cache, current_experts, predicted_next, phase)

    print("\nsummary")
    print(f"  demand hit rate      : {cache.stats.hit_rate:.2%}")
    print(f"  useful prefetch rate : {cache.stats.prefetch_use_rate:.2%}")
    print(f"  wasted prefetch rate : {cache.stats.prefetch_waste_rate:.2%}")


if __name__ == "__main__":
    run_demo()
```

这个脚本可以直接运行。你会看到三类输出：

- `hit`：当前步需要的专家已经在 GPU
- `miss`：当前步需要的专家不在 GPU，必须现加载
- `prefetch`：调度器提前为下一步搬入的专家

这个玩具实现故意省略了真实系统中的 CUDA stream、非阻塞 copy、页锁定内存和多层分组缓存，但调度逻辑是一致的。

把它映射到真实推理系统，通常对应四步：

1. token 进入路由器，得到每层的候选专家。
2. 查询 GPU 专家缓存，命中则直接执行。
3. 未命中则从 CPU pinned memory 或 SSD 发起异步加载。
4. 根据下一步预测结果执行小窗口预取，并更新 LRU 状态。

流程图可以写成：

```text
token
  -> router
  -> current experts
  -> cache lookup
      -> hit: execute on GPU
      -> miss: async load from CPU/SSD
  -> update LRU
  -> predict next experts
  -> prefetch within budget
```

如果再往真实工程靠一步，通常还会补两点：

| 工程细节 | 为什么需要 |
| --- | --- |
| 把单个专家的所有参数整理成连续存储 | 减少碎片化搬运，降低多次 DMA 调用开销 |
| 维护独立传输缓冲区 | 让“加载新专家”和“回收旧专家”尽量非阻塞重叠 |

`mixtral-offloading` 的实现思路就体现了这一点：不是简单地“需要时才从磁盘读文件”，而是为专家维护 GPU 区、CPU 侧存储和传输缓冲区，并尽量用非阻塞交换把复制与计算重叠。否则 offloading 只能证明“能跑起来”，不能证明“延迟能接受”。

---

## 工程权衡与常见坑

最常见的错误不是“不会写缓存”，而是**把专家缓存看成纯算法题**。实际结果同时受下面几件事影响：

- 路由是否有局部性
- GPU 剩余显存是否够放热专家
- CPU 是否有足够 pinned memory
- PCIe/NVLink 带宽是否足够
- SSD 是否只在冷启动时参与，而不是频繁回源
- 预取是否真的和计算重叠，而不是只在逻辑上“提前了一点点”

### 1. 预取窗口过大

预取窗口就是“提前看多远、提前搬多少”。  
窗口过大通常会带来两个直接损失：

- 冷专家占掉本该给热专家的 GPU 空间
- 搬运了很多最后并不会立刻用到的专家，导致无效 I/O

初学者常见误区是：既然预取可以隐藏延迟，那就多预取一些。  
这在工程上通常是错的。因为缓存是稀缺资源，预取每多搬一个不必要的专家，都可能把真正热的专家挤出去。

### 2. 不区分 prefill 和 decode

prefill 和 decode 的负载模式差异很大：

| 阶段 | 运行方式 | 路由特征 | 调度目标 |
| --- | --- | --- | --- |
| prefill | 一次处理整段输入 | 激活更密，层间并行更多 | 防止主计算被缓存挤压 |
| decode | 逐 token 生成 | 激活更稀疏，局部性更明显 | 降低单 token 长尾延迟 |

因此，统一用一套预取窗口和同一套预算，常见结果是：

- prefill 预取不够保守，导致显存被挤爆
- decode 预取不够精准，导致单 token 抖动加大

很多双阶段调度论文的核心贡献，本质上就是把这两个阶段拆开处理，而不是用同一套启发式。

### 3. 缓存容量和带宽不配平

缓存不是越大越好。  
缓存变大，会带来两个反作用：

- 显存占用增加，可能挤压 KV cache 和基础层
- 单次换入换出对象变多，传输缓冲区压力上升

反过来看，如果 CPU 到 GPU 带宽本来就很低，那么即使缓存设计合理，也只能减少损失，不能完全消除瓶颈。  
也就是说，缓存策略决定“你浪费多少”，链路带宽决定“你最多能做到多快”。

下面这个表格是最常见的故障排查入口：

| 常见坑 | 直接后果 | 典型症状 | 修正方向 |
| --- | --- | --- | --- |
| 预取窗口过大 | 冷专家占满缓存 | 命中率不升反降 | 缩小窗口，按阶段设预算 |
| 忽略 decode 稀疏性 | 无效预取过多 | 单 token 延迟抖动 | decode 用更小、更近的预测窗口 |
| 只看命中率，不看重叠 | 指标误判 | hit 很高但延迟仍差 | 增加同步等待时间统计 |
| SSD 回源过多 | 长尾延迟极差 | 偶发尖峰卡顿 | 让 CPU 内存做稳定二级缓存 |
| 按全局统一淘汰 | 层间互相污染 | 某些层反复 miss | 改成分层或分组 LRU |
| 忽略 buffer 开销 | 显存预算失真 | 纸面能放下，实际 OOM | 把传输缓冲区计入预算 |

下面给一个区分 prefill 和 decode 的可运行函数，用来说明“预算应随阶段变化”：

```python
def choose_prefetch_budget(phase: str, free_vram_mb: int) -> int:
    if phase == "prefill":
        # prefill 激活更密，预取要保守，优先给主计算和 KV cache 留空间
        return max(1, free_vram_mb // 4096)
    if phase == "decode":
        # decode 更稀疏，可以在更小窗口内做相对积极的预取
        return max(1, free_vram_mb // 2048)
    raise ValueError("unknown phase")


assert choose_prefetch_budget("prefill", 8192) <= choose_prefetch_budget("decode", 8192)
```

真正落地时，还需要补一层观测指标。至少要记录：

- 每层专家命中率
- 每步同步等待的搬运时间
- 预取成功率和浪费率
- 刚被淘汰又立刻回载的频率
- SSD 回源次数与占比
- 每层热专家集合的稳定度

这些指标比单一的 `token/s` 更有调参价值。  
`token/s` 只能告诉你“整体快不快”，但不能告诉你“到底是命中差、预取差、还是搬运没重叠起来”。

---

## 替代方案与适用边界

LRU + 预取不是通用最优解。它只是在“激活稀疏、局部性存在、I/O 还算可控”的区间内特别有效。超出这个区间，就要换思路。

### 1. 纯缓存，不做预测

优点是实现简单、调试成本低、行为更稳定。  
缺点也很明确：它只能利用“已经发生过的局部性”，不能主动隐藏未来 miss。

适合场景：

- 个人机器
- 实验验证
- 代码复杂度敏感
- 路由规律还没摸清楚

如果你是第一次做 MoE offloading，通常应先把“量化 + 简单 LRU”跑稳定，再考虑预取。因为没有稳定基线时，预取带来的收益和噪声很难分清。

### 2. 混合量化

量化的作用不是“减少专家切换次数”，而是直接**缩小每个专家本身的体积**。这会同时缓解两个问题：

- 降低显存压力
- 降低带宽压力

因此量化和缓存不是替代关系，而是常常叠加使用。  
`mixtral-offloading` 的经验也说明，基础层和专家层往往适合不同量化策略，而不是一刀切。因为两类模块对数值误差的敏感度不同。

可以把它理解成：

| 优化手段 | 解决什么 |
| --- | --- |
| 缓存 | 少搬几次 |
| 预取 | 把等待提前藏起来 |
| 量化 | 每次搬得更小 |

三者叠加时，效果通常最好。

### 3. 双阶段或自适应预测

这类方法回答的是更细的问题：

- prefill 和 decode 应该用同样的窗口吗
- 不同层的专家稳定度是否相同
- 当链路带宽波动时，预取步长是否该动态调整
- 当预测误差变大时，是否该自动退回保守策略

像双阶段调度、自适应 horizon、cache-aware routing 一类工作，本质上都在做这件事。它们的上限更高，但工程复杂度也明显更高，因为你要持续收集运行时统计，并把这些统计反馈给调度器。

### 4. 多卡 expert parallel

这条路线不是 offloading，而是把专家静态分布到多张 GPU 上。  
它的优点是减少主机到 GPU 的慢搬运；缺点是硬件成本高，并且会引入卡间通信与部署复杂度。

因此它适合：

- 服务器场景
- 有多卡资源
- 更关注稳定吞吐而非单机极限压缩

不适合：

- 单卡桌面
- 低成本实验环境
- 只想先验证可运行性的场景

把几种路线并列看会更清楚：

| 方案 | 优点 | 缺点 | 适用边界 |
| --- | --- | --- | --- |
| 纯 LRU 缓存 | 简单、稳、好调试 | 不能提前隐藏 miss | 单卡实验、低并发 |
| LRU + 预取 | 延迟更低 | 依赖预测质量 | 在线解码、路由有规律 |
| 混合量化 | 同时减内存和带宽 | 可能掉精度，需校准 | 资源受限部署 |
| 双阶段自适应预测 | 上限高 | 实现复杂、观测要求高 | 服务化部署 |
| 多卡 expert parallel | 减少主机搬运 | 硬件贵、通信复杂 | 服务器集群 |

如果从新手视角给一个简化决策路径，可以直接这样判断：

- 只有 16GB 显存、32GB 到 64GB 内存：先做“量化 + 简单缓存”。
- 已经能跑，但单 token 延迟高且抖动大：再加“小窗口 decode 预取”。
- 要做正式服务，且请求模式稳定：再考虑双阶段或自适应策略。
- 如果 PCIe 已经长期打满：继续堆预测的收益通常有限，这时更该考虑更强量化、更大的 CPU 二级缓存，或者直接换成多卡方案。

最后再强调一次适用边界：  
如果路由几乎随机，或者 batch 很大导致每步活跃专家覆盖面过宽，那么 LRU 和预取的收益都会下降。这个时候问题不再是“缓存策略不够聪明”，而是“工作负载本身不适合靠热缓存解决”。

---

## 参考资料

| 资料 | 链接 | 重点内容 |
| --- | --- | --- |
| Mixtral of Experts | https://arxiv.org/abs/2401.04088 | Mixtral 8x7B 架构、每层 8 个专家、top-2 路由、约 47B 总参数与约 13B 活跃参数 |
| dvmazur/mixtral-offloading | https://github.com/dvmazur/mixtral-offloading | 真实工程实现：专家拆分、混合量化、LRU 热缓存、消费级硬件运行实践 |
| mixtral-offloading Expert Management System 解读 | https://deepwiki.com/dvmazur/mixtral-offloading/3.3-expert-management-system | GPU/CPU 双驻留、分组 LRU、统一存储与非阻塞专家交换 |
| NVIDIA: Achieving High Mixtral 8x7B Performance with NVIDIA H100 Tensor Core GPUs and TensorRT-LLM | https://developer.nvidia.com/blog/achieving-high-mixtral-8x7b-performance-with-nvidia-h100-tensor-core-gpus-and-tensorrt-llm/ | 从部署角度解释 Mixtral 稀疏路由、吞吐与系统优化背景 |
| ExpertFlow | https://arxiv.org/abs/2510.26730 | 自适应预取窗口、多级内存协调、基于运行时统计调整专家调度 |
| MoE-Infinity | https://github.com/EfficientMoE/MoE-Infinity | 更完整的多级内存 offloading / prefetch 系统参考，适合对运行时设计进一步展开 |
