## 核心结论

MoE，Mixture of Experts，直译是“专家混合”，可以理解为“模型里有很多套前馈网络，但每个 token 只用其中少数几套”。这意味着它的总参数量可以非常大，但单次计算真正参与的参数远小于总量。

对推理系统来说，真正的问题不是“模型总共有多少参数”，而是“当前层和下一层马上要用到哪些专家”。因此，MoE 推理加速的核心不是把所有专家都塞进 GPU，而是让 GPU 只保留活跃专家，把其余专家卸载到 CPU 或 NVMe，并在当前层计算时预取下一层大概率会用到的专家。

Pre-gated MoE 的关键改动是：把第 $l$ 层的路由预测，改造成对第 $l+1$ 层专家集合的提前预测。这样，专家选择和专家执行不再在同一层串行发生，而是跨层流水化。论文给出的结论是，峰值 GPU 显存可近似写成“非 MoE 常驻参数 + 当前层活跃专家 + 下一层活跃专家”，相对 GPU-only 方案可把峰值显存压到约 $23\%$，也就是约 $4.2\times$ 的显存节省。

以 DeepSeek-V3 为代表的细粒度 MoE 更能体现这个思路的价值。它总参数 671B，但每个 token 只激活 37B；每个 MoE 层是 1 个共享专家加 256 个路由专家，每个 token 只选 8 个路由专家，加上共享专家一共 9 个。也就是说，按“专家个数”看，单 token 只激活约 $9/257 \approx 3.5\%$ 的专家；按“参数量”看，激活参数约占 $37/671 \approx 5.5\%$。剩余大部分专家如果仍长期占据 GPU，部署成本会失控。

---

## 问题定义与边界

问题可以精确表述为：在不明显降低模型质量的前提下，如何让大规模 MoE 在有限 GPU 显存上维持可接受的首 token 延迟和稳定吞吐。

这里有三个边界必须先说清：

| 边界 | 含义 | 直接后果 |
| --- | --- | --- |
| 只讨论推理，不讨论训练 | 目标是降低部署成本与延迟 | 优先考虑显存、I/O、调度 |
| 重点在解码阶段 | 解码是逐 token 生成，时间局部性更强 | 缓存和预取更容易生效 |
| 主要优化专家权重，不是注意力 | MoE 的大头通常在专家 FFN | 路由预测比算子微优化更重要 |

传统 MoE-offload 的瓶颈很直接。路由器，router，也叫 gate，可以理解为“决定当前 token 该找哪些专家处理”的模块。它先看输入，再选 top-K 专家。问题在于：只有选完专家，你才知道该从 CPU/NVMe 搬哪些权重到 GPU，于是“选专家”和“搬专家”天然串行，GPU 很容易等数据。

玩具例子可以这样看。假设一层有 8 个专家，GPU 只能放 2 个，每个 token 只激活 2 个专家。如果当前 token 需要专家 3 和 7，而它们都在 CPU，上层算完后你才知道要搬它们，那么 PCIe 传输时间就会直接暴露在这一步里。哪怕专家计算只要 0.4 ms，搬运却要 1.5 ms，GPU 也只能空等。

Pre-gated 想解决的正是这条串行依赖。论文中的峰值显存公式可以写成：

$$
\text{Peak GPU Mem} \approx \text{Non-MoE Mem} + \text{ActExp}^{l} + \text{ActExp}^{l+1}
$$

白话说就是：GPU 不需要长期存所有专家，只需要长期存非 MoE 部分，再临时存当前层和下一层会用到的活跃专家。这个边界很重要，因为它告诉你，预取做得好时，显存规模不再跟“总专家数”线性绑定，而主要跟“每层激活多少专家”绑定。

---

## 核心机制与推导

Pre-gated MoE 的机制可以拆成三步。

第一步，当前层执行自己的专家计算。  
第二步，同一时间用轻量预门函数 $\pi^l(h^l)$ 预测下一层会激活的专家集合 $S^{l+1}$。这里的 $h^l$ 可以理解为“第 $l$ 层当前 token 的中间表示”，也就是路由决策的依据。  
第三步，把 $S^{l+1}$ 对应的专家从 CPU 或 NVMe 提前拉到 GPU，并放入专家缓存。

形式化地写：

$$
S^l = \text{TopK}(g^l(h^{l-1}))
$$

这是传统路由，表示第 $l$ 层根据本层输入选当前层专家。

而 Pre-gated 改成：

$$
S^{l+1} = \pi^l(h^l)
$$

也就是第 $l$ 层额外产出“下一层专家预测结果”。

时间线上，传统 MoE 是：

| 时间片 | 传统 MoE |
| --- | --- |
| T1 | 计算 gate，选第 $l$ 层专家 |
| T2 | 从 CPU/NVMe 拉第 $l$ 层专家 |
| T3 | 执行第 $l$ 层专家 |
| T4 | 再为第 $l+1$ 层重复同样流程 |

Pre-gated 变成：

| 时间片 | Pre-gated MoE |
| --- | --- |
| T1 | 执行第 $l$ 层专家 |
| T1 并行 | 预测第 $l+1$ 层专家并预取 |
| T2 | 第 $l+1$ 层直接执行，大部分专家已在 GPU |
| T2 并行 | 再预测第 $l+2$ 层专家 |

这就是“计算和搬运重叠”。真正提升速度的不是预测本身，而是把 I/O 从关键路径上尽量移开。

真实工程例子是 DeepSeek-V3 这类细粒度 MoE。每层 257 个专家，但单 token 只用 9 个。如果你按全量常驻设计，单机 8 卡显存主要会浪费在“当前并不会被访问的专家”上；如果你只保留热点专家，且能在第 $l$ 层时把第 $l+1$ 层需要的 9 个左右专家预取到位，那么系统就更接近“37B 激活模型”的运行姿态，而不是“671B 总参数模型”的运行姿态。

---

## 代码实现

下面这个 Python 代码不是完整推理框架，而是一个可运行的最小模拟器，用来说明“预测下一层专家并预取”的收益。`assert` 用来验证：预取命中时，总时间应小于按需加载。

```python
from collections import OrderedDict

LOAD_MS = 3.0
COMPUTE_MS = 1.0

class ExpertCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = OrderedDict()

    def has(self, expert_id):
        return expert_id in self.data

    def touch(self, expert_id):
        if expert_id in self.data:
            self.data.move_to_end(expert_id)
        else:
            self.data[expert_id] = True
            if len(self.data) > self.capacity:
                self.data.popitem(last=False)

def run_decode(layers, predicted_next, cache_size=4):
    cache = ExpertCache(cache_size)
    total_ms = 0.0

    for i, experts in enumerate(layers):
        # 当前层如果没命中缓存，就按需加载
        for e in experts:
            if not cache.has(e):
                total_ms += LOAD_MS
                cache.touch(e)

        # 当前层计算
        total_ms += COMPUTE_MS

        # 预取下一层
        if i < len(predicted_next):
            for e in predicted_next[i]:
                if not cache.has(e):
                    cache.touch(e)

    return total_ms

# 三层，每层激活两个专家
layers = [
    [1, 2],
    [2, 3],
    [3, 4],
]

# 理想预测：第 0 层预测到第 1 层，第 1 层预测到第 2 层
prefetch_plan = [
    [2, 3],
    [3, 4],
    [],
]

# 不做预取
no_prefetch = [
    [],
    [],
    [],
]

t_no_prefetch = run_decode(layers, no_prefetch)
t_prefetch = run_decode(layers, prefetch_plan)

assert t_prefetch < t_no_prefetch
print(t_no_prefetch, t_prefetch)
```

如果把这个思路放到真实系统里，最常见的工程分层是：

1. 路由层负责产生当前层激活专家和下一层预测专家。
2. 调度层负责把预测专家映射到 GPU 缓存槽位。
3. 传输层负责 CPU→GPU 或 NVMe→CPU→GPU 的异步 DMA。
4. 执行层负责专家 GEMM 和合并输出。

如果你用 vLLM 部署 DeepSeek-V3，当前官方文档明确支持的是 Expert Parallel，也就是“把专家分散放在多张 GPU 上”，而不是论文里的完整 CPU/NVMe 预门卸载流水线。单机 8 卡示例命令是：

```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend pplx
```

这里 `--enable-expert-parallel` 控制专家并行，意思是专家权重按 EP 组分布到多卡；`--all2all-backend pplx` 控制专家间通信后端。它解决的是“专家如何跨 GPU 放置”，不是“如何从 CPU/NVMe 预取到 GPU”。后者通常要在推理引擎外再叠加 host-side cache 和异步搬运逻辑。

---

## 工程权衡与常见坑

最常见的误解是：只要有预取，I/O 就不是瓶颈。这个结论不成立。预取只有在“预测足够准，而且搬运能和计算重叠”时才有效。

| 问题/症状 | 根因 | 解决策略 |
| --- | --- | --- |
| GPU 仍经常空等 | 预测错，导致关键专家未提前到位 | 提高 pre-gate 准确率，降低预测跨度 |
| 显存抖动严重 | 错误预取太多，占满缓存 | 用 LRU/频次混合替换，给热点专家保留槽位 |
| NVMe 带宽打满 | 冷专家不断换入换出 | 增加 CPU 层缓存，减少直接从 NVMe 读取 |
| 吞吐下降但显存省了 | I/O 被隐藏不充分 | 拉长计算批次，增加传输与计算重叠窗口 |
| 多卡负载不均 | 热门专家集中在少数卡 | 用 expert rebalance 或冗余专家副本 |

一个简单估算很有用。假设每层还剩 $20\%$ 的专家没预取成功，每次冷加载要 3 ms，而该层专家计算只要 1 ms，那么层延迟近似变成：

$$
T_{\text{layer}} \approx 1 + 0.2 \times 3 = 1.6 \text{ ms}
$$

如果这个失配连续出现在几十层，尾延迟会非常明显。对线上服务来说，P99 往往比均值更先失控。

另一个坑是把“专家激活稀疏”误解成“显存一定省”。如果缓存策略很差，系统会不断把错误专家搬上 GPU，又很快驱逐，这种缓存抖动会让 PCIe/NVMe 流量高于按需加载。工程上，预门深度通常先保守设为只看下一层，而不是跨两层、三层预测，因为预测越远，不确定性越大。

---

## 替代方案与适用边界

Pre-gated 不是唯一方案，只是“把路由预测前移一层”这一路线最适合做跨层流水。

| 方案 | 数据流思路 | 缓存/预取策略 | 适合场景 | 局限 |
| --- | --- | --- | --- | --- |
| Pre-gated MoE | 当前层执行时预测下一层专家 | 以层间预测为核心 | 服务器级 GPU，强调吞吐和显存平衡 | 需要改模型结构或路由训练 |
| MoE-Infinity | 跟踪序列级激活轨迹 | activation-aware cache + prefetch | 个人机器、batch size 小的解码 | 更依赖历史轨迹，泛化受工作负载影响 |
| vib3 | 把权重页视作可检索对象 | GPU/RAM/NVMe 三层分页 + predictive prefetch | 单 GPU、本地运行、极致省显存 | 更偏系统重构，不是通用训练后直接替换 |

适用边界可以简单理解为：

如果你有多卡服务器，且能接受对模型路由做定制，Pre-gated 最有价值。  
如果你是单用户、小 batch、长解码，MoE-Infinity 这类“序列级激活跟踪”更现实。  
如果你目标是单 GPU 跑超大 MoE，vib3 这类三层分页方案更激进，但工程侵入性也更强。

需要特别强调一点：论文中 Pre-gated MoE 的性能数字来自 Switch Transformer 系列实验，不应直接当作 DeepSeek-V3 的线上实测值照搬。把它迁移到 DeepSeek-V3 这种 671B/37B 的系统上，结论应理解为“机制上成立、工程上很有必要”，而不是“已有公开论文按同样配置跑出了同样吞吐”。

---

## 参考资料

- Pre-gated MoE 论文，含预门设计、流水线、Equation 1、性能与显存结果: https://www.microsoft.com/en-us/research/uploads/prod/2024/05/isca24_pregated_moe_camera_ready.pdf
- vLLM Expert Parallel 官方文档，含单机 8 卡 DeepSeek-V3-0324 部署命令: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- DeepSeek-V3-0324 模型卡，含 671B 总参数、37B 激活参数: https://huggingface.co/deepseek-ai/DeepSeek-V3-0324
- DeepSeek-V3 技术报告镜像，含 1 shared + 256 routed experts、每 token 激活 8 routed experts: https://www.rivista.ai/wp-content/uploads/2025/01/2412.19437v1.pdf
- MoE-Infinity 论文摘要与入口，含 activation-aware expert cache/prefetch: https://www.emergentmind.com/papers/2401.14361
- vib3 官方站点，含三层分页、HNSW、predictive prefetch 的系统描述: https://vib3.dev/
