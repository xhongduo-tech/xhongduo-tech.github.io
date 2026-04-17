## 核心结论

MoE，Mixture of Experts，意思是“把一个大前馈层拆成很多专家网络，只让每个 token 走其中少数几个”，它解决的是“总参数量继续增大，但每次实际计算不要全量展开”的问题。结论先给出：**MoE 的大瓶颈常常不是算力不够，而是显存容量和显存带宽不够。**

原因很直接。每个 token 只会被路由到 top-k 个专家，例如 top-2，但为了保证下一次路由随时可用，绝大多数专家权重仍要常驻 HBM。HBM 是高带宽显存，白话讲就是 GPU 最快但也最贵的内存。于是出现一个反直觉现象：**总模型很大，真正参与本次计算的参数却很少；GPU 不是忙着算，而是忙着等权重和缓存数据被搬运。**

以 Mixtral 8x7B 为例，FloE 论文给出的数字很典型：FP16 下总共约需 94GB VRAM，但解码时每个 token 真正涉及的激活参数约 27.3GB，约占 30%；其余约 66.8GB 虽然占着 HBM，却不会在这个 token 上被访问。这正是“容量很满，但有效访问很稀疏”的内存墙。

| 项目 | 是否常驻 HBM | 是否每个 token 都访问 | Mixtral 8x7B 量级示意 |
| --- | --- | --- | --- |
| 注意力与共享部分 | 是 | 基本是 | 一直参与 |
| 全部专家权重 | 是 | 否 | 总量很大 |
| 当前 token 激活的 top-k 专家 | 是 | 是 | 约 27.3GB |
| 未激活专家 | 是 | 否 | 约 66.8GB |
| KV Cache | 是 | 是 | 随上下文增长 |

所以理解 MoE，不能只看“激活参数少，FLOP 低”，还要看“未激活专家是否仍占着 HBM，以及当前 token 到底访问了多少字节”。这也是为什么很多 MoE 系统在小 batch、低延迟推理时，表现得更像**带宽受限系统**而不是**计算受限系统**。

---

## 问题定义与边界

本文讨论的是**推理阶段**，尤其是在线服务常见的 decode 阶段，也就是一次只生成一个新 token 的过程。这里最关键的指标不是训练吞吐，而是 TPOT，Time Per Output Token，意思是“每生成一个 token 需要多少时间”。

边界先说明清楚：

| 场景 | 主要瓶颈 | 是否适合本文分析 |
| --- | --- | --- |
| 小 batch、低延迟在线推理 | HBM 带宽、PCIe/NVLink 传输、缓存命中 | 是 |
| 大 batch 离线推理 | 吞吐、跨卡调度、流水线利用率 | 是，但结论要换成吞吐视角 |
| 训练阶段 | 反向传播、优化器状态、激活重算 | 不是本文重点 |
| 稠密模型 | 算法和带宽模型更简单 | 只作为对照 |

一个新手最容易误判的点是：既然每个 token 只激活 2 个专家，那是不是只需要加载这 2 个专家就行？答案通常不是。因为路由结果是运行时才知道的，若专家都不在 GPU，就要临时从 CPU 甚至 NVMe 搬过来。PCIe 是 CPU 和 GPU 之间的高速总线，白话讲就是“比磁盘快很多，但仍远慢于显存内部读写”。FloE 与相关工程资料都说明了同一个现实：**把专家放到更慢的层级，容量问题缓解了，但延迟问题会立刻出现。**

真实工程里，DeepSeek-R1 这类 MoE 部署更能说明边界。Introl 的工程总结给出：在 batch size = 1 的单 token 推理下，仍可能需要 1,040GB/s 以上的带宽预算；如果进入更高激活或更高吞吐场景，带宽需求会继续上升。结论是：**单卡即便勉强装下模型，也不代表能以可接受延迟跑起来。**

玩具例子可以先看一个极简层：1 层、8 个专家、top-2 路由、每个专家 1GB。  
这时总专家容量是 8GB，但某个 token 真正访问的专家权重只有 2GB。  
如果你用“总容量 8GB”去估计这个 token 的带宽压力，会高估 4 倍。  
如果你忽略“其余 6GB 仍要常驻显存”，又会低估容量压力。  
**MoE 的难点正是在这两个维度同时成立：容量按全量算，访问按稀疏算。**

真实工程例子则更接近这样：两张或多张 80GB GPU 存热点专家，CPU DRAM 存冷专家，NVMe 再存更冷的数据页；请求到来时先根据路由历史和预测结果预取下一层可能命中的专家，再把当前层计算与下一层拷贝重叠。这个架构不是优化项，而是很多大模型 MoE 服务能否上线的前提。

---

## 核心机制与推导

传统 MBU，Memory Bandwidth Utilization，意思是“内存带宽利用率”，常写成

$$
\text{MBU}=\frac{B_{\text{achieved}}}{B_{\text{peak}}}
=\frac{(S_{\text{model}}+S_{KV})/TPOT}{B_{\text{peak}}}
$$

这里的问题是，$S_{\text{model}}$ 把所有专家都算进去了，默认它们都在本 token 上被访问。对稀疏激活的 MoE，这个假设不成立。

MoE-CAP 提出的 S-MBU，Sparse Memory Bandwidth Utilization，意思是“稀疏感知的带宽利用率”，核心改动是把总模型大小换成**实际被激活的参数大小**：

$$
\text{S-MBU}=\frac{B_{\text{achieved}}}{B_{\text{peak}}},\quad
B_{\text{achieved}}=\frac{S_{\text{activated}}+S_{KV}}{TPOT}
$$

其中

$$
S_{\text{activated}}=n_{\text{layer}}\cdot S_{\text{attn}}+\sum_{l=1}^{n_{\text{layer}}}\sum_{i=1}^{n_{\text{expert}}}\mathbf{1}[l,i]\cdot S_{\text{expert}}
$$

符号解释如下：

| 符号 | 含义 | 白话解释 |
| --- | --- | --- |
| $TPOT$ | time per output token | 生成一个 token 花多久 |
| $S_{KV}$ | KV cache size | 为注意力保存的历史键值缓存 |
| $\mathbf{1}[l,i]$ | 指示变量 | 第 $l$ 层第 $i$ 个专家这次有没有被用到 |
| $S_{\text{expert}}$ | 单个专家参数大小 | 一个专家权重占多少内存 |
| $B_{\text{peak}}$ | 峰值带宽 | 设备理论最快显存带宽 |

这个定义解决了什么问题？  
它把“模型装了多少”与“本次实际读了多少”区分开了。

继续用玩具例子推导。假设：

- 1 层有 8 个专家
- 每个专家大小为 $S$
- top-2 路由
- 当前 token 只命中专家 3 和专家 5
- 忽略注意力与 KV cache，先只看专家部分

那么：

- 传统 MBU 视角下，访问量近似是 $8S$
- S-MBU 视角下，访问量是 $2S$

如果 GPU 峰值带宽是 1000 GB/s，TPOT 是 2 ms，且 $S=1$ GB，则

- 传统估算带宽需求：$\frac{8}{0.002}=4000$ GB/s
- 稀疏估算带宽需求：$\frac{2}{0.002}=1000$ GB/s

前者会让你误以为系统完全不可跑，后者才接近真实访问压力。这也是论文里报告“传统 MBU 可能高估 260% 以上，而 S-MBU 与实际 profile 误差可低于 1%”的原因。

但要注意，S-MBU 纠正的是**访问估计**，不是在说那 6 个没激活的专家不占显存。它们仍然构成容量压力，仍然决定你能不能单卡部署。于是 MoE 出现了一个非常关键的二分法：

1. **容量问题**：所有专家放不放得下。  
2. **带宽问题**：当前 token 真正访问的数据能否足够快地送到计算单元。

这两个问题同时存在，且常常互相牵制。你把专家全放 HBM，容量压力大但带宽路径最短；你把专家放到 CPU/NVMe，容量压力降了，但 I/O 延迟会放大。

---

## 代码实现

工程上常见做法是把 GPU 显存当作热点缓存，只保留近期最常用专家；CPU DRAM 或 NVMe 作为冷存储。最简单可讲清楚机制的结构是 **LRU 缓存 + 预取**。LRU，Least Recently Used，意思是“最近最少使用优先淘汰”。

下面这个 Python 代码是一个可运行的玩具实现。它不依赖 GPU，只模拟“命中就复用，未命中就加载，缓存满了就淘汰最久没用的专家”。

```python
from collections import OrderedDict

class ExpertCache:
    def __init__(self, cache_size, experts_in_cpu):
        self.cache_size = cache_size
        self.experts_in_cpu = experts_in_cpu
        self.cache_gpu = OrderedDict()
        self.load_count = 0
        self.evict_count = 0

    def _load_to_gpu(self, expert_id):
        if len(self.cache_gpu) >= self.cache_size:
            self.cache_gpu.popitem(last=False)
            self.evict_count += 1
        self.cache_gpu[expert_id] = self.experts_in_cpu[expert_id]
        self.load_count += 1

    def ensure_loaded(self, required_ids):
        for expert_id in required_ids:
            if expert_id in self.cache_gpu:
                self.cache_gpu.move_to_end(expert_id)
            else:
                self._load_to_gpu(expert_id)

    def forward(self, routed_experts):
        unique_ids = []
        seen = set()
        for expert_id in routed_experts:
            if expert_id not in seen:
                seen.add(expert_id)
                unique_ids.append(expert_id)
        self.ensure_loaded(unique_ids)
        return [self.cache_gpu[i] for i in unique_ids]

experts_cpu = {i: f"expert-{i}-weights" for i in range(8)}
cache = ExpertCache(cache_size=3, experts_in_cpu=experts_cpu)

out1 = cache.forward([1, 3])      # miss 2 次
out2 = cache.forward([3, 5])      # 3 命中，5 miss
out3 = cache.forward([6, 1])      # 可能触发淘汰

assert out1 == ["expert-1-weights", "expert-3-weights"]
assert out2 == ["expert-3-weights", "expert-5-weights"]
assert cache.load_count >= 4
assert cache.evict_count >= 1
assert len(cache.cache_gpu) <= 3
```

这个玩具例子说明了最基本的控制流，但真实系统至少还要补上三件事：

| 组件 | 作用 | 如果没有会怎样 |
| --- | --- | --- |
| 异步拷贝 | 后台搬运专家 | 前台计算会被 I/O 卡住 |
| 预取 | 提前猜下一个要用的专家 | miss 时延迟陡增 |
| 分页权重 | 把专家切成更细粒度页 | 整个专家搬运太重 |

更接近真实工程的做法是：

1. 当前层开始前，路由器得到 top-k 候选。
2. 若专家已在 GPU 缓存，直接执行。
3. 若未命中，则发起 CPU→GPU 非阻塞拷贝。
4. 计算当前已就绪部分时，同时预取下一层或下一 token 可能使用的专家。
5. 缓存不足时按 LRU 或负载感知策略淘汰冷专家。

MoE-Lightning 的 CGOPipe 就属于更系统化的实现。它把 CPU、GPU、I/O 三条路径做成流水线，并配合 paged weights 与 HRM 性能模型来决定缓存和调度策略。白话讲，这类系统不是单纯“少拷一点”，而是尽量把“拷贝时间塞进计算空隙里”。

---

## 工程权衡与常见坑

真正落地时，MoE 的坑集中在三类：装不下、搬不动、分不匀。

| 常见坑 | 现象 | 根因 | 常见对策 |
| --- | --- | --- | --- |
| OOM | 模型根本放不进单卡 | 全部专家常驻 HBM | 多卡切片、量化、压缩、offload |
| PCIe 传输过慢 | 每 token 延迟暴涨 | miss 后临时搬专家 | LRU、预取、异步 memcpy、paged weights |
| 专家负载不均 | 某些卡很忙，某些卡接近空转 | 路由偏斜 | 负载均衡 loss、Pre-gated、SiDA、在线监控 |

第一个坑最好理解。即使“激活参数少”，全量专家也可能先把显存占满。Mixtral 8x7B 在 FP16 下约 94GB，就是很多单卡环境跨不过去的门槛。

第二个坑更隐蔽。很多人会想：不常用的专家放 CPU，不就行了吗？问题是 CPU 到 GPU 的链路远慢于 HBM 内部带宽。Hugging Face 论坛里给了一个直观量级：11.3B 参数通过 PCIe 4.0 x16 按需加载，单步可能带来约 0.7 秒级额外延迟。这个数字不一定适用于所有模型，但足够说明问题：**按需加载不是免费午餐。**

第三个坑是负载不均。MoE 路由不是平均撒到所有专家上，热门专家会反复命中，冷门专家长期闲置。如果专家跨 GPU 分布，这就会演化成“部分 GPU 带宽打满，部分 GPU 几乎没活”。Pre-gated 的思路是提前选择下一块会用到的专家，让迁移和当前计算重叠；SiDA 一类方法则更强调服务阶段的数据感知调度。两者共同目标都是减少“算的时候等搬，搬完后又局部热点过载”的情况。

实际排障时，一个很有价值的监控方式是同时看：

- HBM 容量占用
- PCIe/NVLink 吞吐
- 每层专家命中分布
- S-MBU 随 batch 和请求类型的变化

如果只看 GPU 利用率，常常会误判。GPU 利用率低，不一定是模型小，可能是大量时间堵在带宽和拷贝链路上。

---

## 替代方案与适用边界

如果目标是“低延迟在线服务”，最直接的路线仍然是多 GPU 放热点专家，尽量减少跨层级搬运。如果目标变成“单卡也能跑起来”，就要接受额外压缩、额外调度、甚至一定质量损失。

| 策略 | VRAM 压力 | 带宽压力 | 延迟特征 | 适用场景 |
| --- | --- | --- | --- | --- |
| 全量驻留 GPU | 高 | 中 | 最稳定 | 数据中心、多卡高配 |
| GPU + CPU/NVMe 分层 offload | 中 | 高 | miss 时抖动明显 | 容量受限但可做系统优化 |
| 压缩 + 缓存 + offload | 低 | 中 | 依赖压缩质量与预测命中 | 消费级 GPU、个人实验 |
| 激进量化/裁剪 | 很低 | 低 | 可能最快 | 对精度不敏感任务 |

FloE 代表的是“压缩 activated experts，再做 on-the-fly 推理”的路线。它给出的一个强结论是：对 Mixtral-8x7B，可把专家参数压到原来的约 1/9.3，使部署门槛降到约 11GB VRAM 量级。这个方向适合什么边界？适合**内存非常紧、但可以接受少量质量下降**的场景，例如单张消费级显卡上的实验与个人应用。

相对地，MoE-Lightning 更偏向高吞吐 batch 推理。它说明另一件事：**不是所有 offload 都是为低延迟服务设计的。** 有些系统把更多精力放在吞吐最大化，而不是单 token 延迟最小化。若你是做离线批处理、日志总结、批量生成，CPU-GPU-I/O 三级流水可能非常划算；若你是做交互式对话，用户会直接感知 miss 带来的抖动。

因此选型可以用一句话概括：

- 显存够、追求稳定延迟：优先全 GPU 或多 GPU 热点驻留。
- 显存不够、但还能做系统工程：分层 offload + 缓存 + 预取。
- 显存极紧、先求能跑：压缩专家，再结合缓存。
- 如果业务极度依赖稳定低延迟，且硬件预算有限：稠密小模型有时比大 MoE 更实用。

---

## 参考资料

1. FloE: On-the-Fly MoE Inference on Memory-constrained GPU  
https://openreview.net/pdf/d855fa57c8d9ddf99ad636b57872872119a31f7a.pdf

2. MoE-CAP / Sparse Memory Bandwidth Utilization  
https://openreview.net/pdf/d5bcbda04df81a4e636c112b90f96543b3b7fe5b.pdf

3. MoE-CAP Evaluation Framework 概述  
https://www.emergentmind.com/topics/moe-cap-evaluation-framework

4. Mixture of Experts Infrastructure: Scaling Sparse Models for Production AI  
https://introl.com/blog/mixture-of-experts-moe-infrastructure-scaling-sparse-models-guide

5. MoE Expert Offloading to CPU/NVMe  
https://apxml.com/courses/mixture-of-experts-advanced-implementation/chapter-4-efficient-moe-inference/expert-offloading

6. MoE-Lightning: High-Throughput on Memory-Limited GPUs  
https://www.emergentmind.com/papers/2411.11217

7. Hugging Face 论坛关于按需加载专家延迟的讨论  
https://discuss.huggingface.co/t/is-this-possible/163679/2

8. 负载均衡与推理调度相关综述，含 Pre-gated 与 SiDA-MoE 引用  
https://openreview.net/pdf/d8241f7aea6825a2c63da849cc73a980faa2d6dc.pdf
