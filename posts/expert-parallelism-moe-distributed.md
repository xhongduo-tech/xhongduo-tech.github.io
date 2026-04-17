## 核心结论

Expert Parallel，中文常写作“专家并行”，指把 MoE 中的不同 Expert 分到不同 GPU 上存放。Expert 可以理解为“很多个并列的前馈子网络”，每个 token 只激活其中少数几个，而不是激活全部。

它解决的是“参数装不下”的问题，而不是“通信变少”的问题。参数不再在每张卡上完整复制，但 token 激活必须根据路由结果跨卡移动。对训练系统来说，MoE 层的主要跨 GPU 开销几乎都集中在 All-to-All 上。All-to-All 可以理解为“每张卡同时给所有其他卡发不同的数据包”。

最关键的工程事实有三点：

| 资源/负载 | Expert Parallel 下的状态 | 直接后果 |
| --- | --- | --- |
| Expert 参数 | 按 GPU 分片，不全量复制 | 单卡可承载更大的 MoE 参数量 |
| Token 激活 | 按路由结果做 All-to-All 交换 | 网络成为主要瓶颈 |
| Expert 计算 | 在本地 GPU 上执行 | 算力利用率取决于负载是否均衡 |

对一个 MoE 层，若批大小为 $B$，序列长度为 $S$，隐藏维度为 $h$，每个 token 选 top-$k$ 个专家，则一层前向中的总通信元素量可写成：

$$
V_{\text{layer}} = 2 \cdot B \cdot S \cdot k \cdot h
$$

这里的 2 表示“发过去一次，再发回来一次”。如果专家均匀分布到 $E$ 个专家、每组设备均匀承载，则每台设备视角下的跨卡通信可近似写成 $O(B \cdot S \cdot h \cdot k / E)$。这个量会随 batch、sequence、top-k 线性增长，很难靠增加算力自动消失。

---

## 问题定义与边界

MoE，Mixture of Experts，中文可理解为“专家混合层”。它把原来 Transformer 里的一个大前馈层，替换成多个并列专家，再由路由器决定每个 token 去找哪几个专家处理。

Expert Parallel 只讨论其中一个问题：当专家太多、单卡放不下时，如何把专家分散到多张 GPU 上，同时保证 token 能找到对应专家。

边界要讲清楚：

1. 它只改变 MoE 层的参数放置方式，不改变注意力层的基本结构。
2. 它依赖稀疏激活，也就是每个 token 只访问少数专家；如果每个 token 都访问所有专家，就退化成稠密模型。
3. 它的核心通信原语是 All-to-All，而不是 All-Reduce。All-Reduce 是“大家把同一份东西求和同步”，All-to-All 是“每个人给每个人寄不一样的数据”。

可以把边界画成一个简单框图：

```text
token on GPU0
   │
   ├─ router 选出 top-k expert
   │
   ├─ 需要跨卡的 token 片段 -> All-to-All
   │
   ├─ 到达目标 GPU 上的 local experts
   │
   └─ expert 输出 -> 反向 All-to-All -> 回到原 token 所在 GPU
```

玩具例子：

- 一共有 8 个专家，2 张 GPU。
- GPU0 放 expert 0,1,2,3。
- GPU1 放 expert 4,5,6,7。
- 每个 token 只选 top-2。
- 如果某个 token 被路由到 expert 1 和 expert 6，那么它的一份激活留在 GPU0，本地算；另一份激活必须发到 GPU1。

这说明 Expert Parallel 的本质不是“把计算均摊”，而是“把参数拆开后，再把 token 邮寄到参数所在处”。

---

## 核心机制与推导

先看一次前向的数据流。设输入张量形状是 $[B,S,h]$。

1. 路由器对每个 token 计算所有专家的分数。
2. 取 top-$k$，得到每个 token 要访问的专家编号。
3. 按专家编号把 token 重新分桶，并根据专家所在 GPU 做第一次 All-to-All。
4. 每张 GPU 对自己持有的专家执行本地 MLP。
5. 把专家输出按原 token 顺序做第二次 All-to-All，送回源 GPU。
6. 对 top-$k$ 结果做加权求和，恢复成标准 token 表示。

通信量推导并不复杂。总 token 数是 $B \cdot S$。每个 token 要发往 $k$ 个专家，每份激活长度为 $h$。一次 dispatch 需要发送：

$$
B \cdot S \cdot k \cdot h
$$

一次 combine 还要再发回来同样数量，因此单层总量是：

$$
V_{\text{layer}} = 2 \cdot B \cdot S \cdot k \cdot h
$$

如果考虑专家均匀分布，每张设备只持有约 $E_{\text{local}} = E / N$ 个专家，其中 $N$ 为设备数，则在理想均衡情况下，每台设备平均承担的消息量会按专家分摊，常写成近似的设备侧规模：

$$
V_{\text{device}} \approx O(B \cdot S \cdot h \cdot k / E)
$$

这里是近似量，不是严格闭式。因为真实系统里还要乘上“跨卡比例”“是否本地命中”“负载倾斜程度”等因子。它的作用是说明趋势：专家越多、分布越均匀，单设备平均消息负载越容易被摊薄；但 batch、sequence、hidden、top-k 增大时，通信会线性变重。

玩具例子再算一遍：

- $B=2$
- $S=4$
- $h=8$
- $k=2$

则一层总通信元素数为：

$$
2 \cdot 2 \cdot 4 \cdot 2 \cdot 8 = 256
$$

如果用 bf16，每个元素 2 字节，那么总通信字节数约为 $512$ 字节。这个例子很小，但公式结构已经完整。

真实工程例子用常见训练配置看更直观：

- batch = 32
- seq = 2048
- hidden = 4096
- top-k = 2
- 16 个 MoE 层
- bf16，2 字节

按公开示例估算，前向总通信约 32 GB。若互联带宽按 200 GB/s 粗估，顺序传输时间约为：

$$
32 / 200 = 0.16s = 160ms
$$

这就是为什么 MoE 常常不是算不动，而是“网不够快”。

为什么 All-to-All 难与计算完全重叠？原因有三个：

| 原因 | 本质 | 结果 |
| --- | --- | --- |
| 消息长度不规则 | 每张卡发给其他卡的 token 数不同 | buffer 难提前精确规划 |
| 路由结果动态变化 | 每个 batch 的通信图都不同 | 调度不可预测 |
| 依赖链强 | expert 计算要等 token 到齐 | overlap 空间有限 |

这和张量并行不同。张量并行的通信模式通常由层结构固定，比较规则；Expert Parallel 的通信模式受输入内容和路由器输出共同驱动，是“数据相关”的。

---

## 代码实现

下面给一个可运行的 Python 版本，只模拟路由、分桶和通信量统计，不依赖分布式环境。目的是把数据流说明白。

```python
from collections import defaultdict

def assign_experts_to_devices(num_experts: int, num_devices: int):
    assert num_experts % num_devices == 0
    experts_per_device = num_experts // num_devices
    mapping = {}
    for e in range(num_experts):
        mapping[e] = e // experts_per_device
    return mapping

def route_tokens_to_devices(token_topk_experts, expert_to_device):
    """
    token_topk_experts: List[List[int]]
    返回每个 token 需要发送到哪些 device
    """
    token_to_devices = []
    for experts in token_topk_experts:
        devices = [expert_to_device[e] for e in experts]
        token_to_devices.append(devices)
    return token_to_devices

def communication_elements(batch_size, seq_len, hidden_dim, top_k):
    return 2 * batch_size * seq_len * hidden_dim * top_k

def build_dispatch_matrix(token_to_devices, source_device_of_token):
    """
    统计 source_device -> dest_device 的消息数
    """
    matrix = defaultdict(int)
    for token_id, dest_devices in enumerate(token_to_devices):
        src = source_device_of_token[token_id]
        for dst in dest_devices:
            matrix[(src, dst)] += 1
    return dict(matrix)

# 玩具配置
num_experts = 8
num_devices = 2
expert_to_device = assign_experts_to_devices(num_experts, num_devices)

# 4 个 token，每个 token 选 top-2 experts
token_topk_experts = [
    [1, 6],  # 去 GPU0 和 GPU1
    [0, 3],  # 都在 GPU0
    [4, 7],  # 都在 GPU1
    [2, 5],  # 去 GPU0 和 GPU1
]

# 假设前两个 token 来自 GPU0，后两个 token 来自 GPU1
source_device_of_token = [0, 0, 1, 1]

token_to_devices = route_tokens_to_devices(token_topk_experts, expert_to_device)
dispatch = build_dispatch_matrix(token_to_devices, source_device_of_token)

assert expert_to_device[0] == 0
assert expert_to_device[7] == 1
assert token_to_devices[0] == [0, 1]
assert communication_elements(2, 4, 8, 2) == 256
assert dispatch[(0, 0)] == 3
assert dispatch[(0, 1)] == 1
assert dispatch[(1, 1)] == 3
assert dispatch[(1, 0)] == 1

print("expert_to_device =", expert_to_device)
print("token_to_devices =", token_to_devices)
print("dispatch =", dispatch)
print("comm_elements =", communication_elements(2, 4, 8, 2))
```

把它对应到真实训练框架，伪代码通常是下面这样：

```text
1. hidden_states -> router_logits
2. router_logits -> topk(expert_ids, weights)
3. permute tokens by target expert
4. all_to_all(dispatch tokens to expert owner GPUs)
5. local expert MLP forward
6. all_to_all(send outputs back)
7. unpermute + combine by topk weights
8. backward 重复相反方向的数据流
```

真实工程例子可以看 Megatron-Core 的 MoE 路径。它不是只做“路由 + A2A + MLP”，还会继续优化：

- `--overlap-moe-expert-parallel-comm`
- `--moe-router-fusion`
- `--moe-permute-fusion`
- `--moe-grouped-gemm`

这些优化分别对应“尽量重叠通信”“减少 router kernel 数量”“减少 token 重排开销”“把小矩阵乘合并成更大的矩阵乘”。

---

## 工程权衡与常见坑

Expert Parallel 的收益是参数可扩展，代价是系统更脆弱。最常见的问题不是公式算错，而是吞吐突然掉下来却很难定位。

| 坑 | 缓解手段 | 剩余限制 |
| --- | --- | --- |
| 路由倾斜，少数专家过载 | capacity factor、负载均衡损失 | 会引入丢 token、padding 或额外正则 |
| All-to-All 消息长度不一致 | 先分组、预分桶、静态 buffer 池 | 批间波动仍在 |
| top-k 太大导致通信暴涨 | 常用 top-1 或 top-2 | 可能影响模型质量 |
| 小 expert 计算碎片化 | grouped GEMM、专家分组 | 依旧受负载分布影响 |
| 通信难重叠 | A2A overlap、跨 microbatch 调度 | 不能保证完全隐藏 |
| 多并行叠加后拓扑复杂 | 限制 EP 组大小，优先单节点内 EP | 跨节点带宽常更差 |

一个新手容易忽略的点是：top-k 增加一倍，不只是“多算一点”，而是通信、重排、缓存压力一起增加。比如一个 token 选 5 个专家，另一个 token 只选 1 个专家，系统就像快递站点在做不等长分拣，最慢的那条路决定整个批次的收尾时间。这种“拖尾”现象就是 straggler，中文可理解为“慢尾设备”。

另一个真实工程问题是拓扑。若 expert parallel group 刚好跨节点，而节点间只有以太网或较慢互联，那么即使 GPU 算力很强，训练速度也会被网卡锁死。实践上通常优先让 EP 组尽量落在单节点内，用 NVLink 或 NVSwitch 吃下最重的 All-to-All。

---

## 替代方案与适用边界

不是所有 MoE 都必须用 Expert Parallel。要看你究竟卡在“参数容量”还是“网络带宽”。

| 方案 | 参数复制 | 通信压力 | 适用场景 |
| --- | --- | --- | --- |
| Data Parallel + Local MoE | 高，每卡都有全部专家 | 低，MoE 层不做跨卡 dispatch | 专家数量不大，单卡放得下 |
| Tensor Parallel + Expert Replication | 中到高，专家可能复制 | 中，主要是张量切分通信 | 单个 expert 太大，想保留规则通信 |
| Expert Parallel | 低，专家按卡分片 | 高，All-to-All 是核心瓶颈 | 专家总参数很大，单卡放不下 |
| 混合并行 DP + TP + EP | 可控 | 高，但可按拓扑拆分 | 大规模训练集群 |

可以用一句话区分：

- “每卡复制所有专家”适合参数还没大到爆炸，但希望系统简单。
- “专家分散到多卡”适合参数极大，但前提是你有足够好的互联带宽和更复杂的调度能力。

适用边界也很明确：

1. 如果专家总参数远超单卡显存，EP 基本是必选项。
2. 如果网络很弱，EP 可能不如更保守的复制方案。
3. 如果路由极不均衡，EP 的理论节省会被慢尾和空转吃掉。
4. 如果 expert 本身也大到单卡放不下，就要把 EP 和 TP 叠加使用。

所以 Expert Parallel 不是“更先进就一定更好”，而是“在参数规模足够大、互联足够强时才值得”。

---

## 参考资料

- [Training MoEs at Scale with PyTorch](https://docs.pytorch.org/blog/training-moes/)：PyTorch 官方博客，适合建立直觉，重点解释“移动 token 而不是移动权重”的系统思路。
- [Expert Parallelism: Distributed Computing for MoE Models](https://mbrenndoerfer.com/writing/expert-parallelism-distributed-moe-training)：对通信量公式、All-to-All 两阶段、32 GB/160 ms 的估算讲得最清楚。
- [Mixture of Experts, Megatron Core User Guide](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/user-guide/features/moe.html)：偏工程实现，重点是 EP A2A overlap、router fusion、permute fusion、grouped GEMM 等优化手段。
- [Expert Parallelism: Scaling Sparse MoEs](https://www.emergentmind.com/topics/expert-parallel-ep)：偏综述，适合快速看 Expert Parallel 的定义、负载与通信建模。
- [Sparse Expert Parallelism in MoE Architectures](https://www.emergentmind.com/topics/sparse-expert-parallelism)：补充混合并行和稀疏专家系统的更广泛背景。
