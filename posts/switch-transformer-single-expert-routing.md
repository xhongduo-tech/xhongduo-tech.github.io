## 核心结论

Switch Transformer 的核心改动只有两点，但影响很大。

第一，**Top-1 路由**：路由就是“决定一个 token 交给哪个专家处理”的分发规则。Switch 不再像早期 MoE 那样把一个 token 发给两个专家，而是只发给一个专家。这样做的直接结果是，前向计算、反向梯度、跨设备通信都近似减半。

第二，**容量因子 CF**：容量因子就是“每个专家预留多少缓冲位”的放大系数。理想情况下，每个专家平均接收 $\frac{T}{N}$ 个 token，其中 $T$ 是总 token 数，$N$ 是专家数。Switch 实际给每个专家分配的容量是：

$$
\text{capacity} = \left\lceil \gamma \cdot \frac{T}{N} \right\rceil
$$

其中 $\gamma$ 就是 capacity factor，常记作 CF，$\lceil \cdot \rceil$ 表示向上取整。

这两个设计组合起来，得到一个很实用的工程结论：**Top-1 负责把 MoE 做得足够快，CF 负责把 Top-1 做得足够稳。**  
在很多训练与推理场景里，CF 从 1.0 提高到 1.25，通常就能把溢出 token 的丢弃率显著压低，同时仍保留高吞吐。

| 方案 | 每个 token 激活专家数 | 通信量 | 梯度量 | 溢出风险 |
|---|---:|---:|---:|---:|
| Top-1 | 1 | 低 | 低 | 较高，需要 CF 缓冲 |
| Top-2 | 2 | 约翻倍 | 约翻倍 | 较低 |
| Top-k | $k$ | 随 $k$ 增长 | 随 $k$ 增长 | 更低，但更贵 |

玩具例子可以这样理解：一个客服工单只指派给“最匹配的一个处理员”，这就是 Top-1；如果每个处理员的待办列表只留刚好平均数那么多位置，稍微一波峰值就会塞满；CF=1.25 的意思是每个人多留出 25% 空位，用来吸收短时波动。

再把这个结论说得更直接一些：

| 组件 | 解决的问题 | 没解决的问题 |
|---|---|---|
| Top-1 | 降低计算和通信成本 | 不保证负载天然均匀 |
| CF | 降低短时拥塞导致的 drop | 不会主动把热点 token 改派给别的专家 |
| 负载均衡损失 | 约束路由器别长期偏向少数专家 | 不能替代容量设计 |

所以看 Switch Transformer，不要把它理解成“更聪明地选专家”，而应该理解成“**用足够简单的路由规则，把稀疏专家层做成可训练、可扩展、可部署的系统**”。

---

## 问题定义与边界

MoE，中文常叫“混合专家网络”，可以理解成“很多个前馈子网络并排放着，但每次只激活其中少数几个”。它的目标不是让所有参数都参与每次计算，而是用稀疏激活换更大的模型容量。

先把普通 Transformer 的前馈层和 MoE 前馈层区别说清楚：

| 结构 | 每个 token 会经过谁 | 计算特点 |
|---|---|---|
| 稠密 FFN | 同一个前馈网络 | 所有参数每次都参与 |
| MoE FFN | 少数几个专家网络 | 总参数大，但单次激活少 |
| Switch FFN | 只激活 1 个专家 | 更极端的稀疏激活 |

Switch Transformer 讨论的核心问题不是“专家有没有能力”，而是**token 能否被稳定、便宜地送到正确专家**。这里有三个边界条件：

1. token 总量是离散的，而且会波动。  
2. 专家容量是固定 buffer，不是无限队列。  
3. 分发常常跨设备发生，因此通信成本很高。

设：

$$
T = \text{batch\_size} \times \text{seq\_len}
$$

若 `batch_size=32`、`seq_len=512`，则：

$$
T = 32 \times 512 = 16384
$$

如果有 16 个专家，那么每个专家的理想平均负载是：

$$
\bar{N} = \frac{T}{N} = \frac{16384}{16} = 1024
$$

这时如果 `CF=1.0`，每个专家容量就是：

$$
C = \left\lceil 1.0 \times \frac{16384}{16} \right\rceil = 1024
$$

等于“只按平均值开槽位”。这在数学上看起来合理，但在工程上几乎总会偏紧，因为路由器不会把每一批 token 分得完全均匀。只要某个专家短时收到 1100 或 1200 个 token，就会出现溢出。

所以问题的真正定义是：

- 如何在 **Top-1 单专家分发** 的前提下，控制专家拥塞与 token 丢弃？
- 如何在 **不把通信成本重新抬高** 的前提下，让路由足够稳定？

这里的边界也很明确：

- Switch 解决的是**稀疏专家层的路由效率问题**，不是所有 Transformer 层都适用。
- CF 控制的是**静态容量上限**，不是动态调度器。
- 当 batch 很小、token 总量不足时，负载不均会更严重，CF 往往要更保守。
- 当专家跨节点部署时，通信成本比单机更敏感，Top-1 的收益更明显。

再补一个常被忽略的边界：**Switch 不是在“挑最强专家”，而是在“做受容量约束的分流”**。  
路由器即使预测正确，只要目标专家已满，该 token 仍可能被 drop。也就是说，路由质量和系统容量是两套约束，不能混为一谈。

---

## 核心机制与推导

先看路由。对每个 token，路由器会输出一个对专家的打分向量：

$$
p(x) = \text{softmax}(W_r x)
$$

其中 $x$ 是 token 表示，$W_r$ 是路由器参数。Top-1 的做法是只取分数最高的那个专家：

$$
e^\* = \arg\max_j p_j(x)
$$

白话说，路由器只选“当前最合适的一个专家”，不会再复制一份给第二名专家。

如果把这个过程拆成更容易理解的三步，就是：

| 步骤 | 数学动作 | 直白解释 |
|---|---|---|
| 1 | $z=W_r x$ | 给每个专家打分 |
| 2 | $p=\text{softmax}(z)$ | 把打分变成概率分布 |
| 3 | $e^\*=\arg\max_j p_j$ | 只选择概率最高的专家 |

### 1. 容量因子 CF

设总 token 数为 $T$，专家数为 $N$，容量因子为 $\gamma$。每个专家允许接收的最大 token 数是：

$$
C = \left\lceil \gamma \cdot \frac{T}{N} \right\rceil
$$

实际实现中通常都要取整数，常见做法是向上取整，避免容量因为浮点截断而偏小。

继续用前面的例子：

- `T=16384`
- `N=16`

则理想负载为 1024。

当 `CF=1.0` 时：

$$
C = \lceil 1.0 \times 1024 \rceil = 1024
$$

当 `CF=1.25` 时：

$$
C = \lceil 1.25 \times 1024 \rceil = 1280
$$

这多出来的 256 个槽位就是缓冲区。  
如果某个专家实际收到 1400 个 token，那么：

- `CF=1.0` 时要丢 $1400-1024=376$ 个
- `CF=1.25` 时要丢 $1400-1280=120$ 个

这就是 CF 的作用：**不是改变平均负载，而是给波动留余量。**

还可以把它写成“容量利用率”视角：

$$
u_i = \frac{N_i}{C}
$$

其中 $u_i$ 是第 $i$ 个专家的容量利用率。

- 若 $u_i < 1$，说明该专家还有空位
- 若 $u_i = 1$，说明该专家刚好满载
- 若 $u_i > 1$，说明该专家发生溢出

这个写法有助于理解监控面板上常见的“expert utilization”。

### 2. 溢出率 DT

为了度量“专家超载得有多严重”，可以用溢出率：

$$
DT = \frac{\sum_i \text{ReLU}(N_i - \gamma \bar{N})}{\sum_i N_i}
$$

其中：

- $N_i$ 是第 $i$ 个专家实际收到的 token 数
- $\bar{N}=\frac{T}{N}$ 是平均负载
- $\gamma$ 是 CF
- $\text{ReLU}(z)=\max(0,z)$，白话说就是“只统计超出的部分，不统计没满的部分”

这个公式的含义很直接：  
先算每个专家超出容量多少，再把所有超载 token 加起来，最后除以总 token 数，得到“全局有多少 token 因容量不足而面临被丢弃”。

如果实现里确实采用“超出容量直接丢弃”，那么 drop rate 可以近似写成：

$$
\text{drop\_rate} \approx DT
$$

更严格一点地说，在“先 Top-1 分配、再按容量截断”的实现里：

$$
\text{drop\_rate} = \frac{\sum_i \max(0, N_i - C)}{T}
$$

把 $C=\left\lceil \gamma \cdot \frac{T}{N} \right\rceil$ 代入，就能直接对应到代码统计逻辑。

### 3. 为什么 Top-1 更快

Top-2 路由的本质问题不是算法写起来麻烦，而是**每个 token 要复制成两份 dispatch**。这会带来三类额外成本：

- 前向时，两个专家都要算一遍
- 反向时，两个专家路径都要传梯度
- 分布式时，两份 token 都要跨设备搬运

所以 Top-1 的收益主要来自数据移动，而不是某个局部算子的小优化。

把 Top-1 和 Top-2 的系统代价并排看，会更清楚：

| 项目 | Top-1 | Top-2 |
|---|---:|---:|
| 单 token dispatch 次数 | 1 | 2 |
| 专家前向次数 | 1 | 2 |
| 反向梯度路径 | 1 条 | 2 条 |
| all-to-all 传输载荷 | 1 份 | 约 2 份 |
| 容量管理难度 | 低 | 中 |

因此 Switch 的关键不是“理论上更优”，而是“**稀疏性做得足够狠，系统收益立刻出现**”。

### 4. 一个玩具例子

假设有 8 个 token、2 个专家，理想上每个专家接 4 个。

实际路由结果：

- 专家 A 收到 6 个
- 专家 B 收到 2 个

当 `CF=1.0` 时，容量为 4：

- A 超载 2 个
- B 空闲 2 个
- 总 drop 比例是 $\frac{2}{8}=25\%$

当 `CF=1.25` 时，容量为 5：

- A 只超载 1 个
- B 仍有空位
- 总 drop 比例降到 $\frac{1}{8}=12.5\%$

注意，CF 没有让负载变均匀，它只是减少“因为不均匀而立刻损失样本”的程度。真正让负载更均匀的，通常还要靠 load balancing loss，也就是“负载均衡损失”，它会惩罚路由器长期偏爱少数专家。

为了让这个点更清楚，再看一个对照表：

| 现象 | 只加 CF 能否解决 | 需要负载均衡损失吗 |
|---|---|---|
| 偶发的短时超载 | 能缓解 | 不一定 |
| 某几个专家长期过热 | 不能根治 | 需要 |
| 整体通信过大 | 不能 | 需要更换路由策略 |
| token 大量被 drop | 可能缓解 | 通常也要一起看 |

### 5. 负载均衡损失为什么必要

Switch 论文除了 Top-1 和 CF，还强调了辅助负载均衡损失。原因很简单：如果路由器长期把大量 token 发往少数几个专家，那么：

- 热门专家经常爆满
- 冷门专家几乎收不到样本
- 参数利用率会很差
- drop 会持续存在，而不是偶发波动

一种常见写法会同时看两个量：

- 路由概率在专家维度上的平均值
- 真正被分配到专家的 token 比例

记：

$$
P_i = \frac{1}{T}\sum_{t=1}^{T} p_i(x_t)
$$

$$
F_i = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}(e_t^\*=i)
$$

其中：

- $P_i$ 表示路由器“主观上”给专家 $i$ 的平均概率
- $F_i$ 表示专家 $i$ “客观上”实际收到的 token 占比

Switch 的辅助损失会鼓励 $P_i$ 和 $F_i$ 更均衡，防止少数专家长期成为热点。  
对新手来说，只需要记住一句话：**CF 是“扩车位”，负载均衡损失是“别总往同一个停车场开”**。

---

## 代码实现

下面给一个可直接运行的简化版 Switch dispatch。它省略了真实框架里的 all-to-all 通信、padding tensor 和反向传播细节，但保留了最关键的四步：

1. 计算每个 token 的 Top-1 专家  
2. 按 CF 算每个专家容量  
3. 超过容量的 token 直接标记为 dropped  
4. 输出负载、容量利用率和溢出统计

```python
from math import ceil
from typing import Dict, List, Tuple


def softmax(logits: List[float]) -> List[float]:
    max_logit = max(logits)
    exps = [pow(2.718281828459045, x - max_logit) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]


def top1_expert(logits: List[float]) -> int:
    return max(range(len(logits)), key=lambda i: logits[i])


def switch_dispatch(
    router_logits: List[List[float]],
    capacity_factor: float,
) -> Dict[str, object]:
    """
    简化版 Switch Top-1 dispatch。

    参数：
        router_logits: shape = [num_tokens, num_experts]
        capacity_factor: 容量因子 CF

    返回：
        capacity: 每个专家的容量上限
        expert_load: 每个专家实际接收的 token 数（未截断前）
        accepted_counts: 每个专家成功接收的 token 数
        dispatch: (token_id, expert_id, slot_id)
        dropped: 被容量截断的 token_id
        drop_rate: 丢弃率
        utilization: 每个专家的容量利用率
    """
    if not router_logits:
        raise ValueError("router_logits must not be empty")

    num_tokens = len(router_logits)
    num_experts = len(router_logits[0])
    if num_experts == 0:
        raise ValueError("num_experts must be positive")

    for row in router_logits:
        if len(row) != num_experts:
            raise ValueError("All rows in router_logits must have same length")

    capacity = ceil(capacity_factor * num_tokens / num_experts)

    # expert_load 统计“路由器想发给该专家多少 token”
    expert_load = [0] * num_experts
    # accepted_counts 统计“容量限制后实际收下多少 token”
    accepted_counts = [0] * num_experts

    dispatch: List[Tuple[int, int, int]] = []
    dropped: List[int] = []

    for token_id, logits in enumerate(router_logits):
        expert_id = top1_expert(logits)
        expert_load[expert_id] += 1

        if accepted_counts[expert_id] < capacity:
            slot_id = accepted_counts[expert_id]
            dispatch.append((token_id, expert_id, slot_id))
            accepted_counts[expert_id] += 1
        else:
            dropped.append(token_id)

    utilization = [count / capacity for count in accepted_counts]
    overflow = [max(0, load - capacity) for load in expert_load]
    drop_rate = len(dropped) / num_tokens

    return {
        "capacity": capacity,
        "expert_load": expert_load,
        "accepted_counts": accepted_counts,
        "dispatch": dispatch,
        "dropped": dropped,
        "drop_rate": drop_rate,
        "utilization": utilization,
        "overflow": overflow,
    }


def print_report(name: str, result: Dict[str, object]) -> None:
    print(f"=== {name} ===")
    print("capacity:", result["capacity"])
    print("expert_load:", result["expert_load"])
    print("accepted_counts:", result["accepted_counts"])
    print("overflow:", result["overflow"])
    print("utilization:", [round(x, 2) for x in result["utilization"]])
    print("dropped:", result["dropped"])
    print("drop_rate:", round(result["drop_rate"], 4))
    print("dispatch:", result["dispatch"])
    print()


if __name__ == "__main__":
    # 8 个 token，2 个专家，其中前 6 个 token 更偏向专家 0
    router_logits = [
        [4.0, 1.0],
        [3.8, 1.1],
        [3.6, 1.2],
        [4.2, 0.9],
        [3.9, 1.0],
        [4.1, 0.8],
        [0.7, 2.8],
        [0.6, 3.0],
    ]

    res_cf_1 = switch_dispatch(router_logits, capacity_factor=1.0)
    res_cf_125 = switch_dispatch(router_logits, capacity_factor=1.25)

    assert res_cf_1["capacity"] == 4
    assert res_cf_125["capacity"] == 5

    # 专家 0 想接收 6 个 token，所以两种 CF 下分别溢出 2 个和 1 个
    assert res_cf_1["overflow"] == [2, 0]
    assert res_cf_125["overflow"] == [1, 0]

    assert len(res_cf_1["dropped"]) == 2
    assert len(res_cf_125["dropped"]) == 1
    assert res_cf_125["drop_rate"] < res_cf_1["drop_rate"]

    print_report("CF=1.0", res_cf_1)
    print_report("CF=1.25", res_cf_125)
```

如果直接运行，输出的关键信息会是：

| 场景 | capacity | expert\_load | dropped | drop rate |
|---|---:|---|---:|---:|
| `CF=1.0` | 4 | `[6, 2]` | 2 | 25% |
| `CF=1.25` | 5 | `[6, 2]` | 1 | 12.5% |

这段代码对应的工程语义是：

- `expert_load` 是路由器原本想发给各专家的 token 数
- `accepted_counts` 是容量限制后专家真正接收的 token 数
- `dispatch` 记录 token 到专家、以及在专家内部第几个槽位
- `dropped` 记录超载后被丢弃的 token
- `overflow` 记录每个专家理论上超出容量多少
- 真正的分布式实现会把 `dispatch` 转成按专家打包的 tensor，再做 all-to-all

如果写成伪代码，大致是：

| 步骤 | 动作 | 目的 |
|---|---|---|
| 1 | 计算 router logits | 给每个 token 一个专家偏好分数 |
| 2 | 取 Top-1 expert | 保持单专家激活 |
| 3 | 计算 `capacity = ceil(CF * T / N)` | 给每个专家设容量上限 |
| 4 | 若专家未满则写入 buffer | 完成 token 分发 |
| 5 | 若专家已满则 drop | 控制峰值负载 |
| 6 | 统计每层 drop rate | 用于调参和监控 |

真实工程例子里，这段逻辑通常出现在 MoE 层前面。比如大模型推理服务中，一个 batch 可能会被切到多个 GPU 节点上的专家组；Top-1 可以减少跨卡数据复制，而 CF 让热点专家不会因为瞬时拥塞把太多 token 丢掉。

如果再往真实实现靠近一步，通常还会有这些额外环节：

| 真实系统步骤 | 这里为什么省略 |
|---|---|
| 按专家重排 token tensor | 需要框架张量操作，不影响理解 CF |
| all-to-all 通信 | 依赖多卡/多机环境 |
| 专家输出再按原 token 顺序还原 | 属于 dispatch 的逆过程 |
| 辅助负载均衡损失回传 | 需要训练图而不是纯 Python 示例 |

所以这段代码的定位不是“论文完整复现”，而是“把 Switch 的容量与丢弃机制讲明白”。

---

## 工程权衡与常见坑

Top-1 + CF 不是“最优理论方案”，而是“最强工程折中方案”。它的优点很明确，但坑也很集中。

### 1. CF 太低会导致性能抖动

`CF≈1.0` 时，容量几乎等于平均值。只要路由稍微偏斜，就会产生连续 drop。训练中这会表现为：

- 有效参与训练的 token 变少
- 某些专家长期过热，另一些专家几乎没学到
- 验证集指标波动变大

如果把问题说得更具体一点，drop 不是“日志里不好看”这么简单，它会直接改变参与前向和反向的样本集合。也就是说，**路由拥塞会变成训练信号缺失**。

### 2. CF 太高会浪费资源

`CF>1.5` 往往开始明显浪费 buffer 和通信。因为很多专家并没有真的吃满这些槽位，但系统已经为这些空槽完成了预留、padding、甚至传输。

| CF | 典型现象 | drop 率 | 资源浪费 |
|---:|---|---|---|
| 1.0 | 容量贴边运行 | 高 | 低 |
| 1.1 | 略有缓冲 | 中 | 较低 |
| 1.25 | 常见折中点 | 低 | 可接受 |
| 1.5 | 很少 drop | 很低 | 明显上升 |

工程上常见误区是把 CF 当成“越大越稳”的万能旋钮。实际上 CF 增大后，通常会同时推高：

- buffer 预留空间
- padding 比例
- token 打包后的无效传输
- 某些 kernel 的低效计算比例

所以 CF 本质上是在买保险，但保险也有成本。

### 3. 不要只调 CF，不看负载均衡

很多初学者会把“drop 多”直接理解为“容量不够”，于是一路加大 CF。这样可能把问题掩盖掉，但没有解决真正原因。真正常见的问题是路由器偏置，也就是某些专家总被选中。

所以监控指标至少要有三类：

- 每层 drop rate
- 每个专家的 token 占比
- load balancing loss 的收敛情况

调整流程通常是：

| 阶段 | 关注点 | 动作 |
|---|---|---|
| 监控 | 哪些层 drop 高 | 定位热点层 |
| 诊断 | 是否少数专家过热 | 看专家负载分布 |
| 调参 | 先调负载均衡，再调 CF | 避免只靠加容量掩盖问题 |
| 评估 | 吞吐、延迟、精度一起看 | 防止单指标优化 |

可以把这个原则记成一句更简单的话：**CF 解决“车位不够”，负载均衡解决“车辆总往一个入口挤”**。

### 4. 上下文长度变化会改变最优 CF

如果推理场景的 `seq_len` 差异很大，token 总数会直接变化。短上下文、小 batch 时，负载统计噪声更大；长上下文、大 batch 时，平均效应更稳定。  
因此很多系统不会所有场景共用一个 CF，而是按训练、离线推理、在线推理分别调。

这个现象可以从方差角度理解。样本越少，路由分布越容易偏离平均值；样本越多，专家负载越接近期望值。  
所以：

| 场景 | token 总量 | 负载波动 | CF 倾向 |
|---|---:|---:|---:|
| 小 batch 在线推理 | 小 | 大 | 偏保守 |
| 大 batch 离线推理 | 大 | 小 | 可更激进 |
| 大规模训练 | 中到大 | 中 | 常取折中值 |

### 5. 一个真实工程例子

在 Switch-style 或类似稀疏专家推理系统里，工程目标常常不是“零 drop”，而是“在可接受 drop 下最大化吞吐”。例如一些稀疏 MoE 推理实验会发现，某层允许一部分超载 token 被丢弃后，整体延迟显著下降。这说明现实系统追求的是：

- 少量质量损失是否可接受
- 换来的吞吐和时延收益是否足够大

如果一个在线问答系统对延迟极敏感，那么少量 drop 可能比 Top-2 更合理；但如果是离线高质量生成或训练阶段，通常就会更保守。

把这个取舍写成决策表更直观：

| 目标 | 更可能选择 |
|---|---|
| 极致吞吐、严格时延预算 | Top-1 + 较小 CF |
| 稳定训练、尽量少 drop | Top-1 + 中等 CF + 强负载均衡 |
| 更重视质量、成本次要 | Top-2 或更保守容量 |
| 资源受限但通信昂贵 | 优先保留 Top-1 |

---

## 替代方案与适用边界

Switch 不是唯一方案，只是把“实现复杂度 / 训练稳定性 / 系统吞吐”三个目标压到了一个较优点。

### 1. Top-2 路由

Top-2 可以理解成“主负责人 + 副负责人”。同一个 token 会送到两个专家，最后再把输出组合。它的好处是更稳，专家利用率往往更高，drop 压力更小。缺点也直接：

- 通信更多
- 激活更多
- 反向更重
- 实现更复杂

所以 Top-2 更适合对训练质量非常敏感、且硬件预算足够的场景。

### 2. 动态 capacity

静态 CF 是“一开始就给每个专家固定容量”。动态 capacity 则尝试根据当前 batch 分布实时调专家 buffer。这样可以进一步减少空槽浪费，但实现很麻烦，因为：

- buffer 不再规则
- 通信打包更复杂
- 并行硬件不喜欢动态形状

换句话说，动态 capacity 更像“理论上更省”，但未必“系统上更快”。

### 3. 自适应路由

还有一类方案不只是调容量，而是直接让路由策略随负载变化，比如在热点专家过载时，把部分 token 转移给次优专家。这类方法理论上更灵活，但要处理更多训练稳定性问题。

其难点通常不在想法，而在约束冲突：

- 路由要看语义匹配
- 路由又要看当前负载
- 还要兼顾可微训练或近似可微训练
- 最终还要落在硬件友好的规则张量上

| 方案 | 通信成本 | drop 风险 | 实现复杂度 | 适用场景 |
|---|---|---|---|---|
| Top-1 + 静态 CF | 低 | 中 | 低 | 大规模训练与高吞吐推理 |
| Top-2 | 高 | 低 | 中 | 更重视稳定性和精度 |
| Top-k | 很高 | 更低 | 高 | 研究型探索 |
| 动态 CF | 中 | 低 | 高 | 对资源利用率要求极高 |
| 自适应路由 | 中到高 | 低 | 很高 | 有强系统能力的团队 |

对初学者，可以记成一句话：

- Top-1：一个 token 只找一个负责人
- Top-2：一个 token 同时找主、副两个负责人
- CF：不是多派负责人，而是给负责人多留几个座位

适用边界也很重要。若你的模型规模不大、专家数不多、通信不是瓶颈，那么 Switch 的收益未必明显；但在专家跨设备、batch 大、路由拥塞频繁的系统里，Top-1 + CF 往往是性价比很高的默认选择。

最后把适用边界压缩成一张判断表：

| 条件 | Switch 是否合适 | 原因 |
|---|---|---|
| 专家跨卡或跨节点 | 很合适 | Top-1 能直接减通信 |
| batch 足够大 | 合适 | 负载统计更稳定 |
| 模型参数预算很大 | 合适 | 稀疏激活能放大容量 |
| 小模型、单卡部署 | 收益有限 | 通信不是主要矛盾 |
| 对每个 token 的质量极端敏感 | 需要谨慎 | drop 代价可能过高 |

---

## 参考资料

1. **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**  
   原论文，提出 Top-1 路由、辅助负载均衡损失与 Switch 的整体设计。  
   链接：https://arxiv.org/abs/2101.03961

2. **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**  
   更早期的经典 MoE 论文，适合理解为什么“多专家但稀疏激活”在大模型中有意义，也能作为理解 Switch 为什么要简化 Top-2/Top-k 的背景材料。  
   链接：https://arxiv.org/abs/1701.06538

3. **Matthias Brenndoerfer 的 Switch Transformer 讲解笔记**  
   适合理解 CF、专家容量和 overflow/drop 的数值例子，尤其适合快速建立直觉。  
   链接：https://mbrenndoerfer.com/writing/switch-transformer-top-1-routing-trillion-parameter-scaling

4. **deep-paper 对 Switch Transformer 的解读**  
   适合先看概念，再回到原论文，重点是 Top-1 为什么能明显降低系统成本。  
   链接：https://deep-paper.org/en/papers/2025-10/2101.03961/

5. **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding**  
   GShard 是 Switch 的重要前序工作。它能帮助理解 Top-2 路由、分片和专家并行是如何连接起来的。  
   链接：https://arxiv.org/abs/2006.16668

6. **OpenReview 上关于稀疏专家推理与 token drop 的相关实验材料**  
   可用于理解现实系统中“允许少量 drop 以换吞吐”的工程取舍。  
   链接：https://openreview.net/pdf/e518bf31fdf55d21e4a5adacb1fa61ea26a48518.pdf
