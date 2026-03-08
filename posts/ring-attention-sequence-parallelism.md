## 核心结论

Ring Attention 解决的不是“把 attention 的总计算量降到比标准 self-attention 更低”，而是把**全局自注意力**改写成一种适合多设备执行的流程：**每台设备长期保留本地的 Query 块，只让 Key/Value 块在设备之间按环形顺序轮转**。  
如果把 Query 理解成“当前 token 在问什么”，把 Key/Value 理解成“其他 token 能提供什么索引和内容”，那么 Ring Attention 的本质就是：

- `Q` 固定在本地，不到处搬运。
- `K,V` 按块在设备间传递。
- 每台设备经过若干轮后，都让自己的 `Q` 看到了全局所有 `K,V`。
- 因此最终结果仍然是**精确的全局 attention**，不是近似，也不是局部窗口。

它的核心收益有两个。

第一，**上下文长度可以随设备数近似线性扩展**。  
如果单卡最多只能处理长度为 `L` 的序列块，`N` 台设备组成环后，理论上可以把总上下文扩到接近 `N × L`，同时保持标准 attention 的数学结果不变。

第二，**通信有机会被计算掩盖**。  
只要块大小 `c` 满足

$$
c \ge \frac{F}{B}
$$

就有机会让“当前块的 attention 计算时间”不小于“下一块 KV 的传输时间”。这里：

- `F` 是单设备有效计算吞吐，单位可理解为 FLOP/s。
- `B` 是单向链路带宽，单位可理解为 Byte/s。
- `c` 是每个 block 含多少个 token。

满足这个条件后，KV 轮转带来的通信，不再一定直接暴露在总耗时上，而是可能被流水线吸收。

一个最小直觉例子如下。  
假设长文本被切成 4 个 block，分别放在 4 张 GPU 上。每张 GPU 固定保留自己的 `Q`，先和本地 `K,V` 算一次，再把本地 `K,V` 发送给下一张卡，同时接收上一张卡的 `K,V`。经过 4 轮后，每张卡都已经和全局 4 个 block 的 `K,V` 配对过，因此输出与“单机直接做全局 attention”的结果一致。

---

## 问题定义与边界

普通 self-attention 的问题不只是公式里的 $O(n^2)$ 计算量，还包括三件事会一起膨胀：

| 维度 | 为什么会变难 |
|---|---|
| 计算量 | token 数翻倍，`QK^T` 的配对数近似变成四倍 |
| 激活/中间状态 | 长序列训练时需要保存更多中间结果，显存很快吃满 |
| 跨设备同步 | 一旦单卡装不下，就要分布式切分；切分后必须保证不同设备仍能算出全局结果 |

对新手来说，可以先记住一句最重要的话：

> 我们不是想让一张卡硬吃下 100K token，而是让多张卡各自负责一段序列，同时还能得到与单机全局 attention 相同的结果。

Ring Attention 处理的是这个问题里的一个特定子问题：  
**长上下文下，如何在多设备上做精确的全局 self-attention，并把通信组织成可流水化的序列并行。**

它不改变注意力公式本身，仍然是标准的

$$
\text{Attention}(Q,K,V)
=
\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

其中：

- `Q` 是查询矩阵，形状通常为 `n × d`
- `K` 是键矩阵，形状通常为 `n × d`
- `V` 是值矩阵，形状通常为 `n × d_v`
- `d` 是 head dimension

Ring Attention 改的不是公式，而是**数据放置方式和执行顺序**：

- 标准实现：同一设备一次性拿到整段 `Q,K,V`
- Ring 实现：每台设备只长期持有本地 `Q`，`K,V` 分块在设备间轮转

下面这个对比表先建立整体直觉：

| 方案 | 长期驻留数据 | 通信模式 | 最大上下文扩展 | 是否精确全局 attention |
|---|---|---|---|---|
| 单机普通 Attention | 单设备持有全量 `Q,K,V` | 无 | 受单卡显存限制 | 是 |
| Sequence Parallel（简单切分） | 每卡持有部分序列 | 常见为 all-gather / reduce-scatter | 中等 | 可以做到精确 |
| Local Window Attention | 只保留局部窗口 | 很少或无 | 很长 | 否 |
| Ring Attention | 每卡固定本地 `Q`，分轮看到全部 `K,V` | 点对点环形 send/recv | 近似按设备数线性扩展 | 是 |

一个常见误解是：

> “既然所有 `K,V` 最后还是都会被每张卡看到，那是不是内存压力并没有变小？”

不是。关键区别在于**是否同时保存全量上下文**。

- 普通全局 attention：常常需要在同一时刻拿到大量全局数据
- Ring Attention：每一轮只处理当前手里的一个远端 `K,V` block，算完继续转发
- 所以每张设备的**长期驻留内存**只需要保留本地 `Q`、局部状态和少量缓冲区，而不是全量 `K,V`

这就是它适合超长上下文的原因。

但它也有明确边界，不是任何场景都该用：

| 场景 | 是否优先考虑 Ring Attention | 原因 |
|---|---|---|
| 上下文只有几千 token | 通常否 | 普通 FlashAttention 已经足够 |
| 设备数很少，如 2 到 4 张 | 未必 | 通信组织复杂度可能盖过收益 |
| 互联较差，如普通以太网 | 通常否 | 环形流水线容易被延迟打断 |
| 上下文极长，单卡放不下 | 是 | 这是它最擅长的场景 |
| 希望保持精确全局 attention | 是 | 它不是近似方法 |

---

## 核心机制与推导

设：

- 总共有 `N` 台设备
- 每台设备负责一个本地序列块
- 每个 block 长度为 `c`
- 每个 token 的特征维度为 `d`

把整段序列切成多个 block 后，第 `i` 台设备长期持有：

$$
Q_i,\ K_i,\ V_i
$$

在 Ring Attention 中，第 `i` 台设备真正“固定不动”的核心是 `Q_i`。  
`K_i,V_i` 在初始化时也在本地，但之后会被发给下一台设备，并接收上一台设备传来的 `K,V`。

### 每轮到底做什么

第 `i` 台设备在某一轮手里拿着一个当前的远端块 `K_j,V_j`，它会执行三件事：

1. 用固定的 `Q_i` 和当前 `K_j,V_j` 算一轮局部 attention 分块结果
2. 把 `K_j,V_j` 发给下一台设备
3. 从上一台设备接收新的 `K,V`，进入下一轮

经过 `N` 轮后，`Q_i` 已经与所有 `K_0...K_{N-1}`、`V_0...V_{N-1}` 都交互过，因此能恢复出完整的全局 attention 输出。

如果只看数学等价性，可以把它写成：

$$
O_i
=
\text{Attention}(Q_i,\ [K_0;K_1;\dots;K_{N-1}],\ [V_0;V_1;\dots;V_{N-1}])
$$

Ring 做的只是把这个“大矩阵一次算完”的过程，改成“按块分轮累积”。

### 玩具例子

假设有 4 台设备、4 个 block，分别记为 `B0, B1, B2, B3`。

| 轮次 | GPU0 当前 KV | GPU1 当前 KV | GPU2 当前 KV | GPU3 当前 KV |
|---|---|---|---|---|
| 0 | `B0` | `B1` | `B2` | `B3` |
| 1 | `B3` | `B0` | `B1` | `B2` |
| 2 | `B2` | `B3` | `B0` | `B1` |
| 3 | `B1` | `B2` | `B3` | `B0` |

以 GPU0 为例：

- 它的 `Q0` 始终不动
- 第 0 轮见到 `B0`
- 第 1 轮见到 `B3`
- 第 2 轮见到 `B2`
- 第 3 轮见到 `B1`

虽然顺序不是 `B0,B1,B2,B3`，但这不影响最终结果，因为 attention 最终关心的是“`Q0` 是否见到了全部 `K,V`”，而不是块的到达顺序。

### 为什么“不能把每轮 softmax 结果直接相加”

这是新手最容易踩的坑之一。

错误想法是：

1. 每轮算一次 `softmax(Q_i K_j^\top / \sqrt d)V_j`
2. 然后把这些轮次结果直接加起来

这在数学上不对，因为 softmax 的归一化分母是**全局的**，不是每个 block 各算各的。

标准 attention 对某一行查询的输出可写成：

$$
o
=
\frac{\sum_{t} e^{s_t} v_t}{\sum_{t} e^{s_t}}
\quad,\quad
s_t=\frac{q\cdot k_t}{\sqrt d}
$$

如果把 `t` 拆成多个 block，正确做法是维护：

- 当前见过分数的最大值 `m`
- 当前归一化分母 `l`
- 当前加权和 `a`

也就是在线 softmax / log-sum-exp 归并。  
当新到一个 block，先算这个 block 的局部最大值和局部和，再与旧状态做稳定合并，而不是把各块 softmax 后的输出直接相加。

这是 Ring Attention 与 FlashAttention 常一起出现的原因之一：  
**两者都依赖 blockwise 的数值稳定归并。**

### 为什么通信可以被隐藏

设每轮传输一个 KV block 的字节量近似为：

$$
\text{bytes}_{\text{comm}} \approx 4cd
$$

这里把常数项折算进 `4`，重点不是常数，而是通信量随 `c·d` 线性增长。

因此单轮通信时间近似为：

$$
t_{\text{comm}}=\frac{4cd}{B}
$$

再看计算。  
若本轮是一个 `Q` block 与一个 `K,V` block 做 attention，其主要计算量可近似写为：

$$
\text{flops}_{\text{comp}} \approx 4c^2d
$$

于是单轮计算时间近似为：

$$
t_{\text{comp}}=\frac{4c^2d}{F}
$$

如果希望“当前轮计算”足以覆盖“下一轮 KV 传输”，要求：

$$
t_{\text{comp}} \ge t_{\text{comm}}
$$

代入得：

$$
\frac{4c^2d}{F}\ge \frac{4cd}{B}
$$

约掉公共项 `4cd`，得到：

$$
c \ge \frac{F}{B}
$$

这条式子的意思很直接：

- `c` 太小：每轮算得太快，GPU 很快就算完，然后等数据
- `c` 足够大：GPU 正在算当前块时，下一块 `K,V` 已经在传，通信更容易被掩盖

变量解释如下：

| 符号 | 含义 | 白话解释 | 常见单位 |
|---|---|---|---|
| $c$ | block size | 每个 block 有多少 token | token |
| $d$ | hidden dimension | 每个 token 的特征宽度 | 维度数 |
| $F$ | effective FLOPS | 单设备有效计算吞吐 | FLOP/s |
| $B$ | bandwidth | 单向链路带宽 | Byte/s |

这也是为什么工程里常见 `block size ≈ 1K` 一类设定。  
它通常不是随手拍的超参，而是由“计算和通信是否能重叠”共同决定的。

### 一个更完整的执行视角

如果把一层 attention 的分布式执行展开，可以理解成：

| 阶段 | 每台设备在做什么 |
|---|---|
| 初始 | 保留本地 `Q_i`，同时持有初始本地 `K_i,V_i` |
| 第 1 轮 | 用 `Q_i` 对当前 `K_i,V_i` 计算局部块贡献 |
| 发送/接收 | 把当前 `K,V` 发给下一台，从上一台收新的 `K,V` |
| 中间轮次 | 对收到的远端 `K,V` 重复同样过程 |
| 结束 | 已见完整个环上的全部 `K,V`，完成全局输出归并 |

这个流程的关键不是“环”这个词本身，而是：

- **固定本地 `Q`**
- **流动远端 `K,V`**
- **按块归并**
- **通信与计算流水化**

---

## 代码实现

下面先给一个**可直接运行的 Python 示例**。  
它做三件事：

1. 用标准全局 attention 直接计算结果
2. 用“Ring 轮转 + 在线 softmax 归并”计算结果
3. 验证两者数值一致

这个实现不依赖 GPU，也不依赖通信库，但保留了 Ring Attention 的数学本质。

```python
import math


def transpose(x):
    return [list(row) for row in zip(*x)]


def matmul(a, b):
    rows, cols, inner = len(a), len(b[0]), len(b)
    out = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            aik = a[i][k]
            for j in range(cols):
                out[i][j] += aik * b[k][j]
    return out


def full_attention(q, k, v):
    d = len(q[0])
    scores = matmul(q, transpose(k))
    scale = 1.0 / math.sqrt(d)

    out = []
    for row_scores in scores:
        scaled = [s * scale for s in row_scores]
        m = max(scaled)
        exps = [math.exp(x - m) for x in scaled]
        denom = sum(exps)

        row_out = [0.0 for _ in range(len(v[0]))]
        for w, v_row in zip(exps, v):
            for j in range(len(v_row)):
                row_out[j] += w * v_row[j]
        row_out = [x / denom for x in row_out]
        out.append(row_out)
    return out


def block_scores(q, k):
    d = len(q[0])
    scale = 1.0 / math.sqrt(d)
    scores = matmul(q, transpose(k))
    return [[x * scale for x in row] for row in scores]


def ring_attention_single_query_block(q_block, k_blocks_in_ring_order, v_blocks_in_ring_order):
    """
    对一个本地 Q block 做 Ring 风格累积。
    使用在线 softmax 归并，保证与全局 attention 数学等价。
    """
    num_rows = len(q_block)
    value_dim = len(v_blocks_in_ring_order[0][0])

    running_m = [-float("inf")] * num_rows
    running_l = [0.0] * num_rows
    running_a = [[0.0 for _ in range(value_dim)] for _ in range(num_rows)]

    for k_block, v_block in zip(k_blocks_in_ring_order, v_blocks_in_ring_order):
        scores = block_scores(q_block, k_block)

        for r in range(num_rows):
            local_scores = scores[r]
            local_m = max(local_scores)
            local_exp = [math.exp(x - local_m) for x in local_scores]
            local_l = sum(local_exp)

            local_a = [0.0 for _ in range(value_dim)]
            for weight, v_row in zip(local_exp, v_block):
                for j in range(value_dim):
                    local_a[j] += weight * v_row[j]

            new_m = max(running_m[r], local_m)

            old_scale = 0.0 if running_m[r] == -float("inf") else math.exp(running_m[r] - new_m)
            new_scale = math.exp(local_m - new_m)

            new_l = old_scale * running_l[r] + new_scale * local_l
            new_a = [
                old_scale * running_a[r][j] + new_scale * local_a[j]
                for j in range(value_dim)
            ]

            running_m[r] = new_m
            running_l[r] = new_l
            running_a[r] = new_a

    out = []
    for r in range(num_rows):
        out.append([x / running_l[r] for x in running_a[r]])
    return out


def rotate_blocks_right(xs, step):
    n = len(xs)
    return [xs[(i - step) % n] for i in range(n)]


def run_demo():
    # 4 个设备，每个设备 2 个 token，每个 token 维度 2
    q_blocks = [
        [[1.0, 0.0], [0.5, 0.5]],
        [[0.0, 1.0], [1.0, -1.0]],
        [[1.0, 1.0], [0.0, -1.0]],
        [[-1.0, 1.0], [0.2, 0.8]],
    ]
    k_blocks = [
        [[1.0, 0.0], [0.5, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[1.0, 1.0], [-1.0, 1.0]],
        [[-1.0, 1.0], [0.0, -1.0]],
    ]
    v_blocks = [
        [[10.0, 0.0], [5.0, 5.0]],
        [[0.0, 20.0], [8.0, 1.0]],
        [[30.0, 30.0], [-10.0, 10.0]],
        [[7.0, -3.0], [4.0, 12.0]],
    ]

    all_k = []
    all_v = []
    for kb, vb in zip(k_blocks, v_blocks):
        all_k.extend(kb)
        all_v.extend(vb)

    direct_outputs = [full_attention(qb, all_k, all_v) for qb in q_blocks]

    num_devices = len(q_blocks)
    ring_outputs = []

    for i in range(num_devices):
        k_seen = []
        v_seen = []
        for step in range(num_devices):
            owner = (i - step) % num_devices
            k_seen.append(k_blocks[owner])
            v_seen.append(v_blocks[owner])

        ring_out = ring_attention_single_query_block(q_blocks[i], k_seen, v_seen)
        ring_outputs.append(ring_out)

    for dev_id, (direct, ring) in enumerate(zip(direct_outputs, ring_outputs)):
        for row_id, (a_row, b_row) in enumerate(zip(direct, ring)):
            for col_id, (a, b) in enumerate(zip(a_row, b_row)):
                if abs(a - b) > 1e-9:
                    raise AssertionError(
                        f"Mismatch at device={dev_id}, row={row_id}, col={col_id}: {a} vs {b}"
                    )

    print("Direct attention and ring attention are numerically identical.")
    for dev_id, out in enumerate(ring_outputs):
        print(f"device {dev_id}: {out}")


if __name__ == "__main__":
    run_demo()
```

这段代码比“把所有见过的 `K,V` 收集起来再算一次”更接近 Ring 的真实数学过程，因为它显式实现了：

- 分块处理
- 逐轮累积
- 在线 softmax 归并

你可以直接运行：

```bash
python ring_attention_demo.py
```

预期会输出：

```text
Direct attention and ring attention are numerically identical.
...
```

### 代码里最重要的不是轮转，而是归并

新手读这段代码时，建议只盯住三个量：

| 状态 | 作用 |
|---|---|
| `running_m` | 已见块里的分数最大值，用于数值稳定 |
| `running_l` | 归一化分母的累计值 |
| `running_a` | `exp(score) * value` 的累计加权和 |

最终输出就是：

$$
\text{output} = \frac{\text{running\_a}}{\text{running\_l}}
$$

因此，Ring 的难点并不是“怎么写一个 for 循环轮转 KV”，而是“怎么在分块情况下仍然得到与全局 softmax 完全一致的结果”。

### 更接近工程实现的伪代码

真实实现一般会把计算流和通信流拆开，用双缓冲做流水线。伪代码如下：

```python
# pseudo code
q = local_q_block
kv_cur = local_kv_block
recv_buf0 = alloc_kv_buffer()
recv_buf1 = alloc_kv_buffer()
recv_buf = recv_buf0

state = init_online_softmax_state()

for step in range(world_size):
    send_req = isend(kv_cur, dst=next_rank, stream=comm_stream)
    recv_req = irecv(recv_buf, src=prev_rank, stream=comm_stream)

    # 当前计算流处理手里的 kv_cur
    scores = q @ kv_cur.k.T / sqrt(d)
    state = online_softmax_merge(state, scores, kv_cur.v, stream=compute_stream)

    wait(recv_req)
    kv_cur = recv_buf
    recv_buf = recv_buf1 if recv_buf is recv_buf0 else recv_buf0

    wait(send_req)

out = finalize_online_softmax(state)
```

这段伪代码体现了真实工程里的 4 个关键点：

| 关键点 | 含义 |
|---|---|
| `Q` 固定 | 本地查询块不轮转 |
| `K,V` 流动 | 每轮只处理当前拿到的远端块 |
| 双缓冲 | 当前 buffer 在算，下一 buffer 在收 |
| 在线归并 | 保持与全局 softmax 数学一致 |

### 环上的发送与接收关系

设总设备数为 `N`，设备编号为 `i`，则每轮的拓扑关系是：

| 项 | 公式 |
|---|---|
| 接收来源 | $(i - 1 + N)\bmod N$ |
| 发送目标 | $(i + 1)\bmod N$ |
| 本地长期不动的数据 | $Q_i$ |
| 本地会被轮转的数据 | 初始时的 $K_i,V_i$ |

这个关系看起来简单，但它直接决定了系统会不会死锁。  
只要某一轮某个 rank 的发送目标和别的 rank 的接收来源对不上，整个环就可能卡死。

---

## 工程权衡与常见坑

Ring Attention 的价值来自“细粒度流水线”，它的难点也来自“细粒度流水线”。

相比一次性 collective，它更依赖：

- 稳定的设备拓扑
- 正确的 send/recv 协议
- 合适的 block 大小
- 数值稳定的分块归并

下面先看最常见的问题表：

| 问题 | 典型现象 | 根因 | 处理方式 |
|---|---|---|---|
| block 太小 | GPU 经常空转等数据 | $c < F/B$，通信无法被计算盖住 | 增大 block size |
| send/recv 顺序错误 | 程序挂起或死锁 | 环上协议不一致 | 固定每轮通信顺序，统一 rank 逻辑 |
| 只有单缓冲 | 计算与接收串行化 | 当前 buffer 还在用，下一轮数据无处接收 | 使用双缓冲或环形缓冲 |
| 网络抖动大 | 吞吐波动明显 | 跨机延迟或带宽不稳定 | 优先 NVLink / InfiniBand，减少跨交换机跳数 |
| 分块 softmax 合并错误 | 结果与基线不一致，训练不稳定 | 错把局部 softmax 当全局 softmax | 使用 online softmax / log-sum-exp 归并 |
| 序列不够长 | 实测收益不明显 | 通信组织成本高于节省的显存价值 | 退回 FlashAttention 或普通 sequence parallel |

### 坑 1：把“每轮结果相加”当成正确实现

这是最常见的数学错误。  
只要你在代码里看见这种逻辑，就要警惕：

```python
acc += softmax(local_scores) @ local_v
```

如果 `local_scores` 只对应当前 block，这通常不对。  
原因前面已经说过：softmax 的归一化是全局的，不能按块独立归一化后直接相加。

正确思路应该是：

```python
state = merge_block_into_online_softmax(state, local_scores, local_v)
```

也就是维护全局稳定归并状态，而不是把块结果硬拼起来。

### 坑 2：死锁

在环上做点对点通信时，死锁很容易出现。典型错误模式是：

- 所有设备都先做阻塞式 `send`
- 但没有设备先进入匹配的 `recv`
- 于是整个环同时堵住

工程里常见的规避方法有：

| 方法 | 作用 |
|---|---|
| 非阻塞 `isend/irecv` | 避免所有 rank 一起卡在阻塞调用上 |
| 明确的轮次协议 | 每一轮固定“发给谁、收谁”的关系 |
| 双缓冲 | 当前计算数据与下一轮接收数据分开 |
| 单独通信流 | 减少通信和计算互相阻塞 |

下面这个伪代码更接近工程实践：

```cpp
// pseudo C++/CUDA/NCCL
Tensor kv_buf[2];
cudaEvent_t recv_done[2];
int cur = 0;
int nxt = 1;

for (int step = 0; step < world_size; ++step) {
    ncclRecv(kv_buf[nxt].data(), count, dtype, prev_rank, comm, comm_stream);
    ncclSend(kv_buf[cur].data(), count, dtype, next_rank, comm, comm_stream);
    cudaEventRecord(recv_done[nxt], comm_stream);

    launch_block_attention(q, kv_buf[cur], online_state, compute_stream);

    cudaStreamWaitEvent(compute_stream, recv_done[nxt], 0);
    std::swap(cur, nxt);
}

finalize_attention_output(online_state);
```

重点不是 API 名字，而是两个原则：

1. 当前块计算时，下一块尽量已经在接收。
2. 不要让“正在计算的数据”和“即将写入的数据”共用同一块 buffer。

### 坑 3：只盯显存，不看拓扑

很多人看到“上下文可近似线性扩展”就直接想上 Ring Attention，但忽略了互联条件。

实际吞吐受这几项共同决定：

| 因素 | 影响 |
|---|---|
| 设备内带宽（如 NVLink） | 决定环内点对点传输是否高效 |
| 跨机网络（如 IB） | 决定多机环是否稳定 |
| rank 映射方式 | 决定通信路径是否绕远 |
| block size | 决定计算与通信是否能重叠 |
| kernel 效率 | 决定 `F` 是否真的接近理论峰值 |

所以，Ring Attention 的判断标准不是“理论上能扩长上下文”，而是：

> 你的硬件和实现，是否真的能让计算和通信形成稳定流水线。

### 坑 4：只优化 attention，忽略整层系统瓶颈

在长上下文训练里，attention 往往是瓶颈，但不一定是唯一瓶颈。  
如果 attention 被 Ring 优化后，FFN、激活重算、参数并行同步、优化器状态通信反而成为主瓶颈，那么整层吞吐未必线性提升。

因此工程评估通常要同时看：

| 指标 | 为什么要看 |
|---|---|
| attention kernel 时间 | 判断 Ring 本身是否有效 |
| comm wait 时间 | 判断通信是否真被掩盖 |
| step 总时长 | 看整体训练是否变快 |
| 显存峰值 | 看长上下文是否真的能放下 |
| MFU / HFU | 看硬件利用率是否提升 |

---

## 替代方案与适用边界

Ring Attention 不是“比所有 attention 方案都更强”，它只是对某一类问题特别合适：  
**长上下文、精确全局 attention、多设备、高带宽互联。**

先看一个工程向对比表：

| 方案 | 通信模式 | 上下文扩展性 | 是否精确 | 依赖硬件 | 适合什么场景 |
|---|---|---|---|---|---|
| Ring Attention | 环形 P2P | 很强，近似随设备数扩展 | 是 | 高带宽低延迟互联更佳 | 超长上下文训练或推理 |
| Sequence Parallel | all-gather / reduce-scatter 等 | 中等 | 是 | 常规多卡可用 | 中长序列，工程实现较直接 |
| Local Window Attention | 很少或无 | 很强 | 否 | 硬件要求低 | 只关心局部依赖的任务 |
| FlashAttention | 主要是单设备内存与 IO 优化 | 不直接扩总上下文 | 是 | 单卡也受益 | 几乎所有标准 attention |
| Sparse Attention | 依赖稀疏模式 | 强 | 视实现而定 | 中等 | 已知稀疏结构的任务 |

可以用下面这张判断表做快速选型：

| 需求 | 更可能合适的方案 |
|---|---|
| 4K 到 8K、8K 到 16K 的普通扩展 | FlashAttention + 常规并行 |
| 必须保持精确全局 attention，且长度远超单卡容量 | Ring Attention |
| 明确知道远程依赖不重要 | Local Window / Sparse |
| 设备互联普通，部署成本要低 | Sequence Parallel |
| 单卡先榨干性能 | FlashAttention |

### 什么时候优先考虑 Ring Attention

以下条件越多满足，越值得评估：

1. 目标上下文已经长到单卡显存无法承受
2. 模型必须保留精确全局 attention
3. 集群互联较强，比如 NVLink 或高质量 InfiniBand
4. 序列并行维度较大，设备数足够多
5. 团队可以接受更复杂的通信与调度实现

### 什么时候不该优先上 Ring

以下情况通常不划算：

1. 上下文不够长，FlashAttention 已经能解决
2. 设备太少，环形流水带来的收益有限
3. 网络延迟较高，通信难以稳定隐藏
4. 团队没有能力维护复杂的分布式 attention 内核
5. 业务可以接受近似 attention 或局部窗口

### 和 Sequence Parallel 的关系

很多新手会把 Ring Attention 和 Sequence Parallel 当成二选一。  
更准确的说法是：

- **Sequence Parallel**：描述的是“沿序列维度切分到多设备”
- **Ring Attention**：描述的是“在这种切分上，attention 用什么通信执行策略”

也就是说，Ring Attention 通常运行在 sequence parallel 的拓扑上，它不是 mesh 概念的替代品，而是其中一种 attention 实现方式。

下面这个配置伪代码表达的就是这种关系：

```python
# pseudo config
mesh = DeviceMesh(("dp", "tp", "sp"), (1, 1, 64))

config = {
    "sequence_parallel": True,
    "ring_attention": True,
    "sp_size": 64,
    "block_size": 1024,
}

assert config["sequence_parallel"] is True
assert config["ring_attention"] is True
```

这里：

- `dp` 表示 data parallel
- `tp` 表示 tensor parallel
- `sp` 表示 sequence parallel

Ring Attention 不会替代这些并行维度，而是嵌在 attention 层的执行路径里。

---

## 参考资料

下面给出更适合从“原理 → 推导 → 工程实现”顺序阅读的资料表。

| 来源 | URL | 内容摘要 | 推荐阅读顺序 |
|---|---|---|---|
| Ring Attention with Blockwise Transformers for Near-Infinite Context | https://arxiv.org/abs/2310.01889 | 原论文，核心机制、通信隐藏思路、长上下文实验结论都以此为准 | 1 |
| Hugging Face Papers: Ring Attention with Blockwise Transformers for Near-Infinite Context | https://huggingface.co/papers/2310.01889 | 论文入口页，适合先快速确认主题、作者与社区讨论 | 2 |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | https://arxiv.org/abs/2205.14135 | 理解 blockwise attention、在线 softmax 合并的基础材料 | 3 |
| feifeibear/long-context-attention | https://github.com/feifeibear/long-context-attention | 长上下文 attention 的工程实现集合，适合对照分块计算与通信组织 | 4 |
| Selimonder/ring-attention | https://github.com/Selimonder/ring-attention | 一个偏实现导向的 Ring Attention 项目，可看配置和代码结构 | 5 |
| UvA DL Notebooks / Attention and Transformers 类材料 | https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html | 如果对 Q/K/V、softmax attention 还不熟，这类基础材料更适合先补底层概念 | 6 |

如果只读三份，建议顺序是：

1. 原论文，建立“固定 Q、轮转 KV、精确全局 attention”的主框架  
2. FlashAttention 论文，理解“为什么分块后仍能与全局 softmax 等价”  
3. 工程仓库，理解双缓冲、send/recv 顺序和实际配置方式
