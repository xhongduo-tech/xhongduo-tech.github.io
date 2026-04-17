## 核心结论

Transformer 处理长上下文的核心矛盾，不是“模型能不能容纳更多 token”，而是“标准全连接注意力的计算和存储代价会随序列长度平方增长”。注意力的本质是：每个 token 都要决定自己该参考哪些位置的信息。标准 self-attention 中，第 $i$ 个位置会和全部 $N$ 个位置做一次相关性打分，因此序列长度从 $N$ 增长到 $2N$ 时，注意力分数计算量和注意力矩阵面积都会接近变成原来的 4 倍。

现代长上下文模型的主流改造路线，并不是完全放弃注意力，而是把“所有点两两相连的完全图”改成“稀疏、分层、但保持信息可达的图”。最常见的办法有四类：滑动窗口注意力、局部 + 全局注意力、块稀疏注意力、记忆增强注意力。它们的共同目标，是把复杂度从标准 attention 的

$$
O(N^2 d)
$$

压到更接近

$$
O(Nwd + Ngd)
$$

或分段场景下的近线性复杂度。其中 $w$ 是局部窗口宽度，$g$ 是少量全局槽或记忆槽数量，$d$ 是隐藏维度。

对初学者最重要的直觉是：长上下文不等于“每个 token 都逐字看完全部历史”。工程上更常见的真实做法是“近处看细，远处看摘要，关键位置开特权通道”。这就是为什么很多模型可以宣称支持 128K、200K 甚至更长窗口，但真正决定效果的，并不是窗口数字本身，而是稀疏连接方式、位置编码扩展、KV cache 优化、训练时是否见过足够长的依赖，以及远距离信息是否仍然保持连通。

截至 2026 年 3 月 8 日，公开资料能确认：OpenAI 文档仍将 `GPT-4 Turbo` 标为 128K 上下文；Anthropic 官方文档显示 Claude 主流模型长期以 200K 为基础，并在部分 API 模式中提供 1M 上下文；Meta 对 Llama 3.1 的官方介绍则明确给出 128K 上下文长度。要注意，闭源模型内部到底采用了哪种稀疏图、是否使用记忆槽、如何设计跨段通信，通常没有完整公开，因此很多结论只能基于论文、文档和工程惯例做保守推断，不能当作官方架构事实。

---

## 问题定义与边界

问题定义很直接：给定长度为 $N$ 的输入序列，标准 self-attention 会构造一个 $N \times N$ 的打分矩阵。矩阵中第 $i$ 行第 $j$ 列表示“第 $i$ 个 token 对第 $j$ 个 token 的关注强度”。因此其主要代价可以写成：

$$
\text{Time} \approx O(N^2 d), \qquad \text{Memory} \approx O(N^2)
$$

如果只按注意力权重本身粗略估算，单层、单头、`float32` 情况下的矩阵存储约为：

$$
M \approx 4N^2 \text{ bytes}
$$

如果有 $h$ 个头，则变成：

$$
M \approx 4N^2 h
$$

这里的 4 表示一个 `float32` 占 4 字节。这个公式只是下界估算。真实实现里还要叠加 Q、K、V 张量、softmax 临时 buffer、残差激活、KV cache、训练阶段的梯度与优化器状态，因此真实显存占用通常明显更高。

把问题代入数字会更直观。假设序列长度是 1,024：

| 方案 | 每个 token 可见位置数 | 总点积数量级 |
|------|----------------------|--------------|
| 全注意力 | 1,024 | $1{,}024^2 = 1{,}048{,}576$ |
| 滑窗注意力，$w=128$ | 最多 128 | $1{,}024 \times 128 = 131{,}072$ |
| 滑窗 + 4 个全局槽 | 最多 132 | $1{,}024 \times 132 = 135{,}168$ |

这个表的意义不在于“精确到每一次乘加”，而在于给出数量级直觉：如果窗口宽度固定，输入长度翻十倍，成本大致也只翻十倍，而不是一百倍。

再看一个更接近实际部署的估算。假设某层有 32 个注意力头，序列长度 $N=16{,}384$。仅注意力权重矩阵在 `float16` 下的理论大小就已经接近：

$$
16{,}384^2 \times 32 \times 2 \approx 16 \text{ GiB}
$$

这还没有算 Q、K、V、激活、KV cache 和其他层。也就是说，长上下文问题首先是一个硬件与复杂度问题，而不是纯粹的“模型参数够不够大”。

边界也必须说清楚。长上下文方法解决的是“让模型在更长输入上可运行、可训练，并尽量不明显退化”，不是保证“模型一定能稳定理解全文中所有远距离依赖”。长上下文失败通常落在三类问题上：

| 失败类型 | 本质原因 | 典型表现 |
|----------|----------|----------|
| 算不动 | $O(N^2)$ 成本过高 | OOM、吞吐暴跌、延迟不可接受 |
| 连不通 | 稀疏图断裂或路径过长 | 很远的信息无法传回当前 token |
| 记不住 | 摘要压缩过度或训练不足 | 中间事实丢失、摘要失真、`lost in the middle` |

“lost in the middle”可以理解为：模型虽然接受了很长输入，但对中间区域的信息利用率偏低，更容易引用开头或结尾。窗口再大，如果注意力分布过于稀疏、位置编码外推失真、训练时几乎没见过这么长的有效依赖，模型依然可能“形式上读到了，实际上没用好”。

因此，讨论长上下文时最好分清三个层面：

| 层面 | 真正的问题 |
|------|------------|
| 可运行性 | 能不能在硬件上把输入塞进去并跑完 |
| 可达性 | 任意两段信息之间有没有足够短的传播路径 |
| 可用性 | 模型是否真的学会在任务中利用这些远距离信息 |

很多误解来自把这三件事混为一谈。

---

## 核心机制与推导

### 1. 滑动窗口注意力

滑动窗口注意力的定义最简单：第 $i$ 个 token 只允许看它附近固定范围内的 token。若窗口宽度是 $w$，可见边可以写成：

$$
A_{ij} \neq 0 \iff |i-j| \le w
$$

如果是因果语言模型，还要再加一个约束：

$$
j \le i
$$

表示当前位置不能看未来。这样做以后，注意力图从“完全图”变成“带状图”，复杂度近似变为：

$$
O(Nwd)
$$

白话解释就是：每个 token 不再和所有位置比较，只和自己附近的邻居比较。

它的优点非常明确：

| 优点 | 解释 |
|------|------|
| 实现直接 | 本质上只是给 attention 增加规则化 mask |
| 硬件友好 | 连边模式整齐，便于 kernel 优化 |
| 局部建模强 | 对相邻词、短语、句内关系很有效 |

缺点也同样直接。假设只有局部边，两个相距很远的 token 之间必须通过多层“接力”传播信息。若单层最多跨越 $w$ 个位置，经过 $L$ 层后，理论可影响范围大致才扩展到 $L \cdot w$。因此，当距离远大于 $L \cdot w$ 时，远距离依赖很难稳定学习。

一个简单例子：

| 场景 | 纯滑窗效果 |
|------|------------|
| 代码补全，看上一两百行变量定义 | 往往足够 |
| 小说长篇伏笔回收，信息相隔数万 token | 容易失效 |
| 合同条款中前后页定义与例外条款相互引用 | 纯滑窗通常不稳 |

所以滑动窗口更像“把问题规模压下去的第一步”，而不是完整答案。

### 2. 局部 + 全局注意力

为了避免图断裂，工程上常加入少量全局 token。全局 token 可以理解为“拥有广播权限的特殊位置”。普通 token 看本地窗口；全局 token 可以看全局，或者所有 token 额外都允许看这些全局槽。复杂度变为：

$$
O(Nwd + Ngd)
$$

其中 $g \ll N$，通常是常数级或几十级。

关键点不在于“重新让所有位置全连接”，而在于用极少数中转站，显著缩短远距离通信路径。只用滑窗时，最远两点之间的最短路径长度大致与距离成正比；加入全局节点后，很多通信路径会变成：

$$
\text{局部 token} \rightarrow \text{全局槽} \rightarrow \text{远端 token}
$$

路径长度从“随距离增长”变成“近似常数”。

什么位置适合做全局点，取决于任务：

| 全局点类型 | 适用场景 |
|------------|----------|
| 文档标题、目录、章节标题 | 长文档问答、论文阅读 |
| 段落起始 token | 结构化文本 |
| 特殊分隔符、系统提示 | 对话与 agent 流程 |
| 实体标记、时间点、编号字段 | 合同、审计、日志分析 |

对新手来说，可以把它理解成“在超长文本里放少量路由器”。路由器本身数量很少，但它让本来互相很远的内容能更快连上。

### 3. 块稀疏注意力

块稀疏注意力不是按单个 token 连边，而是按 token block 连边。可以把连续的 64、128 或 256 个 token 看成一个 block，再规定每个 block 只与相邻块、固定跳跃块、摘要块或少量全局块通信。

这种做法的优势主要有两点：

1. 更适合底层 kernel 优化。连续块在内存布局上更规则，访存和并行处理更容易做。
2. 更容易设计“规则化稀疏图”。例如“本地块 + 上一块 + 下一块 + 每 8 块跳一次 + 1 个摘要块”。

从图结构角度看，块稀疏注意力是在 token 级稀疏和工程可实现性之间做折中。理论上，完全随机的稀疏连接也可能连通，但那样的图通常不利于 GPU 实现；而块状模式既保留了稀疏性，也更适合硬件。

一个典型的块连边策略如下：

| 当前 block | 允许访问的 block |
|------------|------------------|
| 第 $b$ 块 | $b-1, b, b+1$ |
| 第 $b$ 块 | 每隔 $k$ 个的跳跃块 |
| 第 $b$ 块 | 1 个全局摘要块 |
| 第 $b$ 块 | 文档起始块或章节标题块 |

这类设计常见于“超长文本 + 高吞吐需求”的场景。很多长上下文模型并不是“单一滑窗”，而是“块局部 + 跨块稀疏 + 少量全局”的混合图。

### 4. 记忆增强注意力

记忆增强注意力的思路与前面不同。它不是“把很远的原始 token 也直接连进来”，而是“把旧信息压缩成少量可读写的记忆槽”。这些记忆槽可以理解为“上一段上下文留下来的摘要向量”。

假设每段长度是 $s$，记忆槽数是 $m$。处理某一段时，模型只需要看到：

- 当前段的 $s$ 个 token
- 来自历史的 $m$ 个 memory token

这样单段复杂度近似变为：

$$
O(s^2 d + smd)
$$

如果 segment 大小固定，总长度 $N$ 通过分段推进，那么总体成本就更接近线性增长。

可以把它理解成一种“分段递归”：

1. 先读当前段。
2. 把这一段压缩成有限个 memory slots。
3. 下一段读取这些 memory slots。
4. 用新的段内容更新 memory。
5. 重复整个过程。

它的核心收益是：历史长度不断增长，但模型每一步真正面对的“历史表示大小”保持近似固定。

一个文档阅读的玩具例子如下。假设你要读一本 100 页的技术手册，每页 500 个 token：

| 方法 | 模型看到什么 |
|------|--------------|
| 全注意力 | 100 页原文一次性全部摊开 |
| 记忆增强 | 当前 1 页原文 + 8 个历史摘要槽 |
| RAG | 当前问题相关的几页 + 检索证据 |

记忆增强方法牺牲了“全部历史原文逐字可见”，换来了可控的显存和延迟。它特别适合“主题延续、状态积累、长过程推理”，但对“逐字核对原文证据”的任务要更谨慎，因为压缩就意味着信息可能损失。

### 5. 位置编码也必须同步改

只改注意力图还不够。模型还必须知道“谁在前、谁在后、距离多远”。这件事由位置编码负责。RoPE 可以理解为一种把相对位置信息编码到向量旋转关系中的方法。问题在于，如果训练时主要见过 4K、8K 或 32K 长度，推理时直接外推到 128K 甚至更长，位置关系可能失真。

因此，长上下文改造通常至少包含四个层面：

| 层面 | 作用 |
|------|------|
| 稀疏注意力图 | 降低计算和显存成本 |
| 位置编码扩展 | 让远距离相对位置仍可区分 |
| KV cache / FlashAttention | 让推理在硬件上真正跑得动 |
| 长上下文训练数据 | 让模型学会利用这些远距离连接 |

这四项缺一不可。只做其中一项，往往只能得到“输入更长但效果不稳”的结果。

例如，一个模型理论上支持 128K 输入，但如果：

- 位置编码没有做稳定外推，
- 训练时几乎没见过长依赖，
- 推理时 KV cache 过大导致吞吐崩掉，

那么这个“128K”更多只是接口层能力，而不是可靠的任务能力。

---

## 代码实现

下面先给一个最小可运行的滑动窗口 mask 实现。它不依赖深度学习框架，只用 Python 标准库演示“哪些位置允许互相注意”。

```python
from typing import List

def sliding_window_mask(seq_len: int, window: int, causal: bool = True) -> List[List[int]]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if window < 0:
        raise ValueError("window must be non-negative")

    mask: List[List[int]] = []
    for q_idx in range(seq_len):
        row: List[int] = []
        for kv_idx in range(seq_len):
            if causal:
                visible = (kv_idx <= q_idx) and ((q_idx - kv_idx) <= window)
            else:
                visible = abs(q_idx - kv_idx) <= window
            row.append(1 if visible else 0)
        mask.append(row)
    return mask


def count_visible(mask: List[List[int]]) -> int:
    return sum(sum(row) for row in mask)


m = sliding_window_mask(seq_len=6, window=2, causal=True)

# 第 0 个 token 只能看自己
assert m[0] == [1, 0, 0, 0, 0, 0]

# 第 3 个 token 可以看 [1, 2, 3]
assert m[3] == [0, 1, 1, 1, 0, 0]

# 因果约束下不能看未来
assert m[3][4] == 0

# 第 5 个 token 只能回看 [3, 4, 5]
assert m[5] == [0, 0, 0, 1, 1, 1]

full_causal = sum(i + 1 for i in range(6))  # 因果全注意力的可见边数
swa_causal = count_visible(m)

assert full_causal == 21
assert swa_causal == 15
assert swa_causal < full_causal

print("sliding window mask ok")
```

这个例子可以配一张“可见性示意表”来理解。`1` 表示可见，`0` 表示不可见：

| query \\ key | 0 | 1 | 2 | 3 | 4 | 5 |
|--------------|---|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| 2 | 1 | 1 | 1 | 0 | 0 | 0 |
| 3 | 0 | 1 | 1 | 1 | 0 | 0 |
| 4 | 0 | 0 | 1 | 1 | 1 | 0 |
| 5 | 0 | 0 | 0 | 1 | 1 | 1 |

这就是“带状注意力图”的最基本形态。

如果把它翻译成深度学习框架中的 attention 流程，步骤通常是：

1. 计算 $QK^\top$。
2. 对不可见位置加上一个极小值掩码。
3. 做 softmax。
4. 再乘上 $V$。

公式写成：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top + \text{mask}}{\sqrt{d_k}}\right)V
$$

这里 `mask` 对可见位置加 0，对不可见位置加负无穷。这样 softmax 后，不可见位置的权重就会接近 0。

再看一个“局部 + 全局”的可运行示意。假设第 0 个和第 5 个 token 是全局位置：

```python
from typing import List, Set

def local_global_mask(seq_len: int, window: int, global_ids: Set[int], causal: bool = True) -> List[List[int]]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if window < 0:
        raise ValueError("window must be non-negative")

    mask: List[List[int]] = []
    for q_idx in range(seq_len):
        row: List[int] = []
        for kv_idx in range(seq_len):
            if causal:
                local_ok = (kv_idx <= q_idx) and ((q_idx - kv_idx) <= window)
            else:
                local_ok = abs(q_idx - kv_idx) <= window

            global_ok = kv_idx in global_ids
            row.append(1 if (local_ok or global_ok) else 0)
        mask.append(row)
    return mask


m2 = local_global_mask(seq_len=8, window=2, global_ids={0, 5}, causal=True)

# 第 7 个 token 本地可见 [5, 6, 7]，并且还能看到全局位置 0
assert m2[7][0] == 1
assert m2[7][5] == 1
assert m2[7][6] == 1
assert m2[7][7] == 1
assert m2[7][1] == 0

print("local + global mask ok")
```

这个例子揭示了一个关键变化：即使第 7 个 token 与第 0 个 token 相距很远，只要第 0 个位置是全局槽，它们仍然可以直接建立联系。对长文档来说，这类全局点通常不是随便挑的，而是目录、标题、条款编号、段落摘要、系统提示等结构性位置。

下面再给一个块稀疏注意力的最小示意。这个实现不做真正的矩阵乘法，只生成“哪个 block 可以看哪个 block”的结构。

```python
from typing import Dict, List, Set

def block_sparse_pattern(num_blocks: int) -> Dict[int, Set[int]]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")

    graph: Dict[int, Set[int]] = {}
    for b in range(num_blocks):
        visible = {b}  # 自身
        if b - 1 >= 0:
            visible.add(b - 1)
        if b + 1 < num_blocks:
            visible.add(b + 1)
        if b % 2 == 0:
            visible.add(0)  # 偶数块额外看第 0 块，模拟全局摘要块
        graph[b] = visible
    return graph


pattern = block_sparse_pattern(6)

assert pattern[0] == {0, 1}
assert pattern[2] == {0, 1, 2, 3}
assert pattern[5] == {4, 5}

print("block sparse pattern ok")
```

如果 block size 设成 128，那么“第 2 块看第 0、1、2、3 块”就等价于“第 256 到 383 号 token 可以访问部分局部和摘要区域”。这类模式在工程实现中比“随机 token 级稀疏”更容易优化。

真实工程里更关键的是记忆槽。下面给一个段级记忆更新的简化版。它不是论文复现，只是帮助理解“上一段写 memory，下一段读 memory”的接口。

```python
from typing import List

def update_memory(memory_slots: List[float], segment_states: List[float], m: int) -> List[float]:
    if m <= 0:
        raise ValueError("m must be positive")
    if len(memory_slots) != m:
        raise ValueError("len(memory_slots) must equal m")

    tail = segment_states[-m:]
    if len(tail) < m:
        tail = [0.0] * (m - len(tail)) + tail
    return tail


memory = [0.0, 0.0, 0.0, 0.0]
seg1 = [0.2, 0.3, 0.5, 0.7, 0.9]
seg2 = [1.0, 1.2, 1.4, 1.6]

memory = update_memory(memory, seg1, 4)
assert memory == [0.3, 0.5, 0.7, 0.9]

memory = update_memory(memory, seg2, 4)
assert memory == [1.0, 1.2, 1.4, 1.6]

print("memory update ok")
```

这个实现非常粗糙，但它抓住了核心：历史不再以“全部原 token”存在，而是被压缩进固定大小的 memory slots。论文中的真实做法通常会更复杂，例如用可学习读写、跨层汇总、层次记忆或 segment recurrence，而不是简单地保留最后几个状态。

把这些机制放到企业级长文档问答中，流程通常更像这样：

1. 先把文档切成若干 segment。
2. 每个 segment 内做局部注意力，保证页内推理准确。
3. 把目录、标题、条款编号、实体表等关键节点设为全局位置。
4. 用记忆槽保留前文摘要、变量状态或证据索引。
5. 在推理端使用 FlashAttention、分页 KV cache 或分段缓存来控制吞吐。

目标不是“让最后一层重新逐字读取 300 页原文”，而是“让关键事实能沿着局部边、全局边和记忆边，被传到真正需要生成答案的位置”。

---

## 工程权衡与常见坑

先看大方向：长上下文不是免费能力，而是训练、推理、评测三端一起付成本。

| 上下文长度 | 典型工程状态 | 常见要求 |
|------------|--------------|----------|
| 8K-32K | 比较常规 | 单卡可部署，优化压力中等 |
| 64K-128K | 已进入重优化区 | FlashAttention、KV cache、分段推理几乎必需 |
| 200K+ | 高成本区 | 显存、带宽、调度、评测都明显变难 |
| 1M 级 | 特殊模式 | 常需分层处理、缓存、检索或多阶段推理 |

第一个坑是“窗口很长，但图不连通”。例如你宣称支持 200K token，但每层只是 `window=64` 的纯滑窗，没有全局槽、没有跨段摘要、也没有跳跃边，那么最远距离的信息虽然理论上能逐层传递，实际却可能因为路径过长而在训练中学不出来。可以把它想成一条很长但很窄的管道：信息不是不能走，而是太慢、太弱、太容易在途中衰减。

第二个坑是“窗口数字不等于理解能力”。模型能接受 128K 输入，只说明 tokenizer、位置编码、kernel 和显存路径允许这件事，不说明它能稳定找出第 90K token 附近的关键事实。评测时至少要分开看：

| 指标 | 测什么 |
|------|--------|
| Needle-in-a-Haystack | 能否从极长文本中找到单个目标事实 |
| Multi-hop QA | 能否跨多个远距离位置做联合推理 |
| Topic recall | 能否记住早期主题并在后文调用 |
| Lost-in-the-middle | 中间段信息是否被系统性忽视 |

第三个坑是“内存瓶颈常常比算力更先到”。推理时不仅有 attention 计算，还有 KV cache。KV cache 可以理解为“为了后续生成而保存的历史键值表示”。上下文越长、层数越多、头数越大，cache 就越重。很多长上下文部署真正卡住的不是 FLOPs，而是显存容量、显存带宽和 cache 管理策略。

第四个坑是“摘要式记忆会失真”。记忆槽本质上就是压缩。只要压缩，就可能丢细节。对代码仓库问答、法律条款比对、实验日志审计、金融报表核验这类任务，丢一个否定词、一个版本号、一个时间戳，结果就可能错。因此，记忆增强更适合“主题延续”和“状态概括”，但对“逐字证据检索”不能盲信。

第五个坑是“训练分布不匹配”。如果模型主要在短上下文上训练，只靠推理时把位置编码硬外推到超长长度，效果通常不稳定。长上下文能力必须在训练或继续预训练中显式学习，否则往往只是“支持输入更长”，不等于“学会利用更长”。

第六个坑是“评测设计过于单一”。很多团队只做“把一个针放进草堆里再让模型找出来”这种单点测试，但真实业务常常要求同时满足三件事：

| 任务要求 | 仅靠单针测试能否覆盖 |
|----------|----------------------|
| 找到证据 | 部分能覆盖 |
| 组合多个证据 | 覆盖不足 |
| 严格保持原文细节 | 覆盖不足 |

如果你的业务是合同审阅、代码分析、长链路 agent 规划，那么单一的 needle benchmark 往往远远不够。

---

## 替代方案与适用边界

如果任务的关键信息密度很低，或者总上下文已经远超模型稳定工作区，继续盲目扩窗口往往不是最优路线。更实际的替代方案通常是检索增强生成，也就是 RAG。RAG 可以理解为“先把相关段落找出来，再交给模型推理”。它把问题从“让模型读完全部内容”改成“让模型只读最相关内容”。

下面给出一个选择表：

| 策略 | 适用边界 | 优点 | 局限 |
|------|----------|------|------|
| 纯滑动窗口 | 局部依赖为主，如代码补全、连续对话 | 实现简单，吞吐稳定 | 远距离依赖弱 |
| 局部 + 全局 | 文档结构明显，如标题、目录、关键字段 | 连通性更好 | 需设计全局点 |
| 块稀疏注意力 | 超长文本、硬件优化优先 | 更利于 kernel 实现 | 模式设计复杂 |
| 记忆增强注意力 | 长文档摘要、跨段主题保持 | 内存可控，支持递归 | 摘要可能失真 |
| RAG + chunking | 海量语料、稀疏相关事实检索 | 成本可控，证据路径清晰 | 依赖检索质量 |
| 分层摘要 + 二次推理 | 报告、会议纪要、日志分析 | 易工程化 | 多轮压缩易丢细节 |

真实工程里，常见最优解不是单选，而是组合。典型组合流程通常是：

1. 第一层用 RAG 取回 8 到 16 个相关 chunk。
2. 第二层在 chunk 内用局部或局部 + 全局注意力做精读。
3. 第三层用记忆槽、摘要缓存或外部状态表保留跨轮关键信息。

这通常比“把 50 万 token 原文一次塞进去”更可控，也更便宜。

再看几个模型例子，但必须区分“公开确认的信息”和“架构推断”。截至 2026 年 3 月 8 日，公开资料可确认的信息如下：

| 模型 | 公开确认的上下文信息 | 能确定的结论 | 不能过度下结论的部分 |
|------|----------------------|--------------|----------------------|
| GPT-4 Turbo | OpenAI 官方模型页仍写明 128K | OpenAI 已将长上下文作为正式产品能力提供 | 具体稀疏图和记忆机制未完整公开 |
| Claude 系列 | Anthropic 官方模型文档显示 200K 为基础，部分 API 模式支持 1M | Anthropic 明确把超长输入做成产品能力 | 内部层次注意力与记忆设计并非全部官方披露 |
| Llama 3.1 | Meta 官方发布文明确认 128K | 开源路线也在系统支持长上下文 | 具体训练配方与稀疏机制细节公开有限 |

因此，对初级工程师最实用的判断标准不是“某模型窗口写多大”，而是：

1. 你的任务是局部依赖还是全局依赖？
2. 关键事实是稠密分布还是极稀疏分布？
3. 错一个细节的代价高不高？
4. 你能否接受检索、分段和状态缓存带来的系统复杂度？
5. 你更缺 GPU 显存，还是更缺可靠证据链？

如果答案是“信息稀疏、证据敏感、长度极长”，RAG 往往比纯长上下文更稳。
如果答案是“信息连续、需要保持叙事和篇章状态”，局部 + 全局 + 记忆增强通常更合适。
如果答案是“必须既保留原文证据，又跨段推理”，最现实的方案往往是“检索 + 长上下文精读 + 外部状态”的混合系统。

---

## 参考资料

- OpenAI, “GPT-4 Turbo”  
  https://platform.openai.com/docs/models/gpt-4-turbo
- OpenAI, “New models and developer products announced at DevDay”  
  https://openai.com/index/new-models-and-developer-products-announced-at-devday/
- Anthropic Docs, “Models overview”  
  https://docs.anthropic.com/en/docs/about-claude/models/overview
- Anthropic Docs, “Context windows”  
  https://docs.claude.com/en/docs/build-with-claude/context-windows
- Anthropic Help Center, “How large is the Claude API’s context window?”  
  https://support.claude.com/en/articles/8606395-how-large-is-the-anthropic-api-s-context-window
- Meta, “Introducing Llama 3.1: Our most capable models to date”  
  https://about.fb.com/news/2024/07/introducing-llama-3-1-our-most-capable-models-to-date/
- ACL Anthology, “HMT: Hierarchical Memory Transformer for Efficient Long Context Language Processing”  
  https://aclanthology.org/2025.naacl-long.410/
- THUDM, “LongBench v2”  
  https://github.com/THUDM/LongBench
- OpenBMB, “InfiniteBench”  
  https://github.com/OpenBMB/InfiniteBench
- Michael Brenddoerfer, “Sliding Window Attention”  
  https://mbrenndoerfer.com/writing/sliding-window-attention/
