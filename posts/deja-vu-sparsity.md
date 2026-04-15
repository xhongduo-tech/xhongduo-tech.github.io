## 核心结论

Deja Vu（Contextual Sparsity，上下文稀疏）是一类**推理时动态结构化稀疏**方法。动态，意思是它不会提前永久删除模型参数，而是在每次生成某个 token、经过某一层时，临时判断“这次真正值得算哪些部分”。结构化，意思是它跳过的不是零散标量，而是整个 attention head 或一组 MLP 神经元，这样才更容易映射到 GPU 友好的算子。

它解决的不是“模型参数太大”这个静态问题，而是“每个 token 都把整层全部算一遍，里面有很多计算其实对当前上下文没贡献”这个运行时问题。对新手可以这样理解：不是把模型整体减肥，而是让模型在答题时只调动当前题目会用到的那部分脑细胞。

密集推理和稀疏推理可以写成：

$$
a^{(l)}=\text{Attn}_l(x^{(l)}), \quad m^{(l)}=\text{MLP}_l(a^{(l)})
$$

$$
a^{(l)}=\text{Attn}_l(x^{(l)};S_A^{(l)}), \quad m^{(l)}=\text{MLP}_l(a^{(l)};S_M^{(l)})
$$

其中 $x^{(l)}$ 是第 $l$ 层输入，$S_A^{(l)}$ 是被预测为活跃的 attention heads，$S_M^{(l)}$ 是被预测为活跃的 MLP 神经元集合。

论文的关键结论是：如果活跃单元预测足够准，就能在较高结构化稀疏率下保持模型质量，同时显著减少推理延迟。直白说，结构化稀疏确实能快，但前提是两件事同时成立：**预测要准，执行也要真跳过计算**。

| 方法 | 是否动态 | 是否按输入变化 | 主要目标 | 工程难点 |
|---|---|---:|---|---|
| Dense 推理 | 否 | 否 | 保持原始精度 | 成本高 |
| 静态剪枝 | 否 | 否 | 永久删参数 | 容易伤精度 |
| Deja Vu | 是 | 是 | 只算当前活跃部分 | 预测器与稀疏执行 |

---

## 问题定义与边界

Deja Vu 要优化的是**大模型在线推理中的无效计算**。无效，不是结果完全没用，而是指对于当前 token 和当前上下文，很多 head 和神经元即使参与计算，最终贡献也很小，或者可以被少量关键单元近似替代。

在标准 Transformer 里，每个 token 经过每层时，attention 和 MLP 都默认全量执行：

$$
\text{Dense cost}=\text{all heads}+\text{all neurons}
$$

如果可以先预测出真正需要的子集，那么成本就变成：

$$
\text{Sparse cost}=\text{selected heads}+\text{selected neurons}
$$

这里的“selected”不是手工规则，而是由一个轻量预测器根据当前表示自动给出。

它最适合的场景是：

| 维度 | 适用 | 不适用 |
|---|---|---|
| 服务形态 | 在线聊天、代码补全、搜索问答 | 离线大批量评测 |
| 优化目标 | 单请求延迟、首 token/逐 token 延迟 | 纯吞吐最大化 |
| 模型规模 | 大模型，单层计算重 | 小模型，层内计算本就便宜 |
| 工程条件 | 能改 runtime、能接稀疏算子 | 只能直接用现成 dense 框架 |

一个真实判断标准是：如果你的系统主要痛点是“用户在等回复”，而不是“GPU 每秒总吞吐不够”，那么动态稀疏更值得考虑。

边界也要说清楚。

第一，它优化的是**推理**，不是训练。训练时激活分布更复杂，反向传播还要维护梯度路径，难度高得多。

第二，它不等于量化、蒸馏、MoE。量化是把每次计算变便宜；蒸馏是让小模型学大模型；MoE 是让多个专家里只激活少数几个；Deja Vu 是在已有 dense 模型内部，按输入动态挑出本次该算的 heads 和 neurons。

第三，它也不自动解决 KV cache 的所有问题。KV cache 是 attention 历史键值缓存，用来避免重复计算旧 token。Deja Vu 可以和 KV cache 配合，但 attention 稀疏以后，历史信息怎么读取、哪些头必须保留，都需要额外设计。

---

## 核心机制与推导

Deja Vu 的机制可以拆成四步：

1. 用原始 dense 模型跑样本，收集每层每个 token 的激活情况。
2. 根据激活强弱，标注哪些 heads 和 neurons 算“活跃”。
3. 训练轻量预测器，让它根据当前表示预测下一步活跃集合。
4. 推理时只执行被预测选中的结构化子集。

“激活”这个词第一次出现时可以白话理解为：某个内部单元这次有没有真正参与表达，贡献是否足够大。

### 玩具例子

设某层 MLP 的 5 个神经元预激活值为：

$$
z=[2.4,-1.1,-0.3,-0.7,0.0]
$$

若激活函数用 ReLU，ReLU 的意思是负数直接变成 0，正数保留：

$$
h=\phi(z)=[2.4,0,0,0,0]
$$

这表示只有第 1 个神经元真正活跃。活跃率是 $1/5$，稀疏度是：

$$
1-\frac{1}{5}=80\%
$$

如果输出权重向量是：

$$
v=[1,2,-1,0.5,3]
$$

那么输出为：

$$
y=h^\top v=2.4
$$

因为后 4 个神经元本来就是 0，即使你把它们完全跳过，结果也不变。这个玩具例子说明了一个核心事实：**并不是每次都需要全量神经元参与，很多 token 的内部表示天然是稀疏的**。

### 从 MLP 推到结构化稀疏

标准前馈层可写成：

$$
\text{MLP}(x)=W_2\phi(xW_1)
$$

其中 $\phi$ 是非线性激活函数，$W_1$ 把输入投影到更高维隐藏层，$W_2$ 再投回模型维度。若只保留活跃神经元集合 $S$，则可写成：

$$
\text{MLP}_S(x)=W_2[\phi(xW_1)]_{S}
$$

符号 $[\cdot]_S$ 的意思是只保留集合 $S$ 中的项，其他项不参与后续计算。工程上通常不是“先全算出来再置零”，而是直接只发起与 $S$ 对应的计算。

attention 也是同理。多头注意力里，不同 head 关注的模式不同，有的偏局部，有的偏长程，有的偏语法，有的偏对齐。对于某个具体 token，并不是所有 head 都同样重要。Deja Vu 就是在每层、每个 token 上，预测这一刻真正值得算的 head 子集。

### 为什么可以提前预测

如果每到一层都先停下来做一次同步预测，再决定算哪些单元，预测器本身就可能把节省下来的时间吃掉。论文的关键工程点是 **look-ahead 异步预测**。look-ahead 的白话意思是“提前看一步”。

原因来自残差结构。残差连接意味着层与层之间表示变化通常不是完全跳变，而是在已有表示上做相对平滑的修正。因此可以用当前层或更早层的信息，提前预测下一层哪些单元大概率会活跃。这样预测和主计算可以并行或交叠，减少等待。

可以把流程理解成：

| 步骤 | 输入 | 输出 | 作用 |
|---|---|---|---|
| dense 采样 | 原始模型与样本 | 激活统计 | 生成标签 |
| 标签构造 | 激活强度 | 活跃 head/neuron 集合 | 定义监督目标 |
| 预测器训练 | 隐状态表示 | 活跃概率或 Top-k | 学会“猜哪些要算” |
| 稀疏执行 | 预测结果 | 稀疏 attention / MLP | 真正减少推理计算 |

真实工程例子是在线聊天助手。用户输入“请解释一下为什么 Adam 会比 SGD 收敛更快”。系统逐 token 生成回答时，每一步都要过几十层 Transformer。如果当前 token 是技术解释场景，一部分头会更关注定义对齐，一部分神经元更容易被专业术语触发；如果 token 是标点或格式词，很多内部单元贡献会更小。Deja Vu 的目标就是让 runtime 识别这些差异，只调用当前真正活跃的那部分算子。

---

## 代码实现

实现可以拆成四个模块：`label_collection`、`predictor_training`、`sparse_runtime`、`benchmarking`。这里先给一个最小可运行的 Python 玩具实现，演示“根据激活选 Top-k 神经元，再验证输出近似”的基本思想。

```python
import math

def relu(xs):
    return [max(0.0, x) for x in xs]

def dense_mlp(hidden, out_weight):
    return sum(h * w for h, w in zip(hidden, out_weight))

def topk_indices(values, k):
    pairs = sorted(enumerate(values), key=lambda x: x[1], reverse=True)
    return sorted(i for i, _ in pairs[:k])

def sparse_mlp(hidden, out_weight, active_idx):
    active = set(active_idx)
    return sum(h * w for i, (h, w) in enumerate(zip(hidden, out_weight)) if i in active)

# 玩具例子
z = [2.4, -1.1, -0.3, -0.7, 0.0]
h = relu(z)
v = [1.0, 2.0, -1.0, 0.5, 3.0]

dense_y = dense_mlp(h, v)
active = topk_indices(h, k=1)
sparse_y = sparse_mlp(h, v, active)

assert h == [2.4, 0.0, 0.0, 0.0, 0.0]
assert active == [0]
assert math.isclose(dense_y, 2.4)
assert math.isclose(sparse_y, dense_y)

# 一个近似场景：保留前 2 个最大激活
z2 = [1.2, 0.9, 0.05, 0.01, 0.0]
h2 = relu(z2)
v2 = [0.8, -0.4, 2.0, 1.5, 3.0]

dense_y2 = dense_mlp(h2, v2)
active2 = topk_indices(h2, k=2)
sparse_y2 = sparse_mlp(h2, v2, active2)

assert active2 == [0, 1]
assert abs(dense_y2 - sparse_y2) < 0.2
print("dense:", dense_y, dense_y2)
print("sparse:", sparse_y, sparse_y2)
```

这个代码故意没有引入 GPU 或矩阵库，因为重点是说明逻辑：先定义活跃集合，再只在这个集合上做计算。真正工程实现会复杂很多，关键差异在于你必须避免“虽然逻辑上稀疏了，但底层还是跑了完整 dense matmul”。

一个常见的 runtime 伪代码如下：

```python
for layer in model.layers:
    active_heads = predictor_head(layer_input, layer_id)
    attn_out = sparse_attention(layer_input, active_heads, kv_cache)

    active_neurons = predictor_mlp(attn_out, layer_id)
    mlp_out = sparse_mlp(attn_out, active_neurons)

    layer_input = residual(layer_input, attn_out, mlp_out)
```

各模块职责可以概括为：

| 模块 | 输入 | 输出 | 性能风险 |
|---|---|---|---|
| `label_collection` | dense 隐状态与激活 | 活跃标签 | 标签阈值不合理 |
| `predictor_training` | 隐状态、标签 | 轻量预测器 | 预测器过重 |
| `sparse_runtime` | 预测结果、KV cache | 稀疏层输出 | 伪稀疏、kernel 开销大 |
| `benchmarking` | 请求流与基线模型 | 延迟/吞吐/质量数据 | 只看 FLOPs 不看端到端 |

其中最容易被低估的是 KV cache。attention 稀疏不是简单地少算几个头就完了，还要保证历史 token 的 K/V 读取策略和当前选择一致，否则会出现“这一步快了，但上下文信息取错了”的质量问题。

---

## 工程权衡与常见坑

Deja Vu 的总收益不能只看理论 FLOPs。端到端延迟满足的不是“算得少就一定快”，而是：

$$
\text{End-to-end latency} \neq \text{FLOPs only}
$$

更接近真实工程的估计是：

$$
\text{Net gain}=\text{saved compute}-\text{prediction overhead}-\text{kernel overhead}
$$

这里 `prediction overhead` 是预测器本身的成本，`kernel overhead` 是稀疏执行引入的额外调度、访存和启动开销。

常见坑如下：

| 坑 | 为什么错 | 更合理做法 |
|---|---|---|
| 把它当静态剪枝 | 动态稀疏依赖输入，上下文变了选择也会变 | 按 token、按层评估活跃分布 |
| 只看 FLOPs | GPU 真正瓶颈常是访存和调度 | 直接测端到端延迟 |
| predictor 过重 | 预测本身可能比省下的还贵 | 用轻量模型并做 look-ahead |
| 稀疏实现不结构化 | 零散 mask 难以高效映射 GPU | 按 head、按块、按 neuron group 做结构化 |
| KV cache 处理错误 | 可能破坏历史上下文读取 | 显式设计缓存对齐策略 |
| 各层统一稀疏率 | 层间激活模式不同 | 中间层更激进，边缘层更保守 |

理想收益和实际收益常常差很远：

| 指标 | 理想情况 | 实际情况 |
|---|---|---|
| 稀疏率提升 | 线性带来加速 | 受 kernel 和访存限制 |
| 预测器成本 | 可忽略 | 设计不好会显著吞掉收益 |
| attention 稀疏 | 直接减少计算 | 还要兼顾 KV cache 与并行方式 |
| MLP 稀疏 | 容易高比例跳过 | 若只置零不改算子，收益有限 |

还有一个容易被误解的点：稀疏率不是越高越好。边缘层常更敏感，因为输入层附近负责早期表征，输出层附近更直接影响 logits。中间层通常更容易容纳较高稀疏率。工程上更合理的是分层配置，而不是全模型统一一个 90% 之类的目标。

---

## 替代方案与适用边界

Deja Vu 不是唯一选择，也不是默认首选。它和常见方案的区别如下：

| 方法 | 是否动态 | 是否结构化 | 是否要额外预测器 | 对延迟的直接作用 | 部署复杂度 |
|---|---|---|---:|---|---|
| 量化 | 否 | 通常否 | 否 | 每次计算更便宜 | 低到中 |
| 静态剪枝 | 否 | 可结构化 | 否 | 永久减少参数与计算 | 中 |
| 蒸馏 | 否 | 否 | 否 | 用更小模型替代大模型 | 中到高 |
| MoE | 是 | 是 | 有路由器 | 每次只激活少数专家 | 高 |
| Token Pruning | 是 | 可结构化 | 常需要 | 减少序列上的计算 | 高 |
| Deja Vu | 是 | 是 | 是 | 减少层内单元计算 | 高 |

对新手可以用一句话区分：

- 量化：把每次计算变便宜。
- 静态剪枝：永久删掉一些部分。
- Deja Vu：每次运行时动态决定哪些部分值得算。

什么时候应该优先考虑 Deja Vu？

1. 你做的是在线推理。
2. 你关心的是单请求延迟。
3. 模型足够大，单层内部计算占比高。
4. 你能接受额外预测器和定制 runtime。

什么时候不该先上它？

1. 你只是做离线批处理或大 batch 吞吐。
2. 你用的是现成推理框架，无法改底层算子。
3. 你的模型本来就不大，稀疏调度成本可能盖过收益。
4. 你的基础优化还没做完，比如量化、FlashAttention、批处理调度都还没上。

一个实用决策顺序是：先确认是不是在线延迟问题，再确认 dense 实现是否已较成熟。如果基础实现还很粗糙，先做量化、注意力优化、KV cache 管理，通常比直接上动态稀疏更稳。不是所有模型都值得上动态稀疏，尤其不是所有团队都值得先承担它的实现复杂度。

---

## 参考资料

- 论文主页：<https://proceedings.mlr.press/v202/liu23am.html>
- 论文 PDF：<https://proceedings.mlr.press/v202/liu23am/liu23am.pdf>
- OpenReview：<https://openreview.net/forum?id=wIPIhHd00i>
- 官方代码仓库：<https://github.com/FMInference/DejaVu>
- README：<https://github.com/FMInference/DejaVu#readme>

如果要复现，优先读论文理解方法定义，再看仓库 README 中训练、评测和延迟 benchmark 的实现说明，最后回到论文核对稀疏标签、预测器设计和异步预测细节。
