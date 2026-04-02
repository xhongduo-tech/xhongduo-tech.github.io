## 核心结论

Megatron-LM 的张量并行（Tensor Parallelism，简称 TP，白话说就是“把同一层里的大矩阵拆到多张 GPU 上一起算”）不是随便把参数均分，而是严格按线性层的数学结构切分。

在标准 Transformer 层里，它的核心做法是：

| 模块 | 切分方式 | 每张卡拿到什么 | 是否需要同步 |
| --- | --- | --- | --- |
| Q/K/V 投影 | 按列切分 | $1/N$ 的输出通道，也可理解为 $1/N$ 的注意力头 | 这一层投影后先不需要 |
| Attention 输出投影 $W_o$ | 按行切分 | $1/N$ 的输入通道 | 需要一次 All-Reduce |
| FFN 第一层 $W_1$ | 按列切分 | $1/N$ 的中间神经元 | 不需要立即同步 |
| FFN 第二层 $W_2$ | 按行切分 | $1/N$ 的输入通道 | 需要一次 All-Reduce |

这里的 All-Reduce，白话说就是“每张卡先算出自己那一份结果，再做一次跨卡求和，并把最终和发回所有卡”。Megatron-LM 之所以能保持和单卡训练数学等价，关键就在于列切分和行切分交替出现：列切分保证非线性前的值已经完整，行切分再把部分和汇总回来。

因此，一个标准 Transformer block 在前向里通常有两次核心同步：

1. attention 输出投影后一次
2. FFN 第二层后一次

反向也各有对应同步，所以常见说法是“每层 2 次前向同步，2 次反向同步”。

通信量的近似写法是：

$$
\text{Comm}_{\text{forward, per layer}} \approx 2 \times B \times S \times H \times \frac{N-1}{N}
$$

其中 $B$ 是 batch size，$S$ 是序列长度，$H$ 是 hidden size，$N$ 是 TP size。若把前向和反向都算上，则常用近似是：

$$
\text{Comm}_{\text{fwd+bwd, per layer}} \approx 4 \times B \times S \times H \times \frac{N-1}{N}
$$

这说明 TP 的代价不是“参数切开了就免费”，而是把通信压力直接绑定到激活张量 $B \times S \times H$ 上。结论很直接：TP 适合放在 NVLink/NVSwitch 这种节点内高速互联里，不适合随意跨慢链路扩大。

---

## 问题定义与边界

问题本质是：单张 GPU 放不下某一层的参数、梯度或激活时，怎么在不改变模型数学定义的前提下，把这一层拆到多张卡上训练。

这里要先区分三件事：

| 概念 | 白话解释 | 解决什么问题 |
| --- | --- | --- |
| 数据并行 DP | 每张卡放完整模型，只分样本 | 吞吐扩展 |
| 张量并行 TP | 同一层拆到多张卡 | 单层太大放不下 |
| 流水线并行 PP | 不同层放到不同卡 | 整体模型太深放不下 |

Megatron-LM 的 TP 只解决“层内过大”问题。它不直接解决所有显存问题，也不天然保证高吞吐。因为一旦切层，就要引入频繁同步。

对初学者，可以先用一个玩具例子理解。

玩具例子：假设 hidden size 是 8，attention 头数是 4，TP=2。那就可以让两张卡各负责 2 个头。输入 $X \in \mathbb{R}^{B \times S \times 8}$ 会被广播到两张卡，每张卡各自持有一半 Q/K/V 投影矩阵，于是：

- GPU0 算头 0、1 的 Q/K/V
- GPU1 算头 2、3 的 Q/K/V

每张卡上的注意力头都能独立完成，因为单个头之间本来就互不依赖。真正需要合并的是这些头拼接后的输出再经过 $W_o$ 时，因为 $W_o$ 的输入来自所有头，但每张卡只持有其中一部分输入通道，所以这里必须做一次 All-Reduce。

边界也要说清楚：

1. TP 主要适用于 Transformer 这类大量线性层、隐藏维很大的模型。
2. TP 的收益建立在高带宽互联上，通常建议限制在单节点 NVLink 域内。
3. 如果模型本身不大，只是序列特别长，那么 TP 可能不是第一选择，Context Parallel 或 Sequence Parallel 往往更合适。
4. TP 不会消除激活显存，尤其 LayerNorm、Dropout 这些非张量并行层，常常还需要 SP 配合。

---

## 核心机制与推导

先看线性层。设输入为 $X \in \mathbb{R}^{m \times k}$，权重为 $W \in \mathbb{R}^{k \times n}$，输出为：

$$
Y = XW
$$

### 1. 为什么 Q/K/V 和 FFN 第一层适合按列切分

把 $W$ 按输出维切成 $N$ 份：

$$
W = [W_1, W_2, \dots, W_N]
$$

则：

$$
Y = [XW_1, XW_2, \dots, XW_N]
$$

这说明每张卡只要拿完整输入 $X$ 和自己那一块权重 $W_i$，就能独立算出一部分输出。因为输出本来就是按列拼接的，所以不需要立刻求和。

这就是 ColumnParallelLinear。白话说，它把“输出通道”拆了。Q/K/V 投影和 FFN 第一层都符合这个模式。

注意这里有一个关键约束：如果列切分后立刻接非线性，比如 GELU、SiLU，是安全的，因为每个输出元素已经是完整值，不是“半成品”。

### 2. 为什么 O 投影和 FFN 第二层适合按行切分

再把权重按输入维切分：

$$
W =
\begin{bmatrix}
W_1 \\
W_2 \\
\vdots \\
W_N
\end{bmatrix},
\quad
X = [X_1, X_2, \dots, X_N]
$$

则：

$$
Y = \sum_{i=1}^{N} X_i W_i
$$

每张卡都只能算出一个部分和 $Y_i = X_i W_i$，最后要把所有部分和加起来，才能得到完整输出。

这就是 RowParallelLinear。白话说，它把“输入通道”拆了，所以最后必须求和。

### 3. Transformer 里为什么正好是“列后接行”

FFN 的标准形式是：

$$
\text{FFN}(x) = \phi(xW_1 + b_1)W_2 + b_2
$$

其中 $\phi$ 是激活函数。若第一层用列切分，则每张卡拿到的是完整的局部神经元输出，可以本地做激活；第二层再行切分，把部分和汇总。这样只需要一次同步。

如果反过来，第一层先行切分，那么每张卡拿到的是部分和，激活函数前就必须汇总，否则：

$$
\phi(a_1 + a_2) \neq \phi(a_1) + \phi(a_2)
$$

这就是 Megatron 设计的关键：不是“切分就行”，而是必须让切分顺序和非线性位置匹配。

### 4. 每层为什么是两次同步

一个标准 block 可抽象成：

1. QKV 列切分
2. 本地 attention 计算
3. $W_o$ 行切分并 All-Reduce
4. FFN 第一层列切分
5. 激活函数本地执行
6. FFN 第二层行切分并 All-Reduce

所以前向里有两次核心同步。反向传播要把梯度按相反方向传回去，也对应两次同步。

### 5. 通信量怎么估

每次 attention 输出投影或 FFN 第二层汇总时，都要同步一个形状近似为 $(B, S, H)$ 的张量。若忽略常数细节，一次 All-Reduce 的有效交换量常写成：

$$
B \times S \times H \times \frac{N-1}{N}
$$

前向每层两次，则：

$$
\text{Comm}_{\text{forward, per layer}}
\approx
2 \times B \times S \times H \times \frac{N-1}{N}
$$

前后向都算：

$$
\text{Comm}_{\text{fwd+bwd, per layer}}
\approx
4 \times B \times S \times H \times \frac{N-1}{N}
$$

再看一个数值例子。设：

- $B=8$
- $S=512$
- $H=4096$
- $N=4$

则前向每层通信元素数近似为：

$$
2 \times 8 \times 512 \times 4096 \times \frac{3}{4}
= 25{,}165{,}824
$$

若使用 FP16，每个元素 2 字节，则约为 48 MB。若前后向都算，则接近 96 MB/层。层数一多，通信立刻变成主成本之一。

### 6. SP 为什么能继续省显存

Sequence Parallel，简称 SP，白话说就是“参数仍按 TP 方式切，但某些激活按序列维分到不同卡上存”。它不改变权重切分方式，主要减少非 attention 模块的激活显存。

比如 LayerNorm 是逐 token 做归一化。若序列维被切成 $S/N$，每张卡只处理自己那段 token，就不需要每张卡都存完整 $S$。因此在长序列场景，SP 常和 TP 搭配使用。

---

## 代码实现

下面用一个最小可运行的 Python 例子，模拟“FFN 第一层列切分、第二层行切分”的数学等价。这个例子不依赖分布式库，只验证切分逻辑本身。

```python
import numpy as np

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def ffn_dense(x, w1, w2):
    return gelu(x @ w1) @ w2

def ffn_tensor_parallel(x, w1, w2, tp_size):
    # 第一层按列切分：每张卡拿一部分输出神经元
    w1_shards = np.split(w1, tp_size, axis=1)
    hidden_shards = [gelu(x @ shard) for shard in w1_shards]

    # 第二层按行切分：每张卡拿一部分输入通道，输出是部分和
    w2_shards = np.split(w2, tp_size, axis=0)
    partial_outputs = [h @ w for h, w in zip(hidden_shards, w2_shards)]

    # 模拟 All-Reduce：把所有部分和相加
    y = sum(partial_outputs)
    return y

rng = np.random.default_rng(0)

batch_seq = 6
hidden = 8
ffn_hidden = 16
tp_size = 4

x = rng.normal(size=(batch_seq, hidden))
w1 = rng.normal(size=(hidden, ffn_hidden))
w2 = rng.normal(size=(ffn_hidden, hidden))

y_dense = ffn_dense(x, w1, w2)
y_tp = ffn_tensor_parallel(x, w1, w2, tp_size)

assert np.allclose(y_dense, y_tp, atol=1e-6)
print("tensor parallel FFN matches dense FFN")
```

上面的 `assert` 说明：只要切分位置选对，TP 和单卡 dense 计算在数学上是一致的。

如果把这个思路对应到 Megatron-LM 的模块，可以理解为：

```python
# 伪代码
qkv, _ = ColumnParallelLinear(x)     # 每张卡得到部分头
attn_local = local_attention(qkv)    # 本地头内计算
attn_out, _ = RowParallelLinear(attn_local)   # 这里做 All-Reduce

ffn_mid, _ = ColumnParallelLinear(attn_out)   # 每张卡得到部分中间神经元
ffn_mid = gelu(ffn_mid)
ffn_out, _ = RowParallelLinear(ffn_mid)       # 这里再做 All-Reduce
```

真实工程例子可以看 GPT-3 175B 的训练配置。NVIDIA 在 Selene 上采用了 TP 和 PP 的组合，常见说明是 8-way TP 配合 8-way PP。这个配置的含义不是“所有 64 张卡一起做张量并行”，而是：

1. 单节点内 8 张卡组成一个 TP 组，利用 NVLink 做高频 All-Reduce。
2. 不同节点之间主要用 PP 串接不同层，避免把最贵的 TP 通信拖到跨节点链路上。
3. 再在更外层叠加数据并行，扩大总体吞吐。

这正说明了一个工程原则：TP 解决单层过大，PP 解决整模过深，DP 解决整体吞吐，三者职责不同。

---

## 工程权衡与常见坑

TP 真正难的部分不在“能不能切开”，而在“切开后是不是还划算”。

| 风险 | 具体表现 | 处理方式 |
| --- | --- | --- |
| TP 跨慢链路 | All-Reduce 不能被计算遮住，吞吐明显下降 | 尽量把 TP 限制在 NVLink/NVSwitch 域内 |
| TP 设太大 | 每卡矩阵太小，GEMM 反而不饱和 | 优先选择较小 TP，配合 PP/DP |
| 不开 SP | LayerNorm/Dropout 等激活仍然完整复制 | 长序列场景打开 SP |
| 微批过小 | 计算时间太短，通信占比过高 | 增大 micro-batch 或梯度累积 |
| 误解“每卡只算部分头” | 以为所有 attention 都无需同步 | 只在头内独立，$W_o$ 和 FFN2 必须汇总 |

有两个坑尤其常见。

第一，很多人第一次接触 TP，会觉得“既然每张卡只算自己那几个头，那 attention 不就天然并行，通信应该很少”。这只对前半段成立。Q/K/V 和头内 attention 的确可以局部完成，但 attention block 的输出不是到此为止，后面还有 $W_o$。这一层如果不汇总，就得不到和单卡一致的隐藏状态。

第二，很多人会把 TP 当成显存万能药。实际上 TP 主要切的是权重和部分激活计算，不能自动消掉所有激活副本。尤其长序列训练里，真正让显存爆炸的常常不是参数，而是中间激活，所以 SP 或 CP 常常必须一起用。

再说性能边界。工程实践里，8 卡 TP 已经很常见，但通信代价不低。即使在 A100 + 高速互联环境下，All-Reduce 也往往是单步时间里的主要组成之一；如果把 TP 扩到跨节点，问题会迅速放大。这也是 NVIDIA 文档反复强调“TP 应限制在高带宽节点内”的原因。更直接地说：TP 不是越大越好，而是只开到“刚好让模型装得下”为止。

---

## 替代方案与适用边界

当 TP 不再划算时，不是训练到此为止，而是该换并行维度。

| 方法 | 适用场景 | 优点 | 代价 |
| --- | --- | --- | --- |
| TP | 单层矩阵太大 | 数学结构清晰，Megatron 支持成熟 | 高频 All-Reduce |
| PP | 整模太深、层数太多 | 跨节点更容易扩 | 有 pipeline bubble |
| SP | 长序列、激活显存高 | 明显降低非注意力激活显存 | 需要额外 gather/scatter |
| CP | 超长上下文 | 直接按序列切更多计算 | attention 通信更复杂 |
| FSDP/ZeRO | 模型状态过大但不想做层内切分 | 通用性强 | 参数收发频繁，训练形态不同 |

可以用一个简单判断流程：

1. 如果模型本身能放下，先用 DP，不要急着上 TP。
2. 如果单层太大放不下，先加 TP，但尽量限制在单节点。
3. 如果单节点 TP 还不够，再加 PP，而不是盲目把 TP 扩到跨节点。
4. 如果主要问题是长序列激活爆炸，优先考虑 SP 或 CP。
5. 如果你需要更通用的参数分片，而不是 Megatron 风格的层内结构改造，可以考虑 FSDP。

对初级工程师，一个实用结论是：TP 不是“大模型训练的标准答案”，而是“大隐藏维 Transformer 的一种专用切法”。它在 Megatron-LM 里非常有效，是因为 Transformer 的 QKV、$W_o$、FFN 结构刚好适合列切分和行切分交替出现。换了模型结构，这套方法不一定同样优雅。

---

## 参考资料

- Shoeybi, Mohammad, et al. “Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.” 2019. https://research.nvidia.com/labs/adlr/MegatronLM/
- NVIDIA Technical Blog, “Scaling Language Model Training to a Trillion Parameters Using Megatron.” https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/
- NVIDIA Megatron Core API, `core.tensor_parallel.layers`. https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html
- NVIDIA Megatron Bridge Performance Tuning Guide. https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html
- Hugging Face Accelerate, “Megatron-LM”. https://huggingface.co/docs/accelerate/en/usage_guides/megatron_lm
- DeepWiki, “Core Model Architecture | NVIDIA/Megatron-LM”. https://deepwiki.com/NVIDIA/Megatron-LM/2-core-architecture
- DeepWiki, “Context and Sequence Parallelism | NVIDIA/Megatron-LM”. https://deepwiki.com/NVIDIA/Megatron-LM/4.5-context-and-sequence-parallelism
- Laurens Weitkamp, “Tensor Parallelism”. https://lweitkamp.github.io/posts/numpitron_tensor_parallel/index.html
