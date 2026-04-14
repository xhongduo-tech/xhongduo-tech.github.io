## 核心结论

Grouped Query Attention，简称 GQA，本质上是“让一部分注意力头共享同一套 Key 和 Value 的表示”。白话说，就是很多人在分别提问，但不再给每个人单独准备一整套资料，而是按组共用资料。

它处在 Multi-Head Attention（MHA，多头注意力）和 Multi-Query Attention（MQA，多查询注意力）之间：

| 方案 | Query 头 | Key/Value 头 | KV 缓存占用 | 表示能力 | 推理吞吐 |
|---|---:|---:|---:|---:|---:|
| MHA | $h$ | $h$ | 最高 | 最强 | 最慢 |
| GQA | $h$ | $g$ | 中等 | 中等偏高 | 中等偏快 |
| MQA | $h$ | $1$ | 最低 | 最弱 | 最快 |

其中，$h$ 是 query 头数，$g$ 是 KV 头数。注意这里的核心不是“减少 query 头”，而是“减少需要缓存和读取的 KV 头”。

GQA 的关键价值不在训练省算力，而在自回归推理阶段。自回归推理就是模型一个 token 一个 token 往后生成，每次都要读取历史 token 的 KV cache。这个缓存越大，显存占用越高，带宽压力越大，速度越慢。GQA 通过减少 KV 头数，把这个瓶颈直接压下去。

一个常见结论是：

- 当 $g=h$ 时，GQA 退化为 MHA
- 当 $g=1$ 时，GQA 退化为 MQA
- 当 $1<g<h$ 时，GQA 提供“质量和效率之间的可调中间点”

LLaMA 2 70B 使用的是典型的 GQA 配置：64 个 query 头配 8 个 KV 头。这意味着每 8 个 query 头共享一组 KV，在保留较强表达能力的同时，显著降低了推理阶段 KV cache 的体积和访存成本。

---

## 问题定义与边界

先把问题说清楚：GQA 要解决的不是“注意力算不算得出来”，而是“在长上下文和大模型推理时，KV cache 太大，导致显存和带宽成为瓶颈”。

在标准 MHA 里，每个头都有自己的 $Q,K,V$。如果有 $h$ 个头，那么每生成一个新 token，都要把这个 token 的 $K,V$ 以 $h$ 份的形式写入缓存；之后每一步生成，还要把历史上所有 token 的这些 $K,V$ 再读出来做注意力。序列越长，缓存越大，读写越重。

GQA 的定义是：保留 $h$ 个 query 头，但只保留 $g$ 个 KV 头，并让每个 KV 头服务一组 query 头。通常要求 $h$ 能被 $g$ 整除，这样每组包含 $\frac{h}{g}$ 个 query 头。

设第 $j$ 个 query 头属于第 $\phi(j)$ 个组，则它的注意力写成：

$$
\mathrm{Attn}(Q_j, K_{\phi(j)}, V_{\phi(j)})=
\mathrm{softmax}\left(\frac{Q_jK_{\phi(j)}^\top}{\sqrt{d_k}}\right)V_{\phi(j)}
$$

如果按组写，就是：

$$
\operatorname{Attention}_g(Q^g,K^g,V^g)=
\operatorname{softmax}\left(\frac{Q^g (K^g)^\top}{\sqrt{d_k}}\right)V^g
$$

这里的意思很直接：

- $Q^g$：组内每个 query 头自己的提问向量
- $K^g,V^g$：该组共享的资料向量
- 组内不同头的问题不同，但它们查的是同一组资料

一个玩具例子：

假设模型有 8 个 query 头。如果用 MHA，那么有 8 份 KV；如果用 GQA 且 $g=2$，那么只有 2 份 KV。也就是 8 个头分成 2 组，每组 4 个 query 头共享一套 KV。这样 query 头的“提问角度”还在，但后台要缓存的“资料副本”减少到了原来的四分之一。

这里要明确边界：GQA 主要优化的是推理，尤其是 decoder-only 模型的增量生成。对短序列训练、小模型实验、或不使用 KV cache 的场景，它的收益不会像长文本生成那样明显。

---

## 核心机制与推导

先回顾 MHA。对输入 $x$，通常会做三次线性投影：

$$
Q = xW_Q,\quad K = xW_K,\quad V = xW_V
$$

然后把它们拆成 $h$ 个头，每个头维度为 $d_k$。第 $i$ 个头的输出是：

$$
\mathrm{head}_i=\mathrm{softmax}\left(\frac{Q_iK_i^\top}{\sqrt{d_k}}\right)V_i
$$

最后把所有头拼起来。

GQA 的变化点只有一个：$Q$ 仍然保留 $h$ 个头，但 $K,V$ 不再有 $h$ 个头，而是只有 $g$ 个头。于是每个 query 头不再对应“自己的” $K_i,V_i$，而是映射到某个组的共享 $K,V$。

如果把 query 头编号为 $0,1,\dots,h-1$，一个简单分组函数可以写成：

$$
\phi(j)=\left\lfloor \frac{j}{h/g}\right\rfloor
$$

意思是：每连续 $\frac{h}{g}$ 个 query 头归到一组。

### 为什么它能省缓存

对自回归推理来说，缓存的不是 $Q$，而是历史 token 的 $K,V$。所以缓存规模近似正比于 KV 头数。若序列长度为 $T$，batch 为 $B$，每头维度为 $d_k$，数据类型字节数为 $s$，则：

- MHA 的 KV cache 约为  
  $$
  2 \times B \times T \times h \times d_k \times s
  $$
- GQA 的 KV cache 约为  
  $$
  2 \times B \times T \times g \times d_k \times s
  $$

因此缓存压缩比大致是：

$$
\frac{\text{MHA cache}}{\text{GQA cache}}=\frac{h}{g}
$$

这也是为什么 GQA 的收益通常写成“KV cache 降为原来的 $\frac{g}{h}$”。

### 玩具例子

设：

- $h=8$ 个 query 头
- $g=2$ 个 KV 头
- 每头维度 $d_k=64$

那么：

- MHA 需要缓存 8 份 $K$ 和 8 份 $V$
- GQA 只需要缓存 2 份 $K$ 和 2 份 $V$

缓存压缩比是：

$$
\frac{8}{2}=4
$$

也就是 4:1 压缩，但输出仍然来自 8 个 query 头。注意，这不是“头数减少到 2”，而是“参与提问的头仍是 8 个，只有资料头变成了 2 个”。

### 真实工程例子

LLaMA 2 70B 使用的是 64 个 query 头配 8 个 KV 头。于是缓存压缩比约为：

$$
\frac{64}{8}=8
$$

也就是相比同规模的全 MHA 配置，KV cache 体积和读写量大约缩小到八分之一量级。对长上下文生成，这种差异会直接反映到延迟和吞吐上。原因很简单：现代 GPU 上，很多大模型推理并不是纯计算受限，而是显存带宽受限。KV 更少，访存更轻，速度就能上来。

### 为什么不是直接用 MQA

MQA 相当于 $g=1$，所有 query 头共享同一套 KV。它最省缓存，但共享过头了。不同 query 头虽然仍有不同的 $Q$，但它们访问的“资料视角”完全一样，建模自由度下降更多。GQA 的意义就是：不把共享做得这么极端，而是保留多个 KV 组，让不同 query 头群有不同的上下文读法。

---

## 代码实现

实现 GQA 时，最容易搞混的是“组大小”和“组数”。工程里通常更稳妥的写法是直接用 `num_q_heads` 和 `num_kv_heads` 两个变量，不要用一个 `g` 同时表示两件事。

下面给一个可运行的 Python 版本，用最小实现展示 GQA 的核心逻辑。这里用 `numpy`，并带 `assert` 检查形状与概率归一化。

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def grouped_query_attention(Q, K, V):
    """
    Q: (batch, q_heads, seq_q, d)
    K: (batch, kv_heads, seq_k, d)
    V: (batch, kv_heads, seq_k, d)
    return: (batch, q_heads, seq_q, d)
    """
    batch, q_heads, seq_q, d = Q.shape
    batch2, kv_heads, seq_k, d2 = K.shape
    assert batch == batch2 and d == d2
    assert V.shape == (batch, kv_heads, seq_k, d)
    assert q_heads % kv_heads == 0

    heads_per_group = q_heads // kv_heads
    out = np.zeros((batch, q_heads, seq_q, d), dtype=np.float64)

    for qh in range(q_heads):
        kvh = qh // heads_per_group
        scores = np.matmul(Q[:, qh], K[:, kvh].transpose(0, 2, 1)) / np.sqrt(d)
        attn = softmax(scores, axis=-1)
        out[:, qh] = np.matmul(attn, V[:, kvh])

        # 每个 query 位置对所有 key 位置的概率和应为 1
        assert np.allclose(attn.sum(axis=-1), 1.0, atol=1e-6)

    return out

# 玩具数据
rng = np.random.default_rng(0)
B, q_heads, kv_heads, Tq, Tk, D = 2, 8, 2, 3, 5, 4
Q = rng.normal(size=(B, q_heads, Tq, D))
K = rng.normal(size=(B, kv_heads, Tk, D))
V = rng.normal(size=(B, kv_heads, Tk, D))

O = grouped_query_attention(Q, K, V)
assert O.shape == (B, q_heads, Tq, D)

# 验证：同组 query 头共享同一个 KV 头
assert (0 // (q_heads // kv_heads)) == (3 // (q_heads // kv_heads))
assert (4 // (q_heads // kv_heads)) != (0 // (q_heads // kv_heads))
print("GQA toy example passed.")
```

如果改成 PyTorch 伪代码，核心结构通常是这样：

```python
Q = proj_q(x).view(batch, seq, h, dk).transpose(1, 2)       # (B, h, T, dk)
K = proj_k(x).view(batch, seq, g, dk).transpose(1, 2)       # (B, g, T, dk)
V = proj_v(x).view(batch, seq, g, dk).transpose(1, 2)       # (B, g, T, dk)

group_size = h // g
kv_index = torch.arange(h, device=x.device) // group_size    # (h,)
K_shared = K[:, kv_index]                                    # (B, h, T, dk)
V_shared = V[:, kv_index]                                    # (B, h, T, dk)

scores = torch.matmul(Q, K_shared.transpose(-1, -2)) / math.sqrt(dk)
attn = torch.softmax(scores, dim=-1)
out = torch.matmul(attn, V_shared)                           # (B, h, T, dk)
```

这段逻辑说明了两件事：

1. `Q` 仍有 $h$ 个头。
2. `K,V` 只有 $g$ 个头，但在计算时会按分组关系映射给每个 query 头使用。

工程上还有一个关键点：很多 GQA 模型不是从零开始训练，而是从已有 MHA checkpoint 出发做 uptraining。uptraining 就是在已有模型参数基础上继续训练少量步数，让模型适应新的 KV 分组结构。这样做的原因很现实：从零训练大模型太贵，而把已有 MHA 权重迁移到 GQA 再微调，成本低得多。

---

## 工程权衡与常见坑

GQA 不是“免费午餐”。它解决了推理缓存问题，但会牺牲一部分表示能力。因为多个 query 头共享同一套 KV，本质上减少了“读上下文”的独立视角。

可以用一个工程化表格看这个平衡：

| 配置 | KV 头数 | 缓存压缩比（相对 MHA） | 质量风险 | 适用场景 |
|---|---:|---:|---:|---|
| MHA | $h$ | 1x | 最低 | 质量优先、资源充足 |
| GQA-大组数 | 接近 $h$ | 小幅压缩 | 很低 | 想保质量又要一点加速 |
| GQA-中组数 | 中等 | 明显压缩 | 可控 | 大模型在线推理 |
| MQA | 1 | 最大压缩 | 最高 | 极端带宽/显存受限 |

常见坑主要有五类。

第一，混淆“组数”和“每组多少个头”。  
有的资料把 $g$ 记为组数，有的记为每组 query 头数。如果代码和公式混着看，很容易出错。最稳妥的做法是代码里明确写 `num_q_heads`、`num_kv_heads`、`group_size`。

第二，只看 FLOPs，不看带宽。  
很多人第一次接触 GQA，会去算矩阵乘法规模，发现并没有像预期那样大幅下降，于是误以为收益有限。问题在于，大模型增量推理的主瓶颈常常是 KV cache 读写而不是单步算子 FLOPs。GQA 省的是缓存和带宽，不只是乘法次数。

第三，盲目把 KV 头数压到 1。  
这就是退化成 MQA。MQA 的确更快，但任务一旦对细粒度上下文区分更敏感，质量损失可能会放大。尤其是长文理解、复杂推理、代码生成这类任务，更依赖不同头从不同角度读取上下文。

第四，错误处理 RoPE、mask 和 cache 对齐。  
RoPE 就是旋转位置编码，它通常作用在 $Q,K$ 上。如果从 MHA 改成 GQA，必须确认 query 头和共享 KV 头在位置编码、历史缓存追加、mask 广播上都保持一致。很多实现 bug 不是出在 attention 公式，而是出在 cache 的 shape 和广播规则上。

第五，忽略 checkpoint 迁移方式。  
如果已有模型是 MHA 权重，直接粗暴地把多个头平均成一个 KV 组，再不做后续训练，性能往往会掉。更合理的路径是做 uptraining，让模型重新适应“共享 KV”的约束。

真实工程里，GQA 通常出现在“大模型服务端推理”场景。例如一个多用户并发的对话服务，瓶颈不在单张卡峰值算力，而在显存、KV cache 容量、吞吐稳定性。此时从 MHA 改成中等规模的 GQA，往往比继续堆硬件更划算。

---

## 替代方案与适用边界

GQA 不是唯一方案。至少有三条常见路线：

| 方案 | 核心做法 | 优势 | 劣势 | 适用边界 |
|---|---|---|---|---|
| MHA | 每个头独立 Q/K/V | 表达力最强 | KV cache 最大 | 质量绝对优先 |
| GQA | 多个 Q 头共享组级 KV | 质量和效率平衡 | 仍需调参和迁移 | 大模型主流推理优化 |
| MQA | 所有 Q 头共享一套 KV | 缓存最省、速度快 | 表示能力下降更多 | 资源极度紧张 |
| 稀疏注意力 | 只看部分位置 | 长序列更省 | 实现复杂，模式依赖任务 | 超长上下文 |

可以把它们理解成一个连续谱：

```text
MHA  ------------------  GQA  ------------------  MQA
质量最高                    折中                     资源最省
缓存最大                    可调                     缓存最小
```

再给一个新手容易理解的类比，但不替代定义：  
MHA 像“每个人都有自己的一套资料”；GQA 像“每个小组共享一套资料，但组内每个人问的问题不同”；MQA 像“所有人都看同一套资料”。当任务需要非常细致、很多角度的上下文区分时，越靠左越稳；当部署成本和延迟压力很大时，越靠右越省。

适用边界也要说清楚：

- 如果你做的是研究型基线、追求最高上限，优先 MHA。
- 如果你做的是在线推理、长上下文、并发服务，GQA 往往是更实际的默认选项。
- 如果你的显存或带宽极其紧张，并且能接受一定质量损失，可以考虑 MQA。
- 如果核心问题是“序列太长”，而不是“单步 KV 太贵”，那还要考虑稀疏注意力、滑动窗口注意力、状态空间模型等其他路线。

换句话说，GQA 解决的是“标准全注意力框架下，如何把 KV cache 做小一些”，不是所有注意力效率问题的总解。

---

## 参考资料

1. Ainslie et al.，《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》  
   重点：给出 GQA 定义，并强调从 MHA checkpoint 迁移到 GQA 的 uptraining 路径。

2. IBM 关于 Grouped Query Attention 的技术解读  
   重点：对 MHA、GQA、MQA 的关系和推理效率收益做直观说明，适合先建立整体图景。

3. GeeksforGeeks 关于 GQA 的公式与性能总结  
   重点：整理了基础公式、共享 KV 的实现方式，以及与其他注意力方案的对比。

4. AI Wiki 对 LLaMA 系列 GQA 配置的资料整理  
   重点：帮助把抽象机制和真实模型配置对应起来，例如 LLaMA 2 70B 的 64Q / 8KV 配置。

5. Hugging Face Papers 页面中的论文索引与社区解读  
   重点：适合快速定位原论文、相关实现和后续模型采用情况。
