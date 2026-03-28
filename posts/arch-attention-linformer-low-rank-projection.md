## 核心结论

Linformer 的核心做法是把注意力里的序列维度先压缩，再计算注意力。这里的“低秩”可以先理解成：虽然原始注意力矩阵是 $n \times n$，但其中真正重要的信息常常只集中在少数几个主方向上，不需要把全部 $n^2$ 个关系都精确保留。

标准 self-attention 会先构造 $QK^\top$，其中 $Q,K \in \mathbb{R}^{n \times d}$，所以会得到一个 $n \times n$ 的打分矩阵。Linformer 引入两个可训练投影矩阵 $E,F \in \mathbb{R}^{k \times n}$，把
$$
K' = EK,\quad V' = FV
$$
于是 $K',V' \in \mathbb{R}^{k \times d}$，再计算
$$
\mathrm{LinAttention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK'^\top}{\sqrt d}\right)V'
$$
此时注意力打分不再是 $n \times n$，而是 $n \times k$。当 $k \ll n$ 时，时间和显存都从依赖 $n^2$ 下降到依赖 $nk$。

这件事成立的前提不是“注意力天然线性”，而是“很多任务里的注意力矩阵近似低秩”。经验上，部分编码器任务中前 128 个奇异值就能覆盖大部分能量，因此用较小的 $k$ 逼近原始注意力不会明显掉点。对文本分类、句向量、检索编码器这类非自回归场景，Linformer 是有工程价值的；对生成式解码器，它通常不是优先方案。

---

## 问题定义与边界

Linformer 要解决的问题很直接：Transformer 的全连接注意力在长序列上太贵。

“序列长度”就是 token 个数，记作 $n$；“隐藏维度”就是每个 token 的特征长度，记作 $d$。标准 self-attention 的主要瓶颈有两项：

| 方法 | 打分矩阵形状 | 时间复杂度 | 注意力显存 | 是否保留全局依赖 | 典型限制 |
|---|---|---:|---:|---|---|
| 标准 Attention | $n \times n$ | $O(n^2 d)$ | $O(n^2)$ | 是 | 长序列成本高 |
| Linformer | $n \times k$ | $O(nkd)$ | $O(nk)$ | 是 | 依赖低秩假设、通常要求固定长度 |
| 局部 Attention | 局部窗口 | $O(nwd)$ | $O(nw)$ | 否 | 长程依赖可能丢失 |
| Performer | 近似线性 | 近似 $O(ndm)$ | 近似线性 | 是 | 近似误差来自核映射 |

这里的边界也要说清楚。Linformer 不是把任意任务都“白送”成线性复杂度，它依赖三个前提。

第一，注意力矩阵必须有明显谱衰减。“谱衰减”可以先理解成：把矩阵按重要程度分解后，前几个成分占了绝大多数信息。如果谱分布很平，说明很多方向都重要，低秩压缩就会损失明显。

第二，它更适合编码器任务。因为 Linformer 压缩的是整段序列的 $K,V$，默认这些 token 可以同时看见。自回归生成要求当前位置不能看到未来 token，这和固定投影后的整段可见性天然有冲突。

第三，$k$ 不是永远固定不变的免费参数。理论和经验都说明，序列更长时，想维持同样误差，$k$ 往往也要增大。常见表述是近似误差有界时，$k$ 至少随 $\log n$ 或更保守地随 $\sqrt n$ 增长。

一个玩具例子可以先建立直觉。假设一句话只有 8 个 token，但真正有贡献的关系只有“主语-谓语”“代词-指代对象”“否定词-动词”这几类模式，那么完整的 $8 \times 8$ 打分矩阵里，大部分位置的自由度其实是冗余的。Linformer 的意思就是：既然有效关系远少于全部组合，就先把 $K,V$ 压到更短的表示，再做注意力。

---

## 核心机制与推导

先看标准注意力：
$$
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt d}\right),\qquad Y = AV
$$
其中 $A \in \mathbb{R}^{n \times n}$。真正贵的是 $QK^\top$ 和保存 $A$。

Linformer 不直接近似输出 $Y$，而是近似注意力里的序列交互结构。它假设 $A$ 或与之相关的相似度矩阵具有低秩结构，于是把 $K,V$ 在序列维度上投影：
$$
K' = EK,\quad V' = FV,\quad E,F \in \mathbb{R}^{k \times n}
$$
这样 $Q \in \mathbb{R}^{n \times d}$ 与 $K'^\top \in \mathbb{R}^{d \times k}$ 相乘后，只得到 $n \times k$ 的打分矩阵：
$$
S = \frac{QK'^\top}{\sqrt d}
$$
再做 softmax：
$$
\hat A = \mathrm{softmax}(S),\qquad Y = \hat A V'
$$

为什么压缩 $K,V$ 合理？可以从奇异值分解理解。奇异值分解 SVD 可以先理解成：把一个矩阵拆成若干个由强到弱的模式叠加。如果注意力矩阵的奇异值序列
$$
\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_n
$$
下降很快，那么前 $r$ 个分量已经解释了大部分能量：
$$
\frac{\sum_{i=1}^{r}\sigma_i^2}{\sum_{i=1}^{n}\sigma_i^2} \approx 1
$$
这时用秩为 $k$ 的近似去替代原始矩阵，误差就可控。论文和后续经验分析中，一个常见观察是前 128 个奇异值可覆盖 90% 以上能量，这正是 “小 $k$ 仍能保精度” 的依据。

再给一个数值推导。设 $n=1024,d=512,k=256$。

标准注意力的打分规模是：
$$
1024 \times 1024 = 1{,}048{,}576
$$
Linformer 的打分规模是：
$$
1024 \times 256 = 262{,}144
$$
只看打分矩阵，已经是原来的四分之一。显存中保存的注意力权重也同步从 $O(n^2)$ 变成 $O(nk)$。如果模型有多层多头，这个差距会被进一步放大。

真实工程例子是 BERT 或 RoBERTa 类编码器做分类。比如输入长度从 512 拉到 1024 后，标准 attention 的二次成本增长非常明显；Linformer 若选 $k=256$，往往可以把速度拉回到可接受区间，同时在 GLUE、IMDB 这类任务上保持和全注意力接近的效果。这不是因为它学到了更强表达，而是因为这类任务的注意力结构本身就足够可压缩。

---

## 代码实现

下面给一个可运行的 Python 版本，用 `numpy` 演示单头 Linformer 注意力。它不是高性能实现，但能把公式和张量形状说明白。

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def linformer_attention(Q, K, V, E, F):
    """
    Q, K, V: (n, d)
    E, F:    (k, n)
    return:  (n, d)
    """
    n, d = Q.shape
    k = E.shape[0]

    assert K.shape == (n, d)
    assert V.shape == (n, d)
    assert E.shape[1] == n
    assert F.shape == (k, n)

    K_proj = E @ K          # (k, d)
    V_proj = F @ V          # (k, d)

    scores = (Q @ K_proj.T) / np.sqrt(d)   # (n, k)
    attn = softmax(scores, axis=-1)        # (n, k)

    out = attn @ V_proj                    # (n, d)
    return out, attn

# 玩具例子：8 个 token，4 维特征，压到 3 个投影位置
rng = np.random.default_rng(42)
n, d, k = 8, 4, 3

Q = rng.normal(size=(n, d))
K = rng.normal(size=(n, d))
V = rng.normal(size=(n, d))

E = rng.normal(size=(k, n)) / np.sqrt(n)
F = rng.normal(size=(k, n)) / np.sqrt(n)

out, attn = linformer_attention(Q, K, V, E, F)

assert out.shape == (n, d)
assert attn.shape == (n, k)
assert np.allclose(attn.sum(axis=-1), 1.0, atol=1e-6)

print("output shape:", out.shape)
print("attention row sums:", attn.sum(axis=-1))
```

如果换成 PyTorch，多头实现的核心一般是两步：

1. 先把输入映射成 `q, k, v`，形状通常为 `(batch, heads, n, d_head)`。
2. 再在序列维度上做投影，例如：
```python
k_proj = torch.einsum('bhnd,kn->bhkd', k, E)
v_proj = torch.einsum('bhnd,kn->bhkd', v, F)
scores = torch.einsum('bhnd,bhkd->bhnk', q, k_proj) / math.sqrt(d_head)
attn = torch.softmax(scores, dim=-1)
out = torch.einsum('bhnk,bhkd->bhnd', attn, v_proj)
```

这里的 `einsum` 可以先理解成：用显式下标写张量乘法，减少自己手动 `reshape` 时出错的概率。

工程里通常还会加三种策略。

第一，按层共享 $E,F$。这样参数更省，但表达能力会下降一些。

第二，按头共享或独立。独立投影更灵活，共享投影更省参数。

第三，固定最大长度 `seq_len`。因为 $E,F$ 的形状依赖 $n$，很多实现会直接把最大序列长度写死，短序列靠 padding 对齐。

---

## 工程权衡与常见坑

Linformer 的好处很明确，但它不是“无脑替换标准 attention”的模块。下面这张表更适合工程 review 时逐项核对。

| 问题 | 现象 | 原因 | 对策 |
|---|---|---|---|
| $k$ 设得过小 | 精度明显下降 | 压缩过强，重要关系丢失 | 从 128/256 起扫参，按验证集选 |
| 变长输入不稳定 | 训练正常，推理异常 | 投影矩阵依赖固定长度 $n$ | 固定最大长度并统一 padding |
| 生成任务效果差 | 解码质量不稳 | 自回归掩码和全局投影不兼容 | 生成模型优先考虑其他线性注意力 |
| 盲目共享投影 | 某些层掉点更多 | 不同层的低秩结构并不一致 | 先做分层实验，再决定是否共享 |
| 只看理论不做 profiling | 实际加速不明显 | 小模型上内核和访存开销占主导 | 用真实 batch、真实长度做基准 |
| 忽略谱分析 | 某任务上逼近失败 | 注意力矩阵不够低秩 | 先检查奇异值衰减，再决定是否采用 |

最常见的坑有三个。

第一，把 Linformer 用到自回归生成上。它的原始设计主要面向编码器，因为压缩后的 $K,V$ 默认是整段可见的。对解码器而言，未来 token 不可见是硬约束，不能简单复用这套投影。

第二，把固定长度问题想得太轻。$E,F$ 的维度是 $k \times n$，这意味着训练时的 `n=512` 和推理时的 `n=1024` 并不是同一个参数形状。很多库通过 `Padder` 或固定 `input_size` 规避这个问题，但代价是灵活性变差。

第三，只看平均速度，不看精度-长度曲线。比如在 512 长度上 $k=128$ 没问题，不代表 2048 长度上仍然没问题。更稳妥的做法是：按不同长度画验证集曲线，看误差是否随着 $n$ 增长而系统性扩大。

真实工程中，一个比较稳的流程是：先在候选模型上抽样若干层的注意力矩阵，做 SVD 统计前 64、128、256 个奇异值的累计能量占比；如果前 128 个奇异值长期低于 90%，说明低秩假设不强，Linformer 不是优先选项。如果谱衰减明显，再进入 $k$ 和共享策略搜索。

---

## 替代方案与适用边界

Linformer 适用于“长序列、编码器、注意力谱衰减明显”的场景，但它不是唯一的线性化路线。

| 方法 | 核心思想 | 更适合的场景 | 主要代价 |
|---|---|---|---|
| Linformer | 直接压缩 $K,V$ 的序列维度 | 编码器、分类、嵌入 | 固定长度，自回归不友好 |
| Performer | 用核技巧近似 softmax attention | 更长序列、需更通用近似 | 近似质量依赖随机特征 |
| 局部/稀疏 Attention | 只看邻域或预定义稀疏模式 | 长文档、明显局部结构 | 全局依赖需要额外设计 |
| Nyström 类方法 | 用采样锚点近似全矩阵 | 中长序列、低秩近似 | 锚点选择影响稳定性 |

如果任务是文本分类、双塔检索、句向量编码，Linformer 值得优先考虑。因为这些任务常用编码器结构，而且更关注整体表征，不要求逐 token 自回归生成。

如果任务是长文本生成、代码补全、对话模型解码，则应优先考虑对因果掩码更友好的方案，如 Performer、滑窗加全局 token、KV cache 优化后的标准 attention 变体。原因不是 Linformer 完全不能做，而是它的原始优势和这些任务的约束不匹配。

一句话总结适用边界：Linformer 适合把“全局依赖保留”与“二次复杂度下降”同时拿到，但前提是你的注意力真的可以被低秩近似。

---

## 参考资料

- Linformer 原论文：Wang et al., *Linformer: Self-Attention with Linear Complexity*, arXiv:2006.04768  
- Emergent Mind: Linformer Low-Rank Attention  
- Next Electronics: *Linformer and Performer: Linear Transformers*  
- lucidrains/linformer GitHub 仓库说明  
- `linformer-pytorch` 项目文档  
- 若干关于 GLUE/IMDB 实验复现与谱分析的技术解读文章
