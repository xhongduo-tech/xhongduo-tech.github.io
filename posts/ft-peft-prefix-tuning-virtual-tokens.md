## 核心结论

Prefix Tuning 的核心做法是：在 Transformer 的每一层注意力前，额外插入一小段可学习的“虚拟 token”。这里的“虚拟 token”不是输入文本里的真实词，而是一组专门训练出来的向量，作用是给模型提供任务条件。更准确地说，它们会被拼接到每层的 key/value 前面，让后续真实 token 的 query 在做注意力时，既能看见真实上下文，也能读到这段任务前缀。

它解决的问题是“冻结大模型，只用极少参数完成任务适配”。如果全量微调要改整个模型，LoRA 要改线性层的低秩增量，那么 Prefix Tuning 改的是注意力能访问到的上下文，因此可以概括为：它不是改模型“怎么计算”，而是改模型“先看到什么”。

对初学者，最直白的理解是：给每层 attention 贴一小段可训练记忆，模型推理时会把这段记忆一起读进去，但原模型权重不动。

| 方法 | 改动位置 | 训练参数范围 | 推理额外开销 | 典型特点 |
|---|---|---:|---:|---|
| 全量微调 | 全部权重 | 100% | 无额外结构开销 | 效果强，但存储和训练成本高 |
| LoRA | 线性层增量矩阵 | 很低 | 通常较小 | 改投影权重，部署成熟 |
| Prefix Tuning | 每层 attention 的 K/V 前缀 | 很低，常见 0.1% 到 2% | 随 prefix 长度线性增加 | 不改原权重，易按任务切换 |

---

## 问题定义与边界

问题定义很明确：在冻结预训练模型参数的前提下，用最少的新增参数让模型适配新任务。这里最关键的控制量是 prefix 长度 $k$ 和隐藏维度 $d$。$k$ 可以理解为“每层塞进去多少个虚拟 token”，$d$ 是“每个向量有多宽”。

边界也很明确。第一，prefix 太短，表达能力不够。模型虽然能读到任务条件，但条件信息量太小，无法稳定影响输出。第二，prefix 太长，会直接增加注意力计算量，因为每个 query 都要额外和这些 prefix 做一次匹配；在自回归生成里，还会增加 KV cache，也就是推理时缓存的键值对内存。KV cache 可以白话理解为“为了下一个 token 不重复计算而保存的中间结果”。

一个新手常见误解是：Prefix Tuning 既然参数少，是不是一定更快。答案是否定的。训练参数少，不等于推理零成本。它省掉的是“改大模型权重”的成本，不是“注意力里多看几项”的成本。

下面这个简表能看到 $k$ 的典型权衡：

| prefix 长度 $k$ | 表达力 | 训练稳定性 | 推理时延 | KV cache 压力 | 适用情况 |
|---|---|---|---|---|---|
| 很短，如 2 到 4 | 弱 | 容易欠拟合 | 低 | 低 | 简单分类、风格偏置 |
| 中等，如 8 到 32 | 中等到较强 | 常见最优区间 | 中等 | 中等 | 摘要、问答、格式控制 |
| 很长，如 64 以上 | 强 | 可能更难调 | 高 | 高 | 复杂条件控制，但部署成本上升 |

玩具例子：假设一个 12 层、隐藏维度 768 的模型，每层只插入 4 个 prefix token，那么参数量近似是
$$
2Lkd = 2 \times 12 \times 4 \times 768 = 73728
$$
这里只训练七万多参数，而不是去动上亿参数的主体模型。

---

## 核心机制与推导

先看标准注意力。注意力就是让 query 去匹配 key，再用匹配权重对 value 做加权求和。白话说，query 在问“我现在该关注谁”，key 决定“我是什么信息”，value 决定“我把什么内容给你”。

标准公式是：
$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

Prefix Tuning 不改这个公式，而是扩展其中的 $K$ 和 $V$。对第 $\ell$ 层，定义两组前缀矩阵：
$$
P_K^{(\ell)}, P_V^{(\ell)} \in \mathbb{R}^{k \times d}
$$

然后把它们接到原始键值前面：
$$
K'=[P_K^{(\ell)};K^{(\ell)}], \quad V'=[P_V^{(\ell)};V^{(\ell)}]
$$

于是新的注意力就是：
$$
\text{softmax}\left(\frac{Q{K'}^\top}{\sqrt{d}}\right)V'
$$

注意，query 没变，真实 token 也没变，变的是“候选可读信息”被扩展了。这样真实 token 在做注意力时，就会把 prefix 当成额外的上下文来源。

为什么很多实现会再加一个小 MLP 重参数化？原因是直接把每层的 $P_K^{(\ell)}, P_V^{(\ell)}$ 当独立参数去学，容易初始化敏感、训练不稳。重参数化的思路是：先训练一个更短、更低维的 latent prefix，再通过小 MLP 映射成各层需要的 K/V。这里的 latent 可以理解为“更紧凑的任务编码”。

一个常见形式是：
$$
Z \in \mathbb{R}^{k \times d_z}, \quad \text{MLP}(Z)\rightarrow \mathbb{R}^{L \times 2 \times k \times d}
$$

其中 $2$ 表示每层都要产出一份 K 和一份 V。这样做有两个直接收益：

1. 降低训练不稳定性，因为真正直接优化的是更小的 latent。
2. 引入层间共享结构，因为不同层的 prefix 来自同一个生成器。

玩具例子可以这样理解。原来某个 token 只会在“输入句子里的词”之间分配注意力；加了 prefix 后，它还会分一部分权重给“任务记忆槽”。如果任务是“生成摘要而不是翻译”，这段记忆槽就会把注意力轻轻推向“压缩信息、保留主旨”的方向。

---

## 代码实现

下面用一个可运行的 Python 玩具实现演示“把 prefix 拼到 K/V 前面”这件事。它不是完整训练代码，但逻辑和真实实现一致。

```python
import math
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def attention(q, k, v):
    d = q.shape[-1]
    scores = q @ k.T / math.sqrt(d)
    weights = softmax(scores, axis=-1)
    return weights @ v, weights

# 一个玩具序列：2 个真实 token，维度 2
Q = np.array([[1.0, 0.0],
              [0.0, 1.0]])

K_real = np.array([[1.0, 0.0],
                   [0.0, 1.0]])
V_real = np.array([[10.0, 0.0],
                   [0.0, 10.0]])

# 2 个 prefix token，作为“任务记忆”
P_K = np.array([[1.0, 1.0],
                [2.0, 0.0]])
P_V = np.array([[5.0, 5.0],
                [20.0, 0.0]])

K_ext = np.concatenate([P_K, K_real], axis=0)
V_ext = np.concatenate([P_V, V_real], axis=0)

out_no_prefix, w_no_prefix = attention(Q, K_real, V_real)
out_with_prefix, w_with_prefix = attention(Q, K_ext, V_ext)

# 有 prefix 时，注意力长度应从 2 变成 4
assert w_no_prefix.shape == (2, 2)
assert w_with_prefix.shape == (2, 4)

# 输出应该发生变化，说明 query 真的读到了 prefix
assert not np.allclose(out_no_prefix, out_with_prefix)

print("without prefix:\n", out_no_prefix)
print("with prefix:\n", out_with_prefix)
```

如果换成简化的 PyTorch 伪代码，核心就是下面几步：

```python
# 1. 冻结原模型
for p in model.parameters():
    p.requires_grad = False

# 2. 训练前缀参数或前缀生成器
prefix_encoder = PrefixMLP(...)

def attention_block(x, layer_id):
    q = Wq(x)
    k = Wk(x)
    v = Wv(x)

    prefix_k, prefix_v = prefix_encoder(layer_id)   # [k, d], [k, d]
    prefix_k = prefix_k.unsqueeze(0).expand(x.size(0), -1, -1)
    prefix_v = prefix_v.unsqueeze(0).expand(x.size(0), -1, -1)

    k = torch.cat([prefix_k, k], dim=1)
    v = torch.cat([prefix_v, v], dim=1)

    return attention(q, k, v)

optimizer = Adam(prefix_encoder.parameters(), lr=1e-3)
```

真实工程例子：一个金融团队要给多个品牌生成摘要，但不允许每个品牌都维护一套完整微调模型。做法可以是共享一个冻结主模型，为品牌 A、品牌 B、品牌 C 分别保存各自的 prefix 参数。推理时只切换 prefix 文件，不切换主模型权重。这样存储更省，审计边界也更清楚，因为“品牌行为差异”被集中放在前缀里，而不是分散到整套权重改动中。

---

## 工程权衡与常见坑

Prefix Tuning 的优势很集中，但坑也很集中。

第一类坑是长度选择。prefix 太短时，模型读到的任务条件过少，表现为指标上不去，或者不同随机种子差异很大。prefix 太长时，训练看起来可能更好，但线上时延和显存会明显上升。因为对每一层、每一个头、每一步生成，都多了一段要参与注意力的 K/V。

第二类坑是初始化与收敛。很多失败实验不是方法本身失效，而是直接学习每层前缀时过于敏感。小 MLP 重参数化通常能改善这一点，因为它相当于给前缀加了一个结构先验，让优化空间更平滑。

第三类坑是把“参数少”误当成“效果总更稳”。Prefix Tuning 对任务类型有偏好。它比较擅长把任务条件注入注意力上下文，但如果任务需要系统性改写很多内部变换，LoRA 这类改线性层的办法可能更有效。

一个实用调参流程通常是：

1. 先固定主模型，验证冻结是否正确。
2. 从中等 prefix 长度开始，比如 8、16、32。
3. 优先比较“有无重参数化 MLP”的差异。
4. 记录训练 loss、验证集指标、推理时延、KV cache 占用。
5. 如果效果不足，再加长 prefix；如果时延超标，再缩短 prefix 或减少适配层数。

下面是一个常见经验表：

| 现象 | 常见原因 | 处理办法 |
|---|---|---|
| 训练几乎不收敛 | prefix 初始化差，或长度太短 | 加 MLP 重参数化，增大 $k$ |
| 验证集提升有限 | 任务需要更强表达力 | 增大 $k$，或改用 LoRA |
| 线上时延明显上升 | prefix 太长 | 缩短 $k$，只在部分层加 prefix |
| 多任务切换混乱 | prefix 管理不规范 | 按任务版本化保存 prefix 参数 |
| 小数据集过拟合 | prefix/MLP 过强 | 加 dropout、早停、减小 MLP |

---

## 替代方案与适用边界

Prefix Tuning 不是唯一的 PEFT 方法。PEFT 是 Parameter-Efficient Fine-Tuning，白话就是“用少量附加参数做微调”。

和 LoRA 比，Prefix Tuning 的差异不在“参数量一定更小”，而在“加参数的位置不同”。LoRA 改的是线性层权重增量，Prefix 改的是注意力能读到的上下文。前者像改电路里的变换器，后者像在输入旁边放一本额外说明书。

和 Prompt Tuning 比，Prefix Tuning 更深入。Prompt Tuning 常常只在输入层前面加软提示；Prefix Tuning 是在每一层都注入前缀，所以控制更强，但实现也更复杂。

和 Adapter 比，Prefix Tuning 不需要在层内插入新的残差模块，通常更适合“严格不改原主干结构”的场景；但 Adapter 有时更稳定，也更容易和现有训练框架对接。

| 方法 | 参数加在哪里 | 是否改原权重路径 | 推理额外开销 | 适用场景 |
|---|---|---|---|---|
| Prompt Tuning | 输入嵌入前 | 否 | 低 | 输入层提示足够的简单任务 |
| Prefix Tuning | 每层 attention 的 K/V | 否 | 中到高 | 需要层级条件控制、多任务切换 |
| LoRA | 线性投影矩阵 | 是，以增量方式 | 通常较低 | 通用微调，生态成熟 |
| Adapter | 层内新增小模块 | 是，增加旁路 | 中等 | 需要更强表达力、模块化部署 |

适用边界可以总结为两句。第一，如果你的限制是“主模型不能改，但可以外挂少量任务参数”，Prefix Tuning 很合适。第二，如果你的限制是“线上时延极敏感”，Prefix 长度带来的线性开销就必须严格评估，不能只看训练参数量。

---

## 参考资料

1. EmergentMind, *Prefix-Tuning in Transformer Models*：适合看定义、核心公式、参数量推导，以及 Prefix Tuning 相对全量微调的基本定位。  
2. Avichala GenAI, *What is prefix tuning*：适合看工程视角的解释，包括前缀长度 $k$ 的调优经验、重参数化的作用和多任务部署案例。  
3. Hailey Schoelkopf, *Prefix Linear Attention Can Outspeed Causal Attention*：适合作为拓展阅读，理解 prefix 思想在 linear attention 等变体中的进一步演化。
