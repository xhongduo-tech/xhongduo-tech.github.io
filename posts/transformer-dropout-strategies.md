## 核心结论

Transformer 里的 dropout 不是一个单点技巧，而是三类位置不同、目标相近的正则化手段。这里的“正则化”，可以先理解成：训练时主动加入约束，降低模型对训练集偶然模式的依赖，让它在未见过的数据上更稳。

第一类是 **attention dropout**。它作用在注意力权重上，也就是“当前 token 要看其他 token 多少”的那张分配表。具体做法是在 `softmax` 之后，对部分注意力权重随机置零，再按 $1/(1-p)$ 缩放。它约束的不是 token 本身，而是 token 和 token 之间的连接，目标是防止少数连接长期垄断注意力。

第二类是 **layer dropout**，在标准 Transformer 语境里更准确地说通常是 **residual dropout**。它作用在子层输出上，子层既可能是多头注意力，也可能是前馈网络。做法是在子层输出、加回残差之前做 dropout。它抑制的是某一层输出过强、后续层过分依赖这一路径的问题，让残差分支继续承担稳定的信息通道。

第三类是 **embedding dropout**。它作用在输入表示上。embedding 的作用是把离散 token、位置或结构化特征映射成连续向量。随机屏蔽其中一部分维度后，后续网络就不能长期依赖某几个固定维度，而会更多利用上下文、其他维度和层内组合特征。

三者共同遵循同一个训练期公式：

$$
m \sim \text{Bernoulli}(1-p), \quad y=\frac{m\odot x}{1-p}
$$

其中：

| 符号 | 含义 |
|---|---|
| $x$ | dropout 之前的输入张量 |
| $p$ | dropout 率，即被置零的概率 |
| $m$ | 由 Bernoulli 分布采样得到的掩码 |
| $\odot$ | 逐元素相乘 |
| $y$ | dropout 之后的输出 |

这样处理的关键性质是训练时输出期望不变：

$$
\mathbb{E}[y] = x
$$

推理阶段不再随机屏蔽，因此 dropout 必须关闭。否则同一输入每次前向都会得到不同结果，评估和上线都不可信。

工程上常见的经验值是：预训练里常用 `0.1` 左右；数据量较小、标签较贵或更容易过拟合的微调任务，经常把部分 dropout 提到 `0.2` 到 `0.3`。但这不是越大越好。dropout 太高会把有效信号打散，训练变慢，甚至直接欠拟合。

---

## 问题定义与边界

本文讨论的是标准 Transformer 架构里三类常见 dropout 的作用、位置、实现方式和调参边界，不讨论 RNN 中的 variational dropout，也不讨论 Mixture-of-Experts 里的路由噪声、专家负载均衡噪声或采样式稀疏激活策略。

先把边界说清楚。

| 类型 | 典型位置 | 常见取值 | 直接作用对象 | 主要目标 |
|---|---|---:|---|---|
| attention dropout | `softmax(QK^T/\sqrt{d_k})` 之后 | 0.0-0.1，部分任务到 0.2 | 注意力权重矩阵 | 防止少数 token 连接长期垄断 |
| layer/residual dropout | attention 或 FFN 子层输出上 | 0.1-0.3 | 子层激活 | 减少层间共适应，增强泛化 |
| embedding dropout | token/position/feature embedding 上 | 0.1-0.2，少量数据可更高 | 输入向量维度 | 降低对固定维度的依赖 |

所谓 **共适应**，可以先理解成：若干神经元、若干维度或若干层之间形成了过强的固定搭配。训练时它们总是一起工作，所以模型学会的不是稳健规则，而是“这一组信号一旦同时出现，就直接给答案”。一旦测试集稍微变一下，这种绑定关系就容易失效。

这个问题在 Transformer 里尤其常见，因为它的表达能力很强，层内连接又非常密。模型一旦发现某个 token、某个头、某个 embedding 维度对当前训练集特别有用，就会快速把权重集中到这条路径上。dropout 的作用，就是定期把这些路径打断，让模型学习备用路线。

一个玩具例子：做情感分类时，如果训练集中“excellent”几乎总出现在正样本里，模型可能让某个注意力头长期盯住这个词，甚至只要看到它就倾向于输出正类。如果不开 attention dropout，这条连接会越来越稳，模型对“not excellent”“hardly excellent”这类反转上下文就学得不够。加了 attention dropout 后，这条高权重连接有时会被屏蔽，模型就被迫学习否定词、依存结构、句子位置等替代证据。

一个更接近真实工程的例子：小数据医疗时序预测里，样本可能只有几千条，但输入特征很多，特征之间还存在缺失和噪声。如果微调时几乎不开 residual dropout，模型容易记住训练集里少数高频模式，表现为训练损失持续下降、验证集上下波动很大。这时把 encoder block 的 residual dropout 从 `0.1` 提到 `0.2`，再配合 early stopping，往往比一味加深网络更有效。

这里有两个常被忽略的边界条件。

第一，dropout 只在训练时生效。推理、验证、导出 ONNX、离线评测、线上服务都应关闭 dropout。忘记切到 `eval`，通常会直接造成同一输入多次预测不一致。

第二，dropout 不能脱离训练配置单独看。它和以下因素强相关：

| 因素 | 对 dropout 决策的影响 |
|---|---|
| 数据规模 | 数据越大，dropout 往往可以更保守 |
| 标签噪声 | 标签越脏，适度 dropout 越常见 |
| 模型容量 | 模型越大、越容易记忆训练集，越可能需要更强正则 |
| batch size | 大 batch 往往训练更稳定，可承受不同正则组合 |
| weight decay | 与 dropout 共同影响泛化，不能只调一个 |
| 学习率与 warm-up | dropout 提高后，常需要重新观察收敛速度 |
| 任务头大小 | 小样本微调时，任务头往往比 backbone 更需要 dropout |

因此，本文的讨论范围是：标准 Transformer 里的 dropout 机制本身，以及它在预训练和微调中的典型工程用法，而不是把它当作一个脱离上下文的万能旋钮。

---

## 核心机制与推导

### 1. attention dropout 为什么放在 softmax 后

标准自注意力可以写成：

$$
A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right), \quad O = AV
$$

其中：

| 符号 | 含义 |
|---|---|
| $Q$ | query 矩阵 |
| $K$ | key 矩阵 |
| $V$ | value 矩阵 |
| $M$ | mask，通常用于 padding mask 或 causal mask |
| $A$ | 注意力权重矩阵 |
| $O$ | 注意力输出 |

attention dropout 并不是直接对 $Q$、$K$ 或 $V$ 做丢弃，而是对已经归一化后的注意力权重 $A$ 做随机屏蔽：

$$
\tilde{A} = \frac{m \odot A}{1-p}
$$

于是输出变成：

$$
O = \tilde{A}V
$$

这一步的含义很具体：丢掉的不是 token，而是“query 到某个 key 的连接”。也就是说，模型不是被迫忘掉一个词，而是被迫不能每次都通过同一条注意力边拿信息。

为什么放在 `softmax` 之后？原因有两个。

第一，`softmax` 之后的 $A$ 已经是概率分布，语义明确。每个元素都表示一条连接的相对权重。对它做 dropout，等价于直接正则化连接选择。

第二，`softmax` 之前是 logits。logits 会进入指数函数：

$$
\text{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

如果在 logits 上直接施加随机置零或随机缩放，经过指数变换后，微小扰动可能被成倍放大，整个分布温度都会变，训练噪声更难控制。换句话说，`softmax` 前的 dropout 往往不是“删一条边”，而是“重写整个分布形状”。

看一个具体例子。设某个 query 对 4 个 token 的注意力权重是：

$$
A=[0.5,\,0.2,\,0.2,\,0.1]
$$

若 attention dropout 率为 $p=0.1$，某次训练中最后一个位置被屏蔽，那么：

$$
m=[1,1,1,0], \quad \tilde{A}=\frac{m\odot A}{0.9}
$$

得到：

$$
\tilde{A} \approx [0.556,\,0.222,\,0.222,\,0.0]
$$

这里要注意两点。

第一，这一步通常**不是重新做一次 softmax**。  
第二，缩放的目的是保持期望尺度不变，而不是保持每一行严格和为 1。

从期望上看：

$$
\mathbb{E}[\tilde{A}_{ij}] = A_{ij}
$$

因此，虽然单次前向时权重分布被扰动，但平均意义上它仍然围绕原来的注意模式波动。长期训练后，模型更容易学到分散、稳健、不过度依赖单条边的注意结构。

新手容易混淆的一点是：attention dropout 不等于“鼓励注意力更均匀”。它真正做的是让高权重连接不那么稳定。最终学到的注意力可能仍然尖锐，但会是对扰动更稳的尖锐，而不是脆弱的单点依赖。

### 2. layer dropout 为什么通常配残差

标准 Transformer block 的一个简化写法是：

$$
x' = x + \text{Dropout}(\text{MHA}(x))
$$

$$
y = x' + \text{Dropout}(\text{FFN}(x'))
$$

如果写得更细一点，可以表示为：

$$
h_{\text{attn}} = \text{MHA}(x), \quad x' = x + D(h_{\text{attn}})
$$

$$
h_{\text{ffn}} = \text{FFN}(x'), \quad y = x' + D(h_{\text{ffn}})
$$

其中 $D(\cdot)$ 表示 dropout 运算。

它的核心不是“把这一层去掉”，而是“把这一层输出里的部分激活随机置零，再把剩余激活缩放后加回主干”。因此它约束的是子层输出的强度和稳定性，而不是网络深度本身。

残差连接在这里很关键。可以把它理解成一条稳定的高速通道：即使当前子层输出被打掉一部分，原始表示 $x$ 仍然能继续传下去。于是网络训练时不会因为随机屏蔽而完全失去信息主干。

如果没有残差，dropout 的副作用会更大。因为一旦某层输出被大面积屏蔽，后续层拿到的输入可能立刻变得很弱。残差的存在让“正则化”变成“扰动一条支路”，而不是“切断整条信息流”。

这类 dropout 抑制的是层间共适应。典型症状包括：

| 症状 | 说明 |
|---|---|
| 某个 block 输出幅度越来越大 | 后续层主要依赖该层，结构变脆 |
| 验证集对层数或种子很敏感 | 表示分工不稳定 |
| 训练集收敛很快，验证集提升有限 | 深层表征在记忆训练模式 |
| 最后几层权重更新异常活跃 | 任务头或高层过拟合明显 |

新手常见误解是把 residual dropout 和 **stochastic depth** 混为一谈。两者不是一回事：

| 方法 | 丢弃对象 | 行为 |
|---|---|---|
| residual dropout | 子层输出中的部分激活 | 整层仍然执行，只是输出被部分屏蔽 |
| stochastic depth / drop path | 整个残差分支或整层路径 | 某次前向可能直接跳过整个分支 |

因此，如果你的代码里是：

```python
x = x + dropout(sublayer(x))
```

这是 residual dropout。  
如果你的代码里是按概率直接返回 `x`，完全不走 `sublayer(x)`，那更接近 stochastic depth。

### 3. embedding dropout 为什么对小数据任务常有效

设输入 embedding 为 $e \in \mathbb{R}^d$。embedding dropout 可以写成：

$$
\tilde{e} = \frac{m \odot e}{1-p}
$$

它的含义是：每次训练随机关闭 embedding 的一部分维度，让模型无法把决策建立在少数固定维度上。

从表示学习角度看，这相当于每次训练都在不同的特征子空间中做前向传播。模型需要在“部分特征缺失”的情况下仍然输出合理结果，因此更容易学到冗余表示和替代性证据。

看一个 6 维例子。原始 embedding 为：

$$
e=[0.8,-0.2,0.5,0.0,1.2,-0.6]
$$

若 $p=0.2$，某次训练的掩码为：

$$
m=[1,1,1,1,0,1]
$$

则：

$$
\tilde{e}=\frac{m\odot e}{0.8}=[1.0,-0.25,0.625,0.0,0.0,-0.75]
$$

可以看到，第 5 个维度被关闭，其余维度按 $1/0.8=1.25$ 缩放。

对新手来说，embedding dropout 最容易理解成“输入层的特征缺失模拟”。这种理解不完全严格，但在工程直觉上足够有用。尤其是在以下场景中，它经常有效：

| 场景 | 为什么有效 |
|---|---|
| 小样本文本分类 | 避免模型死记少数高频词的固定表示 |
| 多模态/多特征融合 | 防止某一路特征长期压制其他特征 |
| 传感器数据建模 | 让模型在局部特征缺失时仍能预测 |
| 结构化特征 embedding | 降低对个别离散字段的过强依赖 |

真实工程里，多特征 Transformer 常把不同来源特征先编码成 embedding，再拼接或对齐后输入 encoder。如果某一类特征在训练集上特别强，比如一个传感器分量、某个离散 ID、某个高频关键词，模型就会过度依赖它。embedding dropout 的作用，就是让网络周期性地失去这条捷径，逼它利用上下文和其他特征补足信息。

但它也有边界。如果输入本身已经很稀疏、很弱，embedding dropout 过高会直接伤害可学习性。一个常见现象是：模型一开始就学不动，训练和验证都上不去。这时问题往往不是“正则化不够”，而是“输入信号已经被打没了”。

---

## 代码实现

下面给出一个最小可运行的 Python 例子，演示三类 dropout 的位置和行为。它只依赖 `numpy`，不依赖深度学习框架，可以直接运行。

```python
import numpy as np


def dropout(x, p, training, rng):
    """
    Inverted dropout:
    - training=False: identity
    - training=True : randomly zero out entries and scale by 1 / (1 - p)
    """
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}")

    x = np.asarray(x, dtype=np.float64)

    if (not training) or p == 0.0:
        return x.copy()

    keep_prob = 1.0 - p
    mask = (rng.random(x.shape) < keep_prob).astype(np.float64)
    return x * mask / keep_prob


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention_with_dropout(scores, values, p_attn, training, rng):
    """
    scores: [T, T]
    values: [T, D]
    """
    attn = softmax(scores, axis=-1)
    attn_drop = dropout(attn, p=p_attn, training=training, rng=rng)
    out = attn_drop @ values
    return attn, attn_drop, out


def relu(x):
    return np.maximum(x, 0.0)


def transformer_block_toy(
    x,
    w_q,
    w_k,
    w_v,
    w_o,
    w_ffn1,
    w_ffn2,
    p_attn,
    p_layer,
    p_embed,
    training,
    seed=0,
):
    """
    A toy Transformer block with:
    1) embedding dropout
    2) attention dropout
    3) residual dropout after attention and FFN
    """
    rng = np.random.default_rng(seed)

    # 1) embedding dropout
    x_in = dropout(x, p=p_embed, training=training, rng=rng)

    # single-head self-attention
    q = x_in @ w_q
    k = x_in @ w_k
    v = x_in @ w_v

    d_k = q.shape[-1]
    scores = (q @ k.T) / np.sqrt(d_k)

    # 2) attention dropout
    attn, attn_drop, attn_out = attention_with_dropout(
        scores=scores,
        values=v,
        p_attn=p_attn,
        training=training,
        rng=rng,
    )

    attn_proj = attn_out @ w_o

    # 3) residual dropout after attention sub-layer
    h = x_in + dropout(attn_proj, p=p_layer, training=training, rng=rng)

    # FFN: D -> H -> D
    ffn_hidden = relu(h @ w_ffn1)
    ffn_out = ffn_hidden @ w_ffn2

    # residual dropout after FFN sub-layer
    y = h + dropout(ffn_out, p=p_layer, training=training, rng=rng)
    return {
        "x_after_embed_dropout": x_in,
        "attn": attn,
        "attn_after_dropout": attn_drop,
        "output": y,
    }


def main():
    # toy input: 3 tokens, hidden size 4
    x = np.array(
        [
            [1.0, 0.0, 0.5, -0.5],
            [0.2, 0.8, -0.3, 0.1],
            [0.0, -0.4, 1.2, 0.7],
        ],
        dtype=np.float64,
    )

    # projections: D=4
    w_q = np.array(
        [
            [0.8, 0.1, 0.0, 0.0],
            [0.0, 0.7, 0.2, 0.1],
            [0.1, 0.0, 0.9, 0.1],
            [0.0, 0.2, 0.1, 0.8],
        ],
        dtype=np.float64,
    )
    w_k = np.array(
        [
            [0.7, 0.0, 0.1, 0.1],
            [0.1, 0.8, 0.0, 0.0],
            [0.0, 0.2, 0.9, 0.1],
            [0.1, 0.0, 0.2, 0.7],
        ],
        dtype=np.float64,
    )
    w_v = np.array(
        [
            [0.9, 0.0, 0.1, 0.0],
            [0.0, 0.8, 0.1, 0.1],
            [0.2, 0.0, 0.7, 0.2],
            [0.0, 0.1, 0.0, 0.9],
        ],
        dtype=np.float64,
    )
    w_o = np.eye(4, dtype=np.float64)

    # FFN: 4 -> 6 -> 4
    w_ffn1 = np.array(
        [
            [0.5, -0.2, 0.1, 0.0, 0.3, -0.1],
            [0.0, 0.3, 0.4, -0.1, 0.2, 0.0],
            [0.2, 0.0, 0.6, 0.2, -0.2, 0.1],
            [-0.3, 0.1, 0.0, 0.5, 0.1, 0.4],
        ],
        dtype=np.float64,
    )
    w_ffn2 = np.array(
        [
            [0.4, 0.0, 0.1, -0.2],
            [0.0, 0.5, 0.2, 0.0],
            [0.2, 0.1, 0.6, 0.1],
            [-0.1, 0.2, 0.0, 0.4],
            [0.3, -0.2, 0.1, 0.0],
            [0.0, 0.1, -0.1, 0.5],
        ],
        dtype=np.float64,
    )

    # eval mode: dropout disabled
    eval_result = transformer_block_toy(
        x=x,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        w_ffn1=w_ffn1,
        w_ffn2=w_ffn2,
        p_attn=0.1,
        p_layer=0.2,
        p_embed=0.2,
        training=False,
        seed=42,
    )

    # train mode: dropout enabled
    train_result = transformer_block_toy(
        x=x,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        w_ffn1=w_ffn1,
        w_ffn2=w_ffn2,
        p_attn=0.1,
        p_layer=0.2,
        p_embed=0.2,
        training=True,
        seed=42,
    )

    # In eval mode, attention dropout should do nothing
    assert np.allclose(
        eval_result["attn"],
        eval_result["attn_after_dropout"],
        atol=1e-12,
    )

    # In train mode, attention dropout usually changes the attention weights
    assert not np.allclose(
        train_result["attn"],
        train_result["attn_after_dropout"],
        atol=1e-12,
    )

    # The raw softmax attention is still row-normalized before dropout
    row_sums = eval_result["attn"].sum(axis=-1)
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-12)

    print("eval output shape :", eval_result["output"].shape)
    print("train output shape:", train_result["output"].shape)
    print()
    print("eval attention:")
    print(np.round(eval_result["attn"], 4))
    print()
    print("train attention after dropout:")
    print(np.round(train_result["attn_after_dropout"], 4))
    print()
    print("eval output:")
    print(np.round(eval_result["output"], 4))
    print()
    print("train output:")
    print(np.round(train_result["output"], 4))


if __name__ == "__main__":
    main()
```

如果你运行这个脚本，会看到几件事。

第一，`training=False` 时，attention dropout 之后的矩阵和原始 attention 矩阵一致。  
第二，`training=True` 时，`attn_after_dropout` 中会有部分位置被置零，且其他位置被缩放。  
第三，最终输出 `output` 在训练态和推理态会不同，这正是 dropout 引入的训练期随机性。

把这个最小例子映射到 PyTorch，关键实现点是位置不要错：

```python
import torch
import torch.nn.functional as F

def forward(self, x, attn_scores, values):
    # attention dropout: after softmax
    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_probs = F.dropout(attn_probs, p=self.attn_p, training=self.training)

    attn_out = attn_probs @ values
    attn_out = self.attn_proj(attn_out)

    # residual dropout after attention sub-layer
    x = x + F.dropout(attn_out, p=self.hidden_p, training=self.training)

    # FFN
    ffn_out = self.ffn(x)

    # residual dropout after FFN sub-layer
    x = x + F.dropout(ffn_out, p=self.hidden_p, training=self.training)
    return x
```

embedding dropout 则通常写成：

```python
emb = self.embedding(input_ids)
emb = F.dropout(emb, p=self.embed_p, training=self.training)
```

如果是 Hugging Face 一类的实现，常见配置项通常包括：

| 配置项 | 含义 |
|---|---|
| `attention_probs_dropout_prob` | 注意力概率矩阵上的 dropout |
| `hidden_dropout_prob` | 隐状态或子层输出上的 dropout |
| 某些模型自己的 `embd_pdrop` / `dropout` | embedding 或 block 内部统一 dropout |

这里 `self.training` 是总开关。`model.train()` 时启用，`model.eval()` 时关闭。很多线上事故不是结构写错，而是评估前没切模式。症状往往表现为：同一 batch 多跑几次结果不一样，注意力图也在抖，误以为随机种子没设好，实际只是 dropout 还开着。

真实工程里，微调预训练模型时更稳妥的做法通常不是“把所有 dropout 一起拉高”，而是分层处理：

| 部位 | 更常见的策略 |
|---|---|
| backbone 早期层 | 先保持默认值，例如 `0.1` |
| backbone 后几层 | 视验证集 gap 再微调 |
| 分类头/回归头 | 小数据任务中优先增加 dropout |
| embedding 层 | 只在输入特征确实冗余时再提高 |

原因很简单：预训练 backbone 往往已经在大规模数据上学到了相对稳的表征，真正容易过拟合的部分常常是任务头和最后几层，而不是整个网络的所有位置。

---

## 工程权衡与常见坑

dropout 的价值在于改善泛化，但代价是主动给训练过程加噪。因此它一定不是“免费收益”，而是一个明确的权衡。

第一类权衡是“泛化 vs 收敛速度”。dropout 越大，单步训练使用的有效子网络越随机，训练损失下降通常更慢。对小数据任务，这种代价往往值得；对超大规模预训练，数据本身已经足够多，过高 dropout 反而只会拖慢优化。

第二类权衡是“稳定性 vs 表达能力”。适度 dropout 能防止模型把表达能力集中到少数路径上；但太强的 dropout 会直接削弱模型表达，导致头部分工难形成、浅层表示学不稳、最终性能下降。

把三类 dropout 的风险拆开看，会更清楚：

| 类型 | 开得太小的问题 | 开得太大的问题 |
|---|---|---|
| attention dropout | 容易出现单条连接垄断 | 注意力图过散，头分工不稳定 |
| residual dropout | 层间共适应更强 | 深层信息整合变慢，收敛变差 |
| embedding dropout | 输入特征依赖过强 | 早期表示被打散，模型一开始就学不动 |

下面把常见坑列清楚。

| 坑 | 影响 | 规避方式 |
|---|---|---|
| 把 attention dropout 放在 `softmax` 前 | 改变 logits 温度，分布失真 | 放在 `softmax` 后 |
| 推理前忘记 `model.eval()` | 同一输入多次输出不同，验证波动大 | 评估、导出、上线前统一切 `eval` |
| dropout 率一律设很高 | 欠拟合、收敛慢、注意力过散 | 从 `0.1` 起调，小数据再逐步上调 |
| 只调 dropout，不配学习率/weight decay | 指标不稳，误判超参效果 | 联合调参，至少固定其他关键项 |
| embedding dropout 过强 | 早期表示被打散，浅层更难学 | 对小模型或弱特征任务保守设置 |
| 把 residual dropout 和整层跳过混淆 | 设计和实现都可能出错 | 区分 residual dropout 与 stochastic depth |

一个典型新手坑是：训练结束后直接在 notebook 里验证，却忘了把模型从 `train` 切到 `eval`。这时你会看到：

| 现象 | 实际原因 |
|---|---|
| 同一输入多次预测结果不同 | dropout 仍在生效 |
| attention map 每次都不一样 | attention dropout 未关闭 |
| loss 有明显抖动 | 前向图仍有随机掩码 |
| 以为随机种子没固定 | 其实是模式切换错误 |

另一个真实工程坑出现在小样本微调任务里，比如医学文本分类、材料性质预测、日志异常检测、工业时序告警。这些任务的共同特点是：样本少、标签贵、验证集方差大。如果沿用预训练默认 `0.1`，模型可能仍然过拟合；但如果把 `attention`、`hidden`、`embedding` 三类 dropout 一口气全提到 `0.3`，又可能训练不起来。

更稳的策略通常是：

1. 先保留 backbone 默认值，例如 `0.1`。
2. 优先增加任务头、最后几层或 classifier 前的 dropout。
3. 同时观察训练集与验证集的 gap，而不是只看单次最好分数。
4. 把 dropout 和 warm-up、weight decay、early stopping 一起调。
5. 若训练和验证都上不去，先回头检查学习率、数据质量和模型容量，不要先假设是 dropout 太小。

还有一个容易忽略的点：dropout 不是独立按钮，而是训练噪声预算的一部分。如果你已经用了很强的数据增强、较大的 weight decay、较长的 warm-up，再继续提高 dropout，收益未必叠加，反而可能重复施加约束。

可以用一个简单判断框架：

| 现象 | 更可能的动作 |
|---|---|
| 训练集很好，验证集差 | 适度增强 dropout 或其他正则 |
| 训练集和验证集都差 | 优先检查学习率、数据、模型容量 |
| 训练极慢，验证也没改善 | dropout 可能过高 |
| 指标对随机种子极敏感 | 查看是否过拟合或评估模式错误 |

---

## 替代方案与适用边界

dropout 不是唯一的正则化手段。在一些场景里，替代方法更合适，或者至少应当组合使用。

| 方案 | 主要思路 | 何时优先 |
|---|---|---|
| stochastic depth / drop path | 随机跳过整层或整条路径 | 深层网络很深，想增强梯度传播与结构正则化时 |
| weight decay | 惩罚过大的参数 | 过拟合明显但不想引入前向随机性时 |
| label smoothing | 软化监督目标 | 分类任务置信度过高、校准差时 |
| 数据增强 | 增加输入多样性 | 文本替换、mask、时序扰动可行时 |
| LayerNorm + 合理初始化 | 稳定激活尺度 | embedding dropout 过强会伤输入信号时 |
| early stopping | 在过拟合前停止 | 小数据微调、验证集较可靠时 |

这些方法和 dropout 的差别，不在于“谁更高级”，而在于它们约束的位置不同。

| 方法 | 约束对象 |
|---|---|
| dropout | 前向激活或连接 |
| weight decay | 参数大小 |
| label smoothing | 监督目标分布 |
| 数据增强 | 输入分布 |
| stochastic depth | 网络路径结构 |
| early stopping | 训练时长 |

因此，是否使用 dropout，取决于你要解决的问题到底在哪一层。

如果模型很深，单纯 residual dropout 有时不如 stochastic depth。因为后者直接在路径级别做正则，能更直接处理深层网络中的梯度传播和结构冗余。

如果输入本身已经很稀疏，embedding dropout 要格外谨慎。比如多特征 Transformer 里，某些离散特征本来就弱、某些数值通道本来就缺失严重，这时再做较大 embedding dropout，模型可能连基础信号都拿不到。更常见的替代是保留较小的 embedding dropout，用 LayerNorm、feature scaling、稳健初始化或特征筛选控制训练稳定性。

如果是超大规模预训练，数据量已经非常大，dropout 的边际收益通常会下降。原因不是 dropout 原理失效，而是海量数据本身已经提供了很强的正则化。此时，学习率曲线、batch 规模、优化器配置、数据质量和 token 多样性，往往比把 dropout 从 `0.1` 调到 `0.2` 更重要。

可以把适用边界总结成一个更直接的判断表：

| 场景 | dropout 是否优先 |
|---|---|
| 小数据微调，训练集明显优于验证集 | 通常值得优先尝试 |
| 深层模型训练不稳 | 可能先看 stochastic depth |
| 超大规模预训练 | 通常不是最优先调参项 |
| 输入特征很弱、很稀疏 | embedding dropout 要保守 |
| 分类置信度过高但准确率未必差 | 可优先看 label smoothing |
| 不想引入前向随机性 | 可先看 weight decay |

一个真实工程判断标准是：如果训练集 loss 很低、验证集表现差，而且 attention 图明显集中到少数 token 或少数特征，那么 dropout 往往值得试；如果训练集和验证集都上不去，就应先检查模型容量、学习率、数据质量和预处理流程，而不是先怪 dropout 不够大。

换句话说，dropout 更像是“防止模型走捷径”的工具，而不是“把模型性能硬拉上去”的工具。模型连主路都没学会之前，先去封捷径，通常不会有好结果。

---

## 参考资料

下面按“原始论文、结构讲解、工程实现、训练综述、应用案例”几个维度整理资料。前几项更适合理解机制，后几项更适合解决工程问题。

| 资料 | 角度 | 适合阅读的章节 |
|---|---|---|
| *Attention Is All You Need* | Transformer 原始结构、残差连接、dropout 基本位置 | 问题定义、核心机制 |
| PyTorch `nn.Dropout` 文档 | inverted dropout 的实现语义与 train/eval 行为 | 核心结论、代码实现 |
| PyTorch `nn.MultiheadAttention` 文档 | attention dropout 在框架中的参数位置 | 核心机制、代码实现 |
| Hugging Face Transformers 文档与模型配置 | `attention_probs_dropout_prob`、`hidden_dropout_prob` 等配置项 | 代码实现、常见坑 |
| 结构图类 Transformer Primer | 适合理清 block 内张量流向 | 问题定义、机制推导 |
| 训练 Transformer 的综述文章 | 优化、归一化、正则化之间的关系 | 工程权衡 |
| 多特征 Transformer 的应用论文 | 小样本、多特征融合任务中 dropout 的工程使用 | 问题定义、工程例子 |
| stochastic depth / drop path 相关论文 | 对比 residual dropout 的替代方案 | 替代方案与适用边界 |

如果按阅读顺序给建议，可以这样分。

对刚入门的读者，先看两类资料：

1. 原始 Transformer 结构图和 block 连接图，先搞清楚 attention dropout 和 residual dropout 分别落在哪个张量上。
2. PyTorch 或 Hugging Face 的实现文档，确认 `train/eval` 切换和配置项含义。

对已经在做微调的读者，更值得优先看的反而不是概念文章，而是框架实现和 issue。因为很多“模型训练不稳定”“评估结果随机波动”“注意力图不一致”的问题，本质上不是理论没懂，而是 dropout 放错位置、模式没切对、或者超参数联动没有处理。

如果要给一条最短的实践路线，可以记住下面这张表：

| 问题 | 先看什么 |
|---|---|
| 不理解三类 dropout 的位置 | 结构图和 block 示意图 |
| 不理解为什么期望不变 | PyTorch Dropout 文档和 inverted dropout 公式 |
| 微调过拟合 | Hugging Face 配置项、任务头 dropout 设置 |
| 评估结果不稳定 | `model.eval()`、随机种子、推理路径 |
| 深层网络很深且训练不稳 | stochastic depth / drop path 文献 |

最后给出几条可直接执行的工程建议，作为参考资料部分的落点：

1. 初始配置通常可从 `attention=0.1`、`hidden=0.1` 起步。
2. 小数据微调优先调任务头和最后几层，不要先把全模型所有 dropout 一起拉高。
3. 任何验证、导出、上线前，都检查模型是否处于 `eval` 模式。
4. 如果 attention 图长期只盯少数 token，可以优先检查 attention dropout 和数据偏置。
5. 如果训练与验证都差，不要先加大 dropout，先排查学习率、数据质量和模型容量。
