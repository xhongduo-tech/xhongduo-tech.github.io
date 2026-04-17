## 核心结论

可学习绝对位置编码（Learned Positional Embedding，LPE，白话讲就是“给第 1 个位置、第 2 个位置……分别配一张可训练的向量卡片”）的优点是实现简单、训练稳定、在固定长度内拟合能力强，但它有三个根本局限。

第一，外推失败。LPE 只为 $1 \sim \text{max\_len}$ 这些位置维护参数。超过训练或设计长度后，新位置没有对应参数，模型无法自然理解“第 513 个位置”和“第 1025 个位置”是什么。公式上就是：

$$
z_i = x_i + p_i,\quad i \le \text{max\_len}
$$

当 $i > \text{max\_len}$ 时，$p_i$ 不存在，系统只能截断、补零，或用别的临时策略顶上。

第二，缺乏平移先验。平移先验，白话讲就是“同样的局部结构，整体往后挪几格后，模型仍应认为它们相似”。LPE 中每个位置向量独立学习，$p_{10}$ 和 $p_{11}$ 没有任何必须接近的约束，$p_{100}$ 和 $p_{101}$ 也一样。模型只能记住“绝对编号”，不擅长学习“相对距离”。

第三，参数冗余。冗余，白话讲就是“花了很多参数，学到的却是高度重复的信息”。实际训练里，相邻位置嵌入常常非常相似，常见现象是 $\cos(p_i, p_{i+1}) \approx 0.99$。这说明模型确实在用很多参数反复表达“相邻位置差不多”。

一个适合新手的玩具例子是：把每个位置想成桌面上的一张编号贴纸。训练时你只准备了 512 张贴纸。第 1 到 512 个 token 都能贴上不同颜色的标签；第 513 个开始根本没贴纸，模型就只能“看不见后面的位置”。这不是优化不够，而是表示法本身没有定义。

从工程视角看，LPE 适合固定长度、长度边界清晰的任务；一旦任务转向长文档问答、代码仓检索、长上下文对话，它通常明显弱于 RoPE 和 ALiBi 这类带共享结构的位置方法。

---

## 问题定义与边界

LPE 的定义非常直接：它是一个大小为 $\text{max\_len} \times d$ 的参数表，其中 $\text{max\_len}$ 是最大序列长度，$d$ 是嵌入维度。对于位置 $i$，模型直接查出向量 $p_i$，再与 token embedding $x_i$ 相加：

$$
z_i = x_i + p_i
$$

这里的 token embedding，白话讲就是“词本身的语义向量”；position embedding，白话讲就是“这个词在第几个位置的向量标记”。

以 BERT 为例，若 $\text{max\_len}=512$，$d=768$，则 LPE 参数量为：

$$
512 \times 768 = 393{,}216
$$

这约等于 39.3 万个额外参数。参数量本身不算大，但问题不在“多不多”，而在“这些参数是否学到了可泛化的结构”。

LPE 的边界是硬性的，不是软性的。硬边界，白话讲就是“超出以后不是性能慢慢变差，而是表示直接没有定义”。可以把它写成一个查表条件：

$$
p(i)=
\begin{cases}
\text{EmbeddingTable}[i], & 1 \le i \le \text{max\_len} \\
\text{undefined}, & i > \text{max\_len}
\end{cases}
$$

这意味着两个现实后果：

| 项目 | LPE 在训练长度内 | LPE 超出训练长度 |
|---|---|---|
| 位置向量是否有定义 | 有 | 无或只能人工补丁 |
| 梯度是否可更新 | 可以 | 不存在对应参数 |
| 是否具备自然外推 | 没问题 | 几乎没有 |
| 是否共享位置规律 | 很弱 | 更弱 |

一个家宴座位的例子足够说明边界：你只摆了 512 套餐具，第 513 位客人来了，不是“多等一下就有”，而是“系统里根本没准备这个位置”。在模型里，这种“没准备”通常转化为截断、滑窗、分块，或者强行映射到已有位置。

这里还要区分两个概念。

绝对位置，指“这是第 37 个 token”；相对位置，指“它和前一个词相隔 1，它和标题相隔 12”。LPE主要编码前者，而很多长上下文任务真正依赖的是后者。例如代码里一个变量定义和使用隔了 900 个 token，模型需要的是“二者相关、距离很远”，而不是“一个在 105，另一个在 1005”。

---

## 核心机制与推导

LPE 的前向传播没有复杂技巧，本质就是查表再相加：

$$
z_i = x_i + p_i
$$

自注意力层随后对 $z_i$ 计算 Query、Key、Value。也就是说，后续所有注意力行为都建立在“token 语义 + 绝对位置卡片”这个表示之上。

问题出在梯度传播。设损失为 $L$，对于某个合法位置 $i \le \text{max\_len}$，有：

$$
\frac{\partial L}{\partial p_i} \neq 0
$$

只要该位置出现在训练样本里，位置向量 $p_i$ 就会被更新。可一旦 $i > \text{max\_len}$，模型中没有这个参数，于是：

$$
\frac{\partial L}{\partial p_i} = 0,\quad i > \text{max\_len}
$$

更准确地说，不是“这个参数梯度变小了”，而是“这个参数根本不存在，因此不存在可学习路径”。这就是为什么 LPE 的外推不是困难，而是先天断裂。

从表示结构上看，LPE 还有一个更深的问题：它没有显式建模邻接关系。邻接关系，白话讲就是“第 10 个位置和第 11 个位置应该比第 10 个和第 400 个更接近”。在 LPE 中，$p_{10}, p_{11}, p_{400}$ 都只是独立行向量，只有数据分布间接推动它们形成某种规律，没有任何结构保证。

这会带来位置互换不变性差。互换不变性，白话讲就是“同样一句局部短语，整体后移几格后，模型不该完全陌生”。假设训练里经常见到模式：

- 位置 20: `if`
- 位置 21: `(`
- 位置 22: `x`

模型可能把这种模式和 $p_{20}, p_{21}, p_{22}$ 强绑定。若推理时同样模式出现在位置 600、601、602，而这些位置要么没学过，要么没定义，模型就无法平滑迁移这种认识。

玩具例子可以写成两个三词序列：

- 序列 A：`[猫, 在, 睡]` 出现在位置 1,2,3
- 序列 B：`[猫, 在, 睡]` 出现在位置 101,102,103

对人来说，这两个局部模式完全一样，只是整体后移。对理想的位置编码来说，它们至少应在某种变换下保持相似。LPE 却要分别学 $p_1,p_2,p_3$ 和 $p_{101},p_{102},p_{103}$，没有参数共享的几何约束。

再看冗余问题。如果训练数据主要由自然语言或代码组成，相邻 token 的功能差异通常平滑变化，因此最省损失的做法往往是把相邻位置学得很像。于是我们经常观察到：

$$
\cos(p_i, p_{i+1}) = \frac{p_i \cdot p_{i+1}}{\|p_i\|\|p_{i+1}\|} \approx 1
$$

当这个值长期接近 1，就说明很多位置参数只是“缓慢变化的查表曲线”，而不是强信息密度的独立表示。换句话说，模型用了离散参数表去逼近一个本可由共享函数表达的平滑结构。

这也是为什么 RoPE、ALiBi、KERPLE、CAPE 这类方法更有吸引力。它们不是给每个位置单独发卡片，而是给“位置之间的关系”加规则。

---

## 代码实现

在工程中，LPE 通常直接写在 embedding 层。PyTorch 风格的核心实现大致如下：

```python
import numpy as np

class LearnedPositionEmbedding:
    def __init__(self, max_len: int, dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.max_len = max_len
        self.dim = dim
        self.table = rng.normal(0.0, 0.02, size=(max_len, dim))

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: [seq_len, dim]
        seq_len = x.shape[0]
        if seq_len > self.max_len:
            x = x[:self.max_len]
            seq_len = self.max_len

        pos = np.arange(seq_len)
        return x + self.table[pos]

# 可运行示例
dim = 4
max_len = 8
x = np.zeros((6, dim), dtype=np.float32)

lpe = LearnedPositionEmbedding(max_len=max_len, dim=dim, seed=42)
y = lpe.forward(x)

assert y.shape == (6, dim)
assert not np.allclose(y[0], y[1])  # 不同位置通常不同
assert np.allclose(lpe.forward(np.zeros((20, dim))).shape, (8, dim))  # 超长被截断
```

上面代码刻意展示了一个现实事实：超长序列通常不是“自然外推”，而是“直接裁切”。这就是很多早期模型在前向里做的事情。

如果用深度学习框架实现，结构会更直接：

```python
import torch
import torch.nn as nn

class TokenWithLPE(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, dim: int):
        super().__init__()
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_len, dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch, seq_len]
        batch, seq_len = input_ids.shape
        if seq_len > self.max_len:
            input_ids = input_ids[:, :self.max_len]
            seq_len = self.max_len

        pos = torch.arange(seq_len, device=input_ids.device)
        tok = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos).unsqueeze(0)  # [1, seq_len, dim]
        out = tok + pos_emb

        assert out.shape == (batch, seq_len, tok.size(-1))
        return out
```

这里有两个初学者最容易忽略的点。

第一，位置索引和序列长度必须同步裁切。否则 token 被截断了，但位置没截断，或者反过来，都会产生 shape 错误或语义错位。

第二，LPE 层本身不提供任何超长处理能力。所谓“支持长文档”，往往不是这层变强了，而是系统在它外面加了滑动窗口、重叠分块、检索增强等补丁。

一个真实工程例子是长文档 QA。假设你有一份 600 token 的法律文档，BERT 的 `max_len=512`。如果答案出现在第 530 到 560 token，最直接的前向会把它整个截掉。工程上常见做法是 sliding window，也就是滑动窗口，把文档拆成多个 512 长度片段，窗口之间重叠一部分，再分别跑模型。这样能缓解信息丢失，但代价是：

- 推理次数增加
- 重叠区域重复计算
- 跨窗口依赖仍然困难
- 后处理更复杂

不同框架里的接口差异不大，真正差别在默认行为：

| 框架/实现习惯 | LPE 接口 | 超长处理常见方式 | 风险 |
|---|---|---|---|
| PyTorch `nn.Embedding` | 显式传 `positions` | 手动截断或外部滑窗 | 易忘记同步裁切 |
| TensorFlow/Keras `Embedding` | 显式构造位置张量 | 在 `call()` 内裁切 | 图模式下调试较绕 |
| HuggingFace 早期 BERT 类模型 | 模型内部维护 position ids | tokenizer 侧先截断 | 用户误以为模型“支持长文本” |

从代码层面看，LPE 很简单；真正困难的是它把长度边界问题推给了系统设计。

---

## 工程权衡与常见坑

LPE 不是不能用，而是适用条件很窄。它更像“固定输入长度下的低摩擦方案”，不是“可扩展上下文方案”。

先看常见工程权衡：

| 维度 | LPE | RoPE / ALiBi |
|---|---|---|
| 训练实现复杂度 | 低 | 中 |
| 固定长度内拟合 | 强 | 强 |
| 长度外推 | 弱 | 通常更强 |
| 参数共享 | 很少 | 明显更多 |
| 平移泛化 | 差 | 更好 |
| 长文档任务适配 | 依赖补丁 | 天然更合适 |

最典型的坑有三个。

| 常见坑 | 直接后果 | 常见补救 |
|---|---|---|
| 输入超过 `max_len` | 后半段上下文丢失 | 截断、滑窗、分块 |
| 把未见长度强行映射到已有位置 | 位置语义错乱 | 改用可外推编码 |
| 相邻位置高度相似 | 参数利用率低 | 用共享函数或相对编码 |

真实工程例子是长文档 QA 或代码检索。比如一个代码仓中的函数定义在文件头部，调用点在 3000 token 之后。若模型用的是训练上限 512 的 LPE，那么系统通常只能：

- 把文件切块
- 检索候选块
- 分别编码
- 再靠额外逻辑拼接答案

这类方案不是没效果，但它依赖大量系统侧工程弥补表示层缺陷。相比之下，RoPE 或 ALiBi 至少在编码层面对“更长距离”有连续定义，推理时不会在第 513 个位置突然掉到悬崖下。

另一个常被忽略的坑是“训练内表现不错，误导团队低估风险”。在常规 benchmark 中，如果所有样本都短于 256 token，LPE 可能和更先进方法差别不大，甚至略好，因为它直接为每个位置拟合参数，表达自由度更高。但一旦线上输入分布漂移，出现更长段落、日志、代码、合同文本，模型退化往往非常突然。

可以把这种退化理解成一张只拍前 512 个像素的相机。前 512 个像素内，画面可以很清楚；第 513 个像素之后，不是模糊，而是黑屏。长文本场景里，这种“黑屏式失败”比“平滑变差”更危险，因为它难以靠少量调参修复。

研究综述里经常提到，RoPE 和 ALiBi 在长上下文任务上的退化更平滑，尤其在训练长度外仍保留可解释的相对位置信号。LPE 模型则常见注意力图失稳、关键信息错位、答案依赖前段上下文，后段证据利用率显著下降。

---

## 替代方案与适用边界

如果问题核心是“需要长度泛化”，优先级通常不是继续堆 LPE 参数，而是换编码机制。

RoPE（Rotary Positional Embedding，旋转位置编码，白话讲就是“把位置信息变成向量空间里的旋转角度”）的关键思想是：不再给每个位置一个独立表项，而是让向量随位置做规则旋转。其简化形式可以写成：

$$
q_i' = R(i) q_i,\quad k_j' = R(j) k_j
$$

其中 $R(i)$ 是由位置 $i$ 决定的旋转矩阵。这样注意力分数天然与相对位置 $i-j$ 有关，因此更适合外推和长距离建模。

ALiBi（Attention with Linear Biases，线性注意力偏置，白话讲就是“直接在注意力分数里按距离加一个线性惩罚”）更直接：

$$
\text{score}_{ij} = \frac{q_i k_j^\top}{\sqrt{d}} + m_h \cdot (i-j)
$$

其中 $m_h$ 是每个头的斜率。距离越远，偏置越大或越小，模型因此学到“远近”而不是“绝对编号”。

对 LPE 的改良也存在。KERPLE、CAPE 这类方法的方向不是完全抛弃可学习性，而是给可学习位置编码加入结构先验。

- KERPLE 用核函数约束位置关系。核函数，白话讲就是“不是逐点独立记忆，而是让不同位置之间按某种平滑规则相关联”。
- CAPE 用卷积或增强策略引入平移先验。卷积先验，白话讲就是“局部模式往哪平移，提取规则基本不变”。

它们的共同目标是：保留“可学”，但不要让每个位置完全孤立。

| 方法 | 核心机制 | 外推能力 | 适用场景 |
|---|---|---|---|
| LPE | 每个位置独立查表 | 弱 | 固定短序列、老模型兼容 |
| RoPE | 向量按位置旋转 | 强 | 长文、代码、大模型 |
| ALiBi | 注意力分数加线性距离偏置 | 强 | 需要简单扩展上下文 |
| KERPLE | 核函数建模距离关系 | 中到强 | 想保留可学习性并增强泛化 |
| CAPE | 卷积/增强引入平移结构 | 中 | 需要更强位置先验的变体 |

什么时候 LPE 仍然合理？边界大致有三种。

第一，输入长度上限天然固定且严格，比如短查询分类、固定窗口日志标签、少量模板文本抽取。  
第二，系统必须兼容旧模型结构，改动编码方式成本太高。  
第三，任务更看重训练长度内精细拟合，而不是跨长度泛化。

反过来，如果任务具有以下特征，LPE 通常不是优先选项：

- 文本长度分布长尾明显
- 线上长度可能超过训练长度
- 需要跨段、跨函数、跨章节依赖
- 希望同一模型覆盖短文本和长文本

结论可以压缩成一句话：LPE 的问题不是“学得不够多”，而是“表示假设过于局部、过于离散、过于依赖绝对编号”。

---

## 参考资料

| 资料 | 支撑内容 | URL |
|---|---|---|
| Articsledge, “What Is Positional Encoding?” (2026) | 支撑 LPE 的定义、BERT/GPT 使用方式、与 ALiBi 的对比 | https://www.articsledge.com/post/positional-encoding |
| EmergentMind, Absolute Positional Embeddings 综述 | 支撑外推失败、平移不变性差、参数冗余等局限总结 | https://www.emergentmind.com/topics/absolute-positional-embeddings |
| Michael Brenndoerfer, “Learned Position Embeddings” | 支撑“相邻位置高度相似”的经验现象与直觉解释 | https://mbrenndoerfer.com/writing/learned-position-embeddings |
| Next.gr, Positional Encoding in Transformers | 支撑 $z_i=x_i+p_i$ 及超出 `max_len` 时无定义的基本公式说明 | https://www.next.gr/ai/deep-learning-theory/positional-encoding-in-transformers |
| mlJourney, Positional Encoding Types in Transformers | 支撑 RoPE、ALiBi 在长文档和长上下文应用中的工程对比 | https://mljourney.com/positional-encoding-types-in-transformers/ |
| OpenReview, KERPLE / CAPE 相关论文页面 | 支撑“通过核函数或卷积先验增强可学习位置编码”的改进方向 | https://openreview.net/forum?id=hXzOqPlXDwm |

这些资料分别对应本文的不同部分：Articsledge 和 Next.gr 主要支撑定义与基本公式；EmergentMind 和 Brenndoerfer 主要支撑局限性分析；mlJourney 主要支撑长上下文工程对比；OpenReview 相关工作主要支撑替代方案中的 KERPLE/CAPE 部分。
