## 核心结论

LSTM 语言模型的核心不是“更深的 RNN”，而是把“记忆”拆成两条通路处理：`h_t` 是短期输出，白话说就是当前时刻对外可见的工作区；`C_t` 是细胞状态，白话说就是沿时间轴传递的长期记忆带。它通过门控机制决定哪些旧信息保留、哪些新信息写入，因此比普通 RNN 更适合处理长依赖文本。

标准更新式是：

$$
f_t=\sigma(W_f[h_{t-1},x_t]+b_f),\quad
i_t=\sigma(W_i[h_{t-1},x_t]+b_i)
$$

$$
\tilde C_t=\tanh(W_c[h_{t-1},x_t]+b_c),\quad
C_t=f_t\odot C_{t-1}+i_t\odot \tilde C_t
$$

$$
o_t=\sigma(W_o[h_{t-1},x_t]+b_o),\quad
h_t=o_t\odot \tanh(C_t)
$$

这里的门是 `sigmoid` 输出到 $(0,1)$ 的控制器，白话说就是“通过多少”的阀门。关键点在于 $C_t$ 的更新包含一条加法路径，而不是像普通 RNN 那样每一步都把旧状态彻底乘上新矩阵再覆盖。于是反向传播时：

$$
\frac{\partial C_t}{\partial C_{t-1}} = f_t
$$

当 $f_t \approx 1$ 时，梯度可以沿着这条路径较稳定地往前传很多步，这就是 LSTM 能记住几十步甚至上百步上下文的根本原因。

对语言模型来说，这意味着它更容易记住“前面主语是谁”“引号是否打开”“代码块是否开始”“主题是否切换”。普通 RNN 往往在几十步后把这类信号冲淡，而 LSTM 可以有选择地保留。

---

## 问题定义与边界

语言模型的任务是估计词序列概率：

$$
P(w_1,\dots,w_T)=\prod_{t=1}^{T}P(w_t\mid w_{<t})
$$

也就是：给定前文，预测下一个词。问题不在“会不会预测”，而在“能记多远”。

普通 RNN 的递归形式通常是：

$$
h_t=\tanh(W_hh_{t-1}+W_xx_t+b)
$$

它的问题是，状态每一步都要经过矩阵乘法和非线性激活。时间一长，梯度会反复连乘，若谱半径小于 1 就容易衰减，若过大又可能爆炸。结果是模型更偏向近邻上下文，对远处信息利用差。

下面这个表格能直接看出差别：

| 信号类型 | 例子 | 普通 RNN 的处理 | LSTM 的处理 |
|---|---|---|---|
| 短期信号 | 当前词后面常见的搭配 | 通常能学到 | 能学到 |
| 中程信号 | 10 到 30 个 token 前的修饰关系 | 容易变弱 | 可由门控保留 |
| 长期信号 | 50 步前的主语、话题、括号结构 | 常被覆盖 | 可沿 `C_t` 传递 |
| 噪声信号 | 偶然出现、对后文无用的词 | 容易混入状态 | 可由遗忘门压掉 |

一个玩具例子：句子是“`The key to the cabinets ... is`”。真正决定后面该接 `is` 还是 `are` 的，是很早出现的主语 `key`，不是最近的 `cabinets`。普通 RNN 往往更容易被近处复数词干扰；LSTM 可以把“主语单复数”作为长期状态保留到谓语出现位置。

边界也要说清楚。LSTM 不是无限记忆，也不是自动会“理解语法”。它只是给优化过程提供了更稳定的路径，让模型更有机会学到长依赖。若序列极长、训练数据很少、或任务需要大规模并行与全局注意力，LSTM 仍会吃亏。

---

## 核心机制与推导

LSTM 每个时间步都接收同一组输入源：上一步隐藏状态 $h_{t-1}$ 和当前输入 $x_t$。然后同时计算四组量：

| 量 | 公式 | 作用 |
|---|---|---|
| 忘记门 $f_t$ | $\sigma(W_f[h_{t-1},x_t]+b_f)$ | 保留多少旧记忆 |
| 输入门 $i_t$ | $\sigma(W_i[h_{t-1},x_t]+b_i)$ | 接受多少新信息 |
| 候选记忆 $\tilde C_t$ | $\tanh(W_c[h_{t-1},x_t]+b_c)$ | 准备写入的内容 |
| 输出门 $o_t$ | $\sigma(W_o[h_{t-1},x_t]+b_o)$ | 暴露多少内部状态 |

信息流可以理解成两步：

1. 先更新长期记忆  
   $$
   C_t=f_t\odot C_{t-1}+i_t\odot \tilde C_t
   $$
2. 再从长期记忆里读出当前输出  
   $$
   h_t=o_t\odot\tanh(C_t)
   $$

这个式子最关键的不是门很多，而是“旧记忆”和“新记忆”是相加，不是互相覆盖。线性代数上看，若把各维独立看待，第 $k$ 维满足：

$$
C_t^{(k)}=f_t^{(k)}C_{t-1}^{(k)}+i_t^{(k)}\tilde C_t^{(k)}
$$

它就是一个可学习的指数平滑器：旧值乘一个保留系数，再加上新候选乘一个写入系数。

看一个最小数值例子。设：

- $f_t=[0.2,0.8]$
- $C_{t-1}=[1.0,0.5]$
- $i_t=[0.9,0.1]$
- $\tilde C_t=[0.7,-0.3]$

则：

$$
C_t=[0.2\times1.0+0.9\times0.7,\;0.8\times0.5+0.1\times(-0.3)]=[0.83,0.37]
$$

解释很直接：

- 第一维：旧信息只保留 20%，但强力写入新信息，所以新内容主导。
- 第二维：旧信息保留 80%，新内容写得很少，所以记忆更稳定。

这就是“选择性记忆”。如果把忘记门想成“可调滤纸”，白话说就是它决定旧信息漏掉多少；输入门像“新纸入口”，决定新内容能进来多少。

再看梯度为什么更稳定。对当前步的细胞状态对上一步求导：

$$
\frac{\partial C_t}{\partial C_{t-1}} = f_t
$$

沿时间展开后：

$$
\frac{\partial C_t}{\partial C_{t-k}}=\prod_{j=t-k+1}^{t}f_j
$$

如果多个时间步上 $f_j$ 都接近 1，那么这串乘积不会迅速变成 0。普通 RNN 常见的是矩阵与激活导数连乘，更容易衰减。LSTM 的优势就在这条“常数误差跑道”式的加法通路。

真实工程例子是词级语言模型。在 Penn Treebank 这类基准上，早期强基线是 Zaremba 等人的 large LSTM，常见配置是两层、每层 1500 单元，测试 perplexity 大约 78.4。perplexity 可以白话理解成“模型平均还有多不确定”，越低越好。后来 Gal 与 Ghahramani 的 variational dropout LSTM large 可进一步到约 75.0，而 Merity 等人的 AWD-LSTM 把 PTB 测试集做到 57.3，WikiText-2 做到 65.8。在 Transformer 普及前，这类结果长期代表神经语言模型的强实战水平。

---

## 代码实现

下面给一个可运行的最小 LSTM 单步实现。它只演示门控和状态更新，不涉及训练。

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def lstm_step(x_t, h_prev, c_prev, params):
    # 拼接当前输入和上一步隐藏状态
    z = np.concatenate([h_prev, x_t])

    f_t = sigmoid(params["Wf"] @ z + params["bf"])
    i_t = sigmoid(params["Wi"] @ z + params["bi"])
    g_t = np.tanh(params["Wc"] @ z + params["bc"])   # 候选记忆
    o_t = sigmoid(params["Wo"] @ z + params["bo"])

    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * np.tanh(c_t)
    return h_t, c_t, (f_t, i_t, g_t, o_t)

# 一个确定性的玩具参数，方便验证
hidden_size = 2
input_size = 2
z_size = hidden_size + input_size

params = {
    "Wf": np.zeros((hidden_size, z_size)),
    "Wi": np.zeros((hidden_size, z_size)),
    "Wc": np.zeros((hidden_size, z_size)),
    "Wo": np.zeros((hidden_size, z_size)),
    "bf": np.array([0.0, 0.0]),   # sigmoid(0)=0.5
    "bi": np.array([0.0, 0.0]),
    "bc": np.array([0.0, 0.0]),   # tanh(0)=0
    "bo": np.array([0.0, 0.0]),
}

x_t = np.array([1.0, -1.0])
h_prev = np.array([0.2, -0.1])
c_prev = np.array([0.6, -0.4])

h_t, c_t, gates = lstm_step(x_t, h_prev, c_prev, params)

# 在这个构造里，f=i=o=[0.5,0.5]，g=[0,0]
# 所以 c_t = 0.5 * c_prev, h_t = 0.5 * tanh(c_t)
assert np.allclose(c_t, np.array([0.3, -0.2]))
assert np.allclose(h_t, 0.5 * np.tanh(np.array([0.3, -0.2])))
print("ok")
```

这段代码体现了三个实现细节：

1. LSTM 需要同时维护 `h_t` 和 `c_t`。只传隐藏状态、不传细胞状态，不是完整的 LSTM。
2. 门控是逐元素乘法 `*`，不是矩阵乘法。每个维度都能独立决定保留和写入比例。
3. 训练时通常一次处理整段序列，但本质仍是反复调用这个单步更新。

如果要把它扩展成语言模型，流程是：

1. 把 token id 查成词向量 `x_t`。
2. 逐步滚动 LSTM，得到每步 `h_t`。
3. 用线性层把 `h_t` 投影到词表大小。
4. 对下一个真实词做 softmax 交叉熵。

---

## 工程权衡与常见坑

LSTM 在工程上最常见的问题不是“写不出来”，而是“能训但效果不稳”。

第一类坑是 dropout 用错位置。标准 dropout 指每次前向随机把一部分单元置零，白话说就是训练时临时拔掉部分神经元。如果你在时间维上每一步都换一套掩码，模型前一时刻刚写入的记忆，下一时刻可能又被另一套随机掩码破坏，时序一致性会变差。

| dropout 方式 | 时间维掩码 | 对 LSTM 记忆的影响 | 常见结果 |
|---|---|---|---|
| naive dropout | 每个时间步都变 | 递归路径噪声大 | 容易不稳 |
| variational dropout | 一个序列内固定 | 时序结构更一致 | 更稳，泛化更好 |
| weight dropout / DropConnect | 丢权重不丢激活 | 直接正则 recurrent 权重 | AWD-LSTM 常用 |
| embedding dropout | 丢词向量维度或词 | 抑制词表过拟合 | 小数据常有效 |

variational dropout 的核心不是“dropout 更强”，而是“同一条序列里使用同一套掩码”。可以把它理解成：不是每一步换不同钥匙试门，而是这一整段都用同一把钥匙，系统行为更稳定。

第二类坑是状态管理。训练长文本时常做 truncated BPTT，也就是只反传固定长度窗口。如果你每个 batch 都把状态清零，模型就学不到跨窗口依赖；如果完全不断开图，显存又会爆。标准做法是传递状态值，但在窗口边界 `detach` 计算图。

第三类坑是指标理解错误。perplexity 下降不等于所有下游任务都会同步提升。它说明语言建模更准，但分类、问答、生成质量还受分词、数据域、解码策略影响。

第四类坑是过时口径混用。很多文章会把“1500 维 LSTM 大约 77 PPL”写成一个整数印象，但更严格地说，不同论文配置差别很大：是否两层、是否词嵌入共享、是否 MC dropout、是否 cache，都会影响结果。写工程文档时应注明数据集与配置，而不是只报一个数字。

---

## 替代方案与适用边界

如果你只是想“把 LSTM 用好”，比起自己从 vanilla LSTM 开始堆技巧，更实用的基线是 AWD-LSTM。它不是新单元，而是一套 LSTM 工程配方：weight-dropped LSTM、embedding dropout、activation regularization、NT-ASGD 等组合。它说明了一件事：很多性能差距不是来自单元公式，而是来自正则化和优化细节。

下面给一个简化对比：

| 模型 | 代表结果 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| vanilla LSTM | PTB test 约 78.4 | 结构简单，部署成熟 | 对正则化敏感 | 教学、轻量基线 |
| variational LSTM | PTB test 约 75.0，MC dropout 可到 73.4 | 小数据更稳 | 训练细节更多 | 中小规模文本 |
| AWD-LSTM | PTB 57.3，WikiText-2 65.8 | 经典强基线，参数效率高 | 并行性不如 Transformer | 资源受限、延迟敏感 |
| Transformer LM | 现代大模型主流 | 长程建模强、训练并行 | 显存与数据需求高 | 大数据、高吞吐训练 |

为什么 Transformer 后来全面占优？因为注意力机制能直接建立远距离 token 之间的连接，不必把信息压进一个递归状态里，训练也更易并行。LSTM 每一步依赖上一步，天然串行，长序列训练吞吐差。

但这不等于 LSTM 没用了。在下面这些条件下，LSTM 仍然合理：

- 数据量不大，需要强归纳偏置。
- 线上部署强调低延迟、低内存、逐步流式处理。
- 任务是时间序列或字符级/词级建模，输入连续到达。
- 需要一个简单、稳定、可解释的递归基线。

一个真实工程判断是：如果你在做移动端输入法、嵌入式日志预测、低资源领域文本建模，LSTM 往往仍比“缩小版 Transformer”更容易训、更省资源。如果你在做大规模预训练、长文上下文建模、多卡并行训练，Transformer 基本是默认选择。

---

## 参考资料

- Hochreiter, S., & Schmidhuber, J. 1997. *Long Short-Term Memory*。LSTM 原始论文，重点是如何缓解长期依赖中的梯度衰减问题。
- Gers, F. A., Schmidhuber, J., & Cummins, F. 2000. *Learning to Forget: Continual Prediction with LSTM*。引入忘记门，解释为什么需要可学习地清除旧状态。
- Zaremba, W., Sutskever, I., & Vinyals, O. 2014. *Recurrent Neural Network Regularization*。早期强基线，常被用来引用 PTB 上 medium/large LSTM 的 perplexity。
- Gal, Y., & Ghahramani, Z. 2016. *A Theoretically Grounded Application of Dropout in Recurrent Neural Networks*。variational dropout 的代表工作，核心是时间维共享掩码。
- Merity, S., Keskar, N. S., & Socher, R. 2018. *Regularizing and Optimizing LSTM Language Models*。AWD-LSTM 论文，工程价值很高。
- Sho Takase 的语言模型性能整理页。适合快速查不同年代 PTB/WikiText-2 的 perplexity 对比，但要注意不同论文配置不完全一致。
- 入门向材料如 “LSTM Explained” 一类文章，适合先建立门控直觉；进阶时应回到原始论文核对公式和实验设置。
