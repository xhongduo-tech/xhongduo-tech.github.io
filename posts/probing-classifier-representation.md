## 核心结论

Probing 的核心用途，是检测模型内部表征里是否已经编码了某种信息。这里的“表征”可以先理解成：模型在某一层对输入形成的一组数字向量；“编码了信息”不是指人能直接读懂，而是指另一个简单模型可以从这组向量里把目标属性读出来。

最常见的做法，是在冻结后的中间激活 $h_k$ 上训练一个线性分类器。这里的“线性分类器”可以先理解成：只做加权求和再分类的简单读出器，没有多层非线性网络那么强。如果这个探针在验证集上能稳定预测某个属性，比如“句子是否含动词”“实体类别”“事实是否正确”，通常可以说：该层表征对这个属性具有较强的线性可读性。

但结论必须收窄。Probing 说明的是“信息能不能被读出来”，不是“主模型做决策时一定用了这份信息”。这是很多初学者最容易混淆的点。探针高准确率，不等于模型因果地依赖了该信息；它只说明这类信息存在于表征中，并且分布方式足够简单，简单到一个线性模型就能把它分开。

一个最小玩具例子是语法标签检测。假设我们取 Transformer 第 6 层每个 token 的向量，给每个 token 标上“是否是动词”。冻结 Transformer，只训练一个逻辑回归。如果验证准确率明显高于随机猜测，说明第 6 层已经把“动词性”写进了表示里。

一个真实工程例子，是对大语言模型各层做“事实正确性”探针。输入一批问答样本，提取每层对答案 token 的隐藏状态，再训练探针预测该答案最终是否事实正确。把层号和探针准确率画成曲线，就能看到事实相关信息大致在哪些层开始变得容易读出。这类结果常被用于解释、监控和后续干预设计。

| 指标 | 含义 | 典型解释 |
| --- | --- | --- |
| 探针准确率高 | 目标属性可从该层线性读出 | 该层含有可分离信号 |
| 随机 baseline 低 | 随机猜测效果差 | 任务本身不是“白送分” |
| 对照任务也高 | 探针可能过强或数据泄漏 | 不能直接宣称表征有效 |
| selectivity 高 | 真实任务比对照任务强很多 | 更可信地说明信号来自表征 |

---

## 问题定义与边界

形式化地说，给定模型第 $k$ 层表示 $h_k \in \mathbb{R}^d$，我们训练一个探针：

$$
z = W h_k + b
$$

其中 $W,b$ 是探针参数，$z$ 是分类前的分数。对二分类或多分类任务，再经过 softmax 或 sigmoid 得到预测概率。训练时只更新 $W,b$，主模型参数保持冻结，也就是常说的 stop-gradient。这里的“冻结”可以先理解成：把主模型当成固定特征提取器，不允许它为了适应探针而改变自己。

因此，Probing 要回答的问题非常明确：

1. 某个属性是否已经存在于某层表示中？
2. 该属性在哪些层更容易被读出？
3. 这种可读性是否足够强到超出随机和对照任务？

它不能直接回答的问题也要说清楚：

1. 主模型是否真正依赖该属性做最终决策？
2. 该层信息是“被使用的原因”还是“顺带留下的痕迹”？
3. 准确率高是否来自表征本身，而不是探针容量太强？

可以把任务边界画成一个简单流程：

| 阶段 | 输入 | 输出 | 能说明什么 | 不能说明什么 |
| --- | --- | --- | --- | --- |
| 主模型第 $k$ 层 | 原始样本 | 隐藏状态 $h_k$ | 该层产生了什么表示 | 不知道表示是否被后续使用 |
| 线性 probe | $h_k$ | 目标属性预测 | 属性能否被线性读出 | 不说明因果关系 |
| 对照任务 | 随机标签或随机特征 | 对照准确率 | 探针是否在记忆数据 | 不直接定位机制 |

一个典型新手实验是“按层扫描”。例如，对“事实是否正确”做二分类，分别取第 1 层到第 24 层隐藏状态训练探针，得到层号 vs 准确率曲线。如果前几层接近随机，中间层开始上升，后层更高，说明事实相关信号不是一开始就显式存在，而是在中后层逐步形成。

这类分析很有价值，因为它把“模型会不会”拆成了“模型在哪里开始会”。对解释性、安全和调试都很有帮助。

---

## 核心机制与推导

探针训练本质上是一个标准监督学习问题，只是输入不是原始文本，而是模型内部激活。

设第 $k$ 层表示为 $h_k$，类别数为 $C$。则线性探针输出：

$$
z = W h_k + b, \quad W \in \mathbb{R}^{C \times d},\ b \in \mathbb{R}^{C}
$$

再经过 softmax：

$$
p(y=c \mid h_k) = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}
$$

交叉熵损失为：

$$
\mathcal{L} = - \sum_{c=1}^{C} y_c \log p(y=c \mid h_k)
$$

训练时的关键约束是：

$$
h_k = \operatorname{stopgrad}(f_k(x))
$$

意思是，$h_k$ 来自主模型 $f_k(x)$，但反向传播时梯度不回到主模型，只更新探针参数。这样才能保证我们测的是“已有信息”，而不是“探针训练过程逼着主模型重新编码信息”。

看一个最小数值玩具例子。假设二分类任务中：

$$
h_k = [0.8, 0.2], \quad W = [1, -1], \quad b = 0
$$

那么：

$$
z = 1 \times 0.8 + (-1) \times 0.2 = 0.6
$$

如果把它看成“属于正类的 logit”，则 $z=0.6$ 表示该样本更偏向正类。若另一个负样本的表示是 $[0.2, 0.8]$，则：

$$
z = 1 \times 0.2 + (-1) \times 0.8 = -0.6
$$

于是正负样本在线性平面上被自然分开。这就是“线性可分”的最直观含义：不需要复杂网络，只需一条超平面就能区分目标属性。

为什么工程上经常强调正则化？因为探针太强，就可能自己学出复杂边界，甚至记住训练集，而不是忠实反映主模型表征。常见控制方法有：

| 方法 | 白话解释 | 作用 |
| --- | --- | --- |
| L2 正则 | 限制权重不要过大 | 减少过拟合 |
| 低秩约束 | 限制可用方向数 | 控制探针表达能力 |
| 子维度采样 | 只给探针看部分维度 | 测试信息是否稠密存在 |
| 早停 | 验证集不涨就停止 | 防止记忆训练集 |

所以，Probing 不是“训练一个分类器然后看分数”这么简单，真正可信的结论来自：冻结、低容量、对照组、层间比较，这四件事一起成立。

---

## 代码实现

下面给出一个可运行的最小 Python 示例。它模拟一个已经冻结的隐藏表示，其中第一个维度携带标签信号，后面维度主要是噪声。我们训练一个线性 probe，并与随机标签对照任务比较。

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def train_logistic_probe(X, y, lr=0.1, steps=2000, l2=1e-3):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    # 这里的 X 可看作 stop-gradient 之后的隐藏状态，不会回传到主模型
    for _ in range(steps):
        logits = X @ w + b
        probs = sigmoid(logits)

        grad_logits = probs - y
        grad_w = (X.T @ grad_logits) / n + l2 * w
        grad_b = np.mean(grad_logits)

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b

def predict(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)

def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

rng = np.random.default_rng(42)

# 构造“冻结后的内部表征”
n_train, n_val, d = 400, 200, 8
X_train = rng.normal(size=(n_train, d))
X_val = rng.normal(size=(n_val, d))

# 让第 0 维带有可读标签信号
y_train = (X_train[:, 0] + 0.3 * rng.normal(size=n_train) > 0).astype(int)
y_val = (X_val[:, 0] + 0.3 * rng.normal(size=n_val) > 0).astype(int)

w, b = train_logistic_probe(X_train, y_train)
val_acc = accuracy(y_val, predict(X_val, w, b))

# 对照任务：随机标签
random_y_train = rng.integers(0, 2, size=n_train)
random_y_val = rng.integers(0, 2, size=n_val)
w_ctrl, b_ctrl = train_logistic_probe(X_train, random_y_train)
ctrl_acc = accuracy(random_y_val, predict(X_val, w_ctrl, b_ctrl))

selectivity = val_acc - ctrl_acc

print("probe_acc =", round(val_acc, 4))
print("control_acc =", round(ctrl_acc, 4))
print("selectivity =", round(selectivity, 4))

assert val_acc > 0.75
assert ctrl_acc < 0.65
assert selectivity > 0.15
```

这段代码体现了 Probing 的最小数据流：

1. 主模型先产出隐藏状态 `X_train/X_val`。
2. 训练时只更新探针参数 `w, b`。
3. 同时跑一个随机标签对照任务。
4. 用 `selectivity = 真实准确率 - 对照准确率` 判断结果是否可信。

如果换成真实 Transformer，流程也类似：

```python
# 伪代码
model.eval()
freeze(model)

all_features = []
all_labels = []

for batch in dataloader:
    tokens, labels = batch
    with no_grad():
        hidden_states = model(tokens, output_hidden_states=True)
        h_k = hidden_states[layer_id]          # [batch, seq, dim]
        feat = select_token_representation(h_k) # 例如取最后一个 token
    all_features.append(feat)
    all_labels.append(labels)

probe = LinearProbe(input_dim=dim, num_classes=2, l2=1e-4)
train_probe(probe, features=all_features, labels=all_labels)
evaluate_probe(probe, valid_features, valid_labels)
run_control_task_with_shuffled_labels(...)
```

一个真实工程例子是安全评估。团队想知道模型在生成答案前，哪一层已经暴露“这段回答是否可能包含不实信息”的信号。做法是收集问答样本，标注最终回答是否事实错误，然后对各层最后一个生成 token 的隐藏状态训练二分类 probe。若某几层的 selectivity 明显升高，就说明这些层已经形成较强的“错误倾向”表征。后续可以把这些层作为监控点，做告警、拒答或进一步干预实验。

---

## 工程权衡与常见坑

Probing 好用，但很容易被误用。最常见的问题不是代码写错，而是解释写过头。

| 风险/坑 | 具体表现 | 应对策略 |
| --- | --- | --- |
| 探针过拟合 | 训练集很高，验证集一般 | L2、早停、减少特征维度 |
| 探针容量过强 | 用多层 MLP 也叫 probe，但结论失真 | 优先线性 probe，保持低容量 |
| 标签泄漏 | 元数据暗含答案，比如长度、位置 | 做 baseline 和数据清洗 |
| 把相关当因果 | 准确率高就说模型依赖该特征 | 配合干预实验验证 |
| 对照不充分 | 只有真实任务，没有随机任务 | 加随机标签、随机特征、邻近层对比 |
| 样本不平衡 | 全预测多数类也有高分 | 看 F1、AUC、balanced accuracy |

这里最重要的工程指标之一是 selectivity。它常被定义为：

$$
\text{selectivity} = \text{Acc}_{\text{task}} - \text{Acc}_{\text{control}}
$$

白话讲，就是“真实任务成绩减去对照任务成绩”。如果真实任务 92%，随机标签任务 88%，那这个探针并不说明太多，因为它在无意义任务上也很强，可能只是容量大、数据少、容易记忆。相反，如果真实任务 84%，随机标签任务 54%，selectivity 很高，可信度反而更强。

再看一个玩具坑例子。你用 500 条样本训练 probe，输入维度 4096，但标签其实和样本来源网站相关。结果 probe 准确率 95%。这时你以为模型“编码了事实真伪”，实际上它可能只是读出了站点风格、标点习惯或文本长度等旁路信号。解决办法不是再堆一个更复杂探针，而是做更严格的数据切分和对照设计。

真实工程里还有一个常见问题：层间比较不公平。比如早层取单 token 表示，后层取平均池化表示，结果准确率差异其实来自抽取方式变化，而不是层本身差异。因此，跨层实验必须固定读取协议，例如都取最后 token、都做同样归一化、都使用同一训练超参数。

---

## 替代方案与适用边界

当目标是快速知道“某层有没有某种信息”，线性 probing 往往是第一选择，因为它便宜、快、可重复，也方便做层间扫描。

但如果你要回答的是“模型到底有没有用这份信息”，那 probing 不够。此时需要干预型方法，也就是常说的 causal probing、ablation、activation patching 一类方法。这里的“ablation”可以先理解成：人为遮掉某些维度、某些头或某些层，看模型性能是否下降。

| 方法 | 主要输出 | 好处 | 局限 |
| --- | --- | --- | --- |
| 线性 probing | 信息是否可读 | 快、便宜、容易横向比较 | 不给因果结论 |
| Causal probing / ablation | 去掉某信息后会怎样 | 更接近因果解释 | 干预设计复杂 |
| 信息论分析 | 表征与标签共享多少信息 | 理论表达更一般 | 估计难、对高维不稳定 |
| 表征相似性分析 | 不同层/模型是否学到类似结构 | 适合比较模型 | 不直接回答具体属性可读性 |

可以用一个直观对比来理解：

1. 线性 probe 像“读体检报告”，看某指标是否已经出现在数据里。
2. 因果干预像“做药物试验”，看去掉某因素后结果会不会变。

两者不是互斥关系，而是前后关系。实践中常见路线是：先用 probing 找热点层和热点维度，再做干预验证这些位置是否真的影响任务表现。这样成本最低，也最容易逐步建立可信结论。

适用边界可以概括为：

1. 想看信息形成时机，用 probing。
2. 想看模型是否依赖该信息，用干预。
3. 想做大规模扫描和监控，用低容量 probe。
4. 想下强解释结论，必须把 probing 和因果实验放在一起看。

---

## 参考资料

- Belinkov, Geva, et al. *Probing Classifiers: Promises, Shortcomings, and Advances*. Computational Linguistics, 2022。
  重点：系统总结 probing 的定义、常见误区、控制实验和解释边界。适合作为第一篇综述读物。

- Emergent Mind. *Linear Probes: Neural Network Diagnostics*。
  重点：用更工程化的方式介绍线性探针、训练流程和常见用途，适合先建立直觉再回到论文。

- 2026 Codelab. *Linear Probes for Deep Neural Networks*。
  重点：偏实践路线，适合按步骤复现最小实验，理解“冻结主网 + 提取表征 + 训练 probe”的完整流程。

- 阅读路径建议：
  先读 Belinkov 综述，建立“能说明什么、不能说明什么”的边界；再看 Emergent Mind 的工程化说明；最后结合 codelab 动手做一个按层扫描实验，把 selectivity、随机标签对照和层间曲线都跑一遍。
