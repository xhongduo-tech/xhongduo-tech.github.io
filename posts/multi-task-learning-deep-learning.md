## 核心结论

多任务学习的核心做法是：让多个相关任务共用一套底层表示，再给每个任务接自己的输出头。共享表示可以理解为“模型先学通用能力”，任务头可以理解为“模型最后一步按任务各自作答”。这通常带来三类收益：参数更省、样本利用率更高、泛化更稳。

最常见架构是“共享骨干 + 任务特定头”。骨干就是负责抽特征的主体网络，例如 CNN 或 Transformer；任务头就是挂在骨干后面的轻量模块，例如分类层、序列标注层、回归层。对初学者来说，可以把它看成一个“多科大脑”：前面的大脑负责理解文本或图像，后面不同小模块分别负责分类、打分、抽取标签。

| 组件 | 作用 | 是否共享 | 典型形式 |
| --- | --- | --- | --- |
| 共享层 / 骨干 | 学通用特征 | 是 | ResNet、Transformer Encoder |
| 任务头 | 学任务专属决策边界 | 否 | 分类层、CRF、回归层 |
| 损失聚合器 | 把多个任务目标合成一个训练信号 | 部分共享逻辑 | 加权求和、动态权重 |

多任务学习不是“任务越多越好”。它最适合标签定义相关、输入模态相同、底层特征可复用的任务集。若任务差异过大，反而会出现负迁移。负迁移就是一个任务的训练信号把另一个任务带偏了。

---

## 问题定义与边界

多任务学习是指：给定任务集合 $\{\mathcal{T}_1,\mathcal{T}_2,\dots,\mathcal{T}_K\}$，训练一个模型同时完成这些任务，并在参数层面显式共享一部分表示。表示就是模型内部提取出来的特征向量，可以理解为“机器对输入的压缩理解”。

形式化地看，输入为 $x$，共享骨干为 $f_{\theta_s}(x)$，第 $k$ 个任务头为 $h_{\theta_k}(\cdot)$，则任务输出是：

$$
\hat{y}_k = h_{\theta_k}(f_{\theta_s}(x))
$$

这里 $\theta_s$ 是共享参数，$\theta_k$ 是任务私有参数。边界问题在于：哪些参数共享，哪些参数必须私有。

一个典型 NLP 场景是：同一个 Transformer 编码器处理医疗文本，然后分别做情感分类、命名实体识别、文本相似度。此时词嵌入层、编码器层通常共享；最后的线性分类器、序列标注层、回归头通常私有。共享的是“读懂文本”的能力，私有的是“怎么输出标签”的能力。

| 任务 | 标签空间 | 输入 | 共享参数 | 私有参数 |
| --- | --- | --- | --- | --- |
| 文本分类 | 离散类别 | 句子/段落 | Transformer 编码器 | 分类线性层 |
| 序列标注 | 每个 token 一个标签 | 句子 | Transformer 编码器 | token-level 分类层 / CRF |
| 相似度回归 | 连续分数 | 句对 | Transformer 编码器 | 回归层 |

这里要明确两个边界。

第一，任务边界。多任务学习要求任务之间至少共享一部分底层规律。例如“句子是否有毒”和“句子情感极性”都依赖文本语义，相关性较强；但“图像分割”和“金融时序预测”通常不该强行共用一个骨干。

第二，共享边界。共享过多会让任务互相干扰；共享过少又会退化成多个单任务模型。工程上常见做法是底层多共享，高层少共享，因为底层特征更通用，高层决策更贴近具体标签。

任务冲突本质上来自梯度冲突。梯度就是参数更新方向，可以理解为“每个任务都在拉模型往自己有利的方向走”。如果两个任务梯度夹角接近 $180^\circ$，说明一个任务在推参数向左，另一个在推向右，训练就容易不稳定。

---

## 核心机制与推导

多任务训练的总目标通常写成加权和：

$$
L_{\text{total}}=\sum_{k=1}^{K}\lambda_k L_k
$$

其中 $L_k$ 是第 $k$ 个任务的损失，$\lambda_k$ 是该任务权重。权重决定“这个任务在总更新里有多大话语权”。

### 1. 静态权重

最简单的办法是固定权重，比如两个任务都取 $0.5$：

$$
L_{\text{total}}=0.5L_A+0.5L_B
$$

玩具例子：任务 A 是二分类，当前损失为 $0.3$；任务 B 是回归，当前损失为 $0.6$。若固定等权：

$$
L_{\text{total}}=0.5\times0.3+0.5\times0.6=0.45
$$

这做法简单，但有两个问题。第一，不同任务损失量纲可能不同，MSE 和交叉熵不能直接按数值大小比较。第二，不同任务收敛速度不同，慢任务可能长期被快任务盖住。

### 2. 动态权重

动态权重的思想是：不是手工规定每个任务永远多重要，而是根据训练进展实时调整。一个直观指标是“相对下降比”：

$$
w_k(t)=\frac{L_k(t)}{L_k(1)}
$$

这里 $L_k(1)$ 是任务 $k$ 在训练初期的参考损失，$w_k(t)$ 越大，说明这个任务降得越慢，当前更难学，应该分到更高关注度。

为了把这些相对难度变成可用权重，常见做法是温度 softmax：

$$
\lambda_k(t)=\frac{\exp(w_k(t)/\tau)}{\sum_{j=1}^{K}\exp(w_j(t)/\tau)}
$$

$\tau$ 是温度系数，可以理解为“调权重激进程度的旋钮”。$\tau$ 越小，权重差异越大；$\tau$ 越大，权重越平均。

还是看一个最小数值例子。设两个任务初始损失分别为：

$$
L_A(1)=0.3,\quad L_B(1)=0.6
$$

当前某一步损失变成：

$$
L_A(t)=0.12,\quad L_B(t)=0.42
$$

则下降比分别是：

$$
w_A(t)=0.12/0.3=0.4,\quad w_B(t)=0.42/0.6=0.7
$$

取 $\tau=0.5$，有：

$$
\exp(0.4/0.5)=\exp(0.8)\approx2.23
$$

$$
\exp(0.7/0.5)=\exp(1.4)\approx4.05
$$

归一化后：

$$
\lambda_A(t)=\frac{2.23}{2.23+4.05}\approx0.355,\quad
\lambda_B(t)=\frac{4.05}{2.23+4.05}\approx0.645
$$

于是总损失变成：

$$
L_{\text{total}} \approx 0.355\times0.12+0.645\times0.42 \approx 0.314
$$

这里有个关键点：动态权重不保证“眼前这个标量损失最小”，它优化的是“长期训练更平衡”。B 任务下降更慢，所以系统故意给它更大权重，避免模型只顾着继续压低已经学得比较顺的 A 任务。

### 3. 硬共享与软共享

硬参数共享是指多个任务直接共用同一组隐藏层，只在末端分头。这是现在最常见的设计，参数省，实现也简单。

软参数共享是指每个任务都有自己的网络，但通过正则项约束它们不要差太远：

$$
L = \sum_{k=1}^{K} L_k + \alpha \sum_{i<j}\|\theta_i-\theta_j\|_2^2
$$

这里正则项在鼓励不同任务模型“彼此靠近但不完全相同”。它比硬共享更灵活，但参数和维护成本更高。

---

## 代码实现

下面给一个可运行的玩具实现，演示“共享骨干 + 两个任务头 + 动态权重”的核心逻辑。代码只用 Python 标准库，重点是把训练信号如何组合讲清楚。

```python
import math

def softmax(xs, tau=1.0):
    exps = [math.exp(x / tau) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def dynamic_task_weights(initial_losses, current_losses, tau=0.5):
    ratios = [cur / init for init, cur in zip(initial_losses, current_losses)]
    weights = softmax(ratios, tau=tau)
    return ratios, weights

def total_loss(losses, weights):
    return sum(l * w for l, w in zip(losses, weights))

# 玩具例子：两个任务
initial = [0.3, 0.6]
current = [0.12, 0.42]

ratios, weights = dynamic_task_weights(initial, current, tau=0.5)
loss = total_loss(current, weights)

assert len(weights) == 2
assert abs(sum(weights) - 1.0) < 1e-9
assert weights[1] > weights[0]   # 第二个任务下降更慢，权重更大
assert loss > 0

print("ratios =", ratios)
print("weights =", weights)
print("weighted_loss =", loss)
```

如果换成 PyTorch，结构通常长这样：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedBackbone(nn.Module):
    def __init__(self, d_in=16, d_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class MultiTaskModel(nn.Module):
    def __init__(self, d_in=16, d_hidden=32, num_classes=3):
        super().__init__()
        self.backbone = SharedBackbone(d_in, d_hidden)
        self.cls_head = nn.Linear(d_hidden, num_classes)  # 分类头
        self.reg_head = nn.Linear(d_hidden, 1)            # 回归头

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.cls_head(feat)
        score = self.reg_head(feat)
        return logits, score

def dynamic_weights(init_losses, cur_losses, tau=0.5):
    ratios = torch.tensor([c / i for i, c in zip(init_losses, cur_losses)])
    return torch.softmax(ratios / tau, dim=0)

model = MultiTaskModel()
x = torch.randn(8, 16)
y_cls = torch.randint(0, 3, (8,))
y_reg = torch.randn(8, 1)

logits, score = model(x)
loss_cls = F.cross_entropy(logits, y_cls)
loss_reg = F.mse_loss(score, y_reg)

init_losses = [1.0, 1.0]  # 实际工程里记录第 1 个 epoch 或 warmup 后的值
weights = dynamic_weights(init_losses, [loss_cls.item(), loss_reg.item()])

loss = weights[0] * loss_cls + weights[1] * loss_reg
loss.backward()
```

真实工程例子可以看临床 NLP 中的 MT-Clinical-BERT：共享同一个 BERT 编码器，再挂多个轻量头分别处理实体识别、蕴含判断、语义相似度。这样部署时不需要为每个任务维护一个完整 BERT，推理成本明显下降。

---

## 工程权衡与常见坑

第一类坑是负迁移。最常见现象是：单任务模型都表现正常，一合并训练反而掉点。原因通常不是“多任务这个想法错了”，而是任务相关性不足、损失没校准、采样不平衡，或者梯度直接互相打架。

第二类坑是损失尺度不一致。分类交叉熵可能在 $[0,2]$ 左右，回归 MSE 可能轻易到十几。如果直接相加，大数值任务会主导更新。动态权重、损失归一化、GradNorm 之类方法，本质上都在处理这个问题。

第三类坑是共享层切分不合理。共享得太深，任务差异被压扁；共享得太浅，模型失去多任务带来的泛化收益。经验上，底层共享适合处理通用语法、局部纹理、基本语义；越靠近输出，越应该留给任务私有部分。

| 问题 | 现象 | 常见缓解策略 | 限制 |
| --- | --- | --- | --- |
| 梯度冲突 | 一个任务升，另一个降 | 动态权重、PCGrad、CAGrad | 训练更复杂，开销增加 |
| 损失尺度不一致 | 某任务长期主导 | 损失标准化、uncertainty weighting | 需要额外调参 |
| 共享过度 | 全部任务一起掉点 | 加私有层、Mixture-of-Experts、adapter | 参数量上升 |
| 采样不平衡 | 大数据任务碾压小任务 | round-robin、重采样、batch 配额 | 实现更复杂 |

真实工程里，BERT 类多任务系统往往不是追求每个任务都绝对最优，而是追求“一个统一系统足够强，部署更省”。MT-Clinical-BERT 的结论就很典型：单任务最优模型有时略强，但统一多任务模型在推理资源和系统维护上有明显优势。这是工程视角下很重要的权衡。

另一个常见误区是把“任务头”设计得过重。任务头的职责应该是把共享特征映射到具体标签，不应该再复制一套庞大主干。否则名义上是多任务，实际上又退回多个独立模型。

---

## 替代方案与适用边界

如果任务相关性不强，多任务学习未必是首选。常见替代方案有三类。

| 方案 | 共享程度 | 训练参数量 | 数据需求 | 适用场景 |
| --- | --- | --- | --- | --- |
| 完全独立训练 | 无共享 | 高 | 每任务都要足量数据 | 任务差异大，彼此干扰强 |
| 顺序微调 | 低 | 中 | 依赖任务顺序 | 先通用后专用的迁移流程 |
| 多任务联合训练 | 中到高 | 中 | 适合相关任务联合 | 任务相近，希望统一部署 |
| Adapter / Prompt / LoRA | 冻结大模型，仅训小模块 | 低 | 对小数据更友好 | 大模型多任务适配 |

顺序微调就是先在任务 A 上训练，再拿权重去任务 B。它本质是迁移学习，不是严格的联合多任务。优点是实现简单，缺点是后续任务可能覆盖前面学到的能力，也就是灾难性遗忘。

在大模型场景里，Adapter、LoRA、Prompt Tuning 往往更实际。它们的思路是冻结大部分基础模型，只训练很小的任务特定模块。对 GPT 类模型来说，这通常意味着不必解冻全部参数，只为每个任务增添一个小型适配器、软提示或低秩更新矩阵。这样做牺牲了一部分“全量共享优化”的自由度，但极大降低了显存和部署成本。

所以适用边界可以总结成一句话：如果任务共享输入模态、共享底层规律、并且你在乎统一部署和数据效率，多任务学习值得优先考虑；如果任务差异大、标签体系冲突、或单任务性能必须绝对最优，就应该考虑独立训练或参数高效微调。

---

## 参考资料

- VIFE, *Multi-Task Learning: How to Train One Model to Rule Them All*  
  https://vife.ai/blog/multi-task-learning-guide-shared-representations
- Mulyar, Uzuner, McInnes, *MT-clinical BERT: scaling clinical information extraction with multitask learning*  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8449623/
- MDPI, *Dynamic Tuning and Multi-Task Learning-Based Model for Multimodal Sentiment Analysis*  
  https://www.mdpi.com/2076-3417/15/11/6342
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*  
  https://arxiv.org/abs/1810.04805
- Wang et al., *Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning*  
  https://arxiv.org/abs/2303.02861
- Yu et al., *Gradient Surgery for Multi-Task Learning*  
  https://arxiv.org/abs/2001.06782
- APXML, *Multi-Task Fine-tuning for LLMs*  
  https://apxml.com/courses/fine-tuning-adapting-large-language-models/chapter-5-advanced-fine-tuning-strategies/multi-task-fine-tuning/
- APXML, *Multi-Adapter & Multi-Task PEFT Training*  
  https://apxml.com/courses/lora-peft-efficient-llm-training/chapter-5-peft-optimization-deployment/peft-multi-adapter-training
