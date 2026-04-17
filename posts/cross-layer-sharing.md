## 核心结论

Cross-Layer Sharing，中文通常叫“跨层共享参数”，核心定义是：**多个执行层复用同一套权重，保留网络深度，但不再为每一层单独分配参数。** 这里的“权重”就是模型里真正要学习和存储的数字。

这件事最容易被误解的地方在于，它**不是把 12 层模型缩成 3 层模型**，而是让 12 次计算里有些层使用同一套参数。对新手来说，可以直接记成一句话：**层数可以不变，参数份数可以变少。**

最小公式就是：

$$
h^{(l)} = f\!\left(h^{(l-1)};\theta_{g(l)}\right)
$$

它的意思很直接：第 $l$ 次层计算，把上一层输出 $h^{(l-1)}$ 送进某个变换函数 $f$，但这次用的是第 $g(l)$ 组参数 $\theta_{g(l)}$。如果多个层映射到同一个 $g(l)$，它们就共享参数。

一个新手版理解如下。

- 标准 BERT：12 层 = 12 份 block 参数
- 跨层共享：12 层 = 1 份或少数几份 block 参数重复使用

“block”就是 Transformer 的一层计算单元，通常包含注意力和前馈网络两部分。

下面这张表先把差异压缩成最核心的对比。

| 方案 | 执行层数 | 参数组数 | 参数规模趋势 | 前向计算次数 |
| --- | --- | --- | --- | --- |
| 标准分层 | 12 | 12 | 随层数线性增长 | 12 次 |
| 每 4 层共享一次 | 12 | 3 | 明显下降 | 12 次 |
| 全共享 | 12 | 1 | 最低 | 12 次 |

所以结论很明确：**跨层共享主要解决“模型太大”，不直接解决“算得太慢”。** 如果你的瓶颈是模型文件大小、显存、加载成本，它很有价值；如果你的目标是把单次请求延迟压到最低，它通常不是第一选择。

---

## 问题定义与边界

要理解 Cross-Layer Sharing，必须先分清三个概念：**执行层数、参数组数、实际参数规模**。这三者看起来相关，但并不相等。

- 执行层数：前向传播实际走了多少层。
- 参数组数：真正独立存在的参数集合有多少份。
- 实际参数规模：总共需要训练、保存、加载多少参数。

设：

| 符号 | 含义 |
| --- | --- |
| $L$ | 总执行层数 |
| $N$ | 每组共享覆盖的层数 |
| $G=\lceil L/N \rceil$ | 共享组数 |

例如，一个 12 层 Transformer，如果每 4 层共享一次参数，那么：

- 执行层数仍然是 12
- 每组共享 4 层，所以参数组数变成 3
- 如果每个 block 参数量近似相同，总参数会从“12 份 block”下降到“3 份 block”

这就是为什么跨层共享常被用于“压参数”而不是“降计算”。

一个玩具例子可以这样看。假设你有 12 道作业题，每一道都要批改一次。“批改规则”相当于模型参数，“批改动作”相当于一次层计算。标准做法是 12 位老师各自有一套规则；共享做法是 3 位老师轮流批，甚至 1 位老师反复批 12 次。**题还是 12 道，批改次数没变，但规则手册变少了。**

因此，Cross-Layer Sharing 的问题边界也很清楚。

| 场景 | 是否适合 | 原因 |
| --- | --- | --- |
| 压缩模型参数 | 适合 | 直接减少独立参数份数 |
| 移动端部署 | 适合 | 模型包更小，加载和存储压力更低 |
| 低显存训练 | 适合 | 参数相关显存占用更低 |
| 极致低延迟推理 | 不一定适合 | 前向仍需逐层执行 |
| 强层间差异建模任务 | 可能不适合 | 共享会削弱不同层的独立性 |

这里的“层间差异建模”可以理解成：不同深度的层希望学出明显不同的功能。如果所有层都被迫使用同一套权重，这种差异就会受限。

---

## 核心机制与推导

跨层共享的本质是把多个层映射到同一组参数，只改变“参数怎么分配”，不改变“计算路径有多深”。

设总层数为 $L$，每 $N$ 层共享一组参数，那么共享组数是：

$$
G=\left\lceil \frac{L}{N}\right\rceil
$$

第 $l$ 层的计算可以写成：

$$
h^{(l)} = f\!\left(h^{(l-1)};\theta_{g(l)}\right), \quad g(l)=\left\lceil \frac{l}{N}\right\rceil
$$

其中：

- $h^{(l)}$ 是第 $l$ 层输出
- $f$ 是一层 Transformer block 的计算
- $\theta_{g(l)}$ 是第 $g(l)$ 组共享参数

不共享时，参数量近似为：

$$
P_{\text{base}} \approx P_{\text{emb}} + L\cdot P_{\text{block}}
$$

共享后，参数量近似为：

$$
P \approx P_{\text{emb}} + \sum_{g=1}^{G} P_{\text{block}}(\theta_g)
$$

如果每组 block 的规模接近，还可以近似写成：

$$
P \approx P_{\text{emb}} + G\cdot P_{\text{block}}
$$

这里的 $P_{\text{emb}}$ 是 embedding 参数，指词向量、位置向量这类前端表示层；$P_{\text{block}}$ 是每个 Transformer block 的参数量。

看一个最小数值例子。假设：

- 总层数 $L=12$
- 每层 block 约 1M 参数
- 暂时忽略 embedding

那么：

| 共享方式 | 组数 $G$ | block 参数总量 |
| --- | --- | --- |
| 不共享 | 12 | 12M |
| 每 4 层共享一次 | 3 | 3M |
| 每 6 层共享一次 | 2 | 2M |
| 全共享 | 1 | 1M |

但要特别强调：**前向仍然要走 12 次 block 计算。** 所以它对参数量的压缩非常明显，对 FLOPs 的压缩却通常不成比例。FLOPs 是浮点运算次数，可以粗略理解为模型实际做了多少乘加运算。

可以用一个机制示意图表示：

```text
输入 x
  -> block(theta_1) -> h(1)
  -> block(theta_1) -> h(2)
  -> block(theta_1) -> h(3)
  -> block(theta_1) -> h(4)
  -> block(theta_2) -> h(5)
  -> block(theta_2) -> h(6)
  -> ...
  -> 输出
```

如果是全共享，图会更简单：

```text
输入 x -> 同一个 block(theta) -> 同一个 block(theta) -> ... -> 输出
```

看起来像“重复调用同一个函数”。这也是它和 Universal Transformer、RNN 式递归思想容易被放在一起讨论的原因：**参数不一定随深度增加而增加。**

不过，重复调用同一套参数并不等于完全重复的表示。因为每次输入的 hidden state 不一样，函数虽然相同，进入函数的状态已经变化，所以每一层输出仍然可能不同。这一点很重要，否则会误以为“全共享后 12 层没有意义”。

---

## 代码实现

工程实现里，跨层共享通常不是“复制出 12 个不同层对象”，而是“创建少量层对象，在 forward 里重复调用”。“对象”这里可以理解成框架里的一个可训练模块实例。

先看一个可运行的 Python 玩具实现。它不依赖深度学习框架，只演示“12 次执行，但只用 3 组参数”的映射逻辑。

```python
import math

class SharedLinearBlock:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        return self.weight * x + self.bias

class CrossLayerSharedToyModel:
    def __init__(self, num_layers=12, share_every=4):
        self.num_layers = num_layers
        self.share_every = share_every
        self.num_groups = math.ceil(num_layers / share_every)

        # 这里只创建少量 block，模拟真正被复用的参数组
        self.blocks = [
            SharedLinearBlock(weight=i + 2, bias=i)
            for i in range(self.num_groups)
        ]

    def group_id(self, layer_idx):
        return layer_idx // self.share_every

    def forward(self, x):
        trace = []
        for l in range(self.num_layers):
            g = self.group_id(l)
            x = self.blocks[g](x)
            trace.append((l, g, x))
        return x, trace

model = CrossLayerSharedToyModel(num_layers=12, share_every=4)
y, trace = model.forward(1)

assert model.num_groups == 3
assert trace[0][1] == 0
assert trace[3][1] == 0
assert trace[4][1] == 1
assert trace[8][1] == 2
assert len(trace) == 12

# 验证“层数没变”
assert sum(1 for _ in trace) == 12

# 验证“参数组变少”
assert len(model.blocks) == 3
```

这个例子里：

- `num_layers=12` 表示前向执行 12 次
- `share_every=4` 表示每 4 层共享一组参数
- `blocks` 只有 3 个对象，说明独立参数组只有 3 份

如果换成 PyTorch 风格，结构通常接近下面这样：

```python
import math
import torch.nn as nn

class SharedBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x

class CrossLayerSharedTransformer(nn.Module):
    def __init__(self, hidden_size=128, num_layers=12, share_every=4):
        super().__init__()
        self.num_layers = num_layers
        self.share_every = share_every
        self.num_groups = math.ceil(num_layers / share_every)

        # 只创建 num_groups 个 block，不是 num_layers 个
        self.blocks = nn.ModuleList(
            [SharedBlock(hidden_size) for _ in range(self.num_groups)]
        )

    def forward(self, x):
        for l in range(self.num_layers):
            g = l // self.share_every
            x = self.blocks[g](x)   # 同一个参数对象被重复调用
        return x
```

这里最关键的一点不是语法，而是**参数对象是否真的被复用**。如果你误写成“按层复制模块”，那就只是长得像共享，实际并没有减少参数。

真实工程例子可以看 ALBERT。它是 BERT 的一个压缩版本，核心思路之一就是跨层共享参数。在 Hugging Face 的配置里，常见字段包括：

| 字段 | 含义 |
| --- | --- |
| `num_hidden_layers` | 总执行层数 |
| `num_hidden_groups` | 共享组数 |
| `hidden_size` | 隐状态维度 |
| `intermediate_size` | 前馈网络中间层维度 |

很多新手第一次看会把 `num_hidden_groups` 误认为“层数”，这是错的。`num_hidden_layers` 决定要执行多少次 block，`num_hidden_groups` 决定独立参数有几组。比如 `num_hidden_layers=12, num_hidden_groups=1`，表示 12 层全共享。

---

## 工程权衡与常见坑

跨层共享的收益和代价都很直接，不需要神化。

| 维度 | 收益 | 代价 |
| --- | --- | --- |
| 参数量 | 明显减少 | 无 |
| 模型文件大小 | 更小 | 无 |
| 参数相关显存 | 更低 | 无 |
| 训练/推理计算量 | 不一定同比下降 | 前向仍需逐层执行 |
| 表达能力 | 无法保证不损失 | 层间多样性下降 |
| 实现复杂度 | 无 | checkpoint 和配置更复杂 |

最常见的错误理解是：**“共享后 12 层只算 1 层。”** 正确说法是：**12 层都要算，只是重复使用同一套权重。**

为什么计算量不一定明显下降？因为大多数 Transformer 的主要计算成本来自矩阵乘法，而这些乘法仍然会在每一层执行。你减少的是“需要存和更新多少套参数”，不是“需要调用多少次层函数”。

再看几个常见坑。

1. 误把共享参数当成少计算  
很多人在压缩报告里只盯参数量下降百分比，然后直接推断延迟也会同比下降。这通常不成立。参数量从 12M 变 3M，不代表推理时间从 12ms 变 3ms。

2. 误把 `num_hidden_groups` 当成 `num_hidden_layers`  
这会直接导致配置理解错误。尤其在复现实验时，别人说“12-layer ALBERT”，你却按 3 层去实现，结论会完全错位。

3. 修改结构后没检查 checkpoint 兼容性  
原先是 12 个独立层的 checkpoint，现在改成 3 组共享，权重键名和组织方式都会变。结果通常是：
   - key 不匹配
   - 某些层找不到参数
   - 某些参数维度对不上

4. 共享过强导致性能下降  
如果任务非常依赖不同深度学出不同功能，比如一些复杂理解或生成任务，全共享可能压得过头。此时你看到的不是“训练失败”，而是“能训练，但上限偏低”。

一个实际工程场景是移动端中文文本分类。假设你要把模型放进 App 里做本地推理，约束可能是：

- 模型包大小不能太大
- 首次加载时间要可控
- 设备内存有限

这时，ALBERT 风格共享经常比标准 BERT 更合理，因为你优先要解决“能不能装下、能不能加载”。但如果同一个任务放在服务端，且目标是把 P99 延迟压到最低，那么更小层数的轻量结构、蒸馏模型或量化模型通常更直接。

比较稳妥的验证流程如下。

| 步骤 | 做法 | 目的 |
| --- | --- | --- |
| 1 | 先跑独立层基线 | 拿到性能上界 |
| 2 | 尝试全共享 | 测试最强压缩比 |
| 3 | 尝试分组共享 | 寻找性能与参数平衡点 |
| 4 | 检查 checkpoint 加载 | 排除实现问题 |
| 5 | 对比参数量、显存、延迟、指标 | 避免只看单一维度 |

这个流程的核心思想是：**先确认你在优化哪个指标，再决定共享强度。** 如果你关心的是参数预算，先试全共享没问题；如果你关心的是精度，通常要从“少量组共享”开始看退化幅度。

---

## 替代方案与适用边界

跨层共享是模型压缩的一种手段，但不是唯一手段。它主要压的是“参数份数”，而不是一定压“每次计算成本”。因此，在工程上通常要和蒸馏、剪枝、量化一起比较。

先看总表。

| 方法 | 核心思路 | 主要优化对象 | 对延迟的帮助 | 常见代价 |
| --- | --- | --- | --- | --- |
| 跨层共享参数 | 多层复用同一套权重 | 参数量、模型大小 | 有限或不稳定 | 表达能力可能下降 |
| 知识蒸馏 | 用大模型教小模型 | 模型规模与泛化 | 通常较好 | 训练流程更复杂 |
| 剪枝 | 去掉不重要连接或结构 | 参数量、部分计算 | 取决于实现 | 稀疏加速不一定落地 |
| 量化 | 用更低比特表示权重/激活 | 存储、带宽、算力 | 通常较好 | 精度和部署兼容性风险 |

用一句更白话的话区分：

- 共享参数：同一双手重复做很多步
- 剪枝/量化：把这双手变得更轻、更省电
- 蒸馏：直接训练一个更小、做法更简单的学生模型

再看真实工程例子。假设你做的是移动端中文文本分类：

- 如果设备内存和模型包大小很紧，优先考虑 ALBERT 风格共享是合理的。
- 如果服务端追求极低推理延迟，共享不一定是最佳选择，因为层计算次数还在。
- 如果你既想保留较好性能，又想显著压缩体积，常见路线是“蒸馏 + 量化”，有时再加上适度共享。

所以选择原则可以压缩成三句：

- **参数预算紧，优先看共享。**
- **延迟敏感，优先看轻量结构或量化。**
- **性能优先但还要压缩，可考虑蒸馏 + 共享组合。**

它的适用边界也要说清楚。跨层共享最适合的是“模型能不能放进去、能不能训得动、能不能省下参数相关资源”这类问题；它不适合被当成“万能加速器”。如果你的目标是吞吐和延迟，直接减少层数、改用更小 hidden size、采用蒸馏或量化，通常更有效。

---

## 参考资料

原理来源：

- ALBERT 论文（OpenReview / ICLR 2020）：https://openreview.net/forum?id=H1eA7AEtvS
- ALBERT 论文 PDF：https://openreview.net/pdf?id=H1eA7AEtvS
- Universal Transformers 官方论文页：https://research.google/pubs/universal-transformers/

工程落地来源：

- Google Research 官方实现：https://github.com/google-research/albert
- Hugging Face Transformers ALBERT 文档：https://huggingface.co/docs/transformers/model_doc/albert
