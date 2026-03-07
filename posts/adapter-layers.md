## 核心结论

Adapter 层是一类参数高效微调方法。参数高效微调的意思是：**不改预训练模型主体，只训练少量新增参数**。  
Houlsby et al. 2019 的标准做法是在每个 Transformer 层的两个子层后各插入一个小瓶颈模块：一个接在多头注意力（MHA）子层后，一个接在前馈网络（FFN）子层后。这个模块通常写成：

$$
\mathrm{Adapter}(h)=h+W_{up} f(W_{down} h+b_{down})+b_{up}
$$

其中：

| 符号 | 含义 |
|---|---|
| $h$ | 原子层输出，维度为 $d$ |
| $W_{down}\in\mathbb{R}^{d_b\times d}$ | 降维矩阵，把 $d$ 维压到瓶颈维度 $d_b$ |
| $f$ | 激活函数，常见为 ReLU、GELU、Swish |
| $W_{up}\in\mathbb{R}^{d\times d_b}$ | 升维矩阵，把瓶颈表示投回原维度 |
| $b_{down}, b_{up}$ | 偏置项 |

如果忽略偏置，它就是最常见的简化形式：

$$
h' = h + W_{up} f(W_{down} h)
$$

这条残差公式有两个直接含义：

1. 主干路径仍然保留原模型结果。  
2. 新增分支只学习“任务相关的偏移量”，而不是重写整层表示。

因此，Adapter 的核心不是“加一个小网络”，而是**把任务适配限制在一个更小的可学习子空间中**。  
预训练模型继续提供通用能力，Adapter 只负责补充任务特定修正。

工程上最关键的一点，不是它参数少，而是它**初始时几乎不改原模型行为**。常见初始化方式是把上投影层 $W_{up}$ 初始化为 0，于是训练开始时：

$$
W_{up} f(W_{down} h)\approx 0
\quad\Rightarrow\quad
h' \approx h
$$

也就是说，刚插入的 Adapter 近似恒等映射。这能显著降低训练初期破坏预训练能力的风险，尤其适合小样本任务和已经很强的底座模型。

参数量方面，如果 hidden size 为 $d$，bottleneck size 为 $d_b=d/r$，其中 $r$ 是 reduction factor，那么单个 Adapter 的主要参数量近似为：

$$
d\cdot d_b + d_b\cdot d = 2dd_b = 2d\frac{d}{r}=\frac{2d^2}{r}
$$

以 $d=1024,\ r=16$ 为例：

$$
\frac{2\times 1024^2}{16}=131072
$$

这表示一个 Adapter 只增加约 13.1 万个主参数。  
如果同一 Transformer block 中放两个 Adapter，那么这一层大约增加：

$$
2\times 131072 = 262144
$$

对一整个大模型来说，这通常只占总参数的约 0.5% 到 8%，工程上常见落在 1% 到 3%。

如果只记一个部署结论，可以记下面这句：

| 方案 | 训练成本 | 多任务切换 | 推理延迟 | 是否易于合并回原权重 |
|---|---:|---:|---:|---:|
| Adapter | 低 | 强 | 略增 | 一般 |
| LoRA | 低 | 中 | 更低 | 强 |

所以，**Adapter 更适合多任务共享底座，LoRA 更适合单任务低延迟部署**。这不是谁绝对更先进，而是谁更符合你的部署目标。

---

## 问题定义与边界

Adapter 解决的问题不是“让模型凭空变强”，而是：

> **让同一个预训练模型以更低的训练、存储和运维成本适配多个任务。**

先看全量微调的问题。  
如果你对每个任务都做一次 full fine-tuning，通常会遇到四个成本：

| 成本类型 | 全量微调的典型问题 |
|---|---|
| 存储成本 | 每个任务都要保存一份完整模型 |
| 训练成本 | 需要更新大量参数，优化器状态也大 |
| 运维成本 | 多任务切换时要频繁加载整模型 |
| 能力覆盖风险 | 新任务训练可能破坏原有通用能力 |

这里常提到的“灾难性遗忘”，指的是：**模型在学习新任务后，旧任务能力被覆盖或明显下降**。  
Adapter 的思路是冻结基础模型，只为每个任务新增一组小模块。这样：

- 底座模型只有一份。
- 每个任务只保存一份很小的 Adapter 权重。
- 任务切换时换的是 Adapter，不是整个模型。

这使它特别适合“共享主干、任务很多”的场景。

它适用与不适用的边界可以先看这个表：

| 场景 | Adapter 是否合适 | 原因 |
|---|---|---|
| 多任务服务 | 很合适 | 统一底座，任务差异由小模块表达 |
| 小样本微调 | 合适 | 冻结主干后训练更稳，过拟合风险更低 |
| 显存受限训练 | 合适 | 可训练参数更少，优化器状态更小 |
| 单任务极致低延迟部署 | 不一定 | 前向多了额外计算路径 |
| 需要合并为单一权重文件 | 不一定 | LoRA 往往更直接 |
| 经常热切换多个租户任务 | 很合适 | Adapter 天然模块化 |

一个更直观的工程例子是企业多任务文本系统。  
假设你有一个共享底座模型，需要支持：

| 任务 | 输入 | 输出 |
|---|---|---|
| 法律文本分类 | 合同、判决书片段 | 风险类别 |
| 客服工单路由 | 用户工单文本 | 业务队列 |
| 医疗问答重排序 | 问题与候选答案 | 相关性分数 |

如果用全量微调，你要维护三份完整模型。  
如果用 Adapter，你只需要：

1. 保留一个基础模型。
2. 为三个任务各保存一组 Adapter。
3. 在线切任务时只切换小权重包。

这对多租户平台、A/B 实验系统、内部模型网关都很有价值。

这里还必须说明一个关键超参数：**reduction factor**。  
它定义为：

$$
r=\frac{d_{hidden}}{d_{bottleneck}}
$$

也就是：

$$
d_{bottleneck}=\frac{d_{hidden}}{r}
$$

含义很直接：

- $r$ 越大，瓶颈越窄，参数越省。
- $r$ 越小，瓶颈越宽，表达能力通常越强。

可以把它理解成“给任务偏移预留多少可学习空间”。  
下面这个实验结果很适合当边界示例：

| Reduction factor | Bottleneck 占 hidden 比例 | % PEFT 参数 | SacreBLEU（in-domain） |
|---|---:|---:|---:|
| 2 | 1/2 | 7.62% | 33.34 |
| 8 | 1/8 | 2.03% | 31.05 |
| 32 | 1/32 | 0.52% | 23.81 |

这个表的结论不是“越大越省越好”，而是：

> **Adapter 明确存在参数量与任务性能之间的折中。**

对多数工程任务，`reduction_factor=8` 或 `16` 往往是更稳妥的起点。  
如果你是第一次做 Adapter 实验，可以直接用下面这个经验表：

| 目标 | 推荐起点 |
|---|---|
| 想先跑通、少踩坑 | `r=16` |
| 数据量较小、怕不稳定 | `r=16` 或 `32` |
| 任务复杂、希望更强表达 | `r=8` |
| 明确知道任务很难 | 从 `r=8` 开始做对比实验 |

---

## 核心机制与推导

先看结构。  
一个 Houlsby Adapter 可以抽象为下面这条链路：

| 步骤 | 数学操作 | 作用 |
|---|---|---|
| LayerNorm（可选） | $\tilde{h}=\mathrm{LN}(h)$ | 稳定输入分布 |
| Down-project | $z=W_{down}\tilde{h}+b_{down}$ | 降维到瓶颈空间 |
| Activation | $u=f(z)$ | 引入非线性 |
| Up-project | $\Delta=W_{up}u+b_{up}$ | 升回原维度 |
| Residual Add | $h' = h + \Delta$ | 保留主路径，只加增量修正 |

如果不加 LayerNorm，最核心的部分仍然是中间三步：降维、激活、升维。

### 1. 为什么它能工作

可以从“低维任务偏移”理解。  
预训练模型已经学到大量通用语言知识，但具体下游任务往往只要求模型沿某些方向发生有限调整，而不是推倒重来。于是任务适配可以写成：

$$
h' = h + \Delta(h)
$$

其中 $\Delta(h)$ 表示该任务希望增加的修正项。  
全量微调的做法，本质上是允许模型在整个高维参数空间中改动。  
Adapter 则做了一个更强的假设：

> 这个任务修正项可以被压缩到一个更小的瓶颈子空间中。

于是：

$$
\Delta(h)\approx W_{up} f(W_{down} h)
$$

这个近似有两个重要效果：

| 效果 | 解释 |
|---|---|
| 参数压缩 | 不再直接学习完整大变换，而是在低维空间中学习 |
| 训练约束 | 更新空间更受限，通常更稳、更省数据 |

如果 hidden size 是 $d$，那么完整层内更新常常涉及 $O(d^2)$ 量级的参数；而 Adapter 只需要约：

$$
O\left(\frac{2d^2}{r}\right)
$$

这就是它“省参数”的来源。

### 2. 用一个玩具例子理解

假设某层输出是一个 1024 维向量。  
如果 `reduction_factor=16`，那么瓶颈维度就是：

$$
d_b=\frac{1024}{16}=64
$$

这时 Adapter 做的事可以直白地理解为：

1. 先把 1024 维表示压到 64 维。
2. 在这 64 维空间里学习任务相关变换。
3. 再投回 1024 维，加回原输出。

对新手来说，可以把它理解成一句话：

> 原模型负责“通用表示”，Adapter 负责“在少数任务相关方向上做补偿”。

这里不要把“64 维”理解成随意丢信息。  
更准确地说，模型是在学习：**哪些低维方向足以表达该任务需要的偏移**。

### 3. 为什么插在两个位置

Houlsby 结构通常在每个 Transformer block 里插两个 Adapter：

1. 注意力子层后一个。
2. FFN 子层后一个。

可以画成简化流程：

$$
x \rightarrow \mathrm{MHA} \rightarrow \mathrm{Adapter}_{attn}
\rightarrow \mathrm{FFN} \rightarrow \mathrm{Adapter}_{ffn}
$$

这么做的原因是，Transformer block 里本来就有两类不同功能的子层：

| 子层 | 主要作用 |
|---|---|
| MHA | 建模 token 与 token 之间的依赖关系 |
| FFN | 对每个 token 的表示做逐位置非线性变换 |

在两处都插 Adapter，意味着模型既能在“依赖关系建模”后做修正，也能在“特征变换”后做修正。  
表达能力通常比只插一处更强，但代价是前向路径更长。

AdapterHub 中，Houlsby 风格常见配置就是在两个位置都开启，这也是很多文档里提到的 `DoubleSeqBnConfig` 思路。

### 4. 为什么零初始化上投影很重要

如果把 $W_{up}$ 初始化为 0，那么训练刚开始时有：

$$
\Delta(h)=W_{up} f(W_{down} h)\approx 0
$$

因此：

$$
h' = h + \Delta(h) \approx h
$$

这意味着，**未训练的 Adapter 相当于一条近似空分支**。  
这对稳定性很关键，尤其在下面两种情况里更明显：

| 场景 | 为什么零初始化更重要 |
|---|---|
| 小样本任务 | 数据太少，随机扰动更容易把主干能力冲坏 |
| 强底座模型 | 预训练能力本来已经很好，训练初期不应大幅改动表示 |

很多初学者的问题就出在这里：  
他们把上下投影都用标准随机初始化，结果训练初期 loss 抖动很大、效果不升反降。根本原因往往不是“Adapter 不行”，而是初始化破坏了“先保守、再学习”的训练路径。

### 5. 参数量再精确一点怎么估算

如果考虑偏置，单个 Adapter 的参数量是：

$$
d\cdot d_b + d_b + d_b\cdot d + d
=
2dd_b + d_b + d
$$

代入 $d_b=d/r$，得到：

$$
2d\frac{d}{r} + \frac{d}{r} + d
$$

当 $d$ 很大时，后面的偏置项通常远小于主项，所以工程上常近似成：

$$
\frac{2d^2}{r}
$$

如果一个 Transformer block 中有两个 Adapter，则近似翻倍：

$$
\frac{4d^2}{r}
$$

这也是为什么你在模型规模大、层数多时，虽然“每个 Adapter 不大”，但整模型加起来仍然会形成可观的额外参数包。

### 6. MAM Adapter 是什么

MAM Adapter 可以看成 Adapter 家族中的混合方案。  
它把两种 PEFT 思路组合在一起：

| 组成部分 | 作用 |
|---|---|
| Prefix Tuning | 在注意力计算前引入可训练前缀表示 |
| Bottleneck Adapter | 在层内做低维增量变换 |

它适合的场景不是“默认首选”，而是：

- 你已经试过标准 Adapter。
- 你也试过 Prefix Tuning。
- 两者单独使用都不够好。
- 你愿意接受更复杂的调参空间。

对初学者或第一个生产版本，更合理的路径通常仍然是：  
**先把标准 Houlsby Adapter 跑通，再考虑混合方法。**

---

## 代码实现

下面分三部分给出代码：

1. 一个纯 `numpy` 的最小实现，用来验证公式本身。  
2. 一个可直接运行的 `PyTorch` 版本，演示冻结底座、只训练 Adapter。  
3. 一个基于 Hugging Face AdapterHub 风格接口的工程示例。

### 1. 最小可运行示例：验证恒等映射与非零修正

这段代码只依赖 `numpy`，可以直接运行。  
它演示两件事：

- 上投影零初始化时，Adapter 初始等价于恒等映射。
- 一旦上投影变成非零，输出就会偏离输入。

```python
import numpy as np

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    ))

class Adapter:
    def __init__(self, d_hidden, reduction_factor, seed=42):
        assert d_hidden % reduction_factor == 0, "d_hidden 必须能整除 reduction_factor"
        d_bottleneck = d_hidden // reduction_factor
        rng = np.random.default_rng(seed)

        self.w_down = rng.normal(0.0, 0.02, size=(d_bottleneck, d_hidden))
        self.b_down = np.zeros(d_bottleneck, dtype=np.float64)

        # 关键：零初始化上投影，保证初始近似恒等映射
        self.w_up = np.zeros((d_hidden, d_bottleneck), dtype=np.float64)
        self.b_up = np.zeros(d_hidden, dtype=np.float64)

    def forward(self, h):
        z = self.w_down @ h + self.b_down
        z = gelu(z)
        delta = self.w_up @ z + self.b_up
        return h + delta

if __name__ == "__main__":
    h = np.array([0.2, -0.1, 0.5, 1.0], dtype=np.float64)
    adapter = Adapter(d_hidden=4, reduction_factor=2)

    out = adapter.forward(h)
    print("initial out:", out)

    # 初始时应基本等于输入
    assert np.allclose(out, h), "零初始化上投影时，输出应与输入一致"

    # 模拟训练后出现非零修正
    adapter.w_up[0, 0] = 0.3
    adapter.w_up[2, 1] = -0.2

    out2 = adapter.forward(h)
    print("trained out:", out2)

    assert not np.allclose(out2, h), "上投影非零后，Adapter 应产生修正"
    print("ok")
```

如果你刚接触 Adapter，这段代码要重点看三行：

```python
self.w_up = np.zeros((d_hidden, d_bottleneck))
delta = self.w_up @ z + self.b_up
return h + delta
```

它们直接对应了“零初始化上投影”和“残差加回原输出”这两个核心机制。

### 2. PyTorch 示例：冻结底座，只训练 Adapter

下面这个例子更接近真实训练代码。  
它做了四件事：

1. 定义一个 Houlsby 风格 Adapter 模块。  
2. 定义一个简单的主干层。  
3. 冻结主干参数。  
4. 验证只有 Adapter 参数会参与训练。

```python
import torch
import torch.nn as nn

class HoulsbyAdapter(nn.Module):
    def __init__(self, d_model, reduction_factor=16):
        super().__init__()
        assert d_model % reduction_factor == 0, "d_model 必须能整除 reduction_factor"
        d_bottleneck = d_model // reduction_factor

        self.down = nn.Linear(d_model, d_bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(d_bottleneck, d_model)

        # 关键初始化：上投影置零，初始近似恒等映射
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))

class ToyBlock(nn.Module):
    def __init__(self, d_model=8):
        super().__init__()
        self.core = nn.Linear(d_model, d_model)
        self.adapter = HoulsbyAdapter(d_model=d_model, reduction_factor=4)

    def forward(self, x):
        x = self.core(x)
        x = self.adapter(x)
        return x

def freeze_backbone_except_adapter(model):
    for name, param in model.named_parameters():
        if "adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

if __name__ == "__main__":
    torch.manual_seed(0)

    model = ToyBlock(d_model=8)
    freeze_backbone_except_adapter(model)

    # 检查哪些参数可训练
    trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]

    print("trainable params:")
    for name, shape in trainable:
        print(f"  {name}: {tuple(shape)}")

    print("frozen params:")
    for name in frozen:
        print(f"  {name}")

    x = torch.randn(2, 8)
    y = torch.randn(2, 8)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )
    criterion = nn.MSELoss()

    for step in range(5):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        print(f"step={step}, loss={loss.item():.6f}")
```

这个例子虽然简单，但已经覆盖了真实工程里的两个关键动作：

| 动作 | 原因 |
|---|---|
| 冻结 backbone | 保证训练目标真的是 PEFT，而不是全量微调 |
| 单独收集可训练参数给 optimizer | 避免无效优化器状态占内存 |

如果你把这段代码用于更大的模型，基本模式不会变，只是 `ToyBlock` 会被替换成真实 Transformer block。

### 3. 工程接口示例：AdapterHub 风格

如果你使用 Hugging Face 生态中的 AdapterHub 风格接口，代码通常更短：

```python
from adapters import AutoAdapterModel, BnConfig

model = AutoAdapterModel.from_pretrained("bert-base-uncased")

config = BnConfig(
    mh_adapter=True,
    output_adapter=True,
    reduction_factor=16,
    non_linearity="swish"
)

model.add_adapter("task_cls", config=config)
model.train_adapter("task_cls")
```

这里几个配置项要知道各自控制什么：

| 配置项 | 含义 |
|---|---|
| `mh_adapter=True` | 在注意力子层后插 Adapter |
| `output_adapter=True` | 在 FFN 子层后插 Adapter |
| `reduction_factor=16` | 设置瓶颈压缩比例 |
| `non_linearity="swish"` | 指定激活函数 |

如果你的系统要管理多个任务，通常还会这样组织：

| 对象 | 建议管理方式 |
|---|---|
| 基础模型 | 按模型版本单独管理 |
| Adapter 权重 | 每任务单独保存 |
| 配置文件 | 显式记录 `reduction_factor`、激活函数、插入位置 |
| 推理服务 | 按请求选择对应 Adapter 挂载 |

### 4. 新手最容易漏掉的工程细节

下面这四件事，缺一项都可能让实验结果偏掉：

| 检查项 | 正确做法 | 常见错误 |
|---|---|---|
| 冻结底座 | 主模型参数 `requires_grad=False` | 忘记冻结，变成全量微调 |
| 初始化 | 上投影置零 | 上下投影全随机初始化 |
| 版本绑定 | 底座版本和 Adapter 版本同时记录 | Adapter 训练自 A 模型，却挂到 B 模型上 |
| 单独保存 | 只保存 Adapter 权重包 | 把底座和任务权重混写，后续难维护 |

如果是线上服务，还要再加一条：

| 检查项 | 建议 |
|---|---|
| 延迟评估 | 先做 profiling，再决定是否上生产 |

因为 Adapter 的额外计算虽然小，但它是真实存在的，不应靠直觉判断。

---

## 工程权衡与常见坑

Adapter 的优势很明确，但代价不是假的。  
最实用的判断方式，不是看论文标题，而是看下面这张表：

| 关注点 | Houlsby Adapter | 常见规避方式 |
|---|---|---|
| 训练参数量 | 低，常见 1% 到 3% | 调大 `reduction_factor` |
| 优化器状态 | 显著变小 | 只把 Adapter 参数交给 optimizer |
| 多任务切换 | 很方便 | 每个任务单独管理 adapter |
| 推理延迟 | 有额外层，前向更长 | 上线前做 profiling |
| 权重合并 | 不自然 | 单任务部署时可优先考虑 LoRA |
| 训练稳定性 | 强依赖初始化 | 上投影置零，保持初始近似恒等映射 |

最常见的坑可以归纳为五类。

### 1. `reduction_factor` 设得过大

参数确实省了，但瓶颈太窄时，模型根本学不到足够的任务偏移。  
最典型的表现是：

- 训练 loss 能降一点，但任务指标上不去。
- 换更长训练轮数也没明显帮助。
- 与全量微调或 LoRA 相比差距很大。

这通常不是“Adapter 失效”，而是**你给它的瓶颈空间太小**。  
如果出现这种情况，优先把 `r=32` 调回 `16` 或 `8`，而不是先盲目改学习率。

### 2. 忘记冻结底座参数

这是最常见的实现错误之一。  
如果底座没有冻结，结果会变成“带 Adapter 的全量微调”。这会带来三类后果：

| 后果 | 具体表现 |
|---|---|
| 显存变大 | 主干参数也要保留梯度与优化器状态 |
| 训练行为变化 | 不再是受限更新，而是全空间更新 |
| 实验结论失真 | 你以为自己在比较 Adapter，实际上不是 |

所以每次训练前都要检查可训练参数列表，而不是只看代码“好像冻结了”。

### 3. 初始化不对

如果上投影不是 0 初始化，训练开始时每层输出都会立即被额外分支扰动。  
这在小数据集上尤其危险，因为本来就没有足够数据去纠正错误方向。

可以把它理解为：

- 正确初始化：先保持原模型行为，再逐步学习偏移。
- 错误初始化：一开始就强行改动各层表示。

在实践里，这个差别往往比新手预想得更大。

### 4. 忽略推理延迟

Adapter 的训练成本低，不等于线上完全没有代价。  
它和 LoRA 的一个重要差异是：**Adapter 前向时多了一条真实模块路径**。

举一个简单估算。  
假设原模型单请求延迟是 80ms。加入多层 Adapter 后，延迟变成 84ms。  
这 4ms 在不同业务里的意义完全不同：

| 场景 | 4ms 的含义 |
|---|---|
| 高 QPS 在线检索/广告排序 | 可能很贵 |
| 企业内部低频工具 | 往往可以接受 |
| 多任务统一网关 | 常常值得换取切换便利 |
| 移动端本地部署 | 可能要重新评估 |

所以判断标准不是“4ms 大不大”，而是：**你的业务是否愿意用这点延迟换多任务管理能力。**

### 5. 版本兼容性管理不足

这是工程里比算法本身更常见的坑。  
因为 Adapter 不是独立运行的，它依赖底座模型结构。  
如果底座版本变了，哪怕只是 tokenizer、hidden size、层命名规则或 checkpoint 结构变化，也可能导致：

- 不能加载。
- 能加载但效果异常。
- 配置错位却不报错。

因此建议至少记录下面这组元信息：

| 元信息 | 必须记录吗 |
|---|---|
| base model 名称 | 是 |
| base model 版本或 commit | 是 |
| adapter 训练数据版本 | 是 |
| reduction factor | 是 |
| 插入位置配置 | 是 |
| 激活函数 | 建议 |
| 训练脚本版本 | 建议 |

一句话总结 Adapter 的工程判断标准：

> **你不是在判断它“先进不先进”，而是在判断它是否值得用一点推理开销换取更便宜训练和更清晰的多任务管理。**

---

## 替代方案与适用边界

最常拿来和 Adapter 对比的是 LoRA 和 IA3。  
三者都属于 PEFT，但它们约束更新的方式不同。

### 1. LoRA

LoRA 的思路是：**不显式插入一个额外层，而是在原线性变换上学习一个低秩更新量**。  
如果原权重是 $W$，LoRA 近似学习：

$$
W' = W + BA
$$

其中：

- $A\in\mathbb{R}^{r\times d_{in}}$
- $B\in\mathbb{R}^{d_{out}\times r}$
- $r$ 是低秩维度，通常远小于原矩阵维度

它的优势在于训练后常可把更新合并回原权重，因此推理时不一定保留额外结构。  
所以 LoRA 经常更适合：

- 单任务部署
- 对延迟更敏感的服务
- 希望导出单一权重文件的场景

### 2. IA3

IA3 更激进。  
它通常不是学习完整小矩阵，而是学习若干缩放向量，对注意力或 FFN 中的中间表示做按维缩放。  
因此它参数更少，但表达能力也更受限制。

可以把三者做一个直接对比：

| 方法 | 更新方式 | 可训练参数 | 推理架构变化 | 最佳场景 |
|---|---|---|---|---|
| Houlsby Adapter | 显式插入瓶颈模块 | 约 0.5% 到 8% | 有额外层 | 多任务、动态切换 |
| LoRA | 在线性层上学低秩增量 | 通常更低 | 可合并，结构变化小 | 单任务、低延迟部署 |
| IA3 | 学习缩放向量 | 更低 | 通常可忽略 | 参数预算极紧 |

### 3. 怎么选

如果你只想记一个选择规则，可以用下面这张表：

| 需求 | 更优先考虑 |
|---|---|
| 同一底座服务多个任务 | Adapter |
| 单任务上线且延迟敏感 | LoRA |
| 需要导出单一权重文件 | LoRA |
| 参数预算极度严格 | IA3 |
| 第一次做 PEFT、想先跑稳 | LoRA 或标准 Adapter |
| 需要显式模块化任务切换 | Adapter |

也可以写成更口语化但仍然准确的三句话：

- **频繁切任务，用 Adapter。**
- **只跑一个任务且追求低延迟，用 LoRA。**
- **连 LoRA 都嫌大，再考虑 IA3。**

### 4. MAM Adapter 的边界

MAM Adapter 这类混合方法通常不是“默认更优”，而是“默认更复杂”。  
它的适用边界更窄，通常满足下面几个条件时才值得尝试：

| 条件 | 是否建议尝试 MAM |
|---|---|
| 标准 Houlsby Adapter 效果不够 | 可以考虑 |
| Prefix Tuning 单独效果也不够 | 可以考虑 |
| 你能接受更复杂调参 | 可以考虑 |
| 你是第一次做 PEFT | 不建议一开始就上 |

因此对大多数团队来说，合理路径仍然是：

1. 先用标准 Houlsby Adapter 建立基线。
2. 再和 LoRA、IA3 做横向比较。
3. 只有标准方法都无法满足需求时，再考虑混合方案。

这比一开始就把方案堆复杂更符合工程收益。

---

## 参考资料

下面按“论文原始来源、方法文档、工程实现、实验参考”四类整理，便于按深度阅读。

| 类型 | 资料 | 用途 |
|---|---|---|
| 原始论文 | Houlsby et al., 2019, *Parameter-Efficient Transfer Learning for NLP* | Adapter 方法起点 |
| 方法文档 | AdapterHub, *Adapter Methods* | 看不同 Adapter 结构 |
| 方法文档 | AdapterHub, *Method Combinations* | 看组合方法，如 MAM |
| 工程示例 | AdapterHub, *ukp/gpt2_nli_rte_houlsby* | 看现成 Houlsby 适配器 |
| 工程文档 | NVIDIA NeMo, *Supported PEFT Methods* | 看工业框架中的 PEFT 支持 |
| 实验参考 | Ranaldi et al., 2024 Findings of NAACL | 看 reduction factor 的参数-性能折中 |

具体链接如下：

- Houlsby et al., 2019, *Parameter-Efficient Transfer Learning for NLP*（PMLR）  
  https://proceedings.mlr.press/v97/houlsby19a.html
- AdapterHub 文档，*Adapter Methods*  
  https://docs.adapterhub.ml/methods.html
- AdapterHub 文档，*Method Combinations*（含 MAMConfig）  
  https://docs.adapterhub.ml/method_combinations.html
- AdapterHub 示例，*ukp/gpt2_nli_rte_houlsby*  
  https://adapterhub.ml/adapters/ukp/gpt2_nli_rte_houlsby/
- NVIDIA NeMo，*Supported PEFT Methods*  
  https://docs.nvidia.com/nemo-framework/user-guide/24.12/sft_peft/supported_methods.html
- Ranaldi et al., 2024 Findings of NAACL，reduction factor 对比实验  
  https://aclanthology.org/2024.findings-naacl.263.pdf

如果你只想按最低成本读完，建议顺序是：

1. 先读 Houlsby 2019，建立结构与目标。  
2. 再读 AdapterHub 方法文档，看工程配置项。  
3. 最后看 reduction factor 对比实验，理解参数预算与性能边界。
