## 核心结论

PyTorch 的自动求导机制，本质是一个**反向自动微分引擎**。反向自动微分可以理解成：前向计算时先把“这一步是怎么来的”记下来，反向时再从结果往输入倒推梯度。它不是提前把整个模型翻译成静态计算图，而是在你执行前向代码的同时，动态构造一张有向无环图（DAG，白话就是“节点有方向连接、不会绕回自己的依赖图”）。

这件事成立的核心原因只有两点：

1. 前向阶段，凡是参与梯度计算的操作，都会在结果张量上挂一个 `grad_fn`，表示“这个张量是由哪个求导规则产生的”。
2. 后向阶段，Autograd 从标量损失出发，沿着 `grad_fn` 链按拓扑顺序回溯，对每条边应用链式法则，把各条路径上的梯度贡献累加到叶子张量上。

链式法则是整个机制的数学基础。对任意中间变量 $y_i$，有：

$$
\frac{\partial L}{\partial x} = \sum_i \frac{\partial L}{\partial y_i}\frac{\partial y_i}{\partial x}
$$

这里的“累加”非常重要，因为一个参数可能通过多条路径影响最终损失，Autograd 必须把所有路径的贡献加总，而不是只算其中一条。

PyTorch 选择动态图而不是静态图，直接带来一个工程结果：**写什么就算什么**。`if`、`for`、递归、不同 batch 走不同分支，都可以直接用 Python 控制流表达；本次前向真正执行过的路径，才会被记录并参与这次反向。

一个最小玩具例子就能说明问题。令：

$$
y = \sum \exp(x)
$$

则每个分量满足：

$$
\frac{\partial y}{\partial x_j} = \exp(x_j)
$$

所以如果 `x = [0.5, 0.75]`，反向后 `x.grad` 就应该是 `[e^{0.5}, e^{0.75}]`。PyTorch 做的事情，不是“猜”出这个结果，而是前向先记录 `exp` 和 `sum` 对应的求导节点，反向再把局部导数按顺序乘起来并累加。

---

## 问题定义与边界

讨论 PyTorch 自动求导，先要明确“它到底对谁求导、在什么条件下记录图”。

最关键的边界是**叶子张量**。叶子张量可以理解成：这块数据不是由别的可求导运算算出来的，而是图的起点，通常就是模型参数或你手工创建的输入张量。只有叶子张量的 `.grad` 会默认被填充。中间张量也可能参与求导，但默认不保留自己的 `.grad`。

如果一个叶子张量设置了 `requires_grad=True`，那么从它出发参与的后续算子就会被记录进计算图。反过来，如果你没有打开这个标志，PyTorch 就把这段计算当成普通数值计算，不会追踪梯度。

这意味着“是否求导”不是一个全局魔法开关，而是由三层机制共同决定：

| 机制 | 是否建图 | 是否产生 `grad_fn` | 是否把梯度累加到叶子 `.grad` | 典型用途 |
|---|---|---|---|---|
| 默认 grad 模式 | 是 | 是 | 是 | 训练 |
| `torch.no_grad()` | 否 | 否 | 否 | 参数更新、验证推理前向 |
| `torch.inference_mode()` | 否 | 否 | 否，且更激进优化 | 纯推理 |
| `requires_grad=False` | 对该张量相关路径不追踪 | 否 | 否 | 冻结参数 |

`torch.no_grad()` 和 `requires_grad=False` 看起来都能“关掉梯度”，但边界不同。前者是**上下文级别**禁止记录，进入这个上下文后，即便里面某个张量原本需要梯度，也不会建图；后者是**张量级别**设置，只影响从该张量出发的梯度路径。

一个新手最容易混淆的点是：非叶子张量即使 `requires_grad=True`，也不代表它的 `.grad` 一定可用。默认情况下，Autograd 只把梯度累加到叶子上，因为训练真正需要更新的是参数，而不是每个中间结果。若你确实想查看中间张量梯度，需要额外调用 `retain_grad()`。

还有一个边界是：Autograd 主要处理的是**局部可导**的数值算子。局部可导可以理解成“在当前点附近能写出稳定导数规则”。像 `exp`、`matmul`、`sum` 这类算子没有问题；但 `ReLU` 在 0 点、`sqrt` 在 0 附近、除法在分母接近 0 时，就可能出现子梯度、无穷大或 NaN。自动求导不会替你修复数学上的坏定义，它只会按既定规则传播结果。

---

## 核心机制与推导

Autograd 的执行可以拆成两个阶段：**前向记录**和**反向回放**。

前向记录阶段，每执行一个算子，PyTorch 都会创建对应的后向节点。这个节点可以理解成“该算子的求导规则对象”。例如你写：

```python
y = torch.exp(x).sum()
```

逻辑上发生了两步：

1. `exp(x)` 生成一个中间张量，它知道自己对应 `ExpBackward`。
2. `sum(...)` 再生成一个标量结果，它知道自己对应 `SumBackward`。

于是最终 `y.grad_fn` 会指向 `SumBackward`，它再连到前一个节点，形成一条依赖链。更一般地，多个输入输出会形成一张 DAG，而不是单链表。

为什么前向时还要保存中间值？因为很多局部导数并不只依赖输入输出的形状，还依赖前向真实数值。比如：

$$
y = e^x
$$

其导数是：

$$
\frac{dy}{dx} = e^x
$$

后向如果想算这个导数，就得知道前向的 $e^x$ 是多少。因此 PyTorch 会在前向把某些中间结果以 `SavedTensor` 形式缓存起来，供后向读取。不是所有值都保存，保存什么由具体算子的求导实现决定。

下面用玩具例子展开链式法则。

设：

$$
x = [x_1, x_2], \quad z_i = e^{x_i}, \quad y = z_1 + z_2
$$

则有：

$$
\frac{\partial y}{\partial x_i}
=
\frac{\partial y}{\partial z_i}\frac{\partial z_i}{\partial x_i}
=
1 \cdot e^{x_i}
=
e^{x_i}
$$

反向执行顺序就是：

1. 从标量 `y` 开始，初始化上游梯度为 1，因为 $\partial y / \partial y = 1$。
2. `SumBackward` 把这个 1 分发给每个输入分量，所以传给 `z` 的梯度是全 1。
3. `ExpBackward` 读取前向保存的 `exp(x)`，计算 `grad_output * exp(x)`。
4. 结果累加到叶子 `x.grad`。

这个过程就是**向量-雅可比积**。雅可比矩阵可以理解成“输出每一维对输入每一维的偏导排成的矩阵”，而反向模式不显式构造完整雅可比，而是直接算“上游梯度向量 × 局部雅可比”，因此对“标量损失、海量参数”的训练场景特别高效。

真实工程里，模型通常不是一条链，而是一张多分支图。比如残差网络里同一个参数可能既影响主分支，又影响跳连后的合并结果。这时链式法则变成多路径求和：

$$
\frac{\partial L}{\partial \theta}
=
\sum_{k \in \text{paths}}
\prod_{e \in k}
\frac{\partial \text{child}(e)}{\partial \text{parent}(e)}
$$

白话说，同一个参数对损失的影响可能有很多条路，Autograd 会把每条路上的“局部导数连乘”全部加起来。这也是 `.grad` 具有“累加语义”的根本原因。

动态图的另一个关键性质是：**每次前向都会重建图**。这不是缺点，而是设计目标。因为 Python 控制流在不同输入下可能走不同路径，所以计算图不应该是提前固化的。训练循环中每一轮 forward，PyTorch 都重新生成当轮实际执行过的图；一次 backward 结束后，默认会把不再需要的中间缓冲释放掉，避免内存无限增长。

---

## 代码实现

先看一个最小可运行例子，验证 `.backward()` 和 `torch.autograd.grad()` 的差异。

```python
import math
import torch

x = torch.tensor([0.5, 0.75], requires_grad=True)
y = torch.exp(x).sum()

# grad 返回值，不写入 x.grad
(g,) = torch.autograd.grad(y, x, retain_graph=True)
expected = torch.tensor([math.exp(0.5), math.exp(0.75)])
assert torch.allclose(g, expected, atol=1e-6)

# backward 会把梯度累加到叶子张量 x.grad
y.backward()
assert torch.allclose(x.grad, expected, atol=1e-6)
```

这段代码体现了两个核心事实：

1. `torch.autograd.grad()` 适合“我只想拿这次梯度值做分析或进一步计算”，它直接返回结果，不改写参数的 `.grad`。
2. `.backward()` 适合标准训练流程，它会把梯度累加到所有相关叶子张量的 `.grad` 上，供优化器读取。

如果你连续调用两次 `.backward()` 而不清空梯度，梯度会叠加：

```python
import torch

w = torch.tensor([2.0], requires_grad=True)

loss1 = (w ** 2).sum()      # d/dw = 2w = 4
loss1.backward()
assert torch.allclose(w.grad, torch.tensor([4.0]))

loss2 = (3 * w).sum()       # d/dw = 3
loss2.backward()
assert torch.allclose(w.grad, torch.tensor([7.0]))
```

这就是为什么训练循环里通常要执行 `optimizer.zero_grad()` 或把参数 `.grad` 设为 `None`。否则上一轮梯度会混进下一轮。

再看一个“控制图边界”的例子：

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)

with torch.no_grad():
    y = x * 3.0

assert y.requires_grad is False

z = (x * 3.0).sum()
z.backward()
assert torch.allclose(x.grad, torch.tensor([3.0, 3.0]))
```

这里同样是乘以 3，但在 `no_grad` 内部的 `y` 不会被追踪，因此不属于任何求导图；而下面那次正常前向会建图，反向后把梯度填回 `x.grad`。

真实工程例子可以看“共享前向、多个损失”的情况。比如一个多任务模型同时做分类和回归，共享 backbone，输出两个头：

```python
features = backbone(x)
cls_logits = cls_head(features)
bbox_pred = box_head(features)

loss_cls = cls_criterion(cls_logits, y_cls)
loss_box = box_criterion(bbox_pred, y_box)
loss = loss_cls + loss_box
loss.backward()
```

如果你的目标只是训练，总损失直接相加一次 backward 最简单，因为数学上：

$$
\frac{\partial (L_1 + L_2)}{\partial \theta}
=
\frac{\partial L_1}{\partial \theta}
+
\frac{\partial L_2}{\partial \theta}
$$

但如果你想分别观察 `loss_cls` 和 `loss_box` 对共享参数的影响，比如做梯度冲突分析、动态调权、PCGrad 一类算法，就需要分别求梯度。这时可以用 `torch.autograd.grad()`，或者前几次 `backward(retain_graph=True)`，最后一次不保留图。

---

## 工程权衡与常见坑

自动求导在训练里几乎“默认可用”，但工程上真正出问题的地方，通常不是公式，而是**图生命周期、内存占用、梯度累加和不可导点**。

下面先给出高频问题表：

| 问题 | 现象 | 根因 | 常见修复 |
|---|---|---|---|
| 忘记清空梯度 | 参数梯度越来越大 | `.backward()` 默认累加 | 每轮 `zero_grad()` 或 `grad=None` |
| 多次 backward 报错 | “Trying to backward through the graph a second time” | 默认第一次后向已释放图 | 前几次用 `retain_graph=True` |
| 长时间 OOM | 显存持续上涨 | 图和中间缓存被保留 | 最后一次后向不要 `retain_graph=True` |
| 冻结参数无效 | 某些层仍有梯度 | 只停了优化器，没关 `requires_grad` | 对冻结参数显式设 `requires_grad=False` |
| NaN 梯度 | loss 突然变 NaN | 不可导点、除零、数值爆炸 | `clamp`、加 epsilon、梯度裁剪 |
| 中间张量 `.grad` 是 `None` | 调试时看不到梯度 | 默认只保留叶子梯度 | 对中间张量调用 `retain_grad()` |

最常见的坑是 `retain_graph=True`。它的作用不是“更安全地 backward”，而是“告诉引擎：这张图后面还要继续用，先别释放”。如果你只是照抄教程在每次 backward 都开着它，显存通常会越来越高，最后 OOM。

典型场景是多损失分别求导。正确思路是：只有前几次 backward 需要保留图，最后一次必须释放。

例如共享主干网络、两个损失分别取梯度时：

```python
loss1.backward(retain_graph=True)
loss2.backward()   # 最后一次不保留
```

如果你把两次都写成 `retain_graph=True`，这轮训练结束后图还在，占用的前向缓存不会及时释放。批量大、序列长、模型深时，这个代价非常明显。

第二个常见坑是把“图记录”和“参数更新”混在一起。优化器更新参数时，本质上不需要求导；如果你在参数赋值或 EMA 更新时没进 `no_grad()`，可能会意外把更新步骤也接进图里，造成额外内存和错误依赖。

第三个问题是不可导点。自动求导不是数学豁免证。比如 `sqrt(x)` 在 $x=0$ 附近非常敏感，`log(x)` 在 $x \le 0$ 不合法，归一化时除以一个接近 0 的范数也可能炸掉。工程上通常不会“等 NaN 出来再查”，而是在前向主动约束定义域：

$$
y = \sqrt{\max(x, \epsilon)}
$$

或

$$
y = \log(x + \epsilon)
$$

这里的 $\epsilon$ 是一个很小的正数，白话就是“给数值留安全边界”。

第四个问题是把 `detach()` 用错。`detach()` 的意思是“从当前图中切断这个张量，后面把它当常数看”。这在目标网络、停止某条辅助损失回传、缓存历史状态时非常有用；但如果你在主干路径随手 `detach()`，梯度就真的断了，优化器不可能再更新它前面的参数。

---

## 替代方案与适用边界

PyTorch 并不是只有一种“求梯度方式”。不同 API 对应不同工程目标，关键不是记名字，而是看“要不要建图、要不要累加、要不要支持更高阶梯度”。

| 方案 | 是否建图 | 是否写入 `.grad` | 适用场景 | 边界 |
|---|---|---|---|---|
| `loss.backward()` | 是 | 是 | 标准训练 | 默认面向标量损失 |
| `torch.autograd.grad()` | 是 | 否 | 拿梯度做分析、正则、元学习 | 需要自己管理返回值 |
| `torch.no_grad()` | 否 | 否 | 验证、参数更新、冻结部分计算 | 退出上下文后可恢复 |
| `detach()` | 之前建图，之后切断 | 否 | 停止某条支路回传 | 只切断当前张量之后的依赖 |
| `torch.inference_mode()` | 否 | 否 | 纯推理、极致性能 | 生成的张量更偏只读语义 |
| `create_graph=True` | 是，且为梯度再建图 | 视调用方式而定 | 高阶梯度、元学习 | 内存开销大 |

如果你只是做常规监督学习，`loss.backward()` 已经足够；它和优化器配合最好，也最符合“参数梯度累加”的训练范式。

如果你需要“梯度本身再参与计算”，例如梯度惩罚、隐式元学习、MAML，才需要 `torch.autograd.grad()` 或 `create_graph=True`。这里要特别注意：`create_graph=True` 不是“更完整的求导”，而是“让一阶梯度本身也带着图”，这样你后面才能对梯度继续求导。代价是内存和图复杂度显著上升。

一个真实工程边界是推理服务。在线推理只需要前向输出，不需要任何梯度信息。这时比 `no_grad()` 更激进的 `inference_mode()` 往往更合适，因为它进一步省掉了一些 Autograd 元数据管理成本。反过来，如果你虽然不想更新参数，但还要复用结果进入后续可求导计算，那么 `detach()` 往往比全局 `no_grad()` 更精确。

总结成一句话：

- 训练主路径：优先 `backward()`
- 想拿梯度值但不污染 `.grad`：用 `autograd.grad()`
- 整段不要建图：用 `no_grad()` 或 `inference_mode()`
- 只切断某个张量之后的梯度：用 `detach()`
- 需要高阶梯度：用 `create_graph=True`，同时准备好额外内存预算

---

## 参考资料

- PyTorch 官方《Autograd mechanics》：https://docs.pytorch.org/docs/main/notes/autograd.html
- PyTorch 官方《How Computational Graphs are Executed in PyTorch》：https://pytorch.org/blog/how-computational-graphs-are-executed-in-pytorch/
- PyTorch `torch.autograd.grad` 文档：https://docs.pytorch.wiki/en/generated/torch.autograd.grad.html
- PyTorch 论坛 `retain_graph=True` 与内存问题讨论：https://discuss.pytorch.org/t/retain-graph-true-out-of-memory-issue/85389
- PyTorch 论坛多次 backward 与图保留讨论：https://discuss.pytorch.org/t/use-of-retain-graph-true/179658
- PyTorch 官方关于不可导函数梯度的说明：https://docs.pytorch.org/docs/stable/notes/autograd.html
