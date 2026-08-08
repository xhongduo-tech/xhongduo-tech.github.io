---
title: 反向传播算法：链式法则与计算图
date: 2026-08-07
---

# 反向传播算法：链式法则与计算图

<div class="epigraph">
<p>每一个错误都是一次机会，反向传播就是把机会带回去。</p>
<footer>—— 依据戴维 · 鲁梅尔哈特（David Rumelhart）1986 论文的精神改写</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 花书《深度学习》§6.5、李沐《动手学深度学习》§3.7 ｜ 2026-08-07</p>
</div>

## 为什么从反向传播开始

前面的所有模型都依赖同一个前提：**能高效算出损失对每个参数的梯度**。手动对含几亿参数的网络求导不现实；数值差分（每参数两次前向）也不现实；符号求导的表达式会指数膨胀。**反向传播算法（backpropagation, backprop）**就是那个既精确又高效的正解：它在一个称为**计算图**的表示上做两次遍历，用链式法则把梯度从输出一路「传」回输入。

反向传播的意义怎么强调都不为过：它让「训练任意深度的可微网络」第一次变得实用，是 1986 年神经网络复兴的直接推手，也是今天 PyTorch/TensorFlow 里 `autograd` 背后唯一在做的事。理解它，就是理解整个深度学习引擎的**燃料系统**。<span class="marginnote">反向传播的历史很有趣：它的思想可追溯到控制论的「反向模式自动微分」（1960 年代 Linnainmaa、1970 年代 Werbos），但真正引爆研究界的是 1986 年 Rumelhart、Hinton 与 Williams 的论文《Learning representations by back-propagating errors》。现代框架把它做成了透明的自动微分，但底层机制与 1986 年完全一致。</span>

## 1 计算图：把函数变成一张网

**计算图（computational graph）**：把复合函数拆成**基本运算**节点、用有向边连接依赖关系的图。例如函数

$$
f(\boldsymbol{x}) = \sigma\big(\boldsymbol{W}\boldsymbol{x} + \boldsymbol{b}\big)
$$

被拆成四个节点：矩阵乘法 $\boldsymbol{a} = \boldsymbol{W}\boldsymbol{x}$、加法 $\boldsymbol{z} = \boldsymbol{a} + \boldsymbol{b}$、激活 $\boldsymbol{h} = \sigma(\boldsymbol{z})$，再连到损失节点 $L = \ell(\boldsymbol{h}, \boldsymbol{y})$。每个节点都有一个**局部导数**——对它的每个输入求偏导，这些导数只依赖该节点的公式，简单且预先可知。

**为什么要拆图？** 因为复合函数的导数可以由局部导数**沿图组合**得到。拆成基本运算后，「求任意函数的导数」退化为「图上每个节点记下自己的局部导数，再按链式法则把它们乘起来」——这是计算机能机械完成的工作。<span class="marginnote">计算图也是现代框架的<strong>中间表示</strong>：你写的 PyTorch 代码会被记录成一张动态图，TensorFlow 则用静态图。图的拓扑序决定了前向与反向都能一趟完成——「有向无环图」这个条件保证不存在循环依赖，详见第三级《数据结构》关于拓扑排序的讨论。</span>

## 2 前向传播：算值并缓存

**前向传播（forward propagation）**：按拓扑序从左到右（从输入到输出）计算每个节点的值，并把它们**缓存**起来。

前向缓存有两个目的：一是得到最终输出与损失；二是为反向传播提供「记忆」——许多局部导数的计算需要用到节点的输入值。例如 $\sigma'(z) = \sigma(z)(1-\sigma(z))$ 需要 $z$（或 $\sigma(z)$）的值；线性层 $\frac{\partial \boldsymbol{z}}{\partial \boldsymbol{W}} = \boldsymbol{x}^{\top}$ 需要输入的激活 $\boldsymbol{x}$。<span class="marginnote">「反向时重新算一次前向值」听起来可以省显存，但会浪费算力。现代框架的默认是<strong>用显存换时间</strong>：前向把每个节点的值都缓存，反向直接读取。显存吃紧时的「梯度检查点（gradient checkpointing）」则反其道——少存中间值、反向时重算，用时间换显存，见第九篇《混合精度与显存优化》。</span>

## 3 反向传播：梯度沿图回流

**反向传播（backward propagation）**：从损失节点出发，逆拓扑序往回，对每个节点计算**损失对它的梯度**并传给它的输入。

对节点 $u$，定义它的**伴随量（adjoint）**为 $\bar{u} = \frac{\partial L}{\partial u}$。若 $u$ 是 $g$ 的一个输入（$g$ 在 $u$ 下游），那么沿边回传的规则是：

$$
\bar{u} \mathrel{+}= \bar{g} \cdot \frac{\partial g}{\partial u}
$$

**「+=" 不是笔误**——这正是反向传播最容易出错、也最核心的细节：**当一个节点被多条下游路径使用（分叉节点）时，来自各路径的梯度必须累加**。这是多变量链式法则「总导数 = 各路径贡献之和」的图论表达。以 $u_3 = u_2 + \sin(x)$ 为例（见《微积分与自动微分》），$x$ 的梯度来自两条路，必须相加。

反向传播的**时间复杂度**与前向同阶（约 2–3 倍），因为它对每条边只做一次常数时间操作。这就是它能处理几亿参数网络的根本原因——**梯度计算的总成本不是「对每个参数算一次前向」那样 O(参数 × 前向)，而是 O(前向)**。<span class="marginnote">把「反向传播 = 链式法则」的关系再钉死一次：反向传播不发明任何新的数学，它只是把链式法则组织成高效的图算法。它唯一的「聪明」在于——<strong>知道哪些中间量会被复用，从而只算一次</strong>。这正是它与「对每个参数单独求导」的本质区别。</span>

## 4 公式解析：两个三行推导

反向传播的全部秘密，可以用两段「三行推导」概括。

**推导一：单个节点的回传规则。** 设 $\boldsymbol{z} = \boldsymbol{W}\boldsymbol{h}$（线性层），已知 $\bar{\boldsymbol{z}} = \frac{\partial L}{\partial \boldsymbol{z}}$，求对 $\boldsymbol{W}$、$\boldsymbol{h}$ 的梯度：

$$
\frac{\partial L}{\partial \boldsymbol{W}} = \bar{\boldsymbol{z}}\, \boldsymbol{h}^{\top}, \qquad
\frac{\partial L}{\partial \boldsymbol{h}} = \boldsymbol{W}^{\top} \bar{\boldsymbol{z}}, \qquad
\frac{\partial L}{\partial \boldsymbol{b}} = \bar{\boldsymbol{z}}
$$

- **第一步，读第一行**：对权重梯度是「上游梯度 × 输入激活」的**外积**——为什么是外积已在《自动微分》一节推过。
- **第二步，读第二行**：对输入梯度是「权重转置 × 上游梯度」——梯度沿输入侧回流时被 $\boldsymbol{W}^{\top}$ 映射回输入空间。
- **第三步，读第三行**：偏置梯度就是上游梯度本身（偏置对输出的影响系数为 1）。

**推导二：Softmax + 交叉熵的梯度。** 设 $\hat{\boldsymbol{y}} = \text{softmax}(\boldsymbol{o})$、$L = -\log \hat{y}_{\text{true}}$，则

$$
\frac{\partial L}{\partial \boldsymbol{o}} = \hat{\boldsymbol{y}} - \boldsymbol{y}_{\text{one-hot}}
$$

- **第一步，逐分量看**：对真实类 $j$，$\frac{\partial L}{\partial o_j} = \hat{y}_j - 1$；对非真实类，$= \hat{y}_j - 0$——统一写成 $\hat{\boldsymbol{y}} - \boldsymbol{y}$。
- **第二步，看指数抵消**：$\frac{\partial}{\partial o_j}\log \text{softmax}$ 展开后，$\exp(o_j)$ 的归一化因子贡献恰为 $\hat{y}_j$，与分子贡献相减，指数项神奇消失——这就是「Softmax + 交叉熵必须一起实现」的数值稳定性来源。
- **第三步，看直觉**：梯度 = 预测减真值。预测高估就往低推，低估就往高拉，且距离越大、推力越大——**这就是「学习」的微观动作**。<span class="marginnote">把这个梯度与第一节的「NLL 的统一框架」对照：几乎所有常用损失+输出单元组合的梯度，最终都呈「预测 − 目标」的形式（回归是 $\hat{y}-y$，二分类是 $\hat{y}-y$，多分类是 $\hat{\boldsymbol{y}}-\boldsymbol{y}$）。这个简洁性不是巧合，而是最大似然原理送给我们的礼物。</span>

## 5 反向传播的正确打开方式与常见误解

**为什么反向传播这么高效？** 关键在**共享中间结果**。网络第 $l$ 层的梯度 $\bar{\boldsymbol{h}}^{(l)}$ 会被第 $l-1$ 层复用；每一层的计算都被下游所有参数共享。相比「对每个参数单独做一次前向求导」（$O(n_{\text{param}})$ 次前向），反向传播只做 1 次前向 + 1 次反向，是**革命性的加速**。

**易错点一：「反向传播会过拟合」的迷信。** 反向传播只是一个求导算法，不是归纳偏置——它本身既不导致也不避免过拟合。过拟合由容量与数据决定，与「用什么算法求导」无关。

**易错点二：把「梯度消失」归罪于反向传播。** 反向传播忠实计算链式法则的精确结果；梯度消失是**函数本身**（激活饱和 + 深层连乘）的性质，换个算法求导也一样。反向传播只是「如实报告」了梯度已经消失这一事实。

**易错点三：实现里忘记清梯度。** PyTorch 中 $10^{-7}$ 默认把梯度累加到 `.grad`。常规训练循环若不 `zero_grad()`，上一个 batch 的梯度会叠加进下一个 batch——训练看似进行，实则混乱。**「清梯度 → 前向 → 损失 → 反向 → 更新」五步循环是雷打不动的节拍**。<span class="marginnote">调试反向传播的黄金工具是<strong>数值梯度检验</strong>：用中心差分 $\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}$ 逼近梯度，与反向梯度对比相对误差。若实现正确，相对误差应在 $10^{-7}$ 量级。这个「体检」的具体流程（选小网络、关 Dropout、单样本）在第九篇《调试策略》有完整清单。</span>

## 6 小结

- **计算图**把复合函数拆成基本运算节点，局部导数预先已知，求导退化为图算法。
- **前向传播**按拓扑序算值并**缓存**，为反向提供必要的中间量。
- **反向传播**逆拓扑序回传伴随量 $\bar{u}$，规则 $\bar{u} \mathrel{+}= \bar{g}\frac{\partial g}{\partial u}$；**分叉节点必须累加**。
- 梯度计算总成本与前向同阶，靠**中间结果共享**，这就是它能扩展到几亿参数的原因。
- 核心两段推导：线性层「外积 + 转置回传」；Softmax+交叉熵梯度 $\hat{\boldsymbol{y}}-\boldsymbol{y}$。
- 反向传播是求导引擎，不引入过拟合也不制造梯度消失——它只如实报告。

在下一节，我们把反向传播从「算法」落到「符号」，用完整的记号把每一层的前向与反向式子写出来并推导一遍——这就是**前向传播与反向传播的符号推导**。
