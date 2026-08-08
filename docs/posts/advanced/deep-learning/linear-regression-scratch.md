---
title: 线性回归与从零实现
date: 2026-08-07
---

# 线性回归与从零实现

<div class="epigraph">
<p>所有模型都是错的，但有些是有用的。</p>
<footer>—— 乔治 · 博克斯（George E. P. Box）</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 花书《深度学习》§5.1.4、李沐《动手学深度学习》§3.1–3.3 ｜ 2026-08-07</p>
</div>

## 为什么从线性回归开始

前面几节建立了学习的形式框架：任务、性能度量、经验、容量、正则、验证。现在需要一个**能立刻上手跑起来**的最小完整模型，把这一整套概念在具体数字上演练一遍。**线性回归（linear regression）**正是这样的角色：它是最简单的监督学习模型，却拥有完整的学习闭环——定义模型、定义损失、求解（解析解与梯度解并存）、评估泛化。

更关键的是，线性回归是**深度学习的雏形**：一个线性回归就是一个「没有隐藏层、没有激活函数」的单层网络。从它出发，加一个隐藏层就是多层感知机，加卷积就是 CNN，加自回归就是 RNN——**理解线性回归，就理解了深度学习的一切基本骨架**。李沐《动手学深度学习》把它放在第一章，就是这个道理。<span class="marginnote">线性回归还有两重身份值得记住：在统计学里它是最基本的回归模型（最小二乘），在经济学里是拟合需求曲线的工具（见第一级《经济学基础》中「线性回归拟合价格-销量」的例子）；而在机器学习里，它是理解更复杂模型的「最小可行样例」。</span>

## 1 模型与假设：一条高维直线

**线性回归**：假设输出 $y$ 是输入 $\boldsymbol{x} \in \mathbb{R}^d$ 的**仿射函数**加上噪声：

$$
\hat{y} = \boldsymbol{w}^{\top}\boldsymbol{x} + b
$$

其中 $\boldsymbol{w} \in \mathbb{R}^d$ 是**权重（weight）**，$b \in \mathbb{R}$ 是**偏置（bias）**。把偏置并进权重，用 $\boldsymbol{x}' = [1, x_1, \dots, x_d]$、$\boldsymbol{w}' = [b, w_1, \dots, w_d]$ 可写成更紧凑的 $\hat{y} = \boldsymbol{w}'^{\top}\boldsymbol{x}'$。线性回归的**假设空间**是所有仿射函数构成的集合——它是「容量」概念里最简单的一档：一条直线（$d=1$）、一个平面（$d=2$）、一个超平面（$d \ge 3$）。<span class="marginnote">「线性」一词在不同语境含义不同：在参数上线性（对 $\boldsymbol{w}$ 是线性的）≠ 在特征上线性。$\hat{y} = w_1 x_1 + w_2 x_1^2$ 对参数仍是线性的（叫线性模型），却可以拟合曲线——「把非线性特征喂给线性模型」是特征工程的起源。</span>

数据上我们假设带噪声的观测：$y = \boldsymbol{w}^{\top}\boldsymbol{x} + b + \epsilon$，噪声 $\epsilon$ 常假设为零均值高斯分布。这个假设决定了后面的损失函数选择。

## 2 损失函数：最小二乘从哪来

有了模型，要定义「模型有多差」。对单个样本，定义**平方误差（squared error）**：

$$
\ell^{(i)}(\boldsymbol{w}, b) = \frac{1}{2}\big(y^{(i)} - \hat{y}^{(i)}\big)^2
$$

对所有 $n$ 个样本平均，得到**均方误差（mean squared error, MSE）**损失：

$$
L(\boldsymbol{w}, b) = \frac{1}{n}\sum_{i=1}^{n} \frac{1}{2}\big(y^{(i)} - \boldsymbol{w}^{\top}\boldsymbol{x}^{(i)} - b\big)^2
$$

系数 $\frac{1}{2}$ 纯粹是为了求导时消掉平方的 2，**不改变最优解**。为什么选平方而不是绝对值？因为**最小化均方误差 = 最大似然估计（假设高斯噪声）**：在噪声为独立同分布高斯 $\mathcal{N}(0, \sigma^2)$ 的假设下，观测数据的对数似然正是负的均方误差。这接回了第一篇《最大似然估计与贝叶斯统计》——**损失函数不是随便选的，它编码了「数据如何生成」的假设**。<span class="marginnote">若噪声是拉普拉斯分布，对应的损失就会变成绝对误差（L1），最优解是中位数而非均值——「换噪声假设 = 换损失函数」这条对应关系，在第四篇《损失函数设计》会系统化。</span>

## 3 解析解：正规方程

线性回归的幸运之处在于，它的最优解有**闭式（closed-form）表达式**。把全部数据写成矩阵形式：$\boldsymbol{X} \in \mathbb{R}^{n \times d}$（每行一个样本）、$\boldsymbol{y} \in \mathbb{R}^n$，目标为

$$
\boldsymbol{w}^* = \arg\min_{\boldsymbol{w}} \frac{1}{2}\|\boldsymbol{X}\boldsymbol{w} - \boldsymbol{y}\|_2^2
$$

对 $\boldsymbol{w}$ 求梯度并令为零，得到**正规方程（normal equation）**：

$$
\boldsymbol{w}^* = (\boldsymbol{X}^{\top}\boldsymbol{X})^{-1}\boldsymbol{X}^{\top}\boldsymbol{y}
$$

三步拆解这个公式：

- **第一步，看形状**：$\boldsymbol{X}^{\top}\boldsymbol{X}$ 是 $d \times d$ 矩阵，求逆后仍是 $d \times d$；$\boldsymbol{X}^{\top}\boldsymbol{y}$ 是 $d$ 维向量。矩阵乘向量得 $d$ 维 $\boldsymbol{w}^*$——维度严丝合缝。
- **第二步，看几何**：$\boldsymbol{X}^{\top}\boldsymbol{X}$ 可逆要求特征**线性无关**且 $n \ge d$。当特征相关或样本不足时矩阵奇异，正规方程失效——这正是上一节正则化加 $\lambda\boldsymbol{I}$ 要解决的问题。
- **第三步，看代价**：矩阵求逆是 $O(d^3)$。$d$ 不大时（几千）可行；$d$ 上亿时（现代大模型）完全不可行，必须用迭代法——这解释了为什么深度学习靠梯度下降而非解析解。

## 4 从零实现：梯度下降训练循环

当数据量大或维数高时，用**随机梯度下降（SGD）**迭代求解。参数更新规则：

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} - \eta\,\nabla_{\boldsymbol{w}} L, \qquad b \leftarrow b - \eta\,\frac{\partial L}{\partial b}
$$

其中学习率 $\eta$ 控制步长。用 PyTorch 从零实现（不调用 `nn.Linear`、`optim.SGD` 等高层 API，只靠张量运算与自动微分）的完整循环：

```python
import random
import torch

# 生成合成数据：labels 依赖 features，真实权重 w* = [2, -3.4]、b* = 4.2
num_inputs, num_examples = 2, 1000
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
features = torch.randn(num_examples, num_inputs)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.normal(0, 0.01, size=labels.shape)   # 高斯噪声

def data_iter(batch_size, features, labels):   # 每 epoch 洗牌、切批
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: i + batch_size])
        yield features[batch_indices], labels[batch_indices]

# 参数初始化：requires_grad=True 让 autograd 追踪
w = torch.normal(0, 0.01, size=(num_inputs, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):          # 前向：线性模型
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):   # 损失：均方误差
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):   # 更新：小批量梯度下降
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()      # 清零：防止梯度跨 batch 累加

lr, num_epochs, batch_size = 0.03, 3, 32
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = squared_loss(linreg(X, w, b), y)  # 前向 + 损失
        l.sum().backward()                    # 反向：梯度进 w.grad、b.grad
        sgd([w, b], lr, batch_size)           # 更新（内部已清零）
```

这个循环浓缩了深度学习的**全部骨架**，值得逐行拆解：

- **前向**：`linreg(X, w, b)` 计算预测。
- **损失**：MSE 是一个可导标量。
- **反向**：`backward()` 自动把梯度填进 `w.grad`、`b.grad`。
- **更新**：`sgd([w, b], lr, batch_size)` 沿负梯度走一步。
- **清零**：`param.grad.zero_()` 防止梯度在多个 batch 间累加——正是上一篇《自动微分》强调的「分叉求和」的工程版。

**易错点：** 需要更新（`requires_grad=True`）的张量在更新后必须放在 `with torch.no_grad():` 里操作，否则更新本身也会被记录进计算图，越滚越大、显存爆炸。这是所有 PyTorch 新手都会踩的坑。<span class="marginnote">真实工程里 `backward()` 前往往要 `optimizer.zero_grad()`，就是在这里手动清零梯度的封装。用高层 API 时这些细节被优化器隐藏，但「清零→前向→反向→更新」四步的顺序是任何框架都逃不掉的节拍。</span>

## 5 求解方式对比与泛化评估

| 求解方式 | 代价 | 适用场景 | 局限 |
| --- | --- | --- | --- |
| 解析解（正规方程） | $O(d^3)$ + 求逆 | 特征少、样本中等 | $d$ 大时不可行、需矩阵可逆 |
| 梯度下降（全批量） | 每步 $O(nd)$ | 凸二次，收敛快 | 每次用全部数据，慢 |
| 小批量 SGD | 每步 $O(\text{batch}\cdot d)$ | 深度学习标准 | 有噪声，需调学习率 |

**泛化评估**：训练完成后，用独立**测试集**计算 MSE。注意数据生成时噪声标准差 0.01，因此「理论最优」的测试损失约为 $\frac{1}{2}\times 0.01^2 = 5\times 10^{-5}$ 量级——如果训练损失远低于这个数，说明模型在拟合噪声（过拟合）。**对比训练损失与噪声下限，是诊断过拟合的第一把尺子。**

**易错点：** 合成数据里 `labels` 依赖 `features`，生成时用了真实权重。评估时若把「训练集上拟合的 $\hat{w}$」与「真实权重」直接对比，两者不会完全相等——因为噪声让数据偏离了真值。**学到的 $\hat{w}$ 逼近真值 $w^*$，但不等于真值**，这正是偏差-方差分解中「方差」的来源。

## 6 小结

- **线性回归**假设 $y = \boldsymbol{w}^{\top}\boldsymbol{x} + b + \epsilon$，是最简单的监督模型，也是单层无激活网络。
- **均方误差**在高斯噪声假设下等价于**最大似然估计**，系数 $\frac{1}{2}$ 只为化简求导。
- **解析解** $\boldsymbol{w}^* = (\boldsymbol{X}^{\top}\boldsymbol{X})^{-1}\boldsymbol{X}^{\top}\boldsymbol{y}$ 需要特征无关与 $n \ge d$，代价 $O(d^3)$。
- 数据大时用**小批量 SGD**，训练循环四步曲：清零、前向、反向、更新。
- 用测试损失与「噪声下限」对比，是判断过拟合的第一把尺子。

在下一节，我们把「输出一个数」推广到「输出一个概率分布」，处理多类分类——这就是**Softmax 回归**。
