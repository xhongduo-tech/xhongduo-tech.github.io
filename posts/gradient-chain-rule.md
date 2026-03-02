

## 从导数到梯度：一个变量到多个变量

导数衡量函数对输入的敏感度——输入变化一点点，输出变化多少。

对单变量函数 $f(x) = x^3$，导数定义为：

$$f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x} = 3x^2$$

当函数有多个输入时，比如 $f(x, y) = x^2 y + y^3$，我们对每个变量分别求导，固定其他变量不动，得到**偏导数**：

$$\frac{\partial f}{\partial x} = 2xy, \quad \frac{\partial f}{\partial y} = x^2 + 3y^2$$

把所有偏导数排成一个向量，就是**梯度**：

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} 2xy \\ x^2 + 3y^2 \end{bmatrix}$$

- $\nabla f \in \mathbb{R}^n$，$n$ 是输入变量个数
- 梯度在某一点的值是一个具体向量，指向函数值增长最快的方向
- 梯度的模 $\|\nabla f\|$ 表示最大增长率

---

## 梯度的几何意义：为什么梯度下降有效

在点 $(x_0, y_0)$ 处，沿单位方向 $\mathbf{u} = (\cos\theta, \sin\theta)$ 的**方向导数**为：

$$D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u} = \|\nabla f\| \cos\alpha$$

其中 $\alpha$ 是梯度方向与 $\mathbf{u}$ 的夹角，$D_{\mathbf{u}} f$ 表示沿 $\mathbf{u}$ 方向的变化率。

- $\alpha = 0$ 时（沿梯度方向），$D_{\mathbf{u}} f = \|\nabla f\|$，增长最快
- $\alpha = \pi$ 时（沿梯度反方向），$D_{\mathbf{u}} f = -\|\nabla f\|$，下降最快

梯度下降就是利用这个性质。要最小化损失函数 $L(\mathbf{w})$，每步沿负梯度方向更新参数：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L(\mathbf{w}_t)$$

- $\mathbf{w}_t \in \mathbb{R}^n$：第 $t$ 步的参数向量
- $\eta > 0$：学习率，控制步长
- $\nabla L(\mathbf{w}_t) \in \mathbb{R}^n$：损失函数在当前参数处的梯度

学习率太大会跳过最优点甚至发散，太小则收敛极慢。实际训练中通常配合 Adam 等自适应优化器。

---

## 链式法则：复合函数的求导工具

神经网络本质上是一长串函数的复合：$L = f_n(f_{n-1}(\cdots f_1(\mathbf{x})\cdots))$。要对每一层的参数求梯度，需要**链式法则**。

单变量情形：若 $z = f(y)$，$y = g(x)$，则：

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

多变量推广：若 $L$ 通过中间变量 $z_1, z_2, \dots, z_m$ 依赖于 $x$，则：

$$\frac{\partial L}{\partial x} = \sum_{i=1}^{m} \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial x}$$

每个 $z_i$ 是 $x$ 的函数，$L$ 是所有 $z_i$ 的函数。求和覆盖所有从 $x$ 到 $L$ 的路径。

---

## 计算图：链式法则的可视化

把运算拆解成计算图后，反向传播的规则变得很清晰：**图中每条边对应一次偏导数乘法，每个节点汇总所有流入的梯度**。

以 $L = (a + b) \times c$ 为例，令 $d = a + b$，$L = d \times c$：

```
a ──┐
    ├── [+] ── d ──┐
b ──┘              ├── [×] ── L
               c ──┘
```

前向传播（设 $a=2, b=1, c=3$）：

| 节点 | 值 |
|------|-----|
| $a$ | 2 |
| $b$ | 1 |
| $c$ | 3 |
| $d = a+b$ | 3 |
| $L = d \times c$ | 9 |

反向传播从 $\frac{\partial L}{\partial L} = 1$ 开始，逐层向回传：

| 梯度 | 计算 | 值 |
|------|------|-----|
| $\frac{\partial L}{\partial L}$ | 起点 | 1 |
| $\frac{\partial L}{\partial d}$ | $\frac{\partial L}{\partial L} \cdot c$ | $1 \times 3 = 3$ |
| $\frac{\partial L}{\partial c}$ | $\frac{\partial L}{\partial L} \cdot d$ | $1 \times 3 = 3$ |
| $\frac{\partial L}{\partial a}$ | $\frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial a}$ | $3 \times 1 = 3$ |
| $\frac{\partial L}{\partial b}$ | $\frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial b}$ | $3 \times 1 = 3$ |

每条边就是一次链式法则的应用：上游梯度 $\times$ 本地偏导数 $=$ 传给下游的梯度。

---

## 手推三层网络的反向传播

构建一个具体网络：1 个输入 → 隐藏层（2 个神经元，ReLU）→ 输出层（1 个神经元）→ MSE 损失。

**网络结构**：

$$h_1 = \text{ReLU}(w_1 x + b_1), \quad h_2 = \text{ReLU}(w_2 x + b_2)$$

$$\hat{y} = v_1 h_1 + v_2 h_2 + b_3$$

$$L = \frac{1}{2}(\hat{y} - y)^2$$

- $x \in \mathbb{R}$：输入
- $w_1, w_2, b_1, b_2$：隐藏层权重和偏置
- $v_1, v_2, b_3$：输出层权重和偏置
- $y$：真实标签
- $\text{ReLU}(z) = \max(0, z)$，导数为 $\mathbf{1}_{z > 0}$（$z > 0$ 时为 1，否则为 0）
- 系数 $\frac{1}{2}$ 是为了求导时消去 2，简化计算

**设定初始值**：

| 参数 | $x$ | $y$ | $w_1$ | $w_2$ | $b_1$ | $b_2$ | $v_1$ | $v_2$ | $b_3$ |
|------|-----|-----|-------|-------|-------|-------|-------|-------|-------|
| 值   | 1.0 | 1.0 | 0.5   | -0.3  | 0.1   | 0.2   | 0.8   | -0.6  | 0.0   |

### 前向传播

逐步计算：

$$z_1 = w_1 x + b_1 = 0.5 \times 1.0 + 0.1 = 0.6$$

$$z_2 = w_2 x + b_2 = -0.3 \times 1.0 + 0.2 = -0.1$$

$$h_1 = \text{ReLU}(0.6) = 0.6, \quad h_2 = \text{ReLU}(-0.1) = 0$$

$$\hat{y} = 0.8 \times 0.6 + (-0.6) \times 0 + 0.0 = 0.48$$

$$L = \frac{1}{2}(0.48 - 1.0)^2 = \frac{1}{2}(−0.52)^2 = 0.1352$$

### 反向传播

从损失 $L$ 开始，逐层回传。

**第 1 步：损失对预测值的梯度**

$$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y = 0.48 - 1.0 = -0.52$$

**第 2 步：输出层参数的梯度**

$$\frac{\partial L}{\partial v_1} = \frac{\partial L}{\partial \hat{y}} \cdot h_1 = -0.52 \times 0.6 = -0.312$$

$$\frac{\partial L}{\partial v_2} = \frac{\partial L}{\partial \hat{y}} \cdot h_2 = -0.52 \times 0 = 0$$

$$\frac{\partial L}{\partial b_3} = \frac{\partial L}{\partial \hat{y}} \cdot 1 = -0.52$$

**第 3 步：梯度传到隐藏层输出**

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial \hat{y}} \cdot v_1 = -0.52 \times 0.8 = -0.416$$

$$\frac{\partial L}{\partial h_2} = \frac{\partial L}{\partial \hat{y}} \cdot v_2 = -0.52 \times (-0.6) = 0.312$$

**第 4 步：通过 ReLU**

$$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h_1} \cdot \mathbf{1}_{z_1 > 0} = -0.416 \times 1 = -0.416 \quad (z_1 = 0.6 > 0)$$

$$\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial h_2} \cdot \mathbf{1}_{z_2 > 0} = 0.312 \times 0 = 0 \quad (z_2 = -0.1 \leq 0)$$

$z_2$ 被 ReLU 截断为 0，梯度也被截断。这就是"死神经元"问题的根源：一旦 ReLU 输入持续为负，该神经元不再更新。

**第 5 步：隐藏层参数的梯度**

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z_1} \cdot x = -0.416 \times 1.0 = -0.416$$

$$\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1} \cdot 1 = -0.416$$

$$\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial z_2} \cdot x = 0 \times 1.0 = 0$$

$$\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2} \cdot 1 = 0$$

### 梯度汇总

| 参数 | $w_1$ | $b_1$ | $w_2$ | $b_2$ | $v_1$ | $v_2$ | $b_3$ |
|------|-------|-------|-------|-------|-------|-------|-------|
| 梯度 | -0.416 | -0.416 | 0 | 0 | -0.312 | 0 | -0.52 |

设学习率 $\eta = 0.1$，更新后：

| 参数 | 更新前 | 更新后 |
|------|--------|--------|
| $w_1$ | 0.5 | $0.5 - 0.1 \times (-0.416) = 0.5416$ |
| $b_1$ | 0.1 | $0.1 - 0.1 \times (-0.416) = 0.1416$ |
| $v_1$ | 0.8 | $0.8 - 0.1 \times (-0.312) = 0.8312$ |
| $b_3$ | 0.0 | $0.0 - 0.1 \times (-0.52) = 0.052$ |

$w_2, b_2, v_2$ 梯度为 0，本轮不更新。

---

## 用 PyTorch 验证手推结果

```python
import torch

x = torch.tensor([1.0])
y = torch.tensor([1.0])

w1 = torch.tensor([0.5], requires_grad=True)
w2 = torch.tensor([-0.3], requires_grad=True)
b1 = torch.tensor([0.1], requires_grad=True)
b2 = torch.tensor([0.2], requires_grad=True)
v1 = torch.tensor([0.8], requires_grad=True)
v2 = torch.tensor([-0.6], requires_grad=True)
b3 = torch.tensor([0.0], requires_grad=True)

# 前向传播
z1 = w1 * x + b1
z2 = w2 * x + b2
h1 = torch.relu(z1)
h2 = torch.relu(z2)
y_hat = v1 * h1 + v2 * h2 + b3
loss = 0.5 * (y_hat - y) ** 2

# 反向传播
loss.backward()

# 打印梯度，与手推结果对比
params = {"w1": w1, "w2": w2, "b1": b1, "b2": b2,
          "v1": v1, "v2": v2, "b3": b3}
for name, p in params.items():
    print(f"d(Loss)/d({name}) = {p.grad.item():.4f}")
```

运行输出：

```
d(Loss)/d(w1) = -0.4160
d(Loss)/d(w2) = 0.0000
d(Loss)/d(b1) = -0.4160
d(Loss)/d(b2) = 0.0000
d(Loss)/d(v1) = -0.3120
d(Loss)/d(v2) = -0.0000
d(Loss)/d(b3) = -0.5200
```

与手推结果完全一致。

---

## 反向传播的计算效率

为什么不直接用数值微分（每个参数加 $\epsilon$ 算差分）？假设网络有 $n$ 个参数：

| 方法 | 前向传播次数 | 适用场景 |
|------|-------------|---------|
| 数值微分 | $2n$（每个参数正、负各一次） | 梯度检验、调试 |
| 反向传播 | 1 次前向 + 1 次反向 | 训练 |

GPT-3 有 1750 亿参数。数值微分需要 $3500$ 亿次前向传播来算一次梯度；反向传播只需 2 次遍历。这就是反向传播的价值——将梯度计算从 $O(n)$ 次前向传播压缩到 $O(1)$ 次。

反向传播的代价是内存：需要缓存前向传播中所有中间值（用于计算局部偏导数）。这也是大模型训练需要大量显存的原因之一。梯度检查点（gradient checkpointing）通过牺牲计算时间换内存，只保存部分中间值、需要时重新计算，是常用的工程优化手段。

---

## 常见坑点

**梯度消失与爆炸**：链式法则是连乘。如果每层的局部梯度 $< 1$，经过几十层连乘后梯度趋近于 0（消失）；如果 $> 1$，则指数增长（爆炸）。解决方案包括残差连接（ResNet）、Layer Normalization、合理的权重初始化（如 He initialization）。

**ReLU 死神经元**：上面的例子中 $h_2$ 就是一个死神经元。Leaky ReLU（负区间给一个小斜率如 0.01）或 GELU 可以缓解此问题。

**学习率敏感性**：梯度下降的收敛高度依赖学习率。实践中很少用固定学习率的 SGD，而是用 Adam（自适应学习率 + 动量），配合学习率 warmup 和 cosine decay 调度。

---

## 参考资料

- Rumelhart, Hinton, Williams. *Learning representations by back-propagating errors*. Nature, 1986. [https://www.nature.com/articles/323533a0](https://www.nature.com/articles/323533a0)
- Stanford CS231n: Backpropagation, Intuitions. [https://cs231n.github.io/optimization-2/](https://cs231n.github.io/optimization-2/)
- PyTorch Autograd 机制文档. [https://pytorch.org/docs/stable/autograd.html](https://pytorch.org/docs/stable/autograd.html)
