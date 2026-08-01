# 样式演示

本页集中展示博客支持的全部排版能力：教材式多级标题、数学公式、化学方程式、代码、表格等。
写博文时可直接参考本页的 Markdown 源码。

## 多级标题

文章页的标题自动编号（教材式）：二级标题为章，三级为节，四级为小节。

### 这是一个三级标题

正文内容。编号、字体与间距均由设计系统自动处理，写作时只需要正常写 `#` 标题。

#### 这是一个四级标题

四级标题用于节内的小节，视觉上弱化以拉开层级。

## 数学公式

行内公式：质能方程 $E = mc^2$，以及欧拉恒等式 $e^{i\pi} + 1 = 0$。

块级公式——标准正态分布的概率密度函数：

$$
f(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

多行对齐——反向传播的链式法则：

$$
\begin{aligned}
\frac{\partial L}{\partial W^{[l]}} &= \delta^{[l]} (a^{[l-1]})^T \\
\delta^{[l]} &= (W^{[l+1]})^T \delta^{[l+1]} \odot \sigma'(z^{[l]})
\end{aligned}
$$

矩阵与求和——Softmax 与交叉熵损失：

$$
\mathrm{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \qquad
L = -\sum_{i=1}^{K} y_i \log \hat{y}_i
$$

## 化学方程式

基于 mhchem 宏包，用 `\ce{}` 书写：

$$
\ce{2H2 + O2 ->[点燃] 2H2O}
$$

可逆反应与沉淀气体符号：

$$
\ce{CO2 + C <=> 2CO}, \qquad
\ce{CaCO3 + 2HCl -> CaCl2 + H2O + CO2 ^}
$$

配合物与生物代谢：

$$
\ce{K4[Fe(CN)6]}, \qquad
\ce{ATP + H2O -> ADP + Pi + 能量}
$$

## 代码

```python
import torch

def lora_forward(x, W0, A, B, alpha, r):
    """LoRA: W = W0 + (alpha / r) * B @ A"""
    return x @ W0.T + (alpha / r) * (x @ A.T @ B.T)
```

## 表格与复选框

| 方法 | 显存占用 | 训练参数比例 |
| --- | --- | --- |
| 全参数微调 | 高 | 100% |
| LoRA | 中 | ~0.1–1% |
| QLoRA | 低 | ~0.1–1% |

- [x] 已完成的选题
- [ ] 待写作的选题

## 引用与提示

> 如无必要，勿增实体。 —— 奥卡姆的威廉

> 写作完成后：在本目录新建 `xxx.md`，然后把对应条目改为 `- [x] [标题](./xxx)`。
