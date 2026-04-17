## 核心结论

香农熵（Shannon Entropy，白话解释：一个随机结果到底有多难猜）定义为

$$
H(X)=-\sum_x p(x)\log_2 p(x)
$$

它度量的是离散随机变量 $X$ 的平均不确定性，单位是 bit；如果把对数底换成 $e$，单位就是 nat。结论可以直接记成三条：

1. 分布越均匀，熵越大；分布越偏，熵越小。
2. 条件信息会减少不确定性，所以 $H(Y|X)\le H(Y)$。
3. 互信息（Mutual Information，白话解释：两个变量共享了多少可用信息）满足
   $$
   I(X;Y)=H(X)-H(X|Y)=H(X)+H(Y)-H(X,Y)\ge 0
   $$

最简单的玩具例子是公平硬币。若 $p(H)=p(T)=0.5$，则

$$
H(X)=-(0.5\log_2 0.5+0.5\log_2 0.5)=1
$$

这表示平均每次抛硬币需要 1 bit 才能无损描述。若硬币偏到 $0.9/0.1$，熵会下降，因为结果更容易猜。

这些性质不是孤立公式。它们共同对应三个工程问题：

| 问题 | 对应量 | 含义 |
|---|---:|---|
| 一个源有多难压缩 | $H(X)$ | 无损压缩平均码长下界 |
| 观察了特征后还剩多少不确定性 | $H(Y|X)$ | 条件下的剩余信息量 |
| 特征和标签是否真的相关 | $I(X;Y)$ | 共享信息是否大于 0 |

真实工程例子是语言模型。若模型对下一个 token 的平均负对数概率为 $H$ bit，则困惑度（Perplexity，白话解释：模型平均像在多少个等可能选项里猜）满足

$$
\mathrm{PPL}=2^H
$$

所以 PPL 本质上不是另一个新指标，而是熵的指数形式。

---

## 问题定义与边界

这里讨论的是**离散随机变量**的香农熵，而不是连续变量的微分熵。也就是说，我们默认随机变量的取值集合是可枚举的，例如硬币正反面、词表中的 token、分类标签集合。

问题定义有三个层次：

1. 单变量不确定性：$H(X)$
2. 已知一个变量后的剩余不确定性：$H(Y|X)$
3. 两个变量共享的信息量：$I(X;Y)$

边界先看 $H(X)$。如果 $X$ 只有 $n$ 个可能取值，那么

$$
0 \le H(X)\le \log_2 n
$$

下界 $0$ 出现在“完全确定”的情况，例如某结果概率为 1。上界 $\log_2 n$ 出现在均匀分布，也就是每个结果都一样难猜。

下面用硬币分布做最小计算：

| 分布 $p(H),p(T)$ | 熵 $H(X)$（bit） | 备注 |
|---|---:|---|
| $(1,0)$ | $0$ | 完全确定 |
| $(0.9,0.1)$ | $\approx 0.469$ | 明显偏置 |
| $(0.8,0.2)$ | $\approx 0.722$ | 偏置但不极端 |
| $(0.5,0.5)$ | $1.000$ | 二元分布最大值 |

这个表说明一个常被误解的点：熵不是“结果个数”，而是“概率分布带来的平均不确定性”。两个结果并不自动等于 1 bit，只有在二者等概率时才是 1 bit。

条件熵满足

$$
0\le H(Y|X)\le H(Y)
$$

直观上，知道更多信息不会让你更迷糊。互信息满足

$$
I(X;Y)\ge 0
$$

当且仅当 $X$ 和 $Y$ 独立时，$I(X;Y)=0$。这说明互信息不是“线性相关系数”，它衡量的是更一般的统计相关性。

---

## 核心机制与推导

香农熵的核心不是从“压缩”开始，而是从**惊讶度**开始。惊讶度（self-information，白话解释：某个结果一旦发生，到底有多意外）定义为

$$
I(x)=-\log_2 p(x)
$$

概率越小，惊讶度越大。例如概率 $1/2$ 的事件惊讶度是 1 bit，概率 $1/8$ 的事件惊讶度是 3 bit。香农熵就是惊讶度的期望：

$$
H(X)=\mathbb E[-\log_2 p(X)] = -\sum_x p(x)\log_2 p(x)
$$

这一步很关键。它说明熵不是某个样本的属性，而是整个分布的平均量。

再看联合熵（joint entropy，白话解释：两个变量一起看时还剩多少不确定性）：

$$
H(X,Y)=-\sum_{x,y}p(x,y)\log_2 p(x,y)
$$

利用概率分解 $p(x,y)=p(x)p(y|x)$，可得

$$
\log p(x,y)=\log p(x)+\log p(y|x)
$$

代入联合熵：

$$
\begin{aligned}
H(X,Y)
&=-\sum_{x,y}p(x,y)\log p(x,y) \\
&=-\sum_{x,y}p(x,y)\log p(x)-\sum_{x,y}p(x,y)\log p(y|x) \\
&=H(X)+H(Y|X)
\end{aligned}
$$

这就是链式法则：

$$
H(X,Y)=H(X)+H(Y|X)=H(Y)+H(X|Y)
$$

条件熵的定义因此写成

$$
H(Y|X)=-\sum_{x,y}p(x,y)\log_2 p(y|x)
$$

它表示“知道 $X$ 之后，$Y$ 平均还要多少 bit 才能描述”。

下面给一个玩具例子。设天气 $X\in\{\text{晴},\text{雨}\}$，出门是否带伞 $Y\in\{\text{带},\text{不带}\}$，联合分布为：

| $X$ | $Y$ | 概率 |
|---|---|---:|
| 晴 | 带 | 0.1 |
| 晴 | 不带 | 0.5 |
| 雨 | 带 | 0.3 |
| 雨 | 不带 | 0.1 |

先求边际分布：$p(\text{晴})=0.6,\ p(\text{雨})=0.4$。因此

$$
H(X)=-(0.6\log_2 0.6+0.4\log_2 0.4)\approx 0.971
$$

条件分布中，晴天带伞概率为 $1/6$，雨天带伞概率为 $3/4$。于是

$$
H(Y|X)=0.6\cdot H(1/6,5/6)+0.4\cdot H(3/4,1/4)\approx 0.809
$$

如果直接算联合熵，会得到

$$
H(X,Y)\approx 1.780
$$

可见

$$
H(X)+H(Y|X)\approx 0.971+0.809=1.780
$$

链式法则成立。再算互信息：

$$
I(X;Y)=H(Y)-H(Y|X)
$$

若先算出 $p(Y=\text{带})=0.4,\ p(Y=\text{不带})=0.6$，则 $H(Y)\approx 0.971$，所以

$$
I(X;Y)\approx 0.971-0.809=0.162
$$

说明天气和带伞有相关性，但相关性并不强。

把这个机制放到语言模型里更直观。设序列为 $x_1,\dots,x_N$，模型给出的条件概率为 $P(x_t|x_{<t})$，则平均交叉熵写成

$$
H = -\frac{1}{N}\sum_{t=1}^N \log_2 P(x_t|x_{<t})
$$

这和信息论里的平均惊讶度是同一个结构。若模型越不确定，平均负对数概率越大，熵越高，困惑度也越高：

$$
\mathrm{PPL}=2^H
$$

因此 PPL 可以理解成“平均等价候选数”。

---

## 代码实现

下面的 Python 代码直接实现熵、条件熵、联合熵、互信息和 PPL，并校验链式法则。代码只依赖标准库，可以直接运行。

```python
import math

def entropy(dist):
    """
    dist: dict[value, prob]
    Shannon entropy H(X) in bits.
    """
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-9, f"probabilities must sum to 1, got {total}"
    assert all(p >= 0 for p in dist.values())
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)

def joint_entropy(joint):
    """
    joint: dict[(x, y), prob]
    Joint entropy H(X, Y) in bits.
    """
    total = sum(joint.values())
    assert abs(total - 1.0) < 1e-9, f"probabilities must sum to 1, got {total}"
    assert all(p >= 0 for p in joint.values())
    return -sum(p * math.log2(p) for p in joint.values() if p > 0)

def marginals_from_joint(joint):
    px = {}
    py = {}
    for (x, y), p in joint.items():
        px[x] = px.get(x, 0.0) + p
        py[y] = py.get(y, 0.0) + p
    return px, py

def conditional_entropy_y_given_x(joint):
    """
    H(Y|X) = -sum p(x,y) log p(y|x)
    """
    px, _ = marginals_from_joint(joint)
    value = 0.0
    for (x, y), pxy in joint.items():
        if pxy == 0:
            continue
        py_given_x = pxy / px[x]
        value -= pxy * math.log2(py_given_x)
    return value

def mutual_information(joint):
    px, py = marginals_from_joint(joint)
    hx = entropy(px)
    hy = entropy(py)
    hxy = joint_entropy(joint)
    mi = hx + hy - hxy
    return mi

def perplexity_from_entropy_bits(h_bits):
    return 2 ** h_bits

# 玩具例子 1：公平硬币
coin = {"H": 0.5, "T": 0.5}
h_coin = entropy(coin)
assert abs(h_coin - 1.0) < 1e-9
assert abs(perplexity_from_entropy_bits(h_coin) - 2.0) < 1e-9

# 玩具例子 2：天气-带伞联合分布
joint = {
    ("sunny", "umbrella"): 0.1,
    ("sunny", "no_umbrella"): 0.5,
    ("rainy", "umbrella"): 0.3,
    ("rainy", "no_umbrella"): 0.1,
}

px, py = marginals_from_joint(joint)
hx = entropy(px)
hy = entropy(py)
hxy = joint_entropy(joint)
hy_given_x = conditional_entropy_y_given_x(joint)
mi = mutual_information(joint)

# 链式法则校验
assert abs(hxy - (hx + hy_given_x)) < 1e-9

# 互信息非负
assert mi >= -1e-9

print("H(X) =", round(hx, 6))
print("H(Y) =", round(hy, 6))
print("H(X,Y) =", round(hxy, 6))
print("H(Y|X) =", round(hy_given_x, 6))
print("I(X;Y) =", round(mi, 6))
print("PPL from H(Y) =", round(perplexity_from_entropy_bits(hy), 6))
```

这段实现里最需要注意的是 `if p > 0`。因为 $\log 0$ 不存在，而在信息论约定下，极限 $\lim_{p\to 0^+} p\log p = 0$，所以零概率项应直接跳过。

真实工程例子可以这样理解：如果语言模型在验证集上的平均交叉熵是 1.5 bit/token，那么

$$
\mathrm{PPL}=2^{1.5}\approx 2.828
$$

意思不是“模型只在 2.8 个词里选”，而是“模型的平均不确定性相当于在约 2.8 个等可能候选之间做选择”。

---

## 工程权衡与常见坑

香农熵的公式很短，但工程误用很多，尤其集中在特征选择和知识蒸馏。

第一个坑是**只看边际互信息**。边际互信息（white话解释：单独看一个特征和标签的关系）可能遗漏“只有和别的特征一起看才有用”的交互信号。设二分类任务中，特征 $X_1$、$X_2$ 单看都和标签 $Y$ 相关性很弱，但组合后能决定标签，例如异或结构。此时单独筛特征可能把真正有用的特征误删。

更稳妥的量是条件互信息：

$$
I(X;Y|Z)=H(Y|Z)-H(Y|X,Z)
$$

它衡量“在已经知道 $Z$ 的前提下，$X$ 还能给 $Y$ 带来多少新增信息”。

第二个坑是**把低熵软标签当成高质量知识蒸馏目标**。知识蒸馏里，教师模型输出的 soft label 不只是给出 top-1 类别，还携带类别间相对关系。如果温度太低，softmax 过于尖锐，熵会迅速下降，学生几乎只看到 one-hot 近似，暗知识就消失了。

温度缩放后的 softmax 为

$$
p_i^{(T)}=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

对应熵为

$$
H_T=-\sum_i p_i^{(T)}\log p_i^{(T)}
$$

当 $T$ 上升时，分布通常更平滑，熵更高，学生能接收到更多类别间相似性信息。

| 常见问题 | 触发条件 | 后果 | 缓解策略 |
|---|---|---|---|
| 特征选择漏掉交互特征 | 只看边际 $I(X;Y)$ | 保留表面强特征，删掉组合有效特征 | 用条件互信息或联合选择 |
| MI 估计偏大或偏小 | 样本少、离散化粗糙 | 排序不稳定 | 做分箱敏感性分析或用非参数估计 |
| 熵计算数值错误 | 直接算 $\log 0$ | 得到 `-inf` 或 `nan` | 跳过零概率或加极小常数 |
| 蒸馏信息量不足 | 温度过低，软标签熵太小 | 学生只学到硬标签 | 提高温度并配合校准 |
| PPL 解释错误 | 混淆自然对数和 $\log_2$ | 指标数值不一致 | 明确单位是 bit 还是 nat |

这里给一个真实工程例子。做垃圾邮件分类时，单词 “free” 的边际互信息可能很高，于是它容易入选；但像 “meeting” 这种词单独看几乎没用，和发件域名、历史会话上下文联合后才有意义。如果只做单变量 MI 排序，就会把很多上下文型特征提前丢掉。工程上常用“先粗筛，再做包裹式搜索”或“用条件 MI 追加筛选”来缓解。

---

## 替代方案与适用边界

香农熵最干净的前提是：你知道离散分布，或者至少能较稳定地估计它。但真实任务常常拿不到完整联合分布，此时要考虑替代方法。

第一类替代方案是**基于下界的互信息估计**。如果直接算 $p(x,y)$ 很难，可以用变分分布 $q(x|y)$ 构造下界：

$$
I(X;Y)\ge \mathbb E_{p(x,y)}\left[\log \frac{q(x|y)}{p(x)}\right]
$$

这类方法的优点是可训练、可扩展到高维表示学习；缺点是估计偏差依赖于参数化形式和采样质量。

第二类是**用交叉熵替代直接熵估计**。在语言模型中，通常并不先估计真实数据分布再算熵，而是直接计算模型给测试集的平均负对数似然。这更像“模型视角下的熵”，适合比较模型，不一定等于真实源熵。

第三类是**连续变量场景**。若变量是连续的，例如传感器实值信号，就不能直接套离散熵，而要改用微分熵（differential entropy，白话解释：连续分布版本的熵）：

$$
h(X)=-\int p(x)\log p(x)\,dx
$$

但微分熵和离散熵不同，它可以为负，而且不直接等同于最短编码长度。很多工程任务会先离散化，再计算离散熵或互信息。

| 方法 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| 直接离散熵/MI | 低维离散变量、概率表可得 | 解释最清晰 | 高维时联合表爆炸 |
| 采样估计 MI | 完整分布不可得但可采样 | 可用于模拟和实验 | 方差大，依赖样本量 |
| 变分/KL 下界 | 表示学习、高维神经网络 | 可端到端训练 | 下界松紧受模型限制 |
| 交叉熵/PPL | 语言模型评估 | 易计算、可直接比较模型 | 是模型视角，不一定是真实源熵 |
| 微分熵 | 连续随机变量 | 理论上匹配连续分布 | 解释与编码长度不完全一致 |

一个新手可操作的替代例子是 Monte Carlo。假设拿不到完整联合表，但能不断采样 $(x,y)$，就可以用样本均值近似期望项；不过要明确，这时得到的是估计值，不是解析真值。

因此，香农熵最适合的边界是：变量离散、概率可估、目标是分析不确定性、压缩极限或统计相关性。离开这些前提，公式依然能用，但解释必须换。

---

## 参考资料

1. Shannon entropy and mutual information 的定义与特征选择应用  
   主要贡献：系统给出熵、条件熵、互信息、条件互信息的定义，并讨论特征选择中的使用边界。  
   对应章节：问题定义与边界、工程权衡与常见坑。

2. Conditional entropy 主题资料  
   主要贡献：整理链式法则、联合熵、条件熵、互信息之间的标准关系式。  
   对应章节：核心机制与推导。

3. Perplexity 相关资料  
   主要贡献：说明困惑度与平均负对数概率、熵之间的对应关系，可用于解释语言模型评估。  
   对应章节：核心结论、代码实现。

4. Shannon’s First Theorem / 无噪声源编码定理资料  
   主要贡献：说明任意无损编码的平均码长不能低于信源熵，是“熵等于压缩下界”的理论来源。  
   对应章节：核心机制与推导。

5. 基础信息论教材或课程讲义  
   主要贡献：对“惊讶度”“期望”“链式法则”的推导通常更完整，适合补数学细节。  
   对应章节：全文。
