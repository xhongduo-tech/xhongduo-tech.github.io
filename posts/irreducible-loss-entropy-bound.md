## 核心结论

在 scaling law 里，常数项 $E$ 或 $L_0$ 不是“拟合残差”，而是训练数据分布对应的不可约损失。不可约损失可以直白理解为：数据里还剩多少无法再压缩的不确定性。模型再大、训练再久，loss 也只能逼近它，不能稳定穿透它。

因此，一个更完整的视角是：

$$
L(C)=E+\text{reducible loss}
$$

其中 reducible loss 是“还能靠更多参数、更多数据、更多算力继续压下去的部分”；$E$ 是“天花板以下最后那层地板”。如果把语言建模 loss 看成“预测下一个 token 时平均还会错多少信息量”，那么 $E$ 对应的就是数据本身的熵下界。熵第一次出现时可以先记成一句白话：它衡量的是“这个分布天然有多难预测”。

对新手最重要的认识有两点：

1. 同样是语言模型，loss 的最低可达值并不只由模型决定，还由语料本身决定。
2. 不同语料的 $E$ 不同，所以直接比较 perplexity 往往会把“数据更干净”误看成“模型更强”。

Shannon 的人类猜字实验给英文书面语一个大约 $0.6$ 到 $1.3$ bits/char 的范围，后续模拟工作把实际值推到约 $1.1$ bits/char 或更低。这说明自然语言确实有相当强的冗余，但它不是零冗余，所以交叉熵不可能无限降到 0。注意这里的单位是“每字符 bit”，而现代 LLM 常见训练 loss 是“每 token nat”，二者不能直接数值对比，只能说明同一个原则：语言有可压缩部分，也有不能被消灭的剩余不确定性。

---

## 问题定义与边界

这篇文章只讨论 scaling law 里的常数项 $E$，不讨论 optimizer、架构技巧、RLHF 或推理时采样策略。边界要先划清，否则“不可约损失”这个词很容易被误解成“模型学不会”。

更准确地说：

- irreducible loss：验证分布上的理论下界，来自数据分布本身
- reducible loss：模型还没有学到的部分，随着规模增加会下降
- 总 loss：两者相加

Gadre 等人的写法把这个关系明确写成了公式，而不是把最低点当作经验现象。这样做的价值是：当你换语料时，曲线不只是“斜率不同”，还可能“整条曲线整体上移或下移”。

下面先看论文 Table 6 中三个语料对应的 $E$。原文单位是 nats/token。为了更直观，也给出 bits/token，换算公式是：

$$
\text{bits/token}=\frac{\text{nats/token}}{\ln 2}
$$

| 训练语料 | $E$（nats/token） | 约等于 bits/token | 相对 C4 的差值（nats） | 含义 |
| --- | ---: | ---: | ---: | --- |
| C4 | 1.51 | 2.18 | 0.00 | 三者中最低，下界更低 |
| RefinedWeb | 1.73 | 2.50 | 0.22 | 比 C4 更难压低 |
| RedPajama | 1.84 | 2.65 | 0.33 | 三者中最高，下界最高 |

这张表有两个直接结论。

第一，不同预训练集的 $E$ 差异是可测的，不是概念上的装饰。C4 和 RedPajama 的差距约为 $0.33$ nats/token，也就是约 $0.48$ bits/token。这不小，足以影响你对“同算力下谁更好”的判断。

第二，perplexity 本质上是交叉熵的指数形式。如果 loss 用 nats 表示，那么：

$$
\text{PPL}=e^L
$$

于是即便两个模型的 reducible loss 一样，只要训练语料的 $E$ 不同，最终 perplexity 也会天然错开。这就是为什么“跨语料直接比 perplexity”经常不成立。

玩具例子可以这样理解。假设有两个猜数字游戏：

- 游戏 A：答案永远只在 $\{0,1\}$ 中
- 游戏 B：答案均匀分布在 $\{0,\dots,9\}$ 中

即使你给两个玩家同样强的记忆力、同样多的练习，B 的最低平均错误信息量也一定更高，因为题本身更不确定。这里“题本身更不确定”，就是 $E$ 更高。

---

## 核心机制与推导

Gadre 等人从经典的联合缩放式出发，把 loss 看成参数规模 $N$ 和训练 token 数 $D$ 的函数。再把它重参数化成算力 $C$ 和 token multiplier $M$，得到更适合讨论“过训练”问题的形式。

几个变量先定义清楚：

- $N$：参数量
- $D$：训练 token 数
- $C=6ND$：训练 FLOPs 的近似
- $M=D/N$：token multiplier，可直白理解为“每个参数平均看了多少 token”
- $\eta$：幂律指数，决定缩放收益有多快衰减

论文中的形式是：

$$
L(C,M)=E+\left(aM^{\eta}+bM^{-\eta}\right)C^{-\eta}
$$

这个式子最关键的不是复杂，而是结构非常清楚：

1. $E$ 单独站在前面，表示不可约下界。
2. 后面整项都是 reducible loss。
3. 当 $C$ 变大时，$C^{-\eta}$ 下降，所以 reducible loss 下降。
4. 当 $M$ 改变时，主要影响的是前面的系数 $aM^\eta+bM^{-\eta}$，也就是“曲线往上还是往下平移”。

这也是论文想强调的观察：在不同过训练倍率下，指数 $\eta$ 大体稳定，变化主要体现在系数项，而不是斜率突然变掉。换句话说，增加过训练不是把世界换了一套规律，而是在同一套幂律里换了一个偏移量。

为什么会有 $M^\eta$ 和 $M^{-\eta}$ 两项？直观上是因为模型和数据两个方向都可能成为瓶颈：

- 参数太少，模型容量不够
- 数据太少，模型见识不够

当你把 $D$ 和 $N$ 重写成 $C$ 与 $M$ 的函数时，这两个瓶颈就分别变成了关于 $M$ 的正幂和负幂项。于是存在一个最优 $M^\*$，让这两类瓶颈达到平衡。

如果只看固定语料、固定 $M$，式子还能进一步简化成：

$$
L(C)=E+\lambda(M)\,C^{-\eta}
$$

其中

$$
\lambda(M)=aM^\eta+bM^{-\eta}
$$

所以在 log-log 图上，$\log(L-E)$ 对 $\log C$ 近似是一条斜率为 $-\eta$ 的直线。改变 $M$，通常更像平移；改变 $E$，则像把整张图的“地板高度”换掉。

玩具例子可以用两个语料来理解：

- 语料 A：百科、教材、清洗充分、重复少
- 语料 B：论坛抓取、模板页多、噪声高、上下文断裂多

假设两个语料都用同样的 $a,b,\eta,M,C$，那最后 loss 的差异主要就会落在 $E_A$ 和 $E_B$ 上。你会看到两条曲线下降趋势相似，但 B 始终悬在更高的位置。这不是模型“懒”，而是地板更高。

真实工程例子是 C4、RefinedWeb、RedPajama。按照 Table 6，三者的拟合结果分别是：

- C4：$E=1.51,\ a=141,\ b=190,\ \eta=0.121$
- RedPajama：$E=1.84,\ a=212,\ b=367,\ \eta=0.136$
- RefinedWeb：$E=1.73,\ a=157,\ b=246,\ \eta=0.127$

这组数说明两件事。第一，$E$ 不同，地板不同。第二，连 reducible 部分的系数也不同，意味着“离地板还有多远”这件事也受数据分布影响。工程上如果只盯着最终 loss，而不拆解出 $E$，就会把两种效应混在一起。

---

## 代码实现

下面给一个可以直接运行的 Python 脚本。它做三件事：

1. 按论文形式计算 $L(C,M)$
2. 比较 C4、RedPajama、RefinedWeb 三条曲线
3. 用 `assert` 检查“算力越大，loss 越接近但不会低于 $E$”这个性质

```python
import math

DATASETS = {
    "C4": {"E": 1.51, "a": 141.0, "b": 190.0, "eta": 0.121},
    "RedPajama": {"E": 1.84, "a": 212.0, "b": 367.0, "eta": 0.136},
    "RefinedWeb": {"E": 1.73, "a": 157.0, "b": 246.0, "eta": 0.127},
}

def loss(C, M, E, a, b, eta):
    reducible = (a * (M ** eta) + b * (M ** (-eta))) * (C ** (-eta))
    return E + reducible

def ppl_from_nats(loss_value):
    return math.exp(loss_value)

def nats_to_bits(x):
    return x / math.log(2)

def demo():
    M = 20.0
    computes = [1e18, 3e18, 1e19, 3e19, 1e20]

    for name, cfg in DATASETS.items():
        losses = [loss(C, M, **cfg) for C in computes]

        # loss 随算力增加而下降，但不会低于 E
        assert all(losses[i] > losses[i + 1] for i in range(len(losses) - 1))
        assert all(v > cfg["E"] for v in losses)

        print(f"\n{name}")
        print(f"E = {cfg['E']:.2f} nats/token = {nats_to_bits(cfg['E']):.2f} bits/token")
        for C, L in zip(computes, losses):
            print(f"C={C:.1e}, loss={L:.4f}, ppl={ppl_from_nats(L):.2f}")

    # 同样的算力和 M，下界更低的语料通常更容易达到更低 loss
    c4_loss = loss(1e20, 20.0, **DATASETS["C4"])
    rp_loss = loss(1e20, 20.0, **DATASETS["RedPajama"])
    assert c4_loss < rp_loss

if __name__ == "__main__":
    demo()
```

如果你把它跑起来，会看到一个很稳定的现象：随着 $C$ 增大，三条曲线都在下降，但下降速度逐渐变慢，并分别逼近自己的 $E$。这就是“幂律下降 + 熵下界”的最小复现。

读代码时可以抓住两点：

- `reducible` 是后天可优化项
- `E` 是先天数据下界

如果想继续扩展，可以把多组 $M$ 也扫一遍，例如 $M=10,20,40,80$。这会让你看到：同一个数据集上，改变 $M$ 主要在改变曲线距地板的高度，而不是把地板本身换掉。

---

## 工程权衡与常见坑

工程里最常见的错误，不是公式写错，而是比较对象不一致。

第一类坑是直接比较不同训练语料上的 perplexity。比如：

- 模型 A 在 C4 上训练，eval 也在 C4 风格数据上
- 模型 B 在 RedPajama 上训练，eval 在另一套网页语料上

最后你拿两组 perplexity 一比，得出“B 比 A 强”或“A 比 B 强”。这个结论常常不成立，因为你混入了至少三层变化：

- 训练分布不同
- 验证分布不同
- 各自的 $E$ 不同

第二类坑是把更低 loss 自动解释成“架构更好”。如果一个实验同时改了模型结构和数据清洗流程，那么 loss 下降到底来自哪一部分？很可能是：

- 一部分来自 reducible loss 下降，说明模型或训练法确实更有效
- 另一部分来自 $E$ 下降，说明数据分布本身更干净、更一致、更可预测

这两种收益都重要，但含义不同。

第三类坑是把 Shannon 的字符熵和 LLM 的 token 交叉熵直接数值对齐。两者单位不同，tokenizer 也不同。正确说法应该是：Shannon 类实验证明自然语言存在非零熵下界，支持“不可约损失”这个观念；但不能把 $1.1$ bits/char 直接拿去当现代 tokenizer 下的 token loss 下限。

更稳妥的对比流程可以写成：

1. 固定评测集，避免验证分布漂移。
2. 明确 loss 单位，是 nats/token 还是 bits/token。
3. 尽量拟合出各自的 $E$，把总 loss 拆成“下界 + 可缩减部分”。
4. 比较 reducible loss 的下降速度，再比较最终总 loss。
5. 如果要比较“数据好坏”，优先看 $E$ 与去重、过滤、上下文完整性之间的关系。

真实工程例子：同样预算训练两个 7B 左右模型，一个用偏原始抓取的网页语料，一个用重度去重和质量打分后的语料。后者的最终验证 loss 更低，未必意味着模型结构更强，可能只是数据地板更低。这时如果你要为下一轮训练投更多卡，关键问题不是“继续堆参数吗”，而是“先不先继续降 $E$”。

---

## 替代方案与适用边界

不是所有 scaling law 文献都写 $E$，很多会写成 $L_\infty$。两者在讨论语境里通常指向同一个东西：当规模趋于无穷大时，loss 逼近的常数下界。Emergent Mind 的综述就把这类形式概括为：

$$
L(x)=L_\infty+\left(\frac{x_0}{x}\right)^\alpha
$$

这里的核心思想与本文一致：

- $L_\infty$ 是不可约项
- 幂律项是可缩减项
- 规模增加带来收益，但收益递减

这类写法的优点是通用，适合讲“幂律 + 常数”这个大结构；缺点是对语言模型里的过训练问题不够细，因为它没有显式把 $M=D/N$ 拆出来。

能不能通过工程手段降低 $E$？可以，但边界很明确。

常见手段有：

- 去重：减少模板页、镜像页、训练集内部重复
- 质量过滤：去掉乱码、低信息密度文本、SEO 垃圾页
- 上下文修复：保留更完整的文档边界，减少截断带来的伪随机性
- 语种与领域控制：减少混杂分布，让语料目标更一致

这些做法本质上都在尝试改变训练数据分布，使它更可预测、更一致，于是 $E$ 下降。但它不能无限下降，因为自然语言本身就不是确定性序列。人类写作包含选择、歧义、省略、世界知识依赖和真正的新信息，这些都会留下不可压缩的剩余熵。

Shannon 和 Mahoney 的结果正好给这个边界一个直观支点。即使在字符级、即使让人类参与猜测，英文书面语的熵也不是 0，而是在一个非零区间里。换到现代 LLM，只是单位和 tokenization 变了，不是“不可约下界消失了”。

所以适用边界可以总结为：

- 当你在同类语料、同类 tokenizer、同类评测域下比较模型时，$E$ 是很有用的解释变量。
- 当你跨语种、跨 tokenizer、跨任务、跨评测域时，$E$ 依然有概念价值，但数值比较必须非常谨慎。
- 当训练设置离幂律区间太远，比如数据极小、优化不稳定、架构突变明显时，用单一 $E$ 去解释全部现象会失效。

---

## 参考资料

1. Gadre, S. Y. et al. “Language models scale reliably with over-training and on downstream tasks.”  
   URL: https://arxiv.org/pdf/2403.08540  
   作用：给出本文核心公式 $L(C,M)=E+(aM^\eta+bM^{-\eta})C^{-\eta}$，并在 Table 6 中报告 C4、RedPajama、RefinedWeb 的 $E,a,b,\eta$ 拟合值。

2. Mahoney, Matt. “Refining the Estimated Entropy of English by Shannon Game Simulation.”  
   URL: https://mattmahoney.net/dc/entropy1.html  
   作用：整理 Shannon 猜字实验及后续模拟，给出英文书面语约 $0.6$ 到 $1.3$ bits/char、并进一步估计约 $1.1$ bpc 或更低的讨论，为“自然语言存在非零熵下界”提供直观背景。

3. Emergent Mind. “Data Scaling Laws in ML.”  
   URL: https://www.emergentmind.com/topics/data-scaling-laws  
   作用：总结“power-law plus constant”家族的统一写法，把 $L_\infty$ 明确解释为不可约损失，适合把语言模型的 $E$ 放回更一般的 scaling law 框架中理解。
