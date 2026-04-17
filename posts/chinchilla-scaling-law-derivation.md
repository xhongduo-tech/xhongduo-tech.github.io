## 核心结论

Chinchilla 定律讨论的是“固定训练算力下，参数量和数据量怎么分配最省损失”。结论是：对标准 dense Transformer 语言模型，如果训练损失可以近似写成

$$
L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta},
$$

并且训练算力满足

$$
C \approx 6ND,
$$

那么最优分配是

$$
N^\* = G\left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}, \quad
D^\* = G^{-1}\left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha+\beta}},
$$

其中 $N$ 是参数量，白话说就是“模型有多大”；$D$ 是训练 token 数，白话说就是“模型总共读了多少字”；$C$ 是训练 FLOPs，白话说就是“总算力账单”；$E$ 是不可约误差，白话说就是“再怎么堆资源也消不掉的底噪”；$G=(\alpha A/\beta B)^{1/(\alpha+\beta)}$ 是由拟合常数决定的比例因子。

当 $\alpha \approx \beta$ 时，

$$
N^\* \propto C^{0.5}, \quad D^\* \propto C^{0.5}.
$$

这就是 Chinchilla 的核心：算力增加时，模型参数和训练数据都应按 $\sqrt{C}$ 同步增长，而不是只把钱砸到更大的模型上。

玩具例子：把训练预算想成 100 元。Kaplan 早期结论更像“80 元买更大的模型，20 元买数据”；Chinchilla 说更好的做法是“模型和数据一起扩”。因为只扩模型会遇到“数据喂不饱”，只扩数据会遇到“模型吃不下”，两边都会出现边际收益递减。

真实工程例子：Hoffmann 等人在与 Gopher 相同训练算力下，没有继续做 280B 参数，而是训练了 70B 参数、1.4T tokens 的 Chinchilla，并在 MMLU 上做到 67.5%，比 Gopher 高 7% 以上。这说明“更小但训练充分”的模型，常常比“更大但数据不足”的模型更强。

| 方案 | 参数扩展 | 数据扩展 | 固定算力下的典型问题 |
|---|---:|---:|---|
| Kaplan 式早期理解 | 更偏向增参 | 增长较慢 | 大模型常常欠训练 |
| Chinchilla 式分配 | 与数据近等比 | 与参数近等比 | 模型和数据更平衡 |

---

## 问题定义与边界

这个问题本质上是一个预算分配问题。

我们关心的是测试损失 $L(N,D)$。幂律，白话说就是“资源翻倍，收益按固定指数变化”，它在大模型实验里经常出现。Chinchilla 用下面这个形式描述损失：

$$
L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.
$$

三部分的含义如下：

| 项 | 含义 | 白话解释 |
|---|---|---|
| $E$ | 不可约误差 | 数据和任务本身决定的下限 |
| $A/N^\alpha$ | 容量不足误差 | 模型太小，装不下规律 |
| $B/D^\beta$ | 数据不足误差 | 数据太少，学不全规律 |

约束条件是训练算力：

$$
C \approx 6ND.
$$

这里的 6 来自 Transformer 训练中前向和反向传播的常用 FLOPs 近似。它不是宇宙常数，而是 dense Transformer 里很常用的工程近似。Hoffmann 论文附录还检查了更细致的 FLOP 统计，发现对大方向影响很小。

所以边界也要说清楚：

1. 这套推导主要适用于标准 dense Transformer 预训练。
2. 目标是“固定训练算力下最小化预训练损失”，不是最小化推理成本，也不是最优微调成本。
3. 它依赖幂律拟合有效；如果数据分布、架构、优化器行为明显变了，系数也会变。
4. $C \approx 6ND$ 是近似，不同统计口径会改变常数，甚至改变你拟合出来的指数。

新手可以把它理解成两个漏斗：一个漏斗是模型容量，一个漏斗是训练数据。损失就是漏掉的水。只补一个漏斗，另一个还是会漏，所以总效果不优。

---

## 核心机制与推导

把约束 $D=C/(6N)$ 代入损失：

$$
L(N)=E+\frac{A}{N^\alpha}+B\left(\frac{6N}{C}\right)^\beta.
$$

现在问题变成只对 $N$ 求最优。对 $N$ 求导：

$$
\frac{dL}{dN}
=
-\alpha A N^{-\alpha-1}
+
\beta B 6^\beta C^{-\beta} N^{\beta-1}.
$$

令导数为 0，得到最优点：

$$
\alpha A N^{-\alpha-1}
=
\beta B 6^\beta C^{-\beta} N^{\beta-1}.
$$

整理得

$$
N^{\alpha+\beta}
=
\frac{\alpha A}{\beta B}\left(\frac{C}{6}\right)^\beta.
$$

于是

$$
N^\*=
\left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}
\left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}
=
G\left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}.
$$

再由 $D^\*=C/(6N^\*)$ 得

$$
D^\*=
\left(\frac{\alpha A}{\beta B}\right)^{-\frac{1}{\alpha+\beta}}
\left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha+\beta}}
=
G^{-1}\left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha+\beta}}.
$$

因此最重要的指数是

$$
N^\* \propto C^{\frac{\beta}{\alpha+\beta}}, \quad
D^\* \propto C^{\frac{\alpha}{\alpha+\beta}}.
$$

如果 $\alpha=\beta=0.5$，就直接退化为

$$
N^\* \propto C^{0.5}, \quad D^\* \propto C^{0.5}.
$$

而 token-per-param 比例是

$$
\frac{D^\*}{N^\*}
=
G^{-2}\left(\frac{C}{6}\right)^{\frac{\alpha-\beta}{\alpha+\beta}}.
$$

所以当 $\alpha \approx \beta$ 时，这个比例近似为常数。Hoffmann 的实验表格给出的最优点，大致落在每参数 20 个 token 左右，这就是很多工程实现里常提到的“20 tokens per parameter”。

玩具例子：设 $C=10^{20}$ FLOPs，并取 $D/N=20$。因为 $C=6ND=120N^2$，所以

$$
N\approx \sqrt{\frac{10^{20}}{120}} \approx 9.1\times 10^8,
\quad
D\approx 1.8\times 10^{10}.
$$

这说明在这个预算下，最优点不是“无限堆参数”，而是大约 9 亿参数配 180 亿 token。

真实工程例子：Gopher 是 280B 参数、约 300B tokens；Chinchilla 改成 70B 参数、1.4T tokens，在相近训练算力下效果更好。这里不是“70B 天生优于 280B”，而是“70B 在这个预算下更接近最优分配”。

| 推导步骤 | 数学动作 | 结论 |
|---|---|---|
| 约束代入 | $D=C/(6N)$ | 两变量变一变量 |
| 一阶条件 | $\frac{dL}{dN}=0$ | 找最优平衡点 |
| 解方程 | 求出 $N^\*$ | 得到参数最优缩放 |
| 回代约束 | $D^\*=C/(6N^\*)$ | 得到数据最优缩放 |

---

## 代码实现

下面这段代码把公式直接写成可运行函数。它适合做训练预算粗估、配置搜索前的初始点生成。

```python
import math

def chinchilla_budget(C, alpha=0.5, beta=0.5, A=1.0, B=1.0):
    """
    C: 总训练 FLOPs
    alpha, beta: 损失幂律指数
    A, B: 两个误差项的系数
    返回:
        N_opt: 最优参数量
        D_opt: 最优训练 token 数
        tokens_per_param: 每个参数对应多少 token
    """
    assert C > 0
    assert alpha > 0 and beta > 0
    assert A > 0 and B > 0

    G = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    N_opt = G * (C / 6.0) ** (beta / (alpha + beta))
    D_opt = (1.0 / G) * (C / 6.0) ** (alpha / (alpha + beta))
    tokens_per_param = D_opt / N_opt

    # 一致性检查：应满足 C ≈ 6ND
    assert math.isclose(6.0 * N_opt * D_opt, C, rel_tol=1e-9)
    return N_opt, D_opt, tokens_per_param


def budget_from_ratio(C, tokens_per_param=20.0):
    """
    在 alpha = beta 且 D/N 固定时，用更简单的公式估算。
    """
    assert C > 0 and tokens_per_param > 0
    N = math.sqrt(C / (6.0 * tokens_per_param))
    D = tokens_per_param * N
    assert math.isclose(6.0 * N * D, C, rel_tol=1e-9)
    return N, D


# 玩具例子：C = 1e20 FLOPs, 假设最优 token/param = 20
N, D = budget_from_ratio(1e20, tokens_per_param=20.0)
assert 9e8 < N < 9.2e8
assert 1.8e10 < D < 1.84e10

# 一般公式：alpha = beta = 0.5 且 A = B 时，token/param 应接近 1
N2, D2, r2 = chinchilla_budget(1e20, alpha=0.5, beta=0.5, A=1.0, B=1.0)
assert math.isclose(r2, 1.0, rel_tol=1e-9)

print(f"N ~= {N:.3e}, D ~= {D:.3e}, D/N ~= {D/N:.2f}")
```

如果你在真实项目里做预算，通常做法不是盲信一个常数，而是：

1. 先用上面函数给出初始点。
2. 再结合数据上限、吞吐、上下文长度、显存约束做修正。
3. 最后在几个邻近点上做小规模实验，验证损失曲线。

示例预算表：

| 总算力 $C$ | 若取 $D/N=20$ | 最优参数量 $N$ | 最优 token 数 $D$ |
|---:|---:|---:|---:|
| $10^{20}$ | 20 | $9.1\times10^8$ | $1.8\times10^{10}$ |
| $10^{22}$ | 20 | $9.1\times10^9$ | $1.8\times10^{11}$ |
| $10^{24}$ | 20 | $9.1\times10^{10}$ | $1.8\times10^{12}$ |

---

## 工程权衡与常见坑

Chinchilla 和 Kaplan 最大的表面冲突，是最优参数缩放指数不同：

- Kaplan：$N^\* \propto C^{0.73}$
- Chinchilla：$N^\* \propto C^{0.50}$

后续工作表明，偏差的重要来源不是“谁数学推错了”，而是“统计口径和实验范围不同”。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 只统计 non-embedding 参数 | 会把小模型区间的指数拟合偏大 | 统计总参数 |
| 只统计主干 FLOPs，不含 embedding / final logits | 预算口径不一致 | 统计总训练 FLOPs |
| 主要在小模型区间拟合 | 外推到大模型时失真 | 覆盖更大尺度 |
| 学习率调度不随训练 token 调整 | 低估多训练 token 的收益 | schedule 与 token 数匹配 |
| 把 $6ND$ 当精确值 | 误把近似当定律 | 用它做一阶预算，不做绝对真值 |

Pearce 和 Song 的论文给出一个很重要的解释：Kaplan 主要看 non-embedding 参数，并在更小模型范围上做分析；如果在 Chinchilla 的框架下复现这两个条件，确实能得到接近 0.73 的局部指数。这说明“0.73”更多是小尺度和特定计量方式下的局部现象，不是大尺度 total-parameter 视角下的全局最优规律。

工程上还有一个容易忽略的点：Chinchilla 关心的是预训练 compute-optimal，不等于部署 optimal。你为了线上推理成本，也可能故意选更小模型；你为了已有高质量数据上限，也可能故意少于 20 token/param。这都不是违背定律，而是在改目标函数。

---

## 替代方案与适用边界

Chinchilla 不是“所有模型都必须严格 20 token/param”，它只是 dense Transformer 在标准预训练设定下的一条强经验规律。

需要调整的场景包括：

| 场景 | 为什么要调整 | 典型策略 |
|---|---|---|
| $\alpha \neq \beta$ | 参数和数据的边际收益不对称 | 按通式重算，不强行 $\sqrt{C}$ |
| 专业小语料 | 数据上限先到 | 优先扩数据来源、清洗、tokenizer |
| 稀疏/MoE 架构 | 有效参数与激活参数不同 | 用实际激活 FLOPs 重估 |
| 长上下文训练 | FLOPs 结构变化 | 不只看 $6ND$，要看 attention 开销 |
| 以推理成本为主 | 训练最优不等于部署最优 | 适当偏向更小模型 |

一个真实工程边界例子：你做法律、医疗、芯片设计这类专业模型，可能根本拿不到足够大的高质量语料。此时瓶颈不是“模型不够大”，而是“数据质量和覆盖面不够”。如果你还机械地按 $\sqrt{C}$ 扩参数，最后可能只是得到一个更大的过拟合器。

可以把决策过程写成很短的规则：

1. 先用 Chinchilla 给出 $N^\*, D^\*$。
2. 检查你是否真的拿得到 $D^\*$ 对应的数据。
3. 如果拿不到，优先考虑数据扩充、去重、清洗、tokenizer。
4. 如果架构不是标准 dense Transformer，重算 FLOP 口径。
5. 再在邻近的 2 到 4 个点上做小规模验证。

所以，Chinchilla 更像“默认基线”，不是“不可修改的宗教规则”。

---

## 参考资料

| 来源 | 形式 | 重点内容 | 链接 |
|---|---|---|---|
| Hoffmann et al., *Training Compute-Optimal Large Language Models* | 论文 | 提出 Chinchilla 结论，给出等比扩展与 70B/1.4T 实证 | https://arxiv.org/pdf/2203.15556.pdf |
| OpenAI, *Scaling Laws for Neural Language Models* | 论文/官方页面 | Kaplan 缩放律与早期 compute-optimal 结论来源 | https://openai.com/index/scaling-laws-for-neural-language-models/ |
| EPFL EE-628 Lecture 02 | 讲义 | 用教学视角整理 $L(N,D)$ 与 compute allocation | https://www.epfl.ch/labs/lions/wp-content/uploads/2025/04/ee-628-Lecture_02_Optimization-and-Hyperparameter-Transfer.pdf |
| Pearce and Song, *Reconciling Kaplan and Chinchilla Scaling Laws* | TMLR 论文 | 解释 0.73 与 0.50 的差异主要来自 non-embedding 统计和小尺度拟合 | https://openreview.net/pdf?id=NLoaLyuUUF |
