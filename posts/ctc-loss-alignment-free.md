## 核心结论

CTC（Connectionist Temporal Classification，连接时序分类）是一种用于**输入帧和输出标签没有逐帧对齐**时的损失函数。白话说，就是模型知道一句话里有哪些字符或词，但不知道它们具体出现在第几帧，于是 CTC 不要求人工标注每一帧对应哪个标签，而是把所有可能的对齐方式一起考虑。

它的核心不是“找一条最好路径”，而是**把所有能收缩成目标序列的路径概率求和**。设输入为 $X=(x_1,\dots,x_T)$，目标标签为 $Y=(y_1,\dots,y_U)$，则

$$
P(Y|X)=\sum_{\pi\in\mathcal{B}^{-1}(Y)}\prod_{t=1}^T P(\pi_t|x_t)
$$

其中 $\pi$ 是一条逐帧路径，$\mathcal{B}$ 是 collapse 操作：先合并连续重复符号，再删除 blank。blank 是“空白符号”，白话说就是“这一帧不输出真实标签”。

训练目标就是负对数似然：

$$
L_{CTC}=-\log P(Y|X)
$$

这件事的直接意义有两点：

1. 训练不依赖逐帧对齐标注，只需要整段标签。
2. 梯度来自所有有效路径，而不是单一路径，因此学习更稳定。

先看一个最小玩具例子。输入长度 $T=3$，目标标签是 `"a"`。三帧输出只有两类：blank 和 `a`，概率分别是：

| 帧 | blank | a |
|---|---:|---:|
| t=1 | 0.6 | 0.4 |
| t=2 | 0.3 | 0.7 |
| t=3 | 0.5 | 0.5 |

在这个例子里，按照给定设定，可视作两条有效路径共同贡献目标 `"a"`：

- $[\text{blank}, a, a]$，概率 $0.6\times0.7\times0.5=0.21$
- $[a, a, a]$，概率 $0.4\times0.7\times0.5=0.14$

所以

$$
P(a|X)=0.21+0.14=0.35,\qquad L=-\log 0.35
$$

这个例子说明：**最终 loss 由多条对齐路径共同决定**。CTC 解决的不是“哪一帧对哪个字符”，而是“所有合理对齐加起来有多大概率”。

---

## 问题定义与边界

CTC 处理的是**单调序列映射**问题。单调的意思是输出顺序必须和输入时间顺序一致，不能倒着来，也不能跨时间随意重排。白话说，语音中的第一个字通常不会对应到音频末尾，文本行里的左边字符通常不会对应到右边图像区域。

典型任务有两类：

| 任务 | 输入 | 输出 | 是否常用 CTC |
|---|---|---|---|
| 语音识别 ASR | 声学帧序列 | 字符、拼音、子词 | 是 |
| OCR 行识别 | 图像特征序列 | 字符串 | 是 |
| 机器翻译 | 源语言 token | 目标语言 token | 通常否 |
| 图像描述 | 图像特征 | 自然语言句子 | 通常否 |

原因在于，CTC 默认两个前提。

第一，**输出和输入是单调对齐的**。例如音频从左到右说出 “hello”，标签顺序也必须从左到右生成。

第二，**给定输入后，各时间步输出条件独立**。条件独立的意思是：CTC 在概率定义上通常写成 $\prod_t P(\pi_t|x_t)$，即每一帧的发射概率只由当前时刻的网络输出决定，而不显式依赖前一帧输出了什么。这让训练和流式部署更简单，但也削弱了长程上下文建模能力。

看一个直观例子。假设 5 帧音频要输出 `"hi"`，每帧只能选 `h`、`i`、blank。下面这些路径 collapse 后都可能得到 `"hi"`：

| 帧路径 | collapse 结果 | 说明 |
|---|---|---|
| `[h, blank, i, i, blank]` | `hi` | blank 负责分隔，重复 `i` 被压缩 |
| `[h, h, blank, i, blank]` | `hi` | 连续 `h` 压缩成一个 `h` |
| `[blank, h, blank, i, i]` | `hi` | 开头和结尾都可以留空 |

这就是 CTC 的边界定义：它不需要告诉模型“第 2 帧一定是 `h`”，而是允许模型在所有能变成 `"hi"` 的路径上分配概率。

标签扩展过程通常写成：

$$
Y=(y_1,\dots,y_U)\quad\Rightarrow\quad \bar{Y}=[\phi,y_1,\phi,y_2,\phi,\dots,y_U,\phi]
$$

这里 $\phi$ 表示 blank。比如目标 `"ab"` 会扩展成：

$$
\bar{Y}=[\phi,a,\phi,b,\phi]
$$

这个扩展序列不是最终输出，而是动态规划的状态空间。它的作用是把“停留在 blank”“输出当前字符”“从前一个字符跳到下一个字符”统一进一个递推框架里。

---

## 核心机制与推导

CTC 的关键机制有三步：定义路径、定义 collapse、用前向后向算法高效求和。

### 1. 路径与 collapse

路径 $\pi=(\pi_1,\dots,\pi_T)$ 是逐帧标签序列，每个 $\pi_t$ 属于“真实标签集合 + blank”。

collapse 操作 $\mathcal{B}$ 分两步：

1. 合并连续重复符号
2. 删除 blank

例如：

- $[a,a,\phi,b,b] \Rightarrow [a,\phi,b] \Rightarrow ab$
- $[\phi,h,h,\phi,i] \Rightarrow [\phi,h,\phi,i] \Rightarrow hi$

因此同一个目标序列会对应很多条路径。

### 2. 前向变量

前向变量 $\alpha_t(s)$ 表示：在时刻 $t$，走到扩展标签 $\bar{Y}$ 的第 $s$ 个位置时，所有合法路径的总概率。

设扩展序列长度为 $S=2U+1$。对于目标 `"ab"`，有：

$$
\bar{Y}=[\phi,a,\phi,b,\phi]
$$

在时刻 $t$ 的状态 $s$，发射概率记为 $p_t(\bar{Y}_s)$。递推分两种情况。

如果当前是 blank，或者当前字符和前两个位置字符相同，不能跨两格跳转，只能从本位或前一位来：

$$
\alpha_t(s)=\big(\alpha_{t-1}(s)+\alpha_{t-1}(s-1)\big)\cdot p_t(\bar{Y}_s)
$$

如果当前是新字符，且和 $s-2$ 位置不同，则允许额外从 $s-2$ 跳转：

$$
\alpha_t(s)=\big(\alpha_{t-1}(s)+\alpha_{t-1}(s-1)+\alpha_{t-1}(s-2)\big)\cdot p_t(\bar{Y}_s)
$$

这里“跳两格”的直觉是：从前一个字符的 blank 之后，直接进入新字符，避免把相邻相同字符错误压缩掉。

### 3. 为什么最终概率只看最后两个状态

目标序列全部生成完时，路径可以停在最后一个真实字符，也可以停在结尾 blank，所以

$$
P(Y|X)=\alpha_T(S-1)+\alpha_T(S)
$$

如果用 0 下标写法，就是最后两列相加。

### 4. 后向变量与梯度

后向变量 $\beta_t(s)$ 表示：从时刻 $t$ 的状态 $s$ 出发，到序列结束的所有合法后缀路径总概率。它和 $\alpha_t(s)$ 组合后可以得到“某时刻某状态对总概率的贡献”。

梯度写成：

$$
\frac{\partial L}{\partial y_k^t}
=
-\frac{1}{P(Y|X)}
\sum_{s:\bar{Y}_s=k}
\frac{\alpha_t(s)\beta_t(s)}{y_k^t}
$$

意思是：时刻 $t$ 上所有发射符号 $k$ 的状态贡献都要加起来。白话说，不是某一条路径在负责这个时刻，而是**所有经过该符号的有效路径共同分摊梯度**。

### 简化推导例子：目标 `"ab"`

扩展后：

$$
[\phi,a,\phi,b,\phi]
$$

假设在 $t=2$ 计算状态 `b` 的前向概率。它可以从三处来：

- 上一帧仍在 `b`
- 上一帧在前一个 blank
- 上一帧在 `a`，直接跨过中间 blank 到 `b`

所以 `b` 的前向累积会包含三项。这就是 CTC 动态规划的本质：**把所有合法对齐路径的局部转移压缩成有限状态递推**，避免显式枚举指数级路径数。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它实现的是 CTC 前向算法，输入逐帧概率和目标标签，输出目标序列概率。代码使用普通概率空间，便于理解；真实工程里通常改成 log-space，即对数空间计算，用来避免长序列概率连乘导致的下溢。

```python
import math

BLANK = "_"

def extend_target(target, blank=BLANK):
    ext = [blank]
    for ch in target:
        ext.append(ch)
        ext.append(blank)
    return ext

def ctc_forward_prob(frame_probs, target, blank=BLANK):
    """
    frame_probs: List[Dict[str, float]]
    target: str, e.g. "ab"
    """
    y_ext = extend_target(target, blank)
    T = len(frame_probs)
    S = len(y_ext)

    alpha = [[0.0] * S for _ in range(T)]

    # 初始化
    alpha[0][0] = frame_probs[0].get(blank, 0.0)
    if S > 1:
        alpha[0][1] = frame_probs[0].get(y_ext[1], 0.0)

    for t in range(1, T):
        for s in range(S):
            emit = frame_probs[t].get(y_ext[s], 0.0)

            total = alpha[t - 1][s]  # stay
            if s - 1 >= 0:
                total += alpha[t - 1][s - 1]  # move by 1

            # move by 2 only when current symbol is not blank
            # and not equal to symbol two steps back
            if (
                s - 2 >= 0
                and y_ext[s] != blank
                and y_ext[s] != y_ext[s - 2]
            ):
                total += alpha[t - 1][s - 2]

            alpha[t][s] = total * emit

    if S == 1:
        return alpha[T - 1][0]
    return alpha[T - 1][S - 1] + alpha[T - 1][S - 2]


# 玩具例子：T=3, target="a"
probs = [
    {BLANK: 0.6, "a": 0.4},
    {BLANK: 0.3, "a": 0.7},
    {BLANK: 0.5, "a": 0.5},
]

p = ctc_forward_prob(probs, "a")
assert abs(p - 0.65) < 1e-9, p

loss = -math.log(p)
assert loss > 0
print("P(target|X) =", p)
print("CTC loss =", loss)
```

这个实现有三个关键点。

第一，状态空间不是原始标签 `"ab"`，而是扩展后的 `["_", "a", "_", "b", "_"]`。

第二，状态转移有三种来源：

- 留在当前状态
- 从前一个状态移动
- 必要时从前两个状态跳转

第三，相邻重复字符必须谨慎处理。比如目标 `"ll"`，如果不借助 blank 分隔，collapse 后会错误地变成单个 `"l"`。所以 CTC 需要 blank 来表达“同一字符跨时间延长”和“两个相同字符确实是两个位置”这两种不同语义。

再看解码。训练时我们做的是路径求和；推理时要从模型输出中恢复文本。常见两种方法如下：

| 方法 | 思路 | 优点 | 缺点 |
|---|---|---|---|
| Greedy | 每帧取最大概率符号，再 collapse | 快 | 容易错过全局高概率结果 |
| Beam Search | 保留多个候选前缀，累计路径总概率 | 更准 | 更慢，占用更多内存 |

真实工程例子是流式语音识别。声学模型每 20ms 输出一帧 logits，经过 softmax 得到字符或子词分布，再交给 CTC beam search。beam search 通常还会同时维护“以 blank 结尾”和“以非 blank 结尾”的两种概率，因为它们在重复字符处理上行为不同。这一设计能显著降低词错误率，但会增加延迟和计算量。

---

## 工程权衡与常见坑

CTC 在工程上很好用，但它不是“只要能跑就稳定”的损失函数。

第一类问题是**解码误差**。Greedy 只看逐帧最大值，不看路径总和，所以很容易选错。一个常见现象是：某条路径每一帧都不一定最大，但所有能 collapse 成同一文本的路径加起来，总概率反而更高。Beam search 的价值就在这里，它优化的是前缀总概率，而不是单帧局部最优。

第二类问题是**blank 过多**。训练初期模型很容易学会“多数帧输出 blank”，因为这在长序列上是更安全的策略。结果是输出过短、召回偏低。常见缓解方法包括：

| 问题 | 表现 | 常见对策 |
|---|---|---|
| blank 占比过高 | 预测文本过短，漏字 | 调整初始化、温度缩放、延长训练 |
| 重复字符处理错误 | `ll`、`oo` 等识别差 | 检查 blank 插入与解码逻辑 |
| 长序列数值下溢 | 概率接近 0 | 使用 log-space 前向后向 |
| Greedy 结果偏差大 | 局部最优但整体错误 | 用 beam search，必要时加语言模型 |

第三类问题是**数值稳定性**。因为 CTC 要把很多路径概率相乘再相加，$T$ 稍大时数值就会非常小。工程里一般不直接算概率，而是算对数概率，并使用 log-sum-exp 技巧稳定求和。

第四类问题是**标签集设计**。ASR 中如果输出单位选字符，序列会更长，beam search 压力更大；如果选 BPE 或词片段，序列更短，但词表更大。OCR 中也类似，按字符输出最直观，但中文大词表会明显增加分类头负担。

第五类问题是**条件独立假设带来的上限**。CTC 不擅长处理强语言依赖。例如口语转写里，同音词选择往往依赖前后上下文；CTC 只靠帧级发射概率时，容易在语义层面吃亏。因此工程上常把 CTC 与外部语言模型或 attention 分支结合。

---

## 替代方案与适用边界

CTC 不适合所有序列任务，它适合的是“输入和输出大体单调对应，而且需要高吞吐、低延迟”的场景。

下面给出一个选型表。

| 方案 | 对齐假设 | 解码复杂度 | 延迟 | 适合场景 |
|---|---|---|---|---|
| CTC | 单调 | 中 | 低 | 流式 ASR、OCR |
| RNN-T | 单调，但显式建模历史输出 | 较高 | 低到中 | 在线语音识别 |
| Attention Decoder | 无需单调 | 高 | 高 | 机器翻译、离线高质量 ASR |

三者的差异可以这样理解。

CTC 把对齐问题边缘化，但不强建模输出历史；RNN-T 在保留流式能力的同时，把“已经输出了什么”纳入预测器，因此对语言上下文更敏感；attention decoder 则不要求单调，可以自由关注输入任意位置，适合翻译、摘要这类非严格对齐任务。

一个真实工程判断标准是：

- 如果你做的是**流式语音识别**，用户要求边说边出字，CTC 或 RNN-T 更合适。
- 如果你做的是**扫描文档 OCR 行识别**，字符顺序和图像顺序天然单调，CTC 很合适。
- 如果你做的是**语音翻译**，源语音和目标文本往往不是单调一一对应，attention/seq2seq 更合适。

现在很多系统采用混合方案，例如 **CTC + attention**。CTC 分支提供稳定的对齐监督，attention 分支补充语言建模能力。训练时联合优化，推理时可按场景选择一种或融合两种得分。这类设计常见于现代离线 ASR，因为它兼顾了收敛速度和最终精度。

所以，CTC 的适用边界可以压缩成一句话：**当任务满足单调映射，且你希望训练不依赖逐帧标注、推理延迟可控时，CTC 是非常强的基线；当任务需要更强上下文或非单调建模时，应优先考虑 RNN-T 或 attention 类模型。**

---

## 参考资料

- Emergent Mind, “Connectionist Temporal Classification (CTC)”  
  https://www.emergentmind.com/topics/connectionist-temporal-classification-ctc
- OCR Oldfish, “CTC Loss Function and Training Techniques”  
  https://ocr.oldfish.cn/en/article.aspx?id=84&slug=ctc-loss-training
- Nithish Duvvuru, “CTC Decoding”  
  https://nithish96.github.io/Computer%20Vision/Concepts/CTC%20Decoding/
- Arun Baby, “ASR Beam Search Implementation”  
  https://www.arunbaby.com/speech-tech/0023-asr-beam-search-implementation/
