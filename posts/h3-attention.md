## 核心结论

H3（Hungry Hungry Hippos）的核心价值，不是“把 Transformer 小修小补”，而是给出了一条替代路径：把 attention 里最关键的两件事拆开处理。第一件事是**存储历史**，也就是把过去 token 的信息稳定保留下来；第二件事是**按条件读取**，也就是只有当前 token 满足匹配条件时，才把过去存下来的内容取出来。H3 用两个不同角色的 SSM（State Space Model，状态空间模型，白话说就是“用连续状态压缩并更新序列历史的模型”）加上乘性门控，分别承担这两件事。

它的直观图景可以理解成两个记忆槽。第一个槽由 shift SSM 负责，作用像“把刚看到的东西往后推一格”；第二个槽由 diagonal SSM 负责，作用像“把满足条件的信息长期挂在状态里”。门控则像开关，决定什么时候写，什么时候读。于是，H3 不再像标准 attention 那样显式做所有 token 两两比较，而是把“记住”和“比较”都压进状态更新里完成。

这件事的重要性在于：它证明了纯靠 SSM 并不够，因为纯 SSM 擅长压缩历史，但不擅长做精确匹配；而 SSM 加门控之后，就开始具备类似 attention 的 recall 能力。后来的 Hyena、Mamba 等路线，虽然结构不完全相同，但都延续了这个方向：不要只追求“更长记忆”，还要解决“何时触发记忆”。

从公式上看，H3 的核心可以压缩为：

$$
m_t = S_{shift}(q_{:t}), \quad
h_t = S_{diag}(m_t \odot k_t), \quad
o_t = W_o(h_t \odot v_t)
$$

其中 $\odot$ 表示逐元素乘法，白话说就是“只有对应维度都亮起时，这个通道才通过”。

---

## 问题定义与边界

H3 试图解决的问题很具体：在语言建模里，如果不想保留完整 attention 的 $O(n^2)$ 比较开销，能不能仍然保留长程依赖能力，也就是“前面出现过的重要信息，后面还能被准确取回”。

这里要先区分两类能力。

第一类是**memory**，也就是记忆能力。模型能不能把过去的信息存下来。很多 SSM 在这件事上并不差。

第二类是**comparison / recall**，也就是比较与检索能力。模型能不能在当前 token 到来时，判断它是否与过去某段信息匹配，并只把对应内容取出。这正是 attention 的强项。

纯 SSM 的问题在于，它常常只能“记住一个混合后的历史”，却不知道“当前到底该取哪一段”。H3 的贡献，就是给 SSM 补上这个条件检索机制。

一个新手能理解的说法是：普通 SSM 像一个一直在记笔记的人，但没有目录；H3 则给笔记本加了索引规则，只有 key 对上时，value 才会被读出来。

下面这张表可以把差异看得更清楚：

| 结构 | 能否稳定存历史 | 能否做 token 比较 | 能否做精确 recall | 训练/实现复杂度 |
|---|---|---|---|---|
| 普通 SSM（如 S4D） | 强 | 弱 | 弱 | 中 |
| H3（shift + diagonal + gate） | 强 | 中到强 | 中到强 | 中到高 |
| 标准 Attention | 强 | 强 | 强 | 高，长序列成本更高 |

H3 的边界也必须说清楚。它不是“attention 完全免费替代品”。如果任务非常依赖细粒度 token-to-token 对齐，例如复杂代码编辑、多跳精确引用、长表格逐格对比，纯 H3 仍可能不如保留若干 attention 层的混合结构。换句话说，H3 擅长的是“用状态压缩实现大部分长依赖”，但最硬的逐 token 比较场景，attention 依然更稳。

---

## 核心机制与推导

H3 通常先把输入 $x_t$ 线性投影成三路：$q_t, k_t, v_t$。这三个符号沿用了 attention 里的命名，但它们在 H3 里不是去做显式点积，而是进入两个不同的 SSM。

### 1. shift SSM：把局部信息推进状态

shift SSM 可以理解成一个“下移一格”的状态更新器。它的作用不是长期记忆，而是把前一个位置的信息传给后一个位置，因此特别适合表达“上一 token 是谁”。

如果把状态记为 $s_t$，最理想化的 shift 矩阵长这样：

$$
A_{shift} =
\begin{bmatrix}
0 & 0 & 0 \\
1 & 0 & 0 \\
0 & 1 & 0
\end{bmatrix}
$$

它的含义是：每走一步，旧状态整体下移。于是，$m_t = S_{shift}(q_{:t})$ 可以近似理解为“当前时刻持有了刚才看到的 query 痕迹”。

### 2. diagonal SSM：把触发后的内容长期保留

diagonal SSM 的状态转移矩阵是对角形式，白话说就是“每个状态通道独立演化，不互相搅在一起”。这种结构计算更高效，也更容易做卷积化实现。论文里采用 HiPPO 风格初始化，HiPPO 可以理解成“一种让连续状态尽量保留历史函数信息的初始化方法”。

它接收的不是原始输入，而是门控后的输入：

$$
h_t = S_{diag}(m_t \odot k_t)
$$

这里的关键是 $m_t \odot k_t$。如果当前 $k_t$ 和之前 shift 传来的模式不匹配，对应维度乘出来就接近 0，信息不会被写进去；如果匹配，就会触发写入。于是 diagonal SSM 只记住“被匹配确认过”的内容。

### 3. 输出门：只在需要时放出 value

最终输出是：

$$
o_t = W_o(h_t \odot v_t)
$$

这一步等于再做一次乘性筛选。$h_t$ 是状态里已经保留下来的历史痕迹，$v_t$ 是当前内容或当前读出条件。两者逐维相乘后，只有匹配通道会通过。

### 玩具例子

设 key “a” 编码成 $[1,0]$，另一个 key “b” 编码成 $[0,1]$。设 value “2” 编码成 $[0,1]$。

现在输入序列的逻辑是：“看到 a 后，把后面的 2 记住；之后再看到 a，就把 2 取出来。”

1. 第一步看到 “a”，shift SSM 让 $m_t \approx [1,0]$。
2. 下一步看到 value 相关内容时，若当前 key 仍对应 “a”，则 $m_t \odot k_t = [1,0]$，写入被触发。
3. diagonal SSM 把对应的 value 痕迹维持在状态里。
4. 后面再次遇到 query “a” 时，匹配维度再次打开，输出层从状态里读出“2”的表示。

如果换成 query “b”，则乘法门控无法打开，“2”不会被读出来。这就是 H3 里“只有 key 对上时才写、才读”的基本逻辑。

### 真实工程例子

在真实语言模型里，情况不是“字母配数字”这么简单，而是“前文某个实体、变量、主题词，在后文再次出现时，需要恢复对应上下文”。例如一篇技术文章前面定义了“KV cache”，后文再次出现“cache”时，模型需要恢复“这里讨论的是推理阶段缓存，而不是 CPU cache”。H3 试图把这种检索，用状态和门控的组合来近似，而不显式保留所有 token 的注意力图。

---

## 代码实现

下面给一个极简的可运行玩具实现。它不是论文里的完整 H3，也没有 FlashConv，只是把“shift 记上一步、gate 控制写入、state 持续保留”的机制用最少代码表达出来。

```python
import numpy as np

def h3_toy(keys, values, decay=0.9):
    """
    keys:   每步的 key 向量，shape [T, D]
    values: 每步的 value 向量，shape [T, D]
    返回每一步的输出
    """
    T, D = keys.shape
    shift_state = np.zeros(D)   # 对应 shift SSM 的最简状态
    diag_state = np.zeros(D)    # 对应 diagonal SSM 的最简状态
    outputs = []

    for t in range(T):
        q_t = keys[t]           # 玩具例子里直接复用 key 当 query
        k_t = keys[t]
        v_t = values[t]

        m_t = shift_state.copy()          # 近似 m_t = S_shift(q_:t)
        write_gate = m_t * k_t            # 匹配才写入
        diag_state = decay * diag_state + write_gate * v_t  # 近似 S_diag

        out_t = diag_state * v_t          # 匹配才读出
        outputs.append(out_t.copy())

        shift_state = q_t                 # 把当前 token 推给下一步

    return np.array(outputs)

# 玩具序列：
# t=0: key=a, value=0
# t=1: key=a, value=2  -> 应触发写入
# t=2: key=b, value=3  -> 不应读出 a 的内容
# t=3: key=a, value=2  -> 应再次读出与 a 相关的内容

a = np.array([1.0, 0.0])
b = np.array([0.0, 1.0])
zero = np.array([0.0, 0.0])
two = np.array([0.0, 2.0])
three = np.array([3.0, 0.0])

keys = np.stack([a, a, b, a], axis=0)
values = np.stack([zero, two, three, two], axis=0)

outs = h3_toy(keys, values, decay=1.0)

assert outs.shape == (4, 2)
assert outs[1, 1] > 0          # 第 2 步写入后，相关维度被激活
assert outs[2, 1] == 0         # key 不匹配，不读出 a 对应内容
assert outs[3, 1] > 0          # 再次遇到 a，可恢复相关内容

print(outs)
```

真实模型里的顺序通常是：

```python
q, k, v = proj(x)
m = shift_ssm(q)
h = diag_ssm(m * k)
y = out_proj(h * v)
```

工程上常见的实现细节是：先做输入投影，再经过非线性层如 `gelu`，然后进入 SSM 内核，最后做输出投影。真正决定速度的，不是这几行伪代码本身，而是 `shift_ssm` 和 `diag_ssm` 背后的并行扫描、卷积核生成、状态传递优化。

FlashConv 可以理解成“把 SSM 卷积和状态更新映射到更适合 GPU 的高效内核”。没有这层优化，H3 在纸面上省掉 attention，不代表在硬件上就一定更快。

---

## 工程权衡与常见坑

H3 最容易被误解的地方，是很多人以为“SSM 有长记忆，所以自然能替代 attention”。这不对。长记忆不等于可检索记忆。

下面是常见坑：

| 常见问题 | 现象 | 根因 | 规避方式 |
|---|---|---|---|
| 无门控，只有普通 SSM | 能记住模糊历史，但 recall 失败 | 没有条件写入/读取 | 加 shift + diagonal + 乘性门 |
| 无 FlashConv/高效内核 | 长序列下训练和推理并不快 | 理论复杂度没转化为硬件效率 | 使用卷积化和状态传递优化 |
| 全部层都换成 SSM | 一些 hard case 精确匹配退化 | 最后几层缺少显式 token 比较 | 保留少量 attention 层做混合 |
| 状态衰减设置不当 | 远距离信息过早消失或残留过多 | 状态更新超参数不合适 | 调整初始化、衰减和归一化 |

一个典型失败例子是：如果直接用 S4D 去做“看到 key 后记住对应 value，稍后再按 key 取回”的合成任务，模型往往只能学到模糊相关性，分不清到底该取哪个 value。原因不是它不会记，而是它不会“按条件记、按条件取”。

H3 的补救思路正好相反：先承认纯状态模型不擅长显式比较，再通过 shift SSM 保留局部配对线索，通过 diagonal SSM 保留长期痕迹，最后用乘法门控实现匹配。这是结构层面的修正，不是单纯加大参数量能解决的。

真实工程中还要注意一点：如果任务分布里大量存在“全局精确对齐”，例如长代码补全里变量名跨文件多次回指，那么只靠 H3 可能仍不稳。此时混合少量 attention 层，往往比盲目堆更多 SSM 更有效。

---

## 替代方案与适用边界

从今天回看，H3 更像一个关键中间站。它证明了 SSM 不是只能做平滑记忆，也可以在一定程度上承担 attention 的检索职责。但它并不是所有任务的最终答案。

| 方案 | 长序列效率 | 精细 token 比较 | recall 稳定性 | 适用场景 |
|---|---|---|---|---|
| Transformer | 一般到较弱 | 强 | 强 | 通用建模、精确对齐任务 |
| H3 | 强 | 中 | 中到强 | 长依赖为主、希望降低 attention 依赖 |
| 混合 SSM + Attention | 较强 | 强 | 强 | 希望兼顾速度与 hard case 表现 |

如果硬件预算足够，而且任务里有大量精细比较需求，Transformer 仍然是最稳妥的通用方案。它贵，但能力边界清晰。

如果预算有限，序列很长，且主要需求是“把远处的重要信息带过来”，H3 这种路线就有意义。它把大量历史处理压进状态，不必对所有 token 两两比较。

如果你既想要长序列效率，又不想在 hardest case 上退步，混合方案通常更现实。例如前面大部分层使用 H3 或其他 SSM，最后几层保留 attention，专门处理高精度比较和最终读出。这也是很多后续工作采用的折中路径。

一句话概括边界：H3 适合“长记忆优先、精确对齐次之”的任务；当任务要求严格逐 token 检索时，attention 仍然更强。

---

## 参考资料

- Tri Dao 等，《Hungry Hungry Hippos: Towards Language Modeling with State Space Models》  
  论文原文，理解 H3 机制、实验设置和混合模型结果的权威来源。  
  https://ar5iv.org/pdf/2212.14052

- Hazy Research 博客，《H3: Language Modeling with State Space Models and (Almost) No Attention》  
  图解比论文更直接，适合先建立“shift 负责什么、diagonal 负责什么”的直觉。  
  https://hazyresearch.stanford.edu/blog/2023-01-20-h3

- Together.ai 博客，《Hungry Hungry Hippos: Towards language modeling with state space models》  
  对工程意义和背景脉络有更通俗的说明，适合补充理解。  
  https://www.together.ai/blog/hungry-hungry-hippos-towards-language-modeling-with-state-space-models
