## 核心结论

端到端语音识别里的注意力机制，本质上是在每个解码步动态决定“当前该看哪些帧”。编码器（把连续语音帧压成高层表示的网络）负责产出帧序列 $a_{1:U}$，解码器（按顺序生成字符或子词的网络）在第 $t$ 步根据当前状态 $s_t$ 计算上下文 $c_t$，再输出符号 $y_t$。这一步同时完成两件事：一是对齐，决定当前字符对应哪些语音帧；二是生成，利用这些帧和历史输出推断下一个符号。

直观上，可以把 encoder 帧看成舞台上的灯光，attention 像导演：每当演员要说下一句台词，导演会挑出最该亮的几盏灯。灯打到哪里，就是对齐；演员说什么，就是生成。注意力把这两件事放到一个可学习模块里统一优化，这就是它比“先对齐、再识别”的传统流水线更自然的地方。

离线模型通常用全局 soft attention，也就是每次都能看完整段语音，效果强但延迟高。流式模型必须“边听边写”，因此只能看过去帧或极少量未来帧，典型做法是 MoChA、triggered attention、Emformer 这类受限注意力。2024 年关于 attention-based ASR 的综述把这条演进线总结得很清楚；在真实驾驶舱场景里，基于交叉模态注意力的音视频识别对驾驶员语音命令可达到 98.65% 准确率，说明注意力不仅是理论上优雅，也确实能在噪声和多模态条件下落地。

| 机制 | 所需上下文 | 延迟 | 对齐手段 |
| --- | --- | --- | --- |
| 离线 soft attention | 全部历史 + 全部未来 | 高 | 全局 softmax 权重 |
| MoChA | 过去帧 + 局部窗口 | 低 | 单调选点 + 局部 soft attention |
| Triggered attention | 过去帧 + 少量未来帧 | 低到中 | CTC 触发 + 局部 attention |
| Emformer | chunk 内局部上下文 + memory bank | 低到中 | 分块注意力 + 历史记忆 |

---

## 问题定义与边界

任务定义很直接：输入一串连续的声学帧，或者音频帧加视频唇动帧，输出字符、子词或词。难点不在“分类”，而在“对齐未知”。因为输入长度 $U$ 通常远大于输出长度 $T$，而且没有人工标注每个字符对应哪一帧。

形式化地说，模型在第 $t$ 个解码步需要从帧序列 $a_{1:U}$ 中构造上下文：
$$
c_t = f(s_t, a_{1:U})
$$
其中 $f$ 就是注意力模块。离线与流式的区别，本质上是 $f$ 的可见范围不同：

- 离线：$f$ 可以访问全部 $a_{1:U}$，所以更容易做全局重排和长距离依赖建模。
- 流式：$f$ 只能访问 $a_{1:u_t+\delta}$，其中 $\delta$ 是允许看的未来帧数，通常很小，甚至为 0。

这决定了边界。比如会议转写、字幕离线生成，允许完整读入整段语音，soft attention 通常更合适。车载语音助手、同传字幕、耳机实时控制，则必须在低延迟下输出，不能等一句话说完才开始识别。

一个真实边界场景是车载命令识别：发动机、路噪、空调和多人说话会把信噪比压到 10 dB 以下，这时只靠音频很容易错过辅音或短命令词。若系统还要求边说边响应，就不能依赖全局离线 attention，而要结合视觉唇动，用仅看过去帧或极少未来帧的流式机制。

| 方案 | 上下文规模 | 流式支持 | 对未来依赖 |
| --- | --- | --- | --- |
| 离线 soft attention | 全序列 | 否 | 强依赖 |
| MoChA | 单调终点前的固定窗口 | 是 | 可做到零未来依赖 |
| Triggered attention | CTC 触发附近局部窗口 | 是 | 通常依赖少量未来帧 |
| Emformer | chunk + memory | 是 | 依赖受控的 chunk/look-ahead |

---

## 核心机制与推导

标准 soft attention 的机制可以分成三步。

第一步，打分。打分函数 score 用来衡量“当前解码状态 $s_t$ 和第 $u$ 帧表示 $a_u$ 有多匹配”：
$$
e_{t,u} = \text{score}(s_t, a_u)
$$

第二步，归一化。把所有帧的分数做 softmax，得到注意力权重：
$$
\alpha_{t,u} = \frac{\exp(e_{t,u})}{\sum_{j=1}^{U}\exp(e_{t,j})}
$$

第三步，加权求和。上下文向量就是所有帧的加权平均：
$$
c_t = \sum_{u=1}^{U}\alpha_{t,u}a_u
$$

这里的 softmax 可以理解成“把关注度分成一组非负权重，且总和为 1”。它的好处是可微，训练稳定；问题是每一步都要看全序列，计算是 $O(TU)$，而且无法天然流式。

MoChA 解决的是“既要顺序性，又不想失去 soft attention 的表达力”。它先做单调选择，再做局部 soft attention。单调选择的概率写成：
$$
p_{t,u} = \sigma(\text{score}(s_t, a_u))
$$
其中 $\sigma$ 是 sigmoid，可把任意实数压到 0 到 1。白话说，$p_{t,u}$ 表示“解码器在第 $u$ 帧停下来的概率”。

测试时，模型从上一个停点继续往右扫，遇到首个满足触发条件的位置 $u^\*$ 就停，然后只在局部窗口 $[u^\*-w+1, u^\*]$ 内做 soft attention：
$$
\beta_{t,u'} = \frac{\exp(\text{score}(s_t, a_{u'}))}{\sum_{j=u^\*-w+1}^{u^\*}\exp(\text{score}(s_t, a_j))}
$$
$$
c_t = \sum_{u'=u^\*-w+1}^{u^\*}\beta_{t,u'}a_{u'}
$$

这就是“先选位，再在附近细看”。它保留了顺序约束，所以适合语音这种基本单调对齐的任务。

玩具例子可以直接算。假设窗口宽度 $w=3$，单调模块已经把第 5 帧选为当前终点，那么局部窗口就是第 3、4、5 帧。若对应 logits 是 $[2.0, 0.0, -1.2]$，softmax 后约为：
$$
[0.84, 0.11, 0.03]
$$
上下文就是这三帧的加权和。若把 logits 稍微改成更平缓的一组，例如 $[1.2, 0.2, -0.6]$，权重就会分散一些，模型对边界的决策也更稳。这里最关键的不是某个具体数字，而是注意力把“停在哪”和“停下后看多宽”拆成了两层决策。

Triggered attention 进一步把“停在哪”交给 CTC。CTC（连接时序分类，一种显式偏好单调对齐的训练目标）会在某些帧上给出明显的 token spike，解码器只在这些触发点附近启动 attention。这样做的收益是对齐更稳、计算更省，代价是触发质量高度依赖 CTC 分支，通常还要少量未来帧保证 spike 不漂移。

Emformer 则是另一条路线。它不是把解码器注意力做成单调扫描，而是在编码器侧用 chunk attention 加 memory bank。memory bank 可以理解成“压缩后的长历史摘要”，模型不必重新看完整过去帧，但仍保留长距离上下文。

---

## 代码实现

一个最小可运行实现可以把注意力模块拆成三部分：`encoder_outputs`、单调触发、局部 soft attention。下面的代码演示了 MoChA 风格的一步解码，不依赖深度学习框架，也能看清逻辑。

```python
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def weighted_sum(weights, vectors):
    dim = len(vectors[0])
    out = [0.0] * dim
    for w, v in zip(weights, vectors):
        for i in range(dim):
            out[i] += w * v[i]
    return out

def mocha_step(decoder_state, encoder_outputs, start_idx, window, threshold=0.5):
    """
    decoder_state: 当前解码器状态
    encoder_outputs: 编码器帧表示
    start_idx: 从哪里开始向右扫描
    window: 局部注意力窗口宽度
    """
    trigger = None

    # 1. 单调扫描，决定停点
    for u in range(start_idx, len(encoder_outputs)):
        score = dot(decoder_state, encoder_outputs[u])
        p_choose = sigmoid(score)
        if p_choose >= threshold:
            trigger = u
            break

    if trigger is None:
        return None, None, start_idx

    # 2. 在停点前的局部窗口上做 soft attention
    left = max(0, trigger - window + 1)
    chunk = encoder_outputs[left:trigger + 1]
    logits = [dot(decoder_state, frame) for frame in chunk]
    weights = softmax(logits)
    context = weighted_sum(weights, chunk)

    return trigger, weights, context

# 玩具输入：5 帧，每帧 2 维
frames = [
    [0.1, 0.0],
    [0.2, 0.1],
    [1.0, 0.0],
    [0.0, 0.3],
    [0.2, 0.9],
]
state = [1.0, 0.2]

trigger, weights, context = mocha_step(state, frames, start_idx=0, window=3, threshold=0.7)

assert trigger is not None
assert abs(sum(weights) - 1.0) < 1e-9
assert len(context) == 2
assert trigger >= 0
```

如果换成真实系统，组件会更完整：

- 编码器：把 Mel 频谱、Conformer 表示，或者音频特征加视觉唇动特征编码成帧序列。
- 注意力模块：实现 soft attention、MoChA、triggered attention 或 head-synchronous 变体。
- 解码器：根据上下文 $c_t$ 和历史 token 生成下一个字符或子词。

流式实现里有两个额外细节不能省。

第一，块状态维护。系统通常按 chunk 处理输入，每个 chunk 既要带本地上下文，也要保存历史摘要。

第二，多头同步。多头注意力里的“头”可以理解成多组并行的关注模式。若每个头各自决定停点，流式场景会出现边界抖动，工程上常改成 head-synchronous，让同层头共享边界。

| 多头策略 | 行为 | 优点 | 风险 |
| --- | --- | --- | --- |
| 异步多头 | 每个 head 自己选边界 | 表达力更强 | 流式时边界不一致，输出抖动 |
| 同步多头 | 同层 head 共享边界 | 对齐稳定，易控延迟 | 自由度下降，可能略损精度 |

---

## 工程权衡与常见坑

注意力机制在论文里看起来统一优雅，但到了工程里，核心问题不是“能不能跑”，而是“延迟、稳健性、可控性怎么一起满足”。

最典型的权衡是 chunk 大小。chunk 太小，模型拿不到足够上下文，句内长依赖和同音区分会变差；chunk 太大，延迟直接上升，用户会感知到系统“反应慢”。这不是简单调超参，而是业务目标决定的。如果是车载唤醒词，小 chunk 更重要；如果是长句听写，稍大 chunk 往往值得。

第二个坑是 triggered attention 对未来帧的依赖。CTC 触发点看似天然流式，但真实音频里 spike 往往不够尖锐。未来帧太少，触发早了会漏字，触发晚了会拖延输出。很多系统会允许 2 到 4 帧 look-ahead，用几十毫秒延迟换对齐稳定性。

第三个坑是多模态融合。把音频和视频都接进来，不代表一定更好。视觉帧率、音频帧率、相机遮挡、头部转动都会让跨模态对齐变差。真实车载系统里，视觉更像“补偿通道”，在噪声大或爆破音被掩盖时帮助修正，而不是永远主导。

第四个坑是 Emformer 的 memory 压缩。memory bank 提高了效率，但它不是原始历史的无损副本。长尾依赖、说话风格突变、句首关键词等信息可能在压缩时被削弱，所以 memory 大小、更新频率和摘要方式都需要专门验证。

一个真实工程例子是车载多模态命令识别。系统需要在强噪声下保持低延迟响应，常见做法是：音频走 chunked encoder，视觉唇动走轻量视频分支，两者通过交叉模态注意力对齐；解码侧使用受限 attention 或 CTC 触发机制；训练时再加辅助 loss 稳住对齐。这样做的理由很直接：小 chunk 控制延迟，视觉补偿音频缺口，辅助损失约束边界漂移。

| 问题 | 影响 | 规避策略 |
| --- | --- | --- |
| chunk 过小 | 丢全局语义，错分同音词 | 增大 chunk，或加 memory/context cache |
| chunk 过大 | 延迟升高，实时体验差 | 缩短 chunk，仅保留必要 look-ahead |
| trigger 未来信息不足 | 提前或滞后触发，漏字/重复 | 保留少量未来帧，联合训练更强 CTC 分支 |
| 多头不同步 | 边界抖动，输出不稳定 | 采用 head-synchronous 机制 |
| memory 压缩过强 | 长尾信息丢失 | 增大 memory 容量，分层保留关键信息 |

辅助 loss 的思路也值得一提。比如在训练时同时最小化主解码损失和 CTC 损失：
$$
\mathcal{L} = \lambda \mathcal{L}_{CTC} + (1-\lambda)\mathcal{L}_{Attn}
$$
这样做不是为了让 CTC 取代注意力，而是让模型在早期就学会更稳定的单调边界。

---

## 替代方案与适用边界

注意力不是唯一答案。CTC、RNN-T、Transformer-CTC 都是强有力替代方案，只是它们对“对齐”和“生成”的拆分方式不同。

CTC 的特点是显式单调、训练和推理都相对直接，特别适合低延迟和大规模工程部署。但它的条件独立假设更强，语言建模能力通常不如带解码器的注意力模型。

RNN-T 把对齐和预测网络分开，是工业流式 ASR 的主流之一。它的在线性和稳健性通常优于纯 attention，但训练复杂、调参成本高。

Triggered attention 是折中方案：用 CTC 负责“什么时候可以解码”，用注意力负责“触发附近怎么细看”。当延迟要求很低，但又不想完全放弃 attention 的表达力时，它很实用。

Emformer 则更适合“可接受受控延迟，但又需要更长上下文”的场景。比如长指令、多轮语音交互、连续说话人切换，它比极窄窗口的单调 attention 更稳。

| 方案 | 是否有 explicit alignment | 延迟 | 多模态支持 |
| --- | --- | --- | --- |
| CTC | 有，路径显式单调 | 低 | 中等，常作编码后融合 |
| RNN-T | 有，联合网络隐式建模 | 低 | 中等到强 |
| 纯离线 attention | 无显式硬对齐 | 高 | 强 |
| Triggered attention | 有，依赖 CTC 触发 | 低到中 | 强 |
| Emformer + chunk attention | 有受控边界 | 低到中 | 强 |

可以把选择规则简化成两句：

- 如果延迟极低、系统要稳定、能接受一点表达力损失，优先考虑 CTC、RNN-T 或 triggered attention。
- 如果允许小到中等延迟，希望保留更强上下文建模，Emformer 或受限 attention 更合适。

Triggered attention 的流程很短：编码器先跑出帧表示，CTC 分支检测到 token spike 后发出 hit，解码器仅在 hit 附近的局部窗口执行 attention。Emformer 的流程也可以一句话概括：每次处理当前 chunk，同时读取历史 memory bank，把长历史压缩成摘要并持续滚动更新。

---

## 参考资料

- [Thank you for attention: A survey on attention-based artificial neural networks for automatic speech recognition](https://www.sciencedirect.com/science/article/pii/S2667305324000802)  
  2024 年综述，适合先建立离线 attention、流式 attention、Transformer 流式化的整体地图。

- [Monotonic Chunkwise Attention](https://openreview.net/forum?id=Hko85plCW)  
  ICLR 2018 经典论文，提出“单调选点 + 局部 soft attention”的 MoChA，是在线 attention 的代表性起点。

- [Emformer: Efficient Memory Transformer Based Acoustic Model For Low Latency Streaming Speech Recognition](https://resourcecenter.ieee.org/conferences/icassp-2021/spsicassp21vid1147)  
  ICASSP 2021，重点看 memory bank 和低延迟 chunk 处理，理解 Transformer 如何做流式。

- [Audio-Visual Speech Recognition In-The-Wild: Multi-Angle Vehicle Cabin Corpus and Attention-Based Method](https://openreview.net/forum?id=FZ6uTgpBfU)  
  ICASSP 2024，真实驾驶舱多模态识别案例，能看到交叉模态注意力在噪声环境里的实际收益。

- [Joint CTC/attention decoding for end-to-end speech recognition](https://aclanthology.org/P17-1048/)  
  ACL 2017，理解 CTC 与 attention 如何联合训练、联合解码，是读 triggered attention 前的好铺垫。

建议阅读顺序是：综述 → MoChA → Joint CTC/Attention → Emformer → 车载 AVSR。先建立“为什么要限制可见上下文”的问题意识，再看各方案如何在对齐精度、延迟和鲁棒性之间做折中。
