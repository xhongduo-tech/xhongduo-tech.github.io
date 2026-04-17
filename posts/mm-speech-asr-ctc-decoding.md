## 核心结论

CTC 解码的任务很具体：把声学模型在每一帧给出的“字符概率分布”转换成最终文本。CTC 是一种“允许输入帧数和输出字数不对齐”的训练与解码框架，白话讲，就是模型不用先知道“第几毫秒对应哪个字”，也能学会从整段声音里恢复文字。

它之所以成立，靠的是两条规则：

1. 引入 `blank` 空白符，表示“这一帧不输出字符”。
2. 对路径做 `collapse`，也就是“先合并连续重复，再删除 blank”。

因此，路径 `A, blank, B` 会变成 `AB`，路径 `A, A, blank, B` 会变成 `AB`，而路径 `A, blank, A` 会变成 `AA`。`blank` 的作用不是占位美观，而是把“字符持续发音”和“字符重复输出”区分开。

玩具例子如下。假设 3 帧的最优路径是：

```text
t1: A
t2: blank
t3: B
```

则 collapse 过程是：

```text
原始路径: A   _   B
合并重复: A   _   B
删除blank: A     B
结果: AB
```

如果只做逐帧贪婪选择，系统常常能输出“像样的字串”；但真实语音识别通常要上 `beam search + 语言模型`。Beam search 是“同时保留多个候选前缀的搜索”，白话讲，就是不要太早认死理。语言模型 LM 是“衡量一句话是否像自然语言的模型”，白话讲，就是给“更像人话”的路径加分。

| 帧级路径 | collapse 后 | 说明 |
|---|---|---|
| `A, _, B` | `AB` | 空白把两个字符分开 |
| `A, A, _` | `A` | 连续重复会被合并 |
| `A, _, A` | `AA` | 中间有 blank，因此保留两个 A |
| `_, B, B, _` | `B` | 前后空白不影响结果 |

---

## 问题定义与边界

CTC 解码的输入是长度为 $T$ 的概率序列：
$$
\mathbf{P} = \{p_t(c)\}_{t=1}^T,\quad c \in \mathcal{V} \cup \{\epsilon\}
$$
其中 $\mathcal{V}$ 是字符表，$\epsilon$ 是 blank。输出是一个文本前缀或完整句子，例如“你好”或“speech recognition”。

这里最关键的边界是：CTC 不提供显式对齐。也就是说，模型知道“整段音频对应这句话”，但不知道“第 17 帧一定是这个字”。解码器必须自己在大量可能路径里找最优文本。

可以把每一帧理解成一个“概率骰子”：它会对 `A/B/C/.../blank` 给出不同概率。解码器不会只看当前帧，而是维护多个候选句子，逐帧扩展并剪枝。`beam` 就是这些候选前缀的集合。

参数边界直接影响搜索空间：

| 参数 | 作用 | 太小/太低的后果 | 太大/太高的后果 |
|---|---|---|---|
| `beam_size` | 保留多少候选前缀 | 后期可能翻盘的路径被过早丢掉 | 计算慢，延迟高 |
| `lm_weight=α` | 语言模型强度 | 接近纯声学贪婪，句子不自然 | 语法合理但偏离音频 |
| `word_score=β` | 单词插入偏置 | 容易漏词、句子过短 | 容易插入多余词 |
| `beam_threshold` | 剪枝阈值 | 激进剪枝导致删词 | 保留过多低质量路径 |
| `hotword_weight` | 热词增强 | 关键词难打出来 | 热词被过度偏置 |

一个常见误区是把 CTC 解码理解成“每帧选最大概率，再把 blank 去掉”。这只是 greedy decoding，适合演示，不适合大多数真实系统。原因很简单：某条路径在前两帧看起来很弱，第三帧开始可能突然变强；beam 太窄时，这条路径已经被删了。

---

## 核心机制与推导

CTC beam search 通常对每个前缀维护两种概率：

- $P_b(l, t)$：前缀 $l$ 在时刻 $t$ 以 blank 结尾的概率。
- $P_{nb}(l, t)$：前缀 $l$ 在时刻 $t$ 以非 blank 结尾的概率。

这里“前缀”是当前已经形成的文本片段，白话讲，就是候选句子的前半截。双轨拆分的原因是：CTC 对“重复字符”有特殊规则，必须区分“最后一帧是不是 blank”。

总概率为：
$$
P(l, t)=\operatorname{LogSumExp}(P_b(l,t), P_{nb}(l,t))
$$

当本帧输出 blank 时：
$$
P_b(l,t) = \operatorname{LogSumExp}(P_b(l,t-1), P_{nb}(l,t-1)) + \log p_t(\epsilon)
$$

当扩展字符 $c$ 且 $c \neq$ 前缀最后一个字符时：
$$
P_{nb}(l{+}c,t) = \operatorname{LogSumExp}(P_b(l,t-1), P_{nb}(l,t-1)) + \log p_t(c)
$$

当扩展字符 $c$ 且 $c =$ 前缀最后一个字符时，只能从 blank 轨跳过去：
$$
P_{nb}(l,t) \leftarrow P_b(l,t-1) + \log p_t(c)
$$

这条限制很重要。它保证了 `A, A` 不会被误认为 `AA`，除非中间插入 blank，如 `A, _, A`。

ASCII 流程图可以写成：

```text
前缀 "A"
  ├─ 输出 blank  ──> 仍是 "A"   ，更新 P_b
  ├─ 输出 A      ──> 仍是 "A"   ，但只能从 P_b 转移
  └─ 输出 B      ──> 扩成 "AB" ，更新 P_nb
```

加入语言模型后，常见浅融合打分是：
$$
\text{Score}(y)=\text{Score}_{AM}(y)+\alpha \cdot \text{Score}_{LM}(y)+\beta \cdot \text{WordCount}(y)
$$

其中：

- 声学模型 AM：根据音频给路径打分，白话讲，就是“听起来像不像”。
- 语言模型 LM：根据语言统计给句子打分，白话讲，就是“说出来顺不顺”。
- $\alpha$ 控制 LM 影响力。
- $\beta$ 控制系统偏好更长还是更短的分词结果。

玩具数值例子。假设 3 帧、词表只有 `A/B/_`：

| 时间步 | `P(A)` | `P(_)` | `P(B)` |
|---|---:|---:|---:|
| `t1` | 0.8 | 0.1 | 0.1 |
| `t2` | 0.1 | 0.8 | 0.1 |
| `t3` | 0.1 | 0.1 | 0.8 |

最强路径是 `A, _, B`，概率乘积约为 $0.8 \times 0.8 \times 0.8 = 0.512$，collapse 后得到 `AB`。

如果第三帧 `A` 和 `B` 接近，例如 `P(A)=0.38, P(B)=0.36, P(_)=0.26`，纯声学上 `A` 略强；但如果语言模型认为 `AB` 是合法词、`AA` 很罕见，适当增大 $\alpha$ 后，beam search 可能选择 `AB`。这就是“声学像一点”和“语言更通顺”之间的工程折中。

---

## 代码实现

真实工程里常见两类实现：

1. DeepSpeech 一类的 C++/原生解码器，外接 scorer 和 KenLM。
2. `pyctcdecode` 这类 Python 解码器，接口更轻，便于实验和调参。

实现主循环并不神秘，核心就是“扩展、合并、剪枝、collapse”。伪码如下：

```text
beams = {"": (P_b=0, P_nb=-inf)}

for each frame in logits:
    next_beams = {}
    for prefix, (pb, pnb) in beams:
        1. 用 blank 更新 prefix 自身
        2. 对每个高概率字符 c 扩展 prefix
        3. 若 c 与 prefix 最后字符相同，走重复规则
        4. 若到达词边界，加入 LM 分数
    5. 合并相同前缀
    6. 按总分排序，只保留 top-k beams

返回分数最高的 prefix
```

下面给一个可运行的极简 Python 版本。它没有接 KenLM，但完整体现了 `blank/重复/collapse` 的核心逻辑。

```python
import math

NEG_INF = -1e30

def logsumexp(a, b):
    if a < b:
        a, b = b, a
    if b <= NEG_INF / 2:
        return a
    return a + math.log1p(math.exp(b - a))

def ctc_prefix_beam_search(probs, vocab, beam_size=3, blank="_"):
    beams = {"": (0.0, NEG_INF)}  # prefix -> (P_b, P_nb), log space

    for frame in probs:
        next_beams = {}

        def add_beam(prefix, pb, pnb):
            old_pb, old_pnb = next_beams.get(prefix, (NEG_INF, NEG_INF))
            next_beams[prefix] = (logsumexp(old_pb, pb), logsumexp(old_pnb, pnb))

        for prefix, (pb, pnb) in beams.items():
            total = logsumexp(pb, pnb)

            # 1) 扩展 blank，前缀不变
            add_beam(prefix, total + math.log(frame[blank]), NEG_INF)

            # 2) 扩展非 blank
            for ch in vocab:
                if ch == blank:
                    continue
                p = math.log(frame[ch])
                last = prefix[-1] if prefix else None

                if ch == last:
                    # 重复字符：只能从 blank 轨转到同前缀
                    add_beam(prefix, NEG_INF, pb + p)
                    # 若中间有 blank，才能形成新前缀 prefix + ch
                    add_beam(prefix + ch, NEG_INF, pb + p)
                else:
                    add_beam(prefix + ch, NEG_INF, total + p)

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda kv: logsumexp(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_size]
        )

    best = max(beams.items(), key=lambda kv: logsumexp(kv[1][0], kv[1][1]))[0]
    return best

probs = [
    {"A": 0.8, "_": 0.1, "B": 0.1},
    {"A": 0.1, "_": 0.8, "B": 0.1},
    {"A": 0.1, "_": 0.1, "B": 0.8},
]

text = ctc_prefix_beam_search(probs, vocab=["A", "_", "B"], beam_size=3)
assert text == "AB"
print(text)
```

如果接入 `pyctcdecode`，工程接口通常更短：

```python
# 示例接口形态，不要求本地可运行
from pyctcdecode import build_ctcdecoder

labels = ["_", "a", "b", " ", "c"]
decoder = build_ctcdecoder(
    labels,
    kenlm_model_path="lm.bin",
    alpha=0.5,
    beta=1.0,
)
# text = decoder.decode(logits)
```

真实工程例子是离线转写服务：前端上传音频，后端声学模型输出 `[T, V]` logits，再把 logits 送进 beam decoder。若业务里高频出现品牌名、药名、地名，还会叠加 hotword boosting，让这些词在相同声学证据下更容易被选中。

---

## 工程权衡与常见坑

CTC 解码最常见的问题不是“不会跑”，而是“能跑但结果不稳”。

| 坑 | 典型症状 | 原因 | 规避方式 |
|---|---|---|---|
| `beam_size` 太小 | 杂音下漏词、删词 | 候选过早被剪掉 | 先在验证集扫 `beam_size`，看 WER 与延迟曲线 |
| `beam_size` 太大 | CPU 占用高、实时性差 | 搜索空间膨胀 | 配合 `cutoff_top_n` 与阈值剪枝 |
| `α` 太高 | 输出更像“正常句子”，但和音频不符 | LM 压过声学模型 | 用开发集单独调 α，不要拍脑袋 |
| `β` 太高 | 插入多余词或空格 | 长句偏置过强 | 联合调 α/β，不单独调 |
| 热词权重过大 | 品牌名乱入 | 热词把弱声学路径也拉上来 | 只对封闭域关键词加权 |
| 流式直接每帧出结果 | partial 文本来回跳 | beam 每帧重排导致 flicker | 加稳定窗口与提交阈值 |

流式识别尤其容易出现 hypothesis flicker，也就是“中间结果抖动”。白话讲，用户看到屏幕上的字一会儿变成这个，一会儿又改回去。原因不是系统坏了，而是后续音频到来后，早先前缀的相对分数被改写。

常见缓解办法：

1. 只输出“稳定前缀”，例如连续若干帧未变的部分才提交。
2. 结合 endpoint detection，在停顿处再固化结果。
3. 对增量 beam 做平滑，不让低置信度尾部频繁改写。

例如客服转写场景里，`beam_size=5` 在安静环境可能够用，但电话噪声一上来，经常把后续能修正的路径提前剪掉；再如 `lm_weight=2.0` 可能让“我要查账户余额”被拉向更常见的“我要查账户业务”，语法更像真话，但音频其实不是这么说的。

---

## 替代方案与适用边界

CTC 不是唯一方案。常见替代是 RNN-T 和 Attention/Encoder-Decoder。

| 方案 | 对齐方式 | 延迟特性 | LM 整合 | 优点 | 缺点 |
|---|---|---|---|---|---|
| CTC | 隐式对齐，靠 blank/collapse | 可做低延迟，也可离线 | 容易做浅融合 | 结构简单，解码器成熟 | 条件独立假设较强 |
| RNN-T | 显式建模声学步与输出步交互 | 更适合流式 | 相对更复杂 | 流式效果通常更强 | 训练和解码更复杂 |
| Attention | 直接学输入到输出映射 | 通常偏离线 | 常做内部 LM 化 | 全局上下文强 | 流式支持较弱，稳定性依赖设计 |

什么时候继续用 CTC：

- 设备资源紧张，需要成熟、简单、可控的解码器。
- 已有 KenLM、词典、热词体系，希望快速接入。
- 业务更重视工程稳定性和可解释调参，而不是追求最新架构。

什么时候考虑迁移：

- 强流式、低延迟场景，需要更稳定的逐 token 决策。
- 需要更强上下文依赖，CTC 的条件独立假设开始成为瓶颈。
- 需要更细粒度的时序输出和更一致的在线体验。

一个简单判断是：低资源设备上的命令词、短句转写、可外接词典的行业 ASR，CTC 仍然很实用；如果要做长对话流式字幕、边说边稳定输出、还要强上下文跟踪，RNN-T 往往更合适。

---

## 参考资料

1. [Sequence Modeling with CTC](https://distill.pub/2017/ctc/)
   用途：解释 CTC 的 blank、路径集合与 collapse 规则，适合建立直觉与核对基本公式。

2. [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
   用途：CTC 的原始论文，适合查算法定义、前向后向推导和理论边界。

3. [DeepSpeech CTC beam search decoder](https://deepspeech.readthedocs.io/en/latest/Decoder.html)
   用途：看 DeepSpeech 如何接外部 scorer、alphabet 和语言模型，适合理解工程约束。

4. [DeepSpeech CTC Decoder & Language Models](https://deepwiki.com/mozilla/DeepSpeech/5-ctc-decoder-and-language-models)
   用途：按源码结构解释 `DecoderState`、`Scorer`、`PathTrie`，适合理解真实实现细节。

5. [Torchaudio: ASR Inference with CTC Decoder](https://docs.pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html)
   用途：看 beam search、lexicon constraint、KenLM 接入方式，以及推理侧参数如何影响结果。

6. [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode)
   用途：看 Python 侧 CTC beam decoder 的接口、hotword、BPE 支持与流式状态管理。
