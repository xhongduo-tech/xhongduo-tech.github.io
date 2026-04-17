## 核心结论

实时语音识别的流式处理，目标不是“尽快把整句识别完”，而是“在不等用户说完的前提下，持续做出足够可靠的局部决策”。这里的“流式”可以先理解为：音频一边到达，模型一边输出，而不是等全部音频结束后再统一解码。

Speech ReaLLM 的关键价值，在于把“仅解码器”LLM 接进了 RNN-T 这类天然适合流式处理的框架里，让系统在每收到一个音频 token 或一个小时间块后，都能判断当前应不应该输出文本。它不是简单把大模型塞进语音系统，而是把决策拆成两件事：现在继续等，还是现在输出字词。这个拆分解决了大模型做流式识别时最难的点，即“何时说”的控制问题。

对工程系统来说，实时感通常受三条线约束：

| 指标 | 目标值 | 含义 |
|---|---:|---|
| 首字延迟 | $<200\text{ms}$ | 用户开始说话后，系统第一次给出文字的时间 |
| 词级延迟 | $<100\text{ms}$ | 相邻词或子词更新的平均等待时间 |
| 右侧上下文 | $\le 320\text{ms}$ | 为了判断当前词，最多允许偷看未来多长时间 |

这三个指标分别对应三种用户感知。首字延迟决定“系统是不是醒着”；词级延迟决定“字幕是不是跟得上语速”；右侧上下文决定“模型为了更准，到底允许自己等多久”。如果只追求低 WER 而忽略这三项，系统可能在离线评测里很好，但真实交互里会显得迟钝。

新手可以把它理解成一个固定循环：系统每约 80 到 100ms 听一小段，然后做一次选择，是“再等等”，还是“现在说一个词”。Speech ReaLLM 的进步在于，这个循环不再完全由小型语音模型控制，而是交给有更强语言建模能力的 LLM；但为了防止 LLM 说太早或迟到，先让一个较弱的语言模型把节奏学稳，再让更强的 LLM 接手内容建模。

另一个重要结论是：不能直接拿强 LLM 从零训练流式路径。现实做法是“弱 LM -> 强 LLM”。先用小语言模型学会对齐与节奏，再替换成更强的 LLM，并用 MWER 微调。MWER 可以先理解为“按整句错误率给奖励和惩罚”的训练方法，它关注最终识别质量，而不只看单步概率。

如果把整篇文章压缩成一句话，就是：流式 ASR 的核心不是“会不会识别”，而是“什么时候该输出，什么时候必须继续等”。Speech ReaLLM 和 Transducer-Llama 的价值，都集中在这个决策接口上。

---

## 问题定义与边界

问题定义可以写得很明确：给定持续到来的语音流 $x_{1:t}$，系统要在时刻 $t$ 输出当前最合理的前缀文本 $y_{1:u}$，并且这个输出必须满足低延迟、可持续、可在线更新。

更形式化地说，系统在每个时刻都要解决一个前缀最优问题：

$$
\hat{y}_{1:u}(t)=\arg\max_{y_{1:u}} P(y_{1:u}\mid x_{1:t})
$$

这里的关键不是“最终句子是什么”，而是“在只看到前缀音频 $x_{1:t}$ 时，当前最安全的文本前缀是什么”。这和离线识别不同。离线识别优化的是：

$$
\hat{y}=\arg\max_{y} P(y\mid x_{1:T})
$$

其中 $T$ 表示整段音频已经结束。流式识别没有这个条件，因此它天然更难。

这里有三个边界必须先说清楚。

第一，系统面对的是无端点流。所谓“无端点”，就是没有可靠的“用户已经说完”信号，比如会议字幕、同声传译、长时间语音输入。这和按住按钮录音、说完再松手不是一回事。无端点场景要求模型长期稳定工作，不能靠一句话结束后再回头修正全局。Speech ReaLLM 论文强调的一个点正是：它面向的是 continuous audio，而不是先切句、再识别的伪流式流程。

第二，实时系统不能无限依赖未来上下文。未来上下文可以先理解为“模型为了判断当前词，额外偷看的后面音频”。允许一点点偷看能显著提升准确率，但偷看太远，交互感就没了。因此常见约束是：

$$
\Delta t_{\text{lookahead}} \le 320\text{ms}
$$

这个约束不是拍脑袋定的。因为当系统每 80ms 或 100ms 处理一帧块时，320ms 大致等于额外等待 3 到 4 个 chunk。再往后看，用户通常已经能明显感知到延迟。

第三，流式识别优化的是“边际正确”。它不追求一开始就给出最终最优整句，而是追求每个时刻给出当前信息下最稳妥的输出。非流式模型往往更准，因为它能看到完整句子；但它的等待时间通常不适合实时助手、会议字幕、语音交互。

玩具例子最容易说明这个边界。用户说“今天北京天气怎么样”。

| 时刻 | 已到达音频 | 允许的稳妥输出 | 不理想输出 |
|---|---|---|---|
| $t_1$ | “今天北...” | “今天” 或空 | “今天被” |
| $t_2$ | “今天北京天...” | “今天北京天...” 的稳定前缀 | 直接猜完整句 |
| $t_3$ | 全句结束 | “今天北京天气怎么样” | 仍然长时间空白 |

这个例子说明两件事。过早输出会导致前缀错误，后续难修；过晚输出则直接破坏交互感。流式系统的难点，正是在这两种失败之间持续找平衡。

真实工程例子更典型。手机语音助手通常每 10ms 取帧、每若干帧组成一次特征块，再每 80 到 100ms 触发一次在线解码。log-Mel 可以先理解为把原始波形变成更稳定的频谱表示。系统会持续刷新屏幕上的转写结果，同时把最新文本喂给下游 NLU、翻译或命令执行模块。如果首字延迟超过 300ms，用户会明显感到“机器没跟上”；如果系统频繁改写前文，用户又会觉得结果不稳定。

下面这个时间预算是流式系统的核心约束：

$$
\text{Latency}_{\text{first token}}
=
\text{feature window}
+
\text{encoder compute}
+
\text{decision wait}
+
\text{decode/render}
< 200\text{ms}
$$

可以再拆得更细一点：

| 组成项 | 典型范围 | 说明 |
|---|---:|---|
| 特征窗口 | 25-40ms | 需要先积累最小声学上下文 |
| chunk 聚合 | 40-80ms | 为了稳定判断，通常不逐采样点决策 |
| 编码器计算 | 10-40ms | 取决于模型大小和设备 |
| blank/token 决策 | 5-20ms | 包括 predictor、joiner 或 factorized head |
| 渲染与提交 | 5-20ms | UI 更新、下游消费、网络传输 |

因此问题边界可以概括为：系统必须在严格延迟预算内，长期、单调、可回放地输出稳定前缀，而不能依赖“句子结束后统一修正”这类离线机制。

---

## 核心机制与推导

RNN-T 或 Transducer 系列模型的优势，在于它天然支持“走一步，决定一步”。Speech ReaLLM 和 Transducer-Llama 的关键机制，是把每一步联合决策拆成两部分：blank 路径和 token 路径。

这里的 `blank` 可以理解为“当前还不输出文本，只推进时间”；`token` 则表示“现在输出一个字符、子词或词片段”。

在标准 Transducer 里，模型在声学时间轴和文本标签轴之间交替移动。沿时间轴前进但不发字，就是 blank；沿标签轴前进并发字，就是 token。Factorized Transducer 进一步把这两个动作显式分解，使“是否发字”和“发什么字”分别建模。

形式上，可以写成：

$$
P(k\mid x_{1:t}, y_{1:u-1})
=
\begin{cases}
P_B(t,u), & k=\varnothing \\
(1-P_B(t,u))\cdot P_y(k\mid t,u), & k\in\mathcal{V}
\end{cases}
$$

其中：

- $\varnothing$ 表示 blank；
- $\mathcal{V}$ 表示普通 token 词表；
- $P_B(t,u)$ 表示在状态 $(t,u)$ 下“继续等”的概率；
- $P_y(k\mid t,u)$ 表示在决定输出时，具体输出哪个 token 的条件概率。

这比简单写成 $P(y,t\mid x)=P_B(t\mid x)\cdot P_y(y\mid x)$ 更接近工程实现，因为它明确表达了一个门控关系：先决定“发不发”，再决定“发什么”。

模型在每个流式时刻做两类判断：

| 概率项 | 职责 | 直观含义 | 训练重点 |
|---|---|---|---|
| $P_B$ | 等待 | 现在信息还不够，继续听 | 学对齐、学节奏、学延迟控制 |
| $P_y$ | 输出 | 现在可以吐出哪个 token | 学语言内容、学词序、学词表映射 |

这就是 Factorized Transducer 的核心思想。它的好处不是公式更漂亮，而是工程上可控。因为“是否输出”和“输出什么”被拆开后，系统能稳定地维持保序输出。所谓“保序”，就是输出文本的顺序与音频时间顺序一致，不会乱跳。

为什么这种拆分对 LLM 特别重要？因为仅解码器 LLM 本来擅长的是“给定已有 token，预测下一个 token”，它天然擅长 $P_y$，但并不天然擅长 $P_B$。也就是说，大模型知道“下一词可能是什么”，不代表它知道“现在该不该说”。Speech ReaLLM 的设计，本质上是在 LLM 外面补上时间流控制；Transducer-Llama 则把这种控制做成更明确的 factorized predictor/joiner 结构。

训练通常分两步。

第一步，用弱 LM 学会流式节奏。弱 LM 指参数小、计算便宜的语言模型。它的任务不是把语言能力做到极致，而是把 $P_B$ 和 $P_y$ 的边界学清楚。这个阶段通常依赖 RNN-T 风格的损失，重点是对齐和可训练性。

第二步，swap-in 强 LLM。也就是把 predictor 换成更强的仅解码器 LLM，再做 MWER 微调。MWER 全称 Minimum Word Error Rate，可以理解成“直接按整句词错误率优化”。如果只优化局部 token 概率，模型可能学会在局部很自信，但整体句子并不好；MWER 则更接近真实业务指标。

可把目标近似写成：

$$
\mathcal{L} = \mathcal{L}_{\text{RNNT}} + \lambda \mathcal{L}_{\text{MWER}}
$$

其中 $\lambda$ 控制两者权重。前者负责把流式路径训通，后者负责把最终识别结果拉近业务目标。

如果展开一点，RNN-T 损失可以理解为对所有合法对齐路径求和：

$$
\mathcal{L}_{\text{RNNT}}
=
-\log \sum_{\pi \in \mathcal{A}(x,y)} P(\pi\mid x)
$$

其中 $\mathcal{A}(x,y)$ 是所有能把输入音频 $x$ 对齐为目标文本 $y$ 的 blank/token 路径集合。它的作用是：不要求训练数据显式标出每个字对应哪一帧，但要求模型自己学会一条稳定对齐路径。

MWER 则更接近业务目标，可写成近似形式：

$$
\mathcal{L}_{\text{MWER}}
\approx
\sum_{h\in \mathcal{H}} \tilde{P}(h\mid x)\cdot \operatorname{WER}(h,y^\*)
$$

其中 $\mathcal{H}$ 是候选假设集合，$y^\*$ 是参考文本，$\tilde{P}$ 是归一化后的候选权重。它的直观意思是：模型不只看“当前 token 的概率高不高”，还看“整句错词数到底高不高”。

新手版理解是：先教一个系统学会节奏，再教它学内容。节奏没学会时，强 LLM 也会出错，而且错法很典型：

| 失败模式 | 表现 | 根因 |
|---|---|---|
| 长时间沉默 | 模型总觉得信息还不够 | $P_B$ 过强，延迟控制失衡 |
| 过早猜词 | 还没听清就凭语言先验补全 | $P_y$ 过强，声学约束不足 |
| 频繁抖动 | 同一前缀来回改写 | 决策边界不稳，稳定提交策略缺失 |
| 专名错误 | 通用词正确，实体名常错 | 词表和语音域不匹配 |

还有一个容易被忽略的数学问题：传统 Transducer 往往是局部归一化。局部归一化可以先理解为“每一步只在当前局部选最优”。它的问题是，一旦某一步做了输出决定，后面很难推翻，这会导致流式系统“不能改主意”。Samsung Research 在 ICASSP 2025 的工作讨论的正是这个缺陷，并报告了 9% 到 11% 的相对 WER 改善。直观上可以理解为：模型在早期输出时，如果每步都被 softmax 局部锁死，那么后续即使获得更多信息，也难以回收之前的承诺。

因此，近年的工作开始关注两条线同时优化：

| 方向 | 解决的问题 |
|---|---|
| 更强的流式 LLM 接入 | 提高语言内容建模能力 |
| 全局归一化或其近似 | 降低“过早承诺后无法反悔”的结构性缺陷 |

这两条线叠加后，流式 ASR 才有可能既快，又不至于因为结构缺陷而明显落后于带 lookahead 的模式。

---

## 代码实现

工程实现上，最关键的是把 streaming loop、blank/token 决策、LLM 状态缓存三件事接起来，而不是把 LLM 当成一次性整句生成器。

下面先看一个可运行的玩具例子。这个例子不做真实声学建模，只模拟“分块输入 -> blank 或 token 输出”的节奏，同时补上稳定提交策略。代码不依赖第三方库，直接用 `python3` 可运行。

```python
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class StreamState:
    committed: List[str] = field(default_factory=list)
    tentative: List[str] = field(default_factory=list)
    llm_context: List[str] = field(default_factory=list)
    steps: int = 0


def token_from(score: float) -> str:
    if score >= 0.85:
        return "你"
    if score >= 0.60:
        return "好"
    if score >= 0.45:
        return "吗"
    return "?"


def should_commit(blank_prob: float, token_score: float) -> bool:
    # 这里把“足够确定”近似成两个条件：
    # 1. 当前不是 blank
    # 2. token 分数足够高
    return blank_prob < 0.50 and token_score >= 0.60


def stream_decode(
    chunks: List[Tuple[float, float]],
    blank_threshold: float = 0.70,
) -> StreamState:
    state = StreamState()

    for blank_prob, token_score in chunks:
        state.steps += 1

        if blank_prob > blank_threshold:
            # 继续等待，不输出 token
            continue

        token = token_from(token_score)
        state.tentative.append(token)

        if should_commit(blank_prob, token_score):
            state.committed.append(token)
            state.llm_context.append(token)

    return state


if __name__ == "__main__":
    toy_chunks = [
        (0.95, 0.20),  # 等
        (0.90, 0.25),  # 等
        (0.42, 0.91),  # 输出并提交“你”
        (0.82, 0.10),  # 等
        (0.38, 0.66),  # 输出并提交“好”
    ]

    state = stream_decode(toy_chunks)

    committed_text = "".join(state.committed)
    tentative_text = "".join(state.tentative)

    assert committed_text == "你好"
    assert tentative_text == "你好"
    assert state.llm_context == ["你", "好"]

    print("committed:", committed_text)
    print("tentative:", tentative_text)
    print("steps:", state.steps)
```

这个例子表达了三个核心点：

| 代码对象 | 作用 | 对应真实系统中的概念 |
|---|---|---|
| `blank_prob` | 决定当前是否继续等 | $P_B$ |
| `token_score` | 决定若输出，优先发哪个 token | $P_y$ 的简化版 |
| `llm_context` | 缓存已提交 token 历史 | predictor/LLM 的文本状态 |
| `tentative` / `committed` | 区分试探输出与最终提交 | UI 的增量字幕策略 |

真实流式系统通常不会把所有中间输出都直接展示给用户，而是区分两层文本：

| 层级 | 含义 | 是否容易回退 |
|---|---|---|
| tentative | 当前猜测，允许被后续 chunk 改写 | 是 |
| committed | 已确认前缀，下游可以消费 | 否或很少 |

这一步非常重要。很多“识别模型明明不错，但产品体验差”的问题，并不出在模型，而出在 UI 提交策略。用户对轻微延迟的容忍度，往往高于对前缀来回跳动的容忍度。

下面给出一个更接近工程主干的最小实现。它依然是玩具版，但包含缓存、提交和时间推进逻辑。

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterable


@dataclass
class DecoderState:
    time_index: int = 0
    committed: List[str] = field(default_factory=list)
    llm_tokens: List[str] = field(default_factory=list)
    encoder_cache: List[float] = field(default_factory=list)


def encoder(chunk: List[float], cache: List[float]) -> float:
    # 玩具 encoder：用 chunk 均值模拟一段声学表示，并把最近值放入 cache
    value = sum(chunk) / len(chunk)
    cache.append(value)
    if len(cache) > 8:
        cache.pop(0)
    return value


def predictor(enc_feat: float, llm_tokens: List[str]) -> Dict[str, Any]:
    # 玩具 predictor：根据声学值决定 blank，结合历史 token 调整 token 倾向
    history_bonus = 0.05 * len(llm_tokens)
    blank_prob = max(0.05, min(0.95, 1.0 - enc_feat + history_bonus))
    vocab_scores = {
        "今": enc_feat * 0.90,
        "天": enc_feat * 0.85 + (0.10 if llm_tokens and llm_tokens[-1] == "今" else 0.0),
        "好": enc_feat * 0.70,
    }
    token = max(vocab_scores, key=vocab_scores.get)
    return {
        "blank_prob": blank_prob,
        "token": token,
        "token_score": vocab_scores[token],
    }


def should_emit(blank_prob: float, token_score: float, threshold: float) -> bool:
    return blank_prob < threshold and token_score >= 0.45


def update_llm_state(llm_tokens: List[str], token: str) -> List[str]:
    llm_tokens.append(token)
    return llm_tokens[-32:]  # 模拟前缀缓存窗口


def stream(frames: Iterable[List[float]]):
    for chunk in frames:
        yield chunk


def decode_stream(frames: Iterable[List[float]], blank_threshold: float = 0.60) -> str:
    state = DecoderState()

    for chunk in stream(frames):
        enc_feat = encoder(chunk, state.encoder_cache)
        pred = predictor(enc_feat, state.llm_tokens)

        if should_emit(pred["blank_prob"], pred["token_score"], blank_threshold):
            token = pred["token"]
            state.committed.append(token)
            state.llm_tokens = update_llm_state(state.llm_tokens, token)

        state.time_index += 1

    return "".join(state.committed)


if __name__ == "__main__":
    fake_audio = [
        [0.05, 0.10, 0.08, 0.07],
        [0.20, 0.18, 0.22, 0.24],
        [0.88, 0.91, 0.87, 0.89],
        [0.15, 0.10, 0.12, 0.13],
        [0.83, 0.80, 0.84, 0.82],
    ]

    text = decode_stream(fake_audio)
    print(text)
```

这个骨架对应的真实系统大致是：

```python
for chunk in stream():
    feat = extract_logmel(chunk, feature_cache)
    enc_feat = encoder(feat, encoder_cache)
    p_blank, p_token = predictor(enc_feat, llm_state)
    if p_blank > threshold:
        continue
    token = sample_or_argmax(p_token)
    ui.show_tentative(token)
    if is_stable(token, history, p_blank):
        emit(token)
        llm_state = update_llm_state(llm_state, token)
        ui.commit(token)
```

真实工程例子会多几层模块，但主干不变：

| 模块 | 作用 | 工程要求 |
|---|---|---|
| 特征提取 | 每 80-100ms 产出一批 log-Mel | 必须稳定、低抖动 |
| Encoder | 把声学特征编码成高层表示 | 支持缓存，避免重复算 |
| Predictor | 结合历史 token 和当前编码特征做决策 | 可从弱 LM 切换到强 LLM |
| Joiner / Factorized Head | 输出 $P_B$ 与 $P_y$ | blank 与 token 路径分离 |
| UI / 下游模块 | 实时展示、翻译、执行命令 | 支持增量更新 |

Tokenizer 和 embedding 也经常成为实际瓶颈。因为语音模型的输出粒度、LLM 的词表设计、下游任务的 token 分布并不天然一致。

| 适配项 | 为什么要做 | 常见做法 |
|---|---|---|
| 专门词表 | 原始 LLM 词表不一定适合 ASR | 用更小、更稳定的语音词表 |
| 小词嵌入桥接 | 直接复用大词表代价高 | 增加映射层或重训输入嵌入 |
| 状态缓存 | LLM 每步重算太贵 | KV cache 或前缀状态复用 |
| 中英文混合词表 | 混合语种口语里边界复杂 | 统一子词切分，单独处理数字/缩写 |
| 实体热词注入 | 人名地名缩写频繁出错 | shallow fusion、热词 bias、后处理纠错 |

工程上还要特别注意一次“流式输出”不只是一轮模型前向，它通常包含：

1. 音频缓冲区移动；
2. 特征缓存更新；
3. 编码器增量计算；
4. predictor 读取 LLM 历史状态；
5. blank/token 决策；
6. UI 提交与下游消费。

任何一步抖动过大，用户都会把它感知成“识别不流畅”。

---

## 工程权衡与常见坑

第一类坑是算力和收敛。直接用强 LLM 从零训流式路径，常见结果不是“更强”，而是“不稳定”。原因很简单：流式训练要求模型同时学会声学对齐、blank 决策、语言建模和低延迟节奏，这对大模型来说优化面太复杂。弱 LM -> 强 LLM 的两阶段策略，本质上是在降训练难度。

第二类坑是局部归一化。它会让系统过早承诺。一旦某个 token 在某个时刻被输出，后面即使听到更多信息，也很难修正。这在口音重、噪声大、专有名词多的场景尤其明显。

第三类坑是 lookahead 失控。未来上下文确实能提升准确率，但只要超出交互预算，用户体验就会快速恶化。很多离线评测里表现更好的模型，在线上不一定更好，原因就在这里。

第四类坑是“模型指标正确，但产品体验错误”。例如 WER 降了，但 UI 频繁回退；又如平均延迟不错，但首字延迟抖动太大，用户仍然觉得慢。流式系统必须把模型指标和交互指标一起看。

| 常见坑 | 具体表现 | 根因 | 规避策略 |
|---|---|---|---|
| 强 LLM 直接训练 | 显存爆炸、长期不收敛、输出卡住 | 同时学太多目标 | 先弱 LM 训练，再 swap-in 强 LLM |
| 局部归一化 | 不能改主意，前缀错了就一路错 | 决策在局部 softmax 被锁死 | 引入全局归一化近似或更稳的序列级训练 |
| lookahead 过大 | 准确率上升但实时感消失 | 用未来信息换准确率过头 | 控制在 $\le 320\text{ms}$ |
| 词表不匹配 | 专名、人名、混合语种频繁错 | LLM 词表与 ASR 域不一致 | 重设 tokenizer 和 embedding 桥接 |
| UI 频繁改写 | 用户看到字幕来回跳 | tentative/final 策略缺失 | 设计稳定提交策略 |
| 状态缓存失效 | 延迟突然抖动，吞吐下降 | KV cache 管理不当 | 做增量缓存复用和上限裁剪 |
| 长流漂移 | 说得越久越不稳 | 无端点流中的状态污染 | 定期重同步、分段缓存、热启动重置 |

这些坑里，最容易低估的是“状态缓存”。LLM 接入后，如果每个 chunk 都从头算 predictor，上线后几乎必然超预算。缓存设计通常至少要回答三个问题：

| 问题 | 如果不解决会怎样 |
|---|---|
| 历史 token 保留多长 | 太短丢上下文，太长算力爆炸 |
| cache 何时裁剪 | 长会话显存和延迟持续上升 |
| cache 如何与声学时间同步 | 文本状态和音频时间错位，blank/token 决策失真 |

训练流水线可以概括成一条简单路径：

`弱 LM + RNN-T 对齐 -> 替换成强 LLM predictor -> MWER 微调 streaming path`

这条路径背后的逻辑是先把“节奏”学会，再把“内容”做强。顺序反了，代价通常非常高。

部署时还要补上一条产品流水线：

`tentative 输出 -> 稳定判定 -> committed 提交 -> 下游 NLU/翻译消费`

如果只有模型，没有这条提交链路，系统仍然不会显得稳定。

一个很实用的经验是把监控拆成三层：

| 层级 | 关键指标 |
|---|---|
| 模型层 | WER、CER、blank rate、token rate |
| 时延层 | 首字延迟、词级延迟、P95/P99 延迟 |
| 体验层 | 回退次数、稳定提交比例、长会话漂移率 |

只看第一层，往往会高估系统上线效果。

---

## 替代方案与适用边界

Speech ReaLLM 不是所有业务的默认答案。它适合的是“高交互、低延迟、长流式”的语音场景，比如实时字幕、同传辅助、强交互语音助手。

如果业务允许更大延迟，传统方案依然合理。比如 Conformer + RNN-T + 束搜索，已经能覆盖很多手机助手与车载语音系统。束搜索可以先理解为“同时保留多条候选路径，最后选最优”，它能提升准确率，但会增加计算和延迟。

如果业务可以接受更明显的等待，比如 500ms 到 1s，chunked 非流式也是现实选择。它按块收集音频后统一解码，通常准确率更高，实现也更简单。

还可以把常见路线分成四类：

| 方案 | 首字延迟 | 词级延迟 | 准确率潜力 | 训练/部署复杂度 | 适用场景 |
|---|---:|---:|---|---|---|
| Speech ReaLLM / Transducer-Llama 类 | $<200\text{ms}$ | $<100\text{ms}$ | 高，接近更大非流式模型 | 高 | 实时字幕、强交互助手、连续语音流 |
| 传统 RNN-T / Conformer-T | 低到中 | 低到中 | 中到高 | 中 | 常规语音助手、车载、VAD 明确场景 |
| chunked 非流式 | 300ms-1s+ | 块级更新 | 更高 | 中到低 | 会议整理、字幕草稿、转写后编辑 |
| 全离线 AED/CTC+LM | 秒级 | 无 | 很高 | 低到中 | 文档转写、质检、归档 |

因此边界可以写成两个条件：

$$
\text{latency}_{\text{first token}} < 200\text{ms}
$$

$$
\Delta t_{\text{lookahead}} \le 320\text{ms}
$$

只要业务明确要求这两个条件，Speech ReaLLM 一类方案就有明显意义；如果业务不要求，就未必值得承担更复杂的训练与部署成本。

还可以再加一个更直接的业务判断表：

| 业务问题 | 如果答案是“是” | 更可能需要流式 LLM-ASR |
|---|---|---|
| 用户是否在边说边看屏幕？ | 用户能感知每次更新 | 是 |
| 下游模块是否边听边执行？ | 文本前缀会驱动指令或翻译 | 是 |
| 会话是否很长且无明确结束点？ | 不能等端点再统一处理 | 是 |
| 是否只要求高质量最终稿？ | 可以接受更长等待 | 否 |

一个直观判断标准是：如果用户把系统当“对话对象”，而不是“录音转文字工具”，那么流式机制的价值就会明显放大。

---

## 参考资料

先看整体，再看细节，顺序通常是：先理解 Speech ReaLLM 的系统目标，再看 Transducer-Llama 的训练与归一化问题，最后看工程侧的流水线拆解。

| 资料 | 出处 | 重点贡献 | 链接 |
|---|---|---|---|
| Speech ReaLLM | Seide 等，Interspeech 2024 | 将 decoder-only ASR 与 RNN-T 结合，强调 continuous audio 与无显式端点处理 | https://www.isca-archive.org/interspeech_2024/seide24_interspeech.html |
| Transducer-Llama | Deng 等，CoRR 2024 / OpenReview | 提出 Factorized Transducer、词表适配、弱 LM -> 强 LLM 的训练路径，并在 swap 后用 MWER 微调 | https://openreview.net/forum?id=l8hwUVcOQP |
| Globally Normalising the Transducer for Streaming Speech Recognition | van Dalen，ICASSP 2025 | 讨论局部归一化导致“难以改主意”的结构缺陷，并报告 9% 到 11% 相对 WER 改善 | https://research.samsung.com/research-papers/Globally-Normalising-the-Transducer-for-Streaming-Speech-Recognition |
| Streaming ASR Architecture | 工程实现类总结 | 从 VAD、特征、编码器、缓存、解码器到 UI 提交流水线梳理生产约束 | 工程博客可作为补充阅读，但应以论文结论为主 |

建议阅读顺序如下：

| 阅读顺序 | 目的 |
|---|---|
| 1. Speech ReaLLM | 建立“为什么要把 LLM 接进流式 ASR”的整体图景 |
| 2. Transducer-Llama | 理解 blank/token 分解、词表适配与两阶段训练 |
| 3. Samsung 全局归一化论文 | 理解“不能改主意”的数学根源 |
| 4. 工程总结文章 | 把论文机制映射到真实系统组件和延迟预算 |

参考引用格式可写为：Seide et al., 2024；Deng et al., 2024；van Dalen, 2025。

如果只记三条参考结论，可以记成：

| 结论 | 来源 |
|---|---|
| decoder-only LLM 也可以被改造成真正连续流式 ASR | Speech ReaLLM |
| 大模型不适合直接从零学流式节奏，应采用 weak-to-strong swap | Transducer-Llama |
| 传统 streaming transducer 的局部归一化存在结构性缺陷 | Globally Normalising the Transducer |
