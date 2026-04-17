## 核心结论

TTS（Text-to-Speech，文本转语音）里的“文本到频谱生成”，本质是一个前端声学建模问题：把清洗后的文本或音素序列，映射成一串按时间展开的 mel 频谱帧，再交给声码器生成波形。mel 频谱可以理解为“更贴近人耳感知、压缩过的时频特征图”，它不是声音本身，但已经包含了大部分可听出来的内容。

这一步之所以是核心，不是因为它单纯做了“编码”，而是因为它同时决定了三件事：句子说什么、每个音说多久、每段声音听起来多亮多重多平滑。声码器如 HiFi-GAN 可以把 mel 变成音频，但如果前面的 mel 频谱帧已经错了，后端无法补出正确的节奏、咬字和音色细节。

一个新手可理解的最小链路是：

| 输入阶段 | 处理模块 | 输出特征 |
|---|---|---|
| 原始文本 | 文本清洗 | 可朗读文本 |
| 可朗读文本 | G2P/音素转换 | 音素序列 |
| 音素序列 | duration predictor | 每个音素对应多少帧 |
| 音素+时长+能量/音高条件 | acoustic model | mel 频谱帧 |
| mel 频谱 | HiFi-GAN 等声码器 | 波形 |

例如 `Dr. Smith has 3 apples.` 不会直接送进声学模型。系统通常先做文本归一化，把缩写和数字变成可读形式，例如 `doctor smith has three apples`，再转成音素序列。之后 acoustic 模型预测每个音素该持续几帧、每帧大致的能量和频谱形状，最终输出一串 mel 频谱帧，供 HiFi-GAN 合成语音。

---

## 问题定义与边界

问题定义需要说清楚边界：文本到频谱生成不负责“直接生成最终音频”，它负责把语言信息投影成声学特征。投影，白话说，就是把“字词、语义、停顿、重音、时长”换成“每 10 ms 或 12.5 ms 一帧的频谱表示”。后端声码器再把这些帧还原成连续波形。

所以它的输入通常不是“纯文本”这么简单，而是多种条件的组合：

| 条件 | 白话解释 | 影响什么 |
|---|---|---|
| 文本/音素 | 说什么音 | 发音内容 |
| 时长 | 每个音占多少帧 | 节奏、停顿 |
| 音高 F0 | 声音高低走向 | 语调、情感 |
| 能量 | 响度分布 | 轻重、强调 |
| 说话人/风格向量 | 谁在说、怎么说 | 音色、风格 |

玩具例子：输入 `hi`，经过前端后可写成音素 `[h, aɪ]`。如果 duration predictor 预测时长为 `[4, 5]`，就表示第一个音素占 4 帧，第二个音素占 5 帧，总共输出 9 帧 mel。这里“帧”可以理解为固定 hop 大小下的一小段时间窗口。如果 hop 是 10 ms，那么 9 帧大约覆盖 90 ms。

这件事常用下面的关系来理解。设采样率为 $f_s$，STFT 窗长为 $N$，步长为 $H$，那么总帧数近似由波形长度和 hop 决定：

$$
T \approx \left\lfloor \frac{L-N}{H} \right\rfloor + 1
$$

在非自回归 TTS 中，模型并不是先有波形再切帧，而是反过来先预测每个音素的时长，再推导总帧数：

$$
T = \sum_{i=1}^{M} d_i
$$

其中 $d_i$ 是第 $i$ 个音素对应的帧数，$M$ 是音素个数。这个式子说明：duration predictor 实际上在决定整个频谱时间轴的骨架。一旦它错了，后面的频谱再细腻也会出现跳字、吞音或重复。

因此问题边界很清楚：文本到频谱生成是“多模态条件到频谱序列”的映射问题，核心约束是频谱一致性。所谓一致性，白话说，就是幅度、时间对齐、上下文语义和最终波形要互相说得通，而不是各自正确、组合后失真。

---

## 核心机制与推导

mel 频谱不是凭空定义的，它来自一条固定的信号处理链路。对离散波形 $x[n]$ 做短时傅里叶变换（STFT），得到每一帧每个频率 bin 上的复数谱：

$$
X(m,k)=\sum_{n=0}^{N-1}x[n+mH]\,w[n]\,e^{-j2\pi kn/N}
$$

这里 $m$ 是第几帧，$k$ 是频率索引，$w[n]$ 是窗函数。复数谱的模平方给出功率谱：

$$
P(m,k)=|X(m,k)|^2
$$

再通过 mel 滤波器组，把线性频率压缩到更接近人耳分辨率的 mel 频带：

$$
S_{\text{mel}}(m,i)=\sum_k P(m,k)\cdot H_{\text{mel}}(i,k)
$$

最后做对数压缩，得到模型常用的 log-mel 特征：

$$
M(m,i)=\log(S_{\text{mel}}(m,i)+\epsilon)
$$

这条链路有三个直接后果。

第一，mel 本身丢失了部分细粒度信息，尤其是相位信息。相位可以理解为“波形在周期里的对齐状态”，mel 主要保留幅度分布，所以必须依赖声码器补足。第二，mel 滤波会压缩高频细节，因此如果训练目标只看 L1/L2 误差，高频容易被平均掉，表现成辅音发虚、爆破音不清楚。第三，log 压缩会拉近强弱能量的数值距离，使模型更稳定，但也要求模型在低能量帧上格外小心，否则容易出现静音区脏噪声。

继续看 `hi -> [h, aɪ]` 的玩具例子。假设时长为 `[4,5]`，每帧 80 维 mel，那么输出就是一个 $9 \times 80$ 的矩阵。可以把它理解为：

- 第 1 到 4 帧主要描述 `[h]` 的无声摩擦特征，高频成分相对突出，能量较低。
- 第 5 到 9 帧主要描述 `[aɪ]` 的元音结构，低中频共振峰更明显，能量更高、更平滑。

这也是为什么只知道“文本是 hi”远远不够。模型必须知道边界在哪里、每段持续多久、哪些帧该亮、哪些帧该弱。

为了同时建模全局和局部，很多系统会引入分层潜变量。潜变量，白话说，就是模型内部学出来但不直接写在输入里的控制因素。句子级潜变量控制整体节奏、情感和说话风格；音素级潜变量控制局部音色、辅音爆破、元音过渡。像 HierTTS 这类方法的思路就是把不同语义尺度的信息逐层耦合，而不是让一个单层向量负责全部内容。

可以把机制简化成：

```text
文本 -> 音素 -> 时长展开 -> mel 解码 -> post-net 修正 -> 声码器
```

其中“时长展开”尤其关键。它把音素级序列扩展成帧级序列，相当于先决定时间轴，再填充每一帧的频谱值。如果没有这一步，模型就必须自己同时学“说什么”和“每个音拖多久”，训练会更不稳定。

下面这个表格把关键变量和作用放在一起：

| 变量 | 含义 | 在系统中的角色 |
|---|---|---|
| $X(m,k)$ | STFT 复数谱 | 原始时频表示，含相位 |
| $P(m,k)$ | 功率谱 | 频率能量分布 |
| $S_{\text{mel}}(m,i)$ | mel 滤波后能量 | 压缩后的感知频带表示 |
| $M(m,i)$ | log-mel 特征 | acoustic model 的常见训练目标 |
| $d_i$ | 音素时长 | 决定帧数和对齐边界 |
| $F0$ | 基频 | 控制语调起伏 |
| energy | 帧能量 | 控制轻重和清晰度 |

真实工程例子里，这种机制直接决定部署效果。实时客服场景通常不接受自回归模型一帧一帧慢慢生成，因为并发高、首包时间敏感。于是工程上更偏向 FastSpeech 2 这类非自回归结构：先预测 duration、pitch、energy，再并行生成整段 mel，最后交给 HiFi-GAN。这样整体延迟可以压到 60 ms 左右，但代价是 MOS 往往略低于 Tacotron2 一类高质量自回归模型。

---

## 代码实现

一个最小可运行的思路，不需要真的训练神经网络，也能把“文本到频谱”的骨架说明白。下面的代码演示三件事：文本转音素、根据音素预测帧数、展开成 mel 矩阵，并在低能量帧上做简单噪声填充。

```python
import math
import random

def text_to_phonemes(text: str):
    lexicon = {
        "hi": ["h", "aɪ"],
    }
    return lexicon[text.lower()]

def duration_predictor(phonemes):
    # 玩具规则：辅音短，元音长
    durations = []
    for p in phonemes:
        if p in {"aɪ", "i", "u", "a"}:
            durations.append(5)
        else:
            durations.append(4)
    return durations

def mel_decoder(phonemes, durations, mel_bins=80):
    frames = []
    for p, d in zip(phonemes, durations):
        for t in range(d):
            row = []
            for i in range(mel_bins):
                base = 0.15 if p == "h" else 0.45
                # 简单模拟：元音更平滑、能量更高
                value = base + 0.05 * math.sin((t + 1) * (i + 1) / 17.0)
                row.append(value)
            frames.append(row)
    return frames

def postnet(mel):
    # 简化版后处理：做轻微残差增强
    enhanced = []
    for row in mel:
        enhanced.append([x * 1.05 for x in row])
    return enhanced

def low_energy_noise_fill(mel, threshold=0.18):
    filled = []
    for row in mel:
        avg_energy = sum(row) / len(row)
        if avg_energy < threshold:
            filled.append([x + random.gauss(0.0, 0.01) for x in row])
        else:
            filled.append(row)
    return filled

phonemes = text_to_phonemes("hi")
durations = duration_predictor(phonemes)
mel = mel_decoder(phonemes, durations)
mel = postnet(mel)
mel = low_energy_noise_fill(mel)

assert phonemes == ["h", "aɪ"]
assert durations == [4, 5]
assert len(mel) == 9
assert len(mel[0]) == 80
assert sum(durations) == 9
```

这段代码对应的系统含义如下：

- `text_to_phonemes`：把文字变成更稳定的发音单位。音素就是“发音最小单位”，比字符更适合喂给声学模型。
- `duration_predictor`：决定每个音素展开成多少帧。这一步决定时间轴。
- `mel_decoder`：根据音素和时长，生成帧级 mel。真实系统里这里通常是 Transformer、Conformer 或卷积解码器。
- `postnet`：对初始 mel 做残差修正。残差，白话说，就是“先给粗版本，再补细节”。
- `low_energy_noise_fill`：在低能量帧做噪声补齐，避免静音或弱音处数值塌陷。真实系统里这类策略常配合 mask 与声码器训练一起设计。
- `hifi_gan(mel)` 没在这里实现，但真实部署里它负责把 mel 变成波形，同时学习更合理的相位与细节分布。

所以新手常见的伪代码：

```python
phonemes = text_to_phonemes("hi")
durations = duration_predictor(phonemes)
mel = mel_decoder(phonemes, durations)
mel = postnet(mel)
waveform = hifi_gan(mel)
```

每一步都不是“可选优化”，而是对最终语音质量有明确贡献的模块化职责划分。

---

## 工程权衡与常见坑

工程上最常见的选择，不是“哪种模型理论最好”，而是“延迟、稳定性、质量三者怎么平衡”。非自回归模型的价值在这里非常直接：先预测时长、音高、能量，再并行生成频谱，适合实时系统；自回归模型质量常更高，但速度慢且容易累积错误。

实时客服是典型真实工程例子。并发高、响应预算紧、可接受的 MOS 不是绝对最高而是稳定可用，因此常选 FastSpeech 2。它可能只有约 3.9 的 MOS，但端到端延迟可控制在 60 ms 量级。相比之下，Tacotron2 可能接近 4.2 MOS，但推理更慢、长句更容易出错。

常见坑可以归纳如下：

| 坑 | 原因 | 规避方式 |
|---|---|---|
| 对齐崩盘 | 时长预测错，帧展开错位 | 强对齐模块、时长监督、mask 检查 |
| 跳帧/重复 | 对齐器不稳定或训练数据脏 | Mixture-TTS 类强对齐、清洗标注 |
| 频谱过平滑 | L1/L2 平均化高频细节 | post-net、GAN loss、感知损失 |
| 波形金属感 | mel 不含相位，重建不足 | HiFi-GAN 等神经声码器 |
| 静音区噪点 | 低能量帧数值不稳 | energy mask、噪声填充、静音建模 |
| 长句失稳 | 全局节奏控制不足 | 句子级风格向量、分层潜变量 |

落地时还应注意几件事：

- 对齐 mask 不能省。训练时必须明确哪些帧有效、哪些是 padding，否则 duration predictor 会学偏。
- GAN loss 要适度。太弱时频谱发糊，太强时训练震荡。
- HiFi-GAN 虽然不是“显式恢复相位”，但它通过神经生成过程学习了更合理的波形结构，通常比 Griffin-Lim 更自然。
- 低资源数据集下，duration predictor 往往比 decoder 更先出问题，因为它决定了整个时间骨架。
- 文本归一化必须和训练分布一致。`Dr.`、数字、日期、缩写如果处理不统一，前端错误会直接传到声学模型。

---

## 替代方案与适用边界

常见替代路线主要有两类：自回归声学模型和更激进的端到端并行/扩散式声学模型。

| 架构 | 延迟 | MOS | 稳定性 | 适用场景 |
|---|---|---|---|---|
| Tacotron2 | 高 | 较高 | 中等，长句易失稳 | 品质优先、离线合成 |
| FastSpeech 2 | 低 | 中高 | 高 | 实时服务、并发场景 |
| Diffusion/对抗式并行 decoder | 中低到中 | 潜力高 | 依赖训练设计 | 高质量并行生成探索 |

Tacotron2 的优点是质量通常不错，因为它逐步生成频谱，局部建模细腻；缺点也来自这里：逐帧生成导致延迟高，且错误会累积。FastSpeech 2 的优势是稳定和快，因为先把 duration、pitch、energy 预测好，再一次性并行解码。但前提是 duration predictor 足够可靠，否则会出现“整体快但整体错”。

如果是在线客服、语音助手、边缘设备推理，优先考虑 FastSpeech 2 一类并行方案，因为它更容易满足时延和吞吐要求。如果是有后处理预算的高品质配音、长音频内容生产，自回归方案仍有价值。若采用更轻量或低资源方案，例如并行对抗式 decoder 或扩散式 decoder，也必须确认一点：时长和对齐监督是否足够强，否则质量上限很难稳定达到。

所以适用边界不是“谁先进就用谁”，而是：

- 实时性优先：选非自回归并行生成。
- 绝对音质优先：选高质量自回归或更重的生成器。
- 数据少、标注弱：优先保证对齐与时长建模，别先追求复杂 decoder。
- 多说话人、多风格：优先引入分层条件控制，而不是只堆大模型参数。

---

## 参考资料

1. HackMD, *From Air Pressure to Speech*. 适合先看 text → phoneme → mel 的整体分层链路，重点关注前端表示和声学特征之间的过渡。
2. Emergent Mind, *Mel-Spectrogram Decoder*. 适合查 STFT、mel 滤波、log 压缩的数学基础，重点看频谱表示和 decoder 约束。
3. Deepgram, *Text-to-Speech Architecture and Production Tradeoffs*. 适合看 AR/NAR 的工程权衡，重点看延迟、MOS、稳定性对比。
4. MDPI, 关于 HierTTS 与分层潜变量建模的综述资料。适合理解“句子级语义控制整体风格、音素级变量控制局部音色”的设计思路。
5. PMC 综述资料，涉及对齐失败、过平滑、相位恢复与神经声码器等常见问题。适合查失败模式和规避策略。

新手的快速阅读顺序可以是：先看 HackMD 建立 text→phoneme→mel 的流程感，再看 Emergent Mind 补数学，再看 Deepgram 理解为什么生产环境不总是选最高 MOS 的模型。
