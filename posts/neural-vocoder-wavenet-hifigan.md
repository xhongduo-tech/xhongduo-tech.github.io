## 核心结论

Vocoder，中文常译为“声码器”，在现代 TTS 系统里的职责很单一：它不处理文本，也不决定发音时长，而是把上游声学模型输出的 Mel 频谱 $S$ 还原为可播放的音频波形 $x$。

它要解决的是同一个条件生成问题：

$$
p(x \mid S)
$$

即“在给定 Mel 频谱的条件下，生成与之匹配的波形”。

从 WaveNet 到 WaveGlow 再到 HiFi-GAN，路线变化很大，但目标没有变。三代方法的核心差异在于“如何建模这件事”：

- WaveNet 采用自回归方式，逐采样点生成，音质高，但推理天然串行。
- WaveGlow 采用可逆流模型，通过可逆变换把简单分布映射成真实波形分布，推理可并行。
- HiFi-GAN 采用生成对抗网络，直接学习从 Mel 到波形的并行映射，并用多判别器约束细节，因此在音质、延迟和部署复杂度之间取得了更好的平衡。

“把 Mel 频谱想成语音的乐谱”是一个足够准确的入门直觉，但要补一句：乐谱只告诉你哪些频率成分在什么时候出现，并不直接包含完整波形相位与微观振动细节，所以声码器做的不是简单还原，而是带约束的重建。

| 模型 | 结构类型 | 生成方式 | 并行度 | 主要优点 | 主要缺点 |
| --- | --- | --- | --- | --- | --- |
| WaveNet | 自回归卷积 | 逐采样点预测 | 低 | 音质高，细节建模强 | 推理慢，实时部署成本高 |
| WaveGlow | Flow，可逆流 | 整段并行生成 | 高 | 有显式似然目标，采样快 | 高频细节和自然度常弱于 GAN |
| HiFi-GAN | GAN，对抗生成 | 整段并行生成 | 高 | 音质高、延迟低、工程成熟 | 训练依赖损失设计与稳定性 |

如果只记一句结论：在经典 Mel-spectrogram TTS pipeline 中，HiFi-GAN 往往是更常见的工程选择；WaveNet 更像高质量自回归基线；WaveGlow 是“并行概率建模”的重要过渡方案。

---

## 问题定义与边界

Mel 频谱不是声音本身，而是声音在频域上的压缩表示。它保留了大部分发音结构，例如能量分布、共振峰变化、音节边界和部分韵律信息，但播放器不能直接播放 Mel 图，必须先把它恢复成时域波形。

形式化定义如下。输入是一个 Mel 频谱：

$$
S \in \mathbb{R}^{F \times T}
$$

其中：

- $F$ 是 Mel 频带数，例如 80。
- $T$ 是时间帧数，例如 100、200、500。

输出是一段一维波形：

$$
x \in \mathbb{R}^{N}
$$

其中 $N$ 是采样点数，通常远大于 $T$。如果采样率是 22050 Hz，那么 1 秒音频对应 22050 个采样点，而 1 秒 Mel 序列可能只有约 80 到 100 帧。声码器本质上是在做：

$$
\text{低时间分辨率特征} \rightarrow \text{高时间分辨率波形}
$$

这也是为什么它不能靠普通插值解决。插值只能补长度，不能补真实语音里关键的三类细节：

| 细节类型 | 作用 | 丢失后的现象 |
| --- | --- | --- |
| 周期结构 | 决定基音、音高稳定性 | 元音发虚、发飘 |
| 谐波结构 | 决定音色层次与亮度 | 声音发闷、缺空气感 |
| 瞬态与高频细节 | 决定辅音清晰度和边缘感 | 摩擦音糊、爆破音弱 |

边界也必须说清楚：

1. Vocoder 不负责文本理解。
2. Vocoder 不负责文本到音素的对齐。
3. Vocoder 不直接决定句子“该怎么说”，它接收的是已经生成好的声学特征。
4. Vocoder 的目标是尽量真实地恢复波形，同时满足部署约束。

工程中最常看的指标通常有三类：

| 维度 | 关注点 | 典型问题 |
| --- | --- | --- |
| 音质 | 是否自然、清晰、无噪声 | 发糊、金属音、毛刺 |
| 实时性 | 能否低延迟生成 | 自回归模型吞吐不足 |
| 稳定性 | 长音频、复杂音色下是否稳定 | 断裂、漂移、爆音 |

新手最容易困惑的是长度关系。一个具体例子：

假设 Mel 帧数 $T=100$，生成器的上采样倍率为 $[8,8,2,2]$，那么输出波形长度为：

$$
N = 100 \times 8 \times 8 \times 2 \times 2 = 25600
$$

如果采样率是 22050 Hz，则音频时长约为：

$$
\frac{25600}{22050} \approx 1.16 \text{ 秒}
$$

如果这套上采样倍率与数据预处理里的 `hop_size` 对不上，模型即使训练收敛，输出也会系统性错位。这个约束可以直接写成：

$$
\prod_{i=1}^{K} u_i = \text{hop\_size}
$$

其中 $u_i$ 是第 $i$ 层上采样倍率。这个等式在 HiFi-GAN、MelGAN、GAN-TTS 一类模型里都很关键。

---

## 核心机制与推导

三类模型的区别，不在于“有没有学波形”，而在于“用什么建模假设去学波形”。

### 1. WaveNet：把波形建模成条件自回归序列

WaveNet 的核心思想是：把波形看成一个长序列，每个采样点都依赖之前的采样点与条件特征。其条件概率分解为：

$$
p(x \mid S) = \prod_{n=1}^{N} p(x[n] \mid x[<n], \widetilde S)
$$

其中：

- $x[n]$ 是第 $n$ 个采样点。
- $x[<n]$ 表示前面所有已生成采样点。
- $\widetilde S$ 是已经上采样或对齐到采样点级别的条件特征。

WaveNet 使用两类关键结构：

| 结构 | 作用 | 直观理解 |
| --- | --- | --- |
| 因果卷积 | 保证当前位置不能看到未来 | 推理时不作弊 |
| 膨胀卷积 | 扩大感受野 | 用较少层数覆盖长历史 |

若卷积核大小为 $k=2$，膨胀率依次为 $1,2,4,\dots,2^{L-1}$，则一组堆叠后的感受野近似为：

$$
R = 1 + \sum_{l=0}^{L-1} (k-1)2^l = 2^L
$$

这说明膨胀卷积能指数级扩大可见历史长度，这对语音很重要，因为语音中的基音周期、谐波变化和长时依赖都要求足够大的感受野。

但它的代价同样明确：推理必须按时间顺序一个点一个点生成，因此复杂度与输出长度线性绑定。生成 1 秒 22.05kHz 音频，就要做约 22050 次自回归预测。这个慢不是实现不够优化，而是概率分解方式决定的。

### 2. WaveGlow：把生成问题改写为可逆概率变换

WaveGlow 走的是 Flow 路线。基本思想是：先从一个简单分布采样隐变量 $z$，然后通过一系列可逆变换得到真实音频 $x$。记变换为：

$$
x = f_\theta(z; S), \quad z \sim \mathcal{N}(0, I)
$$

由于 $f_\theta$ 可逆，因此训练时可以使用变量替换公式最大化似然：

$$
\log p(x \mid S)
=
\log p(z) + \log \left| \det \frac{\partial z}{\partial x} \right|
$$

如果将整体变换拆成多层可逆模块，则有：

$$
\log p(x \mid S)
=
\log p(z) + \sum_{k=1}^{K} \log \left| \det J_k \right|
$$

其中 $J_k$ 是第 $k$ 个可逆变换的雅可比矩阵。

对新手来说，这个公式可以这样理解：

- $\log p(z)$ 约束“映射到隐空间后要像高斯分布”。
- 雅可比项修正“从隐空间拉伸、压缩到波形空间时的体积变化”。

WaveGlow 常见的模块有：

| 模块 | 作用 |
| --- | --- |
| 仿射耦合层 | 保持可逆，同时提高表达能力 |
| 可逆 $1 \times 1$ 卷积 | 让通道充分混合 |
| 早期输出机制 | 降低后续层计算负担 |

它的优势在于：训练目标是显式似然，推理时整段并行逆变换，不需要像 WaveNet 一样逐点采样。

### 3. HiFi-GAN：直接并行生成，再用判别器补足细节

HiFi-GAN 不再显式写出条件概率，也不要求可逆结构，而是直接学习一个生成器：

$$
\hat{x} = G(S)
$$

其中 $\hat{x}$ 是生成波形。然后用多个判别器判断 $\hat{x}$ 是否足够像真实音频 $x$。

它通常由三类损失共同训练。

#### 对抗损失

生成器希望骗过判别器。以 least-squares GAN 形式为例：

$$
\mathcal{L}_{Adv}(G;D)=\mathbb{E}_{S}\left[(D(G(S))-1)^2\right]
$$

对应的判别器损失通常写为：

$$
\mathcal{L}_{D}
=
\mathbb{E}_{x}\left[(D(x)-1)^2\right]
+
\mathbb{E}_{S}\left[D(G(S))^2\right]
$$

它约束的是“整体听感是否像真音频”。

#### 特征匹配损失

只看真假分数不够，因为这个监督太粗。HiFi-GAN 会比较判别器中间层特征：

$$
\mathcal{L}_{FM}(G;D)
=
\mathbb{E}_{(x,S)}
\left[
\sum_{l=1}^{L}
\frac{1}{N_l}
\left\|
D^{(l)}(x) - D^{(l)}(G(S))
\right\|_1
\right]
$$

其中：

- $D^{(l)}(\cdot)$ 是判别器第 $l$ 层特征；
- $N_l$ 是该层特征维度，用于归一化。

这项损失的作用不是判断真假，而是强迫生成音频在“判别器看到的多层结构”上接近真实音频。它通常能明显稳定训练。

#### 谱重建损失

HiFi-GAN 论文中常直接对 Mel 频谱做重建损失，也常扩展为多分辨率 STFT 损失。简化写法如下：

$$
\mathcal{L}_{Mel}(G)
=
\left\|
\phi(x) - \phi(G(S))
\right\|_1
$$

其中 $\phi(\cdot)$ 表示 Mel 变换或 STFT 幅度映射。

它负责的不是“真假感”，而是“频域内容别偏太远”。这对辅音清晰度、能量轮廓和整体可懂度很关键。

#### 总损失

因此，生成器训练目标通常是加权和：

$$
\mathcal{L}_{G}
=
\mathcal{L}_{Adv}
+
\lambda_{fm}\mathcal{L}_{FM}
+
\lambda_{mel}\mathcal{L}_{Mel}
$$

这就是 HiFi-GAN 的工程本质：生成器负责并行合成，判别器负责逼细节，谱损失负责兜住内容。

### 4. 为什么多周期判别器有效

HiFi-GAN 的关键创新之一是判别器不是单一路径，而是分成两组：

| 判别器 | 关注对象 | 作用 |
| --- | --- | --- |
| MSD，多尺度判别器 | 不同时间分辨率下的整体结构 | 保证整体波形自然度 |
| MPD，多周期判别器 | 固定周期切分后的局部结构 | 强化基音与谐波规律 |

语音不同于一般噪声序列。浊音部分通常有明显周期性，周期长度与基频 $f_0$ 有关。若采样率为 $f_s$，基频为 $f_0$，则一个周期大致对应：

$$
P \approx \frac{f_s}{f_0}
$$

例如 $f_s = 22050$ Hz、$f_0 = 220$ Hz，则周期约为：

$$
P \approx \frac{22050}{220} \approx 100
$$

多周期判别器虽然不直接显式估计 $f_0$，但它通过多种周期切分方式，让生成器更容易学到“语音该有的周期结构”。这比单个全局判别器更贴近语音本身的统计特性。

### 5. 三类方法的机制差异

| 模型 | 感受野构造 | 采样方式 | 训练目标 |
| --- | --- | --- | --- |
| WaveNet | 因果膨胀卷积堆叠 | 串行逐点采样 | 条件自回归概率 |
| WaveGlow | 可逆耦合层 + 可逆卷积 | 并行逆流采样 | 最大化对数似然 |
| HiFi-GAN | 上采样块 + 残差块 + 多判别器 | 并行一次生成 | 对抗损失 + 特征匹配 + 谱重建 |

一句话概括：

- WaveNet 靠“逐点条件建模”。
- WaveGlow 靠“可逆概率变换”。
- HiFi-GAN 靠“直接生成并让判别器补细节”。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不依赖第三方库，但把 HiFi-GAN 训练中的几个关键结构都保留下来了：

1. 上采样倍率必须与 `hop_size` 对齐。
2. 生成器负责把低帧率序列扩展到波形长度。
3. 多尺度/多周期判别器可以抽象成“不同视角的特征提取器”。
4. 总损失由对抗项、特征匹配项、谱重建项组成。

```python
import math


def upsample_length(mel_frames, rates):
    length = mel_frames
    for r in rates:
        length *= r
    return length


def l1_loss(xs, ys):
    assert len(xs) == len(ys), "length mismatch"
    return sum(abs(a - b) for a, b in zip(xs, ys)) / max(len(xs), 1)


def mse_loss(xs, ys):
    assert len(xs) == len(ys), "length mismatch"
    return sum((a - b) ** 2 for a, b in zip(xs, ys)) / max(len(xs), 1)


def average_pool(xs, stride):
    pooled = []
    for i in range(0, len(xs), stride):
        chunk = xs[i:i + stride]
        pooled.append(sum(chunk) / len(chunk))
    return pooled


def fake_mel_projection(waveform, hop_size):
    """用简单分帧均值模拟 Mel 投影，只演示长度与约束关系。"""
    return average_pool(waveform, hop_size)


class ToyResidualBlock:
    def forward(self, xs):
        out = []
        for i, x in enumerate(xs):
            left = xs[i - 1] if i - 1 >= 0 else xs[i]
            right = xs[i + 1] if i + 1 < len(xs) else xs[i]
            out.append(0.6 * x + 0.2 * left + 0.2 * right)
        return out


class ToyHiFiGenerator:
    def __init__(self, upsample_rates):
        self.upsample_rates = upsample_rates
        self.blocks = [ToyResidualBlock() for _ in upsample_rates]

    def forward(self, mel_frames):
        waveform = mel_frames[:]
        for r, block in zip(self.upsample_rates, self.blocks):
            expanded = []
            for v in waveform:
                for _ in range(r):
                    expanded.append(v)
            waveform = block.forward(expanded)
        return waveform


class ToyDiscriminator:
    def __init__(self, name, view_fn):
        self.name = name
        self.view_fn = view_fn

    def score(self, waveform):
        viewed = self.view_fn(waveform)
        energy = sum(abs(v) for v in viewed) / max(len(viewed), 1)
        sharpness = sum(
            abs(viewed[i] - viewed[i - 1]) for i in range(1, len(viewed))
        ) / max(len(viewed) - 1, 1)
        return 0.7 * energy + 0.3 * sharpness

    def features(self, waveform):
        viewed = self.view_fn(waveform)
        mean_abs = sum(abs(v) for v in viewed) / max(len(viewed), 1)
        max_abs = max(abs(v) for v in viewed) if viewed else 0.0
        roughness = sum(
            abs(viewed[i] - viewed[i - 1]) for i in range(1, len(viewed))
        ) / max(len(viewed) - 1, 1)
        return [mean_abs, max_abs, roughness]


def make_multi_scale_discriminators():
    return [
        ToyDiscriminator("msd_s1", lambda x: x),
        ToyDiscriminator("msd_s2", lambda x: average_pool(x, 2)),
        ToyDiscriminator("msd_s4", lambda x: average_pool(x, 4)),
    ]


def make_multi_period_discriminators(periods):
    discriminators = []
    for p in periods:
        def period_view(x, p=p):
            buckets = []
            for start in range(p):
                group = x[start::p]
                if group:
                    buckets.append(sum(group) / len(group))
            return buckets or [0.0]
        discriminators.append(ToyDiscriminator(f"mpd_p{p}", period_view))
    return discriminators


def adversarial_generator_loss(fake_scores):
    return mse_loss(fake_scores, [1.0] * len(fake_scores))


def adversarial_discriminator_loss(real_scores, fake_scores):
    real_loss = mse_loss(real_scores, [1.0] * len(real_scores))
    fake_loss = mse_loss(fake_scores, [0.0] * len(fake_scores))
    return real_loss + fake_loss


def feature_matching_loss(real_feature_maps, fake_feature_maps):
    assert len(real_feature_maps) == len(fake_feature_maps)
    losses = []
    for real_f, fake_f in zip(real_feature_maps, fake_feature_maps):
        losses.append(l1_loss(real_f, fake_f))
    return sum(losses) / len(losses)


def collect_scores_and_features(discriminators, real_wave, fake_wave):
    real_scores = []
    fake_scores = []
    real_features = []
    fake_features = []
    for disc in discriminators:
        real_scores.append(disc.score(real_wave))
        fake_scores.append(disc.score(fake_wave))
        real_features.append(disc.features(real_wave))
        fake_features.append(disc.features(fake_wave))
    return real_scores, fake_scores, real_features, fake_features


def total_generator_loss(
    discriminators, real_wave, fake_wave, real_mel, hop_size,
    lambda_fm=2.0, lambda_mel=45.0
):
    _, fake_scores, real_features, fake_features = collect_scores_and_features(
        discriminators, real_wave, fake_wave
    )
    adv = adversarial_generator_loss(fake_scores)
    fm = feature_matching_loss(real_features, fake_features)
    fake_mel = fake_mel_projection(fake_wave, hop_size)
    mel = l1_loss(real_mel, fake_mel)
    return adv + lambda_fm * fm + lambda_mel * mel


def main():
    mel_frames = [0.1, 0.2, 0.0, -0.1] * 25  # 100 帧
    hop_size = 256
    upsample_rates = [8, 8, 2, 2]

    assert math.prod(upsample_rates) == hop_size, "upsample rates must match hop_size"

    generator = ToyHiFiGenerator(upsample_rates)
    fake_wave = generator.forward(mel_frames)

    expected_length = upsample_length(len(mel_frames), upsample_rates)
    assert len(fake_wave) == expected_length == 25600

    # 构造一段“真实波形”：与生成波形接近，但加入轻微周期扰动
    real_wave = [
        v + 0.02 * math.sin(2.0 * math.pi * i / 100.0)
        for i, v in enumerate(fake_wave)
    ]

    # 多尺度 + 多周期判别器
    discriminators = make_multi_scale_discriminators()
    discriminators += make_multi_period_discriminators([2, 3, 5, 7, 11])

    real_mel = fake_mel_projection(real_wave, hop_size)
    fake_mel = fake_mel_projection(fake_wave, hop_size)
    assert len(real_mel) == len(mel_frames)
    assert len(fake_mel) == len(mel_frames)

    real_scores, fake_scores, _, _ = collect_scores_and_features(
        discriminators, real_wave, fake_wave
    )
    d_loss = adversarial_discriminator_loss(real_scores, fake_scores)
    g_loss = total_generator_loss(
        discriminators, real_wave, fake_wave, real_mel, hop_size
    )

    print("waveform_length =", len(fake_wave))
    print("duration_seconds =", round(len(fake_wave) / 22050, 4))
    print("disc_count =", len(discriminators))
    print("discriminator_loss =", round(d_loss, 6))
    print("generator_loss =", round(g_loss, 6))


if __name__ == "__main__":
    main()
```

这段代码故意没有实现真正的卷积、STFT 和神经网络参数更新，因为目标不是复刻论文，而是把结构关系讲清楚。它体现了三个工程上必须理解的事实：

| 代码部件 | 对应真实系统中的概念 |
| --- | --- |
| `upsample_rates` | 生成器每层上采样倍率 |
| `assert math.prod(...) == hop_size` | 长度对齐约束 |
| `make_multi_scale_discriminators()` | 多尺度判别器 |
| `make_multi_period_discriminators()` | 多周期判别器 |
| `fake_mel_projection()` | 频域重建损失的抽象 |
| `total_generator_loss()` | 对抗 + FM + 谱损失组合 |

如果换成真实 PyTorch 版本，通常会变成下面的结构：

| 玩具实现 | 真实 HiFi-GAN 常见实现 |
| --- | --- |
| 列表复制上采样 | `ConvTranspose1d` 或插值 + `Conv1d` |
| 简单双边平滑 | 多个 dilation residual block |
| 手写特征 | 判别器中间层 feature map |
| `fake_mel_projection` | Mel spectrogram 或 multi-resolution STFT |
| 标量损失 | mini-batch 上逐项求和/平均 |

一个常见工程组合是 `Tacotron2 + HiFi-GAN`：

1. `Tacotron2` 负责文本到 Mel。
2. `HiFi-GAN` 负责 Mel 到波形。
3. 两者可以独立替换和调优。

这也是现代 TTS pipeline 模块化设计的核心好处：声学模型和声码器解耦，问题边界清楚，系统更容易迭代。

---

## 工程权衡与常见坑

WaveNet、WaveGlow、HiFi-GAN 的优劣不是抽象的“论文指标”，而是直接体现在部署成本、训练稳定性和可维护性上。

### 1. WaveNet 的主要代价是串行推理

WaveNet 通常不是最难训练的，但常常是最难部署的。原因很直接：输出波形有多少采样点，模型就要做多少次条件预测。哪怕单步很快，总量仍然大。

| 问题 | 结果 |
| --- | --- |
| 逐点生成 | 吞吐量低 |
| 高采样率音频 | 推理时间迅速增加 |
| 长上下文依赖 | cache 与工程实现复杂 |

因此它更适合作为：

- 高质量研究基线
- 自回归概率建模分析工具
- 离线生成场景

### 2. WaveGlow 的问题通常出在“够快，但未必最好听”

WaveGlow 的优势是清晰的：

- 并行生成
- 显式似然目标
- 结构上比自回归更适合高吞吐推理

但在实际语音系统里，它经常暴露两个问题：

| 现象 | 常见原因 |
| --- | --- |
| 高频空气感不足 | Flow 目标更重分布拟合，未必最强调感知细节 |
| 摩擦音边缘偏软 | 缺少对抗式细节约束 |

这并不表示 Flow 路线无效，而是说明“概率上可解释”和“听感上最自然”不是完全等价的优化目标。

### 3. HiFi-GAN 的核心挑战是训练稳定性

HiFi-GAN 工程上更常用，但也最容易踩坑。常见问题如下：

| 常见坑 | 现象 | 原因 | 对策 |
| --- | --- | --- | --- |
| 只用对抗损失 | 毛刺多、内容飘 | 判别信号过粗 | 加 Mel/STFT 重建损失 |
| 不加特征匹配 | 训练振荡、音色漂移 | 生成器过度追逐真假分数 | 加 FM loss 稳定梯度 |
| 没有 MPD | 元音空、周期感弱 | 周期结构监督不足 | 引入多周期判别器 |
| 上采样倍率不匹配 | 时长不对、对齐错误 | 总倍率不等于 `hop_size` | 训练前先验算长度 |
| 判别器过强 | 早期崩溃、爆音 | G/D 能力不平衡 | 调整学习率、warmup、loss 权重 |
| 只看训练集短句 | 长句推理时断裂 | 分布覆盖不足 | 加长样本与连续语流验证 |
| 采样率切换后不重配参数 | 高频异常、嘶声 | STFT/Mel 参数不一致 | 成套更新预处理与模型配置 |

新手最该优先检查的是长度对齐。设采样率为 `sr`，帧移为 `hop_size`，Mel 帧数为 $T$，则目标波形长度近似为：

$$
N \approx T \times \text{hop\_size}
$$

如果你的生成器总上采样倍率不是 `hop_size`，那么系统会出现：

- 输出语速异常
- 音节边界漂移
- 波形长度与目标不一致
- 训练损失下降但主观听感始终差

第二个最常见误区是“并行生成一定更差”。这是早期经验，不是定律。HiFi-GAN 证明了只要判别器和损失设计抓住语音统计结构，并行生成同样可以达到很高的自然度。

### 4. 实时系统里真正要看的不是单个模型指标

部署时最好同时看下面几项：

| 指标 | 含义 |
| --- | --- |
| RTF，Real-Time Factor | 生成 1 秒音频需要多少秒 |
| 峰值显存/内存 | 是否能放进目标设备 |
| 首包延迟 | 在线语音系统是否卡顿 |
| 长音频稳定性 | 句子变长后是否退化 |
| 采样率兼容性 | 16k、22.05k、24k、44.1k 是否需要重训 |

如果目标是实时语音助手，“能跑起来且稳定”通常比“离线听起来略好一点”更重要。

---

## 替代方案与适用边界

三类方法的适用边界并不完全重叠，应该按场景选，而不是按论文年代选。

### 1. 什么时候仍然考虑 WaveNet

如果目标是研究自回归语音建模，或者你需要一个高质量、机制清楚的强基线，WaveNet 仍然有价值。它最适合回答的问题是：

- 逐采样点条件建模能做到什么程度
- 长感受野对波形质量有多大影响
- 自回归路径为何天然昂贵

它不适合对低延迟有严格要求的在线系统。

### 2. 什么时候考虑 WaveGlow

如果场景非常强调并行生成，并且团队希望训练目标保持显式概率解释，WaveGlow 仍然是合理路线。它通常适用于：

- 需要较高吞吐的离线合成
- 想研究 flow-based vocoder
- 愿意接受部分听感折中来换取并行采样

### 3. 什么时候优先 HiFi-GAN

如果场景是在线 TTS、语音助手、播报系统或对话系统，HiFi-GAN 往往是更实用的选择。原因不在于它每个指标都绝对第一，而在于它在以下三点上最平衡：

- 并行生成
- 主观音质强
- 工程实现成熟

| 架构 | 延迟 | 并行度 | 建议部署场景 |
| --- | --- | --- | --- |
| WaveNet | 高 | 低 | 研究基线、离线高质量实验 |
| WaveGlow | 中低 | 高 | 并行生成优先、允许一定质量折中 |
| HiFi-GAN | 低 | 高 | 实时 TTS、语音助手、在线对话系统 |

### 4. 本文不覆盖的路线

这里还要明确本文的适用范围。本文讨论的是“Mel 到波形”的经典神经声码器路线，不覆盖以下内容：

| 路线 | 为什么不在本文主线里 |
| --- | --- |
| 扩散式 vocoder | 建模思路和训练目标已明显不同 |
| 端到端文本直接生成波形 | 问题边界不再是独立声码器 |
| codec language model | 表示空间从 Mel 转向离散音频 token |
| 大模型语音系统 | 通常是多模块联合建模，不再是传统 TTS pipeline |

因此，如果任务是经典 TTS pipeline，WaveNet、WaveGlow、HiFi-GAN 足以构成主线；如果任务是最新语音大模型系统，则还必须把扩散模型、神经编码器和离散音频建模一起纳入比较。

---

## 参考资料

| 资料 | 覆盖主题 | 链接文字描述 | 日期/版本 |
| --- | --- | --- | --- |
| WaveNet: A Generative Model for Raw Audio | 因果膨胀卷积、自回归波形建模 | 论文：DeepMind WaveNet，arXiv:1609.03499 | 2016 |
| NVIDIA OpenSeq2Seq WaveNet 文档 | 条件 WaveNet、工程实现背景 | 文档：NVIDIA OpenSeq2Seq WaveNet | 文档页，访问于 2026-03 |
| WaveGlow: A Flow-based Generative Network for Speech Synthesis | Flow、可逆变换、并行生成 | 论文：NVIDIA WaveGlow，arXiv:1811.00002 | 2018 |
| NVIDIA/waveglow | WaveGlow 官方实现、配置与样例 | GitHub：NVIDIA WaveGlow 项目页 | 仓库页，访问于 2026-03 |
| HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | 生成器、MSD、MPD、FM loss | 论文：HiFi-GAN，arXiv:2010.05646 | 2020 |
| jik876/hifi-gan | HiFi-GAN 官方实现、训练配置、推理脚本 | GitHub：HiFi-GAN 官方仓库 | 仓库页，访问于 2026-03 |
| NVIDIA Riva TTS 文档 | Tacotron2 与 HiFi-GAN/WaveGlow 的工程集成 | 文档：NVIDIA Riva TTS 模型说明 | 文档页，访问于 2026-03 |
| Parallel WaveGAN 论文 | GAN 声码器的并行生成对照路线 | 论文：Parallel WaveGAN，arXiv:1910.11480 | 2019 |

推荐阅读顺序如下：

1. 先读 WaveNet，理解为什么语音波形建模需要长感受野，以及自回归为什么慢。
2. 再读 WaveGlow，理解如何把“逐点采样”改写成“可逆并行生成”。
3. 最后读 HiFi-GAN，重点看多周期判别器、特征匹配损失和上采样结构，因为这些最直接决定今天工程里的主流实践。
