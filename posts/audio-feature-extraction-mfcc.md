## 核心结论

MFCC 和 Mel 频谱都不是“直接看波形”，而是先把音频切成很多短时间片，再观察每一片的频率分布。短时傅里叶变换（STFT，意思是“对短时间片做频谱分析”）负责把波形变成时频图；Mel 滤波器组负责把线性频率压到更接近人耳听感的尺度；MFCC 则是在 Log-Mel 的基础上再做一次 DCT 压缩。

两者的关系可以直接写成一条分支链路：

| 步骤 | Log-Mel 频谱 | MFCC |
|---|---|---|
| 分帧 + 加窗 + FFT | 保留 | 保留 |
| Mel 滤波器组 | 保留 | 保留 |
| 对数 | 保留 | 保留 |
| DCT | 不做 | 做 |
| 输出形状 | 时间帧 × Mel 频带 | 时间帧 × 倒谱系数 |

工程上可以把它们看成两类目标：

| 目标 | 更常用特征 | 原因 |
|---|---|---|
| 传统语音识别、轻量分类 | MFCC | 维度低，冗余少，GMM/HMM/SVM 一类模型更容易处理 |
| CNN、Transformer、现代深度学习 | Log-Mel | 保留二维时间-频率结构，卷积更容易学习局部模式 |

一个最常见的入门配置是：16 kHz 语音，25 ms 帧长，10 ms hop，512 点 FFT，26 或 40 个 Mel 滤波器。这样可以直接得到 26/40 维 Log-Mel，或者再取前 13 个 MFCC 系数作为更紧凑的表示。

结论先给清楚：如果你在做现代深度学习音频分类，默认先用 Log-Mel；如果你在做传统机器学习、边缘设备推理，或者希望特征更紧凑，MFCC 仍然非常实用。

---

## 问题定义与边界

音频是非平稳信号，意思是“它的统计性质会随时间变化”。一句话中的元音、辅音、停顿、爆破声，都不可能用一个全局频谱说清楚。所以第一步不是直接对整段音频做 FFT，而是先分帧。每一帧足够短时，可以近似认为“局部稳定”，再做频域分析。

典型配置如下：

| 参数 | 常见取值 | 作用 |
|---|---|---|
| 采样率 | 16 kHz / 22.05 kHz / 44.1 kHz | 决定可分析的最高频率 |
| 帧长 | 20~30 ms | 平衡时间分辨率与频率分辨率 |
| hop | 10 ms 左右 | 控制相邻帧重叠程度 |
| 窗函数 | Hamming / Hann | 降低频谱泄漏 |
| FFT 点数 | 512 / 1024 | 决定频率采样密度 |
| Mel 滤波器数 | 20~40，深度学习常见 64/80 | 控制频带划分细度 |
| MFCC 维度 | 12~13，或 20 | 控制压缩后的特征维度 |

这里有两个边界要先说清楚。

第一，MFCC 和 Log-Mel 都是“短时谱特征”，主要描述局部频谱包络，不直接保留原始相位。相位可以理解为“各频率成分在时间上的对齐关系”。对分类任务，很多时候相位不是第一优先级；但对高保真重建、声源分离、语音增强，相位就很重要，单靠 MFCC 不够。

第二，Mel 尺度是“以人耳感知为目标的频率压缩”。它更重视低频的分辨率，高频更粗。这个设计非常适合语音、情绪、说话人等听感相关任务，但不代表对所有音频都最优。比如超声、机械振动、特定传感器信号，Mel 压缩可能会丢掉你真正关心的频率细节。

玩具例子可以这样理解：你拿到 1 秒、16 kHz 的语音，一共 16000 个采样点。系统不会把这 16000 个点直接送进分类器，而是切成很多个 400 点的小片段（25 ms），每次滑动 160 点（10 ms）。每个小片段先乘窗，再做 FFT，再映射到 26 个 Mel 频带。得到的是“每个时间片上，各感知频带有多强”。

---

## 核心机制与推导

完整流程可以写成：

$$
x[n] \rightarrow \text{Framing} \rightarrow x_m[n] \rightarrow x_m[n]w[n] \rightarrow X_m[k] \rightarrow E_m[j] \rightarrow \log(E_m[j]+\epsilon) \rightarrow \text{DCT}
$$

### 1. 分帧与加窗

第 $m$ 帧的信号通常写成：

$$
x_m[n] = x[n+mH]w[n], \quad 0 \le n < N
$$

其中：

- $N$ 是帧长
- $H$ 是 hop size，也叫帧移
- $w[n]$ 是窗函数

窗函数的白话解释是“把一帧两端逐渐压低，避免硬切片带来的频谱污染”。如果直接矩形截断，边界会非常突兀，频谱泄漏更严重。常见的 Hamming、Hann 都是在做这个事。

为什么通常要重叠？因为窗函数在边界处接近 0，如果帧与帧完全不重叠，被窗压掉的信息就更容易损失。很多场景会用 50% 左右重叠，Hann 窗配 50% overlap 是很常见的工程设置。

### 2. STFT 与功率谱

每帧做 FFT 后得到：

$$
X_m[k] = \sum_{n=0}^{N-1} x_m[n] e^{-j2\pi kn/N}
$$

通常我们更关心功率谱：

$$
P_m[k] = |X_m[k]|^2
$$

白话解释：这一步把“时间里的起伏”换成“这一帧里有哪些频率、强度多大”。

### 3. Mel 频率映射

人耳对频率的感知不是线性的。低频变化更敏感，高频变化更钝。Mel 尺度用下面这个经验公式表示：

$$
f_{\text{mel}} = 2595\log_{10}\left(1+\frac{f}{700}\right)
$$

这意味着：从 200 Hz 到 400 Hz 的变化，在听感上可能比从 6200 Hz 到 6400 Hz 更显著。Mel 滤波器组就是基于这个思想，在 Mel 轴上等间距布置一组三角滤波器，再映射回线性频率轴。

### 4. Mel 滤波器能量

第 $j$ 个 Mel 滤波器的输出能量写成：

$$
E_m[j] = \sum_k P_m[k]H_j[k]
$$

其中 $H_j[k]$ 是第 $j$ 个三角滤波器。白话解释：不是逐个 FFT bin 使用，而是把相邻频率“归并”到一个感知频带里。

举一个玩具例子。假设第 5 个 Mel 滤波器覆盖 400 Hz 到 800 Hz，中心在 600 Hz。落在这个范围内的 FFT 频点会按照三角形权重参与求和，越靠近中心权重越大。这样得到的 $E_m[5]$ 就不是某一个频率点的强度，而是“这一整段感知频带的能量”。

### 5. 取对数得到 Log-Mel

得到滤波器能量后，通常要做对数压缩：

$$
L_m[j] = \log(E_m[j] + \epsilon)
$$

其中 $\epsilon$ 是一个很小的正数，比如 $10^{-10}$。原因很直接：如果某个频带能量为 0，$\log(0)$ 会变成无穷小，数值会炸掉。对数还有第二个好处，它能把巨大的动态范围压缩下来，让模型更容易学习。

Log-Mel 频谱就是 $\{L_m[j]\}$ 本身。它保持了二维结构：横轴是时间帧，纵轴是 Mel 频带。这正是 CNN 喜欢的输入形式。

### 6. 再做 DCT 得到 MFCC

MFCC 会继续对 Log-Mel 做离散余弦变换（DCT，意思是“用一组余弦基做线性压缩”）：

$$
c_m[\ell] = \sum_{j=1}^{J} L_m[j]\cos\left(\frac{\pi \ell}{J}\left(j-\frac{1}{2}\right)\right)
$$

其中 $J$ 是 Mel 滤波器个数，$\ell$ 是第几个倒谱系数。

DCT 的作用不是“再提取一次频谱”，而是把相邻 Mel 频带之间的相关性压缩掉，让信息集中到前几个系数。于是工程上通常只保留前 12 或 13 维。白话解释：MFCC 更像“把谱包络做了一个低维摘要”。

---

## 代码实现

下面用纯 `numpy` 写一个可运行的最小版本，演示从 1 秒合成音频提取 Log-Mel 和 MFCC。代码里没有依赖 `librosa`，目的是把每一步都摊开。

```python
import numpy as np

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)

def frame_signal(x, frame_length, hop_length):
    num_frames = 1 + (len(x) - frame_length) // hop_length
    frames = np.stack([
        x[i * hop_length:i * hop_length + frame_length]
        for i in range(num_frames)
    ])
    return frames

def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)

    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1

        for k in range(left, center):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (right - k) / (right - center)
    return fb

def dct_type_2(x):
    # x shape: (num_frames, n_mels)
    n = x.shape[1]
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    basis = np.cos(np.pi / n * (i + 0.5) * k)
    return x @ basis.T

# 1 秒玩具信号：440 Hz + 880 Hz
sr = 16000
t = np.arange(sr) / sr
x = 0.8 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)

frame_length = 400   # 25 ms
hop_length = 160     # 10 ms
n_fft = 512
n_mels = 26
n_mfcc = 13
eps = 1e-10

frames = frame_signal(x, frame_length, hop_length)
window = np.hamming(frame_length)
windowed = frames * window[None, :]

spec = np.fft.rfft(windowed, n=n_fft, axis=1)
power = np.abs(spec) ** 2

fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
mel_energy = power @ fb.T
log_mel = np.log(mel_energy + eps)

mfcc = dct_type_2(log_mel)[:, :n_mfcc]

assert log_mel.shape[1] == 26
assert mfcc.shape[1] == 13
assert np.isfinite(log_mel).all()
assert np.isfinite(mfcc).all()

print("log_mel shape:", log_mel.shape)
print("mfcc shape:", mfcc.shape)
```

这段代码体现了两个关键事实：

1. Log-Mel 和 MFCC 的前半段完全一样，差别只在最后是否做 DCT。
2. `assert np.isfinite(...)` 很重要，它保证 `log(0)` 这类问题没有把结果变成 `NaN` 或 `Inf`。

真实工程里更常见的是直接用 `librosa`：

```python
import librosa
import numpy as np

y, sr = librosa.load("example.wav", sr=16000)

mel = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=512,
    win_length=400,
    hop_length=160,
    n_mels=40,
    window="hamming",
    power=2.0
)

log_mel = librosa.power_to_db(mel, ref=np.max)

mfcc = librosa.feature.mfcc(
    S=log_mel,
    n_mfcc=13
)

assert log_mel.shape[0] == 40
assert mfcc.shape[0] == 13
```

真实工程例子：车载语音情绪识别常把 Log-Mel 当作二维“声学热图”输入 CNN。2026 年一篇面向司机语音情绪识别的研究中，Log-Mel 频谱与手工声学特征做融合，在 CASIA 中文情绪语料上报告了 71.22% 的识别准确率。这类结果说明，深度模型往往更愿意直接消费 Log-Mel，而不是先被 DCT 压缩过的 MFCC。

---

## 工程权衡与常见坑

最容易踩坑的不是“公式不会”，而是配置和数值稳定性。

| 问题 | 现象 | 规避方式 |
|---|---|---|
| 不重叠或重叠太少 | 短暂事件容易被切碎，帧间不连续 | 常见用 25 ms 帧长 + 10 ms hop |
| 不加窗 | 频谱泄漏严重，能量扩散 | 用 Hamming 或 Hann |
| `log(0)` | 出现 `-inf`、`NaN` | 用 $\log(E+\epsilon)$ |
| Mel 数太少 | 高频细节丢失 | 语音可 26/40，深度学习常 64/80 |
| 采样率不统一 | 特征维度相同但语义不一致 | 全部重采样到固定 `sr` |
| 把 MFCC 当图像输入 CNN | 局部时频结构弱化 | CNN 优先尝试 Log-Mel |

关于 overlap，可以给一个更具体的玩具例子。假设你有 1024 点窗：

| overlap 比例 | hop | 时间采样密度 | 计算量 |
|---|---|---|---|
| 0% | 1024 | 最稀疏 | 最低 |
| 50% | 512 | 更密 | 中等 |
| 75% | 256 | 更密 | 更高 |

如果完全不重叠，8 个窗口就只给你 8 列谱图；50% overlap 会得到 15 列左右。短促的爆破音、敲击音、辅音过渡更容易被捕捉到。代价是计算量上升。

另一个常见误解是“MFCC 一定比 Log-Mel 高级”。这不对。MFCC 是更强的压缩，不是更强的表达。它适合传统模型，不代表它一定保留了对深度网络最有利的结构信息。

还要注意预加重、均值方差归一化、静音裁剪这些前处理。它们不是 MFCC 的定义组成部分，但会显著影响最终效果。尤其是跨设备、跨录音环境时，如果输入动态范围不统一，Log-Mel 的分布会漂移，训练和推理会对不上。

---

## 替代方案与适用边界

如果把选择问题压缩成一句话，就是：你到底更在乎“低维摘要”，还是“保留二维结构”。

| 表示 | 维度特征 | 结构 | 适用模型 | 适用场景 |
|---|---|---|---|---|
| Mel spectrogram | 较高 | 保留 | CNN/Transformer | 需要完整谱能量，不急于压缩 |
| Log-Mel | 较高 | 保留 | CNN/CRNN/Transformer | 现代音频分类、情绪识别、事件检测 |
| MFCC | 较低 | 弱化 | GMM/HMM/SVM/轻量 MLP | 传统 ASR、说话人识别、边缘设备 |

Mel spectrogram 和 Log-Mel 的差别只有一步对数压缩。对数的价值非常实际：它更接近人耳响度感知，也让极大值不过分主导模型。因此在现代系统里，通常直接跳过“原始 Mel 能量”，优先使用 Log-Mel。

MFCC 的边界也很明确：

- 当模型参数量很小、内存受限、推理芯片弱时，MFCC 很有优势。
- 当任务依赖局部时频纹理，比如环境声分类、心音分类、鸟叫检测、车载情绪识别，Log-Mel 往往更自然。
- 当你还需要动态信息时，可以继续扩展到 $\Delta$ MFCC 和 $\Delta^2$ MFCC，也就是一阶、二阶时间差分特征，用来描述“特征随时间怎么变”。

真实工程里可以这样选：

- 边缘离线唤醒词设备：优先 MFCC。原因是每帧十几维，存储和 CPU 代价都更低。
- 服务器端音频事件检测：优先 64 或 80 维 Log-Mel。原因是二维结构更适合卷积和注意力机制。
- 音乐信息检索：通常先试 Log-Mel，而不是默认 MFCC。因为音乐里的谐波和局部纹理往往值得保留。
- 医学心音、机械故障诊断：Mel 是否合适要谨慎验证，因为人耳感知不一定等于任务最优表示。

---

## 参考资料

- MDPI, *Explainable Instrument Classification: From MFCC Mean-Vector Models to CNNs on MFCC and Mel-Spectrograms with t-SNE and Grad-CAM Insights*, 2025. https://www.mdpi.com/2078-2489/16/10/864
- ScienceDirect, *Fusion of Log-Mel spectrogram and acoustic features for driver speech emotion recognition*, 2026. https://www.sciencedirect.com/science/article/abs/pii/S0957417425046366
- APXML, *Practice: Feature Extraction with Python*. https://apxml.com/courses/applied-speech-recognition/chapter-2-feature-extraction-for-speech/practice-extracting-features
- MDPI, *Area-Efficient Short-Time Fourier Transform Processor for Time-Frequency Analysis of Non-Stationary Signals*, 2020. https://www.mdpi.com/2076-3417/10/20/7208
- ScienceDirect Topics, *Mel Frequency Cepstral Coefficient*. https://www.sciencedirect.com/topics/computer-science/mel-frequency-cepstral-coefficient
- Signal Processing Stack Exchange, *STFT: why overlapping the window?* https://dsp.stackexchange.com/questions/19311/stft-why-overlapping-the-window
- Signal Processing Stack Exchange, *Understanding overlapping in STFT*. https://dsp.stackexchange.com/questions/42428/understanding-overlapping-in-stft
- Cross Validated, *How to avoid log(0) term in regression*. https://stats.stackexchange.com/questions/152958/how-to-avoid-log0-term-in-regression
