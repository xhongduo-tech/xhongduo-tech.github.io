---
title: 高斯信源的逆注水（Reverse Water-filling）
date: 2026-08-07
---

# 高斯信源的逆注水（Reverse Water-filling）

<div class="epigraph">
<p>前向注水把功率灌进干净的信道，逆向注水把失真泼给微弱的成分——一正一逆，同一个水池。</p>
<footer>—— 克劳德 · 香农（Claude E. Shannon）</footer>
</div>

<div class="article-byline">
<p>第二级 · 信息论 ｜ Cover &amp; Thomas《Elements of Information Theory》 §10.3 ｜ 2026-08-07</p>
</div>

## 为什么从「把注水倒过来」开始

第 47 篇的注水（water-filling）把功率分配给并联信道：**低噪声信道多分功率**。高斯信源的率失真问题里，出现了它的镜像——**逆注水（reverse water-filling）**：把失真预算分配给并联的独立高斯信源分量。

设信源是一组独立的零均值高斯分量，方差 $\lambda_1, \dots, \lambda_k$（例如：变换编码后的频谱分量、或高斯向量 $\mathbf{K}$ 的特征值）。总失真预算 $D$，怎么分给每个分量最划算？

**逆注水公式**：

$$
D_j = \min(\theta, \lambda_j), \qquad \sum_j D_j = D
$$

**直觉**：给「弱分量」（$\lambda_j$ 小）的失真分配上限是它自己的方差——**小于阈值的分量干脆「全失真」丢弃**（$D_j = \lambda_j$，不花任何比特）；大于阈值的分量「削到阈值」（$D_j = \theta$）。

这一篇推导它、并把它与正向注水对照，看清「功率分配」与「失真分配」的同一水池、两个方向。<span class="marginnote">逆注水在 Cover &amp; Thomas §10.3（Theorem 10.3.3）。它的推导与正向注水完全平行——都是「拉格朗日 + KKT」，只是决策变量从「功率」换成「失真」。名字里的「逆」正是第 47 篇的镜像。</span>

## 1 模型与推导

**设定**：$X_1, \dots, X_k$ 独立，$X_j \sim \mathcal{N}(0, \lambda_j)$。平方误差失真，总失真预算 $D$，总码率

$$
R(D) = \min_{\sum_j D_j \le D} \sum_{j=1}^k \frac12 \log \frac{\lambda_j}{D_j}
$$

（每个分量用单变量高斯率失真公式，总码率 = 逐分量之和。）

**KKT 推导**：对 $D_j$ 求导（$D_j < \lambda_j$ 的分量）：

$$
\frac{d}{dD_j}\left(\frac12\log\frac{\lambda_j}{D_j}\right) = -\frac{1}{2 D_j \ln 2}
$$

拉格朗日 $\sum_j \frac12\log\frac{\lambda_j}{D_j} + \lambda(\sum_j D_j - D)$，求导置零：

$$
-\frac{1}{2D_j\ln 2} + \lambda = 0 \quad\Rightarrow\quad D_j = \frac{1}{2\lambda\ln 2} = \theta \ \text{（常数）}
$$

**加上「$D_j \le \lambda_j$」约束**（失真不能超过方差，否则 $R_j = 0$、$D_j$ 再大无意义）：

$$
D_j = \min(\theta, \lambda_j), \qquad \sum_j \min(\theta, \lambda_j) = D
$$

**这就是逆注水**：失真预算像「水平线 $\theta$」一样切过「方差柱状图」$\lambda_j$——高于水线的分量削到 $\theta$，低于水线的分量完全淹没（$D_j = \lambda_j$）。<span class="marginnote">KKT 推导的关键是「$D_j = \theta$ 常数」：最优时，所有「被编码的分量」分到<strong>相同</strong>的失真预算 $\theta$。这对应「边际码率」均等化——每个分量再省 1 比特失真的代价相同。与正向注水「$P_j + N_j = \nu$」的结构完全平行。</span>

## 2 公式解析：$D_j = \min(\theta, \lambda_j)$ 的几何

把逆注水公式逐项拆开：

- **$\lambda_j$**：第 $j$ 个分量的方差——「柱高」。能量越大的分量越高。
- **$\theta$**：失真水位——由总预算 $\sum \min(\theta, \lambda_j) = D$ 解出的水平线。
- **$D_j$**：第 $j$ 个分量分到的失真——「淹没深度」。
- **$\min(\theta, \lambda_j)$**：失真不超过方差。高于水线的削到 $\theta$；低于水线的（$\lambda_j < \theta$）整个淹没。

**两类分量**：

- $\lambda_j > \theta$：**被编码**的分量，失真 $\theta$，码率 $\frac12\log\frac{\lambda_j}{\theta}$。
- $\lambda_j \le \theta$：**被丢弃**的分量，失真 $\lambda_j$（还原为 0），码率 0。

**直觉**：弱分量（方差 $\le \theta$）的信息量太少，不值得花比特；强分量花比特「削失真」。**总码率**

$$
R(D) = \frac12 \sum_{j: \lambda_j > \theta} \log \frac{\lambda_j}{\theta}
$$

只有「未被淹没」的分量贡献码率。<span class="marginnote">「弱分量直接丢弃」是逆注水最反直觉也最实用的结论：压缩高斯源时，不是「每个分量都保一点」，而是「能量低于阈值的分量整块扔掉」。这正对应图像/音频压缩里的「丢弃小系数」——JPEG 的量化、变换编码的「阈值化」全是逆注水的工程化身。</span>

## 3 正向注水 vs 逆注水：同一水池

把两种注水并排，看它们的镜像结构：

| | 正向注水（功率） | 逆注水（失真） |
| --- | --- | --- |
| 场景 | 并联信道（噪声 $N_j$） | 并联信源（方差 $\lambda_j$） |
| 预算 | 总功率 $P$ | 总失真 $D$ |
| 分配 | $P_j = (\nu - N_j)^+$ | $D_j = \min(\theta, \lambda_j)$ |
| 水位 | $\nu$：功率水位 | $\theta$：失真水位 |
| 丢弃 | 噪声 $> \nu$ 的信道不用 | 方差 $< \theta$ 的分量不编 |
| 目标 | 最大化容量 | 最小化码率 |

**一正一逆，同一座「注水池」**：正向注水在「噪声轮廓」$N_j$ 上注「功率」；逆注水在「方差轮廓」$\lambda_j$ 上注「失真」。一个是「给干净的地方加水」，一个是「给弱小的地方泼脏水」。<span class="marginnote">「泼脏水」的比喻很贴切：逆注水把失真（噪声）预算「泼」给弱分量——它们反正救不回来，不如把失真预算花在它们身上，保住强分量。这种「牺牲弱小保强大」的分配，与人类资源分配里的「重点投入」殊途同归。</span>

**辨析｜易错点：** 三个容易混的地方：

**$\theta$ 与 $\nu$ 的角色**：$\theta$ 是失真水位（越低越好，失真预算小则 $\theta$ 小、丢的分量多）；$\nu$ 是功率水位（越高越好）。方向相反，别记反。
**「丢弃」的含义**：丢弃 = 用 0 还原该分量，失真恰好 $\lambda_j$（即「误差 = 信号本身」）。不是「完全不还原」，而是「还原成 0」。
**独立假设**：逆注水要求分量独立（或正交化后）。相关分量的率失真需要先 K-L 变换解耦——与第 48 篇「白化」是同一个预处理。<span class="marginnote">「还原成 0」的直觉：弱分量的信息量太小，与其花比特保它，不如直接输出 0——代价是「全失真」（误差等于信号方差），但这部分失真被「泼」在弱分量上，几乎不增加总码率。JPEG 里「把小于量化步长的高频系数置零」就是这个操作的实物。</span>

## 4 连续谱版本的逆注水

对功率谱密度 $S_X(f)$ 的连续高斯信源，逆注水变为连续形式：

$$
D(f) = \min(\theta, S_X(f)), \qquad \int D(f)\, df = D
$$

$$
R(D) = \frac12 \int \log \frac{\max(S_X(f), \theta)}{\theta}\, df
$$

**直觉**：信源谱低于水位的频段整段丢弃，高于水位的频段「削到 $\theta$」。这直接对应**变换编码**：把信号分解到频域，低能量频段的系数量化到零，高能量频段精量化——JPEG、MP3、H.264 的全部「心理声学/视觉模型」，本质都是在计算「什么样的 $\theta$ 最合适」。<span class="marginnote">「$\theta$ 的选择」在编码器里就是「量化参数（QP）的选择」：QP 越小，$\theta$ 越小，丢的分量越少、码率越高、失真越小。视频编码的码率控制，就是在「目标码率」下反解 $\theta$——逆注水的工程实现。</span>

**与全课程体系的连接：** 逆注水在图像/视频编码里是「变换编码 + 量化阈值」的理论基础；在《数字信号处理》里对应「谱估计与阈值化」；它也是「压缩感知」「稀疏表示」里「丢弃小系数」行为的信息论解释——「弱分量不配花比特」是一条贯穿压缩世界的铁律。

## 5 小结

- **逆注水**：$D_j = \min(\theta, \lambda_j)$，$\sum D_j = D$——失真预算按「方差轮廓」分配。
- 推导：KKT 给出「被编码分量失真同为 $\theta$」，加约束 $\min(\theta, \lambda_j)$。
- 两类分量：$\lambda_j > \theta$ 被编码（码率 $\frac12\log\frac{\lambda_j}{\theta}$），$\lambda_j \le \theta$ 被丢弃（还原为 0）。
- 总码率：$R(D) = \frac12\sum_{\lambda_j > \theta}\log\frac{\lambda_j}{\theta}$。
- **对照**：正向注水给干净信道加功率，逆注水给弱分量泼失真——同一水池两个方向。
- 连续版：$D(f) = \min(\theta, S_X(f))$，是变换编码、JPEG/MP3 量化阈值的理论基础。

在下一篇，我们把「极限」两个字坐实：**率失真定理：可达性与逆定理**——$R(D)$ 确实是有损压缩的精确极限。
