---
title: Qiskit 环境搭建与第一个量子线路
date: 2026-08-07
---

# Qiskit 环境搭建与第一个量子线路

<div class="epigraph">
<p>只有亲手在模拟器上跑通一个量子程序，你才算真正理解了量子计算。</p>
<footer>—— IBM Quantum 团队（代拟）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Qiskit 官方文档（docs.quantum.ibm.com）｜ 2026-08-07</p>
</div>

## 为什么从 Qiskit 环境开始

前十一篇建立了量子计算的全部理论，现在到了「让理论跑起来」的时刻。**Qiskit** 是 IBM 开源的量子软件开发框架，也是当前最流行的 QML/量子算法实验平台。它用 Python 描述量子线路、在模拟器或真实量子处理器上运行——你只需要一台普通电脑就能开始。<span class="marginnote">Qiskit 生态：`qiskit`（核心：线路、门、量子信息）、`qiskit-ibm-runtime`（连接真实硬件/云）、`qiskit-aer`（高性能模拟器）。2023 年起 Qiskit 进入 1.0 稳定版，API 以 `QuantumCircuit`、`Primitive`（`Sampler`/`Estimator`）、`Backend`/`Session` 为三大支柱。本节教你从零搭好环境、跑通第一个线路。</span>学完本节，你就有了一台「书桌上的量子计算机」（模拟器）和一条「量子你好世界」的流水线。

为什么要「先模拟器、后真机」值得说清：模拟器是理想化的（无噪声、完全确定），适合学算法逻辑；真机有读出误差、退相干、有限连接，适合学「真实硬件约束」。本篇（第十二篇）的全部实验都能在模拟器完成，真机部分在《模拟器与真实后端》一节专门讲。<span class="marginnote">本节对应 Qiskit 官方教程的 "Getting Started" 一章，也是《量子线路模型》《Hadamard 门》等理论篇的第一次代码落地——把「线路图记号」翻译成 `QuantumCircuit` 的 API 调用。</span>

## 1 安装与验证

推荐用 Python 3.9+ 的虚拟环境安装：

```bash
python -m venv qiskit-env
source qiskit-env/bin/activate
pip install qiskit qiskit-aer
```

验证成功的标志：打印出版本号（如 `1.0.0`）。<span class="marginnote">若想连真实硬件，再加 `qiskit-ibm-runtime`，然后在 IBM Quantum 平台注册、用 API token 认证（教程见 IBM Quantum 官方）。模拟器足以完成本篇全部实验——「先用模拟器学对，再上真机」是标准的入门路径。</span>安装遇到问题是初学者的第一道坎：常见有 pip 源慢（换清华/阿里镜像）、Python 版本不匹配（Qiskit 1.0 需要 3.9+）。

安装失败的三种典型场景与对策：

| 问题 | 对策 |
| --- | --- |
| pip 下载慢 / 超时 | 换镜像：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| Python 版本太旧 | 升级到 3.9+ 再建虚拟环境 |
| 已装但 import 失败 | 确认虚拟环境已激活、`pip list` 里能看到 qiskit |

## 2 第一个线路：单比特叠加与测量

「量子你好世界」= 制备 $\lvert+\rangle$、测量、数统计。完整代码：

```python
qc = QuantumCircuit(1, 1)                          # 1 个量子比特 + 1 个经典比特
qc.h(0)                                            # Hadamard：制造叠加
qc.measure(0, 0)                                   # 测量
counts = AerSimulator().run(qc, shots=1024).result().get_counts()
print(counts)                                      # 约 {'0': 512, '1': 512}
```

**第 1–2 行**：`QuantumCircuit(1, 1)` 建「1 个量子比特 + 1 个经典比特」的线路；`qc.h(0)` 对第 0 比特施加 Hadamard。
**第 4–5 行**：`AerSimulator().run(qc, shots=1024)` 把线路编译成模拟器能跑的格式，`AerSimulator` 是本地模拟器，`shots=1024` 表示重复 1024 次。
**输出**：约一半 0、一半 1——叠加态的测量结果。<span class="marginnote">为什么「一半 0 一半 1」恰好证明叠加？因为若比特是确定的 $\lvert0\rangle$，测量永远是 0；出现均匀的 0/1 分布，说明测量前态是 $\frac{1}{\sqrt2}(\lvert0\rangle+\lvert1\rangle)$（在计算基下测，两种结果各 $\frac12$ 概率）。模拟器「谎言」：真实硬件会有偏差（读出误差），模拟器是理想化的。</span>

这段代码里四个关键参数的含义：

| 参数 | 含义 | 例 |
| --- | --- | --- |
| `QuantumCircuit(1, 1)` | 量子比特数, 经典比特数 | 1 个 q + 1 个 c |
| `qc.h(0)` | 对第 0 号量子比特施加 H | $\lvert0\rangle \to \lvert+\rangle$ |
| `shots=1024` | 测量重复次数 | 越大越贴近理论概率 |
| `get_counts()` | 取测量结果的 0/1 计数 | `{'0': 512, '1': 512}` |

别忘了开头的 import——代码能跑通，前两行是关键：

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
```

缺失 import 是最常见的「神秘报错」来源（`NameError: name 'QuantumCircuit' is not defined`）。

## 3 用 draw 看线路图

Qiskit 能直接把线路画出来：

```python
print(qc.draw())     # 输出教科书风格的文本线路图
```

`qc.draw()` 给出教科书风格的线路图，方便检查「门顺序、测量位置」——是调试的第一工具。<span class="marginnote">文本输出里的 `─` 行是量子线、`═` 行是经典线，`┤ H ├` 是 Hadamard 门、`┤M├` 是测量、`╩` 表示经典线汇集。读懂这个文本图，你就能在任何没有图形界面的环境里「看」线路。</span>

## 4 公式解析：线路如何对应量子态

把「线路 = 算子作用」严格对应起来。$\lvert0\rangle$ 经过 $H$ 再测量：

$$
\lvert0\rangle \xrightarrow{H} \frac{1}{\sqrt2}(\lvert0\rangle+\lvert1\rangle) \xrightarrow{\text{measure}} \begin{cases} 0 & \text{概率 } \frac12 \\ 1 & \text{概率 } \frac12 \end{cases}
$$

- **第一步，态演化**：$H\lvert0\rangle = \lvert+\rangle = \frac{1}{\sqrt2}(\lvert0\rangle+\lvert1\rangle)$（第三篇《Hadamard 门》）。
- **第二步，测量概率**：计算基下测 $\lvert+\rangle$，$p(0) = \lvert\langle0\rvert+\rangle\rvert^2 = \frac12$，$p(1) = \frac12$。
- **第三步，统计验证**：1024 次测量 ≈ 512/512——大数定律让实验频率逼近理论概率。<span class="marginnote">这个「理论概率 ↔ 实验频率」的对应是量子实验的基本功：<strong>单次测量随机、多次测量收敛于概率</strong>。模拟器的 counts 就是「掷 1024 次骰子」的记录，与理论分布比对，是验证任何量子程序的第一步。

shots 应该取多大？测量 $N$ 次，频率与理论的偏差约 $\lvert \hat p - p \rvert \sim 1/\sqrt{N}$：$N = 1024$ 时误差约 $3\%$，$N = 10^4$ 时约 $1\%$。这就是「模拟器常用 1024、真机实验多用上万 shots」的原因。<span class="marginnote">这条「$1/\sqrt{N}$」的统计规律贯穿全部量子实验：估计任意概率都要面对它，与《量子查询复杂度》里「采样次数下界」的思想同源——想知道概率，就得花采样预算。</span></span>

## 5 常见错误与调试

**忘写 `measure`**：模拟器跑完没有经典输出，`get_counts()` 报错或为空。
**没 `transpile`**：`AerSimulator().run()` 有时自动转，但显式 `transpile()` 更安全。
**索引越界**：`qc.h(1)` 在 1 比特线路上报错——比特从 0 编号。
**shots 太小**：`shots=10` 时统计噪声巨大，看不出 $1/2$ 分布。<span class="marginnote">调试心法：<strong>先 `draw()` 看线路，再跑小 shots 看输出，最后放大 shots 验证分布</strong>。量子编程的错误多半在「门顺序、测量位置、比特编号」，线路图能一眼揪出大半。</span>

把常见错误做成速查表，排查时逐行对照：

| 症状 | 原因 | 修法 |
| --- | --- | --- |
| `get_counts()` 为空 | 忘了 `measure` | 补 `qc.measure(...)` |
| `IndexError` | 比特编号越界 | 从 0 开始编号 |
| 分布明显偏 0 或 1 | shots 太小 / 噪声 | 加大 shots |
| 结果与理论不符 | 门顺序 / 测量位置错 | 先 `qc.draw()` 检查 |

## 6 小结

- 安装：`pip install qiskit qiskit-aer`，虚拟环境隔离依赖。
- 首个线路：`QuantumCircuit` → `h` → `measure` → `run` → `get_counts`。
- `draw()` 是调试第一工具；理论概率 ↔ 实验频率靠 shots 与大数定律。
- 常见坑：忘测量、索引越界、shots 太小、漏 transpile。
- 进阶路线：`transpile` 优化 → 噪声模拟 → 真机（`qiskit-ibm-runtime`）。

在下一节，我们把「单比特门」放到布洛赫球上——**在 Qiskit 中实现布洛赫球上的单比特门**。
