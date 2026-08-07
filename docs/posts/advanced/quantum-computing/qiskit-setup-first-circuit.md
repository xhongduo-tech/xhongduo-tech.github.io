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

前十一篇建立了量子计算的全部理论，现在到了「让理论跑起来」的时刻。**Qiskit** 是 IBM 开源的量子软件开发框架，也是当前最流行的 QML/量子算法实验平台。它用 Python 描述量子线路、在模拟器或真实量子处理器上运行——你只需要一台普通电脑就能开始。<span class="marginnote">Qiskit 生态：`qiskit`（核心：线路、门、量子信息）、`qiskit-ibm-runtime`（连接真实硬件/云）、`qiskit-aer`（高性能模拟器）。2023 年起 Qiskit 进入 1.0 稳定版，API 以 `QuantumCircuit`、`Sampler`、`Estimator` 为三大支柱。本节教你从零搭好环境、跑通第一个线路。</span>学完本节，你就有了一台「书桌上的量子计算机」（模拟器）和一条「量子你好世界」的流水线。

## 1 安装与验证

推荐用 Python 3.9+ 的虚拟环境安装：

```bash
python -m venv qiskit-env
source qiskit-env/bin/activate      # Windows: qiskit-env\Scripts\activate
pip install qiskit qiskit-aer
python -c "import qiskit; print(qiskit.__version__)"
```

验证成功的标志：打印出版本号（如 `1.4.0`）。<span class="marginnote">若想连真实硬件，再加 `pip install qiskit-ibm-runtime`，然后在 IBM Quantum 平台注册、用 API token 认证（教程见 IBM Quantum 官方）。模拟器足以完成本篇全部实验——「先用模拟器学对，再上真机」是标准的入门路径。</span>安装遇到问题是初学者的第一道坎：常见有 `pip` 源慢（换清华/阿里镜像）、Python 版本不匹配（Qiskit 1.0 需要 3.9+）。

## 2 第一个线路：单比特叠加与测量

「量子你好世界」= 制备 $\lvert+\rangle$、测量、数统计。完整代码：

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile

# 1. 造一条 1 量子比特、1 经典比特的线路
qc = QuantumCircuit(1, 1)

# 2. 加门：H 门把 |0> 变成叠加态
qc.h(0)

# 3. 测量到经典寄存器
qc.measure(0, 0)

# 4. 模拟器上跑 1024 次
sim = AerSimulator()
qc_t = transpile(qc, sim)
result = sim.run(qc_t, shots=1024).result()
counts = result.get_counts()

# 5. 打印统计
print(counts)   # 约 {'0': 512, '1': 512}
```

- **第 1–2 行**：`QuantumCircuit(1, 1)` 建「1 个量子比特 + 1 个经典比特」的线路；`qc.h(0)` 对第 0 比特施加 Hadamard。
- **第 4–5 行**：`transpile` 把线路编译成模拟器能跑的格式，`AerSimulator` 是本地模拟器，`shots=1024` 表示重复 1024 次。
- **输出**：约一半 0、一半 1——叠加态的测量结果。<span class="marginnote">为什么「一半 0 一半 1」恰好证明叠加？因为若比特是确定的 $\lvert0\rangle$，测量永远是 0；出现均匀的 0/1 分布，说明测量前态是 $\frac{1}{\sqrt2}(\lvert0\rangle+\lvert1\rangle)$（在计算基下测，两种结果各 $\frac12$ 概率）。模拟器「谎言」：真实硬件会有偏差（读出误差），模拟器是理想化的。</span>

## 3 用 draw 看线路图

Qiskit 能直接把线路画出来：

```python
qc.draw('mpl')     # 用 matplotlib 画成图片
# 或纯文本版（无额外依赖）：
print(qc.draw())   # 输出： ┌───┐┌─┐
                   #        q: ┤ H ├┤M├
                   #        c: └─╥┘
                   #           ═╩═
```

`draw('mpl')` 给出教科书风格的线路图，方便检查「门顺序、测量位置」——是调试的第一工具。<span class="marginnote">文本输出里的 `q:` 行是量子线、`c:` 行是经典线，`H` 是 Hadamard 门、`M` 是测量、`╩` 表示经典线汇集。读懂这个文本图，你就能在任何没有图形界面的环境里「看」线路。</span>

## 4 公式解析：线路如何对应量子态

把「线路 = 算子作用」严格对应起来。$\lvert0\rangle$ 经过 $H$ 再测量：

$$
\lvert0\rangle \xrightarrow{H} \frac{1}{\sqrt2}(\lvert0\rangle+\lvert1\rangle) \xrightarrow{\text{measure}} \begin{cases} 0 & \text{概率 } \frac12 \\ 1 & \text{概率 } \frac12 \end{cases}
$$

- **第一步，态演化**：$H\lvert0\rangle = \lvert+\rangle = \frac{1}{\sqrt2}(\lvert0\rangle+\lvert1\rangle)$（第三篇《Hadamard 门》）。
- **第二步，测量概率**：计算基下测 $\lvert+\rangle$，$p(0) = \lvert\langle0\rvert+\rangle\rvert^2 = \frac12$，$p(1) = \frac12$。
- **第三步，统计验证**：1024 次测量 ≈ 512/512——大数定律让实验频率逼近理论概率。<span class="marginnote">这个「理论概率 ↔ 实验频率」的对应是量子实验的基本功：<strong>单次测量随机、多次测量收敛于概率</strong>。模拟器的 counts 就是「掷 1024 次骰子」的记录，与理论分布比对，是验证任何量子程序的第一步。</span>

## 5 常见错误与调试

- **忘写 `measure`**：模拟器跑完没有经典输出，`get_counts` 报错或为空。
- **没 `transpile`**：`run` 有时自动转，但显式 `transpile(qc, backend)` 更安全。
- **索引越界**：`qc.h(1)` 在 1 比特线路上报错——比特从 0 编号。
- **shots 太小**：`shots=10` 时统计噪声巨大，看不出 $1/2$ 分布。<span class="marginnote">调试心法：<strong>先 `print(qc.draw())` 看线路，再跑小 shots 看输出，最后放大 shots 验证分布</strong>。量子编程的错误多半在「门顺序、测量位置、比特编号」，线路图能一眼揪出大半。</span>

## 6 小结

- 安装：`pip install qiskit qiskit-aer`，虚拟环境隔离依赖。
- 首个线路：`QuantumCircuit` → `h` → `measure` → `AerSimulator` → `get_counts`。
- `draw()` 是调试第一工具；理论概率 ↔ 实验频率靠 shots 与大数定律。
- 常见坑：忘测量、索引越界、shots 太小、漏 transpile。

在下一节，我们把「单比特门」放到布洛赫球上——**在 Qiskit 中实现布洛赫球上的单比特门**。
