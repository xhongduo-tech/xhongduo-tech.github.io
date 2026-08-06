---
title: HMM 三大问题：评估、解码与学习
date: 2026-08-07
---

# HMM 三大问题：评估、解码与学习

<div class="epigraph">
<p>所有模型都是错的，但其中一些是有用的。</p>
<footer>—— 乔治 · 博克斯（George E. P. Box）</footer>
</div>

<div class="article-byline">
<p>第四级 · 语音技术 ｜ 《语音信号处理》第9章 隐马尔可夫模型 ｜ 2026-08-07</p>
</div>

## 为什么从三大问题开始

上一节我们定义了三要素 $\lambda = (\mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$——模型**长什么样**有了。但一个模型在手，你要会**用它**。HMM 的全部用法收敛为三个经典问题，它们恰好对应语音识别的三个动作：

1. **评估（Evaluation）**：给定 $\lambda$ 与观测 $O$，求 $P(O \mid \lambda)$——「这句话有多像这个模型」。
2. **解码（Decoding）**：给定 $\lambda$ 与 $O$，求最可能的状态序列——「这句话对应哪串音素」。
3. **学习（Learning）**：给定一批观测，估出 $\lambda$——「从数据里把模型训练出来」。

这三个问题不是考试题，而是 ASR 的骨架：评估用于**打分**、解码用于**对齐与识别**、学习用于**训练**。它们共用的数学工具是**动态规划**——把指数级问题化成一阶递推。<span class="marginnote">三大问题的划分出自 Rabiner 1989 年那篇经典综述 "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition"——那是 HMM 在语音领域被引用最多的文献，至今仍是理解 ASR 序列建模的必读材料。</span>

## 1 问题一：评估——前向算法

**目标**：算 $P(O \mid \lambda)$。上一节说过，直接对所有 $N^T$ 条路径求和不可行。解决办法是**前向变量（forward variable）**：

**前向变量 $\alpha_t(j)$**：到时刻 $t$ 为止、当前停在状态 $s_j$ 且已观测到 $o_1, \dots, o_t$ 的联合概率：

$$\alpha_t(j) = P(o_1, \dots, o_t,\; q_t = s_j \mid \lambda)$$

它有一个漂亮的**一阶递推**——$\alpha_{t+1}(j)$ 可以由所有 $\alpha_t(i)$ 算出来：

$$\alpha_{t+1}(j) = \Bigl[\sum_{i=1}^{N} \alpha_t(i)\, a_{ij}\Bigr]\, b_j(o_{t+1})$$

初始：$\alpha_1(j) = \pi_j\, b_j(o_1)$。最终：$P(O \mid \lambda) = \sum_{j=1}^{N} \alpha_T(j)$。

**辨析｜易错点：求和还是取最大？** 前向算法算的是「**所有路径的加和**」，维特比（下一节）算的是「**最优一条路径**」。前者回答「观测的整体概率」，后者回答「最可能的状态序列」——一字之差，用途完全不同，千万别混。

## 2 公式解析：前向递推为什么是对的

把递推式拆开看，理解它为什么能从「算 $N^T$ 条路径」变成「算 $T$ 步、每步 $O(N^2)$」：

$$\alpha_{t+1}(j) = \Bigl[\sum_{i=1}^{N} \alpha_t(i)\, a_{ij}\Bigr]\, b_j(o_{t+1})$$

- **第一步，看方括号**：$\sum_i \alpha_t(i)\, a_{ij}$ 是「从所有可能的上一状态 $s_i$ 转移到 $s_j$ 的概率之和」。$\alpha_t(i)$ 已经包含了到 $t$ 为止的所有历史，乘 $a_{ij}$ 就是「延长一步走到 $s_j$」——**历史被折叠进 $\alpha_t(i)$ 里了**，不必再展开。
- **第二步，看乘号**：到达 $s_j$ 之后还要产生观测 $o_{t+1}$，所以乘发射概率 $b_j(o_{t+1})$。观测只由当前状态决定（观测独立性假设），因此只乘这一步。
- **第三步，看复杂度**：每个 $t$ 要算 $N$ 个 $j$，每个 $j$ 要遍历 $N$ 个 $i$，共 $O(T N^2)$——相比 $O(N^T)$ 是指数级下降。**这就是动态规划「用中间结果换时间」的典型胜利。**
- **第四步，看数值规模**：$T = 100$、$N = 50$ 时，$N^T = 50^{100}$ 大得无法想象，而 $T N^2 = 25$ 万次乘法——前向算法让 HMM 真正可算。

一个前向算法的 Python 实现：

```python
import numpy as np

def forward(A, B, pi):
    """A: (N,N) 转移矩阵, B: (T,N) 每帧每状态的发射概率, pi:(N,) 初始分布
    返回 P(O|lambda)（B 已由发射分布按观测序列算出）"""
    T, N = B.shape
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[0]
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[t]   # 先转移求和，再乘发射
    return alpha.sum(axis=1)                  # P(O|lambda) = sum alpha_T

logP = np.log(forward(A, B, pi).sum())
```

## 3 问题二：解码——维特比算法

**目标**：给定 $O$ 与 $\lambda$，找最可能的状态序列 $Q^* = \arg\max_Q P(Q \mid O, \lambda)$。等价于最大化 $P(Q, O \mid \lambda)$（分母 $P(O \mid \lambda)$ 与 $Q$ 无关）。

把前向递推里的**求和换成取最大**，就得到**维特比变量（Viterbi variable）**：

$$\delta_{t}(j) = \max_{q_1, \dots, q_{t-1}} P(q_1, \dots, q_{t-1}, q_t = s_j, o_1, \dots, o_t \mid \lambda)$$

递推式：

$$\delta_{t}(j) = \Bigl[\max_{i}\, \delta_{t-1}(i)\, a_{ij}\Bigr]\, b_j(o_t)$$

与前面唯一的差别是 $\max$ 替代 $\sum$。同时维护一个**回溯指针（backpointer）** $\psi_t(j) = \arg\max_i \delta_{t-1}(i) a_{ij}$，记下每一步的最优来源；到终点取 $\max_j \delta_T(j)$ 后，沿指针**从后往前回溯**，还原整条最优状态路径。

工程上有三个关键细节：<span class="marginnote">为什么强调工程细节？因为 HMM 在语音上的真实规模是 $T\sim$几百帧、$N\sim$几千状态（三音子模型），任何数值上的不稳健都会直接放大成识别错误。</span>

- **对数域**：概率连乘会下溢（underflow），全程用 $\log$，乘法变加法：$\delta_t(j) = \max_i\bigl(\delta_{t-1}(i) + \log a_{ij}\bigr) + \log b_j(o_t)$。
- **束搜索（beam search）**：每步只保留前 $K$ 大的 $\delta_t(j)$，把复杂度从 $O(TN^2)$ 降到约 $O(TNK)$——语音识别解码的标配。
- **与语言模型结合**：真正的识别还要把语言模型分数并进来，见《加权有限状态转换器（WFST）解码》与《语言模型融合》各节。

维特比在 ASR 里的两个关键用途：**训练时**做强制对齐（把帧对齐到音素，供嵌入式训练使用）；**识别时**给出最优音素/词路径。

## 4 问题三：学习——Baum-Welch 算法

**目标**：给一批观测序列 $\{O^{(1)}, \dots, O^{(K)}\}$，估 $\lambda$ 使 $P(O \mid \lambda)$ 最大。HMM 的状态序列看不见，属于**带隐变量的最大似然估计**——标准解法是 **EM 算法**，在 HMM 里叫 **Baum-Welch 算法**。<span class="marginnote">EM 是带隐变量估计的通用框架，不只在 HMM 里出现——高斯混合模型、潜变量模型的训练都用它，可对照《机器学习》专题对 EM 的统一推导。</span>

EM 的套路是「猜隐变量 → 估参数 → 再猜 → 再估」：

- **E 步**：用当前的 $\lambda$ 估计隐变量的后验——引入两个量。**后向变量** $\beta_t(i) = P(o_{t+1}, \dots, o_T \mid q_t = s_i, \lambda)$ 与前向变量对偶。由 $\alpha, \beta$ 组合出：
  - $\gamma_t(i) = P(q_t = s_i \mid O, \lambda)$——时刻 $t$ 处于状态 $s_i$ 的概率；
  - $\xi_t(i,j) = P(q_t = s_i, q_{t+1} = s_j \mid O, \lambda)$——时刻 $t$ 从 $s_i$ 跳到 $s_j$ 的概率。
- **M 步**：用这些「软计数」重估三要素：

$$\hat{\pi}_i = \gamma_1(i), \qquad \hat{a}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}, \qquad \hat{b}_j(v_k) = \frac{\sum_{t:\, o_t = v_k} \gamma_t(j)}{\sum_{t=1}^{T} \gamma_t(j)}$$

直觉：$\hat{a}_{ij}$ 是「从 $i$ 到 $j$ 的期望转移次数」除以「离开 $i$ 的期望总次数」——**把计数换成期望计数**，因为真实计数不可得。

**辨析｜易错点：Baum-Welch 只能保证收敛到局部最优。** EM 目标函数非凸，初值不同结局不同。工程上常用**多随机重启**，或「先 Viterbi 对齐、再按对齐计数」的**嵌入式训练（embedded training）**——这是 GMM-HMM 声学模型训练的通行做法，细节在《三音子模型与决策树状态绑定》一节展开。对连续密度 HMM，发射分布换成 GMM，M 步还要用 GMM 参数重估，见《高斯混合模型（GMM）与连续密度 HMM》。

## 5 三个问题如何组装成一套识别系统

把三大问题串起来，就是传统 GMM-HMM 声学模型的完整生命线：

1. **训练（学习 + 解码）**：初始化 $\lambda$ → Viterbi 强制对齐 → 按对齐统计 $\gamma, \xi$ → Baum-Welch 重估 → 迭代至收敛。帧被不断「重新对齐」，模型被不断「重新拟合」，这就是嵌入式训练。
2. **建模（评估）**：每个词/音素有一个 HMM，打分就是求 $P(O \mid \lambda_{\text{候选}})$；候选按似然排序。
3. **识别（解码）**：在词典与语言模型约束下，对整个词图（lattice）跑束搜索 Viterbi，输出最可能的词串。<span class="marginnote">从 HMM 的三大问题到端到端模型：CTC 的「对齐自由」绕开了「显式枚举状态路径」，注意力机制则让对齐从「硬性的最大/求和」变成「软性的数据驱动」——但「评估—解码—学习」这个三问题框架，作为理解任何序列模型的透镜，始终成立。</span>

## 6 小结

- **三大问题**：评估（$P(O \mid \lambda)$）、解码（最优状态序列）、学习（估三要素）——分别对应打分、对齐/识别、训练。
- **前向算法**：$\alpha_t(j)$ 一阶递推，把 $O(N^T)$ 降为 $O(TN^2)$；求的是**所有路径之和**。
- **维特比算法**：递推里的 $\sum$ 换 $\max$，加回溯指针；求的是**最优单条路径**，工程上用对数域 + 束搜索。
- **Baum-Welch / EM**：E 步算 $\gamma, \xi$ 软计数，M 步重估 $\pi, \mathbf{A}, \mathbf{B}$；只能保证局部最优，需谨慎初始化。
- **三者合起来**就是 GMM-HMM 的训练—建模—识别流水线，也是理解一切端到端序列模型的透镜。

在下一节，我们将深入前向—后向算法的数值细节与维特比的回溯实现，并正式进入连续密度 HMM 与 GMM 的世界——**前向—后向算法与维特比（Viterbi）算法**。
