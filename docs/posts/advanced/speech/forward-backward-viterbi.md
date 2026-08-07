---
title: 前向—后向算法与维特比（Viterbi）算法
date: 2026-08-07
---

# 前向—后向算法与维特比（Viterbi）算法

<div class="epigraph">
<p>一个最优策略具备这样的性质：无论初始状态与初始决策如何，余下的决策对于由第一步决策所导致的状态而言，必然构成一个最优策略。</p>
<footer>—— 理查德 · 贝尔曼（Richard Bellman），动态规划的最优性原理</footer>
</div>

<div class="article-byline">
<p>第四级 · 语音技术 ｜ 《语音信号处理》第9章 隐马尔可夫模型 ｜ 2026-08-07</p>
</div>

## 为什么从「算法怎么落地」开始

上一节我们把 HMM 的三大问题都摆上了桌：评估、解码、学习。它们对应的四个核心算法——前向、后向、维特比——上一节只讲了「是什么」。这一节回答「**怎么算才不出错**」。原因很实在：语音里的 HMM 规模是 $T$ 几百帧、$N$ 上千状态（三音子模型），任何一个数值陷阱都会被放大成识别错误。

前向算法里所有概率都远小于 1，连乘几十步后必然**下溢（underflow）**到浮点数可表示范围以下；维特比要维护的不只是分数，还有一整张**回溯指针表**。把这两个问题解决，HMM 才从「公式上正确」变成「机器上正确」。这条从公式到实现的路，也正是后面所有端到端模型里「对数空间 + 动态规划」这套工程基因的来源。<span class="marginnote">动态规划（dynamic programming，DP）由贝尔曼在 1950 年代系统化，核心就是「子问题的最优解 → 原问题的最优解」。前向/后向/维特比全是 DP 的特例，区别只在「求和还是取最大」——CTC 的对齐、Transformer 里注意力权重的 softmax 归一化，底层都是同一套递推哲学。</span>

## 1 前向算法先算对：对数域与缩放因子

前向变量的递推，上一节已经给出：

$$\alpha_{t+1}(j) = \Bigl[\sum_{i=1}^{N} \alpha_t(i)\, a_{ij}\Bigr]\, b_j(o_{t+1})$$

直接照抄会在工程上翻车。以 $T=200$ 帧、$N=50$ 状态为例，路径概率是 $200$ 项乘法的连乘，每项平均约 $0.05$，$0.05^{200} \approx 10^{-260}$——远低于双精度浮点能表示的最小正数（约 $10^{-308}$，但中间量的下溢在 $10^{-260}$ 附近已把有效数字吃光）。两条对策：

**对策一：对数域。** 全程用 $\log$，乘法变加法。前向递推的对数形式是：

$$\log \alpha_{t+1}(j) = \log\Bigl[\sum_{i=1}^{N} \exp\bigl(\log \alpha_t(i) + \log a_{ij}\bigr)\Bigr] + \log b_j(o_{t+1})$$

方括号里的「对数和」要用 **log-sum-exp** 技巧算，防止单个 $\exp$ 溢出。实践中常用 PyTorch 的 `torch.logsumexp`，这正是 CTC 损失里每天都会用到的同一个函数。

**对策二：缩放（scaling）。** 不换底数空间，而是每步把 $\alpha_t$ 归一化：令缩放因子 $c_t = \sum_{j=1}^{N} \alpha_t(j)$，然后令 $\hat{\alpha}_t(j) = \alpha_t(j) / c_t$，使每步的和为 1。<span class="marginnote">缩放是 Rabiner 1989 那篇综述给出的经典做法。它还有一个副产品：<strong>缩放因子的连乘就等于观测概率</strong>，即 $\prod_t c_t = P(O \mid \lambda)$——评估问题的答案免费送给你了。在对数域实现时，$P(O\mid\lambda)$ 就是对所有 $\log c_t$ 求和，也就是 Python 里 `numpy` 训练 HMM 时的 `log_likelihood` 累加。</span>

**辨析｜易错点：对数域里没有「缩放」，或者说缩放是隐式的。** 对数域用 log-sum-exp 归一化中间量，返回的最终对数似然就是「对所有 log-缩放因子求和」。两种做法殊途同归——选哪种取决于你后面要什么：要 $\gamma, \xi$ 软计数（Baum-Welch 用），缩放法更直接；只要打分，对数域更省事。

## 2 后向算法：前向的镜像

评估问题只需前向。但下一节 Baum-Welch 要算「时刻 $t$ 停在状态 $s_i$」的后验概率，它需要从序列**尾部**回推的信息，于是定义**后向变量（backward variable）**：

**后向变量 $\beta_t(i)$**：在时刻 $t$ 处于状态 $s_i$、且从 $t+1$ 到 $T$ 的观测 $o_{t+1}, \dots, o_T$ 已经出现的概率：

$$\beta_t(i) = P(o_{t+1}, \dots, o_T \mid q_t = s_i, \lambda)$$

边界条件：$\beta_T(i) = 1$——时刻 $T$ 之后没有观测，概率是 1。递推从后往前：

$$\beta_t(i) = \sum_{j=1}^{N} a_{ij}\, b_j(o_{t+1})\, \beta_{t+1}(j)$$

把它念出来：站在状态 $s_i$，下一步转移到 $s_j$（$a_{ij}$）、$s_j$ 发出观测 $o_{t+1}$（$b_j(o_{t+1})$）、再延续剩下的尾部概率（$\beta_{t+1}(j)$）——三件事相乘，对所有可能的下一状态求和。**后向与前向完全镜像，复杂度同为 $O(TN^2)$。**

## 3 公式解析：$\alpha$ 与 $\beta$ 合体成 $\gamma$ 与 $\xi$

前向管「到 $t$ 为止的过去」，后向管「$t$ 之后的未来」。两者相乘，恰好覆盖整条观测序列，就得到两个后验软计数：

$$
\gamma_t(i) = P(q_t = s_i \mid O, \lambda) = \frac{\alpha_t(i)\, \beta_t(i)}{P(O \mid \lambda)}
$$

$$
\xi_t(i,j) = P(q_t = s_i,\, q_{t+1} = s_j \mid O, \lambda) = \frac{\alpha_t(i)\, a_{ij}\, b_j(o_{t+1})\, \beta_{t+1}(j)}{P(O \mid \lambda)}
$$

拆开看：

- **第一步，看 $\gamma$ 的分子**：$\alpha_t(i)$ 与 $\beta_t(i)$ 都以 $q_t = s_i$ 为「锚点」。锚点两侧的观测互不重叠（$o_1..o_t$ 与 $o_{t+1}..o_T$），由观测独立性假设，它们条件独立，所以相乘合法。
- **第二步，看分母**：$P(O \mid \lambda)$ 是归一化常数，把「联合概率」变成「条件概率」。它可以用 $\sum_i \alpha_T(i)$ 求，也可以用缩放因子的连乘积求。
- **第三步，看 $\xi$ 的分子**：$\alpha_t(i)$ 锚定 $s_i$，乘 $a_{ij}$ 完成转移，乘 $b_j(o_{t+1})$ 发出下一观测，再乘 $\beta_{t+1}(j)$ 延续尾部——四件事沿着时间轴排成一串，是联合概率的又一次「沿路径连乘」。
- **第四步，理解软计数**：$\gamma_t(i)$ 之和（对 $t$）就是「状态 $s_i$ 被访问的期望次数」，$\xi_t(i,j)$ 之和就是「从 $i$ 到 $j$ 的期望转移次数」——下一节 Baum-Welch 的 M 步直接消费这两个量。

缩放版本里，只要用 $\hat{\alpha}, \hat{\beta}$ 计算，$\gamma_t(i) = \hat{\alpha}_t(i) \hat{\beta}_t(i)$ 就直接成立，连分母都不用单独写——这是工程实现喜欢缩放法的根本原因。<span class="marginnote">注意 $\gamma$ 与 $\xi$ 的约束：$\sum_j \xi_t(i,j) = \gamma_t(i)$（从 $i$ 离开的总概率），且 $\sum_i \gamma_t(i) = 1$（每个时刻必在某个状态）。这些等式常被用来<strong>校验代码正确性</strong>——如果你的 $\gamma$ 求和不是 1，说明实现有 bug。</span>

## 4 维特比：把 $\sum$ 换成 $\max$

评估与后验用「求和」，解码求「最可能的那一条路径」用「取最大」。维特比变量：

$$\delta_{t}(j) = \max_{q_1,\dots,q_{t-1}} P(q_1,\dots,q_{t-1},\, q_t = s_j,\, o_1,\dots,o_t \mid \lambda)$$

递推：

$$\delta_{t}(j) = \Bigl[\max_{i}\, \delta_{t-1}(i)\, a_{ij}\Bigr]\, b_j(o_t)$$

与前向的唯一差别是 $\max$ 替代 $\sum$。因为 $\max$ 不涉及「多个数的和」，对数域直接可用、无需缩放：$\delta_t(j) = \max_i\bigl(\delta_{t-1}(i) + \log a_{ij}\bigr) + \log b_j(o_t)$。

但「最优路径长什么样」这个问题，光有分数回答不了。还需要**回溯指针（backpointer）**：每一步记下「这个最大值是从哪个状态来的」：

$$\psi_t(j) = \arg\max_{i}\, \delta_{t-1}(i)\, a_{ij}$$

## 5 公式解析：回溯指针的递归结构

维特比的「解码」是两步：**前向找最优，后向找路径**。

- **第一步，前向累积**：从 $t=1$ 到 $T$，对每个 $j$ 算 $\delta_t(j)$ 并存下 $\psi_t(j)$。$\psi_t(j)$ 记录的是「走到 $s_j$ 最优时，上一步站在哪」。
- **第二步，终点收束**：最优整条路径的终点是 $q_T^* = \arg\max_j \delta_T(j)$。
- **第三步，回溯还原**：从 $q_T^*$ 出发，反复查表 $q_t^* = \psi_{t+1}(q_{t+1}^*)$，一路倒推回 $q_1^*$。把路径倒过来，就是完整的最大后验状态序列。

这条「先贪心地维护局部最优来源、再反查」的流程，正是贝尔曼最优性原理的直接体现：**整条路径最优，则它的每一段前缀也必须是到那个状态为止的最优。** 因此维护每个状态的局部最优来源（而非所有候选），信息不丢失。

用 Python 写一个对数域维特比：

```python
import numpy as np

def viterbi(logA, logB, logpi):
    """logA:(N,N) 对数转移, logB:(T,N) 对数发射, logpi:(N,) 对数初始
    返回最优状态序列 q*(长度 T) 与对数概率"""
    T, N = logB.shape
    delta = np.full((T, N), -np.inf)
    psi = np.zeros((T, N), dtype=int)
    delta[0] = logpi + logB[0]
    for t in range(1, T):
        cand = delta[t-1, :, None] + logA       # (N,N): 从每个 i 到每个 j
        delta[t] = cand.max(axis=0) + logB[t]
        psi[t] = cand.argmax(axis=0)            # 最优来源状态
    q = np.zeros(T, dtype=int)
    q[-1] = delta[-1].argmax()                  # 终点收束
    for t in range(T-2, -1, -1):
        q[t] = psi[t+1, q[t+1]]                 # 回溯还原
    return q, delta[-1].max()
```

**辨析｜易错点：维特比返回「一条路径」，前向返回「一个数」。** 有人把维特比分数当观测似然用，这是错误的：$\max_Q P(Q, O \mid \lambda)$ 恒小于等于 $P(O \mid \lambda) = \sum_Q P(Q, O \mid \lambda)$，两者量纲不同、用途不同——评估用前者、打分用求和，解码用后者、对齐用取最大。<span class="marginnote">维特比在语音里还有一处关键用途：训练时的<strong>强制对齐（forced alignment）</strong>。给定音频与文本，用维特比把每一帧对齐到音素状态，产出「帧-状态」标注——这是嵌入式训练、以及后续三音子决策树聚类（见《三音子模型与决策树状态绑定》）的原材料。</span>

## 6 一份完整的对数域前向后向

把前向、后向、$\gamma$、$\xi$ 串成一个可用的模块，作为 Baum-Welch 的「E 步引擎」：

```python
import numpy as np

def forward_backward(logA, logB, logpi):
    """对数域前向后向。返回 logP(O|lambda), gamma, xi"""
    T, N = logB.shape
    # 前向（log-sum-exp 实现）
    logalpha = np.zeros((T, N))
    logalpha[0] = logpi + logB[0]
    for t in range(1, T):
        m = logalpha[t-1][:, None] + logA       # (N,N)
        mx = m.max(axis=0)
        logalpha[t] = mx + np.log(np.exp(m - mx).sum(axis=0)) + logB[t]
    # 后向
    logbeta = np.zeros((T, N))
    for t in range(T-2, -1, -1):
        m = logA + logB[t+1] + logbeta[t+1]     # (N,N)
        mx = m.max(axis=1)
        logbeta[t] = mx + np.log(np.exp(m - mx).sum(axis=1))
    loglik = logalpha[-1].max() + np.log(np.exp(logalpha[-1] - logalpha[-1].max()).sum())
    # gamma 与 xi
    gamma = np.exp(logalpha + logbeta - loglik)
    xi = np.zeros((T-1, N, N))
    for t in range(T-1):
        m = logalpha[t][:, None] + logA + logB[t+1] + logbeta[t+1]
        mx = m.max()
        xi[t] = np.exp(m - (mx + np.log(np.exp(m - mx).sum())))
    return loglik, gamma, xi
```

这个函数是下一节 Baum-Welch 的骨架：E 步产出 $\gamma, \xi$，M 步直接按公式重估三要素。注意这里每一步的 log-sum-exp 都先减最大值再取指数，正是防止数值下溢的标准姿势。

## 7 小结

- **数值问题是 HMM 落地的第一道坎**：概率连乘必然下溢，对策是对数域（配 log-sum-exp）或逐步缩放。
- **后向变量 $\beta_t(i)$** 从前向后向递归，复杂度 $O(TN^2)$，与前向镜像。
- **$\gamma, \xi$ 软计数**由 $\alpha$ 与 $\beta$ 相乘除以 $P(O\mid\lambda)$ 得到，是 Baum-Welch E 步的产出；$\sum_j \xi_t(i,j)=\gamma_t(i)$ 可用来校验实现。
- **维特比 = 前向取最大 + 回溯指针**：$\psi_t(j)$ 记录最优来源，终点收束后反查还原整条路径；对数域下无需缩放。
- **评估用求和、解码用取最大**，两者含义与数值都不同，不可混用。

在下一节，我们将把 $\gamma$ 与 $\xi$ 喂给 Baum-Welch 的 M 步，回答 HMM 的第三个问题：**从一批语音数据里，把三要素 $\lambda = (\mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$ 真正训练出来**。
