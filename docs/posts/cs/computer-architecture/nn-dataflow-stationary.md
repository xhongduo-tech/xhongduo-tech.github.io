---
title: 神经网络加速中的数据流：Weight/Output/Row Stationary
date: 2026-08-07
---

# 神经网络加速中的数据流：Weight/Output/Row Stationary

<div class="epigraph">
<p>算一次乘加的能耗微乎其微，把操作数搬来搬去的能耗才是大头——数据流决定谁留下、谁奔波。</p>
<footer>—— Eyeriss 数据流研究（Chen et al., 2016）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机体系结构 ｜ Hennessy & Patterson《Computer Architecture: A Quantitative Approach》第 7 章 ｜ 2026-08-07</p>
</div>

## 为什么「谁不动」这么关键

一次矩阵乘加需要三个操作数：**权重、激活、部分和**。如果每个 MAC 都要从内存重新取这三样，数据搬运能耗就把计算能耗淹没了。**数据流（dataflow）** 回答的是：**哪个操作数「驻留」在 PE 里不动、哪个「流动」经过 PE**——这决定了数据搬运的总量，也就决定了能效。<span class="marginnote">[[npu-edge-ai-accelerators]] 提到「访 DRAM ≈ 数百次片上运算」——数据流就是把这个数字做小的设计空间。<strong>同一个算法，不同的数据流可以差 10 倍能耗</strong>。</span>

## 1 三种驻留策略

**核心概念**：**数据流（dataflow）**：在 MAC 阵列上映射矩阵运算时，对「权重（W）、激活（A）、部分和（P）」三个操作数分别决定「驻留 or 流动」。三种经典策略：

| 策略 | 驻留 | 流动 | 适合 |
| --- | --- | --- | --- |
| **Weight Stationary（WS）** | 权重 | 激活流、部分和流 | 权重复用高（全连接、大矩阵） |
| **Output Stationary（OS）** | 部分和 | 权重流、激活流 | 部分和复用高（卷积） |
| **Row Stationary（RS）** | 行级复用 | 混合 | 卷积+全连接兼顾 |

**WS（权重驻留）**：把权重矩阵装入每个 PE 不动，激活从一侧流入，部分和从另一侧流出——**TPU 脉动阵列的默认**（[[google-tpu-systolic-array]]）。
**OS（输出驻留）**：把「部分和」留在 PE 里反复累加，权重与激活轮流送进来——**卷积里每个输出要累加很多项，OS 让累加不搬家**。
**RS（行驻留）**：把一行权重与一行激活都留在阵列里，最大化两种复用——**Eyeriss 芯片的招牌**。

## 2 为什么卷积偏爱 OS

卷积的输出 $O[i][j]$ 是「输入 patch × 权重核」的累加——**每个输出要累加 $K \times K \times C_{in}$ 项**。若部分和来回搬，累加 27 次就要搬 27 次；**OS 让部分和钉在 PE 里，只让权重与激活流过**，累加开销变成零搬运。<span class="marginnote">直觉：<strong>谁「被反复用」谁就留下来</strong>。卷积反复用的是「部分和」（累加项多），全连接反复用的是「权重」（每个权重参与所有输出）——各自选对应的驻留策略。</span>

## 3 数据流背后的本质：循环重排

数据流不是玄学，它等价于**循环的重排（loop order）**。矩阵/卷积本质是嵌套循环，**把哪个循环放在最内层 = 让哪个操作数驻留**：

```c
// 矩阵乘 C = A × B 的三重循环（i、j 定输出元素，k 为累加维）
for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
        for (int k = 0; k < K; k++)
            C[i][j] += A[i][k] * B[k][j];

// Output Stationary：累加维 k 在最内层 → 部分和 C[i][j] 驻留在 PE 里
// Weight Stationary：把 (k, j) 挪到外层、i 放最内层 → B[k][j] 驻留，A 流动
for (int k = 0; k < K; k++)
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++)
            C[i][j] += A[i][k] * B[k][j];   // 内层扫 A 的行，B[k][j] 不动
```

**核心概念**：数据流 = **把矩阵运算的嵌套循环映射到二维 PE 阵列的空间与时间**。这正是 [[simd-programming-autovectorization]] 循环变换思想在专用硬件上的重演——只是「编译器」换成了「映射器（mapper）」。

## 4 公式解析：复用率与能耗

$$
E \approx N_{\text{MAC}} \times E_{\text{on}} + \frac{N_{\text{MAC}}}{R} \times E_{\text{off}}
$$

- **第一步，看 $N_{\text{MAC}} \times E_{\text{on}}$**：每个乘加的计算能耗，固定且小。
- **第二步，看 $\frac{N_{\text{MAC}}}{R} \times E_{\text{off}}$**：数据搬运能耗——**复用率 $R$ 越高，搬运次数越少**。$R$ 是「每个操作数被用几次」。
- **第三步，理解 $R$ 的来源**：驻留策略决定 $R$。WS 让权重复用 $R_w$ 倍、OS 让部分和复用 $R_p$ 倍、RS 让两者都高。**$R$ 差 10 倍，能耗差 10 倍**——数据流设计的全部意义在此。

**辨析｜易错点：** 没有「最优数据流」——它取决于**模型的结构与硬件的存储预算**。全连接用 WS、卷积用 OS 是「一般规律」，但具体到某个网络、某块 SRAM，要**做能耗建模选最优**。**抄别人的数据流，不如算自己的**。

## 5 小结

- 数据搬运能耗 ≫ 计算能耗，**数据流决定搬运总量**。
- 三种驻留：**WS（权重不动）、OS（部分和不动）、RS（行级混合）**。
- 卷积偏爱 **OS**（累加项多、部分和复用高），全连接偏爱 **WS**（权重复用高）。
- 数据流 = **循环重排**：最内层循环决定谁驻留。
- 能耗公式 $E \approx N_{\text{MAC}}E_{\text{on}} + \frac{N_{\text{MAC}}}{R}E_{\text{off}}$：**复用率 $R$ 是能效的关键旋钮**。

在下一节，我们看加速器的另一大能效来源——**稀疏性与量化在 DSA 中的利用**。
