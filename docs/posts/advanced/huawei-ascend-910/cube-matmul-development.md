---
title: Cube 矩阵乘算子开发实战
date: 2026-08-07
---

# Cube 矩阵乘算子开发实战

<div class="epigraph">
<p>纸上得来终觉浅，绝知此事要躬行。</p>
<footer>—— 陆游（南宋诗人，《冬夜读书示子聿》）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 硬件：华为昇腾 910B/910C/950 ｜ 昇腾 CANN 编程指南 第5章 ｜ 2026-08-07</p>
</div>

## 为什么从矩阵乘算子开始

矩阵乘（GEMM）是深度学习的「第一大算子」——卷积、注意力、MLP 的绝大部分计算最终都归结为 GEMM。前一篇我们学了 Ascend C 的编程模型，这一篇要**亲手走一遍「写一个 Cube 矩阵乘算子」的完整流程**：从问题定义到 tiling 策略，从数据搬移到 Cube 计算，从同步到性能验收。这一篇不是「看代码」，而是「理解代码背后每一步为什么这么写」——写完后，你就拥有了「自己写昇腾算子」的完整套路。<span class="marginnote">为什么「实战」值得单独一篇？因为<strong>「知道 tiling 很重要」和「亲手把 tiling 做对」之间，隔着一条巨大的鸿沟</strong>。这一篇填的就是这条鸿沟——用最典型的算子，把抽象原理落到具体决策。</span>

本文的代码以「结构化伪代码 + 讲解」的形式呈现，不追求与某个 CANN 版本逐字一致，而是把「一个 Cube 算子必备的要素」讲透。

## 1 问题定义：要算什么

设矩阵乘为 $\mathbf{C} = \mathbf{A} \times \mathbf{B}$，其中 $\mathbf{A}\in\mathbb{R}^{M\times K}$、$\mathbf{B}\in\mathbb{R}^{K\times N}$、$\mathbf{C}\in\mathbb{R}^{M\times N}$。在昇腾上写这个算子的输入是：

三个 `GlobalTensor`：`a`、`b`、`c`（数据在 HBM）；
形状信息：`M`、`N`、`K`；
- 数据类型：BF16/FP16（输入）、FP32（累加）。

**算子要完成的事**：把 A、B 从 HBM 搬到片上，用 Cube 做矩阵乘，结果写回 C。听起来简单，但「怎么搬、怎么切、怎么算、怎么同步」全是决策点。

## 2 tiling 策略：把大矩阵切成 Cube 能吃的块

Cube 一次做 $16\times16\times16$ 的乘加。一个大矩阵不能一次塞进 Cube，必须切成 tile。tiling 的核心问题是：**怎么切，才能让「搬入量最小、Cube 利用率最高」**？

### 2.1 分块矩阵乘：经典的三层循环

标准的分块 GEMM 把 $\mathbf{A}$ 按「行分块 × 列分块」、$\mathbf{B}$ 按「行分块 × 列分块」切成块，然后：

```
for i in range(M / BM):            # A 的行块
  for j in range(N / BN):          # B 的列块
    for k in range(K / BK):        # 缩减维度块
      load A[i,k]  -> L0 Buffer A   # 搬 A 的一个块
      load B[k,j]  -> L0 Buffer B   # 搬 B 的一个块
      C[i,j] += A[i,k] * B[k,j]     # Cube 计算并累加
```

其中 $B_M$、$B_N$、$B_K$ 是块尺寸，受限于 L0/UB 容量。**tiling 的实质，就是给这三个循环变量选好块尺寸**。<span class="marginnote">分块矩阵乘是「循环分块」的经典应用：<strong>内层循环的数据尽量留在片上，外层循环才与 HBM 打交道</strong>。块越大，数据复用率越高、HBM 流量越少——但块受片上容量约束。这就是第 4 篇「驻留率」的落地。</span>

### 2.2 一个具体的选择

假设 L0 Buffer 能装下 $B_M\times B_K$ 的 A 块与 $B_K\times B_N$ 的 B 块。若取 $B_M=B_N=128$、$B_K=64$，则每个输出块 $128\times128$ 由 Cube 算 $64$ 次累加。**块选得越大，每个字节被复用的次数越多，算术强度越高**——这直接对应第 1 篇的算术强度公式。实践中，tiling 参数由编译器自动求解，但开发者要理解「为什么是 128 而不是 16」：16 太小（复用不足），512 太大（L0 放不下）。

## 3 数据搬移：GM → L1/L0 → UB 的三级接力

在 Ascend C 里，数据搬移用 `DataCopy` 系列指令，典型路径是：

```
// 从 HBM (GM) 搬到 L1
DataCopy(gm_a_tile, l1_a_tile, size_a);
// 从 L1 搬到 L0 Buffer A
DataCopy(l1_a_tile, l0_a_tile, size_a);
// 同样处理 B
DataCopy(gm_b_tile, l1_b_tile, size_b);
DataCopy(l1_b_tile, l0_b_tile, size_b);
// Cube 计算
Matmul(l0_a_tile, l0_b_tile, l0c_out);
```

这串代码里藏着两个关键决策：**为什么经过 L1？为什么搬这么多次？**<span class="marginnote">L1 是「预取中转站」：DMA 先把数据从 HBM 批量搬进 L1，再由 L1 向 L0 供数。<strong>这三级接力让「搬数据」这件事可以流水化</strong>——L1 里存的是下一块，当前块正在 L0 被 Cube 消费，互不等待。回顾第 4 篇的多级双缓冲，这里就是它的代码形态。</span>

### 3.1 双缓冲：让搬运与计算重叠

为了不让 Cube 空等数据，搬移要双缓冲：L0 Buffer A 分两块，一块在算、一块在搬。代码里的体现是「交替使用两个缓冲分区」：

```
// 第一次：搬第 0 块，算第 0 块
DataCopy(..., l0_a_part0, ...);
Matmul(l0_a_part0, l0_b_part0, l0c);
// 循环里：搬第 i+1 块的同时，算第 i 块
for i in ...:
  DataCopy(..., l0_a_part[i % 2], ...);   // 搬到「空闲」的那一半
  WaitFlag(...);                           // 等搬完
  Matmul(l0_a_part[(i+1) % 2], ..., l0c); // 算「上一半」的数据
```

**双缓冲的收益**：只要「搬一块的时间 ≤ 算一块的时间」，Cube 就永不空等，利用率逼近 100%。这是昇腾算子性能的「第一杠杆」。<span class="marginnote">双缓冲的代价是 L0 容量翻倍——所以「按需双缓冲」很重要：<strong>权重（B 矩阵）复用率高，可以常驻；激活（A 矩阵）每块都换，才值得开双缓冲</strong>。回顾第 4 篇的「A 流式、B 常驻」分区直觉，这里就是它的实现。</span>

## 4 Cube 计算与累加

Cube 计算的核心指令是 `Matmul`，它读 L0 里的 A、B 块，把乘加结果累加到 L0C：

```
Matmul(l0_a_tile, l0_b_tile, l0c_accum);
```

关键点有两个。**其一，累加**：$\mathbf{C}$ 需要对 $K$ 维做 $K/B_K$ 次累加，所以 L0C 存的是「部分和」，循环结束才写回。**L0C 用更高精度（FP32）保存累加**，避免精度损失（回顾第 4 篇）。**其二，数据格式**：输入到 Cube 的 A、B 块必须是 Cube 期望的格式（如 FRACTAL_NZ），若不是，搬移时要顺手做格式转换。<span class="marginnote">「累加在 L0C、转换在搬移」是两个常被新手忽略的细节：<strong>累加精度错则结果错，格式错则性能崩</strong>。它们正是「显式数据流」里「隐性的地雷」。</span>

## 5 同步与写回：别让流水「撞车」

矩阵乘算子内部的依赖链是：**搬 B → 搬 A → 等搬完 → Cube 算 → 等算完 → 搬 C 回 HBM**。每一步都要用同步原语（`SetFlag`/`WaitFlag`）保证顺序：

```
SetFlag(Event_DMA_OUT);   // 通知「DMA 搬完了」
WaitFlag(Event_CUBE_IN);  // 等「Cube 可以开始」
Matmul(...);
SetFlag(Event_CUBE_OUT);  // 通知「Cube 算完了」
WaitFlag(Event_DMA_IN);   // 等「可以搬回」
DataCopy(l0c_out, gm_c_tile);
```

**同步点不是越多越好**：同步太多，流水被切碎；同步太少，数据竞争出错。**在「依赖链的拐点」精确插入同步**，是算子写得「又快又对」的平衡术。<span class="marginnote">回顾第 3 篇的「同步 = 依赖的时间投影」：<strong>每一条跨单元的依赖，都需要一条同步指令来担保</strong>。写算子时的全部「工程感」，就是把这些依赖点一个个找出来、用同步钉住。</span>

## 6 公式解析：算子能达到多快

写完后，怎么知道算子「够不够快」？用峰值利用率（MFU）来验收。设算子的实际执行时间为 $T_{\text{actual}}$，理论计算时间为 $T_{\text{ideal}}=2MNK/P_{\text{peak}}$，则：

$$\text{MFU} = \frac{T_{\text{ideal}}}{T_{\text{actual}}}$$

逐项拆解：

- **第一步，$T_{\text{ideal}}$**：$2MNK$ 是总 FLOPs，除以峰值算力 $P_{\text{peak}}$ 得到「理论上最短时间」。
- **第二步，$T_{\text{actual}}$**：实测时间，包含搬移、同步、启动等一切开销。
- **第三步，$\text{MFU}$ 的含义**：MFU 越接近 1，算子越接近「算满」。若 MFU 只有 50%，说明一半时间在等数据或等同步。

**验收的实战流程**：先算 $T_{\text{ideal}}$，再实测 $T_{\text{actual}}$，看 MFU 落在哪。若 MFU < 70%，优先检查「搬算是否重叠、tile 是否偏小、同步是否过多」——这三个是矩阵乘算子最常见的性能杀手。<span class="marginnote">一个「好」的 Cube 算子，MFU 通常能到 80%+。若你写的算子 MFU 只有 40%，别急着怀疑硬件——<strong>先查 tiling 与双缓冲，十有八九是这里出了问题</strong>。这也是昇腾性能工程师排障的第一顺序。</span>

## 7 小结

- **矩阵乘算子 = tiling + 搬移 + Cube 计算 + 同步 + 写回**，五步环环相扣。
- **tiling 的核心是三层循环的块尺寸**：块越大复用越高，但受 L0/UB 容量约束。
- **数据走「GM → L1 → L0」三级接力**，双缓冲让搬运与计算重叠。
- **Cube 用 L0C 做高精度累加**，输入需符合 Cube 期望的数据格式。
- **同步点钉住依赖链的每个拐点**：又多又少都不行，要「恰到好处」。
- **MFU = $T_{\text{ideal}}/T_{\text{actual}}$**：MFU < 70% 先查搬算重叠、tile 大小、同步多少；好算子的 MFU 通常 80%+。

在下一节，我们将从「单算子的实战」转向「多卡如何协作」——正是 **HCCL 集合通信与多卡并行**。