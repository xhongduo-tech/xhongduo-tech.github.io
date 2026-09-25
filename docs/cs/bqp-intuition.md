---
title: BQP 直觉
date: 2026-09-08
section: cs
---

# BQP 直觉

<div class="epigraph">
<p>BQP 是量子电路多项式规模、测量错误可放大的语言类。它含 BPP，被相信不含 NP 完全问题，却能因子分解。</p>
<footer>—— 据 Bernstein and Vazirani；Shor；Watrous；Arora and Barak 整理</footer>
</div>

上一课[单向函数](/cs/one-way-functions-avg) 停在经典平均。缺口是给量子多项式类一个**位置**，不是量子力学课，也不是大模型。BQP：均匀多项式量子电路，接受概率与 BPP 同类缝隙。Shor：因子在 BQP。NPC 是否在 BQP 未知，普遍不信。

## 问题

量子电路：幺正门 + 测量。状态空间 $2^n$ 维，但描述必须均匀生成。BPP $\subseteq$ BQP：经典硬币可模拟。BQP $\subseteq$ PSPACE：用多项式空间算振幅递归（对指数维求和要小心，但配置可递归）。与 NP：无已知把 SAT 放进 BQP 的算法；Grover 是平方加速，不是指数。

广义 Church–Turing：若物理量子可实现，则「有效」大于 BPP。本课不争论物理。

### 不是「量子 = 指数并行」

叠加不是 $2^n$ 份经典答案的免费读取。干涉必须设计；测量只给一帧。把 BQP 写成 PSPACE 的子集已经限制了「全能」。

<span class="marginnote">Bernstein–Vazirani 定义 BQP。Shor 1994 因子与离散对数。Watrous 的量子复杂度综述。本栏不重写量子力学公理，只定位类。</span>

<span class="marginnote">术语翻译：**BQP**=「量子计算机在多项式时间内能解、错误率可压小」的问题类，地位与 BPP 对应——只是把随机硬币换成了量子门与测量。</span>

## 方法

画包含：$P\subseteq BPP\subseteq BQP\subseteq PSPACE$，NP 旁支。点名 Shor 击穿的是因子 / 离散对数假设，不是 OWF 全体（格、后课 LWE 仍候选）。不要给量子门清单当主线。

```mermaid
flowchart TD
  BPP["BPP"] --> BQP["BQP 量子多项式"]
  BQP --> PSP["PSPACE"]
  NP["NP"] -.-> BQP
```

## 机制

复杂性补层到此：时间、空间、随机、交互、PCP、电路、参数、平均、量子，都挂在 TM / 电路模型上。下一单元换尺子：Shannon 信息。BQP 不进入那一单元的公式。

因子在 BQP 而未知在 P，是「BQP 可能真大于 BPP」的最强课堂证据，仍非证明。

Grover 把无结构搜索从 $N$ 降到 $\sqrt{N}$，仍指数于比特数。Shor 用周期发现打因子与离散对数，不打一般 NPC。QMA 是量子 NP 类似物，本课不进。把 BQP 画在 $PSPACE$ 内，已经禁止「读出全部振幅」。复杂性单元到此结束。

同样说「并行」，经典随机、量子叠加、并行机各指什么？

```mermaid
flowchart TD
  C["经典随机：一次抽一条路"] --> CR["重跑多次投票"]
  Q["量子叠加：同时携带全部路径的振幅"] --> I["设计干涉让错路相消"]
  I --> M["测量只给一个答案"]
  PAR["并行机：真同时算"] --> OUT["但内存与算力按问题指数涨"]
```

<span class="marginnote">数字实例：$n=100$ 时状态空间 $2^{100}$ 维，但描述电路只需多项式个门；测量后你只拿到 1 个比特串。Grover 在 $N=2^{100}$ 上也只把 $10^{30}$ 量级的步数压成 $10^{15}$——平方根远不是「一步查完」。</span>

## 边界

本课不推导 Shor，不引入量子纠错、QMA。不把量子退火当 BQP。后课默认：BQP 是量子多项式类，含因子，不自动吞 NP。下一课信源编码定理，接[熵](/cs/entropy-bits)。

BQP 含因子分解，被画在 PSPACE 内，且不默认含 NPC。广义 Church–Turing 是否要扩到量子，是物理与模型的交界，本课只定位类。信息论单元不再用振幅。

<span class="marginnote">常见误区：初学者容易以为「量子计算机=对所有问题指数加速」。实际上加速依赖可设计的干涉结构：因子的周期结构适合 Shor；无结构搜索只有平方加速；对 NP 完全问题，学界普遍相信没有指数加速。</span>

## 小结

- BQP：多项式量子电路 + 可放大错误。
- $BPP\subseteq BQP\subseteq PSPACE$；Shor 把因子放进来。
- 叠加不是免费指数；NPC 不默认在 BQP。
- 出处：Bernstein and Vazirani；Shor；Watrous。
