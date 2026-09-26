---
title: 序列比对与位并行
date: 2026-09-08
section: cs
---

# 序列比对与位并行

<div class="epigraph">
<p>比对是带分的编辑；Myers 用位运算把格图一列打进机器字，$O(nm/w)$ 算单位代价距离。</p>
<footer>—— 据 Needleman and Wunsch, 1970；Smith and Waterman, 1981；Myers, A Fast Bit-Vector Algorithm, 1999；[word RAM](/cs/word-ram) 对照整理</footer>
</div>

上一课[编辑距离与 Hirschberg](/cs/edit-distance-hirschberg)给了单位代价格图。生物比对：匹配加分、错配罚、缺口罚；全局 Needleman–Wunsch，局部 Smith–Waterman。缺口是计分矩阵与**位并行**：后课 word RAM 再抽象模型，本课先用。不重写 Hirschberg。本单元 DP 还剩最优 BST 与 Knuth。

## 问题

NW：全局对齐两端。SW：任意子串对，DP 多一项与 0 取 max。仿射缺口：三种状态（匹配/开缺口）Gotoh $O(nm)$。单位代价 Levenshtein 可用 Myers：把 $\Delta$ 编码成位向量，沿模式预处理，文本扫描 $O(nm/w)$。

缺口是计分与字并行，不是新的最优子结构。

### 不是 Transformer 注意力

序列对齐是 DP 格图，不是注意力权重。本课 CS 算法，不写神经网络对齐。

<span class="marginnote">Needleman–Wunsch 1970。Smith–Waterman 1981。Myers 1999 bit-vector。后课最优 BST 回到树形区间 DP。</span>

<span class="marginnote">数字实例：设匹配 +2、错配 -1、缺口 -2，比较 AGGT 与 ACGT：3 个匹配得 $3\times 2$、1 个错配罚 1，全局得分 5。SW 则先问「哪一段最像」——某条路径分数跌破 0 就整段丢弃，坏段不连累好段。</span>

<span class="marginnote">术语翻译：$O(nm/w)$ 里的 $w$ 是机器字位数。$w=64$ 时，一列 64 个格子的依赖关系被压进一个 64 位整数，一次加法加移位加与或就同时推进 64 格——相当于把 64 行 DP 合成 1 行位运算。</span>

## 方法

NW/SW 填表。需要路径则前驱或 Hirschberg 变体。单位距离用 Myers 位向量。长序列启发式（BLAST）点名，不保证最优。

```mermaid
flowchart TD
  NW["全局 NW"] --> GRD["计分格图"]
  SW["局部 SW"] --> GRD
  MY["Myers 位向量"] --> LEV["单位编辑距离"]
```

字母表小利于位掩码预处理。

## 机制

SW 的 $0$ 截断对应「重新开始」。Myers：水平/垂直差的进位用字内移位模拟格图依赖。与 KMP：KMP 精确无误差；比对允许误差。与 FFT：卷积可做无缺口打分，缺口仍 DP。

上面那张图是三种方法的分工；下面这张拆 SW 单个格子的递推决策——「0 截断」就藏在这一步。

```mermaid
flowchart TD
  A["填格子 i,j"] --> B{"字符相同?"}
  B -- "相同" --> C["左上分 + 匹配分"]
  B -- "不同" --> D["左上分 + 错配罚"]
  C --> E{"分数为负?"}
  D --> E
  E -- "是" --> F["截为 0 从此重新开始"]
  E -- "否" --> G["保留分数继续延伸"]
  G --> H["全表最大分 = 局部最佳对齐"]
```

<span class="marginnote">直觉类比：SW 的 0 截断像收音机找信号最好的片段——某段噪声太大（分数为负）就立刻换台重新收，而不是带着噪声往下放；全局 NW 则必须从第一个字对到最后一个字，负分路段也得走完全程。</span>

## 边界

本课不写多序列比对（NPC）。不写隐马尔可夫 profile。后课默认：全局/局部比对是带分 DP；单位距离可位并行。下一课最优二叉搜索树。

## 小结

- NW 全局、SW 局部；仿射缺口三状态。
- Myers 把单位距离做到 $O(nm/w)$。
- 启发式检索不是精确 DP。
- 出处：Needleman and Wunsch, 1970；Smith and Waterman, 1981；Myers, 1999。
