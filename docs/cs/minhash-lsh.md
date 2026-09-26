---
title: MinHash 与 LSH
date: 2026-09-08
section: cs
---

# MinHash 与 LSH

<div class="epigraph">
<p>每个哈希取集合里最小的像；两集合签名相等的概率等于 Jaccard。多组签名做成桶，相似的对才进同一桶。</p>
<footer>—— 据 Broder, On the Resemblance and Containment of Documents, SEQS 1997；Indyk and Motwani, Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality, STOC 1998 整理</footer>
</div>

[上一课](/cs/quotient-filter) 问成员。[Count-Min](/cs/count-min-sketch) 问频次。近重复文档、相似集合要 Jaccard $J(A,B)=|A\cap B|/|A\cup B|$，穷举对是平方。[通用散列](/cs/universal-hashing) 给独立排列的代理。本课不扫商簇。缺口是 MinHash 签名 + LSH 分桶。

## 问题

精确 Jaccard 要交并。MinHash：随机排列 $\pi$，签名 $\min\pi(A)$。$\Pr[\min\pi(A)=\min\pi(B)]=J(A,B)$。$k$ 个独立哈希得 $k$ 维签名，用相等比例估 $J$。LSH：把签名切成 $b$ 段，每段哈希进桶；至少一段完全相同则成候选，再精确比。缺口是**用碰撞概率对准相似度阈值**，避开高维 k-d 树失效。

<span class="marginnote">MinHash 的直觉像洗牌：把两个集合各按同一随机顺序「洗」一遍，取最上面的那张牌。两个集合越像，最上面那张越可能相同——「签名相等的概率等于 Jaccard 相似度」说的就是这一件事，于是比较集合被换成了比较一个小签名。</span>

<span class="marginnote">Broder 1997 近似文档。Indyk–Motwani STOC 1998 提出 LSH 框架。本课不把 E2LSH 全部度量抄成词条。</span>

## 方法

实现常用 $k$ 个哈希的最小，或一个哈希的 $k$ 个最小（bottom-$k$，分析略不同）。LSH 参数 $(b,r)$：$k=br$，使 $J$ 高的对期望进桶、$J$ 低的少进。候选再算真 Jaccard 或抽查。

```mermaid
flowchart TD
  SET["集合 A"] --> MH["k 个 min 哈希"]
  MH --> BAND["分成 b 段"]
  BAND --> BUCK["段哈希入桶"]
  BUCK --> CAND["候选对"]
```

与 HLL：HLL 是单集合基数；MinHash 是两集合相似。与 k-d：[k-d 树](/cs/kd-tree) 正交范围，LSH 对角度/Jaccard/欧氏有不同族。

<span class="marginnote">常见误区：以为一个 min 哈希就是好估计。单值签名只是方差极大的示性变量——相等记 1、不等记 0，一次观测噪声巨大；$k$ 个独立哈希取相等比例后，估计的波动才随 $\sqrt{k}$ 缩小，$k=100$ 量级才谈得上可用精度。</span>

## 机制

独立性不够则估计偏。对抗哈希同样破坏概率合同。不要把 LSH 写成深度学习检索课的唯一内容——本课是概率数据结构。

$(b,r)$ 放大碰撞的机制值得单独看一眼：单段要求 $r$ 行全等，概率是 $s^r$；$b$ 段独立重复，至少一段命中的概率是 $1-(1-s^r)^b$。这条 S 形曲线就是阈值附近的「断层」。

```mermaid
flowchart TD
  S["相似度 s"] --> ROW["单段 r 行全等？"]
  ROW -->|"概率 s 的 r 次方"| HIT["该段进同桶"]
  ROW -->|"其余"| MISS["该段不同桶"]
  HIT --> ANY["b 段里至少一段命中？"]
  MISS --> ANY
  ANY --> CAND["概率 1−(1−s^r)^b：成为候选对"]
```

<span class="marginnote">数字实例：取 $r=5,b=20$。$s=0.8$ 时单段全等概率 $0.8^5\approx 0.33$，至少一段命中约 $1-0.67^{20}\approx 99.9\%$；$s=0.3$ 时单段仅 $0.3^5\approx 0.0024$，命中约 $4.8\%$。高相似对几乎必进桶、低相似对几乎不进，调 $b,r$ 就是移动这条 S 曲线的拐点。</span>

流上固定大小样本下一课蓄水池，问题从相似转到「均匀抽 $k$ 个」。

## 边界

本课不证明所有度量的 $(r,cr,p1,p2)$ 定义全文。欧氏 LSH 用 p-stable，点名。精确近邻在低维仍可用树。

后课默认：Jaccard 近似用 MinHash；分桶检索用 LSH。流上均匀样本用蓄水池。

## 小结

- MinHash：$\Pr(\text{签名相等})=Jaccard$。
- LSH：分段入桶放大近对碰撞。
- 下一课在未知长度的流上均匀抽样。
- 出处：Broder, 1997；Indyk and Motwani, *STOC*, 1998。
