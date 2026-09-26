---
title: 模拟退火与元启发
date: 2026-09-08
section: cs
---

# 模拟退火与元启发

<div class="epigraph">
<p>以温度 $T$ 接受变差移动 $\mathrm{e}^{-\Delta/T}$，缓冷趋向低能；元启发无最坏近似比，只有实践与马尔可夫直觉。</p>
<footer>—— 据 Kirkpatrick, Gelatt and Vecchi, Optimization by Simulated Annealing, 1983；Metropolis 等, 1953 整理</footer>
</div>

上一课[局部搜索与最大割](/cs/local-search-max-cut)卡在局部最优。模拟退火：差移动也可能走。缺口是 Metropolis 规则与冷却，以及元启发的边界：不替代 2-近似定理。本序列近似在此结束。不写神经网络训练。下一序列模型与下界。

## 问题

状态 $s$，邻域，能量 $E$（如割的负）。$T$ 高：几乎随机游走；$T\to 0$ 成贪心。冷却过快冻在差局部，过慢浪费。没有一般 $\rho$。TSP、布局等实践常用。

缺口是「有保证 vs 无保证」，不是再证 $m/2$。

### 不是物理课相变

Metropolis 来自统计物理。本课优化启发式。不要写配分函数证明全局最优——有限时间没有。

<span class="marginnote">Kirkpatrick 等 1983。Metropolis 1953。遗传算法、禁忌搜索同类元启发，点名。后课 Misra–Gries 换流模型。</span>

<span class="marginnote">数字实例：一个变差移动 $\Delta=1$。$T=1$ 时接受概率 $e^{-1}\approx 0.37$；冷到 $T=0.25$ 时只剩 $e^{-4}\approx 0.018$。同一个坏方向，高温下三分之一能走，低温下基本锁死——这就是「先探索后收敛」的旋钮。</span>

<span class="marginnote">直觉类比：像金属退火——先加热让原子乱动、逃出坏晶格，再缓慢降温让它们落进好位置。也像下山找最低谷的醉汉：年轻时肯翻过眼前小山丘（接受变差），老糊涂前要稳稳走进最深的谷。</span>

## 方法

定义邻域（翻转、2-opt）。几何降温 $T\leftarrow \alpha T$。每温度若干步。多起点。

```mermaid
flowchart TD
  T["温度 T"] --> MET["Metropolis 接受"]
  MET --> COOL["缓冷"]
  COOL --> HEU["实践解，无 ρ"]
```

与局部搜索：$T=0$ 即上一课。

## 机制

平稳分布 $\propto e^{-E/T}$（正则系综直觉）。慢冷若够慢则趋向最优，实践不够慢。与带权随机游走：有限图上的链。与 UCB：bandit 有遗憾界；SA 对组合邻域通常无。

单步内部如何决定「走不走」？Metropolis 判据把上一课的贪心扩成概率规则：

```mermaid
flowchart TD
  S["当前状态 s"] --> C["邻域里取候选 s'"]
  C --> D["算 Δ = E(s') - E(s)"]
  D --> B{"Δ 是否为负?"}
  B -- "是：变好" --> ACC["无条件接受"]
  B -- "变差" --> P["以 e^(-Δ/T) 概率接受"]
  P --> R{"掷骰子接受?"}
  R -- 是 --> ACC
  R -- 否 --> STAY["留在 s 原地"]
  ACC --> COOL["降温 T ← αT<br>进入下一步"]
  STAY --> COOL
```

常见误区：初学者容易把「接受变差移动」当成 bug 或随机噪声。它恰恰是逃离局部最优的唯一出口——上一课局部搜索卡死就是因为 $\Delta\ge 0$ 一律拒绝；SA 给每个上坡一个递减的存活概率，早期多逃、后期锁定。

与局部搜索：$T=0$ 即上一课。

## 边界

本课不写收敛定理的充分条件全文。不把 SA 当 FPTAS。后课默认：要 $\rho$ 用匹配/LP/Christofides；SA 是元启发。下一课流算法 Misra–Gries。

## 小结

- Metropolis 接受差步；缓冷。
- 无一般近似比。
- 与有保证的局部搜索 2-近似分工。
- 出处：Kirkpatrick, Gelatt and Vecchi, 1983；Metropolis 等, 1953。
