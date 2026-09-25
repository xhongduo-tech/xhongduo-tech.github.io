---
title: 多臂老虎机与 UCB
date: 2026-09-08
section: cs
---

# 多臂老虎机与 UCB

<div class="epigraph">
<p>探索未知臂与利用当前最好之间折中；UCB 用均值加 $\sqrt{(\log t)/n_i}$ 上置信界，遗憾 $O(\log T)$。</p>
<footer>—— 据 Lai and Robbins, 1985；Auer, Cesa-Bianchi and Fischer, Finite-time Analysis of the Multiarmed Bandit, 2002 整理</footer>
</div>

上一课[秘书问题](/cs/secretary-problem)一次性选择。多臂：重复拉 $K$ 臂，$T$ 轮，奖励随机。缺口是遗憾（regret）与 UCB。不重写 $1/e$。本序列「不确定输入」在此结束。不写神经网络策略。下一序列近似算法。

## 问题

臂 $i$ 均值 $\mu_i$，$\mu^*=\max\mu_i$。遗憾 $R_T=T\mu^*-\sum$ 收益。贪心只利用会锁死次优。$\varepsilon$-贪心：以 $\varepsilon$ 探索。UCB1：$i$ 的指标 $\bar x_i+\sqrt{2\ln t/n_i}$，拉最大者。次优臂拉 $O(\log T)$ 次，遗憾 $O(\log T)$。

缺口是置信界探索，不是 MDP 全动态规划。

### 不是广告拍卖课

同一公式可出现在推荐，本课算法遗憾。不写限价。

<span class="marginnote">Lai–Robbins 渐近下界。Auer 等 2002 UCB1 有限时间。后课顶点覆盖 2-近似换离线近似。</span>

<span class="marginnote">术语翻译：UCB（Upper Confidence Bound，上置信界）就是「对每个臂估计它最好可能好到哪」，每轮挑这个乐观上界最大的臂——乐观是手段：不确定的臂上界虚高，先给它机会；等拉多了、区间收窄，虚高自动消失。</span>

## 方法

每臂维护次数与和。每轮 UCB 选臂，观察奖励更新。$\varepsilon$ 可递减。下界：Lai–Robbins 对数遗憾必要（一定条件下）。

```mermaid
flowchart TD
  ARM["K 臂"] --> UCB["均值 + 置信"]
  UCB --> PULL["拉臂观察"]
  PULL --> REG["遗憾 O(log T)"]
```

分布有界假设（Hoeffding）。

## 机制

机制是不确定性主动清零：欠拉的臂置信区间宽、上界虚高，指标把它顶到最前，算法被迫探索；拉够之后区间收窄、虚高消失，真次优臂被永久让位——探索不靠外生掷骰子，是置信几何自己产生的。与秘书对照：秘书不能回头拉，bandit 每轮重新选；与混合时间对照：这里奖励 i.i.d.、无状态转移，不是图游走。

```mermaid
flowchart TD
  A["臂 A: 拉 999 次, 均值 0.8"] --> IA["加项窄, 指标约 0.92"]
  B["臂 B: 只拉 1 次, 均值 0.9"] --> IB["加项宽, 指标虚高到约 4.6"]
  IB --> SEL["本轮指标最大, B 被选中"]
  SEL --> UP["n 增大, 加项按根号缩小"]
  UP --> DEC{"均值撑得住上界吗?"}
  DEC -->|"撑不住"| LOSE["回落让位, 遗憾只付这几次"]
  DEC -->|"撑得住"| WIN["继续被选, 成为新最优"]
```

<span class="marginnote">数字实例：$t=1000$ 时 $\sqrt{2\ln t}\approx 3.7$。臂 A 拉过 999 次，加项约 $3.7/\sqrt{999}\approx 0.12$，指标约 $0.92$；臂 B 只拉过 1 次，加项约 $3.7$，哪怕均值只有 $0.9$，指标高达 $4.6$——于是 B 被拉，这就是「欠拉自动被探索」背后的算术。</span>

## 边界

本课不写对抗 bandit（Exp3）、不写上下文 bandit 全文。后课默认：随机 bandit 用 UCB，遗憾对数。下一课顶点覆盖 2-近似。

<span class="marginnote">直觉类比：探索–利用就像选晚饭——常去的那家（当前最优臂）不会难吃，但巷口新开的店（欠拉臂）可能更好；UCB 给「没试过的新店」记一笔「可能更好」的加分，试过几次加分自然衰减，决策慢慢回到真实排名。</span>

## 小结

- 遗憾权衡探索/利用。
- UCB 上置信界，$O(\log T)$。
- 贪心可锁死。
- 出处：Lai and Robbins, 1985；Auer, Cesa-Bianchi and Fischer, 2002。
