---
title: 理性疏忽
date: 2026-09-08
section: econ
---

# 理性疏忽

<div class="epigraph">
<p>注意力是有限信道；最优编码使人对重要、易变的变量反应更灵敏，对其余变量近乎不更新。</p>
<footer>—— Sims, Implications of Rational Inattention, JME 2003；Maćkowiak and Wiederholt, Optimal Sticky Prices under Rational Inattention, AER 2009</footer>
</div>

[上一课](/econ/sticky-information)的 $\lambda$ 是外生更新率。本课缺口是**内生注意力**：Shannon 容量下的最优信号。不把 $\lambda$ 再估一遍，不写神经网络注意力层。

## 问题

Sims：人不是每期免费看见状态，而是选择一个关于状态的信号，互信息受容量约束。结果：对小的、不重要的冲击几乎不反应，对大的冲击反应；预测误差与宏观惯性内生。Maćkowiak–Wiederholt：企业把有限注意力在总需求与特异需求之间分配，加总价格对货币更粘、对特异更灵活——与微观价格频繁改动、宏观通胀惯性可并存。缺口是把粘性信息的外生 $\lambda$ 换成最优信道，而不是再写一次 vintage 加权。

<span class="marginnote">术语翻译：理性疏忽就是「看不过来，所以有选择地看」。不是看不见，而是带宽（容量 $\kappa$）有限，你把清晰度分给最重要的东西，其余只看个模糊轮廓——而且这种分配本身是算过账的。</span>

<span class="marginnote">Sims, *JME* 2003。Maćkowiak and Wiederholt, *AER* 2009。Woodford 的不完全信息定价。Caplin、Dean、Angeletos 等后续。Shannon 是常用的 tractable 约束，不是唯一认知模型。</span>

## 方法

代理人最大化效用减注意力成本（或受 $I(X;S)\le\kappa$）。高斯二次时线性信号最优，Kalman 增益由容量决定。宏观：容量 $\kappa$ 校准到调查误差或 IRF。政策：更嘈杂的政策规则浪费容量，透明规则节省注意力——接到沟通课。与 HANK：穷人可能把容量全用在流动性，不跟踪利率路径，直接欧拉通道更弱。

<span class="marginnote">数字实例：容量以比特计。若 $\kappa=1$ 比特（约一次「是/否」判断的信息量）要在「盯通胀」与「盯利率」之间分配：平均各摊半比特，两件事都只能看得雾；全给通胀，则通胀看得清、利率几乎不更新——预期数据里的迟钝就是这么来的。</span>

```mermaid
flowchart TD
  KAP["容量 κ"] --> SIG["最优信号"]
  SIG --> GAIN["对重要冲击高增益"]
  GAIN --> MICRO["微观可灵活"]
  GAIN --> MACRO["宏观对总量更粘"]
```

与学习：疏忽是每期最优压缩；学习是参数递推。与诊断性：疏忽是噪声，诊断性是偏误（后课）。

## 机制

机制是稀缺注意力的配置。价格系统若已充分统计，本可节省注意力（Hayek）；一旦价格嘈杂或策略互补强（美容竞赛），每人仍须浪费容量猜别人——Angeletos–La’O、Morris–Shin 的高阶信念接口。本课程不重写[美容竞赛](/econ/beauty-contest-higher-order)全文，只借用：疏忽加策略互补 ⇒ 更大惯性。

```mermaid
flowchart LR
  FIRM["企业的容量 κ"] --> Q["注意力往哪放?"]
  Q --> AGG["盯总量: 要读懂政策与经济周期"]
  Q --> IDIO["盯自家顾客: 直接决定利润"]
  IDIO --> MICRO["自家价格反应快"]
  AGG --> STICKY["对总量反应慢, 加总通胀粘"]
  MICRO --> OBS["与微观价格频繁改动的观察并存"]
```

容量随激励变：高通胀时人更跟踪 CPI（经验上预期更「解锚」前的更新），这是后课锚定的微观基础之一。

<span class="marginnote">本课不把信息论当宏观的第一性原理重讲。互信息是约束的方便写法。</span>

## 边界

本课不定 $\kappa$ 的生理值。不把菜单成本与疏忽混成一个参数。资产定价里的疏忽（对新闻反应不足）可点名，不估计横截面。Transformer 与机器学习注意力禁止写入本栏。

<span class="marginnote">常见误区：初学者容易把理性疏忽当成「非理性偷懒」。恰恰相反——疏忽是优化的结果：对影响小的事少花带宽，总效用比样样都盯更高。真正「不理性」的是乱分配注意力，不是省着用。</span>

后课默认：更新率可以是最优注意力配置的结果。下一课：偏差不是噪声，而是对新闻的过度反应——诊断性预期。

## 小结

- 理性疏忽：容量约束下的最优信号。
- 可同时解释微观灵活与宏观惯性。
- 与外生 $\lambda$、与学习、与认知偏差分列。
- 出处：Sims, *JME* 2003。
