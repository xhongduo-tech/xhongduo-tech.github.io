---
title: 适应性预期与学习
date: 2026-09-08
section: econ
---

# 适应性预期与学习

<div class="epigraph">
<p>理性预期把信念钉在真实定律上；学习把信念钉在递推估计上，定律与估计互相追，暂时可以偏离 RE 路径。</p>
<footer>—— Evans and Honkapohja, Learning and Expectations in Macroeconomics, 2001；Marcet and Sargent, Convergence of Least Squares Learning, 1989</footer>
</div>

[上一课](/econ/firm-heterogeneity-investment)仍在理性预期下加总分布。本单元换问题：预期算子本身。本课缺口是**适应性学习**：最小二乘或随机梯度把系数当状态。不重写企业 $(S,s)$，不把粘性信息提前当主模型。

## 问题

RE：$\mathbb{E}_t$ 是模型真实条件期望。估计与 BK 都靠它。现实：系数未知，人用过去数据更新。Marcet–Sargent：若学习增益下降，信念可收敛到 RE。Evans–Honkapohja：E-稳定性——若人们相信的定律在学习动态下是吸引子，RE 才「可学习」。常增益学习永远不完全收敛，持续误设可放大冲击。缺口是给「预期」一条可替代 RE 的递推，而不是从零讲最小二乘。

<span class="marginnote">Evans and Honkapohja, Princeton, 2001。Marcet and Sargent, *JEDC* 1989。Bullard and Mitra 对泰勒规则可学习性。Sargent 的 *The Conquest of American Inflation*。</span>

## 方法

感知定律（PLM）：例如 $\pi_t=\alpha+\beta\pi_{t-1}$。真实定律（ALM）由 PLM 代入模型得出。学习：$\theta_t=\theta_{t-1}+\gamma_t x_t(\pi_t-\theta_{t-1}'x_t)$。E-稳定：在 RE 处 ALM 对 PLM 的映射导数稳定。政策：满足 BK 决定性的规则不一定 E-稳定，反之亦然。与校准：学习参数 $\gamma$ 几乎没有微观钉死，常被用来拟合持续性。

<span class="marginnote">术语翻译：PLM（感知定律）是老百姓心里装的那个简化公式，比如「下期通胀 ≈ 常数 + 0.8 × 本期通胀」；ALM（真实定律）是把这个公式代入经济模型后实际生成数据的定律。学习就是不断用自己的小公式去拟合实际数据，两边互相追。</span>

```mermaid
flowchart TD
  PLM["感知定律"] --> ACT["真实结果"]
  ACT --> UPD["递推更新 θ"]
  UPD --> PLM
  EST["E-稳定"] --> RE["或收敛到 RE"]
```

太阳黑子均衡的可学习性往往更差，学习有时充当选择装置。

## 机制

机制是信念成为额外状态。同一基本面冲击，经错误的 $\beta$ 预报会自我放大（通胀螺旋、资产价格外推）。常增益把结构变化当可能，适合政策体制转换，也适合制造超额波动。HANK 加上学习：高 MPC 家庭若外推收入，间接效应更猛——可叠加，本课先在代表或简单 NK 里钉装置。

<span class="marginnote">数字实例：递减增益常取 $\gamma_t=1/t$。到第 100 期时新观测的权重只有 $1/100=1\%$，旧数据占绝对主导，信念基本冻结；常增益若固定 $\gamma=0.04$，则每期最新数据永远占 4% 的权重，误差不会归零，但政策换轨时信念能跟上新体制。</span>

与卢卡斯批判：学习模型里「参数」会随政策变，因为样本变。这是批判的一种实现，不是回到适应性预期的老 IS–LM。

```mermaid
flowchart TD
  G["学习增益 γ 怎么取"] --> DEC["递减增益: γ 随样本缩小"]
  G --> CON["常增益: γ 固定不归零"]
  DEC --> CVG["信念收敛到 RE: 误差趋零"]
  CON --> LIVE["永远保留近期误差: 信念持续摆动"]
  LIVE --> AMP["小冲击被信念放大: 超额波动, 适合体制转换"]
```

<span class="marginnote">早期 Cagan、Friedman 的适应性预期是固定滞后权重；现代学习让权重由估计得出。本课用后者。</span>

<span class="marginnote">直觉类比：E-稳定性问的是「把 RE 那个点当成谷底，学习这条溪流会不会自己流进去」。若谷底是吸引子，误设的信念会被一次次更新慢慢拽回去；若不是，人再怎么学也停在错误的公式上——所以不是每个 RE 均衡都「学得会」。</span>

## 边界

本课不把粘性信息（Mankiw–Reis）与疏忽（Sims）算作学习——后两课。诊断性预期是认知偏差，也不是最小二乘。调查数据后课才当靶。不估计股市量价。

后课默认：RE 可被学习动态替换；E-稳定性是额外许可证。下一课：即使知道结构，信息也不是每期都更新。

## 小结

- 学习：PLM 的递推估计；E-稳定性选择可达到的 RE。
- 信念是状态，可放大持续性与波动。
- BK 决定性 ≠ 可学习。
- 出处：Marcet and Sargent 1989；Evans and Honkapohja 2001。
