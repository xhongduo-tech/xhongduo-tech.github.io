---
title: 叙事与高频识别
date: 2026-09-08
section: econ
---

# 叙事与高频识别

<div class="epigraph">
<p>冲击的标签若不能从同期零限制放心得到，就从历史记录或政策公告窗里的价格跳变去借。</p>
<footer>—— Romer and Romer, A New Measure of Monetary Policy Shocks, AER 2004；Gertler and Karadi, Monetary Policy Surprises, QE and the Stock Market, AEJ:Macro 2015；Ramey 财政叙事</footer>
</div>

[上一课](/econ/local-projections)把 IRF 估计从 VAR 套牢里拆出，并声明冲击仍须外来。本课缺口是**冲击从哪来**：叙事（读档案、读绿皮书）与高频（公告窗内期货跳变）。本单元在此收束识别；下一单元进入异质性。不重写 LP 公式。

## 问题

Cholesky 对货币、财政都脆。Romer–Romer：用绿皮书预测把「对经济状况的内生反应」从意向利率里减掉，残差当叙事货币冲击。Ramey：军费新闻日期。Mertens–Ravn：叙事减税。Gertler–Karadi、Gürkaynak–Sack–Swanson：FOMC 公告窗内联邦基金或欧洲美元期货的跳变，当意外。缺口是给 SVAR/LP 一条外部 $\varepsilon$ 或工具，而不是再排一次变量顺序。

<span class="marginnote">术语翻译：「内生反应」就是央行的条件反射——经济一软它就降息。直接拿利率变动当「货币政策冲击」，等于把果当成因；Romer–Romer 的做法是先用央行自己的预测把这一层反射减掉，剩下的才算意外。</span>

<span class="marginnote">Romer and Romer, *AER* 94(4), 2004。Gertler and Karadi, *AEJ:Macro* 7(1), 2015。Nakamura and Steinsson, *QJE* 2018 用高频看信息效应。Kuttner 的联邦基金期货是前身。</span>

## 方法

叙事：编码规则必须事前，防止用 IRF 形状反过来挑日期。把叙事序列当冲击直接 LP，或当代理进 SVAR（Mertens–Ravn）。高频：窗要短到宏观数据进不来、长到价格能反应；多目标（路径、前瞻指引）用若干期货期限抽因子。信息效应：若公告同时揭示央行的经济判断，利率升可能伴随股票升，工具就不是纯政策供给——要显式建模或选正交化。

<span class="marginnote">数字实例：FOMC 声明下午 2 点发布，取 2:00 到 2:30 的联邦基金期货跳变当意外——窗再宽，之后的宏观数据与解读新闻就会掺进来；窄于几分钟，价格可能还没反应完。窗宽是拿「干净」换「听得见」。</span>

```mermaid
flowchart TD
  NAR["叙事残差"] --> IV["当 ε 或工具"]
  HF["公告窗期货跳"] --> IV
  IV --> LP["LP / 代理 SVAR"]
  INFO["信息效应"] --> BIAS["污染标签"]
```

财政与货币共用这一逻辑，冲击账户不同。本课不把区域乘数的 Bartik 写进来，以免换数据维度。

## 机制

机制是用制度时间线或市场时钟制造外生变异。叙事靠人读条件信息；高频靠市场比宏观更早把意外资本化。两者都可能弱（工具与真 $\varepsilon$ 相关低）或污染（混进新闻）。弱工具使 LP-IV 的远地平线不可信。DSGE 估计可以把这些序列当额外观测，把结构冲击与代理对齐——回到贝叶斯课的接口，不重估 SW。

```mermaid
flowchart TD
  NAR["叙事识别: 靠制度时间线"] --> N1["读绿皮书或军费新闻"]
  N1 --> N2["减掉对经济的内生反应"]
  N2 --> N3["怕: 事后挑日期, 编码不干净"]
  HF["高频识别: 靠市场时钟"] --> H1["公告窗内期货跳变"]
  H1 --> H2["窗短到宏观数据进不来"]
  H2 --> H3["怕: 信息效应混进政策意外"]
```

<span class="marginnote">常见误区：以为「拿到外部冲击序列」就万事大吉。若它与真冲击相关很弱，LP-IV 的远处脉冲会爆炸性地不可信——估计像被放大镜拉长，标准误却装作不知道。弱工具要报第一阶段强度，不是只画 IRF。</span>

本课程动态方法单元在此停：能解、能估、能识别加总脉冲。加总脉冲对「谁的 MPC」沉默——下一单元从代表性个体走开。

<span class="marginnote">Ramey, *Handbook of Macroeconomics* 第 2 卷，财政与货币识别综述。Jarociński and Karadi 把信息效应与纯政策在符号上拆开。</span>

## 边界

本课不重建绿皮书数据库，不交易期货。不把高频识别写成微观结构（不是 Kyle）。量化宽松的期限溢价通道可点名，不把整本央行资产负债表提前写完（金融摩擦单元后部）。太阳黑子与新闻冲击（Beaudry–Portier）是预期单元的题目。

后课默认：外部冲击序列来自叙事或高频（或两者作代理）；LP/SVAR 只是第二步。下一课：加总欧拉背后，不完全市场的资产分布。

## 小结

- 叙事从档案构造条件残差；高频从公告窗价格跳变借意外。
- 可当冲击或 IV；信息效应会污染标签。
- 识别单元结束；加总 IRF 不回答异质 MPC。
- 出处：Romer and Romer, *AER* 2004；Gertler and Karadi, *AEJ:Macro* 2015；Ramey 手册。
