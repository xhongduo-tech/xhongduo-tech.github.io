---
title: 服务成本模型
date: 2026-09-08
section: llm
---

# 服务成本模型

<div class="epigraph">
<p>token 价格不是模型质量的函数，而是 GPU 时间除以有效吞吐，再叠上利用率、电与失败重试。</p>
<footer>—— 对照 Pope 等对推理扩展的阶段划分，以及 Splitwise 把 prefill 与 decode 分到不同机器的成本动机（Patel et al.）</footer>
</div>

[上一课](/llm/energy-per-token)给出每 token 焦耳。财务账单通常由 *租用的加速器小时* 主导，电是第二项。本课把吞吐、利用率、阶段拆分收成单位 token 成本，使「再开一个 $N=8$ 投票」能写成美元，而不是口号。Splitwise（Patel 等）把 prefill 与 decode 拆机，正是因为两段的屋顶不同、机器单价不同。本课写模型，不写某一云厂商价目表当科学常数。

## 问题

有效成本

$$
C_{\mathrm{tok}}\approx\frac{C_{\mathrm{GPU-h}}/3600}{\eta\cdot \mathrm{TPS}}+C_{\mathrm{energy}}(E_{\mathrm{tok}}),
$$

其中 TPS 是集群级 token/s，$\eta$ 是相对峰值吞吐的利用率（排队、气泡、失败重试、投机拒绝都进 $1-\eta$）。缺口是：产品按提示 token 与生成 token 分开计价，成本也必须分阶段——prefill 算力密、decode 带宽密，混用同一 $C_{\mathrm{GPU-h}}$ 与同一 TPS 会把长提示用户补贴给长生成用户，或反过来。

多样本：$N$ 条生成把分母上的「用户可见 token」变成 $1/N$ 的有效产出（若只交付一条）。自洽的准确率增益要拿 $N\times C_{\mathrm{tok}}$ 比。这是上一课程留下的账，现在可以算。

<span class="marginnote">排队时间消耗租时但不产 token，全部进 $\eta$。只在 GPU busy 时测 TPS 会低估真实 $C_{\mathrm{tok}}$。SLA 拒请求若仍占着 KV 槽，同样进利用率。</span>

<span class="marginnote">直觉类比：花 $N$ 份 GPU 时间生成 $N$ 个候选、只交付最好的 1 条，就像买 $N$ 张彩票只兑奖最高的那张——单张价格没变，但「中一次」的真实成本是 $N$ 倍，所以收益要拿 $N\times C_{\mathrm{tok}}$ 来比，而不是只看最好那次的成绩。</span>

## 方法

分两项吞吐：TTFT 路径的 prompt-TPS，TPOT 路径的 gen-TPS。机器池若分离，用各自的 $C_{\mathrm{GPU-h}}$ 与 $\eta$。若混合，用拐点课的工作点：decode 池保持在 $B^\star$ 左侧以满足 TPOT，多余算力才去 prefill。投机：gen-TPS 用 *接受的* token，成本含草稿。KV 容量限制最大 $B$，从而限制 TPS 上限——成本模型必须受[容量公式](/llm/kv-cache-size-math)约束，不能假设线性加机器就线性加 TPS（互连与调度会弯折，后课程通信课再写）。

```mermaid
flowchart TD
  RENT["加速器小时价"] --> C["C_tok"]
  TPS["有效 token/s"] --> C
  ETA["利用率 η"] --> C
  E["E_tok × 电价"] --> C
  N["交付 1 / 生成 N"] --> TPS
```

## 机制

$C_{\mathrm{tok}}$ 对 $\eta$ 极敏感：空闲热备换尾延迟，直接抬成本。预热权重、长上下文常驻 KV，都是用容量换延迟。Splitwise 的逻辑是：prefill 买算力型、decode 买带宽/显存型，避免在贵的机器上跑错屋顶。成本模型若只有一个池，至少要在表上分列两段的边际成本，否则定价与调度无法对齐。

<span class="marginnote">数字实例：设 GPU 小时价 8 美元、集群 gen-TPS 为 5000。$\eta=0.9$ 时每百万生成 token 约 $8/(3600\times 0.9\times 5000)\times 10^6\approx 0.49$ 美元；$\eta$ 掉到 0.45（一半时间耗在排队、重试、气泡），同一公式立刻变成约 0.98 美元——成本翻倍，而模型一行代码没改。</span>

两段各自贴着哪面「屋顶」跑，决定了该买什么机器：

```mermaid
flowchart TD
  P["prefill 阶段"] --> P1["整段提示一次并行算: 算力密"]
  P1 --> P2["配算力型机器, 换 TTFT"]
  D["decode 阶段"] --> D1["每步读全量 KV: 带宽/显存密"]
  D1 --> D2["配带宽型机器, 稳 TPOT"]
  P2 --> S["两池各按屋顶配机器与利用率"]
  D2 --> S
  S --> O["边际成本分列, 定价与调度对齐"]
```

## 边界

不要把公开聊天产品的每百万 token 报价当成你的 $C_{\mathrm{tok}}$：那是含毛利、缓存命中与套餐的价格。不要忽略失败重试与安全拒绝的 token。后课把 $n$ 拉长：成本与显存都变成曲线，而不是一个点。

<span class="marginnote">常见误区：初学者容易把云厂商「每百万 token 多少元」的牌价直接当自己的成本模型，实际上那里面含毛利、缓存折扣与套餐结构。自建服务要回到租时、利用率与分阶段吞吐来算，否则一上缓存或一拆池，账就对不上了。</span>

出处：Pope et al., 2022；Patel et al., Splitwise（高效生成式 LLM 推理的阶段拆分）。云价格只作数量级对照，不列入公式常数。

## 小结

- $C_{\mathrm{tok}}$ 是租时除以有效吞吐，再加电；$\eta$ 含排队与重试。
- 提示与生成必须分列；混池要用工作点约束。
- 多样本把用户可见产出除以 $N$。
- 容量墙给出 TPS 上限，成本模型不得假设无限线性扩展。
- 拆分 prefill/decode 是为了让机器单价对齐屋顶。
- 下一课：长上下文下显存随 $n$ 的曲线。
- 出处：Pope et al., 2022；Patel et al., Splitwise。
