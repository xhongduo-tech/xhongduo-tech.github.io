---
title: E₀ 与 E_size
date: 2026-09-08
section: litho
---

# E₀ 与 E_size

<div class="epigraph">
<p>$\gamma$ 是曲线有多陡；$E_0$ 问大垫何时清场，$E_\mathrm{size}$ 问目标图形何时到尺寸。两个剂量不是一个数的两个名字。</p>
<footer>—— 据 Mack / Levinson 对 dose-to-clear 与 dose-to-size 的产线定义整理</footer>
</div>

[上一课](/litho/resist-gamma)把 $\gamma$ 钉在大垫 $T(\log E)$ 的中段。缺口是曲线上的锚点：产线说「这胶多少剂量」时，有人报清场，有人报到尺寸。本课拆 $E_0$ 与 $E_\mathrm{size}$。整条剂量–尺寸曲线，留给[下一课](/litho/dose-to-size)。

## 问题

$E_0$（dose-to-clear）通常来自开放区或大垫：正胶在给定显影时间下刚好清到底的剂量。它靠近衬度曲线的脚，强烈依赖胶厚、吸收 $B$、PEB 与显影。$E_\mathrm{size}$（dose-to-size）是某一掩模图形、某一照明下，CD 落到目标时的剂量。密线的 $E_\mathrm{size}$ 往往明显高于 $E_0$，因为边切在空中像斜坡上，当地强度低于开放区。

<span class="marginnote">两个名字直译：$E_0$（dose-to-clear）是「清场剂量」——把一块没有图形的大垫刚好显影干净所需剂量；$E_\mathrm{size}$（dose-to-size）是「到尺寸剂量」——真实线条刚好印到规格 CD 的剂量。产线说「这胶 15 mJ」，先问清报的是哪一个。</span>

缺口不是再定义 $\gamma$，而是：把扫描机配方里的「剂量」默认写成 $E_\mathrm{size}$，并把 $E_0$ 留给材料与清底检查。混用会导致「胶更敏了但密线仍印不动」——其实只是 $E_0$ 降了，$E_\mathrm{size}/E_0$ 没变或更大。

<span class="marginnote">常见误区：「换更灵敏的胶（$E_0$ 更低），机台就能开快」。若 $E_\mathrm{size}/E_0$ 不变，扫描配方用的 $E_\mathrm{size}$ 并没降，产能一分没赚——$E_0$ 低只说明材料端灵敏，产线速度看目标图形那一档剂量。</span>

### 比值读的是光学，不只是胶

$E_\mathrm{size}/E_0$ 随 NILS、节距、MEEF 变。光学对比差，要靠加剂量把斜坡抬过化学阈值，比值变大。胶 $\gamma$ 变陡，比值可以略降，但救不了零级偏置抬死的 $I_\mathrm{min}$。不要把这个比值写成胶数据表上的本征常数。

<span class="marginnote">负胶 / NTD 的「清场」语义翻转：开放区可能留下或去掉，视极性而定。写 $E_0$ 必须声明极性与显影液。金属氧化物负胶常用 dose-to-size 签核，开放区剂量含义要另写。</span>

## 方法

测 $E_0$：大垫或开放框剂量矩阵，光学或轮廓仪看残胶。测 $E_\mathrm{size}$：目标结构 FEM，CD-SEM 插值到目标 CD。两者必须同一 PEB、同一显影、同一栈。换 BARC 只动 $E_\mathrm{size}$ 而 $E_0$ 几乎不动，问题在图形光学；两者一起动，问题在吸收或催化。

LPM 用 $E_0$ 当集总锚，再用空中像把边推到 $E_\mathrm{size}$。标定顺序不能反：先拿密线 $E_\mathrm{size}$ 当 $E_0$，薄膜一改就全部漂。

```mermaid
flowchart TD
  PAD["大垫 / 开放区"] --> E0["E0 清场"]
  FEAT["目标图形 + 照明"] --> ES["E_size 到尺寸"]
  E0 --> RATIO["E_size / E0"]
  ES --> RATIO
  NILS["NILS / 节距"] --> RATIO
```

## 机制

开放区强度接近空中像的局部最大（还含 flare）。清底要求沿厚的有效剂量把 $m$ 降到 $R$ 能挖穿。图形边的强度是 $I_\mathrm{th}\lt I_\mathrm{max}$，要让 $I_\mathrm{th}$ 对应的化学达到同样的 $m$ 阈值，入射剂量必须更高——于是 $E_\mathrm{size}\gt E_0$。flare 抬暗区，会抬 $E_0$ 的「起雾」端，却不一定按同一比例抬 $E_\mathrm{size}$，暗侵蚀与桥接先报警。

<span class="marginnote">数字实例：某正胶大垫测得 $E_0=15$ mJ/cm²，密线在目标节距下 $E_\mathrm{size}/E_0\approx 2.5$，扫描配方剂量约 38 mJ/cm²。同一支胶换到 NILS 更差的孔层，比值升到 3.5，配方剂量就抬到约 53 mJ/cm²——胶没变，光学的账全记在比值里。</span>

```mermaid
flowchart LR
  OPEN["开放区：强度 ≈ Imax"] --> E0["较低剂量即可清底：E0"]
  EDGE["图形边：Ith < Imax"] --> RAISE["入射剂量加码抬过化学阈值"]
  RAISE --> ES["E_size > E0"]
  FLARE["flare 抬暗区台基"] --> FOG["抬 E0 起雾端"]
  FLARE --> BR["暗侵蚀/桥接先报警"]
```

淬灭剂负载同时抬两个锚点，但 $E_\mathrm{size}$ 对负载往往更敏感，因为边沿正在争夺带上。[淬灭剂](/litho/pag-quencher)的阈值语言在这里变成两个可测剂量。

<span class="marginnote">$E_0$ 有时被定义为「开始掉厚」而不是「清底」，教材不统一。对表写清。本课默认正胶清底，除非声明。</span>

## 边界

不把某胶说明书上的 15 mJ/cm² 写成所有节距的 $E_\mathrm{size}$。不重推瑞利公式来「预言」两个剂量。下一课把 $E_\mathrm{size}$ 从单点展开成 CD(E) 曲线。

EUV 随机效应里，平均 $E_\mathrm{size}$ 合格仍可能缺孔——那是后一单元；本课两个锚点仍是均值。

## 小结

- $E_0$ 是大垫清场；$E_\mathrm{size}$ 是目标图形到尺寸。
- 扫描配方默认 $E_\mathrm{size}$；材料比较常报 $E_0$。
- $E_\mathrm{size}/E_0$ 主要读光学对比与节距，不是胶的本征常数。
- 两者必须同栈、同 PEB 测。
- 出处：Mack / Levinson 对 dose-to-clear 与 dose-to-size。
