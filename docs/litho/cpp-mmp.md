---
title: CPP 与 MMP
date: 2026-09-08
section: litho
---

# CPP 与 MMP

<div class="epigraph">
<p>接触多晶节距卡住栅方向密度；最小金属节距卡住互连。两根尺子分开缩，光刻层的分解策略也可以分开选。</p>
<footer>—— 对照 contacted poly pitch 与 minimum metal pitch 作为 DTCO 度量的通称</footer>
</div>

[上一课](/litho/node-naming)丢掉商品名当尺子。缺口是代用尺子。本课钉 CPP 与 MMP。如何合成晶体管密度，留给[下一课](/litho/transistor-density-metric)。

## 问题

CPP（contacted poly pitch）：相邻栅（含接触落地）的周期，约束鳍/片上的逻辑密度。MMP（minimum metal pitch）：最密金属周期，约束布线与 via 网格。前端 EUV/多重与后道 EUV/多重可以不同步——BEOL 课已见混波长。轨道高度 = 轨数 ×（某层金属节距），与 CPP 正交。

缺口是**两根独立节距**，不是一个「节点 nm」。把 CPP 当成所有层的节距，会误判金属是否该上 EUV。

### 与单次极限的关系

浸没单次半节距约 38–40 nm 量级（Mack/ASML 公开叙事，主干已用）。MMP 低于约两倍半节距就要分解或换波长。本课不重推，只把尺子对上那条墙。

<span class="marginnote">公开 CPP/MMP 数字随演讲年份变。课文用定义，不把某年 ISSCC 一页当永远真值。</span>

## 方法

DTCO：分别扫 CPP、MMP 的可印路径（SAQP、LELE、EUV 单/双）。成本按层加总。SRAM 往往两边都最紧。报告节点进展时同时报两数与轨道高度。

给定一个 MMP 目标值，选哪条分解路径？判据是它与浸没单次半节距的倍数关系，倍数每降一档，重数翻一档，成本结构也换一套。

```mermaid
flowchart TD
  M["MMP 目标值"] --> Q{"与单次可印节距比？"}
  Q -->|"不低于单次极限"| S1["浸没单次"]
  Q -->|"约一半"| S2["LELE 或 EUV 单次"]
  Q -->|"约四分之一"| S3["SAQP 或 EUV 双次"]
  Q -->|"更小"| S4["更高重数或下一代光刻"]
  S1 --> C["按层加总成本"]
  S2 --> C
  S3 --> C
  S4 --> C
  C --> D["DTCO 选工作点"]
```

<span class="marginnote">术语翻译：CPP（contacted poly pitch）量的是栅方向——相邻两根栅（算上接触孔落地的位置）中心到中心的距离；MMP（minimum metal pitch）量的是互连方向——同一金属层内最密两条线的中心间距。两把卡尺各卡一个方向，不能互相换算。</span>

<span class="marginnote">数字实例：取浸没单次半节距约 $38$ nm，即一个线加空的周期约 $76$ nm。目标 MMP 若定在 $38$ nm，正好差两倍，要两重图形化（LELE，每重只印一半的线，节距放宽到 $76$ nm）；定在 $19$ nm 就要四重（SAQP）。每翻一倍重数，过机次数和掩模张数跟着涨，这正是「按层加总」要算的账。</span>

<span class="marginnote">常见误区：初学者容易以为某层换了 EUV，整颗芯片的层策略都跟着换。实际上前后道可以不同步：栅层跟 CPP 走、M0/M1 跟 MMP 走，一层用 EUV 单次、邻层用浸没多重是常态，判断依据是每把尺自己的倍数关系。</span>

## 机制

逻辑晶体管占地 ≈ CPP × 单元高度；高度又是 MMP 与轨数的函数。所以密度对两尺都敏感。光刻机台选择按层跟尺走：栅切线跟 CPP，M0/M1 跟 MMP。

```mermaid
flowchart TD
  CPP["CPP"] --> FEOL["栅 / 切线 / 接触"]
  MMP["MMP"] --> BEOL["密金属 / via"]
  CPP --> DENS["密度"]
  HT["轨 × 金属节距"] --> DENS
```

## 边界

尺子不包含 3D 堆叠的垂直密度（后课 3D NAND）。不把 CPP 当 Fin 节距——鳍节距是第三尺，有时单独报。

后课默认：平面密度用 CPP×高度；下一课把多种密度口径（MTr/mm²、NAND2、SRAM）拆开。

## 小结

- CPP 与 MMP 是正交的可缩尺子，层策略可不同步。
- 单元高度把 MMP 接到密度公式。
- 对单次极限时用这些尺，不用商品名。
- 出处：代工/imec 对 CPP、MMP 的 DTCO 通称。
