---
title: 双折射与偏振像差
date: 2026-09-08
section: litho
---

# 双折射与偏振像差

<div class="epigraph">
<p>应力双折射把一块玻璃变成弱波片。高 NA 下 s/p 已经分家，再叠材料波片，Jones 瞳就不再是标量相位。</p>
<footer>—— 对照 [Jones 矩阵](/litho/jones-polarization)、[矢量成像](/litho/vector-imaging)；CaF₂ 应力双折射</footer>
</div>

[上一课](/litho/lens-aberration-metrology)量标量波前。缺口是偏振：材料与镀膜的二向色性、应力双折射、末片的 s/p 菲涅尔。主干 Jones 课已有符号；本课落到物镜硬件。

## 问题

Zernike 相位假定两种偏振看见同一 $W$。高 NA 浸没下，对照明偏振敏感的层（[照明偏振](/litho/illuminator-polarization)）会把材料波片放大成 NILS 损失或套刻的偏振依赖。只补标量操纵器，CD 仍随偏振态变。

## 方法

选低应力双折射牌号、退火、安装应力控制。计量：偏振波前（Jones 瞳）或至少双偏振 Zernike。照明侧用偏振控制匹配物镜特征，而不是「越偏振越好」。

```mermaid
flowchart TD
  STR["安装 / 热应力"] --> BR["双折射"]
  BR --> J["Jones 瞳"]
  NA["高 NA 菲涅尔"] --> J
  J --> NILS["对比 / 套刻"]
```

<span class="marginnote">CaF₂ 比熔石英更易因应力露双折射。上一课的材料分配现在有偏振代价，不是只有透过率。</span>

## 机制

应力光弹性系数把 $\sigma$ 变成 $\Delta n$。快轴沿应力主方向，等效在光瞳上贴位置相关的延迟片，混合 s/p。矢量成像核因此含偏振像差，TCC 标量化失效。

## 边界

下一课污染：薄膜脏了也会改相位和散射，但是颗粒与碳氢，不是体双折射。

## 小结

- 物镜像差有偏振通道；标量 Zernike 不够高 NA。
- 材料应力与末片菲涅尔一起进 Jones 瞳。
- 出处：Jones / 矢量成像课；CaF₂ 应力双折射通识。
