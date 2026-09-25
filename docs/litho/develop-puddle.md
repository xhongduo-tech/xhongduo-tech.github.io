---
title: 显影：搅拌与 puddle
date: 2026-09-08
section: litho
---

# 显影：搅拌与 puddle

<div class="epigraph">
<p>溶解速率曲线给出材料能走多快；轨道给出液膜是静止 puddle 还是流动更新。传质一变，CD 跟着变。</p>
<footer>—— 对照 Mack 溶解模型与轨道 puddle develop 公开流程</footer>
</div>

[上一课](/litho/soft-bake)给出固体膜。主干已有 [衬度曲线](/litho/resist-contrast-curve) 与 [Mack 溶解](/litho/dissolution-mack)。缺口是实现：显影液怎么铺、停多久、要不要搅。本课钉 puddle 与搅拌，化学物种留给下一课 TMAH。

## 问题

实验室喷淋与厂内 puddle 的边界层不同。静止 puddle 里溶解产物积累，局部 pH 与抑制剂浓度漂；旋转喷淋更新液体，但可能引入溅射缺陷。把所有 CD 偏差写成剂量，会漏掉这一段传质。

## 方法

典型正胶：喷嘴铺满 TMAH puddle，静止数秒至数十秒，再漂洗。时间是剂量之外的第二旋钮：过显影吃线宽，欠显影留残胶。搅拌（scan nozzle、轻微旋转）减薄边界层，加快到达体相速率。配方写清：铺液转速、puddle 时间、是否间歇旋转。

<span class="marginnote">为什么重要：显影不足留下的残膜在 SEM 上和离焦、欠剂量很像；过显影吃掉的线宽又会被误记成「刻蚀腰斩」。把轨道参数（puddle 时间、搅拌）写进配方并固定，才谈得上把 CD 漂移归因给光刻剂量——否则两类旋钮互相背锅。</span>

```mermaid
flowchart TD
  SOLID["烤后胶"] --> PUD["铺液 puddle"]
  PUD --> DISS["溶解 / 传质"]
  DISS --> RINSE["后课：冲洗"]
```

<span class="marginnote">EUV 薄胶的绝对溶解量小，但对残膜与残渣更敏感。同一 puddle 时间，DUV 厚胶与 EUV 薄胶不是同一个工艺窗口。</span>

## 机制

溶解是表面反应加产物扩散。边界层厚，有效速率被扩散限制，衬度曲线的「实验室搅拌」假设失效。产物若抑制溶解，静止 puddle 会自减速——看起来像欠剂量。

```mermaid
flowchart TD
  PUD["静止 puddle"] --> BL["溶解产物在界面堆积"]
  BL --> GR["局部 pH 与抑制剂浓度漂移"]
  GR --> SL["表面反应变慢: 自减速"]
  SL --> LOOK["症状看起来像欠剂量"]
  STIR["搅拌 / 间歇旋转"] --> THIN["边界层变薄"]
  THIN --> REF["界面液体不断更新"]
  REF --> FAST["速率回到体相溶解极限"]
```

<span class="marginnote">直觉类比：边界层像咖啡杯底没搅到的那层糖浆——糖化得再快，不搅也堆在杯底，越堆越化不动。喷嘴扫一圈等于用勺子搅一下：界面换上新液体，溶解立刻回全速。CD 偏差就是这杯「搅没搅匀」的读数。</span>

<span class="marginnote">术语翻译：puddle（显影液潭）就是把显影液铺满整片晶圆、让它像一滩静止的水潭停几秒到几十秒的手段——不是喷淋冲洗，而是「泡」。传质快慢全看这潭液体的界面更新程度：死水潭自减速，活水潭贴着体相速率走。</span>

## 边界

本课不写负胶交联显影的溶剂体系细节。下一课钉 TMAH 浓度与表面活性剂，不改 puddle 运动学。

## 小结

- puddle 的传质与溶解曲线同样决定 CD。
- 搅拌与时间是轨道参数，不是扫描仪剂量的别名。
- 出处：Mack 溶解；轨道 puddle 通识。
