---
title: BARC 开口刻蚀
date: 2026-09-08
section: litho
---

# BARC 开口刻蚀

<div class="epigraph">
<p>胶上的窗只开到 BARC 顶；转印还要把抗反射层打开。开口刻蚀吃 CD、吃侧壁，是 OPC 回环的第一段等离子体。</p>
<footer>—— 对照 BARC open etch；接续 [驻波与 BARC](/litho/standing-wave-barc)、[刻蚀偏置](/litho/cdu-etch-bias)</footer>
</div>

[上一课](/litho/backside-bevel-clean)把片夹干净。缺口回到正面：显影停在胶，BARC / SOG 仍盖着衬底。本课钉开口刻蚀，不把主衬底刻蚀写完。

## 问题

BARC 为光学存在，对刻蚀是必须打穿的膜。开口偏置进入 [刻蚀进 OPC](/litho/etch-in-opc) 的第一项。选择比不足会打穿衬底或削胶顶；过刻侧蚀吃线宽。把 AEI–ADI 差全写成主刻蚀，会漏掉这一薄层。

<span class="marginnote">术语翻译：「选择比」就是两种材料被等离子体吃掉的速度之比。选择比大，意味着这一段只打 BARC、几乎不动胶和衬底；选择比不足，就会在打穿 BARC 的同时削掉胶顶、甚至挖进衬底——这就是「窗口窄」的物理来源。</span>

<span class="marginnote">常见误区：以为 AEI（主刻后）与 ADI（显影后）的 CD 差全是主刻蚀的错。中间还隔着 BARC 开口这一段，它先把自己的偏置加进去；不单独量「开口后」的 CD，偏置就会错记到主刻头上，OPC 修错了对象。</span>

## 方法

短等离子体，化学依 BARC 是有机还是含硅（SOG）。终点：光学或时间。三层：先开 SOG 再开 SOC，每段选择比不同。CD 量测应分 ADI、BARC-open 后、主刻后，才能把偏置拆开。

<span class="marginnote">数字实例：假设 20 nm 厚的 BARC 以每秒约 1 nm 的速率被打开，终点探测只要晚两秒就多打 2 nm——这已经能吃掉一大块 CD 偏置预算。所以「时间终点」只能当保险丝，「光学终点」才是主控，两秒的延迟都不是小事。</span>

```mermaid
flowchart TD
  ADI["ADI 胶窗"] --> BO["BARC / SOG 开口"]
  BO --> SOC["可选 SOC 开口"]
  SOC --> MAIN["主刻蚀"]
```

<span class="marginnote">有机 BARC 常用氧化性气体，胶也会被吃。开口时间窗口窄，是随机残膜与过刻之间的缝。</span>

## 机制

各向异性开口尽量保侧壁，但仍有微观侧蚀。驻波造成的胶脚在开口时可能被修掉或被放大。与去渣的差别：去渣清残胶，开口打穿设计存在的底层。

```mermaid
flowchart TD
  WIN["开口时间窗口"] --> U["欠刻：BARC 残膜未打穿"]
  WIN --> OK["刚好：打穿且侧壁直立"]
  WIN --> O["过刻：侧蚀与胶顶损失"]
  U --> F1["主刻后留柱或 CD 漂移"]
  O --> F2["线宽缩小、侧壁变差"]
  F1 --> M["ADI / 开口后 / AEI 三点量测拆偏置"]
  F2 --> M
```

## 边界

下一课胶去除：图形已经转到硬掩模或衬底之后，胶本身要剥离。开口时胶还在当掩模。

## 小结

- BARC 开口是光学层的等离子体代价，偏置要单独量。
- 三层开口是一条链，不是一次刻蚀。
- 出处：BARC / trilayer 开口通识；刻蚀偏置课。
