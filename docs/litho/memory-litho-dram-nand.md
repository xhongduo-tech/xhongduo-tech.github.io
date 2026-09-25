---
title: 存储器光刻：DRAM 与 NAND
date: 2026-09-08
section: litho
---

# 存储器光刻：DRAM 与 NAND

<div class="epigraph">
<p>DRAM 仍在平面上挤半节距，浸没 + 多重是主路径；3D NAND 把负担交给深孔刻蚀，光刻更多是对准与台阶。</p>
<footer>—— 对照 DRAM 与 3D NAND 图形化负担转移的公开论述</footer>
</div>

[上一课](/litho/foundry-roadmap-compare)对比逻辑代工。缺口是存储。本课钉 DRAM 与 NAND 的光刻角色。3D NAND 负担如何具体转移，留给[下一课](/litho/3d-nand-litho-shift)。

## 问题

DRAM：电容与字线/位线在平面周期上，ArF 浸没加 SADP/SAQP、套刻极严，EUV 插入是成本交叉问题，阵列周期性对掩模 CDU 与热点极敏感。NAND：3D 堆叠后，关键变成沟道孔、台阶、字线切割的深宽比刻蚀与沉积，光刻分辨率压力相对逻辑/DRAM 平面密线不同。

缺口是**产品类改瓶颈机台**，不是同一套 CPP 故事。把存储 capex 也写成「全是 EUV」，会错。

<span class="marginnote">可以把 DRAM 与 3D NAND 的差别想成"平房里加挤"与"盖楼"：DRAM 还是在一块地基上把房间隔得更细（平面节距），3D NAND 则是往上叠楼层，难点变成把电梯井（沟道孔）打穿所有楼层还要打直——那是刻蚀的活，不是光刻分辨率的活。</span>

### 阵列 vs 外围

DRAM/NAND 外围逻辑仍可能走类似逻辑的金属光刻；阵列才是产品特征。规划要拆。

<span class="marginnote">存储厂的浸没台数与逻辑厂 mix 完全不同，交叉模型要换输入。</span>

## 方法

分阵列/外围列层策略。DRAM 盯半节距与套刻；NAND 盯孔 AR、对准、键合（若有）。EUV 对 DRAM 的交叉按公开插入叙事理解，不编层数。

```mermaid
flowchart TD
  LAYERS["层分两类"] --> ARRAY["阵列层: 产品特征"]
  LAYERS --> PERI["外围层: 类似逻辑"]
  ARRAY --> DRAMA["DRAM: 半节距 + 套刻 + CDU 敏感"]
  ARRAY --> NANDA["NAND: 深孔对准 + 台阶"]
  PERI --> LOGIC["浸没多重 或 EUV 按成本"]
  DRAMA --> PLAN["排产与 capex 分开算"]
  NANDA --> PLAN
  LOGIC --> PLAN
```

## 机制

比特密度：DRAM 靠平面 F²；NAND 靠层数 × 平面孔密度。后者让刻蚀/薄膜的边际回报在一段时间内高于缩孔节距。光刻仍定义孔位置与套刻，但不总是分辨率英雄。

<span class="marginnote">$F$ 指半节距，DRAM 单元面积按 $F^2$ 计（经典 1T1C 单元约几倍 $F^2$）。半节距每缩一步，单元面积近似按平方缩小——平面 DRAM 的全部密度压力都从"继续缩小 F"来，这也是它对 CDU 与套刻比逻辑更敏感的根源。</span>

<span class="marginnote">常见误区是"存储厂一定最先上 EUV"。实际上 NAND 靠加层就能加密度，EUV 收益低；DRAM 才在节距挤不动时按成本交叉考虑 EUV。所以两家的浸没与 EUV 台数 mix 完全不同，交叉模型不能共用一份输入。</span>

```mermaid
flowchart TD
  DRAM["DRAM 阵列"] --> PITCH["平面节距 + 多重 / EUV"]
  NAND["3D NAND"] --> ETCH["深孔 / 台阶刻蚀"]
  NAND --> LITH["孔位置 对准 台阶"]
```

## 边界

不给各厂堆叠层数当永远数。不把 DRAM EUV 插入当已完成或永不。本课不展开电容介质物理。

后课默认：NAND 的主墙在刻蚀；下一课专门写光刻负担如何转移。

## 小结

- DRAM：平面节距与套刻，浸没多重/EUV 交叉。
- 3D NAND：分辨率压力让位于深孔刻蚀，光刻管位置与台阶。
- 阵列与外围策略不同。
- 出处：DRAM/NAND 图形化负担的公开论述。
