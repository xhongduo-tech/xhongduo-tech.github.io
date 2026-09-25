---
title: 线宽压窄与带宽
date: 2026-09-08
section: litho
---

# 线宽压窄与带宽

<div class="epigraph">
<p>准分子的自然线宽对高 NA 物镜太宽。压窄到皮米量级，色差与焦深才进预算；压得过窄，脉冲能量和相干性又会找上门。</p>
<footer>—— 对照 [色差与光源带宽](/litho/chromatic-bandwidth)；Cymer 等对 narrowing 的公开说明</footer>
</div>

[上一课](/litho/arf-excimer-laser)给出放电光谱。缺口是：这条光谱对 [Zernike 色差](/litho/chromatic-bandwidth) 和焦深是否可接受。本课钉线宽压窄，不重做阿贝或瑞利。

## 问题

高 NA 浸没物镜有残余色散。光源积分带宽 $\Delta\lambda$ 过宽，不同波长的焦面错开，等效于焦深被吃、NILS 掉。自由运行准分子的带宽远大于物镜预算，必须腔内光栅或棱镜压窄。压得过窄：提取效率下降、斑纹（[speckle](/litho/speckle)）上升，因为时间相干变长。

## 方法

公开手段：腔内扩束 + 光栅选频，反馈锁中心波长。指标用 FWHM 或 E95 积分带宽，单位 pm。监测：波长计进剂量/波长闭环。换胶或换 NA 配方可能换带宽规格，不是「越窄越好」。

<span class="marginnote">数字实例：ArF 准分子自由运行带宽有数百皮米，而高 NA 浸没物镜的预算常在皮米量级——差着两个数量级以上，这就是腔内必须加光栅的硬理由，不是「锦上添花」。</span>

```mermaid
flowchart TD
  RAW["放电宽带"] --> GR["光栅压窄"]
  GR --> BW["积分带宽"]
  BW --> CA["色差 / 焦深"]
  BW --> SP["斑纹"]
```

<span class="marginnote">中心波长漂移与带宽是两件事。漂移把整场焦面挪走；带宽把焦面涂厚。稳定环两套传感器。</span>

## 机制

物镜的纵向色差把 $\Delta\lambda$ 映射成 $\Delta z$。预算从镜头设计来，光源必须供得上。时间相干长度 $\sim\lambda^2/\Delta\lambda$，带宽变窄，照明相干性变长，斑纹对比上升——与 [时间与空间相干](/litho/temporal-spatial-coherence) 同一骨架。

<span class="marginnote">数字实例：取 $\lambda=193$ nm。$\Delta\lambda=100$ pm 时相干长度约 $\lambda^2/\Delta\lambda\approx0.37$ mm；压窄到 $\Delta\lambda=1$ pm，相干长度放大百倍到约 37 mm——相干性变长，干涉颗粒（斑纹）就开始显形。</span>

<span class="marginnote">常见误区：以为带宽规格「越窄越先进」。带宽压过镜头预算的部分是白付的：脉冲能量被光栅丢掉，斑纹对比上升，产能与 CD 均匀性反而变差。规格跟着镜头预算走。</span>

```mermaid
flowchart TD
  DL["光源带宽 Δλ"] --> DZ["纵向色差映射成 Δz"]
  DZ -->|"Δz 远小于 DOF 预算"| SAFE["焦面干净"]
  DZ -->|"Δz 逼近 DOF"| EAT["焦面被涂厚，NILS 掉"]
  DL --> COH["时间相干 ~ λ²/Δλ"]
  COH -->|"压得太窄"| SPE["斑纹对比上升"]
```

## 边界

下一课脉冲能量与重复频率：带宽规格约束下还要出功率。不在本课写气体寿命。

## 小结

- 压窄是为色差与焦深，不是为了「光谱好看」。
- 过窄付能量与斑纹。用积分带宽进镜头预算。
- 出处：色差课；准分子 narrowing 公开技术。
