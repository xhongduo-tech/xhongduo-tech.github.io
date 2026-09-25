---
title: 轨道与扫描仪联机
date: 2026-09-08
section: litho
---

# 轨道与扫描仪联机

<div class="epigraph">
<p>涂胶–烘烤–显影在轨道，潜像在扫描仪。联机的是片、配方和时间，不是把两台机器写成一台。</p>
<footer>—— 对照 litho cluster（track + scanner）公开架构</footer>
</div>

[上一课](/litho/hard-bake-uv-cure)停在显影后硬化。把整段工艺串起来，缺口是集群：哪一步在 track、哪一步在 scanner、片如何排队。主干 [Twinscan](/litho/twinscan-dual-stage) 写过曝光产能；本课补轨道节拍与接口，不重写双工件台。

## 问题

扫描仪论场和剂量；轨道论模块占用：涂胶杯、热板、显影碗。节拍不匹配就堆片或饿曝光。PEB 延迟（曝光到 PEB 的时间）是化学放大胶的一等参数：酸在室温也会走，延迟散了 CD。把 PEB 延迟写成「剂量不稳」，会找错旋钮。

## 方法

集群调度：曝光后尽快进 PEB 模块，延迟分布要进 SPC。接口：晶圆 ID、配方号、slot 对齐，避免涂了 A 胶去曝 B 程序。环境：轨道与扫描仪的温湿度、氨污染（胺会淬灭酸）要当一个 mill 的化学环境，不是两家设备商各管一段。

```mermaid
flowchart TD
  COAT["涂胶软烤"] --> EXP["扫描仪曝光"]
  EXP --> PEB["PEB"]
  PEB --> DEV["显影冲洗"]
  DEV --> HB["硬烤"]
```

<span class="marginnote">离线曝光（track 与 scanner 分开）在研发机台常见，PEB 延迟靠人跑秒表。量产集群把延迟收成受控分布。</span>

## 机制

<span class="marginnote">PEB 延迟翻译成大白话：曝光结束到进热板之间的等待时间。化学放大胶的酸从曝光那一刻就在扩散和被淬灭，等待不是免费的——每一秒都在悄悄改写潜像。</span>

CAR 的酸在曝光后即存在，扩散与淬灭从这一刻起算。调度方差直接变 CDU。胺类 airborne base 在排队时中和酸，表现为「等得越久越欠曝」。

<span class="marginnote">可以类比冲印照片：相纸曝光后化学反应在暗处仍继续，放久了亮暗边界自己糊开。集群调度的意义就是让每一片都在同样的等待窗口里进入下一站，把「运气」收成受控分布。</span>

<span class="marginnote">常见误区：CD 漂了先拧剂量。延迟分布散了同样漂 CD，而且方向固定朝欠曝——先调出调度日志看每片的延迟分布，再决定动不动剂量旋钮。</span>

```mermaid
flowchart TD
  EXP["曝光完成"] --> WAIT["排队等待"]
  WAIT --> DL["PEB 延迟拉长"]
  DL --> DIFF["酸继续扩散"]
  AMB["环境胺污染"] --> NEU["中和部分酸"]
  NEU --> UND["欠曝方向漂移"]
  DIFF --> CD["CD 分布变宽"]
  UND --> CD
```

## 边界

下一课热板均匀性：联机之后，仍可能是热板地图主导 CDU，而不是调度。

## 小结

- 集群把 PEB 延迟和环境化学收成受控量。
- 节拍匹配是产能，延迟匹配是 CD。
- 出处：litho cluster 通识；CAR 酸扩散课。
