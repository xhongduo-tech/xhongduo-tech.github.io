---
title: 矢量数据模型与拓扑关系
date: 2026-08-07
---

# 矢量数据模型与拓扑关系

<div class="epigraph">
<p>矢量数据用坐标讲述形状，用拓扑讲述连接——前者让地图精确，后者让地图智能。</p>
<footer>—— 彼得 · 伯勒（Peter A. Burrough）</footer>
</div>

<div class="article-byline">
<p>第九级 · 地理信息科学（GIScience/空间信息科学） ｜ Burrough 等《Principles of Geographical Information Systems》第2章 ｜ 2026-08-07</p>
</div>

## 为什么从矢量模型开始

前面我们建立了「点线面」的几何原语，也认识了场与实体两种视角。**矢量数据模型（vector data model）**是实体视角的工程实现：用坐标串精确记录每个对象的形状，再用**拓扑（topology）**记录对象之间的关系。理解矢量模型，等于理解 GIS 最古老也最精确的一半。<span class="marginnote">矢量模型与拓扑的关系，恰如第三级《数据结构》里「邻接表」之于「图」：没有拓扑的矢量只是几何点集，有了拓扑，道路才知道自己在哪条路上与谁相交、行政区才知道自己和谁相邻。</span>

## 1 矢量的几何基础：从点到多边形

矢量模型的一切都由三类几何构成：

**点（point）**：存储一对坐标 $(x, y)$，可附带属性。**线（polyline）**：一串有序点 $(x_0,y_0), (x_1,y_1), \dots, (x_n,y_n)$ 连成的折线，首尾不相连。**面（polygon）**：首尾相连的闭合折线，内部是有界区域。

一个多边形在磁盘上的裸存储是「一串闭合坐标」：

$$
\text{polygon} = \{(x_1,y_1), (x_2,y_2), \dots, (x_n,y_n), (x_1,y_1)\}
$$