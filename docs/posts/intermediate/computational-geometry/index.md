---
pageClass: plain-doc
---

# 计算几何

周培德《计算几何》、de Berg《Computational Geometry》。按照「学完一个学科 = 写完该学科权威教材对应的全部博文」的标准，每写完一篇勾掉一条。

## 主题规划

<ProgressGrid cat="intermediate/computational-geometry" />

### 第一篇 几何基础与凸包

- [x] [几何对象与表示](./geometric-objects-representation)
- [x] [点的凸包：Graham扫描](./convex-hull-graham-scan)
- [x] [凸包：Jarvis步进](./convex-hull-jarvis-march)
- [x] [凸包：分治算法](./convex-hull-divide-conquer)
- [x] [凸多边形性质](./convex-polygon-properties)
- [x] [直线与线段相交判定](./segment-intersection-detection)
- [x] [线段求交扫描线算法](./segment-intersection-sweep)

### 第二篇 三角剖分与最近邻

- [x] [多边形三角剖分](./polygon-triangulation)
- [x] [点集三角剖分](./point-set-triangulation)
- [x] [Delaunay三角剖分](./delaunay-triangulation)
- [x] [Voronoi图](./voronoi-diagram)
- [x] [最近点对：分治算法](./closest-pair-divide-conquer)
- [x] [最近邻查询](./nearest-neighbor-query)
- [x] [欧氏最小生成树](./euclidean-minimum-spanning-tree)

### 第三篇 范围查询与对偶

- [x] [一维范围树](./one-dimensional-range-tree)
- [x] [kd树](./kd-tree)
- [x] [二维范围树](./two-dimensional-range-tree)
- [x] [几何对偶变换](./geometric-duality)
- [x] [半平面交](./half-plane-intersection)
- [x] [线排列](./line-arrangements)
- [x] [点定位](./point-location)
- [x] [区域树](./interval-tree)

### 第四篇 应用与高级主题

- [x] [可见性图与最短路](./visibility-graph-shortest-path)
- [x] [多边形的重心与几何中心](./polygon-centroid-geometric-center)
- [x] [碰撞检测基础](./collision-detection-basics)
- [x] [空间索引](./spatial-indexing)
- [x] [网格生成](./mesh-generation)
- [x] [计算几何在机器人中的应用](./computational-geometry-robotics)
- [x] [计算几何在GIS中的应用](./computational-geometry-gis)
- [x] [数值稳健性与退化处理](./numerical-robustness-degeneracy)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

### 第1篇

- [ ] 几何基础与凸包（凸包算法、Graham 扫描）
- [ ] 线段求交（平面扫描技术）
- [ ] 多边形三角剖分（可见性、艺术画廊定理）
- [ ] 低维线性规划（随机增量算法）
- [ ] Voronoi 图与 Delaunay 三角剖分（对偶性、应用）
- [ ] 点定位（梯形图、持久结构）
- [ ] 排列与对偶性（线排列、半空间交）
- [ ] 几何数据结构（区间树、线段树、范围树）
- [ ] 运动规划（构形空间、Minkowski 和）
- [ ] 鲁棒性与应用（精确计算、GIS 与图形学应用）
