---
title: 空间数据库与空间索引
date: 2026-08-07
---

# 空间数据库与空间索引

<div class="epigraph">
<p>空间查询的秘诀不是扫描所有数据，而是「先缩小范围，再精确定位」。</p>
<footer>—— 汉斯 · 格特曼（Hans Guttman，R-树的发明者）</footer>
</div>

<div class="article-byline">
<p>第九级 · 地理信息科学（GIScience/空间信息科学） ｜ Longley 等《Geographic Information Science and Systems》第6章 ｜ 2026-08-07</p>
</div>

## 为什么从空间数据库开始

前面几篇把世界装进了各种数据结构，但还缺最后一个环节：**存放与检索**。一个城市有数百万栋建筑、数十万条道路——「找出离事故地点 2 公里以内的救护车」这类查询，不可能扫描全表。**空间数据库（spatial database）**与**空间索引（spatial index）**是回答这类问题的工程基础。<span class="marginnote">这几乎是《数据库》课程的空间版续集：关系数据库解决「按属性查询」，空间数据库补上「按位置查询」。理解空间索引，等于理解 B+ 树的表亲们——R 树、四叉树——如何为「二维」而设计。</span>

## 1 从关系表到空间表

传统关系数据库的每一行是一条记录，字段是标量（数字、字符串）。空间数据库在行里加入**几何字段（geometry column）**——它本身是一种复杂类型（点、线、面），并可附坐标系统。<span class="marginnote">标准几何类型由 <strong>OGC（Open Geospatial Consortium）</strong>的《简单要素规范（Simple Features Specification）》定义，工业界几乎都遵守。PostGIS、Oracle Spatial、SQL Server 的空间扩展都是它的实现。</span>

**核心概念：空间查询（spatial query）**：以几何关系为条件检索数据。典型空间谓词包括：

**相交（ST_Intersects）**：找出与目标范围相交的所有要素。
**包含（ST_Contains / ST_Within）**：找出完全在某个面内的要素。
- **距离（ST_DWithin）**：找出距某点一定距离内的要素。
- **最近邻（ORDER BY 距离 / KNN）**：找出最近的 K 个要素。

空间查询可用扩展 SQL 表达，例如「找出上海市界内所有三级医院」：

```sql
SELECT h.name
FROM hospitals h, districts d
WHERE d.name = '上海市' AND ST_Contains(d.geom, h.geom);
```

这条 SQL 对用户透明，但数据库内部必须回答一个性能问题：**如何不用全表扫描就快速找到「在省界内」的医院？** 答案是空间索引。

## 2 空间索引的基本思想：最小边界框

把二维空间组织成索引，第一个直觉是用**最小边界框（MBR / bounding box）**：给每个几何对象包一个轴对齐的矩形 $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$。查询「点 P 与哪个对象相交」可以先查「P 落在哪些 MBR 里」，再对少数候选做精确几何判断。

**辨析｜易错点：** MBR 判断是「粗筛」不是「精确判定」。一个斜跨的大多边形，其 MBR 会包含很多并不在多边形内的点——所以查询要分两步：**第一步用索引粗筛出候选集，第二步对候选做精确的几何计算**。只做第一步会误报，跳过第一步会慢——两阶段是空间查询的黄金范式，与《数据库》里「索引下推 + 回表验证」异曲同工。

## 3 空间索引的三大家族

MBR 只是思想，组织 MBR 有三种主流结构：

**网格索引（grid index）**：把空间划成规则格网，记录每个格内有哪些对象。简单直观，但对象大小不均时退化。

**四叉树索引（quadtree）**：递归四分空间，叶子节点挂在对象列表。擅长点数据与自适应细分。<span class="marginnote">注意：这里的四叉树索引与第1篇第8条「四叉树编码」同名但侧重不同——那是压缩栅格，这是索引对象。思想同源：递归细分到「够用为止」。</span>

**R-树（R-tree）**：把 MBR 组织成平衡树——每个中间节点存其孩子 MBR 的并集，叶子节点指向实际对象。查询从根向下，剪掉不相交的分支。**R-树是矢量空间索引的事实标准**，PostGIS、Oracle Spatial 的默认空间索引都是它。<span class="marginnote">R-树与 B+ 树的关系是理解它的钥匙：B+ 树按一维键值分页，R-树按二维 MBR 分页；B+ 树保证扇出与平衡，R-树用「最小外接矩形」的启发式保持树的瘦高。所以 R-树的论文标题就是《R-trees: A Dynamic Index Structure for Spatial Searching》。</span>

## 4 公式解析：Z 序曲线与莫顿码

空间索引的另一种思路是把二维坐标**压平成一维键**，然后复用 B+ 树。**空间填充曲线（space-filling curve）**就是这样的映射：它用一条连续曲线扫过整个平面，让「二维相邻」尽可能对应「一维相邻」。

最常用的**Z 序（Z-order / Morton order）**把坐标按位交错。设点 $(x, y)$ 的二进制为 $x = x_1x_2x_3\dots$、$y = y_1y_2y_3\dots$，则 **莫顿码（Morton code）** $M$ 为：

$$
M = y_1x_1y_2x_2y_3x_3\dots \quad \text{（按位交错）}
$$

分三步拆解：

- **第一步，为什么交错位**：把 $x$、$y$ 的二进制位交替排列，相当于把二维坐标的每个「层级」打包进一维码。前两位 $y_1x_1$ 决定点落在整个空间四等分的哪一区，再下两位 $y_2x_2$