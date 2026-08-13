---
pageClass: plain-doc
---

# 数据可视化与科学可视化

以 Tamara Munzner《Visualization Analysis and Design》、Alexandru Telea《Data Visualization: Principles and Practice》与陈为《数据可视化》为纲，系统覆盖从数据抽象、视觉编码与布局设计，到科学可视化（标量/矢量/张量/体场）、信息与大数据可视化、交互设计与可视分析的完整方法体系。学完这些章节，就写完了数据可视化与科学可视化的全部内容。

## 对标教材

- Tamara Munzner, "Visualization Analysis and Design" (CRC Press, 2015)
- Alexandru C. Telea, "Data Visualization: Principles and Practice" (CRC Press, 2nd ed., 2014)
- 陈为 等, 《数据可视化》(电子工业出版社)

## 主题规划

<ProgressGrid cat="cs/data-visualization" />

### 第1篇

- [x] [什么是数据可视化：科学可视化与信息可视化](./what-is-data-visualization)（Munzner 第1章）
- [x] ["是什么—为什么—怎么做"框架与四层嵌套模型](./what-why-how-framework)（Munzner 第1、4章）
- [x] [数据抽象：数据类型、属性与数据集组织](./data-abstraction)（Munzner 第2章）
- [x] [任务抽象：从用户问题到可视化任务](./task-abstraction)（Munzner 第3章）
- [x] [视觉感知与可视化认知：格式塔原则与色彩感知](./perception-and-cognition)（Telea 第3章、陈为 第2章）

### 第2篇

- [x] [从数据到可视化：数据变换与可视化管线](./visualization-pipeline)（Telea 第2章）
- [x] [标记与通道：数据属性到图形元素的映射](./marks-and-channels)（Munzner 第5章）
- [x] [设计法则：视觉通道的有效性排序与经验规则](./design-principles)（Munzner 第6章）
- [x] [表格数据布局：条带、堆叠与分区矩阵](./tabular-layout)（Munzner 第7章）
- [x] [空间数据与地理数据布局：投影与几何映射](./spatial-geographic-layout)（Munzner 第8章）
- [x] [网络与树数据的布局](./network-tree-layout)（Munzner 第9章）
- [x] [颜色映射：顺序、发散与定性色带设计](./color-mapping)（Munzner 第10章）

### 第3篇

- [x] [标量场可视化：色图、高度图与等值线](./scalar-field-visualization)（Telea 第5章、陈为 第5章）
- [x] [等值面提取：Marching Cubes 与 Marching Tetrahedra](./isosurface-extraction)（Telea 第5章）
- [x] [矢量场可视化：箭头、流线、流面与流体积](./vector-field-visualization)（Telea 第6章）
- [x] [矢量场拓扑：临界点分类与拓扑骨架](./vector-field-topology)（Telea 第6章）
- [x] [张量场可视化](./tensor-field-visualization)（Telea 第7章）
- [x] [体数据可视化：体绘制管线与传递函数](./volume-rendering)（Telea 第8章）
- [x] [点云与粒子数据可视化](./point-cloud-visualization)（Telea 第9章）

### 第4篇

- [x] [高维数据可视化：散点图矩阵、平行坐标与降维](./high-dimensional-visualization)（Telea 第10章、陈为 第11章）
- [x] [层次数据可视化：树图、旭日图与嵌套圆圈](./hierarchical-visualization)（Telea 第11章、陈为 第8章）
- [x] [网络与图数据可视化：力导向布局与图布局算法](./network-graph-visualization)（Telea 第12章、陈为 第8章）
- [x] [时变数据可视化：时间线、动画与流图](./time-varying-visualization)（陈为 第7章）
- [x] [文本数据可视化：词云、文档嵌入与主题流](./text-visualization)（陈为 第9章）
- [x] [大数据可视化：采样、聚合与多分辨率层次细节](./big-data-visualization)（Telea 第15-16章）

### 第5篇

- [x] [交互与动画：刷选、缩放、联动与数据叙事](./interaction-and-animation)（Telea 第13章）
- [x] [多视图与聚焦+上下文：分面、导航、过滤与变形](./multiple-views-focus-context)（Munzner 第11-14章）
- [x] [可视分析：探索式数据分析与可视数据挖掘](./visual-analytics)（陈为 第3章、Telea 第14章）
- [x] [可视化评估与验证：四层模型与用户研究](./evaluation-and-validation)（Munzner 第4章）

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
