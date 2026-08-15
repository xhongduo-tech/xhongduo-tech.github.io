---
title: Flex 弹性盒布局
date: 2026-08-07
---

# Flex 弹性盒布局

<div class="epigraph">
<p>完善的境界，不在无可再增，而在无可再减。</p>
<footer>—— 安托万 · 德 · 圣埃克苏佩里（Antoine de Saint-Exupéry）</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ MDN Web Docs CSS 弹性盒 ｜ 2026-08-07</p>
</div>

## 为什么从 Flexbox 开始

在前 Flex 时代，CSS 完成「水平垂直居中」要靠 `margin: auto` + 绝对定位 + 表格法……一行居中要三招，多列等高更是噩梦。2009 年，W3C 启动了 **Flexbox（弹性盒布局）** 规范，2012 年前后主流浏览器全面落地——它把「单方向上的分布与对齐」从 hack 变成了声明式能力。

**Flexbox 解决的核心问题是「一维布局」**：一根主轴上的排列——一行按钮怎么分布、一列卡片怎么对齐、导航栏怎么自动伸缩。它不负责「二维网格」（那是下一节 Grid 的战场）。判断是否用 Flexbox：**你的元素主要在一条线上排，且希望它们能灵活伸缩——用 Flex；既要行又要列、想画「表格状的网格」——用 Grid。**

## 1 主轴与交叉轴：Flex 的坐标系

给容器写 `display: flex`，它的直接子元素就变成了**弹性项目（flex item）**，进入一套以**两根轴**为基准的布局：

- **主轴（main axis）**：项目排列的方向，由 `flex-direction` 决定。
- **交叉轴（cross axis）**：与主轴垂直的方向。

**`flex-direction`** 是整套布局的「朝向开关」，四个值：

| 值 | 主轴方向 | 效果 |
| --- | --- | --- |
| `row`（默认） | 左 → 右 | 横向排列 |
| `row-reverse` | 右 → 左 | 横向反向 |
| `column` | 上 → 下 | 纵向排列 |
| `column-reverse` | 下 → 上 | 纵向反向 |

**记住一个要点**：主轴变了，`justify-content`（主轴分布）与 `align-items`（交叉轴对齐）的「作用对象」就跟着变——横向时它们管「左右」与「上下」，切成 `column` 后全部互换。Flexbox 的轴是**相对概念**，不是固定坐标。

## 2 容器属性：分布与对齐

**`justify-content`**：主轴方向的分布方式——`flex-start`（起点对齐，默认）、`flex-end`（终点）、`center`（居中）、`space-between`（两端顶满、中间均分）、`space-around`（每项两侧各留一半间距）、`space-evenly`（间距完全相等）。一行三个按钮想「两端一个、中间一个」，`space-between` 一步到位。

**`align-items`**：交叉轴方向的对齐——`stretch`（拉满容器高度，默认）、`flex-start`、`flex-end`、`center`、`baseline`（按文字基线对齐）。**`align-items: center` 就是「垂直居中」的标准答案**——在此之前，这一行代码是无数布局 hack 才换来的。

**`gap`**：项目间距。`gap: 16px` 给主轴与交叉轴都加 16px 间距；`column-gap`/`row-gap` 可分别控制。它是近年才补进 Flexbox 的属性——早年项目间距要靠 margin，还总遇到「最后一个项目多了个 margin」的尴尬。

```css
.nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}
```

**`flex-wrap`**：默认 `nowrap`，项目一行塞不下就**压缩**而不是换行；设 `wrap` 后溢出项目折行，配合 `align-content` 控制多行之间的分布。<span class="marginnote">`flex-wrap: wrap` 让 Flexbox 从「一行」升级成「多行流」——这也常被用于「自适应卡片墙」：项目宽度用 `flex-basis` 定，塞不下自动折行，不需要媒体查询就能响应容器宽度。这是 Flexbox 与「响应式」第一次握手。</span>

## 3 项目属性：谁胖谁瘦、谁先谁后

容器管整体，**项目属性**让每个子元素有「个体意志」：

**`flex-grow`**：空间富余时，项目按比例「抢」剩余空间。`flex-grow: 1` 表示「把剩余空间分给我一份」。三个项目都 `1`，剩余空间三等分；其中一个 `2`，它拿两份。

**`flex-shrink`**：空间不足时，项目按比例「缩水」。默认都是 1，大家均匀变瘦；`flex-shrink: 0` 表示「我宁死不缩」——做「永不压缩的图标」就靠它。

**`flex-basis`**：项目的「理想尺寸」，未伸未缩前的基准。可以写像素、百分比或 `auto`（默认，看内容宽度）。它取代了 `width` 在 Flex 项目里的地位——项目最终尺寸由「basis + grow/shrink 调整」共同决定。

**`flex` 简写**：`flex: 1` 等价于 `flex: 1 1 0%`——「能伸能缩、基准从 0 算起」，是「均分」的经典写法；`flex: 1 1 auto` 则是「以内容为基准再均分剩余」。初学最容易踩的坑就是分不清 `flex: 1` 与 `flex: 1 1 auto`。

**`order`**：视觉重排。`order: -1` 把项目提到最前。它只改显示顺序、不改源码顺序——读屏仍按源码读，所以 `order` 只用于「视觉微调」，不能用来颠倒「逻辑顺序」（呼应第5篇的文档顺序原则）。

**`align-self`**：单项目覆盖容器的 `align-items`——「大家都在顶部，就这一个垂直居中」。

## 4 公式解析：flex-grow 的剩余空间分配

`flex-grow` 的「按比例分空间」可以写成一条公式。设容器主轴可用空间为 $W$，项目 $i$ 的 `flex-basis` 之和为 $\sum B_i$，则剩余空间为：

$$
W_{\text{left}} = W - \sum_{i} B_i
$$

若 $W_{\text{left}} > 0$（空间富余），每个项目分到的实际宽度为：

$$
\text{width}_i = B_i + W_{\text{left}} \cdot \frac{g_i}{\sum_{k} g_k}
$$

其中 $g_i$ 是项目 $i$ 的 `flex-grow` 值。拆解：

- **$\sum_i B_i$ 是「基准账本」**：先把每个项目按 `flex-basis` 摆好，剩下的才是可分配的。
- **$\frac{g_i}{\sum_k g_k}$ 是「份额」**：把剩余空间按 grow 值加权分配——grow 之和是分母，自己的 grow 是分子。
- **直觉**：`flex-grow: 1` 的项目在均分剩余空间，`flex-grow: 2` 拿双份。全部 grow 为 0 时，剩余空间闲置在尾部，这正是 `justify-content: flex-start` 的默认观感。

举个例子：容器宽 400px，两个项目 `flex-basis` 各 100px，grow 各 1。则 $W_{\text{left}} = 400 - 200 = 200$px，每项分 $100 + 200 \times \frac{1}{2} = 200$px——两项正好各占一半。若把 grow 改成 1 和 3，则第一项 $100 + 200 \times \frac{1}{4} = 150$px、第二项 $100 + 200 \times \frac{3}{4} = 250$px——**比例决定胖瘦**，这就是「弹性」的数学内核。

## 5 核心对比表：Flex 与 Grid 怎么选

Flexbox 讲完，先把「一维 vs 二维」的边界钉清楚，下一节 Grid 会更细：

| 维度 | Flexbox | Grid |
| --- | --- | --- |
| 布局维度 | 一维（单根主轴） | 二维（行 + 列） |
| 心智模型 | 一条线上的项目流 | 一张带轨道的网格 |
| 分布控制 | `justify-content` + `align-items` | 轨道尺寸 + 区域放置 |
| 典型场景 | 导航栏、按钮组、卡片行 | 页面骨架、相册、仪表盘 |
| 项目自主 | 强（grow/shrink/basis/order） | 强（grid-area 定位） |
| 组合用法 | 可嵌在 Grid 单元格里 | 可嵌在 Flex 项目里 |

**实践共识**：页面大骨架用 Grid，骨架内的某一行、某一列再用 Flex 精细排列——两者嵌套使用，是现代布局的标准姿势。Vue/React 组件库的 `flex` 与 `grid` 工具类，也完全是这套语法的封装。

## 6 小结

- `display: flex` 激活弹性布局；**主轴/交叉轴**由 `flex-direction` 决定，两轴属性随之互换。
- 容器三件套：`justify-content`（主轴分布）、`align-items`（交叉轴对齐）、`gap`（间距）。
- 项目三件套：`flex-grow`（抢富余）、`flex-shrink`（让不足）、`flex-basis`（基准尺寸）；`flex` 简写别混淆。
- 分配公式：实际宽度 $= B_i + W_{\text{left}} \times g_i / \sum g_k$——grow 是比例，不是固定像素。
- `order` 只改视觉顺序、不改读屏顺序；`align-self` 让单个项目突围。
- **选型一句话**：一条线排布用 Flex，二维网格用 Grid——骨架 Grid、内层 Flex，嵌套是常态。

在下一节，我们把布局从「一条线」升级成「一整张网」：**Grid 网格布局**——二维轨道系统，页面骨架的终极答案。