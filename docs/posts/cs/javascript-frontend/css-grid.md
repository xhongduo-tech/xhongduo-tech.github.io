---
title: Grid 网格布局
date: 2026-08-07
---

# Grid 网格布局

<div class="epigraph">
<p>网格系统是一种辅助，而非保证——它允许无数种可能的用法，每个设计者都可以找到契合自己风格的答案。</p>
<footer>—— 约瑟夫 · 米勒-布罗克曼（Josef Müller-Brockmann），《平面设计中的网格系统》</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ MDN Web Docs CSS 网格布局 ｜ 2026-08-07</p>
</div>

## 为什么从 Grid 开始

上一节的 Flexbox 是「一条线」上的布局大师，但真实页面是**二维**的：正文、侧栏、页头、页脚构成一张棋盘，相册是一排排缩略图。在 Grid 出现之前，页面骨架靠 `float` + 清除浮动、表格布局等一套「用错误工具做正确的事」的 hack——1996 年的表格布局、2010 年代的 float 栅格，都是时代的补丁。

**CSS Grid Layout（网格布局）**在 2017 年前后随主流浏览器全面落地，第一次把「二维网格」变成 CSS 的原生能力：**你声明轨道，浏览器负责排布**。它的心智模型很直白——在容器上画出「列轨道 + 行轨道」这张网，再把项目放进网里的指定区域。学完它，页面骨架将首次不再依赖任何 hack。

## 1 网格的三个概念：轨道、线与区域

`display: grid` 激活网格容器，其直接子元素成为**网格项目（grid item）**。理解三个基础概念：

**轨道（track）**：一行或一列的「跑道」，由 `grid-template-columns`（列轨道）与 `grid-template-rows`（行轨道）定义。

**网格线（grid line）**：轨道之间的分界线，**从 1 开始编号**。三列网格有 4 条垂直线：线 1、线 2、线 3、线 4。项目用「从线几到线几」来定位——这是 Grid 放置的核心语言。

**网格区域（grid area）**：四条线围出的矩形，是项目的「落脚处」。项目默认按源码顺序自动放进网格（自动放置），但也可以精确指定区域。

```css
.container {
  display: grid;
  grid-template-columns: 1fr 2fr 1fr;  /* 三列，宽度 1:2:1 */
  grid-template-rows: auto 1fr auto;   /* 三行：页头、主体、页脚 */
}
```

## 2 定义轨道：fr、repeat 与 gap

**`fr`（fraction，分数）**是 Grid 的「弹性单位」：`1fr` 表示「把剩余空间分一份」。`1fr 2fr 1fr` 三列按 1:2:1 分——`fr` 与 Flexbox 的 `flex-grow` 异曲同工，但它是**轨道**层面的弹性，写起来更干净。

**`repeat(n, 值)`** 避免重复书写：`repeat(3, 1fr)` 等于 `1fr 1fr 1fr`；`repeat(auto-fill, minmax(200px, 1fr))` 是「自动装满」——列数随容器宽度增减，这是响应式网格的一行式答案。

**`gap`**：轨道间距，`gap: 20px` 行列都加，`column-gap`/`row-gap` 分别控制。**注意 Grid 的 `gap` 不折叠**——它永远是你写的大小，与 Flex 里的 `gap` 一致。

**尺寸关键字**：`auto`（按内容）、`minmax(min, max)`（区间）、`1fr`（弹性）、固定 px。写 `minmax(200px, 1fr)` 的意思是「至少 200px，能多则多」——卡片墙不塌、不留大洞。

## 3 放置项目：grid-column、grid-row 与命名区域

项目按源码自动流进网格，但骨架布局通常需要「指定位置」。两个定位语法：

**按线定位**：`grid-column: 1 / 3` 表示「从第 1 条线到第 3 条线」，跨越两列；`grid-row: 1 / 2` 占一行。**`span` 写法** `grid-column: span 2` 表示「横跨 2 列」，不关心起点。经典页面骨架：

```css
header { grid-column: 1 / 4; }   /* 页头横跨三列 */
nav    { grid-column: 1 / 2; }   /* 侧栏占第一列 */
main   { grid-column: 2 / 3; }   /* 正文占第二列 */
footer { grid-column: 1 / 4; }   /* 页脚横跨三列 */
```

**按命名区域定位**：`grid-template-areas` 用 ASCII 图直接画出骨架，再用 `grid-area` 把项目对号入座——**布局意图一目了然**，是团队协作的福音：

```css
.container {
  display: grid;
  grid-template-columns: 200px 1fr;
  grid-template-areas:
    "header header"
    "nav    main"
    "footer footer";
}
nav    { grid-area: nav; }
main   { grid-area: main; }
header { grid-area: header; }
footer { grid-area: footer; }
```

`grid-template-areas` 里的每个字符串是一行，列名对应轨道列；同一名字出现多次就是「合并单元格」。改骨架只需改 ASCII 图，不用改项目定位——这是 Grid 最优雅的抽象。<span class="marginnote">`grid-template-areas` 的 ASCII 图是「先图纸后施工」的典范：一眼看出「页头横跨、侧栏在左、正文居中」。响应式时只需换一张 ASCII 图（比如移动端把侧栏移到正文下方），项目代码零改动——这就是「布局即数据」的威力。</span>

## 4 隐式轨道与 auto-fit

项目多到轨道不够用时，Grid 会**自动创建新轨道**——这些自动生成的叫**隐式轨道（implicit track）**，默认 `auto`（按内容撑开），可用 `grid-auto-rows: 1fr`、`grid-auto-columns` 指定它们的尺寸。

**`auto-fit` 与 `auto-fill`**：`repeat(auto-fit, minmax(200px, 1fr))` 是响应式卡片墙的标配。区别微妙：`auto-fill` 倾向「保留空轨道数」，`auto-fit` 倾向「把空轨道折叠、让现有卡片拉宽」。做卡片墙用 `auto-fit`，想要「固定列数不塌」用 `auto-fill`——两者配合 `minmax` 是「无媒体查询响应式」的核心配方（呼应第11篇《响应式设计与媒体查询》的「断点越少越好」）。

**`align-items` / `justify-items`**：Grid 里项目在单元格内的对齐，`stretch` 默认拉满、`center` 居中；容器侧还有 `align-content`/`justify-content` 管「轨道组整体」的分布。这些与 Flex 的对应属性名几乎一致，但作用对象从「项目」换成了「轨道/单元格」——别混淆。

## 5 公式解析：fr 的剩余空间分配

`fr` 的分配与 Flexbox 的 `flex-grow` 是同一套数学。设容器内容宽为 $W$，各列轨道中「固定部分」（px、auto、minmax 的 min）合计为 $F$，gap 合计为 $G$，则剩余空间为：

$$
W_{\text{fr}} = W - F - G
$$

所有 `fr` 轨道按比例瓜分 $W_{\text{fr}}$，第 $i$ 条轨道宽：

$$
\text{col}_i = F_i + W_{\text{fr}} \cdot \frac{f_i}{\sum_k f_k}
$$

- **$F_i$**：该轨道自己的固定部分（没有则 0）。
- **$\frac{f_i}{\sum_k f_k}$**：`fr` 值的份额——`1fr 2fr 1fr` 里，`2fr` 那列拿剩余空间的一半。
- **直觉**：先满足固定轨道与间距，再按比例分弹性——这就是为什么「`1fr` 是剩余空间的一份」而不是「总宽的百分比」。

例：容器宽 400px，gap 合计 20px，轨道 `100px 1fr 2fr`。则 $W_{\text{fr}} = 400 - 100 - 20 = 280$px，`1fr` 列得 $280 \times \frac{1}{3} \approx 93$px，`2fr` 列得 $280 \times \frac{2}{3} \approx 187$px。**`fr` 永远只在「扣完固定量」后的剩余里分**——写 `grid-template-columns: 300px 1fr` 时，侧栏固定 300px，正文拿光剩余，这是「固定侧栏 + 弹性正文」的标准骨架。

## 6 小结

- Grid 是**二维**布局：`grid-template-columns`/`grid-template-rows` 定义轨道，`gap` 定间距。
- `fr` 是轨道级弹性单位，分配公式：$W_{\text{fr}} = W - F - G$，`fr` 按份额分剩余。
- 放置项目两种语法：**按线**（`grid-column: 1 / 3`、`span 2`）与**命名区域**（`grid-template-areas` + `grid-area`）。
- 隐式轨道自动补齐；`auto-fit`/`auto-fill` + `minmax(200px, 1fr)` 是「无断点响应式卡片墙」。
- **选型**：骨架用 Grid、骨架内一行用 Flex——二者嵌套是标准姿势（见上一节对比表）。
- 对齐属性在 Grid 里作用于「轨道/单元格」，别与 Flex 的「项目」混淆。
- 项目默认按源码顺序自动放置；`grid-auto-flow` 可改自动排布方向（默认 row，可设 column / dense 补洞）。

在下一节，我们让页面适配一切屏幕：**响应式设计与媒体查询**——Grid 与 Flex 提供了弹性布局，媒体查询则负责「关键的形态切换」。