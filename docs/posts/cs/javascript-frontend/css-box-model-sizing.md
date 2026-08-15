---
title: 盒模型与尺寸控制
date: 2026-08-07
---

# 盒模型与尺寸控制

<div class="epigraph">
<p>在 CSS 里，万物皆盒子；盒子套盒子，最终叠出整个页面。</p>
<footer>—— 埃里克 · 迈耶（Eric A. Meyer）</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ MDN Web Docs CSS ｜ 2026-08-07</p>
</div>

## 为什么从盒模型开始

上一节解决了「样式听谁的」，这一节解决「元素占多大」。CSS 的世界里没有「自由漂浮的内容」——**每个元素都是一个矩形盒子**，盒子由内到外依次是：内容（content）、内边距（padding）、边框（border）、外边距（margin）。理解这个四层盒子，是理解布局（Flex/Grid）与尺寸控制的前提。

盒模型是整个 CSS 里**最容易算错账**的地方。一个经典困境：你写 `width: 100px`，但元素在页面上占了 150px 宽——因为它默认是 `content-box` 模型，`width` 只管内容区，padding 和 border 另算。无数「为什么布局超出了」的 bug 都源自这里。<span class="marginnote">盒模型的概念来自 CSS 规范对「视觉格式化模型」的定义：任何元素都被视为一个矩形盒子，样式引擎按 content → padding → border → margin 逐层往外画。这个模型决定了一切布局计算。</span>

## 1 盒子的四层结构

把一个盒子从内到外拆开：

**content（内容区）**：文字、图片真正占的区域，尺寸由 `width`/`height` 或内容自然撑开决定。
**padding（内边距）**：内容与边框之间的内衬，透明，背景色会延伸到它。
- **border（边框）**：包围 padding 的可见或不可见边界。
- **margin（外边距）**：盒子与相邻盒子之间的间隔，透明，**不会延伸到背景色**。

上下左右四个方向都能分别设置：`padding-top`、`padding-left`、`border-width`、`margin: 10px 20px 30px 40px`（上右下左，顺时针）。简写有记忆法：**从顶部开始顺时针转一圈**。

![CSS 盒模型四层结构：margin 在外、border 围边、padding 内衬、content 居中](/images/javascript-frontend/css-box-model-sizing-1.svg)

**margin 与 padding 的语义差异**值得单独拎出来：padding 是「盒子的内衬」，**影响背景与边框的覆盖范围**；margin 是「盒子之间的距离」，**背景不延伸进去**。给按钮加「内留白」用 padding（点按区域变大、背景变宽），给卡片之间加「空隙」用 margin（只是拉开距离）。<span class="marginnote">从行为看：padding 撑大「可点击/可看见」的区域，margin 只推动邻居。同一视觉间距，用错二者，背景色范围和 hover 热区会完全不同——这是样式细节的常见败笔。</span>

## 2 尺寸控制：width 与 box-sizing

`width`/`height` 设置内容区尺寸，但真正决定「盒子总宽」的是计算方式，由 **`box-sizing`** 属性切换：

**`content-box`**（默认）：`width` = 内容区宽，**总宽 = width + padding + border**。
**`border-box`**：`width` = 内容区 + padding + border 的总宽，**总宽 = width 本身**。

```css
.box {
  box-sizing: border-box;
  width: 200px;
  padding: 20px;
  border: 2px solid black;
}
```

`border-box` 下，内容区自动缩为 `200 - 20×2 - 2×2 = 156px`，而盒子总宽牢牢钉在 200px。<span class="marginnote">业界几乎公认：用 `border-box` 的直觉更接近人的预期——「我说这个盒子多宽，它就是多宽」。现代框架（Bootstrap、Tailwind）都默认 `border-box`，常见做法是加一条全局规则 `* { box-sizing: border-box; }`。</span>

`min-width` / `max-width` / `min-height` / `max-height` 是尺寸的「护栏」：内容太多时盒子不会无限撑破布局，用 `max-width: 100%` 让图片永不超出容器——这是响应式图像最基本的保护。`width: auto` 则是默认行为：块级元素自动撑满父容器，行内元素按内容收缩。

## 3 公式解析：盒子总宽与总高

把两种盒模型的账算清楚，就是一条**总尺寸公式**：

$$
\text{outer} = \text{inner} + \text{padding}_{\text{LR}} + \text{border}_{\text{LR}} + \text{margin}_{\text{LR}}
$$

这条式子里的四个加数分别来自盒子四层，下标 LR 表示「左右两边的合计」（上下同理）。代入两种盒模型，式子变成两副面孔：

- **content-box（默认）**：`width` 只声明内容区，于是 $\text{outer} = \text{width} + \text{padding} + \text{border} + \text{margin}$。一个 `width: 100px; padding: 20px; border: 2px solid` 的盒子，实际占据 $100 + 40 + 4 = 144px$——比你写下的数字宽了 44px。这就是「为什么布局超了」的头号元凶。
- **border-box**：`width` 已经包含 padding 与 border，内容区被自动压缩，于是 $\text{outer} = \text{width} + \text{margin}$。同样的设置下盒子总宽钉死在 100px，内容区只剩 $100 - 40 - 4 = 56px$。

**一句话记住**：content-box 是「先定内容，尺寸外溢」；border-box 是「先定总宽，内容收缩」。要让两列 `width: 50%` 并排严丝合缝，border-box 保证无论 padding 多大都不溢出——这正是它成为现代默认的原因。

## 4 margin 折叠：盒子距离怎么算

垂直方向上，两个相邻块级盒子的 margin **不会相加，而是取较大者**——这叫作 **margin 折叠（margin collapsing）**，是 CSS 里最反直觉的规则之一。

```css
.a { margin-bottom: 40px; }
.b { margin-top: 30px; }
/* .a 与 .b 之间的间距是 40px，而不是 70px */
```

为什么？规范把「块级盒子之间的垂直空隙」视为**同一段间距**，谁给得多就用谁。折叠发生的三种场合：**相邻兄弟之间**、**父盒子与第一个/最后一个子元素之间**、**空盒子的上下 margin 之间**。水平方向（`margin-left` / `margin-right`）不折叠，Flex 与 Grid 容器内部的垂直 margin 也不折叠——现代布局顺手解决了一批老问题。

**辨析｜易错点：** 想隔离折叠，用 **BFC（块格式化上下文）**——一种「内部布局与外界互不干扰」的渲染上下文。给父容器加 `overflow: hidden` 或 `display: flow-root` 就触发 BFC，子元素的 margin 从此「漏不出去」，父容器也不再被内部 margin 撑得看起来高度不对。<span class="marginnote">触发 BFC 的常见手段：`overflow: hidden`、`display: flow-root`、`float`、`position: absolute`。其中 `flow-root` 是为此而生的现代写法，语义最干净——「让这个元素自成一块格式化根」。</span>

## 5 实战：尺寸控制的三个经典场景

**场景一：水平居中。** `margin: 0 auto` 让块级元素在父容器内水平居中——前提是它有确定宽度（`width` 或 `max-width`），否则块级元素默认撑满整行，「居中」无从谈起。这是最常用的居中手法，也是「宽度与 margin 的配合」的第一次实战。

**场景二：永不溢出的图片。** `max-width: 100%; height: auto`——图片随容器收缩、保持宽高比。这行代码是响应式布局的第一道防线，也是《响应式设计》那一篇会反复使用的铁律。

**场景三：底部吸附的页脚。** `min-height: 100vh` 让页面至少占满一屏，配合 Flex 容器的 `margin-top: auto` 把页脚推到底部——内容不足一屏时页脚也在最下方，内容超高时页面自然滚动。`vh` 是视口单位，见第9节的单位体系。

**尺寸控制三原则**：宽度优先用百分比、`fr`、`vw` 等相对值（为响应式铺路）；高度尽量交给内容（写死 `height` 容易截断文本）；间距从 margin / padding 二选一时，想清楚「间距是谁的一部分」。

**百分比宽度的细节**：`width: 50%` 的「50%」指父容器内容区的 50%。父容器若 `box-sizing: content-box` 且自身带 padding，子元素的百分比要小心——父内容区变窄，子元素的 50% 也相应变窄。这一层「百分比的参照物」关系，是理解响应式栅格（第9、10节）的伏笔。

## 6 display：盒子以什么身份存在

盒子的尺寸行为，很大程度上由 **`display`** 属性决定——它决定盒子是块级、行内，还是两者的结合体。三种最常用的值：

**`block`**：占满整行，上下可设 margin/padding，`width` 默认铺满父容器。`<div>`、`<p>`、`<h1>` 默认就是它。
**`inline`**：不换行、像文字一样在行内流动，**`width` / `height` 无效**，上下 margin 也不生效。`<span>`、`<a>`、`<strong>` 默认是它。
**`inline-block`**：既在行内流动，又**保留块级盒子的尺寸能力**——`width` / `height`、上下 margin 都有效。`<img>` 天然就是 inline-block。

```css
.box { display: inline-block; width: 120px; height: 60px; margin: 10px; }
```

`inline` 与 `inline-block` 的差别，正是「行内元素能不能设宽高」这一经典困惑的答案：`<a>` 想变成可点击的大按钮，要先 `display: inline-block` 再设宽高与 padding。这解释了为什么「`width` 对 `<span>` 不生效」——不是 bug，而是它默认的行内身份让宽度属性无从谈起。<span class="marginnote">`display` 的取值远不止三种：`flex` 与 `grid` 本身就是两种 `display` 值，它们把直接子元素放进专门的布局上下文（见本专题第9、10节）；`none` 让盒子彻底不渲染，与 `visibility: hidden` 的「占位但透明」不同。</span>

## 7 小结

- 盒子四层由内到外：**content → padding → border → margin**；padding 属于盒子本身，margin 属于盒子之间。
- **content-box** 的 `width` 只管内容、总宽另加 padding+border；**border-box** 的 `width` 就是总宽、内容自动收缩——现代开发默认 border-box。
- 垂直 margin 会**折叠**（取较大者），水平方向不折叠；需要隔离时用 BFC（`overflow: hidden` / `display: flow-root`）。
- 尺寸护栏 `min-width` / `max-width` 防止内容撑破布局；`margin: 0 auto` 居中、`max-width: 100%` 防溢出是两个高频配方。
- `display` 决定盒子的身份：`block` 占行、`inline` 不占行且宽高无效、`inline-block` 兼得二者。

在下一节，我们将处理页面观感的另一半：**字体、文本排版与颜色**——盒子已经会量尺寸了，接下来让盒子里的文字读起来舒服。