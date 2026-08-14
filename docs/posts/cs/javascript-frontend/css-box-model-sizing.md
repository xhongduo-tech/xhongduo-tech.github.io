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