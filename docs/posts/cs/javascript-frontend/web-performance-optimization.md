---
title: Web 性能优化基础
date: 2026-08-07
---

# Web 性能优化基础

<div class="epigraph">
<p>性能是用户体验的货币：每慢一毫秒，用户都以「下次不来」投票。</p>
<footer>—— 塔米 · 埃弗茨（Tammy Everts）</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ MDN Web Docs 性能 ｜ 2026-08-07</p>
</div>

## 为什么从性能优化开始

代码能跑、测试能过，但**用户体感慢**——一切白搭。性能是前端工程「体验」的最后一环，也是搜索引擎排名（Core Web Vitals 是 Google 排名信号）的硬指标。这一节建立性能的完整认知：**衡量什么**（核心指标）、**加载为什么慢**（关键渲染路径）、**如何优化**（加载/渲染/资源三层）。

性能优化不是「堆技巧」，而是**分层治理**：构建阶段（压缩、摇树、分割——第26篇）、加载阶段（缓存、CDN、预加载）、渲染阶段（布局抖动、合成）。每个阶段有各自的抓手，而衡量的指标决定优化的方向——**先测再优，别凭感觉**。<span class="marginnote">性能的本质是「资源经济学」：带宽、CPU、时间都是稀缺资源。优化的每一招，本质都是在「少用资源」或「把资源用到刀刃上」之间做取舍。理解资源流向，就理解了所有优化技巧为什么存在。</span>

## 1 核心指标：Core Web Vitals

**Core Web Vitals** 是 Google 定义的三个「以用户为中心」的指标，覆盖「加载、交互、稳定」三个维度：

| 指标 | 全称 | 测量什么 | 好（绿） |
| --- | --- | --- | --- |
| LCP | Largest Contentful Paint | 最大内容元素何时可见 | ≤ 2.5s |
| INP | Interaction to Next Paint | 交互到下一次绘制的延迟 | ≤ 200ms |
| CLS | Cumulative Layout Shift | 布局累计偏移量 | ≤ 0.1 |

**LCP**：加载体验——页面「主要内容」（大图、大标题）出现的时刻。之前用 FCP（首屏第一个像素）衡量，LCP 更贴近「用户觉得页面好了没」。
**INP**：交互体验——用户点击/按键到页面响应的时间（取代老的 FID）。>200ms 用户就感到「卡」。
- **CLS**：稳定性——元素加载后突然跳动（图加载、广告插入）的累计量。第2篇说的「图片预声明宽高」就是防 CLS。

**为什么这三个？** 它们各自回答一个用户问题：「页面出来了吗？」「能立刻点吗？」「会不会跳来跳去？」——**加载、交互、稳定**，是体感的三大维度。Lighthouse 会给出这三个 + 更多指标的诊断与优化建议。<span class="marginnote"><strong>Lighthouse</strong> 是 Google 的开源审计工具：给页面打「性能/可访问性/SEO/最佳实践」分，并逐条给优化建议。Chrome DevTools 的 Lighthouse 面板一键可跑。优化的第一动作永远是「先跑一次 Lighthouse 看基线」。</span>

## 2 关键渲染路径：页面是怎么画出来的

浏览器把 URL 变成像素的路径叫**关键渲染路径（critical rendering path）**：

```
HTML → DOM
CSS  → CSSOM
DOM + CSSOM → 渲染树 → 布局(layout) → 绘制(paint) → 合成(composite)
```

每步都是「上游决定下游」：

1. **解析 HTML → DOM 树**；解析 CSS → CSSOM 树。
2. **合并成渲染树**：只含「会显示的节点」。
3. **布局（layout/reflow）**：计算每个节点的大小位置。
4. **绘制（paint）**：画像素。
5. **合成（composite）**：把图层拼到屏幕上。

**阻塞点在哪？**

**CSS 是渲染阻塞**：CSSOM 没建好，渲染树无法生成——CSS 文件下载解析完之前，页面白屏。
**JS 是解析阻塞**：`<script>` 在执行时暂停 HTML 解析——脚本越大、放越靠前，首屏越慢。
- **JS 还可能操作 DOM/CSSOM**：改变布局 → 重新 layout/paint。

**`<script>` 的两个救兵**（第1篇见过，此处讲透）：

```html
<script defer src="app.js"></script>   <!-- 下载不阻塞，HTML 解析完再执行 -->
<script async src="ads.js"></script>   <!-- 下载不阻塞，下载完立即执行（顺序无关） -->
```

- **`defer`**：HTML 解析完、DOMContentLoaded 前执行——**保序**，适合主脚本。
- **`async`**：下载完就执行——**不保序**，适合独立脚本（广告、统计）。<span class="marginnote">为什么 JS 放 `<body>` 底部不够了？`defer`/`async` 让「下载」与「HTML 解析」并行——脚本 1MB 的话，底部方案要多等一次下载时间，defer 则下载期间解析并行。现代构建产物一个入口脚本，配 `defer` 是标配。</span>

## 3 加载优化：让资源又快又少

**「快」与「少」是两个抓手**——资源加载快的策略：让文件小、让请求少、让缓存多。

**让文件小（构建阶段）**：

压缩（minify）+ tree-shaking + 代码分割（第26篇）。
图片压缩 + 现代格式（WebP/AVIF）。
- gzip / brotli 传输压缩（服务器开）。

**让请求少**：

- 打包合并（构建做）。
- **HTTP/2 多路复用**：一个连接并发多个请求，弱化了「合并文件」的必要（反而可能拆小更优）。
- **CDN（内容分发网络）**：静态资源部署到离用户近的边缘节点——地理距离决定网络延迟，CDN 把「绕地球」变成「隔壁机房」。

**让缓存多**：

- 静态资源加**内容哈希文件名**（第26篇）——变则更新、不变则命中缓存。
- 设 **`Cache-Control`** 响应头：`Cache-Control: max-age=31536000` 一年缓存。

**预加载（preload / preconnect）**：

```html
<link rel="preload" href="font.woff2" as="font">   <!-- 提前下载关键资源 -->
<link rel="preconnect" href="https://api.example.com">  <!-- 提前建立连接 -->
```

- **`preload`**：告诉浏览器「这个资源很关键，现在就开始下载」——用于字体、首屏大图。
- **`preconnect`**：提前完成 DNS/握手——用于「马上要请求的跨域源」（API、CDN）。

**辨析｜易错点：** 缓存与更新的矛盾——「缓存越多越快」但「用户看到的可能是旧版」。解法是**内容哈希**：文件名含内容哈希，内容变哈希变 → 新 URL 自然绕过缓存；内容不变哈希不变 → 永久命中缓存。**「变则失效、不变则复用」**，两头都赢。<span class="marginnote">图片加载的现代三件套：`loading="lazy"`（视口外图片延迟加载）、`decoding="async"`（解码不阻塞渲染）、`srcset`/`sizes`（按设备选分辨率）。三件都是「用户没看见的，别急着花流量」。</span>

## 4 渲染优化：别让主线程抖

资源都加载完了，交互还卡——问题出在**渲染**。核心原则只有一条：**别频繁触发 layout 与 paint**。

**布局抖动（layout thrashing）**：JS 在「读布局属性」与「写布局属性」之间反复横跳，每次写都强制重新布局：

```js
// 抖动：读-写-读-写，每轮都触发重排
for (const box of boxes) {
  const w = box.offsetWidth;   // 读 → 强制布局
  box.style.width = w * 2 + "px";  // 写 → 下次读又强制布局
}
```

**优化：批量读写**——先全部读、再全部写；或用 `requestAnimationFrame` 把写聚到一帧。

**合成（composite）友好**：第12篇说过——**只动画 `transform`/`opacity`**，它们只走合成层、不触发 layout/paint，能在 GPU 上跑 60fps。改 `width`/`top`/`left` 则每帧都 layout+paint，主线程必然卡。

**减少重绘面积**：固定高度、避免大范围 `box-shadow` 与滤镜动画。

**长任务（long task）**：主线程被一段 >50ms 的同步 JS 占住，交互就「没反应」。对策：

大计算拆块（`setTimeout`/`requestIdleCallback`）。
大量 DOM 操作合并（用 `DocumentFragment` 一次性插入）。
- Web Worker 跑重计算（并行线程，不占主线程）。<span class="marginnote">`requestIdleCallback` 把「不紧急的任务」排到浏览器空闲时跑——适合统计上报、非关键预计算。配合第19篇的事件循环心智：主线程是有预算的，长任务就是「超预算占用」，让用户的点击排不上队。</span>

## 5 公式解析：LCP 的构成

LCP 的优化最怕「盲人摸象」——先看清它由什么组成。一条**公式**把 LCP 拆成四个累加的阶段：

$$
\text{LCP} = T_{\text{TTFB}} + T_{\text{resource}} + T_{\text{render}} + T_{\text{largest}}
$$