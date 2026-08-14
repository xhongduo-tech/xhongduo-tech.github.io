---
title: Web 标准、浏览器兼容性与特性检测
date: 2026-08-07
---

# Web 标准、浏览器兼容性与特性检测

<div class="epigraph">
<p>Web 之所以伟大，是因为它不归任何一家公司所有——它归一套共同承诺的标准所有。</p>
<footer>—— 蒂姆 · 伯纳斯-李（Tim Berners-Lee）</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ MDN Web Docs ｜ 2026-08-07</p>
</div>

## 为什么从 Web 标准收束

全专题从 HTML 讲到了安全，是时候回答一个「地基级」问题：**这些语法是谁定的？为什么有的浏览器支持、有的不支持？写代码怎么应对差异？** 这一节把「标准 → 浏览器实现 → 兼容策略」整条链讲清。

Web 标准的特殊之处在于**多方博弈**：规范由 W3C/WHATWG/ECMA 制定，但 Chrome、Safari、Firefox 各自实现——实现有快慢、有取舍、有差异。前端工程师的日常是「**写一套代码，跑在 N 种浏览器上**」——兼容性不是选修课，是 Web 开发的默认约束。而**特性检测**（feature detection）是应对差异的标准姿势：不问「你是哪个浏览器」，只问「你支持这个功能吗」。<span class="marginnote">Web 标准组织的分工：<strong>WHATWG</strong>（HTML/DOM/Fetch 等，浏览器厂商主导）、<strong>W3C</strong>（CSS、SVG、ARIA 等）、<strong>ECMA TC39</strong>（ECMAScript/JS 语言）。MDN Web Docs 是这三者的权威参考合辑——所以本专题的对标教材是 MDN。</span>

## 1 标准从哪来：规范与实现的两条线

一条新特性从想法到能用，走一条「规范 + 实现」双轨：

```
提案 → 规范草稿 → 规范定稿
     ↘ 浏览器实验实现 → 稳定实现 → 全浏览器支持 → "可以放心用了"
```

**规范（specification）**：描述「应该怎样」的文字——语法、行为、边界。它是契约，不是代码。

**实现（implementation）**：浏览器真正写出来的代码。同一规范，各浏览器实现可能有细微差异——这就是「兼容性 bug」的源头。

**兼容性数据从哪查**——两个权威工具：

**MDN 兼容性表**：每个 API 页面底部有浏览器支持矩阵（Chrome/Firefox/Safari/Edge × 版本号）。
**caniuse.com**：输入特性名，看全球支持率与版本门槛。

**支持率的现实**：一个特性「全浏览器支持」需要 Chrome、Firefox、Safari 都稳定落地——通常要几年。写代码时查一下「我想用的 API 支持到哪个版本」，是每个前端的基本习惯。<span class="marginnote"><strong>Babel</strong> 的浏览器目标（browserslist）就是「按支持率决定转译多少」：`"browserslist": ["> 0.5%, not dead"]` 表示「支持全球使用率超 0.5% 且仍在维护的浏览器」——构建工具据此决定哪些语法要降级（第26篇的转译在此有了明确目标）。</span>

## 2 兼容性的三层应对：检测、降级、垫片

遇到「想用但部分浏览器不支持」的特性，标准答案是**三层组合拳**：

**1. 特性检测（feature detection）**——先问「支持吗」，再决定用不用：

```js
if ("localStorage" in window) {
  useLocalStorage();
} else {
  useFallback();      // 不支持就降级
}
```

**`in` 操作符**检查属性是否存在。更精确的「真的可用吗」检测：

```js
const supportsJSON = typeof JSON === "object" && typeof JSON.parse === "function";
```

**特性检测的黄金法则**：**检测「特性」本身，不检测「浏览器」**——不要写 `if (navigator.userAgent.includes("Chrome"))` 猜浏览器。浏览器能改 userAgent、会有新版本、Chrome 能装插件改行为——猜浏览器极不可靠，测特性才贴近「你到底能不能用」。

**2. 渐进增强（progressive enhancement）**——**先保证基本功能人人可用，再在支持的浏览器上增强**：

```html
<!-- 基础：所有浏览器都能提交 -->
<form action="/search">
  <input type="search" name="q">
  <button>搜索</button>
</form>
<script>
  // 增强：支持的浏览器上，改成异步即时搜索
  if (window.fetch) { form.addEventListener("submit", async (e) => { … }); }
</script>
```

**优雅降级（graceful degradation）** 是反向的：先做高级版，再为老浏览器兜底。现代实践更推荐**渐进增强**——以「最差环境可用」为底线往上加。

**3. Polyfill（垫片）**——用 JS 把「缺失的 API」补上：

```js
// 老浏览器没有 Array.prototype.includes，用脚本补一个
if (!Array.prototype.includes) {
  Array.prototype.includes = function (search) {
    return this.indexOf(search) !== -1;
  };
}
```

**Polyfill vs 转译（transpile）的分工**：

| 手段 | 解决什么 | 例子 |
| --- | --- | --- |
| 转译 | **语法**层面 | 箭头函数 → function、class → 老写法 |
| Polyfill | **API** 层面 | `Array.includes`、`fetch`、`Promise` |

语法降级用 Babel 转译；新 API 用 polyfill 补。两者配合，现代语法才能在老浏览器跑通——`core-js` 是工业级 polyfill 集，构建工具通常自动引入。

**辨析｜易错点：** Polyfill 不是万能的——`Proxy`、`WeakRef` 这类「改变引擎语义」的 API 无法 polyfill（补不了引擎级能力）。**Polyfill 也有体积与性能成本**——只引入需要的（`core-js` 支持按特性引），别把整包塞进去。判断标准：**补得了的补，补不了的降级**。<span class="marginnote">「语法能转译、API 能垫片、引擎级补不了」——这个三层判断是兼容性决策的核心。`Proxy` 补不了，所以 Vue 3 用 Proxy 实现响应式后，明确放弃了对 IE11 的支持——「新特性驱动的底线提升」是行业常态。</span>

## 3 特性检测 vs 浏览器嗅探：一条边界

「怎么判断环境」是兼容性的分水岭，两种姿势天差地别：

| 维度 | 特性检测 | 浏览器嗅探 |
| --- | --- | --- |
| 依据 | 「这个功能在不在」 | 「userAgent 是谁」 |
| 可靠度 | 高（测真实能力） | 低（可伪造、会过时） |
| 维护 | 新浏览器自动适配 | 每个新版本/新浏览器都要更新 |
| 典型 | `if ("fetch" in window)` | `if (ua.includes("Chrome"))` |

**嗅探为什么坏**：`navigator.userAgent` 是浏览器自称的身份——可被改、会随版本变、还可能被插件污染。写 `ua.includes("Chrome")` 等于「靠身份证照片认人」——证件能造假，功能测试才是「当面考他」。

**唯一的嗅探例外**：极少数「按浏览器必须差异化」的场景（如已知某浏览器特有的 bug），也要**缩到最小范围**并注释原因。默认路径永远是特性检测。

**`@supports`** 是 CSS 的特性检测——只应用「支持某属性」时的样式：

```css
@supports (display: grid) {
  .layout { display: grid; }
}
```

`@supports` 让 CSS 也能「渐进增强」：不支持 Grid 的浏览器跳过这段、用兜底布局。<span class="marginnote">CSS 的 `@supports` 与 JS 的 `if ("x" in window)` 是同一思想在两个语言的体现：<strong>能力探测</strong>。配合「先写兜底、再写增强」的顺序，即使不支持也能优雅降级——这是响应式与兼容性共用的优雅哲学。</span>

## 4 前缀与自动处理：兼容的工程化

**供应商前缀（vendor prefix）** 是历史的产物：规范未定稿时，浏览器先实现自己的版本并加前缀区分：

```css
.example {
  -webkit-transform: rotate(45deg);  /* WebKit（Chrome/Safari 系）早期 */
  -ms-transform: rotate(45deg);      /* IE/Edge 早期 */
  transform: rotate(45deg);          /* 标准 */
}
```

前缀的尴尬：规范定稿后，标准写法与带前缀写法并存，且「该写哪个前缀」随版本漂移——手写前缀是兼容性的重灾区。

**工程解法**：**Autoprefixer**——构建时根据 caniuse 数据自动加/删前缀：

```js
// 你只写标准写法
.example { transform: rotate(45deg); }
// Autoprefixer 按目标浏览器自动产出带前缀版本
```

**结论：别手写前缀，交给工具**。现代 CSS 属性大多已标准化，前缀需求逐年减少——但「工具自动处理」的习惯仍是最稳的。

**辨析｜易错点：** 兼容性决策的三个「别」——

1. **别写死浏览器版本**：目标列表用 browserslist（百分比、版本范围），而不是「Chrome 80+」写死。
2. **别过度兼容**：为目标用户群定底线——支持 IE11 要付出巨大成本，明确「要不要支持」是产品决策不是技术决策。
3. **别「看着能跑」就放行**：用真实浏览器（或 BrowserStack 之类云真机）验证，而不是只在 Chrome 里看一眼。<span class="marginnote">「目标浏览器底线」决定了你的语法选择、polyfill 成本、测试矩阵。2020 年代的主流底线是「最近两个主版本 + Safari 全系 + 份额超 0.5%」——在此之上，ES2020+ 语法、`clamp()`、Grid 都敢直接用（配合转译与 Autoprefixer）。</span>

## 5 公式解析：特性检测的决策树

兼容性的每一次「用不用某特性」的决策，都能套进一棵**决策树**——它是特性检测的完整模型：

$$
\text{use}(F) = \begin{cases} \text{直接用} & \text{if } \text{supported}(F) \wedge \text{baseline} \ge \text{threshold} \\ \text{polyfill} & \text{if } \text{patchable}(F) \\ \text{降级} & \text{otherwise} \end{cases}
$$