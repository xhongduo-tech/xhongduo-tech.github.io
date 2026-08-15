---
title: HTML 文档结构与元数据
date: 2026-08-07
---

# HTML 文档结构与元数据

<div class="epigraph">
<p>万维网与其说是技术的创造，不如说是社会的创造。</p>
<footer>—— 蒂姆 · 伯纳斯-李（Tim Berners-Lee），万维网发明者</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ MDN Web Docs HTML 基础 ｜ 2026-08-07</p>
</div>

## 为什么从文档结构开始

网页的一切都要从「一张文档怎么写」说起。1991 年，蒂姆 · 伯纳斯-李在瑞士日内瓦的欧洲核子研究中心（CERN）发布了第一个网页，用的就是 **HTML（HyperText Markup Language，超文本标记语言）**。它的核心理念至今未变：**用一套约定的标记，给纯文本穿上结构**——告诉浏览器「这是标题、这是段落、这是链接」，剩下的交给浏览器去渲染。

这一节是整棵知识树的根。后面所有内容——CSS 样式、JavaScript 交互、性能、安全——都寄生在 HTML 文档这具「骨架」上。如果把网页比作一栋房子，HTML 是承重墙与房间划分，CSS 是涂料与家具，JavaScript 是水电与门禁。**先立骨架，再谈装修。**

## 1 HTML 语法基础：标签、元素与属性

一条最基本的 HTML 由三样东西构成：**标签（tag）、元素（element）、属性（attribute）**。

**标签**是带尖括号的标记，如 `<p>`、`</p>`；**元素**是「开标签 + 内容 + 闭标签」的完整组合；**属性**是写在开标签里的附加信息，用「属性名 = 值」的形式给出。

```html
<p class="intro">你好，世界。</p>
```

拆开看：`<p>` 是开标签，`class="intro"` 是属性（声明这个段落属于 `intro` 类），`你好，世界。` 是内容，`</p>` 是闭标签。**开闭标签必须成对**。浏览器对「忘写闭标签」容忍度很高——它会自己猜，但猜错的位置就是诡异的布局 bug。

**元素的两种类型**：

- **容器元素**：有开有闭、包住内容，如 `<div>`、`<p>`、`<ul>`。
- **空元素（void element）**：没有内容、没有闭标签，如 `<img>`、`<br>`、`<input>`、`<meta>`——一切信息都在属性里。

**属性的几条通用规则**：属性值一般用引号包裹（单双皆可，保持一致）；布尔属性只写属性名即可，如 `<input disabled>`；全局属性 `class` 与 `id` 对所有元素通用——`class` 可重复、是样式钩子，`id` 必须全页唯一、用于精确定位。<span class="marginnote">`id` 的唯一性不只是规范要求：`href="#xxx"` 锚点跳转、CSS 的 `#` 选择器、JS 的 `document.getElementById` 都假设 `id` 全局唯一。同一 `id` 重复出现时第一个生效、其余被无视——这是新手常踩的坑。</span>

## 2 文档骨架：DOCTYPE、html、head 与 body

一份最小的合法 HTML 文档长这样：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>我的第一个网页</title>
</head>
<body>
  <p>正文内容。</p>
</body>
</html>
```

四层骨架，各有职责：

**`<!DOCTYPE html>`（文档类型声明）**：不是标签，而是一条指令，告诉浏览器「请按标准模式（standards mode）解析我」。它是 HTML5 的唯一写法。缺了它，浏览器可能退回**怪异模式（quirks mode）**——盒模型等行为全部回到 IE 时代的怪癖，布局随之错乱。<span class="marginnote">怪异模式是历史的包袱：当年 IE 与标准差异巨大，浏览器只好「看 `<!DOCTYPE>` 决定用哪套行为」。`<!DOCTYPE html>` 一刀切让所有现代浏览器进入标准模式，这也是现代开发几乎不再谈论怪异模式的原因。</span>

**`<html lang="zh-CN">`**：文档根元素，`lang` 属性声明页面主语言。它看似不起眼，却同时服务搜索引擎、屏幕阅读器（决定用哪套语音读）、以及浏览器的自动翻译——中文页面写 `lang="zh-CN"`，英文写 `lang="en"`。

**`<head>`（头部）**：**不给用户看**，而是「关于这份文档的信息」——元数据、标题、外部资源的引入都在这里。

**`<body>`（主体）**：用户真正看到的全部内容。

## 3 元数据：head 里藏着什么

`<head>` 是「文档的说明书」，核心成员：

```html
<meta charset="utf-8">                    <!-- 字符编码，写在最前 -->
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="一篇关于 HTML 结构的入门教程">
<meta name="author" content="从极限到大模型">
<title>HTML 文档结构与元数据 - 从极限到大模型</title>
```

逐一解释：

**`charset="utf-8"`**：字符编码声明，写在 `<head>` 的第一个 `<meta>`。UTF-8 覆盖地球上几乎全部文字，中文页面漏掉它就会出现乱码——**这是全站最重要的一个 `<meta>`**。<span class="marginnote">早期中文网页用 GB2312/GBK 编码，漏声明就会看到「鎴戠殑缃戝」这类乱码。2010 年后 UTF-8 一统天下：它向后兼容 ASCII、可变长编码、中文只占 3 字节。规范要求 `charset` 必须在文档前 1024 字节内出现。</span>

**`viewport`**：移动端布局的开关。`width=device-width` 让页面按「设备物理宽度」渲染，`initial-scale=1` 设初始缩放为 1——没有它，手机浏览器会把桌面版网页整体缩成邮票再让你放大，这正是响应式设计要解决的第一件事（详见本专题《响应式设计与媒体查询》）。

**`description`**：页面的一句话简介，搜索引擎把它显示在搜索结果标题下方；写得好坏直接影响点击率，一般 120–160 个字符。

**`<title>`**：浏览器标签页的文字，也是搜索结果的大标题——**每个页面都该有且只有一句 `<title>`**。

## 4 核心对比表：head 与 body 的分工

「哪些东西放 `head`、哪些放 `body`」是初学者最容易纠结的分界。核心判据只有一条：**用户看得见的放 `body`，看不见、但关于文档的放 `head`**。

| 元素 | 放哪 | 用户可见？ | 职责 |
| --- | --- | --- | --- |
| `<title>` | head | 标签页文字 | 文档标题，搜索大标题 |
| `<meta charset>` | head | 否 | 字符编码声明 |
| `<meta description>` | head | 否（搜索摘要可见） | 页面简介 |
| `<meta viewport>` | head | 否 | 移动端视口控制 |
| `<link rel="stylesheet">` | head | 否 | 引入 CSS |
| `<style>` | head | 否 | 内嵌样式块 |
| `<h1>` | body | 是 | 页面主标题 |
| `<p>`、`<img>`、`<a>` | body | 是 | 正文内容 |

**辨析｜易错点：** `<h1>` 与 `<title>` 不是一回事——`<title>` 是浏览器标签页与搜索结果的标题，`<h1>` 是页面里用户看到的大标题。一份规范文档二者内容往往相近但各有定位：`<title>` 面向「搜索与标签页」，`<h1>` 面向「页面正文的大纲起点」。

## 5 连接外部资源：link、favicon 与脚本

`<head>` 里最常见的连接工作有两类：样式与图标。

```html
<link rel="stylesheet" href="style.css">
<link rel="icon" href="/favicon.ico" type="image/x-icon">
<link rel="preconnect" href="https://fonts.googleapis.com">
```

**`<link rel="stylesheet">`**：引入外部 CSS，`rel`（relationship）声明「这是样式表」。CSS 放在 `<head>` 而 HTML 在 `<body>`，是为了让样式先就位、避免「无样式内容闪现（FOUC）」——用户看到裸排版的内容一闪而过。

**`<link rel="icon">`**：站点图标（favicon），显示在标签页与书签旁。现代站点常用多尺寸 PNG 图标，甚至直接用 SVG。

**`<link rel="preconnect">`**：性能预热——提前与第三方域名建立连接，减少字体、脚本的加载等待。这是本专题《Web 性能优化基础》的前哨。

**脚本放哪？** 传统做法是 `<script>` 放在 `<body>` 末尾：脚本执行会**阻塞**页面解析，放在末尾能让正文先渲染。现代做法则是给 `<head>` 里的脚本加 `defer`（HTML 解析完再执行）——既保住「先正文」体验，又符合依赖顺序。这一套「阻塞 / 非阻塞」机制，在第4篇与性能篇会反复出现。

## 6 小结

- HTML 的三件套：**标签、元素、属性**；容器元素成对闭合，空元素（`<img>`、`<br>`、`<meta>`）无闭标签。
- 文档骨架四层：`<!DOCTYPE html>`（标准模式）→ `<html lang="zh-CN">` → `<head>`（关于文档的信息）→ `<body>`（可见内容）。
- 元数据三宝：**`charset="utf-8"`（防乱码）、`viewport`（移动端开关）、`description`（搜索摘要）**；`<title>` 是标签页与搜索大标题。
- 分界判据一句话：**用户看得见的放 `body`，看不见的关于文档的信息放 `head`**。
- `<link>` 连接外部资源：样式表、favicon、`preconnect` 性能预热；脚本用 `defer` 平衡「先渲染」与「执行顺序」。

在下一节，我们开始填 `body` 的内容：**文本、链接与图像元素**——从「一张空白骨架」走向「第一个真正可看、可点的页面」。