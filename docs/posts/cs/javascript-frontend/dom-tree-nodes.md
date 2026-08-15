---
title: DOM 树结构与节点操作
date: 2026-08-07
---

# DOM 树结构与节点操作

<div class="epigraph">
<p>任何足够先进的技术，都与魔法无异。</p>
<footer>—— 阿瑟 · 克拉克（Arthur C. Clarke），《2001 太空漫游》作者</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ MDN Web Docs DOM ｜ Zakas《JavaScript高级程序设计》第4版 ｜ 2026-08-07</p>
</div>

## 为什么从 DOM 开始

前面学完 HTML、CSS 与 JavaScript 语言核心，但三者至今还是「分开的」：HTML 定义结构、CSS 定义外观、JS 在空地上跑。把它们缝在一起的那根线，就是 **DOM（Document Object Model，文档对象模型）**——浏览器把 HTML 解析成一棵**节点树**，然后把这棵树暴露给 JavaScript，于是「改页面」变成了「改树」。

这是前端编程的「跃迁时刻」：之前 JS 只能算数学，现在它能**操作真实页面**——改文字、增删元素、响应点击。Zakas《JavaScript高级程序设计》第4版把 DOM 放在核心参考类型之后、事件之前，正是这个逻辑：**先认识树，再谈对它动手，最后谈它如何与你交互**。本节是第4篇的基石，也是「从静态网页到动态应用」的分水岭。

## 1 文档被解析成树

浏览器收到 HTML 后，解析器把标记流转换成一棵**树**：

- **根**是 `document` 对象——整棵树的入口。
- 每个标签成为一个**元素节点**，标签内的文字成为**文本节点**，属性挂在元素节点上。
- 嵌套结构变成父子关系，先后顺序变成兄弟关系。

`<ul><li>甲</li><li>乙</li></ul>` 就长成：一个 `ul` 节点，挂着两个 `li` 子节点，每个 `li` 各有一个文本子节点「甲」「乙」。

**为什么必须理解「树」**：因为 JS 对页面的所有操作都是**沿树行走**——找父节点 `parentNode`、找子节点 `children`、找兄弟节点 `nextSibling`。树的直觉一旦立住，DOM API 就不再是零散函数，而是一套「在树上走、在树上改」的词汇表。<span class="marginnote">DOM 的树形态不是偶然：W3C 在 1998 年发布 DOM 1 规范时，就把它设计成「独立于平台与语言的接口」——不只是浏览器能用，任何能解析 XML 的宿主都能用同一棵模型。你在这里学的 `parentNode`/`children`，在解析 RSS、SVG 等 XML 类文档时同样成立。</span>

## 2 节点类型：元素、文本与文档

树上的每个节点都有类型，用 `nodeType` 常量区分。最重要的三种：

| nodeType | 含义 | 例子 |
| --- | --- | --- |
| `1` | **元素节点（ELEMENT_NODE）** | `<p>`、`<div>` |
| `3` | **文本节点（TEXT_NODE）** | 「你好」 |
| `9` | **文档节点（DOCUMENT_NODE）** | `document` |

**重点：** 文本也是节点——这是新手最容易忽略的。`<p>你好</p>` 不是「一个节点」，而是「一个元素节点包着一个文本节点」。所以遍历 `childNodes` 时你会看到文本节点（包括空白换行），这也是为什么推荐用 `children`（只收元素）而非 `childNodes`（收所有节点）。

**节点的通用属性**：

- `nodeType`：类型常量。
- `nodeName`：元素节点是标签名（大写 `"P"`），文本节点是 `"#text"`。
- `parentNode` / `childNodes` / `firstChild` / `lastChild` / `nextSibling` / `previousSibling`：树的关系指针。

**元素的独有属性**：`id`、`className`、`classList`（现代的类集合，支持 `add`/`remove`/`toggle`）、`attributes`、`dataset`（读取 `data-*` 属性）。<span class="marginnote">`classList` 是操作类名的现代答案：`el.classList.toggle('active')` 一行切换状态类，而老式写法 `el.className = el.className.replace(...)` 既长又易错。类名操作是「状态 → 样式」的桥，配合第4篇事件篇「切类」模式，是前端状态管理的最朴素形态。</span>

## 3 查询节点：找到你要的那棵树

操作之前先「定位」。现代 DOM 的查询 API 简洁统一：

```js
document.getElementById('app')      // 按 id，返回单个元素（最常用）
document.querySelector('.card')     // CSS 选择器，返回第一个匹配
document.querySelectorAll('.card')  // CSS 选择器，返回 NodeList
document.getElementsByTagName('p')  // 按标签，返回 HTMLCollection
```

**`querySelector` 一族是首选**：它接受完整 CSS 选择器语法——`.card > .title`、`#app input[type="email"]`，与 CSS 知识无缝衔接。`getElementById` 因性能极快仍是「找 id」的默认，但通用查询几乎都走 `querySelector`。

**两个集合类型要分清**：

- **`NodeList`**（`querySelectorAll` 返回）：是**静态快照**——查询那一刻的结果拷贝，之后 DOM 再变它也不变。
- **`HTMLCollection`**（`getElementsBy*` 返回）：是**动态引用**——它始终反映当前 DOM，元素被删除它自动减少。

```js
const cols = document.getElementsByTagName('li'); // 动态
const nodeList = document.querySelectorAll('li'); // 静态
// 之后往文档里加一个 <li>：cols.length 会 +1，nodeList.length 不变
```

**辨析｜易错点：** 遍历 `NodeList` 用 `forEach` 很方便，但遍历 `HTMLCollection` 时 `forEach` **不可用**（它只是类数组）——统一做法是 `Array.from(cols)` 转真数组，或 `for...of` 循环（两者都支持迭代）。分不清集合类型，就会遇到「我删了元素，遍历却还在」或「`forEach` 不存在」的怪错。

## 4 创建、插入与删除节点

定位之后是「改树」。四步操作是全部基本功：

```js
const li = document.createElement('li');   // 1. 创建元素
li.textContent = '新条目';                  // 2. 填内容
parent.appendChild(li);                     // 3. 插入到末尾
// parent.insertBefore(li, refNode);        //    插到 refNode 之前
// parent.removeChild(li);                  // 4. 删除
```

**`createElement`**：创建孤立元素（还没进树）；**`textContent`**：设置/读取纯文本，安全无副作用；**`appendChild`**：挂到某节点末尾；**`insertBefore(new, ref)`**：插到指定节点前（注意没有 `appendAfter`，要靠 `nextSibling` 组合）；**`removeChild`**：删除。

**现代更简洁的接口**：`append`（可同时插多个节点或字符串）、`prepend`、`before`、`after`、`remove`、`replaceWith`——2022 年后在主流浏览器全部可用，写起来远比老 API 顺手：

```js
parent.append('追加文本', li);   // 直接插字符串也能
li.remove();                     // 就地删除，不用先找父节点
```

**辨析｜易错点：** `textContent` 与 `innerHTML` 是「安全」与「便利」的对立：`textContent` 把一切当纯文本（`<b>` 会原样显示），**永远安全**；`innerHTML` 把字符串当 HTML 解析（`<b>` 会加粗），**方便但危险**——如果字符串来自用户输入（用户名、评论、搜索词），`innerHTML` 会执行其中的脚本，正是 XSS 攻击的温床（第5篇《前端安全基础》的主角）。**规则：凡是可能含用户数据的地方，用 `textContent`；`innerHTML` 只用于你完全掌控的、无用户输入的结构**。<span class="marginnote">XSS 的典型引爆点就是 `innerHTML`：用户把 `<img src=x onerror="steal()">` 填进评论区，你的 `innerHTML` 把「图片标签」当成真标签执行了它的 `onerror`。React/Vue 默认转义、提供 `dangerouslySetInnerHTML`/`v-html` 且反复警告——就是这段历史留下的教训。写原生 DOM 时，把这条规矩记牢。</span>

## 5 核心对比表：DOM 操作高频辨析

| 对比 | 选择谁 | 关键差别 |
| --- | --- | --- |
| `getElementById` vs `querySelector` | id 用前者，其余用后者 | 前者只认 id、极快；后者吃全套 CSS 选择器 |
| `NodeList` vs `HTMLCollection` | 一般用 `querySelectorAll` 的 NodeList | 静态快照 vs 动态引用；`forEach` 只有 NodeList 有 |
| `textContent` vs `innerHTML` | 默认 `textContent` | 纯文本（安全）vs 解析 HTML（有 XSS 风险） |
| `childNodes` vs `children` | 遍历元素用 `children` | 前者含文本节点（含空白），后者只收元素 |
| `appendChild` vs `append` | 现代代码用 `append` | 前者只收节点、返回节点；后者收节点+字符串、一次多个 |
| `classList` vs `className` | 现代用 `classList` | 前者可 `add/remove/toggle`；后者整体覆盖字符串 |

**辨析｜易错点：** 一次改多个元素？`querySelectorAll` 的 `NodeList` 支持 `forEach`，配合 `classList` 可以做「给所有 `.card` 加 active」的批量操作；`HTMLCollection` 记得先 `Array.from`。这些「集合与遍历」的差异，是原生 DOM 入门时出错率最高的区域。

## 6 小结

- DOM 是浏览器把 HTML 解析成的**节点树**；`document` 是根，元素、文本都是节点（`nodeType` 1/3/9）。
- 沿树行走：`parentNode` / `children` / `firstChild` / `nextSibling`；遍历元素用 `children` 避开空白文本节点。
- 查询首选 `querySelector`（全套 CSS 选择器）；`getElementById` 留给明确的 id 定位。
- `NodeList` 静态、`HTMLCollection` 动态；`forEach` 只在 `NodeList` 上直接可用。
- 改树四步：`createElement` → `textContent` → `append`/`appendChild` → `remove`；现代接口更简洁。
- **安全铁律**：用户数据只走 `textContent`；`innerHTML` 只用于可信、无用户输入的结构——这是 XSS 的第一道闸。

在下一节，我们让页面「活」起来：**事件模型与事件委托**——树有了、也能改了，接下来学习浏览器如何把用户的每一次点击、按键翻译成可响应的事件。