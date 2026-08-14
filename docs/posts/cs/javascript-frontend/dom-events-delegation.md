---
title: 事件模型与事件委托
date: 2026-08-07
---

# 事件模型与事件委托

<div class="epigraph">
<p>用户的行为像投入湖中的石子，事件就是那圈涟漪——从中心荡开，路过每一个岸边。</p>
<footer>—— 尼古拉斯 · 扎卡斯（Nicholas C. Zakas）</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ MDN Web Docs 事件 ｜ 2026-08-07</p>
</div>

## 为什么从事件开始

页面不是「运行一遍就完」的程序——它要**响应**用户的每一次点击、按键、滚动、输入。这套「用户动作 → 浏览器通知 → 代码响应」的机制，就是**事件（event）**。上一节学会了操作 DOM，这一节让 DOM「听见」用户的动作。

事件模型是浏览器最核心的交互机制，值得精确理解两件事：**事件是怎么传播的**（捕获/冒泡三阶段），以及**事件委托**——用「一个监听器管一片元素」的优雅模式。事件委托不只是性能优化，更是「动态内容也能响应」的解决方案：新插入的元素不需要重新绑监听，因为它们的事件一样会冒泡到容器。<span class="marginnote">DOM 事件规范（DOM Events）经历了从「老式 inline onclick」到「标准 addEventListener」的演进。理解传播三阶段与 `currentTarget`/`target` 的差别，就掌握了事件调试的核心。</span>

## 1 监听事件：addEventListener

给元素绑定事件处理器的标准方式：

```js
const btn = document.querySelector("#save");
btn.addEventListener("click", (e) => {
  console.log("点击了", e);
});
```

**`addEventListener(type, listener, options)` 三参**：

`type`：事件类型字符串，如 `"click"`、`"keydown"`、`"submit"`。
`listener`：回调函数，收到事件对象 `e`。
- `options`：配置对象或布尔值。

常用选项：

```js
btn.addEventListener("click", handler, { once: true });   // 只触发一次
btn.addEventListener("click", handler, { passive: true }); // 声明不调 preventDefault，利于滚动性能
btn.removeEventListener("click", handler);                // 移除监听
```

- **`once: true`**：触发后自动移除——适合「首次完成」类交互。
- **`passive: true`**：告诉浏览器「我这个监听器不会阻止默认行为」——滚轮/触摸滚动监听加上它可让滚动不卡顿（性能优化，呼应第28篇）。
- **`removeEventListener` 必须传同一个函数引用**：匿名函数无法移除——想「先绑后解」就得用具名函数。

**事件类型家族**：`click`/`dblclick`（鼠标）、`keydown`/`keyup`（键盘）、`focus`/`blur`（焦点）、`submit`/`input`/`change`（表单）、`scroll`/`resize`（窗口）、`touchstart`/`touchmove`（触摸）。<span class="marginnote">区分三个「长得像」的表单事件：`input` 每次输入都触发（实时）；`change` 失焦或回车才触发（最终值）；`submit` 是表单提交。用 `input` 做实时预览，用 `change` 做「改完再处理」，用 `submit` 做提交拦截。</span>

## 2 事件对象：target 与 currentTarget

监听器收到的 `e`（Event 对象）携带大量信息，最常用的是四个：

**`e.target`**：事件**实际发生**的元素（用户点的那个）。
**`e.currentTarget`**：**当前正在执行监听器**的元素（绑定的那个）。
- **`e.preventDefault()`**：阻止默认行为（如链接跳转、表单提交）。
- **`e.stopPropagation()`**：阻止事件继续传播。

```js
list.addEventListener("click", (e) => {
  console.log(e.target);         // 用户实际点的元素（可能是个 <li> 里的 <span>）
  console.log(e.currentTarget);  // list 本身（监听器绑在这里）
});
```

**`target` 与 `currentTarget` 的差别是事件调试的第一课**——点击 `<li>` 里的 `<span>` 时，`target` 是 `<span>`，而监听器所在的 `currentTarget` 是 `<ul>`。两者只在「事件恰好发生在绑定元素本身」时相等。

**`preventDefault` 是「取消」，`stopPropagation` 是「截流」**——前者阻止浏览器的默认行为（链接跳转、表单提交、右键菜单），后者阻止事件传播到其他节点。两者常一起用但作用维度不同：一个管「浏览器做什么」，一个管「事件走到哪」。<span class="marginnote">经典用例：表单校验失败时 `e.preventDefault()` 阻止提交；`keydown` 里对回车键 `preventDefault()` 阻止表单被回车触发。注意 `passive: true` 的监听器里调 `preventDefault` 会被忽略——浏览器早已认为你不会阻止。</span>

## 3 传播三阶段：捕获、目标、冒泡

事件在 DOM 树中的传播分成**三个阶段**：

```
捕获阶段：window → document → html → body → … → target
目标阶段：事件到达 target
冒泡阶段：target → … → body → html → document → window
```

```html
<div id="outer"><div id="inner"><button id="btn">点我</button></div></div>
```

点击 `btn` 时，事件先从 `window` **向下捕获**到 `btn`（捕获阶段），在 `btn` 触发目标阶段后，再**向上冒泡**回 `window`（冒泡阶段）。

```js
outer.addEventListener("click", () => console.log("冒泡：outer"));
outer.addEventListener("click", () => console.log("捕获：outer"), true);  // true 开启捕获
btn.addEventListener("click", () => console.log("目标：btn"));
// 点击 btn 输出：捕获：outer → 目标：btn → 冒泡：outer
```

**第三参 `true` 表示在捕获阶段监听**（默认 false 即冒泡阶段）。事件**总是先捕获、后冒泡**，即使你没有捕获阶段的监听器，事件也照样走完全程——只是「没人听见」而已。<span class="marginnote">为什么要有捕获？历史上 Netscape 用捕获、IE 用冒泡，标准两全其美都保留。冒泡是默认、也最常用（事件委托依赖它）；捕获留给「要在目标之前截胡」的场景，如全局错误捕获 `window.addEventListener("error", …)`。</span>

## 4 事件委托：一个监听器管一片

**事件委托（event delegation）** 利用冒泡：与其给每个子元素绑监听器，不如在**共同的祖先**绑一个，用 `target` 判断是谁触发的：

```js
// 不用委托：1000 个 li 绑 1000 个监听器（且新增 li 还得重新绑）
document.querySelectorAll("li").forEach(li => li.addEventListener("click", handler));

// 委托：1 个监听器管所有 li（含未来新增的）
list.addEventListener("click", (e) => {
  const li = e.target.closest("li");
  if (!li) return;                 // 点的不在 li 里，忽略
  console.log(li.textContent);
});
```

**委托的三大优势**：

1. **省内存**：一个监听器 vs N 个——页面元素越多差距越大。
2. **动态内容免维护**：`innerHTML` 或 JS 新增的 `li` 无需重新绑定——事件冒泡到 `list` 自然被捕获。
3. **逻辑集中**：所有同类交互的处理集中在一处，可读性高。

**`closest("li")` 是关键技巧**：`e.target` 可能是 `<li>` 里的 `<span>`、`<strong>`，`closest` 沿父链找到所属的 `li`；找不到（点在列表外的区域）就 `return`——「不是我的事就放行」。

**辨析｜易错点：** 委托不是万能的——

**`mouseenter`/`mouseleave` 不冒泡**，无法委托，得用冒泡版的 `mouseover`/`mouseout` 自行判断。
`focus`/`blur` 不冒泡，但 `focusin`/`focusout` 冒泡——委托焦点事件要用后者。
- `scroll` 事件不冒泡（但在 window 上可统一监听）。

选对「可冒泡的等价事件」，委托才成立。<span class="marginnote">事件委托的边界：它适合「许多同类子元素」的场景（列表、表格行、菜单项）。如果子元素形态各异、交互各自独立，直接绑定可能更清晰。委托是工具不是教条。</span>

## 5 自定义事件：dispatchEvent 与 CustomEvent

除了浏览器触发的事件，你还能**自己发明事件**，让组件之间解耦通信：

```js
// 派发自定义事件
const el = document.getElementById("box");
el.dispatchEvent(new CustomEvent("app:loaded", {
  detail: { count: 42 },      // detail 携带自定义数据
}));

// 监听自定义事件
el.addEventListener("app:loaded", (e) => {
  console.log(e.detail.count);   // 42
});
```

**`CustomEvent(type, { detail })`**：`detail` 是携带任意数据的字段，是自定义事件的「载荷」。
**`dispatchEvent`**：在指定元素上「模拟触发」一个事件——浏览器事件也能用（如程序化触发表单校验）。
- **命名惯例**：事件名用 `namespace:action` 风格（如 `app:loaded`、`cart:update`），避免与原生事件冲突。

自定义事件是「观察者模式」的 DOM 实现：组件 A 派发「数据就绪」，组件 B 监听并响应——两者互不引用，实现解耦。<span class="marginnote">自定义事件与第19篇的事件循环联动：`dispatchEvent` 派发的事件会<strong>同步</strong>执行监听器（在当前调用栈内）。若想「下一轮再响应」，配合 `setTimeout(0)` 或 `queueMicrotask` 延迟派发。</span>

## 6 公式解析：事件传播路径

事件传播的完整行为可以写成一条「**路径公式**」，它解释了 `target`、`currentTarget` 与 stopPropagation 的一切：

$$
\text{path} = [w, d, h, b, \ldots, \text{target}, \ldots, b, h, d, w]
$$