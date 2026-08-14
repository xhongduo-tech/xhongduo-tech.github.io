---
title: ES Module 模块化
date: 2026-08-07
---

# ES Module 模块化

<div class="epigraph">
<p>模块不是文件，而是边界——它是你告诉代码「这里结束、那里开始」的声明。</p>
<footer>—— 凯尔 · 辛普森（Kyle Simpson）</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ JS高级程序设计 第26章 ｜ 2026-08-07</p>
</div>

## 为什么从模块化开始

代码写到一定规模，单文件必然失控：变量互相覆盖、依赖顺序靠人肉维护、复用只能复制粘贴。**模块化**是软件工程的必然答案——把代码按职责拆成独立单元，明确「谁能看到谁」。JS 的模块化历程尤其曲折：从全局变量、IIFE、CommonJS（Node）、AMD，最终收敛到语言标准 **ES Module（ESM）**。

这一节是 JS 语言部分（第3篇）的收官，也是通向工程化（第5篇）的入口。**静态分析**——ESM 的 import/export 在代码运行前就可被解析——直接支撑了第26篇《构建工具与模块打包》里的 tree-shaking 与代码分割。不理解 ESM，就不理解打包器在干什么。<span class="marginnote">Zakas 第26章「模块」。ESM 是 ES2015 引入的语言级模块系统，如今浏览器与 Node.js 原生支持；但它真正横扫工程界，靠的是配合打包器（webpack、Vite）在浏览器里落地。</span>

## 1 模块的痛点：为什么历史这么曲折

在没有模块的远古时代，JS 靠**全局变量**共享代码：

```html
<script src="a.js"></script>
<script src="b.js"></script>
```

问题立刻浮现：

1. **全局污染**：`a.js` 声明的 `window.count` 可能被 `b.js` 同名覆盖——谁后加载谁赢，无声无息。
2. **依赖顺序靠人**：`b.js` 用 `a.js` 的东西，必须保证 `<script>` 顺序——错了就 `ReferenceError`，且错误在运行时才暴露。
3. **复用是复制**：没有「引入」，只能把代码抄进新文件。

于是出现一系列方案，一路演进到标准：

| 方案 | 机制 | 状态 |
| --- | --- | --- |
| IIFE | 立即执行函数制造私有作用域 | 历史方案 |
| AMD / RequireJS | 浏览器端异步加载 | 历史方案 |
| CommonJS | `require`/`module.exports`，同步 | Node 默认，浏览器需打包 |
| **ES Module** | `import`/`export`，静态、异步 | **语言标准** |

**CommonJS 与 ESM 的本质差异**：CommonJS 的 `require` 是**运行时**的——返回值是一个对象，可以出现在任何表达式里；ESM 的 `import` 是**编译期**的——必须写在顶层，模块结构静态可解析。这条差异带来连锁反应：静态可解析 → 工具能分析依赖 → 能裁剪无用代码（tree-shaking）→ 能并行加载。<span class="marginnote">为什么 CommonJS 不能 tree-shaking？因为 `require(x)` 里 x 可以是变量、可以是条件分支，工具在编译期无法确定「到底依赖谁」。ESM 的 import 路径是字面量字符串、位置固定，依赖图完全确定——这是打包优化的前提。</span>

## 2 导出：named 与 default

模块的「出口」用 `export` 声明。两种形式：

**命名导出（named export）**——模块可以有多个：

```js
// utils.js
export const VERSION = "1.0";
export function format(n) { return n.toFixed(2); }
export class Store {}
```

**默认导出（default export）**——每个模块至多一个，适合「本模块的主角」：

```js
// store.js
export default class Store {
  constructor() { this.data = []; }
}
```

**辨析｜易错点：** 命名导出与默认导出是**两套不同机制**，可同时存在：

```js
export const VERSION = "1.0";
export default class Store { … }
```

导入时对应两种写法——默认导入不加大括号，命名导入要加大括号且**名字必须与导出名一致**：

```js
import Store from "./store.js";            // 默认导入：名可随便起
import { VERSION } from "./utils.js";      // 命名导入：名必须对上
import Store, { VERSION } from "./mod.js"; // 混合
```

**`export default` 的来历**：它提供了「每个模块一个主角」的惯例，让默认导入的语法最简——但在「同时导多个」的场景里，命名导出更清晰、更利于静态分析和重命名（alias `import { f as g }`）。

## 3 导入：import 的三种形态

```js
import { a, b } from "./x.js";        // 1. 按名导入
import { a as alias } from "./x.js";  // 2. 改名导入
import * as utils from "./x.js";      // 3. 命名空间导入：整体当对象
import "./polyfill.js";               // 4. 副作用导入：只执行，不取值
```

**命名空间导入** `* as utils` 把模块所有导出打包成一个对象——注意它只在 `import` 语句里合法，且不要过度使用（会丢失静态分析的部分收益）。
**副作用导入**用于「执行模块代码但不需要导出」——如引入一个全局补丁、注册一个自定义元素。

**导入的规则**：

- 必须在**模块顶层**（不能写在 `if` 或函数里）——因为 import 是静态的。
- 路径是字符串字面量，浏览器里**必须带扩展名** `./x.js`（Node/打包器里可省略）。
- 模块是**单例**：同一路径多次 import，模块只执行一次，导出对象共享——「导入的是同一个引用」。

**模块的作用域是独立的**：每个模块的顶层 `const`/`let`/`function` 都只在模块内可见，**不进全局**——这直接解决了「全局污染」的百年痛点。模块默认**严格模式**（`"use strict"` 自动开启），`this` 是 `undefined`。<span class="marginnote">「模块是单例且共享引用」意味着：导出对象被多处导入时，大家看到的是<strong>同一个对象</strong>——一处修改，处处可见。这在「配置对象」「全局状态」场景是特性，但在「不可变数据」场景要小心。</span>

## 4 实时绑定：import 是引用不是拷贝

ESM 最反直觉也最优雅的特性：**import 进来的不是值的拷贝，而是实时绑定（live binding）**。

```js
// counter.js
export let count = 0;
export function inc() { count++; }

// main.js
import { count, inc } from "./counter.js";
console.log(count);   // 0
inc();
console.log(count);   // 1 —— 读取的是最新值！
```

导出模块改了 `count`，导入方读到的**永远是最新值**——因为导入方持有的是对 `count` 的引用。这对比 CommonJS 的 `require`（拷贝当时的导出对象快照），是完全不同的模型。

**辨析｜易错点：** 实时绑定**只读不写**——导入方不能给 `count` 赋值（`count = 5` 抛 `TypeError`），只能通过导出模块自己的函数修改。所以模块的「状态」天然被封装在模块内部，对外只暴露「读」与「操作入口」——这恰好就是模块作为「边界」的意义：**数据在边界内，操作在边界内，外界只能按你给的通道用**。<span class="marginnote">实时绑定是 ESM「静态分析」的另一面：编译器知道哪个绑定对应哪个导出，于是能生成「同步更新」的引用。这对 tree-shaking 也友好——没被 import 的导出，可以确定地删掉。</span>

## 5 动态 import()：按需加载

静态 `import` 必须在顶层，但有些模块**运行时才知道要不要加载**（用户点开某个弹窗才需要）。`import()` **函数形式**（动态导入）填补了这个缺口——它返回 Promise：

```js
// 按需加载，点击才下载
button.addEventListener("click", async () => {
  const { openChart } = await import("./chart.js");
  openChart(data);
});
```

动态导入的三个价值：

1. **代码分割（code splitting）**：打包器会把动态导入的模块拆成独立文件，用到才下载——首屏体积变小（呼应第28篇性能优化）。
2. **条件加载**：按环境/权限决定加载哪套实现。
3. **模块路径可动态**：`import(\`./locale/${lang}.js\`)` 运行时拼路径。

动态导入返回 Promise，所以天然配合 `async/await`——第19篇的异步武器在此汇合。<span class="marginnote">代码分割是构建工具的核心能力：`import()` 变成打包器划分 chunk 的标记，配合 `preload`/懒加载，是「首屏快」的工程基石。第26篇《构建工具与模块打包》会深入。</span>

## 6 公式解析：依赖图与加载时序

ESM 的加载机制可以用一张「**依赖图**」来理解——这也是构建工具的思维模型：

$$
\text{load}(M) = \text{parse}(M) \Rightarrow \text{resolve}(\text{imports}(M)) \Rightarrow \text{instantiate} \Rightarrow \text{evaluate}
$$

**逐步拆解：**

- **第一步，解析（parse）**：读源码，建立模块的导出表与导入表——这是**静态**阶段，只读语法结构，不执行。
- **第二步，解析依赖（resolve）**：对每个 import 路径，找到目标文件——这一步**递归**，最终得到一张完整依赖图（谁依赖谁）。
- **第三步，实例化（instantiate）**：为每个模块创建「绑定」容器，连接 import 与 export——此刻实时绑定建立起来。
- **第四步，求值（evaluate）**：执行模块体代码，按依赖顺序**先执行依赖、再执行自身**。

**代入一个实例：** 页面加载 `main.js`，它 import `a.js`，`a.js` 又 import `b.js`——加载顺序是 `b → a → main`（依赖先跑）。这个「**深度优先、依赖优先**」的顺序保证：`main.js` 执行时，`a.js` 和 `b.js` 都已求值完毕。

**直觉是什么？** 静态的「解析 + 解析依赖」先行，动态的「求值」在后——**先画出依赖图，再执行**。依赖图就是模块世界的「施工蓝图」：构建工具拿着它做 tree-shaking、拆 chunk、做缓存。<span class="marginnote">循环依赖（a import b、b import a）在 ESM 里是合法的——因为「实例化」阶段就建立了绑定，求值阶段即使 a 未跑完、b 已经能通过绑定读 a 的导出（可能是 undefined）。CommonJS 遇循环依赖则更容易踩「读到半成品」的坑。理解「先连后跑」就理解了为什么。</span>

## 7 小结

- 模块化解决**全局污染、依赖顺序、复用**三大痛点；演进路线 IIFE → CommonJS → ESM 语言标准。
- **命名导出**（多个、按名导入）与**默认导出**（至多一个、自由命名）；两者可共存。
- import 四种形态：按名、改名、命名空间、副作用；import 必须在顶层、路径是静态字面量。
- **实时绑定**：导入的是引用不是拷贝，读到的永远是最新值，但只读不写。
- `import()` 动态导入返回 Promise，实现**代码分割与按需加载**。
- ESM 加载是「**解析 → 解析依赖 → 实例化 → 求值**」，依赖优先执行；依赖图是打包器的工作蓝图。

在下一节，我们跨入浏览器 API 世界——**DOM 树结构与节点操作**。JS 语言部分告一段落，从此刻起，我们用 JS 与页面本身对话。
