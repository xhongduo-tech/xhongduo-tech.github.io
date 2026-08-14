---
title: 构建工具与模块打包
date: 2026-08-07
---

# 构建工具与模块打包

<div class="epigraph">
<p>构建工具解决的不是「能不能跑」，而是「在一个真实的世界里能不能快、稳、省地跑」。</p>
<footer>—— 埃文 · 尤（Evan You），Vue/Vite 作者</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ MDN Web Docs 工具 ｜ 2026-08-07</p>
</div>

## 为什么从构建工具开始

第20篇学了 ES Module——但把源代码直接丢给浏览器，现实会打脸：几百个 `.js` 文件意味着几百次 HTTP 请求；`.jsx`、`.ts`、`.scss` 浏览器根本看不懂；开发时改一行要等全站刷新……**构建工具（build tool）**把这些「现实问题」集中解决：它把源码**编译、打包、压缩**成浏览器能高效加载的产物。

这是从「会写前端」到「能做工程」的分水岭。现代前端几乎不可能绕过构建：Vite、webpack、Rollup 是当前三巨头，它们解决同一类问题但各有侧重。理解「构建在做什么」，比记住某个工具的配置更重要——工具会换代，**依赖图、tree-shaking、代码分割、source map** 这些概念是长青的。<span class="marginnote">为什么不能直接让浏览器加载源码？两个硬伤：一是文件数——上百个模块=上百个请求，首屏慢到爆炸；二是语法——JSX、TypeScript、Sass、下一代语法浏览器要么不支持要么性能差。构建工具把「开发便利」与「生产高效」用一道转换隔开。</span>

## 1 构建工具解决的四件事

一次构建（build），本质上做四件事：

**1. 转译（transpile）**：把「写的时候方便」的语法转成「浏览器懂」的语法。

JSX → 普通 JS（`React.createElement` 调用）。
TypeScript → JS（抹掉类型注解）。
- 下一代 JS 语法 → 当前浏览器支持的版本（Babel 做）。
- Sass/SCSS → 纯 CSS。

**2. 打包（bundle）**：把散落的模块**合并**成少量文件。从**入口（entry）**出发，沿 `import` 关系递归收集全部依赖，得到**依赖图（dependency graph）**，再合并输出成 `bundle.js`——几百个模块变几个文件，请求数骤降。

**3. 优化（optimize）**：压缩（minify）——去空白、缩短变量名，体积减半；tree-shaking——删掉「导出了但没被用到」的代码（依赖第20篇 ESM 的静态分析）；代码分割——按需拆成 chunk。

**4. 开发体验（DX）**：开发服务器（dev server）——改代码即时热更新（HMR），不用手动刷新；source map——把压缩后的代码映射回源码，断点调试不受影响。<span class="marginnote"><strong>热更新（HMR, Hot Module Replacement）</strong> 是开发体验的核心：保存文件，浏览器<strong>只更新改动的那块</strong>，不刷新整页、不丢状态（表单填到一半不重置）。Vite 用原生 ESM 按需编译实现「毫秒级启动」，这是它取代 webpack 成为新默认的重要原因。</span>

## 2 核心概念：入口、依赖图与输出

所有打包器共享同一个心智模型，理解它就理解了构建工具的共同内核：

**入口（entry）**：打包从哪个文件开始，通常是 `src/main.js`。它是依赖图的根。

**依赖图（dependency graph）**：从入口递归扫描所有 `import`/`require`，得到的「谁依赖谁」的网络。每个模块是图里的节点，依赖关系是边。

```
main.js
 ├─ components/Header.js
 │   └─ icons.js
 ├─ utils/api.js
 └─ styles/main.css
```

**输出（output）**：把依赖图合并成可部署的文件，通常 `dist/` 目录：

```
dist/
 ├─ index.html
 ├─ assets/index-abc123.js    # 带内容哈希，用于缓存
 └─ assets/index-abc123.css
```

**内容哈希（content hash）** 是输出的关键设计：文件名里嵌入内容的哈希（`index-abc123.js`）——内容变了哈希才变。浏览器缓存旧文件名，哈希变了就重新下载——「文件名即版本号」的缓存策略（呼应第28篇性能优化）。

**辨析｜易错点：** 依赖图是**静态分析**的产物——打包器解析 `import` 语句，但**不执行代码**。所以「动态拼路径的 require」无法被准确分析，`import()`（第20篇）是唯一被支持的「动态」形态——它成为代码分割的标记点。<span class="marginnote">依赖图的思维也解释了「为什么循环依赖能工作」：图是「结构」而非「执行顺序」，打包器先建立全部节点与边，再按依赖拓扑序求值——所以第20篇说「先连后跑」，在打包层面同样成立。</span>

## 3 tree-shaking：摇掉没用的代码

**Tree-shaking（摇树优化）** 是打包器最著名的优化：把「导出了但没人 import」的代码删掉。

```js
// utils.js
export function used() { return 1; }
export function unused() { return 2; }   // 没人 import → 会被摇掉

// main.js
import { used } from "./utils.js";
```

**为什么只有 ESM 能摇树？** 第20篇的核心伏笔在此兑现——ESM 的 import/export 是**静态**的：路径是字面量、位置在顶层、结构编译期可确定。打包器能精确知道「`unused` 从没被 import」，于是安全删除。而 CommonJS 的 `require` 可以是变量、条件分支——**运行时才确定依赖，无法静态摇树**。

摇树的三个前提：

1. **副作用标记**：模块顶层若有副作用（`console.log`、修改全局），删除会改变行为——所以 `package.json` 的 `"sideEffects": false` 告诉打包器「本包无副作用，可放心摇」。
2. **纯 ESM 依赖**：依赖库必须是 ESM（提供 `"module"` 入口）才能被摇。
3. **生产构建**：tree-shaking 只在生产模式生效——开发模式保留完整代码便于调试。

**辨析｜易错点：** 摇树对「对象属性」不生效——`import { format } from "utils"` 且 utils 是个大对象 `export default { format, parse, … }`，**默认导出整个对象无法摇掉未用属性**（对象是运行时结构）。所以「按需引入」的库要提供**具名导出**而非把一切塞进 default 对象——这是库作者的设计责任。<span class="marginnote">这就是为什么现代库（如 lodash-es、Ant Design）要额外提供 ESM 版本：让使用者能摇树，否则一个 `import { get } from "lodash"` 会带进整个库。选依赖时「有没有 ESM 入口」直接影响产物体积。</span>

## 4 代码分割与懒加载

**代码分割（code splitting）** 把「一个巨大 bundle」拆成「按需加载的多个 chunk」——与第20篇的 `import()` 是一对：

```js
// 动态导入 → 打包器自动拆成独立 chunk
button.addEventListener("click", async () => {
  const { openChart } = await import("./chart.js");
  openChart(data);
});
```

**打包器看到 `import()` 就把它标记为分割点**：`chart.js` 被拆进单独文件，**只有点击时才下载**。收益：

首屏只加载必需的代码——体积变小，加载变快（第28篇核心指标之一）。
不常用功能（图表、编辑器、设置页）延迟加载。

**分割的粒度策略**：

- **按路由**：单页应用每页一个 chunk，切页才加载。
- **按功能**：重组件（编辑器、图表库）独立 chunk。
- **按供应商**：第三方库（React、Vue）单独 chunk，长期不变、缓存友好。

**辨析｜易错点：** 分割不是越细越好——每个 chunk 都有网络往返开销。太碎的 chunk 会让「加载大功能」变成「十几个小请求」，反而更慢。实践是「**按路由/按重功能**分块，别把每个小组件都拆出去」。<span class="marginnote">分割的本质是「首屏 vs 全量」的时间权衡：把「现在不需要的」推迟到「需要时」。现代框架的路由懒加载（React.lazy、Vue 的动态组件）底层都是 `import()` + 代码分割——你写的懒加载代码，打包器负责拆，浏览器负责按需取。</span>

## 5 主流工具对比：webpack、Rollup、esbuild、Vite

当前构建工具谱系，各有定位：

| 工具 | 定位 | 特点 | 适用 |
| --- | --- | --- | --- |
| webpack | 全能打包器 | 生态最大、配置最繁 | 大型工程、需要深度定制 |
| Rollup | ESM 打包器 | tree-shaking 强、产物干净 | 库的开发发布 |
| esbuild | 极速打包器 | Go 写的、快 10–100 倍 | 作为底层引擎 |
| Vite | 开发服务器 + 打包 | 原生 ESM 按需编译、秒启动 | 现代新项目默认 |

**webpack**：老牌王者，插件系统庞大，但配置重、构建慢。
**Rollup**：以「对 ESM 的极致优化」著称，**写库**用它（产物精简、可摇树）。
- **esbuild**：用 Go 重写的打包器，速度碾压 JS 系——被 Vite 用作生产构建引擎。
- **Vite**：开发时**不打包**——直接用浏览器原生 ESM 按需加载（毫秒启动），生产时用 Rollup/esbuild 打包。**当前新项目的默认选择**。

**辨析｜易错点：** 「开发不打包」是 Vite 的核心创新——开发服务器把源码按浏览器原生 ESM 直接提供给浏览器，只转译不合并，所以启动与热更新都快到感知不到。但它依赖浏览器原生 ESM 支持（现代浏览器都支持），且生产仍是「打包后分发」——**「开发快」与「生产快」是两个独立目标，Vite 分别用不同引擎达成**。<span class="marginnote">工具选择的现实指南：2020 年后新项目默认 Vite；维护老 webpack 项目要能读它的配置；写 npm 库用 Rollup。esbuild 本身很少直接用作构建器，更多是「被别的工具当引擎」。理解「工具是手段、构建概念是目的」，换工具就不慌。</span>

## 6 公式解析：构建管线

一次构建可以浓缩成一条**管线公式**——它把「源码」加工成「产物」的每一步都标出来：

$$
\text{src} \xrightarrow{\text{resolve}} \text{deps} \xrightarrow{\text{transpile}} \text{es5/js} \xrightarrow{\text{bundle}} \text{chunks} \xrightarrow{\text{optimize}} \text{assets}
$$