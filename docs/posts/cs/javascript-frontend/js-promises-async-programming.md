---
title: Promise 与异步编程
date: 2026-08-07
---

# Promise 与异步编程

<div class="epigraph">
<p>异步不是并发，而是「不等结果就继续做别的事」；Promise 让这种「不等」变得有序而可预测。</p>
<footer>—— 尼古拉斯 · 扎卡斯（Nicholas C. Zakas）</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ JS高级程序设计 第11章 ｜ 2026-08-07</p>
</div>

## 为什么从 Promise 与异步开始

浏览器里最耗时的操作——**网络请求、定时器、用户事件、文件读写**——全都是**异步**的：发起时不阻塞界面，结果在「不知道什么时候」回来。JS 是单线程语言，处理异步靠的是事件循环（event loop）与回调。而 **Promise** 是把「异步结果」封装成对象的现代方案，**async/await** 则让它长得像同步代码。

这一节是全专题最重要的章节之一：它是 JS 语言部分（第3篇）与浏览器 API 部分（第4篇）的**桥梁**。第24篇《Fetch 与网络请求》就是 Promise 的最大实战——`fetch()` 返回 Promise。不理解 Promise，就谈不上做任何真实的前端应用。<span class="marginnote">Zakas 第11章「Promise 与异步编程」：从事件循环讲起，到 Promise 基础、链式、错误处理、async/await。注意：Promise 是语言规范，不依赖浏览器——Node.js 也完全一样。</span>

## 1 事件循环：单线程如何不卡死

JS 在**单线程**上运行——一次只执行一段代码。但浏览器还需要响应点击、滚动、网络……答案就是**事件循环（event loop）**，它把代码分成「现在的」与「稍后的」：

```
执行栈（call stack） → 同步代码跑完 → 事件循环轮询队列 → 取下一个任务执行
```

模型里有三个关键队列：

**调用栈（call stack）**：当前正在执行的同步代码。
**任务队列（task queue / macrotask）**：`setTimeout`、`setInterval`、I/O、UI 事件——**按序执行**。
- **微任务队列（microtask queue）**：`Promise.then`、`queueMicrotask`——**在每轮宏任务结束后、渲染前清空**。

```js
console.log("1");                       // 同步，先跑
setTimeout(() => console.log("2"), 0);  // 宏任务，排队
Promise.resolve().then(() => console.log("3"));  // 微任务，插队
console.log("4");                       // 同步

// 输出顺序：1, 4, 3, 2
```

**为什么微任务先于宏任务？** 每个宏任务结束后，事件循环会**清空整个微任务队列**再取下一个宏任务。`Promise.then` 是微任务——它总比下一个 `setTimeout` 先跑，即使后者延时为 0。<span class="marginnote">记忆：<strong>同步 → 微任务 → 宏任务</strong>。微任务里再排微任务，会在一轮内全部清空（直到队列空）；宏任务则要等下一轮。这个顺序是无数「为什么输出是这样」面试题的钥匙。</span>

## 2 从回调地狱到 Promise

**回调（callback）** 是最原始的异步处理：把「结果回来之后做什么」传进函数。但嵌套多了就变成「回调地狱」：

```js
getUser(id, (user) => {
  getPosts(user.id, (posts) => {
    getComments(posts[0].id, (comments) => {
      render(comments);   // 一层套一层，越缩越深
    });
  });
});
```

**Promise** 解决这个问题的思路：异步操作返回一个**状态对象**，你在这个对象上**声明**「成功时做什么、失败时做什么」——而不是把回调传进去。嵌套的「金字塔」变成了平的「链条」。

```js
getUser(id)
  .then(user => getPosts(user.id))
  .then(posts => getComments(posts[0].id))
  .then(render)
  .catch(err => handleError(err));
```

**Promise 的三种状态**：`pending`（进行中）→ `fulfilled`（成功）或 `rejected`（失败）。**状态一旦确定就不可逆转**——成功不能变失败，失败不能变成功。这是 Promise 可靠性的根基。<span class="marginnote">「状态一经落定不可变」意味着：同一结果被多个 `.then` 监听，它们都会拿到同一个值——Promise 可安全地到处传递，不用担心竞态。这是它比「每次都要重新发起回调」的裸回调强得多的地方。</span>

## 3 Promise 的创建与消费

**创建**：`new Promise((resolve, reject) => …)`，执行器立即同步运行，`resolve` 置为成功、`reject` 置为失败：

```js
const p = new Promise((resolve, reject) => {
  const ok = doWork();
  if (ok) resolve("结果数据");
  else    reject(new Error("失败原因"));
});
```

**消费**：`then(onFulfilled, onRejected)`、`catch(onRejected)`、`finally(onFinally)`：

```js
p
  .then(data => console.log(data))      // 成功分支
  .catch(err => console.error(err))     // 失败分支（等价 then 的第二参）
  .finally(() => cleanup());            // 无论成败都会执行（清理）
```

**`.then` 返回新 Promise**——这使链式调用成为可能，且链上任意一处 `throw` 都会被后面的 `.catch` 捕获：

```js
fetchUser()
  .then(user => {
    if (!user) throw new Error("用户不存在");  // 抛错 → 跳过后续 then → 进 catch
    return fetchPosts(user.id);
  })
  .then(posts => render(posts))
  .catch(err => showError(err.message));
```

**错误传播**是 Promise 链最优雅的设计：**链中任何一个环节失败，都会沿链滑到最近的 `catch`**——不需要每层都手动判断。这解决了回调地狱里「每个回调都要处理错误」的噩梦。<span class="marginnote">`.catch` 只抓「它之前的链段」的错误。想在链中某段后「局部兜底再继续」，就在那个位置放 `.catch`，返回默认值后链继续走——「捕获 → 修复 → 继续」模式。</span>

## 4 Promise 组合：all、allSettled、race 与 any

真实场景常要**同时发起多个异步**，四个静态方法对应四种「多合一」策略：

```js
// 全部成功才继续，任一个失败就整体失败
const [users, posts] = await Promise.all([fetchUsers(), fetchPosts()]);

// 全部结算后返回结果数组（含成功与失败），永不整体拒绝
const results = await Promise.allSettled([a(), b()]);

// 谁先落定就返回谁（用于超时竞赛）
const winner = await Promise.race([request(), timeout(3000)]);

// 第一个成功即返回；全失败才拒绝（用于多路备选）
const first = await Promise.any([cdnA(), cdnB()]);
```

| 方法 | 成功条件 | 失败条件 | 典型场景 |
| --- | --- | --- | --- |
| `all` | 全部成功 | 任一失败即拒绝 | 并行依赖多个数据 |
| `allSettled` | 永不整体拒绝 | — | 并行任务都要结果，个别失败可容忍 |
| `race` | 第一个落定 | 第一个落定失败也拒绝 | 超时控制、竞速 |
| `any` | 第一个成功 | 全部失败才拒绝 | 多 CDN 容灾 |

**辨析｜易错点：** `all` 与 `allSettled` 的取舍——「必须全部成功才能继续」用 `all`（如并行加载几个必填数据）；「失败个别没关系，都要拿到结果」用 `allSettled`（如批量上报，个别失败不阻塞）。`race` 配 `Promise.reject` 可以实现**超时**——请求与「3 秒后拒绝」赛跑，谁先谁赢。<span class="marginnote">`Promise.race` 做超时：`race([fetch(url), new Promise((_, rej) => setTimeout(() => rej(new Error("超时")), 3000))])`——请求没在 3 秒内完成就拒绝。注意 race 不会取消已发出的请求，只是「先到先得」。</span>

## 5 async/await：让异步像同步

**`async`/`await`** 是 Promise 的语法糖——生成器思想的正式产品。`async` 函数**总是返回 Promise**，`await` 在 Promise 上暂停直到落定：

```js
async function loadPage() {
  try {
    const user = await fetchUser();        // 暂停，等 Promise 落定
    const posts = await fetchPosts(user.id);
    render(posts);
  } catch (err) {
    showError(err);                        // await 的拒绝在 try/catch 里接住
  }
}
```

**await 的规则**：`await` 只能在 `async` 函数内使用；`await` 一个 Promise 会「解开」它——拿到 fulfilled 值，rejected 则抛出异常（被 `try/catch` 捕获）；`await` 非 Promise 值则原样返回（可放心 `await 普通值`）。

**async/await 与 .then 链的对比**：

| 维度 | `.then` 链 | `async/await` |
| --- | --- | --- |
| 代码形态 | 回调式，平铺 | 同步式，直读 |
| 错误处理 | `.catch` 集中 | `try/catch` 就近 |
| 调试 | 需在链里打断点 | 像同步代码，断点直观 |
| 并行 | `Promise.all` | `Promise.all` 同样配合 |

**辨析｜易错点：** `await` 是「逐个等待」，不是「并行」——`await a(); await b()` 串行。想并行，先发起再 await：

```js
// 串行：b 要等 a 完成
const ra = await a(); const rb = await b();
// 并行：两个同时跑
const [ra, rb] = await Promise.all([a(), b()]);
```

并行是前端性能的常见优化点——两个无关请求不该排队等待，应同时发出。<span class="marginnote">`await` 在 `for...of` 里可以逐项等待，配合异步迭代器（`for await...of`）能流式处理分页数据。生成器 + Promise 的组合就是「异步生成器」，第18篇的伏笔在此兑现。</span>

## 6 公式解析：Promise 状态机

Promise 的全部行为可以建模成一个**状态机**——这也是它可靠性的数学基础：

$$
S \in \{\text{pending}, \text{fulfilled}, \text{rejected}\}, \qquad \text{transitions: } \text{pending} \rightarrow \begin{cases} \text{fulfilled} & (\text{resolve}) \\ \text{rejected} & (\text{reject}) \end{cases}
$$