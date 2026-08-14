---
title: 迭代器与生成器
date: 2026-08-07
---

# 迭代器与生成器

<div class="epigraph">
<p>生成器让函数学会暂停——它不是把结果一次算完，而是「每次要一点，每次停一下」。</p>
<footer>—— 尼古拉斯 · 扎卡斯（Nicholas C. Zakas）</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ JS高级程序设计 第7章 ｜ 2026-08-07</p>
</div>

## 为什么从迭代器与生成器开始

`for...of` 能遍历数组、字符串、`Map`、`Set`——为什么它们都能被 `for...of` 遍历？因为背后有一个统一的协议：**迭代器协议（iterator protocol）**。这一节把这个「隐形的引擎」点破：什么是可迭代对象、迭代器怎么工作、生成器如何让「产生值」变成「按需暂停」。

学习它的意义远超语法本身。**惰性求值（lazy evaluation）**——用多少算多少——是这套机制的灵魂：无穷数列、流式处理、超大文件逐行读，都靠它不必一次性造出全部数据。而生成器（generator）是「写了暂停点的函数」，它把「状态机」写成了普通代码——这是从「命令式循环」到「声明式数据流」的又一次升级。第19篇的异步迭代、第4篇的 DOM 遍历，也都建立在它之上。<span class="marginnote">Zakas 第7章「迭代器与生成器」。迭代器是语言层面的「遍历协议」，生成器是「便捷生成迭代器」的语法。两者配合，是 ES6 为「序列处理」提供的完整方案。</span>

## 1 迭代器协议：next() 与 done

**可迭代对象（iterable）**：实现了 `Symbol.iterator` 方法的对象。任何 `for...of` 能遍历的东西都是可迭代的。

**迭代器（iterator）**：调用 `Symbol.iterator` 返回的对象，它有 `next()` 方法，每次调用返回一个**迭代结果对象**：

```js
{ value: 下一个值, done: boolean }
```

```js
const arr = [10, 20, 30];
const it = arr[Symbol.iterator]();   // 拿到迭代器

it.next();  // { value: 10, done: false }
it.next();  // { value: 20, done: false }
it.next();  // { value: 30, done: false }
it.next();  // { value: undefined, done: true }  数据耗尽
```

`for...of` 就是它的语法糖：循环内部反复调 `next()`，`done` 为 true 就停止。

**为什么需要这套协议？** 因为「遍历」应该与「数据结构内部实现」解耦——数组、Set、Map 内部结构完全不同，但都通过同一协议暴露「顺序访问」，`for...of`、展开运算符、解构就都能统一工作。<span class="marginnote">展开运算符 `[...x]`、`Array.from(x)`、解构 `const [a, b] = x`，全都依赖迭代器协议。所以「让我的类支持 for...of、支持展开」只需实现 `[Symbol.iterator]()`——协议的力量在于一处实现、处处受益。</span>

## 2 自定义可迭代对象

让一个普通对象可迭代，只需实现 `Symbol.iterator`：

```js
const range = {
  from: 1, to: 5,
  [Symbol.iterator]() {
    let current = this.from;
    return {
      next: () => current <= this.to
        ? { value: current++, done: false }
        : { value: undefined, done: true },
    };
  },
};

for (const n of range) console.log(n);   // 1 2 3 4 5
```

闭包在这里扮演核心角色：`current` 被迭代器闭包捕获，每次 `next()` 更新它。迭代器本质是一个**记住当前位置的可调用对象**——这是「惰性序列」的最小实现。

**辨析｜易错点：** 迭代器是**一次性**的——`next()` 走到 `done: true` 后，再调也只会返回 `{ done: true }`。想重新遍历，得重新调用 `[Symbol.iterator]()` 拿新迭代器。`for...of` 每次都重新拿迭代器，所以能反复遍历；手动保存的迭代器则「用一次就没」——这是最常见的理解偏差。<span class="marginnote">迭代器「一次性」的直觉来源：它是<strong>游标（cursor）</strong>，不是数据本身。数据库游标、文件指针都是同一思想——记住位置、按需取、用尽即弃。</span>

## 3 生成器：暂停的函数

**生成器函数（generator function）** 是返回迭代器的一种特殊函数，用 `function*` 声明、`yield` 暂停：

```js
function* counter() {
  yield 1;    // 暂停点：产出 1，等待下次 next()
  yield 2;
  yield 3;
}

const gen = counter();
gen.next();   // { value: 1, done: false }  停在第一个 yield
gen.next();   // { value: 2, done: false }  继续走到第二个 yield
```

**生成器与普通函数的根本区别**：普通函数一次跑完、返回一个值；生成器**跑到第一个 `yield` 就暂停**，把控制权交回调用方，下次 `next()` 再从中断处继续。每次 `next()` 都对应「跑到下一个 `yield`」。

**`yield` 可以接收值**——`next(传入值)` 会成为 `yield` 表达式的值，实现「函数与调用方双向通信」：

```js
function* ask() {
  const name = yield "你叫什么？";   // 先发问，等回答
  yield `你好，${name}`;
}
```

这是协程（coroutine）的雏形——两个执行流交替推进。生成器是 JS 实现「可暂停计算」的最轻量机制。

## 4 惰性求值：无穷序列

生成器最大的实际价值是**惰性**：值在需要时才计算，因此可以表达「无穷」：

```js
function* fibonacci() {
  let [a, b] = [0, 1];
  while (true) {            // 永不结束的循环——但它不是死循环！
    yield a;
    [a, b] = [b, a + b];
  }
}

const fib = fibonacci();
fib.next().value;  // 0
fib.next().value;  // 1
fib.next().value;  // 1
fib.next().value;  // 2   // 每次只算一个，永远不会爆内存
```

**为什么 `while (true)` 不是死循环？** 因为 `yield` 让循环每次在产出后**暂停**——计算被「冻结」在下一次 `next()` 之前。这就是惰性求值的本质：**循环结构 + 暂停点 = 无限可续流**。<span class="marginnote">对比「一次算完」：若用数组造前 1 亿个斐波那契数，内存直接爆掉；生成器方案内存恒定 O(1)。处理超大文件、无限数据流时，「不全部载入」是唯一的正确姿势。</span>

**`yield*` 委托**：在一个生成器里把另一个可迭代对象「让位出去」：

```js
function* combined() {
  yield* [1, 2, 3];        // 等价于逐个 yield 1,2,3
  yield* "ab";
}
```

`yield*` 自动迭代被委托的对象，把它的每个值转发出去——是「拼接多个序列」的优雅写法，也是实现递归生成器（如深度优先遍历树）的关键。

## 5 生成器驱动：手动迭代到同步控制流

生成器可以接收 `next(arg)` 注入数据，这催生了「生成器驱动的流程控制」——早期异步库（`co`）的核心思想：把异步代码写成同步样子的生成器，外层驱动它推进：

```js
function* flow() {
  const user = yield fetchUser();      // 假装同步，实际暂停等结果
  const posts = yield fetchPosts(user.id);
  return posts;
}
```

`next()` 把 `fetchUser()` 的结果塞回 `yield` 处，`flow` 就像「同步代码」一样继续——异步被生成器「暂停 + 续传」机制抹平了。这套思想后来被 **async/await**（第19篇）继承，await 就是「yield + Promise」的语法糖——理解生成器，等于提前看到了 async/await 的引擎。<span class="marginnote">历史上 `co` 库用生成器让异步代码看起来同步，那时 Promise 还没有 `async/await`。如今 `async/await` 内置了这套机制，但「yield 暂停、注入续传」的心智模型与 await 完全一致——学透生成器，await 只是换个名字。</span>

## 6 核心对比表：迭代器与生成器

| 维度 | 迭代器（协议） | 生成器（语法） |
| --- | --- | --- |
| 本质 | 一个协议：`next()` 返回 `{value, done}` | 一种函数：`function*` + `yield` |
| 谁在写 | 手动实现 `[Symbol.iterator]` | 引擎自动生成迭代器 |
| 暂停能力 | 无（调用方控制节奏） | 有（`yield` 内建暂停点） |
| 适合 | 让自定义对象可遍历 | 惰性序列、状态机、协程 |
| 与异步 | 同步为主 | `yield` 可接 Promise，async/await 前身 |

**辨析｜易错点：** 生成器函数调用后**不会立即执行**——`counter()` 返回的是待命迭代器，函数体要等第一次 `next()` 才开始跑。所以：

```js
function* f() { console.log("开始"); yield 1; }
const g = f();     // 什么都没打印
g.next();          // 此刻才打印 "开始"
```

生成器是「延迟到第一次 next 才执行」的——这与普通函数「调用即执行」截然不同，是生成器最容易踩的认知差。<span class="marginnote">生成器的「冷启动」特性让它可以表达「按需初始化」：昂贵的资源准备放到第一个 `yield` 之前，不 `next()` 就不花钱。这也让生成器成为「懒加载数据源」的理想实现。</span>

## 7 小结

- **迭代器协议**：可迭代对象实现 `[Symbol.iterator]`，返回带 `next()` 的迭代器，`next()` 产出 `{ value, done }`。
- `for...of`、展开、解构都依赖迭代器协议；**一处实现，处处受益**。
- 迭代器是**一次性游标**：用尽即弃，重新遍历要重新拿迭代器。
- **生成器** `function*` + `yield`：跑至 `yield` 暂停，`next()` 续传，还能用 `next(arg)` 回传数据。
- **惰性求值**让 `while (true)` 的无穷序列内存恒定——用多少算多少，处理大数据的正确姿势。
- `yield*` 委托拼接序列；生成器是 async/await 的前身，理解它等于提前看到异步的引擎。

在下一节，我们正式进入异步世界——**Promise 与异步编程**。浏览器里最耗时的操作（网络请求、定时器、文件读写）都是异步的，Promise 与 async/await 是驾驭它们的现代武器。
