---
title: 函数、作用域与闭包
date: 2026-08-07
---

# 函数、作用域与闭包

<div class="epigraph">
<p>JavaScript 最强大的特性，是把函数当作可以传递、可以返回、可以随身携带的值。</p>
<footer>—— 凯尔 · 辛普森（Kyle Simpson），《你不知道的 JavaScript》</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ JS高级程序设计 第10章 ｜ 2026-08-07</p>
</div>

## 为什么从函数与作用域开始

数据有了，流程有了，但把代码组织成「可复用、可组合」的单元，靠的是**函数**。函数是 JS 的**一等公民（first-class citizen）**：它可以像数字、字符串一样被赋值、传参、返回。这个看似简单的性质，催生了 JS 的整个函数式风格——数组的 `map`/`filter`、事件的回调、异步的 Promise，全部建立在「函数可以传来传去」之上。

而**闭包（closure）**，是「函数 + 词法作用域」碰撞出的最精妙的机制。理解闭包，等于理解 JS 里一半的高级技巧与一半的经典 bug——循环里的 `var` 陷阱、模块封装、防抖节流，都逃不出它的掌心。这一节是全专题的知识密度高峰之一，值得放慢脚步。<span class="marginnote">Zakas 第10章「函数」是全书最重要的章节之一：函数声明/表达式、参数、this、闭包、尾调用。这里按「定义方式→作用域→this→闭包」的认知链重排，让「闭包为什么成立」建立在「作用域如何嵌套」之上。</span>

## 1 三种函数写法

```js
// 1. 函数声明：会提升，可在声明前调用
function add(a, b) {
  return a + b;
}

// 2. 函数表达式：不提升，赋值后才可用
const add = function (a, b) {
  return a + b;
};

// 3. 箭头函数：无自己的 this，最简洁
const add = (a, b) => a + b;
```

三者差异的关键维度：

| 维度 | 函数声明 | 函数表达式 | 箭头函数 |
| --- | --- | --- | --- |
| 提升 | 整体提升 | 只提升变量名 | 同左 |
| 自己的 `this` | 有 | 有 | **无**（继承外层） |
| `arguments` | 有 | 有 | **无** |
| 可作构造函数 | 可 | 可 | **不可** |
| 简洁度 | 中 | 中 | 高 |

**箭头函数是「瘦身版」**：语法最简（单参可去括号、单句可省 `return`），但代价是——它没有自己的 `this`、没有 `arguments`、不能当构造函数。这个「没有自己的 this」恰恰是它最实用的地方：回调函数里 `this` 不会丢失（见第3节）。<span class="marginnote">箭头函数没有 `arguments`，但有 rest 参数（`(...args) =>`）替代。现代代码里「纯计算、无 this 需求」的函数一律箭头函数，有 this 需求或要当构造函数才用 function。这个约定让代码一眼可读。</span>

## 2 参数：默认值、rest 与 arguments

函数参数有三种进阶写法：

```js
function greet(name, greeting = "你好") {   // 默认值：undefined 时生效
  return `${greeting}，${name}`;
}

function sum(...nums) {                     // rest 参数：收拢所有实参为数组
  return nums.reduce((a, b) => a + b, 0);
}

function log() {
  console.log(arguments);                   // 类数组：所有实参（箭头函数无）
}
```

**默认值**在实参为 `undefined` 时生效——注意传 `null` 不算缺省，默认值不触发。
**rest 参数** `...nums` 是「收集剩余实参」的现代写法，比 `arguments` 更好用——它是真正的数组，可用所有数组方法。
- `arguments` 是历史遗留的类数组对象，箭头函数里不可用；新代码优先 rest。

**`...`（展开运算符）是 rest 的镜像**：rest 是「收集成数组」，展开是「摊开成元素」：

```js
const arr = [1, 2, 3];
const copy = [...arr];          // 浅拷贝数组
Math.max(...arr);               // 把数组摊成独立实参
const merged = { ...obj1, ...obj2 };  // 对象展开合并
```

展开运算符是现代 JS 处理不可变数据、合并对象数组的默认工具，第17篇还会大量用到。<span class="marginnote">展开做的是<strong>浅拷贝</strong>：`[...arr]` 只复制最外一层，嵌套的对象仍是引用。深拷贝需要 `structuredClone` 或递归——这个「浅 vs 深」的边界是面试高频题，也是日常 bug 的来源。</span>

## 3 this：四种绑定与箭头函数的豁免

`this` 是 JS 里最著名的难点。它在**调用时**确定，取决于「怎么调」，有四种绑定规则：

```js
// 1. 默认绑定：裸调用，非严格模式指向全局，严格模式 undefined
function f() { console.log(this); }
f();                              // window / undefined

// 2. 隐式绑定：obj.method()，this 指向 obj
const obj = { name: "o", f: function() { console.log(this.name); } };
obj.f();                          // "o"

// 3. 显式绑定：call/apply/bind 指定 this
f.call(obj);                      // this 指向 obj

// 4. new 绑定：this 指向新建对象
function Person(n) { this.name = n; }
new Person("x");                  // this 指向新对象
```

**this 丢失的经典陷阱**：把方法当回调传递时，隐式绑定就断了：

```js
const obj = { name: "o", f: function() { console.log(this.name); } };
setTimeout(obj.f, 1000);   // undefined！this 变回了默认绑定
```

三个修复：`obj.f.bind(obj)`、箭头函数包装 `() => obj.f()`、或者干脆方法就用箭头函数定义。<span class="marginnote">箭头函数<strong>没有自己的 this</strong>——它的 this 是「词法继承」：定义时外层函数的 this 是什么，它就是什么，且永远不变（`call`/`apply`/`bind` 也无法改它）。所以回调里用箭头函数，this 天然指向定义处的外层——这是它在 React 事件回调里大行其道的原因。</span>

## 4 作用域：词法作用域与链

**作用域（scope）** 是「名字到值的查找范围」。JS 是**词法作用域（lexical scope）**：函数能访问哪些变量，**在定义时**就由代码的书写位置决定了，与调用位置无关。

三种作用域层次：

**全局作用域**：顶层声明的变量，处处可访问。
**函数作用域**：函数内部（`var` 声明）——函数外访问不到，但函数内可访问外层。
- **块作用域**：`let`/`const` 只在最近的 `{}` 内。

作用域形成一条**链**：函数内用到一个变量时，先查自己，再往外层一层层查，直到全局——查不到就 `ReferenceError`。这就是「作用域链（scope chain）」的查找机制。

```js
const x = 1;                 // 全局
function outer() {
  const y = 2;               // outer 的局部
  function inner() {
    const z = 3;             // inner 的局部
    console.log(x + y + z);  // 沿链向上全部可见
  }
}
```

**辨析｜易错点：** 变量查找靠「定义位置」而非「调用位置」。`inner` 在 `outer` 外部被调用，它依然能访问 `outer` 里的 `y`——因为 `y` 在它定义时就是其外层作用域。这正是闭包能成立的根基：**函数定义时就「记住了」它的外层环境**。<span class="marginnote">「词法」二字的意思是「跟代码文本的位置有关」：你读代码时的嵌套结构，就是作用域的嵌套结构。对比动态作用域（看调用栈），词法作用域让函数「自包含」——理解这一点，闭包就只剩「记忆」二字。</span>

## 5 闭包：函数带着它的背包

**闭包（closure）** 的定义：**一个函数连同它定义时捕获的外部变量，一起构成的组合**。当内层函数引用了外层函数的变量，即使外层函数已经返回，内层函数依然「记得」那些变量：

```js
function counter() {
  let count = 0;
  return function () {
    count++;                  // 引用外层局部变量
    return count;
  };
}
const c = counter();
c();  // 1
c();  // 2   count 没有被回收，被闭包「抓住」了
```

**闭包是怎么工作的？** 每次调用 `counter()`，都创建一个新的词法环境，`count` 活在其中。返回的匿名函数引用这个环境，于是环境因被引用而不被垃圾回收——`count` 就「活了下来」，且**每次调用 `counter()` 都是独立的一份**。

闭包的三大实用场景：

1. **私有变量**：模拟「模块」——外部只能通过返回的函数操作内部状态。
2. **工厂函数**：一次性生成「绑定好配置」的函数。
3. **回调记忆**：事件监听、防抖节流里「记住上一次的时间戳」。

**防抖（debounce）** 是闭包的经典工程用例：连续触发的事件，只在停止后执行一次——闭包负责「记住定时器」：

```js
function debounce(fn, delay) {
  let timer = null;                  // 闭包变量：记住定时器
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}
```

<span class="marginnote">闭包的内存代价：被闭包引用的变量不会被回收。若闭包长期存活且持有大对象，就造成内存泄漏——这也是为什么「监听器不再需要时要 removeEventListener」。权衡：闭包换来了封装，但要为它的生命周期负责。</span>

## 6 公式解析：循环与闭包的经典陷阱

闭包最难的一课，藏在「循环 + 回调」里。看这个「经典错误」：

$$
\text{for}\,(i) \;\Rightarrow\; \text{closure}_{j} \;\text{captures}\; i
$$