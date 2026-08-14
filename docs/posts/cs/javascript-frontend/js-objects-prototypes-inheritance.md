---
title: 对象、原型与继承
date: 2026-08-07
---

# 对象、原型与继承

<div class="epigraph">
<p>JavaScript 的对象系统是一张由原型织成的网，而不是一棵从类长出的树。</p>
<footer>—— 凯尔 · 辛普森（Kyle Simpson）</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ JS高级程序设计 第8章 ｜ 2026-08-07</p>
</div>

## 为什么从对象与原型开始

原始类型是 JS 的基本积木，而**对象（object）**是让积木成形的胶水——几乎所有非原始值都是对象，几乎所有复杂数据都由对象承载。这一节讲清三件事：**对象怎么写**（字面量、属性、解构）、**原型链怎么工作**（JS 的继承机制）、**class 语法**（现代写法下的原型继承）。

理解原型链是理解 JS 的关键分水岭：别的语言用「类继承」，JS 用「**原型继承**」——对象通过一条内部链（`[[Prototype]]`）找到不属于自己的属性。DOM 元素、函数、数组……一切对象的「自带方法」都来自原型链。甚至 `class` 也只是原型继承的**语法糖**——看清这层糖纸，你就再不会被「JS 到底有没有类」这类问题困扰。<span class="marginnote">Zakas 第8章「对象、类与面向对象编程」：从属性特性到创建对象模式、原型、继承、ES6 class 全覆盖。DOM 的 `document.querySelector`、数组的 `map` 全是原型方法——原型链不是理论，是浏览器每一刻都在运行的机制。</span>

## 1 对象字面量与属性

创建对象的最简方式——**对象字面量（object literal）**：

```js
const user = {
  name: "小明",
  age: 18,
  greet() {                     // 方法简写
    return `你好，我是${this.name}`;
  },
  ["tag_" + "1"]: true,          // 计算属性名：方括号内放表达式
};
```

**属性名**默认是字符串（或 Symbol）；访问用点号 `user.name` 或方括号 `user["name"]`——方括号里可以是变量或表达式，是「动态取属性」的途径。
**方法简写** `greet() {…}` 等价于 `greet: function () {…}`。
- **计算属性名** `[expr]` 允许在字面量里动态生成键名。

**对象内容的三种遍历方式**：

```js
Object.keys(user);        // ["name", "age", "tag_1"]
Object.values(user);      // ["小明", 18, true]
Object.entries(user);     // [["name","小明"], ...]
```

`Object.entries()` 配 `for...of` 是遍历对象的现代标准写法，返回 `[键, 值]` 对数组，可配合解构。

**解构（destructuring）** 从对象/数组里提取值：

```js
const { name, age } = user;              // 对象解构：按名取
const [first, second] = [1, 2, 3];       // 数组解构：按位取
const { name: alias = "匿名" } = user;   // 改名 + 默认值
```

解构配合展开、rest，是现代 JS 处理数据的「三件套」——函数参数、状态更新、组件取值处处可见。<span class="marginnote">函数参数解构是高频模式：`function render({ title, items }) {…}` 直接把对象参数拆开用，调用方 `render({ title, items })` 也一目了然——「参数即文档」。</span>

## 2 属性访问的防弹衣：可选链与空值合并

访问嵌套对象属性时，任何一层为 `null`/`undefined` 都会抛 `TypeError`。两个现代运算符根治这个问题：

```js
const city = user?.address?.city ?? "未知城市";
//  ?. 可选链：中间任一层为 null/undefined，整个表达式短路为 undefined
//  ?? 空值合并：左边是 null/undefined 时取右边
```

**可选链（optional chaining）** `?.`：访问前先检查「值存在吗」，不存在就短路返回 `undefined`——不再需要一长串 `if (a && a.b && a.b.c)`。
**空值合并（nullish coalescing）** `??`：只在左边为 `null`/`undefined` 时取默认值——与 `||` 不同，`||` 对 `0`、`""`、`false` 也会取默认，而 `??` 不会。

```js
const n = 0;
const a = n || 100;   // 100 —— 0 是假值，被替换了！
const b = n ?? 100;   // 0 —— 0 不是 null/undefined，保留
```

**辨析｜易错点：** `??` 与 `||` 的选择——想让「0、空串」也合法保留，用 `??`；想做「假值兜底」，用 `||`。还要注意 `??` **不能与 `||`/`&&` 混用不括号**：`a ?? b || c` 会抛语法错误，必须写 `(a ?? b) || c`。<span class="marginnote">`?.` 与 `??` 是 ES2020 加入的，如今浏览器全面支持。它们的组合 `data?.items?.[0]?.name ?? "空"` 一行写出「深层安全取值 + 兜底」——传统 JS 要写五层 if。</span>

## 3 原型与原型链：JS 的继承引擎

每个对象都有一个内部属性 **`[[Prototype]]`**（可通过 `Object.getPrototypeOf(obj)` 读取），指向**另一个对象**——它的「原型」。当访问一个对象没有的属性时，JS 会**沿原型链向上查找**：

```js
const arr = [1, 2, 3];
arr.map(x => x * 2);        // map 不在 arr 上，在 Array.prototype 上

Object.getPrototypeOf(arr) === Array.prototype;   // true
```

**原型链的结构**：

```
arr → Array.prototype → Object.prototype → null
```

`arr` 自身只有元素；`map`、`forEach` 等住在 `Array.prototype`。
任何对象的原型链最终都通到 `Object.prototype`，再往上是 `null`。
- `Object.prototype` 上有 `toString`、`hasOwnProperty` 等「万物皆有的方法」。

**为什么这样设计？** 原型让「共享行为」不用复制——所有数组共用同一份 `map` 函数，内存省、更新方便。**继承**的本质就是：新对象把自己的原型指向「父对象」，于是「父对象有的，我查得到」。

**构造函数的原型**：用 `new` 调用的函数（构造函数），其 `prototype` 属性会成为实例的原型：

```js
function Animal(name) {
  this.name = name;
}
Animal.prototype.speak = function () {
  return `${this.name}在叫`;
};
const dog = new Animal("旺财");
dog.speak();   // "旺财在叫" —— 沿原型链找到 speak
```

`new` 做的事：创建新对象 → 把新对象的 `[[Prototype]]` 指向 `Animal.prototype` → 执行构造函数（`this` 指向新对象）→ 返回新对象。<span class="marginnote">区分两个相似词：函数的 `prototype` 属性（给 new 出的实例当原型）与实例的 `[[Prototype]]`（内部链）。`dog` 没有 `.prototype`，`Animal.prototype` 才是 `dog` 的原型。术语不混，原型链就清楚了。</span>

## 4 class：原型继承的语法糖

ES6 的 **`class`** 不引入新机制，只是把「构造函数 + 原型方法」包成更清晰的语法：

```js
class Animal {
  constructor(name) { this.name = name; }
  speak() { return `${this.name}在叫`; }       // 自动进 Animal.prototype
  static create(name) { return new Animal(name); }  // 静态方法：挂在类上
}

class Dog extends Animal {                      // 继承
  constructor(name) { super(name); this.loyal = true; }  // super 调父构造函数
  speak() { return `${super.speak()}，汪！`; }   // super 调父方法
}
```

**`constructor`** 是初始化钩子，`new` 时自动执行。
类内定义的方法自动放在 `prototype` 上，实例共享。
- **`extends`** 建立原型链：`Dog.prototype` 的原型指向 `Animal.prototype`。
- **`super`** 在子类里调用父类构造函数或方法——**子类构造函数里必须先 `super(...)` 再用 `this`**，否则报错（这是 JS 强制「先初始化父类」的规矩）。

**辨析｜易错点：** class 与「类继承」语言的区别——class 的继承链仍是**原型链**：`dog instanceof Dog` 为 true，因为它沿原型链找到了 `Dog.prototype`。而且原型链是**动态的**：`Dog.prototype.speak = 新函数` 会在运行时影响所有实例（包括已创建的），这是静态类语言做不到的。class 只是「好看的原型」。<span class="marginnote">`instanceof` 的语义：沿 `左侧` 的原型链查找，看是否能遇到 `右侧.prototype`。它查的是「原型链」而不是「类型标签」——理解这点，就能解释为什么 `[] instanceof Object` 也是 true。</span>

## 5 对象创建与拷贝

对象创建的几种方式，对应不同需求：

```js
const obj = {};                          // 字面量
const obj = Object.create(proto);        // 以 proto 为原型创建
const obj = new Object();                // 等价于 {}
```

**`Object.create(proto)`** 是原型继承的「裸操作」——指定原型直接创建对象。它也是「纯原型继承」写法的核心（不借助 class）：

```js
const animal = { speak() { return "在叫"; } };
const dog = Object.create(animal);       // dog 的原型是 animal
dog.speak();                             // 沿原型链找到
```

**拷贝对象**有三档：

| 方式 | 深度 | 说明 |
| --- | --- | --- |
| `{ ...obj }` | 浅 | 常用，嵌套对象仍是引用 |
| `Object.assign({}, obj)` | 浅 | 老写法，等价于展开 |
| `structuredClone(obj)` | 深 | 现代深拷贝，可复制循环引用 |

**辨析｜易错点：** 浅拷贝的陷阱——`const b = { ...a }; b.nested.x = 1` 会**同时改到 `a.nested.x`**，因为 `nested` 是共享引用。想完全独立，必须深拷贝（`structuredClone`）或逐层复制。这也是 React 里「更新嵌套状态要每一层都展开」的原因（呼应第4篇的不可变更新）。

## 6 公式解析：属性查找沿原型链上行

原型继承的一切行为，都可以收敛成一条**查找公式**：

$$
\text{value}(p) = \begin{cases} \text{own}(p) & \text{if } p \in \text{ownProperties} \\ \text{value}(\text{proto}(p)) & \text{if } p \notin \text{ownProperties 且 proto}(p) \ne \text{null} \\ \text{undefined} & \text{otherwise} \end{cases}
$$