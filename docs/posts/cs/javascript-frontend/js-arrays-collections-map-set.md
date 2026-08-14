---
title: 数组与集合类型 Map/Set
date: 2026-08-07
---

# 数组与集合类型 Map/Set

<div class="epigraph">
<p>数据结构的价值不在存储，而在它让哪些操作变快、哪些表达变短。</p>
<footer>—— 尼古拉斯 · 扎卡斯（Nicholas C. Zakas）</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ JS高级程序设计 第6章 ｜ 2026-08-07</p>
</div>

## 为什么从数组与集合开始

真实程序处理的是**批量数据**：一组列表项、一批用户、一列成绩。JS 提供三类主力容器——**数组（Array）**：有序可重复，按索引访问；**Map**：键值对集合，键可为任意类型；**Set**：唯一值集合，自动去重。这三者覆盖了「有序列表」「字典」「集合」三种基本需求。

数组尤其重要——它是 JS 里**最常用**的数据结构，几乎所有批量数据操作都在数组上展开。而数组的 `map`/`filter`/`reduce` 三件套，是**函数式风格**的精华：把「怎么遍历」交给标准库，你只声明「每个元素怎么处理」。学会它们，代码会从「怎么做的循环」变成「要什么的声明」，可读性上一个台阶。<span class="marginnote">Zakas 第6章「集合引用类型」覆盖 Array、Map、Set、WeakMap 等。注意：Map/Set 在第16篇的对象之后讲正合适——它们的实现都建立在对象的哈希能力之上，但 API 更友好。</span>

## 1 数组基础：创建、增删与访问

```js
const arr = [1, 2, 3];           // 字面量
const arr2 = new Array(5);       // 长度 5 的空数组（元素都是 empty）
const arr3 = Array.from("abc");  // ['a','b','c'] 类数组转真数组
const arr4 = [...new Set([1,1,2])]; // 配合展开
```

**增删元素**的两端操作：

| 方法 | 作用 | 返回值 |
| --- | --- | --- |
| `push(x)` | 尾部添加 | 新长度 |
| `pop()` | 尾部删除 | 被删元素 |
| `unshift(x)` | 头部添加 | 新长度 |
| `shift()` | 头部删除 | 被删元素 |

注意：`push`/`unshift` 返回**新长度**而非数组本身——想「改完再链式调用」需要别的设计。而 `arr[i]` 访问第 i 个，`arr.length` 是长度；`arr[arr.length - 1]` 取最后一个。

**辨析｜易错点：** `length` 是可写的——`arr.length = 2` 会**截断**数组（删掉后面元素）；`arr.length = 5` 会补空位。别把它当普通属性随手改，这是数组的「裁剪」语义。

**类数组对象**（`arguments`、`NodeList`、`HTMLCollection`）不是真数组，没有 `map`/`filter`。转换用 `Array.from(x)` 或 `[...x]`——第4篇操作 DOM 集合时这是家常便饭。

## 2 遍历与查找：from 到 forEach

遍历数组有四种主流方式：

```js
for (let i = 0; i < arr.length; i++)     // 传统：需要索引
for (const v of arr)                      // 现代：只要值
arr.forEach((v, i) => …)                  // 声明式：有值有索引
arr.map(v => …)                           // 变换并返回新数组
```

**查找类方法**（返回元素而非遍历）：

| 方法 | 找到时返回 | 未找到返回 |
| --- | --- | --- |
| `arr.find(fn)` | 第一个满足的元素 | `undefined` |
| `arr.findIndex(fn)` | 第一个满足的下标 | `-1` |
| `arr.indexOf(v)` | 值首次出现的下标 | `-1` |
| `arr.includes(v)` | `true` | `false` |
| `arr.some(fn)` | `true`（任一满足） | `false` |
| `arr.every(fn)` | `true`（全部满足） | `false` |

**辨析｜易错点：** `find` 与 `filter` 一字之差——`find` 返回**第一个**匹配的元素（单个），`filter` 返回**所有**匹配的数组。判「有没有」用 `some`/`includes`，判「满不满足全部」用 `every`——选对方法，语义自明。<span class="marginnote">`indexOf`/`includes` 用<strong>严格相等</strong>（`===`）比较，所以 `[NaN].includes(NaN)` 是 true（includes 用 SameValueZero），但 `[NaN].indexOf(NaN)` 是 -1——NaN 的「不等于自己」在两者间表现不同，细节决定行为。</span>

## 3 三件套：map、filter 与 reduce

这是数组的**函数式核心三件套**，全部**返回新数组/新值**，不改原数组：

```js
const prices = [10, 25, 40];

const withTax = prices.map(p => p * 1.13);        // 每个元素变换
const affordable = prices.filter(p => p <= 30);   // 按条件筛选
const total = prices.reduce((sum, p) => sum + p, 0);  // 归约成单值
```

**`map`**：一对一变换——长度不变，元素被加工。**不要用 map 做副作用**（`arr.map(console.log)` 是反模式，副作用用 `forEach`）。
**`filter`**：按谓词筛选——只保留返回真值的元素。
- **`reduce`**：把数组「折叠」成一个值——累加器、统计、扁平化都能做，是三者中最强也最抽象的一个。

**链式组合**让数据处理成为「流水线」：

```js
const result = prices
  .map(p => p * 1.13)
  .filter(p => p >= 20)
  .reduce((sum, p) => sum + p, 0);
```

**辨析｜易错点：** 三件套都不改原数组——想「原地排序」用 `sort()`（改原数组），想「得到副本」用 `[...arr].sort()`。`sort()` 还有个经典坑：**默认按字符串排序**——`[10, 2, 1].sort()` 得 `[1, 10, 2]`，必须传比较函数：`arr.sort((a, b) => a - b)`。<span class="marginnote">`sort` 默认把元素转字符串按字典序排，所以数字必须给比较函数。`(a, b) => a - b` 升序、`(b - a)` 降序——返回负/正决定前后，这是「比较器」约定，几乎所有语言通用。</span>

## 4 切片与拼接：slice 与 splice

两个长相相似、语义天差地别的方法：

```js
const copy = arr.slice(1, 3);   // 返回 [1,3) 的新数组，不改原数组
const cut = arr.splice(1, 2);   // 从下标 1 删 2 个，改原数组，返回被删的
```

**`slice(begin, end)`**：纯读取，返回一段**副本**，`end` 不包含。`arr.slice()` 无参即整体浅拷贝——克隆数组的首选。
**`splice(start, deleteCount, ...items)`**：**原地**增删——删了还能插入，是「数组手术刀」。

**辨析｜易错点：** `slice` 是「复制一段」，`splice` 是「切掉一段」——一个不动原数组、一个动原数组，一个不含 end、一个删 count 个。拼错它们，要么数据被意外修改，要么拿到空数组。记忆：splice 里那个 `p` 像「pluck（摘除）」。

## 5 Map 与 Set：现代集合类型

**`Map`** 是键值对集合，键可以是**任意类型**（对象、函数都行），并按插入序迭代：

```js
const cache = new Map();
cache.set("user", { name: "小明" });   // 写入
cache.get("user");                     // 读取，无则 undefined
cache.has("user");                     // 判断存在
cache.delete("user");                  // 删除
cache.size;                            // 条数

for (const [key, value] of cache) {    // 解构遍历
  console.log(key, value);
}
```

**Map 与普通对象的对比**：

| 维度 | Map | 对象 |
| --- | --- | --- |
| 键类型 | 任意值 | 只能是字符串/Symbol |
| 迭代顺序 | 插入序 | 数字键排前、无保证 |
| 常用操作 | `get`/`set`/`has` | 属性访问 |
| 原型污染 | 无 | 继承属性需 `hasOwnProperty` 防御 |

需要**动态增删键**、键非字符串、频繁查存在性时——用 Map；只是「固定结构的数据」，对象更自然。<span class="marginnote">对象作为 Map 的历史陷阱：`obj[key]` 会把 key 隐式转字符串，`obj[{}]` 与 `obj["[object Object]"]` 撞键。Map 没有这个问题——对象键按引用区分。存「以对象为键的缓存」只能用 Map。</span>

**`Set`** 是唯一值集合，自动去重：

```js
const set = new Set([1, 2, 2, 3]);   // {1, 2, 3} 自动去重
set.add(4);                          // 已有则忽略
set.has(2);                          // true
set.delete(2);
[...set];                            // 转回数组

const unique = [...new Set(arr)];    // 数组去重一行版
```

**Set 与数组的取舍**：判「在不在」Set 是 O(1)、数组是 O(n)；Set 保证唯一、数组允许重复。需要「唯一 + 快速查重」用 Set，需要「保序可重复 + 随机访问」用数组。<span class="marginnote">`new Set(arr)` 去重对原始类型精确；对对象，`Set` 按<strong>引用</strong>判等——两个内容相同但不同引用的对象算两个元素。想按内容去重对象，得自己用 Map 以序列化为键。</span>

## 6 公式解析：reduce 如何折叠数组

`reduce` 是三者中最需「公式感」的。它的签名是：

$$
\text{reduce}(f, init) \;\Rightarrow\; v_0 = \text{init},\; v_{i} = f(v_{i-1}, a_i) \;\text{for}\; i = 1 \ldots n
$$