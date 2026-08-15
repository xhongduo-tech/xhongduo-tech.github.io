---
title: 函数式特性：lambda、闭包与高阶函数
date: 2026-08-07
---

# 函数式特性：lambda、闭包与高阶函数

<div class="epigraph">
<p>函数和数据一样，可以被传递、被返回、被组合。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 官方 Python 教程 第4章 ｜ 2026-08-07</p>
</div>

## 为什么从函数式开始

官方 Python 教程第 4 章把 `lambda` 表达式放在「更多流程控制工具」一节。它标志着一次思维转变：此前函数是「被调用的命令」，本节之后函数是**一等公民（first-class citizen）**——可以像整数、字符串一样被存入变量、传入参数、作为返回值。

本节学习四个概念：一等函数、`lambda` 匿名函数、**高阶函数**（`map`/`filter`/`sorted` 的 `key`）、以及**闭包（closure）**。它们是函数式风格的基石，也是装饰器（下一节）的直接前置知识。

## 1 函数是一等公民

**一等公民**：函数可以作为值被操作——赋给变量、存进列表、传给别的函数。

```python
def shout(text):
    return text.upper()

f = shout                 # 不加括号，不调用，只引用
print(f("hello"))         # HELLO，通过 f 调用

functions = [shout, str.lower]
print(functions[0]("hi")) # HI
```

**重点：`shout`（无括号）是函数对象，`shout()`（有括号）才是调用。** 这一字之差是函数式编程的入口——把函数当值传递时**不加括号**。把一个函数作为参数传给另一个函数，后者就叫**高阶函数（higher-order function）**。<span class="marginnote">「函数即值」的思想源自 λ 演算——数学里把它形式化为「一切皆函数」。你在第二级《离散数学》与《函数式程序设计》会看到它的完整展开；Python 只是借用了它的实用子集。</span>

## 2 lambda：一句话函数

**lambda 表达式**：匿名地写一个单表达式函数，用于「用完即扔」的场合：

```python
square = lambda x: x ** 2        # 等价于 def square(x): return x ** 2
print(square(5))                 # 25
```

**重点：`lambda` 只能有一个表达式、没有语句。** 它不能含 `if` 块、`for`、赋值——这些要靠**条件表达式** `x if cond else y` 或三元来绕。它的正确打开方式是「作为参数传给高阶函数」：

```python
pairs = [(1, "apple"), (3, "cherry"), (2, "banana")]
pairs.sort(key=lambda p: p[1])    # 按第二个元素（字符串）排序
print(pairs)                      # [(1, 'apple'), (2, 'banana'), (3, 'cherry')]
```

**辨析｜易错点：** `sorted` 的 `key` 参数接一个「从元素提取排序依据」的函数，`lambda` 是它的最佳搭档。但「有名字、要复用、逻辑超过一行」的函数**不该**用 `lambda`——`def` 更清晰。PEP 8 也建议：能 `def` 就 `def`，`lambda` 只做「内联的短函数」。

## 3 高阶函数：map、filter 与 sorted

函数式风格里，处理序列的三件套：

```python
nums = [1, 2, 3, 4, 5]

squares = list(map(lambda x: x ** 2, nums))       # [1, 4, 9, 16, 25]
evens   = list(filter(lambda x: x % 2 == 0, nums))  # [2, 4]
total   = reduce(lambda a, b: a + b, nums)        # 15（需 from functools import reduce）
```

**重点：`map` 做变换、`filter` 做筛选、`reduce` 做归约。** 三者都返回可迭代对象（`map`/`filter` 惰性），所以外面要套 `list()` 才能看到结果。<span class="marginnote">`functools` 是标准库的「函数式工具箱」：`reduce`、`partial`（部分应用）、`lru_cache`（记忆化）都在这里。我们会在《标准库导览》一节见到它。</span>

**辨析｜易错点：** 现代 Python 更推荐**列表推导**取代 `map`/`filter`——`[x**2 for x in nums]` 与 `[x for x in nums if x % 2 == 0]` 通常更可读。何时用哪个？推导式直观；`map` 与既有函数（如 `map(str, numbers)`）搭配、以及惰性需要时才显优势。**两者等价，选可读的**。

## 4 公式解析：map 与 filter 的展开语义

**`map(f, seq)` 与 `filter(pred, seq)` 可以用集合论语言精确翻译。**

变换（map）：对序列的每个元素施加函数 $f$：

$$
\text{map}(f, S) = \{f(x) \mid x \in S\}
$$

筛选（filter）：保留使谓词 $P$ 为真的元素：

$$
\text{filter}(P, S) = \{x \in S \mid P(x) \ \text{为真}\}
$$

对这两条式子做三步拆解：

- **第一步，读 map**：`map(lambda x: x**2, [1,2,3])` 就是 $\{x^2 \mid x \in \{1,2,3\}\}$——「对集合里的每个元素做变换」。这与列表推导 $\{x^2 \mid x \in S\}$ 完全同构，只差语法。
- **第二步，读 filter**：`filter(lambda x: x % 2 == 0, [1,2,3,4])` 就是「只留下偶数」——$\{x \in S \mid P(x)\}$。它是推导式里 `if` 子句的来源。
- **第三步，看组合**：两者可嵌套——`map(f, filter(P, S))` 先筛选再变换，正好对应「清洗 → 加工」的数据管线。这就是函数式「把流水线写出来」的精髓：每个环节一个纯函数，数据流经它。

**为何引入集合论？** 因为函数式的本质是「描述数据如何流动」，而非「命令机器如何一步步做」——这与第一级《集合的概念》里「用规则定义集合」的思想遥相呼应。理解这一点，装饰器、管道、`itertools` 的用法都会自然浮现。

## 5 闭包：函数记住它的环境

**闭包（closure）**：内层函数捕获并「记住」外层函数的局部变量，即使外层函数已经返回。

```python
def make_multiplier(n):
    def multiplier(x):           # 内层函数
        return x * n             # 捕获了外层变量 n
    return multiplier            # 返回内层函数

double = make_multiplier(2)      # double 记住了 n=2
triple = make_multiplier(3)
print(double(5))                 # 10
print(triple(5))                 # 15
```

**重点：`multiplier` 记住的不只是 `n` 的值，而是整个外围作用域。** 即使 `make_multiplier` 返回了，`double` 仍能读取 `n=2`。这个「随身携带的上下文」就是闭包。<span class="marginnote">闭包与作用域（《函数：定义、参数与返回值》一节的 LEGB）直接相连。若内层函数想<strong>修改</strong>外层变量，需要 `nonlocal` 声明——比如写计数器闭包时，`nonlocal count` 才能让内层函数累加外层计数。</span>

闭包的应用很广：**惰性求值**（暂存参数、稍后再算）、**函数工厂**（按参数定制函数）、以及下一节装饰器的底层机制——装饰器本质上就是一个「接收函数、返回增强版函数」的闭包。

## 6 小结

- 函数是**一等公民**：可赋值、可入参、可返回；引用不加括号，调用才加。
- `lambda x: 表达式` 是单表达式匿名函数，最佳用途是作为 `key` 等参数。
- `map`（变换）、`filter`（筛选）、`reduce`（归约）构成函数式三件套；推导式是更 Pythonic 的等价写法。
- `map(f, S) = {f(x) | x ∈ S}`、`filter(P, S) = {x ∈ S | P(x)}`——与集合论一一对应。
- **闭包**让内层函数记住外层环境，是函数工厂与装饰器的地基。

在下一节，我们将用闭包造出 Python 最优雅的语法糖之一——装饰器，给函数「穿上一层又一层外套」。
