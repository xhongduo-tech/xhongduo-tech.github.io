---
title: 迭代器与生成器
date: 2026-08-07
---

# 迭代器与生成器

<div class="epigraph">
<p>真正的优雅，是按需生产，而非提前囤积。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 官方 Python 教程 第5、9章 ｜ 2026-08-07</p>
</div>

## 为什么从迭代开始

从列表到字典，我们一直在用 `for x in 容器:` 遍历——它方便得像魔法。但 Python 里这种「魔法」背后有一套清晰的协议：**迭代协议（iteration protocol）**。一旦理解了它，你就能让**自己的类**也能被 `for` 遍历，还能写出**生成器（generator）**——一种按需产出数据、几乎不占内存的函数。

官方 Python 教程第 5 章（数据结构）与第 9 章（类）都触及迭代主题。本节把三条线索收拢：迭代协议、生成器、惰性求值。

## 1 可迭代对象与迭代器

先分清两个概念：

**可迭代对象（iterable）**：可以放进 `for` 循环的东西——列表、元组、字典、字符串、`range`。它实现了 `__iter__`，返回一个迭代器。

**迭代器（iterator）**：记录「当前走到哪了」的对象，实现了 `__next__`，每次调用吐出下一个元素，元素耗尽后抛 `StopIteration`。

```python
numbers = [1, 2, 3]
it = iter(numbers)         # iter() 调用 __iter__，拿到迭代器
print(next(it))            # 1，next() 调用 __next__
print(next(it))            # 2
print(next(it))            # 3
# print(next(it))          # StopIteration：耗尽
```

**重点：迭代器是「一次性」的。** 迭代器就像一支已插入的书签，读过头就回不去了。但可迭代对象（列表）不是——每次 `iter()` 都给你一支新的书签。所以「列表可以反复遍历，迭代器只能走一遍」：`for x in it:` 第二次就是空的。<span class="marginnote">字典的迭代默认只产键：`for k in d:` 等价于 `for k in d.keys():`。若要同时拿键值，用 `d.items()`——它返回的就是一个视图迭代器，这就是 `for k, v in d.items()` 的底层机制。</span>

## 2 生成器：yield 暂停与恢复

**生成器（generator）**：用 `yield` 定义的特殊函数。调用它不会执行函数体，而是返回一个**生成器对象**；每次 `next()` 执行到下一个 `yield` 就暂停，把值交给调用方。

```python
def countdown(n):
    print("开始倒数")
    while n > 0:
        yield n            # 产出一个值并暂停
        n -= 1
    print("倒计时结束")

for num in countdown(3):
    print(num)
# 开始倒数 → 3 → 2 → 1 → 倒计时结束
```

**重点：`yield` 把函数变成「可暂停的序列」。** 关键在于**状态保留**——生成器记得自己的局部变量与执行位置，下次 `next()` 从暂停处继续。这与普通函数的「每次从头跑」完全不同。函数里的 `return` 与 `yield` 还有一个关系：`return` 可以提前结束生成器（并终止迭代）。

**生成器表达式**是列表推导的惰性兄弟：

```python
squares = (x ** 2 for x in range(1000000))    # 圆括号，不是方括号
print(sum(squares))                            # 一千万以内平方和，内存占用极小
```

## 3 惰性求值：按需生产的大数据哲学

**惰性求值（lazy evaluation）**：需要时才计算，而不是提前算好存起来。生成器是惰性的典型代表：

```python
def read_large_file(path):
    with open(path, encoding="utf-8") as f:
        for line in f:            # 逐行读取，不把整个文件装进内存
            yield line

for line in read_large_file("data.csv"):
    pass                          # 处理一行、丢弃一行
```

**重点：惰性让「无限」成为可能。** 生成器不存储全部结果，所以能描述「无限序列」而不爆内存：

```python
def fibonacci():
    a, b = 0, 1
    while True:                    # 无限循环，但不会卡死
        yield a
        a, b = b, a + b

fib = fibonacci()
for _ in range(10):
    print(next(fib), end=" ")      # 0 1 1 2 3 5 8 13 21 34
```

只要按需取前几个，无限序列也安全——这正是「先看取多少，再谈生成多快」。数据科学里处理超大文件、流式 API、无限数据流时，惰性求值是标配哲学。<span class="marginnote">惰性思想在本专题反复出现：`range()` 的惰性本质（《深入流程控制》）、`readlines` vs 逐行迭代（《输入输出与文件读写》）。它也是函数式语言（如 Haskell）的根基，与第二级《函数式程序设计》的精神一脉相承。</span>

## 4 公式解析：for 循环的机械展开

**`for x in seq:` 其实是一条迭代协议的机械展开。** 把「语法糖」剥开，等价于：

$$
\begin{aligned}
\text{it} &= \text{iter}(\text{seq}) \\
\text{while True:} &\quad x = \text{next}(\text{it}) \\
&\quad \text{执行循环体} \\
&\quad \text{捕获 StopIteration 后退出}
\end{aligned}
$$

对这条流程做三步拆解：

- **第一步，取迭代器**：`iter(seq)` 调用 `seq.__iter__()`，得到迭代器 `it`。字符串、列表、字典都能被 `iter`，因为它们都实现了 `__iter__`。
- **第二步，反复 next**：每轮循环调用 `next(it)`，等价于 `it.__next__()`，取下一个元素赋给 `x`。
- **第三步，捕异常退出**：元素耗尽时 `__next__` 抛 `StopIteration`，`for` 循环悄悄捕获它并结束——所以你不必手动处理「迭代到头」。

**为何要理解展开？** 因为它解释了三条经验：迭代器是**一次性**的（书签机制）、自定义类实现 `__iter__` + `__next__` 即可被 `for` 遍历（见《特殊方法》）、以及 `for` 比 `while + 下标` 更接近「数据流」的本来面貌——后者无法处理不支持下标的数据源。

## 5 小结

- **可迭代对象**实现 `__iter__`，**迭代器**实现 `__next__` 并在耗尽时抛 `StopIteration`。
- 迭代器是**一次性**的，列表等可迭代对象可反复遍历。
- **生成器**用 `yield` 定义：调用不执行、每次 `next` 到 `yield` 暂停并保留状态。
- **惰性求值**让超大文件、无限序列得以处理；生成器表达式是推导式的惰性版。
- `for x in seq` 的底层是 `iter` + 循环 `next` + 捕获 `StopIteration`。

在下一节，我们将从「对象会出什么错」转向「出错后怎么办」——异常处理与调试技巧。
