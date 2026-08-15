---
title: 特殊方法（dunder）与运算符重载
date: 2026-08-07
---

# 特殊方法（dunder）与运算符重载

<div class="epigraph">
<p>运算符不过是穿了正装的函数调用。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 官方 Python 教程 第9章 ｜ 2026-08-07</p>
</div>

## 为什么从 dunder 开始

前几节我们定义过 `__init__`——它就是**特殊方法（dunder method）**。这类前后各带双下划线的方法，是 Python 留给对象的「协议钩子」：只要实现了它们，你的对象就能参与 `+`、`==`、`len()`、`str()` 等语言级操作。官方 Python 教程第 9 章把它们放在类这一章的最后，正是「对象的最终形态」。

理解 dunder，你就理解了 Python「一切皆对象」的深层机制：**运算符是语法糖，背后都是方法调用。**

## 1 dunder：运算符背后的钩子

**特殊方法（dunder）**：名字形如 `__xxx__`（double underscore）的方法，由 Python 在特定场景自动调用，而非你显式调用。

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"    # 给开发者的表示

p = Point(3, 4)
print(p)                     # Point(3, 4)，背后调用了 __repr__ 或 __str__
str(p)                       # 优先调 __str__，缺省用 __repr__
```

**重点：`print(obj)`、`str(obj)`、`repr(obj)` 背后都是方法调用。** `__str__` 面向用户（`print` 用它），`__repr__` 面向开发者（`repr()` 与 REPL 用它）。实现了它们，你的对象才能「打印得好看」——否则打印出来是一串晦涩的内存地址 `<__main__.Point object at 0x...>`。

**辨析｜易错点：** `__init__` 是**构造后的初始化**，真正「造对象」的是 `__new__`。日常几乎只写 `__init__`；`__new__` 是单例、不可变类型才需要碰的高级工具。<span class="marginnote">「dunder」取自 double underscore 的连读（dunder）。PEP 8 明确：不要发明自己的 `__名字__`——这种命名空间留给 Python 自身。标准库里 `__file__`、`__name__`、`__main__` 都是语言约定的元信息。</span>

## 2 运算符重载：让 +、==、< 有意义

**运算符重载（operator overloading）**：通过实现对应 dunder，让你的对象支持算术与比较运算。

```python
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __add__(self, other):                  # 支持 v1 + v2
        return Vector(self.x + other.x, self.y + other.y)

    def __eq__(self, other):                   # 支持 v1 == v2
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):                   # 支持 v1 < v2（排序用）
        return (self.x, self.y) < (other.x, other.y)

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)              # Vector(4, 6)
print(v1 == Vector(1, 2))   # True
```

**重点：`v1 + v2` 就是 `v1.__add__(v2)` 的语法糖。** 让自定义类支持算术，只需实现对应 dunder。常用映射：`+`→`__add__`，`==`→`__eq__`，`<`→`__lt__`，`len()`→`__len__`，`str()`→`__str__`。标准库里 `datetime` 支持「日期 + 天数」，靠的正是 `__add__`。<span class="marginnote">实现了 `__lt__`，`sorted([...])` 就能对这类对象排序——排序算法只需一个比较键。官方教程里多次用「定义 `__lt__` 后列表可排序」来说明「协议」思想：实现一个方法，换来整套语言能力。</span>

## 3 容器协议与上下文管理器

想让自定义类像列表一样支持下标、像字典一样支持成员判断，实现容器协议：

```python
class TodoList:
    def __init__(self):
        self.items = []

    def __len__(self):                 # len(todo) 可用
        return len(self.items)

    def __getitem__(self, i):          # todo[0] 可用
        return self.items[i]

    def __contains__(self, item):      # "task" in todo 可用
        return item in self.items
```

**上下文管理器（context manager）** 是另一个著名协议——`with` 语句能工作的秘密：

```python
class File:
    def __enter__(self):               # 进入 with 块时调用
        print("打开文件")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):   # 离开 with 块时调用
        print("关闭文件")

with File() as f:
    pass                              # 自动触发 __enter__ 与 __exit__
```

**重点：`with` 背后就是 `__enter__` + `__exit__`。** 这就是文件对象能 `with open(...) as f` 的原因，也是 `@contextmanager` 装饰器把普通函数变成上下文管理器的原理。<span class="marginnote">实现 `__iter__` 与 `__next__` 的对象还能被 `for` 循环遍历——这就是下一节《迭代器与生成器》的入口。容器协议、迭代协议、上下文协议，是 dunder 世界三大支柱。</span>

## 4 核心对比表：常用 dunder 一览

| 类别 | 方法 | 触发的语法 |
| --- | --- | --- |
| 表示 | `__str__`、`__repr__` | `str()`、`print()`、REPL |
| 算术 | `__add__`、`__sub__`、`__mul__` | `+`、`-`、`*` |
| 比较 | `__eq__`、`__lt__`、`__le__` | `==`、`<`、`<=` |
| 容器 | `__len__`、`__getitem__`、`__contains__` | `len()`、`obj[i]`、`in` |
| 迭代 | `__iter__`、`__next__` | `for` 循环 |
| 上下文 | `__enter__`、`__exit__` | `with` 语句 |

**核心观察：一个方法，换来整套语法。** dunder 的本质是「协议」——Python 提供语法骨架，你通过实现方法填入语义。这套设计让第三方对象与内建对象享受同一套语言能力，是「一切皆对象」的落地。学习 dunder，就是在学习语言为你预留的接口清单。

## 5 小结

- **dunder**（前后双下划线）方法由语言自动调用，`__init__` 只是第一个。
- `__str__` 面向用户、`__repr__` 面向开发者，实现后可让对象「打印得好看」。
- 运算符重载：`v1 + v2` 就是 `v1.__add__(v2)`，实现对应 dunder 即支持该运算符。
- 容器协议 `__len__`/`__getitem__`/`__contains__` 让对象支持 `len()`、下标、`in`。
- `with` 语句的秘密是 `__enter__` + `__exit__`；实现 dunder 即获得整套语法能力。

在下一节，我们将学习 Python 的惰性引擎——迭代器与生成器，看看 `for` 循环如何被 `yield` 重塑。
