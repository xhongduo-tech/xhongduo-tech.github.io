---
title: 装饰器（Decorator）与语法糖
date: 2026-08-07
---

# 装饰器（Decorator）与语法糖

<div class="epigraph">
<p>不修改函数的代码，却改变函数的行为——这就是装饰器的魔法。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 官方 Python 教程 第9章 ｜ 2026-08-07</p>
</div>

## 为什么从装饰器开始

上一节的闭包留了一个悬念：内层函数记住外层环境，那「用一个函数包住另一个函数」能做什么？答案是——**装饰器（decorator）**：在**不修改函数源码**的前提下，给函数附加行为（计时、日志、权限检查）。官方 Python 教程第 9 章里的 `@classmethod`、`@staticmethod` 就是装饰器的内建实例。

本节拆解三件事：`@` 语法到底是什么、如何自己写一个装饰器、以及如何让装饰器保存被装饰函数的名字。

## 1 语法糖的真相：@decorator = f = dec(f)

装饰器的底层，是上一节的闭包。先看「纯函数版本」——手写一个包装函数：

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("调用前：准备中...")
        result = func(*args, **kwargs)
        print("调用后：收尾。")
        return result
    return wrapper

def say_hi():
    print("你好！")

say_hi = my_decorator(say_hi)   # 手动套上装饰器
say_hi()
```

`@` 语法只是把最后一步「赋值回去」变成一行：

```python
@my_decorator
def say_hi():
    print("你好！")
```

**重点：`@my_decorator` 等价于 `say_hi = my_decorator(say_hi)`。** `my_decorator` 接收原函数、返回一个 `wrapper`，名字 `say_hi` 现在指向 `wrapper`——此后每次调用 `say_hi()`，真正执行的是 `wrapper`，它会在调用前后做额外动作。<span class="marginnote">`wrapper(*args, **kwargs)` 用 `*`/`**` 把任意参数原样转发给原函数（见《函数：定义、参数与返回值》的打包/解包），这样装饰器能套在签名各异的函数上而互不干扰。</span>

## 2 动手写一个计时装饰器

最常见的装饰器用途是**计时**与**日志**——不污染业务代码，把横切关注点抽出来：

```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} 耗时 {elapsed:.4f} 秒")
        return result
    return wrapper

@timer
def slow_task(n):
    total = 0
    for i in range(n):
        total += i
    return total

slow_task(1_000_000)          # 打印：slow_task 耗时 0.0xx 秒
```

**重点：装饰器把「横切关注点」从业务逻辑里剥离。** 计时、日志、鉴权、缓存这些「每个函数都可能要」的行为，不必在每个函数里各写一遍——写进装饰器，`@` 一行即可复用。这正是「DRY（Don't Repeat Yourself）」原则的工程落地。

**辨析｜易错点：** 装饰后，`slow_task` 的名字与文档会变成 `wrapper` 的——`help(slow_task)` 不再显示原函数信息。修复方法是用 `functools.wraps`：

```python
from functools import wraps

def timer(func):
    @wraps(func)                      # 复制 func 的名字与文档到 wrapper
    def wrapper(*args, **kwargs):
        ...
```

`@wraps` 本身就是一个装饰器，它把原函数的 `__name__`、`__doc__` 拷贝到 `wrapper` 上，让「被装饰后」的函数在工具看来仍像原函数。<span class="marginnote">`functools.wraps` 的原理正是把 `func.__name__`、`func.__doc__` 赋值给 `wrapper`，并更新 `wrapper.__wrapped__` 指向原函数。写装饰器时<strong>始终加 @wraps(func)</strong>，是社区约定，也是调试与文档的基本保障。</span>

## 3 带参数的装饰器与堆叠

装饰器还可以「套参数」——但那时 `@` 后面是一个**装饰器工厂**：

```python
def repeat(times):                    # 返回真正的装饰器
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet():
    print("你好！")

greet()                               # 你好！× 3
```

**重点：`@repeat(3)` 先调用 `repeat(3)` 得到 `decorator`，再 `greet = decorator(greet)`。** 三层嵌套（工厂 → 装饰器 → wrapper）是带参数装饰器的固定形状。

**堆叠**：多个 `@` 从下往上依次应用：

```python
@timer            # 外层：最晚应用
@my_decorator     # 内层：最先应用
def f():
    pass
```

**辨析｜易错点：** 堆叠时顺序很重要——`f = timer(my_decorator(f))`，下面先套、上面后套。习惯上「功能性的、最基础的」放下面，「关心开销的」放上面，读起来才顺。

## 4 公式解析：装饰器的等价展开

**`@dec` 装饰的本质是一条「函数替换」恒等式。**

对任意函数 $f$ 与装饰器 $\mathrm{dec}$，以下两种写法完全等价：

$$
f' = \mathrm{dec}(f) \quad \Longleftrightarrow \quad \texttt{@dec} \; \texttt{def } f'(\cdots)
$$

对这条式子做三步拆解：

- **第一步，读方向**：`@dec` 的意思是「定义完成后，把函数对象交给 `dec` 处理，把结果重新绑定到原来的名字」。它不是「给函数打标签」，而是「换掉函数」。
- **第二步，看对象**：`dec` 接收一个函数（对象），返回一个函数（对象）——签名是「函数 → 函数」。任何「接收并返回函数」的调用，都可以被 `@` 改写，无论它叫 `dec`、`timer` 还是 `functools.wraps`。
- **第三步，算复杂度**：装饰只发生**一次**（定义时），之后每次调用走 `wrapper`，多一层调用栈、少许额外开销。所以「装饰器是定义期的投资、调用期的微小税负」——用它简化代码结构，完全值得。

**为何要理解展开？** 因为它解释了三个现象：`@` 只是赋值语法糖（真相在闭包）；`functools.wraps` 为何必要（否则 `f.__name__` 丢失）；以及「带参数的装饰器为何三层嵌套」（`@dec(参数)` 先求值工厂再应用）。剥掉 `@`，一切回到函数。

## 5 小结

- `@decorator` 等价于 `f = decorator(f)`：接收函数、返回增强版函数。
- 装饰器把**横切关注点**（计时、日志、鉴权）从业务中剥离，实现 DRY。
- `wrapper(*args, **kwargs)` 原样转发参数，让装饰器适用任意签名。
- 用 `functools.wraps` 保留原函数的名字与文档，是写装饰器的铁律。
- 带参数的装饰器是「工厂 → 装饰器 → wrapper」三层结构；堆叠时自下而上应用。

在下一节，我们将把散落的字符串操作收拢——字符串处理与格式化，让文本既读得进也排得美。
