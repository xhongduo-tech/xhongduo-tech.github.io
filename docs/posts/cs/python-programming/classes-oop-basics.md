---
title: 类与面向对象编程基础
date: 2026-08-07
---

# 类与面向对象编程基础

<div class="epigraph">
<p>把数据和行为装进同一个名字，这就是「对象」。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 《Python编程：从入门到实践》（第3版）第9章 ｜ 2026-08-07</p>
</div>

## 为什么从类开始

前面我们一直在用「数据」与「函数」两种分离的构件：列表装着数据，函数处理数据。可现实模型往往要求两者同行——一只狗既有名字、年龄（数据），又会吠叫、奔跑（行为）。**类（class）** 把「一类东西共有的数据与行为」封装成蓝图，再通过**对象（object）** 造出一个个具体实例。

这一节是《Python编程》第 9 章的核心，也是从「过程式」转向「面向对象」的第一步。学完本节，你会看懂 `random`、`datetime`、`requests` 等大量库「为什么能 `对象.方法()`」地调用。

## 1 类与实例：蓝图与成品

**类（class）**：一类对象的模板，定义它们的共同属性与行为。**对象 / 实例（instance）**：按类造出的具体个体。

```python
class Dog:
    """一次模拟小狗的尝试"""

    def __init__(self, name, age):
        """初始化属性 name 和 age"""
        self.name = name
        self.age = age

    def sit(self):
        print(f"{self.name} 现在坐下了。")

    def roll_over(self):
        print(f"{self.name} 打滚！")
```

**重点：`__init__` 是构造器，`self` 是实例本身。** 创建实例时 `__init__` 自动被调用，`self` 指向正在创建的那个对象，`self.name = name` 把数据挂到实例上。

```python
my_dog = Dog("Willie", 6)          # 自动调用 __init__
print(my_dog.name)                  # Willie，访问属性
my_dog.sit()                        # 调用方法
```

**方法（method）** 就是类内定义的函数；**属性（attribute）** 就是实例上保存的数据。`Dog` 是**类**，`my_dog` 是**实例**——类定义了「所有狗都是这样」，实例是「这条具体的狗」。<span class="marginnote">`__init__` 里外各有两个下划线，这种「前后双下划线」的名字叫 <strong>dunder</strong>（double underscore）。Python 用它们标记「有特殊含义」的成员，下一节《特殊方法与运算符重载》会专门展开。首字母大写的 `Dog` 是类名约定，小写 `my_dog` 是实例名约定。</span>

## 2 三种方法：实例、类与静态

类里可以定义三种方法，区别在于「第一个参数是谁」：

```python
class Pizza:
    def __init__(self, size):
        self.size = size

    def area(self):                  # 实例方法：第一个参数 self
        return 3.14 * (self.size / 2) ** 2

    @classmethod
    def from_dict(cls, data):        # 类方法：第一个参数 cls
        return cls(size=data["size"])

    @staticmethod
    def validate(size):              # 静态方法：无 self 也无 cls
        return size > 0
```

**实例方法**接收实例，最常用；**类方法**接收类，常用于「替代构造器」（从别的格式造实例）；**静态方法**既不收实例也不收类，只是逻辑上归属该类，用作工具函数。

**如何选择三者？** 判断口诀：方法需要**实例数据**（`self.xxx`）→ 实例方法；只需**类数据**或要「按类造实例」→ 类方法；与类本身无关的纯工具函数 → 静态方法。多数方法都是实例方法，后两种是特例，用对能显著提升代码意图的清晰度——读者一眼就知道「这个方法依赖什么」。<span class="marginnote">`@classmethod`、`@staticmethod` 前面的 `@` 是<strong>装饰器</strong>——一种「给函数附加行为」的语法。它在《装饰器（Decorator）与语法糖》一节会被彻底拆解，这里只需知道「`@` 标记了方法的种类」。</span>

## 3 属性的可见性：公开、受保护与「魔术」

Python 没有严格的私有属性，但有**约定**与**改名机制**两层防护：

```python
class Account:
    def __init__(self):
        self.balance = 0          # 公开：外部可自由读写
        self._pending = 0         # 约定私有：下划线开头，勿直接访问
        self.__secret = "hidden"  # 名称改写：__x 变为 _ClassName__x
```

- `balance`：公开，谁都能 `acc.balance`。
- `_pending`：单下划线只是**约定**——告诉别的程序员「这是内部实现，别碰」，但技术上仍可访问。
- `__secret`：双下划线触发**名称改写（name mangling）**，类外访问 `acc.__secret` 会报错。

**重点：Python 的封装靠「君子协定」。** 单下划线靠自觉，双下划线靠改名。真正的约束力来自你与同事的共识：公开接口用 `property` 提供读写控制，内部状态用下划线标记。这与 Java 的 `private` 完全不同——Python 信任使用者的纪律。<span class="marginnote">类属性与实例属性的差别也在这里浮现：定义在 `class` 体内、方法外的变量是<strong>类属性</strong>，所有实例共享；`__init__` 里 `self.x = ...` 是<strong>实例属性</strong>，各实例独立。二者同名时实例属性优先——「先实例、后类」的查找顺序。</span>

要给属性加**读写规则**，用 `@property` 装饰器——把「方法」伪装成「属性」：

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def area(self):
        return 3.14159 * self._radius ** 2

c = Circle(2)
print(c.area)              # 12.56636，像属性一样读，实为方法计算
```

`c.area` 不写括号却返回计算结果，把「访问的简洁」与「计算的严格」结合。`@property` 的机制在《装饰器（Decorator）与语法糖》一节会彻底拆解，这里先认识它的用途：当属性需要「算出来的」或「校验过的」值时，它就是正解。

## 4 核心对比表：面向过程 vs 面向对象

| 维度 | 面向过程 | 面向对象 |
| --- | --- | --- |
| 数据与行为 | 分离：函数处理数据 | 封装：对象同时拥有 |
| 组织单位 | 函数、模块 | 类、对象 |
| 复用方式 | 函数调用 | 继承、组合 |
| 心智模型 | 「做这件事的步骤」 | 「这一类东西是什么样」 |
| 典型场景 | 脚本、算法 | 大型系统、GUI、游戏 |

**核心观察：OOP 是「为变化而生」。** 面向过程适合「逻辑固定、写一次就行」的脚本；面向对象适合「需求多变、要持续扩展」的工程。Python 两者都支持，选择标准是**代码是否在变**——会变的地方用对象封装，不变的地方用函数即可。这是工程判断，不是教条。

## 5 小结

- **类**是蓝图，**实例**是成品；`__init__` 构造实例，`self` 指代实例本身。
- 属性是实例上的数据，方法是类内定义的函数；`对象.属性`、`对象.方法()` 访问。
- 三种方法：**实例方法**（`self`）、**类方法**（`cls`，替代构造器）、**静态方法**（工具函数）。
- Python 无严格私有：单下划线是约定，双下划线触发名称改写。
- 面向过程 vs 面向对象的选择，取决于**代码是否在变**。

在下一节，我们将学习类的「遗传」——继承、多态与鸭子类型，让子类复用并改造父类的行为。
