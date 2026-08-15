---
title: 继承、多态与鸭子类型
date: 2026-08-07
---

# 继承、多态与鸭子类型

<div class="epigraph">
<p>如果它走起来像鸭子、叫起来像鸭子，那它就是鸭子。</p>
<footer>—— 谚语（James Whitcomb Riley 的「鸭子测试」）</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 官方 Python 教程 第9章 ｜ 2026-08-07</p>
</div>

## 为什么从继承开始

上一节我们把「数据 + 行为」封进了类。但现实中，类与类之间常有**层级**：一只柯基是狗，狗又是动物。「柯基」不该重复定义「狗」已有的属性和方法。**继承（inheritance）** 让子类自动获得父类的成员，并在需要时改写——这是官方 Python 教程第 9 章的进阶主题。

本节还回答一个更本质的问题：**多态（polymorphism）** 是什么，以及 Python 独特的**鸭子类型（duck typing）** 如何让「类型检查」这件事变得轻松而危险。

## 1 继承：子类复用父类

**继承**：定义子类时把父类名放进括号，子类自动拥有父类的全部属性与方法。

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print(f"{self.name} 发出了声音。")

class Dog(Animal):                # Dog 继承自 Animal
    def fetch(self):
        print(f"{self.name} 跑去捡球了！")

d = Dog("Willie")
d.speak()                        # Willie 发出了声音（来自父类）
d.fetch()                        # Willie 跑去捡球了！（子类独有）
```

**重点：子类「是一个」父类。** 柯基是狗、狗是动物，所以柯基能调用动物的方法。子类可以**新增**方法（`fetch`），也可以**重写（override）** 父类方法。

若子类有自己的 `__init__`，要显式调用父类的初始化：

```python
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)        # 调用父类 __init__
        self.breed = breed
```

`super()` 返回父类代理，`super().__init__(...)` 把父类该初始化的部分初始化好。**省略 `super().__init__()` 是最常见的继承错误**——父类属性没初始化，子类就访问不到。

## 2 多态：同一接口，不同实现

**多态（polymorphism）**：同一个方法名，在不同类上有不同的实现；调用者不必关心对象的「具体类型」，只要它「会做这件事」。

```python
class Dog(Animal):
    def speak(self):
        print(f"{self.name} 汪汪叫！")

class Cat(Animal):
    def speak(self):
        print(f"{self.name} 喵喵叫！")

def make_speak(animal):          # 一个函数，处理任何「会 speak」的对象
    animal.speak()

make_speak(Dog("Willie"))        # Willie 汪汪叫！
make_speak(Cat("Whiskers"))      # Whiskers 喵喵叫！
```

**重点：`make_speak` 不关心参数是 `Dog` 还是 `Cat`，只要求它有 `speak` 方法。** 这就是多态的价值——调用方与具体类型解耦。要新增一种动物，只需写它的类，`make_speak` 一行不改。<span class="marginnote">多态在数学/工程里处处可见：`print(3)` 打印数字、`print("hi")` 打印字符串、`print([1,2])` 打印列表——同一个 `print`，对不同类型的参数做了不同的事，靠的是每个类型自己实现的 `__str__`。这就是「统一接口、分而实现」。</span>

## 3 鸭子类型：看行为，不看血缘

Python 对多态的实现更宽松——**不要求继承关系**，只看「是否具备某行为」：

```python
class Person:
    def speak(self):
        print("人在说话。")

make_speak(Person())             # 人在说话。Person 与 Animal 毫无血缘
```

**鸭子类型（duck typing）**：只要对象「走起来像鸭子、叫起来像鸭子」，就把它当鸭子用——判断标准是**行为**而非**类型**。`Person` 没继承 `Animal`，但它有 `speak` 方法，就能传给 `make_speak`。

**重点：鸭子类型把「检查」从调用前挪到了调用时。** 好处是代码极其灵活、耦合低；代价是**错误滞后**——传错对象时，只有真正调用 `.speak()` 的那一行才会报 `AttributeError`。应对之道是「三思而行」：用 `isinstance()` 做显式检查，或用更 Pythonic 的 **EAFP**（try/except）处理可能的失败。<span class="marginnote">`isinstance(x, Animal)` 检查血缘，`hasattr(x, "speak")` 检查行为。第三级《异常处理与调试技巧》会讲 EAFP（先试、再捕获）与 LBYL（先看、再动）之争——鸭子类型的灵活性正是在那里获得纪律。</span>

**多态与鸭子类型在标准库中俯拾皆是**：`len()` 接受任何实现 `__len__` 的对象，`sorted()` 接受任何可迭代对象，`open()` 接受任何「文件类」对象——它们都不检查类型，只检查行为。写通用函数时，继承是「显式的契约」，鸭子类型是「隐式的约定」：前者靠类型约束，后者靠测试兜底。

**辨析｜易错点：** 鸭子类型不是「不检查」的借口。当函数对参数有明确要求（如必须实现 `speak`）时，docstring 应写明；必要时用 `hasattr` 先探测、或 `try/except` 捕获 `AttributeError` 给出友好提示。灵活与纪律并存，才是生产级代码——这与《异常处理与调试技巧》一节的 EAFP 哲学正好衔接。

## 4 核心对比表：继承 vs 组合

| 维度 | 继承 | 组合 |
| --- | --- | --- |
| 关系 | 「是一个」（is-a） | 「有一个」（has-a） |
| 实现 | 子类继承父类 | 对象持有另一个对象 |
| 耦合 | 强：子类依赖父类细节 | 弱：通过接口协作 |
| 扩展 | 加子类 | 换组件 |
| 风险 | 层级过深、脆弱的基类 | 需显式委托 |

**核心观察：优先组合，而非继承。** 这是面向对象工程的一条经典经验。继承建立的是静态的「血缘」，改父类会波及所有子类；组合建立的是「零件组装」，换零件不影响整体。`collections.deque`、`random.Random` 这类标准库对象，大量用组合而非继承来获得能力。<span class="marginnote">「组合优于继承」不是禁止继承——它来自《设计模式》里「Favor object composition over class inheritance」的告诫。判断口诀：如果子类只是「借用」父类方法，组合通常更干净；如果确实是「天然的血缘」，继承无可厚非。</span>

## 5 小结

- 子类加父类名于括号即**继承**，自动获得父类成员，可新增、可重写。
- 子类自带 `__init__` 时用 `super().__init__(...)` 初始化父类部分，勿省略。
- **多态**：同一方法名不同实现，调用方只认「会做这件事」。
- **鸭子类型**：看行为不看血缘，灵活但错误会滞后到调用时。
- 设计上**优先组合、慎用继承**，用 `isinstance`/`hasattr` 显式约束行为。
- 鸭子类型的边界：行为检查不取代文档——公共接口写明约定，用异常或探测给出友好失败。

在下一节，我们将深入 Python 对象的底层约定——dunder 特殊方法，看看 `+`、`==`、`len()` 背后发生了什么。
