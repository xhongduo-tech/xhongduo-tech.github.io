---
title: 函数：定义、参数与返回值
date: 2026-08-07
---

# 函数：定义、参数与返回值

<div class="epigraph">
<p>函数是代码的最小「积木」：命名一段逻辑，然后反复复用。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 《Python编程：从入门到实践》（第3版）第8章 ｜ 2026-08-07</p>
</div>

## 为什么从函数开始

到目前为止，代码都是「一行行平铺」：从文件顶部读到尾部。可一旦逻辑多起来，这样的代码就成了一锅粥——同样的判断写三遍、改一处要翻三处。**函数（function）** 正是拆解之道：把一段命名、参数化、可复用的逻辑抽出来，需要时调用。

这一节回答三个问题：函数怎么定义、参数有多少种形态、以及变量在函数内外如何「看得见」。函数也是后续模块、类、装饰器的地基。

## 1 定义与调用：从语句到表达式

**函数定义**用 `def` 关键字，**函数调用**用函数名加圆括号：

```python
def greet_user(username):
    """显示简单的问候语"""        # 文档字符串 docstring
    print(f"Hello, {username.title()}!")

greet_user("jesse")                # Hello, Jesse!
greet_user("sarah")                # Hello, Sarah!
```

函数体是缩进块；`def` 后第一行的 `"""..."""` 是**文档字符串（docstring）**，记录函数用途，可用 `help(greet_user)` 查看。**抽象是函数的第一价值**：调用者只需知道「输入什么、得到什么」，不必关心内部实现——这正是一级一级往上搭建筑的起点。

**返回值**用 `return` 语句，让函数的结果成为表达式的一部分：

```python
def get_formatted_name(first, last):
    full_name = f"{first} {last}".title()
    return full_name

name = get_formatted_name("ada", "lovelace")
print(name)                        # Ada Lovelace
```

**辨析｜易错点：** 没有 `return` 的函数返回 `None`。`result = print("hi")` 会得到 `None`——`print` 只负责输出，不返回内容。区分「函数做什么」（副作用）与「函数返回什么」（值），是理解函数的第一步。

## 2 参数形态：位置、关键字与默认值

函数可以接收零到多个参数，按三种形态提供：

```python
def describe_pet(pet_name, animal_type="dog"):   # animal_type 有默认值
    print(f"I have a {animal_type} named {pet_name}.")

describe_pet("willie")                    # 位置参数：省略 animal_type
describe_pet("harry", "hamster")          # 位置参数：按顺序对应
describe_pet(animal_type="cat", pet_name="mia")   # 关键字参数：乱序也行
```

**位置参数**按定义顺序对应；**关键字参数**用 `形参名=值` 指定，顺序无关；**默认值参数**在调用时可省略。三者可以混用，但位置参数必须排在关键字参数之前。

**重点：默认值参数应放在参数列表末尾。** 因为省略时按位置匹配，`def f(a=1, b)` 这种「无默认值的参数跟在有默认值的后面」是语法错误——Python 无法判断调用 `f(2)` 的 `2` 给谁。

## 3 任意数量参数：*args 与 **kwargs

有时参数数量不确定：汇总一份成绩单、传递任意配置。用星号收集：

```python
def make_pizza(size, *toppings):          # *toppings 收集成元组
    print(f"制作 {size} 英寸披萨，加料：{toppings}")

make_pizza(12, "mushrooms", "peppers")    # toppings = ('mushrooms', 'peppers')
```

```python
def build_profile(first, last, **user_info):   # **user_info 收集成字典
    info = {"first": first, "last": last}
    info.update(user_info)
    return info

profile = build_profile("albert", "einstein",
                        location="princeton", field="physics")
```

**重点：`*` 收集多余的位置参数成元组，`**` 收集多余的关键字参数成字典。** 这层「打包」机制在函数调用时还有镜像的**解包**用法：`f(*args)` 把列表展开成多个位置参数，`f(**kwargs)` 把字典展开成关键字参数。<span class="marginnote">`*args` 与 `**kwargs` 的名字约定俗成（args = arguments，kwargs = keyword arguments），但星号才是真正的语法；换成 `*items`、`**opts` 同样有效。在《模块与包》一节，导入语句里的 `from mod import *` 也是同一个星号家族。</span>

## 4 作用域：变量在哪里「看得见」

函数内的变量与函数外的变量分属不同的**作用域（scope）**。

```python
x = "global"          # 全局作用域
def f():
    y = "local"       # 局部作用域
    print(x)          # 可以读取全局变量
    print(y)          # 局部变量，只在本函数内可见

f()
# print(y)            # NameError：函数外看不到 y
```

**重点：读取是自由的，写入是要声明的。** 在函数内对 `x` 赋值，Python 会认为你想创建一个新的局部变量 `x`，而不是修改全局 `x`——除非用 `global x` 声明。多数情况下应避免在函数内修改全局变量，改为「函数返回值、调用处重新赋值」，这让数据流向清晰可追踪。<span class="marginnote">「读写不对称」是作用域最常见的坑：`x = x + 1` 在函数内若 `x` 未定义，会抛 `UnboundLocalError`——因为赋值让 `x` 变成了局部变量，但右边的读取找不到局部 `x`。理解「赋值即声明局部」这一条，就能避开它。</span>

## 5 公式解析：递归与阶乘

函数能调用自己，这叫**递归（recursion）**。最经典的例子是阶乘：

$$
n! = \begin{cases} 1 & n = 0 \\ n \cdot (n-1)! & n \ge 1 \end{cases}
$$

写成函数：

```python
def factorial(n):
    if n == 0:              # 递归出口（base case）
        return 1
    return n * factorial(n - 1)   # 递归调用
```

对这条式子做三步拆解：

- **第一步，找出口**：`n == 0` 时直接返回 1，不再调用自身。**没有出口的递归就是死循环的孪生兄弟**——会无限压栈直到 `RecursionError`。
- **第二步，看递推**：`factorial(5)` 展开为 `5 * factorial(4)`，后者又展开为 `4 * factorial(3)`……直到触底 `factorial(0) = 1`。
- **第三步，理解成本**：`factorial(5)` 展开为 `5 * (4 * (3 * (2 * (1 * 1))))`，共 5 层嵌套。每一次递归调用都占用**调用栈**的一层，层数过多会栈溢出——所以迭代（`while`/`for`）通常比递归省内存。数学里的递推定义与编程里的递归一一对应，这是你从第一级《数列》里的递推公式迁移过来的同一思想。

## 6 小结

- 函数用 `def` 定义，`return` 返回值，`"""docstring"""` 记录用途。
- 三种参数形态：**位置**、**关键字**、**默认值**；默认值参数排在末尾。
- `*args` 收集成元组，`**kwargs` 收集成字典；调用时用 `*`/`**` 解包。
- 函数内赋值即声明**局部变量**；读取全局可以，写入需 `global` 声明。
- 递归 = 出口 + 递推；数学递推公式与递归函数一一对应，但迭代更省栈。

在下一节，我们将把函数装进更大的单元——模块与包，学习如何组织多文件的大型程序。
