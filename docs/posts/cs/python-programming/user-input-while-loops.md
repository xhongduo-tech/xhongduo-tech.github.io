---
title: 用户输入与 while 循环
date: 2026-08-07
---

# 用户输入与 while 循环

<div class="epigraph">
<p>程序真正的生命力，从接收用户的第一句话开始。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 《Python编程：从入门到实践》（第3版）第7章 ｜ 2026-08-07</p>
</div>

## 为什么从用户输入开始

前几节的程序都是「闭门造车」：数据写死在代码里，跑一遍就结束。但真实的程序要**与用户对话**——让用户输入姓名、选择菜单、回答「是否继续」。这一节学习两个配合紧密的工具：**`input()`** 让程序停下来听用户说话，**`while` 循环**让程序反复做一件事直到满足条件。

两者结合，就能写出真正的「交互式」小程序：积木问答、用户注册、猜数字——这也是《Python编程》第 7 章的经典实践。

## 1 input()：让程序停下来听用户说话

**`input(prompt)`**：打印提示信息，等待用户在终端输入一行，回车后把输入**以字符串形式**返回。

```python
name = input("请输入你的名字：")
print(f"你好，{name}！")
```

**重点：`input()` 永远返回字符串。** 哪怕用户输入的是 `42`，得到的也是 `"42"` 而非整数 `42`。要当数字用，必须先转换：

```python
age = int(input("请输入你的年龄："))   # 字符串 → 整数
height = float(input("请输入身高（米）："))
```

**辨析｜易错点：** 如果用户输入了无法解析的内容，`int("abc")` 会抛 `ValueError` 导致程序崩溃。稳健做法是先读取字符串，再在 `try/except` 中转换——异常处理会在本专题《异常处理与调试技巧》一节系统讲解，这里先记住「先读后转」的原则。

`input()` 还有一个用途：让程序**暂停**。交互式脚本常在结尾加一行 `input("按回车退出...")`，防止终端窗口一闪而过。

## 2 while 循环：条件成立就重复

**`while` 循环**：只要条件为真，就反复执行循环体。与 `for`（遍历已知序列）不同，`while` 适合「不知道要循环几次、直到条件改变」的场景。

```python
current = 1
while current <= 5:
    print(current)
    current += 1
```

最常见的模式是**标志（flag）**：用一个布尔变量控制循环是否继续。

```python
active = True
while active:
    message = input("输入 'quit' 退出：")
    if message == "quit":
        active = False        # 翻转标志，循环即将结束
    else:
        print(f"你说的是：{message}")
```

**重点：先判断、后改变。** `while` 的核心风险是**死循环**——条件永远为真。所以循环体内必须有一条路径最终让条件变为假（改变 `current`、翻转 `active`），这是编程里最早、也是最严肃的教训。<span class="marginnote">死循环不一定都是坏事——服务器、操作系统事件循环本质上是「故意死循环」。区别在于：恶意死循环不做事还占满 CPU，良性事件循环会主动等待事件。在《操作系统》专题里，「忙等」与「阻塞等待」是两条不同的哲学。</span>

## 3 循环控制：break 与 continue

`break` 立即结束整个循环，`continue` 跳过本次迭代的剩余语句：

```python
while True:                        # 无限循环 + break 退出，常见交互模式
    city = input("输入城市（'quit' 退出）：")
    if city == "quit":
        break                      # 跳出循环
    print(f"我想去 {city.title()}！")
```

```python
number = 0
while number < 10:
    number += 1
    if number % 2 == 0:
        continue                   # 偶数直接跳过打印
    print(number)                  # 打印 1 3 5 7 9
```

**辨析｜易错点：** `break` 只跳出**最内层**循环；`continue` 跳过后要确保改变条件变量的语句仍然执行——把 `number += 1` 写在 `continue` 之后，会导致死循环。这两个控制语句与 `for` 循环完全通用，我们已在《深入流程控制》一节见过它们的完整语义。

## 4 while 循环处理列表与字典

`for` 循环遍历列表时不应修改列表；而 `while` 循环可以边遍历边删——这正是处理「待办队列」的姿势。

```python
unconfirmed = ["alice", "brian", "candace"]
confirmed = []
while unconfirmed:
    user = unconfirmed.pop()        # 从队尾取出
    confirmed.append(user)
print(confirmed)                    # ['candace', 'brian', 'alice']
```

**重点：`while` + 容器 = 队列。** 用 `while 列表:`（非空即真）不断 `pop()`，可模拟先入先出或后入先出；配合 `append`，就是最简单的**队列/栈**。这个模式在后续《数据结构》专题的队列章节、以及消息处理脚本里会反复出现。<span class="marginnote">`while 列表:` 利用了「空列表为假、非空为真」的真值规则——Python 里空容器（`[]`、`{}`、`""`、`0`、`None`）一律为假，这是非常 Pythonic 的判断习惯，下一节《if 语句与条件逻辑》会专门讲真值测试。</span>

## 5 核心对比表：for 与 while

| 维度 | `for` 循环 | `while` 循环 |
| --- | --- | --- |
| 适用场景 | 遍历已知序列 | 直到条件改变 |
| 循环次数 | 由序列长度决定 | 由条件决定，可能无限 |
| 常见风险 | 修改正在遍历的列表 | 忘记更新条件导致死循环 |
| 退出方式 | 自然结束 / `break` | 条件变假 / `break` |
| 典型搭档 | `range`、`enumerate` | 标志 `active`、`input()` |

**核心观察：选循环不靠喜好，靠「是否知道次数」。** 知道要遍历 N 个元素，用 `for`；不知道要循环多少次、要由运行时状态决定，用 `while`。两者共享 `break`、`continue`、`else` 三个控制语句，语义完全一致。

## 6 小结

- `input()` 返回**字符串**，先读取、再 `int()`/`float()` 转换。
- `while` 循环「条件为真就重复」，用**标志** `active` 控制最常见。
- 循环体内必须有一条路径让条件最终变假，否则陷入**死循环**。
- `break` 立即退出，`continue` 跳过本次迭代；`break` 只作用于最内层。
- `while 列表:` + `pop()` + `append()` 可模拟队列/栈处理待办。

在下一节，我们将系统学习判断的语法核心——`if` 语句与条件逻辑，让程序学会「看情况办事」。
