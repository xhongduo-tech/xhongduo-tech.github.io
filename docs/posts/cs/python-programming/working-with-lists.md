---
title: 操作列表：遍历、切片与列表推导
date: 2026-08-07
---

# 操作列表：遍历、切片与列表推导

<div class="epigraph">
<p>让数据动起来，而不是一个元素一个元素地搬。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 《Python编程：从入门到实践》（第3版）第4章 ｜ 2026-08-07</p>
</div>

## 为什么从整批操作开始

上一节我们学会构造列表、按下标取元素。但真实任务很少只处理一个元素——要对**整批数据**做同一件事：把每个分数乘以权重、把每封邮件标记已读、把名单排序。这一节就是「整批操作」的专场。

核心武器有三个：`for` 循环（逐个访问）、切片（复制与取段）、以及 Python 引以为傲的**列表推导式（list comprehension）**——一行写出一个完整的新列表。

## 1 用 for 循环遍历列表

**for 循环**：把列表中的每个元素依次取出来、执行一遍循环体。它是「遍历」的最基本形态。

```python
magicians = ["alice", "david", "carolina"]
for magician in magicians:
    print(f"{magician.title()}, 该你上场了！")
```

每次迭代，变量 `magician` 被绑定到列表中的下一个元素。循环体是缩进块——**缩进即层级**，这是 Python 语法里最容易也最关键的习惯。

需要下标时，用 `range(len(...))` 配合，或用更 Pythonic 的 `enumerate`：

```python
for i, magician in enumerate(magicians, start=1):
    print(f"第 {i} 位：{magician}")
```

**重点：`for` 读作「对每一个」。** 它隐藏了下标管理，让你聚焦于「对每个元素做什么」，这正是把命令式思维转换为声明式思维的起点。

**辨析｜易错点：** 遍历列表时**不要**同时增删元素——`for x in lst` 的迭代进度依赖长度，中途改动会让迭代「跳项」或越界。需要边遍历边删时，改用 `while lst:` 的「队列模式」（见《用户输入与 while 循环》），或先构建新列表再替换。

## 2 数值列表：range、min、max 与 sum

`range()` 生成一个整数序列，配合 `list()` 可转成列表：

```python
squares_1 = []
for n in range(1, 6):
    squares_1.append(n ** 2)
print(squares_1)          # [1, 4, 9, 16, 25]
```

对数值列表还有三个内置助手：

```python
scores = [83, 91, 76, 88, 95]
print(min(scores))        # 76
print(max(scores))        # 95
print(sum(scores))        # 433
```

`min`、`max`、`sum` 是统计类任务的高频入口——第一级《概率与统计》里「样本均值 = 总和 ÷ 个数」，在 Python 里就是 `sum(scores) / len(scores)`。

`min` 与 `max` 同样支持 `key` 参数：`max(pairs, key=lambda p: p[1])` 返回「按第二项最大」的元素——与 `sorted` 的 `key` 机制同源，排序与取极值共享同一套「按什么比」的约定。<span class="marginnote">`range(1, 6)` 同样遵循半开区间：到 6 <strong>之前</strong>结束，正好生成 1 到 5。这与上一节切片 $[start, stop)$ 的语义完全同构——Python 把「半开区间」贯彻到了所有序列操作。</span>

## 3 排序与切片：sorted 与副本复制

**排序**有两种：`list.sort()` 原地修改，`sorted(list)` 返回新列表。

```python
cars = ["bmw", "audi", "toyota", "subaru"]
print(sorted(cars))        # ['audi', 'bmw', 'subaru', 'toyota']，原列表不变
cars.sort()                # 原地排序，直接修改 cars
print(cars)
```

**辨析｜易错点：** `sorted()` 默认返回**新列表**，而 `sort()` 返回 `None` 并**原地修改**。用 `cars = cars.sort()` 会把列表变成 `None`——这是初学高频事故，牢记「sort 原地、sorted 出新」。

**排序还能「按键」。** `sorted(cars, key=len)` 按字符串长度排，`sorted(pairs, key=lambda p: p[1])` 按元组第二项排——`key` 参数让「按什么比」完全自定义。这个机制在《函数式特性：lambda、闭包与高阶函数》一节会配合 `lambda` 系统展开，这里先记住它的存在。

切片做复制的用法在前一节已见：`players[:]` 得到副本；`players[0:3]` 取前 3 名。这两者结合 `sorted`，可以做出「保持原名单、输出排序结果」的常见需求。

## 4 公式解析：列表推导式的语法展开

**列表推导式**：用一条表达式生成整个列表的紧凑写法。它本质上是一个**集合论构造**在编程里的翻译——数学里写

$$
\{n^2 \mid n \in \{1,2,3,4,5\}\}
$$

Python 写作：

```python
squares = [n ** 2 for n in range(1, 6)]
```

对这条式子做三步拆解：

- **第一步，读顺序**：把式子读成「对 `range(1, 6)` 里的每一个 `n`，产生一个 `n ** 2`」。顺序是**从后往前**：先看 `for`，再看最前面的表达式。
- **第二步，加过滤**：可在末尾追加 `if` 条件——`[n ** 2 for n in range(1, 10) if n % 2 == 0]` 生成 `[4, 16, 36, 64]`，对应数学里的 $\{n^2 \mid n \in S,\ n\ \text{为偶数}\}$。
- **第三步，与循环对照**：`[expr for item in seq if cond]` 完全等价于「`for` 循环 + `append` + `if` 判断」的三行代码。推导式只是把这三行折叠成一行——这正是《Python编程》第 4 章把推导式放在循环之后的用意：先懂展开，再懂缩写。

推导式能涵盖九成「从旧列表造新列表」的场景，比手工 `for + append` 更短、更不易出错，也是后续数据清洗、特征工程里最常用的写法。

**推导式的边界**：当变换逻辑超过「一个表达式」时——需要 `if/else` 双分支、需要多行计算——推导式就会变得晦涩，此时回到 `for + append` 反而更清晰。可读性始终是第一优先级：推导式省的是行数，不是理解成本。

## 5 小结

- `for x in lst` 逐个遍历，`enumerate(lst, start=1)` 同时拿到下标与元素。
- `range()` 生成整数序列；`min`、`max`、`sum` 是数值列表的三件套。
- `sort()` 原地排序、返回 `None`；`sorted()` 返回新列表。
- 切片 `lst[:]` 复制列表，避免别名共享。
- 列表推导式 `[expr for item in seq if cond]` 是「循环 + 判断 + 生成」的一行化，读法从 `for` 往前推。
- 遍历时勿修改列表；需要边遍历边删时用 `while` 队列模式或先建新列表。

在下一节，我们将转向另一种容器——字典，用「键值对」描述「名字→值」的映射关系。
