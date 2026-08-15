---
title: if 语句与条件逻辑
date: 2026-08-07
---

# if 语句与条件逻辑

<div class="epigraph">
<p>程序的分岔路，由条件来决定。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 《Python编程：从入门到实践》（第3版）第5章 ｜ 2026-08-07</p>
</div>

## 为什么从条件逻辑开始

顺序、分支、循环是结构化的三大基石。前面我们已经掌握顺序（一句句执行）与循环（重复执行），这一节补上第三块——**分支**：根据条件选择执行不同的代码。`if` 语句是分支的全部语法，而它背后依赖两个更基本的概念：**布尔值**与**比较运算**。

本节从「真与假」讲起，再到 `if/elif/else` 的三种形态，最后落到 `and/or/not` 的逻辑组合与三目表达式。

## 1 布尔值与比较运算

**布尔（bool）** 只有两个值：`True` 与 `False`。比较运算返回布尔值：

```python
print(5 > 3)      # True
print(5 == 3)     # False，== 是相等判断
print(5 != 3)     # True，!= 是不等
print(7 % 2 == 1) # True，7 是奇数
```

**辨析｜易错点：** `=` 是赋值，`==` 是相等判断。`if x = 3:` 是语法错误——Python 不允许在条件里直接赋值（避免把 C 语言式的「意外赋值」带进来）。这两个符号的区别是新手第一大坑。

Python 支持**链式比较**，读起来和数学一致：

```python
age = 25
print(18 <= age < 65)    # True，等价于 18 <= age and age < 65
```

每个对象都有**真值（truthiness）**：在布尔语境下，`0`、空字符串 `""`、空列表 `[]`、`None` 为**假**，其余为**真**。这意味着 `if name:` 能直接判断「名字非空」。

## 2 if/elif/else：分支的三种形态

**`if` 语句**：条件为真则执行缩进块，为假则跳过。

```python
age = 19
if age >= 18:
    print("已成年，可以投票")
```

需要「否则」时用 `else`，需要「多分支」时用 `elif`（else + if 的缩写）：

```python
age = 25
if age < 4:
    price = 0
elif age < 18:
    price = 25
elif age < 65:
    price = 40
else:
    price = 20
print(f"门票价格：${price}")
```

**重点：`elif` 自上而下、命中即止。** 一旦某个条件为真，后面的 `elif` 与 `else` 一律不再判断。这决定了**条件的顺序很关键**——把范围窄的条件放前面，范围宽的放后面，否则窄条件永远没机会命中。<span class="marginnote">上面的票价阶梯就是典型：`age < 4` 先判断，`age < 18` 再看——顺序反过来，4 岁以下的会先被 `age < 18` 拦下，价格就错了。条件分支的顺序，是一种隐藏的 bug 来源。</span>

## 3 逻辑运算：and、or 与 not

单个条件往往不够，需要**组合**条件。三个逻辑运算符：

```python
age = 22
print(age >= 18 and age < 65)   # True，同时满足
print(age < 13 or age >= 65)    # False，满足其一
print(not age >= 18)            # False，取反
```

**成员判断 `in` / `not in`** 也很常用，判断一个值是否在容器里：

```python
toppings = ["mushrooms", "peppers", "extra cheese"]
print("mushrooms" in toppings)     # True
print("pineapple" not in toppings) # True
```

**辨析｜易错点：** `and` 与 `or` 是**短路**求值——`and` 左侧为假就不再求右侧，`or` 左侧为真就不再求右侧。短路让 `if user and user.is_admin():` 这种写法安全：`user` 为空时根本不会调用 `.is_admin()`。<span class="marginnote">短路求值对性能与安全都有意义：`if a != 0 and b / a > 1:` 里，当 `a == 0` 时不会执行 `b / a`，从而避免了除零错误。这个习惯在《异常处理》一节会再次强化。</span>

## 4 公式解析：三目条件表达式

**三目表达式（conditional expression）** 把「二选一赋值」压成一行：

$$
\text{结果} = A \ \text{if}\ \text{条件}\ \text{else}\ B
$$

Python 写法：

```python
status = "成年" if age >= 18 else "未成年"
```

对这条式子做三步拆解：

- **第一步，读顺序**：读作「如果 `age >= 18` 则取 `"成年"`，否则取 `"未成年"`」。注意顺序是「真值分支在前，条件居中，假值分支在后」——与自然语言「如果……那么……否则……」的顺序不完全一致，容易看反。
- **第二步，对照等价 if**：`x = A if cond else B` 完全等价于「`if cond: x = A else: x = B`」两行。三目只是语法糖，不引入新语义。
- **第三步，理解嵌套与选择**：三目适合**单值二选一**；一旦分支内有复杂逻辑或需要多个赋值，回到完整的 `if/else` 更清晰。可读性是第一优先级——**一行能说清就用三目，说不清就老老实实写 if**。

## 5 结合实践：列表中的条件判断

`if` 常与前面的列表、字典配合，构成「过滤与分类」：

```python
requested_toppings = ["mushrooms", "green peppers", "extra cheese"]
for topping in requested_toppings:
    if topping == "green peppers":
        print("抱歉，青椒卖完了。")
    else:
        print(f"添加 {topping}。")
```

这种「遍历 + 条件 + 动作」的组合，是数据处理里最频繁的骨架——过滤无效数据、分类标签、检查权限，本质都是同一个模式。

## 6 小结

- 布尔值 `True`/`False` 来自**比较运算**；`=` 赋值、`==` 判断，不可混用。
- `if/elif/else` 自上而下、命中即止；**条件顺序决定结果**。
- `and`、`or`、`not` 组合条件，均**短路求值**；`in`/`not in` 做成员判断。
- 空值 `0`、`""`、`[]`、`None` 为假，`if name:` 可测非空。
- 三目表达式 `A if 条件 else B` 是二选一赋值的语法糖，读作「真值在前」。

在下一节，我们将深入 `for` 循环本身——`range`、循环控制与嵌套，把「重复执行」用得又准又巧。
