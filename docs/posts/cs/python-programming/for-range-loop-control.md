---
title: 深入流程控制：for 循环、range 与循环控制语句
date: 2026-08-07
---

# 深入流程控制：for 循环、range 与循环控制语句

<div class="epigraph">
<p>重复是学习之母，循环是算法之母。</p>
<footer>—— 改编自拉丁谚语 Repetitio est mater studiorum</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 官方 Python 教程 第4章 ｜ 2026-08-07</p>
</div>

## 为什么把流程控制讲深一层

前面两节分别学了 `for` 与 `while`，但流程控制的细节远不止「会写一个循环」。真实代码里的循环往往嵌套、带提前退出、需要同时遍历多个序列——这些进阶形态，官方 Python 教程第 4 章做了系统梳理，也正是本节的内容。

把循环从「会写」提升到「写得准」，关键在四个点：`range` 的惰性本质、`break/continue/else` 三个控制语句的精确语义、嵌套循环的节奏、以及 `zip`/`enumerate` 这类「同时遍历」的利器。

## 1 range 的惰性本质与步进

`range()` 在遍历数值序列时几乎无处不在，但它的真实身份常被误解。

**`range` 返回的是「可迭代对象」，不是列表。** 它按需生成数字，不一次性占用内存。看一个证据：

```python
big = range(10 ** 9)      # 十亿个数，但内存几乎为零
print(10 ** 9 in big)     # 很快判断出 True，而不是真的生成十亿个数
```

**重点：`range` 是惰性（lazy）的。** 它只存储起点、终点、步长三个参数，每次迭代才算出下一个值。这一设计让 `range(10**9)` 这种「天文数字的序列」也能安全创建。<span class="marginnote">惰性求值是大数据与流式处理的核心思想：能按需生成就不提前生成。本专题《迭代器与生成器》会把它推广到任意自定义序列；第二级《数值分析》里「向量不整块分配、按索引访问」也是同一哲学。</span>

**步长的三种典型用法**：

```python
list(range(5))            # [0, 1, 2, 3, 4]          默认从 0、步长 1
list(range(2, 8))         # [2, 3, 4, 5, 6, 7]       指定起点
list(range(0, 10, 2))     # [0, 2, 4, 6, 8]          步长 2
list(range(5, 0, -1))     # [5, 4, 3, 2, 1]          负步长倒序
```

## 2 循环控制三兄弟：break / continue / else

**`break`**：立即终止整个循环，程序跳到循环之后的语句。

```python
for n in range(2, 100):
    if n * n > 50:
        break            # 找到第一个平方超过 50 的数就停
    print(n)             # 打印 2..7
```

**`continue`**：跳过本次迭代剩余语句，进入下一轮。

```python
for n in range(10):
    if n % 3 == 0:
        continue         # 跳过 3 的倍数
    print(n, end=" ")    # 1 2 4 5 7 8
```

**`else` 子句**：这是 Python 循环的独门武器——**循环正常结束（没被 break 打断）时才执行 `else` 块**。

```python
for n in range(2, 10):
    for d in range(2, n):
        if n % d == 0:
            break        # 找到因子，n 不是质数
    else:                # 内层没 break，说明 n 是质数
        print(n, end=" ")   # 2 3 5 7
```

**重点：`for...else` 的语义是「循环没被 break 就执行 else」。** 上面的质数判定是经典示范——用 `break` 标记「找到了」，用 `else` 标记「没找到」。这个写法在需要「搜索是否存在」的场景非常优雅。

**辨析｜易错点：** `else` 不是在「循环结束后」执行，而是在「循环未被 break 中断而自然结束」时执行。如果循环被 `break` 打断，`else` 会被跳过。搞混这一点，是 `for-else` 最常见的误用。

## 3 嵌套循环：外层换一次，内层跑一圈

循环体内再套循环，就是**嵌套循环（nested loop）**。外层每执行一次，内层完整跑一遍。

```python
for i in range(3):
    for j in range(2):
        print(f"({i}, {j})", end=" ")
    print()
```

输出：

```
(0, 0) (0, 1) 
(1, 0) (1, 1) 
(2, 0) (2, 1) 
```

**重点：嵌套循环的总执行次数 = 外层次数 × 内层次数。** 用 `range` 嵌套生成的 `(i, j)` 对，正是「笛卡尔积」的编程表达——遍历二维表格、枚举坐标、生成全排列的起点都在这里。<span class="marginnote">笛卡尔积在数学里是集合运算：$A \times B = \{(a,b) \mid a \in A, b \in B\}$</span>