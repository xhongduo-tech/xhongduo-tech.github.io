---
title: 控制流：循环与分支
date: 2026-08-07
---

# 控制流：循环与分支

<div class="epigraph">
<p>结构程序设计的原则是：程序的控制流应当仅由顺序、选择与循环三种基本结构构成。</p>
<footer>—— 艾兹格 · 迪科斯彻（Edsger W. Dijkstra）</footer>
</div>

<div class="article-byline">
<p>第三级 · C 语言编程 ｜ K&R《C 程序设计语言》第3章 ｜ 2026-08-07</p>
</div>

## 为什么从控制流讲起

顺序执行的程序只能做「一条道走到黑」的计算，而真实世界的算法需要**判断**与**重复**：判断成绩是否及格，重复累加一亿个数。<span class="marginnote">1966 年 Böhm 与 Jacopini 证明了「顺序、选择、循环」三种结构足以表达一切算法——这就是结构化编程（structured programming）的理论基础，C 语言的 `if/else`、`switch`、`while/for` 正是这三种结构的直接落地。</span>这一节把 C 的分支与循环语句逐个讲透，并回答一个核心问题：面对同一任务，该选哪种循环。

## 1 分支：if / else 与 else-if

`if` 语句让程序在两种路径中二选一：

```c
int score = 85;
if (score >= 60) {
    printf("及格\n");
} else {
    printf("不及格\n");
}
```

多条件分级用 **`else if` 链**：

```c
if (score >= 90)
    printf("优秀\n");
else if (score >= 60)
    printf("及格\n");
else
    printf("不及格\n");
```

注意 `else` 的**就近配对**：`else` 永远与**最近的未配对 `if`** 结合，而与缩进无关。

```c
if (a > 0)
    if (b > 0)
        printf("A\n");
else            /* 这个 else 匹配的是 if(b>0)，不是 if(a>0) */
    printf("B\n");
```

**条件表达式（conditional expression）** 是 `if/else` 的表达式形态：`条件 ? 表达式1 : 表达式2`。它**有值**，可以嵌入更大的表达式：

```c
int max = (a > b) ? a : b;   /* max 取 a、b 中的较大者 */
printf("%s\n", (score >= 60) ? "pass" : "fail");
```

## 2 分支：switch 语句

当分支条件是「与一个整数值精确相等」的多种情况时，`switch` 比一长串 `else if` 更清晰：

```c
int day = 3;
switch (day) {
case 1:
    printf("星期一\n");
    break;
case 2:
    printf("星期二\n");
    break;
case 3:
    printf("星期三\n");
    break;
default:
    printf("未知\n");
    break;
}
```

三个要点：

**`case` 后必须是整型常量表达式**，不能是变量或浮点。
**每个 `case` 末尾通常要 `break`**：否则会**贯穿（fall-through）**，继续执行下一个 `case`。贯穿本身是合法特性（可以用来合并多个 case：`case 1: case 2:` 共享同一段代码），但忘记 `break` 是高频 bug。<span class="marginnote">贯穿（fall-through）在合并同类项时很实用：`case 'a': case 'e': case 'i':` 三个字母共用一段处理代码。但多数编译器会警告「可能存在贯穿」，最好用注释显式标注这是有意为之。</span>
`default` 处理所有未匹配的情况，通常放在最后。

## 3 循环：while 与 do-while

**`while` 循环**：先判断条件，条件为真才进入循环体。

```c
int i = 0;
while (i < 10) {
    printf("%d ", i);
    i++;
}
/* 输出：0 1 2 3 4 5 6 7 8 9 */
```

**`do-while` 循环**：先执行循环体，再判断条件——**循环体至少执行一次**。

```c
int n;
do {
    printf("请输入一个正数：");
    scanf("%d", &n);
} while (n <= 0);
/* 不管第一次输入什么，都会先执行一次，然后判断是否需要重来 */
```

`do-while` 适合「先做后判断」的场景，比如菜单驱动程序的「至少显示一次菜单」。

**一个经典陷阱**：`while` 与分号的组合。

```c
int i = 0;
while (i < 10);    /* 条件后多了分号：死循环！ */
{
    printf("%d", i);
    i++;
}
```

`while` 后面的单独分号构成空语句，条件永远不变，程序死循环。<span class="marginnote">编译器对空循环体通常会给警告。`while(1)` 或 `for(;;)` 是故意写的无限循环，常与 `break` 配合做「读到输入结束为止」的模式。</span>

## 4 循环：for 语句

`for` 把循环的三要素——**初始化、条件、步进**——集中到一行，是最紧凑也最常用的循环：

```c
for (int i = 0; i < 10; i++)
    printf("%d ", i);
```

`for (初始化; 条件; 步进)` 的执行顺序是：初始化 → 判断条件 → 执行循环体 → 步进 → 再判断条件 → …… 三部分都可以省略，但两个分号必须保留：

```c
int i = 0;
for (; i < 10;) {    /* 等价于 while 循环 */
    printf("%d ", i);
    i++;
}
```

C99 起允许在初始化部分**声明局部变量**：`for (int i = 0; ...)` 中的 `i` 只在整个 `for` 语句内可见。

**循环嵌套**是二维问题（矩阵、表格、九九乘法表）的天然写法：

```c
for (int i = 1; i <= 9; i++) {
    for (int j = 1; j <= 9; j++)
        printf("%d ", i * j);
    printf("\n");
}
```

## 5 break / continue / goto

- **`break`**：跳出**最内层**的循环或 `switch`，程序继续执行循环后面的语句。
- **`continue`**：跳过本次循环体的**剩余部分**，直接进入下一次迭代（`for` 中会先执行步进）。
- **`goto`**：无条件跳转到指定的标签。由于它会破坏结构化，一般只在「从多层嵌套循环中一次性跳出」时使用。

```c
for (int i = 0; i < 10; i++) {
    if (i == 5)
        break;       /* i 到 5 就彻底结束循环 */
    if (i % 2 == 0)
        continue;    /* 偶数跳过打印，直接下一轮 */
    printf("%d ", i);
}
/* 输出：1 3 */
```

多层嵌套的提前退出，`goto` 比层层 `break` 更干净：

```c
for (int i = 0; i < 10; i++)
    for (int j = 0; j < 10; j++) {
        if (grid[i][j] == target)
            goto found;
    }
found:
printf("找到了！\n");
```

## 6 核心对比表：三种循环怎么选

| 循环 | 先判后做 / 先做后判 | 适用场景 | 典型写法 |
| --- | --- | --- | --- |
| `while` | 先判后做 | 循环次数未知，靠条件退出 | 读文件直到 `EOF` |
| `do-while` | 先做后判 | 至少执行一次 | 菜单、重试输入 |
| `for` | 先判后做 | 次数已知或可计数 | 遍历数组 `0..n-1` |

选型口诀：**知道次数用 `for`，只知道条件用 `while`，必须至少跑一次用 `do-while`**。三者可相互改写，但选择合适的形态能让意图一目了然——`for (int i = 0; i < n; i++)` 一眼就知道「遍历 n 次」，这是选型最重要的价值。

## 7 公式解析：`for` 循环的迭代模型

`for` 循环的语义可以用一个等价变换精确描述：

$$
\text{for}(init;\ cond;\ step)\ \text{body} \quad\equiv\quad init;\ \text{while}(cond)\ \{\ \text{body};\ step;\ \}
$$