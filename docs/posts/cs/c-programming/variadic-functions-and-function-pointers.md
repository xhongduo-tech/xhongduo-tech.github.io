---
title: 可变参数与函数指针
date: 2026-08-07
---

# 可变参数与函数指针

<div class="epigraph">
<p>把行为当作数据传递，是 C 语言通往「高阶」的第一步。</p>
<footer>—— 对函数指针的中文转述</footer>
</div>

<div class="article-byline">
<p>第三级 · C 语言编程 ｜ K&R《C 程序设计语言》第5、7章 ｜ 2026-08-07</p>
</div>

## 为什么从可变参数与函数指针讲起

`printf("x=%d, y=%s", x, s)` 的参数数量是可变的——它怎么能做到？函数能接收「行为」吗——让排序函数不知道元素类型也能排？这两个问题分别由**可变参数（variadic functions）** 与**函数指针（function pointers）** 回答。<span class="marginnote">可变参数让 `printf`/`scanf` 成为可能；函数指针让 `qsort` 这样的「通用算法」得以实现——算法不知道数据的类型与比较规则，只通过一个函数指针调用调用者提供的「比较行为」。这两者合起来，是 C 实现「泛型」与「回调」的两大支柱。</span>这一节先讲可变参数的标准机制（`stdarg.h`），再深入函数指针的声明、调用与回调应用。

## 1 可变参数：数量不定的参数

`printf` 的签名形如：

```c
int printf(const char *format, ...);
```

省略号 `...` 表示「后面还有任意数量的参数」。C 提供 `stdarg.h` 来遍历这些参数，核心是**四步套路**：

```c
#include <stdarg.h>

int sum(int count, ...)   /* count 是参数个数，... 是可变部分 */
{
    va_list ap;           /* 参数列表游标 */
    int total = 0;

    va_start(ap, count);  /* 初始化：从 count 之后开始取 */
    for (int i = 0; i < count; i++)
        total += va_arg(ap, int);   /* 依次取出一个 int */
    va_end(ap);           /* 清理 */

    return total;
}

printf("%d\n", sum(3, 10, 20, 30));   /* 60 */
```

四步：**`va_start` 初始化 → `va_arg` 逐个取参 → `va_end` 清理**。关键限制：

必须有至少一个**命名参数**（这里是 `count`），`va_start` 以它为起点。
**`va_arg` 无法知道类型**——必须由你（或格式串）告诉它取多大：`va_arg(ap, int)` 会按 `int` 从栈上取 4 字节。类型传错，读出的字节全错。<span class="marginnote">这就是 `printf` 的格式串为什么必须匹配参数类型：`%d` 对应 `int`、`%f` 对应 `double`。`va_arg` 是「无类型」的裸读取，全靠调用者的约定。类型不匹配时编译期无提示，运行时输出乱码——这是可变参数的主要风险。</span>
- **`float` 会被提升为 `double`**：可变参数里取 `float` 要写 `va_arg(ap, double)`。

**安全惯例**：可变参数函数必须有一个「告诉有多少个/什么类型」的信息来源——要么是显式的计数参数（`sum(3, ...)`），要么是格式串（`printf`）。缺了它，函数无法知道取到何时为止。

## 2 实现一个 printf 风格的格式化函数

把可变参数与 `vprintf` 家族结合，可以包装日志函数：

```c
#include <stdio.h>
#include <stdarg.h>

void log_msg(const char *level, const char *fmt, ...)
{
    printf("[%s] ", level);
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);     /* 把 va_list 直接交给 vprintf */
    va_end(ap);
    printf("\n");
}

log_msg("INFO", "用户 %s 登录，id=%d", "alice", 1001);
/* 输出：[INFO] 用户 alice 登录，id=1001 */
```

`vprintf`/`vfprintf`/`vsnprintf` 接受 `va_list` 而非直接参数——它们让「自己的可变参数函数」能复用标准库的格式化逻辑。**把 `va_list` 传给别的函数前，应在该函数内 `va_start`/`va_end` 成对使用**。

## 3 函数指针：把函数当作值

第 3 篇《复杂声明与 typedef》见过函数指针的声明，这里正式讲透**用法**。函数指针保存**函数的入口地址**，声明、赋值、调用的完整链条：

```c
int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }

int (*op)(int, int);      /* 声明：op 是函数指针 */
op = add;                 /* 赋值：函数名退化为指针 */
int r = op(10, 5);        /* 调用：等价于 add(10, 5) */
op = sub;                 /* 换一个函数 */
r = op(10, 5);            /* 现在调用的是 sub，结果是 5 */
```

要点：

- 函数名在表达式中**退化为指向函数的指针**（与数组名退化同理），所以 `op = add` 不带括号。
- 调用 `op(10, 5)` 与直接 `add(10, 5)` 语法相同——编译器见 `op` 是函数指针，自动解引用调用。
- 签名必须匹配：`op` 是「接收两个 `int`、返回 `int`」，赋值时 `add` 的签名必须一致，否则编译错误。

**函数指针数组**（第 3 篇）是查表派发的基础；这里看一个更贴近业务的应用——**用函数指针实现「策略」**。

## 4 回调：把行为传给通用算法

**回调（callback）** 指「把一个函数作为参数传给另一个函数，由后者在适当时候调用」。`qsort` 是 C 标准库最经典的回调例子：

```c
#include <stdlib.h>

int cmp_int(const void *a, const void *b)
{
    int x = *(const int *)a;
    int y = *(const int *)b;
    return (x > y) - (x < y);     /* 三态比较：-1/0/1 */
}

int main(void)
{
    int arr[] = {42, 7, 13, 99, 5};
    qsort(arr, 5, sizeof(int), cmp_int);   /* 回调 cmp_int */
    /* arr 变为 {5, 7, 13, 42, 99} */
    return 0;
}
```

`qsort` 的第四个参数是**比较函数指针**。关键在于 `qsort` **完全不关心元素的类型**——它只知道每个元素占 `sizeof(int)` 字节，比较规则由调用者通过回调提供。换一组数据、换一个比较函数，`qsort` 不用改一行：

```c
int cmp_str(const void *a, const void *b)
{
    return strcmp(*(const char **)a, *(const char **)b);
}

char *words[] = {"pear", "apple", "orange"};
qsort(words, 3, sizeof(char *), cmp_str);   /* 按字典序排序 */
```

**「不知道类型也能排序」** 正是回调的威力：通用算法（排序、遍历、查找）与具体行为（怎么比较、怎么处理）解耦。这个思想一路延伸到 C++ 的模板、Python 的 `sorted(key=...)`、现代框架的事件监听——**回调是「把函数当参数」的祖师爷**。

## 5 核心对比表：函数调用 vs 函数指针

| 维度 | 直接调用 `add(1,2)` | 通过指针 `op(1,2)` |
| --- | --- | --- |
| 函数名身份 | 编译期绑定 | 运行期可换 |
| 灵活性 | 写死 | 可运行时决定调用谁 |
| 编译优化 | 可内联 | 间接调用，难内联 |
| 典型用途 | 普通调用 | 回调、派发表、事件处理 |
| 性能 | 直接跳转 | 多一次间接跳转 |

核心权衡：**直接调用是「写死的直连」，函数指针是「可换的接线」**——多一层间接换来运行时灵活性，代价是微小性能开销与更难内联。排序、信号处理、GUI 事件、插件系统这些「行为可插拔」的场景，函数指针是不可替代的。

## 6 公式解析：`op = add` 后 `op(a, b)` 的调用链

函数指针调用展开成一次「取地址 → 间接跳转」：

$$
op = add \;\Rightarrow\; op(a,b) \;\equiv\; (\ast\text{op})(a,b) \;\equiv\; add(a,b)
$$