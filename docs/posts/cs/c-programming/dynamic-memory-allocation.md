---
title: 动态内存分配 malloc/free
date: 2026-08-07
---

# 动态内存分配 malloc/free

<div class="epigraph">
<p>动态内存是程序在运行时向堆要地盘的机制——给得好是灵活，管理不好是泄漏与悬垂。</p>
<footer>—— 对 C 内存管理的中文转述</footer>
</div>

<div class="article-byline">
<p>第三级 · C 语言编程 ｜ C Primer Plus 第12章 ｜ 2026-08-07</p>
</div>

## 为什么从动态内存讲起

之前的所有数组，大小都在编译期定死：`int arr[100];` 需要多少就得提前猜。可真实程序往往**运行时才知道**需要多少内存——读入多少行数据、链表要挂多少个节点、图像有多大。<span class="marginnote">C 提供<strong>堆（heap）</strong>：程序运行时向操作系统「现要现还」的内存区。`malloc` 从堆上借一块、`free` 归还。这是 C 灵活性的来源，也是内存泄漏与悬垂指针这两大「C 专属杀手」的根源。现代语言用垃圾回收接管了这一切，但理解 C 的做法，才能真正理解「内存到底怎么被管理」。</span>这一节讲清 `malloc` 家族的用法、生命周期管理，以及三类经典内存错误。

## 1 malloc：向堆借一块内存

**`malloc`（memory allocation）** 从堆上分配一块指定字节数的内存，返回指向它的指针：

```c
#include <stdlib.h>

int *p = (int *)malloc(10 * sizeof(int));  /* 申请 10 个 int 的空间 */
if (p == NULL) {
    /* 分配失败：内存不足 */
    fprintf(stderr, "内存不足\n");
    return 1;
}
```

要点：

参数是**字节数**，所以通常写 `n * sizeof(int)`，而不是 `n`。`sizeof(int)` 保证在不同平台上都正确。
返回 `void *`，C 中会自动转换为任意指针类型，但显式 `(int *)` 转换是惯用法（C++ 必需）。
- **返回值必须检查**：分配失败返回 `NULL`。不检查就使用，解引用 `NULL` 是未定义行为。<span class="marginnote">写 `n * sizeof(int)` 还有个溢出隐患：若 `n` 极大，`n * sizeof(int)` 可能溢出成一个小数，分配一块远小于预期的内存。安全写法是 `n > SIZE_MAX / sizeof(int)` 先做检查——这是审计 CVE 时常讨论的整数溢出攻击面。</span>

**`calloc` 与 `realloc`** 是 `malloc` 的姊妹：

```c
int *a = (int *)calloc(10, sizeof(int));   /* 分配并全部清零 */
int *b = (int *)realloc(a, 20 * sizeof(int)); /* 调整大小：可能移动 */
```

- `calloc(n, size)`：分配 n 块 size 字节，**全部初始化为 0**——比 `malloc` + 手动清零更方便。
- `realloc(ptr, new_size)`：调整已分配块的大小。可能原地扩大，也可能搬到新地址（原指针失效）。返回 `NULL` 表示失败，**此时原块仍有效**，别把返回值直接赋给原指针（会丢）。

## 2 free：归还内存

用完之后必须用 **`free`** 归还：

```c
free(p);       /* 把 p 指向的块还给堆 */
p = NULL;      /* 习惯：置空，避免悬垂指针 */
```

`free` 只接受 `malloc`/`calloc`/`realloc` 返回的指针。三大禁忌：

- **重复释放（double free）**：对同一指针 `free` 两次，是未定义行为，多数实现会崩溃。
- **释放后仍使用（use-after-free）**：`free(p)` 之后还访问 `*p`，是未定义行为——内存可能已被复用，数据随时会变。
- **释放非堆内存**：`free` 一个栈上数组的地址，同样是未定义行为。

**悬垂指针（dangling pointer）** 是「释放后仍指向那块内存的指针」——它像一张过期地图，指向的地方可能已属于别人。

## 3 内存泄漏：借了不还

**内存泄漏（memory leak）**：分配了内存但不再使用、也不 `free`，直到程序结束也不归还。程序长期运行（服务器、浏览器、游戏）时，泄漏会**逐渐耗尽内存**：

```c
for (int i = 0; i < 1000000; i++) {
    char *buf = (char *)malloc(1024);
    /* 忘了 free(buf) —— 每轮泄漏 1024 字节 */
}
```

防治手段：

- **配对检查**：每次 `malloc` 都问自己「对应 `free` 在哪」。
- **尽早释放**：用完立即 `free`，不要等到函数末尾。
- **用工具检测**：Valgrind 是 Linux 下检测泄漏与内存错误的标配工具：`valgrind ./a.out` 会在程序结束时报告「definitely lost: N bytes」。
- **约定所有权**：谁分配谁释放（resource acquisition is initialization 的 C 版纪律）。

## 4 三类经典内存错误速览

| 错误 | 场景 | 后果 |
| --- | --- | --- |
| 缓冲区溢出 | 写入超过分配大小的范围 | 破坏相邻内存，可能崩溃或被利用 |
| 悬垂指针 / use-after-free | 释放后仍访问 | 数据随机、段错误 |
| 内存泄漏 | 分配后不释放 | 长期运行内存耗尽 |

这三类错误**不一定立刻崩溃**——可能运行几小时后才爆炸，或者数据「偶尔变错」。这正是 C 内存错误难调试的原因：**错误发生与症状出现之间隔着很长的距离**。这也是第 5 篇《调试、断言与常见错误陷阱》要专门展开的话题。

## 5 动态内存与指针：一个完整例子

动态数组是最典型的用法：

```c
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int n;
    printf("要存几个数？");
    if (scanf("%d", &n) != 1 || n <= 0)
        return 1;

    int *arr = (int *)malloc(n * sizeof(int));
    if (arr == NULL) {
        fprintf(stderr, "分配失败\n");
        return 1;
    }

    for (int i = 0; i < n; i++)
        arr[i] = i * i;

    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += arr[i];
    printf("前 n 个平方和：%d\n", sum);

    free(arr);
    return 0;
}
```

这个例子串起本节全部要点：**用 `malloc` 按运行时长度申请 → 检查 `NULL` → 像数组一样用下标访问 → 用完 `free`**。动态数组与静态数组的唯一差别是「谁在何时分配内存」，访问语法完全一致——因为 `arr[i]` 就是 `*(arr + i)`，指针与数组的等价在这里再次体现。

## 6 公式解析：`malloc(n * sizeof(int))` 的字节账

动态分配最容易错的是字节数，把账算清楚：

$$
\text{申请字节数} = n \times \text{sizeof}(int)
$$

- **第一步，定每元素大小**：`int` 在当前平台占 `sizeof(int)` 字节，64 位 Linux 下是 4。
- **第二步，乘元素个数**：要存 n 个 `int`，需要 $n \times 4$