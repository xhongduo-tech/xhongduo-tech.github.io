---
title: 标准库常用函数 string.h/stdlib.h
date: 2026-08-07
---

# 标准库常用函数 string.h/stdlib.h

<div class="epigraph">
<p>标准库是你不必重写的轮子——但要知道每个轮子往哪转。</p>
<footer>—— 编程教学常言</footer>
</div>

<div class="article-byline">
<p>第三级 · C 语言编程 ｜ K&R《C 程序设计语言》第7章 ｜ 2026-08-07</p>
</div>

## 为什么从标准库讲起

写程序不必从零发明一切：C 标准库提供了字符串处理、内存操作、数值转换、随机数、排序等**高频通用功能**。<span class="marginnote">C 标准库分两大部分：`string.h`（字符串与内存操作）、`stdlib.h`（通用工具：转换、随机、内存分配、排序、进程）。K&R 第 7 章把它们当「标准库就是 C 的一部分」来教——用好标准库，是「会写 C」与「写得快」的分水岭。</span>这一节把两个头文件里最常用的函数梳理成表，并讲清几个「长得像、用法不同」的易混点。

## 1 string.h：字符串与内存操作

`string.h` 的函数分为**字符串类**（以 `'\0'` 判长度）与**内存类**（以显式字节数操作）：

### 字符串类

| 函数 | 作用 | 返回值 |
| --- | --- | --- |
| `strlen(s)` | 长度（不含 `'\0'`） | `size_t` |
| `strcmp(a, b)` | 字典序比较 | `<0` / `0` / `>0` |
| `strcpy(dst, src)` | 复制 | `dst` |
| `strcat(dst, src)` | 追加 | `dst` |
| `strchr(s, c)` | 找字符 | 指针或 `NULL` |
| `strrchr(s, c)` | 从右找字符 | 指针或 `NULL` |
| `strstr(h, n)` | 找子串 | 指针或 `NULL` |
| `strtok(s, delim)` | 按分隔符切分 | 下一个片段或 `NULL` |

**`strtok` 的用法**（注意它的「记忆状态」特性）：

```c
char s[] = "one,two,three";
char *token = strtok(s, ",");
while (token != NULL) {
    printf("%s\n", token);
    token = strtok(NULL, ",");   /* 第一次传 s，之后传 NULL */
}
/* 输出：one / two / three */
```

`strtok` 内部记住了「切到哪了」，所以后续调用传 `NULL` 表示继续。它**会修改原字符串**（把分隔符替换成 `'\0'`），且不是线程安全的（有线程安全版本 `strtok_r`）。<span class="marginnote">`strtok` 的「第一次传字符串、之后传 `NULL`」是 C 里少见的「有状态」接口，源自早期 C 没有闭包。它修改原串这一行为常让新手困惑——若需保留原字符串，先复制一份再切。`strtok_r` 用显式上下文解决了线程安全问题。</span>

### 内存类

| 函数 | 作用 | 与字符串类的区别 |
| --- | --- | --- |
| `memcpy(dst, src, n)` | 复制 n 字节 | 不看 `'\0'`，按字节数 |
| `memmove(dst, src, n)` | 复制 n 字节（处理重叠） | 源与目标重叠时安全 |
| `memset(ptr, v, n)` | 把 n 字节都设为 `v` | 清零常用 |
| `memcmp(a, b, n)` | 比较前 n 字节 | 二进制比较 |

`memcpy` 与 `memmove` 的区别很重要：**`memcpy` 不允许源与目标重叠，`memmove` 允许**。两者处理重叠时行为不同，`memmove` 更安全但略慢：

```c
char buf[20] = "hello world";
memmove(buf + 2, buf, 5);   /* 重叠：把 "hello" 复制到 buf+2，得到 "hehelloworld" */
/* memcpy(buf + 2, buf, 5);   重叠场景行为未定义 */
```

`memset(buf, 0, sizeof(buf))` 是「把缓冲区清零」的标准写法。

## 2 stdlib.h：转换、随机与进程

`stdlib.h` 是「通用工具库」：

### 数值转换

| 函数 | 作用 | 安全版本 |
| --- | --- | --- |
| `atoi(s)` / `atol(s)` | 字符串转整数 | `strtol` |
| `atof(s)` | 字符串转浮点 | `strtod` |
| `itoa`（非标准） | 整数转字符串 | `snprintf(buf, sz, "%d", n)` |

`atoi("42")` 返回 `42`，但**无法检测错误**——输入不是数字时行为未定义（通常返回 0）。健壮的程序用 `strtol`，它能给出「转换到哪结束、是否出错」：

```c
char *end;
long v = strtol("  123abc", &end, 10);   /* 十进制 */
printf("%ld, 剩余=%s\n", v, end);        /* 123, 剩余=abc */
```

`strtol` 第三个参数是**进制**（`10` 十进制、`16` 十六进制、`0` 自动识别前缀）。

### 随机数

| 函数 | 作用 |
| --- | --- |
| `rand()` | 返回 `0` ~ `RAND_MAX` 的伪随机数 |
| `srand(seed)` | 设置随机种子 |
| `random()` / `srandom()` | 更高质量的变体（非标准但常见） |

```c
#include <stdlib.h>
#include <time.h>

srand(time(NULL));                       /* 用当前时间做种子 */
int r = rand() % 100;                    /* 0~99 的随机数 */
int dice = rand() % 6 + 1;               /* 1~6，模拟骰子 */
```

**不调用 `srand` 直接 `rand`，每次运行得到相同的序列**（默认种子固定）——这正是「伪随机」的含义。`rand() % n` 有轻微分布不均（`RAND_MAX` 未必能被 `n` 整除），对要求严格的场景可用取整方法或 `arc4random` 等更均匀的来源。

### 进程与退出

| 函数 | 作用 |
| --- | --- |
| `exit(status)` | 立即终止程序，返回状态码 |
| `atexit(f)` | 注册程序正常退出时要调用的函数 |
| `system(cmd)` | 调用系统命令（如 `system("ls")`） |
| `abort()` | 异常终止（发 `SIGABRT`） |

`exit(0)` 与 `return 0` 效果相近，但 `exit` 可在任意位置终止，且会执行 `atexit` 注册的清理函数与刷新标准流缓冲。

### 内存与排序

`malloc`/`calloc`/`realloc`/`free`（第 3 篇已详述）与 `qsort`（下节《回调函数与 qsort》专门展开）都在 `stdlib.h` 中。

## 3 易混函数辨析

`string.h` 与 `stdlib.h` 里藏着几组「长得像、意思差很远」的函数：

| 易混组 | 区别 |
| --- | --- |
| `strcmp` vs `strncmp` | 后者只比较前 n 个字符 |
| `strcpy` vs `strncpy` | 后者限制复制长度；`strncpy` 超长时**不补 `'\0'`** |
| `atoi` vs `strtol` | 前者不报错，后者返回结束位置与错误信息 |
| `rand` vs `random` | 后者范围更大、更均匀，但非 C 标准 |
| `memcpy` vs `memmove` | 后者允许源目标重叠 |

**`strncpy` 的坑值得再强调**：`strncpy(dst, src, n)` 在 `strlen(src) >= n` 时**不会**写 `'\0'`，`dst` 不是合法字符串。安全复制的现代写法：

```c
char dst[16];
snprintf(dst, sizeof(dst), "%s", src);   /* 自动截断并保证 '\0' 结尾 */
```

## 4 公式解析：`strtol` 的转换模型

`strtol` 是「把字符串按进制解析成数字」的完整模型，理解它就能理解所有转换函数：

$$
v = \text{strtol}(s,\ \&end,\ base) \quad\Rightarrow\quad s = \underbrace{\text{空白}}_{可选} + \underbrace{[\pm]}_{符号} + \underbrace{\text{数字序列}}_{按 base 进制} + \underbrace{\text{剩余}}_{end 指向这里}
$$