---
title: 头文件、多文件程序与编译单元
date: 2026-08-07
---

# 头文件、多文件程序与编译单元

<div class="epigraph">
<p>模块化的目标很简单：让每个文件只做一件事，让每个文件只被别人需要它的部分。</p>
<footer>—— 对软件模块化的中文转述</footer>
</div>

<div class="article-byline">
<p>第三级 · C 语言编程 ｜ C Primer Plus 第9章 ｜ 2026-08-07</p>
</div>

## 为什么从多文件讲起

真实程序不会只有一个 `.c` 文件。把几十万行代码拆成一个个源文件，各自编译、最后拼装——这套「**分开编译、统一链接**」的流程，是 C 工程化的基石。<span class="marginnote">Linux 内核有上万个子目录、数千个源文件。能让它们协作，靠的正是「编译单元」边界与头文件契约：每个 `.c` 独立编译成 `.o`，链接器再按需拼装。理解了这套机制，你才算真正理解了 C 工程。</span>这一节讲清头文件的作用、多文件项目的组织方式，以及一个反复强调的纪律：**头文件声明、源文件定义**。

## 1 为什么需要头文件

设想两个源文件需要共享一个函数与一个全局变量：

```c
/* math_utils.c */
int add(int a, int b) { return a + b; }
int shared = 100;
```

```c
/* main.c */
#include <stdio.h>

int add(int a, int b);   /* 手写原型 */
extern int shared;       /* 手写 extern 声明 */

int main(void)
{
    printf("%d\n", add(3, 4));
    printf("%d\n", shared);
    return 0;
}
```

这能工作，但**手写声明很脆弱**：`math_utils.c` 若改了函数签名，`main.c` 的声明会悄悄失配，编译不报错、运行出乱子。**头文件（header file）** 就是「把声明集中到一处」的解法：

```c
/* math_utils.h */
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

int add(int a, int b);
extern int shared;

#endif
```

`main.c` 只需 `#include "math_utils.h"`，声明永远与定义保持同步——因为**头文件只有一份，谁改谁负责**。<span class="marginnote">`<>` 与 `""` 的区别：`#include <stdio.h>` 让预处理器在<strong>系统目录</strong>查找；`#include "math_utils.h"` 先查<strong>当前目录</strong>，再查系统目录。惯例：标准库用 `<>`，自己的头文件用 `""`。</span>

**头文件里放什么**：函数原型、`extern` 变量声明、宏定义、结构体定义（第 3 篇）、`typedef`。**不放什么**：函数的定义（实现）与变量定义——否则多个源文件都包含它，链接时重定义。

## 2 编译单元：文件之间的墙

一个 `.c` 文件经过预处理、编译后，产出一个 `.o` 目标文件，称为一个**编译单元（translation unit）**。关键性质：

**每个编译单元独立编译**：`main.c` 编译时看不到 `math_utils.c` 的任何内容，只依赖头文件给出的声明。
**编译单元之间靠「声明 + 链接」协作**：`main.c` 调用 `add` 时，只知道它的签名；真正找到实现是链接阶段的事。
- **头文件保护（include guard）**：`#ifndef ... #define ... #endif` 防止同一个头文件被重复包含。现代编译器还支持 `#pragma once` 作为等效、更简洁的写法。

```c
#ifndef MATH_UTILS_H     /* 若未定义过 MATH_UTILS_H */
#define MATH_UTILS_H     /* 就定义它 */
/* ... 头文件内容 ... */
#endif                   /* 若已定义，整段跳过 */
```

为什么需要保护？因为 A 头文件可能包含 B 头文件，而 C 头文件也包含 B——若 B 没保护，B 的内容会在一个编译单元里出现两次，导致结构体重定义等错误。<span class="marginnote">`#pragma once` 是更现代的写法，绝大多数编译器支持，且比 `#ifndef` 更不易写错。但 `#ifndef` 是标准语言特性，可移植性最稳。两种都常见，选一种保持一致即可。</span>

## 3 多文件项目的编译与链接

两个源文件 + 一个头文件的项目，编译命令是：

```bash
$ gcc -c main.c -o main.o          # 编译 main.c → main.o
$ gcc -c math_utils.c -o math_utils.o   # 编译 math_utils.c → math_utils.o
$ gcc main.o math_utils.o -o app   # 链接两个目标文件 → app
```

也可以一步到位 `gcc main.c math_utils.c -o app`。**分开编译的价值在于增量构建**：改一个文件只需重编那一个文件，再链接——大项目里省下的时间极其可观。`make` 工具正是自动追踪「哪个文件改了、需要重编什么」的（见《链接、库与 Makefile》一节）。

**链接错误（link error）** 与编译错误区分开：

- 编译错误：单个文件内，语法/类型问题，报错时给出文件名与行号。
- 链接错误：目标文件拼装时发现缺失或冲突，如 `undefined reference to 'add'`（没人实现 `add`）或 `multiple definition of 'shared'`（`shared` 被定义两次）。

常见链接错误与成因：

| 链接错误 | 典型成因 |
| --- | --- |
| `undefined reference to 'f'` | 声明了 `f` 但没人提供定义，或忘了链接对应的 `.o`/库 |
| `multiple definition of 'x'` | 变量 `x` 在多个 `.c` 里都写了定义 |
| 段错误发生在 main 之前 | 全局对象初始化顺序问题（少见但诡异） |

## 4 头文件的设计原则

**原则一：每个 `.c` 配一个同名 `.h`**。`math_utils.c` 的公开接口放进 `math_utils.h`，其他文件只 `#include` 头文件、绝不手写声明。

**原则二：头文件应当是「自包含」的**——任何源文件包含它时，不必先包含别的头文件。若头文件用到 `size_t`，就自己 `#include <stddef.h>`。

**原则三：头文件只放声明，不放定义**。函数定义与变量定义留在 `.c`。唯一的例外是 `static inline` 函数与宏——它们必须在头文件里定义，因为每个使用点都需要完整的函数体。

**原则四：头文件的内容应当「低调」**——尽量少用全局变量，多用函数接口。暴露的符号越少，改动的冲击面越小。

```c
/* counter.h */
#ifndef COUNTER_H
#define COUNTER_H

void counter_reset(void);
int  counter_value(void);
void counter_increment(void);

#endif
```

```c
/* counter.c */
#include "counter.h"

static int count = 0;      /* 内部链接：外部看不到、也改不了 */

void counter_reset(void)    { count = 0; }
int  counter_value(void)    { return count; }
void counter_increment(void){ count++; }
```

这个例子展示了 C 模块化的精髓：**数据藏在 `.c` 内部（`static`），对外只暴露函数接口**——调用者无法直接改 `count`，只能通过三个函数操作它。这比暴露全局变量安全得多，也是面向对象「封装」思想的 C 版本。

## 5 核心对比表：声明与定义

| 维度 | 声明（declaration） | 定义（definition） |
| --- | --- | --- |
| 作用 | 告诉编译器名字的类型/签名 | 分配内存/提供实现 |
| 出现次数 | 可多次 | 只能一次 |
| 存放位置 | 头文件 | 源文件 |
| 函数例子 | `int add(int, int);` | `int add(int a, int b) { ... }` |
| 变量例子 | `extern int shared;` | `int shared = 100;` |

**「声明可重复、定义唯一」是 C 的全部模块化纪律的浓缩**。声明让其他文件「知道有这个接口」，定义让链接器「找到唯一实现」。违反它就会撞上 `undefined reference` 或 `multiple definition` 这两种典型链接错误。

## 6 小结

- 头文件集中存放函数原型、`extern` 声明、宏与结构体定义，让多文件共享接口且永不失配。
- 编译单元 = 一个 `.c` 预处理编译后的 `.o`；文件间靠声明协作、链接器拼装。
- 头文件保护 `#ifndef/#define/#endif`（或 `#pragma once`）防止重复包含。
- 编译错误是单文件内的；链接错误是拼装时的，常见 `undefined reference` 与 `multiple definition`。
- 设计原则：`.c` 配同名 `.h`、头文件自包含、只放声明、数据用 `static` 隐藏、只暴露函数。

在下一节，我们将深挖 `#` 开头的预处理世界——宏、条件编译与 `include` 的完整机制，这是多文件工程最依赖的编译期工具。
