---
title: 字符输入输出与 getchar/putchar
date: 2026-08-07
---

# 字符输入输出与 getchar/putchar

<div class="epigraph">
<p>一切输入最终都是字符，一切输出最终也是字符。</p>
<footer>—— 对「文本流」哲学的中文转述</footer>
</div>

<div class="article-byline">
<p>第三级 · C 语言编程 ｜ C Primer Plus 第8章 ｜ 2026-08-07</p>
</div>

## 为什么从字符输入输出讲起

`printf` 与 `scanf` 是格式化的「整块」输入输出，而更底层、也更贴近操作系统的读法是一次一个**字符（character）**。字符是文本世界的原子：文件、命令行、网络报文，最终都可以展开成一串字符。<span class="marginnote">UNIX 的哲学是「一切皆文本」：命令之间用管道传递的正是字符流。`getchar`/`putchar` 让你站在了这条文本流的源头，也理解了为什么 `wc -l`、`grep` 这类工具能那么轻巧地处理大文件——它们也是逐字符扫描的。</span>理解字符输入，还绕不开一个关键概念：**`EOF`（文件结束）**——程序如何知道「输入已经没有了」。

## 1 getchar 与 putchar：最基础的字符读写

`getchar()` 从标准输入读取**一个字符**并返回，`putchar(c)` 向标准输出写出**一个字符**：

```c
#include <stdio.h>

int main(void)
{
    int c = getchar();        /* 读取一个字符 */
    putchar(c);               /* 原样输出它 */
    putchar('\n');
    return 0;
}
```

运行它，输入 `A` 回车，程序输出 `A`。注意 `c` 被声明为 `int` 而不是 `char`——**这个细节是整个字符 I/O 的关键**，下面马上解释。

一个字符读一个字符，看起来平淡无奇，但组合上循环，`getchar` 就能逐字吞下整段输入。K&R 的经典例子是「字符计数」：

```c
long nc = 0;
int c;
while ((c = getchar()) != EOF)   /* 一直读到文件结束 */
    nc++;
printf("%ld 个字符\n", nc);
```

`while ((c = getchar()) != EOF)` 是 C 里最著名的惯用法之一：**读取一个字符、与 `EOF` 比较、若不等就进入循环**。括号的层级很重要：先赋值，再比较——若写成 `c = getchar() != EOF`，由于 `!=` 优先级高于 `=`，实际等价于 `c = (getchar() != EOF)`，`c` 会先被赋成 `0` 或 `1`，从而丢掉读到的字符。

## 2 EOF：文件结束的哨兵

**`EOF`（End of File）** 是一个在 `stdio.h` 中定义的常量，通常是 `-1`，表示「没有更多的输入了」。两个来源会产生 `EOF`：

**文件真正读完了**：用输入重定向 `./a.out < data.txt` 时，读到最后一行之后就会遇到 `EOF`。
**用户主动结束终端输入**：在终端按 `Ctrl-D`（Linux/macOS）或 `Ctrl-Z` 回车（Windows）。

`EOF` 的取值 `-1` 直接解释了为什么 `getchar` 的返回值必须用 `int`：`char` 通常是有符号的 8 位整数，能表示 `-128` 到 `127`，恰好能装下 `-1`——但标准只保证 `char` 能容纳**全部字符编码**，不保证它是无符号还是有符号。为了稳妥，`getchar` 返回的字符值加上 `EOF` 共 257 种可能，必须用至少 `int` 来容纳。<span class="marginnote">若把 `c` 声明成 `char`，在某些实现上 `getchar()` 返回 `0xFF`（如 `ÿ`）会被截断、与 `EOF`（`-1` 的位模式 `0xFFFFFFFF`）比较时可能永远不等或永远相等——「`char` 装不下返回值」是经典未定义行为来源。所以 K&R 的约定是：字符 I/O 一律用 `int c`。</span>

## 3 行缓冲与「读不到字符」之谜

在终端运行时，你会发现 `getchar` 并不是**每按一个键就立刻返回**——程序看起来「卡住」了。原因是终端输入默认是**行缓冲（line buffered）**的：

1. 你的按键先进入**输入缓冲（input buffer）**，屏幕上的回显也由终端处理；
2. 直到你按下**回车键**，缓冲区里的整行内容才交给程序的 `getchar`；
3. `getchar` 从缓冲区一次取一个字符，取空了就等待下一行。

所以 `while ((c = getchar()) != EOF)` 在终端下，通常是**按一个字符 → 按回车 → 程序读到一个字符**的节奏。要让程序「即时响应按键」而不等回车，需要关闭行缓冲（在 Unix 上可用 `termios` 设置原始模式），那是系统编程的话题。

**输入重定向**可以绕开终端交互，把文件喂给程序：

```bash
$ ./count \lt  data.txt
128 个字符
```

管道则把前一个命令的输出直接接成程序的输入：

```bash
$ echo "hello" | ./count
6 个字符
```

`echo "hello"` 输出 `hello\n` 共 6 个字符，`./count` 读完后遇到 `EOF`，输出 `6`。<span class="marginnote">`<` 重定向让 `getchar` 从文件读，`|` 管道让它从上一个命令读——两者都是标准输入的不同「来源」。这正是 UNIX「程序即过滤器」模型的基础：程序只关心从 stdin 读、往 stdout 写，来源与去向由 Shell 决定。</span>

## 4 从字符到词：一个完整的词计数程序

把字符、词、行三个计数器合起来，就是 K&R 里著名的 **`wc` 词计数**程序：

```c
#include <stdio.h>

#define IN  1   /* 在词内 */
#define OUT 0   /* 在词外 */

int main(void)
{
    int c, nl = 0, nw = 0, nc = 0, state = OUT;
    while ((c = getchar()) != EOF) {
        nc++;                 /* 字符数 +1 */
        if (c == '\n')
            nl++;             /* 行数 +1 */
        if (c == ' ' || c == '\n' || c == '\t')
            state = OUT;      /* 空白把词与词隔开 */
        else if (state == OUT) {
            state = IN;       /* 由外入内：一个新词开始 */
            nw++;
        }
    }
    printf("%d %d %d\n", nl, nw, nc);
    return 0;
}
```

这段程序展示了「**状态机（state machine）**」的思想：用一个 `state` 变量记住「当前是否在词里」，每当字符由空白变为非空白，就断定一个新词开始。**程序的正确性取决于状态转移的完备性**——所有可能的输入（字母、空白、换行、`EOF`）都必须有确定的去向。

`wc` 程序是 `getchar` 循环的集大成者，也印证了一个通用模式：**逐字符处理 → 边读边统计/变换 → 遇 `EOF` 收尾**。文本分析、词频统计、语法高亮的雏形都是这套骨架。

## 5 公式解析：`(c = getchar()) != EOF` 的优先级拆解

这个惯用法能成立，完全取决于 C 的运算符优先级，拆开看：

$$
\underbrace{(c = \underbrace{getchar()}_{\text{函数调用，最高}} )}_{\text{括号内赋值}} \neq \underbrace{EOF}_{\text{宏，-1}}
$$