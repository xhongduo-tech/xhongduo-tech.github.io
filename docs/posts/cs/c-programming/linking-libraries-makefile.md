---
title: 链接、库与 Makefile
date: 2026-08-07
---

# 链接、库与 Makefile

<div class="epigraph">
<p>编译是让每个文件自洽，链接是让它们彼此找到对方，Makefile 是记住这一切的分工表。</p>
<footer>—— 对 C 构建流程的中文转述</footer>
</div>

<div class="article-byline">
<p>第三级 · C 语言编程 ｜ C Primer Plus 第16章 ｜ 2026-08-07</p>
</div>

## 为什么从链接与 Makefile 讲起

单文件程序用一条 `gcc hello.c` 就够了，但真实项目有成百上千个源文件。把它们组织起来、按依赖增量编译、打包成可复用的库——这套**构建（build）** 体系是 C 工程的日常。<span class="marginnote">《头文件、多文件程序与编译单元》讲过「分开编译、统一链接」；这一节回答剩下的问题：链接器到底做了什么、静态库与动态库有何区别、如何用一个 Makefile 自动管理编译。学会它，你就能读懂任何开源 C 项目的构建配置。</span>这一节讲清链接的两个阶段、两类库的差异，以及 Makefile 的核心语法。

## 1 链接器：把符号对号入座

回顾第 1 篇的编译流程：每个 `.c` 文件编译成 `.o` 目标文件后，链接器把它们与库拼装成可执行文件。链接器解决的核心问题是**符号解析（symbol resolution）** 与**重定位（relocation）**：

**符号解析**：把「调用 `printf`」与「`printf` 的实现」对上号。每个目标文件都有「定义了哪些符号」「引用了哪些符号」两张表。
**重定位**：把相对地址换算成真实的内存地址。`.o` 里的地址是「假设从 0 开始」的，合并进最终程序后要重新计算。

```bash
$ gcc -c math_utils.c -o math_utils.o   # 编译：产生目标文件
$ gcc main.c math_utils.o -o app        # 编译 + 链接
```

链接错误的根源（第 2 篇讲过）再归拢一次：

| 错误信息 | 含义 |
| --- | --- |
| `undefined reference to 'foo'` | 有人引用 `foo`，但所有目标文件/库里都没定义 |
| `multiple definition of 'foo'` | `foo` 被定义了多次（如两个 `.c` 都定义了同名全局变量） |

**链接时还会自动做一件重要的事**：搜索标准库。`gcc main.c` 会自动链接 libc（C 标准库），所以 `printf` 的定义不用你操心；但链接第三方库（如数学库 `libm`）需要显式加 `-lm`。

## 2 静态库：把代码打包进程序

**静态库（static library）** 把一组目标文件打包成一个档案文件，链接时其内容**直接复制**进可执行文件。创建与使用：

```bash
$ gcc -c math_utils.c -o math_utils.o
$ ar rcs libmathutils.a math_utils.o     # 打包成静态库
$ gcc main.c -L. -lmathutils -o app      # 链接：-L 指定目录，-l 指定库名
```

- `ar rcs` 是归档工具：`r` 插入、`c` 创建、`s` 生成索引。
- **库命名规则**：文件叫 `libmathutils.a`，链接时写 `-lmathutils`——`-l` 会自动补上 `lib` 前缀与 `.a` 后缀。
- `-L.` 告诉链接器在当前目录找库。

**静态库的缺点**：每个可执行文件都复制一份库代码，体积大；库更新后必须重新链接。

## 3 动态库：运行时加载共享

**动态库（shared library）** 不把代码复制进程序，而是让程序**运行时**从系统里加载它。一个动态库可被多个程序共享一份内存副本。

```bash
$ gcc -fPIC -c math_utils.c -o math_utils.o   # -fPIC：位置无关代码
$ gcc -shared -o libmathutils.so math_utils.o  # 生成 .so 动态库
$ gcc main.c -L. -lmathutils -o app           # 链接时记录依赖
$ export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH   # 运行时告诉系统去哪找
$ ./app
```

**静态库 vs 动态库**：

| 维度 | 静态库 `.a` | 动态库 `.so`/`.dylib` |
| --- | --- | --- |
| 链接时机 | 编译时复制进程序 | 运行时加载 |
| 可执行文件体积 | 大（含库代码） | 小 |
| 库更新 | 需重新链接 | 替换库文件即可（需 ABI 兼容） |
| 多程序共享 | 各自一份 | 内存共享一份 |
| 部署 | 单文件自足 | 需同时部署 `.so` |

`-fPIC`（Position Independent Code）让代码可以被加载到任意内存地址——动态库加载时地址不固定，必须用位置无关代码。<span class="marginnote">Windows 上对应的是静态库 `.lib` 与动态库 `.dll`；macOS 是 `.dylib`。动态库的「地狱」是依赖版本冲突：程序 A 需要 libX 1.0，程序 B 需要 1.2，系统只能装一个——这就是著名的 DLL hell。Linux 用 SONAME 版本机制缓解。</span>

## 4 Makefile：自动化的构建脚本

手动输入一长串 `gcc` 命令很痛苦，尤其只改了一个文件时。**`make`** 工具读 `Makefile`，只重编「依赖变化了的目标」：

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -g

app: main.o math_utils.o
	$(CC) $(CFLAGS) main.o math_utils.o -o app

main.o: main.c math_utils.h
	$(CC) $(CFLAGS) -c main.c

math_utils.o: math_utils.c math_utils.h
	$(CC) $(CFLAGS) -c math_utils.c

clean:
	rm -f *.o app
```

Makefile 的基本单元是**规则（rule）**：

```
目标: 依赖项
	命令
```

- **目标（target）**：要生成的文件（`main.o`、`app`）。
- **依赖（prerequisites）**：目标依赖的文件。依赖比目标新时，执行命令重建目标。
- **命令（recipe）**：以 **Tab 缩进**的 shell 命令——不能用空格，这是 make 的铁律。

`make` 的工作方式：从最终目标 `app` 开始，递归检查依赖；任一依赖比目标新，就重新生成。`make` 会建立一张**依赖图**，自动决定执行顺序。

- `make` 或 `make app`：构建最终程序。
- `make main.o`：只编某一个目标文件。
- `make clean`：删除中间产物——注意 `clean` 不是文件，是**伪目标（phony target）**，需声明 `clean:` 前加 `.PHONY: clean` 防冲突。

## 5 自动变量与常用模式

Makefile 里常用**自动变量**减少重复：

| 自动变量 | 含义 |
| --- | --- |
| `$@` | 当前目标名 |
| `$\lt ` | 第一个依赖 |
| `$^` | 所有依赖 |

用**模式规则（pattern rule）** 把「`.c` → `.o`」的规则写成一条：

```makefile
CC = gcc
CFLAGS = -Wall -g

%.o: %.c
	$(CC) $(CFLAGS) -c $\lt  -o $@

app: main.o math_utils.o
	$(CC) $(CFLAGS) $^ -o $@
```

`%.o: %.c` 匹配任意「同名 `.c` 编译成 `.o`」的规则。维护一个项目只需改「源文件清单」，规则自动套用。更现代的构建工具（CMake、Meson）是对 Makefile 的更高层封装，但**读懂 Makefile 仍是理解构建全貌的基本功**。

## 6 公式解析：`-lmathutils` 的库名展开

链接器的库名规则是「反直觉但必须记住」的一环：

$$
-l\text{name} \;\Rightarrow\; \text{lib}\text{name}.a \text{（静态）或 lib}\text{name}.so \text{（动态）}
$$

- **第一步，补前缀**：`-lmathutils` → 在名字前补 `lib`，得 `libmathutils`。
- **第二步，补后缀**：链接器在 `-L` 指定的目录里先找 `libmathutils.so`（动态优先），没有再找 `libmathutils.a`。
- **第三步，解析**：`-L.` 表示「当前目录」，于是它查找 `./libmathutils.so` / `./libmathutils.a`。

所以你的库文件**必须**命名为 `lib<名字>.a` 或 `lib<名字>.so`，`-l<名字>` 才能对上。自己创建的库名不按这个规范，链接器就找不到——这是新手最常见的「`undefined reference` 却明明有库」的坑。

## 7 小结

- 链接器做**符号解析**与**重定位**；`undefined reference` 与 `multiple definition` 是两类典型链接错误。
- 静态库 `ar rcs libx.a ...` 把代码复制进程序；动态库 `-shared -fPIC` 运行时共享加载。
- `-l名字` 自动展开为 `lib名字.a`/`.so`；`-L目录` 指定搜索目录。
- Makefile 规则 = 目标 + 依赖 + Tab 缩进的命令；`make` 按依赖图增量构建。
- 自动变量 `$@`/`$<`/`$^` 与模式规则 `%.o: %.c` 让构建脚本高度复用。

在下一节，我们进入第 5 篇，也是本专题的应用篇——用 C 实现真正的数据结构：**链表与基本数据结构实现**。
