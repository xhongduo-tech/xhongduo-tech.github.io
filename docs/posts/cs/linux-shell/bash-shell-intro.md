---
title: 认识 BASH：Shell 类型、命令别名与历史命令
date: 2026-08-07
---

# 认识 BASH：Shell 类型、命令别名与历史命令

<div class="epigraph">
<p>Shell 是用户与内核之间的那个「壳」：你敲下的每一行，都由它翻译给系统。</p>
<footer>—— Unix 系统设计的基本分层（Shell 词源）</footer>
</div>

<div class="article-byline">
<p>第三级 · Linux 命令行与 Shell 脚本 ｜ 鸟哥《Linux私房菜》 第10章 ｜ 2026-08-07</p>
</div>

## 为什么从认识 BASH 开始

前几节我们一直在「用命令」，本节抬头看清「命令是谁在解释」。你在终端里敲下 `ls -l`，真正接手这一行的不是内核，而是一个叫 **Shell** 的程序——它负责把命令拆词、按规则展开、找到可执行文件、拉起进程、传递结果。绝大多数 Linux 的默认 Shell 是 **bash（Bourne Again SHell）**。理解 bash 的命令处理流程，是后面所有脚本写作的地基：别名、历史、通配符、变量、命令替换，全是同一台机器上的不同齿轮。今天先装好三颗最常用的：**命令查找顺序、别名、历史**。

## 1 Shell 是什么：壳与内核的分层

**Shell（壳）** 是操作系统中「用户与内核之间」的那层命令解释器。你看到的终端窗口只是门面，真正干活的是：

```
用户键盘 → 终端模拟器 → Shell（bash）→ 内核（Kernel）→ 硬件
```

内核只认系统调用，不认 `ls -l`；把 `ls -l` 翻译成「找 `ls` 这个程序并执行它」的，正是 Shell。<span class="marginnote">Linux 上有多种 Shell：`sh`（Bourne Shell，最古老）、`bash`（默认，sh 的超集）、`zsh`（macOS 默认）、`fish`（对新手友好）。`echo $SHELL` 看当前登录 Shell，`cat /etc/shells` 看系统装了哪些。</span>

**bash 与 sh 的关系**：bash 是 POSIX 兼容的 sh 的超集，绝大多数 `sh` 脚本在 bash 下也能跑，但 bash 扩展了数组、`[[ ]]`、`$(( ))` 等特性。写脚本时若用 `#!/bin/bash` 声明，就能放心用这些增强。

**shell 的四种分类**是判断「哪些启动文件会生效」的关键：

| 维度 | 类型 | 例子 |
| --- | --- | --- |
| 是否登录 | 登录 shell / 非登录 shell | ssh 登录 / 登录后再开终端 |
| 是否交互 | 交互式 / 非交互式 | 人敲命令 / 脚本执行 |

这个分类矩阵决定了 bash 读取 `~/.bashrc`、`~/.bash_profile` 等配置文件的顺序——我们到第 4 篇《环境变量与 Bash 启动文件》专门拆解，今天先记住结论：**登录 shell 读登录配置，非登录交互 shell 读 `~/.bashrc`**。

## 2 命令执行顺序：alias 排在很前面

bash 拿到一行命令后，**按固定顺序**决定「这个命令是什么」：

**公式解析：bash 的命令解析优先级**

$$
\text{alias} \;\rightarrow\; \text{关键字} \;\rightarrow\; \text{函数} \;\rightarrow\; \text{内建命令} \;\rightarrow\; \text{PATH 中的外部程序}
$$

逐项拆解：

- **第一步，别名（alias）**：用户自定义的快捷键，优先级最高。
- **第二步，关键字**：`if`、`for`、`while`、`case` 等 shell 语法词。
- **第三步，函数**：`function` 定义的 shell 函数。
- **第四步，内建命令（builtin）**：`cd`、`echo`、`pwd` 这类 shell 自己实现的命令，不启动外部进程。
- **第五步，外部程序**：在 `PATH` 目录里找同名可执行文件。

这个顺序解释了很多「诡异」现象：你给 `ls` 起了个别名，它就永远先于系统的 `ls` 生效；你定义了一个叫 `cd` 的函数，就能覆盖内建 `cd`。用 `type 命令名` 可以查看一条命令到底属于哪一类：<span class="marginnote">`type -a ls` 会列出所有可能的匹配（别名、内建、路径），`which ls` 只查 PATH 中的外部程序。想知道一条命令是不是内建，`type cd` 会回答「cd is a shell builtin」。</span>

```bash
type ls        # ls is aliased to `ls --color=auto'
type cd        # cd is a shell builtin
type -a echo   # echo 可能是内建，也是 /usr/bin/echo
```

**易错点**：别以为 `echo` 只有一种。多数系统里 bash 的内建 `echo` 优先于 `/bin/echo`，两者选项略有差异。跨脚本时若行为不一致，用 `command echo` 或绝对路径强制调用外部版本。

## 3 别名：给常用命令起小名

**别名（alias）** 是命令的快捷方式，定义与查看都极简单：

```bash
alias ll='ls -alF'          # 定义别名
alias grep='grep --color'   # 给 grep 加默认高亮
unalias ll                  # 删除别名
alias                       # 不带参数，列出全部别名
```

别名只对**交互式 shell** 有意义，脚本里默认不展开别名（脚本要可移植）。别名也不递归：`alias ls='ls --color'` 里的右侧 `ls` 用的是**原始** ls，不会无限套娃。

**易错点**：别名的定义只在当前 shell 会话生效，关掉终端就没了。想让别名永久生效，写进 `~/.bashrc`。而 `~/.bashrc` 里常见的 `alias rm='rm -i'` 这类「保命别名」，在脚本或别人的机器上并不存在——**别依赖别名保护自己，习惯本身才可靠**。<span class="marginnote">用单引号还是双引号定义别名有讲究：单引号在定义时不展开变量，双引号会立即展开。`alias lx="ls $PWD"` 会绑定当前的 $PWD，而单引号版本每次执行时才取 $PWD。</span>

## 4 历史命令：bash 的记忆

bash 会把敲过的命令记进历史文件（默认 `~/.bash_history`），支持快速重放与搜索：

| 操作 | 按键/命令 | 含义 |
| --- | --- | --- |
| 上一条 | `↑` / `!!` | 重跑上一条命令 |
| 上上条 | `↑↑` / `!-2` | 重跑上上条 |
| 关键字搜索 | `ctrl-r` | 增量反向搜索历史 |
| 查看历史 | `history` | 列出带编号的历史 |
| 清空历史 | `history -c` | 清除当前会话历史 |

`ctrl-r` 是历史功能里最值得练的一个：按 `ctrl-r` 后输入几个字母，bash 实时匹配你敲过的最接近的命令，再按 `ctrl-r` 继续往前翻。<span class="marginnote">`!$` 表示上一条命令的最后一个参数——`mkdir newdir` 之后接 `cd !$` 就是「进入刚创建的目录」。这类历史扩展缩写是 bash 的高阶玩具，`shopt -s histverify` 可以让你在真正执行前先看到展开结果，避免误执行。</span>

**易错点**：历史文件默认是「退出时才写入」的，多个终端同时开时可能互相覆盖。生产服务器排查问题时，`history` 还能看别人敲过什么——但真正重要的是记住：**不要把密码、token 直接敲在命令行里**，它们会明文躺在 `~/.bash_history` 里，被下一个 `grep` 人轻易翻出。

## 5 命令行编辑：Readline 快捷键

bash 的行编辑由 **Readline** 库实现，意味着你在命令行里其实在一个**微型编辑器**中工作——支持光标移动、剪切粘贴、撤销，只是默认是 emacs 风格按键。以下五个快捷键值得最先掌握：

```text
ctrl-a / ctrl-e    光标跳到行首 / 行尾
ctrl-w            删除光标前一个单词
ctrl-u            删除从光标到行首
ctrl-k            删除从光标到行尾
ctrl-y            粘贴刚删除的内容
```

配合历史使用威力更大：敲了一半发现前面写错，`ctrl-a` 回行首改掉，再 `ctrl-e` 回行尾继续；想重敲一条历史命令，`ctrl-u` 清空当前行，再 `ctrl-r` 搜索。这一套组合下来，**在命令行里「改」的效率会超过「删了重敲」**。

**易错点**：`ctrl-w` 删除的是「以空白分隔的单词」，`a_b_c` 会被整体删掉，因为它没有空格。想要按路径段删除，可用 `alt-backspace`（在终端里有时要配置）。这些快捷键全部由 Readline 提供，`bind -P` 可以列出所有当前绑定。

## 6 小结

- **Shell 是命令解释器**：bash 是 Linux 默认 Shell，是 sh 的超集，`echo $SHELL` 可查。
- **命令解析有固定顺序**：别名 → 关键字 → 函数 → 内建 → PATH 外部程序，`type` 可诊断。
- **别名是会话级快捷方式**：`alias ll='ls -alF'`，写进 `~/.bashrc` 才永久生效。
- **历史命令 `ctrl-r` 增量搜索**，`!!` 重跑上一条、`!$` 取上一条最后的参数。
- **别让密码进历史**：`~/.bash_history` 是明文，敏感信息要避免直接敲在命令行。

在下一节，我们把「过滤文本」升级成「匹配文本的模式语言」——**正则表达式与文件格式化处理**，它同时是 grep、sed、vim 替换背后的统一语法。
