---
title: 构建基础脚本：变量、命令替换与数学运算
date: 2026-08-07
---

# 构建基础脚本：变量、命令替换与数学运算

<div class="epigraph">
<p>命令行让你手动做事，脚本让命令替你做事。</p>
<footer>—— Shell 脚本的初心（Automation first）</footer>
</div>

<div class="article-byline">
<p>第三级 · Linux 命令行与 Shell 脚本 ｜ Blum《Shell 脚本编程大全》 第11章 ｜ 2026-08-07</p>
</div>

## 为什么从基础脚本开始

前两篇批我们用命令行「手动」完成一件事。但运维与开发的分水岭在于：**同样的事，第二次做就该交给脚本**。Shell 脚本是把一连串命令、判断与循环写进一个文件，让 bash 逐行执行。本章是第一篇真正的编程课：从「写一个能跑的最小脚本」开始，掌握三个最核心的构件——**变量**（存东西）、**命令替换**（取别的命令的输出）、**数学运算**（算东西）。这三样拼起来，你已经能写出「带逻辑」的自动化脚本；后续的结构化命令、循环、函数，全是建立在这三块地基之上。

## 1 第一个脚本：shebang 与执行

任何 Shell 脚本的第一行几乎都是这一行：

```bash
#!/bin/bash
```

这一行叫 **shebang**（发音 sh-bang）——`#!` 告诉内核「用哪个解释器来跑这个文件」。`#!/bin/bash` 表示「请用 /bin/bash 解释下面所有行」。<span class="marginnote">shebang 的名称由来：`#` 读作 sharp、`!` 读作 bang，合起来就是 she-bang。写 `#!/usr/bin/env bash` 也是常见写法——它让系统在 PATH 里找 bash，兼容各种安装路径（如 macOS 的 `/opt/homebrew/bin/bash`）。</span>

写一个最小脚本：

```bash
#!/bin/bash
echo "Hello, Linux!"
echo "今天是 $(date +%F)"
```

执行有两种方式：

```bash
bash hello.sh        # 用 bash 直接解释，无需执行权限
chmod +x hello.sh && ./hello.sh   # 先加执行权限，再直接运行
```

**易错点**：`./hello.sh` 的 `./` 不能省。`. hello.sh`（一个点）表示「用当前 shell 执行」，`./hello.sh`（点斜杠）表示「作为程序执行」——前者会污染当前 shell 环境，后者是规范做法。脚本没有 `chmod +x` 就运行会得到 `Permission denied`。

## 2 变量：给数据起名字

**变量（variable）** 是脚本里存放数据的名字。定义与使用：

```bash
name="world"              # 赋值：等号两边不能有空格！
echo "Hello, $name"       # 使用：$ 前缀取值
echo "Hello, ${name}!"    # 用花括号隔开边界
```

**赋值规则**：`=` 两边**绝不能有空格**——`name = "world"` 会被 bash 解释成「执行 name 命令并传两个参数」。变量名由字母、数字、下划线组成，**不能以数字开头**。

**变量种类**：脚本内自定义的是**用户变量**；bash 启动时从环境里继承的（`PATH`、`HOME`、`USER`）叫**环境变量**，脚本内也能直接读。`$0` 是脚本名、`$1` `$2` 是位置参数——下一章《处理用户输入》专门展开。

**易错点**：`$` 后面的花括号 `{}` 用来**划清变量名的边界**。`echo "$name_file"` 会把 `name_file` 当整个变量名（多半为空）；`echo "${name}_file"` 才能输出 `world_file`。
<span class="marginnote">变量大小写敏感：`$Name` 与 `$name` 是两个变量。惯例是用户变量用小写、环境变量用大写。`. env.sh` 之后脚本里定义的变量会留在当前 shell，所以脚本内定义的变量别指望「脚本跑完还能用」——除非你用 `export` 导出到环境。</span>

## 3 公式解析：命令替换

**命令替换（command substitution）** 把一条命令的输出**当作值**赋给变量——这是脚本连接「命令世界」与「数据世界」的桥梁：

$$
\text{变量} = \$( \text{命令} ) \quad \Longrightarrow \quad \text{命令的输出成为变量的值}
$$

写法有两种，推荐第一种：

```bash
today=$(date +%Y-%m-%d)        # 新式：$(...)
users=$(wc -l </etc/passwd)   # 命令的输出作为值
count=`grep -c error app.log`  # 旧式：反引号
```

拆解 <code>count=&#36;(grep -c error app.log)</code>：

- **第一步**：bash 先执行括号里的 `grep -c error app.log`，得到输出（一个数字）。
- **第二步**：这个数字替换掉整个 <code>&#36;(...)</code> 表达式，成为变量 `count` 的值。
- **第三步**：之后 `echo "$count"` 就能打印它。

命令替换在脚本里无处不在：拿当前时间、读命令输出、把动态值拼进消息。**反引号 `` ` `` 与 <code>&#36;(...)</code> 等价，但反引号里嵌套反引号时需要转义，可读性差**——新脚本一律用 <code>&#36;(...)</code>。
<span class="marginnote">命令替换执行的是<strong>子 shell</strong>：<code>&#36;(cd /tmp && pwd)</code> 里 `cd` 只影响子 shell，不会改变当前脚本的工作目录。想在同一 shell 里改状态，用 `source` 而不是命令替换。</span>

## 4 公式解析：数学运算

bash 里做数学有三种姿势，但正统只有一种。**双括号算术展开 <code>&#36;(( ))</code>** 是 POSIX 标准推荐：

$$
\$(( \;\text{算术表达式}\; )) \quad \Longrightarrow \quad \text{计算出数值并替换}
$$

```bash
a=7
b=3
echo $(( a + b ))       # 10
echo $(( a * b ))       # 21
echo $(( a % b ))       # 1，取余
echo $(( (a + b) * 2 )) # 20，支持嵌套括号
c=$(( a ** 2 ))         # 49，乘方（部分 bash 版本）
```

拆解这条规则：

- **第一步**：<code>&#36;(( ))</code> 里的内容被当作**整数算术**求值，`a`、`b` 自动被替换成它们的值。
- **第二步**：支持的运算符包括 `+ - * / % **`，以及 `+=`、`++` 等简写。
- **第三步**：整个表达式的结果替换 <code>&#36;(( ))</code>，可赋给变量或直接 echo。

**易错点**：<code>&#36;(( ))</code> 里写 `a + b` 不需要 `$` 前缀——bash 自动取值。而**只有整数运算**：<code>echo &#36;(( 7 / 2 ))</code> 得 3，不会得 3.5。需要小数运算，得用 `bc` 或 `awk`。另一个经典坑：<code>a = &#36;(( a + 1 ))</code> 两边不能有空格，且 `=` 前不能有 `$`。<span class="marginnote">旧式算术 `expr`（`expr 7 + 3`）需要手动传空格、参数还要转义，早已被 <code>&#36;(( ))</code> 取代。`let` 命令也是老写法。现代 bash 脚本统一用 <code>&#36;(( ))</code>——简洁、无转义、纯整数。</span>

## 5 把三样拼起来：一个真实小脚本

把变量、命令替换、算术组合起来，写一个「统计日志里 503 数量并报告」的脚本：

```bash
#!/bin/bash
# 统计 access.log 中 503 的次数并输出报告

log="/var/log/nginx/access.log"
total=$(wc -l <"$log")
errors=$(grep -c " 503 " "$log")
ratio=$(( errors * 100 / total ))

echo "总请求: $total"
echo "503 次数: $errors"
echo "503 占比: ${ratio}%"
```

这个脚本把本章三件套全用上了：`log` 变量存路径、<code>&#36;(...)</code> 两次取命令输出、<code>&#36;(( ))</code> 算占比。加上一句 `chmod +x report.sh` 就能跑——**第一次体验「把排查步骤变成可复用工具」**，正是脚本存在的全部意义。<span class="marginnote">写脚本的黄金习惯：变量名有意义、顶部加注释说明用途、所有外部命令写绝对路径或先 `export PATH`。这样换一台机器、换一个用户，脚本依然行为一致。</span>

## 6 小结

- **shebang 声明解释器**：`#!/bin/bash` 第一行，`chmod +x` 后可直接执行。
- **变量 = 名字存值**：`=` 两边无空格、`$var` 取值、`${var}` 划清边界。
- **命令替换 <code>&#36;(...)</code>**：把命令输出变成值，新脚本一律用它而非反引号。
- **算术 <code>&#36;(( ))</code>**：整数四则与取余，<code>&#36;(( 7 / 2 ))</code> 得 3，小数要用 bc/awk。
- **脚本 = 变量 + 替换 + 算术 + 逻辑**：从最小可跑脚本开始，逐步拼出自动化工具。

在下一节，我们给脚本装上「判断力」——**使用结构化命令**：if-then、test 与 case。
