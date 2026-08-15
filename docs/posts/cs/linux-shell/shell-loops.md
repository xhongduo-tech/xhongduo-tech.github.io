---
title: 更多结构化命令：for、while 与 until 循环
date: 2026-08-07
---

# 更多结构化命令：for、while 与 until 循环

<div class="epigraph">
<p>让计算机做它最擅长的事：重复一千次，而人类只思考一次。</p>
<footer>—— 循环存在的意义（Repetition is the point）</footer>
</div>

<div class="article-byline">
<p>第三级 · Linux 命令行与 Shell 脚本 ｜ Blum《Shell 脚本编程大全》 第13章 ｜ 2026-08-07</p>
</div>

## 为什么从循环开始

有了分支，脚本能「看情况走」；但真正拉开脚本与手动操作差距的，是**重复**——备份 10 个目录、给 50 个用户发信、检查 100 个端口。手动做这些是体力活，而循环让脚本把「一个」的规则放大到「所有」。Shell 提供三种循环：**`for`**（已知集合，逐个处理）、**`while`**（条件为真就一直做）、**`until`**（条件为假才一直做）。掌握它们，是脚本从「一次性工具」走向「规模化工具」的分水岭。

## 1 for 循环：遍历一个集合

**`for`** 循环遍历一个列表，每轮把列表中的一个元素赋给循环变量：

```bash
for name in alice bob carol; do
    echo "Hello, $name"
done
```

三要素：`for 变量 in 列表`、`do`、`done`。列表的写法很灵活：

```bash
for file in *.log; do            # 通配符展开成文件列表
    echo "处理 $file"
done

for num in {1..5}; do            # 花括号展开 1 到 5
    echo "$num"
done

for line in $(cat list.txt); do  # 命令替换：把文件内容当列表
    echo "$line"
done
```

**易错点**：`for line in $(cat list.txt)` 按**空格**切分列表——文件里含空格的行会被拆开。要按行处理，用 `while read` 更稳（见下节）。另外 `{1..5}` 是 bash 的花括号展开，`{1..100}` 写 100 也没问题，但不能是变量：`for i in {$a..$b}` 不生效。
<span class="marginnote">`for` 不写 `in 列表` 时默认遍历命令行参数：`for arg` 等价 `for arg in "$@"`。脚本里写 `for arg in "$@"` 更清晰。列表元素带空格时，`in "a b" c` 中引号内的 `a b` 是一个整体。</span>

## 2 C 风格 for：当计数器存在时

bash 的 `for` 还有一套 C 语言风格，适合「明确的起止与步长」：

```bash
for (( i = 1; i <= 10; i++ )); do
    echo "第 $i 次"
done
```

**公式解析：C 风格 for 的三段式**

$$
\text{for} \;( \underbrace{i=1}_{\text{初始化}} ; \underbrace{i \le 10}_{\text{循环条件}} ; \underbrace{i++}_{\text{每轮更新}} ) \; \text{do} \cdots \text{done}
$$

拆解：

- **第一步，初始化**：进入循环前执行一次，把计数器设为初值。
- **第二步，条件判断**：每轮开头检查条件，为真才进入循环体。
- **第三步，更新**：每轮结束执行更新（`i++` 自增 1）。
- **第四步**：回到第二步，直到条件为假——计数循环的骨架。

`i++` 等价 `i=i+1`；`i+=2` 是步长为 2。这套写法适合「明确知道要循环多少次」的场景，比如给数组的每个下标赋值。

## 3 while 与 until：条件驱动循环

**`while`** 在条件**为真**时反复执行；**`until`** 在条件**为假**时反复执行。两者互为镜像：

**核心对比表：while 与 until**

| 循环 | 何时进入循环体 | 何时退出 |
| --- | --- | --- |
| `while 条件` | 条件为真 | 条件变假 |
| `until 条件` | 条件为假 | 条件变真 |

`while` 读文件的经典用法（正确处理含空格的行）：

```bash
while IFS= read -r line; do
    echo "行: $line"
done < list.txt
```

`< list.txt` 把文件重定向为 `read` 的输入，`read` 每次读一行、直到文件读完（read 返回非 0），循环自然结束。<span class="marginnote">`IFS=` 让 read 不按空白切分行、`-r` 保留行内反斜杠——这是「按行处理文件」最稳妥的写法。相比之下 `for line in $(cat f)` 遇到空格就裂，运维脚本里这是高频 Bug。</span>

`while` 做「直到成功」的等待：

```bash
while ! nc -z localhost 8080; do
    echo "等待服务就绪..."
    sleep 2
done
```

服务没起时 `nc` 失败、条件为真，循环一直等；服务一起来，`!` 取反使条件为假，循环退出。

**易错点**：`while` 循环最怕**死循环**——条件永远为真。三个保命手段：循环体内一定有改变条件的语句（`i++`、消费输入）、加 `sleep` 降频、必要时用 `timeout` 包裹或 `Ctrl-C` 中断。测试时先想清楚「这个条件怎么变假」。

## 4 控制循环：break 与 continue

**`break`** 立即跳出整个循环，**`continue`** 跳过本轮、进入下一轮：

```bash
for i in {1..10}; do
    if [ "$i" -eq 5 ]; then
        continue        # 跳过 5，继续 6
    fi
    if [ "$i" -eq 8 ]; then
        break           # 到 8 就彻底停
    fi
    echo "$i"
done
```

输出会是 `1 2 3 4 6 7`。`break 2` 可跳出**两层**嵌套循环，`continue 2` 同理作用于外层。<span class="marginnote">嵌套循环里，`break` 默认只跳出最内层。要在两层 for 里「找到就全停」，`break 2` 是简洁答案；想更可控就用函数 + return（见第 19 章函数）。</span>

**易错点**：`continue` 与 `break` 最容易与 `case` 的 `;;` 记混——`case` 里 `;;` 只是结束一个分支，不是循环控制。循环与 case 都常出现在「处理参数的脚本」里，写着写着容易串。

## 5 实战：批量巡检脚本

把三种循环拼起来，写一个「批量检查端口是否开放」的脚本：

```bash
#!/bin/bash
# 用法: ./check_ports.sh 8080 9090 10000

for port in "$@"; do
    if nc -z -w 2 localhost "$port"; then
        echo "端口 $port: 开放"
    else
        echo "端口 $port: 未开放"
    fi
done
```

`"$@"` 拿到所有命令行参数当列表，`for` 逐个检查，`if` 判结果——**一个 for 配一个 if，就覆盖了「对每个对象做判断」的绝大多数场景**。再加一层 `while` 做「重试直到通过」，几乎能应对任何巡检需求。

## 6 小结

- **`for` 遍历集合**：`in` 列表、通配符、花括号展开、命令替换都能生成列表。
- **C 风格 `for (( ;; ))`**：初始化 + 条件 + 更新三段式，适合明确的计数循环。
- **`while` 为真才跑、`until` 为假才跑**：读文件用 `while IFS= read -r`，最稳。
- **`break` 跳出、`continue` 跳过**：`break 2` 跳出两层，测试时防死循环。
- **`for` + `if` 覆盖大半巡检**：逐个对象做判断，是脚本最常用的组合。

在下一节，我们让脚本接收外部输入——**处理用户输入**：read、位置参数与命令行选项。
