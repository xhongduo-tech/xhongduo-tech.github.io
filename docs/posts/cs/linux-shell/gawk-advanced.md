---
title: gawk 进阶：字段、数组与自定义函数
date: 2026-08-07
---

# gawk 进阶：字段、数组与自定义函数

<div class="epigraph">
<p>awk 不是又一个文本工具，它是一整套按字段思考的数据语言。</p>
<footer>—— awk 的设计者之一，阿尔弗雷德 · 艾侯（Alfred Aho）</footer>
</div>

<div class="article-byline">
<p>第三级 · Linux 命令行与 Shell 脚本 ｜ Blum《Shell 脚本编程大全》 第22章 ｜ 2026-08-07</p>
</div>

## 为什么从 gawk 进阶开始

初识 awk 时我们把它当「字段提取器」：`gawk '{print $1}'`。但 awk 的真实身份是**一门完整的小语言**——它有关联数组、条件、循环、字符串函数、自定义函数。这意味着：统计每个 IP 的次数、按字段排序输出、跨文件累积、生成格式化报表，都能在 awk 内部一步完成，不必反复借用 sort/uniq。本章把 awk 从「会用」推到「会写程序」：**字段控制**（FS/OFS）、**关联数组**（awk 最锋利的武器）、**内建函数**、**自定义函数**与 **BEGINFILE** 等进阶结构。

## 1 字段控制的完整图景

awk 默认用**连续空白**当字段分隔符（FS），用**空格**当输出分隔符（OFS）。但真实数据是冒号分隔的 passwd、逗号分隔的 CSV、制表符分隔的 TSV——字段控制是 awk 的第一门必修：

```bash
gawk -F: '{print $1, $3}' /etc/passwd        # 冒号分隔，取用户名与 UID
gawk -F, '{print $1}' data.csv               # CSV
gawk 'BEGIN { FS=":"; OFS="|" } {print $1, $2}' file   # 程序内设置
```

**公式解析：FS 与 OFS 的对称性**

$$
\text{输入行} = \$1 \; \underbrace{\text{FS}}_{\text{输入分隔符}} \; \$2 \; \cdots \qquad
\text{输出} = \$1 \; \underbrace{\text{OFS}}_{\text{输出分隔符}} \; \$2 \; \cdots
$$

拆解：

- **第一步**：awk 按 **FS** 把输入行切开成字段——`-F:` 等价 `BEGIN { FS=":" }`。
- **第二步**：`print $1, $2` 里的逗号输出时会被 **OFS** 连接（默认空格）。
- **第三步**：`BEGIN { FS=":"; OFS="|" }` 同时设好两个，`print` 自动用 `|` 连接。
- **第四步**：FS 也可以是正则：`-F'[ \t,]+'` 一次处理多种分隔符。

**易错点**：`print $1 $2`（无逗号）是**拼接**，`print $1, $2`（有逗号）才按 OFS 分隔。字段数 `NF`、整行 `$0`、行号 `NR`、文件号 `FNR`——`NR` 跨文件累计，`FNR` 每个文件从 1 起，多文件处理时极易混。

## 2 关联数组：awk 最锋利的武器

awk 的数组是**关联数组**——下标不必是数字，可以是任意字符串。这让「按 key 聚合」变得极其自然：

```bash
gawk '{ count[$1]++ } END { for (ip in count) print ip, count[ip] }' access.log
```

**公式解析：count[key]++ 的聚合模型**

$$
\text{count}[\, \underbrace{\$1}_{\text{key（如 IP）}}\, ] = \text{count}[\$1] + 1 \qquad \Longrightarrow \qquad \text{每个 key 一个计数器}
$$

拆解：

- **第一步**：`count[$1]++` 以第 1 字段（如 IP）为下标，每出现一次就把该下标的值加一。
- **第二步**：awk 数组是「字典」——`count["1.2.3.4"]` 与 `count["5.6.7.8"]` 是各自独立的计数器。
- **第三步**：`END` 块里 `for (key in count)` 遍历所有键，打印 `IP 次数`。
- **第四步**：整条命令在 awk 内部完成「分组计数」，无需 sort/uniq 管线。

再进一步：统计「状态码分布」只需把下标换成 `$9`：

```bash
gawk '{ code[$9]++ } END { for (c in code) print c, code[c] }' access.log
```

**易错点**：awk 数组**没有顺序**——`for (key in arr)` 的遍历顺序不保证与插入一致。想按次数排序输出，在 END 里二次处理或仍交给 `sort -rn` 收尾。另一个坑：`arr[$1]` 访问不存在的键会**自动创建**它——判断键是否存在用 `if ($1 in arr)`，而不是 `if (arr[$1])`。<span class="marginnote">`delete arr[key]` 删除单个键、`delete arr` 清空数组。awk 数组也能当「集合」用：`if (!seen[$1]++) print $1` 只输出每个键第一次出现的行——天然去重。</span>

## 3 模式与结构：比逐行更精细

awk 的主体块可以用**模式（pattern）** 限定「哪些行才执行」：

```bash
gawk '/^ERROR/ {print NR": "$0}' app.log        # 只处理 ERROR 行
gawk '$3 > 1000 {print $1, $3}' data.txt        # 只处理第 3 字段大于 1000 的行
gawk 'NR >= 2 && NR <= 10 {print $1}' file      # 行号范围
```

模式可以是正则、比较表达式、逻辑组合。`BEGIN`/`END` 之外的第三种特殊块是 **`BEGINFILE`**（gawk 扩展）：在**每个输入文件**处理前执行一次，多文件时做「每文件独立初始化」：

```bash
gawk 'BEGINFILE { print "== 处理文件", FILENAME } { total += $1 } ENDFILE { print "小计:", total; total=0 }' a.txt b.txt
```

`BEGINFILE`/`ENDFILE` 让「每个文件一个报告」在 awk 内部完成。<span class="marginnote">`next` 跳过当前行进入下一行、`exit` 结束所有处理。`getline` 手动读下一行（高级，易出错）。模式里的 `!` 取反：`!/^#/` 跳过注释行。</span>

## 4 内建函数与自定义函数

awk 自带一箱子字符串与数学函数：

| 函数 | 作用 |
| --- | --- |
| `length(s)` | 字符串长度 |
| `substr(s, m, n)` | 取子串 |
| `index(s, t)` | 找子串位置 |
| `split(s, arr, fs)` | 按分隔符拆成数组 |
| `sprintf(fmt, ...)` | 格式化字符串 |
| `int(x)`、`sqrt(x)`、`rand()` | 数学函数 |

**自定义函数**用 `function` 定义，写在程序任何位置：

```bash
gawk '
function percent(part, whole) {
    return (whole > 0) ? (part * 100 / whole) : 0
}
{ err += ($9 ~ /^5/) }      # 5xx 记一次
{ tot++ }
END { printf "5xx 占比: %.2f%%\n", percent(err, tot) }
' access.log
```

拆解这个例子：

**第一步**：`function percent(...)` 定义返回百分比的函数，带默认保护（whole 为 0 时返回 0）。
**第二步**：主体块里 `err` 统计 5xx 次数、`tot` 统计总请求数。
- **第三步**：END 里 `printf` 用 `%.2f` 保留两位小数，调用自定义函数完成计算。

**易错点**：awk 函数的参数是**值传递**，但数组参数是**引用传递**（函数内改数组会生效）。多返回值可以靠全局变量或传入数组让函数填充。函数定义要放在使用之前，或至少与调用在同一个程序文本里。<span class="marginnote">awk 的字符串比较按字典序：`"100" \lt  "9"` 为真——排序数值型字段要么先 `+0` 转数字，要么用 `sort -n`。awk 脚本变复杂后可以存成 `.awk` 文件，用 `gawk -f script.awk data` 执行。</span>

## 5 小结

- **FS 切输入、OFS 连输出**：`-F:` 与 `BEGIN { OFS="|" }` 成对使用；`NF`/`NR`/`FNR` 分清累计与独立。
- **关联数组是 awk 的杀手锏**：`count[$