---
title: 字符串处理与格式化
date: 2026-08-07
---

# 字符串处理与格式化

<div class="epigraph">
<p>文本是数据的常态，字符串处理是数据科学的地基。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 官方 Python 教程 第7章 ｜ 2026-08-07</p>
</div>

## 为什么从字符串开始

从第一行 `print("Hello")` 起，字符串就无处不在：读进来的数据是字符串、网页返回的是字符串、用户输入的是字符串。官方 Python 教程第 7 章（输入与输出）把字符串格式化系统化，而它的**处理**能力则散布在 `str` 的四十多个方法里。

本节把两件事讲透：**字符串处理工具箱**（分割、拼接、清洗）与**格式化**（把数据排成想要的样子）。文本清洗——CSV 解析、日志分析、网页抓取——九成工作就是这两件事的组合。

## 1 字符串方法工具箱：split、join 与 strip

字符串**不可变**，所以「修改」方法都返回新字符串。最常用的三件套：

```python
text = "  hello, python, world  "

print(text.strip())                    # 'hello, python, world'，去首尾空白
print(text.split(","))                 # ['  hello', ' python', ' world  ']，按分隔符切
print("-".join(["a", "b", "c"]))       # 'a-b-c'，把列表用分隔符拼起来
```

**重点：`split` 切、`join` 拼、`strip` 洗。** 三者是文本处理的核心动作。`split()`（无参数）按任意空白切，`split(",")` 按指定分隔符切，`splitlines()` 按行切。<span class="marginnote">`strip()` 家族还有 `lstrip()` 与 `rstrip()`（只去左/右）。处理带 `\n` 的文本时，`rstrip()` 去掉行尾换行几乎成了标准动作——这正是《输入输出与文件读写》里逐行读文件后的标配。</span>

其他常用方法：

```python
print("Hello".upper().lower())         # 'hello'，链式调用
print("abc123".isalpha())              # False，含数字
print("data.csv".endswith(".csv"))     # True，后缀判断
print("abc".replace("b", "X"))         # 'aXc'
print(text.startswith("hello"))        # 前缀判断
```

**判断类**方法（`isalpha`、`isdigit`、`startswith`、`endswith`）返回布尔值，常与 `if` 组合做「文本校验」。

## 2 三种格式化方式：%、format 与 f-string

把数据「填」进字符串模板，Python 历史上经历了三代写法：

```python
name, score = "alice", 95.5

# 1. % 格式化（旧式，源自 C 语言）
print("%s 的分数是 %.1f" % (name, score))

# 2. str.format()（Python 3 中坚）
print("{} 的分数是 {:.1f}".format(name, score))

# 3. f-string（Python 3.6 起，推荐）
print(f"{name} 的分数是 {score:.1f}")
```

**重点：优先 f-string。** 它把模板与变量写在**同一处**，可读性最好，执行也最快。`%` 式是历史遗留，读旧代码时认识即可；`str.format` 用于模板在运行时才确定（如配置文件）的场景。<span class="marginnote">f-string 在 `{}` 里能放任意表达式：`f"{len(items)} 项"`、`f"{x!r}"`（`!r` 用 `repr` 显示）。3.12 起还支持嵌套引号，`f"{'大' if ok else '小'}"` 这类条件表达式可以直接写。</span>

**辨析｜易错点：** f-string 里的 `{}` 是插值符号，若字符串本身要显示花括号（如 JSON 片段、字典字面量），要写 <code>&#123;&#123;</code> 与 <code>&#125;&#125;</code> 转义。这是「模板引擎」类工具共有的转义规则。

## 3 格式化迷你语言：对齐、宽度与精度

f-string 的 `{}` 内部有一套**格式说明符**，用于控制对齐、宽度、小数位：

```python
price = 1234.5

print(f"{price:10.2f}")     # '   1234.50'，宽 10、右对齐、两位小数
print(f"{'left':<10}|")     # 'left      |'，< 左对齐
print(f"{'right':>10}")     # '     right'，> 右对齐
print(f"{'center':^10}")    # '  center  '，^ 居中
print(f"{price:,.2f}")      # '1,234.50'，逗号千分位
```

**重点：`{值:宽.精度f}` 是数字排版的固定句式。** 报表、对齐的表格、日志里把数字「钉」在固定列宽，靠的就是这套说明符。<span class="marginnote">日期与十六进制也有对应：`f"{x:#x}"` 输出 `0x...`，`f"{x:08b}"` 输出 8 位二进制补零。对齐宽度在《数据可视化》专题做坐标轴刻度标签时还会重逢——把数字格式化成统一宽度的文本，是图表「不挤作一团」的前提。</span>

## 4 公式解析：格式说明符的语法

**f-string 插值字段是一个小型公式。**

$$
\{ \text{值} \; [\text{!转换}] \; [:\text{填充对齐}[宽度][.精度][类型]] \}
$$

对这条式子做四步拆解：

- **第一步，读结构**：`{name!r:>10.2f}` 拆成三块——值 `name`、可选转换 `!r`、格式 `:>10.2f`。
- **第二步，读对齐**：`>` 右对齐、`<` 左对齐、`^` 居中，后接**宽度** `10`，表示结果至少占 10 个字符宽度，不足补空格。
- **第三步，读精度**：`.` 后是精度 `2`，对浮点表示小数位数，对字符串表示截断长度；`f` 是类型码（float 定点表示）。`{price:10.2f}` 意思是「占 10 宽、保留 2 位小数」。
- **第四步，看类型码**：`f` 定点浮点、`e` 科学计数、`d` 十进制整数、`x` 十六进制、`,` 千分位。`{1e6:,.1f}` 输出 `1,000,000.0`——数据报告里的标准数字形态。

**为何要记这套语法？** 因为「把数据排成可读的文本」是输出层的核心工程：对齐表格、控制小数位、统一千分位，全部靠它。从数值计算到数据分析再到展示层，这套说明符贯穿始终。

## 5 综合应用：文本清洗流水线

把方法组合起来，就是一条清洗流水线——解析 CSV 行、清洗文本、重组输出：

```python
raw = "  Alice,92.5,B\n  Bob,87.0,A\n"
for line in raw.splitlines():
    name, score, grade = line.strip().split(",")      # 洗 → 切 → 解包
    print(f"{name:<8} {float(score):5.1f}  {grade}")
# Alice     92.5  B
# Bob       87.0  A
```

**重点：清洗 = strip + split + 校验，输出 = f-string 排版。** 这一行管道式的写法，正是数据工程师处理原始文件的日常。到第五级《数据清洗与特征工程》专题，你会看到同一套思想放大到百万行数据的规模。

## 6 小结

- 字符串**不可变**，「修改」都返回新对象；`split` 切、`join` 拼、`strip` 洗是三件套。
- 判断类方法（`isalpha`、`startswith`、`endswith`）常与 `if` 组合做文本校验。
- 三代格式化：`%`、`str.format`、f-string；**优先 f-string**，`{}` 内可用任意表达式。
- 格式说明符 `{值:对齐宽度.精度类型}` 控制排版；`:` 后可写 `>10`、`.2f`、`,`。
- 清洗流水线 = `strip` + `split` + 校验，输出 = f-string 排版；管道式写法贯穿数据工作。

在下一节，我们将把视野从语法拉回「用什么」——标准库导览，看看 Python 自带电池里的常用模块与 collections。
