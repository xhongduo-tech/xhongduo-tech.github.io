---
title: 标准库导览：常用模块与 collections
date: 2026-08-07
---

# 标准库导览：常用模块与 collections

<div class="epigraph">
<p>Python 自带电池——标准库就是它随语言附赠的武器库。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 官方 Python 教程 第10章 ｜ 2026-08-07</p>
</div>

## 为什么从标准库开始

语法学的再多，也只是「会用语言」；能交付工作，靠的是**标准库**——Python 随语言附赠的几百个模块，覆盖文件、网络、数学、日期、数据容器。官方 Python 教程第 10 章「标准库巡礼」就是它的地图。

本节带你认识两个层次：**日常高频模块**（`os`、`sys`、`math`、`random`、`datetime`、`json`）与**数据容器增强**（`collections` 的四件套）。学会「找标准库、用标准库」，是「不重复造轮子」的开始。

## 1 标准库：Python 的「自带电池」

**标准库（standard library）**：随 Python 一起安装、无需额外下载即可 `import` 的模块集合。「batteries included」——从《起步》一节那句设计哲学到这里落地。

```python
import os, sys, math, random, datetime, json
```

六大高频模块一句话定位：

- **`os`**：与操作系统交互——路径、目录、环境变量。`os.getcwd()`、`os.mkdir("data")`。
- **`sys`**：与解释器交互——`sys.argv` 拿命令行参数，`sys.path` 是导入搜索路径。
- **`math`**：数学函数——`math.sqrt(2)`、`math.pi`、`math.floor(3.7)`。
- **`random`**：随机数——`random.randint(1, 6)`、`random.choice(seq)`。
- **`datetime`**：日期时间——`datetime.date.today()`、`datetime.timedelta`。
- **`json`**：JSON 读写——`json.load`/`json.dump`，见《输入输出与文件读写》。

**重点：先查标准库，再考虑写代码。** 大量「看起来要自己实现」的功能，标准库早已有之——`math.gcd` 求最大公约数、`statistics.mean` 求均值、`itertools.product` 求笛卡尔积。用 `dir(模块)` 或 `help(模块)` 在 REPL 里探索，比闭门造车快得多。<span class="marginnote">「标准库优先」是工程铁律：标准库经过多年打磨、有官方文档与安全维护。要处理路径，`os.path`/`pathlib` 提供了跨平台的正确做法，自己拼接字符串路径在 Windows 上必踩坑（反斜杠 vs 正斜杠）。</span>

## 2 collections：容器家族的增强件

`collections` 模块是对内建容器（列表、字典、元组）的「增强版」，其中最常用四件套：

```python
from collections import namedtuple, defaultdict, Counter, deque
```

**namedtuple**：有名字的元组，像元组一样不可变，却能用属性名访问：

```python
Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)
print(p.x, p.y)              # 3 4，属性访问比 p[0] 可读
```

**defaultdict**：字典的「缺省值」版本——访问不存在的键不会抛 `KeyError`，而是自动用默认工厂创建：

```python
from collections import defaultdict
word_counts = defaultdict(int)
for w in "abracadabra":
    word_counts[w] += 1      # 首次访问 w 自动初始化为 0
print(word_counts["a"])      # 5
```

**Counter**：计数器的开箱即用——`most_common()` 直接给出频次排行：

```python
from collections import Counter
c = Counter("abracadabra")
print(c.most_common(2))      # [('a', 5), ('r', 2)]
```

**deque**：双端队列——两端都能高效插入/弹出，`deque` 是队列任务的正确容器：

```python
from collections import deque
q = deque([1, 2, 3])
q.append(4)                  # 右端入队
q.appendleft(0)              # 左端入队，O(1)
print(q.popleft())           # 0，左端出队，O(1)
```

**重点：容器选择是性能与语义的双重决定。** `list` 首部插入是 $O(n)$，`deque` 两端都是 $O(1)$——「先进先出」的队列用 `deque` 而非 `list`；「按键统计」用 `Counter`；「缺省键」用 `defaultdict`。语义对、性能也对，代码才站得住。<span class="marginnote">`defaultdict(int)` 之所以能 `+= 1`，是因为首次访问不存在的键时，`int()` 返回 0 作为初始值。把 `int` 换成 `list`、`set`，就得到「按键分组的容器」——`defaultdict(list)` 是分组数据的标准姿势。</span>

## 3 实战：用标准库组装一个小程序

把上面模块串起来，做一次「掷骰子模拟」：

```python
import random
from collections import Counter

rolls = [random.randint(1, 6) for _ in range(10000)]
counts = Counter(rolls)
for face in range(1, 7):
    print(f"{face}: {'#' * (counts[face] // 100)}  {counts[face]}")
```

**重点：六行代码完成 1 万次模拟与统计。** `random.randint` 负责随机，列表推导负责批量，`Counter` 负责计数，f-string 负责可视化。这展示了标准库的组合价值——每个模块解决一小步，拼起来就是完整功能。这正是「自带电池」的意义：你不用从零写随机数、计数表、可视化。到第四级《机器学习》专题做蒙特卡洛模拟，这套组合仍是骨架。

## 4 核心对比表：collections 四件套

| 容器 | 本质 | 解决什么 | 典型场景 |
| --- | --- | --- | --- |
| `namedtuple` | 有名字的元组 | 可读性、不可变性 | 坐标、记录、返回多个值 |
| `defaultdict` | 带默认值的字典 | 免 `KeyError`、自动初始化 | 计数、按键分组 |
| `Counter` | 计数器字典 | 频次统计、排行 | 词频、投票、分布 |
| `deque` | 双端队列 | 两端 $O(1)$ 插入删除 | 队列、滚动窗口 |

**核心观察：选容器 = 选数据结构的语义。** 从第二级《数据结构》的角度看，每种容器都对应一种抽象：元组是「不可变的复合值」，字典是「键值映射」，队列是「先进先出」。理解底层数据结构，标准库的每个类就不只是「魔法 API」，而是「我知道它为什么快」。

## 5 小结

- 标准库随语言附带，`os`、`sys`、`math`、`random`、`datetime`、`json` 覆盖日常六成需求。
- 用 `dir()`/`help()` 在 REPL 里探索模块；**先查标准库，再考虑自己写**。
- `namedtuple` 给元组加名字；`defaultdict` 免 `KeyError`；`Counter` 一键计数；`deque` 两端 $O(1)$。
- 容器选择 = 语义 + 性能：队列用 `deque`，分组用 `defaultdict(list)`。
- 标准库模块组合能快速拼出完整功能，是「不重复造轮子」的开始。

在下一节，我们将处理工程化的第一道门槛——虚拟环境与包管理，用 venv 与 pip 管理第三方依赖。
