---
title: 输入输出与文件读写
date: 2026-08-07
---

# 输入输出与文件读写

<div class="epigraph">
<p>程序一旦能读文件、写文件，它就从「玩具」变成了「工具」。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 《Python编程：从入门到实践》（第3版）第10章 ｜ 2026-08-07</p>
</div>

## 为什么从文件开始

到目前为止，程序的数据都活在内存里：程序一退出，数据就消失。真实程序要**持久化**——把数据存到磁盘，下次运行再读回来。这就是**文件读写**。

本节学习三件事：如何安全地打开与关闭文件（`open` 与 `with`）、如何按行读取与按块写入、以及如何用 **JSON** 把结构化数据存下来再读回。文件读写也是《Python编程》第 10 章「文件与异常」的主题，异常部分我们留到《异常处理与调试技巧》一节。

## 1 打开与关闭：open 与 with

Python 用内置函数 `open()` 打开文件，得到**文件对象（file object）**。最稳妥的打开方式是 `with` 语句：

```python
with open("pi_digits.txt") as f:
    contents = f.read()
print(contents)
```

**重点：`with` 是「上下文管理器」，负责自动关闭文件。** `with open(...) as f:` 块结束时，文件会被自动关闭——即使块内抛了异常。手动 `f.close()` 容易漏：文件没关会导致数据没落盘、占用句柄。

**路径辨析：** 相对路径相对的是**当前工作目录**（启动终端时的目录），而不是 `.py` 文件所在的目录——这常导致「换个终端跑就找不到文件」。要「以脚本自身为基准」，用 `pathlib` 拼接：

```python
from pathlib import Path
base = Path(__file__).parent        # 脚本所在目录
data_path = base / "data" / "notes.txt"
```

`Path(__file__).parent` 拿到脚本所在目录，`/` 运算符拼接路径——`pathlib` 是标准库处理路径的现代方式，跨平台、免去字符串拼接的坑。

`open()` 的第一参数是文件名，第二参数是**打开模式**，默认 `'r'`（只读）。文件路径是相对的：相对当前工作目录解析。用 `pathlib.Path` 或 `os.path.join` 处理跨平台路径更健壮。<span class="marginnote">`with` 语句后面会再次出现：在《特殊方法与运算符重载》一节，`with open(...) as f` 能工作的原因，是文件对象实现了 `__enter__` 与 `__exit__` 两个特殊方法——`with` 的自动清理机制源自这里。</span>

## 2 读取与写入：文本文件的完整操作

读取有几种姿势，按需选用：

```python
with open("poem.txt") as f:
    lines = f.readlines()        # 一次读出所有行，成列表
with open("poem.txt") as f:
    for line in f:               # 逐行迭代，省内存，大文件推荐
        print(line.rstrip())
```

**辨析｜易错点：** `read()` 返回整段字符串，`readline()` 读一行，`readlines()` 返回行列表。逐行 `for line in f:` 是最省内存的——它惰性读取，不会把整个文件一次性装进内存，处理几百 MB 日志时这是唯一可行的姿势。<span class="marginnote">行尾的 `\n` 会被保留，所以 `print(line)` 会多空一行，常用 `rstrip()` 去掉行尾空白。「每行读出来带换行、要自己处理」是文本处理里第一个意外。</span>

写入需要打开模式 `'w'`（覆盖写）或 `'a'`（追加）：

```python
with open("notes.txt", "w", encoding="utf-8") as f:
    f.write("第一条笔记\n")
with open("notes.txt", "a", encoding="utf-8") as f:
    f.write("追加的第二条\n")
```

**辨析｜易错点：** `'w'` 会**清空原文件再写**；想保留原有内容要用 `'a'`。`write()` 不会自动加换行，需自己补 `\n`。文本文件建议显式指定 `encoding="utf-8"`，否则在 Windows 上中文会乱码。

## 3 JSON：把数据存下来、再读回来

文本文件只能存字符串；要存列表、字典这类结构化数据，标准做法是 **JSON**（JavaScript Object Notation）。

```python
import json

numbers = [1, 2, 3, 4, 5]
with open("numbers.json", "w") as f:
    json.dump(numbers, f)          # 写入：列表 → JSON 文本

with open("numbers.json") as f:
    loaded = json.load(f)          # 读取：JSON 文本 → 列表
print(loaded)                      # [1, 2, 3, 4, 5]
```

**重点：`json.dump` 是「对象 → 文件」，`json.load` 是「文件 → 对象」。** JSON 与 Python 的对应关系很直接：对象 ↔ 字典，数组 ↔ 列表，字符串 ↔ 字符串，数字 ↔ 数字。配置文件、API 响应、模型参数几乎都以 JSON 为载体——它在《字典与映射》一节已经以「字典的孪生兄弟」身份出现过。<span class="marginnote">`json.loads` 处理字符串、`json.load` 处理文件，`dumps`/`dump` 同理（末尾的 `s` 表示 string）。JSON 不能存元组、集合、自定义对象，但 `default=` 参数可自定义序列化规则——这个细节在数据工程里常用来把 `datetime` 对象转成字符串。</span>

**辨析｜易错点：** `json.dump` 默认把中文转成 `\uXXXX` 转义序列、并输出单行紧凑文本。想要人类可读、中文直出的 JSON，加参数：`json.dump(data, f, ensure_ascii=False, indent=2)`——`indent` 控制缩进，`ensure_ascii=False` 保留中文原样。读回时 `json.load` 自动还原，无需额外处理。

## 4 核心对比表：文件打开模式

| 模式 | 含义 | 文件不存在时 | 对已有内容 |
| --- | --- | --- | --- |
| `'r'` | 只读（默认） | 报错 `FileNotFoundError` | 不变 |
| `'w'` | 写入 | 创建 | **清空后覆盖** |
| `'a'` | 追加 | 创建 | 在末尾追加 |
| `'r+'` | 读 + 写 | 报错 | 可读可写，光标在开头 |
| `'b'` | 二进制（可与上者组合） | 依模式 | 按字节处理 |

**核心观察：`'w'` 是破坏性的。** 打开前建议确认路径正确；对重要文件，先备份或先读后写。`'b'` 二进制模式用于图片、音视频等非文本数据，`read()` 返回字节对象而非字符串。模式选错，轻则白读，重则覆盖数据——这是文件操作里唯一的「不可撤销」事故。

## 5 异常的出现：文件不存在的处理

文件读写天然伴随失败：文件不存在、权限不足、磁盘满。这些**运行时错误**在 Python 里叫**异常（exception）**：

```python
try:
    with open("missing.txt") as f:
        print(f.read())
except FileNotFoundError:
    print("抱歉，文件不存在。")
```

**重点：程序不该因文件缺失而崩溃。** 用 `try/except` 捕获 `FileNotFoundError` 后程序继续运行——这是异常处理的第一次实战。完整的 `try/except/else/finally` 语法、异常层级与自定义异常，将在《异常处理与调试技巧》一节系统展开；这里先记住一句话：**可能失败的操作，要准备它的失败。**

## 6 小结

- 用 `with open(路径, 模式) as f:` 打开文件，块结束自动关闭，无需手动 `close()`。
- 读取三兄弟：`read()`（全读）、`readline()`（一行）、`readlines()`（行列表）；逐行 `for line in f` 最省内存。
- 写入用 `'w'`（覆盖）或 `'a'`（追加），`'w'` 会清空原文件；文本建议指定 `encoding="utf-8"`。
- JSON 是结构化数据的标准载体：`json.dump`/`json.load` 配对使用。
- 文件不存在会抛 `FileNotFoundError`，用 `try/except` 捕获，避免程序崩溃。

在下一节，我们将进入面向对象的世界——类与对象，把数据和操作它的一组函数打包成一个个「模型」。
