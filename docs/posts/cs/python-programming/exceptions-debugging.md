---
title: 异常处理与调试技巧
date: 2026-08-07
---

# 异常处理与调试技巧

<div class="epigraph">
<p>程序出错不可怕，可怕的是出错后不知道错在哪里。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 官方 Python 教程 第8章 ｜ 2026-08-07</p>
</div>

## 为什么从异常开始

运行中的程序随时可能出错：文件不存在、用户输入了无法解析的数字、除数为零。**异常（exception）** 是 Python 报告「运行期出错」的机制——它不是程序的终点，而是「可被拦截并处理」的信号。官方 Python 教程第 8 章的全部主题，就是「如何优雅地面对错误」。

本节学两件事：**异常处理**（`try/except` 家族）与**调试技巧**（把错误定位出来）。

## 1 异常：错误也是一种值

程序运行到非法操作时，Python 抛出一个**异常对象**，包含错误类型与信息；若不处理，程序会在那一行**终止**并打印 traceback（回溯）。常见异常类型：

```python
int("abc")            # ValueError：无法把 "abc" 转成整数
[1, 2][5]             # IndexError：下标越界
{"a": 1}["b"]         # KeyError：键不存在
1 / 0                 # ZeroDivisionError：除数为零
```

这些异常构成了一个**层级体系**：`ValueError`、`TypeError`、`IndexError`、`KeyError` 都是 `Exception` 的子类，`Exception` 又是 `BaseException` 的子类。<span class="marginnote">异常层级决定了「捕获粒度」：`except Exception:` 能抓住绝大多数普通错误，但抓不住 `KeyboardInterrupt`（Ctrl+C）与 `SystemExit`——它们直接继承 `BaseException`。写程序时 `except Exception` 是常用底线，`except BaseException` 几乎总是错的。</span>

**辨析｜易错点：** 语法错误（SyntaxError）是**编译期**错误，不是运行时异常——`if x` 少个冒号，程序根本跑不起来。异常处理的只有**运行期**错误。

## 2 try/except/else/finally：完整的异常语法

Python 的异常处理有四个子句，各司其职：

```python
try:
    age = int(input("年龄："))
except ValueError:                    # 捕获指定异常
    print("请输入数字！")
else:                                 # 没有异常时执行
    print(f"明年你就 {age + 1} 岁了。")
finally:                              # 无论是否异常都执行
    print("本次输入结束。")
```

**重点：四个子句的分工。** `try` 里放**可能失败**的代码；`except` 捕获并处理异常；`else` 只在**一切正常**时执行；`finally` **无论如何**都会执行（用于清理资源）。`except` 可同时捕获多种异常：

```python
try:
    result = 10 / int(input("除数："))
except (ValueError, ZeroDivisionError):
    print("除数必须是整数且不为零。")
```

**raise** 是主动抛异常——「这个条件不该出现」：

```python
def set_age(age):
    if not 0 <= age <= 150:
        raise ValueError(f"年龄 {age} 不合理")
    return age
```

`raise ValueError(...)` 把「函数使用方传了坏参数」显式上报，调用方才能捕获处理——**在错误最早出现的地方抛异常，是防御式编程的第一原则**。<span class="marginnote">`else` 与 `finally` 容易被忽略，但它们有意义：`else` 让「成功路径」与「异常路径」分离，避免把不该被 try 保护的代码也包进去；`finally` 保证 `with` 之外的资源（网络连接、锁）一定能被释放。</span>

## 3 调试技巧：print、assert、pdb 与日志

异常处理是「事后」的兜底，**调试**是「事前」的定位。四个由浅入深的武器：

**1. `print` 探针**：在可疑处打印中间变量，是最快也最笨的办法。

**2. `assert` 断言**：检查「此时此处的假设」：

```python
def average(nums):
    assert len(nums) > 0, "空列表无法求平均"
    return sum(nums) / len(nums)
```

断言失败抛 `AssertionError`，用于「不变量」检查。`python -O` 运行时断言会被禁用，所以**不要用断言做真正的数据校验**——它只是开发期护栏。

**3. `pdb` 调试器**：让程序在任何一行暂停、逐行查看：

```python
import pdb
pdb.set_trace()          # 在此处进入调试器
```

进入后可用 `n`（下一步）、`p 变量`（打印）、`q`（退出）、`c`（继续）。现代 IDE（如 VS Code）把断点做成了可视化，原理相同。

**4. `logging` 日志**：生产环境用日志而非 `print`，因为日志带时间戳、等级、可重定向到文件：

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.info("开始处理")
logging.error("处理失败：%s", "文件缺失")
```

**重点：调试的次序是「先复现、再定位、后修复」。** 一个典型的思路：最小化复现用例 → 沿调用栈向上排查 → 修好后跑回归测试确认没破坏别的。这与《测试代码：unittest 与 pytest》一节的 TDD 精神一脉相承。

## 4 核心对比表：EAFP 与 LBYL

| 维度 | EAFP | LBYL |
| --- | --- | --- |
| 全称 | Easier to Ask for Forgiveness than Permission | Look Before You Leap |
| 策略 | 先尝试，出错再捕获 | 先检查条件，再执行 |
| 写法 | `try: ... except:` | `if ...: ... else:` |
| Python 风格 | 官方推荐、Pythonic | 传统语言风格 |
| 风险 | 异常多时性能略低 | 检查与执行间有竞态 |

**核心观察：Python 社区偏爱 EAFP。** 两个例子的对照：

```python
# LBYL
if "key" in d:
    value = d["key"]

# EAFP
try:
    value = d["key"]
except KeyError:
    value = None
```

EAFP 的好处是**没有竞态**——「检查后、执行前」数据可能已变（多线程、文件系统）；坏处是异常路径靠 `try` 包裹，读起来不如 `if` 直白。取舍标准：**可能失败的频率**——经常失败用 LBYL 显式判断，偶发失败用 EAFP 兜底。<span class="marginnote">字典的 `d.get("key", 默认值)` 其实是「EAFP 精神的内建实现」——它把「存在则取值、不存在给默认」封装成了一个调用，既无 `try` 也无 `if`。标准库把常用错误路径都做成了这种「不用你写异常」的 API。</span>

## 5 小结

- **异常**是运行期错误的信号，有层级体系（`ValueError`、`IndexError`、`KeyError` 等都是 `Exception` 子类）。
- `try/except` 捕获处理，`else` 只在无异常时执行，`finally` 无论怎样都执行。
- `raise` 主动抛异常，在错误最早出现处上报；防御式编程的第一原则。
- 调试四武器：`print` 探针、`assert` 断言、`pdb` 调试器、`logging` 日志。
- 捕获策略：**EAFP**（先试再捕）是 Python 主流，**LBYL**（先查再做）适合经常失败的场景。

在下一节，我们将转向函数式编程的特性——lambda、闭包与高阶函数，把函数当作一等公民来传递。
