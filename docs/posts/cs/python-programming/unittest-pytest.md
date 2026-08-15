---
title: 测试代码：unittest 与 pytest 入门
date: 2026-08-07
---

# 测试代码：unittest 与 pytest 入门

<div class="epigraph">
<p>没有测试的代码，是一栋没有地基的楼——看起来不错，经不起改动。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 《Python编程：从入门到实践》（第3版）第11章 ｜ 2026-08-07</p>
</div>

## 为什么从测试开始

写到这一节，本专题的语法已基本齐备。但「能跑」和「可靠」是两回事：你怎么知道改动没有破坏旧功能？怎么证明一个函数在各种输入下都对？答案是**自动化测试**——把「期望」写成代码，让机器替你反复检查。这是《Python编程》第 11 章的主题，也是工程化的最后一块拼图。

本节学习两个框架：标准库的 **`unittest`** 与社区事实标准的 **`pytest`**，以及它们背后的思想——测试让代码**可回归、可验证、可重构**。

## 1 为什么测试：回归与信心

**单元测试（unit test）**：针对程序最小的可测单元（通常是函数）写的自动化验证。它解决的问题是**回归（regression）**——改一处、坏另一处。

```python
def add(a, b):
    return a + b

assert add(2, 3) == 5          # 最朴素的测试：断言期望
```

**重点：测试是「对未来的承诺」。** 写好测试后，每次修改代码就运行一遍——全绿代表「没破坏什么」，红了代表「这里被改坏了」。这让**重构**变得安全：敢改代码，是因为有测试兜底。这个「先写断言、看它失败、再写实现让它通过」的循环，就是 **TDD（测试驱动开发）** 的核心节奏。<span class="marginnote">「回归测试」一词来自软件工程史：每次发布后要确保旧功能没「倒退（regress）」。著名的「红—绿—重构」节奏（red-green-refactor）是 TDD 的三拍：先写会失败的测试（红）→ 写最小实现让它通过（绿）→ 重构代码保持测试绿。</span>

## 2 unittest：标准库自带的测试框架

`unittest` 随 Python 安装，零依赖。它的写法受 Java 的 JUnit 启发：测试写在 `TestCase` 子类里，方法名以 `test_` 开头，用 `assert*` 系列方法断言：

```python
import unittest

def get_formatted_name(first, last):
    return f"{first} {last}".title()

class NameTestCase(unittest.TestCase):
    def test_first_last_name(self):
        formatted = get_formatted_name("janis", "joplin")
        self.assertEqual(formatted, "Janis Joplin")

    def test_first_last_middle_name(self):
        formatted = get_formatted_name("wolfgang", "mozart", "amadeus")
        self.assertEqual(formatted, "Wolfgang Amadeus Mozart")

if __name__ == "__main__":
    unittest.main()
```

运行 `python test_name.py`，输出：

```text
..
----------------------------------------------------------------------
Ran 2 tests in 0.001s

OK
```

**重点：断言用的是 `self.assertEqual`，不是 Python 的 `assert`。** `assertEqual(a, b)` 检查 `a == b`，失败时输出两个值的具体差异；`assertTrue`、`assertIn`、`assertRaises` 等是一整族断言方法。<span class="marginnote">`setUp()` 方法在每个测试执行前自动运行，用于准备共享数据——`self.xxx` 在 `setUp` 里赋值、在测试方法里使用。这是 unittest「一次准备、多个测试复用」的标准姿势。</span>

**辨析｜易错点：** 内置 `assert` 在 `python -O` 下会被禁用，而 `unittest` 的 `assertEqual` 是真实的方法调用，永远生效——**测试框架的断言永远比裸 `assert` 可靠**。

## 3 pytest：更简洁的测试体验

`pytest` 是第三方框架（`pip install pytest`），却成为 Python 社区事实标准。它最大的卖点是**零样板**——普通函数 + 普通 `assert` 就是测试：

```python
def add(a, b):
    return a + b

def test_add():                    # 函数名以 test_ 开头即可
    assert add(2, 3) == 5          # 用原生 assert，pytest 会捕获失败详情

def test_add_negative():
    assert add(-1, 1) == 0
```

运行 `pytest` 命令，pytest 自动发现并运行 `test_*.py` 里的 `test_*` 函数。

**重点：pytest 用原生 `assert` + 智能失败报告。** 失败时它重写 `assert` 的表达式，直接告诉你「左边是 4，右边是 5，哪里不相等」——比 unittest 更直观。参数化与夹具则把重复降到最低：

```python
import pytest

@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5), (0, 0, 0), (-1, 1, 0),
])
def test_add(a, b, expected):
    assert add(a, b) == expected     # 一组数据跑一遍测试
```

`@pytest.mark.parametrize` 用一组组输入跑同一测试，覆盖更多分支只添一行。<span class="marginnote">pytest 的<strong>夹具（fixture）</strong>用 `@pytest.fixture` 定义，通过参数注入给测试——`def test_x(db)` 自动拿到 `db` 夹具。这套依赖注入设计让「测试前置准备」与测试逻辑分离，是大型测试套件的关键结构。</span>

## 4 核心对比表：unittest 与 pytest

| 维度 | unittest | pytest |
| --- | --- | --- |
| 来源 | 标准库，零依赖 | 第三方，需 `pip install` |
| 测试形态 | `TestCase` 类 + `assertEqual` | 普通函数 + 原生 `assert` |
| 学习成本 | 中等（类与断言家族） | 低（函数即测试） |
| 失败报告 | 简洁 | 详细（重写 assert 显示差异） |
| 参数化 | 需 `subTest` 或手写 | `@pytest.mark.parametrize` 一行 |
| 夹具 | `setUp`/`tearDown` | `@pytest.fixture` 注入 |

**核心观察：选 pytest，除非你无法装第三方包。** 社区统计显示 pytest 已成主流——简洁、报告好、生态强。unittest 的价值在于「标准库自带」，在离线环境、最小依赖场景不可替代。两者思想相通：断言期望、自动化运行、报告结果。**无论选哪个，「有测试」都远比「测试框架有多高级」重要。**

## 5 小结：全专题收束

- **自动化测试**让代码可回归、可重构；「红—绿—重构」是 TDD 的节奏。
- `unittest`：标准库自带，`TestCase` 子类 + `assertEqual` 断言，`setUp` 准备共享数据。
- `pytest`：普通函数 + 原生 `assert` 即测试，`parametrize` 一行参数化、`fixture` 注入依赖。
- 测试框架的断言永远比裸 `assert` 可靠（`-O` 下 `assert` 会失效）。
- 工程顺序：写测试 → 跑测试 → 红 → 实现 → 绿 → 重构。

到这里，本专题画上句点：从环境搭建出发，我们走过了变量与数据类型、列表与字典、流程控制与函数、模块与包、文件读写、面向对象、特殊方法、迭代器与生成器、异常与调试、函数式特性与装饰器、字符串与标准库、虚拟环境与测试。这 21 节课共同构成了 Python 的完整骨架——再往上，无论是数据分析、Web 开发、自动化脚本还是机器学习，都只是在这副骨架上生长血肉。下一站，你可以顺着「从极限到大模型」的主线，进入本系列《数据可视化》《机器学习》等专题，让这些基础语法开始解决真实问题。
