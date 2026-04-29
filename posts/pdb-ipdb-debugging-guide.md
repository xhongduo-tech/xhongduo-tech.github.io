## 核心结论

`pdb` 是 Python 标准库自带的命令行调试器，`ipdb` 是基于 IPython 交互能力包装出来的增强版调试器。两者最重要的共同点是：都能在程序运行到某个位置时暂停，让你检查“这一刻的真实状态”；最重要的差异是：`ipdb` 的交互体验通常更顺手，比如补全、语法高亮、对象展示更友好，但它不是标准库，部署环境里未必默认可用。

如果目标是稳定、零依赖、任何 Python 环境基本都能用，优先选 `pdb`。如果你已经安装了 IPython，日常又常在终端里做交互探索，希望调试时输入更舒服、输出更清晰，再用 `ipdb`。

最小入口通常不是 `import pdb; pdb.set_trace()`，而是 `breakpoint()`。它是 Python 3.7+ 推荐的统一断点入口，白话解释就是“把暂停调试这件事交给 Python 运行时处理”，默认会进入标准调试器，也可以被环境变量或 hook 替换。

| 对比项 | `pdb` | `ipdb` |
|---|---|---|
| 依赖 | 标准库，无需安装 | 需要额外安装 |
| 默认可用性 | 几乎所有 Python 环境可用 | 取决于环境是否装了 `ipdb`/IPython |
| 交互体验 | 基础、稳定 | 更强的补全和展示 |
| 调试语义 | 标准命令集 | 基本兼容 `pdb` |
| 适合场景 | 线上容器、最小环境、CI 复现 | 本地开发、交互探索、复杂对象查看 |

一个新手可立即上手的例子如下。程序在 `breakpoint()` 处停住后，可以用 `p x`、`p y` 查看变量，用 `n` 执行下一行，用 `c` 继续运行到下一个断点或结束。

```python
def add_and_scale(x, y):
    total = x + y
    breakpoint()
    result = total * 10
    return result

assert add_and_scale(2, 3) == 50
```

这个例子本身没有 bug，但它准确展示了调试器的价值：你不是猜 `total` 是否等于 5，而是在运行时直接看见它。

---

## 问题定义与边界

本文讨论的是“在 Python 进程内部暂停执行，并检查运行时状态”的命令行调试。运行时状态，白话解释就是“程序此刻手里拿着哪些变量、正走到哪一行、从哪层函数调用过来”。这和日志系统、性能分析器、远程观测平台不是一类工具。

`pdb/ipdb` 擅长回答的问题是：

| 问题类型 | 更合适的工具 | 原因 |
|---|---|---|
| 某次调用里变量为什么不对 | `pdb/ipdb` | 需要看单次执行现场 |
| 某个异常到底从哪层传出来 | `pdb/ipdb` | 需要看栈和局部变量 |
| 函数是否被调用过 | `print`/`logging` | 只需轻量观测 |
| 代码覆盖某分支没有 | `pytest` + 断言 | 需要可重复验证 |
| 程序整体为什么变慢 | profiler | 需要统计热点，不是单次停住 |
| 分布式链路哪里超时 | tracing/logging | 单进程调试器看不到全局 |

一个典型场景是：程序偶发报错，日志只告诉你最后抛了 `ValueError`，但你真正想知道的是“哪一个输入字段把逻辑带偏了”。如果继续加 `print`，你需要重新部署、重跑、筛输出；而断点调试可以直接在异常发生前停住，看局部变量和调用链。

下面是一个“只看最终异常不够”的玩具例子：

```python
def calc_discounted_price(price, user_type, coupon):
    if user_type == "vip":
        discount = 0.8
    else:
        discount = 0.95

    if coupon:
        discount -= coupon  # bug: coupon 可能是 20，语义应是 0.2

    final_price = price * discount
    if final_price < 0:
        raise ValueError("final price must be non-negative")
    return final_price

try:
    calc_discounted_price(100, "vip", 20)
except ValueError:
    pass
```

这里只看异常信息，只知道价格变成负数；用调试器停在 `discount -= coupon` 或异常抛出前，你才能立刻确认问题是“券值单位设计错了”，不是 `price` 或 `user_type` 出错。

边界也要讲清楚。`pdb/ipdb` 能检查 Python 层的 frame。frame，白话解释就是“一层函数调用在当前时刻的执行现场”。它能看局部变量、调用栈、当前行号，但不能直接把 C 扩展内部每一步都展开给你看。比如 NumPy、正则引擎、某些数据库驱动内部的 C 代码，不会像 Python 行号那样逐行可见。

---

## 核心机制与推导

`pdb/ipdb` 的工作基础是 trace hook。trace hook，白话解释就是“Python 解释器在执行关键事件时，给外部一个可拦截的回调入口”。常见事件包括 `call`、`line`、`return`、`exception`，分别表示进入函数、执行到一行、函数返回、抛出异常。

所以“断点命中”并不是魔法暂停，而是执行流走到某个事件点后，调试器判断：这里是不是我关心的位置；如果是，就接管控制权。这个过程可以概括成：

$$
S = hit(line) \land cond(line, state) \land \neg skip(frame)
$$

其中：

- $hit(line)$：当前执行事件到达了目标行
- $cond(line, state)$：如果设置了条件断点，当前状态满足条件
- $\neg skip(frame)$：当前 frame 不在跳过范围内

理解这个式子后，很多命令的行为就统一了。比如：

| 命令 | 语义 | 什么时候用 |
|---|---|---|
| `step` / `s` | 进入下一次可停的执行事件，可能进入子函数 | 你怀疑 bug 在被调用函数内部 |
| `next` / `n` | 执行当前行，但不进入当前行调用的子函数 | 你只关心本层逻辑 |
| `return` / `r` | 继续执行到当前函数返回 | 你确认本层没问题，想看返回值 |
| `continue` / `c` | 继续跑到下一个断点或程序结束 | 现场看完了，继续执行 |

“当前 frame”也因此变得容易理解。假设 `A()` 调 `B()`，`B()` 再调 `C()`，你在 `C()` 里停住时，当前 frame 是 `C()` 的执行现场；`up` 是往调用者方向看上一层，也就是 `B()`；`down` 是回到更深的一层。调用栈，白话解释就是“函数一层套一层调用时形成的路径记录”。

下面这个例子展示“暂停发生在什么时刻”：

```python
def inner(x):
    y = x * 2
    return y

def outer(n):
    a = n + 1
    b = inner(a)   # 对 `next` 来说，通常会整行走完；对 `step` 会进入 inner
    c = b + 3
    return c

assert outer(4) == 13
```

如果断点下在 `b = inner(a)` 这一行：

- 用 `n`，调试器通常会执行完整行，再停到下一行，结果是你直接看到 `b` 已经算完。
- 用 `s`，调试器会进入 `inner`，你可以继续看 `x`、`y` 的变化。

真实工程里，这个差别非常关键。比如 Web 服务里一个请求经过参数解析、鉴权、数据库访问、序列化输出四层调用。你如果只怀疑业务分支判断错了，应该多用 `next`；如果怀疑 ORM 返回数据异常，才值得 `step` 进入更深层。

---

## 代码实现

现代 Python 里最推荐的入口是 `breakpoint()`。它最终会调用 `sys.breakpointhook()`。hook，白话解释就是“预留出来、允许你替换的处理入口”。默认情况下，这个 hook 会进入标准调试器；如果你设置了 `PYTHONBREAKPOINT`，也可以改成别的实现。

先看最小可运行例子：

```python
def normalize_score(raw_score):
    score = int(raw_score)
    breakpoint()
    if score < 0:
        score = 0
    if score > 100:
        score = 100
    return score

assert normalize_score("120") == 100
```

调试时常用命令速查如下：

| 命令 | 含义 | 常见用途 |
|---|---|---|
| `p expr` | 打印表达式值 | 看单个变量 |
| `pp expr` | 美化打印 | 看字典、列表、嵌套对象 |
| `l` | 查看附近源码 | 确认当前上下文 |
| `n` | 下一步，不进子函数 | 跟本层逻辑 |
| `s` | 下一步，可能进子函数 | 钻进可疑调用 |
| `c` | 继续执行 | 放行到下个断点 |
| `w` / `where` | 查看调用栈 | 看从哪层调用进来 |
| `u` | 上移一层 frame | 看调用者局部变量 |
| `d` | 下移一层 frame | 回到更深层 |

如果你使用 `ipdb`，最常见入口是 `ipdb.set_trace()`：

```python
import ipdb

def choose_handler(event):
    event_type = event["type"]
    ipdb.set_trace()
    if event_type == "click":
        return "mouse-handler"
    if event_type == "submit":
        return "form-handler"
    return "default-handler"

assert choose_handler({"type": "click"}) == "mouse-handler"
```

真实工程例子可以更贴近服务端排障。假设你在批处理任务里发现只有某个用户样本算错，不想让每一条数据都停住，就应该用条件断点。条件断点，白话解释就是“只有满足条件时才暂停”。

```python
def process_orders(orders):
    total = 0
    for order in orders:
        user_id = order["user_id"]
        amount = order["amount"]
        total += amount  # 在这里下条件断点: user_id == 1003
    return total

orders = [
    {"user_id": 1001, "amount": 10},
    {"user_id": 1002, "amount": 20},
    {"user_id": 1003, "amount": 999},
]
assert process_orders(orders) == 1029
```

在 `pdb` 里可以先 `b 文件名:行号, user_id == 1003`，也可以在交互后对已存在断点加条件。这样不会让前 1002 个样本都停住。

还有一种常见方式是事后调试，也就是程序已经抛异常了，再进入现场。`pdb.pm()` 里的 `pm` 是 post-mortem，白话解释就是“死后检视”，即异常抛出后的现场分析。

```python
import pdb

def divide(a, b):
    return a / b

try:
    divide(10, 0)
except ZeroDivisionError:
    pdb.pm()
```

这种方式特别适合“错误已经稳定复现，但触发路径很长”的场景。你不用提前埋断点，只要在异常发生后接管现场即可。

---

## 工程权衡与常见坑

调试器不是没有成本。它会改变执行节奏，会阻塞当前进程，还可能因为打印大对象而显著拖慢程序。一个简单的定性表达是：

$$
Cost \approx N_{stop} \times (T_{interaction} + T_{render})
$$

其中 $N_{stop}$ 是停顿次数，$T_{interaction}$ 是你每次人工查看和输入命令的时间，$T_{render}$ 是变量展示本身的时间。停得越多、打印对象越大，开销越高。

常见坑如下：

| 常见坑 | 现象 | 规避方式 |
|---|---|---|
| 混淆 `step` 和 `next` | 一脚踩进库函数，越调越深 | 先用 `next`，只在怀疑子函数时再 `step` |
| 多进程里断点下错进程 | 主进程停了，真正工作进程没停 | 先确认执行代码的是哪一个 PID/worker |
| 大对象频繁 `p` | 调试器本身卡顿 | 只打印关键字段，优先 `p obj["id"]` 这种局部表达式 |
| C 扩展内部不可见 | 看到调用入口，看不到内部每步 | 转向文档、日志、源码或更底层工具 |
| 在 `p` 里执行有副作用表达式 | 调试时把状态改坏了 | 只读表达式优先，避免调用会写库、发请求、修改全局状态的函数 |

错误使用和正确使用的对比如下。先看错误方式：

```python
def bad_debug(records):
    for record in records:
        breakpoint()
        # 每条都停，而且下面如果 record 很大，p record 会非常慢
        result = record["value"] * 2
    return True

records = [{"value": i} for i in range(3)]
assert bad_debug(records) is True
```

如果真实数据有 50 万条，这种调法几乎不可用。更合理的是只对可疑样本停住：

```python
def good_debug(records):
    for record in records:
        record_id = record["id"]
        value = record["value"]
        # 在这一行配条件断点: record_id == 424242
        result = value * 2
        if result < 0:
            return False
    return True

records = [
    {"id": 1, "value": 3},
    {"id": 424242, "value": 5},
]
assert good_debug(records) is True
```

再看一个真实工程里的坑。很多人在线上 Celery worker、Gunicorn worker、pytest xdist 多进程测试里打了断点，却发现终端没反应。根因通常不是调试器失效，而是“你交互的终端”和“实际执行代码的子进程”不是一个控制台。此时应先缩小并发、只跑单 worker、只复现单 case，再进入调试。

另一个常见失败模式是把调试器当成长期观测方案。调试器适合“我想看这一次到底发生了什么”，不适合“我想连续观察三天线上流量”。后者应该用日志、指标、链路追踪。

---

## 替代方案与适用边界

`pdb/ipdb` 是交互式定位逻辑错误的利器，但它不是万能工具。很多问题如果用别的手段，成本更低、结果更稳。

| 工具 | 最适合解决的问题 | 不适合解决的问题 |
|---|---|---|
| `print` / `logging` | 轻量确认分支、记录长期行为 | 复杂现场、临时深入排障 |
| `pdb/ipdb` | 单次执行中看变量、栈、分支 | 长期观测、性能统计、分布式全局分析 |
| `pytest` | 复现 bug、固定回归、建立断言 | 临时探索未知现场 |
| profiler | 找热点、看耗时分布 | 某次输入为什么走错分支 |
| 远程调试器/观测平台 | 容器、远端服务、多人协作排障 | 本地最小化单步排查 |

“何时用 `pdb`、何时用 `ipdb`、何时不用”可以直接这样判断：

| 场景 | 建议 |
|---|---|
| 线上最小容器、依赖越少越好 | 用 `pdb` |
| 本地开发，常看复杂字典、对象树 | 用 `ipdb` |
| 只是想确认函数是否被调用 | 先用日志或测试 |
| 想知道性能瓶颈在哪 | 用 profiler |
| 需要排查分布式请求跨服务行为 | 用 tracing / logging，不要硬上命令行断点 |

一个最小对比例子：

```python
def route_request(path):
    print(f"route_request called with {path}")  # 适合确认“有没有被调用”
    if path.startswith("/admin"):
        return "admin"
    return "public"

assert route_request("/admin/users") == "admin"
```

这类问题只需知道函数是否被调用，用日志就够了。但如果问题变成“为什么同样是 `/admin/users`，某一次却没走到 `admin` 分支”，那就应该下断点，直接检查 `path` 是否被中间件改写、是否带了空格、是否被前置规则截断。

所以边界很明确：`pdb/ipdb` 解决的是“现场不透明”，不是“所有诊断问题”。真正稳健的工程实践通常是组合使用：先用测试固定复现，再用调试器理解机制，最后用断言和日志把问题永久收口。

---

## 参考资料

| 来源 | 用途 | 推荐顺序 |
|---|---|---|
| Python `pdb` 官方文档 | 学命令语义和入口方式 | 1 |
| Python `bdb` 官方文档 | 理解调试器抽象层 | 2 |
| Python `sys.settrace` 官方文档 | 理解 trace 事件机制 | 3 |
| Python `breakpoint()` 官方文档 | 理解统一断点入口 | 4 |
| IPython 调试文档 | 理解增强交互体验 | 5 |
| `ipdb` 项目页 | 看安装和兼容说明 | 6 |

1. [Python 官方文档：pdb](https://docs.python.org/3/library/pdb.html)
2. [Python 官方文档：bdb](https://docs.python.org/3/library/bdb.html)
3. [Python 官方文档：sys.settrace](https://docs.python.org/3/library/sys.html#sys.settrace)
4. [Python 官方文档：breakpoint()](https://docs.python.org/3/library/functions.html#breakpoint)
5. [IPython 文档：Debugger](https://ipython.readthedocs.io/en/stable/interactive/reference.html#debugger)
6. [ipdb 项目主页](https://pypi.org/project/ipdb/)
