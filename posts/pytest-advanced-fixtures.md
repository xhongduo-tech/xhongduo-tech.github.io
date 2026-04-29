## 核心结论

`fixture` 管环境，`parametrize` 管输入。这是理解 Pytest 进阶用法的最短路径。

`fixture` 的职责是准备、注入、回收测试上下文。测试上下文就是测试运行前后需要的依赖和资源，比如临时文件、数据库连接、鉴权 token。`parametrize` 的职责是在收集阶段把一组输入展开成多个独立 case。收集阶段就是 Pytest 先扫描测试文件、决定“总共要跑哪些测试”的阶段。

如果没有 `fixture`，测试里常会重复写 setup 和 teardown；如果没有 `parametrize`，同一段断言往往只验证一个输入，覆盖面不够，回归时容易漏掉边界样本。

最小对比例子：

- 没有 `fixture`：每个测试自己创建用户、自己删临时目录，重复且容易漏清理。
- 没有 `parametrize`：只测 `add(1, 2) == 3`，但没测 `0`、负数、非法输入。

可以把它公式化记成：

$$
fixture = Arrange,\quad parametrize = \text{批量生成 case}
$$

| 维度 | `fixture` | `@pytest.mark.parametrize` |
| --- | --- | --- |
| 职责 | 准备与回收依赖 | 展开测试输入 |
| 发生阶段 | setup / teardown | collection |
| 典型用途 | 数据库、临时目录、客户端、登录态 | 合法值、边界值、异常值 |
| 结果形态 | 给测试函数注入对象 | 生成多个独立测试 case |

工程上真正重要的不是“写法更优雅”，而是输入、依赖、执行次数被显式化后，CI 才能稳定统计 case 数、失败样本、慢测试和回归范围。

---

## 问题定义与边界

先把测试里最容易混在一起的三个对象拆开：

| 对象 | 含义 | 典型例子 |
| --- | --- | --- |
| 依赖 | 测试运行所需的外部条件 | 数据库连接、HTTP client、临时目录 |
| 输入 | 送进被测逻辑的数据 | 用户角色、订单 payload、地区码 |
| 执行次数 | 同一断言要跑多少组 | 3 个角色、5 个边界值、2 个环境配置 |

Pytest 要把它们显式化，是因为这三者决定了测试是否可维护：

- 依赖不显式，setup 会散落在测试函数里。
- 输入不显式，覆盖范围只能靠人肉记忆。
- 执行次数不显式，CI 很难判断“这次是不是少跑了”。

边界要讲清楚：

- `fixture` 不是业务逻辑容器。不要把“下单流程本身”写进 fixture，再让测试只做一个空断言。
- `parametrize` 不是运行时数据读取器。不要指望它在执行时临时查数据库、拉接口、读线上配置。
- 两者本质上都服务于测试编排。测试编排就是决定“先准备什么、测哪些输入、每个 case 怎么清理”。

用“订单服务接口测试”做边界例子：

- 适合放进 `fixture` 的内容：
  - 测试数据库
  - 已登录 client
  - 基线订单数据
  - 清理逻辑
- 适合放进 `parametrize` 的内容：
  - `role in {guest, user, admin}`
  - `payload in {合法, 缺字段, 越权}`
  - `region in {cn, us}`
- 不适合直接放进参数化的内容：
  - 实时查询线上商品库存
  - 随机生成且不可复现的数据
  - 需要长时间初始化的大对象

边界表：

| 适合 `fixture` 的内容 | 适合 `parametrize` 的内容 | 不适合放进去的内容 |
| --- | --- | --- |
| 数据库连接与回收 | 稳定、可枚举的输入集合 | 线上实时状态 |
| 临时目录 | 边界值、异常值、权限组合 | 大量不可复现随机数据 |
| 鉴权 token | 期望状态码、地区、角色 | 重型初始化逻辑放进参数列表 |
| mock 服务 | 业务维度的离散组合 | 把业务主流程本身封进 fixture |

一个简化流程图：

`collection phase -> setup phase -> test execution -> teardown`

这里最关键的点是：`parametrize` 主要影响左边，`fixture` 主要影响中间和右边。两者关注点不同，所以不要互相替代。

---

## 核心机制与推导

先看 `parametrize` 的机制。它不是在测试函数内部做 `for` 循环，而是在 collection phase 直接生成多个独立 case。独立 case 的意思是：每一组输入都有自己的测试名、自己的通过失败结果、自己的重跑能力。

单层参数化时，总 case 数为：

$$
N = |P|
$$

其中 $P$ 是参数集合。

多层叠加时，总 case 数为：

$$
N = \prod_{i=1}^{k} |P_i|
$$

也就是笛卡尔积。笛卡尔积就是“每一层都和其他层全部组合一遍”。

玩具例子：

- `@pytest.mark.parametrize("x", [10, 20, 30])`
- `@pytest.fixture(params=[1, 2])`

如果测试函数同时依赖 `x` 和这个带参数的 fixture，那么总 case 数就是：

$$
N = 3 \times 2 = 6
$$

展开后是这 6 个 case：

| fixture 参数 | `x` | case 数 |
| --- | --- | --- |
| 1 | 10 | 1 |
| 1 | 20 | 1 |
| 1 | 30 | 1 |
| 2 | 10 | 1 |
| 2 | 20 | 1 |
| 2 | 30 | 1 |

如果断言写成 `x + base == 11`，只有 `(x=10, base=1)` 会通过，其余 5 个都失败。这个例子虽然小，但很适合先用 `pytest --collect-only` 检查总 case 数是不是 6。对新手来说，这比先盯着失败日志更容易建立正确心智模型。

再看 `fixture(params=[...])`。它也是参数化，但参数是挂在 fixture 上，而不是测试函数上。官方文档和源码视图都说明了：带 `params` 的 fixture 会让依赖它的测试被多次调用，当前参数值通过 `request.param` 取得。

`indirect=True` 又是什么？它的作用是把参数先送进 fixture，而不是直接送进测试函数。这样可以把“数据选择”和“昂贵初始化”拆开。

例如：

- 参数里只写 `"sqlite"`、`"postgres"`
- fixture 里根据 `request.param` 决定真正创建哪种 client

这样做的价值是：测试收集时只关心有哪些输入标签，真正的数据库连接、容器启动、客户端初始化仍然留在 fixture 生命周期里处理。

机制对照表：

| 输入集合 | 展开层数 | 总 case 数 | collection 阶段完成 | 是否触发 fixture 初始化 |
| --- | --- | --- | --- | --- |
| `parametrize("x", [1,2,3])` | 1 | 3 | 是 | 按每个 case 需要时触发 |
| `fixture(params=[1,2])` | 1 | 2 | 是 | 是 |
| 两者组合 | 2 | 6 | 是 | 是 |
| `indirect=True` | 取决于参数层数 | 同参数总数 | 是 | 参数先进入 fixture |

真实工程例子：订单接口回归

假设我们测试 `POST /orders`，需要覆盖：

- 角色：`guest`、`user`、`admin`
- payload：合法、缺字段、越权字段
- 地区：`cn`、`us`

理论总 case 数：

$$
N = 3 \times 3 \times 2 = 18
$$

这 18 不是“循环 18 次”这么简单。它意味着：

- CI 会收集到 18 个独立样本
- 每个失败样本都能直接定位到输入组合
- 你可以只重跑某一个失败 case
- 你可以统计这次改动有没有让 case 数意外下降

这就是工程视角里参数化真正有价值的地方。

---

## 代码实现

先看一个最小可运行示例，覆盖普通 fixture 和普通参数化。

```python
import pytest

def add(a, b):
    return a + b

@pytest.fixture
def base():
    return 10

@pytest.mark.parametrize(
    "x, expected",
    [
        (1, 11),
        (5, 15),
        (0, 10),
    ],
    ids=["plus-1", "plus-5", "plus-0"],
)
def test_add_with_base(base, x, expected):
    assert add(base, x) == expected

def test_fixture_is_fresh(base):
    assert base == 10
```

这段代码里：

- `base` 是普通 fixture，负责提供基线值。
- `x, expected` 是参数集合，负责定义三组输入。
- `ids` 是给 case 起稳定名字，方便日志和 CI 定位。

再看一个带 `params=`、`indirect=True` 的例子：

```python
import pytest

class FakeClient:
    def __init__(self, backend):
        self.backend = backend

    def create_order(self, amount):
        if amount <= 0:
            return 400
        if self.backend == "readonly":
            return 503
        return 201

@pytest.fixture(params=["memory", "readonly"], ids=["mem", "ro"])
def backend_client(request):
    client = FakeClient(request.param)
    yield client
    # 这里可以放 teardown，例如关闭连接

@pytest.fixture
def user_client(request):
    backend = request.param
    return FakeClient(backend)

@pytest.mark.parametrize("amount, expected", [(1, 201), (-1, 400)], ids=["ok", "bad"])
def test_backend_client(backend_client, amount, expected):
    assert backend_client.create_order(amount) in {201, 400, 503}
    if amount <= 0:
        assert backend_client.create_order(amount) == expected

@pytest.mark.parametrize("user_client, expected", [("memory", 201), ("readonly", 503)], indirect=True)
def test_indirect_client(user_client, expected):
    assert user_client.create_order(10) == expected
```

这个例子里有四种关系：

- 普通 fixture：`user_client`、`backend_client`
- 带 `params` 的 fixture：`backend_client`
- 普通 `parametrize`：`amount, expected`
- `indirect=True`：字符串参数先进入 `user_client` 的 `request.param`

执行结果示意表：

| 测试 | 输入 | 实际初始化位置 | 期望结果 |
| --- | --- | --- | --- |
| `test_backend_client[mem-ok]` | backend=`memory`, amount=1 | `backend_client` | 201 |
| `test_backend_client[mem-bad]` | backend=`memory`, amount=-1 | `backend_client` | 400 |
| `test_backend_client[ro-ok]` | backend=`readonly`, amount=1 | `backend_client` | 503 |
| `test_backend_client[ro-bad]` | backend=`readonly`, amount=-1 | `backend_client` | 400 或按业务定义调整 |
| `test_indirect_client[memory]` | `"memory"` | `user_client(request.param)` | 201 |
| `test_indirect_client[readonly]` | `"readonly"` | `user_client(request.param)` | 503 |

下面这个 `python` 代码块不是 Pytest 文件，而是一个“可直接运行的玩具推导”，用来帮助理解 case 数计算：

```python
from itertools import product

fixture_params = [1, 2]
x_params = [10, 20, 30]

cases = list(product(fixture_params, x_params))

assert len(cases) == 6
assert cases[0] == (1, 10)
assert cases[-1] == (2, 30)

def passed(case):
    base, x = case
    return x + base == 11

results = [passed(case) for case in cases]
assert results.count(True) == 1
assert results.count(False) == 5
```

工程化写法还有三个常用点：

- 用 `conftest.py` 放共享 fixture，避免每个测试文件复制。
- 默认优先 `scope="function"`，只有明确确认资源昂贵且无状态污染时才提高到 `module` 或 `session`。
- 用 `ids=` 给参数命名，否则日志里可能出现难读的对象表示。

---

## 工程权衡与常见坑

先说结论：`parametrize` 适合枚举稳定输入，不适合承担外部状态读取；fixture 应默认低 scope、低共享，否则很容易产生 flake。flake 就是“同样代码有时过有时不过”的不稳定测试。

真实工程例子：订单回归测试

错误写法 1：把数据库查询结果直接生成进参数列表。问题是 collection 变重，甚至在收集阶段就依赖网络和数据库，导致：

- `pytest --collect-only` 变慢
- 本地无网时连收集都失败
- CI 很难区分“没收集到 case”和“测试执行失败”

错误写法 2：把一个可变共享对象放进 `session` scope fixture，让所有 case 共用。问题是某个 case 修改状态后，下一个 case 读到脏数据，失败位置和原因会错位。

常见坑表：

| 常见坑 | 现象 | 规避方式 |
| --- | --- | --- |
| 参数组合爆炸 | case 数从几十涨到几百几千，CI 变慢 | 只对真正独立维度做笛卡尔积，其余合并成业务 case |
| `xpass` 被忽略 | 预期失败的 case 突然通过，但没人处理 | 在 CI 单独关注 `xpass`，必要时开启 strict |
| 空参数集被静默处理 | 看起来流程跑了，实际关键测试没执行 | 检查 `empty_parameter_set_mark`，并监控 collected case 数 |
| 共享状态污染 | 单独跑通过，整套跑失败 | 默认 `function` scope；避免可变对象跨 case 共享 |
| 在参数里放可变对象 | 后续 case 读到被前一个 case 改过的数据 | 参数值尽量不可变，或在测试内复制 |
| fixture 过度封装 | 测试读起来像“黑箱脚本” | fixture 只做依赖准备，不吞掉断言和业务主流程 |

工程门槛建议也要明确，否则“会写 pytest”不等于“能进流水线”：

| 指标 | 建议门槛 |
| --- | --- |
| `collect count` | 不低于预期基线 |
| `xpass` | 必须为 0，除非明确调整预期 |
| 慢 fixture 初始化时间 | 不超过历史基线的 `+20%` |
| flake rate | 核心回归集接近 0 |
| 可回滚条件 | case 数下降、慢测试显著上升、共享污染出现 |

换句话说，Fixture 与参数化测试要放进完整工程流程讨论：输入输出是什么、评测口径是什么、回归风险是什么、触发回滚的条件是什么。只讲“这个功能很方便”没有工程价值。

---

## 替代方案与适用边界

不是所有测试都该上 `fixture + parametrize`。

三种常见写法可以这样理解：

| 问题特征 | 推荐方案 | 不推荐方案 | 原因 |
| --- | --- | --- | --- |
| 输入稳定且可枚举 | 纯 `parametrize` | 手写 `for` 循环 | 独立 case 更可观测 |
| 既有外部依赖又有多组输入 | `fixture + parametrize` | 把所有逻辑塞进 fixture | 依赖与输入职责分离 |
| 输入来自动态服务或 CLI 选项 | `pytest_generate_tests` 或自定义收集逻辑 | 硬编码到装饰器 | 输入源本身是动态的 |
| 只有一个局部小常量 | 局部变量 | 专门写 fixture | 没有复用价值 |
| case 之间不应做笛卡尔积 | 手工组织 case 或单独测试函数 | 多层叠加参数化 | 组合会失真或爆炸 |

什么时候不该用 `parametrize`：

- 输入来自实时外部服务
- 需要根据运行现场临时生成
- 多个维度之间不是独立组合关系
- 某些 case 需要完全不同的断言路径

什么时候不该用 fixture：

- 只是一个测试内一次性常量
- 不涉及生命周期管理
- 没有复用价值
- 抽出来反而让阅读成本更高

可以用一个简单决策树判断：

1. 这个东西是不是“测试依赖”而不是“业务输入”？
2. 如果是依赖，它是否需要复用或清理？
3. 如果是输入，它是否稳定、可枚举、值得成为独立 case？
4. 如果多个输入维度叠加，笛卡尔积是否仍然有业务意义？

只要第 4 步答案是否定的，就不要机械叠参数化。

一个很常见的误区是：为了“看起来高级”，把所有测试都改成参数化和 fixture 化。结果是：

- 简单测试变难读
- case 名称失真
- 调试路径变长
- 新人更难判断输入和依赖的边界

所以最佳实践不是“尽量多用”，而是“在输入、依赖、生命周期三者分离后再用”。

---

## 参考资料

本文依据官方文档和源码视图整理。关于 fixture 的定义、作用域、`yield` 清理、`request.param` 与参数化展开机制，结论主要来自官方文档和 API 说明；关于 CI 门槛、慢 fixture、回滚条件与组合爆炸的处理，属于工程实践归纳。

1. [How to use fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
2. [How to parametrize fixtures and test functions](https://docs.pytest.org/en/stable/how-to/parametrize.html)
3. [API Reference: `Metafunc.parametrize`](https://docs.pytest.org/en/latest/reference/reference.html)
4. [源码视图：`_pytest/fixtures.py`](https://docs.pytest.org/en/stable/_modules/_pytest/fixtures.html)
