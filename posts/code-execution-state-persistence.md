## 核心结论

代码执行里的“状态持久化”，可以先理解成一句话：前一次运行留下来的变量、对象、文件，下一次运行还能不能继续用。

在交互式执行环境里，最常见的两种设计是有状态执行和无状态执行：

| 维度 | 有状态执行 | 无状态执行 |
|---|---|---|
| 状态保留 | 保留前面运行过的变量、对象、缓存、打开的连接 | 每次运行都从干净环境开始 |
| 典型代表 | Jupyter Notebook 内核、长期存活容器、会话型代码执行器 | CI 任务、一次性脚本、每次新建容器的批处理 |
| 变量跨块复用 | 支持 | 不支持，必须显式传入 |
| 隐式依赖 | 高，后续代码可能偷偷依赖前面某个 cell | 低，依赖必须写全 |
| 可重现性 | 较弱，运行顺序会影响结果 | 较强，同样输入更容易得到同样输出 |
| 调试体验 | 强，适合探索式试错 | 弱一些，但行为更稳定 |
| 主要风险 | stale state、内存累积、顺序错乱、文件与内存不一致 | 重复初始化、性能损耗、交互效率低 |

最大差别不在“能不能保存文件”，而在“内存白板是否会被清空”。有状态执行像多人共用一块白板，后一个人看到的是前一个人留下的内容；无状态执行像每次都拿一张新表格，只看这次提交的输入。

因此，结论可以直接写成四条：

1. 有状态执行提升交互效率，因为变量可以跨 cell 复用，中间结果不用重复计算。
2. 有状态执行降低透明度，因为最终结果不仅取决于当前代码，还取决于“之前到底运行过什么、按什么顺序运行”。
3. 无状态执行更适合自动化、批处理、回归验证，因为依赖关系被迫显式化。
4. 工程上最稳妥的做法通常不是二选一，而是“短生命周期的有状态会话 + 显式文件持久化 + 定期全量重跑验证”。

---

## 问题定义与边界

这里讨论的不是普通的 `python main.py` 一次性脚本，而是“一个执行环境会持续存在一段时间”的场景，例如：

- Jupyter Notebook 的 kernel。kernel 就是实际执行 Python 代码的后台进程。
- 代码解释器容器。容器可以理解为隔离出来的一台小机器，里面有自己的内存、文件和运行时。
- 智能体的会话型执行器。会话型表示同一个任务可以连续多轮调用同一环境。

问题核心不是“有没有保存功能”，而是两件事：

1. 状态如何在不同代码块之间传播。
2. 这种传播什么时候会破坏结果的可重复性。

先看一个最小例子。假设你有两个 cell：

- Cell 1: `x = 10`
- Cell 2: `y = x + 5`

第一次按顺序运行，`y = 15`。后来你把 Cell 1 改成 `x = 2`，但忘了重跑 Cell 2。此时页面上“看起来”代码已经变了，但 kernel 内存里 `y` 还是旧值 `15`。这就是典型的 hidden state，也就是“页面里看不见，但运行环境里确实还在影响结果的隐藏状态”。

可以把依赖链画成最简流程：

```text
Cell n 的代码
    ↓
读取当前 kernel 状态 S(n-1)
    ↓
执行并产生增量 Delta(cell_n)
    ↓
写回新的 kernel 状态 S(n)
    ↓
Cell n+1 再读取 S(n)
```

这条链说明一个关键边界：只要后续 cell 读取的是“之前累积下来的状态”，它就不再只是“执行这段代码”，而是在“执行这段代码 + 依赖历史”。这也是为什么 Notebook 很适合探索，但不天然适合当作严格可复现的生产流水线。

还要再区分三类状态，否则讨论会混淆：

| 类型 | 例子 | 默认是否跨 cell 保留 | 默认是否跨会话保留 |
|---|---|---|---|
| 内存状态 | 变量、DataFrame、模型对象、缓存、导入模块 | 是 | 否 |
| 文件状态 | `csv`、图片、日志、模型文件 | 取决于目录是否仍存在 | 取决于存储介质 |
| 外部状态 | 数据库记录、对象存储文件、消息队列偏移量 | 与 cell 无关 | 通常是 |

这张表很重要，因为很多新手会把“文件还在”误认为“整个执行状态还在”。实际上不是。`result.csv` 仍然存在，不代表内存里的 `df`、模型对象、网络连接、随机数生成器状态也还存在。

另一个边界是会话生命周期。以 OpenAI 官方的 Code Interpreter 文档为例，容器被明确建议视为临时资源；如果 20 分钟未使用，容器会过期，关联数据会被丢弃。这意味着“会话内有状态”不等于“永久持久化”。它解决的是短期连续执行问题，不解决长期归档问题。

---

## 核心机制与推导

先把状态演进写成一个简单模型。记第 $n$ 个 cell 执行后的命名空间为 $S_n$。命名空间可以理解为“当前环境里所有变量名到对象的映射表”。那么状态转移可以写成：

$$
S_n = \operatorname{merge}(S_{n-1}, \Delta(\text{cell}_n))
$$

其中：

- $S_{n-1}$ 表示执行第 $n$ 个 cell 之前，执行器已经记住的全部状态。
- $\Delta(\text{cell}_n)$ 表示这个 cell 对状态造成的增量，比如新建变量、覆盖变量、导入模块、修改全局配置。
- $\operatorname{merge}$ 表示把增量合并回已有状态。若同名变量冲突，通常以后者覆盖前者。

如果只关心“哪些名字存在”，也可以近似写成：

$$
S_n \approx S_{n-1} \cup \Delta(\text{cell}_n)
$$

但工程上更准确的理解是“覆盖式合并”，不是严格数学上的集合并。

这个模型有两个直接推论。

第一，cell 的输出不是只由 cell 自身决定，而是由“当前代码”和“历史状态”共同决定：

$$
\text{output}_n = F(S_{n-1}, \text{cell}_n)
$$

这意味着同一段代码，在不同历史状态下，可能得到不同结果。

第二，只要执行顺序改变，状态就可能改变。若先执行 Cell A 再执行 Cell B，得到的是：

$$
S_{AB} = \operatorname{merge}(\operatorname{merge}(S_0, \Delta(A)), \Delta(B))
$$

若顺序颠倒，得到的是：

$$
S_{BA} = \operatorname{merge}(\operatorname{merge}(S_0, \Delta(B)), \Delta(A))
$$

当 $\Delta(A)$ 和 $\Delta(B)$ 互相依赖、覆盖同名变量、改变全局配置或写入同一路径时，通常有：

$$
S_{AB} \neq S_{BA}
$$

这就是为什么“执行顺序”本身就是程序语义的一部分。

把前面的玩具例子写成状态演进会更直观：

- 初始状态：$S_0 = \emptyset$
- Cell 1 执行 `x = 10`，得到 $S_1 = \{x: 10\}$
- Cell 2 执行 `y = x + 5`，得到 $S_2 = \{x: 10, y: 15\}$

后来你把 Cell 1 改成 `x = 2` 并重跑，只得到新的：

$$
S_1' = \{x: 2\}
$$

如果没有重跑 Cell 2，那么页面显示的是“代码里 `x = 2`”，但执行环境里可能仍然保留旧的：

$$
y = 15
$$

这就是 stale state。它不是“程序崩了”，而是“程序还能跑，但结果已经和当前代码语义不一致”。

这类错误难排查，原因在于它往往不会触发异常。程序仍然返回合法结果，只是结果属于旧世界。

再把问题推广一步。假设数据分析流程如下：

1. 前一个 cell 从数据库拉 200 万行数据到 DataFrame。
2. 后一个 cell 做清洗、聚合、训练、绘图。
3. 你修改了筛选条件，只重跑了前一个 cell。
4. 聚合结果、模型、图表如果没有重跑，就仍然基于旧数据。

这时错误类型不是语法错误，而是语义过期。对新手来说，最容易忽略的一点是：交互式环境里的“变量是否存在”不等于“变量是否仍然有效”。存在只是内存层面的事实；有效则取决于它是否仍和当前上游逻辑一致。

可以把这个判断写成一个更工程化的条件：

$$
\text{valid}(v_n) = \bigwedge_{u \in \operatorname{deps}(v_n)} \text{fresh}(u)
$$

意思是：某个结果变量 $v_n$ 是否有效，取决于它依赖的所有上游输入是否仍然是最新版本。只要其中任一上游变了而下游没重算，这个结果就不该再被信任。

---

## 代码实现

要管理状态，先要把“状态在哪里”拆清楚：

| 状态层 | 具体内容 | 生命周期 | 典型问题 |
|---|---|---|---|
| 内存状态 | 变量、对象、缓存、导入模块、随机数种子 | 随 kernel/容器存活 | 隐式依赖、内存泄漏、结果过期 |
| 文件状态 | CSV、图片、模型、日志、中间产物 | 随目录和容器策略存活 | 过期丢失、版本漂移、文件名冲突 |
| 外部状态 | 数据库、对象存储、消息队列 | 独立于当前执行器 | 一致性、幂等性、权限问题 |

下面用一个可直接运行的 Python 例子，模拟“有状态 cell 执行器 + 文件持久化 + 依赖追踪”。例子不解析真实 Python 语法，而是用函数代表 cell，重点展示状态转移和 stale state 检测。

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from typing import Callable, Dict, Any, List, Set

Serializable = (int, float, str, bool, list, dict, type(None))


@dataclass
class CellRecord:
    name: str
    depends_on: List[str]
    produced_keys: List[str]
    version: int
    executed_at: float


@dataclass
class SessionExecutor:
    persist_root: Path
    state: Dict[str, Any] = field(default_factory=dict)
    cell_versions: Dict[str, int] = field(default_factory=dict)
    key_producers: Dict[str, str] = field(default_factory=dict)
    key_upstream_versions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    execution_log: List[CellRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.persist_root.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        name: str,
        cell_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        depends_on: List[str] | None = None,
    ) -> Dict[str, Any]:
        depends_on = depends_on or []
        current_state = self.state.copy()
        delta = cell_fn(current_state)

        if not isinstance(delta, dict):
            raise TypeError(f"{name} must return dict, got {type(delta).__name__}")

        version = self.cell_versions.get(name, 0) + 1
        self.cell_versions[name] = version

        upstream_versions = {
            dep: self.cell_versions.get(dep, 0)
            for dep in depends_on
        }

        for key, value in delta.items():
            self.state[key] = value
            self.key_producers[key] = name
            self.key_upstream_versions[key] = upstream_versions.copy()

        self.execution_log.append(
            CellRecord(
                name=name,
                depends_on=depends_on,
                produced_keys=sorted(delta.keys()),
                version=version,
                executed_at=time.time(),
            )
        )

        self._persist_snapshot()
        return delta

    def is_stale(self, key: str) -> bool:
        if key not in self.key_upstream_versions:
            return False

        recorded = self.key_upstream_versions[key]
        for dep, old_version in recorded.items():
            if self.cell_versions.get(dep, 0) != old_version:
                return True
        return False

    def stale_keys(self) -> Set[str]:
        return {key for key in self.state if self.is_stale(key)}

    def _persist_snapshot(self) -> None:
        serializable_state = {
            key: value
            for key, value in self.state.items()
            if isinstance(value, Serializable)
        }

        snapshot = {
            "state": serializable_state,
            "cell_versions": self.cell_versions,
            "key_producers": self.key_producers,
            "stale_keys": sorted(self.stale_keys()),
            "execution_log": [
                {
                    "name": record.name,
                    "depends_on": record.depends_on,
                    "produced_keys": record.produced_keys,
                    "version": record.version,
                    "executed_at": int(record.executed_at),
                }
                for record in self.execution_log
            ],
        }

        target = self.persist_root / "session_snapshot.json"
        target.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def cell_load_data(_: Dict[str, Any]) -> Dict[str, Any]:
    rows = [
        {"user": "u1", "amount": 100},
        {"user": "u2", "amount": 200},
        {"user": "u3", "amount": 50},
    ]
    return {"raw_rows": rows}


def cell_build_metric(state: Dict[str, Any]) -> Dict[str, Any]:
    total = sum(item["amount"] for item in state["raw_rows"])
    return {"total_amount": total}


def cell_load_data_modified(_: Dict[str, Any]) -> Dict[str, Any]:
    rows = [
        {"user": "u1", "amount": 100},
        {"user": "u2", "amount": 200},
    ]
    return {"raw_rows": rows}


if __name__ == "__main__":
    executor = SessionExecutor(Path("/tmp/state_demo"))

    executor.execute("load_data", cell_load_data)
    executor.execute("build_metric", cell_build_metric, depends_on=["load_data"])

    assert executor.state["total_amount"] == 350
    assert executor.is_stale("total_amount") is False

    executor.execute("load_data", cell_load_data_modified)

    assert len(executor.state["raw_rows"]) == 2
    assert executor.state["total_amount"] == 350
    assert executor.is_stale("total_amount") is True

    executor.execute("build_metric", cell_build_metric, depends_on=["load_data"])

    assert executor.state["total_amount"] == 300
    assert executor.is_stale("total_amount") is False

    print("current_state =", executor.state)
    print("stale_keys =", executor.stale_keys())
    print("snapshot =", executor.persist_root / "session_snapshot.json")
```

这个例子说明四件事。

第一，`state` 就是有状态执行器的内存命名空间。每次 `execute` 都会读取旧状态，再写入新状态。

第二，`session_snapshot.json` 是显式文件持久化。它记录的是“当前状态摘要”和“执行轨迹”，而不是把整个 Python 进程原样冻结。工程上这是更现实的做法，因为很多对象根本不能直接序列化，比如数据库连接、线程锁、模型句柄、打开的文件描述符。

第三，`is_stale(key)` 说明 stale state 不是玄学，而是可以检测的：如果某个结果依赖的上游 cell 版本已经变化，但该结果没有重新生成，那么这个结果就是过期的。

第四，文件持久化不能替代依赖管理。即使你把当前状态写进了 JSON，`total_amount` 仍可能是旧值。因为“被保存下来”不等于“仍然正确”。

如果想再看一个更接近真实 Notebook 的最小示例，可以用下面这段纯 Python 演示“改上游、不重跑下游”：

```python
state = {}

def cell1():
    state["x"] = 10

def cell2():
    state["y"] = state["x"] + 5

cell1()
cell2()
print(state)  # {'x': 10, 'y': 15}

state["x"] = 2
print(state)  # {'x': 2, 'y': 15}

cell2()
print(state)  # {'x': 2, 'y': 7}
```

它简单，但足够说明问题：在有状态环境里，下游结果不会因为你改了上游变量而自动刷新。Notebook 不是电子表格，不具备默认的依赖重算机制。

如果把这个模型往真实工程再推进一步，一个更可靠的执行器通常会补三类元数据：

| 元数据 | 用途 | 不记录会怎样 |
|---|---|---|
| 执行顺序 | 解释状态是怎么形成的 | 很难复盘“当前结果从哪里来” |
| 依赖关系 | 判断哪些结果需要重算 | 结果可能悄悄过期 |
| 持久化快照 | 跨步骤共享中间产物 | 故障后无法恢复可审计现场 |

真实工程例子可以看会话型数据任务：

1. 从 API 拉原始数据。
2. 清洗后保存为 `cleaned.csv`。
3. 生成报告图表 `chart.png`。
4. 下一轮继续读取这些文件回答问题。

这个流程的交互体验很强，因为不必每轮都重新抓数、重新画图。但从系统设计角度看，它已经不是纯函数式执行，而是会话状态机。状态机的重点不是“能不能多跑几次”，而是“系统每走一步，内部状态都发生了什么变化”。

---

## 工程权衡与常见坑

有状态执行的优势非常实际：减少重复计算、支持探索式分析、适合多步 ETL 和交互调试。但它带来的坑也同样具体，不是抽象理论问题。

先看最常见的问题和对应规避：

| 常见坑 | 表现 | 根因 | 规避方式 |
|---|---|---|---|
| 隐式依赖 | 代码改了，但输出没变或半变 | 依赖 cell 没重跑 | `Restart & Run All`，或显式声明依赖 |
| stale state | 页面代码与内存状态不一致 | 历史状态残留 | 改基础变量后全量重跑 |
| 内存累积 | 反复运行后 RAM 持续上涨 | 大对象、缓存、绘图库对象未释放 | 重启 kernel，把大中间结果写盘 |
| 文件误当永久存储 | 会话结束后文件丢失 | 容器是临时资源 | 及时下载或同步到外部存储 |
| 顺序耦合 | 从中间某个 cell 开始运行就报错 | 前置状态未建立 | 把初始化逻辑集中到前几个 cell |
| 结果不可审计 | 不知道当前结果由哪些步骤产生 | 缺少执行日志和快照 | 保存状态摘要、版本和时间戳 |
| 随机性漂移 | 同一 notebook 重跑结果变了 | 随机种子、采样、时间函数未固定 | 固定种子，隔离时间依赖 |
| 外部状态污染 | 第二次运行结果不同 | 数据库已写入、缓存已命中、文件被覆盖 | 幂等设计，区分读写路径 |

对新手来说，最容易误解的是前两项。

隐式依赖的意思不是“代码写得神秘”，而是“这段代码真正需要的前置条件，没有在当前 cell 里说清楚”。例如当前 cell 只写了：

```python
model.fit(X_train, y_train)
```

但 `X_train`、`y_train`、`model` 都来自更早之前的多个 cell。这样一来，这个 cell 表面上只有一行，实际上却依赖很长的历史链条。

stale state 则更隐蔽。它最大的危险不是报错，而是不报错。图能正常出，模型能正常训，表格能正常打印，但语义已经过期。

内存问题也不是少数个案。HoloViews 的一个历史 issue 记录了在 Jupyter 里反复执行复杂绘图 cell 后，kernel 内存持续增长的现象。这里不必把问题理解成“Jupyter 一定泄漏”，更准确的理解是：长期存活的交互式环境，会把很多对象生命周期拉长，某些库在重复构图、缓存、引用管理上处理不当时，内存问题会被放大。

所以工程上比较稳妥的做法通常有四条。

1. 把 `Restart & Run All` 当作基线检查  
只要全量重跑不过，就说明当前文档并不自洽。它可能能“继续用”，但已经不适合作为可复现资产。

2. 大对象尽量落盘，小对象再留内存  
原始数据、特征矩阵、模型中间产物、图表文件，尽量写到目录里；小型参数、标量统计量、轻量配置，才保留在内存中。这样做的目的不是省事，而是让依赖链更可见。

3. 缩短状态窗口  
状态窗口可以理解为“允许同一个执行环境持续存在的时长”。窗口越长，交互越方便；但隐藏状态越多，内存和排障成本也越高。OpenAI 官方文档对 Code Interpreter 的设计就是明确的有限窗口：容器 20 分钟未使用即过期。这个边界本身就在表达一个系统设计选择，即它更像短期工作区，而不是长期数据库。

4. 把关键逻辑从 cell 历史里抽出来  
探索可以在 cell 里完成，但稳定步骤最好沉淀为函数、脚本或 pipeline。原因很简单：cell 历史适合试验，不适合治理。

可以再用一张表把“探索效率”和“工程可靠性”的关系讲清楚：

| 做法 | 探索效率 | 可审计性 | 可重现性 |
|---|---|---|---|
| 全靠内存变量连续跑 | 高 | 低 | 低 |
| 内存变量 + 文件快照 | 中高 | 中 | 中 |
| 脚本化 + 无状态重跑 | 中 | 高 | 高 |

这也是为什么很多团队的实际工作流不是单一模式，而是分阶段模式：前期允许有状态试验，后期要求无状态验证。

---

## 替代方案与适用边界

有状态和无状态没有绝对优劣，只有适配场景的差异。

先看一张小表：

| 方案 | 操作体验 | 可重复性 | 资源利用 | 适合场景 |
|---|---|---|---|---|
| 长期有状态 kernel | 最强 | 最弱 | 容易累积内存和隐藏状态 | 探索式分析、教学演示、交互调试 |
| 短生命周期容器 + 会话目录 | 较强 | 中等 | 可控，需处理过期 | 智能体任务、多步数据处理、短会话工作流 |
| 完全无状态执行 | 较弱 | 最强 | 最稳定 | CI、批处理、定时任务、生产流水线 |

如果目标是“边看边改、边跑边试”，有状态模式更合适。典型场景：

- 数据分析时先抽样，再改图表，再调特征。
- 智能体分多轮生成中间文件，再继续处理。
- 教学或实验中需要逐步展示推导过程。

如果目标是“任何时候重跑都应该得到同样结果”，无状态模式更合适。典型场景：

- 自动化训练流水线。
- 每日批处理任务。
- 回归测试和结果审计。
- 对外提供可复现报告。

边界条件需要说清楚，否则容易把“会话内可用”误判成“系统级持久化”。

第一，有限时长容器不是严格意义上的长期持久化。以 OpenAI 官方的 Code Interpreter 文档为例，容器 20 分钟未使用即过期。这个数字不是实现细节，而是设计边界：说明它只能被当作会话工作区，不能被当作长期数据库。

第二，`/mnt/data` 这类目录的价值，本质上是“把本来只存在内存里的中间产物，转换成在当前会话窗口内可见、可下载、可复用的文件状态”。这比纯内存变量更可追踪，但仍然不是长期归档方案。长期归档、跨会话恢复、版本治理，仍然要交给对象存储、数据库或代码仓库。

第三，有状态系统一旦规模变大，就必须主动把隐式依赖转成显式依赖。否则它会慢慢退化成“只有原作者知道如何正确重跑”的个人工作台，而不是可交接的工程资产。

因此，更实用的组合策略通常是：

- 交互开发阶段：用有状态环境提高试验速度。
- 提交和发布前：用 `Restart & Run All` 或无状态脚本验证。
- 中间产物：写入会话目录，同时同步到外部持久化存储。
- 关键依赖：写成脚本、函数或 pipeline，不只留在 cell 历史里。
- 外部副作用：单独治理幂等性、回滚和审计，不把它们混进 Notebook 状态里。

这套做法的本质，是把“探索效率”和“结果可复现性”分阶段处理，而不是试图让同一种执行模式同时把两件事做到极致。

---

## 参考资料

1. OpenAI, “Code Interpreter”  
   https://platform.openai.com/docs/guides/tools-code-interpreter/  
   支持点：官方说明容器应被视为临时资源；20 分钟未使用会过期；过期后数据不可恢复；会话期间可创建和下载文件。

2. Jupyter, “What is a kernel?” 与 Notebook/Server 相关文档  
   https://docs.jupyter.org/  
   支持点：说明 kernel 是独立执行进程，负责保存解释器状态；前端与执行内核分离。

3. Programming Central, “Why Jupyter Notebooks Are Killing the Traditional Python Script”  
   https://dev.to/programmingcentral/why-jupyter-notebooks-are-killing-the-traditional-python-script-1l6b  
   支持点：对 Jupyter 的持久状态、kernel 机制和 stale state 风险给出了面向初学者的解释。

4. HoloViews Issue #1821, “Memory leak / increasing usage in Jupyter for repeated cell execution”  
   https://github.com/holoviz/holoviews/issues/1821  
   支持点：给出交互式 notebook 中重复执行 cell 后内存持续增长的真实案例，说明长期有状态环境会放大对象生命周期与缓存管理问题。

5. nbformat Documentation  
   https://nbformat.readthedocs.io/  
   支持点：Notebook 文件本身是结构化文档，保存代码、文本和输出，但它并不等于完整的运行时内存快照。这有助于理解“文档可见内容”和“执行环境真实状态”之间的差异。
