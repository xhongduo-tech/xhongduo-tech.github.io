## 核心结论

代码执行 Agent 的包管理，最稳妥的做法不是只选一种，而是采用“三腿策略”：

1. 预装常用包。预装，白话说，就是把高频依赖提前做进基础环境，启动后直接可用。
2. 按需安装低频包。按需安装，白话说，就是任务真需要时再临时补包，而不是每次启动都全装一遍。
3. 安全扫描与白名单兜底。白名单，白话说，就是只允许事先认可的内容通过；扫描，白话说，就是在安装前先做风险检查。

这三者解决的是三类不同问题：

| 目标 | 主要手段 | 直接收益 | 主要代价 |
|---|---|---|---|
| 启动快 | 预装高频包 | 首次响应稳定 | 镜像更大、构建更慢 |
| 覆盖广 | 按需安装低频包 | 长尾任务能完成 | 延迟不稳定、流程更复杂 |
| 风险可控 | 扫描 + 白名单 | 减少恶意包与越权安装 | 需要规则、审批和维护 |

对“零基础到初级工程师”最重要的一点是：不要把 `pip install` 当成默认路径。默认路径应当是“先用预装环境跑”，只有缺包时才进入“扫描 -> 审批/白名单 -> 安装”的分支。

一个最小结论可以写成：

$$
总体收益 \approx 预装覆盖率 \times 复用次数 \times 单次节省时间 - 按需安装成本 - 安全检查成本
$$

如果常见任务都集中在 `numpy`、`pandas`、`matplotlib` 这类数据科学库，那么把它们放入模板或镜像，通常比每个会话临时安装更划算。E2B 官方文档明确支持两条路：在模板里预装，或者在运行时安装。Azure SRE Agent 则更严格，官方文档直接说明 `pip install` 不可用，因此这类环境只能高度依赖预装。

| 策略 | 首次启动体验 | 后续复用收益 | 安全控制难度 | 适合场景 |
|---|---:|---:|---:|---|
| 只预装 | 快 | 高 | 中 | 任务类型稳定、强安全约束 |
| 只按需安装 | 慢 | 低 | 高 | 实验环境、任务分布极不稳定 |
| 预装 + 按需 + 安全 | 中到快 | 高 | 中 | 通用代码执行 Agent |

可以把它记成一句工程原则：**默认走快路径，例外走受控慢路径。**

---

## 问题定义与边界

这里讨论的“包管理策略”，不是普通开发机上的依赖管理，而是**代码执行沙箱**里的依赖管理。沙箱，白话说，就是给 Agent 一个隔离出来的运行环境，代码能跑，但权限被严格限制。

问题本质是三件事同时成立：

1. Agent 要能完成大多数常见任务，比如数据清洗、画图、CSV 处理、简单数值计算。
2. Agent 不能因为缺包就频繁卡在安装阶段，导致响应时间不可预测。
3. Agent 不能随意引入高风险能力，比如网络探测、子进程拉起、系统命令执行。

如果只看“能不能跑”，你会自然偏向“缺什么装什么”。但 Agent 不是人工开发机。人工开发机允许开发者自己判断依赖是否可信、是否值得装；Agent 场景里，发起安装动作的往往是模型，而模型的判断并不等于安全策略。这里的核心不是“装包技术”，而是“谁有权限触发安装，以及安装前后怎么控风险”。

因此边界要先划清：

| 类别 | 典型内容 | 默认策略 |
|---|---|---|
| 高频安全包 | `numpy`、`pandas`、`matplotlib` | 预装 |
| 中低频业务包 | `openpyxl`、`lxml`、`pyyaml` | 按需安装，但要走扫描 |
| 高风险能力相关 | 依赖安装脚本复杂、会调用系统能力的包 | 白名单或人工审批 |
| 明确受限能力 | `subprocess`、`os.system`、`socket` 一类能力链路 | 在沙箱或执行策略层阻断 |

这里要特别澄清一个常见误解：`os`、`socket` 这些通常是 Python 标准库模块，不是通过 `pip install` 获取的第三方包。真正的风险控制点不是“禁止安装这些名字”，而是**禁止代码在沙箱中调用相关危险能力**，或者禁止引入会偷偷触发这些能力的恶意第三方包。

还可以把边界拆成两层：

| 控制层 | 负责什么 | 典型措施 |
|---|---|---|
| 依赖层 | 决定能装什么第三方包 | 预装、白名单、扫描、审批 |
| 执行层 | 决定代码运行时能做什么 | 禁网、禁子进程、禁文件越界、资源限额 |

很多新人会把这两层混在一起，结果出现两个错误判断：

1. 以为“不让 `pip install`”就等于安全。
2. 以为“允许装的是安全包”就等于运行安全。

这两个判断都不成立。原因很简单：安装限制控制的是“能引入什么依赖”，执行限制控制的是“代码最终能做什么动作”。二者都要有。

“玩具例子”可以这样理解。你做一个面向初学者的数据分析 Agent，90% 的任务只是“读 CSV、算均值、画折线图”。这类任务如果每次都先执行：

```bash
pip install pandas numpy matplotlib
```

那用户每次都在等准备环境，而不是等结果。相反，如果这些库在模板里已经就绪，Agent 启动后就能直接读文件和画图。

“真实工程例子”则更严格。Azure SRE Agent 官方文档说明，它的代码解释器沙箱不允许网络访问、不允许创建子进程，也不允许 `pip install` 或 `conda install`。这意味着在这类环境里，你根本没有“运行时补包”这个选项，策略天然退化成“预装优先 + 严格限制”。

所以这篇文章的边界很明确：我们讨论的是**有代码执行能力的 Agent 如何设计依赖可用性与安全性**，不是讨论 Python 项目如何写 `requirements.txt`，也不是讨论本地开发机怎么提速。

---

## 核心机制与推导

这套策略能否划算，可以用一个简化公式看：

$$
效能 = C \times R \times T_s - T_p - T_g
$$

其中：

- $C$：预装覆盖率。白话说，就是预装环境能直接满足多少比例的任务。
- $R$：复用次数。白话说，就是同一套预装环境会被重复用多少次。
- $T_s$：每次跳过安装节省的时间。
- $T_p$：按需安装延迟。
- $T_g$：安全扫描或审批带来的额外延迟。

这个公式不是严格的学术模型，但足够指导工程判断。它的价值不在于算出绝对精确的数字，而在于提醒你：**预装是否划算，不只看单次安装时间，还要看任务集中度和复用频率。**

如果要再细一点，可以把期望耗时写成：

$$
E[T] = C \times T_{pre} + (1-C) \times (T_{pre} + T_p + T_g)
$$

其中：

- $T_{pre}$：预装环境启动并开始执行任务的基线时间。
- $(1-C)$：落入“缺包分支”的概率。

这个式子表达的是：并不是所有任务都走安装分支，只有没被预装覆盖到的那部分任务才会额外付出安装和检查成本。因此真正应该优化的是**让高频任务尽量留在 $C$ 的覆盖范围内**。

### 玩具例子：80% 覆盖率是否值得预装

假设：

- 模板预装覆盖率 $C = 0.8$
- 同一模板平均复用 $R = 5$
- 每次避免一次 `pip install` 可省 $T_s = 8.5s$
- 安全扫描平均消耗 $T_g = 1s$
- 低频任务仍需一次按需安装，代价 $T_p = 8.5s$

则：

$$
效能 = 0.8 \times 5 \times 8.5 - 8.5 - 1 = 24.5
$$

这里的 `24.5` 可以理解成“在一个复用周期内净赚约 24.5 秒”。这说明只要高频任务足够集中，预装就不是“浪费镜像空间”，而是在买稳定延迟。

如果把覆盖率换成 30%，其他条件不变：

$$
效能 = 0.3 \times 5 \times 8.5 - 8.5 - 1 = 3.25
$$

还能赚，但收益已经明显下降。如果覆盖率继续下降，或者模板几乎不复用，那么预装的边际价值就会快速变小。工程上这意味着一条很实用的判断标准：

| 情况 | 解释 | 策略倾向 |
|---|---|---|
| 高频任务集中、模板复用高 | 大多数会话都在用同一批包 | 扩大预装 |
| 任务很散、模板复用低 | 每次需求都不太一样 | 缩小预装，保留按需装 |
| 安全要求极高 | 不能接受在线引入第三方依赖 | 压缩按需装，转为离线审核后预装 |

再看冷缓存和热缓存。冷缓存，白话说，就是本地没有已下载的 wheel 或源码包，要重新联网拉取；热缓存，白话说，就是之前下过，这次直接复用缓存。PythonSpeed 给出的一个实测例子中，`pip install` 冷缓存约 8.5 秒，热缓存约 6.3 秒。它说明一件事：即使缓存热了，运行时安装仍然有明显成本。

| 安装方式 | 缓存状态 | 墙钟时间 |
|---|---|---:|
| `pip install` | 冷缓存 | 8.5s |
| `pip install` | 热缓存 | 6.3s |

因此，“每次缺包就装”虽然灵活，但它有持续存在的弹性税。这个“税”至少包括三部分：

| 成本项 | 说明 |
|---|---|
| 下载成本 | 拉取 wheel、源码包和依赖元数据 |
| 解析成本 | 依赖解析、版本选择、环境检查 |
| 安全成本 | 扫描、审批、白名单匹配、审计记录 |

### 安全机制怎么接进来

Open Interpreter 的 `safe mode` 官方文档强调两点：禁用自动执行，以及用 `semgrep` 扫描生成代码。文档还给出一个建议做法：在自定义指令里要求模型在 `pip` 或 `npm` 安装前先用 `guarddog` 扫描包。这里要准确表述：**`guarddog` 不是 Open Interpreter safe mode 默认内建的安装网关，而是官方文档建议你接入的额外流程。**

所以一个更准确的机制图是：

1. 模型先尝试用预装依赖完成任务。
2. 缺包时，提出安装请求。
3. 执行层先跑包扫描，例如 `guarddog pypi scan 包名`。
4. 通过白名单或扫描规则后，才允许安装。
5. 如果环境像 Azure SRE Agent 一样完全禁装，则直接回退到“换方案”或“提示当前环境不支持”。

可以把这条链路写成一个判定式：

$$
允许安装 = 环境允许 \land 包在白名单内 \land 扫描通过
$$

只要有一个条件不成立，就不安装。这个逻辑比“模型觉得需要所以就装”稳定得多，也更容易审计。

---

## 代码实现

先给一个可以直接运行的 Python 玩具程序，用来估算“预装是否值得”。`assert` 的作用，白话说，就是把预期条件写进代码里，条件不满足就立刻报错。

```python
from dataclasses import dataclass


@dataclass
class StrategyInput:
    coverage: float      # 预装覆盖率，0~1
    reuse_count: int     # 模板复用次数
    pip_delay: float     # 一次按需安装延迟（秒）
    scan_delay: float    # 一次安全扫描延迟（秒）


def strategy_score(cfg: StrategyInput) -> float:
    assert 0 <= cfg.coverage <= 1
    assert cfg.reuse_count >= 0
    assert cfg.pip_delay >= 0
    assert cfg.scan_delay >= 0

    saved = cfg.coverage * cfg.reuse_count * cfg.pip_delay
    cost = cfg.pip_delay + cfg.scan_delay
    return saved - cost


def expected_runtime(base_time: float, cfg: StrategyInput) -> float:
    """
    期望总耗时 = 命中预装分支的时间 + 落入缺包分支的时间
    """
    assert base_time >= 0
    miss_rate = 1 - cfg.coverage
    return base_time + miss_rate * (cfg.pip_delay + cfg.scan_delay)


cfg = StrategyInput(
    coverage=0.8,
    reuse_count=5,
    pip_delay=8.5,
    scan_delay=1.0,
)

score = strategy_score(cfg)
runtime = expected_runtime(base_time=2.0, cfg=cfg)

assert round(score, 1) == 24.5
assert round(runtime, 1) == 3.9

print(f"strategy_score={score:.1f}s")
print(f"expected_runtime={runtime:.1f}s")
```

这段代码可以直接运行，输出大致为：

```text
strategy_score=24.5s
expected_runtime=3.9s
```

它表达的不是“真实生产环境一定是 3.9 秒”，而是告诉你一个方法：**先把覆盖率、复用次数、安装延迟拆开，再讨论是否预装。**

接着看模板层实现。E2B 官方现在推荐用模板 API，也兼容从 Dockerfile 构建模板。对初学者来说，Dockerfile 最直观。Dockerfile，白话说，就是描述镜像如何构建的脚本。

```dockerfile
FROM e2bdev/code-interpreter:latest

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    openpyxl
```

这段代码表达的是：把 `numpy`、`pandas`、`matplotlib`、`openpyxl` 烧进镜像层，沙箱启动后直接可用。对于“数据分析 Agent”或“报表 Agent”这种高频场景，这是最值钱的一步。

如果你希望版本更稳定，建议显式写版本范围，而不是直接吃最新版本：

```dockerfile
FROM e2bdev/code-interpreter:latest

RUN pip install --no-cache-dir \
    "numpy>=1.26,<3" \
    "pandas>=2.2,<3" \
    "matplotlib>=3.8,<4" \
    "openpyxl>=3.1,<4"
```

这样做的意义是：减少“今天能跑、下周构建失败”的概率。Agent 平台通常追求稳定重复，而不是每次都追最新包。

运行时的一个可执行简化示例可以这样写。下面这段代码不依赖真实 `guarddog`，而是用一个占位扫描函数模拟“白名单 + 扫描 + 环境能力”三重判断，便于初学者先理解流程：

```python
from dataclasses import dataclass


PREINSTALLED = {"numpy", "pandas", "matplotlib"}
ALLOWLIST = {"openpyxl", "lxml", "pyyaml"}
DENYLIST = {"tensorflow", "torch"}  # 例子：体积大或不允许在线装


@dataclass
class EnvironmentPolicy:
    runtime_install_enabled: bool = True
    require_scan: bool = True


def mock_scan_package(pkg_name: str) -> dict:
    """
    用玩具逻辑模拟安全扫描结果。
    真实工程里这里可以接 guarddog 或自研扫描器。
    """
    suspicious_keywords = {"hack", "exploit", "malware"}
    issues_found = int(any(word in pkg_name.lower() for word in suspicious_keywords))
    return {"package": pkg_name, "issues_found": issues_found}


def ensure_package(pkg_name: str, policy: EnvironmentPolicy) -> str:
    if pkg_name in PREINSTALLED:
        return "ready: preinstalled"

    if not policy.runtime_install_enabled:
        raise RuntimeError("runtime installation is disabled in this environment")

    if pkg_name in DENYLIST:
        raise PermissionError(f"{pkg_name} is explicitly denied")

    if pkg_name not in ALLOWLIST:
        raise PermissionError(f"{pkg_name} is not in allowlist")

    if policy.require_scan:
        report = mock_scan_package(pkg_name)
        if report["issues_found"] > 0:
            raise RuntimeError(f"{pkg_name} failed security scan")

    # 真实工程里此处才会调用 pip install
    return "approved: install now"


policy = EnvironmentPolicy(runtime_install_enabled=True, require_scan=True)

assert ensure_package("numpy", policy) == "ready: preinstalled"
assert ensure_package("openpyxl", policy) == "approved: install now"

try:
    ensure_package("unknown-lib", policy)
except PermissionError as e:
    print(e)

try:
    ensure_package("hack-toolkit", policy)
except PermissionError as e:
    print(e)
```

这段示例解决了原始伪代码的两个问题：

1. 它本身能运行，不依赖外部命令存在。
2. 它把判断流程拆开了，新手更容易看懂“为什么这个包能装、那个包不能装”。

如果换成更接近真实工程的流程，逻辑通常是这样：

| 步骤 | 执行者 | 目的 |
|---|---|---|
| 识别依赖缺失 | Agent 或执行器 | 判断是不是缺包导致失败 |
| 生成安装提案 | Agent | 提出“需要什么包、为什么需要” |
| 校验环境能力 | 执行器 | 判断当前平台是否允许运行时安装 |
| 白名单匹配 | 策略层 | 判断包名是否被允许 |
| 扫描 | 安全工具 | 检查供应链风险 |
| 执行安装 | 沙箱执行器 | 真正调用安装命令 |
| 记录审计 | 平台层 | 记录谁在什么任务中装了什么 |

再进一步，若你接的是 Open Interpreter 一类执行器，核心不是让模型自己决定一切，而是把约束前置到系统策略里：

1. 默认关闭自动执行。
2. 缺包时只允许提案，不允许直接安装。
3. 安装请求必须先经过扫描工具。
4. 执行器最终依据白名单和环境能力决定是否放行。

真实工程例子可以这样落地。假设你做一个企业内报表 Agent：

- 在基础镜像里预装 `numpy`、`pandas`、`matplotlib`、`openpyxl`
- 扫描和审批后，才允许低频包进入镜像更新计划
- 在线运行态默认不开放任意 `pip install`
- 对于完全受限的平台，例如 Azure SRE Agent，则只使用平台预装库完成任务

这种设计的重点不是“功能最多”，而是“最常见任务最快，异常路径最可控”。

---

## 工程权衡与常见坑

包管理策略不是越灵活越好，而是要看你在买什么、放弃什么。

第一类权衡是**镜像体积 vs 启动速度**。把常用数据科学库预装进镜像，会增加镜像体积，现实里常常是数百 MB 级别；但如果这些库是高频任务的基础，那换来的就是更稳定的首响应时间。对 Agent 产品来说，稳定常常比理论最小镜像更重要。

第二类权衡是**任务覆盖率 vs 安全面**。允许任意安装第三方包，任务覆盖率会上升，但攻击面也会上升。很多风险并不来自“包名可疑”，而来自安装脚本、依赖链、隐蔽下载、执行外部命令等行为。

第三类权衡是**平台能力 vs 统一策略**。如果底层平台像 Azure SRE Agent 一样禁用 `pip install`、禁用网络、禁用子进程，那你的上层 Agent 根本不应设计“自动补包”逻辑，否则只是制造失败路径。

第四类权衡是**灵活性 vs 可审计性**。线上随时装包最灵活，但很难复盘“某次失败用了哪个版本、谁批准的、为什么安装成功”。把新增依赖转成离线审核后再预装，灵活性下降，但审计能力显著提升。

常见坑和规避如下：

| 常见坑 | 表现 | 规避措施 |
|---|---|---|
| 忽略冷缓存延迟 | 首次运行卡在下载依赖 | 把高频包做进模板 |
| 误把热缓存当常态 | 本地测得很快，线上新容器却慢 | 以冷启动数据评估 |
| 放任模型任意装包 | 安全风险暴露、依赖不可控 | 强制扫描 + 白名单 |
| 把标准库和第三方包混为一谈 | 风险点判断错误 | 区分“能力限制”和“安装限制” |
| 在禁装环境里还设计补包分支 | 线上稳定报错 | 启动时检测环境能力 |
| 预装范围过大 | 镜像臃肿、构建慢 | 只预装高频且稳定的包 |
| 不锁版本或版本范围 | 今天能跑，明天构建失败 | 固定版本或约束版本区间 |
| 不记录安装审计 | 出问题后无法追查 | 记录包名、版本、任务、审批结果 |

一个典型坑是：开发者本地跑通了，因为本地 pip 缓存是热的；一上线到短生命周期容器，每次都是冷缓存，Agent 每轮都在等下载。这就是为什么 PythonSpeed 的冷缓存 8.5 秒、热缓存 6.3 秒差异很有参考意义。它不是告诉你“8.5 秒一定成立”，而是告诉你“运行时安装的时间并不稳定”。

另一个典型坑是误解 Open Interpreter 的 safe mode。safe mode 官方能力重点是“禁自动执行 + 扫描生成代码”。如果你需要“安装前检查第三方包”，应该把 `guarddog` 之类扫描流程显式接入，而不是假设 safe mode 已经替你完成了全部供应链安全。

还有一个经常被忽略的问题是“补包失败后的回退策略”。真正稳定的 Agent，不只要定义“什么时候能装”，还要定义“装不了怎么办”。常见回退路径有三类：

| 回退路径 | 适用情况 | 例子 |
|---|---|---|
| 改写方案 | 能用已预装库完成同一目标 | 用 `csv` 标准库替代 `pandas` 做简单读取 |
| 显式报错 | 环境能力不支持时 | 当前沙箱禁用运行时安装 |
| 延后到镜像升级 | 低频但确实需要 | 把新依赖加入下一个模板版本 |

这类回退设计，会直接影响用户体验。没有回退时，用户看到的是“安装失败”；有回退时，用户看到的是“当前环境不支持该依赖，已改用简化方案”或“该能力需进入下个镜像版本”。二者差别很大。

---

## 替代方案与适用边界

如果你的环境允许 `pip install`，推荐的默认方案仍然是“预装 + 按需 + 安全”。但这不是唯一解。

| 环境类型 | 是否允许运行时安装 | 推荐策略 | 主要限制 |
|---|---|---|---|
| 通用沙箱，如可自定义 E2B 模板 | 可 | 高频包预装，低频包扫描后按需装 | 需要维护模板和扫描链路 |
| 严格企业沙箱 | 部分可 | 白名单极小，更多依赖镜像升级 | 流程慢，但可控 |
| 禁装环境，如 Azure SRE Agent | 否 | 完全依赖预装，任务设计围绕现有库展开 | 无法临时扩展依赖 |
| 多镜像调度环境 | 视平台而定 | 按任务类型切换模板 | 需要更复杂的调度和镜像管理 |

第一种替代方案是**多模板/多镜像**。例如拆成：

- 数据分析模板：`numpy` / `pandas` / `matplotlib`
- 文档处理模板：`pypdf` / `python-docx` / `openpyxl`
- 机器学习模板：`scikit-learn` / `xgboost`

这样做的好处是减少单镜像臃肿，同时避免在线安装。但代价是模板数量增加，路由逻辑更复杂。你需要一个前置判断器，决定“这次任务应该进入哪个模板”。如果判断错了，就会出现“模板切错，还是缺包”的问题。

第二种替代方案是**完全禁用运行时安装**。这在高安全环境里很常见。Azure SRE Agent 就是现实例子：平台直接声明不支持 `pip install` 和 `conda install`。这种设计适合安全优先、任务范围明确的场景，比如企业内部报表分析、离线审计、固定数据处理流程。

第三种替代方案是**把低频依赖升级为“离线审核后再预装”**。也就是说，线上根本不开放安装能力；真正新增依赖时，由平台维护者先扫描、测试、打入下一版镜像。这个模式牺牲灵活性，但换来最强的可审计性。

第四种替代方案是**能力降级优先，而不是依赖补齐优先**。意思是：当缺少高级库时，先尝试用标准库或已有预装库实现一个简化版本，而不是立刻触发安装。举例：

| 目标 | 高级方案 | 降级方案 |
|---|---|---|
| 读取简单 CSV | `pandas.read_csv()` | Python `csv` 标准库 |
| 解析简单 JSON | `orjson` | Python `json` 标准库 |
| 生成简单图表 | `matplotlib` | 输出表格或统计摘要 |

这种方案的优点是稳，尤其适合“任务完成率优先于功能豪华程度”的产品。

适用边界可以总结成一句话：任务分布越稳定，越应该偏向预装；环境安全要求越高，越不能把“在线补包”交给模型自由发挥。

如果再压缩成一个可执行判断表，可以写成：

| 条件 | 优先策略 |
|---|---|
| 高频任务明显集中 | 扩大预装 |
| 长尾依赖很多但风险可控 | 白名单 + 按需装 |
| 平台强安全、强审计 | 禁止在线装，走镜像升级 |
| 任务路由能力较强 | 多模板调度 |
| 用户更在意稳定 than 灵活 | 缩小在线安装能力 |

最终建议仍然不变：**通用代码执行 Agent 的默认解，应当是预装高频包、对低频包采用受控按需安装，并用白名单与扫描兜底。**

---

## 参考资料

- E2B, Install custom packages  
  https://e2b.dev/docs/quickstart/install-custom-packages  
  说明 E2B 同时支持“模板预装”和“运行时安装”两种路径。

- E2B, Sandbox templates  
  https://e2b.dev/docs/sandbox-template  
  说明基于模板或 Dockerfile 构建自定义沙箱的方式。

- E2B, Template File / Dockerfile  
  https://e2b.dev/docs/legacy/sandbox/templates/template-file  
  说明 Dockerfile 作为模板文件时的支持指令范围。

- Open Interpreter, Safe Mode  
  https://docs.openinterpreter.com/safety/safe-mode  
  说明 safe mode 的核心能力是禁自动执行和代码扫描，并建议在自定义指令中接入 `guarddog` 包扫描流程。

- Microsoft Learn, Run code with code interpreter in Azure SRE Agent  
  https://learn.microsoft.com/en-us/azure/sre-agent/code-interpreter  
  说明 Azure SRE Agent 沙箱不允许网络访问、进程创建以及 `pip install` / `conda install`。

- PythonSpeed, Faster pip installs: caching, bytecode compilation, and uv  
  https://pythonspeed.com/articles/faster-pip-installs/  
  给出 `pip install` 冷缓存约 8.5 秒、热缓存约 6.3 秒的实测示例，可用于理解运行时安装的延迟成本。

- GuardDog on PyPI  
  https://pypi.org/project/guarddog/  
  说明 `guarddog pypi scan`、`guarddog pypi verify` 等包扫描能力，适合作为安装前的供应链检查工具。
