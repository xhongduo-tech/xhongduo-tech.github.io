## 核心结论

Code Interpreter 的沙箱，本质上是一个给智能体执行“模型刚刚生成的代码”的隔离计算环境。隔离，白话说，就是这段代码能运行，但它默认碰不到宿主机、碰不到不该碰的网络，也拿不到超出授权的资源。

它成立的关键不在“能跑 Python”，而在“能把任意代码执行这件事收进边界”。这个边界通常由三部分组成：

1. 运行隔离：用 Firecracker microVM 这类轻量虚拟机承接执行过程。
2. 资源配额：限制 CPU、内存、磁盘和最长执行时间，避免无限消耗。
3. 审计闭环：把 `stdout`、`stderr`、退出状态、执行日志和 Trace 返回给 Agent，而不是让代码直接影响外部系统。

简化流程可以写成：

`提交代码 -> 沙箱 S 执行 -> 输出/错误/Trace 返回给 Agent -> microVM 销毁`

所以，Code Interpreter 不是普通“工具调用”的增强版，而是把智能体从“只能调用预先写好的接口”推进到“可以先写脚本，再在受控环境里自己跑”的执行代理。

---

## 问题定义与边界

问题定义很直接：如果智能体会生成代码，那么系统就必须回答一个更难的问题，怎样让“任意代码”在企业环境中可运行、可限制、可审计。

这里的“任意代码”不是完全无限制，而是指调用方事先并不知道模型会写出什么具体脚本。今天可能是 CSV 清洗，明天可能是图表生成，后天可能是统计检验。业务希望保留这种灵活性，但安全团队不能接受一段新生成的脚本直接在主机或生产网络里裸跑。

边界主要有三类：

| 行为项 | 沙箱策略 | 目的 |
|---|---|---|
| 代码执行位置 | 仅在独立 microVM 中执行 | 避免影响宿主环境 |
| 网络访问 | 可配置为 Sandbox / Public / VPC | 控制数据外流与内网访问 |
| CPU/内存/磁盘 | 强制上限，如 2vCPU、8GiB、10GiB | 防止资源耗尽 |
| 执行时长 | 强制超时，如 15 分钟 | 避免死循环长期占用 |
| 请求大小 | 限制单次输入体积，如 100MB | 防止大包滥用 |
| 身份权限 | 最小 IAM 权限 | 避免脚本越权访问云资源 |
| 会话生命周期 | 执行后回收或会话结束销毁 | 防止状态泄漏 |

一个典型反例是：开发者手动把模型生成的脚本复制到自己机器执行。这样脚本可能直接访问公网、读取本地凭据、扫内网地址，甚至误删文件。沙箱要解决的不是“脚本能不能跑”，而是“脚本跑起来时，最坏情况也不能突破设计边界”。

玩具例子很简单。假设用户说：“把这份 5MB 的日志统计每个错误码出现次数。”模型生成一个 Python 脚本，脚本进入沙箱执行。沙箱只给它受限 CPU、内存、磁盘和时间，执行完成后只返回统计结果与错误日志。对用户来说，它像“自动运行了一次脚本”；对系统来说，它其实是一轮严格隔离的受控执行。

---

## 核心机制与推导

可以把一次沙箱执行抽象成：

$$
S(c, ctx, s) \rightarrow (output, error, trace)
$$

这里：

- $c$ 是代码内容，也就是智能体生成的脚本。
- $ctx$ 是上下文，白话说，就是本轮执行所需的输入数据、文件、参数和历史信息。
- $s$ 是沙箱状态，比如本轮会话的运行时环境、挂载目录、临时文件系统。
- `output` 是标准输出或产出文件索引。
- `error` 是标准错误、异常信息、超时状态。
- `trace` 是执行轨迹，白话说，就是“这次到底做了什么”的记录，便于审计与后续决策。

但仅有函数形式还不够，真正关键的是它受到资源集合 $R$ 和网络策略 $N$ 的约束：

$$
R = \{CPU \le 2vCPU,\ MEM \le 8GiB,\ disk \le 10GiB,\ timeout \le 15min,\ request \le 100MB\}
$$

$$
N \in \{Sandbox,\ Public,\ VPC\}
$$

所以更准确地说，是：

$$
S_{R,N}(c, ctx, s) \rightarrow (output, error, trace)
$$

这意味着执行不是“尽力跑”，而是“在给定约束下跑”。如果模型提交一个 8MB 的代码与数据包，请求体没有超过 100MB，代码在 15 分钟内完成，那么系统返回结果；如果超过时间上限，则直接终止并记录超时；如果尝试访问未开放网络，则连接失败并留下错误日志。

这里有一个重要推导：智能体的多轮推理，实际上依赖这个闭环。

第 $t$ 轮可以写成：

$$
(c_t, action_t) = A(history_t)
$$

$$
(output_t, error_t, trace_t) = S_{R,N}(c_t, ctx_t, s_t)
$$

$$
history_{t+1} = history_t \cup \{output_t, error_t, trace_t\}
$$

也就是说，沙箱不是推理系统外面的附属物，而是下一轮决策的数据来源。模型之所以能“修脚本”，不是因为它突然变得会调试，而是因为上一轮 `stderr` 和 Trace 被结构化地送回来了。

再看一个玩具例子。模型第一次写出：

- 读取 `numbers.csv`
- 计算均值与标准差
- 打印结果

如果文件名拼错，沙箱返回 `FileNotFoundError`。下一轮模型根据错误信息把文件名修正，再次提交执行。这个过程就是“代码生成 + 受控执行 + 反馈修正”的闭环。

真实工程例子更能说明价值。假设审计团队需要分析 S3 中的交易明细，计算异常转账分布，并生成一张内部报告图。传统做法是工程师手写 ETL 程序、部署任务、开权限。Code Interpreter 的做法是：Agent 根据审计问题生成分析脚本，在受控网络和最小 IAM 权限下读取指定数据，执行统计和可视化，把图表与日志返回，同时通过平台日志体系留痕。灵活性来自“脚本动态生成”，安全性来自“执行环境静态受控”。

---

## 代码实现

从工程实现看，智能体不需要自己管理虚拟机生命周期。它只需要提交代码和输入，底层平台完成启动、执行、收集、销毁。

下面是一个可运行的 Python 玩具实现。它不是 Firecracker 本身，而是用代码模拟“配额检查 + 执行结果回传”的核心思想。

```python
from dataclasses import dataclass

@dataclass
class Limits:
    cpu: int = 2
    mem_gib: int = 8
    disk_gib: int = 10
    timeout_min: int = 15
    request_mb: int = 100

def sandbox_execute(code_size_mb: int, runtime_min: int, limits: Limits):
    trace = []
    trace.append("boot microVM")
    if code_size_mb > limits.request_mb:
        trace.append("reject request: request too large")
        return "", "RequestTooLarge", trace

    trace.append("run user code")
    if runtime_min > limits.timeout_min:
        trace.append("terminate: timeout")
        trace.append("collect logs")
        trace.append("destroy microVM")
        return "", "Timeout", trace

    output = f"finished within {runtime_min} min"
    trace.append("collect stdout/stderr")
    trace.append("destroy microVM")
    return output, "", trace

limits = Limits()

out, err, trace = sandbox_execute(code_size_mb=8, runtime_min=3, limits=limits)
assert out == "finished within 3 min"
assert err == ""
assert trace[-1] == "destroy microVM"

out2, err2, trace2 = sandbox_execute(code_size_mb=8, runtime_min=20, limits=limits)
assert out2 == ""
assert err2 == "Timeout"
assert "terminate: timeout" in trace2

out3, err3, trace3 = sandbox_execute(code_size_mb=120, runtime_min=1, limits=limits)
assert out3 == ""
assert err3 == "RequestTooLarge"
assert "reject request: request too large" in trace3
```

上面的代码刻意保留了最核心的四步生命周期：

1. 启动：`boot microVM`
2. 执行：`run user code`
3. 收集：`collect stdout/stderr`
4. 销毁：`destroy microVM`

如果映射到更接近真实系统的伪代码，可以写成：

```python
def run_in_agentcore(py_file, inputs, limits, network_mode):
    session = agentcore.start_microvm(
        runtime="python",
        cpu=limits.cpu,
        memory_gib=limits.mem_gib,
        disk_gib=limits.disk_gib,
        timeout_min=limits.timeout_min,
        network=network_mode,
    )
    try:
        session.upload(py_file)
        session.upload(inputs)
        result = session.execute("python main.py")
        return {
            "output": result.stdout,
            "error": result.stderr,
            "trace": result.trace,
        }
    finally:
        session.destroy()
```

这段伪代码要表达的重点只有一个：智能体只负责“交付代码与目标”，环境管理交给平台。否则，模型不仅要写脚本，还要管容器、网络、清理和权限，这会把系统复杂度推到不可控。

---

## 工程权衡与常见坑

沙箱不是万能保险箱，它只是把风险压进了可管理区间。工程上最常见的问题，不是“沙箱失效”，而是“边界没有定义完整”。

| 常见坑 | 结果 | 对策 |
|---|---|---|
| 网络模式放得过宽 | 可能访问公网或错误内网资源 | 默认用 `Sandbox`，只有必要时才开 `VPC/Public` |
| IAM 权限过大 | 脚本可越权访问存储或服务 | 使用最小权限原则 |
| 不设超时 | 无限循环长期占用资源 | 强制 `timeout <= 15min` |
| 不设请求大小限制 | 大包输入拖垮执行系统 | 限制单次请求如 `<=100MB` |
| 会话结束不销毁 | 上轮状态泄漏到下轮 | 执行后立即销毁 microVM |
| 只看输出不看错误日志 | 调试与审计困难 | 保留 `stdout/stderr/trace` |
| 把沙箱当生产任务引擎 | 长任务、重任务表现差 | 把它用于受限分析与推理闭环，而非大规模批处理 |

一个典型失败模式是：团队只关注“模型能不能把图跑出来”，没关注“图是怎么跑出来的”。结果脚本需要联网下载第三方包、请求外部 API、写入不受控对象存储。功能短期看是成功了，但安全边界已经失守。

另一个常见坑是资源估计过于理想化。模型生成的代码经常不稳定，比如：

- 忘记流式读取大文件，导致内存暴涨。
- 在循环里重复解析同一数据，导致 CPU 时间浪费。
- 写出无终止条件的重试逻辑，导致执行卡死。

因此，硬性 quota 不是可选项，而是前提条件。对零基础工程师来说，可以把它理解成“不是让脚本更快，而是让坏脚本也伤不到系统”。

执行后的清理同样关键。简单流程是：

`执行完成/失败 -> 收集输出与日志 -> 删除临时文件 -> 销毁 microVM -> 结束会话`

如果这一环缺失，多轮会话之间可能共享残留文件、缓存数据甚至身份上下文。那时问题已经不是“脚本执行失败”，而是“隔离假设被破坏”。

真实工程例子里，金融审计或内部报表是很适合沙箱的，因为它们通常需要灵活脚本，但又要求日志留痕、权限可核查、执行边界固定。相反，把它拿去跑长时间训练任务或高并发在线推理，就会遇到延迟和成本上的明显劣势。

---

## 替代方案与适用边界

沙箱执行并不是唯一方案。很多场景下，传统工具调用更合适。

工具调用，白话说，就是系统提前定义好接口，模型只能在固定入口里填参数；Code Interpreter 则允许模型先写脚本，再在受控环境里运行。两者核心区别不是“会不会写代码”，而是“系统把灵活性放在哪一层”。

| 维度 | 传统工具调用 | Code Interpreter |
|---|---|---|
| 输入输出结构 | 固定 | 动态，可由脚本定义中间步骤 |
| 灵活性 | 低到中 | 高 |
| 安全控制 | 强，因接口预定义 | 强，但依赖沙箱边界是否完整 |
| 调试方式 | 看 API 返回值 | 看 `stdout/stderr/trace` |
| 延迟 | 通常更低 | 通常更高，需启动执行环境 |
| 适合任务 | 稳定流程、固定动作 | 探索性分析、动态脚本处理 |
| 失败模式 | 参数错、接口错 | 代码错、资源超限、权限边界错 |

举个对比：

- 调用数据分析 API：你必须事先定义“输入是什么、输出是什么、图表类型有哪些”。它适合重复性高的统计任务。
- 用 Code Interpreter：模型可以根据问题临时生成清洗、聚合、画图脚本。它适合临时分析、半结构化数据处理、报告生成。

所以适用边界很清楚：

1. 如果业务动作固定、延迟敏感、合规要求明确，优先工具调用。
2. 如果问题形态变化大，需要“先推理、再写脚本、再根据错误修正”，优先 Code Interpreter。
3. 如果任务是长时批处理、大规模数据管道、持续服务，不应把沙箱当通用计算平台。

换句话说，Code Interpreter 解决的是“动态计算能力不足”的问题，不是取代所有后端系统。

---

## 参考资料

1. Amazon Bedrock AgentCore Code Interpreter 官方博客  
用途：说明 Code Interpreter 作为受控沙箱执行环境的定位、Firecracker microVM 的隔离思路，以及合规审计背景。

2. Amazon Bedrock AgentCore limits 文档  
用途：确认资源限制与配额细节，如 `2vCPU`、`8GiB`、`10GiB`、`15 分钟`、请求体限制与网络模式等硬边界。

3. Amazon Bedrock AgentCore built-in tools/how it works 文档  
用途：补充内置工具执行流程、会话隔离与执行后清理等机制，支持“输出回传 + 生命周期销毁”的工程描述。
