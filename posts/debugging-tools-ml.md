## 核心结论

`pdb` 是 Python 标准库自带的命令行调试器，作用是在程序运行到某个位置时暂停，让你检查“当前执行到哪一行、当前堆栈是什么、当前变量是什么”。`ipdb` 是把同一套调试能力放进 IPython 环境，白话说就是“命令没变，但交互体验更好”，补全、高亮、异常显示都更友好。VS Code 调试器则通常通过 `debugpy` 接到目标 Python 进程上，本地编辑器负责显示断点、变量和调用栈，远端进程负责真正执行。

对初学者最重要的不是先记完整工具链，而是先掌握一组最小命令：`n`、`s`、`c`、`p`、`l`、`b`。它们分别表示下一行、不跳过子调用地进入、继续执行、打印表达式、列出附近源码、设置断点。大多数“代码为什么和我想的不一样”问题，先靠这 6 个命令就能定位到。

如果程序已经报错，优先使用 post-mortem 调试。post-mortem 的意思是“程序死后再回到现场看尸检结果”，对应 `pdb.pm()` 或 `ipdb.post_mortem()`。如果程序跑在远端服务器或分布式训练环境，不要在所有进程同时打交互断点，否则极易卡死；通常只让 `rank 0` 停下，其余进程在同步屏障等待。

---

## 问题定义与边界

调试的目标不是“让代码停下来”，而是理解运行时状态。运行时状态至少包含三部分：

| 状态维度 | 含义 | 典型问题 |
|---|---|---|
| 当前执行位置 | 程序此刻停在哪个文件、哪一行 | 为什么没有走到我以为的分支 |
| 调用栈 | 当前函数是被谁调用进来的 | 错误是在哪一层传进来的 |
| 变量视图 | 当前帧里的局部变量和表达式值 | 为什么 `x` 是 `None`、为什么 shape 不对 |

可以把一次调试看成状态观察问题。记当前调试状态为

$$
S=(F, \ell, V)
$$

其中 $F$ 是调用栈，$\ell$ 是当前源码行号，$V$ 是当前帧可见变量集合。调试命令的作用，本质上是让你改变或观察这个状态。

边界要先分清：

| 工具 | 适用场景 | 优势 | 边界 |
|---|---|---|---|
| `pdb` | 本地单进程、终端脚本 | 零依赖、随 Python 自带 | 交互界面原始 |
| `ipdb` | 本地单进程、Notebook/终端 | 补全和高亮更好 | 需要额外安装 |
| VS Code + `debugpy` | 远端进程、容器、服务器 | 图形化断点、变量面板、调用栈面板 | 依赖端口与路径映射 |
| `pdb.pm()` / `ipdb.post_mortem()` | 已经抛异常后的定位 | 适合复盘异常现场 | 不能回到异常前更早状态 |

玩具例子是一个脚本里 `10 // x` 在 `x=0` 时抛异常；真实工程例子则是远端 GPU 训练作业或 DDP 多进程训练，这时调试问题不再只是“变量错了”，还包括“哪个进程该停、哪个进程必须继续等待”。

---

## 核心机制与推导

`pdb` 工作时，总是围绕“当前帧”展开。帧可以理解为“某个函数调用在运行时的现场记录”，里面包括局部变量、当前行号、代码对象等信息。你在提示符里输入的命令，实际上是在控制解释器如何跨帧或跨行推进，并读取当前帧里的数据。

常用命令可以这样理解：

| 命令 | 完整名 | 白话解释 | 主要作用 |
|---|---|---|---|
| `n` | `next` | 执行下一行，但不钻进子函数内部 | 适合先看主流程 |
| `s` | `step` | 执行下一步，遇到子函数会进去 | 适合怀疑某个函数内部有问题 |
| `c` | `continue` | 一路跑到下一个断点或程序结束 | 适合跳过无关路径 |
| `p expr` | `print` | 计算并显示表达式值 | 直接看变量、属性、shape |
| `l` | `list` | 显示当前附近源码 | 重新建立上下文 |
| `b lineno` | `break` | 在指定行设断点 | 精确停在目标位置 |

这套机制可以抽象成状态转移：

$$
S_t=(F_t,\ell_t,V_t)\xrightarrow{\texttt{n/s/c}} S_{t+1}
$$

其中 `n/s/c` 主要改变执行位置和栈结构，`p/l` 主要读取当前状态，`b` 则是在未来状态转移路径上插入一个停点。

一个最小的 post-mortem 场景如下。假设函数 `f(0)` 抛出 `ZeroDivisionError`。异常发生后调用 `pdb.pm()`，调试器会直接把你放到异常对应的调用栈现场。此时如果输入 `p x`，你看到的是触发异常那一帧里的 `x=0`，而不是程序后续任何状态。这就是 post-mortem 的价值：它不是重新跑程序，而是回到“崩溃现场”。

VS Code 远程调试的机制不同。它不是在终端里直接接管标准输入输出，而是让目标进程启动 `debugpy` 监听端口，再让本地 VS Code attach 上去。这里有两个关键点：

1. `host/port` 解决“IDE 去哪里连接进程”。
2. `pathMappings` 解决“本地看到的源码路径，怎样对应到远端实际执行的源码路径”。

如果这两个点有一个错，现象通常就是断点打上了但不生效，或者能连上但源码定位漂移。

简化链路如下：

```text
本地 VS Code
    |
    | attach(host, port)
    v
远端 debugpy
    |
    | 控制目标 Python 进程
    v
运行中的脚本 / 训练任务
```

---

## 代码实现

先看最小玩具例子。这个例子覆盖 `set_trace()`、常用命令和异常定位，代码可以直接运行：

```python
import pdb

def safe_div(a, b):
    pdb.set_trace()
    result = a // b
    return result

def validate():
    ok = safe_div(10, 2)
    assert ok == 5
    return ok

if __name__ == "__main__":
    value = validate()
    assert value == 5
```

运行后程序会在 `pdb.set_trace()` 处停下。你可以输入：

```text
(Pdb) p a
10
(Pdb) p b
2
(Pdb) n
(Pdb) p result
5
(Pdb) c
```

如果你要演示异常后的 post-mortem，可以改成下面这样：

```python
import pdb

def f(x):
    return 10 // x

def main():
    try:
        f(0)
    except Exception:
        pdb.post_mortem()

if __name__ == "__main__":
    main()
```

进入 post-mortem 后，常见操作是先 `l` 看源码，再 `p x` 看异常输入值，再 `w` 看调用栈。`w` 不是题目要求的核心命令，但在实际定位异常时非常常用。

如果你更习惯 IPython 风格，把 `pdb` 换成 `ipdb` 即可：

```python
import ipdb

def parse_ratio(text):
    parts = text.split("/")
    ipdb.set_trace()
    value = int(parts[0]) / int(parts[1])
    assert value == 2
    return value
```

这里命令基本不变，但你会得到更好的补全、语法高亮和对象查看体验。

真实工程例子通常是远端服务或训练作业。下面是一个最小 `debugpy` 监听脚本：

```python
import debugpy

debugpy.listen(("0.0.0.0", 5678))
print("waiting for debugger...")
debugpy.wait_for_client()

def train_step(x):
    y = x * 2
    assert y == 8
    return y

if __name__ == "__main__":
    train_step(4)
```

对应的 VS Code `launch.json` 可以写成：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Remote Attach",
      "type": "debugpy",
      "request": "attach",
      "host": "127.0.0.1",
      "port": 5678,
      "pathMappings": [
        {
          "localRoot": "/local/code",
          "remoteRoot": "/remote/code"
        }
      ],
      "justMyCode": false
    }
  ]
}
```

如果你通过 SSH 隧道把本地 `5678` 转发到远端，那么 `host` 往往填 `127.0.0.1` 就够了。流程是：

```text
目标脚本启动 -> debugpy.listen() -> wait_for_client()
-> VS Code attach -> 命中断点 -> 查看变量和调用栈
```

---

## 工程权衡与常见坑

分布式训练是最容易把调试工具用坏的地方。原因不是工具不会用，而是多进程环境下，交互式断点天然会放大同步问题。正确思路通常是“只停一个进程，其他进程等待”。

典型写法如下：

```python
if trainer.global_rank == 0:
    import pdb
    pdb.set_trace()

trainer.strategy.barrier()
```

这里的 `global_rank` 指全局进程编号，白话说就是“第几个训练进程”。只在 `rank 0` 进入断点，其他进程在 `barrier()` 等待，可以避免多个进程同时抢终端输入。`barrier` 是同步屏障，意思是“大家都走到这里后才能一起继续”。

常见坑可以直接列出来：

| 坑 | 现象 | 规避方式 |
|---|---|---|
| 所有 rank 都进 `pdb` | 训练卡死，终端不可控 | 只在 `global_rank == 0` 打断点 |
| 少了 `barrier()` | 其他 rank 继续跑，状态失配 | 在断点后显式同步 |
| `pathMappings` 错误 | VS Code 显示断点未绑定 | 使用本地/远端绝对路径，逐字核对 |
| 端口未开放 | attach 超时或连不上 | 开放端口或走 SSH 隧道 |
| `wait_for_client()` 放太晚 | 关键早期代码错过断点 | 在入口尽早调用 |
| 只看报错信息不看栈 | 修复停留在猜测层面 | 进入 post-mortem 后先看调用栈和局部变量 |

还有一个容易被忽略的权衡：`pdb` 足够轻，但可视化弱；VS Code 体验好，但前置配置更多。对于一次性脚本、数据清洗、LeetCode 风格代码，`pdb/ipdb` 通常更快。对于长时间运行的训练作业、容器内服务、远端 API 进程，VS Code attach 的收益明显更高，因为图形化调用栈和变量面板能大幅减少心智负担。

---

## 替代方案与适用边界

不是所有问题都必须上交互式调试。很多场景下，日志、traceback 和单元测试就足够。交互式调试适合“我需要看到现场状态”，不适合“我要长期记录行为轨迹”。

可以这样选：

| 方案 | 适用场景 | 优点 | 限制 |
|---|---|---|---|
| `print` / 日志 | 简单脚本、批处理 | 最轻量、最稳定 | 上下文有限，回看成本高 |
| `pdb` | 本地脚本、CLI 程序 | 无依赖、立即可用 | 交互界面粗糙 |
| `ipdb` | 需要更好交互体验 | 补全和高亮更强 | 需安装依赖 |
| `pdb.pm()` / `ipdb.post_mortem()` | 已抛异常的问题 | 直接回到崩溃现场 | 只能看异常时刻 |
| VS Code + `debugpy` | 远端、容器、服务器 | 图形化强，适合复杂工程 | 配置成本更高 |

一个实用边界是：如果环境 headless，也就是“没有图形界面、只有命令行”，那么 `ipdb.post_mortem()` 往往比强上 VS Code 更稳。反过来，如果你需要在远端训练脚本里跨多个模块追踪张量 shape、对象属性和调用链，VS Code attach 比反复 `print` 更有效。

初学者容易把工具选型理解成“哪个更高级”。实际上它是“哪个更匹配问题形态”。本地单脚本先学 `pdb`；需要更舒适就换 `ipdb`；涉及远端和长时任务，再上 `debugpy` 和 IDE attach。这是成本最低的一条学习路径。

---

## 参考资料

- Python 官方文档：`pdb` 命令、`set_trace()`、`post_mortem()`、`pm()` 的权威说明  
  https://docs.python.org/3/library/pdb.html
- Python 3.9 `pdb` 文档：包含 `pdb.pm()` 的示例说明，便于理解异常后的调试入口  
  https://docs.python.org/3.9/library/pdb.html
- `ipdb` PyPI 页面：说明 `ipdb.set_trace()`、`ipdb.pm()` 以及 IPython 增强能力  
  https://pypi.org/project/ipdb/
- VS Code 文档：`launch.json` 通用配置与 `attach` 调试配置机制  
  https://code.visualstudio.com/docs/debugtest/debugging-configuration
- VS Code 容器调试文档：给出 Python Remote Attach 示例，含 `host`、`port`、`pathMappings`  
  https://code.visualstudio.com/docs/containers/docker-compose
- Lightning 官方文档：分布式训练中只在 `global_rank == 0` 进入 `pdb`，并用 `barrier()` 同步  
  https://lightning.ai/docs/pytorch/stable/debug/debugging_advanced.html
