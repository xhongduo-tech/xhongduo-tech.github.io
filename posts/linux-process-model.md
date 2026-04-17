## 核心结论

Linux 进程模型可以压缩成三个动作：`fork()` 复制、`execve()` 替换、`waitpid()` 回收。`fork` 的本质不是“立刻复制整块内存”，而是先共享物理页并启用 COW，写时复制；`execve` 的本质不是“创建新进程”，而是“在原进程里换一套程序映像”；`waitpid` 的本质不是“顺便等一下”，而是通知内核可以释放子进程退出后留下的内核记录，避免僵尸进程。

可以用一条公式概括：

$$
fork() \Rightarrow
\begin{cases}
\text{parent: } ret = child\_pid \\
\text{child: } ret = 0
\end{cases}
\quad\to\quad
execve(path, argv, envp)
\quad\to\quad
waitpid(child\_pid, \&status, 0)
$$

对初学者最重要的结论有三条：

| 调用 | 做了什么 | PID 是否变化 | 地址空间是否变化 | 典型用途 |
|---|---|---:|---:|---|
| `fork()` | 复制当前进程的执行上下文 | 子进程新 PID，父不变 | 初始逻辑上相同，物理页经 COW 延迟复制 | 创建子进程 |
| `execve()` | 用新程序替换当前进程映像 | 不变 | 完全替换 | 让子进程运行新程序 |
| `waitpid()` | 读取子进程状态并回收内核记录 | 不涉及 | 不涉及 | 防止僵尸进程 |

玩具例子：父进程调用 `fork()` 后，子进程马上执行 `execve("/bin/ls", ...)`；父进程调用 `waitpid(child_pid, &status, 0)`，最后打印退出码。这已经覆盖了 Linux 里最常见的“启动一个子命令并等待结果”的流程。

---

## 问题定义与边界

这里讨论的是 Linux 中“进程如何被创建、替换、结束以及如何彼此通信”。进程可以理解为“操作系统分配资源并调度执行的独立运行单元”。本文只覆盖以下边界：

| 范围 | 说明 |
|---|---|
| 讨论 | `fork(2)`、`execve(2)`、`waitpid(2)` |
| 讨论 | 管道、FIFO、消息队列、共享内存、信号量、Socket 这六类 IPC |
| 不讨论 | 线程模型、容器隔离、调度器细节、`clone()` 的旗标组合 |
| 不讨论 | 跨机器通信协议、分布式系统一致性问题 |

“亲缘进程”指存在父子或共同祖先进程关系的进程，比如一个 Web 服务拉起的工作子进程。“无亲缘进程”指彼此独立启动、只是碰巧需要通信的进程，比如系统监控服务与独立业务进程。

进程生命周期可抽象成下面的状态变化：

| 事件 | 进程状态变化 | 说明 |
|---|---|---|
| `fork` | 一个运行中进程变成父子两个运行实体 | 子进程得到新的 PID |
| `exec` | 同一个 PID 换成新程序 | 不是新建进程 |
| `exit` | 子进程结束，但保留退出状态 | 此时可能成为僵尸 |
| `wait/waitpid` | 父进程读取状态并回收 | 僵尸记录从进程表删除 |

一个最小数值例子：

- 父进程 PID 为 `9001`
- 调用 `fork()` 后，父进程返回 `9002`，表示新子进程 PID
- 子进程返回 `0`
- 子进程执行 `execve("/bin/ls", argv, envp)`
- 父进程随后执行 `waitpid(9002, &status, 0)`

这里最容易误解的是：子进程的“返回 0”不是 PID，是真正的函数返回值；子进程自己的真实 PID 仍然是内核分配的一个正整数。

COW 可以写成一个简化条件：

$$
\text{初始时父子共享同一物理页}
\quad\land\quad
\text{页被标记为只读}
\quad\land\quad
\text{任一方首次写入}
\Rightarrow
\text{触发缺页异常并复制该页}
$$

---

## 核心机制与推导

### 1. `fork()` 到底复制了什么

`fork()` 之后，内核会为子进程建立新的任务描述结构。任务描述结构可以理解为“内核里代表一个进程的记录”。用户态看到的是“代码段、数据段、堆、栈都像被复制了一份”，但内核不会在 `fork` 瞬间把全部物理内存真的拷贝一遍，那样成本太高。

真实做法是：

1. 复制进程元数据和页表结构
2. 让父子映射到同一批物理页
3. 把这些页临时标成只读
4. 谁先写，谁触发页错误，再复制该页

所以 `fork` 适合“复制后立刻 `exec`”的模式，因为这种模式几乎不发生用户页写入，COW 成本很低。

### 2. `execve()` 为什么不产生新 PID

`execve()` 会装载新程序，替换当前进程的用户态映像。进程映像可以理解为“当前进程所运行程序的代码、数据、堆、栈以及相关内存布局”。

替换后的事实是：

- PID 不变
- 打开的文件描述符默认保留
- 已安装的某些进程属性会重置，具体以系统语义为准
- 之前程序的代码、全局变量、堆内容全部无效

这就是为什么 shell 运行外部命令时通常要先 `fork`，再让子进程 `exec`。如果 shell 自己直接 `execve()`，那 shell 进程本身就消失了。

可以把 `execve` 看成：

$$
\text{new\_image} = \{text, data, heap, stack, argv, envp\}
$$

然后用 `new_image` 覆盖旧的用户空间布局，而不是创建第二个进程。

### 3. `waitpid()` 为什么是必须的

子进程退出时，内核不能立刻把所有记录都删掉，因为父进程可能还要读取退出码、终止信号等信息。所以子进程会短暂处于“僵尸”状态。僵尸进程可以理解为“已经死了，但户口还没注销”。

`waitpid()` 做两件事：

1. 读取退出状态
2. 允许内核删除该子进程在进程表中的残留项

常见判断公式：

$$
WIFEXITED(status) = 1 \Rightarrow \text{子进程正常退出}
$$

$$
exit\_code = WEXITSTATUS(status)
$$

如果父进程一直不调用 `wait` 或 `waitpid`，僵尸会积累，最终可能耗尽 PID 或进程表项。

### 4. IPC 为什么跟 `fork/exec` 经常绑在一起

IPC 是 Inter-Process Communication，意思是“进程间通信”。因为父进程常常 `fork` 出子进程，再把工作交给子进程执行，所以通信问题马上出现：

- 父子之间如何传一段文本
- 多个工作进程如何共享一块数据
- 独立服务之间如何双向通信

六种常见 IPC 的直觉区分如下：

| 机制 | 白话解释 | 亲缘要求 | 延迟特点 | 典型场景 |
|---|---|---|---|---|
| 匿名管道 | 一根只能在相关进程间继承的字节流管子 | 通常需要亲缘 | 低到中 | shell 管道、父子同步 |
| FIFO | 有名字的管道文件 | 无亲缘也可 | 中 | 简单进程间单机传输 |
| 消息队列 | 内核维护的消息收发箱 | 无亲缘也可 | 中 | 有消息边界和优先级 |
| 共享内存 | 多个进程映射同一块内存 | 无亲缘也可 | 低 | 大数据量、低拷贝 |
| 信号量 | 控制谁能进临界区的计数器 | 常与共享内存配套 | 不传业务数据 | 同步与互斥 |
| Socket | 双向通信端点 | 无亲缘也可 | 中 | 本机服务通信、请求响应 |

真实工程例子：视频处理流水线里，采集进程和编码进程往往使用共享内存传帧，再用信号量或事件通知“新帧已就绪”。原因很直接，单帧数据大，频繁复制会浪费 CPU 和内存带宽。

---

## 代码实现

先给一个可运行的 Python 玩具例子，用来模拟 `fork -> exec -> wait` 的返回语义。它不是系统调用实现，但能帮助理解控制流。

```python
def simulate_fork_exec_wait(parent_pid: int, child_pid: int, child_exit_code: int):
    assert parent_pid > 0
    assert child_pid > 0
    assert 0 <= child_exit_code <= 255

    fork_result_parent = child_pid
    fork_result_child = 0

    # child branch: execve("/bin/ls", ...)
    exec_replaces_image = True
    waitpid_return = child_pid
    status = child_exit_code

    assert fork_result_parent == child_pid
    assert fork_result_child == 0
    assert exec_replaces_image is True
    assert waitpid_return == child_pid
    assert status == child_exit_code

    return {
        "parent_ret": fork_result_parent,
        "child_ret": fork_result_child,
        "waitpid_ret": waitpid_return,
        "exit_code": status,
    }

result = simulate_fork_exec_wait(9001, 9002, 0)
assert result["parent_ret"] == 9002
assert result["child_ret"] == 0
assert result["waitpid_ret"] == 9002
assert result["exit_code"] == 0
print(result)
```

下面是更接近真实 Linux 的 C 代码：子进程通过匿名管道给父进程发一个 `"pong"`，然后 `execve("/bin/echo", ...)`；父进程先读管道，再 `waitpid` 回收。

```c
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

int main(void) {
    int pipefd[2];
    if (pipe(pipefd) < 0) {
        perror("pipe");
        return 1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }

    if (pid == 0) {
        close(pipefd[0]);

        const char *msg = "pong";
        if (write(pipefd[1], msg, strlen(msg)) < 0) {
            perror("write");
            _exit(2);
        }
        close(pipefd[1]);

        char *argv[] = {"echo", "child exec ok", NULL};
        execve("/bin/echo", argv, environ);

        perror("execve");
        _exit(127);
    }

    close(pipefd[1]);

    char buf[16] = {0};
    ssize_t n = read(pipefd[0], buf, sizeof(buf) - 1);
    if (n < 0) {
        perror("read");
        close(pipefd[0]);
        return 1;
    }
    close(pipefd[0]);

    int status = 0;
    pid_t w = waitpid(pid, &status, 0);
    if (w < 0) {
        perror("waitpid");
        return 1;
    }

    printf("parent read: %s\n", buf);

    if (WIFEXITED(status)) {
        printf("child exit code = %d\n", WEXITSTATUS(status));
    } else {
        printf("child terminated abnormally\n");
    }

    return 0;
}
```

错误检查至少要覆盖下面几类：

| 调用 | 失败判定 | 常见原因 | 处理方式 |
|---|---|---|---|
| `fork` | `< 0` | 进程数或内存限制 | 记录错误并终止或降级 |
| `execve` | 返回才算失败 | 路径不存在、权限不足、参数错误 | 打印 `errno`，子进程 `_exit(127)` |
| `waitpid` | `< 0` | 子进程不存在、被信号中断 | 视 `errno` 重试或报错 |
| `pipe/read/write` | `< 0` | FD 错误、对端关闭 | 清理资源，避免死锁 |

共享内存 + POSIX 信号量的低延迟模式可抽象成伪代码：

```text
producer:
  shm = mmap(shared_region)
  sem_wait(empty)
  write_frame(shm)
  sem_post(full)

consumer:
  shm = mmap(shared_region)
  sem_wait(full)
  read_frame(shm)
  sem_post(empty)
```

这个模式适合“数据很大、频率很高”的真实工程场景，例如音视频帧、机器视觉图像块、共享缓存页。

---

## 工程权衡与常见坑

最常见的坑不是 API 不会写，而是系统语义理解错。

第一类坑是僵尸进程。子进程退出后，如果父进程不调用 `waitpid`，`ps` 里会看到 `<defunct>`。短时间一个两个问题不大，但守护进程长期运行时会积累成故障。解决方式通常是：

- 同步等待某个明确的子进程
- 或在 `SIGCHLD` 处理逻辑里循环调用非阻塞 `waitpid(-1, &status, WNOHANG)`

第二类坑是文件描述符泄漏。文件描述符可以理解为“进程操作文件、管道、socket 的句柄编号”。`execve` 默认会继承未标记关闭的 FD。如果父进程打开了监听 socket，而子进程 `exec` 后意外继承，会导致：

- 端口迟迟不释放
- 管道 EOF 不出现
- 敏感 FD 暴露给不该持有的程序

设置 `FD_CLOEXEC` 的示例：

```c
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

int set_cloexec(int fd) {
    int flags = fcntl(fd, F_GETFD);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
}
```

第三类坑是误以为 COW 没成本。COW 只是把复制成本从 `fork` 时刻推迟到了“首次写入”时刻。如果父子都在 `fork` 后对大块内存频繁写入，就会触发大量页复制，内存和 TLB 压力都会上升。所以“`fork` 后尽快 `exec`”不是编码习惯问题，而是性能策略。

第四类坑是把“数据传输”和“同步控制”混在一起。共享内存只解决“大家看到同一块数据”，不解决“谁先写谁后读”。因此共享内存通常必须配合信号量、互斥锁或事件通知。

| 坑 | 现象 | 根因 | 规避方式 |
|---|---|---|---|
| 忘记 `waitpid` | 出现僵尸进程 | 子进程状态未回收 | 明确回收策略或处理 `SIGCHLD` |
| FD 泄漏到 `exec` 后程序 | 管道不结束、端口不释放 | 未设置 `FD_CLOEXEC` | 创建后立即设置 close-on-exec |
| `fork` 后大量写内存 | 内存暴涨、性能下降 | COW 被频繁触发 | `fork` 后尽快 `exec`，避免大写入 |
| 共享内存数据错乱 | 读到半写入数据 | 缺少同步 | 配合信号量或锁 |
| 管道双端未关闭 | 读阻塞不返回 EOF | 某端 FD 仍被持有 | 父子各自关闭不用的一端 |

真实工程例子：一个视频采集服务将原始帧放入共享内存，多个编码 worker 读取。这里若只做共享内存映射、不做信号量同步，消费者可能读到“前半帧是旧数据、后半帧是新数据”的撕裂状态。解决办法通常是环形缓冲区 + 信号量/原子状态位。

---

## 替代方案与适用边界

IPC 没有“永远最优”的单一答案，选择取决于四个条件：是否有亲缘关系、数据量大小、延迟要求、是否需要消息边界或优先级。

| 机制 | 是否适合亲缘进程 | 是否适合无亲缘进程 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|---|---|
| 匿名管道 | 是 | 否 | 简单、系统调用少 | 单向字节流、通常限父子 | 父子进程短消息 |
| FIFO | 是 | 是 | 易于用文件路径接入 | 性能一般、语义简单 | 独立进程低复杂度通信 |
| POSIX 消息队列 | 是 | 是 | 有消息边界、可设优先级 | 容量有限、管理成本更高 | 控制消息、优先级任务 |
| 共享内存 | 是 | 是 | 延迟低、适合大块数据 | 必须自行同步 | 高频大数据 |
| 信号量 | 是 | 是 | 适合同步和互斥 | 不传数据 | 配合共享内存 |
| UNIX 域 Socket | 是 | 是 | 双向、可传 FD、接口统一 | 比共享内存多拷贝 | 本机服务通信 |

如果需要“优先级消息”，POSIX 消息队列比普通管道更合适。简化逻辑可以写成：

$$
mq\_send(q, msg, len, prio)
\Rightarrow
\text{内核按优先级组织消息}
\Rightarrow
mq\_receive(q, buf, size, \&prio)
$$

如果需要“最低延迟的大块数据传输”，共享内存常优于管道和普通 socket。若还要减少用户态复制，工程上会进一步考虑 `memfd_create`、`splice` 之类的零拷贝路径，但这已经超出入门主线。

新手可按下面的决策顺序选：

1. 父子进程、小消息、一次性命令：匿名管道
2. 独立本机服务、需要双向请求响应：UNIX 域 Socket
3. 大块数据、高频传输：共享内存 + 信号量
4. 需要消息优先级：POSIX 消息队列
5. 只想做最简单的无亲缘单机通信：FIFO

玩具例子：父进程给子进程发送 `"start"`，子进程回复 `"done"`，匿名管道就够了。

真实工程例子：本机监控代理从多个业务进程收集状态，如果还需要传递文件描述符、做双向握手、支持多客户端并发，UNIX 域 Socket 通常比 FIFO 更稳妥。

---

## 参考资料

| 来源 | 作用 |
|---|---|
| `fork(2)` man page | 确认 `fork` 返回值、错误码、语义边界 |
| `exec(3p)` / `execve(2)` 文档 | 确认程序映像替换与参数格式 |
| `waitpid(2)` man page | 确认子进程回收、状态解析宏 |
| Linux IPC 对比文章 | 补充 IPC 机制差异与选型直觉 |
| IPC 性能讨论文章 | 补充低延迟场景下共享内存等方案的工程背景 |

- `fork(2)`: https://man7.org/linux/man-pages/man2/fork.2.html
- `waitpid(2)`: https://man7.org/linux/man-pages/man2/waitpid.2.html
- `exec(3p)`: https://man7.org/linux/man-pages/man3/exec.3p.html
- Linux IPC 比较文章: https://linuxvox.com/blog/which-linux-ipc-technique-is-best-for-passing-information/
- IPC 性能讨论: https://www.codestudy.net/blog/fastest-technique-to-pass-messages-between-processes-on-linux/
