---
title: 常见系统调用速查：open/read/write/fork/exec/wait
date: 2026-08-07
---

# 常见系统调用速查：open/read/write/fork/exec/wait

<div class="epigraph">
<p>写 C 程序真正需要的系统调用不超过二十个——把它们记熟，操作系统的大门就对你敞开。</p>
<footer>—— 佚名，Linux 内核课堂</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 恐龙书 §2.3 与 Linux man 手册 ｜ 2026-08-07</p>
</div>

## 为什么从系统调用速查开始

第十五篇到此收官——把前面讲过的系统调用做一次**速查总结**。文件、进程、IPC 三大家族是系统编程的核心，掌握它们的签名与语义，是「会用操作系统」的最低门槛。这一节用表格 + 代码把最常用的 open/read/write/fork/exec/wait 串起来。<span class="marginnote">回顾《系统调用的类型》：进程控制、文件管理、设备管理、信息维护、通信五大家族。这一节是「五大家族中最常用成员」的浓缩手册——<strong>记住签名、理解语义，其余调用都是它们的变体</strong>。</span>

## 1 文件系统调用：open/read/write/close

**文件访问五件套**：

| 调用 | 原型 | 作用 |
| --- | --- | --- |
| `open` | `open(path, flags, mode)` | 打开/创建文件，返回 fd |
| `read` | `read(fd, buf, count)` | 从 fd 读 count 字节 |
| `write` | `write(fd, buf, count)` | 向 fd 写 count 字节 |
| `lseek` | `lseek(fd, offset, whence)` | 移动读写位置 |
| `close` | `close(fd)` | 关闭 fd |

**典型读写循环**：

```c
int fd = open("data.txt", O_RDONLY);
if (fd < 0) { perror("open"); exit(1); }

char buf[256];
ssize_t n;
while ((n = read(fd, buf, sizeof buf)) > 0) {
    write(STDOUT_FILENO, buf, n);   // 把读到的内容写回标准输出
}
close(fd);
```

**要点**：**每次调用都要检查返回值**——open 失败返回 -1 并置 errno；read/write 返回实际字节数（可能小于请求数，要循环处理）。

## 2 进程系统调用：fork/exec/wait/exit

**进程生命周期四件套**：

| 调用 | 原型 | 作用 |
| --- | --- | --- |
| `fork` | `fork()` | 创建子进程（返回子 PID / 0） |
| `execve` | `execve(path, argv, envp)` | 加载新程序替换当前进程 |
| `wait` | `wait(&status)` | 等待子进程结束并回收 |
| `exit` | `exit(status)` | 终止当前进程 |

**经典父子进程模式**：

```c
pid_t pid = fork();
if (pid == 0) {
    /* 子进程：加载新程序 */
    execl("/bin/ls", "ls", "-l", (char *)NULL);
    perror("execl");          /* 只有失败才会执行到这里 */
    exit(1);
} else if (pid > 0) {
    int status;
    wait(&status);            /* 父进程：等待并回收子进程 */
} else {
    perror("fork");
}
```

**要点**（回顾《进程创建与终止》）：

- `fork` **一次调用两次返回**：子进程得 0、父进程得子 PID。
- `exec` **成功后不返回**（进程被替换）；只有失败才返回 -1。
- `wait` **回收僵尸**：不 wait 就积累僵尸进程。

## 3 IPC 与信息系统调用

**IPC 常用调用**（回顾 IPC 篇）：

| 调用 | 作用 |
| --- | --- |
| `pipe(fds)` | 创建匿名管道（返回读端/写端 fd） |
| `shmget` + `shmat` | 创建/映射共享内存段 |
| `msgget`/`msgsnd` | 消息队列操作 |
| `socket(...)` | 网络套接字 |

**信息维护常用调用**：

| 调用 | 作用 |
| --- | --- |
| `getpid()` | 当前进程 PID |
| `gettimeofday(&tv, NULL)` | 当前时间（vDSO 加速） |
| `getrlimit(RLIMIT_NOFILE, &rl)` | 资源限制查询 |

## 4 公式解析：系统调用的统一心智模型

所有系统调用都遵循同一个「调用-检查-处理」模式：

$$\text{系统调用} \rightarrow \text{检查返回值} \rightarrow \begin{cases} \text{成功：继续} \\ \text{失败（-1 + errno）：处理错误} \end{cases}$$

- **返回值 < 0**：失败，查 errno（ENOENT 不存在、EACCES 权限、EAGAIN 重试）。
- **返回值 >= 0**：成功，按语义解读（fd、字节数、PID）。

**记忆框架**：把系统调用分成四问——

1. **我要碰文件/设备？** → open/read/write/close/lseek。
2. **我要管进程？** → fork/exec/wait/exit。
3. **我要通信？** → pipe/socket/shm/msg。
4. **我要查信息？** → getpid/gettimeofday/getrlimit。

**直觉**：**系统调用是「操作系统服务的电话号码」**——记住号码（调用名）+ 约定（返回值检查），就能调用操作系统的一切能力。**libc 封装只是更友好的皮，核心语义不变。**

## 5 核心对比表：五大家族的代表调用

| 家族 | 代表调用 | 核心语义 |
| --- | --- | --- |
| 文件管理 | open/read/write/close | fd 句柄 + 字节流 |
| 进程控制 | fork/exec/wait/exit | 创建/替换/回收 |
| 设备管理 | open/read/write/ioctl | 设备即文件 |
| 信息维护 | getpid/gettimeofday | 查询系统状态 |
| 通信 | pipe/socket/msg/shm | 进程间数据交换 |

**辨析｜易错点：** 「libc 函数 = 系统调用」是常见混淆。**libc 函数（`printf`、`fwrite`）是「带缓冲的封装」，系统调用（`read`、`write`）是「无缓冲的内核服务」**——`printf` 会先查用户态缓冲区，不足才调 `write`。**「带缓冲的库函数」与「无缓冲的系统调用」是两层**，混用可能导致数据不一致（如 `printf` 后不 `fflush` 就 `_exit`，缓冲数据可能丢失）。

## 6 小结

- **文件五件套**：open/read/write/lseek/close——fd 句柄 + 字节流 + 返回值检查。
- **进程四件套**：fork（两次返回）/exec（成功不返回）/wait（回收僵尸）/exit。
- **IPC 家族**：pipe/socket/shm/msg——进程间通信。
- **信息家族**：getpid/gettimeofday/getrlimit——查询状态。
- 统一心智：**调用 → 检查返回值 → 成功继续/失败看 errno**；库函数（带缓冲）≠ 系统调用（无缓冲）。

至此，第十五篇「Linux 专题：系统调用」收官。在下一节，我们进入进程管理——**Linux 进程描述符 task_struct 详解**。
