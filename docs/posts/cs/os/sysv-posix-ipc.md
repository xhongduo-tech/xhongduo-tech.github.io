---
title: System V 与 POSIX IPC：共享内存、信号量与消息队列
date: 2026-08-07
---

# System V 与 POSIX IPC：共享内存、信号量与消息队列

<div class="epigraph">
<p>同一门功夫，两套拳法——System V 是资历最老的宗师，POSIX 是接口更干净的传人。</p>
<footer>—— 佚名，Linux 内核课堂</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 恐龙书 §3.6 与 APUE §15 ｜ 2026-08-07</p>
</div>

## 为什么从 System V 与 POSIX IPC 开始

回顾《IPC 经典机制》时用的 API 大多是 System V 的（`shmget`、`semget`、`msgget`）。Linux 上实际有**两套 IPC 接口**：**System V IPC**（老牌经典）与 **POSIX IPC**（现代替代）。这一节对比它们的三件套——**共享内存、信号量、消息队列**，并给出选型建议。<span class="marginnote">回顾《IPC 经典机制》的四兄弟：管道/FIFO/消息队列/共享内存。System V 与 POSIX 是后两者（消息队列、共享内存）加信号量的<strong>两套 API 风格</strong>——功能等价，接口与语义有细节差异。</span>

## 1 System V IPC：经典三件套

**System V IPC**（Unix System V 引入）用**标识符（id）**与 **key** 管理对象：

| 机制 | 创建 | 操作 | 特点 |
| --- | --- | --- | --- |
| 共享内存 | `shmget` | `shmat`/`shmdt` | 映射共享内存段 |
| 消息队列 | `msgget` | `msgsnd`/`msgrcv` | 有边界消息 |
| 信号量 | `semget` | `semop`/`semctl` | 计数信号量 |

**System V 的关键概念**：

- **key**：外部标识（`ftok` 生成），用于定位/创建对象。
- **id**：内核返回的句柄（类似 fd）。
- **生命周期独立**：对象**不随进程消亡**——进程退出后共享内存/队列仍在，需显式删除（`IPC_RMID`）或 `shmctl`/`msgctl` 显式控制。
- **权限**：对象有属主/组/其他权限（类似文件）。

**System V 的缺点**：**接口古老**——key 是整数、权限与文件系统不统一、对象生命周期难管理（遗留下「孤儿 IPC 对象」）。

**孤儿 IPC 对象的实战影响**：进程崩溃后，它创建的共享内存段/信号量/队列不会自动消失，会一直占用系统资源。排查工具是 `ipcs`（列出当前所有 IPC 对象及其属主/大小），清理用 `ipcrm -m <id>`、`ipcrm -s <id>`、`ipcrm -q <id>`。<span class="marginnote">`ipcs -m` 看共享内存时，被多个进程 attach 的段会显示 `nattch > 1`——这是判断「还有谁在用这段内存」的直接证据。写 daemon 时应在退出路径上 `shmctl(id, IPC_RMID, ...)`，否则重启后旧段泄漏。</span>相比之下，POSIX 对象「名字即文件」、可 `unlink`，孤儿问题在概念上被「文件即所有权」化解——`/dev/shm` 下的残留一看便知。

## 2 POSIX IPC：现代替代

**POSIX IPC** 用**名字**（路径字符串）管理对象，更接近文件接口：

| 机制 | 创建 | 操作 | 特点 |
| --- | --- | --- | --- |
| 共享内存 | `shm_open` | `mmap` | 名字 + mmap 映射 |
| 消息队列 | `mq_open` | `mq_send`/`mq_receive` | 名字 + 消息 |
| 信号量 | `sem_open` | `sem_wait`/`sem_post` | 命名/匿名信号量 |

**POSIX 的关键概念**：

- **名字**：`/name` 形式的路径——**与文件系统统一**（`/dev/shm` 下可见）。
- **mmap 集成**：共享内存用 `mmap` 映射——与文件映射统一。
- **更现代的语义**：消息队列支持优先级、超时；信号量有命名/匿名两种。

**POSIX 的优点**：接口统一（名字 + 文件风格）、与 mmap/文件集成、**生命周期更像文件**（可 unlink）。

## 3 对比：两个 API 的核心差异

| 维度 | System V IPC | POSIX IPC |
| --- | --- | --- |
| 对象标识 | key + id | **名字（路径）** |
| 接口风格 | 专用函数（shmget...） | 文件风格（open 类似） |
| 生命周期 | 独立，需显式删除 | 类似文件，可 unlink |
| 权限 | 自带权限 | 文件权限 |
| 与文件系统 | 不统一 | **统一（/dev/shm）** |
| 消息队列特性 | 类型字段 | 优先级 + 超时 |

**辨析｜易错点：** 「System V 已废弃，用 POSIX 就行」是过于绝对的选型观。**两者在 Linux 上都广泛支持且都在用**：System V 历史悠久、教科书常用、某些老应用依赖；POSIX 接口现代、与文件系统统一、新代码更推荐。**选型看「与现有代码的兼容」与「是否需要跨平台 POSIX 标准」**——没有「唯一正确」。

**公式解析：ftok 生成 key**

System V 用 `ftok` 生成 key：

$$\text{key} = \text{ftok}(\text{path}, \text{proj\_id})$$

- **path**：一个存在的文件路径（取它的 inode 号/设备号）。
- **proj_id**：一个字符（区分多个 key）。
- 返回值是一个整数 key——**同 path + 同 id 的进程得到同一个 key**，从而定位同一个 IPC 对象。

**直觉**：`ftok` 把「文件路径」翻译成「整数 key」——**两个进程只要约定同一个文件路径，就能拿到同一个 key**。这是 System V 时代的「命名约定」。

## 4 工程实践：共享内存的典型用法

**POSIX 共享内存 + mmap 的典型流程**：

```c
int fd = shm_open("/myshm", O_CREAT | O_RDWR, 0666);  /* 创建命名共享内存对象 */
ftruncate(fd, 4096);                                  /* 设定段大小 */
void *p = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
               MAP_SHARED, fd, 0);                    /* 映射进地址空间 */
/* p 即共享数据首地址：多个进程映射同一对象，读写同一物理页 */
```

- **多个进程 `shm_open` 同一名字 → mmap 同一物理页 → 共享数据**。
- 配合**信号量**（`sem_wait`/`sem_post`）做同步——共享内存 + 信号量是 IPC 的「黄金搭档」（回顾 IPC 篇：共享内存快但需自同步）。

**设计启示**：POSIX 共享内存把「共享内存」与「文件映射」统一到 `mmap` 一个机制——**共享内存只是「mmap 一个命名对象」**。这与《mmap》篇的「文件与内存统一」哲学一致：**抽象越统一，概念越少，越容易用对**。

## 5 核心对比表：三件套的选型

| 场景 | 推荐 | 理由 |
| --- | --- | --- |
| 大块数据、高频共享 | 共享内存（mmap） | 零拷贝、快 |
| 低频结构化消息 | 消息队列 | 有边界、带优先级 |
| 进程间同步 | 信号量 | 计数、阻塞语义 |
| 新代码 | POSIX IPC | 接口现代、与文件统一 |
| 维护老代码 | System V IPC | 兼容既有系统 |

## 6 术语速查表

| 术语 | 含义 | 一句话记忆 |
| --- | --- | --- |
| `key` | System V 外部标识 | ftok 约定 |
| `id` | 内核返回的句柄 | 类似 fd |
| `ftok` | 路径→key | 路径当暗号 |
| `shm_open` | 命名共享内存 | 名字即文件 |
| `mq_open` / `sem_open` | 命名队列/信号量 | 文件风格 |
| 孤儿 IPC 对象 | 进程退出后残留 | 需 ipcrm 清理 |
| `ipcs` / `ipcrm` | 查看 / 删除 IPC | 系统管家 |

## 7 小结

- **System V IPC**：key + id 管理共享内存/消息队列/信号量，生命周期独立、接口古老。
- **POSIX IPC**：名字 + mmap 集成，文件风格、生命周期似文件、接口现代。
- 功能等价、细节不同——选型看兼容与标准需求，无唯一正确。
- `ftok` 把路径翻译成 key，是 System V 的命名约定。
- POSIX 共享内存 = `shm_open` + `mmap`，与文件映射统一，配信号量做同步。

在下一节，我们从用户态线程走向内核视角——**Linux 线程实现：NPTL 与 pthread 的内核视角**。
