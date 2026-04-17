## 核心结论

Linux 文件访问可以拆成四层：`inode`、`dentry`、`VFS`、页缓存访问方式。`inode` 是“文件元数据记录”，保存权限、大小、时间戳、数据块位置等信息；`dentry` 是“目录项缓存”，把文件名和 inode 关联起来；`VFS` 是“虚拟文件系统抽象层”，把 `open/read/write/stat` 这套系统调用统一到 ext4、XFS、tmpfs 等不同文件系统之上；`mmap` 与 `Direct I/O` 则决定数据是否经过页缓存。

这个分层的工程意义很直接：

| 组件 | 白话解释 | 核心职责 | 对性能/行为的影响 |
|---|---|---|---|
| inode | 文件的“身份证+属性表” | 记录元数据、定位数据块 | 决定权限检查、大小获取、块定位 |
| dentry | 路径里的“名字索引” | 建立文件名到 inode 的映射 | 命中 dcache 时可减少路径查找成本 |
| VFS | 文件系统“统一接口层” | 屏蔽 ext4/XFS/tmpfs 差异 | 同一套 syscall 可复用到不同 FS |
| mmap | 把文件页映射进进程地址空间 | 共享页缓存、减少拷贝和 syscall | 适合重复读、多进程共享 |
| Direct I/O | 绕过页缓存直接访问设备 | 降低 cache pollution | 适合大流量顺序扫描 |

一个具体路径可以把这几层串起来：进程打开 `/home/user/data.bin` 时，内核先按路径分段查找目录项，优先命中 dcache 中的 dentry；拿到 dentry 后找到 inode，检查权限和文件大小；再由 VFS 调用 ext4 或 XFS 的具体实现；如果后续使用 `mmap`，访问的通常是页缓存中的 4KB 页面；如果使用 `O_DIRECT`，数据则直接从块设备进用户缓冲区，尽量不污染页缓存。

因此，结论不是“哪种 I/O 最快”，而是“访问模式决定最优路径”。多轮重读、多个进程共享同一批文件、随机访问较多时，`mmap` 往往更合适；顺序读取超大文件、工作集远大于内存、页缓存复用价值很低时，`Direct I/O` 更有优势。

---

## 问题定义与边界

问题可以表述为：在 Linux 中，如何既理解文件元数据和路径解析过程，又根据访问模式选择合适的 I/O 方式，避免错误地把“文件系统性能”理解成单一指标。

这里有几个边界要先定清楚。

第一，`inode` 不存文件名。文件名保存在目录项里，目录项再指向 inode。所以“改文件名”通常不需要改 inode 内容，而是改目录中的名字映射。这个设计让“同一个 inode 被多个名字引用”成为可能，也就是硬链接。

第二，VFS 统一的是接口，不统一底层实现。你在用户态调用的都是 `open()`、`read()`、`write()`、`stat()`，但 VFS 会把这些操作分发到 ext4、XFS、tmpfs 各自的实现上。也就是说，编程接口统一，不等于行为细节完全一致，尤其是 `O_DIRECT` 支持程度和限制会因文件系统不同而不同。

第三，默认 `read()`/`write()` 和 `mmap()` 都与页缓存强相关。页缓存可以理解为“内核维护的一层文件页内存副本”，目的是减少真实设备访问。只要页已经在 cache 里，下一次访问通常不需要再读磁盘。

第四，`Direct I/O` 不是“完全没有内核参与”，而是“尽量绕过页缓存，不把这次数据访问纳入通用 cache 路径”。它通常要求缓冲区地址、长度、偏移满足对齐约束。没对齐时，常见结果是失败，报 `EINVAL`。

一个新手可观察的例子是：

- `cat file`：通常走默认页缓存路径。
- `dd if=file of=/dev/null bs=4M`：通常也是页缓存路径。
- `dd if=file of=/dev/null bs=4M iflag=direct`：尝试走 Direct I/O。

它们的差异不在“命令长短”，而在是否要求内核把文件页纳入 page cache 管理。

可以把整体流程先记成一条线：

`路径解析 -> dcache/dentry -> inode -> VFS -> 具体文件系统 -> 页缓存或 Direct I/O`

这个边界划分很重要，因为很多性能问题并不是出在“磁盘慢”，而是出在路径缓存没命中、页缓存被污染、访问模式和 I/O 策略不匹配。

---

## 核心机制与推导

先看 inode。inode 至少包含这几类信息：文件类型、权限位、UID/GID、文件大小、时间戳、链接计数、扩展属性，以及指向数据块或 extent 的索引信息。白话说，inode 负责回答“这是什么文件、谁能访问、多大、数据大致在哪”。

再看 dentry。dentry 是目录项对象，核心作用是把“路径中的名字”映射到 inode。因为路径解析频繁发生，Linux 会维护 dcache，也就是 dentry cache。路径解析先查缓存，再决定是否需要进一步访问底层文件系统。于是，打开同一路径的成本不只看磁盘，还看缓存命中情况。

VFS 再往上抽象一层。VFS 维护统一的数据结构和操作表，比如 inode 对象、file 对象、superblock 对象、dentry 对象。白话说，它像一个“总调度接口”，让 ext4、XFS、tmpfs 都能挂在相同的系统调用入口后面。用户态不需要关心某个文件到底位于哪种文件系统上。

接着看 `mmap`。`mmap` 的关键不是“把整个文件复制到内存”，而是“把文件页映射到进程虚拟地址空间”。真正访问某个字节时，CPU 可能先触发缺页异常，由内核把对应文件页装入页缓存，再建立页表映射。公式上，映射一个文件至少涉及多少页，可写成：

$$
page\_count = \left\lceil \frac{file\_size}{PAGE\_SIZE} \right\rceil
$$

如果文件大小是 10 MiB，且 `PAGE_SIZE = 4096`，那么：

$$
page\_count = \left\lceil \frac{10 \times 1024 \times 1024}{4096} \right\rceil = 2560
$$

也就是说，`mmap` 逻辑上会覆盖 2560 个页框对应的地址区间。注意，这不意味着程序一开始就真实占有 2560 个物理页；它通常是按需触发、按页装入。

下面给一个玩具例子。假设一个 16KB 的小文件，在 4KB 页大小下恰好占 4 页。若两个进程同时 `mmap` 这个文件，并都只读，那么这 4 页可以共享同一份页缓存；两个进程看到的是不同的虚拟地址，但底层可以指向相同的缓存页。这就是 `mmap` 在“多进程共享只读数据”场景中的核心价值。

而 `Direct I/O` 的关键机制是 bypass page cache。白话说，这次 I/O 不想让内核把数据纳入通用文件缓存。这样做的典型收益是避免 cache pollution，也就是“大量顺序扫描把真正有复用价值的热页挤出去”。代价是你失去了页缓存带来的 read-ahead、缓存命中和共享收益，而且要满足对齐约束，常见形式是：

$$
buffer\_addr \bmod PAGE\_SIZE = 0
$$

$$
length \bmod PAGE\_SIZE = 0
$$

通常偏移量也要求按设备或文件系统要求对齐。新手可以先把它记成“地址、长度、offset 都不能随便写”。

真实工程例子是 ML 训练数据加载。假设一个 2TB 数据集存成许多大文件，而机器只有 256GB 内存。若训练阶段主要是顺序扫描，而且下一轮很久以后才会再次读取同一页，那么让 page cache 装下这些页的收益很低，反而会把别的热数据挤出去。这时 `Direct I/O` 更像“告诉内核不要替我缓存这一大波流式数据”。但如果后续变成多个 worker 反复读取同一批样本，`mmap` 共享页缓存就会更有价值。

---

## 代码实现

先给一个可运行的 Python 代码块，用来推导页数和 Direct I/O 的基础对齐检查。它不是直接调用 `mmap` 或 `O_DIRECT`，但可以把机制先算清楚。

```python
import math

PAGE_SIZE = 4096

def page_count(file_size: int, page_size: int = PAGE_SIZE) -> int:
    assert file_size >= 0
    assert page_size > 0
    return math.ceil(file_size / page_size)

def is_aligned(value: int, align: int = PAGE_SIZE) -> bool:
    assert align > 0
    return value % align == 0

# 10 MiB 文件需要多少页
size = 10 * 1024 * 1024
pages = page_count(size)
assert pages == 2560

# 16 KiB 文件正好 4 页
assert page_count(16 * 1024) == 4

# Direct I/O 常见的对齐要求
buf_addr = 8192
length = 16384
offset = 0
assert is_aligned(buf_addr)
assert is_aligned(length)
assert is_aligned(offset)

print("page_count:", pages)
print("alignment checks passed")
```

如果你把这个公式和断言先理解了，后面的 `mmap` 与 `O_DIRECT` 代码就更容易读。

下面是 `mmap` 的典型流程，适合说明“fd -> fstat -> mmap -> 按字节访问”的主线：

```c
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

int main() {
    int fd = open("data.bin", O_RDONLY);
    if (fd < 0) return 1;

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return 1;
    }

    if (st.st_size == 0) {
        close(fd);
        return 0;
    }

    void *p = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        close(fd);
        return 1;
    }

    unsigned char first = ((unsigned char *)p)[0];
    printf("first byte = %u\n", first);

    munmap(p, st.st_size);
    close(fd);
    return 0;
}
```

这里最重要的点有三个。

第一，`mmap` 不是替代 `open`。你仍然要先拿到文件描述符，再基于它创建映射。

第二，`fstat` 常用于获取文件大小，因为映射长度通常要明确给出。

第三，真正访问 `((char*)p)[i]` 时，底层才会按需拉起页缓存和页表映射。也就是说，映射创建成功不等于数据已经全部读进内存。

下面再看 `O_DIRECT` 版本。关键是缓冲区必须对齐，所以常用 `posix_memalign`：

```c
#define _GNU_SOURCE
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

int main() {
    const size_t ALIGN = 4096;
    const size_t SIZE = 4096;

    int fd = open("data.bin", O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    void *buf = NULL;
    if (posix_memalign(&buf, ALIGN, SIZE) != 0) {
        close(fd);
        return 1;
    }

    ssize_t n = read(fd, buf, SIZE);
    if (n < 0) {
        fprintf(stderr, "read failed: %s\n", strerror(errno));
        free(buf);
        close(fd);
        return 1;
    }

    printf("read %zd bytes\n", n);

    free(buf);
    close(fd);
    return 0;
}
```

这个版本强调的不是“代码更复杂”，而是“责任从内核缓存策略转移到应用侧”。你需要自己保证：

- 打开的文件系统支持 `O_DIRECT`
- 缓冲区地址按要求对齐
- 长度和偏移按要求对齐
- 错误处理不能偷懒，因为失败并不罕见

---

## 工程权衡与常见坑

很多误解来自“把 mmap 和 Direct I/O 当成对立面的快慢比较”。更准确的说法是，它们优化的是不同目标。

`mmap` 的目标是复用页缓存、共享只读数据、减少显式 `read()` 系统调用和用户态拷贝。它在下面几类场景里很强：

- 多次重读同一批文件页
- 多个进程共享同一个只读数据集
- 随机访问较多，访问粒度细碎
- 访问延迟比纯吞吐更重要

`Direct I/O` 的目标是避免页缓存干预，让大流量、低复用的数据不要挤占 cache。它更适合：

- 大文件顺序扫描
- 工作集远大于内存
- 读过即丢，短期内不重复使用
- 应用本身有独立缓存层，例如数据库 buffer pool

下面这张表把常见坑集中列出来：

| 常见坑 | 现象 | 原因 | 规避建议 |
|---|---|---|---|
| 对齐失败 | `open/read` 报 `EINVAL` 或行为异常 | `O_DIRECT` 对地址、长度、偏移有要求 | 用 `posix_memalign`，把 I/O 粒度固定到 4KB 或设备要求 |
| 混用 `mmap` 和 `O_DIRECT` | 读到旧数据、性能波动、语义难判断 | 一部分走页缓存，一部分绕过页缓存 | 同一文件生命周期尽量保持单一 I/O 模式 |
| 文件系统不支持 Direct I/O | `open(..., O_DIRECT)` 失败 | 并非所有 FS、挂载参数都支持 | 先验证目标 FS 和部署环境 |
| 误把 `mmap` 当“整文件加载” | 映射很大就担心马上吃满内存 | `mmap` 多数情况下是按页触发 | 监控缺页、RSS、page cache，而不是只看映射长度 |
| 顺序流量仍走默认页缓存 | 热数据被大文件冲掉 | cache pollution | 改用 `O_DIRECT` 或配合 `posix_fadvise` |
| 小随机读强行用 Direct I/O | 吞吐和延迟反而变差 | 失去页缓存和预读收益 | 小随机访问优先默认 page cache |

真实工程里，ML 训练是最容易把这个问题放大的场景。比如 2TB 数据集、单机 256GB RAM、多个 epoch 顺序读 parquet 或二进制样本文件。如果当前阶段是一次性扫描，而且预处理线程不会短时间内重读同一批页，用默认 page cache 可能只是在把无复用数据塞进内核缓存。此时 `Direct I/O` 或类似的“尽量别缓存”策略更稳妥。相反，如果训练框架采用多 worker，且样本存在高复用窗口，`mmap` 能让多个进程共享文件页，减少重复装载和内存浪费。

---

## 替代方案与适用边界

实际选型很少是“三选一的教条题”，更像“默认 page cache、mmap、Direct I/O”之间按模式切换。

可以先用一个矩阵判断：

| 访问模式 | 推荐方式 | 理由 |
|---|---|---|
| 多轮重读、跨进程共享只读数据 | `mmap` | 共享页缓存，减少重复 I/O |
| 顺序大批量流式读写 | `Direct I/O` | 降低 cache pollution |
| 小文件、普通工具链、通用随机读写 | 默认 `read/write` | 兼容性最好，页缓存收益高 |
| 数据库自带缓存池 | 常见为 `O_DIRECT` | 避免“双重缓存” |
| 偶尔顺序扫描但仍想保留通用接口 | 默认 I/O + `posix_fadvise` | 用 hint 调整缓存策略 |

这里的替代方案主要有两个。

第一个是继续走默认 `read/write`，但用 `posix_fadvise` 给内核提示。白话说，它不是强制命令，而是“建议内核怎么对待这些页”。例如顺序扫完后，可以用 `POSIX_FADV_DONTNEED` 提示这些页可优先回收。这种方式的优点是兼容性和代码复杂度都更好，缺点是控制力度不如 `O_DIRECT`。

第二个是应用层自建缓存。数据库就是典型例子。数据库已经维护自己的 buffer pool，如果再让 OS page cache 为同一批页再缓存一次，就会形成双重缓存，浪费内存，也让淘汰策略变得难以预期。因此数据库 bulk load、日志或表空间访问常见 `O_DIRECT`。但普通文本编辑器、日志查看器、脚本工具并不值得承担这套复杂度，默认页缓存就够了。

一个玩具例子可以帮助建立边界感：如果你只是反复读取一个 2MB 的词典文件做查询，这种文件很快会常驻页缓存，默认 `read()` 甚至就已经很好；如果你非要上 `O_DIRECT`，你反而要自己处理对齐和小粒度随机访问的低效问题。也就是说，复杂方案只有在复用模式和数据规模确实匹配时才成立。

因此，适用边界可以压缩成一句话：

- 数据有复用，优先考虑页缓存路径，`mmap` 是增强版共享入口。
- 数据几乎无复用且量很大，优先考虑 `Direct I/O`。
- 既想保留缓存，又想减轻污染，可以先尝试 `posix_fadvise`，不要过早引入 `O_DIRECT` 的复杂度。

---

## 参考资料

- Linux ext4 inode 相关文档：支持 inode 的元数据字段、数据块定位方式、inode 在文件系统中的角色，对应“问题定义与边界”“核心机制与推导”。
- Linux VFS 文档：支持 dentry、inode、file、superblock 等对象关系，以及 VFS 如何统一系统调用接口，对应“核心结论”“核心机制与推导”。
- Linux 内存管理与页缓存概念文档：支持 page cache、页大小、缺页和文件页映射等机制，对应“核心机制与推导”“代码实现”。
- Linux Direct I/O 相关文章与内核资料：支持 `O_DIRECT` 的绕缓存语义、对齐要求、适用场景与限制，对应“工程权衡与常见坑”“替代方案与适用边界”。
- 文件系统与页缓存实践分析资料：支持 ML 训练、数据库 bulk load 等真实工程场景中的选型依据，对应“工程权衡与常见坑”“替代方案与适用边界”。
