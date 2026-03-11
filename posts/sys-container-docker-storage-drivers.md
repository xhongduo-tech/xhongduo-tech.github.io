## 核心结论

Docker 存储驱动决定“镜像只读层”和“容器可写层”如何叠加成一个统一文件系统。对大多数现代 Linux 环境而言，经典存储驱动里主流选择是 `overlay2`，它基于 Linux `overlayfs` 实现联合挂载，也就是把多层目录合并成一个可见目录。

对初学者最重要的结论有三条：

1. `overlay2` 的性能优势来自“共享只读层”。同一个镜像层可以被多个容器复用，不需要每个容器都复制一份。
2. `overlay2` 的主要成本来自 Copy-on-Write，简称 CoW，意思是“第一次改动旧文件时先复制，再修改”。它是文件粒度，不是块粒度。
3. 写密集型数据不要放在容器可写层里，尤其是数据库、日志索引、缓存落盘目录。更合适的做法是 volume、bind mount 或 tmpfs。

可以用一个玩具例子先建立直觉。把 `overlay2` 想成两张纸：下层纸是镜像，不能改；上层纸是容器，能写。系统把两张纸贴在一起给你看。你第一次修改下层已有文件时，不能直接改底纸，只能先把整张文件复制到上层，再在上层改。于是首次写成本近似为：

$$
CoW_{first\ write} \approx copy\_up(file\_size) + \Delta metadata
$$

这里的 `copy_up` 指“把下层文件整份复制到上层”，`metadata` 指权限、属主、时间戳等元数据。

---

## 问题定义与边界

“Docker 存储驱动”讨论的不是网络、CPU 调度，也不是 volume 本身，而是容器写入自己文件系统时，Docker 怎样管理镜像层和写层。

边界先划清：

| 操作 | 是否触发 copy_up | 说明 |
| --- | --- | --- |
| 读取已有文件 | 否 | 直接读下层或缓存结果 |
| 创建新文件 | 否 | 只写入 `upperdir` |
| 修改已有文件内容 | 是 | 先整文件复制到上层 |
| 修改权限/属主等元数据 | 是 | 元数据变化也可能触发复制 |
| 删除文件 | 否，但会生成白化文件 | 白化文件用于遮住下层旧文件 |
| 删除目录 | 否，但会生成 opaque 目录 | 表示这个目录以下不再透出下层内容 |

这里的 `upperdir` 可以理解为“容器自己的写层”，`lowerdir` 是“镜像提供的只读层”，`merged` 是“容器最终看到的统一目录”。

一个具体边界例子：容器第一次修改一个来自镜像层的 1MB 文件，即使你只改其中 4KB，`overlay2` 仍要先把 1MB 整个复制到 `upperdir`。所以这次操作的 I/O 更接近“读 1MB + 写 1MB + 少量元数据更新”，而不是只写 4KB。

这也是为什么“写密集”和“读多写少”要分开讨论。读多写少的 Web 服务、CLI 工具镜像、编译环境，一般很适合 `overlay2`。数据库、消息队列、搜索引擎索引目录，如果直接写容器层，就会被 CoW 放大。

---

## 核心机制与推导

`overlay2` 的基本流程是“层搜索 -> 找到首个可见文件 -> copy_up -> 在上层写”。

Docker 官方文档说明，`overlay2` 会从较新的镜像层开始向下搜索文件；一旦找到第一份可见版本，就把结果加入缓存，并在首次写入时执行 `copy_up`。这带来两个直接推论：

1. 首次写成本和文件大小强相关。
2. 镜像层很多、目录很深时，查找路径本身也会增加延迟。

简化后的成本可以写成：

$$
CoW_{\Delta} = copy\_up(file\_size) + copy\_meta
$$

如果同一个文件已经被复制到 `upperdir`，后续写入就不再重复 `copy_up`，这时成本才更接近真实修改量。

再看删除机制。容器不能真的删除下层只读文件，所以会在上层生成“白化文件”，也就是一个特殊标记，告诉联合挂载层“这个名字虽然下层有，但对当前容器应该视为不存在”。目录删除类似，只是标记换成 opaque 目录。

还要注意两个实现细节：

1. `overlay2` 原生支持最多 128 个 lower 层，这比早期方案更适合多层镜像。
2. `rename(2)` 对跨层目录不完全支持，可能返回 `EXDEV`，应用需要回退到“复制后删除”。

可以把整个过程看成下面这张文字流程图：

`读路径` -> `按层查找文件` -> `首次写?` -> `是: copy_up 到 upperdir` -> `在 upperdir 修改`
`删除路径` -> `在 upperdir 写 whiteout/opaque 标记` -> `merged 视图中隐藏 lowerdir 内容`

真实工程例子最典型的是 PostgreSQL。假设镜像里预置了一些数据库文件，容器启动后 WAL、表空间、索引文件频繁更新。对 `overlay2` 来说，这不是“改几个块”，而是“首次改某个文件就整文件复制”。如果这样的文件很多，写层容量和磁盘 I/O 会突然上升，延迟也更不稳定。

---

## 代码实现

下面用一个可运行的 Python 玩具模型模拟 `overlay2` 的核心行为：读取、首次写触发 `copy_up`、删除生成白化标记。代码不是 Linux 内核实现，但足够表达机制。

```python
class OverlayFS:
    def __init__(self, lowers):
        self.lowers = lowers          # 从新到旧的只读层
        self.upper = {}               # 容器写层
        self.whiteouts = set()        # 被删除的文件名
        self.copy_up_count = 0

    def read(self, path):
        if path in self.whiteouts:
            raise FileNotFoundError(path)
        if path in self.upper:
            return self.upper[path]
        for layer in self.lowers:
            if path in layer:
                return layer[path]
        raise FileNotFoundError(path)

    def write(self, path, content):
        if path in self.whiteouts:
            self.whiteouts.remove(path)

        # 首次修改 lowerdir 中已有文件时，先 copy_up 一次
        if path not in self.upper:
            for layer in self.lowers:
                if path in layer:
                    self.upper[path] = layer[path]
                    self.copy_up_count += 1
                    break

        self.upper[path] = content

    def delete(self, path):
        self.upper.pop(path, None)
        self.whiteouts.add(path)

# 玩具例子：首次写触发 copy_up，后续写不再重复 copy_up
base = {"/etc/app.conf": "v1", "/bin/tool": "ELF"}
ovl = OverlayFS([base])

assert ovl.read("/etc/app.conf") == "v1"
ovl.write("/etc/app.conf", "v2")
assert ovl.read("/etc/app.conf") == "v2"
assert ovl.copy_up_count == 1

ovl.write("/etc/app.conf", "v3")
assert ovl.read("/etc/app.conf") == "v3"
assert ovl.copy_up_count == 1

ovl.delete("/bin/tool")
try:
    ovl.read("/bin/tool")
    assert False, "deleted file should be hidden by whiteout"
except FileNotFoundError:
    pass
```

这个例子里，`copy_up` 只发生一次。后续再改 `/etc/app.conf`，都是直接操作 `upper`。

把它翻成更接近实现的伪代码，就是：

```python
for layer in lower_layers_from_new_to_old:
    if file in layer:
        copy_up(layer[file], upperdir)   # 只在首次写时发生
        write(upperdir[file], payload)   # 之后都写 upperdir
        break
else:
    write(upperdir[file], payload)       # 新文件直接写 upperdir
```

如果你的程序依赖目录 `rename` 原子成功，也要额外处理 `EXDEV`，回退到“copy + unlink”。这是 OverlayFS 已知限制，不是应用代码写错。

---

## 工程权衡与常见坑

`overlay2` 之所以成为主流，不是因为它没有缺点，而是因为它在“兼容性、维护成本、共享层能力、内存效率”之间取得了较好平衡。官方文档还提到它支持页缓存共享，也就是多个容器读同一底层文件时，可以共用缓存页，这对高密度部署很有价值。

但坑也很明确：

| 坑 | 后果 | 规避方式 |
| --- | --- | --- |
| 数据库直接写容器层 | 首次写放大，写层膨胀，I/O 抖动 | 用 volume 或 bind mount |
| 大量小文件频繁修改 | copy_up 次数很多，延迟波动 | 把热写目录独立挂卷 |
| 深层镜像、层数多 | 路径查找更慢，首次写更明显 | 控制镜像层数，合并无意义层 |
| 依赖跨层 `rename` | 可能收到 `EXDEV` | 代码里实现 copy-and-unlink 回退 |
| XFS 未启用 `d_type/ftype=1` | `overlay2` 不受支持 | 重新规划 backing filesystem |
| 仍使用 AUFS / legacy overlay / devicemapper | 升级新版本 Docker 失败 | 迁移到 `overlay2` |

真实工程建议可以写得很直接。比如 PostgreSQL：

```bash
docker run -d \
  --name pg \
  -v pgdata:/var/lib/postgresql/data \
  postgres:16
```

原因不是“volume 更高级”，而是 volume 绕过存储驱动，直接写宿主机文件系统，避免 WAL、表文件、索引文件在容器层上反复触发 CoW。官方文档对写密集型 workload 的建议也是优先用 volumes。

---

## 替代方案与适用边界

如果你讨论的是当前 Docker Engine 的经典存储驱动，现实结论已经比较明确：大多数 Linux 发行版默认推荐 `overlay2`，其他历史驱动多数已退出主流。

| 驱动 | CoW 粒度 | 关键依赖 | 当前状态 |
| --- | --- | --- | --- |
| `overlay2` | 文件级 | Linux overlayfs；XFS 需 `ftype=1` | 主流推荐 |
| `devicemapper` | 块级 | thin pool、`direct-lvm`、元数据监控 | 已在 Docker v25.0 移除 |
| `aufs` | 文件级 | AUFS 内核补丁 | 已在 Docker v24.0 移除 |
| `overlay` | 文件级 | 旧版 overlayfs | 已在 Docker v24.0 移除 |
| `vfs` | 无共享优化 | 几乎无特殊要求 | 主要用于调试，不适合生产 |

`devicemapper` 常被拿来和 `overlay2` 对比。它的白话解释是“在块设备层做写时复制”，也就是复制的单位更接近磁盘块，不是整个文件。理论上它能减少某些文件级复制成本，但代价是维护复杂，需要 thin pool、`direct-lvm`、容量监控，而且已经在 Docker v25.0 被移除。对现在的新部署来说，这基本不是应优先考虑的路线。

还有一个容易混淆的边界：Docker Engine 29.0 的 fresh install 默认开始使用 containerd image store，但这不等于“存储驱动概念失效”。只要你仍在经典存储驱动路径上运行，`overlay2`、CoW、volume 绕过写层这些规律仍然成立。

所以适用边界可以压缩成一句话：读多写少、镜像层共享明显的场景，`overlay2` 很合适；高频持久写入场景，核心数据应尽量离开容器可写层。

---

## 参考资料

- Docker Docs: Storage drivers  
  https://docs.docker.com/engine/storage/drivers/
- Docker Docs: OverlayFS storage driver  
  https://docs.docker.com/engine/storage/drivers/overlayfs-driver/
- Docker Docs: Select a storage driver  
  https://docs.docker.com/engine/storage/drivers/select-storage-driver/
- Docker Docs: Volumes  
  https://docs.docker.com/engine/storage/volumes/
- Docker Docs: Deprecated features  
  https://docs.docker.com/engine/deprecated/
- Docker Docs: Device Mapper storage driver  
  https://docs.docker.com/engine/storage/drivers/device-mapper-driver/
