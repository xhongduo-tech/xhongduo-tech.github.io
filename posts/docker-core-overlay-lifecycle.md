## 核心结论

Docker 的核心不是“把应用打包成一个大文件”，而是把文件系统拆成多层来复用。对默认的 `overlay2` 存储驱动来说，这个机制可写成：

$$
merged = lowerdir \cup upperdir
$$

其中，`lowerdir` 是只读层，白话说就是“已经做好的基础文件层”；`upperdir` 是可写层，白话说就是“这个容器自己新增或修改的部分”；`merged` 是联合视图，白话说就是“容器进程真正看到的目录树”。

这带来两个直接结果：

| 结论 | 含义 | 工程价值 |
|---|---|---|
| 镜像按层存储 | 多个镜像可以共享公共基础层 | 节省磁盘与拉取时间 |
| 容器只增加写层 | 同一个镜像启动多个容器时，只需各自维护一个 `upperdir` | 创建快，隔离清晰 |
| 生命周期是显式状态机 | `create -> start -> stop -> rm` 对应 `Created -> Running -> Stopped -> Removed` | 状态可观测、可自动化 |

一个最小例子是：`docker pull ubuntu` 拉下若干只读层；随后 `docker run ubuntu` 并不是复制一整套 Ubuntu 文件，而是在这些只读层之上再挂一个新的写层。假设这个镜像有 5 个只读层，那么运行一个容器时，本质上是“5 个 `lowerdir` + 1 个 `upperdir`”。

---

## 问题定义与边界

本文只回答两个问题。

第一，Docker 为什么能让很多容器共享同一个基础镜像，却又互不影响。  
第二，容器为什么会有 `Created`、`Running`、`Stopped` 这些状态，这些状态与命令之间是什么关系。

边界也要先说清楚：

| 本文覆盖 | 本文不展开 |
|---|---|
| OverlayFS 的 `lowerdir / upperdir / merged / workdir` | `btrfs`、`zfs`、远程快照等其他高级驱动细节 |
| `docker create/start/stop/rm` 的基础生命周期 | Kubernetes 编排、Volume 持久化、多节点调度 |
| `docker ps` 中常见状态字段的解释 | `docker inspect` 的所有 JSON 字段 |

新手可以先把分层文件系统想成“透明胶片叠加”。这个比喻只用来帮助建立直觉，不替代定义：

- 最底下几张胶片是镜像层，内容固定，不让你直接改。
- 最上面一张胶片是容器层，你写的改动只记在这里。
- 最终肉眼看到的是所有胶片叠加后的结果，也就是 `merged`。

可以用一个简化图表示读写关系：

| 层级 | 是否可写 | 谁拥有 | 读取优先级 |
|---|---|---|---|
| `upperdir` | 是 | 当前容器 | 先查这里 |
| `lowerdir` | 否 | 镜像或父层，可被多个容器共享 | `upperdir` 没有时再查 |
| `merged` | 视图，不直接存储 | 容器运行时看到的文件系统 | 对进程表现为完整目录树 |

所以，关键边界是：只有 `upperdir` 可写，`lowerdir` 只读且可共享。没有这条边界，就没有镜像层复用，也没有容器间隔离。

---

## 核心机制与推导

OverlayFS 是 Linux 内核里的联合挂载机制。联合挂载，白话说就是“把多个目录拼成一个看起来完整的目录”。Docker 利用它实现镜像层和容器层。

### 1. 读路径

当容器进程读取 `/app/config.yaml` 时，内核先查 `upperdir`。  
如果 `upperdir` 里存在同名文件，就直接返回它。  
如果没有，再按顺序去 `lowerdir` 中查找。

因此，读取规则不是“把所有层真的合并成一份新文件”，而是“访问时按优先级查找”。

### 2. 写路径

当容器修改文件时，写入不会落到 `lowerdir`，因为镜像层必须保持只读。  
写操作只会进入 `upperdir`。如果要修改一个原本只存在于下层的文件，通常会发生 `copy-up`，也就是“先把下层文件复制到上层，再在上层修改”。

### 3. 删除路径

删除也不会真的把下层文件物理删掉，因为共享镜像层不能被某个容器私自改坏。  
OverlayFS 会在 `upperdir` 里写一个 whiteout。whiteout 可以理解为“遮罩标记”，意思是：虽然下层还有这个文件，但在这个容器的 `merged` 视图里，它应该被视为已删除。

### 4. 为什么能节省磁盘

假设 `ubuntu` 镜像有 5 个只读层：

- `L1` 基础根文件系统
- `L2` 一组系统库
- `L3` 工具链
- `L4` 默认配置
- `L5` 元数据或后续增量

当你启动两个容器 `c1` 和 `c2` 时，不需要复制两份 Ubuntu。真实结构更像这样：

| 容器/镜像对象 | lowerdir 数量 | upperdir 数量 | 是否共享 |
|---|---:|---:|---|
| `ubuntu` 镜像 | 5 | 0 | 这 5 层可共享 |
| 容器 `c1` | 5 | 1 | 共享 5 层，新增自己 1 层 |
| 容器 `c2` | 5 | 1 | 共享 5 层，新增自己 1 层 |

这就是“镜像层共享，容器层独立”。共享只发生在 `lowerdir`，不会发生在 `upperdir`。

### 5. 从 `docker pull ubuntu` 到 `docker run`

可以把过程抽象成以下几步：

1. `docker pull ubuntu`
   - 下载镜像元数据和多层只读文件层
   - 这些层通常落到 `/var/lib/docker/overlay2/` 的若干目录中
2. `docker create ubuntu`
   - 创建容器元数据
   - 为该容器分配新的 `upperdir`、`workdir`、`merged`
3. `docker start <container>`
   - 用 `lowerdir + upperdir + workdir` 挂载出 `merged`
   - 启动容器主进程
4. 容器运行期间
   - 读：先上层后下层
   - 写：只写上层
   - 删：在上层记录 whiteout
5. `docker stop`
   - 主进程退出，状态转为 `Stopped`
6. `docker rm`
   - 删除容器元数据和对应写层

这里的“5 个只读层 + 1 个写层”不是固定数字，而是用来说明结构。真实镜像层数取决于镜像构建历史。

---

## 代码实现

先看 OverlayFS 在内核中的最小挂载形式。`workdir` 是 OverlayFS 的内部工作目录，白话说就是“内核做临时协调时要用的辅助目录”，不能省略。

```bash
mount -t overlay overlay \
  -o lowerdir=/layers/l5:/layers/l4:/layers/l3:/layers/l2:/layers/l1,upperdir=/containers/c1/upper,workdir=/containers/c1/work \
  /containers/c1/merged
```

这条命令表达的是：

- `lowerdir`：多个只读层，从高到低排列
- `upperdir`：当前容器自己的写层
- `workdir`：OverlayFS 内部工作区
- 挂载点 `/containers/c1/merged`：容器进程看到的最终文件系统

Docker 的 `create/start` 可以用伪代码理解：

```text
docker create ubuntu
  -> 读取镜像配置
  -> 分配 container id
  -> 创建容器元数据（如 config.v2.json、hostconfig.json）
  -> 创建 upperdir/workdir/merged 目录
  -> 状态 = Created

docker start <id>
  -> 执行 overlay mount(lowerdir, upperdir, workdir, merged)
  -> 设置 namespace/cgroup/network
  -> 在 merged 中启动 entrypoint/cmd
  -> 状态 = Running
```

下面给一个可运行的玩具例子，用 Python 模拟“上层覆盖下层”的读取规则。这里不是在真的实现内核文件系统，而是在模拟最核心的查找逻辑。

```python
def overlay_merge(lower_layers, upper_layer, whiteouts=None):
    """
    lower_layers: 从低到高排列的只读层
    upper_layer: 当前容器写层
    whiteouts: 被上层标记删除的路径集合
    """
    whiteouts = whiteouts or set()
    merged = {}

    # 先叠加 lower，再叠加 upper，后者覆盖前者
    for layer in lower_layers:
        for path, content in layer.items():
            if path not in whiteouts:
                merged[path] = content

    for path, content in upper_layer.items():
        if path not in whiteouts:
            merged[path] = content

    for path in whiteouts:
        merged.pop(path, None)

    return merged


base = {
    "/etc/os-release": "Ubuntu 24.04",
    "/app/app.py": "print('v1')",
    "/tmp/debug.log": "old",
}

security_patch = {
    "/app/requirements.txt": "flask==3.0.0",
}

container_upper = {
    "/app/app.py": "print('v2 from container')",
    "/data/result.txt": "job done",
}

whiteouts = {"/tmp/debug.log"}

merged = overlay_merge(
    lower_layers=[base, security_patch],
    upper_layer=container_upper,
    whiteouts=whiteouts,
)

assert merged["/etc/os-release"] == "Ubuntu 24.04"
assert merged["/app/app.py"] == "print('v2 from container')"
assert merged["/app/requirements.txt"] == "flask==3.0.0"
assert merged["/data/result.txt"] == "job done"
assert "/tmp/debug.log" not in merged

print("overlay toy example passed")
```

这个玩具例子体现了三条规则：

- 下层内容会出现在最终视图中
- 上层同名文件会覆盖下层
- 删除通过 whiteout 表示，而不是改写下层

真实工程例子可以看 CI 任务容器。假设流水线执行 Node 构建：

```bash
docker create --name ci-base node:20
docker start ci-base
docker exec ci-base sh -lc "npm ci && npm test"
docker stop ci-base
docker rm ci-base
```

这里 `node:20` 的基础层可被反复共享，但每次 CI 容器自己的中间文件、日志、缓存写入，都会落入它独立的 `upperdir`。如果任务结束后不删除容器，这个写层就会继续占磁盘。

---

## 工程权衡与常见坑

### 1. `docker ps` 默认看不到全部状态

很多新手以为“容器没了”，其实只是 `docker ps` 默认只显示正在运行的容器，也就是常见的 `Up` 状态。  
如果你刚执行了 `docker create` 还没 `start`，或者已经 `stop`，默认列表里都可能看不到。应使用：

```bash
docker ps -a
```

常见字段可以这样理解：

| 字段 | 含义 | 新手易错点 |
|---|---|---|
| `CONTAINER ID` | 容器短 ID | 以为这是镜像 ID |
| `IMAGE` | 容器基于哪个镜像创建 | 误以为容器本身就是镜像 |
| `COMMAND` | 启动命令 | 看见命令退出就误判 Docker 崩溃 |
| `CREATED` | 容器创建时间 | 不是启动时间 |
| `STATUS` | 当前生命周期状态 | 默认 `ps` 常只看到 `Up` |
| `PORTS` | 端口映射 | 没映射不代表服务没启动 |
| `NAMES` | 容器名称 | 名称可变，ID 更稳定 |

### 2. 生命周期必须按状态推进

容器不是一个随便操作的黑盒，而是显式状态机：

| 命令 | 前置状态 | 结果状态 | 说明 |
|---|---|---|---|
| `docker create` | 无 | `Created` | 只创建，不运行 |
| `docker start` | `Created` 或 `Stopped` | `Running` | 启动主进程 |
| `docker stop` | `Running` | `Stopped` | 请求主进程优雅退出 |
| `docker rm` | `Created` 或 `Stopped` | `Removed` | 删除容器及写层元数据 |

错误顺序会直接失败。比如：

- 对未创建对象执行 `start`：对象不存在
- 对运行中的容器直接 `rm`：通常需要先停掉，除非强制删除
- 把 `Created` 当作“已经启动”：这是最常见误解之一

### 3. 写层不会共享，清理不及时会膨胀

镜像层能共享，不代表容器层也共享。  
如果你频繁启动短命容器，但不做 `rm`，每个容器都可能留下自己的 `upperdir`。这些层不会自动合并回镜像，也不会彼此复用。

真实工程里，CI/CD 很容易踩这个坑。一个典型流程是：

1. `docker create --name ci-base node:20`
2. `docker start ci-base`
3. 执行测试和构建
4. `docker stop ci-base`
5. 查看日志、收集结果
6. `docker rm ci-base`

如果第 6 步省略，`docker ps` 默认又看不到已退出容器，团队常常会误以为机器很干净，直到磁盘被 `/var/lib/docker` 吃满。

### 4. 删除不是“真的删掉下层文件”

容器里执行 `rm /some/file`，你看到的是文件消失，但底层镜像层里的文件大概率还在。只是当前容器的 `upperdir` 记录了一个 whiteout，让 `merged` 视图把它隐藏了。  
这也是为什么不同容器之间不会互相删坏基础镜像。

---

## 替代方案与适用边界

OverlayFS 之所以常见，是因为它简单、性能好、在现代 Linux 上支持广。但它不是唯一方案。

| 驱动 | 层复用 | 写入代价 | 宿主兼容性 | 适用边界 |
|---|---|---|---|---|
| OverlayFS / `overlay2` | 强 | 中等，依赖 `copy-up` | 现代 Linux 最好 | Docker 默认推荐 |
| AUFS | 强 | 也支持联合层，但维护生态较弱 | 某些环境可用 | 旧环境历史兼容 |
| devicemapper | 不是典型目录联合视图 | 块设备层面管理，复杂度更高 | 某些老系统可用 | 特定旧部署场景 |

如果宿主机不支持 OverlayFS，通常先用 `docker info` 检查当前驱动，再考虑替代方案。这个场景常见于旧内核、定制发行版或受限内核特性环境。

还要注意，`docker compose up/down` 这类更高层工具并没有替代底层生命周期，只是把多个容器的 `create/start/stop/rm` 包装起来。  
换句话说：

- 你看到的是 Compose 服务
- Docker 实际执行的仍是容器创建、启动、停止、删除
- 状态监控的基本单位仍是容器

因此，适用边界可以总结为：

- 想理解“为什么镜像能复用、容器为什么启动快”，看 OverlayFS
- 想理解“为什么某个容器是 `Exited` 而不是消失”，看生命周期状态机
- 如果问题已经进入“跨主机编排、持久卷、一致性存储”，本文范围就不够了

---

## 参考资料

1. Docker Docs, *OverlayFS storage driver*。支撑“第 1 节核心结论”“第 2 节问题边界”“第 3 节机制推导”“第 4 节代码实现”。  
   https://docs.docker.com/engine/storage/drivers/overlayfs-driver/

2. Dev.to, *The Docker Container Lifecycle: Docker made easy*。支撑“第 1 节生命周期结论”“第 5 节工程流程中的 create/start/stop/rm”。  
   https://dev.to/alubhorta/the-docker-container-lifecycle-docker-made-easy-3-554o

3. Dev.to, *A complete guide to listing Docker containers*。支撑“第 5 节 `docker ps` 字段与默认过滤行为”。  
   https://dev.to/refine/a-complete-guide-to-listing-docker-containers-e9a
