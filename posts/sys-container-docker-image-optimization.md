## 核心结论

Docker 镜像构建优化的核心，不是“把 Dockerfile 写短”，而是把最终镜像限制为“运行时真正需要的内容”，同时让构建缓存尽可能复用。

先给结论：

| 关注点 | 优化方向 | 效果 |
| --- | --- | --- |
| 层数冗余 | 合并 `RUN`、删除临时文件 | 减少镜像大小与构建时间 |
| 缓存命中 | 固定依赖前置，频繁变动源码后置 | 减少重复构建与重复上传 |
| 多阶段构建 | `COPY --from=builder` 只复制产物 | 运行镜像只保留运行期内容 |
| 基础镜像 | 从完整发行版切到 Alpine 或 distroless | 降低体积与攻击面 |

这里的“层”指镜像内部按指令叠加的文件快照，一层可以理解成一次不可变的增量包。Dockerfile 中常见的 `RUN`、`COPY`、`ADD`，都会影响层结构和缓存复用。

一个最重要的经验是：**把不常变的东西放前面，把经常变的东西放后面。**

玩具例子很简单。假设一个 Python 项目有两类输入：

1. `requirements.txt`，一个月改一次
2. 业务代码 `app/`，一天改十次

如果你先 `COPY . .`，再 `RUN pip install -r requirements.txt`，那么每次改业务代码，依赖安装层也会失效。反过来，先单独复制依赖清单并安装，再复制源码，缓存才有意义。

多阶段构建是第二个关键点。它把“编译环境”和“运行环境”分开。前者保留编译器、头文件、包管理器；后者只保留最终二进制或运行文件。这样最终镜像通常会明显变小，推送更快，暴露给生产环境的内容也更少。

---

## 问题定义与边界

先定义问题：Docker 镜像构建优化，是在不破坏运行正确性的前提下，减少镜像体积、缩短构建时间、提高缓存命中率，并控制运行时依赖集合。

边界也要说清楚。优化对象主要有三类：

| 内容类型 | 是否应该进入最终镜像 | 典型例子 |
| --- | --- | --- |
| 固定依赖 | 视运行期是否需要而定 | `libssl`、证书、时区数据 |
| 可变源码 | 通常只在解释型语言中需要 | Python 源码、Node.js 源码 |
| 构建工具链 | 通常不应该进入最终镜像 | `gcc`、`make`、`build-essential` |

可以用一个近似公式表示：

$$
最终镜像 \approx 固定依赖层 + 可变源码层 + 构建产物层
$$

如果再细一点：

$$
Size_{final} \approx Size_{base} + Size_{deps} + Size_{artifact}
$$

其中：

- `Size_base` 是基础镜像本身的体积。
- `Size_deps` 是运行时依赖，不是编译依赖。
- `Size_artifact` 是最终应用产物，比如二进制、jar、静态文件。

关键在于：**构建依赖不应该默认算进 `Size_final`。** 多阶段构建的价值，就是让编译工具和中间文件停留在 builder 阶段，不进入最终运行镜像。

看一个初学者常写的 Dockerfile：

```dockerfile
FROM ubuntu:latest
COPY . /app
WORKDIR /app
RUN apt update && apt install -y build-essential
RUN make
CMD ["./bin/app"]
```

这个写法的问题有三个：

1. 源码、编译器、构建缓存、最终二进制被混在一个镜像里。
2. 每次改源码，前面的构建步骤容易失效。
3. `ubuntu:latest` 体积较大，且 `latest` 还会引入不可控变化。

所以优化问题不是单纯“减几层”，而是明确边界：哪些内容属于开发期，哪些内容属于构建期，哪些内容必须进入运行期。

---

## 核心机制与推导

Docker 构建缓存的基本规则可以粗略理解为：某一步及其输入没变，就可以复用旧结果；某一步变了，后续依赖它的步骤通常也要重新执行。

因此镜像大小和构建代价都可以近似写成：

$$
镜像大小 \approx \sum_{i=1}^{n} layer_i
$$

其中：

$$
layer_i \in \{基础层, 依赖层, 源码层, 构建层, 运行层\}
$$

这直接推出两条工程规则。

第一条：**把稳定输入前置。**  
例如 `go.mod`、`go.sum`、`requirements.txt`、`package-lock.json` 变化频率远低于业务源码，就应该先复制这些文件并安装依赖。这样源码改动不会拖累依赖层重建。

第二条：**把最终输出和构建环境分离。**  
如果 builder 阶段大小是 900MB，其中编译器和头文件占了 700MB，而最终可执行文件只有 25MB，那么多阶段构建可以让最终镜像接近：

$$
Size_{final} \approx Size_{base(runtime)} + Size_{artifact}
$$

而不再是：

$$
Size_{base(build)} + Size_{toolchain} + Size_{artifact}
$$

这就是为什么多阶段构建经常能把数百 MB 的镜像压到几十 MB。

一个玩具例子：

- 依赖安装层：300MB，很少变化
- 源码层：20MB，经常变化
- 构建临时层：150MB
- 最终二进制：15MB

未优化时，最终镜像可能接近 `300 + 20 + 150 + 基础镜像`。  
多阶段后，最终镜像接近 `运行时基础镜像 + 15MB`。

下面这个 Python 代码不是在调用 Docker，而是模拟“变更顺序如何影响重建成本”。它可以直接运行，用来理解缓存命中逻辑。

```python
from dataclasses import dataclass

@dataclass
class Layer:
    name: str
    size_mb: int
    changed: bool = False

def rebuilt_size(layers):
    rebuild = False
    total = 0
    for layer in layers:
        if layer.changed:
            rebuild = True
        if rebuild:
            total += layer.size_mb
    return total

# 方案 A：先复制全部源码，再安装依赖
plan_a = [
    Layer("base", 80, False),
    Layer("copy_all_source", 20, True),   # 代码改了
    Layer("install_deps", 300, False),
    Layer("build", 150, False),
]

# 方案 B：先安装依赖，再复制源码
plan_b = [
    Layer("base", 80, False),
    Layer("install_deps", 300, False),
    Layer("copy_source", 20, True),       # 代码改了
    Layer("build", 150, False),
]

assert rebuilt_size(plan_a) == 470
assert rebuilt_size(plan_b) == 170
assert rebuilt_size(plan_b) < rebuilt_size(plan_a)

print("A 需要重建:", rebuilt_size(plan_a), "MB")
print("B 需要重建:", rebuilt_size(plan_b), "MB")
```

这个例子表达的不是精确 Docker 内部实现，而是工程上的主结论：**把高频变动层后置，重建和上传的体积都会下降。**

真实工程例子更典型。一个 Go 服务在 CI 中构建：

1. builder 使用 `golang:1.21`
2. 先复制 `go.mod`、`go.sum`
3. 下载依赖
4. 再复制业务源码
5. 编译为静态二进制
6. 最终镜像使用 distroless，只复制二进制和必要证书

结果通常是：

- 构建更快，因为依赖层复用
- 推送更快，因为最终镜像更小
- 安全面更好，因为运行镜像没有编译器、包管理器、shell

---

## 代码实现

先看一个推荐的多阶段 Dockerfile：

```dockerfile
FROM golang:1.21 AS builder
WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o app ./cmd/server

FROM gcr.io/distroless/static-debian12
COPY --from=builder /src/app /app
CMD ["/app"]
```

这段代码可以拆成两部分理解。

第一阶段 `builder` 的作用只有一个：编译。  
这里的 `CGO_ENABLED=0` 表示关闭 CGO，也就是不依赖本机 C 工具链和动态 C 库，白话说就是尽量产出更独立的静态二进制。`GOOS=linux` 表示目标运行系统是 Linux，避免在非 Linux 构建机上产出错误平台的可执行文件。

第二阶段使用 `distroless`。distroless 是一类极简运行镜像，白话说就是只保留程序运行所需的最少系统文件，通常没有 shell、包管理器、调试命令。它的优点是小且干净，代价是排障不方便。

如果程序需要 HTTPS，请注意证书问题。有些极简镜像需要你显式提供 `ca-certificates`。例如可以在 builder 或中间阶段准备证书，再复制进去：

```dockerfile
FROM golang:1.21 AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o app ./cmd/server

FROM debian:12-slim AS certs
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

FROM gcr.io/distroless/static-debian12
COPY --from=builder /src/app /app
COPY --from=certs /etc/ssl/certs /etc/ssl/certs
CMD ["/app"]
```

如果你用 Alpine，也要知道一个差异：Alpine 默认基于 musl libc，而很多预编译程序或某些语言扩展默认围绕 glibc 构建。`libc` 可以理解为 C 语言运行库，大量系统程序都会依赖它。  
所以 Alpine 不是“越小越一定更好”，而是“更小，但兼容性要单独验证”。

一个更贴近前端或 Node.js 项目的示例：

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
```

这里真正进入最终镜像的是打包后的 `dist/`，而不是完整源码、`node_modules`、构建工具链。对静态站点来说，这是非常常见且有效的优化方式。

---

## 工程权衡与常见坑

镜像优化不是没有代价。它本质上是在体积、速度、可读性、可调试性之间做取舍。

常见坑如下：

| 坑 | 影响 | 规避方式 |
| --- | --- | --- |
| 重复 `COPY` 相同路径 | 层膨胀，缓存行为混乱 | 合并 `COPY`，控制目录结构 |
| 不清理 apt 缓存 | 临时文件残留在层里 | `rm -rf /var/lib/apt/lists/*` |
| 过早 `COPY . .` | 依赖层频繁失效 | 先复制依赖描述文件 |
| 把 `node_modules` 打进镜像 | 体积暴涨，且常含本地无用文件 | 使用 `.dockerignore` |
| 使用 Alpine 但未验证依赖 | 运行时报库兼容错误 | 验证 musl/glibc 差异 |
| 使用 distroless 直接上线 | 出问题后难排查 | 保留 debug stage |

先说 `.dockerignore`。它和 `.gitignore` 类似，用来排除不该进入构建上下文的文件。构建上下文可以理解为 Docker 在执行 `docker build` 时会打包上传给 daemon 的输入目录。  
如果你没排除 `node_modules/`、`.git/`、`dist/`、测试数据、日志文件，镜像还没开始构建，上传输入就已经浪费很多时间和空间。

一个新手常见错误是：

```dockerfile
COPY . .
RUN npm install
```

如果本地目录里已经有 `node_modules/`，而 `.dockerignore` 又没排除，它就可能被整个复制进镜像。结果不是“装得更快”，而是把宿主机产物、平台差异文件、无用缓存一并带进去。

再说 `RUN` 合并。下面两种写法在功能上相近，但体积可能不同：

```dockerfile
RUN apt-get update
RUN apt-get install -y curl
RUN rm -rf /var/lib/apt/lists/*
```

和：

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
```

后者更常见，因为清理动作发生在同一层里，临时文件不会残留到前一层中。但代价也明显：可读性略差，调试单步失败信息不如拆开直观。

真实工程里，一个折中做法是：

- 开发镜像保留较完整工具链，便于调试
- 生产镜像使用瘦身运行层
- CI 中同时产出 `debug` 标签和 `release` 标签

这样既不牺牲排障能力，也不把调试工具永久放进生产镜像。

---

## 替代方案与适用边界

多阶段构建不是唯一手段，但通常是第一优先级最高的手段，因为它直接改变“最终镜像里有什么”。

如果多阶段已经做了，仍然需要继续优化，可以考虑下面这些方案：

| 方案 | 优点 | 适用边界 |
| --- | --- | --- |
| 多阶段 + distroless | 最小运行镜像，攻击面小 | 生产部署 |
| BuildKit `cache-from` | 提高跨机器缓存复用 | 多分支频繁构建 |
| 完整基础镜像 | 调试简单，工具齐全 | 本地开发、临时排障 |
| Alpine 运行镜像 | 通常更小 | 对 musl 兼容性已验证的服务 |
| layer squashing | 减少层历史残留 | 特殊发布流程，需谨慎使用 |

这里的 BuildKit 是 Docker 新构建后端，白话说就是更现代的构建执行器，支持更强的缓存、并行和挂载能力。对于 CI 很关键，因为很多团队的瓶颈不是本地构建，而是“每次都在新机器上从零开始”。

例如：

- 主分支每天构建几十次
- 每次都要重新下载语言依赖
- 每次都要重新上传大镜像

这时 `buildx`、`cache-from`、远端缓存导出会明显降低重复网络开销。

但它们有边界。BuildKit 解决的是“怎么更聪明地复用已有结果”，不是“最终镜像里到底放了什么”。如果 Dockerfile 本身把编译器、源码、缓存都塞进了生产镜像，再强的缓存也只是让构建快一点，不能替代镜像内容治理。

所以适用顺序通常是：

1. 先把 Dockerfile 结构改对
2. 再用多阶段清边界
3. 再选更合适的基础镜像
4. 最后才是 BuildKit 缓存、远端缓存、squash 等附加优化

一个实用建议是：本地开发和生产发布可以故意使用不同目标阶段。  
例如本地用 `builder` 或 `dev` 阶段，保留 shell、调试器、源码映射；生产只发布 `runtime` 阶段。这样不会为了“极致压缩”牺牲开发效率。

---

## 参考资料

- Docker 官方文档，构建优化与缓存策略：<https://docs.docker.com/build-cloud/optimization/>
- Docker 官方文档，多阶段构建：<https://docs.docker.com/build/building/multi-stage/>
- CubePath，Go 镜像优化示例：<https://cubepath.com/en/docs/docker-kubernetes/docker-image-optimization>
- DevDojo，多阶段与 distroless 实战：<https://devdojo.com/post/bobbyiliev/optimizing-docker-image-sizes-advanced-techniques-and-tools>
