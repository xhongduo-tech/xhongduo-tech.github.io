## 核心结论

Docker 多阶段构建的本质，不是“在同一个镜像里做完再删文件”，而是把**构建环境**和**运行环境**拆成多个阶段，最终镜像只保留运行时真正需要的内容。这里的“阶段”可以理解成 Dockerfile 里多个独立的 `FROM` 起点，每个起点都有自己的文件系统。

如果只看体积，最核心的近似关系是：

$$
S_{final} \approx S_{runtime\_base} + S_{artifact} + S_{runtime\_only}
$$

其中，$S_{final}$ 是最终镜像大小，$S_{runtime\_base}$ 是最终阶段所选基础镜像大小，$S_{artifact}$ 是构建产物大小，$S_{runtime\_only}$ 是运行时必须额外带上的证书、时区、动态库等文件大小。

“玩具例子”可以这样理解：前一个阶段像工厂车间，里面有机床、原料、包装废料；最后一个阶段像门店，门店只需要摆上成品，不需要把机床一起搬进去。Docker 多阶段构建做的就是这个隔离。

真实工程里，多阶段构建通常能同时带来三件事：镜像更小、拉取更快、攻击面更低。这里的“攻击面”指潜在可被利用的组件范围，镜像里工具越少，可暴露的风险点通常越少。

| 对比项 | 单阶段镜像 | 多阶段镜像 |
| --- | --- | --- |
| 编译器 | 常被保留 | 通常不进入最终镜像 |
| 源码 | 常被保留 | 通常不进入最终镜像 |
| 包管理缓存 | 常被保留 | 通常不进入最终镜像 |
| 测试工具 | 可能被保留 | 通常不进入最终镜像 |
| 最终内容 | 构建环境 + 运行环境混合 | 只保留运行必需内容 |

---

## 问题定义与边界

问题不在于“镜像大”本身，而在于**构建期依赖**和**运行期依赖**混在了一起。这里的“构建期依赖”是指为了编译、打包、测试而临时需要的工具；“运行期依赖”是指程序启动后真的要用到的内容。

很多初学者第一次写 Dockerfile，会把下面这些东西一并留在最终镜像里：

- 编译器
- 包管理器下载缓存
- 单元测试工具
- 源码仓库元数据
- 中间构建目录

这会导致镜像比实际运行需求大很多。一个常见 Node 项目就是典型例子：如果你把完整 `node_modules`、构建工具链、测试依赖、源码、缓存都放进最终镜像，体积就会迅速膨胀。

但边界也很明确：多阶段构建只能移除**可以不进入运行期**的内容，不能消除程序真正依赖的运行时文件。比如：

- `glibc`
- CA 证书
- 时区数据库
- 动态链接库
- 运行时插件

这些东西如果程序启动时要用，就不能为了“瘦身”直接删掉。

可以用一个近似式描述节省空间的来源：

$$
\Delta S \approx S_{build\_tools} + S_{temp} + S_{cache} + S_{unused\_files} - S_{extra\_runtime\_files}
$$

其中 $\Delta S$ 是多阶段相对单阶段节省的体积。最后那一项减号表示：为了让极简运行镜像可用，你有时还要补回一些必要运行文件，比如证书和时区。

下面这个单阶段 Dockerfile 片段能直观看到问题来源：

```dockerfile
FROM node:22
WORKDIR /app
COPY . .
RUN npm install
RUN npm run build
CMD ["node", "server.js"]
```

这个写法的问题不是“不能跑”，而是它默认把源码、安装缓存、构建工具和运行内容都塞进了同一个阶段。

| 内容 | 属于哪类 | 是否应进入最终镜像 |
| --- | --- | --- |
| `gcc`、`make`、JDK | 构建期依赖 | 通常不应进入 |
| 源码与测试目录 | 可丢弃内容 | 通常不应进入 |
| `dist/`、二进制文件 | 运行产物 | 应进入 |
| CA 证书、时区数据 | 运行期依赖 | 视程序需求而定 |
| 包管理缓存 | 可丢弃内容 | 通常不应进入 |

---

## 核心机制与推导

Docker 镜像是**分层**的。这里的“分层”可以理解成一层层追加变更记录，而不是每次都从零拷贝整块磁盘。多阶段构建的关键不是“让前面的层变小”，而是：**最终镜像只会包含最后阶段可见的层，以及 `COPY --from` 显式复制进去的文件**。

这句话要拆开理解：

1. `builder` 阶段里有什么，不代表最终镜像里就有什么。
2. 只有最后一个目标阶段真正决定产物长什么样。
3. `COPY --from=builder /path/a /path/b` 的语义是“只复制指定路径”，不是“继承整个构建环境”。

看一个数值化“玩具例子”：

- 构建工具：500MB
- 源码和中间文件：120MB
- 包管理缓存：100MB
- 编译产物：30MB
- 最终运行基础镜像：12MB
- 运行时额外文件：8MB

那么：

$$
S_{final} \approx 12 + 30 + 8 = 50MB
$$

而不是把前面所有内容相加。真正节省的，不是 Docker 替你“清理”了 720MB，而是这 720MB 根本没有被复制进最后阶段。

一个标准多阶段示例如下：

```dockerfile
FROM golang:1.22 AS builder
WORKDIR /src
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/app ./cmd/app

FROM scratch
COPY --from=builder /out/app /app
CMD ["/app"]
```

这个例子里，`golang:1.22` 阶段拥有编译器、标准库、源码和临时文件，但最终 `scratch` 阶段只拿走 `/out/app` 这个编译产物。

| 阶段 | 内容 | 是否进入最终镜像 | 作用 |
| --- | --- | --- | --- |
| `builder` | Go 编译器、源码、缓存 | 否 | 负责构建 |
| `builder` 输出目录 | `/out/app` | 是，被显式复制 | 运行产物 |
| `runtime` | `scratch` 基础层 | 是 | 提供最终根文件系统 |
| 其他临时文件 | 测试结果、缓存 | 否 | 构建时临时使用 |

真实工程例子更能说明机制。一个 Go 服务若使用静态编译，`builder` 阶段可能是几百 MB，但最终只需要一个 20MB 左右二进制和少量证书文件。于是交付镜像从几百 MB 降到几十 MB，这不是“优化器变魔术”，而是“未复制的内容天然消失”。

---

## 代码实现

最小实现建议优先用 Go，因为 Go 的静态二进制场景最容易理解：编译完成后，运行期通常只需要一个文件和少量附加资源。

```dockerfile
FROM golang:1.22 AS builder
WORKDIR /src

# 先复制依赖声明，利用缓存
COPY go.mod go.sum ./
RUN go mod download

# 再复制源码
COPY . .

# 关闭 CGO，尽量生成独立二进制
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/app ./cmd/app

FROM scratch
COPY --from=builder /out/app /app
CMD ["/app"]
```

这段 Dockerfile 每一行的作用如下：

| 行 | 作用 |
| --- | --- |
| `FROM golang:1.22 AS builder` | 定义构建阶段，包含完整 Go 工具链 |
| `WORKDIR /src` | 设置工作目录 |
| `COPY go.mod go.sum ./` | 先复制依赖清单，便于缓存复用 |
| `RUN go mod download` | 下载依赖 |
| `COPY . .` | 再复制源码，避免源码变更导致依赖层失效 |
| `RUN ... go build` | 编译产物到 `/out/app` |
| `FROM scratch` | 定义极简运行阶段 |
| `COPY --from=builder /out/app /app` | 只复制最终产物 |
| `CMD ["/app"]` | 启动程序 |

为了把“只复制指定路径”的效果说清楚，下面给一个可运行的 Python 玩具程序。它不调用 Docker，而是模拟“最终镜像只包含运行阶段和显式复制产物”的规则。

```python
builder_files = {
    "/src/main.go": 12,
    "/src/go.mod": 1,
    "/cache/mod.zip": 80,
    "/toolchain/go": 150,
    "/out/app": 30,
}

runtime_base_files = {
    "/bin/sh": 0,  # scratch 里通常没有，这里仅做演示
}

copied_from_builder = ["/out/app"]
runtime_only_files = {
    "/etc/ssl/certs/ca-certificates.crt": 3
}

final_files = {}
final_files.update(runtime_base_files)
for path in copied_from_builder:
    final_files[path] = builder_files[path]
final_files.update(runtime_only_files)

final_size = sum(final_files.values())
builder_total = sum(builder_files.values())

assert "/toolchain/go" not in final_files
assert "/cache/mod.zip" not in final_files
assert "/out/app" in final_files
assert final_size == 33
assert builder_total == 273
assert builder_total > final_size

print("final_size_mb =", final_size)
```

这个例子里，`builder` 阶段总量是 273MB，但最终只带走 30MB 的产物和 3MB 的证书文件。逻辑与多阶段构建完全一致：**没被复制，就不会进入最终镜像**。

如果是 Node 项目，思路也类似，只是最终阶段通常不能像 Go 那样只放一个二进制，而是要保留 `dist/`、生产依赖和运行时：

```dockerfile
FROM node:22-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci

FROM node:22-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules /app/node_modules
COPY . .
RUN npm run build

FROM node:22-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production
COPY package*.json ./
RUN npm ci --omit=dev
COPY --from=build /app/dist /app/dist
CMD ["node", "dist/server.js"]
```

---

## 工程权衡与常见坑

多阶段构建解决的是**内容隔离**，不是自动最佳化。最终镜像能缩多少，仍然取决于你选了什么运行基镜像、运行时依赖有多少、构建上下文是否干净。

最常见的误区有四类。

第一类，前面用了多阶段，最后阶段却还是很重。比如最终仍然 `FROM ubuntu`，再安装一堆运行库，体积下降会很有限。

第二类，盲目追求 `scratch`。`scratch` 是空白基础镜像，优点是极小，缺点是没有 shell、没有证书、没有时区、没有排障工具。真实工程里，Go 程序放进 `scratch` 后做 HTTPS 请求失败，常见原因就是缺少 CA 证书。

第三类，缓存顺序写反。比如先 `COPY . .` 再安装依赖，只要源码有一点变化，依赖安装层就会失效，构建时间显著增加。这里优化的是构建效率，不一定直接影响最终体积，但会影响开发和 CI 体验。

第四类，忽略 `.dockerignore`。构建上下文是发送给 Docker 守护进程的文件集合，它不等于最终镜像体积，但会影响上传时间、缓存命中和敏感信息泄露风险。

$$
S_{context} \neq S_{final}
$$

但通常有：

$$
S_{context} \uparrow \Rightarrow T_{build} \uparrow \text{，泄露风险也上升}
$$

常见坑可以整理成表：

| 常见坑 | 表现 | 解决方式 |
| --- | --- | --- |
| 最终阶段基础镜像过大 | 多阶段后仍然很胖 | 换成 `alpine`、`distroless` 或 `scratch` |
| 缺少 CA 证书 | HTTPS 请求失败 | 复制证书或选择带证书的运行镜像 |
| 缺少时区数据 | 时间显示异常 | 显式安装或复制 `tzdata` |
| `COPY . .` 太早 | 构建频繁全量重跑 | 先复制依赖清单，再复制源码 |
| 忘记 `.dockerignore` | 构建慢、上下文泄露 | 排除无关目录和敏感文件 |

一个基础 `.dockerignore` 示例：

```gitignore
node_modules/
.git/
dist/
coverage/
.env
*.pem
*.key
.DS_Store
```

真实工程里还要考虑可维护性。如果生产排障经常需要进入容器查看文件、执行诊断命令，那么极简镜像会增加运维成本。这时可以用更平衡的方案，比如 `distroless` 用于正式发布，单独保留一个 debug 镜像用于排障。

---

## 替代方案与适用边界

多阶段构建不是唯一手段，它更适合“构建环境重、运行环境轻”的项目。Go、Rust、前端静态站点都很适合；而某些运行时本身就依赖完整解释器、动态模块或系统工具链的项目，收益会小一些。

常见方案对比如下：

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 多阶段构建 | 清晰隔离构建与运行内容 | 仍需手动处理运行依赖 | 大多数可编译或可打包项目 |
| 更小基础镜像 | 见效直接 | 兼容性可能下降 | 运行依赖明确的服务 |
| 精简依赖 | 从源头减小体积 | 需要治理依赖树 | Node、Python 项目 |
| 关闭缓存残留 | 降低无效文件 | 对最终体积帮助有限时需结合别法 | 包管理器安装流程 |
| 优化产物 | 直接减少可执行文件大小 | 可能增加构建复杂度 | Go、Java、前端构建产物 |

“玩具例子”里，Go 服务很适合多阶段，因为最终常常只需一个二进制。相反，如果你的容器本质上就是一个脚本执行环境，运行时需要 `bash`、`curl`、`git`、诊断工具和动态插件，那就不适合极端追求 `scratch`。

几个典型写法如下：

```dockerfile
# alpine：小，但仍有基础用户态工具
FROM alpine:3.20

# distroless：更偏运行态，缺少常规 shell
FROM gcr.io/distroless/base-debian12

# scratch：极简，只有你复制进去的文件
FROM scratch
```

判断是否适合多阶段，可以用一条简单规则：

$$
最终镜像大小 = 基镜像 + 运行产物 + 必要运行时依赖
$$

如果你的“构建环境”远大于“运行环境”，多阶段通常收益明显；如果运行期本身就需要完整工具链，收益就会下降。

真实工程例子里，一个前端项目常见做法是：第一阶段用 Node 执行 `npm ci` 和 `npm run build`，第二阶段改用 `nginx:alpine` 只复制静态 `dist/`。这种场景非常典型，因为 Node 构建工具链很重，但最终线上只需要静态文件和 Web 服务器。

---

## 参考资料

下表给出一个适合初学者的阅读顺序：

| 资料 | 用途 | 建议阅读顺序 |
| --- | --- | --- |
| Docker Multi-stage builds | 理解定义与基本语义 | 1 |
| Docker Building best practices | 理解层、缓存、上下文与镜像设计原则 | 2 |
| Docker BuildKit | 理解现代构建后端与缓存机制 | 3 |
| `moby/buildkit` | 深入实现细节 | 4 |

1. [Docker Docs: Multi-stage builds](https://docs.docker.com/build/building/multi-stage/)
2. [Docker Docs: Building best practices](https://docs.docker.com/build/building/best-practices/)
3. [Docker Docs: BuildKit](https://docs.docker.com/build/buildkit/)
4. [moby/buildkit](https://github.com/moby/buildkit)
