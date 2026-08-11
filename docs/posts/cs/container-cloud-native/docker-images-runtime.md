---
title: Docker 镜像与运行时
date: 2026-08-11
---

# Docker 镜像与运行时

<div class="epigraph">
<p>镜像是一棵层叠的树，每层都是不可变的快照；容器是这棵树顶上临时长出的可变叶子。</p>
<footer>—— 意译自 Solomon Hykes（Docker 创始人）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 容器与云原生 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么先讲镜像

上一节我们把容器还原成了「加了 namespaces 与 cgroups 的进程」，但日常里我们从不直接 `clone()` 进程——我们操作的是**镜像**。镜像解决了容器技术能否普及的关键问题：**可分发性**。一个环境能不能被原样复制到另一台机器、另一批人手里，决定了它能不能变成工程师的通用语言。<span class="marginnote">镜像（image）与容器（container）的关系可以类比「类与实例」或「烧录的固件与运行中的设备」：镜像是静态蓝图，容器是蓝图的运行态。</span>

## 1 分层文件系统：镜像的核心

Docker 镜像由若干**只读层（layer）**堆叠而成，每一层对应 Dockerfile 中的一条指令。写 Dockerfile 时你看到的每条 `RUN`、`COPY`、`ADD`，构建时都会产生一个新层。

$$\text{Image} = L_0 \oplus L_1 \oplus \cdots \oplus L_n, \qquad \text{Container} = \text{Image} + L_{\text{rw}}$$

- $L_i$：第 $i$ 个只读层，内容是文件集合（含删除标记）。
- $\oplus$：层的叠加合并——「叠在一起看」构成完整的根文件系统。
- $L_{\text{rw}}$：容器启动时新增的**可写层**，容器的写入全部落在这一层，退出后被丢弃（除非 commit 回镜像）。

层之所以能高效叠加，靠的是**写时复制（copy-on-write）**与**联合挂载（union mount）**：读文件时从上到下逐层查找，第一次命中即返回；写文件时把该文件从下层复制到可写层再修改。因此镜像层可以多容器共享——同一台机器上跑 10 个基于同一镜像的容器，磁盘只存一份底层。<span class="marginnote">共享 + 只读 + 写时复制，让「镜像体积虽大、但每容器边际成本极低」成为可能。这也解释了为什么 Docker 官方建议「小的基础镜像（如 alpine/distroless）」与「尽量合并 RUN 指令」——层数越多，拉取与共享的开销越大。</span>

## 2 内容寻址：镜像如何保证一致性

每一层都有一个**内容摘要（digest）**，即该层内容的加密哈希：

$$D(L) = \text{SHA256}(L)$$

镜像清单（manifest）记录所有层的摘要与顺序。于是：

- 同一内容一定产生同一摘要，镜像可以按摘要去重、按摘要寻址（content-addressed storage）。
- 拉取、校验、分发都基于摘要——**「你拉到的，就是发布者签名的那份」**由哈希保证，传输损坏会被发现。
- 镜像可以设置**不可变标签**（如 `sha256:...`）代替易变的 `latest`，这在生产与安全审计中极其重要。

## 3 从 Dockerfile 到镜像：一次构建

```dockerfile
FROM golang:1.22 AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /app/server

FROM alpine:3.20
COPY --from=builder /app/server /server
EXPOSE 8080
ENTRYPOINT ["/server"]
```

这个例子演示了三个核心概念：

- **多阶段构建**：先在一个含完整工具链的镜像里编译，再把产物复制进精简的运行镜像——最终镜像里没有编译器与源码，体积骤降。
- **每层只存增量**：`COPY go.mod` 与 `COPY .` 分开写，让依赖层可以被缓存——改一行业务代码，不会让依赖层作废。
- **`ENTRYPOINT` 与 `CMD`**：定义容器启动时执行什么、带什么默认参数。容器启动的进程（PID 1）由此决定，也决定了「优雅退出」的信号处理——这在《弹性伸缩》与可观测性课里都会回来。

## 4 运行时：谁把镜像变成进程

镜像只是蓝图，把蓝图还原成进程的是**容器运行时（container runtime）**。现代栈做了分层：

- **上层运行时（high-level）**：如 `containerd`（Kubernetes 默认）、`CRI-O`。负责镜像管理、生命周期编排，通过 **CRI（Container Runtime Interface）** 与 kubelet 通信。
- **底层运行时（low-level / OCI runtime）**：如 `runc`——真正调用 `clone()` + namespaces/cgroups、执行 `pivot_root` 的引擎。<span class="marginnote">OCI（Open Container Initiative）定义了两份规范：<strong>image spec</strong>（镜像的层、清单、配置格式）与 <strong>runtime spec</strong>（config.json 如何描述一个容器）。任何按规范实现的运行时都可互换，这正是生态互操作的根基。</span>
- **沙箱运行时**：`gVisor`（用用户态内核拦截系统调用）、`Kata Containers`（轻量 VM）——牺牲一点性能换更强的隔离，见上一节「容器安全 ≠ 虚拟机安全」。

Docker 只是最外层 UX（`docker build` / `docker run`），它调用 containerd，containerd 再调用 runc。**Docker 与 Kubernetes 并不冲突，Docker 只管「构建与本地体验」，Kubernetes 通过 containerd 直接使用镜像。**

## 5 辨析｜易错点：层是只读的，可写层是临时的

**最常踩的坑**：以为在容器里写的文件会被保留。真相是：

- 所有写操作进入**可写层**，容器删除即丢失。
- 「持久化」必须依赖**卷（volume）**——把宿主机目录或网络存储挂载进容器，绕过可写层。卷的挂载与存储类，见《持久化存储与 CSI》。
- 不要在生产镜像里把数据写进 `/` 或 `/var/lib`，也不要依赖容器 PID 稳定。

**第二个易错点**：`docker build` 的缓存（builder cache）按「指令 + 前置层摘要」匹配。没有变化的层会被复用，但 `COPY` 的校验基于文件内容——**把 `go.sum` 与源码分开 COPY，正是为了最大化缓存命中**。盲目把所有东西塞进一条 `RUN`，缓存就废了。

## 6 小结

- 镜像是**只读层**的堆叠，容器 = 镜像 + 临时**可写层**。
- 层靠**写时复制**共享，靠 **SHA256 摘要**保证一致性与可寻址性。
- Dockerfile 用指令产生层；**多阶段构建**与**合理的 COPY 顺序**能大幅优化体积与缓存。
- 运行时分层：`containerd`（CRI）→ `runc`（OCI）→ 可选的 gVisor/Kata 沙箱；Docker 只是最外层的 UX。
- 容器内写入不持久，持久化必须走**卷**。

在下一节，我们将这些「会跑的进程」交给一个编排系统统一管理——进入 **Kubernetes 集群架构与控制平面**。
