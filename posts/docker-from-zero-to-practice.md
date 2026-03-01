## Docker 是什么

Docker 是一个**容器化平台（Containerization Platform）**，核心功能是将程序及其全部运行依赖打包为标准化单元（容器），确保在任何环境中行为一致。

软件运行不仅依赖代码本身，还依赖操作系统版本、语言运行时、依赖库版本、环境变量、端口配置等运行环境。环境不一致是导致"本地正常、部署报错"的根本原因。Docker 通过将程序与其完整环境打包在一起解决这个问题。

---

## 容器 vs 虚拟机

容器和**虚拟机（Virtual Machine）**都是环境隔离方案，但实现方式和性能特征差异显著。

### 虚拟机

虚拟机通过 **Hypervisor（虚拟化层）**在一台物理机上模拟出多台完整的计算机，每台拥有独立的操作系统内核、驱动程序和文件系统。隔离彻底，但资源开销大：

- 启动需要几十秒到几分钟
- 每个 VM 占用 GB 级内存（操作系统本身就要几百 MB）
- 资源利用率低：运行一个 Python 脚本需要先启动整个 Linux 系统

### 容器

Docker 使用的**容器（Container）**技术共享宿主机的操作系统内核，只隔离文件系统、进程、网络等运行上下文：

```
虚拟机                        Docker 容器
┌──────────────────┐          ┌──────────────────┐
│  App A  │  App B │          │  App A  │  App B │
├─────────┼────────┤          ├─────────┼────────┤
│  OS(2G) │ OS(2G) │          │  Lib A  │  Lib B │  ← 只打包依赖
├─────────┴────────┤          ├─────────┴────────┤
│    Hypervisor    │          │  Docker Engine   │  ← 共享内核
├──────────────────┤          ├──────────────────┤
│  Host OS Kernel  │          │  Host OS Kernel  │  ← 同一个内核
└──────────────────┘          └──────────────────┘
```

| 对比项    | 虚拟机       | Docker 容器 |
| --------- | ------------ | ----------- |
| 启动时间  | 分钟级       | 秒级（通常 < 1s） |
| 内存占用  | GB 级        | MB 级       |
| 隔离程度  | 完全隔离     | 进程级隔离  |
| 镜像大小  | 几 GB        | 几十 MB 起  |
| 适合场景  | 强安全隔离   | 微服务、CI/CD、开发环境 |

> Docker 底层依赖 Linux 内核的两个关键特性：**Namespace**（隔离进程视图）和 **cgroups**（限制资源使用）。容器本质上是一种受限的进程，而非独立的虚拟机。

---

## 三个核心概念：镜像、容器、仓库

### 镜像（Image）

镜像是一个**只读的文件系统快照**，包含运行程序所需的全部内容：操作系统基础层、运行时（如 Python、Node.js）、依赖库、配置文件和应用代码。

镜像采用**分层构建（Layered Build）**：

```
┌─────────────────────────────┐  ← 应用代码层（最上层，通常只有几 KB）
├─────────────────────────────┤  ← 依赖层（pip install 的结果）
├─────────────────────────────┤  ← Python 3.11 运行时
├─────────────────────────────┤  ← Ubuntu 22.04 基础层（底层，可跨项目共享）
└─────────────────────────────┘
```

每一层可被缓存和复用。如果多个项目使用同一个 Ubuntu 基础层，该层只需下载一次、存储一份。

### 容器（Container）

容器是镜像的**运行实例**。Docker 在镜像的只读层之上添加一个**可写层（Writable Layer）**，程序运行时产生的文件变更写入可写层，镜像本身保持不变。

同一个镜像可以启动多个容器，它们共享镜像的只读内容，各自拥有独立的可写层。

### 仓库（Registry）

仓库是镜像的存储与分发中心。

- **Docker Hub**（`hub.docker.com`）：官方公共仓库，托管大量官方和社区镜像
- **私有仓库**：企业通常搭建自己的仓库（如 AWS ECR、阿里云 ACR）存放内部镜像

基本操作：`docker pull` 从仓库拉取镜像，`docker push` 将本地镜像推送到仓库。

---

## 安装 Docker

### macOS / Windows

前往 [Docker Desktop](https://www.docker.com/products/docker-desktop/) 官网下载安装包。Docker Desktop 提供 GUI 界面，同时安装 Docker CLI。

### Linux（以 Ubuntu 为例）

```bash
# 1. 更新包索引，安装依赖
sudo apt update
sudo apt install -y ca-certificates curl gnupg

# 2. 添加 Docker 官方 GPG 密钥
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 3. 添加 Docker 软件源
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 4. 安装 Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# 5. 让当前用户无需 sudo 就能运行 docker
sudo usermod -aG docker $USER
newgrp docker
```

验证安装：

```bash
docker run hello-world
```

输出 `Hello from Docker!` 表示安装成功。

---

## 基础命令

### 拉取与运行镜像

```bash
# 从 Docker Hub 拉取 nginx 官方镜像
docker pull nginx:1.27

# 运行 nginx 容器
# -d：后台运行（detached mode）
# -p 8080:80：宿主机 8080 端口映射到容器 80 端口
# --name my-nginx：指定容器名称
docker run -d -p 8080:80 --name my-nginx nginx:1.27
```

访问 `http://localhost:8080` 可看到 nginx 欢迎页。

### 管理容器

```bash
# 查看正在运行的容器
docker ps

# 查看所有容器（包括已停止的）
docker ps -a

# 停止容器
docker stop my-nginx

# 启动已停止的容器
docker start my-nginx

# 删除容器（必须先停止）
docker rm my-nginx

# 停止并立即删除
docker rm -f my-nginx
```

### 进入容器内部

```bash
# 在容器中打开交互式 bash 终端
docker exec -it my-nginx bash

# 退出容器终端（容器继续运行）
exit
```

`exec` 用于调试——可以进入容器内部查看文件、检查进程状态。

### 查看日志

```bash
# 查看容器日志
docker logs my-nginx

# 实时跟踪日志（类似 tail -f）
docker logs -f my-nginx
```

### 管理镜像

```bash
# 查看本地所有镜像
docker images

# 删除镜像（需先删除使用该镜像的容器）
docker rmi nginx:1.27

# 清理所有未被使用的镜像、容器、网络
docker system prune -a
```

---

## Dockerfile：构建自定义镜像

Dockerfile 是一个文本文件，每一行是一条指令，Docker 按顺序执行这些指令构建镜像。

### 示例：打包 Python Flask 应用

应用代码：

```python
# app.py
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello from Docker!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

```
# requirements.txt
flask==3.0.3
```

Dockerfile：

```dockerfile
# 1. 指定基础镜像（Python 3.12 精简版）
FROM python:3.12-slim

# 2. 设置容器内工作目录
WORKDIR /app

# 3. 先复制依赖文件（利用层缓存，避免每次改代码都重新安装依赖）
COPY requirements.txt .

# 4. 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制应用代码
COPY app.py .

# 6. 声明容器监听端口（文档作用，不自动映射）
EXPOSE 5000

# 7. 容器启动时执行的命令
CMD ["python", "app.py"]
```

构建并运行：

```bash
# 在 Dockerfile 所在目录执行，-t 指定镜像名和标签
docker build -t my-flask-app:1.0 .

# 运行容器
docker run -d -p 5000:5000 --name flask-demo my-flask-app:1.0

# 验证
curl http://localhost:5000
# 输出：Hello from Docker!
```

### 关键指令速查

| 指令        | 作用                                           |
| ----------- | ---------------------------------------------- |
| `FROM`      | 指定基础镜像，每个 Dockerfile 必须以此开头    |
| `WORKDIR`   | 设置后续指令的工作目录（自动创建）            |
| `COPY`      | 将本地文件复制到镜像内                        |
| `RUN`       | **构建时**执行命令（每条 RUN 产生一个新层）   |
| `ENV`       | 设置环境变量                                   |
| `EXPOSE`    | 声明容器监听的端口（仅作文档说明）            |
| `CMD`       | **容器启动时**执行的默认命令（可被覆盖）      |
| `ENTRYPOINT`| 容器入口点，与 CMD 配合使用，不易被覆盖       |

### 层缓存机制

Dockerfile 中先 `COPY requirements.txt`、再 `COPY app.py`，而非一次性复制所有文件，这利用了 Docker 的**层缓存（Layer Cache）**机制：

- Docker 构建时，如果某一层的输入没有变化，直接使用上次的缓存结果
- `requirements.txt` 的改动频率远低于 `app.py`
- 每次修改代码重新构建时，`pip install` 层直接命中缓存，构建时间从几十秒降到几秒

---

## 数据持久化：Volume

容器的可写层随容器删除而消失。对于数据库、日志等需要持久化的数据，使用 **Volume（数据卷）**。

### 具名 Volume（推荐）

```bash
# 创建一个名为 pgdata 的 volume
docker volume create pgdata

# 运行 PostgreSQL，将数据目录挂载到 volume
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=secret \
  -v pgdata:/var/lib/postgresql/data \
  postgres:16

# 即使删除容器，pgdata 里的数据依然存在
docker rm -f postgres
docker volume ls  # pgdata 还在

# 重新启动并挂载，数据完整恢复
docker run -d --name postgres -v pgdata:/var/lib/postgresql/data postgres:16
```

### Bind Mount（绑定挂载）

将宿主机目录直接挂载到容器内，常用于开发时实时同步代码：

```bash
# 将当前目录挂载到容器的 /app 目录
# 修改本地代码，容器内立即生效（无需重新构建镜像）
docker run -d \
  -p 5000:5000 \
  -v $(pwd):/app \
  my-flask-app:1.0
```

---

## Docker Compose：多容器编排

实际项目通常包含多个服务。一个典型的 Web 应用至少需要：Web 服务、数据库、缓存（Redis）。Docker Compose 用一个 `docker-compose.yml` 文件描述所有服务配置，一条命令启动整个系统。

### 示例：Web + PostgreSQL + Redis

```yaml
# docker-compose.yml
version: "3.9"

services:

  # Web 应用
  web:
    build: .                        # 用当前目录的 Dockerfile 构建
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://user:secret@db:5432/mydb
      - REDIS_URL=redis://cache:6379
    depends_on:
      db:
        condition: service_healthy  # 等 db 健康检查通过再启动
      cache:
        condition: service_started
    volumes:
      - .:/app                      # 开发时挂载代码（热更新）

  # PostgreSQL 数据库
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: mydb
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d mydb"]
      interval: 5s
      timeout: 5s
      retries: 5

  # Redis 缓存
  cache:
    image: redis:7-alpine
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:
```

操作命令：

```bash
# 启动所有服务（-d 后台运行）
docker compose up -d

# 查看所有服务状态
docker compose ps

# 查看某个服务的日志
docker compose logs -f web

# 进入某个服务的容器
docker compose exec web bash

# 停止并删除所有容器（保留 volume）
docker compose down

# 停止并删除容器 + volume（清空数据）
docker compose down -v

# 重新构建镜像后启动（代码变更时使用）
docker compose up -d --build
```

### Compose 的核心价值

- **网络自动配置**：Compose 为所有服务创建私有网络，服务间通过服务名互相访问（如 `db:5432`），无需关心 IP
- **启动顺序控制**：`depends_on` 确保依赖服务先启动
- **一键管理**：整个应用栈的启动、停止、重建只需一条命令

---

## 生产实践：构建更小更安全的镜像

### 多阶段构建（Multi-stage Build）

以 Go 应用为例。Go 编译后产出静态二进制文件，运行时不需要编译环境：

```dockerfile
# 阶段一：构建
FROM golang:1.22-alpine AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
# 编译为静态二进制，禁用 CGO
RUN CGO_ENABLED=0 GOOS=linux go build -o server .

# 阶段二：运行（极简镜像）
FROM scratch
# scratch 是空镜像，只放入编译好的二进制文件
COPY --from=builder /build/server /server
EXPOSE 8080
ENTRYPOINT ["/server"]
```

镜像大小对比：

| 方案                | 镜像大小 |
| ------------------- | -------- |
| `golang:1.22`       | ~800 MB  |
| `golang:1.22-alpine`| ~300 MB  |
| 多阶段 + `scratch`  | ~10 MB   |

镜像缩小约 80 倍，攻击面也大幅减少——没有 shell、没有包管理器。

### 其他最佳实践

**1. 使用非 root 用户运行进程**

```dockerfile
FROM python:3.12-slim
# 创建普通用户
RUN useradd -m -u 1001 appuser
WORKDIR /app
COPY --chown=appuser:appuser . .
# 切换到普通用户
USER appuser
CMD ["python", "app.py"]
```

**2. 使用 .dockerignore**

`.dockerignore` 防止无关文件被打包进镜像，功能类似 `.gitignore`：

```
# .dockerignore
.git
.env
__pycache__
*.pyc
node_modules
*.log
.DS_Store
```

**3. 固定镜像版本**

```dockerfile
# 不推荐：latest 可能在某天更新，引入破坏性变更
FROM python:latest

# 推荐：固定版本，确保可复现
FROM python:3.12.3-slim-bookworm
```

---

## 网络：容器间通信

### Bridge 网络（默认）

每个容器有独立的网络命名空间，通过虚拟网桥连接。Docker Compose 自动为每组服务创建一个 bridge 网络。

```bash
# 手动创建自定义网络
docker network create my-net

# 将两个容器加入同一网络，可通过容器名互相访问
docker run -d --name app --network my-net my-flask-app:1.0
docker run -d --name db  --network my-net postgres:16

# app 容器内可以通过 "db" 主机名访问数据库
```

### 端口映射

容器内部端口对外部不可见，必须通过 `-p` 显式映射：

```bash
# 格式：-p <宿主机端口>:<容器端口>
docker run -p 8080:80 nginx        # 访问宿主机 8080 → 容器 80
docker run -p 127.0.0.1:8080:80 nginx  # 只允许本地访问
```

---

## 总结

Docker 的核心体系可以分为五个层次：

```
概念层  →  镜像 / 容器 / 仓库
操作层  →  docker run / exec / logs / ps / rm
构建层  →  Dockerfile / 层缓存 / 多阶段构建
编排层  →  Docker Compose
生产层  →  非 root / .dockerignore / 固定版本 / 多阶段构建
```

Docker 改变的是软件交付方式——从"发布代码"变成"发布环境"。这也是 Kubernetes、微服务、CI/CD 流水线等现代基础设施得以普及的基础。

**进阶方向：**

- **Kubernetes（k8s）**：容器数量从个位数增长到数百个时，需要自动调度和弹性扩缩容
- **Docker Swarm**：Docker 原生的轻量集群方案，学习成本低于 k8s
- **镜像安全扫描**：`docker scout` / `trivy` 检测镜像中的 CVE 漏洞
- **BuildKit**：Docker 下一代构建引擎，支持并行构建和缓存导出
