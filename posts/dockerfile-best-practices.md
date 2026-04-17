## 核心结论

Dockerfile 最佳实践的目标只有三个：镜像更小、构建更快、运行更安全。最有效的手段是把“编译”和“运行”拆开，也就是多阶段构建。所谓多阶段构建，就是前面的阶段负责安装编译器、下载依赖、产出制品，最后一个阶段只保留运行程序真正需要的内容。

核心结论可以压缩成四条：

| 结论 | 直接作用 | 典型写法 |
|---|---|---|
| 多阶段构建 | 减小镜像体积，隔离编译环境 | `FROM ... AS builder` + `COPY --from=builder ...` |
| 缓存友好的指令顺序 | 减少重复安装依赖 | 先 `COPY package*.json`，再 `RUN npm ci`，最后 `COPY src` |
| 控制构建上下文 | 减少无效传输和缓存失效 | `.dockerignore` 排除 `node_modules`、`.git`、日志 |
| 非 root + 明确入口 | 提升运行安全与可维护性 | `USER app`，`ENTRYPOINT` 放主程序，`CMD` 放默认参数 |

缓存命中的本质可以写成：

$$
Cache\_hit(L)=true \iff \text{指令文本不变} \land \text{输入校验和不变}
$$

这条规则决定了 Dockerfile 的顺序不是“写起来顺手就行”，而是必须按照“最稳定的输入在前，最常变的输入在后”来安排。

一个最小新手例子：

```dockerfile
FROM node:18 AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:18-slim
WORKDIR /app
COPY --from=builder /app/dist ./dist
ENTRYPOINT ["node", "dist/server.js"]
```

如果你把“安装依赖”和“复制全部源码”写反，任何一个业务文件变动都会导致依赖层重跑，构建时间会明显变差。

---

## 问题定义与边界

Dockerfile 最佳实践讨论的不是“容器能不能跑起来”，而是“在长期工程中，镜像是否容易维护、迭代是否足够快、默认行为是否安全”。这里有几个边界必须先说清楚。

第一，Docker 构建不是只读 Dockerfile。它还读取构建上下文。构建上下文就是 `docker build` 发送给 Docker 引擎的文件集合，通常是当前目录。白话说，目录里你没排除的东西，Docker 都可能拿去参与计算。

第二，缓存不是“只看这一行命令”。缓存依赖上游层和输入文件。一个 `COPY . .` 如果把 `.git`、`node_modules`、测试产物一起带进去，哪怕程序逻辑没变，也可能让后续层全部失效。

第三，运行安全不只等于“镜像里没漏洞”。容器默认以 root 用户运行，意味着进程拥有更高权限。白话说，一旦应用被利用，攻击者拿到的是管理员身份，不是普通账号。

第四，启动命令也有边界。`ENTRYPOINT` 用来定义主程序，`CMD` 用来提供默认参数。很多初学者把两者都写成完整命令，结果是参数覆盖、信号传递异常，容器停止不干净。

可以把问题、影响和对策先压缩成一张表：

| 问题 | 影响 | 对策 |
|---|---|---|
| `.dockerignore` 缺失 | 上下文过大，构建慢，缓存不稳定 | 显式排除依赖目录、版本库、日志、临时文件 |
| `COPY . .` 过早出现 | 依赖安装层频繁失效 | 先复制依赖描述文件，再安装依赖 |
| 单阶段包含编译器 | 镜像膨胀，攻击面变大 | 用多阶段构建只拷贝制品 |
| 默认 root 用户 | 权限过大，安全风险上升 | 创建专用用户并 `USER` 切换 |
| `ENTRYPOINT`/`CMD` 混用错误 | 参数不可控，信号传播差 | 主程序放 `ENTRYPOINT`，默认参数放 `CMD` |

构建时间的粗略关系可以写成：

$$
BuildTime \propto ContextSize \times LayersRebuilt
$$

这不是精确公式，但足够说明工程事实：上下文越大、重建层越多，构建越慢。

一个新手常见反例是：目录里有 `node_modules`，但 `.dockerignore` 没排除。结果每次 `COPY . .` 都把几百 MB 文件重新纳入校验和计算，后续 `RUN npm ci` 和构建层都容易失效。

---

## 核心机制与推导

Dockerfile 优化的核心机制其实只有两条：缓存边界设计，以及构建产物隔离。

先看缓存边界。假设有一层负责安装依赖：

$$
Cache\_hit(RUN\ deps) \iff depsFilesChecksumUnchanged
$$

这句话的意思是：只要依赖文件没变，安装依赖这层就应该复用缓存。因此，正确顺序是先复制依赖描述文件，再执行安装，再复制经常变化的源码。

玩具例子：

一个 Node 项目有四个文件：

| 文件 | 是否变化 | 对哪层有影响 |
|---|---|---|
| `package.json` | 偶尔变化 | 依赖安装层 |
| `package-lock.json` | 偶尔变化 | 依赖安装层 |
| `src/index.js` | 经常变化 | 编译层 |
| `README.md` | 偶尔变化 | 如果不复制进镜像，最好不影响 |

正确 Dockerfile 片段：

```dockerfile
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY src ./src
RUN npm run build
```

当你只修改 `src/index.js` 时，`RUN npm ci` 这一层可以直接命中缓存；当你修改 `package-lock.json` 时，依赖层才需要重跑。这就是“把变化频率低的输入放前面”的含义。

再看多阶段构建。多阶段的本质不是“写两个 `FROM`”，而是把文件流切开：

```text
builder stage
  ├─ 安装编译依赖
  ├─ 拉取源码
  ├─ 编译/打包
  └─ 产出 artifact
           │
           ▼
runtime stage
  ├─ 只复制 artifact
  ├─ 配置运行用户
  └─ 启动主程序
```

这样做有三个直接收益：

| 做法 | 结果 |
|---|---|
| 编译器留在 builder | 最终镜像不带 gcc、make、headers |
| 只复制运行文件 | 体积更小，漏洞面更少 |
| builder 与 runtime 解耦 | 可以替换更轻的运行基镜像 |

真实工程例子是 ML 或 HPC 镜像。CUDA 基础镜像通常分成 `devel` 和 `runtime` 两类。`devel` 含编译工具链，适合构建；`runtime` 只保留运行库，适合部署。如果你在单阶段里直接用 `nvidia/cuda:*devel*` 运行线上服务，镜像会把大量头文件、编译器和工具链一起带上，体积和攻击面都明显偏大。更稳妥的做法是：

1. 用 `devel` 镜像编译扩展、安装需要编译的 Python 包。
2. 在最终阶段切到对应版本的 `runtime` 镜像。
3. 只拷贝虚拟环境、wheel、二进制制品和必需配置。

这也是为什么实际 GPU 项目里，依赖安装顺序尤其重要。CUDA 版本、PyTorch 版本、系统库版本之间有强耦合，先后顺序错了，不只是缓存差，还可能直接安装失败。

---

## 代码实现

下面给出一个完整的新手版实现，涵盖 `.dockerignore`、多阶段构建、非 root 用户、`ENTRYPOINT`/`CMD` 分工。

`.dockerignore`：

```gitignore
node_modules
dist
.git
.gitignore
Dockerfile
*.log
.env
coverage
__pycache__
*.pyc
```

Node 服务示例 Dockerfile：

```dockerfile
# builder: 负责安装依赖和构建产物
FROM node:18 AS builder
WORKDIR /app

# 先复制依赖描述文件，保证依赖层缓存稳定
COPY package.json package-lock.json ./
RUN npm ci

# 再复制业务代码，避免代码改动冲掉依赖缓存
COPY . .
RUN npm run build

# runtime: 只保留运行需要的内容
FROM node:18-slim
WORKDIR /app

# 创建非 root 用户
RUN groupadd -r app && useradd -r -g app app

# 只拷贝运行时所需文件
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/node_modules ./node_modules

USER app
EXPOSE 3000

# ENTRYPOINT 定义主程序，CMD 提供默认参数
ENTRYPOINT ["node", "dist/server.js"]
CMD ["--port=3000"]
```

文件流可以理解为：

```text
builder stage -> build artifact -> runtime stage
```

如果是 Python 小项目，不需要编译本地扩展，可以更简单；如果需要编译扩展，建议把 wheel 构建和运行环境拆开。下面给一个可运行的 Python 玩具程序，用来模拟“指令顺序是否会扩大重建范围”的问题：

```python
from hashlib import sha256

def checksum(*parts: str) -> str:
    h = sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()

def cache_hit(instruction: str, *inputs: str, old_key: str | None = None) -> tuple[bool, str]:
    new_key = checksum(instruction, *inputs)
    return (old_key == new_key, new_key)

# 第一次构建
deps_key_hit, deps_key = cache_hit("RUN npm ci", "package.json:v1", "package-lock.json:v1", old_key=None)
assert deps_key_hit is False

# 只改源码，不改依赖文件，依赖层应命中缓存
deps_key_hit2, deps_key2 = cache_hit("RUN npm ci", "package.json:v1", "package-lock.json:v1", old_key=deps_key)
assert deps_key_hit2 is True
assert deps_key2 == deps_key

# 改 lock 文件，依赖层必须失效
deps_key_hit3, deps_key3 = cache_hit("RUN npm ci", "package.json:v1", "package-lock.json:v2", old_key=deps_key)
assert deps_key_hit3 is False
assert deps_key3 != deps_key

print("cache model ok")
```

`ENTRYPOINT` 和 `CMD` 的职责建议这样区分：

| 项 | 含义 | 适用场景 |
|---|---|---|
| `ENTRYPOINT` | 容器启动时必须执行的主程序 | Web 服务、worker、固定二进制 |
| `CMD` | 默认参数或默认命令 | 端口、模式、可覆盖参数 |
| 两者组合 | `ENTRYPOINT` 固定程序，`CMD` 传默认参数 | 最常见、最稳定 |
| 只写 `CMD` | 适合临时调试镜像 | 交互式或一次性任务 |

---

## 工程权衡与常见坑

工程里最常见的问题不是“不会写 Dockerfile”，而是“写得能跑，但迭代代价很高”。

先看常见坑：

| 坑 | 触发条件 | 修复措施 |
|---|---|---|
| `.git`、`node_modules` 被送入上下文 | `.dockerignore` 不完整 | 明确排除所有无关目录 |
| 把 `COPY . .` 放在依赖安装前 | 业务代码高频变动 | 先复制锁文件，再装依赖 |
| 最终镜像包含编译器 | 单阶段构建 | 使用 builder/runtime 两阶段 |
| 容器默认 root | 未创建业务用户 | `groupadd`/`useradd` 后 `USER` 切换 |
| `ENTRYPOINT ["npm","start"]` | 用包管理器做主进程 | 直接运行应用进程，如 `node dist/server.js` |
| `ENTRYPOINT` 与 `CMD` 都写完整命令 | 参数覆盖和可读性差 | `ENTRYPOINT` 放程序，`CMD` 放参数 |

一个典型错误是：

```dockerfile
ENTRYPOINT ["npm", "start"]
CMD ["npm", "start"]
```

这会导致职责混乱。更稳妥的修正是：

```dockerfile
ENTRYPOINT ["node", "dist/server.js"]
CMD ["--port=3000"]
```

原因有两个。第一，`npm` 不是你的业务进程，它只是启动器。第二，容器信号处理要尽量交给真正的主进程，否则停止、重启和优雅退出都可能不稳定。

再看 ML 镜像的专项坑。GPU 镜像里最常见的是基础镜像选型和依赖顺序错误：

| 场景 | 常见错误 | 更合理做法 |
|---|---|---|
| 训练镜像 | 直接在 `runtime` 里编译 | builder 用 `devel`，runtime 用 `runtime` |
| 推理镜像 | 把 Jupyter、编译器、调试工具一起带上 | 线上镜像只保留推理依赖 |
| Python 依赖安装 | 先复制全量源码再 `pip install` | 先复制 `requirements.txt` 或 `pyproject.toml` |
| CUDA 版本切换 | 基础镜像与框架版本不对齐 | 先固定 CUDA，再匹配框架 wheel |

真实工程里，GPU 镜像往往很大，基础镜像一旦选错，后续所有层都被放大。比如你需要 `torch+cu121`，就应该优先固定到兼容的 CUDA 12.1 基镜像，并把 Python 依赖描述文件单独复制，先完成依赖安装，再复制业务代码。这样一方面缓存更稳定，另一方面问题定位也更清楚：失败是出在系统库、框架轮子，还是业务代码。

---

## 替代方案与适用边界

不是所有项目都必须上复杂的多阶段构建。关键在于“是否有明显的编译步骤”和“是否进入长期维护”。

可以这样判断：

| 方案 | 适用场景 | 优点 | 边界 |
|---|---|---|---|
| 单阶段构建 | 小脚本、临时调试、PoC | 简单直接 | 体积和安全性通常较差 |
| 多阶段构建 | 正式服务、需编译项目、前后端构建 | 体积小、边界清晰 | Dockerfile 稍复杂 |
| BuildKit 缓存导入 | CI/CD、跨机器复用缓存 | 重建更快 | 依赖构建系统支持 |

一个简单 Python 脚本的边界例子：

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "main.py"]
```

如果它只是内部工具、没有本地扩展、构建频率低，这样完全可以接受。但当项目变成正式服务，或者开始依赖需要编译的包，就应当转向多阶段。

BuildKit 是进一步的增强方案。它不是替代多阶段，而是在多阶段之外继续优化缓存复用。常见示例：

```bash
docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t myapp:latest .
```

在 CI 中再配合 `buildx` 的远程缓存导入导出，可以让不同机器之间共享构建结果。适用边界很明确：如果团队构建频率高、依赖安装慢、CI 时间昂贵，BuildKit 很值得；如果只是个人本地偶尔构建，先把 Dockerfile 顺序写对，收益往往更大。

最终可以把选择压缩成一句话：单阶段适合短期和简单场景，多阶段适合正式交付，BuildKit 适合高频构建团队。

---

## 参考资料

| 来源 | 摘要 | 用途 |
|---|---|---|
| Docker 官方 Best Practices | 官方推荐多阶段、排序、最小镜像、缓存思路 | 作为 Dockerfile 写法基线 |
| Docker 官方关于 `RUN`/`CMD`/`ENTRYPOINT` 说明 | 解释三者职责和覆盖关系 | 规范容器启动方式 |
| Better Stack Docker Build Best Practices | 强调缓存命中、上下文控制、层顺序 | 理解构建性能优化 |
| NVIDIA HPC/容器相关文章 | 展示多阶段在 CUDA/HPC 镜像中的体积收益 | 说明 ML/GPU 专项实践 |

引用列表：

1. Docker Docs, *Building best practices*.
2. Docker Blog/Docs, *RUN vs CMD vs ENTRYPOINT*.
3. Better Stack, *Docker Build Best Practices*.
4. NVIDIA Developer Blog, *Building HPC Containers Demystified*.

继续深究的顺序建议是：先看 Docker 官方文档确认推荐指令顺序，再看 `ENTRYPOINT`/`CMD` 的职责区别，最后看 Better Stack 与 NVIDIA 的案例，理解为什么这些规则在真实工程里能节省时间和镜像体积。
