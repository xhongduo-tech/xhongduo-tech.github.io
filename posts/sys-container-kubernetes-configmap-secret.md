## 核心结论

ConfigMap 和 Secret 都是 Kubernetes 的 API 对象。API 对象就是“保存在集群控制面的结构化记录”，可以被 `kubectl` 创建、查询和更新。它们解决的是同一个问题：把配置从镜像里拆出来，让同一个镜像在不同环境下复用。

区别主要有三点：

| 项 | ConfigMap | Secret |
|---|---|---|
| 目的 | 非敏感设定 | 敏感凭据 |
| 存储格式 | 纯文本键值 | Base64 编码（可启用加密） |
| 注入方式 | `env` / `envFrom` / `volume` | `env` / `envFrom` / `volume` |
| 更新策略 | 环境变量需重启，卷可延迟刷新 | 同上 |

第一，ConfigMap 适合数据库地址、端口、开关、模板片段这类“可公开但需可变”的配置。Secret 适合密码、令牌、证书、私钥这类“泄露后有安全后果”的数据。

第二，二者注入 Pod 的方式基本一致，要么变成环境变量，要么变成挂载文件。环境变量就是“进程启动时读入内存的一组键值”，它通常只在容器启动时生效；卷挂载就是“把对象内容映射成容器里的文件”，后续有机会热更新，但应用不一定会自动重读。

第三，Secret 默认不是“天然加密存储”，而是先做 Base64 编码。Base64 是“把二进制转成可打印字符的编码方式”，不是加密算法。真正的加密能力依赖 API Server 的 `EncryptionConfiguration`、etcd 加密和 KMS。

一个最常见的组合是：把数据库地址写进 ConfigMap，把密码和证书写进 Secret，Pod 通过 `envFrom` 读取前者，再把后者挂载到 `/etc/creds`。这样镜像构建完成后，配置仍然可以改，不需要重新打镜像。

---

## 问题定义与边界

问题可以表述为：

> 如何在不重建镜像的前提下，为 Pod 提供可变配置，并在需要时安全地管理敏感信息？

这里有两个目标：

1. 配置与镜像解耦。解耦就是“两个东西可以独立变化”。镜像负责程序代码，ConfigMap/Secret 负责运行时数据。
2. 敏感信息单独治理。治理就是“有权限、审计、轮换、加密的一整套管理方式”。

但 ConfigMap/Secret 不是无限制的万能存储。它们有明确边界。

第一条边界是容量。单个对象总大小通常受限于：

$$
ObjectSize \le 1{,}048{,}576 \text{ bytes}
$$

也就是 1 MiB。这个限制不是建议，而是工程边界。比如一个 ConfigMap 放了 1024 个键，每个值约 1 KB，再加上键名和对象元数据，就已经非常接近上限。超出后，创建或更新可能失败，Pod 也可能因为引用失败而无法按预期启动。

第二条边界是 Secret 的安全语义。很多初学者看到 Base64 后会误以为已经“加密”。不是。只要拿到对象内容，任何人都可以轻易解码。如果 etcd 未启用加密，或者权限控制过宽，那么 Secret 的安全性就只是“比直接写在 YAML 里好一点”，不是高强度保护。

第三条边界是更新传播机制。ConfigMap/Secret 不是“改完立刻全局生效”的分布式配置中心。环境变量方式几乎总是要重启 Pod；卷方式虽然可以刷新，但刷新存在延迟，而且应用程序未必监听文件变化。

第四条边界是挂载方式。若你使用 `subPath`，也就是“只把卷中的某个文件单独挂进容器某个路径”，那么后续对象更新通常不会反映到容器里。这是生产环境里非常常见的坑。

一个玩具例子可以直接说明这些边界：

假设你有一个小服务，只需要两个配置：

- `DB_HOST=db.prod.svc.cluster.local`
- `FEATURE_X=true`

这两个值适合放进 ConfigMap。然后再加一个：

- `DB_PASSWORD=abc123`

这就应该放进 Secret。这样设计的核心不是“分类美观”，而是后续可以分别控制谁能看、谁能改、如何轮换。

一个真实工程例子是多环境部署。开发、测试、生产三套环境共用同一个镜像，但每个环境的数据库地址、缓存地址、第三方 API Token 都不同。把这些值固化进镜像意味着每切一次环境就要重新构建、重新发布，流程会非常脆弱；而把它们拆到 ConfigMap/Secret 后，镜像只构建一次，环境差异通过对象注入解决。

---

## 核心机制与推导

先看对象层。ConfigMap 和 Secret 都存储在 Kubernetes 控制平面里。控制平面就是“负责保存期望状态并驱动集群达成该状态的一组组件”。当你执行 `kubectl apply -f xxx.yaml` 时，本质是在创建或更新这些对象。

再看容器读取路径，主要有两条：

1. 环境变量注入
2. 卷挂载注入

环境变量路径的机制最简单。Pod 创建时，kubelet 会把 ConfigMap/Secret 的键值展开成进程环境变量，再启动容器。结论也最直接：

$$
EnvVarUpdate \Rightarrow \text{Pod restart is usually required}
$$

原因不是 Kubernetes “不支持更新”，而是操作系统进程模型决定了已启动进程不会自动重新装载父级为它准备好的环境变量。除非应用自己额外提供控制接口并重启自身，否则你改了对象，老进程并不会感知。

卷挂载路径稍复杂。kubelet 会把对象内容投影成文件，写入节点上的卷目录，再挂入容器。对象更新后，kubelet 会周期性同步这些文件，所以你看到的是“文件可能变了”。这里的传播延迟可粗略理解为：

$$
VolumeRefreshDelay \approx \text{kubelet sync period} + \text{cache propagation delay}
$$

在默认设置下，常被观察到的量级是几十秒，常见描述是 30 到 90 秒左右。它不是严格 SLA，而是经验量级。也就是说，卷方式可以热更新，但不是实时推送。

Secret 的存储链路也要拆开看：

$$
SecretStorage = base64(data) + optional\ encryption
$$

这条式子很重要。`base64(data)` 只是编码，目的是让二进制字节能安全出现在 JSON/YAML 里；`optional encryption` 才是可选的真正加密。如果没有后半段，Secret 在控制面和存储层面并不等于“强安全”。

玩具例子可以这样推导：

- `app-config` 中有 `LOG_LEVEL=INFO`
- Pod 用 `envFrom` 读取它
- 你把值改成 `DEBUG`

结果是：对象确实更新了，但容器里运行中的进程仍旧看到旧值。因为它启动时已经拿到一份环境变量快照。要让新值生效，通常要滚动重启 Deployment。

再看卷方式：

- `app-secret` 中有 `tls.crt`
- Pod 把它挂到 `/etc/certs/tls.crt`
- 你更新了 Secret

结果是：文件有机会在一段时间后刷新。但如果应用程序在启动时只读一次证书并缓存到内存，那即便磁盘文件更新，连接层仍然继续使用旧证书。这就是“卷可热更新，不代表应用自动热重载”。

真实工程里常见的做法是把配置内容做哈希。哈希就是“把任意长度输入映射成固定长度指纹”。例如把 ConfigMap/Secret 内容算成一个 `sha256`，写入 Deployment 注解 `checksum/config`。一旦配置变化，注解也变化，Deployment 模板发生改变，控制器就会触发滚动更新。这不是 Kubernetes 内建的“配置热更新”，而是利用 Deployment 的变更检测机制来显式重启 Pod。

---

## 代码实现

下面给一个面向新手的最小可运行思路。先定义 ConfigMap 和 Secret，再让 Pod 同时以环境变量和卷的方式读取。

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DB_HOST: "postgres.default.svc.cluster.local"
  FEATURE_X: "true"
  LOG_LEVEL: "INFO"
---
apiVersion: v1
kind: Secret
metadata:
  name: app-secret
type: Opaque
data:
  password: c2VjcmV0MTIz
  tls.crt: ZmFrZS1jZXJ0LWRhdGE=
---
apiVersion: v1
kind: Pod
metadata:
  name: app
spec:
  containers:
  - name: web
    image: nginx
    envFrom:
    - configMapRef:
        name: app-config
    env:
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: app-secret
          key: password
    volumeMounts:
    - name: certs
      mountPath: /etc/certs
      readOnly: true
  volumes:
  - name: certs
    secret:
      secretName: app-secret
```

这个例子里有两条不同读取路径：

- `envFrom` 把 `app-config` 里的键直接变成环境变量。
- `secretKeyRef` 把 Secret 里的 `password` 注入到 `DB_PASSWORD`。
- `volumeMounts` 把 Secret 的内容作为文件挂到 `/etc/certs`。

如果用命令行创建，Secret 经常这样生成：

```bash
kubectl create secret generic app-secret \
  --from-literal=password=secret123
```

然后 Kubernetes 会帮你把值转成 Base64 写入对象。这里要注意：Base64 不是你手工“加密”了一次，而只是对象序列化格式的一部分。

下面给一个 Python 小程序，演示 Base64 只是可逆编码，并顺手模拟对象大小检查。它可以直接运行。

```python
import base64

def encode_secret(raw: str) -> str:
    return base64.b64encode(raw.encode("utf-8")).decode("ascii")

def decode_secret(encoded: str) -> str:
    return base64.b64decode(encoded.encode("ascii")).decode("utf-8")

def object_size_bytes(data: dict[str, str]) -> int:
    total = 0
    for k, v in data.items():
        total += len(k.encode("utf-8"))
        total += len(v.encode("utf-8"))
    return total

# Base64 是编码，不是加密
encoded = encode_secret("secret123")
assert encoded == "c2VjcmV0MTIz"
assert decode_secret(encoded) == "secret123"

# 玩具例子：简单配置大小远低于 1 MiB
config = {
    "DB_HOST": "postgres.default.svc.cluster.local",
    "FEATURE_X": "true",
    "LOG_LEVEL": "INFO",
}
assert object_size_bytes(config) < 1_048_576

# 接近上限的构造例子：1024 个 1KB 值
large_config = {f"k{i}": "x" * 1024 for i in range(1024)}
assert object_size_bytes(large_config) > 1_000_000
```

这个代码说明两件事：

1. Secret 的“密文”可以被直接解码回原文，所以不能把 Base64 当作安全边界。
2. ConfigMap/Secret 的大小是可量化的工程约束，不是抽象概念。

真实工程例子可以进一步写成 Deployment：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
      annotations:
        checksum/config: "sha256:9d9e-example"
    spec:
      containers:
      - name: api
        image: my-api:1.0.0
        envFrom:
        - configMapRef:
            name: api-config
        volumeMounts:
        - name: certs
          mountPath: /etc/certs
          readOnly: true
      volumes:
      - name: certs
        secret:
          secretName: api-secret
```

这里的 `checksum/config` 本身不会自动计算，但在 CI/CD 或模板系统里通常会根据 ConfigMap/Secret 内容生成。只要这个注解变化，Deployment 就会滚动替换 Pod，间接解决环境变量不会热更新的问题。

---

## 工程权衡与常见坑

先说权衡。

把配置做成环境变量，优点是简单，应用读取成本低，大多数语言框架原生支持；缺点是更新几乎必然依赖重启，而且变量值容易被错误日志、进程转储或调试接口暴露。

把配置做成卷文件，优点是更适合证书、私钥、多行配置文件，也更接近“可动态更新”；缺点是应用必须显式支持 reload，且传播有延迟。

再说常见坑。下面这张表在生产里非常常见。

| 坑 | 影响 | 规避 |
|---|---|---|
| Secret 只是 Base64 | 值仍可能在 etcd 中被直接解码 | 启用 `EncryptionConfiguration` + KMS，收紧 RBAC |
| Env 变量无法热更新 | 改对象后业务仍读旧值 | 使用滚动重启，或改为卷挂载 |
| `subPath` 无刷新 | 文件长期停留在旧版本 | 避免对会变的数据使用 `subPath` |
| kubelet 同步延迟 | 配置变更不能立刻生效 | 评估传播窗口，应用侧加监听与重试 |
| 应用只启动时读取一次文件 | 文件虽更新，进程仍用旧配置 | 实现 SIGHUP / fsnotify / 定时 reload |
| 密钥轮换只改 Secret 不重启 | 新旧凭据混用，连接异常 | 设计明确的轮换顺序和回滚方案 |

这里重点展开三个最容易出事故的问题。

第一，`subPath`。假设你把证书文件通过 `subPath` 单独挂到 `/etc/nginx/tls.crt`。Secret 更新后，很多人以为 kubelet 会替换这个文件。实际通常不会。因为 `subPath` 绑定的是一个具体路径映射，不会像整个投影卷那样跟随更新。结果就是集群里的 Secret 已经是新证书，Pod 里 Nginx 仍在用老证书，直到重启。

第二，密钥轮换。轮换就是“把旧凭据替换为新凭据的流程”。它不是单次修改，而是一个时序问题。比如数据库密码轮换，正确做法通常不是“先改 Secret，再希望业务自己变好”，而是：

1. 数据库先接受新旧两套密码的过渡期。
2. 更新 Kubernetes Secret。
3. 滚动重启应用 Pod，让新凭据加载。
4. 观察连接成功率。
5. 再撤掉旧密码。

如果你跳过第 3 步，很多进程会继续拿旧环境变量跑；如果你没有第 1 步，新的 Pod 起来前旧连接可能全部失效。

第三，权限模型。RBAC 就是“谁可以对哪些资源执行哪些动作的授权规则”。如果某个命名空间里的普通运维账号被授予了读取全部 Secret 的权限，那么即使你做了对象拆分，也只是把敏感信息集中到了另一个容易被读的位置。Secret 的价值很大程度上依赖访问控制，而不是名字叫 Secret。

一个真实工程例子是 TLS 证书更新失败。某团队把证书 Secret 挂给 Ingress 控制器，同时为了兼容旧目录结构使用了 `subPath`。证书续签后，Secret 对象内容是新的，但 Ingress 仍提供旧证书。排查时会发现：

- `kubectl get secret` 看起来正确
- Pod 没有重建
- 挂载路径文件没有变化

最终修复通常是：移除 `subPath`、直接挂整个目录、让进程支持 reload，或者在 Secret 变化时强制滚动重启。

---

## 替代方案与适用边界

ConfigMap/Secret 适合大多数“配置量不大、生命周期可控、更新频率中低”的场景。比如服务地址、开关项、数据库连接参数、证书文件、API Token。这已经覆盖了大量常规业务。

但当需求超出它的边界，就要考虑替代方案。

第一类替代方案是自动重启工具，例如 Stakater Reloader。它的作用不是替代 ConfigMap/Secret，而是补齐“对象变了，Pod 不会自动重启”的缺口。适合那些主要用环境变量注入、又希望少做手工重启的团队。

```yaml
metadata:
  annotations:
    reloader.stakater.com/auto: "true"
```

第二类替代方案是外部密钥系统，例如 HashiCorp Vault、云厂商 Secret Manager、External Secrets Operator。它们更适合下面这些场景：

- 凭据轮换频繁
- 需要审计谁在何时读取了密钥
- 需要短期动态凭据，例如数据库临时账号
- 希望不把长期敏感值直接存进 Kubernetes 原生 Secret

第三类替代方案是 CSI 驱动注入。CSI 就是“容器存储接口”，这里可把外部系统中的 Secret 作为卷直接提供给 Pod。例如 Vault CSI Provider 可以在挂载阶段从 Vault 动态获取数据，减少把机密长期落盘到 Kubernetes 对象中的必要性。

第四类方案是应用启动时自行拉取配置。这适合你已经有成熟配置中心或服务发现系统，且应用侧能处理鉴权、缓存、失败重试。但这会把复杂性从平台层转移到应用层，不一定适合初级团队。

选择边界可以简单记成一张表：

| 场景 | 推荐方案 |
|---|---|
| 普通非敏感配置 | ConfigMap |
| 普通敏感配置，更新不频繁 | Secret |
| 配置更新后需自动重启 | Secret/ConfigMap + Reloader |
| 高频轮换、强审计需求 | Vault / External Secrets |
| 单对象超过 1 MiB | 拆分对象、外部存储、初始化拉取 |
| 需要动态短期凭据 | 外部密钥系统 + CSI 或 SDK |

所以结论不是“ConfigMap/Secret 不够好”，而是它们适合解决 Kubernetes 内部最常见、最标准化的配置注入问题；一旦进入高频轮换、强安全审计、超大配置、跨系统统一治理，就应该转向专用方案。

---

## 参考资料

- Kubernetes Visual Handbook《ConfigMaps & Secrets》  
  https://k8s.info/docs/core/config-secrets
- Kubernetes 官方文档《Secrets》  
  https://kubernetes.io/docs/concepts/configuration/secret/
- Stakater Reloader 官方 GitHub  
  https://github.com/stakater/Reloader
