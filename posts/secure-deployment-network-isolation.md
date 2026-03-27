## 核心结论

安全部署不是“把模型放到服务器上再加一个密码”，而是把访问路径拆成四层分别收紧：网络隔离、访问控制、加密传输、监控审计。可以用一个简化公式表示：

$$
S = N + A + E + M
$$

其中，$S$ 是整体安全态；$N$ 是网络隔离，意思是先把机器和流量关进边界明确的网络里；$A$ 是访问控制，意思是只有被授权的身份才能调用；$E$ 是加密传输，意思是链路上的内容不能被旁路窃听；$M$ 是监控审计，意思是异常行为必须被记录、发现和追查。

对大模型部署来说，真正的风险通常不是“模型会不会跑起来”，而是以下三类问题同时出现：

| 风险 | 白话解释 | 常见后果 |
| --- | --- | --- |
| 模型窃取 | 别人持续高频调用接口，把模型能力“抄走” | 成本暴涨、核心能力泄露 |
| 数据泄露 | 训练数据、提示词、输出结果或日志被越权访问 | 合规风险、业务信息外泄 |
| 恶意攻击 | 攻击者用异常流量、恶意请求或越权凭证打接口 | 服务中断、资源耗尽、横向入侵 |

因此，部署安全的最小闭环不是单点措施，而是“私有网络承载模型 + 网关统一入口 + IAM/RBAC 鉴权 + HTTPS/TLS 加密 + 全量日志与告警联动”。如果只做其中一层，系统仍然会留下明显缺口。

---

## 问题定义与边界

“网络隔离与访问控制”要解决的问题，不是抽象的“更安全”，而是明确回答三个边界问题。

第一，谁能进来。也就是入站访问边界。模型服务是否暴露公网、是否只允许 API 网关访问、是否只接受来自特定子网或私有端点的流量，这些都属于入站控制。

第二，谁能出去。也就是出站访问边界。推理容器、训练节点、运维主机能否直接访问公网、能否下载任意依赖、能否把数据发往未知地址，这些属于出站控制。很多团队只看入站，不管出站，结果模型权重和日志从内部被带走。

第三，谁能调用什么。也就是身份和权限边界。一个调用方拿到的是不是“只允许推理、不允许列举模型、不允许读日志、不允许下载权重”的最小权限，而不是一把能访问整个账户资源的万能钥匙。

可以把部署对象分成四个层次：

| 层次 | 主要对象 | 目标 |
| --- | --- | --- |
| 边界入口层 | DNS、CDN、WAF、API 网关、负载均衡 | 吸收公网流量，挡住明显恶意请求 |
| 应用接入层 | 鉴权服务、签名校验、限流、Schema 校验 | 只让合法请求进入业务逻辑 |
| 模型执行层 | 推理容器、GPU 节点、模型仓库 | 放在私有子网，不直接暴露公网 |
| 审计响应层 | 日志、告警、IDS、审计平台 | 发现异常并触发封禁或回滚 |

这里的“VPC”可以理解为专属虚拟网络，相当于云上独立园区；“子网”可以理解为园区内不同功能区域；“安全组”可以理解为实例级门禁规则；“防火墙”可以理解为统一边界盘查系统；“IAM”是身份与访问管理，白话说就是“给谁什么权限”；“RBAC”是基于角色分配权限，白话说就是“按岗位给权限，不按个人随便发权限”。

玩具例子可以帮助建立直觉。假设你只有一个小型推理服务，内部只有两台机器：

- `gateway`：负责接收外部 HTTPS 请求。
- `model-server`：负责真正执行模型推理。

正确做法不是让两台机器都暴露公网，而是让 `gateway` 放在公有子网，`model-server` 放在私有子网，且私有子网只允许来自 `gateway` 的 443 端口访问。这样即使有人知道模型服务地址，也无法直接从公网打到推理进程。

---

## 核心机制与推导

安全体系为什么必须分层，而不是“一个防火墙全搞定”？原因在于攻击链是分阶段发生的。攻击者通常先探测入口，再绕过身份校验，再窃听或伪造流量，最后才是横向移动或批量抽取。单层防御只能挡住其中一个阶段。

从机制上看，可以把安全链路写成下面的顺序：

$$
\text{Request} \rightarrow \text{Network Boundary} \rightarrow \text{Identity Check} \rightarrow \text{Encrypted Transport} \rightarrow \text{Inference Service} \rightarrow \text{Audit Trail}
$$

如果某一层缺失，攻击面就会放大：

| 缺失层 | 直接后果 | 典型攻击方式 |
| --- | --- | --- |
| 网络隔离缺失 | 模型服务直接暴露公网 | 端口扫描、未授权访问、DDoS |
| 访问控制缺失 | 任何拿到地址的人都能调用 | 越权调用、模型抽取 |
| 加密缺失 | 传输内容可被监听或篡改 | 中间人攻击、密钥泄露 |
| 监控缺失 | 攻击发生后无法及时发现 | 持续滥用、事后难追踪 |

核心推导可以概括为三步。

第一步，先缩小可达范围。模型节点放入私有子网后，公网攻击者不能直接建立 TCP 连接，攻击面从“所有互联网来源”缩到“少数受控入口”。这一步并不依赖调用方是否诚实，它先在网络层减小暴露面。

第二步，再缩小合法身份范围。即使攻击者能打到 API 网关，仍然需要通过 IAM、RBAC、API 密钥或请求签名验证。请求签名的白话意思是：客户端要用受保护的密钥对请求做一次数学签名，服务端验证签名后才接受请求。这样即使报文被别人看到，也不能轻易伪造同样的合法调用。

第三步，给每次调用留下可审计痕迹。日志审计的价值不在“记录很多”，而在于能把“谁、何时、从哪、调用了什么、是否异常”串成完整证据链。没有这一层，前面三层即使配置了，也难以及时发现凭证被盗用或策略被误改。

一个真实工程例子是典型的云上推理部署。前端负载均衡和 API 网关位于公有子网，对外只开放 443；WAF 在七层过滤明显恶意模式，比如异常 User-Agent、超长请求体、违反 Schema 的 JSON；网关完成 IAM 签名校验和限流；真正的推理容器位于私有子网，只允许来自网关安全组的访问；出站流量统一经过 NAT 或云防火墙，只允许访问对象存储、日志服务、镜像仓库等白名单目标；所有请求、鉴权失败、限流命中、模型调用耗时和异常响应都进入审计日志，再由告警系统触发自动化封禁。这种架构的关键不是某个单独组件，而是每层只做自己最擅长的那部分控制。

---

## 代码实现

下面用一个最小可运行的 Python 示例，演示“请求签名 + 时间戳有效期 + 角色校验”的基本思路。它不是生产级鉴权系统，但足够解释机制。

```python
import hmac
import hashlib
import time

SECRETS = {
    "client_a": b"super-secret-key-a",
    "client_b": b"super-secret-key-b",
}

ROLES = {
    "client_a": {"invoke:model"},
    "client_b": set(),
}

def canonical_string(method: str, path: str, body: str, ts: int) -> str:
    return f"{method}\n{path}\n{body}\n{ts}"

def sign(secret: bytes, message: str) -> str:
    return hmac.new(secret, message.encode("utf-8"), hashlib.sha256).hexdigest()

def verify_request(client_id: str, method: str, path: str, body: str, ts: int, signature: str) -> bool:
    if client_id not in SECRETS:
        return False
    # 只接受 5 分钟内的请求，防止重放
    if abs(int(time.time()) - ts) > 300:
        return False
    expected = sign(SECRETS[client_id], canonical_string(method, path, body, ts))
    return hmac.compare_digest(expected, signature)

def authorize(client_id: str, action: str) -> bool:
    return action in ROLES.get(client_id, set())

def handle_inference(client_id: str, prompt: str, ts: int, signature: str) -> str:
    method = "POST"
    path = "/v1/infer"
    body = prompt

    if not verify_request(client_id, method, path, body, ts, signature):
        raise PermissionError("invalid signature or expired request")

    if not authorize(client_id, "invoke:model"):
        raise PermissionError("role not allowed")

    return f"ok:{prompt[:10]}"

now = int(time.time())
msg = canonical_string("POST", "/v1/infer", "hello model", now)
sig = sign(SECRETS["client_a"], msg)

assert verify_request("client_a", "POST", "/v1/infer", "hello model", now, sig) is True
assert authorize("client_a", "invoke:model") is True
assert handle_inference("client_a", "hello model", now, sig).startswith("ok:")

bad_sig = sign(SECRETS["client_b"], msg)
assert verify_request("client_a", "POST", "/v1/infer", "hello model", now, bad_sig) is False
assert authorize("client_b", "invoke:model") is False
```

这个例子展示了三个关键点：

1. 只有持有正确密钥的客户端才能生成合法签名。
2. 时间戳有效期限制了重放攻击，也就是旧请求被截获后再次发送。
3. 即使签名合法，也还要过角色权限校验，避免“能登录就能做所有事”。

在网关层，除了签名校验，还应做限流和请求体约束。下面是常见的 Ingress 限流配置片段：

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/limit-rps: "20"
    nginx.ingress.kubernetes.io/limit-burst-multiplier: "3"
    nginx.ingress.kubernetes.io/proxy-body-size: "2m"
```

它的含义可以直白地理解为：单个调用方每秒最多 20 个请求，短时突发最多放大到 60 个，同时请求体大小限制为 2MB。这样做不是为了“更严格”本身，而是为了防止两类问题：

| 规则 | 解决的问题 |
| --- | --- |
| `limit-rps` | 避免被持续高频抽取模型能力 |
| `burst` | 允许正常短时抖动，不至于误伤 |
| `proxy-body-size` | 避免超大输入拖垮链路或绕过解析器 |

如果部署在 Kubernetes 或云原生网关上，常见落地顺序是：

1. 公网流量先进入 WAF 或 CDN。
2. 再进入 API 网关或 Ingress，做 TLS 终止、Schema 校验、签名验证、限流。
3. 通过内部服务发现访问私有子网中的推理服务。
4. 推理服务把访问日志、耗时指标、异常码发送到统一监控系统。
5. 告警系统根据阈值或规则自动禁用 API 密钥、更新黑名单或拉高防护等级。

---

## 工程权衡与常见坑

安全部署不是“组件越多越好”，而是“风险和维护成本是否匹配”。对初级团队来说，最容易踩的坑不是不会配，而是配错优先级。

最常见的误区如下：

| 常见坑 | 为什么危险 | 更合理的做法 |
| --- | --- | --- |
| 推理服务直接暴露公网 | 攻击面过大，绕过网关控制 | 模型只放私有子网，对外只暴露网关 |
| IAM 使用 `*` 通配权限 | 一旦凭证泄露，影响范围极大 | 每个服务单独角色，只授予必要动作 |
| 只看入站不管出站 | 内部节点可把数据发往未知地址 | 出站统一走 NAT/Firewall 和白名单 |
| API 密钥长期不轮换 | 密钥泄露后可长期滥用 | 设置过期时间、轮换策略和吊销机制 |
| 开了日志但没人看 | 审计数据存在但无法响应 | 告警规则和自动处置联动 |
| TLS 只在边界终止 | 内网仍可能明文传输 | 敏感链路继续使用端到端 TLS |

这里有一个经常被忽略的工程权衡：隔离越严格，联调和排障越复杂。比如私有子网不出公网是对的，但如果镜像拉取、漏洞库同步、系统补丁更新都被一刀切阻断，运维会转而手动开白名单，最终形成“临时规则永久存在”。因此，安全设计应从一开始就明确“必要的合法出站目标”，而不是事后靠人工补洞。

再看一个真实工程例子。某金融团队把模型推理服务放进了私有子网，这一步做对了；但他们关闭了持续监控告警，且共用了一组权限过大的运行角色。结果某个自动化任务的凭证泄露后，攻击者没有直接打公网接口，而是利用已有权限在夜间从对象存储读取模型相关文件并发起批量导出。由于异常检测没有及时触发，团队直到后续审计才发现数据被大量访问。这个例子说明：网络隔离只能减少暴露面，不能替代身份收敛和审计响应。

---

## 替代方案与适用边界

不同云平台会给出不同产品名字，但底层思路基本一致，都是把“入口、身份、链路、审计”分层控制。差异主要在自动化程度和可定制程度。

| 方案 | 适用场景 | 优势 | 代价 |
| --- | --- | --- | --- |
| 托管型网络隔离 | 团队小、希望快速上线 | 默认规则较完整，维护成本低 | 可定制空间较少 |
| 自定义 VPC/VNet 架构 | 有复杂拓扑、需接入本地机房或多云 | 控制力强，适合高合规环境 | 设计和运维复杂度高 |
| Service Perimeter / VPC Controls | 多项目、多团队、多数据域 | 适合跨服务统一边界控制 | 学习成本高，策略调试复杂 |

如果是零基础到初级工程师，可以用一个简单判断来选型：

1. 如果团队没有专职安全和平台工程能力，优先选托管型隔离能力，把默认最佳实践先吃满。
2. 如果存在本地数据中心、专线、跨云访问、严格审计要求，再考虑自定义网络拓扑。
3. 如果业务需要强制限制“某些数据绝不能离开某个安全域”，就要引入更强的服务边界控制，而不只是普通安全组。

还有一个适用边界需要讲清楚：网络隔离与访问控制主要保护“谁能访问、怎么访问、访问是否可信”，但它不能直接解决“模型输出是否安全”“提示注入是否成功”“训练数据本身是否合规”这类应用层问题。也就是说，它是部署安全的底座，不是全部安全问题的终点。

---

## 参考资料

- AWS Security Blog: Build secure network architectures for generative AI applications using AWS services  
  https://aws.amazon.com/blogs/security/build-secure-network-architectures-for-generative-ai-applications-using-aws-services/
- Microsoft Learn: Network isolation planning for Azure Machine Learning  
  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-network-isolation-planning
- Dino Cajic: Secure Model Deployment in MLOps  
  https://www.dinocajic.com/secure-model-deploument-mlops/
- Braincuber: AWS security best practices for AI applications  
  https://www.braincuber.com/blog/6-aws-security-best-practices-ai-applications
- Google Cloud: Security best practices for GenAI with VPC controls  
  https://cloud.google.com/docs/security/security-best-practices-genai/vpc-controls
