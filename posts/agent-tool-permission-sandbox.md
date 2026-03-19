## 核心结论

Agent 的工具调用安全，核心不是先给一个“大身份”，再靠代码里四处判断能不能做事；核心是把每次可做的事拆成一个个不可伪造的能力令牌。能力令牌可以理解为“资源钥匙卡”：只有拿到对应卡片，Agent 才能访问那个文件、那个网络出口、那个执行器。

最常见的能力结构可以写成：

$$
Capability = (object\_id,\ access\_rights,\ validity\_period,\ signature)
$$

各字段含义如下：

| 字段 | 含义 | 白话解释 |
| --- | --- | --- |
| `object_id` | 资源标识 | 这把钥匙到底开哪一扇门 |
| `access_rights` | 访问动作集合 | 能读、能写、能执行，还是只能其中一种 |
| `validity_period` | 有效期 | 这把钥匙什么时候失效 |
| `signature` | 签名 | 平台签过字，Agent 自己伪造不了 |

这套模型与最小权限原则天然一致。最小权限原则的意思是“只给完成当前任务所需的最低权限”。如果 Agent 只是把结果写到 `/tmp/results`，那就只发 `fs:write:/tmp/results`，而不是直接给整个文件系统写权限，更不能给 shell 的执行权限。

玩具例子很直接：把 `tool:execute:shell`、`fs:read:/data`、`net:connect:api.example.com` 看成三张不同钥匙卡。Agent 要读数据时，只插 `fs:read:/data`；要发请求时，只插 `net:connect:api.example.com`；没有 `tool:execute:shell` 这张卡，就算它运行在同一台机器上，也不能启动 shell。

真实工程里，能力令牌必须和沙箱一起工作。沙箱是“隔离房间”，意思是即使 Agent 获得某项能力，它的活动范围仍被限制在容器、用户态内核或微虚拟机里。Docker、gVisor、Firecracker分别代表三个常见隔离层级：Docker 轻、快，但共享宿主机内核；gVisor 用用户态内核拦截系统调用，隔离更强；Firecracker 用微虚拟机直接提供硬隔离，安全边界最清晰。Agent 场景里，常见额外延迟税大致在 5-50ms 这个量级，代价通常是可接受的，因为它换来的是越权风险显著下降。

---

## 问题定义与边界

问题定义可以压缩成一句话：Agent 不只是“生成文本”，它会调用外部工具，因此它已经接近一个会自动操作系统和网络的程序执行体。权限边界如果还停留在“这个用户是不是管理员”，通常就太粗了。

传统身份权限模型的思路是“你是谁，所以你能做什么”。这在人类用户系统里常见，例如管理员、开发者、访客。问题在于 Agent 的行为粒度更细。一次任务里，它可能只需要：

| 资源类型 | 对应令牌示例 | 最小权限动作 |
| --- | --- | --- |
| 文件系统 | `fs:read:/workspace/docs` | 只读某目录 |
| 文件系统 | `fs:write:/tmp/results` | 只写临时结果目录 |
| 工具执行 | `tool:execute:python` | 只运行 Python，不给 shell |
| 工具执行 | `tool:execute:shell` | 只在需要时启动 shell |
| 网络 | `net:connect:api.example.com:443` | 只访问指定主机和端口 |
| 凭证 | `secret:use:github-token` | 只允许代调用，不允许明文读出 |

这里的边界有三个。

第一，控制对象是“调用能力”，不是“用户身份”。“读取 `/etc/passwd`”和“执行 shell”必须被当成两把不同钥匙。即使 Agent 在同一进程、同一容器、同一会话中运行，也不能因为已经有一种能力，就自动推导出另一种能力。

第二，控制目标是“最小授权且可审计”。可审计的意思是平台能够回答：是谁在什么时间拿到了哪个令牌，用它访问了什么资源，是否做了转授。没有审计链路，权限系统只是摆设。

第三，本文边界聚焦在三类高风险工具：文件系统、网络请求、代码执行。数据库、浏览器自动化、消息队列本质上也适用同一模型，但为了讲清机制，不展开到更多资源类型。

一个新手容易混淆的点是：“都已经在沙箱里了，为什么还要能力令牌？”原因很简单。沙箱解决的是“跑在哪里”，能力令牌解决的是“能做什么”。没有令牌，沙箱里的 Agent 可能仍能访问沙箱内部不该碰的目录；没有沙箱，令牌一旦校验实现有漏洞，攻击面会直接落到宿主机。两者是正交关系，不是替代关系。

---

## 核心机制与推导

能力模型的核心判定链可以写成：

$$
HasCapability \land VerifySignature \land NotExpired \land MatchScope \Rightarrow Allow
$$

也可以写得更细一点：

$$
Allow(op, obj) =
Present(cap) \land
SignedBy(platform, cap) \land
Now \in validity\_period \land
obj = object\_id \land
op \in access\_rights
$$

含义是：只有当 Agent 持有令牌、令牌签名正确、令牌未过期、目标对象匹配、请求动作属于授权范围时，工具调用才被放行。任何一个条件不满足，都应该拒绝。

这条链路的价值在于，它把“权限判断”从模糊的上下文里拉成了显式数据结构。传统做法里，经常是工具代码里写一堆 `if role == admin`、`if env == prod`、`if task_type == xxx`。这样做的问题是权限分散在多个模块里，后续很难证明“到底哪里还能执行 shell”。能力模型把授权收敛为“先发令牌，再验令牌”。

一个最小玩具例子：

1. Agent 需要读取 `/data/report.csv`
2. 平台签发 `fs:read:/data/report.csv`，有效期 60 秒
3. Agent 读取成功
4. 任务下一步需要把结果写入 `/tmp/results.json`
5. 平台额外签发 `fs:write:/tmp/results.json`，有效期 30 秒
6. 写入完成后，平台主动撤销或等待令牌过期

这个过程体现了运行时授予。运行时授予的意思是权限不是在进程启动时一次性塞满，而是在任务推进过程中按需下发。它带来的直接结果是：长期持有的高权限变少，令牌泄露后的爆炸半径也更小。

真实工程例子可以看一个“代码审查 Agent”。它需要：

- 读取仓库只读副本
- 连接企业内部的 diff API
- 调用 Python 执行静态分析脚本
- 把结果写到临时目录
- 最后通过代理调用评论接口

如果直接给一个容器 root 权限，再挂上全部仓库、全部网络、全部凭证，这个 Agent 一旦被提示注入或工具链污染，就可能读出无关目录、横向访问内网，或者把凭证带出。更稳妥的做法是：

1. 只给 `fs:read:/repo`
2. 只给 `tool:execute:python`
3. 只给 `net:connect:diff.internal:443`
4. 评论阶段才给 `secret:use:review-bot-token`
5. 写结果只给 `fs:write:/tmp/review-output`

这时即使模型想执行 `bash -c "curl ..."`，如果没有 `tool:execute:shell` 或没有对应网络能力，也过不了校验链。

能力验证流程可以抽象为：

`令牌获取 -> 签名校验 -> 有效期校验 -> 作用域匹配 -> 工具调用`

这里还有两个工程上必须补齐的点。

第一是委托。委托的意思是某个受信组件可以把一部分能力转交给另一个组件。例如调度器把“只读仓库”能力交给分析器，但不能自动把“写回评论”能力一并下发。委托必须是降权的，不能凭空扩权。

第二是撤销。撤销的意思是即使令牌还没自然过期，平台也能提前让它失效。最简单的实现是短 TTL，也就是很短的有效期；更稳妥的实现是加入吊销列表或版本号检查。否则，令牌一旦通过日志、缓存、错误上报泄露，风险窗口会变长。

---

## 代码实现

下面给一个可以运行的 Python 玩具实现，用 HMAC 模拟平台签名。HMAC 可以理解为“用共享密钥做签名校验的方法”，适合演示令牌不可伪造这一点。生产环境更常用非对称签名，这样验证方不需要持有签名私钥。

```python
import time
import json
import hmac
import hashlib
from copy import deepcopy

SECRET = b"platform-signing-key"

def canonical_payload(cap):
    data = {
        "object_id": cap["object_id"],
        "access_rights": sorted(cap["access_rights"]),
        "valid_from": int(cap["valid_from"]),
        "valid_to": int(cap["valid_to"]),
    }
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()

def sign_capability(cap):
    msg = canonical_payload(cap)
    return hmac.new(SECRET, msg, hashlib.sha256).hexdigest()

def issue_capability(object_id, rights, ttl_seconds):
    now = int(time.time())
    cap = {
        "object_id": object_id,
        "access_rights": list(rights),
        "valid_from": now,
        "valid_to": now + ttl_seconds,
    }
    cap["signature"] = sign_capability(cap)
    return cap

def verify_capability(cap, object_id, action, now=None):
    now = int(time.time()) if now is None else int(now)
    expected = sign_capability(cap)
    if not hmac.compare_digest(expected, cap.get("signature", "")):
        return False, "bad_signature"
    if now < cap["valid_from"] or now > cap["valid_to"]:
        return False, "expired"
    if cap["object_id"] != object_id:
        return False, "object_mismatch"
    if action not in cap["access_rights"]:
        return False, "action_denied"
    return True, "ok"

def request_tool(tool_name, cap):
    object_id = f"tool:{tool_name}"
    ok, reason = verify_capability(cap, object_id, "execute")
    if not ok:
        return f"reject:{reason}"
    return f"spawn:{tool_name}"

shell_cap = issue_capability("tool:shell", ["execute"], ttl_seconds=30)
py_cap = issue_capability("tool:python", ["execute"], ttl_seconds=30)

assert request_tool("python", py_cap) == "spawn:python"
assert request_tool("shell", py_cap).startswith("reject:")

tampered = deepcopy(py_cap)
tampered["access_rights"].append("admin")
assert verify_capability(tampered, "tool:python", "execute")[0] is False

expired = issue_capability("fs:/tmp/results.json", ["write"], ttl_seconds=1)
future = expired["valid_to"] + 10
assert verify_capability(expired, "fs:/tmp/results.json", "write", now=future) == (False, "expired")
```

上面这段代码对应的伪接口逻辑就是：

```text
request_tool("shell")
if hasCapability("tool:shell", "execute"):
    spawnShell()
else:
    reject()
```

如果把结构写成更接近平台 API 的 JSON，可以是这样：

```json
{
  "object_id": "fs:/tmp/results.json",
  "access_rights": ["write"],
  "validity_period": {
    "valid_from": 1760000000,
    "valid_to": 1760000030
  },
  "delegation_depth": 0,
  "issuer": "tool-gateway",
  "signature": "base64-or-hex-signature"
}
```

这里的 `delegation_depth` 表示还能不能继续转授，`issuer` 表示由哪个受信组件签发。生产实现里通常还会加入 `nonce`、`audience`、`request_id` 等字段，用于防重放和日志关联。

沙箱初始化接口可以抽象成两层。

第一层是创建执行环境：

- `sandbox = create_sandbox(kind="firecracker" | "gvisor" | "docker")`

第二层是挂载能力：

- `sandbox.attach_capability(cap)`
- `sandbox.exec(tool="python", args=[...])`

关键点不在 API 名字，而在责任边界：沙箱负责隔离进程、文件视图、网络命名空间；能力网关负责判断这次调用有没有资格发生。不要把“是否允许访问”塞进沙箱镜像脚本里临时判断，那样会把授权逻辑散落回各处。

---

## 工程权衡与常见坑

先看三种常见沙箱方案的对比。

| 方案 | 隔离机制 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- | --- |
| Docker | namespace + cgroup + seccomp，共享宿主机内核 | 启动快，生态成熟，资源开销低 | 共享内核，若配置弱或内核有漏洞，逃逸风险更高 | 低风险任务、内部受控环境 |
| gVisor | 用户态内核拦截 syscall | 比普通容器更强的系统调用隔离 | 高频 I/O 性能下降明显，调试复杂 | 多租户 Agent、需要更强隔离 |
| Firecracker | 微虚拟机，独立内核边界 | 隔离最强，边界清晰，适合不可信代码 | 启动和管理成本更高，镜像与网络编排更复杂 | 执行不可信代码、外部插件市场 |

一个常见误区是“用了 Docker 就已经安全”。这不成立。Docker 本质上仍共享宿主机内核，如果 seccomp、capabilities、挂载点、用户映射没收紧，攻击面依旧大。对 Agent 来说尤其危险，因为它会自动试错、自动组合工具链，探索性比普通服务更强。能力令牌在这里的作用不是替代容器，而是把容器里默认可做的事继续缩小。

第二个常见坑是忽视 gVisor 的 I/O 成本。gVisor 的核心手段是截获和模拟系统调用。系统调用可以理解为“程序向操作系统申请服务的入口”，例如 `open()`、`read()`、`write()`。如果你的 Agent 工作负载是频繁小文件读写、不断 `open/close`，gVisor 可能比原生容器慢 2 到 10 倍。这时优化方向不是粗暴换回“全开放 Docker”，而是先问一句：这些高频 I/O 是否真的需要动态发生？

一个典型改法是，把大批量文件预检查前置。比如索引阶段由受信服务先扫描并打包允许访问的文件列表，执行阶段只授予一个只读 bundle 能力，而不是让 Agent 自己在沙箱里到处 `open()` 探测。

第三个坑是忽略能力传播。传播的意思是 Agent 拿到一个能力后，又把它塞进子任务、脚本参数、缓存、日志、错误信息里。这样就算主流程设计得很细，令牌仍可能外泄。工程上至少要做到：

- 令牌短期有效
- 默认不可转授
- 审计派发链路
- 日志脱敏
- 对撤销结果做快速生效

第四个坑是把“凭证注入”做成明文暴露。正确做法通常是平台代理访问外部服务，Agent 只持有“可请求平台代理代调用”的能力，而不是直接拿到完整 API Key。否则模型输出、调试日志、异常回溯都可能把秘密带出。

真实工程里，如果选择 Firecracker，不应只看“冷启动是不是慢”。关键在于能否预热。预热的意思是预先准备好可复用的微虚拟机或快照，等任务来了直接恢复。经验上，冷启动大约可落在 10-50ms，snapshot/restore 额外延迟可压到 5-25ms。这意味着即使要求较低延迟，微虚拟机也不一定不可用，前提是你愿意维护镜像池、快照池和网络代理。

---

## 替代方案与适用边界

能力模型不是唯一方案，也不应该被说成万能方案。它最适合解决的是“细粒度、可组合、运行时变化”的授权问题。对 Agent 工具调用，这正好是主矛盾。

替代方案之一是传统身份或角色权限，也就是 RBAC/ABAC 一类模型。RBAC 可以理解为“按角色给权限”，ABAC 可以理解为“按属性做策略判断”。它们在平台入口层仍然重要，例如先判断这个租户有没有资格创建代码执行任务，这一步完全可以靠角色与组织策略处理。然后进入具体任务执行时，再下沉到能力令牌控制单次工具调用。两者不是对立关系，而是“平台身份认证 + 执行期能力授权”的分层结构。

第二种替代思路是“只做沙箱，不做能力控制”。它的优点是实现简单，缺点是粒度太粗。比如某任务其实只需要读数据，你却因为统一策略直接启动一个完整 Firecracker，并挂上过大的文件视图和网络能力。这会让成本、复杂度和风险一起上升。对于只读任务，更合理的方式可能是只发一个 `fs:read:` 令牌，甚至放在只读容器里执行，根本不必启完整微虚拟机。

第三种替代思路是“应用层白名单”。例如只允许固定几个 API、固定几个命令。这对简单场景有效，但一旦任务路径变多，就会退化成大量散落的条件判断，难以审计，也难以支持运行时临时授权。

如何选型，可以按风险和延迟要求分层：

| 场景 | 推荐组合 | 原因 |
| --- | --- | --- |
| 只读检索、低风险内网任务 | Docker + 只读能力令牌 | 启动快，足够轻 |
| 需要执行代码、但 I/O 中等 | gVisor + 能力网关 | 隔离强于普通容器 |
| 执行不可信代码、租户隔离强 | Firecracker + 代理网络 + 短 TTL 令牌 | 安全边界最清晰 |
| 极低延迟要求 | 预热 Firecracker 或轻量容器 + 更严格能力范围 | 用预热抵消启动税 |

如果把性能税单独拿出来看，5-50ms 的额外延迟并不总是瓶颈。多数 Agent 任务真正的大头在模型推理、网络往返、仓库扫描和外部 API 响应。只有在高并发、短任务、强实时场景下，这几十毫秒才会成为主导因素。此时可用的折中手段是：

- 微虚拟机预热
- snapshot/restore
- 将只读任务下沉到轻量容器
- 把高吞吐预处理放到受信服务
- 将高风险能力改为平台代理代执行

最终的适用边界很明确：如果你的 Agent 永远不出网、不执行代码、不读写敏感文件，能力模型的收益会下降；但只要它开始调用真实工具，能力令牌加沙箱几乎就是默认应选项，而不是锦上添花。

---

## 参考资料

- Capability-based security 定义与基本思想 — https://en.wikipedia.org/wiki/Capability-based_security
- Capability 结构、最小权限原则的工程化说明 — https://codelucky.com/capability-based-security/
- Agent 系统安全技巧与能力令牌直观解释 — https://www.linkedin.com/pulse/security-techniques-ai-agent-systems-building-safe-scalable-r-scdbf
- Agentic RAG、shell sandboxing、工具隔离工程实践 — https://www.codeant.ai/blogs/agentic-rag-shell-sandboxing
- 微虚拟机预热、snapshot 与延迟测量讨论 — https://michaellivs.com/
