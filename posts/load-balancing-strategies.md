## 核心结论

负载均衡的定义很直接：它用一套分发规则，把进入系统的请求送到多台后端实例上。白话讲，它就是流量分配器。它的目标通常有三个：提高吞吐量，缩短平均响应时间，避免单点过载。

从工程上看，负载均衡策略不是“谁更高级”，而是“谁更匹配当前负载形态”。

1. 请求处理时间接近、实例能力接近时，`Round Robin` 足够，成本最低。
2. 实例能力不同，或机器规格不一致时，`Weighted Round Robin` 更合理。
3. 长连接、WebSocket、慢查询较多时，`Least Connections` 往往比简单轮询更稳。
4. 需要会话保持时，`IP Hash` 或其他粘性会话策略更合适。
5. 当负载波动大、实例状态变化快时，才需要自适应负载均衡，即根据实时指标动态调权。

一个初学者可以先记住这条主线：负载均衡策略的演进，本质是从“只看顺序分发”，走向“看容量差异”，再走向“看实时状态”。

| 负载均衡形态 | 代表产品 | 典型约束 |
|---|---|---|
| 硬件负载均衡 | F5、A10 | 价格高，扩容慢，可控性强 |
| 软件负载均衡 | NGINX、HAProxy | 需要自己运维，灵活度高 |
| 云负载均衡 | Google Cloud Load Balancing、AWS ALB/NLB | 易用但抽象层高，厂商特性明显 |

玩具例子可以用“两台收银机”理解。顾客是请求，收银机是后端实例，排队指挥员是负载均衡器。轮流分配就是 Round Robin；让处理快的收银机多接人，就是加权轮询；谁前面排队短就优先安排给谁，就是 Least Connections。

---

## 问题定义与边界

先定义问题。我们讨论的负载均衡，边界是“多个后端副本对外提供同一种服务”，前面有一个入口负责转发。它解决的是分流问题，不直接解决业务逻辑、数据库瓶颈或代码慢查询本身。

几个术语先说明。

负载均衡：把请求分到多台机器上。  
吞吐量：单位时间内系统能处理多少请求。  
延迟：一个请求从进入到返回结果花的时间。  
健康检查：周期性判断某个后端是不是还能正常服务。  
粘性会话：尽量让同一用户持续落到同一台机器。  
异构资源：不同机器能力不一样，比如 16 核和 8 核混用。

真正的难点不在“有没有多台机器”，而在“这些机器和请求都不一样”。

| 边界条件 | 会造成什么限制 | 可行策略 |
|---|---|---|
| 后端规格不同 | 平均分发会压垮慢机器 | Weighted Round Robin |
| 请求耗时差异大 | 简单轮询会让慢实例积压 | Least Connections |
| 需要会话保持 | 请求不能随意切换实例 | IP Hash、Cookie Sticky |
| 跨区域部署 | 最近区域和最空闲区域可能冲突 | 区域优先 + 容量溢出 |
| 长连接很多 | “正在处理中的连接数”比请求数更重要 | Least Connections、自适应 |
| 后端健康波动 | 继续分流会放大故障 | 健康检查、自动摘除 |

这里有一个常见误解：把请求平均分到每台机器，不等于系统最优。  
如果三台机器能力分别是 $16$ 核、$8$ 核、$8$ 核，而你仍然按 $1:1:1$ 分配，请求会优先把第一台以外的机器打满，结果整体吞吐下降，尾延迟升高。尾延迟就是最慢那部分请求的延迟，白话讲，是“最容易让用户感觉卡”的延迟。

因此，负载均衡的边界可以概括成一句话：它不是追求“绝对平均”，而是追求“在约束下尽量稳、尽量快”。

---

## 核心机制与推导

先看最基础的机制。

`Round Robin` 是轮询。第 1 个请求给 A，第 2 个给 B，第 3 个给 C，然后重复。它的优点是简单、稳定、实现便宜；缺点是它假设所有后端处理能力接近，而且请求耗时差不多。

`Weighted Round Robin` 是加权轮询。权重可以理解为“这台机器应该多吃多少流量”。如果三台机器权重是 $5:1:1$，那么在足够长的时间内，每 $7$ 个请求大约有 $5$ 个去第一台，另外两台各 $1$ 个。

`Least Connections` 是最少连接。它不看固定顺序，而是优先把新请求交给当前活跃连接数最少的实例。活跃连接数就是“这台机器现在手上还有多少没处理完的连接”。它特别适合连接持续时间差异大的场景。

`IP Hash` 是按客户端 IP 做哈希。哈希可以理解为“把一个输入稳定映射到一个结果”。这样同一用户通常会落到同一台后端，适合依赖本地会话状态的旧系统，但会带来流量倾斜问题。

从静态算法走向自适应，核心变化是：开始引入实时观测值。一个常见思路是同时观察响应时间、连接数、CPU、队列长度，再动态调整权重。

如果第 $i$ 个请求在某台机器上的响应时间是

$$
RT_i = S_{\text{end}, i} - S_{\text{start}, i}
$$

那么第 $j$ 台机器在一个观测窗口内的“负载压力”可以抽象写成

$$
U_j = \sum_{i=1}^{n} \frac{RT_i}{O_i}
$$

这里 $O_i$ 可以理解为该请求对应的处理能力或归一化容量。机器之间负载差异可以写成

$$
\Delta U = U_{\max} - U_{\min}
$$

自适应调度的目标，不是让每台机器请求数完全一样，而是尽量让 $\Delta U$ 变小，也就是让各实例的真实压力更接近。

机制演进可以概括成这条链路：

`Round Robin`  
$\rightarrow$ `Weighted Round Robin`  
$\rightarrow$ 采集连接数、响应时间、失败率  
$\rightarrow$ 动态调权或改路由  
$\rightarrow$ 自适应负载均衡

玩具例子：

有三台实例 A、B、C。A 处理一次请求平均要 100ms，B 和 C 都要 20ms。  
如果用 Round Robin，三台每秒都收到差不多的请求数，但 A 会先堆积。  
如果改成权重 $1:3:3$，或者直接用 Least Connections，新请求会更少地流向 A，系统整体平均延迟会下降。

真实工程例子：

某个推理服务前面挂了 6 台 API 实例，其中 2 台承接大量 WebSocket 长连接，另外 4 台主要处理短请求。如果还用纯 Round Robin，长连接实例会长期占满连接槽位，新来的短请求继续被送过去，结果排队严重。把策略切到 `leastconn` 后，新请求会更多地流向连接较少的实例，排队深度明显下降。这类场景下，连接数比“累计请求数”更有解释力。

---

## 代码实现

先给一个可运行的 Python 玩具实现，模拟三种策略：轮询、加权轮询、最少连接。这里不追求网络细节，只展示调度逻辑。

```python
from dataclasses import dataclass

@dataclass
class Server:
    name: str
    weight: int
    active_connections: int = 0

class RoundRobinBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.idx = 0

    def pick(self):
        s = self.servers[self.idx % len(self.servers)]
        self.idx += 1
        return s.name

class WeightedRoundRobinBalancer:
    def __init__(self, servers):
        self.sequence = []
        for s in servers:
            self.sequence.extend([s.name] * s.weight)
        self.idx = 0

    def pick(self):
        s = self.sequence[self.idx % len(self.sequence)]
        self.idx += 1
        return s

class LeastConnectionsBalancer:
    def __init__(self, servers):
        self.servers = servers

    def pick(self):
        s = min(self.servers, key=lambda x: x.active_connections)
        s.active_connections += 1
        return s.name

servers = [
    Server("srv-a", weight=5, active_connections=3),
    Server("srv-b", weight=1, active_connections=1),
    Server("srv-c", weight=1, active_connections=0),
]

rr = RoundRobinBalancer(servers)
assert [rr.pick() for _ in range(4)] == ["srv-a", "srv-b", "srv-c", "srv-a"]

wrr = WeightedRoundRobinBalancer(servers)
assert [wrr.pick() for _ in range(7)] == [
    "srv-a", "srv-a", "srv-a", "srv-a", "srv-a", "srv-b", "srv-c"
]

lc = LeastConnectionsBalancer(servers)
picked = lc.pick()
assert picked == "srv-c"
assert servers[2].active_connections == 1
```

如果你只想理解策略差异，这段代码已经足够。  
如果你要上线真实服务，通常不会自己手写负载均衡器，而是使用 NGINX、HAProxy 或云负载均衡。

下面是一个典型的 NGINX 配置：

```nginx
upstream api_pool {
    least_conn;
    server 10.0.1.5:8080 weight=5 max_fails=3 fail_timeout=30s;
    server 10.0.1.6:8080 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.1.7:8080 backup;
}

server {
    listen 80;
    location / {
        proxy_pass http://api_pool;
    }
}
```

这段配置表达了三件事：

1. `least_conn`：按当前活跃连接数选更空闲的实例。
2. `weight=5`：第一台机器能力更强，应接更多流量。
3. `backup`：前两台都不可用时，第三台作为备份接流。
4. `max_fails` 和 `fail_timeout`：连续失败达到阈值后，暂时把实例判为不可用。

如果是 HAProxy，对应思路通常是：

```haproxy
backend api_pool
    balance leastconn
    option httpchk GET /healthz
    server s1 10.0.1.5:8080 check weight 5
    server s2 10.0.1.6:8080 check weight 1
    server s3 10.0.1.7:8080 check backup
```

监控不能缺。没有监控，负载均衡只是“猜”。  
常见观察项包括：

| 指标 | 说明 | 用途 |
|---|---|---|
| `active connections` | 当前活跃连接数 | 判断是否适合 leastconn |
| `response time` | 后端响应时间 | 识别慢实例 |
| `5xx rate` | 错误率 | 识别异常节点 |
| `queue length` | 排队长度 | 判断是否已过载 |
| `health check status` | 健康检查状态 | 自动摘除故障节点 |

真实工程里，策略选择往往不是一次性决定，而是“配置 + 指标 + 回滚预案”一起上线。

---

## 工程权衡与常见坑

第一个坑是把“平均分配”误认为“公平分配”。  
对异构机器来说，平均请求数不公平；对不同耗时请求来说，平均请求数也不公平。公平应该接近“平均压力”。

第二个坑是只配算法，不配健康检查。  
没有健康检查，负载均衡器会继续把请求发给已经异常的实例，故障会被放大而不是被隔离。

第三个坑是只看 QPS，不看连接和排队。  
QPS 是每秒请求数，白话讲是“来单速度”。但 WebSocket、流式推理、长轮询这类场景里，活跃连接数和队列深度更重要。

第四个坑是粘性会话滥用。  
IP Hash 看上去能解决会话保持，但用户如果集中来自某个 NAT 出口，流量可能高度集中到少数实例，导致热点。

第五个坑是切换策略时一次性全量切。  
例如从 `roundrobin` 切到 `leastconn`，如果业务里混有大量长连接，最好先让一小部分流量试运行，确认 `scur`、`qcur`、响应时间和错误率趋势正常，再扩大比例。

| 常见坑 | 规避措施 |
|---|---|
| 静态权重长期不调整 | 定期按容量和指标重估权重 |
| 没有健康检查 | 配置主动或被动健康检查 |
| 只看平均延迟 | 同时看 P95/P99 尾延迟 |
| 长连接场景还用简单 RR | 改用 leastconn 或自适应策略 |
| 新实例刚上线就全量接流 | 使用 slow start 或灰度引流 |
| 跨区流量一刀切平均 | 先区域优先，再做容量溢出 |

还要注意一个事实：自适应并不天然更好。  
如果你的指标采集延迟大、控制回路太敏感，动态调权可能产生抖动。也就是今天觉得 A 忙就降权，下一秒又觉得 B 忙再降权，结果流量来回摆动。控制系统里这叫振荡，白话讲就是“调得太勤，反而不稳”。

因此，工程上的优先级通常是：

1. 先把健康检查和监控补齐。
2. 再用最简单可解释的算法。
3. 只有当静态策略确实不能满足目标时，才引入动态或自适应。

---

## 替代方案与适用边界

如果你自己运维 NGINX 或 HAProxy，优势是控制细、成本低、调优自由。缺点是要自己处理高可用、升级、监控和跨区域复杂度。

如果使用云负载均衡，优势是托管程度高，天然支持区域级或全球级分发。缺点是很多高级行为被平台抽象掉，排障时需要熟悉厂商的模型。

可以把常见方案这样区分：

| 方案 | 优势 | 适用边界 |
|---|---|---|
| NGINX | 配置直观，生态广 | 中小规模 HTTP 服务，预算有限 |
| HAProxy | 连接调度与观测能力强 | 长连接、代理层控制要求高 |
| 云负载均衡 | 跨区能力强，托管运维 | 全球业务、多区域容灾 |
| 多层负载均衡 | 边缘与内部职责分离 | 大规模平台型系统 |

一个典型替代方案是多层架构：

1. 最外层用 DNS、Anycast 或云全局负载均衡做地域级入口。
2. 区域内再用 NGINX 或 HAProxy 做实例级转发。
3. 服务内部再结合服务发现和健康检查做细粒度调度。

这种分层的好处是，把“跨区域延迟最优”和“单区域内实例负载最优”拆开处理。前者看地理位置和容量，后者看连接数、响应时间和错误率。

真实工程里，多区域服务常见做法不是全局绝对平均，而是“近端优先，满载溢出”。例如云平台提供的 `WATERFALL_BY_REGION` 更偏向最近区域优先，而 `SPRAY_TO_REGION` 更偏向区域内更均匀分布。再配合 `auto-capacity-drain`，当某区域后端不健康时，可以自动把有效容量降到零，让流量撤走。这类能力适合全球应用，但前提是你接受平台抽象和对应的运维模型。

所以，适用边界可以压缩成一句话：

小而稳的系统，先用简单静态策略。  
连接时长差异明显的系统，优先考虑 leastconn。  
异构资源明显的系统，必须引入权重。  
跨区域、多活和容量溢出要求高的系统，更适合云负载均衡或多层架构。  
只有当静态规则解释不了性能问题时，再上自适应。

---

## 参考资料

- NGINX: Using nginx as HTTP load balancer  
  https://nginx.org/en/docs/http/load_balancing.html
- NGINX Documentation: HTTP Load Balancing  
  https://docs.nginx.com/nginx/admin-guide/load-balancer/http-load-balancer/
- HAProxy: What is Load Balancing and How Does It Work?  
  https://www.haproxy.com/blog/what-is-load-balancing
- GeeksforGeeks: Software vs. Hardware Load Balancers  
  https://www.geeksforgeeks.org/software-vs-hardware-load-balancers/
- Google Cloud Documentation: Advanced load balancing optimizations / Service LB Policy  
  https://docs.cloud.google.com/load-balancing/docs/service-lb-policy
- Springer: Dynamic scheduling strategies for cloud-based load balancing in parallel and distributed systems  
  https://link.springer.com/article/10.1186/s13677-025-00757-6
- ServerDevWorker: leastconn vs roundrobin 调优经验  
  https://serverdevworker.com/4202572a2/
