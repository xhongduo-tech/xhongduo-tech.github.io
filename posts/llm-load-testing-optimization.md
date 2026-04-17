## 核心结论

LLM 应用的压测与调优，本质上是在回答两个问题：系统最多能扛多少请求，以及在接近上限时哪里先坏。压测是“人为制造负载并观测系统反应”，调优是“根据观测结果改系统，再重复测量”。

对初级工程师，最先要记住的不是工具命令，而是三组核心指标：

| 指标 | 白话解释 | 用途 |
| --- | --- | --- |
| QPS / RPS | 每秒能处理多少请求 | 看吞吐上限 |
| p95 / p99 延迟 | 100 个请求里最慢那 5 个或 1 个大概有多慢 | 看尾部抖动和拥堵 |
| 错误率 | 有多少请求失败、超时、被限流 | 看稳定性 |

只看平均延迟通常会误判。平均值会把“少量特别慢的请求”冲淡，但用户往往正是被这些慢请求伤害。对 LLM 应用尤其如此，因为生成式接口天然有长尾：有些请求 prompt 更长，有些请求输出 token 更多，有些请求刚好碰到模型队列排队或数据库锁等待。

一个新手能直接上手的最小结论是：先用 `k6` 或 `Locust` 固定低并发做基线测试，再逐步增加虚拟用户或到达率；一旦发现 p95/p99 突然抬升、错误率同步上涨，就说明系统从“平滑区”进入“拥堵区”。接下来不是盲目扩容，而是拆解链路，看瓶颈在模型推理、网络连接、数据库查询还是缓存未命中。

玩具例子：一个 `/api/chat` 接口在 5 个并发用户下，平均 320ms，p95 450ms，错误率 0%。增加到 20 个并发后，平均变成 500ms，但 p95 直接跳到 1800ms，错误率升到 6%。这说明问题不是“大家都慢了一点”，而是“有一批请求开始严重排队甚至失败”，通常意味着某个共享资源已经满了。

---

## 问题定义与边界

这篇文章讨论的是在线 LLM 应用的压测与调优，边界限定为：用户通过 HTTP 接口发起请求，系统执行鉴权、检索、提示词拼装、模型推理、后处理并返回结果。也就是典型的聊天、问答、代码解释、客服助手这一类交互式场景。

不在本文主定义中的内容有三类：

| 维度 | 包含 | 不包含 |
| --- | --- | --- |
| 测试对象 | `/api/chat`、`/api/completions` 等在线接口 | 离线批量摘要、夜间批处理任务 |
| 性能目标 | 延迟、吞吐、错误率、可用性 | 准确率评测、业务满意度 |
| 调优手段 | 缓存、异步、批处理、连接复用 | 模型训练、参数微调 |

这里要特别区分“压测”和“基准测试”。压测强调系统在并发和高负载下是否稳定，常用 `k6`、`Locust` 之类工具模拟用户流量。基准测试更偏向单模型或单服务的能力测量，比如 TTFT、TPOT、tokens/s，强调模型本身的推理特性。工程上通常二者都需要：前者告诉你系统什么时候崩，后者告诉你模型本身快不快。

一个常见边界错误是把离线任务混进在线链路。比如团队有个 nightly job 会批量生成 embedding，如果它和在线问答共用数据库或 GPU，而你在压测时没有暂停它，那么测出来的结果既不能代表纯在线能力，也不能精确定位问题。这类情况应先隔离环境，或者明确说明“这是混部环境下的峰值表现”。

另一个边界错误是没有明确输入分布。LLM 请求不是同质的，输入 token 长度、输出 token 上限、是否走 RAG、是否调用工具，都会显著改变延迟。如果测试脚本全是“你好，请介绍一下自己”这种短 prompt，那么测出的吞吐量通常远高于真实业务。

真实工程例子：一个 RAG 问答系统的线上链路是“API 网关 -> 鉴权 -> 向量检索 -> 重排 -> LLM 生成 -> 审核 -> 返回”。如果压测脚本直接打模型服务，得到的只是模型服务上限；如果产品接口实际还要多一次向量库查询和一次审核服务调用，那么用户真正感知的延迟会更高。因此测试边界必须和真实请求路径一致。

---

## 核心机制与推导

LLM 接口和普通 CRUD API 最大的不同，是响应时间不只是“服务器处理完成所需的固定时间”，而是和生成 token 数直接相关。核心公式可以写成：

$$
\text{End-to-End Latency} = \text{TTFT} + \text{TPOT} \times N
$$

其中：

- TTFT，Time To First Token，第一 Token 时间，白话就是“模型多久开始吐出第一个字”。
- TPOT，Time Per Output Token，单个输出 Token 的平均生成时间，白话就是“后续每个字大概多久生成一个”。
- $N$ 是输出 token 数。

如果一个请求的 TTFT 是 120ms，TPOT 是 18ms，输出 20 个 token，那么总延迟大约是：

$$
120 + 18 \times 20 = 480 \text{ms}
$$

这解释了为什么同一个接口会出现明显长尾：并不是服务器偶尔发神经，而是请求内容天然导致计算量不同。如果你把 `max_tokens` 从 128 提到 1024，哪怕模型和网络完全没变，尾部延迟也会被显著拉长。

吞吐和延迟之间还存在一个非常实用的近似关系：

$$
\text{QPS} \approx \frac{\text{并发中的完成请求数}}{\text{平均请求耗时}}
$$

在单机、稳定负载、请求类型相近时，常可简化理解为“平均每个请求越慢，单位时间能完成的请求越少”。这不是严格物理定律，但对排查拥堵区很有用。

再看拥堵拐点。系统在低并发下，新增一些请求通常只会让延迟平滑上升；但当队列、连接池、GPU batch capacity 或数据库连接数触顶后，延迟会非线性上涨，p95/p99 先爆，再出现超时和 5xx/429。这就是工程上说的“拐点”。

可以用一个玩具例子理解：

| 并发用户数 | 平均延迟 | p95 延迟 | 错误率 |
| --- | --- | --- | --- |
| 5 | 300ms | 420ms | 0% |
| 10 | 360ms | 520ms | 0% |
| 20 | 470ms | 980ms | 0.5% |
| 30 | 680ms | 2300ms | 4% |

从表里看，20 到 30 之间就是明显拐点。平均延迟只涨了 210ms，但 p95 增加了 1320ms，错误率也开始出现。这类模式通常不是“机器有点忙”，而是“系统排队开始堆积”。

网络层也会制造假瓶颈。如果每个请求都重新建 TCP/TLS 连接，那么 `connecting` 和 `tls_handshaking` 时间会被重复计入。Web-PSQC 的 k6 案例提到，启用 keep-alive 和 HTTP/2 后，这部分时间在 p95 上可下降约 70ms。对短回复接口，这已经足以改变整条曲线。

因此，LLM 压测的正确拆法通常是：

1. 先看端到端延迟、p95/p99、错误率，确认是否真的过载。
2. 再拆阶段指标：网络连接、网关、检索、数据库、模型 TTFT、TPOT。
3. 最后结合资源使用率：CPU、GPU、显存、连接池、线程池、数据库慢查询。

---

## 代码实现

下面先给一个可运行的 Python 小脚本，用来演示如何从一批请求样本中计算平均延迟、p95 和理论 QPS。这个例子不依赖第三方库，直接能跑：

```python
from math import ceil

def percentile(values, p):
    assert values, "values cannot be empty"
    assert 0 < p <= 100
    ordered = sorted(values)
    idx = ceil(len(ordered) * p / 100) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]

def estimate_latency(ttft_ms, tpot_ms, output_tokens):
    assert ttft_ms >= 0
    assert tpot_ms >= 0
    assert output_tokens >= 0
    return ttft_ms + tpot_ms * output_tokens

samples_ms = [
    estimate_latency(120, 15, 10),
    estimate_latency(110, 16, 12),
    estimate_latency(130, 15, 11),
    estimate_latency(125, 18, 30),  # 一个长回复请求，制造长尾
    estimate_latency(118, 15, 9),
]

avg_ms = sum(samples_ms) / len(samples_ms)
p95_ms = percentile(samples_ms, 95)
qps_per_worker = 1000 / avg_ms  # 近似：单 worker 每秒完成请求数

assert round(avg_ms, 2) > 0
assert p95_ms >= avg_ms
assert qps_per_worker > 0

print("samples_ms =", samples_ms)
print("avg_ms =", round(avg_ms, 2))
print("p95_ms =", p95_ms)
print("qps_per_worker =", round(qps_per_worker, 2))
```

这个脚本故意放进一个“长回复请求”，你会发现平均值变化不算夸张，但 p95 会明显升高。这就是为什么尾部指标比平均值更适合发现问题。

实际压测时，更常见的是 `k6` 或 `Locust`。`k6` 的优点是单文件、上手快、适合 CI；`Locust` 的优点是 Python 编写，便于嵌入复杂业务逻辑。下面给一个适合 LLM API 的 `k6` 最小脚本：

```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 5 },
    { duration: '30s', target: 10 },
    { duration: '30s', target: 20 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<1500'],
  },
};

const prompts = [
  '解释什么是索引失效',
  '给我一个 Python 二分查找示例',
  '总结 Redis 缓存穿透与缓存雪崩',
];

export default function () {
  const prompt = prompts[Math.floor(Math.random() * prompts.length)];

  const payload = JSON.stringify({
    model: 'your-model',
    messages: [{ role: 'user', content: prompt }],
    max_tokens: 128
  });

  const res = http.post('https://example.com/api/chat', payload, {
    headers: {
      'Content-Type': 'application/json',
      'Connection': 'keep-alive'
    },
    timeout: '20s',
  });

  check(res, {
    'status is 200': (r) => r.status === 200,
    'no server error': (r) => r.status < 500,
  });

  sleep(Math.random() * 2);
}
```

这个脚本体现了三个关键点：

- `stages` 逐步爬升负载，方便观察拐点。
- `sleep(Math.random() * 2)` 引入随机 think time，避免所有虚拟用户同步打点。
- `Connection: keep-alive` 避免每次重建连接，减少网络握手噪声。

如果你更偏向 Python，可以用 `Locust` 写一个接近真实业务的脚本，把不同请求路径、不同 prompt 长度、甚至登录态都纳入模型。真实工程里，建议至少做三轮测试：

| 轮次 | 目标 | 做法 |
| --- | --- | --- |
| 基线测试 | 确认系统健康 | 小并发、短时间、稳定输入 |
| 容量爬坡 | 找到拐点 | 逐级提高并发或 arrival rate |
| 回归验证 | 验证调优是否有效 | 复用同一脚本和同一输入分布 |

真实工程例子：假设你在做一个企业知识库问答系统，链路里有向量检索和关系数据库。压测时要同时记录四类数据：API 总延迟、向量库查询耗时、数据库慢查询、模型 TTFT/TPOT。如果总延迟升高而模型 TTFT 稳定，问题往往不在模型；如果 TTFT 飙升但数据库稳定，则更可能是模型队列或 GPU 饱和。

---

## 工程权衡与常见坑

调优不是“把所有东西都调快”，而是在成本、复杂度和稳定性之间做取舍。最常见的四种调优手段是批处理、缓存、异步、连接复用。

批处理的白话解释是“把多个小请求攒成一批一起算”，优势是提高 GPU 利用率，劣势是会增加单个请求的等待时间。对实时聊天，过度批处理会伤害 TTFT；对批量摘要或审核任务，则通常很划算。

缓存的白话解释是“重复问题直接复用旧结果”。它对热门 FAQ、固定系统提示词、重复检索结果非常有效，但必须处理失效和一致性问题。AWS 在 2026 年 2 月的文章中提到，合适的缓存能把缓存命中的响应降到毫秒级，并显著降低模型调用成本；但缓存命中率、TTL 和语义相似阈值如果设置不当，也可能返回过期或错误内容。

异步的白话解释是“先接住请求，再排队慢慢做”。它适合非交互任务，如批量生成摘要、文档标签、离线分类，不适合要求即时答案的聊天首响应。把本该同步返回的交互接口改成异步，虽然吞吐更高，但用户体验会下降。

下面是常见坑和规避方式：

| 常见坑 | 结果 | 规避方式 |
| --- | --- | --- |
| 只看平均延迟 | 看不见少量极慢请求 | 同时看 p95、p99、错误率 |
| 固定 think time | 人为制造同步流量 | 使用随机 sleep 或 arrival-rate 模型 |
| 未预热就采样 | 冷缓存、冷模型污染结果 | 先 warm-up，再记录正式数据 |
| 输入样本过于单一 | 压测结果偏乐观 | 按真实 prompt 长度和类型构造样本 |
| 忽略连接/TLS 时间 | 把网络问题误判为模型慢 | 观察 connecting、TLS、waiting 分项 |
| 共用脏环境压测 | 难以归因 | 隔离批处理、定时任务和外部流量 |

其中“预热”非常重要。预热就是先跑一小段请求，让模型加载、JIT 编译、缓存填充、连接池建立都完成，再开始正式采样。很多团队第一次压测失败，不是因为系统真有那么差，而是因为把冷启动代价当成了稳定态性能。

另一个常见误区是“调优后只看吞吐，不看尾部”。例如你加了激进批处理后，QPS 提升了 25%，但 p95 从 900ms 涨到 2600ms。对于在线客服，这通常不是优化，而是退化。工程上必须先明确 SLO。SLO 就是服务目标，例如“99% 请求在 2 秒内返回且错误率低于 1%”。没有 SLO，调优会变成随意拉扯指标。

---

## 替代方案与适用边界

`k6` 和 `Locust` 不是唯一方案，但它们是最通用、成本最低的起点。什么时候该继续用，什么时候该换工具，取决于你的目标。

| 方案 | 适用情况 | 局限 |
| --- | --- | --- |
| k6 | 快速压 HTTP API、接入 CI、脚本简单 | 复杂用户态逻辑不如 Python 灵活 |
| Locust | 需要 Python 逻辑、复杂登录和多步骤流程 | 部署和分布式管理略重 |
| 模型专用基准工具 | 关注 TTFT、ITL、TPS 等模型指标 | 不等于真实业务全链路压测 |
| 云厂商压测/监控平台 | 已部署在特定云环境，希望少维护工具链 | 自定义程度受平台限制 |
| 异步队列模拟高吞吐 | 非交互任务、批量处理 | 不适用于首屏响应要求高的场景 |

如果你的目标是“知道模型服务本身能吐多快”，就应该补充使用 NVIDIA GenAI-Perf、AIPerf、LLMPerf 这类更偏模型指标的工具。NVIDIA 文档明确区分了负载测试和推理基准测试：前者看系统能否承受真实流量，后者看模型吞吐和延迟特征。两者不能互相替代。

如果你的场景运行在 SageMaker、Bedrock、Azure AI 或自建推理网关上，也可以借助平台侧监控观察实例扩缩容、GPU 利用率、429/5xx、连接复用情况。但云平台监控通常更像“事后体检”，而不是“可控实验”。要定位细节，还是需要自己设计负载脚本。

一个真实工程上的替代路径是“分级服务”：把高频简单请求先交给轻量模型或缓存命中，只有复杂请求才转大模型。这不是纯性能优化，而是架构优化。它的收益往往大于单纯在大模型服务上硬挤 10% QPS，但前提是你能接受路由复杂度和答案一致性管理成本。

---

## 参考资料

| 来源 | 重点内容 | 访问用途 |
| --- | --- | --- |
| [NVIDIA NIM Benchmarking Guide](https://docs.nvidia.com/nim/benchmarking/llm/latest/) | 区分负载测试与推理基准测试，介绍 TTFT、ITL、TPS、RPS 等指标 | 建立指标体系与测试方法 |
| [AWS Neuron LLM Inference Benchmarking Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/llm-inference-benchmarking-guide.html) | 给出 TTFT、TPOT、端到端延迟公式，说明 token 级指标定义 | 支撑延迟公式与术语定义 |
| [Web-PSQC: How Load Testing with K6 Unveils Performance Bottlenecks](https://www.web-psqc.com/blog/performance/how-load-testing-with-k6-unveils-performance-bottlenecks) | 说明 p95/p99、连接复用、keep-alive、HTTP/2 与现实流量节奏 | 支撑 k6 压测实践与网络瓶颈判断 |
| [AWS Database Blog: Optimize LLM Response Costs and Latency with Effective Caching](https://aws.amazon.com/blogs/database/optimize-llm-response-costs-and-latency-with-effective-caching/) | 总结提示词缓存、语义缓存、多层缓存与失效策略 | 支撑缓存调优部分 |
| [Gary Stafford: Load Testing SageMaker Real-Time Inference Endpoints with Locust](https://garystafford.medium.com/finding-your-llms-breaking-point-load-testing-sagemaker-real-time-inference-endpoints-with-locust-5b60cd1dfbf5) | 展示对实时推理端点进行容量爬坡和拐点识别的工程案例 | 作为云端真实工程参考 |
