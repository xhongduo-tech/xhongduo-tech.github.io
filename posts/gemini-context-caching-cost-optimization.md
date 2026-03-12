## 核心结论

Gemini 的上下文缓存，本质上是在服务端复用一段已经做过预填充计算的前缀。预填充计算可以白话理解为：模型先把长提示词读一遍，并把后续生成要用的中间状态准备好。这样下一次再遇到同样的开头，就不必从头重新算一遍。

它有两种机制：

| 机制 | 是否自动启用 | 命中条件 | 折扣是否保证 | TTL 控制 | 额外存储费 |
| --- | --- | --- | --- | --- | --- |
| Implicit caching | 是 | 短时间内出现相同前缀 | 否 | 否 | 否 |
| Explicit caching | 否 | 请求显式引用 `cache id` | 是 | 是，默认 1 小时 | 是 |

对 Gemini 2.5 系列，官方公开的缓存命中折扣是输入价的约 1 折，也就是约 90% 折扣。区别不在折扣比例，而在“能不能稳定命中”和“要不要自己管理生命周期”。Implicit 适合“前缀经常重复，但你不想维护缓存对象”；explicit 适合“有稳定的大前缀，而且希望命中可控”。

结论可以压缩成一句话：如果你的请求里有大段不变的系统指令、文档、代码库摘要、长 PDF 或视频说明，那么 Gemini context caching 不是“锦上添花”，而是直接影响推理成本和首 token 延迟的基础优化项。

---

## 问题定义与边界

问题很简单：同一段长上下文被反复发送，重复付费，也重复等待。

这里的“上下文”通常是这几类内容：

| 内容类型 | 白话解释 | 是否适合缓存 |
| --- | --- | --- |
| 系统指令 | 模型每次都要先读的规则说明 | 很适合 |
| 知识文档 | 产品文档、FAQ、设计文档 | 很适合 |
| 多模态大文件 | PDF、视频、音频的解析输入 | 很适合 |
| 用户问题 | 每次都不同的提问 | 不适合放前缀缓存 |
| 对话历史尾部 | 最近轮次经常变化 | 只适合部分稳定段 |

边界主要有四个。

第一，缓存只对“前缀”起作用。前缀就是 prompt 开头那一段连续 token。不是任意位置重复都能命中，而是开头相同才最有机会命中。

第二，存在最小 token 门槛。按照当前官方文档，Gemini API 的 implicit caching 最低输入长度为：

| 模型 | Implicit 最低 token |
| --- | --- |
| Gemini 2.5 Flash | 1024 |
| Gemini 2.5 Pro | 4096 |

如果连这个门槛都没到，缓存命中基本无从谈起。比如一个只有 800 token 的输入，即使完全重复，也不会触发 Gemini 2.5 Flash 的 implicit caching。

第三，explicit caching 有 TTL。TTL 是 time to live，白话解释就是“这份缓存还能活多久”。默认 1 小时，但可以自定义延长。TTL 越长，缓存越稳定；TTL 越长，存储费也越高。

第四，缓存不会改变模型语义。它优化的是“前缀重复计算”和“重复输入计费”，不是让模型少看内容。被缓存的内容在语义上仍然属于 prompt 的一部分，只是中间状态被复用了。

一个玩具例子可以先建立直觉。

你有一个数学助教机器人，每次都带 1500 token 的固定规则：

- 先定义符号
- 再列公式
- 不要跳步
- 输出用 Markdown

然后学生不断问不同题目：

- 第 1 次问：解一元二次方程
- 第 2 次问：解释导数定义
- 第 3 次问：证明等比数列求和

如果这 1500 token 总是在开头保持不变，那么后两次、第三次请求就有机会命中缓存。你节省的不是“学生问题”的成本，而是这 1500 token 的重复成本。

---

## 核心机制与推导

要理解 Gemini context caching，先区分两个阶段。

1. 预填充阶段：模型读取输入前缀，构建注意力层里的 KV 状态。
2. 生成阶段：模型基于这些状态继续预测后续 token。

这里的 KV cache，可以白话理解为：模型已经把“前面读过的东西”压成一组可继续使用的中间表示。上下文缓存复用的正是这部分结果，而不是把最终回答直接缓存起来。

### 1. Implicit caching 的命中逻辑

Implicit caching 不需要你先建缓存对象。平台自动观察最近请求，如果多个请求的开头高度一致，并且时间上足够接近，就可能把重复前缀识别为 cached tokens。

它不是“全文去重”，而是“前缀匹配”。所以工程上最关键的不是“内容重复”，而是“重复内容是否稳定地放在最前面”。

可以把它抽象成：

$$
\text{prompt} = P_{\text{stable}} \Vert P_{\text{dynamic}}
$$

其中：

- $P_{\text{stable}}$ 是稳定前缀，比如系统规则、文档摘要、工具说明
- $P_{\text{dynamic}}$ 是动态尾部，比如用户问题、会话变量、最新检索结果

只有当多个请求共享同一个 $P_{\text{stable}}$ 时，缓存才有意义。

### 2. Explicit caching 的工作方式

Explicit caching 是先创建一个缓存对象，再在后续请求里引用它。你可以把它理解成“把一段长前缀注册为可复用资源”。

它比 implicit 多了一层资源管理：

- 创建缓存
- 获得 `cache.name`
- 在生成请求里传入 `cached_content`
- 必要时延长 TTL
- 不再需要时删除

这里的核心不是“模型少看一遍文档”，而是“服务端不用重新把这段文档编码成新的前缀状态”。

### 3. 费用模型

显式缓存的总成本可近似写成：

$$
\text{Total Cost} \approx C_{\text{storage}} + C_{\text{cached input}} + C_{\text{uncached input}} + C_{\text{output}}
$$

进一步展开：

$$
C_{\text{storage}} \approx \text{cached\_tokens} \times \text{storage\_rate} \times \text{TTL}
$$

$$
C_{\text{cached input}} \approx \text{cached\_tokens} \times \text{discounted\_input\_rate}
$$

$$
C_{\text{uncached input}} \approx (\text{prompt\_tokens} - \text{cached\_tokens}) \times \text{normal\_input\_rate}
$$

输出 token 仍按正常输出价计费，不因为前缀命中而免费。

一个数值化的玩具推导：

- 固定前缀：6000 token
- 每次用户追加问题：300 token
- 共请求 20 次
- TTL：2 小时

如果不用缓存，输入总量近似是：

$$
20 \times (6000 + 300) = 126000
$$

如果用 explicit caching，第一次要写入缓存，并支付 2 小时存储费；后续 20 次请求里，6000 token 那部分按缓存价读入，300 token 按普通输入价计费。只要重复次数足够多，存储费通常会被大量重复前缀带来的折扣覆盖掉。

### 4. 真实工程例子

真实工程里最常见的是客服或企业知识问答系统。

假设一个客服 agent 每次都需要这些固定输入：

- 1800 token 系统规则
- 9000 token 产品手册摘要
- 6000 token 售后流程说明
- 2000 token 风险回复规范

总共接近 18000 token，而且一天要处理几千个问题。真正变化的只有：

- 用户当前问题
- 最近几轮对话
- 当前订单状态

这种场景如果不做缓存，你每次都在重传并重算一份大前缀。对 Gemini 来说，这是 context caching 最标准的收益区间。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现，用来模拟“前缀稳定时，缓存如何降低重复计算次数”。它不是 Gemini SDK 调用，而是一个帮助理解机制的本地版本。

```python
from dataclasses import dataclass

@dataclass
class CacheStats:
    prefill_calls: int = 0
    cache_hits: int = 0

class PrefixCacheEngine:
    def __init__(self):
        self.cache = {}
        self.stats = CacheStats()

    def _prefill(self, prefix: str) -> str:
        # 模拟昂贵的前缀预处理
        self.stats.prefill_calls += 1
        return f"kv::{hash(prefix)}"

    def generate(self, prefix: str, suffix: str) -> dict:
        if prefix in self.cache:
            kv_state = self.cache[prefix]
            self.stats.cache_hits += 1
        else:
            kv_state = self._prefill(prefix)
            self.cache[prefix] = kv_state

        # 这里只模拟“基于前缀状态继续生成”
        output = f"[{kv_state}] answer for: {suffix}"
        return {
            "cached": prefix in self.cache,
            "output": output,
        }

engine = PrefixCacheEngine()

stable_prefix = "SYSTEM: follow policy\nDOC: refund rules v1\n"
r1 = engine.generate(stable_prefix, "用户问：退款多久到账？")
r2 = engine.generate(stable_prefix, "用户问：过保后还能维修吗？")
r3 = engine.generate("SYSTEM: user_id=42\nDOC: refund rules v1\n", "用户问：退款多久到账？")

assert engine.stats.prefill_calls == 2
assert engine.stats.cache_hits == 1
assert "answer for" in r1["output"]
assert "answer for" in r2["output"]
assert "answer for" in r3["output"]
```

这段代码说明两件事：

- 相同前缀第二次出现时，可以直接命中。
- 只要你把动态字段 `user_id=42` 塞进前缀开头，就会形成一个新前缀，导致缓存失效。

下面再看更接近 Gemini API 的显式缓存写法。接口名会随 SDK 版本小幅变化，但流程是稳定的。

```python
from google import genai
from google.genai import types

client = genai.Client()

model = "gemini-2.5-flash"

cache = client.caches.create(
    model=model,
    config=types.CreateCachedContentConfig(
        system_instruction=(
            "你是企业知识库问答助手。回答必须引用规则，不得编造。"
        ),
        contents=[
            "这里放长文档、FAQ 摘要或上传后的文件引用"
        ],
        ttl="2h",
    ),
)

response = client.models.generate_content(
    model=model,
    contents="用户问题：订单取消后发票如何处理？",
    config=types.GenerateContentConfig(
        cached_content=cache.name
    ),
)

assert cache.name
assert response is not None
```

如果你不想自己管理缓存对象，而是想吃 implicit caching，代码层面的重点反而更少：

- 把稳定大前缀放到最前面
- 动态字段放到末尾
- 连续请求尽量保持相同开头
- 读取返回里的 `usage_metadata`，看 `cachedContentTokenCount` 或同类字段

工程上推荐把 prompt 拆成三段：

| 段落 | 内容 | 位置 |
| --- | --- | --- |
| Stable prefix | 系统指令、文档、规则、工具说明 | 最前面 |
| Semi-stable middle | 最近会共享的一小段上下文 | 中间 |
| Dynamic suffix | 用户问题、变量、实时检索结果 | 最后面 |

这样即使你只用 implicit caching，也能显著提高命中率。

---

## 工程权衡与常见坑

缓存不是“只要打开就一定省钱”。真正决定收益的是重复率、前缀稳定度和 TTL 管理。

### 常见坑 1：把动态字段放在前缀里

最常见错误是把这些内容放在最前面：

- 用户 ID
- 时间戳
- trace id
- 当前实验开关
- 实时检索片段

这会让每个请求的开头都不同。对 prefix cache 来说，哪怕只改动前面一点点，也可能形成完全不同的缓存键。

### 常见坑 2：前缀太短

如果你的稳定前缀只有几百 token，就算内容完全相同，也可能达不到最低缓存门槛。此时不要“为了缓存而缓存”，而要先确认这部分是否真的占了主要输入成本。

### 常见坑 3：TTL 设置错误

TTL 太短，结果是缓存刚建好就过期，频繁重建。
TTL 太长，结果是对象长期占用存储费，尤其是显式缓存的大文档场景。

一个实用经验是：按业务访问波峰设置 TTL，而不是按“理论最长可能复用时间”设置。比如工作台客服系统白天请求密集，2 到 4 小时可能比 24 小时更合理。

### 常见坑 4：把缓存当成知识库更新机制

缓存不是文档版本管理。文档一旦变化，旧缓存不会自动同步成新知识。你要自己决定：

- 是延长旧缓存继续用
- 还是创建新版本缓存
- 或者删除旧缓存

如果你的知识库一天变十几次，explicit caching 的管理成本可能会明显升高。

### 常见坑 5：只看 token 价格，不看延迟和吞吐

缓存优化的收益不只在账单。对于长前缀，大量请求会把 prefill 阶段的计算压到服务端缓存命中路径上，通常也能改善首 token 延迟，并提升系统在高并发时的吞吐稳定性。

下面是一个更实用的排查表：

| 坑 | 现象 | 规避方式 |
| --- | --- | --- |
| 前缀混入动态字段 | 命中率极低 | 动态内容全部后移 |
| 低于最小 token 阈值 | `cached tokens` 一直为 0 | 合并稳定内容，先达到门槛 |
| TTL 过短 | 缓存频繁失效重建 | 按访问峰值延长 TTL |
| 文档更新未切版本 | 回答引用旧规则 | 用版本号管理 cache |
| 只依赖 implicit | 成本波动大 | 高重复场景改用 explicit |

真实工程里，一个好用的做法是给缓存前缀做“规范化”：

- 固定段落顺序
- 固定标题命名
- 固定空行数量
- 固定工具描述顺序
- 固定 schema 输出格式

因为 token 序列只要变了，哪怕语义没变，也可能影响匹配。

---

## 替代方案与适用边界

Gemini context caching 不是所有场景都值得上。

### 1. 适合使用 Gemini explicit caching 的场景

- 大前缀稳定，重复次数多
- 命中必须可控
- 希望多个请求共享同一段固定上下文
- 可以接受 TTL 和缓存对象管理

比如企业客服、代码库分析、长 PDF 连续问答、视频内容多轮分析。这类任务通常“固定材料很长，用户问题很短”，是最典型的缓存收益模型。

### 2. 只适合 implicit caching 的场景

- 你不想维护缓存对象
- 前缀经常重复，但生命周期不值得单独管理
- 业务波动大，希望平台自动处理

比如一个轻量问答服务，白天某些热门文档会被频繁访问，但热度只持续几十分钟。这时 implicit 往往已经够用。

### 3. 不适合上下文缓存的场景

- 每次输入都完全不同
- 稳定前缀很短
- 文档变化频率极高
- 用户个性化内容必须放在最前面

这类场景里，更有效的优化通常不是 cache，而是缩短 prompt、做检索裁剪、改写系统指令，或者拆成两阶段推理。

### 4. 与 OpenAI prompt caching 的差异

两者目标相似，都是降低重复前缀成本，但产品形态不同。

| 维度 | Gemini explicit caching | OpenAI prompt caching |
| --- | --- | --- |
| 是否需手动创建缓存对象 | 是 | 否 |
| 是否可显式引用 cache id | 是 | 否 |
| 是否支持 TTL 生命周期控制 | 是 | 是，按请求设置 retention policy |
| 是否有单独存储费 | 有 | 无单独缓存写入费 |
| 优化核心 | 可管理的上下文资源 | 自动前缀路由与命中 |

这意味着 Gemini explicit 更像“可管理的长期前缀资源”，而 OpenAI prompt caching 更像“平台自动维护的前缀命中机制”。如果你需要明确控制哪段大上下文被多个请求复用，Gemini explicit 更直接；如果你只想让平台自动对相同前缀降费，自动缓存方案更省运维。

---

## 参考资料

1. Google AI for Developers, Context caching: https://ai.google.dev/gemini-api/docs/caching/
2. Google Developers Blog, Gemini 2.5 Models now support implicit caching: https://developers.googleblog.com/gemini-2-5-models-now-support-implicit-caching/
3. Google Cloud, Context caching overview on Vertex AI: https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview
4. Google Cloud, Vertex AI pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing
5. OpenAI, Prompt caching guide: https://platform.openai.com/docs/guides/prompt-caching
