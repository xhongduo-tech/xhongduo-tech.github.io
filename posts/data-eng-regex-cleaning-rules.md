## 核心结论

正则表达式清洗规则库，本质上是一套“可配置、可复用、可审计”的文本处理链。它不是把几条正则随手写进业务代码，而是把敏感信息提取与脱敏、HTML 实体解码、零宽字符和控制字符清理、性能防护这几类动作统一进同一个规则系统里。

对工程实践来说，首要目标不是“正则写得多花”，而是三件事：

1. 同一类文本在不同服务、不同环境里使用同一套规则，不因为复制粘贴出现行为漂移。
2. 敏感字段在进入日志平台、搜索系统、监控系统之前就完成脱敏，避免原文外泄。
3. 规则执行可控，能避免灾难性回溯，也就是某些输入让正则匹配时间急剧膨胀，形成 ReDoS 风险。

一个最小玩具例子：

原始文本是 `alice@example.com 登录成功`，邮箱规则命中后，把用户名部分替换成 `***`，结果变成 `***@example.com 登录成功`。这说明规则库不是只做“找出来”，还要定义“怎么替换、在哪一层替换、替换后是否继续执行后续规则”。

下表可以先建立整体视图：

| 分类 | 典型规则 | 作用 |
| --- | --- | --- |
| 敏感信息 | 邮箱、URL、手机号、身份证 | 提取或脱敏敏感字段 |
| 文本规范化 | HTML 实体解码 | 把 `&amp;` 还原为 `&` |
| Unicode 清理 | 零宽字符、控制字符 | 消除看不见但会影响匹配的字符 |
| 性能防护 | 预编译、超时、限制模式 | 避免回溯爆炸和 ReDoS |
| 规则治理 | JSON/YAML 配置化 | 版本管理、复用、按环境发布 |

---

## 问题定义与边界

先定义问题。这里的“清洗”不是自然语言润色，也不是全文理解，而是针对文本中的特定模式做确定性处理。确定性，意思是输入相同，规则相同，输出就应该相同，不能依赖模型猜测。

规则库通常处理这几类输入：

| 输入类型 | 常见内容 | 目标 |
| --- | --- | --- |
| 应用日志 | 登录、支付、异常堆栈 | 脱敏后再写入监控平台 |
| 客服文本 | 用户留言、转写内容 | 清理隐私字段和脏字符 |
| 抓取文本 | HTML 页面、富文本片段 | 解码实体并清理不可见字符 |
| 导入数据 | CSV、TSV、表单导出 | 规范格式，便于下游处理 |

边界也必须说清楚。规则库适合处理“模式明确、可枚举”的问题，不适合处理“语义依赖很强”的问题。比如：

- “手机号、邮箱、身份证号”适合正则，因为形式特征明显。
- “这段话是不是包含商业机密”不适合只靠正则，因为它依赖语义理解。

另一个边界是处理层级。最稳妥的做法，是在采集端或入库前处理，而不是等数据进入外部平台后再补救。

| 规则应用层 | 触发条件 | 优点 | 风险 |
| --- | --- | --- | --- |
| 采集前 | 日志生成后、发送前 | 原文不出本地，最安全 | 部署点更多 |
| 入库前 | 网关、清洗服务 | 统一治理 | 已经过了一层传输 |
| 云端平台内 | SaaS 日志处理 | 配置集中 | 原文可能已上传 |

真实工程例子：客服系统把通话转写文本发送到日志平台。如果文本里包含“我的手机号是 13800138000，邮箱是 alice@example.com”，正确做法是在本地采集器或日志网关先执行脱敏，再把结果发往平台，而不是把原文直接送上去后再想办法屏蔽。

---

## 核心机制与推导

规则库可以抽象成一个处理函数：

$$
Cleanup(Text) = Normalize(Decode(ApplyRules(Text, P)))
$$

其中：

- `Text` 是输入文本。
- `P = \{p_1, p_2, ..., p_n\}` 是规则集合。
- `ApplyRules` 表示按顺序执行规则。
- `Decode` 表示 HTML 实体解码。
- `Normalize` 表示清理零宽字符、控制字符等不可见污染。

更细一点，可以写成：

$$
T_0 = Text,\quad T_i = Replace(T_{i-1}, p_i),\quad Result = Normalize(HtmlUnescape(T_n))
$$

这里的顺序很重要。因为不同规则之间会相互影响。

一个常见顺序是：

1. 先做结构性脱敏，比如邮箱、手机号、身份证。
2. 再做 HTML 实体解码。
3. 最后做 Unicode 规范化和不可见字符清理。

为什么不把顺序反过来？因为解码后可能会出现新的符号边界，影响后续匹配。比如文本里是：

`联系我：alice&#64;example.com`

如果先匹配邮箱，往往匹配不到；先把 `&#64;` 解码成 `@`，后面邮箱规则才能命中。

零宽字符也类似。零宽字符，白话说，就是“看不见但真实存在的字符”。例如 `U+200B` 零宽空格会让两个肉眼看起来相同的字符串，实际上字节序列不同。于是搜索、去重、哈希、精确匹配都会出问题。

玩具例子：

- 肉眼看到的是 `abc123`
- 实际内容可能是 `abc\u200b123`

这时普通字符串比较会失败，数据库唯一键、缓存键、分词结果都可能出错。

因此，规则库不仅是“正则替换器”，还是一个受控的文本标准化流水线。

再看性能问题。很多初学者以为正则慢，是因为“正则本来就慢”。这不准确。真正的问题常常是模式写法触发了大量回溯。回溯，白话说，就是正则引擎走错路以后不断退回重试。模式中如果出现嵌套量词、模糊边界、超长输入，回溯次数会爆炸。

典型危险写法：

```text
(a+)+$
```

当输入接近但不匹配时，匹配器会尝试大量路径，时间复杂度急剧上升。

所以规则库设计的核心推导是：

- 规则必须有优先级。
- 规则必须配置化。
- 执行器必须缓存编译结果。
- 危险模式必须限制，必要时加超时、长度阈值或更换实现。

---

## 代码实现

下面给出一个可运行的 Python 版本，演示“配置驱动规则库”的最小实现。配置驱动，白话说，就是规则写在配置里，执行器只负责读取并执行，不把具体模式硬编码进业务流程。

```python
import re
import html
from dataclasses import dataclass
from typing import Callable, Pattern

@dataclass
class Rule:
    name: str
    pattern: Pattern[str]
    replacer: Callable[[re.Match[str]], str]

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")

def mask_email(m: re.Match[str]) -> str:
    local, domain = m.group(1), m.group(2)
    return "***@" + domain

def mask_phone(m: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    if len(digits) == 11:
        return digits[:3] + "****" + digits[-4:]
    return "***"

def mask_idcard(m: re.Match[str]) -> str:
    s = m.group(0)
    return s[:6] + "********" + s[-4:]

RULES = [
    Rule(
        name="email",
        pattern=re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"),
        replacer=mask_email,
    ),
    Rule(
        name="phone",
        pattern=re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
        replacer=mask_phone,
    ),
    Rule(
        name="idcard",
        pattern=re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
        replacer=mask_idcard,
    ),
]

def clean_text(text: str) -> str:
    text = html.unescape(text)
    for rule in RULES:
        text = rule.pattern.sub(rule.replacer, text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    return text

sample = "用户 alice&#64;example.com 手机 13800138000 身份证 11010519491231002X\u200b 登录成功"
result = clean_text(sample)

assert "***@example.com" in result
assert "138****8000" in result
assert "110105********002X" in result
assert "\u200b" not in result
assert "&" not in result  # 这里的输入是 &#64;，应被解码为 @，不会保留实体写法

print(result)
```

这段代码体现了几个关键点：

| 实现点 | 作用 | 为什么需要 |
| --- | --- | --- |
| `html.unescape` | HTML 实体解码 | 让 `&#64;`、`&amp;` 先恢复原字符 |
| 预编译 `re.compile(...)` | 复用模式对象 | 避免每次调用重复编译 |
| 独立 `replacer` | 自定义脱敏逻辑 | 比固定替换字符串更灵活 |
| 零宽与控制字符单独处理 | 统一规范化 | 不把所有问题都塞进一个大正则 |

如果要做成工程可维护的规则库，建议把规则定义迁移到 JSON 或 YAML。比如：

```json
[
  {
    "name": "email",
    "pattern": "\\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\\.[A-Za-z]{2,})\\b",
    "action": "mask_email",
    "enabled": true
  },
  {
    "name": "phone",
    "pattern": "(?<!\\d)(?:\\+?86[- ]?)?1[3-9]\\d{9}(?!\\d)",
    "action": "mask_phone",
    "enabled": true
  }
]
```

然后由执行器读取配置、构建规则池、按优先级执行。这样一来，新增规则不需要改主流程代码。

真实工程例子：一个日志网关同时服务 20 个微服务。支付服务需要脱敏银行卡和身份证，客服服务需要脱敏手机号和邮箱，搜索服务只需要清理控制字符。最合理的做法不是在 20 个仓库里各写一份正则，而是把规则集中到一个配置仓库，按服务标签或日志来源选择性启用。

---

## 工程权衡与常见坑

第一类坑是误报和漏报。

正则越宽松，误报越多；越严格，漏报越多。比如手机号规则如果只写 `1\d{10}`，会误伤很多长数字片段；如果严格到只允许纯 11 位且两边不能有数字，误报降低，但某些带国家码或分隔符的形式又可能漏掉。

第二类坑是隐形字符。

| 陷阱 | 后果 | 规避方式 |
| --- | --- | --- |
| 零宽字符 | 搜索失败、去重失败、哈希不一致 | 清理 `Cf` 类字符或显式列举 |
| 控制字符 | 日志换行错乱、CSV 解析失败 | 清理 `C` 类或常见控制区间 |
| HTML 实体未解码 | 本该命中的邮箱/URL 匹配不到 | 先解码再匹配 |
| 规则顺序错误 | 先替换后破坏边界 | 固定处理链顺序 |

第三类坑是性能和安全。

很多系统上线时数据量不大，几条正则看起来都正常；但一到日志洪峰或者出现恶意输入，性能问题就集中爆发。常见诱因有：

- 使用 `.*` 吞掉过大范围。
- 嵌套量词，如 `(.*a)*`。
- 在长文本上反复执行高成本模式。
- 每次请求都重新编译正则。

工程上应做这些约束：

1. 预编译并缓存规则对象。
2. 为单条文本设置长度上限。
3. 把高风险模式加入评审清单。
4. 对关键链路增加超时或隔离执行。
5. 用更具体的字符类代替泛化写法。

例如邮箱提取，不要轻易写成：

```text
.+@.+\..+
```

因为它虽然“看起来能用”，但边界极差，容易跨越过多文本。更合理的是把用户名、域名、后缀分别限制在明确字符集里。

第四类坑是规则治理。

如果规则散落在多个项目中，问题不会马上出现，但几个月后通常会变成：

- A 服务脱敏了邮箱，B 服务没脱敏。
- 测试环境和生产环境规则版本不同。
- 某条规则修复后，只有部分服务更新。

所以规则库必须版本化。版本化，白话说，就是像管理代码一样管理规则，让每次改动都有历史记录、评审记录和回滚能力。

---

## 替代方案与适用边界

正则规则库不是唯一方案，它适合“文本非结构化、字段边界不稳定”的场景。如果输入已经高度结构化，字段级处理通常更可靠。

| 方案 | 优势 | 适用场景 | 局限 |
| --- | --- | --- | --- |
| 字段级 Mask | 精准、性能稳定 | JSON、数据库记录、固定表单 | 只适合已知字段 |
| 正则规则库 | 灵活、覆盖未知文本 | 日志、客服文本、混合输入 | 有误报漏报和平衡问题 |
| 词法/解析器方案 | 语法边界更清晰 | URL、HTML、代码片段 | 实现成本更高 |
| 模型识别 | 能处理复杂语义 | 命名实体、语义敏感内容 | 成本高，不够确定性 |

举个对比：

- 如果请求体是固定 JSON：`{"email":"a@b.com","phone":"13800138000"}`，直接按字段名脱敏最稳。
- 如果输入是一段自由文本：`请联系我 alice@example.com，或者拨打 13800138000`，字段边界不固定，就更适合正则规则库。

还有一个经常被忽略的边界是“规则共享方式”。小项目里把规则放在本地文件足够；多团队协作时，最好进入统一配置源，比如 Git 仓库或配置中心，再由 CI/CD 分发到各个运行点。

这类设计的价值不在“形式优雅”，而在于你能回答下面这些问题：

- 当前线上到底是哪一版规则？
- 哪条规则是谁在什么时候改的？
- 某次误脱敏或漏脱敏是否能回滚？
- 不同环境是否保证规则一致？

如果这些问题答不上来，规则库通常还不算工程化。

---

## 参考资料

- Dynatrace, Mask sensitive information in log analytics: https://docs.dynatrace.com/docs/analyze-explore-automate/log-monitoring-v1/mask-sensitive-information-in-log-analytics
- Dynatrace, Methods of masking sensitive data: https://docs.dynatrace.com/docs/analyze-explore-automate/logs/lma-use-cases/methods-of-masking-sensitive-data
- Microsoft Learn, Backtracking in regular expressions: https://learn.microsoft.com/en-us/dotnet/standard/base-types/backtracking-in-regular-expressions
- Microsoft Learn, Regular expression options: https://learn.microsoft.com/en-us/dotnet/standard/base-types/regular-expression-options
- Cloud Native Now, Config management best practices: https://cloudnativenow.com/features/5-cloud-native-app-config-management-best-practices/
- AI Text Cleaner, Zero-width characters guide: https://ai-text-cleaner.com/guides/zero-width-characters/
