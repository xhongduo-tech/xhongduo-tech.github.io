---
title: 日期时间 API 与国际化
date: 2026-08-07
---

# 日期时间 API 与国际化

<div class="epigraph">
<p>时间不是钟表上的数字，而是「时区 + 时刻 + 日历」的三重叠加；国际化不是翻译，而是「语言 + 地区 + 格式」的整套适配。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第2卷第6章 ｜ 2026-08-07</p>
</div>

## 为什么从日期时间与国际化开始

程序处理的「时间」比你以为的复杂得多：一个订单时间，在纽约用户眼里和北京用户眼里是**不同的钟点**；同一串 `2026-08-07`，美式写法是 `08/07/2026`，中式是 `2026/08/07`。Java 8 之前的 `Date`/`Calendar` 因为设计缺陷（可变、线程不安全、月份从 0 开始、时区处理混乱）成了历史包袱；Java 8 的 **`java.time`**（JSR-310）重新设计，不可变、线程安全、语义清晰。**国际化（i18n）**则解决「一套代码服务多语言多地区」——用 `Locale` 描述「哪里的用户」，用 `ResourceBundle` 装各语言的文案。这一篇把新时间 API 与国际化两件套讲透。

## 1 java.time 的核心类型：时刻、日期与区间

`java.time` 的设计把「时间」拆成几个语义各异的类型，先分清它们：

| 类型 | 语义 | 示例 |
| --- | --- | --- |
| `LocalDate` | 本地日期（无时间无时区） | `2026-08-07` |
| `LocalTime` | 本地时间（无日期无时区） | `14:30:00` |
| `LocalDateTime` | 日期 + 时间（无时区） | `2026-08-07T14:30` |
| `Instant` | **时刻**（UTC 绝对时间点） | `2026-08-07T06:30:00Z` |
| `ZonedDateTime` | 时刻 + 时区 | `2026-08-07T14:30+08:00[Asia/Shanghai]` |
| `Duration` | 时间量（秒/纳秒） | `PT2H30M`（2 小时 30 分） |
| `Period` | 日期间隔（年/月/日） | `P2Y3M4D` |

**重点结论：区分「本地时间」与「时刻」是理解全部的关键。** `LocalDateTime` 没有时区——它不知道「这一刻在格林尼治是几点」，只适合「不考虑时区的业务」（闹钟时间、营业时间）；**跨时区的绝对时间（下单时刻、日志时间戳）必须用 `Instant` 或 `ZonedDateTime`**。用 `LocalDateTime` 存「北京时间 14:30 的订单」会在服务器迁移时区后变质——这是分布式系统里著名的时区 bug 温床。<span class="marginnote">记忆口诀：<strong>`Instant` 是「世界统一的一刻」，`LocalDateTime` 是「某地钟面上显示的数字」，`ZonedDateTime` = 时刻 + 时区。</strong> 数据库里存时间戳用 `Instant`/`OffsetDateTime`，展示给用户再转成本地时区。</span>

**构造与运算**——都返回**新对象**（不可变），绝不改原对象：

```java
LocalDate today = LocalDate.now();
LocalDate day = LocalDate.of(2026, 8, 7);
LocalDate nextWeek = today.plusWeeks(1);
boolean after = today.isAfter(day);
int year = today.getYear();
```

`Duration.between(t1, t2)` 算时间差，`ChronoUnit.DAYS.between(d1, d2)` 按指定单位算差——比老 `Date` 的减法直观得多。

## 2 时区与时区转换

**时区（time zone）**是「地区 ↔ UTC 偏移」的映射，由 IANA 管理（如 `Asia/Shanghai`、`America/New_York`）。Java 8 里的时区处理：

```java
ZonedDateTime beijing = ZonedDateTime.of(
        LocalDateTime.of(2026, 8, 7, 14, 30),
        ZoneId.of("Asia/Shanghai"));          // 2026-08-07T14:30+08:00
ZonedDateTime newYork = beijing.withZoneSameInstant(
        ZoneId.of("America/New_York"));       // 同一时刻，纽约是 02:30
```

`withZoneSameInstant` 把**同一时刻**换到另一个时区显示——这是「给用户看他当地时间」的正确做法。**不要**用 `withZoneSameLocal`（那会改时刻，只剩钟面数字相同，语义已变）。

**为什么老 `Date` 的时区处理是灾难**：`Date` 本身是「绝对时刻」，但 `toString()` 打印的是**当前 JVM 默认时区**的本地时间——同一 `Date` 在不同机器打印不同，调试时极易误解。而 `java.time` 把「时刻」与「在哪个时区看它」彻底分开，从根上消除了这类错乱。

**公式解析：Unix 时间戳与时刻的关系**：

$$

\text{Instant} = \text{epochSecond} \times 10^9 + \text{nanoAdjustment}

$$

三步拆解：

- **第一步，看懂 epoch**：Unix 时间戳是「自 1970-01-01T00:00:00Z 以来的秒数」——它是**全局统一**的整数，与任何时区无关。
- **第二步，Instant 的存储**：`Instant` 内部就是「秒 + 纳秒」两个字段——`epochSecond` 与 `nanoAdjustment`。所以 `Instant` 本质是时间戳的类型化封装。
- **第三步，为什么它可靠**：因为存储的是「绝对秒数」而非「钟面时间」，无论服务器在哪个时区、哪台机器，同一个 `Instant` 永远代表同一个物理时刻。<span class="marginnote">这条「用绝对秒数表示时刻」的思想，是分布式系统时间处理的地基——日志、缓存、消息，所有跨机器的「时间点」都应该用 `Instant`/`long` 时间戳表达，绝不存「本地时间字符串」。它与第三级《分布式系统》里「逻辑时钟与物理时钟」的讨论直接接轨。</span>

**格式化**用 `DateTimeFormatter`：

```java
DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");
String s = beijing.format(fmt);                 // 2026-08-07 14:30
LocalDateTime parsed = LocalDateTime.parse(s, fmt);   // 解析回去
```

`yyyy` 是年份、`MM` 月份、`dd` 日、`HH` 24 小时、`mm` 分钟。`DateTimeFormatter` 是**线程安全**的（老 `SimpleDateFormat` 不是！多线程共享会出错）。

## 3 国际化：Locale、ResourceBundle 与格式化

**国际化（internationalization，i18n）**是让程序「适配不同语言与地区」，而不是硬编码一种。核心概念是 **`Locale`**——「语言 + 地区」的标识：

```java
Locale zhCN = Locale.SIMPLIFIED_CHINESE;   // zh_CN：简体中文（中国大陆）
Locale enUS = Locale.US;                    // en_US：英语（美国）
Locale current = Locale.getDefault();       // 当前 JVM 默认地区
```

**`ResourceBundle`（资源包）**按 Locale 装载各语言文案——文件命名带语言后缀：

```
messages.properties          # 默认（英文兜底）
messages_zh_CN.properties    # 简体中文
```

```properties
# messages_zh_CN.properties
greeting=你好，{0}！
# messages.properties
greeting=Hello, {0}!
```

```java
ResourceBundle bundle = ResourceBundle.getBundle("messages", userLocale);
String msg = MessageFormat.format(bundle.getString("greeting"), "张三");
// zh_CN 用户 → "你好，张三！"，en_US 用户 → "Hello, 张三！"
```

**加载顺序**：`ResourceBundle.getBundle` 先找最匹配的 locale（`messages_zh_CN`），没有就退化到基名（`messages`）——所以**永远提供默认资源包兜底**。文案中的占位符用 `MessageFormat` 的 `{0}`、`{1}` 替换，避免字符串拼接造成的翻译顺序问题。

**数字与日期的本地化格式**：同样一个数，不同地区写法不同。用 `NumberFormat` 与 `DateTimeFormatter` 的 `withLocale`：

```java
NumberFormat nf = NumberFormat.getCurrencyInstance(Locale.US);
nf.format(1234.5);     // "$1,234.50"
NumberFormat nfCN = NumberFormat.getCurrencyInstance(Locale.CHINA);
nfCN.format(1234.5);   // "¥1,234.50"

String dateStr = DateTimeFormatter.ofPattern("yyyy年M月d日", Locale.CHINA)
        .format(LocalDate.of(2026, 8, 7));    // 2026年8月7日
```

**辨析｜易错点：把「语言」与「格式」混为一谈。** `Locale` 同时影响**语言**（用哪套文案）与**格式**（日期/数字/货币怎么写）。用户可能母语是中文却住在日本（`zh_JP`）——文案用中文、货币却该显示日元。因此有些框架区分「显示语言」与「显示地区」，这是 i18n 的进阶话题；初学先把「`Locale` 决定一切本地化行为」这个整体模型建立起来。

## 4 核心对比表：旧 API 与新 API 的换代

把历史包袱与新一代并排，看为何必须迁移：

| 维度 | 老 `Date`/`Calendar` | 新 `java.time` |
| --- | --- | --- |
| 可变性 | 可变（线程不安全） | **不可变**（线程安全） |
| 月份 | 从 0 开始（1 月是 0） | 从 1 开始 |
| 时区 | 藏在默认时区里，混乱 | 类型显式区分 |
| 格式化 | `SimpleDateFormat` 线程不安全 | `DateTimeFormatter` 线程安全 |
| 语义 | 一个类包办一切 | 类型细分（日期/时间/时刻/区间） |

**重点结论：新代码一律用 `java.time`，别再用 `Date`/`Calendar`/`SimpleDateFormat`。** 老 API 的缺陷（可变、线程不安全、月份 0 基）在并发与大型系统里是隐藏炸弹。与老代码交接时，用转换方法：`Date.from(instant)`、`instant.toEpochMilli()`、`LocalDate.ofInstant(instant, zone)`——把老 `Date` 挡在边界，内部一律新 API。

## 5 小结

- `java.time` 类型细分：**`LocalDate` 日期、`LocalTime` 时间、`Instant` 绝对时刻、`ZonedDateTime` 时刻+时区**；全部不可变。
- **跨时区场景用 `Instant`/`ZonedDateTime`**，别用 `LocalDateTime` 存绝对时间；`withZoneSameInstant` 换时区显示。
- `Instant` = Unix 时间戳的类型化封装（秒+纳秒），全局统一、与时区无关。
- **i18n 三件套**：`Locale` 描述用户地区、`ResourceBundle` 装各语言文案、`NumberFormat`/`DateTimeFormatter.withLocale` 本地化格式。
- 老 `Date`/`Calendar`/`SimpleDateFormat` 有可变、线程不安全、0 基月份等缺陷，**新代码一律 `java.time`**。

到这里，从「Java 语言概述」到「日期时间 API」，本专题的 27 篇博文就全部完成了。回头看看这条路线：语言基础 → 面向对象 → 编码规范 → 并发与 IO——你已经把 Java 从「会写语法」推进到「能设计可靠后端系统」。下一站，你可以沿着第三级《计算机基础》继续深入分布式、数据库或云计算，Java 会是你在那些领域的得力母语。
