---
title: 方法设计：参数校验、重载与返回值
date: 2026-08-07
---

# 方法设计：参数校验、重载与返回值

<div class="epigraph">
<p>一个方法签名的好坏，决定了它被调用时是「顺手」还是「处处提防」。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Effective Java》第8章 ｜ 2026-08-07</p>
</div>

## 为什么从方法设计开始

类与接口是「骨架」，**方法**是骨架上真正干活的肌肉。一个设计糟糕的方法，调用方要么提心吊胆地处理各种边界，要么在「传错参数」时收到一条莫名的 `NullPointerException`。Effective Java 第 8 章（第 49–56 条）系统讲了方法设计的八条规范：参数校验、防御性拷贝、方法签名、重载与可变参数、返回值与 `null`。<span class="marginnote">方法签名是「调用契约」：参数要求什么、返回承诺什么、什么情况下抛什么异常。写得清楚，调用方读 javadoc 就能放心用；写得含糊，调用方只能在踩坑后猜测。这一章的目标，是让你把「预期」与「意外」在方法边界上划清。</span>这一篇把最核心的四条讲透：参数校验、防御性拷贝、重载设计、返回值的 `null` 纪律。

## 1 参数校验：尽早失败，讲清条件

Effective Java 第 49 条：**检查参数的有效性。** 一个方法收到非法参数（负数、null、越界下标），应当**尽早抛异常**，而不是带着坏数据跑一段再神秘地挂掉。

**「尽早失败（fail fast）」的价值**：非法参数被越早发现，错误离源头越近，排障越容易。如果方法里跑了一大段才在深处撞出 `NullPointerException`，你很难判断是哪一步传错了。

```java
public void deposit(double amount) {
    if (amount <= 0) {
        throw new IllegalArgumentException("金额必须为正：" + amount);
    }
    // 正常逻辑……
}
```

**校验的位置**：**公开方法必须校验**（调用方不可控）；`private` 方法可以宽松（只有你自己调，你自己守规矩）。

**常用校验手段：**

| 手段 | 用途 |
| --- | --- |
| `Objects.requireNonNull(obj, "obj 不能为 null")` | null 校验，带消息 |
| `if (x < 0) throw new IllegalArgumentException(...)` | 范围校验 |
| `if (index < 0 \|\| index >= size) throw new IndexOutOfBoundsException(...)` | 下标校验 |
| 构造器里的校验 | 把不变量挡在对象诞生之前 |

**辨析｜易错点：校验 vs 防御性编程不要混淆。** 参数校验是「拒绝坏输入」；防御性拷贝是「防止好输入被外部篡改」。前者管「调用方传错了」，后者管「调用方传的是对的，但它会事后改掉」——两者配合才完整，见下节。

## 2 防御性拷贝：别让外部篡改你的内部

Effective Java 第 50 条：**必要时进行防御性拷贝（defensive copy）**。当你的方法接收一个**可变对象**作为参数，或返回一个**可变对象**时，调用方可能在不该碰的地方动手脚：

```java
// 反例：日期被外部篡改
public final class Period {
    private final Date start;
    public Period(Date start) { this.start = start; }   // 危险：外部持有同一个 Date 引用
}
Date d = new Date();
Period p = new Period(d);
d.setYear(70);       // 外部改了 Date，Period 的内部「只读」失效了！
```

**正例：构造时与返回时都做防御性拷贝**：

```java
public Period(Date start) {
    this.start = new Date(start.getTime());   // 拷贝：外部怎么改都影响不到内部
}
public Date getStart() {
    return new Date(start.getTime());          // 返回拷贝：内部也不被外部改到
}
```

**重点结论：当「可变对象穿过方法边界」时，问一句「外部会不会改它」。** 用 `java.time` 的不可变 `LocalDate` 就没有这个问题——所以新代码根本不用 `Date`。**防御性拷贝是「与不可变并用」的：对象本身不可变，就无需拷贝；对象可变，出入边界都要拷贝。**<span class="marginnote">防御性拷贝不是「多此一举」：Effective Java 引用的真实案例里，某公司的 `Period` 类因为没做拷贝，外部改 `Date` 后「历史记录」被篡改，酿成数据事故。看似低效的一次 `new Date(...)`，其实是把「内部状态」的安全握回自己手里。</span>

**拷贝的成本与界限**：拷贝有性能开销，且 `clone()` 对子类不可靠，一般用「拷贝构造器」或静态工厂。Effective Java 的原则：**信任前提不存在时做防御性拷贝；文档明确说「调用方不可改」的，可以省略。**

## 3 重载与可变参数：签名设计的两个坑

**重载（overload）**：同名方法、不同参数列表。重载最著名的坑是**「装箱 + 重载」的歧义**：

```java
public void f(int x) { ... }
public void f(Object o) { ... }
f(null);       // 编译错误：null 既能装箱成 Integer 匹配 Object，又……其实是两个都能匹配，编译器报歧义
```

Effective Java 第 52 条：「**谨慎使用重载**」——两个重载参数类型「都能被对方转型匹配」时，编译器选的是最具体的那个，规则隐蔽。规避办法：**给不同行为起不同名字**（`writeInt` vs `writeObject`），而不是靠重载区分语义。

**可变参数（varargs）**：`void sum(int... nums)` 接受任意个数参数，编译器自动包装成数组。两个注意点：

- 可变参数在**性能敏感**路径上有数组分配开销——被高频调用时，考虑提供「固定数量参数」的重载兜底（JDK 里 `EnumSet` 就是这么做的）。
- **可变参数必须是参数列表的最后一个**。

**公式解析：重载决议的「最具体」规则**

当实参能匹配多个重载时，编译器按「最具体参数类型优先」决议。这个选择规则可以形式化：

$$
\text{可选重载} = \{ m_i \mid \text{实参兼容 } m_i \} \qquad
\text{选中} = \arg\max_{m_i}\big(\text{参数类型的「具体度」}\big)
$$

「具体度」即参数类型的子类型关系：`int` 比 `long` 具体，`Integer` 比 `Object` 具体，`int` 与 `Object` 不可比时——若实参是 `null`，就出现歧义、编译报错。**与其背规则，不如设计时避免「int 与 Integer 并存」这类暧昧重载**：语义不同就改名，是 Effective Java 第 52 条的忠告。

## 4 返回值的 null 纪律

Effective Java 第 54 条：**返回空集合/空数组，别返回 null。** 一个返回 `List` 的方法找不到结果时，返回 `null` 还是空集合？

```java
// 反例：返回 null，调用方忘记判空就 NPE
List<Item> items = findItems(keyword);
for (Item i : items) { ... }      // 若 items 是 null，这里 NPE

// 正例：返回空集合
List<Item> items = findItems(keyword);    // 找不到就返回 Collections.emptyList()
```

**重点结论：返回 `null` 是给调用方埋雷。** 空集合语义清晰、for-each 直接可遍历、还可用 `Collections.emptyList()` 共享同一个空实例（零开销）。返回 `null` 唯一的正当理由是「null 有明确业务含义」——但那样更要写在 javadoc 里。

**`Optional<T>` 的合理使用**：Java 8 的 `Optional` 表达「可能缺失」的返回值——`stream.findFirst()` 返回 `Optional`，调用方被迫面对「可能没有」：

```java
Optional<Employee> boss = findBoss();
boss.ifPresent(b -> System.out.println(b.getName()));
```

**辨析｜易错点：`Optional` 不是万能的。** Effective Java 第 55 条建议：**`Optional` 适合「可能缺失的返回值」，但不要**——用在字段上（类膨胀、不可序列化）、用在方法参数上（应当重载）、用在「始终有值」的集合上（空集合就是答案）。`Optional` 是「返回值可能缺失」的信号，别滥用在别处。

**从 Optional 取值**：`orElse(默认值)`、`orElseThrow(异常提供者)`、`orElseGet(惰性提供者)`——优先 `orElseGet` 避免默认值也参与构建。

## 5 核心对比表：方法设计四律

纯概念主题用**核心对比表**替代公式解析的展开：

| 纪律 | 原则 | 反例 |
| --- | --- | --- |
| 参数校验 | 尽早失败、讲清条件 | 带着坏参数跑一段才炸 |
| 防御性拷贝 | 可变对象过边界要拷贝 | 外部 `Date` 被改，内部「只读」失效 |
| 重载设计 | 语义不同就改名 | `f(null)` 的装箱歧义 |
| 返回值 | 空集合而非 null；可选值用 `Optional` | 忘记判空导致的 NPE |

**重点结论：方法边界的质量 = 调用方的安心程度。** 把「非法输入」挡在方法入口（校验）、把「内部状态」护在方法内部（拷贝）、把「没有结果」表达清楚（空集合/`Optional`）——这三件事做扎实，你的方法就不需要调用方提防。这也是 Effective Java 第 8 章的全部要义：**让方法容易用对、难以用错**。

## 6 小结

- **公开方法必须校验参数**，用标准异常尽早失败、带清晰消息。
- 可变对象穿过方法边界要**防御性拷贝**；不可变对象免拷贝。
- 重载警惕「装箱 + null」歧义；语义不同就**改名**，别靠重载。
- **返回空集合而非 null**；可能缺失的返回值用 `Optional`（别滥用在字段/参数上）。
- 方法设计的目标：让调用方「容易用对、难以用错」。

在下一节，我们回到「日常编码」的细节——**通用编程规范：局部变量、数值与循环**。
