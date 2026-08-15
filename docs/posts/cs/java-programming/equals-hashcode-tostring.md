---
title: 覆盖 equals、hashCode 与 toString
date: 2026-08-07
---

# 覆盖 equals、hashCode 与 toString

<div class="epigraph">
<p>每个类都该重写 toString；重写 equals 就必须重写 hashCode——这是一条刻在 Java 血液里的契约。</p>
<footer>—— 改编自 Joshua Bloch《Effective Java》第3章</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Effective Java》第3章 ｜ 2026-08-07</p>
</div>

## 为什么从相等性契约开始

`Object` 提供了三个「每个类都在用」的方法，默认实现却几乎总是错的：`equals` 默认按**引用**比较——两个内容相同的员工对象，`equals` 返回 `false`；`hashCode` 默认按**内存地址**散列——内容相同的对象 hash 不同；`toString` 默认打印 `类名@地址`——人根本读不懂。数据库里有主键、数学里有相等公理，Java 的对象相等性也有一套**必须遵守的契约**：违反它，`HashSet` 会漏判重复、`HashMap` 会找不到键、调试日志全是乱码。这一篇把三条契约与它们的实现配方讲透。

## 1 equals：相等性的通用约定

`Object.equals` 的默认实现是引用相等（`==`）——「只有同一个对象才算相等」。当你想表达「**内容相等**」（两个员工工号相同即相等）时，就必须重写 `equals`。

`equals` 方法必须满足**五个性质**：

| 性质 | 含义 | 违反的后果 |
| --- | --- | --- |
| **自反性** | `x.equals(x)` 必为 true | 集合查找自己的元素都可能失败 |
| **对称性** | `x.equals(y)` ⟺ `y.equals(x)` | 一方 true 一方 false，行为错乱 |
| **传递性** | `x=y` 且 `y=z` ⇒ `x=z` | 等价类瓦解 |
| **一致性** | 字段未变则结果不变 | 缓存里的结果过期 |
| **非空性** | `x.equals(null)` 必为 false | 空指针风险 |

**重点结论：相等关系必须是数学意义上的等价关系。** 违反任一条，对象放进集合后就表现诡异——`contains` 结果飘忽、`remove` 删不掉。这五个性质不是道德要求，而是 `HashSet`、`HashMap` 等数据结构正确性的前提。

**经典配方**（Effective Java 第 10 条的四步）：

```java
@Override
public boolean equals(Object o) {
    if (this == o) return true;                    // 1. 引用相等，直接返回
    if (o == null || getClass() != o.getClass()) return false;  // 2. 类型检查
    Employee e = (Employee) o;                     // 3. 强转
    return id == e.id                              // 4. 逐字段比较
        && Objects.equals(name, e.name);
}
```

`Objects.equals(name, e.name)` 内部做了空值安全比较（null 与 null 相等）。**字段比较顺序**：先比最可能不同的字段、先比基本类型（`==`）、对象字段用 `Objects.equals`。

**辨析｜易错点：用 `getClass()` 还是 `instanceof`？** `getClass() != o.getClass()` 要求**精确同类**才相等；`instanceof` 则允许子类与父类「相等」。Effective Java 主张前者——否则对称性极难保证（子类新增字段后，`Parent.equals(Child)` 与 `Child.equals(Parent)` 很难同时成立）。**继承体系下设计 equals 极难，如果子类要新增相等性字段，干脆不让它们重写 equals。**

## 2 hashCode：与 equals 锁死的伴侣

`hashCode` 返回一个 32 位整数，供 `HashMap`、`HashSet` 定位桶。它的**约定**是：

$$

\text{若 } a.\text{equals}(b) \text{ 为 true，则 } a.\text{hashCode}() = b.\text{hashCode}()

$$

反过来不成立——两个 hash 相同的对象可以 `equals` 为 false（哈希碰撞是合法的）。**关键性质只有一条方向**：equals 相等 ⟹ hashCode 相等。违反它，两个「相等」的对象被放进不同的桶，`HashSet` 里就会出现两个重复元素。

**计算公式**（Effective Java 第 11 条的 31 因子配方）：

$$

\text{result} = 31 \times \text{result} + c \quad (\text{逐个字段累积})

$$

对这条公式做三步拆解：

- **第一步，初始化**：`int result = 1;` 作为累积起点。
- **第二步，逐个字段累乘加和**：对每个参与 `equals` 的字段 `f`，先算它的散列码 `c`（基本类型用 `Integer.hashCode(f)` 等，对象用 `f.hashCode()`，null 用 0），再 `result = 31 * result + c`。
- **第三步，返回**：`return result;`。

**为什么是 31？** 它是个奇素数，且 `31 * i == (i << 5) - i`——JVM 能把乘法优化成一次移位加一次减法，又快又能把不同字段的取值「搅匀」，减少碰撞。**要点不是「非 31 不可」，而是「`equals` 相等的字段必须全部参与散列」**——漏掉任何字段，`equals` 相等而 `hashCode` 不等的违约就会出现。

**一个常用捷径**：`Objects.hash(f1, f2, f3)` 内部按同样的配方累积，一行搞定（代价是多一次装箱分配，非性能热点时可用）：<span class="marginnote">`Objects.hash` 内部用的是 `Arrays.hashCode(Object[])`——它把每个参数装箱后按 31 因子累积。简洁的代价是分配一个 `Object[]`，非热点路径上完全可接受；性能敏感时回到手写配方。</span>

```java
@Override
public int hashCode() { return Objects.hash(id, name); }
```

## 3 toString：每个类都该有的自述

**`toString`** 的默认实现是 `类名@十六进制哈希`（如 `Employee@1b6d3586`）——对调试毫无帮助。Effective Java 第 12 条：**始终重写 `toString`**，让它返回「对象的可读自述」。

```java
@Override
public String toString() {
    return "Employee{id=" + id + ", name='" + name + "'}";
}
```

**为什么要重写**：`System.out.println(obj)`、日志、调试器、字符串拼接（`"员工是：" + e`）都会调用 `toString`——不重写，你看到的就是一堆地址。**好的 `toString` 应包含对象的全部关键信息**，让看日志的人「不用再翻代码」就明白对象状态。

**解析器友好格式**：`toString` 的格式最好能**反过来构造对象**——`Employee{id=1, name='Alice'}` 这种自描述格式，既人眼可读，也方便日后写工具解析。Effective Java 还建议在 javadoc 里**文档化你承诺的格式**（或明确说「格式可能变化」），避免调用方把 `toString` 输出当协议来解析。<span class="marginnote">现代 IDE（IntelliJ IDEA）能一键生成 `equals`/`hashCode`/`toString`，生成的模板已符合 Effective Java 的配方。自己写仍值得掌握——遇到「生成模板与领域语义冲突」时，你知道要改哪里、为什么改。</span>

**辨析｜易错点：`toString` 不要抛异常、不要有副作用。** 它可能在任何地方被调用（日志框架、调试器、异常消息），抛异常会掩盖真正的错误。也别在 `toString` 里做重计算——它可能被频繁调用。

## 4 核心对比表：Object 三方法契约

纯概念主题用**核心对比表**替代公式解析的展开：

| 方法 | 契约 | 不重写的后果 |
| --- | --- | --- |
| `equals` | 等价关系（5 性质） | `HashSet` 漏判重复、`contains` 失败 |
| `hashCode` | `equals` 相等 ⟹ hash 相等 | 相等对象进不同桶，集合出现「重复元素」 |
| `toString` | 可读的自描述 | 日志全是 `类名@地址`，排障困难 |

**重点结论：三条契约是一套「数据驱动」的地基。** `equals` 定义「何为相同」，`hashCode` 让散列集合能快速定位，`toString` 让对象可观察。**重写 `equals` 必重写 `hashCode`**（否则散列集合崩），`toString` 则独立存在、值得每个类都写。这三件小事做好了，你的对象才真正「可比较、可散列、可观测」——这也是进入集合框架前最后一层地基。

## 5 小结

- `equals` 五性质：**自反、对称、传递、一致、非空**；经典配方四步，用 `getClass()` 保对称。
- `hashCode` 与 `equals` 锁死：`equals` 相等 ⟹ `hashCode` 相等；31 因子配方逐个字段累积。
- **重写 `equals` 必重写 `hashCode`**，否则 `HashSet`/`HashMap` 行为错乱。
- `toString` 每个类都重写，返回「可读 + 含全部关键信息 + 可解析」的自述。
- 别在 `toString` 里抛异常、别依赖默认 `类名@地址`。

在下一节，我们把「单个对象」的纪律升级到「整个类」——**类和接口的设计规范**。