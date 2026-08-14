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