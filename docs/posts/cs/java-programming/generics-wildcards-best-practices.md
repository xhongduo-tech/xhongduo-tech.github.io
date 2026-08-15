---
title: 泛型的正确使用与通配符边界
date: 2026-08-07
---

# 泛型的正确使用与通配符边界

<div class="epigraph">
<p>泛型的全部优雅，藏在「生产者用 extends，消费者用 super」这句口诀里。</p>
<footer>—— 改编自 Joshua Bloch《Effective Java》第5章</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Effective Java》第5章 ｜ 2026-08-07</p>
</div>

## 为什么从泛型通配符开始

上一章学了泛型的基本语法：类型参数 `<T>`、类型擦除、有界类型参数。但一写真实代码你就会撞上一堵墙：`List<Number>` 与 `List<Integer>` 之间**没有父子关系**——虽然 `Integer` 是 `Number` 的子类，`List<Integer>` 却既不是 `List<Number>` 的子类型也不是父类型。这堵墙是类型安全必须的（否则往 `List<Number>` 里塞 `Double` 就能污染你的 `List<Integer>`），但它也让「通用方法」写起来处处受限。**通配符（wildcard）**是突破这堵墙的钥匙：它表达「某个未知但受约束的类型」。这一篇把通配符、PECS 口诀与泛型方法的使用边界一次讲透。

## 1 不要使用裸类型（raw type）

**裸类型（raw type）**是不带类型参数的泛型——`List`、`Map`，而不是 `List<String>`、`Map<K,V>`。它存在于 JDK 5 之前的旧代码里，现在只应出现在「与旧代码互操作」的边缘。

```java
// 反例：裸类型
List myList = new ArrayList();
myList.add("hello");
myList.add(42);          // 编译期没人拦
String s = (String) myList.get(1);   // 运行期 ClassCastException！

// 正例：参数化类型
List<String> myList = new ArrayList<>();
myList.add(42);          // 编译错误，当场拦住
```

**重点结论：裸类型放弃了泛型的全部安全收益。** 裸 `List` 的每个元素都是 `Object`，取出必须强转，强转错类型到运行期才炸——这正是泛型要消灭的东西。编译器甚至会对裸类型的使用给出警告（unchecked），因为 JVM 里它们与参数化类型擦除后是同一个类，无法区分。

**辨析｜易错点：`List`（裸）与 `List<?>`（通配符）不是一回事。** 裸 `List` 可以往里 `add` 任何东西；`List<?>` 是「元素类型未知」的只读视图，不能 `add`（除了 null）。**「未知」≠「任意」**：`List<?>` 是「我知道它是某一种类型的 List，但不知道具体哪种」，所以往里写是不安全的。

## 2 无界通配符：List`<?>`

**无界通配符 `<?>`** 表示「某种未知类型」。它解决的问题是：**只读遍历一个类型未知的集合**。

```java
public static void printAll(List<?> list) {
    for (Object o : list) {   // 读：安全，任何类型都能赋给 Object
        System.out.println(o);
    }
    // list.add("hi");        // 写：编译错误，类型未知不能写入
}
```

为什么不能写？如果 `printAll` 收到一个 `List<Integer>`，`add("hi")` 会往里面塞 `String`，运行时类型系统就崩了。编译器宁可禁止一切写入来保证安全——**除了 `null`**，因为 null 是任何类型都能接受的。<span class="marginnote">`List<?>` 的成员都只能读成 `Object`。它比裸 `List` 安全的地方正在于此：裸 `List` 允许乱写（埋雷），`List<?>` 拒绝写入（拆雷）。</span>

**什么时候用 `<?>`**：方法只需要「读一个通用集合」时。它等价于 `<? extends Object>`，是通配符的底线形态。

## 3 公式解析：PECS——生产者 extends，消费者 super

上界与下界通配符的选用规则，就是著名的 **PECS（Producer Extends, Consumer Super）**：方法从参数里**读**东西，参数是**生产者**，用 `extends`；方法向参数里**写**东西，参数是**消费者**，用 `super`。

$$ 读（产出 T）\Rightarrow \text{? extends T} \qquad \text{写（消费 T）} \Rightarrow \text{? super T} $$

对这条公式做三步拆解：

- **第一步，判断方向**：方法是对参数「读」（取出元素用）还是「写」（往里面放元素）？
- **第二步，读 → 生产者 → `? extends T`**：方法要从集合里**读**出 `T`（集合是 `T` 的**生产者**），用上界通配符——这样 `List<Integer>` 也能传给「读 `List<? extends Number>`」的方法。
- **第三步，写 → 消费者 → `? super T`**：方法要向集合里**放** `T`（集合是 `T` 的**消费者**），用下界通配符——这样 `List<Object>` 也能接受「写 `List<? super Integer>`」的元素。

看两个应用，PECS 才有血肉：

```java
// 生产者：从 src 里读（复制出来）
public static <T> void copy(List<? extends T> src, List<? super T> dst) {
    for (T item : src) dst.add(item);   // 从 extends 读，往 super 写
}
```

`copy(List<Integer>, List<Number>)` 能编译——`List<Integer>` 是 `List<? extends Number>`（生产者），`List<Number>` 是 `List<? super Integer>`（消费者）。**没有通配符，这段通用代码根本写不出来**——这正是 `Collections.copy` 等 JDK 方法背后的签名设计。

**重点结论：PECS 不是口号，而是「读写安全」的推导。** `? extends T` 只保证「能读成 T」——往里写不安全（你不知具体类型）；`? super T` 只保证「能放进 T」——往里读只能读成 `Object`。**用 PECS 判断，比死记规则可靠**：每次写泛型签名，先问「我读它还是写它」。<span class="marginnote">PECS 的直觉：`extends` 让「子类都能放进集合，于是里面可能是任意子类，只能按父类读」；`super` 让「父类都能容纳 T，于是往里写 T 一定安全」。读方向朝上、写方向朝下——两个方向各锁一边，安全就成立了。</span>

## 4 泛型方法与通配符：优先泛型方法

Effective Java 第 30 条与第 31 条给出两条选型原则：

**第一，泛型方法优先于「无界通配符 + 强转」**。裸 `Object` 参数的方法（内部强转）会「静默吞掉类型信息」，应在编译期就由泛型方法守住：

```java
public static <E> Set<E> union(Set<? extends E> s1, Set<? extends E> s2);
```

**第二，若类型参数在方法体内出现多次，考虑用泛型方法而非通配符**。通配符适合「类型只在签名出现一次」；类型参数需要「同名约束」时，泛型方法更清晰。

**辨析｜易错点：`List<?>` 与 `List<Object>` 不是一回事。** `List<Object>` 是「能装任何类型的列表」（往里面 add 什么都行）；`List<?>` 是「类型未知的列表」（不能 add 非 null）。`List<String>` 可以传给 `List<?>`，**但不能**传给 `List<Object>`（那会允许往里面塞 `Integer` 污染）。这个区别，是理解「不变性（invariance）」的关键——`List<String>` 与 `List<Object>` 无父子关系，是泛型安全的第一道闸门。

## 5 核心对比表：三种通配符

纯概念主题用**核心对比表**替代公式解析的展开，把三种通配符一次分清：

| 维度 | `? extends T`（上界） | `?`（无界） | `? super T`（下界） |
| --- | --- | --- | --- |
| 含义 | T 或 T 的任意子类 | 某种未知类型 | T 或 T 的任意父类 |
| 能读 | 能读成 T | 只能读成 Object | 只能读成 Object |
| 能写 | **不能** | **不能**（除 null） | **能写 T** |
| 角色 | 生产者（读） | 只读视图 | 消费者（写） |
| 典型 | `copy(src)` 的源 | `printAll` 的参数 | `copy(dst)` 的目标 |

**再补一个高级细节——通配符捕获（wildcard capture）**。当你想把 `List<?>` 里的元素「拿到一个泛型方法里去处理」时，编译器不允许直接传 `List<?>` 给 `List<E>` 形参——这时可以用一个辅助泛型方法来「捕获」未知类型：

```java
void printReverse(List<?> list) {
    revHelper(list);          // 编译器把 ? 捕获成某个具体类型 W
}
private static <W> void revHelper(List<W> list) {   // 通配符捕获
    // 现在 W 是「已知的未知」，可以在方法体内安全使用
}
```

**重点结论：通配符读多写少，写多要靠 `super`，完全未知则只读。** 记住「读 extends、写 super、纯读无界」的三角——写代码时先判断方法对参数的读写方向，再选通配符。这套规则在 `Collections` 的每个泛型方法签名里都能看到，是 JDK 自身也在遵守的纪律。

## 6 小结

- **别用裸类型**：`List` 放弃全部类型安全；新代码一律参数化。
- **`List<?>`** 是「类型未知」的只读视图，能读不能写（除 null）。
- **PECS**：读用 `? extends T`（生产者），写用 `? super T`（消费者）。
- **`List<?>` ≠ `List<Object>`**；泛型容器是不变的。
- 优先**泛型方法**；通配符用于「类型只在签名出现一次」的场合。

在下一节，我们把泛型与 lambda 结合成更强大的数据处理——**Lambda 与 Stream 流式编程**。