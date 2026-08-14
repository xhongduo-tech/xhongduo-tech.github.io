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