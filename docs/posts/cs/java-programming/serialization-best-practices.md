---
title: 序列化的最佳实践
date: 2026-08-07
---

# 序列化的最佳实践

<div class="epigraph">
<p>序列化让对象能离开内存、穿过网络、越过时间——但每个被序列化的类都背负一份永久的兼容性债务。</p>
<footer>—— 改编自 Joshua Bloch《Effective Java》第12章</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Effective Java》第12章 ｜ 2026-08-07</p>
</div>

## 为什么从序列化开始

上一章的对象都活在内存里，进程一结束就烟消云散。**序列化（serialization）**让对象「走出去」：把对象的内存状态转换成**字节序列**，存盘、传网络、进缓存；需要时再**反序列化**还原成对象。`Session` 存储、分布式消息、RPC 调用都靠它。但 Effective Java 第 12 章反复警告：**序列化是「易用而难精」**——默认序列化会把内部结构暴露成契约，改字段就破坏兼容；反序列化还能被攻击者利用构造恶意对象（历史上著名的序列化漏洞）。这一篇讲清序列化的机制、`serialVersionUID`、以及三条避坑原则。

## 1 序列化机制：Serializable 与流

Java 的序列化由两个内建类驱动：

```java
ObjectOutputStream out = ...;
out.writeObject(obj);          // 序列化：对象 → 字节
ObjectInputStream in = ...;
Object obj = in.readObject();  // 反序列化：字节 → 对象
```

对象要能序列化，必须实现 **`Serializable`** 接口——它是一个**标记接口**（没有方法），实现它等于向 JVM 声明「这个类可以被序列化」：

```java
public class Employee implements Serializable {
    private static final long serialVersionUID = 1L;   // 版本号，见下节
    private String name;
    private double salary;
}
```

**序列化写什么**：默认序列化把对象的所有**非 transient 实例字段**递归写出——字段是对象就继续序列化那个对象，形成**对象图**。不想序列化的字段（如缓存、密码）标 `transient`，反序列化时它们恢复为默认值（0、null、false）。

**重点结论：默认序列化是「逐字段复制」**，它朴素、易用，但有两个隐患——**把实现细节固化成了对外契约**（字段名、字段类型都不能随便改），以及**反序列化可以绕过构造器**（不调任何构造器/校验，直接用字节重建对象）。

**辨析｜易错点：序列化与 equals 无关，却与单例相关。** 反序列化**不调用构造器**，用 `ObjectInputStream` 直接重建对象——所以序列化一个单例类，反序列化会得到**第二个实例**，单例被打破。解法是单例类里加 `readResolve()`：反序列化后返回既有的单例实例。

## 2 serialVersionUID：版本兼容的身份证

**`serialVersionUID`** 是序列化版本的标识。序列化时写入，反序列化时比对：

- **相同**：反序列化继续。
- **不同**：抛 `InvalidClassException`——两个版本「不兼容」。

```java
private static final long serialVersionUID = 1L;
```

**为什么必须显式声明它？** 不声明时 JVM 会根据类名、接口、字段、方法自动算一个哈希值当 UID。问题是：**自动算的 UID 对「无关紧要的改动」也敏感**——你只是加了个注释、改了个方法名，UID 就变了，旧数据反序列化直接失败。**显式固定 `serialVersionUID`，把版本兼容的决定权握在自己手里**：加个字段？只要 UID 不变，旧数据反序列化后新字段用默认值，兼容；确实不兼容的改动，才主动改 UID 让旧数据「优雅失败」。<span class="marginnote">「UID 不变 = 兼容」是 Java 序列化兼容性的核心约定。改动时按「加字段兼容、删字段不兼容、改类型不兼容」的直觉判断，配合 UID 控制——这与你数据库加列的迁移思路同构：向后兼容要留默认值。</span>

**辨析｜易错点：反序列化是「不受信任的输入」。** 字节流可以被人为构造——攻击者精心构造一个字节流，反序列化时可能触发危险代码（反序列化漏洞）。**永远不要反序列化不受信任的数据**；对受信任数据也要在白名单校验 `readObject` 读出的对象类型。

## 3 谨慎实现 Serializable，首选自定义序列化形式

Effective Java 第 86 条「谨慎实现 Serializable」：**不要轻易让一个类实现 `Serializable`**。它的代价是长久的：

- **兼容性枷锁**：实现后，类的字段布局被固化，**改字段就破坏兼容性**（除非有 UID 与自定义形式兜底）。
- **安全暴露**：默认序列化把**私有字段**也写出去——内部表示对外泄露。
- **测试负担**：每个版本都要验证「旧数据还能读」。

**何时才实现 Serializable**：类的主要用途就是「跨越进程/时间传输」（DTO、事件、缓存值），或框架明确要求（某些分布式组件）。**否则别为「也许以后要用」而提前实现**——一旦发布，就是永久的承诺。

**自定义序列化形式**（Effective Java 第 87 条）：当类的**逻辑状态**与**物理布局**不一致时（比如内部用 `Map` 存数据、逻辑上是个集合），默认逐字段序列化是错的——它把内部 `HashMap` 的实现细节暴露了。正确做法是实现 `writeObject`/`readObject` 自定义「写什么、怎么读」：

```java
private void writeObject(ObjectOutputStream s) throws IOException {
    s.defaultWriteObject();          // 先写非 transient 字段
    s.writeInt(size);                // 再按逻辑状态写
    for (Entry e : entries) s.writeObject(e.getKey());
}
```

**重点结论：序列化形式应该反映「逻辑数据」，而不是「内存布局」。** 默认形式好用，但只在你「字段即逻辑」时正确；两者不一致，就要自定义。好消息是——**现代实践大多用 JSON 等文本格式替代 Java 原生序列化**，DTO 直接 `Jackson`/`Gson` 转 JSON，字段名就是协议，兼容性管理更直观。<span class="marginnote">为什么 JSON 更安全？反序列化漏洞的前提是「字节流能被构造来触发代码执行」，而 JSON 只有「数据结构 + 值」，没有对象引用图，攻击面小得多。这也是 Effective Java 第 85 条「优先选择替代序列化技术」的理由。</span>

## 4 公式解析：readResolve 与单例的守卫

单例类若坚持可序列化，必须用 **`readResolve()`** 守住「只有一个实例」的不变量：

$$

\text{readObject()} \to \text{readResolve()} \to \text{返回规范实例}

$$

对这条公式做三步拆解：

- **第一步，反序列化发生**：`readObject()` 从字节流重建对象——注意它**不调用构造器**，直接按字节填字段。
- **第二步，拦截点触发**：`readObject()` 返回前会检查对象是否有 `readResolve()` 方法；有则调用它，把返回值当作「真正的反序列化结果」。
- **第三步，返回规范实例**：`readResolve()` 返回那个**已存在的单例**，临时重建的对象被丢弃——于是「反序列化出第二个实例」被彻底堵死。

```java
public class Singleton {
    private static final Singleton INSTANCE = new Singleton();
    private Singleton() { }
    public static Singleton getInstance() { return INSTANCE; }

    private Object readResolve() {        // 反序列化后强制返回既有单例
        return INSTANCE;
    }
}
```

**重点结论：`readResolve()` 是「反序列化绕过构造器」的补丁。** 它不阻止「重建对象」这个动作，而是让重建结果**被替换**——调用方拿到的永远是那个规范的单例。同理，任何「反序列化后必须校验」的类（`readObject` 里检查合法性），都是在这一层补上「构造器里才有的不变量检查」。

**更彻底的防御：避免原生序列化。** 现代实践里，跨进程传输多用 **JSON/Protobuf** 而非 Java 原生序列化——DTO 直接 `Jackson` 序列化成 JSON，字段名就是协议、兼容性由 JSON schema 管理、也不存在「字节流被构造器绕过」的原生问题。**原生 `Serializable` 的适用面越来越窄**：Java 生态内的缓存、RMI 等特定场景才需要它。

## 5 小结

- 序列化 = 对象 → 字节流，反序列化 = 字节流 → 对象；对象必须实现 `Serializable`（标记接口）。
- 默认序列化**逐字段复制**，`transient` 字段跳过；它把**内部结构固化成了契约**。
- **显式声明 `serialVersionUID`**：`UID 不变 = 兼容`，加字段兼容、删字段/改类型不兼容。
- **永远别反序列化不受信任的数据**——字节流可被人为构造触发危险代码。
- **谨慎实现 `Serializable`**；逻辑状态 ≠ 物理布局时自定义 `writeObject`/`readObject`；单例加 `readResolve()`。

在下一节，我们把程序送进网络——**网络编程与套接字**。