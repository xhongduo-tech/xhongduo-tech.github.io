---
title: 异常的正确使用与最佳实践
date: 2026-08-07
---

# 异常的正确使用与最佳实践

<div class="epigraph">
<p>异常处理的要诀，是只在真正异常的情况下使用异常——异常不该是流程控制，也不该被吞掉。</p>
<footer>—— 改编自 Joshua Bloch《Effective Java》第10章</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Effective Java》第10章 ｜ 2026-08-07</p>
</div>

## 为什么从异常最佳实践开始

上一章学了异常的「语法」：`Throwable` 家族、受检/非受检异常、`try-catch`。但「会用」与「用对」之间隔着一整片雷区——Effective Java 第 10 章用整整九条规则（第 69–77 条）讲异常的使用纪律。核心问题是：**异常是什么？** 它不是流程控制，而是「真的出事了」的信号。用错异常，轻则日志刷屏却查不出问题，重则吞掉错误让系统在错误状态里继续运行、数据悄悄损坏。<span class="marginnote">异常在 JVM 里是「昂贵的」：每次抛出都要填栈、构建异常对象。Effective Java 第 69 条提醒：异常应该<strong>稀罕</strong>——正常路径用不到异常，性能也更好。频繁用异常做流程判断，是「拿火箭筒打蚊子」。</span>这一篇把「何时抛、抛什么、怎么吞、怎么补」四条纪律讲透。

## 1 只在真正异常时使用异常

Effective Java 第 69 条：**只对异常情况使用异常。** 反例是最经典的：

```java
// 反例：用异常做循环终止判断
try {
    int i = 0;
    while (true) {
        System.out.println(array[i++]);
    }
} catch (ArrayIndexOutOfBoundsException e) {
    // 假装「数组遍历完了」
}
```

这段「用越界异常当循环结束标志」的代码：编译能过、结果碰巧也对，但 JVM 每轮循环都做数组边界检查，一旦越界就抛异常、填栈——**性能差几十倍**，而且掩盖了「数组真的可能越界」这个本该暴露的 bug。正确的循环用 `for-each`，异常留给「真的意外」。

**重点结论：异常捕获的是「意外」，不是「预期」。** 预期会发生的事（遍历到末尾、查无此人）用正常控制流判断；真正意外的事（磁盘 IO 失败、网络断了）才用异常。写 `try-catch` 前先问：这个「错误」正常会发生吗？会，就别用异常。

**「检查所有异常」的成本**：异常对象要构建栈跟踪、对象可能被多次包装，正常路径上每抛一次都是纯开销。让异常走「异常路径」、让正常路径畅通无阻——这也是性能优化里常被忽视的一课。

## 2 优先使用标准异常

Effective Java 第 72 条：**复用标准异常，别自己发明。** JDK 提供的常见异常足够覆盖绝大多数场景：

| 标准异常 | 适用场景 |
| --- | --- |
| `IllegalArgumentException` | 参数值不合法（如负数传给「必须为正」的参数） |
| `IllegalStateException` | 对象状态不合法（如「未初始化就调用」） |
| `NullPointerException` | 参数/字段意外为 null |
| `IndexOutOfBoundsException` | 下标越界 |
| `ConcurrentModificationException` | 并发修改了不该并发修改的集合 |
| `UnsupportedOperationException` | 对象不支持该操作 |
| `ArithmeticException` | 算术异常（如除零） |

**重点结论：标准异常自解释、人人认识，还免费获得「语义一致性」。** 调用方看到 `IllegalArgumentException` 就知道「传参错了」，看到 `IllegalStateException` 就知道「状态不对」——比自创的 `MyException` 更省沟通成本。自创异常只在「这个错误类型有跨方法共享的语义」时才有必要（如领域层的 `OrderNotFoundException`）。<span class="marginnote">JDK 的 `java.lang` 异常集合是「标准异常词典」：先翻它，找不到合适的才自定义。Effective Java 第 72 条的原话是「优先使用标准异常」——自创异常的成本是维护与文档，收益只有「更具体的类型名」。</span>

**抛异常时把信息写清楚**：

```java
throw new IllegalArgumentException("金额必须为正，收到：" + amount);
```

好的异常消息要能「不看代码就定位问题」——包含出错的输入值、期望的约束。这比「参数错误」有用一个数量级。

## 3 受检异常 vs 非受检异常：抛什么

Effective Java 第 71 条讨论了一个关键决策：**方法的异常是声明为受检（checked）还是非受检（unchecked）？**

- **受检异常（checked）**：编译器强制调用方处理。用于「可以合理恢复的情况」——文件不存在、网络超时、参数非法输入导致的操作失败。调用方被逼着思考「怎么办」。
- **非受检异常（`RuntimeException` 及其子类）**：编译器不强制处理。用于「编程错误」——空指针、越界、非法参数、断言失败。这些是「你写错了」，不该期望调用方优雅恢复。

**辨析｜易错点：把「编程错误」声明成受检异常是负担。** 如果一个异常「调用方无论如何都无法合理恢复」，把它做成受检只会逼出大片空 `catch`。Effective Java 第 71 条的经验：**方法在文档里写清「会抛什么、什么条件下抛」，让调用方决定怎么处理。**

**「编译时强制」是把双刃剑**：受检异常让「忘了处理」不可能发生，却也逼着调用方要么 `catch` 要么 `throws`——处理不了还要向上抛时，就会污染一路的签名。选择原则是：**能恢复 → 受检，是 bug → 非受检。**

## 4 别吞异常：catch 后怎么办

**吞掉异常（swallowing）**是异常处理里最臭名昭著的坏习惯：

```java
// 反例：空 catch 块——异常被无声吞掉
try {
    loadConfig();
} catch (IOException e) {
    // 什么都不做！！错误被静默吞掉
}
```

程序继续往下跑，但配置没加载——错误状态下的程序可能「看起来正常、行为全错」。**空 catch 是隐患放大器**：它把「出错了」的信息彻底抹掉，排障时连线索都没有。

**catch 后的三种正当处理：**

- **恢复**：能处理就处理——重试、用默认值、走备用路径。
- **记录 + 重新抛出**：记日志后把异常（或包装后的异常）继续抛给上层。
- **包装**：用 `throw new BusinessException("加载配置失败", e)` 把底层 `IOException` 包装成上层语义，并把原异常**作为 cause 传进去**——保留栈跟踪，这是 Effective Java 第 75 条「在细节信息中包含失败-捕获信息」的实践。

**公式解析：异常的「因果链」传递**

异常在传递中要保留「原始原因」，否则排查时只见表层、不见根源。包装异常的传递关系：

$$
\text{cause}(新异常) = \text{原异常} \quad \Rightarrow \quad \text{栈跟踪} = \text{新异常栈} \oplus \text{原异常栈}
$$

写 `throw new WrapperException("context", e)` 时把 `e` 作为第二个参数（cause），Java 会把两层栈跟踪都保留在 `printStackTrace()` 里——你既能看见「哪一层处理失败」，又能顺着 cause 找到「最初在哪出错」。**丢了 cause，等于丢了事故现场**。

**finally 与 try-with-resources**：`finally` 保证「无论是否异常都执行」，是释放资源的传统方式；Java 7 的 **try-with-resources** 让实现了 `AutoCloseable` 的资源自动关闭，且**同时发生两个异常时，finally 会覆盖原异常、try-with-resources 会保留原异常并把关闭异常作为 suppressed 记录**——这正是 Effective Java 第 9 条偏爱 try-with-resources 的原因。

## 5 核心对比表：三种异常处理姿势

纯概念主题用**核心对比表**替代公式解析的展开，把「遇到异常做什么」钉死：

| 姿势 | 代码 | 后果 |
| --- | --- | --- |
| 空 `catch`（吞掉） | `catch (E e) {}` | 错误被抹掉，程序在错误态运行 |
| 记录但继续 | `catch (E e) { log(e); }` | 有线索，但调用方不知道失败 |
| 处理并恢复 | `catch (E e) { retry / 默认值 }` | 最好：错误被纠正 |
| 包装后抛出 | `catch (E e) { throw new W(e, e); }` | 错误被上抛，保留 cause |

**重点结论：异常的归宿必须是「被处理」或「被显式上抛」，绝不能是「被遗忘」。** 三条黄金纪律：能用 try-with-resources 就别手写 `finally`；catch 之后要么恢复要么重抛，不许留空；给异常写清楚上下文信息。记住 Effective Java 第 77 条的总纲——**「别忽略异常」**。

## 6 小结

- 只在**真正异常**时用异常；正常流程用普通控制流，别拿异常当分支。
- **优先标准异常**；抛异常时带上具体信息，方便不看代码就定位。
- 能恢复 → 受检异常；编程错误 → 非受检异常（`RuntimeException`）。
- **绝不空 catch**；要么恢复、要么记录并重抛、要么包装（保留 cause）。
- try-with-resources 自动关闭资源，且比 `finally` 更不易丢失原始异常。

在下一节，我们进入集合的高级话题——**泛型程序设计**，让容器与算法脱离具体类型而存在。
