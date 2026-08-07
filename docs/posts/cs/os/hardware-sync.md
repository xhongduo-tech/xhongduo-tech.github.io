---
title: 硬件同步原语：Test-and-Set、Compare-and-Swap 与原子变量
date: 2026-08-07
---

# 硬件同步原语：Test-and-Set、Compare-and-Swap 与原子变量

<div class="epigraph">
<p>当软件无法保证「读-改-写」不可分割时，硬件伸出了手：一条指令，搞定一切。</p>
<footer>—— 佚名，并发编程课堂</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 《操作系统概念》§6.5 与 §6.8 硬件同步 ｜ 2026-08-07</p>
</div>

## 为什么从硬件同步原语开始

Peterson 算法证明了软件互斥可能，但也暴露了软件方案的命门：**「读-改-写」三段式永远可能被交错**。解决办法釜底抽薪——让硬件提供**一条原子指令**，把「读旧值、改、写新值」变成不可分割的单个操作。Test-and-Set 与 Compare-and-Swap 是其中最著名的两条，它们是锁、自旋锁、无锁编程的全部地基。<span class="marginnote">回顾《竞态条件》的根源：`counter++` 的读-改-写不原子。硬件原语把「读-改-写」压缩成一条 CPU 指令，从物理上消灭了交错的可能——这就是「原子性」的最底层来源。</span>

## 1 Test-and-Set：最原始的自旋锁内核

**Test-and-Set（TS）**：一条原子指令，其行为等价于：

```c
bool test_and_set(bool *target) {
    bool rv = *target;   // 读旧值
    *target = true;      // 写 true
    return rv;           // 返回旧值
}
```

关键：**读与写在同一指令周期内完成，不可被打断**。无论多核怎么并发，这条指令都不会交错。

用 TS 实现互斥锁：

```c
bool lock = false;   // false = 锁空闲

// 进入区
while (test_and_set(&lock))   // 抢锁：返回 true 说明已被占，继续自旋
    ;                          // 忙等待

/* 临界区 */

// 退出区
lock = false;                  // 释放锁
```

**test_and_set 返回旧值**：若返回 `false`，说明锁原本空闲，本进程抢到并把它置为 `true`，进临界区；若返回 `true`，说明锁已被占，循环重试——这就是**自旋锁（spinlock）**的雏形。<span class="marginnote">自旋锁的「自旋」就是 while 忙等：进程在临界区很短时，忙等的开销（几十纳秒）远小于睡眠-唤醒的切换开销（微秒级），所以内核短临界区都用自旋锁。临界区长的话，忙等就是在烧 CPU。</span>

## 2 Compare-and-Swap：更强大的「条件写」

**Compare-and-Swap（CAS）**：一条原子指令，比较目标值与期望值，相等则写入新值：

```c
int compare_and_swap(int *value, int expected, int new_value) {
    int temp = *value;         // 读当前值
    if (*value == expected)    // 若等于期望值
        *value = new_value;    // 则写入新值
    return temp;               // 返回旧值
}
```

CAS 的威力：它支持「读-比较-条件写」的复杂原子逻辑。用 CAS 实现「无锁加一」：

```c
void atomic_increment(int *counter) {
    int old;
    do {
        old = *counter;                          // 读当前值
    } while (compare_and_swap(counter, old, old + 1) != old);
    // 若期间被别的线程改了，CAS 返回的不是 old，重试
}
```

**CAS 的无锁思路**：先乐观地读，再 CAS 检查「读完之后没人改过吗」——若 CAS 失败（说明并发改了），重新读再试。这就是**乐观并发控制**，是后续《无锁编程》的基石。<span class="marginnote">CAS 失败重试的循环被称为「CAS 循环」。它把「加锁-更新-解锁」变成「读-试-重试」，避免了锁的睡眠与切换。但 CAS 有个著名陷阱——ABA 问题，见第六篇《无锁编程、CAS 的 ABA 问题》。</span>

## 3 原子变量：把原语封装成数据类型

**原子变量（atomic variable）**：把原子指令封装成带类型的变量及操作，如 `atomic_t`、`std::atomic<int>`。程序员不再手动写 while+CAS，而是调用 `atomic_fetch_add`、`counter.fetch_add(1)`。

原子变量带来的保证：

- **原子性**：`fetch_add` 是单条原子指令，读-改-写不分割。
- **可见性**：配合内存屏障，保证别的线程立刻看到更新。
- **顺序性**：可指定内存序（顺序一致、宽松等），控制编译优化与乱序。

**辨析｜易错点：** 「原子变量 = 什么都不用管了」是危险误解。**原子变量保证「单次操作原子」，不保证「多步操作的组合原子」**。`counter.fetch_add(1)` 是原子的，但「先读 counter、再决定写别的」这种多步逻辑不是原子的，需要锁。原子变量是「更快的锁」而不是「不用锁」。

另一个易错点：**「`volatile` 就是原子」**。`volatile` 只防止编译器优化读写，**没有原子指令的支持，`volatile int` 的 `++` 仍然可能交错**；真正的原子必须用 `atomic` 类型或显式原子指令。C11 的 `_Atomic` 与 C++11 的 `std::atomic` 才是「原子」的正确写法。

## 4 核心对比表：TS vs CAS

| 维度 | Test-and-Set | Compare-and-Swap |
| --- | --- | --- |
| 原子操作 | 置 true 并返回旧值 | 比较相等才写，返回旧值 |
| 能否做条件写 | 不能（总是置 true） | **能**（仅当等于期望值） |
| 实现锁 | 直接、经典 | 可以，但更常用在无锁上 |
| 实现无锁结构 | 难 | **易（CAS 循环）** |
| 现代 CPU 指令 | x86 `xchg`（配合） | x86 `cmpxchg`、ARM `LDXR/STXR` |

**设计启示**：TS 是「无脑置位」，适合做互斥锁；CAS 是「有脑比较」，适合做无锁数据结构。两者都是「单条不可分割的读-改-写」，区别在「改」的逻辑——这正是它们分道扬镳的支点。

## 5 小结

- **Test-and-Set**：原子地「读旧值、置 true、返回旧值」，直接实现自旋锁。
- **Compare-and-Swap**：原子地「比较-条件写-返回旧值」，是乐观并发与无锁编程的基础。
- **原子变量**把原子指令封装成类型（`std::atomic`），保证单次操作原子性与可见性。
- **`volatile` ≠ 原子**，原子必须靠硬件原子指令。
- 原子变量保证「单步原子」，不保证「多步组合原子」，复杂逻辑仍需锁。

在下一节，我们把「硬件原语」组装成两类真实锁——**互斥锁 Mutex 与自旋锁 Spinlock**。
