---
title: 异常处理、断言与日志
date: 2026-08-07
---

# 异常处理、断言与日志

<div class="epigraph">
<p>程序最诚实的时刻，是它出错的时候。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第1卷第7章 ｜ 2026-08-07</p>
</div>

## 为什么从异常处理开始

至今为止的代码都假设「一切顺利」——文件一定存在、网络一定连通、整数一定不溢出。但真实世界不是这样：磁盘会满、用户会输入乱码、第三方接口会超时。**异常处理（exception handling）**是 Java 对「程序出错时怎么办」的体系化回答：它把「错误信号」与「正常返回值」分开，让错误沿着调用栈向上传递，直到有人愿意处理它。配合**断言**（开发期检查「不可能发生」的条件）与**日志**（记录运行时发生了什么），三件套构成了 Java 程序的「容错与可观测性」基础设施。

## 1 异常层次：Throwable 家族

Java 的所有错误都以**对象**形式存在，它们的根是 `java.lang.Throwable`，下分两支：

$$

\text{Throwable} \begin{cases} \text{Error} & \text{—— 虚拟机层面的严重问题（OOM、StackOverflow）} \\ \text{Exception} & \begin{cases} \text{RuntimeException} & \text{—— 运行时异常（不受检查）} \\ \text{其他受检异常} & \text{—— 编译器强制处理} \end{cases} \end{cases}

$$