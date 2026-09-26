---
title: printk
date: 2026-09-08
section: cs
---

# printk

<div class="epigraph">
<p>printk 把内核消息写入环形缓冲：级别、时间戳、可从 /dev/kmsg 读；它必须在几乎不能睡的上下文里仍能用。</p>
<footer>—— 据 Linux printk 文档；[audit](/cs/kernel-audit) 为安全日志对照</footer>
</div>

[上一课](/cs/suspend-resume) 失败时常只剩串口。[NMI](/cs/nmi) 也不能睡。缺口是 **printk**：环形缓冲、console、与延迟打印。观测课序从此开始。

## 问题

`printk` 格式化进 `log_buf`。级别：KERN_ERR 等。缺口：锁与递归；nbcon 多 console；丢消息当缓冲满。本课不把每条格式说明符当作业。

<span class="marginnote">`pr_debug` 可编译去掉。early printk 在控制台驱动前用特定硬件。对象是内核日志，不是 systemd journal 全文。</span>

<span class="marginnote">术语翻译：环形缓冲就是一段定长数组加「写到底就绕回开头」的用法——像一条首尾相接的传送带：新消息永远写在写指针处，缓冲一满，最旧的消息就被悄悄顶掉。</span>

## 方法

调用 printk → 写入 ring → 唤醒 klogd/journal。对照 [sk_buff](/cs/skbuff)：一个包，一个字符环。对照 ftrace 后课：printk 太重不能每包用。对照 [fsync](/cs/fsync)：kmsg 默认不持久。

```mermaid
flowchart TD
  CALL["printk"] --> RING["log_buf 环形"]
  RING --> CONS["console 输出"]
  RING --> KMSG["/dev/kmsg"]
```

## 机制

printk 是内核最旧的观测面，保证「还能说话」。税是锁与串口慢。不要写成 ELK 栈。与 [KPTI](/cs/kpti-os)：用户读 kmsg 要权限。

<span class="marginnote">为什么重要：printk 必须能在持锁、关抢占甚至 NMI 上下文里调用，所以它不能睡眠、不能分配内存——这正是它只能往一块预留的静态环里写的全部原因；普通日志库的 malloc 和互斥锁在这里都是禁区。</span>

风暴：中断里 printk 可活锁，需限速。

环写满时发生什么、丢的是谁的消息：

```mermaid
flowchart TD
  NEW["新消息到达"] --> CHK{"环还有空位吗"}
  CHK -->|"有"| APP["追加 尾指针前移"]
  CHK -->|"满"| DROP["覆盖最旧消息 头指针前移"]
  DROP --> LOSE["早期日志无声消失"]
  APP --> R["读者按序号从 /dev/kmsg 读"]
  LOSE --> R
```

<span class="marginnote">数字实例：log_buf 常见配置 128 KB 到 1 MB；按一条消息约 100 字节算，一千来条就能写满，重启后环归零——要留崩溃现场得靠 pstore 或 kdump，不能指望 printk 自己记着。</span>


实现上：console_lock 曾让多核 printk 变成串行灾难，nbcon 在改这。rate limit 防止中断风暴自激。持久要靠 pstore 或用户态 journal，ring 重启即逝。 读法上只引用[上一课](/cs/suspend-resume)的结论，不把对象换成训练推理或限价簿。

本课在操作系统进阶的「安全、启动与调试 / 观测与调试」课序里，对象是 **printk**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

## 边界

本课不引入所有 index/seq。不保证崩溃时环还在——kdump 后课。下一课低开销跟踪：ftrace。

<span class="marginnote">常见误区：以为内核日志会像应用日志那样落盘。printk 只保证「写进内存环」，往文件、网络里搬是用户态 journal 的事；机器断电那一刻还在环里没被搬走的消息就没了。</span>


版本字段会变，课序钉的是机制对象「printk」，不是某一主线内核的结构体名。
后课默认：内核消息进 ring buffer。静态与动态 tracepoint，下一课 ftrace。

## 小结

- printk 写环形缓冲并可选打到 console。
- 必须能在受限上下文使用；满则丢。
- ftrace 是下一课。
- 出处：Linux printk；kernel logging。
