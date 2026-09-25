---
title: 队头阻塞对照
date: 2026-09-08
section: cs
---

# 队头阻塞对照

<div class="epigraph">
<p>同名「队头」出现在交换结构、HTTP/1.1 串行响应、以及 TCP 字节流上的 HTTP/2；解法一层一层不同。</p>
<footer>—— 据 RFC 9112 / 9113；Karol HOL；RFC 9000 对照整理</footer>
</div>

[HTTP/1.1 持久](/cs/http11-persistent-chunked) 把请求排成队。[HTTP/2 多路](/cs/http2) 已给应用层解。[交换机 HOL](/cs/output-queue-hol) 是结构。[上一课](/cs/http11-persistent-chunked) 留下管道化失败。缺口是**对照三层 HOL**，避免混名。本课不把 HTTP/3 写完。

## 问题

1.1：连接上前一响应未完，后一请求的响应不能开始——应用 HOL。开多连接是笨解，伤 cc。HTTP/2：多流交错，应用 HOL 解了，但 TCP 丢一个段挡住所有流——运输 HOL。QUIC 再解运输。交换机 VOQ 解的是第三种：入端口 FIFO。PFC 暂停是第四种「整口冻」。

不要用一种队列算法「统一解决」。

<span class="marginnote">教学对照。Kurose 分章讲，这里收口进阶课的词汇。</span>

<span class="marginnote">直觉类比：1.1 像单窗口柜台，前一人的慢业务堵住全队；H2 像叫号取件但整箱按序运输，箱子在途中碎一格、后面全部压着；交换机 HOL 则是所有卸货口共用一条进货通道，互不相干的货也互相堵。</span>

### 写 HOL 必须加层

1.1 应用串行；H2 解之但仍受 TCP；H3 再解运输。交换 VOQ 是第三对象。混名无法排障。

## 方法

三列对照：层、现象、对策。画三条路径。与 SCTP/QUIC 流对照。

<span class="marginnote">术语翻译：HOL（Head-of-Line Blocking，队头阻塞）就是「排在最前面的没走成，后面全走不了」；VOQ（Virtual Output Queue，虚拟输出队列）是入端口为每个出端口单独开一条队，队头只堵自己要去的那一个出口。</span>

```mermaid
flowchart TD
  H11["HTTP/1.1 串行"] --> H2["H2 多流"]
  H2 --> TCP["仍受 TCP HOL"]
  TCP --> H3["QUIC 多流"]
  SW["输入 FIFO"] --> VOQ["VOQ"]
```

## 机制

TSO 合并不造 1.1 HOL，只改段大小。WFQ 不修 TCP HOL。数据中心 incast 是出口队列，不是 HTTP HOL，但浏览器多连接会加重 incast。CDN 用多连接与 H2 混合。

命名纪律：写「HOL」必须加层。

H2 的运输层 HOL 是怎么发生的：

```mermaid
flowchart TD
  S1["流 A 的帧"] --> Q["同一条 TCP 连接按序交付"]
  S2["流 B 的帧"] --> Q
  S3["流 C 的帧"] --> Q
  Q --> L{"中间丢了一个段?"}
  L -- "没丢" --> OK["各流帧按时到达"]
  L -- "丢了" --> W["重传期间后续所有帧被扣留"]
  W --> R["重传到达后按序补交"]
```

<span class="marginnote">常见误区：以为上了 HTTP/2 多路复用就不卡了。应用层确实解了串行，但底下仍是一条 TCP 字节流——一个段丢了，重传期间后面所有流的帧都得干等；这正是 QUIC 给每个流独立编号、独立交付的动机。</span>

## 边界

本课不引入 HTTP/2 优先级树的全部。HTTP/3 是下一课。后课默认：H2 解应用 HOL，H3 解运输 HOL。

把所有卡顿都叫 HOL 会无法排障。

下一课[HTTP/3](/cs/http3)。

## 小结

- 1.1 应用 HOL；H2 解之、TCP 仍堵。
- QUIC 解运输 HOL。
- 交换 HOL 是第三对象。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 9112/9113/9000；Karol et al.。
