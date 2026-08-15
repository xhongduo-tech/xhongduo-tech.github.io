---
title: TCP 的有限状态机
date: 2026-08-07
---

# TCP 的有限状态机

<div class="epigraph">
<p>一条 TCP 连接的一生，就是在一张状态图上的漫长旅行：从 CLOSED 出发，绕一圈，再回到 CLOSED。</p>
<footer>—— 网络教材中的通俗说法</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机网络 ｜ 谢希仁《计算机网络》§5.7 ｜ 2026-08-07</p>
</div>

## 为什么从 TCP 状态机开始

三次握手与四次挥手讲了「事件的顺序」，但 TCP 连接的实际管理是一台**有限状态机（FSM）**——它明确规定了：**在哪个状态、收到什么报文、做什么动作、转到哪个状态。**<span class="marginnote">状态机是协议的「宪法」：<strong>不靠代码逻辑临时判断，而是严格按「状态 × 事件 → 动作 + 新状态」迁移</strong>。TCP 有 11 个状态，覆盖「未开、监听、握手、传输、挥手、等待」全部生命周期。读懂它，抓包时的每一个状态转换都能对上号。</span>

这一节讲：**11 个状态的含义、从握手到挥手的状态转换、以及状态机的排障价值。**

## 1 TCP 的 11 个状态

TCP 状态机共有 11 个状态，先逐一立住：<span class="marginnote">把 11 个状态按「生命周期」分组记：<strong>建连三兄弟（SYN_SENT、SYN_RCVD、ESTABLISHED）、断连六兄弟（FIN_WAIT_1、FIN_WAIT_2、CLOSE_WAIT、CLOSING、LAST_ACK、TIME_WAIT）、基础两个（CLOSED、LISTEN）</strong>。分组记忆比死记快得多。</span>

| 状态 | 含义 |
| --- | --- |
| CLOSED | 初始状态，无连接 |
| LISTEN | 服务器监听，等待连接 |
| SYN_SENT | 已发 SYN，等待 SYN+ACK |
| SYN_RCVD | 已收 SYN 并回 SYN+ACK，等待 ACK |
| ESTABLISHED | 连接已建立，正常传输 |
| FIN_WAIT_1 | 已发 FIN，等待对方 ACK |
| FIN_WAIT_2 | 已收对 FIN 的 ACK，等待对方 FIN |
| CLOSE_WAIT | 已收对方 FIN，等待本地关闭 |
| CLOSING | 双方几乎同时发 FIN（罕见） |
| LAST_ACK | 已发自己的 FIN，等待最后 ACK |
| TIME_WAIT | 主动关闭后等待 2×MSL |

**辨析｜易错点：** **CLOSING 是最容易漏的状态**——它发生在「双方同时主动关闭」的罕见场景。另外，**服务端不会进入 FIN_WAIT_1/2 与 TIME_WAIT**（它是被动方），这些是主动关闭方的状态。**「谁主动关闭，谁进 TIME_WAIT」**是最常考的判断。

## 2 建立连接的状态迁移

以 A（主动）与 B（被动）为例，握手的迁移路径：

| 动作 | A 的状态 | B 的状态 |
| --- | --- | --- |
| 初始 | CLOSED | CLOSED → LISTEN |
| A 发 SYN | CLOSED → SYN_SENT | LISTEN |
| B 回 SYN+ACK | SYN_SENT | LISTEN → SYN_RCVD |
| A 回 ACK | SYN_SENT → ESTABLISHED | SYN_RCVD → ESTABLISHED |

**辨析｜易错点：** 状态迁移的触发是「**收到特定报文**」而非「时间流逝」——**SYN_SENT 只有收到 SYN+ACK 才进 ESTABLISHED**，收到其他就超时或 RST。**「状态是事件的产物，不是时间的产物」**是理解一切状态机的关键。

## 3 释放连接的状态迁移

四次挥手对应的完整迁移（A 主动）：

| 步骤 | A 的状态 | B 的状态 |
| --- | --- | --- |
| A 发 FIN | ESTABLISHED → FIN_WAIT_1 | ESTABLISHED |
| B 回 ACK | FIN_WAIT_1 → FIN_WAIT_2 | ESTABLISHED → CLOSE_WAIT |
| B 发 FIN | FIN_WAIT_2 | CLOSE_WAIT → LAST_ACK |
| A 回 ACK | FIN_WAIT_2 → TIME_WAIT | LAST_ACK → CLOSED |
| 2×MSL 后 | TIME_WAIT → CLOSED | — |

**辨析｜易错点：** **B 的状态序列是 ESTABLISHED → CLOSE_WAIT → LAST_ACK → CLOSED**——它从不进 FIN_WAIT。而 **A 的状态序列是 FIN_WAIT_1 → FIN_WAIT_2 → TIME_WAIT → CLOSED**。把两条「状态链」分开记，抓包时一眼就能判断「谁主动关闭」。<span class="marginnote">排障的实战价值：<strong>`netstat`（或 `ss`）</strong>里看到大量 FIN_WAIT_2，说明主动关闭方在等对方发 FIN（对方迟迟不关）；看到大量 CLOSE_WAIT，说明被动方应用没调用 close（bug）。状态机不只是理论，它是生产排障的第一张地图。</span>

## 4 状态机 = 排障地图

状态机最大的实用价值是排障定位。常见的「卡状态」场景：

| 观察到的状态 | 含义 | 可能的根因 |
| --- | --- | --- |
| SYN_SENT 长时间 | 请求没被响应 | 防火墙丢包、目标端口未开 |
| SYN_RCVD 堆积 | 半连接积压 | 半连接队列满、SYN 洪水 |
| FIN_WAIT_2 大量 | 等对端 FIN | 对端半关闭未完成 |
| CLOSE_WAIT 大量 | 等本地 close | 应用没关闭 socket |
| TIME_WAIT 大量 | 等旧报文消亡 | 大量短连接主动关闭 |

**辨析｜易错点：** **状态机的「卡点」直接指向「故障点」**——这是它排障价值的内核。但要注意：**TIME_WAIT 大量不等于故障**——它是正常现象（短连接场景下必然出现），只有当它耗尽本地端口资源时才需要优化。**「状态本身无好坏，要看它堆积在哪」**是排障的正确姿势。

## 5 为什么 TIME_WAIT 要等 2×MSL

TIME_WAIT 是最容易被质疑的状态：「连接都关了，为什么还赖着 2×MSL 不走？」两个理由：

- **确保最后的 ACK 送达**：主动关闭方的最后一个 ACK 可能丢失。若不等，对方会重发 FIN，此时本方已进 CLOSED，收不到也不处理——连接无法干净关闭。等 2×MSL，给了「重发 FIN → 再回 ACK」的余地。
- **让旧报文消亡**：网络上可能还有「绕路迟到」的旧报文段。等够 2×MSL，等于保证「本连接的所有报文段都已在网络中消亡」，不会混入同四元组的新连接。

**公式解析：为什么是「2 倍」MSL**

$$
\text{等待时长} = 2 \times \text{MSL} \;\Rightarrow\; \text{保证一个报文「从发送到消亡」至多 MSL + 最多往返一次 MSL}
$$

- **第一步，MSL 是什么**：报文段在网络中的**最长存活时间**（如 120 秒）。
- **第二步，为什么 2 倍**：我方发出 ACK 后，若丢失，对方会在 ≤ MSL 内重发 FIN；我方重发 ACK 后再等 MSL 保证其消亡。一来一回各一个 MSL。
- **第三步，代价**：2×MSL 期间四元组被占用，**短连接风暴下端口可能耗尽**——工程上才需要 `SO_REUSEADDR`、长连接池等手段。<span class="marginnote">这也是为什么「<strong>服务端主动关闭会背上 TIME_WAIT 的包袱</strong>」：谁主动关闭，谁就要等 2×MSL。所以高并发服务通常「让客户端先关」或「服务端复用连接」，尽量让自己少当「主动关闭方」。</span>

**辨析｜易错点：** TIME_WAIT 属于**主动关闭方**，被动方（CLOSE_WAIT → LAST_ACK → CLOSED）**没有 TIME_WAIT**。若你在服务器上看到大量 TIME_WAIT，且连接都是「服务端主动 close」导致的，就要思考「服务端是否过早关闭连接」的设计问题了。

## 6 半关闭与同时关闭：状态机的两个边角

TCP 是全双工的，所以「关闭」其实可以**不对称**——这就是**半关闭（half-close）**：A 用 `shutdown(SHUT_WR)` 只关发送方向，但还能继续接收 B 的数据。此时 A 在 FIN_WAIT_2，B 在 CLOSE_WAIT，但数据还能单向流动。**在应用里，只有收到对方 FIN 才真正结束**。

另一个边角是**同时关闭（simultaneous close）**：双方同时发 FIN。此时双方都进入 FIN_WAIT_1，收到对方的 FIN 后都进 CLOSING，最后都进 TIME_WAIT——这就是 CLOSING 状态存在的意义。它罕见，但状态机必须覆盖。

**辨析｜易错点：** **半关闭要靠应用层 API 主动触发**，普通 `close()` 是「双向都关」，不会出现半关闭。同时关闭的识别标志是**双方都从 CLOSING 进 TIME_WAIT**。这两个边角状态（半关闭、CLOSING）在教科书里占比小，却经常是「看起来对不上状态机」考题的出处。

## 7 小结

- **11 个状态**：CLOSED、LISTEN、SYN_SENT、SYN_RCVD、ESTABLISHED、FIN_WAIT_1/2、CLOSE_WAIT、CLOSING、LAST_ACK、TIME_WAIT。
- **分组记忆**：建连三兄弟 + 断连六兄弟 + 基础两个。
- **握手迁移**：CLOSED→SYN_SENT→ESTABLISHED（主动）；LISTEN→SYN_RCVD→ESTABLISHED（被动）。
- **挥手迁移**：FIN_WAIT_1→FIN_WAIT_2→TIME_WAIT→CLOSED（主动）；CLOSE_WAIT→LAST_ACK→CLOSED（被动）。
- **核心规律**：状态是事件的产物；谁主动关闭谁进 TIME_WAIT。
- **排障价值**：卡在哪个状态，就指向哪一类问题。

在下一节，我们将学习 TCP 的「刹车」——**利用滑动窗口实现流量控制**。
