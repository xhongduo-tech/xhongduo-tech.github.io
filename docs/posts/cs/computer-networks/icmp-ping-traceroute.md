---
title: 网际控制报文协议（ICMP）：ping 与 traceroute
date: 2026-08-07
---

# 网际控制报文协议（ICMP）：ping 与 traceroute

<div class="epigraph">
<p>IP 自己不会说话，但它的「信使」ICMP 会——出错时报信，被问时应答。</p>
<footer>—— 网络教材中的通俗说法</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机网络 ｜ 谢希仁《计算机网络》§4.6 ｜ 2026-08-07</p>
</div>

## 为什么从 ICMP 开始

IP 是「尽力而为」的——包丢了、路由器找不着路、TTL 耗尽……这些「事故」谁来报告？IP 本身太忙太简单，没空写事故报告。于是有了 **网际控制报文协议（ICMP，Internet Control Message Protocol）**：**IP 层的「信使」，专门报告差错与回答询问。**<span class="marginnote">ICMP 报文本身装在 IP 数据报里传输（协议号 1），但它不是「上层数据」，而是<strong>IP 层的控制辅助协议</strong>。它的两大职责：<strong>差错报告</strong>（告诉发送方「你的包出了什么问题」）与<strong>询问</strong>（ping 的请求/应答）。你每天用的 `ping` 与 `traceroute`，都是 ICMP 的经典应用。</span>

这一节讲：**ICMP 的两类报文、ping 的原理、traceroute 的原理。**

## 1 ICMP 的两类报文

ICMP 报文分两大类：

**差错报告报文**（报告 IP 数据报的问题）：<span class="marginnote">常见差错报文：<strong>终点不可达</strong>（目的网络/主机/端口找不到）、<strong>超时</strong>（TTL 减到 0）、<strong>参数问题</strong>（首部字段有误）、<strong>重定向</strong>（让发送方换一条更优路由）。这些报文的共同特点是「<strong>只报告、不纠正</strong>」——ICMP 告诉发送方出错了，但重传是上层（TCP）的事。</span>

**终点不可达（Destination Unreachable）**：目的网络、主机或端口不可达时报告。
**超时（Time Exceeded）**：TTL 减到 0 时报告——traceroute 就靠它。
**重定向（Redirect）**：路由器告诉发送方「有更近的下一跳，请改道」。
**参数问题**：首部字段有误时报告。

**询问报文**（主动询问对方状态）：**回显请求/应答（Echo Request/Reply）**——ping 用的就是它。

**辨析｜易错点：** 差错报文里会**携带出错的 IP 数据报的首部与部分数据**（通常前 8 字节）——目的是让接收方认出「哪个数据报出了错」。但注意：**对差错报文本身不再发差错报文**（避免「报错的报错」陷入循环）。这个「不再报错」的规则是防止 ICMP 风暴的关键。

## 2 ping 的原理：回显请求与应答

**ping**（Packet Internet Groper）用 ICMP 的**回显请求（Echo Request）** 与**回显应答（Echo Reply）** 测试连通性：

1. 发送方给目的主机发一个 **Echo Request** 报文。
2. 目的主机收到后，回一个 **Echo Reply** 报文。
3. 发送方若收到 Reply，说明「目标可达」；并可从往返时间估算链路时延。

**ping 能告诉你什么**：目标是否可达、往返时延（RTT）多少、丢包率多少。<span class="marginnote">ping 的「往返时间」衡量的是「<strong>整个往返</strong>」——去程加回程。所以 ping 显示的 30 ms，是去程约 15 ms 加回程约 15 ms。判断网络质量时，ping 的稳定值比平均值更有参考意义（抖动才是大问题）。</span>

**辨析｜易错点：** **ping 通 ≠ 应用可用**。ping 只测 IP 层的连通性——即使目标主机存活、IP 层通，也不代表它的 80 端口开着、Web 服务正常。**「ping 通但网站打不开」**是经典的排障起点：问题多半在端口或应用层，不在 IP 连通性。反过来，**很多网络默认禁用 ICMP**，ping 不通也不一定代表主机宕机——「防火墙禁 ping」是常见配置。

## 3 traceroute 的原理：用 TTL 逐跳「逼供」

**traceroute**（路由追踪）利用 ICMP 的**超时报告**来探测每一跳：

1. 发送方先发一个 **TTL=1** 的探测包：第一跳路由器把 TTL 减到 0，丢弃并回 ICMP「超时」报文——从报文里取第一跳的 IP。
2. 再发 **TTL=2** 的包：第一跳转发（TTL 变 1），第二跳减到 0 丢弃回超时——取第二跳的 IP。
3. 依次递增 TTL，直到到达目的主机——目的主机不回「超时」，而是回「终点不可达」或 Echo Reply，标记探测结束。<span class="marginnote">每一跳的「超时」都来自那跳的路由器，traceroute 就从这些报文里读出每跳的地址——<strong>用 TTL 的递减，把一个不可见的转发路径「逼」成了可见的地址列表</strong>。这是「用协议行为做测量」的经典技巧，也是 TTL 字段「防止环路」之外的第二大用途。</span>

**traceroute 能告诉你什么**：数据报实际走的路径（哪些路由器）、每跳的时延、哪里堵了、哪里断了。<span class="marginnote">Windows 下命令是 `tracert`，Linux/macOS 是 `traceroute`。看到某几跳都是「*」时，可能是路由器不响应 ICMP（被防火墙拦），不一定是链路断了——排障时别被星号吓到。</span>

**辨析｜易错点：** traceroute 的三次探测（默认）是为了让时延显示更可靠——**同一跳发三次包，看时延的稳定性**。三次时延差异大（抖动大）说明该跳拥塞。另外，**traceroute 依赖中间路由器回「超时」报文**——如果某台路由器不响应 ICMP，那跳就显示为星号。

## 4 ICMP 的角色定位：信使不是主角

最后厘清 ICMP 在协议栈中的位置：<span class="marginnote">ICMP 与 IP 的关系像「秘书与领导」：<strong>IP 负责把数据送走（主角），ICMP 负责把事故与询问传递给相关人员（辅助）</strong>。ICMP 报文装在 IP 数据报里，但它的「协议号」是 1——网络层据此把载荷交给 ICMP 模块，而不是 TCP/UDP。所以 ICMP 不是「IP 之上的应用」，而是「IP 的左右手」。</span>

**辨析｜易错点：** ICMP 虽装在 IP 数据报里，但**它不是「上层协议」**——它属于网际层，与 IP 平级。抓包时你会看到 ICMP 报文没有 TCP/UDP 端口——它是「端口号」缺席的协议，靠「类型/代码」字段区分各种报文。**「有端口的是 TCP/UDP，没端口的是 ICMP」**是抓包识别的快速判断。

## 5 小结

- **ICMP 的职责**：IP 层的差错报告与询问，装在 IP 数据报里（协议号 1）。
- **两大类报文**：差错报告（终点不可达、超时、重定向、参数问题）与询问（回显请求/应答）。
- **ping 原理**：Echo Request/Reply，测连通性与往返时延；「ping 通 ≠ 应用可用」。
- **traceroute 原理**：递增 TTL，利用「超时」报告逐跳逼出路径；`ping`/`traceroute`。
- **角色定位**：ICMP 是 IP 的辅助协议，不是上层应用；无端口，靠类型/代码区分。
- **排障心法**：ping 测连通、traceroute 定位断点，配合抓包使用效果最佳。

在下一节，我们将进入路由的世界——**互联网的路由选择：静态路由与动态路由**。
