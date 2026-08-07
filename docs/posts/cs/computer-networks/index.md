---
pageClass: plain-doc
---

# 计算机网络

对标谢希仁《计算机网络》与 Tanenbaum《Computer Networks》的完整章节体系，从体系结构到现代专题逐节写透。

## 主题规划

<ProgressGrid cat="cs/computer-networks" />


### 第 1 章 概述

- [x] [计算机网络在信息时代中的作用](./networks-in-information-age)
- [x] [互联网概述：网络的网络](./internet-overview)
- [x] [互联网的组成：边缘部分与核心部分](./internet-composition)
- [x] [电路交换、报文交换与分组交换](./packet-switching)
- [x] [计算机网络的定义与分类](./network-definition-classification)
- [x] [计算机网络的性能指标：速率、带宽、吞吐量](./network-performance-metrics)
- [x] [时延、时延带宽积、往返时间与利用率](./delay-bandwidth-product)
- [x] [计算机网络体系结构：协议与分层](./network-architecture-protocol-layering)
- [x] [OSI 七层参考模型](./osi-seven-layer-model)
- [x] [TCP/IP 四层体系结构](./tcp-ip-four-layer-architecture)
- [x] [五层协议体系结构与封装、解封装](./five-layer-protocol-architecture)
- [x] [实体、协议、服务与服务访问点](./entity-protocol-service-sap)

### 第 2 章 物理层

- [x] [物理层的基本概念与四大特性](./physical-layer-basic-concepts)
- [x] [数据通信的基础知识：信道、信号与码元](./data-communication-basics)
- [x] [信道的极限容量：奈氏准则与香农公式](./channel-capacity-nyquist-shannon)
- [x] [传输媒体：双绞线、同轴电缆、光纤](./transmission-media)
- [x] [无线传输：无线电、微波、红外与卫星通信](./wireless-transmission)
- [x] [数字调制：ASK、FSK、PSK 与正交振幅调制（QAM）](./digital-modulation)
- [x] [模拟信号与数字信号的编码：NRZ、曼彻斯特、差分曼彻斯特](./line-coding)
- [x] [信道复用技术：频分复用（FDM）与时分复用（TDM）](./fdm-tdm)
- [x] [统计时分复用（STDM）](./stdm)
- [x] [波分复用（WDM）与密集波分复用（DWDM）](./wdm-dwdm)
- [x] [码分复用（CDM）与码分多址（CDMA）](./cdm-cdma)
- [x] [宽带接入技术：ADSL 与 HFC](./adsl-hfc)
- [x] [FTTx 光纤接入](./fttx)

### 第 3 章 数据链路层

- [x] [数据链路层的功能与点对点信道](./data-link-layer-functions)
- [x] [封装成帧与帧定界](./framing)
- [x] [透明传输：字节填充与比特填充](./transparent-transmission)
- [x] [差错检测：奇偶校验与循环冗余检验（CRC）](./crc-error-detection)
- [x] [停止-等待协议](./stop-and-wait)
- [x] [连续 ARQ 协议与滑动窗口](./sliding-window-arq)
- [x] [回退 N 帧（GBN）与选择重传（SR）](./gbn-sr)
- [x] [点对点协议（PPP）的组成与帧格式](./ppp)
- [x] [PPP 的工作状态与认证](./ppp-states-authentication)
- [x] [使用广播信道的数据链路层：局域网概述](./broadcast-channel-lan)
- [x] [CSMA/CD 协议：载波监听、碰撞检测与退避算法](./csma-cd)
- [x] [以太网的信道利用率与最短帧长](./ethernet-min-frame-length)
- [x] [以太网的 MAC 层：MAC 地址与帧格式](./ethernet-mac-layer)
- [x] [在物理层扩展以太网：集线器与冲突域](./hub-collision-domain)
- [x] [在数据链路层扩展以太网：网桥与交换机](./bridge-switch)
- [x] [以太网交换机的自学习与转发](./switch-learning-forwarding)
- [x] [虚拟局域网（VLAN）与 802.1Q](./vlan-8021q)
- [x] [高速以太网：100BASE-T、吉比特与万兆以太网](./fast-gigabit-10g-ethernet)

### 第 4 章 网络层

- [x] [网络层提供的两种服务：虚电路与数据报](./virtual-circuit-datagram)
- [x] [网际协议（IP）与虚拟互连网络](./ip-virtual-internet)
- [x] [IPv4 地址：分类编址与点分十进制](./ipv4-addressing-classful)
- [x] [IP 地址与硬件地址（MAC）的关系](./ip-vs-mac-address)
- [x] [划分子网与子网掩码](./subnetting-subnet-mask)
- [x] [使用子网时分组的转发](./subnet-forwarding)
- [x] [无分类编址（CIDR）与最长前缀匹配](./cidr-longest-prefix)
- [x] [IPv4 数据报格式与分片](./ipv4-datagram-fragmentation)
- [x] [地址解析协议（ARP）](./arp)
- [x] [网际控制报文协议（ICMP）：ping 与 traceroute](./icmp-ping-traceroute)
- [x] [互联网的路由选择：静态路由与动态路由](./routing-static-dynamic)
- [x] [自治系统与内部网关协议、外部网关协议](./as-igp-egp)
- [x] [路由信息协议（RIP）：距离向量算法](./rip-distance-vector)
- [x] [开放最短路径优先（OSPF）：链路状态算法](./ospf-link-state)
- [x] [边界网关协议（BGP）：路径向量与路由策略](./bgp-path-vector)
- [x] [路由器的构成与分组转发流程](./router-structure)
- [x] [IPv6 的基本首部与地址表示](./ipv6-basic-header)
- [x] [IPv4 向 IPv6 过渡：隧道与双协议栈](./ipv4-ipv6-transition)
- [x] [网络地址转换（NAT）](./nat)

### 第 5 章 传输层

- [x] [传输层协议概述：进程之间的通信](./transport-layer-overview)
- [x] [端口、套接字与多路复用/分用](./port-socket-multiplexing)
- [x] [用户数据报协议（UDP）：首部与特点](./udp)
- [x] [传输控制协议（TCP）概述：面向连接的字节流](./tcp-overview)
- [x] [TCP 报文段的首部格式](./tcp-header)
- [x] [可靠传输的工作原理：确认与重传](./reliable-transfer-principles)
- [x] [TCP 的编号、确认与累积确认](./tcp-sequence-ack)
- [x] [超时重传时间的选择：RTT 估计与 Karn 算法](./rtt-karn-algorithm)
- [x] [TCP 连接的建立：三次握手](./tcp-three-way-handshake)
- [x] [TCP 连接的释放：四次挥手](./tcp-four-way-release)
- [x] [TCP 的有限状态机](./tcp-state-machine)
- [x] [利用滑动窗口实现流量控制](./tcp-flow-control)
- [x] [TCP 的拥塞控制：慢开始与拥塞避免](./tcp-congestion-control)
- [x] [快重传与快恢复](./fast-retransmit-recovery)
- [x] [主动队列管理（AQM）与 RED](./aqm-red)

### 第 6 章 应用层

- [x] [应用层协议与进程通信模型：C/S 与 P2P](./client-server-p2p)
- [x] [域名系统（DNS）：域名结构与域名服务器](./dns-domain-structure)
- [x] [DNS 的解析过程：递归查询与迭代查询](./dns-resolution)
- [x] [文件传送协议（FTP）与 TFTP](./ftp-tftp)
- [x] [万维网（WWW）概述：URL 与 Web 体系](./www-url)
- [x] [超文本传送协议（HTTP）：报文结构与请求方法](./http-message-methods)
- [x] [HTTP 的持久连接、Cookie 与会话管理](./http-persistent-cookie)
- [x] [Web 缓存与代理服务器](./web-cache-proxy)
- [x] [电子邮件系统概述：用户代理与邮件服务器](./email-overview)
- [x] [简单邮件传送协议（SMTP）](./smtp)
- [x] [邮件读取协议（POP3 与 IMAP）](./pop3-imap)
- [x] [动态主机配置协议（DHCP）](./dhcp)
- [x] [基于 P2P 的应用：文件分发与 BitTorrent](./bittorrent)

### 第 7 章 网络安全

- [x] [网络安全问题概述：威胁模型与攻击类型](./network-security-threats)
- [x] [两类密码体制：对称密钥与公钥密码](./symmetric-public-key-crypto)
- [x] [对称密钥密码：DES 与 AES](./des-aes)
- [x] [公钥密码体制：RSA 与椭圆曲线密码（ECC）](./rsa-ecc)
- [x] [数字签名与报文鉴别](./digital-signature)
- [x] [密码散列函数：MD5、SHA 与完整性校验](./hash-functions)
- [x] [密钥分配与公钥基础设施（PKI）](./key-distribution-pki)
- [x] [数字证书与证书颁发机构（CA）](./digital-certificate-ca)
- [x] [运输层安全协议：TLS/SSL 的握手与记录协议](./tls-ssl)
- [x] [防火墙：分组过滤与状态检测](./firewall)
- [x] [入侵检测系统（IDS）与入侵防御系统（IPS）](./ids-ips)

### 第 8 章 无线网络和移动网络

- [x] [无线局域网（WLAN）的组成与 802.11 体系结构](./wlan-80211)
- [x] [802.11 的 MAC 层：CSMA/CA 与 RTS/CTS](./csma-ca-rts-cts)
- [x] [802.11 的帧格式与关联过程](./80211-frame-association)
- [x] [无线个人区域网：蓝牙（Bluetooth）](./bluetooth)
- [x] [蜂窝移动通信网：GSM、LTE 与 5G 的演进](./cellular-gsm-lte-5g)
- [x] [蜂窝网络中的移动性管理：切换与漫游](./handoff-roaming)
- [x] [无线网络对高层协议的影响](./wireless-impact-on-upper-layers)

### 第 9 章 现代网络专题

- [x] [HTTP/2：二进制分帧、多路复用与头部压缩（HPACK）](./http2)
- [x] [HTTP/3 与 QUIC：基于 UDP 的可靠传输](./http3-quic)
- [x] [内容分发网络（CDN）的工作原理](./cdn)
- [x] [软件定义网络（SDN）与 OpenFlow](./sdn-openflow)
- [x] [数据中心网络与网络功能虚拟化（NFV）](./datacenter-network-nfv)

