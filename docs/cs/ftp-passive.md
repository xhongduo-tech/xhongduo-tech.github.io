---
title: FTP 被动模式
date: 2026-09-08
section: cs
---

# FTP 被动模式

<div class="epigraph">
<p>控制连接在 21，数据连接另开端口；主动模式让服务器回连客户，NAT 失败；PASV 让客户去连服务器宣告的端口。</p>
<footer>—— 据 RFC 959 FTP；RFC 2428 IPv6 PASV；RFC 1579 防火墙友好整理</footer>
</div>

[NAT](/cs/dhcp-nat) 改了谁能回连。[上一课](/cs/ptp-precision-time) 结束时间。缺口是 **FTP 双连接**与被动模式：为何当代少用但仍能讲清 NAT 与 ALG。本课结束名字、时间与邮件课序。

## 问题

FTP：PORT 命令告诉服务器客户数据口，服务器向客户发起——客户在 NAT 后无映射则失败。PASV：服务器说「来连我的 (h1,h2,p1,p2)」，客户主动连，NAT 友好。IPv6 用 EPSV。应用层网关改包内地址，脆弱。SFTP（SSH 子系统）与 HTTP 上传用单连接，避开这坑。

<span class="marginnote">被动模式（PASV）的「被动」站在服务器角度：服务器不再主动回连客户，而是开一个数据端口「被动地」等你来连。响应里六个数如 (203,0,113,10,200,35) 是 IP 与端口的字节编码：前四个是 203.0.113.10，后两个 200×256+35=51235 是端口号。</span>

不要把 FTP 当推荐文件协议。

<span class="marginnote">RFC 959。本课对照 NAT，不教匿名镜像运营。</span>

### 载荷嵌地址与 NAT 冲突

主动回连失败；PASV 让客户发起。ALG 在加密后失效。当代用单连接协议替代。

## 方法

画主动 vs 被动谁发起数据 TCP。对照 HTTP 单连接 GET。与负载均衡：数据口随机，防火墙要动态开洞或 ALG。

```mermaid
flowchart TD
  CTRL["控制 21"] --> ACT["主动: 服务器回连"]
  CTRL --> PASV["被动: 客户连数据口"]
  NAT["NAT"] --> FAIL["主动常失败"]
```

## 机制

PMTUD、TLS（FTPS）让双连接更痛。Cookie 无。多播无。当代对象是理解「载荷里嵌地址」与 NAT 冲突——SIP 同类，点名。

```mermaid
flowchart TD
  C1["客户连服务器 21：控制连接"] --> CMD["客户发 PASV"]
  CMD --> RES["服务器响应端口六元组"]
  RES --> CALC["客户端算出数据端口"]
  CALC --> DATA["客户主动连该数据端口"]
  DATA --> XFER["传文件，控制连接待命"]
```

<span class="marginnote">这张图回答「被动模式一次传输到底谁连谁」：控制连接由客户发起，数据连接仍由客户发起——两次都是客户主动，服务器只负责宣告端口。主动模式则反过来要服务器回连客户，NAT 后的客户没有可供回连的映射，于是失败。</span>

安全：明文 FTP 泄露口令，应淘汰。

<span class="marginnote">ALG 可以想象成翻译官：NAT 只会改信封（IP 头），信纸里嵌的地址要靠 FTP 应用层网关代改。可一旦信纸上了锁（TLS 加密），翻译官读不了也改不了——这就是「把 ALG 当安全边界会在加密后失效」的原因。</span>

<span class="marginnote">初学者容易以为 NAT 后 FTP 失败是防火墙拦了 21 端口——实际上控制连接常常是通的，坏在 PORT 命令把客户的内网地址（如 192.168.x.x）写进了载荷，服务器照着去连一个不可路由的地址；地址藏在载荷里，NAT 的头部改写规则看不见它。</span>

## 边界

本课不引入 FTP 的每一条命令。BitTorrent 与 DHT 是下一课序第一课。后课默认：被动模式为了 NAT；新协议避免分离数据连接。

把 ALG 当安全边界会在加密后失效。

下一课[BitTorrent 与 DHT](/cs/bittorrent-dht)。

## 小结

- 控制与数据分离，主动回连破 NAT。
- PASV 让客户发起数据连接。
- 单连接协议是更干净的替代。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 959；RFC 1579。
