---
title: SSH 协议
date: 2026-09-08
section: cs
---

# SSH 协议

<div class="epigraph">
<p>SSH 在一条 TCP 上先密钥交换再开信道：会话、端口转发、SFTP 都是信道类型，不是三个端口三个协议。</p>
<footer>—— 据 RFC 4253 SSH 运输；RFC 4254 连接协议整理</footer>
</div>

[TLS 入口](/cs/http-tls-entry) 保护 HTTP。[上一课](/cs/spf-dkim-dmarc) 是邮件。缺口是**交互登录与隧道的 SSH**：版本交换、KEX、认证、多信道。本课不把 NTP 写完。

## 问题

telnet 明文。SSH：二进制包，运输层加密完整性，然后用户认证（公钥/密码），再 multiplex 信道。端口转发把其它应用塞进这条管道，像应用层 GRE。与 QUIC 对照：都有加密+多流，SSH 更老、跑 TCP，有 TCP HOL。主机密钥钉防中间人，类似证书钉但常 TOFU。

<span class="marginnote">「多路复用」翻译一下：认证完成后那条加密连接并不只服务一个终端——每个逻辑流（shell、转发的端口、SFTP）各领一个信道编号，数据包上带着编号走同一条 TCP，接收端按编号分拣。像一条高速公路划出多条车道：路面只有一条，车流互不挡道。</span>

不要把 ssh 命令行选项当协议规范。

<span class="marginnote">RFC 4251–4254。本课不写爆破密码的操作。</span>

### 加密运输加多信道

转发是信道不是独立端口协议。主机密钥 TOFU 防中间人。TCP-over-TCP 当 VPN 有代价。Nagle 对按键有害。

## 方法

画：TCP 22 → KEX → 认证 → 信道。对照 HTTPS：证书 PKI vs known_hosts。与 keepalive：SSH 有应用层 ping。

```mermaid
flowchart TD
  TCP["TCP"] --> KEX["密钥交换"]
  KEX --> AUTH["用户认证"]
  AUTH --> CH["多信道"]
```

## 机制

Nagle 对按键延迟有害，SSH 常关。窗口与流量控制在信道上。TSO 对小交互帮助有限。跳板与 ProxyJump 是会话层，不是 BGP。安全：认证失败限速，不写绕过。

SFTP 是子系统，不是另起 FTP 课的被动模式。

```mermaid
flowchart TD
  TRANS["一条加密运输层"] --> CH1["信道：交互 shell"]
  TRANS --> CH2["信道：端口转发"]
  TRANS --> CH3["信道：SFTP 子系统"]
  CH1 --> FLOW["每信道独立窗口与流控"]
  CH2 --> FLOW
  CH3 --> FLOW
  FLOW --> SAME["全部复用同一条 TCP 连接"]
```

<span class="marginnote">常见误区：把 SSH 隧道当 VPN 用。隧道里跑的应用自己做 TCP，外面 SSH 又是一层 TCP——两层各自重传、各自计时，丢包时内层还在等重传，外层已经超时重发，性能会塌得比任何一层单独存在都难看。这就是本课说的 TCP-over-TCP 代价。</span>

## 边界

本课不引入 SSH 证书 CA 的全部。NTP 是下一课。后课默认：SSH = 加密运输 + 认证 + 多信道。

把 SSH 当通用 VPN 要接受 TCP-over-TCP 的性能。

下一课[NTP](/cs/ntp)。

## 小结

- 一层加密运输，上面多信道。
- 主机密钥钉身份。
- 转发是信道，不是独立协议。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 4253；RFC 4254。
