---
title: SMTP / IMAP
date: 2026-09-08
section: cs
---

# SMTP / IMAP

<div class="epigraph">
<p>SMTP 在 MTA 之间推送邮件；IMAP 让客户端在服务器上管理邮箱。提交与投递端口、TLS 时机不同，都不是 HTTP。</p>
<footer>—— 据 RFC 5321 SMTP；RFC 9051 IMAP4rev2 整理</footer>
</div>

[上一课](/cs/geodns) 不传邮箱。[DNS MX](/cs/dns-rr) 已点名记录。缺口是**邮件运输与存取**：SMTP 对话、IMAP 选文件夹。本课不把 SPF 写完。

## 问题

邮件要跨管理域、要存、要多设备。SMTP：MAIL FROM、RCPT TO、DATA，MTA 跳到跳，MX 决定下一跳。提交（587）与传输（25）政策不同。IMAP：TCP 上命令取消息，状态在服务器；POP 对照为下载删除，点名。持久连接、大附件 MTU、超时与无线半开都出现。不是 REST 资源模型。

不要把 Gmail 网页当协议。

<span class="marginnote">术语翻译：MUA（Mail User Agent）是你手上的邮件客户端，MTA（Mail Transfer Agent）是服务器之间的搬运工。信从 MUA 交给 MTA 只走一次提交，之后 MTA 到 MTA 像接力赛，每一跳只查「下一个收件域的 MX 记录在哪」，并不关心这封信最终落在谁的收件箱里。</span>

<span class="marginnote">RFC 5321。STARTTLS 升级。本课不把每条 SMTP 代码背完。</span>

### 投递链与邮箱存取

SMTP 跳到跳，IMAP 状态在服务器。逐跳 TLS 不是端到端密信。网页邮箱不是协议。

## 方法

画：MUA → 提交 → MTA 链 → 投递 → IMAP 取。对照 HTTP POST：无统一缓存，有队列与重试。与 QUIC：邮件仍多 TCP。

<span class="marginnote">数字实例：端口各记一个——客户端发信走 587（submission，通常要求登录并 STARTTLS），服务器之间走 25（中继，公网可达），IMAP 是 143（明文后升级）或 993（TLS 直连）。配邮件客户端时端口选错，症状多半是「连得上但发不出去」，而不是直接连不上。</span>

```mermaid
flowchart TD
  MUA["客户端"] --> SUB["SMTP 提交"]
  SUB --> MTA["MTA 到 MTA"]
  MTA --> STORE["邮箱存储"]
  STORE --> IMAP["IMAP 存取"]
```

## 机制

DNS 缓存 MX 影响切换。TLS 对 SMTP 是逐跳，不是端到端密信（那是 OpenPGP/S/MIME，点名）。负载均衡对 25 端口要小心会话。分块附件像 HTTP 体，但边界是 MIME。

```mermaid
flowchart TD
  G["客户端连上提交或中继端口"] --> EH["EHLO 问候 协商能力"]
  EH --> MF["MAIL FROM 声明发件人"]
  MF --> RT["RCPT TO 逐个给收件人"]
  RT --> DA["DATA 传正文 以单独一行点号结束"]
  DA --> Q{"服务器接受吗?"}
  Q -- "接受 250" --> MO["入队 继续下一跳"]
  Q -- "拒绝 4xx/5xx" --> RP["4xx 延迟重试，5xx 退信"]
  MO --> QU["QUIT 结束会话"]
```

<span class="marginnote">常见误区：以为逐跳 TLS 等于端到端加密。每段 TLS 只保护「这一段链路」，中途每个 MTA 都看得见、也改得了明文；真正的内容保密要靠 OpenPGP 或 S/MIME 在信体上另加一层——MTA 只剩信封，看不见信纸。</span>

垃圾与伪造下一课认证。保活防止 NAT 拆 IMAP。

## 边界

本课不引入 JMAP 的全部。SPF/DKIM/DMARC 是下一课。后课默认：SMTP 推送投递；IMAP 远程邮箱。

把 25 对全世界开放中继会成为垃圾跳板。

下一课[SPF / DKIM / DMARC](/cs/spf-dkim-dmarc)。

## 小结

- SMTP 管投递链；IMAP 管邮箱访问。
- MX 与逐跳 TLS 是核心机制。
- 与 HTTP 会话模型不同。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 5321；RFC 9051。
