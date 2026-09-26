---
title: 反向代理
date: 2026-09-08
section: cs
---

# 反向代理

<div class="epigraph">
<p>客户端只看见代理；代理终止 TLS、补头、缓存、再开到后端的连接。它是 L7 的具体形态，不是又一种路由协议。</p>
<footer>—— 据 RFC 9110 网关/代理；HTTP 实践整理</footer>
</div>

[L7 LB](/cs/l4-l7-load-balancing) 常以反向代理实现。[CDN](/cs/cdn-cache-hierarchy) 边缘也是。[上一课](/cs/maglev-lb) 可把流量先送到代理层。缺口是**代理职责**：Hop-by-hop 头、连接复用、缓冲。本课不把服务发现写完。

## 问题

浏览器不直连应用进程。反向代理：证书、HTTP/1.1 到后端 H2 的转换、压缩、WAF、限流令牌桶。`X-Forwarded-For` 传原 IP，信任边界要钉。<span class="marginnote">常见误区：以为后端见到 X-Forwarded-For 就能无条件相信。客户端可以伪造这个头——只有「最外层代理负责写入、后端只信任与自己直连的那一跳」这条链才是安全的。</span>缓冲：代理若把 SSE 攒满再发，实时死——接 SSE 课。WS 要 Upgrade 透传。

不要把反向代理写成正向代理（客户端配置出去）。<span class="marginnote">方向记法：正向代理替「客户端」出门办事，服务器不知道真客户是谁；反向代理替「服务器」接客，客户端不知道后面有几台真服务器。两者共用同一套 HTTP 语义，方向相反。</span>

<span class="marginnote">RFC 9110 定义 proxy/gateway。本课不点名 nginx 配置项当标准。</span>

### 面向客户的网关

终止 TLS、改头、池化后端。hop-by-hop 头不盲目转发。缓冲会杀 SSE。与正向代理方向相反。

## 方法

画：客户 → 代理（TLS 终）→ 后端池。对照 Maglev 纯 L4：不终结 TLS。与 SSH 反向隧道不同方向。

```mermaid
flowchart TD
  CLI["客户端"] --> RP["反向代理"]
  RP --> TLS["终止 TLS"]
  RP --> POOL["后端连接池"]
```

## 机制

连接池摊销后端握手，像 HTTP 持久。TSO 在代理两侧。健康检查由代理做。Cookie 会话可钉后端。QUIC 在代理终止，后端常 HTTP/1.1。PMTUD 在两段各自发生。

```mermaid
flowchart TD
  IN["客户端请求到达（TLS 已终止）"] --> HDR["改写头：补 X-Forwarded-For"]
  HDR --> HOP["剥掉 hop-by-hop 头"]
  HOP --> BUF["缓冲决策：普通请求可缓冲，SSE 必须直通"]
  BUF --> POOL["从后端连接池取一条连接"]
  POOL --> FWD["转发到后端并等待响应"]
  FWD --> HEALTH["健康检查失败则摘除该后端"]
```

<span class="marginnote">hop-by-hop 头就是「只属于这一跳」的头，如 Connection、Transfer-Encoding：代理收到就该自己消化，不原样转给下一跳；Authorization 这类端到端头才要透传。转错方向，协议状态机会对不上。</span>

安全：代理是集中点，配置错误会泄露内网。

## 边界

本课不引入缓存键的全部边角。服务发现是下一课。后课默认：反向代理终结客户协议并转发；hop-by-hop 头不盲目转发。

把代理超时设太短会拆合法 SSE/WS。

下一课[服务发现](/cs/service-discovery)。

## 小结

- 反向代理是面向客户的 L7 网关。
- 管 TLS、头、池、缓冲。
- 与纯 L4 Maglev 分层。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 9110；HTTP 网关实践。
