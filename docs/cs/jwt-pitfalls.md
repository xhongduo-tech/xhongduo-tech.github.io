---
title: JWT 陷阱
date: 2026-09-08
section: cs
---

# JWT 陷阱

<div class="epigraph">
<p>JWT 把声明做成可验的紧凑串。陷阱是算法可切换成 none、用对称密钥去验非对称、不验 aud/exp，于是「有令牌」再次变成装饰。</p>
<footer>—— RFC 7519；RFC 7515 JWS；Jones、Bradley 与 Sakimura；对照 auth0/jwt 误用报告一类公开分析</footer>
</div>

上一课[ARP](/cs/arp-spoofing)结束信道映射。身份课序从**应用令牌**起。主干会话可能已有 cookie；JWT 常被当成无状态银弹。缺口是声明合同与算法混淆，不给伪造步骤。

后课默认已经读完本课钉下的合同，只补差，不从该领域第一性原理重开。

## 问题

`alg=none`、把公钥当 HMAC 密钥、不检查 `iss`/`aud`、过长有效期、把权限写进不可吊销的胖令牌。缺口是验证库的默认值，不是 JSON 语法。

<span class="marginnote">术语翻译：声明（claim）就是令牌里「我担保你是谁、权限到什么时候」的键值对，像一张盖了章的工牌——门禁读的是章上的部门和有效期，而不是重新盘问你本人。</span>

### 签名≠加密

JWS 可验不可隐藏。敏感声明要 JWE 或根本不要放进去。

<span class="marginnote">RFC 7519。本课禁止构造恶意 JWT 的操作指导。密钥轮换与 `kid` 注入是实现问题：只接受已知 kid。</span>

<span class="marginnote">常见误区：以为令牌「签过名」别人就看不到内容。实际上 JWS 的 payload 只是 Base64URL 编码，随手可解——把身份证号、手机号塞进去等于明文广播；要保密得用 JWE 加密。</span>

## 方法

列出必须检查的头与声明。对照服务端会话：可吊销、可短。JWT 适合短命访问令牌，刷新走服务端。下一课会话管理把 cookie 旗标收全。

```mermaid
flowchart TD
  TOK["JWT"] --> ALG["钉死允许算法"]
  TOK --> SIG["用正确密钥验签"]
  SIG --> CLAIM["iss aud exp nbf"]
  CLAIM --> OK["接受声明"]
```

## 机制

无状态的代价是吊销难。证明在「验签+声明」游戏内；`none` 与错钥是游戏外。OAuth 后课会发这种令牌——先把验法钉死。

<span class="marginnote">数字实例：`exp` 是 Unix 时间戳，比如 `exp = 1767225600` 表示 2026-01-01 零点 UTC 后作废。把有效期写成一年，等于把令牌泄漏后的可利用窗口放大一整年；短命令牌加刷新机制才是常规做法。</span>

```mermaid
flowchart TD
  V["验证库收到令牌"] --> H["读头里的 alg"]
  H -->|"alg = none"| N["跳过验签，直接信声明"]
  H -->|"alg = HS256"| K["用哪个密钥验？"]
  K -->|"误用公钥当 HMAC 密钥"| W["伪造令牌也能通过"]
  K -->|"正确的对称密钥"| R["验签有效"]
  R --> C["再查 iss / aud / exp"]
  C --> OK["接受"]
```

## 边界

本课不写全部 JOSE 算法套件。会话管理下一课：cookie、CSRF、固定。

## 小结

- 信道映射之后，应用身份常是 JWT。
- 钉算法、验签、验 aud/exp；JWS 不保密。
- 胖令牌难吊销；宜短命。
- 下一课会话管理。
- 出处：RFC 7519；RFC 7515；对照 OWASP 对 JWT 的讨论（机制，非利用手册）。
