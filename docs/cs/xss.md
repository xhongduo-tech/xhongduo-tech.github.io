---
title: XSS
date: 2026-09-08
section: cs
---

# XSS

<div class="epigraph">
<p>脚本与数据若在 HTML/JS 上下文里分不开，攻击者的字符串就会在受害者浏览器里以该源的身份执行。这是注入课在 Web 上的实例：完整中介失败在页面拼接。</p>
<footer>—— 据 OWASP XSS；CWE-79；对照[同源与 CSRF](/cs/same-origin-csrf)、[注入](/cs/injection-boundary)</footer>
</div>

上一课[二进制加固](/cs/binary-hardening-obfuscation)结束软件安全。Web 单元从浏览器开始。主干同源课管 Cookie；缺口是**脚本注入**：反射、存储、DOM。不给可复制的恶意脚本。

后课默认已经读完本课钉下的合同，只补差，不从该领域第一性原理重开。

## 问题

模板把用户输入写进 HTML。若未按上下文编码，输入可变成标签或事件处理器。存储型让每位访客执行。缺口是上下文编码与可信类型，不是过滤尖括号清单。

### 不是只防 script 标签

事件属性、CSS、URL 方案都是上下文。编码函数必须匹配上下文。

<span class="marginnote">术语翻译：「上下文编码」就是把用户输入翻译成"纯文字"再交给页面：在 HTML 文本里把 <code>&lt;</code> 变成 <code>&amp;lt;</code>，在属性值里把引号变成数字实体。同一个字符在不同上下文要变不同的样子——"一个转义函数走天下"是常见误区。</span>

<span class="marginnote">OWASP。本课禁止 XSS payload 教程。CSP 下一课是纵深，不是根治。</span>

## 方法

分三种 XSS。防御：模板自动转义、Trusted Types、HttpOnly 不阻止 XSS 操 DOM。下一课 CSP 限制脚本源。

<span class="marginnote">数字实例：三种型的差别在"注入点"。反射型的串躺在 URL 参数里；存储型的串进了数据库；DOM 型的串被前端 JS 自己拼进页面。前两种的脏数据经过服务器，第三种可能全程不离开浏览器。</span>

```mermaid
flowchart TD
  IN["不可信串"] --> CTX["HTML/JS/URL 上下文"]
  CTX --> EXEC["以该源执行"]
  IN --> ESC["按上下文编码"]
```

## 机制

SOP 把脚本权限给源。XSS 等于把权限借给输入。会话 Cookie 可被同源于脚本读（除非 HttpOnly，但仍可发请求）。

<span class="marginnote">直觉类比：源像一间上了锁的办公室，脚本是持全权工牌的员工。XSS 就是攻击者把你的一段话伪装成"工牌"刷进门——门禁没坏、锁也没坏，坏的是允许把外来文字当员工放行。后续 CSP 是给工牌发证机关加白名单。</span>

```mermaid
flowchart TD
  ATK["攻击者投放恶意串"] --> Q{"串停在哪里"}
  Q --> R["URL 参数：反射型"]
  Q --> DB["数据库：存储型"]
  Q --> JS["前端变量：DOM 型"]
  R --> SRV["服务端拼进响应页"]
  DB --> EVERY["每位访问者触发一次"]
  JS --> DOM["JS 直接拼进 DOM"]
  SRV --> BROWSER["受害者浏览器执行"]
  EVERY --> BROWSER
  DOM --> BROWSER
  BROWSER --> ACT["以该源身份发请求改页面"]
```

## 边界

零 payload。CSP 下一课。

## 小结

- 内存利用之后，Web 的经典注入是 XSS。
- 按上下文编码；过滤清单不够。
- XSS 借用源的权限。
- 下一课 CSP。
- 出处：OWASP XSS；CWE-79；[injection-boundary](/cs/injection-boundary)。
