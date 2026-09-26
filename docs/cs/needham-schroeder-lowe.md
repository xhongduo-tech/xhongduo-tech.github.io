---
title: Needham–Schroeder 与 Lowe
date: 2026-09-08
section: cs
---

# Needham–Schroeder 与 Lowe

<div class="epigraph">
<p>Needham–Schroeder 公钥协议看起来互相证明了新鲜 nonce；Lowe 指出缺少身份绑定，第三者可以把会话转包。修复是在消息里显式写上对端名字。</p>
<footer>—— Needham and Schroeder, 1978；Lowe, Breaking and Fixing the Needham-Schroeder Public-Key Protocol, TACAS 1996</footer>
</div>

上一课[ProVerif/Tamarin](/cs/protocol-verification)要一个经典标本。缺口就是 **NS 公钥协议与 Lowe 修正**：认证对应性失败的最小例子。软件安全下一单元从这里离开协议逻辑，进入内存。

后课默认已经读完本课钉下的合同，只补差，不从该领域第一性原理重开。

## 问题

A 与 B 交换加密 nonce。攻击者 I 让 A 以为在和 I 谈，同时让 B 以为在和 A 谈——只陈述身份错绑。Lowe：在加密载荷里加入对端身份。缺口是「认证要钉住是谁」，不是 nonce 算术作业。

<span class="marginnote">直觉类比：NS 协议的密文像一份没写抬头的合同——只有签字和防伪暗号（nonce），没写「本人承诺交给谁」。中介截下签字页，转手附到自己与第三方的合同上，双方各执一份都对得上暗号，却根本没和彼此成交。</span>

<span class="marginnote">常见误区：初学者以为「每条消息都加密、nonce 都新鲜，协议就安全」。NS 公钥协议每一步单看都无懈可击，漏洞出在消息的**组合**层面——密文里没写承诺对象是谁。这类洞只有把协议当整体建模（对应性检查）才找得到。</span>

### 符号攻击≠实现包

本课不给报文级利用。要的是：设计时把身份放进认证消息，再用工具查对应性。

<span class="marginnote">Needham–Schroeder 1978 还有对称密钥版与服务器。Lowe 1996 针对公钥版。这是协议验证课的标准例。</span>

## 方法

写三消息形状与 Lowe 的字段补丁。对照 TLS 的签名 Finished 与证书名：同一思想——名字绑进密钥交换。指出软件漏洞不在此模型里。

<span class="marginnote">术语翻译：「对应性认证」就是账要两头对得上——A 的会话记录写「我与 B 完成对话」，B 那边必须也有一条「与 A」的对应记录。攻击得手时 A 记的是 I、B 记的是 A，两头对不上，自动工具就据此报协议失败。</span>

```mermaid
flowchart TD
  NS["NS 公钥交换"] --> GAP["身份未绑进密文"]
  GAP --> CONF["对应性失败"]
  LOWE["密文内写入对端名"] --> FIX["Lowe 修正"]
```

## 机制

身份单元封口：令牌、会话、联邦、因素、形式化，最后用 NSL 说明最小补丁。下一课程单元软件安全：栈上的比特如何破坏「进程按源码语义执行」这一假设。

```mermaid
flowchart TD
  A1["A → I：加密给 I 的 Na"] --> I1["I 解出 Na，冒充 A 发给 B"]
  I1 --> B1["B → 『A』：加密给 A 的 Na, Nb"]
  B1 --> MID["I 截下，谎称是自己的回应"]
  MID --> A2["A 解出 Nb，以为在与 I 会话"]
  A2 --> LEAK["A → I：加密给 I 的 Nb"]
  LEAK --> WIN["I 解出 Nb，向 B 完成冒充"]
```

## 边界

本课不把 Kerberos 全部消息重写（Windows 课更后）。下一课栈溢出到 shellcode：空间安全的开口，只讲机制与防御，不给可运行载荷。

## 小结

- 工具课需要标本：NS 公钥协议的认证洞。
- Lowe：把对端身份写进认证密文。
- 对应性是「谁与谁完成了会话」。
- 下一单元：栈溢出与内存破坏（防御视角）。
- 出处：Needham and Schroeder, 1978；Lowe, 1996。
