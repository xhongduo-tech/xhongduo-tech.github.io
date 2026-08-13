---
title: 公钥基础设施 PKI 与数字证书
date: 2026-08-07
---

# 公钥基础设施 PKI 与数字证书

<div class="epigraph">
<p>密码学保证「只有你能签名」，但保证不了「你真的是你」——那是信任基础设施的事。</p>
<footer>—— 公钥基础设施实践的通识表述</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机安全综合 ｜ Stallings《Computer Security: Principles and Practice》第2章 ｜ 2026-08-07</p>
</div>

## 为什么从 PKI 开始

上一篇的数字签名解决了「消息是谁签的」，但留下一个悬而未决的前提：**验证者拿到的公钥，真的属于声称的主体吗？** 攻击者可以伪造一个「银行官网」的公钥，用它签名、用它解密——只要受害者的浏览器信了，密码学再强也白搭。**公钥基础设施（PKI）** 就是回答「公钥怎么可信地绑定到身份」的信任体系。<span class="marginnote">这是密码学从「数学安全」走向「系统工程」的关键一步：算法可以做到无懈可击，但「信任谁、由谁背书、如何撤销」是社会与治理问题。PKI 的骨干标准是 X.509，HTTPS 的「小锁头」背后就是整套 PKI。</span>

## 1 公钥认证的核心难题

上一篇的结束处埋了一个问题：**公钥分发被中间人攻击**。设想 Bob 想给 Alice 发加密消息：

1. Alice 把公钥 $K_A$ 发给 Bob——但 Mallory 在中间截获，换成自己的公钥 $K_M$；
2. Bob 用 $K_M$ 加密发给「Alice」，Mallory 用自己的私钥解密，再换成 Alice 的公钥重发；
3. Alice 解密成功，但机密已被 Mallory 读走，且双方毫无察觉。

这就是**中间人攻击（MITM, man-in-the-middle attack）**——它不攻破任何算法，只破坏「公钥归属」的信任。<span class="marginnote">解决公钥认证有三条路：① 公钥指纹直接核对（SSH 首次连接的 `known_hosts` 提示，TOFU——trust on first use）；② 面对面交换（PGP 密钥签名聚会）；③ <strong>证书授权机构（CA）背书</strong>——本篇的主角。HTTPS 走第③条路。</span>

## 2 数字证书：把公钥与身份绑定的「身份证」

**数字证书（digital certificate）**：由可信第三方（CA）用其私钥签名的数据结构，声明「公钥 $K$ 属于身份 $X$