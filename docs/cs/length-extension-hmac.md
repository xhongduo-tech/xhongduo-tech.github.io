---
title: 长度扩展与 HMAC
date: 2026-09-08
section: cs
---

# 长度扩展与 HMAC

<div class="epigraph">
<p>知道 Merkle–Damgård 的 $H(m)$ 往往等于知道链状态，于是可伪造 $H(m\|\mathrm{pad}\|s)$ 而不知 $m$。HMAC 把密钥以嵌套哈希拌进去，关掉这条，并成为标准 MAC。</p>
<footer>—— Krawczyk, Bellare and Canetti, HMAC, RFC 2104；Bellare, Canetti and Krawczyk 的分析</footer>
</div>

上一课[海绵](/cs/sha3-sponge)从结构上弱化了长度扩展。缺口是：SHA-2 仍是 MD，**裸哈希不能当 MAC**。主干[MAC 与签名](/cs/mac-signature)已点名 HMAC；本课把长度扩展收成机制。不给伪造操作步骤。

后课默认已经读完本课钉下的合同，只补差，不从该领域第一性原理重开。

## 问题

若认证做成 $H(k\|m)$，MD 下敌手可在未知 $m$ 时扩展。$H(m\|k)$ 另有填充与碰撞问题。HMAC：$H(k\oplus\mathrm{opad}\|H(k\oplus\mathrm{ipad}\|m))$。缺口是理解「为何嵌套」，不是再定义 MAC 游戏。

### 海绵也不要用裸 $H(k\|m)$ 凑合

KMAC 按海绵合同拌密钥。算法换了，域分离不能省。

<span class="marginnote">RFC 2104。安全归约依赖压缩函数的 PRF 性质（Bellare 后来的证明）。本课不提供长度扩展的字节级配方。</span>

<span class="marginnote">「MAC（消息认证码）」就是用一把共享密钥给消息盖防伪章的手段：持钥者能验证、无钥者伪造不出。裸哈希只给出摘要、不含密钥，任何人都能对篡改后的消息重算同样的摘要——这就是「裸哈希不能当 MAC」的直观含义。</span>

## 方法

陈述 MD 状态与摘要的关系。给出 HMAC 嵌套。对照 AEAD 内建标签：HMAC 仍用于旧协议、HKDF、许多 API 签名。指出时序上 HMAC 比较应常数时间。

```mermaid
flowchart TD
  K["密钥"] --> IN["内层哈希 ipad"]
  M["消息"] --> IN
  IN --> OUTH["外层哈希 opad"]
  OUTH --> TAG["MAC 标签"]
```

<span class="marginnote">数字实例：SHA-256 输出 32 字节，链状态也是 8 个 32 位字共 32 字节——摘要恰好是「链的快照」，拿到摘要就等于拿到可续算的中间状态，这是 MD 结构下长度扩展成立的根源；海绵结构输出与内部状态不同宽，问题从结构上被弱化。</span>

## 机制

MAC 证明持有 $k$ 的人认过消息。长度扩展说明「抗碰撞哈希」≠「带密钥认证」。HMAC 把 PRF 需求放到哈希上，工程上可换 SHA-256/SHA-3。签名仍是非对称，下一课生日界先回来管碰撞。

<span class="marginnote">常见误区：初学者容易以为「抗碰撞的哈希天然能做认证」。抗碰撞说的是没人能找到两输入同摘要，与密钥无关；认证要求只有持钥者能造出有效标签。两个性质要分别论证，HMAC 的安全证明正是一次独立的归约。</span>

```mermaid
flowchart TD
  A["裸 H 前缀密钥 H(k||m): 密钥先进链"] --> B["输出摘要即暴露链内部状态"]
  B --> C["敌手无需 k 即可续链: 伪造 H(k||m||pad||s) 的标签"]
  C --> D["收件人误认扩展消息已认证"]
  E["HMAC: 内外两层拌密钥"] --> F["外层以含 k 的状态收尾 敌手无从续链"]
  F --> G["扩展伪造失败"]
  D -. "教训" .-> G
```

## 边界

本课不把 Poly1305、CMAC 写全。生日攻击下一课：碰撞的平方根界，既管哈希也管 CTR 分组。

## 小结

- MD 摘要可当链状态，裸 $H(k\|m)$ 不够。
- HMAC 嵌套关长度扩展，是标准 MAC。
- 常数时间比较标签，避免再开预言。
- 下一课生日界：碰撞的定量。
- 出处：RFC 2104；Bellare, Canetti and Krawczyk；对照 FIPS 198-1。
