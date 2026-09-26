---
title: SHA-3 海绵
date: 2026-09-08
section: cs
---

# SHA-3 海绵

<div class="epigraph">
<p>海绵把一个宽置换反复用：先吸收消息进状态，再挤出摘要。输出不必等于整个内部状态，长度扩展那类 MD 遗产从结构上被拆掉。</p>
<footer>—— Bertoni, Daemen, Peeters and Van Assche, Cryptographic sponge functions；NIST FIPS 202</footer>
</div>

上一课[Merkle–Damgård](/cs/merkle-damgard)的链接值即摘要前缀。缺口是**另一族**：Keccak 海绵，SHA-3 与 SHAKE。不重写碰撞三条性质，只换构造。

后课默认已经读完本课钉下的合同，只补差，不从该领域第一性原理重开。

## 问题

MD 的最终摘要往往就是链状态（或截断但仍泄漏可继续的状态）。海绵：状态分比率 $r$ 与容量 $c$，吸收阶段把消息块异或进 $r$，再做置换 $f$；挤出阶段读 $r$。容量不直接输出，长度扩展不再「接着压一块」。缺口是海绵的吸收/挤出，不是再背 SHA-256 圈数。

### SHAKE 是 XOF

挤出可任意长，当 PRG 式派生时要换域分离，避免与哈希同一上下文。

<span class="marginnote">FIPS 202。Keccak 置换是 1600 比特。本课不把 SHA-3 写成「SHA-2 被破所以必须换」——SHA-2 仍广泛安全；SHA-3 是结构多样性。</span>

<span class="marginnote">数字实例：Keccak-f[1600] 的状态共 1600 比特。SHA3-256 取比率 $r=1088$、容量 $c=512$——输出 256 比特时抗碰撞强度是 $2^{128}$ 次操作，因为碰撞必须同时打穿容量；想要更强的 SHAKE128/256，只是换一对 $r/c$ 切分，置换本身不变。</span>

## 方法

画吸收循环与挤出循环。对比 MD 的单向链。指出 SHAKE128/256 的可变输出。HMAC 对 SHA-3 仍可用，但海绵有 KMAC 更贴结构。

```mermaid
flowchart TD
  M["消息块"] --> ABS["吸收: ⊕进比率再置换"]
  ABS --> ABS
  ABS --> SQ["挤出: 读比率再置换"]
  SQ --> DIG["摘要或 XOF"]
```

## 机制

容量 $c$ 对应安全边际：内部有不输出的部分。这服务抗碰撞与原像，前提是 $f$ 像随机置换。它不自动给 MAC；密钥如何拌进海绵，是域分离问题，下一课与 HMAC 一起收。

```mermaid
flowchart LR
  subgraph MD["Merkle-Damgard 链"]
    A["块1"] --> B["链值直接当摘要前缀"]
    B --> C["攻击者拿到 h 可续写新块"]
  end
  subgraph SP["海绵构造"]
    D["块异或进比率 r"] --> E["容量 c 从不输出"]
    E --> F["只读 r 挤出, 完整状态不可得"]
  end
```

<span class="marginnote">直觉类比：容量 $c$ 像沉在水下的底仓——装卸货（消息进出）都只发生在水面（比率 $r$），旁观者永远看不到底舱里装了什么。攻击者拿不到完整内部状态，「接着压一块」的长扩展戏法就没了舞台。</span>

## 边界

本课不写 Keccak 圈函数的 θρπχι 全套。长度扩展与 HMAC 下一课专门关 MD 遗产，并说明为何「$H(k\|m)$」不够。

<span class="marginnote">常见误区：初学者容易以为 SHA-3 安全是因为它的圈函数更复杂；其实起决定作用的是构造层面的隔离——容量部分从不输出。圈函数 $f$ 再花哨，若把整个状态亮出来，长度扩展照样成立。</span>

## 小结

- MD 链状态等于可续写的内部。
- 海绵：吸收/挤出，容量不直接输出。
- SHA-3 与 SHAKE 是 FIPS 202；与 SHA-2 并存。
- 下一课：长度扩展为何逼出 HMAC。
- 出处：Bertoni et al., sponge；NIST FIPS 202。
