---
title: HKDF
date: 2026-09-08
section: cs
---

# HKDF

<div class="epigraph">
<p>HKDF 先用 HMAC 提取，把不匀的输入密钥材料收成伪随机密钥，再扩展出多条用途分离的输出。它不是慢哈希，也不从弱口令里变出熵。</p>
<footer>—— Krawczyk, Cryptographic Extraction and Key Derivation, 2010；RFC 5869</footer>
</div>

## 定位

上一课[CSPRNG](/cs/csprng)从熵池抽新鲜随机。缺口是**已经有共享秘密**（ECDH 输出、PSK）时如何派生 AEAD 键、IV 盐、下一跳棘轮键。主干[KDF 与慢哈希](/cs/kdf-slow-hash)管口令；本课 HKDF 管高熵 IKM。

后课默认已经读完本课钉下的合同，只补差，不从该领域第一性原理重开。

## 问题

ECDH 的共享坐标不是均匀密钥。直接截断当 AES 键会留结构。HKDF-Extract(salt, IKM)→PRK，HKDF-Expand(PRK, info, L)→OKM。`info` 做域分离：同一 IKM 不能在两条协议里撞键。缺口是提取–扩展，不是再讲 bcrypt。

### 不能当口令哈希

口令熵低，要慢哈希与盐。HKDF 快，假设 IKM 已有足够熵。


<span class="marginnote">RFC 5869。TLS 1.3 的 HKDF-Expand-Label 是加了标签的扩展。Signal 棘轮更后用同一思想。</span>

## 方法

画 Extract 与 Expand。强调 salt 与 info 的角色。对照：ChaCha/AES 的密钥从 OKM 切开，nonce 另计。指出长度扩展已由 HMAC 关。

```mermaid
flowchart TD
  IKM["ECDH 等 IKM"] --> EX["HKDF-Extract"]
  SALT["盐"] --> EX
  EX --> PRK["PRK"]
  PRK --> EXP["HKDF-Expand + info"]
  EXP --> OKM["用途分离的键"]
```

图中节点是本课的机制骨架；课程不把图展开成可运行的攻击步骤。

## 机制

域分离让「同一共享秘密」长出握手密钥、应用密钥、导出密钥而不串用。这是协议级构造的零件。密钥如何存放、谁能碰，下一课 HSM 与生命周期——算法再对，内存里的明文键仍是威胁模型里的端点。

前提写进合同之后，游戏外的误用只当失败模式点名，不在本课写成操作程序。

## 边界

本课不把 Argon2 重写一遍。原语课序到 HKDF 收口；下一课序协议级构造从密钥管理与 HSM 起，问「键在哪」。

## 小结

- CSPRNG 产熵；HKDF 从已有 IKM 提取并扩展。
- info 做域分离；不能当慢口令哈希。
- TLS 1.3 与棘轮都吃 HKDF。
- 下一课序：密钥管理与 HSM。
- 出处：Krawczyk, 2010；RFC 5869；对照 RFC 8446。
