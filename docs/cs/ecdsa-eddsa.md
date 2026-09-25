---
title: ECDSA / EdDSA
date: 2026-09-08
section: cs
---

# ECDSA / EdDSA

<div class="epigraph">
<p>曲线上的签名：ECDSA 沿用 DSA 的随机 $k$ 合同，$k$ 泄漏或复用即私钥泄漏；Ed25519 把 $k$ 从私钥与消息确定性派生，去掉这口坑。</p>
<footer>—— Johnson, Menezes and Vanstone 对 ECDSA；Bernstein, Duif, Lange, Schwabe and Yang, Ed25519；RFC 8032</footer>
</div>

上一课[ECDH 与 X25519](/cs/ecdh-x25519)用曲线做协商。缺口是**认证**：签名。主干[MAC 与签名](/cs/mac-signature)已分对称与非对称；本课落到曲线族与确定性。不重写证书链。

后课默认已经读完本课钉下的合同，只补差，不从该领域第一性原理重开。

## 问题

ECDSA 验证用公钥点；签名生成需要每次唯一且保密的 nonce $k$。Sony 一类事故是 $k$ 复用——本课只陈述「复用则线性方程出私钥」，不给求解步骤。EdDSA（Ed25519）：$k$ 由哈希确定，去掉 RNG 失败。缺口是签名合同与密钥分离（签钥 ≠ DH 钥）。

<span class="marginnote">术语翻译：nonce 就是「只用一次的数」。签名里的 $k$ 必须每次全新且保密——两枚签名共用同一个 $k$，两条方程一相减私钥就现形；$k$ 可预测同样致命。这不是实现上的小瑕疵，而是数学层面注定翻车的规格级失败，Ed25519 的做法就是干脆把 $k$ 的命运从随机数发生器手里拿回来。</span>

### 规范编码

点压缩与规范签名可防可塑。协议应验规范形式，避免同一逻辑消息多种编码。

<span class="marginnote">RFC 8032。FIPS 186 含 ECDSA。阈值与硬件签名更后。本课不提供从两枚签名提取私钥的程序。</span>

## 方法

对照：ECDSA 要好 RNG；Ed25519 确定性。消息先哈希。域分离：不同协议用不同上下文。指出 JWT 等后课若允许 `alg=none` 是协议误用，不是曲线的锅。

<span class="marginnote">常见误区：把签名理解成「用私钥加密消息」。实际流程是先哈希、再对摘要做曲线运算，签名长度固定、与消息多长无关；Ed25519 把消息直接喂进内部哈希，长消息可以流式分块算。理解成「加密」会把信任模型整个搞错方向。</span>

```mermaid
flowchart TD
  SK["签钥"] --> SIG["签名"]
  MSG["消息哈希"] --> SIG
  K["ECDSA: 随机 k / EdDSA: 派生 k"] --> SIG
  SIG --> VRF["公钥可验"]
```

## 机制

签名服务不可否认与握手认证（证书里的公钥）。ECDH 服务保密密钥。两者都在曲线上，威胁模型不同：签钥泄漏是伪造，DH 长期钥泄漏在无前向保密时是解密历史。CSPRNG 下一课专收「$k$ 从哪来」。

<span class="marginnote">为什么签钥和 DH 钥必须分开？因为泄漏后果完全不同：签名钥丢了，攻击者能长期冒充你；DH 长期钥丢了，历史记录可能被解密。一把钥匙身兼两职，等于把两种风险捆在一起，还让跨协议交叉攻击多出一条现成的路。</span>

同一条曲线上两把钥匙，泄漏后果差在哪？

```mermaid
flowchart TD
  Q["同一曲线族上的两把钥匙"] --> SIGN["签名钥 ECDSA/EdDSA"]
  Q --> DHK["协商钥 ECDH"]
  SIGN --> L1["泄漏后果：可冒充你签任意消息"]
  DHK --> L2["泄漏后果：无前向保密时历史会话可解"]
  L1 --> SEP["语义不混用：签钥不进握手，DH 钥不发证书"]
  L2 --> SEP
```


## 边界

本课不把双线性对和 BLS 插进主干。随机数生成器下一课：不仅服务 ECDSA，也服务 OAEP 种子与 nonce。

## 小结

- ECDH 协商；ECDSA/EdDSA 认证。
- ECDSA 的 $k$ 复用是规格级失败；Ed25519 确定性派生。
- 签钥与 DH 钥分离。
- 下一课 CSPRNG：种子与熵。
- 出处：RFC 8032；Johnson, Menezes and Vanstone；Bernstein et al., Ed25519。
