---
title: RSA 与 OAEP
date: 2026-09-08
section: cs
---

# RSA 与 OAEP

<div class="epigraph">
<p>RSA 函数是模幂；直接加密短消息既不确定也不抗选择密文。OAEP 用哈希做随机化填充，把 RSA 收成在随机预言机模型下可达到 CCA 目标的信封零件。</p>
<footer>—— Rivest, Shamir and Adleman, 1978；Bellare and Rogaway, Optimal Asymmetric Encryption Padding, EUROCRYPT 1994；PKCS #1 v2</footer>
</div>

上一课[生日攻击](/cs/birthday-attack)钉了哈希宽度。主干[公钥信封](/cs/pubkey-envelope)与[DH 与 RSA 分工](/cs/dh-vs-rsa)已把 RSA 当封装选项。缺口是**明文不能直接当模幂输入**：必须填充。本课收 OAEP，不重推欧拉定理。

后课默认已经读完本课钉下的合同，只补差，不从该领域第一性原理重开。

## 问题

教科书 RSA $c=m^e\bmod n$ 确定、可乘性，选择密文下可被操纵。OAEP：用随机种子和哈希把 $m$ 扩成类似一次一密的掩码块再模幂。TLS 1.3 已不用 RSA 密钥传输，但签名与遗留封装仍要理解填充。缺口是填充合同，不是再讲因数分解叙事。<span class="marginnote">术语翻译：OAEP（最优非对称加密填充）就是用「真随机种子加两轮哈希掩码，把消息搅成看不出结构的一团」的手段来做「同一条消息每次密文都不同、密文改一个比特就整块作废」的事——RSA 本体只负责搬这团东西。</span>

### 签名填充是另一套

PSS 与 PKCS#1 v1.5 签名填充下一课攻击面才对照。加密填充与签名填充不能混用同一密钥语义而不加域分离。

<span class="marginnote">Bellare–Rogaway OAEP。Fujisaki–Okamoto 等是同类「把陷门置换收成 CCA」的框架，本课只钉 OAEP。不要发明 arXiv 编号。</span>

## 方法

写 RSA 陷门置换角色：公钥 $(n,e)$ 公开，私钥 $d$ 解幂。OAEP 的两轮 Feistel 式掩码（用 $G,H$ 哈希）。对照：哈希宽度受生日界约束，种子必须真随机。<span class="marginnote">数字实例：若种子只有 16 位随机，攻击者枚举 65536 个候选就能把密文重放出明文；真实方案里种子宽如哈希输出（如 256 位），穷举要 $2^{256}$ 次——填充的随机性上限就是种子的随机性，种子弱则整套 OAEP 形同虚设。</span>

```mermaid
flowchart TD
  M["消息"] --> OAEP["OAEP 随机化填充"]
  R["随机种子"] --> OAEP
  OAEP --> POW["模幂 e"]
  POW --> C["密文"]
```

## 机制

信封仍是：RSA 只封对称键，数据走 AEAD。OAEP 让封装在理想哈希下可归约到 RSA 陷门。实现必须常数时间模幂与严格的填充校验——校验失败不可区分，否则又是预言机。<span class="marginnote">常见误区：初学者容易以为「解封失败时返回不同的错误码方便排查」——每一种可分辨的失败提示都是一台免费预言机；正确做法是把所有失败统一成同一句「解封失败」，连报错耗时也要抹平。</span>

```mermaid
flowchart TD
  C2["收到密文"] --> DEC["私钥 d 解幂"]
  DEC --> UNMASK["反向拆两轮掩码 H 与 G"]
  UNMASK --> CHK{"种子与掩码自洽？"}
  CHK -- "是" --> OUT["交出被封装的对称密钥"]
  CHK -- "否" --> FAIL["统一报 解封失败"]
  FAIL --> ND["错误不可区分：堵死预言机"]
```

## 边界

本课不给选择密文操纵的步骤。下一课 RSA 攻击面：填充实现、共模、广播、小指数，全部当失败模式与防御，不当配方。

## 小结

- 生日界约束填充里的哈希；RSA 要随机化填充。
- OAEP 把陷门置换收成 CCA 向的封装。
- TLS 1.3 不用 RSA 传输密钥；遗留与签名仍要填充。
- 下一课：填充与数论误用的攻击面清单。
- 出处：Rivest, Shamir and Adleman, 1978；Bellare and Rogaway, 1994；PKCS #1。
