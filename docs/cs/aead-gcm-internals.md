---
title: AEAD 与 GCM 内部
date: 2026-09-08
section: cs
---

# AEAD 与 GCM 内部

<div class="epigraph">
<p>AEAD 一次调用同时给出保密与完整性。GCM 用 CTR 加密，用有限域上的 GHASH 吃密文与附加数据；标签错了就该丢整条消息。</p>
<footer>—— McGrew and Viega, The Galois/Counter Mode；NIST SP 800-38D</footer>
</div>

上一课[ChaCha20](/cs/chacha20)给出软件密钥流。主干[分组模式](/cs/block-modes)已把 GCM 当 AEAD 名字。缺口是**内部**：CTR 与 GHASH 如何咬合，附加数据（AAD）为何不加密却进标签。[MAC 与签名](/cs/mac-signature)的 Encrypt-then-MAC 在此收成单一原语。

后课默认已经读完本课钉下的合同，只补差，不从该领域第一性原理重开。

## 问题

只加密的 CBC 留下主动篡改。GCM：密钥流来自 AES-CTR（或等价），认证来自 $ \mathrm{GF}(2^{128}) $ 上的多项式求值 GHASH，最后再加密得到标签。AAD 让头部明文仍被认证。缺口是理解「一次 nonce、一块标签」的合同，不是再背套件名。

### 标签失败即拒绝

解密先验标签；失败则不释放明文。把失败当「再试试填充」会重开预言机——下一课。

<span class="marginnote">SP 800-38D 限制同一密钥下 nonce 不得重复。96 比特 nonce 最常见。GHASH 不是抗碰撞哈希；它是带密钥的通用哈希，安全靠密钥保密与 nonce 唯一。</span>

<span class="marginnote">术语翻译：AAD（附加数据）就是用「不加密、但进标签」的手段来做「头部可读却不可篡改」的事——比如 TLS 的记录头必须明文传输让网络设备看懂，又不许中间人改一个比特：改了，标签对不上，整条记录作废。</span>

## 方法

画：J0 从 nonce 来，CTR 加密块，GHASH 扫 AAD 与密文，再与 E(K,J0) 混合成标签。对照 ChaCha20-Poly1305：同一 AEAD 形状，多项式不同。指出 TLS 1.3 只保留 AEAD。

```mermaid
flowchart TD
  N["nonce"] --> CTR["AES-CTR 密钥流"]
  PT["明文"] --> CTR
  CTR --> CT["密文"]
  AAD["附加数据"] --> GH["GHASH"]
  CT --> GH
  GH --> T["认证标签"]
```

## 机制

AEAD 把[CIA](/cs/cia-triad)的 C 与 I 绑到同一密钥与同一 nonce。主动改密文或 AAD 会使标签失败。它仍假设密钥未泄漏、实现常数时间、nonce 不重复。公钥与身份不在本课。

方法一节的图回答「发送方怎么把保密与认证拧在一起」，下面这张回答另一个问题：接收方收到一条消息后，标签校验与解密的先后与分支。

```mermaid
flowchart TD
  IN["收到: 密文 + AAD + 标签"] --> GH["用同一密钥与 nonce 重算 GHASH"]
  GH --> MIX["与 E(K, J0) 混合得候选标签"]
  MIX --> CMP{"与收到的标签一致?"}
  CMP -- "否" --> REJ["整条丢弃, 不释放任何明文"]
  CMP -- "是" --> DEC["AES-CTR: 同一密钥流异或还原明文"]
```

<span class="marginnote">数字实例：128 比特标签意味着攻击者盲改一次密文蒙混过关的概率约 $2^{-128}\approx 3\times 10^{-39}$——比连中十几次彩票还稀罕得多。但nonce 只有 96 比特且不得重复：NIST 建议同一密钥加密次数不超过 $2^{32}$，靠生日界保证 nonce 不撞。</span>

<span class="marginnote">常见误区：初学者容易把「标签失败」当成另一种「解密出错」去细究原因、返回不同报错——这条差异路径正是下一课填充预言机攻击的原料。正确动作只有一种：标签不对，整条丢掉，错误信息与别的一模一样。</span>

## 边界

本课不写 GHASH 的全部域算术作业，不给 nonce 复用下的密钥恢复步骤。下一课专收 **nonce 误用与重放**：合同一旦破，GCM 的证明作废。

## 小结

- 流密码与分组密码都要 AEAD 包装。
- GCM：CTR 保密，GHASH 认证，AAD 进标签。
- 标签失败则丢消息；不要当填充预言。
- nonce 唯一是下一课的主缺口。
- 出处：McGrew and Viega；NIST SP 800-38D；RFC 5116 AEAD 接口。
