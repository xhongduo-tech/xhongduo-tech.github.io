---
title: 密码学误用清单
date: 2026-09-08
section: cs
---

# 密码学误用清单

<div class="epigraph">
<p>破掉产品的常常不是新的密码分析，而是 ECB、nonce 复用、裸哈希当 MAC、自制 RNG、把身份写进 JWT 却不验签。清单用来对照合同，不是利用手册。</p>
<footer>—— 据 Anderson, *Security Engineering*；Egele et al. 对 Android 密码学误用的测量；对照 NIST 与 RFC 的合同条款</footer>
</div>

上一课[归约](/cs/provable-security-reduction)说明证明有前提。缺口是把本单元散见的失败收成**检查单**，作为密码学进阶的封口，并交给下一单元「证书链到底验什么」。

后课默认已经读完本课钉下的合同，只补差，不从该领域第一性原理重开。

## 问题

工程上重复出现：ECB 或未认证加密、静态 IV、CBC+可区分填充错误、$H(k\|m)$、MD5 当签名哈希、RSA 无填充、ECDSA 坏 $k$、硬编码密钥、自签还关校验。缺口是清单与「每条对应哪一课合同」，不是新算法。

### 库默认值也是合同

有的 API 默认 ECB 或允许 `None` 算法。选库等于选游戏前提。

<span class="marginnote">Anderson 反复强调实现与流程。学术测量（如 CryptoLint 一类）表明误用普遍。本课不提供扫描目标站点的操作指南。</span>

<span class="marginnote">术语翻译：「游戏前提」是归约证明里的那套规则——攻击者拿到的输入、能问的问题都被约定好。误用等于把规则改了还沿用旧证明：证明只在合同内有效，出了合同什么都没保证。</span>

## 方法

按原语课序对照列出「正确形状」。要求：AEAD、唯一 nonce、HMAC/HKDF、OAEP/PSS、CSPRNG、密钥分离、常数时间比较。指出下一单元从 PKI 验证细节开始，因为「有证书」仍可能什么都没验。

```mermaid
flowchart TD
  L1["选错模式/nonce"] --> FAIL["游戏前提破"]
  L2["裸哈希当认证"] --> FAIL
  L3["坏 RNG / 密钥硬编码"] --> FAIL
  FAIL --> FIX["回到 AEAD 与 HKDF 合同"]
```

<span class="marginnote">数字实例：AES-GCM 的 nonce 是 96 位，同一密钥下只要重用一次，认证密钥就可能被恢复、任意密文可被伪造。nonce 不需要保密，但在同一密钥下绝不能重复——「唯一」比「随机」更重要。</span>

## 机制

误用把可证明对象变回「有加密外观」。审查应问合同而非算法名。协议与身份单元将看到：TLS 配置、JWT `alg`、OAuth 重定向，是同一类失败在协议层的投影。

```mermaid
flowchart TD
  PT["明文含两块相同内容"] --> ECB["ECB: 各块独立加密"]
  ECB --> CT["两块密文也相同, 图案可见"]
  PT --> AE["AEAD: 唯一 nonce 参与每一块"]
  AE --> CT2["相同明文得出不同密文"]
```

<span class="marginnote">常见误区：初学者容易以为「用了 AES 就安全了」。算法名只说明原语本身；模式、nonce、填充、密钥来源任何一环用错，都会把它变回「有加密外观的明文」——所以审查要问合同，不是问算法名。</span>

## 边界

本课不把清单写成攻击脚本。下一课程单元「协议与身份」第一课：证书链验证细节——主干 PKI 课之后仍被跳过的那些检查。

## 小结

- 归约在合同内；本课列合同外高频失败。
- AEAD、nonce、HMAC、填充、RNG、密钥分离。
- 库默认与「有加密」不是证明。
- 下一单元：证书链验证细节。
- 出处：Anderson, *Security Engineering*；Egele et al.；NIST SP 800-38 系列与 RFC 8446。
