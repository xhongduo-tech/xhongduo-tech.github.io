---
pageClass: plain-doc
---

# 密码学与信息安全

密码学是信息安全的数学根基，信息安全则是密码学在系统与网络中的工程落地。本分类对标 William Stallings《密码编码学与网络安全——原理与实践》（Cryptography and Network Security: Principles and Practice）的章节体系，覆盖从古典密码到后量子密码、从协议到系统安全的完整知识链路。

## 主题规划

<ProgressGrid cat="cs/cryptography-security" />


### 第一篇 古典密码与密码分析基础

- [x] [密码学的基本概念：明文、密文、密钥与密码体制五元组](./basic-concepts)
- [x] [攻击模型与威胁分类：被动攻击与主动攻击](./attack-models-threats)
- [x] [科克霍夫原则（Kerckhoffs's Principle）与现代密码设计假设](./kerckhoffs-principle)
- [x] [单表替换密码：凯撒密码、仿射密码与任意单表替换](./monoalphabetic-ciphers)
- [x] [单表替换密码的统计分析：字母频率攻击](./frequency-analysis)
- [x] [多表替换密码：维吉尼亚密码（Vigenère Cipher）](./polyalphabetic-ciphers-vigenere)
- [x] [多表替换密码的分析：卡西斯基试验与重合指数](./kasiski-test-coincidence-index)
- [x] [置换密码：栅栏密码与列置换密码](./transposition-ciphers)

### 第二篇 信息论安全与一次一密

- [x] [香农保密系统理论：熵、条件熵与疑义度](./shannon-entropy-equivocation)
- [x] [完善保密性（Perfect Secrecy）的定义与判定](./perfect-secrecy)
- [x] [一次一密（One-Time Pad）及其安全性证明](./one-time-pad)
- [x] [一次一密的工程局限与唯密文攻击的意义](./one-time-pad-limitations)
- [x] [无条件安全与计算安全的分野](./unconditional-vs-computational-security)

### 第三篇 分组密码（Block Cipher）

- [x] [分组密码的设计思想：混淆与扩散](./block-cipher-design-confusion-diffusion)
- [x] [Feistel 网络结构及其可逆性](./feistel-network)
- [x] [DES 的整体结构：初始置换、16 轮迭代与逆置换](./des-overall-structure)
- [x] [DES 的轮函数：扩展置换、S 盒与 P 置换](./des-round-function)
- [x] [DES 的安全性：弱密钥、差分分析与线性分析](./des-security)
- [x] [多重 DES 与中间相遇攻击：2DES 为何无效、3DES 的构造](./multi-des-meet-in-the-middle)
- [x] [AES 的数学基础：有限域 GF(2^8) 上的运算](./aes-finite-field)
- [x] [AES 的轮变换：字节代换、行移位、列混合与轮密钥加](./aes-round-transformations)
- [x] [AES 的密钥扩展算法与安全性分析](./aes-key-expansion)
- [x] [分组密码工作模式：ECB 与 CBC](./block-cipher-modes-ecb-cbc)
- [x] [分组密码工作模式：CFB、OFB 与 CTR](./block-cipher-modes-cfb-ofb-ctr)
- [x] [认证加密模式：GCM 与 CCM](./aead-gcm-ccm)
- [x] [分组密码的填充方案与填充预言攻击（Padding Oracle）](./padding-oracle)

### 第四篇 流密码（Stream Cipher）

- [x] [流密码的基本模型：密钥流生成器](./stream-cipher-basic-model)
- [x] [线性反馈移位寄存器（LFSR）及其周期特性](./lfsr)
- [x] [LFSR 的组合生成与非线性滤波](./lfsr-combination-nonlinear-filter)
- [x] [RC4 算法及其偏差攻击](./rc4)
- [x] [ChaCha20 流密码与现代流密码设计](./chacha20)

### 第五篇 公钥密码的数论基础

- [x] [整除性、欧几里得算法与扩展欧几里得算法](./euclidean-algorithm-extended)
- [x] [模运算与同余：剩余类环 Zn](./modular-arithmetic-zn)
- [x] [费马小定理与欧拉定理](./fermat-euler-theorem)
- [x] [素性测试：Miller-Rabin 算法](./miller-rabin)
- [x] [离散对数问题与循环群](./discrete-logarithm-cyclic-groups)
- [x] [中国剩余定理（CRT）及其在密码学中的应用](./chinese-remainder-theorem)

### 第六篇 公钥密码体制（Public-Key Cryptography）

- [x] [公钥密码的思想起源：单向函数与陷门单向函数](./public-key-origin-trapdoor)
- [x] [RSA 算法：密钥生成、加密与解密](./rsa-algorithm)
- [x] [RSA 的正确性证明与安全性假设](./rsa-correctness-security)
- [x] [RSA 的实现优化：CRT 加速与低指数攻击](./rsa-implementation-optimization)
- [x] [RSA 填充方案：PKCS#1 v1.5 与 OAEP](./rsa-padding-oaep)
- [x] [Diffie-Hellman 密钥交换协议](./diffie-hellman)
- [x] [ElGamal 加密体制](./elgamal)
- [x] [椭圆曲线基础：椭圆曲线群与椭圆曲线离散对数问题](./elliptic-curve-basics)
- [x] [椭圆曲线密码（ECC）：ECDH 密钥交换](./ecc-ecdh)
- [x] [ECC 与 RSA 的对比：密钥长度与安全强度](./ecc-vs-rsa)

### 第七篇 哈希函数与消息认证

- [x] [密码学哈希函数的性质：抗原像、抗第二原像与抗碰撞](./hash-function-properties)
- [x] [迭代式哈希结构：Merkle-Damgård 构造](./merkle-damgard)
- [x] [MD5 与 SHA-1 的碰撞攻击](./md5-sha1-collision)
- [x] [SHA-2 系列：SHA-256 与 SHA-512](./sha2)
- [x] [SHA-3 与海绵结构（Sponge Construction）](./sha3-sponge)
- [x] [生日悖论与生日攻击](./birthday-paradox)
- [x] [消息认证码（MAC）的原理](./mac-principles)
- [x] [HMAC 的构造与安全性](./hmac)
- [x] [基于哈希的长度扩展攻击](./length-extension-attack)

### 第八篇 数字签名（Digital Signature）

- [x] [数字签名的概念与安全需求](./digital-signature-concepts)
- [x] [RSA 数字签名方案](./rsa-digital-signature)
- [x] [ElGamal 数字签名方案](./elgamal-digital-signature)
- [x] [数字签名标准 DSA](./dsa)
- [x] [椭圆曲线数字签名算法 ECDSA](./ecdsa)
- [x] [EdDSA 与 Ed25519](./eddsa-ed25519)
- [x] [签名的不可伪造性：EUF-CMA 安全模型](./euf-cma-security)

### 第九篇 密钥分配与公钥基础设施（PKI）

- [x] [对称密钥分配问题与密钥分配中心（KDC）](./symmetric-key-distribution-kdc)
- [x] [Needham-Schroeder 协议](./needham-schroeder)
- [x] [Kerberos 认证体系](./kerberos)
- [x] [公钥证书与 X.509 标准](./x509-certificates)
- [x] [证书链、信任模型与证书颁发机构（CA）](./certificate-chains-ca)
- [x] [证书吊销：CRL 与 OCSP](./certificate-revocation-crl-ocsp)

### 第十篇 认证协议

- [x] [身份认证的基本方式：口令、令牌与生物特征](./authentication-factors)
- [x] [口令存储：加盐哈希与慢哈希函数（PBKDF2、bcrypt、Argon2）](./password-storage-salted-hash)
- [x] [挑战-响应认证协议](./challenge-response-authentication)
- [x] [重放攻击及其防护：随机数、时间戳与序列号](./replay-attack-defense)
- [x] [中间人攻击与认证的密钥交换](./mitm-authenticated-key-exchange)
- [x] [多因子认证（MFA）与 FIDO2/WebAuthn](./mfa-fido2-webauthn)

### 第十一篇 TLS/SSL 详解

- [x] [SSL/TLS 的历史与版本演进](./tls-history)
- [x] [TLS 握手协议：密钥协商与参数协商](./tls-handshake)
- [x] [TLS 记录协议：加密与完整性保护](./tls-record-protocol)
- [x] [TLS 1.2 与 TLS 1.3 的差异](./tls12-vs-tls13)
- [x] [HTTPS 与证书校验流程](./https-certificate-validation)
- [x] [TLS 的著名攻击：BEAST、POODLE、Heartbleed 与降级攻击](./tls-attacks)

### 第十二篇 网络安全

- [x] [防火墙：包过滤、状态检测与应用层网关](./firewall)
- [x] [入侵检测系统（IDS）与入侵防御系统（IPS）](./ids-ips)
- [x] [拒绝服务攻击（DoS）与分布式拒绝服务攻击（DDoS）](./dos-ddos)
- [x] [VPN 原理：IPSec 协议族](./vpn-ipsec)
- [x] [TLS VPN 与 WireGuard](./tls-vpn-wireguard)
- [x] [无线网络安全：WEP 的缺陷与 WPA2/WPA3](./wireless-security-wep-wpa)
- [x] [电子邮件安全：PGP 与 S/MIME](./email-security-pgp-smime)

### 第十三篇 软件安全

- [x] [缓冲区溢出：栈溢出的原理与利用](./buffer-overflow)
- [x] [缓冲区溢出的防护：栈金丝雀、DEP/NX 与 ASLR](./buffer-overflow-defenses)
- [x] [格式化字符串漏洞与整数溢出](./format-string-integer-overflow)
- [x] [SQL 注入：原理、利用与参数化查询防护](./sql-injection)
- [x] [跨站脚本攻击（XSS）：反射型、存储型与 DOM 型](./xss)
- [x] [跨站请求伪造（CSRF）及其防护](./csrf)
- [x] [反序列化漏洞与命令注入](./deserialization-command-injection)
- [x] [软件供应链安全：依赖混淆、投毒与 SBOM](./software-supply-chain-security)

### 第十四篇 系统安全

- [x] [操作系统的权限模型：用户、组与访问控制表](./os-permission-model)
- [x] [自主访问控制（DAC）与强制访问控制（MAC）](./dac-mac)
- [x] [权能（Capability）与最小特权原则](./capability-least-privilege)
- [x] [沙箱机制：chroot、命名空间与 seccomp](./sandboxing-chroot-namespaces-seccomp)
- [x] [可信计算：TPM 与可信启动](./tpm-trusted-boot)
- [x] [恶意软件：病毒、蠕虫、木马与勒索软件](./malware-types)

### 第十五篇 隐私保护技术

- [x] [匿名化与 k-匿名：去匿名化攻击](./k-anonymity-deanonymization)
- [x] [差分隐私（Differential Privacy）的基本概念](./differential-privacy)
- [x] [差分隐私的机制：拉普拉斯机制与指数机制](./differential-privacy-mechanisms)
- [x] [同态加密（Homomorphic Encryption）：半同态与全同态](./homomorphic-encryption)
- [x] [安全多方计算（MPC）与秘密共享](./mpc-secret-sharing)
- [x] [零知识证明（Zero-Knowledge Proof）初步：交互式证明与 Schnorr 协议](./zero-knowledge-proofs)

### 第十六篇 后量子密码（Post-Quantum Cryptography）

- [x] [量子计算的威胁：Shor 算法与 Grover 算法](./quantum-threat-shor-grover)
- [x] [后量子密码的主要技术路线概览](./post-quantum-routes-overview)
- [x] [基于格的密码：LWE 问题与 Kyber](./lattice-crypto-lwe-kyber)
- [x] [基于哈希的签名：Lamport 签名与 SPHINCS+](./hash-based-signatures-lamport-sphincs)
- [x] [NIST 后量子密码标准化进程](./nist-pqc-standardization)
- [x] [混合模式与密码敏捷性（Crypto-Agility）](./hybrid-crypto-agility)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
