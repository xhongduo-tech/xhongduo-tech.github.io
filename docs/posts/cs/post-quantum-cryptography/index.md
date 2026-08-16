---
pageClass: plain-doc
---

# 后量子密码（格密码/Kyber/Dilithium/PQC 迁移）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Bernstein, Buchmann & Dahmen (eds.), "Post-Quantum Cryptography" (2009)
- NIST, "FIPS 203/204/205 — ML-KEM/ML-DSA/SLH-DSA 标准" (2024)
- Hoffstein, Pipher & Silverman, "An Introduction to Mathematical Cryptography" (2nd ed., 2014)

## 主题规划

<ProgressGrid cat="cs/post-quantum-cryptography" />

### 第1篇

- [x] [量子威胁（Shor 算法对 RSA/ECC 的毁灭性打击、先存后解攻击）](./quantum-threats-shor-algorithm)
- [x] [格密码基础（LWE/RLWE/Module-LWE 困难问题、格基约化）](./lattice-cryptography-basics)
- [x] [密钥封装 ML-KEM（CRYSTALS-Kyber 的构造与实现）](./ml-kem-crystals-kyber)
- [x] [数字签名 ML-DSA（CRYSTALS-Dilithium 的 Fiat-Shamir 变换）](./ml-dsa-crystals-dilithium)
- [x] [哈希签名 SLH-DSA（SPHINCS+ 的无状态设计）](./slh-dsa-sphincs-plus)
- [x] [其他路线（编码密码 McEliece、多变量、同源密码 SIKE 的兴衰）](./other-pqc-approaches)
- [x] [侧信道与实现安全（NTT 实现的时序攻击、掩码防护）](./side-channel-implementation-security)
- [x] [标准化进程（NIST PQC 竞赛五轮评审、CNSA 2.0 时间表）](./nist-pqc-standardization-cnsa2)

### 第2篇

- [x] [迁移工程（混合模式、证书双签、密码敏捷性 crypto-agility）](./pqc-migration-hybrid-modes)
- [x] [性能与部署（嵌入式/TLS/固件签名中的 PQC 开销）](./pqc-performance-deployment)
- [x] [量子密码的边界（QKD 与 PQC 的互补与争论）](./qkd-vs-pqc)
- [x] [全同态加密的格基础（与《隐私计算》互链）](./fhe-lattice-foundations)
