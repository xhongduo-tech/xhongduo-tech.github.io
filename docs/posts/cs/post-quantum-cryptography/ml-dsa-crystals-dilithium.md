---
title: 数字签名 ML-DSA（CRYSTALS-Dilithium 的 Fiat-Shamir 变换）
date: 2026-08-07
---

# 数字签名 ML-DSA（CRYSTALS-Dilithium 的 Fiat–Shamir 变换）

<div class="epigraph">
<p>信任，但务必核实。</p>
<footer>—— 罗纳德 · 里根（Ronald Reagan，借自俄谚）</footer>
</div>

<div class="article-byline">
<p>第三级 · 后量子密码 ｜ NIST FIPS 204《ML-DSA》标准 §5–§7 ｜ 2026-08-07</p>
</div>

## 为什么从签名开始

ML-KEM 解决了「共享密钥」，但互联网的信任骨架是另一件东西：**数字签名**。你浏览的每一个 HTTPS 网站，证书链的每一环都是签名；你下载的每一个软件、固件、更新包，都要靠签名验证真伪；代码签名、邮件签名、车联网 V2X 消息、区块链交易——签名无处不在。<span class="marginnote">签名与加密的根本差异：加密是「公开加密、私钥解密」，签名是「私钥签名、公钥验签」。前者保证机密性，后者保证<b>真实性、完整性与不可抵赖</b>。一旦量子计算机出现，RSA/ECDSA 签名全部可被伪造，整个信任骨架随之崩塌。</span>

ML-KEM 与 ML-DSA 是一对姊妹：都出自 CRYSTALS 项目（Kyber + Dilithium），都基于 Module-LWE，都走 NTT 实现。但它们在数学上有一个关键分野：**加密靠「隐藏」短向量，签名靠「证明」短向量**。签名需要一个「零知识」过程：证明自己知道私钥（那个短向量）而不泄露它。Dilithium 用 **Fiat–Shamir 变换** 把交互式证明变成一次性签名——这是这一篇的核心戏法。

在讲戏法之前，先明确「签名」与《密码学与信息安全》里见过的 MAC 的区别：MAC 用对称密钥、双方共享同一把钥匙，能验签的人也能伪造；签名用公钥密码，**验签人只有公钥、无法伪造**。后量子迁移因此必须同时换掉「密钥交换」与「签名」两套原语——ML-DSA 就是后者在 FIPS 204 里的正式答案，2024 年 8 月与 ML-KEM 同日定稿。

## 1 Fiat–Shamir 变换：把「对话」变成「签名」

要理解 Dilithium，先理解一个更古老的结构。1986 年，阿莫斯 · 菲亚特（Amos Fiat）与阿迪 · 沙米尔（Adi Shamir）提出一个著名技巧：**任何「公开随机挑战、承诺—应答」的交互式零知识证明，都可以把验证者换成哈希函数，变成非交互的一次性签名**。

一个典型的交互式证明（如 Schnorr 签名）有三步：

1. **承诺（Commit）**：签名者先随机取 $y$，公开承诺 $w$。
2. **挑战（Challenge）**：验证者随机抛硬币 $c$。
3. **应答（Response）**：签名者计算 $z = f(y, c, \text{私钥})$ 并发送。

**Fiat–Shamir 的魔法**：把第二步的「验证者随机挑战」替换成「$c = H(\text{消息} \| w)$」——签名者自己算出挑战。因为哈希是公开函数、无法预知，签名者必须在知道 $c$ 之前就锁定 $w$，于是「伪造者」无法像真签名者那样事后调整应答。<span class="marginnote">安全性直觉：交互式证明里验证者的随机性保证「承诺在先、挑战在后」，防伪造；Fiat–Shamir 用哈希把这一约束<b>固化进签名本身</b>。这是 1986 年以来几乎所有格基签名（含 Dilithium、Falcon）的共同骨架。在 ROM（随机预言机模型）里，Fiat–Shamir 变换保安全性。</span>

## 2 Dilithium 的代数：还是那套 Module-LWE

Dilithium 与 Kyber 共用同一片代数土壤：环 $R_q = \mathbb{Z}_{3329}[x]/(x^{256}+1)$，模块维数 $k, \ell$。密钥生成：

$$
\mathbf{A} \in R_q^{k \times \ell}, \qquad
\mathbf{t} = \mathbf{A}\mathbf{s}_1 + \mathbf{s}_2
$$

其中 $\mathbf{s}_1, \mathbf{s}_2$ 是小系数秘密（取自窄分布），$\mathbf{A}$ 由种子经 SHAKE128 展开。<span class="marginnote">公钥是 $(\mathbf{A}, \mathbf{t})$，私钥是 $\mathbf{s}_1, \mathbf{s}_2$。注意这里与 Kyber 几乎同构——但用途相反：Kyber 用 $\mathbf{t} = \mathbf{A}\mathbf{s}+\mathbf{e}$ 藏信息，Dilithium 用同样的等式做「知识证明」的证据。</span>给定公钥求私钥就是 Module-LWE 难题——这是签名不可伪造性的地基。

关键设计是**把公钥中的误差项 $\mathbf{s}_2$ 显式保存为私钥的一部分**。这让验签方程能「精确」成立：签名者用 $\mathbf{s}_2$ 微调应答，使验签方得到一个**接近但不完全等于**承诺的小量——正是这份「有误差的成立」，让攻击者学不走签名过程。

## 3 签名与验签：三条核心等式

### 公式解析：验签方程为什么能「约等于」

**签名（Sign）**，输入消息 $m$ 与私钥：

$$
\mathbf{w} = \mathbf{A}\mathbf{y}, \qquad
c = H(\mu \| \mathbf{w}_{\text{高}}), \qquad
\mathbf{z} = \mathbf{y} + c\,\mathbf{s}_1
$$

其中 $\mathbf{y}$ 是随机采样的「遮蔽」，$c$ 是由哈希导出的挑战，$\mathbf{z}$ 是应答。若 $\mathbf{z}$ 太大（超出边界）就**拒绝重来**（见下节）。

**验签（Verify）**，输入 $m$、公钥 $(\mathbf{A}, \mathbf{t})$、签名 $(\mathbf{z}, c)$：

$$
\mathbf{w}' = \mathbf{A}\mathbf{z} - c\,\mathbf{t}
= \mathbf{A}(\mathbf{y} + c\mathbf{s}_1) - c(\mathbf{A}\mathbf{s}_1 + \mathbf{s}_2)
= \mathbf{A}\mathbf{y} - c\,\mathbf{s}_2
\approx \mathbf{A}\mathbf{y} = \mathbf{w}
$$

逐项拆解：

- **第一步，代入并展开**：把 $\mathbf{z} = \mathbf{y} + c\mathbf{s}_1$ 与 $\mathbf{t} = \mathbf{A}\mathbf{s}_1 + \mathbf{s}_2$ 代入，$\mathbf{A}\mathbf{s}_1$ 两项恰好相消。
- **第二步，留下误差**：残差是 $-c\,\mathbf{s}_2$——因为 $\mathbf{s}_2$ 系数小，$c$ 是短挑战多项式，所以 $c\,\mathbf{s}_2$ 是小量。
- **第三步，比较高位**：验签方只比较 $\mathbf{w}'$ 与 $\mathbf{w}$ 的**高位（丢弃小误差）**，若一致则通过。若攻击者没有私钥 $\mathbf{s}_1$，就无法构造让 $\mathbf{A}\mathbf{z} - c\mathbf{t}$ 与承诺高位匹配的 $\mathbf{z}$——这就是「不可伪造」的数学来源。

### 为什么拒绝采样必不可少

如果直接发 $\mathbf{z} = \mathbf{y} + c\mathbf{s}_1$，验签方能从多个签名里减去 $\mathbf{z}$ 反推出 $c\mathbf{s}_1$，进而恢复 $\mathbf{s}_1$——**私钥泄露**。Dilithium 的防护是**拒绝采样（rejection sampling）**：只当 $\mathbf{z}$ 落在某个「安全区间」内才输出，否则丢弃重采。<span class="marginnote">拒绝采样让 $\mathbf{z}$ 的分布与私钥 $\mathbf{s}_1$ 无关（分布均匀、条件独立），于是签名不泄露私钥信息。代价是约 10–20% 的签名尝试被拒绝——参考实现用「确定性种子重试」控制时间。这正是 Kyber 里没有、Dilithium 独有的机制。</span>

## 4 参数与尺寸：签名比密钥大

| 参数集 | 安全级别 | 公钥大小 | 签名大小 | 哈希种子 |
| --- | --- | --- | --- | --- |
| ML-DSA-44 | 2 | 1312 字节 | 2420 字节 | SHAKE256 |
| ML-DSA-65 | 3 | 1952 字节 | 3309 字节 | SHAKE256 |
| ML-DSA-87 | 5 | 2592 字节 | 4627 字节 | SHAKE256 |

对比 ECDSA（P-256）签名 64 字节、RSA-2048 签名 256 字节，**ML-DSA 签名大了 10–70 倍**。这是格签名最显著的工程代价：对证书链、代码签名这类「签名次数少、验签次数多」的场景，签名尺寸直接进带宽预算；对链上交易（区块链）、汽车 V2X 这种「每条消息都要签名」的场景，签名大小几乎决定吞吐量。

还有一个常被忽略的点：**验签速度**。Dilithium 的验签主要是矩阵乘法（NTT 加速后约几十微秒），远快于 RSA 的模幂运算——这在「一次签名、万次验签」的证书链场景里是净收益。选型时要同时看尺寸、签名速度、验签速度三个维度，不能只看字节数就下结论。<span class="marginnote">NIST 对 ML-DSA 的定位是「默认签名」：性能均衡、实现成熟、基于与 ML-KEM 相同的信任假设。CNSA 2.0 要求国家保密系统用 ML-DSA-87；而 SLH-DSA（哈希签名）作为「保守备份」备用，防止格假设崩盘。</span>

## 5 对比 Kyber：同源不同路

把姊妹俩摆在一起，格密码的「两面」就清晰了：

| 维度 | ML-KEM（Kyber） | ML-DSA（Dilithium） |
| --- | --- | --- |
| 用途 | 密钥封装 | 数字签名 |
| 私钥形态 | 秘密 $\mathbf{s}$ | 秘密 $\mathbf{s}_1, \mathbf{s}_2$ |
| 输出 | 密文 + 会话密钥 | 签名 $(\mathbf{z}, c)$ |
| 关键机制 | FO 变换（CCA） | 拒绝采样（防泄露） |
| 挑战来源 | 无挑战 | $c = H(\mu \| \mathbf{w})$ |

**辨析｜易错点：** 两者都叫「CRYSTALS」、都基于 Module-LWE、都用 NTT，但一个核心差异决定了它们的安全证明结构：**加密不需要公开「零知识」**，而签名必须公开一份「不泄露秘密的证明」。所以 Dilithium 额外背上了拒绝采样与窄分布，签名尺寸也远大于密文。把「Kyber 是加密版 Dilithium」记成等式会踩坑——它们的数学同源，安全目标与机制完全不同。

**补充一句历史**：Dilithium 在 NIST 竞赛中胜出的关键之一，是它的参数在「小尺寸、快验签、简单实现」三者之间取得了最佳平衡——决赛对手 Falcon 签名更小（基于 NTRU + 高斯采样，约 666–1949 字节）但实现复杂、专利与侧信道风险更高；Dilithium 则凭「简单、可测、难出错」赢得 FIPS 204 正选，Falcon 则进入后续的 FIPS 206。这一取舍值得记住：**标准化比赛里，「实现上的稳健」与「理论上的优雅」同样重要**。

## 6 小结

- **签名 vs 加密**：签名保真实性、完整性、不可抵赖，私钥签名、公钥验签；量子威胁下 RSA/ECDSA 签名全部可伪造。
- **Fiat–Shamir 变换（1986）**：用哈希把「承诺—挑战—应答」的交互式证明变成非交互签名；Dilithium 继承这一骨架。
- **代数同源**：Dilithium 与 Kyber 共用环 $\mathbb{Z}_{3329}[x]/(x^{256}+1)$ 与 Module-LWE，私钥是 $\mathbf{s}_1, \mathbf{s}_2$，公钥 $\mathbf{t} = \mathbf{A}\mathbf{s}_1 + \mathbf{s}_2$。
- **验签方程**：$\mathbf{A}\mathbf{z} - c\mathbf{t} = \mathbf{A}\mathbf{y} - c\mathbf{s}_2 \approx \mathbf{w}$——相消 + 小误差 + 比较高位的三连击。
- **拒绝采样**：让 $\mathbf{z}$ 分布与私钥无关，防签名泄露私钥；这是 Dilithium 独有的机制。
- **尺寸税更大**：ML-DSA-87 签名 4627 字节，是 ECDSA 的 70 倍；NIST 定位 ML-DSA 为默认签名、SLH-DSA 为保守备份。
- **三因子选型**：尺寸、签名速度、验签速度要一起看；Dilithium 验签快，证书链场景受益。

在下一节，我们将介绍那份「保守备份」——**SLH-DSA（SPHINCS+）**。它完全不依赖格假设，只靠哈希函数的性质，用 Merkle 树把一次性签名编织成无状态签名，是格密码崩盘时的「最后的防线」。你会看到哈希签名如何用「空间换时间、树状认证」的朴素思想，把签名安全建立在 SHA 系列之上。
