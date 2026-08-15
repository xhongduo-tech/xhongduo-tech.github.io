---
title: Needham-Schroeder 协议
date: 2026-08-07
---

# Needham-Schroeder 协议

<div class="epigraph">
<p>认证协议的全部艺术，在于让双方在交换密钥的同时确信对方真的在场——而每一次「确信」都可能被重放打破。</p>
<footer>—— 罗杰 · 尼德汉姆与迈克尔 · 施罗德（R. Needham, M. Schroeder）</footer>
</div>

<div class="article-byline">
<p>第三级 · 密码学与信息安全 ｜ Stallings《密码编码学与网络安全》第十四章 ｜ 2026-08-07</p>
</div>

## 为什么从 Needham-Schroeder 开始

1978 年，Roger Needham 与 Michael Schroeder 发表了第一个完整的对称密钥分配协议——**Needham-Schroeder 协议**。它是 KDC 思想的严格化：用 **nonce（一次性随机数）** 来防止重放攻击，用**票据**让会话密钥安全抵达双方。它是 Kerberos 的直接前身，也是「协议分析」这门学问的第一个研究对象——它后来被发现存在一个著名漏洞（Denning-Sacco 攻击），这个漏洞直接催生了 Kerberos 的时间戳设计。<span class="marginnote">Needham-Schroeder 的贡献不只是协议本身，而是它开创了<strong>「用 nonce 证明新鲜性」</strong>的方法：一条消息只有在「刚刚生成」时才有意义，重放它没用。这个思想渗透进一切现代认证协议。</span>

## 1 协议流程：六条消息

Needham-Schroeder 在 Alice（想通信）、Bob（目标）、KDC（可信第三方）之间交换 6 条消息。Alice 持有与 KDC 的长期密钥 $K_A$，Bob 持有 $K_B$：

1. **A → KDC**：$\text{ID}_A \| \text{ID}_B \| N_A$——Alice 请求「与 Bob 通信的密钥」，附上她的 nonce $N_A$。
2. **KDC → A**：$E_{K_A}(K_{AB} \| \text{ID}_B \| N_A \| \text{Ticket}_B)$——KDC 生成会话密钥 $K_{AB}$，用 $K_A$ 加密发给 Alice，并附上给 Bob 的票据。
3. **A → B**：$\text{Ticket}_B$——Alice 转发票据给 Bob。
4. **B → A**：$E_{K_{AB}}(N_B)$——Bob 用 $K_{AB}$ 回一个自己的 nonce，证明「我确实拿到了票据且在场」。
5. **A → B**：$E_{K_{AB}}(N_B - 1)$——Alice 证明「我也持有 $K_{AB}$」。
6. 双方用 $K_{AB}$ 开始通信。

其中**票据**是 $\text{Ticket}_B = E_{K_B}(K_{AB} \| \text{ID}_A)$——用 Bob 的长期密钥加密，Alice 解不开、也改不了，只能原样转发。<span class="marginnote">票据的设计巧思：<strong>KDC 只需给 Alice 发一份密文，Alice 把「属于 Bob 的那半」转交，Bob 用自己的密钥解开</strong>。票据是「KDC 写给 Bob 的介绍信」，Alice 只是信使——她碰不到信的内容。</span>

用三步推演理解「nonce 如何让 Alice 确信响应是新鲜的」：设 Alice 之前发给 KDC 的请求里 nonce $N_A = 314159$。

1. KDC 的响应（消息 2）里**原样带回 $N_A$**，并用 $K_A$ 加密。
2. Alice 解开后，看到 $N_A = 314159$——**确认这条响应是对「她刚才那次请求」的回应**，不是攻击者重放的旧响应（旧响应里是别的 nonce）。
3. 若攻击者重放 KDC 的旧响应，Alice 解出的是旧 nonce——立即识破，丢弃。

**nonce 的作用域**：它证明了「KDC 的响应新鲜」（消息 2），却没证明「票据新鲜」（旧票据仍可重放，见第 3 节的 Denning-Sacco）。这正是「nonce 派」协议的共同盲区。

## 2 公式解析：消息 2 里的双重加密

$$
\text{KDC} \to A: \quad E_{K_A}\big(K_{AB} \| \text{ID}_B \| N_A \| E_{K_B}(K_{AB} \| \text{ID}_A)\big)
$$

三步拆解这条「消息 2」：

- **第一步，外层用 $K_A$**：整个响应用 Alice 的长期密钥加密——只有 Alice 能解开，且 $N_A$ 让她确认「这是对我这次请求的回应」（防重放）。
- **第二步，内层票据用 $K_B$**：$E_{K_B}(K_{AB}\|\text{ID}_A)$ 只有 Bob 能解——Alice 拿到的是「打不开的包裹」，转交即可。
- **第三步，安全属性**：Alice 知道 $K_{AB}$（从外层解出）、Bob 能从票据解出 $K_{AB}$——双方有了共享密钥，且都通过 KDC 的信任背书。
- **辨析｜易错点：** 消息 2 里的「$N_A$ 回显」是防重放的关键——若 KDC 响应里不带 $N_A$，攻击者可以重放旧的 KDC 响应，Alice 无法区分「新响应」与「旧响应的重放」。nonce 的「回显验证」是协议新鲜性的第一道闸门。

## 3 著名漏洞：Denning-Sacco 攻击（重放的旧票据）

1981 年，Denning 与 Sacco 发现了 Needham-Schroeder 的一个缺陷：**票据没有过期时间**。

**攻击场景**：

1. 攻击者（曾经合法地）截获了一份旧票据 $\text{Ticket}_B$ 与对应的旧会话密钥 $K_{AB}$（比如通过渗透或暴力破解 $K_{AB}$）。
2. 攻击者**冒充 Alice**，把这份旧票据发给 Bob。
3. Bob 解开票据，得到 $K_{AB}$ 与「Alice」——他**相信**这就是 Alice 在发起新会话。
4. 攻击者用旧 $K_{AB}$ 与 Bob 通信——**成功冒充 Alice**。

**根因**：协议用 nonce 保证了「消息 2 是新生成的」，但**票据本身没有绑定时间**——旧票据可以无限重放。Bob 无法区分「新会话」与「重放的旧会话」。

**修复方向**：给票据加**时间戳**，让 Bob 拒绝过期票据——这正是 Kerberos 的核心改进。<span class="marginnote">Denning-Sacco 攻击的教训：<strong>认证协议要防的不只是「伪造」，还有「重放」</strong>。nonce 能证明「消息新鲜」，却证明不了「会话新鲜」——除非把时间信息显式放进票据。Kerberos 的时间戳，就是对这个漏洞的教科书式修复。</span>

用一张时间线把 Denning-Sacco 的「旧票据重放」具体化（票据有效期假设为永久）：

| 时间 | 事件 | 攻击者的动作 |
| --- | --- | --- |
| 周一 | Alice 与 Bob 正常会话，票据含 $K_{AB}$ | 攻击者截获票据与 $K_{AB}$ 副本 |
| 周一晚 | 会话结束，$K_{AB}$ 本应作废 | 攻击者保留副本，等时机 |
| 三个月后 | 攻击者冒充 Alice 向 Bob 发旧票据 | Bob 解开票据，以为 Alice 发起新会话 |
| 三个月后 | Bob 与「Alice」用旧 $K_{AB}$ 通信 | 攻击者完成冒充 |

**关键差异**：nonce 只证明了「消息 2（KDC 的响应）是新生成的」——但这条消息**没有把「这个会话是什么时候的」写进票据**。一旦会话结束、$K_{AB}$ 应作废，攻击者仍能靠旧票据让 Bob「重启」信任。Kerberos 给票据加时间戳 + 生命周期，正是为了把「旧票据」变成「过期票据」。

## 4 协议分析的诞生

Needham-Schroeder 的价值还在方法论层面：它是第一个被**系统性分析**的认证协议。此后出现了专门的协议分析工具与逻辑：

**BAN 逻辑**（Burrows-Abadi-Needham，1989）：用形式逻辑证明协议「相信」什么——哪些主体相信哪些密钥是新鲜的、可信的。
**形式化验证**：用工具（ProVerif、Tamarin）自动搜索协议的攻击。
**Dolev-Yao 模型**：抽象攻击者「能读能改能删能重放一切消息」。

这些工具从 Needham-Schroeder 时代发展至今，是现代安全协议设计的标配。<span class="marginnote">一个观念转变：<strong>「协议看起来对」≠「协议安全」</strong>。Needham-Schroeder 的漏洞（以及其他协议的分析）让人们明白：认证协议必须用形式化工具证明，而不是靠「聪明设计 + 肉眼审查」。这个教训直接塑造了今天的密码学协议工程。</span>

**辨析｜易错点：** 「分析出漏洞」不等于「协议不能修」——Needham-Schroeder 在加时间戳后变得安全，其后的许多变体（Kerberos v4/v5、Otway-Rees、Woo-Lam）都是「原协议 + 针对已知攻击的修补」。**协议安全是「设计 + 分析 + 修补」的循环**，不是一次成型。这跟「算法安全靠公开分析迭代」是同一个方法论。

## 5 从 Needham-Schroeder 到 Kerberos

Needham-Schroeder 的改进方向在 1980 年代逐步清晰，最终汇聚成 Kerberos：

| 特性 | Needham-Schroeder | Kerberos |
| --- | --- | --- |
| 防重放 | nonce | nonce + **时间戳** |
| 票据过期 | 无 | **有**（票据生命周期） |
| KDC 结构 | 单一 | AS + TGS 分工 |
| 目标用途 | 理论协议 | 企业域认证（MIT） |

Kerberos 保留「KDC + 票据 + nonce」的骨架，加上时间戳与生命周期，从根上修复 Denning-Sacco。<span class="marginnote">一个贯穿性的记忆：<strong>Needham-Schroeder 是「nonce 派」的巅峰，Kerberos 是「时间戳派」的胜利</strong>。现代协议（TLS、Kerberos、OAuth）通常两者并用：nonce 防即时重放、时间戳/序号防跨会话重放。</span>

**选型判断**：nonce 与时间戳不是二选一，而是互补——nonce 需要交互往返（先发挑战再回应），时间戳需要时钟同步（依赖 NTP）。**需要离线验证（无交互）选时间戳，需要精确防重放（不信任时钟）选 nonce**，两者能叠加就叠加。现代协议几乎都是「nonce 做握手 + 时间戳/序号做凭证」的组合。

## 6 小结

- **Needham-Schroeder**：6 条消息，KDC 颁发会话密钥，nonce 防重放，票据传递密钥。
- **票据**：$E_{K_B}(K_{AB}\|\text{ID}_A)$——Alice 只转交、不碰内容。
- **nonce 作用域**：回显验证证明「响应新鲜」，但证明不了「票据新鲜」——这正是下一行漏洞的伏笔。
- **Denning-Sacco 攻击**：旧票据无限重放，冒充合法用户——因为**票据没有时间戳**。
- **修复的通用形态**：把「新鲜性」从消息层（nonce）下沉到凭证层（时间戳/生命周期）——现代协议两者都用。
- **方法论遗产**：BAN 逻辑、Dolev-Yao 模型、形式化验证——协议分析成为独立学科。
- 演进：加时间戳与生命周期 → **Kerberos**。
- 一句话总结：**Needham-Schroeder 用 nonce 证明了「响应新鲜」，却忘了证明「票据新鲜」**——这个「漏一层」的教训，正是「认证协议要逐层追问新鲜性」的起点。
- 记忆锚点：**nonce 防「消息重放」，时间戳防「会话重放」**——Kerberos 的改进本质上就是把第二层补上。

在下一节，我们看企业认证的事实标准——**Kerberos 认证体系**。
