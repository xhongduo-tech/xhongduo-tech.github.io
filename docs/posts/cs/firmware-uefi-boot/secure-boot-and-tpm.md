---
title: 安全启动（Secure Boot 信任链、TPM 度量启动）
date: 2026-08-07
---

# 安全启动（Secure Boot 信任链、TPM 度量启动）

<div class="epigraph">
<p>能被度量的东西，才可能被管理。</p>
<footer>—— 彼得 · 德鲁克（Peter Drucker）</footer>
</div>

<div class="article-byline">
<p>第三级 · 固件与启动链（BIOS/UEFI/嵌入式引导） ｜ UEFI Forum《UEFI Specification》2.10 第32章 ｜ 2026-08-07</p>
</div>

## 为什么从安全启动开始

前几节都在回答「机器怎么跑起来」，这一节回答「机器怎么能被信任地跑起来」。固件是整个平台的**第一段代码**，如果它在启动早期被篡改，之后的操作系统、杀毒软件、可信执行环境全部建立在流沙之上。于是安全启动要回答两个问题：**执行谁**（Secure Boot 管签名），**记录跑了什么**（TPM 度量启动管审计）。

两条线合起来，构成了现代平台的**信任根（Root of Trust）**。它从固件向下延伸到 TPM 芯片，向上覆盖操作系统加载器——这是《UEFI Specification》第32章与 TCG 组织 PC Client 平台固件规范共同定义的地盘。<span class="marginnote">TCG（Trusted Computing Group）是定义 TPM 的组织，它与 UEFI Forum 的边界大致是：UEFI 规范管「怎么验证签名」，TCG 规范管「怎么度量进 PCR」。两台各自分工，共同支撑启动信任。</span>

## 1 签名数据库：PK、KEK、db、dbx

Secure Boot 的信任不是「信任某个厂商」，而是「信任一组密钥」。UEFI 把决策依据存放在四个非易失变量里：

- **PK（Platform Key，平台密钥）**：整个 Secure Boot 体系的根。只有用 PK 签名的请求，才能修改 KEK。
- **KEK（Key Exchange Key，密钥交换密钥）**：受 PK 保护，用于给 db 与 dbx 签名。操作系统厂商（微软等）的证书就挂在这里。
- **db（Signature Database，签名数据库）**：受信任的签名者/哈希白名单。
- **dbx（Forbidden Signature Database）**：被撤销的签名/哈希黑名单。

**验证链条是单向的**：`PK → KEK → db/dbx → 具体镜像`。每个环节都用上一层的私钥签名，用下一层的公钥验证。这种「证书链 + 吊销列表」的结构与 HTTPS 里的 CA 体系同构。

一个重要的工程细节：**PK/KEK/db/dbx 不是存盘的普通文件，而是存进 UEFI 非易失变量（NVRAM）的「认证变量」**。它们被写进 Flash 的变量区，只能按签名规则修改（改 db 需要 KEK 签名的请求）。这也解释了为什么「Secure Boot 密钥被清空」往往意味着「固件变量区被重置」——攻击者想装自己的密钥，第一件事是寻找变量写保护的漏洞。

## 2 Secure Boot 的验证流程

当固件要加载一个 `.efi` 镜像（驱动、Shell、OS Loader）时，验证逻辑是：

1. 计算镜像哈希，先查 **dbx**：命中则直接拒绝，返回 `EFI_SECURITY_VIOLATION`；
2. 对镜像内嵌的签名（或其证书链），在 **db** 里找匹配项：找到则放行；
3. 两者都不命中，按平台的验证策略决定（PCD 可配置为「信任所有可移动介质」或「全量验证」）。

OEM 出厂时会把 **Microsoft 的 KEK 证书**预先灌进变量区，这样 Windows 的内核（`bootmgfw.efi`）天然可信；第三方系统则需要走 shim 或自签路径（见下文）。整个生命周期里「谁在什么时刻持有什么私钥」，就是**平台密钥治理**的全部内容——这也是企业采购服务器时要求「客户自管 PK」的原因。

**易错点｜辨析：** 很多人把「Secure Boot 开启」误解为「第三方系统装不了」。实际上 Secure Boot 只拦截「未签名或签名被撤销的引导代码」，开源的 shim + 自签证书完全可以加入 db。**真正决定能否安装的是 db/dbx 的内容，而不是开关本身。**<span class="marginnote">实践中最常见的踩坑是：主板默认 `Secure Boot = 开`，却用旧版 USB 启动盘引导未签名内核，结果「卡在引导不进入安装器」。对策是把启动盘的签名证书导入 db，或临时关闭 Secure Boot——这正体现了 db 可管理性。</span>

当 PK 为空时，平台处于 **Setup Mode（设置模式）**：此时任何签名都能写进 db，用户可以自行灌入自己的密钥；一旦写入 PK，平台进入 **User Mode（用户模式）**，非授权修改被拒。

Linux 世界里还有一个常被提起的组件——**shim**：微软签名的「第一级加载器」，它本身在 db 里，然后通过第二级证书 **MokList（Machine Owner Key）** 让发行版可以继续加载自己的自签内核。这套「微软签名 shim + 发行版自签内核」的桥接设计，是 Secure Boot 与开源生态共存的经典案例。

## 3 公式解析：TPM 的 PCR 扩展

Secure Boot 解决「让什么运行」，**度量启动（Measured Boot）** 解决「把运行了什么记下来」。记录的载体是 TPM 里的 **PCR（Platform Configuration Register）**，而 PCR 的写入只能通过**扩展（extend）**完成：

$$
\text{PCR}_i^{\text{new}} = H\big( \text{PCR}_i^{\text{old}} \,\|\, \text{measured\_data} \big)
$$

拆成三步理解：

- **第一步，`H` 是什么**：TPM 2.0 里通常取 SHA-256（也可用 SHA-1/SM3 等算法）。一次扩展就是把新数据哈希后，与 PCR 旧值拼在一起再哈希。
- **第二步，为什么必须哈希拼接而非覆盖**：如果直接覆盖，攻击者可以「先量度正常值、再量度恶意值」把记录洗白。PCR 扩展的**单向累积性**保证了：一旦某个早期组件被改成恶意版本，最终 PCR 值就与「干净启动」时的值对不上——不可能伪造出一条「看起来正常」的扩展序列。
- **第三步，测量与引用的分工**：固件把每一段代码的度量**扩展进 PCR**，同时把「谁在什么时刻度量了什么」写进**事件日志（Event Log）**。远程验证者拿到 PCR 值做比对，本地审计者读事件日志核对明细——一个管「结果」，一个管「过程」。

事件日志的验证有个细节：**日志本身在普通内存里，可被篡改**。所以验证者必须「重放」：把日志里的每一条事件重新 extend 一遍，看最终算出的 PCR 是否等于 TPM 硬件当前的真实 PCR 值。**日志是软的、PCR 是硬的**——只有 TPM 的硬件 PCR 才是不可伪造的证据，日志只是可读的「过程解说词」。

### 哪些 PCR 装了什么

TPM 2.0 有至少 24 个 PCR（0–23），PC Client 规范给它们分配了明确职责。读懂这张「寄存器分工表」，就懂了度量启动的整个语义：

| PCR | 度量的内容 |
| --- | --- |
| 0 | CRTM、平台固件代码（SEC 起） |
| 1 | 平台固件与配置数据 |
| 2 | 可扩展固件的选项 ROM |
| 3 | 选项 ROM 配置 |
| 4 | IPL 代码（引导加载器/OS Loader） |
| 5 | IPL 配置数据 |
| 6 | 状态转换与唤醒事件 |
| 7 | 厂商自定义的平台状态 |

这套分工让**验证者可以只认 PCR 0–7 的「干净启动基线」**：如果 PCR 0 与出厂基线一致，说明 CRTM 到 OS Loader 之间的链路没被换过。<span class="marginnote">不同 TPM 的 PCR 数量与算法不同：TPM 1.2 只有 16 个 PCR 且用 SHA-1；TPM 2.0 扩到 24 个并支持 SHA-256 等多算法。Windows 11 要求 TPM 2.0，一个重要原因就是 SHA-1 的碰撞风险与 PCR 数的不足。</span>

## 4 信任链：CRTM 到内核

完整链条从 SEC 阶段的 CRTM 开始，逐级向上：

- **CRTM（Core Root of Trust for Measurement）**：不可被度量自身的最小起点，通常烧在 Boot Firmware Volume 里；
- **SEC → PEI → DXE → BDS**：每一阶段在把控制权交给下一阶段前，先度量下一阶段代码再扩展进 PCR；
- **OS Loader**：度量后再执行，之后把事件日志与 PCR 值交给内核；
- **内核与安全启动协同**：内核可检查事件日志是否与自身期望一致（IOMMU 启动、内存完整性等）。

度量结果的最终消费场景是**远程证明（Remote Attestation）**：机器把 PCR 值、事件日志与一段「证明」发到验证服务器，服务器对照「可信基线库」判定这台机器是否处于预期状态。这是企业合规、机密计算（如 Intel TDX 的信誉服务）与「零信任」网络的底层支点——**TPM 不只是给本地看的，更是给远端看的**。

一个具体到日常的例子是 **Windows 的 BitLocker 与 VBS**：BitLocker 把磁盘加密密钥「钉」在 TPM 的 PCR 上（通常 PCR 0/2/4/7/11），只有启动链度量值「干净」时 TPM 才释放密钥；VBS（基于虚拟化的安全）则要求 Secure Boot 与 TPM 2.0 同时在线，用 Hyper-V 隔离内核凭证。**对普通用户而言，「固件安全」最直接的体验就是开机密码与磁盘加密，而它们都压在这条信任链上。**

**核心对比表**（纯概念主题，以表代替公式）：Secure Boot 与 Measured Boot 常被混为一谈，区别很重要：

| 维度 | Secure Boot | Measured Boot |
| --- | --- | --- |
| 问题 | 能不能执行？ | 执行了什么？ |
| 机制 | 签名 + db/dbx 验证 | 哈希扩展 + 事件日志 |
| 载体 | UEFI 变量 | TPM PCR |
| 行为 | 拒绝恶意镜像 | 记录但不拒绝 |
| 依赖 | 密钥管理 | 信任锚 CRTM |
| 典型用途 | 防引导前 rootkit | 远程证明、合规审计 |

<span class="marginnote">两者互补：Secure Boot 负责「拦截」，Measured Boot 负责「留痕」。只有拦截没有留痕，攻破后无从追责；只有留痕没有拦截，恶意代码照样运行。现代平台把两个都打开——Windows 的 VBS（基于虚拟化的安全）就同时依赖 UEFI Secure Boot 与 TPM 2.0。</span>

## 5 小结

- Secure Boot 的核心是四个变量：**PK → KEK → db/dbx**，证书链单向验证，dbx 提供撤销能力。
- 验证流程先查黑名单再查白名单，策略可由平台 PCD 微调；**Setup Mode 与 User Mode** 决定密钥能否被改。
- **TPM PCR 扩展** `PCR' = H(PCR ‖ data)` 是度量启动的地基：单向累积，不可洗白。
- **CRTM → 各阶段 → OS Loader → 内核** 形成完整信任链，度量结果通过事件日志交给上层。
- Secure Boot 管「拦截」，Measured Boot 管「留痕」，两者共同构成平台信任根。
- PCR 分工（0–7）把「固件、Option ROM、OS Loader、配置」分开度量，验证者可据此做「干净基线」比对。
- 事件日志可被篡改，**验证必须重放日志与硬件 PCR 比对**——「日志是软的、PCR 是硬的」。
- **PK/KEK/db/dbx 存于 NVRAM 认证变量**，修改须签名，Setup Mode 与 User Mode 是密钥治理的两个阶段。

- 工程落地：**shim + MokList** 让开源系统在 Secure Boot 下存活；**BitLocker/VBS** 让安全启动惠及普通用户。

在下一节，我们把信任链暂时放回物理世界，看内存是怎么被点亮与训练的：MRC、SPD 读取与 DDR 训练——这是整个启动链里最像「硬件魔术」的一步。
